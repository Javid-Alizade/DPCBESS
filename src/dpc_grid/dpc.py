"""Milestone M3: the closed loop, the training loss and the two reference controllers.

One step of the loop:  measurements -> controller -> (projection) -> network and battery -> next measurements.
The controller sees only the present and the past. The future load and solar are used only by the simulated network.

Sign convention from docs/decisions.md: P_bess > 0 is discharge, Q_bess > 0 injects reactive power, and both raise the voltage.
Powers in the solver are per unit, while the battery equation uses MW and MWh.
"""
import numpy as np
import pandas as pd
import torch

from .data import SPLITS, window_starts
from .metrics import summarize_rq2
from .physics import bess_soc_step
from .policy import DPCPolicy
from .projection import project_to_feasible_set

N_STATE = 8        # SoC (2), V at battery buses (2), previous action (4)
N_DISTURBANCE = 8  # total load, PV at 4 buses, latest r_t, time of day (2)


class ClosedTestPartError(RuntimeError):
    """Raised when code asks for the test part, which is sealed."""


class M3Data:
    """Training and validation steps of the M2 dataset. Steps of the test part are dropped when the file is read."""

    def __init__(self, dataset_path, s_base_mva, dt_min, splits=("train", "val")):
        if "test" in splits:
            raise ClosedTestPartError("The test part is sealed and is not loaded (docs/decisions.md).")
        raw = np.load(dataset_path)
        keep = np.isin(raw["labels"], [SPLITS.index(s) for s in splits])
        self.s_base, self.dt_min = s_base_mva, dt_min
        self.dt_h = dt_min / 60
        self.labels = raw["labels"][keep]
        self.stamps = pd.to_datetime(raw["stamps"][keep], unit="s")
        self.valid = raw["valid"][keep]
        self.extreme = raw["extreme"][keep]
        self.r = raw["r_mw_per_min"][keep].astype(float)
        self.p_load = torch.as_tensor(raw["p_load_x1"][keep], dtype=torch.float32)
        self.q_load = torch.as_tensor(raw["q_load_x1"][keep], dtype=torch.float32)
        self.p_solar = torch.as_tensor(np.nan_to_num(raw["p_solar"][keep]), dtype=torch.float32)  # NaN marks non-PV buses too

    def prepare(self, pv_buses, bess_buses):
        """Normalise the measured inputs with statistics of the TRAINING part only."""
        self.pv_buses, self.bess_buses = list(pv_buses), list(bess_buses)
        train = torch.as_tensor(self.labels == 0) & torch.as_tensor(self.valid)
        total = self.p_load.sum(dim=1)
        self.total_load = total
        self.load_mean, self.load_std = total[train].mean(), total[train].std()
        columns = [self.p_solar[:, b] for b in self.pv_buses]
        columns.append(torch.as_tensor(np.nan_to_num(self.r), dtype=torch.float32))
        exogenous = torch.stack(columns, dim=1)
        self.exogenous_mean, self.exogenous_std = exogenous[train].mean(dim=0), exogenous[train].std(dim=0)
        if (self.exogenous_std == 0).any() or self.load_std == 0:
            raise ValueError("an input is constant on the training part, and dividing by its spread would give NaN")
        self.exogenous =(exogenous - self.exogenous_mean) / self.exogenous_std
        minutes = torch.as_tensor(np.asarray(self.stamps.hour * 60 + self.stamps.minute), dtype=torch.float32)
        angle = 2 * torch.pi * minutes / (24 * 60)
        self.time_of_day = torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)
        return self

    def windows(self, split, horizon):
        """Starts of all fully valid, consecutive windows inside one split."""
        return window_starts(self.valid, self.labels, SPLITS.index(split), horizon, self.stamps, self.dt_min)


def disturbance_features(data, t, scale):
    """Measured inputs at the steps t, as (B, 8). `scale` is the load scale of each window, which the load input reflects."""
    load = (scale * data.total_load[t] - data.load_mean) / data.load_std
    return torch.cat([load.unsqueeze(1), data.exogenous[t], data.time_of_day[t]], dim=1)


def state_features(soc, v_bess, previous_action, battery):
    """SoC, V and previous action on fixed scales, as (B, 8)."""
    return torch.cat([(soc - 0.5) / 0.25, (v_bess - 1.0) / 0.05, previous_action.flatten(1) / battery["p_max_pu"]], dim=1)


class PolicyController:
    """The DPC policy. Its raw outputs are scaled so that 1.0 means the 0.5 MW limit."""

    def __init__(self, policy, battery):
        self.policy, self.battery = policy, battery

    def __call__(self, state, disturbance, v_bess):
        return self.policy(state, disturbance).view(-1, v_bess.shape[1], 2) * self.battery["p_max_pu"]


class IdleController:
    """Reference: the batteries do nothing."""

    def __call__(self, state, disturbance, v_bess):
        return torch.zeros(v_bess.shape[0], v_bess.shape[1], 2, dtype=v_bess.dtype)


class DroopController:
    """Reference: Volt-VAR droop on each inverter, from V at its own bus on the previous step. P stays 0, so SoC is unused.

    Q is 0 inside the dead band and grows linearly to the full limit at `full_at` from 1.0 pu. Q > 0 raises V, so low V injects.
    """

    def __init__(self, battery, dead_band_pu, full_at_pu):
        self.q_max, self.dead_band, self.full_at = battery["q_max_pu"], dead_band_pu, full_at_pu

    def __call__(self, state, disturbance, v_bess):
        low_by = 1.0 - v_bess  # positive when the voltage is low
        size = ((low_by.abs() - self.dead_band) / (self.full_at - self.dead_band)).clamp(0.0, 1.0)
        return torch.stack([torch.zeros_like(v_bess), torch.sign(low_by) * size * self.q_max], dim=2)


def per_unit_battery(battery, s_base):
    """Add the inverter limits in per unit to the battery settings."""
    return dict(battery, p_max_pu=battery["p_max_mw"] / s_base, q_max_pu=battery["q_max_mvar"] / s_base,
                s_max_pu=battery["s_max_mva"] / s_base)


def energy_power_limits(soc, battery, dt_h, s_base):
    """Largest discharge and charge power (per unit, both >= 0) that keep SoC inside its limits after one step.

    From the SoC equation:  SoC_next = SoC - P_dis dt / (eta_dis E)  and  SoC_next = SoC + eta_ch P_ch dt / E.
    A SoC already outside its limits gives 0 in that direction, so it can only move back.
    """
    energy_pu = battery["energy_mwh"] / s_base
    discharge = torch.relu(soc - battery["soc_min"]) * energy_pu * battery["eta_dis"] / dt_h
    charge = torch.relu(battery["soc_max"] - soc) * energy_pu / (battery["eta_ch"] * dt_h)
    return discharge, charge


def safety_layer(action, soc, battery, dt_h, s_base):
    """The projection with the energy limit: P is clipped to what SoC allows, then to P, Q and S of the inverter.

    Radial scaling for S can only shrink P, so it cannot break the energy limit. Returns the safe action and
    a mask of the battery steps where the energy limit, and not the inverter, cut P.
    """
    discharge, charge = energy_power_limits(soc, battery, dt_h, s_base)
    p_upper = torch.minimum(discharge, torch.full_like(discharge, battery["p_max_pu"]))
    p_lower = -torch.minimum(charge, torch.full_like(charge, battery["p_max_pu"]))
    p = action[..., 0]
    clipped = ((p > p_upper) & (discharge < battery["p_max_pu"])) | ((p < p_lower) & (charge < battery["p_max_pu"]))
    safe = project_to_feasible_set(action, p_lower, p_upper, -battery["q_max_pu"], battery["q_max_pu"], battery["s_max_pu"])
    return safe, clipped


def control_step(controller, data, t, scale, soc, v_bess, previous_action, battery, solver, use_projection, diagnostics=None):
    """One step of the closed loop at the data steps t. Returns network output, action (B, n_bess, 2) and the new SoC.

    If a `diagnostics` dict is given, it receives "soc_clip", the mask of battery steps where the energy limit cut P.
    """
    state = state_features(soc, v_bess, previous_action, battery)
    if getattr(controller, "needs_context", False):  # a model-based controller (the MPC) also needs the step, scale and SoC
        action = controller(state, disturbance_features(data, t, scale), v_bess, t=t, scale=scale, soc=soc)
    else:
        action = controller(state, disturbance_features(data, t, scale), v_bess)
    clipped = torch.zeros_like(soc, dtype=torch.bool)
    if use_projection:
        action, clipped = safety_layer(action, soc, battery, data.dt_h, data.s_base)
    if diagnostics is not None:
        diagnostics["soc_clip"] = clipped
    index = torch.as_tensor(data.bess_buses)
    zeros = torch.zeros_like(data.p_load[t])
    p_bess = zeros.index_copy(1, index, action[..., 0])  # index_copy returns a new tensor, so gradients keep flowing
    q_bess = zeros.index_copy(1, index, action[..., 1])
    out = solver(scale.unsqueeze(1) * data.p_load[t], scale.unsqueeze(1) * data.q_load[t], data.p_solar[t],
                 p_bess=p_bess, q_bess=q_bess)
    soc_next = bess_soc_step(soc, action[..., 0] * data.s_base, battery["energy_mwh"], data.dt_h,
                             battery["eta_ch"], battery["eta_dis"])
    return out, action, soc_next


def idle_voltage(data, t, scale, solver):
    """V with idle batteries at the steps t, used as the "previous V" at the start of a window."""
    with torch.no_grad():
        return solver(scale.unsqueeze(1) * data.p_load[t], scale.unsqueeze(1) * data.q_load[t], data.p_solar[t])["v"]


def step_terms(out, action, soc, battery):
    """All terms of proposal eq. (16) at one step, per window, not yet weighted. Degradation is off in this milestone."""
    v = out["v"]
    limits_low = torch.relu(0.95 - v) ** 2
    limits_high = torch.relu(v - 1.05) ** 2
    p, q = action[..., 0], action[..., 1]
    s = torch.sqrt(p ** 2 + q ** 2 + 1e-12)  # the small constant keeps the gradient finite at P = Q = 0
    inverter = (torch.relu(p.abs() - battery["p_max_pu"]) ** 2 + torch.relu(q.abs() - battery["q_max_pu"]) ** 2
                + torch.relu(s - battery["s_max_pu"]) ** 2)
    return {"loss": out["loss"],
            # Excess over the inverter limits. Always 0 with the safety layer. Used in the loss only by penalty-only (D1).
            "inverter_penalty": inverter.sum(dim=1),
            "voltage": ((v - 1.0) ** 2).sum(dim=1),
            "effort": (action ** 2).sum(dim=(1, 2)),
            "soc_penalty": (torch.relu(battery["soc_min"] - soc) ** 2 + torch.relu(soc - battery["soc_max"]) ** 2).sum(dim=1),
            "voltage_penalty": (limits_low + limits_high).sum(dim=1)}


def rollout(controller, data, starts, scale, soc0, horizon, battery, solver, use_projection, truncate=None):
    """Run the loop over windows and return the unweighted terms, as a mean over windows and steps.

    truncate: if set, gradients stop every `truncate` steps (truncated BPTT). The loss still covers the whole horizon,
    but SoC, previous V and previous action are detached at each segment border.
    """
    t = torch.as_tensor(starts)
    soc = soc0
    index = torch.as_tensor(data.bess_buses)
    v_bess = idle_voltage(data, t, scale, solver)[:, index]
    action = torch.zeros(len(t), len(data.bess_buses), 2)
    totals = {}
    for k in range(horizon):
        if truncate and k > 0 and k % truncate == 0:
            soc, v_bess, action = soc.detach(), v_bess.detach(), action.detach()
        out, action, soc = control_step(controller, data, t + k, scale, soc, v_bess, action, battery, solver, use_projection)
        for name, value in step_terms(out, action, soc, battery).items():
            totals[name] = totals.get(name, 0.0) + value
        v_bess = out["v"][:, index]
    terms = {name: value.mean() / horizon for name, value in totals.items()}
    # Not in eq. (16): keeps the short horizon from emptying the battery. Inside the dead band there is no penalty.
    terms["terminal_soc"] = (torch.relu((soc - 0.5).abs() - battery.get("terminal_soc_dead_band", 0.0)) ** 2).mean()
    return terms


def total_objective(terms, weights, use_penalties, use_inverter_penalty=False):
    """Weighted sum. Variants without penalties drop the soft constraints but keep the voltage-regulation term.
    use_inverter_penalty adds the soft inverter-limit penalty (only the penalty-only ablation, decision D1 of 2026-11-07)."""
    names = ["loss", "voltage", "effort", "terminal_soc"] + (["soc_penalty", "voltage_penalty"] if use_penalties else [])
    names += ["inverter_penalty"] if use_inverter_penalty else []
    return sum(weights[n] * terms[n] for n in names)


def is_unstable(grad_norms, stopped_on_nonfinite, after_step=100, limit=10.0, max_share=0.05):
    """The pre-specified rule of Experiment 2: a run is unstable if a loss or gradient was not finite, or if the gradient
    norm before clipping is above `limit` on more than `max_share` of the steps after `after_step`.
    grad_norms: one value per step, step 1 first."""
    if stopped_on_nonfinite:
        return True
    later = np.asarray(grad_norms, dtype=float)[after_step:]
    if len(later) == 0:
        return False
    return bool(not np.isfinite(later).all() or (later > limit).mean() > max_share)


def sample_windows(rng, all_starts, extreme_starts, batch, extreme_share):
    """A share of the windows contain an extreme ramp and the rest are uniform. This bias is stated in docs/decisions.md."""
    n_extreme = int(round(batch * extreme_share))
    return np.r_[rng.choice(extreme_starts, n_extreme), rng.choice(all_starts, batch - n_extreme)]


def windows_with_extreme(data, starts, horizon):
    """The window starts whose window holds at least one extreme-ramp step."""
    count = np.r_[0, np.cumsum(data.extreme.astype(int))]
    return starts[(count[starts + horizon] - count[starts]) > 0]


def closed_loop(controller, data, split, scale_value, battery, solver, use_projection, only=None,
                resume=None, on_checkpoint=None, checkpoint_every=288):
    """Run one split continuously. SoC carries over between steps and resets after a data gap. Steps not valid stay NaN.

    only: an optional boolean mask over all steps, to run a subset (for example the 2-day MPC pilot). Steps left out count
    as a gap, so SoC resets after them.
    on_checkpoint: if given, it is called every `checkpoint_every` steps with a dict that holds the outputs so far and the
    loop state (SoC, previous action, previous V). Passing that dict back as `resume` continues the loop exactly where it
    stopped, so a long run can survive an interruption. Without these arguments nothing changes.
    """
    mask = data.labels == SPLITS.index(split)
    if only is not None:
        mask = mask & only
    steps = np.flatnonzero(mask & data.valid)
    if len(steps) == 0:
        raise ValueError(f"no valid steps in {split}")
    n_total, n_bus, n_bess = len(data.labels), data.p_load.shape[1], len(data.bess_buses)
    v_all = np.full((n_total, n_bus), np.nan, dtype=np.float32)
    action_all = np.full((n_total, n_bess, 2), np.nan, dtype=np.float32)
    soc_all = np.full((n_total, n_bess), np.nan, dtype=np.float32)
    loss_all = np.full(n_total, np.nan, dtype=np.float32)
    clip_all = np.zeros((n_total, n_bess), dtype=bool)
    diagnostics = {}
    step = np.timedelta64(data.dt_min, "m")
    stamps = data.stamps.values
    scale = torch.tensor([scale_value], dtype=torch.float32)
    index = torch.as_tensor(data.bess_buses)
    first, soc, action, v_bess = 0, None, None, None
    if resume is not None:
        v_all, action_all, soc_all = resume["v"].copy(), resume["action"].copy(), resume["soc"].copy()
        loss_all, clip_all = resume["loss"].copy(), resume["soc_clip"].copy()
        first = int(resume["next_position"])
        if not np.array_equal(resume["steps"], steps):
            raise ValueError("the saved state belongs to a different set of steps")
        soc, action, v_bess = (torch.as_tensor(resume[k]) for k in ("state_soc", "state_action", "state_v_bess"))

    def state(next_position):
        return {"v": v_all, "action": action_all, "soc": soc_all, "loss": loss_all, "soc_clip": clip_all, "steps": steps,
                "next_position": next_position, "state_soc": soc.numpy(), "state_action": action.numpy(),
                "state_v_bess": v_bess.numpy()}

    with torch.no_grad():
        for position in range(first, len(steps)):
            t_value = steps[position]
            t = torch.tensor([t_value])
            if position == 0 or t_value != steps[position - 1] + 1 or stamps[t_value] - stamps[t_value - 1] != step:
                soc = torch.full((1, n_bess), battery["soc_after_gap"], dtype=torch.float32)
                action = torch.zeros(1, n_bess, 2)
                v_bess = idle_voltage(data, t, scale, solver)[:, index]
            out, action, soc = control_step(controller, data, t, scale, soc, v_bess, action, battery, solver, use_projection,
                                            diagnostics)
            v_bess = out["v"][:, index]
            v_all[t_value], action_all[t_value], soc_all[t_value] = out["v"][0].numpy(), action[0].numpy(), soc[0].numpy()
            loss_all[t_value] = out["loss"][0].item()
            clip_all[t_value] = diagnostics["soc_clip"][0].numpy()
            if on_checkpoint is not None and (position + 1) % checkpoint_every == 0 and position + 1 < len(steps):
                on_checkpoint(state(position + 1))
    return {"v": v_all, "action": action_all, "soc": soc_all, "loss": loss_all, "soc_clip": clip_all}


def closed_loop_batched(controller, batch, data, split, scale_value, battery, solver, use_projection):
    """The same continuous loop as closed_loop, for `batch` independent copies of one controller that differ only in their
    own settings (for example a DroopController whose dead band is a (batch, 1) tensor). Every copy sees the same steps.
    Returns a list of `batch` result dicts shaped exactly like those of closed_loop.
    """
    mask = data.labels == SPLITS.index(split)
    steps = np.flatnonzero(mask & data.valid)
    if len(steps) == 0:
        raise ValueError(f"no valid steps in {split}")
    n_total, n_bus, n_bess = len(data.labels), data.p_load.shape[1], len(data.bess_buses)
    v_all = np.full((batch, n_total, n_bus), np.nan, dtype=np.float32)
    action_all = np.full((batch, n_total, n_bess, 2), np.nan, dtype=np.float32)
    soc_all = np.full((batch, n_total, n_bess), np.nan, dtype=np.float32)
    loss_all = np.full((batch, n_total), np.nan, dtype=np.float32)
    clip_all = np.zeros((batch, n_total, n_bess), dtype=bool)
    diagnostics = {}
    step = np.timedelta64(data.dt_min, "m")
    stamps = data.stamps.values
    scale = torch.full((batch,), scale_value, dtype=torch.float32)
    index = torch.as_tensor(data.bess_buses)
    with torch.no_grad():
        for position, t_value in enumerate(steps):
            t = torch.full((batch,), int(t_value))
            if position == 0 or t_value != steps[position - 1] + 1 or stamps[t_value] - stamps[t_value - 1] != step:
                soc = torch.full((batch, n_bess), battery["soc_after_gap"], dtype=torch.float32)
                action = torch.zeros(batch, n_bess, 2)
                v_bess = idle_voltage(data, t, scale, solver)[:, index]
            out, action, soc = control_step(controller, data, t, scale, soc, v_bess, action, battery, solver, use_projection,
                                            diagnostics)
            v_bess = out["v"][:, index]
            v_all[:, t_value], action_all[:, t_value], soc_all[:, t_value] = out["v"].numpy(), action.numpy(), soc.numpy()
            loss_all[:, t_value] = out["loss"].numpy()
            clip_all[:, t_value] = diagnostics["soc_clip"].numpy()
    return [{"v": v_all[b], "action": action_all[b], "soc": soc_all[b], "loss": loss_all[b], "soc_clip": clip_all[b]}
            for b in range(batch)]


def share_q_sign_changes(result, data, split):
    """Share of battery steps where Q has the opposite sign to Q on the previous step. Only steps whose previous step is
    consecutive and valid in the same split count, and a zero Q has no sign. A sign of oscillation in a feedback rule."""
    used = (data.labels == SPLITS.index(split)) & data.valid
    gap_free = np.zeros_like(used)
    gap_free[1:] = used[1:] & used[:-1] & (np.diff(data.stamps.values) == np.timedelta64(data.dt_min, "m"))
    q = result["action"][..., 1]
    flips = (q[1:] * q[:-1]) < 0
    return float(flips[gap_free[1:]].mean())


def continuous_objective(result, data, split, battery, weights):
    """Mean over valid steps of the eq. (16) terms, with no added terminal term, so all controllers share one scale."""
    used = (data.labels == SPLITS.index(split)) & data.valid
    v, a, soc = result["v"][used], result["action"][used], result["soc"][used]
    terms = {"loss": result["loss"][used].mean(),
             "voltage": ((v - 1.0) ** 2).sum(axis=1).mean(),
             "effort": (a ** 2).sum(axis=(1, 2)).mean(),
             "soc_penalty": (np.maximum(battery["soc_min"] - soc, 0) ** 2 + np.maximum(soc - battery["soc_max"], 0) ** 2).sum(axis=1).mean(),
             "voltage_penalty": (np.maximum(0.95 - v, 0) ** 2 + np.maximum(v - 1.05, 0) ** 2).sum(axis=1).mean()}
    terms = {k: float(value) for k, value in terms.items()}
    return {"objective": sum(weights[k] * terms[k] for k in terms), "terms": terms}


def end_of_day_soc(result, data, split):
    """SoC (n_days, n_batteries) at the step stamped 00:00 that closes a scenario day, on days where that step is valid."""
    midnight = np.asarray(data.stamps.hour == 0) & np.asarray(data.stamps.minute == 0)
    return result["soc"][(data.labels == SPLITS.index(split)) & data.valid & midnight]


def share_soc_clip_active(result, data, split):
    """Share of battery steps where the energy limit cut P."""
    return float(result["soc_clip"][(data.labels == SPLITS.index(split)) & data.valid].mean())


def share_at_s_limit(result, data, split, battery, fraction=0.98):
    """Share of battery steps with apparent power at or above `fraction` of S_max."""
    a = result["action"][(data.labels == SPLITS.index(split)) & data.valid]
    return float((np.hypot(a[..., 0], a[..., 1]) >= fraction * battery["s_max_pu"]).mean())


def select_soc_variant(candidates, min_mean_abs_p_mw=0.025, soc_range=(0.2, 0.8)):
    """The pre-specified rule in docs/decisions.md: the lowest objective among candidates that use P and keep end-of-day SoC in range.

    candidates: {name: {"objective": float, "regimes": {regime: {"mean_abs_p_mw": [per battery], "end_of_day_soc": array}}}}
    Returns (selected name or None, {name: {"uses_active_power": bool, "soc_stays_in_range": bool, "passes": bool}}).
    """
    checks = {}
    for name, c in candidates.items():
        regimes = c["regimes"].values()
        uses_p = all(min(r["mean_abs_p_mw"]) >= min_mean_abs_p_mw for r in regimes)
        in_range = all(np.min(r["end_of_day_soc"]) >= soc_range[0] and np.max(r["end_of_day_soc"]) <= soc_range[1] for r in regimes)
        checks[name] = {"uses_active_power": bool(uses_p), "soc_stays_in_range": bool(in_range), "passes": bool(uses_p and in_range)}
    passing = [n for n in candidates if checks[n]["passes"]]
    return (min(passing, key=lambda n: candidates[n]["objective"]) if passing else None), checks


def closed_loop_summary(result, data, split, battery, metrics_config, include_rq2=True):
    """Metrics of one closed-loop run. All voltage metrics use only the valid steps of the split."""
    k = SPLITS.index(split)
    in_split = data.labels == k
    used = in_split & data.valid
    v = result["v"][used]
    outside = (v < 0.95) | (v > 1.05)
    a = result["action"][used]
    soc = result["soc"][used]
    limit_broken = (np.abs(a[..., 0]) > battery["p_max_pu"] * (1 + 1e-6)) | (np.abs(a[..., 1]) > battery["q_max_pu"] * (1 + 1e-6)) \
        | (np.hypot(a[..., 0], a[..., 1]) > battery["s_max_pu"] * (1 + 1e-5))
    rq2 = None
    if include_rq2:
        rq2 = summarize_rq2(result["v"][in_split], data.extreme[in_split], data.valid[in_split], ~np.isnan(data.r[in_split]),
                            data.labels[in_split], k, data.stamps[in_split], metrics_config, data.dt_min)
    return {"steps": int(used.sum()),
            "vvr_share_of_steps_with_any_violation": float(outside.any(axis=1).mean()),
            "vvr_share_of_bus_steps": float(outside.mean()),
            "v_rmse_pu": float(np.sqrt(((v - 1.0) ** 2).mean())),
            "v_min": float(v.min()), "v_max": float(v.max()),
            "mean_network_loss_kw": float(result["loss"][used].mean() * data.s_base * 1000),
            "mean_abs_p_mw": float(np.abs(a[..., 0]).mean() * data.s_base),
            "mean_abs_q_mvar": float(np.abs(a[..., 1]).mean() * data.s_base),
            "battery_stress_mean_p2_q2_pu": float((a ** 2).sum(axis=(1, 2)).mean()),
            "soc_min": float(soc.min()), "soc_max": float(soc.max()),
            "share_of_battery_steps_outside_soc_limits": float(((soc < battery["soc_min"]) | (soc > battery["soc_max"])).mean()),
            "share_of_battery_steps_breaking_inverter_limits": float(limit_broken.mean()),
            "rq2": rq2}


def make_policy(config):
    return DPCPolicy(N_STATE, N_DISTURBANCE, control_dim=4, hidden=config["policy"]["hidden"])
