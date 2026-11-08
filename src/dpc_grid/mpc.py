"""Milestone M4: online MPC baseline with a convex LinDistFlow model (plan in docs/decisions.md, "M4 plan").

At every step the MPC solves one convex problem over the horizon H and applies only the first action (receding horizon).
The plant is NOT this model: the closed loop still uses the nonlinear DistFlow solver, exactly as for DPC.

Model inside the MPC (per unit, the same sign conventions as physics.py; P_bess > 0 is discharge, Q > 0 injects):
  branch flows   P = P0 - P_bess @ M,  Q = Q0 - Q_bess @ M,  with P0, Q0 the flows without batteries and
                 M[b, k] = 1 if battery b sits in the subtree of branch k (so its power changes the flow on branch k)
  voltages       v2 = V_slack^2 - 2 (r P + x Q) @ subtree       (LinDistFlow: the loss term z^2 I^2 is dropped)
                 V ~ (1 + v2) / 2                                  (sqrt linearised around 1 pu)
  losses         sum_k r_k (P_k^2 + Q_k^2)                         (I^2 ~ P^2 + Q^2 at V ~ 1 pu)
  battery        soc_model="lossless" (item 7 as changed on 2026-11-06, used from then on): one signed P,
                   SoC_{k+1} = SoC_k - P dt / E, so simultaneous charge and discharge is impossible; the plant keeps eta = 0.95
                 soc_model="efficiency" (the original item 7, superseded): P_bess = P_dis - P_ch, both >= 0, a convex
                   relaxation, SoC_{k+1} = SoC_k + (eta_ch P_ch - P_dis / eta_dis) dt / E. It burned energy by charging and
                   discharging at once (docs/decisions.md, overlap diagnostic).
                 10% <= SoC <= 90% (hard) in both
  inverter       |P| <= P_max, |Q| <= Q_max, sqrt(P^2 + Q^2) <= S_max (a second-order cone, exact and convex)
Objective: the eq. (16) terms with the configured weights, as a mean over the horizon steps like the DPC rollout, voltage limits
as squared slack with weight 1000 (soft), and the same terminal SoC term with the dead zone. The SoC penalty is zero because SoC
is a hard constraint here.

Differences from the proposal (stated wherever MPC is shown): convex LinDistFlow instead of AC OPF, and soft voltage limits.
"""
import time

import cvxpy as cp
import numpy as np
import torch


class LinDistFlowMPC:
    """Receding-horizon MPC. Build once; each call only updates parameter values (cvxpy DPP), then solves with Clarabel."""

    needs_context = True  # control_step passes t, scale and SoC to controllers that set this

    def __init__(self, solver_net, data, battery, weights, horizon, forecast="persistence", solver="CLARABEL",
                 overlap_penalty=0.0, soc_model="efficiency"):
        """forecast: "persistence" (the current step held over the horizon) or "oracle" (the exact future, an upper bound).
        soc_model: "efficiency" (P split into P_ch, P_dis with efficiencies, the original item 7) or "lossless" (one signed P,
        SoC_{k+1} = SoC_k - P dt / E; the changed item 7 of 2026-11-06, which makes simultaneous charge and discharge
        impossible). overlap_penalty: weight of sum(P_ch + P_dis) / H, only for the efficiency model; 0 is the original."""
        if forecast not in ("persistence", "oracle"):
            raise ValueError(f"unknown forecast {forecast}")
        if soc_model not in ("efficiency", "lossless"):
            raise ValueError(f"unknown soc_model {soc_model}")
        if overlap_penalty and soc_model == "lossless":
            raise ValueError("the overlap penalty only exists in the efficiency model")
        self.data, self.battery, self.weights, self.horizon, self.solver = data, battery, weights, horizon, solver
        self.forecast, self.soc_model = forecast, soc_model
        # Steps whose data the oracle may use: valid steps of the validation part (M3Data cannot read the test part).
        self.available = np.asarray((data.labels == 1) & data.valid)
        subtree = solver_net.subtree.numpy()
        r, x = solver_net.r.numpy(), solver_net.x.numpy()
        n_bus, bess = subtree.shape[0], list(data.bess_buses)
        n_bess = len(bess)
        self.subtree, self.r, self.x, self.slack = subtree, r, x, solver_net.v_slack ** 2
        M = subtree[:, bess].T  # (n_bess, n_bus)
        H = horizon

        self.P0 = cp.Parameter((H, n_bus))
        self.Q0 = cp.Parameter((H, n_bus))
        self.soc0 = cp.Parameter(n_bess)
        q = cp.Variable((H, n_bess))
        soc = cp.Variable((H + 1, n_bess))
        s_low = cp.Variable((H, n_bus), nonneg=True)
        s_high = cp.Variable((H, n_bus), nonneg=True)
        if soc_model == "efficiency":
            p_dis = cp.Variable((H, n_bess), nonneg=True)
            p_ch = cp.Variable((H, n_bess), nonneg=True)
            p = p_dis - p_ch
        else:
            p = cp.Variable((H, n_bess))

        P = self.P0 - p @ M
        Q = self.Q0 - q @ M
        v2 = self.slack - 2 * (P @ np.diag(r) + Q @ np.diag(x)) @ subtree
        V = (1 + v2) / 2
        s_base, dt_h, energy = data.s_base, data.dt_h, battery["energy_mwh"]

        if soc_model == "efficiency":
            constraints = [p_dis <= battery["p_max_pu"], p_ch <= battery["p_max_pu"],
                           soc[1:] == soc[:-1] + (battery["eta_ch"] * p_ch - p_dis / battery["eta_dis"]) * s_base * dt_h / energy]
        else:
            constraints = [cp.abs(p) <= battery["p_max_pu"], soc[1:] == soc[:-1] - p * s_base * dt_h / energy]
        constraints += [cp.abs(q) <= battery["q_max_pu"], soc[0] == self.soc0,
                        soc >= battery["soc_min"], soc <= battery["soc_max"],
                        V >= 0.95 - s_low, V <= 1.05 + s_high]
        for b in range(n_bess):
            constraints.append(cp.norm(cp.vstack([p[:, b], q[:, b]]), 2, axis=0) <= battery["s_max_pu"])

        sqrt_r = np.sqrt(r)
        terms = {"loss": (cp.sum_squares(P @ np.diag(sqrt_r)) + cp.sum_squares(Q @ np.diag(sqrt_r))) / H,
                 "voltage": cp.sum_squares(V - 1) / H,
                 "effort": (cp.sum_squares(p) + cp.sum_squares(q)) / H,
                 "voltage_penalty": (cp.sum_squares(s_low) + cp.sum_squares(s_high)) / H}
        dead_band = battery.get("terminal_soc_dead_band", 0.0)
        terminal = cp.sum_squares(cp.pos(cp.abs(soc[H] - 0.5) - dead_band)) / n_bess
        objective = sum(weights[k] * terms[k] for k in terms) + weights["terminal_soc"] * terminal
        if overlap_penalty:
            objective = objective + overlap_penalty * cp.sum(p_ch + p_dis) / H
        self.problem = cp.Problem(cp.Minimize(objective), constraints)
        assert self.problem.is_dcp(dpp=True)
        self.vars = {"p": p, "q": q, "soc": soc, "v": V}
        if soc_model == "efficiency":
            self.vars.update({"p_dis": p_dis, "p_ch": p_ch})
        self.log = []  # one record per call: raw action, solve time, status, simultaneous charge and discharge

    def forecast_steps(self, t):
        """Data steps used for the horizon t .. t+H-1, and how many of them are held values.

        persistence: step t for every horizon step. oracle: the true step t+k, but where t+k is outside the validation part
        or not valid (data end or gap), the last known step along the horizon is held instead."""
        if self.forecast == "persistence":
            return np.full(self.horizon, t), 0
        steps, held, last = np.empty(self.horizon, dtype=int), 0, t
        for k in range(self.horizon):
            s = t + k
            if s < len(self.available) and self.available[s]:
                last = s
            else:
                held += 1
            steps[k] = last
        return steps, held

    def base_flows(self, t, scale):
        """Branch flows without batteries over the horizon, from the forecast steps (held values for persistence)."""
        steps, held = self.forecast_steps(t)
        p_net = (scale * self.data.p_load[steps] - self.data.p_solar[steps]).double().numpy()
        q_net = (scale * self.data.q_load[steps]).double().numpy()
        self.last_held = held
        return p_net @ self.subtree.T, q_net @ self.subtree.T

    def __call__(self, state, disturbance, v_bess, t=None, scale=None, soc=None):
        assert len(t) == 1, "the MPC acts on one trajectory at a time"
        self.P0.value, self.Q0.value = self.base_flows(int(t[0]), float(scale[0]))
        self.soc0.value = soc[0].double().numpy()
        start = time.perf_counter()
        try:
            self.problem.solve(solver=self.solver, warm_start=True)
            status = self.problem.status
        except cp.error.SolverError:
            status = "solver_error"
        seconds = time.perf_counter() - start
        n_bess = soc.shape[1]
        if status in ("optimal", "optimal_inaccurate"):
            q = self.vars["q"].value[0]
            if self.soc_model == "efficiency":
                p_dis, p_ch = self.vars["p_dis"].value[0], self.vars["p_ch"].value[0]
                action = np.stack([p_dis - p_ch, q], axis=1)
                simultaneous = ((p_dis > 1e-6) & (p_ch > 1e-6)).tolist()
            else:  # one signed P: simultaneous charge and discharge cannot happen
                action = np.stack([self.vars["p"].value[0], q], axis=1)
                simultaneous = [False] * soc.shape[1]
        else:  # no solution: the batteries stay idle on this step, and the failure is logged
            action, simultaneous = np.zeros((n_bess, 2)), [False] * n_bess
        self.log.append({"t": int(t[0]), "status": status, "solve_seconds": seconds, "action": action.tolist(),
                         "simultaneous_charge_discharge": simultaneous, "held_forecast_steps": self.last_held})
        return torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
