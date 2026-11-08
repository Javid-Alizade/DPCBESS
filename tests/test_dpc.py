"""M3: the closed loop, the loss, the references, the test-part refusal and the run log, on small artificial data."""
import json
import subprocess

import numpy as np
import pandas as pd
import pytest
import torch

from dpc_grid import load_case33bw
from dpc_grid.dpc import (ClosedTestPartError, DroopController, IdleController, M3Data, PolicyController, closed_loop,
                          closed_loop_batched, closed_loop_summary, share_q_sign_changes, continuous_objective, control_step, end_of_day_soc, energy_power_limits,
                          is_unstable, make_policy, per_unit_battery, rollout, safety_layer, sample_windows, select_soc_variant,
                          share_at_s_limit, share_soc_clip_active, step_terms, total_objective, windows_with_extreme)
from dpc_grid.physics import bess_soc_step
from dpc_grid.runlog import git_executable, start_run

BATTERY = {"p_max_mw": 0.5, "q_max_mvar": 0.5, "s_max_mva": 0.5, "energy_mwh": 1.0, "eta_ch": 0.95, "eta_dis": 0.95,
           "soc_min": 0.1, "soc_max": 0.9, "soc_start_range": [0.3, 0.7], "soc_after_gap": 0.5}
WEIGHTS = {"loss": 1.0, "voltage": 1.0, "effort": 0.01, "soc_penalty": 100.0, "voltage_penalty": 1000.0, "terminal_soc": 10.0}
PV, BESS = [14, 17, 29, 32], [17, 32]
H = 6


@pytest.fixture(scope="module")
def net():
    return load_case33bw()


@pytest.fixture(scope="module")
def battery(net):
    return per_unit_battery(BATTERY, net.s_base_mva)


@pytest.fixture(scope="module")
def solver(net):
    return net.make_solver(n_iter=15)


@pytest.fixture()
def data_path(net, tmp_path):
    """A small file shaped like the M2 dataset: 100 train, 100 val and 40 test steps, with test values set to a marker."""
    n = 240
    labels = np.r_[np.zeros(100), np.ones(100), np.full(40, 2)].astype(np.int8)
    stamps = pd.date_range("2012-06-01 00:05", periods=n, freq="5min").to_numpy().astype("datetime64[s]").astype(np.int64)
    shape = 0.6 + 0.2 * np.sin(np.arange(n) / 20)[:, None]
    p_solar = np.full((n, 33), np.nan, dtype=np.float32)
    p_solar[:, PV] = (0.04 * (1 + np.cos(np.arange(n) / 15)))[:, None]
    valid = np.ones(n, dtype=bool)
    valid[150] = False  # a gap in the validation part
    extreme = np.zeros(n, dtype=bool)
    extreme[[10, 30, 120]] = True
    r = 0.01 * np.sin(np.arange(n) / 7)
    r[0] = np.nan
    p_load = shape * net.p_load
    p_load[labels == 2] = 99.0  # nothing may ever read these values
    path = tmp_path / "small.npz"
    np.savez(path, stamps=stamps, labels=labels, valid=valid, extreme=extreme, r_mw_per_min=r, p_solar=p_solar,
             p_load_x1=p_load.astype(np.float32), q_load_x1=(shape * net.q_load).astype(np.float32))
    return path


@pytest.fixture()
def data(data_path):
    return M3Data(data_path, 10.0, 5).prepare(PV, BESS)


def test_test_part_is_refused_and_dropped(data_path, data):
    with pytest.raises(ClosedTestPartError):
        M3Data(data_path, 10.0, 5, splits=("train", "val", "test"))
    assert 2 not in set(data.labels.tolist())
    assert len(data.labels) == 200
    assert data.p_load.max() < 99.0


def test_a_constant_input_stops_with_a_clear_error_instead_of_nan(data_path):
    bad = M3Data(data_path, 10.0, 5)
    bad.r[:] = 0.0
    with pytest.raises(ValueError, match="constant"):
        bad.prepare(PV, BESS)


def test_normalisation_comes_from_the_training_part(data_path, data):
    train_total = data.total_load[torch.as_tensor(data.labels == 0)]
    assert data.load_mean == pytest.approx(train_total.mean().item())
    assert data.exogenous[torch.as_tensor(data.labels == 0), 0].mean().item() == pytest.approx(0.0, abs=1e-5)


def one_step(controller, data, battery, solver, use_projection=True, t=20, scale=0.7, soc=0.5):
    t_tensor = torch.tensor([t])
    v_bess = torch.ones(1, 2)
    return control_step(controller, data, t_tensor, torch.tensor([scale]), torch.full((1, 2), soc), v_bess,
                        torch.zeros(1, 2, 2), battery, solver, use_projection)


class Fixed:
    """A controller that always returns the same action, in per unit."""

    def __init__(self, p, q):
        self.p, self.q = p, q

    def __call__(self, state, disturbance, v_bess):
        return torch.tensor([[[self.p, self.q], [self.p, self.q]]]).repeat(v_bess.shape[0], 1, 1)


def test_discharge_raises_voltage_and_lowers_soc_with_the_right_energy(data, battery, solver):
    idle, _, _ = one_step(IdleController(), data, battery, solver)
    out, action, soc = one_step(Fixed(0.05, 0.0), data, battery, solver)
    assert (out["v"][0, BESS] > idle["v"][0, BESS]).all()
    # 0.5 MW for 5 minutes through 95% efficiency, from a 1 MWh battery that started at 50%
    assert soc[0, 0].item() == pytest.approx(0.5 - 0.5 * (5 / 60) / 0.95 / 1.0, rel=1e-5)


def test_charging_raises_soc_with_the_right_energy(data, battery, solver):
    _, _, soc = one_step(Fixed(-0.05, 0.0), data, battery, solver)
    assert soc[0, 0].item() == pytest.approx(0.5 + 0.95 * 0.5 * (5 / 60), rel=1e-5)


def test_reactive_power_raises_voltage_and_leaves_soc_alone(data, battery, solver):
    idle, _, _ = one_step(IdleController(), data, battery, solver)
    out, _, soc = one_step(Fixed(0.0, 0.05), data, battery, solver)
    assert (out["v"][0, BESS] > idle["v"][0, BESS]).all()
    assert soc[0, 0].item() == 0.5


def test_projection_keeps_the_inverter_limits_and_can_be_turned_off(data, battery, solver):
    _, action, _ = one_step(Fixed(0.3, 0.3), data, battery, solver)
    assert torch.hypot(action[..., 0], action[..., 1]).max().item() <= battery["s_max_pu"] * (1 + 1e-5)
    _, free, _ = one_step(Fixed(0.3, 0.3), data, battery, solver, use_projection=False)
    assert free[0, 0, 0].item() == pytest.approx(0.3)


def test_energy_limits_match_the_soc_equation(battery):
    soc = torch.tensor([[0.5, 0.11]])
    dt, s_base = 5 / 60, 10.0
    discharge, charge = energy_power_limits(soc, battery, dt, s_base)
    assert discharge[0, 0].item() * s_base == pytest.approx(0.4 * 1.0 * 0.95 / dt, rel=1e-4)
    assert charge[0, 0].item() * s_base == pytest.approx(0.4 * 1.0 / (0.95 * dt), rel=1e-4)
    # Using exactly the allowed power lands exactly on the limit.
    after = bess_soc_step(soc, discharge * s_base, 1.0, dt, 0.95, 0.95)
    assert after[0, 0].item() == pytest.approx(0.1, abs=1e-5) and after[0, 1].item() == pytest.approx(0.1, abs=1e-5)
    after = bess_soc_step(soc, -charge * s_base, 1.0, dt, 0.95, 0.95)
    assert after[0, 0].item() == pytest.approx(0.9, abs=1e-5)


def test_safety_layer_cuts_p_near_the_limits_only_in_the_blocked_direction(battery):
    dt, s_base = 5 / 60, 10.0
    wants_discharge = torch.tensor([[[0.05, 0.0], [0.05, 0.0]]])
    safe, clipped = safety_layer(wants_discharge, torch.tensor([[0.1, 0.5]]), battery, dt, s_base)
    assert safe[0, 0, 0].item() == 0.0 and safe[0, 1, 0].item() == pytest.approx(0.05, rel=1e-4)
    assert clipped.tolist() == [[True, False]]
    wants_charge = -wants_discharge
    safe, clipped = safety_layer(wants_charge, torch.tensor([[0.1, 0.9]]), battery, dt, s_base)
    assert safe[0, 0, 0].item() == pytest.approx(-0.05, rel=1e-4)  # an empty battery may still charge
    assert safe[0, 1, 0].item() == 0.0                             # a full battery may not
    assert clipped.tolist() == [[False, True]]


def test_safety_layer_keeps_inverter_limits_and_does_not_count_them_as_soc_clip(battery):
    big = torch.tensor([[[0.3, 0.3], [-0.3, -0.3]]])
    safe, clipped = safety_layer(big, torch.tensor([[0.5, 0.5]]), battery, 5 / 60, 10.0)
    assert torch.hypot(safe[..., 0], safe[..., 1]).max().item() <= battery["s_max_pu"] * (1 + 1e-5)
    assert safe[..., 1].abs().max().item() <= battery["q_max_pu"] * (1 + 1e-6)
    assert not clipped.any()  # at 50% SoC only the inverter limits P


def test_q_is_not_blocked_by_an_empty_battery(battery):
    action = torch.tensor([[[0.05, 0.03], [0.05, 0.03]]])
    safe, _ = safety_layer(action, torch.tensor([[0.1, 0.1]]), battery, 5 / 60, 10.0)
    assert safe[0, 0, 0].item() == 0.0 and safe[0, 0, 1].item() == pytest.approx(0.03, rel=1e-4)


def test_gradient_flows_through_the_energy_limit(battery):
    soc = torch.tensor([[0.11, 0.5]], requires_grad=True)
    raw = torch.tensor([[[0.05, 0.0], [0.01, 0.0]]], requires_grad=True)
    safe, _ = safety_layer(raw, soc, battery, 5 / 60, 10.0)
    safe[..., 0].sum().backward()
    assert soc.grad[0, 0].item() > 0   # the first battery sits on the energy limit, so its P depends on SoC
    assert raw.grad[0, 1, 0].item() > 0  # the second is free, so its P follows the policy


@pytest.mark.parametrize("p", [0.05, -0.05, 5.0, -5.0])
def test_soc_stays_inside_its_limits_over_a_long_loop(data, battery, solver, p):
    result = closed_loop(Fixed(p, 0.0), data, "val", 0.7, battery, solver, True)
    used = (data.labels == 1) & data.valid
    soc = result["soc"][used]
    assert soc.min() >= 0.1 - 1e-5 and soc.max() <= 0.9 + 1e-5
    assert share_soc_clip_active(result, data, "val") > 0  # the limit is reached and reported


def test_without_the_projection_there_is_no_energy_limit(data, battery, solver):
    result = closed_loop(Fixed(0.05, 0.0), data, "val", 0.7, battery, solver, False)
    assert result["soc"][(data.labels == 1) & data.valid].min() < 0.1
    assert share_soc_clip_active(result, data, "val") == 0


def test_droop_dead_band_full_output_and_sign(battery):
    droop = DroopController(battery, dead_band_pu=0.01, full_at_pu=0.05)
    v = torch.tensor([[1.005, 0.995], [0.95, 1.05], [0.97, 1.03], [0.90, 1.10]])
    action = droop(None, None, v)
    q = battery["q_max_pu"]
    assert action[0, :, 1].tolist() == [0.0, 0.0]                 # inside the dead band
    assert action[1, :, 1].numpy() == pytest.approx([q, -q], rel=1e-4)      # full output, injecting at low V
    assert action[2, :, 1].numpy() == pytest.approx([q / 2, -q / 2], rel=1e-4)
    assert action[3, :, 1].numpy() == pytest.approx([q, -q], rel=1e-4)      # saturates
    assert (action[..., 0] == 0).all()                            # no active power, so no SoC use


def test_droop_moves_a_low_voltage_up(data, battery, solver):
    droop = DroopController(battery, 0.01, 0.05)
    t = torch.tensor([20])
    args = (data, t, torch.tensor([1.0]), torch.full((1, 2), 0.5), torch.full((1, 2), 0.95), torch.zeros(1, 2, 2), battery, solver, True)
    idle = control_step(IdleController(), *args)[0]
    out = control_step(droop, *args)[0]
    assert (out["v"][0, BESS] > idle["v"][0, BESS]).all()


def test_terms_match_their_formulas(battery):
    v = torch.tensor([[1.0, 0.94, 1.06]])
    out = {"v": v, "loss": torch.tensor([0.002])}
    action = torch.tensor([[[0.03, 0.04], [0.0, 0.0]]])
    soc = torch.tensor([[0.05, 0.95]])
    terms = step_terms(out, action, soc, battery)
    assert terms["loss"].item() == pytest.approx(0.002, rel=1e-4)
    assert terms["voltage"].item() == pytest.approx(0.06 ** 2 * 2, rel=1e-4)
    assert terms["effort"].item() == pytest.approx(0.03 ** 2 + 0.04 ** 2, rel=1e-4)
    assert terms["soc_penalty"].item() == pytest.approx(2 * 0.05 ** 2, rel=1e-4)
    assert terms["voltage_penalty"].item() == pytest.approx(2 * 0.01 ** 2, rel=1e-4)  # float32 limits the precision here


def test_inverter_penalty_matches_its_formula(battery):
    out = {"v": torch.ones(1, 3), "loss": torch.zeros(1)}
    p_max, q_max, s_max = battery["p_max_pu"], battery["q_max_pu"], battery["s_max_pu"]
    action = torch.tensor([[[p_max + 0.01, 0.0], [0.8 * s_max, 0.8 * s_max]]])  # battery 1 over |P|, battery 2 over S only
    terms = step_terms(out, action, torch.full((1, 2), 0.5), battery)
    s2 = (0.8 * s_max) * 2 ** 0.5
    expected = 0.01 ** 2 + (p_max + 0.01 - s_max) ** 2 + (s2 - s_max) ** 2
    assert terms["inverter_penalty"].item() == pytest.approx(expected, rel=1e-3)


def test_inverter_penalty_is_zero_inside_limits_with_finite_gradient_at_zero(battery):
    out = {"v": torch.ones(1, 3), "loss": torch.zeros(1)}
    action = torch.zeros(1, 2, 2, requires_grad=True)
    terms = step_terms(out, action, torch.full((1, 2), 0.5), battery)
    assert terms["inverter_penalty"].item() == 0
    terms["inverter_penalty"].sum().backward()
    assert torch.isfinite(action.grad).all()


def test_objective_adds_the_inverter_penalty_only_when_asked():
    terms = {k: torch.tensor(1.0) for k in list(WEIGHTS) + ["inverter_penalty"]}
    weights = dict(WEIGHTS, inverter_penalty=1000.0)
    assert (total_objective(terms, weights, True, True) - total_objective(terms, weights, True)).item() == pytest.approx(1000.0)


def test_voltages_inside_limits_have_no_penalty(battery):
    out = {"v": torch.tensor([[1.0, 0.95, 1.05]]), "loss": torch.zeros(1)}
    terms = step_terms(out, torch.zeros(1, 2, 2), torch.full((1, 2), 0.5), battery)
    assert terms["voltage_penalty"].item() == 0 and terms["soc_penalty"].item() == 0


def test_objective_without_penalties_drops_only_the_soft_constraints():
    terms = {k: torch.tensor(1.0) for k in WEIGHTS}
    full = total_objective(terms, WEIGHTS, True).item()
    bare = total_objective(terms, WEIGHTS, False).item()
    assert full == pytest.approx(sum(WEIGHTS.values()))
    assert full - bare == pytest.approx(WEIGHTS["soc_penalty"] + WEIGHTS["voltage_penalty"])


def make_controller(battery):
    torch.manual_seed(0)
    return PolicyController(make_policy({"policy": {"hidden": 16}}), battery)


def test_gradients_reach_the_policy_through_the_whole_horizon(data, battery, solver):
    controller = make_controller(battery)
    starts = data.windows("train", H)[:4]
    terms = rollout(controller, data, starts, torch.tensor([0.7, 1.0, 0.7, 1.0]), torch.full((4, 2), 0.5), H, battery, solver, True)
    total_objective(terms, WEIGHTS, True).backward()
    grads = [p.grad for p in controller.policy.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0


def test_action_does_not_depend_on_future_load_or_solar(data, battery, solver):
    controller = make_controller(battery)
    t = 40
    _, before, _ = one_step(controller, data, battery, solver, t=t)
    data.p_load[t + 1:] *= 5
    data.p_solar[t + 1:] *= 5
    data.exogenous[t + 1:] *= 5
    _, after, _ = one_step(controller, data, battery, solver, t=t)
    assert torch.equal(before, after)


def test_idle_rollout_keeps_soc_and_matches_no_battery_voltages(data, battery, solver):
    t = data.windows("train", H)[:1]
    terms = rollout(IdleController(), data, t, torch.tensor([0.7]), torch.full((1, 2), 0.5), H, battery, solver, True)
    assert terms["effort"].item() == 0 and terms["terminal_soc"].item() == 0
    total = 0.0
    for k in range(H):
        step = t[0] + k
        v = solver(0.7 * data.p_load[[step]], 0.7 * data.q_load[[step]], data.p_solar[[step]])["v"]
        total += ((v - 1) ** 2).sum().item()
    assert terms["voltage"].item() == pytest.approx(total / H, rel=1e-5)


def test_window_sampling_share_and_extreme_selection(data):
    starts = data.windows("train", H)
    extreme = windows_with_extreme(data, starts, H)
    assert set(extreme.tolist()) == set(range(5, 11)) | set(range(25, 31))
    batch = sample_windows(np.random.default_rng(0), starts, extreme, 10, 0.3)
    assert len(batch) == 10
    assert set(batch[:3].tolist()) <= set(extreme.tolist())


def test_windows_stay_inside_a_split_and_skip_the_gap(data):
    starts = data.windows("val", H)
    assert starts.min() == 100 and starts.max() == 200 - H
    assert not any(s <= 150 < s + H for s in starts)


def test_closed_loop_carries_soc_and_resets_after_a_gap(data, battery, solver):
    result = closed_loop(Fixed(0.05, 0.0), data, "val", 0.7, battery, solver, True)
    soc = result["soc"][:, 0]
    one_step_drop = 0.5 * (5 / 60) / 0.95
    assert soc[100] == pytest.approx(0.5 - one_step_drop, rel=1e-4)
    assert soc[104] == pytest.approx(0.5 - 5 * one_step_drop, rel=1e-3)   # it carries over step after step
    assert soc[149] == pytest.approx(0.1, abs=1e-5)                       # and stops at the lower limit
    assert np.isnan(soc[150])                                             # the gap has no result
    assert soc[151] == pytest.approx(0.5 - one_step_drop, rel=1e-4)       # it restarts after the gap
    assert np.isnan(soc[:100]).all()                                      # the other split is untouched


def test_closed_loop_summary_reports_limits_and_both_kinds_of_violation(data, battery, solver):
    config = {"limits_pu": [0.95, 1.05], "safe_band_pu": [0.96, 1.04], "window_steps": 3, "bootstrap_samples": 20, "confidence": 0.95, "seed": 0}
    result = closed_loop(Fixed(0.3, 0.3), data, "val", 0.7, battery, solver, False)  # no projection: limits are broken
    summary = closed_loop_summary(result, data, "val", battery, config)
    assert summary["steps"] == 99
    assert summary["share_of_battery_steps_breaking_inverter_limits"] == 1.0
    assert summary["share_of_battery_steps_outside_soc_limits"] > 0
    assert summary["rq2"]["primary_over_ramp_windows"]["extreme_steps"] == 1  # the extreme step of val
    idle = closed_loop_summary(closed_loop(IdleController(), data, "val", 0.7, battery, solver, True), data, "val", battery, config)
    assert idle["share_of_battery_steps_breaking_inverter_limits"] == 0.0 and idle["mean_abs_p_mw"] == 0.0


def test_terminal_penalty_dead_band(data, battery, solver):
    t = data.windows("train", 1)[:1]
    def terminal(soc, band):
        settings = dict(battery, terminal_soc_dead_band=band)
        return rollout(IdleController(), data, t, torch.tensor([0.7]), torch.full((1, 2), soc), 1, settings, solver, True)["terminal_soc"].item()
    assert terminal(0.65, 0.2) == 0                                 # inside 30-70%: no penalty
    assert terminal(0.8, 0.2) == pytest.approx(0.1 ** 2, rel=1e-4)  # outside: squared distance to the zone
    assert terminal(0.2, 0.2) == pytest.approx(0.1 ** 2, rel=1e-4)
    assert terminal(0.8, 0.0) == pytest.approx(0.3 ** 2, rel=1e-4)  # no dead band: the distance to 50%


def test_continuous_objective_matches_a_hand_calculation_and_leaves_out_the_terminal_term(data, battery):
    n = len(data.labels)
    v = np.ones((n, 33), dtype=np.float32)
    v[:, 1] = 0.94
    action = np.zeros((n, 2, 2), dtype=np.float32)
    action[:, 0, 0] = 0.03
    result = {"v": v, "action": action, "soc": np.full((n, 2), 0.05, dtype=np.float32), "loss": np.full(n, 0.002, dtype=np.float32)}
    out = continuous_objective(result, data, "val", battery, WEIGHTS)
    assert out["terms"]["voltage"] == pytest.approx(0.06 ** 2, rel=1e-4)
    assert out["terms"]["voltage_penalty"] == pytest.approx(0.01 ** 2, rel=1e-3)
    assert out["terms"]["effort"] == pytest.approx(0.03 ** 2, rel=1e-4)
    assert out["terms"]["soc_penalty"] == pytest.approx(2 * 0.05 ** 2, rel=1e-4)
    assert "terminal_soc" not in out["terms"]
    expected = 0.002 + 0.06 ** 2 + 0.01 * 0.03 ** 2 + 100 * 2 * 0.05 ** 2 + 1000 * 0.01 ** 2
    assert out["objective"] == pytest.approx(expected, rel=1e-3)


def test_continuous_objective_ignores_the_other_split_and_invalid_steps(data, battery):
    n = len(data.labels)
    base = {"v": np.ones((n, 33), dtype=np.float32), "action": np.zeros((n, 2, 2), dtype=np.float32),
            "soc": np.full((n, 2), 0.5, dtype=np.float32), "loss": np.full(n, 0.002, dtype=np.float32)}
    changed = {k: value.copy() for k, value in base.items()}
    changed["loss"][data.labels == 0] = 9.0  # the training part
    changed["loss"][150] = 9.0               # the invalid step in validation
    assert continuous_objective(changed, data, "val", battery, WEIGHTS)["objective"] == continuous_objective(base, data, "val", battery, WEIGHTS)["objective"]


def test_end_of_day_uses_only_valid_midnight_steps(data):
    n = len(data.labels)
    soc = np.arange(2 * n, dtype=np.float32).reshape(n, 2)
    days = end_of_day_soc({"soc": soc}, data, "val")
    midnight = np.flatnonzero((data.stamps.hour == 0) & (data.stamps.minute == 0) & (data.labels == 1) & data.valid)
    assert len(days) == len(midnight) and (days == soc[midnight]).all()


def test_s_limit_share_counts_battery_steps(data, battery):
    n = len(data.labels)
    action = np.zeros((n, 2, 2), dtype=np.float32)
    action[:, 0, 1] = battery["s_max_pu"]  # one of two batteries is at the limit on every step
    assert share_at_s_limit({"action": action}, data, "val", battery) == 0.5


def candidate(objective, p, low, high):
    soc = np.array([[low, high], [0.5, 0.5]])
    regime = {"mean_abs_p_mw": p, "end_of_day_soc": soc}
    return {"objective": objective, "regimes": {"base": regime, "stress": regime}}


def test_selection_rule_takes_the_lowest_objective_among_those_that_pass():
    candidates = {"no_p": candidate(0.001, [0.0, 0.0], 0.5, 0.5),       # the lowest, but it does not use P
                  "drifts": candidate(0.002, [0.1, 0.1], 0.1, 0.5),     # uses P, but SoC leaves the range
                  "good": candidate(0.005, [0.1, 0.1], 0.3, 0.7),
                  "worse": candidate(0.006, [0.1, 0.1], 0.3, 0.7)}
    selected, checks = select_soc_variant(candidates)
    assert selected == "good"
    assert not checks["no_p"]["uses_active_power"] and not checks["drifts"]["soc_stays_in_range"]


def test_selection_rule_needs_every_battery_and_every_regime():
    one_idle = candidate(0.005, [0.1, 0.0], 0.3, 0.7)
    assert select_soc_variant({"x": one_idle})[0] is None
    stress_drift = {"objective": 0.005, "regimes": {
        "base": {"mean_abs_p_mw": [0.1, 0.1], "end_of_day_soc": np.array([[0.5, 0.5]])},
        "stress": {"mean_abs_p_mw": [0.1, 0.1], "end_of_day_soc": np.array([[0.5, 0.85]])}}}
    assert select_soc_variant({"x": stress_drift})[0] is None


def test_truncated_bptt_cuts_the_gradient_path_but_keeps_the_loss(data, battery, solver):
    starts = data.windows("train", H)[:2]
    scale = torch.tensor([0.7, 1.0])
    def run(truncate):
        soc0 = torch.full((2, 2), 0.6, requires_grad=True)
        controller = make_controller(battery)
        terms = rollout(controller, data, starts, scale, soc0, H, battery, solver, True, truncate=truncate)
        terms["terminal_soc"].backward()  # depends only on the last step
        return terms, soc0.grad, [p.grad for p in controller.policy.parameters()]
    full, soc_grad_full, _ = run(None)
    cut, soc_grad_cut, policy_grads = run(2)
    assert soc_grad_full.abs().sum() > 0      # without truncation the last step reaches the start
    assert soc_grad_cut is None or soc_grad_cut.abs().sum() == 0  # with truncation it does not
    assert sum(g.abs().sum().item() for g in policy_grads if g is not None) > 0  # the policy still learns
    for name in full:
        assert cut[name].item() == pytest.approx(full[name].item(), rel=1e-5)  # the same loss value


@pytest.mark.parametrize("norms, nonfinite, expected", [
    ([1.0] * 200, False, False),
    ([1.0] * 200, True, True),                           # a NaN or infinity stopped the run
    ([50.0] * 100 + [1.0] * 100, False, False),          # large norms only before step 100 do not count
    ([1.0] * 100 + [11.0] * 6 + [1.0] * 94, False, True),  # 6% of later steps above 10
    ([1.0] * 100 + [11.0] * 5 + [1.0] * 95, False, False),  # 5% is not more than 5%
    ([1.0] * 100 + [float("nan")] + [1.0] * 99, False, True),
])
def test_instability_rule_matches_the_pre_registration(norms, nonfinite, expected):
    assert is_unstable(norms, nonfinite) is expected


def test_run_log_records_config_seed_commit_and_refuses_changed_code(tmp_path):
    git = git_executable()
    def run(*args):
        subprocess.run([git, *args], cwd=tmp_path, check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.name", "test")
    run("config", "user.email", "test@example.com")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    run("add", "-A")
    run("commit", "-q", "-m", "first")
    folder = start_run(tmp_path, "pilot", "no-deg", 3, {"lr": 0.1}, require_clean_code=True)
    record = json.loads((folder / "run.json").read_text())
    assert record["seed"] == 3 and record["config"] == {"lr": 0.1} and len(record["commit"]) == 40
    assert not record["code_differs_from_commit"]
    assert "pilot" in (tmp_path / "results" / "runs" / "index.csv").read_text()
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError):
        start_run(tmp_path, "full", "no-deg", 3, {}, require_clean_code=True)


def test_batched_closed_loop_matches_the_ordinary_loop_for_each_droop_setting(data, battery, solver):
    dead, full = [0.0, 0.01, 0.005], [0.02, 0.05, 0.03]
    batched = DroopController(battery, torch.tensor(dead).unsqueeze(1), torch.tensor(full).unsqueeze(1))
    results = closed_loop_batched(batched, 3, data, "val", 1.0, battery, solver, True)
    assert len(results) == 3
    for b in range(3):
        single = closed_loop(DroopController(battery, dead[b], full[b]), data, "val", 1.0, battery, solver, True)
        for key in ("v", "action", "soc", "loss"):
            np.testing.assert_allclose(results[b][key], single[key], atol=1e-5, equal_nan=True)  # float32 sums in another order
        assert np.abs(np.nan_to_num(single["action"][..., 1])).sum() > 0  # Q is really used, so the check means something


def test_share_q_sign_changes_counts_only_consecutive_valid_steps_and_ignores_zero(data):
    action = np.zeros((len(data.labels), 2, 2), dtype=np.float32)
    val = np.flatnonzero(data.labels == 1)
    action[val, 0, 1] = np.where(np.arange(len(val)) % 2 == 0, 0.01, -0.01)  # battery 1 flips every step
    action[val, 1, 1] = 0.0  # battery 2 never has a sign
    share = share_q_sign_changes({"action": action}, data, "val")
    # 100 val steps with one invalid step (150): pairs with both steps valid = 99 - 2 = 97, battery 1 flips on all of them
    assert share == pytest.approx(0.5)


def test_mpc_lindistflow_voltages_are_close_to_the_nonlinear_plant(net, data, battery, solver):
    from dpc_grid.mpc import LinDistFlowMPC
    mpc = LinDistFlowMPC(solver, data, battery, WEIGHTS, horizon=3)
    t, scale, soc = torch.tensor([120]), torch.tensor([1.0]), torch.tensor([[0.5, 0.5]])
    action = mpc(None, None, None, t=t, scale=scale, soc=soc)
    assert mpc.log[-1]["status"] == "optimal"
    index = torch.as_tensor(data.bess_buses)
    zeros = torch.zeros_like(data.p_load[t])
    out = solver(data.p_load[t], data.q_load[t], data.p_solar[t], p_bess=zeros.index_copy(1, index, action[..., 0]),
                 q_bess=zeros.index_copy(1, index, action[..., 1]))
    error = np.abs(mpc.vars["v"].value[0] - out["v"][0].numpy()).max()
    assert error < 0.01  # LinDistFlow drops the losses, so it is close but not exact


def test_mpc_respects_limits_and_its_soc_equation_matches_the_plant(data, battery, solver):
    from dpc_grid.mpc import LinDistFlowMPC
    mpc = LinDistFlowMPC(solver, data, battery, WEIGHTS, horizon=4)
    soc0 = torch.tensor([[0.12, 0.88]])
    action = mpc(None, None, None, t=torch.tensor([110]), scale=torch.tensor([0.7]), soc=soc0)[0].numpy()
    assert np.all(np.abs(action) <= battery["p_max_pu"] + 1e-6)
    assert np.all(np.hypot(action[:, 0], action[:, 1]) <= battery["s_max_pu"] + 1e-6)
    planned = mpc.vars["soc"].value[1]
    assert np.all(planned >= battery["soc_min"] - 1e-6) and np.all(planned <= battery["soc_max"] + 1e-6)
    if not any(mpc.log[-1]["simultaneous_charge_discharge"]):
        plant = bess_soc_step(soc0, torch.as_tensor(action[:, 0]) * data.s_base, 1.0, data.dt_h, 0.95, 0.95)
        np.testing.assert_allclose(planned, plant[0].numpy(), atol=1e-5)


def test_closed_loop_runs_the_mpc_on_a_subset_of_steps(data, battery, solver):
    from dpc_grid.mpc import LinDistFlowMPC
    mpc = LinDistFlowMPC(solver, data, battery, WEIGHTS, horizon=3)
    only = np.zeros(len(data.labels), dtype=bool)
    only[110:120] = True
    result = closed_loop(mpc, data, "val", 0.7, battery, solver, True, only=only)
    assert len(mpc.log) == 10
    assert np.isfinite(result["v"][110:120]).all() and np.isnan(result["v"][120:200]).all()


class _SocController:
    """A test controller whose P depends on SoC, so that a resume must restore SoC to give the same result."""
    needs_context = True

    def __call__(self, state, disturbance, v_bess, t=None, scale=None, soc=None):
        p = 0.04 * (soc - 0.3)  # pu, discharges while SoC is above 30%
        q = 0.01 * torch.ones_like(soc)
        return torch.stack([p, q], dim=2)


def test_closed_loop_resumed_from_a_checkpoint_matches_an_uninterrupted_run(data, battery, solver):
    full = closed_loop(_SocController(), data, "val", 1.0, battery, solver, True)
    saved = []

    class Stop(Exception):
        pass

    def checkpoint(state):
        saved.append({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in state.items()})
        if len(saved) == 2:
            raise Stop  # an interruption after the second checkpoint

    with pytest.raises(Stop):
        closed_loop(_SocController(), data, "val", 1.0, battery, solver, True, on_checkpoint=checkpoint, checkpoint_every=30)
    assert saved[-1]["next_position"] == 60
    resumed = closed_loop(_SocController(), data, "val", 1.0, battery, solver, True, resume=saved[-1])
    for key in ("v", "action", "soc", "loss", "soc_clip"):
        np.testing.assert_array_equal(resumed[key], full[key])
    assert np.nanmax(np.abs(full["action"][..., 0])) > 0  # P really depends on SoC here

def test_lossless_mpc_soc_model_has_one_signed_p_and_matches_the_plant_without_efficiency(data, battery, solver):
    from dpc_grid.mpc import LinDistFlowMPC
    mpc = LinDistFlowMPC(solver, data, battery, WEIGHTS, horizon=4, soc_model="lossless")
    assert "p_ch" not in mpc.vars and "p_dis" not in mpc.vars  # no split, so no simultaneous charge and discharge
    soc0 = torch.tensor([[0.6, 0.4]])
    action = mpc(None, None, None, t=torch.tensor([110]), scale=torch.tensor([1.0]), soc=soc0)[0].numpy()
    assert mpc.log[-1]["status"] == "optimal" and mpc.log[-1]["simultaneous_charge_discharge"] == [False, False]
    planned = mpc.vars["soc"].value[1]
    lossless = soc0[0].numpy() - action[:, 0] * data.s_base * data.dt_h / 1.0
    np.testing.assert_allclose(planned, lossless, atol=1e-6)
    # Against the real battery (efficiency 0.95) the one-step gap is within the bound written in docs/decisions.md.
    plant = bess_soc_step(soc0, torch.as_tensor(action[:, 0]) * data.s_base, 1.0, data.dt_h, 0.95, 0.95)[0].numpy()
    bound = battery["p_max_pu"] * data.s_base * (1 / 0.95 - 0.95) * data.dt_h / 1.0
    assert np.all(np.abs(planned - plant) <= bound + 1e-6)


def test_oracle_forecast_uses_the_true_future_and_holds_the_last_known_value(data, battery, solver):
    from dpc_grid.mpc import LinDistFlowMPC
    mpc = LinDistFlowMPC(solver, data, battery, WEIGHTS, horizon=6, forecast="oracle", soc_model="lossless")
    available = np.asarray((data.labels == 1) & data.valid)
    t = int(np.flatnonzero(available)[0])
    steps, held = mpc.forecast_steps(t)
    expected = np.arange(t, t + 6)
    ok = available[expected]
    if ok.all():
        np.testing.assert_array_equal(steps, expected)
        assert held == 0
    last = int(np.flatnonzero(available)[-1])
    steps, held = mpc.forecast_steps(last - 2)  # runs past the end of the validation part
    np.testing.assert_array_equal(steps, [last - 2, last - 1, last, last, last, last])
    assert held == 3
    persistence = LinDistFlowMPC(solver, data, battery, WEIGHTS, horizon=6, soc_model="lossless")
    np.testing.assert_array_equal(persistence.forecast_steps(t)[0], np.full(6, t))