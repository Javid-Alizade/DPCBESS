"""Milestone M4: the oracle MPC (exact future, an UPPER BOUND, not a real controller) on a pre-specified subset of
validation days, next to every other controller under identical conditions. Plan: docs/decisions.md, "Oracle plan".

Run:  python scripts/run_oracle_subset_m4.py
Days: every 4th full validation day starting from the first. Each day runs on its own (the days are not adjacent, so the loop
sees a gap before each one and SoC resets to 0.5 at its start). Controllers, both regimes: oracle MPC and persistence MPC
(both with the lossless SoC model), DPC H144_dead_zone_30_70 seeds 0/1/2, original droop, final tuned droop, no battery.
Each finished (controller, regime) is saved to results/m4_oracle_subset/ and skipped if the job is started again; a restart
is allowed only at the same clean commit. RQ2 metrics are not computed. The test part stays closed (M3Data cannot read it).
"""
import copy
import json
import platform
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import torch

from dpc_grid import load_case33bw
from dpc_grid.dpc import (DroopController, IdleController, M3Data, PolicyController, closed_loop, closed_loop_summary,
                          continuous_objective, end_of_day_soc, make_policy, per_unit_battery, share_at_s_limit,
                          share_soc_clip_active)
from dpc_grid.mpc import LinDistFlowMPC
from dpc_grid.runlog import git_state

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results" / "m4_oracle_subset"
REGIMES = {"base_x0.7": 0.7, "stress_x1.0": 1.0}
HORIZON = 144
DEAD_BAND = 0.2
CHANGED = 1e-6
EVERY_NTH_DAY = 4
DPC_RUNS = {"seed_0": "results/runs/20261106-124919_candidate_no-deg_seed0",
            "seed_1": "results/runs/20261106-141248_candidate_no-deg_seed1",
            "seed_2": "results/runs/20261106-144916_candidate_no-deg_seed2"}


def evaluate(controller, sub, battery, solver, weights, value, s_base):
    """One controller in one regime on the subset days. `sub` is the data with only the subset days marked valid."""
    wall = time.perf_counter()
    result = closed_loop(controller, sub, "val", value, battery, solver, True)
    wall = time.perf_counter() - wall
    used = (sub.labels == 1) & sub.valid
    a = result["action"][used] * s_base
    days = end_of_day_soc(result, sub, "val")
    summary = closed_loop_summary(result, sub, "val", battery, None, include_rq2=False)
    summary.pop("rq2", None)
    objective = continuous_objective(result, sub, "val", battery, weights)
    row = {"objective": objective["objective"], "terms": objective["terms"], "wall_seconds": wall,
           "mean_abs_p_mw": np.abs(a[..., 0]).mean(axis=0).tolist(), "mean_p_mw": a[..., 0].mean(axis=0).tolist(),
           "mean_abs_q_mvar": np.abs(a[..., 1]).mean(axis=0).tolist(),
           "share_at_s_limit": share_at_s_limit(result, sub, "val", battery),
           "share_soc_clip_active": share_soc_clip_active(result, sub, "val"),
           "days_at_end_of_day": int(len(days)), "end_of_day_soc_min": float(days.min()) if len(days) else None,
           "end_of_day_soc_max": float(days.max()) if len(days) else None, "summary": summary}
    if isinstance(controller, LinDistFlowMPC):
        steps = [r["t"] for r in controller.log]
        raw = np.array([r["action"] for r in controller.log])
        solve = np.array([r["solve_seconds"] for r in controller.log])
        statuses = {}
        for r in controller.log:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        row.update({"solver_statuses": statuses,
                    "solve_seconds": {"mean": float(solve.mean()), "median": float(np.median(solve)),
                                      "p95": float(np.percentile(solve, 95)), "max": float(solve.max())},
                    "share_battery_steps_safety_layer_changes_action":
                        float((np.abs(result["action"][steps] - raw).max(axis=2) > CHANGED).mean()),
                    "share_held_forecast_horizon_steps":
                        float(sum(r["held_forecast_steps"] for r in controller.log) / (len(steps) * HORIZON))})
    return row


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    state = git_state(ROOT)
    if state["code_differs_from_commit"]:
        raise RuntimeError("Code differs from the last commit. Commit first, so this run can be repeated exactly.")
    m2 = json.loads((ROOT / "configs" / "m2.json").read_text())
    m3 = json.loads((ROOT / "configs" / "m3.json").read_text())
    net = load_case33bw()
    data = M3Data(ROOT / "data" / "processed" / "m2_dataset.npz", net.s_base_mva, m2["time"]["dt_min"]).prepare(
        m2["placement"]["pv_buses"], m2["placement"]["bess_buses"])
    battery = per_unit_battery(dict(m3["battery"], terminal_soc_dead_band=DEAD_BAND), net.s_base_mva)
    solver = net.make_solver(n_iter=m3["evaluation"]["n_iter"])
    weights = m3["weights"]

    val = (data.labels == 1) & data.valid
    start = data.stamps - np.timedelta64(data.dt_min, "m")
    day = np.asarray(start.normalize())
    full_days = [d for d in np.unique(day[val]) if (val & (day == d)).sum() == 288]
    subset_days = full_days[::EVERY_NTH_DAY]
    sub = copy.copy(data)  # the closed loop sees only the subset days; the oracle still reads the true data around them
    sub.valid = data.valid & np.isin(day, subset_days)
    plan = {"part": "validation only, oracle subset", **state, "n_full_validation_days": len(full_days),
            "every_nth_day": EVERY_NTH_DAY, "days": [str(np.datetime_as_string(d, unit="D")) for d in subset_days],
            "steps_per_regime": int(((sub.labels == 1) & sub.valid).sum()), "horizon": HORIZON,
            "soc_model_of_both_mpcs": "lossless", "terminal_soc_dead_band": DEAD_BAND}
    plan_file = FOLDER / "plan.json"
    if plan_file.exists():
        saved = json.loads(plan_file.read_text())
        if saved["commit"] != state["commit"] or saved["days"] != plan["days"]:
            raise RuntimeError(f"Earlier parts were run at commit {saved['commit'][:7]}; continue only at that commit.")
    else:
        plan_file.write_text(json.dumps(plan, indent=2))
    print(f"{len(subset_days)} days of {len(full_days)}, {plan['steps_per_regime']} steps per regime", flush=True)

    round2 = json.loads((ROOT / "results" / "m3_droop_tuning_round2.json").read_text())["final_tuned"]

    def controllers():  # built lazily, so that each MPC starts with an empty log
        yield "oracle_mpc_upper_bound", lambda: LinDistFlowMPC(solver, data, battery, weights, HORIZON, forecast="oracle",
                                                               soc_model="lossless")
        yield "persistence_mpc", lambda: LinDistFlowMPC(solver, data, battery, weights, HORIZON, soc_model="lossless")
        for seed, folder in DPC_RUNS.items():
            def load(folder=folder):
                policy = make_policy(m3)
                policy.load_state_dict(torch.load(ROOT / folder / "model.pt"))
                policy.eval()
                return PolicyController(policy, battery)
            yield f"dpc_H144_dead_zone_30_70_{seed}", load
        yield "droop_original", lambda: DroopController(battery, m3["droop"]["dead_band_pu"], m3["droop"]["full_at_pu"])
        yield "droop_final_tuned", lambda: DroopController(battery, round2["dead_band_pu"], round2["full_at_pu"])
        yield "no_battery", lambda: IdleController()

    for name, build in controllers():
        for regime, value in REGIMES.items():
            out = FOLDER / f"{name}__{regime}.json"
            if out.exists():
                print(f"{name}, {regime}: done earlier, skipped", flush=True)
                continue
            print(f"{name}, {regime}", flush=True)
            row = evaluate(build(), sub, battery, solver, weights, value, net.s_base_mva)
            row.update({"controller": name, "regime": regime, "commit": state["commit"],
                        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                     "torch": torch.__version__, "cvxpy": cp.__version__}})
            out.write_text(json.dumps(row, indent=2))
            print(f"  objective {row['objective']:.5f}", flush=True)
    print("oracle subset finished", flush=True)


if __name__ == "__main__":
    main()
