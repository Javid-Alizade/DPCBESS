"""Milestone M4: the 2-day pilot of the persistence MPC with the LOSSLESS SoC model (changed item 7), run once, with the
pre-specified pass criteria (docs/decisions.md, "CHANGE OF M4 PLAN ITEM 7").

Run:  python scripts/pilot_mpc_lossless_m4.py
Exit code 0 if all criteria pass, 1 if any fails (the night run starts only after exit code 0).
  (1) status "optimal" on at least 99.9% of steps in each regime
  (2) no simultaneous charge and discharge by construction (one signed P, no P_ch/P_dis in the model)
  (3) one-step SoC mismatch vs the real battery reported; above 0.0043 + 1e-4 is a FAIL (a model or code error)
  (4) safety-layer share, solve times, mean |P| and |Q| reported
The test part stays closed (M3Data cannot read it).
"""
import json
import platform
import sys
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import torch

from dpc_grid import load_case33bw
from dpc_grid.dpc import M3Data, closed_loop, per_unit_battery
from dpc_grid.mpc import LinDistFlowMPC
from dpc_grid.runlog import git_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "m4_mpc_pilot_lossless.json"
REGIMES = {"base_x0.7": 0.7, "stress_x1.0": 1.0}
HORIZON = 144
DEAD_BAND = 0.2
CHANGED = 1e-6
AHEAD = 12
MIN_OPTIMAL_SHARE = 0.999
MISMATCH_BOUND = 0.5 * (1 / 0.95 - 0.95) * (5 / 60) / 1.0  # 0.0043 SoC per step, from docs/decisions.md
MISMATCH_SLACK = 1e-4


class PlanRecordingMPC(LinDistFlowMPC):
    """The same MPC; after each solve it also stores the planned SoC for the next 12 steps."""

    def __call__(self, *args, **kwargs):
        action = super().__call__(*args, **kwargs)
        if self.log[-1]["status"] in ("optimal", "optimal_inaccurate"):
            self.log[-1]["soc_plan"] = self.vars["soc"].value[: AHEAD + 1].tolist()
        return action


def stats(values):
    values = np.asarray(values, dtype=float).ravel()
    return {"n": int(values.size), "mean": float(values.mean()), "p95": float(np.percentile(values, 95)),
            "max": float(values.max())}


def main():
    if OUT.exists():
        raise FileExistsError(f"{OUT} exists. Earlier results are never overwritten.")
    m2 = json.loads((ROOT / "configs" / "m2.json").read_text())
    m3 = json.loads((ROOT / "configs" / "m3.json").read_text())
    net = load_case33bw()
    data = M3Data(ROOT / "data" / "processed" / "m2_dataset.npz", net.s_base_mva, m2["time"]["dt_min"]).prepare(
        m2["placement"]["pv_buses"], m2["placement"]["bess_buses"])
    battery = per_unit_battery(dict(m3["battery"], terminal_soc_dead_band=DEAD_BAND), net.s_base_mva)
    solver = net.make_solver(n_iter=m3["evaluation"]["n_iter"])

    val = (data.labels == 1) & data.valid
    start = data.stamps - np.timedelta64(data.dt_min, "m")
    day = np.asarray(start.normalize())
    full_days = [d for d in np.unique(day[val]) if (val & (day == d)).sum() == 288]
    pilot_days = full_days[:2]
    only = np.isin(day, pilot_days) & val
    stamps = data.stamps.values
    step = np.timedelta64(data.dt_min, "m")
    n_val_steps = int(val.sum())
    output = {"part": "validation only, 2-day pilot, lossless SoC model in the MPC", **git_state(ROOT),
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                           "cvxpy": cp.__version__},
              "soc_model": "lossless", "forecast": "persistence", "horizon": HORIZON, "terminal_soc_dead_band": DEAD_BAND,
              "pilot_days": [str(np.datetime_as_string(d, unit="D")) for d in pilot_days],
              "criteria": {"min_optimal_share": MIN_OPTIMAL_SHARE, "mismatch_bound": MISMATCH_BOUND,
                           "mismatch_slack": MISMATCH_SLACK}, "regimes": {}}
    failures = []
    for regime, value in REGIMES.items():
        print(f"pilot lossless, {regime}", flush=True)
        mpc = PlanRecordingMPC(solver, data, battery, m3["weights"], HORIZON, soc_model="lossless")
        by_construction = "p_ch" not in mpc.vars and "p_dis" not in mpc.vars
        wall = time.perf_counter()
        result = closed_loop(mpc, data, "val", value, battery, solver, True, only=only)
        wall = time.perf_counter() - wall
        steps = [r["t"] for r in mpc.log]
        raw = np.array([r["action"] for r in mpc.log])
        safe = result["action"][steps]
        solve = np.array([r["solve_seconds"] for r in mpc.log])
        statuses = {}
        for r in mpc.log:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        optimal_share = statuses.get("optimal", 0) / len(mpc.log)
        gap1, gap12 = [], []
        for r in mpc.log:
            if "soc_plan" not in r:
                continue
            t, plan = r["t"], np.array(r["soc_plan"])
            gap1.append(np.abs(plan[1] - result["soc"][t]))
            last = t + AHEAD - 1
            if last < len(stamps) and only[t:last + 1].all() and (np.diff(stamps[t:last + 1]) == step).all():
                gap12.append(np.abs(plan[AHEAD] - result["soc"][last]))
        per_step = wall / len(steps)
        row = {"steps": len(steps), "wall_seconds": wall, "wall_seconds_per_step": per_step,
               "solve_seconds": {"mean": float(solve.mean()), "median": float(np.median(solve)),
                                 "p95": float(np.percentile(solve, 95)), "max": float(solve.max())},
               "estimated_hours_for_all_validation_steps": per_step * n_val_steps / 3600,
               "solver_statuses": statuses, "optimal_share": optimal_share,
               "no_simultaneous_by_construction": by_construction,
               "share_battery_steps_safety_layer_changes_action": float((np.abs(safe - raw).max(axis=2) > CHANGED).mean()),
               "max_safety_change_pu": float(np.abs(safe - raw).max()),
               "share_battery_steps_soc_clip_active": float(result["soc_clip"][steps].mean()),
               "soc_mismatch_1_step": stats(gap1), "soc_gap_12_steps": stats(gap12),
               "mean_abs_p_mw": (np.abs(safe[..., 0]) * net.s_base_mva).mean(axis=0).tolist(),
               "mean_p_mw": (safe[..., 0] * net.s_base_mva).mean(axis=0).tolist(),
               "mean_abs_q_mvar": (np.abs(safe[..., 1]) * net.s_base_mva).mean(axis=0).tolist(),
               "soc_min": float(np.nanmin(result["soc"][steps])), "soc_max": float(np.nanmax(result["soc"][steps]))}
        row["pass"] = {"optimal_share": optimal_share >= MIN_OPTIMAL_SHARE,
                       "no_simultaneous_by_construction": by_construction,
                       "mismatch_within_bound": row["soc_mismatch_1_step"]["max"] <= MISMATCH_BOUND + MISMATCH_SLACK}
        failures += [f"{regime}: {k}" for k, ok in row["pass"].items() if not ok]
        output["regimes"][regime] = row
        print(json.dumps(row, indent=1), flush=True)
    output["all_criteria_pass"] = not failures
    output["failures"] = failures
    OUT.write_text(json.dumps(output, indent=2))
    print(f"Saved {OUT}", flush=True)
    print("ALL CRITERIA PASS" if not failures else f"FAIL: {failures}", flush=True)
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
