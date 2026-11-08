"""Milestone M4: 2-day pilot of the persistence MPC on VALIDATION, both regimes, with timing. No full run from here.

Run:  python scripts/pilot_mpc_m4.py
Pilot days: the first two full validation days (all 288 steps valid), fixed in docs/decisions.md before the run.
Reports: per-step solve time, wall time per step of the whole loop, estimated total time per regime for all 48,217
validation steps, solver statuses, the share of battery steps where the safety layer changes the MPC action, the SoC-clip
share and the share of simultaneous charge and discharge. The test part stays closed (M3Data cannot read it).
"""
import json
import platform
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
OUT = ROOT / "results" / "m4_mpc_pilot.json"
REGIMES = {"base_x0.7": 0.7, "stress_x1.0": 1.0}
HORIZON = 144
DEAD_BAND = 0.2  # the 30-70% dead zone of the selected DPC candidate, weight 10 (configs/m3.json)
CHANGED = 1e-6  # pu: a safety-layer change smaller than this is numerical noise


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
    weights = m3["weights"]

    val = (data.labels == 1) & data.valid
    start = data.stamps - np.timedelta64(data.dt_min, "m")
    day = np.asarray(start.normalize())
    full_days = [d for d in np.unique(day[val]) if (val & (day == d)).sum() == 288]
    pilot_days = full_days[:2]
    only = np.isin(day, pilot_days) & val
    n_val_steps = int(val.sum())
    output = {"part": "validation only, 2-day pilot", **git_state(ROOT),
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                           "cvxpy": cp.__version__},
              "horizon": HORIZON, "forecast": "persistence", "terminal_soc_dead_band": DEAD_BAND, "weights": weights,
              "pilot_days": [str(np.datetime_as_string(d, unit="D")) for d in pilot_days], "pilot_steps": int(only.sum()),
              "validation_steps_per_regime": n_val_steps, "regimes": {}}
    for regime, value in REGIMES.items():
        print(f"pilot, {regime}", flush=True)
        build = time.perf_counter()
        mpc = LinDistFlowMPC(solver, data, battery, weights, HORIZON)
        build = time.perf_counter() - build
        wall = time.perf_counter()
        result = closed_loop(mpc, data, "val", value, battery, solver, True, only=only)
        wall = time.perf_counter() - wall
        steps = [r["t"] for r in mpc.log]
        raw = np.array([r["action"] for r in mpc.log])  # (steps, batteries, 2), before the safety layer
        safe = result["action"][steps]
        changed = np.abs(safe - raw).max(axis=2) > CHANGED  # (steps, batteries)
        solve = np.array([r["solve_seconds"] for r in mpc.log])
        statuses = {}
        for r in mpc.log:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        per_step = wall / len(steps)
        output["regimes"][regime] = {
            "build_seconds": build, "steps": len(steps), "wall_seconds": wall, "wall_seconds_per_step": per_step,
            "solve_seconds": {"mean": float(solve.mean()), "median": float(np.median(solve)),
                              "p95": float(np.percentile(solve, 95)), "max": float(solve.max())},
            "estimated_hours_for_all_validation_steps": per_step * n_val_steps / 3600,
            "solver_statuses": statuses,
            "share_battery_steps_safety_layer_changes_action": float(changed.mean()),
            "share_battery_steps_soc_clip_active": float(result["soc_clip"][steps].mean()),
            "share_battery_steps_simultaneous_charge_discharge": float(np.mean([r["simultaneous_charge_discharge"]
                                                                                for r in mpc.log])),
            "max_safety_change_pu": float(np.abs(safe - raw).max()),
            "mean_abs_p_mw": (np.abs(safe[..., 0]) * net.s_base_mva).mean(axis=0).tolist(),
            "mean_abs_q_mvar": (np.abs(safe[..., 1]) * net.s_base_mva).mean(axis=0).tolist(),
            "soc_min": float(np.nanmin(result["soc"][steps])), "soc_max": float(np.nanmax(result["soc"][steps])),
        }
        print(json.dumps(output["regimes"][regime], indent=1), flush=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
