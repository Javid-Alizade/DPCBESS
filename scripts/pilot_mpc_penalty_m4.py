"""Milestone M4: the ONE re-run of the 2-day pilot with the pre-specified overlap penalty (1e-5 x sum(P_ch + P_dis) / H).

Run:  python scripts/pilot_mpc_penalty_m4.py
Triggered by the rule in docs/decisions.md ("Overlap diagnostic and decision rule"): the mean overlap exceeded 2% of the limit.
Same pilot days, regimes, horizon, terminal term and safety layer as scripts/pilot_mpc_m4.py. Reports the pilot numbers
(times, statuses, safety layer, SoC clip) and the diagnostic numbers (overlap, SoC gaps), so both runs can be compared.
The value of the penalty is fixed in the protocol and is not tuned. The test part stays closed.
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
from dpc_grid.runlog import git_state
from diagnose_mpc_overlap_m4 import AHEAD, LIMIT_MW, RecordingMPC, stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "m4_mpc_pilot_penalty.json"
REGIMES = {"base_x0.7": 0.7, "stress_x1.0": 1.0}
HORIZON = 144
DEAD_BAND = 0.2
PENALTY = 1e-5  # fixed in docs/decisions.md before the diagnostic was run
CHANGED = 1e-6


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
    output = {"part": "validation only, 2-day pilot, re-run with the overlap penalty", **git_state(ROOT),
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                           "cvxpy": cp.__version__},
              "overlap_penalty": PENALTY, "horizon": HORIZON, "terminal_soc_dead_band": DEAD_BAND,
              "pilot_days": [str(np.datetime_as_string(d, unit="D")) for d in pilot_days], "regimes": {}}
    for regime, value in REGIMES.items():
        print(f"pilot with penalty, {regime}", flush=True)
        mpc = RecordingMPC(solver, data, battery, m3["weights"], HORIZON, overlap_penalty=PENALTY)
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
        log = [r for r in mpc.log if "p_ch" in r]
        overlap = np.minimum(np.array([r["p_ch"] for r in log]), np.array([r["p_dis"] for r in log])) * net.s_base_mva
        gap1, gap12 = [], []
        for r in log:
            t = r["t"]
            plan = np.array(r["soc_plan"])
            gap1.append(np.abs(plan[1] - result["soc"][t]))
            last = t + AHEAD - 1
            if last < len(stamps) and only[t:last + 1].all() and (np.diff(stamps[t:last + 1]) == step).all():
                gap12.append(np.abs(plan[AHEAD] - result["soc"][last]))
        per_step = wall / len(steps)
        output["regimes"][regime] = {
            "steps": len(steps), "wall_seconds": wall, "wall_seconds_per_step": per_step,
            "solve_seconds": {"mean": float(solve.mean()), "median": float(np.median(solve)),
                              "p95": float(np.percentile(solve, 95)), "max": float(solve.max())},
            "estimated_hours_for_all_validation_steps": per_step * n_val_steps / 3600,
            "solver_statuses": statuses,
            "share_battery_steps_safety_layer_changes_action": float((np.abs(safe - raw).max(axis=2) > CHANGED).mean()),
            "share_battery_steps_soc_clip_active": float(result["soc_clip"][steps].mean()),
            "share_battery_steps_simultaneous_charge_discharge": float(np.mean([r["simultaneous_charge_discharge"]
                                                                                for r in mpc.log])),
            "overlap_mw": stats(overlap), "overlap_mean_share_of_limit": float(overlap.mean() / LIMIT_MW),
            "soc_gap_1_step": stats(gap1), "soc_gap_12_steps": stats(gap12),
            "mean_abs_p_mw": (np.abs(safe[..., 0]) * net.s_base_mva).mean(axis=0).tolist(),
            "mean_abs_q_mvar": (np.abs(safe[..., 1]) * net.s_base_mva).mean(axis=0).tolist(),
            "soc_min": float(np.nanmin(result["soc"][steps])), "soc_max": float(np.nanmax(result["soc"][steps])),
        }
        print(json.dumps(output["regimes"][regime], indent=1), flush=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
