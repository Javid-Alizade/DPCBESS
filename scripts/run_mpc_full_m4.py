"""Milestone M4: full run of the persistence MPC (lossless SoC model, item 7 as changed on 2026-11-06) on ALL validation steps, both regimes one after the other, resumable.

Run:  python scripts/run_mpc_full_m4.py
Started only when the pre-specified rule allows it (docs/decisions.md, "Overlap diagnostic and decision rule").
Progress is saved every 288 steps (one day) to results/m4_mpc_full/<regime>_checkpoint.npz and <regime>_log.json.
If the job stops, running the same command again continues from the last checkpoint. A resume is allowed only at the same,
clean commit, and every resume is logged in the result. A finished regime is never run again.
Reports the same validation check as the DPC candidates (objective and its terms, mean |P|, end-of-day SoC, S-limit share,
SoC-clip share, VVR, losses), plus solver times and statuses, the safety-layer share and the overlap share.
The test part stays closed (M3Data cannot read it).
"""
import json
import platform
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import torch

from dpc_grid import load_case33bw
from dpc_grid.dpc import (M3Data, closed_loop, closed_loop_summary, continuous_objective, end_of_day_soc, per_unit_battery,
                          share_at_s_limit, share_soc_clip_active)
from dpc_grid.mpc import LinDistFlowMPC
from dpc_grid.runlog import git_state

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results" / "m4_mpc_full"
REGIMES = {"base_x0.7": 0.7, "stress_x1.0": 1.0}
HORIZON = 144
DEAD_BAND = 0.2
CHANGED = 1e-6
EVERY = 288
STATE_KEYS = ("v", "action", "soc", "loss", "soc_clip", "steps", "next_position", "state_soc", "state_action",
              "state_v_bess")


def save_checkpoint(regime, state, mpc, history):
    """Write the checkpoint to a temporary file first, so that a stop during writing cannot destroy the last good one."""
    tmp = FOLDER / f"{regime}_checkpoint.tmp.npz"
    np.savez(tmp, **{k: state[k] for k in STATE_KEYS})
    (FOLDER / f"{regime}_log.tmp.json").write_text(json.dumps({"log": mpc.log, "history": history}))
    tmp.replace(FOLDER / f"{regime}_checkpoint.npz")
    (FOLDER / f"{regime}_log.tmp.json").replace(FOLDER / f"{regime}_log.json")


def summarize(result, mpc, data, battery, weights, s_base):
    used = (data.labels == 1) & data.valid
    a = result["action"][used] * s_base
    days = end_of_day_soc(result, data, "val")
    summary = closed_loop_summary(result, data, "val", battery, None, include_rq2=False)
    summary.pop("rq2", None)
    objective = continuous_objective(result, data, "val", battery, weights)
    steps = [r["t"] for r in mpc.log]
    raw = np.array([r["action"] for r in mpc.log])
    changed = np.abs(result["action"][steps] - raw).max(axis=2) > CHANGED
    solve = np.array([r["solve_seconds"] for r in mpc.log])
    statuses = {}
    for r in mpc.log:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return {"objective": objective["objective"], "terms": objective["terms"],
            "mean_abs_p_mw": np.abs(a[..., 0]).mean(axis=0).tolist(), "mean_p_mw": a[..., 0].mean(axis=0).tolist(),
            "mean_abs_q_mvar": np.abs(a[..., 1]).mean(axis=0).tolist(),
            "share_at_s_limit": share_at_s_limit(result, data, "val", battery),
            "share_soc_clip_active": share_soc_clip_active(result, data, "val"),
            "days_at_end_of_day": int(len(days)), "end_of_day_soc_min": float(days.min()),
            "end_of_day_soc_max": float(days.max()), "end_of_day_soc_mean": float(days.mean()),
            "share_of_end_of_day_values_inside_20_80": float(((days >= 0.2) & (days <= 0.8)).mean()),
            "summary": summary, "steps": len(steps),
            "solve_seconds": {"mean": float(solve.mean()), "median": float(np.median(solve)),
                              "p95": float(np.percentile(solve, 95)), "max": float(solve.max())},
            "solver_statuses": statuses,
            "share_battery_steps_safety_layer_changes_action": float(changed.mean()),
            "share_battery_steps_simultaneous_charge_discharge": float(np.mean([r["simultaneous_charge_discharge"]
                                                                                for r in mpc.log]))}


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    state_now = git_state(ROOT)
    if state_now["code_differs_from_commit"]:
        raise RuntimeError("Code differs from the last commit. Commit first, so this run can be repeated and resumed exactly.")
    m2 = json.loads((ROOT / "configs" / "m2.json").read_text())
    m3 = json.loads((ROOT / "configs" / "m3.json").read_text())
    net = load_case33bw()
    data = M3Data(ROOT / "data" / "processed" / "m2_dataset.npz", net.s_base_mva, m2["time"]["dt_min"]).prepare(
        m2["placement"]["pv_buses"], m2["placement"]["bess_buses"])
    battery = per_unit_battery(dict(m3["battery"], terminal_soc_dead_band=DEAD_BAND), net.s_base_mva)
    solver = net.make_solver(n_iter=m3["evaluation"]["n_iter"])
    weights = m3["weights"]

    for regime, value in REGIMES.items():
        final = FOLDER / f"{regime}.json"
        if final.exists():
            print(f"{regime}: already finished, skipped", flush=True)
            continue
        mpc = LinDistFlowMPC(solver, data, battery, weights, HORIZON, soc_model="lossless")
        resume, history = None, []
        checkpoint = FOLDER / f"{regime}_checkpoint.npz"
        if checkpoint.exists():
            saved = json.loads((FOLDER / f"{regime}_log.json").read_text())
            history = saved["history"]
            started = history[0]
            if started["commit"] != state_now["commit"] or state_now["code_differs_from_commit"]:
                raise RuntimeError(f"{regime}: the checkpoint was made at commit {started['commit'][:7]}; a resume is "
                                   "allowed only at that same commit with no code changes.")
            with np.load(checkpoint) as f:
                resume = {k: f[k] for k in STATE_KEYS}
            mpc.log = saved["log"]
            print(f"{regime}: resuming at step {int(resume['next_position'])} of {len(resume['steps'])}", flush=True)
        history.append({"event": "resume" if resume is not None else "start", **state_now,
                        "at_position": int(resume["next_position"]) if resume is not None else 0,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        clock = time.perf_counter()
        last = {"t": clock}

        def on_checkpoint(state):
            save_checkpoint(regime, state, mpc, history)
            now = time.perf_counter()
            done, total = state["next_position"], len(state["steps"])
            print(f"{regime}: {done}/{total} steps, {now - last['t']:.0f} s for the last {EVERY}", flush=True)
            last["t"] = now

        result = closed_loop(mpc, data, "val", value, battery, solver, True, resume=resume, on_checkpoint=on_checkpoint,
                             checkpoint_every=EVERY)
        row = summarize(result, mpc, data, battery, weights, net.s_base_mva)
        row.update({"regime": regime, "part": "validation only, all steps", "horizon": HORIZON, "forecast": "persistence", "soc_model": "lossless",
                    "terminal_soc_dead_band": DEAD_BAND, "weights": weights, "history": history,
                    "wall_seconds_this_session": time.perf_counter() - clock,
                    "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                                 "cvxpy": cp.__version__}})
        np.savez_compressed(FOLDER / f"{regime}_trajectory.npz", **{k: result[k] for k in result})
        (FOLDER / f"{regime}_solver_log.json").write_text(json.dumps(mpc.log))
        final.write_text(json.dumps(row, indent=2))
        print(f"{regime}: finished, objective {row['objective']:.5f}", flush=True)


if __name__ == "__main__":
    main()
