"""Milestone M4: diagnostic of simultaneous charge and discharge in the persistence MPC, on the two pilot days, both regimes.

Run:  python scripts/diagnose_mpc_overlap_m4.py
Protocol and decision rule: docs/decisions.md, "Overlap diagnostic and decision rule" (written before this script was run).
The MPC is NOT changed: a subclass only reads the solution after each solve (P_ch, P_dis of the first step and the planned SoC).
  overlap         min(P_ch, P_dis) of the applied step, per battery step, in MW
  SoC gap, 12     |SoC the MPC plans at step t for after 12 steps - real SoC after those 12 steps| (the rule uses its mean)
  SoC gap, 1      the same after one step (the efficiency mismatch only)
The test part stays closed (M3Data cannot read it).
"""
import json
import platform
from pathlib import Path

import cvxpy as cp
import numpy as np
import torch

from dpc_grid import load_case33bw
from dpc_grid.dpc import M3Data, closed_loop, per_unit_battery
from dpc_grid.mpc import LinDistFlowMPC
from dpc_grid.runlog import git_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "m4_mpc_overlap_diagnostic_rerun.json"  # the published run, made at a clean commit
REGIMES = {"base_x0.7": 0.7, "stress_x1.0": 1.0}
HORIZON = 144
DEAD_BAND = 0.2
AHEAD = 12  # 1 hour
LIMIT_MW = 0.5
OVERLAP_SHARE_LIMIT = 0.02  # 2% of the 0.5 MW limit
GAP_LIMIT = 0.02  # SoC


class RecordingMPC(LinDistFlowMPC):
    """The same MPC; after each solve it also stores P_ch, P_dis of the first step and the planned SoC trajectory."""

    def __call__(self, *args, **kwargs):
        action = super().__call__(*args, **kwargs)
        record = self.log[-1]
        if record["status"] in ("optimal", "optimal_inaccurate"):
            record["p_ch"] = self.vars["p_ch"].value[0].tolist()
            record["p_dis"] = self.vars["p_dis"].value[0].tolist()
            record["soc_plan"] = self.vars["soc"].value[: AHEAD + 1].tolist()  # soc_plan[k] = planned SoC after k steps
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

    # The same pilot days as scripts/pilot_mpc_m4.py: the first two full validation days.
    val = (data.labels == 1) & data.valid
    start = data.stamps - np.timedelta64(data.dt_min, "m")
    day = np.asarray(start.normalize())
    full_days = [d for d in np.unique(day[val]) if (val & (day == d)).sum() == 288]
    pilot_days = full_days[:2]
    only = np.isin(day, pilot_days) & val
    stamps = data.stamps.values
    step = np.timedelta64(data.dt_min, "m")

    output = {"part": "validation only, the two pilot days", **git_state(ROOT),
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                           "cvxpy": cp.__version__},
              "protocol": "docs/decisions.md, Overlap diagnostic and decision rule",
              "pilot_days": [str(np.datetime_as_string(d, unit="D")) for d in pilot_days], "ahead_steps": AHEAD,
              "regimes": {}}
    triggered = []
    for regime, value in REGIMES.items():
        print(f"diagnostic, {regime}", flush=True)
        mpc = RecordingMPC(solver, data, battery, m3["weights"], HORIZON)
        result = closed_loop(mpc, data, "val", value, battery, solver, True, only=only)
        log = [r for r in mpc.log if "p_ch" in r]
        p_ch = np.array([r["p_ch"] for r in log]) * net.s_base_mva
        p_dis = np.array([r["p_dis"] for r in log]) * net.s_base_mva
        overlap = np.minimum(p_ch, p_dis)  # (steps, batteries), MW

        gap1, gap12 = [], []
        for r in log:
            t = r["t"]
            plan = np.array(r["soc_plan"])  # (AHEAD + 1, batteries)
            gap1.append(np.abs(plan[1] - result["soc"][t]))
            last = t + AHEAD - 1  # the 12 applied steps are t .. t+11
            if last < len(stamps) and only[t:last + 1].all() and (np.diff(stamps[t:last + 1]) == step).all():
                gap12.append(np.abs(plan[AHEAD] - result["soc"][last]))
        row = {"steps_with_solution": len(log), "steps": len(mpc.log),
               "overlap_mw": stats(overlap), "overlap_mean_share_of_limit": float(overlap.mean() / LIMIT_MW),
               "overlap_mean_mw_per_battery": overlap.mean(axis=0).tolist(),
               "share_battery_steps_overlap_above_1e-6_pu": float((overlap / net.s_base_mva > 1e-6).mean()),
               "soc_gap_1_step": stats(gap1), "soc_gap_12_steps": stats(gap12)}
        row["rule_overlap_exceeds"] = row["overlap_mean_share_of_limit"] > OVERLAP_SHARE_LIMIT
        row["rule_gap_exceeds"] = row["soc_gap_12_steps"]["mean"] > GAP_LIMIT
        if row["rule_overlap_exceeds"] or row["rule_gap_exceeds"]:
            triggered.append(regime)
        output["regimes"][regime] = row
        print(json.dumps(row, indent=1), flush=True)
    output["rule"] = {"overlap_share_limit": OVERLAP_SHARE_LIMIT, "gap_limit": GAP_LIMIT, "triggered_in": triggered,
                      "outcome": "add the 1e-5 penalty and re-run the pilot once" if triggered
                      else "run the full persistence MPC as it is"}
    OUT.write_text(json.dumps(output, indent=2))
    print(f"rule outcome: {output['rule']['outcome']}", flush=True)
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
