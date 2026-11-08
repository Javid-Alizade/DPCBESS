"""Milestone M3: compare the SoC-penalty candidates in the continuous loop and apply the pre-specified rule in docs/decisions.md.

Run:  python scripts/check_soc_variants_m3.py --out results/<name>.json weight_10=results/runs/<id> weight_0=results/runs/<id> ...
VALIDATION part only, both regimes, with both references. RQ2 metrics are not computed, so they cannot affect the choice.
"""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

from dpc_grid import load_case33bw
from dpc_grid.dpc import (DroopController, IdleController, M3Data, PolicyController, closed_loop, closed_loop_summary,
                          continuous_objective, end_of_day_soc, make_policy, per_unit_battery, select_soc_variant,
                          share_at_s_limit, share_soc_clip_active)
from dpc_grid.runlog import git_state

ROOT = Path(__file__).resolve().parents[1]
REGIMES = {"base_x0.7": 0.7, "stress_x1.0": 1.0}


def evaluate(controller, project, data, battery, solver, weights):
    """One controller, both regimes: objective terms, active-power use, end-of-day SoC, S-limit share and VVR."""
    regimes, objectives, soc_values = {}, [], {}
    for regime, value in REGIMES.items():
        result = closed_loop(controller, data, "val", value, battery, solver, project)
        objective = continuous_objective(result, data, "val", battery, weights)
        objectives.append(objective["objective"])
        used = (data.labels == 1) & data.valid
        a = result["action"][used] * data.s_base
        days = end_of_day_soc(result, data, "val")
        soc_values[regime] = days
        nonfinite_v = float((~np.isfinite(result["v"][used])).any(axis=1).mean())
        summary = closed_loop_summary(result, data, "val", battery, None, include_rq2=False)
        summary.pop("rq2")
        regimes[regime] = {
            "objective": objective["objective"], "terms": objective["terms"],
            "mean_abs_p_mw": np.abs(a[..., 0]).mean(axis=0).tolist(),
            "mean_p_mw": a[..., 0].mean(axis=0).tolist(),
            "mean_abs_q_mvar": np.abs(a[..., 1]).mean(axis=0).tolist(),
            "share_at_s_limit": share_at_s_limit(result, data, "val", battery),
            "share_soc_clip_active": share_soc_clip_active(result, data, "val"),
            "days_at_end_of_day": int(len(days)),
            "end_of_day_soc_min": float(days.min()), "end_of_day_soc_max": float(days.max()),
            "end_of_day_soc_mean": float(days.mean()),
            "share_of_end_of_day_values_inside_20_80": float(((days >= 0.2) & (days <= 0.8)).mean()),
            "projection": project, "share_of_steps_with_nonfinite_voltage": nonfinite_v,
            "summary": summary}
        print(f"  {regime}: objective {objective['objective']:.5f}", flush=True)
    return {"objective": float(np.mean(objectives)), "regimes": regimes}, soc_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-selection", action="store_true",
                        help="report conditions (a) and (b) for each run but select nothing (seed report, docs/decisions.md)")
    parser.add_argument("--no-references", action="store_true",
                        help="skip the no-battery and droop references (they exist already on the same steps)")
    parser.add_argument("runs", nargs="+", help="name=run folder")
    args = parser.parse_args()
    runs = dict(arg.split("=", 1) for arg in args.runs)
    m2 =json.loads((ROOT / "configs" / "m2.json").read_text())
    net = load_case33bw()
    data = M3Data(ROOT / "data" / "processed" / "m2_dataset.npz", net.s_base_mva, m2["time"]["dt_min"]).prepare(
        m2["placement"]["pv_buses"], m2["placement"]["bess_buses"])

    # Every candidate is judged with the same weights and the same battery settings. The terminal term never enters here.
    m3 = json.loads((ROOT / "configs" / "m3.json").read_text())
    weights = m3["weights"]
    battery = per_unit_battery(m3["battery"], net.s_base_mva)
    solver = net.make_solver(n_iter=m3["evaluation"]["n_iter"])

    output = {"part": "validation only", "rule": "docs/decisions.md, SoC-penalty selection rule", **git_state(ROOT),
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
              "candidates": {}, "references": {}}
    candidates = {}
    for name, folder in runs.items():
        record = json.loads((Path(folder) / "run.json").read_text())
        print(f"{name} ({record['run_id']}, commit {record['commit'][:7]})", flush=True)
        policy = make_policy(m3)
        policy.load_state_dict(torch.load(Path(folder) / "model.pt"))
        policy.eval()
        # The run's own variant decides whether the safety layer is used; without it nothing is clipped (M3 item 20).
        project = record["config"]["m3"]["variants"][record["variant"]]["projection"]
        out, soc_values = evaluate(PolicyController(policy, battery), project, data, battery, solver, weights)
        out["variant"] = record["variant"]
        out["run_id"] = record["run_id"]
        out["training_steps"] = record["config"]["steps"]
        out["horizon_steps"] = record["config"]["m3"]["horizon_steps"]
        out["truncate"] = record["config"]["m3"]["train"].get("truncate")
        out["terminal_soc_weight"] = record["config"]["m3"]["weights"]["terminal_soc"]
        out["terminal_soc_dead_band"] = record["config"]["m3"]["battery"]["terminal_soc_dead_band"]
        output["candidates"][name] = out
        candidates[name] = {"objective": out["objective"], "regimes": {
            r: {"mean_abs_p_mw": out["regimes"][r]["mean_abs_p_mw"], "end_of_day_soc": soc_values[r]} for r in REGIMES}}
    references = () if args.no_references else (("no_battery_action", IdleController()),
                                                  ("volt_var_droop", DroopController(battery, **m3["droop"])))
    for name, controller in references:
        print(name, flush=True)
        output["references"][name] = evaluate(controller, True, data, battery, solver, weights)[0]

    selected, checks = select_soc_variant(candidates)
    if args.no_selection:
        selected = "not applied: seed report, no re-selection (docs/decisions.md, seed reporting rule)"
    output["checks"], output["selected_by_rule"] = checks, selected
    out_path = ROOT / args.out
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists. Earlier results are never overwritten.")
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps({"checks": checks, "selected_by_rule": selected}, indent=1))
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
