"""Milestone M3: Volt-VAR droop tuning ROUND 2 on the TRAINING part, added after seeing round 1, then a validation comparison.

Run:  python scripts/tune_droop_round2_m3.py
Protocol: docs/decisions.md, "Droop tuning round 2". 12 new settings in one batched continuous loop on the training part, the same
objective and tie rule as round 1. The final tuned droop is the minimum over all 28 settings of rounds 1 and 2. No further tuning.
Validation: original, round-1 tuned and final tuned droop, and the selected DPC with seeds 0, 1, 2 (median and range).
The test part stays closed (M3Data cannot read it).
"""
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import torch

from dpc_grid import load_case33bw
from dpc_grid.dpc import (DroopController, M3Data, PolicyController, closed_loop, closed_loop_batched, closed_loop_summary,
                          continuous_objective, make_policy, per_unit_battery, share_q_sign_changes)
from dpc_grid.runlog import git_state

ROOT = Path(__file__).resolve().parents[1]
ROUND1 = ROOT / "results" / "m3_droop_tuning.json"
OUT = ROOT / "results" / "m3_droop_tuning_round2.json"
REGIMES = {"base_x0.7": 0.7, "stress_x1.0": 1.0}
DEAD_BANDS = [0.0, 0.0025, 0.005]
FULL_AT = [0.01, 0.015, 0.02, 0.025]
TIE = 1e-9
DPC_RUNS = {"seed_0": "results/runs/20261106-124919_candidate_no-deg_seed0",
            "seed_1": "results/runs/20261106-141248_candidate_no-deg_seed1",
            "seed_2": "results/runs/20261106-144916_candidate_no-deg_seed2"}


def mean_abs_q_mvar(result, data, split, s_base):
    """Mean |Q| per battery over the valid steps of a split, in Mvar."""
    used = (data.labels == {"train": 0, "val": 1}[split]) & data.valid
    return (np.abs(result["action"][used][..., 1]) * s_base).mean(axis=0).tolist()


def validation_row(controller, data, battery, solver, weights, s_base):
    """Objective, VVR, network losses, mean |Q| and the share of Q sign changes, per regime, on validation."""
    row, objectives = {}, []
    for regime, value in REGIMES.items():
        result = closed_loop(controller, data, "val", value, battery, solver, True)
        objective = continuous_objective(result, data, "val", battery, weights)
        summary = closed_loop_summary(result, data, "val", battery, None, include_rq2=False)
        objectives.append(objective["objective"])
        row[regime] = {"objective": objective["objective"], "terms": objective["terms"],
                       "vvr_share_of_steps_with_any_violation": summary["vvr_share_of_steps_with_any_violation"],
                       "vvr_share_of_bus_steps": summary["vvr_share_of_bus_steps"],
                       "mean_network_loss_kw": summary["mean_network_loss_kw"],
                       "mean_abs_q_mvar": mean_abs_q_mvar(result, data, "val", s_base),
                       "share_q_sign_changes": share_q_sign_changes(result, data, "val")}
        print(f"  {regime}: objective {objective['objective']:.5f}", flush=True)
    return {"objective": float(np.mean(objectives)), "regimes": row}


def main():
    if OUT.exists():
        raise FileExistsError(f"{OUT} exists. Earlier results are never overwritten.")
    round1 = json.loads(ROUND1.read_text())
    m2 = json.loads((ROOT / "configs" / "m2.json").read_text())
    m3 = json.loads((ROOT / "configs" / "m3.json").read_text())
    net = load_case33bw()
    data = M3Data(ROOT / "data" / "processed" / "m2_dataset.npz", net.s_base_mva, m2["time"]["dt_min"]).prepare(
        m2["placement"]["pv_buses"], m2["placement"]["bess_buses"])
    battery = per_unit_battery(m3["battery"], net.s_base_mva)
    solver = net.make_solver(n_iter=m3["evaluation"]["n_iter"])
    weights = m3["weights"]
    output = {"protocol": "docs/decisions.md, Droop tuning round 2 (added after seeing round 1)", **git_state(ROOT),
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
              "original": m3["droop"], "round1_tuned": round1["tuned"],
              "grid_round2": {"dead_band_pu": DEAD_BANDS, "full_at_pu": FULL_AT}}

    # 1. Round 2 on the training part only. All 12 settings run together.
    grid = list(itertools.product(DEAD_BANDS, FULL_AT))
    controller = DroopController(battery, torch.tensor([g[0] for g in grid]).unsqueeze(1),
                                 torch.tensor([g[1] for g in grid]).unsqueeze(1))
    settings = [{"round": 2, "dead_band_pu": d, "full_at_pu": f, "regimes": {}} for d, f in grid]
    for regime, value in REGIMES.items():
        print(f"training part, {regime}, {len(grid)} settings at once", flush=True)
        results = closed_loop_batched(controller, len(grid), data, "train", value, battery, solver, True)
        for s, result in zip(settings, results):
            objective = continuous_objective(result, data, "train", battery, weights)
            s["regimes"][regime] = {"objective": objective["objective"], "terms": objective["terms"],
                                    "mean_abs_q_mvar": mean_abs_q_mvar(result, data, "train", net.s_base_mva),
                                    "share_q_sign_changes": share_q_sign_changes(result, data, "train")}
        del results
    for s in settings:
        s["objective"] = float(np.mean([r["objective"] for r in s["regimes"].values()]))
        s["mean_abs_q_mvar_overall"] = float(np.mean([r["mean_abs_q_mvar"] for r in s["regimes"].values()]))

    # 2. The choice over all 28 settings of both rounds.
    everything = [dict(s, round=1) for s in round1["training_part"]] + settings
    best = min(s["objective"] for s in everything)
    tied = [s for s in everything if s["objective"] - best <= TIE]
    chosen = min(tied, key=lambda s: s["mean_abs_q_mvar_overall"])
    output["training_part_round2"] = settings
    output["final_tuned"] = {"round": chosen["round"], "dead_band_pu": chosen["dead_band_pu"], "full_at_pu": chosen["full_at_pu"],
                             "training_objective": chosen["objective"], "n_tied": len(tied)}
    print(f"final tuned droop (round {chosen['round']}): dead band {chosen['dead_band_pu']}, full at {chosen['full_at_pu']}, "
          f"training objective {chosen['objective']:.5f}", flush=True)

    # 3. Validation, after the choice. Nothing below can change it.
    controllers = {"droop_original": DroopController(battery, m3["droop"]["dead_band_pu"], m3["droop"]["full_at_pu"]),
                   "droop_round1_tuned": DroopController(battery, round1["tuned"]["dead_band_pu"], round1["tuned"]["full_at_pu"]),
                   "droop_final_tuned": DroopController(battery, chosen["dead_band_pu"], chosen["full_at_pu"])}
    for seed, folder in DPC_RUNS.items():
        policy = make_policy(m3)
        policy.load_state_dict(torch.load(ROOT / folder / "model.pt"))
        policy.eval()
        controllers[f"dpc_H144_dead_zone_30_70_{seed}"] = PolicyController(policy, battery)
    output["validation"] = {}
    for name, c in controllers.items():
        print(f"validation: {name}", flush=True)
        output["validation"][name] = validation_row(c, data, battery, solver, weights, net.s_base_mva)

    # 4. DPC over seeds: median and range of every reported number.
    seeds = [output["validation"][f"dpc_H144_dead_zone_30_70_{s}"] for s in DPC_RUNS]
    over = {"objective": [float(np.median([s["objective"] for s in seeds])), min(s["objective"] for s in seeds),
                          max(s["objective"] for s in seeds)], "regimes": {}}
    for regime in REGIMES:
        over["regimes"][regime] = {}
        for key in ("objective", "vvr_share_of_steps_with_any_violation", "mean_network_loss_kw", "share_q_sign_changes"):
            values = [s["regimes"][regime][key] for s in seeds]
            over["regimes"][regime][key] = [float(np.median(values)), min(values), max(values)]
        values = np.array([s["regimes"][regime]["mean_abs_q_mvar"] for s in seeds])
        over["regimes"][regime]["mean_abs_q_mvar"] = [np.median(values, axis=0).tolist(), values.min(axis=0).tolist(),
                                                      values.max(axis=0).tolist()]
    output["dpc_over_seeds_median_min_max"] = over
    OUT.write_text(json.dumps(output, indent=2))
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
