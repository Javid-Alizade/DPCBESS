"""Milestone M3: the pre-specified Volt-VAR droop tuning on the TRAINING part, then a validation comparison.

Run:  python scripts/tune_droop_m3.py
Protocol: docs/decisions.md, "Volt-VAR droop tuning protocol". 16 settings in one batched continuous loop on the training part, both
regimes, the same objective as the DPC candidates. The minimum wins, and a tie (within 1e-9) goes to the smaller mean |Q|.
No re-tuning afterwards. Then, on validation only: the original droop, the tuned droop and the selected DPC side by side.
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
OUT = ROOT / "results" / "m3_droop_tuning.json"
REGIMES = {"base_x0.7": 0.7, "stress_x1.0": 1.0}
DEAD_BANDS = [0.0, 0.005, 0.01, 0.02]
FULL_AT = [0.03, 0.04, 0.05, 0.06]
TIE = 1e-9
DPC_RUN = "results/runs/20261106-124919_candidate_no-deg_seed0"  # H144_dead_zone_30_70, seed 0, selected and confirmed


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
    m2 = json.loads((ROOT / "configs" / "m2.json").read_text())
    m3 = json.loads((ROOT / "configs" / "m3.json").read_text())
    net = load_case33bw()
    data = M3Data(ROOT / "data" / "processed" / "m2_dataset.npz", net.s_base_mva, m2["time"]["dt_min"]).prepare(
        m2["placement"]["pv_buses"], m2["placement"]["bess_buses"])
    battery = per_unit_battery(m3["battery"], net.s_base_mva)
    solver = net.make_solver(n_iter=m3["evaluation"]["n_iter"])
    weights = m3["weights"]
    output = {"protocol": "docs/decisions.md, Volt-VAR droop tuning protocol", **git_state(ROOT),
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
              "original": m3["droop"], "grid": {"dead_band_pu": DEAD_BANDS, "full_at_pu": FULL_AT}}

    # 1. Tuning on the training part only. All 16 settings run together.
    grid = list(itertools.product(DEAD_BANDS, FULL_AT))
    dead = torch.tensor([g[0] for g in grid]).unsqueeze(1)
    full = torch.tensor([g[1] for g in grid]).unsqueeze(1)
    controller = DroopController(battery, dead, full)
    settings = [{"dead_band_pu": d, "full_at_pu": f, "regimes": {}} for d, f in grid]
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
    best = min(s["objective"] for s in settings)
    tied = [s for s in settings if s["objective"] - best <= TIE]
    chosen = min(tied, key=lambda s: s["mean_abs_q_mvar_overall"])
    output["training_part"] = settings
    output["tuned"] = {"dead_band_pu": chosen["dead_band_pu"], "full_at_pu": chosen["full_at_pu"],
                       "training_objective": chosen["objective"], "n_tied": len(tied)}
    print(f"tuned droop: dead band {chosen['dead_band_pu']}, full at {chosen['full_at_pu']}, "
          f"training objective {chosen['objective']:.5f}", flush=True)

    # 2. Validation, after the choice. Nothing below can change it.
    policy = make_policy(m3)
    policy.load_state_dict(torch.load(ROOT / DPC_RUN / "model.pt"))
    policy.eval()
    controllers = {"droop_original": DroopController(battery, m3["droop"]["dead_band_pu"], m3["droop"]["full_at_pu"]),
                   "droop_tuned": DroopController(battery, chosen["dead_band_pu"], chosen["full_at_pu"]),
                   "dpc_H144_dead_zone_30_70_seed0": PolicyController(policy, battery)}
    output["validation"] = {}
    for name, c in controllers.items():
        print(f"validation: {name}", flush=True)
        output["validation"][name] = validation_row(c, data, battery, solver, weights, net.s_base_mva)
    OUT.write_text(json.dumps(output, indent=2))
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
