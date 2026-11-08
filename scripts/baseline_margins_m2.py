"""RQ2 metrics with no controller: PV as measured, batteries idle. TRAINING part only, in both regimes.

Run after prepare_m2.py:  python scripts/baseline_margins_m2.py
Writes results/m2_margin_baseline.json. Voltages are never computed on validation or test here.
"""
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dpc_grid import load_case33bw
from dpc_grid.metrics import network_voltages, summarize_rq2

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "configs" / "m2.json").read_text())
    metrics_config = dict(config["rq2_metrics"], seed=config["seed"])
    raw = np.load(ROOT / "data" / "processed" / "m2_dataset.npz")
    train = raw["labels"] == 0  # everything below sees only the training part
    r = raw["r_mw_per_min"][train]
    stamps = pd.to_datetime(raw["stamps"][train], unit="s")
    labels = raw["labels"][train]
    p_load, q_load = raw["p_load_x1"][train], raw["q_load_x1"][train]
    p_solar = np.nan_to_num(raw["p_solar"][train])

    net = load_case33bw()
    result = {"part": "train only", "controller": "none, batteries idle",
              "pv_total_mw": config["placement"]["pv_total_mw"], "pv_buses": config["placement"]["pv_buses"],
              "metrics_config": metrics_config, "regimes": {},
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    for name, key in (("base_x0.7", "base_load_scale"), ("stress_x1.0", "stress_load_scale")):
        print(f"Running {name}...", flush=True)
        v = network_voltages(net, p_load, q_load, p_solar, config["scenarios"][key])
        result["regimes"][name] = summarize_rq2(v, raw["extreme"][train], raw["valid"][train], ~np.isnan(r), labels, 0,
                                                stamps, metrics_config, config["time"]["dt_min"])
    out = ROOT / "results" / "m2_margin_baseline.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["regimes"], indent=1))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
