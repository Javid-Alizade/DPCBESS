"""Milestone M2: are the extreme ramps real, or caused by PV systems dropping in and out of the data?

Run after prepare_m2.py:  python scripts/check_ramps_m2.py
For every TRAINING step, the ramp is computed again using only the systems that report on both this step and the
previous one. If this same-set ramp is close to the normal one, a changing set of systems is not creating the ramp.
Writes results/m2_ramp_check.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main():
    config = json.loads((ROOT / "configs" / "m2.json").read_text())
    d = np.load(PROCESSED / "m2_dataset.npz")
    groups = json.loads((PROCESSED / "m2_groups.json").read_text())["pv_groups"]
    cache = np.load(PROCESSED / "pv_5min_systems.npz")
    pv_stamps = pd.to_datetime(cache["stamps"], unit="s")
    names = cache["names"].astype(str).tolist()
    meta = pd.read_csv(ROOT / "data" / "raw" / "solar" / "metadata.csv", sep=";", usecols=range(12)).set_index("ID")

    train = d["labels"] == 0
    source = pd.to_datetime(d["source_stamps"][train], unit="s")
    position = pv_stamps.get_indexer(source)
    r = d["r_mw_per_min"][train].astype(float)
    extreme = d["extreme"][train]
    threshold = float(d["threshold"])
    bus_mw = config["placement"]["pv_total_mw"] / len(groups)
    dt = config["time"]["dt_min"]

    same_set = np.zeros(len(r))
    changed = np.zeros(len(r), dtype=bool)
    for group in groups:
        columns = [names.index(x) for x in group]
        values = cache["values_w"][np.clip(position, 0, None)][:, columns].astype(float)
        values[position < 0] = np.nan
        capacity = meta.loc[group, "estimated_dc_capacity"].to_numpy(float)
        ok = ~np.isnan(values)
        both = ok[1:] & ok[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            now = np.where(both, values[1:], 0).sum(axis=1) / (both * capacity).sum(axis=1)
            before = np.where(both, values[:-1], 0).sum(axis=1) / (both * capacity).sum(axis=1)
        same_set[1:] += (now - before) * bus_mw / dt
        changed[1:] |= (ok[1:] != ok[:-1]).any(axis=1)

    known = ~np.isnan(r)
    daytime = known & (d["total_pv_mw"][train] > 0)
    gap = np.abs(np.abs(r) - np.abs(same_set))
    same_threshold = float(np.nanpercentile(np.abs(same_set[known]), 100 - config["ramp"]["extreme_top_percent"]))
    result = {
        "part": "train only",
        "threshold_mw_per_min": threshold,
        "extreme_steps": int(extreme.sum()),
        "daytime_steps": int(daytime.sum()),
        "extreme_share_of_daytime_steps": float(extreme.sum() / daytime.sum()),
        "falling_share_of_extreme_steps": float((r[extreme] < 0).mean()),
        "reporting_set_changes_share_daytime_steps": float(changed[daytime].mean()),
        "reporting_set_changes_share_extreme_steps": float(changed[extreme].mean()),
        "extreme_steps_also_extreme_with_same_set": float((np.abs(same_set[extreme]) >= threshold).mean()),
        "gap_over_threshold_median": float(np.median(gap[extreme]) / threshold),
        "gap_over_threshold_p95": float(np.quantile(gap[extreme], 0.95) / threshold),
        "gap_over_threshold_max": float(gap[extreme].max() / threshold),
        "same_set_threshold_over_threshold": same_threshold / threshold,
    }
    out = ROOT / "results" / "m2_ramp_check.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
