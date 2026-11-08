"""Milestone M2: build the time-series dataset from the downloaded files.

Run:  python scripts/cache_pv_m2.py     (once, a few minutes)
      python scripts/prepare_m2.py
Writes data/processed/m2_dataset.npz (the time series) and results/m2_summary.json (config, versions, counts).

Everything measured with values of the network is on the TRAINING part. For val and test this script reports only
counts of usable data, the feeder peak and the share of steps labelled extreme, which are properties of the data.
"""
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandapower
import pandas as pd
import torch

from dpc_grid import load_case33bw
from dpc_grid.data import SPLITS, build_dataset, read_uci_load, uci_local_to_utc_end, window_starts

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SUMMARY = ROOT / "results" / "m2_summary.json"
EVERY_NTH_TRAIN_STEP = 6  # the no-control check below uses half-hourly training steps


def no_control_check(net, d, scale, batch=4096):
    """Voltages with PV but no battery action, on valid training steps. Shows whether the data make sense for the feeder."""
    steps = np.flatnonzero((d["labels"] == 0) & d["valid"])[::EVERY_NTH_TRAIN_STEP]
    solver = net.make_solver()
    v = []
    with torch.no_grad():
        for part in np.array_split(steps, max(1, len(steps) // batch)):
            pl = torch.as_tensor(scale * d["p_load"][part], dtype=torch.float32)
            ql = torch.as_tensor(scale * d["q_load"][part], dtype=torch.float32)
            ps = torch.as_tensor(np.nan_to_num(d["p_solar"][part]), dtype=torch.float32)
            v.append(solver(pl, ql, ps)["v"].numpy())
    v = np.concatenate(v)
    outside = (v < 0.95) | (v > 1.05)
    return {"load_scale": scale, "steps_checked": int(len(steps)), "v_min": float(v.min()), "v_max": float(v.max()),
            "share_of_steps_with_any_bus_below_0.95": float((v < 0.95).any(axis=1).mean()),
            "share_of_steps_with_any_bus_above_1.05": float((v > 1.05).any(axis=1).mean()),
            "share_of_bus_steps_outside_band": float(outside.mean())}


def main():
    t0 = time.perf_counter()
    config = json.loads((ROOT / "configs" / "m2.json").read_text())
    net = load_case33bw()
    config["s_base_mva"] = float(net.s_base_mva)

    print("Reading the load table...", flush=True)
    first = pd.Timestamp(config["time"]["train"][0])
    last = pd.Timestamp(config["time"]["test"][1]) - pd.Timedelta(minutes=15)
    load = uci_local_to_utc_end(read_uci_load(RAW / "load" / "electricityloaddiagrams20112014.zip", first, last))

    print("Reading the PV cache and building the dataset...", flush=True)
    cache = np.load(PROCESSED / "pv_5min_systems.npz")
    pv_stamps = pd.to_datetime(cache["stamps"], unit="s")
    names = cache["names"].astype(str).tolist()
    meta = pd.read_csv(RAW / "solar" / "metadata.csv", sep=";", usecols=range(12)).set_index("ID")
    capacity = meta.loc[names, "estimated_dc_capacity"].to_numpy(float)
    d = build_dataset(load, pv_stamps, cache["values_w"], names, capacity, config, net.p_load, net.q_load)

    stamps, labels, valid = d["stamps"], d["labels"], d["valid"]
    H = config["time"]["horizon_steps"]
    per_split = {}
    for k, name in enumerate(SPLITS):
        mask = labels == k
        known = mask & ~np.isnan(d["r_mw_per_min"])
        total_mw = d["p_load"][mask].sum(axis=1) * net.s_base_mva
        per_split[name] = {
            "first_step": str(stamps[mask][0]), "last_step": str(stamps[mask][-1]),
            "solar_first": str(d["source_stamps"][mask][0]), "solar_last": str(d["source_stamps"][mask][-1]),
            "steps": int(mask.sum()), "valid_steps": int(valid[mask].sum()), "valid_share": float(valid[mask].mean()),
            "full_windows": int(len(window_starts(valid, labels, k, H, stamps, config["time"]["dt_min"]))),
            "feeder_load_peak_mw_at_x1": float(total_mw.max()), "feeder_load_mean_mw_at_x1": float(total_mw.mean()),
            "steps_with_known_ramp": int(known.sum()), "extreme_steps": int(d["extreme"][mask].sum()),
            "extreme_share_of_known": float(d["extreme"][mask].sum() / known.sum()),
        }

    # Training-only description of the data.
    train = labels == 0
    r_train = d["r_mw_per_min"][train]
    extreme_train = d["extreme"] & train
    moved = np.r_[False, (np.diff(d["reporting"], axis=0) != 0).any(axis=1)]  # the set of reporting systems changed
    pv_train = d["total_pv_mw"][train]
    peak_ratio = d["p_load"][train].max(axis=0)[d["load_buses"]] / net.p_load[d["load_buses"]]
    group_mw = [g for g in d["load_groups"]]
    client_energy = load[d["clients"]][(load.index > first) & (load.index <= pd.Timestamp(config["time"]["train"][1]))].sum()
    largest_in_group = [float(client_energy[g].max() / client_energy[g].sum()) for g in group_mw]

    summary = {
        "note": "Solar and load come from different countries and years and share no real weather.",
        "per_split": per_split,
        "load": {"clients_used": len(d["clients"]), "clients_total": int(load.shape[1]),
                 "clients_per_bus": sorted({len(g) for g in d["load_groups"]}),
                 "train_bus_peak_over_case33bw_min": float(peak_ratio.min()),
                 "train_bus_peak_over_case33bw_max": float(peak_ratio.max()),
                 "largest_client_share_within_a_bus_median": float(np.median(largest_in_group)),
                 "largest_client_share_within_a_bus_max": float(np.max(largest_in_group))},
        "solar": {"systems_usable": d["n_pv_usable"], "systems_total": len(names),
                  "systems_per_bus": [len(g) for g in d["pv_groups"]],
                  "train_max_total_mw": float(np.nanmax(pv_train)), "train_mean_total_mw": float(np.nanmean(pv_train)),
                  "train_share_of_valid_steps_with_output": float(np.nanmean(pv_train > 0))},
        "ramp": {"unit": "MW per minute, total feeder PV", "definition": config["ramp"],
                 "threshold_mw_per_min_from_train": d["threshold"],
                 "threshold_in_percent_of_pv_capacity_per_5_min": 100 * d["threshold"] * 5 / config["placement"]["pv_total_mw"],
                 "train_max_abs_ramp_mw_per_min": float(np.nanmax(np.abs(r_train))),
                 "train_extreme_share_when_reporting_systems_changed": float(moved[extreme_train].mean()),
                 "train_all_steps_share_when_reporting_systems_changed": float(moved[train].mean())},
        "train_no_control_check": [no_control_check(net, d, config["scenarios"][k])
                                   for k in ("base_load_scale", "stress_load_scale")],
        "config": config,
        "downloads": json.loads((ROOT / "results" / "m2_download.json").read_text()),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "torch": torch.__version__, "pandapower": pandapower.__version__},
    }

    np.savez_compressed(PROCESSED / "m2_dataset.npz",
                        stamps=stamps.to_numpy().astype("datetime64[s]").astype(np.int64),
                        source_stamps=d["source_stamps"].to_numpy().astype("datetime64[s]").astype(np.int64),
                        labels=labels, valid=valid, extreme=d["extreme"], threshold=d["threshold"],
                        p_load_x1=d["p_load"].astype(np.float32), q_load_x1=d["q_load"].astype(np.float32),
                        p_solar=d["p_solar"].astype(np.float32), total_pv_mw=d["total_pv_mw"].astype(np.float32),
                        r_mw_per_min=d["r_mw_per_min"].astype(np.float32))
    (PROCESSED / "m2_groups.json").write_text(json.dumps(
        {"load_buses": d["load_buses"], "load_groups": d["load_groups"], "pv_buses": config["placement"]["pv_buses"],
         "pv_groups": d["pv_groups"]}, indent=1))
    SUMMARY.write_text(json.dumps(summary, indent=2))

    printable = {k: summary[k] for k in ("per_split", "load", "solar", "ramp", "train_no_control_check")}
    print(json.dumps(printable, indent=1))
    print(f"\nDone in {time.perf_counter() - t0:.0f} s. Saved {SUMMARY}")


if __name__ == "__main__":
    main()
