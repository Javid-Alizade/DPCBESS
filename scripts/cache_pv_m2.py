"""Milestone M2, step 1: average the 3 GB 1-minute PV table to 5-minute steps and cache it.

Run:  python scripts/cache_pv_m2.py
This takes minutes and only has to run once. It reads the years selected in configs/m2.json.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dpc_grid.data import aggregate_pv_to_steps

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "raw" / "solar" / "filtered_pv_power_measurements_ac.csv"
CACHE = ROOT / "data" / "processed" / "pv_5min_systems.npz"


def main():
    full = json.loads((ROOT / "configs" / "m2.json").read_text())
    config = full["time"]
    first_year = min(config["load_years"]) + config["solar_year_offset"]["train"]
    last_year = max(config["load_years"]) + config["solar_year_offset"]["test"]
    # Scenario time = solar UTC + 1 h - offset years, so the solar data start 1 h before the first year.
    start = pd.Timestamp(f"{first_year}-01-01") - pd.Timedelta(hours=1)
    end = pd.Timestamp(f"{last_year + 1}-01-01") - pd.Timedelta(hours=1)
    t0 = time.perf_counter()
    stamps, values, names = aggregate_pv_to_steps(CSV, start, end, step_min=config["dt_min"],
                                                  min_samples=full["selection"]["pv_min_valid_minutes_per_step"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, stamps=stamps.to_numpy().astype("datetime64[s]").astype(np.int64),
                        values_w=values.astype(np.float32), names=np.array(names))
    print(f"{values.shape[0]} steps, {values.shape[1]} systems, {stamps[0]} to {stamps[-1]}, "
          f"{time.perf_counter() - t0:.0f} s. Saved {CACHE}")


if __name__ == "__main__":
    main()
