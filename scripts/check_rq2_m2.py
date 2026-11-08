"""Before M3: can the data test RQ2, voltage violations during extreme solar-ramp events?

Run after prepare_m2.py:  python scripts/check_rq2_m2.py
TRAINING part only, no controller: PV as measured, batteries idle, no reactive power from any device.
Writes results/m2_rq2_check.json.

Definitions:
  violation    any bus below 0.95 pu or above 1.05 pu at this step
  extreme      |r_t| >= the training threshold of total feeder PV, as labelled by prepare_m2.py
  steps used   valid training steps whose ramp is known
"""
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dpc_grid import load_case33bw
from dpc_grid.data import any_in_next_steps

ROOT = Path(__file__).resolve().parents[1]
V_LOW, V_HIGH = 0.95, 1.05
BATCH = 4096


def bus_voltages(net, p_load, q_load, p_solar, scale):
    """Solver voltages for all steps, in batches."""
    solver = net.make_solver()
    out = []
    with torch.no_grad():
        for start in range(0, len(p_load), BATCH):
            stop = start + BATCH
            out.append(solver(torch.as_tensor(scale * p_load[start:stop]), torch.as_tensor(scale * q_load[start:stop]),
                              torch.as_tensor(p_solar[start:stop]))["v"].numpy())
    return np.concatenate(out)


def share(numerator, denominator):
    """Share of `denominator` steps where `numerator` holds, with the counts next to it."""
    count, total = int((numerator & denominator).sum()), int(denominator.sum())
    return {"share": count / total if total else None, "count": count, "of": total}


def analyse(net, d, scale, horizon, step_min):
    v = bus_voltages(net, d["p_load"], d["q_load"], d["p_solar"], scale)
    low, high = (v < V_LOW).any(axis=1), (v > V_HIGH).any(axis=1)
    violation = low | high
    used, extreme, r, stamps = d["used"], d["extreme"], d["r"], d["stamps"]
    daytime = used & (d["total_pv_mw"] > 0)
    rising, falling = extreme & (r > 0), extreme & (r < 0)
    previous = np.r_[False, violation[:-1]]
    soon = any_in_next_steps(violation, stamps, horizon, step_min)
    days = pd.DatetimeIndex(stamps - pd.Timedelta(minutes=step_min)).normalize()

    def n_days(mask):
        return int(pd.Series(days[mask]).nunique())

    return {
        "load_scale": scale,
        "v_min": float(v[used].min()), "v_max": float(v[used].max()),
        "requested": {
            "extreme_among_violation_steps": share(extreme, used & violation),
            "violation_among_extreme_steps": share(violation, used & extreme),
        },
        "for_comparison": {
            "extreme_among_all_steps": share(extreme, used),
            "violation_among_all_steps": share(violation, used),
            "violation_among_daytime_steps": share(violation, daytime),
            "violation_among_daytime_steps_that_are_not_extreme": share(violation, daytime & ~extreme),
        },
        "by_type_and_direction": {
            "below_0.95_among_extreme": share(low, used & extreme),
            "above_1.05_among_extreme": share(high, used & extreme),
            "below_0.95_among_daytime": share(low, daytime),
            "above_1.05_among_daytime": share(high, daytime),
            "below_0.95_among_falling_extreme": share(low, falling),
            "above_1.05_among_falling_extreme": share(high, falling),
            "below_0.95_among_rising_extreme": share(low, rising),
            "above_1.05_among_rising_extreme": share(high, rising),
        },
        "new_violation": {
            "starts_at_this_step_among_extreme": share(violation & ~previous, used & extreme),
            "starts_at_this_step_among_daytime": share(violation & ~previous, daytime),
        },
        f"violation_within_next_{horizon}_steps": {
            "among_extreme": share(soon, used & extreme),
            "among_daytime": share(soon, daytime),
        },
        "events": {
            "days_in_training_part": int(pd.Series(days[used]).nunique()),
            "days_with_extreme_step": n_days(used & extreme),
            "days_with_extreme_step_and_violation_at_it": n_days(used & extreme & violation),
            "extreme_and_violation_steps_by_month": {
                int(m): int(c) for m, c in pd.Series(days[used & extreme & violation].month).value_counts().sort_index().items()},
        },
    }


def main():
    config = json.loads((ROOT / "configs" / "m2.json").read_text())
    place = config["placement"]
    if place["pv_total_mw"] != 3.9 or place["pv_buses"] != [14, 17, 29, 32]:
        raise ValueError("this check is defined for scheme B with 3.9 MW of PV")
    raw = np.load(ROOT / "data" / "processed" / "m2_dataset.npz")
    train = raw["labels"] == 0  # everything below sees only the training part
    r = raw["r_mw_per_min"][train].astype(float)
    data = {"stamps": pd.to_datetime(raw["stamps"][train], unit="s"), "r": r, "extreme": raw["extreme"][train],
            "used": raw["valid"][train] & ~np.isnan(r), "total_pv_mw": np.nan_to_num(raw["total_pv_mw"][train]),
            "p_load": raw["p_load_x1"][train], "q_load": raw["q_load_x1"][train],
            "p_solar": np.nan_to_num(raw["p_solar"][train])}

    net = load_case33bw()
    horizon, step_min = config["time"]["horizon_steps"], config["time"]["dt_min"]
    result = {"part": "train only", "controller": "none, batteries idle",
              "band_pu": [V_LOW, V_HIGH], "threshold_mw_per_min": float(raw["threshold"]),
              "regimes": {}, "pv_total_mw": place["pv_total_mw"], "pv_buses": place["pv_buses"],
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    for name, key in (("base_x0.7", "base_load_scale"), ("stress_x1.0", "stress_load_scale")):
        print(f"Running {name}...", flush=True)
        result["regimes"][name] = analyse(net, data, config["scenarios"][key], horizon, step_min)
    out = ROOT / "results" / "m2_rq2_check.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
