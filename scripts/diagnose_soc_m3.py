"""Milestone M3: descriptive diagnostics of active power and SoC in the continuous loop. VALIDATION part only.

Run:  python scripts/diagnose_soc_m3.py --out results/<name>.json name=results/runs/<id> ...
These numbers only describe how candidates already chosen by the rule behave. They never feed back into the selection.

Definitions (written here, before the first run):
- A step's day is the scenario-clock date of the interval START (stamps mark the end), as in the RQ2 metrics.
- A full day has all 288 steps valid. Daily quantities (throughput, SoC curves, swings) use full days only.
- Charging: P < -0.001 MW. Discharging: P > 0.001 MW. Otherwise idle (|P| <= 0.001 MW, 0.2% of the 0.5 MW limit).
- Throughput: sum of |P| x dt over a day, on the AC side, in MWh/day. Equivalent full cycles = throughput / (2 x 1 MWh).
- Midday charge: energy charged with interval start in 10:00-15:00. Evening discharge: energy discharged in 17:00-22:00.
- Daily SoC curve: SoC after each step of a full day; mean and 10-90% band over days; 3 example days drawn with seed 0.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dpc_grid import load_case33bw
from dpc_grid.dpc import M3Data, PolicyController, closed_loop, end_of_day_soc, make_policy, per_unit_battery
from dpc_grid.runlog import git_state

ROOT = Path(__file__).resolve().parents[1]
REGIMES = {"base_x0.7": 0.7, "stress_x1.0": 1.0}
IDLE_MW = 0.001
STEPS_PER_DAY = 288


def hours_of(index, start, end):
    """Mask of steps whose interval starts in [start, end) hours."""
    return (index >= start * 12) & (index < end * 12)


def describe(result, data, s_base, dt_h, energy_mwh, rng):
    """All descriptive numbers for one controller in one regime."""
    val = data.labels == 1
    used = val & data.valid
    p = result["action"][..., 0] * s_base  # MW, (steps, batteries), P > 0 is discharge
    soc = result["soc"]
    p_used = p[used]

    start = data.stamps - np.timedelta64(data.dt_min, "m")
    day = np.asarray(start.normalize())
    index_in_day = np.asarray(start.hour * 12 + start.minute // 5)
    full_days = [d for d in np.unique(day[used]) if (used & (day == d)).sum() == STEPS_PER_DAY]

    daily_p, daily_soc = [], []
    for d in full_days:
        rows = np.flatnonzero(used & (day == d))
        rows = rows[np.argsort(index_in_day[rows])]
        daily_p.append(p[rows])
        daily_soc.append(soc[rows])
    daily_p, daily_soc = np.stack(daily_p), np.stack(daily_soc)  # (days, 288, batteries)
    k = np.arange(STEPS_PER_DAY)
    throughput = np.abs(daily_p).sum(axis=1) * dt_h
    midday = hours_of(k, 10, 15)
    evening = hours_of(k, 17, 22)
    midday_charge = np.clip(-daily_p[:, midday], 0, None).sum(axis=1) * dt_h
    evening_discharge = np.clip(daily_p[:, evening], 0, None).sum(axis=1) * dt_h

    hourly_p = np.stack([daily_p[:, hours_of(k, h, h + 1)].mean(axis=(0, 1)) for h in range(24)])
    examples = np.sort(rng.choice(len(full_days), 3, replace=False))
    eod = end_of_day_soc(result, data, "val")

    def per_battery(x):
        return [float(v) for v in x]

    return {
        "valid_steps": int(used.sum()), "full_days": len(full_days),
        "mean_p_mw": per_battery(p_used.mean(axis=0)),
        "mean_abs_p_mw": per_battery(np.abs(p_used).mean(axis=0)),
        "share_charging": per_battery((p_used < -IDLE_MW).mean(axis=0)),
        "share_discharging": per_battery((p_used > IDLE_MW).mean(axis=0)),
        "share_idle": per_battery((np.abs(p_used) <= IDLE_MW).mean(axis=0)),
        "throughput_mwh_per_day_mean": per_battery(throughput.mean(axis=0)),
        "equivalent_full_cycles_per_day_mean": per_battery(throughput.mean(axis=0) / (2 * energy_mwh)),
        "midday_10_15_charge_mwh_per_day_mean": per_battery(midday_charge.mean(axis=0)),
        "evening_17_22_discharge_mwh_per_day_mean": per_battery(evening_discharge.mean(axis=0)),
        "daily_soc_swing_mean": per_battery((daily_soc.max(axis=1) - daily_soc.min(axis=1)).mean(axis=0)),
        "hourly_mean_p_mw": hourly_p.tolist(),
        "soc_curve_mean": daily_soc.mean(axis=0).tolist(),
        "soc_curve_p10": np.percentile(daily_soc, 10, axis=0).tolist(),
        "soc_curve_p90": np.percentile(daily_soc, 90, axis=0).tolist(),
        "example_days": [str(np.datetime_as_string(full_days[i], unit="D")) for i in examples],
        "example_soc": [daily_soc[i].tolist() for i in examples],
        "end_of_day_soc_quantiles": {q: per_battery(np.percentile(eod, q, axis=0)) for q in (0, 5, 25, 50, 75, 95, 100)},
        "end_of_day_soc_histogram": {"edges": np.linspace(0.1, 0.9, 17).tolist(),
                                     "counts": [np.histogram(eod[:, b], np.linspace(0.1, 0.9, 17))[0].tolist()
                                                for b in range(eod.shape[1])]},
        "end_of_day_days": int(len(eod)),
    }


def figure(output, data, path):
    """Rows: candidate x regime. Columns: daily SoC curve, hourly mean P with mean PV, end-of-day SoC histogram."""
    rows = [(n, r) for n in output["candidates"] for r in REGIMES]
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 3.2 * len(rows)), squeeze=False)
    hours = (np.arange(STEPS_PER_DAY) + 1) / 12
    for i, (name, regime) in enumerate(rows):
        d = output["candidates"][name][regime]
        mean, low, high = (np.array(d[k]) for k in ("soc_curve_mean", "soc_curve_p10", "soc_curve_p90"))
        ax = axes[i, 0]
        for b, colour in enumerate(("C0", "C1")):
            ax.fill_between(hours, low[:, b], high[:, b], color=colour, alpha=0.2)
            ax.plot(hours, mean[:, b], color=colour, label=f"battery {b + 1} mean")
            for example in d["example_soc"]:
                ax.plot(hours, np.array(example)[:, b], color=colour, lw=0.6, ls="--")
        ax.axhspan(0.3, 0.7, color="grey", alpha=0.08)
        ax.set(ylim=(0.1, 0.9), xlim=(0, 24), xlabel="hour of scenario day", ylabel="SoC",
               title=f"{name}, {regime}: SoC (band 10-90%, dashed {', '.join(d['example_days'])})")
        ax.title.set_fontsize(8)
        ax.legend(fontsize=7)
        ax = axes[i, 1]
        hourly = np.array(d["hourly_mean_p_mw"])
        ax.bar(np.arange(24) + 0.3, hourly[:, 0], 0.4, label="battery 1")
        ax.bar(np.arange(24) + 0.7, hourly[:, 1], 0.4, label="battery 2")
        ax.axhline(0, color="k", lw=0.5)
        ax.set(xlim=(0, 24), xlabel="hour (interval start)", ylabel="mean P, MW (>0 discharge)", title="hourly mean P")
        twin = ax.twinx()
        twin.plot(np.arange(24) + 0.5, output["mean_pv_mw_by_hour"], color="goldenrod", label="mean feeder PV")
        twin.set_ylabel("PV, MW")
        ax.legend(fontsize=7, loc="upper left")
        ax = axes[i, 2]
        h = d["end_of_day_soc_histogram"]
        edges = np.array(h["edges"])
        for b in range(2):
            ax.bar(edges[:-1] + 0.025 * (b + 0.5), h["counts"][b], 0.025, label=f"battery {b + 1}")
        ax.set(xlabel="end-of-day SoC", ylabel="days", title=f"end-of-day SoC ({d['end_of_day_days']} days)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("runs", nargs="+", help="name=run folder")
    args = parser.parse_args()
    out_path = ROOT / args.out
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists. Earlier results are never overwritten.")
    m2 = json.loads((ROOT / "configs" / "m2.json").read_text())
    m3 = json.loads((ROOT / "configs" / "m3.json").read_text())
    net = load_case33bw()
    data = M3Data(ROOT / "data" / "processed" / "m2_dataset.npz", net.s_base_mva, m2["time"]["dt_min"]).prepare(
        m2["placement"]["pv_buses"], m2["placement"]["bess_buses"])
    battery = per_unit_battery(m3["battery"], net.s_base_mva)
    solver = net.make_solver(n_iter=m3["evaluation"]["n_iter"])

    used = (data.labels == 1) & data.valid
    start = data.stamps - np.timedelta64(data.dt_min, "m")
    pv = data.p_solar.sum(dim=1).numpy() * net.s_base_mva
    hour = np.asarray(start.hour)
    output = {"part": "validation only", "purpose": "descriptive, does not change the selection", **git_state(ROOT),
              "definitions": __doc__, "idle_threshold_mw": IDLE_MW,
              "mean_pv_mw_by_hour": [float(pv[used & (hour == h)].mean()) for h in range(24)], "candidates": {}}
    for name, folder in dict(arg.split("=", 1) for arg in args.runs).items():
        record = json.loads((Path(folder) / "run.json").read_text())
        policy = make_policy(m3)
        policy.load_state_dict(torch.load(Path(folder) / "model.pt"))
        policy.eval()
        output["candidates"][name] = {"run_id": record["run_id"]}
        for regime, value in REGIMES.items():
            print(f"{name}, {regime}", flush=True)
            result = closed_loop(PolicyController(policy, battery), data, "val", value, battery, solver, True)
            rng = np.random.default_rng(0)  # the same example days for every candidate and regime
            output["candidates"][name][regime] = describe(result, data, net.s_base_mva, data.dt_h,
                                                          m3["battery"]["energy_mwh"], rng)
    out_path.write_text(json.dumps(output, indent=1))
    figure(output, data, out_path.with_suffix(".png"))
    print(f"Saved {out_path} and {out_path.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
