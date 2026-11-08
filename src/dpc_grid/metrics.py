"""RQ2 metrics: voltage margin and violations during extreme solar-ramp events.

The exact definitions are in docs/decisions.md, section "RQ2 metrics". They were fixed before any controller existed.
All functions take plain numpy arrays, so the same code will score the no-control case, DPC and every baseline.
"""
import numpy as np
import pandas as pd
import torch

from .data import window_starts


def network_voltages(net, p_load, q_load, p_solar, scale=1.0, p_bess=None, q_bess=None, batch=4096):
    """Solver voltages (T, N) for all steps, in batches. `scale` multiplies load only."""
    solver = net.make_solver()
    def tensor(a):
        return None if a is None else torch.as_tensor(a)
    out = []
    with torch.no_grad():
        for start in range(0, len(p_load), batch):
            s = slice(start, start + batch)
            out.append(solver(tensor(scale * p_load[s]), tensor(scale * q_load[s]), tensor(p_solar[s]),
                              p_bess=None if p_bess is None else tensor(p_bess[s]),
                              q_bess=None if q_bess is None else tensor(q_bess[s]))["v"].numpy())
    return np.concatenate(out)


def step_margin(v, limits=(0.95, 1.05)):
    """m_t = min over buses of min(V - low, high - V). Negative means at least one bus violates a limit. v: (T, N)."""
    low, high = limits
    return np.minimum(v - low, high - v).min(axis=1)


def ramp_window_starts(extreme, valid, labels, split, window_steps, stamps, step_min=5):
    """Extreme-ramp steps whose whole forward window is valid, consecutive and inside `split`."""
    complete = window_starts(valid, labels, split, window_steps, stamps, step_min)
    return np.intersect1d(np.flatnonzero(extreme & (labels == split)), complete)


def ramp_window_metrics(v, starts, window_steps, step_min=5, limits=(0.95, 1.05), safe_band=(0.96, 1.04)):
    """The three primary metrics, one value per window. `starts` must have complete windows (see ramp_window_starts)."""
    if len(starts) and starts.max() + window_steps > len(v):
        raise ValueError("a window runs past the end of the voltages")
    margin = step_margin(v, limits)
    deviation = np.abs(v - 1.0).max(axis=1)
    inside = ((v >= safe_band[0]) & (v <= safe_band[1])).all(axis=1)
    worst_margin = np.empty(len(starts))
    max_deviation = np.empty(len(starts))
    minutes = np.empty(len(starts))
    censored = np.zeros(len(starts), dtype=bool)
    for i, t in enumerate(starts):
        w = np.arange(t, t + window_steps)
        worst_margin[i] = margin[w].min()
        max_deviation[i] = deviation[w].max()
        if inside[w].any():
            minutes[i] = step_min * int(np.argmax(inside[w]))  # argmax finds the first True
        else:
            minutes[i] = step_min * window_steps  # never inside: recorded at the window length, and flagged
            censored[i] = True
    return {"worst_margin": worst_margin, "max_deviation": max_deviation,
            "minutes_to_safe_band": minutes, "censored": censored}


def day_bootstrap_ci(values, days, statistic=np.mean, n_samples=2000, confidence=0.95, seed=0):
    """Percentile interval from resampling whole days, because events on one day are not independent."""
    values = np.asarray(values, dtype=float)
    groups = [values[idx] for idx in pd.Series(np.arange(len(values))).groupby(np.asarray(days)).indices.values()]
    rng = np.random.default_rng(seed)
    results = np.empty(n_samples)
    for k in range(n_samples):
        picks = rng.integers(len(groups), size=len(groups))
        results[k] = statistic(np.concatenate([groups[p] for p in picks]))
    tail = 100 * (1 - confidence) / 2
    low, high = np.percentile(results, [tail, 100 - tail])
    return float(low), float(high)


def day_of_step(stamps, step_min=5):
    """Scenario-clock date of each interval start. Stamps mark interval ends, so midnight belongs to the previous day."""
    return pd.DatetimeIndex(stamps - pd.Timedelta(minutes=step_min)).normalize().to_numpy()


def summarize_rq2(v, extreme, valid, known_ramp, labels, split, stamps, config, step_min=5):
    """All RQ2 numbers for one regime and one split, from voltages (T, N) on the same steps as the other arrays."""
    limits, band = tuple(config["limits_pu"]), tuple(config["safe_band_pu"])
    window = config["window_steps"]
    boot = dict(n_samples=config["bootstrap_samples"], confidence=config["confidence"], seed=config.get("seed", 0))
    days = day_of_step(stamps, step_min)

    events = extreme & valid & known_ramp & (labels == split)
    starts = ramp_window_starts(events, valid, labels, split, window, stamps, step_min)
    m = ramp_window_metrics(v, starts, window, step_min, limits, band)
    d = days[starts]

    def describe(values, statistic=np.mean):
        return {"mean": float(np.mean(values)), "median": float(np.median(values)),
                "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95)),
                "mean_ci": list(day_bootstrap_ci(values, d, statistic, **boot))}

    violated = step_margin(v[events], limits) < 0
    event_days = days[events]
    return {
        "primary_over_ramp_windows": {
            "extreme_steps": int(events.sum()), "windows_used": int(len(starts)),
            "windows_excluded_incomplete": int(events.sum() - len(starts)),
            "distinct_days": int(len(set(d.tolist()))),
            "worst_margin_pu": describe(m["worst_margin"]),
            "max_deviation_pu": describe(m["max_deviation"]),
            "minutes_to_safe_band": describe(m["minutes_to_safe_band"]),
            "share_inside_safe_band_at_ramp_step": float((m["minutes_to_safe_band"] == 0).mean()),
            "censored_share_never_inside_in_window": float(m["censored"].mean()),
            "censored_share_ci": list(day_bootstrap_ci(m["censored"], d, np.mean, **boot)),
            "share_of_windows_with_any_violation": float((m["worst_margin"] < 0).mean()),
        },
        "secondary_violations_at_extreme_step": {
            "steps": int(violated.sum()), "of_extreme_steps": int(events.sum()),
            "distinct_days_with_violation": int(len(set(event_days[violated].tolist()))),
            "distinct_days_with_extreme_step": int(len(set(event_days.tolist()))),
            "share": float(violated.mean()),
            "share_ci": list(day_bootstrap_ci(violated, event_days, np.mean, **boot)),
        },
    }
