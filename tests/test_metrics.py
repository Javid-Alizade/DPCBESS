"""RQ2 metrics, checked on small voltage tables where the right answer can be worked out by hand."""
import numpy as np
import pandas as pd
import pytest

from dpc_grid.metrics import (day_bootstrap_ci, day_of_step, ramp_window_metrics, ramp_window_starts, step_margin,
                              summarize_rq2)


def test_margin_uses_the_nearest_limit_and_the_worst_bus():
    v = np.array([[1.00, 1.00],   # 0.05 from both limits
                  [0.97, 1.00],   # the low limit is nearest: 0.02
                  [1.03, 0.99],   # the high limit is nearest: 0.02
                  [0.94, 1.00],   # violation: -0.01
                  [1.00, 1.06]])  # violation: -0.01
    assert step_margin(v) == pytest.approx([0.05, 0.02, 0.02, -0.01, -0.01])


def table(values):
    """One column of voltages plus the slack bus at 1.0, as (T, 2)."""
    return np.column_stack([np.ones(len(values)), values])


def test_window_metrics_look_only_inside_the_forward_window():
    v = table([0.90, 1.00, 0.97, 1.00, 1.00, 0.90])  # steps 0 and 5 are bad but outside the window
    m = ramp_window_metrics(v, np.array([1]), window_steps=4)
    assert m["worst_margin"][0] == pytest.approx(0.02)
    assert m["max_deviation"][0] == pytest.approx(0.03)


def test_max_deviation_counts_both_directions():
    m = ramp_window_metrics(table([1.04, 0.93]), np.array([0]), window_steps=2)
    assert m["max_deviation"][0] == pytest.approx(0.07)
    assert m["worst_margin"][0] == pytest.approx(-0.02)


def test_time_to_safe_band_zero_first_entry_and_censored():
    inside_now = ramp_window_metrics(table([1.00, 0.90, 0.90]), np.array([0]), 3)
    assert inside_now["minutes_to_safe_band"][0] == 0 and not inside_now["censored"][0]
    enters = ramp_window_metrics(table([0.95, 0.955, 0.97]), np.array([0]), 3)
    assert enters["minutes_to_safe_band"][0] == 10 and not enters["censored"][0]
    never = ramp_window_metrics(table([0.95, 0.95, 0.95]), np.array([0]), 3)
    assert never["minutes_to_safe_band"][0] == 15 and never["censored"][0]


def test_safe_band_edges_count_as_inside():
    m = ramp_window_metrics(table([1.04]), np.array([0]), 1)
    assert m["minutes_to_safe_band"][0] == 0 and not m["censored"][0]


def test_window_past_the_end_is_refused():
    with pytest.raises(ValueError):
        ramp_window_metrics(table([1.0, 1.0]), np.array([1]), 2)


def grid(n):
    return pd.date_range("2012-01-01 00:05", periods=n, freq="5min")


def test_only_complete_windows_are_used():
    extreme = np.zeros(10, dtype=bool)
    extreme[[0, 3, 8]] = True
    valid = np.ones(10, dtype=bool)
    valid[5] = False  # breaks the window that starts at step 3
    labels = np.zeros(10, dtype=np.int8)
    starts = ramp_window_starts(extreme, valid, labels, 0, 3, grid(10))
    assert starts.tolist() == [0]  # step 3 reaches missing data, and step 8 runs past the end


def test_windows_do_not_cross_into_the_next_split():
    extreme = np.zeros(6, dtype=bool)
    extreme[[1, 2]] = True
    labels = np.array([0, 0, 0, 0, 1, 1], dtype=np.int8)
    starts = ramp_window_starts(extreme, np.ones(6, dtype=bool), labels, 0, 3, grid(6))
    assert starts.tolist() == [1]


def test_midnight_stamp_belongs_to_the_previous_day():
    days = day_of_step(pd.DatetimeIndex(["2012-01-01 23:55", "2012-01-02 00:00", "2012-01-02 00:05"]))
    assert [str(x)[:10] for x in days] == ["2012-01-01", "2012-01-01", "2012-01-02"]


def test_bootstrap_is_repeatable_and_contains_the_mean():
    rng = np.random.default_rng(1)
    values = rng.normal(size=300)
    days = np.repeat(np.arange(30), 10)
    a = day_bootstrap_ci(values, days, seed=3)
    assert a == day_bootstrap_ci(values, days, seed=3)
    assert a != day_bootstrap_ci(values, days, seed=4)
    assert a[0] < values.mean() < a[1]


def test_bootstrap_resamples_days_not_single_steps():
    # Within a day all values are equal, so there are really only 20 observations and not 400.
    day_level = np.random.default_rng(2).normal(size=20)
    values = np.repeat(day_level, 20)
    days = np.repeat(np.arange(20), 20)
    low, high = day_bootstrap_ci(values, days)
    rng = np.random.default_rng(0)
    step_means = [rng.choice(values, len(values)).mean() for _ in range(2000)]
    step_low, step_high = np.percentile(step_means, [2.5, 97.5])
    assert high - low > 3 * (step_high - step_low)


def test_bootstrap_with_one_day_has_no_spread():
    low, high = day_bootstrap_ci(np.array([1.0, 2.0, 3.0]), np.zeros(3))
    assert low == high == pytest.approx(2.0)


def test_summary_counts_steps_and_days():
    n = 288 * 2  # two days
    stamps = grid(n)
    v = np.ones((n, 2))
    v[100, 1] = 0.94                       # a violation exactly at an extreme step of day 1
    v[400:412, 1] = 0.955                  # day 2: inside the limits, outside the safe band for the whole window
    extreme = np.zeros(n, dtype=bool)
    extreme[[100, 200, 400]] = True
    valid = np.ones(n, dtype=bool)
    config = {"limits_pu": [0.95, 1.05], "safe_band_pu": [0.96, 1.04], "window_steps": 12,
              "bootstrap_samples": 50, "confidence": 0.95}
    out = summarize_rq2(v, extreme, valid, np.ones(n, dtype=bool), np.zeros(n, dtype=np.int8), 0, stamps, config)
    primary, secondary = out["primary_over_ramp_windows"], out["secondary_violations_at_extreme_step"]
    assert primary["windows_used"] == 3 and primary["windows_excluded_incomplete"] == 0
    assert primary["distinct_days"] == 2
    assert primary["worst_margin_pu"]["mean"] == pytest.approx(np.mean([-0.01, 0.05, 0.005]))
    assert primary["censored_share_never_inside_in_window"] == pytest.approx(1 / 3)
    assert primary["share_of_windows_with_any_violation"] == pytest.approx(1 / 3)
    assert secondary["steps"] == 1 and secondary["distinct_days_with_violation"] == 1
    assert secondary["share"] == pytest.approx(1 / 3)
