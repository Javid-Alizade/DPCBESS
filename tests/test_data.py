"""M2: every rule of the data pipeline, checked on small artificial tables built only for these tests."""
import numpy as np
import pandas as pd
import pytest
import torch

from dpc_grid import load_case33bw
from dpc_grid.data import (aggregate_pv_to_steps, any_in_next_steps, assign_groups, build_dataset, bus_load_profiles, bus_pv_profiles,
                           extreme_threshold, hold_to_steps, net_consumption, ramp_rate, scenario_grid,
                           solar_source_stamps, split_labels, uci_local_to_utc_end, window_starts)

TIME = {"dt_min": 5, "solar_year_offset": {"train": 2, "val": 3, "test": 3},
        "train": ["2012-01-01", "2013-07-01"], "val": ["2013-07-01", "2014-01-01"], "test": ["2014-01-01", "2015-01-01"]}


def test_pv_aggregation_labels_by_interval_end_and_needs_enough_minutes(tmp_path):
    path = tmp_path / "pv.csv"
    rows = ["DateTime,A,B"]
    for minute in range(1, 11):  # A reports in all 10 minutes, B only in 2 minutes of the first step
        b = "7.0" if minute <= 2 else ""
        rows.append(f"2015-06-01 12:{minute:02d}:00+00:00,{float(minute)},{b}")
    path.write_text("\n".join(rows))
    stamps, values, names = aggregate_pv_to_steps(path, "2015-06-01 12:00", "2015-06-01 12:10", min_samples=3, chunk_rows=4)
    assert names == ["A", "B"]
    assert list(stamps) == [pd.Timestamp("2015-06-01 12:05"), pd.Timestamp("2015-06-01 12:10")]
    assert values[:, 0] == pytest.approx([3.0, 8.0])  # means of minutes 1-5 and 6-10, across chunk borders
    assert np.isnan(values[:, 1]).all()               # too few minutes, so missing instead of invented


def uci_like(start, end):
    index = pd.date_range(start, end, freq="15min")
    return pd.DataFrame({"c": np.ones(len(index))}, index=index)


def test_march_clock_change_removes_the_hour_that_does_not_exist():
    table = uci_like("2012-03-25 00:00", "2012-03-25 03:45")  # the clock jumps from 01:00 to 02:00 on this day
    out = uci_local_to_utc_end(table)
    assert len(out) == len(table) - 4
    assert out.index[0] == pd.Timestamp("2012-03-25 00:15")
    assert out.index[-1] == pd.Timestamp("2012-03-25 03:00")  # local 03:45 + 15 min, in summer time, is 03:00 UTC


def test_october_clock_change_splits_the_double_hour():
    table = uci_like("2012-10-28 00:00", "2012-10-28 03:45")
    table.loc["2012-10-28 01:00":"2012-10-28 01:45", "c"] = 2.0  # the file holds the sum of two hours
    out = uci_local_to_utc_end(table)
    assert len(out) == len(table) + 4
    assert (out["c"] == 1.0).all()  # the energy is preserved and the jump is gone
    assert out.index[0] == pd.Timestamp("2012-10-27 23:15")  # 00:00 summer time starts at 23:00 UTC


def test_full_year_gives_a_regular_utc_grid():
    out = uci_local_to_utc_end(uci_like("2013-01-01 00:00", "2013-12-31 23:45"))
    assert len(out) == 365 * 96
    assert out.index[0] == pd.Timestamp("2013-01-01 00:15") and out.index[-1] == pd.Timestamp("2014-01-01 00:00")


def test_grid_has_no_february_29_and_splits_do_not_overlap():
    stamps = scenario_grid("2012-01-01", "2015-01-01")
    assert len(stamps) == 3 * 365 * 288
    begin = stamps - pd.Timedelta(minutes=5)
    assert not ((begin.month == 2) & (begin.day == 29)).any()
    assert pd.Timestamp("2012-02-29 00:00") in stamps      # this step ends February 28, so it stays
    assert pd.Timestamp("2012-03-01 00:00") not in stamps  # this step ends February 29
    labels = split_labels(stamps, TIME)
    assert set(labels) == {0, 1, 2}
    assert (np.diff(labels) >= 0).all()  # strictly chronological: train, then val, then test
    assert stamps[labels == 0].max() < stamps[labels == 1].min()
    assert stamps[labels == 1].max() < stamps[labels == 2].min()
    assert (labels == 0).sum() == (365 + 181) * 288 and (labels == 1).sum() == 184 * 288 and (labels == 2).sum() == 365 * 288


def test_solar_mapping_uses_the_decided_years():
    stamps = pd.DatetimeIndex(["2012-06-15 12:05", "2013-06-30 12:00", "2013-07-01 12:00", "2014-06-15 12:05"])
    source = solar_source_stamps(stamps, split_labels(stamps, TIME), TIME["solar_year_offset"])
    expected = ["2014-06-15 11:05", "2015-06-30 11:00", "2016-07-01 11:00", "2017-06-15 11:05"]
    assert list(source) == [pd.Timestamp(x) for x in expected]
    # The step that ends February 28 of a leap year must map cleanly onto a year without February 29.
    edge = pd.DatetimeIndex(["2012-02-29 00:00"])
    assert list(solar_source_stamps(edge, split_labels(edge, TIME), TIME["solar_year_offset"])) == [pd.Timestamp("2014-02-28 23:00")]


def test_groups_are_a_seeded_partition():
    names = [f"c{i}" for i in range(50)]
    a = assign_groups(names, 7, np.random.default_rng(0))
    b = assign_groups(names, 7, np.random.default_rng(0))
    assert a == b
    assert sorted(x for g in a for x in g) == sorted(names)
    assert {len(g) for g in a} <= {7, 8}
    assert a != assign_groups(names, 7, np.random.default_rng(1))


def test_hold_gives_each_15_minute_value_to_its_own_steps():
    index = pd.date_range("2012-01-01 00:15", periods=2, freq="15min")
    table = pd.DataFrame({"x": [1.0, 2.0]}, index=index)
    stamps = pd.date_range("2012-01-01 00:05", periods=6, freq="5min")
    assert hold_to_steps(table, stamps)[:, 0].tolist() == [1, 1, 1, 2, 2, 2]


def load_example():
    rng = np.random.default_rng(3)
    group_kw = rng.uniform(10, 30, (200, 4))
    return group_kw, np.arange(200) < 100, np.array([.1, .2, .3, .4]), np.array([.05, .1, .1, .3])


def test_load_scale_sets_the_training_peak_of_the_feeder_total():
    group_kw, train, p, q = load_example()
    out_p, out_q = bus_load_profiles(group_kw, train, p, q)
    assert out_p[train].sum(axis=1).max() == pytest.approx(p.sum())
    assert out_q / out_p == pytest.approx(np.broadcast_to(q / p, out_p.shape))  # constant Q/P of each bus


def test_load_scale_does_not_depend_on_later_data():
    group_kw, train, p, q = load_example()
    before = bus_load_profiles(group_kw, train, p, q)[0]
    changed = group_kw.copy()
    changed[~train] *= 100  # a huge change after the training part
    after = bus_load_profiles(changed, train, p, q)[0]
    assert after[train] == pytest.approx(before[train])
    assert after[~train] == pytest.approx(100 * before[~train])


def test_pv_bus_power_uses_only_the_systems_reporting():
    values = np.array([[500.0, 1000.0], [500.0, np.nan], [np.nan, np.nan]])
    capacity = np.array([1000.0, 2000.0])
    power, reporting = bus_pv_profiles(values, capacity, [np.array([0, 1])], np.array([0.065]))
    assert power[0, 0] == pytest.approx(0.5 * 0.065)  # 1500 W of 3000 Wp
    assert power[1, 0] == pytest.approx(0.5 * 0.065)  # a missing system does not look like a cloud
    assert np.isnan(power[2, 0])                      # no system: missing, not zero
    assert reporting[:, 0].tolist() == [2, 1, 0]


def test_ramp_formula_and_breaks():
    stamps = pd.DatetimeIndex(["2012-01-01 00:05", "2012-01-01 00:10", "2012-01-01 00:15", "2012-01-01 00:25",
                               "2012-01-01 00:30", "2012-01-01 00:35"])
    labels = np.array([0, 0, 0, 0, 0, 1], dtype=np.int8)
    power = np.array([1.0, 2.0, 1.5, 3.0, np.nan, 4.0])
    r = ramp_rate(power, stamps, labels, 5)
    assert np.isnan(r[0])                      # no previous step
    assert r[1] == pytest.approx(1.0 / 5)      # (P_t - P_{t-1}) / dt
    assert r[2] == pytest.approx(-0.5 / 5)     # a falling ramp is negative
    assert np.isnan(r[3])                      # gap in the clock
    assert np.isnan(r[4])                      # missing value
    assert np.isnan(r[5])                      # first step of a new split


def test_extreme_threshold_comes_from_training_only():
    r = np.r_[np.arange(1000.0), np.full(1000, 1e9)]
    train = np.arange(2000) < 1000
    threshold = extreme_threshold(r, train, 1.0)
    assert threshold == pytest.approx(np.percentile(np.arange(1000.0), 99))
    r2 = r.copy()
    r2[~train] = -5e12
    assert extreme_threshold(r2, train, 1.0) == threshold
    assert extreme_threshold(-r, train, 1.0) == threshold  # uses |r_t|, so falling and rising ramps count equally


def test_windows_stay_inside_a_split_and_avoid_missing_and_clock_gaps():
    stamps = pd.date_range("2012-01-01 00:05", periods=12, freq="5min").delete([7])  # a gap in the clock before index 7
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8)
    valid = np.ones(11, dtype=bool)
    valid[1] = False
    starts = window_starts(valid, labels, 0, 3, stamps)
    # steps 2..6 are one stretch, steps 7, 8 are the next (too short), and step 1 is missing
    assert starts.tolist() == [2, 3, 4]
    assert window_starts(valid, labels, 1, 3, stamps).tolist() == []  # val has only 2 steps


def test_any_in_next_steps_looks_forward_only_and_stops_at_clock_gaps():
    stamps = pd.date_range("2012-01-01 00:05", periods=9, freq="5min").delete([6])  # a gap before index 6
    flags = np.array([0, 0, 0, 1, 0, 0, 0, 1], dtype=bool)
    out = any_in_next_steps(flags, stamps, 3, 5)
    # Steps 1, 2, 3 reach the flag at 3. Step 0 is too far, and step 4 is after it. Step 5 cannot cross the gap.
    assert out.tolist() == [False, True, True, True, False, False, True, True]


def test_sign_convention_discharge_reduces_net_consumption_and_raises_voltage():
    assert net_consumption(1.0, 0.25, 0.5) == 0.25
    net = load_case33bw()
    pl = torch.as_tensor(net.p_load, dtype=torch.float64).unsqueeze(0)
    ql = torch.as_tensor(net.q_load, dtype=torch.float64).unsqueeze(0)
    bess = torch.zeros_like(pl)
    bess[0, 17] = 0.05
    solver = net.make_solver()
    idle = solver(pl, ql, torch.zeros_like(pl))
    discharging = solver(pl, ql, torch.zeros_like(pl), p_bess=bess)
    assert discharging["v"][0, 17] > idle["v"][0, 17]
    assert discharging["P"][0, 1] < idle["P"][0, 1]  # the substation supplies less power


def small_inputs():
    """Artificial inputs for the whole pipeline, with values that are obviously not real."""
    rng = np.random.default_rng(11)
    index = pd.date_range("2012-01-01 00:15", "2015-01-01 00:00", freq="15min")
    load = pd.DataFrame(rng.uniform(1, 5, (len(index), 70)), index=index, columns=[f"c{i}" for i in range(70)])
    load["silent"] = 0.0  # a client that never reports in training has to be dropped
    pv_stamps = pd.date_range("2013-12-31 23:05", "2017-12-31 23:00", freq="5min")
    hours = np.asarray(pv_stamps.hour + pv_stamps.minute / 60)
    shape = np.clip(np.sin((hours - 5) / 14 * np.pi), 0, None)[:, None]
    pv = (shape * rng.uniform(0.2, 1.0, (len(pv_stamps), 24)) * 1000).astype(np.float32)
    pv[(pv_stamps >= "2017-03-01") & (pv_stamps < "2017-03-05")] = np.nan  # a gap in the test part
    return load, pv_stamps, pv, [f"s{i}" for i in range(24)], np.full(24, 1000.0)


CONFIG = {"seed": 0, "s_base_mva": 10.0, "time": TIME,
          "selection": {"load_max_zero_share_in_train": 0.01, "pv_min_availability_in_train": 0.9,
                        "pv_min_systems_per_bus": 5},
          "placement": {"pv_buses": [14, 17, 29, 32], "pv_total_mw": 2.6},
          "ramp": {"extreme_top_percent": 1.0}}


@pytest.fixture(scope="module")
def built():
    net = load_case33bw()
    return build_dataset(*small_inputs(), CONFIG, net.p_load, net.q_load), net


def test_pipeline_applies_the_rules(built):
    d, net = built
    train = d["labels"] == 0
    assert "silent" not in d["clients"]
    assert d["p_load"][train].sum(axis=1).max() == pytest.approx(net.p_load.sum())
    assert (d["p_load"][:, 0] == 0).all()
    missing = (d["source_stamps"] >= "2017-03-01") & (d["source_stamps"] < "2017-03-05")
    assert not d["valid"][missing].any() and np.isnan(d["p_solar"][missing]).all()  # the gap stays a gap
    assert not d["extreme"][missing].any()
    assert d["threshold"] == pytest.approx(np.nanpercentile(np.abs(d["r_mw_per_min"][train]), 99))
    assert np.nanmax(d["p_solar"][:, [14, 17, 29, 32]]) <= 0.065 + 1e-9  # 2.6 MW over 4 buses, and the input is <= 1 W/Wp
    other = np.delete(np.arange(33), [14, 17, 29, 32])
    assert np.nan_to_num(d["p_solar"][:, other]).max() == 0  # PV only at the chosen buses


def test_pipeline_is_repeatable_and_training_part_ignores_test_values(built):
    d, net = built
    load, pv_stamps, pv, names, capacity = small_inputs()
    again = build_dataset(load, pv_stamps, pv, names, capacity, CONFIG, net.p_load, net.q_load)
    assert again["threshold"] == d["threshold"]
    assert np.array_equal(again["p_load"], d["p_load"])

    # Change only the data of the test part: nothing derived from training may move.
    changed_load = load.copy()
    changed_load.loc[changed_load.index > "2014-01-01"] *= 50
    changed_pv = pv.copy()
    changed_pv[pv_stamps > "2017-01-01"] *= 0.1
    other = build_dataset(changed_load, pv_stamps, changed_pv, names, capacity, CONFIG, net.p_load, net.q_load)
    train = d["labels"] == 0
    assert other["threshold"] == d["threshold"]
    assert other["clients"] == d["clients"]
    assert np.array_equal(other["p_load"][train], d["p_load"][train])
