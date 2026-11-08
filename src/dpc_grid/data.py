"""Milestone M2: turn the raw solar and load datasets into time series for the 33-bus network.

Time convention: every final timestamp is the END of the interval it describes, so the value stamped 12:05 is the
average over (12:00, 12:05]. A controller acting at 12:05 therefore never uses information from the future.

The clock of the final dataset is "scenario time". It has no daylight-saving changes:
    load:  UCI Portuguese clock converted to UTC (Portuguese standard time is UTC+0)
    solar: Utrecht UTC + 1 hour (Dutch standard time), moved back by a whole number of years
Lisbon (about 9 W) sees solar noon near 12:36 UTC and Utrecht (about 5 E) near 12:40 UTC+1, so the two clocks match
to within a few minutes on average. Solar and load do not share real weather, and this must always be stated.

Nothing here fills a gap in the solar data. A step without enough systems is marked invalid and windows skip it.
All thresholds and scales are computed on the training part only.
"""
import zipfile

import numpy as np
import pandas as pd

UCI_TXT = "LD2011_2014.txt"
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------- raw solar
def aggregate_pv_to_steps(csv_path, start_utc, end_utc, step_min=5, min_samples=3, chunk_rows=200_000):
    """Average the 1-minute PV table into steps of `step_min` minutes, per system.

    A step is NaN if fewer than `min_samples` valid minutes exist in it. The file is read in pieces because it is 3 GB.
    Returns (stamps, values [W], system_names). Stamps are the end of each step, as naive UTC.
    """
    start, end = pd.Timestamp(start_utc), pd.Timestamp(end_utc)
    n_steps = int((end - start) / pd.Timedelta(minutes=step_min))
    names = pd.read_csv(csv_path, nrows=0).columns[1:].tolist()
    total = np.zeros((n_steps, len(names)))
    count = np.zeros((n_steps, len(names)), dtype=np.int16)
    for chunk in pd.read_csv(csv_path, chunksize=chunk_rows):
        stamps = pd.to_datetime(chunk.iloc[:, 0], utc=True).dt.tz_localize(None)
        minutes = ((stamps - start) / pd.Timedelta(minutes=1)).to_numpy()
        keep = (minutes > 0) & (minutes <= n_steps * step_min)
        if not keep.any():
            continue
        steps = np.ceil(minutes[keep] / step_min).astype(int) - 1
        values = chunk.iloc[:, 1:].to_numpy(dtype=float)[keep]
        np.add.at(total, steps, np.nan_to_num(values))
        np.add.at(count, steps, ~np.isnan(values))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    mean[count < min_samples] = np.nan
    stamps = start + pd.to_timedelta(step_min * np.arange(1, n_steps + 1), unit="min")
    return stamps, mean, names


# ---------------------------------------------------------------- raw load
def read_uci_load(zip_path, first=None, last=None):
    """Read the UCI table from its zip: rows = local Portuguese time, columns = clients, values = kW."""
    with zipfile.ZipFile(zip_path) as archive, archive.open(UCI_TXT) as f:
        table = pd.read_csv(f, sep=";", decimal=",", index_col=0, parse_dates=True)
    return table.loc[first:last].astype(float)


def _last_sunday(year, month):
    """Last Sunday of March or October (both months have 31 days)."""
    day = pd.Timestamp(year=year, month=month, day=31)
    return day - pd.Timedelta(days=(day.weekday() + 1) % 7)


def uci_local_to_utc_end(table):
    """Convert the UCI clock to a regular UTC grid labelled by the END of each 15-minute interval.

    Measured on the training years: the abnormal hour of the clock-change days carries the labels 01:00-01:45, which
    are the starts of the intervals because Portugal changes the clock at 01:00. So labels are treated as starts.
      March:   the hour does not exist, and the file holds near-zero values in it. Those 4 rows are removed.
      October: the hour happens twice, and the file holds the sum of both. Each row becomes two rows of half the value.
    """
    idx = table.index
    utc = idx.to_series(index=idx)
    drop = np.zeros(len(idx), dtype=bool)
    twice = np.zeros(len(idx), dtype=bool)
    for year in sorted(set(idx.year)):
        spring = _last_sunday(year, 3) + pd.Timedelta(hours=1)
        autumn = _last_sunday(year, 10) + pd.Timedelta(hours=1)
        hour = pd.Timedelta(hours=1)
        drop |= (idx >= spring) & (idx < spring + hour)
        twice |= (idx >= autumn) & (idx < autumn + hour)
        summer = (idx >= spring + hour) & (idx < autumn)
        utc[summer] = idx[summer] - hour  # summer time is UTC+1
    normal = ~drop & ~twice
    parts = [table[normal].set_axis(pd.DatetimeIndex(utc[normal])),
             (table[twice] / 2).set_axis(idx[twice] - pd.Timedelta(hours=1)),  # first pass, still summer time
             (table[twice] / 2).set_axis(idx[twice])]                          # second pass, winter time
    result = pd.concat(parts).sort_index()
    result.index = result.index + pd.Timedelta(minutes=15)
    if not (np.diff(result.index.values) == np.timedelta64(15, "m")).all():
        raise ValueError("the UTC grid is not regular after the clock-change correction")
    return result


# ---------------------------------------------------------------- scenario time
def scenario_grid(first_day, last_day_exclusive, step_min=5):
    """End-labelled steps of the scenario clock. February 29 is removed, because the mapped solar year may not have it."""
    stamps = pd.date_range(pd.Timestamp(first_day) + pd.Timedelta(minutes=step_min),
                           pd.Timestamp(last_day_exclusive), freq=f"{step_min}min")
    interval_start = stamps - pd.Timedelta(minutes=step_min)
    return stamps[~((interval_start.month == 2) & (interval_start.day == 29))]


def split_labels(stamps, time_config):
    """0 = train, 1 = val, 2 = test, by the end-labelled stamps. Ranges are half open, so they cannot overlap."""
    labels = np.full(len(stamps), -1, dtype=np.int8)
    for k, name in enumerate(SPLITS):
        start, end = pd.Timestamp(time_config[name][0]), pd.Timestamp(time_config[name][1])
        labels[(stamps > start) & (stamps <= end)] = k
    return labels


def solar_source_stamps(stamps, labels, year_offset, step_min=5):
    """The Utrecht UTC stamp used for each scenario stamp: same calendar date in the mapped year, minus 1 hour.

    The year is moved using the START of each interval. The stamp 2012-02-29 00:00 ends February 28, and moving the
    stamp itself would ask for a February 29 that the mapped year may not have.
    """
    step = pd.Timedelta(minutes=step_min)
    begin = stamps - step
    offsets = np.array([year_offset[name] for name in SPLITS])[labels]
    shifted = pd.to_datetime(pd.DataFrame({"year": begin.year + offsets, "month": begin.month, "day": begin.day,
                                           "hour": begin.hour, "minute": begin.minute}))
    return pd.DatetimeIndex(shifted) + step - pd.Timedelta(hours=1)


# ---------------------------------------------------------------- mapping to buses
def assign_groups(names, n_groups, rng):
    """Random partition of systems or clients into groups of nearly equal size."""
    shuffled = rng.permutation(len(names))
    return [sorted(np.asarray(names)[part].tolist()) for part in np.array_split(shuffled, n_groups)]


def hold_to_steps(table_15min, stamps):
    """Repeat each 15-minute value over its 5-minute steps. The data have no information inside the 15 minutes."""
    return table_15min.reindex(stamps.ceil("15min")).to_numpy()


def bus_load_profiles(group_kw, train, p_nominal, q_nominal):
    """Per-unit bus load at scale x1.0 from the summed client power of each bus group.

    group_kw: (T, n_load_buses) raw kW. p_nominal, q_nominal: the case33bw values of those buses.
    Each group is divided by its training mean, so the average spatial pattern follows case33bw.
    One common factor then sets the feeder total peak ON THE TRAINING PART to the case33bw total.
    Q follows P with the constant Q/P ratio of each bus, because the datasets contain no reactive power.
    """
    shape = group_kw / group_kw[train].mean(axis=0)
    total = (shape * p_nominal).sum(axis=1)
    p = shape * p_nominal * (p_nominal.sum() / total[train].max())
    return p, p * (q_nominal / p_nominal)


def bus_pv_profiles(values_w, capacity_w, groups, bus_capacity_pu):
    """Per-unit PV power of each PV bus: measured power / capacity of the systems reporting x bus capacity.

    values_w: (T, n_systems). groups: one array of column indices per PV bus.
    Returns (T, n_pv_buses) power and (T, n_pv_buses) the number of systems reporting.
    """
    power = np.zeros((len(values_w), len(groups)))
    reporting = np.zeros((len(values_w), len(groups)), dtype=np.int16)
    for b, columns in enumerate(groups):
        ok = ~np.isnan(values_w[:, columns])
        reporting[:, b] = ok.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            power[:, b] = (np.nansum(values_w[:, columns], axis=1) / (ok * capacity_w[columns]).sum(axis=1)
                           * bus_capacity_pu[b])
    return power, reporting


# ---------------------------------------------------------------- ramps and windows
def ramp_rate(power, stamps, labels, dt):
    """r_t = (P_t - P_{t-1}) / dt. NaN at the first step, at a gap in the clock, at a new split, and next to missing data."""
    r = np.full(len(power), np.nan)
    r[1:] = (power[1:] - power[:-1]) / dt
    step = np.timedelta64(int(dt), "m")
    broken = (np.diff(stamps.values) != step) | (labels[1:] != labels[:-1])
    r[1:][broken] = np.nan
    return r


def extreme_threshold(r, train, top_percent):
    """Threshold of |r_t|, computed from the training part only."""
    return float(np.nanpercentile(np.abs(r[train]), 100 - top_percent))


def window_starts(valid, labels, split, horizon, stamps, step_min=5):
    """Indices t where the steps t .. t+horizon-1 are all valid, consecutive, and inside one split."""
    good = valid & (labels == split)
    steady = np.r_[True, np.diff(stamps.values) == np.timedelta64(step_min, "m")]
    run = 0
    starts = []
    for t in range(len(good) - 1, -1, -1):  # walk backwards and count how long the good stretch ahead is
        run = run + 1 if good[t] else 0
        if t + 1 < len(good) and not steady[t + 1]:
            run = 1 if good[t] else 0  # the next step belongs to a different stretch of the clock
        if run >= horizon:
            starts.append(t)
    return np.array(starts[::-1], dtype=np.int64)


def any_in_next_steps(flags, stamps, n, step_min=5):
    """True at t if `flags` holds at any of the steps t .. t+n-1, without crossing a gap in the clock."""
    out = np.zeros(len(flags), dtype=bool)
    gap = np.r_[False, np.diff(stamps.values) != np.timedelta64(step_min, "m")]
    nearest = -1  # the nearest step at or after t where flags holds
    for t in range(len(flags) - 1, -1, -1):
        if t + 1 < len(flags) and gap[t + 1]:
            nearest = -1
        if flags[t]:
            nearest = t
        out[t] = nearest != -1 and nearest < t + n
    return out


def net_consumption(p_load, p_solar, p_bess):
    """docs/decisions.md sign convention: P_bess > 0 is discharge, and discharge reduces net consumption."""
    return p_load - p_solar - p_bess


# ---------------------------------------------------------------- the whole pipeline
def build_dataset(load_15min, pv_stamps, pv_values_w, pv_names, pv_capacity_w, config, p_load, q_load):
    """Build the dataset from cleaned inputs. The same inputs and seed always give the same output.

    load_15min: end-labelled UTC table from uci_local_to_utc_end. pv_*: 5-minute Utrecht data (naive UTC).
    p_load, q_load: per-unit case33bw values for all buses, with bus 0 = slack.
    """
    time_cfg, select, place = config["time"], config["selection"], config["placement"]
    rng = np.random.default_rng(config["seed"])
    dt = time_cfg["dt_min"]
    stamps = scenario_grid(time_cfg["train"][0], time_cfg["test"][1], dt)
    labels = split_labels(stamps, time_cfg)
    train = labels == 0

    # Load: choose clients on the training part only, then split them over the load buses.
    train_15 = (load_15min.index > pd.Timestamp(time_cfg["train"][0])) & (load_15min.index <= pd.Timestamp(time_cfg["train"][1]))
    zero_share = (load_15min[train_15] == 0).mean()
    clients = zero_share.index[zero_share <= select["load_max_zero_share_in_train"]].tolist()
    load_buses = np.flatnonzero(p_load > 0)
    load_groups = assign_groups(clients, len(load_buses), rng)
    group_table = pd.DataFrame({i: load_15min[g].sum(axis=1) for i, g in enumerate(load_groups)})
    p_nom, q_nom = p_load[load_buses], q_load[load_buses]
    p_part, q_part = bus_load_profiles(hold_to_steps(group_table, stamps), train, p_nom, q_nom)
    p_out = np.zeros((len(stamps), len(p_load)))
    q_out = np.zeros((len(stamps), len(p_load)))
    p_out[:, load_buses], q_out[:, load_buses] = p_part, q_part

    # Solar: same idea. Only systems that report often enough in the training part are used.
    source = solar_source_stamps(stamps, labels, time_cfg["solar_year_offset"], dt)
    position = pd.DatetimeIndex(pv_stamps).get_indexer(source)
    values = pv_values_w[position].astype(float)
    values[position < 0] = np.nan
    availability = (~np.isnan(values[train])).mean(axis=0)
    usable = np.flatnonzero(availability >= select["pv_min_availability_in_train"])
    pv_buses = place["pv_buses"]
    pv_groups_idx = [np.sort(part) for part in np.array_split(rng.permutation(usable), len(pv_buses))]
    s_base = config["s_base_mva"]
    bus_capacity = np.full(len(pv_buses), place["pv_total_mw"] / len(pv_buses) / s_base)
    pv_part, reporting = bus_pv_profiles(values, pv_capacity_w, pv_groups_idx, bus_capacity)

    # A step counts only if every PV bus has enough systems reporting. Nothing is filled in.
    valid = (reporting >= select["pv_min_systems_per_bus"]).all(axis=1)
    p_solar = np.full((len(stamps), len(p_load)), np.nan)
    p_solar[np.ix_(valid, pv_buses)] = pv_part[valid]

    # Ramps of total feeder PV in MW per minute, and the extreme label from the training threshold.
    total_mw = np.where(valid, np.nansum(p_solar, axis=1), np.nan) * s_base
    r = ramp_rate(total_mw, stamps, labels, dt)
    threshold = extreme_threshold(r, train, config["ramp"]["extreme_top_percent"])
    extreme = np.abs(r) >= threshold  # NaN compares as False

    return {"stamps": stamps, "labels": labels, "source_stamps": source, "p_load": p_out, "q_load": q_out,
            "p_solar": p_solar, "valid": valid, "reporting": reporting, "total_pv_mw": total_mw,
            "r_mw_per_min": r, "extreme": extreme, "threshold": threshold,
            "clients": clients, "load_groups": load_groups, "load_buses": load_buses.tolist(),
            "pv_groups": [[pv_names[i] for i in g] for g in pv_groups_idx], "n_pv_usable": len(usable)}
