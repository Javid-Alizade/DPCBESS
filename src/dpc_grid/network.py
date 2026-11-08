"""Load a radial feeder from pandapower and convert it to the per-unit arrays used by physics.py.

Data source: pandapower `case33bw` (IEEE 33-bus feeder, Baran & Wu 1989).

Conventions (same as physics.py):
  * Arrays have one entry per bus. Line values are stored at the index of the line's child bus,
    so r[j], x[j] belong to line parent[j] -> j. Entry 0 is unused (bus 0 is the slack).
  * Per-unit bases: S_base = net.sn_mva, V_base = bus vn_kv, Z_base = V_base^2 / S_base.

pandapower is imported inside the functions, so physics.py and training code do not depend on it.
"""
import copy
from dataclasses import dataclass

import numpy as np

from .physics import RadialDistFlow


@dataclass
class RadialNetwork:
    name: str
    parent: list         # parent[j] = upstream bus of bus j; parent[0] = -1
    r: np.ndarray        # (N,) line resistance [pu], index = child bus
    x: np.ndarray        # (N,) line reactance [pu], index = child bus
    p_load: np.ndarray   # (N,) nominal active load [pu]
    q_load: np.ndarray   # (N,) nominal reactive load [pu]
    s_base_mva: float
    v_base_kv: float
    z_base_ohm: float
    v_slack: float       # slack voltage magnitude [pu]
    pp_net: object       # the source pandapower network, kept for validation

    @property
    def n_bus(self):
        return len(self.parent)

    def make_solver(self, n_iter=15):
        return RadialDistFlow(self.parent, self.r, self.x, v_slack=self.v_slack, n_iter=n_iter)


def network_from_pandapower(net, name="network"):
    """Convert a pandapower network to RadialNetwork. Raises ValueError if our solver cannot represent it."""
    # Our DistFlow model has only series r + jx lines and PQ injections, so refuse anything else
    # instead of silently ignoring it.
    if len(net.trafo) or len(net.shunt) or len(net.gen) or len(net.ext_grid) != 1:
        raise ValueError("expected one ext_grid and no transformers, shunts or PV generators")
    if not np.array_equal(net.bus.index.to_numpy(), np.arange(len(net.bus))):
        raise ValueError("bus indices must be 0..N-1")
    if net.ext_grid.bus.iloc[0] != 0:
        raise ValueError("bus 0 must be the slack")
    if net.bus.vn_kv.nunique() != 1:
        raise ValueError("expected one voltage level")
    if net.load.bus.duplicated().any():
        raise ValueError("expected at most one load per bus")

    lines = net.line[net.line.in_service]  # case33bw has 5 open tie lines; these are skipped
    if (lines.c_nf_per_km != 0).any() or (lines.g_us_per_km != 0).any():
        raise ValueError("line shunt admittance is not modelled")
    n = len(net.bus)
    if len(lines) != n - 1:
        raise ValueError(f"not radial: {n} buses need {n - 1} closed lines, found {len(lines)}")

    s_base = float(net.sn_mva)
    v_base = float(net.bus.vn_kv.iloc[0])
    z_base = v_base ** 2 / s_base

    # Neighbours of each bus, with the line impedance in ohm.
    neighbours = [[] for _ in range(n)]
    for line in lines.itertuples():
        r_ohm = line.r_ohm_per_km * line.length_km / line.parallel
        x_ohm = line.x_ohm_per_km * line.length_km / line.parallel
        neighbours[line.from_bus].append((line.to_bus, r_ohm, x_ohm))
        neighbours[line.to_bus].append((line.from_bus, r_ohm, x_ohm))

    # Walk away from the slack to orient each line parent -> child.
    parent = [None] * n
    parent[0] = -1
    r, x = np.zeros(n), np.zeros(n)
    stack = [0]
    while stack:
        i = stack.pop()
        for j, r_ohm, x_ohm in neighbours[i]:
            if parent[j] is None:
                parent[j] = i
                r[j], x[j] = r_ohm / z_base, x_ohm / z_base
                stack.append(j)
    if any(p is None for p in parent):
        raise ValueError("not radial: some buses are not connected to the slack")
    # The solver needs this order. We check it rather than renumber, so that bus numbers stay
    # identical to pandapower and to the published network.
    if any(parent[j] >= j for j in range(1, n)):
        raise ValueError("buses are not in topological order (parent[j] < j)")

    p_load, q_load = np.zeros(n), np.zeros(n)
    loads = net.load[net.load.in_service]
    p_load[loads.bus.to_numpy()] = (loads.p_mw * loads.scaling).to_numpy() / s_base
    q_load[loads.bus.to_numpy()] = (loads.q_mvar * loads.scaling).to_numpy() / s_base

    return RadialNetwork(name=name, parent=parent, r=r, x=x, p_load=p_load, q_load=q_load,
                         s_base_mva=s_base, v_base_kv=v_base, z_base_ohm=z_base,
                         v_slack=float(net.ext_grid.vm_pu.iloc[0]), pp_net=net)


def load_case33bw():
    """IEEE 33-bus test feeder (Baran & Wu 1989), via pandapower."""
    import pandapower.networks as pn
    return network_from_pandapower(pn.case33bw(), name="case33bw")


def run_pandapower(network, p_load, q_load, p_solar=None, q_solar=None, p_bess=None, q_bess=None):
    """Full AC power flow (Newton-Raphson) for one operating point; the reference for validation.

    All inputs are (N,) per-unit arrays with the same meaning as RadialDistFlow.forward.
    Solar and BESS are modelled as pandapower static generators, whose sign is positive = injection.
    This matches P_bess > 0 = discharge, and the conversion is independent from our solver's
    p_net = load - solar - P_bess, so a sign error in the solver would show up as a voltage error.
    """
    import pandapower as pp
    zeros = np.zeros(network.n_bus)
    p_solar = zeros if p_solar is None else p_solar
    q_solar = zeros if q_solar is None else q_solar
    p_bess = zeros if p_bess is None else p_bess
    q_bess = zeros if q_bess is None else q_bess

    s_base = network.s_base_mva
    net = copy.deepcopy(network.pp_net)
    load_buses = net.load.bus.to_numpy()
    net.load["p_mw"] = np.asarray(p_load)[load_buses] * s_base
    net.load["q_mvar"] = np.asarray(q_load)[load_buses] * s_base
    net.load["scaling"] = 1.0
    p_inj = (np.asarray(p_solar) + np.asarray(p_bess)) * s_base
    q_inj = (np.asarray(q_solar) + np.asarray(q_bess)) * s_base
    for bus in np.flatnonzero((p_inj != 0) | (q_inj != 0)):
        pp.create_sgen(net, int(bus), p_mw=float(p_inj[bus]), q_mvar=float(q_inj[bus]))

    pp.runpp(net, algorithm="nr", init="flat", tolerance_mva=1e-9, numba=False)
    return {
        "v": net.res_bus.vm_pu.to_numpy(),
        "loss": float(np.nansum(net.res_line.pl_mw.to_numpy())) / s_base,
        "p_slack": float(net.res_ext_grid.p_mw.iloc[0]) / s_base,
        "q_slack": float(net.res_ext_grid.q_mvar.iloc[0]) / s_base,
    }
