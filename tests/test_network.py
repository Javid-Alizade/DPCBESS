"""M1: 33-bus loader and validation of the DistFlow solver against pandapower."""
import copy

import numpy as np
import pytest
import torch

from dpc_grid import load_case33bw, network_from_pandapower, run_pandapower

TARGET = 1e-3  # pu, from docs/decisions.md


@pytest.fixture(scope="module")
def net():
    return load_case33bw()


def solve(net, dtype=torch.float32, **kw):
    """Our solver for a single operating point, given (N,) numpy arrays."""
    t = {k: torch.as_tensor(v, dtype=dtype).unsqueeze(0) for k, v in kw.items()}
    return net.make_solver()(t["p_load"], t["q_load"], t.get("p_solar", torch.zeros_like(t["p_load"])),
                             p_bess=t.get("p_bess"), q_bess=t.get("q_bess"))


def test_loader_structure(net):
    assert net.n_bus == 33
    assert net.parent[0] == -1 and all(net.parent[j] < j for j in range(1, 33))
    assert net.parent[18] == 1 and net.parent[22] == 2 and net.parent[25] == 5  # the three laterals


def test_loader_per_unit_values(net):
    assert net.z_base_ohm == pytest.approx(12.66 ** 2 / 10)
    assert net.r[1] == pytest.approx(0.0922 / net.z_base_ohm)
    assert net.x[17] == pytest.approx(0.5740 / net.z_base_ohm)
    assert net.p_load.sum() * net.s_base_mva == pytest.approx(3.715)
    assert net.q_load.sum() * net.s_base_mva == pytest.approx(2.3)
    assert net.p_load[0] == 0 and net.r[0] == 0


def test_loader_rejects_meshed_network(net):
    meshed = copy.deepcopy(net.pp_net)
    meshed.line.loc[32, "in_service"] = True  # close a tie line, which creates a loop
    with pytest.raises(ValueError, match="not radial"):
        network_from_pandapower(meshed)


def test_matches_pandapower_nominal(net):
    case = dict(p_load=net.p_load, q_load=net.q_load)
    ref = run_pandapower(net, **case)
    out = solve(net, **case)
    assert np.abs(out["v"][0].numpy() - ref["v"]).max() < TARGET
    assert abs(out["loss"][0].item() - ref["loss"]) < 1e-5
    assert abs(out["P"][0, 1].item() - ref["p_slack"]) < 1e-5  # bus 1 is the only child of the slack
    assert abs(out["Q"][0, 1].item() - ref["q_slack"]) < 1e-5


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_matches_pandapower_with_bess(net, sign):
    """A wrong BESS sign in either direction would fail here."""
    p_bess, q_bess = np.zeros((2, 33))
    p_bess[17], q_bess[17] = sign * 0.05, sign * 0.02  # 0.5 MW, 0.2 Mvar
    case = dict(p_load=net.p_load, q_load=net.q_load, p_bess=p_bess, q_bess=q_bess)
    assert np.abs(solve(net, **case)["v"][0].numpy() - run_pandapower(net, **case)["v"]).max() < TARGET


def test_matches_pandapower_reverse_flow(net):
    p_solar = np.zeros(33)
    p_solar[[17, 24, 32]] = 0.15, 0.10, 0.15
    case = dict(p_load=0.3 * net.p_load, q_load=0.3 * net.q_load, p_solar=p_solar)
    ref = run_pandapower(net, **case)
    assert ref["p_slack"] < 0  # confirms this case really exports power to the substation
    assert np.abs(solve(net, **case)["v"][0].numpy() - ref["v"]).max() < TARGET


def test_gradient_matches_pandapower_finite_difference(net):
    """dV/dP_bess from autograd must match the sensitivity of the reference AC power flow."""
    bus, h = 17, 1e-3
    def v_ref(p):
        p_bess = np.zeros(33)
        p_bess[bus] = p
        return run_pandapower(net, net.p_load, net.q_load, p_bess=p_bess)["v"][bus]
    fd = (v_ref(h) - v_ref(-h)) / (2 * h)

    p_bess = torch.zeros(1, 33, dtype=torch.float64, requires_grad=True)
    pl = torch.as_tensor(net.p_load, dtype=torch.float64).unsqueeze(0)
    ql = torch.as_tensor(net.q_load, dtype=torch.float64).unsqueeze(0)
    out = net.make_solver()(pl, ql, torch.zeros_like(pl), p_bess=p_bess)
    out["v"][0, bus].backward()
    assert p_bess.grad[0, bus].item() == pytest.approx(fd, rel=1e-3)
    assert fd > 0  # discharge raises the voltage
