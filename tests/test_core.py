"""Run: pytest -q   (needs torch). These tests encode the corrected conventions."""
import torch
from dpc_grid import RadialDistFlow, project_to_feasible_set, bess_soc_step, DPCPolicy

PARENT = [-1, 0, 1, 2, 1]
R = [0, .01, .02, .02, .03]
X = [0, .02, .03, .03, .04]
LOAD = torch.tensor([[0, .3, .2, .4, .25]])


def net():
    return RadialDistFlow(PARENT, R, X, n_iter=30)


def test_power_balance():
    out = net()(LOAD, 0.4 * LOAD, torch.zeros_like(LOAD))
    assert abs(out["P"][0, 1] - (LOAD.sum() + out["loss"][0])) < 1e-5


def test_bess_discharge_raises_voltage():
    m = net(); pb = torch.zeros_like(LOAD); pb[0, 3] = 0.2
    v0 = m(LOAD, 0.4 * LOAD, torch.zeros_like(LOAD))["v"][0, 3]
    v1 = m(LOAD, 0.4 * LOAD, torch.zeros_like(LOAD), p_bess=pb)["v"][0, 3]
    assert v1 > v0


def test_projection_limits():
    u = torch.tensor([[5., 5.], [-5., 0.3], [0.1, 0.1]])
    s = project_to_feasible_set(u, -1., 1., -1., 1., 1.0)
    assert (s[:, 0].abs() <= 1 + 1e-6).all() and (s[:, 1].abs() <= 1 + 1e-6).all()
    assert (s.norm(dim=1) <= 1 + 1e-4).all()


def test_soc_direction():
    assert bess_soc_step(torch.tensor(.5), torch.tensor(10.), 100., 1.) < .5   # discharge
    assert bess_soc_step(torch.tensor(.5), torch.tensor(-10.), 100., 1.) > .5  # charge


def test_gradient_flows_and_training_reduces_loss():
    torch.manual_seed(0)
    m = net(); pol = DPCPolicy(state_dim=5, dist_dim=5)
    opt = torch.optim.AdamW(pol.parameters(), lr=3e-3)
    load = LOAD.repeat(16, 1) * (1 + 0.2 * torch.rand(16, 1))
    def loss_fn():
        d = load
        u = project_to_feasible_set(pol(torch.ones(16, 5), d), -.3, .3, -.3, .3, .3)
        pb = torch.zeros_like(load); qb = torch.zeros_like(load)
        pb[:, 3] = u[:, 0]; qb[:, 3] = u[:, 1]
        out = m(load, 0.4 * load, torch.zeros_like(load), p_bess=pb, q_bess=qb)
        return ((out["v"][:, 1:] - 1) ** 2).sum(1).mean() + out["loss"].mean()
    l0 = loss_fn().item()
    for _ in range(150):
        opt.zero_grad(); l = loss_fn(); l.backward(); opt.step()
    assert l.item() < l0
