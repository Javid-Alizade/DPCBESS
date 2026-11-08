"""Differentiable DistFlow (branch-flow) solver for radial feeders.

Conventions (per-unit, one time step):
  * Bus 0 is the slack/substation. Buses are numbered in topological order:
    parent[j] < j for every j >= 1. Line j connects parent[j] -> j.
  * p_net[j] = load[j] - solar[j] - P_bess[j]   (net CONSUMPTION at bus j)
    BESS sign: P_bess > 0 = discharging (injects power, so it REDUCES net consumption).
  * Sending-end flows (Baran-Wu / Farivar-Low branch flow model):
      P_j = p_net[j] + r_j * I2_j + sum_{k in children(j)} P_k
      Q_j = q_net[j] + x_j * I2_j + sum_{k in children(j)} Q_k
      I2_j = (P_j^2 + Q_j^2) / V2_parent
      V2_j = V2_parent - 2 (r_j P_j + x_j Q_j) + (r_j^2 + x_j^2) I2_j
    where V2 = |V|^2 and P_j, Q_j are the flows entering line j from its parent.
Solved by fixed-point backward/forward sweep, written with one matrix instead of Python loops over buses:
  subtree[j, k] = 1 if bus k is fed through line j (k is j or a descendant of j).
  Backward sweep: the flow into line j is the sum of (p_net + r I2) over the buses in its subtree.
  Forward sweep:  V2 at bus j is V2 at the slack minus the voltage drops of all lines on the path to j,
                  and line k is on the path to bus j exactly when subtree[k, j] = 1.
Every operation is differentiable, and no tensor that requires grad is written in place.
"""
import torch
import torch.nn as nn


class RadialDistFlow(nn.Module):
    def __init__(self, parent, r, x, v_slack=1.0, n_iter=15):
        super().__init__()
        self.parent = list(parent)
        self.n = len(self.parent)
        assert self.parent[0] == -1 and all(self.parent[j] < j for j in range(1, self.n)), \
            "buses must be in topological order, bus 0 = slack"
        self.children = [[k for k in range(self.n) if self.parent[k] == j] for j in range(self.n)]
        subtree = torch.zeros(self.n, self.n, dtype=torch.float64)
        for k in range(self.n - 1, 0, -1):  # children come after parents, so walk from the last bus up to the root
            subtree[k, k] = 1.0
            for child in self.children[k]:
                subtree[k] += subtree[child]
        self.register_buffer("subtree", subtree)  # the row and column of the slack stay zero
        self.register_buffer("parent_index", torch.as_tensor([max(p, 0) for p in self.parent]))
        self.register_buffer("r", torch.as_tensor(r, dtype=torch.float64))
        self.register_buffer("x", torch.as_tensor(x, dtype=torch.float64))
        self.v_slack = v_slack
        self.n_iter = n_iter

    def forward(self, p_load, q_load, p_solar, q_solar=None, p_bess=None, q_bess=None):
        """All inputs: (B, N) per-unit tensors (use zeros where absent).
        Returns dict: v (B,N) magnitudes, v2, P, Q, I2 (flows per line, index = child bus), loss (B,)."""
        zeros = torch.zeros_like(p_load)
        q_solar = zeros if q_solar is None else q_solar
        p_bess = zeros if p_bess is None else p_bess
        q_bess = zeros if q_bess is None else q_bess
        p_net = p_load - p_solar - p_bess
        q_net = q_load - q_solar - q_bess
        r, x = self.r.to(p_net.dtype), self.x.to(p_net.dtype)
        subtree = self.subtree.to(p_net.dtype)
        z2 = r ** 2 + x ** 2
        slack = self.v_slack ** 2
        I2 = torch.zeros_like(p_net)
        v2 = torch.full_like(p_net, slack)
        for _ in range(self.n_iter):
            P = (p_net + r * I2) @ subtree.T  # backward sweep
            Q = (q_net + x * I2) @ subtree.T
            I2 = (P ** 2 + Q ** 2) / torch.clamp(v2[:, self.parent_index], min=1e-6)
            v2 = slack - (2 * (r * P + x * Q) - z2 * I2) @ subtree  # forward sweep
        loss = (r * I2).sum(dim=1)
        return {"v": torch.sqrt(torch.clamp(v2, min=1e-8)), "v2": v2, "P": P, "Q": Q, "I2": I2, "loss": loss}


def bess_soc_step(soc, p_bess, e_nom, dt, eta_ch=0.95, eta_dis=0.95):
    """Signed-power SoC update (p_bess > 0 discharge). Same units for p_bess*dt and e_nom."""
    p_dis = torch.clamp(p_bess, min=0.0)
    p_ch = torch.clamp(-p_bess, min=0.0)
    return soc + (eta_ch * p_ch - p_dis / eta_dis) * dt / e_nom
