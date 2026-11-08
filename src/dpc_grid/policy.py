import torch
import torch.nn as nn


class DPCPolicy(nn.Module):
    """u_raw = pi_theta(x_t, d_t). Raw output; feasibility is enforced by project_to_feasible_set."""

    def __init__(self, state_dim, dist_dim, control_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + dist_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, control_dim),
        )

    def forward(self, x, d):
        return self.net(torch.cat([x, d], dim=-1))
