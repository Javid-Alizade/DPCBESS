import torch


def project_to_feasible_set(u, p_min, p_max, q_min, q_max, s_max, eps=1e-8):
    """Differentiable actuator-level safety operator for BESS inverter limits.

    u: (..., 2) raw [P, Q]. Clip P and Q to their boxes, then scale radially if
    sqrt(P^2+Q^2) > s_max. This guarantees |P|,|Q| box limits and the apparent-power
    limit, but it is NOT the exact Euclidean projection and does NOT guarantee
    network-wide voltage feasibility (state this in the paper).
    Note: radial scaling keeps the box limits because the scale factor is <= 1
    and the box contains the origin (p_min <= 0 <= p_max, q_min <= 0 <= q_max).
    p_min/p_max/q_min/q_max/s_max may be floats or tensors broadcastable to u[..., 0].
    """
    p = torch.clamp(u[..., 0], p_min, p_max) if not torch.is_tensor(p_min) \
        else torch.minimum(torch.maximum(u[..., 0], p_min), p_max)
    q = torch.clamp(u[..., 1], q_min, q_max) if not torch.is_tensor(q_min) \
        else torch.minimum(torch.maximum(u[..., 1], q_min), q_max)
    s = torch.sqrt(p ** 2 + q ** 2 + eps)
    scale = torch.clamp(s_max / s, max=1.0)
    return torch.stack([p * scale, q * scale], dim=-1)
