"""Milestone M1: validate the differentiable DistFlow solver against pandapower AC power flow.

Run:  python scripts/validate_m1.py
Prints the comparison and writes results/m1_validation.json (results + config + library versions).

PV and BESS locations below are only test points for the solver. They are not a design decision.
"""
import json
import platform
from pathlib import Path

import numpy as np
import pandapower
import torch

from dpc_grid import load_case33bw, run_pandapower

SEED = 0
N_RANDOM = 200
N_ITER = 15          # solver default
TARGET = 1e-3        # pu, from docs/decisions.md
RESULTS = Path(__file__).resolve().parents[1] / "results" / "m1_validation.json"


def injection(net, mw_by_bus):
    """(N,) per-unit array from {bus: MW or Mvar}."""
    a = np.zeros(net.n_bus)
    for bus, mw in mw_by_bus.items():
        a[bus] = mw / net.s_base_mva
    return a


def named_scenarios(net):
    """Fixed operating points, from nominal to reverse power flow."""
    pl, ql = net.p_load, net.q_load
    return {
        "nominal load": dict(p_load=pl, q_load=ql),
        "load x0.5": dict(p_load=0.5 * pl, q_load=0.5 * ql),
        "load x1.5": dict(p_load=1.5 * pl, q_load=1.5 * ql),
        "BESS discharge": dict(p_load=pl, q_load=ql, p_bess=injection(net, {17: 0.5, 32: 0.5}),
                               q_bess=injection(net, {17: 0.2, 32: -0.2})),
        "BESS charge": dict(p_load=pl, q_load=ql, p_bess=injection(net, {17: -0.5, 32: -0.5}),
                            q_bess=injection(net, {17: -0.2, 32: 0.2})),
        "PV + BESS": dict(p_load=pl, q_load=ql, p_solar=injection(net, {13: 0.5, 24: 0.8, 29: 0.8}),
                          p_bess=injection(net, {17: 0.3}), q_bess=injection(net, {17: 0.1})),
        "reverse flow (PV > load)": dict(p_load=0.3 * pl, q_load=0.3 * ql,
                                         p_solar=injection(net, {17: 1.5, 24: 1.0, 32: 1.5})),
    }


def random_scenarios(net, n, rng):
    """Random load levels with PV at 5 buses and BESS (charge or discharge) at 3 buses."""
    out = []
    for _ in range(n):
        p_solar, p_bess, q_bess = np.zeros((3, net.n_bus))
        scale = rng.uniform(0.3, 1.5, net.n_bus)
        buses = rng.choice(np.arange(1, net.n_bus), size=8, replace=False)
        p_solar[buses[:5]] = rng.uniform(0, 0.8, 5) / net.s_base_mva
        p_bess[buses[5:]] = rng.uniform(-0.5, 0.5, 3) / net.s_base_mva
        q_bess[buses[5:]] = rng.uniform(-0.25, 0.25, 3) / net.s_base_mva
        out.append(dict(p_load=scale * net.p_load, q_load=scale * net.q_load,
                        p_solar=p_solar, p_bess=p_bess, q_bess=q_bess))
    return out


def solve_batch(solver, scenarios, dtype):
    """Run our solver on a list of scenarios as one batch."""
    zeros = np.zeros_like(scenarios[0]["p_load"])
    def stack(key):
        return torch.as_tensor(np.stack([s.get(key, zeros) for s in scenarios]), dtype=dtype)
    with torch.no_grad():
        return solver(stack("p_load"), stack("q_load"), stack("p_solar"),
                      p_bess=stack("p_bess"), q_bess=stack("q_bess"))


def main():
    torch.manual_seed(SEED)
    net = load_case33bw()
    solver = net.make_solver(n_iter=N_ITER)
    rng = np.random.default_rng(SEED)

    named = named_scenarios(net)
    randoms = random_scenarios(net, N_RANDOM, rng)
    all_scenarios = list(named.values()) + randoms

    # Reference: one pandapower AC power flow per scenario.
    ref = [run_pandapower(net, **s) for s in all_scenarios]
    v_ref = np.stack([r["v"] for r in ref])
    loss_ref = np.array([r["loss"] for r in ref])
    p_slack_ref = np.array([r["p_slack"] for r in ref])
    q_slack_ref = np.array([r["q_slack"] for r in ref])
    slack_children = solver.children[0]

    def errors(out):
        """Absolute errors of one solver output, per scenario."""
        p_slack = out["P"][:, slack_children].sum(1).double().numpy()
        q_slack = out["Q"][:, slack_children].sum(1).double().numpy()
        return {"v": np.abs(out["v"].double().numpy() - v_ref),
                "loss": np.abs(out["loss"].double().numpy() - loss_ref),
                "p_slack": np.abs(p_slack - p_slack_ref),
                "q_slack": np.abs(q_slack - q_slack_ref)}

    # float32 is what training will use. float64 shows the error without single-precision rounding.
    err32 = errors(solve_batch(solver, all_scenarios, torch.float32))
    err64 = errors(solve_batch(solver, all_scenarios, torch.float64))

    print(f"Network: {net.name}, {net.n_bus} buses, S_base = {net.s_base_mva} MVA, "
          f"V_base = {net.v_base_kv} kV, Z_base = {net.z_base_ohm:.4f} ohm")
    print(f"Total load: {net.p_load.sum() * net.s_base_mva:.3f} MW, "
          f"{net.q_load.sum() * net.s_base_mva:.3f} Mvar")
    print(f"Solver: n_iter = {N_ITER}\n")

    header = f"{'scenario':<27}{'V min':>8}{'V max':>8}{'loss kW':>10}{'max|dV| f32':>13}{'max|dV| f64':>13}"
    print(header)
    print("-" * len(header))
    per_scenario = {}
    for k, name in enumerate(named):
        row = {"v_min": float(v_ref[k].min()), "v_min_bus": int(v_ref[k].argmin()),
               "v_max": float(v_ref[k].max()), "loss_kw": float(loss_ref[k] * net.s_base_mva * 1e3),
               "max_dv_float32": float(err32["v"][k].max()), "max_dv_float64": float(err64["v"][k].max())}
        per_scenario[name] = row
        print(f"{name:<27}{row['v_min']:>8.4f}{row['v_max']:>8.4f}{row['loss_kw']:>10.2f}"
              f"{row['max_dv_float32']:>13.2e}{row['max_dv_float64']:>13.2e}")
    n_named = len(named)
    print(f"{f'{N_RANDOM} random (worst)':<27}{v_ref[n_named:].min():>8.4f}{v_ref[n_named:].max():>8.4f}"
          f"{'':>10}{err32['v'][n_named:].max():>13.2e}{err64['v'][n_named:].max():>13.2e}")

    summary = {
        "max_voltage_error_pu_float32": float(err32["v"].max()),
        "max_voltage_error_pu_float64": float(err64["v"].max()),
        "mean_voltage_error_pu_float32": float(err32["v"].mean()),
        "max_loss_error_pu_float32": float(err32["loss"].max()),
        "max_slack_p_error_pu_float32": float(err32["p_slack"].max()),
        "max_slack_q_error_pu_float32": float(err32["q_slack"].max()),
        "passes_target": bool(err32["v"].max() < TARGET),
    }
    print(f"\nAll {len(all_scenarios)} scenarios, {net.n_bus} buses each:")
    print(f"  max voltage error (float32) = {summary['max_voltage_error_pu_float32']:.3e} pu   target < {TARGET:.0e}")
    print(f"  max voltage error (float64) = {summary['max_voltage_error_pu_float64']:.3e} pu")
    print(f"  max loss error    (float32) = {summary['max_loss_error_pu_float32'] * net.s_base_mva * 1e3:.4f} kW")
    print(f"  max slack P error (float32) = {summary['max_slack_p_error_pu_float32'] * net.s_base_mva * 1e3:.4f} kW")
    print(f"  max slack Q error (float32) = {summary['max_slack_q_error_pu_float32'] * net.s_base_mva * 1e3:.4f} kvar")
    print(f"  PASS: {summary['passes_target']}")

    # How many sweeps does the solver need? This checks the default n_iter instead of assuming it.
    print("\nConvergence, max voltage error over all scenarios (float64):")
    sweep = {}
    for n_iter in [1, 2, 3, 4, 5, 8, 10, 15, 20, 30]:
        e = errors(solve_batch(net.make_solver(n_iter=n_iter), all_scenarios, torch.float64))
        sweep[n_iter] = float(e["v"].max())
        print(f"  n_iter = {n_iter:>2}: {sweep[n_iter]:.3e} pu")

    # The proposal uses a +-5% voltage band; how does the unmodified feeder sit against it?
    v_nom = v_ref[0]
    below = np.flatnonzero(v_nom < 0.95)
    print(f"\nNominal load, no PV/BESS: V min = {v_nom.min():.4f} pu at bus {v_nom.argmin()}, "
          f"{len(below)} of {net.n_bus} buses below 0.95 pu")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "summary": summary, "named_scenarios": per_scenario, "n_iter_sweep": sweep,
        "nominal_buses_below_0.95": below.tolist(),
        "config": {"seed": SEED, "n_random": N_RANDOM, "n_iter": N_ITER, "target_pu": TARGET,
                   "pandapower_tolerance_mva": 1e-9},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandapower": pandapower.__version__},
    }, indent=2))
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
