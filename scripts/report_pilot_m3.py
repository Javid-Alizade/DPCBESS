"""Milestone M3: read one run and show its learning curve, every loss term, and what the policy actually does.

Run:  python scripts/report_pilot_m3.py results/runs/<run id>
Writes pilot_report.json and pilot.png into the run folder. Uses the training curves and the fixed validation windows only.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dpc_grid import load_case33bw
from dpc_grid.dpc import (DroopController, IdleController, M3Data, PolicyController, control_step, idle_voltage,
                          make_policy, per_unit_battery)

ROOT = Path(__file__).resolve().parents[1]
TERMS = ["voltage_penalty", "voltage", "loss", "soc_penalty", "terminal_soc", "effort", "inverter_penalty"]
COLORS = {"voltage_penalty": "#2a78d6", "voltage": "#eb6834", "loss": "#1baf7a", "soc_penalty": "#eda100",
          "terminal_soc": "#e87ba4", "effort": "#008300", "inverter_penalty": "#7d4fd0"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
REGIMES = {"base_x0.7": 0.7, "stress_x1.0": 1.0}


def behaviour(controller, data, starts, value, horizon, battery, solver, project):
    """Run the fixed validation windows and describe the actions, the voltages and the SoC."""
    t = torch.as_tensor(starts)
    scale = torch.full((len(t),), value)
    index = torch.as_tensor(data.bess_buses)
    soc = torch.full((len(t), len(data.bess_buses)), 0.5)
    action = torch.zeros(len(t), len(data.bess_buses), 2)
    with torch.no_grad():
        v_bess = idle_voltage(data, t, scale, solver)[:, index]
        actions, voltages, socs = [], [], []
        for k in range(horizon):
            out, action, soc = control_step(controller, data, t + k, scale, soc, v_bess, action, battery, solver, project)
            v_bess = out["v"][:, index]
            actions.append(action)
            voltages.append(out["v"])
            socs.append(soc)
    a = torch.stack(actions).numpy() * data.s_base  # (H, windows, batteries, 2), in MW and Mvar
    v = torch.stack(voltages).numpy()
    apparent = np.hypot(a[..., 0], a[..., 1])
    s_all = torch.stack(socs).numpy()
    broken = ((np.abs(a[..., 0]) > battery["p_max_mw"] * (1 + 1e-6)) | (np.abs(a[..., 1]) > battery["q_max_mvar"] * (1 + 1e-6))
              | (apparent > battery["s_max_mva"] * (1 + 1e-5)))
    return {"mean_P_mw_per_battery": a[..., 0].mean(axis=(0, 1)).tolist(), "std_P_mw_per_battery": a[..., 0].std(axis=(0, 1)).tolist(),
            "mean_Q_mvar_per_battery": a[..., 1].mean(axis=(0, 1)).tolist(), "std_Q_mvar_per_battery": a[..., 1].std(axis=(0, 1)).tolist(),
            "share_at_inverter_limit": float((apparent >= 0.98 * battery["s_max_pu"] * data.s_base).mean()),
            "share_battery_steps_breaking_inverter_limits": float(broken.mean()),
            "share_battery_steps_soc_outside_limits": float(((s_all < battery["soc_min"]) | (s_all > battery["soc_max"])).mean()),
            "soc_min_over_windows": float(s_all.min()), "soc_max_over_windows": float(s_all.max()),
            "max_apparent_power_mva": float(apparent.max()),
            "soc_end_min": float(soc.min()), "soc_end_mean": float(soc.mean()), "soc_end_max": float(soc.max()),
            "share_of_steps_with_any_violation": float(((v < 0.95) | (v > 1.05)).any(axis=2).mean()),
            "share_of_steps_below_0.95": float((v < 0.95).any(axis=2).mean()),
            "share_of_steps_above_1.05": float((v > 1.05).any(axis=2).mean()),
            "v_min": float(v.min()), "v_max": float(v.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    run = parser.parse_args().run
    record = json.loads((run / "run.json").read_text())
    m2, m3 = record["config"]["m2"], record["config"]["m3"]
    weights = m3["weights"]
    curves = pd.read_csv(run / "curves.csv")
    references = json.loads((run / "references.json").read_text())
    val = curves.dropna(subset=["val_objective"])
    terms = [k for k in TERMS if f"train_{k}" in curves and k in weights]  # older runs have no inverter_penalty column

    def weighted(row, prefix):
        return {k: float(weights[k] * row[f"{prefix}_{k}"]) for k in terms}

    report = {"run_id": record["run_id"], "commit": record["commit"], "seed": record["seed"],
              "weighted_terms": {
                  "train": {int(s): weighted(curves[curves.step == s].iloc[0], "train") for s in (1, 100, int(curves.step.max()))},
                  **{f"val_{r}": {int(row.step): weighted(row, f"val_{r}") for _, row in val.iloc[[0, len(val) // 2, -1]].iterrows()} for r in REGIMES}},
              "references_weighted_val": {
                  c: {r: {k: weights[k] * references["terms"][c][r][k] for k in terms} for r in REGIMES} for c in references["terms"]},
              "references_objective_val": references["objective"],
              "best_val_objective": float(val.val_objective.min()), "last_val_objective": float(val.val_objective.iloc[-1]),
              "grad_norm_first_last_max": [float(curves.grad_norm.iloc[0]), float(curves.grad_norm.iloc[-1]), float(curves.grad_norm.max())],
              "seconds": float(curves.seconds.iloc[-1])}

    # What the chosen model actually does, compared with the two references, on the same validation windows.
    net = load_case33bw()
    battery = per_unit_battery(m3["battery"], net.s_base_mva)
    horizon = m3["horizon_steps"]
    data = M3Data(ROOT / "data" / "processed" / "m2_dataset.npz", net.s_base_mva, m2["time"]["dt_min"]).prepare(
        m2["placement"]["pv_buses"], m2["placement"]["bess_buses"])
    val_all = data.windows("val", horizon)
    starts = np.sort(np.random.default_rng(12345).choice(val_all, m3["evaluation"]["val_windows"], replace=False))
    solver = net.make_solver(n_iter=m3["evaluation"]["n_iter"])
    policy = make_policy(m3)
    policy.load_state_dict(torch.load(run / "model.pt"))
    variant = m3["variants"][record["variant"]]
    controllers = {"dpc": (PolicyController(policy, battery), variant["projection"]), "idle": (IdleController(), True),
                   "droop": (DroopController(battery, **m3["droop"]), True)}
    report["behaviour_on_val_windows"] = {
        r: {name: behaviour(c, data, starts, value, horizon, battery, solver, project) for name, (c, project) in controllers.items()}
        for r, value in REGIMES.items()}
    (run / "pilot_report.json").write_text(json.dumps(report, indent=2))

    # Two panels, each with one axis. Both use a log scale because the terms differ by orders of magnitude.
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="#fcfcfb")
    left.plot(curves.step, curves.train_objective, color="#b9b8b2", lw=1.2, label="Training batches")
    left.plot(val.step, val.val_objective, color="#2a78d6", lw=2, marker="o", ms=5, label="DPC on validation windows")
    left.axhline(references["objective"]["idle"], color="#eb6834", lw=2, ls="--", label="No battery action")
    left.axhline(references["objective"]["droop"], color="#1baf7a", lw=2, ls="--", label="Volt-VAR droop")
    left.set_yscale("log")
    left.set_title("Objective, mean of both regimes", loc="left", fontsize=11)
    left.legend(frameon=False, fontsize=9)
    for name in terms:
        values = np.mean([weights[name] * val[f"val_{r}_{name}"] for r in REGIMES], axis=0)
        right.plot(val.step, np.where(values > 0, values, np.nan), color=COLORS[name], lw=2, marker="o", ms=4, label=name)
    right.set_yscale("log")
    right.set_title("Weighted loss terms on validation windows, mean of both regimes", loc="left", fontsize=11)
    right.legend(frameon=False, fontsize=9, ncol=2)
    for axis in (left, right):
        axis.set_facecolor("#fcfcfb")
        axis.set_xlabel("Gradient step")
        axis.grid(axis="y", color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig(run / "pilot.png", dpi=130)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
