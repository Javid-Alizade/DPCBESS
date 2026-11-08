"""Milestone M3: train a DPC variant on the training part and check it on the validation part.

Pilot, which is short and always comes first:  python scripts/train_m3.py --pilot
Full run, after the pilot has been checked:    python scripts/train_m3.py --variant no-deg --seed 0

The test part cannot be reached from here: M3Data refuses it and drops its steps while reading the file.
Each run writes results/runs/<id>/: run.json (config, seed, commit), curves.csv, references.json, summary.json, model.pt.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from dpc_grid import load_case33bw
from dpc_grid.dpc import (DroopController, IdleController, M3Data, PolicyController, closed_loop, closed_loop_summary,
                          make_policy, per_unit_battery, rollout, sample_windows, total_objective, windows_with_extreme)
from dpc_grid.runlog import start_run

ROOT = Path(__file__).resolve().parents[1]
TERMS = ["loss", "voltage", "effort", "soc_penalty", "voltage_penalty", "terminal_soc", "inverter_penalty"]
REGIMES = {"base_x0.7": 0.7, "stress_x1.0": 1.0}


def validation_terms(controller, data, starts, battery, solver, horizon, use_projection):
    """The same fixed validation windows at both load scales, starting at 50% SoC, per regime."""
    out = {}
    with torch.no_grad():
        for name, value in REGIMES.items():
            scale = torch.full((len(starts),), value)
            soc = torch.full((len(starts), len(data.bess_buses)), 0.5)
            out[name] = {k: v.item() for k, v in rollout(controller, data, starts, scale, soc, horizon, battery, solver,
                                                         use_projection).items()}
    return out


def objective_of(terms_by_regime, weights, penalties, inverter_penalty=False):
    """Mean of the objective over both regimes, as a plain number."""
    values = [total_objective({k: torch.tensor(v) for k, v in terms.items()}, weights, penalties, inverter_penalty).item()
              for terms in terms_by_regime.values()]
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="no-deg")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--candidate", action="store_true",
                        help="full length, for the SoC-penalty check: no closed-loop scoring and no RQ2 metrics here")
    parser.add_argument("--terminal-soc-weight", type=float, default=None,
                        help="pilot or candidate only: another terminal SoC weight. The value is logged with the run.")
    parser.add_argument("--terminal-soc-dead-band", type=float, default=None,
                        help="pilot or candidate only: half width of the no-penalty zone around 50%% SoC. Logged with the run.")
    parser.add_argument("--horizon", type=int, default=None,
                        help="pilot or candidate only: another horizon in steps (Experiment 2). Logged with the run.")
    parser.add_argument("--truncate", type=int, default=None,
                        help="pilot or candidate only: truncated BPTT segment length in steps. Logged with the run.")
    args = parser.parse_args()
    if (args.horizon is not None or args.truncate is not None) and not (args.pilot or args.candidate):
        parser.error("--horizon and --truncate work only together with --pilot or --candidate")
    if args.pilot and args.candidate:
        parser.error("choose --pilot or --candidate, not both")
    if (args.terminal_soc_weight is not None or args.terminal_soc_dead_band is not None) and not (args.pilot or args.candidate):
        parser.error("the --terminal-soc options work only together with --pilot or --candidate")

    m2 = json.loads((ROOT / "configs" / "m2.json").read_text())
    m3 = json.loads((ROOT / "configs" / "m3.json").read_text())
    if args.terminal_soc_weight is not None:
        m3["weights"]["terminal_soc"] = args.terminal_soc_weight
    if args.terminal_soc_dead_band is not None:
        m3["battery"]["terminal_soc_dead_band"] = args.terminal_soc_dead_band
    if args.horizon is not None:
        m3["horizon_steps"] = args.horizon
    m3["train"]["truncate"] = args.truncate
    variant = m3["variants"][args.variant]
    use_penalties, use_projection = variant["penalties"], variant["projection"]
    use_inverter_penalty = variant.get("inverter_penalty", False)  # penalty-only only, decision D1 of 2026-11-07
    steps = m3["pilot"]["steps"] if args.pilot else m3["train"]["steps"]
    val_every = m3["pilot"]["val_every"] if args.pilot else m3["train"]["val_every"]

    kind = "pilot" if args.pilot else "candidate" if args.candidate else "full"
    folder = start_run(ROOT, kind, args.variant, args.seed,
                       {"m2": m2, "m3": m3, "steps": steps, "val_every": val_every}, require_clean_code=True)
    print(f"Run folder: {folder}", flush=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = load_case33bw()
    weights = m3["weights"]
    horizon = m3["horizon_steps"]
    battery = per_unit_battery(m3["battery"], net.s_base_mva)
    data = M3Data(ROOT / "data" / "processed" / "m2_dataset.npz", net.s_base_mva, m2["time"]["dt_min"]).prepare(
        m2["placement"]["pv_buses"], m2["placement"]["bess_buses"])
    train_solver = net.make_solver(n_iter=m3["train"]["n_iter"])
    eval_solver = net.make_solver(n_iter=m3["evaluation"]["n_iter"])

    train_starts = data.windows("train", horizon)
    extreme_starts = windows_with_extreme(data, train_starts, horizon)
    val_rng = np.random.default_rng(12345)  # the same validation windows for every run and every controller
    val_all = data.windows("val", horizon)
    val_starts = np.sort(val_rng.choice(val_all, m3["evaluation"]["val_windows"], replace=False))
    print(f"train windows {len(train_starts)}, with extreme ramp {len(extreme_starts)}, val windows used {len(val_starts)}", flush=True)

    droop = DroopController(battery, **m3["droop"])
    references = {"idle": validation_terms(IdleController(), data, val_starts, battery, eval_solver, horizon, True),
                  "droop": validation_terms(droop, data, val_starts, battery, eval_solver, horizon, True)}
    references_objective = {k: objective_of(v, weights, use_penalties, use_inverter_penalty) for k, v in references.items()}
    (folder / "references.json").write_text(json.dumps({"terms": references, "objective": references_objective}, indent=2))
    print(f"references on validation windows: {references_objective}", flush=True)

    policy = make_policy(m3)
    controller = PolicyController(policy, battery)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=m3["train"]["lr"], weight_decay=m3["train"]["weight_decay"])
    low, high = m3["battery"]["soc_start_range"]
    scales = list(REGIMES.values())
    batch = m3["train"]["batch"]

    best, rows = float("inf"), []
    stopped_on_nonfinite = None
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        starts = sample_windows(rng, train_starts, extreme_starts, batch, m3["train"]["extreme_window_share"])
        scale = torch.as_tensor(rng.choice(scales, batch), dtype=torch.float32)
        soc0 = torch.as_tensor(rng.uniform(low, high, (batch, len(data.bess_buses))), dtype=torch.float32)
        terms = rollout(controller, data, starts, scale, soc0, horizon, battery, train_solver, use_projection,
                        truncate=m3["train"]["truncate"])
        objective = total_objective(terms, weights, use_penalties, use_inverter_penalty)
        optimizer.zero_grad()
        objective.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), m3["train"]["grad_clip"]).item()  # norm before clipping
        if not (np.isfinite(objective.item()) and np.isfinite(grad_norm)):
            stopped_on_nonfinite = step  # a NaN or infinity: stop, record it, and never take this step
            rows.append({"step": step, "train_objective": objective.item(), "grad_norm": grad_norm,
                         "seconds": time.perf_counter() - t0})
            print(f"step {step}: loss or gradient is not finite, training stops", flush=True)
            break
        optimizer.step()

        row = {"step": step, "train_objective": objective.item(), "grad_norm": grad_norm,
               "seconds": time.perf_counter() - t0}
        row.update({f"train_{k}": terms[k].item() for k in TERMS})
        if step == 1 or step % val_every == 0 or step == steps:
            val = validation_terms(controller, data, val_starts, battery, eval_solver, horizon, use_projection)
            row["val_objective"] = objective_of(val, weights, use_penalties, use_inverter_penalty)
            for regime, values in val.items():
                row.update({f"val_{regime}_{k}": values[k] for k in TERMS})
            if row["val_objective"] < best:  # model selection uses the validation objective only
                best = row["val_objective"]
                torch.save(policy.state_dict(), folder / "model.pt")
            print(f"step {step:>4}: train {row['train_objective']:.5f}, val {row['val_objective']:.5f}, "
                  f"grad {grad_norm:.3f}, {row['seconds']:.0f} s", flush=True)
        rows.append(row)

    columns = sorted({k for r in rows for k in r}, key=lambda k: (k != "step", k))
    with open(folder / "curves.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    summary = {"best_val_objective": best, "seconds": rows[-1]["seconds"], "steps": steps,
               "steps_done": rows[-1]["step"], "stopped_on_nonfinite_at_step": stopped_on_nonfinite,
               "horizon_steps": horizon, "truncate": m3["train"]["truncate"],
               "references_val_objective": references_objective}
    if kind == "full":  # a full run also scores the chosen model on validation with the continuous loop, in both regimes
        policy.load_state_dict(torch.load(folder / "model.pt"))
        metrics_config = dict(m2["rq2_metrics"], seed=args.seed)
        summary["validation_closed_loop"] = {}
        controllers = {"dpc": (controller, use_projection), "idle": (IdleController(), True), "droop": (droop, True)}
        for regime, value in REGIMES.items():
            summary["validation_closed_loop"][regime] = {}
            for name, (ctrl, project) in controllers.items():
                print(f"closed loop on validation: {regime}, {name}", flush=True)
                result = closed_loop(ctrl, data, "val", value, battery, eval_solver, project)
                summary["validation_closed_loop"][regime][name] = closed_loop_summary(result, data, "val", battery, metrics_config)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done. Saved {folder}", flush=True)


if __name__ == "__main__":
    main()
