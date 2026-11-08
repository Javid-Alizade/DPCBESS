"""Milestone M3: the 9 ablation runs (3 variants x seeds 0, 1, 2), then their validation check. Protocol: docs/decisions.md,
"M3 ablations: protocol" and the decisions D1-D5 of 2026-11-07.

Run:  python scripts/run_ablations_m3.py
Each variant-seed is trained exactly like the selected candidate (1000 steps, H = 144, terminal weight 10 with the 30-70% dead
zone). The stability rule and the truncated-BPTT fallback are those of Experiment 2. Then every stable run is checked in the
continuous loop on ALL validation steps, both regimes, with its own safety-layer setting (nothing is clipped without it).
Resumable: progress is saved after every finished training run and every finished check in results/m3_ablations/progress.json.
Running the same command again skips finished parts; a run that was cut off is started again from the beginning (its partial
folder stays in results/runs and is listed as interrupted). A resume is allowed only at the same clean commit.
Results: results/m3_ablations/<variant>_seed<k>.json per run and results/m3_ablations.json over seeds. Test part closed.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dpc_grid.dpc import is_unstable
from dpc_grid.runlog import git_state

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results" / "m3_ablations"
OUT = ROOT / "results" / "m3_ablations.json"
VARIANTS = ["penalty-only-no-deg", "projection-only", "no-constraints-no-deg"]
SEEDS = [0, 1, 2]
OPTIONS = ["--horizon", "144", "--terminal-soc-weight", "10", "--terminal-soc-dead-band", "0.2"]
TRUNCATE = 24
REGIMES = ["base_x0.7", "stress_x1.0"]
REFERENCE = "results/m3_exp2_seeds.json"  # the selected candidate, seeds 0/1/2, the same check on the same steps
# Scalars reported as median and range over seeds, per regime (paths inside one regime entry of the check output).
METRICS = {"objective": ("objective",), "vvr_share_of_steps": ("summary", "vvr_share_of_steps_with_any_violation"),
           "v_rmse_pu": ("summary", "v_rmse_pu"), "mean_network_loss_kw": ("summary", "mean_network_loss_kw"),
           "mean_abs_p_mw": ("summary", "mean_abs_p_mw"), "mean_abs_q_mvar": ("summary", "mean_abs_q_mvar"),
           "share_at_s_limit": ("share_at_s_limit",), "share_soc_clip_active": ("share_soc_clip_active",),
           "end_of_day_soc_min": ("end_of_day_soc_min",), "end_of_day_soc_max": ("end_of_day_soc_max",),
           "soc_min": ("summary", "soc_min"), "soc_max": ("summary", "soc_max"),
           "share_battery_steps_outside_soc_limits": ("summary", "share_of_battery_steps_outside_soc_limits"),
           "share_battery_steps_breaking_inverter_limits": ("summary", "share_of_battery_steps_breaking_inverter_limits"),
           "share_of_steps_with_nonfinite_voltage": ("share_of_steps_with_nonfinite_voltage",)}


def run(*args):
    """Run one script, show its output as it comes, and return the output lines."""
    process = subprocess.Popen([sys.executable, "-I", *args], cwd=ROOT, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line.strip())
    if process.wait() != 0:
        raise RuntimeError(f"failed: {args}")
    return lines


def save(progress):
    tmp = FOLDER / "progress.tmp.json"
    tmp.write_text(json.dumps(progress, indent=2))
    tmp.replace(FOLDER / "progress.json")


def train(variant, seed, truncate=None):
    """Train one variant-seed and judge its stability with the pre-specified rule."""
    extra = ["--truncate", str(truncate)] if truncate else []
    lines = run("scripts/train_m3.py", "--candidate", "--variant", variant, "--seed", str(seed), *OPTIONS, *extra)
    folder = Path(next(x.split("Run folder: ", 1)[1] for x in lines if x.startswith("Run folder: ")))
    summary = json.loads((folder / "summary.json").read_text())
    grad_norms = pd.read_csv(folder / "curves.csv")["grad_norm"].tolist()
    unstable = is_unstable(grad_norms, summary["stopped_on_nonfinite_at_step"] is not None)
    return {"run": str(folder.relative_to(ROOT)), "truncate": truncate, "unstable": bool(unstable),
            "stopped_on_nonfinite_at_step": summary["stopped_on_nonfinite_at_step"],
            "max_grad_norm": float(max(grad_norms)), "seconds": summary["seconds"]}


def over_seeds(checks):
    """Median and range of each metric over the seeds of one variant, per regime (per battery where the metric has two)."""
    out = {}
    for regime in REGIMES:
        out[regime] = {}
        for name, path in METRICS.items():
            values = []
            for check in checks:
                value = check["regimes"][regime]
                for key in path:
                    value = value.get(key) if isinstance(value, dict) else None
                values.append(value)
            if any(v is None for v in values):  # older check files lack the newer diagnostics
                out[regime][name] = None
                continue
            values = np.array(values, dtype=float)
            out[regime][name] = {"median": float(np.median(values)), "min": float(values.min()), "max": float(values.max())}
        per_battery = np.array([c["regimes"][regime]["mean_abs_p_mw"] for c in checks])
        out[regime]["mean_abs_p_mw_per_battery"] = {"median": np.median(per_battery, axis=0).tolist(),
                                                    "min": per_battery.min(axis=0).tolist(),
                                                    "max": per_battery.max(axis=0).tolist()}
    objectives = np.array([c["objective"] for c in checks])
    out["objective_mean_of_regimes"] = {"median": float(np.median(objectives)), "min": float(objectives.min()),
                                        "max": float(objectives.max())}
    return out


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    state = git_state(ROOT)
    if state["code_differs_from_commit"]:
        raise RuntimeError("Code differs from the last commit. Commit first, so this job can be repeated and resumed exactly.")
    progress_file = FOLDER / "progress.json"
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
        if progress["commit"] != state["commit"]:
            raise RuntimeError(f"The job was started at commit {progress['commit'][:7]}; resume only at that commit.")
        progress["history"].append({"event": "resume", "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    else:
        progress = {"commit": state["commit"], "history": [{"event": "start", "time": time.strftime("%Y-%m-%d %H:%M:%S")}],
                    "training": {}, "interrupted_attempts": {}}
    save(progress)

    for variant in VARIANTS:
        for seed in SEEDS:
            name = f"{variant}_seed{seed}"
            if name in progress["training"]:
                print(f"=== {name}: trained earlier, skipped", flush=True)
                continue
            before = {p.name for p in (ROOT / "results" / "runs").iterdir()}
            print(f"=== training {name}", flush=True)
            try:
                records = [train(variant, seed)]
                if records[0]["unstable"]:
                    print(f"=== {name} is unstable, repeating once with truncated BPTT of {TRUNCATE} steps", flush=True)
                    records.append(train(variant, seed, truncate=TRUNCATE))
            except BaseException:
                new = sorted({p.name for p in (ROOT / "results" / "runs").iterdir()} - before)
                progress["interrupted_attempts"].setdefault(name, []).extend(new)
                save(progress)
                raise
            progress["training"][name] = {"records": records, "failed": records[-1]["unstable"],
                                          "used_run": None if records[-1]["unstable"] else records[-1]["run"]}
            save(progress)

    for name, training in progress["training"].items():
        out = FOLDER / f"{name}.json"
        if training["failed"]:
            print(f"=== {name}: unstable with truncation too, reported as failed and not checked", flush=True)
            continue
        if out.exists():
            print(f"=== {name}: checked earlier, skipped", flush=True)
            continue
        print(f"=== continuous-loop check on validation: {name}", flush=True)
        run("scripts/check_soc_variants_m3.py", "--no-selection", "--no-references", "--out", str(out.relative_to(ROOT)),
            f"{name}={training['used_run']}")

    reference = json.loads((ROOT / REFERENCE).read_text())
    output = {"part": "validation only, continuous loop, all steps", "protocol": "docs/decisions.md, M3 ablations: protocol (D1-D5)",
              **state, "labels": "degradation off in every variant; penalty-only includes the D1 inverter-limit penalty",
              "training": progress["training"], "interrupted_attempts": progress["interrupted_attempts"],
              "history": progress["history"], "variants": {},
              "selected_candidate_reference": {"file": REFERENCE,
                                               "over_seeds": over_seeds(list(reference["candidates"].values()))}}
    for variant in VARIANTS:
        checks, per_seed = [], {}
        for seed in SEEDS:
            file = FOLDER / f"{variant}_seed{seed}.json"
            if file.exists():
                check = next(iter(json.loads(file.read_text())["candidates"].values()))
                checks.append(check)
                per_seed[f"seed_{seed}"] = check
        output["variants"][variant] = {"seeds_checked": list(per_seed), "per_seed": per_seed,
                                       "over_seeds": over_seeds(checks) if checks else None}
    OUT.write_text(json.dumps(output, indent=2))
    progress["history"].append({"event": "finished", "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    save(progress)
    print(f"Saved {OUT}", flush=True)
    print("ablations finished", flush=True)


if __name__ == "__main__":
    main()
