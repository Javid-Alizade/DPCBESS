"""Milestone M3: seeds 1 and 2 of the selected candidate H144_dead_zone_30_70, then the validation check of seeds 0, 1, 2.

Run:  python scripts/run_seeds_m3.py
Follows the seed reporting rule in docs/decisions.md: every seed is shown, nothing is re-selected. Validation only, test closed.
Seed 0 is the Experiment 2 run itself, not retrained. The instability rule and its truncated-BPTT fallback are the same as in
Experiment 2. The over-seed summary (counts of passing seeds, median and range of mean |P|) is added to the same file.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dpc_grid.dpc import is_unstable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "m3_exp2_seeds.json"
SEED0 = "results/runs/20261106-124919_candidate_no-deg_seed0"  # Experiment 2, H144_dead_zone_30_70
OPTIONS = ["--horizon", "144", "--terminal-soc-weight", "10", "--terminal-soc-dead-band", "0.2"]
TRUNCATE = 24
REGIMES = ["base_x0.7", "stress_x1.0"]


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


def train(seed, truncate=None):
    """Train one seed and judge its stability with the pre-specified rule."""
    extra = ["--truncate", str(truncate)] if truncate else []
    lines = run("scripts/train_m3.py", "--candidate", "--seed", str(seed), *OPTIONS, *extra)
    folder = Path(next(x.split("Run folder: ", 1)[1] for x in lines if x.startswith("Run folder: ")))
    summary = json.loads((folder / "summary.json").read_text())
    grad_norms = pd.read_csv(folder / "curves.csv")["grad_norm"].tolist()
    unstable = is_unstable(grad_norms, summary["stopped_on_nonfinite_at_step"] is not None)
    return folder, {"run": folder.name, "truncate": truncate, "unstable": unstable,
                    "stopped_on_nonfinite_at_step": summary["stopped_on_nonfinite_at_step"],
                    "max_grad_norm": float(max(grad_norms)), "seconds": summary["seconds"]}


def main():
    if OUT.exists():
        raise FileExistsError(f"{OUT} exists. Earlier results are never overwritten.")
    runs, stability = {"seed_0": SEED0}, {"seed_0": "Experiment 2 run, see results/m3_exp2_stability.json"}
    for seed in (1, 2):
        print(f"=== training seed {seed}", flush=True)
        folder, record = train(seed)
        stability[f"seed_{seed}"] = [record]
        if record["unstable"]:
            print(f"=== seed {seed} is unstable, repeating once with truncated BPTT of {TRUNCATE} steps", flush=True)
            folder, record = train(seed, truncate=TRUNCATE)
            stability[f"seed_{seed}"].append(record)
        if record["unstable"]:
            print(f"=== seed {seed} is unstable with truncation too; it is reported as failed and not checked", flush=True)
        else:
            runs[f"seed_{seed}"] = str(folder)
    print("=== continuous-loop check on validation, selection switched off", flush=True)
    run("scripts/check_soc_variants_m3.py", "--no-selection", "--out", str(OUT.relative_to(ROOT)),
        *(f"{name}={folder}" for name, folder in runs.items()))

    output = json.loads(OUT.read_text())
    checks = output["checks"]
    over = {"seeds_checked": list(checks), "seeds_failed_by_instability": [s for s in stability if s not in runs],
            "n_pass_a": sum(c["uses_active_power"] for c in checks.values()),
            "n_pass_b": sum(c["soc_stays_in_range"] for c in checks.values()),
            "n_pass_both": sum(c["passes"] for c in checks.values()), "mean_abs_p_mw": {}}
    for regime in REGIMES:
        values = np.array([output["candidates"][s]["regimes"][regime]["mean_abs_p_mw"] for s in checks])  # (seeds, batteries)
        over["mean_abs_p_mw"][regime] = {"median_per_battery": np.median(values, axis=0).tolist(),
                                         "min_per_battery": values.min(axis=0).tolist(),
                                         "max_per_battery": values.max(axis=0).tolist()}
    minima = [min(min(output["candidates"][s]["regimes"][r]["mean_abs_p_mw"]) for r in REGIMES) for s in checks]
    over["condition_a_minimum_per_seed"] = dict(zip(checks, minima))
    over["condition_a_minimum_median"], over["condition_a_minimum_range"] = float(np.median(minima)), [min(minima), max(minima)]
    output["stability"], output["over_seeds"] = stability, over
    OUT.write_text(json.dumps(output, indent=2))
    print(json.dumps(over, indent=1))


if __name__ == "__main__":
    main()
