"""Milestone M3, Experiment 2: the four SoC-penalty candidates at H = 72 and H = 144, then the continuous-loop check.

Run:  python scripts/run_exp2_m3.py
Pre-specified in docs/decisions.md ("Experiment 2"). Validation only, seed 0, test part closed.
An unstable run (rule in dpc.is_unstable) is repeated once with truncated BPTT of 24 steps. If that is unstable too, the
candidate fails and is not checked. Stability of every run is written to results/m3_exp2_stability.json.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from dpc_grid.dpc import is_unstable

ROOT = Path(__file__).resolve().parents[1]
OUT = "results/m3_exp2_soc_variants.json"
STABILITY = ROOT / "results" / "m3_exp2_stability.json"
HORIZONS = [72, 144]
TRUNCATE = 24
CANDIDATES = {"weight_10": ["--terminal-soc-weight", "10"],
              "weight_0": ["--terminal-soc-weight", "0"],
              "weight_0.1": ["--terminal-soc-weight", "0.1"],
              "dead_zone_30_70": ["--terminal-soc-weight", "10", "--terminal-soc-dead-band", "0.2"]}


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


def train(horizon, options, truncate=None):
    """Train one candidate and judge its stability with the pre-specified rule."""
    extra = ["--truncate", str(truncate)] if truncate else []
    lines = run("scripts/train_m3.py", "--candidate", "--seed", "0", "--horizon", str(horizon), *options, *extra)
    folder = Path(next(x.split("Run folder: ", 1)[1] for x in lines if x.startswith("Run folder: ")))
    summary = json.loads((folder / "summary.json").read_text())
    grad_norms = pd.read_csv(folder / "curves.csv")["grad_norm"].tolist()
    unstable = is_unstable(grad_norms, summary["stopped_on_nonfinite_at_step"] is not None)
    later = pd.Series(grad_norms[100:])
    return folder, {"run": folder.name, "truncate": truncate, "unstable": unstable,
                    "stopped_on_nonfinite_at_step": summary["stopped_on_nonfinite_at_step"],
                    "share_of_steps_after_100_with_grad_norm_above_10": float((later > 10).mean()) if len(later) else None,
                    "max_grad_norm": float(max(grad_norms)), "seconds": summary["seconds"]}


def main():
    if (ROOT / OUT).exists() or STABILITY.exists():
        raise FileExistsError("Experiment 2 results exist. Earlier results are never overwritten.")
    stability, passing = {}, []
    for horizon in HORIZONS:
        for name, options in CANDIDATES.items():
            label = f"H{horizon}_{name}"
            print(f"=== training {label}", flush=True)
            folder, record = train(horizon, options)
            stability[label] = [record]
            if record["unstable"]:
                print(f"=== {label} is unstable, repeating once with truncated BPTT of {TRUNCATE} steps", flush=True)
                folder, record = train(horizon, options, truncate=TRUNCATE)
                stability[label].append(record)
            if record["unstable"]:
                print(f"=== {label} is unstable with truncation too, so it fails", flush=True)
            else:
                passing.append(f"{label}={folder}")
            STABILITY.write_text(json.dumps(stability, indent=2))
    print("=== continuous-loop check on validation", flush=True)
    run("scripts/check_soc_variants_m3.py", "--out", OUT, *passing)


if __name__ == "__main__":
    main()
