"""Milestone M3: train the four SoC-penalty candidates for the full 1000 steps, then run the continuous-loop check.

Run:  python scripts/run_soc_candidates_m3.py
Same seed for all. Validation only, and the rule is in docs/decisions.md. Each training is a logged run in results/runs/.
Any failure stops everything, so a partial table can never look like a finished one.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = "results/m3_soc_variants_1000.json"
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


def main():
    if (ROOT / OUT).exists():
        raise FileExistsError(f"{OUT} exists. Earlier results are never overwritten.")
    folders = []
    for name, options in CANDIDATES.items():
        print(f"=== training {name}", flush=True)
        lines = run("scripts/train_m3.py", "--candidate", "--seed", "0", *options)
        folder = next(x.split("Run folder: ", 1)[1] for x in lines if x.startswith("Run folder: "))
        folders.append(f"{name}={folder}")
    print("=== continuous-loop check on validation", flush=True)
    run("scripts/check_soc_variants_m3.py", "--out", OUT, *folders)


if __name__ == "__main__":
    main()
