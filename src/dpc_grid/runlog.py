"""Every run writes its config, seed and git commit, and is listed in results/runs/index.csv."""
import csv
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

CODE_PATHS = ["src", "scripts", "configs", "tests", "pyproject.toml"]


def git_executable():
    found = shutil.which("git") or r"C:\Program Files\Git\cmd\git.exe"
    if not Path(found).exists():
        raise RuntimeError("git was not found, so this run cannot record its commit")
    return found


def git_state(root):
    """The current commit, and whether code differs from it. Result files do not count as code."""
    def run(*args):
        return subprocess.run([git_executable(), *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"),
            "code_differs_from_commit": bool(run("status", "--porcelain", "--", *CODE_PATHS))}


def start_run(root, kind, name, seed, config, require_clean_code):
    """Create results/runs/<name>/, save everything needed to repeat the run, and add it to the index."""
    root = Path(root)
    state = git_state(root)
    if require_clean_code and state["code_differs_from_commit"]:
        raise RuntimeError("Code differs from the last commit. Commit first, so this run can be repeated exactly.")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{kind}_{name}_seed{seed}"
    folder = root / "results" / "runs" / run_id
    folder.mkdir(parents=True)
    record = {"run_id": run_id, "kind": kind, "variant": name, "seed": seed, "started": stamp, **state,
              "command": sys.argv,
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
              "config": config}
    (folder / "run.json").write_text(json.dumps(record, indent=2))
    index = root / "results" / "runs" / "index.csv"
    is_new = not index.exists()
    with open(index, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["run_id", "kind", "variant", "seed", "commit", "code_differs_from_commit"])
        writer.writerow([run_id, kind, name, seed, state["commit"], state["code_differs_from_commit"]])
    return folder
