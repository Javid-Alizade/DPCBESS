"""Shared helpers for the site notebooks.

The notebooks only READ committed result files in results/ (JSON, CSV run curves). They never train, solve or simulate.
"""
import json
from pathlib import Path

import matplotlib as mpl
import pandas as pd


def _find_root():
    here = Path.cwd().resolve()
    for folder in [here, *here.parents]:
        if (folder / "pyproject.toml").exists() and (folder / "results").is_dir():
            return folder
    raise FileNotFoundError("Run the notebook from inside the repository.")


ROOT = _find_root()
RESULTS = ROOT / "results"
REGIMES = {"base_x0.7": "base (load ×0.7)", "stress_x1.0": "stress (load ×1.0)"}

# Reference categorical palette (fixed order; colour follows the entity, never its rank).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10,
    "axes.prop_cycle": mpl.cycler(color=SERIES), "figure.dpi": 110,
})


def load(name):
    """Load one committed result file, e.g. load("m4_mpc_full/base_x0.7.json")."""
    return json.loads((RESULTS / name).read_text())


def provenance(record, name):
    """One line saying where a number comes from: file, commit and whether the code differed from that commit."""
    commit = record.get("commit")
    differs = record.get("code_differs_from_commit")
    text = f"Source: `results/{name}`"
    if commit:
        text += f", commit `{commit[:7]}` (private development history)"
    if differs is not None:
        text += f", code differed from that commit: **{differs}**"
    return text


def fmt(value, digits=5):
    """Plain number formatting for tables."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if value != 0 and abs(value) < 10 ** (-digits + 1):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def pct(value, digits=2):
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def table(rows, index=None):
    """A DataFrame of strings, for display."""
    frame = pd.DataFrame(rows)
    if index:
        frame = frame.set_index(index)
        # show the index label in the header row, so the header has one line and stays aligned
        frame.columns.name, frame.index.name = frame.index.name, None
    return frame
