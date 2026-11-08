# DPC for battery voltage regulation on the IEEE 33-bus feeder

Differentiable Predictive Control (DPC) of two battery inverters (P and Q) for voltage regulation in a distribution feeder
with rooftop solar, compared with no battery action, a Volt-VAR droop rule and an online MPC.

> **Preliminary, validation part only. Test part sealed, not yet evaluated.**
> All results so far come from the validation part of the data and one network. The code refuses to load the test part
> (`ClosedTestPartError` in `src/dpc_grid/dpc.py`).

- Site (Jupyter Book, built from the notebooks in `site/`): https://javid-alizade.github.io/DPCBESS/
- Decisions and protocols, with dates: [`docs/decisions.md`](docs/decisions.md)
- Licences: code MIT ([`LICENSE`](LICENSE)); text and figures CC BY 4.0 ([`LICENSE-CONTENT.md`](LICENSE-CONTENT.md))

## Repository layout

| Path | Content |
|---|---|
| `src/dpc_grid/` | network loader, differentiable DistFlow solver, data pipeline, policy, safety layer, closed loop, MPC, metrics, run log |
| `configs/` | all numeric settings (`m2.json` data, `m3.json` training and controllers) |
| `scripts/` | one script per step; the docstring at the top of each script says how to run it |
| `tests/` | `pytest` tests |
| `results/` | committed result files (`*.json`, figures); `results/runs/<id>/` holds the log of every training run |
| `site/` | Jupyter Book: landing page and notebooks 01-07, which only read `results/` |
| `docs/decisions.md` | dated decisions, pre-specified protocols and their outcomes |

<!-- reproduce-start -->
## How to reproduce

The results were produced on Windows 11 with Python 3.14 on a CPU. Versions used: torch 2.15.0, numpy 2.4.6, pandas
2.3.3, cvxpy 1.9.3 with Clarabel 0.11.1, pandapower 3.5.6, matplotlib 3.11.2.

### 1. Install

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .        # on Linux or macOS: .venv/bin/python
.venv/Scripts/python -m pytest -q
```

### 2. Data (open, CC BY 4.0)

```bash
python scripts/download_m2.py      # downloads from Zenodo and UCI, checks MD5, writes results/m2_download.json
python scripts/cache_pv_m2.py      # once, a few minutes
python scripts/prepare_m2.py       # builds the 5-minute scenario, the split and the ramp labels
```

Sources: solar, Visser et al. (2022), [doi:10.5281/zenodo.10953360](https://doi.org/10.5281/zenodo.10953360);
load, Trindade (2015), [doi:10.24432/C58C86](https://doi.org/10.24432/C58C86). Gaps are never filled.

### 3. Results, in the order they were made

Every script writes its result file in `results/` and records the configuration, the seed, the git commit and whether the
code differed from that commit. Times are rough, from the original runs on one CPU.

| Step | Command | Result file(s) | Time |
|---|---|---|---|
| M1 solver check | `python scripts/validate_m1.py` | `m1_validation.json` | minutes |
| M2 data checks | `python scripts/check_ramps_m2.py`, `check_rq2_m2.py`, `baseline_margins_m2.py` | `m2_*.json` | minutes |
| M3 Experiment 1 (H = 24) | `python scripts/run_soc_candidates_m3.py` | `m3_soc_variants_1000.json` | under 1 h |
| M3 Experiment 2 (H = 72, 144) | `python scripts/run_exp2_m3.py` | `m3_exp2_soc_variants.json`, `m3_exp2_stability.json` | about 4 h |
| M3 seeds 1 and 2 | `python scripts/run_seeds_m3.py` | `m3_exp2_seeds.json` | about 1 h |
| M3 droop tuning | `python scripts/tune_droop_m3.py`, then `tune_droop_round2_m3.py` | `m3_droop_tuning*.json` | about 1.5 h |
| M4 MPC, lossless pilot, full run, oracle subset | `powershell -File scripts/night_m4.ps1` (or the three scripts it calls) | `m4_mpc_pilot_lossless.json`, `m4_mpc_full/`, `m4_oracle_subset/` | more than 11 h |
| M3 ablations | `python scripts/run_ablations_m3.py` | `m3_ablations.json`, `m3_ablations/` | about 4 h |
| Site | see below | `site/_build/html/` | minutes |

Training on a different machine or library version will not give bit-identical numbers. The run logs in `results/runs/`
record what was used.

### 4. Build the site locally

```bash
python -m pip install jupyter-book
jupyter-book build site
```

The notebooks only read `results/`; they do not train or solve anything.
### Commit hashes

Text labels in the exported result files were normalised (file-name references replaced); no numeric value was changed. Commit hashes stored in the result files (`results/`) and cited in the decision record refer to the private development history, which is not published. The public repository starts from a single initial commit.
<!-- reproduce-end -->

## Citation

Alizade, J. (2026). *Differentiable Predictive Control of Battery Storage for Voltage Regulation in a Distribution Grid*
(preliminary). Network data: Baran and Wu (1989), IEEE 33-bus test feeder, via pandapower `case33bw`.
