# Differentiable Predictive Control of Battery Storage for Voltage Regulation in a Distribution Grid

**Javid Alizade** (M.Sc. student, Istanbul Gedik University)

:::{important}
**Preliminary, validation part only. Test part sealed, not yet evaluated.**
:::

[Results](#results) · [Decisions and protocols](decisions.md) · [How to reproduce](reproduce.md) ·
Code repository: [github.com/Javid-Alizade/DPCBESS](https://github.com/Javid-Alizade/DPCBESS)

## Abstract

Rooftop solar can change its output within minutes. In a distribution feeder, fast changes in solar power move bus
voltages towards or beyond the ±5% limits. Battery inverters can help, through both active power (P) and reactive power
(Q), but deciding their set-points every few minutes with an online optimiser can be slow on large feeders.
This project studies **Differentiable Predictive Control (DPC)**: a small neural-network policy is trained offline by
backpropagating through a differentiable model of the grid (DistFlow power flow), the batteries and a safety layer. Once
trained, the policy needs only one forward pass per control step.
The project is an open, reproducible implementation on the IEEE 33-bus feeder with two batteries and four PV plants, using
real, openly licensed solar and load measurements. It compares DPC with no battery action, a local Volt-VAR droop rule
and an online model predictive controller (MPC). The results shown here are **preliminary**: they come from the
validation part of the data, and the test part has not been opened.

## Problem

- **Grid.** IEEE 33-bus radial feeder (Baran and Wu, 1989), per-unit system, bus 0 is the substation. Two load levels are
  always reported: a base scenario with the load scaled by 0.7 and a stress scenario at the standard load (×1.0).
- **Disturbances.** 3.9 MW of PV at buses 14, 17, 29 and 32 and time-varying load at every bus, at 5-minute resolution.
- **Actuators.** Two batteries at buses 17 and 32, 0.5 MW / 1 MWh each, which can set both P and Q (inverter rating
  0.5 MVA). A positive P means discharge, so the net consumption at a bus is load − solar − P.
- **Goal.** Keep all bus voltages inside 0.95-1.05 pu, and close to 1 pu, with low network losses and modest battery use,
  in particular around extreme solar ramps.
- **Research questions** (from the proposal; RQ2 reworded, see the caveats):
  - RQ1. How can nonlinear DistFlow power flow, inverter constraints and battery dynamics be represented as
    differentiable computation graphs, and how do gradients propagate through the closed loop over a finite horizon?
  - RQ2. To what extent can physics-informed penalties combined with a differentiable feasibility projection **improve
    voltage margin and reduce voltage violations** during extreme solar-ramp events, compared with black-box deep
    reinforcement learning and online MPC?
  - RQ3. How effectively can DPC include battery stress and degradation costs in policy training while keeping voltage
    regulation? *(Not started; see [Next steps](#next-steps).)*

## Method

1. **Differentiable plant.** Branch-flow DistFlow solved by a fixed number of backward/forward sweeps, written in PyTorch
   so that gradients pass through it (8 sweeps in training, 15 in evaluation). It is checked against pandapower.
2. **Policy.** A multilayer perceptron (2 hidden layers of 128, Tanh) maps the current state to P and Q of both batteries.
   Inputs: state of charge (SoC), voltages at the battery buses on the previous step, total load, PV at the four PV buses,
   the latest solar ramp rate, the previous action and the time of day. **No forecast** of future load or solar is given.
3. **Safety layer.** The raw action is limited in three steps: P is limited so that SoC stays in 10-90% within the step,
   then P and Q are clipped to their limits, then (P, Q) is scaled radially onto the 0.5 MVA circle if needed. The layer is
   differentiable. It enforces battery and inverter limits only; **it does not guarantee voltage limits**, which are
   penalised in the loss and measured afterwards.
4. **Training loss.** The terms of the proposal's eq. (16): network losses, voltage deviation from 1 pu, control effort, a
   soft SoC penalty and a soft voltage-limit penalty, plus a terminal SoC term with a 30-70% dead zone. Degradation and
   C-rate terms are off. The policy is rolled out over a horizon of 144 steps (12 h) and trained by backpropagation
   through time.
5. **Selection and evaluation.** Candidates were compared by pre-specified rules on the validation part only, in a
   continuous closed loop over all validation steps. Every rule was written down, with its date, before the run it judged
   ([Decisions](decisions.md)).
6. **References.** No battery action; a Volt-VAR droop rule (original setting, and a setting tuned on the training part);
   an online MPC with a convex LinDistFlow model and a persistence forecast; and an oracle MPC with exact future data, as an
   upper bound.

## Data

- **Solar.** Visser et al. (2022), rooftop PV measurements from Utrecht, the Netherlands, Zenodo,
  [doi:10.5281/zenodo.10953360](https://doi.org/10.5281/zenodo.10953360), CC BY 4.0.
- **Load.** Trindade (2015), ElectricityLoadDiagrams20112014, UCI Machine Learning Repository,
  [doi:10.24432/C58C86](https://doi.org/10.24432/C58C86), CC BY 4.0.
- **Split by time, no overlap.** Training: 18 months, validation: 6 months, test: 12 months. Solar and load years are paired
  by calendar, so they **share no real weather**. Gaps in the solar data are never filled; windows with missing data are
  skipped.
- **Extreme ramp.** A step whose absolute change of total PV is in the top 1% of the training part.

Details: [notebook 02](notebooks/02_data_pipeline.ipynb).

(results)=
## Results (validation part, preliminary)

```{include} _caveats.md
```

The results are in seven notebooks. Each one only reads saved result files from the repository and draws tables and
figures from them. Nothing is trained or optimised while the site is built.

| Notebook | What it shows |
|---|---|
| [01 Network and solver](notebooks/01_network_and_solver.ipynb) | The 33-bus model and the check of the differentiable solver against pandapower |
| [02 Data pipeline](notebooks/02_data_pipeline.ipynb) | Solar and load data, the split, ramp rates, extreme events, voltages without control |
| [03 DPC training and selection](notebooks/03_dpc_training_and_selection.ipynb) | Experiments 1 and 2, the pre-specified selection rule and its outcome |
| [04 Seeds and droop](notebooks/04_seeds_and_droop.ipynb) | Seeds 0, 1 and 2 of the selected policy, and the Volt-VAR droop tuned on training data |
| [05 MPC baseline](notebooks/05_mpc_baseline.ipynb) | Pilots, the overlap diagnostic, and the full persistence MPC on validation |
| [06 Oracle subset](notebooks/06_oracle_subset.ipynb) | All controllers on 28 validation days, including the oracle upper bound |
| [07 Ablations](notebooks/07_ablations.ipynb) | Removing the soft penalties, the safety layer, or both (degradation off) |

(next-steps)=
## Next steps

- **Sealed test run.** One run on the test part, with a protocol written down before it, reporting both load scenarios
  and all seeds. It has not been done.
- **RQ2 metrics** (worst voltage margin, maximum deviation and time to the safe band around extreme ramps) are defined and
  implemented, but have not yet been computed for any controller.
- **Open:** a deep reinforcement learning baseline and the degradation extension (RQ3). Neither has been started.

## Reproducibility

All code, configurations and result files are versioned. Every run records its configuration, random seed and git commit,
and whether the code differed from that commit. The data are downloaded from their original sources with checksums.
See [How to reproduce](reproduce.md).

## Citation and license

```bibtex
@misc{alizade2026dpcbess,
  author = {Alizade, Javid},
  title  = {Differentiable Predictive Control of Battery Storage
            for Voltage Regulation in a Distribution Grid
            (preliminary)},
  year   = {2026},
  url    = {https://javid-alizade.github.io/DPCBESS/},
  note   = {Preliminary results on the validation part;
            test part not yet evaluated}
}
```

Code: MIT License. Text and figures: CC BY 4.0. The datasets keep their own licences (CC BY 4.0) and are cited above.
