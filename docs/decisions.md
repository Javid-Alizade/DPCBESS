# Decisions and protocols

This is the dated record of the project's decisions, pre-specified protocols and their outcomes, in the order they were
made. It is the public version of the project's working notes.

**Labels.** "Option proposed" marks a default that was proposed as an option and then accepted; "chosen" marks a choice
made directly. Every rule can be changed only in a new dated entry.

**Pre-specification.** A protocol marked "written BEFORE" was written down and committed to the repository before the run it governs.

**Commit hashes.** Text labels in the exported result files were normalised (file-name references replaced); no numeric value was changed. Commit hashes stored in the result files (`results/`) and cited in the decision record refer to the private development history, which is not published. The public repository starts from a single initial commit.

**Status.** All results are from the validation part. The test part is sealed: the code refuses to load it, and it has not
been evaluated.

---

## Conventions (fixed from the start)

- Per-unit system. Bus 0 is the slack bus. Buses are in topological order (parent[j] < j).
- Battery sign: P > 0 means DISCHARGE. Net consumption p_net = load − solar − P_bess. The proposal had `+ P_bess`, which
  is a sign error: it would make discharge increase the load.
- Proposal eq. (9): the left side is P_ij,t, not 2 P_ij,t, and the battery enters with a minus sign.
- Physics: branch-flow DistFlow with a backward/forward sweep (`src/dpc_grid/physics.py`), kept differentiable.
- Safety layer: an energy limit on P from SoC, then clipping and radial scaling of (P, Q). It acts on battery and inverter
  limits only and does NOT guarantee voltage feasibility. Voltage violations are measured (VVR) and penalised in the loss.
- Strict split on time (train / validation / test, non-overlapping). Nothing is tuned on the test part.
- Only open datasets, cited. IEEE 33-bus: Baran and Wu (1989), pandapower `case33bw`.
- Results are never adjusted to look better. If DPC does not beat a baseline, that is reported.
- Both load regimes (×0.7 and ×1.0) are reported for every result.

## Load scenarios (chosen, before any data or controller result)

- Base scenario: load ×0.7. Stress scenario: standard load ×1.0. The standard case33bw already violates the ±5% limit at
  peak without any control, so the load is reduced to study the effect of solar and battery control. The scale was chosen
  before looking at any time series or controller result and is not changed after seeing results.
- The ×0.7 load scale was chosen from peak-load voltages of case33bw, before any real time series was used. The PV
  placement and the first PV size (2.6 MW) came from a table that assumed a midday load of 30% of the scenario load; the
  real load is about 66% of peak at midday, which changed the PV size to 3.9 MW (see the data entries).

## M2 data decisions (chosen, after seeing option tables)

- Solar: Utrecht rooftop PV (Visser et al. 2022, Zenodo, CC BY 4.0). Load: UCI ElectricityLoadDiagrams20112014
  (Trindade 2015, CC BY 4.0).
- Years paired by calendar: train UCI 2012 + 2013 H1 ↔ Utrecht 2014 + 2015 H1; validation UCI 2013 H2 ↔ Utrecht 2016 H2;
  test UCI 2014 ↔ Utrecht 2017. **Solar and load share no real weather.** Train was moved to 2014-2015 because the Utrecht
  data have large gaps from Sep 2015 to Apr 2016 (decided from data availability only, before seeing any power values of
  validation or test). Gaps are never filled; windows with missing solar are excluded.
- Load scale refers to the feeder total peak on the training set (scale × 3.715 MW).
- Placement: PV at buses 14, 17, 29, 32 (3.9 MW total); batteries at buses 17, 32 (1.0 MW total, 2 h of energy).
- **PV was changed from 2.6 to 3.9 MW** after a no-control check on the TRAINING part only, before any controller existed:
  at 2.6 MW no training step exceeded 1.05 pu in either scenario.
- dt = 5 min, H = 24 (2 h) at first. Split: 18 months train / 6 months validation / 12 months test.
- Extreme ramp: top 1% of |r_t| of total feeder PV, threshold computed on the training set only.

## RQ2 wording (chosen, 2026-11-04)

- Old: "... reduce voltage violations during extreme solar-ramp events ...".
- New: "... improve voltage margin and reduce voltage violations during extreme solar-ramp events ...".
- Reason: extreme ramps and violations rarely coincide in this data. On training data with no controller, 90 of 1567
  extreme-ramp steps have a violation at ×0.7 and 25 of 1567 at ×1.0. The proposal document itself is not edited.
- Optional extension, documented and NOT implemented: a separate hard-event scenario pairing real strong-ramp days with
  real low-load days, reported apart from the main regimes, with selection rules fixed on training data.

## RQ2 metrics (2026-11-04, before any controller existed; definitions: option proposed)

Implemented in `src/dpc_grid/metrics.py`, values in `configs/m2.json`.

- Step margin: m_t = min over all buses of min(V − 0.95, 1.05 − V), in pu. Negative means a violation.
- Ramp window: the extreme-ramp step t and the next 11 steps (60 min, forward only). Every extreme-ramp step starts its own
  window; windows may overlap. Only fully valid, consecutive windows inside one split are used; the number excluded is
  reported. 60 min is long enough for a controller to act after the ramp and shorter than the 2 h horizon.
- Primary, over all windows, both regimes: (1) worst margin, min of m_t over the window (higher is better);
  (2) maximum deviation, max |V − 1| over buses and the window (lower is better); (3) time to the safe band, minutes from t
  until all buses are inside 0.96-1.04 pu (0 = already inside; if never, censored at 60 min, with the censored share
  reported).
- Secondary: violations at the extreme-ramp step itself (count of steps, of distinct days, and share).
- Uncertainty: 95% percentile interval from a bootstrap over whole days (2000 samples, seed from the config), because
  extreme ramps cluster within a day.
- Status: not yet computed for any controller.

## Data assumptions (option proposed unless marked "chosen")

Numeric rules were set from training data or before seeing data. Values: `configs/m2.json`, `src/dpc_grid/data.py`.

**Load (UCI).**
- A client is kept if at most 1% of its training values are exact zeros (313 of 370 kept).
- Clients are assigned to buses at random with seed 0, 9-10 per bus. Bus load = plain sum of client kW, so one client can
  dominate a bus (largest client share in a bus: median 39%, max 82% on train).
- Each bus group is divided by its training mean and multiplied by its case33bw P. One common factor sets the feeder total
  training peak to 3.715 MW at ×1.0 ("feeder total peak" basis: chosen). Bus peaks reach 0.85-1.87 of their case33bw
  values on train; the validation peak is 3.815 MW at ×1.0.
- Q = P × the constant Q/P of the bus in case33bw (the data contain no reactive power).
- 15-minute values are held constant over their three 5-minute steps.

**Solar (Utrecht).**
- File `filtered_pv_power_measurements_ac.csv` (the authors' quality-controlled data).
- Normalised by `estimated_dc_capacity` (W/Wp), then multiplied by the bus PV capacity.
- A system is kept if it reports on at least 90% of training steps (121 of 175 kept).
- Systems are assigned to PV buses at random with seed 0, 30-31 per bus. Bus PV uses only the systems reporting at a step.
- A 5-minute value needs at least 3 valid minutes. A step is valid only if each PV bus has at least 5 systems reporting.
- Gaps are never filled. Total PV and total battery capacity are split equally between their buses (totals: chosen).

**Time.**
- Time stamps mark the END of an interval; UTC is the base clock.
- UCI labels are treated as the START of 15 minutes (inferred, not documented: the abnormal hours on clock-change days
  carry labels 01:00-01:45, and Portugal changes the clock at 01:00).
- March: the abnormal hour is removed. October: the hour holds the sum of two hours, so each row becomes two rows of half
  the value (the doubling was seen in the training data, as the UCI description says).
- Utrecht 1-minute labels are treated as the END of the minute (undocumented; matters by at most 1 minute).
- Scenario clock: UCI in UTC, Utrecht in UTC+1, so solar noon aligns within minutes. February 29 is removed.

**Ramps and windows.**
- r_t is in MW per minute on total feeder PV; NaN at the first step, at gaps, at a new split and next to missing data.
- The 99th percentile uses all valid training steps including night, so about 2% of daytime steps are labelled extreme.
- Seed 0 everywhere.

**PV size and placement.** Placement schemes A, B and C were proposed as options; scheme B was chosen, using a table
that assumed midday load = 0.3 × scenario load. That assumption proved wrong (real midday load is
about 0.66 of peak). 2.6 MW (chosen, 2026-11-04) was replaced by 3.9 MW (chosen, 2026-11-04) after the training-only check.

## M3 design (2026-11-05; 24 options proposed and chosen)

Values in `configs/m3.json`.

1. Policy: MLP, 2 hidden layers of 128, Tanh, 4 outputs (P and Q of each battery), scaled by the 0.5 MW limit.
2. Inputs: SoC, V at the battery buses on the previous step, total load, PV at the 4 PV buses, latest r_t, previous action,
   time of day as sin/cos.
3. No forecast of the future. Adding one requires a new dated decision.
4. Exogenous inputs normalised with training mean and std only.
5. Inverter: |P| and |Q| ≤ 0.5, S ≤ 0.5 MVA per battery.
6. Battery: 1 MWh each, efficiency 0.95, SoC limits 10-90%, training starts uniform in 30-70%.
7. H = 24 steps (chosen).
8. A terminal soft penalty on SoC away from 50%, which is NOT in proposal eq. (16).
9. Starting weights: α_L = 1, α_V = 1, α_B = 0.01, α_SoC = 100, α_V,g = 1000, α_C = 0, α_D = 0, changed only using validation.
10. C-rate term off: 0.5 MW on 1 MWh never exceeds 0.5 C.
11. No extra loss term for margin, to avoid training directly on the metric.
12. AdamW, lr 1e-3, grad clip 1.0, batch 64, up to 1000 steps, validation every 100.
13. Solver: 8 iterations in training, 15 in evaluation.
14. One seed first, then 3, and all seeds are reported.
15. One policy for both regimes: each window draws ×0.7 or ×1.0 at random; results reported per regime.
16. 30% of training windows contain an extreme ramp, the rest are uniform. This bias is stated with the results.
17. Model selection by the eq. (16) loss on validation, not by an RQ2 metric.
18. Degradation off (DPC-no-deg, chosen).
19. Ablations without degradation: no-deg, penalty-only, projection-only, no-constraints, labelled "degradation off".
20. Without projection, actions are not clipped in evaluation, and the share of inverter-limit violations is reported.
21. References: no battery action AND a rule-based local Volt-VAR droop (chosen).
22. Evaluation runs the loop continuously, carries SoC forward, and resets SoC to 50% after a data gap.
23. Solver vectorised.
24. Git repository, local only.

Additions (2026-11-05): a short pilot comes before any full run, with its curve and every loss term shown;
the test part is not run until a separate, explicit decision; every run logs its config, seed and git commit.

Implementation details (option proposed, 2026-11-05):
- Terminal SoC penalty weight 10, on (SoC − 0.5)² at the last step of the horizon, averaged over batteries.
- Volt-VAR droop, set before any evaluation and not tuned at that point: each battery uses V at its own bus on the previous
  step; Q = 0 inside 0.99-1.01 pu, rising linearly to the full 0.5 Mvar at 0.95 and 1.05 pu (inject when V is low); P = 0.
- At the first step of a window, "previous V" comes from a solve with idle batteries, and the previous action is 0.
- SoC, V and actions use fixed scaling: (SoC − 0.5)/0.25, (V − 1)/0.05, action / 0.5 MW.

## SoC-penalty selection rule (written 2026-11-05, BEFORE the check; rule chosen)

- Candidates: terminal SoC weight 10, weight 0, weight 0.1, and a 30-70% dead zone (no penalty inside 30-70%, weight 10 on
  the squared distance outside). Seed 0, variant no-deg, one commit.
- Check: continuous loop on the VALIDATION part, 15 solver iterations, both regimes. References: no battery action and the
  droop. RQ2 metrics are not computed.
- Rule: among the candidates that pass (a) and (b), choose the lowest validation objective.
  (a) Active power is used: mean |P| ≥ 0.025 MW (5% of the limit) for each battery in each regime.
  (b) No SoC drift: every end-of-day SoC (the step stamped 00:00 that closes a scenario day, when valid) is inside 20-80%,
  both batteries, both regimes.
  If no candidate passes, nothing is selected and the next step is decided separately.
- Validation objective: mean over valid steps of the eq. (16) terms only (losses, voltage deviation, effort, SoC penalty,
  voltage penalty, configured weights), then the mean of both regimes. The terminal term is left out because it differs
  between candidates. ((a), (b) and this objective were made precise.)

## Documented change: SoC physically bounded (2026-11-05)

- Reason: a simulation defect. The weight-0 candidate reached SoC −0.062 and 1.077, because the limits were only soft
  penalties and the safety layer did not know the stored energy.
- Change: the safety layer limits P with a differentiable clamp so that SoC stays inside [0.10, 0.90] within each step,
  using the SoC equation with its efficiencies. The P, Q and S inverter limits stay; Q is not limited by SoC.
- This is a bug fix made BEFORE any new results. All results obtained before this fix (the 200-step candidates and the
  pilot runs of 2026-11-05) are SUPERSEDED: they are not reported, and their files are not published.
- It goes beyond proposal eqs. (4)-(7), which define only inverter limits. The soft SoC penalty stays in the loss.
- The limit is part of the safety layer, so ablation variants without the safety layer have no energy limit; their share
  of steps outside the SoC limits is reported as physically impossible operation.
- Rerun: all four candidates retrained for 1000 steps (not 200), seed 0, with the fixed safety layer. Rule unchanged.

## Experiment 1 result (H = 24, 2026-11-05), as it is

- The rule selected NO candidate. Weight 0 failed (b): SoC ended every day at 10%. Weights 10 and 0.1 and the dead zone
  failed (a); the dead zone missed by one battery in the stress regime (0.0228 MW).
- Voltage was regulated mostly with Q (|Q| about 10-20 times |P|); the inverter was at its S limit on 43-79% of steps.

## Experiment 2 (pre-specified 2026-11-05, ADDED AFTER seeing Experiment 1)

- Reason: in Experiment 1 the policy used almost no active power. Hypothesis (not tested): with a 2 h horizon and no
  forecast, training cannot see why it should store energy for later, while Q helps the voltage at no energy cost. A
  longer horizon may give P a purpose. Experiment 1 stays reported.
- Horizons H = 72 (6 h) and H = 144 (12 h); everything else as in Experiment 1. Same four SoC-penalty variants at each
  horizon (8 candidates). Inputs unchanged, NO forecast.
- Instability rule, defined before running: a run is unstable if any loss or gradient is not finite, or if the gradient
  norm before clipping is above 10 on more than 5% of steps after step 100. Then it is repeated once with truncated BPTT
  (gradients stop every 24 steps); both runs are reported; if that is unstable too, the candidate fails.
- Same selection rule (a), (b) and objective, over all 8 candidates. The continuous loop is the same for every horizon.

## Experiment 2 result (2026-11-06), as it is

- All 8 runs were stable; no truncated rerun.
- The rule selected **H144_dead_zone_30_70** (objective 0.00870), the only candidate that passed (a) and (b). It passes (a)
  by a very small margin: lowest mean |P| 0.02504 MW against the 0.025 MW threshold. H72_dead_zone missed (a) (0.0225 MW in
  stress). Weight 0 had the lowest objective at both horizons but failed (b): SoC ended almost every day at 10%.
- Voltage is still regulated mostly with Q: |Q| about 14-20 times |P|, at the S limit on 53-77% of steps.
- Confirmed on 2026-11-06 as the outcome of the pre-specified rule. Any claim that P is used is MARGINAL.
- Descriptive diagnostics (cannot change the selection; validation, 110 full days): a small daily cycle, charging around
  9-14 h and discharging around 15-21 h; mean SoC about 0.31 → 0.53 → 0.33; daily swing 0.23-0.25; throughput about
  0.5 MWh/day per battery (0.25 equivalent full cycles); SoC rests near 0.30-0.33 overnight.

## Seed reporting rule (written 2026-11-06, BEFORE seeds 1 and 2)

- H144_dead_zone_30_70 is trained again with seeds 1 and 2, with exactly the Experiment 2 settings. Every seed is shown.
- NO re-selection after seeing the seeds; no seed is chosen by its result. If a seed fails (a) or (b), the use of P is
  stated as not robust across seeds.

## Seed result (2026-11-06), as it is

- Condition (a) passes for 1 of 3 seeds, (b) for 3 of 3. The per-seed minimum of mean |P| is 0.02504 (seed 0), 0.02300
  (seed 1, battery 2) and 0.02499 (seed 2) MW. **The use of P is NOT robust across seeds.**
- Otherwise very similar across seeds: objective 0.00869-0.00870, VVR 0 in both regimes, end-of-day SoC 0.29-0.35.

## Difference from proposal eq. (16)

Any candidate with a nonzero terminal weight or the dead zone adds a term that is NOT in eq. (16). Every candidate at this
stage also differs from the full eq. (16) because degradation (α_D) and C-rate (α_C) are off.

## Volt-VAR droop tuning (protocol written 2026-11-06, BEFORE the run; confirmed)

- Reason: the original droop setting (0.01 / 0.05 pu) was never tuned on data, while DPC was trained. To make the comparison
  fair, the droop is tuned once on the TRAINING part only.
- Grid of 16: dead band {0, 0.005, 0.01, 0.02} pu × full-Q point {0.03, 0.04, 0.05, 0.06} pu. Same objective as the DPC
  candidates, mean of both regimes, continuous loop, 15 iterations. Ties (within 1e-9) go to the smaller mean |Q|.
- Also described: the share of steps where Q changes sign.
- Result (round 1): dead band 0, full Q at 0.03 pu (training objective 0.01158). The optimum lies at a corner of the grid.
  The tuned droop oscillates: Q changes sign on 8.7-10.8% of validation steps, against 0% for the original.
- Round 2 (written BEFORE its run, ADDED AFTER seeing round 1): extended grid of 12, dead band
  {0, 0.0025, 0.005} × full-Q point {0.01, 0.015, 0.02, 0.025}; final choice = minimum over all 28 settings; no further
  tuning whatever it shows.
- Round 2 result: none of the 12 settings beats round 1 (training objectives 0.01362-0.01615). The final tuned droop stays
  dead band 0, full Q at 0.03 pu.
- Validation, mean of regimes: original droop 0.01507, final tuned droop 0.01213, DPC seeds 0/1/2 median 0.00870
  (range 0.00869-0.00870). DPC: VVR 0 in both regimes for all seeds.

## Seeds in later comparisons (chosen, 2026-11-06, BEFORE any M4 result)

- Numeric comparisons of DPC use all three seeds (0, 1, 2) of H144_dead_zone_30_70 and report the median and range.
- Single-trajectory plots and example days use seed 0, the seed on which the candidate was selected.

## M4 plan: online MPC baseline (all options proposed and chosen, 2026-11-06, before any M4 code)

1. Model inside the MPC: LinDistFlow (linearised around 1.0 pu, no losses in the voltage equation), convex. The plant is
   still the nonlinear DistFlow with 15 iterations, on the same steps as DPC.
2. Forecast: persistence (the current load and PV held over the horizon), the same information as DPC. Extra variant:
   oracle (exact future data), labelled as an upper bound.
3. H = 144, as the selected DPC.
4. Objective: the eq. (16) terms with the same weights; losses approximated by Σ r (P² + Q²); the same terminal dead-zone
   term.
5. Voltage limits SOFT (squared slack, weight 1000), as in DPC.
6. Inverter limits exact and convex; SoC 10-90% hard.
7. Efficiencies via a charge/discharge split (changed later, see below).
8. Solver: Clarabel in cvxpy, parameterised once (DPP).
9. The MPC action passes through the same safety layer as DPC.
10. No tuning of the MPC.
11. A 2-day pilot first; the full run only after a separate decision.

**Differences from the proposal, stated wherever MPC is shown:** (i) the MPC uses a convex LinDistFlow model, NOT the
non-convex AC OPF named in the proposal; (ii) the voltage constraints are SOFT. Any speed comparison of DPC is therefore
against a convex MPC.

## MPC pilots and the change of item 7 (2026-11-06)

- First pilot (efficiency model): optimal on all 576 steps per regime; median solve 0.31-0.32 s per step. Simultaneous
  charge and discharge on 30.7% (×0.7) and 23.9% (×1.0) of battery steps, so the relaxation was NOT tight.
- Overlap diagnostic (rule written BEFORE it ran; rule chosen): mean overlap min(P_ch, P_dis) 0.129 MW (×0.7) and
  0.096 MW (×1.0), above the 0.01 MW trigger, so a fixed penalty (1e-5) was added and the pilot re-run once. It changed
  almost nothing (overlap 0.131 / 0.098 MW). Hypothesis (not tested): in its own model the MPC burns stored energy to reach
  the terminal dead zone.
- **Change of item 7 (chosen, 2026-11-06, written BEFORE any run of the new MPC):** the MPC uses a LOSSLESS SoC model (one
  signed P per battery, no efficiency inside the MPC); the plant keeps efficiency 0.95. Motivated ONLY by the relaxation
  artefact, not by any comparison with DPC (none existed). Cost: the MPC's SoC model differs from the plant by at most
  0.0043 SoC per step. The earlier pilots are superseded.
- Pre-specified pass criteria for the lossless pilot: optimal on ≥ 99.9% of steps per regime; no charge/discharge split
  in the model; a one-step SoC mismatch above 0.0043 + 1e-4 counts as a failure. If all pass, the full run starts.
- Oracle plan (chosen, 2026-11-06): exact future load and PV for all 144 steps (last known value held where a future step
  is invalid; share reported). Every 4th full validation day (28 days), SoC reset to 0.5 each day; on exactly these days:
  oracle MPC, persistence MPC, DPC seeds 0/1/2, both droops, no battery.

## M4 result (2026-11-06/07, one commit, no resume), as it is

- Lossless pilot: all criteria pass in both regimes.
- Full persistence MPC on all 48,217 validation steps per regime: optimal on every step; safety layer changed nothing;
  median solve 0.34-0.37 s per step.
  Objective: MPC 0.00529 (×0.7) and 0.01296 (×1.0); DPC seeds 0/1/2 0.00497-0.00498 and 0.01241-0.01242; tuned droop
  0.00808 and 0.01619. **On this objective the persistence MPC is worse than DPC in both regimes.**
  VVR: MPC 0 (×0.7) and 0.039% = 19 steps (×1.0); DPC 0 in both.
- Oracle subset (28 days): oracle MPC (UPPER BOUND) 0.00379 / 0.01078; DPC 0.00484-0.00486 / 0.01251-0.01253; persistence
  MPC 0.00511 / 0.01296; tuned droop 0.00827 / 0.01646; original droop 0.01022 / 0.02001; no battery 0.01528 / 0.20154.
- Reading, descriptive only: on validation DPC lies between the persistence MPC and the oracle bound. One network, one
  untuned convex MPC, validation only.

## M3 ablations (protocol written 2026-11-07, BEFORE any ablation run)

- Base configuration for every variant: exactly as the selected candidate (H = 144, dead-zone terminal term, 1000 steps,
  …). Degradation and C-rate off in all variants. The selected candidate (penalties + projection) is the reference.
- penalty-only-no-deg: the WHOLE safety layer is removed; the loss keeps the soft SoC and voltage penalties.
  projection-only: the soft SoC and voltage penalties are removed; the safety layer stays.
  no-constraints-no-deg: both removed.
- Without the safety layer the policy output is used as it is. The MLP output layer is linear, so P, Q and SoC are not
  bounded. Nothing is clipped. Steps breaking the inverter or SoC limits are reported as physically impossible operation.
- Seeds 0, 1, 2 for each variant (9 runs). Same stability rule. Validation only; all steps; RQ2 metrics not computed.
- Decisions D1-D5 (options proposed and chosen on 2026-11-07):
  D1. penalty-only gets a soft penalty on inverter-limit excess, relu(|P| − P_max)² + relu(|Q| − Q_max)² + relu(S − S_max)²,
      weight 1000 like the voltage penalty. **This is a change from the original code**: without it, "penalty-only" would
      not penalise the constraints that the projection enforces, and the ablation would mostly measure unlimited reactive
      power. It enters the loss only in penalty-only and is not part of the common comparison objective.
  D2. Model selection by each variant's own training objective; the common objective is reported for comparison.
  D3. The terminal dead-zone term is kept in all variants.
  D4. A 200-step pilot of each variant before the full runs.
  D5. The ablations enter the future test run as secondary results, all 3 seeds.
- Pilots (200 steps, seed 0): all stable.

## Ablation result (2026-11-07, one commit, no resume), as it is

- All 9 runs stable; no truncated rerun.
- Common objective, median [min-max] over seeds, ×0.7 / ×1.0:
  selected 0.00497 [0.00497-0.00498] / 0.01241 [0.01241-0.01242];
  projection-only 0.00498 [0.00498-0.00500] / 0.01243 [0.01242-0.01243];
  penalty-only (D1) 0.00507 [0.00503-0.00508] / 0.01260 [0.01254-0.01262];
  no-constraints 0.00458 [0.00458-0.00460] / 0.00908 [0.00907-0.00910].
- Battery steps breaking an inverter limit (physically impossible): selected and projection-only 0; penalty-only 1.5% /
  14.3%; no-constraints 40.6% / 72.3% (battery 2 mean |Q| up to 0.88 Mvar against the 0.5 Mvar limit). The lower objective
  of no-constraints therefore comes from impossible operation and is NOT achievable performance.
- In all 9 runs: VVR 0, SoC always inside 10-90%, end-of-day SoC 0.29-0.36. Condition (a) fails descriptively in all 9.
- Reading, descriptive only: with the safety layer, removing the soft penalties changes almost nothing; without it, the D1
  penalty keeps violations small but not zero, and the policy without constraints exceeds the inverter rating on most steps.

## Site v1 (decisions S1-S8, options proposed and chosen, 2026-11-07)

- Site v1 is built ONLY from existing validation results, labelled PRELIMINARY, with the test part stated as sealed.
- S1. The public repository is a new repository with one initial commit; the private working history stays private.
- S2. The working notes are not published; this file is their public version.
- S3. Jupyter Book; the landing page follows the order of the reference site; notebooks 01-07 read committed results only.
- S4. Licences: MIT for code, CC BY 4.0 for text and figures; the datasets keep their CC BY 4.0 licences and are cited.
- S5. No headline numbers in the page header until the test run.
- S6. Ablations are included if finished (they are).
- S7. Before publishing, two diagnostics are rerun at a clean commit.
- Done on 2026-11-07: the descriptive P and SoC diagnostics and the MPC overlap diagnostic were rerun at a clean commit
  (results/m3_exp2_diagnostics_rerun.json and results/m4_mpc_overlap_diagnostic_rerun.json). The first is numerically
  identical to the earlier run; the second differs by at most 1.2e-10 (solver rounding). The earlier files were made while
  the code differed from its commit; they are superseded and not published.
- S8. The repository and site go under the project's own GitHub account.

## Related work

- Safarzadeh Ravajiri, Drgoňa, Mehrtash, Hobbs (2026), "End-to-End Battery Dispatch with Exact Rainflow Degradation via
  Mixed-Integer DPC", arXiv 2609.12968. Their work is single-battery price arbitrage with no network or voltage; this
  project is network-level voltage regulation with P and Q control. Degradation is a secondary term here, and no novelty is
  claimed there.

## Open questions

- Degradation model (start simple: SoC / C-rate polynomial; rainflow as an extension).
- Deep reinforcement learning baseline.
