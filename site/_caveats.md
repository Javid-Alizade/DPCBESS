:::{admonition} Read these results with the following caveats
:class: warning

- **Preliminary, validation part only.** The test part is sealed and has not been evaluated. Every number comes from one
  network (IEEE 33-bus) and from the validation part, which was also used to select the DPC policy, so the DPC numbers are
  optimistically biased.
- **Data.** Solar (Utrecht, 2014-2017) and load (Portugal, 2012-2014) are real measurements, but they come from different
  places and years and share no real weather. The PV size of 3.9 MW was chosen after a no-control check on the training part
  only (at 2.6 MW no training step exceeded 1.05 pu).
- **RQ2 was reworded** on 2026-11-04, before any controller existed, because extreme ramps and voltage violations rarely
  coincide in this data (see {doc}`Decisions </decisions>`).
- **Training bias.** 30% of the training windows contain an extreme solar ramp.
- **Differences from the proposal's eq. (16).** Degradation and C-rate terms are off in every variant ("degradation off").
  A terminal SoC term with a 30-70% dead zone is added and is not in eq. (16).
- **The safety layer goes beyond the proposal**: it also limits P by the stored energy (SoC kept within 10-90%), not only by
  the inverter limits of eqs. (4)-(7). It does not guarantee voltage feasibility.
- **Active power use is marginal and not seed-robust.** Voltage is regulated mostly with reactive power Q; the selected
  policy passes the pre-specified "P is used" threshold on 1 of 3 seeds.
- **The MPC baseline** is a convex LinDistFlow MPC with soft voltage limits and a lossless SoC model, not the AC-OPF MPC
  named in the proposal, and it was not tuned. Speed comparisons are therefore against a convex MPC. The oracle MPC uses
  exact future data and is an upper bound, not a real controller.
- **Ablations.** "penalty-only" includes a soft inverter-limit penalty that was added on purpose (decision D1). Variants
  without the safety layer can break the inverter limits; such steps are physically impossible operation and are reported
  as shares, not as achievable performance.
- **Corrections to the proposal**: a discharging battery enters the net load with a minus sign, and the left side of
  eq. (9) is P, not 2P. Results computed before the SoC energy limit was added (2026-11-05) are superseded and not shown.
:::
