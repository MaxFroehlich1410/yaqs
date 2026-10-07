# Fine-tuning every depth candidate: results and interpretation

**The depth search improves average noisy fidelity for both B and D, at roughly
6–7 times the full CPU cost of the corresponding first-90% method.** D delivers
the highest average fidelity and slightly smaller final circuits than B in this
search, but takes about twice B's CPU time. Improvements are not universal.

The experiment covers all **nine six-qubit fixture targets**, optimizer seeds
300 and 400, pruning B/D, and both existing IBM/Google logical noise profiles:
**72 searches, 417 attempted candidates, 413 completed candidates, and 72 valid
selected circuits**. Training uses 64-amplitude statevectors; final assessment
uses independent 64-by-64 Qiskit density matrices. The random-MPS targets are
loaded as dense vectors; simulation does not use MPS truncation.

## What changed

- Removed the 90% ideal-fidelity early exit inside construction and between depths.
- Saved every noiseless, unpruned checkpoint, with 100 online plus 100 batch
  updates at each depth. Subsequent growth always uses this noiseless parent.
- Pruned a separate copy with B or D, then fine-tuned every visited candidate
  with the original **100 frozen training histories and exact empirical no-jump
  compression**, using the unchanged 100 online plus 100 batch noisy updates.
- Used **32,768 separate validation histories per completed candidate**. Stopping
  requires three extensions without a material improvement over the last material
  record: 0.001 plus twice the combined validation standard error. The depth cap
  remains 16. The highest validation score wins, even if its improvement is too
  small to reset patience. Selection is locked before exact density scoring.

Pruning thresholds, repair policy, optimizer rates, final-iterate retention,
noise laws, training seeds and reference compilation were preserved. B is the
simultaneous exact-deletion policy; D is sequential exact deletion with ten
noiseless repair updates after each removal. See the
[frozen protocol](topdown_depth_search_protocol.md) and
[concise mathematical specification](topdown_state_preparation_method.md).

## Paired fidelity and computation

Each mean includes 18 target/seed cases. The old method was **rerun** for fresh,
construction-inclusive timings. All 72 old endpoints reproduced the archived
parameters exactly, with identical training banks: maximum parameter error 0.

| Noise profile | Pruning | Old mean fidelity | Depth-search mean | Gain      | Improved cases | Old CPU | Search CPU | Cost ratio |
| ------------- | ------- | ----------------- | ----------------- | --------- | -------------- | ------- | ---------- | ---------- |
| IBM           | B       | 86.656%           | **90.214%**       | +3.557 pp | 16/18          | 7.26 s  | 45.42 s    | 6.26x      |
| IBM           | D       | 89.167%           | **92.040%**       | +2.873 pp | 16/18          | 13.42 s | 95.41 s    | 7.11x      |
| Google        | B       | 82.745%           | **85.143%**       | +2.398 pp | 12/18          | 7.44 s  | 44.59 s    | 5.99x      |
| Google        | D       | 84.161%           | **87.194%**       | +3.033 pp | 17/18          | 14.68 s | 90.92 s    | 6.19x      |

CPU includes the full visited construction prefix, pruning/repair, noisy training,
sampling/compression and validation, including discarded and failed candidates.
Shared construction is charged in full to each standalone search. Exact final
assessment, compilation and general checkpoint bookkeeping are separate; validation
bank persistence is included. The comparison is **not compute matched**.

The number of selected circuits reaching at least 90% noisy fidelity changed:
IBM B **6→11**, IBM D **11→12**, Google B **3→10**, Google D **9→10**, out of 18.

## Every target

Values average the two optimizer seeds. Each cell gives the new exact noisy
fidelity followed by its change from the corresponding old B or D method in
percentage points. The [full measurement report](../output/topdown-depth-search-q6-20260914-v2/REPORT.md)
also lists old values, all 72 individual endpoints, and phase timings.

| Target             | IBM B          | IBM D           | Google B       | Google D        |
| ------------------ | -------------- | --------------- | -------------- | --------------- |
| Gaussian           | 96.37% (+1.20) | 96.77% (+1.58)  | 94.19% (+0.22) | 94.48% (+0.49)  |
| TFIM ferromagnetic | 90.01% (−3.23) | 92.88% (+14.88) | 86.44% (−2.89) | 87.90% (+13.82) |
| TFIM critical      | 95.78% (+7.34) | 96.31% (+4.37)  | 92.79% (+6.56) | 93.33% (+3.58)  |
| TFIM paramagnetic  | 95.60% (+7.32) | 96.82% (+1.12)  | 93.47% (+6.69) | 94.71% (+1.20)  |
| Haar 1             | 79.60% (+0.73) | 83.98% (+0.94)  | 69.93% (−1.28) | 75.72% (+3.44)  |
| Haar 2             | 81.66% (+4.79) | 84.79% (+0.43)  | 72.32% (+2.77) | 76.53% (+1.77)  |
| Haar 3             | 82.38% (+3.01) | 85.24% (+0.98)  | 72.43% (+0.70) | 76.15% (+1.82)  |
| Random MPS, bond 2 | 95.71% (+6.50) | 96.45% (+0.46)  | 93.05% (+5.79) | 93.52% (−0.03)  |
| Random MPS, bond 3 | 94.81% (+4.35) | 95.12% (+1.09)  | 91.67% (+3.01) | 92.40% (+1.21)  |

B gains most on critical/paramagnetic TFIM and the low-bond MPS targets. It
loses on ferromagnetic TFIM under both profiles, and on Haar 1 under Google.
Haar targets remain substantially harder than the structured targets.

D's large ferromagnetic gain partly repairs a known poor baseline: for seed 400,
IBM improves **63.03%→91.96%** and Google **60.64%→85.89%**. Excluding the entire
ferromagnetic target, D still gains **1.37 pp** under IBM and **1.68 pp** under
Google. Its overall mean therefore should not be read as a uniform three-point
improvement on every target.

## Final circuit resources and depth selection

These are mean resource counts of the selected circuits, not of all candidates.

| Profile | Pruning | Logical 2Q old → new | Compiled CX old → new | Compiled 2Q depth old → new | Mean selected / visited depth index |
| ------- | ------- | -------------------- | --------------------- | --------------------------- | ----------------------------------- |
| IBM     | B       | 16.39 → 23.00        | 32.56 → 45.56         | 16.00 → 23.22               | 4.17 / 6.56                         |
| IBM     | D       | 21.17 → 20.94        | 42.33 → 41.89         | 21.33 → 21.67               | 2.83 / 5.39                         |
| Google  | B       | 16.39 → 21.06        | 32.56 → 41.78         | 16.00 → 21.22               | 3.44 / 6.11                         |
| Google  | D       | 21.17 → 19.06        | 42.33 → 38.11         | 21.33 → 20.00               | 2.56 / 5.11                         |

One construction depth index adds a complete even/odd brickwall cycle (45 scalar
gates at six qubits). It is distinct from compiled two-qubit depth. No search hit
the depth-16 cap; the largest visited index was 11. Selected indices ranged from
1 to 9 for B and 1 to 6 for D.

**33 of 72 winners came from noiseless parents below 90% ideal fidelity.** This
supports admitting such parents to the search. It does not mean their final ideal
fidelity stayed below 90%: subsequent pruning/repair and fine-tuning changed their
parameters; only two selected final circuits had ideal fidelity below 90%.

All visited candidates and both Pareto frontiers are in
[candidates.json](../output/topdown-depth-search-q6-20260914-v2/candidates.json).
The four recorded quantities are logical 2Q count, compiled 2Q depth, final ideal
fidelity and exact noisy fidelity. We also report the frontier that treats ideal
fidelity as diagnostic and optimizes only resources/noisy fidelity. Frontiers are
retrospective and never replace the validation-selected winner.

## The main optimization limitation

**Removing the 90% early exit exposes substantial instability in the unchanged
noiseless optimizer.** Of 124 distinct construction checkpoints, 64 end more than
10 percentage points below a fidelity reached earlier during the same 200 updates.

For ferromagnetic TFIM, seed 400, depth 3, ideal fidelity is **99.533%** after
the 100 online updates, peaks at **99.542%** on update 102, and ends at **7.037%**
after the 100 batch updates. For Haar 3, seed 400,
depth 10, the corresponding peak is **99.840%**, with a final value of **12.070%**.

The old first-90% exit often stopped before these later updates could damage the
state. The new experiment preserves the final-iterate policy and rates to avoid
silently changing the optimizer as well. Consequently, the measured depth curves
combine circuit expressivity, noise, pruning and optimization failures. They are
**not a clean measurement of the best physically attainable fidelity at each depth**.
Three-step patience can also miss improvements beyond the visited depths.

This is the strongest next issue to isolate in a separate experiment: retaining
the best noiseless iterate, or stabilizing the batch step. Neither change was
introduced into these results.

## Trajectories, independent evaluation and reliability

- Every successful candidate used the original 100-history weighted fine-tuning.
  The 413 successful banks contain **41,300 raw histories**, compressed to
  **5,626 representatives**: an average of **13.62 simulated histories per bank**.
  This preserves the empirical weights exactly; it is not a measured 7.34x
  wall-time speedup.
- Four additional banks, totaling **400 histories**, failed the unchanged
  zero-jump guard. They are the depth-1 D candidates for both ferromagnetic TFIM
  seeds under both profiles, each reduced to six gates. They were not resampled;
  the next depth was considered. No circuit was pruned to zero gates.
- Validation used **13,533,184 independent raw histories**, compressed to
  **1,612,851 representatives**. All 413 successful training banks and all 413
  validation banks contained jumps. Final density evaluation integrates every
  modeled noise outcome without using a sampled test bank.
- Validation selected the exact-best visited candidate in **67/72 searches**.
  The largest loss from choosing by validation rather than retrospective exact
  ranking was **0.1541 percentage points**. The mean selected validation standard
  error was **0.1367 pp**; the largest absolute validation-versus-exact discrepancy
  was **0.5064 pp**. Finite validation sampling remains a source of uncertainty.
- Every validation bank and all 417 stopping decisions were replayed. Validation
  replay error was zero; independent Qiskit trajectory checks agreed within
  **3.4e-16**. Vector/density ideal-fidelity agreement was within **1.6e-15**,
  and compiled-unitary residuals within **1.3e-13**. Baseline parameter and bank
  parity passed for all 72 controls.

The existing profile labels represent logical simulation coefficients, not
calibrated IBM/Google device predictions. Resource compilation uses the same
fixed CX reference as before; the noise fidelity is evaluated on the logical
circuit. The largest sampler-law versus full-Lindblad fidelity difference across
completed candidates was **0.015 percentage points**.

## Practical interpretation

**Use B for the cheaper depth search; use D when higher fidelity and smaller
deployed circuits justify more offline computation.** Within the new search,
D gains **1.83 pp** over B under IBM and **2.05 pp** under Google, at **2.10x**
and **2.04x** B's CPU cost. D wins those paired comparisons in 17/18 and 15/18
cases, respectively, and has fewer mean compiled CX gates in both profiles.

The search is an improvement in average fidelity, not in computational efficiency
relative to fine-tuning one final circuit. B spends most of its extra CPU on
construction and noisy fine-tuning; D spends most on sequential pruning/repair.
Validation accounts for only about 2–6% of the new search CPU.

These are descriptive results on nine previously used targets and two optimizer
seeds. There is no compute-matched control and no untouched-target confirmation.
The extra budget and removal of the inner construction cutoff contribute alongside
depth selection. Further depth increases should wait for an isolated study of the
noiseless optimizer's loss of good intermediate solutions.

## Artifacts and reproduction

- [Full measurement report, all 72 paired endpoints and plots](../output/topdown-depth-search-q6-20260914-v2/REPORT.md)
- [Per-search results and costs](../output/topdown-depth-search-q6-20260914-v2/searches.json)
- [All candidates and Pareto frontiers](../output/topdown-depth-search-q6-20260914-v2/candidates.json)
- [Complete audit](../output/topdown-depth-search-q6-20260914-v2/audit.json)
- [Frozen manifest](../output/topdown-depth-search-q6-20260914-v2/manifest.json)

Run the search with `uv run python -m experiments.topdown_depth_search --output NEW_DIRECTORY --workers 4`.
Generate the report with `uv run --with matplotlib python -m experiments.topdown_depth_analysis --output OUTPUT_DIRECTORY --workers 4`.
The driver uses the authenticated old B/D checkpoints as its baseline reference.

A serialization-only correction converted a NumPy Boolean to a Python Boolean
before writing the selection record. The numerical code and protocol were
unchanged; 458 authenticated checkpoint/bank files were reused, with provenance
in [recovery.json](../output/topdown-depth-search-q6-20260914-v2/recovery.json).
Combined active execution took approximately **19.4 minutes** with four workers,
including baseline replay and exact assessment, excluding debugging and the
separate audit/report run. Per-search CPU counts include the reused stages once.

Verification: **49 targeted tests passed**, including independent Qiskit channel
and trajectory checks, raw-sample standard errors, unchanged fine-tuning parity,
checkpoint checksums, failed-bank accounting and stopping/selection behavior.
The new Python files pass targeted Ruff and `ty` checks. Repository-wide
`uvx nox -s lint` remains blocked by existing repository errors and missing
plotting dependencies in the separate lint environment; its full log is retained
with the experiment.
