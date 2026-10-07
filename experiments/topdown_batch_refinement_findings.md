# Batch refinement: completed BB and L-BFGS comparison

**Use L-BFGS for noiseless batch construction with pruning B in the optional
depth search. Keep best-only retention with D.** L-BFGS improves B's mean final
noisy fidelity by **0.556 / 0.483 percentage points** under the historical
IBM/Google profiles, at **21.2% / 11.5% less measured search CPU**. With D, its
fidelity effect is mixed and its search costs more. BB improves noiseless
optimization but does not improve mean final noisy fidelity against either
current primary baseline.

These are results for all nine six-qubit fixture targets, optimizer seeds
300/400, pruning B/D and both noise profiles: **288 completed searches** plus
**496 identical-input construction optimizations**. Training uses dense
statevectors; final assessment uses exact density matrices. Random-MPS targets
are also simulated as dense vectors. The cheaper first-90% pipeline was not
rerun, and the online and noisy optimizers were not changed.

[Frozen protocol](topdown_batch_refinement_protocol.md) ·
[All target/seed results, resources and phase costs](../output/topdown-batch-refinement-q6-20260919/COMPARISON.md) ·
[Machine-readable comparison](../output/topdown-batch-refinement-q6-20260919/comparison.json)

## 1. What changed, and which baseline was used?

The prior completed study supplies the primary baseline: **best retention plus
backtracking for B**, and **best retention alone for D**. Both controls were
freshly rerun with both pruning methods. All 36 control target/seed cases
reproduce archived parameters, training banks, validation and selections
exactly. We do not compare the new methods only against the obsolete,
unprotected final-iterate optimizer.

All four settings use the same 100 online sweeps, followed by 100 batch updates,
and return the best actual construction parameters, including iteration zero.

| Setting      | Noiseless batch update                                        |
| ------------ | ------------------------------------------------------------- |
| Best only    | Existing fixed step 0.7; retain the best checkpoint           |
| Backtracking | True loss gradient; Armijo search restarted at 0.7            |
| BB           | True loss gradient; safeguarded BB1 initial step, then Armijo |
| L-BFGS       | Limited-memory inverse-Hessian direction, then Armijo         |

BB uses $s^Ts/(s^Ty)$ from consecutive accepted batch states and true gradients,
with positive-curvature checks and initial proposals clipped to $[10^{-6},10]$.
L-BFGS keeps ten positive curvature pairs, skips unsuitable pairs and resets
non-descent directions. Both reuse the current gradient during line search.
Armijo uses $c_1=10^{-4}$, halving and at most 13 trials. This is an Armijo
L-BFGS implementation, not SciPy L-BFGS-B or strong-Wolfe L-BFGS. Curvature
history starts empty at each online/batch transition.

Pruning remains B/D with cutoff 0.01. D repair remains five online plus five
fixed-rate batch updates, returning the final iterate. Noisy fine-tuning remains
100 online plus 100 cross-trajectory batch updates at rate 0.2, with **M=100**
frozen, unconditional histories and exact empirical no-jump compression.
Separate 32,768-history validation banks, patience three, depth cap 16 and
selection-before-exact-evaluation are unchanged. The new optimizers apply only
to the noiseless batch phase, where the implemented signal is the true gradient.

## 2. Construction improves on identical inputs

The screen replays 124 saved inputs from the latest backtracking-grown pipeline.
Every arm receives the same starting vector and has an identical online prefix.
These are not the earlier stability study's original-grown fixed inputs; even
matching target/seed/depth labels do not make those parameter vectors identical.

| Batch policy | Mean returned ideal fidelity | Optimizer CPU vs backtracking | Gain at backtracking's batch CPU budget |
| ------------ | ---------------------------- | ----------------------------- | --------------------------------------- |
| Best only    | 92.111%                      | 0.878x                        | —                                       |
| Backtracking | 92.886%                      | 1.000x                        | —                                       |
| BB           | 94.393%                      | 0.964x                        | +1.323 pp                               |
| L-BFGS       | **96.022%**                  | **0.883x**                    | **+3.113 pp**                           |

Both new methods improve all 124 equal-update endpoints against both controls.
At each control's measured batch CPU budget, both win 123/124 cases; the one
small loss comes from truncating before a beneficial final update. The timing
comparison uses completed checkpoints without interpolation. It is a descriptive
analysis of measured timings, not a second optimization run.

All arms perform 12,400 true batch-gradient evaluations. Additional batch-only
forward evaluations are **24,402 for backtracking, 23,189 for BB and 12,922 for
L-BFGS**. Rejected trials fall from 12,002 to 10,789 and 522, respectively.
This reduced trial work supports the timing advantage. No safeguarded batch
step decreases fidelity, and no line search fails. Best-only still has 69/124
raw peak-to-final losses exceeding ten percentage points; retention protects
its returned states.

The means above weight checkpoints equally. Giving each target equal weight
after averaging depths within each seed still gives positive gains for all
nine targets: **+1.263 pp for BB and +2.666 pp for L-BFGS** versus backtracking.
L-BFGS reaches at least 99% ideal fidelity on 101/124 inputs, versus 81 for BB
and 72 for backtracking. This is finite-budget performance, not a guarantee of
convergence or global optimality.

![Construction traces and CPU curves](../output/topdown-batch-refinement-q6-20260919/batch_refinement_traces.png)

The trace examples use the new inputs described above; the historical
99.542% to 7.037% collapse is not being replayed in this figure.

## 3. Final noisy fidelity and complete search cost

Each fidelity averages the same 18 target/seed cases. CPU includes all visited
construction, pruning/repair, noisy training and validation work, including
discarded and failed candidates. Shared work is charged in full to each
hypothetical standalone search; summing branch costs duplicates that work and
is not actual suite CPU or parallel wall time. Exact final assessment,
compilation and general checkpoint bookkeeping are excluded from search CPU.
All timings below are fresh measurements from this run.

| Construction                        | Pruning | IBM fidelity | Google fidelity | Mean search CPU, IBM / Google |
| ----------------------------------- | ------- | ------------ | --------------- | ----------------------------- |
| Best only                           | B       | 90.793%      | 85.771%         | 39.99 / 44.89 s               |
| Backtracking — prior B baseline     | B       | 91.235%      | 86.179%         | 39.38 / 42.52 s               |
| BB                                  | B       | 91.115%      | 86.011%         | 35.19 / 37.81 s               |
| **L-BFGS**                          | **B**   | **91.790%**  | **86.662%**     | **31.02 / 37.64 s**           |
| **Best only — retained D baseline** | **D**   | **92.094%**  | **87.470%**     | **73.35 / 83.73 s**           |
| Backtracking                        | D       | 92.148%      | 87.272%         | 77.67 / 88.16 s               |
| BB                                  | D       | 91.780%      | 86.839%         | 81.03 / 77.90 s               |
| L-BFGS                              | D       | 92.172%      | 87.137%         | 85.37 / 89.06 s               |

Against the predeclared current baseline for each pruning method:

| New policy | Pruning/profile | Mean gain     | Median gain | Wins / losses | Search CPU ratio |
| ---------- | --------------- | ------------- | ----------- | ------------- | ---------------- |
| BB         | B / IBM         | -0.120 pp     | +0.024 pp   | 10 / 8        | 0.894x           |
| BB         | B / Google      | -0.168 pp     | -0.158 pp   | 7 / 11        | 0.889x           |
| BB         | D / IBM         | -0.314 pp     | -0.288 pp   | 5 / 13        | 1.105x           |
| BB         | D / Google      | -0.631 pp     | -0.318 pp   | 5 / 13        | 0.930x           |
| L-BFGS     | B / IBM         | **+0.556 pp** | +0.378 pp   | 13 / 5        | **0.788x**       |
| L-BFGS     | B / Google      | **+0.483 pp** | +0.129 pp   | 10 / 8        | **0.885x**       |
| L-BFGS     | D / IBM         | +0.078 pp     | -0.049 pp   | 9 / 9         | 1.164x           |
| L-BFGS     | D / Google      | -0.333 pp     | -0.218 pp   | 7 / 11        | 1.064x           |

L-BFGS therefore offers B the best observed fidelity/classical-compute tradeoff.
It does not improve every case: its worst B losses are 0.598 pp under IBM and
1.079 pp under Google. For D, a tiny IBM mean gain does not offset worse Google
fidelity, more compute and mixed paired results. Even against D's secondary
backtracking control, L-BFGS adds only 0.024 pp under IBM and loses 0.136 pp
under Google.

BB is approximately 11% cheaper for B, but sacrifices mean fidelity and circuit
resources. It is not an overall method improvement. Comparing BB only with
B's weaker best-only control would misleadingly suggest a fidelity improvement;
the declared primary B baseline is backtracking.

![Noisy fidelity and full search CPU](../output/topdown-batch-refinement-q6-20260919/batch_refinement_overview.png)

In the right panel, I/G denote the IBM/Google logical profiles and B/D identify
pruning. Full searches are not compute matched: new policies can visit different
depths and follow different parameter histories. The whole four-worker run,
including the fixed screen and final assessments, took 3,481.65 seconds
(58.03 minutes); this wall time is separate from standalone search costs.

## 4. Two-qubit resources and why construction gains need not survive noise

Values are means for selected circuits. Construction depth is not compiled
two-qubit depth, and compiled resources are not the circuit used for noise
evaluation: noise is applied to the logical gates under the frozen model.

| Construction/pruning | Logical 2Q, IBM / Google | Compiled CX, IBM / Google | Compiled 2Q depth, IBM / Google |
| -------------------- | ------------------------ | ------------------------- | ------------------------------- |
| Backtracking / B     | 20.72 / 19.89            | 41.44 / 39.67             | 20.33 / 19.11                   |
| BB / B               | 21.33 / 20.78            | 42.67 / 41.56             | 21.56 / 20.89                   |
| L-BFGS / B           | 21.06 / 19.78            | 42.11 / 39.56             | 21.11 / 20.00                   |
| Best only / D        | 19.06 / 18.33            | 38.11 / 36.67             | 19.22 / 18.44                   |
| Backtracking / D     | 19.94 / 18.61            | 39.89 / 37.22             | 20.33 / 18.56                   |
| BB / D               | 20.72 / 19.61            | 41.33 / 39.11             | 20.78 / 19.56                   |
| L-BFGS / D           | 20.28 / 19.17            | 40.56 / 38.33             | 20.00 / 18.67                   |

L-BFGS with B is not strict Pareto dominance: compiled two-qubit depth rises
slightly even though mean noisy fidelity improves and classical CPU falls.
Mean selected construction depth falls from 4.28 to 2.67 under IBM and from
3.61 to 2.44 under Google; pruning and compilation explain why shallower
construction does not imply lower final compiled depth.

Three post-hoc case studies illustrate the downstream tradeoff:

- **L-BFGS helps B:** Haar 1, seed 400, Google selects depth 2 instead of 7.
  Final ideal fidelity is lower, 90.096% versus 96.295%, but logical two-qubit
  count falls from 48 to 26. Exact noisy fidelity rises from 70.325% to
  **75.134%**, a 4.809 pp gain. L-BFGS also improves the common depth-2 candidate;
  this is not explained solely by selecting a different depth.
- **L-BFGS does not help D:** Haar 2, seed 300, Google selects depth 3 in both
  methods. L-BFGS raises final ideal fidelity from 96.066% to 99.046%, while
  logical two-qubit count rises from 29 to 40. Exact noisy fidelity falls from
  78.496% to **75.249%**. Neither a large repair collapse at the selected
  checkpoint nor validation selection error explains this example.
- **BB's largest losses:** Haar 1, seed 300 loses 2.155/2.840 pp with B and
  2.231/3.280 pp with D under IBM/Google. D selects depth 3 in both methods and
  attains almost identical final ideal fidelity, but BB retains 38 rather than
  31 logical two-qubit gates. B also retains more gates; collective pruning
  damage occurs in both pipelines and is not uniformly worse for BB.

These are descriptive associations, not isolated causal tests of gate count.
The [BB case audit](../output/topdown-batch-refinement-q6-20260919/bb_case_haar_random_1__seed300.json)
and [L-BFGS case audits](../output/topdown-batch-refinement-q6-20260919/lbfgs_partial_scope_case_studies.json)
record stage scores, resources and selection details. The L-BFGS examples were
selected from a frozen 16/18-case snapshot, not claimed to be extrema of the
completed suite. Full-pipeline parents diverge after the first depth, whereas
the construction screen holds every initial vector fixed.

## 5. Audits, remaining limitations and recommendation

- All 288 searches have valid endpoints; none hits the depth cap. Of 1,622
  candidates, 1,610 finish and 12 fail the unchanged zero-jump training-bank
  guard. No circuit is emptied by pruning. Failed banks are not resampled or
  forced to contain jumps; their raw histories are not saved by the frozen
  runner, so their status is recorded without claiming a raw-bank audit.
- Successful training banks contain 161,000 raw histories, compressed to
  21,260 representatives, including 19,650 jump histories. Every successful
  training bank contains jumps. Separate validation includes 52,756,480 raw
  histories and 6,014,123 jump histories; all validation banks contain jumps.
- Exact final density propagation occurs after validation selection is locked.
  It averages all modeled noise outcomes, including jumps, without reusing a
  training bank. It has no trajectory-sampling error; validation sampling can
  still influence which circuit wins.
- The full construction audit verifies 463 checkpoints and 10,812 artifact
  hashes, with zero saved-parameter fidelity replay error. All 33,800
  safeguarded full-pipeline batch updates are nondecreasing, with no exhausted
  line searches. Source and input hashes, reference parity, trajectory weights,
  validation regeneration and selection decisions pass the analysis audits.
- D repair remains unstable. Repairs losing over ten percentage points from
  their post-deletion input number 367/9,590 for best-only, 484/10,813 for
  backtracking, 348/9,519 for BB and 364/10,118 for L-BFGS. These count different
  visited circuits and are not a controlled comparison of repair policies.
  All 1,610 noisy-training final scores equal their recorded post-update
  maxima; the initial noisy score and best-iterate parameter history are not
  saved, so no broader retention/generalization claim follows.
- The 88 targeted/reused optimizer tests pass, as do scoped lint, formatting
  and typing checks. Repository-wide lint remains blocked by repository and
  local dependency/tooling issues, including the lint environment's missing
  optional plotting dependency; see the saved [check logs](../output/topdown-batch-refinement-q6-20260919/checks/).
- The profiles use historical logical simulation coefficients, not calibrated
  hardware error rates. Nine development targets and two optimizer seeds do
  not establish statistical significance or generalization. Small timing
  differences are single-run measurements, not precise universal speedups.

For the optional depth search, carry forward **L-BFGS batch construction plus
best retention with B**, and **fixed-step construction plus best retention
with D**. Retain backtracking/B as the historical control for this completed
comparison. D with best-only still gives higher mean noisy fidelity than the
new B choice, by 0.304 pp under IBM and 0.807 pp under Google, at approximately
2.36x and 2.22x search CPU. Thus B remains the default efficiency choice and D
the more expensive fidelity option. Applying L-BFGS to D repair, noisy training
or the first-90% pipeline would be a separate experiment; none was changed here.
