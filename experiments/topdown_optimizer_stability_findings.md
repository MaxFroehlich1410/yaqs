# Construction optimizer stability: completed comparison

**Best-iterate retention prevents discarding a good noiseless solution. Adding
backtracking also stabilizes the batch updates and gives the best fidelity/cost
tradeoff with pruning B in this experiment. With D, the downstream benefit is
small and inconsistent; its unchanged repair optimizer still exhibits severe
collapses.**

This is the requested three-optimizer by two-pruning comparison: all nine
six-qubit fixture targets, optimizer seeds 300/400 and both historical logical
noise profiles, giving **216 completed searches**. It uses the optional
every-depth search, not the cheaper first-90% pipeline. All training uses dense
statevectors; final fidelity uses exact density matrices. Random-MPS targets
are loaded as dense vectors. Existing production defaults are unchanged.

The [frozen protocol](topdown_optimizer_stability_protocol.md) specifies the
experiment. The [complete measurement report](../output/topdown-optimizer-stability-q6-20260919/COMPARISON.md)
contains every target/seed result, phase timings, resources and paired changes;
[comparison.json](../output/topdown-optimizer-stability-q6-20260919/comparison.json)
contains the machine-readable measurements and audits.

## What the three settings change

| Setting             | Noiseless construction at each depth                                            | Returned checkpoint                                 |
| ------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------- |
| Original            | 100 online + 100 batch updates, rate 0.7                                        | Final iterate                                       |
| Best only           | Exactly the same raw updates                                                    | Highest ideal fidelity, including the initial state |
| Best + backtracking | Same 100 online updates; 100 batch updates with Armijo step rejection/reduction | Highest ideal fidelity, including the initial state |

For $L=1-F_0$ and its noiseless gradient $g$, backtracking accepts the first
$\eta=0.7\,2^{-j}$, $j=0,\ldots,12$, satisfying

$$
L(\theta-\eta g)\leq L(\theta)-10^{-4}\eta\lVert g\rVert^2.
$$

Rejected trials reuse the gradient. An exhausted line search keeps the previous
parameters and records failure. No additional early exit was introduced.

Only construction changes. B remains simultaneous exact gate-removal pruning
with cutoff 0.01. D remains sequential exact gate removal with five online and
five fixed-rate batch repair updates after each removal. Both retain **M=100**
frozen noisy training histories, empirical no-jump compression and weighting,
unchanged noisy fine-tuning, separate validation, and the same depth-search
stopping/selection rules. D repair and noisy fine-tuning still return their
final iterates.

## 1. Does the construction fix work on identical inputs?

Each setting replayed the same 124 archived depth inputs, for 372 optimizations.
Unlike the subsequent full searches, these inputs do not depend on the new
policy's preceding output. The original parameters and complete traces reproduce
the archive exactly; best-only follows that same raw trajectory exactly.

| Setting             | Mean returned ideal fidelity | Raw peak-to-final losses >10 pp | Returned losses from peak >10 pp | Optimizer CPU vs original |
| ------------------- | ---------------------------- | ------------------------------- | -------------------------------- | ------------------------- |
| Original            | 73.025%                      | 64/124                          | 64/124                           | 1.000x                    |
| Best only           | 92.157%                      | 64/124                          | 0/124                            | 1.003x                    |
| Best + backtracking | 92.933%                      | 0/124                           | 0/124                            | 1.098x                    |

These means average archived depths, not selected final circuits. Retention
protects the returned state without fixing the intervening unstable updates.
Backtracking produced **12,400 accepted batch steps with no fidelity decrease**,
8,596 rejected trial steps, and no exhausted line searches. The online phase
was unchanged and does not acquire a monotonicity guarantee from this result.

| Fixed input                           | Original returned | Best only returned | Best + backtracking returned |
| ------------------------------------- | ----------------- | ------------------ | ---------------------------- |
| TFIM ferromagnetic, seed 400, depth 3 | 7.0367%           | 99.5422%           | 99.6652%                     |
| Haar 3, seed 400, depth 10            | 12.0699%          | 99.8396%           | 99.8938%                     |

![Matched construction traces](../output/topdown-optimizer-stability-q6-20260919/stability_traces.png)

Independent Qiskit statevector checks agree within $4.45\times10^{-15}$.
Twenty-four directional finite-difference checks around the online/batch
transition have maximum absolute gradient error $3.83\times10^{-9}$.
The evidence supports an excessive fixed batch step as a cause of the recorded
collapses, rather than an incorrect noiseless gradient in those checked cases.

## 2. What happens to final noisy fidelity and computation?

Each table entry averages the same 18 target/seed cases. All 216 searches have
a valid validation-selected endpoint. CPU includes construction, pruning and
repair, noisy fine-tuning and validation for the full visited search, including
discarded/failed candidates. Shared construction is charged in full to each
standalone branch. Final exact assessment, compilation and general checkpoint
bookkeeping are separate. Timings are fresh measurements from this run.

| Construction        | Pruning | IBM fidelity | Google fidelity | Mean search CPU, IBM / Google |
| ------------------- | ------- | ------------ | --------------- | ----------------------------- |
| Original            | B       | 90.214%      | 85.143%         | 51.65 / 51.76 s               |
| Best only           | B       | 90.793%      | 85.771%         | 48.74 / 54.92 s               |
| Best + backtracking | B       | **91.235%**  | **86.179%**     | **48.20 / 52.18 s**           |
| Original            | D       | 92.040%      | 87.194%         | 103.61 / 100.29 s             |
| Best only           | D       | 92.094%      | **87.470%**     | **89.49 / 102.43 s**          |
| Best + backtracking | D       | **92.148%**  | 87.272%         | 95.52 / 108.51 s              |

For B, backtracking improves mean noisy fidelity over the original by **1.021 pp
(IBM)** and **1.036 pp (Google)**. It improves 14/18 and 12/18 paired cases;
median gains are 0.322 and 0.761 pp. Relative to best-only, it adds 0.442 and
0.408 pp at 0.989x and 0.950x total CPU. Although individual construction
optimizations cost more, changed pruning outcomes and visited candidates can
reduce the full search cost. These adaptive searches are not compute matched,
and small timing differences should not be interpreted as precise speedups.

For D, backtracking adds only 0.107 and 0.078 pp over the original means, while
losing in **9/18 cases under each profile**. Median paired changes are negative
under both profiles. Relative to best-only, its mean change is +0.054 pp under
IBM and -0.197 pp under Google, at about 6–7% more CPU. There is no consistent
downstream advantage for D from this construction change alone.

Regressions matter: backtracking versus original loses 2.025 pp (IBM) and
2.535 pp (Google) for Haar 2, seed 300, pruning D. Versus best-only, that Google
case loses 3.888 pp. B's worst regressions versus original are 0.518 pp for
critical TFIM seed 400 under IBM and 0.907 pp for bond-3 random MPS seed 400
under Google. Better noiseless construction does not guarantee a better result
after discrete pruning, repair, noisy optimization and validation selection.

## 3. Final two-qubit resources

Values are means for selected circuits, not for the candidates visited during
optimization. Construction depth and compiled two-qubit depth are different.

| Construction        | Pruning | Logical 2Q, IBM / Google | Compiled CX, IBM / Google | Compiled 2Q depth, IBM / Google |
| ------------------- | ------- | ------------------------ | ------------------------- | ------------------------------- |
| Original            | B       | 23.00 / 21.06            | 45.56 / 41.78             | 23.22 / 21.22                   |
| Best only           | B       | 21.72 / 20.56            | 43.11 / 40.78             | 21.78 / 20.78                   |
| Best + backtracking | B       | 20.72 / 19.89            | 41.44 / 39.67             | 20.33 / 19.11                   |
| Original            | D       | 20.94 / 19.06            | 41.89 / 38.11             | 21.67 / 20.00                   |
| Best only           | D       | 19.06 / 18.33            | 38.11 / 36.67             | 19.22 / 18.44                   |
| Best + backtracking | D       | 19.94 / 18.61            | 39.89 / 37.22             | 20.33 / 18.56                   |

B with backtracking improves average fidelity and reduces both compiled CX
count and depth relative to either alternative. D with best-only uses fewer
resources than D with backtracking and has a better computation/fidelity
tradeoff except for the latter's very small IBM mean-fidelity advantage.

![Fidelity and computation comparison](../output/topdown-optimizer-stability-q6-20260919/stability_overview.png)

## 4. Remaining instability in D repair

The repair policy was deliberately held fixed, and it still damages good
states. Counting each saved D-pruning path once, independently of its two
hardware branches:

| Construction        | Repair calls | Repairs losing >10 pp from their post-deletion input |
| ------------------- | ------------ | ---------------------------------------------------- |
| Original            | 10,309       | 275                                                  |
| Best only           | 9,590        | 367                                                  |
| Best + backtracking | 10,813       | 484                                                  |

These are descriptive counts over different visited circuits and depths, not
a controlled comparison of repair policies. They do not establish that
construction stabilization makes repair intrinsically worse.

A concrete example is backtracking construction, TFIM ferromagnetic seed 300,
depth 3: after its first gate deletion, ideal fidelity is **98.9957%**. D repair
reaches **99.0798%**, then its remaining fixed-rate batch updates finish at
**0.2389%**. The [saved checkpoint](../output/topdown-optimizer-stability-q6-20260919/pipeline/best_backtracking/tfim_ferro__seed300/search/depth_03_D_pruned.json)
and [repair diagnostic](../output/topdown-optimizer-stability-q6-20260919/repair_diagnostic.json)
record this failure. Stabilized construction therefore does not imply a stable
complete method. Noisy fine-tuning also remains unchanged and is not covered
by the noiseless Armijo guarantee.

## 5. Trajectories, checks and limits

- The successful candidates use **1,248 training banks**, totaling 124,800 raw
  histories, compressed to 16,644 representatives with 15,396 jump-containing
  histories. Each successful bank contains jumps. Twelve additional banks fail
  the unchanged zero-jump guard: four per construction policy. They are recorded
  without resampling or forced events; their searches continue to later depths.
- Separate validation banks contain **40,894,464 raw histories**, including
  4,701,777 jump-containing histories. Every successful candidate has independent
  validation; every validation bank contains jumps. No search hits the depth cap.
- Final noisy fidelity comes from exact density propagation after locking
  validation selection. It integrates all modeled noise outcomes, including
  jumps, rather than reusing a training bank. There is no Monte Carlo error on
  this final score, but finite validation sampling can affect which circuit wins.
- All 18 original pipeline cases reproduce archived construction, pruning,
  noisy parameters, training/validation banks and selection exactly. Source and
  artifact hashes, multiplicities, validation RNG separation, stopping decisions
  and fixed-input checkpoint semantics pass the full analysis audit.
- The 36 new targeted tests pass, as do 50 tests covering reused backend,
  pruning and depth-search code. Repository-wide lint remains blocked by
  existing repository issues and local tooling/dependency issues; see the saved
  [check logs](../output/topdown-optimizer-stability-q6-20260919/checks/).
- These IBM/Google labels are historical logical noise coefficients, not
  calibrated hardware predictions. Compiled CX resources are reported
  separately from the logical-circuit noise simulation. Nine previously used
  development targets and two optimizer seeds do not establish significance
  or generalization to unseen states; depths and searches are correlated.

## Recommendation

Use **best retention plus backtracking with B** as the next construction
baseline for the optional depth search. It resolves the observed construction
collapse and improves average fidelity, circuit resources and the measured
overall tradeoff.

For D, **best retention alone currently offers the better practical tradeoff**;
the small mean fidelity differences do not justify treating backtracking as a
clear downstream winner. D still delivers higher mean noisy fidelity than B,
at roughly twice the search CPU. Preserve both in comparisons.

The next isolated experiment should stabilize **D's repair updates** while
holding construction, pruning cutoff, M=100, noisy fine-tuning and validation
fixed. This comparison has not yet been run. Applying the construction policy
to the cheaper first-90% pipeline also requires a separate check; the current
results should not be presented as measurements of that pipeline.
