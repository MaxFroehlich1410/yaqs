# Hybrid construction extension: results

Completed 2026-09-06 under `yaqs.compression_hybrid_construction_extension.v1`.
The [predeclared extension](state_preparation_hybrid_extension.md) added only
25/30 online updates at rates 0.7/0.9, as clarified by the author. All 32
new jobs completed in **255.7 seconds (4.3 minutes)** on eight workers.

## Main finding

**25 online updates at rate 0.7 per growth stage is the strongest setting
in this screen:** mean exact noisy fidelity **75.970853%**, versus
**74.926143%** for the previous 20/0.5 setting and **71.502713%** for batch-only.
The additional gain is **1.044710 percentage points** over the old best,
or **4.468140 points** over batch-only. All settings keep the same update
and final gate budgets.

The new best improves **7/8 target/seed pairs and 3/4 target-family means**
over the old best, but it is not uniformly better: the bond-2 MPS mean
falls slightly. All four new settings improve all eight pairs over batch-only.
Increasing the online duration or rate further does **not** improve the mean
consistently. This is evidence against blindly extending both boundaries.

## Complete comparison

Fidelities are percentages; gains are percentage points. Each fixed target
has equal weight, with seeds 11 and 29 averaged within it. Old B and H20/0.5
are authenticated stored references, not newly executed jobs. All numbers
use the final iterate after 570 updates, not a best intermediate checkpoint.

| Online updates / rate | Noisy fidelity, % | Noiseless fidelity, % | Gain vs old best, pp | Better pairs vs old best |
| --------------------- | ----------------- | --------------------- | -------------------- | ------------------------ |
| Batch reference       | 71.502713         | 87.674909             | -3.423431            | 0/8                      |
| 20 / 0.5 (old best)   | 74.926143         | 91.875935             | +0.000000            | reference                |
| 25 / 0.7              | 75.970853         | 93.120859             | +1.044710            | 7/8                      |
| 25 / 0.9              | 75.034839         | 91.973822             | +0.108696            | 5/8                      |
| 30 / 0.7              | 74.894414         | 91.745143             | -0.031729            | 5/8                      |
| 30 / 0.9              | 75.209904         | 92.055304             | +0.283760            | 5/8                      |

The strongest setting has 100 online and 470 batch updates overall: each of
four depth-growth stages uses 25 online updates at 0.7 and 75 batch updates
at 0.2. The unchanged six pruning rounds each have 20 batch repair updates
at 0.1; final continuation has 50 batch updates at 0.1. The 30-prefix settings
replace another 20 batch updates with online updates; they do not receive
additional iterations. Every final circuit retains 78 one-qubit and 30
two-qubit gates, including the protected initial U3 layer.

## Target-level results

Noisy fidelity in percent, averaged over the two initialization seeds:

| Setting             | Gaussian  | Critical TFIM | Haar random 1 | Bond-2 random MPS |
| ------------------- | --------- | ------------- | ------------- | ----------------- |
| Batch reference     | 80.709135 | 80.216672     | 44.003510     | 81.081533         |
| 20 / 0.5 (old best) | 81.401798 | 80.660929     | 55.731317     | 81.910528         |
| 25 / 0.7            | 81.859787 | 81.410414     | 58.786038     | 81.827173         |
| 25 / 0.9            | 81.747974 | 81.772646     | 54.760734     | 81.858003         |
| 30 / 0.7            | 81.749623 | 81.780025     | 54.205318     | 81.842691         |
| 30 / 0.9            | 81.956382 | 81.866132     | 55.125278     | 81.891823         |

For 25/0.7 versus 20/0.5, the family gains are +0.457989 pp (Gaussian),
+0.749484 pp (TFIM), +3.054721 pp (Haar), and -0.083355 pp (MPS).
The Haar state contributes about 73% of the summed family gains. Both Haar
seeds improve at 25/0.7, whereas both deteriorate at each of the other three
new settings. Conversely, 30/0.9 is strongest here on Gaussian and TFIM.
The best average is not the best setting for every family.

All paired changes relative to the stored 20/0.5 endpoint, in percentage points:

| Target / seed      | 25 / 0.7  | 25 / 0.9  | 30 / 0.7  | 30 / 0.9  |
| ------------------ | --------- | --------- | --------- | --------- |
| Gaussian / 11      | +0.462180 | +0.665318 | +0.542592 | +0.570531 |
| Gaussian / 29      | +0.453798 | +0.027034 | +0.153056 | +0.538637 |
| Critical TFIM / 11 | +0.621007 | +0.773221 | +0.858062 | +0.868255 |
| Critical TFIM / 29 | +0.877961 | +1.450211 | +1.380128 | +1.542149 |
| Haar random 1 / 11 | +2.742154 | -0.974709 | -2.789602 | -1.018106 |
| Haar random 1 / 29 | +3.367287 | -0.966457 | -0.262396 | -0.193972 |
| Bond-2 MPS / 11    | -0.254456 | -0.367096 | -0.349006 | -0.280771 |
| Bond-2 MPS / 29    | +0.087745 | +0.262045 | +0.213331 | +0.243360 |

## Why more aggressive is not always better

The stronger prefixes do accelerate early training. The table gives mean
noiseless training fidelity in percent at selected fixed diagnostics, before
any pruning:

| Setting             | Depth 1, update 20 | Depth 1, update 100 | Depth 4, update 100 |
| ------------------- | ------------------ | ------------------- | ------------------- |
| Batch reference     | 13.148664          | 46.723082           | 86.302450           |
| 20 / 0.5 (old best) | 20.754230          | 55.172458           | 90.424641           |
| 25 / 0.7            | 28.705744          | 69.349010           | 92.602504           |
| 25 / 0.9            | 41.115450          | 71.906301           | 91.868051           |
| 30 / 0.7            | 28.705744          | 70.858407           | 90.796112           |
| 30 / 0.9            | 41.115450          | 73.383771           | 91.800499           |

Although 30/0.9 has the strongest mean after the first growth stage, 25/0.7
has the strongest mean after the fourth. Thus the final ranking difference
already appears **before pruning**; it cannot be attributed solely to gate
removal. These nonconvex runs follow different parameter paths and may also
prune different gates. The observations are consistent with path dependence,
but do not by themselves identify a particular local minimum or causal mechanism.

All recorded parameters and diagnostics were finite. There were no fidelity
decreases larger than 1e-10 between consecutive recorded online updates in
the new growth-stage traces. This check excludes the comparison from each
block's initial state to its first recorded update. There is no observed
numerical blow-up that would explain the weaker final means. Monotonic
progress within a run does not imply that a larger rate or longer online
prefix will find a better final solution.

## Noise interpretation and limits

This is still **noiseless hybrid construction plus magnitude pruning and
batch-only repair**, followed by exact evaluation under the calibrated noise
channel. It is not a new noisy-trajectory-training result. The noise model
and rates are unchanged: local Pauli-Lindblad channels calibrated to complete
one-/two-qubit window errors 0.00064/0.0051. Training uses untruncated MPS;
the small-system density-matrix oracle is evaluation-only. No old warm-0 or
warm-N refinements were rerun or combined with these new cold endpoints.

Use 25/0.7 as the provisional development setting. Before expanding the grid
again, the more informative checks are a **matched batch-only 25-step
rate-0.7 prefix**, followed by rate 0.2, and confirmation on fresh targets
and more seeds. The first separates learning-rate effects from online gate
ordering; the second tests whether the observed gain generalizes. These
controls have not been run here. This remains one fixed state per family
and two seeds, with adaptive tuning on the same development data, not a
held-out superiority or state-of-the-art claim.

## Reproducibility and verification

The post-run audit verified:

- All **386 JSON checksums**, 32 distinct jobs, 352 training blocks, 64 endpoints,
  18240 actual updates (3520 online), and 1824 executed checkpoint chunks.
- Every source-stage, repair, and final block signature, reconstructing initial
  parameters, circuit growth, exact phase boundaries, per-step learning rates,
  paired seed keys, and nonempty batch tails.
- Every pruning score, deletion choice, original gate ID, and final gate count.
  Recomputed pruning fidelities and both endpoint fidelities agree exactly
  with their saved values in this environment (maximum difference 0.0).
- The final aggregate recomputes exactly in the saved job order. No failed
  job or unfavorable setting was omitted.
- Both reference settings share the same source checksums, targets, seeds,
  environment, calibrated noise, and construction budgets. The core runtime
  source fingerprint is unchanged; the runner revision adds the explicit
  construction-only mode, whose engineering tests reproduce the full runner's
  corresponding cold endpoints exactly.

The complete optimization suite passed all **325 tests** in 118.25 seconds.
The focused hybrid-runner suite also passed all **19 tests**, covering the new
25/30 boundaries at both rates, chunked versus uninterrupted noiseless/noisy
training, compatible resume, serial/parallel results, unchanged batch controls,
and the four-setting CLI dispatch. Changed files pass their applicable
format/lint/type hooks. Full-repository lint still encounters the previously
existing manuscript, plotting, MPS/test, BibTeX Node dependency, and lint-
environment issues; unrelated hook modifications were reverted.

Artifact directories (not committed):

- New: `/Users/maximilianfrohlich/Desktop/yaqs-experiment-results/q6-hybrid-construction-extension-20260906`.
- Stored reference: `/Users/maximilianfrohlich/Desktop/yaqs-experiment-results/q6-hybrid-compression-screen-20260906`.
- Original batch source: `/Users/maximilianfrohlich/Desktop/yaqs-experiment-results/q6-compression-pilot-20260905`.

Checksums use the canonical JSON content digest, excluding the checksum envelope:

| Record             | SHA-256                                                            |
| ------------------ | ------------------------------------------------------------------ |
| Extension manifest | `82258104e377e971c05a2490a30a963716c2e07e035f6241ba976825954419e0` |
| Extension result   | `70902334dd5cdb1d4a2b8469a9e4906af2d113fdf16e46d03f833f4fdef4f276` |
| Reference manifest | `c869b62bd94fd7983290c1a7a65a81a79d0e312f8269daaadef6329c7c3cc138` |
| Reference result   | `23e292dba2a7a463d2b72a1e8499d9a3c8f237946ca7432bcbded28c61756d65` |

The original result directories remain unchanged. See the extension protocol
for the exact launch command; compatible interrupted runs accept `--resume`.
