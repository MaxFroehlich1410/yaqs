# Experiment 1 results: exact-noisy versus noiseless Adam

Completed 2026-09-07 under `yaqs.exact_adam_headroom.v1`, following the
[predeclared protocol](state_preparation_exact_adam.md). All **16 continuations /
1600 Adam updates** finished. The measured experiment interval, including
preflights, worker-pool startup, training and checkpoint evaluations, was
**16.94 seconds on eight single-threaded workers**; initial imports/source loading,
implementation, tests, and the subsequent audit are outside that interval.

## Main finding

**Noise-aware parameter optimization has a real but modest accessible benefit
in this fixed-circuit, fixed-budget diagnostic.** Mean exact noisy fidelity is
**77.768830%**, versus **77.362375%** after matched noiseless Adam and
**75.970853%** at the frozen start.

- Exact-noisy minus noiseless Adam: **+0.406455 percentage points**.
- Exact-noisy minus frozen start: **+1.797977 percentage points**.
- Noiseless Adam minus frozen start: **+1.391522 percentage points**.
- Exact-noisy beats noiseless Adam in **7/8 paired runs and all four family means**.
- Exact-noisy beats the frozen start in all eight runs.

The predeclared practical screen required at least **0.5 percentage points**
target-balanced improvement and positive gains in at least three families.
The family criterion is met, but **the magnitude criterion is not**. Do not
relabel this as passing the screen or as a state-of-the-art result. The data do
rule out the blanket interpretation that this noise model offers no useful
parameter-refinement opportunity from these starts.

## Matched conditions

- Four fixed six-qubit targets, seeds 11 and 29; all starts are the previously
  saved `H25_lr0p7:cold_final` checkpoints after 570 construction/repair updates.
- Identical starting parameters in both arms, independently zeroed Adam moments.
- 100 additional updates, constant learning rate 0.005, beta1=0.9,
  beta2=0.999, epsilon=1e-8; no hyperparameter selection or restarts.
- Every circuit remains at 78 one-qubit / 30 two-qubit gates and 108 parameters.
  No new pruning, compilation, gate deletion, or angle clipping.
- Complex128 exact density-matrix adjoint in both arms. The only objective
  difference is the training channel: zero noise versus calibrated local
  Pauli-Lindblad noise, with complete-window average errors 0.00064 / 0.0051.
- The independent Qiskit density implementation evaluates both fidelities at
  updates 0,10,...,100. The final iterate is the primary result, not the best
  intermediate checkpoint.

This is **Adam**, not a new noisy hybrid-Krotov experiment. The initial circuits
were constructed with noiseless hybrid Krotov and pruning; neither continuation
uses trajectories, common random numbers, or a cross-trajectory estimator.
The model remains a homogeneous hardware-motivated abstract-gate proxy, not a
native-device hardware experiment with coherent, readout, or idle errors.

## Target-level noisy fidelities

All fidelities are percentages. Average the two initialization seeds within
each target, then weight the four targets equally. Gains are percentage points.

| Target                |  Frozen start | Noiseless Adam | Exact-noisy Adam | Noisy minus noiseless |
| --------------------- | ------------: | -------------: | ---------------: | --------------------: |
| Gaussian              |     81.859787 |      81.945785 |        82.436033 |             +0.490248 |
| Critical TFIM         |     81.410414 |      82.102237 |        82.480390 |             +0.378152 |
| Haar random 1         |     58.786038 |      63.705584 |        63.806178 |             +0.100593 |
| Bond-2 random MPS     |     81.827173 |      81.695893 |        82.352718 |             +0.656825 |
| **Equal-target mean** | **75.970853** |  **77.362375** |    **77.768830** |         **+0.406455** |

Every paired difference, including the unfavorable Haar initialization:

| Target / seed      | Noisy minus noiseless, pp | Noisy minus frozen, pp |
| ------------------ | ------------------------: | ---------------------: |
| Gaussian / 11      |                 +0.329575 |              +0.492169 |
| Gaussian / 29      |                 +0.650922 |              +0.660323 |
| Critical TFIM / 11 |                 +0.331830 |              +0.767217 |
| Critical TFIM / 29 |                 +0.424475 |              +1.372734 |
| Haar random 1 / 11 |             **-0.139753** |              +2.429754 |
| Haar random 1 / 29 |                 +0.340939 |              +7.610526 |
| Bond-2 MPS / 11    |                 +0.651419 |              +0.492073 |
| Bond-2 MPS / 29    |                 +0.662231 |              +0.559017 |

There is no trajectory-evaluation uncertainty: these are exact channel
expectations. That does not remove initialization variability or establish
generalization. There is still only one fixed state per family, and the same
development states were used in earlier adaptive hybrid-parameter screens.
No significance or target-population confidence claim is made.

## Ideal-fidelity tradeoff and convergence

| Target                | Frozen ideal fidelity, % | Noiseless Adam ideal, % | Exact-noisy Adam ideal, % |
| --------------------- | -----------------------: | ----------------------: | ------------------------: |
| Gaussian              |                99.408465 |               99.752739 |                 99.601990 |
| Critical TFIM         |                98.964467 |               99.899143 |                 99.832527 |
| Haar random 1         |                74.602526 |               81.093262 |                 80.601740 |
| Bond-2 random MPS     |                99.507978 |               99.823046 |                 99.660775 |
| **Equal-target mean** |            **93.120859** |           **95.142047** |             **94.924258** |

Noise-aware refinement accepts **0.217790 pp lower mean ideal fidelity** than
noiseless Adam while attaining higher noisy fidelity. Most notably, both MPS
noiseless continuations improve ideal fidelity but lose noisy fidelity relative
to their frozen starts (mean **-0.131280 pp**); noisy refinement instead gains
**+0.525545 pp** over the frozen MPS mean. This directly demonstrates why ideal
fidelity alone is an insufficient parameter-selection objective here.

Changes in exact noisy fidelity during updates 80 to 100:

| Target            | Noiseless Adam, pp | Exact-noisy Adam, pp |
| ----------------- | -----------------: | -------------------: |
| Gaussian          |          +0.003228 |            +0.047943 |
| Critical TFIM     |          +0.022987 |            +0.079802 |
| Haar random 1     |          +0.850761 |            +0.706189 |
| Bond-2 random MPS |          -0.018612 |            +0.092104 |

The Haar runs are **not converged** at the cutoff: noisy seed 29 alone gains
1.138877 pp during the last 20 updates. Its final noisy-loss gradient norm is
0.026089, versus 0.002041-0.003775 for the six structured-target noisy runs.
Thus the Haar outcome mixes unfinished approximation/optimization with noise
sensitivity; it does not identify a fundamental noise ceiling. All final noisy
gradient norms are nonzero, and no convergence tolerance was a stopping rule.

Some Adam training traces have small uphill transitions (0-4 per run among
successive recorded pre-update losses); the implementation imposes no monotonic
line search. There were no nonfinite updates or physicality-check failures.

## Cost and scientific conclusion

Summed per-arm measured training intervals were 24.66 worker-seconds for
noiseless Adam and 27.78 for exact-noisy Adam. Summed checkpoint/final-diagnostic
evaluation intervals were 24.38 and 19.48 worker-seconds, respectively. These
are concurrent wall intervals, not process CPU-time measurements. The different
evaluation timings are not a scientific speed ranking: both arms use the same
evaluation procedure, and concurrent execution/warm-up affects timings.
Observed worker-process high-water RSS was **150.5-160.4 MiB**, including imports
and dense-adjoint storage; it is not incremental optimizer memory.

This establishes a concrete, modest noisy-refinement reference at fixed circuit
resources. It does **not** demonstrate a Krotov advantage, a trajectory memory
advantage, device performance, a global optimum, or superiority over published
state-preparation methods. In particular, the +1.797977 pp improvement over the
frozen start must not all be attributed to noise awareness: matched noiseless
continuation already provides +1.391522 pp.

A small subsequent trajectory-versus-exact diagnostic would now have a measured
signal to resolve, approximately 0.4 pp on average. The missed practical screen
and unfinished Haar convergence do not justify automatically launching the full
confirmation/scaling matrix. No additional learning rates, longer continuations,
noise models, or Experiment 2 runs were added after seeing these results.

## Verification and reproducibility

- **36 new tests passed**, covering all eight gate generators, shared parameters,
  nonunit scales/offsets, reversed and nonadjacent sites, disabled/zero noise,
  invalid inputs, Adam weighted moments/bias correction/input immutability,
  interrupted resume, corrupt checkpoints, identical starts, and serial/parallel
  reproducibility. The complete optimization suite passed **361 tests in 93.39 s**.
- Before training, all eight selected starts reproduced their saved ideal/noisy
  fidelities; all 16 objective/directional-gradient checks passed. The maximum
  central-difference discrepancy was **9.715e-11** (directional epsilon=1e-5).
- The post-run audit authenticated **36 JSON records**, recomputed all **176
  saved evaluation checkpoints / 352 fidelity values** and all 16 final gradient
  norms. Maximum discrepancies were **0.0** in this environment. It independently
  reconstructed the aggregates, checked all 1600 update counters, and confirmed
  moment-preserving completed resume leaves the result unchanged.
- Source manifests and all eight selected source-record checksums were unchanged
  after the experiment. No previous results were overwritten.
- All applicable hooks on the changed/new files passed, including type checking
  with the project environment. Required `uvx nox -s lint` was run, but full-repo
  lint still fails on pre-existing manuscript/plotting/MPS-test issues, the broken
  Homebrew Node `simdjson.28.dylib` dependency, and missing plotting imports in the
  lint environment. Unrelated automatic hook edits were reverted.

Implementation: [exact density adjoint](../benchmarks/state_preparation/exact_density.py),
[runner](state_preparation_exact_adam.py), and
[tests](../tests/optimization/test_state_preparation_exact_adam.py).

Artifacts are at
`/Users/maximilianfrohlich/Desktop/GitHub/yaqs/output/state-preparation-exact-adam-20260906/`:
`manifest.json`, `preflight.json`, `result.json`, `audit.json`, the JUnit test
report, and each target/seed's copied start, two Adam checkpoints, and result.
The directory label retains the protocol's start date; completion was September 7.
Use the protocol's launch command with `--resume` for an integrity-checked no-op
on this completed run. Do not change code/configuration and reuse its manifest.

Checksums use canonical JSON excluding the checksum envelope:

| Artifact  | SHA-256                                                            |
| --------- | ------------------------------------------------------------------ |
| Manifest  | `2cc9beed1e8a981448f12056fb0ec85846a37db1e6d36757bf3af4df92049c3f` |
| Result    | `190693f3b16940ff2dd23eb6301fcf782aa1826d5ddcbbaddceba2f8fdc17c18` |
| Preflight | `4939cc3813b2fd8cbc91b57a416ad118f2aee9bc2e6f56a437bda0bfd52d06fb` |
| Audit     | `289395ab030fce199b708ac09eca81633bf5c289092c3e4563d014bda0c2f4a0` |
