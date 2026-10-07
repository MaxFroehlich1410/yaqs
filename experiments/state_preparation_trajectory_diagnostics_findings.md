# Statevector development findings: noisy gradients and a weak batch tail

Completed 2026-09-11 using the unchanged
[development protocol](state_preparation_trajectory_diagnostics.md). These are
exploratory results on four previously used development targets, not a new
confirmation panel or evidence of state-of-the-art performance.

## Outcome and execution decisions

The most useful finding is that **R=32 gradients are usually downhill but not
accurate near the saved endpoints**. Keeping all 50 updates online improved
final fidelity consistently, but only modestly. Reusing noise maps reduced
training cost, with a fidelity penalty. Cross estimation offered a small,
budget-dependent error/cost benefit, not a solution to the gradient noise.

| Planned component                                    | Execution                        | Outcome                                                                                                                          |
| ---------------------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 1. Gradient quality and batch tail                   | Completed                        | 12 checkpoint diagnostics and 32 refinements; all-online won all eight paired comparisons against the 0.1-tail baseline.         |
| 2. CRN refresh policies                              | Completed                        | 24 refinements; refresh-five was cheaper but slightly less accurate; fixed CRN lost more fidelity.                               |
| 3. Cross versus independent                          | Completed as a paired diagnostic | 576 estimates on 288 shared ensembles; no cross refinement was launched.                                                         |
| 4. Adaptive tails, conditional on accurate gradients | Deferred                         | R=32 did not satisfy the qualitative accuracy condition across families. This does not establish that adaptive tails would fail. |
| Optional CRN backtracking                            | Not run                          | Remains a separate, cost-counted follow-up.                                                                                      |

The adaptive decision was made after inspecting the diagnostic and before any
adaptive run. No numerical pass threshold had been predeclared; this is an
explicit qualitative decision, not a retrospectively defined hypothesis test.
No trajectory count, learning rate, target, circuit or budget was changed in
response to the results. The studies ran sequentially, each with up to eight
single-numerical-thread workers.

## Fixed setup and interpretation limits

- Six qubits, 108 retained gates/parameters (78 one-qubit and 30 two-qubit),
  with no additional construction or pruning.
- One old development target per family: `gaussian_mu0p5_sigma0p1`,
  `tfim_critical`, `haar_random_1`, and `random_mps_bond2`.
- Frozen construction replica 11 for each target. Refinements use two paired
  noise replicas, 11 and 29, not two new target states or construction starts.
- Calibrated homogeneous local Pauli-Lindblad noise after each retained gate:
  average gate infidelities `r1=0.00064`, `r2=0.0051`; per-axis integrated
  coefficients approximately `0.00032020497493061084` and
  `0.0010664702651014273`. Exact commuting Pauli-factor sampling, not the
  legacy at-most-one-jump approximation.
- Complex128 statevector training. Exact density-matrix gradients provide the
  diagnostic reference; exact density-matrix fidelity scores refinement after
  every update. Scoring does not determine updates, stopping or acceptance.
- Each refinement starts from the same saved update-0 parameters, draws R=32
  paths per new ensemble, and performs 50 updates. The usual schedule is 25
  online sweeps at 0.7, then 25 independent-gradient batch steps at 0.1.
- Root seed `2026091101`; method-independent paired streams. Diagnostic
  gradients use the old saved hybrid checkpoints at updates 0, 25 and 50, not
  checkpoints selected from the new screens.

There were **56 refinements, 2,800 updates and 2,856 exact refinement scores**.
The eight fresh-map baseline cases were deliberately executed in both the tail
and CRN screens and reproduced exactly. Thus there are 48 distinct refinement
configurations, not 56 independent experimental replicates. Means weight the
four targets and their two noise replicas equally. Endpoints are update 50,
never the best observed iterate. Percentage points are abbreviated pp.

## 1. Gradient quality

Eight independent ensemble repeats were evaluated for each target/checkpoint
and R. Counts are nested prefixes of the same per-repeat random stream;
independent and cross signals use identical maps. The exact reference is the
gradient of the channel-averaged infidelity at those same parameters.

Pooled means across all 12 target/checkpoint settings:

| Estimator   |   R | Relative L2 error | Cosine with exact gradient | CPU per estimate (s) | MSE times CPU (s) |
| ----------- | --: | ----------------: | -------------------------: | -------------------: | ----------------: |
| Independent |   8 |             4.572 |                      0.330 |               0.0156 |       0.000137446 |
| Independent |  32 |             2.506 |                      0.449 |               0.0435 |      0.0000946482 |
| Independent | 128 |             1.265 |                      0.659 |               0.1617 |      0.0000853342 |
| Cross       |   8 |             4.484 |                      0.333 |               0.0165 |       0.000141823 |
| Cross       |  32 |             2.381 |                      0.464 |               0.0461 |      0.0000894932 |
| Cross       | 128 |             1.220 |                      0.675 |               0.1808 |      0.0000876815 |

Independent gradients had a positive dot product with the exact gradient in
85/96, 94/96 and 96/96 estimates at R=8, 32 and 128 respectively. A downhill
direction is weaker evidence than an accurate gradient, and does not guarantee
that a finite learning-rate step improves the exact objective.

The end-of-run diagnostic is particularly important:

| Target family, saved update 50 | R=32 mean relative error | R=32 mean cosine | R=128 mean relative error | R=128 mean cosine |
| ------------------------------ | -----------------------: | ---------------: | ------------------------: | ----------------: |
| Gaussian                       |                    4.701 |            0.337 |                     2.305 |             0.490 |
| TFIM                           |                    4.073 |            0.179 |                     2.217 |             0.316 |
| Haar                           |                    1.178 |            0.673 |                     0.774 |             0.791 |
| Random MPS                     |                    4.774 |            0.299 |                     2.080 |             0.519 |

At saved update 25, the R=32 mean cosine was 0.499 and relative error 1.771;
at update 50 these were 0.372 and 3.681. The corresponding exact gradient norms
at update 50 were approximately 0.00991, 0.01177, 0.01522 and 0.01173, so the
ratios are not artifacts of dividing by a numerically zero reference.

Increasing R from 32 to 128 reduced pooled independent-gradient MSE to about
24.4% of its former value, at about 3.72 times the CPU per estimate. That is
consistent with a substantial sampling-variance limitation. R=128 still did
not produce accurate endpoint gradients for every family. These diagnostics
support investigating variance reduction or trajectory budgeting; they do not
prove that noise is the sole cause of slow optimization.

## 2. Tail-rate and all-online screen

Every variant had identical new-noise-stream prefixes through update 25. Only
the remaining 25 updates differed. CPU includes sampling and optimizer work,
but excludes exact scoring, setup and checkpoint I/O.

| Variant                      | Mean final exact noisy fidelity (%) | Gain during updates 26-50 (pp) | Difference from 0.1-tail (pp) | Paired wins / 8 | Mean training CPU (s) |
| ---------------------------- | ----------------------------------: | -----------------------------: | ----------------------------: | --------------: | --------------------: |
| Hybrid, batch tail 0.1       |                           77.655255 |                       0.034993 |                             0 |               — |                2.6138 |
| Hybrid, batch tail 0.2       |                           77.676392 |                       0.056130 |                     +0.021136 |               7 |                2.6261 |
| Hybrid, batch tail 0.3       |                           77.694438 |                       0.074176 |                     +0.039182 |               7 |                2.6177 |
| All 50 updates online at 0.7 |                           77.768017 |                       0.147755 |                     +0.112762 |               8 |                2.9212 |

Mean endpoints by family, averaging the two noise replicas:

| Family     | Tail 0.1 (%) | Tail 0.2 (%) | Tail 0.3 (%) | All-online (%) |
| ---------- | -----------: | -----------: | -----------: | -------------: |
| Gaussian   |    82.062081 |    82.071891 |    82.080087 |      82.115479 |
| TFIM       |    82.163122 |    82.185377 |    82.204612 |      82.282598 |
| Haar       |    64.441127 |    64.491216 |    64.536111 |      64.700243 |
| Random MPS |    81.954691 |    81.957083 |    81.956941 |      81.973748 |

All-online improved every target/noise-replica pair by 0.0178 to 0.2690 pp over
the 0.1-tail baseline, at approximately 11.8% extra training CPU. It also beat
the 0.3-tail endpoint in every family mean. The improvement is consistent but
small, and Haar remains the least well-prepared family.

This supports retaining all-online as a control and questioning the early
switch to a low-rate batch tail. It does **not** isolate update ordering from
learning rate: the all-online continuation uses 0.7, versus 0.1-0.3 in the
batch tails. Nor does this study compare the new endpoints with a matched
statevector Adam run. It cannot establish a new advantage over Adam.

## 3. CRN screen

The schedule is fixed at 25 online updates at 0.7, then 25 batch updates at
0.1. Only the across-update reuse policy changes. Within-update map replay is
used in all arms. Refresh-five and fixed CRN recompute their entire prefix from
update 0; they do not inherit the fresh-map prefix from the tail screen.

| Policy             | Mean final exact noisy fidelity (%) | Difference from fresh (pp) | Paired wins / 8 | Mean training CPU (s) | CPU relative to fresh |
| ------------------ | ----------------------------------: | -------------------------: | --------------: | --------------------: | --------------------: |
| Fresh each update  |                           77.655255 |                          0 |               — |                2.6309 |                 1.000 |
| Refresh every five |                           77.616142 |                  -0.039113 |               0 |                1.5274 |                 0.581 |
| Fully fixed        |                           77.436009 |                  -0.219246 |               2 |                1.2732 |                 0.484 |

| Family     | Fresh (%) | Refresh-five (%) | Fixed (%) |
| ---------- | --------: | ---------------: | --------: |
| Gaussian   | 82.062081 |        82.001236 | 81.654852 |
| TFIM       | 82.163122 |        82.100642 | 81.840527 |
| Haar       | 64.441127 |        64.411270 | 64.455755 |
| Random MPS | 81.954691 |        81.951421 | 81.792904 |

Each fixed ensemble contained 5-12 sampled Pauli factors and 22-27 factor-free
paths out of 32. Thus **none of the eight fixed ensembles was entirely
noise-free**, unlike one older R=3 example. Factor counts do not necessarily
equal net physical errors because Pauli factors can cancel.

Fixed CRN reduced mean exact fidelity during the batch tail in three families;
the Haar mean improved. Its two endpoint wins against fresh sampling were both
Haar replicas. This is consistent with optimizing a limited frozen ensemble,
but no separate training-versus-test gap was measured here.

Refresh-five lost to fresh sampling in every pair, but by only 0.0008-0.1029 pp,
while saving approximately 42% of measured training CPU. It is therefore a
plausible **cost/fidelity trade-off**, not a fidelity winner. Sampling is charged
in these costs; reusing maps avoids most sampling work. A fixed-CPU comparison
would be needed to establish a budget-normalized advantage. Exact scoring and
shared construction are excluded from these ratios.

## 4. Cross and adaptive decisions

At R=32, cross reduced pooled MSE by about 10.4%, but used about 6.1% more CPU
per estimate. Mean per-setting MSE-times-CPU improved by only 5.4%. The
cost/error product favored cross in 11 of 12 settings at R=32, but only two of
12 at R=128; pooled R=128 cost/error was about 2.8% worse. These are small,
observational differences from eight repeats, not precision speed benchmarks.
Cross should remain a candidate, but there is no compelling reason here to
replace independent averaging in all refinements.

Raw records retain empirical mean-gradient bias and variance. With only eight
repeats, a nonzero error of the mean does not establish asymptotic estimator
bias. The cross signal is not assumed to differentiate the same finite-sample
objective as independent averaging; MSE-times-CPU is descriptive only.

The planned adaptive screen was conditional on gradients being accurate but
progress remaining slow. That premise is not supported across the four
families at R=32, so **momentum, diagonal scaling, Krotov-Adam and its full-Adam
control were not run**. This defers, rather than rejects, those methods.
Backtracking was optional and was also not launched.

The next priority is a small, explicitly budgeted sampling/variance-reduction
comparison, retaining all-online and fresh-map hybrid controls. R=128 is a
useful tested diagnostic point, not an automatically selected new training
default. Keep refresh-five as a possible low-cost control. Any future
competitiveness claim still needs matched Adam and fresh target states.

## Artifacts, accounting and verification

All outputs contain frozen manifests, authenticated per-job JSON, `result.json`
and a generated `summary.md`:

- [Gradient and cross results](../output/statevector-gradient-development-20260911/summary.md)
  ([raw result](../output/statevector-gradient-development-20260911/result.json)).
- [Tail/all-online results](../output/statevector-tail-development-20260911/summary.md)
  ([raw result](../output/statevector-tail-development-20260911/result.json)).
- [CRN results](../output/statevector-crn-development-20260911/summary.md)
  ([raw result](../output/statevector-crn-development-20260911/result.json)).

The source was `output/state-preparation-trajectory-comparison-20260910`,
including its authenticated original construction link. Runtime: Python
3.11.9, NumPy 2.4.6, SciPy 1.17.1, Qiskit 2.4.1, eight-core Apple M1 Pro,
16 GiB RAM. All studies used `--workers auto`. No source code was changed
during the scientific runs; older experiment artifacts were read only.

Tail refinements consumed 86.23 training CPU seconds and 169.50 exact-scoring
CPU seconds; CRN refinements consumed 43.45 and 125.97 seconds respectively.
The diagnostic charged 44.57 CPU seconds across both estimators, including
sampling separately for each standalone estimator, plus 0.44 seconds for exact
reference gradients. Physically shared diagnostic sampling must not be counted
twice when reconstructing total execution cost. Setup and I/O are additional.
No peak-memory study or matched MPS-versus-statevector speed benchmark was run.

Post-run audits checked all result/manifest/job checksums, finite parameters,
50-update traces, 51 exact scores per refinement, the required refresh counts
(50/10/1), identical tail prefixes through update 25, and exact equality of the
fresh baselines across studies. Explicit single-worker resume reproduced every
artifact byte-for-byte without rerunning completed numerical work.
The targeted driver suite passed: **44 tests in 4.56 seconds**. Tests ran after
the scientific studies, not concurrently with their timing measurements.
