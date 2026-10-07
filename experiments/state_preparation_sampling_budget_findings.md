# Sampling-budget and weighted-gradient findings

Later follow-up: weighted circuit optimization has now been tested with the
same estimator for Krotov and Adam. See the
[completed optimization findings](state_preparation_weighted_optimization_findings.md).
The diagnostic-only conclusions below describe the earlier stage of the work.

Completed 2026-09-12 on the four old development targets. The
[optimizer/budget protocol](state_preparation_sampling_budget.md) was frozen
before step 1. The [weighted diagnostic protocol](state_preparation_weighted_sampling.md)
was frozen after step 1 justified investigating variance, before any weighted
estimates were computed. No new targets, construction, pruning or noise models
were introduced, and no method was frozen for a confirmation experiment.

## Main interpretation

**Adam remains the stronger optimizer in this screen. Weighted sampling is a
promising estimator improvement, but it has not yet been used to train a
circuit.** These are two distinct conclusions.

- Increasing R from 32 to 128 improved all-online Krotov by only 0.011157
  percentage points (pp), but improved Adam by 0.143217 pp. Adam beat
  all-online Krotov in every R=128 pair. Sampling noise is relevant, but it
  does not explain the whole optimizer gap.
- The matched-rate control confirms an online-ordering benefit: all-online
  0.7 beat a hybrid with a batch tail at the same 0.7 rate by 3.556342 pp,
  winning eight of eight pairs. High-rate batch continuation became unstable
  on several cases. This is a stability advantage, not an advantage over Adam.
- The subsequent weighted R=32 diagnostic reduced pooled gradient MSE by
  79.9%, used 33.2% less measured CPU, and reduced mean MSE-times-CPU by 86.6%
  relative to ordinary R=32 sampling. All 12 target/checkpoint settings
  improved in both MSE and the cost/error product.
- Weighted R=32 had slightly lower pooled MSE than ordinary R=128 while using
  only 17.6% of its CPU per estimate. This is an observed estimator result;
  it is not a 5.7x optimization speedup or a demonstrated preparation-fidelity
  improvement.

## Step 1: matched optimizer and trajectory count

Six qubits; one Gaussian, critical TFIM, Haar and random bond-2 MPS target;
the same four frozen construction-replica-11 circuits and initial parameters.
Each circuit has 108 parameters/gates, including 30 two-qubit gates. Noise
replicas 11/29 do not represent distinct construction starts or target states.

The calibrated homogeneous local Pauli-Lindblad channel is unchanged:
`r1=0.00064`, `r2=0.0051`, integrated per-axis coefficients approximately
`0.00032020497493061084` and `0.0010664702651014273`. All trajectory training
uses complex128 statevectors and exact independent Bernoulli factors. Noise
maps refresh every update, with paired streams across methods and nested path
indices across R. Root seed and seed derivation remain unchanged.

All refinements use 50 updates from saved update 0. The three primary methods
are all-online at 0.7; 25 online at 0.7 then 25 batch at 0.3; and Adam at
0.02, the earlier selected rate. Adam moments start at zero, with beta1=0.9,
beta2=0.999 and epsilon=1e-8. The extra ordering control has the same hybrid
prefix and a batch tail at 0.7, only at R=32.

There are 48 primary cases, eight ordering controls and four deterministic
exact-noisy Adam references. Sixteen compatible R=32 Krotov cases were reused
read-only, preserving their original artifacts and timings. **44 new cases
were executed, with 2,200 new optimization updates.** Exact Adam is executed
once per target, not repeated as if it had stochastic noise replicas.

### Final exact noisy fidelity and cost

Endpoints are update 50, never the best observed iterate. Training CPU includes
sampling and optimizer work, but excludes exact scoring, setup and I/O.

| Method                          |     R | Mean noisy fidelity (%) | Mean training CPU (s) |
| ------------------------------- | ----: | ----------------------: | --------------------: |
| All-online Krotov 0.7           |    32 |               77.768017 |              2.921245 |
| All-online Krotov 0.7           |   128 |               77.779174 |              8.278365 |
| Hybrid, batch tail 0.3          |    32 |               77.694438 |              2.617726 |
| Hybrid, batch tail 0.3          |   128 |               77.701232 |              7.495020 |
| Trajectory Adam 0.02            |    32 |               78.002904 |              1.875531 |
| Trajectory Adam 0.02            |   128 |               78.146122 |              6.751116 |
| Hybrid, batch tail 0.7 control  |    32 |               74.211675 |              2.146353 |
| Exact-noisy Adam 0.02 reference | Exact |               78.232451 |              1.415964 |

| Change from R=32 to R=128 | Mean fidelity change (pp) | Positive pairs / 8 | Training CPU ratio |
| ------------------------- | ------------------------: | -----------------: | -----------------: |
| All-online Krotov         |                 +0.011157 |                  4 |               2.83 |
| Hybrid, batch tail 0.3    |                 +0.006794 |                  6 |               2.86 |
| Trajectory Adam           |                 +0.143217 |                  8 |               3.60 |

All-online minus Adam is -0.234887 pp at R=32 (two wins out of eight), and
-0.366948 pp at R=128 (zero wins). At R=128, all-online also uses about 22.6%
more training CPU than Adam. The R=128 Adam-to-exact-Adam gap is 0.086329 pp.
The exact reference is attainable under its specified optimizer/budget, not
a global optimum or a proof about density-method scaling at larger sizes.

Mean final fidelity by family, averaging the two noise replicas where present:

| Family     | All-online R32 (%) | All-online R128 (%) | Adam R32 (%) | Adam R128 (%) | Exact Adam (%) |
| ---------- | -----------------: | ------------------: | -----------: | ------------: | -------------: |
| Gaussian   |          82.115479 |           82.128147 |    82.139625 |     82.263538 |      82.362062 |
| TFIM       |          82.282598 |           82.281261 |    82.442291 |     82.519009 |      82.578083 |
| Haar       |          64.700243 |           64.708868 |    65.457744 |     65.712639 |      65.842591 |
| Random MPS |          81.973748 |           81.998418 |    81.971958 |     82.089301 |      82.147068 |

The matched-rate batch-tail-0.7 control ends at 75.846944%, 75.961210%,
64.679451% and 80.359097% in those four families. All-online wins every pair
by 0.0207-9.7799 pp at that matched rate. This more cleanly supports online
ordering than comparing online 0.7 against lower-rate batch updates. It still
does not demonstrate superiority to a well-scaled batch or Adam method.

### Fidelity versus training CPU

The result file retains every trajectory update's exact fidelity and cumulative
training CPU, and the exact reference's scores at 0,10,25,50. As a **descriptive,
post-run cost slice**, take the last completed update costing at most one CPU
second, without interpolation or best-iterate selection. Every primary case
ran for more than 1.84 training CPU seconds, so this slice requires no extension
or extrapolation beyond any run's budget.

| Method                 |   R | Mean noisy fidelity at the one-second slice (%) | Completed update range |
| ---------------------- | --: | ----------------------------------------------: | ---------------------: |
| All-online             |  32 |                                       77.549321 |                  16-17 |
| Hybrid, batch tail 0.3 |  32 |                                       77.546004 |                  16-17 |
| Adam                   |  32 |                                       77.807228 |                  25-27 |
| All-online             | 128 |                                       77.407577 |                    5-6 |
| Hybrid, batch tail 0.3 | 128 |                                       77.409058 |                      6 |
| Adam                   | 128 |                                       77.455234 |                      7 |

The early Krotov rows still lie inside the common online prefix; their tiny
differences reflect timing boundaries, not distinct prefix algorithms. At this
slice, buying more trajectories gives fewer updates and worse mean endpoints
for each method. R=128 is therefore not automatically the better fixed-cost
choice. This is a curve-based diagnostic, not a separately run deadline study.

Timings are observational on heterogeneous laptop cores. The 16 reused cases
retain measurements from the previous session; they were not retimed or
presented as concurrent measurements. All methods share construction costs,
which are excluded here. New cases consumed 218.03 training CPU seconds;
reused cases retain 44.31 seconds. No memory/scaling advantage was measured.

## Conditional follow-up: weighted estimator

The step-1 decision was to pursue a **gradient diagnostic**, not a weighted
optimizer sweep: higher R benefited Adam consistently, and the earlier saved
checkpoint diagnostics showed large sampling errors. At the same time, the
small Krotov response to higher R argues against assuming variance reduction
alone will make Krotov competitive.

For this fixed channel, the exact probability of selecting no Pauli factors
over the circuit is **0.7658428756735005**. Separate this deterministic stratum
and sample the remaining 23.4157% probability mass conditionally. For a total
budget R, propagate one deterministic no-factor path and R-1 conditional paths,
using the probability weights specified in the protocol. Multiple factors and
their possible cancellations are retained. The physical noise strengths do
not change.

This is an established sampling strategy adapted to the discrete channel, not
a new sampling-principle claim. The conditional kernel is experimental and
separate; it does not modify the production sampler or implement weighted
online Krotov updates.

The diagnostic recomputed ordinary and weighted estimators at the same old
updates 0,25,50 for all four targets, R=8,32,128, with eight repeats per setting:
**576 estimates across 12 checkpoint jobs**. Estimator order alternated, and
standalone cost includes weighted sampler setup and its deterministic gradient.

| Estimator | Total R | Pooled gradient MSE | Mean cosine | Mean CPU per estimate (s) | Mean MSE times CPU |
| --------- | ------: | ------------------: | ----------: | ------------------------: | -----------------: |
| Ordinary  |       8 |          0.00929247 |      0.3284 |                  0.012083 |        0.000109908 |
| Weighted  |       8 |          0.00217043 |      0.4319 |                  0.011992 |       0.0000256988 |
| Ordinary  |      32 |          0.00234642 |      0.4360 |                  0.034717 |       0.0000806335 |
| Weighted  |      32 |         0.000471988 |      0.6598 |                  0.023182 |       0.0000107655 |
| Ordinary  |     128 |         0.000543900 |      0.6381 |                  0.131827 |       0.0000702824 |
| Weighted  |     128 |         0.000120115 |      0.8488 |                  0.071742 |      0.00000855530 |

Weighted-to-ordinary MSE ratios are 0.234, 0.201 and 0.221 at R=8,32,128;
CPU ratios are 0.992, 0.668 and 0.544. Every one of the 12 checkpoint settings
improves in both MSE and MSE-times-CPU at each R. All 96 weighted estimates at
R=32, and all 96 at R=128, have positive dot products with their exact gradients.
These signs alone do not guarantee improvement under a finite optimizer step.

Weighted R=32 has 86.8% of the pooled MSE of ordinary R=128 and uses 17.6% of
its CPU. It is therefore a promising replacement for increasing the ordinary
trajectory count. However, weighted R=32 endpoint relative errors still exceed
one for Gaussian, TFIM and random MPS; not every endpoint gradient is accurate.
The larger R=128 weighted estimator has mean cosine 0.8488 across checkpoints,
but no parameter updates were performed with it in this study.

The cost reduction includes a faster specialized conditional sampler as well
as statistical variance reduction. It does not isolate a theoretical sampling
gain from implementation overhead. Eight repeats support a development lead,
not a precision speed ratio or population-level confidence claim. Empirical
mean-gradient error is not proof of asymptotic bias. Exhaustive tests establish
the decomposition for this parameter-independent model; they do not extend it
to state-dependent/nonunitary jumps.

## Next decision

Do **not** freeze a configuration or evaluate new targets yet. The next small
comparison should integrate the weighted estimator into both Krotov and Adam
with matched weighted paths and cost accounting. Giving it only to Krotov
would confound an estimator improvement with an optimizer advantage.

Retain all-online 0.7 and the stable hybrid tail as controls. A single
Krotov-to-Adam tail can be a separately labeled combination, rather than a broad
adaptive sweep. Because ordinary R=128 barely helped Krotov, a weighted estimator
alone should not be presumed to close its gap. Any integration into online
sweeps must weight contributions before each gate update, not average complete
independently updated circuits afterward.

No weighted optimizer training, fresh-target evaluation, new noise profile,
memory benchmark or state-of-the-art claim is part of the completed work.

## Artifacts and verification

- [Step-1 generated summary](../output/statevector-sampling-budget-20260912/summary.md)
  and [raw results/CPU curves](../output/statevector-sampling-budget-20260912/result.json).
- [Weighted generated summary](../output/statevector-weighted-diagnostic-20260912/summary.md)
  and [raw diagnostic results](../output/statevector-weighted-diagnostic-20260912/result.json).
- [Step-1 wrapper](state_preparation_sampling_budget.py) and
  [weighted diagnostic/sampler](state_preparation_weighted_sampling.py).

The weighted summary reuses the earlier generic gradient-report format; its
generic cross-promotion caution does not mean that cross estimation was tested
in this run. The estimator labels and raw rows are independent/weighted only.

Both runs used up to eight single-numerical-thread workers on the eight-core
Apple M1 Pro with 16 GiB RAM; Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1,
Qiskit 2.4.1. Existing scientific source files and earlier artifacts remain
unchanged. Manifests record the new wrapper/diagnostic hashes separately.

**89 targeted tests passed in 5.52 seconds**, including the shared statevector
backend, old diagnostics and 23 new tests for these additions. Tests cover
read-only source reuse, stale/corrupt source rejection, serial/parallel
reproducibility, exact-Adam equivalence, unbiased exhaustive one-/two-site
factor expectations, conditional pattern frequencies, cancellation events,
finite differences, zero noise, invalid inputs, costs and completed resume.

Both scientific outputs passed checksum/progress audits and explicit
single-worker completed resume without changing any artifact bytes. Step-1
Krotov prefixes match exactly through update 25 at each R. No tests or lint
overlapped the scientific timing runs. A validation-only overlap between lint
and tests initially triggered source-fingerprint checks; the unrelated hook
edits were restored and the full targeted check was rerun sequentially.

Changed-file hooks and scoped type checks pass. Required repository-wide
`uvx nox -s lint` was run, but still reports the pre-existing manuscript,
plotting/MPS-test issues, broken Homebrew Node `simdjson.28.dylib` dependency,
and missing plotting imports in its lint environment. Those unrelated sources
were not changed to make this experiment pass lint.
