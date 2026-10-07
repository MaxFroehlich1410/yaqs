# Frozen q8 fine-tuning with 100 compressed histories

This rerun changes only the fine-tuning ensemble: three sampled histories become
100 sampled histories, with duplicate empty histories compressed using their
empirical multiplicities. Circuit construction, pruning, learning rates, update
counts, online/batch phases, noise profiles, seeds, and CRN schedules are read
from the preceding experiment's saved checkpoints.

## Completed exact-fidelity results

All 13 paired reruns completed. The two archived 150-gate top-down cases show
mixed effects: TFIM seed 300 improves from **57.5241% to 71.0356%**, whereas
seed 400 changes from **81.6983% to 81.0609%**. Thus this run verifies the
compression and larger-ensemble implementation, but does not establish a
consistent fidelity improvement from 100 histories under the fixed settings.

Every score below is an exact noisy density-matrix fidelity in percent.
“Before” scores the common checkpoint immediately before noisy fine-tuning.
“Change” is 100-history minus 3-history fidelity in percentage points.

| Target and case                       | Noise  | Gates |  Before | 3 histories | 100 histories |   Change |
| ------------------------------------- | ------ | ----: | ------: | ----------: | ------------: | -------: |
| TFIM 300, archived layerwise          | IBM    |   150 | 80.8208 |     86.0596 |       86.0544 |  -0.0052 |
| TFIM 300, archived top-down           | IBM    |   150 |  0.2711 |     57.5241 |       71.0356 | +13.5115 |
| TFIM 400, archived layerwise          | IBM    |   150 | 78.3826 |     85.3954 |       85.3668 |  -0.0286 |
| TFIM 400, archived top-down           | IBM    |   150 |  1.0820 |     81.6983 |       81.0609 |  -0.6374 |
| TFIM 400, separate layerwise baseline | IBM    |   150 | 79.8251 |     85.4688 |       85.4410 |  -0.0277 |
| TFIM 300, literal cutoff              | IBM    |     0 | 10.0023 |     10.0023 |       10.0023 |  +0.0000 |
| TFIM 300, literal cutoff              | Google |     0 | 10.0023 |     10.0023 |       10.0023 |  +0.0000 |
| Haar 300, literal cutoff              | IBM    |     0 |  0.4723 |      0.4723 |        0.4723 |  +0.0000 |
| Haar 300, literal cutoff              | Google |     0 |  0.4723 |      0.4723 |        0.4723 |  +0.0000 |
| TFIM 300, unpruned hybrid control     | IBM    |    87 | 83.9027 |     85.1721 |       84.8630 |  -0.3091 |
| TFIM 300, unpruned hybrid control     | Google |    87 | 78.7819 |     72.0566 |       79.4590 |  +7.4024 |
| Haar 300, unpruned hybrid control     | IBM    |   780 | 40.4818 |     40.6702 |       40.9250 |  +0.2548 |
| Haar 300, unpruned hybrid control     | Google |   780 | 20.0401 |     18.6951 |       20.2100 |  +1.5149 |

The literal cutoff still deletes all 87 gates for the TFIM target and all 780
gates for the Haar target. More trajectories cannot change an empty circuit.
The pruning rule remains deferred for a separate controlled change.

For the seed-400 archived top-down result, the old three-history bank had no
jumps and reported 95.9186% training fidelity against 81.6983% exact fidelity.
The new bank has 85 empty paths and 15 jump paths: training fidelity is
82.6350%, with 81.0609% exact fidelity. Final scoring integrates all noise
outcomes and never uses these training banks.

The unchanged 276-gate VQA reference from the separate seed-400 baseline
settings remains at 70.5084% exact IBM fidelity. The corresponding layerwise
baseline now scores 85.4410% at 150 gates. These fixed-case results do not show
a top-down advantage over that layerwise reference. They are an intervention
on fine-tuning, not a new baseline hyperparameter or compute-budget search.

## Exact implemented procedure

1. Load the saved target vector, gate list, and parameters immediately before
   noisy fine-tuning. Authenticate the historical checkpoint JSON and record
   hashes of every input and the executing source.
2. Rerun the original three-history fine-tuning. Require its final parameters to
   match the previous result within `1e-10` before running the intervention.
3. Sample 100 complete noise histories using the same provider and counter.
   Inspect every history for jumps before propagating any state. Record all
   sampled events, jump counts, multiplicities, and replay hashes.
4. Let `n0` be the number of histories with no operators at any gate. If `n0>0`,
   retain one such history with weight `n0/100`. Retain every jump history with
   weight `1/100`. Explicit operators, including cancelling jumps, are retained.
   The weights sum to one. Do not reject, resample, or force events into a bank.
5. Propagate the retained histories with dense complex statevectors. At each
   parameter update recompute the empty representative too. Fidelity uses
   `sum_r w[r] * abs(<target|psi[r]>)**2`; the existing cross signal uses
   `sum_rs w[r] * w[s] * pair_contribution(r,s)`. Weights act on both indices.
   Perform exactly the stored update schedule.
6. Preserve frozen CRN in the top-down and literal arms. Preserve the archived
   layerwise baseline's per-step refresh; each of its 100 steps samples a new
   100-history bank. Verify every bank's first three histories equal the paired
   three-history run. No additional seed search is performed.
7. Evaluate final fidelity independently by propagating a `256 x 256` Qiskit
   density matrix through the full circuit and analytical noise channels. This
   integrates jump and no-jump outcomes without reusing a training ensemble.
   Store both archived profiles and both their TJM and full Lindblad laws.
   Main comparisons use the exact ensemble law of the training sampler (TJM).

For nonempty noisy circuits the runner stops if a sampled bank has no jumps;
it does not condition the distribution by silently drawing another bank. Empty
circuits have no gate noise windows and therefore correctly have no jumps.

Training uses 256-amplitude vectors, with no MPS truncation. The cross signal is
the historical all-pairs estimator; changing its mathematical objective or
replacing it with the independent gradient is outside this intervention.

## Fixed comparison cases

- Disordered TFIM **ground-state** targets at seeds 300 and 400: archived
  top-down and layerwise endpoints, four comparisons in total.
- The separately archived layerwise baseline settings at TFIM seed 400: one
  comparison. This is the same target but a different saved initialization.
- Literal hybrid/cutoff diagnostics at TFIM seed 300 and Haar seed 300: cutoff
  and unpruned control under IBM and Google profiles, eight comparisons.

These are three distinct target vectors, all eight qubits. They are the same
small cases from the preceding verification, not newly selected states. The
absolute `0.01` cutoff still leaves zero gates in both literal cases. Its
diagnostics remain in the results unchanged; revising pruning is deferred.
The direct noisy-from-scratch VQA references have no separate fine-tuning stage
and retain their previous results. The archived layerwise baseline keeps its
historical label and is not newly asserted to implement canonical ADAPT-VQE.

## Retained paths and measured fine-tuning cost

All runs use one BLAS thread. These are training wall times, including sampling,
compression, replay logging, and parameter updates, excluding exact final
scoring. The layerwise cases refresh their bank on every update; other cases
reuse one frozen bank. The budget is unchanged in updates, not in compute.

| Target and case                       | Noise  | Retained paths per bank | 3-history time (s) | 100-history time (s) |
| ------------------------------------- | ------ | ----------------------: | -----------------: | -------------------: |
| TFIM 300, archived layerwise          | IBM    |                    9–26 |               6.60 |               180.48 |
| TFIM 300, archived top-down           | IBM    |                      24 |               2.77 |                13.08 |
| TFIM 400, archived layerwise          | IBM    |                    8–30 |               6.76 |               179.87 |
| TFIM 400, archived top-down           | IBM    |                      16 |               2.86 |                 9.93 |
| TFIM 400, separate layerwise baseline | IBM    |                    8–30 |               6.72 |               183.58 |
| TFIM 300, unpruned hybrid control     | IBM    |                       8 |               1.85 |                 4.30 |
| TFIM 300, unpruned hybrid control     | Google |                      14 |               2.01 |                 6.80 |
| Haar 300, unpruned hybrid control     | IBM    |                      54 |              17.32 |               169.02 |
| Haar 300, unpruned hybrid control     | Google |                      86 |              17.27 |               268.41 |

Across all 310 sampled banks, 31,000 histories reduce to 5,430 retained
histories. Frozen banks are reused over multiple updates, so this count is a
bank inventory rather than a count of all propagated path-steps. All 306 banks
for nonempty circuits contain jumps, with at least 7 jump-containing histories
per bank. The four empty-circuit banks correctly retain one path of weight 1.

## Reproduction and evidence

```sh
uv run python -m experiments.legacy_topdown_weighted \
  --output output/legacy-topdown-weighted100-20260914
```

Use a fresh directory for another run; `--resume` requires the same input and
source hashes. Noise coefficients and gate conventions follow the
[preceding verification](legacy_topdown_dense_findings.md). The saved `manifest.json` contains hashes of three source files
and 49 input files. Source text snapshots accompany it. Each case stores its
target, initial circuit/parameters, original three-history replay, weighted
100-history result, complete event records, and exact matrix scores. The root
`comparisons.json` collects completed comparisons. The prior output directories
remain unchanged.

## Compression verification

On the archived TFIM seed-400 top-down circuit at counter 5400, 100 samples
contain 85 no-jump histories and 15 histories with jumps. Compression retains
16 paths with weights `0.85, 0.01, ..., 0.01`.

Against the identical uncompressed 100 samples, the maximum cross-gradient
error is `1.39e-17`, fidelity error `1.73e-18`, and maximum online parameter
update error is zero at floating-point precision. Three alternating timings
give median gradient times of 158.3 ms uncompressed and 30.1 ms compressed, a
5.26x kernel speedup. Sampling the raw bank costs 1.84 s; compression costs
1.35 ms. This timing separates propagation savings from sampling overhead.
It is not a claim that 100-history training is faster than three-history
training. Frozen CRN amortizes sampling; per-step refresh pays it every step.

The compression is an exact representation of the sampled empirical ensemble,
up to floating-point roundoff. Increasing from 3 to 100 samples changes that
ensemble; its effect on optimization must be assessed from exact final scores.

## Completed artifact audit

All original three-history runs reproduce their saved final parameters
**bit for bit**. Every gate list, initial parameter vector, update count,
online/batch phase, rate, seed, and CRN schedule is unchanged. All bank event
counts, multiplicities, probabilities, hashes, and nested first-three histories
were checked. All three source hashes and 49 input hashes remain unchanged.

The maximum noiseless vector/Qiskit fidelity discrepancy is `6.66e-16`.
Across all final matrix checks, the maximum trace error is `2.18e-14`,
maximum Hermiticity error `6.35e-17`, and minimum eigenvalue `-6.84e-16`
(floating-point roundoff). Detailed checks are in `validation/audit.json`.

## Software checks

111 targeted tests passed, covering compressed/full replication, cross and
independent signals, hybrid updates, shared/affine parameters, empty circuits,
input validation, frozen/refreshed replay, checkpoint resume, and independent
Qiskit checks. Ruff 0.15.15 and the project's `ty` pass on changed Python files.

The required `uvx nox -s lint` was run and fails on existing repository issues:
manuscript/plot/MPS lint findings, the Homebrew Node binary's missing
`libsimdjson.28.dylib`, and type checking of archived output source snapshots
with unresolved dependencies. Its unrelated automatic edits were restored to
their exact pre-lint contents. The log is preserved with the run's validation
artifacts.
