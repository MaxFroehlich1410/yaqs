# Experiment 2 findings: trajectory hybrid Krotov is not yet competitive

Completed 2026-09-10 UTC (2026-09-11 at completion in Berlin), using the
[frozen Experiment 2 protocol](state_preparation_trajectory_comparison.md).
All 60 development runs, 16 shared constructions and 80 confirmation runs
completed. No target, replica, or unsuccessful starting circuit was excluded.

## Answer to the research question

**Can trajectory-based hybrid Krotov achieve the noise-aware improvement
competitively? Not with the configuration tested here.**

On eight fresh targets from four families, each with two paired replicas:

- Trajectory hybrid reached **75.540%** mean exact noisy fidelity, versus
  **76.853%** for trajectory Adam: **1.313 percentage points lower**, while
  using approximately **13% more refinement CPU time**. Adam won 15 of 16
  paired endpoints and had the higher mean in every family.
- Hybrid was slightly better than matched-rate trajectory batch Krotov:
  **+0.058 percentage points**, winning 14 of 16 pairs, at about **7% extra CPU**.
- Noisy hybrid was **0.186 percentage points below noiseless hybrid overall**.
  Noise-aware gains on Gaussian and TFIM did not extend to Haar and random MPS.
- The online phase nevertheless has a clear positive result: it substantially
  reduced the early instability of aggressive batch updates. This is useful
  evidence for the hybrid mechanism, but not an advantage over Adam in final
  fidelity, optimization cost, or cross-family final-result robustness.

Neither predeclared competitiveness screen was met: a gain of at least 0.5
percentage points at comparable CPU cost across families, or fidelity within
0.2 percentage points of the strongest reference with at least 1.5x lower CPU
cost. This rejects a competitiveness claim for this configuration, not every
possible trajectory Krotov variant.

## What was compared

Six-qubit circuits, zero input state, pure targets, and the existing calibrated
local-depolarizing Lindblad channel after every retained gate:

| Quantity                                           |                  Value |
| -------------------------------------------------- | ---------------------: |
| One-qubit average gate infidelity                  |                0.00064 |
| Two-qubit average gate infidelity                  |                 0.0051 |
| Per-axis one-qubit integrated Lindblad coefficient | 0.00032020497493061084 |
| Per-axis two-qubit integrated Lindblad coefficient |  0.0010664702651014273 |
| Retained one-/two-qubit gates                      |                78 / 30 |
| Trainable parameters                               |                    108 |
| Refinement updates                                 |                     50 |
| Trajectories per noisy update                      |                     32 |

The exact commuting Pauli-factor unraveling was used, not the legacy
at-most-one-jump approximation. These are hardware-motivated homogeneous
abstract-gate noise assumptions, not a native-device execution claim. Only
this state-independent noise model was tested; no amplitude-damping or general
nonunitary-jump gradient conclusions follow.

All methods started from exactly the same circuit and parameters for each
case. The shared, noiseless 570-update growth/pruning/repair recipe was unchanged.
There was no further pruning during refinement. The comparison therefore tests
noisy **refinement**, not noise-aware construction or noise-aware pruning.

Trajectory Adam and both noisy Krotov methods used the same independent-path
fidelity estimator, with paths refreshed at every update and paired across
methods. Each noisy refinement drew 1,600 training paths; the online sweep also
propagated updated prefixes, so equal path counts did not mean equal work.
Untruncated MPS trajectories were used throughout. Exact density evaluation was
independent of the sampled training objectives.

Three configurations per method were tested on the four old development
targets, replica 11, with equal 50-update budgets. Selection was frozen before
any fresh-target construction:

| Method                   | Selected setting                  | Development mean noisy fidelity (%) |
| ------------------------ | --------------------------------- | ----------------------------------: |
| Noiseless hybrid Krotov  | 25 online at 0.7; 25 batch at 0.1 |                           77.597568 |
| Exact-noisy Adam         | learning rate 0.02                |                           78.232451 |
| Trajectory Adam          | learning rate 0.02                |                           77.985068 |
| Trajectory hybrid Krotov | 25 online at 0.7; 25 batch at 0.1 |                           77.642244 |
| Trajectory batch Krotov  | 25 batch at 0.7; 25 batch at 0.1  |                           77.612340 |

The matched batch setting was also its independently best development setting.
Adam used beta1=0.9, beta2=0.999 and epsilon=1e-8. All selected rates were at
the upper edge of their three-point grids: this was a limited screen, not
exhaustive optimization of any method. No settings were changed after seeing
the confirmation results. Noiseless hybrid's training objective was noiseless,
but its rate was selected using noisy development evaluation.

## Primary endpoints and measured cost

Fidelities below are exact channel-averaged fidelities at update 50, not
training-trajectory estimates or best-iterate selections. Every family, target
and replica has equal weight. CPU and wall figures are mean **refinement-only**
times per case, excluding exact evaluation, checkpoint I/O and shared construction.

| Method                   | Mean noisy fidelity (%) | Mean CPU seconds | Mean wall seconds |
| ------------------------ | ----------------------: | ---------------: | ----------------: |
| Frozen common start      |               70.025887 |                — |                 — |
| Noiseless hybrid Krotov  |               75.725944 |             3.02 |              3.72 |
| Exact-noisy Adam         |               77.085822 |             1.40 |              1.76 |
| Trajectory Adam          |               76.853414 |           103.03 |            127.29 |
| Trajectory hybrid Krotov |               75.540336 |           116.69 |            144.54 |
| Trajectory batch Krotov  |               75.482492 |           109.40 |            136.74 |

The increase from the frozen start is **not** a noise-aware advantage: much of
it is ordinary additional optimization, especially recovery of one weak MPS
start. The matched noiseless continuation is the relevant Krotov control.

| Hybrid minus comparator  | Mean difference (percentage points) | Median difference | Hybrid wins / 16 | Hybrid/comparator CPU |
| ------------------------ | ----------------------------------: | ----------------: | ---------------: | --------------------: |
| Noiseless hybrid         |                           -0.185608 |         +0.050677 |               11 |                 38.60 |
| Exact-noisy Adam         |                           -1.545486 |         -0.457569 |                0 |                 83.09 |
| Trajectory Adam          |                           -1.313078 |         -0.294170 |                1 |                  1.13 |
| Matched trajectory batch |                           +0.057844 |         +0.074166 |               14 |                  1.07 |

The single endpoint win over trajectory Adam was only 0.009849 percentage
points. The largest deficit was 5.053617 points. The trajectory-Adam versus
hybrid comparison is particularly informative because both use the same MPS
backend, noise model, sampled-gradient estimator, path budget and paired starts.

The run used eight single-numerical-thread workers on an eight-core Apple M1 Pro
with 16 GiB RAM. The full scientific run took approximately **28 minutes**:
11 minutes of development selection, 2 minutes 15 seconds of fresh construction,
then 14 minutes 39 seconds of confirmation. Validation/auditing was additional.
Development consumed 3,922.88 CPU seconds; confirmation consumed 5,336.82 CPU
seconds. These totals exclude shared construction; its recorded block wall times
sum to 1,057.30 seconds across 16 cases and are not serial CPU measurements.

Timing is observational under concurrent execution on heterogeneous laptop
cores, not an isolated latency benchmark. Part of the optimization test suite
ran concurrently with construction/early confirmation. Small timing differences
should therefore not be treated as precise implementation speed ratios.
The observed timings offer no speed advantage for hybrid. Exact density training
is exceptionally cheap at six qubits; its roughly 83x CPU advantage here is not
a claim about larger-system scaling. There was **no method-specific peak-memory
or memory-scaling benchmark**; process high-water RSS includes imports, earlier
jobs and dense evaluation. Shared construction must also be charged to every
method before making an end-to-end speedup claim.

## Results by target family

Each entry averages two genuinely different targets and two replicas per target.

| Family     | Noiseless hybrid (%) | Exact-noisy Adam (%) | Trajectory Adam (%) | Trajectory hybrid (%) | Matched batch (%) |
| ---------- | -------------------: | -------------------: | ------------------: | --------------------: | ----------------: |
| Gaussian   |            82.150238 |            82.644597 |           82.448755 |             82.256965 |         82.156238 |
| TFIM       |            81.618428 |            82.110728 |           81.977225 |             81.741516 |         81.709038 |
| Haar       |            60.357314 |            63.807112 |           63.286978 |             59.974506 |         59.852155 |
| Random MPS |            78.777796 |            79.780851 |           79.700700 |             78.188357 |         78.212536 |

Noise-aware hybrid minus noiseless hybrid was +0.107 points on Gaussian,
+0.123 on TFIM, -0.383 on Haar, and -0.589 on MPS. Thus positive results on the
structured families are real, but they do not support a target-independent
noise-aware improvement under this budget.

Haar targets remain substantially under-optimized: final mean noiseless
fidelity was 75.850% for noisy hybrid and 80.113% for trajectory Adam.
Consequently, not all fidelity differences in this experiment can be attributed
to exploiting noise-specific headroom. Optimizer progress remains important.

## The online phase did help, especially early

The final endpoint alone understates the positive hybrid-versus-batch result.
Both used the same 0.7 prefix rate, 0.1 tail rate and sampled paths; only the
first 25 updates' gate-sequential versus batch ordering differed.

| Update | Trajectory hybrid (%) | Matched trajectory batch (%) | Trajectory Adam (%) |
| ------ | --------------------: | ---------------------------: | ------------------: |
| 0      |             70.025887 |                    70.025887 |           70.025887 |
| 10     |             70.452223 |                    63.482541 |           72.554951 |
| 25     |             75.365011 |                    71.680340 |           75.782207 |
| 50     |             75.540336 |                    75.482492 |           76.853414 |

At update 10, batch was below its starting fidelity in 11/16 cases, with a worst
drop of 26.835 percentage points. Hybrid was above its start in all 16 cases at
both updates 10 and 25. This is checkpoint-level stability evidence, **not** a
claim of monotonicity at every unobserved update or of guaranteed stability.
Adam had nine small below-start deviations at update 10 (worst 0.287 points)
and only one at update 25 (0.027 points).

Hybrid's mean advantage over matched batch was 6.970 points at update 10 and
3.685 points at update 25. Batch largely recovered during the lower-rate tail.
This supports the value of the aggressive online phase relative to high-rate
batch updates. It does not imply superiority to every possible batch schedule.

Adam remained ahead at each recorded nonzero checkpoint. At update 25, hybrid
used 64.90 mean CPU seconds versus Adam's 52.03 seconds. Over updates 26–50,
hybrid gained only 0.175 points, while Adam gained 1.071 points. The implemented
batch tail did not close the gap.

## Weak-start case and sensitivity of the aggregate

`mps_fresh_1`, replica 29, entered refinement at 0.259104% noiseless fidelity
and approximately 0.241928% noisy fidelity. Both replicas of this target had
stayed below 0.1% noiseless fidelity through all four growth stages. Replica 11
escaped during the final 50-step continuation and reached 86.619709%; replica
29 remained at 0.259104%. The trace shows slow escape from a low-overlap start,
not a catastrophic fidelity loss caused by pruning.

All methods subsequently recovered substantial fidelity in replica 29:

| Refinement               | Final noisy fidelity (%) |
| ------------------------ | -----------------------: |
| Noiseless hybrid         |                78.430686 |
| Exact-noisy Adam         |                81.001602 |
| Trajectory Adam          |                80.825361 |
| Trajectory hybrid        |                75.771744 |
| Matched trajectory batch |                76.007764 |

This case contributes strongly to the large gain over the frozen-start mean.
It also reveals sensitivity of the fixed construction budget to the target and
initialization, rather than establishing a noise-specific benefit.

As a **post hoc diagnostic only**, excluding this one case still leaves hybrid
1.064 percentage points below trajectory Adam and 0.021 points below noiseless
hybrid on the remaining 15 cases. The negative overall competitiveness result
is therefore not solely an artifact of this weak start. The primary results
above retain all 16 cases; no exclusion or retuning was used for selection.

## Relationship to Experiment 1 and next decision

[Experiment 1](state_preparation_exact_adam_findings.md) showed a modest exact
noise-aware gain when comparing noisy and noiseless **Adam** on old targets.
This new experiment used fresh targets and a shorter continuation budget. It
did not include a fresh-target noiseless-Adam arm, so the difference between
exact-noisy Adam and noiseless Krotov must not be labeled a pure noise benefit.
The within-hybrid noisy/noiseless comparison is the appropriate matched control
for that claim here.

The evidence supports three separate statements:

1. The trajectory implementation is operational and independently validated;
   its failure to win is not evidence that trajectories cannot model the noise.
2. Gate-sequential online updates are useful relative to aggressive batch
   Krotov updates, particularly for early stability. This is compatible with
   the earlier positive noiseless construction results.
3. The present R=32, 25-online/25-batch noisy refinement is **not ready to support
   an Adam-superiority or broad robustness claim**, and this small test does
   not establish comparison with the field's state of the art.

Do not scale this configuration to a weeks-long campaign yet. The next bounded
diagnostic should compare sampled gradients against the existing exact-density
gradient oracle on **development targets**, separating sampling error from
step-size/update-scaling limitations. The current data do not establish which
of these causes the gap. Any subsequent rate, estimator or tail modification
needs a newly frozen evaluation; these confirmation targets are no longer
unseen. Improving weak-start construction is a separate issue and must be
tested with the same construction budget for every competing refinement.

## Reproducibility and verification

Implementation:

- [Comparison runner](state_preparation_trajectory_comparison.py).
- [Independent audit and summaries](state_preparation_trajectory_analysis.py).
- [Protocol, target definitions, seeds and commands](state_preparation_trajectory_comparison.md).

Artifacts are in `output/state-preparation-trajectory-comparison-20260910/`:

- `manifest.json`: exact fresh target vectors, noise parameters, old-source
  provenance, configuration, dependency versions and training-source fingerprint.
- `selection.json`: all 15 development configuration scores and frozen nominees.
- `starts/`: all 16 common circuits, parameters, construction blocks and pruning events.
- `development/` and `confirmation/`: all moments, phase/rate traces, seeds,
  timings and exact evaluations at updates 0, 10, 25 and 50.
- `result.json`: authenticated canonical development and confirmation results.
- `audit.json`: independent fidelity checks, paired summaries and checkpoint curves.
- `validation/`: full lint output, completed-run resume log, and audit output.

The manifest was written at 21:32:27 UTC, selection frozen at 21:43:32 UTC,
all fresh starts completed by 21:45:47 UTC, and the scientific run completed at
22:00:26 UTC on September 10. Completed resume revalidated the same artifacts
without retraining, changing nominees or replacing original timing records.

The independent audit recomputed **640 fidelities**, with maximum absolute
disagreement **1.67e-15**, and verified checkpoint identities, phase schedules,
noise seeds and summary values. The complete optimization suite passed:
**406 tests**, including 45 new comparison/analysis tests. Checks cover target
mathematics, gradients, public-trainer equivalence, input immutability, paired
sampling, phase boundaries, moments/counters on resume, construction/pruning
budgets, development-only nomination and serial/parallel reproducibility.

All applicable lint hooks pass for the new experiment files and updated release
notes. `uvx nox -s lint` was also run: repository-wide lint still fails on
pre-existing manuscript/script/core-test issues, missing matplotlib in the lint
environment, and the existing Node/simdjson dependency error in bibtex-tidy.
Unrelated automatic hook edits were reverted; those files were not changed as
part of this experiment. The full repository test suite beyond optimization
was not run.

Key SHA-256 identities (canonical records exclude their checksum envelopes):

```text
manifest: 84f7f2cfdbf466aeb253d4e134db6848b61a88e3f74fa44dbe3afa2eecc1140f
selection: 09427f6194fdd29c625f0497a928ef49156e23cc1e7d8aa563eb714d38052059
result: abc3aab1f9a771d175596e2f41262fb12ce798ba7e7a4fbd01dde3186df0679e
training sources: 8dcfaab5b1e56c1ca19f6b6f0f8f4e4e085886b7df59c85162e402b467eabb06
analysis source: 10ca6c5bbabd8b68bcccffc8defbf21e4ff9bd09379514fab29f602820947d01
```

## All paired confirmation endpoints

Every number is exact noisy fidelity in percent, rounded only for display.
H0 = noiseless hybrid; EA = exact-noisy Adam; TA = trajectory Adam;
TH = trajectory hybrid; TB = matched trajectory batch.

| Target           | Replica | Frozen |     H0 |     EA |     TA |     TH |     TB |
| ---------------- | ------: | -----: | -----: | -----: | -----: | -----: | -----: |
| gaussian_fresh_1 |      11 | 82.016 | 82.058 | 82.421 | 82.289 | 82.178 | 81.978 |
| gaussian_fresh_1 |      29 | 82.234 | 82.282 | 82.541 | 82.383 | 82.306 | 82.242 |
| gaussian_fresh_2 |      11 | 81.860 | 81.951 | 82.734 | 82.467 | 82.004 | 81.998 |
| gaussian_fresh_2 |      29 | 82.299 | 82.310 | 82.883 | 82.657 | 82.541 | 82.407 |
| tfim_fresh_1     |      11 | 80.404 | 80.431 | 80.885 | 80.775 | 80.571 | 80.559 |
| tfim_fresh_1     |      29 | 81.174 | 81.245 | 81.991 | 81.812 | 81.428 | 81.342 |
| tfim_fresh_2     |      11 | 82.198 | 82.272 | 82.744 | 82.587 | 82.392 | 82.365 |
| tfim_fresh_2     |      29 | 82.460 | 82.526 | 82.822 | 82.735 | 82.575 | 82.570 |
| haar_fresh_1     |      11 | 57.460 | 60.835 | 64.984 | 64.336 | 60.451 | 60.329 |
| haar_fresh_1     |      29 | 59.415 | 61.778 | 64.957 | 65.268 | 61.330 | 61.178 |
| haar_fresh_2     |      11 | 59.775 | 62.058 | 66.044 | 65.314 | 61.780 | 61.675 |
| haar_fresh_2     |      29 | 53.722 | 56.759 | 59.243 | 58.230 | 56.338 | 56.226 |
| mps_fresh_1      |      11 | 71.433 | 72.961 | 73.934 | 74.011 | 73.087 | 73.002 |
| mps_fresh_1      |      29 |  0.242 | 78.431 | 81.002 | 80.825 | 75.772 | 76.008 |
| mps_fresh_2      |      11 | 82.027 | 82.040 | 82.195 | 82.058 | 82.068 | 82.010 |
| mps_fresh_2      |      29 | 81.695 | 81.679 | 81.992 | 81.908 | 81.827 | 81.830 |
