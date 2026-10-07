# State-preparation reruns with hardware-error-calibrated noise

Date: 2026-09-05. Branch: `codex/minimal-noisy-krotov`.
Scope: the recent six-qubit critical-TFIM causal study and small hybrid
diagnostic, not the full multi-target benchmark matrix.

## Calibration and controlled comparison

The intended complete-gate average infidelities are 0.064% (one qubit) and
0.51% (two qubits). These are the reference hardware-motivated targets in the
[benchmark definition](../benchmarks/state_preparation_benchmarks.md#common-rate-convention),
not per-operator rates. Their relationship to the Google specification and
their limitations as a model of native Google/IBM instructions are documented
there. The reference pair is a conservative hardware-motivated model, not a
claim that a particular device currently has these two measured errors.

The corrected provider retains the same local Pauli operators, unit-duration
TJM windows, and noise after every abstract gate. Only coefficients change:

| Quantity                                    |  One-qubit window |  Two-qubit window |
| ------------------------------------------- | ----------------: | ----------------: |
| Number of local Pauli operators             |                 3 |                 6 |
| Legacy coefficient per operator             |           0.00064 |            0.0051 |
| Corrected Lindblad coefficient per operator | 0.000320204974931 | 0.001066470265101 |
| Legacy sampler average gate error           |        0.1278772% |        2.4109247% |
| Corrected sampler average gate error        |       0.06401025% |       0.51027142% |

Coefficients calibrate the exact Lindblad exponential, not a sampler-specific
fit. The existing at-most-one-jump discretization is retained and checked
below. Legacy presets, manifests, and results are not overwritten or relabeled.

The causal study retains the same target, circuit, seeds, and budgets:

- one shared noiseless warm start grown through depths 1, 2, 3, and 4, with
  100 updates per depth;
- five arms: frozen checkpoint, noiseless continuation, fixed common random
  numbers (CRN), CRN refreshed every five updates, and per-update resampling;
- three independent training replicates, 200 updates per nonfrozen arm, three
  training trajectories, and a fixed disjoint 12-trajectory validation set;
- checkpoint selection every five updates, followed by 500 fresh paired test
  trajectories per replicate; no tuning based on those test results;
- the original batch cross-trajectory update and learning-rate schedule.

Independent replicates run in three processes. A bounded serial-versus-parallel
test verifies identical numerical aggregates and random-stream audits. Worker
count is not part of the scientific configuration.

The old and corrected pretrained parameter vectors are bitwise identical.
Comparing their configuration documents changes only calibration fields,
provider identity, and the resulting configuration checksum. The circuit has
138 one-qubit and 60 two-qubit gates.

## Full causal rerun

Completed successfully: three replicates, five arms, and 500 paired held-out
trajectories per replicate. Wall time was 1,996.5 seconds (33.3 minutes),
including shared pretraining. Values below are fidelities, not percentages.
Intervals use the three independent replicate-level paired effects as the
sampling unit; they do not treat 1,500 trajectories as 1,500 training runs.

| Arm                    | Legacy held-out mean | Corrected held-out mean | Corrected gain over frozen | Paired 95% Student-t interval |
| ---------------------- | -------------------: | ----------------------: | -------------------------: | ----------------------------- |
| Frozen                 |           0.25975916 |              0.67773401 |                  reference | reference                     |
| Noiseless continuation |           0.25975916 |              0.68072420 |                +0.00299020 | [-0.00344449, 0.00942489]     |
| Fixed CRN              |           0.26190310 |              0.68301548 |                +0.00528147 | [0.00052620, 0.01003674]      |
| Refresh-5 CRN          |           0.25884306 |              0.67767344 |                -0.00006056 | [-0.00692997, 0.00680884]     |
| Per-update resampling  |           0.26210681 |              0.68252862 |                +0.00479461 | [0.00131928, 0.00826994]      |

Fixed CRN and resampling beat the frozen checkpoint in all three replicates.
Their mean gains are approximately **0.53 and 0.48 percentage points**. Unlike
the legacy higher-noise test, their nominal 95% intervals now exclude zero.
These intervals are exploratory and are not adjusted for multiple comparisons;
three replicates on one target are not confirmatory evidence or a general
method ranking.

The important causal qualification is that further noiseless optimization also
helps. Supplementary paired contrasts, recomputed from the retained arm means,
give:

- fixed CRN minus noiseless continuation: +0.00229128, interval
  [-0.00883939, 0.01342194];
- resampling minus noiseless continuation: +0.00180441, interval
  [-0.00741532, 0.01102415].

Neither distinguishes a noise-aware advantage from continued noiseless
optimization. Refresh-5 does not improve on frozen, and it loses to noiseless
continuation in all three replicates: -0.00305076, interval
[-0.00533660, -0.00076492]. This rerun does not support refresh-5 as the preferred
setting.

Validation-selected update numbers are:

| Arm                    | Replicate 0 | Replicate 1 | Replicate 2 |
| ---------------------- | ----------: | ----------: | ----------: |
| Frozen                 |           0 |           0 |           0 |
| Noiseless continuation |         200 |         200 |           0 |
| Fixed CRN              |         200 |         200 |         200 |
| Refresh-5 CRN          |          30 |          30 |          30 |
| Resampling             |         200 |         155 |         140 |

The noiseless control retains its initial checkpoint in replicate 2. The noisy
arms' positive mean contrasts against that control are driven mainly by this
replicate, underscoring the uncertainty from the small validation ensemble.
Training still ran the complete 200-update budget in every nonfrozen arm; early
checkpoint selection is not early termination.

**Interpretation:** the excessive old noise strongly depressed absolute
fidelity, and two corrected-noise settings now show modest, consistent gains
over doing no further training. This is a useful positive result, but not yet
evidence that accounting for noise itself is better than simply optimizing
longer without noise.

### Training-sample diagnostic

At the corrected rates the total circuit jump hazard is 0.5164941551. Thus a
trajectory is entirely jump-free with probability 0.59660850, and a fixed set
of three trajectories is entirely jump-free with probability 0.21235785.
The retained fixed-CRN ensembles contain event counts `[0, 1, 0]`, `[0, 0, 0]`,
and `[0, 1, 1]` in replicates 0, 1, and 2. For this Pauli model the normalized
no-jump evolution is noiseless, so replicate 1's fixed training set supplies
no sampled noise events. This is a concrete limitation of the original tiny
training budget, not evidence that the corrected provider is inactive.
The trajectory budget is unchanged in this rerun to avoid confounding the
calibration comparison.

## Hybrid diagnostic: completed

The rerun uses the identical verified pretrained parameters, 10 online plus
10 batch updates, constant step size 0.1, three training trajectories,
cross-trajectory updates, and per-update resampling. Training seed 314159,
optimizer seed 271828, and the original 200-trajectory paired test stream are
preserved. No parameter or budget tuning was performed.

| Held-out result               |         Legacy high noise |           Corrected noise |
| ----------------------------- | ------------------------: | ------------------------: |
| Frozen fidelity               |                0.31002423 |                0.70177720 |
| Hybrid fidelity               |                0.31091014 |                0.70195362 |
| Paired hybrid minus frozen    |               +0.00088591 |               +0.00017642 |
| Paired 95% Student-t interval | [-0.00203707, 0.00380890] | [-0.00113861, 0.00149144] |

The corrected hybrid endpoint has noiseless fidelity 0.97127467 versus
0.97159853 for the shared checkpoint. Training took 63.3 seconds and training
plus evaluation took 189.6 seconds while the causal workers also ran.

**Conclusion:** lowering noise greatly improves absolute fidelity but does not
produce a measurable hybrid advantage in this small, untuned diagnostic. The
interval above measures trajectory-sampling uncertainty conditional on this
single training run, not variation across training seeds. The deterministic
channel check below finds a small negative expected hybrid change, consistent
with the interval containing zero.
The hybrid and causal studies use different held-out streams and different
budgets. Their absolute sampled means must not be compared to rank hybrid
against batch optimization; use the within-study paired contrasts.

## Independent density-matrix and discretization checks

For six qubits, an independent Qiskit density matrix can propagate the complete
ensemble channel without sampling. The calculation begins in the zero state,
applies each canonical gate in matching little-endian convention, then applies
either the exact local Lindblad channel or the one-jump ensemble channel. Trace,
Hermiticity, and nonnegative eigenvalues are checked. The noiseless result
0.9715985320982362 agrees with the MPS result to less than 2e-14.

| Circuit and channel                    | Expected fidelity (no Monte Carlo error) |
| -------------------------------------- | ---------------------------------------: |
| Frozen, legacy one-jump noise          |                           0.258438151510 |
| Frozen, corrected one-jump noise       |                           0.677962701236 |
| Frozen, corrected exact Lindblad noise |                           0.677990816066 |
| Hybrid, corrected one-jump noise       |                           0.677632099254 |
| Hybrid, corrected exact Lindblad noise |                           0.677660407888 |

Thus the frozen circuit's finite-step fidelity bias is approximately 2.81e-5,
or 0.00281 percentage points. The expected hybrid-minus-frozen change is
-0.00033060 under the sampled channel and -0.00033041 under the exact Lindblad
channel. The small discretization error does not explain away the absent hybrid
advantage. These channel checks are evaluation-only and do not alter training
or select checkpoints.

The complete selected-checkpoint audit gives the following means across the
three causal replicates:

| Arm                    | One-jump ensemble fidelity | Exact Lindblad ensemble fidelity |
| ---------------------- | -------------------------: | -------------------------------: |
| Frozen                 |             0.677962701236 |                   0.677990816066 |
| Noiseless continuation |             0.681756840259 |                   0.681790308013 |
| Fixed CRN              |             0.683586914581 |                   0.683618444718 |
| Refresh-5 CRN          |             0.678566326965 |                   0.678594378205 |
| Resampling             |             0.683281333116 |                   0.683308528208 |

The largest finite-step difference among the 15 selected causal checkpoints
and the hybrid endpoint is 3.615e-5 in fidelity. The deterministic checks
therefore confirm the modest gains over frozen without Monte Carlo ambiguity,
while still not establishing a clear gain over the noiseless continuation
control across training replicates.

For reproducibility, the one-jump update on a gate with support S and uniform
per-operator coefficient a is

```text
A = 3 * len(S) * a
rho <- exp(-A)*rho + (1-exp(-A))/(3*len(S)) * sum_{s in S, P in X/Y/Z} P_s rho P_s
```

The exact update applies, independently to each site in S,

```text
w = (1-exp(-4*a))/4
rho <- (1-3*w)*rho + w*(X_s rho X_s + Y_s rho Y_s + Z_s rho Z_s)
```

Final fidelity is the target expectation of rho. Noise follows gates regardless
of their current angle, matching the unchanged abstract-gate placement in the
experiment; this is not a compiled native-device noise model.

## Artifacts and implementation checks

All raw outputs are beneath
`/Users/maximilianfrohlich/Desktop/yaqs-experiment-results/`:

| Run              | Directory                                              |
| ---------------- | ------------------------------------------------------ |
| Original causal  | `q6-tfim-critical-causal-8f8308ad`                     |
| Corrected causal | `q6-tfim-critical-causal-hardware-calibrated-20260905` |
| Original hybrid  | `q6-hybrid-smoke-6bed8ea5`                             |
| Corrected hybrid | `q6-hybrid-hardware-calibrated-20260905`               |

The causal directories contain sealed configuration/result JSON, all training
and evaluation ensembles, validation-selected checkpoints, and per-trajectory
paired fidelities. The corrected causal directory additionally retains the
standalone analysis artifact `density_crosscheck.py` and its complete
`density_crosscheck_result.json`. This small-system verification script is
outside the minimal repository and is not a training backend. Reproduce it
from the repository with `PYTHONPATH=.` and the same thread-limiting environment
variables used for the experiments. It checks all 15 selected causal
checkpoints and the hybrid endpoint against both ensemble channels.
The corrected hybrid directory contains `config.json`,
`hybrid_checkpoint.npz`, `test_ensemble.json`, and `hybrid_smoke_result.json`.
Commands are in the benchmark definition's corrected-reruns section.

Important identities:

- Shared pretrained parameters:
  `sha256:14665001ea5a09838a3b7324700d92d18eb475ab38ce3d9e6f061ee6713c22b9`.
- Corrected causal configuration:
  `sha256:b186a9059108480ff34f3778fe04416815987e4fccbcda274a310255e7cfd452`.
- Corrected provider:
  `sha256:1e66fbc4d3fe739a46f617bd9062385797d74608d1e7ea1415e38307a97501ab`.
- Completed corrected causal result:
  `sha256:15be8ccdfaf977ee1d8e3ae622e5d00fad480e914f963c985071fe2879da155a`.
- Corrected hybrid configuration:
  `sha256:24059267e667b9f16b9655a1801d2e3ec0875944eb19290f0850e943531649e5`.

The run uses uncommitted focused additions on base commit
`6bed8ea542f1bd03918254432e7fb8abb60cb21f`. Source SHA-256 values:

| File                                          | SHA-256                                                            |
| --------------------------------------------- | ------------------------------------------------------------------ |
| `benchmarks/state_preparation/calibration.py` | `da077c436ddd200ecab2d4fff0827a403ca800937a5cc0858a96d2469d258e92` |
| `experiments/noisy_krotov_causal_study.py`    | `32a0778a83b6f6e75228c11dee924d864e69e82352eb408b13deef342b015921` |
| `experiments/noisy_krotov_hybrid_smoke.py`    | `95ff439fd06125018610e5ed538bb005eddaedca81d5ebae8ca2cdbe85ca9c02` |

Verification: all 283 optimization tests pass; the fixed-target regeneration
check passes; scoped lint/type checks pass for all changed implementation and
documentation files. Full `uvx nox -s lint` remains blocked by pre-existing
manuscript/script/MPS-test lint issues, a missing Homebrew Node `simdjson`
library, and missing matplotlib in the nox lint environment. Incidental
formatter changes outside this work were restored.

The completed artifact audit verifies all 1,500 paired test rows, result
checksums, shared initialization, recomputed replicate-level intervals, and
pairwise-disjoint seeds: 1,800 unique training seeds, 36 validation seeds, and
1,500 test seeds. Both the original and corrected complete causal studies
resume without changing their result bytes. Runtime versions match the
original study: Python 3.11.9, NumPy 2.4.6, and SciPy 1.17.1.

## Implications for the next experiment

Keep the corrected calibration and the noiseless-continuation control.
Before scaling to the full target matrix, test a larger training ensemble and
a larger disjoint validation ensemble, with more independent training seeds.
Fixed CRN and resampling are viable comparison arms; this run provides no
reason to privilege refresh-5 or hybrid. Keep work-normalized controls in the
next design: equal iteration counts do not imply equal computational work.
These are recommendations, not additional experiments performed in this rerun.
