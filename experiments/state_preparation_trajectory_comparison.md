# Experiment 2: trajectory hybrid Krotov versus Adam

Protocol frozen before development tuning and fresh-target training on 2026-09-10.
This is a small, descriptive competitiveness screen, not a state-of-the-art claim.
It asks whether trajectory optimization can recover the noise-aware headroom
observed in [Experiment 1](state_preparation_exact_adam_findings.md), and whether
the hybrid online phase helps relative to a matched batch-only update.

Implementation update (2026-09-11): `--backend statevector` now runs construction
and pure-state optimizers on the shared dense backend; `mps` remains the default.
The frozen run described here used MPS. This option changes representation, not
the noise law, estimator, schedule or target panel. Exact Adam and scoring remain
density-matrix based. Use a fresh output directory and record timings separately;
see the [backend guide](statevector_backend.md). No new scientific run is implied
by this implementation update.

## Scientific settings

- Six qubits, pure target, input zero state, loss
  `1 - E[|<target|psi_trajectory>|^2]`. Report exact mixed-state target fidelity.
- Same hardware-motivated homogeneous local depolarizing model as Experiment 1:
  average gate infidelities `r1 = 0.00064`, `r2 = 0.0051`; exact local Pauli
  Lindblad exponential after every retained abstract gate. These are not native
  device execution/calibration claims. No change to the original benchmark rates.
- Per-axis dimensionless Lindblad exposures:
  `a1 = -log(1 - 2*r1)/4` and
  `a2 = -log((sqrt(16 - 20*r2) - 1)/3)/4`.
  Each site's independent X/Y/Z factor is sampled with
  `q = (1 - exp(-2*a))/2`, including all multiple-factor events.
- Untruncated complex128 MPS trajectories, independent-trajectory fidelity
  estimator (not the cross-trajectory estimator), `R = 32` per noisy update.
- Fresh paths every update. Paths are fixed within the old-forward/adjoint and
  updated-prefix sweep. Across optimizers and learning-rate candidates, the same
  case/update/path uses the same random stream. This is paired common randomness,
  not freezing one training ensemble for all iterations (`use_crn=False`).
- SHA-256 role/case seeds rooted at `2026091001`; base seeds separated into
  disjoint 2^32-sized blocks. Iterations stay numbered 1 through 50 across
  checkpoints and the online/batch switch.

## Development selection, then fresh-target confirmation

Only four old targets (one per family, initialization replica 11) from the
authenticated `H25_lr0p7:cold_final` extension are used for learning-rate selection.
Each method receives three configurations, each run for all 50 updates. The
selection criterion is equal-family mean final exact noisy fidelity, with the
smaller tested rate breaking exact ties. No per-target choices or best-iterate
selection. Development is 60 runs, separate from the 80 confirmation runs.

| Method                   | Three development rates         | Main comparison                                |
| ------------------------ | ------------------------------- | ---------------------------------------------- |
| Noiseless hybrid Krotov  | online 0.1, 0.3, 0.7; batch 0.1 | selected globally                              |
| Exact-noisy Adam         | 0.001, 0.005, 0.02              | selected globally                              |
| Trajectory Adam          | 0.001, 0.005, 0.02              | selected globally                              |
| Trajectory hybrid Krotov | online 0.1, 0.3, 0.7; batch 0.1 | selected globally                              |
| Trajectory batch Krotov  | prefix 0.1, 0.3, 0.7; tail 0.1  | **rate matched to selected trajectory hybrid** |

Both hybrid arms use 25 gate-sequential online sweeps then 25 batch gradient
steps. Batch ablation uses 25 batch steps at the hybrid's online rate, then 25
at 0.1; thus ordering is the only change. Its independent development winner is
reported but does not override this matching rule. Adam uses beta1=0.9,
beta2=0.999, epsilon=1e-8, standard bias correction and no clipping/line search.
Even noiseless hybrid is selected using noisy development evaluation: its
training is noiseless, but this comparator is not wholly noise-blind.

The selection artifact is written before any fresh-target starts are trained.
Fresh targets are frozen in a separate vector/metadata manifest; the original
collaborator fixture is not changed:

| Family                   | Fresh target 1                      | Fresh target 2              |
| ------------------------ | ----------------------------------- | --------------------------- |
| Gaussian amplitudes      | mean 0.35, sigma 0.08               | mean 0.65, sigma 0.14       |
| Uniform open-chain TFIM  | J=1, h/J=0.7, eigensolver seed 9201 | J=1, h/J=1.3, seed 9202     |
| Complex Haar             | seed 9101                           | seed 9102                   |
| Real Gaussian random MPS | bond dimension 2, seed 9001         | bond dimension 2, seed 9002 |

Gaussian amplitudes are the square root of density, using the original
coarsest-bit-first quantics grid and little-endian vector convention. All vectors
are normalized and have a fixed global phase. TFIM is `-sum ZZ - h sum X`.
Each of the eight targets has two construction replicas (11, 29), giving 16
paired starts, not 16 distinct targets.

## Common construction and refinement

Use the previous winning noiseless construction recipe unchanged: four BMPD
growth stages, 100 updates each (25 online at 0.7 then 75 batch at 0.2); initial
parameters 0.1 times a standard normal master vector, newly grown parameters
0.001 times that same vector. Six magnitude-pruning rounds remove 10 one-qubit
and 5 two-qubit gates each, with 20 batch repair steps at 0.1 after each round;
protect the initial single-qubit layer. Finish with 50 batch steps at 0.1.
This is 570 shared preparation updates per start and leaves 78 one-qubit and
30 two-qubit gates, with 108 parameters. Deleting gates also deletes their
noise windows. There is **no further pruning during the five refinements**.

All five methods receive exactly the same circuit and parameters for each case.
Each gets 50 additional refinement updates. No early stopping, target-specific
retuning, extra optimizer restarts, or silently dropped failures. Exact fidelity
evaluations at updates 0, 10, 25 and 50 are diagnostic; update 50 is primary.

## Cost, reproducibility, and decision rule

Measure optimizer CPU and wall time per update with one numerical thread per
worker, separately from target setup, shared construction, checkpoint I/O and
exact density evaluation. Up to eight process workers run independent jobs.
CPU time is primary under concurrent execution; summed wall times are not an
isolated latency benchmark. The shared construction cost belongs to every
end-to-end method and cannot be omitted from any end-to-end speedup claim.

Use the existing Krotov one-update kernels directly to avoid charging Krotov for
the public trainer's extra per-update diagnostic trajectory evaluations when
Adam has none. Tests compare these updates with the public hybrid trainer.
Trajectory Adam and batch Krotov use the identical sampled gradient routine.
Each noisy arm draws 1,600 training paths, but online sweeps also propagate the
updated prefix: equal update/path counts do **not** imply equal computation.
Process high-water RSS includes imports, previous jobs, and exact evaluation;
it is recorded as context only, not a method-specific memory benchmark.

Report paired changes versus frozen/noiseless starts, exact Adam, trajectory
Adam, and matched batch, per family and per target with both replicas. Equal
weighting: families, then targets, then replicas. Practical screen:

1. At least +0.5 percentage points of noisy fidelity over a comparator at
   comparable measured optimization cost (within 20% CPU), with gains in every
   family; or
2. Within 0.2 percentage points of the strongest reference with at least 1.5x
   lower refinement CPU cost, without hiding family-level regressions.

These thresholds are screening choices, not statistical significance tests.
With two targets per family, do not claim population robustness. Six qubits
cannot establish tensor-network memory scaling; an exact dense optimizer is a
small-system reference, not a proposed scalable baseline. If neither criterion
is met, report that competitiveness is not demonstrated, even if mean fidelity
improves. No post-result expansion of the sweep is part of this experiment.

Run with:

```console
uv run --no-sync python -m experiments.state_preparation_trajectory_comparison \
  --source-root /Users/maximilianfrohlich/Desktop/yaqs-experiment-results/q6-hybrid-construction-extension-20260906 \
  --output-root output/state-preparation-trajectory-comparison-20260910 --workers 8
```

`--stage development` stops at the frozen selection. `--resume` authenticates
the scientific/runtime manifest and all completed checkpoints, preserving
parameters, Adam moments, noise iteration counters and recorded timings.

After completion, reproduce the independent fidelity audit and paired summaries:

```console
uv run --no-sync python -m experiments.state_preparation_trajectory_analysis \
  --output-root output/state-preparation-trajectory-comparison-20260910
```

The audit recomputes noisy and noiseless fidelity at all four checkpoints for
all 80 confirmation runs (640 independent checks), checks phase/rate traces and
artifact identities, and writes `audit.json` without changing training records.
