# Exact-anchor SVRG Adam development screen

Status: prepared implementation; no scientific optimization has been launched by
this agent. The coordinator must allocate workers before execution. This is a
known variance-reduction method adapted to weighted trajectories and Adam,
not a claim of algorithmic novelty or of full-benchmark superiority.

## Evidence motivating this one setting

The authenticated `estimator-diagnostic` experiment contains 512 paired CV32 /
weighted64 gradient replications, four development targets, 32 replications per
Euclidean displacement 0, 0.05, 0.2 and 1.0 from the original parameters. An
independent audit recomputed every squared gradient error from saved vectors and
verified each reported MSE. All four targets have 32/32 lower CV errors at 0.05
and 0.2. Including an exact snapshot amortized over only five uses, the empirical
MSE-times-CPU ratios are 0.00155–0.00406 at 0.05 and 0.02482–0.06106 at 0.2.
At distance 1.0, the corresponding ratios are 0.739 and 0.740 for Gaussian and
TFIM, but 1.266 and 1.179 for Haar and random MPS. The latter two have no finite
asymptotic MSE-cost break-even reuse count at that displacement. Distance zero
cancels by construction and is not positive scientific evidence.

These results support an accurate nearby-anchor estimator for the fixed local
Pauli channel. They do not establish useful Adam anchor lifetime, favorable
optimization trajectories, or improvement across the full noise family. The
original diagnostic and its separately charged typing-only rerun are distinct
physical executions; neither is silently discarded from development cost.

## Mathematical adaptation and fixed protocol

For the local calibrated channel let `g(theta, omega)` be a trajectory gradient
of infidelity. The weighted estimator is

`G_R(theta; omega) = p0 g(theta, 0) + (1-p0)/(R-1) sum[j=1..R-1] g(theta, omega_j)`,

where the conditional paths contain at least one independent Pauli factor and
retain every selected factor, including multiple events. Noise probabilities do
not depend on theta. The exact-anchor correction is

`v_t = grad L(a) + G_16(theta_t; omega_t) - G_16(a; omega_t)`.

The two terms use identical physical maps and weights. The exact expectation is
`E[v_t | theta_t, a] = grad L(theta_t)`. The first correction at every anchor is
exactly zero. We still execute and charge both trajectory batches at that step.
An exact snapshot costs one separate density forward/adjoint gradient call.
The signed loss control variate is logged as an estimator, not treated as a
bounded physical fidelity or used for endpoint selection.

All three methods use the unchanged implementation of bias-corrected Adam:
learning rate 0.02, beta1 0.9, beta2 0.999, epsilon 1e-8, 150 accepted updates.
First and second moments persist across every anchor refresh. The methods are:

- `cv_adam`: 16 weighted correction paths, exact anchor refreshed immediately
  before updates 1, 6, 11, ..., 146. No clipping, restarts or adaptive refresh.
- `weighted_adam`: 32 weighted paths and the same Adam settings. This has equal
  trajectory-direction work per update and receives twice the conditional-path
  count up to the deterministic-stratum offset.
- `exact_adam`: exact local-channel gradient at each update, same Adam settings.
  This is the serious deterministic baseline, not an equal-trajectory comparator.

The fixed original q6 108-gate circuits and source parameters are authenticated
against `output/overnight-state-preparation-20260913-0108/starts.json`. Only
construction replica 11 of Gaussian, critical TFIM, Haar 1 and random MPS bond 2
is used. No held-out target is accessed. Every run starts from physical zero.
No architecture, gate-noise support, strength, cutoff, or target is changed.

## Randomness, checkpointing and endpoint

Seeds are SHA256-derived with the separate `development_training_svrg_v1`
domain and fixed target, construction replica and one-based update. The complete
600-update seed table is frozen and checked for overlapping per-path intervals.
CV16 and weighted32 share the no-factor path and first 15 conditional maps at
each update. The ordinary estimator additionally has 16 conditional maps;
its conditional weights are `(1-p0)/31` while the CV weights are `(1-p0)/15`.
Map order, matrix bytes, site placement, weights and shared-prefix checksums are
recorded. Different updates resample; a finite fixed ensemble is never optimized.

The checkpoint retains all parameters, Adam moments, completed update count,
exact anchor theta/gradient/loss/problem signature, anchor refresh index, every
completed update's stream/map audit, and separate work/timing counters. Durable
checkpoints are written every 10 accepted updates and at the endpoint. A caught
interruption can replay work after the last durable checkpoint, using the same
seeds and anchor schedule. An independent attempts ledger charges the discarded
work too. Uncatchable kills leave an unfinished attempt and require an external
resource record before automatic resume; their work is not silently set to zero.

All endpoints are update 150, never a best checkpoint. A distinct exact forward
local score is taken initially and after updates 10, 20, ..., 150 for CPU curves;
these diagnostics do not influence training or selection. Final evaluation then
calls all twelve fixed benchmark conditions independently, in explicit stable
order, including noiseless and Ballarin. There is no final-evaluation randomness
to overlap with training. The local final score must agree with the separate
last diagnostic to 1e-10. Completed scientific files are read-only on resume;
derived CSV ordering is explicit and stable across JSON key reordering.

## Resource accounting and comparisons

Per target before diagnostics and final scoring:

| Method          | Updates | Exact joint gradients | Batched trajectory-gradient calls | Propagated paths | Trajectory directions | Density gradient directions |
| --------------- | ------: | --------------------: | --------------------------------: | ---------------: | --------------------: | --------------------------: |
| CV16 Adam       |     150 |            30 anchors |                               300 |             4800 |                  9600 |                          60 |
| Weighted32 Adam |     150 |                     0 |                               150 |             4800 |                  9600 |                           0 |
| Exact Adam      |     150 |                   150 |                                 0 |                0 |                     0 |                         300 |

Each method additionally pays 16 exact forward curve diagnostics and 12 exact
forward final evaluations. A trajectory-gradient call processes a vectorized
batch; the paths column counts both current and anchor propagation. A forward
score is one direction, a joint gradient has two. Density and statevector
directions have different costs and must not be called equivalent compute.

Process CPU and wall time include backend setup, sampler construction, sampling,
map hashing, snapshot integrity checks, numerical estimators and Adam updates.
Checkpoint JSON I/O, custody checks, imports and orchestration are excluded from
training CPU and reported separately through whole-invocation timings. Diagnostic
CPU and final-evaluation CPU are separate and fully charged. The attempts ledger
retains physical work including failed/discarded invocations; counters for an
interrupted vectorized call are attempted-work upper bounds. On ordinary complete
runs they are exact. Common-CPU curves must use training plus diagnostic CPU and
the last available iterate at or before the common cutoff, never the largest
fidelity seen. If an interruption occurred, do not use logical curve times as
physical-cost matched comparisons without adding the discarded-work ledger.

Compare local-channel endpoints and target-balanced mean over all eleven noisy
conditions separately. Report target-level mean differences, mean, median, worst
target, worst noise family and paired wins, including Haar. Report equal-update,
equal-trajectory-direction and common-CPU comparisons with their appropriate
labels. Exact Adam can be faster in q6 despite more expensive individual oracles.
Unchanged starts and archived structural/adaptive-online endpoints remain
unmatched references from the coordinator's existing scoreboard; they cannot
establish a fair method win against this fixed-architecture training screen.

## Promotion, falsification and next setting

The previous estimator experiment has passed only the nearby-anchor mechanism
gate. This screen asks whether the benefit survives actual parameter updates.
Record anchor distance and age at every step, and inspect whether four intervening
Adam updates leave the supported nearby-anchor regime. A poor endpoint cannot
be rescued by selecting a better earlier iterate.

Promote the training candidate for a compact follow-up only if, across all four
targets, the all-eleven-noise-condition mean improves over weighted Adam by at
least 0.5 percentage points or has a smaller positive gain on all four targets;
it must also be no more than 0.1 percentage points below exact Adam in the overall
mean, have no target-level mean loss larger than one point against the stronger
of the two controls, and have no unexplained failure. A substantial compute
advantage at comparable fidelity can instead justify promotion only if the
common-CPU curves and complete physical ledger establish it explicitly. Local
channel gains alone remain a restricted local-channel result.

Retire this training configuration if neither the fidelity nor compute gate
passes. Preserve the established estimator result even if Adam training fails.
Only after the refresh-5 training gate passes may the coordinator preregister
one refresh-10 alternate with the other settings fixed; there is no initial grid,
learning-rate rescue, or confirmation-data tuning. This development panel alone
cannot establish a new benchmark baseline or justify MPS scaling.

## Provenance and reproduction

The anchor gradient plus paired stochastic difference is adapted from
[Johnson and Zhang, NIPS 2013](https://proceedings.neurips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html).
Original SVRG convergence claims concern their assumptions and algorithm;
combining it with persistent Adam on this nonconvex circuit is an experimental
adaptation, not a transfer of their guarantee. Adam follows
[Adam: A Method for Stochastic Optimization, 2014](https://arxiv.org/abs/1412.6980). The exact conditional Pauli
sampler and calibrated density/statevector kernels are existing repository code.
The reduction in nearby gradient error is empirical evidence from the saved
repository diagnostic, not a claim made by these papers.

Prepare without running:

```sh
uv run python -m experiments.overnight_state_preparation_20260913_0108.svrg_experiment --output output/overnight-state-preparation-20260913-0108/svrg-refresh5 --prepare-only
```

Execute only after coordinator allocation, with single-thread numerical kernels:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run python -m experiments.overnight_state_preparation_20260913_0108.svrg_experiment --output output/overnight-state-preparation-20260913-0108/svrg-refresh5 --run --workers 1
```

Repeating the run command authenticates sources and resumes complete accepted
updates. The module does not overwrite a mismatched prior protocol or numerical
result. Engineering tests use q2 and reduced budgets; they are not scientific
benchmark outcomes.
