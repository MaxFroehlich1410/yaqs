# Weighted trajectory optimization: frozen development protocol

Frozen before execution on 2026-09-12. This tests whether the earlier weighted
gradient diagnostic translates into better optimization, giving exactly the
same estimator to Krotov and Adam. No new targets, reconstruction, pruning,
learning-rate search, early stopping or validation-based step selection.

## Matched design

Use the four authenticated Experiment-2 development starts in
`output/state-preparation-trajectory-comparison-20260910`: Gaussian
`gaussian_mu0p5_sigma0p1`, `tfim_critical`, `haar_random_1`, and
`random_mps_bond2`. Each is q6, construction replica 11, with its original
108-gate frozen circuit and parameter vector. Noise replicas 11 and 29 are
repeated noise streams, not additional targets or construction starts.

| Method            | Updates 1-25     | Updates 26-50    |
| ----------------- | ---------------- | ---------------- |
| All-online Krotov | Online, rate 0.7 | Online, rate 0.7 |
| Hybrid Krotov     | Online, rate 0.7 | Batch, rate 0.3  |
| Trajectory Adam   | Adam, rate 0.02  | Adam, rate 0.02  |

Adam uses beta1=0.9, beta2=0.999, epsilon=1e-8, zero initial moments and
standard bias correction. Every method runs 50 updates with fresh maps each
update. Use both ordinary and weighted sampling at total path budgets R=32
and R=128: 4 targets x 2 replicas x 3 methods x 2 samplers x 2 budgets =
96 optimization runs and 4,800 updates. Ordinary controls are rerun under the
same implementation and timing conditions, not imported timing measurements.

Use the unchanged calibrated local Pauli channel: one-/two-qubit average gate
errors 0.00064/0.0051, integrated per-axis Lindblad strengths
0.00032020497493061084/0.0010664702651014273. Apply independent factors with
probabilities `(1-exp(-2*a))/2` to each operand of each noise-enabled gate.
The zero input, little-endian statevector convention, and exact density scoring
remain unchanged. This is a parameter-independent random-unitary channel,
not normalized nonunitary jump optimization.

## Weighted estimator and online update

Let p0 be the exact probability of selecting no Pauli factor anywhere in the
fixed circuit (approximately 0.7658428757 for these circuits). Propagate one
deterministic no-factor path with weight p0, and R-1 independent conditional
paths given at least one selected factor, each with weight `(1-p0)/(R-1)`.
The loss is the weighted sum of individual infidelities, not an overlap with
an averaged state. The conditional sampler retains cancellations and multiple
factors. It samples the first event exactly and all subsequent Bernoulli
factors without rejection or truncation. Probabilities do not depend on theta.

Both batch Krotov and Adam receive the gradient of this same weighted sample
loss. Online Krotov computes old costates with trajectory weights applied once,
then sums all weighted path contributions **before each gate update**, followed
by propagation of every path through that updated gate. Never optimize paths
individually and average their final parameters. The batch estimator is
unbiased for the exact channel gradient; this does not assert that a finite
stale-adjoint online sweep is an unbiased batch-gradient step or monotone.

Root seed 2026091101 and the existing `training` seed namespace are retained.
Within each sampler and path budget, all methods use identical maps and
weights at an update, checked by physical-ensemble hashes. R32 paths are a
prefix of R128 paths (conditional prefixes for weighted sampling); conditional
weights change with R. Ordinary and conditional draws use the same seed roots
but are different sampling laws, not identical physical paths. Maps are replayed
for all forward/backward operations within an update, refreshed across updates.

## Assessment and execution

Primary endpoint: exact noisy fidelity after update 50, equally averaged over
the four targets and two replicas. Report paired per-target/replica differences,
not only pooled winners. Record update-25 and update-50 scores, fidelity versus
cumulative training CPU, setup/sampling/update costs, scoring costs and path
passes (online 3R versus batch/Adam 2R per update, not gate operations).
CPU includes sampler and engine setup; exact evaluation, checkpoint hashing
and disk logging are excluded and should not be called total runtime. Scores
are descriptive only. Compare weighted versus ordinary within each method,
and weighted Krotov versus weighted Adam at equal updates, and with a
descriptive common-CPU checkpoint comparison. No interpolation or best-checkpoint
selection is needed for the latter; use the last completed update within a
shared CPU budget. Eight pairs are only four distinct, repeatedly used targets.

Use independent processes with one numerical thread each, up to eight workers,
fixed shuffled job order, atomic checkpoints and source/runtime-checked resume.
Software changes intentionally invalidate older resume fingerprints; earlier
scientific artifacts are preserved. Do not mix old and new timings.

```bash
uv run --no-sync python -m experiments.state_preparation_weighted_optimization \
  --source-root output/state-preparation-trajectory-comparison-20260910 \
  --output-root output/statevector-weighted-optimization-20260912 \
  --workers auto
```

This screen is exploratory, not held-out confirmation or a state-of-the-art
claim. Freeze and test a genuinely new target panel only after a configuration
is competitive. A variance-reduction benefit shared by Adam is not a
Krotov-specific novelty claim.
