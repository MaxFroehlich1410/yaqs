# Adaptive online Krotov: frozen development protocol

Frozen before execution on 2026-09-12. This is the agreed 64-run follow-up to
the [weighted optimization findings](state_preparation_weighted_optimization_findings.md).
The question is whether parameter-wise adaptive scaling helps the **sequential
online** method compete with weighted Adam at matched training CPU. It is not
an adaptive batch tail relabeled as an online Krotov result.

## Fixed cases and noise

Reuse the authenticated Experiment-2 development cases from
`output/state-preparation-trajectory-comparison-20260910`: q6 Gaussian
`gaussian_mu0p5_sigma0p1`, `tfim_critical`, `haar_random_1` and
`random_mps_bond2`, each with its construction-replica-11 circuit and parameters.
Each fixed circuit has 108 parameterized gates (78 one-qubit, 30 two-qubit).
No new target states, construction starts, pruning or circuit changes.

Use noise replicas 11/29 and root seed 2026091101 in the existing training
namespace. All methods receive the same maps at each of 50 updates, refreshed
every update and replayed consistently within it. Noise replicas are not
additional target states. Numerical training is complex128 statevector based.

The calibrated local Pauli-Lindblad model remains unchanged: average gate
errors 0.00064/0.0051, integrated per-axis coefficients
0.00032020497493061084/0.0010664702651014273, independent Bernoulli factors
with probabilities `(1-exp(-2*a))/2` after enabled gates on their operands.

Use weighted R=32 throughout: one deterministic no-selected-factor path of
weight p0=0.7658428756735005, plus 31 paths conditioned on at least one selected
factor, each with weight `(1-p0)/31`. Multiple events and cancellations remain
in the conditional law. Probabilities are independent of parameters; this
does not extend to normalized nonunitary jumps. The objective is the weighted
sum of individual trajectory infidelities, never an averaged state overlap.

## Eight methods, eight target/noise pairs each

| Method                          | Updates 1-25 | Updates 26-50     | Rates screened    |
| ------------------------------- | ------------ | ----------------- | ----------------- |
| Weighted Adam                   | Adam         | Adam              | 0.005, 0.01, 0.02 |
| All-online Krotov control       | Online 0.7   | Online 0.7        | One fixed control |
| Adaptive-online method          | Online 0.7   | RMS-scaled online | 0.005, 0.01, 0.02 |
| Krotov-Adam combination control | Online 0.7   | Adam 0.02         | One fixed control |

This is 3+1+3+1 = 8 methods x 4 targets x 2 replicas = **64 runs, 3,200
updates**. Rerun all controls under the same code revision and timing
conditions; do not import old timings. Use fixed shuffled job order and eight
independent single-threaded workers. No additional rate grid or target panel
will be launched based on these results within this screen.

Adam uses beta1=0.9, beta2=0.999, epsilon=1e-8 and standard bias correction.
The combination control initializes both Adam moments and its local counter
at the phase switch, and is labeled a combination rather than a pure Krotov win.

## Exact adaptive-online rule

At the switch, initialize second moments v_j=0 and occurrence counters n_j=0.
There is **no first moment** and no adaptive statistics collected in the prefix.
For each subsequent sweep:

1. Draw the paired weighted ensemble. Propagate at the old parameters and
   build weighted old costates, applying trajectory weights exactly once.
2. Walk gates in their fixed order. With the currently updated forward prefix,
   sum the weighted trajectory contributions to the local scalar signal s_j.
3. At each trainable parameter occurrence, update
   `v_j = beta2*v_j + (1-beta2)*s_j**2`, `n_j = n_j+1`, then
   `theta_j -= alpha*s_j/(sqrt(v_j/(1-beta2**n_j)) + epsilon)`.
4. Recompute that gate at the updated parameter and propagate all paths
   through it and their original noise maps before proceeding to the next gate.

Here beta2=0.999 and epsilon=1e-8. Shared parameters, if present, update their
counter and moment at every occurrence; bias correction is not a global
sweep counter for such circuits. The current scientific circuits have one
occurrence per parameter. Square the **ensemble-summed** local signal, not
the individual path signals. There is no extra batch gradient, adaptive
clipping, line search, exact-loss acceptance or early stopping.

This is an adaptively preconditioned Krotov-inspired sequential scheme. It is
not classical monotone Krotov, nor ordinary batch RMSProp, and no convergence
guarantee is asserted for this finite stochastic update.

## Assessment and stopping decision

Record exact density fidelity at update 0 and after every update; these scores
are descriptive only and never used in updates. Record full phase/moment/
counter state for resume, physical-ensemble hashes, update norms, CPU/wall
costs, and path passes. Both online rules require 3R trajectory-direction
passes per sweep; Adam uses 2R. Pass counts are not local gate operations.

Primary comparison: exact noisy fidelity at matched measured training CPU.
For each fixed pair of methods and target/replica, choose the smaller final
CPU total and the last completed checkpoint inside that budget; never choose
the maximum-fidelity checkpoint or interpolate. Also report all fixed
update-50 endpoints, late gains, per-target paired differences and win counts.
Account for engine/sampler setup, sampling and updates; report exact scoring
separately. Replay hashing, disk I/O and orchestration are excluded from
training CPU. Laptop all-core timings are descriptive, not a microbenchmark.

Report every rate. If identifying the strongest rate, use one global rate per
method family for the whole development panel, never per-target cherry-picking.
Describe any selection as exploratory and retain the total three-rate tuning
costs. The combination gets only its predeclared 0.02 rate; it is a practical
control, not a matched hyperparameter-sweep superiority claim.

A promising result must improve the cost-fidelity comparison over the Adam
rate screen consistently enough to justify further work, rather than simply
beat unscaled Krotov. With only four previously used states and two noise
replicas, no formal significance or target-independent advantage is claimed.
If the adaptive method fails this check, pause expansion of these schedules
and reassess the method. Do not proceed automatically to new targets.

```bash
uv run --no-sync python -m experiments.state_preparation_adaptive_online \
  --source-root output/state-preparation-trajectory-comparison-20260910 \
  --output-root output/statevector-adaptive-online-20260912 \
  --workers auto
```

Use `--resume` only with the same source/runtime/configuration. The new online
kernel and checkpoint fields change scientific fingerprints; old artifacts
remain preserved but are not resumed under the new implementation.
