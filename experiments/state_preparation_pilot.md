# Small compression-first state-preparation pilot

Version: `yaqs.compression_first_pilot.v1`, frozen 2026-09-05 before launch.

This is a separate exploratory screen of promising construction/refinement
routes, not a reduced execution of the
[full research protocol](state_preparation_next_experiments.md). It cannot
establish state-of-the-art performance or target-independent generalization.
No hyperparameters, targets, seeds, budgets, or selection rules are changed
after looking at this pilot's results.

## Fixed panel and numerical settings

- Six qubits, initial state `|000000>`; normalized targets from the restored
  fixture: `gaussian_mu0p5_sigma0p1`, `tfim_critical`, `haar_random_1`, and
  `random_mps_bond2`.
- Two optimization seeds per target: 11 and 29. Root seed: 2026090520.
  Seeds derive from SHA-256 of version, root seed, target, replica, and training
  role, never worker order. Target randomness is frozen in the fixture.
- Scalar BMPD adjoint circuit with the initial U3 product-state layer enabled.
  Trainable initial-layer gates are protected from pruning. Circuit parameters
  retain the canonical builder's parameter order, not chronological gate order.
- Noiseless loss is `1 - |<target|psi(theta)>|^2`. Noisy loss is its expectation
  under the channel below. All optimization is batch Krotov-inspired discrete
  adjoint gradient descent: `theta <- theta - eta * gradient`. No hybrid/online
  phase, momentum, clipping, line search, or best-checkpoint selection.
- Complex128 MPS evolution; no bond cap, zero SVD truncation threshold, minimum
  bond dimension one. No density-matrix training.

## Stage 1: construction and compression

A shared depth-four master standard-normal parameter vector is drawn separately
for each target/seed. Depth one starts at `0.1 * z`; new parameter slices added
at later depths start at `0.001 * z`, retaining the optimized earlier slice.
The source grows through depths 1, 2, 3, 4 with 100 noiseless updates per depth
at learning rate 0.2. Save all source blocks.

| Route | Construction after shared prefixes                                | Final one-qubit / two-qubit gates | Total noiseless updates |
| ----- | ----------------------------------------------------------------- | --------------------------------- | ----------------------- |
| L2    | Depth-two prefix; 200 more updates at 0.2, then 120 at 0.1        | 78 / 30                           | 520                     |
| L4    | Depth-four source; 120 control updates at 0.1                     | 138 / 60                          | 520                     |
| M     | Depth-four source; six magnitude-pruning rounds with repair       | 78 / 30                           | 520                     |
| D     | Depth-four source; six deletion-impact-pruning rounds with repair | 78 / 30                           | 520                     |

The shared source is computed once but charged to every route that uses it.
Matching update counts is not matching compute: D additionally evaluates every
eligible single-gate deletion, and gradient cost depends on circuit size.
Retain timings and work counts; do not present a same-time optimizer ranking.

Every pruning round removes exactly ten one-qubit gates and five two-qubit
gates simultaneously, followed by twenty noiseless repair updates at learning
rate 0.1. Rescore after each repair. The initial 18 parameter origins remain
protected. Six rounds remove 90 gates/parameters, leaving 108 from the original
198, with exactly the depth-two resource counts.

- M score: absolute physical angle wrapped into `[-pi, pi)`, including gate
  angle scale/offset.
- D score: actual finite-deletion noiseless loss minus the current noiseless
  loss. Negative scores are allowed. This is not a local-gradient proxy.
- Select the smallest scores independently by gate arity. Deterministic tie
  key: `(round(score / 1e-12), magnitude, original_parameter_id)`.
- A deletion removes the entire unitary and its noise window. Compact remaining
  parameters in original parameter-index order; preserve chronological gate
  order and immutable original IDs. Merely setting an angle to zero is not
  deletion. Do not opportunistically remove any additional gates.
- Preserve all scores, removed origins, before/after-deletion fidelity, repair
  traces, repaired fidelity, and final parameter vectors.

Evaluate all four final construction endpoints with the exact noisy channel.
After all eight target/seed jobs finish, nominate **one global pruning rule**
M or D by equal-target mean exact noisy fidelity, averaging the two seeds first.
An improvement at most `1e-12` chooses cheaper M. Do not pick a different rule
for each target. This nomination uses development data and is not held-out
evidence. Both construction rules remain in the reported results.

## Stage 2: matched refinements

For each target/seed, take L2, L4, and the globally nominated pruning endpoint.
Fork each fixed parameter vector into two arms:

- `0`: 50 additional noiseless batch updates, learning rate 0.1.
- `N`: 50 noisy batch updates, learning rate 0.1, eight independent MPS
  trajectories per update, resampled at every update. The gradient averages
  eight single-trajectory contributions exactly once. No cross-trajectory
  estimator, frozen training sample, or CRN search is used.

Record identical starting-parameter checksums for each pair and retain frozen
pre-refinement controls. Evaluate the final iterate at update 50, not the best
training or evaluation checkpoint. There are 24 noisy and 24 matched noiseless
refinement runs; noisy training uses 1200 ensemble updates / 9600 trajectories.
Together the two stages evaluate 32 construction and 48 refinement endpoints.

## Calibrated noise and exact evaluation

Use the [benchmark's hardware-motivated local depolarization calibration](../benchmarks/state_preparation_benchmarks.md#common-rate-convention):
whole-window average gate infidelities `r1 = 0.00064` and `r2 = 0.0051`.
These are reference magnitudes, not a native compilation or device-specific
model of every BMPD primitive. No additional noise model or rate sweep is
included in this pilot.

For every retained abstract gate, follow its ideal unitary by a unit-duration
Lindblad window on each site of that gate. With
`D[P](rho) = P rho P - rho`, the integrated coefficient per X/Y/Z process is:

```text
a1 = -log(1 - 2*r1)/4                  = 0.000320204974931...
lambda2 = (4*sqrt(1 - 5*r2/4) - 1)/3
a2 = -log(lambda2)/4                  = 0.001066470265101...
E_k = exp(a_k * sum_{site on gate, P in X,Y,Z} D[P_site])
```

Independent local Pauli dissipators commute as superoperators. For each site
and each P, sample a Bernoulli event with `q = (1-exp(-2*a_k))/2` and apply P
when selected. This exactly samples each factor of `E_k`; it is a pure-state
unraveling of the same Lindblad channel, not the legacy at-most-one-jump
approximation. The pilot's provider/version explicitly records this distinction.
Previous artifacts and samplers are left unchanged. This exact factorization
is specific to this commuting Pauli model, not a claim about arbitrary noise.

The independent evaluation oracle propagates a 64-by-64 density matrix using
per-site Pauli mixture weights `(1-exp(-4*a_k))/4`. It checks trace, Hermiticity,
positivity, and fidelity bounds. Pure-state trajectory norms, gradients,
parameters, and losses are checked for finite values throughout training.
Sampler branch enumeration is tested against a separately built Lindblad
matrix exponential and against the requested average gate errors. The noiseless
oracle is tested against MPS evolution for all four target families; sampled
gradients are tested against fixed-map finite differences.

Report exact `F = <target|rho|target>` and infidelity, not finite-trajectory
evaluation noise. Exact evaluation removes measurement/sampling uncertainty
from an endpoint, not the variability caused by training seeds. With only two
seeds and one state per family, results remain descriptive.

## Execution, artifacts, and interpretation

Run from the repository using a fresh external output directory:

```bash
uv run python -m experiments.state_preparation_pilot \
  --output-root /Users/maximilianfrohlich/Desktop/yaqs-experiment-results/q6-compression-pilot-20260905 \
  --workers auto
```

Eight independent target/seed jobs use eight processes on this laptop, each
with one BLAS/OpenMP numerical thread. Construction completes for the whole
panel before global nomination and the parallel refinement stage. `--workers 1`
is a reproducibility/debugging option, not a different scientific configuration.

The manifest records exact source, target-fixture and configuration checksums,
noise-provider metadata, seeds and dependency versions. Each job owns its
directory. Training blocks checkpoint atomically every ten updates and at the
last update. The parent alone aggregates completed jobs. Use the same command
with `--resume` only for a compatible interrupted run; incompatible manifests
and corrupt checkpoints are rejected. A failed job is not silently omitted.

Main artifacts: `manifest.json`, `nomination.json`, `result.json`, and each
target/seed's `construction.json`, `refinement.json`, and `blocks/*.json`.
All JSON artifacts carry content checksums. Wall times include evaluation and
pruning scoring where recorded; training blocks preserve update traces.

Assess, in order:

1. Does pruning improve exact noisy fidelity compared with L4, and does it beat
   the equally compact L2 control, rather than merely reducing gate counts?
2. Does noisy refinement improve on **matched noiseless continuation**, as well
   as the frozen start, across several target families?
3. Are any gains consistent across the two seeds, worth their additional compute,
   and large enough to justify a separately specified confirmation study?

Report every target and seed, including failures and unfavorable outcomes.
Do not call a post-nomination comparison unbiased confirmation. Do not enlarge
this pilot, retune rates, or start the full research matrix based on an early
partial result.
