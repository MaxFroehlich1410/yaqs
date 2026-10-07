# Experiment 1: exact-noisy versus noiseless Adam continuation

Version `yaqs.exact_adam_headroom.v1`, specified 2026-09-06 before the run.
This is a small development-data diagnostic, not Experiment 2, a trajectory
experiment, or a claim of competitive performance against published methods.

## Frozen panel, starts, and budget

Use exactly the eight `H25_lr0p7:cold_final` checkpoints (after 570 updates)
from `q6-hybrid-construction-extension-20260906`: six qubits, four fixed targets
(`gaussian_mu0p5_sigma0p1`, `tfim_critical`, `haar_random_1`,
`random_mps_bond2`), initialization seeds 11 and 29. These are one state per
family and two initialization replicas, not eight independently sampled states.
Authenticate source manifests/results, target fixtures, circuit/parameter
checksums, and recompute starting fidelities before continuation. Preserve all
source artifacts unchanged and copy the selected starting records to new output.

Freeze each circuit at 78 one-qubit and 30 two-qubit gates / 108 parameters.
There is no circuit growth, pruning, compilation, angle clipping, or gate removal.
Fork each parameter vector into `noiseless` and `exact_noisy` arms, retaining a
frozen-start reference. Both arms receive exactly 100 Adam updates:

- Constant learning rate 0.005; beta1=0.9, beta2=0.999, epsilon=1e-8.
- Initialize both moment vectors to zero. Bias correction uses t=1,...,100.
- Standard Adam: theta <- theta - eta \* m_hat / (sqrt(v_hat) + epsilon).
- No weight decay, AMSGrad, clipping, line search, learning-rate sweep, early
  stopping, restarts, or best-checkpoint selection. Final iterate is primary.
- Exactly 16 continuations / 1600 gradient updates; no training randomness.

The learning rate is a fixed, modest Adam reference (also used in RobustState),
not a conversion of the previous Krotov rate or a claim of optimal tuning.
A negative result warrants a separately declared convergence/rate diagnostic,
not an automatic conclusion that no useful noisy optimum exists.

## Objectives and exact derivative

Both arms use the same complex128 density-matrix adjoint implementation, without
tensor truncation or trajectory sampling. The initial state is |000000> and
the loss is L=1-<target|rho(theta)|target>.

For `noiseless`, all channel strengths are zero. For `exact_noisy`, preserve the
pilot's hardware-motivated homogeneous local Pauli-Lindblad channel. Complete
one-/two-qubit window average errors are r1=0.00064 and r2=0.0051, giving
integrated coefficients per site and X/Y/Z process:

```text
a1 = 0.00032020497493061084
a2 = 0.0010664702651014273
E_site(A) = lambda*A + (1-lambda)*(I_site/2 tensor Tr_site(A))
lambda = exp(-4*a)
```

Apply independent site channels after every retained abstract gate on its support,
including zero-angle gates. These are not native-device calibrated circuits;
there are no coherent control, readout, or idle errors in this model.

Store the density matrix immediately after each ideal gate V*k=U_k rho*(k-1)
U*k^dagger. Back-propagate C_K=|target><target| using
C_pre=E_k^dagger(C_k), C*(k-1)=U_k^dagger C_pre U_k. The noise channel is
self-adjoint. For dU/dtheta=D U, the loss derivative contributed by gate k is
-Re Tr[C_pre^dagger (D V_k + V_k D^dagger)], including the gate's angle scale.
Sum contributions for shared parameters. The channel is independent of theta;
this formula must not be reused unchanged for angle-dependent noise.

Validate against central finite differences of the independent Qiskit density
oracle, including reversed/nonadjacent sites, scales/offsets, shared parameters,
disabled noise, and zero noise. At zero noise also compare with the existing
MPS adjoint. Recheck saved q6 starting values and directional derivatives on
every target/seed before launching the full continuation matrix.

## Evaluation, artifacts, and interpretation

Record both fidelities at updates 0,10,20,...,100 using the independently
implemented Qiskit density oracle. Save all pre-update losses, gradient/update
norms, parameters at evaluation checkpoints, Adam moments, and update counters.
Record final gradient norms and the last-20-update change to flag incomplete
convergence; never substitute an intermediate checkpoint for the final iterate.
Atomic checksummed checkpoints support exact moment-preserving resume and reject
configuration, source, fixture, or runtime-fingerprint changes.

Primary comparison: exact noisy fidelity at t=100 for exact-noisy minus
noiseless Adam. Secondary: each arm minus the frozen start. Average seeds within
targets and targets equally. Report every paired result, target mean, win count,
and noiseless fidelity tradeoff. A practically promising screening result is a
target-balanced gain of at least 0.5 percentage points with positive gains in
at least three of four families. Smaller gains remain reportable, not failures.
Do not use significance claims or generalization intervals from these four
adaptively reused target states. This is accessible improvement from a fixed
start and optimizer budget, not a global optimality bound.

Run up to eight independent single-threaded processes. Record training compute,
checkpoint/evaluation time, total invocation wall time, and process high-water RSS.
RSS includes interpreter/library/tape overhead and previous work in a reused
worker; concurrent timings are descriptive, not a memory/speed advantage claim.
No comparison of trajectory algorithms or scalable memory usage is made here.

```bash
uv run python -m experiments.state_preparation_exact_adam \
  --source-root /Users/maximilianfrohlich/Desktop/yaqs-experiment-results/q6-hybrid-construction-extension-20260906 \
  --output-root output/state-preparation-exact-adam-20260906 \
  --workers auto
```

Append `--resume` only for an existing compatible output directory. All previous
experiment defaults and artifacts remain unchanged.

Completed results: [Experiment 1 findings](state_preparation_exact_adam_findings.md).

Reference: [RobustState training protocol](https://arxiv.org/html/2311.16035v1).
