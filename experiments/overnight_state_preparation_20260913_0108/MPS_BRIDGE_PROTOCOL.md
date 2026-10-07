# Weighted MPS replay overlap gate

This support component does not authorize larger-system experiments. The q6
candidate confirmation gate still applies. No confirmation target is used for
this backend audit and no MPS scalability claim follows from implementation.

`mps_bridge.gradient` replays the exact same fixed unitary local noise maps and
nonnegative normalized weights as `StatevectorBackend`. It calls the unchanged
public MPS single-trajectory state-preparation adjoint for each path, then sums
the complete gradients and fidelities with those weights. It always physically
starts from the zero MPS. An MPS target is accepted directly; a dense target is
factorized inside the measured interval and that exponential input cost is not
hidden. This is a backend adapter for known trajectory adjoints, not a new
estimator or optimizer. Noises with state-dependent normalization are rejected.

A passive, reversible wrapper records every operator application and each
actual `linalg.truncate` decision. It reads the singular values already computed
by the numerical implementation, measures the squared discarded tail, and does
not recompute or change the SVD. Record per-split tails, their sum and maximum,
hard-cap reductions, actual internal bond axes, maximum individual tensor-state
storage, observed forward bonds, total forward/adjoint/derivative/target bond
peak, final forward norm error, path/direction counts and CPU/wall time. The
sum of local discarded weights is a diagnostic, not a global fidelity bound.
Maximum individual MPS bytes is not the memory required by complete tapes.

The exact overlap mode freezes `KrotovTruncation()` with no cap, threshold zero,
minimum bond one and discarded-weight mode. Truncated modes are engineering
probes only: the adjoint does not differentiate discrete SVD truncation choices.
They must never be labelled an exact gradient of the truncated simulation.
No private production numerical routine is modified. Instrumentation is
process-local and must run serially within a worker; all patches restore on
normal return and exceptions. A scientific driver must retain attempted work
and failure costs around the kernel, since a raised exception returns no result.

Preflight tests cover q2/q3 complex targets, unequal mixture weights, shared and
signed affine parameters, reversed/distant two-qubit gates, independent dense
finite differences, exact no-discard behavior, true bond axes, cap-forced loss
of weight/norm, invalid maps/weights, and wrapper restoration after failures.

Before any q8 run, the allocated overlap driver will use all four development
targets and both their original108 and mean-two structural-L-BFGS endpoints at
seed11. Freeze these endpoint checksums before execution. Use four paths per
channel (eight total) from an independent `mps_overlap` seed namespace. Compare
per-path fidelity, weighted loss/gradient and three fresh-stream exact-Adam
updates (rate0.02, beta1 0.9, beta2 0.999, epsilon1e-8) between dense and MPS.
Acceptance requires every gradient/parameter/loss discrepancy <=1e-10, all
complete cases, exact zero discarded tail, and measured bonds within the q6
Hilbert-space bound. Report true cost and input factorization separately where
possible. This gate is implementation agreement, not optimization evidence.
A q8/q12 experiment still requires confirmation success or the user's declared
exceptionally clear resource-advantage route, plus a new bounded scaling protocol.
