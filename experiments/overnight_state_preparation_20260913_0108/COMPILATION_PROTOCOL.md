# Fixed endpoint physical compilation screen

Preregistered before compilation of development endpoints or noisy comparisons.
This is a deterministic postprocessing test with no optimizer, rank selection,
target-dependent threshold, family switching or residual layer growth.

## Fixed compiler

Accept only scalar RX/RY/RZ/RXX/RYY/RZZ circuits. Resolve every gate's supplied
endpoint angle, including fixed offsets, scales and shared source parameters.
Reject data maps, unsupported gates and nonfinite or malformed inputs. Output
one independent trainable scalar per retained gate with unit scale/zero offset.
This is a fixed-endpoint reparameterization, not a claim of equality for other
values of the original shared parameters. Preserve the input objects unchanged.

Reduce each resolved angle modulo 2π, retaining the resulting global sign in
metadata. Remove a gate when its reduced angle has magnitude at most 1e-12;
record the discarded residual and a numerical error bound. Explicit local
trigonometric checks reject argument reductions with excessive roundoff.

Process gates in original order. For each retained incoming rotation, scan
backward only through gates on disjoint qubits with the same `noise_enabled`
value. Fuse with the first overlapping gate only when its generator, ordered
sites and noise flag are identical. Any other overlap or retained noise-flag
boundary stops the scan. Add angles, reduce again, and remove a fused identity.
Do not exploit other commutation identities, approximate small-angle pruning,
layout changes, arbitrary unitaries or optimizations conditioned on fidelity.

Each deleted or fused physical gate removes its own gate window. This changes
the noisy circuit intentionally while preserving its ideal unitary within the
declared numerical tolerance. No remaining noisy gate becomes noiseless.
All output operations remain supported scalar gates with compact parameters.
Record every source-to-output mapping, fusion, removed identity, global sign,
input/output physical noise-window count and complete compiler CPU/wall time.

## Engineering validation

Independently compare full Qiskit unitaries, not just one state, at q1/q2/q3:
all primitive generators, positive/negative periods, shared/scaled/offset and
fixed angles, reversed/nonadjacent sites, cancellations across disjoint gates,
overlap barriers and noise-flag barriers. Verify source immutability, compact
parameterization, complete mapping and deterministic repeated compilation.
Exact noisy q2 tests must show that removed/fused gates actually remove their
noise windows. Invalid data maps and unsupported gates must fail explicitly.

## First scientific panel, pending execution allocation

All four unchanged q6 development targets, replica 11, independently process:

- rank-two unchanged synthesis endpoint;
- rank-two exact Adam 600 endpoint;
- original authenticated 108-gate construction start;
- structural L-BFGS25 final full-budget endpoint.

Retain Haar in every panel and aggregate. No conditional TT-approximation rule
is implemented; a later threshold-based initializer remains a separate idea.

Authenticate the immutable benchmark and every input artifact/protocol. For
each source, freeze the same exact source circuit and parameter vector for
both uncompiled and compiled arms. Compile once, with no training afterward.
Independently check ideal statevector overlap between both complete physical
preparations from zero before scoring. Evaluate all twelve fixed profiles for
both final circuits. Compare each compiled arm only with its own source as the
paired compilation effect; cross-source differences retain their different
construction and optimization costs. Report full source and compiled gate
counts, every fixed condition, compilation CPU/wall time and separate audit
cost. Ballarin alone applies its unchanged materializer/pruning rules.

Preparation freezes sources and code without compilation or evaluation.
Scientific execution requires a separate explicit `--run`; this implementation
task does not authorize it. Preserve failures and unfavorable outcomes, and
abort on custody or configuration mismatches. Completed artifacts are read-only
on identical resume. No claim about hardware idle-time removal is implied by
the benchmark's abstract gate-window deletion semantics.
