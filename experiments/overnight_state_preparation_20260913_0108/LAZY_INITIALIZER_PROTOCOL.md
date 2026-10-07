# Lazy conditional initialization: development equivalence gate

This changes orchestration, not synthesis, the 0.99 threshold, scalar lowering,
the generic unitary compiler, or the legacy constructor. No held-out vector is
loaded or simulated during implementation. Before confirmation use, all four
q6 development targets at replica 11 must reproduce the existing conditional
initializer circuit and parameters exactly, with the same overlap decision.
Every case is retained. Scientific execution requires coordinator allocation.

The prior two-source initializer kernel always receives and compiles a legacy
endpoint. That was appropriate for its legacy-versus-conditional development
comparison. A standalone conditional pipeline cannot claim to skip construction
if it eagerly executes that constructor simply to satisfy the helper API.
This adapter makes the claimed early exit real.

## Fixed numerical behavior

First call the frozen `synthesis.synthesize(target)`. Recompute the ideal TT
overlap from its approximation and independently audit the complete physical
preparation from zero with the same Qiskit state helper used by
`initializer_policy.build_initializers`. Apply the frozen
`initializer_policy.use_rank_two` rule: choose rank two iff overlap >=0.99,
with no target identity, family label, noisy score or relaxed tolerance.
Always compile the rank-two physical probe using the exact existing
`initializer_policy.compile_source`, including its independent full-state audit.
Always pay for this probe, including on fallback.

If the overlap meets the threshold, return that compiled circuit immediately.
Do not invoke the legacy constructor, compile legacy gates, copy an archived
legacy endpoint or create a legacy child directory. Otherwise invoke the
unchanged constructor inside the adapter-owned `legacy/` directory, authenticate
its complete result, full block states and attempt ledger, then compile that
endpoint with the same frozen generic compiler. The scientific fallback recipe
is exactly the previously reproduced H25/rate-0.7 MPS construction:570 updates,
78 one-qubit gates,30 two-qubit gates and108 parameters. See
[CONSTRUCTION_PROTOCOL.md](CONSTRUCTION_PROTOCOL.md).

The selected physical circuit, affine parameters, gate order, noise flags and
logical/native IDs must equal the prior policy's selected output. The only
numerical work omitted is the unused legacy constructor/compiler on a rank-two
selection. There is no new state-aware compiler, structural search, optimizer
or noise scoring in this adapter. Any later compiler opportunity remains a
separate validated composition choice.

## Ownership, custody and complete costs

The target vector is supplied explicitly. Development mode accepts exactly
the four fixed q6 development identifiers at seed11; engineering accepts only
synthetic q1-q4 vectors. A future confirmation invocation requires the complete
checksummed parent schema from `confirmation_construction`, and its initializer
section must equal `lazy_initializer.specification()`. That whole parent binds
all arm choices, budgets, acceptance, source hashes and per-case vector hashes.
No standalone confirmation flag or target loader is provided.

Prepare one fresh root containing only `PROTOCOL.json`. Freeze the target hash,
identity, constructor configuration, compiler/synthesis sources, benchmark
custody and environment. A callback receives only the owned `legacy/` path; its
returned endpoint must match the complete checked RESULT, PROTOCOL, ATTEMPTS
and block receipts there. Source target ID, replica, vector hash and recipe must
match the initializer case. An external cached constructor cannot be substituted
as an uncharged free callback.

Persist the complete `PROBE.json` before any fallback. Persist legacy output in
its original child schema, and legacy compilation once. Keep only references
and compact selected source metadata in the final result; do not duplicate
constructor histories. Resume only those exact stage states. Full source,
parent, case and durable dependency hashes are checked before any cache return.
Reconstruct every cached initializer RESULT field from its authenticated probe,
selected compiler/constructor receipts and complete attempt totals; merely
preserving its signature does not authenticate an edited angle or cost. The
development wrapper additionally stores a compact completion receipt binding
its entire audit/result record, and requires the exact complete child-file map,
including detection of unexpected added records.

One inclusive outer attempt ledger defines actual standalone initializer CPU
and wall time. It includes target copies, constructor invocation, all numerical
probe/compiler/audit work, metadata/circuit copies, checks and ordinary stage
I/O. Caught failures and repeated unfinished work stay charged. The nested
constructor's own complete CPU is separately reported for diagnosis; it is
already inside the outer timer and is **not added again**. Do not reuse the
older policy's component-sum initializer cost as the new complete cost.
Prepare-only administration, read-only completed cache authentication and final
result/terminal-ledger serialization are outside this interval. The caller
separately charges loading its supplied target. No overlapping component times
are added to the inclusive total.

Running/null-duration attempts block resume until external accounting is
resolved. Orphan stages without an attempt ledger, changed child checkpoints,
unknown constructor timing and changed source/parent data also fail closed.
Every caught numerical failure retains its message, known CPU/wall and complete
stage dependency snapshot. No fallback is triggered by a numerical failure;
only the frozen overlap rule controls the constructor branch.

## Bounded development audit

The development wrapper must prepare all four cases before executing any one.
It authenticates existing initializer-policy PROTOCOL/SUMMARY/case RESULT and
INITIALIZERS snapshots and dependencies. Compare the selected circuit and theta
exactly to that prior conditional endpoint, compare TT/lowering overlaps, and
independently compare complete ideal physical preparation vectors. No new noisy
scores or optimizer tuning are needed for this orchestration equivalence gate.
Confirm that the constructor was called exactly on below-threshold cases and
that no legacy directory exists on rank-two-selected cases. All source checks,
fresh construction and audit costs remain visible. Retain failures and explicit
mismatches rather than rescuing them with a changed threshold or seed.

Only after this entire panel passes and the coordinator completes lint may the
new lazy orchestration enter a final complete confirmation parent. Its absolute
cost replaces the earlier constructed-two-source bookkeeping; earlier scientific
files and their reported costs remain untouched.
