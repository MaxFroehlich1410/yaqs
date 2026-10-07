# Legacy construction reproduction, development only

This adapter is preregistered before any new q6 construction. It does not load
confirmation targets or authorize confirmation. A target vector is supplied
explicitly. The CLI accepts only the four existing development identifiers,
q6 and replicas 11/29. Small q1–q3 synthetic engineering cases use an explicitly
different mode. Preparation authenticates inputs without simulating them.

## Compatibility decision

The current overnight starts are `H25_lr0p7:cold_final` from
`/Users/maximilianfrohlich/Desktop/yaqs-experiment-results/q6-hybrid-construction-extension-20260906`.
Use the construction branch of `state_preparation_hybrid_pilot.run_job`, not
the newer `state_preparation_trajectory_comparison.construct` seed namespace.
The latter has the same schedule but hashes version
`yaqs.trajectory_competitiveness.v1`, root 2026091001 and role `construction`,
then shifts the result left 32 bits. It creates different initial parameters.

The compatible master vector uses `pilot.pilot_seed(PilotConfig(),
"initialization", target_id, replica)`: SHA-256 of canonical JSON containing
`yaqs.compression_first_pilot.v1`, root 2026090520 and those identity fields,
with the first 16 hexadecimal digits interpreted as an integer. Sample one
NumPy default_rng standard-normal vector at depth four. Initialize depth one
with 0.1 times its prefix; newly grown parameters use 0.001 times the matching
master slice. Growth retains old parameter indices, not gate-list positions.

Build the existing scalar BMPD preparation adjoint with the physical initial
U3 product layer. Depths 1–4 have 63/108/153/198 scalar gates at q6. Every
initial U3 factor and every BMPD factor is physical and counts. Original gate
IDs equal original parameter indices, even though adjoint gate order differs.
The first 18 parameter origins are protected throughout pruning.

At EACH depth, restart 25 gate-sequential stale-adjoint online updates at 0.7,
then run 75 simultaneous batch updates at 0.2. All rates are constant. The
online update uses costates computed at the old circuit and the updated
forward prefix. Use the unchanged public noiseless MPS hybrid optimizer and
ten-update chunks, including the 25-update switch inside its third chunk.
No optimizer moments are present. Chunk state is exact theta, completed update
counter, full scalar trace and phase; global counter restarts at each block.

The old top-level runner also reads the preceding batch pilot's construction
and refinement archives for provenance and optional warm-arm comparisons.
Those parameters do not initialize this cold construction branch. Reproduction
therefore needs the legacy master seed and target, not a free warm state or
the older batch endpoint as its circuit input.

Then perform six fixed magnitude-pruning rounds. Score the true signed affine
gate angle modulo 2π using the existing magnitude helper. Protect origins <18.
Select simultaneously ten one-qubit and five two-qubit gates per round, using
the existing `(round(score / 1e-12), magnitude, origin)` ordering within arity.
Remove actual gates/noise windows and compact parameters in old parameter
order while retaining origins. After each deletion run 20 batch updates at
0.1. Save the 520-update construction state; finish with 50 batch updates at
0.1. The final rule is update 570, never a best checkpoint. Final q6 circuit:
78 one-qubit +30 two-qubit gates, 108 independent scalar parameters, all noisy.

Training seeds preserve the old `pilot_seed(config, "training", target_id,
replica, block_name)` namespace (`source_1`–`source_4`, `M_repair_0`–`5`,
`refine_M_0`). They do not introduce randomness into noiseless updates.

## Numerical custody, costs and reproduction

Default backend is MPS, complex128. Freeze actual `KrotovTruncation()` fields:
max_bond_dim=None, svd_threshold=0, min_bond_dim=1,
trunc_mode="discarded_weight". Freeze all underlying YAQS numerical sources,
benchmark helpers, pilot/hybrid source, this adapter/protocol, dependency
versions and numerical-thread environment. No statevector substitution or
new comparison seed namespace is allowed. A future faster backend must be
separately named and checked on paired development cases before use.

The archived extension source fingerprint is
`90e41762c64ddb6c2c02de065b0793a6a99ec95a33d58250abfa997bb8eb081b`.
Current code has a different fingerprint, and newer block signatures include
backend metadata. The old strict top-level runner rejects current source
compatibility; its archives must not be resumed or rewritten. This adapter
starts in a new root, reproduces the old mathematical recipe, and compares
full circuit/parameter endpoints against authenticated archived starts. Until
an allocated q6 development reproduction succeeds (circuit equality and
max absolute parameter error <=1e-11), exact archive compatibility remains
unproven. Preserve mismatches as results; do not tune around them.

Historical artifacts record wall time, not process CPU. Historical CPU stays
null. New construction CPU/wall time includes all attempted construction work,
pruning, checkpoint I/O, replay of completed stages and final diagnostics.
The final artifact serialization and initial manifest custody work are outside
the timed interval. Interruption attempts have their own ledger; a hard-killed
attempt with unknown duration makes physical total cost unknown, never zero.
Complete blocks and final artifacts are authenticated on resume. Store one
full circuit/parameter endpoint and compact block digests, without duplicating
circuits in every trace row. Every downstream method must share and charge
the same complete construction source and cost.

## MPS scaling audit, no scaling run

`MPS.from_statevector` exactly normalizes and factorizes a dense vector in
little-endian order; its input already costs exponential storage. Passing an
MPS directly to `state_preparation_metrics` avoids dense target conversion.
`MPS.scalar_product(other)` contracts the full network; do not pass its `sites`
argument for a full-state fidelity. `MPS.to_vec()` materializes a dense vector
and must remain an audit-only small-system operation. The state-preparation
metric uses the complete physical circuit from `MPS(..., state="zeros")` and
the squared absolute overlap with the target MPS.

Read actual internal bonds from tensor left/right axes; `get_max_bond()` also
includes the physical axis and therefore reports at least two for qubits,
even for a product MPS. This adapter records target bonds, every final-forward
gate's actual maximum internal bond, final norm/overlap and truncation controls.
Those are final-forward diagnostics, not a measured training-wide costate peak.
The exact q6 Hilbert-space bond bound is eight. No bond cap or truncation is
silently introduced to achieve that size.

For later scaling, freeze target representation/construction cost, forward
and costate peak bonds, cap-hit counts, all SVD thresholds/modes, per-split and
accumulated discarded squared singular values, norm drift and dense-versus-MPS
overlap error on small audit cases. Current `apply_local_operator`/split APIs
do not return discarded-weight diagnostics; zero threshold/no cap is exact,
but truncated scaling needs explicit instrumentation rather than inferred
zero truncation error. Also charge target resolution, copies, contractions
and any SWAP-chain work for nonadjacent gates. No MPS scaling experiment is
part of this implementation.
