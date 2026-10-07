# State-preparation eigenstate simplification diagnostic

This is a new development-only compiler diagnostic. No scientific compile or
channel evaluation is authorized until the coordinator freezes source/input hashes,
passes the complete lint and test batch, and allocates execution. No confirmation
target is touched. Existing compilers and optimization pipelines remain unchanged.

## Known mechanism and mathematical scope

Known-input and constant/eigenstate propagation are established compiler ideas,
including [Assertion-Based Optimization of Quantum Programs](https://arxiv.org/abs/1810.00375)
and [Quantum Constant Propagation](https://cliplab.org/jd1mOkw/SAS2023/142840162/142840162.pdf).
Input-specific equivalence also appears in
[Robust quantum compilation and circuit optimisation via energy minimisation](https://arxiv.org/abs/1811.03147).
The present implementation is a small dense numerical adaptation for this repository,
not a new optimizer, proof system or general unitary-equivalence compiler.

The only scientific input state is the physical all-zero vector. Supported systems
have one to six qubits, and only `rx`, `ry`, `rz`, `rxx`, `ryy`, `rzz` rotations.
Each gate must own one distinct scalar parameter. Permuted parameter indices,
finite signed/zero affine scales and offsets, reversed/nonadjacent sites and
logical/native/noise metadata are preserved. Shared, fixed-parameter, data-dependent
and arbitrary-unitary gates are rejected, not silently converted.

Start with the all-zero state and process gates in their original order. For each
gate, apply its noiseless matrix to the current retained ideal prefix. Let `before`
and `after` be those two vectors. Use the unit-modulus phase of
`vdot(before, after)` (phase 1 if the overlap vanishes), and compute the direct
Euclidean norm `norm(after - phase * before)`. Remove the gate iff that residual is
at most `1e-12`. A removed gate contributes its phase and residual to the audit;
the retained prefix stays unchanged. Otherwise retain the gate and advance the
prefix to `after`. There are no permutations, fusions, angle reduction, structure
search, threshold tuning, or noisy-score-based acceptance decisions.

The direct residual avoids the catastrophic cancellation of a formula based on
subtracting an overlap from one. The sum of removed residuals bounds the ideal
state error by the triangle inequality and unitary norm preservation. A separately
reported allowance `128 * machine_epsilon * (input_gates+1) * 2**qubits` covers
floating-point application, overlap and phase-accumulation error conservatively.
This is a numerical allowance, not an interval-arithmetic proof. An independent
Qiskit source and compiled preparation must agree with the recorded accumulated
global phase within that bound; otherwise the compiler fails. Report both the
best-phase and recorded-phase vector errors and normalization errors. Equivalence
is asserted only on the all-zero input, not on arbitrary input states or under
control of another qubit.

The output keeps the exact ordered subset of original gate objects' fields, apart
from explicit scalar-index compaction. Surviving scalar parameters retain their
original relative parameter order. Source-to-output gate and parameter maps,
original logical/native IDs, every decision residual/phase, retained noise flags,
removed windows and the accumulated phase remain in the artifact. Output vectors
are not substituted as free inputs to later gates or noisy evaluation.

## Complete paired panel

Freeze these six source labels on all four existing development targets at replica 11:

- the authenticated physically compiled legacy initializer;
- the authenticated physically compiled conditional initializer;
- the completed conditional-composition structural exact L-BFGS final endpoint;
- the same composition structural exact Adam final endpoint;
- the same composition structural weighted Adam final endpoint;
- the same composition structural adaptive-online final endpoint.

The target identities are `gaussian_mu0p5_sigma0p1`, `tfim_critical`, `haar_random_1`
and `random_mps_bond2`. This is exactly 24 source/compiled pairs. Preparation requires
all source cases complete; it does not shrink the panel if a case is absent.
Authenticate initializer/composition protocols, case signatures, all source files
and circuit-bank identities before extraction. Freeze complete small endpoint
records, source case hashes and numerical/benchmark hashes before compiling any
scientific endpoint. Source selection is fixed by method identity and final
endpoint, never by this diagnostic's ideal or noisy scores.

Run the single deterministic simplification above on every pair. Independently
evaluate source and compiled physical circuits under all twelve unchanged exact
calibrated conditions, including native Ballarin lowering. The ideal prefix used
for a deletion need not equal its noisy prefix, so no noisy-fidelity guarantee
follows from the proof. In particular a removed gate's noise window disappears;
it is not retained as a zero-angle noisy gate. Keep unfavorable noisy differences.

Report the equal-eleven-noisy-condition delta for every pair, each source label's
equal-target mean, each target's equal-source mean and the complete 24-pair mean.
Keep noiseless fidelity, every individual noisy condition, one-/two-qubit counts,
depth, scalar count, original and Ballarin-lowered counts, and compiler/audit CPU.
Do not replace an unfavorable result with an earlier checkpoint or a different
tolerance. Retire broad adoption if the complete-panel mean noisy fidelity falls
by more than `1e-10` in absolute fidelity (a numerical reporting margin). Any
negative source-label or target mean also blocks a claim of uniformly beneficial
compilation and must be stated explicitly. Neutral or positive noisy performance
with fewer gates is a compiler/resource result, not an optimizer-superiority result.

The prefix simulation, both independent Qiskit preparations, metadata work and
all 12 source/compiled scoring have explicit time and operation counts. Durable
attempt ledgers charge caught failures/replay and reject unobserved process-death
cost. Compiled and uncompiled scores are separate physical evaluations unless an
identical endpoint is reused through an authenticated cache; cache hits are stated.
No score is used to decide which compiled circuits are retained in the report.

Adoption would give the same compiler opportunity to every compatible optimizer.
Removing unobservable scalar coordinates may also alter optimization trajectories;
therefore a newly composed pipeline must be rerun and validated on development
before any final confirmation recipe is frozen. This endpoint diagnostic alone
does not establish that earlier training results transfer to a changed parameter
domain. No held-out tuning or scientific launch occurs in this implementation task.

## Frozen driver and custody

`state_preparation_compilation_experiment.py prepare --output OUTPUT
--composition-root COMPLETED_COMPOSITION` performs authenticated reads only.
It freezes all 24 complete endpoint records, target-vector hashes, all source child
artifact hashes, numerical sources, the environment and immutable benchmark files
in `PROTOCOL.json`. `run --output OUTPUT --workers N` accepts one to four workers;
the coordinator allocates the actual worker count only after lint and source freeze.
The driver does not call any constructor, initializer or training entry point.

Each pair has `COMPILATION.json`, compiler attempt records, separate `before` and
`after` all-channel score caches, an inclusive `CASE_ATTEMPTS.json` and a signed
`RESULT.json` covering all child artifacts. Separate score roots intentionally
perform all 576 physical channel evaluations on a fresh complete panel, even for
identical endpoints. Completed cache hits perform no numerical work. Caught
compiler failures retain measured CPU; their partial gate-application counts are
explicitly unknown, not reported as zero. Completed compiler application counts
and all attempted/completed channel calls remain explicit. Unclosed running
markers block resume because their elapsed physical work is unknown.

Inclusive per-pair CPU contains target loading, compilation, both Qiskit audits,
exact channel evaluation and inner bookkeeping. Source validation and final
result/attempt writes are outside this clock. Compiler and scoring timers are
nested diagnostics and must not be added again to the inclusive CPU. This is the
incremental endpoint-compilation diagnostic cost; original construction/training
costs remain in the authenticated input studies. No equal-CPU optimizer comparison
is inferred from this compiler-only panel.

`scoreboard.csv` preserves 576 original/compiled cells. `SUMMARY.json` retains
missing pair identities, checksum links, complete-panel means only when all 24
pairs finish, and all negative target/source-label means. Failures remain in
`FAILURES.json` and their per-pair artifact folders. Incomplete panels receive no
selected-subset aggregate. No compiler tolerance or endpoint choice is changed
after preparation.
