# Circuit synthesis and architecture scout

Prepared 2026-09-13. All ten normative files were read completely. The target
fixture was parsed through `load_target_collection`, including all 74,880 real
amplitude components across 18 q6/q12 records. Its SHA-256 is
`d53cd272b6c665b3a92de2d6b1e335b85ef77f74016b0f20cf7ea6a0f486b839`.
The inspected worktree contains extensive user modifications; this track changes
only its new experiment module, tests, and this report.

## Strongest reproducible source endpoints

The strongest archived structural comparison is
`output/statevector-adjoint-restructuring-20260912/`. The manifest canonical hash
is `baa8d1f5c27cf9bf887cff10c2b5b92bbc0c40596a316ca5e00115d8d777c150`;
the aggregate result hash is
`44713751ab99c23c125c64ded70d6988639647adbe1369d742df0bcb0e232908`.
`experiments/state_preparation_pilot.read_record` authenticates the canonical
checksum envelope. Do not read `refined[method]['state']` as the primary endpoint:
the final crossing block can exceed the case budget. Select with
`state_preparation_noisy_restructuring.at_budget(arm, record['budget_seconds'])`.

Each private record is at
`optimization/{target_id}/seed_{11_or_29}.json`. Its schema includes `prefixes`,
`searches`, `refined`, and `fixed`, each keyed by `adam` or `adaptive_online`.
`at_budget(...)['state']` contains `circuit`, `theta`, `first`, `second`, `counts`,
and `updates`. `pilot.restore_circuit` restores complete gate metadata; retain
the complete state if resuming a historical optimizer. Historical statistics use
exact calibrated local depolarization only, not the complete benchmark channels.

| Target / start     | Structural Adam | Structural Krotov |   Fixed Adam | Fixed Krotov |
| ------------------ | --------------: | ----------------: | -----------: | -----------: |
| Gaussian / 11      |    0.8447508791 |      0.8442907603 | 0.8267506821 | 0.8269787787 |
| Gaussian / 29      |    0.8467999391 |      0.8438306792 | 0.8292480784 | 0.8257450536 |
| Critical TFIM / 11 |    0.8457894214 |      0.8444454857 | 0.8280941495 | 0.8268697364 |
| Critical TFIM / 29 |    0.8457904506 |      0.8453634411 | 0.8266175353 | 0.8262245381 |
| Haar 1 / 11        |    0.7164413376 |      0.7171952760 | 0.7371906547 | 0.7471898415 |
| Haar 1 / 29        |    0.6709292924 |      0.6879771159 | 0.6789460046 | 0.6765357857 |
| Bond-2 MPS / 11    |    0.8481512884 |      0.8470822804 | 0.8290498485 | 0.8282423128 |
| Bond-2 MPS / 29    |    0.8484899004 |      0.8462161322 | 0.8281630688 | 0.8285069124 |

Structural Adam has 103 retained gates: 25 two-qubit gates on structured states,
26/28 on the two Haar starts. Structural Krotov has 78 one-qubit and 25 two-qubit
gates everywhere. Fixed controls have 78/30. The Krotov mean advantage is driven
entirely by Haar; an oracle per-case mixture is not a reproducible single method.

Earlier common 108-gate starts can be loaded with
`experiments.state_preparation_exact_adam.load_starts(source_root)`; the existing
structural manifest records the appropriate construction source root. This is
preferable to guessing the earliest compatible output directory.

## Ranked independent hypotheses

### 1. Exact noisy coordinate minimization

**Provenance:** Ostaszewski, Grant, Benedetti, _Structure optimization for
parameterized quantum circuits_, [arXiv:1905.09692](https://arxiv.org/abs/1905.09692).
Original field: variational quantum algorithms; known Rotosolve/Rotoselect.

**Mechanism and adaptation:** For a unique scalar Pauli rotation and channels
independent of its angle, target fidelity is `a + b cos(theta) + c sin(theta)`.
An exact coordinate optimum comes from three values and `atan2(c,b)`. Reuse a
reverse observable tape and an updated density prefix so each coordinate can
be solved using local contractions, rather than whole-circuit evaluations.
This adaptation changes the objective to noisy target fidelity, not the known
coordinate method. It may eliminate inefficient small sequential updates.

**Smallest test:** Four development starts, 20 sweeps versus exact Adam/L-BFGS,
paired starting circuits and measured CPU; compare finite-step monotonicity and
fidelity per second. **Cost:** about 150–250 lines plus analytic/noisy tests.
**Failure:** narrow-coordinate valleys; cached tape misuse for shared parameters;
angle-dependent Ballarin channels invalidate the sinusoid. A bounded scalar
search is then needed. **Status:** selected for another track's implementation.

### 2. Adaptive stopping for noisy deletion and repair

**Provenance:** Bilkis et al., VAns, _A semi-agnostic ansatz with variable
structure for quantum machine learning_,
[arXiv:2103.06712](https://arxiv.org/abs/2103.06712).
Original field: variational quantum algorithms; variable circuit structures
with insertion and simplification are known.

**Mechanism and adaptation:** Replace the historical fixed five deletions with
bounded sequential deletion until no repaired deletion beats an equally repaired
retained circuit. Use the existing exact finite-deletion score; this test is
deletion-only, not a full VAns implementation. Exact noisy acceptance and a
matched retained repair prevent pruning from getting credit for ordinary
optimization. **Smallest test:** all four development starts with one fixed
shortlist and repair budget, comparing the same optimizer without deletions.
**Cost:** small; implemented in `structural.py`. **Failure:** greediness,
inadequate repair misleading the ranking, and inability to undo an early
deletion. Prefer retiring it to retuning repeatedly if Haar regresses.

### 3. Beam search and reversible gate relocation

**Provenance:** Davis et al., _Towards Optimal Topology Aware Quantum Circuit
Synthesis_, [original QSearch implementation](https://github.com/BQSKit/qsearch)
and [algorithm documentation](https://qsearch.readthedocs.io/en/latest/autoapi/qsearch/compiler/index.html).
Original field: heuristic search with numerical synthesis. QSearch uses A\*;
a width-two deletion beam would be an adaptation, not a faithful reproduction.

**Mechanism:** Keep two distinct structures after each repair instead of only
one, optionally restore a previously removed gate at a new position. Rank by
true noisy fidelity with full branch costs. **Smallest test:** Haar 1 and one
structured development state, width one versus two under equal compute.
**Cost:** medium; copying optimizer state and exact branch accounting dominate.
**Failure:** tiny repaired differences choose the wrong basin, or the beam cost
exceeds value of longer single-path refinement. Identity insertion is not
noise-neutral: inserted zero-angle gates still add fixed noise windows here.

### 4. Target-informed sequential MPS initialization

**Provenance:** Schön et al., _Sequential generation of entangled multiqubit
states_, [arXiv:quant-ph/0501096](https://arxiv.org/abs/quant-ph/0501096);
Ran, _Encoding of Matrix Product States into Quantum Circuits of One- and
Two-Qubit Gates_, [arXiv:1908.07958](https://arxiv.org/abs/1908.07958).
Original field: tensor-network state synthesis; established methods.

**Mechanism and adaptation:** Truncate the target to a rank-two MPS, derive its
sequential isometries, complete each to a two-qubit unitary, lower it, then refine
the actual noisy circuit. Repeat low-rank residual disentangling only if needed.
The bond-2 target provides an exact ideal check; TFIM/Gaussian provide a meaningful
compression test; Haar exposes expressivity loss. No ancilla or alternative
initial state can be supplied free of charge. **Smallest test:** one rank-two
construction per development target and exact noisy evaluation before refinement.
**Cost:** medium; repository already contains
`create_sequential_matrix_product_disentangler_parameterized_circuit`, which
KAK-lowers supplied one-/two-qubit matrices to scalar rotations and CX gates.
**Failure:** generic KAK completion wastes gates; truncation misses Haar;
overcounting an arbitrary unitary as one gate would invalidate the result.

### 5. Proximal entangler sparsity and continuation

**Provenance:** Parikh and Boyd, _Proximal Algorithms_,
[original article and code](https://web.stanford.edu/~boyd/papers/prox_algs.html).
Original field: convex/nonsmooth optimization; transfer here is nonconvex and
does not inherit convex global convergence guarantees.

**Mechanism and adaptation:** Add `lambda * sum(abs(wrap(theta_2q)))` or a
group norm over each entangling block, follow a short increasing-lambda path,
delete exact zero gates and refine the unpenalized true noisy objective.
**Smallest test:** two predeclared penalty settings, equal full-gradient counts,
versus hard exact deletion on the four development starts. **Cost:** low–medium.
**Failure:** angle size is a poor gate-importance proxy; a zero-angle gate still
has a full fixed noise window until physically deleted. Periodic parameters and
coordinate rescalings complicate naive Euclidean shrinkage. This looks less
informative than exact deletion while the exact q6 scorer is so cheap.

### 6. Differentiable gate-presence search

**Provenance:** Zhang et al., _Differentiable Quantum Architecture Search_,
[arXiv:2010.08561](https://arxiv.org/abs/2010.08561).
Original field: differentiable/probabilistic programming applied to circuits.

**Mechanism and adaptation:** Optimize probabilities over identity and a small
gate dictionary, jointly with angles, then freeze one discrete architecture and
refine it. **Smallest test:** probabilities only for existing two-qubit-gate
presence; compare with exact deletion at identical total work.
**Cost:** medium/high; stochastic architecture draws and final discretization
need their own seed domain. **Failure:** a relaxed mixture channel looks better
than a realizable circuit, or shared weights favor the wrong discrete endpoint.
Only the frozen, lowered physical circuit is a valid final competitor.

### 7. Riemannian block optimization

**Provenance:** Luchnikov et al., _QGOpt: Riemannian optimization for quantum
technologies_, [arXiv:2011.01894](https://arxiv.org/abs/2011.01894);
Le, Sun, Mendl, _Riemannian quantum circuit optimization based on matrix product
operators_, [arXiv:2501.08872](https://arxiv.org/abs/2501.08872) and
[original code](https://github.com/INMLe/rqcopt-mpo).
Original field: constrained numerical optimization and quantum simulation.

**Mechanism and adaptation:** Optimize each two-qubit block on U(4), projecting
Euclidean gradients onto the tangent space and retracting to unitarity. It may
avoid bad scalar parameter conditioning. **Smallest test:** one noiseless block
update versus scalar L-BFGS, then lower and score both circuits under full noise.
**Cost:** high for a scientifically valid noisy comparison: each changing U(4)
must be lowered and noise must act at the lowered gates. **Failure:** treating
an optimized block as a single noise window gives an artificial win; KAK charts
and native gate-count discontinuities spoil a naive smooth noisy objective.
Retire for this first wave unless simpler methods reveal conditioning as the
dominant bottleneck.

## Implemented adaptive-deletion contract

`adaptive_delete_refine(circuit, theta, target, calibration, refine, *, config,
protected_origins)` returns a JSON-compatible endpoint and full search trace.
`refine(circuit_copy, theta_copy, evaluation_cap)` must return
`RefinementResult(theta, evaluations, iterations, diagnostics)`. Each round
reserves equal caps for one unchanged branch and every shortlisted deletion.
Actual evaluations and reserved caps are both saved; discarded and failed
repairs remain in the ledger. Numerical failures are charged the full cap and
cannot promote a fallback deletion. A callback overrun invalidates the run.
The original endpoint remains eligible if refinement regresses; numerical ties
retain it. Set `protected_origins=frozenset(range(18))` to preserve the historical
q6 product layer. The optimizer adapter must define its own restart or moment
policy consistently for every branch; this stateless callback API does not
claim historical Adam-moment continuation.

All rankings and branches are serializable, with circuit and parameter endpoints.
Initial-state preparation is always the actual retained circuit acting on
`|0...0>`. Deletion removes both ideal gate and its channel; remaining windows
are unchanged. The first implementation does not insert gates, route operations,
or purport to implement all of VAns.

Validation: 23 targeted tests passed in 1.05 seconds, including analytic one-qubit
repair, a noisy Bell-state circuit with a removable zero-angle entangler,
protected preparation, independent exact evaluation, equal repair caps,
retained-branch improvement, deterministic ties, failed-branch accounting,
budget exhaustion, input immutability and strict JSON serialization. Changed
files pass Ruff and project-environment `ty`; the coordinator owns full lint.
