# Preregistered rank-two sequential preparation test

Recorded before any development fidelity comparison or q6 construction in this
track. This is a target-informed synthesis candidate, distinct from circuit
refinement and from a new optimizer. No confirmation target may be screened.

## Fixed mathematical procedure

1. Read the unchanged normalized target vector in little-endian ordering.
2. Form its tensor-train approximation with maximum bond dimension two using
   one deterministic left-to-right SVD sweep in physical order q0,...,q(n-1).
   Keep both largest singular directions and normalize only the final vector.
   Save each discarded squared singular mass and final approximation fidelity.
3. Sequentially disentangle q0,...,q(n-2): take the leading two left singular
   vectors of the current pair-versus-rest unfolding as its occupied support,
   complete with the remaining SVD columns, fix every column's largest component
   to positive real, and apply the adjoint basis matrix. This maps the occupied
   pair support to |0> on the earlier qubit. Finally disentangle the last qubit
   by the corresponding one-qubit basis matrix.
4. Reverse and adjoint these local matrices to prepare the approximate target
   from the physical all-zero state. The final single-qubit preparation is an
   explicit first gate; no nonzero initial state or free ancilla is supplied.
5. Lower each two-qubit matrix exactly through Qiskit's Weyl decomposition with
   `fidelity=None`, Euler basis ZYZ, `simplify=False`, and tolerance zero. This
   directly emits scalar RZ/RY and RXX/RYY/RZZ rotations. No arbitrary U(4) or CX
   remains at evaluation. Lower each one-qubit matrix with the same ZYZ Euler
   convention. Every emitted rotation has its own parameter and enabled noise.
   Zero-angle rotations are retained and counted with full N2/N3 gate-window
   noise. There is no synthesis-specific noisy angle-pruning threshold.

The logical count is one one-qubit matrix and n-1 two-qubit matrices; all
scientific noisy comparisons use the separately reported lowered primitive
counts: 12 one-qubit and 3 two-qubit rotations per unsimplified U(4), plus
three rotations for the final one-qubit matrix. The q6 candidate therefore
contains 63 one-qubit and 15 two-qubit rotations.
All N2/N3 single-qubit lowering rotations receive their ordinary gate-window
noise. Ballarin's existing materializer alone supplies its specified noiseless
basis changes and 1e-4 RZZ pruning. No calibration is changed.

## Bounded engineering validation before screening

- q1/q2/q3 random vectors: reconstruction up to global phase, unchanged inputs,
  correct little-endian order, exact rank-two representation at q<=3.
- Random U(4) matrices and reversed/nonadjacent sites: primitive versus original
  unitary on all basis inputs, not only on one favorable preparation input.
- A genuinely rank-four q4 state: nonzero truncation, reported discarded weight,
  normalized approximation, and rank<=2 on every cut.
- q2 exact noise evaluator: lowering is physical; disabled compilation noise
  would yield a different channel and is not used.
- One q6 bond-two fixture engineering check: ideal fidelity exactly one within
  1e-10 and all primitive operations supported. This checks the mathematical
  class guarantee, not comparative noisy performance.
- Reject invalid dimensions, nonfinite/unnormalized states, and unsupported
  gates. Test deterministic repeated construction in the same environment.

No q6 development noisy score or optimization will be run until the coordinator
allocates the comparison. The first proposed screen is one construction for
each of the four development targets, final evaluation on all unchanged
benchmark channels, and a matched postconstruction refinement budget from a
single frozen optimizer. Charge construction time and all lowered gates to the
candidate. Falsify if approximation losses or compilation overhead prevent a
fidelity/gate-count/compute Pareto advantage. No family-specific completion,
ordering, rank, or compiler tuning is allowed after seeing that panel.

## Provenance and limits

Sequential MPS preparation is known; this implements a dense q<=6 diagnostic
of the rank-two construction, inspired by
[Schön et al., 2005](https://arxiv.org/abs/quant-ph/0501096) and
[Ran, 2020](https://arxiv.org/abs/1908.07958).
Local primitive lowering uses the established Cartan/Weyl factorization in
[Qiskit's official synthesis API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/synthesis).
The new component is its controlled application to the fixed noisy benchmark,
not an original state-synthesis theorem. Dense vector/SVD processing here does
not establish tensor-network scaling. Arbitrary isometry completion may waste
gates; changing completion is a separate preregistered method.

Engineering protocol clarification before q6 construction: the coordinator
required all zero-angle emitted primitives to remain physical. The compiler
setting above therefore uses `simplify=False, atol=0` rather than the initial
proposed identity simplification. This clarification occurred before any
development fidelity comparison.

Engineering validation completed: 25 targeted tests pass, including the q6
bond-two class guarantee and full local-unitary checks on reversed and
nonadjacent sites. The unsimplified Qiskit Weyl circuit can carry an additional
global sign relative to its declared phase. Each local lowering is therefore
audited against its complete input matrix, with the common global phase derived
from their Hilbert-Schmidt overlap. Only that physically irrelevant phase is
omitted; a maximum entrywise discrepancy above 1e-10 rejects the lowering. The
phase correction and matrix discrepancy are serialized per logical gate.
No q6 noisy development comparison was performed during this validation.
