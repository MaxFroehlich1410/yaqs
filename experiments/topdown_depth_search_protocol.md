# Frozen q6 depth-search protocol

This changes construction stopping and candidate selection, retaining the tested
[B/D pruning and weighted fine-tuning](topdown_state_preparation_method.md).
All nine six-qubit fixture targets, optimizer seeds 300/400, pruning B/D and
the existing IBM/Google logical noise profiles give 72 searches. These names
denote historical simulation coefficients, not calibrated hardware predictions.

1. Grow the original BMPD circuit for depth indices 1 through 16. One index adds
   a full even/odd brickwall cycle (45 scalar gates at six qubits), not one
   physical two-qubit layer. Keep the initial 18 single-qubit rotations.
   Preserve parameter order and RandomState initialization: seed+1000+depth,
   sigma 0.05 initially and 0.001 for extensions.
2. At each depth perform the unchanged 100 online plus 100 batch noiseless
   updates at rate 0.7, with no decay, retaining the final iterate. Remove the
   90% early exit. Continue growth from this **unpruned noiseless checkpoint**;
   pruning and fine-tuning act on separate copies. No optimizer stabilization,
   best-iterate rollback, extra restarts or repair-policy changes are introduced.
3. Apply B or D with its frozen 0.01 exact-deletion threshold. B deletes all
   individually eligible gates simultaneously. D deletes sequentially with
   5 online + 5 batch noiseless repair updates at rate 0.7 after each deletion.
4. Fine-tune every visited candidate with 100 online + 100 batch cross-trajectory
   updates at rate 0.2. Sample 100 unconditional histories using seed+5000,
   freeze them, merge only identical no-jump histories with empirical weight
   m0/100 and retain each jump history with weight 1/100. Preserve the existing
   rule: a nonempty circuit whose bank contains zero jumps fails that candidate;
   no forced events, alternate seeds or resampling. Empty circuits are recorded.
5. Independently sample 32,768 validation histories for that candidate using
   NumPy SeedSequence entropy [20260914,6,target_index,optimizer_seed,
   variant_index,hardware_index,depth,90000]. Training and validation use separate
   RNG streams. Draw the complete gate-level categorical bank before simulation,
   with exactly the same at-most-one-Pauli-jump law. Compress no-jump duplicates
   only; compute sample mean and standard error with original multiplicities.
   Persist the complete raw bank and its checksum. A nonempty zero-jump validation
   bank also fails without resampling.
6. Track the last **material record** and the highest validation score separately.
   A material record resets patience when

   $$
   \widehat F_d>\widehat F_{\rm record}+0.001+
   2\sqrt{\mathrm{SE}_d^2+\mathrm{SE}_{\rm record}^2}.
   $$

   Otherwise increment patience, including failed candidates. The first valid
   candidate establishes the record. Stop that B/D/profile branch after three
   consecutive unsuccessful extensions, or at depth 16. The uncertainty allowance
   is a practical heuristic, not a sequential hypothesis test or a guarantee of
   globally optimal depth. The material record is not updated by tiny gains, so
   accumulated small improvements can eventually reset patience.

7. Select the highest validation fidelity among **all successful visited
   candidates**, even when its gain did not reset patience; exact ties prefer
   earlier depth. Lock this decision to disk before density evaluation. All-failed
   searches have no selected candidate and are reported explicitly.
8. After selection is frozen, independently evaluate the selected circuit and
   all visited successful candidates with Qiskit 64x64 density matrices. Primary
   fidelity exactly integrates the training sampler's law; the full Lindblad
   channel is a sensitivity diagnostic. Retrospective exact ranking/Pareto
   frontiers never change the validation-selected winner. Compile with the same
   fixed CX reference as before and report logical two-qubit count, compiled CX
   count and compiled two-qubit depth. Noise scoring acts on the logical circuit.

The original first-90% construction plus B/D is rerun for every target/seed.
Its final parameters and training banks must reproduce the archived comparison.
This supplies a paired control and fresh construction-inclusive CPU timings.
Per-search cost charges all visited construction, pruning/repair, fine-tuning,
sampling/compression, and validation work, including discarded/failed candidates.
Shared work is charged in full to each standalone search; actual parallel suite
wall time is reported separately. Exact assessment and compilation costs are
reported separately from the decision process. This is not a compute-matched
optimizer comparison; two optimizer seeds do not support broad significance claims.

Run with `uv run python -m experiments.topdown_depth_search --output NEW_DIRECTORY --workers 4`.
The manifest freezes this protocol, source hashes, fixture, prior inputs and
package versions before results exist. Resume requires them to match.
