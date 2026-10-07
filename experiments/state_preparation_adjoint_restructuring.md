# Shared-adjoint deletion scoring and budget-reinvested refinement

Version: `yaqs.shared_adjoint_restructuring.v1`. Frozen before verification/run.

## Objective

Replace repeated whole-circuit deletion scoring by mathematically equivalent
shared forward/backward density adjoints. Reproduce all historical scores and
the same five-round search, then spend the saved computation on further noisy
optimization of the chosen circuit. Keep targets, starts, optimizers, sampling,
noise model and structural-search settings unchanged.

This is an exact-density-assisted small-system diagnostic, not a new trajectory
estimator for structural decisions, native compilation or hardware experiment.

## Source and fixed settings

Historical evidence: `output/statevector-noisy-restructuring-20260912`.
Authenticate its manifest, result and all eight case records, plus the original
construction provenance and target vectors. Do not modify or resume that old
root with the new runtime. Earlier artifacts remain historical evidence.

Use precisely the old four q6 development targets (Gaussian, critical TFIM,
Haar random 1, bond-2 random MPS) and both construction starts, 11 and 29.
The following are copied from the historical configuration without retuning:

- Weighted R32, fresh ensemble per update, root seed 2026091207; original
  per-case seed namespace is retained, not changed to the new driver's version.
- Adam at 0.02, beta1=0.9, beta2=0.999, epsilon=1e-8.
- Adaptive online Krotov: 25 unscaled updates at 0.7 followed by RMS-scaled
  online updates at 0.02, beta2=0.999, epsilon=1e-8.
- 50 warmup updates; five deletion rounds; two shortlisted single deletions;
  ten repair updates per candidate; unchanged repaired and unrefined controls;
  protected original 18 product-layer gates. Identical 1e-12 ranking rounding
  and 1e-10 acceptance threshold. Preserve moments and counters after deletion.
- Homogeneous abstract-gate Pauli-Lindblad errors 0.00064 / 0.0051. Deleting a
  gate removes only its unitary and associated noise window, changing no others.
- Complex128 statevector trajectory updates and independent Qiskit exact
  density endpoint scoring. No new rates, targets, noise models or gate insertions.

## Shared exact scorer

For the original circuit channels A_k = E_k composed with U_k, store each
rho_before(k) from one forward density propagation. Propagate the target
projector backward through the unchanged suffix to obtain B_after(k).

`F_delete(k) = Tr(B_after(k) * rho_before(k))`.

Score this contraction **before** applying the current gate/channel adjoint to
the costate. This removes both the gate and its noise window. The result is
the exact finite-deletion fidelity, not a first-order parameter-gradient proxy.
One forward/backward pair scores every gate; only then apply the existing
eligibility filter and ranking rule. Fixed gates, shared parameters, disabled
noise windows and reversed/nonadjacent sites are covered by kernel tests.

The density tape uses O(L \* 4^n) entries. This replacement removes repeated
full-circuit propagation across candidate gates; it does not establish an MPS
or trajectory memory advantage. It assumes fixed, parameter-independent local
channels and no rerouting/rescheduling after deletion.

## Mandatory verification before training

1. Recompute all **7040** archived eligible deletion scores at all **80** saved
   round-input states with the shared adjoint.
2. Require absolute agreement within 1e-10 and **identical full ordered gate
   rankings**, including tie rules. Preserve every recomputed score.
3. At each case/method's first-round state, freshly run both implementations
   for a paired timing comparison and independent score check: 16 pairs in
   total. Alternate which implementation executes first by optimizer label.
   All other verification states compare against authenticated stored exhaustive
   scores. Verification/timing benchmark cost is recorded separately from
   deployed algorithm cost, not hidden as a synthesis operation.
4. Finish verification of every case and write the aggregate verification record
   before starting any optimization job. A mismatch stops the experiment.

## Reinvestment experiment

Each case has **one identical CPU budget for all four arms**, frozen before
this run to the larger of its two historical structural CPU costs. These are
52.899165-58.241136 process-CPU seconds per case. Using a common case budget
avoids a different baseline cutoff for Adam and Krotov. No outcome in this new
experiment changes a budget.

Rerun the original warmup and five-round search with the shared scorer. Require
the warmup and **all repaired candidate states, accepted gates and final search
optimizer state to reproduce history exactly** before beginning extra refinement.
The old exhaustive implementation is retained as a reference, not charged to
the fast search. No saved optimizer state is substituted for the freshly timed
search. Warmup is physically reused between each method's arms but charged to
each, as in the original protocol.

After the five search rounds, freeze the selected structure and continue the
same optimizer in ten-update blocks until its total charged CPU covers the case
budget. Carry all moments and counters forward without resetting the online
phase. Continue the noise stream after the last consumed structural round,
even if a round rejected every update; do not replay discarded-candidate epochs.

Rerun both original fixed-structure controls from their identical warmups to
the same case budget. They use the same original streams, moments and rates.
The four arms are:

| Circuit treatment                                | Adam      | Adaptive online Krotov |
| ------------------------------------------------ | --------- | ---------------------- |
| Fixed 108-gate circuit, continued refinement     | Control   | Control                |
| Shared-adjoint search, then continued refinement | Candidate | Candidate              |

Charge setup, sampling, all candidate updates, endpoint exact scores, shared
deletion scoring and rejected repairs. Exclude historical construction,
artifact I/O, worker import/launch, verification and post-run audit. Preserve
the separate ledgers for the search and the additional refinement; do not
describe total search speedup as optimizer speedup.

Both fixed and structural arms may export the best scored checkpoint available
at the budget. Continue the trajectory after a bad score; select only completed
checkpoints at or before the cutoff. The block crossing the cutoff is excluded,
with no interpolation. Retain a 4000-update safety cap and flag any failure to
cover a budget; such a case cannot establish a matched-budget win.

## Analysis and scope

Report four-arm exact noisy fidelities, target-level paired differences,
equal-target means, gates, selected/executed update counts, and matched timing
ratios. Separate the unchanged search endpoint from the later refined endpoint
so extra refinement is not attributed to a different deletion decision.

Retain the prior practical screen: at least +0.5 percentage points mean gain
over fixed Adam and positive mean gains in at least three families. Report both
structural optimizers, without a per-target winner mixture. Also compare them
directly at the common budget, with wins/losses and family differences.

This remains four repeatedly used development states, not held-out generalization.
No automatic new-target or hardware run follows any result. Noisy versus ideal
ranking, other synthesis methods, general Lindblad channels, realistic native
gate schedules and hardware calibration uncertainty remain outside this run.

## Execution

```bash
uv run --no-sync python -m experiments.state_preparation_adjoint_restructuring \
  --history-root output/statevector-noisy-restructuring-20260912 \
  --output-root output/statevector-adjoint-restructuring-20260912 \
  --workers auto
```

Eight case jobs use eight single-threaded numerical processes. `--stage verify`
runs the mandatory verification only; resume with `--resume` to proceed after
that stage. All completed stage/case artifacts are checksummed and atomic;
an incomplete case reruns from its source. Identical-manifest completed resume
must preserve bytes. Scientific source changes require a fresh output root.
