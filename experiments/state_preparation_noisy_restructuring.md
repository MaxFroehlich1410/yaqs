# Noisy deletion-and-repair: small-system headroom test

Version: `yaqs.noisy_restructuring_headroom.v1`. Frozen before the scientific run.

## Question and limits

Does selecting actual gate deletions by noisy fidelity, followed by noisy
parameter repair, outperform continued optimization of a fixed circuit? Does
adaptive online Krotov add an advantage over Adam inside the same procedure?

This is an exact-score-assisted, six-qubit **diagnostic**, not an entirely
trajectory-based synthesis algorithm, native compilation benchmark, hardware
result or new-target confirmation. Exact density scores select structures and
accept/reject candidates; gradients and online signals use weighted trajectories.
No claim that architecture search or pruning is itself novel is made.

## Frozen panel and source

Use the four existing development states: `gaussian_mu0p5_sigma0p1`,
`tfim_critical`, `haar_random_1`, `random_mps_bond2`. Use both existing
construction replicas 11 and 29, not two noise replicas of the same construction.
The source is the authenticated `H25_lr0p7:cold_final` construction extension:

`/Users/maximilianfrohlich/Desktop/yaqs-experiment-results/q6-hybrid-construction-extension-20260906`

Each source has 108 scalar parameters/gates: 78 one-qubit and 30 two-qubit gates.
These are earlier construction endpoints, **not** the later four checkpoints
used in the adaptive-rate screen. Consequently absolute means from the two
screens must not be compared as a matched continuation.

All four arms within each of the eight cases start from identical parameters:
fixed Adam, structural Adam, fixed adaptive online, structural adaptive online.
No new targets, hyperparameter search, insertion, replacement, layout search,
native compilation, additional noise models or extrapolated hardware claims.

## Noise and optimization

- Initial state: all zeros. Complex128 statevector trajectories; no MPS training.
- Same calibrated homogeneous local Pauli-Lindblad proxy as the previous pilot:
  complete-window average errors 0.00064 / 0.0051 for one-/two-qubit gates.
  The exact independent-factor unraveling and exact density evaluator agree.
- Weighted R32: one deterministic no-factor path with its exact probability,
  and 31 paths conditional on at least one selected Pauli factor. Rebuild the
  sampler and weights after every physical deletion; never reuse the old p0.
- Adam: learning rate 0.02, beta1=0.9, beta2=0.999, epsilon=1e-8.
- Adaptive online Krotov: first 25 updates at unscaled rate 0.7, then online
  RMS-scaled updates at 0.02, beta2=0.999 and epsilon=1e-8. This is the previously
  selected schedule, with a continued adaptive tail, not a new rate screen.
- Each method starts with a 50-update noisy warmup. Its structural/fixed arms
  share that prefix; compute is charged to each arm, even when reused.
- After deletion, retain moments and occurrence counters of surviving parameters
  in compact parameter order. Never restart the aggressive phase or moments
  for each candidate. No shared-parameter circuits are allowed in this driver.
- Fresh ensemble at every update. Root seed 2026091207; SHA-256-derived per-case
  seed independent of method, execution order and worker. Within the same circuit
  and epoch, both optimizers receive identical maps and weights. Different
  structures receive the same seed, but **not** a claimed gate-aligned CRN coupling.
  Within each update, forward/costate propagation replays the same maps.
- Repair round r uses epochs 50+(r-1)*10+1 through 50+r*10, even if the previous
  round retained its input. Rejection therefore does not freeze training noise.

## Structural procedure

Five rounds, at most one gate accepted for deletion in each round:

1. Protect the original 18 initial product-layer parameters, as in the earlier
   compression pilot. Exhaustively score each other single-gate deletion by
   **exact noisy fidelity before repair**. Remove both gate and its abstract
   noise window; setting its angle to zero is not deletion.
2. Rank highest fidelity first, with 1e-12 rounding and original gate ID as the
   deterministic tie rule. Shortlist two candidates, without arity quotas or
   family-dependent settings.
3. Fork the current state into unchanged and two deleted candidates. Apply
   ten noisy updates to each, using the same optimizer and preserved moments.
4. Score all three repaired candidates exactly. Also retain the unmodified,
   unrefined input as a free known candidate. Accept a candidate only if it
   improves the incumbent by more than 1e-10; ties prefer retention, then
   unchanged repair, then shortlist order. Continue all five rounds even if
   a round makes no deletion. Record all rejected candidates and scores.

The same procedure may select different gates/counts for different targets; no
per-target hand-tuned settings or outcome-dependent optimizer choice is allowed.
This tests greedy single deletions, not the full space of useful circuit edits.

## Cost controls and decision rule

Charge process CPU for initial scoring, complete warmup/repair blocks (including
engine and sampler construction, sampling, updates, trace creation and exact
endpoint score), exhaustive deletion scoring and **all rejected repairs**.
Keep a role-resolved ledger. Exclude historical construction, artifact I/O,
worker imports/launch, final audit and report generation. Report these exclusions.
The physical prefix is computed once per optimizer but charged to both arms.

For each case, first measure both structural arms. Continue both fixed-circuit
optimizers until each covers the larger structural CPU cost. Evaluate every ten
updates; carry moments without resetting. Continue the trajectory even after
an unfavorable score, but allow the fixed arm to export its best exactly scored
checkpoint, so exact model-based selection is not reserved for restructuring.
Stop at at most 4000 total updates as an engineering safety cap. If the cap
prevents budget coverage, flag it; do not claim a matched-budget win.

For each structural endpoint, compare with both fixed arms at that structural
arm's CPU cutoff, using only completed scored checkpoints within the cutoff.
Do not use the block crossing the cutoff, interpolation, or a future checkpoint.
Report fixed-control update counts, final structural gate counts, target-level
paired differences and equal-target means (average construction replicas first).
Also compare the two structural arms at their common CPU cutoff; at that cutoff,
only completed structural rounds are available. CPU-stopped control lengths may
vary with machine load; seeds and per-update scientific trajectories do not.

The proposed practical screen is at least **0.5 percentage points** mean gain
over compute-matched continued fixed Adam, with positive mean gains in at least
three of the four families. Report each structural optimizer separately, without
constructing a per-target winner mixture. Passing this diagnostic does not prove
novelty, population generalization, hardware transfer or competitiveness with
published synthesis methods. No automatic new-target launch follows it.

The deployed-circuit limitation is explicit: deleting an abstract gate removes
its abstract noise window here. A real scheduled circuit may retain idle time
and decoherence. Native gate decomposition, routing and idle noise are not yet
modeled and must be validated before a hardware-fidelity claim.

## Execution and reproducibility

```bash
uv run --no-sync python -m experiments.state_preparation_noisy_restructuring \
  --source-root /Users/maximilianfrohlich/Desktop/yaqs-experiment-results/q6-hybrid-construction-extension-20260906 \
  --output-root output/statevector-noisy-restructuring-20260912 \
  --workers auto
```

Eight independent case jobs use eight processes with single-threaded numerical
libraries. Each case is written atomically when complete, including the source
identity, all optimizer states, decision scores, update replay hashes and costs.
Completed cases are authenticated and reused by `--resume`; an interrupted
incomplete case is rerun from its source. No partial-iteration timing equivalence
is claimed. An existing output root without explicit identical-manifest resume
is rejected. The manifest fingerprints scientific code, protocol and sources.

Tests cover finite noisy deletion scores, removal rather than zeroing, moment
compaction, refreshed weights, optimizer wiring and phase continuation, exact
acceptance/ties, rejected-work charging, budget cutoff selection, safety caps,
case-level resume/authentication and serial/parallel scientific reproduction.
