# Conditional rank-two resource initializer, development only

This is a new development hypothesis informed by the earlier rank-two screen.
It does not modify confirmation or claim a unique optimizer advantage. Keep
all four fixed development targets at replica 11, including Haar. The policy
does not inspect family names, noisy scores, optimizer results or resource
limits when choosing its initial circuit.

## Fixed initializer

Always run the frozen deterministic rank-two TT-SVD physical synthesis and
its exact scalar lowering. Independently calculate
`abs(vdot(target, normalized_TT_approximation))**2`. Use the rank-two initializer
iff this ideal overlap is >=0.99; otherwise use the reconstructed legacy
H25/rate-0.7 cold-final 108-gate initializer. The threshold has no tolerance
band, learned exceptions or family-specific overrides. Independently audit
the prepared scalar circuit against its TT vector. Apply the frozen physical
compiler to the rank-two probe even if it is rejected. Also compile the
legacy initial circuit for both policies. Every retained physical operation
and noise window follows the unchanged compiler/channel rules.

Charge the full TT synthesis, measured overlap/audit and probe compilation
even on fallback. Charge the complete measured legacy construction and its
initial compilation whenever legacy is selected. A selected rank-two policy
does not inherit the legacy construction cost that exists solely to support
its comparator. Report actual shared execution costs separately from full
per-arm logical charges. Keep raw source counts and compilation mappings.
Unknown historical CPU is never substituted with zero or inferred from wall
time. Scientific preparation requires the fresh four-case construction batch
to have reproduced its archived circuits, parameters and ideal states, with
authenticated complete timing and source/checkpoint custody.

## First mechanistic comparison

Six independent arms per target: legacy or conditional initial source, each
with unchanged, mean-two exact Adam, or mean-two exact L-BFGS. Both source
policies receive initial compilation before training. Optimize the equal
mean of unchanged calibrated local depolarization and dephasing. Adam uses
550 updates at 0.02; L-BFGS uses at most 550 joint objective-gradient calls,
history ten. Both caps are 1100 constituent channel-gradient calls. Use final
Adam updates/final accepted L-BFGS iterates, never best-checkpoint selection.
No deletion search, objective retuning or extra optimizer restart is included.

After every ten accepted optimizer updates, and at the final accepted
endpoint, compile an independent copy for deployment; the training circuit
and optimizer memory keep their fixed parameterization. Compilation changes
physical noise windows and is intentionally applied to every trained arm.
Persist full optimizer state and compact theta/digest/CPU curves. Store each
compiled physical circuit once by its digest. Do not copy circuits or full
optimizer histories into each outer result/curve record. All twelve exact
benchmark profiles are evaluated on every final deployed circuit, with the
same final channel audits for both policies and both optimizers.

## Cost clocks and comparisons

Report complete initializer cost, optimizer CPU, all checkpoint/final compiler
CPU, independent audit CPU and all attempted/replayed work. Initialization is
physically performed once per target and its full measured cost is charged
to each independent arm using it. Probe work on rejected rank-two cases stays
visible. Hard-killed attempts with unmeasured duration make total physical CPU
unknown; do not silently fill it with zero.

If the shared initializer-building invocation is interrupted, its measured
unfinished setup overhead is charged equally to both source policies and
reported separately. A successful retry uses the identical deterministic
policy. Unattributable hard-kill time makes both total-cost clocks unavailable;
the refinement-only comparison remains distinct.

The four trained arms define the common CPU cutoff (the shortest completed
trained arm), with fractions 0, .25, .5, .75 and 1. Unchanged arms are references
and do not collapse the training budget to their near-zero duration. Select
the latest available deployed checkpoint at or before each cutoff, without
interpolation or score-based selection. The total-cost clock starts before
construction; an arm whose initializer is not yet available has no point at
that budget. Preserve this unavailability and do not average over a reduced
panel. Also report a separately labeled refinement-only comparison excluding
construction/probe/initial compilation while charging the deployed checkpoint
compiler work. This separates resource-policy effects from optimizer work.

Every run remains development-only. Preparation authenticates the fresh
construction reproduction and freezes all sources before any new rank-two
synthesis, training or noisy evaluation. Explicit execution is a separate
step. Retain all failures, fallbacks and unfavorable outcomes. Do not change
the policy or confirmation construction from these results without a new
protocol and development comparison.
