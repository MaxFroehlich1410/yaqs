# Strong structural baseline screen

Preregistered before these reruns. Four unchanged q6 development targets,
construction replica 11 only. No confirmation states or hyperparameter tuning.

## Source audit

`state_preparation_noisy_restructuring.py` and
`state_preparation_adjoint_restructuring.py` used weighted R32 trajectory Adam,
not exact-density Adam. Their selected adaptive online schedule used 25
unscaled updates at 0.7 followed by RMS-scaled updates at 0.02, preserving
surviving moments and occurrence counters after deletion. The old search used
five rounds, warmup 50 and repair 10. The shared-adjoint replacement reproduced
its entire trajectory before reinvesting saved CPU. Its 53–56 second seed-11
budgets reached roughly 1040–1060 structural adaptive updates; a 150-update
baseline therefore does not cover the established adaptive tail.

Those historical controls exported best-scored incumbents. Their `refined.state`
may additionally belong to a block crossing the CPU budget. Historical
references must use the last archived incumbent available at the budget
(`at_budget`), remain explicitly unmatched, and never substitute for a fresh
final-iterate rerun.

## Frozen new arms

Start every arm at the identical authenticated 108-gate construction in
`starts.json` (78 one-qubit and 30 two-qubit gates), also used by
`structural_lbfgs25`. Reuse only historical construction, with its unavailable
original cost equally excluded from all matched continuation arms.

1. Structural exact-density Adam, learning rate 0.02.
2. Structural weighted R32 Adam, learning rate 0.02: the actual repository Adam
   comparator, using its unchanged weighted gradient kernel.
3. Structural adaptive online R32: one initial 25-update phase at 0.7, then
   the established 0.02 RMS tail.
4. Fixed-circuit adaptive online R32 extended to 1500 updates.

Adam uses beta1=0.9, beta2=0.999, epsilon=1e-8. Adaptive online uses the existing
beta2=0.999, epsilon=1e-8 kernel. Each update refreshes its weighted R32 ensemble.
The old per-case seed namespace and epoch formula are retained. Within a repair
round, candidates receive the same epoch seeds; different structures are not
claimed to have aligned noise factors. Discarded repair epochs are consumed
before polishing, even when the original parameter state is retained.

Every structural arm uses warmup 150, at most 15 rounds, shortlist two, repair
25, and a total 1500 update cap charging warmup, all retained/deleted repairs
(including rejected branches), and final polishing. The unchanged repaired
branch and two deleted branches receive equal update caps. Use exactly
`structural.adaptive_delete_refine`: shared exact finite-deletion ranking,
1e-12 rank rounding, 1e-10 strict acceptance, initial 18 gates protected, and
stop when no repaired deletion beats retention. Deleting a gate removes its
unitary and noise window and compacts moments in surviving parameter order.

Preserve Adam moments and adaptive moments/counters through selection,
deletion and polishing; never restart the aggressive adaptive phase. The
L-BFGS comparator deliberately uses its own frozen fresh-curvature repair
policy. This is a comparison of fully specified optimizers with their
appropriate state policies, not an isolated test of one identical memory rule.
The fixed adaptive arm shares its physical warmup with the structural adaptive
arm; charge that warmup fully to both.

## Fairness and final-iterate rule

The 1500 update caps are deterministic work limits, not equal computation:
one density gradient, one R32 batch gradient, and one R32 online sweep have
different costs. L-BFGS instead has its already frozen 1500 channel-call cap,
including structural scoring. Report these full-cap endpoints descriptively.

For the primary CPU comparison, include these four new arms plus the frozen
structural/fixed L-BFGS pair. For each target, choose the smallest complete
arm's measured charged process CPU. Select each algorithm's last deployable
checkpoint available at or before that cutoff, never its highest-fidelity
checkpoint and never an interpolated or future iterate. Structural checkpoints
become available only after all candidate repairs and exact acceptance finish;
discarded branches are not deployable checkpoints. Polishing and fixed control
continue their last numerical iterate after unfavorable scores. A missing or
failed arm disables the complete-panel CPU comparison for that case.

Charge setup, sampling, updates, exact endpoint scoring, shared ranking, rejected
repairs, trace creation and checkpoint I/O. Distinguish exact gradient calls,
exact forward scores, density ranking passes, trajectory paths and trajectory
direction passes. No equality between density and trajectory passes is claimed.
Record physical caught-interruption replay CPU and work separately; interrupted
search attempts with unrecoverable score counts are explicitly flagged, not
silently assigned zero. Worker imports/launch and final twelve-profile auditing
remain outside algorithm CPU. This follows the new L-BFGS accounting rather
than the historical protocol's exclusion of checkpoint I/O.

Before execution, a reporting-cost audit replaced the legacy block's mandatory
Qiskit endpoint score with the fast exact local-channel evaluator for every
optimizer checkpoint. A local adapter retains the historical update loop,
ensemble hashes, moments and counters exactly; automated tests compare them
directly with the legacy implementation. This changes diagnostic implementation
cost, not the update algorithm or mathematical score. Initial diagnostics use
the same fast evaluator. The shared structural helper's postrepair scorer is
unchanged and identical across baseline and L-BFGS search. Independent Qiskit
scoring is reserved for the final twelve-profile audit outside algorithm CPU.

Evaluate all twelve unchanged benchmark profiles on every full endpoint and
every available common-CPU endpoint. Include original zero-angle physical gates;
only the existing Ballarin materializer performs its fixed pruning and supplies
its specified noiseless basis changes. Reuse already authenticated historical
fan-out records only as unmatched references. Equal-weight all eleven noisy
conditions and four targets; retain every unfavorable or failed case.

## Artifacts and execution gate

New `baseline_experiment.py` only; existing frozen modules remain unchanged.
The prepare-only mode authenticates construction starts, immutable benchmark
hashes, numerical sources, this protocol and the structural-L-BFGS protocol.
It does not require or inspect unfinished L-BFGS endpoint scores. Execution
requires its completed authenticated case records and a separate `--run`.

Save full optimizer states, deterministic ensemble hashes, branch/epoch
decisions, all costs, final curves and all twelve-profile results. Accepted
checkpoint resume preserves moments, counters and noise epochs. A completed
artifact is read-only on identical resume. Custody/configuration changes abort;
numerical failures remain explicit rows. No scientific launch is authorized by
this preparation task.
