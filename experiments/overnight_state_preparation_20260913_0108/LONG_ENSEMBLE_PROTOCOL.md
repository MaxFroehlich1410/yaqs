# Full-family convergence diagnostic

Preparation only until the coordinator freezes actual source hashes and allocates
execution. No held-out identity is constructed, trained or scored by this protocol.

The earlier fixed original-circuit mean11 panel received only 100 joint gradients
at its 1,100 individual-channel cap; mean2 received 550. That comparison could
confound objective choice with finite optimization progress. This diagnostic tests
whether additional joint iterations resolve the observed full-family deficit,
especially on the development Haar target. It does not establish a new optimizer.
Exact gradients, ordinary Adam, Armijo L-BFGS and a uniform finite mixture objective
are known methods. The numerical optimizer and derivatives remain the unchanged
`ensemble_experiment.train_arm` and `ensemble_objective` implementations, including
the Ballarin probability derivative. Source provenance and citations remain in
`LITERATURE_SCOUT.md` and `ENSEMBLE_PROTOCOL.md`.

The panel contains exactly the original q6 development targets
`gaussian_mu0p5_sigma0p1`, `tfim_critical`, `haar_random_1` and
`random_mps_bond2`, each at construction replica11. Every arm starts from the
same authenticated original 108 scalar parameters and 78 one-qubit plus 30
two-qubit gates. No structure changes, compilation, conditional initializer,
warm start from an earlier endpoint, or target-dependent optimizer setting is used.

| Arm          | True objective                            | Optimizer | Joint cap | Individual-channel cap |
| ------------ | ----------------------------------------- | --------- | --------- | ---------------------- |
| mean2_lbfgs  | Equal local depolarizing/dephasing mean   | L-BFGS    | 3,025     | 6,050                  |
| mean11_lbfgs | Equal mean of all eleven noisy conditions | L-BFGS    | 550       | 6,050                  |
| mean2_adam   | Equal local depolarizing/dephasing mean   | Adam      | 3,025     | 6,050                  |
| mean11_adam  | Equal mean of all eleven noisy conditions | Adam      | 550       | 6,050                  |

The mean2 profiles are precisely `depolarizing_1s_all` and `dephasing_1s_all`.
The mean11 order is the frozen `ensemble_objective.NOISY_IDS`, including Ballarin.
The noiseless condition is evaluated but never optimized. Adam learning rate is
0.02; other Adam constants are inherited from the frozen primitive. L-BFGS uses
history10, its original Armijo/backtracking/curvature constants, and a maximum
iteration count equal to its joint-evaluation cap. All failed line-search trials
consume calls. Early convergence is retained and unused calls are not recycled.
There is no learning-rate, temperature, optimizer-history or additional-budget grid.
Arm execution order rotates by frozen target index among the four methods.

The endpoint is the final accepted L-BFGS iterate or final Adam update. The last
pre-update Adam loss is not presented as that endpoint's quality. Each final
endpoint receives all twelve exact calibrated channel scores. Post-training CPU
comparisons use the latest chronologically available iterate at fractions
0, 0.25, 0.5, 0.75 and 1 of the shortest completed arm's training CPU for that
target. Iterates are selected by time before looking at their all12 scores.
The primary CPU comparison is fraction1 among these four arms only. The other
fractions describe convergence; they are not candidates for quality selection.
Every selected point is rescored on all12, cached by exact circuit/target/theta
identity, and accounted as separate forward work. Full-cap and common-CPU rows
remain separate reporting strata.

The shared CPU scope is the unchanged primitive's optimization time: gradients,
optimizer operations, checkpoint writes, failed line searches and observed replay
are included. Its final/attempt-record writes, wrapper work, imports, custody,
the existing shared original construction and all12 scoring are excluded from
every training curve. Wrapper invocation CPU and its difference from primitive
CPU are reported separately; scoring has a separate attempted/completed component
ledger, including failed points without a completed result. Thus the compute claim
is a refinement claim, not a claim of lower total state-construction cost. Logical
joint/component calls, forward/gradient direction passes, physical attempted and
completed work, replayed work, primitive CPU, invocation CPU and scoring CPU remain
available. Neither heterogeneous joint iterations nor equal component caps alone
are described as equal wall or CPU time.

Training checkpoints preserve the complete optimizer state and accepted-iterate
history every ten accepted iterations. The original primitive resumes caught
interruptions exactly and charges all observed lost work before replay. A durable
outer marker is written before each invocation and closed on every catchable exit.
A marker still running on restart means process-death work was not observed; that
case is rejected for resume and resource comparison, and remains a visible failed
case. The same rule applies to scoring attempts. Cached training verifies the
primitive attempt checksum, and completed cases bind all durable JSON bytes.
No unknown kill cost is imputed as zero or silently repaired. Numerical failures
are retained, and a failed arm prevents a complete common-CPU four-arm panel.
There is no method/target omission or changed hyperparameter retry after a failure.

For promotion of this objective/budget recipe, one fixed mean11 optimizer must
improve the equal-target, equal-eleven-condition mean by at least 0.5 percentage
points over the best overall mean2 optimizer at the common-CPU endpoint, with a
positive paired difference on each of the four targets. The comparator is selected
once by its aggregate across all four targets, never separately per target.
The alternative resource route requires one fixed mean11 optimizer's full-cap
mean to be within 0.1 percentage points of the best mean2 full-cap mean while its
full-cap primitive CPU is at most 80% of that comparator on every target. This
alternative is a predeclared refinement-resource claim and must retain every
per-target/channel score and the common-CPU comparison. Otherwise retire this
6,050-call full-family budget. Any development success still requires a separately
frozen untouched confirmation panel and does not itself support an optimizer claim.

`PROTOCOL.json` must exist before scientific execution and freezes all settings,
ordered target identities and raw state-vector hashes, complete original starts
and their source-file hash, benchmark hashes, imported numerical source hashes,
this protocol, endpoint rules and CPU scope. `--prepare-only` performs custody
and construction validation only. `--run` verifies the complete protocol before
and after each case. Up to four independent target workers are allowed only with
single-thread numerical libraries and explicit coordinator allocation. Scientific
results, all intermediate parameter curves, full optimizer/attempt state,
all12 scoring records and checksums are retained under the newly allocated output
root. Tiny q1-q3 engineering tests may use a smaller divisible-by22 budget; those
settings cannot pass the scientific protocol verifier.
