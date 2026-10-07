# Untouched q6 confirmation design — DRAFT, NOT FROZEN

This is an independent design review, not an execution protocol. **No
confirmation target has been constructed, optimized, scored, or used to select
a setting for this draft.** It does not authorize a run. The coordinator must
finish all queued development comparisons, resolve the choices explicitly
listed below, implement/test the complete selected pipeline, pass full lint,
and write a new immutable protocol/configuration/source hash before accessing
confirmation outcomes. Existing frozen development files remain unchanged.

## 1. What is being confirmed

Select exactly one complete candidate pipeline using development evidence only:
constructor/initializer policy, physical compiler, training objective,
optimizer, estimator, optional structural search, memory policy, budgets and
deployment rule. Do not assemble individually successful components for the
first time on confirmation. Any proposed composition must first pass a complete
development run on the four already exercised targets, with its equally treated
controls. In particular, conditional initialization plus structural mean-two
training plus deployment compilation is not automatically validated by their
separate component screens.

Before freezing, state one primary claim:

- **Optimizer claim:** the chosen optimizer improves on distinct, serious
  optimizers given the same initialization, objective, estimator opportunity,
  compiler and structural opportunity.
- **Pipeline/repository-baseline claim:** the complete recipe improves the
  existing repository baseline, with explicit ablations locating an objective,
  initialization, architecture or compilation contribution.

If the winner is an upgraded Adam recipe, label it Adam or the actual hybrid.
Giving an Adam control every identical upgrade can produce the identical
algorithm. Deduplicate that arm by its complete numerical configuration; record
the candidate as also satisfying the corresponding fair Adam-control role.
Do not demand a strictly positive difference against itself, rename an
identical run as independent evidence, or retain a deliberately handicapped
Adam solely to manufacture an optimizer win. Compare the pipeline against the
strongest **distinct** fair controls and its preregistered component ablations.
Improvement over the prior recipe can establish a new in-repository baseline
without establishing a new optimizer or general state-of-the-art result.

The strongest distinct primary baseline B* is selected once from completed
development results, using the same primary metric and cost scope as the
candidate. Freeze its identity before confirmation. Also retain all mandatory
fair controls below. Do not choose a different weak baseline for each held-out
target or replace B* after seeing confirmation scores. Report whether any other
distinct fair control beats the candidate in aggregate; such a result prevents
a broad strongest-method claim even if the candidate beats B\*.

## 2. Fixed panel and paired randomness

Use exactly these five q6 target identities:

1. `tfim_ferro`
2. `tfim_para`
3. `haar_random_2`
4. `haar_random_3`
5. `random_mps_bond3`

All five remain in every table and aggregate. A family label must not influence
algorithm settings. Reading/validating the complete normative vector fixture
was required at startup and is distinct from using held-out outcomes. Audit
the overnight record before freezing: if held-out construction, fidelity or
optimizer outcomes have already influenced a design decision, disclose that
contamination and do not call this panel untouched.

**Draft replication proposal:** two paired construction/training replicas,
11 and 29, on every target, with no best-seed selection. This means ten
target–replica cases per unique trained arm. The final number must be confirmed
against remaining compute before freezing. An inability to finish the entire
panel is a reason to defer confirmation, not to remove Haar, drop a failed seed,
or add extra seeds only to a method with an unfavorable result.

Use the validated legacy constructor's actual master-vector seed formula from
[CONSTRUCTION_PROTOCOL.md](CONSTRUCTION_PROTOCOL.md), including its source
version, root and target/replica identity. All arms sharing that constructor
receive the same physical start and theta. Do not replace it with the similarly
named later construction namespace. Conditional rank-two synthesis is
deterministic; identical deterministic repetitions are not additional
independent training samples.

Derive stochastic training seeds from a new `confirmation_training` domain,
target and replica, then the frozen channel/update/path allocation. Optimizer
labels must not change paired sampler opportunities. Persist every counter.
Sibling structural repairs may use the same new epoch interval, as already
preregistered in development; later selected-state training must advance beyond
all those repair epochs. Different architectures are not asserted to realize
the same physical events from the same integer seeds. Debugging and diagnostics
use disjoint domains. Exact all-channel final evaluation consumes no random
stream; any later trajectory evaluation uses a separately frozen
`confirmation_final_evaluation` domain, disjoint from all other purposes.

## 3. Mandatory fair control roles, with identical arms deduplicated

The final protocol must expand this role table into an explicit finite arm list,
complete configuration hashes and budgets before confirmation. It is not a
license for a factorial grid or outcome-dependent additions.

| Role                                                     | Required opportunity and interpretation                                                                                                                                                                                                                                                                  |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Unchanged start                                          | The actual selected initializer, with the same initial/deployment compiler and all physical preparation gates. No training; a reference, excluded from the trained-arm CPU cutoff.                                                                                                                       |
| Matched noiseless optimization                           | Serious noiseless Adam, or the stronger validated noiseless optimizer selected on development. Same initializer, compiler, endpoint rule and meaningful budget. This intentionally different objective is the noise-awareness ablation.                                                                  |
| Exact noise-aware Adam                                   | The strongest validated exact Adam implementation on the candidate's training objective. Give it the same constructor/compiler and appropriate structural opportunity. Use the fast validated exact oracle where applicable.                                                                             |
| Weighted trajectory Adam                                 | Fresh weighted sampling for the same objective, with the same validated estimator and trajectory allocation as any stochastic candidate. For mean-two, use 32 paths per channel unless a different allocation was fully developed and frozen.                                                            |
| Strong structural Adam                                   | The strongest applicable validated structural Adam memory/update recipe, including equal retained/deleted repairs, exact finite ranking and all rejected work. If identical to another arm, record the role once.                                                                                        |
| Strong adaptive online/Krotov control                    | The strongest applicable validated schedule, currently the 25-update online phase at 0.7 followed by adaptive 0.02 scaling, with surviving moments/counts retained and the same objective/initializer/compiler opportunities. A pure Krotov claim additionally needs a win against equally treated Adam. |
| Strong exact alternative                                 | Mean-two exact L-BFGS and/or the distinct optimizer that is B\*, with its actual validated memory/line-search policy. Do not omit a stronger exact method just because it is not the proposed method.                                                                                                    |
| Component ablation, only if needed for the primary claim | One preregistered earlier objective or legacy initializer/architecture recipe, with other components matched. It isolates the claimed component; it does not replace the strongest fair control.                                                                                                         |

“Same opportunity” means a competent validated configuration and accounting,
not forcing every optimizer into an identical but inappropriate memory policy.
The existing exact structural screen resets moments/secants per stage, whereas
the strongest stochastic controls preserve surviving memory. State this
explicitly. If a new memory policy, estimator, compiler or initializer changes
which baseline is strongest, finish that development comparison before freezing.

If the selected objective is nonlinear entropic risk, the current unbiased
mean-two sampler is not automatically an unbiased risk-gradient estimator.
Softmax weights obtained from noisy channel-loss estimates introduce a separate
estimation problem. Develop and validate a correct, charged stochastic control
or explicitly limit the claim to the compatible exact setting before
confirmation; do not silently substitute a mean-objective stochastic control
and call it objective-matched.

## 4. Constructor, conditional policy and compiler custody

If conditional initialization survives development and is selected, freeze its
existing input-only rule: always form the rank-two TT approximation, use its
physical scalar initializer iff ideal target overlap is at least 0.99, otherwise
use the reproduced legacy H25/rate-0.7 cold-final initializer. No family names,
noisy scores, target-specific thresholds, favorable seeds or runtime limits
enter this decision. Freeze TT rank, SVD/gauge conventions, scalar lowering,
normalization and audit tolerances. Offer this same policy to every corresponding
fair control. A legacy-only arm is a specifically labelled constructor ablation.

Charge the TT synthesis/probe, overlap audit and probe compilation even on
fallback; also charge complete legacy construction and initial compilation when
selected. A rank-two-selected arm does not inherit legacy construction done
solely for another comparator. Conversely, a comparator's expensive construction
is never assigned zero because a shared copy already exists. Record actual
shared physical execution separately from full standalone per-arm charges.
Historical starts with missing CPU are unsuitable for an end-to-end CPU claim.
Fresh confirmation construction needs measured complete CPU/wall and exact
resume from the now-validated construction recipe.

Freeze the supported scalar gate set, angle reduction, 1e-12 identity threshold,
disjoint-support fusion rules, ordered sites, noise-flag barriers and source
mapping from [COMPILATION_PROTOCOL.md](COMPILATION_PROTOCOL.md). Compilation
changes physical gate-noise windows and is therefore an algorithmic component,
not cost-free cosmetic postprocessing. Give it to every compatible fair control.

If using the currently developed deployment-copy approach, compile independent
copies initially, after every ten accepted updates, and at the final accepted
endpoint. Training retains its own parameterization and memory. Do not silently
feed a compiled copy back into optimization; that is a different algorithm
requiring development validation. Charge compiler/audit work and make a deployed
checkpoint available only when both finish.

Original construction parameter indices are permuted and logical construction
IDs are sparse. Preserve all metadata, compact moments by surviving original
parameter index, and track initial physical gate positions separately for
ranking ties/protection. The current mean-two structural and mixture controls
use those physical origins; their first-18 protected set agrees on the original
starts. A compiled or rank-two circuit needs its own explicitly frozen protected
set and mapping. Blindly applying “protect first 18” to a different initializer
is not yet a validated shared structural opportunity.

## 5. Budgets, available endpoints and the total CPU clock

Before freeze, write one exact budget table for every distinct arm. Include
constructor depth/sweeps/pruning, gradient/joint/channel calls, trajectory counts,
warmup, rounds, shortlist, per-branch repair, polishing, line-search limits,
compiler schedule, convergence tests and emergency resource ceilings. Current
development recipes supply candidate values such as 550 mean-objective calls
for fixed refinement, 3000 constituent exact calls for structural mean-two,
and 1500 aggregate stochastic updates including rejected repairs. These are
**examples to resolve, not interchangeable or frozen confirmation budgets**.
An upgraded pipeline must receive budgets already validated with its controls;
do not transfer numbers solely because they have the same “update” label.

Prefer the existing chronological **final-iterate** policy. Fixed/polishing arms
keep their last update even after an unfavorable diagnostic. Structural search
uses its frozen internal acceptance rule, but only exposes the chosen incumbent
after all required branch work and acceptance scoring finish. Rejected branch
checkpoints are never retrospectively deployable. Accepted L-BFGS points and
completed Adam/online updates are distinct legitimate iterate definitions;
their pre-update losses are not final fidelity scores.

For each target–replica case, the draft primary cost comparison uses the
minimum completed **trained-arm end-to-end CPU** among the frozen fair primary
panel. Exclude unchanged-start reference arms from that minimum. Select each
arm's latest fully constructed, compiled and deployable checkpoint at or before
the cutoff, using time only; score it afterward. Do not interpolate, extrapolate,
choose the highest score, or borrow an iterate whose compilation finished later.
If an initializer is not yet available at the cutoff, that arm has no endpoint.
Report the complete-panel comparison as unavailable; do not average the subset
that happened to finish. The final development rehearsal must check that this
rule is informative for the proposed constructor/control combination.

The new primary total-cost clock starts before target-specific construction
and includes target conversion needed by the algorithm, initialization/probes,
lowering, setup, sampling, all objective/gradient calls, failed line searches,
ranking, retained/rejected repairs, structural acceptance scores, compilation,
algorithmic audits, normal checkpoint work and caught-interruption replay.
Freeze one consistent exclusion boundary for worker startup/imports and purely
administrative manifest/final-summary writes, and report those separately.
Final all-twelve scoring and post hoc diagnostics have separate measured ledgers.
Do not add component sub-times again when already inside an enclosing timer.

Existing development continuation clocks omit unknown original construction
cost and have specific ledger-I/O exclusions. They cannot simply be relabelled
as end-to-end confirmation CPU. A secondary refinement-only plot may reproduce
that clearly defined exclusion, but the primary pipeline/resource claim must
use complete comparable construction and compiler costs. Also report full-cap
endpoints, actual CPU/wall, peak memory and logical/physical call counts; equal
caps are not equal computation. Never equate density directions with trajectory
directions. Report method-development/tuning compute separately from each frozen
confirmation inference/training cost, including all abandoned development runs.

## 6. Primary score and conservative acceptance

For each target, average its final expected fidelity equally over the eleven
noisy conditions and equally over the two frozen replicas. Then average equally
over the five targets. Noiseless fidelity is separate. Report every one of the
twelve conditions, both Haar cases, target means, family summaries, mean/median,
worst target/family/channel, paired wins and gate/CPU Pareto points.

Let delta_t be the candidate minus frozen B\* difference in each target's
replica-balanced eleven-noisy mean at the primary common CPU. The recommended
**fidelity route**, to be finalized before confirmation, requires all of:

1. Positive target-balanced mean gain and at least **0.005 absolute fidelity
   (0.5 percentage points)** over B\*.
2. Positive delta_t on **all five targets**, above the frozen numerical audit
   tolerance. This is the conservative deterministic-consistency route allowed
   by the user goal; exact conditional fidelity evaluation alone does not imply
   a population-level statistical confidence interval.
3. No unexplained numerical failure, no omitted target/replica, and no broad
   regression hidden by one large gain. Show every replica-level contrast;
   conservatively flag any target–replica mean loss exceeding one point or any
   individual-channel loss exceeding five points for mandatory failure analysis.
   An unexplained such loss blocks promotion rather than being averaged away.
4. No distinct mandatory fair control has a superior target-balanced mean at
   the same comparison point; otherwise narrow or reject the strongest-method
   claim. Freeze which comparisons support the primary claim and avoid a
   post hoc choice of one favorable baseline.
5. Complete reproducibility and measured, known physical work.

If development instead justifies a **resource route**, freeze it as the primary
route before confirmation; it is not a rescue exception chosen afterward.
A conservative proposal is positive overall mean fidelity gain, no target mean
loss larger than 0.1 points, and either at least 20% fewer deployed two-qubit
gates or at least 25% less complete standalone CPU against B\*, while never
obtaining that resource mean through one exceptional target. Require the chosen
resource advantage on every target and report all other resource axes. Gate
counts use actual lowered physical operations, with Ballarin's further native
materialization reported separately. A CPU reduction compares frozen full-cap
endpoints with preserved fidelity, not a best checkpoint selected to hit a
favorable quality level; the common-CPU fidelity panel remains mandatory.
The coordinator must resolve the resource metric, percentage, per-target rule
and comparison endpoint before freeze, or omit this route entirely.

All final scores are exact channel expectations for the saved seeded circuits,
up to validated numerical error. Report that exactness rather than a spurious
Monte Carlo error bar. Training/construction randomness still affects optimizer
outcomes: publish paired replica results and any preregistered paired interval.
Do not treat 11 correlated channels as 11 independent observations or inflate
sample size by repeatedly evaluating a deterministic circuit. With five fixed
targets and two replicas, broad population claims remain limited. Any optional
interval procedure, resampling unit and multiplicity handling must be frozen;
it cannot replace failed deterministic consistency after the results arrive.

## 7. Minimum reproducibility package and no-adaptation rule

Save the benchmark Markdown and target-fixture hashes, canonical target IDs and
dimensions, exact target entries or authenticated fixture references, current
repository revision plus dirty-source hashes, dependency/lock versions,
hardware/threads/precision, all numerical source and protocol hashes, and the
complete arm/configuration/constructor/compiler identity. Preserve master seeds,
state RNG/counter snapshots, initial/source parameters, full gate metadata,
TT approximation/rank/gauge diagnostics when used, scalar lowering and compiler
source mappings, global-phase/error audits, all decision scores, accepted/rejected
branches and every final raw/deployed circuit and parameter vector.

Keep compact circuit-by-digest curves plus current full optimizer states:
Adam moments/age, adaptive moments and occurrence counts, L-BFGS secants and
pending line-search state, structural original-position maps and stream epochs.
Retain logical calls, attempted/completed physical calls, component CPU/wall,
compiler/constructor costs, failure traces, unknown-duration markers and all
checksums. A clean reproducibility command must authenticate these inputs and
reproduce numerical endpoints, not merely copy a scoreboard.

Numerical scientific failures are results. Do not retune, restart from another
seed, drop the failing/Haar case, change the threshold, compiler, objective,
memory policy, arm list, stopping rule or acceptance route after any confirmation
outcome is visible. Mechanical interruption resumes only the same authenticated
state/streams, with lost/replayed work charged. Unobserved process death blocks
CPU claims until its resource ledger is resolved. A discovered numerical bug
invalidates affected confirmation; a changed method returns to development and
requires a genuinely untouched panel. The same five states cannot be relabelled
as fresh confirmation after using their failures to repair the method.

## 8. Scaling gate

No q8/q12 performance claim follows from passing this q6 panel. Proceed only
after the primary confirmation gate or the explicitly frozen exceptional
resource gate. First run overlapping q6 dense-versus-untruncated-MPS audits with
identical physical circuits, parameters, target ordering, Pauli maps and weights.
Compare forward states/fidelities, gradients or independent directional
derivatives, online updates and full short optimizer trajectories. Use exact
zero-input preparation, not a free target-dependent MPS input.

Freeze bond caps, SVD thresholds/modes, normalization, local ordering and SWAP
handling; record forward and costate peak internal bonds, cap hits, per-split and
accumulated discarded weights, norm drift, memory, runtime and target-resolution
cost. The current zero-threshold/no-cap path supplies an exact overlap check;
truncated runs need actual discarded-weight instrumentation, not an assumed zero.
Dense target conversion remains exponential and must be charged and disclosed.
Only then try q8 before q12 with fresh bounded allocations. Prefer MPS
trajectories; an MPO optimizer needs its own demonstrated advantage and controlled
bond growth. Stop scaling if agreement or resource scaling fails.

## Choices still unresolved — therefore this remains a draft

The candidate, distinct B\*, deduplicated arm list, any constructor/component
ablation, complete validated composition, two-replica feasibility, exact numerical
and CPU budgets, compiler/protected-set integration, CPU inclusion implementation,
fidelity versus resource route, and any interval procedure must all be explicitly
settled from development evidence. Finalize these once, before confirmation
construction or scoring. This draft records no held-out outcome and supplies no
permission to launch an incomplete or selectively shortened panel.
