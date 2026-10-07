# Preregistered candidate registry

Recorded before new candidate execution, 2026-09-13 UTC. All work uses the
startup benchmark hashes in MANIFEST.json. Only the four development targets
may be used until an algorithm and confirmation protocol have been frozen.
Original proposals and preregistered amendments are retained below alongside
their chronological dispositions. A historical "pending" statement is not the
current experiment status; the latest dispositions and REPORT.md govern. The final disposition is negative:
all ten confirmation cases completed, but the selected original composition
failed the strongest-control and fixed-comparator primary gates. The chronological
registry below preserves the original proposals rather than rewriting them as
if those outcomes were known before execution.

## Common protocol

Physical input is the all-zero state. Source construction replicas 11 and 29
are authenticated 108-gate cold-final H25_lr0p7 starts. Phase-0 reproduction
precedes screening. First-stage optimizer screens use replica 11, followed by
replica 29 only for promising methods. All run endpoints are final iterates;
when a CPU cutoff is shown, use the last completed iterate within the cutoff,
never retrospectively select the highest score. Saved starting historical
structural best checkpoints are separately labeled historical comparisons.

Primary cheap screening channel: the existing exact calibrated local
Pauli-Lindblad channel (whole-window errors 0.00064/0.0051). Every final circuit
is additionally scored on all twelve normative calibrated exact conditions;
legacy executable presets, if evaluated, get separate columns/convention IDs.
The same final circuit is used across this evaluation fan-out. No new noise
strength or arbitrary ideal multi-qubit operation is permitted.

First wave refinement budgets: 100 optimizer iterations (maximum 150 joint
objective/gradient calls for line searches), 20 full coordinate sweeps, and
exact Adam at rates 0.005 and 0.02 with 150 updates; count all objective calls
and CPU, then compare both endpoint and common CPU checkpoints. This first
wave is a mechanism screen, not a fair final superiority comparison. Weighted
R32 Adam and adaptive online controls retain the existing rates/schedules and
receive 150 updates. Structural candidates are screened only after comparing
fixed-structure optimizers. All CPU and failed/tuning work is charged.

Promotion: >=0.5 percentage points target-balanced mean improvement, smaller
consistent improvement on almost every case, comparable fidelity with >=20%
gate/compute reduction, or >=2x estimator error-times-cost improvement with a
credible optimization path. Stop an unstable or clearly inferior method; no
more than four settings for a promoted family. Confirmation must compare the
strongest applicable structural Adam and Krotov methods with equivalent
budgets and endpoint selection.

## Candidates, ranked by expected information per implementation/compute cost

1. **Exact noisy L-BFGS (curvature/line search).** Hypothesis: deterministic
   quasi-Newton steps resolve late conditioning and Haar under-optimization.
   Change: SciPy L-BFGS-B on the existing exact loss/gradient, no bounds.
   Tunables: history 10 or 20; initial screen uses 10. Fixed: gtol=1e-8,
   ftol=1e-12, max line-search steps 20, final iterate. Budget: above.
   Falsify: inferior to exact Adam at common CPU on three of four targets.
   Confounders: function evaluations exceed accepted steps; optimizer basin.

2. **Exact Rotosolve (coordinate optimization).** Hypothesis: finite analytic
   coordinate minimization improves update scaling without tuning a rate.
   Change: solve a+b cos(angle)+c sin(angle) using updated density prefixes
   and cached untouched suffix observables. Fixed ordering, no shared
   parameter occurrences; fixed parameter-independent channels only. Tunable:
   sweep count; first screen 20. Budget: all local contractions and tapes.
   Falsify: accurate monotone sweeps progress too slowly at common CPU.
   Confounders: coordinate coupling, stale suffix if implementation is wrong;
   sinusoidal assumption explicitly invalid for Ballarin channel training.

3. **Adaptive exact deletion plus repair (architecture).** Hypothesis: allowing
   variable deletion count while comparing equally repaired retained branch
   avoids forced Haar loss and removes more harmless structured gates.
   Change: top-two exact single-deletion candidates plus retained control;
   same refinement oracle/budget for each; accept strict improvement only.
   Tunables: repair calls 25/50, rounds cap 15; protect initial 18 gates.
   Budget: every rejected repair and scoring pass, not only accepted path.
   Falsify: no fidelity/gate/compute Pareto advantage over five-round baseline.
   Confounders: local greedy trap and winner-selection compute.

4. **Anchor gradient control variate (variance reduction).** Hypothesis: paired
   g(theta,omega)-g(anchor,omega)+exact_gradient(anchor) cuts residual variance.
   Fixed: weighted conditional sampler, independent anchor snapshot if sampled,
   common correction maps. Tunables: anchor distance and refresh interval.
   Budget: 32 paired correction paths versus 64 ordinary weighted paths;
   charge exact snapshot and report break-even reuse. Falsify: no >=2x
   error-times-cost benefit or rapid variance degradation with anchor age.
   Confounders: frozen sampled-anchor conditional bias; no-jump subtraction
   alone does not remove conditional-stratum variance.

5. **Noise homotopy (continuation).** Hypothesis: weak-to-full noise continuation
   follows a better basin than direct full-noise optimization. Change: exact
   noisy optimizer at error scales 0,0.5,1 with final full-channel objective.
   Tunables: three stage fractions; first 25/25/50 percent. Fixed final channel.
   Budget: exactly the direct optimizer's total oracle-call allocation.
   Falsify: no paired common-cost gain, especially on Haar. Confounders:
   spending early steps on irrelevant objective; initial circuit already
   noiseless-pretrained. Scaling training noise is a surrogate, never evaluation.

6. **Group-sparse entangler optimization (sparsity/proximal).** Hypothesis:
   angle-group shrinkage steers toward compressible circuits before exact
   deletion. Change: penalty on two-qubit rotation groups during training,
   followed by physical removal and unpenalized refinement. Tunables: two
   penalty weights; fixed gate-noise placement. Budget equal to structural
   baseline including all polish. Falsify: penalty reduces ideal expressivity
   without noisy Pareto gain. Confounders: arbitrary parameter gauge,
   periodic-angle wrapping and implicit group choice. Deferred until wave one.

7. **Regularized pure-state QGT preconditioning (geometry).** Hypothesis:
   metric-aware noisy-gradient descent resolves redundant directions.
   Change: solve (G+lambda I) delta=-gradient using a noiseless-state metric
   as a surrogate. Tunables: lambda=1e-3/1e-2, line search. Budget includes
   metric construction/solve. Falsify: metric cost overwhelms gain or noisy
   gradient directions disagree with pure-state geometry. Confounders:
   rank deficiency and surrogate mismatch. Deferred pending curvature screen.

8. **Multi-channel mean/minimax training (robust control).** Hypothesis:
   optimizing average or soft-min fidelity across the fixed benchmark noise
   family improves target/channel-balanced final performance. Change: exact
   general-channel objectives, including angle derivatives if Ballarin enters
   gradient training. Tunables: mean versus soft-min temperature. Budget:
   count each constituent channel evaluation; same cost granted to baseline.
   Falsify: local-depolarization training already dominates general family
   average at common cost. Confounders: sacrifices easy profiles to worst
   channel, differing calibration, nonsmooth Ballarin pruning.

9. **Stratified randomized QMC (sampling design).** Hypothesis: scrambled Sobol
   points for first-selected-factor conditional coordinates improve weighted
   gradient accuracy. Change sampling only, same estimator for Adam/Krotov.
   Tunable scramble replicas; fixed exact branch probabilities. Budget equal
   directions and sampler CPU. Falsify: high dimensional discontinuities erase
   benefit. Confounders: independent scramble uncertainty and factor ordering.

10. **Reversible insertion/deletion (architecture search).** Hypothesis:
    recovery insertion avoids irreversible greedy deletion traps. Change:
    test a small retained-gate insertion pool with explicit noisy preparation
    and paid repair. Fixed deterministic ties and resource ledger. Tunables:
    beam width 2, one alternate structure. Falsify: added search cost buys
    no held-out Pareto gain. Confounders: broad search overfits development.

## Dispositions

Pending first-wave tests: 1-5. Deferred for independent evidence: 6-10.
Implementation-derived protocol refinements must be recorded before execution.

## Implementation freeze amendment, before first-wave execution

The curvature arm uses an explicit limited-memory BFGS two-loop recursion and
Armijo backtracking instead of SciPy's opaque L-BFGS-B state. This permits exact
resume with all secant pairs and counters retained. Settings: history 10,
Armijo coefficient 1e-4, backtracking factor 0.5, maximum 20 line-search trials,
gradient tolerance 1e-8; invalid curvature pairs skipped. Every rejected
objective/gradient evaluation is charged. This is a known algorithm adaptation.
No first-wave optimizer results existed when this change was recorded.

The homotopy arm is specifically Adam with full persistent moments and rates
0.02, linearly scaling the **average-error calibration targets** r1/r2 by
0, 0.5, 1 over 25/25/50 percent of its 150 steps. Lindblad coefficients are
recomputed from the same exact calibration map. This modifies only the
training surrogate; all final-evaluation profiles keep their fixed strengths.

Curves use the final iterate at every recorded step. For coordinate and
L-BFGS, exact local-channel scores are produced by the optimization itself;
other methods receive periodic independent exact diagnostic scores at every
ten updates. Diagnostic CPU is separately reported and included in charged
CPU comparisons. Artifact I/O is excluded from optimizer CPU but included in
invocation wall time. This asymmetry is an explicit measurement overhead,
not permission to provide either method additional training compute.

## Wave-one disposition and focused curvature protocol

All36 arms completed. L-BFGS is promoted for its compute signal: endpoint
79.671135% versus exact Adam0.02 at79.717726%, with3/4 wins; at common charged
CPU it wins4/4 by mean+0.920022pp. Mean charged CPU2.784827s versus4.277385s.
These are exploratory laptop timings and one construction seed.

Coordinate20-sweep endpoint loses1.744385pp; commonCPU difference is-0.038757pp
with2/4 wins. Retire this standalone schedule: no promotion signal. Homotopy
loses0.205711pp at endpoint, wins only0.095929pp at commonCPU and costs about7%
less; retire under the preregistered practical gate. Adaptive-online loses all
four pairs at both endpoint/commonCPU. Preserve these controls; no rate rescue.

Focused curvature freezes600 oracle calls (at most599 accepted L-BFGS steps)
and600 Adam updates, with no new hyperparameter search, on BOTH construction
replicas11/29. Comparators are exact noisy Adam0.02, weightedR32 Adam0.02, and
matched noiseless Adam0.02. All start from original common108-gate circuits.
Original150-update runs remain separate artifacts; this panel charges fresh
runs in full. Final iterates, all12 condition fan-out, and common charged-CPU
curves unchanged. Promotion requires positive paired gains or >=20% compute
advantage at comparable fidelity, with consistency on both starts. This
focused protocol is recorded before its results. No confirmation target read
for candidate selection or evaluated in either screen.

## Completed development cycles and revised dispositions

The focused 600-call local-channel L-BFGS gain is +0.436545 pp over exact Adam,
with 8/8 starts improving the training channel, but the all-eleven-noise mean
loses 0.657588 pp. The entire structured panel loses this latter comparison.
Retain L-BFGS as a component, not a broad noise-robust winner.

The 1,500-call structural L-BFGS screen improves all-noisy fidelity over its fixed
control by 3.448222 pp on seed 11 and 3.651101 pp on seed 29. However, serious
structural controls eliminate the optimizer claim: at common CPU weighted Adam
scores 89.441241% versus L-BFGS 89.269189%. Adaptive-online reaches 89.769798%
at its full cap. Promote the structural mechanism; do not promote L-BFGS over
these stronger baselines. Original and failed hypotheses remain in the ledger.

Rank-2 synthesis fails as a universal architecture: its exact-Adam endpoint mean
is 80.792490%, including Haar at 46.240992%. The structured cases are near 92%.
The Haar failure is consistent with measured TT approximation fidelity 0.464183;
structured approximation fidelities are 0.998456, 0.999574 and 1.0. No Haar case
is removed from any aggregate. A conditional rank-based initializer is only a
future hypothesis and has not been evaluated on confirmation targets.

Control variates pass the declared small training signal: refresh every 5 updates
improves the all-noisy mean 0.532805 pp over equal-path weighted Adam, costs more
CPU, and nearly matches exact Adam (+0.016473 pp). The mean is dominated by Haar;
retain this qualification. A refresh-10 follow-up is allowed by the original
protocol but requires a new frozen driver/output, not editing the completed run.

Mean-two-channel L-BFGS (depolarizing_1s_all and dephasing_1s_all equally) reaches
89.408523% with the 1,100 individual-channel-gradient cap. It improves all four
targets against local exact Adam (+3.226948 pp mean) and noiseless L-BFGS
(+2.023100 pp). Mean-eleven training reaches 88.392926% at the same cap, including
only 100 joint calls and a Haar loss. Promote mean-two for combination testing;
retain mean-eleven as a finite-budget negative, not an inferior-optimum claim.

## New composed protocols, before their scientific runs

11. **Physical algebraic compilation (architecture/compiler adaptation).**
    Hypothesis: exact identity removal and same-generator fusion reduce noisy
    exposure while preserving the endpoint unitary up to global phase. Mechanism:
    resolve the fixed scalar endpoint; remove rotations modulo 2pi within 1e-12;
    fuse only identical generators on identical ordered sites through intervening
    disjoint gates, respecting noise-enabled boundaries. Four development targets,
    seed11; source circuits are rank2 unchanged, rank2 Adam600, original108 start,
    and structural L-BFGS25 final. All source and compiled endpoints receive all
    twelve profile scores. No optimizer grid, no target omissions. Charge complete
    compilation and scoring; report gates/parameters/depth and ideal-equivalence
    error. Promote on >=0.5 pp mean gain or significant resource reduction with
    comparable fidelity. Falsify on unitary audit failure or lack of noisy Pareto
    gain. Confounders: floating-point identity tolerance, global phases, and
    changed future parameter sharing; this first screen is endpoint-only.

12. **Mean-two objective plus exact structural repair (composed candidate).**
    Hypothesis: channel-balanced training and removal of noisy gates address
    complementary deficits. Exact finite-deletion scores remove both gate and
    its noise window, then average the two unchanged training profiles. First
    mechanistic screen: original108 fourdev seed11, structural mean2 L-BFGS,
    fixed mean2 L-BFGS, and structural mean2 exact Adam, with 3,000 individual
    channel calls (1,500 mean calls), warmup150 mean calls, repairs25, shortlist2,
    max15 rounds and protected initial18. All scoring/ranking/discarded branches
    charged; final iterates and all12 evaluations. These settings are fixed before
    results. Promotion requires the stated full-noise practical gain at matched
    cost, then comparison with strong weighted/adaptive controls granted the same
    mean-two objective. Confounders: heterogeneous oracle costs, greedy deletion,
    preserved versus reset optimizer memory, and surrogate versus full11 objective.

Pure-state QGT candidate7 proceeds with its originally declared damping values
0.001/0.01, cap300, exact Adam0.02 and L-BFGS10 controls. No new tuning values.
The independently tested mechanism and full protocol live in GEOMETRY_PROTOCOL.md.

## Additional completed development dispositions

- Geometry damping0.001 versus exact Adam: all-noisy endpoint -0.139941 pp;
  common CPU -0.107855 pp (2/4 wins), with more full-run CPU. Damping0.01 loses
  0.724715 pp at common CPU (1/4 wins). Retire the frozen pure-state geometry
  schedule as a general candidate. Haar improvement does not rescue the mean.
- Compiler: sixteen ideal-equivalent endpoint pairs complete. Mean noisy gains
  are +0.197042 pp for original108, +0.827719 pp for rank2 unchanged,
  +0.122356 pp for rank2 Adam, and +0.283352 pp for structural L-BFGS. Retain as
  common postprocessing for all methods; it is not an optimizer-specific gain.
- Mean-two objective: common-CPU gains over noiseless L-BFGS, noiseless Adam
  and local exact Adam are +2.003759, +1.975654 and +3.093587 pp, all four wins.
  Promote to matched structural testing; no confirmation claim yet.

13. **Entropic risk objective (risk-sensitive control adaptation, preregistered).**
    Use tau\*log(mean(exp(loss_j/tau))) over the two fixed training profiles,
    with tau 0.02/0.1, stable log-sum-exp and exact softmax-weighted gradients.
    Original108 fourdev seed11; each arm receives 1,100 individual channel
    gradients (550 joint evaluations). L-BFGS and exact Adam receive each tau;
    both also receive equal-mean controls. Adam rate0.02; unchanged L-BFGS10.
    Compare all11 mean and worst benchmark channel under common CPU and cap.
    Promote a primary full-mean gain against the strongest matched control, or
    at least +0.5 pp worst-channel mean with <=0.1 pp all11-mean sacrifice.
    Retire otherwise; no target omission or extra temperature grid. Expected
    failure: worst-channel emphasis sacrifices easier profiles or approaches
    nonsmooth conditioning. This is known entropic-risk optimization, adapted
    to circuit fidelity; no novelty claim. Full implementation protocol pending.

14. **Target-conditioned rank-2 initializer (resource policy, before execution).**
    Compute the fixed rank-2 TT approximation for every development target.
    If ideal squared overlap is at least 0.99, use its physically lowered
    circuit; otherwise use the legacy H25 construction. The same algebraic
    compiler is granted to both source choices and every trained endpoint.
    Family labels do not select a branch; Haar remains in the panel. Charge
    TT/overlap/compile work even for fallback. Fresh legacy construction costs
    must be authenticated before freeze; archived CPU is never inferred.
    First six-arm screen: both initializer choices with unchanged, mean2
    L-BFGS550 and mean2 exact Adam550 (rate0.02); all12 final evaluations.
    Compare complete preparation+training CPU and separately labeled training
    CPU, lowered gates and mean fidelity. An initializer unavailable at an
    early common budget is explicitly unavailable. Promote comparable fidelity
    with substantial gate/construction-CPU advantage, or the usual fidelity
    gate; otherwise retire. Strong stochastic optimizers receive the same
    chosen initializer opportunity before a general superiority claim.
    Expected failure: ideal low-rank overlap does not imply a good noisy
    realization or an easily repaired landscape. This is a conditional
    composition of known TT synthesis and legacy variational construction.

Mean-two structural composition result: all11 averages structural L-BFGS
91.571432%, structural exact Adam90.884894%, fixed L-BFGS89.493593%. The
+0.686537 pp gain over structural Adam and +2.077839 pp over fixed L-BFGS
passes the mechanistic screen. Strong same-objective stochastic controls are
now complete: at common CPU structural weighted Adam scores90.802145% versus
L-BFGS91.264849%, leaving +0.462704 pp and only2/4 wins. Full-cap L-BFGS gains
0.769286 pp with4/4 wins but costs more CPU. Keep the structural/objective
composition promoted; the optimizer-specific practical gate remains unresolved.
No confirmation target was evaluated.

Entropic risk candidate13 is retired for the frozen temperature/budget screen.
Temperature0.1 L-BFGS has a small consistent mean gain (+0.107425 pp,4/4) and
worst-condition gain (+0.121830 pp) over mean-two L-BFGS, below the predeclared
0.5 pp primary/tail threshold. Temperature0.02 L-BFGS loses1.024246 pp mean;
both Adam temperatures also fail their criteria. No temperature sweep follows.
This does not falsify all risk-sensitive optimization or full-family minimax.

The conditional initializer's prerequisite now passes: all four fresh legacy
construction reproductions match scalar circuits, parameters and states exactly.
Measured construction CPU totals147.384197 seconds, with full per-target timing.

Candidate14's complete screen passes its initializer/resource gate. Conditional
L-BFGS averages91.808357%, versus legacy89.651145% (+2.157213 pp); conditional
Adam91.422781%, versus legacy89.630829% (+1.791952 pp). All three structured
targets improve; Haar follows the legacy fallback and is numerically identical.
Structured total CPU falls from roughly60–62s to12–15s. Common total CPU is
explicitly unavailable for legacy on those three targets because its initializer
is not ready at the shorter cutoff. Common refinement CPU remains available.
Promote the target-only initializer policy, granted to all serious controls.

15. **Complete conditional structural composition (before execution).**
    Combine the accepted initializer/resource rule with mean-two exact finite
    deletion and matched repair. All gates become eligible under a common empty
    protected set; this is a declared new policy for compiled physical starts.
    Compile initially and after final training only. Raw accepted checkpoints
    remain deployable during training; final compilation is available only after
    its measured cost. Four primary structural arms: exact L-BFGS, exact Adam,
    weighted Adam and adaptive online. Add fixed mean-two L-BFGS, fixed noiseless
    Adam and unchanged references. Keep all four development targets, seed11,
    with the previously developed exact3000/stochastic1500 caps. Freeze separate
    four-primary and all-six-trained common-CPU panels. Promote an optimizer only
    for at least0.5 pp primary common-CPU gain with4/4 signs, or the explicit
    all-target fidelity/resource rule in COMPOSITION_PROTOCOL.md. Otherwise keep
    the component resource result without an optimizer claim. No target-specific
    threshold, repair or rate rescue. This is a composition of known ideas.

The full-family objective receives one bounded convergence diagnostic, not a
new methodological-family entry: mean-two/mean-eleven times L-BFGS/Adam at6050
constituent calls (3025 versus550 joint calls), original108 fourdev seed11.
LONG_ENSEMBLE_PROTOCOL.md fixes its mean/common-CPU and resource criteria before
execution. The earlier100-joint-call failure motivates this sole longer-budget
test; no additional objective weighting or hyperparameter grid is introduced.

The 6,050-call convergence diagnostic is complete and retires that mean-eleven
recipe. At common CPU its L-BFGS89.337888% trails the strongest mean-two control,
Adam89.497042%, by0.159154 pp, with only1/4 wins. Mean-eleven Adam88.853517%
loses0.643525 pp. Full-cap means are mean-eleven L-BFGS89.337888%, mean-eleven
Adam88.939780%, mean-two L-BFGS89.489328%, and mean-two Adam89.503145%.
Mean-eleven L-BFGS has CPU ratios0.249–0.768 against full-cap mean-two Adam,
but its0.165257 pp mean deficit misses the predeclared0.1 pp resource tolerance.
Retain the two-channel objective; no additional convergence budget is tested.

Candidate15 passes its primary four-structural common-total-CPU development
gate: L-BFGS92.612204%, weighted Adam91.891505%, adaptive online91.722731%,
exact Adam91.345127%. The gain over the strongest aggregate control is
0.720699 pp, with all four targets improving (0.125020–1.890321 pp).
Full-cap L-BFGS92.935685% exceeds weighted Adam by1.044180 pp, also4/4.
The stricter common cutoff across all six trained methods gives L-BFGS92.319196%
and fixed mean-two L-BFGS91.841250%, a0.477946 pp gain with4/4 wins. Preserve
this sensitivity qualification. The objective convergence and endpoint compiler
screens are now resolved. Candidate15 was retained after the compiled alternative
failed its strongest-control gate. The complete original parent subsequently
finished confirmation and failed its primary acceptance gate, as detailed in
the final disposition below. This remains a known-method composition.

The first MPS overlap gate fails all eight endpoint audits: parameter differences
after three fresh Adam updates are1.25e-9–3.95e-9, above its fixed1e-10 tolerance.
Most gradient/path-fidelity differences remain around floating-point roundoff;
one later TFIM structural gradient difference reaches1.40e-10. The recorded
failure is retained while diagnosing update sensitivity. It does not authorize
scaling and must not be reclassified by silently loosening its tolerance.

16. **Known-input eigenstate compiler (before execution).**
    Starting from physical zero, greedily remove a supported scalar rotation
    only when its action on the retained ideal prefix is a global phase within
    direct vector residual1e-12. Preserve every surviving gate field and noise
    flag and compact scalar indices. Independently verify the complete ideal
    preparation and accumulated phase with Qiskit. This is known-input compiler
    specialization, not arbitrary-unitary equivalence or a new optimizer.
    Freeze24pairs: both compiled initializers and all four complete structural
    endpoints on all four development targets, replica11. Evaluate source and
    compiled circuits under all12 unchanged conditions (576 fresh profile calls).
    Retire broad adoption if the complete-panel mean worsens beyond1e-10;
    any negative target/source-label mean blocks uniform-benefit claims.
    Fewer gates with neutral/positive fidelity supports a shared compiler
    resource result. No threshold sweep or noisy-score selection. Because the
    parameter domain changes, adoption requires a new complete development
    composition with the same compiler opportunity for every control before
    confirmation. Full mechanism, provenance, error allowance, costs and frozen
    panel are in STATE_PREPARATION_COMPILATION_PROTOCOL.md.

The COMMON_CONTROL_PROTOCOL.md sidecar will score fixed/noiseless controls at
candidate15's already frozen four-structural CPU cutoff. It freezes selected
points before new evaluations, reuses the original four structural scores and
tests the strongest aggregate control across all six trained methods. This
development diagnostic cannot alter the original cutoff or promotion rule.

That all-control audit is complete: weighted Adam91.891505% remains strongest,
above fixed mean-two L-BFGS91.841917%, adaptive91.722731%, exact Adam91.345127%,
noiseless Adam90.400908% and unchanged87.423780%. Candidate92.612204% therefore
keeps its0.720699 pp/4-of-4 development gate against the strongest aggregate
control. Against fixed L-BFGS its gain is0.770287 pp with4/4 wins. Adaptive online
retains a small0.013897 pp TFIM advantage; do not claim all-control/all-target
dominance. The original four structural points and every old artifact remain
unchanged; only fixed-control same-time scores were newly evaluated.

Candidate16's paired endpoint diagnostic passes its shared compiler/resource
screen. All24 pairs finish with576 fresh profile calls and48 independent ideal
preparations;61 RZ gates/parameters disappear across15 pairs, nine stay unchanged,
and no two-qubit gates disappear. The overall gain is0.036813 pp; every target
and source-label mean is positive, with no negative pair mean. Structural
L-BFGS finals gain only0.008195 pp. This is a small compilation result, not a
reason to reinterpret the previous optimizer comparison.

**Focused compiled composition, before execution:** test whether eliminating
known zero-input phase coordinates before optimization makes the unchanged
15-round structural budget more effective. Apply the same state-preparation
compiler to the shared generic-compiled conditional initializer and, after
training and generic final compilation, to every final arm. Preserve all raw
intermediate checkpoints; each final compiler becomes available only after its
measured cost. Use all four development targets, seed11, all seven roles,
exact3000/stochastic1500 caps and unchanged optimizer/repair/protection settings.
Use fresh compiled_composition_training randomness. At the four-structural
cutoff score all six trained controls and unchanged, and retain the all-six
minimum sensitivity and full caps. Candidate L-BFGS must gain at least0.5 pp
with4/4 target signs against weighted Adam and the strongest aggregate control;
otherwise retire this new composition and retain the prior candidate. No
threshold, round-budget or rate rescue. Confirmation remains untouched until
this final declared method choice and lazy-construction equivalence are settled.

The lazy-constructor reproduction is now complete on all four development
targets. Scalar circuits and parameters reproduce the prior conditional
endpoints bit for bit. Gaussian, TFIM critical and random MPS select rank two
and never invoke the legacy constructor; their measured initializer CPU is
0.042004, 0.041078 and 0.040402 seconds. Haar invokes the unchanged legacy
fallback, including the preceding rank-two proposal, for 35.830342 seconds.
These measurements establish real conditional execution, not a new fidelity
experiment, and do not replace the earlier studies' recorded procedure costs.

The compiled-composition development run is complete and **retired** under its
frozen strongest-control rule. Candidate92.765744% exceeds weighted Adam92.211126%
by0.554618 pp with4/4 target wins, but the strongest aggregate control is adaptive
online92.387971%. The candidate gains only0.377773 pp against that control, with2/4
wins (MPS−0.195890 pp; TFIM−0.021060 pp). The raw candidate increase does not rescue
that failure. Retain the original recipe; no additional setting is tested. No outcome has selected its controls, endpoint rule or settings. The
original candidate15 comparison remains +0.720699 pp and four target-mean
wins against the strongest aggregate control, weighted Adam; 8 of its 44
individual noisy target/condition comparisons lose. Its smaller all-six-cutoff
gain remains below 0.5 pp. Both qualifications carry forward to reporting.

The pre-confirmation recipe-selection record was frozen at 04:34 UTC before
any compiled case completed. Retain the original unless the new composition
passes its complete primary gate against fixed comparator B\* (structural
weighted Adam) and the strongest aggregate control, and exceeds the
original candidate's 92.6122043083% primary noisy mean by more than 1e-10 in
fidelity units. Ties retain the original. Full-cap or all-six-cutoff scores do
not rescue a primary failure. This is a choice between development recipes
with fresh training streams and measured CPU endpoints, not an isolated
compiler treatment-effect estimate. The authoritative record is
`validation/PRE_CONFIRMATION_RECIPE_SELECTION.json` in the output root.

The separate MPS physical follow-up passes all 24 saved steps under its own
frozen sensitivity criterion. It leaves the original eight parameter-tolerance
failures unchanged. The 2688-byte maximum measures aggregate tensor storage
for one observed MPS state, not retained tapes or process peak memory. This is
a q6 backend prerequisite and grants no automatic q8/q12 scientific allocation.

The recipe decision is now resolved: ORIGINAL confirmation began at about04:49 UTC
only after its complete DESIGN and parent freeze. Five held-out identities and
paired replicas11/29 receive all seven roles and twelve exact conditions. The
configuration hash is `7ca2de674a9e13e863dfaa0c1b4312dee90a1d7cb437fa1efc9a3bad302d9068`;
the DESIGN canonical content checksum is
`203ee474a2500b752c0f46063a32838e51aa73dd2bd1dad48f3797916e9ca976`.
Do not use partial outcomes for selection or omit any failing target.

The q8 scaling design is ineligible: the unchanged canonical fixture contains
only q6 and q12 target entries. No substitute q8 target may be generated. Any
q12 MPS extension requires a new protocol and a complete passing q6 confirmation;
no scaling experiment is authorized by the backend overlap checks alone.

## Final disposition after complete untouched confirmation

Candidate 15's original recipe completed all ten target/replica cases and all
seven roles. At the frozen primary common CPU cutoff, its eleven-condition mean
is 88.534223%, versus 88.095145% for structural weighted Adam and 88.209294% for
the strongest aggregate control, structural adaptive online. The gains are
0.439079 and 0.324929 percentage points, each with only three of five target
wins. They miss 0.5 points by 0.060921 and 0.175071 respectively. TFIM para and
MPS bond 3 lose to both. The positive aggregate is concentrated in Haar, with
both TFIM and MPS family means below adaptive online.
**Retire the selected recipe's promotion claim.**

The full-cap candidate mean of 88.830668% gains 0.600137 points over adaptive
online with five wins, but still loses MPS to weighted Adam. Neither full cap,
an alternate cutoff, the stronger development panel nor a subset of channels
rescues the primary failure. There is no newly selected recipe, held-out
retuning, exceptional resource override or q12 scientific run. The contribution
is a decisive negative comparison plus reusable objective, initialization,
compilation, estimator and exact-oracle components; it is not a new optimizer
or external state-of-the-art result.

All target/channel cells remain published, including the difficult Haar 3
replica 29 constructor outcome. Exact density evaluation removes trajectory
sampling error at each saved point, not finite-target uncertainty. The public
CLI cache resume reproduces all ten completed cases with 12,080 files unchanged.
Fresh CPU timing may select a neighboring chronological checkpoint even with
identical optimizer trajectories; the immutable receipt proof and fresh-run
bitwise equivalence are distinct claims.

The read-only structural history reduction finds matching signs for the
training mean-two and full eleven-channel advantages in all 60 paired
case/panel/control comparisons against weighted and adaptive methods. It does
not support objective-surrogate reversal as the explanation for these losses.
TFIM para primary checkpoints precede terminal polishing; MPS bond 3 stops
L-BFGS structure when the repaired retained branch wins at round 11, whereas
weighted Adam accepts 15 deletions and retains fewer two-qubit gates. These
observations motivate a separately frozen development-only repair-allocation
probe, not a revised interpretation of confirmation.

The retained-memory MPS diagnostic passes numerical agreement on eight older
development endpoints but is about 13.25 times slower than dense gradients at
q6; it does not measure the complete selected stochastic batch or establish a
full-method memory advantage. The first parameter-tolerance failure stays failed.
The q8 fixture entries are absent, and no q12 allocation follows.

All nine fixed q6 identities are now used. Extra seeds can quantify variability
but cannot create untouched target identities. Any changed method remains
fixed-benchmark development until a separately authorized untouched panel is
available. Complete signed outcomes and scope are in
[REPORT.md](REPORT.md), [CONFIRMATION_DIAGNOSTICS.md](CONFIRMATION_DIAGNOSTICS.md)
and the [confirmation SUMMARY](../../output/overnight-state-preparation-20260913-0108/confirmation-best-candidate/SUMMARY.json).
