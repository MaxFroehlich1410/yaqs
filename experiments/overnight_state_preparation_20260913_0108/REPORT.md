# Overnight noise-aware state preparation: final scientific report

The scientific search and the complete q6 confirmation are finished. All ten
confirmation cases are retained. Reporting, provenance and archive work do not
change the frozen algorithm or create new scientific results.

## 1. Executive conclusion

**No candidate passed the untouched confirmation gate.** The strongest candidate
selected during development combines conditional physical initialization,
a two-channel noisy objective, exact gate-deletion ranking, symmetric repair,
L-BFGS and generic physical compilation. At the predeclared common-CPU endpoint
it achieves **88.534223%** target-balanced noisy fidelity, against **88.209294%**
for the strongest fair aggregate control, structural adaptive online. The gain
is **0.324929 percentage points**, below the required 0.5 points by **0.175071
points**, with only **three of five target wins**. Against the fixed development
comparator, structural weighted Adam, the gain is 0.439079 points, short by
0.060921 points, again with three wins. TFIM para and random MPS bond 3 lose to
both required controls.

Full-cap results are better: 88.830668% for the candidate and 88.230532% for
adaptive online, a 0.600137-point gain with five target wins. They cannot rescue
the failed primary gate. Even at full cap, weighted Adam still wins random MPS
bond 3. The earlier cutoff across all six trained arms narrows the candidate's
lead over adaptive online to 0.117057 points with only two target wins.

The useful outcome is a decisive negative result with reusable components:
noise-aware architecture reduction, a broadly useful conditional initializer,
a fair mean-objective trajectory sampler, exact channel/deletion oracles and
complete resumable experiment custody. This is a **candidate method**, not a
confirmed new repository baseline, a Krotov advantage or state-of-the-art result.
No external method was faithfully benchmarked across the complete fixed task.
No q12 scaling run is justified by this result or the measured MPS resources.

## 2. Best method found: mathematics and algorithm

The selected `original` recipe prepares the physical zero state. For profile
`c`, the exact benchmark fidelity is:

```math
rho_0 = |0...0><0...0|,
E_{c,j}(rho) = N_{c,j}(U_j(theta) rho U_j(theta)^dagger),
F_c(C,theta) = <psi| E_{c,L} ... E_{c,1}(rho_0) |psi>.
```

Every local unitary is lowered into supported scalar one- and two-qubit
rotations before applying the calibrated gate-noise semantics. The training
loss is the equal mean of `depolarizing_1s_all` and `dephasing_1s_all` infidelities:

```math
L(C,theta) = 1 - [F_depolarizing_1s_all(C,theta)
                  + F_dephasing_1s_all(C,theta)] / 2.
```

The continuous gradient is an exact density adjoint. The two training profiles
have parameter-independent Pauli probabilities. Final scoring always uses all
twelve unchanged conditions, including angle-dependent Ballarin materialization.
The general ensemble implementation differentiates Ballarin probabilities when
that channel is trained; the selected recipe does not train it.

The complete procedure is:

1. Form a deterministic rank-two TT approximation of the supplied classical
   target amplitudes. If its ideal squared overlap is at least 0.99, synthesize
   and lower its physical preparation. Otherwise pay for the entire probe and
   execute the unchanged legacy H25/rate-0.7 constructor freshly. Target-family
   labels and noisy fidelity do not choose the branch. The legacy constructor
   uses its original MPS backend, seed root 2026090520, 570 updates and final
   78 one-qubit plus 30 two-qubit gates before compilation.
2. Apply the generic endpoint compiler. It resolves scalar angles, removes
   periodic identities and fuses matching generators only across allowed
   disjoint operations. Ideal equivalence is audited independently. Removing
   gates changes physical noise exposure; noisy equivalence is not assumed.
3. Warm up for 150 joint mean-two objective calls. L-BFGS uses ten secant pairs,
   two-loop inverse-Hessian recursion, curvature safeguards and Armijo
   backtracking. Failed line-search calls are charged.
4. For at most 15 rounds, compute exact single-gate deletion fidelities. If
   `rho_{c,j-1}` is the prefix density and `A_{c,j}` is the suffix target effect,
   deleting the entire gate-plus-noise layer has score
   `Tr(A_{c,j} rho_{c,j-1})`. Average the two channel scores and shortlist two
   deletions. No gate is protected at this noisy refinement stage.
5. Repair the unchanged incumbent and each shortlisted deletion for at most
   25 calls apiece with fresh exact-stage optimizer memory. Accept a deletion
   only when its repaired true mean-two fidelity exceeds both the original
   incumbent and the equally repaired retained circuit. Charge rejected
   repairs, ranking and scoring; use the remaining budget to polish.
6. Retain the final iterate and apply the same generic final compiler granted
   to every control. The exact branch uses a 3,000 individual-channel-call
   budget, corresponding to at most 1,500 joint mean-two calls with the frozen
   ranking/score reservations. Raw chronological checkpoints remain raw;
   final compilation is available only after its measured cost.

The algorithm and all numerical constants are explained more fully in
[METHOD.md](METHOD.md), with code links, the legacy construction schedule,
compiler rules, optimizer state and cost semantics. The selected recipe does
**not** include the separately tested known-input state compiler.

## 3. Exact difference from existing methods

The legacy layerwise noiseless constructor remains a fallback; it is not
replaced by a free target state. The new conditional branch saves its execution
on eligible structured targets while still preparing zero physically. The
noisy objective changes from one local-depolarizing channel to an equal mean of
two fixed channels. This addresses the development observation that optimizing
a single channel improved depolarization while harming dephasing transfer.

Architecture changes use exact removal of a complete physical gate and its
noise window, rather than magnitude quotas, first-order saliency or setting an
angle to zero while leaving its noise in place. Symmetric repair gives retention
and deletion the same optimization opportunity; it does not force a prescribed
number of deletions. L-BFGS replaces Adam or online updates only inside that
shared structural framework. A generic compiler is shared across all roles.

Weighted Adam and adaptive online receive the same mean-two objective through
an exact mixture of two weighted Pauli ensembles: 32 paths per channel, 64 in
total, with a deterministic no-selected-factor stratum and conditional weighted
remainder. Their seeds, estimator opportunity, initializer, shortlist, repair
lengths and compiler opportunity match the candidate's corresponding policies.
No fixed-CRN empirical objective substitutes for the true channel expectation.

These are changes to an entire preparation procedure. Improvements from the
objective, initialization or architecture are not evidence that a new optimizer
was invented. An upgraded Adam that becomes identical to a proposed combination
is a valid shared baseline contribution; an impossible self-comparison would
not establish an optimizer-specific claim.

## 4. Literature provenance and novelty status

All implemented mechanisms are known or adapted combinations. The independent
catalogues contain more than eight materially different ideas spanning
optimization, quantum control, Monte Carlo, geometry and circuit synthesis.
[LITERATURE.md](LITERATURE.md) records exact adaptations and original sources;
[CANDIDATES.md](CANDIDATES.md) preserves hypotheses, budgets and dispositions.

| Mechanism                       | Primary prior source                                                                                                            | Adaptation in this study                                                         |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Limited-memory curvature        | [Nocedal, 1980](https://doi.org/10.1090/S0025-5718-1980-0572855-7); [Liu and Nocedal, 1989](https://doi.org/10.1007/BF01589116) | Exact noisy infidelity, Armijo search, fully resumable history                   |
| Ensemble quantum control        | [Chen et al.](https://arxiv.org/abs/1308.1454)                                                                                  | Exact finite means over fixed benchmark channels                                 |
| Structural circuit search       | [Bilkis et al., VAns](https://arxiv.org/abs/2103.06712)                                                                         | Deletion-only exact gate-plus-noise scoring and symmetric repair; not full VAns  |
| Weighted trajectories           | [Abdelhafez, Schuster and Koch](https://link.aps.org/accepted/10.1103/PhysRevA.99.052327)                                       | Discrete calibrated Pauli-factor stratification and an equal two-channel mixture |
| Snapshot control variates       | [Johnson and Zhang](https://proceedings.neurips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html)              | Exact-density anchor plus paired weighted trajectory differences inside Adam     |
| Pure-state geometry             | [Stokes et al.](https://quantum-journal.org/papers/q-2020-05-25-269/)                                                           | Damped pure-state metric applied to an exact noisy gradient; not mixed-state QNG |
| Sequential physical preparation | [Schön et al.](https://arxiv.org/abs/quant-ph/0501096); [Ran](https://arxiv.org/abs/1908.07958)                                 | TT-SVD, physical isometry lowering and a fixed overlap fallback rule             |
| Known-input compilation         | [Häner, Hoefler and Troyer](https://arxiv.org/abs/1810.00375)                                                                   | Dense zero-input eigenstate simplification with an independent ideal-state audit |

The coordinate, entropic-risk, compiler and cross-field alternatives have their
own original citations in the literature synthesis. Those papers establish
prior mechanisms, not the fidelity gains measured here. No faithful external
SOTA comparison, hardware experiment or scalability claim follows from naming
those methods.

## 5. Development results and recipe selection

The four q6 development identities were Gaussian, TFIM critical, Haar 1 and
random MPS bond 2, normally replica 11, with a compact paired replica-29 check
where preregistered. Every mean below weights the eleven noisy conditions
equally, then replicas within target, then targets. Budgets differ between rows;
this table is a research sequence, not an equal-compute ranking.

| Development comparison                                |            Candidate/setting |                  Comparator | Result and disposition                                                      |
| ----------------------------------------------------- | ---------------------------: | --------------------------: | --------------------------------------------------------------------------- |
| 600-call local optimization, replicas 11/29           |  L-BFGS 85.024223% all-noisy |       Exact Adam 85.681811% | −0.657588 pp despite +0.436545 pp on the training channel; transfer deficit |
| 1,500-call local structure, replica 11                | Structural L-BFGS 89.169670% |     Fixed L-BFGS 85.721448% | +3.448222 pp, 4/4 wins; architecture signal                                 |
| Strong local structural controls, common CPU          |            L-BFGS 89.269189% |    Weighted Adam 89.441241% | −0.172053 pp; optimizer claim fails                                         |
| Mean-two fixed 108 gates, 1,100 constituent gradients |            L-BFGS 89.408523% | Local exact Adam 86.181575% | +3.226948 pp, 4/4 wins; promote objective mechanism                         |
| Mean-two structure, common CPU                        |            L-BFGS 91.264849% |    Weighted Adam 90.802145% | +0.462704 pp, only 2/4 wins; insufficient strongest-control gate            |
| Conditional complete recipe, primary common CPU       |            L-BFGS 92.612204% |    Weighted Adam 91.891505% | +0.720699 pp, 4/4 wins; selected candidate                                  |
| Known-input compiler added before/after training      |            L-BFGS 92.765744% |  Adaptive online 92.387971% | +0.377773 pp, only 2/4 wins; rejected                                       |

Local structural compression cut 30 two-qubit gates to 15–26 and improved all
four targets against fixed L-BFGS. A second construction seed repeated that
architecture advantage, but neither repetition replaced serious structural
weighted/adaptive controls. Older structural Adam/adaptive artifacts are
explicitly unmatched historical references because their budgets, estimator
opportunity and checkpoint rules differ.

Conditional initialization plus mean-two L-BFGS reached 91.808357%, versus
89.651145% with equally compiled legacy initialization. Adam also benefited
by 1.791952 points. Three structured targets improved while Haar used the
bit-identical fallback. The actual lazy implementation reproduced all four
development circuits and parameter vectors exactly; it skipped legacy work on
Gaussian, TFIM and MPS, with initializer CPU 0.042004, 0.041078 and 0.040402
seconds. Haar cost 35.830342 seconds including its rejected TT probe. These fresh
measurements do not retroactively replace earlier attributed construction costs.

The original complete recipe's all-control sidecar held its structural cutoff
fixed and scored all seven roles. Weighted Adam remained the strongest aggregate
control, above fixed L-BFGS at 91.841917%. Adaptive online retained a small TFIM
advantage, and 8 of 44 target/noisy-condition pairs lost against weighted Adam.
The candidate's gain was therefore an average improvement, not universal dominance.
At the earlier all-six cutoff its lead over fixed L-BFGS was only 0.477946 points.

The pre-confirmation selection record was saved before any compiled case
finished. The new known-input-compiled recipe needed both its strongest-control
gate and a higher primary candidate mean. It beat weighted Adam by 0.554618
points with four wins but lost MPS and TFIM against adaptive online. Its higher
raw candidate fidelity did not rescue the failed gate. ORIGINAL was retained;
this is recipe selection across fresh streams and CPU availability, not an
isolated causal estimate of compiler benefit.

## 6. Untouched q6 confirmation results

Before any confirmation construction, the complete design fixed the five
remaining identities, replicas 11 and 29, all seven roles, construction and
training seeds, compiler schedule, budgets, trajectory counts, endpoint rules
and acceptance test. Ten cases and all 840 case/role/channel endpoint cells
completed; retaining primary, all-six sensitivity and full-cap strata gives
2,520 published cells. All are finite. An independent read-only audit verified
12,080 artifact files unchanged and recomputed the complete analysis exactly.
No confirmation result selected a setting, target subset or alternate recipe.
The later central reconciliation matches all 2,520 fidelity/count/evaluation-hash
cells exactly; 21 aggregated points agree to at most 1.11e-16 from floating-point
summation order. [Reconciliation receipt](../../output/overnight-state-preparation-20260913-0108/validation/CENTRAL_CONFIRMATION_RECONCILIATION.json).

The primary rule required at least 0.5 percentage points mean gain and a
strictly positive target-mean gain on all five identities against **both** the
fixed weighted-Adam comparator and the strongest aggregate noncandidate role.
It failed both comparisons. The following target means average both replicas:

| Held-out target  | Candidate noisy % | Change vs weighted Adam, pp | Change vs adaptive online, pp |
| ---------------- | ----------------: | --------------------------: | ----------------------------: |
| tfim_ferro       |         94.249159 |                   +1.831105 |                     +0.359466 |
| tfim_para        |         95.581861 |                   -0.804953 |                     -0.712797 |
| haar_random_2    |         83.496938 |                   +1.113170 |                     +1.107243 |
| haar_random_3    |         73.530761 |                   +0.480171 |                     +0.994531 |
| random_mps_bond3 |         95.812397 |                   -0.424100 |                     -0.123797 |

The candidate's median target fidelity is 94.249159%; its worst target is Haar 3
at 73.530761%. Its worst target-family mean is Haar at 78.513850%, versus
77.717179% for weighted Adam and 77.462963% for adaptive online. TFIM-family
mean is 94.915510%, below adaptive online's 95.092176%; MPS also favors both
stochastic controls. Against that strongest control the candidate gains
1.050887 points on the Haar-family mean, but loses 0.176666 on TFIM and
0.123797 on MPS. Thus the aggregate advantage is concentrated in Haar even
though three individual target means improve. Both non-Haar family means lose;
this is not broad family improvement. All five identities remain in every
aggregate.

Every target-balanced condition mean favors the candidate, but this does not
imply consistent target improvements. After averaging paired replicas, 21 of
55 target/noisy-condition cells lose to weighted Adam and 14 lose to adaptive
online. The largest losses are TFIM para under `dephasing_1s2s_all`: 1.781250
and 2.032263 points respectively.

| Fixed evaluation condition | Candidate % | Weighted Adam % | Adaptive online % |
| -------------------------- | ----------: | --------------: | ----------------: |
| ballarin_coupled           |   92.601954 |       92.238449 |         92.449881 |
| dephasing_1s2s_all         |   86.549388 |       86.075146 |         86.280846 |
| dephasing_1s_1q            |   92.948414 |       92.524118 |         92.601391 |
| dephasing_1s_2q            |   88.972386 |       88.398164 |         88.752944 |
| dephasing_1s_all           |   86.548565 |       86.016956 |         86.287902 |
| dephasing_2s_2q            |   88.996595 |       88.614703 |         88.758677 |
| depolarizing_1s2s_all      |   84.676185 |       84.316923 |         84.174970 |
| depolarizing_1s_1q         |   92.177479 |       91.867117 |         91.622667 |
| depolarizing_1s_2q         |   87.903191 |       87.362785 |         87.637617 |
| depolarizing_1s_all        |   84.802367 |       84.401933 |         84.297252 |
| depolarizing_2s_2q         |   87.699934 |       87.230296 |         87.438093 |
| noiseless                  |   95.630375 |       95.175711 |         95.346092 |

Noiseless is the same selected physical circuit evaluated without noise and is
excluded from the eleven-noisy mean. These are exact channel expectations, so
there is no Monte Carlo standard error or sampled-confidence-interval claim.
Exactness does not remove uncertainty about generalization from five target
identities and two construction/training replicas. The required deterministic
consistency was not achieved.

Haar 3 replica 29 is visibly difficult for every method. Its unchanged circuit
has 54.066884% ideal and 47.192516% noisy fidelity; candidate primary fidelity
is 73.290413% ideal and 65.772773% noisy, versus weighted Adam's 73.953837% and
66.092517%. Replica 11 is substantially better for all methods. This demonstrates
strong inherited construction/optimization-basin sensitivity; it is not a
missing or nonfinite run, and no seed is discarded or rerun for a favorable
score. These observations diagnose stored outputs without establishing a unique
cause or changing the recipe.

Authoritative results are
[confirmation SUMMARY](../../output/overnight-state-preparation-20260913-0108/confirmation-best-candidate/SUMMARY.json),
[all 2,520 cells](../../output/overnight-state-preparation-20260913-0108/confirmation-best-candidate/SCOREBOARD.csv)
and [strict authentication](../../output/overnight-state-preparation-20260913-0108/validation/CONFIRMATION_AUTHENTICATION.json).

## 7. Comparison with all strong baselines

All structural roles receive the same physical conditional initializer,
mean-two objective, exact deletion ranking, repair shortlist and final compiler.
Exact Adam uses rate 0.02. Weighted Adam uses rate 0.02 and the 32-per-channel
estimator. Adaptive online uses its declared first 25 online updates at rate
0.7, then adaptive scaling at 0.02 with the same weighted mixture. Fixed mean-two
L-BFGS isolates structural changes; fixed noiseless Adam isolates noisy training;
unchanged isolates all refinement.

| Method                        | Primary noisy % | Full-cap noisy % | All-six-cutoff noisy % | Primary noiseless % |
| ----------------------------- | --------------: | ---------------: | ---------------------: | ------------------: |
| Structural L-BFGS (candidate) |       88.534223 |        88.830668 |              87.857135 |           95.630375 |
| Structural weighted Adam      |       88.095145 |        88.095029 |              87.293365 |           95.175711 |
| Structural adaptive online    |       88.209294 |        88.230532 |              87.740078 |           95.346092 |
| Structural exact Adam         |       86.966933 |        87.378780 |              86.549863 |           94.384515 |
| Fixed mean-two L-BFGS         |       87.689991 |        87.692425 |              87.681941 |           95.889920 |
| Fixed noiseless Adam          |       86.470996 |        86.470996 |              86.470996 |           95.990104 |
| Unchanged initializer         |       77.633670 |        77.633670 |              77.633670 |           85.021395 |

At the primary cutoff the candidate beats fixed mean-two L-BFGS by 0.844232
points, exact structural Adam by 1.567291 points and noiseless Adam by 2.063228
points, with five target wins in each comparison. Those wins do not substitute
for beating the strongest fair controls. Adaptive online, not exact Adam, is
the strongest aggregate confirmation comparator.

Full-cap and primary endpoints differ because later structural work, polishing
and final compilation can be unavailable at the earlier cutoff. For TFIM para,
the candidate primary checkpoints follow structural rounds 14 and 13; its full
run completes round 15, then 165 polishing gradient calls and three one-qubit
fusions. The candidate gains about 1.10 points between primary and full cap.
The saved data do not isolate a compiler-only fidelity effect, because raw-final
all-twelve scoring was not recorded. Applying final compilation retroactively
would change the endpoint rule and is forbidden.

For MPS bond 3, L-BFGS stops structural search at round 11 when the repaired
retained branch wins, then spends 481 calls on polishing. Weighted Adam accepts
all 15 structural rounds and retains fewer two-qubit gates. The full-cap
candidate still loses MPS to weighted Adam despite beating adaptive online
there. These histories motivate studying repair and stopping policies, without
establishing that a specific replacement would succeed.

The read-only history reduction finds the same sign for candidate advantage in
the training mean-two objective and the full eleven-channel mean in **all 60**
paired case/panel/control comparisons: ten cases, three endpoint panels, and
weighted Adam plus adaptive online. Thus these results do not support objective
surrogate mismatch as the primary explanation of confirmation failure. They do
not prove the two-channel surrogate universally sufficient. See
[the authenticated history reduction](../../output/overnight-state-preparation-20260913-0108/validation/CONFIRMATION_HISTORY_REDUCTION.json)
and [complete case diagnosis](CONFIRMATION_DIAGNOSTICS.md).
None of these post-confirmation explanations changes the failed primary gate.

Exact structural stages reset optimizer memory; stochastic structural controls
preserve surviving moments and counters, following their strong established
implementation. This difference is explicit in the frozen procedure, rather
than silently presenting every arm as the same optimizer with a renamed update.
A stage resume preserves complete state for every method. Online sweeps use
stale adjoints and are not claimed to be unbiased finite-step gradient descent.
Heterogeneous exact calls and stochastic updates are never treated as equal
compute. The common-CPU comparison is primary; full-cap comparisons are
supporting evidence only.

## 8. Fidelity, gates, compute and memory tradeoffs

For each target/replica the primary refinement cutoff is the minimum final
compiled endpoint CPU among the four structural roles. **All seven roles** are
scored at that same cutoff using the last chronologically available checkpoint.
A shorter completed arm remains available. There is no interpolation or
quality-based checkpoint selection. Input cost, the measured lazy initializer
and amortized preparation are added equally to define deployable total CPU.
The ten primary total cutoffs range from 25.684801 to 89.551277 seconds; their
variation is retained rather than hidden behind one global time budget.

Physical resource ranges below cover all ten primary endpoints. They count a
single physical circuit before profile-specific Ballarin materialization.
Ballarin lowering counts are separate in the raw rows and receipts; gate counts
are not averaged across noise conditions to change the resource metric.

| Primary endpoint method       | Physical 1q gates | Physical 2q gates | Parameters | Depth |
| ----------------------------- | ----------------: | ----------------: | ---------: | ----: |
| Structural L-BFGS (candidate) |             33–68 |             10–26 |      43–92 | 24–31 |
| Structural weighted Adam      |             28–66 |              6–27 |      34–88 | 17–36 |
| Structural adaptive online    |             35–71 |              7–27 |      44–98 | 23–32 |
| Structural exact Adam         |             32–68 |              9–27 |      41–91 | 22–36 |
| Fixed mean-two L-BFGS         |             41–74 |             12–30 |     53–104 | 29–38 |
| Fixed noiseless Adam          |             41–74 |             12–30 |     53–104 | 29–38 |
| Unchanged initializer         |             41–74 |             12–30 |     53–104 | 29–38 |

MPS bond 3 exposes a concrete fidelity/resource conflict: weighted Adam retains
8 and 6 two-qubit gates across its two replicas, versus 10 and 10 for L-BFGS,
and has higher noisy fidelity despite lower noiseless fidelity. This prevents
a claim of uniform candidate gate efficiency. A nondominated endpoint in a
multiobjective tradeoff is not itself proof of the required 0.5-point advantage.

The Pareto utility uses all four declared objectives: maximize mean noisy
fidelity, minimize physical one-qubit gates, minimize two-qubit gates and minimize
complete deployable CPU. Each recipe and endpoint stratum is analyzed separately.
Two-dimensional projections mark membership of that four-objective frontier;
they are not silently substituted for it. Authenticated development plot inputs
and arrays are saved beside
[the original/compiled comparison](../../output/overnight-state-preparation-20260913-0108/figures/development_pareto_primary_v3.png).
Original and compiled points have different streams and CPU availability;
that plot is a development comparison, not a causal compiler experiment.

The final confirmation figures retain all five targets, both replicas, all seven
roles and the three separate endpoint strata:
[resource/Pareto projections](../../output/overnight-state-preparation-20260913-0108/figures/confirmation_tradeoffs_v2_scatter.png)
and [paired condition heatmaps](../../output/overnight-state-preparation-20260913-0108/figures/confirmation_tradeoffs_v2_heatmaps.png).
Their [exact arrays and custody](../../output/overnight-state-preparation-20260913-0108/figures/confirmation_tradeoffs_v2.json)
and [point table](../../output/overnight-state-preparation-20260913-0108/figures/confirmation_tradeoffs_v2.csv)
make every plotted value recoverable. Primary candidate averages are 47.0
one-qubit gates, 15.3 two-qubit gates and 51.86937 deployable CPU seconds;
weighted Adam averages 45.1, 15.6 and 52.52947 respectively. The small CPU/two-qubit
savings accompany more one-qubit gates and inconsistent fidelity gains. They
are not an exceptionally clear resource advantage or a route around the failed
confirmation gate.

The fresh complete confirmation cost is **2,818.474129 CPU seconds**, counted as
preparation once plus the ten inclusive physical case invocations. Shared
initialization is not multiplied by seven when totaling actual execution.
Nested training, score, constructor and compiler clocks diagnose work but must
not be added again. Worker-lifetime peak RSS ranges from 302,497,792 to
404,176,896 bytes and includes previous work in reused workers; it is not
incremental optimizer memory.

The final **28-panel ledger** retains **12,318 disjoint physical events** and
**13,765.937587 measured CPU seconds (3.823872 CPU hours)**. Every one of the
10,624 historical 27-panel events is preserved exactly; the 1,694 new events add
47.715553 CPU seconds for independent verification. The ledger includes the
complete confirmation, MPS memory diagnostic and independent evaluator. Its
category totals are: optimization 12,810.036432; construction 335.585177;
evaluation 120.507105; compilation 11.536556; diagnostics 254.344908; bookkeeping
231.696625; setup 2.230784 CPU seconds. **Complete overnight/development CPU
remains unknown**, because some early and engineering work was not instrumented.
This measured subtotal is neither full elapsed time nor the complete research
cost. The old 27-panel ledger remains a historical artifact; do not add its
subtotal or the verification run again to the consolidated total.

The 28-panel accounting audit costs 87.241795 CPU and 149.320147 wall seconds,
outside scientific event sums. The central scoreboard audit costs 317.282637 CPU
and 360.936062 wall seconds, also engineering. All 12,318 event rows in the new
CSV match their JSON measurements. Attempted/completed trajectory, gradient and
forward counters have different missing coverage; there is no honest single
total number of interchangeable evaluations.
[LEDGER.json](../../output/overnight-state-preparation-20260913-0108/resource-accounting-28panels/LEDGER.json)
and its `EVENTS.csv` retain detailed proof links and unknown fields, with
[the independent accounting audit](../../output/overnight-state-preparation-20260913-0108/validation/RESOURCE_28PANELS_AUDIT.json).
The ledger canonical checksum is
`925da91a2d27f29ea17d0246edce2025349bf5fdacaef91ff8fb9f5102e2106d`.

## 9. Negative, failed and retired approaches

These conclusions apply to the frozen settings tested, not entire mathematical
families. Failed attempts and tuning work remain in the ledgers.

| Tested idea                                        | Decisive observation                                                                                                     | Disposition                                                                    |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Three-probe exact coordinate sweeps                | −1.744385 pp local endpoint versus first-wave Adam; common CPU −0.038757 pp, 2/4 wins                                    | Retire the standalone 20-sweep schedule                                        |
| Three-stage noise homotopy                         | −0.205711 pp endpoint; +0.095929 pp common CPU, about 7% cost reduction                                                  | Below practical gate; no rate rescue                                           |
| Short adaptive-online screen                       | Loses all four exact-Adam endpoint/common-CPU pairs                                                                      | Reject that short schedule; longer strong adaptive controls remain competitive |
| Universal rank-two TT construction                 | 80.792490% overall after Adam; Haar only 46.240992% despite structured targets near 92%                                  | Expressivity failure; conditional fallback investigated separately             |
| Exact-anchor control variates                      | Strong nearby-anchor gradient error/cost reduction; displacement-one advantage disappears for Haar/MPS after anchor cost | Useful local mechanism, no general optimizer promotion                         |
| Refresh-five SVRG Adam                             | 86.396072%, versus weighted Adam 85.863267% and exact Adam 86.379599%; CPU 4.050286 vs 2.888228 and 3.911794 s           | Nearly reproduces exact Adam; no strong q6 advantage over exact gradients      |
| Mean-eleven objective, 1,100 constituent gradients | 88.392926% vs mean-two 89.408523%; only 100 vs 550 joint calls                                                           | Finite-budget warning, not a theorem about global optima                       |
| Mean-eleven, sole 6,050-call extension             | Common CPU 89.337888% vs mean-two Adam 89.497042%; 1/4 wins                                                              | Retire; full-cap −0.165257 pp misses 0.1-point resource tolerance              |
| Pure-state geometric preconditioning               | Damping 0.001: −0.107855 pp common CPU and more compute; 0.01: −0.724715 pp                                              | Both tested schedules retired                                                  |
| Entropic risk, temperatures 0.1/0.02               | Best mean +0.107425 pp, worst-condition +0.121830 pp; smaller temperature loses about one point                          | All four optimizer/temperature settings retired                                |
| Local-channel structural optimizer claim           | Weighted/adaptive controls remove the apparent L-BFGS advantage                                                          | Architecture signal retained; optimizer claim rejected                         |
| Mean-two structure on original 108 gates           | Strongest common-CPU gain only +0.462704 pp with 2/4 wins                                                                | Insufficient before complete conditional composition                           |
| Known-input endpoint compiler                      | 61 RZ gates/parameters removed in 15 of 24 pairs; +0.036813 pp mean; no 2q removal                                       | Small shared resource result, not an optimizer win                             |
| Known-input compiler before/after training         | +0.377773 pp vs strongest adaptive control, only 2/4 development wins                                                    | Reject recipe; do not rescue with better candidate raw mean                    |
| Selected original composition                      | Confirmation primary +0.324929 pp vs strongest control, only 3/5 wins                                                    | No promotion; no held-out retuning                                             |
| First MPS overlap gate                             | All eight parameter-coordinate checks fail 1e-10 after Adam roundoff amplification                                       | Original failure remains; separate physical diagnostic only                    |

Nearby-anchor control-variate diagnostics used 32 independent paired
replications. At displacements 0.05 and 0.2, every target/replication had lower
gradient squared error than equal-direction-work weighted sampling. Exact-anchor
cost amortized over five uses gave error-times-CPU ratios about 0.00155–0.00406
and 0.02482–0.06106. That local evidence does not establish a global optimization
or MPS advantage.

Generic algebraic compilation preserved all sixteen tested ideal endpoints and
improved their group means, but did not repair rank-two Haar expressivity.
The separate known-input screen completed all 24 source/compiled pairs, 576
fresh profile evaluations and 48 independent ideal-state preparations. Nine
pairs were unchanged, and no pair-mean loss was resolved. Structural L-BFGS
finals gained only 0.008195 points. It remains a small endpoint effect, while
its newly matched before-training recipe failed development selection.

## 10. Validity, selection bias and engineering custody

The immutable benchmark Markdown and target fixture retain startup hashes
`40685856bb052b88aab924792df74159a2b3bdf47d047adfb84183f5f4076f11`
and `d53cd272b6c665b3a92de2d6b1e335b85ef77f74016b0f20cf7ea6a0f486b839`.
All eighteen q6/q12 fixture vectors were initially parsed and validated.
Every preparation includes its physical input construction. No arbitrary
multiqubit ideal gate is counted as one noisy operation. Exact final scoring
has no reused stochastic training/test stream.

The completed independent evaluator audit reproduced all **2,520 saved
fidelity values** across ten cases, seven roles, three endpoint strata and twelve
profiles. The 210 role endpoints reduce to 141 unique target/circuit/parameter
combinations and **1,692 independent profile calls**. Maximum absolute fidelity
discrepancy is **9.992007221626409e-15**, below the frozen 1e-10 tolerance. Every
row remains present, with no failed call, missing row or CPU-cap overrun.

This oracle constructs Qiskit gates and starts from physical zero. Standard
channels use literal Pauli-Liouvillian matrix exponentials calibrated through
Qiskit's average-gate-fidelity function, independently of the selected evaluator's
Pauli-Fourier construction. Ballarin native lowering, threshold rules and local
Kraus channels are transcribed separately. The maximum trace defect is
1.343414939952874e-14, maximum reported Hermitian residual 3.910125086057837e-16,
and minimum reported Hermitian-part eigenvalue -6.932262332683731e-16; each meets
the fixed 1e-10 physicality tolerance.

A separate read-only audit reconstructs every comparison from raw oracle
receipts and sealed endpoints, verifies physical and Ballarin counts, and proves
all **12,080 original confirmation files unchanged**. It recomputes trace defects
from saved trace components; final density matrices are not retained, so this
later audit reduces reported Hermitian/eigenvalue scalars rather than repeating
their numerical calculations. Independent fixed-circuit agreement passes;
**the original performance promotion remains failed**. See
[INDEPENDENT_VERIFICATION_FINDINGS.md](INDEPENDENT_VERIFICATION_FINDINGS.md),
[the complete verification result](../../output/overnight-state-preparation-20260913-0108/independent-confirmation-validation/RESULT.json)
and [the independent result audit](../../output/overnight-state-preparation-20260913-0108/validation/INDEPENDENT_VERIFICATION_RESULT_AUDIT.json).

The verification costs **47.715553 CPU seconds and 68.135319917 wall seconds**,
including its 3.846407-CPU-second preparation. It is separate from training and
is already included once in the consolidated 28-panel ledger in section 8. Its later reduction costs 4.255584
CPU seconds of engineering. The exact public verifier CLI also passes a cache
resume with all **7,197 verification files unchanged**, no new oracle calls,
and 7.037975 CPU / 16.456421 wall seconds of engineering.
[Verifier resume audit](../../output/overnight-state-preparation-20260913-0108/validation/INDEPENDENT_VALIDATION_CLI_RESUME_AUDIT.json).
Neither verification nor cache reuse changes a selected endpoint, algorithm or
performance gate, establishes hardware accuracy, or justifies larger scaling.

The primary threats are gains concentrated in Haar while both other family
means lose to the strongest control, selection over many development hypotheses, only four
development and five confirmation identities, two paired seeds, classical
access to full target amplitudes, surrogate-objective transfer, nonconvex
construction sensitivity, laptop CPU variability and different estimator costs.
Exact channel expectations remove sampling error at a saved circuit, not
selection bias or target-population uncertainty. The common-CPU stopping rule
is fixed before outcomes but observed CPU still depends on machine execution.
Chronological selection can retain a raw endpoint just before a beneficial
final compiler; this is part of the declared deployability comparison. Fresh
process timing or different hardware can move a cutoff between checkpoints even
when optimizer trajectories are deterministic. Immutable saved receipts reproduce
the reported selection exactly; cache-resume proof is distinct from fresh-run
bitwise equivalence across machines. Replicas cover constructor and stochastic
training variability, and some deterministic TT branches have identical full-cap
outputs. They are not additional independent target samples.

The initial Gaussian 100-update exact-Adam reproduction was bit-identical at
local fidelity 0.8243802176093276. Noiseless density/statevector gradients agreed
within 6.53e-16. New kernels have independent small-system analytic, dense,
finite-difference, Qiskit, unbiasedness and exact-resume checks. The reporting
adapter has independent receipt/tampering tests. The authoritative corrected
full owned cohort passes **1,409 tests in 116.09 seconds**, with zero failures,
errors or skips. All **53 source modules and 51 test files** retain their 104
pre-run hashes. Corrected combined coverage is **80.835381%** (81% displayed):
**10,140/12,298 statements, 82.452431%**, and **3,020/3,982 branches, 75.841286%**.
There are 2,158 uncovered statements, 962 missing branches and **169 excluded
lines**, independently classified as 52 TYPE_CHECKING headers and 117 associated
import lines. No numerical function lines remain excluded. This is measured
coverage of the owned package, not complete repository or behavioral coverage.

The new measurement uses a frozen temporary configuration with the ellipsis
exclusion anchored to a standalone statement; repository settings and scientific
sources are unchanged. All 13 historical and two new oracle functions previously
hidden by tuple annotations now have nonzero denominators and no excluded lines
in their function summaries. The earlier 1,249-test and separate 152-test
coverage reports remain historical; their counts are not added to the complete
1,409-test result. [Coverage review](TEST_COVERAGE_REVIEW.md),
[corrected coverage JSON](../../output/overnight-state-preparation-20260913-0108/validation/coverage-final-corrected.json)
and [final cohort audit](../../output/overnight-state-preparation-20260913-0108/validation/FINAL_CORRECTED_COHORT_REVIEW.json)
record the current scope, residual gaps and all restored function measurements.
[An independent coverage audit](../../output/overnight-state-preparation-20260913-0108/validation/INDEPENDENT_CORRECTED_COVERAGE_AUDIT.json)
confirms all 104 hashes, fifteen restored bodies and the typing-only exclusions.

The public best-candidate CLI also completed a cache resume of all ten cases,
leaving all 12,080 files and every attempt/score receipt byte-for-byte unchanged;
[its audit](../../output/overnight-state-preparation-20260913-0108/validation/CONFIRMATION_CLI_RESUME_AUDIT.json)
records no new scientific execution. Full nox batch 32 records zero new-code
violations, all typing checks passing and all new files equal to the reviewed
snapshot. Known unrelated legacy Ruff/license/toolchain/manuscript issues and a
frozen-citation spelling false positive remain. This is not a clean
full-repository lint result.

Full `uvx nox -s lint` runs use source snapshots so autofix cannot change frozen
or unrelated user files. One accidental live invocation was interrupted; six
autoformatted user files were restored byte-for-byte from pre-hook copies,
not reset to Git HEAD. The restoration and complete pre-existing-file custody
are audited. The estimator's annotation-only rerun reproduced all 512 gradient/
loss replications bit-for-bit; both physical attempts are retained and charged.

A source-recovery extraction found that the direct 114-file confirmation map
was not a complete Python import closure. Three additional imports are required:
trajectory comparison, the benchmark generator and the lazy-reproduction
wrapper. The first two match authoritative startup hashes; the third is bound
by the lazy-reproduction protocol frozen inside DESIGN's development evidence.
No drift was found. They are archived as supplementary sources with these
proofs, and the omission is recorded honestly. Neither the frozen DESIGN nor
any numerical source was retrospectively changed.

The final code/benchmark custody audit verifies all 287 startup files after
removing only the exact owned additive CHANGELOG/UPGRADING blocks, all 114
selected source hashes, 97 selected development-evidence hashes and the 104
final source/test hashes. It also reads back all 153 selected-source archive
members. [Final custody receipt](../../output/overnight-state-preparation-20260913-0108/validation/FINAL_CODE_AND_BENCHMARK_CUSTODY.json).
The 0.241830-CPU-second audit is engineering; it does not add scientific runs.

The complete confirmation analysis is negative, so no confidence claim is
rescued by post-hoc exclusions, alternate full-cap selection or a new recipe.
All current q6 target identities have now been used. Fresh seeds alone cannot
make those target identities untouched again.

## 11. Is MPS scaling justified?

**No.** Confirmation failed, and there is no exceptionally clear measured
resource advantage that would justify bypassing that gate. Construction uses
MPS, but noisy density training and exact deletion are not thereby shown to
scale as an MPS algorithm.

The first q6 MPS overlap experiment failed all eight parameter-coordinate
checks: three Adam updates differed by 1.25e-9–3.95e-9, above its frozen 1e-10
threshold. A separate physical follow-up passed all 24 saved steps under its
own declared sensitivity criterion. At identical theta, maximum gradient error
was 2.290269e-15 and Qiskit/MPS state distance 8.784010e-15. Different saved
parameters produced state distances at most 1.509320e-10, within the explicit
sensitivity-plus-numerical allowance. All 23,280 audited splits had zero
discarded weight and cap reductions, maximum bond 8. This explains update
amplification of roundoff without making the original coordinate gate pass.

The later retained-memory diagnostic covers eight **older development
endpoints**, not the complete selected conditional candidate. Its predeclared
analytic path zero has no noise factors, so it is not a representative full
64-path training batch. Full retained tapes occupy 122,816–365,952 bytes plus
2,688 target bytes, with bonds 4 or 8. MPS gradient CPU totals 0.367551 seconds
against dense 0.027735 seconds, about **13.25 times slower at q6**. The inclusive
experiment costs 1.032438 CPU seconds; nested gradient/inspection timings are
not added again. Norm and gradient errors stay below 6e-15.

The earlier 2,688-byte quantity is aggregate tensor storage for one MPS state,
not a whole retained tape or process peak. Even the new tape snapshot excludes
unreturned adjoints, temporary contractions, Python containers and allocation
overhead. Worker-lifetime RSS of 168,574,976–169,426,944 bytes includes dense
validation and earlier cases. Neither memory superiority nor a scaling speedup
is established. See [MPS_MEMORY_FINDINGS.md](MPS_MEMORY_FINDINGS.md).

The canonical fixture contains q6 and q12 entries only. The proposed q8 design
is ineligible because its required canonical targets do not exist; do not
generate replacements. No q8/q12 scientific run was launched. A future larger
study requires a new protocol, a scientifically justified admission gate,
overlapping untruncated gradients/updates, measured bond growth/discarded
weights and complete fidelity/runtime/memory accounting.

## 12. Precise recommendation for the next run

Do not scale or retune the confirmed recipe. The most informative next
**development-only** experiment is one controlled budget-allocation comparison:
move part of terminal polishing into symmetric structural repair, keeping the
same total constituent-gradient cap and all other method components fixed.
Predeclare 50 joint-gradient evaluations per exact repair instead of 25:
100 rather than 50 individual-channel gradients for the two training profiles.
Retain the same two deletion candidates and equally repaired retained branch,
keep the 15-round cap, and spend only the remaining budget on terminal polishing. Give exact Adam, weighted Adam and
adaptive online the same repair-allocation opportunity in their own declared
update units. Compare each altered arm with its original repair schedule at
common measured CPU; do not equate heterogeneous updates with cost. Do not add
an insertion, objective change or compiler change to this experiment.

Use only the original four development identities with paired replicas 11/29,
the same conditional starts and all twelve exact endpoint scores. Charge every
rank, failed line search, rejected repair, compiler and selection evaluation.
Freeze the schedule and source hashes before running; use one alternative,
without a budget or learning-rate rescue. The falsifiable question is whether
more repair before a greedy structural stop improves the strongest-control
comparison enough to repay its cost. A useful signal must improve the full
mean by at least 0.5 points with consistent target signs, or meet a separately
predeclared substantial resource criterion. If it fails, retain the negative
result and the strong weighted/adaptive references. This proposal is informed
by saved stopping histories; it is not a causal conclusion or an experiment
performed during the completed search.

Any changed method is a new development method. All nine fixed q6 target
identities have now been used. Fresh seeds can quantify variability but do not
restore untouched-target status. A new confirmation claim requires a separately
authorized, independently fixed untouched panel outside this exhausted split;
this study's immutable fixture must not be changed. Further runs on the fixed
benchmark alone remain descriptive development or reproduction.

## 13. Reproduction commands, exact artifacts and source recovery

Repository root:
`/Users/maximilianfrohlich/Desktop/GitHub/yaqs`.
Research source package:
`experiments/overnight_state_preparation_20260913_0108`.
Output root:
`output/overnight-state-preparation-20260913-0108`.

The single best-candidate CLI authenticates and resumes the original complete
parent. With the existing complete output it verifies stored receipts rather
than starting a new recipe. Do not alter numerical sources or install/build
into the frozen tree before verification:

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
uv run python -m experiments.overnight_state_preparation_20260913_0108.best_candidate --run --workers 1 --output output/overnight-state-preparation-20260913-0108/confirmation-best-candidate
uv run python -m experiments.overnight_state_preparation_20260913_0108.independent_confirmation_validation --output output/overnight-state-preparation-20260913-0108/independent-confirmation-validation
uv run pytest -n 0 --no-cov tests/optimization/test_overnight_best_candidate_20260913.py tests/optimization/test_overnight_reporting_20260913.py
```

Fresh independent numerical reproduction requires a new output directory and
explicit allocation; it must preserve the chosen configuration and cannot be
used to replace an unfavorable seed. Freeze/prepare commands and source-specific
resume instructions are in [HANDOFF.md](HANDOFF.md). Full lint uses
`uvx nox -s lint` inside a complete owned source snapshot, with all failures
retained. The exact selected-source archive is not such a full lint snapshot.

| Artifact                                                                                                                                                                                                                                                                                                                                        | Purpose                                                                                       |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| [MANIFEST.json](MANIFEST.json), [STARTUP_FILES.json](STARTUP_FILES.json)                                                                                                                                                                                                                                                                        | Immutable benchmark hashes, hardware/environment, seeds and pre-existing-file custody         |
| [LITERATURE.md](LITERATURE.md), [CANDIDATES.md](CANDIDATES.md), [METHOD.md](METHOD.md)                                                                                                                                                                                                                                                          | Original sources, all hypotheses/dispositions and complete algorithm                          |
| [Central SCOREBOARD.csv](../../output/overnight-state-preparation-20260913-0108/SCOREBOARD.csv) and [SCOREBOARD.md](../../output/overnight-state-preparation-20260913-0108/SCOREBOARD.md)                                                                                                                                                       | Twenty authenticated source panels, 10,632/10,632 cells, zero missing; strata remain separate |
| [Confirmation DESIGN](../../output/overnight-state-preparation-20260913-0108/confirmation-best-candidate/DESIGN.json), [PROTOCOL](../../output/overnight-state-preparation-20260913-0108/confirmation-best-candidate/PROTOCOL.json), [SUMMARY](../../output/overnight-state-preparation-20260913-0108/confirmation-best-candidate/SUMMARY.json) | Frozen choices, complete ten-case parent and failed primary gate                              |
| [Confirmation authentication](../../output/overnight-state-preparation-20260913-0108/validation/CONFIRMATION_AUTHENTICATION.json)                                                                                                                                                                                                               | All case checksums, 12,080 unchanged files and exact recomputation                            |
| [Case diagnostics](../../output/overnight-state-preparation-20260913-0108/validation/CONFIRMATION_CASE_DIAGNOSTICS.json)                                                                                                                                                                                                                        | Stored endpoint/compiler/initializer details without new optimization                         |
| [Resource ledger](../../output/overnight-state-preparation-20260913-0108/resource-accounting-28panels/LEDGER.json)                                                                                                                                                                                                                              | Disjoint physical work, source proofs, unknown cost coverage                                  |
| [Exact-source archive](../../output/overnight-state-preparation-20260913-0108/reproducibility/selected-original-7ca2de674a9e.tar.gz) and [readback verification](../../output/overnight-state-preparation-20260913-0108/reproducibility/selected-original-7ca2de674a9e.verification.json)                                                       | Recover the selected bytes from the dirty worktree                                            |

Configuration SHA256:
`7ca2de674a9e13e863dfaa0c1b4312dee90a1d7cb437fa1efc9a3bad302d9068`.
DESIGN canonical content checksum:
`203ee474a2500b752c0f46063a32838e51aa73dd2bd1dad48f3797916e9ca976`;
its file-byte SHA256 is
`b606b4fd1b0d6024f567de06abb216911e01e1d605d9e241bcf13888f4b2a024`.
Parent canonical checksum:
`5a8a4980116a955ab5146c4e146c423d7f84ff92ce66a3c23167cfe394e0cb4e`.
Confirmation SUMMARY canonical checksum:
`426c02e029dc7db5bf0664520f46db556364f45c2879f5d5a516a6ec6a5820c9`.

The archive is 1,175,328 compressed bytes, SHA256
`5cb516c704242262a8976b072a88405e2a7cd8343da9a392df3818808e00e4ba`.
All 153 members were read back and hashed. It contains all 114 directly frozen
source files, both benchmark inputs, supplementary import/support files,
19 compact research/custody receipts, dirty-Git state and 130 installed
distribution names/versions. An owned temporary extraction resolved 101 local
modules inside the archive, reproduced the exact configuration hash and passed
`runner.custody()` without loading targets or running numerics. No Hatch build
regenerated the frozen version file.

This is **exact source recovery**, not a fully relocatable study deposit.
The complete raw training/evidence graph is separately retained and contains
absolute paths; compact receipts do not make those dependencies disappear.
The archive README states the import-closure omission, source proofs, environment
limits and verification procedure. Historical scoreboards and prior ledgers
remain preserved; no benchmark, manuscript or unrelated user work was changed
to manufacture a positive result.
