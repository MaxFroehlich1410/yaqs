# Selected method: conditional physical preparation with noisy structural L-BFGS

This document describes the `original` recipe frozen for the complete
confirmation experiment. It is an explanatory research document, not an
algorithm input or an amendment to the frozen protocol. It contains no
confirmation results. The authoritative choices are in the
[frozen design](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/output/overnight-state-preparation-20260913-0108/confirmation-best-candidate/DESIGN.json)
and [parent protocol](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/output/overnight-state-preparation-20260913-0108/confirmation-best-candidate/PROTOCOL.json).
The implementation composes a target-only initializer, physical circuit
simplification, exact optimization of two noisy fidelities, and adaptive
deletion with equally funded repair branches. All comparison methods receive
the same constructed and initially compiled circuit.

## Physical problem and objective

For a normalized target vector $|\psi\rangle$, every preparation begins with
$\rho_0=|0\cdots0\rangle\langle0\cdots0|$. A scalar gate has the form
$U_j(\theta)=\exp[-i\alpha_j(\theta)P_j/2]$, where $P_j$ is a supported
one-qubit Pauli or a two-qubit XX, YY or ZZ generator. Its physical layer under
noise profile $c$ is

$$
\mathcal E_{c,j}(\rho)
 =\mathcal N_{c,j}\!\left(U_j\rho U_j^\dagger\right),\qquad
F_c(\theta)=\langle\psi|
 (\mathcal E_{c,L}\circ\cdots\circ\mathcal E_{c,1})(\rho_0)|\psi\rangle.
$$

The calibrated local channel is attached to each enabled physical gate. An
explicitly disabled noise flag makes that gate's channel the identity. The
selected training objective is

$$
\mathcal L(\theta)
 =1-\tfrac12\bigl(F_{\mathrm{depolarizing\_1s\_all}}(\theta)
                  +F_{\mathrm{dephasing\_1s\_all}}(\theta)\bigr).
$$

These two fixed channels have angle-independent Pauli probabilities. The
candidate differentiates the complete density evolution exactly, including
each gate and its local channel, and averages the two gradients. This is a
dense q6 method; its use of an MPS during construction does not establish an
MPS implementation of noisy refinement. See the
[exact objective](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/ensemble_objective.py:167),
[joint training oracle](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/ensemble_experiment.py:254)
and [calibrated evaluator](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/channels.py:166).

## Target-only construction and a conditional fallback

The initializer first forms a tensor-train approximation of the target by
successive singular-value decompositions, retaining at most two singular
directions at every cut. It does not renormalize between cuts; it normalizes
the contracted approximate vector at the end. Let that vector be
$|\widetilde\psi_2\rangle$, and define
$\eta=|\langle\psi|\widetilde\psi_2\rangle|^2$. This sequential compression
is not asserted to be the globally optimal bond-two approximation.

If $\eta\geq0.99$, the method uses its deterministic physical preparation.
Otherwise it constructs the legacy circuit described below. The rule inspects
neither a target-family label nor noisy fidelity. The probe, overlap audit,
lowering and initial compilation are executed and charged even when the
fallback is selected. The legacy constructor is invoked only after that
decision, so a successful rank-two branch has no unexecuted legacy cost.
See [TT compression](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/synthesis.py:64)
and the [lazy probe and branch](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/lazy_initializer.py:191).

The rank-two preparation is a sequence of adjacent two-qubit unitaries and one
single-qubit unitary obtained by reversing deterministic disentanglers. It
prepares the approximate state from the complete all-zero register. Every
local unitary is lowered before any noisy evaluation: ZYZ Euler rotations for
one qubit and an unsimplified exact Weyl decomposition for two qubits, using
`fidelity=None`. The result contains only scalar rotations, including the
Cartan RXX, RYY and RZZ factors. Independent matrix comparisons require local
unitary agreement up to a global phase within $10^{-10}$. There is no free
arbitrary-unitary gate or supplied initial quantum state. At q6 the raw
lowering has 63 one-qubit and 15 two-qubit physical gates, including zero
angles. Those extra Euler rotations receive their normal physical noise.
See [sequential preparation](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/synthesis.py:110)
and [exact primitive lowering](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/synthesis.py:156).

The fallback is the archive-compatible H25/rate-0.7 construction, executed
freshly with its original MPS backend and seed namespace rooted at 2026090520. Its scalar brickwall matrix-product-disentangler ansatz includes
the initial product-state rotation layer. The construction grows through
depths one to four. At each depth it performs 25 online sweeps at rate 0.7
followed by 75 batch updates at rate 0.2. The first parameters are 0.1 times
the appropriate prefix of one seeded normal vector; newly introduced
parameters use 0.001 times the corresponding entries of that same vector.
It then performs six magnitude-pruning rounds, each removing ten one-qubit
and five two-qubit gates and applying 20 batch repair updates at rate 0.1.
The initial eighteen rotation origins are protected during this construction
pruning. Fifty final batch updates at rate 0.1 complete the schedule:
$4\times100+6\times20+50=570$ updates, ending with 78 one-qubit and 30
two-qubit gates before initial compilation. MPS truncation has no maximum
bond, zero SVD threshold and minimum bond one; the numerical backend and its
dependencies are frozen. Historical CPU is unknown and is not substituted
for the newly measured construction cost. See the
[ansatz](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/src/mqt/yaqs/optimization/parameterized_circuit.py:585),
[constructor schedule](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/construction.py:292),
[magnitude quotas](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/benchmarks/state_preparation/pilot_components.py:173)
and [parent-bound construction wrapper](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/confirmation_construction.py:350).

## Shared physical compilation

The selected recipe applies the generic endpoint compiler to either initial
source and to every trained final endpoint. At a supplied parameter vector,
it resolves fixed, shared and affine scalar angles into physical angles.
It reduces angles modulo $2\pi$, removes an identity factor when its reduced
angle has magnitude at most $10^{-12}$, and fuses equal generators on the
same sites only across intervening disjoint gates. It preserves noise flags
and does not cross an incompatible overlapping gate or noise boundary.
The output has compact independent parameters and an explicit map to the
original factors, with the omitted global phase recorded.

This is equivalence of a fixed ideal endpoint, not a claim that the old and
new parameterized families coincide everywhere. Removing or fusing factors
also removes physical noise windows; noisy fidelity may therefore change.
An independent statevector comparison audits ideal preparation equivalence.
The selected `original` recipe does not apply the separately implemented
state-preparation-specific compiler. See the
[generic compiler](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/compilation.py:50),
[initial audit](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/initializer_policy.py:48)
and [final compiler accounting](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/composed_experiment.py:121).

## Exact finite deletion followed by symmetric repair

The candidate first spends 150 joint objective calls on the compiled start.
Then, for at most fifteen rounds, it ranks every remaining physical gate;
no initial gate is protected at this noisy refinement stage. The ranking
measures removal of the entire gate-plus-noise layer, not a derivative at
zero angle or the magnitude of its current parameter.

For a profile $c$, let $\rho_{c,j-1}$ be the state before layer $j$, and
let
$A_{c,j}=(\mathcal E_{c,j+1}^{\dagger}\circ\cdots\circ
\mathcal E_{c,L}^{\dagger})(|\psi\rangle\langle\psi|)$
be the backward-propagated target effect immediately after that layer.
Its exact finite-deletion fidelity is

$$
F_{c,-j}=\operatorname{Tr}(A_{c,j}\rho_{c,j-1}).
$$

One forward and one adjoint pass per profile produce all such scores. Their
two-profile averages determine the shortlist of two gates. Scores are
quantized at $10^{-12}$ for deterministic ranking, with immutable origin
positions breaking ties. This ranking uses dense tapes and is exact for the
fixed training maps. See the
[finite-deletion contraction](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/ensemble_deletion.py:115).

Each round evaluates three branches: the undeleted circuit and each of the
two single-gate deletions. Every branch receives a fresh L-BFGS repair capped
at 25 joint evaluations, then an exact mean-two fidelity score. A deletion
must beat both the incoming state and the equally repaired undeleted branch
by more than $10^{-10}$. Deterministic strict comparison retains earlier
choices on ties. An improving undeleted repair may be adopted, but structural
search stops if no repaired deletion wins. Remaining budget is used for a
final fresh L-BFGS polishing stage. Gate deletion removes its physical channel
and compacts the surviving parameters and origins. All proposed branches,
including rejected branches, consume budget and measured CPU.

The hard exact budget is 3,000 individual channel calls, or at most 1,500
joint two-profile oracle opportunities. Ranking and explicit forward scores
also consume this budget; it is not a promise of 1,500 gradient updates.
The driver reserves a complete round plus final polishing before starting
another round. See the
[structural driver](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/ensemble_structural_experiment.py:365)
and [explicit role budgets](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/best_candidate.py:195).

## Continuous optimization and controls

L-BFGS stores at most ten curvature pairs and uses the standard two-loop
inverse-Hessian product. Its line search starts at step one and accepts
$\mathcal L(\theta+\alpha p)\leq\mathcal L(\theta)
+10^{-4}\alpha g^Tp$. Rejection halves the step, with at most twenty
trials. Nonfinite trials and rejected trials are charged evaluations. A
non-descent direction resets the history and uses $-g$. A new curvature
pair is retained only when
$s^Ty>10^{-10}\|s\|\|y\|$; the gradient infinity-norm stopping threshold
is $10^{-8}$. The result is the last accepted iterate, never the best
historical score. A checkpoint includes the complete parameter vector,
gradient, secant history, counters and pending line-search trial so exact
resume does not silently restart a stage. See
[L-BFGS and constants](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/optimizers.py:27)
and [resumable minimization](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/optimizers.py:150).

The complete comparison retains seven roles:

| Role                               | Structure and optimizer                                                               |
| ---------------------------------- | ------------------------------------------------------------------------------------- |
| `structural_mean2_lbfgs`           | Candidate: exact mean-two gradients and shared deletion/repair policy                 |
| `structural_mean2_adam`            | Exact Adam at rate 0.02 with the same finite-deletion ranking and repair policy       |
| `structural_mean2_weighted_adam`   | Weighted trajectory Adam at rate 0.02; fixed primary comparator                       |
| `structural_mean2_adaptive_online` | Weighted online control: first 25 updates at rate 0.7, then adaptive online rate 0.02 |
| `fixed_mean2_lbfgs`                | Exact mean-two L-BFGS without structural deletion                                     |
| `fixed_noiseless_adam`             | Noiseless Adam at rate 0.02 without structural deletion                               |
| `unchanged`                        | Shared constructed and initially compiled circuit, without refinement                 |

The two stochastic structural controls use 32 weighted trajectories per
training channel and a 1,500-update cap. Their exact deletion ranking and
branch scoring are additionally charged as physical work. Surviving Adam or
adaptive moments and update counters are preserved through their deletions,
matching the strong historical policy. Exact structural stages use fresh
optimizer memory, including fresh Adam memory in the exact-Adam control.
These memory policies differ and are part of the frozen comparison. Resume
within a stage preserves its full existing state in either case. Weighted
online sweeps use stale adjoints; they are not claimed to be unbiased
finite-step gradient descent. See the
[weighted update adapter](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/mixture_baseline_experiment.py:206),
[stochastic structural policy](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/mixture_baseline_experiment.py:515)
and [weighted trajectory contract](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/trajectory_mixture.py:136).

## Deployable endpoints and measured work

All twelve fixed exact profiles evaluate the same selected physical circuit:
noiseless, Ballarin coupled, and the five support/window variants for each
of dephasing and depolarizing. The eleven noisy fidelities form the noisy
mean; noiseless fidelity is retained separately. Physical gate counts include
every remaining emitted gate, even if a trained angle happens to be zero.
The benchmark's fixed Ballarin materialization and pruning are applied only
as that channel requires, with its materialized counts reported separately.
See the [profile list](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/channels.py:41)
and [endpoint evaluation and counts](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/composed_experiment.py:141).

For each case, the primary refinement cutoff is the minimum final compiled
endpoint CPU among the four structural roles. All seven roles are scored
at that cutoff using their last chronologically available checkpoint.
A final compiler result becomes available only after its own cost has been
paid; an earlier raw checkpoint remains eligible. A shorter completed control
remains deployable at its final point. There is no interpolation or
best-fidelity checkpoint selection. The minimum over all six trained roles
defines a separate sensitivity cutoff, and full-cap final results are also
retained. See the [pure chronological selector](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/common_control_audit.py:35)
and [all-role comparison](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/best_candidate.py:692).

Costs distinguish the developed refinement clock from the total physical
invocation. Per-case input, inclusive initializer cost and amortized batch
preparation enter each arm's deployable total clock once. Optimization,
rankings, repair branches and final compilation enter the relevant refinement
clock. Final evaluation and outer orchestration are recorded in the full
physical case clock. Nested clocks are not summed twice, and shared
construction is counted once when totaling the actually executed experiment.
Caught failures retain attempted CPU and work counts; an unobserved kill with
unknown cost blocks normal resume. Completed caches bind result fields to
the full child-file inventory, attempt ledger and completion receipt.
Worker-lifetime peak RSS is reported with explicit platform units, rather
than presented as incremental algorithm memory. See
[phase accounting](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/best_candidate.py:241)
and [complete case execution](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/best_candidate.py:771).

## Relation to the earlier local methods

The legacy method remains a fully specified construction fallback, including
its noiseless schedule and magnitude-pruning quotas. Subsequent noisy
structural search instead ranks exact removal of each complete physical
layer, compares deletion with equally repaired retention, and stops when
repair cannot establish a deletion advantage. It therefore does not force
a fixed number of deletions or interpret a small angle as small noisy damage.

The initializer extends the earlier universal rank-two proposal with a fixed
target-overlap eligibility test and a complete legacy fallback. All targets
remain in the experiment. Its contribution concerns construction resources
and the starting physical architecture; all strong controls share it, so
the full pipeline should not be described as an optimizer-only improvement.
The common objective is the fixed mean of two noisy profiles, distinct from
earlier single-channel optimization and the separately studied mean-eleven
objective. L-BFGS supplies the continuous optimizer while weighted Adam and
adaptive online remain serious matched structural controls. Historical
best-checkpoint artifacts are provenance references, not matched-cost
comparators for this complete experiment.

The original generic compiler is a shared endpoint opportunity. Neither the
separate state-preparation-aware compiler nor identity insertion, residual
rank-two layers, a family-specific rule, or an MPS noisy optimizer is part of
this selected recipe. The complete composition is implemented by
[the original kernel](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/composed_experiment.py:241)
and frozen by [the parent algorithm specification](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/best_candidate.py:159).

## Frozen confirmation contract

The parent fixes five targets (`tfim_ferro`, `tfim_para`, `haar_random_2`,
`haar_random_3`, `random_mps_bond3`) and replicas 11 and 29. Training uses an
independent `confirmation_training` namespace rooted at 202609130108 and
paired by target and replica, without method labels changing the sampling
root. The six trained roles rotate by complete case index. Constructor seeds
retain the legacy root. Scientific execution requires all four native thread
limits explicitly set to one.

For method $m$, target $t$ and replica $r$, let
$f_{mtr}$ be the mean of its eleven noisy fidelities at the primary cutoff.
The target score is $f_{mt}=(f_{mt,11}+f_{mt,29})/2$, and the aggregate is
$\overline f_m=\frac15\sum_t f_{mt}$. The fixed comparator is
`structural_mean2_weighted_adam`. The strongest overall comparator is also
computed as the maximum aggregate among all six noncandidate roles at that
same primary cutoff. Against each required comparator, the candidate must
gain at least 0.005 absolute fidelity in aggregate and strictly more than
$10^{-10}$ on every one of the five target scores. Full-cap or sensitivity
results cannot rescue a failed primary rule. All ten cases, seven roles and
twelve exact profile scores must be complete, finite and authenticated;
failure or missing cost prevents promotion rather than defining a smaller
panel. See the [acceptance specification](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/best_candidate.py:136)
and [complete-panel analysis](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/best_candidate.py:971).

The configuration SHA is
`7ca2de674a9e13e863dfaa0c1b4312dee90a1d7cb437fa1efc9a3bad302d9068`;
the design digest is
`203ee474a2500b752c0f46063a32838e51aa73dd2bd1dad48f3797916e9ca976`.
An independent read-only audit verified all 114 source hashes, 97 evidence
hashes, and all ten parent-bound case manifests without loading target
vectors or invoking numerical work. The design file precedes preparation
in the recorded filesystem ordering; the complete parent and case manifests
precede execution. This document records that frozen method and audit scope,
not the outcome of confirmation.
