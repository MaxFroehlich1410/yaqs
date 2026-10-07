# Mean-two-channel deletion and exact refinement

This is a new composed development candidate. It combines the established
finite-deletion mechanism with the equally weighted local depolarizing/dephasing
objective. Earlier local-objective structural results remain unchanged. No
scientific execution is authorized for this module until coordinator allocation
and the full lint batch pass.

## Hypothesis and provenance

The original 108-gate mean2 L-BFGS panel produced a target-balanced eleven-noise
mean of 89.4085%, with gains of 3.2269 points over exact Adam 1100 and 2.0231 points
over noiseless L-BFGS, each positive on all four development targets. Exact noisy
deletion is independently promising. Their composition may retain the broader
noise-family improvement while reducing gate exposure and parameter dimension.
This is a hypothesis about their composition, not evidence that it already works.

The strongest matched local-objective structural results require caution:
adaptive-online achieved 89.7698% at the full cap, weighted Adam 89.4412%,
L-BFGS 89.1697%, and exact Adam 89.0986%. At the paired common CPU, weighted Adam
led at 89.4412%, versus L-BFGS 89.2692%, adaptive-online 89.1803%, and exact Adam
89.03396%. A deterministic two-channel comparison therefore establishes only a
mechanistic candidate. A strongest-baseline claim requires matching the new
objective opportunity for weighted/adaptive methods or directly beating the
strongest applicable existing baseline at matched CPU and declared resources.

The grow/simplify perspective is known from
[VAns, Bilkis et al.](https://arxiv.org/abs/2103.06712). This experiment restricts
it to exact finite deletion and independent parameter repair. The prefix/suffix
counterfactual extends `density_deletion_fidelities` in the existing repository;
VAns is not claimed to provide this exact estimator or these benchmark results.
The two-channel mean is the already implemented ensemble-control objective.
There is no new channel, target, arbitrary ideal multi-qubit gate, or novel
optimizer claim.

## Finite deletion identity

For each fixed channel profile c, write a physical layer as
`C[c,j](rho) = E[c,j](U[j] rho U[j]^dagger)`. Let rho[c,j-1] be the density just
before that layer, and B[c,j] the target projector propagated backward through
all later layers. Removing both the gate and its complete noise window gives

`F[c,-j] = Tr(B[c,j] rho[c,j-1])`.

The suffix recursion is `B[c,j-1] = U[j]^dagger E[c,j]^*(B[c,j]) U[j]`.
Every supported constant Pauli channel is self-adjoint, so the same complete
fixed-window probabilities implement E and its adjoint. The ranking score is
`(F[depolarizing_1s_all,-j] + F[dephasing_1s_all,-j])/2`.

One ranking uses a forward and backward density pass for each profile: two
individual channel calls and four density directions. It computes all deletions,
not one complete circuit simulation per gate. Profiles are processed sequentially
with `O(L*4^n)` peak density tape storage. The kernel supports all ten fixed noisy
profiles and noiseless, but rejects angle-dependent Ballarin ranking. Final
Ballarin evaluation remains the unchanged normative evaluator. All first 18
original physical gate positions are protected, even after index shifts.

## Frozen first screen

Use only Gaussian, critical TFIM, Haar 1 and random MPS bond 2 at construction
replica 11, with the authenticated original q6 108-gate starts. Every circuit
physically prepares from zero. Train all three arms on exactly the same equally
weighted `depolarizing_1s_all` / `dephasing_1s_all` objective:

- structural mean2 L-BFGS, history 10;
- fixed mean2 L-BFGS, history 10;
- structural mean2 exact Adam, rate 0.02, beta1 0.9, beta2 0.999, epsilon 1e-8.

Each arm has a full cap of 3000 individual channel calls, equivalent to 1500
mean-objective calls. Structural warmup has a cap of 150 mean gradient calls,
followed by at most 15 deletion rounds. Each round shortlists two eligible gates
using exact mean finite-deletion fidelity, quantized at 1e-12 for stable ties;
original physical gate index breaks ties. It gives the retained branch and both
deleted branches the same fresh optimizer memory and 25 mean gradient calls.
Each repaired endpoint is independently scored with two exact forward channel
calls. A deleted endpoint must beat both the incoming endpoint and the repaired
retained branch by more than 1e-10. Retention can improve parameters even when no
deletion wins. Stop search when no repaired deletion wins, no gate is eligible,
or the remaining cap cannot reserve a complete symmetric round plus final
refinement. Spend the remaining cap on fresh-memory final refinement of the
selected structure. Every rejected branch and line-search call is charged.
A numerical branch failure is retained as a failed arm, not rescued by retuning.

The fixed control runs continuously for its 1500 mean gradient-call cap from the
same original start. Its first 150 calls provide the same L-BFGS warmup opportunity;
its curvature is retained thereafter. Warmups are physically executed within each
arm and charged individually; no shared execution is subtracted. Structural Adam
keeps moments within each stage and resets them for every repaired branch and
final refinement, exactly as structural L-BFGS resets curvature. The existing
resumable ensemble trainer is reused with a two-profile budget configuration;
its numerical optimizer implementations are unchanged.

A full three-branch round uses `1 ranking + 3*(25 gradient + 1 forward score) = 79`
mean-call units. Ranking and gradient calls each use two density directions per
profile; forward scores use one. Thus equal individual-channel-call caps are
not claimed to imply identical compute. Report gradient, ranking, forward-score,
physical attempted/completed, and direction counts separately. Stage caps may
finish early on convergence; unused calls are reported and the frozen remaining
refinement rule determines their subsequent allocation.

## Endpoint, CPU matching, failure and storage rules

Every final arm is its last optimizer iterate, never its best checkpoint. Every
final circuit is scored under all twelve fixed benchmark conditions. Training
and ranking consume no stochastic randomness. Final exact evaluation uses
separate forward calls, including unchanged Ballarin materialization; there is
no final random stream to overlap with training.

Record parameter curves at accepted optimizer iterates. Search curves expose
only completed retained/selected round endpoints after paying every candidate's
cost. Select the last available iterate at or before the shortest completed-arm
training CPU, then evaluate all twelve channels. Selection uses CPU only, never
fidelity. Include measured interruption replay work in these CPU curves. Exact
final/common-CPU scores may be reused only for identical circuit, theta, target
and channel identities, with physical evaluation reuse explicitly recorded.

Training CPU follows the frozen ensemble trainer: optimization and intermediate
checkpoint I/O are included; final ledger writes, imports, custody, cached-resume
bookkeeping and orchestration are excluded and reported through invocation
measurements. Ranking and forward-score numerical CPU are included separately.
All raw attempt ledgers survive failures and interruptions. Unknown killed work
requires an external ledger; it is never assigned zero cost.

Store every circuit once by digest. Curves contain that digest and theta, never
repeated full circuits or complete optimizer histories. The current durable
checkpoint retains full Adam moments or L-BFGS secants. Completed stages and
operation results are immutable and cached, permitting deterministic replay of
the short structural search after interruption. Disk use must remain bounded;
no large numerical or circuit arrays belong in every round's reporting row.

## Promotion and falsification

Report each target and all eleven noisy conditions equally, with noiseless,
N1, N2 and N3 summaries separate. Promote the composition for focused development
if it improves all-eleven mean over both matched deterministic controls by at
least 0.5 points, or gives a smaller positive target-level gain on all four
cases, without an unexplained failure or a target loss exceeding one point.
Comparable fidelity with a substantial verified gate/CPU advantage is another
permitted promotion route. Keep local/mean2 gains distinct from all-family gains.

Failure against these controls retires this composition setting without a
learning-rate grid. Passing remains development evidence; it does not establish
confirmation performance, the strongest repository baseline, or MPS scalability.
The next comparison must include equally treated strong weighted/adaptive
baselines before any general method superiority claim.
