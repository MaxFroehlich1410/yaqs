# Literature, provenance, and implementation decisions

This work combines established optimization, sampling, control and compilation
methods for the repository's fixed state-preparation benchmark. The completed contribution is a reproducible in-repository comparison,
measured component tradeoffs and a negative strongest-control confirmation. It is not a newly invented optimizer, a faithful reproduction of every
cited algorithm, or evidence of external state-of-the-art performance. Statements
about this benchmark below are our development results or inferences; the cited
papers establish the prior mechanisms, not those results.

The independent hypothesis catalogues retain the original mechanisms, assumptions,
falsifiable experiments, costs and novelty assessments:

- [General literature catalogue](LITERATURE_SCOUT.md)
- [Stochastic optimization and robust control](STOCHASTIC_SCOUT.md)
- [Circuit synthesis and architecture](ARCHITECTURE_SCOUT.md)

This mutable synthesis records what was actually implemented. The frozen
per-experiment protocols retain the predeclared settings and endpoint rules;
later interpretations do not amend those protocols.

## Curvature, coordinates and circuit structure

**Exact noisy L-BFGS.** Limited-memory inverse-Hessian updates originate in
[Nocedal (1980)](https://doi.org/10.1090/S0025-5718-1980-0572855-7) and
[Liu and Nocedal (1989)](https://doi.org/10.1007/BF01589116), also listed on
[Nocedal's original software/reference page](https://users.iems.northwestern.edu/~nocedal/lbfgs.html).
The quantum-control connection includes
[de Fouquieres et al., second-order GRAPE](https://arxiv.org/abs/1102.4096).
Our implementation uses exact noisy infidelity gradients, two-loop recursion,
Armijo backtracking, curvature safeguards and complete resumable optimizer state.
It is not SciPy L-BFGS-B and does not inherit that solver's precise line search.
Every rejected objective-gradient call is charged. Its early local-channel cost
signal motivated further development, but improving local depolarization alone
reduced transfer to dephasing in stronger comparisons. Curvature is consequently
one component of the complete recipe, not sufficient evidence of a superior
state-preparation optimizer.

**Exact coordinate minimization.**
[Nakanishi, Fujii and Todo](https://arxiv.org/abs/1903.12166) and
[Ostaszewski, Grant and Benedetti](https://arxiv.org/abs/1905.09692) provide the
trigonometric coordinate mechanisms. Our adaptation contracts cached noisy
observable suffixes against sequentially updated density prefixes. A unique
scalar Pauli rotation with parameter-independent surrounding channels has an
exact sinusoidal coordinate objective. This assumption is not extended to
angle-dependent Ballarin noise or arbitrary shared coordinates. The tested
standalone sweep budget did not beat the serious Adam comparison; monotone
coordinate improvement was not sufficient for competitive convergence.
[Wave-one evidence](../../output/overnight-state-preparation-20260913-0108/wave1/ANALYSIS.json)
retains that negative result.

**Exact finite deletion and symmetric repair.**
[Bilkis et al., VAns](https://arxiv.org/abs/2103.06712) already grows and removes
circuit gates during optimization. Our smaller implementation is deletion-only:
exact prefix/suffix contractions rank removal of the physical gate and its noise
window, then the retained circuit and shortlisted deletions receive equal repair
opportunities. Actual repaired noisy fidelity decides acceptance. This is not a
Taylor or Hessian-saliency approximation and is not a faithful full VAns search.
The exact ranking was extended from one calibrated channel to the equal mean of
two channels. Rejected repairs, retained-branch work and ranking costs remain
charged. The complete composition gives exact Adam, weighted Adam and adaptive
online the same structural and initialization opportunities.
[Raymond and He, Rotolasso](https://www.ijcai.org/proceedings/2025/0680.pdf) is direct
prior art for combining coordinate optimization with sparsity; no novelty claim
follows merely from combining pruning and parameter optimization.

## Mean objectives and risk-sensitive control

**Two-channel and eleven-channel ensembles.** Simultaneous optimization over
an ensemble is established in quantum control, for example
[Chen et al., sampling-based learning control](https://arxiv.org/abs/1308.1454).
Our finite exact objective averages calibrated benchmark-condition expectations,
rather than sampling unknown Hamiltonian parameters. The implemented mean-two
objective equally weights `depolarizing_1s_all` and `dephasing_1s_all`; mean-eleven
equally weights every noisy benchmark condition. All final evaluations still use
the same complete twelve-condition panel, with noiseless fidelity kept separate
from the primary eleven-noisy-condition mean. The general density adjoint includes
the Ballarin probability derivative and its declared piecewise domain; omitting
that derivative would not optimize the stated mean-eleven objective.

The two-channel objective addressed a measured transfer deficit of a specialist
local-depolarizing objective. The first mean-eleven comparison had only 100 joint
updates at 1,100 constituent gradients, versus 550 for mean-two, so it could not
establish an inferior global objective. A separate 6,050-constituent-gradient
screen gave mean-eleven 550 joint calls and mean-two 3,025, with both L-BFGS and
exact Adam. Mean-eleven L-BFGS remained 0.159154 percentage points below the
strongest mean-two control at common CPU, with its main deficit on Haar. Neither
mean-eleven optimizer passed the fidelity or predeclared resource gate. This
budget is retired, not converted into a theorem about global optima.
[Long-ensemble analysis](../../output/overnight-state-preparation-20260913-0108/long-ensemble6050/ANALYSIS.json).

**Entropic risk.** Exponential-utility control predates this experiment; see
[Howard and Matheson (1972)](https://pubsonline.informs.org/doi/10.1287/mnsc.18.7.356)
and the quantum-control adaptation of
[Ge and Wu (2021)](https://arxiv.org/abs/2104.01323).
Our two-channel loss is `tau * log(mean(exp(loss_j/tau)))`; exact channel gradients
receive normalized exponential weights. This emphasizes the worse expected
channel, not rare stochastic trajectories. It is not an implementation of full
benchmark minimax control. The tested temperatures were 0.02 and 0.1, each with
L-BFGS and exact Adam. Temperature 0.1 L-BFGS gained only 0.107425 points in the
primary mean and 0.121830 in the mean worst-condition fidelity; both fall below
the declared 0.5-point routes. Temperature 0.02 lost about one mean point.
All tested risk settings were retired without a temperature rescue.
[Risk analysis](../../output/overnight-state-preparation-20260913-0108/risk-wave1/ANALYSIS.json).

## Weighted trajectories, control variates and geometry

**Weighted Pauli trajectories and objective-matched mixtures.**
[Abdelhafez, Schuster and Koch (2019)](https://link.aps.org/accepted/10.1103/PhysRevA.99.052327)
already combine quantum trajectories, stochastic optimization and improved
sampling that separates no-jump behavior. Our discrete Pauli-factor adaptation
integrates the no-selected-factor stratum deterministically and samples the
conditional remainder with exact weights, preserving multiple factors and their
cancellations. It is known stratification, not a new Monte Carlo principle.
The implemented depolarizing/dephasing mixture concatenates their separate
weighted ensembles with half weights, so weighted Adam and adaptive online train
the same mean-two objective as exact methods. Each receives 32 paths per channel,
fresh paired seed streams and the same structural repair opportunity. Equal path
or update counts are not equated with equal CPU. The earlier objective-matched
structural comparison narrowed the candidate's common-CPU lead over weighted
Adam to 0.462704 points with only two target wins; that weaker result remains
visible despite later composition gains.
[Mixture baseline analysis](../../output/overnight-state-preparation-20260913-0108/mixture-baseline1500/ANALYSIS.json).

**Exact-anchor control variates and SVRG Adam.**
[Johnson and Zhang (2013)](https://proceedings.neurips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html)
provides the snapshot-gradient plus paired stochastic-difference mechanism.
Our estimator adds the exact density gradient at an anchor to a weighted
trajectory difference evaluated with identical maps at current and anchor
parameters. It is unbiased for a parameter-independent path law. A sampled
snapshot instead has conditional error equal to its frozen snapshot error;
resampling corrections does not remove that error. Original finite-sum
convergence guarantees do not transfer automatically to persistent Adam on
this nonconvex expectation.

The nearby-anchor diagnostic substantially reduced gradient error times CPU,
but at displacement one the Haar/MPS cost advantage disappeared. The refresh-five
Adam screen reached 86.396072% mean noisy fidelity, versus 85.863267% for weighted
Adam and 86.379599% for exact Adam, while costing about 4.05, 2.89 and 3.91 training
CPU seconds respectively. It nearly reproduces the exact-gradient baseline;
it does not establish an advantage over that serious q6 control or useful
scaling. Snapshot, correction, replay and diagnostic costs remain separate.
[SVRG protocol and interpretation](SVRG_PROTOCOL.md),
[complete training panel](../../output/overnight-state-preparation-20260913-0108/svrg-refresh5/RESULT.json).

**Pure-state quantum-geometric preconditioning.**
[Stokes et al., Quantum Natural Gradient](https://quantum-journal.org/papers/q-2020-05-25-269/)
is the pure-state geometric precedent; compare the genuinely mixed-state
construction of [Koczor and Benjamin](https://arxiv.org/abs/1912.08660).
We form the exact pure-state tangent Gram matrix with the global-phase component
removed and solve a damped system against the noisy exact gradient. This is a
surrogate preconditioner, not the exact mixed-state natural gradient. Shared
parameter occurrences are combined before forming the metric. Metric formation,
linear solves and failed line searches are charged. Damping 0.001 and 0.01 did
not pass the common-CPU comparison; the smaller damping's difference from exact
Adam was -0.107855 points, and the larger damping was worse. Both tested schedules
are retired as general candidates.
[Geometry analysis](../../output/overnight-state-preparation-20260913-0108/geometry-wave1/ANALYSIS.json).

## Target-informed construction and compilation

**Conditional rank-two TT/MPS preparation.**
[Schön et al.](https://arxiv.org/abs/quant-ph/0501096) establishes sequential
entangled-state generation, and [Ran](https://arxiv.org/abs/1908.07958) connects
MPS encoding to physical one-/two-qubit circuits. Our implementation uses a
rank-two TT-SVD approximation, explicit isometry completion and scalar lowering;
its gates act on the physical zero state and incur all prescribed noise windows.
This uses the supplied classical target amplitudes, not an efficient unknown-state
oracle, and never supplies a target MPS as a free prepared input.

Unconditional rank two failed on Haar and lost the complete-panel comparison,
despite strong structured-target results. The subsequent frozen policy selects
the rank-two preparation only when its ideal target overlap is at least 0.99;
otherwise it executes the original legacy constructor. The rule uses overlap,
not target-family labels or noisy scores. Its rejected probe, synthesis, lowering,
construction and deployment costs are explicit. The conditional policy improves
structured cases while retaining the Haar fallback; it is an initialization and
resource contribution available to every optimizer. The lazy implementation
executes legacy construction only on fallback, avoiding unnecessary comparator
work in a deployed policy. It does not change the mathematical branch rule.
[Initializer policy](INITIALIZER_POLICY_PROTOCOL.md),
[development evidence](../../output/overnight-state-preparation-20260913-0108/initializer-policy/ANALYSIS.json).

**Algebraic and known-input compilation.** Periodic rotation simplification,
identity removal and matching-generator fusion are established compiler
operations. Our algebraic endpoint pass preserves the ideal unitary up to its
recorded phase, but changes the actual number of noisy gates; noisy benefit is
therefore measured separately. Known-input simplification has prior art in
[Häner, Hoefler and Troyer, assertion-based optimization](https://arxiv.org/abs/1810.00375)
and [Quantum Constant Propagation](https://cliplab.org/jd1mOkw/SAS2023/142840162/142840162.pdf).
The separate dense q1-q6 pass removes a gate only when it acts as a global phase
on the current ideal retained prefix from zero, using a fixed 1e-12 residual
threshold. It preserves all surviving gate metadata, scalar origins and phase/error
records and independently audits the result with Qiskit. It is not arbitrary-input
unitary equivalence and provides no a priori guarantee about noisy prefixes.

All 24 development source/compiled pairs completed. The pass removed 61 RZ gates
and scalar coordinates in 15 pairs, with no two-qubit removal, and increased the
complete noisy mean by 0.036813 points. Every target and source-label mean was
nonnegative; other than the relevant one-qubit depolarizing profiles, differences
were floating-point roundoff. Structural L-BFGS final endpoints gained only
0.008195 points. This supports a small common endpoint-compiler opportunity.
It does not demonstrate faster training: a coordinate that is phase-only at one
endpoint need not remain redundant when earlier parameters change. The subsequently completed comparison applied the same known-input compiler
before and after training to every role. Although candidate fidelity increased,
its strongest-control gain was only 0.377773 points with two of four target wins;
the compiled recipe was rejected under the rule frozen before its results.
The original recipe was retained for confirmation. This was development recipe
selection with different training streams and measured CPU endpoints, not an
isolated compiler treatment-effect estimate. Final compilation cannot
retroactively replace earlier common-CPU checkpoints.
[Paired endpoint compiler analysis](../../output/overnight-state-preparation-20260913-0108/state-preparation-compilation/ANALYSIS.json),
[complete compiled comparison authentication](../../output/overnight-state-preparation-20260913-0108/validation/COMPILED_COMPOSITION_AUTHENTICATION.json).

## What the evidence does and does not establish

The tested noise-continuation and standalone coordinate schedules, both geometric
damping settings, all tested entropic-risk settings, unconditional rank-two
construction, and the larger-budget mean-eleven hypothesis did not justify their
proposed general promotion. Nearby-anchor variance reduction remains useful
mechanistic evidence, with finite anchor lifetime and a strong exact-gradient
baseline. Negative results are evidence about these frozen settings, not proofs
that their entire method families cannot work.

Combining the conditional initializer, a mean-two objective, exact deletion and
repair, curvature and physical compilation produced a useful development recipe:
its original composition gained 0.720699 points over the strongest control with
four development-target wins at the primary CPU cutoff. The untouched five-target,
two-replica confirmation then failed. Its primary 88.534223% mean exceeds adaptive
online's 88.209294% by only 0.324929 points, with three of five target wins;
weighted Adam is also beaten on only three targets. TFIM para and MPS bond 3 lose.
The positive aggregate is concentrated in Haar; both other family means lose
against the strongest adaptive control.
The better full-cap result does not override the frozen primary rule. This is a
measured negative result for the selected complete method, not an optimizer
superiority or new external state-of-the-art claim.
[Complete confirmation](../../output/overnight-state-preparation-20260913-0108/confirmation-best-candidate/SUMMARY.json),
[final report](REPORT.md).

Its controls must receive the same compatible initialization, objective,
estimator, architecture and compilation opportunities. Where an upgraded Adam
control becomes the same algorithm, an impossible self-comparison is not required;
the valid claim concerns the objective/architecture/initialization contribution,
not invention of a superior Adam optimizer. Complete target-balanced outcomes,
strongest-control comparisons, chronological CPU endpoints and all failures take
precedence over individual favorable settings.

No cited external method has been faithfully benchmarked against this recipe
across the complete fixed q6/q12 conditions with matched construction, lowering,
training and endpoint costs. Literature names are provenance, not measured
competitors. MPS overlap and gradient checks are engineering validation; the
original failed parameter-agreement gate remains failed even though a separate
physical-state follow-up passes its newly declared gate. Neither that follow-up
nor dense q6 fidelity establishes a q12 speedup, tensor-network scalability,
hardware performance or state-of-the-art superiority. The later q6 memory audit
measures complete retained tapes for eight older development endpoints, and MPS
gradients are about 13.25 times slower than dense gradients at that size. It does
not establish a full-method memory advantage or an exceptional resource reason
to override failed confirmation. No larger numerical campaign followed.

The post-confirmation read-only history audit finds identical training-mean-two
and full-eleven advantage signs in all 60 paired case/panel comparisons against
weighted and adaptive controls. It therefore does not support blaming objective
surrogate mismatch for this failure; repair allocation and structural stopping
remain plausible mechanisms to test separately. This interpretation is our
inference from stored histories, not a claim from the cited literature.
All nine canonical q6 identities are now used. Any changed method remains
fixed-benchmark development until a separately authorized untouched panel is
available; new seeds alone do not create one.
