# Independent literature scout

Prepared 2026-09-13. This is a ranked hypothesis catalogue, not experimental
evidence of an improvement. Statements about benchmark fit and experiment
outcomes expected below are the scout's inferences. Source claims are identified
separately. All numerical comparisons must use the unchanged benchmark vectors
and channels, physical preparation from zero, and lowered one-/two-qubit gates.

## Custody and interpretation

The complete user request, AGENTS.md, benchmark definition, target loader, and
all six required findings documents were read. The complete numerical fixture
was parsed and validated with the repository loader. All 18 vectors have finite
amplitudes, correct lengths, unit norm within 2.3e-16, and the prescribed phase.
All root metadata and each record's generation parameters were inspected.

- Benchmark SHA-256: `40685856bb052b88aab924792df74159a2b3bdf47d047adfb84183f5f4076f11`.
- Fixture SHA-256: `d53cd272b6c665b3a92de2d6b1e335b85ef77f74016b0f20cf7ea6a0f486b839`.
- Development means previously achieved are evidence of local headroom; the
  exact-Adam endpoint is not an upper bound. Existing structural Adam/Krotov
  comparisons are stronger than a 50-update fixed-Adam baseline.
- The reference calibrated local Pauli channel is parameter independent. N1
  Ballarin is angle dependent and prunes rotations at a finite threshold. These
  facts materially change valid derivatives and local optimization formulas.

The ranking considers information gained per implementation and compute cost.
It intentionally includes optimizer, circuit structure, estimation, geometry,
and robust-objective families. Suggested budgets are development proposals to
be frozen by the coordinator before execution, not permission to tune held-out
targets. A common first screen can use the four prescribed q6 development
targets, construction seed 11, fixed starts, exact final channel evaluation,
and the same endpoint rule for all arms. Use evaluation or local-kernel counts
as well as measured CPU when update costs differ.

## Ranked catalogue

### 1. Exact coordinate Rotosolve with reused density suffixes

**Source and original field.** Nakanishi, Fujii, and Todo, _Sequential minimal
optimization for quantum-classical hybrid algorithms_, PR Research 2, 043158
(2020), [original preprint](https://arxiv.org/abs/1903.12166); Ostaszewski, Grant,
and Benedetti, _Structure optimization for parameterized quantum circuits_,
Quantum 5, 391 (2021), [original paper](https://arxiv.org/abs/1905.09692).
The source mechanism is exact minimization of a trigonometric coordinate model.

**Benchmark adaptation.** A single occurrence of a supported rotation gives
`F(delta)=a+b*cos(delta)+c*sin(delta)` even with arbitrary fixed surrounding
linear channels. Fit with three local contractions of an updated density prefix
against the unchanged noisy observable suffix; choose `atan2(c,b)`. This avoids
learning-rate and diagonal scaling choices. The eight currently supported
one-angle gates have suitable two-eigenvalue generators, including P and CP.

**Fit/conflicts.** Excellent for the existing exact local-Pauli oracle. Shared
parameters require a richer Fourier model. Ballarin's changing channel weights
invalidate this three-point fit. Zero-angle retained gates still incur noise.

**Smallest falsification and cost.** Ten exact sweeps on the four starts versus
Adam at equal local propagation cost, plus equal-CPU curves. Reject if exact
fidelity fails monotonicity, or improvement per CPU is consistently worse. Low
implementation cost; known method, adapted density implementation. Coordinate
stationary traps and parameter coupling are the expected failure modes.

### 2. Exact noisy L-BFGS, with line-search costs fully charged

**Source and original field.** De Fouquieres, Schirmer, Glaser, and Kuprov,
_Second order gradient ascent pulse engineering_, JMR 212, 412-417 (2011),
[original preprint](https://arxiv.org/abs/1102.4096). The authors combine accurate
control gradients with BFGS curvature to accelerate pulse optimization.
The underlying limited-memory algorithm originates in Nocedal, _Updating
Quasi-Newton Matrices with Limited Storage_, Mathematics of Computation 35,
773-782 (1980), and Liu and Nocedal, _On the limited memory BFGS method for
large scale optimization_, Mathematical Programming 45, 503-528 (1989):
[original author's software and reference page](https://users.iems.northwestern.edu/~nocedal/lbfgs.html),
[1989 journal article](https://link.springer.com/article/10.1007/BF01589116).

**Mechanism and fit.** Apply limited-memory secant curvature to the repository's
exact density loss/gradient and use a deterministic line search. It directly
tests whether Haar deficits arise from conditioning and unfinished refinement.
Keep starting circuits fixed before combining it with structural search.

**Conflicts.** Smooth fixed-channel assumptions hold locally. Ballarin needs its
actual channel derivative and piecewise/discrete treatment near zero. Ordinary
BFGS curvature is unreliable with freshly sampled noisy gradients.

**Smallest falsification and cost.** Budget 100 exact loss-gradient calls per
start against 100 exact-Adam calls; also compare at common CPU. Record line-search
failures, gradient norm, and calls rather than only iterations. Very low code
cost using SciPy, or low cost for an explicitly resumable two-loop solver.
Retire if a fair converged comparison gives no advantage.
Known optimizer adapted from control; expected failure is convergence to the
same poor basin or loss of savings to line searches.

**Implementation refinement before execution.** The coordinator will use a
deterministic resumable two-loop L-BFGS implementation with Armijo backtracking
rather than SciPy L-BFGS-B, so curvature history can be authenticated and resumed
exactly. This is an implementation choice, not an optimization novelty. It
needs a descent-direction safeguard, curvature-pair acceptance rule, and frozen
line-search constants. Armijo alone does not guarantee positive secant curvature,
so pairs with invalid curvature must be skipped or explicitly damped. Report
the actual variant rather than claiming equivalence to the full L-BFGS-B solver.

### 3. Curvature-assisted repair after exact finite deletion

**Source and original field.** Hassibi and Stork, _Second order derivatives for
network pruning: Optimal Brain Surgeon_, NeurIPS 1992,
[original proceedings](https://papers.neurips.cc/paper_files/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html).
OBS uses inverse-Hessian information to compensate other parameters when
removing a weight.

**Adapted mechanism.** For deletion of coordinate k near a stationary point,
the quadratic constrained repair is proportional to
`-theta_k * Hinv[:,k]/Hinv[k,k]`. Use a damped positive curvature approximation
only to initialize repair; rank and accept actual gate-plus-noise deletion with
the exact channel. The known Haar deficit makes repair quality informative.

**Fit/conflicts.** Noise-window removal is a finite model change; Hessian
saliency alone cannot correctly rank it. Rotation periodicity, zero modes, and
nonstationary starts require care. Native gates, not parameter zeros, count.

**Smallest falsification and cost.** Compare two shortlisted deletions with
ordinary repair versus curvature-initialized repair under identical refinement
calls. Medium code cost; no extra broad search. Reject if true repaired fidelity
falls or curvature cost dominates. Known cross-field adaptation, with no novelty
claim. Main failure is a poor quadratic approximation across the deletion.

### 4. Reversible structural search with an explicit no-deletion option

**Source and original field.** Bilkis et al., _A semi-agnostic ansatz with
variable structure for quantum machine learning_,
[VAns preprint](https://arxiv.org/abs/2103.06712), published under the title
_A semi-agnostic ansatz with variable structure for variational quantum
algorithms_ (2023). VAns grows and removes gates while optimizing parameters.

**Mechanism and fit.** Keep a small beam of original, deleted, and repaired
circuits. Reinsert one supported entangler where deletion harms expressivity.
Score complete finite circuits with exact noise. Letting structure stay fixed
prevents a fixed deletion quota from forcing Haar losses.

**Conflicts.** Identity insertion adds physical noise unless lowering removes
the gate; do not assume an identity-initialized block is noiseless. Different
branches require full cost accounting and independent checkpoint rules.

**Smallest falsification and cost.** Beam width two, at most two structural
rounds, one local insertion candidate and one deletion shortlist, same total
repair budget as greedy deletion. Medium implementation cost. Reject if the
Pareto frontier does not improve. Known architecture-search adaptation; main
failure is beam cost outweighing fewer greedy mistakes.

### 5. Gate sparsity with coordinate or proximal updates

**Source and original field.** Raymond and He, _Optimizing Parameters of Quantum
Circuits with Sparsity-Inducing Coordinate Descent_, IJCAI 2025,
[original paper](https://www.ijcai.org/proceedings/2025/0680.pdf). Their Rotolasso
combines trigonometric coordinate fitting and L1 sparsity, including group
extensions; this is direct prior art for coordinate-plus-sparsity proposals.

**Mechanism and fit.** Optimize noisy infidelity plus an entangler penalty,
then explicitly lower/delete selected gates and refine without the penalty.
An L1 surrogate promotes zeros; an exact gate-present/gate-absent comparison
better matches the discontinuous N2/N3 gate cost. Protected initial preparation
must still be counted and simulated.

**Smallest falsification and cost.** Two fixed regularization strengths and one
unregularized control, equal calls including cleanup. Compare noisy-fidelity
versus two-qubit counts. Low-to-medium cost if coordinate code exists. Failure:
angle shrinkage pays the same gate noise until actual deletion, or removes
expressive Haar gates. Known method and adaptation, not novel pruning.

### 6. Mixed-state natural gradient or Hilbert-Schmidt preconditioning

**Source and original field.** Koczor and Benjamin, _Quantum natural gradient
generalised to noisy and non-unitary circuits_, PRA 106, 062416 (2022),
[original paper](https://arxiv.org/abs/1912.08660). The source extends QNG to mixed
states through the quantum Fisher metric and discusses Hilbert-Schmidt
approximations. The pure-state block-metric precursor is Stokes et al.,
[Quantum 4, 269](https://quantum-journal.org/papers/q-2020-05-25-269/).

**Mechanism and fit.** Solve a damped metric system `G delta=-eta*g`, initially
per layer or gate block. Geometry can distinguish redundant parameter motion
from useful physical-state changes.

**Conflicts.** A pure-state metric used with noisy loss is a heuristic
preconditioner, not the exact noisy natural gradient. Rank deficiency, damping,
and metric-estimation costs matter.

**Smallest falsification and cost.** At one frozen checkpoint per target,
compare predicted directions and exact finite-step improvement against Adam
under two damping levels; then only promote if improvement per total work is
credible. Medium cost. Known adapted method. Main failure is near-null metric
directions amplifying errors while the true loss remains poorly conditioned.

### 7. Exact-anchor SVRG for fresh trajectory updates

**Source and original field.** Johnson and Zhang, _Accelerating Stochastic
Gradient Descent using Predictive Variance Reduction_, NeurIPS 2013,
[original proceedings](https://papers.nips.cc/paper_files/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html).
SVRG subtracts a sampled reference gradient and restores its full mean.

**Adapted mechanism.** Use
`g_hat(theta,omega)-g_hat(anchor,omega)+g_exact(anchor)`, refreshing the anchor
after a short epoch. Paths refresh across updates but are paired inside each
difference. This addresses finite-ensemble overfit without discarding CRN's
local cancellation benefit.

**Fit/conflicts.** Valid under parameter-independent path laws. An approximate
anchor mean adds uncertainty and must be accounted for. At q6, exact gradients
may be cheaper than this estimator, so q6 is mainly a diagnostic oracle.

**Smallest falsification and cost.** Measure error-times-CPU at starting and
nearby parameters with 32 paths and 8 independent batches; compare weighted
sampling with the same additional work. Medium cost. Retire if coupling fails
after a few steps. Known cross-field adaptation; finite-sum convex guarantees
do not transfer automatically to this nonconvex expectation.

### 8. Low-factor-count control variate for exact Pauli trajectories

**Source and original field.** Peherstorfer, Willcox, and Gunzburger, _Optimal
Model Management for Multifidelity Monte Carlo Estimation_, SIAM SISC 38
(2016), [author manuscript](https://kiwi.oden.utexas.edu/papers/multi-fidelity-monte-carlo-peherstorfer-willcox-gunzburger.pdf).
The source combines correlated cheap and expensive model outputs while
optimizing sample allocation by correlation and cost.

**Adapted mechanism.** A first-order path surrogate is
`C(omega)=g0+sum_m I_m(omega)*(g_single_m-g0)`, whose expectation is known using
fixed factor probabilities. Estimate `E[C]+mean(g_exact_path-C)` so all
multi-factor interactions remain in the correction. Sparse factor events give
a concrete low-noise rationale.

**Conflicts.** Subtracting the same deterministic no-jump gradient from every
sample alone does not reduce variance. Computing every single-factor response
could dominate cost; adjoint/local contractions or a selected subset are needed.

**Smallest falsification and cost.** One checkpoint per target, compare gradient
MSE-times-CPU against weighted R32 over eight batches. Medium-high code cost.
Known multifidelity principle; this specific circuit surrogate is a potentially
useful adaptation, with novelty unestablished. Failure: strong multi-event
interactions or expensive surrogate construction.

### 9. Gradient-signal-based progressive batching

**Source and original field.** Byrd, Chin, Nocedal, and Wu, _Sample Size Selection
in Optimization Methods for Machine Learning_, Mathematical Programming 134
(2012), [author manuscript](https://users.iems.northwestern.edu/~nocedal/PDFfiles/dss.pdf).
The source varies batch size using gradient variance and analyzes sampled
second-order optimization.

**Mechanism and fit.** Increase trajectory count when estimated gradient-noise
norm becomes a large fraction of the gradient norm, retaining weighted
no-factor stratification. Start cheaply when the deterministic signal is large;
spend paths near convergence.

**Smallest falsification and cost.** Use a predeclared batch ladder 16/32/64/128,
one variance ratio, and the same total trajectory passes as weighted Adam.
Track exact gradient cosine as a diagnostic. Low code cost. Known adaptation.
Failure: noisy variance estimates force large batches too early, or Adam's
adaptivity already tolerates the error. This cannot cure the demonstrated
Krotov scaling deficit by itself.

### 10. Joint-channel ensemble optimization and worst-channel trust regions

**Sources and original field.** Chen et al., _Sampling-based learning control of
inhomogeneous quantum ensembles_, [2013 preprint](https://arxiv.org/abs/1308.1454);
Kosut, Grace, and Brif, _Robust control of quantum gates via sequential convex
programming_, [2013 preprint](https://arxiv.org/abs/1301.3235). The latter studies
worst-case fidelity under modeled uncertainty with resource constraints.

**Adapted mechanism.** Optimize the equal-weight mean of the fixed benchmark
channels, or maximize a local lower bound to their minimum within a trust
region. Recompute true channel scores for acceptance. This aligns a single
delivered circuit with evaluation across several conditions.

**Fit/conflicts.** Benchmark channels are fixed categories, not unidentified
device uncertainty. Minimax can sacrifice mean fidelity, so mean-trained and
worst-trained methods have different hypotheses. Never modify test strengths.

**Smallest falsification and cost.** Four development targets, two objectives,
same total exact channel evaluations; assess per-channel tradeoffs and mean.
Medium cost, especially once all exact evaluators are validated. Known adapted
control. Failure: the hardest channel dominates and unnecessarily harms others.

### 11. Noise homotopy as a training curriculum

**Source and original field.** Bengio, Louradour, Collobert, and Weston,
_Curriculum Learning_, ICML 2009,
[conference record](https://mlanthology.org/icml/2009/bengio2009icml-curriculum/).
The source interprets easier-to-harder objectives as a form of continuation.

**Adapted mechanism.** Follow minimizers of channels with scaled training
strengths `s*alpha`, with `s` increasing from zero to one, then finish with
the unchanged full-strength channel. This is allowed as a labelled training
surrogate and does not change the final benchmark.

**Fit/conflicts.** Starts are already noiseless-trained, so additional low-noise
work may repeat solved optimization. Increasing noise need not make the
landscape harder, and no continuation guarantee is inherited.

**Smallest falsification and cost.** A frozen three-stage schedule versus direct
full-noise optimization, same total calls and final full-noise tail. Very low
code cost. Known continuation principle adapted here. Retire on no consistent
gain; likely failure is wasting scarce refinement steps or following the wrong
basin.

### 12. Scrambled quasi-Monte Carlo conditional paths

**Source and original field.** Owen, _Randomly Permuted (t,m,s)-Nets and
(t,s)-Sequences_, 1995; [original Stanford report](https://statistics.stanford.edu/technical-reports/randomly-permuted-t-m-s-nets-and-t-s-sequences).
The source randomizes low-discrepancy sampling while preserving stratification.

**Adapted mechanism.** Transform scrambled Sobol coordinates into conditional
Pauli-factor draws, rescrambling per update. Prefer factor ordering by influence
and keep the deterministic no-factor branch. Use independent scrambles as the
replication unit for error estimates.

**Fit/conflicts.** Rare Bernoulli thresholds are discontinuous and effective
dimension may be large; smooth-integrand rates cannot be claimed. Conditional
sampling must preserve the exact factor distribution and cancellations.

**Smallest falsification and cost.** Eight independent scrambles at 32/128 paths
on one checkpoint per target, paired against weighted pseudo-random sampling.
Low-to-medium cost. Known estimator adapted to circuit paths. Failure: no
effective-dimensionality reduction or implementation overhead erases benefit.

### 13. Parameter sharpness-aware optimization as a low-priority negative control

**Source and original field.** Foret et al., _Sharpness-Aware Minimization for
Efficiently Improving Generalization_, [2020 original preprint](https://arxiv.org/abs/2010.01412).
SAM approximately optimizes the worst loss in a parameter neighborhood.

**Adapted mechanism.** Apply a bounded adversarial parameter perturbation before
the descent step, using exact noisy fidelity. This could favor broader basins.

**Conflict.** Parameter perturbations model coherent control error, whereas
the fixed benchmark contains specific Pauli channels. Parameter flatness is
not equivalent to stochastic-Pauli robustness; the main rationale is basin
selection, not correcting the benchmark noise.

**Smallest falsification and cost.** One fixed radius against exact Adam with
the same gradient-call budget. Low code cost, approximately two gradient calls
per update. Known cross-field adaptation. Expected failure is sacrificing true
fidelity for robustness to an error absent from the benchmark. Do not promote
on parameter-noise tests that replace the benchmark objective.

## Relevant state-preparation competitors and exclusions

- [Ballarin et al., _Efficient quantum state preparation of multivariate
  functions using tensor networks_](https://arxiv.org/abs/2511.15674) is especially
  relevant for the Gaussian and angle-dependent N1 model. Its tensor-network
  synthesis and native entangler accounting are a meaningful external baseline;
  a general claim needs a faithful implementation and complete resource cost.
- [_Scalable Preparation of Matrix Product States with Sequential and Brick
  Wall Quantum Circuits_](https://arxiv.org/abs/2602.12042) directly overlaps the
  benchmark's MPS/BMPD construction family. Compare state preparation from zero
  after supported-gate lowering, not merely a target MPS representation.
- [Long et al., PRA 109, 042413 (2024)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.109.042413)
  distinguish benefits of reducing depth under global noise from reducing gate
  count under local depolarizing noise. For this benchmark, reordering disjoint
  gates without reducing noisy windows has no automatic noise benefit.
- [Agrawal et al., _Cheaper and more noise-resilient quantum state preparation
  using eigenvector continuation_](https://arxiv.org/abs/2406.17037) provides
  an external subspace method. A reconstructed lower energy is not itself the
  physical target state. Its coherent combination must be prepared, lowered,
  and subjected to all noise before comparison; absent that, exclude it from
  the fidelity scoreboard.
- [Abdelhafez, Schuster, and Koch, PRA 99, 052327 (2019)](https://link.aps.org/accepted/10.1103/PhysRevA.99.052327)
  already combine quantum trajectories, stochastic optimization, and improved
  sampling that isolates no-jump behavior. The repository's weighted no-factor
  idea has clear prior art; it is not a novelty contribution.
- [QuantumNAS, original project](https://hanlab.mit.edu/projects/quantumnas)
  includes noise-adaptive circuit search and iterative gate pruning/refinement.
  A broad claim of inventing noise-aware architecture search is therefore
  unsupported. Its complete search/mapping procedure is a larger external
  implementation project, not a free comparator obtained by borrowing its name.

## Implemented adaptation and limits

`coordinate.py` implements only the first candidate's fixed-structure sweep.
Its public API is `sweep(circuit, theta, target, calibration)`. It returns a
copied updated vector and exact initial/final fidelity, each coordinate's true
local scores, kernel-operation counts, stored-tape array bytes, and measured
CPU/wall time. It accepts the exact `CalibratedLocalDepolarizingProvider` type,
supports one to six qubits and all eight currently supported one-angle gate
types, and rejects shared parameters, data maps, nonfinite data, and zero scales.
Nonunit/negative scales, offsets, unused parameters, fixed gates, and disabled
noise windows are handled. Zero-angle gates remain present and noisy.

The adaptation stores a backward noisy observable tape and then updates the
density prefix sequentially. Later gates retain their original parameters,
making each local subproblem exact. This is different from reusing an old
costate with a small gradient step: the full coordinate maximum is contracted
and explicitly checked. It is still Rotosolve, not a Krotov-specific method.

Validation uses an independent full-circuit Qiskit channel implementation for
each sequential coordinate on zero, calibrated, and strong-noise examples with
all gate generators. Additional tests cover immutability, unsupported-domain
rejection, flat coordinates, retained identity noise, and multi-sweep
monotonicity. These are numerical correctness tests, not evidence of competitive
benchmark fidelity or scalability. The coordinator owns experimental screening,
all method selection, full lint, and final conclusions.

`estimator_diagnostic.py` implements entry 7's exact-anchor lifetime experiment.
The CLI preregisters the four development construction-seed-11 starts before
any scientific numerical work. Each target has one deterministic normalized
parameter-space direction and offsets 0, 0.05, 0.2, and 1.0. At each offset,
32 independent replications pair CV32 with weighted64 at 128 directional
passes per method. All seeds use the separate `development_diagnostics`
domain; consecutive per-path seed intervals are checked for overlap across
the complete panel. Method order alternates within the paired replications.

The complete panel costs four exact anchor gradients, twelve displaced exact
oracle gradients, and 65,536 directional passes for each sampled method.
Exact anchor and displaced oracle costs are kept distinct from estimator costs.
Anchor cost is charged once per target and retained on resume. The diagnostic
reports raw gradients, gradient norms, MSE, relative error, cosine similarity,
MSE times CPU, and the smallest strict snapshot break-even reuse count K under
`MSE_cv * (CPU_cv + CPU_anchor / K) < MSE_weighted * CPU_weighted`.
It also reports whether CPU alone amortizes, which is a different criterion.
The exact cancellation at distance zero is a correctness check, not evidence
of useful variance reduction away from the anchor. Ratios with zero reference
MSE and directions with zero gradient norm are reported as undefined.

Checksummed protocols, anchor records, paired-replication checkpoints, and final
ledgers authenticate the benchmark, selected source starts, and numerical source
files. Completed numerical work is reused exactly. A failed or interrupted
uncommitted pair must be recorded separately in the coordinator's run log;
its partial work cannot be recovered from a completed-pair checkpoint.
Small q2 tests verify raw estimator identities, oracle metrics, strict
amortization arithmetic, rejection of overlapping randomness, frozen protocol
identities, and byte-identical complete-panel resume. They do not run the
development panel or constitute scientific evidence for estimator efficiency.

Prepare only with `uv run python -m
experiments.overnight_state_preparation_20260913_0108.estimator_diagnostic
--prepare-only`. The same command without `--prepare-only` executes or resumes
the frozen serial panel. Single-thread BLAS limits must be set by the runner.

## Candidate-registry provenance audit

The coordinator's `CANDIDATES.md` was reviewed before candidate execution. All
ten candidates map to known families or explicitly labelled adaptations:

| Registry candidate               | Primary provenance in this catalogue                          | Novelty assessment                                                                  |
| -------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| 1 Exact noisy L-BFGS             | Entry 2: Nocedal/Liu, control connection de Fouquieres et al. | Known optimizer; resumability and exact noise oracle are implementation adaptations |
| 2 Exact Rotosolve                | Entry 1: Nakanishi et al., Ostaszewski et al.                 | Known coordinate rule; local density implementation                                 |
| 3 Adaptive exact deletion/repair | Entry 4 VAns; existing repository exact scorer                | Adapted structural search; exact noisy finite ranking must be described precisely   |
| 4 Anchor control variate         | Entry 7 Johnson/Zhang SVRG                                    | Known variance reduction with an exact expectation anchor                           |
| 5 Noise homotopy                 | Entry 11 curriculum/continuation                              | Known strategy applied to noise strength                                            |
| 6 Group sparsity                 | Entry 5 Raymond/He Rotolasso                                  | Strong direct prior art; no novel sparsity claim                                    |
| 7 Pure-state QGT preconditioner  | Entry 6 Stokes et al.; contrast Koczor/Benjamin               | Known surrogate preconditioner, not exact mixed-state QNG                           |
| 8 Mean/minimax channels          | Entry 10 ensemble and robust control                          | Known objective family adapted to fixed benchmark channels                          |
| 9 Stratified randomized QMC      | Entry 12 Owen                                                 | Known sampling design with a conditional-path adaptation                            |
| 10 Insertion/deletion            | Entry 4 VAns; QuantumNAS comparison                           | Known architectural search                                                          |

No candidate name currently asserts a novel algorithm or state-of-the-art
performance. The following details should be frozen unambiguously before runs:

1. Registry candidate 1 must record the resumable Armijo solver in place of its
   earlier SciPy L-BFGS-B wording and define its curvature safeguards.
2. Noise homotopy must say whether the schedule scales average errors r or
   integrated coefficients alpha. Those are nonlinear transforms; either is a
   valid declared training curriculum but their identities must not be confused.
3. Candidate 3's retained branch requires identical repair work and endpoint
   selection. Comparing a repaired deletion to an unrepaired original confounds
   architecture with refinement.
4. The all-condition fan-out should report the mean across the eleven noisy
   conditions separately from noiseless fidelity; including noiseless in a
   quantity called mean noisy fidelity changes the primary metric.
5. Historical best-checkpoint structural endpoints can be contextual references,
   but a new final-iterate arm cannot fairly claim superiority over or inferiority
   to them as if endpoint rules were matched.

The initial registry explicitly calls the first wave a mechanism screen with
different work budgets and requires subsequent matched comparisons. That is an
appropriate limit. Final reporting must preserve this distinction and charge
line-search and rejected structural evaluations as well as accepted updates.
