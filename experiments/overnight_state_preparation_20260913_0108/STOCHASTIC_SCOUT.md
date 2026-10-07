# Stochastic optimization and robust-control scout

Research date: 2026-09-13. This catalogue was written before new scientific
experiments. The candidate ranking is a judgment of information value, not a
performance result. The complete required benchmark and findings were read;
all 18 target records and all amplitudes were parsed through the validated
fixture loader. Fixture SHA-256:
`d53cd272b6c665b3a92de2d6b1e335b85ef77f74016b0f20cf7ea6a0f486b839`.

## Existing evidence that constrains the search

The fixed 108-gate starts have no-selected-factor probability approximately
0.76584 under calibrated local depolarization. The existing `FactorSampler`
already integrates that entire stratum deterministically. Weighted R32 Adam
achieved 78.209294% versus 78.002904% ordinary R32 Adam on the earlier eight
development pairs; it also beat ordinary R128 Adam at lower measured CPU.
Those comparisons combine estimator and implementation improvements.

The subsequent structural study is a substantially stronger reference:
80.839281% structural Adam and 80.955015% structural adaptive-online Krotov at
matched 52.9-58.2 CPU-second case budgets. Krotov's aggregate lead came only
from its two Haar wins. A variance improvement alone does not establish a
better final state-preparation procedure.

`StatevectorBackend.gradient(theta, maps, weights=weights)` provides the
independent weighted infidelity gradient and physical map replay needed below.
`density_loss_gradient` provides an exact q1-q6 local-depolarizing gradient.
Both require parameter-independent probabilities for the claimed gradient.
They cannot be reused unmodified for angle-dependent Ballarin gradients:
those also require probability derivatives, including handling native angle
canonicalization and the benchmark's deletion threshold. The exact density
module currently does not implement every fixed dephasing/correlated profile.

## Ranked hypotheses

### 1. Refreshed anchor-gradient control variates

- **Hypothesis:** near a retained checkpoint, noise-map gradients at current
  and anchor parameters are sufficiently correlated that their difference
  beats ordinary weighted sampling at equal trajectory-direction work.
- **Field/source:** finite-sum stochastic optimization;
  [Johnson and Zhang, SVRG, NIPS 2013](https://proceedings.neurips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html).
  Adapt the anchor correction, not its strong-convexity guarantees.
- **Mechanism:** for a fixed anchor `a`, exact gradient `G(a)`, and fresh
  weighted physical maps `(omega_r,w_r)`, use
  `v(theta)=G(a)+sum_r w_r[g(theta,omega_r)-g(a,omega_r)]`.
  This is unbiased conditional on the anchor when `G(a)` is exact. At
  `theta=a` every correction cancels exactly. With an independently sampled
  anchor `Ghat(a)`, the conditional bias is exactly `Ghat(a)-G(a)`; resampling
  corrections cannot remove that frozen error. No global optimization
  convergence theorem is transferred to this nonconvex adaptive setting.
- **Exact change:** replace the gradient estimator only. Keep Adam, its
  moments, learning rate, circuit and endpoint rule unchanged.
- **First falsification:** use the four development starts and their saved
  update-10/update-25 parameters, anchor each at its start. At 24 independent
  correction seeds measure gradient MSE, cosine, descent fraction and
  MSE-times-CPU against the exact density oracle. Test R=8 and R=16 correction
  paths, paired against ordinary weighted R=16 and R=32 respectively. Include
  the exact anchor's full cost and report amortization at actual reuse counts
  1, 5 and 10. Do not call amortized hypothetical timings a measured speedup.
- **Promotion/falsification:** promote if target-balanced MSE-times-total-CPU
  drops by at least 2x in at least three targets and no target doubles its MSE.
  Retire if covariance is weak away from the anchor or costs dominate. If
  promoted, compare 100-update Adam with refresh periods 5 and 10; these are
  the entire compact screen, with final iterate endpoints.
- **Fixed/tunable:** benchmark noise and all initial circuits fixed; tune
  only R and anchor refresh interval. Exact and sampled anchors are distinct
  method identities. A scalable follow-up could use an independent R512
  weighted snapshot; its residual error must be measured and charged.
- **Costs/confounders:** correction uses 4R directional passes versus 2R for
  ordinary weighted gradients. Density anchor calls, setup, failed branches
  and snapshot memory are extra. On q6, exact Adam may itself be cheaper and
  more accurate. Structural changes invalidate the snapshot immediately.
- **Novelty/status:** known estimator adapted to exact Pauli trajectories.
  Implemented diagnostic in `variance.py`; numerical tests only so far.

### 2. Randomized QMC of the conditional first-event path

- **Hypothesis:** variation in the location/type of the first rare Pauli event
  dominates the conditional gradient variance and benefits from stratification.
- **Field/source:** numerical quadrature and variational inference;
  [Buchholz, Wenzel and Mandt, 2018](https://arxiv.org/abs/1807.01604).
  [SciPy Sobol documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html)
  specifies the available LMS-plus-shift implementation and power-of-two
  sampling constraints.
- **Mechanism:** sample scrambled Sobol vectors `u` of dimension M+1. Set
  first event `j=CDF_first^{-1}(u_0)` using the existing exact conditional CDF;
  set later events `mask_k=(u_(k+1)<q_k)` for k>j and no earlier events.
  Each point retains the correct conditional distribution; all multiple
  events and Pauli cancellations remain. Independently scramble each update.
- **Exact change:** only uniforms in `FactorSampler.draw` change. One
  deterministic no-factor path remains. R33 and R129 mean 32/128 conditional
  Sobol points; compare to IID weighted R33/R129, not the historical R32/R128.
- **First falsification:** same four starts and 24 independent scrambled
  replicates, 33 total paths; compare MSE-times-CPU and gradient covariance
  to IID weighted and one-dimensional stratified-first-event sampling. The
  latter uses `(permutation(arrange(N))+Uniform(N))/N` for the first coordinate
  and IID tail uniforms. All methods get identical oracle diagnostics.
- **Promotion/falsification:** require >=25% lower MSE-times-CPU on at least
  three targets and no increased aggregate MSE. Then a 100-update matched
  Adam comparison with globally frozen sampler; abandon if gains vanish.
- **Fixed/tunable/cost:** factor order is frozen gate/site/XYZ order; only
  N=32/128 screened. Cost adds Sobol setup O(MN) but no trajectory propagation.
  Charge setup rather than attributing cheap precomputation to the optimizer.
- **Expected failure/confounders:** discrete thresholds are nonsmooth, M is
  several hundred, and low-discrepancy order need not align with the important
  events. Smooth-integrand QMC rates do not apply. Uncertainty is across
  independent scrambles; individual Sobol points are not IID observations.
- **Novelty:** known QMC/stratification adapted to the existing conditional
  Pauli sampler. This should be offered equally to Adam and Krotov.

### 3. Noise-strength homotopy with a full-noise finishing stage

- **Hypothesis:** gradual deformation of the loss landscape can avoid a poor
  basin or a destructive initial noisy update, improving full-strength fidelity.
- **Field/source:** homotopy and stochastic optimal control;
  [Villanueva and Kappen, 2025](https://arxiv.org/html/2410.18635v2)
  study noise-strength annealing within Quantum Diffusion Control. Their
  path-integral control and control/noise matching assumptions are not being
  imported into this gate-circuit experiment; only the continuation idea is.
- **Mechanism:** train at integrated strengths `alpha(s)=s*alpha_ref`, ending
  at s=1. If constructing the existing calibration provider from errors, use
  `r1(s)=(1-exp(-4*s*alpha1))/2` and
  `r2(s)=4/5*[1-((1+3*exp(-4*s*alpha2))/4)^2]`. These are intermediate training
  objectives; final scoring uses the unchanged reference channel only.
- **Exact change/protocol:** 100 updates from each of the four starts.
  Candidate schedule s=0 for 10 updates, s=0.5 for 20, s=1 for 70. Compare
  exact Adam at full strength for all 100 and matched noiseless Adam; repeat
  with the same schedule under weighted Adam only if the exact mechanism
  succeeds. Preserve optimizer moments through transitions. Final iterate.
- **Promotion/falsification:** >=0.5 pp mean gain or a smaller gain in all
  four targets at equal total CPU. Retire after this single schedule if weak.
- **Fixed/tunable/cost:** no architecture changes and no target-specific
  schedules. One globally declared schedule; identical gradient-call budget.
  Evaluation/setup costs count toward equal CPU. Implementation cost low.
- **Expected failure/confounders:** starts were already constructed noiselessly,
  so continuation may simply waste full-noise refinement. A gain over the
  frozen start is not evidence for homotopy. The ending noise must exactly
  reproduce the benchmark, not a nearby lower-strength channel.
- **Novelty:** conventional continuation adapted to these frozen circuits.

### 4. Damped quantum-geometric preconditioning

- **Hypothesis:** redundant scalar rotations create coupled ill-conditioning
  that diagonal Adam scaling cannot resolve, especially on Haar targets.
- **Field/source:** information geometry/variational quantum algorithms;
  [Stokes et al., Quantum 4, 269 (2020)](https://quantum-journal.org/papers/q-2020-05-25-269/).
- **Mechanism:** compute noiseless state tangents `J_i=partial_i psi` and
  `S_ij=Re(<J_i|J_j>-<J_i|psi><psi|J_j>)`. Use
  `d=-(S+lambda*I)^(-1)*g_noisy`, with an exact full-noise line search. This
  pure-state Fubini-Study metric is a surrogate preconditioner, not the exact
  mixed-state quantum Fisher metric. Shared parameter occurrences must add
  before forming tangents; gauge/null directions require damping.
- **Exact change/protocol:** four development starts, 50 outer updates,
  globally declared lambda in {1e-3,1e-2}; maximum 5 backtracking trials and
  final-iterate endpoint. Compare exact Adam and exact L-BFGS at equal CPU
  and report equal-update numbers separately. Use exact gradients first to
  distinguish geometry from estimator variance.
- **Promotion/falsification:** >=0.5 pp mean or comparable fidelity with >=2x
  lower compute, without broad losses hidden by one Haar improvement. A
  singular solve or repeated line-search rejection is a recorded failure.
- **Fixed/tunable/cost:** only two damping constants; same initialization and
  gate family. A full tangent sweep costs O(L*P*2^n), Gram formation
  O(P^2*2^n), solve O(P^3), and storage O(P*2^n+P^2). Charge metric refreshes
  and line-search evaluations. Implementation cost moderate.
- **Expected failure/confounders:** the ideal metric may poorly describe
  dissipative sensitivity; damping can erase the geometry benefit. Strong
  baseline L-BFGS may capture useful curvature more cheaply.
- **Novelty:** known QNG with a clearly labeled noiseless-metric adaptation.

### 5. Pilot-controlled progressive weighted batches

- **Hypothesis:** late gradients require larger ensembles while early progress
  tolerates cheaper ones; a fixed R wastes work or creates a noise floor.
- **Field/source:** stochastic optimization;
  [Bollapragada, Byrd and Nocedal, 2018](https://arxiv.org/abs/1710.11258),
  and [progressive-batch L-BFGS, 2018](https://proceedings.mlr.press/v80/bollapragada18a.html).
- **Mechanism:** estimate the conditional-path gradient covariance `V` with
  a separate pilot. For direction d, require
  `(1-p0)^2*d^T V d/N <= eta^2*(ghat^T d)^2`. Choose a production batch from
  N in {15,31,63,127}; draw it independently of the pilot so adaptive sample
  selection does not silently create a stopped-sample gradient bias. Reuse
  the pilot only for batch choice, and charge all of it.
- **Exact change/protocol:** 100 weighted Adam updates, pilot size 8 every
  10 updates, eta=0.5 fixed, otherwise carry the chosen batch forward. Compare
  fixed weighted R32/R128 at equal CPU using the same endpoint rule.
- **Promotion/falsification:** retain only if a >=25% reduction in gradient
  MSE-times-CPU produces an endpoint benefit or a meaningful resource saving.
- **Costs/failure:** per-path gradient covariance needs an API returning
  individual contributions or small separate calls; either may be expensive.
  Noisy near-zero directional mean forces maximum batches. Pilot cost can
  erase savings; existing weighted R32 may already suffice.
- **Novelty/fairness:** known adaptive sampling. Give every optimizer the
  same production batch sequence in an estimator-isolation comparison.

### 6. Risk-sensitive training over fixed channel conditions

- **Hypothesis:** a compromise circuit can avoid a severe channel-specific
  failure without sacrificing target-balanced average fidelity.
- **Field/source:** robust control and adversarial optimization;
  [Ge and Wu, 2021](https://arxiv.org/abs/2104.01323) and
  [Ge et al., 2020](https://arxiv.org/abs/1909.02296).
- **Mechanism:** for unchanged benchmark-condition losses L*c, minimize
  `J_tau=tau*log(mean_c exp(L_c/tau))`; its gradient is
`sum_c softmax(L/tau)\_c*gradient(L_c)`. This approaches worst-channel loss
as tau decreases and equal-channel mean for large tau. Robustify over
complete channel expectations, not arbitrary rare trajectory branches.
A CVaR alternative is `min_z z+(1/(1-beta))\*E[(L-z)*+]`, but selecting only
  high-fidelity trajectories would optimize the wrong tail.
- **Exact change/protocol:** after all-channel oracle validation, compare
  mean-channel Adam against tau in {0.05,0.1}, 100 updates, on the four
  development targets. Include specialist local-noise Adam and structural
  baselines. The primary endpoint remains equal-weight mean benchmark
  fidelity; worst-channel and worst-family improvements are secondary.
- **Promotion/falsification:** require >=0.5 pp mean improvement or comparable
  mean with a predeclared >=2 pp worst-channel improvement. Retire if worst-case
  protection predictably trades away the primary mean with no resource gain.
- **Costs/confounders:** multiple exact-channel gradients per update, plus
  extending the current oracle correctly. Ballarin probability derivatives
  are essential. All methods must get the same multi-channel training budget.
- **Expected failure/novelty:** known risk-sensitive ensemble optimization;
  fixed benchmark noise gives no sampling-distribution generalization motive.
  A risk-neutral mean objective may already be the correct and strongest
  choice for the benchmark's primary metric.

## Ideas to defer or reject on mechanism grounds

- Subtracting the same deterministic no-jump gradient from every conditional
  trajectory and adding it back does not change within-stratum variance.
  A useful control variate must vary with the random path, such as an anchor
  gradient or precomputed first-event response with an analytic expectation.
- Plain antithetic Bernoulli pairs U and 1-U do not guarantee negative
  covariance of fidelity/gradient: quantum interference is not monotone in
  selected error factors. Treat this as a cheap diagnostic only after QMC.
- A first-event response surrogate can be Rao-Blackwellized, but enumerating
  all M single-event gradients costs approximately M trajectories per anchor;
  compare its actual amortized cost against the exact density anchor on q6.
- Multilevel truncation can be unbiased only with coupled exact corrections.
  Simply retaining at most one event changes the fixed channel. At q6 the
  deterministic exact oracle makes a multilevel implementation low priority.
- [Sharpness-aware minimization, Foret et al., 2020](https://arxiv.org/abs/2010.01412)
  would train at `theta+rho*g/||g||` with two gradient calls. The benchmark
  specifies stochastic Pauli channels, not parameter miscalibration, so
  parameter-flatness has no automatic physical robustness interpretation.
- SARAH-like recursive differences accumulate estimator error between
  refreshes. The explicit snapshot correction above is easier to audit first;
  no need to implement both before observing a variance advantage.

## Implemented diagnostic API and test results

`variance.build_snapshot(engine, anchor, calibration, snapshot_paths=0, seed=0)`
returns a snapshot. Zero paths selects the exact density gradient; positive
paths must be at least two and use the existing weighted sampler.

`variance.control_variate_estimate(engine, theta, calibration, snapshot,
paths=R, seed=...)` returns the signed corrected gradient and loss, cost and
bias metadata. The two gradients share every map and weight. A sampled
snapshot's per-path seed interval must be disjoint from correction seeds.
Snapshot costs remain explicit and separate from each correction estimate.

`variance.weighted_estimate(..., paths=2*R, seed=...)` is the matched-direction
ordinary weighted comparator. Actual CPU still must be measured: equal
trajectory-direction work is not automatically equal compute.

The low-level `control_variate_from_maps` supports analytic enumeration. Eleven
tests pass, including exhaustive all-pattern conditional expectation with
multi-factor cancellation, exact anchor cancellation for exact and sampled
snapshots, the sampled-anchor bias formula, disjoint streams, changed-problem
rejection, zero noise and explicit work accounting. Tests are not evidence of
optimization improvement. No scientific q6 estimator or optimizer run was
launched by this scout.
