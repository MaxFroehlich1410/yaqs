# Paper-based noisy state-preparation competitors

The new [IQSP tensor-network adapter](IQSP_TN.md) is implemented and tested for
all nine fixtures under logical noise. It resolves the old KAK initialization
trap using the pinned author SU(4) exponential. Its eight non-Gaussian paths
are explicitly labeled extensions. Use `run_iqsp_tn` for new IQSP experiments;
the legacy dense `iqsp` entry described below remains diagnostic. No full
IQSP-TN performance results are available yet.

Future experiments compare **noise-aware training methods only**. The primary
external comparison is **NACL and RobustState on all nine six-qubit fixtures**,
with Fontana on the six real fixtures. AQCE and BAA-LRSP are excluded from future
performance experiments because their synthesis does not optimize with noise.
Their implementations and completed results remain available as historical
references; this scope decision does not change any previous ranking.

Fixed VQSP and unpruned layerwise remain optional internal noisy-training
controls, not external publication baselines. Legacy dense IQSP is **on hold**: the Gaussian
pilot exposed a zero-initialized KAK parameterization that keeps every
entangling coordinate zero. Correct and test that initialization before any
further IQSP performance experiment; more unchanged iterations do not fix it.
Its existing implementation is retained for diagnosis. See the
[Gaussian pilot report](../../output/gaussian-all-methods-pilot-20260920/REPORT.md).

Unsupported method/target pairs appear explicitly in
`summary.json.exclusions`. They are never assigned artificial zero fidelities.
The incumbent remains B + online Krotov + L-BFGS construction, pruning B, and
the unchanged weighted M=100 noisy refinement described in the
[current-method specification](../topdown_state_preparation_method.md).

These are local implementations of published algorithmic procedures with
documented benchmark adaptations. The new IQSP-TN adapter reuses the verified
author fork's numerical SU(4) routine; the complete published experiment driver
was not found. They do not reproduce the papers' hardware experiments. No additional
pruning from our method is applied to competitors. A successful smoke run or
passing tests establishes implementation behavior, not a fidelity ranking.

| Method         | Published procedure retained                                                                                                                 | Reused implementation                                                                     | Benchmark scope                                                                                                                                                                    |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NACL           | Exact noisy fidelity, structural proposals, identity insertion, removal, simplification, annealed acceptance, continuous OrthoMADS search    | Official PyNomadBBO 4.6.0; existing local density kernels and noise channels              | All nine targets; underspecified search choices are explicit settings. See [provenance](NACL_PROVENANCE.md).                                                                       |
| RobustState    | CNOT followed by local RZ/RX/RZ blocks, Hilbert-Schmidt loss, noisy-output/noiseless-Jacobian gradient, ideal then noisy Adam, final iterate | Existing tested Adam update, shared dense adjoint, Qiskit gates                           | All nine targets; exact-density feedback or finite-shot Pauli shadows. See [provenance](ROBUSTSTATE_PROVENANCE.md).                                                                |
| Fontana        | RY/CX layers, noiseless BFGS followed by noisy BFGS                                                                                          | SciPy BFGS, shared exact objective                                                        | Gaussian, three TFIM and both random MPS fixtures, real up to global phase. Six-qubit line extension and common deployment noise are explicit adaptations.                         |
| IQSP — on hold | Smooth Gaussian interpolation, uniform start, generic SU(4) blocks, Adam continuation, compilation followed by native noisy refinement       | Existing Adam, Qiskit SU(4) synthesis with audited exact fallback, shared dense objective | Gaussian only; local initialization must be corrected before further performance comparisons. Exact amplitudes replace tensor interpolation; no invented Haar-state interpolation. |

Fontana and IQSP have a separate [subset provenance](SUBSET_PROVENANCE.md).
In particular, IQSP's H2 angle-dependent ZZ noise and associated small-angle
removal rule are not transferred to a CX-based noise model. Its supported
comparison is explicitly the Gaussian procedure adapted to the common native
gate basis and channel, not an unchanged H2 implementation.

## Numerical contract

Target amplitudes come directly from the checked-in target JSON. The final
metric is always

\[
F=\langle t|\rho_C|t\rangle.
\]

The legacy implementations below train using exact density matrices at six
qubits. The separate IQSP-TN adapter uses pure and purified circuit tensor
networks. Native noise reuses the
existing synthetic IBM/Google single-jump Pauli distributions, with single-site
and correlated two-site terms. A channel follows **every native RZ/SX/X/CX**,
including RZ. These profile names are historical labels, not current hardware
calibrations. There is no idle, measurement, or readout noise in this comparison.

NACL searches directly in its native alphabet. Fixed ansatzes compile
symbolically using the shared compiler before noisy training. IQSP compiles
resolved learned SU(4) blocks at the transition to native noisy refinement.
The resulting native gate schedule is frozen. Final evaluation audits that
schedule without recompiling or canceling zero-angle gates, since those gates
carry noise windows in the trained objective. All outputs obey the same linear
topology and native evaluation channel. This schedule preservation differs from
post-training compilation of the older noiseless synthesis baselines.
The final audit's compiler field records the shared policy reference; it does
not claim that NACL's own native simplifier invokes the Qiskit compiler.

The shared `noisy_core.DenseObjective` reuses the local dense kernels from
`benchmarks/state_preparation/exact_density.py`. It provides the exact noisy
infidelity gradient and the noiseless density Jacobian contraction. For
RobustState specifically,

\[
L=\|\rho*{\rm feedback}-|t\rangle\langle t|\|\_F,
\qquad
g_i=\operatorname{Tr}\!\left[
\frac{\rho*{\rm feedback}-|t\rangle\langle t|}{L}
\frac{\partial\rho\_{\rm ideal}}{\partial\theta_i}\right].
\]

The feedback is held fixed in that derivative. Replacing it with the true noisy
fidelity gradient would change RobustState. `pauli_shadows` samples fresh
measurement feedback with a separate RNG and leaves the unbiased raw shadow
estimate unprojected; `exact_density` is an explicitly labeled simulator
feedback adaptation. Readout mitigation is unnecessary under the declared
zero-readout-error model. Hardware measurement costs are not inferred from
simulator timings; requested shadow shots are reported separately.
For shadow learning curves, plot `exact_loss_diagnostic` consistently: raw
`loss` is estimated from measurements during updates but uses exact feedback at
the final diagnostic point. RobustState's reported hardware advantage chiefly
concerns coherent errors; this Pauli-only comparison establishes performance
only for the declared incoherent noise profiles.

## Running and reproducibility

Use the isolated optional dependency set, without changing the main project
lock. `run_noisy` defaults to NACL and RobustState; add Fontana explicitly for
its supported targets. The older `run` driver now defaults only to the archived
incumbent, fixed VQSP and layerwise controls. AQCE/BAA require explicit historical
selection and are not part of the active experiment plan.

```sh
uv run --with-requirements experiments/state_preparation_competitors/requirements-noisy.txt \
  python -m experiments.state_preparation_competitors.run_noisy \
  --output output/noisy-paper-smoke \
  --methods nacl robuststate fontana \
  --depths 1 --seeds 300 --nacl-steps 0 2 --nacl-evaluations 8 \
  --construction-steps 2 --noisy-steps 2 \
  --validation-paths 512
```

This small smoke covers every eligible target under both profiles. These tiny
budgets intentionally do not establish scientific performance. Omit the update
overrides for method defaults, after choosing a justified search budget:

- RobustState: 500 noiseless and 50 noisy updates, learning rate 0.005.
- Fontana: 200 BFGS iterations per phase as a declared benchmark budget.
- IQSP (retained diagnostic defaults; performance runs on hold): 20 interpolation steps, 200 Adam updates **per step** as the declared
  construction budget, then 10000 noisy updates, learning rate 0.01. Its starting
  state is deterministic, so seed replicas are deduplicated per hardware profile.
- NACL: structural checkpoints 0/10/50 per restart, up to 100 black-box
  evaluations per inner solve. These are declared reconstruction settings,
  not quoted optimal settings from the publication.

The next performance pilot therefore retains the agreed TFIM-critical and
Haar-1 cases with our incumbent, NACL and RobustState; Fontana applies only to
TFIM. Fixed VQSP and layerwise may be included as internal controls. Do not
schedule AQCE, BAA-LRSP or the unresolved IQSP reconstruction. Keep the existing
training budgets, noise profiles and independent assessment rules when making
this scope change.

The manifest fingerprints sources, fixture, settings and pinned package
versions. Resume authenticates existing construction, selection and evaluation
bundles and rejects changed inputs. Scalar curves, seeds, parameters, timings
and circuits are retained using the existing compact binary history storage.
NACL saves each checkpoint interval's search history once. IQSP's different
parameter dimensions before and after native compilation have separate histories.

Independent sampled validation selects among the predeclared candidates. A
selection lock is written before the independent Qiskit density evaluator
produces final scores for every candidate. Exact final evaluation includes all
modeled histories and therefore has no trajectory sampling error. Optimizer
seeds and sampled-shadow feedback can still produce run-to-run variability.

Full grid construction time includes initialization, compilation, rejected
proposals and all refinement work. Validation and final assessment have separate
timings. Interrupted unsealed attempts are not recovered into cost totals. NACL
prefix checkpoints record cumulative and interval costs; do not add cumulative
costs together. Budget counts across methods are not interchangeable. Pareto
comparisons must use final fidelity, native two-qubit count/depth, and measured
total search cost, with target and noise profile held fixed.

## Verification

Tests include independent Qiskit channel propagation, finite-difference
gradients for complex states, the RobustState frozen-feedback derivative,
shadow unbiasedness, actual official OrthoMADS integration, structural identity
and simplification unitary oracles, real/Gaussian target restrictions, IQSP's
SU(4)-to-native transition, six-qubit execution, seed replay, compact storage
and authenticated resume.

To exercise all existing competitor adapters as well as the new methods, install
both optional dependency lists:

```sh
uv run --with-requirements experiments/state_preparation_competitors/requirements.txt \
  --with-requirements experiments/state_preparation_competitors/requirements-noisy.txt \
  python -m pytest -n 0 tests/optimization/test_state_preparation_competitors_*.py
```

Reported passing tests are numerical and software checks. They do not resolve
unpublished implementation choices or validate broader scaling claims from a
six-qubit study.
