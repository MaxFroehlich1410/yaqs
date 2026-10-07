<!-- Entries in each category are sorted by merge time, with the latest PRs appearing first. -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on a mixture of [Keep a Changelog] and [Common Changelog].
This project adheres to [Semantic Versioning], with the exception that minor releases may include breaking changes.

## [Unreleased]

- added independent authenticated fixed-depth and growing-depth Haar capacity checks targeting 99.9% ideal fidelity with the unchanged statevector optimizers and explicit resource bounds ([**@MaxFroehlich1410**])

### Added

- made the historical plot scripts portable and safe to import, with PNG smoke tests ([**@MaxFroehlich1410**])

- published the experimental source, supporting tests, protocols, and manuscript source while keeping generated experiment and manuscript artifacts local ([**@MaxFroehlich1410**])

- added a matched eighteen-case twelve-qubit sequential-C pruning continuation, reusing the structured ideal endpoints and unchanged noisy refinement and assessment budgets for comparison with simultaneous B ([**@MaxFroehlich1410**])

- added authenticated twelve-qubit pruning and noisy continuation from saved ideal endpoints, with unchanged B pruning, 100 online and 100 batch updates, sealed phase recovery and independent paired final assessments ([**@MaxFroehlich1410**])

- added a separate authenticated 100-online/100-batch sequential-pruning refinement experiment with direct longer-budget prefix checks and explicit checkpoint-search CPU accounting ([**@MaxFroehlich1410**])

- added authenticated noisy refinement of sequential-ideal checkpoints with 100 online sweeps and 200 batch updates using the unchanged cross-trajectory signal and a shared fixed noise bank ([**@MaxFroehlich1410**])

- added an independent eleven-arm six-qubit pruning comparison on authenticated unpruned checkpoints, with isolated noiseless repair, exact fidelity replay and no noisy fine-tuning ([**@MaxFroehlich1410**])

- added a matched 27-case unpruned noisy fine-tuning ablation reusing the exact three-seed six-qubit incumbent checkpoints, with sealed recovery and independent exact endpoint replay ([**@MaxFroehlich1410**])

- added a twelve-qubit best-of-1,000 product-start comparison targeting 99% by depth 30 with unchanged longer hybrid updates, sealed sampling costs and reuse of both historical initialization controls ([**@MaxFroehlich1410**])

- added isolated continuation of all eighteen completed Haar initialization cases from authenticated depth-16 parameters to at most depth 30, preserving the original optimizer and separately accounting for historical and new costs ([**@MaxFroehlich1410**])

- added zero-protected best-of-1,000 product-state initialization with batched exact overlaps, sealed sampling provenance and a matched six-qubit noisy comparison driver preserving B pruning and weighted updates ([**@MaxFroehlich1410**])

- added a complete twelve-qubit initialization comparison with 48 fresh paired runs, authenticated reuse of six random-MPS runs and separate overlap-distribution diagnostics, preserving the tested numerical updates ([**@MaxFroehlich1410**])

- added a matched random-MPS initialization experiment with nine statevector arms, longer per-depth L-BFGS refinement, exact 99% stopping, compact sealed histories and recovery tests ([**@MaxFroehlich1410**])

- documented the completed twelve-qubit Haar refinement result in the current method description: longer fixed-depth optimization can reduce circuit depth at 99% ideal fidelity, with explicit costs and unresolved 99.9% attainment ([**@MaxFroehlich1410**])

- added authenticated fixed-depth Haar refinements at depths 66 and 87 with continuous L-BFGS, exact threshold-crossing costs, compact histories and independent statevector replay ([**@MaxFroehlich1410**])

- added authenticated statevector depth continuations for matched Gaussian and Haar checks, preserving the original optimizer and distinguishing shared MPS parents from new vector training ([**@MaxFroehlich1410**])

- added isolated noiseless depth continuation from authenticated unpruned checkpoints, with scale-aware ideal patience, explicit resource exhaustion, exact statevector replay and preserved optimizer settings ([**@MaxFroehlich1410**])

- added a frozen twelve-qubit logical-noise campaign with 400-basis RobustState feedback, independent depth jobs, capacity-aware Leonhard distribution, authenticated staging and guarded resume; launching remains explicit ([**@MaxFroehlich1410**])

- clarified RobustState's noise-dependent refinement benefit in the manuscript discussion, citing the original incoherent-noise simulator result without claiming general competitor failure ([**@MaxFroehlich1410**])

- added a four-qubit RobustState validation control with the full published training and measurement budgets, matched exact-feedback diagnostics, and independently scored compact histories ([**@MaxFroehlich1410**])

- added a qualified manuscript discussion of twelve-qubit IQSP resource diagnostics and Pauli-noise representation costs, with source provenance and no claim of comparative scaling ([**@MaxFroehlich1410**])

- added explicit RobustState diagnostic basis budgets and independent trial ranges with preserved seeds, unchanged warm parameters and recorded parameter/moment hashes for remote checks ([**@MaxFroehlich1410**])

- added phase-resolved Ballarin memory diagnostics with unchanged forward propagation, exact SVD-input capture, isolated replay and a supervised disk-space guard ([**@MaxFroehlich1410**])

- added bounded twelve-qubit compression and post-ideal-training measurement audits, with unchanged competitor updates, preserved Adam moments, paired gradients, compact checkpoint storage, streamed SVD derivative accumulation and repeated-feedback uncertainty ([**@MaxFroehlich1410**])

- added pinned published two-site TCI construction for compressed IQSP targets, independent fixture accuracy checks, complete setup-cost receipts and unchanged training/noise policies ([**@MaxFroehlich1410**])

- added explicit twelve-qubit competitor support with exact density checkpoint recomputation, statevector ideal derivatives, streamed logical-noise assessment and unchanged-initialization Ballarin startup audits ([**@MaxFroehlich1410**])

- added an optional Ballarin compressed comb backend with pinned author kernels, pure and locally purified propagation, separate bond/environment truncation, explicit paper/common noise channels, numerical regression tests and documented identity-gradient limitations ([**@MaxFroehlich1410**])

- added matched full-refinement MPS truncation diagnostics with frozen weighted histories, exact statevector assessment and paired confidence intervals ([**@MaxFroehlich1410**])

- accelerated weighted MPS updates with bounded reusable mixed environments, batched pair contractions, shared immutable tape tensors, identical local gate reuse and removal of an unused cross-estimator forward pass, preserving update and truncation semantics ([**@MaxFroehlich1410**])

- added the current pruning-B state-preparation pipeline with exact/truncated MPS backends, weighted cross-trajectory online and batch updates, shared state-free noise sampling, streamed independent assessment, sealed resume, resource diagnostics and dense/MPS parity tests ([**@MaxFroehlich1410**])

- added a separately runnable tensor-network IQSP competitor with pinned author SU(4) code, all-target interpolation adapters, matched logical-noise training and assessment, small-angle deletion, independent numerical tests, and authenticated compact results ([**@MaxFroehlich1410**])

- added a separate six-qubit state-preparation manuscript package with the confirmed author list, verified result figures, protocol and native-comparison tables, bibliography, and portable Overleaf sources ([**@MaxFroehlich1410**])

- added the supplied paper-writing guide and a source/results handoff index for the completed six-qubit logical-noise study ([**@MaxFroehlich1410**])

- extended the matched logical-noise comparison to all nine six-qubit targets, with authenticated reuse of completed results, unchanged method budgets, and explicit full-grid runtime/Pareto accounting ([**@MaxFroehlich1410**])

- added native-noise refinement and independent depth selection for the frozen top-down three-target pilot, preserving parameter coordinates, weighted trajectory budgets and authenticated construction reuse ([**@MaxFroehlich1410**])

- added a matched logical-noise three-target pilot with original-gate Fontana/RobustState training, authenticated reuse of existing logical training, separate validation and locked exact evaluation, and explicit historical runtime accounting ([**@MaxFroehlich1410**])

- added paper-based NACL and RobustState noisy state-preparation competitors, real-target Fontana and Gaussian IQSP adapters, exact native-noise derivatives, independent numerical tests, and an authenticated six-qubit runner with explicit target exclusions ([**@MaxFroehlich1410**])

- added pinned upstream AQCE and QCLib BAA-LRSP adapters, conventional fixed-ansatz and unpruned layerwise controls, independent complex-state tests, and a q6 compiled-deployment experiment runner with validation locked before exact final evaluation ([**@MaxFroehlich1410**])
- documented B with online Krotov and L-BFGS noiseless batch refinement as the main state-preparation research method and future baseline, retaining D as a reference and weighted M=100 noisy fine-tuning ([**@MaxFroehlich1410**])
- added a controlled BB/L-BFGS noiseless batch-refinement study against retained-best current baselines, with identical-input CPU comparisons and unchanged B/D pruning and M=100 noisy training ([**@MaxFroehlich1410**])
- added a controlled three-policy construction-optimizer comparison with best-iterate retention, Armijo batch backtracking, identical-input diagnostics, and unchanged B/D pruning and weighted noisy fine-tuning ([**@MaxFroehlich1410**])
- added a q6 depth search that fine-tunes every B/D candidate, selects with independent weighted validation histories, stops with patience, and compares against authenticated 90% baselines with full search cost accounting ([**@MaxFroehlich1410**])
- documented the current top-down state-preparation method mathematically, retaining exact-deletion pruning B and sequential-repair variant D with weighted cross-trajectory updates and independent evaluation ([**@MaxFroehlich1410**])
- added a controlled four-setting q6 pruning experiment with exact shared-vector deletion scores, sequential removal and repair ablations, unchanged weighted fine-tuning, baseline parity checks, and compiled two-qubit resource reporting ([**@MaxFroehlich1410**])
- extended the frozen legacy weighted-trajectory comparison to all nine q6 fixture targets with two optimizer seeds, unchanged pruning/noise settings, and independently scored paired checkpoints ([**@MaxFroehlich1410**])
- added empirical no-jump trajectory compression with probability-weighted cross updates and a paired 100-history rerun of frozen q8 fine-tuning checkpoints ([**@MaxFroehlich1410**])
- added an isolated q8 vector/matrix reproduction of archived impact pruning, with independent exact noisy scoring and a separate literal hybrid/cutoff diagnostic ([**@MaxFroehlich1410**])
- added isolated q6 state-preparation research with diverse optimizer and circuit comparisons, frozen confirmation, complete score and resource audits, source recovery, and dense/MPS diagnostics ([**@MaxFroehlich1410**])
- added shared forward/backward exact deletion scores, exhaustive-score verification, and budget-reinvested noisy refinement with unchanged Krotov/Adam settings and compute-matched controls ([**@MaxFroehlich1410**])
- added an exact-score-assisted noisy deletion-and-repair development driver with weighted Adam/online-Krotov repairs, preserved optimizer moments, and scoring-inclusive compute-matched fixed-circuit controls ([**@MaxFroehlich1410**])
- added per-parameter RMS-scaled online statevector sweeps with occurrence-aware bias correction, phase-safe resume and a matched weighted Krotov/Adam adaptive-rate development screen ([**@MaxFroehlich1410**])
- added probability-weighted statevector gradients and joint online sweeps, plus a matched ordinary/weighted Krotov and Adam development experiment with paired replay hashes and setup-inclusive cost accounting ([**@MaxFroehlich1410**])
- added an experimental weighted no-factor/conditional-Pauli gradient diagnostic with exact conditioning, matched total path budgets, exhaustive channel checks and separate estimator-cost accounting ([**@MaxFroehlich1410**])
- added a paired statevector optimizer/trajectory-budget screen with authenticated read-only reuse, exact-noisy Adam references, matched-rate ordering control and fidelity-versus-training-CPU records ([**@MaxFroehlich1410**])
- documented completed four-family statevector gradient/cross, tail-rate and CRN screens, with paired artifact audits and an explicit gradient-quality decision to defer adaptive tails ([**@MaxFroehlich1410**])
- added a compact statevector development driver for exact-reference gradient diagnostics, opt-in cross estimation, tail-rate/all-online controls, fixed/periodic CRN, cost-counted backtracking and adaptive tails, with authenticated sources and resumable paired multi-target execution ([**@MaxFroehlich1410**])
- added a shared small-system statevector backend for noiseless and unitary-trajectory adjoints, independent/cross batch signals and online Krotov sweeps, with an opt-in backend for the five-method trajectory experiment including construction and checkpoint-safe resume ([**@MaxFroehlich1410**])
- added a fresh-target trajectory hybrid Krotov/Adam comparison with development-only rate selection, matched batch ablation, resumable paired streams, and separate optimization cost accounting ([**@MaxFroehlich1410**])
- added a matched exact-noisy/noiseless Adam continuation diagnostic from frozen hybrid checkpoints, with independently validated density-matrix gradients and moment-preserving resume ([**@MaxFroehlich1410**])
- added a construction-only hybrid extension for 25/30 online updates at rates 0.7/0.9, retaining matched update budgets without rerunning old settings or warm refinements ([**@MaxFroehlich1410**])
- added a matched four-target hybrid Krotov screen with 5/10/20 online updates, online rates 0.1/0.2/0.5, separate early-construction and identical-warm-start refinement comparisons, and phase/resume/batch-reproduction tests ([**@MaxFroehlich1410**])
- added a separate resumable four-family q6 compression-first pilot, true magnitude/deletion-impact pruning, exact calibrated Pauli-factor trajectory sampling, paired noiseless/noisy refinement, and independent channel/pruning/resume tests ([**@MaxFroehlich1410**])
- added bounded all-core process execution for independent state-preparation jobs, single-threaded numerical workers, deterministic parallel results, and resumable parent-owned aggregation; benchmark and causal-study CLIs now support `--workers auto` ([**@MaxFroehlich1410**])
- specified a multi-target Krotov pruning protocol with matched gate budgets, ten construction/refinement variants, held-out assessment, calibrated noise transfer, and implementation gates ([**@MaxFroehlich1410**])
- added exact Lindblad-coefficient calibration for the local-depolarizing q6 causal rerun, deterministic parallel replicates, and a reproducible calibrated hybrid diagnostic, without changing legacy profiles ([**@MaxFroehlich1410**])
- added a resumable canonical q6 causal study comparing a shared layerwise warm start with matched noiseless continuation, fixed CRN, periodically refreshed CRN, and per-update resampling using disjoint validation and paired held-out trajectory streams ([**@MaxFroehlich1410**])
- restored the reproducible state-preparation targets and concise scientific benchmark definition, with validated target loading, exact standard and Ballarin noise profiles, method adapters, and a resumable train-once/evaluate-many runner ([**@MaxFroehlich1410**])
- added a direct, resumable q8 noisy-Krotov refresh experiment that reuses stored noiseless parameters, refreshes three training trajectories every five updates, and performs paired fresh-map evaluation without the former benchmark-study infrastructure ([**@MaxFroehlich1410**])
- added MPS-based Krotov-inspired discrete adjoint optimizer for parameterized circuits (online/batch/hybrid variants) in new `mqt.yaqs.optimization` subpackage ([**@MaxFroehlich1410**])
- added MPO zip-up as default long-range gate application method ([#449]) ([**@aaronleesander**])
- parallelized MPO equivalence checker ([#448]) ([**@aaronleesander**])
- added tebd/hybrid/tdvp options for circuit simulation ([#441]) ([**@aaronleesander**])
- added Krylov tolerance as high-level parameter ([#439]) ([**@aaronleesander**])
- added accuracy presets to simulation parameters ([#438]) ([**@aaronleesander**])
- added linalg submodule to open a new path for optimizations and stop BLAS thread oversubscription for stability ([#429]) ([**@aaronleesander**])
- added high-level State and Hamiltonian classes at user-facing level ([#426]) ([**@aaronleesander**])
- added Fermionic and Jordan-Wigner MPO encodings of 1D Fermi-Hubbard model ([#220]) ([**@thilomueller**])
- added deterministic ensemble evolution with optional autocorrelator and two-time correlator outputs, including periodic-wrap two-site observable support on `(L-1, 0)` ([#409]) ([**@Gauthameshwar**])

### Changed

- restricted future state-preparation comparisons and default control-runner methods to noisy-training approaches, retaining AQCE/BAA code and results as historical references and recording the IQSP initialization hold ([**@MaxFroehlich1410**])
- stored competitor optimizer parameter histories once in authenticated binary archives and replaced repeated construction metadata with portable references, preserving full scalar traces and best checkpoints ([**@MaxFroehlich1410**])
- specified best-iterate retention and restoration for noiseless construction in the current-method guide, with tested scope and unchanged repair/noisy endpoints made explicit ([**@MaxFroehlich1410**])
- recorded the completed M=400 comparison in the current-method specification, retaining M=100 and directing future work to reuse the results unless a new scientific question justifies repetition ([**@MaxFroehlich1410**])
- made the research depth-search training-history count configurable for an isolated M=400 rerun, retaining the M=100 default, common RNG prefixes and unchanged optimization/selection settings ([**@MaxFroehlich1410**])
- marked noisy-validation depth search as optional in the current-method specification, retaining the cheaper single-circuit default and documenting its modest fidelity gains versus roughly 6–7x CPU cost ([**@MaxFroehlich1410**])
- revised the planned multi-target protocol with a separate batch/hybrid and trajectory-gradient ablation, explicit update conventions, and limits on conclusions from the supplied optimization-paper abstract ([**@MaxFroehlich1410**])
- clarified the state-preparation benchmark's hardware-error targets, derived model-specific Lindblad conversions, and documented the legacy local-depolarizing tests' approximately 2.0x/4.7x excess average gate errors; legacy standard profiles remain unchanged ([**@MaxFroehlich1410**])
- changed default to TEBD in digital circuit simulation ([#445]) ([**@aaronleesander**])
- refactored public API to use Simulator, Result, and EquivalenceChecker classes ([#430]) ([**@aaronleesander**])
- sped up and stabilized test suite ([#428]) ([**@aaronleesander**])
- changed `simulator.run` to accept `State | list[State]` and `Hamiltonian` for analog simulations instead of `MPS` / `MPO` ([#422]) ([**@aaronleesander**])
- changed solver to representation and updated noise-free simulation paths ([#422]) ([**@aaronleesander**])

### Removed

- removed the unreleased state-preparation benchmark matrix, Phase II work-package machinery, and unrelated tracked experiment drivers in favor of the focused noisy-Krotov experiment ([**@MaxFroehlich1410**])

### Fixed

- corrected the AQCE research competitor to the original paper's one-time initialization, identity-block enlargement and complete forward/backward sweeps while reusing pinned upstream tensor routines; exposed and recorded its initialization and growth controls ([**@MaxFroehlich1410**])

- preserved exact opaque two-qubit unitaries in the competitor compiler when Qiskit specializes near-symmetric gates, with audited exact three-CX fallback and regression tests ([**@MaxFroehlich1410**])
- refactored observable handling in simulator to preserve user order ([#447]) ([**@aaronleesander**])
- added tests to ensure qubit ordering matches Qiskit ([#446]) ([**@aaronleesander**])
- minor cleanup ([#420]) ([**@aaronleesander**])

## [0.5.0] - 2026-05-12

### Added

- added `MPS(..., state="haar-random")` initializer using Haar-random isometries with optional bond-dimension cap via `pad` ([#400]) ([**@Gauthameshwar**])
- added process tomography for non-Markovian noise ([#344]) ([**@aaronleesander**])
- added ability to measure in X or Y basis ([#339]) ([**@aaronleesander**])
- added Arnoldi iteration as alternative to Lanczos method ([#338]) ([**@aaronleesander**])
- added Monte Carlo Wavefunction solver ([#333]) ([**@aaronleesander**])
- added MPO.from_matrix() method ([#331]) ([**@lucello**])
- added Lindblad solver for small systems ([#330]) ([**@aaronleesander**])
- Noise model strengths can now be sampled from a normal distribution ([#329]) ([**@aaronleesander**])

### Changed

- changed Lindblad and MCWF solvers to use sparse implementation ([#338]) ([**@aaronleesander**])

### Fixed

- fixed numba attempting to parallelize in already parallelized processes ([#337]) ([**@aaronleesander**])
- fixed potential memory leak in parallelization ([#336]) ([**@aaronleesander**])

## [0.4.0] - 2026-02-05

### Added

- added ability to schedule jumps ([#319]) ([**@aaronleesander**])
- added Bose-Hubbard Hamiltonian option ([#309]) ([**@lucello**])
- added multi-threading setting for noise-free ones ([#316]) ([**@aaronleesander**])
- Minor improvements to TDVP performance ([#311]) ([**@aaronleesander**])
- ⚡️ Improve Lanczos iteration and use numba for significant speedup ([#310]) ([**@aaronleesander**])

### Changed

- ♻️ Change Pauli summation to use finite state machine construction ([#308]) ([**@aaronleesander**])
- 🔧 Replace `mypy` with `ty` ([#304]) ([**@denialhaag**])

### Removed

- 👷‍♂️ Stop testing on x86 macOS systems ([#310]) ([**@aaronleesander**])

## [0.3.3] - 2026-01-12

### Added

- updated bib and readme with Nature Communications publication ([#298]) ([**@aaronleesander**])
- updates MPO class to allow construction of arbitrary Pauli Hamiltonians ([#216]) ([**@aaronleesander**, **@thilomueller**])
- added faster paths for TDVP dense effective Hamiltonian ([#280]) ([**@aaronleesander**])
- added more stable and faster SVD and QR implementation ([#278]) ([**@aaronleesander**])
- TDVP now utilizes a dense Hamiltonian for small tensor sizes (adjusted by global variable DENSE_THRESHOLD) ([#261]) ([**@aaronleesander**])
- Updated readability of TDVP subfunctions and Lanczos method ([#261]) ([**@aaronleesander**])

### Fixed

- Fixed bug where two-site dissipative processes were computed in loop ([#290]) ([**@aaronleesander**])
- Fixed a truncation bug in TDVP sometimes leading to over-truncation ([#274]) ([**@aaronleesander**])
- Updated Lanczos method's vdot order to match mathematical expectations ([#261]) ([**@aaronleesander**])

## [0.3.2] - 2025-10-16

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#032)._

### Added

- ✨ Make it possible to return final state without setting dummy observables ([#214]) ([**@aaronleesander**])
- 👷 Enable testing on Python 3.14 ([#212]) ([**@denialhaag**])

### Removed

- 🔥 Drop support for Python 3.9 ([#180]) ([**@denialhaag**])

### Fixed

- 🐛 Fix project bar of simulator ([#213]) ([**@aaronleesander**])

## [0.3.1] - 2025-08-29

_📚 Refer to the [GitHub Release Notes](https://github.com/munich-quantum-toolkit/yaqs/releases) for previous changelogs._

<!-- Version links -->

[Unreleased]: https://github.com/munich-quantum-toolkit/yaqs/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/munich-quantum-toolkit/yaqs/compare/v0.5.0
[0.4.0]: https://github.com/munich-quantum-toolkit/yaqs/releases/tag/v0.4.0
[0.3.3]: https://github.com/munich-quantum-toolkit/yaqs/releases/tag/v0.3.3
[0.3.2]: https://github.com/munich-quantum-toolkit/yaqs/releases/tag/v0.3.2
[0.3.1]: https://github.com/munich-quantum-toolkit/yaqs/releases/tag/v0.3.1

<!-- PR links -->

[#449]: https://github.com/munich-quantum-toolkit/yaqs/pull/449
[#448]: https://github.com/munich-quantum-toolkit/yaqs/pull/448
[#447]: https://github.com/munich-quantum-toolkit/yaqs/pull/447
[#446]: https://github.com/munich-quantum-toolkit/yaqs/pull/446
[#445]: https://github.com/munich-quantum-toolkit/yaqs/pull/445
[#441]: https://github.com/munich-quantum-toolkit/yaqs/pull/441
[#439]: https://github.com/munich-quantum-toolkit/yaqs/pull/439
[#438]: https://github.com/munich-quantum-toolkit/yaqs/pull/438
[#430]: https://github.com/munich-quantum-toolkit/yaqs/pull/430
[#429]: https://github.com/munich-quantum-toolkit/yaqs/pull/428
[#428]: https://github.com/munich-quantum-toolkit/yaqs/pull/428
[#426]: https://github.com/munich-quantum-toolkit/yaqs/pull/426
[#422]: https://github.com/munich-quantum-toolkit/yaqs/pull/422
[#220]: https://github.com/munich-quantum-toolkit/yaqs/pull/220
[#420]: https://github.com/munich-quantum-toolkit/yaqs/pull/420
[#409]: https://github.com/munich-quantum-toolkit/yaqs/pull/409
[#344]: https://github.com/munich-quantum-toolkit/yaqs/pull/344
[#400]: https://github.com/munich-quantum-toolkit/yaqs/pull/400
[#339]: https://github.com/munich-quantum-toolkit/yaqs/pull/339
[#338]: https://github.com/munich-quantum-toolkit/yaqs/pull/338
[#337]: https://github.com/munich-quantum-toolkit/yaqs/pull/337
[#336]: https://github.com/munich-quantum-toolkit/yaqs/pull/336
[#333]: https://github.com/munich-quantum-toolkit/yaqs/pull/333
[#331]: https://github.com/munich-quantum-toolkit/yaqs/pull/331
[#330]: https://github.com/munich-quantum-toolkit/yaqs/pull/330
[#329]: https://github.com/munich-quantum-toolkit/yaqs/pull/329
[#319]: https://github.com/munich-quantum-toolkit/yaqs/pull/319
[#316]: https://github.com/munich-quantum-toolkit/yaqs/pull/316
[#311]: https://github.com/munich-quantum-toolkit/yaqs/pull/311
[#310]: https://github.com/munich-quantum-toolkit/yaqs/pull/310
[#309]: https://github.com/munich-quantum-toolkit/yaqs/pull/309
[#308]: https://github.com/munich-quantum-toolkit/yaqs/pull/308
[#304]: https://github.com/munich-quantum-toolkit/yaqs/pull/304
[#298]: https://github.com/munich-quantum-toolkit/yaqs/pull/298
[#216]: https://github.com/munich-quantum-toolkit/yaqs/pull/216
[#290]: https://github.com/munich-quantum-toolkit/yaqs/pull/290
[#280]: https://github.com/munich-quantum-toolkit/yaqs/pull/280
[#278]: https://github.com/munich-quantum-toolkit/yaqs/pull/278
[#274]: https://github.com/munich-quantum-toolkit/yaqs/pull/274
[#261]: https://github.com/munich-quantum-toolkit/yaqs/pull/261
[#214]: https://github.com/munich-quantum-toolkit/yaqs/pull/214
[#213]: https://github.com/munich-quantum-toolkit/yaqs/pull/213
[#212]: https://github.com/munich-quantum-toolkit/yaqs/pull/212
[#180]: https://github.com/munich-quantum-toolkit/yaqs/pull/180

<!-- Contributor -->

[**@aaronleesander**]: https://github.com/aaronleesander
[**@denialhaag**]: https://github.com/denialhaag
[**@Gauthameshwar**]: https://github.com/Gauthameshwar
[**@thilomueller**]: https://github.com/thilomueller
[**@lucello**]: https://github.com/lucello

<!-- General links -->

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[Common Changelog]: https://common-changelog.org
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[GitHub Release Notes]: https://github.com/munich-quantum-toolkit/yaqs/releases
