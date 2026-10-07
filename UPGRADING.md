# Upgrade Guide

This document describes breaking changes and how to upgrade. For a complete list of changes including minor and patch releases, please refer to the [changelog](CHANGELOG.md).

## [Unreleased]

### Source-only experiment checkout

Generated experiment output, checkpoints, plots, and manuscript data and build
artifacts are excluded from Git. Archive-driven analysis and manuscript
compilation require the corresponding local inputs. Core tests retain their
small reference fixtures; manuscript data tests skip when the result snapshot
is absent. The experimental code and configuration files remain available.

### Matched twelve-qubit sequential pruning

`experiments.topdown_q12_sequential_noisy_continuation` prepares the six
structured targets with seeds 300/400/500 and applies the archived sequential-C
pruning rule before the unchanged q12 noisy continuation. Use a fresh output
directory; its separate identity rejects simultaneous-B checkpoints. Haar
targets are excluded. See `experiments/topdown_q12_sequential_noisy_continuation.md`.
Production defaults and public APIs are unchanged.

### Twelve-qubit noisy continuation

`experiments.topdown_q12_noisy_continuation` authenticates and copies saved
ideal endpoints into a fresh 27-case input panel. Explicit independent runs
apply the existing B pruning and 100-online/100-batch noisy refinement without
repeating construction or reselecting depth. Final evaluations use independent
trajectory banks and retain historical preparation costs separately. See
`experiments/topdown_q12_noisy_continuation.md`. Production defaults and public
APIs are unchanged.

### Shorter sequential-pruning refinement budget

`experiments.topdown_sequential_noisy_finetuning_100` uses the unchanged driver
and the same authenticated inputs, with 100 online plus 100 batch noisy updates.
Use a fresh output directory; budget identities prevent resuming a 200-batch
result as a 100-batch run. The original 200-batch module and sealed experiment
remain unchanged. See `experiments/topdown_sequential_noisy_finetuning_100.md`
for parallel execution, prefix verification and CPU accounting. Production
defaults and public APIs are unchanged.

### Sequential pruning with noisy cross updates

`experiments.topdown_sequential_noisy_finetuning` copies all 27 sequential-ideal
pruning endpoints into a fresh input panel. Explicit `--action run` performs
100 noisy online sweeps and 200 batch updates using the unchanged cross signal,
rate 0.2 and a fixed weighted 100-history IBM noise bank. Recovery authenticates
both stage seals and their shared bank. Exact test scores do not select iterates.
See `experiments/topdown_sequential_noisy_finetuning.md` for budgets and the
qualified comparison with historical B. Production defaults and public APIs
are unchanged.

### Independent pruning comparison

`experiments.topdown_pruning_screen` reuses the 27 authenticated inputs of the
matched unpruned campaign. Prepare a fresh directory with `--action prepare
--parent ... --output ...`, then launch each checkpoint and `--method` separately
for parallel execution. Each arm starts from the full original circuit. Only
`ideal_repair` adjusts retained parameters, using ten safeguarded noiseless
updates after each accepted deletion. There is no construction or noisy
fine-tuning. See `experiments/topdown_pruning_screen.md` for the frozen rules.
Production defaults and public APIs are unchanged.

### Matched unpruned fine-tuning

`experiments.topdown_unpruned_finetuning` prepares all nine six-qubit near-zero
targets with seeds 300/400/500 from the completed screened-product campaign.
Use a fresh output directory and the original frozen numerical source. Explicit
`--action run --input ... --output ...` skips pruning and fine-tunes the saved
unpruned parent at the incumbent's selected depth with unchanged weighted noisy
updates. It repeats no construction and performs no depth reselection. See
`experiments/topdown_unpruned_finetuning.md` for provenance and recovery rules.
Production defaults and public APIs are unchanged.

### Twelve-qubit screened product starts

`experiments.topdown_product_screening_q12` plans 27 new ideal statevector runs
and references all 54 original near-zero/single-product controls. It uses the
existing selector and unchanged 100-online/2,000-batch depth calls, targeting
99% by depth 30. Supply an audited `--references` file for planning; training
requires explicit `--action run --target-id ... --seed ...`. Use new folders and
the original frozen numerical source, as described in
`experiments/topdown_product_screening_q12.md`. No production defaults or public
APIs change.

### Haar matrix continuation to depth 30

`experiments.topdown_haar_matrix_continuation` authenticates the eighteen completed
Haar matrix parents and prepares separate continuation commands. Use the original
frozen numerical source with the new wrapper, rather than modifying the parent
budgets or output folders. Explicit `--action run` extends one selected depth-16
endpoint with the original per-depth optimizer, targeting 99% by at most depth 30.
New CPU cost is bounded and reported separately from the historical preparation.
See `experiments/topdown_haar_matrix_continuation.md` for guards and recovery.
Production defaults and public APIs are unchanged.

### Product-state screening with a zero fallback

`experiments.topdown_product_screening` samples 1,000 random product states plus
zero and encodes the maximum-overlap candidate in the existing initial rotation
layer. Its default action plans 54 matched six-qubit pipelines and launches
nothing; explicit `--action screen` seals one initialization, and `--action run`
executes the unchanged B/weighted-noise method with that input. Defaults in the
original runner remain unchanged. `topdown_mps.run_case` now accepts an optional
`initial_theta` vector, authenticated in the manifest and applied only at depth
one. Use fresh output directories and source snapshots; old sealed experiments
continue to require their original frozen source. See
`experiments/topdown_product_screening.md`. No public library API migration is
required.

### Complete twelve-qubit initialization matrix

`experiments.topdown_initialization_matrix` plans near-zero/random-product
comparisons for all nine twelve-qubit fixtures. Supply `--reuse-random-mps` with
the original complete experiment root to authenticate six matching references
and prepare 48 fresh jobs. Explicit `--action run --target-id ... --arm ...
--seed ...` delegates to the tested noiseless driver; `--action overlaps` saves
separate product-overlap diagnostics without selecting optimizer seeds. Planning
does not launch jobs. See `experiments/topdown_initialization_matrix.md` for
resource guards, fidelity limits and recovery. Numerical defaults and public APIs
are unchanged.

### Random-MPS initialization and longer refinement

`experiments.topdown_random_mps_initialization` prepares a nine-job plan by
default. Explicit `--action run --arm ... --seed ...` trains one twelve-qubit
fixture with 100 online sweeps and up to 2,000 L-BFGS updates per depth, targeting
99% ideal fidelity. Use separate output directories for the three initializations
and seeds, and consult `experiments/topdown_random_mps_initialization.md` for
resource guards and incomplete-call recovery. This is an optional diagnostic;
production budgets, public APIs and historical results are unchanged.

### Fixed-depth Haar refinement

`experiments.topdown_haar_refinement` reuses sealed depth-66 and depth-87 growth
checkpoints in separate output directories with 2,000 additional L-BFGS updates.
It records 99% crossing costs and stops at 99.9% or the update budget. Numerical
updates and existing results remain unchanged. Incomplete calls require an
explicit restart from the parent in a new directory; live snapshots lack
optimizer history. See `experiments/topdown_haar_refinement.md`.
No public API migration is required.

The current method description now records the completed 99% fidelity result
as evidence for optional longer refinement before further depth growth. Default
optimizer budgets, stopping rules, pruning and noisy updates are unchanged;
99.9% attainment remains unverified. See section 1.3 of
`experiments/topdown_state_preparation_method.md`.

### Haar capacity audit

`experiments.topdown_haar_capacity` adds independent fixed-depth L-BFGS and
unchanged per-depth growth branches from the sealed statevector continuation.
Use separate output directories and `experiments/topdown_haar_capacity.md`.
The audit targets 99.9% ideal fidelity with explicit resource limits; it does
not change the production optimizer, noisy procedure, or historical runs.
Fixed-depth live snapshots lack optimizer history and are observational only;
an interrupted fixed call must replay from its original parent. Growth resumes
from complete authenticated depth checkpoints. No public API migration is needed.

### Noiseless depth continuation

`experiments.topdown_noiseless_continuation` continues the three non-TFIM
twelve-qubit targets from their saved unpruned construction checkpoints.
It uses separate ideal-fidelity patience and explicit depth/CPU ceilings,
without changing the original comparison, pruning or noisy updates. Use a
separate output directory and the protocol in
`experiments/topdown_noiseless_continuation.md`. No public API migration is required.
The optional `--backend statevector` trains new extensions from the same sealed
MPS parent with vectors; the default remains MPS. Version-2 manifests record
both parent and execution backends and require a separate output directory.
Use the archived version-1 snapshot to resume older runs; their records and
scientific settings are not migrated or overwritten.

### Distributed twelve-qubit comparison

The optional `experiments.state_preparation_competitors.q12_cluster` driver
prepares and stages the eleven-group logical-noise campaign without launching
it. RobustState uses the explicitly chosen 400 bases and 1,024 shots per basis;
the existing optimizers and weighted top-down updates are reused. The default
action only plans capacity; `launch` is a separate explicit action. Use a new
output directory and consult `experiments/state_preparation_competitors/Q12_CAMPAIGN.md`
for authenticated snapshots, resource limits and interruption recovery.
No public API or historical run changes; no migration is required.

### Four-qubit RobustState validation control

The optional `experiments.state_preparation_competitors.robuststate_small_control`
driver runs the existing algorithm with 500 ideal and 50 noisy updates, all 81
Pauli bases, and 1,024 shots per basis. Its exact-feedback arm is a labelled
diagnostic. It preserves the logical benchmark noise and changes no defaults
or public API; no migration is required.

### State-preparation manuscript clarification

The manuscript now distinguishes preliminary twelve-qubit IQSP resource
diagnostics from the six-qubit ranked results and documents the noise-model
comparison's scope. It also relates RobustState's limited refinement gains
under incoherent noise to the original paper's analysis. This editorial update
changes no algorithm, noise model,
benchmark ranking, or public API; no migration is required.

### Twelve-qubit final numerical audits

The RobustState diagnostic accepts `--num-bases` (default 40). This changes only
the declared feedback sampling budget; shots per basis, ideal training and Adam
updates remain unchanged. Default seeds retain their previous values. Different
budgets use explicit separate seed streams and output directories, with warm
parameter and moment hashes recorded in each result.
`--trial-start` assigns a global trial index to an independent diagnostic batch.
Together with `--trials`, it permits disjoint batches with the same seeds and
feedback as a serial run. Only global trials zero and one receive the prescribed
one-step fidelity scores. This does not parallelize sequential optimizer updates.

`experiments.state_preparation_competitors.ballarin_memory` provides observational
forward-only SVD tracing and isolated replay of one captured matrix. Use fresh
output directories; no optimizer or compression default changes. Its supervisor
retains the memory/time limits and also stops below a 512 MiB free-disk floor.
Capturing a matrix requires space for that file plus a one-GiB reserve.

`experiments.state_preparation_competitors.q12_final_checks` runs standalone
compression and sampled-feedback diagnostics. Its ideal RobustState checkpoint
uses the exact pure-state identity for the same Hilbert-Schmidt loss, preserving
the full Adam path and phase-switch moments. No production optimizer default
changes. Use separate output directories and the bounds described in
[Q12_FINAL_CHECKS.md](experiments/state_preparation_competitors/Q12_FINAL_CHECKS.md).
The diagnostic's opt-in `--compact-checkpoints` owns compact copies of segment
outputs instead of retaining views into larger SVD buffers. It preserves
the values and derivatives and does not alter production optimizer defaults.
The vendored SVD first derivative now accumulates its terms into one buffer in
the original arithmetic order. Compression and regularization remain unchanged;
saved reference derivatives and independent density-oracle tests cover the change.

### TCI construction for compressed IQSP

Compressed `iqsp_tn_candidates` runs now use TensorCrossInterpolation.jl targets
by default. Instantiate the isolated Julia project described in
[IQSP_TCI.md](experiments/state_preparation_competitors/IQSP_TCI.md) before use.
Set `target_construction="dense_svd"` explicitly to repeat the historical
diagnostic adapter. With TCI, leave `CombOptions.target_tolerance=0`; target
accuracy is controlled by `TCIOptions` and independently checked before training.
The old exact backend is unchanged. Start new output directories and include
the recorded Julia setup cost in runtime comparisons. Published optimizer
policies and the benchmark's common noise/target inputs are unchanged.

### Twelve-qubit competitor prerequisites

Fontana and RobustState accept up to twelve-qubit fixtures and automatically
use exact recursive density checkpoints above six qubits. Noiseless derivatives
use statevectors. Their losses, optimizers, noise schedules and retention rules
are unchanged; legacy six-qubit numerical execution remains available.
IQSP's compressed-comb candidate path accepts twelve-qubit Gaussian fixtures
on a single twelve-site tooth. Its local SU(4) lowering verifies local unitaries
and the prepared state without allocating a global twelve-qubit unitary.
For larger noisy comb runs, explicitly set `CombOptions(checkpoint_interval=8)`
to recompute autograd segments without changing their derivatives. Monitor
physical footprint as well as resident memory on macOS.
Logical assessment streams batches of at most 256 statevectors by default.

Use new output directories and rerun feasibility checks before a campaign.
The original Ballarin identity-gradient caveat remains: the Gaussian startup
audit checks actual progress and later gradient agreement without jitter or a
replacement gradient. It is not a guarantee for arbitrary targets or later
rank transitions. The sampled-basis RobustState budget remains a separately
reported extrapolation, not a validated twelve-qubit tomography prescription.
See [readiness documentation](experiments/state_preparation_competitors/Q12_READINESS.md).

### Optional compressed IQSP comb

`CombTensorNetworkObjective` adds a separate compressed backend using pinned
Ballarin comb kernels. `iqsp_tn_candidates` accepts optional `comb_options`;
omitting it preserves all previous exact-backend settings. Dense target
conversion is limited to twelve qubits; larger inputs require comb tensors.
Use a new experiment output directory and record the backend, compression and
noise settings. The author's SVD gradient can suppress entangling directions at
the all-identity initialization, so compressed and exact training must not be
assumed equivalent there. No initialization or optimizer is silently changed.
See [compressed comb documentation](experiments/state_preparation_competitors/IQSP_COMB.md).

### Weighted MPS state preparation

`experiments.topdown_mps_truncation` provides a separate diagnostic comparing
bond caps and SVD thresholds against full statevector fine-tuning from sealed
checkpoints. It changes no defaults, compresses no target, and evaluates learned
parameters using exact statevectors on fresh, independent histories.

The weighted MPS backend now reuses mixed overlap environments, batches pairs
and shares unchanged private tape tensors. Optimizer rules, noise samples,
weights and SVD truncation settings are unchanged; floating-point contraction
order changes. Start a fresh output directory for performance comparisons:
source-authenticated checkpoints from the earlier implementation cannot be
resumed under the new code. Environment-cache memory is reported separately
from retained tensor memory. No public caller-owned MPS tensor is shared.

`MPSBackend`, `PreparationBackend`, `create_backend` and `TrajectorySampler` are
new optional APIs. Existing dense and uniform-weight MPS training entry points
retain their defaults. The new `experiments.topdown_mps` runner executes the
current B/online-Krotov/L-BFGS method with either backend, including weighted
cross updates. Start a new output directory: numerical sources and backend
settings are authenticated and old six-qubit result archives are not caches
for this runner. Final assessment defaults to independent trajectory batches;
`--exact-density-check` adds the old small-system oracle. Truncation is opt-in,
and allocation profiling is a separate option. See the
[MPS runner documentation](experiments/topdown_mps.md).

### Paper-writing references

The new `manuscript/state_preparation/` directory contains a separate first
draft, vector figures, LaTeX tables, verified bibliography, and portable data
for regenerating the visual material. Compile `main.tex` with pdfLaTeX and
BibTeX, or run its `compile.sh`. This documentation package leaves the older
manuscript and the experiment archives intact and changes no library API or
experiment settings. Full optimization replay still requires the referenced
archives; the Overleaf package supports manuscript compilation and figure
regeneration. The author list is maintained separately in `authors.tex`.

The supplied [paper-writing guide](PAPER_WRITING.md) is now stored unchanged at
the repository root. The [paper handoff](experiments/state_preparation_competitors/PAPER_HANDOFF.md)
indexes the current method, completed logical-noise results, experiment scripts,
source archives and interpretation limits. These documentation additions do not
change algorithms, dependencies or stored results.

### Optional state-preparation competitors

The separate `run_iqsp_tn` CLI implements IQSP with exact circuit tensor-network
contractions and the same IBM logical noise in training and assessment. It uses
the author fork's SU(4) exponential to avoid the earlier zero-initialized KAK
trap. Install `requirements-iqsp-tn.txt` in a separate environment and use a new
output directory; previous IQSP results are not compatible. All nine fixtures
are supported, with a declared geodesic extension for the eight non-Gaussian
targets. The adapter includes the paper's small-angle two-qubit deletion but
does not reproduce the authors' compressed-comb runtime or scaling. See the
[IQSP tensor-network protocol](experiments/state_preparation_competitors/IQSP_TN.md).

The separate `run_logical_suite` CLI extends the existing logical pilot to all
nine six-qubit targets with the same IBM profile and seed 300. It references
completed pilot groups without retraining or reselection, reuses the archived
incumbent, and trains only missing method-target combinations. Retain the old
output directories for compact-history references. See the
[full logical suite protocol](experiments/state_preparation_competitors/LOGICAL_SUITE.md).

The separate `run_native_topdown` CLI reruns only the current method on the
three pilot targets, with native noise during refinement and depth selection.
It preserves archived noiseless construction and pruning B, compiles the symbolic
parameters once, and keeps that native schedule for assessment. Start a new
output directory; retain the referenced archives for reproducibility. See the
[native pilot protocol](experiments/state_preparation_competitors/NATIVE_TOPDOWN.md).

The separate `run_logical` CLI compares Gaussian, TFIM critical and Haar 1 under
IBM noise after every original elementary gate, using seed 300 and the existing
pilot budgets. It reuses authenticated logical training and trains Fontana and
RobustState with `noise_basis="logical"`; their default remains `"native"`.
Training, validation and final assessment use the same logical channel. Native
compilation supplies resource counts only. Start a new output directory and keep
the referenced old construction bundles, which own reused parameter histories.
See the [logical-noise protocol](experiments/state_preparation_competitors/LOGICAL_NOISE.md).

Future performance comparisons include only methods that train with noise.
The `run` CLI now defaults to `current_b_lbfgs_frozen`, `fixed_vqsp` and
`layerwise`; AQCE and BAA-LRSP require explicit historical selection and are
excluded from the active experiment plan. Keep their existing results and
implementations. The `run_noisy` defaults remain NACL and RobustState; add
Fontana for real targets. The old dense IQSP reconstruction remains on hold;
new IQSP runs use the separately tested tensor-network adapter above. See the
[active scope](experiments/state_preparation_competitors/NOISY_METHODS.md).

The new [noisy-method runner](experiments/state_preparation_competitors/NOISY_METHODS.md)
adds NACL using optional official PyNomadBBO 4.6.0 and a local RobustState
implementation for all nine six-qubit targets. Fontana supports the six real
fixtures, and IQSP supports the Gaussian fixture. Unsupported pairs are recorded
as exclusions. These are documented paper-based simulator adaptations, without
additional competitor pruning or a claim to reproduce the papers' hardware results.

Use `python -m experiments.state_preparation_competitors.run_noisy` with
`requirements-noisy.txt` in a separate optional environment. The original `run`
driver retains explicit historical method support. Noisy candidates preserve their
trained native schedule during final evaluation; post-training cancellation
would change their noise windows. Start a new output directory for the new
source fingerprint and protocol. Production YAQS APIs are unchanged.

AQCE now follows Algorithms 1–5 of the original published paper, reusing the
pinned MNISQ tensor primitives with a local scheduler. `--aqce-sweeps` counts
complete forward/backward sweeps, replacing the former forward-only meaning.
`--aqce-initial-gates` (M0, default 1) initializes once; subsequent growth uses
`--aqce-growth-increment` (delta M, default 1) identity gates optimized backwards
against the residual target. These are declared benchmark settings, not a claim
to reproduce the paper's numerical experiments. Positive gate budgets must be
reachable from M0 by delta M; the zero-budget empty control remains available.
Use a new output directory: source fingerprints reject resuming the former
MNISQ variant. Old AQCE fidelity and timing results describe that variant only;
they cannot be mixed with the corrected algorithm. Other methods are unchanged.

The [competitor protocol](experiments/state_preparation_competitors/README.md)
adds AQCE by reusing pinned MNISQ numerical primitives and BAA-LRSP through
official QCLib 0.1.15. Optional dependencies are isolated from the project lock:

```sh
uv run --with-requirements experiments/state_preparation_competitors/requirements.txt python -m experiments.state_preparation_competitors.run --output NEW_DIRECTORY
```

The default sweep uses all nine q6 fixture targets, two optimizer seeds for
variational controls, and both historical synthetic noise profiles. Deterministic
synthesis is shared across profiles. The frozen B + L-BFGS reference requires
the existing batch-refinement archive; select `fixed_vqsp` and `layerwise`
with `--methods` to run the active controls without it. See the protocol
for a small smoke command, grids, upstream variant limitations, and tests.

All circuits receive the same native-gate compilation and independent validation;
selection is saved before exact density-matrix final assessment. This introduces
a **compiled deployment evaluation**, including noise after RZ, and must not be
merged with historical logical-gate fidelities. Archived reference selection and
training parameters remain unchanged. No production API migration is required.

The competitor compiler now detects Qiskit's near-symmetric two-qubit
approximations and uses an audited exact three-CX fallback when needed. The
original \(10^{-9}\) equivalence tolerance is unchanged. Use a new experiment
output directory after this correction; the source fingerprint deliberately
rejects resuming old compiled receipts. Historical source circuits remain valid.

Competitor storage schema v2 also replaces repeated inline optimizer parameter
histories with a single lossless `histories.npz` in each construction bundle.
Scalar curves, timings, seeds, best checkpoints and reproducibility metadata
are retained. Selection/final/summary `records[].method` objects now contain
construction references and candidate indices instead of the full metadata.
Use `load_candidate_metadata(output_root, method_record)` and
`load_parameter_history(construction_folder, phase["parameter_history"])` from
the competitor runner to authenticate and read them. The protocol contains a
complete example. Start v2 in a new output directory; existing v1 archives are
left intact. Scientific settings and the production YAQS API are unchanged.

### Main state-preparation research method

The [current-method specification](experiments/topdown_state_preparation_method.md)
now designates **B + online Krotov + L-BFGS** as the main method and baseline for
future experiments. L-BFGS applies to noiseless batch construction, with
best-iterate retention. Noisy fine-tuning retains the existing cross-trajectory
updates and weighted M=100 histories. D is kept as a research reference and
need not be rerun in every development experiment.

The measured L-BFGS advantage comes from the optional every-depth search;
that extension remains optional, and no new first-90% result or stopping policy
is introduced. This records the research-method decision; archived drivers and
frozen comparisons retain their settings and require no API migration.

### BB and L-BFGS batch refinement

The opt-in [batch-refinement protocol](experiments/topdown_batch_refinement_protocol.md)
compares safeguarded BB1 and L-BFGS with the then-current retained-best construction
controls. For that completed comparison, the primary baseline is backtracking for B
and best-only retention for D; both controls are reported for each pruning variant. Only the
noiseless batch phase changes. Online sweeps, D repair, weighted M=100 noisy
fine-tuning, validation and depth selection retain their previous settings.

Run `uv run python -m experiments.topdown_batch_refinement --output NEW_DIRECTORY --workers 4`.
Generate the report with `uv run --with matplotlib python -m experiments.topdown_batch_refinement_analysis --output NEW_DIRECTORY`.
This research driver requires no production API migration; existing driver
defaults and frozen earlier experiment sources are unchanged.
The [completed comparison](experiments/topdown_batch_refinement_findings.md)
reports 496 fixed-input optimizations and 288 full searches. For the optional
depth search, L-BFGS improves B's mean noisy fidelity at lower search CPU;
D retains the best-only construction baseline. BB does not improve either
primary baseline's mean noisy fidelity. Section 1.2 of the current-method
specification records the tested scope and the new pruning-specific recommendation.

### Construction-optimizer stability comparison

The opt-in [stability protocol](experiments/topdown_optimizer_stability_protocol.md)
compares the original final-iterate construction, best-iterate retention, and
best retention with Armijo batch backtracking. It first replays identical
archived construction inputs, then compares all three policies with pruning B/D
and unchanged M=100 noisy fine-tuning in the optional depth search. D's repair
optimizer is deliberately unchanged. Existing drivers and production APIs retain
their previous behavior.

Run `uv run python -m experiments.topdown_optimizer_stability --output NEW_DIRECTORY --workers 4`.
Generate the comparison with `uv run --with matplotlib python -m experiments.topdown_optimizer_stability_analysis --output NEW_DIRECTORY`.
The [completed comparison](experiments/topdown_optimizer_stability_findings.md)
reports all 216 searches. Best retention with backtracking stabilizes construction
and improves B's average noisy fidelity at similar overall cost; D's downstream
benefit is mixed, and its unchanged repair optimizer still exhibits collapses.
Section 1.1 of the [current-method specification](experiments/topdown_state_preparation_method.md)
now prescribes saving and restoring the best noiseless construction parameters,
including the initial state. This documentation update does not change driver
defaults: the tested implementation remains in the stability driver, and D repair
and noisy fine-tuning still return their final iterates.

### Noisy-validation depth search

The optional `--training-paths 400` argument repeats both the depth search and
its single-circuit control with 400 raw histories. The default remains 100.
The [M=400 protocol](experiments/topdown_m400_protocol.md) changes only this
sampling budget, preserves common RNG prefixes, and retains independent validation
and exact final evaluation. Use a new output directory for a different budget.
The [completed M=400 comparison](experiments/topdown_m400_findings.md) found
negligible mean fidelity changes at higher cost; M=100 remains the default.
Section 3.1 of the [current-method specification](experiments/topdown_state_preparation_method.md)
records this as a completed experiment: reuse its results instead of repeating
the same settings, unless there is a documented new scientific question.
Generate its paired report with `uv run --with matplotlib python -m experiments.topdown_m400_analysis --reference REFERENCE_DIRECTORY --output OUTPUT_DIRECTORY`.

The [current-method specification](experiments/topdown_state_preparation_method.md)
marks this search as optional: its mean fidelity gains of 2.40–3.56 percentage
points cost roughly 6–7x more CPU than the default single-circuit B/D pipeline.
Both modes retain compressed 100-history noisy fine-tuning and independent final evaluation.

`uv run python -m experiments.topdown_depth_search --output NEW_DIRECTORY --workers 4`
removes the 90% construction cutoff and fine-tunes every visited depth with the
existing B/D pruning and compressed 100-history training. Independent validation
controls patience and selection; exact density scoring follows locked decisions.
The [frozen protocol](experiments/topdown_depth_search_protocol.md) specifies all
72 searches, stopping tolerances, RNG separation and construction-inclusive costs.
The old 90% pipeline remains available and is replayed as a paired control.
This opt-in research driver requires no production API migration.
The [completed findings](experiments/topdown_depth_search_findings.md) report all
72 paired searches, their full CPU costs, trajectory audits, and the exposed
noiseless-optimizer instability. Generate the detailed tables and figures with
`uv run --with matplotlib python -m experiments.topdown_depth_analysis --output OUTPUT_DIRECTORY --workers 4`.

### Controlled q6 pruning comparison

The [concise mathematical method](experiments/topdown_state_preparation_method.md)
specifies pruning B (the current default) and D (the repair variant), including
the actual weighted update signal and independent final evaluation.

`uv run python -m experiments.legacy_pruning_ablation --output NEW_DIRECTORY --workers 4`
compares the existing saliency cutoff with exact noiseless removal scores,
sequential deletion, and ten-update repair. All four settings reuse the same
18 six-qubit starting checkpoints and retain 100 weighted training histories.
The [frozen protocol](experiments/legacy_pruning_ablation_protocol.md) specifies
all 144 conditions, independent exact density scoring, explicit coverage
failures, fixed reference compilation, and optimization resource accounting.
This is an opt-in experiment and requires no production API migration.
The [completed comparison](experiments/legacy_pruning_ablation_findings.md)
reports all nine targets, matched fidelity changes, compiled two-qubit
resources, repair instability and the eight unchanged-policy coverage failures.
`uv run python -m experiments.legacy_pruning_analysis --output OUTPUT_DIRECTORY --workers 4`
independently audits the saved deletion decisions and generates the full report.

### Full q6 target extension of the legacy comparison

`uv run python -m experiments.legacy_topdown_q6 --output NEW_DIRECTORY --workers 4`
runs the existing 3-versus-100-history fine-tuning experiment on all nine
six-qubit vectors from `benchmarks/state_preparation_target_states.json`, using
optimizer seeds 300 and 400. It preserves the archived and literal construction,
pruning, update budgets, and IBM/Google coefficients. This target extension is
separate from the base benchmark's noise-profile matrix. Statevectors train the
circuits and exact density matrices score them. Checksummed checkpoints support
`--resume` with identical source, fixture, and package versions. Cases failing
the 90% construction threshold or jump-coverage check are reported explicitly.
The [completed q6 findings](experiments/legacy_topdown_q6_findings.md) report all
108 paired conditions, per-target and per-seed fidelities, and cutoff failures.

### Compressed 100-history fine-tuning

`StatevectorBackend.compress_no_jump_maps` merges sampled empty histories and
returns their multiplicity weights. Cross gradients and online sweeps now accept
probability weights on both trajectory indices. Existing unweighted calls keep
their behavior. Compression does not resample histories or replace empirical
counts with analytic probabilities.

`uv run python -m experiments.legacy_topdown_weighted --output NEW_DIRECTORY`
reruns the 13 saved q8 fine-tuning endpoints with 100 sampled histories, retaining
construction, pruning, optimizer settings, seeds, and CRN schedules. It verifies
the original three-path result and nested histories, records jump coverage, and
evaluates final fidelity with independent exact density matrices. `--resume`
requires identical source and input hashes.
The [completed findings](experiments/legacy_topdown_weighted_findings.md) record
all paired scores, retained path counts, timing, and the unchanged cutoff issue.

### Archived top-down vector/matrix verification

`python -m experiments.legacy_topdown_dense --output NEW_DIRECTORY` reproduces
the archived eight-qubit comparison and separately tests layerwise hybrid
training with the absolute 0.01 impact cutoff. It stores target vectors,
parameters, circuits, traces, source evidence, and independent Qiskit density
scores. Use a new output directory. This experimental driver requires no API
migration; its logical noise rates and baseline labels retain the historical
definitions for comparison.

### Overnight state-preparation research

The opt-in package `experiments.overnight_state_preparation_20260913_0108`
contains tested q6 optimization, conditional circuit construction, exact noisy
scoring, complete comparison audits, and dense/MPS diagnostics. The
[research report](experiments/overnight_state_preparation_20260913_0108/REPORT.md)
records the completed development search and the candidate's failed confirmation
gate. It does not recommend a production optimizer change or q12 scaling.

The [handoff](experiments/overnight_state_preparation_20260913_0108/HANDOFF.md)
provides the frozen candidate CLI, exact source archive and artifact paths.
Identical-source reruns resume saved state and verify the complete input and
result records. A changed scientific configuration needs a new output root and
protocol. New random seeds do not restore untouched status to target identities
already inspected during this study. These additions require no production API
migration; the fixed benchmark definitions remain the startup versions.

### Shared-adjoint deletion scoring

`density_deletion_fidelities` in the experimental exact-density helper returns
all finite single-gate/channel deletion fidelities using one forward/backward
tape. The exhaustive scorer remains available for verification. The restructuring
helper accepts an optional ranker and continuation epoch offset; defaults retain
the previous behavior.

`python -m experiments.state_preparation_adjoint_restructuring` follows the
[verification and reinvestment protocol](experiments/state_preparation_adjoint_restructuring.md).
It verifies all historical scores before rerunning the same search, then spends
saved CPU on further refinement while rerunning fixed-circuit controls. The
old output is read-only evidence: use a fresh root for the changed runtime and
`--resume` only for identical-manifest stage/case continuation in that new root.
This exact-density-assisted experiment is not a trajectory structural scorer or
native-device compiler; production defaults are unchanged.

### Noisy circuit restructuring diagnostic

`python -m experiments.state_preparation_noisy_restructuring` runs the
[small-system deletion-and-repair protocol](experiments/state_preparation_noisy_restructuring.md).
This opt-in experiment uses exact density scores for structure selection and
weighted statevector trajectories for parameter updates. It preserves retained
optimizer moments after true gate deletion, charges rejected work, and compares
with fixed circuits continued to the corresponding CPU budget. It is not a
native-device compiler or an entirely trajectory-based synthesis algorithm.
Use a fresh output root; identical-manifest resume reuses completed cases and
reruns any incomplete case. Existing drivers and production defaults are unchanged.

### Adaptive online statevector updates

`StatevectorBackend.adaptive_online_update` adds optional RMS scaling within
the existing gate-sequential update order. It returns new parameters, raw
signals, squared-signal moments and per-parameter occurrence counters without
mutating inputs. Ordinary online updates retain their original rule.

`python -m experiments.state_preparation_adaptive_online` runs the
[64-case development protocol](experiments/state_preparation_adaptive_online.md)
with weighted R32 sampling, matched three-rate Adam/adaptive-online screens,
and fixed all-online and Krotov-Adam controls. The shared development driver
now supports `online_rmsprop` tails and validates saved occurrence counters.
Use a fresh output root: source fingerprints and the update-kernel checkpoint
signature intentionally reject older revisions. Earlier artifacts are unchanged.

The [completed adaptive-online findings](experiments/state_preparation_adaptive_online_findings.md)
document the improvements and the remaining matched-cost deficit to Adam;
no production default or new-target confirmation setting is selected.

### Weighted trajectory optimization

`python -m experiments.state_preparation_weighted_optimization` runs the
[matched development protocol](experiments/state_preparation_weighted_optimization.md)
on the existing four targets only. It compares weighted and ordinary sampling
for all-online Krotov, hybrid Krotov and Adam at R32/R128. The statevector
`metrics`, `gradient` and `online_update` methods now accept optional fixed
probability `weights`; defaults are unchanged. Cross updates now also support
weights, as described in the compressed 100-history fine-tuning entry above.
The conditional sampler is shared with the earlier gradient diagnostic.

Use a fresh output root for this revision. Scientific source fingerprints and
the sampling-policy checkpoint field intentionally reject resumes of older
driver revisions; completed earlier artifacts remain unchanged. Setup costs
are included in this experiment's training totals, unlike the earlier budget
screen. Do not mix timing conventions or code revisions.

### Weighted trajectory estimator diagnostic

The [completed development findings](experiments/state_preparation_sampling_budget_findings.md)
separate the optimizer comparison from the weighted estimator result; no
production defaults or confirmation configuration are selected from these runs.

`python -m experiments.state_preparation_weighted_sampling` implements the
[weighted development diagnostic](experiments/state_preparation_weighted_sampling.md).
It is opt-in, limited to the existing parameter-independent local Pauli channel,
and does not change the production trajectory sampler or optimizer APIs. Its
R counts one deterministic path and R-1 conditional paths. No new target panel
or weighted optimizer sweep is launched by this command.

### Statevector trajectory-budget comparison

`python -m experiments.state_preparation_sampling_budget` runs the
[paired development screen](experiments/state_preparation_sampling_budget.md).
Provide the completed Experiment 2 source, a fresh output root and optionally
the compatible old tail root for read-only reuse. It does not change existing
optimizer defaults or old driver fingerprints, and does not launch new targets.
Use `--resume` for interrupted work with identical source/runtime metadata.

### Statevector development-study driver

The [2026-09-11 findings](experiments/state_preparation_trajectory_diagnostics_findings.md)
document the completed gradient/cross, tail and CRN screens. Adaptive tails were
deferred under the planned gradient-quality condition. These results do not
change optimizer defaults, require a migration or modify earlier artifacts.

`python -m experiments.state_preparation_trajectory_diagnostics` implements the
[compact development protocol](experiments/state_preparation_trajectory_diagnostics.md).
Pass `--source-root` pointing to the completed trajectory comparison (Experiment
2), not directly to its construction source, and a fresh `--output-root`.
The default is independent-gradient diagnostics; `--include-cross` adds a paired
cross diagnostic. Tail, CRN, backtracking and adaptive studies require explicit
`--study` choices. There is no automatic launch of all studies or cross training.

The driver reuses the four development targets and frozen circuits, not the
confirmation panel. It supports all-online schedules and tail-local Adam counters
without changing older runner defaults or signatures. Per-update checkpoints
preserve moments, sampling epochs and accounting; `--resume` validates source,
configuration and runtime while permitting a different `--workers` setting.
Existing scientific artifacts are read only. No study is executed by installing
or importing the new driver.

### Small-system statevector optimization

`mqt.yaqs.optimization.StatevectorBackend` provides dense pure-state metrics,
batch gradients and gate-sequential online updates using the existing circuit
and noise-provider formats. See the [backend guide](experiments/statevector_backend.md).
The trajectory comparison CLI now accepts `--backend statevector`, covering
construction, repair, noiseless hybrid and trajectory Adam/batch/hybrid refinement.
Exact-noisy Adam and exact scoring remain density-matrix based. The default is
still `mps`; older standalone experiment CLIs are not automatically converted.

Use a fresh output directory. Backend identity is included in checkpoint and
manifest signatures; source fingerprints and the new block signatures reject
resuming earlier runner revisions, even with `mps` selected. Existing scientific
artifacts are unchanged. Do not combine timing measurements from different
backends. General nonunitary jumps (including amplitude damping) and derivatives
of state-/parameter-dependent noise are not supported by this small backend.

### Trajectory competitiveness experiment

`python -m experiments.state_preparation_trajectory_comparison` runs
[Experiment 2](experiments/state_preparation_trajectory_comparison.md), selecting
rates on old development targets before constructing and comparing five methods
on eight fresh targets with two replicas each. Supply `--source-root` for the
completed hybrid construction extension and a fresh `--output-root`.
`--stage development` stops after nomination; `--resume` validates all retained
artifacts and preserves Adam moments, noise counters, and original timings.
This benchmark adapter reuses existing optimizer kernels without changing
their public API, the canonical target fixture, or earlier experiments.

### Exact-noisy Adam diagnostic

The new `python -m experiments.state_preparation_exact_adam` entry point runs
only [Experiment 1](experiments/state_preparation_exact_adam.md): 100 matched
noiseless/exact-noisy Adam updates from each of eight saved `H25_lr0p7` final
checkpoints. Supply `--source-root`, a fresh `--output-root`, and optionally
`--workers auto`. Resume requires `--resume` and identical scientific/runtime
metadata; Adam moments and bias-correction counters are preserved. This separate
six-qubit density-matrix training diagnostic does not change the production
trajectory optimizer, earlier evaluation-only pilot, or existing artifacts.

### Hybrid compression screen

Add `--construction-extension` to run only the four new 25/30-update,
0.7/0.9-rate settings described in the
[extension protocol](experiments/state_preparation_hybrid_extension.md).
This construction-only mode has a separate manifest version and omits the
warm-refinement arms. Use a fresh output directory; previous results remain
unchanged. Source fingerprints reject resuming artifacts from a different
runner revision, including the earlier completed screen.

The separate `python -m experiments.state_preparation_hybrid_pilot` entry point
requires `--source-root` pointing to the completed compatible magnitude-pruning
pilot and a fresh `--output-root`. It compares nine hybrid settings against a
recomputed batch control, which must reproduce the original saved endpoints.
The [hybrid screen protocol](experiments/state_preparation_hybrid_pilot.md)
separates early layerwise construction from identical-start noiseless/noisy
refinement. Existing pilot code, optimizer defaults, and result directories
are unchanged. `--resume` preserves phase and trajectory counters exactly;
`--workers auto` uses bounded single-threaded process workers.

### Small compression-first pilot

The separate `python -m experiments.state_preparation_pilot --output-root PATH`
entry point implements the [four-family q6 pilot](experiments/state_preparation_pilot.md),
not the full multi-target research protocol. Its default `--workers auto` runs
eight target/seed jobs with one numerical thread each on an eight-core machine.
Resume requires `--resume`, matching source/configuration/fixture checksums, and
the same dependency versions; worker allocation may change.

The pilot uses exact independent Pauli-factor sampling of the calibrated local
Lindblad exponential. This removes the older at-most-one-jump finite-step bias
for this commuting channel only. Training remains pure-state MPS trajectories;
the six-qubit density-matrix oracle is evaluation-only. Legacy providers,
existing experiment artifacts, and standard benchmark presets are unchanged.

### Parallel experiment execution

The state-preparation benchmark and causal-study CLIs now default to
`--workers auto`, using up to all detected CPU cores when enough independent
jobs are available. Use `--workers 1` for serial execution or a positive count
to reduce CPU/RAM pressure. Python APIs retain their serial defaults.
Each process uses one BLAS/OpenMP numerical thread; completed jobs are
aggregated by one parent process. Parallel custom method adapters must be
picklable, and mutations of adapter state remain local to each worker.

Worker count does not enter scientific configuration hashes or random seeds.
Existing output can resume with a different worker allocation. Checkpoints
are written atomically, and result rows use canonical run-ID order. The
causal study still parallelizes only its independent replicas; three replicas
cannot occupy eight workers, and their count is not increased automatically.
The general benchmark parallelizes target/method/depth/initialization jobs,
training once before evaluating that job's noise conditions.

This execution support does not implement the planned pruning/hybrid study or
change any noise model. Profile memory before running large jobs concurrently;
`auto` caps CPU usage by core/job count, not by a memory-usage prediction.

### Planned multi-target pruning experiments

The [next-experiment protocol](experiments/state_preparation_next_experiments.md)
specifies a separate architecture-changing, noise-aware extension with several
targets in every experiment. It is a research specification, not a new preset
or implemented CLI. Existing runners and defaults are unchanged; the protocol
lists the pruning, stream-coupling, and multi-target orchestration work required
before execution.

Protocol v1.1 also specifies a separate four-target cold/warm batch-versus-hybrid
ablation and independent-versus-cross trajectory checks. It does not change
the main ten-method matrix or any runtime defaults. Its numerical hybrid
settings are provisional: only the optimization paper's abstract was supplied,
not its hyperparameter tables. Because the protocol version enters seed keys,
do not resume a v1 manifest as v1.1.

### State-preparation noise calibration clarification

The [benchmark definition](benchmarks/state_preparation_benchmarks.md#common-rate-convention)
now distinguishes hardware average gate errors from per-operator Lindblad
coefficients and derives the conversion for each noise structure. The reference
0.064% and 0.51% values are complete-window average-error targets, not coefficients
to copy to every operator.

Existing standard profiles and benchmark runner presets still use the legacy
coefficients. With a unit time step, the earlier
local-depolarizing tests correspond to approximately 0.128% and 2.411% average
gate errors. Preserve their configurations and results as higher-noise runs.

For the focused q6 causal experiment, explicitly select
`--noise-calibration hardware_average_error` and a new output directory. This
uses `CalibratedLocalDepolarizingProvider` with unit-window per-operator
coefficients approximately `0.000320205` and `0.001066470`. Its configuration
and provider checksums distinguish it from legacy runs; incompatible resume
requests are rejected. The default remains `legacy` for reproducibility.
`--workers 3` runs independent replicates concurrently without changing their
seeds or scientific configuration. The standalone
`experiments.noisy_krotov_hybrid_smoke` entry point repeats the historical
20-update hybrid diagnostic with the calibrated provider.

The coefficients calibrate the exact Lindblad exponential. The unchanged
one-jump circuit sampler gives average errors approximately 0.0640102% and
0.5102714%; its finite-step bias is not removed by increasing trajectories.
Calibration of other operator families and integration of calibrated strength
sweeps into the general benchmark runner remain pending. Do not reinterpret
old results or resume old manifests under the corrected calibration.

### `simulator.run` becomes `Simulator(...).run(...)`

The free `mqt.yaqs.simulator.run` function has been replaced by a `Simulator` class.
`Simulator` owns the execution-side configuration (parallel vs. serial execution, worker count,
progress reporting, multiprocessing context, retry policy); the physics inputs are passed to
`Simulator.run`. `Simulator.run` returns a `Result` (from
`mqt.yaqs.core.data_structures.result`) that holds all simulation outputs. The `*SimParams`
object you pass in is never mutated.

**Before:**

```python
from mqt.yaqs import simulator

simulator.run(state, op, sim_params, noise_model, parallel=True)
```

**After:**

```python
from mqt.yaqs import Simulator

sim = Simulator()
result = sim.run(state, op, sim_params, noise_model)
```

`show_progress` and `num_threads` were removed from `AnalogSimParams`, `StrongSimParams`, and
`WeakSimParams`. Pass `show_progress` to `Simulator` instead; `num_threads` was unused and has been
deleted.

### `threshold` renamed to `svd_threshold` on `*SimParams`

The SVD bond-truncation setting on `AnalogSimParams`, `StrongSimParams`, and `WeakSimParams` is now
`svd_threshold` (attribute and constructor keyword). Simulation presets use the key `svd_threshold`
in `SIMULATION_PRESETS`. This distinguishes SVD truncation from `krylov_tol` (Krylov/Lanczos matrix
exponential) and from unrelated `threshold` parameters elsewhere (for example
`EquivalenceChecker(threshold=...)`).

**Before:**

```python
params = AnalogSimParams(threshold=1e-8)
value = params.threshold
```

**After:**

```python
params = AnalogSimParams(svd_threshold=1e-8)
value = params.svd_threshold
```

### `digital.equivalence_checker.run` becomes `EquivalenceChecker(...).check(...)`

The free `mqt.yaqs.digital.equivalence_checker.run` function has been replaced by
`EquivalenceChecker` (now exposed at `mqt.yaqs.EquivalenceChecker`). `EquivalenceChecker` owns the
numerical thresholds (`threshold`, `fidelity`); the two circuits are passed to
`EquivalenceChecker.check`. The return value is unchanged: a `dict` with keys `equivalent` and
`elapsed_time`.

**Before:**

```python
from mqt.yaqs.digital.equivalence_checker import run

result = run(circuit1, circuit2, threshold=1e-6, fidelity=1 - 1e-13)
```

**After:**

```python
from mqt.yaqs import EquivalenceChecker

checker = EquivalenceChecker(threshold=1e-6, fidelity=1 - 1e-13)
result = checker.check(circuit1, circuit2)
```

### Read outputs from `Result`, not `*SimParams`

`Simulator.run` no longer writes outputs onto the `*SimParams` instance you pass in. Capture the
return value and read fields from `Result`. `result.sim_params` still references your original
configuration object (unchanged).

| Old (`sim_params`)                          | New (`result`)                 |
| ------------------------------------------- | ------------------------------ |
| `sim_params.observables[i].results`         | `result.expectation_values[i]` |
| `sim_params.output_state`                   | `result.output_state`          |
| `sim_params.noise_model`                    | `result.noise_model`           |
| `sim_params.results` (weak)                 | `result.counts`                |
| `sim_params.measurements`                   | `result.measurements`          |
| `sim_params.multi_time_observables_times`   | `result.multi_time_times`      |
| `sim_params.multi_time_observables_results` | `result.multi_time_results`    |

Removed from `*SimParams`: `noise_model`, `output_state`, `multi_time_observables_times`,
`multi_time_observables_results`, `measurements`, `results`, `aggregate_trajectories`,
`aggregate_measurements`. Observable configuration (`observables`, `multi_time_observables`, etc.)
remains on `*SimParams`.

`Observable` no longer carries run outputs. After `Simulator.run`, read
`result.expectation_values[i]` (aggregated expectations), `result.trajectories[i]` (per-trajectory
data), and `result.times` (shared analog time grid). `result.observables[i]` is still the gate/sites
metadata for observable _i_.

### MPS bond diagnostics are automatic on `Result`

`runtime_cost`, `max_bond`, and `total_bond` are no longer configured as `Observable` instances.
For MPS-backed analog and strong-digital runs, `Simulator.run` fills `result.runtime_cost`,
`result.max_bond`, and `result.total_bond` (1D arrays aligned with `result.times` or the
strong-sim layer grid). MCWF, Lindblad, and weak digital runs leave these fields as `None`.

**Before:**

```python
sim_params = AnalogSimParams(observables=[Observable(Z(), 0), Observable("max_bond")])
result = sim.run(state, H, sim_params)
max_bond_curve = result.expectation_values[-1]
```

**After:**

```python
sim_params = AnalogSimParams(observables=[Observable(Z(), 0)])
result = sim.run(state, H, sim_params)
max_bond_curve = result.max_bond
```

**Before:**

```python
sim = Simulator()
sim.run(state, op, sim_params, noise_model)
print(sim_params.observables[0].results)
```

**After:**

```python
sim = Simulator()
result = sim.run(state, op, sim_params, noise_model)
print(result.expectation_values[0])
```

### `simulator.run` uses `State` and `Hamiltonian`

Analog and circuit entry points no longer accept raw `MPS` / `MPO` objects. Use `State` (from
`mqt.yaqs.core.data_structures.state`) and `Hamiltonian` (from
`mqt.yaqs.core.data_structures.hamiltonian`) instead.

**Before:**

```python
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.simulator import run

psi = MPS(4, state="zeros")
H = MPO.ising(4, J=1.0, g=0.5)
params = AnalogSimParams(..., solver="MCWF")
run(psi, H, params, noise_model)
```

**After:**

```python
from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.core.data_structures.state import State

psi = State(4, initial="zeros", representation="vector")
H = Hamiltonian.ising(4, J=1.0, g=0.5)
params = AnalogSimParams(...)
sim = Simulator()
result = sim.run(psi, H, params, noise_model)
```

### End of support for x86 macOS systems

Starting with this release, we can no longer guarantee support for x86 macOS systems.
x86 macOS systems are no longer tested in our CI and we can no longer guarantee that MQT YAQS installs and runs correctly on them.

## [0.3.2]

### End of support for Python 3.9

Starting with this release, MQT YAQS no longer supports Python 3.9.
This is in line with the scheduled end of life of the version.
As a result, MQT YAQS is no longer tested under Python 3.9 and requires Python 3.10 or later.

<!-- Version links -->

[Unreleased]: https://github.com/munich-quantum-toolkit/yaqs/compare/v0.3.3...HEAD
[0.3.2]: https://github.com/munich-quantum-toolkit/yaqs/compare/v0.3.1...v0.3.2
