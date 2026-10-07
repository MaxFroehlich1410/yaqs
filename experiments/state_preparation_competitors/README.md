# Six-qubit state-preparation comparison protocol

Future performance experiments use **noise-aware training methods only**:
our incumbent, NACL, RobustState, and Fontana on its real-target subset. Use the
[active noisy-method protocol and runner](NOISY_METHODS.md). Fixed VQSP and
layerwise remain internal controls; IQSP is on hold pending its initialization
fix. AQCE and BAA-LRSP are excluded from future comparisons because they do not
train with noise. Their code and completed results remain historical references.

The older `run` driver defaults to `current_b_lbfgs_frozen`, `fixed_vqsp` and
`layerwise`. Historical synthesis adapters require explicit `--methods`
selection. The broader protocol below documents their retained implementation
and historical grids; it is not the active experiment scope.

This experiment compares the current pruning-B method with independently
implemented preparation methods under one common compiled-circuit evaluation.
It reuses unchanged MNISQ numerical routines with the original paper's AQCE
schedule, and the official QCLib synthesis code. The full
comparison sweep has not been run; implementation validation and small smoke
checks are separate from benchmark results.

This is a new deployment experiment. Its compiled-circuit fidelities are not
expected to equal earlier logical-circuit results. The protocol alone establishes
neither a ranking nor a claim of superiority over the state of the art.

## Targets and repetitions

Use exactly the nine **six-qubit** vectors in
[`state_preparation_target_states.json`](../../benchmarks/state_preparation_target_states.json),
as specified by the
[benchmark definition](../../benchmarks/state_preparation_benchmarks.md):

| Target family      | Fixture identifiers                               |
| ------------------ | ------------------------------------------------- |
| Gaussian           | `gaussian_mu0p5_sigma0p1`                         |
| TFIM ground states | `tfim_ferro`, `tfim_critical`, `tfim_para`        |
| Haar random        | `haar_random_1`, `haar_random_2`, `haar_random_3` |
| Random MPS         | `random_mps_bond2`, `random_mps_bond3`            |

Load the stored complex amplitudes directly, in their documented little-endian
order. Do not regenerate targets, discard imaginary parts, or substitute
approximately reconstructed states. Evaluation uses dense state vectors and
64-by-64 density matrices, without MPS truncation.

The variational methods use optimizer seeds **300 and 400**, under both synthetic
profiles labelled **IBM** and **Google**. AQCE and BAA are deterministic and are
constructed once per target and candidate grid. Repeating a nominal seed does
not create an independent optimizer replicate for those methods. Report their
individual outcomes without inflating the replication count.

These targets have already been used to develop the current method. Results
are comparisons on known development fixtures, not an unseen-target
generalization test. Two optimizer seeds support descriptive paired comparisons;
they do not establish broad statistical robustness.

## Methods and declared initial grids

| Method identifier        | Implementation and candidate grid                                                                                                                                                                                                                                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `current_b_lbfgs_frozen` | Frozen selected pruning-B + online Krotov + L-BFGS endpoint for each target, optimizer seed, and profile from `output/topdown-batch-refinement-q6-20260919`. Reuse the circuit and parameters, then apply the common deployment evaluation below.                                                                              |
| `aqce`                   | Published AQCE Algorithms 1–5, using unchanged complex MNISQ tensor routines and local scheduling; full two-qubit unitary budgets **0, 5, 10, 20, 40, 80**, with **M0=1**, **delta M=1**, and **100 complete forward/backward sweeps** after initialization and each growth step. All independent budget restarts are charged. |
| `baa_lrsp`               | Original QCLib **0.1.15** BAA-LRSP, `strategy="brute_force"`, `use_low_rank=True`, squared-infidelity bounds **0, 0.001, 0.01, 0.03, 0.1, 0.2**. Zero is an exact-synthesis control.                                                                                                                                           |
| `fixed_vqsp`             | Fixed-depth BMPD circuits, independently initialized at depths **1–16**, optimized using the original SciPy **1.17.1** `minimize(method="L-BFGS-B")`; up to **200 noiseless + 200 noisy iterations** per depth. No pruning or parent warm start.                                                                               |
| `layerwise`              | Internal unpruned control, depths **1–16**, reusing the current online-Krotov-then-L-BFGS construction and unchanged legacy noisy fine-tuning. Each depth inherits its best **noiseless** parent, not its noisy endpoint. Default construction and noisy budgets are each **200 updates**, with **100 online updates** first.  |

These are initial declared grids, not grids chosen after inspecting test
fidelities. Record all overrides in the run manifest before starting. The method
adapters retain all requested candidates; they do not stop at 90% ideal fidelity.
The archived reference retains its historical depth-selection decision and
construction stopping rule rather than receiving a new search budget here.

The default AQCE grid is substantial: because each requested budget restarts,
it schedules **868,150 SVD gate updates per target**: 868,000 updates in complete
sweeps and 150 in the new growth blocks, plus five density-initialization gate
choices. Each choice scans all 15 qubit pairs at six qubits. The budget-80 run
alone schedules 648,079 SVD updates plus one density-initialization choice.
These are schedule counts, not runtime predictions. Use an explicitly
labelled smaller grid for smoke checks; do not interpret such checks as the
full-grid experiment or assume the default command is a quick test.

The reference was selected previously under logical noise, whereas new candidate
grids are selected under the common compiled validation model. This asymmetry is
explicit: the reference measures deployment of an already selected method. It
does not establish what that method would achieve if every archived depth were
reselected or retrained for native noise.

### Upstream reuse and exact attribution

AQCE follows [Algorithms 1–5 of the original published paper](https://doi.org/10.1103/PhysRevResearch.6.043008).
The local `aqce_schedule.py` reuses the pinned MNISQ `evaluate_F` and
`evaluate_Rho` routines unchanged. It initializes `M0` gates once from density
eigenvectors, performs complete forward/backward sweeps, and grows by inserting
`delta M` identities at the input end. Only the new block receives the initial
backward update against the residual target before whole-circuit sweeps resume.
Environment states propagate between gate updates as in the paper. Every
update may change both a gate's matrix and its qubit pair. The original MNISQ
forward-only driver remains preserved but is not called by the active adapter.
No pruning, noisy fine-tuning, or optimizer substitution is added.

The configurable defaults `M0=1`, `delta M=1`, and `N=100` are declared benchmark
choices, not the original paper's particular numerical experiment settings.
Small `M0` can leave zero initial overlap for some sparse targets; no special
remedy is added to the published procedure. Each budget starts independently.
The method is deterministic and all-to-all; routing costs are charged by the
shared compiler. See [AQCE provenance](AQCE_PROVENANCE.md) for source-search
scope, the upstream commit, retained license, and numerical limitations.

Earlier AQCE smoke results and runtime estimates used the forward-only driver
and repeated density-based growth. They do not validate performance or timing
of the corrected schedule. Use fresh outputs and new timing measurements; the
manifest's source and configuration checks prevent resuming old results as new.

BAA calls the official QCLib initializer without replacing its approximation
search or LRSP synthesis. The wrapper expands existing gate definitions and
independently measures actual **squared** fidelity against each requested bound.
See [BAA provenance](BAA_PROVENANCE.md) for the release, audited source hashes,
paper, options, and optional dependency pins.

The fixed VQSP control uses SciPy's conventional optimizer and the repository's
exact objective/gradient backend. It is a generic fixed-ansatz state-preparation
control, not a claimed reproduction of a particular VQA paper. The layerwise
control is an internal unpruned method, not an external competitor. Its full
depth grid differs from the archived reference's stopping rule, so this
comparison alone does not isolate a causal pruning effect.
**Genuine operator-pool ADAPT is deferred**; the unpruned layerwise control must
not be labelled ADAPT.

## One common deployment circuit

Compile every source circuit, including the archived reference, with Qiskit
**2.4.1** and identical settings:

- Native basis: **`rz`, `sx`, `x`, `cx`**.
- Six-qubit nearest-neighbor line coupling map, with identity initial placement.
- `routing_method="basic"`, `optimization_level=1`,
  `seed_transpiler=1729`, `approximation_degree=1.0`.
- No ancillary qubits, measurements, resets, classical control, or unresolved
  parameters.

Check ideal-unitary equivalence independently, allowing global phase and
accounting for the compiler's final wire permutation. Evaluate against the
correspondingly permuted target. Charge inserted routing gates, including their
noise, and report native one-qubit count, **CX count**, **two-qubit depth**, and
total depth. Raw AQCE SU(4) counts, logical rotation counts, and compiled CX counts
remain different metrics; they are never treated as interchangeable gate costs.

Qiskit 2.4.1 can approximate near-symmetric opaque two-qubit unitaries even
with these settings. If the full unitary audit fails, the compiler replaces
affected opaque gates using Qiskit's unspecialized exact three-CX decomposition,
then applies the same native compilation and repeats the audit against the
original source at the unchanged \(10^{-9}\) tolerance. Receipts record this
fallback and its gate count; all resulting gates and noise are charged. Use a
new output directory after this fix: old compiled receipts are not interchangeable.

## Synthetic noise law and training boundary

The IBM/Google names identify the historical synthetic profiles; they are **not
hardware calibration snapshots**. Retain their literal coefficients:

| Profile | Local coefficient `a1` | Correlated coefficient `a2` | Two-qubit correlated Paulis |
| ------- | ---------------------: | --------------------------: | --------------------------- |
| IBM     |                 `3e-4` |                      `3e-3` | `XX`, `ZZ`                  |
| Google  |                 `6e-4` |                      `5e-3` | `XX`, `YY`, `ZZ`            |

Each acted-on qubit contributes `X`, `Y`, and `Z` with coefficient `a1/3` each.
A two-qubit gate additionally contributes each listed correlated Pauli with
coefficient `a2 / number_of_correlated_Paulis`. Thus total coefficient
\(\Lambda=a_1\) for a one-qubit operation and
\(\Lambda=2a_1+a_2\) for a two-qubit operation. These coefficients are not
average gate infidelities or Bernoulli error probabilities.

The primary law is the existing at-most-one-jump gate-window channel:

\[
\mathcal E(\rho)=e^{-\Lambda}\rho+
(1-e^{-\Lambda})\sum_j\frac{a_j}{\Lambda}P_j\rho P_j^\dagger.
\]

Apply this channel **after every compiled operation, including `rz`**. No virtual
zero-error RZ assumption, idle noise, readout noise, device-specific connectivity,
or device calibration is implied. This convention makes every method's native
operation costs comparable under the same specified synthetic model. It is not
the full repeated-jump Lindblad semigroup; any such sensitivity result must be
labelled separately.

Existing variational **training retains the historical logical-gate noise
placement**. The current and layerwise methods keep their existing noisy update
policy; fixed VQSP uses SciPy on its frozen weighted noisy objective. AQCE and BAA
are used as published synthesis implementations, without adding our optimizer or
noise fine-tuning. Consequently this study compares complete preparation
procedures at deployment, not optimizers trained on an identical native-noise
objective.

## Training, selection, and final assessment

1. For each noisy variational candidate, draw **M=100 unconditional training
   trajectories once**, then freeze the common random numbers. Merge duplicate
   no-jump histories into one representative with weight `n_no_jump / 100`;
   retain each jump history with its empirical multiplicity. This exactly
   preserves that sampled objective and its weighted gradient.
2. Inspect and record raw jump counts. A nonempty noisy circuit whose bank has
   no jump history is explicitly flagged as a coverage failure and is ineligible
   for ordinary winner selection. Do not force jumps, resample until a jump
   appears, or silently change M. An empty circuit legitimately has no noise
   events and is not a coverage failure.
3. Compile every eligible candidate and score it using **32,768 fresh fixed
   validation trajectories** from a declared namespace disjoint from all training
   streams. Compression may again merge duplicate no-jump paths with their
   original empirical weights. Keep the raw sample count, event coverage, bank
   identity, and sampling standard error. A nonempty compiled circuit with no
   validation jump history is also ineligible; an empty circuit remains valid.
   Never condition or resample the validation bank.
4. Select using compiled **validation fidelity**. Exact ties prefer fewer native
   CX gates, then smaller two-qubit depth, then lexical candidate identifier.
   Persist the selected index, candidate records, and compiled circuits as an
   authenticated selection lock. The frozen reference remains its historical
   selected endpoint. If no candidate is eligible, save `selected_index=null`
   and status `no_eligible_candidate`; do not substitute a failed candidate.
5. **Only after selection is locked**, propagate every candidate circuit through
   the exact density-matrix channel and report
   \(F*{\rm exact}=\langle\psi*{\rm target}|\rho|\psi\_{\rm target}\rangle\).
   This integrates all events in the specified channel and introduces no final
   trajectory sampling error. The primary endpoint is the already selected
   candidate. Other exact scores, including scores for coverage-failed
   candidates, support explicitly post hoc descriptions of the candidate grid;
   they do not restore eligibility. Never use these final values to tune the
   grid, choose a depth, or replace a locked winner.

Independent final density evaluation prevents training/validation trajectory
reuse from becoming a test metric. Validation uncertainty and optimizer-seed
variability remain relevant and should be reported separately from the exact
channel value. A failure must remain an explicit record rather than becoming an
invented zero fidelity or an omitted unfavorable result.

## Costs and interpretation

Record actual construction, noisy training, upstream synthesis, all candidate
restarts, compilation, validation, and final-evaluation costs. Report objective
and gradient calls, trajectory counts before and after compression, and CPU and
wall time where available. CPU belongs to the whole requested candidate grid,
not only the winning candidate. Include completed coverage-failed candidates and
charged prechecks. The saved timing scope excludes work lost in interrupted,
unsealed stages; account for that separately if interruptions occur during a
scientific timing campaign. Deterministic synthesis shared between profiles or seed rows must not
be charged as independent repeated execution.

Keep **historical archived optimization cost** separate from **cost incurred in
this run**. Reusing an archived endpoint does not mean that constructing it was
free, nor does it justify charging that historical computation as newly spent
CPU. These grids do not impose equal wall-time, CPU, oracle, or gate budgets;
report fidelity/resource/cost tradeoffs without calling this a matched-budget
study. Any later matched-budget or new-target study needs its own declared
protocol.

Report target-level outcomes and paired variational-seed differences before
aggregates. Keep logical and compiled statistics separate. A favorable result
on these nine fixtures would support a benchmark-specific claim, not general
state-preparation superiority or a hardware-performance claim.

## Running the comparison

### Compact, authenticated histories

Storage schema **v2** keeps each candidate's full construction metadata in its
construction bundle only. Per-iteration parameter vectors, where collected by
the optimizer, are stored once as lossless float64 matrices in
`histories.npz`. Matrix rows correspond exactly to the optimizer's scalar
`records` entries, including the initial iterate when present. Fidelity curves,
timings, evaluation counts, seeds, initial/best/final parameters, noise-bank
receipts and provenance remain in `record.json`. No optimizer iterates are
dropped, and no training or selection setting changes.

Selection, final and summary records contain candidate identifiers and a
`construction_reference` with a campaign-relative path and construction-seal
hash, plus the candidate index. They no longer embed the full metadata or
parameter histories. These references survive moving the complete output
directory. Construction seals cover JSON, QPY and the optional NPZ; resumed
runs authenticate all three, and readers reject altered or missing arrays,
invalid shapes and mismatched candidate identities. NPZ loading never enables
object pickles.

For a fixed-VQSP result, read its noiseless parameter history with:

```python
import json
from pathlib import Path

from experiments.state_preparation_competitors.run import (
    load_candidate_metadata,
    load_parameter_history,
)

root = Path("YOUR_OUTPUT_DIRECTORY")
summary = json.loads((root / "summary.json").read_text())
method = summary["groups"][0]["records"][0]["method"]  # Choose a fixed-VQSP row.
metadata = load_candidate_metadata(root, method)
phase = metadata["construction"]
folder = root / method["construction_reference"]["path"]
theta_history = load_parameter_history(folder, phase["parameter_history"])
fidelities = [entry["fidelity"] for entry in phase["records"]]
```

Use `metadata["noisy_training"]` for its noisy history. Methods whose existing
diagnostics contain scalar curves without per-iteration parameters retain those
curves as before and have no artificial binary history added.

Use a **new output directory** for v2. Earlier v1 archives are preserved; their
large nested JSON layout is not silently rewritten or resumed under v2. The
earlier approximately 30 GB storage projection described that v1 duplication,
not this compact format.

### Commands

Optional synthesis dependencies are installed in an isolated `uv` environment;
the project lock is not changed. This pinned environment was validated with
Python 3.11:

```sh
uv run -q --with-requirements experiments/state_preparation_competitors/requirements.txt \
  python -m experiments.state_preparation_competitors.run \
  --output NEW_DIRECTORY
```

The runner also accepts a restricted target/method selection and an explicit
archive root, for example:

```sh
uv run -q --with-requirements experiments/state_preparation_competitors/requirements.txt \
  python -m experiments.state_preparation_competitors.run \
  --output NEW_DIRECTORY \
  --targets gaussian_mu0p5_sigma0p1 \
  --methods current_b_lbfgs_frozen fixed_vqsp layerwise \
  --reference-root output/topdown-batch-refinement-q6-20260919
```

Additional controls are `--hardware IBM Google`, `--seeds 300 400`, `--depths`,
`--aqce-gates`, `--aqce-sweeps`, `--aqce-initial-gates`,
`--aqce-growth-increment`, `--baa-losses`, `--construction-steps`,
`--online-steps`, `--noisy-steps`, `--noisy-online-steps`, `--training-paths`, and
`--validation-paths`. The layerwise method requires contiguous depths starting
at one. AQCE zero is a separate empty-circuit control; each positive budget must
be at least `--aqce-initial-gates` and differ from it by an integer multiple of
the positive `--aqce-growth-increment`. `--aqce-sweeps` counts complete
forward/backward sweeps. BAA CLI bounds must lie in `[0, 1)`; its adapter adds zero if omitted.
The archived reference accepts only its original seeds 300 and 400. Controls
must be internally valid even when a chosen method does not use every control.

For a deliberately small internal-control smoke check:

```sh
uv run -q --with-requirements experiments/state_preparation_competitors/requirements.txt \
  python -m experiments.state_preparation_competitors.run \
  --output NEW_SMOKE_DIRECTORY \
  --targets gaussian_mu0p5_sigma0p1 --methods fixed_vqsp layerwise \
  --hardware IBM --seeds 300 --depths 1 \
  --construction-steps 2 --online-steps 1 \
  --noisy-steps 2 --noisy-online-steps 1 --validation-paths 512
```

This small bank is for integration testing only; a coverage failure is an
expected possible outcome and must not trigger resampling.

Use a new output directory for a new protocol. Preserve the fixture, source,
upstream, dependency, compiler, grid, and reference-artifact provenance alongside
the run. Repeating the same command automatically resumes sealed completed
stages; no `--resume` flag is required. A changed configuration or hashed source
requires a new directory. Metadata and QPY circuit bundles are checked against
their saved seals before reuse. The runner writes `summary.json` once the
requested groups complete.

The adapter/evaluation validation includes actual pinned numerical routines,
not only mocked calls. AQCE's published schedule is checked against independent
transition-matrix, gate-order, and multi-gate growth oracles; direct historical
MNISQ executions remain useful for validating the reused tensor conventions,
not as expected outputs of the corrected schedule. BAA is
checked against original gate definitions, complex q2/q3 examples, its fidelity
bounds, deterministic repetition, and a complex q6 exact endpoint.
This validates implementation boundaries, not scientific performance.
Run the targeted tests with:

```sh
uv run -q --with-requirements experiments/state_preparation_competitors/requirements.txt \
  pytest -n 0 tests/optimization/test_state_preparation_competitors_*.py
```

The commands do not imply that the full sweep has already completed. Report
smoke checks and full benchmark results separately.
