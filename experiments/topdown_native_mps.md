# Native-MPS thirty-qubit scaling pilot

This opt-in extension creates a real random target directly as open-boundary
MPS tensors. It uses the same independent standard-normal tensor construction
as the existing random-MPS family, with target seed 5002 and bond dimension 2
by default. The optimizer seed remains separately fixed at 300. No dense target,
state-vector screening pool, or density matrix is constructed.

The NPZ payload contains one complex128 `site_i` array per qubit in
physical/left/right order. Its JSON manifest records the width, generation seed,
bond, basis convention, tensor bytes and SHA-256 checksum. Loading checks those
bytes, boundaries, finite entries, normalization, phase and regeneration from the
declared seed. Each load returns private arrays. Existing files are never
overwritten. Thirty qubits at bond 2 require 3,712 bytes of tensor data, plus
the small archive and manifest overhead.

The global phase fixes the all-zero amplitude positive, rather than searching
exponentially many amplitudes for the largest. This changes no physical state.
Small-width tests compare against the archived dense fixtures up to this global
phase. The archived six/twelve/twenty/twenty-four-qubit fixture APIs and their
supported-width constants are unchanged; this format is a separate interface.

## Preparation and precision

Product screening contracts exactly the same zero-protected pool of 1,000
random single-qubit products using MPS transfer environments. Random streams,
little-endian site order, selected initial U3 angles and the remaining brickwall
parameters match the earlier dense screening procedure.

The new runner reuses our existing depth-growth preparation, B/G deletion rules
and fixed-bank noisy refinements. It explicitly constructs the already tested
`CanonicalMPSBackend` with relative propagation cutoff **1e-8** for preparation,
pruning replay and noisy refinement. Singular values below that fraction of the
largest are discarded; there is no hard bond cap or trajectory renormalization.
Target/input canonicalization retains machine epsilon independently. The cutoff,
source hashes, full budgets and target identity are sealed in the pilot plan;
stage claims prevent accidental retries or mixing checkpoints.

Using this backend for pruning means its deletion scores are approximate replay
scores, unlike the original exact twenty-qubit pruning stage. This is recorded
as a new pilot, not pooled silently with the archived experiments. Final noisy
fidelities use the existing **untruncated forward MPO** on retained circuits.
That assessment has a separate workspace guard and bond-growth diagnostics;
the training cutoff never silently truncates the final score. The target is
defined by its stored tensors, with roundoff limitations recorded separately.

## Execution

Create a new directory and tensor target without starting optimization:

```sh
uv run python -m experiments.topdown_native_mps create \
  --root output/native-mps-q30-NEW --qubits 30 --bond 2 \
  --target-seed 5002 --seed 300 --max-depth 4
```

Run preparation and then an explicit B branch and its final assessment:

```sh
uv run python -m experiments.topdown_native_mps run \
  --root output/native-mps-q30-NEW --stage prepare
uv run python -m experiments.topdown_native_mps run \
  --root output/native-mps-q30-NEW --stage B
uv run python -m experiments.topdown_native_mps run \
  --root output/native-mps-q30-NEW --stage B_selected_mpo
```

`--stage all` runs the original seven stages, including both B/G branches and
their pruned/final assessments. `create` defaults to 30 qubits, bond 2 and depth
at most 4; the inherited per-depth and refinement budgets remain in `plan.json`.
For a deliberately shortened smoke test, pass explicit `pilot.Settings` to
`create` from Python instead of modifying a sealed plan.

Server deployment must retain full existing job reservations and use an external
process memory/wall guard. Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
`RAYON_NUM_THREADS=1` and `QISKIT_PARALLEL=FALSE` before launching Python. The CLI
also limits numerical thread pools to one. This module does not deploy a remote
campaign or change any active comparator job.

## Validation

Tests cover the archived dense family, complex product overlaps, identical
initialization streams, target corruption and relabelled seeds, private loads,
zero-state retention, real thirty-qubit generation/screening/preparation and a
complete weighted noisy update with dense APIs trapped. Small full B/G pilots
use actual pruning and optimizer kernels, followed by an independent Qiskit
full-density comparison of the final MPO scores. A tiny Schmidt component test
checks actual propagation truncation while preserving target precision.

Run the targeted regression set:

```sh
uv run pytest -n 0 tests/optimization/test_native_mps_targets.py \
  tests/optimization/test_topdown_native_mps.py \
  tests/optimization/test_topdown_product_screening.py \
  tests/optimization/test_topdown_structured_q20.py \
  tests/optimization/test_topdown_structured_q20_canonical.py \
  tests/optimization/test_canonical_mps_backend.py \
  tests/optimization/test_state_preparation_binary_targets.py
```
