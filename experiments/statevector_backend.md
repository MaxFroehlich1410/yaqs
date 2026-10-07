# Small shared statevector backend

`mqt.yaqs.optimization.StatevectorBackend` executes state-preparation circuits on
complex128 vectors. Each trajectory stores `2**n` amplitudes (64 at six qubits).
Local gates act directly on batched vectors; optimization does not construct MPS,
MPO or full-system gate matrices. Forward/adjoint storage scales as
`O(gates * trajectories * 2**n)`; this is for small systems, not a replacement for
MPS at large qubit counts. There is no claimed speedup until measured.

## Supported operations

| Operation                             | Dense implementation                                                           |
| ------------------------------------- | ------------------------------------------------------------------------------ |
| Noiseless fidelity and batch gradient | Exact pure-state adjoint                                                       |
| Independent noisy gradient            | Average individual trajectory gradients once                                   |
| Online Krotov sweep                   | Old adjoints, updated forward prefix, ensemble average before each gate update |
| Hybrid Krotov                         | Caller schedules online sweeps followed by batch steps                         |
| Trajectory Adam                       | Same batch gradient, existing Adam moments and bias correction                 |
| Cross-trajectory signal               | Existing all-pairs estimator, with `O(trajectories**2)` pair work              |
| CRN and refresh                       | Caller retains maps or samples a new global counter                            |

The independent objective is `1 - mean(|<target|psi_r>|**2)`, not the fidelity of
an averaged statevector. The cross signal reproduces the MPS implementation; it
is not generally the derivative of this same finite-sample objective. Do not
attribute a new convergence guarantee to it. Shared parameters, affine gate
angles, reversed and nonadjacent two-qubit gates are supported. Vectors must be
finite and normalized and use little-endian indices; local matrix tensor factors
are ordered by ascending site as in `ParameterizedCircuit`. Data-encoding gates
are deliberately outside the state-preparation interface.

## Noise and randomness

The backend accepts the existing parameter-independent random-unitary providers,
composite instructions and Pauli TJM instructions. `ExactLocalPauliProvider` keeps
the exact commuting local Lindblad exponential, including multiple Pauli-factor
events. Legacy `TJMNoiseInstruction` keeps its different at-most-one-jump law;
switching representation does not silently replace that approximation.

Pauli no-jump scalar drift cancels on normalization, so the dense sampler stores
physical unitary maps without nonlinear normalization checkpoints. General
nonunitary jumps, normalized nonunitary replay, and noise-probability derivatives
are unsupported and must not be approximated by this adjoint. In particular,
amplitude damping requires additional implementation. Custom provider authors
must ensure probabilities/operators do not depend on the state or parameters;
that semantic restriction cannot be inferred from an arbitrary callable.

Sampling uses the production seed formula
`base_seed + 1_000_003 * iteration + trajectory_index`. The `iteration` is a
global counter, not a checkpoint-local step. Keep maps for fixed-ensemble CRN or
candidate-step comparisons; change the counter for refresh. `use_crn` does not
manage hidden sampler state. Separate training and validation base seeds.
Sampling respects gate masks; explicit replay uses exactly the supplied maps.

`compress_no_jump_maps(maps)` replaces duplicate empty operator histories with
one representative and returns empirical multiplicity weights. Thus 90 empty
histories and 10 jump histories become 11 paths with weights `0.90, 0.01, ...`.
The representative is propagated again at each parameter update. Jump histories
remain separate, even if their operators cancel. Probability weights are
supported by metrics, independent/cross gradients, and joint online sweeps;
cross signals use `w[r] * w[s]` for every pair. Compression matches the original
finite ensemble; it does not remove its sampling error. Inspect jump coverage
before propagation without rejecting or resampling banks based on their events.

```python
from mqt.yaqs.optimization import KrotovTJMOptions, StatevectorBackend
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider

engine = StatevectorBackend(circuit, target_vector)
sampling = KrotovTJMOptions(num_trajectories=32, random_seed=123)
provider = ExactLocalPauliProvider()

# Sample once and reuse within a gate-sequential sweep.
maps = engine.sample_noise_maps(theta, sampling, provider, iteration=1)
theta, signal = engine.online_update(theta, 0.7, maps)

# For batch updates or Adam, use the same shared gradient engine.
gradient, loss, fidelity = engine.gradient(theta, maps)
theta = theta - 0.1 * gradient

# Omitting maps selects the noiseless objective.
noiseless_loss, noiseless_fidelity = engine.metrics(theta)
```

## Experiment integration

The [compact development driver](state_preparation_trajectory_diagnostics.md)
adds explicit gradient/cross diagnostics, all-online and tail-rate controls,
fixed/periodic CRN, backtracking and adaptive-tail studies on this backend.
These are opt-in studies; they do not change the frozen five-method experiment.

The latest five-method runner accepts `--backend statevector`:

```sh
uv run python -m experiments.state_preparation_trajectory_comparison \
  --source-root PATH_TO_COMPLETED_HYBRID_EXTENSION \
  --output-root NEW_OUTPUT_DIRECTORY \
  --backend statevector --workers auto
```

It covers noiseless construction, pruning repair, and noiseless hybrid plus
trajectory Adam/batch/hybrid refinement. Pruning rules are unchanged. Exact-noisy
Adam and exact fidelity scoring still use the existing density-matrix oracle.
The reusable `state_preparation_hybrid_pilot.optimize_block` also accepts the
backend keyword; older standalone runner CLIs remain MPS-only. This is not a
universal switch for every historical experiment.

Defaults, hyperparameter grids, budgets, noise rates and existing outputs remain
unchanged. Backend choice enters manifests and checkpoint signatures. A fresh
output directory is required for a different backend or source revision; resume
preserves global noise counters, online/batch phase, Adam moments and timing.
Floating-point differences can grow under many aggressive online sweeps, so
numerical equivalence of individual operations is not a claim of bitwise-identical
long training runs. Scientific comparisons must record the backend and must not
mix their timing measurements.

## Verification

Tests compare gradients, losses, online sweeps and multi-update refinements with
untruncated MPS using identical sampled paths. They also check finite differences,
an enumerated exact Pauli ensemble against the density-matrix adjoint, Qiskit
bit ordering, input immutability, masks, unsupported noise rejection, interrupted
resume, checkpoint backend mismatches and serial/parallel reproducibility.
Dense-path tests forbid MPS construction to detect accidental fallback.
