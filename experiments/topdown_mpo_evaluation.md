# Forward MPO assessment of final noisy circuits

`topdown_mpo_evaluation` independently assesses frozen circuits; it does not train,
prune, select a depth, sample histories, or modify the input checkpoints. The
supported profiles are the existing historical logical `IBM` and `Google`
profiles and an explicit `noiseless` reference. Every original enabled gate,
including a zero-angle rotation, retains its original noise window. Noise-disabled
gates remain noise-free. The implementation sums the at-most-one-Pauli-jump
channel used by the archived trajectories, including the correlated two-qubit
Paulis. It does not substitute a full Lindblad exponential or native-gate noise.

## Representation and contraction

The initial state is the bond-one MPO for `|0><0|`. Local site indices are
`(ket, bra)` and YAQS site 0 is the least significant qubit. Each original
unitary conjugation and following noise channel are composed into a local
4-by-4 or 16-by-16 map. Nearest-neighbor updates split the fused operator by SVD;
QR sweeps put the orthogonality center at the active bond before each split.
The final fidelity is the direct MPS–MPO–MPS contraction `<target|rho|target>`.
The production evaluator never materializes a global density matrix, target
projector, statevector, or unitary. Dense fixtures may still require target
factorization before evaluation; supply native target MPS tensors where possible.

No bond cap and `--svd-atol 0` retain all SVD directions. Reports label a result
`exact_up_to_roundoff` only if no nonzero singular tail was discarded. A cap or
nonzero tolerance can make a run approximate. `--svd-atol` is an **absolute
Frobenius tail-norm budget per canonical split**, not an individual singular-value
threshold or a relative weight. Results are neither trace-renormalized nor
clipped to `[0,1]`. Generic SVD compression does not enforce positivity.

For these unital random-unitary channels, Hilbert–Schmidt distance is contractive.
Canonical splits give the actual discarded Frobenius norm of the represented
operator. Therefore the sum of these discarded norms bounds the final
Frobenius density error and the absolute pure-target fidelity error, excluding
floating-point roundoff. The report includes this conservative bound, trace
drift, retained bonds/storage, split counts, and estimated operation workspace.
This argument is specific to the supported unital channels; it must not be
extended to arbitrary nonunital channels without a new bound.

`--max-workspace-mib` defaults to 1024 and rejects oversized estimated local
operations before their main allocation. It is a numerical-workspace estimate,
not an operating-system RSS limit. Dense target factorization and native-library
workspace must also fit the host. Exact MPO bonds can grow exponentially; an
out-of-budget run must not silently switch to truncation.

## Assessing a saved final circuit

Supply a sealed checkpoint containing `theta` and a sealed checkpoint containing
its retained `circuit`. The circuit may also be nested under `training`. Existing
`complete.json` hashes are verified with the shared checkpoint loader.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run python -m experiments.topdown_mpo_evaluation \
  --checkpoint CASE/selection \
  --circuit-checkpoint CASE/pruning \
  --target-id tfim_critical --hardware IBM \
  --output output/mpo-assessment/final.json
```

Read the original final record's `selected_parameters` to choose the returned
endpoint. Historical cases can select `batch`; newer cases may select
`selection`, `online`, or another saved iterate. A standard `topdown_mps` case
uses `depth_XX/refinement` and `depth_XX/pruning`. Do not assume the last iterate
is the selected one. Outputs must use a fresh file. The report identifies the
parameter/circuit seals, target amplitudes, evaluator source, profile, and all
approximation settings. Frozen campaign manifests and ranking files are untouched.

Use `--max-bond-dim N` and/or `--svd-atol TOL` only for separately labelled
convergence runs. An approximate MPO score is not an exact replacement for
held-out trajectory assessment. The runner can accept twenty-qubit target
MPS inputs, but no twenty-qubit campaign is launched by this tool or validation.

## Independent six/twelve-qubit validation

The separate diagnostic `topdown_mpo_validation` uses Qiskit gate constructors
and the independently constructed literal Pauli channel from
`legacy_topdown_dense.noise_channel`. It propagates the full density and compares
all matrix entries through a Frobenius norm, as well as fidelity, trace, and
Hermiticity. Full eigenvalue checks are included at six qubits. At twelve qubits,
positivity is supported by the independent CPTP propagation and agreement of the
full output matrices; a full 4096-by-4096 spectral decomposition is not performed.
Dense conversion is explicitly rejected above twelve qubits.

The controlled circuits cover all six scalar rotation primitives, shared affine
parameters, reversed gate order, asymmetric CX gates, zero-angle noise windows,
and disabled noise. They connect the entire register. Targets can be selected
from the unchanged six/twelve-qubit fixture.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run python -m experiments.topdown_mpo_validation \
  --output output/mpo-validation/controlled.json
```

This checks both widths under IBM, Google, and noiseless profiles. Repeatable
`--num-qubits` and `--hardware` select subsets. To check an actual returned
circuit instead, provide `--checkpoint`, `--circuit-checkpoint`, and `--target-id`
with the same meanings as above. Compression checks accept `--max-bond-dim` and
`--svd-atol`. Reports distinguish complete validation from partially written
rows and preserve source/checkpoint provenance.

The automated tests additionally check canonical compression bounds, sufficient
bond caps, matrix-free assessment, workspace rejection, target/register validation,
and sealed input/output behavior. The local validation artifacts are in
`output/forward-mpo-validation-20261019/`; generated reports remain excluded
from Git under the repository's existing output policy.

## Completed validation on 2026-10-19

All cases below retained every SVD direction and passed. Errors compare the
MPO result with independent full-density propagation, not a trajectory estimate.

| Circuit                                      | Width | Profiles               | Maximum fidelity error | Maximum full-density Frobenius error | Maximum retained MPO bond |
| -------------------------------------------- | ----: | ---------------------- | ---------------------: | -----------------------------------: | ------------------------: |
| Controlled connected circuit                 |     6 | IBM, Google, noiseless |               4.03e-16 |                             3.86e-15 |                         4 |
| Controlled connected circuit                 |    12 | IBM, Google, noiseless |               1.96e-18 |                             3.05e-15 |                         4 |
| Saved final Gaussian circuit, 35 gates       |     6 | IBM, Google            |               1.34e-15 |                             1.83e-15 |                         4 |
| Saved final Gaussian circuit, 66 gates       |    12 | IBM                    |               1.12e-15 |                             2.44e-15 |                         4 |
| Saved final critical-TFIM circuit, 118 gates |    12 | IBM                    |               5.56e-15 |                             2.00e-14 |                       256 |

The saved q6 Gaussian endpoint is from
`topdown-mps-implementation-20260923/gaussian-dense`. The q12 endpoints are
`gaussian_mu0p5_sigma0p1__seed300` and `tfim_critical__seed300` from the retrieved
`selected-G-noisy-q12-20261017` campaign; their final records select the `batch`
parameters. Parameter and circuit seals are recorded in the JSON reports.
Source snapshots corresponding to those reports are retained in the local
validation output's `source/` directory. `controlled-final.json` additionally
uses the current diagnostic's explicit completion-status reporting.

The critical-TFIM MPO retained approximately 4.4 MiB of tensor data at its peak.
This excludes contraction/SVD temporaries, target tensors, imports, and the
separate dense reference. These checks establish numerical parity for the
listed circuits; they do not establish affordable exact bond growth for every
structured twenty-qubit circuit. No twenty-qubit evaluation was run.
