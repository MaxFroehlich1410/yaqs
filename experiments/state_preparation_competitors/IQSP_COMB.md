# Compressed comb backend for Ballarin IQSP

The new `iqsp_comb_backend.py` executes pure and locally purified states on a
comb tree using the pinned author's compression and contraction kernels. A
comb has several chain-like teeth whose first sites form a backbone. This is
not the older uncompressed circuit-network contraction or merely a renamed
linear MPS implementation.

The backend supports local one-qubit gates, SU(4) gates during ideal training,
and two-qubit gates along tooth and backbone edges. It moves the orthogonality
center before compression. Closed-system bonds and local noise-environment
legs have separate caps and discarded-weight tolerances. Positivity follows
from the purification representation; the author's norm-restoring SVD rule is
retained. Tolerances are local squared-weight criteria, not guarantees on final
fidelity or its gradient.

## Scope and fidelity to the source

See [vendor provenance](vendor/ballarin_comb/PROVENANCE.md) for the pinned code,
source hashes, license, two tested indexing corrections, and explicit channel
adaptations. Adam, interpolation, final-iterate retention, and the existing
small-angle postprocessing have not changed. Archived experiments retain their
exact backend; no historical result is overwritten or relabeled.

- `hardware=None`: pure comb propagation.
- `hardware="H2"`: the paper's angle-dependent RZZ channel; noiseless one-qubit
  gates. Both endpoints, including backbone roots, receive noise.
- `hardware="IBM"` / `"Google"`: the existing common logical noise, including
  one-qubit and correlated two-qubit errors. These are benchmark adaptations.

The compressed candidate pipeline now defaults to published two-site TCI for
its single-tooth targets. See [TCI construction and settings](IQSP_TCI.md).
The wrapper also accepts an already constructed `CombTarget`. For existing fixtures
up to twelve qubits, `comb_target_from_vector` performs an exact leaf-by-leaf
SVD into the requested comb as an explicit historical diagnostic. This converter does not implement TCI target
construction and does not claim scalable dense target loading. Larger targets
must be supplied directly as comb tensors. Circuit truncation settings do not implicitly compress the target. An explicit
`relative_tolerance` in the converter (or `CombOptions.target_tolerance` in the
candidate pipeline) can remove numerical null directions; its default is zero.
Record and independently verify the target approximation before a campaign.

## Usage

```python
from experiments.state_preparation_competitors.iqsp_comb_backend import (
    CombLayout,
    CombOptions,
    CombTensorNetworkObjective,
    comb_target_from_vector,
)
from experiments.state_preparation_competitors.iqsp_tn import optimize_tn_adam

# target_vector and circuit are matched inputs in Qiskit qubit ordering.
layout = CombLayout(sites_per_tooth=6, teeth=2)
target = comb_target_from_vector(target_vector, layout)
objective = CombTensorNetworkObjective(
    circuit,
    target,
    hardware="IBM",
    options=CombOptions(
        checkpoint_interval=8,  # Bound saved autograd history by recomputation.
        closed_bond=64,
        environment_bond=16,
        closed_tolerance=1e-9,
        environment_tolerance=1e-12,
    ),
)
# Use an explicitly supplied initialization; none is silently substituted.
theta, history = optimize_tn_adam(objective, initial_parameters, steps=200)
receipt = objective.diagnostics(theta)
```

A circuit must follow the declared comb edges. The `iqsp_tn_candidates(..., comb_options=CombOptions())` API uses TCI targets and runs its complete
ideal/interpolation/noisy/postprocessing pipeline on two to twelve qubits with
a single-tooth comb. Twelve-qubit execution requires this explicit backend.
Omitting this option preserves exact execution and all existing defaults.
The old campaign CLI remains unchanged; this implementation does not start a
new experiment or select a twelve-qubit campaign's ansatz/budget.

Diagnostics report actual closed and environment ranks, stored state elements,
largest local merged tensor, local discarded weights, trace, channel, layout,
compression settings and source commit. `max_tensor_elements` guards local gate
and channel merges; it is not a limit on total autograd or overlap memory.

## Gradient caveat before a campaign

**The compressed author backend is not automatically interchangeable with the
exact backend at all-identity initialization.** The author's rank selection and
regularized SVD derivative can suppress entangling directions there. A Bell
state regression explicitly records equal fidelities but different gradients.
The implementation adds no random perturbation or alternative derivative to
hide this limitation. Resolving the initialization policy and auditing gradient
accuracy at the intended settings are prerequisites for a defensible comparison.

Away from rank transitions, tests compare both compressed gradients and
nontruncated gradients with independent finite differences. Tight compression
is compared with independent statevector/density calculations. A twelve-qubit
noisy comb is checked against an analytic product-channel reference, including
its angle derivative and bounded stored tensor sizes. These checks establish
numerical correctness within the stated scope, not twelve-qubit performance
or a global error guarantee.

## Tests

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-iqsp-tn/bin/python -m pytest \
  -o addopts='' \
  tests/optimization/test_state_preparation_competitors_iqsp_comb.py \
  tests/optimization/test_state_preparation_competitors_iqsp_tn_backend.py \
  tests/optimization/test_state_preparation_competitors_iqsp_tn.py \
  tests/optimization/test_state_preparation_competitors_run_iqsp_tn.py
```

## Twelve-qubit readiness audits

`q12_readiness.audit_gaussian_initialization` tests the actual Gaussian path at
lambda=0.05 using the unchanged deterministic initialization. It checks progress
and recovery of derivative agreement within twenty normal Adam updates. The
identity derivative is still approximate and is explicitly recorded; no exact
reference gradient is substituted into optimization. Passing this audit is
specific to this target/startup and does not remove the general limitation.
See [readiness documentation](Q12_READINESS.md) for the numerical criteria,
noisy compression checks and limitations on target construction and runtime.
