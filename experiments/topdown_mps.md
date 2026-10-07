# Current top-down method: MPS and statevector backends

`python -m experiments.topdown_mps` implements the current \*\*B + online Krotov

- L-BFGS construction + weighted cross-trajectory refinement\*\* pipeline with a
  selectable numerical representation. Its default search fine-tunes every depth
  candidate, the optional search used by the current reference comparison.
  Implementation tests establish numerical agreement, not a new fidelity or
  scalability result. The old benchmark archives remain historical references;
  start this runner in a new output directory.

## Preserved scientific settings

- Construction: 100 online sweeps at 0.7, followed by 100 safeguarded L-BFGS
  updates; restore the best actual parameters, including the initial iterate.
- Growth: original brickwall circuit and initialization, unpruned best parent,
  depth cap 16, independent validation and the existing three-step patience rule.
- Pruning B: signed noiseless single-gate deletion damage, scored once;
  simultaneously delete gates with damage strictly below 0.01. No repair or
  cumulative-fidelity safeguard is added. Empty circuits remain valid diagnostics.
- Refinement: 100 online and 100 batch updates at 0.2; return the final iterate.
  Use the original state-independent logical IBM/Google one-jump Pauli law.
- Sample 100 complete training histories once. Keep every jump history and one
  representative of the empty histories with their empirical multiplicity.
  Never condition on a jump, force jumps, or replace a zero-jump bank.
- Both indices of the cross signal carry their own weights:

  ```math
  s_i=-2\operatorname{Re}\sum_{r,s}w_rw_s
      \langle\chi_{r,i}|A_i|\psi_{s,i}\rangle
      \langle\psi_{s,i}|\chi_{r,i}\rangle.
  ```

  The same weights apply inside every sequential online gate update. This is
  the historical cross signal, not a claim that it differentiates the finite
  sampled mean fidelity. L-BFGS remains confined to noiseless construction.

The MPS backend lives in `src/mqt/yaqs/optimization/mps_backend.py`; it reuses
YAQS local MPS operators and circuit generators. Both numerical backends share
`TrajectorySampler`, which allocates no quantum state. The noiseless L-BFGS
policy and noisy optimizer loops are reused directly from the reference code.
The earlier public MPS training functions retain their historical uniform-weight
behavior; the full weighted method uses the new `MPSBackend` interface.

### Contraction implementation

Mixed overlaps use batched matrix multiplications and cached left/right
environments, without assuming the two states share a canonical basis. The
ordinary overlap and local derivative insertion reuse the same outside
environments. Unchanged private tape tensors are shared; local gates replace
affected arrays. Changes to either bra or ket invalidate every affected cached
prefix/suffix. Caches are bounded (32 MiB), pair batches adapt to tensor sizes,
and all cached references are released after each optimizer call, including
exceptions. Cache bytes exclude transient operands and are reported separately.

The derivative still uses the existing local gate/SVD kernel, preserving the
previous behavior even with bond truncation. No additional QR sweep, truncation,
renormalization, trajectory compression or parameter update is introduced.
Cross-estimator online sweeps omit the unused preliminary forward pass;
independent-gradient sweeps retain it because their costates need its overlaps.
Local gate/SVD results are also reused across paths with object-identical local
input tensors. This shares computation only: all trajectories and both cross
weights remain present. Distinct or merely close tensor inputs are never merged.
SVD work counters and summed discarded weights describe executed kernels;
an identically shared split is counted once, not once per receiving trajectory.
Online sweeps record the norm error of the returned forward states as well.
These changes preserve the mathematical algorithm, with a different floating-
point contraction order. Use a fresh source-authenticated run for new timings.

## Exact and approximate execution

Default MPS execution has no bond cap and zero SVD threshold. Targets are
factored once per case and remain cached as tensors. No dense state or density
matrix is needed when an MPS target is supplied directly to `run_case`.
The fixture loader still reads dense JSON amplitudes for six/twelve qubits;
this input format is not a scalable representation for arbitrary large targets.

`--max-bond-dim` and `--svd-threshold` explicitly enable approximation.
`--compress-target` additionally compresses the cached target and records its
infidelity against the original. Validation and final assessment always use
the original target and untruncated propagation, including after approximate
training. Dense execution rejects MPS-only settings.

With truncation, adjoints omit derivatives of SVD truncation decisions, and
prefix/suffix pruning scores are approximate contractions. They are not exact
sensitivities of a nonlinear truncated simulation. No automatic normalization
hides lost trajectory norm. Report convergence before using approximate runs
for scientific comparisons. Summed local discarded SVD weights are diagnostic,
not a rigorous bound on final infidelity.

## Assessment and resource accounting

Validation uses the original independent 32,768-history bank and its original
raw-sample standard error. Histories are propagated one at a time, compressing
no-jump duplicates with their multiplicities. This avoids an ensemble-sized
state allocation. The source noise law and logical gate windows are unchanged.

Selection is written to `selection_locked.json` before final assessment. By
default the selected circuit is tested on four fresh independent batches of
8,192 histories. Reports contain a normal-approximation 95% interval from
trajectory sampling and the standard error of the independent batch means.
Training, validation and final banks use separate seed roles; jump coverage is
reported, and coverage failures are explicit rather than silently resampled.
The default final score is therefore a Monte Carlo estimate. The old exact
six-qubit score is available through `--exact-density-check`, limited to q6.

Resource reporting includes logical and compiled two-qubit gate counts and
dependency-aware depths. Compilation is verified by untruncated MPS prepared-state
agreement up to global phase; it no longer constructs full-system unitaries.
This is preparation-state verification, not a full-unitary certificate. A changed
output layout is rejected by this current brickwall-specific resource check.

Each stage records CPU time, including discarded depth candidates. MPS
receipts record forward/adjoint/derivative/target bond maxima, observed retained
tensor storage, actual SVD tails, cap reductions and forward norm drift.
Worker-lifetime peak RSS includes imports and previous work in that process.
Use a fresh process per case/backend for memory comparisons.

`--profile-memory` enables allocation tracing around the full case, including
target factorization, retained forward/backward tapes and temporary allocations.
Tracing can slow execution; compare runtimes with it disabled for both backends.
Traced peaks exclude some native-library allocations, while RSS is a lifetime
high-water mark. Tensor snapshots alone are not complete peak process memory.

## Running and resuming

Example for the proposed twelve-qubit pilot, with unchanged training budgets:

```sh
uv run python -m experiments.topdown_mps \
  --output output/topdown-mps-q12-new \
  --qubits 12 --backend mps --hardware IBM \
  --targets gaussian_mu0p5_sigma0p1
```

Use `--backend statevector` and another output root for the matched dense run.
Additional target identifiers are the existing fixture identifiers. For a
short implementation smoke check, explicitly reduce `--max-depth`,
`--construction-steps`, `--construction-online`, `--noisy-steps`, `--noisy-online`
and assessment budgets. Such a run is not a full-budget benchmark result.

Run the same command to resume. A manifest authenticates the target, fixture,
numerical sources, backend, truncation and budgets. Every completed stage has a
checksum seal; incomplete stages restart from the preceding sealed checkpoint.
A completed resume authenticates all stage payloads and the selection lock.
Curves and parameter arrays are stored in stage-local `arrays.npz` files once;
selection and final summaries reference those files rather than copy histories.

Case output:

- `manifest.json`: configuration, seeds, source and target checksums.
- `depth_XX/construction/`: best noiseless parameters and L-BFGS diagnostics.
- `depth_XX/pruning/`: B scores, removed indices and retained circuit.
- `depth_XX/refinement/`: fixed noise bank, multiplicities, final parameters and curve.
- `depth_XX/validation/`: independent validation bank and statistics.
- `selection_locked.json`: selected depth and all stopping decisions.
- `test_XX/`: independent final banks, coverage and per-batch statistics.
- `final/record.json`: compact result, resources and references.

To compare exact MPS with dense execution and probe controlled bond caps on the
**same saved circuit, parameters, noise maps and weights**:

```sh
uv run python -m experiments.topdown_backend_probe \
  --case-root output/topdown-mps-q12-new/gaussian_mu0p5_sigma0p1 \
  --output output/topdown-mps-q12-new/gaussian-backend-probe.json \
  --bond-caps 4 8 16 32 64
```

This diagnostic reports fidelity, cross-signal, online-update and pruning-score
errors plus runtime/memory diagnostics. It never selects new parameters for the
scientific run. Its dense reference is limited to twelve qubits. Larger-system
studies need tensor-native target inputs and an explicitly budgeted assessment
strategy; this implementation does not establish an advantage for Haar targets.

## Controlled truncation checks

`experiments.topdown_mps_truncation` compares full noisy refinement from sealed
pruning checkpoints, holding initial parameters, logical IBM noise, the 100 raw
training histories, empirical multiplicity weights and both cross-trajectory
weight factors fixed. It runs the existing 100 online plus 100 batch updates at
rate 0.2 and retains the final noisy iterate. Noiseless construction and pruning
are reused, so this isolates truncation during fine-tuning.

The frozen grid compares exact statevectors, untruncated MPS, bond caps
2/4/8/16/32, and local discarded-weight thresholds 1e-8/1e-6/1e-4 without a hard
cap. Target compression is disabled. Local thresholds and accumulated SVD tails
are not bounds on global fidelity error; truncation also approximates adjoint
and derivative states, and their rank derivatives are not included.

Every resulting parameter history is assessed with exact statevectors. Final
circuits share a fresh independent bank of 32,768 unconditional noise histories
per depth, with paired confidence intervals based on raw multiplicities. Exact
duplicate histories are collapsed only for assessment. Training and evaluation
CPU are reported separately, with allocation tracing disabled.

```console
uv run python -m experiments.topdown_mps_truncation --root OUTPUT --prepare SEALED_CASE --depths 1 3
uv run python -m experiments.topdown_mps_truncation --root OUTPUT --run-all
```

The output includes authenticated input/source snapshots, all parameter and
fidelity curves in compact arrays, exact held-out observations, numerical
truncation diagnostics, and `summary.json`. Completed rows can be resumed only
with the same frozen numerical sources.

The completed [20-run truncation study](../output/topdown-mps-truncation-20260923/REPORT.md)
uses twelve-qubit critical-TFIM checkpoints at depths 1 and 3. Untruncated MPS
matches the full statevector refinement numerically. Threshold 1e-6 changes
held-out fidelity by at most 0.000213 percentage points; cap 32 by at most
0.000220 points. Caps 2/4/8/16 produce substantial losses, up to 14.86 points.
These are convergence checks for one target family and one training bank per
circuit, not universal safe tolerances. All assessment uses exact statevectors
on fresh independent noise histories, with paired confidence intervals.

A separate identity-gate probe confirms a remaining truncation limitation:
the current local kernel does not move the orthogonality center to each split.
At cap 4, a central identity split gives 76.0355% normalized target fidelity
without centering, versus 99.999842% with centering. Centered splits match direct
dense Schmidt truncation. Consequently, the hard-cap results must not be treated
as intrinsic MPS rank limits. Center-aware truncation should precede adoption of
aggressive caps; its full optimization accuracy has not yet been measured.
