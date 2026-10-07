# Tensor-network IQSP with matched logical noise

An optional [compressed comb backend](IQSP_COMB.md) is now available. The
exact backend and archived campaign described below retain their defaults.

**Implementation and correctness tests are ready; the performance campaign has
not been run.** This is a declared six-qubit adaptation of Ballarin et al.'s
[interpolative quantum state preparation (IQSP)](https://arxiv.org/html/2511.15674v1),
with the same logical IBM noise as the completed comparison. It supersedes the
old Gaussian-only KAK reconstruction for new experiments. Historical IQSP
results remain diagnostic and must not be included in a ranking.

## Implemented procedure

Start from the uniform state, prepared with Hadamards. At each declared depth,
optimize a nearest-neighbor brickwork of generic SU(4) gates against a sequence
of normalized target states. Warm-start each stage with the preceding stage's
final parameters. Optimize infidelity with Adam at learning rate 0.01. Then
decompose the learned circuit and refine its angles against noisy fidelity at
the final target. Remove two-qubit rotations with magnitude at most 1e-4 and
evaluate the resulting circuit. These steps retain the publication's
interpolation, final noisy optimization, and small-angle deletion strategy.

The concrete choices below are recorded, not claimed to reproduce the paper's
complete experimental configuration:

| Setting             | This adapter                                                                      |
| ------------------- | --------------------------------------------------------------------------------- |
| Targets             | All nine frozen six-qubit fixtures, including complex Haar states                 |
| Depth candidates    | 2 and 4 even/odd brickwork cycles; 10 and 20 SU(4) blocks                         |
| Initialization      | Zero SU(4) exponential coordinates after Hadamards                                |
| Interpolation       | 20 equally spaced nonzero coordinates, ending at 1                                |
| Construction budget | 200 Adam updates **per stage**, a benchmark budget                                |
| Noisy budget        | 10,000 Adam updates, matching the paper's noisy-study epoch count                 |
| Adam details        | Rate 0.01, beta1 0.9, beta2 0.999, epsilon 1e-8; moments reset at each stage      |
| Retention           | Final iterate; best iterate retained only as a diagnostic                         |
| Logical lowering    | Exact unsimplified ZYZ/Weyl: 12 single-qubit and 3 two-qubit rotations per SU(4)  |
| Small-angle cutoff  | Once after noisy refinement; absolute signed RXX/RYY/RZZ angle <= 1e-4; no repair |
| Validation          | 8,192 unconditional independent trajectories per candidate                        |
| Final metric        | Independent exact logical density-matrix fidelity, after selection is locked      |

The paper does not specify the complete Adam-state reset policy or precise
deletion cadence. Those are explicit implementation choices here. Short test
budgets establish correctness only. The construction budget is not a
convergence claim. No pruning B/D, Krotov, L-BFGS, or additional fidelity-driven
deletion is applied to this competitor.

### Target paths

The Gaussian uses the publication's exponential family, with the **fixture's**
covariance and quantics bit ordering. Its width is not replaced with the paper's
experimental width. For the other eight fixtures, define the uniform state
`u`, phase-align the target `t` so `<u|t>` is nonnegative, and use

\[
\alpha=\arccos\langle u|t\rangle,\qquad
|f(\lambda)\rangle=
\frac{\sin((1-\lambda)\alpha)|u\rangle+
\sin(\lambda\alpha)|t\rangle}{\sin\alpha}.
\]

The zero-angle limit is handled continuously. This geodesic is an explicit
**all-target extension**, not a published Ballarin benchmark or a claim that
the paper's gradient guarantee covers complex Haar states. The runner labels
these eight targets separately from the Gaussian path.

### Tensor representation and reused code

Pure training contracts local circuit tensors against an exact target MPS.
Noisy training introduces a local Kraus purification leg after each elementary
operation, contracts its bra and ket, and differentiates the resulting exact
fidelity using PyTorch. No global statevector or density-matrix propagation is
used in training. Dense fixture input is converted to MPS once per objective;
that constant SVD is outside the parameter gradient path.

The SU(4) exponential is extracted from the
[author-maintained TorchQuantum fork](https://github.com/mballarin97/torchquantum/tree/5bbd7353f63dfc868c3b0a0d6f29f5c86506f207):
`exp(+i sum(theta_j P_j))` over the 15 nonidentity Pauli products. Its MIT
license, pinned commit and changes are recorded in
[vendor provenance](vendor/iqsp_tensor/PROVENANCE.md). Existing tested Adam,
Qiskit synthesis, channel definitions, assessment and storage are reused.

The backend is an **exact, uncompressed circuit tensor network**, not the
authors' compressed comb implementation. Their device-specific noise does not
match the common logical channel, and parameter-dependent rank truncation can
lose entangling derivatives at the identity. Exact contractions avoid this
problem at six qubits. They can still require exponentially large intermediate
tensors; contraction cost and largest intermediate are recorded. These timings
must not be presented as the authors' software runtime or large-system scaling.

### Same logical channel in training and assessment

The existing `pauli_distribution("IBM", arity)` defines every window. A
one-qubit window has X/Y/Z rates 1e-4 each. A two-qubit window also includes
single-site terms on both qubits and correlated XX/ZZ rates 0.0015 each. For
total rate `r`, identity has probability `exp(-r)` and each listed Pauli has
probability `(1-exp(-r))*rate/r`. This is the historical synthetic IBM profile,
not a new hardware calibration or an independent product channel.

All present gates, including Hadamards and zero-angle rotations, carry their
arity-dependent channel. An opaque SU(4) block never counts as one noisy gate.
There is no idle or readout noise. The cutoff explicitly deletes windows;
post-cutoff TN fidelity is recomputed and recorded separately from the last
optimizer iterate. Validation and testing then use this exact resulting
schedule. Native compilation supplies resource counts only.

Training evaluates the full noisy expectation, so there are no sampled
training trajectories to overfit or reuse. Validation samples fresh paths,
records the number with jumps and rejects a nonempty circuit without observed
jumps. It does not force jumps or change their probabilities. Selection is
locked before final exact evaluation, which includes every modeled jump history
and has no Monte Carlo error. Confidence intervals belong to sampled
validation, not the deterministic exact final fidelity.

## Running

Use a separate environment; the project lock and prior benchmark environments
are unchanged. The prepared local environment is `.venv-iqsp-tn`. For a fresh
environment, install the repository and the pinned optional requirements:

```sh
uv venv --python 3.12 .venv-iqsp-tn
uv pip install --python .venv-iqsp-tn/bin/python -e . \
  -r experiments/state_preparation_competitors/requirements-iqsp-tn.txt
```

Prepare an authenticated plan without training:

```sh
.venv-iqsp-tn/bin/python -m experiments.state_preparation_competitors.run_iqsp_tn \
  --output output/iqsp-tn-logical-q6 --dry-run
```

Remove `--dry-run` to run all nine targets with the table's defaults. Smaller
budgets or subsets require a new output directory. No full campaign was
launched as part of implementation. The CLI does not rerun any other method.

Each output directory contains the configuration/source manifest, sealed
construction bundles, one compact binary parameter-history archive per target,
locked validation, exact final results and a summary. Resume authenticates
sources, settings, targets and circuits. All stages and both depths count toward
training CPU; selection and final assessment have separate timings. Final
fidelity and total search cost must both be used for comparisons.

## Verification

The new tests cover all nine paths, complex phases and qubit ordering; a Bell
state reachability regression from the identity; SU(4) gradients; noisy
finite-difference gradients; independent Qiskit density parity; correlated
noise and zero-angle windows; unitary-preserving lowering; cutoff boundaries;
trace and positivity; the full six-qubit depth-four schedule; Adam updates and
warm starts; compact histories; independent validation with jumps; and sealed
resume without retraining or reselection.

```sh
.venv-iqsp-tn/bin/python -m pytest -n 0 \
  tests/optimization/test_state_preparation_competitors_iqsp_tn.py \
  tests/optimization/test_state_preparation_competitors_iqsp_tn_backend.py \
  tests/optimization/test_state_preparation_competitors_run_iqsp_tn.py
```
