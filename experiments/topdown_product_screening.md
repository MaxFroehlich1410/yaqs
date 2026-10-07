# Product-state screening before state preparation

For each target, sample **1,000 independent random product states**, include
the exact zero state, and choose the largest noiseless target overlap:

$$
|\phi_r\rangle=\bigotimes_{i=0}^{n-1}
\left(\sqrt{1-p_{ri}}|0\rangle+
e^{i\varphi_{ri}}\sqrt{p_{ri}}|1\rangle\right),
\quad p_{ri}\sim U[0,1],\quad \varphi_{ri}\sim U[0,2\pi),
$$

$$
|\phi_0\rangle=|0^n\rangle,\qquad
r_* = \arg\max_{0\leq r\leq1000}|\langle t|\phi_r\rangle|^2.
$$

Exact ties retain the earlier candidate, with zero first. This guarantees the
selected **product-layer** fidelity is at least the zero-state overlap. It does
not guarantee higher final fidelity, better gradients or faster convergence.
The existing initialized entanglers can change the full depth-one fidelity;
record that score separately. These samples are product states, not generic
entangled Haar states requiring a separate preparation circuit.

Encode the winner with the **existing initial U3 layer**, leaving all other
depth-one parameter values unchanged. All later layers inherit their own trained
parent and use the unchanged extension stream. Pruning B, optimizer budgets,
weighted cross-trajectory updates, noise and validation selection remain fixed.
The physical input to execution remains zero; its existing initial rotations
prepare the selected product state.

## Reproducibility and cost

Use `SeedSequence([optimizer_seed, 730002])` for the candidate pool, independently
of initial entangler draws and the original unscreened random-product stream.
The three optimizer seeds 300/400/500 therefore have independent pools. There is
no selection of optimizer seeds or noise samples using held-out results.

Score in batches of 64 vectors; retain the 1,001 scores and product angles once
in the sealed `initialization/arrays.npz`, together with the full initial parameter
vector. This uses `O(batch_size * 2**n + count * n)` working storage, rather than
keeping all dense candidate vectors. Initialization time includes sampling,
scoring, selection and actual rotation-layer/depth-one replay, and contributes to
accounted preparation CPU. Serialized arrays, manifests and source hashes prevent
resampling a completed initialization or mixing different inputs on resume.

## Six-qubit full-pipeline comparison

The default plan contains **54 new jobs**: nine six-qubit targets, three seeds,
and near-zero versus screened-product initialization. An unscreened
`random_product` single-job control is also available. Existing competitor runs
can be reused only when noise, assessment and other settings match.

Use statevectors, current default 100 online Krotov plus 100 safeguarded L-BFGS
construction updates per depth, pruning B at 0.01, and 100 online plus 100 batch
noisy cross-trajectory updates on 100 frozen weighted histories. The default
profile is IBM logical gate noise. The existing independent 32,768-history
validation and depth-patience rule select a circuit. Four independent final
trajectory batches of 8,192 histories preserve jump-coverage reporting. Exact
density-matrix fidelities are then evaluated for the selected circuit before and
after noisy fine-tuning, after the validation decision is locked. The result also
records ideal fidelity before and after pruning, resources and stage costs.

This comparison changes initialization only. It does not import the longer
2,000-update budget or 99% early stop from the twelve-qubit capacity diagnostic.
The full noisy runner is limited to targets of at most six qubits for its exact
density assessment; twelve-qubit standalone screening is supported.

```sh
# Plan only: launches nothing.
python -m experiments.topdown_product_screening \
  --output output/q6-product-screening

# Explicitly sample and seal one initialization without training.
python -m experiments.topdown_product_screening --action screen \
  --target-id random_mps_bond2 --seed 300 \
  --output output/q6-product-screening/screen-only

# Explicitly run one complete matched noisy pipeline.
python -m experiments.topdown_product_screening --action run \
  --target-id random_mps_bond2 --seed 300 --arm screened_product \
  --output output/q6-product-screening/jobs/random_mps_bond2__screened_product__seed300
```

Use fresh output directories and frozen source snapshots for distributed runs.
The existing twelve-qubit workers and their frozen snapshots are unaffected.
Incomplete noisy stages retain the existing runner's recovery policy; sealed
initialization is reused. No new optimizer or noise-bank resampling is introduced.

The prior 1,000-sample twelve-qubit diagnostics were not used for training.
Their distributions are strongly right-skewed, with useful high-overlap extremes
for Gaussian and random-MPS fixtures. Even the best sampled product state fell
below the exact zero-state overlap for the three TFIM fixtures. See the
[descriptive overlap audit](../output/q12-initialization-matrix-20261005/OVERLAP_DISTRIBUTIONS.md).
These observations motivate screening with the zero fallback; a full noisy
performance improvement remains to be measured.
