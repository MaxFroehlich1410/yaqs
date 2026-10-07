# Restricted paper-based noisy preparation baselines

`fontana.py` and `iqsp.py` are local, explicitly adapted implementations of
published procedures. They are **not downloaded author code**, and do not claim
to reproduce the papers' hardware models or numerical results. They reuse
SciPy BFGS, the repository's tested bias-corrected Adam update, Qiskit synthesis,
and the shared exact-density objective. No upstream algorithm code is copied.

## Fontana real-state VQSP

Primary source: Fontana et al.,
[_Evaluating the noise resilience of variational quantum algorithms_,
arXiv:2011.01125v3](https://arxiv.org/abs/2011.01125v3), Sec. IV and Fig. 6.
The figure was inspected in the PDF. Its four-qubit layer has RY rotations on
every qubit, parallel directed CX pairs `(0,1)`/`(2,3)`, then `(1,2)`. The study
uses real target states, noiseless optimization followed by noisy
reoptimization, and BFGS.

Our extension repeats RY on each qubit followed by even then odd neighboring CX
pairs. A depth `d` on `n` qubits has `n*d` parameters and `(n-1)*d` CX gates.
This is a declared six-qubit extension of the figure, not the paper's original
four-qubit experiment. Inputs must be real up to a single global phase;
relative complex phases are rejected rather than removed.

The actual eligible q6 fixtures are **Gaussian, all three TFIM targets, and both
random-MPS targets**. The stored random-MPS fixtures are real. The three complex
Haar targets are outside this ansatz's scope and must be reported as unsupported.

Each depth independently samples `Uniform[-pi,pi]` using NumPy SeedSequence
entropy `[seed, 201101125, depth]`. This initialization, six-qubit size, optimizer
budgets, and stopping tolerances are declared implementation choices. Original
SciPy `minimize(method="BFGS", jac=True)` receives the shared exact-density loss
and analytic gradient. Defaults are 200 ideal and 200 noisy iterations,
`gtol=1e-8`, and a hard joint loss/gradient-call cap of `1+20*iterations` per
phase. Returned parameters are the best accepted iterate including the initial
one; rejected line-search probes cannot become output. The ideal endpoint
initializes noisy training. Both phases retain complete iterate/cost receipts.

The paper's independent single-qubit channels after entangling sublayers are
**replaced by the common native IBM/Google synthetic Pauli channel**, including
noise after single-qubit operations. Thus this is a common-noise method
adaptation, not an exact reproduction of the paper's damping or product-layer
noise experiments.

## Gaussian-only IQSP

Primary source: Ballarin et al.,
[_Efficient quantum state preparation of multivariate functions using tensor
networks_, arXiv:2511.15674v1](https://arxiv.org/abs/2511.15674v1), Eqs. (3)–(6),
the noise-aware section, and End Matter Fig. 6. The circuit figure was inspected
in the PDF. IQSP follows a smooth function family from an easy initial state,
warm-starting successive ideal optimizations; the published procedure uses
Adam and adds hardware-native noisy optimization at the final target. Its
one-dimensional circuit specialization has initial Hadamards and SU(4)
brickwork, without inter-variable comb links.

Here eligibility is restricted to the exact **six-qubit
`gaussian_mu0p5_sigma0p1` fixture**, including copies differing only by a global
phase. The adapter checks amplitudes, not just a caller-provided family label.
It does not invent a homotopy for Haar or other target families. The interpolation
is

\[
\psi\_\lambda(x)\propto
\exp[-\lambda(x-0.5)^2/(4\times0.1^2)],\qquad 0\le\lambda\le1,
\]

on the fixture's endpoint-excluded quantics grid, with qubit 0 the most
significant coordinate bit. The uniform `lambda=0` state is exact. Because our
fixture contains square-root probability amplitudes, its amplitude covariance
in the paper's Eq. (6) is **0.02**, not the paper's example covariance 0.05.

At six qubits all 64 function values are represented exactly. We do not execute
TCI, introduce tensor truncation, or claim the paper's large-system scaling.
Each nearest-neighbor SU(4) uses a universal 15-coordinate KAK parameterization:
local ZYZ rotations, commuting XX/YY/ZZ rotations, then local ZYZ rotations.
The paper specifies generic SU(4); this coordinate choice is ours. All coordinates
start at zero after the initial Hadamards, so initialization is deterministic.
Each requested depth restarts independently; different nominal seeds are not
independent IQSP replicas.

**Known issue; performance experiments on hold.** The
[Gaussian pilot](../../output/gaussian-all-methods-pilot-20260920/REPORT.md)
found every entangling coordinate exactly zero throughout all saved construction
iterates. With these zero-initialized KAK coordinates and a real target, only
local RY angles acquire a nonzero first-order update; the implementation stays
in product circuits. This is a local reconstruction failure, not evidence about
published IQSP performance. Correct and test the initialization/coordinates
before using IQSP in another performance comparison. Its existing code and
diagnostic results are retained; increasing the unchanged budget is insufficient.

Defaults use 20 nonzero interpolation stages (`delta_lambda=0.05`), **200 Adam
updates per stage**, and 200 final noisy updates, with learning rate 0.01,
betas `(0.9,0.999)`, and epsilon `1e-8`. This is **4,000 ideal updates per depth**,
not 200 total, and differs from the paper's longer reported epoch budgets.
Those are the Python function defaults. The `run_noisy` campaign CLI explicitly
requests 10000 noisy IQSP updates unless `--noisy-steps` overrides it; every
effective budget appears in the candidate receipt. The smoke command overrides
both phase budgets and the interpolation count.
The implementation calls the existing `AdamConfig`/`adam_step` in
`experiments/state_preparation_exact_adam.py`. Moments reset per stage; final
parameters warm-start the next stage. Final and best observed fidelities are
both recorded, but the final iterate is returned. Moment resets and finite
budgets are explicit reproducibility choices where the paper does not fix our
local implementation details.

After the final ideal stage, each learned KAK block is replaced by its exact
4-by-4 unitary. The common compiler then synthesizes the resolved SU(4) gates;
a generic isolated block needs at most three CX gates. This avoids retaining
the unnecessary six-CX expansion of unbound XX/YY/ZZ rotations. Full ideal
unitary equivalence is checked before and after compilation. Every compiled RZ
angle then becomes an independent noisy-training parameter, with SX/X/CX fixed.
The native and original parameter counts can differ and their histories remain
separate. The final noisy phase differentiates exactly this native schedule.

The paper's angle-dependent H2 ZZ error model and tiny-ZZ-angle removal are
**not transferred to CX gates**. This baseline instead trains with the common
native synthetic IBM/Google model. Existing Ballarin/Quantinuum channel helpers
remain unchanged but are not invoked because they describe a different native
gate set and channel. This is not an H2-calibrated baseline or a reproduction
of the published hardware experiment. IQSP's final circuits depend on the
chosen noise profile even though optimizer seeds are unused.

## Evaluation boundary, cost, and tests

Both adapters return `preserve_native_schedule=True`. The shared evaluator must
retain the trained native schedule so that further gate cancellation does not
silently change the optimized noise windows. Training integrates all modeled
noise outcomes exactly; it does not use M=100 sampled trajectories. Those exact
training objectives are distinct from the independent validation used for
candidate selection and the final post-selection reporting stage. No claim of
equal training budgets or equal statistical access across methods is made.

All requested depths, interpolation stages, BFGS probes, compilation, and noisy
updates are charged; metadata records initial/final parameters, fidelity curves,
oracle calls, seeds, timings, and the model adaptation. IQSP must be deduplicated
across seeds but not across hardware profiles. Unsupported target/method pairs
must remain explicit exclusions, not fabricated zero fidelities.

Tests cover target eligibility and global phases; the inspected ansatz supports;
independent Gaussian values and fixture matching; direct SciPy/analytical
optimizer oracles; actual generic SU(4)-to-three-CX compilation and free-native-RZ
equivalence; stage warm starts; invalid configurations; nonfinite failures; and
actual tiny six-qubit runs checked against independent Qiskit density evolution.
These are implementation checks, not a full performance comparison.

```sh
.venv/bin/python -m pytest -n 0 \
  tests/optimization/test_state_preparation_competitors_fontana.py \
  tests/optimization/test_state_preparation_competitors_iqsp.py
```
