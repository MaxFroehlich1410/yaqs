# NACL: paper-based reconstruction and upstream optimizer

This adapter reconstructs the state-preparation algorithm in Cincio, Rudinger,
Sarovar and Coles, **Machine Learning of Noise-Resilient Quantum Circuits**,
PRX Quantum **2**, 010324 (2021),
[DOI 10.1103/PRXQuantum.2.010324](https://doi.org/10.1103/PRXQuantum.2.010324),
[arXiv:2007.01210](https://arxiv.org/html/2007.01210).

It is **not the authors' implementation**. No public author implementation was
located during the repository audit. The local structural-search orchestration
is a paper-based reproduction; the continuous optimizer is reused from the
official NOMAD implementation rather than rewritten or replaced with BFGS.
The unrelated third-party `nishant34/Variational_Algorithms/Nacl` code is not used:
it combines other architecture-search policies and optimizers.

## Published algorithm retained

The state-preparation objective is the exact noisy infidelity

\[
C(\boldsymbol\theta,\boldsymbol k)
=1-\langle\psi*{\mathrm{target}}|
\mathcal E*{\boldsymbol\theta,\boldsymbol k}
(|0\rangle\langle0|)|\psi\_{\mathrm{target}}\rangle.
\]

The inner loop optimizes continuous angles using gradient-free **OrthoMADS**.
The outer loop proposes changes in structure, including gate shuffling, gate
removal and insertion of one- or two-qubit resolutions of the identity.
After reoptimizing the angles, a lower cost is accepted, and an increase
\(\Delta C>0\) is accepted with probability \(\exp(-\Delta C/T)\).
The paper also describes multiple restarts and simple cancellation/commutation
compilation. These operations belong to NACL itself; they are not our pruning B
or an additional noise-fine-tuning method.

Arbitrary complex pure target amplitudes are accepted for one through six qubits.
The prepared noisy state uses the shared exact density-matrix simulator. No
Monte Carlo training samples or shots are used for NACL.

## Explicit reconstruction and benchmark choices

The manuscript does not provide a complete executable specification of proposal
probabilities, identity blocks, annealing temperatures, stopping rules or restart
counts. The following choices are therefore disclosed and configurable, not
claimed to be recovered settings from the authors' experiments:

| Choice               | Implementation/default                                                                                                                               |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Initial circuit      | Empty, independently seeded chains                                                                                                                   |
| Native gates         | RZ, SX, directed nearest-neighbor CX on a line                                                                                                       |
| Structural moves     | Uniform over the currently available move types: one-qubit identity insertion, two-qubit identity insertion, one-gate removal, adjacent gate shuffle |
| Proposal positions   | Uniform over valid insertion/removal/shuffle positions                                                                                               |
| One-qubit identity   | RZ(a), SX, RZ(pi), SX, RZ(pi-a), with a uniform in [-pi, pi); equals -i I                                                                            |
| Two-qubit identity   | CX, RZ(0) on the target, CX; either direction on a uniformly sampled adjacent pair                                                                   |
| Continuous variables | Every RZ angle independent; existing optimized angles warm-start the next proposal                                                                   |
| Annealing            | Initial T=0.01, multiplied by 0.95 each structural step; per chain                                                                                   |
| Inner budget         | At most 100 exact noisy objective evaluations per OrthoMADS call                                                                                     |
| Mesh frames          | Initial 0.5 radians, minimum 1e-6 radians                                                                                                            |
| Checkpoints          | 0, 10, 50 outer steps per chain by default                                                                                                           |
| Restarts             | One by default; exposed as an explicit control                                                                                                       |
| Resource cap         | 300 native gates, exposed explicitly                                                                                                                 |
| Simple compilation   | Every five outer steps; exact commuting RZ combination, identical CX-pair cancellation, groups of four SXs and exact zero-modulo-2pi RZ removal      |

The identity blocks are initially identity but can leave identity during angle
optimization. Combining arbitrary one-qubit rotations with the variable CX-RZ-CX
entangling blocks does not impose a real-amplitude restriction.

Simplification follows continuous optimization so newly inserted identities are
not canceled before they can explore a new state. Its changed noisy channel is
evaluated and the remaining angles reoptimized; both pre- and post-simplification
circuits are eligible for the recorded best noisy fidelity. There is no saliency
threshold, approximate angle deletion, extra optimizer or externally added
pruning stage. The fixed structural budget is not presented as convergence.

The shared IBM/Google synthetic Pauli channels are applied after every native
RZ/SX/CX operation, including RZ; there is no idle noise. These are supplied
benchmark device models, **not** the Ourense GST calibration from the paper.
The returned circuit keeps precisely the trained native schedule. Its metadata
sets `preserve_native_schedule=True` so deployment evaluation cannot silently
cancel gates or change the trained noise placement.

## Official NOMAD reuse

- Upstream: [bbopt/nomad](https://github.com/bbopt/nomad).
- Package: `PyNomadBBO==4.6.0`, imported as `PyNomad`.
- Release tag: `v.4.6.0`, commit
  `945fa3e2d801361e3dba69746c30cea656377a2a`.
- License: LGPL-3.0. Upstream is an installed dependency, not copied into the
  local MIT-licensed implementation.
- Verified macOS ARM64 CPython 3.11 wheel:
  `pynomadbbo-4.6.0-cp311-cp311-macosx_11_0_arm64.whl`, SHA256
  `ade82141df0f5d0dbdf4e94926918b956b26e1f2b796125099da875537bab662`.

This modern dependency implements the cited OrthoMADS algorithm; it is not the
authors' historical binary. Both primary and secondary poll directions are set
to `ORTHO 2N`. Quadratic-model, SGTELIB, Nelder-Mead, speculative and Latin-hypercube
searches are disabled. No bounds are added to periodic rotation angles. Every
actual optimizer parameter, seed, stop reason and scalar loss evaluation is
recorded in the candidate receipt.

Integration details established by actual upstream tests:

- `setSeed` and `resetRandomNumberGenerator` must be called for every solve.
  Repeating the `SEED` parameter alone does not reset the native global state.
- The driver draws inner seeds uniformly from **0 through 65535** using each
  restart's recorded NumPy generator. Upstream's default `RNG::reset()` advances
  the generator `seed` times, so sampling 31-bit seeds wasted seconds per inner
  solve in RNG initialization alone. The bounded seed distribution avoids this
  artifact without changing the optimizer or its search budget. Direct callers
  of `ortho_mads` can supply signed 32-bit seeds, but pay this O(seed) setup cost.
  See [upstream RNG.cpp](https://github.com/bbopt/nomad/blob/v.4.6.0/src/Math/RNG.cpp).
- The 4.6.0
  native parser can reject/abort for larger seeds despite broader documentation.
  The adapter therefore rejects direct inner seeds above the signed 32-bit limit.
  Root seeds remain unsigned 32-bit and generate safe 16-bit inner seeds.
- At minimum-frame stopping, 4.6.0 can return a negative `run_flag` even with a
  valid feasible incumbent. The adapter requires both recorded successful
  objective evaluations and an upstream feasible incumbent, accepts this
  specific frame-stop condition, and preserves the original flag and stop
  reason. Callback errors and interrupted/failed solves remain errors.
- NOMAD has process-global native state; the experiment runner calls its inner
  solves sequentially. Independent parallel runs should use separate processes.

Install without changing the shared experiment environment:

```sh
uv venv --python .venv/bin/python .venv-nacl
uv pip install --python .venv-nacl/bin/python -r experiments/state_preparation_competitors/requirements-nacl.txt
```

For the current Python 3.11 workspace, expose only that dependency when running
the normal project Python:

```sh
PYTHONPATH=.venv-nacl/lib/python3.11/site-packages .venv/bin/python -m pytest -n 0 tests/optimization/test_state_preparation_competitors_nacl.py
```

Alternatively install the pinned optional requirement in a dedicated environment
containing the usual benchmark dependencies. Import/version failures stop NACL;
there is no silent fallback to a different optimizer.

## Accounting and validation

Search checkpoints reuse a single prefix of every seeded chain. Each checkpoint
stores **only the histories added since the previous checkpoint**, identified by
`previous_checkpoint`; earlier histories are not duplicated. Receipts retain
scalar inner fidelity curves, move/annealing choices, seeds, best parameters,
resolved circuit, actual objective counts and interval/cumulative CPU and wall
time. All rejected proposals, restarts, optimizer work and simplification solves
are charged. Timing includes initialization and scoring bookkeeping. Per-evaluation
parameter vectors are not stored; the returned best parameters are preserved.

Tests exercise full-unitary identity/commutation/simplification oracles,
independent noisy-density scoring of complex six-qubit runs under both profiles,
actual upstream OrthoMADS optimization and seed replay, complex one-qubit
reachability, checkpoint equivalence, budget accounting and invalid inputs.
The six-qubit runs are deliberately tiny integration checks, not convergence or
competitive-fidelity evidence.
