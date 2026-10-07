# RobustState: published algorithm with simulated feedback

Reference: Wang et al., [RobustState](https://arxiv.org/html/2311.16035v1),
Algorithm 1, Section III.B, Figure 10 and Section IV.A. This module is a
paper-based reimplementation, not recovered author code or a reproduction of
the original hardware experiment.

## Preserved method

For target projector \(P=|\psi\rangle\langle\psi|\), the training objective is
the **Hilbert-Schmidt norm**, not infidelity:

\[
L(\theta)=\|\rho*{\mathrm{feedback}}(\theta)-P\|\_F,\qquad
g_j=\operatorname{Tr}\!\left[
\frac{\rho*{\mathrm{feedback}}-P}{L}
\frac{\partial\rho\_{\mathrm{ideal}}}{\partial\theta_j}
\right].
\]

At zero loss, the implementation uses the zero norm subgradient. The noisy
output is held fixed while constructing this surrogate gradient; the simulator
Jacobian is noiseless. Final evaluation separately reports
\(F=\langle\psi|\rho\_{\mathrm{noisy}}|\psi\rangle\).

- The gate-level Figure 10 block applies CNOT, then RZ-RX-RZ on both endpoints.
  Blocks traverse the line's neighboring pairs in order, then repeat. Disjoint
  blocks can execute in parallel. A `depth` means one complete traversal, not
  compiled two-qubit depth. No pulse calibration or RZX pulse variant is claimed.
- Defaults are 500 noiseless and 50 noisy updates with Adam learning rate 0.005.
- The final iterate is returned. Best parameters/fidelities are diagnostic only;
  they do not introduce our baseline's rollback policy into RobustState.
- There is no added pruning, architecture search or substituted optimizer.

## Explicit benchmark adaptations and unspecified conventions

`feedback="exact_density"` uses the exact simulated noisy density matrix.
This is a shot-free feedback adaptation, not simulated finite-shot tomography.
`feedback="pauli_shadows"` instead measures all \(3^n\) local Pauli bases with
1024 shots per basis by default. Optional `num_bases` uniformly samples a subset
without replacement at each update. Its linear estimate is

\[
\widehat\rho=\frac1{BS}\sum*{b,s}\bigotimes_q
\left(3U*{b,q}|s*q\rangle\langle s_q|U*{b,q}^{\dagger}-I\right).
\]

The raw shadow is Hermitian and trace one but can have negative eigenvalues.
No positivity projection is introduced. Input densities for Born sampling must
be physical. Measurement rotations and readout are ideal: the current benchmark
specifies preparation noise only, so there is no readout error to mitigate.
This does not reproduce the paper's device calibration, readout mitigation,
hardware execution cost, or pulse-control experiments.

Six qubits, the target fixtures, depth grid, and synthetic IBM/Google noise
profiles are benchmark inputs. Training and final scoring use the same native
RZ/SX/X/CX schedule. Symbolic compilation occurs before optimization and the
returned circuit must be evaluated without subsequent gate cancellation or
parameter-dependent recompilation (`preserve_native_schedule=True`). The
benchmark's noise acts after every native operation, including RZ. It is not a
historical IBM calibration or an assertion about actual virtual-Z error.

The paper does not specify every implementation convention. This implementation
records uniform initial angles in [-pi, pi], standard Adam beta values
(0.9, 0.999) and epsilon 1e-8, continuous Adam moments across the phase switch,
and independently derived initialization/measurement seeds. These are disclosed
choices, not attributed numerical details. Existing
`experiments.state_preparation_exact_adam.py` supplies Adam; the shared
`noisy_core.DenseObjective` supplies the native schedule and ideal adjoint.

## Upstream search and reuse

The author framework [TorchQuantum](https://github.com/mit-han-lab/torchquantum)
was inspected at commit `8dc3255c51477dd4c28892049571df032c77e2ff` (MIT license).
Its [state-preparation example](https://github.com/mit-han-lab/torchquantum/blob/8dc3255c51477dd4c28892049571df032c77e2ff/examples/train_state_prep/train_state_prep.py)
is a two-qubit noiseless overlap-loss example, not the published noisy-feedback
algorithm. No verified paper implementation was found after searching the
paper, author pages, repository names, 282 main-branch source/documentation
files, and relevant density/noise development branch trees. This records the
search result, not proof that the authors have never released code elsewhere.

NumPy and Qiskit already present in this repository supply the numerical and
circuit primitives. TorchQuantum is not installed: its current dependency
requirements differ from the benchmark's pinned Qiskit environment, and its
generic example would not supply the missing RobustState algorithm anyway.

## Tests and receipts

The targeted suite checks Figure 10 gate order; complex conjugation; the
surrogate gradient against finite differences with frozen noisy feedback;
exact integration of the shadow estimator on complex entangled mixed states;
shot-seed replay; raw estimator nonpositivity; Adam state across the phase
switch; final-iterate retention; and actual six-qubit circuit training/scoring.
Candidate metadata stores curves, seeds, parameters, timings, feedback mode,
measurement counts, and the simulator adaptation. No timing claim for original
hardware runs follows from these classical simulations.
