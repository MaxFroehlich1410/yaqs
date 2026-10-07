# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""RobustState's published surrogate gradient with simulated density feedback.

The objective remains the Hilbert-Schmidt norm, not infidelity. The noisy
output supplies its costate, while only the noiseless circuit is differentiated.
The default exact feedback is explicitly a simulator adaptation; optional Pauli
shadow feedback samples the publication's measurement protocol without readout
errors. See ROBUSTSTATE_PROVENANCE.md for reproduction limits.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from experiments.state_preparation_competitors.checkpointed_dense import make_objective
from experiments.state_preparation_competitors.common import Candidate, validate_target
from experiments.state_preparation_competitors.noisy_core import DenseObjective, nonnegative_integer, real_parameters
from experiments.state_preparation_exact_adam import AdamConfig, adam_step

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

PAPER = "https://arxiv.org/html/2311.16035v1"
LEARNING_RATE = 0.005
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
_PAULI_BASES = np.array(
    [np.array([[1, 1], [1, -1]]) / np.sqrt(2), np.array([[1, 1], [1j, -1j]]) / np.sqrt(2), np.eye(2)],
    dtype=np.complex128,
)


def topology_aware_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    """Build Figure 10's CNOT-first blocks on a linear nearest-neighbor device.

    A depth is one forward traversal of all neighboring pairs. Each block is
    CNOT followed by independent RZ, RX, RZ rotations on both endpoints. This
    sequence can be packed into overlapping traversals exactly as in the figure.

    Args:
        num_qubits: Register size, from two to twelve qubits.
        depth: Nonnegative number of complete pair traversals.

    Returns:
        A symbolic gate-level circuit with six parameters per CNOT block.

    Raises:
        ValueError: If the register or depth is invalid.
    """
    if type(num_qubits) is not int or not 2 <= num_qubits <= 12:
        msg = "RobustState's two-qubit-block ansatz requires two to twelve qubits."
        raise ValueError(msg)
    nonnegative_integer(depth, "depth")
    circuit = QuantumCircuit(num_qubits)
    parameters = iter(ParameterVector("theta", 6 * depth * (num_qubits - 1)))
    for _ in range(depth):
        for left in range(num_qubits - 1):
            circuit.cx(left, left + 1)
            for qubit in (left, left + 1):
                circuit.rz(next(parameters), qubit)
                circuit.rx(next(parameters), qubit)
                circuit.rz(next(parameters), qubit)
    return circuit


def _hermitian_matrix(value: ArrayLike) -> NDArray[np.complex128]:
    """Copy a finite Hermitian q1-q12 matrix, permitting nonpositive shadows.

    Raises:
        ValueError: If the shape, finiteness or Hermiticity is invalid.
    """
    matrix = np.asarray(value, dtype=np.complex128)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not 2 <= matrix.shape[0] <= 4096
        or matrix.shape[0] & (matrix.shape[0] - 1)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=0)
    ):
        msg = "Expected a finite Hermitian matrix for one to twelve qubits."
        raise ValueError(msg)
    return matrix.copy()


def hilbert_schmidt_loss_and_costate(
    prepared: ArrayLike, target_density: ArrayLike
) -> tuple[float, NDArray[np.complex128]]:
    """Return the published Frobenius norm and its Hermitian-matrix derivative.

    Args:
        prepared: Exact density output or an unprojected shadow estimate.
        target_density: Target projector, using conjugated complex amplitudes.

    Returns:
        ``(norm(prepared-target), (prepared-target)/norm)``. At exactly zero
        loss the returned zero costate is a valid norm subgradient.

    Raises:
        ValueError: If matrices are invalid or have different dimensions.
    """
    actual, target = _hermitian_matrix(prepared), _hermitian_matrix(target_density)
    if actual.shape != target.shape:
        msg = "Prepared and target density matrices must have the same shape."
        raise ValueError(msg)
    difference = actual - target
    loss = float(np.linalg.norm(difference))
    return loss, difference / loss if loss else np.zeros_like(difference)


def pauli_basis_probabilities(density: NDArray[np.complex128], basis: tuple[int, ...]) -> NDArray[np.float64]:
    """Compute outcome probabilities in a Pauli product basis, high qubit first."""
    n = len(basis)
    tensor = density.reshape((2,) * (2 * n))
    for measured, label in enumerate(basis):
        unitary = _PAULI_BASES[label]
        projectors = np.einsum("is,js->sij", unitary.conj(), unitary)
        # Contract a row/bra pair immediately: each measured qubit halves the
        # remaining tensor instead of rotating a full density twice per site.
        tensor = np.moveaxis(np.tensordot(projectors, tensor, axes=((1, 2), (measured, n))), 0, measured)
    probabilities = tensor.reshape(-1).real.copy()
    probabilities = np.maximum(probabilities, 0)
    return probabilities / probabilities.sum()


def inverse_pauli_shadow(frequencies: NDArray[np.float64], basis: tuple[int, ...]) -> NDArray[np.complex128]:
    """Apply the local shadow inverse to a distribution over one product basis."""
    n = len(basis)
    operands: list[Any] = [frequencies.reshape((2,) * n), list(range(n))]
    for qubit, label in enumerate(basis):
        unitary = _PAULI_BASES[label]
        inverse = np.stack(
            [3 * np.outer(unitary[:, bit], unitary[:, bit].conj()) - np.eye(2) for bit in (0, 1)], axis=2
        )
        operands.extend((inverse, [n + qubit, 2 * n + qubit, qubit]))
    operands.append(list(range(n, 3 * n)))
    return np.asarray(np.einsum(*operands, optimize=True), dtype=np.complex128).reshape((2**n, 2**n))


def classical_shadow_feedback(
    density: ArrayLike,
    *,
    rng: np.random.Generator,
    shots_per_basis: int = 1024,
    num_bases: int | None = None,
    trusted_cptp_output: bool = False,
) -> NDArray[np.complex128]:
    """Sample ideal Pauli measurements of a noisy density matrix.

    Args:
        density: Physical density matrix after the preparation circuit.
        rng: Explicit measurement RNG, independent of parameter initialization.
        shots_per_basis: Positive number of shots in each measured basis.
        num_bases: None measures all ``3**n`` bases as in Section IV.A; an
            integer selects a uniform subset without replacement. Bases are
            resampled independently on each call.

        trusted_cptp_output: Skip the cubic PSD eigensolve only for output already
            known to come from CPTP circuit propagation. Shape, trace and
            Hermiticity checks still apply. Never use for untrusted matrices.

    Returns:
        The unbiased raw Hermitian trace-one shadow average. It need not be
        positive semidefinite. No positivity projection, clipping of estimator
        eigenvalues, or readout mitigation is added to the published estimator.

    Raises:
        ValueError: If the input is not a density matrix or counts are invalid.
        TypeError: If rng is not an explicit NumPy Generator.
    """
    if type(trusted_cptp_output) is not bool:
        msg = "trusted_cptp_output must be an explicit boolean."
        raise TypeError(msg)
    rho = _hermitian_matrix(density)
    if abs(np.trace(rho) - 1) > 1e-10 or (not trusted_cptp_output and np.linalg.eigvalsh(rho).min() < -1e-10):
        msg = "Shadow sampling requires a positive trace-one density matrix."
        raise ValueError(msg)
    nonnegative_integer(shots_per_basis, "shots_per_basis")
    if shots_per_basis == 0:
        msg = "shots_per_basis must be positive."
        raise ValueError(msg)
    if not isinstance(rng, np.random.Generator):
        msg = "rng must be an explicit NumPy Generator."
        raise TypeError(msg)
    n = rho.shape[0].bit_length() - 1
    total_bases = 3**n
    count = total_bases if num_bases is None else num_bases
    if type(count) is not int or not 1 <= count <= total_bases:
        msg = "num_bases must lie between one and 3**num_qubits."
        raise ValueError(msg)
    selected = range(total_bases) if count == total_bases else rng.choice(total_bases, size=count, replace=False)
    estimate = np.zeros_like(rho)
    for index in selected:
        basis = tuple(int(index) // 3**power % 3 for power in reversed(range(n)))
        counts = rng.multinomial(shots_per_basis, pauli_basis_probabilities(rho, basis))
        estimate += inverse_pauli_shadow(counts / shots_per_basis, basis)
    return estimate / count


def robuststate_loss_gradient(
    engine: DenseObjective,
    theta: ArrayLike,
    *,
    noisy: bool = True,
    feedback_density: ArrayLike | None = None,
) -> tuple[float, NDArray[np.float64]]:
    """Contract a noisy loss costate with the noiseless circuit Jacobian.

    Args:
        engine: Exact circuit model supplying the target and ideal Jacobian.
        theta: Parameters in the engine's original template order.
        noisy: Whether unsupplied feedback is noisy or noiseless.
        feedback_density: Optional sampled tomography result, held fixed when
            differentiating. Only the simulator Jacobian is differentiated.

    Returns:
        Hilbert-Schmidt loss and the publication's surrogate gradient.
    """
    rho = engine.density(theta, noisy=noisy) if feedback_density is None else feedback_density
    projector = np.outer(engine.target, engine.target.conj())
    loss, costate = hilbert_schmidt_loss_and_costate(rho, projector)
    return loss, engine.density_vjp(theta, costate)


def train_robuststate(
    engine: DenseObjective,
    initial: ArrayLike,
    *,
    seed: int,
    noiseless_steps: int = 500,
    noisy_steps: int = 50,
    learning_rate: float = LEARNING_RATE,
    feedback: str = "exact_density",
    shots_per_basis: int = 1024,
    num_bases: int | None = None,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Run the published ideal-then-noisy Adam schedule and retain its final iterate.

    Args:
        engine: Fixed compiled circuit, target, and benchmark noise model.
        initial: Finite initial parameters in the original template order.
        seed: Independent local measurement RNG seed.
        noiseless_steps: Ideal Adam updates; publication default 500.
        noisy_steps: Noisy surrogate Adam updates; publication default 50.
        learning_rate: Constant learning rate; publication default 0.005.
        feedback: ``exact_density`` or ``pauli_shadows`` simulator feedback.
        shots_per_basis: Samples per basis for optional shadow feedback.
        num_bases: Optional uniformly sampled subset of the Pauli bases.

    Returns:
        Final parameters and finite JSON diagnostics. Best parameters are only
        reported, never substituted into the published final-iterate policy.

    Raises:
        ValueError: If budgets, settings, or initial parameters are invalid.
    """
    for value, name in ((seed, "seed"), (noiseless_steps, "noiseless_steps"), (noisy_steps, "noisy_steps")):
        nonnegative_integer(value, name)
    if feedback not in {"exact_density", "pauli_shadows"}:
        msg = "feedback must be exact_density or pauli_shadows."
        raise ValueError(msg)
    if engine.num_qubits > 6 and feedback == "pauli_shadows" and num_bases is None:
        msg = "Above six qubits, declare an explicit num_bases measurement budget."
        raise ValueError(msg)
    nonnegative_integer(shots_per_basis, "shots_per_basis")
    if shots_per_basis == 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
        msg = "Require positive shots_per_basis and a finite positive learning_rate."
        raise ValueError(msg)
    if num_bases is not None and (type(num_bases) is not int or not 1 <= num_bases <= 3**engine.num_qubits):
        msg = "num_bases must lie between one and 3**num_qubits."
        raise ValueError(msg)
    cpu, wall = time.process_time(), time.perf_counter()
    theta = real_parameters(initial, len(engine.parameters))
    first, second = np.zeros_like(theta), np.zeros_like(theta)
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    projector = np.outer(engine.target, engine.target.conj())
    total = noiseless_steps + noisy_steps
    config = AdamConfig(updates=max(1, total), learning_rate=learning_rate)
    best_loss = None
    best_theta = theta.copy()
    best_step = 0
    measurement_calls = 0
    final_phase = "noisy" if noisy_steps else "noiseless"
    for iteration in range(total + 1):
        phase = "noiseless" if iteration < noiseless_steps or not noisy_steps else "noisy"
        density = engine.density(theta, noisy=phase == "noisy")
        # Final diagnostics do not consume a new tomography sample or select an
        # iterate using hidden access to exact noisy fidelity.
        measured = density
        if phase == "noisy" and feedback == "pauli_shadows" and iteration < total:
            measured = classical_shadow_feedback(
                density, rng=rng, shots_per_basis=shots_per_basis, num_bases=num_bases, trusted_cptp_output=True
            )
            measurement_calls += 1
        loss, _ = hilbert_schmidt_loss_and_costate(measured, projector)
        exact_loss, _ = hilbert_schmidt_loss_and_costate(density, projector)
        fidelity = float(np.vdot(engine.target, density @ engine.target).real)
        if phase == final_phase and (best_loss is None or exact_loss < best_loss):
            best_loss, best_theta, best_step = exact_loss, theta.copy(), iteration
        records.append({
            "iteration": iteration,
            "phase": phase,
            "theta": theta.tolist(),
            "loss": loss,
            "exact_loss_diagnostic": exact_loss,
            "fidelity": fidelity,
            "feedback_sampled": bool(phase == "noisy" and feedback == "pauli_shadows" and iteration < total),
            "cumulative_cpu_seconds": time.process_time() - cpu,
        })
        if iteration < total:
            _, gradient = robuststate_loss_gradient(engine, theta, feedback_density=measured)
            theta, first, second = adam_step(theta, gradient, first, second, iteration + 1, config)
    return theta, {
        "optimizer": "experiments.state_preparation_exact_adam.adam_step",
        "learning_rate": learning_rate,
        "betas": list(ADAM_BETAS),
        "epsilon": ADAM_EPSILON,
        "adam_state_at_phase_switch": "preserved",
        "noiseless_steps": noiseless_steps,
        "noisy_steps": noisy_steps,
        "gradient_evaluations": total,
        "feedback": feedback,
        "measurement_seed": seed,
        "measurement_calls": measurement_calls,
        "shots_per_basis": shots_per_basis if feedback == "pauli_shadows" else None,
        "bases_per_call": (num_bases or 3**engine.num_qubits) if feedback == "pauli_shadows" else 0,
        "total_shots": measurement_calls * (num_bases or 3**engine.num_qubits) * shots_per_basis,
        "retention": "final_iterate_published",
        "initial_theta": real_parameters(initial, len(engine.parameters)).tolist(),
        "final_theta": theta.tolist(),
        "best_theta_diagnostic": best_theta.tolist(),
        "best_step_diagnostic": best_step,
        "best_exact_loss_diagnostic": best_loss,
        "best_diagnostic_phase": final_phase,
        "records": records,
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall,
    }


def robuststate_candidates(
    target: ArrayLike,
    *,
    hardware: str = "IBM",
    seed: int = 300,
    depths: Sequence[int] = (6,),
    noiseless_steps: int = 500,
    noisy_steps: int = 50,
    learning_rate: float = LEARNING_RATE,
    feedback: str = "exact_density",
    shots_per_basis: int = 1024,
    num_bases: int | None = None,
    noise_basis: str = "native",
) -> list[Candidate]:
    """Train separate published-style line ansatzes for the supplied depth grid.

    Args:
        target: Normalized complex target in Qiskit's little-endian order.
        hardware: Synthetic IBM or Google benchmark profile.
        seed: Root seed; each depth has independent initialization/measurement.
        depths: Distinct nonnegative full traversals of the linear coupling map.
        noiseless_steps: Ideal Adam update budget.
        noisy_steps: Noisy surrogate Adam update budget.
        learning_rate: Constant Adam learning rate.
        feedback: Explicit exact-density or sampled-shadow simulator adaptation.
        shots_per_basis: Samples in each selected Pauli basis.
        num_bases: Number of bases to sample; None means all bases.
        noise_basis: ``native`` applies noise after compiled native gates;
            ``logical`` after each original CX/RZ/RX gate without compilation.

    Returns:
        Final-iterate circuits with the exact trained operation schedule preserved.

    Raises:
        ValueError: If targets, profile, noise basis, grid, settings or seeds are invalid.
    """
    if noise_basis not in {"native", "logical"}:
        msg = "noise_basis must be native or logical."
        raise ValueError(msg)
    state = validate_target(target, max_qubits=12)
    nonnegative_integer(seed, "seed")
    grid = tuple(depths)
    if not grid or any(type(depth) is not int or depth < 0 for depth in grid) or len(set(grid)) != len(grid):
        msg = "depths must be a nonempty sequence of distinct nonnegative integers."
        raise ValueError(msg)
    if hardware not in {"IBM", "Google"}:
        msg = "hardware must be IBM or Google."
        raise ValueError(msg)
    result = []
    for depth in grid:
        cpu, wall = time.process_time(), time.perf_counter()
        n = state.size.bit_length() - 1
        circuit = topology_aware_ansatz(n, depth)
        engine = make_objective(circuit, state, hardware, compile_native=noise_basis == "native")
        initialization_seed, measurement_seed = [
            int(x) for x in np.random.SeedSequence([seed, depth]).generate_state(2)
        ]
        initial = np.random.default_rng(initialization_seed).uniform(-np.pi, np.pi, len(engine.parameters))
        theta, training = train_robuststate(
            engine,
            initial,
            seed=measurement_seed,
            noiseless_steps=noiseless_steps,
            noisy_steps=noisy_steps,
            learning_rate=learning_rate,
            feedback=feedback,
            shots_per_basis=shots_per_basis,
            num_bases=num_bases,
        )
        ideal = engine.ideal_density(theta)
        result.append(
            Candidate(
                engine.bind(theta),
                {
                    "method": "robuststate",
                    "execution_backend": "exact_recursive_checkpoints_and_ideal_statevectors"
                    if n > 6
                    else "exact_dense_tape",
                    "candidate_id": f"robuststate_depth{depth}_{hardware}_seed{seed}_{feedback}",
                    "status": "complete",
                    "paper": PAPER,
                    "implementation": "paper_reimplementation_with_Qiskit_and_NumPy",
                    "ansatz": "Figure10_CNOT_then_RZ_RX_RZ_line_traversals",
                    "depth": depth,
                    "num_qubits": n,
                    "logical_cnot_blocks": depth * (n - 1),
                    "seed": seed,
                    "initialization_seed": initialization_seed,
                    "initialization": "uniform[-pi,pi]; unpublished convention disclosed",
                    "noise_model": hardware,
                    "noise_fine_tuning": noisy_steps > 0,
                    "training_loss": "Hilbert_Schmidt_norm",
                    "gradient": "noisy_output_costate_times_noiseless_density_Jacobian",
                    "feedback_adaptation": feedback,
                    "readout_noise": False,
                    "hardware_execution": False,
                    "additional_pruning": False,
                    "preserve_native_schedule": True,
                    "ideal_fidelity": float(np.vdot(state, ideal @ state).real),
                    "noisy_fidelity": engine.fidelity(theta),
                    "training": training,
                    "cpu_seconds": time.process_time() - cpu,
                    "wall_seconds": time.perf_counter() - wall,
                },
            )
        )
        if noise_basis == "logical":
            result[-1].metadata.update({
                "noise_basis": "logical",
                "preserve_native_schedule": False,
                "training_noise": (
                    "exact synthetic Pauli channel after each original one- or two-qubit gate; no native compilation"
                ),
            })
    return result
