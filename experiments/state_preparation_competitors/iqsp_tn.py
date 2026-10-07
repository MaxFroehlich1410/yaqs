# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Interpolative state preparation using differentiable tensor networks.

The Gaussian path follows Ballarin et al. Other targets use an explicitly
declared geodesic extension. Ideal blocks use the author fork's exponential
SU(4) coordinates. Final noisy refinement uses elementary logical rotations
with the same Pauli channel as independent logical assessment.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, ParameterVector
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator, Statevector
from qiskit.synthesis import TwoQubitWeylDecomposition
from scipy.linalg import polar

from experiments.state_preparation_competitors.common import Candidate, validate_target
from experiments.state_preparation_competitors.evaluation import pauli_distribution
from experiments.state_preparation_competitors.iqsp import GAUSSIAN_TARGET_ID, gaussian_state, validate_gaussian_target
from experiments.state_preparation_competitors.iqsp_comb_backend import (
    CombLayout,
    CombTarget,
    CombTensorNetworkObjective,
    comb_target_from_vector,
)
from experiments.state_preparation_competitors.iqsp_tci import DEFAULT_TCI_OPTIONS, TCIOptions, tci_target_bank
from experiments.state_preparation_competitors.iqsp_tn_backend import TensorNetworkObjective
from experiments.state_preparation_competitors.logical_evaluation import NOISE_PLACEMENT, validate_logical_circuit
from experiments.state_preparation_competitors.noisy_core import nonnegative_integer, real_parameters
from experiments.state_preparation_exact_adam import AdamConfig, adam_step

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

    from experiments.state_preparation_competitors.iqsp_comb_backend import CombOptions

METHOD = "iqsp_tn"
TWO_QUBIT_CUTOFF = 1e-4


def interpolation_target(target: ArrayLike, coordinate: float, *, target_id: str) -> NDArray[np.complex128]:
    """Return a normalized interpolation state in Qiskit little-endian order.

    The Gaussian uses the published exponential family with the fixture's
    amplitude covariance. Other targets follow the shortest phase-aligned
    great-circle path from the uniform state. This latter path is a benchmark
    extension, not a claim of reproducing the paper's real-function theorem.

    Args:
        target: Normalized pure target, including relative complex phases.
        coordinate: Interpolation coordinate in [0, 1].
        target_id: Gaussian fixture identifier or a name for a generalized path.

    Returns:
        Private normalized interpolated amplitudes.

    Raises:
        ValueError: If inputs are invalid or the Gaussian identifier is misused.
    """
    value = validate_target(target, max_qubits=12)
    if (
        isinstance(coordinate, (bool, np.bool_))
        or not isinstance(coordinate, (int, float, np.integer, np.floating))
        or not math.isfinite(float(coordinate))
        or not 0 <= coordinate <= 1
        or not isinstance(target_id, str)
        or not target_id
    ):
        msg = "Require a target identifier and a finite interpolation coordinate in [0,1]."
        raise ValueError(msg)
    if target_id == GAUSSIAN_TARGET_ID:
        validate_gaussian_target(value, max_qubits=12)
        return gaussian_state(value.size.bit_length() - 1, float(coordinate))
    uniform = np.full(value.size, 1 / math.sqrt(value.size), dtype=np.complex128)
    overlap = np.vdot(uniform, value)
    phase = np.angle(overlap) if abs(overlap) > 1e-14 else np.angle(value[np.argmax(abs(value))])
    aligned = value * np.exp(-1j * phase)
    cosine = float(np.clip(np.vdot(uniform, aligned).real, 0, 1))
    angle = math.acos(cosine)
    if coordinate == 0:
        return uniform
    if coordinate == 1:
        return aligned.copy()
    if angle < 1e-7:
        result = (1 - coordinate) * uniform + coordinate * aligned
    else:
        result = (math.sin((1 - coordinate) * angle) * uniform + math.sin(coordinate * angle) * aligned) / math.sin(
            angle
        )
    return np.asarray(result / np.linalg.norm(result), dtype=np.complex128)


def iqsp_tn_ansatz(qubits: int, depth: int) -> QuantumCircuit:
    """Build Hadamards and nearest-neighbor exponential SU(4) brickwork.

    Each block has 15 Pauli coordinates and is identity at zero. The backend
    interprets ``su4`` as exp(+i sum_j theta_j P_j), matching the author fork.

    Args:
        qubits: Register size between two and twelve.
        depth: Positive number of even/odd brickwork cycles.

    Returns:
        Symbolic circuit; SU(4) gates are resolved by the tensor backend.

    Raises:
        ValueError: If the dimensions are unsupported.
    """
    nonnegative_integer(depth, "depth")
    if type(qubits) is not int or not 2 <= qubits <= 12 or depth < 1:
        msg = "IQSP-TN requires two to twelve qubits and positive depth."
        raise ValueError(msg)
    circuit = QuantumCircuit(qubits)
    circuit.h(range(qubits))
    parameters = ParameterVector("iqsp_su4", 15 * (qubits - 1) * depth)
    offset = 0
    for _ in range(depth):
        for parity in (0, 1):
            for first in range(parity, qubits - 1, 2):
                circuit.append(Gate("su4", 2, list(parameters[offset : offset + 15])), [first, first + 1])
                offset += 15
    return circuit


def _weyl_block(matrix: NDArray[np.complex128]) -> tuple[QuantumCircuit, float]:
    """Remove only roundoff nonunitarity if Qiskit's Weyl eigensolve fails.

    Returns:
        The original 15-rotation decomposition and maximum polar correction.

    Raises:
        QiskitError: If synthesis fails for a reason other than the known eigensolve.
        ValueError: If a polar correction exceeds 1e-12, beyond roundoff repair.
    """
    correction = 0.0
    try:
        weyl = TwoQubitWeylDecomposition(matrix, fidelity=None)
    except QiskitError as error:
        if "failed to diagonalize M2" not in str(error):
            raise
        unitary, _ = polar(matrix)
        correction = float(np.max(abs(unitary - matrix)))
        if correction > 1e-12:
            msg = "Weyl roundoff repair exceeds the 1e-12 local correction limit."
            raise ValueError(msg) from error
        weyl = TwoQubitWeylDecomposition(unitary, fidelity=None)
    return weyl.circuit(euler_basis="ZYZ", simplify=False), correction


def logical_parameterization(bound: QuantumCircuit) -> tuple[QuantumCircuit, NDArray[np.float64], dict[str, Any]]:
    """Exactly lower ideal learned SU(4) blocks into logical scalar rotations.

    Every block uses the same unsimplified 15-rotation ZYZ/Weyl decomposition.
    Independent angles are then optimized with noise after every elementary
    operation. Zero-angle rotations remain noise windows throughout training.
    The paper's small-angle two-qubit removal is a separate final operation.
    This transition adapts hardware-native refinement to the logical alphabet.

    Args:
        bound: Resolved H-plus-two-qubit-unitary circuit from ideal training.

    Returns:
        Symbolic logical circuit, initial angles, and local equivalence audit.

    Raises:
        ValueError: If the schedule is invalid or exact decomposition fails.
    """
    if not isinstance(bound, QuantumCircuit) or bound.parameters or not 2 <= bound.num_qubits <= 12:
        msg = "Expected a resolved two-to-twelve-qubit IQSP circuit."
        raise ValueError(msg)
    lowered = QuantumCircuit(bound.num_qubits)
    lowered.global_phase = bound.global_phase
    errors = []
    phase_corrections = []
    polar_corrections = []
    for item in bound.data:
        sites = [bound.find_bit(qubit).index for qubit in item.qubits]
        if len(sites) == 1 and item.operation.name == "h":
            lowered.h(sites[0])
        elif len(sites) == 2 and abs(sites[0] - sites[1]) == 1 and item.operation.name == "unitary":
            matrix = Operator(item.operation).data
            block, polar_correction = _weyl_block(matrix)
            polar_corrections.append(polar_correction)
            # Weyl branch choices can differ by a global sign. Correct only the
            # global phase, then independently require full matrix equivalence.
            phase = float(np.angle(np.vdot(matrix, Operator(block).data)))
            block.global_phase = float(block.global_phase) - phase
            error = float(np.max(abs(Operator(block).data - matrix)))
            if error > 1e-10 or len(block.data) != 15:
                msg = "Exact unsimplified SU(4) logical lowering failed."
                raise ValueError(msg)
            errors.append(error)
            phase_corrections.append(phase)
            lowered.compose(block, qubits=sites, inplace=True)
        else:
            msg = "Expected only Hadamards and resolved neighboring SU(4) blocks."
            raise ValueError(msg)
    if not errors:
        msg = "The ideal IQSP endpoint contains no SU(4) blocks."
        raise ValueError(msg)
    rotations = {"rz", "ry", "rxx", "ryy", "rzz"}
    angles = np.array([float(item.operation.params[0]) for item in lowered.data if item.operation.name in rotations])
    parameters = ParameterVector("iqsp_logical", len(angles))
    template = QuantumCircuit(bound.num_qubits)
    template.global_phase = lowered.global_phase
    cursor = 0
    for item in lowered.data:
        sites = [lowered.find_bit(qubit).index for qubit in item.qubits]
        if item.operation.name in rotations:
            getattr(template, item.operation.name)(parameters[cursor], *sites)
            cursor += 1
        else:
            template.append(item.operation, sites)
    restored = template.assign_parameters(dict(zip(template.parameters, angles, strict=True)))
    validate_logical_circuit(restored)
    # Local unitary checks prove circuit equivalence without allocating a 4**n operator.
    if bound.num_qubits <= 6:
        error = float(np.max(abs(Operator(restored).data - Operator(bound).data)))
    else:
        error = float(
            np.max(abs(Statevector.from_instruction(restored).data - Statevector.from_instruction(bound).data))
        )
    if error > 1e-9:
        msg = "Logical reparameterization changed the IQSP unitary."
        raise ValueError(msg)
    return (
        template,
        angles,
        {
            "su4_blocks": len(errors),
            "rotation_count": len(angles),
            "maximum_local_unitary_error": max(errors),
            "local_global_phase_corrections": phase_corrections,
            "weyl_polar_corrections": polar_corrections,
            "global_unitary_error": error if bound.num_qubits <= 6 else None,
            "prepared_state_error": error if bound.num_qubits > 6 else None,
            "equivalence_verification": "all local blocks, plus full unitary at q<=6 or prepared state above",
            "lowering": "Qiskit exact Weyl/ZYZ; fidelity=None; simplify=False; no simplification",
            "noisy_parameterization": "independent angles on fixed logical H/RY/RZ/RXX/RYY/RZZ schedule",
        },
    )


def remove_small_two_qubit_rotations(circuit: QuantumCircuit) -> tuple[QuantumCircuit, dict[str, Any]]:
    """Apply the paper's small-angle cutoff once after noisy refinement.

    Qiskit angles are signed, so the threshold applies to their magnitude.
    No single-qubit rotations or other two-qubit operations are removed.
    Final assessment must use the returned schedule with its new noise windows.

    Args:
        circuit: Fully bound elementary logical circuit.

    Returns:
        Circuit after deletion and a receipt identifying every removed window.
    """
    validate_logical_circuit(circuit)
    result = QuantumCircuit(circuit.num_qubits)
    result.global_phase = circuit.global_phase
    removed = []
    for index, item in enumerate(circuit.data):
        sites = [circuit.find_bit(qubit).index for qubit in item.qubits]
        if item.operation.name in {"rxx", "ryy", "rzz"} and abs(float(item.operation.params[0])) <= TWO_QUBIT_CUTOFF:
            removed.append({
                "original_index": index,
                "gate": item.operation.name,
                "qubits": sites,
                "angle": float(item.operation.params[0]),
            })
        else:
            result.append(item.operation, sites)
    return result, {
        "policy": "paper small-angle two-qubit removal; absolute signed rotation angle <= 1e-4",
        "placement": "once after noisy optimization; timing chosen for this benchmark adapter",
        "threshold": TWO_QUBIT_CUTOFF,
        "removed": removed,
        "input_windows": len(circuit.data),
        "output_windows": len(result.data),
        "additional_optimization_steps": 0,
    }


def optimize_tn_adam(
    objective: TensorNetworkObjective | CombTensorNetworkObjective,
    initial: ArrayLike,
    *,
    steps: int,
    learning_rate: float = 0.01,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Apply the existing tested Adam update to a differentiable TN objective.

    Args:
        objective: Pure or locally purified network objective.
        initial: Starting parameters in the objective's parameter order.
        steps: Nonnegative number of Adam updates.
        learning_rate: Positive finite Adam learning rate.

    Returns:
        Final parameters and complete compactable optimization history.

    Raises:
        ValueError: If the budget, learning rate or parameters are invalid.
        FloatingPointError: If an objective or gradient is invalid.
    """
    nonnegative_integer(steps, "steps")
    if isinstance(learning_rate, (bool, np.bool_)) or not np.isfinite(learning_rate) or learning_rate <= 0:
        msg = "Adam learning rate must be positive and finite."
        raise ValueError(msg)
    config = AdamConfig(updates=max(steps, 1), learning_rate=learning_rate)
    theta = real_parameters(initial, len(objective.parameters))
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    first, second = np.zeros_like(theta), np.zeros_like(theta)
    best_theta, best_loss, best_iteration = theta.copy(), math.inf, 0
    records = []
    for iteration in range(steps + 1):
        loss, gradient = objective.loss_gradient(theta)
        if (
            not np.isfinite(loss)
            or not -1e-9 <= loss <= 1 + 1e-9
            or gradient.shape != theta.shape
            or not np.all(np.isfinite(gradient))
        ):
            msg = "TN infidelity and gradient must be finite and valid."
            raise FloatingPointError(msg)
        if loss < best_loss:
            best_theta, best_loss, best_iteration = theta.copy(), float(loss), iteration
        records.append({
            "iteration": iteration,
            "theta": theta.tolist(),
            "fidelity": 1 - float(loss),
            "gradient_norm": float(np.linalg.norm(gradient)),
            "cumulative_objective_evaluations": iteration + 1,
            "cumulative_gradient_evaluations": iteration + 1,
            "cumulative_cpu_seconds": time.process_time() - start_cpu,
        })
        if iteration < steps:
            theta, first, second = adam_step(theta, gradient, first, second, iteration + 1, config)
    return theta, {
        "optimizer": "existing standard bias-corrected Adam",
        "update_source": "experiments/state_preparation_exact_adam.py:adam_step",
        "retention": "final_iterate; best recorded only",
        "moments": "reset per interpolation stage and final noisy phase",
        "learning_rate": config.learning_rate,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "epsilon": config.epsilon,
        "steps": steps,
        "initial_theta": records[0]["theta"],
        "initial_fidelity": records[0]["fidelity"],
        "final_theta": theta.tolist(),
        "final_fidelity": records[-1]["fidelity"],
        "best_theta": best_theta.tolist(),
        "best_fidelity": 1 - best_loss,
        "best_iteration": best_iteration,
        "records": records,
        "cpu_seconds": time.process_time() - start_cpu,
        "wall_seconds": time.perf_counter() - start_wall,
    }


def _training_objective(
    template: QuantumCircuit,
    target: ArrayLike,
    hardware: str | None,
    comb_options: CombOptions | None,
    target_comb: CombTarget | None = None,
) -> TensorNetworkObjective | CombTensorNetworkObjective:
    """Select an explicit backend without changing the training or noise schedule."""
    if comb_options is None:
        return TensorNetworkObjective(template, target, hardware)

    if target_comb is None:
        target_comb = comb_target_from_vector(
            target, CombLayout(template.num_qubits), relative_tolerance=comb_options.target_tolerance
        )
    return CombTensorNetworkObjective(template, target_comb, hardware, options=comb_options)


def iqsp_tn_candidates(
    target: ArrayLike,
    *,
    target_id: str,
    depths: Sequence[int] = (2, 4),
    interpolation_steps: int = 20,
    construction_steps: int = 200,
    noisy_steps: int = 10000,
    hardware: str = "IBM",
    learning_rate: float = 0.01,
    comb_options: CombOptions | None = None,
    target_construction: str = "tci",
    tci_options: TCIOptions = DEFAULT_TCI_OPTIONS,
) -> list[Candidate]:
    """Train each declared depth with interpolation and final logical noise.

    Args:
        target: Normalized target amplitudes, including complex targets.
        target_id: Fixture identifier or explicit generalized-path label.
        depths: Unique positive even/odd brickwork cycle counts.
        interpolation_steps: Number of nonzero equally spaced path coordinates.
        construction_steps: Adam updates per interpolation stage.
        noisy_steps: Adam updates at the final target with logical noise.
        hardware: Existing synthetic IBM or Google Pauli profile.
        learning_rate: Positive finite learning rate used in every phase.
        comb_options: Explicit compressed-comb execution; None preserves the exact backend.
        target_construction: Compressed targets use TCI by default. Explicit
            ``dense_svd`` preserves the historical diagnostic adapter.
        tci_options: Recorded TCI construction and target validation settings.

    Returns:
        Final logical circuits with complete training and transition receipts.

    Raises:
        ValueError: If target, grid, budgets or noise profile are invalid.
    """
    value = validate_target(target, max_qubits=12)
    qubits = value.size.bit_length() - 1
    if target_construction not in {"tci", "dense_svd"} or not isinstance(tci_options, TCIOptions):
        msg = "Require target_construction tci or dense_svd and valid TCIOptions."
        raise ValueError(msg)
    if comb_options is not None and target_construction == "tci" and comb_options.target_tolerance != 0:
        msg = "TCI target construction cannot use the dense-SVD target_tolerance."
        raise ValueError(msg)
    if qubits > 6 and comb_options is None:
        msg = "Above six qubits, explicitly select compressed-comb execution."
        raise ValueError(msg)
    nonnegative_integer(interpolation_steps, "interpolation_steps")
    nonnegative_integer(construction_steps, "construction_steps")
    nonnegative_integer(noisy_steps, "noisy_steps")
    if qubits < 2 or interpolation_steps < 1 or not depths or len(set(depths)) != len(depths):
        msg = "Require q2-q12, a nonempty unique depth grid and at least one interpolation stage."
        raise ValueError(msg)
    for depth in depths:
        iqsp_tn_ansatz(qubits, depth)
    if isinstance(learning_rate, (bool, np.bool_)) or not np.isfinite(learning_rate) or learning_rate <= 0:
        msg = "Adam learning rate must be positive and finite."
        raise ValueError(msg)
    # Validate the path and both arity channels before any optimization starts.
    interpolation_target(value, 0, target_id=target_id)
    channels = {}
    for arity in (1, 2):
        probabilities, matrices = pauli_distribution(hardware, arity)
        channel_hash = hashlib.sha256(probabilities.tobytes())
        for matrix in matrices:
            channel_hash.update(matrix.tobytes())
        channels[str(arity)] = {"probabilities": probabilities.tolist(), "sha256": channel_hash.hexdigest()}
    candidates = []
    for depth in depths:
        started_cpu, started_wall = time.process_time(), time.perf_counter()
        template = iqsp_tn_ansatz(qubits, depth)
        theta = np.zeros(len(template.parameters))
        target_bank: tuple[CombTarget, ...] = ()
        construction_receipt = None
        if comb_options is not None and target_construction == "tci":
            coordinates = [stage / interpolation_steps for stage in range(1, interpolation_steps + 1)]
            target_bank, construction_receipt = tci_target_bank(
                [interpolation_target(value, coordinate, target_id=target_id) for coordinate in coordinates],
                gaussian_coordinates=coordinates if target_id == GAUSSIAN_TARGET_ID else None,
                options=tci_options,
            )
        stages = []
        for stage in range(1, interpolation_steps + 1):
            coordinate = stage / interpolation_steps
            intermediate = interpolation_target(value, coordinate, target_id=target_id)
            objective = _training_objective(
                template, intermediate, None, comb_options, target_bank[stage - 1] if target_bank else None
            )
            theta, receipt = optimize_tn_adam(objective, theta, steps=construction_steps, learning_rate=learning_rate)
            stages.append({
                "coordinate": coordinate,
                "optimization": receipt,
                "tensor_network": objective.diagnostics(theta),
            })
        ideal = objective.bind(theta)
        transition_cpu = time.process_time()
        noisy_template, noisy_initial, transition = logical_parameterization(ideal)
        transition["cpu_seconds"] = time.process_time() - transition_cpu
        final_target = target_bank[-1] if target_bank else None
        noisy_objective = _training_objective(noisy_template, value, hardware, comb_options, final_target)
        refined, noisy_receipt = optimize_tn_adam(
            noisy_objective, noisy_initial, steps=noisy_steps, learning_rate=learning_rate
        )
        circuit, postprocessing = remove_small_two_qubit_rotations(noisy_objective.bind(refined))
        validate_logical_circuit(circuit)
        final_objective = _training_objective(circuit, value, hardware, comb_options, final_target)
        postprocessing["pre_removal_noisy_fidelity"] = noisy_receipt["final_fidelity"]
        postprocessing["final_noisy_fidelity"] = final_objective.fidelity([])
        postprocessing["final_tensor_network"] = final_objective.diagnostics([])
        candidates.append(
            Candidate(
                circuit,
                {
                    "method": METHOD,
                    "candidate_id": f"depth_{depth:02d}",
                    "status": "complete",
                    "depth": depth,
                    "target_id": target_id,
                    "target_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                    "hardware": hardware,
                    "noise_basis": "logical",
                    "noise_placement": NOISE_PLACEMENT,
                    "channels": channels,
                    "interpolation_scope": "published_gaussian_path"
                    if target_id == GAUSSIAN_TARGET_ID
                    else "generalized_complex_geodesic",
                    "initialization": "zero exponential-SU4 parameters after Hadamards; deterministic",
                    "tensor_representation": "Ballarin compressed pure/locally purified comb"
                    if comb_options is not None
                    else "exact pure and Kraus-purification circuit tensor networks; target MPS",
                    "compression": asdict(comb_options) if comb_options is not None else None,
                    "target_construction": target_construction if comb_options is not None else "legacy_exact",
                    "target_construction_receipt": construction_receipt,
                    "scope": "q2-q12 fixture adaptation with author compressed-comb kernels"
                    if comb_options is not None
                    else "six-qubit TN adaptation; no compressed-comb runtime or large-system scaling claim",
                    "interpolation_steps": interpolation_steps,
                    "construction_steps_per_stage": construction_steps,
                    "stages": stages,
                    "logical_transition": transition,
                    "noisy": noisy_receipt,
                    "noisy_tensor_network": noisy_objective.diagnostics(refined),
                    "postprocessing": postprocessing,
                    "cpu_seconds": time.process_time()
                    - started_cpu
                    + (construction_receipt["child_cpu_seconds"] if construction_receipt else 0),
                    "wall_seconds": time.perf_counter() - started_wall,
                },
            )
        )
    return candidates
