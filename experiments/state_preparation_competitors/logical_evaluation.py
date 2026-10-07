# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent assessment with one noise window after each original operation.

This module never compiles, merges, cancels or reorders gates. Its logical-gate
windows match the original archived rotation-circuit sampler. Different source
gate alphabets remain explicit in the resource receipt. Validation and exact
assessment are separate entry points so a runner can lock selection first.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator, Statevector

from experiments.legacy_topdown_dense import noise_channel
from experiments.state_preparation_competitors.common import validate_target
from experiments.state_preparation_competitors.evaluation import pauli_distribution, validate_circuit
from mqt.yaqs.optimization.statevector import _apply  # noqa: PLC2701  # Reuse tested kernels.

if TYPE_CHECKING:
    from numpy.typing import NDArray

ONE_QUBIT_GATES = frozenset({"id", "rx", "ry", "rz", "sx", "sxdg", "x", "y", "z", "h", "s", "sdg", "t", "tdg"})
TWO_QUBIT_GATES = frozenset({"rxx", "ryy", "rzz", "rzx", "cx", "cy", "cz", "swap"})
NOISE_PLACEMENT = "after every original logical operation, including zero-angle gates; no idle or readout noise"


def validate_logical_circuit(circuit: QuantumCircuit) -> None:
    """Require resolved elementary one/two-qubit gates on an unchanged line.

    Args:
        circuit: Source circuit in its original logical wire ordering.

    Raises:
        TypeError: If the input is not a circuit or contains a nonunitary instruction.
        ValueError: If its register, layout, matrices, gate alphabet or supports are invalid.
    """
    if not isinstance(circuit, QuantumCircuit):
        msg = "Expected a Qiskit logical circuit."
        raise TypeError(msg)
    validate_circuit(circuit, max_qubits=12)
    if circuit.layout is not None:
        msg = "Logical assessment requires the original circuit without a transpiler layout."
        raise ValueError(msg)
    for item in circuit.data:
        sites = [circuit.find_bit(qubit).index for qubit in item.qubits]
        allowed = ONE_QUBIT_GATES if len(sites) == 1 else TWO_QUBIT_GATES
        if (
            len(sites) not in {1, 2}
            or item.operation.name not in allowed
            or (len(sites) == 2 and abs(sites[0] - sites[1]) != 1)
        ):
            msg = "Expected elementary logical gates on one qubit or neighboring pairs; opaque blocks are forbidden."
            raise ValueError(msg)


def _target(circuit: QuantumCircuit, target: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Validate an original logical circuit and its little-endian target together.

    Args:
        circuit: Original source circuit without a transpiler layout.
        target: Normalized target amplitudes in Qiskit order.

    Returns:
        Validated target with the circuit's register size.

    Raises:
        ValueError: If the target register differs from the circuit.
    """
    validate_logical_circuit(circuit)
    value = validate_target(target, max_qubits=12)
    if value.size != 2**circuit.num_qubits:
        msg = "Logical circuit and target must have the same register size."
        raise ValueError(msg)
    return value


def logical_resources(circuit: QuantumCircuit, target: NDArray[np.complex128]) -> dict[str, Any]:
    """Measure the original gate schedule and ideal fidelity without compilation.

    Args:
        circuit: Resolved elementary logical circuit; zero-angle gates stay present.
        target: Target amplitudes in the unchanged logical wire ordering.

    Returns:
        Ideal fidelity and explicitly logical gate counts, depths and alphabet.
    """
    value = _target(circuit, target)
    return {
        "ideal_fidelity": float(abs(np.vdot(value, Statevector.from_instruction(circuit).data)) ** 2),
        "source_gate_count": len(circuit.data),
        "logical_one_qubit": sum(len(item.qubits) == 1 for item in circuit.data),
        "logical_two_qubit": sum(len(item.qubits) == 2 for item in circuit.data),
        "logical_two_qubit_depth": circuit.depth(lambda item: len(item.qubits) == 2),
        "logical_total_depth": circuit.depth(),
        "logical_operations": dict(circuit.count_ops()),
        "noise_basis": "logical",
        "noise_placement": NOISE_PLACEMENT,
        "compilation_performed": False,
    }


def sample_paths(
    circuit: QuantumCircuit, hardware: str, validation_paths: int, entropy: list[int]
) -> NDArray[np.uint8]:
    """Sample unconditional logical-gate Pauli histories from a declared stream.

    Args:
        circuit: Original resolved logical gate schedule.
        hardware: Existing IBM or Google synthetic profile.
        validation_paths: Number of raw paths, at least two.
        entropy: Nonempty list of nonnegative integer stream identifiers.

    Returns:
        One Pauli code per raw trajectory and original gate. No jumps are forced.

    Raises:
        ValueError: If the path count, profile or entropy is invalid.
    """
    validate_logical_circuit(circuit)
    pauli_distribution(hardware, 1)
    if (
        type(validation_paths) is not int
        or validation_paths < 2
        or not isinstance(entropy, list)
        or not entropy
        or any(type(value) is not int or value < 0 for value in entropy)
    ):
        msg = "Validation needs at least two paths and a nonempty nonnegative integer entropy list."
        raise ValueError(msg)
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    codes = np.zeros((validation_paths, len(circuit.data)), dtype=np.uint8)
    for index, item in enumerate(circuit.data):
        probabilities, _ = pauli_distribution(hardware, len(item.qubits))
        codes[:, index] = rng.choice(len(probabilities), size=validation_paths, p=probabilities)
    return codes


def validation_from_paths(
    circuit: QuantumCircuit,
    target: NDArray[np.complex128],
    hardware: str,
    codes: NDArray[np.uint8],
    *,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Score frozen paths with exact empirical no-jump compression and raw-sample SE.

    Args:
        circuit: Original resolved logical gate schedule.
        target: Little-endian target amplitudes.
        hardware: Existing IBM or Google synthetic profile.
        codes: At least two unconditional histories with one integer code per gate.
        batch_size: Maximum simultaneously propagated statevectors.

    Returns:
        Fidelity, raw-sample standard error, jump coverage, norms and bank digest.

    Raises:
        ValueError: If the bank shape, gate codes or propagated states are invalid.
    """
    value = _target(circuit, target)
    if type(batch_size) is not int or batch_size < 1:
        msg = "batch_size must be a positive integer."
        raise ValueError(msg)
    pauli_distribution(hardware, 1)
    codes = np.asarray(codes)
    if (
        codes.ndim != 2
        or codes.shape[1] != len(circuit.data)
        or len(codes) < 2
        or codes.dtype.kind not in "iu"
        or np.any(codes < 0)
    ):
        msg = "Validation bank must contain integer gate codes matching the logical circuit."
        raise ValueError(msg)
    has_jump = np.any(codes != 0, axis=1)
    indices = np.flatnonzero(has_jump).tolist()
    multiplicities = [1] * len(indices)
    no_jump = int(np.count_nonzero(~has_jump))
    if no_jump:
        indices.insert(0, int(np.flatnonzero(~has_jump)[0]))
        multiplicities.insert(0, no_jump)
    retained = codes[indices]
    analytic_no_jump = 1.0
    schedule = []
    for index, item in enumerate(circuit.data):
        sites = tuple(reversed([circuit.find_bit(qubit).index for qubit in item.qubits]))
        probabilities, matrices = pauli_distribution(hardware, len(sites))
        if np.any(codes[:, index] >= len(probabilities)):
            msg = "Invalid logical Pauli code for the gate support."
            raise ValueError(msg)
        analytic_no_jump *= float(probabilities[0])
        schedule.append((sites, Operator(item.operation).data, matrices))
    fidelities = np.empty(len(retained))
    norm_error = 0.0
    for start in range(0, len(retained), batch_size):
        batch = retained[start : start + batch_size]
        states = np.zeros((len(batch), value.size), dtype=np.complex128)
        states[:, 0] = 1
        for index, (sites, unitary, matrices) in enumerate(schedule):
            states = _apply(states, unitary, sites, circuit.num_qubits)
            for code in range(1, len(matrices)):
                selected = np.flatnonzero(batch[:, index] == code)
                if selected.size:
                    states[selected] = _apply(states[selected], matrices[code], sites, circuit.num_qubits)
        fidelities[start : start + len(batch)] = abs(states @ value.conj()) ** 2
        norm_error = max(norm_error, float(np.max(abs(np.sum(abs(states) ** 2, axis=1) - 1))))
    weights = np.array(multiplicities, dtype=float) / len(codes)
    mean = float(weights @ fidelities)
    if not np.isfinite(mean) or not -1e-9 <= mean <= 1 + 1e-9 or norm_error > 1e-9:
        msg = "Logical trajectory propagation produced an invalid normalized state or fidelity."
        raise ValueError(msg)
    return {
        "fidelity": mean,
        "standard_error": float(np.sqrt(weights @ (fidelities - mean) ** 2 / (len(codes) - 1))),
        "sampled_paths": len(codes),
        "no_jump_paths": no_jump,
        "jump_paths": int(has_jump.sum()),
        "retained_paths": len(retained),
        "analytic_no_jump_probability": analytic_no_jump,
        "norm_max_error": norm_error,
        "bank_sha256": hashlib.sha256(codes.astype(np.uint8).tobytes()).hexdigest(),
    }


def score_validation(
    circuit: QuantumCircuit,
    target: NDArray[np.complex128],
    hardware: str,
    validation_paths: int,
    entropy: list[int],
) -> dict[str, Any]:
    """Sample and assess a fixed validation bank without computing an exact score.

    Args:
        circuit: Unchanged elementary logical gate schedule.
        target: Little-endian target amplitudes.
        hardware: Existing synthetic IBM or Google profile.
        validation_paths: Raw validation sample count, at least two.
        entropy: Predeclared stream independent of training and other roles.

    Returns:
        Validation-only statistics, bank hash and a copy of the declared entropy.
    """
    _target(circuit, target)
    codes = sample_paths(circuit, hardware, validation_paths, entropy)
    return {**validation_from_paths(circuit, target, hardware, codes), "entropy": entropy.copy()}


def score_exact(circuit: QuantumCircuit, target: NDArray[np.complex128], hardware: str) -> dict[str, float]:
    """Integrate the original one-jump logical law after the caller locks selection.

    Args:
        circuit: Original elementary logical circuit, with every window preserved.
        target: Little-endian target amplitudes.
        hardware: Existing synthetic IBM or Google profile.

    Returns:
        Exact fidelity and density validity diagnostics, without trajectory error.

    Raises:
        ValueError: If the profile, circuit, target or final density is invalid.
    """
    value = _target(circuit, target)
    pauli_distribution(hardware, 1)
    rho = DensityMatrix.from_label("0" * circuit.num_qubits)
    for item in circuit.data:
        sites = [circuit.find_bit(qubit).index for qubit in item.qubits]
        rho = rho.evolve(item.operation, qargs=sites)
        rho = rho.evolve(noise_channel(hardware, len(sites), "tjm"), qargs=sites)
    matrix = rho.data
    result = {
        "fidelity": float(np.vdot(value, matrix @ value).real),
        "trace_error": float(abs(np.trace(matrix) - 1)),
        "hermiticity_error": float(np.max(abs(matrix - matrix.conj().T))),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(matrix).min()),
    }
    if (
        not all(np.isfinite(number) for number in result.values())
        or not -1e-9 <= result["fidelity"] <= 1 + 1e-9
        or result["trace_error"] > 1e-9
        or result["hermiticity_error"] > 1e-9
        or result["minimum_eigenvalue"] < -1e-9
    ):
        msg = "Exact logical channel propagation produced an invalid density matrix."
        raise ValueError(msg)
    return result
