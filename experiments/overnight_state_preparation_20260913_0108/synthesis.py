# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Target-informed rank-two preparation with fully physical Weyl lowering.

Known sequential MPS preparation (Schön et al., quant-ph/0501096; Ran,
arXiv:1908.07958), implemented here as a bounded dense diagnostic. No arbitrary
multi-qubit unitary is passed to the benchmark evaluator.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer, TwoQubitWeylDecomposition

from experiments.state_preparation_pilot import circuit_record
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

VERSION = "overnight.rank_two_sequential_weyl.v1"
COMPILER_ATOL = 0.0
PRIMITIVES = frozenset({"rx", "ry", "rz", "rxx", "ryy", "rzz"})


def _qubit_count(vector: NDArray[np.complex128]) -> int:
    """Return the size of a normalized finite q1-q6 vector.

    Raises:
        ValueError: If the vector violates the small-system physical domain.
    """
    if vector.ndim != 1 or vector.size < 2 or vector.size > 64:
        msg = "Expected a normalized finite q1-q6 vector."
        raise ValueError(msg)
    n = vector.size.bit_length() - 1
    if 2**n != vector.size or not np.all(np.isfinite(vector)) or abs(np.vdot(vector, vector) - 1) > 1e-10:
        msg = "Expected a normalized finite q1-q6 vector."
        raise ValueError(msg)
    return n


def apply_local(
    vector: NDArray[np.complex128], matrix: NDArray[np.complex128], sites: tuple[int, ...], num_qubits: int
) -> NDArray[np.complex128]:
    """Return a local action with sites ordered from most to least significant."""
    axes = tuple(num_qubits - 1 - site for site in sites)
    front = tuple(range(len(sites)))
    moved = np.moveaxis(vector.reshape((2,) * num_qubits), axes, front)
    result = (matrix @ moved.reshape(2 ** len(sites), -1)).reshape(moved.shape)
    return np.moveaxis(result, front, axes).reshape(vector.shape)


def rank_two_approximation(target: NDArray[np.complex128]) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    """Return a normalized rank-two tensor-train approximation in fixed site order.

    The SVD order is physical q0,...,q(n-1), with no intermediate renormalization.
    Discarded squared singular values therefore share the original norm scale.
    This is a sequential truncated SVD, not a globally optimal low-rank fit.
    """
    target = np.asarray(target, dtype=complex)
    n = _qubit_count(target)
    current = target.reshape((2,) * n).transpose(tuple(reversed(range(n))))
    cores = []
    discarded = []
    bond = 1
    for _ in range(n - 1):
        left, singular, right = np.linalg.svd(current.reshape(bond * 2, -1), full_matrices=False)
        retained = min(2, len(singular))
        discarded.append(float(np.sum(singular[retained:] ** 2)))
        cores.append(left[:, :retained].reshape(bond, 2, retained))
        current = singular[:retained, None] * right[:retained]
        bond = retained
    cores.append(current.reshape(bond, 2, 1))
    contracted = cores[0]
    for core in cores[1:]:
        contracted = np.tensordot(contracted, core, axes=([-1], [0]))
    vector = contracted.reshape((2,) * n).transpose(tuple(reversed(range(n)))).reshape(-1)
    retained_norm_squared = float(np.vdot(vector, vector).real)
    vector = np.divide(vector, np.sqrt(retained_norm_squared))
    return vector, {
        "svd_site_order": list(range(n)),
        "max_bond_dimension": 2,
        "discarded_squared_singular_values": discarded,
        "total_discarded_weight": sum(discarded),
        "retained_norm_squared": retained_norm_squared,
        "approximation_fidelity": float(abs(np.vdot(target, vector)) ** 2),
    }


def _support_basis(unfolding: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Return a full orthonormal SVD basis with deterministic column phases."""
    basis = np.linalg.svd(unfolding, full_matrices=True)[0]
    for column in range(basis.shape[1]):
        pivot = basis[int(np.argmax(abs(basis[:, column]))), column]
        basis[:, column] *= np.conj(pivot) / abs(pivot)
    return basis


def rank_two_preparation_gates(
    vector: NDArray[np.complex128],
) -> tuple[list[tuple[NDArray[np.complex128], tuple[int, ...]]], dict[str, Any]]:
    """Return exact local preparation matrices for a supplied rank-two state.

    Every pair basis maps its occupied support to zero on the earlier site.
    A final one-qubit basis change maps the last site to zero. Reversing these
    adjoints gives the complete preparation from the physical all-zero state.

    Raises:
        ValueError: If the vector is not rank two across every physical cut.
    """
    current = np.array(vector, dtype=complex, copy=True)
    n = _qubit_count(current)
    disentanglers: list[tuple[NDArray[np.complex128], tuple[int, ...]]] = []
    residual_weights = []
    for site in range(n - 1):
        sites = (site, site + 1)
        axes = tuple(n - 1 - q for q in sites)
        unfolding = np.moveaxis(current.reshape((2,) * n), axes, (0, 1)).reshape(4, -1)
        matrix = _support_basis(unfolding).conj().T
        current = apply_local(current, matrix, sites, n)
        occupied = np.moveaxis(current.reshape((2,) * n), n - 1 - site, 0)[1]
        residual_weights.append(float(np.vdot(occupied, occupied).real))
        if residual_weights[-1] > 1e-10:
            msg = "Sequential support does not have rank at most two."
            raise ValueError(msg)
        disentanglers.append((matrix, sites))
    sites = (n - 1,)
    unfolding = current.reshape(2, -1)
    matrix = _support_basis(unfolding).conj().T
    current = apply_local(current, matrix, sites, n)
    disentanglers.append((matrix, sites))
    zero_fidelity = float(abs(current[0]) ** 2)
    if abs(zero_fidelity - 1) > 1e-10:
        msg = "Sequential support does not have rank at most two."
        raise ValueError(msg)
    preparation = [(operator.conj().T, sites) for operator, sites in reversed(disentanglers)]
    return preparation, {
        "disentangled_zero_fidelity": zero_fidelity,
        "residual_excited_weights": residual_weights,
        "logical_one_qubit_gates": 1,
        "logical_two_qubit_gates": n - 1,
    }


def lower_local_unitaries(
    num_qubits: int,
    local_gates: Sequence[tuple[NDArray[np.complex128], tuple[int, ...]]],
) -> tuple[ParameterizedCircuit, NDArray[np.float64], dict[str, Any]]:
    """Return scalar primitives for local unitaries, with all physical noise enabled.

    Matrices use the supplied site order, most significant site first. Qiskit's
    local two-qubit basis is reversed explicitly. The returned circuit omits a
    harmless global phase, reported separately. No native CX or arbitrary matrix
    is counted as an ideal gate in the physical output.

    Raises:
        ValueError: If a matrix, gate support, or primitive is unsupported.
    """
    if type(num_qubits) is not int or not 1 <= num_qubits <= 6:
        msg = "Expected one to six physical qubits."
        raise ValueError(msg)
    gates, theta, mapping = [], [], []
    global_phase = 0.0
    maximum_unitary_error = 0.0
    for origin, (raw_matrix, sites) in enumerate(local_gates):
        matrix = np.asarray(raw_matrix, dtype=complex)
        arity = len(sites)
        if (
            arity not in {1, 2}
            or len(set(sites)) != arity
            or any(type(site) is not int or not 0 <= site < num_qubits for site in sites)
            or matrix.shape != (2**arity, 2**arity)
            or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix.conj().T @ matrix, np.eye(2**arity), atol=1e-10, rtol=0)
        ):
            msg = "Expected valid one-/two-qubit unitary matrices and distinct physical sites."
            raise ValueError(msg)
        if arity == 1:
            reference = matrix
            compiled = OneQubitEulerDecomposer("ZYZ")(matrix, simplify=False, atol=COMPILER_ATOL)
        else:
            swapped = matrix.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
            reference = swapped
            compiled = TwoQubitWeylDecomposition(swapped, fidelity=None).circuit(
                euler_basis="ZYZ", simplify=False, atol=COMPILER_ATOL
            )
        # Compare the complete unitary, accepting only a common global phase.
        # Qiskit's unsimplified Weyl output can differ by a sign, so derive the
        # phase from matrices rather than trusting its reported phase alone.
        actual = Operator(compiled).data
        correction = float(np.angle(np.vdot(actual, reference)))
        unitary_error = float(np.max(abs(np.exp(1j * correction) * actual - reference)))
        if unitary_error > 1e-10:
            msg = "Weyl lowering did not reproduce the complete local unitary."
            raise ValueError(msg)
        maximum_unitary_error = max(maximum_unitary_error, unitary_error)
        global_phase += float(compiled.global_phase) + correction
        first = len(gates)
        for instruction in compiled.data:
            name = instruction.operation.name
            if name not in PRIMITIVES or len(instruction.operation.params) != 1:
                msg = f"Weyl lowering emitted unsupported physical primitive {name}."
                raise ValueError(msg)
            physical_sites = tuple(sites[compiled.find_bit(qubit).index] for qubit in instruction.qubits)
            index = len(theta)
            theta.append(float(instruction.operation.params[0]))
            gates.append(ParameterizedGate(name, physical_sites, index, logical_gate_id=index, noise_enabled=True))
        mapping.append({
            "logical_gate": origin,
            "sites": list(sites),
            "primitive_indices": list(range(first, len(gates))),
            "unitary_error": unitary_error,
            "compiler_phase_correction": correction,
        })
    circuit = ParameterizedCircuit(num_qubits, gates, num_params=len(theta))
    return (
        circuit,
        np.asarray(theta, dtype=float),
        {
            "compiler": "qiskit.TwoQubitWeylDecomposition(fidelity=None), ZYZ",
            "compiler_identity_tolerance": COMPILER_ATOL,
            "compiler_simplification": False,
            "omitted_global_phase": global_phase,
            "maximum_unitary_error": maximum_unitary_error,
            "lowered_one_qubit_gates": sum(len(gate.sites) == 1 for gate in gates),
            "lowered_two_qubit_gates": sum(len(gate.sites) == 2 for gate in gates),
            "mapping": mapping,
        },
    )


def synthesize(target: NDArray[np.complex128]) -> dict[str, Any]:
    """Return an authenticated-serializable physical rank-two preparation candidate."""
    cpu, wall = time.process_time(), time.perf_counter()
    approximation, compression = rank_two_approximation(target)
    local, preparation = rank_two_preparation_gates(approximation)
    circuit, theta, lowering = lower_local_unitaries(_qubit_count(approximation), local)
    return {
        "version": VERSION,
        "circuit": circuit_record(circuit),
        "theta": theta.tolist(),
        "approximation": [[float(value.real), float(value.imag)] for value in approximation],
        "compression": compression,
        "preparation": preparation,
        "lowering": lowering,
        "construction_cpu_seconds": time.process_time() - cpu,
        "construction_wall_seconds": time.perf_counter() - wall,
    }
