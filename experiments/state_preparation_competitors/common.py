# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Small circuit boundary shared by independent state-preparation implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RXXGate, RYGate, RYYGate, RZGate, RZZGate

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

ROTATIONS = {"rx": RXGate, "ry": RYGate, "rz": RZGate, "rxx": RXXGate, "ryy": RYYGate, "rzz": RZZGate}


@dataclass(frozen=True)
class Candidate:
    """One preparation circuit and its method-specific construction receipt.

    Circuits use Qiskit's little-endian state-vector convention. Metadata must
    contain only finite JSON values when the experiment persists a candidate.
    Failed construction/training candidates carry an explicit metadata status;
    a failure is never represented by a fabricated zero fidelity.
    """

    circuit: QuantumCircuit
    metadata: dict[str, Any]


def validate_target(target: ArrayLike, *, max_qubits: int = 6) -> NDArray[np.complex128]:
    """Return a private normalized complex state vector for one to the explicitly allowed number of qubits.

    Args:
        target: Amplitudes in little-endian computational-basis order.
        max_qubits: Explicit backend limit; legacy callers remain capped at six.

    Returns:
        A copy, preserving amplitudes and global phase without renormalization.

    Raises:
        ValueError: If the input is nonfinite, nonnormalized, or outside the declared register limit.
    """
    if type(max_qubits) is not int or not 1 <= max_qubits <= 12:
        msg = "max_qubits must be an integer from one to twelve."
        raise ValueError(msg)
    value = np.asarray(target, dtype=np.complex128)
    if (
        value.ndim != 1
        or not 2 <= value.size <= 2**max_qubits
        or value.size & (value.size - 1)
        or not np.all(np.isfinite(value))
        or abs(np.vdot(value, value) - 1) > 1e-10
    ):
        msg = f"Expected a normalized finite state vector for one to {max_qubits} qubits."
        raise ValueError(msg)
    return value.copy()


def from_parameterized(circuit: ParameterizedCircuit, theta: ArrayLike) -> QuantumCircuit:
    """Materialize the existing primitive-rotation ansatz without simplification.

    Args:
        circuit: Existing YAQS circuit made of scalar Pauli rotations.
        theta: Resolved finite real parameters, including shared parameters.

    Returns:
        A circuit preserving every primitive gate and its ordered qubit support.

    Raises:
        ValueError: If a parameter or gate cannot be represented faithfully.
    """
    values = np.asarray(theta)
    if (
        values.shape != (circuit.num_params,)
        or not np.isrealobj(values)
        or values.dtype.kind not in "fiu"
        or not np.all(np.isfinite(values))
        or not 1 <= circuit.num_qubits <= 6
    ):
        msg = "Expected finite real parameters matching a q1-q6 circuit."
        raise ValueError(msg)
    result = QuantumCircuit(circuit.num_qubits)
    for gate in circuit.gates:
        if gate.name not in ROTATIONS:
            msg = f"Unsupported primitive rotation: {gate.name}."
            raise ValueError(msg)
        angle = gate.angle_offset
        if gate.param_index is not None:
            angle += gate.angle_scale * values[gate.param_index]
        if not np.isfinite(angle):
            msg = "Resolved gate angles must be finite."
            raise ValueError(msg)
        result.append(ROTATIONS[gate.name](float(angle)), list(gate.sites))
    return result
