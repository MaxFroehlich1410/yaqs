# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Adapt Qiskit's exact three-CX synthesis for near-symmetric opaque gates.

Qiskit 2.4.1's ordinary SU(4) synthesis specializes the Weyl decomposition at
average fidelity 1 - 1e-9 even with approximation_degree=1. This may discard
small but resolved rotations. The adapter explicitly disables specialization;
it reuses Qiskit's factorization and only assembles the returned local factors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CXGate, UnitaryGate
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer, TwoQubitWeylDecomposition

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike


def exact_two_qubit_gate(matrix: ArrayLike) -> QuantumCircuit:
    """Use Qiskit's unspecialized Weyl factors to assemble an exact three-CX gate.

    Args:
        matrix: Finite unitary four-by-four matrix in Qiskit wire order.

    Returns:
        Circuit equal to the input including its global phase, with three CXs.

    Raises:
        ValueError: If the input is not a finite four-by-four unitary.
        RuntimeError: If Qiskit's pinned bridge changes or synthesis is inaccurate.
    """
    value = np.asarray(matrix, dtype=np.complex128)
    if (
        value.shape != (4, 4)
        or not np.all(np.isfinite(value))
        or not np.allclose(value.conj().T @ value, np.eye(4), atol=1e-10, rtol=0)
    ):
        msg = "Exact two-qubit synthesis requires a finite four-by-four unitary."
        raise ValueError(msg)
    weyl = TwoQubitWeylDecomposition(value, fidelity=None)
    # Qiskit 2.4.1's public decomp3 method expects its Rust object, not the public
    # Python wrapper. Guard this one private bridge rather than reimplement KAK.
    inner = getattr(weyl, "_inner_decomposition", None)
    if inner is None:
        msg = "Pinned Qiskit Weyl bridge is unavailable; review exact synthesis before upgrading."
        raise RuntimeError(msg)
    factors = TwoQubitBasisDecomposer(CXGate()).decomp3_supercontrolled(inner)
    circuit = QuantumCircuit(2)
    for stage in range(4):
        circuit.unitary(factors[2 * stage], [0])
        circuit.unitary(factors[2 * stage + 1], [1])
        if stage < 3:
            circuit.cx(0, 1)
    overlap = np.vdot(value, Operator(circuit).data)
    circuit.global_phase = -float(np.angle(overlap))
    if np.max(abs(Operator(circuit).data - value)) > 1e-10:
        msg = "Qiskit's exact three-CX synthesis failed its independent matrix check."
        raise RuntimeError(msg)
    return circuit


def repair_opaque_two_qubit_gates(source: QuantumCircuit, compiler: Mapping[str, Any]) -> tuple[QuantumCircuit, int]:
    """Replace only inaccurate opaque two-qubit syntheses by exact Qiskit factors.

    Call only after the full compilation fails its ideal-unitary audit. Normal
    accurate gates retain the ordinary compilation path. The local 1e-12 check
    is deliberately stricter than the unchanged full-circuit 1e-9 acceptance.

    Args:
        source: Original logical circuit without a previous transpiler layout.
        compiler: The same basis, optimization, approximation and seed settings.

    Returns:
        Equivalent source circuit and the number of replaced opaque gates.
    """
    result = source.copy_empty_like()
    replacements = 0
    for instruction in source.data:
        sites = [source.find_bit(qubit).index for qubit in instruction.qubits]
        operation = instruction.operation
        if isinstance(operation, UnitaryGate) and operation.num_qubits == 2:
            local = QuantumCircuit(2)
            local.append(operation, [0, 1])
            compiled = transpile(local, **compiler)
            expected, actual = Operator(local).data, Operator(compiled).data
            overlap = np.vdot(expected, actual)
            phase = overlap / abs(overlap) if abs(overlap) else 1
            if np.max(abs(actual - phase * expected)) > 1e-12:
                result.compose(exact_two_qubit_gate(expected), qubits=sites, inplace=True)
                replacements += 1
                continue
        result.append(operation, sites)
    return result, replacements
