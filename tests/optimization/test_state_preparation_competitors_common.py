# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent checks of the common state-preparation circuit boundary."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from experiments.state_preparation_competitors.common import Candidate, from_parameterized, validate_target
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate, StatevectorBackend


@pytest.mark.parametrize("qubits", range(1, 7))
def test_target_copy_preserves_complex_amplitudes_and_phase(qubits: int) -> None:
    """Adapters accept the full complex space without using magnitudes as amplitudes."""
    rng = np.random.default_rng(qubits)
    target = rng.normal(size=2**qubits) + 1j * rng.normal(size=2**qubits)
    target /= np.linalg.norm(target)
    actual = validate_target(target)
    np.testing.assert_array_equal(actual, target)
    assert actual.dtype == np.complex128
    assert not np.shares_memory(actual, target)


@pytest.mark.parametrize(
    "target", [[], [1], [1, 0, 0], [0, 0], [2, 0], [np.nan, 0], [np.inf, 0], [[1, 0]], [1] + [0] * 127]
)
def test_target_rejects_invalid_inputs(target: list) -> None:
    """Invalid inputs must not be silently truncated or renormalized."""
    with pytest.raises(ValueError, match="state vector"):
        validate_target(target)


def test_materialization_preserves_sites_shared_parameters_and_offsets() -> None:
    """Qiskit and the independent dense backend agree on complex, nonsymmetric states."""
    gates = [
        ParameterizedGate("rx", (0,), 0),
        ParameterizedGate("ry", (2,), 1, angle_scale=-0.7, angle_offset=0.4),
        ParameterizedGate("rz", (1,), None, angle_offset=0.6),
        ParameterizedGate("rxx", (2, 0), 2),
        ParameterizedGate("ryy", (1, 2), 0),
        ParameterizedGate("rzz", (0, 1), 3),
    ]
    circuit = ParameterizedCircuit(3, gates, 4)
    theta = np.array([0.3, -0.5, 0.7, 1.1])
    compiled = from_parameterized(circuit, theta)
    target = Statevector.from_instruction(compiled).data
    assert len(compiled.data) == len(gates)
    assert StatevectorBackend(circuit, target).metrics(theta)[1] == pytest.approx(1, abs=1e-13)
    candidate = Candidate(compiled, {"status": "complete"})
    assert candidate.metadata["status"] == "complete"
    assert candidate.circuit is compiled


@pytest.mark.parametrize("theta", [[], [np.inf], [0.1j], [[0.1]]])
def test_materialization_rejects_invalid_parameters(theta: list) -> None:
    """The adapter fails closed instead of dropping imaginary or missing parameters."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), 0)], 1)
    with pytest.raises(ValueError, match="finite real parameters"):
        from_parameterized(circuit, theta)


def test_materialization_empty_circuit_and_unsupported_gate() -> None:
    """An empty preparation is valid; unsupported gates cannot disappear."""
    circuit = ParameterizedCircuit(2, [], 0)
    assert from_parameterized(circuit, np.empty(0)) == QuantumCircuit(2)
    unsupported = ParameterizedCircuit(2, [ParameterizedGate("cx", (0, 1), None)], 0)
    with pytest.raises(ValueError, match="Unsupported primitive"):
        from_parameterized(unsupported, np.empty(0))
