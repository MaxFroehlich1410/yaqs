# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Regression checks for exact compilation of nearly symmetric opaque gates."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, RXXGate, RYYGate, RZZGate
from qiskit.quantum_info import Operator, Statevector, random_unitary

from experiments.state_preparation_competitors import evaluation, exact_synthesis
from experiments.state_preparation_competitors.common import Candidate
from experiments.state_preparation_competitors.evaluation import COMPILER, compile_candidate
from experiments.state_preparation_competitors.exact_synthesis import (
    exact_two_qubit_gate,
    repair_opaque_two_qubit_gates,
)

if TYPE_CHECKING:
    from qiskit.circuit import Gate


@pytest.mark.parametrize("seed", [19, 23, 101])
@pytest.mark.parametrize("phase", [-2.7, 0.0, 2.3])
def test_exact_three_cx_matches_complex_unitary_and_phase(seed: int, phase: float) -> None:
    """Reconstruct independent complex matrices including their global phase."""
    expected = np.exp(1j * phase) * random_unitary(4, seed=seed).data
    circuit = exact_two_qubit_gate(expected)
    assert circuit.count_ops()["cx"] == 3
    np.testing.assert_allclose(Operator(circuit).data, expected, atol=2e-12, rtol=0)


@pytest.mark.parametrize("operation", [CXGate(), RXXGate(1e-5), RYYGate(1e-5), RZZGate(1e-5)])
def test_exact_three_cx_resolves_symmetric_and_nearly_symmetric_gates(operation: Gate) -> None:
    """Small resolved rotations survive the unspecialized decomposition."""
    expected = Operator(operation).data
    result = exact_two_qubit_gate(expected)
    assert result.count_ops()["cx"] == 3
    np.testing.assert_allclose(Operator(result).data, expected, atol=2e-12, rtol=0)


@pytest.mark.parametrize("matrix", [np.eye(2), np.zeros((4, 4)), np.full((4, 4), np.nan)])
def test_exact_three_cx_rejects_invalid_input(matrix: np.ndarray) -> None:
    """Nonunitary or nonfinite data cannot reach the synthesis backend."""
    with pytest.raises(ValueError, match="finite four-by-four unitary"):
        exact_two_qubit_gate(matrix)


def test_private_bridge_is_guarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed Qiskit private bridge fails explicitly rather than changing results."""
    monkeypatch.setattr(exact_synthesis, "TwoQubitWeylDecomposition", lambda *_a, **_k: SimpleNamespace())
    with pytest.raises(RuntimeError, match="Pinned Qiskit Weyl bridge"):
        exact_two_qubit_gate(np.eye(4))


def test_bad_factorization_fails_independent_matrix_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapter does not trust an incorrect backend decomposition."""
    monkeypatch.setattr(
        exact_synthesis,
        "TwoQubitBasisDecomposer",
        lambda *_a: SimpleNamespace(decomp3_supercontrolled=lambda _w: [np.eye(2)] * 8),
    )
    with pytest.raises(RuntimeError, match="independent matrix check"):
        exact_two_qubit_gate(np.eye(4))


def test_repair_preserves_accurate_gates_and_original_phase() -> None:
    """Only the opaque gate with a demonstrably inaccurate local synthesis changes."""
    source = QuantumCircuit(3, global_phase=1.3)
    source.h(0)
    source.cx(0, 1)
    source.unitary(random_unitary(4, seed=13), [2, 0])
    source.unitary(Operator(RXXGate(1e-5)), [1, 2])
    result, count = repair_opaque_two_qubit_gates(source, COMPILER)
    assert count == 1
    assert result.data[:3] == source.data[:3]
    np.testing.assert_allclose(Operator(result).data, Operator(source).data, atol=2e-12, rtol=0)
    assert len(source.data) == 4
    assert source.global_phase == pytest.approx(1.3)


def test_repair_does_not_change_an_accurate_circuit() -> None:
    """Ordinary instructions and accurate opaque matrices remain unchanged."""
    source = QuantumCircuit(2, global_phase=0.9)
    source.ry(0.3, 0)
    source.unitary(random_unitary(4, seed=17), [1, 0])
    result, count = repair_opaque_two_qubit_gates(source, COMPILER)
    assert count == 0
    assert source == result


@pytest.mark.parametrize("sites", [[0, 1], [5, 0]])
@pytest.mark.parametrize("rotation", [RXXGate, RYYGate, RZZGate])
def test_compiler_recovers_near_degenerate_opaque_gate(sites: list[int], rotation: type) -> None:
    """Native lowering remains accurate on reversed and routed six-qubit supports."""
    source = QuantumCircuit(6, global_phase=1.1)
    source.h(0)
    source.rz(0.37, 5)
    source.unitary(Operator(rotation(1e-5)), sites)
    target = Statevector.from_instruction(source).data
    result = compile_candidate(Candidate(source, {"method": "regression"}), target)
    assert result.metadata["exact_synthesis_fallback_count"] == 1
    assert result.metadata["initial_ideal_unitary_error"] > 1e-9
    assert result.metadata["ideal_unitary_error"] < 1e-9
    assert result.metadata["ideal_fidelity"] == pytest.approx(1.0, abs=2e-12)
    expected, actual = Operator(source).data, Operator.from_circuit(result.circuit).data
    phase = np.vdot(expected, actual)
    phase /= abs(phase)
    np.testing.assert_allclose(actual, phase * expected, atol=2e-12, rtol=0)


def test_accurate_compilation_does_not_invoke_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback leaves the previous compilation path intact when its audit passes."""

    def unexpected_repair(*_args: object) -> None:
        pytest.fail("Exact synthesis repair should not run for an accurate compile.")

    monkeypatch.setattr(evaluation, "repair_opaque_two_qubit_gates", unexpected_repair)
    source = QuantumCircuit(2)
    source.h(0)
    source.cx(0, 1)
    result = compile_candidate(Candidate(source, {}), Statevector.from_instruction(source).data)
    assert result.metadata["exact_synthesis_fallback_count"] == 0


def test_bad_repair_cannot_pass_audit_against_original_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fallback is independently compared with the untouched original circuit."""
    source = QuantumCircuit(2)
    source.unitary(Operator(RXXGate(1e-5)), [0, 1])
    wrong = QuantumCircuit(2)
    wrong.x(0)
    monkeypatch.setattr(evaluation, "repair_opaque_two_qubit_gates", lambda *_a: (wrong, 1))
    with pytest.raises(ValueError, match="independent ideal-unitary equivalence check"):
        compile_candidate(Candidate(source, {}), Statevector.from_instruction(source).data)
