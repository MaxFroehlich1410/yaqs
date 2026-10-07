# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Full-unitary lowering and physical-zero rank-two preparation checks."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider, exact_density_fidelity
from benchmarks.state_preparation.targets import load_target
from experiments.overnight_state_preparation_20260913_0108 import synthesis
from experiments.state_preparation_pilot import restore_circuit
from mqt.yaqs.optimization import ParameterizedCircuit

if TYPE_CHECKING:
    from numpy.typing import NDArray


def random_vector(n: int, seed: int) -> NDArray[np.complex128]:
    """Return a deterministic normalized engineering target."""
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return vector / np.linalg.norm(vector)


def physical_qiskit(circuit: ParameterizedCircuit, theta: NDArray[np.float64]) -> QuantumCircuit:
    """Return an independent Qiskit circuit of only the emitted primitives."""
    reference = QuantumCircuit(circuit.num_qubits)
    for gate in circuit.gates:
        angle = circuit.angle(gate, theta, None)
        getattr(reference, gate.name)(angle, *gate.sites)
    return reference


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("seed", [1729, 29])
def test_low_qubit_preparation_exact_from_physical_zero(n: int, seed: int) -> None:
    """All q<=3 pure states have bond dimension at most two on this ordering."""
    target = random_vector(n, seed)
    before = target.copy()
    result = synthesis.synthesize(target)
    circuit = restore_circuit(result["circuit"])
    theta = np.asarray(result["theta"])
    output = Statevector.from_instruction(physical_qiskit(circuit, theta)).data
    assert abs(np.vdot(target, output)) ** 2 == pytest.approx(1, abs=1e-12)
    assert result["compression"]["approximation_fidelity"] == pytest.approx(1, abs=1e-12)
    assert result["preparation"]["disentangled_zero_fidelity"] == pytest.approx(1, abs=1e-12)
    assert result["preparation"]["logical_one_qubit_gates"] == 1
    assert result["preparation"]["logical_two_qubit_gates"] == n - 1
    assert result["lowering"]["lowered_one_qubit_gates"] == 3 + 12 * (n - 1)
    assert result["lowering"]["lowered_two_qubit_gates"] == 3 * (n - 1)
    assert all(gate.noise_enabled and gate.name in synthesis.PRIMITIVES for gate in circuit.gates)
    np.testing.assert_array_equal(target, before)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("sites", [(0, 1), (1, 0), (2, 0)])
def test_weyl_lowering_matches_entire_unitary_on_reversed_nonadjacent_sites(sites: tuple[int, int]) -> None:
    """The compiler reproduces all input states, not only the prepared vector."""
    rng = np.random.default_rng(19)
    matrix, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    n = max(sites) + 1
    circuit, theta, metadata = synthesis.lower_local_unitaries(n, [(matrix, sites)])
    reference = QuantumCircuit(n)
    reference.unitary(matrix, list(reversed(sites)))
    actual = np.exp(1j * metadata["omitted_global_phase"]) * Operator(physical_qiskit(circuit, theta)).data
    np.testing.assert_allclose(actual, Operator(reference).data, atol=1e-12, rtol=0)
    assert circuit.num_params == 15
    assert all(gate.noise_enabled for gate in circuit.gates)


def test_rank_four_target_reports_truncation_and_retains_rank_two_on_every_cut() -> None:
    """Two crossing Bell pairs require rank four and lose exactly half their norm."""
    target = np.zeros(16, dtype=complex)
    target[[0, 5, 10, 15]] = 0.5
    vector, detail = synthesis.rank_two_approximation(target)
    assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-13)
    assert detail["retained_norm_squared"] == pytest.approx(0.5, abs=1e-13)
    assert detail["total_discarded_weight"] == pytest.approx(0.5, abs=1e-13)
    assert detail["approximation_fidelity"] == pytest.approx(0.5, abs=1e-13)
    physical = vector.reshape((2,) * 4).transpose(3, 2, 1, 0)
    for cut in range(1, 4):
        singular = np.linalg.svd(physical.reshape(2**cut, -1), compute_uv=False)
        assert np.sum(singular[2:] ** 2) < 1e-24
    with pytest.raises(ValueError, match="rank at most two"):
        synthesis.rank_two_preparation_gates(target)
    local, _ = synthesis.rank_two_preparation_gates(vector)
    circuit, theta, _ = synthesis.lower_local_unitaries(4, local)
    output = Statevector.from_instruction(physical_qiskit(circuit, theta)).data
    assert abs(np.vdot(vector, output)) ** 2 == pytest.approx(1, abs=1e-12)


def test_identity_rotations_remain_counted_with_enabled_noise() -> None:
    """A compiler identity does not become a free gate-window deletion."""
    circuit, theta, metadata = synthesis.lower_local_unitaries(2, [(np.eye(4, dtype=complex), (0, 1))])
    assert metadata["compiler_identity_tolerance"] == 0
    assert metadata["compiler_simplification"] is False
    assert circuit.num_params == 15
    np.testing.assert_allclose(Operator(physical_qiskit(circuit, theta)).data, np.eye(4), atol=1e-13)
    assert all(gate.noise_enabled for gate in circuit.gates)


def test_single_qubit_lowering_gates_receive_actual_standard_noise() -> None:
    """The physical noisy fidelity includes all lowering rotations."""
    target = random_vector(2, 19)
    result = synthesis.synthesize(target)
    circuit, theta = restore_circuit(result["circuit"]), np.asarray(result["theta"])
    actual = exact_density_fidelity(circuit, theta, target, ExactLocalPauliProvider())
    artificial = ParameterizedCircuit(
        2,
        [replace(gate, noise_enabled=len(gate.sites) == 2) for gate in circuit.gates],
        circuit.num_params,
    )
    falsely_free = exact_density_fidelity(artificial, theta, target, ExactLocalPauliProvider())
    assert falsely_free > actual + 1e-4
    assert 0 < actual < 1


def test_q6_bond_two_fixture_exact_ideal_preparation() -> None:
    """Check the class guarantee on the fixed bond-two vector without noisy screening."""
    target = load_target(6, "random_mps_bond2").state_vector_copy()
    result = synthesis.synthesize(target)
    circuit, theta = restore_circuit(result["circuit"]), np.asarray(result["theta"])
    output = Statevector.from_instruction(physical_qiskit(circuit, theta)).data
    assert abs(np.vdot(target, output)) ** 2 == pytest.approx(1, abs=1e-10)
    assert result["compression"]["total_discarded_weight"] < 1e-24
    assert result["lowering"]["lowered_two_qubit_gates"] == 15
    assert result["lowering"]["lowered_one_qubit_gates"] == 63
    assert all(gate.name in synthesis.PRIMITIVES and gate.noise_enabled for gate in circuit.gates)


def test_repeated_construction_same_environment_is_deterministic() -> None:
    """SVD phase fixing and Weyl lowering yield repeatable circuit parameters."""
    target = random_vector(3, 117)
    first, second = synthesis.synthesize(target), synthesis.synthesize(target)
    for key in ("version", "circuit", "theta", "approximation", "compression", "preparation", "lowering"):
        assert first[key] == second[key]


@pytest.mark.parametrize(
    "target",
    [
        np.zeros(4, dtype=complex),
        np.ones(4, dtype=complex),
        np.array([np.nan, 0], dtype=complex),
        np.ones(3, dtype=complex) / np.sqrt(3),
        np.ones(128, dtype=complex) / np.sqrt(128),
        np.eye(2, dtype=complex),
    ],
)
def test_invalid_target_rejected(target: NDArray[np.complex128]) -> None:
    """Reject malformed state fixtures without silently normalizing them."""
    with pytest.raises(ValueError, match="normalized finite q1-q6 vector"):
        synthesis.synthesize(target)


@pytest.mark.parametrize(
    ("matrix", "sites"),
    [
        (np.eye(4) * 2, (0, 1)),
        (np.eye(8), (0, 1, 2)),
        (np.eye(4), (0, 0)),
        (np.eye(4), (0, 3)),
        (np.eye(2), (0, 1)),
    ],
)
def test_invalid_local_unitary_rejected(matrix: NDArray[np.complex128], sites: tuple[int, ...]) -> None:
    """No arbitrary-size, nonunitary, or malformed gate reaches evaluation."""
    with pytest.raises(ValueError, match="valid one-/two-qubit unitary"):
        synthesis.lower_local_unitaries(3, [(matrix, sites)])
