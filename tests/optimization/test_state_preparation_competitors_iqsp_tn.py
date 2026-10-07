# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent path, reachability, logical-transition and training checks."""

from __future__ import annotations

import json

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, Statevector, partial_trace

pytest.importorskip("torch")

import itertools

from benchmarks.state_preparation.constants import TARGET_IDS
from benchmarks.state_preparation.targets import load_target_collection
from experiments.state_preparation_competitors import iqsp_tn as method
from experiments.state_preparation_competitors.iqsp_tn_backend import TensorNetworkObjective
from experiments.state_preparation_competitors.logical_evaluation import score_exact


@pytest.mark.parametrize("target_id", TARGET_IDS)
def test_paths_preserve_every_fixture_and_relative_complex_phases(target_id: str) -> None:
    """All nine paths start uniformly and end at the true target up to phase."""
    target = load_target_collection().load(6, target_id).state_vector_copy()
    original = target.copy()
    path = [method.interpolation_target(target, step / 20, target_id=target_id) for step in range(21)]
    np.testing.assert_allclose(path[0], np.ones(64) / 8, atol=1e-14)
    assert abs(np.vdot(path[-1], target)) ** 2 == pytest.approx(1, abs=1e-13)
    assert max(abs(np.linalg.norm(state) - 1) for state in path) < 1e-13
    assert max(np.linalg.norm(right - left) for left, right in itertools.pairwise(path)) < 0.15
    shifted = method.interpolation_target(target * np.exp(0.73j), 0.35, target_id=target_id)
    np.testing.assert_allclose(shifted, path[7], atol=1e-13)
    np.testing.assert_array_equal(target, original)


def test_gaussian_path_matches_analytic_amplitudes_with_fixture_bit_order() -> None:
    """Gaussian interpolation is the paper's function family, not a geodesic."""
    target = load_target_collection().load(6, method.GAUSSIAN_TARGET_ID).state_vector_copy()
    indices = np.arange(64)
    x = sum(((indices >> q) & 1) * 2.0 ** (-q - 1) for q in range(6))
    expected = np.exp(-0.4 * (x - 0.5) ** 2 / 0.04)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(method.interpolation_target(target, 0.4, target_id=method.GAUSSIAN_TARGET_ID), expected)
    with pytest.raises(ValueError, match="requires gaussian"):
        method.interpolation_target(np.ones(64) / 8, 0.4, target_id=method.GAUSSIAN_TARGET_ID)


def test_geodesic_uniform_orthogonal_and_near_uniform_limits() -> None:
    """Degenerate path endpoints do not divide by zero or erase complex phases."""
    uniform = np.ones(4, dtype=complex) / 2
    orthogonal = np.array([1, -1, 1j, -1j]) / 2
    state = method.interpolation_target(orthogonal, 0.5, target_id="complex_test")
    np.testing.assert_allclose(state, (uniform + orthogonal) / np.sqrt(2), atol=1e-14)
    for delta in (0, 1e-9):
        target = uniform + delta * orthogonal
        target /= np.linalg.norm(target)
        actual = method.interpolation_target(target, 0.5, target_id="near_uniform")
        assert np.all(np.isfinite(actual))
        assert abs(np.vdot(actual, uniform)) ** 2 > 1 - 1e-12


@pytest.mark.parametrize("coordinate", [-0.1, 1.1, np.nan, np.inf, True, "0.5"])
def test_invalid_path_coordinates_fail(coordinate: float) -> None:
    """Malformed coordinates cannot silently change the continuation schedule."""
    with pytest.raises(ValueError, match="coordinate"):
        method.interpolation_target(np.ones(4) / 2, coordinate, target_id="test")


@pytest.mark.parametrize("target_id", ["", None, 4])
def test_invalid_path_labels_fail(target_id: str) -> None:
    """A reproducible interpolation requires an explicit target label."""
    with pytest.raises(ValueError, match="identifier"):
        method.interpolation_target(np.ones(4) / 2, 0.5, target_id=target_id)


@pytest.mark.parametrize(("qubits", "depth"), [(1, 1), (13, 1), (True, 1), (2, 0), (2, -1), (2, True)])
def test_ansatz_invalid_dimensions_fail(qubits: int, depth: int) -> None:
    """Only the declared nonempty neighboring-qubit brickwork is supported."""
    with pytest.raises(ValueError, match=r"(depth|qubits)"):
        method.iqsp_tn_ansatz(qubits, depth)


def test_ansatz_register_order_and_identity_are_correct() -> None:
    """The complete zero-coordinate circuit prepares the uniform state."""
    circuit = method.iqsp_tn_ansatz(6, 2)
    pairs = [tuple(circuit.find_bit(q).index for q in gate.qubits) for gate in circuit.data if len(gate.qubits) == 2]
    assert pairs == [(0, 1), (2, 3), (4, 5), (1, 2), (3, 4)] * 2
    assert len(circuit.parameters) == 150
    objective = TensorNetworkObjective(circuit, np.ones(64) / 8)
    bound = objective.bind(np.zeros(150))
    np.testing.assert_allclose(Statevector.from_instruction(bound).data, np.ones(64) / 8, atol=1e-13)


def test_origin_gradient_can_create_entanglement_and_adam_reaches_bell_state() -> None:
    """Regression: zero initialization must not trap IQSP in product states."""
    target = np.array([1, 0, 0, 1]) / np.sqrt(2)
    objective = TensorNetworkObjective(method.iqsp_tn_ansatz(2, 1), target)
    initial = np.zeros(15)
    loss, gradient = objective.loss_gradient(initial)
    two_qubit_coordinates = [index - 1 for index in range(1, 16) if index // 4 and index % 4]
    assert np.linalg.norm(gradient[two_qubit_coordinates]) > 0.1
    assert loss == pytest.approx(0.5, abs=1e-12)
    theta, receipt = method.optimize_tn_adam(objective, initial, steps=80)
    state = Statevector.from_instruction(objective.bind(theta))
    assert abs(np.vdot(target, state.data)) ** 2 > 0.99
    marginal = partial_trace(state, [1]).data
    assert np.trace(marginal @ marginal).real < 0.52
    assert receipt["final_fidelity"] > 0.99
    np.testing.assert_array_equal(initial, np.zeros(15))


@pytest.mark.parametrize("qubits", [2, 6])
def test_logical_transition_preserves_complex_unitary_and_all_windows(qubits: int) -> None:
    """Weyl lowering changes coordinates, not the ideal learned circuit."""
    original = method.iqsp_tn_ansatz(qubits, 1)
    objective = TensorNetworkObjective(original, np.ones(2**qubits) / np.sqrt(2**qubits))
    angles = np.random.default_rng(45).normal(scale=0.3, size=len(original.parameters))
    bound = objective.bind(angles)
    template, initial, receipt = method.logical_parameterization(bound)
    actual = template.assign_parameters(initial)
    np.testing.assert_allclose(Operator(actual).data, Operator(bound).data, atol=1e-12)
    assert set(actual.count_ops()) == {"h", "ry", "rz", "rxx", "ryy", "rzz"}
    assert len(initial) == 15 * (qubits - 1)
    assert len(actual.data) == qubits + 15 * (qubits - 1)
    assert receipt["maximum_local_unitary_error"] < 1e-12
    assert receipt["global_unitary_error"] < 1e-12
    zero = objective.bind(np.zeros(len(angles)))
    zero_template, zero_initial, _ = method.logical_parameterization(zero)
    assert len(zero_template.data) == len(actual.data)
    assert sum(abs(angle) < 1e-14 for angle in zero_initial) >= 3 * (qubits - 1)


def test_logical_transition_rejects_unexpected_or_unbound_gates() -> None:
    """Unsupported gates cannot acquire an undefined logical noise window."""
    for circuit in (QuantumCircuit(2), method.iqsp_tn_ansatz(2, 1), QuantumCircuit(1)):
        with pytest.raises(ValueError, match=r"(resolved|blocks)"):
            method.logical_parameterization(circuit)
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    with pytest.raises(ValueError, match="Hadamards"):
        method.logical_parameterization(circuit)


def test_small_angle_cutoff_is_signed_inclusive_two_qubit_only_and_nonmutating() -> None:
    """Paper cutoff removes only tiny two-qubit rotations, with no repair."""
    circuit = QuantumCircuit(2)
    circuit.global_phase = 0.37
    circuit.h(0)
    circuit.rxx(0, 0, 1)
    circuit.ryy(1e-4, 0, 1)
    circuit.rzz(-1e-4, 0, 1)
    circuit.rzz(-1.01e-4, 0, 1)
    circuit.ry(0, 0)
    circuit.cx(0, 1)
    original = circuit.copy()
    result, receipt = method.remove_small_two_qubit_rotations(circuit)
    assert circuit == original
    assert [item.operation.name for item in result.data] == ["h", "rzz", "ry", "cx"]
    assert result.global_phase == circuit.global_phase
    assert [item["original_index"] for item in receipt["removed"]] == [1, 2, 3]
    assert [item["angle"] for item in receipt["removed"]] == [0, 1e-4, -1e-4]
    assert receipt["input_windows"] - receipt["output_windows"] == 3
    assert receipt["additional_optimization_steps"] == 0
    second, repeated = method.remove_small_two_qubit_rotations(result)
    assert second == result
    assert repeated["removed"] == []


def test_zero_angle_cutoff_changes_noise_windows_but_preserves_ideal_state() -> None:
    """Deployment uses the reduced schedule, independently checked with density matrices."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.rxx(0, 0, 1)
    circuit.ryy(0, 0, 1)
    target = Statevector.from_instruction(circuit).data
    result, _ = method.remove_small_two_qubit_rotations(circuit)
    np.testing.assert_allclose(Operator(result).data, Operator(circuit).data, atol=1e-14)
    before = TensorNetworkObjective(circuit, target, hardware="IBM").fidelity([])
    after = TensorNetworkObjective(result, target, hardware="IBM").fidelity([])
    assert after > before
    assert after == pytest.approx(score_exact(result, target, "IBM")["fidelity"], abs=1e-13)


def test_small_angle_cutoff_rejects_unbound_circuit() -> None:
    """Do not accidentally delete a gate whose final angle is unresolved."""
    circuit = QuantumCircuit(2)
    circuit.rxx(Parameter("angle"), 0, 1)
    with pytest.raises(ValueError, match=r"(resolved|bound|parameter)"):
        method.remove_small_two_qubit_rotations(circuit)


def test_tn_adam_matches_independent_first_update() -> None:
    """The TN optimizer retains the existing Adam rule rather than a new policy."""
    circuit = QuantumCircuit(1)
    circuit.ry(Parameter("angle"), 0)
    target_angle = 1.1
    objective = TensorNetworkObjective(circuit, [np.cos(target_angle / 2), np.sin(target_angle / 2)])
    theta, receipt = method.optimize_tn_adam(objective, [0.2], steps=1)
    gradient = np.sin(0.2 - target_angle) / 2
    expected = 0.2 - 0.01 * gradient / (abs(gradient) + 1e-8)
    assert theta[0] == pytest.approx(expected, abs=1e-13)
    assert receipt["final_fidelity"] == pytest.approx(np.cos((expected - target_angle) / 2) ** 2, abs=1e-13)
    assert len(receipt["records"]) == 2
    assert receipt["best_iteration"] == 1
    json.dumps(receipt, allow_nan=False)


def test_tn_adam_returns_final_iterate_and_detects_invalid_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Policy parity: best checkpoints are diagnostic, not a substituted optimizer."""
    circuit = QuantumCircuit(1)
    circuit.ry(Parameter("angle"), 0)
    objective = TensorNetworkObjective(circuit, [1, 0])
    values = iter([(0.1, np.array([1.0])), (0.2, np.array([1.0]))])
    monkeypatch.setattr(objective, "loss_gradient", lambda _theta: next(values))
    theta, receipt = method.optimize_tn_adam(objective, [0.2], steps=1)
    assert theta[0] < 0.2
    assert receipt["best_theta"] == [0.2]
    assert receipt["final_fidelity"] < receipt["best_fidelity"]
    for broken in [(np.nan, np.array([0.0])), (0.1, np.array([np.nan])), (0.1, np.zeros(2))]:
        monkeypatch.setattr(objective, "loss_gradient", lambda _theta, result=broken: result)
        with pytest.raises(FloatingPointError):
            method.optimize_tn_adam(objective, [0.2], steps=0)


@pytest.mark.parametrize("rate", [0, -0.01, np.inf, np.nan, True])
def test_tn_adam_rejects_invalid_rate(rate: float) -> None:
    """Invalid learning rates fail before any expensive TN evaluation."""
    circuit = QuantumCircuit(1)
    circuit.ry(Parameter("angle"), 0)
    with pytest.raises(ValueError, match="learning rate"):
        method.optimize_tn_adam(TensorNetworkObjective(circuit, [1, 0]), [0.1], steps=1, learning_rate=rate)


def test_candidate_warm_starts_phases_and_matches_independent_noisy_density() -> None:
    """A tiny complete run covers real training, logical lowering and noise parity."""
    target = np.array([1, 1j, -0.3, 0.2j])
    target /= np.linalg.norm(target)
    result = method.iqsp_tn_candidates(
        target, target_id="complex_test", depths=(1,), interpolation_steps=2, construction_steps=2, noisy_steps=2
    )
    assert len(result) == 1
    candidate = result[0]
    receipt = candidate.metadata
    stages = receipt["stages"]
    assert stages[0]["optimization"]["final_theta"] == stages[1]["optimization"]["initial_theta"]
    assert receipt["interpolation_scope"] == "generalized_complex_geodesic"
    assert receipt["method"] == "iqsp_tn"
    assert receipt["noise_basis"] == "logical"
    assert receipt["noisy"]["steps"] == 2
    exact = score_exact(candidate.circuit, target, "IBM")
    assert receipt["postprocessing"]["final_noisy_fidelity"] == pytest.approx(exact["fidelity"], abs=1e-11)
    assert receipt["cpu_seconds"] >= sum(stage["optimization"]["cpu_seconds"] for stage in stages)
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("target_id", TARGET_IDS)
def test_all_nine_targets_have_finite_tn_training_and_correct_logical_noise(target_id: str) -> None:
    """One update per phase tests coverage, not benchmark performance."""
    target = load_target_collection().load(6, target_id).state_vector_copy()
    candidate = method.iqsp_tn_candidates(
        target, target_id=target_id, depths=(1,), interpolation_steps=1, construction_steps=1, noisy_steps=1
    )[0]
    receipt = candidate.metadata
    assert receipt["status"] == "complete"
    assert receipt["postprocessing"]["final_noisy_fidelity"] == pytest.approx(
        score_exact(candidate.circuit, target, "IBM")["fidelity"], abs=1e-10
    )
    assert receipt["interpolation_scope"] == (
        "published_gaussian_path" if target_id == method.GAUSSIAN_TARGET_ID else "generalized_complex_geodesic"
    )


@pytest.mark.parametrize(
    "options",
    [
        {"depths": ()},
        {"depths": (1, 1)},
        {"depths": (0,)},
        {"interpolation_steps": 0},
        {"construction_steps": -1},
        {"noisy_steps": True},
        {"learning_rate": 0},
        {"hardware": "unknown"},
    ],
)
def test_candidate_invalid_protocol_fails_before_training(options: dict) -> None:
    """A malformed protocol must not emit a completed candidate."""
    with pytest.raises(ValueError, match=r"(Require|depth|steps|learning rate|profile)"):
        method.iqsp_tn_candidates(np.ones(4) / 2, target_id="test", **options)
