# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent Gaussian-family, SU4-transition, and Adam tests for IQSP."""

from __future__ import annotations

import json

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import DensityMatrix, Operator, Statevector

from benchmarks.state_preparation.targets import load_target_collection
from experiments.legacy_topdown_dense import noise_channel
from experiments.state_preparation_competitors.iqsp import (
    compile_su4_endpoint,
    gaussian_state,
    iqsp_ansatz,
    iqsp_candidates,
    optimize_adam,
    validate_gaussian_target,
)
from experiments.state_preparation_competitors.noisy_core import DenseObjective


def test_gaussian_family_matches_fixture_and_quantically_ordered_analytic_values() -> None:
    """The path interpolates the Gaussian amplitude function, not arbitrary states."""
    fixture = load_target_collection().load(6, "gaussian_mu0p5_sigma0p1").state_vector_copy()
    np.testing.assert_allclose(gaussian_state(6, 1), fixture, atol=1e-15)
    np.testing.assert_allclose(gaussian_state(6, 0), np.ones(64) / 8, atol=1e-15)
    # For q2, little-endian bit indices encode x=(0,1/2,1/4,3/4).
    x = np.array([0, 0.5, 0.25, 0.75])
    expected = np.exp(-0.4 * (x - 0.5) ** 2 / 0.04)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(gaussian_state(2, 0.4), expected, atol=1e-15)
    assert np.linalg.norm(gaussian_state(6, 0.05) - gaussian_state(6, 0)) < 0.1


@pytest.mark.parametrize(
    ("qubits", "value"), [(0, 0), (13, 0), (True, 0), (2, -0.1), (2, 1.1), (2, np.nan), (2, True), (2, "0.1")]
)
def test_invalid_gaussian_domain_is_rejected(qubits: int, value: float) -> None:
    """Malformed path coordinates cannot silently extrapolate a different function."""
    with pytest.raises(ValueError, match="Require"):
        gaussian_state(qubits, value)


def test_gaussian_eligibility_accepts_only_matching_fixture_up_to_phase() -> None:
    """Neither a wrong family name nor a hidden relative phase can enter IQSP."""
    target = gaussian_state(6, 1)
    np.testing.assert_allclose(validate_gaussian_target(np.exp(0.4j) * target), target, atol=1e-15)
    with pytest.raises(ValueError, match="six-qubit"):
        validate_gaussian_target(gaussian_state(3, 1))
    for other in (gaussian_state(6, 0.5), target * np.exp(1j * np.linspace(0, 0.4, 64))):
        with pytest.raises(ValueError, match="requires gaussian"):
            validate_gaussian_target(other)
    collection = load_target_collection()
    for name in ("tfim_ferro", "tfim_para", "tfim_critical", "haar_random_1", "random_mps_bond2"):
        with pytest.raises(ValueError, match="requires gaussian"):
            validate_gaussian_target(collection.load(6, name).state_vector_copy())


def test_su4_brickwork_starts_exactly_uniform_and_has_expected_supports() -> None:
    """One-dimensional Fig.6 has H then even/odd neighboring SU4 blocks per layer."""
    for qubits in (2, 6):
        template = iqsp_ansatz(qubits, 2)
        assert len(template.parameters) == 30 * (qubits - 1)
        zero = template.assign_parameters(np.zeros(len(template.parameters)))
        np.testing.assert_allclose(
            Statevector.from_instruction(zero).data, np.ones(2**qubits) / np.sqrt(2**qubits), atol=1e-14
        )
        pairs = [
            tuple(template.find_bit(q).index for q in item.qubits)
            for item in template.data
            if item.operation.name == "rxx"
        ]
        expected = [(site, site + 1) for parity in (0, 1) for site in range(parity, qubits - 1, 2)]
        assert pairs == expected * 2
    generic = iqsp_ansatz(2, 1).assign_parameters(np.linspace(-0.8, 0.7, 15))
    assert Operator(generic).is_unitary()
    assert abs(np.linalg.det(Operator(generic).data) - 1) < 1e-12


@pytest.mark.parametrize(("qubits", "depth"), [(1, 1), (7, 1), (2, 0), (2, -1), (2, True)])
def test_iqsp_ansatz_rejects_invalid_dimensions(qubits: int, depth: int) -> None:
    """Unspecified one-qubit and empty-layer variants are not silently introduced."""
    with pytest.raises(ValueError, match=r"requires|nonnegative"):
        iqsp_ansatz(qubits, depth)


def test_su4_compilation_uses_three_cx_and_frees_native_angles_without_changing_unitary() -> None:
    """Hardware compilation occurs after learning full SU4 matrices, before noisy tuning."""
    theta = np.linspace(-0.63, 1.37, 15)
    source = iqsp_ansatz(2, 1).assign_parameters(theta)
    target = gaussian_state(2, 1)
    native, initial, receipt = compile_su4_endpoint(2, 1, theta, target)
    bound = native.assign_parameters(initial)
    expected, actual = Operator(source).data, Operator(bound).data
    phase = np.vdot(expected, actual) / abs(np.vdot(expected, actual))
    np.testing.assert_allclose(actual, phase * expected, atol=1e-12)
    assert bound.count_ops()["cx"] == 3
    assert len(native.parameters) == native.count_ops()["rz"]
    assert receipt["original_parameter_count"] == 15
    assert receipt["native_parameter_count"] == len(initial)
    assert receipt["grouping_unitary_error"] < 1e-12
    assert receipt["native_reparameterization_error"] < 1e-12
    assert set(bound.count_ops()).issubset({"rz", "sx", "x", "cx"})
    with pytest.raises(ValueError, match="parameter vector"):
        compile_su4_endpoint(2, 1, theta[:-1], target)
    with pytest.raises(ValueError, match="target"):
        compile_su4_endpoint(2, 1, theta, gaussian_state(3, 1))


def test_adam_step_agrees_with_closed_form_bias_correction() -> None:
    """The first actual update matches Adam's independent analytical scalar formula."""
    circuit = QuantumCircuit(1)
    circuit.ry(Parameter("theta"), 0)
    target_angle = 1.1
    target = [np.cos(target_angle / 2), np.sin(target_angle / 2)]
    objective = DenseObjective(circuit, target, None, compile_native=False)
    initial = np.array([0.2])
    theta, trace = optimize_adam(objective, initial, steps=1)
    gradient = np.sin(initial[0] - target_angle) / 2
    expected = initial[0] - 0.01 * gradient / (abs(gradient) + 1e-8)
    assert theta[0] == pytest.approx(expected, abs=1e-14)
    assert trace["returned_fidelity"] == pytest.approx(np.cos((expected - target_angle) / 2) ** 2, abs=1e-14)
    assert trace["objective_evaluations"] == trace["gradient_evaluations"] == 2
    assert len(trace["records"]) == 2
    np.testing.assert_array_equal(initial, [0.2])
    json.dumps(trace, allow_nan=False)


def test_adam_zero_updates_and_nonfinite_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero steps preserve parameters; a broken oracle is not a fake completed run."""
    circuit = QuantumCircuit(1)
    circuit.ry(Parameter("theta"), 0)
    objective = DenseObjective(circuit, [1, 0], None)
    actual, trace = optimize_adam(objective, [0.3], steps=0)
    np.testing.assert_array_equal(actual, [0.3])
    assert trace["objective_evaluations"] == 1
    monkeypatch.setattr(objective, "loss_gradient", lambda _theta: (0.2, np.array([np.nan])))
    with pytest.raises(FloatingPointError, match="must be finite"):
        optimize_adam(objective, [0.3], steps=1)


@pytest.mark.parametrize(
    "options", [{"steps": -1}, {"steps": 1, "learning_rate": 0}, {"steps": 1, "learning_rate": np.nan}]
)
def test_adam_rejects_invalid_options(options: dict) -> None:
    """Adam configuration is checked by the reused tested implementation."""
    circuit = QuantumCircuit(1)
    circuit.ry(Parameter("theta"), 0)
    objective = DenseObjective(circuit, [1, 0], None)
    with pytest.raises(ValueError, match=r"nonnegative|Adam configuration"):
        optimize_adam(objective, [0.0], **options)


def test_actual_six_qubit_path_warm_starts_then_trains_the_compiled_native_channel() -> None:
    """A small real run follows Gaussian stages and matches an independent noisy oracle."""
    target = gaussian_state(6, 1)
    candidate = iqsp_candidates(
        target, hardware="IBM", seed=300, depths=(1,), construction_steps=1, noisy_steps=1, interpolation_steps=2
    )[0]
    metadata = candidate.metadata
    assert metadata["interpolation"][0]["lambda"] == pytest.approx(0.5)
    assert metadata["interpolation"][1]["lambda"] == 1
    assert metadata["interpolation"][0]["final_theta"] == metadata["interpolation"][1]["initial_theta"]
    assert metadata["construction_theta"] == metadata["interpolation"][-1]["final_theta"]
    assert metadata["native_initial_theta"] == metadata["noisy_training"]["initial_theta"]
    assert metadata["native_transition"]["compiled"]["compiled_two_qubit"] <= 15
    assert metadata["preserve_native_schedule"] is True
    assert metadata["deterministic"] is True
    assert metadata["seed_used"] is False
    rho = DensityMatrix.from_label("000000")
    for item in candidate.circuit.data:
        sites = [candidate.circuit.find_bit(q).index for q in item.qubits]
        rho = rho.evolve(item.operation, qargs=sites).evolve(noise_channel("IBM", len(sites)), qargs=sites)
    expected = float(np.vdot(target, rho.data @ target).real)
    assert metadata["training_fidelity"] == pytest.approx(expected, abs=1e-12)
    assert metadata["cpu_seconds"] > 0
    json.dumps(metadata, allow_nan=False)


def test_identity_start_is_deterministic_and_does_not_invent_seed_replicas() -> None:
    """A zero-update path verifies that the declared seed is deliberately unused."""
    target = gaussian_state(6, 1)
    first = iqsp_candidates(
        target, hardware="Google", seed=300, depths=(1,), construction_steps=0, noisy_steps=0, interpolation_steps=1
    )[0]
    second = iqsp_candidates(
        target, hardware="Google", seed=400, depths=(1,), construction_steps=0, noisy_steps=0, interpolation_steps=1
    )[0]
    np.testing.assert_array_equal(Operator(first.circuit).data, Operator(second.circuit).data)
    assert first.metadata["theta"] == second.metadata["theta"]
    assert first.metadata["construction_steps_total"] == 0


@pytest.mark.parametrize(
    "options",
    [
        {"depths": ()},
        {"depths": (2, 1)},
        {"depths": (1, 1)},
        {"depths": (0,)},
        {"depths": (True,)},
        {"hardware": "H2"},
        {"seed": -1},
        {"interpolation_steps": 0},
        {"construction_steps": -1},
        {"noisy_steps": True},
        {"learning_rate": 0},
    ],
)
def test_iqsp_candidate_configuration_is_validated(options: dict) -> None:
    """Invalid settings cannot silently change family, hardware or stage count."""
    config: dict = {"hardware": "IBM", "seed": 300, "depths": (1,), **options}
    with pytest.raises(ValueError, match=r"Require|Depths|nonnegative|Adam configuration"):
        iqsp_candidates(gaussian_state(6, 1), **config)
