# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent oracle tests for the restricted real-state Fontana adaptation."""

from __future__ import annotations

import json

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import DensityMatrix, Statevector
from scipy.optimize import minimize

from benchmarks.state_preparation.targets import load_target_collection
from experiments.legacy_topdown_dense import noise_channel
from experiments.state_preparation_competitors.fontana import (
    fontana_ansatz,
    fontana_candidates,
    optimize_bfgs,
    real_target,
)
from experiments.state_preparation_competitors.noisy_core import DenseObjective


def test_real_target_removes_only_one_global_phase() -> None:
    """Negative amplitudes are retained and genuinely relative complex phases rejected."""
    target = np.array([1, -2, 3, -4]) / np.sqrt(30)
    actual = real_target(np.exp(0.73j) * target)
    np.testing.assert_allclose(actual, -target, atol=1e-15)
    assert actual.dtype == np.complex128
    with pytest.raises(ValueError, match="real target"):
        real_target([1 / np.sqrt(2), 1j / np.sqrt(2)])
    with pytest.raises(ValueError, match="state vector"):
        real_target([1, 1])


def test_fixture_eligibility_includes_real_mps_and_excludes_complex_haar() -> None:
    """The benchmark's real and complex target subsets must not be conflated."""
    collection = load_target_collection()
    for name in (
        "gaussian_mu0p5_sigma0p1",
        "tfim_ferro",
        "tfim_critical",
        "tfim_para",
        "random_mps_bond2",
        "random_mps_bond3",
    ):
        value = collection.load(6, name).state_vector_copy()
        assert abs(np.vdot(value, real_target(value))) ** 2 == pytest.approx(1, abs=1e-12)
    for name in ("haar_random_1", "haar_random_2", "haar_random_3"):
        with pytest.raises(ValueError, match="real target"):
            real_target(collection.load(6, name).state_vector_copy())


def test_ansatz_matches_published_four_qubit_layer_and_six_qubit_extension() -> None:
    """The two alternating directed CX sublayers follow the inspected Fig.6."""
    for qubits in (4, 6):
        circuit = fontana_ansatz(qubits, 2)
        assert len(circuit.parameters) == 2 * qubits
        assert circuit.count_ops() == {"ry": 2 * qubits, "cx": 2 * (qubits - 1)}
        pairs = [
            tuple(circuit.find_bit(q).index for q in item.qubits)
            for item in circuit.data
            if item.operation.name == "cx"
        ]
        expected = [(site, site + 1) for parity in (0, 1) for site in range(parity, qubits - 1, 2)]
        assert pairs == expected * 2
        angles = np.linspace(-1, 1, len(circuit.parameters))
        bound = circuit.assign_parameters(angles)
        assert np.max(abs(Statevector.from_instruction(bound).data.imag)) < 1e-14


@pytest.mark.parametrize(("qubits", "depth"), [(0, 1), (13, 1), (True, 1), (2, 0), (2, -1), (2, True)])
def test_ansatz_rejects_invalid_dimensions(qubits: int, depth: int) -> None:
    """Invalid dimensions cannot create a silently different gate grid."""
    with pytest.raises(ValueError, match=r"requires|nonnegative|positive|Require|Depths"):
        fontana_ansatz(qubits, depth)


def test_bfgs_matches_direct_scipy_on_an_analytic_single_qubit_problem() -> None:
    """The adapter's minimization agrees with BFGS on the closed-form infidelity."""
    circuit = QuantumCircuit(1)
    circuit.ry(Parameter("theta"), 0)
    angle = 1.1
    target = np.array([np.cos(angle / 2), np.sin(angle / 2)])
    objective = DenseObjective(circuit, target, None, compile_native=False)
    initial = np.array([-0.4])
    actual, trace = optimize_bfgs(objective, initial, steps=20)
    expected = minimize(
        lambda theta: (np.sin((theta[0] - angle) / 2) ** 2, np.array([np.sin(theta[0] - angle) / 2])),
        initial,
        method="BFGS",
        jac=True,
        options={"maxiter": 20, "gtol": 1e-8},
    )
    np.testing.assert_allclose(actual, expected.x, atol=1e-10)
    assert trace["returned_fidelity"] == pytest.approx(1, abs=1e-12)
    assert trace["records"][0]["theta"] == initial.tolist()
    assert trace["objective_evaluations"] == trace["gradient_evaluations"]
    assert trace["cpu_seconds"] > 0
    json.dumps(trace, allow_nan=False)


def test_bfgs_zero_steps_and_budget_stop_retain_the_actual_initial_iterate() -> None:
    """A rejected/unavailable optimizer probe cannot replace an accepted endpoint."""
    objective = DenseObjective(fontana_ansatz(1, 1), [0, 1], None)
    initial = np.array([0.2])
    for options, reason in (({"steps": 0}, "zero_steps"), ({"steps": 5, "max_evaluations": 1}, "evaluation_budget")):
        theta, trace = optimize_bfgs(objective, initial, **options)
        np.testing.assert_array_equal(theta, initial)
        assert trace["steps"] == 0
        assert trace["objective_evaluations"] == 1
        assert trace["stop_reason"] == reason


@pytest.mark.parametrize("options", [{"steps": -1}, {"steps": True}, {"steps": 1, "max_evaluations": 0}])
def test_bfgs_invalid_budgets_fail(options: dict) -> None:
    """Invalid optimizer budgets are rejected before any update."""
    objective = DenseObjective(fontana_ansatz(1, 1), [1, 0], None)
    with pytest.raises(ValueError, match=r"requires|nonnegative|positive|Require|Depths"):
        optimize_bfgs(objective, [0.0], **options)


def test_bfgs_nonfinite_oracle_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A numerical failure remains a failure rather than a low-fidelity result."""
    objective = DenseObjective(fontana_ansatz(1, 1), [1, 0], None)
    monkeypatch.setattr(objective, "loss_gradient", lambda _theta: (np.nan, np.zeros(1)))
    with pytest.raises(FloatingPointError):
        optimize_bfgs(objective, [0.0], steps=1)


@pytest.mark.parametrize("noise_basis", ["native", "logical"])
def test_actual_six_qubit_candidate_preserves_its_training_channel(noise_basis: str) -> None:
    """A tiny end-to-end run agrees with independent density propagation gate by gate."""
    target = load_target_collection().load(6, "tfim_para").state_vector_copy()
    candidate = fontana_candidates(
        target, hardware="IBM", seed=300, depths=(1,), construction_steps=1, noisy_steps=1, noise_basis=noise_basis
    )[0]
    metadata = candidate.metadata
    assert metadata["preserve_native_schedule"] is (noise_basis == "native")
    logical = fontana_ansatz(6, 1)
    if noise_basis == "logical":
        assert metadata["noise_basis"] == "logical"
        assert "after each original" in metadata["training_noise"]
        assert [item.operation.name for item in candidate.circuit.data] == [
            item.operation.name for item in logical.data
        ]
        assert len(candidate.circuit.data) == 11
    else:
        assert len(candidate.circuit.data) > len(logical.data)
    assert metadata["construction_theta"] == metadata["noisy_training"]["initial_theta"]
    assert metadata["noisy_training"]["returned_fidelity"] >= metadata["noisy_training"]["initial_fidelity"] - 1e-14
    rho = DensityMatrix.from_label("000000")
    for item in candidate.circuit.data:
        sites = [candidate.circuit.find_bit(q).index for q in item.qubits]
        allowed = {"rz", "sx", "x", "cx"} if noise_basis == "native" else {"ry", "cx"}
        assert item.operation.name in allowed
        rho = rho.evolve(item.operation, qargs=sites).evolve(noise_channel("IBM", len(sites)), qargs=sites)
    expected = float(np.vdot(target, rho.data @ target).real)
    assert metadata["training_fidelity"] == pytest.approx(expected, abs=1e-12)
    assert metadata["cpu_seconds"] > 0
    json.dumps(metadata, allow_nan=False)


def test_native_default_matches_explicit_native_training() -> None:
    """Opting into logical noise cannot change the established default result."""
    config: dict = {"hardware": "IBM", "seed": 300, "depths": (1,), "construction_steps": 1, "noisy_steps": 1}
    default = fontana_candidates([1, 0, 0, 0], **config)[0]
    explicit = fontana_candidates([1, 0, 0, 0], noise_basis="native", **config)[0]
    assert default.circuit == explicit.circuit
    assert default.metadata["theta"] == explicit.metadata["theta"]
    assert default.metadata["training_fidelity"] == explicit.metadata["training_fidelity"]
    assert (
        default.metadata["training_noise"]
        == "exact common-native synthetic Pauli channel; not paper product-layer noise"
    )
    assert "noise_basis" not in default.metadata


@pytest.mark.parametrize("noise_basis", ["compiled", "LOGICAL", ""])
def test_candidate_rejects_unknown_noise_basis(noise_basis: str) -> None:
    """A misspelled noise placement must fail rather than silently compiling."""
    with pytest.raises(ValueError, match="noise_basis"):
        fontana_candidates([1, 0], hardware="IBM", seed=300, depths=(1,), noise_basis=noise_basis)


def test_initializations_replay_and_vary_with_declared_seed() -> None:
    """Seeds correspond to real independent angle initializations, with exact replay."""
    target = np.array([1, 0, 0, 0], dtype=complex)
    values = []
    for seed in (300, 300, 400):
        item = fontana_candidates(
            target, hardware="Google", seed=seed, depths=(1, 2), construction_steps=0, noisy_steps=0
        )
        values.append(item[0].metadata["initial_theta"])
        assert item[1].metadata["cumulative_cpu_seconds"] >= item[0].metadata["cumulative_cpu_seconds"]
    assert values[0] == values[1]
    assert values[0] != values[2]


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
        {"construction_steps": -1},
        {"noisy_steps": True},
    ],
)
def test_candidate_configuration_is_validated(options: dict) -> None:
    """Unsupported configurations do not get remapped to a different study."""
    config: dict = {"hardware": "IBM", "seed": 300, "depths": (1,), **options}
    with pytest.raises(ValueError, match=r"requires|nonnegative|positive|Require|Depths"):
        fontana_candidates([1, 0], **config)
