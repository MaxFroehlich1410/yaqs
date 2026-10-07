# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Startup auditing and large-register lowering without global unitary allocation."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

pytest.importorskip("torch")

from benchmarks.state_preparation.targets import load_target
from experiments.state_preparation_competitors.iqsp_comb_backend import CombOptions
from experiments.state_preparation_competitors.iqsp_tn import (
    GAUSSIAN_TARGET_ID,
    interpolation_target,
    iqsp_tn_ansatz,
    iqsp_tn_candidates,
    logical_parameterization,
)
from experiments.state_preparation_competitors.iqsp_tn_backend import CircuitTensorGates
from experiments.state_preparation_competitors.q12_readiness import (
    StatevectorAuditObjective,
    audit_gaussian_initialization,
)


def test_audit_oracle_matches_qiskit_directional_derivative() -> None:
    """An independent circuit simulator verifies the audit reference itself."""
    circuit = iqsp_tn_ansatz(3, 2)
    oracle = StatevectorAuditObjective(circuit)
    rng = np.random.default_rng(42)
    theta = rng.normal(size=circuit.num_parameters) * 0.2
    target = rng.normal(size=8) + 1j * rng.normal(size=8)
    target /= np.linalg.norm(target)
    loss, gradient = oracle.loss_gradient(theta, target)
    direction = rng.normal(size=theta.size)

    def exact(values: np.ndarray) -> float:
        return 1 - float(abs(np.vdot(target, Statevector.from_instruction(oracle.bind(values)).data)) ** 2)

    assert loss == pytest.approx(exact(theta), abs=2e-14)
    assert gradient @ direction == pytest.approx(
        (exact(theta + 1e-5 * direction) - exact(theta - 1e-5 * direction)) / 2e-5, abs=2e-8
    )
    with pytest.raises(ValueError, match="target"):
        oracle.loss_gradient(theta, [1, 0])
    noisy = StatevectorAuditObjective(oracle.bind(theta), "IBM")
    with pytest.raises(ValueError, match="noiseless"):
        noisy.loss_gradient([], target)


def test_actual_gaussian_startup_records_origin_error_and_recovers() -> None:
    """The audit neither hides the first derivative error nor supplies a fallback."""
    result = audit_gaussian_initialization(num_qubits=6, depth=1, steps=5)
    assert result["passed"]
    assert not result["gradient_fallback"]
    assert result["records"][0]["relative_gradient_error"] > 0.9
    assert result["records"][-1]["relative_gradient_error"] < 0.01
    assert not audit_gaussian_initialization(num_qubits=6, depth=1, steps=1)["passed"]
    for qubits, steps in [(5, 1), (12, 0), (12, 21), (12, True)]:
        with pytest.raises(ValueError, match="startup updates"):
            audit_gaussian_initialization(num_qubits=qubits, steps=steps)


def test_twelve_qubit_gaussian_path_and_logical_lowering() -> None:
    """The fixture remains one-dimensional and all 11 local SU4 blocks survive."""
    target = load_target(12, GAUSSIAN_TARGET_ID).state_vector
    np.testing.assert_allclose(interpolation_target(target, 0, target_id=GAUSSIAN_TARGET_ID), np.ones(4096) / 64)
    np.testing.assert_allclose(interpolation_target(target, 1, target_id=GAUSSIAN_TARGET_ID), target, atol=2e-15)
    circuit = iqsp_tn_ansatz(12, 1)
    theta = np.random.default_rng(12).normal(size=circuit.num_parameters) * 0.05
    bound = CircuitTensorGates(circuit).bind(theta)
    template, values, receipt = logical_parameterization(bound)
    assert receipt["su4_blocks"] == 11
    assert receipt["global_unitary_error"] is None
    assert receipt["prepared_state_error"] < 1e-12
    restored = template.assign_parameters(dict(zip(template.parameters, values, strict=True)))
    np.testing.assert_allclose(
        Statevector.from_instruction(restored).data, Statevector.from_instruction(bound).data, atol=2e-14
    )
    with pytest.raises(ValueError, match="compressed-comb"):
        iqsp_tn_candidates(target, target_id=GAUSSIAN_TARGET_ID)


def test_complete_twelve_qubit_compressed_candidate_smoke() -> None:
    """Construction, lowering, noisy Adam and original cutoff all accept q12."""
    target = load_target(12, GAUSSIAN_TARGET_ID).state_vector
    candidate = iqsp_tn_candidates(
        target,
        target_id=GAUSSIAN_TARGET_ID,
        depths=(1,),
        interpolation_steps=1,
        construction_steps=1,
        noisy_steps=1,
        comb_options=CombOptions(checkpoint_interval=8, target_tolerance=1e-28),
        target_construction="dense_svd",
    )[0]
    assert candidate.circuit.num_qubits == 12
    assert candidate.metadata["status"] == "complete"
    assert (
        candidate.metadata["postprocessing"]["final_tensor_network"]["target_source"]
        == "dense_fixture_leaf_svd_rtol_1e-28"
    )
    assert candidate.metadata["noise_basis"] == "logical"
    assert candidate.metadata["logical_transition"]["prepared_state_error"] < 1e-12
    assert candidate.metadata["noisy"]["steps"] == 1
    corrections = candidate.metadata["logical_transition"]["weyl_polar_corrections"]
    assert 0 < max(corrections) <= 1e-12
    assert candidate.metadata["logical_transition"]["rotation_count"] == 165
    np.testing.assert_array_equal(
        candidate.metadata["noisy"]["final_theta"], candidate.metadata["noisy"]["records"][-1]["theta"]
    )
