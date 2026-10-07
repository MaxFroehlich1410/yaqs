# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact finite deletion scores against independent whole-circuit propagation."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_deletion_fidelities
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider, exact_density_fidelity
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate


def problem() -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Return all gate generators, fixed gates, reversed sites and shared parameters."""
    gates = [ParameterizedGate("h", (2,)), ParameterizedGate("cx", (2, 0))]
    for i, name in enumerate(("rx", "ry", "rz", "p", "rxx", "ryy", "rzz", "cp")):
        gates.append(
            ParameterizedGate(
                name,
                (2, 0) if i >= 4 else (1,),
                param_index=i % 3,
                angle_scale=-0.7 + i * 0.3,
                angle_offset=0.21,
                noise_enabled=i % 3 != 1,
            )
        )
    rng = np.random.default_rng(907)
    target = rng.normal(size=8) + 1j * rng.normal(size=8)
    target /= np.linalg.norm(target)
    return ParameterizedCircuit(3, gates), rng.normal(size=3), target


@pytest.mark.parametrize("errors", [(0, 0), (0.00064, 0.0051), (0.13, 0.24)])
def test_all_deletions_equal_independent_exhaustive_density(errors: tuple[float, float]) -> None:
    """Every score removes the gate and channel, retaining all other windows exactly."""
    circuit, theta, target = problem()
    before_theta, before_target = theta.copy(), target.copy()
    provider = ExactLocalPauliProvider(CalibratedLocalDepolarizingProvider(*errors))
    with patch.object(circuit, "gate_matrix", wraps=circuit.gate_matrix) as matrices:
        scores, original = density_deletion_fidelities(circuit, theta, target, provider.calibration)
    assert matrices.call_count == len(circuit.gates)
    assert original == pytest.approx(exact_density_fidelity(circuit, theta, target, provider), abs=1e-12)
    for index, score in enumerate(scores):
        deleted = ParameterizedCircuit(3, [g for i, g in enumerate(circuit.gates) if i != index], circuit.num_params)
        assert score == pytest.approx(exact_density_fidelity(deleted, theta, target, provider), abs=1e-12)
    np.testing.assert_array_equal(theta, before_theta)
    np.testing.assert_array_equal(target, before_target)


def test_empty_circuit_and_identity_rotation_deletion() -> None:
    """Deleting the only zero-angle rotation removes noise; no gates returns empty scores."""
    target = np.array([1, 0], dtype=complex)
    calibration = CalibratedLocalDepolarizingProvider()
    empty, fidelity = density_deletion_fidelities(ParameterizedCircuit(1, []), np.array([]), target, calibration)
    assert empty.size == 0
    assert fidelity == pytest.approx(1)
    gate = ParameterizedGate("rz", (0,), param_index=0)
    scores, original = density_deletion_fidelities(ParameterizedCircuit(1, [gate]), np.zeros(1), target, calibration)
    assert original < 1
    assert scores[0] == pytest.approx(1)
    gate.noise_enabled = False
    scores, original = density_deletion_fidelities(ParameterizedCircuit(1, [gate]), np.zeros(1), target, calibration)
    assert scores[0] == original == pytest.approx(1)


@pytest.mark.parametrize(
    "invalid", ["large", "theta_shape", "complex_theta", "theta_nan", "target_shape", "target_nan", "norm", "data"]
)
def test_invalid_inputs(invalid: str) -> None:
    """Validate inputs before constructing an adjoint tape."""
    circuit, theta, target = problem()
    if invalid == "large":
        circuit.num_qubits = 7
    elif invalid == "theta_shape":
        theta = theta[:-1]
    elif invalid == "complex_theta":
        theta = theta.astype(complex)
    elif invalid == "theta_nan":
        theta[0] = np.nan
    elif invalid == "target_shape":
        target = target[:-1]
    elif invalid == "target_nan":
        target[0] = np.nan
    elif invalid == "norm":
        target *= 2
    else:
        circuit.gates[0].data_map = lambda _: 0.1
    with pytest.raises(ValueError, match="Exact deletion adjoint"):
        density_deletion_fidelities(circuit, theta, target, CalibratedLocalDepolarizingProvider())
