# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Finite ensemble deletion versus independent complete Qiskit propagation."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from experiments.overnight_state_preparation_20260913_0108 import channels, ensemble_deletion
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate


def problem() -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Return q2 noncommuting gates with sharing, reversed sites and disabled noise."""
    gates = [ParameterizedGate("h", (1,)), ParameterizedGate("cx", (1, 0))]
    for index, name in enumerate(("rx", "ry", "rz", "p", "rxx", "ryy", "rzz", "cp")):
        gates.append(
            ParameterizedGate(
                name,
                (1, 0) if index >= 4 else (0,),
                param_index=index % 3,
                angle_scale=-0.7 + 0.3 * index,
                angle_offset=0.21,
                noise_enabled=index % 3 != 1,
            )
        )
    rng = np.random.default_rng(724)
    target = rng.normal(size=4) + 1j * rng.normal(size=4)
    target /= np.linalg.norm(target)
    return ParameterizedCircuit(2, gates), rng.normal(size=3), target


@pytest.mark.parametrize("profiles", [ensemble_deletion.MEAN2, ensemble_deletion.FIXED_PROFILES])
def test_every_finite_deletion_matches_independent_whole_circuit(profiles: tuple[str, ...]) -> None:
    """Removing each full gate/window matches exact independent Qiskit evaluation."""
    circuit, theta, target = problem()
    original_theta, original_target = theta.copy(), target.copy()
    with patch.object(circuit, "gate_matrix", wraps=circuit.gate_matrix) as matrices:
        result = ensemble_deletion.deletion_fidelities(circuit, theta, target, profiles)
    assert matrices.call_count == len(circuit.gates)
    for noise in profiles:
        assert result.profile_fidelities[noise] == pytest.approx(
            channels.evaluate(circuit, theta, target, noise), abs=2e-13
        )
        for index, score in enumerate(result.profile_scores[noise]):
            deleted = ParameterizedCircuit(
                2, [gate for i, gate in enumerate(circuit.gates) if i != index], circuit.num_params
            )
            assert score == pytest.approx(channels.evaluate(deleted, theta, target, noise), abs=2e-13)
    np.testing.assert_allclose(result.mean_scores, np.mean(list(result.profile_scores.values()), axis=0), atol=1e-15)
    assert result.mean_fidelity == pytest.approx(np.mean(list(result.profile_fidelities.values())))
    assert result.individual_channel_calls == len(profiles)
    assert result.density_direction_passes == 2 * len(profiles)
    assert result.cpu_seconds >= 0
    assert result.wall_seconds >= 0
    assert not result.mean_scores.flags["W"]
    np.testing.assert_array_equal(theta, original_theta)
    np.testing.assert_array_equal(target, original_target)


def test_zero_angle_gate_deletion_also_removes_its_noise_and_empty_circuit_is_valid() -> None:
    """Zero angle is not equivalent to deleting its calibrated noisy window."""
    target = np.array([1, 1], dtype=complex) / np.sqrt(2)
    circuit = ParameterizedCircuit(
        1, [ParameterizedGate("h", (0,), noise_enabled=False), ParameterizedGate("rz", (0,), 0)]
    )
    result = ensemble_deletion.deletion_fidelities(circuit, np.zeros(1), target)
    assert result.mean_fidelity < 1
    assert result.mean_scores[1] == pytest.approx(1, abs=1e-14)
    assert all(result.profile_scores[noise][1] > result.profile_fidelities[noise] for noise in ensemble_deletion.MEAN2)
    empty = ensemble_deletion.deletion_fidelities(ParameterizedCircuit(1, []), np.array([]), target)
    assert empty.mean_scores.size == 0
    assert empty.mean_fidelity == pytest.approx(0.5)


def test_protected_original_origins_survive_index_shifts_and_ties_are_stable() -> None:
    """Protection follows original positions; equal scores use original-index order."""
    circuit, theta, target = problem()
    result = ensemble_deletion.deletion_fidelities(circuit, theta, target)
    result = replace(result, mean_scores=np.array([0.5, 0.5, 0.5]))
    assert ensemble_deletion.ranked_indices(result, (17, 31, 25)) == [2, 1]
    assert ensemble_deletion.ranked_indices(result, (17, 31, 25), protected_origins=frozenset()) == [0, 2, 1]
    with pytest.raises(ValueError, match="original gate identities"):
        ensemble_deletion.ranked_indices(result, (17, 17, 25))


def test_component_progress_preserves_attempted_and_completed_accounting() -> None:
    """An interrupted second profile leaves the first profile explicitly complete."""
    circuit, theta, target = problem()
    events: list[tuple[str, str]] = []

    def progress(noise: str, phase: str) -> None:
        events.append((noise, phase))
        if noise == ensemble_deletion.MEAN2[1] and phase == "start":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        ensemble_deletion.deletion_fidelities(circuit, theta, target, progress=progress)
    assert events == [
        (ensemble_deletion.MEAN2[0], "start"),
        (ensemble_deletion.MEAN2[0], "complete"),
        (ensemble_deletion.MEAN2[1], "start"),
    ]


@pytest.mark.parametrize(
    "invalid",
    [
        "ballarin",
        "duplicate",
        "empty_profiles",
        "large",
        "complex_theta",
        "theta_shape",
        "target_norm",
        "angle_nan",
        "data",
    ],
)
def test_invalid_domain_rejected_before_any_forward_pass(invalid: str) -> None:
    """Angle-dependent channels and malformed inputs cannot enter the fixed map proof."""
    circuit, theta, target = problem()
    profiles = ensemble_deletion.MEAN2
    if invalid == "ballarin":
        profiles = ("ballarin_coupled",)
    elif invalid == "duplicate":
        profiles = ("dephasing_1s_all", "dephasing_1s_all")
    elif invalid == "empty_profiles":
        profiles = ()
    elif invalid == "large":
        circuit.num_qubits = 7
    elif invalid == "complex_theta":
        theta = theta.astype(complex)
    elif invalid == "theta_shape":
        theta = theta[:-1]
    elif invalid == "target_norm":
        target *= 2
    elif invalid == "angle_nan":
        circuit.gates[0].angle_offset = np.nan
    else:
        circuit.gates[0].data_map = lambda _: 0.1
    with (
        patch.object(circuit, "gate_matrix", wraps=circuit.gate_matrix) as matrices,
        pytest.raises(ValueError, match="Ensemble deletion requires"),
    ):
        ensemble_deletion.deletion_fidelities(circuit, theta, target, profiles)
    assert matrices.call_count == 0
