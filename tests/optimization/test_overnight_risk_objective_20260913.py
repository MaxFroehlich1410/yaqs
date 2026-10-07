# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent entropic-risk gradient, limit, stability and work-accounting checks."""

from __future__ import annotations

import copy
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

from experiments.overnight_state_preparation_20260913_0108 import channels, ensemble_objective, risk_objective
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate


def problem() -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Return q2 shared, reversed and scaled scalar gates with an unused parameter."""
    gates = [ParameterizedGate("h", (1,)), ParameterizedGate("x", (0,))]
    gates.extend(
        ParameterizedGate(
            name,
            (1, 0) if index >= 4 else (index % 2,),
            param_index=index % 4,
            angle_scale=(-1) ** index * (0.4 + 0.1 * index),
            angle_offset=0.1 * index,
            noise_enabled=index % 3 != 0,
        )
        for index, name in enumerate(("rx", "ry", "rz", "p", "rxx", "ryy", "rzz", "cp"))
    )
    rng = np.random.default_rng(810)
    target = rng.normal(size=4) + 1j * rng.normal(size=4)
    target /= np.linalg.norm(target)
    return ParameterizedCircuit(2, gates, num_params=5), rng.normal(size=5), target


@pytest.mark.parametrize("temperature", [0.02, 0.1, None])
def test_risk_gradient_matches_independent_complete_channel_finite_differences(temperature: float | None) -> None:
    """The physical channel expectation is differentiated before applying entropic risk."""
    circuit, theta, target = problem()
    original_gates, original_theta, original_target = copy.deepcopy(circuit.gates), theta.copy(), target.copy()
    events: list[tuple[str, str]] = []
    result = risk_objective.loss_gradient(
        circuit, theta, target, temperature=temperature, progress=lambda noise, phase: events.append((noise, phase))
    )

    def independent_value(parameters: np.ndarray) -> float:
        losses = np.array([
            1 - channels.evaluate(circuit, parameters, target, noise) for noise in risk_objective.TRAINING_NOISE_IDS
        ])
        return (
            float(np.mean(losses))
            if temperature is None
            else float(temperature * np.log(np.mean(np.exp(losses / temperature))))
        )

    assert result.loss == pytest.approx(independent_value(theta), abs=2e-13)
    differences = []
    for index in range(theta.size):
        offset = np.zeros(theta.size)
        offset[index] = 1e-6
        differences.append((independent_value(theta + offset) - independent_value(theta - offset)) / 2e-6)
    np.testing.assert_allclose(result.gradient, differences, atol=3e-9, rtol=2e-7)
    assert result.gradient[-1] == 0
    assert result.individual_channel_calls == 2
    assert result.density_direction_passes == 4
    assert events == [(noise, phase) for noise in risk_objective.TRAINING_NOISE_IDS for phase in ("start", "complete")]
    assert result.cpu_seconds >= 0
    assert result.wall_seconds >= 0
    assert all(not array.flags["W"] for array in (result.gradient, result.weights, result.component_losses))
    assert circuit.gates == original_gates
    np.testing.assert_array_equal(theta, original_theta)
    np.testing.assert_array_equal(target, original_target)


def test_mean_control_matches_frozen_ensemble_oracle() -> None:
    """Mean uses exactly the same two physical components and parameter convention."""
    circuit, theta, target = problem()
    actual = risk_objective.loss_gradient(circuit, theta, target, temperature=None)
    gradient, loss, fidelities = ensemble_objective.mean_loss_gradient(
        circuit, theta, target, risk_objective.TRAINING_NOISE_IDS
    )
    np.testing.assert_array_equal(actual.gradient, gradient)
    assert actual.loss == loss
    np.testing.assert_array_equal(actual.weights, [0.5, 0.5])
    np.testing.assert_allclose(actual.component_losses, [1 - value for value in fidelities.values()], atol=1e-15)


@pytest.mark.parametrize("temperature", [0.02, 0.1, 1e-300, 1e300])
def test_risk_limits_and_worst_channel_weights_are_stable(temperature: float) -> None:
    """Mean-to-max bounds hold without overflow and with the correct softmax direction."""
    losses = np.array([0.1, 0.9])
    loss, gradient, weights = risk_objective.aggregate(losses, np.eye(2), temperature)
    assert 0.5 - 1e-14 <= loss <= 0.9 + 1e-14
    assert loss >= 0.9 - temperature * np.log(2) - 1e-14
    assert weights[1] >= weights[0]
    assert sum(weights) == pytest.approx(1, abs=1e-15)
    np.testing.assert_array_equal(gradient, weights)
    if temperature < 1e-10:
        assert loss == pytest.approx(0.9)
        np.testing.assert_array_equal(weights, [0, 1])
    if temperature > 1e10:
        assert loss == pytest.approx(0.5, abs=1e-14)
        np.testing.assert_array_equal(weights, [0.5, 0.5])


def test_equal_losses_translation_and_entropy_variational_identity() -> None:
    """Normalizing by channel count preserves equal losses and the finite risk identity."""
    equal, gradient, weights = risk_objective.aggregate(np.array([0.3, 0.3]), np.array([[1.0], [-1.0]]), 0.02)
    assert equal == pytest.approx(0.3, abs=1e-16)
    np.testing.assert_array_equal(gradient, [0])
    np.testing.assert_array_equal(weights, [0.5, 0.5])
    losses = np.array([0.2, 0.7])
    value, _, weights = risk_objective.aggregate(losses, np.eye(2), 0.1)
    translated, _, translated_weights = risk_objective.aggregate(losses + 0.1, np.eye(2), 0.1)
    assert translated == pytest.approx(value + 0.1, abs=1e-15)
    np.testing.assert_allclose(translated_weights, weights, atol=1e-15)
    assert value == pytest.approx(weights @ losses - 0.1 * np.sum(weights * np.log(2 * weights)), abs=1e-15)


def test_interrupted_second_component_retains_first_completion_event() -> None:
    """A driver can charge completed and attempted profiles when the next one fails."""
    circuit, theta, target = problem()
    events: list[tuple[str, str]] = []

    def progress(noise: str, phase: str) -> None:
        events.append((noise, phase))
        if noise == risk_objective.TRAINING_NOISE_IDS[1] and phase == "start":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        risk_objective.loss_gradient(circuit, theta, target, progress=progress)
    assert len(events) == 3
    assert events[-1] == (risk_objective.TRAINING_NOISE_IDS[1], "start")


@pytest.mark.parametrize("temperature", [0, -0.1, np.inf, np.nan, True, 0.02j])
def test_invalid_temperature_rejected_before_component_work(temperature: object) -> None:
    """Neither malformed temperature nor mean-limit shorthand can change the objective silently."""
    circuit, theta, target = problem()
    with (
        patch.object(ensemble_objective, "loss_gradient", side_effect=AssertionError("must reject before work")),
        pytest.raises(ValueError, match="temperature must"),
    ):
        risk_objective.loss_gradient(circuit, theta, target, temperature=cast("float", temperature))


@pytest.mark.parametrize("invalid", ["shape", "nan", "complex", "bounds", "empty"])
def test_invalid_component_arrays_rejected(invalid: str) -> None:
    """Malformed component arrays cannot yield a seemingly valid risk objective."""
    losses, gradients = np.array([0.1, 0.2]), np.eye(2)
    if invalid == "shape":
        gradients = gradients[:1]
    elif invalid == "nan":
        gradients[0, 0] = np.nan
    elif invalid == "complex":
        losses = losses.astype(complex)
    elif invalid == "bounds":
        losses[0] = 1.1
    else:
        losses, gradients = np.array([]), np.empty((0, 2))
    with pytest.raises(ValueError, match="finite real infidelities"):
        risk_objective.aggregate(losses, gradients, 0.02)
