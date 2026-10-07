# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact expectation, cancellation, custody and work tests for anchor estimates."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from benchmarks.state_preparation.weighted_sampling import FactorSampler
from experiments.overnight_state_preparation_20260913_0108 import variance
from mqt.yaqs.optimization import KrotovNoiseMap, ParameterizedCircuit, ParameterizedGate, StatevectorBackend


def small_case() -> tuple[StatevectorBackend, np.ndarray, CalibratedLocalDepolarizingProvider]:
    """Return a shared-parameter complex-target circuit with one noisy window."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), param_index=0, angle_scale=-1.3, angle_offset=0.2),
            ParameterizedGate("rxx", (1, 0), param_index=1, noise_enabled=False),
            ParameterizedGate("rz", (1,), param_index=0, noise_enabled=False),
        ],
    )
    target = np.array([1, 2j, -3, 4j]) / np.sqrt(30)
    return StatevectorBackend(circuit, target), np.array([0.4, -0.7]), CalibratedLocalDepolarizingProvider(0.2, 0.3)


@pytest.mark.parametrize("snapshot_paths", [0, 17])
def test_anchor_cancellation_is_exact_and_work_is_not_hidden(snapshot_paths: int) -> None:
    """At the anchor every correction cancels, including an inexact snapshot."""
    engine, theta, calibration = small_case()
    original = theta.copy()
    snapshot = variance.build_snapshot(engine, theta, calibration, snapshot_paths=snapshot_paths, seed=100)
    for seed in (1000, 2000):
        result = variance.control_variate_estimate(engine, theta, calibration, snapshot, paths=5, seed=seed)
        np.testing.assert_array_equal(result.gradient, snapshot.estimate.gradient)
        assert result.loss == snapshot.estimate.loss
        assert result.paths == 10
        assert result.directional_passes == 20
        assert result.exact_channel_evaluations == 0
        assert result.cpu_seconds >= 0
        assert result.wall_seconds >= 0
    np.testing.assert_array_equal(theta, original)
    assert not snapshot.theta.flags["W"]
    assert not snapshot.estimate.gradient.flags["W"]
    assert snapshot.estimate.paths == snapshot_paths
    assert snapshot.estimate.directional_passes == 2 * snapshot_paths
    assert snapshot.estimate.exact_channel_evaluations == int(snapshot_paths == 0)


@pytest.mark.parametrize("snapshot_paths", [0, 5])
def test_complete_expectation_and_sampled_snapshot_bias(snapshot_paths: int) -> None:
    """Enumerate every conditional Pauli event, retaining multiple cancellations."""
    engine, anchor, calibration = small_case()
    theta = anchor + np.array([0.1, -0.2])
    snapshot = variance.build_snapshot(engine, anchor, calibration, snapshot_paths=snapshot_paths, seed=100)
    sampler = FactorSampler(engine.circuit, calibration)
    weights = np.array([sampler.no_factor_probability, sampler.event_probability])
    zero = (KrotovNoiseMap(),) * len(engine.circuit.gates)
    mean_gradient, mean_loss = np.zeros_like(theta), 0.0
    for pattern in product([False, True], repeat=len(sampler.factors)):
        mask = np.array(pattern)
        if not mask.any():
            continue
        probability = float(np.prod(np.where(mask, sampler.probabilities, 1 - sampler.probabilities)))
        probability /= sampler.event_probability
        maps = (zero, *sampler.maps(mask[None, :]))
        gradient, loss = variance.control_variate_from_maps(engine, theta, calibration, snapshot, maps, weights)
        mean_gradient += probability * gradient
        mean_loss += probability * loss
    truth, truth_loss, _ = density_loss_gradient(engine.circuit, theta, engine.target, calibration)
    anchor_truth, anchor_loss, _ = density_loss_gradient(engine.circuit, anchor, engine.target, calibration)
    np.testing.assert_allclose(mean_gradient - truth, snapshot.estimate.gradient - anchor_truth, atol=2e-14, rtol=0)
    assert mean_loss - truth_loss == pytest.approx(snapshot.estimate.loss - anchor_loss, abs=2e-14)


def test_independent_snapshot_streams_and_equal_direction_comparator() -> None:
    """Reject intersecting seed ranges and include independent snapshot work."""
    engine, anchor, calibration = small_case()
    snapshot = variance.build_snapshot(engine, anchor, calibration, snapshot_paths=9, seed=100)
    for seed in (97, 100, 107):
        with pytest.raises(ValueError, match="must be disjoint"):
            variance.control_variate_estimate(engine, anchor, calibration, snapshot, paths=5, seed=seed)
    control = variance.control_variate_estimate(engine, anchor + 0.02, calibration, snapshot, paths=5, seed=108)
    ordinary = variance.weighted_estimate(engine, anchor + 0.02, calibration, paths=10, seed=1000)
    assert control.directional_passes == ordinary.directional_passes
    assert control.paths == ordinary.paths
    assert control.bias_status == "conditional_bias_equals_frozen_snapshot_error"
    assert snapshot.estimate.directional_passes + control.directional_passes == 38
    repeated = variance.control_variate_estimate(engine, anchor + 0.02, calibration, snapshot, paths=5, seed=108)
    np.testing.assert_array_equal(control.gradient, repeated.gradient)
    assert control.loss == repeated.loss


def test_snapshot_rejects_changed_problem_and_nonzero_initial_state() -> None:
    """A control variate cannot silently change its channel, target or input."""
    engine, anchor, calibration = small_case()
    snapshot = variance.build_snapshot(engine, anchor, calibration)
    with pytest.raises(ValueError, match="signature differs"):
        variance.control_variate_estimate(
            engine, anchor, CalibratedLocalDepolarizingProvider(), snapshot, paths=3, seed=100
        )
    engine.target = np.roll(engine.target, 1)
    with pytest.raises(ValueError, match="signature differs"):
        variance.control_variate_estimate(engine, anchor, calibration, snapshot, paths=3, seed=100)
    engine.initial = np.roll(engine.initial, 1)
    with pytest.raises(ValueError, match="all-zero"):
        variance.build_snapshot(engine, anchor, calibration)


@pytest.mark.parametrize("paths", [-1, 1, True, 1.5])
def test_invalid_snapshot_budgets(paths: int) -> None:
    """Reject malformed snapshot budgets before propagation."""
    engine, anchor, calibration = small_case()
    with pytest.raises(ValueError, match="Snapshot paths"):
        variance.build_snapshot(engine, anchor, calibration, snapshot_paths=paths)


def test_zero_noise_correction_recovers_the_noiseless_gradient() -> None:
    """The deterministic no-noise limit agrees away from the anchor too."""
    engine, anchor, _ = small_case()
    calibration = CalibratedLocalDepolarizingProvider(0, 0)
    snapshot = variance.build_snapshot(engine, anchor, calibration)
    theta = anchor + 0.13
    estimate = variance.control_variate_estimate(engine, theta, calibration, snapshot, paths=4, seed=200)
    np.testing.assert_allclose(estimate.gradient, engine.gradient(theta)[0], atol=1e-14, rtol=0)
    assert estimate.loss == pytest.approx(engine.metrics(theta)[0], abs=1e-14)
