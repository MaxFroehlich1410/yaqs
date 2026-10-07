# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Checkpoint selection, archived parity, and noiseless Armijo safeguards."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from experiments import topdown_optimizer_policies as study
from experiments.legacy_topdown_dense import train
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate, StatevectorBackend


def case() -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Return a signed, entangling circuit with an affine and shared parameter."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), 0),
            ParameterizedGate("rxx", (0, 1), 1, angle_scale=-1.7, angle_offset=0.2),
            ParameterizedGate("rz", (1,), 2),
            ParameterizedGate("ry", (1,), 0, angle_scale=0.4),
        ],
        3,
    )
    theta = np.array([0.7, -0.6, 1.1])
    target = np.array([1, 2j, -3, 4j], dtype=complex) / np.sqrt(30)
    return circuit, theta, target


@pytest.mark.parametrize(("steps", "online", "stop"), [(12, 5, 2.0), (4, 0, 2.0), (6, 8, 0.0)])
def test_original_and_best_only_reproduce_every_archived_update(steps: int, online: int, stop: float) -> None:
    """Retaining a checkpoint must not modify any original update or stop rule."""
    circuit, theta, target = case()
    original_input = theta.copy()
    expected, legacy_trace = train(circuit, theta, target, steps=steps, online=online, rate=0.7, seed=400, stop=stop)
    for policy in ("original_final", "best_only"):
        returned, trace = study.train_noiseless(
            circuit, theta, target, policy=policy, steps=steps, online=online, stop=stop
        )
        np.testing.assert_array_equal(trace["fidelity_trace"], legacy_trace["fidelity_trace"])
        np.testing.assert_array_equal(trace["final_theta"], expected)
        np.testing.assert_array_equal(theta, original_input)
        assert trace["steps"] == legacy_trace["steps"]
        assert trace["online_steps"] == legacy_trace["online_steps"]
        assert trace["objective_evaluations"] == trace["steps"] + 1
        assert trace["gradient_evaluations"] == trace["steps"] - trace["online_steps"]
        assert StatevectorBackend(circuit, target).metrics(returned)[1] == trace["returned_fidelity"]
        if policy == "original_final":
            np.testing.assert_array_equal(returned, expected)
        json.dumps(trace, allow_nan=False)


def test_best_checkpoint_contains_actual_nonfinal_parameters() -> None:
    """An oversized fixed step loses a good iterate, whose vector is retained."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), 0)], 1)
    theta, target = np.array([0.5]), np.array([0, 1], dtype=complex)
    returned, trace = study.train_noiseless(circuit, theta, target, policy="best_only", steps=2, online=0, rate=8)
    assert trace["best_iteration"] == 1
    assert trace["best_fidelity"] > trace["final_fidelity"] + 0.4
    expected, _ = train(circuit, theta, target, steps=1, rate=8, seed=400)
    np.testing.assert_array_equal(returned, expected)
    assert StatevectorBackend(circuit, target).metrics(returned)[1] == trace["best_fidelity"]
    assert returned.tolist() != trace["final_theta"]


def test_initial_checkpoint_is_eligible_even_if_every_update_is_worse() -> None:
    """Iteration zero protects a circuit from a harmful first repair step."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), 0)], 1)
    theta, target = np.array([2.9]), np.array([0, 1], dtype=complex)
    returned, trace = study.train_noiseless(circuit, theta, target, policy="best_only", steps=1, online=0, rate=8)
    assert trace["best_iteration"] == 0
    assert trace["final_fidelity"] < trace["initial_fidelity"]
    np.testing.assert_array_equal(returned, theta)


@pytest.mark.parametrize("policy", study.POLICIES)
def test_zero_budget_returns_initial_checkpoint(policy: str) -> None:
    """No updates still yields complete, accurately scored endpoint metadata."""
    circuit, theta, target = case()
    returned, trace = study.train_noiseless(circuit, theta, target, policy=policy, steps=0)
    np.testing.assert_array_equal(returned, theta)
    assert trace["steps"] == trace["online_sweeps"] == trace["gradient_evaluations"] == 0
    assert trace["objective_evaluations"] == 1
    assert trace["best_iteration"] == 0
    assert trace["initial_fidelity"] == trace["final_fidelity"] == trace["returned_fidelity"]
    assert trace["records"] == []


def test_backtracking_preserves_online_prefix_and_starts_from_its_endpoint() -> None:
    """Best tracking does not silently restart the batch phase from another vector."""
    circuit, theta, target = case()
    _, original = study.train_noiseless(circuit, theta, target, policy="original_final", steps=12, online=5)
    returned, adaptive = study.train_noiseless(circuit, theta, target, policy="best_backtracking", steps=12, online=5)
    np.testing.assert_array_equal(adaptive["fidelity_trace"][:5], original["fidelity_trace"][:5])
    assert adaptive["records"][5]["before_fidelity"] == adaptive["online_final_fidelity"]
    assert adaptive["online_sweeps"] == 5
    assert adaptive["gradient_evaluations"] == 7
    assert all(record["gradient_norm"] is None for record in adaptive["records"][:5])
    assert StatevectorBackend(circuit, target).metrics(returned)[1] == adaptive["returned_fidelity"]


def test_backtracking_counts_rejections_and_accepts_only_armijo_steps() -> None:
    """An oversized batch rate is reduced while reusing each gradient exactly once."""
    circuit, theta, target = case()
    returned, trace = study.train_noiseless(
        circuit, theta, target, policy="best_backtracking", steps=8, online=0, rate=16
    )
    assert trace["rejected_trials"] > 0
    assert trace["failed_line_searches"] == 0
    assert trace["gradient_evaluations"] == 8
    assert trace["objective_evaluations"] == 1 + 8 + trace["rejected_trials"]
    for record in trace["records"]:
        assert record["accepted"]
        assert record["fidelity"] >= record["before_fidelity"]
        assert record["objective_evaluations"] == len(record["trial_rates"]) == record["rejections"] + 1
        loss = 1 - record["fidelity"]
        bound = 1 - record["before_fidelity"] - study.ARMIJO * record["rate"] * record["gradient_norm"] ** 2
        assert loss <= bound + 1e-15
        assert record["rate"] == pytest.approx(16 * 0.5 ** record["rejections"])
    np.testing.assert_array_equal(returned, trace["final_theta"])


@pytest.mark.parametrize("trial_loss", [1.0, float("nan")])
def test_failed_line_searches_keep_parameters_and_count_every_trial(
    monkeypatch: pytest.MonkeyPatch, trial_loss: float
) -> None:
    """Worse or nonfinite trials never replace the valid active checkpoint."""
    circuit, theta, target = case()
    original_metrics = StatevectorBackend.metrics
    metrics_calls = 0

    def failing_trials(self: StatevectorBackend, value: np.ndarray) -> tuple[float, float]:
        nonlocal metrics_calls
        metrics_calls += 1
        if metrics_calls == 1:
            return original_metrics(self, value)
        return trial_loss, 1 - trial_loss

    monkeypatch.setattr(StatevectorBackend, "metrics", failing_trials)
    returned, trace = study.train_noiseless(circuit, theta, target, policy="best_backtracking", steps=2, online=0)
    np.testing.assert_array_equal(returned, theta)
    np.testing.assert_array_equal(trace["final_theta"], theta)
    assert trace["gradient_evaluations"] == 2
    assert trace["failed_line_searches"] == 2
    assert trace["rejected_trials"] == 2 * (study.MAX_BACKTRACKS + 1)
    assert metrics_calls == trace["objective_evaluations"] == 1 + trace["rejected_trials"]
    assert trace["final_fidelity"] == trace["initial_fidelity"]
    assert all(record["failure_reason"] == "trial_budget_exhausted" for record in trace["records"])
    json.dumps(trace, allow_nan=False)


def test_exact_loss_gradient_matches_finite_differences_at_input_and_online_endpoint() -> None:
    """Check the descent direction, including angle scales and shared parameters."""
    circuit, theta, target = case()
    engine = StatevectorBackend(circuit, target)
    endpoint, _ = study.train_noiseless(circuit, theta, target, policy="original_final", steps=5, online=5)
    for point in (theta, endpoint):
        gradient, _, _ = engine.gradient(point, estimator="independent")
        finite_difference = np.zeros_like(point)
        for index in range(len(point)):
            displacement = np.zeros_like(point)
            displacement[index] = 1e-6
            finite_difference[index] = (
                engine.metrics(point + displacement)[0] - engine.metrics(point - displacement)[0]
            ) / 2e-6
        np.testing.assert_allclose(gradient, finite_difference, atol=3e-10, rtol=1e-7)


def test_nonfinite_gradient_is_rejected_without_evaluating_invalid_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject an invalid direction before propagation and retain strict JSON metadata."""
    circuit, theta, target = case()
    engine = StatevectorBackend(circuit, target)
    loss, fidelity = engine.metrics(theta)

    def invalid_gradient(*args: object, **kwargs: object) -> tuple[np.ndarray, float, float]:
        del args, kwargs
        return np.full_like(theta, np.nan), loss, fidelity

    monkeypatch.setattr(StatevectorBackend, "gradient", invalid_gradient)
    returned, trace = study.train_noiseless(circuit, theta, target, policy="best_backtracking", steps=2, online=0)
    np.testing.assert_array_equal(returned, theta)
    assert trace["failed_line_searches"] == trace["gradient_evaluations"] == 2
    assert trace["objective_evaluations"] == 1
    assert trace["rejected_trials"] == 0
    assert all(record["failure_reason"] == "nonfinite_gradient" for record in trace["records"])
    json.dumps(trace, allow_nan=False)


@pytest.mark.parametrize(
    "options",
    [
        {"policy": "unknown"},
        {"steps": -1},
        {"online": True},
        {"steps": 1.5},
        {"rate": -1},
        {"rate": float("nan")},
        {"stop": float("nan")},
    ],
)
def test_invalid_options_fail_before_training(options: dict[str, Any]) -> None:
    """Reject invalid comparison configurations rather than silently changing them."""
    circuit, theta, target = case()
    arguments: dict[str, Any] = {"policy": "original_final"}
    arguments.update(options)
    with pytest.raises(ValueError, match=r"policy|budgets|Rate"):
        study.train_noiseless(circuit, theta, target, **arguments)
