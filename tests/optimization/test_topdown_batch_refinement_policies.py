# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Frozen-control parity and curvature-aware batch-refinement safeguards."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from experiments import topdown_batch_refinement_policies as study
from experiments import topdown_optimizer_policies as control
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate, StatevectorBackend


def case() -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Return an entangling circuit with scaled and shared rotation parameters."""
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
    return circuit, np.array([0.7, -0.6, 1.1]), np.array([1, 2j, -3, 4j], dtype=complex) / np.sqrt(30)


@pytest.mark.parametrize("policy", ["best_only", "best_backtracking"])
@pytest.mark.parametrize(("steps", "online", "rate"), [(30, 5, 0.7), (12, 0, 16.0), (0, 5, 0.7)])
def test_controls_reproduce_frozen_parameters_traces_and_work_counts(
    policy: str, steps: int, online: int, rate: float
) -> None:
    """New instrumentation must leave both scientific control paths unchanged."""
    circuit, theta, target = case()
    before = theta.copy()
    expected, old = control.train_noiseless(
        circuit, theta, target, policy=policy, steps=steps, online=online, rate=rate
    )
    returned, new = study.train_noiseless(circuit, theta, target, policy=policy, steps=steps, online=online, rate=rate)
    np.testing.assert_array_equal(returned, expected)
    np.testing.assert_array_equal(theta, before)
    for key in ("fidelity_trace", "final_theta", "online_final_theta", "best_theta"):
        np.testing.assert_array_equal(new[key], old[key])
    for key in (
        "best_iteration",
        "objective_evaluations",
        "gradient_evaluations",
        "online_sweeps",
        "rejected_trials",
        "failed_line_searches",
        "returned_fidelity",
        "final_fidelity",
        "online_final_fidelity",
    ):
        assert new[key] == old[key]
    for record, previous in zip(new["records"], old["records"], strict=True):
        for key in ("rate", "accepted", "objective_evaluations", "gradient_evaluations", "rejections", "trial_rates"):
            assert record[key] == previous[key]
    json.dumps(new, allow_nan=False)


@pytest.mark.parametrize("policy", study.POLICIES)
def test_online_prefix_and_transition_are_unchanged(policy: str) -> None:
    """Only batch refinement changes, with no online gradient or history creation."""
    circuit, theta, target = case()
    _, expected = control.train_noiseless(circuit, theta, target, policy="best_only", steps=6, online=6)
    _, trace = study.train_noiseless(circuit, theta, target, policy=policy, steps=14, online=6)
    np.testing.assert_array_equal(trace["online_final_theta"], expected["final_theta"])
    np.testing.assert_array_equal(trace["fidelity_trace"][:6], expected["fidelity_trace"])
    assert trace["records"][6]["before_fidelity"] == expected["final_fidelity"]
    assert trace["gradient_evaluations"] == 8
    assert trace["online_sweeps"] == 6
    assert trace["records"][6]["initial_trial_rate"] == pytest.approx(0.7)
    assert trace["records"][6]["curvature_valid"] is None
    assert all(record["history_pairs"] == 0 for record in trace["records"][:7])
    assert trace["records"][5]["cumulative_cpu_seconds"] <= trace["batch_start_cpu_seconds"]
    assert trace["batch_start_cpu_seconds"] <= trace["records"][6]["cumulative_cpu_seconds"]


def test_bb_formula_bounds_and_nonpositive_curvature_fallback() -> None:
    """Use BB1, clamp initial rates, and reject zero, negative, or tiny curvature."""
    step, info = study.bb_initial_rate(np.array([2.0, 1.0]), np.array([1.0, 3.0]), 0.7)
    assert step == pytest.approx(1.0)
    assert info["curvature_valid"]
    assert not info["bb_fallback_used"]
    for difference, expected in ((1e-12, study.BB_MAX_RATE), (1e12, study.BB_MIN_RATE)):
        step, info = study.bb_initial_rate(np.array([1.0]), np.array([difference]), 0.7)
        assert step == pytest.approx(expected)
        assert info["bb_bound_hit"]
    for difference in (np.zeros(2), np.array([-1.0, 0.0]), np.array([1e-12, 1.0]), np.array([np.nan, 0.0])):
        step, info = study.bb_initial_rate(np.array([1.0, 0.0]), difference, 0.175)
        assert step == pytest.approx(0.175)
        assert not info["curvature_valid"]
        assert info["bb_fallback_used"]
        json.dumps(info, allow_nan=False)


@pytest.mark.parametrize("policy", ["best_backtracking", "bb_backtracking", "lbfgs"])
def test_safeguarded_batch_updates_are_monotone_and_cost_every_trial(policy: str) -> None:
    """Armijo uses the actual directional derivative, including quasi-Newton steps."""
    circuit, theta, target = case()
    returned, trace = study.train_noiseless(circuit, theta, target, policy=policy, steps=16, online=0, rate=8)
    assert trace["gradient_evaluations"] == 16
    assert trace["rejected_trials"] > 0
    assert trace["objective_evaluations"] == 1 + sum(r["objective_evaluations"] for r in trace["records"])
    previous_cpu = trace["initial_cpu_seconds"]
    for index, record in enumerate(trace["records"], start=1):
        assert record["cumulative_gradient_evaluations"] == index
        assert record["cumulative_cpu_seconds"] >= previous_cpu
        previous_cpu = record["cumulative_cpu_seconds"]
        assert record["fidelity"] >= record["before_fidelity"]
        assert record["objective_evaluations"] == len(record["trial_rates"])
        if record["accepted"]:
            bound = 1 - record["before_fidelity"] + study.ARMIJO * record["rate"] * record["directional_derivative"]
            assert 1 - record["fidelity"] <= bound + 1e-15
            assert record["rejections"] + 1 == record["objective_evaluations"]
    assert StatevectorBackend(circuit, target).metrics(returned)[1] == trace["returned_fidelity"]
    json.dumps(trace, allow_nan=False)


class QuadraticBackend:
    """Provide an exact positive-curvature oracle independently of quantum gates."""

    def __init__(self, diagonal: np.ndarray) -> None:
        """Store the exact diagonal Hessian and initialize the call counter."""
        self.diagonal = diagonal
        self.gradient_calls = 0

    def metrics(self, theta: np.ndarray) -> tuple[float, float]:
        """Return the quadratic loss and one minus that loss.

        Returns:
            Exact objective and corresponding fidelity-like score.
        """
        loss = float(0.5 * theta @ (self.diagonal * theta))
        return loss, 1 - loss

    def gradient(self, theta: np.ndarray, *_args: object, **_kwargs: object) -> tuple[np.ndarray, float, float]:
        """Count one exact gradient and its jointly computed objective.

        Returns:
            Gradient, exact loss, and one minus the loss.
        """
        self.gradient_calls += 1
        return self.diagonal * theta, *self.metrics(theta)


@pytest.mark.parametrize("policy", ["bb_backtracking", "lbfgs"])
def test_curvature_updates_use_consecutive_true_gradients_without_terminal_extra_call(
    monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    """Validate the first secant pair and direction on an independent curved loss."""
    oracle = QuadraticBackend(np.array([1.0, 10.0]))
    monkeypatch.setattr(study, "StatevectorBackend", lambda *_args: oracle)
    circuit, _, target = case()
    initial = np.array([1.0, 1.0])
    _, trace = study.train_noiseless(circuit, initial, target, policy=policy, steps=8, online=0)
    assert oracle.gradient_calls == trace["gradient_evaluations"] == 8
    first, second = trace["records"][:2]
    displacement = -first["rate"] * oracle.diagonal * initial
    difference = oracle.diagonal * displacement
    assert second["curvature_dot"] == pytest.approx(displacement @ difference)
    assert second["curvature_valid"]
    if policy == "bb_backtracking":
        assert second["initial_trial_rate"] == pytest.approx(displacement @ displacement / (displacement @ difference))
    else:
        assert second["history_pairs"] == 1
        assert second["initial_trial_rate"] == pytest.approx(1.0)
    assert trace["final_fidelity"] > trace["initial_fidelity"] + 5.0


def test_lbfgs_two_loop_reproduces_scalar_inverse_curvature() -> None:
    """For one dimension, the secant pair determines the exact Newton direction."""
    direction = study.two_loop(np.array([8.0]), [np.array([0.5])], [np.array([2.0])])
    np.testing.assert_allclose(direction, [-2.0], atol=1e-15)


def test_lbfgs_memory_is_limited_and_bad_direction_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cap accepted secant memory and replace an ascent direction by steepest descent."""
    circuit, _, target = case()
    oracle = QuadraticBackend(np.arange(1.0, 31.0))
    monkeypatch.setattr(study, "StatevectorBackend", lambda *_args: oracle)
    _, trace = study.train_noiseless(circuit, np.ones(30), target, policy="lbfgs", steps=30, online=0)
    assert max(record["history_pairs"] for record in trace["records"]) == study.LBFGS_HISTORY
    assert trace["final_fidelity"] > 0.99
    monkeypatch.setattr(study, "two_loop", lambda gradient, *_args: gradient.copy())
    _, reset = study.train_noiseless(circuit, np.ones(30), target, policy="lbfgs", steps=3, online=0)
    assert reset["direction_resets"] == 3
    assert all(record["history_pairs"] == 0 for record in reset["records"])
    assert all(record["initial_trial_rate"] == pytest.approx(0.7) for record in reset["records"])
    assert reset["final_fidelity"] > reset["initial_fidelity"]


@pytest.mark.parametrize("policy", ["bb_backtracking", "lbfgs"])
@pytest.mark.parametrize("trial_loss", [1.0, float("nan")])
def test_failed_trials_preserve_state_and_do_not_create_secant_pairs(
    monkeypatch: pytest.MonkeyPatch, policy: str, trial_loss: float
) -> None:
    """An exhausted line search consumes all trials but no invalid curvature pair."""
    circuit, theta, target = case()
    original = StatevectorBackend.metrics
    calls = 0

    def failing(self: StatevectorBackend, value: np.ndarray) -> tuple[float, float]:
        nonlocal calls
        calls += 1
        return original(self, value) if calls == 1 else (trial_loss, 1 - trial_loss)

    monkeypatch.setattr(StatevectorBackend, "metrics", failing)
    returned, trace = study.train_noiseless(circuit, theta, target, policy=policy, steps=2, online=0)
    np.testing.assert_array_equal(returned, theta)
    np.testing.assert_array_equal(trace["final_theta"], theta)
    assert trace["failed_line_searches"] == trace["gradient_evaluations"] == 2
    assert trace["rejected_trials"] == 2 * (study.MAX_BACKTRACKS + 1)
    assert trace["objective_evaluations"] == calls == 1 + trace["rejected_trials"]
    assert trace["curvature_skips"] == 0
    assert all(record["curvature_valid"] is None for record in trace["records"])
    json.dumps(trace, allow_nan=False)


def test_best_retention_includes_initial_even_after_online_damage() -> None:
    """A batch-policy comparison retains the initial checkpoint across all phases."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), 0)], 1)
    theta, target = np.array([2.9]), np.array([0, 1], dtype=complex)
    for policy in study.POLICIES:
        returned, trace = study.train_noiseless(circuit, theta, target, policy=policy, steps=1, online=1, rate=8)
        np.testing.assert_array_equal(returned, theta)
        assert trace["best_iteration"] == 0
        assert trace["final_fidelity"] < trace["initial_fidelity"]


@pytest.mark.parametrize("policy", ["bb_backtracking", "lbfgs"])
def test_zero_gradient_preserves_full_budget_and_rejects_degenerate_curvature(
    monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    """At a stationary point, the method records attempted updates without hidden stopping."""
    circuit, _, target = case()
    oracle = QuadraticBackend(np.array([1.0, 2.0]))
    monkeypatch.setattr(study, "StatevectorBackend", lambda *_args: oracle)
    returned, trace = study.train_noiseless(circuit, np.zeros(2), target, policy=policy, steps=5, online=0)
    np.testing.assert_array_equal(returned, np.zeros(2))
    assert trace["steps"] == trace["gradient_evaluations"] == oracle.gradient_calls == 5
    assert trace["curvature_skips"] == 4
    assert trace["failed_line_searches"] == 0
    json.dumps(trace, allow_nan=False)


@pytest.mark.parametrize("policy", ["bb_backtracking", "lbfgs"])
def test_nonfinite_gradient_never_reaches_trial_propagation(monkeypatch: pytest.MonkeyPatch, policy: str) -> None:
    """Nonfinite directions retain valid checkpoints and JSON-safe diagnostics."""
    circuit, theta, target = case()
    loss, fidelity = StatevectorBackend(circuit, target).metrics(theta)

    def invalid(*_args: object, **_kwargs: object) -> tuple[np.ndarray, float, float]:
        return np.full_like(theta, np.nan), loss, fidelity

    monkeypatch.setattr(StatevectorBackend, "gradient", invalid)
    returned, trace = study.train_noiseless(circuit, theta, target, policy=policy, steps=2, online=0)
    np.testing.assert_array_equal(returned, theta)
    assert trace["failed_line_searches"] == trace["gradient_evaluations"] == 2
    assert trace["objective_evaluations"] == 1
    assert all(record["failure_reason"] == "invalid_descent_direction" for record in trace["records"])
    json.dumps(trace, allow_nan=False)


@pytest.mark.parametrize("policy", study.POLICIES)
def test_empty_budget_and_post_update_stopping_preserve_existing_semantics(policy: str) -> None:
    """No implicit gradient tolerance or pre-update fidelity stopping is introduced."""
    circuit, theta, target = case()
    returned, empty = study.train_noiseless(circuit, theta, target, policy=policy, steps=0)
    np.testing.assert_array_equal(returned, theta)
    assert empty["objective_evaluations"] == 1
    assert empty["gradient_evaluations"] == 0
    assert empty["batch_start_cpu_seconds"] is None
    _, stopped = study.train_noiseless(circuit, theta, target, policy=policy, steps=5, stop=0)
    assert stopped["steps"] == 1


@pytest.mark.parametrize(
    "options", [{"policy": "bad"}, {"steps": -1}, {"online": True}, {"rate": -1}, {"stop": float("nan")}]
)
def test_invalid_configuration_is_rejected(options: dict[str, Any]) -> None:
    """Invalid optimizer options must not silently alter the controlled comparison."""
    circuit, theta, target = case()
    arguments: dict[str, Any] = {"policy": "best_only"}
    arguments.update(options)
    with pytest.raises(ValueError, match=r"policy|budgets|Rate"):
        study.train_noiseless(circuit, theta, target, **arguments)
