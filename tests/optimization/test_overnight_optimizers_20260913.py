# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""L-BFGS mathematical, cumulative-budget and exact-resume regressions."""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pytest

from experiments.overnight_state_preparation_20260913_0108 import optimizers


def quadratic(theta: np.ndarray) -> tuple[float, np.ndarray]:
    """Return a coupled positive-definite objective with a known minimizer."""
    matrix = np.array([[4.0, 1.0], [1.0, 20.0]])
    residual = theta - np.array([0.3, -0.7])
    return float(residual @ matrix @ residual / 2), matrix @ residual


def test_two_loop_matches_explicit_inverse_bfgs_matrix() -> None:
    """Compare the recursion to independent dense inverse-Hessian updates."""
    rng = np.random.default_rng(432)
    matrix = rng.normal(size=(4, 4))
    matrix = matrix.T @ matrix + np.eye(4)
    s_history = [rng.normal(size=4) for _ in range(3)]
    y_history = [matrix @ s for s in s_history]
    scale = float(s_history[-1] @ y_history[-1] / (y_history[-1] @ y_history[-1]))
    inverse = np.eye(4) * scale
    for s, y in zip(s_history, y_history, strict=True):
        rho = 1 / (s @ y)
        left = np.eye(4) - rho * np.outer(s, y)
        inverse = left @ inverse @ left.T + rho * np.outer(s, s)
    gradient = rng.normal(size=4)
    np.testing.assert_allclose(optimizers.two_loop(gradient, s_history, y_history), -inverse @ gradient, atol=1e-14)
    np.testing.assert_array_equal(optimizers.two_loop(gradient, [], []), -gradient)


def test_quadratic_converges_with_monotone_accepted_losses_and_is_json_safe() -> None:
    """The endpoint solves a coupled problem while every callback is accepted."""
    initial = np.array([2.0, 3.0])
    original = initial.copy()
    checkpoints = []
    result = optimizers.minimize(quadratic, initial, max_evaluations=100, callback=checkpoints.append)
    assert result["status"] == "converged"
    np.testing.assert_allclose(result["theta"], [0.3, -0.7], atol=1e-8)
    np.testing.assert_array_equal(initial, original)
    assert result["evaluations"] == 1 + result["iterations"] + result["rejected_evaluations"]
    assert result["evaluations"] <= 100
    assert all(row["pending"] is None for row in checkpoints)
    assert np.all(np.diff([row["loss"] for row in checkpoints]) <= 0)
    assert json.loads(json.dumps(result)) == result
    checkpoints[0]["theta"][0] = 999
    assert result["initial"][0] == 2


def test_callback_interruption_resumes_bit_for_bit_with_history() -> None:
    """Accepted-boundary resume neither restarts memory nor repeats evaluations."""
    initial = np.array([2.0, 3.0])
    full = optimizers.minimize(quadratic, initial, max_evaluations=100)
    saved: dict[str, Any] = {}

    class CheckpointInterruptionError(Exception):
        """Stop exactly after a callback saved an accepted iterate."""

    def checkpoint(row: dict[str, Any]) -> None:
        if row["iterations"] == 3:
            saved.update(row)
            raise CheckpointInterruptionError

    with pytest.raises(CheckpointInterruptionError):
        optimizers.minimize(quadratic, initial, max_evaluations=100, callback=checkpoint)
    assert saved["s_history"]
    serialized = json.loads(json.dumps(saved))
    resumed = optimizers.minimize(quadratic, initial, max_evaluations=100, state=serialized)
    assert resumed == full
    assert serialized == saved
    assert optimizers.minimize(quadratic, initial, max_evaluations=100, state=resumed) == full


def test_hard_cap_and_partially_rejected_line_search_resume() -> None:
    """A spent trial budget preserves the exact next line-search evaluation."""
    initial = np.array([1.0])
    calls = []

    def steep(theta: np.ndarray) -> tuple[float, np.ndarray]:
        calls.append(theta.copy())
        return float(50 * theta @ theta), 100 * theta

    partial = optimizers.minimize(steep, initial, max_evaluations=3)
    assert len(calls) == 3
    assert partial["evaluations"] == 3
    assert partial["iterations"] == 0
    assert partial["rejected_evaluations"] == 2
    assert partial["pending"]["trials"] == 2
    assert partial["status"] == "evaluation_limit"
    resumed = optimizers.minimize(steep, initial, max_evaluations=100, state=partial)
    assert len(calls) == resumed["evaluations"]
    expected = optimizers.minimize(steep, initial, max_evaluations=100)
    assert resumed == expected


def test_negative_curvature_is_skipped_and_failed_trials_are_accounted() -> None:
    """Nonconvex negative secants cannot poison the positive inverse Hessian."""
    result = optimizers.minimize(lambda x: (-float(x @ x), -2 * x), np.ones(1), max_evaluations=4)
    assert result["iterations"] == 3
    assert result["skipped_curvature_pairs"] == 3
    assert not result["s_history"]
    failed = optimizers.minimize(lambda x: (1.0, np.ones_like(x)), np.ones(1), max_evaluations=100)
    assert failed["status"] == "line_search_failed"
    assert failed["evaluations"] == 21
    assert failed["rejected_evaluations"] == 20
    assert failed["theta"] == [1.0]


def test_nonfinite_trials_backtrack_and_preserve_a_valid_accepted_endpoint() -> None:
    """Out-of-domain trial values are counted and rejected without corrupting state."""

    def positive_domain(theta: np.ndarray) -> tuple[float, np.ndarray]:
        return (np.nan, np.full_like(theta, np.nan)) if theta[0] < 0 else (float(theta @ theta), 2 * theta)

    result = optimizers.minimize(positive_domain, np.ones(1), max_evaluations=10)
    assert result["status"] == "converged"
    assert result["nonfinite_evaluations"] == 1
    assert result["rejected_evaluations"] == 1
    assert result["evaluations"] == 3
    assert result["theta"] == [0.0]


@pytest.mark.parametrize("field", ["gradient", "s_history", "config", "evaluations", "initial"])
def test_corrupt_or_incompatible_checkpoints_are_rejected(field: str) -> None:
    """Resume must not reinterpret a changed state or drop curvature memory."""
    initial = np.array([2.0, 3.0])
    state = optimizers.minimize(quadratic, initial, max_evaluations=20, max_iterations=2)
    broken = copy.deepcopy(state)
    if field == "gradient":
        broken[field][0] = np.nan
    elif field == "s_history":
        broken[field] = []
    elif field == "config":
        broken[field]["history"] = 3
    elif field == "evaluations":
        broken[field] += 1
    else:
        broken[field][0] = 999
    with pytest.raises(ValueError, match="checkpoint"):
        optimizers.minimize(quadratic, initial, max_evaluations=100, state=broken)


def test_zero_iteration_budget_initial_value_and_bad_return_shapes() -> None:
    """The initial objective is charged even when no optimizer step is allowed."""
    result = optimizers.minimize(quadratic, np.ones(2), max_evaluations=1, max_iterations=0)
    assert result["evaluations"] == 1
    assert result["iterations"] == 0
    assert result["status"] == "iteration_limit"
    with pytest.raises(ValueError, match="matching initial"):
        optimizers.minimize(lambda _x: (1.0, np.ones(3)), np.ones(2), max_evaluations=5)
    with pytest.raises(ValueError, match="must be finite"):
        optimizers.minimize(lambda x: (np.nan, x), np.ones(2), max_evaluations=5)
