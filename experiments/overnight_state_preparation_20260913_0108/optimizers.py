# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Deterministic resumable L-BFGS with a hard joint objective-call budget.

Known L-BFGS two-loop recursion (Nocedal, Mathematics of Computation 35,
773-782, 1980), adapted with Armijo backtracking, positive-curvature skips,
and transparent checkpoints. This is not SciPy L-BFGS-B or a new algorithm.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

VERSION = "overnight.armijo_lbfgs.v1"
ARMIJO = 1e-4
BACKTRACK = 0.5
MAX_TRIALS = 20
GTOL = 1e-8
CURVATURE_TOL = 1e-10


def two_loop(
    gradient: NDArray[np.float64],
    s_history: list[NDArray[np.float64]],
    y_history: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Return the negative limited-memory inverse-Hessian gradient.

    Histories are oldest first and must contain positive-curvature pairs.
    The initial diagonal is the latest sTy/yTy ratio, clipped to [1e-8,1e8].
    """
    q = gradient.copy()
    alphas = []
    pairs = list(zip(s_history, y_history, strict=True))
    for s, y in reversed(pairs):
        alpha = float(s @ q / (s @ y))
        alphas.append(alpha)
        q -= alpha * y
    scale = float(np.clip(pairs[-1][0] @ pairs[-1][1] / (pairs[-1][1] @ pairs[-1][1]), 1e-8, 1e8)) if pairs else 1.0
    r = scale * q
    for (s, y), alpha in zip(pairs, reversed(alphas), strict=True):
        beta = float(y @ r / (s @ y))
        r += s * (alpha - beta)
    return -r


def _result(
    objective: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]], theta: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64]]:
    """Evaluate on private parameters and require a scalar real loss/gradient.

    Nonfinite values are returned to the line search for explicit rejection.

    Raises:
        ValueError: If the return shape or number type is invalid.
    """
    value, raw_gradient = objective(theta.copy())
    gradient = np.asarray(raw_gradient)
    if not np.isrealobj(value) or np.ndim(value) != 0 or not np.isrealobj(gradient) or gradient.shape != theta.shape:
        msg = "Objective must return a real scalar and real gradient matching initial parameters."
        raise ValueError(msg)
    return float(value), gradient.astype(float, copy=True)


def _config(history: int) -> dict[str, float | int]:
    """Return every fixed numerical setting affecting the optimizer path."""
    return {
        "history": history,
        "gtol": GTOL,
        "armijo": ARMIJO,
        "backtrack": BACKTRACK,
        "max_trials": MAX_TRIALS,
        "curvature_tolerance": CURVATURE_TOL,
    }


def _state_valid(state: dict[str, Any], initial: NDArray[np.float64], history: int) -> bool:
    """Check numerical checkpoint fields, allowing malformed fields to raise."""
    valid = (
        state["version"] == VERSION
        and state["config"] == _config(history)
        and np.array_equal(initial, np.asarray(state["initial"]))
        and len(state["s_history"]) == len(state["y_history"]) <= history
        and np.isfinite(state["loss"])
    )
    for key in ("theta", "gradient", "initial"):
        array = np.asarray(state[key])
        valid = valid and array.shape == initial.shape and np.isrealobj(array) and bool(np.all(np.isfinite(array)))
    for key in (
        "iterations",
        "evaluations",
        "rejected_evaluations",
        "nonfinite_evaluations",
        "skipped_curvature_pairs",
        "resets",
    ):
        valid = valid and type(state[key]) is int and state[key] >= 0
    valid = valid and state["evaluations"] == 1 + state["iterations"] + state["rejected_evaluations"]
    for raw_s, raw_y in zip(state["s_history"], state["y_history"], strict=True):
        s, y = np.asarray(raw_s), np.asarray(raw_y)
        valid = valid and s.shape == y.shape == initial.shape and np.all(np.isfinite(s)) and np.all(np.isfinite(y))
        valid = valid and float(s @ y) > CURVATURE_TOL * np.linalg.norm(s) * np.linalg.norm(y)
    pending = state["pending"]
    if pending is not None:
        direction = np.asarray(pending["direction"])
        valid = valid and direction.shape == initial.shape and np.all(np.isfinite(direction))
        valid = valid and type(pending["trials"]) is int and 0 <= pending["trials"] < MAX_TRIALS
        valid = valid and pending["step"] == BACKTRACK ** pending["trials"]
        valid = valid and np.isfinite(pending["slope"]) and pending["slope"] < 0
    valid = valid and state["status"] in {
        "running",
        "converged",
        "evaluation_limit",
        "iteration_limit",
        "line_search_failed",
    }
    return bool(valid)


def _validate_state(state: dict[str, Any], initial: NDArray[np.float64], history: int) -> None:
    """Reject incompatible or malformed numerical checkpoints.

    The runner must separately authenticate objective/circuit/source identities.

    Raises:
        ValueError: If state is incompatible, nonfinite or structurally invalid.
    """
    try:
        valid = _state_valid(state, initial, history)
    except (KeyError, TypeError, ValueError, IndexError):
        valid = False
    if not valid:
        msg = "L-BFGS checkpoint is malformed or incompatible with initial parameters/configuration."
        raise ValueError(msg)


def minimize(
    objective: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]],
    initial: NDArray[np.float64],
    *,
    max_evaluations: int,
    max_iterations: int = 100,
    history: int = 10,
    state: dict[str, Any] | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Minimize a deterministic objective and return the final accepted iterate.

    Args:
        objective: Callable returning joint loss and gradient for given theta.
        initial: Finite real initial parameters, also identifying resumed runs.
        max_evaluations: Hard cumulative joint-call cap, including rejections.
        max_iterations: Hard cumulative accepted-step cap.
        history: Maximum retained positive-curvature secant pairs.
        state: Complete previous plain-dictionary checkpoint, copied on entry.
        callback: Receives private complete accepted-iterate checkpoints,
            including initial iteration zero. Raising interrupts at this boundary.

    Returns:
        Plain JSON-compatible state containing parameters, objective, gradient,
        all secant pairs, counters, status and any unfinished line-search trial.
        Raising a budget and resuming even a partially spent line search uses
        the same next evaluation as an uninterrupted run. Callback checkpoints
        always have pending=None and represent complete accepted boundaries.

    Raises:
        ValueError: If arguments/checkpoint are invalid or the initial joint
            objective evaluation is nonfinite.
    """
    initial = np.asarray(initial)
    if (
        initial.ndim != 1
        or not np.isrealobj(initial)
        or not np.all(np.isfinite(initial))
        or type(max_evaluations) is not int
        or max_evaluations < 1
        or type(max_iterations) is not int
        or max_iterations < 0
        or type(history) is not int
        or history < 1
    ):
        msg = "Invalid L-BFGS parameters, history or cumulative iteration/evaluation budgets."
        raise ValueError(msg)
    initial = initial.astype(float, copy=True)
    if state is None:
        loss, gradient = _result(objective, initial)
        if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
            msg = "Initial L-BFGS objective and gradient must be finite."
            raise ValueError(msg)
        current: dict[str, Any] = {
            "version": VERSION,
            "config": _config(history),
            "initial": initial.tolist(),
            "theta": initial.tolist(),
            "loss": loss,
            "gradient": gradient.tolist(),
            "s_history": [],
            "y_history": [],
            "iterations": 0,
            "evaluations": 1,
            "rejected_evaluations": 0,
            "nonfinite_evaluations": 0,
            "skipped_curvature_pairs": 0,
            "resets": 0,
            "pending": None,
            "status": "running",
        }
        if callback is not None:
            callback(copy.deepcopy(current))
    else:
        _validate_state(state, initial, history)
        current = copy.deepcopy(state)
        if current["evaluations"] > max_evaluations or current["iterations"] > max_iterations:
            msg = "Resume budgets cannot be below already consumed work."
            raise ValueError(msg)
        if current["status"] in {"converged", "line_search_failed"}:
            return current
        current["status"] = "running"
    while True:
        theta, gradient = np.array(current["theta"]), np.array(current["gradient"])
        if not gradient.size or np.linalg.norm(gradient, ord=np.inf) <= GTOL:
            current["status"] = "converged"
            break
        if current["iterations"] >= max_iterations:
            current["status"] = "iteration_limit"
            break
        if current["evaluations"] >= max_evaluations:
            current["status"] = "evaluation_limit"
            break
        if current["pending"] is None:
            direction = two_loop(
                gradient, [np.array(s) for s in current["s_history"]], [np.array(y) for y in current["y_history"]]
            )
            slope = float(gradient @ direction)
            if not np.all(np.isfinite(direction)) or slope >= -1e-12 * np.linalg.norm(gradient) * np.linalg.norm(
                direction
            ):
                current["s_history"], current["y_history"] = [], []
                current["resets"] += 1
                direction, slope = -gradient, -float(gradient @ gradient)
            current["pending"] = {"direction": direction.tolist(), "step": 1.0, "trials": 0, "slope": slope}
        pending = current["pending"]
        trial_theta = theta + pending["step"] * np.asarray(pending["direction"])
        trial_loss, trial_gradient = _result(objective, trial_theta)
        current["evaluations"] += 1
        finite = np.isfinite(trial_loss) and np.all(np.isfinite(trial_gradient))
        accepted = finite and trial_loss <= current["loss"] + ARMIJO * pending["step"] * pending["slope"]
        if accepted:
            s, y = trial_theta - theta, trial_gradient - gradient
            if float(s @ y) > CURVATURE_TOL * np.linalg.norm(s) * np.linalg.norm(y):
                current["s_history"] = [*current["s_history"], s.tolist()][-history:]
                current["y_history"] = [*current["y_history"], y.tolist()][-history:]
            else:
                current["skipped_curvature_pairs"] += 1
            current.update(theta=trial_theta.tolist(), loss=trial_loss, gradient=trial_gradient.tolist(), pending=None)
            current["iterations"] += 1
            if callback is not None:
                callback(copy.deepcopy(current))
        else:
            current["rejected_evaluations"] += 1
            current["nonfinite_evaluations"] += int(not finite)
            pending["trials"] += 1
            pending["step"] *= BACKTRACK
            if pending["trials"] >= MAX_TRIALS:
                current["status"] = "line_search_failed"
                current["pending"] = None
                break
    return current
