# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Controlled noiseless checkpoint-retention and batch-stability comparison.

The fixed-step arms preserve the archived arithmetic and accept every update.
All arms use the same gate-sequential online prefix. Backtracking only changes
the simultaneous batch phase; noisy training and pruning are outside this module.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from mqt.yaqs.optimization import StatevectorBackend

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

POLICIES = ("original_final", "best_only", "best_backtracking")
ARMIJO = 1e-4
MAX_BACKTRACKS = 12


def _backtracking_step(
    engine: StatevectorBackend,
    theta: NDArray[np.float64],
    gradient: NDArray[np.float64],
    before: float,
    fidelity: float,
    rate: float,
) -> tuple[NDArray[np.float64], float, dict[str, Any]]:
    """Reuse one exact loss gradient for at most thirteen forward trials."""
    info: dict[str, Any] = {
        "accepted": False,
        "rate": 0.0,
        "objective_evaluations": 0,
        "rejections": 0,
        "trial_rates": [],
        "trial_losses": [],
        "failure_reason": None,
    }
    with np.errstate(over="ignore", invalid="ignore"):
        norm_squared = float(gradient @ gradient)
    if not np.all(np.isfinite(gradient)) or not np.isfinite(norm_squared):
        info["failure_reason"] = "nonfinite_gradient"
        return theta, fidelity, info
    for attempt in range(MAX_BACKTRACKS + 1):
        trial_rate = rate * 0.5**attempt
        info["trial_rates"].append(float(trial_rate))
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = theta - trial_rate * gradient
            required_loss = before - ARMIJO * trial_rate * norm_squared
        loss = float("nan")
        candidate_fidelity = float("nan")
        if np.all(np.isfinite(candidate)):
            loss, candidate_fidelity = engine.metrics(candidate)
            info["objective_evaluations"] += 1
        info["trial_losses"].append(float(loss) if np.isfinite(loss) else None)
        if np.isfinite(loss) and np.isfinite(candidate_fidelity) and loss <= required_loss:
            info.update({"accepted": True, "rate": float(trial_rate)})
            return candidate, float(candidate_fidelity), info
        info["rejections"] += 1
    info["failure_reason"] = "trial_budget_exhausted"
    return theta, fidelity, info


def train_noiseless(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    *,
    policy: str,
    steps: int = 200,
    online: int = 100,
    rate: float = 0.7,
    stop: float = 2.0,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Train one fixed circuit with a controlled noiseless optimizer policy.

    Args:
        circuit: Parameterized circuit, unchanged by this function.
        theta: Initial parameters; caller-owned values are never modified.
        target: Target state vector.
        policy: ``original_final``, ``best_only``, or ``best_backtracking``.
        steps: Maximum update attempts, including failed batch line searches.
        online: Number of initial gate-sequential sweeps.
        rate: Fixed online rate and initial batch trial rate.
        stop: Stop after an update reaches this fidelity, as in the archive.
            The default exceeds one and therefore uses the full update budget.

    Returns:
        Returned parameters and a JSON-serializable trace. ``fidelity_trace``
        contains post-update fidelities, excluding iteration zero. Best-iterate
        selection includes iteration zero and never rolls back the active loop.
        ``objective_evaluations`` counts standalone ``metrics`` calls, including
        the initial call and rejected trials. ``gradient_evaluations`` counts
        batch calls, which also compute the baseline objective. Online sweeps
        are counted separately; their accumulated signal is not a batch gradient.
        Timings include initial scoring, diagnostics, and checkpoint copying.

    Raises:
        ValueError: If the policy, budgets, rate, or threshold are invalid.
    """
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    if policy not in POLICIES:
        msg = f"Unknown noiseless optimizer policy: {policy}."
        raise ValueError(msg)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (steps, online)):
        msg = "Update budgets must be nonnegative integers."
        raise ValueError(msg)
    if isinstance(rate, bool) or not np.isfinite(rate) or rate < 0 or not np.isfinite(stop):
        msg = "Rate must be finite and nonnegative; stopping threshold must be finite."
        raise ValueError(msg)
    theta = theta.copy()
    engine = StatevectorBackend(circuit, target)
    _, initial = engine.metrics(theta)
    best_fidelity, final_fidelity, online_final = initial, initial, initial
    best_theta = theta.copy()
    online_final_theta = theta.copy()
    best_iteration = 0
    fidelity_trace: list[float] = []
    records: list[dict[str, Any]] = []
    objective_evaluations = 1
    gradient_evaluations = 0
    online_sweeps = 0
    for iteration in range(1, steps + 1):
        is_online = iteration <= online
        phase_iteration = iteration if is_online else iteration - online
        # Match the archived zero-decay arithmetic, including the in-place batch update.
        step = rate * np.exp(-0.0 * (phase_iteration - 1))
        before_fidelity = final_fidelity
        info: dict[str, Any] = {
            "iteration": iteration,
            "phase": "online" if is_online else "batch",
            "before_fidelity": before_fidelity,
            "accepted": True,
            "rate": float(step),
            "objective_evaluations": 1,
            "gradient_evaluations": int(not is_online),
            "rejections": 0,
            "trial_rates": [],
            "trial_losses": [],
            "failure_reason": None,
        }
        if is_online:
            theta, signal = engine.online_update(theta, step, None, estimator="independent", weights=None)
            final_fidelity = engine.metrics(theta)[1]
            online_final = final_fidelity
            online_final_theta = theta.copy()
            online_sweeps += 1
            info["gradient_norm"] = None
            info["signal_norm"] = float(np.linalg.norm(signal))
        else:
            gradient, loss, before_fidelity = engine.gradient(theta, None, estimator="independent", weights=None)
            gradient_evaluations += 1
            info["before_fidelity"] = before_fidelity
            with np.errstate(over="ignore", invalid="ignore"):
                gradient_norm = float(np.linalg.norm(gradient))
            info["gradient_norm"] = gradient_norm if np.isfinite(gradient_norm) else None
            info["signal_norm"] = info["gradient_norm"]
            if policy == "best_backtracking":
                theta, final_fidelity, line_info = _backtracking_step(
                    engine, theta, gradient, loss, before_fidelity, float(step)
                )
                info.update(line_info)
            else:
                theta -= step * gradient
                final_fidelity = engine.metrics(theta)[1]
        objective_evaluations += info["objective_evaluations"]
        info["fidelity"] = float(final_fidelity)
        records.append(info)
        fidelity_trace.append(float(final_fidelity))
        if final_fidelity > best_fidelity:
            best_fidelity, best_iteration = final_fidelity, iteration
            best_theta = theta.copy()
        if final_fidelity >= stop:
            break
    returned = theta.copy() if policy == "original_final" else best_theta.copy()
    returned_fidelity = final_fidelity if policy == "original_final" else best_fidelity
    trace = {
        "policy": policy,
        "steps": len(fidelity_trace),
        "requested_steps": steps,
        "online_steps": online_sweeps,
        "rate": rate,
        "decay": 0.0,
        "noisy": False,
        "estimator": "independent",
        "fidelity_trace": fidelity_trace,
        "initial_fidelity": float(initial),
        "online_final_fidelity": float(online_final),
        "online_final_theta": online_final_theta.tolist(),
        "best_fidelity": float(best_fidelity),
        "final_fidelity": float(final_fidelity),
        "returned_fidelity": float(returned_fidelity),
        "best_iteration": best_iteration,
        "best_theta": best_theta.tolist(),
        "final_theta": theta.tolist(),
        "returned_theta": returned.tolist(),
        "records": records,
        "objective_evaluations": objective_evaluations,
        "gradient_evaluations": gradient_evaluations,
        "online_sweeps": online_sweeps,
        "rejected_trials": sum(record["rejections"] for record in records),
        "failed_line_searches": sum(not record["accepted"] for record in records),
        "armijo": ARMIJO if policy == "best_backtracking" else None,
        "max_backtracks": MAX_BACKTRACKS if policy == "best_backtracking" else None,
        "cpu_seconds": time.process_time() - cpu_start,
        "wall_seconds": time.perf_counter() - wall_start,
    }
    return returned, trace
