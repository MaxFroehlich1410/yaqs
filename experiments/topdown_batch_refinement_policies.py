# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compare exact-gradient batch refinements after an unchanged online prefix.

The two controls reproduce the frozen optimizer-stability arithmetic. BB changes
the initial Armijo trial rate; L-BFGS additionally changes the search direction.
All curvature information comes from true gradients at accepted batch states.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from experiments.overnight_state_preparation_20260913_0108.optimizers import two_loop
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import StatevectorBackend

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit
    from mqt.yaqs.optimization.backends import PreparationBackend

POLICIES = ("best_only", "best_backtracking", "bb_backtracking", "lbfgs")
ARMIJO = 1e-4
MAX_BACKTRACKS = 12
CURVATURE_TOL = 1e-10
BB_MIN_RATE = 1e-6
BB_MAX_RATE = 10.0
LBFGS_HISTORY = 10


def curvature_info(displacement: NDArray, difference: NDArray) -> dict[str, Any]:
    """Check a secant pair before using its positive curvature estimate."""
    with np.errstate(over="ignore", invalid="ignore"):
        displacement_norm = float(np.linalg.norm(displacement))
        difference_norm = float(np.linalg.norm(difference))
        curvature = float(displacement @ difference)
        threshold = float(CURVATURE_TOL * displacement_norm * difference_norm)
    valid = (
        np.all(np.isfinite(displacement))
        and np.all(np.isfinite(difference))
        and np.isfinite(displacement_norm)
        and np.isfinite(difference_norm)
        and displacement_norm > 0
        and difference_norm > 0
        and np.isfinite(curvature)
        and np.isfinite(threshold)
        and curvature > threshold
    )
    return {
        "curvature_valid": bool(valid),
        "curvature_dot": curvature if np.isfinite(curvature) else None,
        "curvature_threshold": threshold if np.isfinite(threshold) else None,
    }


def bb_initial_rate(displacement: NDArray, difference: NDArray, fallback: float) -> tuple[float, dict[str, Any]]:
    """Return safeguarded BB1 or the previous accepted rate after a bad pair.

    Bounds apply to the initial trial rate. Armijo may subsequently accept a
    smaller rate. Angles must be unwrapped, and pairs must join accepted states.
    """
    info = curvature_info(displacement, difference)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw = float(displacement @ displacement / info["curvature_dot"]) if info["curvature_valid"] else float("nan")
    usable = bool(info["curvature_valid"] and np.isfinite(raw) and raw > 0)
    candidate = raw if usable else fallback
    rate = float(np.clip(candidate, BB_MIN_RATE, BB_MAX_RATE))
    info.update({
        "bb_raw_rate": raw if np.isfinite(raw) else None,
        "bb_fallback_used": not usable,
        "bb_bound_hit": bool(rate != candidate),
        "initial_trial_rate": rate,
    })
    return rate, info


def _line_search(
    engine: PreparationBackend,
    theta: NDArray,
    direction: NDArray,
    loss: float,
    fidelity: float,
    slope: float,
    rate: float,
) -> tuple[NDArray, float, dict[str, Any]]:
    """Perform bounded Armijo trials without recomputing the true gradient."""
    info: dict[str, Any] = {
        "accepted": False,
        "rate": 0.0,
        "objective_evaluations": 0,
        "rejections": 0,
        "trial_rates": [],
        "trial_losses": [],
        "failure_reason": None,
    }
    if not np.all(np.isfinite(direction)) or not np.isfinite(slope) or slope > 0:
        info["failure_reason"] = "invalid_descent_direction"
        return theta, fidelity, info
    for attempt in range(MAX_BACKTRACKS + 1):
        trial_rate = rate * 0.5**attempt
        info["trial_rates"].append(float(trial_rate))
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = theta + trial_rate * direction
            required_loss = loss + ARMIJO * trial_rate * slope
        candidate_loss, candidate_fidelity = float("nan"), float("nan")
        if np.all(np.isfinite(candidate)):
            candidate_loss, candidate_fidelity = engine.metrics(candidate)
            info["objective_evaluations"] += 1
        info["trial_losses"].append(float(candidate_loss) if np.isfinite(candidate_loss) else None)
        if np.isfinite(candidate_loss) and np.isfinite(candidate_fidelity) and candidate_loss <= required_loss:
            info.update({"accepted": True, "rate": float(trial_rate)})
            return candidate, float(candidate_fidelity), info
        info["rejections"] += 1
    info["failure_reason"] = "trial_budget_exhausted"
    return theta, fidelity, info


def train_noiseless(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128] | MPS,
    *,
    policy: str,
    steps: int = 200,
    online: int = 100,
    rate: float = 0.7,
    stop: float = 2.0,
    engine: PreparationBackend | None = None,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Retain the best iterate while comparing four noiseless batch policies.

    Args:
        circuit: Circuit whose parameter order and topology stay fixed.
        theta: Initial unwrapped angles; the input array is never changed.
        target: Target state vector.
        policy: One of ``POLICIES``.
        engine: Optional numerical backend; defaults to the archived dense implementation.
        steps: Total attempted updates, including failed line searches.
        online: Initial unchanged sequential sweeps, without curvature history.
        rate: Online rate, control batch rate, and initial BB/L-BFGS rate.
        stop: Post-update fidelity stopping threshold; two disables early exit.

    Returns:
        Best actual parameters, including iteration zero, and JSON diagnostics.
        Standalone metrics calls and batch gradient calls are counted separately;
        each gradient call also provides the current loss. Rejected trials count.
        Curvature uses the next iteration's gradient, never an extra terminal
        gradient. CPU curves include all training diagnostics and checkpoint
        copies, but no artifact serialization. No angle wrapping is performed.

    Raises:
        ValueError: If the policy, budgets, rate, or threshold are invalid.
    """
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    if policy not in POLICIES:
        msg = f"Unknown batch-refinement policy: {policy}."
        raise ValueError(msg)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (steps, online)):
        msg = "Update budgets must be nonnegative integers."
        raise ValueError(msg)
    if isinstance(rate, bool) or not np.isfinite(rate) or rate < 0 or not np.isfinite(stop):
        msg = "Rate must be finite and nonnegative; stopping threshold must be finite."
        raise ValueError(msg)
    theta = theta.copy()
    initial_theta = theta.copy()
    if engine is None:
        if isinstance(target, MPS):
            msg = "MPS targets require an explicit preparation backend."
            raise ValueError(msg)
        engine = StatevectorBackend(circuit, target)
    _, initial = engine.metrics(theta)
    best_fidelity, final_fidelity, online_final = initial, initial, initial
    best_theta, online_final_theta = theta.copy(), theta.copy()
    best_iteration = 0
    fidelity_trace: list[float] = []
    records: list[dict[str, Any]] = []
    objective_evaluations, gradient_evaluations, online_sweeps = 1, 0, 0
    batch_start_cpu = None
    initial_cpu = time.process_time() - cpu_start
    last_accepted_rate = rate
    pending: tuple[NDArray, NDArray] | None = None
    s_history: list[NDArray] = []
    y_history: list[NDArray] = []
    for iteration in range(1, steps + 1):
        is_online = iteration <= online
        phase_iteration = iteration if is_online else iteration - online
        step = rate * np.exp(-0.0 * (phase_iteration - 1))
        info: dict[str, Any] = {
            "iteration": iteration,
            "phase": "online" if is_online else "batch",
            "before_fidelity": float(final_fidelity),
            "accepted": True,
            "rate": float(step),
            "objective_evaluations": 1,
            "gradient_evaluations": int(not is_online),
            "rejections": 0,
            "trial_rates": [],
            "trial_losses": [],
            "failure_reason": None,
            "curvature_valid": None,
            "curvature_dot": None,
            "curvature_threshold": None,
            "curvature_skipped": False,
            "bb_raw_rate": None,
            "bb_fallback_used": False,
            "bb_bound_hit": False,
            "direction_reset": False,
            "history_pairs": 0,
            "initial_trial_rate": float(step),
        }
        if is_online:
            theta, signal = engine.online_update(theta, step, None, estimator="independent", weights=None)
            final_fidelity = engine.metrics(theta)[1]
            online_final, online_final_theta = final_fidelity, theta.copy()
            online_sweeps += 1
            info["gradient_norm"] = None
            info["signal_norm"] = float(np.linalg.norm(signal))
            info["directional_derivative"] = None
        else:
            if batch_start_cpu is None:
                batch_start_cpu = time.process_time() - cpu_start
            gradient, loss, before_fidelity = engine.gradient(theta, None, estimator="independent", weights=None)
            gradient_evaluations += 1
            info["before_fidelity"] = float(before_fidelity)
            with np.errstate(over="ignore", invalid="ignore"):
                norm = float(np.linalg.norm(gradient))
                slope = -float(gradient @ gradient)
            info["gradient_norm"] = norm if np.isfinite(norm) else None
            info["signal_norm"] = info["gradient_norm"]
            direction = -gradient
            if pending is not None:
                old_theta, old_gradient = pending
                displacement, difference = theta - old_theta, gradient - old_gradient
                if policy == "bb_backtracking":
                    step, pair_info = bb_initial_rate(displacement, difference, last_accepted_rate)
                    info.update(pair_info)
                    info["curvature_skipped"] = not pair_info["curvature_valid"]
                elif policy == "lbfgs":
                    pair_info = curvature_info(displacement, difference)
                    info.update(pair_info)
                    if pair_info["curvature_valid"]:
                        s_history.append(displacement.copy())
                        y_history.append(difference.copy())
                        s_history, y_history = s_history[-LBFGS_HISTORY:], y_history[-LBFGS_HISTORY:]
                    else:
                        info["curvature_skipped"] = True
                pending = None
            elif policy == "bb_backtracking":
                step = float(np.clip(last_accepted_rate, BB_MIN_RATE, BB_MAX_RATE))
                info["bb_bound_hit"] = bool(step != last_accepted_rate)
            if policy == "lbfgs":
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    direction = two_loop(gradient, s_history, y_history)
                    slope = float(gradient @ direction)
                if not np.all(np.isfinite(direction)) or not np.isfinite(slope) or slope >= 0:
                    s_history, y_history = [], []
                    direction = -gradient
                    with np.errstate(over="ignore", invalid="ignore"):
                        slope = -float(gradient @ gradient)
                    info["direction_reset"] = True
                step = 1.0 if s_history else rate
            info["history_pairs"] = len(s_history)
            info["initial_trial_rate"] = float(step)
            info["directional_derivative"] = slope if np.isfinite(slope) else None
            if policy == "best_only":
                theta -= step * gradient
                final_fidelity = engine.metrics(theta)[1]
            else:
                old_theta = theta
                theta, final_fidelity, line_info = _line_search(
                    engine, theta, direction, loss, before_fidelity, slope, float(step)
                )
                info.update(line_info)
                if info["accepted"] and policy in {"bb_backtracking", "lbfgs"}:
                    pending = old_theta.copy(), gradient.copy()
                    last_accepted_rate = info["rate"]
        objective_evaluations += info["objective_evaluations"]
        fidelity_trace.append(float(final_fidelity))
        if final_fidelity > best_fidelity:
            best_fidelity, best_iteration, best_theta = final_fidelity, iteration, theta.copy()
        info.update({
            "fidelity": float(final_fidelity),
            "best_fidelity": float(best_fidelity),
            "best_iteration": best_iteration,
            "cumulative_objective_evaluations": objective_evaluations,
            "cumulative_gradient_evaluations": gradient_evaluations,
            "cumulative_online_sweeps": online_sweeps,
        })
        records.append(info)
        info["cumulative_cpu_seconds"] = time.process_time() - cpu_start
        info["cumulative_wall_seconds"] = time.perf_counter() - wall_start
        if final_fidelity >= stop:
            break
    return best_theta.copy(), {
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
        "initial_theta": initial_theta.tolist(),
        "initial_cpu_seconds": initial_cpu,
        "batch_start_cpu_seconds": batch_start_cpu,
        "online_final_fidelity": float(online_final),
        "online_final_theta": online_final_theta.tolist(),
        "best_fidelity": float(best_fidelity),
        "final_fidelity": float(final_fidelity),
        "returned_fidelity": float(best_fidelity),
        "best_iteration": best_iteration,
        "best_theta": best_theta.tolist(),
        "final_theta": theta.tolist(),
        "returned_theta": best_theta.tolist(),
        "records": records,
        "objective_evaluations": objective_evaluations,
        "gradient_evaluations": gradient_evaluations,
        "online_sweeps": online_sweeps,
        "rejected_trials": sum(record["rejections"] for record in records),
        "failed_line_searches": sum(not record["accepted"] for record in records),
        "curvature_skips": sum(record["curvature_skipped"] for record in records),
        "bb_fallbacks": sum(record["bb_fallback_used"] for record in records),
        "bb_bound_hits": sum(record["bb_bound_hit"] for record in records),
        "direction_resets": sum(record["direction_reset"] for record in records),
        "armijo": ARMIJO if policy != "best_only" else None,
        "max_backtracks": MAX_BACKTRACKS if policy != "best_only" else None,
        "curvature_tolerance": CURVATURE_TOL if policy in {"bb_backtracking", "lbfgs"} else None,
        "bb_rate_bounds": [BB_MIN_RATE, BB_MAX_RATE] if policy == "bb_backtracking" else None,
        "lbfgs_history_limit": LBFGS_HISTORY if policy == "lbfgs" else None,
        "cpu_seconds": time.process_time() - cpu_start,
        "wall_seconds": time.perf_counter() - wall_start,
    }
