# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Pure-state geometric preconditioning for the fixed noisy density objective.

This is the known real quantum geometric tensor/Fubini-Study metric, used as
a surrogate preconditioner. It is not the quantum Fisher geometry of the noisy
mixed state. No new algorithm or mixed-state natural-gradient claim is made.
"""

from __future__ import annotations

import copy
import hashlib
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from experiments import state_preparation_pilot as pilot

# Reuse the validated local batched-state kernel in this isolated experiment.
# PLC2701 is the private-import rule in the repository's pinned Ruff version.
from mqt.yaqs.optimization.statevector import _apply  # noqa: PLC2701

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

VERSION = "overnight.pure_qgt_noisy_armijo.v1"
DAMPINGS = (0.001, 0.01)
ARMIJO = 1e-4
BACKTRACK = 0.5
MAX_TRIALS = 20
GTOL = 1e-8


def _validate(circuit: ParameterizedCircuit, theta: NDArray) -> None:
    """Restrict the exact derivative kernel to finite scalar q1-q6 circuits.

    Raises:
        ValueError: If a circuit or parameter vector is unsupported.
    """
    if (
        type(circuit.num_qubits) is not int
        or not 1 <= circuit.num_qubits <= 6
        or theta.shape != (circuit.num_params,)
        or not np.isrealobj(theta)
        or not np.all(np.isfinite(theta))
        or any(
            gate.data_map is not None
            or not np.isrealobj(gate.angle_scale)
            or not np.isrealobj(gate.angle_offset)
            or not np.isrealobj(gate.fixed_params)
            or not np.isfinite(gate.angle_scale)
            or not np.isfinite(gate.angle_offset)
            or not np.all(np.isfinite(gate.fixed_params))
            for gate in circuit.gates
        )
    ):
        msg = "Pure geometry requires a finite real scalar q1-q6 circuit without data maps."
        raise ValueError(msg)


def state_jacobian(circuit: ParameterizedCircuit, theta: NDArray) -> tuple[NDArray, NDArray, dict[str, Any]]:
    """Propagate the exact pure state and all first derivatives from |0...0>.

    Args:
        circuit: Logical ideal circuit; physical noise flags do not affect its metric.
        theta: Parameters including shared occurrences, offsets and signed scales.

    Returns:
        Final state, Jacobian rows ``jacobian[i] = d state / d theta[i]``, and
        measured cost. One batch propagates the state and all tangent rows.
        Derivative insertions use dU/dangle = D U and accumulate shared indices.
        Cost is O(L P 2**n), with (P+1)*L row-operator applications plus one
        insertion for each nonzero-scale trainable gate occurrence.

    Raises:
        ValueError: If inputs are outside the exact small-system domain.
        FloatingPointError: If state normalization or derivative finiteness fails.
    """
    _validate(circuit, theta)
    cpu, wall = time.process_time(), time.perf_counter()
    n, p = circuit.num_qubits, circuit.num_params
    batch = np.zeros((p + 1, 2**n), dtype=complex)
    batch[0, 0] = 1
    insertions = 0
    for gate in circuit.gates:
        unitary, sites = circuit.gate_matrix(gate, theta)
        batch = _apply(batch, unitary, sites, n)
        if gate.param_index is not None and gate.angle_scale != 0:
            derivative, derivative_sites = circuit.derivative_operator(gate)
            batch[gate.param_index + 1] += gate.angle_scale * _apply(batch[:1], derivative, derivative_sites, n)[0]
            insertions += 1
    if not np.all(np.isfinite(batch)) or abs(np.vdot(batch[0], batch[0]) - 1) > 1e-10:
        msg = "Pure-state Jacobian failed finite or state-normalization checks."
        raise FloatingPointError(msg)
    result: dict[str, Any] = {
        "batch_unitary_kernel_calls": len(circuit.gates),
        "derivative_insertion_kernel_calls": insertions,
        "statevector_row_operator_applications": (p + 1) * len(circuit.gates) + insertions,
        "propagated_rows": p + 1,
        "batch_array_bytes": batch.nbytes,
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall,
        "stochastic_trajectory_paths": 0,
    }
    return batch[0].copy(), batch[1:].copy(), result


def metric_from_jacobian(state: NDArray, jacobian: NDArray) -> NDArray:
    """Compute the real Fubini-Study tensor with no factor-of-four convention.

    G_ij = Re(<di|dj> - <di|psi><psi|dj>). Projecting out the state direction
    before the Gram product is numerically stable and removes parameter-dependent
    global phases. The pure-state quantum Fisher information would be 4G.

    Returns:
        Real symmetric positive-semidefinite parameter metric, up to roundoff.

    Raises:
        ValueError: If the state/Jacobian dimensions or normalization are invalid.
    """
    if (
        state.ndim != 1
        or not state.size
        or jacobian.ndim != 2
        or jacobian.shape[1] != state.size
        or not np.all(np.isfinite(state))
        or not np.all(np.isfinite(jacobian))
        or abs(np.vdot(state, state) - 1) > 1e-10
    ):
        msg = "Metric requires a finite normalized state and parameter-by-state Jacobian."
        raise ValueError(msg)
    horizontal = jacobian - (jacobian @ state.conj())[:, None] * state[None, :]
    gram = np.real(horizontal.conj() @ horizontal.T)
    return (gram + gram.T) / 2


def pure_state_metric(circuit: ParameterizedCircuit, theta: NDArray) -> tuple[NDArray, dict[str, Any]]:
    """Construct the exact pure-state metric and charge Jacobian and Gram work.

    Returns:
        Metric and complete construction timing plus explicit array/kernel costs.
    """
    cpu, wall = time.process_time(), time.perf_counter()
    state, jacobian, diagnostics = state_jacobian(circuit, theta)
    metric = metric_from_jacobian(state, jacobian)
    diagnostics.update({
        "jacobian_cpu_seconds": diagnostics["cpu_seconds"],
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall,
        "metric_array_bytes": metric.nbytes,
        "gram_shape": list(metric.shape),
        "metric_identity": "real_pure_state_QGT_without_factor_four",
    })
    return metric, diagnostics


def precondition(gradient: NDArray, metric: NDArray, damping: float) -> tuple[NDArray, dict[str, Any]]:
    """Solve (G + damping I) direction = -gradient for one preregistered damping.

    Returns:
        Descent direction and solve timing/conditioning. Metric construction
        remains separately charged. The real symmetric metric is never inverted.

    Raises:
        ValueError: If shapes, damping, finiteness, symmetry or PSD are invalid.
        FloatingPointError: If the damped solve fails finite descent checks.
    """
    cpu, wall = time.process_time(), time.perf_counter()
    if (
        isinstance(damping, bool)
        or damping not in DAMPINGS
        or gradient.ndim != 1
        or metric.shape != (gradient.size, gradient.size)
        or not np.isrealobj(gradient)
        or not np.isrealobj(metric)
        or not np.all(np.isfinite(gradient))
        or not np.all(np.isfinite(metric))
        or (metric.size and np.max(abs(metric - metric.T)) > 1e-10)
    ):
        msg = "Preconditioner requires finite real gradient, symmetric metric, and damping 0.001 or 0.01."
        raise ValueError(msg)
    eigenvalues = np.linalg.eigvalsh(metric)
    if eigenvalues.size and eigenvalues[0] < -1e-10:
        msg = "Pure-state preconditioning metric is not positive semidefinite."
        raise ValueError(msg)
    direction = np.linalg.solve(metric + damping * np.eye(gradient.size), -gradient)
    slope = float(gradient @ direction)
    if not np.all(np.isfinite(direction)) or (np.linalg.norm(gradient) > 0 and slope >= 0):
        msg = "Regularized metric solve did not produce a finite descent direction."
        raise FloatingPointError(msg)
    diagnostics: dict[str, Any] = {
        "damping": damping,
        "slope": slope,
        "direction_norm": float(np.linalg.norm(direction)),
        "minimum_metric_eigenvalue": float(eigenvalues[0]) if eigenvalues.size else None,
        "maximum_metric_eigenvalue": float(eigenvalues[-1]) if eigenvalues.size else None,
        "damped_condition_number": float((eigenvalues[-1] + damping) / (eigenvalues[0] + damping))
        if eigenvalues.size
        else None,
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall,
        "linear_solves": int(bool(gradient.size)),
        "symmetric_eigensolves": int(bool(gradient.size)),
    }
    return direction, diagnostics


def _validate_state(state: dict[str, Any], signature: str, initial: NDArray) -> None:
    """Validate a complete JSON-compatible geometry/line-search checkpoint.

    Raises:
        ValueError: If stored identity, parameters, counters, or pending state differ.
    """
    valid = state["signature"] == signature and state["version"] == VERSION
    valid = valid and state["status"] in {"running", "evaluation_limit", "converged", "line_search_failed"}
    for key in ("channel_cpu_seconds", "metric_cpu_seconds", "solve_cpu_seconds", "total_cpu_seconds"):
        valid = valid and np.isfinite(state[key]) and state[key] >= 0
    for key in ("theta", "gradient"):
        value = np.asarray(state[key])
        valid = valid and value.shape == initial.shape and np.isrealobj(value) and bool(np.all(np.isfinite(value)))
    for key in ("evaluations", "iterations", "rejected_evaluations", "nonfinite_evaluations", "metric_evaluations"):
        valid = valid and type(state[key]) is int and state[key] >= 0
    valid = (
        valid
        and np.isfinite(state["loss"])
        and state["evaluations"] == 1 + state["iterations"] + state["rejected_evaluations"]
    )
    valid = valid and len(state["metric_records"]) == state["metric_evaluations"] <= state["iterations"] + 1
    if state["pending"] is not None:
        pending = state["pending"]
        direction = np.asarray(pending["direction"])
        valid = (
            valid and direction.shape == initial.shape and np.isrealobj(direction) and np.all(np.isfinite(direction))
        )
        valid = valid and type(pending["trials"]) is int and 0 <= pending["trials"] < MAX_TRIALS
        valid = (
            valid
            and pending["step"] == BACKTRACK ** pending["trials"]
            and np.isfinite(pending["slope"])
            and pending["slope"] < 0
        )
    if not valid:
        msg = "Geometry optimizer checkpoint is malformed or incompatible."
        raise ValueError(msg)


def minimize(
    circuit: ParameterizedCircuit,
    initial: NDArray,
    target: NDArray,
    *,
    damping: float,
    max_evaluations: int = 300,
    state: dict[str, Any] | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Minimize the exact calibrated local noisy loss using pure-state geometry.

    Args:
        circuit: Fixed logical q1-q6 circuit prepared from zero.
        initial: Original finite parameters, also authenticating resumed calls.
        target: Normalized pure target for the true noisy density objective.
        damping: Exactly 0.001 or 0.01; no adaptive damping or wider grid.
        max_evaluations: Cumulative joint true-channel-gradient cap, at most 300.
        state: Previous complete accepted state and optional pending line search.
        callback: Receives a private serializable state after the initial call
            and every completed line-search trial, including rejected trials.

    Returns:
        Final accepted parameters, gradient, counters, pending direction/step,
        complete metric-cost records and accepted-iterate curve. A resumed
        partially spent line search reuses its direction and does not reconstruct
        its metric. Every trial uses the true noisy channel, including its gradient;
        rejected and nonfinite trials consume the hard channel-call budget.
        Metric construction, spectral checks and linear solves are additional
        charged work, never relabelled as free channel-gradient evaluations.

    Raises:
        ValueError: If arguments or checkpoint identity/counters are invalid.
        FloatingPointError: If initial objective or preconditioner is invalid.

    Notes:
        The caller owns source/custody hashes and durable checkpoint files.
        Caught interruption overhead or work after its last durable checkpoint
        needs an external attempt ledger; this kernel returns completed-call
        accounting and supports trial-boundary checkpoints without hidden state.
    """
    _validate(circuit, initial)
    if (
        isinstance(damping, bool)
        or damping not in DAMPINGS
        or type(max_evaluations) is not int
        or not 1 <= max_evaluations <= 300
        or target.shape != (2**circuit.num_qubits,)
        or not np.all(np.isfinite(target))
        or abs(np.vdot(target, target) - 1) > 1e-10
    ):
        msg = "Geometry optimizer requires a normalized target, registered damping and a 1-300 call cap."
        raise ValueError(msg)
    calibration = CalibratedLocalDepolarizingProvider()
    signature = pilot.digest({
        "version": VERSION,
        "circuit": pilot.circuit_record(circuit),
        "initial": initial.tolist(),
        "target": hashlib.sha256(target.tobytes()).hexdigest(),
        "calibration": calibration.content_checksum,
        "damping": damping,
        "armijo": ARMIJO,
        "backtrack": BACKTRACK,
        "max_trials": MAX_TRIALS,
        "gtol": GTOL,
    })
    current: dict[str, Any]
    if state is None:
        current = {
            "signature": signature,
            "version": VERSION,
            "theta": initial.tolist(),
            "gradient": [],
            "loss": None,
            "evaluations": 0,
            "iterations": 0,
            "rejected_evaluations": 0,
            "nonfinite_evaluations": 0,
            "metric_evaluations": 0,
            "channel_cpu_seconds": 0.0,
            "metric_cpu_seconds": 0.0,
            "solve_cpu_seconds": 0.0,
            "total_cpu_seconds": 0.0,
            "metric_records": [],
            "curve": [],
            "trials": [],
            "pending": None,
            "status": "running",
        }
    else:
        current = copy.deepcopy(state)
        try:
            _validate_state(current, signature, initial)
        except (KeyError, TypeError, IndexError) as error:
            msg = "Geometry optimizer checkpoint is malformed or incompatible."
            raise ValueError(msg) from error
        if current["evaluations"] > max_evaluations:
            msg = "Geometry resume cannot reduce the consumed channel-call budget."
            raise ValueError(msg)
        if current["status"] in {"converged", "line_search_failed"}:
            return current
        current["status"] = "running"
    prior_cpu = current["total_cpu_seconds"]
    started = time.process_time()

    def objective(theta: NDArray) -> tuple[float, NDArray]:
        cpu = time.process_time()
        current["evaluations"] += 1
        try:
            gradient, loss, _ = density_loss_gradient(circuit, theta, target, calibration)
        except FloatingPointError:
            return float("inf"), np.full_like(theta, np.nan)
        finally:
            current["channel_cpu_seconds"] += time.process_time() - cpu
        return loss, gradient

    def publish(*, accepted: bool) -> None:
        current["total_cpu_seconds"] = prior_cpu + time.process_time() - started
        current["density_direction_passes"] = 2 * current["evaluations"]
        current["statevector_row_operator_applications"] = sum(
            row["construction"]["statevector_row_operator_applications"] for row in current["metric_records"]
        )
        current["training_trajectories"] = 0
        if accepted:
            current["curve"].append({
                "theta": current["theta"],
                "fidelity": 1 - current["loss"],
                "evaluations": current["evaluations"],
                "iterations": current["iterations"],
                "cpu_seconds": current["total_cpu_seconds"],
            })
        if callback is not None:
            callback(copy.deepcopy(current))

    if state is None:
        loss, gradient = objective(initial)
        if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
            msg = "Initial geometry objective must be finite."
            raise FloatingPointError(msg)
        current.update(loss=loss, gradient=gradient.tolist())
        publish(accepted=True)
    while True:
        theta, gradient = np.asarray(current["theta"]), np.asarray(current["gradient"])
        if not gradient.size or np.linalg.norm(gradient, ord=np.inf) <= GTOL:
            current["status"] = "converged"
            break
        if current["evaluations"] >= max_evaluations:
            current["status"] = "evaluation_limit"
            break
        if current["pending"] is None:
            metric, construction = pure_state_metric(circuit, theta)
            direction, solve = precondition(gradient, metric, damping)
            current["metric_evaluations"] += 1
            current["metric_cpu_seconds"] += construction["cpu_seconds"]
            current["solve_cpu_seconds"] += solve["cpu_seconds"]
            current["metric_records"].append({
                "iteration": current["iterations"],
                "construction": construction,
                "solve": solve,
            })
            current["pending"] = {"direction": direction.tolist(), "slope": solve["slope"], "step": 1.0, "trials": 0}
        pending = current["pending"]
        trial_theta = theta + pending["step"] * np.asarray(pending["direction"])
        loss, gradient = objective(trial_theta)
        finite = bool(np.isfinite(loss) and np.all(np.isfinite(gradient)))
        accepted = finite and loss <= cast("float", current["loss"]) + ARMIJO * pending["step"] * pending["slope"]
        current["trials"].append({
            "evaluation": current["evaluations"],
            "iteration": current["iterations"],
            "step": pending["step"],
            "accepted": accepted,
            "loss": loss if finite else None,
            "slope": pending["slope"],
        })
        if accepted:
            current.update(theta=trial_theta.tolist(), gradient=gradient.tolist(), loss=loss, pending=None)
            current["iterations"] += 1
        else:
            current["rejected_evaluations"] += 1
            current["nonfinite_evaluations"] += int(not finite)
            pending["trials"] += 1
            pending["step"] *= BACKTRACK
            if pending["trials"] == MAX_TRIALS:
                current.update(status="line_search_failed", pending=None)
        publish(accepted=accepted)
        if current["status"] == "line_search_failed":
            break
    if current["status"] == "running":
        current["status"] = "evaluation_limit"
    current["total_cpu_seconds"] = prior_cpu + time.process_time() - started
    current["density_direction_passes"] = 2 * current["evaluations"]
    current["statevector_row_operator_applications"] = sum(
        row["construction"]["statevector_row_operator_applications"] for row in current["metric_records"]
    )
    current["training_trajectories"] = 0
    return current
