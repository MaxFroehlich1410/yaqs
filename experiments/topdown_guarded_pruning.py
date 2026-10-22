# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact ideal pruning with adaptive batches and a cumulative fidelity budget."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from benchmarks.state_preparation.pilot_components import delete_gates
from experiments import topdown_backend_support as support
from mqt.yaqs.optimization import StatevectorBackend

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.mps import MPS
    from mqt.yaqs.optimization import ParameterizedCircuit
    from mqt.yaqs.optimization.backends import PreparationBackend

PROTOCOL: dict = {
    "version": "guarded-adaptive-ideal-pruning-v1",
    "cutoff": 0.01,
    "loss_budget": 0.05,
    "guard_tolerance": 1e-12,
    "score": "signed exact current ideal fidelity minus fidelity after single deletion",
    "batch": "all scores strictly below cutoff; halve weakest prefix on rejection",
    "ties": "stable original gate order",
    "guard": "candidate fidelity >= original starting fidelity minus loss budget",
    "rescore": "after each accepted batch or single deletion",
    "repair_updates": 0,
    "noisy_updates": 0,
}


def prune(
    circuit: ParameterizedCircuit,
    theta: NDArray,
    target: NDArray | MPS,
    *,
    cutoff: float = 0.01,
    loss_budget: float = 0.05,
    backend_factory: Callable[..., PreparationBackend] = StatevectorBackend,
) -> tuple[ParameterizedCircuit, NDArray, dict]:
    """Delete weak gates in guarded batches, falling back to a C-style single gate.

    Args:
        circuit: Circuit with one independent scalar parameter per gate.
        theta: Existing parameters; retained values are never optimized.
        target: Normalized target state vector or MPS.
        backend_factory: Backend constructor; default preserves the original dense execution.
        cutoff: Strict signed single-deletion damage threshold.
        loss_budget: Absolute fidelity loss allowed relative to the original input.

    Returns:
        Reduced circuit, private retained parameters and complete decision trace.

    Raises:
        ValueError: If parameters or policy options are invalid.
    """
    indices = [gate.param_index for gate in circuit.gates]
    if (
        any(index is None for index in indices)
        or sorted(index for index in indices if index is not None) != list(range(circuit.num_params))
        or theta.shape != (circuit.num_params,)
        or not np.all(np.isfinite(theta))
    ):
        msg = "Require one independent scalar parameter per gate."
        raise ValueError(msg)
    if not np.isfinite(cutoff) or cutoff <= 0 or not np.isfinite(loss_budget) or not 0 <= loss_budget <= 1:
        msg = "Require a positive finite cutoff and a loss budget in [0, 1]."
        raise ValueError(msg)
    circuit = support.restore_circuit(support.circuit_record(circuit))
    theta = theta.copy()
    origins = list(range(len(circuit.gates)))
    initial = float(backend_factory(circuit, target).metrics(theta)[1])
    floor = initial - loss_budget
    trace: dict = {
        "protocol": {**PROTOCOL, "cutoff": cutoff, "loss_budget": loss_budget},
        "initial_fidelity": initial,
        "fidelity_floor": floor,
        "iterations": [],
        "scored_candidates": 0,
        "repair_updates": 0,
        "stop": "empty_circuit",
    }
    while circuit.gates:
        removed, baseline = backend_factory(circuit, target).deletion_fidelities(theta)
        scores = baseline - removed
        eligible = np.flatnonzero(scores < cutoff)
        ranked = eligible[np.argsort(scores[eligible], kind="stable")].tolist()
        entry: dict = {"baseline": float(baseline), "scores": scores.tolist(), "attempts": []}
        trace["iterations"].append(entry)
        trace["scored_candidates"] += len(scores)
        if not ranked:
            trace["stop"] = "no_score_below_cutoff"
            break
        count = len(ranked)
        accepted = False
        while count:
            selected = sorted(ranked[:count])
            candidate, retained = delete_gates(circuit, theta, selected)
            fidelity = float(backend_factory(candidate, target).metrics(retained)[1])
            accepted = fidelity + PROTOCOL["guard_tolerance"] >= floor
            entry["attempts"].append({
                "removed_origins": [origins[index] for index in selected],
                "fidelity": fidelity,
                "accepted": accepted,
            })
            if accepted:
                circuit, theta = candidate, retained
                selected_set = set(selected)
                origins = [origin for index, origin in enumerate(origins) if index not in selected_set]
                break
            count //= 2
        if not accepted:
            trace["stop"] = "cumulative_budget"
            break
    trace["remaining_origins"] = origins
    trace["remaining_parameter_indices"] = sorted(indices[origin] for origin in origins)
    trace["final_fidelity"] = float(backend_factory(circuit, target).metrics(theta)[1])
    return circuit, theta, trace
