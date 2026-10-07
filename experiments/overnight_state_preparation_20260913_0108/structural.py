# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Budgeted adaptive exact deletion with an equally repaired retained branch.

This adapts variable-ansatz grow/simplify ideas (arXiv:2103.06712), restricting
the initial test to deletion and exact noisy acceptance. It reuses the existing
shared finite-deletion adjoint. No new quantum channel or initial state is used.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks.state_preparation.exact_density import density_deletion_fidelities
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider, delete_gates, exact_density_fidelity
from experiments.state_preparation_pilot import circuit_record

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
    from mqt.yaqs.optimization import ParameterizedCircuit


@dataclass(frozen=True)
class StructuralConfig:
    """Freeze the entire search budget and deterministic acceptance policy."""

    max_rounds: int = 20
    shortlist: int = 2
    repair_evaluations: int = 30
    max_refinement_evaluations: int = 1800
    acceptance_tolerance: float = 1e-10
    ranking_quantum: float = 1e-12

    def __post_init__(self) -> None:
        """Reject empty or nonfinite budgets before any objective evaluations."""
        integers = (self.max_rounds, self.shortlist, self.repair_evaluations, self.max_refinement_evaluations)
        reals = (self.acceptance_tolerance, self.ranking_quantum)
        if any(type(value) is not int or value < 1 for value in integers) or any(
            isinstance(value, bool) or not np.isfinite(value) or value <= 0 for value in reals
        ):
            msg = "Structural budgets and tolerances must be finite and positive."
            raise ValueError(msg)


@dataclass
class RefinementResult:
    """Return a callback endpoint, its actual cost, and serializable diagnostics.

    ``evaluations`` counts all objective/gradient evaluations, including rejected
    line-search trials. A callback must respect its supplied evaluation cap.
    Optimizer moments may be kept in an external callback adapter when required;
    this driver supplies independent copied circuit/parameter inputs per branch.
    """

    theta: NDArray[np.float64]
    evaluations: int
    iterations: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _validated_refinement(result: RefinementResult, size: int, cap: int) -> NDArray[np.float64]:
    """Return valid callback parameters after checking its physical work cap.

    Raises:
        ValueError: If the callback overran or misreported its budget.
        FloatingPointError: If callback parameters are nonfinite or malformed.
    """
    if (
        type(result.evaluations) is not int
        or not 0 <= result.evaluations <= cap
        or type(result.iterations) is not int
        or result.iterations < 0
    ):
        msg = "Refinement callback violated its evaluation budget or iteration accounting."
        raise ValueError(msg)
    finish = np.asarray(result.theta)
    if finish.shape != (size,) or not np.isrealobj(finish) or not np.all(np.isfinite(finish)):
        msg = "Refinement returned invalid parameters."
        raise FloatingPointError(msg)
    return np.asarray(finish, dtype=float)


def adaptive_delete_refine(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    calibration: CalibratedLocalDepolarizingProvider,
    refine: Callable[[ParameterizedCircuit, NDArray[np.float64], int], RefinementResult],
    *,
    config: StructuralConfig | None = None,
    protected_origins: frozenset[int | str] = frozenset(),
) -> dict[str, Any]:
    """Delete only when repaired compression beats equally repaired retention.

    Every complete round reserves equal callback caps for the retained circuit
    and all shortlisted deletions. A round never starts if these caps exceed
    the remaining budget. Actual and reserved costs, discarded trials, and all
    finite-deletion rankings are retained. Refinement failures consume their
    full reserved cap because their exact evaluation count is unavailable.

    Args:
        circuit: Already lowered one-/two-qubit circuit, preparing from zero.
        theta: Initial real parameters. Inputs and gate metadata are preserved.
        target: Fixed normalized benchmark target in little-endian ordering.
        calibration: Unchanged local Pauli-Lindblad evaluation semantics.
        refine: Independent parameter optimizer with a hard evaluation cap.
        config: Frozen round, shortlist, refinement and numerical limits.
        protected_origins: Logical IDs of physical preparation gates to retain.

    Returns:
        JSON-compatible endpoint, charged work ledger, every branch and stop rule.

    Raises:
        ValueError: If inputs or callback evaluation accounting are invalid.
    """
    if config is None:
        config = StructuralConfig()
    if not np.isrealobj(theta):
        msg = "Structural refinement requires real parameters."
        raise ValueError(msg)
    current = copy.deepcopy(circuit)
    parameters = np.array(theta, dtype=float, copy=True)
    provider = ExactLocalPauliProvider(calibration)
    origins = list(range(len(current.gates)))
    ledger: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    consumed = 0

    def score(candidate: ParameterizedCircuit, values: NDArray[np.float64], role: str) -> float:
        """Independently score and charge a complete physical circuit."""
        wall, cpu = time.perf_counter(), time.process_time()
        value = exact_density_fidelity(candidate, values, target, provider)
        ledger.append({
            "role": role,
            "cpu_seconds": time.process_time() - cpu,
            "wall_seconds": time.perf_counter() - wall,
            "exact_channel_evaluations": 1,
            "refinement_evaluations": 0,
        })
        return value

    # Validate shape/finiteness with the strict shared oracle before callbacks.
    wall, cpu = time.perf_counter(), time.process_time()
    initial_scores, fidelity = density_deletion_fidelities(current, parameters, target, calibration)
    ledger.append({
        "role": "initial_score_validation",
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall,
        "exact_channel_evaluations": 1,
        "refinement_evaluations": 0,
        "density_direction_passes": 2,
    })
    del initial_scores
    initial_fidelity = fidelity
    stop_reason = "max_rounds"
    for round_index in range(config.max_rounds):
        eligible = [i for i, gate in enumerate(current.gates) if gate.logical_gate_id not in protected_origins]
        if not eligible:
            stop_reason = "no_eligible_gates"
            break
        count = min(config.shortlist, len(eligible))
        reserved = (count + 1) * config.repair_evaluations
        if consumed + reserved > config.max_refinement_evaluations:
            stop_reason = "refinement_budget"
            break
        wall, cpu = time.perf_counter(), time.process_time()
        values, reference = density_deletion_fidelities(current, parameters, target, calibration)
        ledger.append({
            "role": "deletion_ranking",
            "cpu_seconds": time.process_time() - cpu,
            "wall_seconds": time.perf_counter() - wall,
            "exact_channel_evaluations": 1,
            "refinement_evaluations": 0,
            "density_direction_passes": 2,
            "finite_deletion_scores": len(values),
        })
        if abs(reference - fidelity) > 1e-10:
            msg = "Independent endpoint and shared deletion oracle disagree."
            raise ValueError(msg)
        ranking = sorted(eligible, key=lambda i: (-round(values[i] / config.ranking_quantum), origins[i]))
        branches: list[dict[str, Any]] = []
        branch_circuits: list[ParameterizedCircuit] = []
        branch_thetas: list[NDArray[np.float64]] = []
        for index in (None, *ranking[:count]):
            candidate, start = (
                (copy.deepcopy(current), parameters.copy())
                if index is None
                else delete_gates(current, parameters, [index])
            )
            branch: dict[str, Any] = {
                "kind": "retain" if index is None else "delete",
                "index": index,
                "origin": None if index is None else origins[index],
                "logical_gate_id": None if index is None else current.gates[index].logical_gate_id,
                "raw_fidelity": fidelity if index is None else float(values[index]),
                "evaluation_cap": config.repair_evaluations,
            }
            wall, cpu = time.perf_counter(), time.process_time()
            try:
                repaired = refine(copy.deepcopy(candidate), start.copy(), config.repair_evaluations)
                finish = _validated_refinement(repaired, len(start), config.repair_evaluations)
                branch.update({
                    "status": "complete",
                    "evaluations": repaired.evaluations,
                    "iterations": repaired.iterations,
                    "diagnostics": copy.deepcopy(repaired.diagnostics),
                })
            except (ArithmeticError, RuntimeError) as error:
                finish = start.copy()
                branch.update({
                    "status": "failed",
                    "failure": f"{type(error).__name__}: {error}",
                    "evaluations": config.repair_evaluations,
                    "iterations": 0,
                })
            ledger.append({
                "role": f"{branch['kind']}_repair",
                "cpu_seconds": time.process_time() - cpu,
                "wall_seconds": time.perf_counter() - wall,
                "exact_channel_evaluations": 0,
                "refinement_evaluations": branch["evaluations"],
                "reserved_refinement_evaluations": config.repair_evaluations,
                "status": branch["status"],
            })
            consumed += branch["evaluations"]
            branch["fidelity"] = score(candidate, finish, f"{branch['kind']}_postrepair_score")
            branch["theta"] = finish.tolist()
            branch["circuit"] = circuit_record(candidate)
            branch_circuits.append(candidate)
            branch_thetas.append(finish.copy())
            branches.append(branch)
        # Preserve the initial endpoint if the retained repair itself regressed.
        selected: int | None = 0 if branches[0]["fidelity"] > fidelity + config.acceptance_tolerance else None
        best = max(fidelity, branches[0]["fidelity"])
        for index in range(1, len(branches)):
            branch = branches[index]
            if branch["status"] == "complete" and branch["fidelity"] > best + config.acceptance_tolerance:
                selected, best = index, branch["fidelity"]
        rounds.append({
            "round": round_index + 1,
            "initial_fidelity": fidelity,
            "ranking": [{"index": i, "origin": origins[i], "fidelity": float(values[i])} for i in ranking],
            "branches": branches,
            "selected_branch": selected,
            "reserved_refinement_evaluations": reserved,
            "cumulative_refinement_evaluations": consumed,
        })
        if selected is not None:
            current, parameters = branch_circuits[selected], branch_thetas[selected]
            fidelity = branches[selected]["fidelity"]
        if selected is None or selected == 0:
            stop_reason = "no_repaired_deletion_advantage"
            break
        removed = branches[selected]["index"]
        origins.pop(removed)
    return {
        "status": "complete",
        "circuit": circuit_record(current),
        "theta": parameters.tolist(),
        "initial_fidelity": initial_fidelity,
        "fidelity": fidelity,
        "stop_reason": stop_reason,
        "rounds": rounds,
        "ledger": ledger,
        "refinement_evaluations": consumed,
        "cpu_seconds": sum(entry["cpu_seconds"] for entry in ledger),
        "wall_seconds": sum(entry["wall_seconds"] for entry in ledger),
    }
