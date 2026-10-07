# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Physical deletion semantics and fair adaptive search stopping tests."""

from __future__ import annotations

import copy
import json
from typing import cast

import numpy as np
import pytest

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider, exact_density_fidelity
from experiments.overnight_state_preparation_20260913_0108.structural import (
    RefinementResult,
    StructuralConfig,
    adaptive_delete_refine,
)
from experiments.state_preparation_pilot import circuit_record, restore_circuit
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate


def unchanged(circuit: ParameterizedCircuit, theta: np.ndarray, cap: int) -> RefinementResult:
    """Return a branch endpoint unchanged while charging one evaluation."""
    assert theta.shape == (circuit.num_params,)
    assert cap >= 1
    return RefinementResult(theta, evaluations=1, iterations=0)


def test_repaired_deletion_removes_physical_noise_and_stops_at_essential_gate() -> None:
    """Repair recovers useful compression; an essential final gate is retained."""
    circuit = ParameterizedCircuit(
        1,
        [
            ParameterizedGate("ry", (0,), 0, logical_gate_id=11),
            ParameterizedGate("ry", (0,), 1, logical_gate_id=22),
        ],
    )
    theta = np.full(2, np.pi / 4)
    target = np.ones(2, dtype=complex) / np.sqrt(2)
    original_circuit, original_theta = copy.deepcopy(circuit_record(circuit)), theta.copy()
    caps = []

    def repair(candidate: ParameterizedCircuit, start: np.ndarray, cap: int) -> RefinementResult:
        """Return plus preparation wherever a retained Y rotation permits it."""
        caps.append(cap)
        if candidate.num_params:
            start[:] = 0
            start[0] = np.pi / 2
        return RefinementResult(start, evaluations=1, iterations=1)

    calibration = CalibratedLocalDepolarizingProvider()
    result = adaptive_delete_refine(
        circuit, theta, target, calibration, repair, config=StructuralConfig(repair_evaluations=2)
    )
    assert result["stop_reason"] == "no_repaired_deletion_advantage"
    assert len(result["circuit"]["gates"]) == 1
    assert len(result["rounds"]) == 2
    assert result["fidelity"] > result["initial_fidelity"]
    assert caps == [2] * 5
    assert result["refinement_evaluations"] == 5
    assert result["rounds"][0]["selected_branch"] == 1
    assert result["rounds"][0]["branches"][1]["raw_fidelity"] < result["initial_fidelity"]
    expected = exact_density_fidelity(
        restore_circuit(result["circuit"]), np.asarray(result["theta"]), target, ExactLocalPauliProvider(calibration)
    )
    assert result["fidelity"] == pytest.approx(expected, abs=1e-14)
    assert circuit_record(circuit) == original_circuit
    np.testing.assert_array_equal(theta, original_theta)
    assert result["cpu_seconds"] >= 0
    assert result["wall_seconds"] >= 0
    json.dumps(result, allow_nan=False)


def test_retention_receives_equal_repair_and_can_win_despite_initial_headroom() -> None:
    """Compression is judged against the repaired control, not the weak start."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), 0), ParameterizedGate("ry", (0,), 1)])
    target = np.ones(2, dtype=complex) / np.sqrt(2)

    def repair(candidate: ParameterizedCircuit, theta: np.ndarray, cap: int) -> RefinementResult:
        """Return retention improved farther than either repaired deletion."""
        theta[:] = np.pi / 4 if candidate.num_params == 2 else np.pi / 8
        return RefinementResult(theta, evaluations=cap, iterations=1)

    result = adaptive_delete_refine(circuit, np.full(2, 0.01), target, CalibratedLocalDepolarizingProvider(), repair)
    assert len(result["rounds"]) == 1
    assert result["rounds"][0]["selected_branch"] == 0
    assert len(result["circuit"]["gates"]) == 2
    assert result["fidelity"] > result["initial_fidelity"]
    assert result["refinement_evaluations"] == 90
    assert all(b["fidelity"] > result["initial_fidelity"] for b in result["rounds"][0]["branches"])


def test_noise_only_two_qubit_gate_can_be_deleted_with_preparation_protected() -> None:
    """A zero-angle entangler still has a channel; remove both atomically."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), 0, logical_gate_id=0),
            ParameterizedGate("cx", (0, 1), logical_gate_id=1),
            ParameterizedGate("rxx", (0, 1), 1, logical_gate_id=2),
        ],
    )
    target = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    result = adaptive_delete_refine(
        circuit,
        np.array([np.pi / 2, 0]),
        target,
        CalibratedLocalDepolarizingProvider(),
        unchanged,
        protected_origins=frozenset({0, 1}),
    )
    assert result["stop_reason"] == "no_eligible_gates"
    assert len(result["circuit"]["gates"]) == 2
    assert result["fidelity"] > result["initial_fidelity"]
    assert result["rounds"][0]["branches"][1]["logical_gate_id"] == 2


def test_insufficient_budget_never_runs_partial_round() -> None:
    """No candidate gets an optimizer opportunity missing from its comparator."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), 0)])
    result = adaptive_delete_refine(
        circuit,
        np.zeros(1),
        np.array([1, 0], dtype=complex),
        CalibratedLocalDepolarizingProvider(),
        unchanged,
        config=StructuralConfig(repair_evaluations=3, max_refinement_evaluations=5),
    )
    assert result["rounds"] == []
    assert result["refinement_evaluations"] == 0
    assert result["stop_reason"] == "refinement_budget"
    assert result["ledger"][0]["role"] == "initial_score_validation"


def test_ties_keep_original_and_ranking_is_stable() -> None:
    """Numerically equal deletions do not silently become gate-count wins."""
    circuit = ParameterizedCircuit(
        1,
        [
            ParameterizedGate("ry", (0,), 0, logical_gate_id=900),
            ParameterizedGate("rz", (0,), 1, logical_gate_id=100),
        ],
    )
    calibration = CalibratedLocalDepolarizingProvider(one_qubit_error=0, two_qubit_error=0)
    result = adaptive_delete_refine(circuit, np.zeros(2), np.array([1, 0], dtype=complex), calibration, unchanged)
    assert result["rounds"][0]["selected_branch"] is None
    assert [r["origin"] for r in result["rounds"][0]["ranking"]] == [0, 1]
    assert len(result["circuit"]["gates"]) == 2


def test_failed_repairs_are_charged_and_cannot_accept_deletion() -> None:
    """A failed compressed branch is not promoted by its fallback raw score."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), 0)])

    def fail(_circuit: ParameterizedCircuit, _theta: np.ndarray, _cap: int) -> RefinementResult:
        """Represent a numerical optimizer failure.

        Raises:
            FloatingPointError: Always, after an unknown amount of work.
        """
        msg = "Nonfinite line search."
        raise FloatingPointError(msg)

    result = adaptive_delete_refine(
        circuit,
        np.zeros(1),
        np.array([1, 0], dtype=complex),
        CalibratedLocalDepolarizingProvider(),
        fail,
        config=StructuralConfig(repair_evaluations=3),
    )
    assert result["refinement_evaluations"] == 6
    assert len(result["circuit"]["gates"]) == 1
    assert all(branch["status"] == "failed" for branch in result["rounds"][0]["branches"])
    assert result["rounds"][0]["selected_branch"] is None


def test_overbudget_callback_is_rejected() -> None:
    """Misreported work invalidates the run rather than producing a fair-looking score."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), 0)])

    def overbudget(_circuit: ParameterizedCircuit, theta: np.ndarray, cap: int) -> RefinementResult:
        return RefinementResult(theta, evaluations=cap + 1, iterations=1)

    with pytest.raises(ValueError, match="violated its evaluation budget"):
        adaptive_delete_refine(
            circuit, np.zeros(1), np.array([1, 0], dtype=complex), CalibratedLocalDepolarizingProvider(), overbudget
        )


@pytest.mark.parametrize("field", ["max_rounds", "shortlist", "repair_evaluations", "max_refinement_evaluations"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_integer_budgets_rejected(field: str, value: object) -> None:
    """Invalid experimental budgets fail before the first physical evaluation."""
    with pytest.raises(ValueError, match="finite and positive"):
        StructuralConfig(**{field: cast("int", value)})
