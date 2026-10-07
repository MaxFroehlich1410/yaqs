# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Complex-state, adapter and provenance checks for published AQCE scheduling."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector, random_statevector, random_unitary

from experiments.state_preparation_competitors import aqce as study

if TYPE_CHECKING:
    from pathlib import Path
    from types import FrameType

    from _typeshed import TraceFunction

pytest.importorskip("qulacs")


def test_vendored_files_are_byte_identical_to_pinned_upstream() -> None:
    """Preserve the upstream algorithm and its actual root license verbatim."""
    for filename, expected in study.UPSTREAM_HASHES.items():
        assert hashlib.sha256((study.UPSTREAM_DIRECTORY / filename).read_bytes()).hexdigest() == expected
    assert "Attribution-ShareAlike 4.0 International" in (study.UPSTREAM_DIRECTORY / "LICENSE.txt").read_text()
    assert callable(study.load_upstream().AQCE_python)


def test_loader_rejects_a_changed_source_before_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not silently benchmark a modified upstream implementation."""
    for filename in study.UPSTREAM_HASHES:
        shutil.copyfile(study.UPSTREAM_DIRECTORY / filename, tmp_path / filename)
    (tmp_path / "AQCE_from_python.py.txt").write_text("raise AssertionError('must not execute')\n")
    monkeypatch.setattr(study, "UPSTREAM_DIRECTORY", tmp_path)
    with pytest.raises(RuntimeError, match="integrity"):
        study.load_upstream()


def test_adapter_preserves_published_schedule_output_on_complex_six_qubit_target() -> None:
    """Circuit conversion and metadata must describe the published scheduler."""
    target = random_statevector(64, seed=61).data
    before = target.copy()
    actual = study.aqce_candidates(target, gate_budgets=(0, 2, 4), sweeps=2)
    upstream = study.load_upstream()
    for candidate, budget in zip(actual, (0, 2, 4), strict=True):
        state = upstream.QuantumState(6)
        state.load(target)
        initial = 1 if budget else 0
        gates = study.aqce_schedule.encode(
            upstream, state, initial_gates=initial, growth_increment=1, max_gates=budget, sweeps=2
        )
        expected = upstream.get_state(6, gates).get_vector()
        obtained = Statevector.from_instruction(candidate.circuit).data
        np.testing.assert_allclose(obtained, expected, atol=2e-14, rtol=0)
        assert candidate.metadata["ideal_fidelity"] == pytest.approx(abs(np.vdot(target, expected)) ** 2)
        assert np.linalg.norm(obtained) == pytest.approx(1.0)
        assert len(candidate.circuit.data) == budget
        assert candidate.metadata["gate_budget"] == budget
        assert candidate.metadata["scheduled_unitary_updates"] == 2 * budget * (budget + 1) + budget - initial
        assert candidate.metadata["scheduled_initialization_steps"] == initial
        assert candidate.metadata["algorithm"] == study.ALGORITHM
        assert candidate.metadata["sweep_definition"] == "forward_then_backward"
        json.dumps(candidate.metadata, allow_nan=False)
        for instruction in candidate.circuit.data:
            unitary = Operator(instruction.operation).data
            np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(4), atol=2e-14, rtol=0)
    np.testing.assert_array_equal(target, before)


def test_adaptive_pair_selection_prepares_a_nonlocal_complex_bell_state() -> None:
    """An all-to-all update must choose the nonadjacent entangled pair."""
    target = np.zeros(8, dtype=complex)
    target[[0, 5]] = [np.sqrt(0.3), 1j * np.sqrt(0.7)]
    candidate = study.aqce_candidates(target, gate_budgets=(1,), sweeps=2)[0]
    assert candidate.metadata["ideal_fidelity"] == pytest.approx(1.0)
    instruction = candidate.circuit.data[0]
    assert {candidate.circuit.find_bit(qubit).index for qubit in instruction.qubits} == {0, 2}


def test_qulacsto_qiskit_conversion_preserves_nonsymmetric_gate_order() -> None:
    """Use reversed, nonlocal targets and noncommuting complex gates as oracle."""
    upstream = study.load_upstream()
    first = random_unitary(4, seed=101).data
    second = random_unitary(4, seed=102).data
    gates = [upstream.DenseMatrix([2, 0], first), upstream.DenseMatrix([0, 1], second)]
    actual = study.to_qiskit(gates, 3)
    expected = QuantumCircuit(3)
    expected.unitary(first, [2, 0])
    expected.unitary(second, [0, 1])
    np.testing.assert_allclose(Operator(actual).data, Operator(expected).data, atol=1e-14, rtol=0)
    direct_state = upstream.get_state(3, gates).get_vector()
    np.testing.assert_allclose(Statevector.from_instruction(actual).data, direct_state, atol=1e-14, rtol=0)


def test_upstream_transition_tensor_matches_independent_complex_overlap() -> None:
    """Check conjugation and pair ordering through a full-state Qiskit oracle."""
    upstream = study.load_upstream()
    phi = random_statevector(16, seed=104).data
    psi = random_statevector(16, seed=105).data
    left, right = upstream.QuantumState(4), upstream.QuantumState(4)
    left.load(phi)
    right.load(psi)
    matrix = upstream.evaluate_F(right, left, right, (0, 3))
    for seed in range(5):
        trial = random_unitary(4, seed=seed).data
        evolved = Statevector(psi).evolve(Operator(trial.conj().T), qargs=[3, 0]).data
        assert np.trace(matrix @ trial.conj().T) == pytest.approx(np.vdot(phi, evolved), abs=2e-14)
    x, singular_values, y_dagger = np.linalg.svd(matrix)
    optimal = x @ y_dagger
    evolved = Statevector(psi).evolve(Operator(optimal.conj().T), qargs=[3, 0]).data
    assert abs(np.vdot(phi, evolved)) == pytest.approx(float(singular_values.sum()), abs=2e-14)


def test_original_upstream_gate_updates_are_monotone_and_scan_every_pair() -> None:
    """Observe pinned source execution without patching any numerical routine."""
    upstream = study.load_upstream()
    target = random_statevector(8, seed=77).data
    state = upstream.QuantumState(3)
    state.load(target)
    records: list[tuple[float, float]] = []
    pairs: list[tuple[int, int]] = []
    before: list[float] = []

    def trace(frame: FrameType, event: str, _arg: object) -> TraceFunction:
        """Record full circuit fidelities immediately around a gate assignment.

        Returns:
            The same tracer to continue observing this invocation.
        """
        if frame.f_code is upstream.evaluate_F.__code__ and event == "call":
            pairs.append(frame.f_locals["idx"])
        if frame.f_code is upstream.AQCE_python.__code__ and event == "line" and frame.f_lineno in {125, 126}:
            circuit = study.to_qiskit(frame.f_locals["C"], 3)
            fidelity = float(abs(np.vdot(target, Statevector.from_instruction(circuit).data)) ** 2)
            if frame.f_lineno == 125:
                before.append(fidelity)
            else:
                records.append((before.pop(), fidelity))
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        gates = upstream.AQCE_python(state, M_0=0, M_delta=1, M_max=4, N=2)
    finally:
        sys.settrace(previous)
    assert len(gates) == 4
    assert len(records) == 2 * (1 + 2 + 3 + 4)
    assert all(after >= initial - 2e-13 for initial, after in records)
    assert pairs == [(0, 1), (0, 2), (1, 2)] * len(records)
    assert records[-1][1] > records[0][0]


def test_deterministic_replay_ignores_seed() -> None:
    """A deterministic library must not be presented as independent seed trials."""
    target = random_statevector(8, seed=99).data
    first = study.aqce_candidates(target, gate_budgets=(3,), sweeps=2, seed=300)[0]
    second = study.aqce_candidates(target, gate_budgets=(3,), sweeps=2, seed=400)[0]
    np.testing.assert_array_equal(Operator(first.circuit).data, Operator(second.circuit).data)
    assert first.metadata["deterministic"]
    assert not first.metadata["seed_used"]
    assert not first.metadata["noise_fine_tuning"]
    assert first.metadata["sweep_definition"] == "forward_then_backward"


def test_zero_budget_is_empty_and_zero_sweeps_retain_published_growth_update() -> None:
    """Support explicit zero controls without silently adding optimization."""
    target = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    empty = study.aqce_candidates(target, gate_budgets=(0,), sweeps=0)[0]
    assert not empty.circuit.data
    assert empty.metadata["ideal_fidelity"] == pytest.approx(0.5)
    grown = study.aqce_candidates(random_statevector(8, seed=83).data, gate_budgets=(2,), sweeps=0)[0]
    assert len(grown.circuit.data) == 2
    assert grown.metadata["scheduled_sweep_updates"] == 0
    assert grown.metadata["scheduled_growth_updates"] == 1
    assert grown.metadata["scheduled_unitary_updates"] == 1


def test_explicit_initialization_and_block_growth_are_recorded() -> None:
    """Non-default M0/delta M must reach exact budgets and report all SVD work."""
    candidates = study.aqce_candidates(
        random_statevector(16, seed=83).data,
        gate_budgets=(0, 2, 4),
        sweeps=3,
        initial_gates=2,
        growth_increment=2,
    )
    assert [candidate.metadata["initial_gates"] for candidate in candidates] == [0, 2, 2]
    assert [len(candidate.circuit.data) for candidate in candidates] == [0, 2, 4]
    assert candidates[-1].metadata["scheduled_unitary_updates"] == 6 * (2 + 4) + 2
    assert candidates[-1].metadata["requested_initial_gates"] == 2
    assert candidates[-1].metadata["growth_increment"] == 2


def test_costs_charge_setup_once_and_each_independent_budget_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cumulative CPU must include all restarted budgets, not only the winner."""
    readings = iter([0.0, 1.0, 2.0, 5.0, 6.0, 10.0])
    monkeypatch.setattr(study.time, "process_time", lambda: next(readings))
    first, second = study.aqce_candidates(random_statevector(4, seed=84).data, gate_budgets=(0, 1), sweeps=1)
    assert first.metadata["cpu_seconds"] == pytest.approx(4.0)
    assert first.metadata["setup_cpu_seconds"] == pytest.approx(1.0)
    assert first.metadata["cumulative_cpu_seconds"] == pytest.approx(4.0)
    assert second.metadata["cpu_seconds"] == pytest.approx(4.0)
    assert second.metadata["setup_cpu_seconds"] == pytest.approx(0.0)
    assert second.metadata["cumulative_cpu_seconds"] == pytest.approx(8.0)
    assert second.metadata["budget_reuse"] == "independent_restarts"


@pytest.mark.parametrize("budgets", [(), (-1,), (1, 1), (2, 1), (1.5,), (True,), [1, 2]])
def test_invalid_budgets_are_rejected(budgets: tuple[Any, ...] | list[int]) -> None:
    """Gate growth must have an explicit nonnegative strictly increasing budget."""
    with pytest.raises(ValueError, match="gate_budgets"):
        study.aqce_candidates([1, 0, 0, 0], gate_budgets=cast("tuple[int, ...]", budgets))


@pytest.mark.parametrize(
    "controls",
    [
        {"sweeps": -1},
        {"sweeps": 1.5},
        {"sweeps": True},
        {"seed": -1},
        {"initial_gates": -1},
        {"initial_gates": True},
        {"growth_increment": 1.5},
    ],
)
def test_invalid_controls_are_rejected(controls: dict[str, Any]) -> None:
    """Prevent silent changes to iteration counts or misleading seed metadata."""
    with pytest.raises(ValueError, match="nonnegative integer"):
        study.aqce_candidates([1, 0, 0, 0], gate_budgets=(1,), **controls)


@pytest.mark.parametrize(
    "controls",
    [
        {"growth_increment": 0},
        {"initial_gates": 3},
        {"growth_increment": 2},
    ],
)
def test_unreachable_growth_budgets_are_rejected(controls: dict[str, Any]) -> None:
    """Do not round, overshoot or silently change the paper's controls."""
    with pytest.raises(ValueError, match="reachable"):
        study.aqce_candidates([1, 0, 0, 0], gate_budgets=(2,), **controls)


def test_unsupported_connectivity_and_one_qubit_growth_are_explicit() -> None:
    """Unsupported pair sets and dimensions must fail explicitly."""
    with pytest.raises(ValueError, match="all_to_all"):
        study.aqce_candidates([1, 0, 0, 0], gate_budgets=(1,), connectivity="line")
    with pytest.raises(ValueError, match="at least two qubits"):
        study.aqce_candidates([1, 0], gate_budgets=(1,))


@pytest.mark.parametrize("target", [[1, 0, 0], [0, 0, 0, 0], [np.nan, 0, 0, 0], [2, 0, 0, 0]])
def test_invalid_targets_follow_shared_validation(target: list[float]) -> None:
    """Do not quietly normalize or discard phase information from bad inputs."""
    with pytest.raises(ValueError, match="normalized finite"):
        study.aqce_candidates(target, gate_budgets=(1,))
