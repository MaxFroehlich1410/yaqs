# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent dense-oracle checks of the published AQCE update schedule."""

from __future__ import annotations

import itertools
import operator
from collections import UserList
from functools import cache
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector, random_statevector, random_unitary

from experiments.state_preparation_competitors import aqce
from experiments.state_preparation_competitors import aqce_schedule as study

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from types import ModuleType
    from typing import SupportsIndex

    from numpy.typing import NDArray

pytest.importorskip("qulacs")


def state(upstream: ModuleType, vector: NDArray[np.complex128]) -> study.StateProtocol:
    """Load a Qulacs state without changing the supplied amplitudes.

    Returns:
        An independently allocated state.
    """
    result = upstream.QuantumState(vector.size.bit_length() - 1)
    result.load(vector)
    return result


def dense_circuit(gates: list[Any], num_qubits: int) -> Operator:
    """Build the full operator independently of the AQCE adapter.

    Returns:
        A Qiskit operator with the original ordered, complex gate matrices.
    """
    circuit = QuantumCircuit(num_qubits)
    for gate in gates:
        circuit.unitary(gate.get_matrix(), gate.get_target_index_list())
    return Operator(circuit)


@cache
def matrix_units(num_qubits: int, pair: tuple[int, int]) -> NDArray[np.complex128]:
    """Embed the sixteen matrix units using Qiskit's subsystem composition.

    Returns:
        Dense full-system operators indexed by their local row and column.
    """
    units = np.zeros((4, 4, 2**num_qubits, 2**num_qubits), dtype=complex)
    for row, column in itertools.product(range(4), repeat=2):
        local = np.zeros((4, 4), dtype=complex)
        local[row, column] = 1
        units[row, column] = Operator(np.eye(2**num_qubits)).compose(Operator(local), qargs=list(pair)).data
    return units


def overlap_matrix(
    left: NDArray[np.complex128], right: NDArray[np.complex128], pair: tuple[int, int]
) -> NDArray[np.complex128]:
    """Obtain the local transition matrix from sixteen full-state overlaps.

    Returns:
        A matrix whose polar factor maximizes the gate-local overlap.
    """
    units = matrix_units(left.size.bit_length() - 1, pair)
    return np.einsum("i,abij,j->ab", right.conj(), units, left).conj()


def fidelity(target: NDArray[np.complex128], gates: list[Any]) -> float:
    """Evaluate the complete circuit with Qiskit's dense operator.

    Returns:
        Squared target overlap from the computational zero state.
    """
    vector = dense_circuit(gates, target.size.bit_length() - 1).data[:, 0]
    return float(abs(np.vdot(target, vector)) ** 2)


class RecordingGates(UserList[Any]):
    """Observe individual assignments without replacing a numerical routine."""

    def __init__(self, values: list[Any], audit: Callable[[int, list[Any], list[Any]], None] | None = None) -> None:
        """Retain a callback for the circuit immediately around each update."""
        super().__init__(values)
        self.audit = audit

    @overload
    def __setitem__(self, i: SupportsIndex, item: object) -> None: ...

    @overload
    def __setitem__(self, i: slice, item: Iterable[Any]) -> None: ...

    def __setitem__(self, i: SupportsIndex | slice, item: object) -> None:
        """Record every scalar replacement and leave ordinary list semantics."""
        before = list(self)
        if isinstance(i, slice):
            super().__setitem__(i, cast("Iterable[Any]", item))
        else:
            super().__setitem__(i, item)
            if self.audit is not None:
                self.audit(operator.index(i), before, list(self))


@pytest.mark.parametrize(("num_qubits", "count"), [(3, 1), (4, 3)])
def test_initialization_chooses_density_eigenvectors_in_residual_order(num_qubits: int, count: int) -> None:
    """Verify principal-eigenvalue selection and all four columns at each step."""
    upstream = aqce.load_upstream()
    target = random_statevector(2**num_qubits, seed=127).data
    original = target.copy()
    gates = study.initialize(upstream, state(upstream, target), count)
    assert len(gates) == count
    residual = target.copy()
    for gate in reversed(gates):
        indices = gate.get_target_index_list()
        pair = (indices[0], indices[1])
        matrix = gate.get_matrix()
        density = overlap_matrix(residual, residual, pair)
        maximum = max(
            np.linalg.eigvalsh(overlap_matrix(residual, residual, candidate))[-1]
            for candidate in itertools.combinations(range(num_qubits), 2)
        )
        diagonal = matrix.conj().T @ density @ matrix
        expected = np.linalg.eigvalsh(density)[::-1]
        np.testing.assert_allclose(diagonal, np.diag(expected), atol=3e-14, rtol=0)
        assert diagonal[0, 0].real == pytest.approx(maximum, abs=2e-14)
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(4), atol=2e-14, rtol=0)
        residual = Statevector(residual).evolve(Operator(matrix.conj().T), qargs=list(pair)).data
    np.testing.assert_array_equal(target, original)


@pytest.mark.parametrize("backward", [False, True])
def test_every_gate_update_reaches_the_all_pair_dense_oracle_maximum(*, backward: bool) -> None:
    """Check order, adaptivity and monotonicity using full-state matrix units."""
    upstream = aqce.load_upstream()
    target = random_statevector(16, seed=351).data
    pairs = [(3, 0), (1, 2), (0, 2), (3, 1)]
    initial = [upstream.DenseMatrix(pair, random_unitary(4, seed=400 + index).data) for index, pair in enumerate(pairs)]
    order: list[int] = []

    def audit(index: int, before: list[Any], after: list[Any]) -> None:
        """Compare the local optimum with all independently embedded pairs."""
        order.append(index)
        left = dense_circuit(before[:index], 4).data[:, 0]
        right = dense_circuit(before[index + 1 :], 4).data.conj().T @ target
        expected = max(
            np.linalg.svd(overlap_matrix(left, right, pair), compute_uv=False).sum() ** 2
            for pair in itertools.combinations(range(4), 2)
        )
        assert fidelity(target, after) == pytest.approx(expected, abs=3e-13)
        assert fidelity(target, after) >= fidelity(target, before) - 3e-13
        assert all(old is new for offset, (old, new) in enumerate(zip(before, after, strict=True)) if offset != index)
        matrix = after[index].get_matrix()
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(4), atol=3e-14, rtol=0)

    gates = RecordingGates(initial, audit)
    study.optimize_pass(upstream, state(upstream, target), cast("list[Any]", gates), backward=backward)
    assert order == (list(reversed(range(4))) if backward else list(range(4)))


def test_sweep_repeats_forward_then_backward_without_skipping_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """The paper counts two complete directional passes as one sweep."""
    upstream = aqce.load_upstream()
    target = state(upstream, random_statevector(8, seed=421).data)
    gates = study.initialize(upstream, target, 2)
    calls: list[bool] = []

    def record_pass(
        actual_upstream: ModuleType, actual_target: study.StateProtocol, actual_gates: list[Any], *, backward: bool
    ) -> None:
        """Record the public directional pass contract without evolving states."""
        assert actual_upstream is upstream
        assert actual_target is target
        assert actual_gates is gates
        calls.append(backward)

    monkeypatch.setattr(study, "optimize_pass", record_pass)
    study.sweep(upstream, target, gates, 3)
    assert calls == [False, True] * 3


@pytest.mark.parametrize("increment", [1, 3])
def test_growth_updates_only_new_identity_block_against_old_circuit_residual(
    increment: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A growth block is optimized backward while all established gates stay fixed."""
    upstream = aqce.load_upstream()
    target = random_statevector(16, seed=451).data
    gates = [
        upstream.DenseMatrix([3, 0], random_unitary(4, seed=460).data),
        upstream.DenseMatrix([1, 2], random_unitary(4, seed=461).data),
    ]
    original = list(gates)
    residual = dense_circuit(original, 4).data.conj().T @ target
    original_pass = study.optimize_pass
    calls: list[bool] = []

    def inspect_pass(
        actual_upstream: ModuleType, actual_target: study.StateProtocol, block: list[Any], *, backward: bool
    ) -> None:
        """Check the exact initial block and effective target before refinement."""
        calls.append(backward)
        assert actual_upstream is upstream
        assert len(block) == increment
        np.testing.assert_allclose(actual_target.get_vector(), residual, atol=2e-14, rtol=0)
        np.testing.assert_allclose(dense_circuit(block, 4).data, np.eye(16), atol=2e-14, rtol=0)
        original_pass(actual_upstream, actual_target, block, backward=backward)

    monkeypatch.setattr(study, "optimize_pass", inspect_pass)
    before = fidelity(target, gates)
    study.grow(upstream, state(upstream, target), gates, increment)
    assert calls == [True]
    assert len(gates) == len(original) + increment
    assert all(old is new for old, new in zip(original, gates[increment:], strict=True))
    assert fidelity(target, gates) >= before - 3e-13


@pytest.mark.parametrize("basis", [1, 2, 3])
def test_two_qubit_initialization_escapes_zero_overlap_basis_targets(basis: int) -> None:
    """Initialization must handle states orthogonal to the computational zero."""
    upstream = aqce.load_upstream()
    target = np.zeros(4, dtype=complex)
    target[basis] = np.exp(0.37j)
    gates = study.encode(upstream, state(upstream, target), initial_gates=1, growth_increment=1, max_gates=1, sweeps=1)
    assert fidelity(target, gates) == pytest.approx(1.0, abs=2e-14)


def test_nonlocal_complex_target_preserves_order_and_relative_phase() -> None:
    """A single adaptive gate must encode a complex nonadjacent Bell state."""
    upstream = aqce.load_upstream()
    target = np.zeros(8, dtype=complex)
    target[[0, 5]] = np.sqrt([0.23, 0.77]) * np.exp(1j * np.array([0.29, -0.63]))
    gates = study.encode(upstream, state(upstream, target), initial_gates=1, growth_increment=1, max_gates=1, sweeps=2)
    assert set(gates[0].get_target_index_list()) == {0, 2}
    assert fidelity(target, gates) == pytest.approx(1.0, abs=2e-14)


def test_encode_initializes_once_then_alternates_growth_and_full_sweeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Paper control parameters must determine an exact initialization/growth schedule."""
    upstream = aqce.load_upstream()
    target = state(upstream, random_statevector(8, seed=472).data)
    calls: list[tuple[str, int, int]] = []
    identity = upstream.DenseMatrix([0, 1], np.eye(4))

    def initialize(actual_upstream: ModuleType, actual_target: study.StateProtocol, count: int) -> list[Any]:
        """Record the initial density-based construction.

        Returns:
            An identity placeholder for each requested gate.
        """
        assert actual_upstream is upstream
        assert actual_target is target
        calls.append(("initialize", count, count))
        return [identity.copy() for _ in range(count)]

    def sweep(actual_upstream: ModuleType, actual_target: study.StateProtocol, gates: list[Any], count: int) -> None:
        """Record the full fixed-length optimization budget."""
        assert actual_upstream is upstream
        assert actual_target is target
        calls.append(("sweep", len(gates), count))

    def grow(actual_upstream: ModuleType, actual_target: study.StateProtocol, gates: list[Any], increment: int) -> None:
        """Record growth without changing the public list identity."""
        assert actual_upstream is upstream
        assert actual_target is target
        calls.append(("grow", len(gates), increment))
        gates[:0] = [identity.copy() for _ in range(increment)]

    monkeypatch.setattr(study, "initialize", initialize)
    monkeypatch.setattr(study, "sweep", sweep)
    monkeypatch.setattr(study, "grow", grow)
    gates = study.encode(upstream, target, initial_gates=2, growth_increment=2, max_gates=6, sweeps=3)
    assert len(gates) == 6
    assert calls == [
        ("initialize", 2, 2),
        ("sweep", 2, 3),
        ("grow", 2, 2),
        ("sweep", 4, 3),
        ("grow", 4, 2),
        ("sweep", 6, 3),
    ]


def test_zero_budget_supports_one_qubit_and_empty_directional_passes() -> None:
    """Empty controls have no two-qubit requirement and preserve the target."""
    upstream = aqce.load_upstream()
    target = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    actual_target = state(upstream, target)
    gates = study.encode(upstream, actual_target, initial_gates=0, growth_increment=1, max_gates=0, sweeps=0)
    assert gates == []
    study.optimize_pass(upstream, actual_target, gates, backward=False)
    study.optimize_pass(upstream, actual_target, gates, backward=True)
    study.sweep(upstream, actual_target, gates, 0)
    np.testing.assert_array_equal(actual_target.get_vector(), target)


@pytest.mark.parametrize(
    "controls",
    [
        {"initial_gates": -1},
        {"initial_gates": True},
        {"initial_gates": 1.5},
        {"initial_gates": 5},
        {"growth_increment": 0},
        {"growth_increment": -1},
        {"growth_increment": True},
        {"growth_increment": 1.5},
        {"growth_increment": 2},
        {"max_gates": -1},
        {"max_gates": True},
        {"max_gates": 1.5},
        {"sweeps": -1},
        {"sweeps": True},
        {"sweeps": 1.5},
    ],
)
def test_encode_rejects_invalid_controls_before_work(controls: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject ambiguous iteration budgets and unreachable endpoint sizes."""
    upstream = aqce.load_upstream()
    target = state(upstream, random_statevector(8, seed=489).data)

    def unexpected_initialization(*_args: object, **_kwargs: object) -> list[Any]:
        """Fail if validation starts expensive work before checking controls.

        Raises:
            AssertionError: Always, because initialization must not occur.
        """
        msg = "invalid controls reached initialization"
        raise AssertionError(msg)

    monkeypatch.setattr(study, "initialize", unexpected_initialization)
    kwargs = {"initial_gates": 1, "growth_increment": 1, "max_gates": 4, "sweeps": 1} | controls
    with pytest.raises(ValueError, match="AQCE"):
        study.encode(upstream, target, **kwargs)


def test_positive_gate_budget_rejects_one_qubit() -> None:
    """Two-qubit updates cannot silently operate on an unsupported register."""
    upstream = aqce.load_upstream()
    target = state(upstream, np.array([1, 0], dtype=complex))
    with pytest.raises(ValueError, match="at least two qubits"):
        study.encode(upstream, target, initial_gates=1, growth_increment=1, max_gates=1, sweeps=1)
