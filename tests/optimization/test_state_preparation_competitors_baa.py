# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Verify the pinned official BAA adapter against independent Qiskit oracles."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit import AncillaRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Statevector

from experiments.state_preparation_competitors import baa

# Direct helper tests exercise fail-closed dependency and output guards.
from experiments.state_preparation_competitors.baa import (
    _load_upstream,  # ruff: ignore[PLC2701]
    _loss_grid,  # ruff: ignore[PLC2701]
    _verify_candidate,  # ruff: ignore[PLC2701]
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture
def upstream() -> type:
    """Skip numerical adapter tests when the optional upstream package is absent.

    Returns:
        The authenticated original initializer class.
    """
    pytest.importorskip("qclib.state_preparation")
    return _load_upstream()


def complex_target(qubits: int, seed: int) -> NDArray[np.complex128]:
    """Create complex amplitudes that detect conjugation and ordering errors.

    Returns:
        A normalized complex state vector.
    """
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=2**qubits) + 1j * rng.normal(size=2**qubits)
    return vector / np.linalg.norm(vector)


@pytest.mark.usefixtures("upstream")
@pytest.mark.parametrize("qubits", [2, 3])
def test_official_candidates_are_unitary_bounded_and_fully_accounted(qubits: int) -> None:
    """Every candidate satisfies its actual squared-fidelity bound with honest costs."""
    target = complex_target(qubits, 3)
    original = target.copy()
    candidates = baa.baa_candidates(target, fidelity_losses=(0.2, 0.05, 0.2))
    assert [item.metadata["requested_fidelity_loss"] for item in candidates] == [0.0, 0.05, 0.2]
    assert len({item.metadata["candidate_id"] for item in candidates}) == 3
    np.testing.assert_array_equal(target, original)
    cumulative = 0.0
    for index, item in enumerate(candidates):
        metadata = item.metadata
        circuit = item.circuit
        prepared = Statevector.from_instruction(circuit).data
        expected = float(abs(np.vdot(target, prepared)) ** 2)
        assert metadata["raw_squared_fidelity"] == pytest.approx(expected, abs=1e-14)
        assert metadata["actual_squared_fidelity"] >= 1 - metadata["requested_fidelity_loss"] - 1e-9
        assert metadata["actual_squared_fidelity"] == metadata["ideal_fidelity"]
        assert metadata["actual_fidelity_loss"] == 1 - metadata["ideal_fidelity"]
        assert circuit.num_qubits == qubits
        assert circuit.num_ancillas == circuit.num_clbits == 0
        assert Operator(circuit).is_unitary()
        assert all(gate.operation.num_qubits <= 2 for gate in circuit.data)
        assert metadata["method"] == "baa_lrsp"
        assert metadata["status"] == "complete"
        assert metadata["deterministic"] is True
        assert metadata["synthesis_cpu_seconds"] > 0
        assert metadata["verification_cpu_seconds"] > 0
        assert metadata["cpu_seconds"] >= metadata["synthesis_cpu_seconds"] + metadata["verification_cpu_seconds"]
        assert metadata["setup_cpu_seconds"] > 0 if index == 0 else metadata["setup_cpu_seconds"] == 0
        cumulative += metadata["cpu_seconds"]
        assert metadata["cumulative_cpu_seconds"] == pytest.approx(cumulative, abs=1e-12)
        json.dumps(metadata, allow_nan=False)
    assert candidates[0].metadata["ideal_fidelity"] == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize(("strategy", "use_low_rank"), [("brute_force", True), ("greedy", False)])
def test_definition_expansion_preserves_the_original_upstream_state(
    upstream: type, strategy: str, *, use_low_rank: bool
) -> None:
    """The adapter's exact expansion preserves official BAA output up to global phase."""
    target = complex_target(3, 3)
    candidate = baa.baa_candidates(target, fidelity_losses=(0.2,), strategy=strategy, use_low_rank=use_low_rank)[1]
    gate = upstream(target, opt_params=candidate.metadata["upstream_options"])
    raw_state = Statevector.from_instruction(gate.definition).data
    expanded_state = Statevector.from_instruction(candidate.circuit).data
    assert abs(np.vdot(raw_state, expanded_state)) ** 2 == pytest.approx(1, abs=1e-12)
    assert candidate.metadata["upstream_fidelity_loss_estimate"] == pytest.approx(gate.node.total_fidelity_loss)
    assert candidate.metadata["actual_squared_fidelity"] < 0.999
    assert candidate.metadata["actual_squared_fidelity"] >= 0.8 - 1e-9


@pytest.mark.usefixtures("upstream")
@pytest.mark.parametrize("qubits", [1, 6])
def test_exact_endpoint_handles_complex_targets_at_both_supported_extremes(qubits: int) -> None:
    """The zero-loss endpoint supports a full complex six-qubit state without ancillas."""
    target = complex_target(qubits, 17)
    candidate = baa.baa_candidates(target, fidelity_losses=(0.0,))[0]
    assert candidate.metadata["ideal_fidelity"] == pytest.approx(1, abs=1e-12)
    assert candidate.metadata["requested_fidelity_loss"] == 0
    assert candidate.metadata["upstream_fidelity_loss_estimate"] == pytest.approx(0, abs=1e-12)


@pytest.mark.usefixtures("upstream")
def test_repeated_synthesis_is_deterministic() -> None:
    """Independent calls use the same deterministic original synthesis, not a random seed."""
    target = complex_target(3, 3)
    first = baa.baa_candidates(target, fidelity_losses=(0.2, 0.5))
    second = baa.baa_candidates(target, fidelity_losses=(0.5, 0.2))
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(Operator(left.circuit).data, Operator(right.circuit).data)
        assert left.metadata["candidate_id"] == right.metadata["candidate_id"]
        assert left.metadata["raw_squared_fidelity"] == right.metadata["raw_squared_fidelity"]


def test_loss_grid_canonicalizes_bounds_and_includes_zero_once() -> None:
    """Bounds represent squared infidelity and include a reproducible exact control."""
    assert _loss_grid((1, 0.2, np.float64(0.2), -0.0)) == (0.0, 0.2, 1.0)


@pytest.mark.parametrize("bounds", [(), [], (np.nan,), (np.inf,), (-0.01,), (1.01,)])
def test_invalid_loss_ranges_fail_before_synthesis(bounds: tuple) -> None:
    """Nonfinite, missing, or out-of-range bounds cannot silently alter the experiment."""
    with pytest.raises(ValueError, match=r"fidelity|Fidelity"):
        _loss_grid(bounds)


@pytest.mark.parametrize("bounds", [(True,), (np.bool_(0),), ("0.1",), (0.1j,)])
def test_nonreal_or_boolean_loss_bounds_are_rejected(bounds: tuple) -> None:
    """Boolean, textual, and complex bounds are not valid numerical options."""
    with pytest.raises(TypeError, match="real numbers"):
        _loss_grid(bounds)


@pytest.mark.parametrize("target", [[1, 0, 0], [2, 0], [np.nan, 0], [1] + [0] * 127])
def test_invalid_targets_fail_without_loading_upstream(target: list, monkeypatch: pytest.MonkeyPatch) -> None:
    """Target validation precedes loading optional dependencies or running synthesis."""
    monkeypatch.setattr(baa, "_load_upstream", lambda: pytest.fail("Should validate first"))
    with pytest.raises(ValueError, match="state vector"):
        baa.baa_candidates(target, fidelity_losses=(0.0,))


@pytest.mark.parametrize("options", [{"strategy": "unknown"}, {"use_low_rank": 1}])
def test_unknown_options_fail_without_loading_upstream(options: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Typos cannot fall through to upstream defaults and change the declared method."""
    monkeypatch.setattr(baa, "_load_upstream", lambda: pytest.fail("Should validate first"))
    with pytest.raises(ValueError, match=r"strategies|bool"):
        baa.baa_candidates([1, 0], fidelity_losses=(0.0,), **options)


@pytest.mark.parametrize("mismatch", ["qclib", "qiskit"])
def test_upstream_version_mismatch_is_rejected(mismatch: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrapper never substitutes an unaudited installed algorithm version."""
    versions = {"qclib": baa.QCLIB_VERSION, "qiskit": baa.QISKIT_VERSION, mismatch: "0.0.0"}
    monkeypatch.setattr(baa, "version", versions.__getitem__)
    with pytest.raises(RuntimeError, match=f"requires {mismatch}=="):
        _load_upstream()


def test_missing_optional_package_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dependency errors point to the isolated installation recipe."""

    def missing(package: str) -> str:
        raise PackageNotFoundError(package)

    monkeypatch.setattr(baa, "version", missing)
    with pytest.raises(RuntimeError, match=r"requirements\.txt"):
        _load_upstream()


def test_missing_runtime_dependency_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """QCLib's undeclared Aer dependency is diagnosed rather than hidden."""

    def missing(module: str) -> None:
        raise ImportError(module)

    monkeypatch.setattr(baa, "version", {"qclib": baa.QCLIB_VERSION, "qiskit": baa.QISKIT_VERSION}.__getitem__)
    monkeypatch.setattr(baa, "import_module", missing)
    with pytest.raises(RuntimeError, match="qiskit-aer"):
        _load_upstream()


@pytest.mark.usefixtures("upstream")
def test_changed_upstream_source_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A matching version string alone is insufficient to authenticate patched source."""
    monkeypatch.setitem(baa.UPSTREAM_HASHES, "qclib.state_preparation.baa_lowrank", "0" * 64)
    with pytest.raises(RuntimeError, match="source differs"):
        _load_upstream()


@pytest.mark.parametrize("kind", ["width", "ancilla", "classical", "parameter", "opaque", "reset", "nonunitary"])
def test_invalid_upstream_circuits_fail_closed(kind: str) -> None:
    """Unsupported width, ancillas, and nonunitary circuits cannot enter benchmarks."""
    target = np.array([1, 0], dtype=np.complex128)
    circuit = QuantumCircuit(1)
    if kind == "width":
        circuit = QuantumCircuit(2)
    elif kind == "ancilla":
        circuit = QuantumCircuit(AncillaRegister(1))
    elif kind == "classical":
        circuit = QuantumCircuit(1, 1)
    elif kind == "parameter":
        circuit.ry(Parameter("theta"), 0)
    elif kind == "opaque":
        circuit = QuantumCircuit(3)
        circuit.append(UnitaryGate(np.eye(8)), [0, 1, 2])
        target = np.array([1] + [0] * 7, dtype=np.complex128)
    elif kind == "reset":
        circuit.reset(0)
    else:
        circuit.append(UnitaryGate(np.eye(2) * 0.5, check_input=False), [0])
    with pytest.raises(RuntimeError, match="BAA"):
        _verify_candidate(circuit, target, 1.0)


def test_fidelity_bound_uses_actual_squared_overlap() -> None:
    """An optimistic approximation estimate cannot bypass independently measured loss."""
    target = np.array([1, 0], dtype=np.complex128)
    circuit = QuantumCircuit(1)
    circuit.ry(np.pi / 3, 0)
    result = _verify_candidate(circuit, target, 0.25)
    assert result["actual_squared_fidelity"] == pytest.approx(0.75, abs=1e-14)
    with pytest.raises(RuntimeError, match="fidelity-loss bound"):
        _verify_candidate(circuit, target, 0.1)


def test_nonfinite_upstream_estimate_cannot_be_persisted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid upstream metadata fails rather than producing nonstandard JSON."""

    def initializer(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(definition=QuantumCircuit(1), node=SimpleNamespace(total_fidelity_loss=np.nan))

    monkeypatch.setattr(baa, "_load_upstream", lambda: initializer)
    with pytest.raises(RuntimeError, match="nonfinite approximation estimate"):
        baa.baa_candidates([1, 0], fidelity_losses=(0.0,))
