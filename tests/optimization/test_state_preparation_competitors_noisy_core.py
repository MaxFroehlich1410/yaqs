# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent Qiskit and finite-difference oracles for shared noisy objectives."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import DensityMatrix, Operator, Statevector

from experiments.legacy_topdown_dense import noise_channel
from experiments.state_preparation_competitors import noisy_core as study


def _target(n: int, seed: int = 42) -> np.ndarray:
    """Return a complex state with independently generated amplitudes."""
    rng = np.random.default_rng(seed)
    state = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return state / np.linalg.norm(state)


def _oracle(circuit: QuantumCircuit, hardware: str | None) -> np.ndarray:
    """Return Qiskit's independently implemented superoperator density output."""
    rho = DensityMatrix.from_label("0" * circuit.num_qubits)
    for item in circuit.data:
        sites = [circuit.find_bit(q).index for q in item.qubits]
        rho = rho.evolve(Operator(item.operation), qargs=sites)
        if hardware is not None:
            rho = rho.evolve(noise_channel(hardware, len(sites)), qargs=sites)
    return rho.data


@pytest.mark.parametrize("hardware", [None, "IBM", "Google"])
@pytest.mark.parametrize("n", [1, 2, 3, 6])
def test_density_and_gradient_against_independent_oracles(hardware: str | None, n: int) -> None:
    """Complex states, reversed CNOT supports and shared angles obey the same channel."""
    p = ParameterVector("a", 3)
    circuit = QuantumCircuit(n)
    circuit.ry(2 * p[0] + 0.3, 0)
    circuit.rx(p[1], n - 1)
    if n > 1:
        circuit.cx(1, 0)
        circuit.rzz(p[2], 0, 1)
    circuit.rz(p[0] + p[2], n - 1)
    circuit.rx(p[1] * p[2], 0)
    target = _target(n)
    objective = study.DenseObjective(circuit, target, hardware)
    theta = np.array([0.23, -0.47, 0.81])
    expected = _oracle(objective.bind(theta), hardware)
    actual = objective.density(theta)
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    assert np.linalg.eigvalsh(actual).min() > -1e-13
    loss, gradient = objective.loss_gradient(theta)
    assert loss == pytest.approx(1 - np.vdot(target, expected @ target).real, abs=2e-14)
    assert objective.fidelity(theta) == pytest.approx(1 - loss)
    delta = 1e-5
    finite_difference = []
    for axis in np.eye(theta.size):
        plus = _oracle(objective.bind(theta + delta * axis), hardware)
        minus = _oracle(objective.bind(theta - delta * axis), hardware)
        finite_difference.append(-np.vdot(target, (plus - minus) @ target).real / (2 * delta))
    np.testing.assert_allclose(gradient, finite_difference, atol=2e-10, rtol=2e-7)


@pytest.mark.parametrize("native", [True, False])
def test_ideal_vjp_uses_fixed_upstream_costate(*, native: bool) -> None:
    """A noisy residual drives the ideal Jacobian, without differentiating feedback."""
    p = ParameterVector("p", 2)
    qc = QuantumCircuit(2)
    qc.rx(p[0], 1)
    qc.ry(p[1], 0)
    qc.cx(1, 0)
    qc.rxx(p[0] - p[1], 0, 1)
    target = _target(2)
    engine = study.DenseObjective(qc, target, "Google", compile_native=native)
    theta = np.array([0.6, -0.9])
    residual = engine.density(theta) - np.outer(target, target.conj())
    costate = residual / np.linalg.norm(residual)
    actual = engine.density_vjp(theta, costate)
    eps = 1e-5
    expected = [
        np.vdot(
            costate,
            _oracle(engine.bind(theta + eps * direction), None) - _oracle(engine.bind(theta - eps * direction), None),
        ).real
        / (2 * eps)
        for direction in np.eye(2)
    ]
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_allclose(engine.ideal_density(theta), _oracle(engine.bind(theta), None), atol=1e-14)


def test_parameter_order_fixed_phase_and_cancellation_are_preserved() -> None:
    """A zero-angle gate still incurs noise and is never removed during binding."""
    a, z = Parameter("a"), Parameter("z")
    qc = QuantumCircuit(1)
    qc.rz(z, 0)
    qc.rz(-z, 0)
    qc.rx(a, 0)
    qc.global_phase = z
    saved = qc.copy()
    engine = study.DenseObjective(qc, np.array([1, 0]), "IBM")
    assert engine.parameters == (a, z)
    assert qc == saved
    assert len(engine.bind([0, 0]).data) == len(engine.bind([0.8, 0.3]).data)
    assert engine.fidelity([0, 0]) < 1
    np.testing.assert_allclose(
        Operator(engine.bind([0.8, 0.3])).data, Operator(qc.assign_parameters({a: 0.8, z: 0.3})).data, atol=1e-14
    )
    assert engine.loss_gradient([0.8, 0.3])[1][1] == pytest.approx(0, abs=1e-14)


def test_empty_and_parameter_free_circuits() -> None:
    """Zero-dimensional optimization returns real empty gradients and exact overlaps."""
    for n in (1, 6):
        target = _target(n)
        engine = study.DenseObjective(QuantumCircuit(n), target)
        loss, gradient = engine.loss_gradient([])
        assert loss == pytest.approx(1 - abs(target[0]) ** 2)
        assert gradient.shape == (0,)
        assert engine.density_vjp([], np.eye(2**n)).shape == (0,)
    fixed = QuantumCircuit(2)
    fixed.h(0)
    fixed.cx(0, 1)
    target = Statevector.from_instruction(fixed).data
    engine = study.DenseObjective(fixed, target, None)
    assert engine.fidelity([]) == pytest.approx(1)


@pytest.mark.parametrize("values", [[True], [1j], [float("nan")], [float("inf")], [[0.0]], []])
def test_invalid_parameters(values: list) -> None:
    """Reject malformed angles before circuit or channel simulation."""
    with pytest.raises(ValueError, match="parameter vector"):
        study.real_parameters(values, 1)


@pytest.mark.parametrize("value", [True, -1, 1.5, "2"])
def test_invalid_budgets(value: object) -> None:
    """Integer budgets cannot silently round values or interpret booleans."""
    with pytest.raises(ValueError, match="integer"):
        study.nonnegative_integer(cast("int", value), "steps")


def test_validation_and_nonmutation() -> None:
    """Fail explicitly for unsupported instructions, targets, models and costates."""
    study.nonnegative_integer(0, "steps")
    source = np.array([0.2])
    result = study.real_parameters(source, 1)
    result[0] = 9
    assert source[0] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="normalized"):
        study.DenseObjective(QuantumCircuit(1), [1, 1])
    with pytest.raises(ValueError, match="match"):
        study.DenseObjective(QuantumCircuit(2), [1, 0])
    with pytest.raises(ValueError, match="IBM/Google"):
        study.DenseObjective(QuantumCircuit(1), [1, 0], "bad")
    with pytest.raises(ValueError, match="classical"):
        study.DenseObjective(QuantumCircuit(1, 1), [1, 0])
    nonlocal_gate = QuantumCircuit(3)
    nonlocal_gate.cx(0, 2)
    with pytest.raises(ValueError, match="nearest-neighbor"):
        study.DenseObjective(nonlocal_gate, _target(3))
    reset = QuantumCircuit(1)
    reset.reset(0)
    with pytest.raises(TypeError, match="unitary"):
        study.DenseObjective(reset, [1, 0])
    p = Parameter("p")
    unsupported = QuantumCircuit(1)
    unsupported.u(p, 0, 0, 0)
    with pytest.raises(ValueError, match="Pauli rotations"):
        study.DenseObjective(unsupported, [1, 0], compile_native=False)
    engine = study.DenseObjective(QuantumCircuit(1), [1, 0])
    for bad in (np.eye(3), np.array([[0, 1], [0, 0]]), np.full((2, 2), np.nan)):
        with pytest.raises(ValueError, match="Hermitian"):
            engine.density_vjp([], bad)
