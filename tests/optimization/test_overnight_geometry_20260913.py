# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent dense derivative, gauge, PSD, and resume checks for pure geometry."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from experiments.overnight_state_preparation_20260913_0108 import channels, geometry
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _problem() -> tuple[ParameterizedCircuit, NDArray, NDArray]:
    """Create all scalar generators with shared parameters and reversed supports.

    Returns:
        A q3 circuit, original theta and asymmetric normalized target.
    """
    names = ["rx", "ry", "rz", "p", "rxx", "ryy", "rzz", "cp"]
    gates = [ParameterizedGate("h", (2,)), ParameterizedGate("cx", (2, 0))]
    gates.extend(
        ParameterizedGate(
            name,
            (2, 0) if index >= 4 else (index % 3,),
            param_index=index % 6,
            angle_scale=(-1) ** index * (0.7 + index / 4),
            angle_offset=0.21 * index,
            noise_enabled=index % 2 == 0,
        )
        for index, name in enumerate(names)
    )
    rng = np.random.default_rng(93)
    target = rng.normal(size=8) + 1j * rng.normal(size=8)
    target /= np.linalg.norm(target)
    return ParameterizedCircuit(3, gates, num_params=7), rng.normal(size=7), target


def _qiskit_state(circuit: ParameterizedCircuit, theta: NDArray) -> NDArray:
    """Use Qiskit's own gate matrices and native little-endian state ordering.

    Returns:
        Independently propagated final pure state.
    """
    reference = QuantumCircuit(circuit.num_qubits)
    for gate in circuit.gates:
        method = getattr(reference, gate.name)
        if gate.is_parametric:
            angle = gate.angle_offset + (0 if gate.param_index is None else gate.angle_scale * theta[gate.param_index])
            method(float(angle), *gate.sites)
        else:
            method(*gate.sites)
    return np.asarray(Statevector.from_instruction(reference).data)


def test_exact_jacobian_matches_independent_qiskit_finite_differences() -> None:
    """Check each full state derivative, not merely a particular target loss."""
    circuit, theta, _ = _problem()
    original_theta, original_gates = theta.copy(), copy.deepcopy(circuit.gates)
    state, jacobian, diagnostics = geometry.state_jacobian(circuit, theta)
    np.testing.assert_allclose(state, _qiskit_state(circuit, theta), atol=2e-14)
    expected = np.empty_like(jacobian)
    for index in range(theta.size):
        plus, minus = theta.copy(), theta.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        expected[index] = (_qiskit_state(circuit, plus) - _qiskit_state(circuit, minus)) / 2e-6
    np.testing.assert_allclose(jacobian, expected, atol=4e-10, rtol=2e-8)
    np.testing.assert_array_equal(jacobian[-1], 0)
    np.testing.assert_array_equal(theta, original_theta)
    assert circuit.gates == original_gates
    assert diagnostics["statevector_row_operator_applications"] == (theta.size + 1) * len(circuit.gates) + 8
    assert diagnostics["stochastic_trajectory_paths"] == 0


def test_metric_matches_formula_is_symmetric_psd_and_obeys_state_space_rank_bound() -> None:
    """The real projected Gram matrix has the mathematical QGT normalization."""
    circuit, theta, _ = _problem()
    state, jacobian, _ = geometry.state_jacobian(circuit, theta)
    overlap = jacobian @ state.conj()
    expected = np.real(jacobian.conj() @ jacobian.T - np.outer(overlap.conj(), overlap))
    metric, diagnostics = geometry.pure_state_metric(circuit, theta)
    np.testing.assert_allclose(metric, expected, atol=1e-14)
    np.testing.assert_array_equal(metric, metric.T)
    assert np.linalg.eigvalsh(metric).min() > -1e-12
    assert np.linalg.matrix_rank(metric, tol=1e-10) <= 2 * state.size - 2
    assert diagnostics["metric_array_bytes"] == metric.nbytes
    assert diagnostics["cpu_seconds"] >= diagnostics["jacobian_cpu_seconds"] >= 0


def test_metric_is_invariant_to_parameter_dependent_global_phase() -> None:
    """Adding an arbitrary vertical gauge component to every derivative changes no geometry."""
    circuit, theta, _ = _problem()
    state, jacobian, _ = geometry.state_jacobian(circuit, theta)
    gauge_gradient = np.arange(theta.size) - 2.3
    phase = np.exp(0.71j)
    transformed = phase * (jacobian + 1j * gauge_gradient[:, None] * state[None, :])
    np.testing.assert_allclose(
        geometry.metric_from_jacobian(phase * state, transformed),
        geometry.metric_from_jacobian(state, jacobian),
        atol=2e-14,
    )


def test_q2_metric_rank_and_local_infidelity_quadratic_form() -> None:
    """An overparameterized q2 circuit has at most six real physical tangent directions."""
    names = ["rx", "ry", "rz", "rx", "ry", "rz", "rxx", "rzz"] * 2
    gates = [
        ParameterizedGate(name, (1, 0) if name in {"rxx", "rzz"} else (index % 2,), param_index=index)
        for index, name in enumerate(names)
    ]
    circuit = ParameterizedCircuit(2, gates)
    theta = np.random.default_rng(47).normal(size=len(gates))
    metric, _ = geometry.pure_state_metric(circuit, theta)
    eigenvalues = np.linalg.eigvalsh(metric)
    assert eigenvalues.min() > -1e-12
    assert np.count_nonzero(eigenvalues > 1e-10) == 6
    direction = np.random.default_rng(87).normal(size=theta.size)
    direction /= np.linalg.norm(direction)
    initial = _qiskit_state(circuit, theta)
    displaced = _qiskit_state(circuit, theta + 1e-4 * direction)
    observed = (1 - abs(np.vdot(initial, displaced)) ** 2) / 1e-8
    assert observed == pytest.approx(direction @ metric @ direction, rel=2e-4, abs=2e-7)


@pytest.mark.parametrize("scales", [(1.0, 1.0), (0.4, -0.7), (1.0, -1.0)])
def test_shared_rotation_metric_includes_all_cross_terms(scales: tuple[float, float]) -> None:
    """Two shared RX occurrences differentiate the sum of their physical angles."""
    circuit = ParameterizedCircuit(
        1, [ParameterizedGate("rx", (0,), param_index=0, angle_scale=scale) for scale in scales]
    )
    metric, _ = geometry.pure_state_metric(circuit, np.array([0.3]))
    assert metric[0, 0] == pytest.approx(sum(scales) ** 2 / 4, abs=1e-14)


def test_q1_gauge_directions_zero_scales_and_empty_circuit() -> None:
    """RZ on zero is global phase; unused and zero-scale directions remain exactly flat."""
    circuit = ParameterizedCircuit(
        1,
        [
            ParameterizedGate("rz", (0,), param_index=0),
            ParameterizedGate("rx", (0,), param_index=1, angle_scale=0),
        ],
        num_params=3,
    )
    metric, _ = geometry.pure_state_metric(circuit, np.array([0.3, 0.7, -0.2]))
    np.testing.assert_allclose(metric, 0, atol=1e-30)
    empty = ParameterizedCircuit(1, [])
    state, jacobian, _ = geometry.state_jacobian(empty, np.empty(0))
    np.testing.assert_array_equal(state, [1, 0])
    assert jacobian.shape == (0, 2)
    metric, _ = geometry.pure_state_metric(empty, np.empty(0))
    assert metric.shape == (0, 0)


@pytest.mark.parametrize("damping", geometry.DAMPINGS)
def test_preconditioner_solves_regularized_system_and_is_descent(damping: float) -> None:
    """Singular geometry is handled by the fixed positive damping, not a pseudoinverse cutoff."""
    metric = np.array([[0.25, 0.25], [0.25, 0.25]])
    gradient = np.array([1.0, -1.0])
    direction, diagnostics = geometry.precondition(gradient, metric, damping)
    np.testing.assert_allclose((metric + damping * np.eye(2)) @ direction, -gradient, atol=1e-12)
    np.testing.assert_allclose(direction, -gradient / damping, rtol=1e-12)
    assert gradient @ direction < 0
    assert diagnostics["linear_solves"] == diagnostics["symmetric_eigensolves"] == 1


@pytest.mark.parametrize("invalid", ["damping", "negative", "asymmetric", "complex", "shape", "nan"])
def test_invalid_preconditioners_are_rejected(invalid: str) -> None:
    """Invalid geometry cannot be hidden by regularization or a silent cast."""
    metric, gradient, damping = np.eye(2), np.ones(2), 0.001
    if invalid == "damping":
        damping = 0.1
    elif invalid == "negative":
        metric[0, 0] = -1
    elif invalid == "asymmetric":
        metric[0, 1] = 0.4
    elif invalid == "complex":
        metric = metric.astype(complex)
    elif invalid == "shape":
        metric = np.eye(3)
    else:
        gradient[0] = np.nan
    with pytest.raises(ValueError, match=r"Preconditioner requires|not positive semidefinite"):
        geometry.precondition(gradient, metric, damping)


@pytest.mark.parametrize("invalid", ["complex", "nan", "data", "offset", "shape"])
def test_invalid_jacobian_domains_are_rejected(invalid: str) -> None:
    """Validate the pure kernel independently from the noisy objective."""
    circuit, theta, _ = _problem()
    if invalid == "complex":
        theta = theta.astype(complex)
    elif invalid == "nan":
        theta[0] = np.nan
    elif invalid == "data":
        circuit.gates[-1].data_map = lambda _x: 0.2
    elif invalid == "offset":
        circuit.gates[-1].angle_offset = np.inf
    else:
        theta = theta[:-1]
    with pytest.raises(ValueError, match="finite real scalar q1-q6"):
        geometry.state_jacobian(circuit, theta)


def _numerical_state(state: dict[str, Any]) -> dict[str, Any]:
    """Extract all optimization decisions without nondeterministic timing measurements.

    Returns:
        Parameters, gradients, pending line search, counters and trial decisions.
    """
    return {
        key: state[key]
        for key in (
            "theta",
            "gradient",
            "loss",
            "evaluations",
            "iterations",
            "metric_evaluations",
            "pending",
            "rejected_evaluations",
            "nonfinite_evaluations",
            "status",
            "trials",
            "density_direction_passes",
            "statevector_row_operator_applications",
        )
    }


@pytest.mark.parametrize("damping", geometry.DAMPINGS)
def test_noisy_armijo_is_monotone_and_exact_json_resume_preserves_pending_work(damping: float) -> None:
    """Budget-limited restart reuses the same metric, direction and backtracking trial."""
    circuit, theta, target = _problem()
    full = geometry.minimize(circuit, theta, target, damping=damping, max_evaluations=17)
    partial = geometry.minimize(circuit, theta, target, damping=damping, max_evaluations=2)
    restored = json.loads(json.dumps(partial, allow_nan=False))
    resumed = geometry.minimize(circuit, theta, target, damping=damping, max_evaluations=17, state=restored)
    assert _numerical_state(full) == _numerical_state(resumed)
    assert restored == partial
    assert full["density_direction_passes"] == 2 * full["evaluations"]
    assert full["evaluations"] == 1 + full["iterations"] + full["rejected_evaluations"]
    fidelities = [row["fidelity"] for row in full["curve"]]
    assert np.min(np.diff(fidelities)) >= -1e-13
    assert 1 - full["loss"] == pytest.approx(
        channels.evaluate(circuit, np.asarray(full["theta"]), target, "depolarizing_1s_all"),
        abs=2e-13,
    )
    assert full["metric_cpu_seconds"] >= 0
    assert full["solve_cpu_seconds"] >= 0
    assert full["total_cpu_seconds"] >= full["channel_cpu_seconds"]


@pytest.mark.parametrize(
    ("budget", "evaluations", "status"), [(8, 8, "evaluation_limit"), (40, 21, "line_search_failed")]
)
def test_every_rejected_nonfinite_trial_and_metric_are_charged(
    monkeypatch: pytest.MonkeyPatch, budget: int, evaluations: int, status: str
) -> None:
    """One reusable metric can drive multiple rejected true-objective calls."""
    circuit, theta, target = _problem()
    original = geometry.density_loss_gradient
    calls = 0

    def fail_trials(
        candidate: ParameterizedCircuit, values: NDArray, desired: NDArray, calibration: object
    ) -> tuple[NDArray, float, float]:
        nonlocal calls
        calls += 1
        if calls > 1:
            msg = "Injected nonfinite density"
            raise FloatingPointError(msg)
        return original(candidate, values, desired, cast("Any", calibration))

    monkeypatch.setattr(geometry, "density_loss_gradient", fail_trials)
    result = geometry.minimize(circuit, theta, target, damping=0.001, max_evaluations=budget)
    assert result["evaluations"] == evaluations
    assert result["rejected_evaluations"] == result["nonfinite_evaluations"] == evaluations - 1
    assert result["metric_evaluations"] == 1
    assert result["status"] == status
    if status == "evaluation_limit":
        assert result["pending"]["trials"] == 7
        assert result["pending"]["step"] == pytest.approx(0.5**7)
    else:
        assert result["pending"] is None
    assert result["density_direction_passes"] == 2 * evaluations
    assert all(row["loss"] is None for row in result["trials"])
    json.dumps(result, allow_nan=False)


def test_callback_interruption_checkpoint_resumes_full_numerical_path() -> None:
    """A durable trial-boundary callback needs no hidden simulator or metric cache state."""
    circuit, theta, target = _problem()
    retained: list[dict[str, Any]] = []

    def interrupt(state: dict[str, Any]) -> None:
        retained.append(json.loads(json.dumps(state, allow_nan=False)))
        if state["evaluations"] == 5:
            raise InterruptedError

    with pytest.raises(InterruptedError):
        geometry.minimize(circuit, theta, target, damping=0.01, max_evaluations=15, callback=interrupt)
    resumed = geometry.minimize(circuit, theta, target, damping=0.01, max_evaluations=15, state=retained[-1])
    full = geometry.minimize(circuit, theta, target, damping=0.01, max_evaluations=15)
    assert _numerical_state(resumed) == _numerical_state(full)
    incompatible = copy.deepcopy(retained[-1])
    incompatible["evaluations"] += 1
    with pytest.raises(ValueError, match="malformed or incompatible"):
        geometry.minimize(circuit, theta, target, damping=0.01, state=incompatible)
