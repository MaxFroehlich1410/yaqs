# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent tests of RobustState's published loss, feedback and update policy."""

from __future__ import annotations

import itertools
import json
from typing import Any, cast

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import DensityMatrix, Operator, Statevector, random_statevector

from experiments.legacy_topdown_dense import noise_channel
from experiments.state_preparation_competitors import robuststate as study
from experiments.state_preparation_competitors.noisy_core import DenseObjective
from experiments.state_preparation_exact_adam import AdamConfig, adam_step


def test_ansatz_matches_cnot_first_blocks_in_paper_figure_ten() -> None:
    """Check gate order, pair traversal, axes, parameter independence and count."""
    circuit = study.topology_aware_ansatz(4, 2)
    assert circuit.num_parameters == 36
    assert circuit.count_ops() == {"cx": 6, "rz": 24, "rx": 12}
    for block, pair in enumerate([(0, 1), (1, 2), (2, 3)] * 2):
        rows = circuit.data[7 * block : 7 * (block + 1)]
        assert [row.operation.name for row in rows] == ["cx", "rz", "rx", "rz", "rz", "rx", "rz"]
        assert tuple(circuit.find_bit(q).index for q in rows[0].qubits) == pair
        assert [circuit.find_bit(row.qubits[0]).index for row in rows[1:]] == [pair[0]] * 3 + [pair[1]] * 3
    assert not study.topology_aware_ansatz(6, 0).data


@pytest.mark.parametrize(("qubits", "depth"), [(1, 1), (13, 1), (True, 1), (2.0, 1), (2, -1), (2, True), (2, 1.5)])
def test_ansatz_rejects_invalid_dimensions(qubits: int, depth: int) -> None:
    """Do not invent one-qubit blocks or round noninteger depths."""
    with pytest.raises(ValueError, match=r"qubits|depth"):
        study.topology_aware_ansatz(qubits, depth)


def test_loss_and_costate_use_hermitian_complex_matrices() -> None:
    """Complex-target fidelity and Frobenius loss need conjugated amplitudes."""
    target = np.array([1, 1j]) / np.sqrt(2)
    projector = np.outer(target, target.conj())
    prepared = np.array([[0.3, 0.1 - 0.2j], [0.1 + 0.2j, 0.7]])
    loss, costate = study.hilbert_schmidt_loss_and_costate(prepared, projector)
    assert loss == pytest.approx(np.sqrt(np.trace((prepared - projector) @ (prepared - projector)).real))
    np.testing.assert_allclose(costate, (prepared - projector) / loss, atol=1e-15)
    direction = np.array([[0.2, -0.1j], [0.1j, -0.2]])
    h = 1e-6
    finite_difference = (
        np.linalg.norm(prepared + h * direction - projector) - np.linalg.norm(prepared - h * direction - projector)
    ) / (2 * h)
    assert np.trace(costate @ direction).real == pytest.approx(finite_difference, abs=1e-9)
    zero_loss, zero_costate = study.hilbert_schmidt_loss_and_costate(projector, projector)
    assert zero_loss == 0
    np.testing.assert_array_equal(zero_costate, np.zeros((2, 2)))


@pytest.mark.parametrize(
    "bad", [np.zeros(4), np.zeros((2, 3)), np.eye(3), np.eye(129), [[1, 1j], [1j, 0]], [[np.nan, 0], [0, 0]]]
)
def test_loss_rejects_invalid_matrices(bad: np.ndarray) -> None:
    """Reject shapes and non-Hermitian or nonfinite inputs before differentiation."""
    with pytest.raises(ValueError, match="Hermitian"):
        study.hilbert_schmidt_loss_and_costate(bad, np.eye(2))


def test_loss_rejects_dimension_mismatch_but_accepts_nonpositive_shadows() -> None:
    """A raw shadow is not a physical density; rejecting it would bias feedback."""
    with pytest.raises(ValueError, match="same shape"):
        study.hilbert_schmidt_loss_and_costate(np.eye(2), np.eye(4))
    loss, _ = study.hilbert_schmidt_loss_and_costate(np.diag([2, -1]), np.diag([1, 0]))
    assert loss == pytest.approx(np.sqrt(2))


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_shadow_inverse_is_unbiased_for_complex_entangled_mixed_states(num_qubits: int) -> None:
    """Exactly integrate measurement outcomes and all bases without Monte Carlo."""
    state = random_statevector(2**num_qubits, seed=17).data
    rho = 0.7 * np.outer(state, state.conj()) + 0.3 * np.eye(state.size) / state.size
    expected = np.zeros_like(rho)
    for basis in itertools.product(range(3), repeat=num_qubits):
        probabilities = study.pauli_basis_probabilities(rho, basis)
        # An independent Kronecker change of basis catches little-endian errors.
        unitary = np.array([[1]], dtype=complex)
        bases = [np.array([[1, 1], [1, -1]]) / np.sqrt(2), np.array([[1, 1], [1j, -1j]]) / np.sqrt(2), np.eye(2)]
        for label in basis:
            unitary = np.kron(unitary, bases[label])
        np.testing.assert_allclose(probabilities, np.diag(unitary.conj().T @ rho @ unitary).real, atol=1e-14)
        expected += study.inverse_pauli_shadow(probabilities, basis)
    np.testing.assert_allclose(expected / 3**num_qubits, rho, atol=3e-15, rtol=0)


def test_sampled_shadows_are_seeded_finite_hermitian_and_trace_one() -> None:
    """Repeat the shot experiment and confirm agreement with the physical input."""
    state = random_statevector(4, seed=91).data
    density = np.outer(state, state.conj())
    before = density.copy()
    first = study.classical_shadow_feedback(density, rng=np.random.default_rng(7), shots_per_basis=4096)
    second = study.classical_shadow_feedback(density, rng=np.random.default_rng(7), shots_per_basis=4096)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(density, before)
    np.testing.assert_allclose(first, first.conj().T, atol=1e-14)
    assert np.trace(first) == pytest.approx(1)
    np.testing.assert_allclose(first, density, atol=0.035, rtol=0)
    raw = study.classical_shadow_feedback(np.diag([1, 0]), rng=np.random.default_rng(7), shots_per_basis=1)
    assert np.linalg.eigvalsh(raw).min() < 0


def test_uniform_subset_shadow_feedback_replays_without_conditioning_on_results() -> None:
    """Subset basis choice is independently drawn and leaves the raw estimator."""
    rho = np.eye(4, dtype=complex) / 4
    first = study.classical_shadow_feedback(rho, rng=np.random.default_rng(4), shots_per_basis=50, num_bases=3)
    second = study.classical_shadow_feedback(rho, rng=np.random.default_rng(4), shots_per_basis=50, num_bases=3)
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.trace(first) == pytest.approx(1)


@pytest.mark.parametrize("density", [np.diag([0.5, 0.6]), np.diag([1.1, -0.1])])
def test_shadow_sampler_rejects_nonphysical_input(density: np.ndarray) -> None:
    """The state sampled by Born's rule must be positive and normalized."""
    with pytest.raises(ValueError, match="trace-one"):
        study.classical_shadow_feedback(density, rng=np.random.default_rng(0))


@pytest.mark.parametrize(
    "options",
    [{"shots_per_basis": 0}, {"shots_per_basis": True}, {"num_bases": 0}, {"num_bases": 4}, {"num_bases": True}],
)
def test_shadow_sampler_rejects_invalid_measurement_counts(options: dict[str, Any]) -> None:
    """Invalid shots or basis budgets must not become silent defaults."""
    with pytest.raises(ValueError, match=r"shots_per_basis|num_bases"):
        study.classical_shadow_feedback(np.eye(2) / 2, rng=np.random.default_rng(0), **options)


def test_shadow_sampler_requires_explicit_rng() -> None:
    """Avoid an implicit global RNG that leaks across trials."""
    with pytest.raises(TypeError, match="Generator"):
        study.classical_shadow_feedback(np.eye(2) / 2, rng=cast("np.random.Generator", None))


@pytest.mark.parametrize("compile_native", [True, False])
def test_surrogate_gradient_matches_independent_frozen_feedback_finite_difference(*, compile_native: bool) -> None:
    """Differentiate noisy output plus the ideal-state displacement, not noise."""
    target = random_statevector(4, seed=72).data
    engine = DenseObjective(study.topology_aware_ansatz(2, 2), target, "Google", compile_native=compile_native)
    theta = np.random.default_rng(19).normal(size=len(engine.parameters))
    rho_noisy, rho_ideal = engine.density(theta), engine.ideal_density(theta)
    target_density = np.outer(target, target.conj())
    loss, gradient = study.robuststate_loss_gradient(engine, theta)
    assert loss == pytest.approx(np.linalg.norm(rho_noisy - target_density))
    h = 1e-6
    finite_difference = []
    true_noisy_gradient = []
    for index in range(theta.size):
        delta = np.zeros_like(theta)
        delta[index] = h
        plus = rho_noisy + engine.ideal_density(theta + delta) - rho_ideal
        minus = rho_noisy + engine.ideal_density(theta - delta) - rho_ideal
        finite_difference.append(
            (np.linalg.norm(plus - target_density) - np.linalg.norm(minus - target_density)) / (2 * h)
        )
        true_noisy_gradient.append(
            (
                np.linalg.norm(engine.density(theta + delta) - target_density)
                - np.linalg.norm(engine.density(theta - delta) - target_density)
            )
            / (2 * h)
        )
    np.testing.assert_allclose(gradient, finite_difference, atol=3e-9, rtol=0)
    assert np.linalg.norm(gradient - true_noisy_gradient) > 1e-5
    exact_loss, exact_gradient = study.robuststate_loss_gradient(engine, theta, noisy=False)
    assert exact_loss == pytest.approx(np.linalg.norm(rho_ideal - target_density))
    assert np.isfinite(exact_gradient).all()
    shadow = study.classical_shadow_feedback(rho_noisy, rng=np.random.default_rng(5), shots_per_basis=2)
    shadow_loss, shadow_gradient = study.robuststate_loss_gradient(engine, theta, feedback_density=shadow)
    assert shadow_loss == pytest.approx(np.linalg.norm(shadow - target_density))
    assert not np.allclose(shadow_gradient, gradient)


def test_training_preserves_adam_moments_and_step_count_at_phase_switch() -> None:
    """Independently execute the published optimizer recurrence across the switch."""
    engine = DenseObjective(study.topology_aware_ansatz(2, 1), random_statevector(4, seed=4).data)
    initial = np.linspace(-0.6, 0.7, len(engine.parameters))
    theta = initial.copy()
    first, second = np.zeros_like(theta), np.zeros_like(theta)
    config = AdamConfig(updates=5)
    for iteration in range(5):
        _, gradient = study.robuststate_loss_gradient(engine, theta, noisy=iteration >= 3)
        theta, first, second = adam_step(theta, gradient, first, second, iteration + 1, config)
    actual, receipt = study.train_robuststate(engine, initial, seed=29, noiseless_steps=3, noisy_steps=2)
    np.testing.assert_array_equal(actual, theta)
    np.testing.assert_array_equal(initial, np.linspace(-0.6, 0.7, initial.size))
    assert receipt["adam_state_at_phase_switch"] == "preserved"
    assert [record["phase"] for record in receipt["records"]] == ["noiseless"] * 3 + ["noisy"] * 3
    assert receipt["gradient_evaluations"] == 5
    assert receipt["total_shots"] == 0
    json.dumps(receipt, allow_nan=False)


def test_training_returns_final_iterate_even_when_a_step_worsens_loss() -> None:
    """Do not silently add our own best-iterate rollback to the competitor."""
    circuit = QuantumCircuit(1)
    circuit.rx(Parameter("a"), 0)
    engine = DenseObjective(circuit, [1, 0], hardware=None)
    theta, receipt = study.train_robuststate(engine, [0.1], seed=0, noiseless_steps=1, noisy_steps=0, learning_rate=2)
    assert receipt["records"][-1]["fidelity"] < receipt["records"][0]["fidelity"]
    assert receipt["best_step_diagnostic"] == 0
    np.testing.assert_array_equal(receipt["best_theta_diagnostic"], [0.1])
    np.testing.assert_array_equal(theta, receipt["final_theta"])
    assert receipt["retention"] == "final_iterate_published"


@pytest.mark.parametrize("budgets", [(0, 0), (0, 2), (2, 0)])
def test_zero_or_single_phase_budgets_are_explicit(budgets: tuple[int, int]) -> None:
    """Support small engineering probes without inventing skipped update counts."""
    engine = DenseObjective(study.topology_aware_ansatz(2, 0), [1, 0, 0, 0])
    theta, receipt = study.train_robuststate(engine, [], seed=0, noiseless_steps=budgets[0], noisy_steps=budgets[1])
    assert theta.size == 0
    assert len(receipt["records"]) == sum(budgets) + 1
    assert receipt["records"][-1]["fidelity"] == pytest.approx(1)
    json.dumps(receipt, allow_nan=False)


def test_shadow_training_repeats_seed_and_records_actual_shots() -> None:
    """Noiseless updates and final diagnostics must not draw tomography shots."""
    engine = DenseObjective(study.topology_aware_ansatz(2, 1), random_statevector(4, seed=7).data)
    initial = np.linspace(0.2, 1.1, len(engine.parameters))
    kwargs: dict[str, Any] = {
        "seed": 30,
        "noiseless_steps": 2,
        "noisy_steps": 3,
        "feedback": "pauli_shadows",
        "shots_per_basis": 20,
        "num_bases": 5,
    }
    theta, receipt = study.train_robuststate(engine, initial, **kwargs)
    repeated, second = study.train_robuststate(engine, initial, **kwargs)
    np.testing.assert_array_equal(theta, repeated)
    assert receipt["measurement_calls"] == 3
    assert receipt["total_shots"] == 300
    assert [row["feedback_sampled"] for row in receipt["records"]] == [False, False, True, True, True, False]
    assert [row["loss"] for row in receipt["records"]] == [row["loss"] for row in second["records"]]


@pytest.mark.parametrize(
    "options",
    [
        {"noiseless_steps": -1},
        {"noisy_steps": True},
        {"seed": -2},
        {"learning_rate": 0},
        {"learning_rate": float("nan")},
        {"feedback": "hardware"},
        {"shots_per_basis": 0},
        {"num_bases": 0},
        {"num_bases": 10},
    ],
)
def test_training_rejects_bad_options_before_work(options: dict[str, Any]) -> None:
    """Reject invalid configurations even for zero-length engineering runs."""
    engine = DenseObjective(study.topology_aware_ansatz(2, 0), [1, 0, 0, 0])
    invalid: dict[str, Any] = {"seed": 0, "noiseless_steps": 0, "noisy_steps": 0} | options
    with pytest.raises(ValueError, match=r"steps|seed|learning_rate|feedback|shots_per_basis|num_bases"):
        study.train_robuststate(engine, [], **invalid)


@pytest.mark.parametrize("noise_basis", ["native", "logical"])
def test_real_six_qubit_candidates_preserve_schedule_and_complex_fidelity(noise_basis: str) -> None:
    """Exercise genuine 64-dimensional training, conversion and reproducibility."""
    target = random_statevector(64, seed=72).data
    before = target.copy()
    kwargs: dict[str, Any] = {
        "hardware": "Google",
        "seed": 31,
        "depths": (1,),
        "noiseless_steps": 1,
        "noisy_steps": 1,
        "noise_basis": noise_basis,
    }
    first = study.robuststate_candidates(target, **kwargs)[0]
    second = study.robuststate_candidates(target, **kwargs)[0]
    np.testing.assert_array_equal(target, before)
    np.testing.assert_array_equal(Operator(first.circuit).data, Operator(second.circuit).data)
    assert first.metadata["method"] == "robuststate"
    assert first.metadata["status"] == "complete"
    assert first.metadata["preserve_native_schedule"] is (noise_basis == "native")
    assert not first.metadata["hardware_execution"]
    logical = study.topology_aware_ansatz(6, 1)
    if noise_basis == "logical":
        assert first.metadata["noise_basis"] == "logical"
        assert "after each original" in first.metadata["training_noise"]
        assert [item.operation.name for item in first.circuit.data] == [item.operation.name for item in logical.data]
        assert len(first.circuit.data) == 35
        assert set(first.circuit.count_ops()) == {"rz", "rx", "cx"}
    else:
        assert len(first.circuit.data) > len(logical.data)
        assert set(first.circuit.count_ops()) <= {"rz", "sx", "x", "cx"}
    state = Statevector.from_instruction(first.circuit).data
    assert first.metadata["ideal_fidelity"] == pytest.approx(abs(np.vdot(target, state)) ** 2)
    exact = DensityMatrix.from_label("000000")
    for item in first.circuit.data:
        sites = [first.circuit.find_bit(qubit).index for qubit in item.qubits]
        exact = exact.evolve(item.operation, qargs=sites).evolve(noise_channel("Google", len(sites)), qargs=sites)
    assert exact.is_valid()
    assert first.metadata["noisy_fidelity"] == pytest.approx(float(np.vdot(target, exact.data @ target).real))
    assert [len(row["theta"]) for row in first.metadata["training"]["records"]] == [30, 30, 30]
    json.dumps(first.metadata, allow_nan=False)


def test_native_default_matches_explicit_native_training() -> None:
    """The optional logical schedule leaves default native Adam updates unchanged."""
    config: dict[str, Any] = {"hardware": "IBM", "seed": 300, "depths": (1,), "noiseless_steps": 1, "noisy_steps": 1}
    default = study.robuststate_candidates([1, 0, 0, 0], **config)[0]
    explicit = study.robuststate_candidates([1, 0, 0, 0], noise_basis="native", **config)[0]
    assert default.circuit == explicit.circuit
    assert default.metadata["training"]["final_theta"] == explicit.metadata["training"]["final_theta"]
    assert default.metadata["noisy_fidelity"] == explicit.metadata["noisy_fidelity"]
    assert default.metadata["preserve_native_schedule"] is True
    assert "noise_basis" not in default.metadata
    assert "training_noise" not in default.metadata


@pytest.mark.parametrize("noise_basis", ["compiled", "LOGICAL", ""])
def test_candidate_rejects_unknown_noise_basis(noise_basis: str) -> None:
    """Reject unsupported noise placement before allocating a training objective."""
    with pytest.raises(ValueError, match="noise_basis"):
        study.robuststate_candidates([1, 0, 0, 0], noise_basis=noise_basis)


def test_grid_order_does_not_change_per_depth_initialization_or_output() -> None:
    """Depth seeds are keyed by depth, independent of enumeration order."""
    target = random_statevector(4, seed=12).data
    normal = study.robuststate_candidates(target, depths=(0, 1), noiseless_steps=0, noisy_steps=0)
    reverse = study.robuststate_candidates(target, depths=(1, 0), noiseless_steps=0, noisy_steps=0)
    for first, second in zip(normal, reversed(reverse), strict=True):
        assert first.metadata["candidate_id"] == second.metadata["candidate_id"]
        assert first.metadata["training"]["initial_theta"] == second.metadata["training"]["initial_theta"]
        np.testing.assert_array_equal(Operator(first.circuit).data, Operator(second.circuit).data)


@pytest.mark.parametrize(
    "options",
    [{"depths": ()}, {"depths": (1, 1)}, {"depths": (True,)}, {"depths": (-1,)}, {"hardware": "ibm"}, {"seed": True}],
)
def test_candidates_reject_invalid_grids_and_profiles(options: dict[str, Any]) -> None:
    """Invalid candidate configuration must not be interpreted as another run."""
    with pytest.raises(ValueError, match=r"depths|hardware|seed"):
        study.robuststate_candidates([1, 0, 0, 0], noiseless_steps=0, noisy_steps=0, **options)


def test_candidates_reject_invalid_or_single_qubit_target() -> None:
    """Target normalization and block-supported register size remain explicit."""
    for target in ([1, 1, 0, 0], [1, 0]):
        with pytest.raises(ValueError, match=r"normalized|qubits"):
            study.robuststate_candidates(target, noiseless_steps=0, noisy_steps=0)
