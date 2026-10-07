# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent parity, derivative and register tests for twelve-qubit readiness."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import DensityMatrix, Statevector, random_statevector

from experiments.state_preparation_competitors.checkpointed_dense import CheckpointedDenseObjective, make_objective
from experiments.state_preparation_competitors.common import validate_target
from experiments.state_preparation_competitors.fontana import fontana_ansatz, real_target
from experiments.state_preparation_competitors.logical_evaluation import (
    logical_resources,
    sample_paths,
    validation_from_paths,
)
from experiments.state_preparation_competitors.noisy_core import DenseObjective
from experiments.state_preparation_competitors.robuststate import (
    classical_shadow_feedback,
    pauli_basis_probabilities,
    robuststate_candidates,
    topology_aware_ansatz,
    train_robuststate,
)


def _circuit(n: int = 3) -> QuantumCircuit:
    """Use shared, nonlinear coordinates and both pair orientations.

    Returns:
        A nontrivial symbolic circuit for chain-rule tests.
    """
    theta = ParameterVector("a", 3)
    circuit = QuantumCircuit(n)
    for _ in range(3):
        circuit.rx(theta[0] * theta[2], 0)
        circuit.ry(0.7 * theta[1] + 0.2, n - 1)
        circuit.cx(1, 0)
        circuit.rzz(theta[2], 0, 1)
    return circuit


@pytest.mark.parametrize("hardware", [None, "IBM", "Google"])
@pytest.mark.parametrize("native", [True, False])
def test_checkpoint_gradient_matches_full_tape_and_finite_difference(hardware: str | None, *, native: bool) -> None:
    """Recomputation changes neither the noise law nor chain-rule derivatives."""
    target = random_statevector(8, seed=8).data
    circuit = _circuit()
    full = DenseObjective(circuit, target, hardware, compile_native=native)
    bounded = CheckpointedDenseObjective(circuit, target, hardware, compile_native=native)
    theta = np.array([0.43, -0.7, 0.23])
    loss, gradient = bounded.loss_gradient(theta)
    expected_loss, expected_gradient = full.loss_gradient(theta)
    assert loss == pytest.approx(expected_loss, abs=2e-14)
    np.testing.assert_allclose(gradient, expected_gradient, atol=2e-14)
    delta = 1e-5
    fd = [
        (full.fidelity(theta - delta * axis) - full.fidelity(theta + delta * axis)) / (2 * delta) for axis in np.eye(3)
    ]
    np.testing.assert_allclose(gradient, fd, atol=1e-10)
    np.testing.assert_allclose(bounded.density(theta), full.density(theta), atol=2e-14)
    assert bounded.last_checkpoint_count <= math.ceil(math.log2(len(bounded.windows))) + 1


def test_statevector_vjp_matches_frozen_costate_density_derivative() -> None:
    """A complex, indefinite shadow costate still gives the exact ideal VJP."""
    target = random_statevector(8, seed=23).data
    full = DenseObjective(_circuit(), target, "IBM", compile_native=False)
    bounded = CheckpointedDenseObjective(_circuit(), target, "IBM", compile_native=False)
    rng = np.random.default_rng(2)
    costate = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    costate += costate.conj().T
    theta = [0.1, 0.3, -0.4]
    np.testing.assert_allclose(bounded.density_vjp(theta, costate), full.density_vjp(theta, costate), atol=2e-14)
    with pytest.raises(ValueError, match="Hermitian"):
        bounded.density_vjp(theta, costate + 1j)
    with pytest.raises(ValueError, match="Hermitian"):
        bounded.density_vjp(theta, np.eye(2))


def test_empty_circuit_and_zero_parameter_checkpoint_paths() -> None:
    """Empty/no-parameter inputs work without a fabricated gradient."""
    for fixed_gate in [False, True]:
        circuit = QuantumCircuit(2)
        if fixed_gate:
            circuit.x(0)
        target = np.array([1, 0, 0, 0])
        for hardware in [None, "IBM"]:
            objective = CheckpointedDenseObjective(circuit, target, hardware)
            loss, gradient = objective.loss_gradient([])
            expected = DenseObjective(circuit, target, hardware).loss_gradient([])[0]
            assert loss == pytest.approx(expected, abs=1e-14)
            assert gradient.size == 0


def test_twelve_qubit_ideal_gradient_without_density_tape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full twelve-qubit pure training has an independent statevector oracle."""

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("The large-register path must never allocate the legacy density tape")

    monkeypatch.setattr(DenseObjective, "_forward", forbidden)
    circuit = fontana_ansatz(12, 1)
    target = random_statevector(4096, seed=10).data
    objective = make_objective(circuit, target, None, compile_native=False)
    assert isinstance(objective, CheckpointedDenseObjective)
    theta = np.linspace(-0.4, 0.3, 12)
    loss, gradient = objective.loss_gradient(theta)
    direction = np.linspace(0.1, 1, 12)

    def loss_at(values: np.ndarray) -> float:
        state = Statevector.from_instruction(objective.bind(values)).data
        return 1 - float(abs(np.vdot(target, state)) ** 2)

    expected = (loss_at(theta + 1e-5 * direction) - loss_at(theta - 1e-5 * direction)) / 2e-5
    assert loss == pytest.approx(loss_at(theta), abs=2e-14)
    assert gradient @ direction == pytest.approx(expected, abs=1e-10)
    assert topology_aware_ansatz(12, 4).num_parameters == 264
    assert logical_resources(objective.bind(theta), target)["logical_two_qubit"] == 11
    assert isinstance(make_objective(QuantumCircuit(2), [1, 0, 0, 0]), DenseObjective)


def test_larger_register_is_explicit_and_fontana_stays_real() -> None:
    """New callers opt into twelve while legacy methods keep their old guard."""
    target = np.ones(4096) / 64
    with pytest.raises(ValueError, match="state vector"):
        validate_target(target)
    np.testing.assert_array_equal(validate_target(target, max_qubits=12), target)
    for invalid in (True, 0, 13):
        with pytest.raises(ValueError, match="max_qubits"):
            validate_target(target, max_qubits=invalid)
    np.testing.assert_allclose(real_target(target * 1j), target, atol=1e-15)
    with pytest.raises(ValueError, match="real target"):
        real_target(random_statevector(4096, seed=6).data)


def test_shadow_sampling_preserves_seed_and_validates_untrusted_inputs() -> None:
    """Skipping only the PSD eigensolve for known CPTP output preserves samples."""
    state = random_statevector(8, seed=9).data
    rho = np.outer(state, state.conj())
    checked = classical_shadow_feedback(rho, rng=np.random.default_rng(7), shots_per_basis=21, num_bases=7)
    trusted = classical_shadow_feedback(
        rho, rng=np.random.default_rng(7), trusted_cptp_output=True, shots_per_basis=21, num_bases=7
    )
    np.testing.assert_array_equal(checked, trusted)
    with pytest.raises(ValueError, match="positive"):
        classical_shadow_feedback(np.diag([2, -1]), rng=np.random.default_rng(7))


def test_seven_qubit_robuststate_full_candidate_path() -> None:
    """The larger wrapper reaches ideal, sampled-noisy and final circuit stages."""
    target = random_statevector(128, seed=7).data
    result = robuststate_candidates(
        target,
        depths=(1,),
        noiseless_steps=1,
        noisy_steps=1,
        feedback="pauli_shadows",
        num_bases=2,
        shots_per_basis=5,
        noise_basis="logical",
    )[0]
    assert result.circuit.num_qubits == 7
    assert result.metadata["training"]["total_shots"] == 10
    assert result.metadata["training"]["retention"] == "final_iterate_published"


def test_chunked_trajectory_evaluation_preserves_weighted_estimate() -> None:
    """Changing chunk size preserves every sample, jump and raw-sample error bar."""
    circuit = QuantumCircuit(12)
    circuit.h(0)
    circuit.cx(0, 1)
    target = Statevector.from_instruction(circuit).data
    codes = sample_paths(circuit, "IBM", 31, [300, 99])
    codes[3, 0] = 1
    codes[7, 1] = 2
    codes[9, 1] = 3
    first = validation_from_paths(circuit, target, "IBM", codes, batch_size=1)
    second = validation_from_paths(circuit, target, "IBM", codes, batch_size=31)
    for key in first:
        assert first[key] == pytest.approx(second[key]) if isinstance(first[key], float) else first[key] == second[key]
    assert first["jump_paths"] >= 3
    assert first["no_jump_paths"] > 0
    with pytest.raises(ValueError, match="batch_size"):
        validation_from_paths(circuit, target, "IBM", codes, batch_size=0)


def test_large_shadow_training_requires_explicit_measurement_budget() -> None:
    """A default must not accidentally enumerate 531,441 twelve-qubit bases."""
    target = np.zeros(4096)
    target[0] = 1
    engine = CheckpointedDenseObjective(QuantumCircuit(12), target)
    with pytest.raises(ValueError, match="explicit num_bases"):
        train_robuststate(engine, [], seed=3, feedback="pauli_shadows")
    with pytest.raises(TypeError, match="explicit boolean"):
        classical_shadow_feedback(np.eye(2) / 2, rng=np.random.default_rng(3), trusted_cptp_output=cast("bool", 1))


def test_shrinking_pauli_probabilities_match_independent_qiskit_rotations() -> None:
    """Pair contraction preserves complex mixed-state Born probabilities."""
    states = [random_statevector(16, seed=seed).data for seed in (2, 3)]
    density = sum(weight * np.outer(state, state.conj()) for weight, state in zip((0.3, 0.7), states, strict=True))
    for basis in [(0, 1, 2, 1), (2, 0, 1, 0), (1, 1, 1, 1)]:
        circuit = QuantumCircuit(4)
        for site, label in enumerate(reversed(basis)):
            if label == 1:
                circuit.sdg(site)
            if label != 2:
                circuit.h(site)
        expected = DensityMatrix(density).evolve(circuit).probabilities()
        np.testing.assert_allclose(pauli_basis_probabilities(density, basis), expected, atol=2e-15)
