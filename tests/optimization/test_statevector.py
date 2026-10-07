# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Dense trajectory gradients, online updates, sampling, and rejection contracts."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization import (
    CompositeGateNoiseInstruction,
    KrotovNoiseMap,
    KrotovTJMOptions,
    KrotovTruncation,
    LocalOperator,
    ParameterizedCircuit,
    ParameterizedGate,
    RandomUnitaryInstruction,
    StatevectorBackend,
    TJMNoiseInstruction,
    noisy_state_preparation_contribution,
)

# Numerical parity checks deliberately exercise the production gate-sweep kernel.
from mqt.yaqs.optimization.krotov import _noisy_state_preparation_online_update  # noqa: PLC2701

if TYPE_CHECKING:
    from mqt.yaqs.optimization.gate_noise import GateNoiseContext, GateNoiseInstruction
    from mqt.yaqs.optimization.statevector import Estimator


def case(qubits: int = 3) -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Return complex targets, a long-range gate, shared parameters, and affine angles."""
    circuit = ParameterizedCircuit(
        qubits,
        [
            ParameterizedGate("ry", (0,), param_index=0, angle_scale=-1.3, angle_offset=0.2),
            ParameterizedGate("rx", (qubits - 1,), param_index=1),
            ParameterizedGate("rxx", (qubits - 1, 0), param_index=2),
            ParameterizedGate("rz", (0,), param_index=0, angle_scale=0.7),
            ParameterizedGate("h", (1,), noise_enabled=False),
            ParameterizedGate("cx", (qubits - 1, 0)),
        ],
    )
    rng = np.random.default_rng(42)
    target = rng.normal(size=2**qubits) + 1j * rng.normal(size=2**qubits)
    return circuit, rng.normal(size=3), target / np.linalg.norm(target)


@pytest.mark.parametrize("qubits", [2, 3, 6])
@pytest.mark.parametrize("estimator", ["independent", "cross"])
def test_noisy_signals_and_online_sweeps_match_mps(qubits: int, estimator: Estimator) -> None:
    """Identical sampled maps reproduce MPS gradients, mean losses, and sequential updates."""
    circuit, theta, target = case(qubits)
    backend = StatevectorBackend(circuit, target)
    provider = ExactLocalPauliProvider(CalibratedLocalDepolarizingProvider(0.15, 0.3))
    options = KrotovTJMOptions(num_trajectories=4, random_seed=900, trajectory_update=estimator)
    maps = backend.sample_noise_maps(theta, options, provider, iteration=2)
    expected, loss, fidelity, trajectories = noisy_state_preparation_contribution(
        circuit,
        theta,
        target,
        None,
        options,
        MPS(qubits),
        KrotovTruncation(),
        iteration=2,
        noise_provider=provider,
    )
    for dense_row, trajectory in zip(maps, trajectories, strict=True):
        for dense, mps in zip(dense_row, trajectory.noise_maps, strict=True):
            assert dense.outcome_labels == mps.outcome_labels
            assert len(dense.operators) == len(mps.operators)
            for (a, sites_a), (b, sites_b) in zip(dense.operators, mps.operators, strict=True):
                np.testing.assert_array_equal(a, b)
                assert sites_a == sites_b
    actual, dense_loss, dense_fidelity = backend.gradient(theta, maps, estimator=estimator)
    np.testing.assert_allclose(actual, expected, atol=2e-12)
    assert dense_loss == pytest.approx(loss, abs=2e-12)
    assert dense_fidelity == pytest.approx(fidelity, abs=2e-12)
    replay = [list(row) for row in maps]
    expected_theta, expected_signal = _noisy_state_preparation_online_update(
        circuit,
        theta,
        MPS.from_statevector(target),
        None,
        options,
        0.7,
        None,
        KrotovTruncation(),
        2,
        fixed_noise_maps=replay,
    )
    updated, signal = backend.online_update(theta, 0.7, maps, estimator=estimator)
    np.testing.assert_allclose(updated, expected_theta, atol=2e-12)
    np.testing.assert_allclose(signal, expected_signal, atol=2e-12)


def test_fixed_map_gradient_finite_difference_and_ensemble_mean() -> None:
    """Average individual fidelities/gradients once, never average state amplitudes."""
    circuit, theta, target = case()
    backend = StatevectorBackend(circuit, target)
    provider = ExactLocalPauliProvider(CalibratedLocalDepolarizingProvider(0.2, 0.4))
    maps = backend.sample_noise_maps(theta, KrotovTJMOptions(num_trajectories=7), provider, iteration=3)
    gradient, loss, fidelity = backend.gradient(theta, maps)
    single = [backend.gradient(theta, [row]) for row in maps]
    np.testing.assert_allclose(gradient, np.mean([row[0] for row in single], axis=0), atol=1e-14)
    assert fidelity == pytest.approx(np.mean([row[2] for row in single]), abs=1e-14)
    assert loss + fidelity == 1
    for k in range(len(theta)):
        shift = np.eye(len(theta))[k] * 1e-6
        difference = (backend.metrics(theta + shift, maps)[0] - backend.metrics(theta - shift, maps)[0]) / 2e-6
        assert gradient[k] == pytest.approx(difference, abs=1e-9)


def test_noiseless_matches_density_and_qiskit_without_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dense execution stays dense and handles nonzero initial complex states."""
    circuit, theta, target = case(6)
    expected_gradient, _, expected_fidelity = density_loss_gradient(
        circuit,
        theta,
        target,
        CalibratedLocalDepolarizingProvider(0, 0),
    )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Dense execution constructed an MPS")

    monkeypatch.setattr(MPS, "__init__", forbidden)
    backend = StatevectorBackend(circuit, target)
    actual, _, fidelity = backend.gradient(theta)
    np.testing.assert_allclose(actual, expected_gradient, atol=1e-13)
    assert fidelity == pytest.approx(expected_fidelity, abs=1e-13)
    nonzero = StatevectorBackend(circuit, target, initial_state=target)
    state = Statevector(target)
    for gate in circuit.gates:
        matrix, sites = circuit.gate_matrix(gate, theta)
        state = state.evolve(Operator(matrix), qargs=list(reversed(sites)))
    assert nonzero.metrics(theta)[1] == pytest.approx(abs(np.vdot(target, state.data)) ** 2, abs=1e-13)
    np.testing.assert_allclose(backend.gradient(theta, estimator="cross")[0], actual, atol=1e-13)


def test_sampling_masks_replay_and_input_immutability() -> None:
    """Seed counters are stateless; retained CRN maps and input arrays are never changed."""
    circuit, theta, target = case()
    original_theta, original_target = theta.copy(), target.copy()
    backend = StatevectorBackend(circuit, target)
    calls = []

    def provider(context: GateNoiseContext, rng: np.random.Generator) -> RandomUnitaryInstruction:
        calls.append((context.gate_index, rng.random()))
        return RandomUnitaryInstruction((LocalOperator(np.array([[0, 1], [1, 0]]), (context.sites[0],)),))

    options = KrotovTJMOptions(num_trajectories=2, random_seed=50, apply_noise_to="two-qubit", noisy_gate_indices=(2,))
    maps = backend.sample_noise_maps(theta, options, provider, iteration=4)
    assert [index for index, _ in calls] == [2, 2]
    uniforms = [value for _, value in calls]
    calls.clear()
    backend.sample_noise_maps(theta, options, provider, iteration=4)
    assert [value for _, value in calls] == uniforms
    calls.clear()
    backend.sample_noise_maps(theta, options, provider, iteration=5)
    assert [value for _, value in calls] != uniforms
    before = backend.gradient(theta, maps)
    backend.online_update(theta, 0.7, maps)
    np.testing.assert_array_equal(before[0], backend.gradient(theta, maps)[0])
    np.testing.assert_array_equal(theta, original_theta)
    np.testing.assert_array_equal(target, original_target)
    target[:] = 0
    assert backend.metrics(theta, maps)[1] == before[2]


@pytest.mark.parametrize("composite", [False, True])
def test_legacy_pauli_tjm_distribution_and_seed_pairing(*, composite: bool) -> None:
    """Pauli one-jump and composite providers agree with production sampled maps."""
    circuit, theta, target = case()
    backend = StatevectorBackend(circuit, target)

    def provider(context: GateNoiseContext, rng: np.random.Generator) -> GateNoiseInstruction:
        del rng
        instruction = TJMNoiseInstruction(
            NoiseModel([
                {"name": "pauli_x", "sites": [context.sites[0]], "strength": 0.4},
                {"name": "pauli_z", "sites": [context.sites[0]], "strength": 0.2},
            ])
        )
        return CompositeGateNoiseInstruction((instruction, instruction)) if composite else instruction

    options = KrotovTJMOptions(num_trajectories=5, random_seed=111)
    maps = backend.sample_noise_maps(theta, options, provider, iteration=9)
    expected, loss, _, _ = noisy_state_preparation_contribution(
        circuit,
        theta,
        target,
        None,
        options,
        MPS(3),
        KrotovTruncation(),
        iteration=9,
        noise_provider=provider,
    )
    actual, dense_loss, _ = backend.gradient(theta, maps)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert dense_loss == pytest.approx(loss, abs=1e-12)


@pytest.mark.parametrize("bad", [np.zeros(8), np.ones(8), np.ones(4), np.full(8, np.nan)])
def test_invalid_vectors_rejected(bad: np.ndarray) -> None:
    """Invalid dimensions, normalization and nonfinite vectors fail before execution."""
    circuit, _, target = case()
    with pytest.raises(ValueError, match="normalized statevector"):
        StatevectorBackend(circuit, bad)
    with pytest.raises(ValueError, match="normalized statevector"):
        StatevectorBackend(circuit, target, bad)


@pytest.mark.parametrize("bad", [np.zeros(4), np.full(3, np.nan), np.ones(3, dtype=complex)])
def test_invalid_parameters_rejected(bad: np.ndarray) -> None:
    """Do not silently discard imaginary parameters or propagate NaNs."""
    circuit, _, target = case()
    with pytest.raises(ValueError, match="real parameters"):
        StatevectorBackend(circuit, target).gradient(bad)


def test_invalid_maps_estimators_and_counters_rejected() -> None:
    """Unsupported maps and invalid controls fail loudly rather than changing the model."""
    circuit, theta, target = case()
    backend = StatevectorBackend(circuit, target)
    empty = [KrotovNoiseMap()] * len(circuit.gates)
    for maps in ([], [[]], [empty[:-1]]):
        with pytest.raises(ValueError, match="one map"):
            backend.gradient(theta, maps)
    for changed in (KrotovNoiseMap(normalized=True), KrotovNoiseMap(normalization_checkpoints=(0,))):
        with pytest.raises(NotImplementedError, match="nonunitary replay"):
            backend.gradient(theta, [[changed, *empty[1:]]])
    with pytest.raises(ValueError, match="unitary"):
        backend.gradient(theta, [[KrotovNoiseMap(operators=((np.diag([1, 0.5]), (0,)),)), *empty[1:]]])
    with pytest.raises(ValueError, match="outside"):
        backend.gradient(theta, [[KrotovNoiseMap(operators=((np.eye(2, dtype=complex), (3,)),)), *empty[1:]]])
    with pytest.raises(ValueError, match="estimator"):
        backend.gradient(theta, estimator=cast("Estimator", "invalid"))
    for rate in (-1.0, np.nan, np.inf, True):
        with pytest.raises(ValueError, match="Step size"):
            backend.online_update(theta, rate)
    for options, iteration in ((KrotovTJMOptions(), -1), (KrotovTJMOptions(num_trajectories=cast("int", 1.5)), 0)):
        with pytest.raises(ValueError, match="integers"):
            backend.sample_noise_maps(theta, options, ExactLocalPauliProvider(), iteration=iteration)

    def damping(context: GateNoiseContext, rng: np.random.Generator) -> TJMNoiseInstruction:
        del rng
        return TJMNoiseInstruction(NoiseModel([{"name": "lowering", "sites": [context.sites[0]], "strength": 0.2}]))

    with pytest.raises(NotImplementedError, match="Pauli unitary"):
        backend.sample_noise_maps(theta, KrotovTJMOptions(), damping)
    dependent = ParameterizedCircuit(3, [replace(circuit.gates[0], data_map=lambda _x: 1.0)])
    with pytest.raises(ValueError, match="data-free"):
        StatevectorBackend(dependent, target)


def test_empty_circuit_and_zero_step() -> None:
    """Identity circuits and zero-rate diagnostics retain their mathematical meaning."""
    backend = StatevectorBackend(ParameterizedCircuit(1, []), np.array([1, 0], dtype=complex))
    gradient, loss, fidelity = backend.gradient(np.array([]))
    assert not len(gradient)
    assert (loss, fidelity) == (0, 1)
    circuit, theta, target = case()
    updated, _ = StatevectorBackend(circuit, target).online_update(theta, 0)
    np.testing.assert_array_equal(updated, theta)


@pytest.mark.parametrize("estimator", ["independent", "cross"])
def test_enumerated_noise_matches_exact_density_gradient(estimator: Estimator) -> None:
    """A complete joint ensemble agrees with the independent Lindblad adjoint."""
    circuit = ParameterizedCircuit(
        1,
        [
            ParameterizedGate("ry", (0,), param_index=0),
            ParameterizedGate("rz", (0,), param_index=1),
        ],
    )
    target = np.array([1, 1j]) / np.sqrt(2)
    theta = np.array([0.3, -0.8])
    # At r1=1/4 the exact Pauli mixture has weights (5,1,1,1)/8.
    maps = [KrotovNoiseMap()] * 5 + [
        KrotovNoiseMap(operators=((matrix, (0,)),))
        for matrix in (
            np.array([[0, 1], [1, 0]], dtype=complex),
            np.array([[0, -1j], [1j, 0]]),
            np.diag([1, -1]).astype(complex),
        )
    ]
    actual = StatevectorBackend(circuit, target).gradient(theta, list(product(maps, repeat=2)), estimator=estimator)
    expected = density_loss_gradient(circuit, theta, target, CalibratedLocalDepolarizingProvider(0.25, 0))
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-14)
    np.testing.assert_allclose(actual[1:], expected[1:], atol=1e-14)
