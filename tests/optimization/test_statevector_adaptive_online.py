# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent dense checks for weighted per-gate adaptive online updates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.weighted_sampling import FactorSampler
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate, StatevectorBackend

if TYPE_CHECKING:
    from mqt.yaqs.optimization.statevector import NoiseMaps


def example() -> tuple[StatevectorBackend, np.ndarray, NoiseMaps, np.ndarray]:
    """Return a weighted complex-target example with a shared affine parameter."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), param_index=0, angle_scale=-1.3, angle_offset=0.2),
            ParameterizedGate("rxx", (1, 0), param_index=1),
            ParameterizedGate("rz", (0,), param_index=0, angle_scale=0.7),
            ParameterizedGate("h", (1,), noise_enabled=False),
        ],
    )
    engine = StatevectorBackend(circuit, np.array([1, 2j, -3, 4j]) / np.sqrt(30))
    sampler = FactorSampler(circuit, CalibratedLocalDepolarizingProvider(0.1, 0.2))
    maps, weights = sampler.ensemble(4, 543)
    return engine, np.array([0.4, -0.7]), maps, weights


def dense_reference(
    engine: StatevectorBackend,
    theta: np.ndarray,
    maps: NoiseMaps,
    weights: np.ndarray,
    second: np.ndarray,
    counts: np.ndarray,
    beta2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reimplement sequential adjoints using full Qiskit-embedded matrices.

    Returns:
        Parameters, raw signals, moments and counters after one RMS-scaled sweep.
    """
    circuit = engine.circuit
    dimension = len(engine.target)

    def embed(matrix: np.ndarray, sites: tuple[int, ...]) -> np.ndarray:
        return Operator(np.eye(dimension)).compose(Operator(matrix), qargs=list(reversed(sites))).data

    unitary = [embed(*circuit.gate_matrix(gate, theta)) for gate in circuit.gates]
    noises = []
    costates = []
    for r, row in enumerate(maps):
        noise_row = []
        state = engine.initial.copy()
        for u, noise in zip(unitary, row, strict=True):
            n = np.eye(dimension, dtype=complex)
            for matrix, sites in noise.operators:
                n = embed(matrix, sites) @ n
            noise_row.append(n)
            state = n @ u @ state
        chi = -weights[r] * np.vdot(engine.target, state) * engine.target
        backwards = []
        for u, n in zip(reversed(unitary), reversed(noise_row), strict=True):
            chi = n.conj().T @ chi
            backwards.append(chi)
            chi = u.conj().T @ chi
        noises.append(noise_row)
        costates.append(list(reversed(backwards)))
    updated, moment, counter = theta.copy(), second.copy(), counts.copy()
    current = [engine.initial.copy() for _ in maps]
    signals = np.zeros_like(theta)
    for k, gate in enumerate(circuit.gates):
        u = embed(*circuit.gate_matrix(gate, updated))
        outputs = [u @ state for state in current]
        if gate.param_index is not None:
            index = gate.param_index
            derivative = embed(*circuit.derivative_operator(gate))
            local = (
                2
                * gate.angle_scale
                * sum(float(np.vdot(costates[r][k], derivative @ out).real) for r, out in enumerate(outputs))
            )
            signals[index] += local
            moment[index] = beta2 * moment[index] + (1 - beta2) * local**2
            counter[index] += 1
            updated[index] -= 0.02 * local / (np.sqrt(moment[index] / (1 - beta2 ** counter[index])) + 1e-8)
            u = embed(*circuit.gate_matrix(gate, updated))
            outputs = [u @ state for state in current]
        current = [noises[r][k] @ state for r, state in enumerate(outputs)]
    return updated, signals, moment, counter


@pytest.mark.parametrize("beta2", [0.0, 0.9, 0.999])
def test_adaptive_sweep_matches_independent_dense_reference(beta2: float) -> None:
    """Match each updated prefix, weighted signal, affine angle and shared-parameter counter."""
    engine, theta, maps, weights = example()
    second, counts = np.array([0.03, 0.001]), np.array([7, 3], dtype=np.int64)
    inputs = [x.copy() for x in (theta, weights, second, counts)]
    actual = engine.adaptive_online_update(theta, 0.02, second, counts, maps, weights=weights, beta2=beta2)
    expected = dense_reference(engine, theta, maps, weights, second, counts, beta2)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, atol=2e-14, rtol=0)
    for a, b in zip(inputs, (theta, weights, second, counts), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(actual[-1], [9, 4])
    # Scaling a full batch gradient is a different operation from sequential updates.
    gradient = engine.gradient(theta, maps, weights=weights)[0]
    batch_second = beta2 * second + (1 - beta2) * gradient**2
    batch = theta - 0.02 * gradient / (np.sqrt(batch_second / (1 - beta2 ** (counts + 1))) + 1e-8)
    assert np.linalg.norm(actual[0] - batch) > 1e-6


def test_one_gate_closed_form_and_zero_signal() -> None:
    """The first occurrence uses bias correction, with finite behavior at zero signal."""
    engine = StatevectorBackend(
        ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)]), np.array([1, 0j])
    )
    theta = np.array([0.4])
    gradient = engine.gradient(theta)[0]
    updated, _, moment, counts = engine.adaptive_online_update(theta, 0.02, np.zeros(1), np.zeros(1, dtype=np.int64))
    np.testing.assert_allclose(updated, theta - 0.02 * gradient / (np.abs(gradient) + 1e-8), atol=1e-14)
    np.testing.assert_allclose(moment, 0.001 * gradient**2, atol=1e-16)
    np.testing.assert_array_equal(counts, [1])
    zero = engine.adaptive_online_update(np.zeros(1), 0.02, np.zeros(1), np.zeros(1, dtype=np.int64))
    for value in zero[:3]:
        np.testing.assert_array_equal(value, [0])


def test_weighted_replication_and_zero_rate() -> None:
    """Moment accumulation uses the weighted total signal, not per-path moments."""
    engine, theta, maps, _ = example()
    weights = np.array([0.5, 0.25, 0.125, 0.125])
    repeated = tuple(row for row, n in zip(maps, (4, 2, 1, 1), strict=True) for _ in range(n))
    moment, counts = np.zeros_like(theta), np.zeros(len(theta), dtype=np.int64)
    a = engine.adaptive_online_update(theta, 0.02, moment, counts, maps, weights=weights)
    b = engine.adaptive_online_update(theta, 0.02, moment, counts, repeated)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_allclose(x, y, atol=1e-14)
    zero_rate = engine.adaptive_online_update(theta, 0, moment, counts, maps, weights=weights)
    np.testing.assert_array_equal(zero_rate[0], theta)
    np.testing.assert_allclose(zero_rate[1], engine.gradient(theta, maps, weights=weights)[0], atol=1e-14)


@pytest.mark.parametrize(
    "options",
    [
        {"step_size": -0.1},
        {"step_size": True},
        {"beta2": 1},
        {"beta2": -1},
        {"beta2": np.nan},
        {"epsilon": 0},
        {"epsilon": np.inf},
        {"second_moment": np.array([-1.0, 0.0])},
        {"second_moment": np.array([np.nan, 0.0])},
        {"second_moment": np.zeros(3)},
        {"counts": np.array([1.0, 0.0])},
        {"counts": np.array([-1, 0])},
        {"counts": np.array([True, False])},
        {"counts": np.zeros(3, dtype=int)},
        {"counts": np.full(2, np.iinfo(np.int64).max)},
    ],
)
def test_invalid_adaptive_state(options: dict[str, Any]) -> None:
    """Malformed state/rates cannot silently corrupt online moments or counters."""
    engine, theta, maps, weights = example()
    kwargs: dict[str, Any] = {"step_size": 0.02, "second_moment": np.zeros(2), "counts": np.zeros(2, dtype=np.int64)}
    kwargs.update(options)
    with pytest.raises(ValueError, match=r"adaptive|parameters"):
        engine.adaptive_online_update(theta, maps=maps, weights=weights, **kwargs)
