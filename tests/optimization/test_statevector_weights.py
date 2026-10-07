# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Probability-weighted statevector gradients and truly joint online sweeps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.weighted_sampling import FactorSampler
from experiments.state_preparation_weighted_sampling import weighted_gradient
from mqt.yaqs.optimization import KrotovNoiseMap, ParameterizedCircuit, ParameterizedGate, StatevectorBackend

if TYPE_CHECKING:
    from mqt.yaqs.optimization.statevector import NoiseMaps


def example() -> tuple[StatevectorBackend, np.ndarray, NoiseMaps]:
    """Return a complex two-qubit problem with affine angles and a shared parameter."""
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
    return engine, np.array([0.4, -0.7]), sampler.ensemble(3, 543)[0]


def test_unequal_weights_gradient_finite_difference_and_online_replication() -> None:
    """Rational weights equal explicit path replication, including each online gate update."""
    engine, theta, maps = example()
    weights = np.array([0.5, 0.375, 0.125])
    repeated = tuple(row for row, count in zip(maps, (4, 3, 1), strict=True) for _ in range(count))
    gradient, loss, fidelity = engine.gradient(theta, maps, weights=weights)
    reference, ref_loss, ref_fidelity = engine.gradient(theta, repeated)
    np.testing.assert_allclose(gradient, reference, atol=1e-14)
    assert (loss, fidelity) == pytest.approx((ref_loss, ref_fidelity), abs=1e-14)
    for direction in np.eye(len(theta)) * 1e-6:
        plus = engine.metrics(theta + direction, maps, weights=weights)[0]
        minus = engine.metrics(theta - direction, maps, weights=weights)[0]
        assert gradient @ direction / 1e-6 == pytest.approx((plus - minus) / 2e-6, abs=2e-9)
    updated, signal = engine.online_update(theta, 0.7, maps, weights=weights)
    expected, ref_signal = engine.online_update(theta, 0.7, repeated)
    np.testing.assert_allclose(updated, expected, atol=1e-14)
    np.testing.assert_allclose(signal, ref_signal, atol=1e-14)
    separately_updated = np.sum(
        [weight * engine.online_update(theta, 0.7, [row])[0] for weight, row in zip(weights, maps, strict=True)], axis=0
    )
    assert np.linalg.norm(updated - separately_updated) > 1e-5


def test_uniform_zero_and_one_path_weights() -> None:
    """Explicit equal weights reproduce default signals; zero weights remove paths."""
    engine, theta, maps = example()
    for weights, expected in [(np.ones(3) / 3, maps), (np.array([1, 0, 0]), [maps[0]])]:
        np.testing.assert_allclose(
            engine.gradient(theta, maps, weights=weights)[0], engine.gradient(theta, expected)[0], atol=1e-14
        )
        np.testing.assert_allclose(
            engine.online_update(theta, 0.7, maps, weights=weights)[0],
            engine.online_update(theta, 0.7, expected)[0],
            atol=1e-14,
        )
    np.testing.assert_array_equal(engine.gradient(theta, weights=np.ones(1))[0], engine.gradient(theta)[0])
    np.testing.assert_array_equal(
        engine.online_update(theta, 0.7, weights=np.ones(1))[0], engine.online_update(theta, 0.7)[0]
    )


@pytest.mark.parametrize(
    "weights",
    [
        [],
        [1],
        [[1, 0, 0]],
        [0, 0, 0],
        [-1, 1, 1],
        [0.1, 0.1, 0.1],
        [np.nan, 0, 1],
        [np.inf, 0, 1],
        [1j, 0, 0],
        [True, False, False],
    ],
)
def test_invalid_probability_weights(weights: list[Any]) -> None:
    """Bad weights fail before either gradient or online propagation."""
    engine, theta, maps = example()
    for function in (engine.gradient, engine.metrics):
        with pytest.raises(ValueError, match="weights"):
            function(theta, maps, weights=np.asarray(weights))
    with pytest.raises(ValueError, match="weights"):
        engine.online_update(theta, 0.7, maps, weights=np.asarray(weights))


def test_weighted_cross_matches_replication_in_batch_and_online_updates() -> None:
    """Both cross indices receive multiplicities, including through sequential updates."""
    engine, theta, maps = example()
    weights = np.array([0.9, 0.07, 0.03])
    repeated = tuple(row for row, count in zip(maps, (90, 7, 3), strict=True) for _ in range(count))
    compressed, empirical = engine.compress_no_jump_maps(repeated)
    assert len(compressed) == 11
    assert empirical[0] == pytest.approx(0.9)
    original, reduced = theta.copy(), theta.copy()
    for iteration in range(6):
        full_signal, full_loss, full_fidelity = engine.gradient(original, repeated, estimator="cross")
        short_signal, short_loss, short_fidelity = engine.gradient(
            reduced, compressed, estimator="cross", weights=empirical
        )
        np.testing.assert_allclose(full_signal, short_signal, atol=2e-13)
        assert (full_loss, full_fidelity) == pytest.approx((short_loss, short_fidelity), abs=2e-13)
        direct = engine.gradient(original, maps, estimator="cross", weights=weights)[0]
        np.testing.assert_allclose(full_signal, direct, atol=2e-13)
        if iteration < 3:
            original, _ = engine.online_update(original, 0.2, repeated, estimator="cross")
            reduced, _ = engine.online_update(reduced, 0.2, compressed, estimator="cross", weights=empirical)
        else:
            original -= 0.2 * full_signal
            reduced -= 0.2 * short_signal
        np.testing.assert_allclose(original, reduced, atol=2e-13)


def test_compression_edge_cases_and_input_immutability() -> None:
    """Empty histories merge, whereas identity operators and cancelling jumps do not."""
    engine, theta, maps = example()
    empty = maps[0]
    identity = list(empty)
    identity[0] = KrotovNoiseMap(operators=((np.eye(2, dtype=complex), (0,)),))
    source = (tuple(identity), empty, empty, tuple(identity))
    retained, weights = engine.compress_no_jump_maps(source)
    assert len(source) == 4
    assert len(retained) == 3
    np.testing.assert_array_equal(weights, [0.25, 0.5, 0.25])
    assert retained[0][0].operators
    assert not retained[1][0].operators
    for histories in ((empty,) * 100, (tuple(identity),) * 3):
        reduced, mass = engine.compress_no_jump_maps(histories)
        np.testing.assert_allclose(engine.metrics(theta, histories), engine.metrics(theta, reduced, weights=mass))
    single, mass = engine.compress_no_jump_maps((empty,) * 100)
    assert len(single) == 1
    np.testing.assert_array_equal(mass, [1])
    with pytest.raises(ValueError, match="nonempty ensemble"):
        engine.compress_no_jump_maps(())
    bad = list(empty)
    bad[0] = KrotovNoiseMap(normalized=True)
    with pytest.raises(NotImplementedError, match=r"normalization|Normalized"):
        engine.compress_no_jump_maps((bad,))


def test_joint_stratified_kernel_matches_diagnostic_and_nested_paths() -> None:
    """The same stratum weights used in the diagnostic now feed optimizer kernels."""
    engine, theta, _ = example()
    sampler = FactorSampler(engine.circuit, CalibratedLocalDepolarizingProvider())
    maps, weights = sampler.ensemble(32, 43)
    expected, loss = weighted_gradient(engine, theta, sampler, maps[1:])
    actual, actual_loss, _ = engine.gradient(theta, maps, weights=weights)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert actual_loss == pytest.approx(loss, abs=1e-14)
    assert weights[0] == sampler.no_factor_probability
    np.testing.assert_array_equal(weights[1:], np.full(31, sampler.event_probability / 31))
    larger, _ = sampler.ensemble(128, 43)
    for small, large in zip(maps, larger[:32], strict=True):
        for a, b in zip(small, large, strict=True):
            assert len(a.operators) == len(b.operators)
            for (x, sites_x), (y, sites_y) in zip(a.operators, b.operators, strict=True):
                assert sites_x == sites_y
                np.testing.assert_array_equal(x, y)
    zero = FactorSampler(engine.circuit, CalibratedLocalDepolarizingProvider(0, 0))
    zero_maps, zero_weights = zero.ensemble(4, 0)
    np.testing.assert_allclose(
        engine.gradient(theta, zero_maps, weights=zero_weights)[0], engine.gradient(theta)[0], atol=1e-14
    )
    for count, seed in [(1, 0), (True, 0), (3, -1), (3, True)]:
        with pytest.raises(ValueError, match="two total paths"):
            sampler.ensemble(count, seed)
