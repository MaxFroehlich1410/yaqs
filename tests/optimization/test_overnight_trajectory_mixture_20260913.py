# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact finite-outcome checks for the equal-channel trajectory mixture."""

from __future__ import annotations

import copy
from itertools import product
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from experiments.overnight_state_preparation_20260913_0108 import channels
from experiments.overnight_state_preparation_20260913_0108 import trajectory_mixture as mixture
from experiments.overnight_state_preparation_20260913_0108.ensemble_objective import mean_loss_gradient
from experiments.state_preparation_pilot import circuit_record, digest
from mqt.yaqs.optimization import (
    KrotovNoiseMap,
    ParameterizedCircuit,
    ParameterizedGate,
    StatevectorBackend,
    statevector,
)

# Instrument the existing primitive to verify all declared row-operation costs.
from mqt.yaqs.optimization.statevector import _apply as apply_local  # noqa: PLC2701

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from benchmarks.state_preparation.weighted_sampling import FactorSampler
    from mqt.yaqs.optimization.statevector import NoiseMaps


def _problem(qubits: int, *, disabled: bool = False) -> tuple[StatevectorBackend, NDArray[np.float64]]:
    """Use a single noise window with reversed support and shared signed parameters.

    Returns:
        A q1-q3 backend and parameters, including one unused parameter.
    """
    gate = ParameterizedGate(
        "ry" if qubits == 1 else "rxx",
        (0,) if qubits == 1 else (qubits - 1, 0),
        param_index=0,
        angle_scale=-1.3,
        angle_offset=0.2,
        logical_gate_id="logical",
        native_gate_id="native",
        noise_enabled=not disabled,
    )
    circuit = ParameterizedCircuit(
        qubits,
        [
            ParameterizedGate("h", (0,), noise_enabled=False),
            gate,
            ParameterizedGate("rz", (0,), param_index=0, angle_scale=0.7, noise_enabled=False),
            ParameterizedGate("p", (0,), param_index=1, angle_scale=0.0, angle_offset=0.3, noise_enabled=False),
        ],
        num_params=3,
    )
    values = np.arange(1, 2**qubits + 1) + 1j * np.arange(2**qubits, 0, -1)
    return StatevectorBackend(circuit, values / np.linalg.norm(values)), np.array([0.4, -0.7, 0.9])


def _enumerate(sampler: FactorSampler) -> tuple[NoiseMaps, NDArray[np.float64]]:
    """Enumerate every independent Bernoulli factor pattern including empty.

    Returns:
        Replay maps and unconditioned exact product probabilities.
    """
    masks = np.asarray(list(product([False, True], repeat=len(sampler.factors))), dtype=bool)
    zero = (KrotovNoiseMap(),) * sampler.gate_count
    maps = (zero, *sampler.maps(masks[1:]))
    probabilities = np.prod(np.where(masks, sampler.probabilities, 1 - sampler.probabilities), axis=1)
    return maps, probabilities


def _map_key(maps: NoiseMaps) -> tuple:
    """Compare all replayed operator values, sites, order and source gate indices.

    Returns:
        An immutable replay identity with exact operator bytes.
    """
    return tuple(
        tuple(
            (gate.source_gate_index, tuple((matrix.tobytes(), sites) for matrix, sites in gate.operators))
            for gate in row
        )
        for row in maps
    )


@pytest.mark.parametrize("qubits", [1, 2, 3])
def test_exhaustive_mixture_matches_exact_channel_gradient(qubits: int) -> None:
    """All 8+2 or 64+4 paths reproduce the exact mean-two derivative and loss."""
    engine, theta = _problem(qubits)
    sampler = mixture.MixtureSampler(engine.circuit)
    depolarizing, dephasing = [_enumerate(value) for value in sampler.samplers]
    maps = (*depolarizing[0], *dephasing[0])
    weights = np.concatenate([depolarizing[1] / 2, dephasing[1] / 2])
    actual, loss, fidelity = engine.gradient(theta, maps, weights=weights)
    expected, expected_loss, per_channel = mean_loss_gradient(engine.circuit, theta, engine.target, mixture.NOISE_IDS)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-14)
    assert loss == pytest.approx(expected_loss, abs=3e-14)
    assert fidelity == pytest.approx(np.mean(list(per_channel.values())), abs=3e-14)
    assert weights.sum() == pytest.approx(1, abs=2e-15)
    np.testing.assert_array_equal(actual[1:], [0, 0])
    for factor_sampler, (channel_maps, channel_weights), noise_id in zip(
        sampler.samplers, [depolarizing, dephasing], mixture.NOISE_IDS, strict=True
    ):
        assert len(factor_sampler.factors) == (3 if noise_id.startswith("depolarizing") else 1) * min(2, qubits)
        assert engine.metrics(theta, channel_maps, weights=channel_weights)[1] == pytest.approx(
            channels.evaluate(engine.circuit, theta, engine.target, noise_id), abs=3e-14
        )


def test_independent_dephasing_calibration_and_two_site_order() -> None:
    """Closed-form average errors determine distinct local-Z probabilities."""
    circuit = ParameterizedCircuit(
        2,
        [ParameterizedGate("ry", (0,), param_index=0), ParameterizedGate("rzz", (1, 0), param_index=0)],
    )
    sampler = mixture.DephasingFactorSampler(circuit)
    expected_one = 1.5 * 0.00064
    expected_two = 1 - np.sqrt(1 - 1.25 * 0.0051)
    np.testing.assert_allclose(sampler.probabilities, [expected_one, expected_two, expected_two], atol=2e-15, rtol=0)
    assert sampler.factors == [(0, 0, "Z"), (1, 1, "Z"), (1, 0, "Z")]
    assert 2 / 3 * expected_one == pytest.approx(0.00064, abs=1e-16)
    assert 4 / 5 * (1 - (1 - expected_two) ** 2) == pytest.approx(0.0051, abs=2e-16)
    depolarizing = mixture.MixtureSampler(circuit).samplers[0]
    assert not np.isclose(depolarizing.probabilities[0], expected_one)
    assert not np.isclose(depolarizing.probabilities[-1], expected_two)


def test_conditional_factor_law_and_draw_for_every_nonempty_pattern(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first-event inversion and unmodified tail give the exact conditional law."""

    class FixedGenerator:
        """Supply inverse-CDF inputs selecting one prescribed pattern."""

        def __init__(self, values: list[float | NDArray]) -> None:
            """Retain prescribed first-factor and tail uniform inputs."""
            self.draws = values.copy()

        def random(self, size: int | None = None) -> float | NDArray:
            """Return a valid uniform input for the first or tail factors."""
            result = self.draws.pop(0)
            if size is not None:
                assert np.asarray(result).shape == (size,)
            return result

    circuit = ParameterizedCircuit(2, [ParameterizedGate("h", (0,)), ParameterizedGate("rxx", (1, 0), param_index=0)])
    sampler = mixture.DephasingFactorSampler(circuit)
    first_masses = np.diff(np.r_[0, sampler.first_cdf])
    total = 0.0
    for pattern in product([False, True], repeat=3):
        mask = np.asarray(pattern)
        if not mask.any():
            continue
        first = int(np.flatnonzero(mask)[0])
        tail = np.where(mask[first + 1 :], sampler.probabilities[first + 1 :], 1 - sampler.probabilities[first + 1 :])
        conditional = first_masses[first] * float(np.prod(tail))
        exact = (
            float(np.prod(np.where(mask, sampler.probabilities, 1 - sampler.probabilities))) / sampler.event_probability
        )
        assert conditional == pytest.approx(exact, abs=3e-15)
        total += conditional
        lower = 0.0 if first == 0 else sampler.first_cdf[first - 1]
        draws: list[float | NDArray] = [
            float((lower + sampler.first_cdf[first]) / 2),
            np.where(
                mask[first + 1 :], sampler.probabilities[first + 1 :] / 2, (1 + sampler.probabilities[first + 1 :]) / 2
            ),
        ]

        monkeypatch.setattr(np.random, "default_rng", lambda _, values=draws: FixedGenerator(values))
        np.testing.assert_array_equal(sampler.draw(1, 0)[0], mask)
    assert total == pytest.approx(1, abs=3e-15)


def test_mixture_weights_nested_paths_fresh_streams_and_metadata() -> None:
    """Both channels retain their strata and nested prefixes across budget changes."""
    engine, theta = _problem(3)
    original = copy.deepcopy(circuit_record(engine.circuit))
    sampler = mixture.MixtureSampler(engine.circuit)
    small = sampler.ensemble(16, seed=81, update_index=4)
    large = sampler.ensemble(seed=81, update_index=4)
    again = sampler.ensemble(seed=81, update_index=4)
    assert len(large.maps) == 64
    assert large.record["paths_per_channel"] == 32
    assert large.record["channel_order"] == list(mixture.NOISE_IDS)
    assert large.record["factor_counts"] == [6, 2]
    assert large.record["execution_channel"] == "exact_independent_pauli_factors"
    assert large.record["conditional_paths"] == 62
    assert large.record["deterministic_strata"] == 2
    assert all(isinstance(seed, str) for seed in large.record["channel_base_seeds"])
    assert _map_key(again.maps) == _map_key(large.maps)
    assert _map_key(small.maps[:16]) == _map_key(large.maps[:16])
    assert _map_key(small.maps[16:]) == _map_key(large.maps[32:48])
    np.testing.assert_array_equal(large.weights, again.weights)
    for index, factor_sampler in enumerate(sampler.samplers):
        block = large.weights[index * 32 : (index + 1) * 32]
        assert block.sum() == pytest.approx(0.5, abs=2e-15)
        assert block[0] == pytest.approx(factor_sampler.no_factor_probability / 2)
        np.testing.assert_array_equal(block[1:], np.full(31, factor_sampler.event_probability / 62))
    assert not large.weights.flags["W"]
    large.gradient(engine, theta)
    assert circuit_record(engine.circuit) == original
    assert sampler.circuit_signature == digest(original)
    assert large.record["sampling_cpu_seconds"] >= 0
    assert sampler.setup_cpu_seconds >= 0
    purposes: list[mixture.Namespaces] = ["training", "development_diagnostics", "final_evaluation"]
    bases = [
        mixture.stream_seed(81, step, channel, purpose)
        for step, channel, purpose in product([0, 4, 2**32 - 1], [0, 1], purposes)
    ]
    assert len(set(bases)) == len(bases)
    assert all(abs(a - b) >= 2**32 for a, b in product(bases, repeat=2) if a != b)
    assert sampler.ensemble(seed=81, update_index=5).record["channel_base_seeds"] != large.record["channel_base_seeds"]


def test_sample_gradient_is_linear_half_mixture_and_sampled_finite_difference() -> None:
    """The two channel weights are halved exactly once, including shared parameters."""
    engine, theta = _problem(2)
    batch = mixture.MixtureSampler(engine.circuit).ensemble(5, seed=17, update_index=2)
    gradient, loss, fidelity = batch.gradient(engine, theta)
    left = engine.gradient(theta, batch.maps[:5], weights=2 * batch.weights[:5])
    right = engine.gradient(theta, batch.maps[5:], weights=2 * batch.weights[5:])
    np.testing.assert_allclose(gradient, (left[0] + right[0]) / 2, atol=3e-16, rtol=0)
    assert loss == pytest.approx((left[1] + right[1]) / 2, abs=2e-16)
    assert fidelity == pytest.approx((left[2] + right[2]) / 2, abs=2e-16)
    for direction, value in zip(np.eye(3), gradient, strict=True):
        upper = engine.metrics(theta + 1e-6 * direction, batch.maps, weights=batch.weights)[0]
        lower = engine.metrics(theta - 1e-6 * direction, batch.maps, weights=batch.weights)[0]
        assert value == pytest.approx((upper - lower) / 2e-6, abs=4e-10)


def test_online_and_adaptive_wrappers_use_weighted_backend_and_shared_occurrences() -> None:
    """Finite online updates match existing weighted replay; zero-step signals match gradients."""
    engine, theta = _problem(2)
    batch = mixture.MixtureSampler(engine.circuit).ensemble(4, seed=61, update_index=0)
    np.testing.assert_allclose(batch.online_update(engine, theta, 0)[1], batch.gradient(engine, theta)[0], atol=2e-16)
    expected = engine.online_update(theta, 0.04, batch.maps, weights=batch.weights)
    actual = batch.online_update(engine, theta, 0.04)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left, right)
    second, counts = np.zeros(3), np.zeros(3, dtype=np.int64)
    expected_adaptive = engine.adaptive_online_update(
        theta, 0.02, second, counts, batch.maps, weights=batch.weights, beta2=0.8, epsilon=1e-6
    )
    actual_adaptive = batch.adaptive_online_update(engine, theta, 0.02, second, counts, beta2=0.8, epsilon=1e-6)
    for left, right in zip(actual_adaptive, expected_adaptive, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(actual_adaptive[3], [2, 1, 0])
    np.testing.assert_array_equal(second, np.zeros(3))
    np.testing.assert_array_equal(counts, np.zeros(3, dtype=int))
    np.testing.assert_array_equal(theta, [0.4, -0.7, 0.9])


@pytest.mark.parametrize("mode", ["gradient", "online", "adaptive_online", "metrics"])
def test_resource_counts_match_actual_local_operator_calls(
    mode: mixture.Modes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Instrument the backend primitive so repeated tangents and zero weights are charged."""
    engine, theta = _problem(2)
    batch = mixture.MixtureSampler(engine.circuit).ensemble(4, seed=11, update_index=0)
    expected = batch.resource_counts(mode)
    rows = 0

    def counted(states: NDArray, operator: NDArray, sites: tuple[int, ...], qubits: int) -> NDArray:
        """Count statevector rows passed to every local operator application.

        Returns:
            The unchanged primitive's numerical result.
        """
        nonlocal rows
        rows += len(states)
        return apply_local(states, operator, sites, qubits)

    monkeypatch.setattr(statevector, "_apply", counted)
    if mode == "gradient":
        batch.gradient(engine, theta)
    elif mode == "online":
        batch.online_update(engine, theta, 0.02)
    elif mode == "adaptive_online":
        batch.adaptive_online_update(engine, theta, 0.02, np.zeros(3), np.zeros(3, dtype=np.int64))
    else:
        engine.metrics(theta, batch.maps, weights=batch.weights)
    assert rows == expected["total_local_operator_applications"]
    assert expected["paths"] == 8
    assert expected["exact_channel_gradient_calls"] == 0


def test_all_disabled_paths_remain_counted_and_equal_noiseless() -> None:
    """The zero-noise limit keeps both requested blocks with zero-weight duplicates."""
    engine, theta = _problem(2, disabled=True)
    batch = mixture.MixtureSampler(engine.circuit).ensemble(4, seed=1, update_index=0)
    np.testing.assert_array_equal(batch.weights, [0.5, 0, 0, 0, 0.5, 0, 0, 0])
    assert batch.resource_counts("gradient")["paths"] == 8
    assert batch.resource_counts("gradient")["noise_factor_applications"] == 0
    actual, loss, fidelity = batch.gradient(engine, theta)
    expected = engine.gradient(theta)
    np.testing.assert_allclose(actual, expected[0], atol=2e-16, rtol=0)
    assert loss == pytest.approx(expected[1], abs=3e-16)
    assert fidelity == pytest.approx(expected[2], abs=3e-16)


def test_cancellation_paths_are_retained_as_nonempty_factor_events() -> None:
    """Selecting all X/Y/Z factors can cancel physically but stays in the conditional stratum."""
    engine, theta = _problem(1)
    sampler = mixture.MixtureSampler(engine.circuit).samplers[0]
    maps = sampler.maps(np.ones((1, 3), dtype=bool))
    assert sum(len(gate.operators) for gate in maps[0]) == 3
    assert engine.metrics(theta, maps)[1] == pytest.approx(engine.metrics(theta)[1], abs=2e-16)


@pytest.mark.parametrize("paths", [0, 1, True, 2.0, 2**32])
def test_invalid_path_counts(paths: int) -> None:
    """Reject unweighted or ambiguous budgets before allocating arrays."""
    engine, _ = _problem(1)
    with pytest.raises(ValueError, match="Each channel"):
        mixture.MixtureSampler(engine.circuit).ensemble(paths, seed=0, update_index=0)


@pytest.mark.parametrize(
    ("seed", "update", "channel", "namespace"),
    [
        (-1, 0, 0, "training"),
        (True, 0, 0, "training"),
        (0, -1, 0, "training"),
        (0, True, 0, "training"),
        (0, 2**32, 0, "training"),
        (0, 0, 2, "training"),
        (0, 0, True, "training"),
        (0, 0, 0, "unknown"),
    ],
)
def test_invalid_stream_arguments(seed: int, update: int, channel: int, namespace: mixture.Namespaces) -> None:
    """Reject overlapping counter allocations and unnamed random purposes."""
    with pytest.raises(ValueError, match="root seed"):
        mixture.stream_seed(seed, update, channel, namespace)


def test_invalid_circuit_and_stale_or_nonzero_replay_are_rejected() -> None:
    """Do not reuse a structural sampler after changing gate contexts or preparation."""
    engine, theta = _problem(2)
    batch = mixture.MixtureSampler(engine.circuit).ensemble(2, seed=0, update_index=0)
    bad = copy.deepcopy(engine.circuit)
    bad.gates[0].data_map = lambda _: 1.0
    with pytest.raises(ValueError, match="fixed metadata"):
        mixture.MixtureSampler(bad)
    bad = copy.deepcopy(engine.circuit)
    bad.gates[1].angle_scale = float("nan")
    with pytest.raises(ValueError, match="fixed metadata"):
        mixture.DephasingFactorSampler(bad)
    engine.circuit.gates[1].noise_enabled = False
    with pytest.raises(ValueError, match="recorded circuit"):
        batch.gradient(engine, theta)
    engine, _ = _problem(2)
    engine.initial[:] = 0
    engine.initial[1] = 1
    with pytest.raises(ValueError, match="all-zero"):
        batch.online_update(engine, theta, 0.02)
    with pytest.raises(ValueError, match="Unknown"):
        batch.resource_counts(cast("mixture.Modes", "cross"))
