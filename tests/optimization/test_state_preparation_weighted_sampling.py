# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Conditional Pauli sampling, unbiased weights and diagnostic accounting tests."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from experiments import state_preparation_pilot as pilot
from experiments import state_preparation_trajectory_comparison as comparison
from experiments import state_preparation_trajectory_diagnostics as development
from experiments import state_preparation_weighted_sampling as weighted
from mqt.yaqs.optimization import KrotovNoiseMap, ParameterizedCircuit, ParameterizedGate, StatevectorBackend

if TYPE_CHECKING:
    from pathlib import Path


def engine_case(*, two_sites: bool = False) -> tuple[StatevectorBackend, np.ndarray]:
    """Build a complex-target circuit with reversed sites and one disabled noise gate.

    Returns:
        A small statevector engine and nonzero parameters.
    """
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate(
                "ry", (0,), param_index=0, angle_scale=-1.3, angle_offset=0.2, noise_enabled=not two_sites
            ),
            ParameterizedGate("rxx", (1, 0), param_index=1, noise_enabled=two_sites),
        ],
    )
    return StatevectorBackend(circuit, np.array([1, 2j, -3, 4j]) / np.sqrt(30)), np.array([0.4, -0.7])


@pytest.mark.parametrize("two_sites", [False, True])
def test_exhaustive_weighted_expectation_equals_exact_density(*, two_sites: bool) -> None:
    """Enumerate all 3-/6-factor outcomes, including canceling multiple factors."""
    engine, theta = engine_case(two_sites=two_sites)
    calibration = CalibratedLocalDepolarizingProvider(0.15, 0.3)
    sampler = weighted.FactorSampler(engine.circuit, calibration)
    assert len(sampler.factors) == (6 if two_sites else 3)
    gradient = np.zeros_like(theta)
    loss = 0.0
    probability_sum = 0.0
    for pattern in product([False, True], repeat=len(sampler.factors)):
        mask = np.asarray(pattern)
        if not mask.any():
            continue
        probability = (
            float(np.prod(np.where(mask, sampler.probabilities, 1 - sampler.probabilities))) / sampler.event_probability
        )
        maps = sampler.maps(mask[None, :])
        sample_gradient, sample_loss = weighted.weighted_gradient(engine, theta, sampler, maps)
        gradient += probability * sample_gradient
        loss += probability * sample_loss
        probability_sum += probability
    expected, expected_loss, _ = density_loss_gradient(engine.circuit, theta, engine.target, calibration)
    assert probability_sum == pytest.approx(1, abs=2e-15)
    np.testing.assert_allclose(gradient, expected, atol=2e-14, rtol=0)
    assert loss == pytest.approx(expected_loss, abs=2e-14)


def test_conditional_distribution_nested_streams_and_cancellations() -> None:
    """First-event conditioning has the correct joint law and does not erase cancellations."""
    engine, theta = engine_case()
    sampler = weighted.FactorSampler(engine.circuit, CalibratedLocalDepolarizingProvider(0.2, 0.3))
    masks = sampler.draw(20000, 7100)
    assert np.all(masks.any(axis=1))
    np.testing.assert_array_equal(masks[:7], sampler.draw(7, 7100))
    for pattern in product([False, True], repeat=3):
        mask = np.array(pattern)
        if mask.any():
            expected = (
                np.prod(np.where(mask, sampler.probabilities, 1 - sampler.probabilities)) / sampler.event_probability
            )
            observed = np.mean(np.all(masks == mask, axis=1))
            assert observed == pytest.approx(expected, abs=0.015)
    canceling_maps = sampler.maps(np.ones((1, 3), dtype=bool))
    assert development.noise_events(canceling_maps)["sampled_factors"] == 3
    assert engine.metrics(theta, canceling_maps)[0] == pytest.approx(engine.metrics(theta)[0], abs=1e-14)


def test_weighted_gradient_matches_finite_difference_and_zero_noise() -> None:
    """Weights differentiate the sampled physical loss, with a deterministic zero-noise limit."""
    engine, theta = engine_case(two_sites=True)
    sampler = weighted.FactorSampler(engine.circuit, CalibratedLocalDepolarizingProvider())
    maps = sampler.maps(sampler.draw(7, 43))
    gradient, _ = weighted.weighted_gradient(engine, theta, sampler, maps)
    epsilon = 1e-6
    finite_difference = []
    for direction in np.eye(len(theta)):
        upper = weighted.weighted_gradient(engine, theta + epsilon * direction, sampler, maps)[1]
        lower = weighted.weighted_gradient(engine, theta - epsilon * direction, sampler, maps)[1]
        finite_difference.append((upper - lower) / (2 * epsilon))
    np.testing.assert_allclose(gradient, finite_difference, atol=2e-9, rtol=0)
    zero = weighted.FactorSampler(engine.circuit, CalibratedLocalDepolarizingProvider(0, 0))
    assert zero.no_factor_probability == 1
    assert zero.event_probability == 0
    np.testing.assert_array_equal(weighted.weighted_gradient(engine, theta, zero, None)[0], engine.gradient(theta)[0])
    with pytest.raises(ValueError, match="zero-probability"):
        zero.draw(2, 1)
    with pytest.raises(ValueError, match="conditional maps"):
        weighted.weighted_gradient(engine, theta, sampler, None)


@pytest.mark.parametrize(("count", "seed"), [(0, 0), (True, 0), (2, -1), (2, True)])
def test_invalid_sampling_arguments(count: int, seed: int) -> None:
    """Reject malformed sample counts and seeds before drawing randomness."""
    engine, _ = engine_case()
    sampler = weighted.FactorSampler(engine.circuit, CalibratedLocalDepolarizingProvider())
    with pytest.raises(ValueError, match="integers"):
        sampler.draw(count, seed)


def test_bad_masks_and_all_disabled_noise() -> None:
    """Shape/type/conditioning validation also covers a circuit without noise windows."""
    engine, theta = engine_case()
    sampler = weighted.FactorSampler(engine.circuit, CalibratedLocalDepolarizingProvider())
    for masks in [np.zeros((1, 3), dtype=bool), np.ones((1, 2), dtype=bool), np.ones((1, 3)), np.ones(3, dtype=bool)]:
        with pytest.raises(ValueError, match="Boolean conditional masks"):
            sampler.maps(cast("Any", masks))
    circuit = ParameterizedCircuit(2, [ParameterizedGate("ry", (0,), param_index=0, noise_enabled=False)])
    zero = weighted.FactorSampler(circuit, CalibratedLocalDepolarizingProvider())
    assert zero.event_probability == 0
    assert not zero.factors
    with pytest.raises(ValueError, match="sampled factor"):
        weighted.weighted_gradient(engine, theta, sampler, ((KrotovNoiseMap(), KrotovNoiseMap()),))


def test_weighted_diagnostic_parallel_resume_and_costs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Paired diagnostics preserve gradients, budgets, checksums and completed resume."""
    engine, theta = engine_case()
    fidelity = density_loss_gradient(engine.circuit, theta, engine.target, CalibratedLocalDepolarizingProvider())[2]
    cases: list[dict[str, Any]] = [
        {
            "target": comparison.target_record(name, "haar", engine.target),
            "circuit": pilot.circuit_record(engine.circuit),
            "initial": theta.tolist(),
            "checkpoints": [{"update": k, "theta": theta.tolist(), "exact_noisy_fidelity": fidelity} for k in (0, 25)],
        }
        for name in ("a", "b")
    ]
    monkeypatch.setattr(development, "load_cases", lambda _: ({"source": "test"}, cases))
    config = development.Config(checkpoints=(0, 25), diagnostic_counts=(2, 4), diagnostic_repeats=2)
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    result = weighted.run(serial, tmp_path, config=config, workers=1)
    before = {p: p.read_bytes() for p in serial.rglob("*") if p.is_file()}
    assert weighted.run(serial, tmp_path, config=config, workers=2, resume=True) == result
    assert before == {p: p.read_bytes() for p in serial.rglob("*") if p.is_file()}
    other = weighted.run(parallel, tmp_path, config=config, workers=2)
    for a, b in zip(result["rows"], other["rows"], strict=True):
        left = cast("dict[str, Any]", pilot.read_record(serial / a["path"]))
        right = cast("dict[str, Any]", pilot.read_record(parallel / b["path"]))
        assert pilot.digest(left) == a["checksum"]
        for x, y in zip(left["rows"], right["rows"], strict=True):
            assert x["gradient"] == y["gradient"]
            assert (
                x["cpu_seconds"]
                == x["sampling"]["cpu_seconds"] + x["gradient_cost"]["cpu_seconds"] + x["setup_cpu_seconds"]
            )
            if x["estimator"] == "weighted":
                assert x["conditional_trajectories"] == x["trajectories"] - 1
                assert x["event_free_trajectories"] == 0
    record_path = serial / result["rows"][0]["path"]
    record = pilot.read_record(record_path)
    record["signature"] = "changed"
    pilot.write_record(record_path, record)
    with pytest.raises(ValueError, match="configuration mismatch"):
        weighted.run(serial, tmp_path, config=config, resume=True)


def test_diagnostic_rejects_one_path(tmp_path: Path) -> None:
    """The matched total budget must leave at least one conditional path."""
    with pytest.raises(ValueError, match="two total paths"):
        weighted.run(tmp_path / "new", tmp_path, config=development.Config(diagnostic_counts=(1,)))
