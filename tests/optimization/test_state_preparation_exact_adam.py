# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent adjoint checks and reproducibility tests for exact Adam."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider, exact_density_fidelity
from benchmarks.state_preparation.targets import load_target, load_target_collection
from experiments import state_preparation_exact_adam as experiment
from experiments import state_preparation_pilot as pilot
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KrotovTruncation,
    ParameterizedCircuit,
    ParameterizedGate,
    state_preparation_contribution,
)

if TYPE_CHECKING:
    from pathlib import Path


def small_problem() -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Return a nontrivial ordered circuit with shared/scaled parameters."""
    circuit = ParameterizedCircuit(
        3,
        [
            ParameterizedGate("h", (0,)),
            ParameterizedGate("ry", (2,), param_index=0, angle_scale=-1.7, angle_offset=0.2),
            ParameterizedGate("cx", (2, 0)),
            ParameterizedGate("rxx", (2, 0), param_index=1, angle_scale=0.6),
            ParameterizedGate("p", (1,), param_index=0, angle_offset=0.8, noise_enabled=False),
            ParameterizedGate("ryy", (1, 2), param_index=2),
            ParameterizedGate("cp", (2, 0), param_index=3, angle_scale=-0.9),
        ],
    )
    rng = np.random.default_rng(909)
    target = rng.normal(size=8) + 1j * rng.normal(size=8)
    target /= np.linalg.norm(target)
    return circuit, rng.normal(size=4), target


@pytest.mark.parametrize("errors", [(0.0, 0.0), (0.00064, 0.0051), (0.13, 0.24)])
def test_density_adjoint_matches_independent_oracle_and_finite_differences(errors: tuple[float, float]) -> None:
    """Check both loss and every shared-parameter derivative at several rates."""
    circuit, theta, target = small_problem()
    calibration = CalibratedLocalDepolarizingProvider(*errors)
    provider = ExactLocalPauliProvider(calibration)
    gradient, loss, fidelity = density_loss_gradient(circuit, theta, target, calibration)
    assert fidelity == pytest.approx(exact_density_fidelity(circuit, theta, target, provider), abs=1e-12)
    assert loss == 1 - fidelity
    for index in range(theta.size):
        shift = np.eye(theta.size)[index] * 1e-6
        upper = 1 - exact_density_fidelity(circuit, theta + shift, target, provider)
        lower = 1 - exact_density_fidelity(circuit, theta - shift, target, provider)
        assert gradient[index] == pytest.approx((upper - lower) / 2e-6, abs=2e-8)
    if errors == (0.0, 0.0):
        expected, expected_loss, _ = state_preparation_contribution(circuit, theta, target, MPS(3), KrotovTruncation())
        np.testing.assert_allclose(gradient, expected, atol=1e-11, rtol=0)
        assert loss == pytest.approx(expected_loss, abs=1e-12)


@pytest.mark.parametrize("gate_name", ["rx", "ry", "rz", "p", "rxx", "ryy", "rzz", "cp"])
def test_all_trainable_gate_generators_and_reversed_sites(gate_name: str) -> None:
    """Test each supported generator, offset, scale, and noise-disabled gate."""
    circuit, theta, target = small_problem()
    sites = (2, 0) if gate_name in {"rxx", "ryy", "rzz", "cp"} else (1,)
    circuit.gates.append(
        ParameterizedGate(gate_name, sites, param_index=1, angle_scale=2.3, angle_offset=0.3, noise_enabled=False)
    )
    calibration = CalibratedLocalDepolarizingProvider()
    gradient, _, _ = density_loss_gradient(circuit, theta, target, calibration)
    shift = np.eye(4)[1] * 1e-6
    provider = ExactLocalPauliProvider(calibration)
    difference = (
        exact_density_fidelity(circuit, theta - shift, target, provider)
        - exact_density_fidelity(circuit, theta + shift, target, provider)
    ) / 2e-6
    assert gradient[1] == pytest.approx(difference, abs=2e-8)


@pytest.mark.parametrize("target_id", pilot.TARGETS)
def test_q6_target_families_match_density_oracle_and_noiseless_mps(target_id: str) -> None:
    """Check the production-sized gate convention across all four families."""
    circuit = pilot.pilot_circuit(2)
    theta = np.random.default_rng(42).normal(0, 0.3, circuit.num_params)
    target = load_target(6, target_id).state_vector_copy()
    zero = CalibratedLocalDepolarizingProvider(0, 0)
    gradient, loss, _ = density_loss_gradient(circuit, theta, target, zero)
    reference, expected_loss, _ = state_preparation_contribution(circuit, theta, target, MPS(6), KrotovTruncation())
    np.testing.assert_allclose(gradient, reference, atol=1e-11, rtol=0)
    assert loss == pytest.approx(expected_loss, abs=1e-11)
    _, _, fidelity = density_loss_gradient(circuit, theta, target, CalibratedLocalDepolarizingProvider())
    assert fidelity == pytest.approx(
        exact_density_fidelity(circuit, theta, target, ExactLocalPauliProvider()), abs=1e-11
    )


def test_zero_angle_gate_keeps_noise_and_empty_circuit_is_supported() -> None:
    """An identity-valued retained rotation is still a noisy window."""
    target = np.array([1, 0], dtype=complex)
    circuit = ParameterizedCircuit(1, [ParameterizedGate("rz", (0,), param_index=0)])
    _, _, fidelity = density_loss_gradient(circuit, np.zeros(1), target, CalibratedLocalDepolarizingProvider())
    assert fidelity < 1
    gradient, loss, fidelity = density_loss_gradient(
        ParameterizedCircuit(1, []), np.zeros(0), target, CalibratedLocalDepolarizingProvider()
    )
    assert gradient.size == 0
    assert loss == 0
    assert fidelity == 1


@pytest.mark.parametrize(
    "invalid", ["large", "theta_shape", "theta_nan", "target_norm", "target_nan", "target_shape", "data_map"]
)
def test_invalid_density_inputs_are_rejected(invalid: str) -> None:
    """Reject malformed inputs before optimization can write an endpoint."""
    circuit, theta, target = small_problem()
    if invalid == "large":
        circuit.num_qubits = 7
    elif invalid == "theta_shape":
        theta = theta[:-1]
    elif invalid == "theta_nan":
        theta[0] = np.nan
    elif invalid == "target_norm":
        target *= 2
    elif invalid == "target_nan":
        target[0] = np.nan
    elif invalid == "target_shape":
        target = target[:-1]
    else:
        circuit.gates[1].data_map = lambda _x: 0.1
    with pytest.raises(ValueError, match="Exact density adjoint"):
        density_loss_gradient(circuit, theta, target, CalibratedLocalDepolarizingProvider())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("updates", 0),
        ("updates", True),
        ("learning_rate", -0.1),
        ("learning_rate", np.nan),
        ("beta1", 1.0),
        ("beta2", -0.1),
        ("epsilon", 0),
    ],
)
def test_invalid_adam_config(field: str, value: float) -> None:
    """Require finite valid Adam parameters."""
    with pytest.raises(ValueError, match="configuration"):
        replace(experiment.AdamConfig(), **{field: value})


def test_adam_moments_bias_correction_and_input_immutability() -> None:
    """Compare changing scalar gradients to an explicit weighted-history formula."""
    config = experiment.AdamConfig()
    theta = np.array([0.3])
    first = second = np.array([0.0])
    expected_theta = 0.3
    history = [2.0, -0.5, 0.0, 0.8]
    for step, gradient in enumerate(history, 1):
        original = theta.copy()
        result, first, second = experiment.adam_step(theta, np.array([gradient]), first, second, step, config)
        np.testing.assert_array_equal(theta, original)
        first_expected = sum(
            (1 - config.beta1) * config.beta1 ** (step - i) * g for i, g in enumerate(history[:step], 1)
        )
        second_expected = sum(
            (1 - config.beta2) * config.beta2 ** (step - i) * g**2 for i, g in enumerate(history[:step], 1)
        )
        expected_theta -= (
            config.learning_rate
            * (first_expected / (1 - config.beta1**step))
            / (np.sqrt(second_expected / (1 - config.beta2**step)) + config.epsilon)
        )
        assert result[0] == pytest.approx(expected_theta, abs=1e-15)
        theta = result
    with pytest.raises(ValueError, match="state or gradient"):
        experiment.adam_step(theta, np.array([np.nan]), first, second, 5, config)


@pytest.mark.parametrize("arm", experiment.ARMS)
def test_interruption_preserves_adam_moments_and_exact_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    """Interrupted runs reproduce every parameter, moment, loss, and evaluation."""
    circuit, theta, target = small_problem()
    config = experiment.AdamConfig(updates=12)
    baseline = experiment.optimize(tmp_path / "full.json", circuit, theta, target, arm, config, "test")
    writer = pilot.write_record

    def interrupt(path: Path, record: dict[str, object]) -> None:
        writer(path, record)
        if record.get("completed_updates") == 10:
            msg = "simulated interruption"
            raise RuntimeError(msg)

    monkeypatch.setattr(pilot, "write_record", interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        experiment.optimize(tmp_path / "resumed.json", circuit, theta, target, arm, config, "test")
    monkeypatch.setattr(pilot, "write_record", writer)
    resumed = experiment.optimize(tmp_path / "resumed.json", circuit, theta, target, arm, config, "test")
    for key in (
        "theta",
        "first_moment",
        "second_moment",
        "trace",
        "evaluations",
        "completed_updates",
        "final_gradient_norm",
    ):
        assert resumed[key] == baseline[key]
    assert len(cast("list[object]", resumed["trace"])) == 12
    with pytest.raises(ValueError, match="configuration mismatch"):
        experiment.optimize(
            tmp_path / "resumed.json", circuit, theta, target, arm, replace(config, learning_rate=0.01), "test"
        )
    corrupt = pilot.read_record(tmp_path / "resumed.json")
    corrupt["second_moment"] = [-1] * 4
    pilot.write_record(tmp_path / "resumed.json", corrupt)
    with pytest.raises(ValueError, match="progress or moments"):
        experiment.optimize(tmp_path / "resumed.json", circuit, theta, target, arm, config, "test")


def test_nonfinite_gradient_stops_without_saving_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never label a failed update as a completed experiment."""
    circuit, theta, target = small_problem()
    monkeypatch.setattr(experiment, "density_loss_gradient", lambda *_: (np.full(4, np.nan), 0.5, 0.5))
    with pytest.raises(ValueError, match="state or gradient"):
        experiment.optimize(
            tmp_path / "failed.json", circuit, theta, target, "exact_noisy", experiment.AdamConfig(updates=1), "test"
        )
    assert not (tmp_path / "failed.json").exists()


@pytest.fixture
def source_panel(tmp_path: Path) -> Path:
    """Create a complete small synthetic source panel without external artifacts.

    Returns:
        Synthetic source root with all eight compatible start records.
    """
    original = tmp_path / "original"
    original_manifest: dict[str, object] = {"target_fixture_checksum": load_target_collection().checksum}
    pilot.write_record(original / "manifest.json", original_manifest)
    root = tmp_path / "source"
    manifest: dict[str, object] = {
        "version": experiment.SOURCE_VERSION,
        "source_root": str(original),
        "source_manifest_checksum": pilot.digest(original_manifest),
        "targets": list(pilot.TARGETS),
        "noise": ExactLocalPauliProvider().scientific_dict(),
        "config": {
            "source_updates": 100,
            "repair_updates": 20,
            "refinement_updates": 50,
            "training_trajectories": 8,
            "seeds": [11, 29],
            "root_seed": 2026090520,
        },
    }
    pilot.write_record(root / "manifest.json", manifest)
    pilot.write_record(root / "result.json", {"status": "complete", "manifest_checksum": pilot.digest(manifest)})
    circuit = pilot.pilot_circuit(2)
    for target_id in pilot.TARGETS:
        target = load_target(6, target_id).state_vector_copy()
        for seed in experiment.SEEDS:
            theta = np.random.default_rng(seed).normal(0, 0.2, circuit.num_params)
            candidate = {
                **experiment.evaluate(circuit, theta, target, 0),
                "route": experiment.SOURCE_ROUTE,
                "circuit": pilot.circuit_record(circuit),
            }
            record: dict[str, object] = {
                "target_id": target_id,
                "replica": seed,
                "setting": experiment.SOURCE_SETTING,
                "schedule": {"online_updates": 25, "online_rate": 0.7},
                "include_warm": False,
                "candidates": [candidate],
            }
            pilot.write_record(root / experiment.SOURCE_SETTING / target_id / f"seed_{seed}" / "result.json", record)
    return root


def test_complete_panel_serial_parallel_resume_and_source_mutation(source_panel: Path, tmp_path: Path) -> None:
    """All 16 arms use identical starts and aggregate independently of worker order."""
    config = experiment.AdamConfig(updates=1)
    serial = experiment.run_experiment(tmp_path / "serial", source_panel, config=config, workers=1)
    parallel = experiment.run_experiment(tmp_path / "parallel", source_panel, config=config, workers=2)
    assert serial["aggregate"] == parallel["aggregate"]
    assert serial["paired_differences"] == parallel["paired_differences"]
    assert serial["completed_continuations"] == 16
    assert serial["gradient_updates"] == 16
    resumed = experiment.run_experiment(tmp_path / "serial", source_panel, config=config, workers=2, resume=True)
    assert resumed == serial
    for relative in cast("dict[str, object]", serial["record_checksums"]):
        record = pilot.read_record(tmp_path / "serial" / relative)
        endpoints = cast("list[dict[str, object]]", record["candidates"])
        assert {row["route"] for row in endpoints} == {"frozen", *experiment.ARMS}
        directory = (tmp_path / "serial" / relative).parent
        zero, noisy = [pilot.read_record(directory / f"{arm}.json") for arm in experiment.ARMS]
        assert cast("list[object]", zero["evaluations"])[0] == cast("list[object]", noisy["evaluations"])[0]
    checkpoint = tmp_path / "parallel" / pilot.TARGETS[0] / "seed_11" / "noiseless.json"
    damaged = pilot.read_record(checkpoint)
    damaged["final_gradient_norm"] = 0
    pilot.write_record(checkpoint, damaged)
    with pytest.raises(ValueError, match="Adam checkpoints"):
        experiment.run_experiment(tmp_path / "parallel", source_panel, config=config, resume=True)
    with pytest.raises(ValueError, match="explicit resume"):
        experiment.run_experiment(tmp_path / "serial", source_panel, config=config)
    with pytest.raises(ValueError, match="identical manifest"):
        experiment.run_experiment(
            tmp_path / "serial", source_panel, config=replace(config, learning_rate=0.01), resume=True
        )
    path = source_panel / experiment.SOURCE_SETTING / pilot.TARGETS[0] / "seed_11" / "result.json"
    record = pilot.read_record(path)
    record["replica"] = 99
    pilot.write_record(path, record)
    with pytest.raises(ValueError, match="selected source"):
        experiment.load_starts(source_panel)


def test_cli_reports_failure_without_starting_other_experiments(tmp_path: Path) -> None:
    """Missing source data returns failure rather than silently constructing new starts."""
    assert experiment.main(["--source-root", str(tmp_path / "missing"), "--output-root", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()
