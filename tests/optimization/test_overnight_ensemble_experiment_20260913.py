# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Engineering tests for the frozen ensemble experiment and interruption ledger."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from experiments import state_preparation_exact_adam as adam
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import channels
from experiments.overnight_state_preparation_20260913_0108 import ensemble_experiment as experiment
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def _problem() -> tuple[ParameterizedCircuit, NDArray, NDArray]:
    """Build a nontrivial q2 circuit compatible with every benchmark profile.

    Returns:
        Circuit, original theta, and normalized target.
    """
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("rx", (0,), param_index=0),
            ParameterizedGate("ry", (1,), param_index=1),
            ParameterizedGate("rzz", (1, 0), param_index=2),
            ParameterizedGate("rxx", (0, 1), param_index=3),
        ],
    )
    target = np.array([1, -0.3j, 0.2, 0.8 + 0.1j], dtype=complex)
    target /= np.linalg.norm(target)
    return circuit, np.array([0.4, -0.5, 0.2, 0.6]), target


def test_declared_six_arms_have_equal_individual_channel_caps() -> None:
    """Mean-two and mean-eleven receive fewer joint calls, not extra hidden work."""
    specs = experiment.arm_specs(experiment.Config())
    assert len(specs) == 6
    assert specs["local_lbfgs"]["joint_cap"] == 1100
    assert specs["mean2_lbfgs"]["joint_cap"] == 550
    assert specs["mean11_lbfgs"]["joint_cap"] == 100
    assert specs["mean11_lbfgs"]["noise_ids"] == list(channels.NOISE_IDS[1:])
    assert {len(spec["noise_ids"]) * spec["joint_cap"] for spec in specs.values()} == {1100}
    assert experiment.Config().adam_learning_rate == pytest.approx(0.02)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("channel_evaluations", 23),
        ("channel_evaluations", True),
        ("history", 0),
        ("checkpoint_every", 0),
        ("adam_learning_rate", True),
    ],
)
def test_invalid_configuration_is_rejected(field: str, value: int) -> None:
    """Invalid budgets cannot silently change the registered comparison."""
    with pytest.raises(ValueError, match="budgets must be positive"):
        experiment.Config(**{field: value})


def test_adam_full_moments_resume_exactly_and_final_loss_is_not_inferred() -> None:
    """Checkpointing moments and the bias-correction step preserves the full path."""
    initial = np.array([0.4, -0.7])
    config = adam.AdamConfig(updates=7, learning_rate=0.02)
    snapshots: list[dict[str, Any]] = []

    def objective(theta: NDArray) -> tuple[float, NDArray]:
        return float(theta @ theta / 2), theta.copy()

    def interrupt(state: dict[str, Any]) -> None:
        snapshots.append(state)
        if state["iterations"] == 3:
            raise InterruptedError

    expected = experiment.minimize_adam(objective, initial, config)
    with pytest.raises(InterruptedError):
        experiment.minimize_adam(objective, initial, config, callback=interrupt)
    prior = copy.deepcopy(snapshots[-1])
    resumed = experiment.minimize_adam(objective, initial, config, state=prior)
    assert resumed == expected
    assert prior == snapshots[-1]
    final_theta = np.asarray(resumed["theta"])
    assert resumed["last_pre_update_loss"] > objective(final_theta)[0]
    assert resumed["iterations"] == resumed["evaluations"] == 7
    bad = copy.deepcopy(prior)
    bad["second_moment"] = [-1, 0]
    with pytest.raises(ValueError, match="malformed numerical arrays"):
        experiment.minimize_adam(objective, initial, config, state=bad)


@pytest.mark.parametrize(
    "method", ["local_lbfgs", "mean2_lbfgs", "mean11_lbfgs", "local_adam", "noiseless_adam", "noiseless_lbfgs"]
)
def test_training_counts_actual_component_work_and_completed_resume_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    """Count joint and component gradients and ensure completed runs are never retrained."""
    circuit, theta, target = _problem()
    config = experiment.Config(channel_evaluations=22, checkpoint_every=2)
    spec = experiment.arm_specs(config)[method]
    result = experiment.train_arm(tmp_path, circuit, theta, target, spec, config)
    components = len(spec["noise_ids"])
    assert result["individual_channel_gradient_calls"] == components * result["joint_gradient_calls"]
    assert result["individual_channel_gradient_calls"] <= 22
    assert result["physical_completed_channel_calls"] == result["individual_channel_gradient_calls"]
    assert result["physical_attempted_channel_calls"] == result["physical_completed_channel_calls"]
    assert result["density_direction_passes"] == 2 * result["individual_channel_gradient_calls"]
    assert result["replayed_completed_channel_calls"] == 0
    assert result["training_trajectories"] == 0
    assert result["curve"][-1]["theta"] == result["theta"]
    if spec["optimizer"] == "adam":
        assert result["joint_gradient_calls"] == 22
        assert all(point["training_loss"] is None for point in result["curve"])
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.json")}

    def fail(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Completed training cannot evaluate any gradient again.")

    monkeypatch.setattr(experiment, "component_gradient", fail)
    assert experiment.train_arm(tmp_path, circuit, theta, target, spec, config) == result
    assert {path.name: path.read_bytes() for path in tmp_path.glob("*.json")} == before


@pytest.mark.parametrize("method", ["local_adam", "local_lbfgs"])
def test_interrupted_training_restores_full_state_and_charges_replayed_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    """Lost accepted work is replayed deterministically and retained in the physical ledger."""
    circuit, theta, target = _problem()
    config = experiment.Config(channel_evaluations=22, checkpoint_every=2 if method == "local_adam" else 100)
    spec = experiment.arm_specs(config)[method]
    reference = experiment.train_arm(tmp_path / "reference", circuit, theta, target, spec, config)
    original = experiment.component_gradient
    called = 0
    interruption_call = 4 if method == "local_adam" else 7

    def interrupted(
        candidate: ParameterizedCircuit, parameters: NDArray, desired: NDArray, noise_id: str, *, fast: bool
    ) -> tuple[NDArray, float, float]:
        nonlocal called
        called += 1
        if called == interruption_call:
            raise InterruptedError
        return original(candidate, parameters, desired, noise_id, fast=fast)

    monkeypatch.setattr(experiment, "component_gradient", interrupted)
    root = tmp_path / "interrupted"
    with pytest.raises(InterruptedError):
        experiment.train_arm(root, circuit, theta, target, spec, config)
    attempts = cast("dict[str, Any]", pilot.read_record(root / "ATTEMPTS.json"))
    assert attempts["attempts"][0]["attempted_channel_calls"] == interruption_call
    assert attempts["attempts"][0]["completed_channel_calls"] == interruption_call - 1
    assert attempts["attempts"][0]["failure"] == "InterruptedError: "
    monkeypatch.setattr(experiment, "component_gradient", original)
    resumed = experiment.train_arm(root, circuit, theta, target, spec, config)
    assert resumed["optimizer"] == reference["optimizer"]
    assert resumed["replayed_completed_channel_calls"] == (1 if method == "local_adam" else 5)
    assert resumed["physical_attempted_channel_calls"] == resumed["physical_completed_channel_calls"] + 1
    saved_attempts = cast("dict[str, Any]", pilot.read_record(root / "ATTEMPTS.json"))
    assert resumed["cpu_seconds"] == sum(row["cpu_seconds"] for row in saved_attempts["attempts"])


def test_fast_single_channel_and_generic_ensemble_dispatch_are_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not impose the generic evaluator overhead on the strong single-channel baselines."""
    circuit, theta, target = _problem()
    calls = []

    def fast(*_args: object) -> tuple[NDArray, float, float]:
        calls.append("fast")
        return np.zeros_like(theta), 0.0, 1.0

    def general(*_args: object) -> tuple[NDArray, float, float]:
        calls.append("general")
        return np.zeros_like(theta), 0.0, 1.0

    monkeypatch.setattr(experiment, "density_loss_gradient", fast)
    monkeypatch.setattr(experiment.ensemble_objective, "loss_gradient", general)
    for noise_id, use_fast in (
        ("noiseless", True),
        ("depolarizing_1s_all", True),
        ("depolarizing_1s_all", False),
        ("dephasing_1s_all", True),
    ):
        experiment.component_gradient(circuit, theta, target, noise_id, fast=use_fast)
    assert calls == ["fast", "fast", "general", "general"]


def test_common_cpu_selection_uses_latest_available_parameters_even_when_loss_worsens() -> None:
    """Common-time points are selected before reading any scientific evaluation scores."""
    curves = [
        {"theta": [index], "cpu_seconds": float(index), "training_loss": loss}
        for index, loss in enumerate([0.3, 0.1, 0.9])
    ]
    arms = {"first": {"cpu_seconds": 2.0, "curve": curves}, "second": {"cpu_seconds": 3.0, "curve": curves}}
    common, points = experiment.common_cpu_points(arms)
    assert common == 2
    assert points[0]["selected"]["first"]["theta"] == [0]
    assert points[-1]["selected"]["first"]["theta"] == [2]
    assert points[1]["selected"]["first"]["theta"] == [0]


def test_gradient_alignment_diagnostic_preserves_negative_and_undefined_cosines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative alignment is not clipped away and zero-norm comparisons stay undefined."""
    circuit, theta, target = _problem()
    profiles = experiment.ensemble_objective.NOISY_IDS

    def gradient(
        _circuit: object, _theta: object, _target: object, key: str, **_kwargs: object
    ) -> tuple[NDArray, float, float]:
        vector = np.zeros_like(theta)
        vector[0] = 1 if key == profiles[0] else (-1 if key == profiles[1] else 0)
        return vector, 0.0, 1.0

    monkeypatch.setattr(experiment, "component_gradient", gradient)
    result = experiment.gradient_cosines(tmp_path, circuit, theta, target)
    assert result["cosines"][profiles[0]][profiles[1]] == -1
    assert result["cosines"][profiles[0]][profiles[2]] is None
    assert result["individual_channel_gradient_calls"] == 11
    assert result["density_direction_passes"] == 22
    assert experiment.gradient_cosines(tmp_path, circuit, theta, target) == result


def test_complete_q2_panel_has_all12_fanout_common_cpu_and_read_only_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise all six arms and the cached exact score ledger on an isolated q2 case."""
    circuit, theta, target = _problem()
    source = {
        "target_id": "engineering_fixture",
        "replica": 11,
        "candidate": {"circuit": pilot.circuit_record(circuit), "theta": theta.tolist()},
    }
    monkeypatch.setattr(experiment, "load_target", lambda *_args: SimpleNamespace(state_vector_copy=target.copy))
    config = experiment.Config(channel_evaluations=22, checkpoint_every=2)
    result = experiment.run_case(tmp_path, source, config)
    assert result["status"] == "complete"
    assert len(result["arms"]) == 6
    for arm in result["arms"].values():
        assert tuple(arm["final_score"]["fidelities"]) == channels.NOISE_IDS
        assert arm["final_score"]["logical_counts"]["one_qubit"] == 2
        assert arm["final_score"]["logical_counts"]["two_qubit"] == 2
        assert arm["training"]["individual_channel_gradient_calls"] <= 22
    assert len(result["common_cpu_comparison"]["points"]) == 5
    ledger = result["post_training_scoring"]
    assert ledger["forward_channel_evaluations"] == 12 * ledger["unique_parameter_points"]
    assert ledger["density_direction_passes"] == ledger["forward_channel_evaluations"]
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*.json")}
    assert experiment.run_case(tmp_path, source, config) == result
    assert {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*.json")} == before
    experiment.write_scoreboard(tmp_path, [result])
    assert len((tmp_path / "SCOREBOARD.csv").read_text().splitlines()) == 1 + 6 * 12


def test_protocol_prepare_only_freezes_original108_gate_starts_and_rejects_custody_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparation validates metadata only and never executes a scientific gradient."""
    circuit = ParameterizedCircuit(6, [ParameterizedGate("rx", (0,), param_index=0) for _ in range(108)])
    sources: list[dict[str, Any]] = [
        {"target_id": target_id, "replica": 11, "candidate": {"circuit": pilot.circuit_record(circuit), "theta": [0.2]}}
        for target_id in experiment.TARGETS
    ]
    starts_path = tmp_path / "starts.json"
    pilot.write_record(starts_path, {"starts": sources})
    monkeypatch.setattr(experiment.runner, "OUTPUT", tmp_path)
    monkeypatch.setattr(experiment.runner, "custody", lambda: {"benchmark_hashes": {"benchmark": "frozen"}})
    monkeypatch.setattr(experiment, "numerical_sources", lambda: {"implementation": "frozen"})
    output = tmp_path / "prepared"
    assert experiment.main(["--output", str(output), "--prepare-only"]) == 0
    assert len(list(output.iterdir())) == 1
    protocol = cast("dict[str, Any]", pilot.read_record(output / "PROTOCOL.json"))
    assert len(protocol["arms"]) == 6
    assert protocol["config"]["channel_evaluations"] == 1100
    assert len(protocol["starts"]) == 4
    assert experiment.freeze_protocol(output) == protocol
    experiment.verify_protocol(protocol)
    sources[0]["candidate"]["theta"] = [0.3]
    pilot.write_record(starts_path, {"starts": sources})
    with pytest.raises(ValueError, match="custody or numerical source identity changed"):
        experiment.verify_protocol(protocol)


def test_case_level_failure_is_preserved_for_every_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A diagnostic failure cannot silently remove a target from the final table."""
    monkeypatch.setattr(experiment, "verify_protocol", lambda _protocol: None)

    def fail(*_args: object, **_kwargs: object) -> None:
        msg = "Injected diagnostic failure"
        raise ValueError(msg)

    monkeypatch.setattr(experiment, "run_case", fail)
    config = experiment.Config()
    job = {
        "protocol": {"arms": experiment.arm_specs(config), "config": vars(config)},
        "source": {"target_id": "fixture", "replica": 11},
        "root": str(tmp_path / "case"),
        "order_index": 0,
    }
    result = experiment.execute_case(job)
    assert result["status"] == "failed"
    assert len(result["arms"]) == 6
    assert experiment.execute_case(job) == result
    experiment.write_scoreboard(tmp_path, [result])
    text = (tmp_path / "SCOREBOARD.csv").read_text()
    assert len(text.splitlines()) == 73
    assert text.count("Injected diagnostic failure") == 72
