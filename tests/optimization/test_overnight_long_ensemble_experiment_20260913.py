# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tiny engineering checks for the preregistered full-family convergence panel."""

from __future__ import annotations

import csv
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import long_ensemble_experiment as experiment
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def problem() -> tuple[ParameterizedCircuit, NDArray, NDArray]:
    """Return a nontrivial two-qubit circuit and normalized complex target."""
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


def source_fixture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Return a q2 source whose target loader cannot read scientific data."""
    circuit, theta, target = problem()
    monkeypatch.setattr(experiment, "load_target", lambda *_args: SimpleNamespace(state_vector_copy=target.copy))
    return {
        "target_id": "engineering",
        "replica": 11,
        "candidate": {"circuit": pilot.circuit_record(circuit), "theta": theta.tolist()},
    }


def test_exact_four_arm_budget_and_optimizer_freeze() -> None:
    """The longer full-family test changes only cap and includes both optimizers."""
    config = experiment.Config()
    specs = experiment.arm_specs(config)
    assert set(specs) == {"mean2_lbfgs", "mean11_lbfgs", "mean2_adam", "mean11_adam"}
    assert {len(spec["noise_ids"]) * spec["joint_cap"] for spec in specs.values()} == {6050}
    assert specs["mean2_adam"]["joint_cap"] == 3025
    assert specs["mean11_lbfgs"]["joint_cap"] == 550
    assert specs["mean11_adam"]["noise_ids"] == list(experiment.channels.NOISE_IDS[1:])
    assert config.adam_learning_rate == pytest.approx(0.02)
    assert config.history == config.checkpoint_every == 10
    with pytest.raises(ValueError, match="divisible by 22"):
        experiment.Config(channel_evaluations=6051)


@pytest.mark.parametrize("method", ["mean2_lbfgs", "mean11_lbfgs", "mean2_adam", "mean11_adam"])
def test_wrapper_matches_frozen_primitive_and_has_read_only_resume(tmp_path: Path, method: str) -> None:
    """The diagnostic never changes the optimizer trajectory or channel dispatch."""
    circuit, theta, target = problem()
    config = experiment.Config(channel_evaluations=22, checkpoint_every=2)
    spec = experiment.arm_specs(config)[method]
    reference = experiment.base.train_arm(tmp_path / "reference", circuit, theta, target, spec, config)
    root = tmp_path / "wrapped"
    result = experiment.train_arm(root, circuit, theta, target, spec, config)
    assert result["optimizer"] == reference["optimizer"]
    assert result["theta"] == reference["theta"]
    assert result["individual_channel_gradient_calls"] <= 22
    assert result["physical_completed_channel_calls"] == result["individual_channel_gradient_calls"]
    assert result["cpu_scope"] == experiment.CPU_SCOPE
    before = {path.name: path.read_bytes() for path in root.glob("*.json")}
    assert experiment.train_arm(root, circuit, theta, target, spec, config) == result
    assert before == {path.name: path.read_bytes() for path in root.glob("*.json")}


@pytest.mark.parametrize("method", ["mean2_lbfgs", "mean11_lbfgs", "mean2_adam", "mean11_adam"])
def test_exact_interruption_resume_preserves_all_observed_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Partial component gradients and lost accepted work remain charged on resume."""
    circuit, theta, target = problem()
    config = experiment.Config(channel_evaluations=22, checkpoint_every=2)
    spec = experiment.arm_specs(config)[method]
    expected = experiment.train_arm(tmp_path / "reference", circuit, theta, target, spec, config)
    original = experiment.base.component_gradient
    calls = 0

    def interrupted(
        candidate: ParameterizedCircuit, parameters: NDArray, desired: NDArray, noise: str, *, fast: bool
    ) -> tuple[NDArray, float, float]:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise InterruptedError
        return original(candidate, parameters, desired, noise, fast=fast)

    monkeypatch.setattr(experiment.base, "component_gradient", interrupted)
    root = tmp_path / "interrupted"
    with pytest.raises(InterruptedError):
        experiment.train_arm(root, circuit, theta, target, spec, config)
    failed = experiment.physical_training(root)
    assert failed["physical_attempted_channel_calls"] == 4
    assert failed["physical_completed_channel_calls"] == 3
    assert not failed["unknown_kill_cost"]
    monkeypatch.setattr(experiment.base, "component_gradient", original)
    resumed = experiment.train_arm(root, circuit, theta, target, spec, config)
    assert resumed["optimizer"] == expected["optimizer"]
    assert resumed["physical_attempted_channel_calls"] == resumed["physical_completed_channel_calls"] + 1
    assert resumed["replayed_completed_channel_calls"] >= 1
    attempts = cast("dict[str, Any]", pilot.read_record(root / "ATTEMPTS.json"))["attempts"]
    assert resumed["cpu_seconds"] == sum(attempt["cpu_seconds"] for attempt in attempts)


@pytest.mark.parametrize("change", ["running", "changed_attempts", "missing_invocations"])
def test_training_unknown_kill_or_changed_ledger_cannot_resume(tmp_path: Path, change: str) -> None:
    """A cached primitive result does not legitimize unobserved resource consumption."""
    circuit, theta, target = problem()
    config = experiment.Config(channel_evaluations=22)
    spec = experiment.arm_specs(config)["mean11_adam"]
    experiment.train_arm(tmp_path, circuit, theta, target, spec, config)
    path = tmp_path / ("ATTEMPTS.json" if change == "changed_attempts" else "INVOCATIONS.json")
    if change == "missing_invocations":
        path.unlink()
    else:
        ledger = cast("dict[str, Any]", pilot.read_record(path))
        ledger["attempts"][-1]["status" if change == "running" else "cpu_seconds"] = (
            "running" if change == "running" else 0
        )
        pilot.write_record(path, ledger)
    with pytest.raises(ValueError, match=r"unknown kill|attempt ledger changed|lacks its invocation"):
        experiment.train_arm(tmp_path, circuit, theta, target, spec, config)


def test_all12_scoring_charges_partial_failure_and_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Failed score components remain in the physical ledger and cannot be free retries."""
    circuit, theta, target = problem()
    original = experiment.channels.evaluate
    calls = 0

    def interrupted(candidate: ParameterizedCircuit, parameters: NDArray, desired: NDArray, noise: str) -> float:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InterruptedError
        return original(candidate, parameters, desired, noise)

    monkeypatch.setattr(experiment.channels, "evaluate", interrupted)
    with pytest.raises(InterruptedError):
        experiment.score_point(tmp_path, circuit, theta, target)
    failed = experiment.physical_scoring(tmp_path)
    assert failed["physical_attempted_channel_calls"] == 2
    assert failed["physical_completed_channel_calls"] == 1
    assert failed["complete_parameter_points"] == 0
    monkeypatch.setattr(experiment.channels, "evaluate", original)
    result = experiment.score_point(tmp_path, circuit, theta, target)
    assert set(result["fidelities"]) == set(experiment.channels.NOISE_IDS)
    assert result["physical_attempted_channel_calls"] == 14
    assert result["physical_completed_channel_calls"] == result["density_direction_passes"] == 13
    assert experiment.score_point(tmp_path, circuit, theta, target) == result
    path = next((tmp_path / "scores").glob("*/ATTEMPTS.json"))
    ledger = cast("dict[str, Any]", pilot.read_record(path))
    ledger["attempts"][-1]["status"] = "running"
    pilot.write_record(path, ledger)
    with pytest.raises(ValueError, match="unknown kill"):
        experiment.score_point(tmp_path, circuit, theta, target)


def test_q2_panel_all12_fanout_common_cpu_and_complete_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every method gets its final iterate and all fixed chronological CPU points."""
    source = source_fixture(monkeypatch)
    config = experiment.Config(channel_evaluations=22, checkpoint_every=2)
    result = experiment.run_case(tmp_path, source, config)
    assert result["status"] == "complete"
    assert len(result["arms"]) == 4
    for arm in result["arms"].values():
        assert len(arm["final_score"]["fidelities"]) == 12
        assert arm["final_score"]["theta"] == arm["training"]["theta"]
        assert arm["training"]["individual_channel_gradient_calls"] <= 22
    comparison = result["common_cpu_comparison"]
    assert len(comparison["points"]) == 5
    assert comparison["common_cpu_seconds"] == min(arm["training"]["cpu_seconds"] for arm in result["arms"].values())
    for point in comparison["points"]:
        for method, selected in point["selected"].items():
            assert selected["cpu_seconds"] <= point["cpu_budget_seconds"]
            assert point["scores"][method]["theta"] == selected["theta"]
            available = [
                value
                for value in result["arms"][method]["training"]["curve"]
                if value["cpu_seconds"] <= point["cpu_budget_seconds"]
            ]
            assert selected == available[-1]
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*.json")}
    assert experiment.run_case(tmp_path, source, config) == result
    assert before == {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*.json")}
    experiment.write_scoreboard(tmp_path, [result])
    with (tmp_path / "SCOREBOARD.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 * 2 * 12
    assert {row["panel"] for row in rows} == {"full_cap", "common_cpu"}
    path = next(tmp_path.glob("*/ATTEMPTS.json"))
    path.write_text("changed")
    with pytest.raises(ValueError, match="durable artifact changed"):
        experiment.run_case(tmp_path, source, config)


def test_failed_arm_keeps_physical_cost_and_invalidates_common_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A derivative failure retains its observed attempted work and paired placeholders."""
    source = source_fixture(monkeypatch)
    original = experiment.base.component_gradient
    calls = 0

    def fail_once(
        candidate: ParameterizedCircuit, parameters: NDArray, desired: NDArray, noise: str, *, fast: bool
    ) -> tuple[NDArray, float, float]:
        nonlocal calls
        calls += 1
        if calls == 2:
            msg = "injected derivative failure"
            raise ValueError(msg)
        return original(candidate, parameters, desired, noise, fast=fast)

    monkeypatch.setattr(experiment.base, "component_gradient", fail_once)
    result = experiment.run_case(tmp_path, source, experiment.Config(channel_evaluations=22))
    assert result["status"] == "complete_with_failed_arms"
    assert result["arms"]["mean2_lbfgs"]["status"] == "failed"
    assert result["arms"]["mean2_lbfgs"]["physical_work"]["physical_attempted_channel_calls"] == 2
    assert result["arms"]["mean2_lbfgs"]["physical_work"]["physical_completed_channel_calls"] == 1
    assert not result["common_cpu_comparison"]["points"]
    experiment.write_scoreboard(tmp_path, [result])
    with (tmp_path / "SCOREBOARD.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 96
    assert all(row["status"] == "incomplete" for row in rows if row["panel"] == "common_cpu")


def preparation_fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock scientific custody with correct q6 construction metadata and no evaluations."""
    gates = [ParameterizedGate("rx", (index % 6,), param_index=index) for index in range(78)]
    gates.extend(ParameterizedGate("rzz", (0, 1), param_index=index) for index in range(78, 108))
    circuit = ParameterizedCircuit(6, gates)
    starts = [
        {
            "target_id": target,
            "replica": 11,
            "candidate": {"circuit": pilot.circuit_record(circuit), "theta": [0.2] * 108},
        }
        for target in experiment.TARGETS
    ]
    pilot.write_record(root / "starts.json", {"starts": starts})
    monkeypatch.setattr(experiment.runner, "OUTPUT", root)
    monkeypatch.setattr(experiment.runner, "custody", lambda: {"benchmark_hashes": {"benchmark": "frozen"}})
    monkeypatch.setattr(experiment, "numerical_sources", lambda: {"source": "frozen"})
    monkeypatch.setattr(
        experiment, "load_target", lambda *_args: SimpleNamespace(state_vector_copy=lambda: np.ones(64))
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Preparation must not train or score a target.")

    monkeypatch.setattr(experiment, "train_arm", fail)
    monkeypatch.setattr(experiment, "score_point", fail)


@pytest.mark.parametrize("change", ["none", "source", "target", "config", "case_order"])
def test_prepare_only_freezes_settings_sources_targets_and_execution_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Engineering settings and changed inputs cannot pass the scientific verifier."""
    preparation_fixture(tmp_path, monkeypatch)
    output = tmp_path / "prepared"
    assert experiment.main(["--output", str(output), "--prepare-only"]) == 0
    assert [path.name for path in output.iterdir()] == ["PROTOCOL.json"]
    protocol = cast("dict[str, Any]", pilot.read_record(output / "PROTOCOL.json"))
    assert len(protocol["target_sha256"]) == 4
    assert experiment.freeze_protocol(output) == protocol
    if change == "source":
        monkeypatch.setattr(experiment, "numerical_sources", lambda: {"source": "changed"})
    elif change == "target":
        monkeypatch.setattr(
            experiment, "load_target", lambda *_args: SimpleNamespace(state_vector_copy=lambda: np.zeros(64))
        )
    elif change == "config":
        protocol["config"]["channel_evaluations"] = 22
    if change in {"source", "target", "config"}:
        with pytest.raises(ValueError, match="custody or frozen settings"):
            experiment.verify_protocol(protocol)
    elif change == "case_order":
        with pytest.raises(ValueError, match="construction/order panel"):
            experiment.execute_case({"protocol": protocol, "source": protocol["starts"][0], "order_index": 1})
    else:
        experiment.verify_protocol(protocol)


@pytest.mark.parametrize("workers", ["0", "5"])
def test_worker_count_is_bounded(tmp_path: Path, workers: str) -> None:
    """CLI preparation and execution both reject accidental CPU oversubscription."""
    with pytest.raises(SystemExit):
        experiment.main(["--output", str(tmp_path), "--prepare-only", "--workers", workers])
