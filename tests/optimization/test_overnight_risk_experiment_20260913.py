# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Small exact risk-screen endpoint, resume, custody and failure checks."""

from __future__ import annotations

import copy
import csv
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import channels, risk_experiment, risk_objective
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def problem() -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Return a q2 noncommuting problem compatible with all twelve endpoint profiles."""
    gates = [
        ParameterizedGate(name, sites, index)
        for index, (name, sites) in enumerate([
            ("ry", (0,)),
            ("rx", (1,)),
            ("rxx", (0, 1)),
            ("rz", (0,)),
            ("ryy", (1, 0)),
            ("ry", (1,)),
        ])
    ]
    rng = np.random.default_rng(382)
    target = rng.normal(size=4) + 1j * rng.normal(size=4)
    target /= np.linalg.norm(target)
    return ParameterizedCircuit(2, gates), rng.normal(size=6), target


def test_six_final_endpoints_call_caps_cpu_selection_and_byte_stable_resume(tmp_path: Path) -> None:
    """Six equal-opportunity arms retain every final condition and select common CPU blindly."""
    circuit, theta, target = problem()
    config = risk_experiment.Config(channel_evaluations=40, checkpoint_every=2)
    result = risk_experiment.run_case(tmp_path, circuit, theta, target, config)
    assert result["status"] == "complete"
    assert result["physical_accounting_status"] == "known"
    assert list(result["arms"]) == list(risk_experiment.METHODS)
    assert len(list((tmp_path / "circuits").glob("*.json"))) == 1
    for method, arm in result["arms"].items():
        assert arm["individual_channel_calls"] <= 40
        assert arm["density_direction_passes"] == 2 * arm["individual_channel_calls"]
        assert arm["individual_channel_calls"] == arm["physical_work"]["completed_channel_calls"]
        if method.endswith("adam"):
            assert arm["individual_channel_calls"] == 40
            assert arm["optimizer"]["iterations"] == 20
        assert arm["evaluation"]["individual_channel_calls"] == 12
        assert arm["evaluation"]["density_direction_passes"] == 12
        assert set(arm["fidelities"]) == set(channels.NOISE_IDS)
        for noise in channels.NOISE_IDS:
            assert arm["fidelities"][noise] == pytest.approx(
                channels.evaluate(circuit, np.asarray(arm["theta"]), target, noise), abs=2e-13
            )
        assert arm["metrics"]["mean11"] == pytest.approx(
            np.mean([value for noise, value in arm["fidelities"].items() if noise != "noiseless"])
        )
        selected = result["common_cpu"]["methods"][method]["point"]
        assert (
            selected
            == [point for point in arm["curve"] if point["cpu_seconds"] <= result["common_cpu"]["cpu_seconds"]][-1]
        )
        assert all(
            set(point) == {"theta", "cpu_seconds", "joint_evaluations", "individual_channel_calls"}
            for point in arm["curve"]
        )
    assert result["physical_training_cpu_seconds"] == pytest.approx(
        sum(arm["cpu_seconds"] for arm in result["arms"].values())
    )
    assert (
        result["case_invocation_cpu_seconds"]
        >= result["physical_training_cpu_seconds"] + result["physical_evaluation_cpu_seconds"]
    )
    risk_experiment.write_scoreboard(tmp_path, [result])
    rows = list(csv.DictReader((tmp_path / "SCOREBOARD.csv").read_text().splitlines()))
    assert len(rows) == 72
    assert [row["noise_id"] for row in rows[:12]] == list(channels.NOISE_IDS)
    assert all(row["ballarin_depth"] and row["logical_depth"] for row in rows)
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with patch.object(risk_objective, "loss_gradient", side_effect=AssertionError("cache must not propagate")):
        resumed = risk_experiment.run_case(tmp_path, circuit, theta, target, config)
    assert resumed == result
    risk_experiment.write_scoreboard(tmp_path, [resumed])
    assert {
        str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    } == before


@pytest.mark.parametrize("method", risk_experiment.METHODS)
def test_exact_optimizer_resume_for_every_risk_temperature_and_control(tmp_path: Path, method: str) -> None:
    """Temperature, full optimizer memory and interrupted component costs remain fixed."""
    circuit, theta, target = problem()
    config = risk_experiment.Config(channel_evaluations=28, checkpoint_every=2)
    reference = risk_experiment.train(tmp_path / "reference", circuit, theta, target, method, config)
    calls = 0
    original = risk_objective.ensemble_objective.loss_gradient

    def interrupted(
        local_circuit: ParameterizedCircuit, local_theta: np.ndarray, local_target: np.ndarray, noise: str
    ) -> tuple[np.ndarray, float, float]:
        nonlocal calls
        calls += 1
        if calls == 8:
            raise KeyboardInterrupt
        return original(local_circuit, local_theta, local_target, noise)

    with (
        patch.object(risk_objective.ensemble_objective, "loss_gradient", side_effect=interrupted),
        pytest.raises(KeyboardInterrupt),
    ):
        risk_experiment.train(tmp_path / "resume", circuit, theta, target, method, config)
    checkpoint = cast("dict[str, Any]", pilot.read_record(tmp_path / "resume/PARTIAL.json"))
    assert ("first_moment" if method.endswith("adam") else "s_history") in checkpoint["optimizer"]
    resumed = risk_experiment.train(tmp_path / "resume", circuit, theta, target, method, config)
    assert resumed["optimizer"] == reference["optimizer"]
    assert resumed["theta"] == reference["theta"]
    assert resumed["last_oracle"] == reference["last_oracle"]
    assert resumed["individual_channel_calls"] == reference["individual_channel_calls"]
    physical = resumed["physical_work"]
    assert physical["completed_channel_calls"] > resumed["individual_channel_calls"]
    assert physical["attempted_channel_calls"] == physical["completed_channel_calls"] + 1
    attempts = cast("dict[str, Any]", pilot.read_record(tmp_path / "resume/ATTEMPTS.json"))["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["completed_channel_calls"] == 7
    assert attempts[0]["status"] == "interrupted"
    assert physical["cpu_seconds"] == pytest.approx(sum(attempt["cpu_seconds"] for attempt in attempts))


def test_failed_component_remains_visible_and_paid_while_other_arms_complete(tmp_path: Path) -> None:
    """The first failed risk arm cannot disappear from the panel or its work ledger."""
    circuit, theta, target = problem()
    calls = 0
    original = risk_objective.ensemble_objective.loss_gradient

    def failed(
        local_circuit: ParameterizedCircuit, local_theta: np.ndarray, local_target: np.ndarray, noise: str
    ) -> tuple[np.ndarray, float, float]:
        nonlocal calls
        calls += 1
        if calls == 2:
            msg = "engineered risk component failure"
            raise FloatingPointError(msg)
        return original(local_circuit, local_theta, local_target, noise)

    with patch.object(risk_objective.ensemble_objective, "loss_gradient", side_effect=failed):
        result = risk_experiment.run_case(
            tmp_path, circuit, theta, target, risk_experiment.Config(channel_evaluations=16)
        )
    assert result["status"] == "complete_with_failures"
    assert result["common_cpu"] is None
    first = result["arms"][risk_experiment.METHODS[0]]
    assert first["status"] == "failed"
    assert first["physical_work"]["completed_channel_calls"] == 1
    assert first["physical_work"]["attempted_channel_calls"] == 2
    assert first["physical_work"]["cpu_seconds"] > 0
    assert all(result["arms"][method]["status"] == "complete" for method in risk_experiment.METHODS[1:])
    assert result["physical_training_cpu_seconds"] == pytest.approx(
        sum(arm["physical_work"]["cpu_seconds"] for arm in result["arms"].values())
    )
    risk_experiment.write_scoreboard(tmp_path, [result])
    rows = list(csv.DictReader((tmp_path / "SCOREBOARD.csv").read_text().splitlines()))
    assert len([row for row in rows if row["status"] == "failed"]) == 12
    assert all(not row["fidelity"] for row in rows if row["status"] == "failed")


def test_case_interruption_reuses_completed_arm_and_charges_invocation(tmp_path: Path) -> None:
    """Whole-case replay preserves cached scientific bytes and retains both invocation costs."""
    circuit, theta, target = problem()
    config = risk_experiment.Config(channel_evaluations=16, checkpoint_every=2)
    original = risk_experiment.train

    def interrupted(
        root: Path,
        local_circuit: ParameterizedCircuit,
        local_theta: np.ndarray,
        local_target: np.ndarray,
        method: str,
        local_config: risk_experiment.Config,
    ) -> dict[str, Any]:
        if method == risk_experiment.METHODS[1]:
            raise KeyboardInterrupt
        return original(root, local_circuit, local_theta, local_target, method, local_config)

    with patch.object(risk_experiment, "train", side_effect=interrupted), pytest.raises(KeyboardInterrupt):
        risk_experiment.run_case(tmp_path, circuit, theta, target, config)
    complete = tmp_path / "training" / risk_experiment.METHODS[0] / "TRAINING.json"
    before = complete.read_bytes()
    result = risk_experiment.run_case(tmp_path, circuit, theta, target, config)
    assert result["status"] == "complete"
    assert complete.read_bytes() == before
    attempts = cast("dict[str, Any]", pilot.read_record(tmp_path / "CASE_ATTEMPTS.json"))["attempts"]
    assert [row["status"] for row in attempts] == ["interrupted", "complete"]
    assert result["case_invocation_cpu_seconds"] == pytest.approx(sum(row["cpu_seconds"] for row in attempts))


def test_tail_metrics_use_all_eleven_and_reject_missing_conditions() -> None:
    """Noiseless and the selected training mean cannot replace all-condition tail metrics."""
    fidelities = {noise: index / 12 for index, noise in enumerate(channels.NOISE_IDS)}
    result = risk_experiment.metrics(fidelities)
    assert result["mean11"] == pytest.approx(0.5)
    assert result["worst11"] == pytest.approx(1 / 12)
    assert result["lowest3_mean"] == pytest.approx(2 / 12)
    assert result["noiseless"] == 0
    assert result["noise_families"]["N1_angle"] == pytest.approx(1 / 12)
    assert result["worst_family"] == "N1_angle"
    del fidelities["ballarin_coupled"]
    with pytest.raises(ValueError, match="all twelve"):
        risk_experiment.metrics(fidelities)


@pytest.mark.usefixtures("isolated_overnight_custody")
def test_prepare_only_freezes_exact_six_method_panel_and_original_sources(tmp_path: Path) -> None:
    """Protocol creation does no numerical work and rejects re-signed temperature/start changes."""
    with patch.object(risk_objective, "loss_gradient", side_effect=AssertionError("no preparation simulation")):
        protocol = risk_experiment.freeze_protocol(tmp_path)
        risk_experiment.verify_protocol(protocol)
        assert risk_experiment.freeze_protocol(tmp_path) == protocol
    assert len(protocol["methods"]) == 6
    assert protocol["config"]["channel_evaluations"] == 1100
    assert len(protocol["starts"]) == 4
    changed = copy.deepcopy(protocol)
    changed["methods"]["risk020_lbfgs"]["temperature"] = 0.03
    with pytest.raises(ValueError, match="custody mismatch"):
        risk_experiment.verify_protocol(changed)
    changed = copy.deepcopy(protocol)
    changed["starts"][0]["candidate"]["theta"][0] += 0.1
    with pytest.raises(ValueError, match="custody mismatch"):
        risk_experiment.verify_protocol(changed)


def test_unknown_killed_training_work_fails_closed(tmp_path: Path) -> None:
    """An unclosed attempt cannot become a seemingly complete zero-cost cached run."""
    circuit, theta, target = problem()
    config = risk_experiment.Config(channel_evaluations=8)
    risk_experiment.train(tmp_path, circuit, theta, target, "risk020_lbfgs", config)
    path = tmp_path / "ATTEMPTS.json"
    attempts = cast("dict[str, Any]", pilot.read_record(path))
    attempts["attempts"].append({"status": "running"})
    pilot.write_record(path, attempts)
    with pytest.raises(ValueError, match="external resource ledger"):
        risk_experiment.train(tmp_path, circuit, theta, target, "risk020_lbfgs", config)
    assert risk_experiment.attempt_totals(attempts)["accounting_status"] == "unknown_requires_external_ledger"


@pytest.mark.parametrize("cap", [0, 3, -2, True])
def test_invalid_budgets_are_rejected(cap: int) -> None:
    """All methods must fit complete two-channel gradient calls."""
    with pytest.raises(ValueError, match="positive even"):
        risk_experiment.Config(channel_evaluations=cap)
