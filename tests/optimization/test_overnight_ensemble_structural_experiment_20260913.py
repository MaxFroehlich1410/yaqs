# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Small exact composed-search checks; no benchmark targets or scientific jobs."""

from __future__ import annotations

import copy
import csv
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import (
    channels,
    ensemble_experiment,
)
from experiments.overnight_state_preparation_20260913_0108 import (
    ensemble_structural_experiment as experiment,
)
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path


def problem() -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Return q2 scalar native rotations compatible with all twelve final conditions."""
    gates = [
        ParameterizedGate(name, sites, index)
        for index, (name, sites) in enumerate([
            ("ry", (0,)),
            ("rx", (1,)),
            ("rxx", (0, 1)),
            ("rz", (0,)),
            ("ryy", (1, 0)),
            ("ry", (1,)),
            ("rzz", (0, 1)),
            ("rx", (0,)),
        ])
    ]
    rng = np.random.default_rng(619)
    target = rng.normal(size=4) + 1j * rng.normal(size=4)
    target /= np.linalg.norm(target)
    return ParameterizedCircuit(2, gates), rng.normal(size=len(gates)), target


def config(**updates: int) -> experiment.Config:
    """Return an engineering cap that funds symmetric search and final refinement."""
    values = {
        "channel_evaluations": 80,
        "warmup_evaluations": 3,
        "max_rounds": 2,
        "shortlist": 1,
        "repair_evaluations": 4,
        "protected_initial_gates": 2,
        "checkpoint_every": 2,
    }
    values.update(updates)
    return experiment.Config(**values)


def test_three_arms_budget_final_scores_compact_curves_and_byte_preserving_resume(tmp_path: Path) -> None:
    """Every paid branch is charged, and common-CPU selection uses only availability."""
    circuit, theta, target = problem()
    result = experiment.run_case(tmp_path, circuit, theta, target, config())
    assert result["status"] == "complete"
    assert list(result["arms"]) == list(experiment.METHODS)
    store = experiment.Store(tmp_path, target, config())
    for method, arm in result["arms"].items():
        assert arm["individual_channel_calls"] <= 80
        assert arm["individual_channel_calls"] == (
            arm["gradient_channel_calls"] + arm["ranking_channel_calls"] + arm["forward_channel_calls"]
        )
        assert arm["density_direction_passes"] == (
            2 * (arm["gradient_channel_calls"] + arm["ranking_channel_calls"]) + arm["forward_channel_calls"]
        )
        assert arm["physical_work"]["training"]["completed_channel_calls"] == arm["individual_channel_calls"]
        assert arm["physical_work"]["accounting_status"] == "known"
        assert arm["physical_work"]["training"]["cpu_seconds"] == pytest.approx(arm["cpu_seconds"])
        assert arm["physical_work"]["evaluation"]["completed_channel_calls"] == 24
        assert set(arm["fidelities"]) == set(channels.NOISE_IDS)
        final_circuit = store.restore(arm["circuit_hash"])
        for noise, value in arm["fidelities"].items():
            assert value == pytest.approx(channels.evaluate(final_circuit, np.asarray(arm["theta"]), target, noise))
        for point in arm["curve"]:
            assert set(point) == {"theta", "circuit_hash", "cpu_seconds", "individual_channel_calls"}
            assert len(point["theta"]) == store.restore(point["circuit_hash"]).num_params
        for round_record in arm["rounds"]:
            assert all(origin >= 2 for origin in round_record["ranking_origins"])
            assert len(round_record["branches"]) == 2
            choice = round_record["selected_branch"]
            if choice not in {None, 0}:
                assert (
                    round_record["branches"][choice]["fidelity"]
                    > max(round_record["incoming_fidelity"], round_record["branches"][0]["fidelity"])
                    + config().acceptance_tolerance
                )
        selected = result["common_cpu"]["methods"][method]["point"]
        available = [point for point in arm["curve"] if point["cpu_seconds"] <= result["common_cpu"]["cpu_seconds"]]
        assert selected == available[-1]
    assert result["arms"]["structural_mean2_adam"]["individual_channel_calls"] == 80
    assert len(list((tmp_path / "circuits").glob("*.json"))) <= 1 + 2 * config().max_rounds
    experiment.write_scoreboard(tmp_path, [result])
    rows = list(csv.DictReader((tmp_path / "SCOREBOARD.csv").read_text().splitlines()))
    assert len(rows) == 36
    assert [row["noise_id"] for row in rows[:12]] == list(channels.NOISE_IDS)
    assert all(row["ballarin_depth"] and row["logical_depth"] for row in rows)
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with patch.object(ensemble_experiment, "component_gradient", side_effect=AssertionError("cached result")):
        resumed = experiment.run_case(tmp_path, circuit, theta, target, config())
    assert resumed == result
    experiment.write_scoreboard(tmp_path, [resumed])
    after = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after


@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_stage_exact_resume_retains_full_optimizer_and_charges_discarded_work(tmp_path: Path, optimizer: str) -> None:
    """Interrupted partial mean calls and optimizer replay survive in physical ledgers."""
    circuit, theta, target = problem()
    reference = experiment.Store(tmp_path / "reference", target, config()).train("stage", circuit, theta, 14, optimizer)
    store = experiment.Store(tmp_path / "resume", target, config())
    calls = 0
    original = ensemble_experiment.component_gradient

    def interrupted(
        local_circuit: ParameterizedCircuit,
        local_theta: np.ndarray,
        local_target: np.ndarray,
        noise_id: str,
        *,
        fast: bool,
    ) -> tuple[np.ndarray, float, float]:
        nonlocal calls
        calls += 1
        if calls == 8:
            raise KeyboardInterrupt
        return original(local_circuit, local_theta, local_target, noise_id, fast=fast)

    with (
        patch.object(ensemble_experiment, "component_gradient", side_effect=interrupted),
        pytest.raises(KeyboardInterrupt),
    ):
        store.train("stage", circuit, theta, 14, optimizer)
    partial = cast("dict[str, Any]", pilot.read_record(tmp_path / "resume/stages/stage/PARTIAL.json"))["optimizer"]
    assert ("first_moment" if optimizer == "adam" else "s_history") in partial
    resumed = store.train("stage", circuit, theta, 14, optimizer)
    assert resumed["optimizer"] == reference["optimizer"]
    assert resumed["theta"] == reference["theta"]
    assert resumed["individual_channel_gradient_calls"] == reference["individual_channel_gradient_calls"]
    assert resumed["physical_attempted_channel_calls"] == resumed["physical_completed_channel_calls"] + 1
    assert resumed["physical_completed_channel_calls"] > reference["physical_completed_channel_calls"]
    attempts = cast("dict[str, Any]", pilot.read_record(tmp_path / "resume/stages/stage/ATTEMPTS.json"))["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["completed_channel_calls"] == 7
    assert attempts[0]["completed"] is False
    assert resumed["cpu_seconds"] == pytest.approx(sum(row["cpu_seconds"] for row in attempts))


def test_each_structural_branch_gets_fresh_curvature_or_moments(tmp_path: Path) -> None:
    """Retained, deleted and polish stages start without inherited optimizer memory."""
    circuit, theta, target = problem()
    store = experiment.Store(tmp_path, target, config(max_rounds=1))
    with patch.object(
        ensemble_experiment.optimizers, "minimize", wraps=ensemble_experiment.optimizers.minimize
    ) as lbfgs:
        experiment.train_method(store, circuit, theta, "structural_mean2_lbfgs")
    assert len(lbfgs.call_args_list) == 4
    assert all(call.kwargs["state"] is None for call in lbfgs.call_args_list)
    assert [call.kwargs["max_evaluations"] for call in lbfgs.call_args_list[:3]] == [3, 4, 4]
    with patch.object(ensemble_experiment, "minimize_adam", wraps=ensemble_experiment.minimize_adam) as adam:
        experiment.train_method(store, circuit, theta, "structural_mean2_adam")
    assert len(adam.call_args_list) == 4
    assert all(call.kwargs["state"] is None for call in adam.call_args_list)


def test_search_replay_reuses_completed_candidates_and_preserves_final_decisions(tmp_path: Path) -> None:
    """Interrupting after retained repair reconstructs the same complete structural search."""
    circuit, theta, target = problem()
    method = "structural_mean2_lbfgs"
    reference = experiment.train_method(
        experiment.Store(tmp_path / "reference", target, config()), circuit, theta, method
    )
    store = experiment.Store(tmp_path / "resume", target, config())
    original = store.train

    def interrupted(
        name: str, local_circuit: ParameterizedCircuit, start: np.ndarray, cap: int, optimizer: str
    ) -> dict[str, Any]:
        if "_delete" in name:
            raise KeyboardInterrupt
        return original(name, local_circuit, start, cap, optimizer)

    with patch.object(store, "train", side_effect=interrupted), pytest.raises(KeyboardInterrupt):
        experiment.train_method(store, circuit, theta, method)
    retained_path = tmp_path / "resume/stages/structural_mean2_lbfgs_round0_retain/TRAINING.json"
    retained = retained_path.read_bytes()
    resumed = experiment.train_method(store, circuit, theta, method)
    assert retained_path.read_bytes() == retained
    assert resumed["theta"] == reference["theta"]
    assert resumed["circuit_hash"] == reference["circuit_hash"]
    assert resumed["origins"] == reference["origins"]
    assert resumed["individual_channel_calls"] == reference["individual_channel_calls"]
    assert resumed["density_direction_passes"] == reference["density_direction_passes"]
    for actual, expected in zip(resumed["rounds"], reference["rounds"], strict=True):
        assert actual["selected_branch"] == expected["selected_branch"]
        assert actual["ranking_origins"] == expected["ranking_origins"]
        assert actual["ranking_scores"] == expected["ranking_scores"]
        assert [row["theta"] for row in actual["branches"]] == [row["theta"] for row in expected["branches"]]


def test_failed_arm_preserves_failed_component_costs_and_remaining_controls(tmp_path: Path) -> None:
    """A failed mean gradient is a visible failed arm with its first component charged."""
    circuit, theta, target = problem()
    calls = 0
    original = ensemble_experiment.component_gradient

    def failure(
        local_circuit: ParameterizedCircuit,
        local_theta: np.ndarray,
        local_target: np.ndarray,
        noise_id: str,
        *,
        fast: bool,
    ) -> tuple[np.ndarray, float, float]:
        nonlocal calls
        calls += 1
        if calls == 2:
            msg = "engineered component failure"
            raise FloatingPointError(msg)
        return original(local_circuit, local_theta, local_target, noise_id, fast=fast)

    with patch.object(ensemble_experiment, "component_gradient", side_effect=failure):
        result = experiment.run_case(tmp_path, circuit, theta, target, config(max_rounds=1))
    assert result["status"] == "complete_with_failed_arms"
    assert result["common_cpu"] is None
    failed = result["arms"]["structural_mean2_lbfgs"]
    assert failed["status"] == "failed"
    assert "engineered component failure" in failed["failure"]
    assert failed["physical_work"]["training"]["completed_channel_calls"] == 1
    assert failed["physical_work"]["training"]["attempted_channel_calls"] == 2
    assert failed["physical_work"]["training"]["cpu_seconds"] > 0
    assert result["physical_training_cpu_seconds"] == pytest.approx(
        sum(arm["physical_work"]["training"]["cpu_seconds"] for arm in result["arms"].values())
    )
    assert all(result["arms"][method]["status"] == "complete" for method in experiment.METHODS[1:])
    experiment.write_scoreboard(tmp_path, [result])
    rows = list(csv.DictReader((tmp_path / "SCOREBOARD.csv").read_text().splitlines()))
    assert len([row for row in rows if row["status"] == "failed"]) == 12
    assert all(not row["fidelity"] for row in rows if row["status"] == "failed")


def test_ranking_interruption_counts_each_profile_and_reuses_complete_cache(tmp_path: Path) -> None:
    """A restarted two-channel ranking charges its discarded completed first profile."""
    circuit, theta, target = problem()
    store = experiment.Store(tmp_path, target, config())
    original = experiment.ensemble_deletion.deletion_fidelities

    def interrupted(
        local_circuit: ParameterizedCircuit,
        local_theta: np.ndarray,
        local_target: np.ndarray,
        profiles: Sequence[str],
        *,
        progress: Callable[[str, str], None],
    ) -> experiment.ensemble_deletion.DeletionResult:
        def halt(noise: str, phase: str) -> None:
            progress(noise, phase)
            if noise == experiment.ensemble_deletion.MEAN2[1] and phase == "start":
                raise KeyboardInterrupt

        return original(local_circuit, local_theta, local_target, profiles, progress=halt)

    with (
        patch.object(experiment.ensemble_deletion, "deletion_fidelities", side_effect=interrupted),
        pytest.raises(KeyboardInterrupt),
    ):
        store.operation("ranking", circuit, theta, ranking=True)
    result = store.operation("ranking", circuit, theta, ranking=True)
    assert result["individual_channel_calls"] == 2
    assert result["physical_completed_channel_calls"] == 3
    assert result["physical_attempted_channel_calls"] == 4
    assert result["physical_completed_direction_passes"] == 6
    assert store.operation("ranking", circuit, theta, ranking=True) == result
    with pytest.raises(ValueError, match="Cached exact operation"):
        store.operation("ranking", circuit, theta + 0.1, ranking=True)


def test_unknown_killed_work_is_never_assigned_zero_cost(tmp_path: Path) -> None:
    """An unclosed invocation fails closed and requires an external resource ledger."""
    circuit, theta, target = problem()
    store = experiment.Store(tmp_path, target, config())
    store.train("fixed_mean2_lbfgs", circuit, theta, 3, "lbfgs")
    marker_path = tmp_path / "stages/fixed_mean2_lbfgs/INVOCATIONS.json"
    marker = cast("dict[str, Any]", pilot.read_record(marker_path))
    marker["attempts"].append({"status": "running"})
    pilot.write_record(marker_path, marker)
    with pytest.raises(ValueError, match="external resource ledger"):
        store.train("fixed_mean2_lbfgs", circuit, theta, 3, "lbfgs")
    assert (
        experiment.physical_work(tmp_path, "fixed_mean2_lbfgs")["accounting_status"]
        == "unknown_requires_external_ledger"
    )


@pytest.mark.parametrize(
    "updates",
    [{"channel_evaluations": 79}, {"channel_evaluations": 10}, {"shortlist": 0}, {"protected_initial_gates": -1}],
)
def test_invalid_budgets_rejected_before_simulation(updates: dict[str, Any]) -> None:
    """Invalid caps cannot silently bypass symmetric repair or final-refinement reserves."""
    with pytest.raises(ValueError, match="Invalid mean2 structural"):
        config(**updates)


@pytest.mark.usefixtures("synthetic_overnight_starts")
def test_prepare_only_authenticates_every_start_without_numerical_work(tmp_path: Path) -> None:
    """Prepare fixes one production setting and rejects even re-signed altered starts."""
    with patch.object(
        ensemble_experiment, "component_gradient", side_effect=AssertionError("prepare must not simulate")
    ):
        protocol = experiment.freeze_protocol(tmp_path)
        experiment.verify_protocol(protocol)
        assert experiment.freeze_protocol(tmp_path) == protocol
    assert protocol["config"]["channel_evaluations"] == 3000
    assert len(protocol["starts"]) == 4
    assert protocol["methods"] == list(experiment.METHODS)
    tampered = copy.deepcopy(protocol)
    tampered["starts"][0]["candidate"]["theta"][0] += 0.1
    with pytest.raises(ValueError, match="original starts changed"):
        experiment.verify_protocol(tampered)
    tampered = copy.deepcopy(protocol)
    tampered["noise_ids"] = ["depolarizing_1s_all"]
    with pytest.raises(ValueError, match="custody mismatch"):
        experiment.verify_protocol(tampered)
