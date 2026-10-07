# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Paired map, persistent-moment, exact-resume and ledger tests for SVRG Adam."""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider as Calibration
from benchmarks.state_preparation.exact_density import density_loss_gradient
from experiments import state_preparation_exact_adam as adam
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import svrg_experiment as experiment
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def small_case() -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Return a q2 complex-target problem with shared and scaled parameters."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), 0, angle_scale=-1.3, angle_offset=0.2),
            ParameterizedGate("rxx", (1, 0), 1),
            ParameterizedGate("rz", (1,), 0),
            ParameterizedGate("ry", (1,), 2),
        ],
    )
    return circuit, np.array([0.4, -0.7, 0.2]), np.array([1, 2j, -3, 4j]) / np.sqrt(30)


def numerical(value: object) -> object:
    """Return values without timing and the timing-dependent anchor checksum."""
    if isinstance(value, dict):
        mapping = cast("dict[str, Any]", value)
        return {key: numerical(item) for key, item in mapping.items() if "seconds" not in key and key != "checksum"}
    if isinstance(value, list):
        return [numerical(item) for item in value]
    return value


def small_config() -> experiment.Config:
    """Return the refresh-five algorithm with bounded engineering work."""
    return experiment.Config(updates=7, correction_paths=3, checkpoint_every=2)


@pytest.mark.parametrize("method", experiment.METHODS)
@pytest.mark.parametrize("boundary", [3, 5])
def test_exact_resume_keeps_anchor_moments_counters_and_maps(method: str, boundary: int) -> None:
    """Pause both inside and exactly at an anchor epoch, then reproduce all numerics."""
    circuit, theta, target = small_case()
    config = small_config()
    full = experiment.train(circuit, theta, target, config, method, target_id="toy")
    part = experiment.train(circuit, theta, target, config, method, target_id="toy", stop_after=boundary)
    saved = json.loads(json.dumps(part))
    resumed = experiment.train(circuit, theta, target, config, method, target_id="toy", state=saved)
    assert numerical(resumed) == numerical(full)
    assert saved == part
    assert resumed["iterations"] == config.updates
    assert resumed["status"] == "complete"
    assert resumed["curve"][-1]["theta"] == resumed["theta"]
    if method == "cv_adam":
        assert [row["anchor_age"] for row in resumed["audit"]] == [0, 1, 2, 3, 4, 0, 1]
        assert resumed["anchor"]["created_before_update"] == 6
        assert resumed["audit"][0]["anchor_distance"] == resumed["audit"][5]["anchor_distance"] == 0


def test_equal_direction_control_shares_physical_prefix_and_distinct_weights() -> None:
    """Both stochastic arms pay equal path directions, with fully additional anchors."""
    circuit, theta, target = small_case()
    config = small_config()
    cv = experiment.train(circuit, theta, target, config, "cv_adam", target_id="toy")
    weighted = experiment.train(circuit, theta, target, config, "weighted_adam", target_id="toy")
    assert cv["trajectory_paths"] == weighted["trajectory_paths"] == 42
    assert cv["trajectory_direction_passes"] == weighted["trajectory_direction_passes"] == 84
    assert cv["trajectory_gradient_calls"] == 14
    assert weighted["trajectory_gradient_calls"] == 7
    assert cv["exact_anchor_calls"] == cv["exact_gradient_calls"] == 2
    assert cv["density_gradient_direction_passes"] == 4
    assert weighted["exact_gradient_calls"] == 0
    assert cv["diagnostic_forward_calls"] == weighted["diagnostic_forward_calls"] == 5
    assert cv["diagnostic_direction_passes"] == 5
    for correction, ordinary in zip(cv["audit"], weighted["audit"], strict=True):
        assert correction["seed"] == ordinary["seed"]
        assert correction["paired_prefix_checksum"] == ordinary["paired_prefix_checksum"]
        assert correction["map_checksum"] != ordinary["map_checksum"]
        assert correction["weights"][0] == ordinary["weights"][0]
        assert correction["weights"][1] / ordinary["weights"][1] == pytest.approx(5 / 2)
    assert len({row["seed"] for row in cv["audit"]}) == config.updates


def test_first_correction_is_exact_and_adam_uses_persistent_moments() -> None:
    """Known cancellation yields the exact first update; refresh does not reset Adam."""
    circuit, theta, target = small_case()
    config = small_config()
    cv = experiment.train(circuit, theta, target, config, "cv_adam", target_id="toy", stop_after=1)
    gradient, _, _ = density_loss_gradient(circuit, theta, target, Calibration())
    expected = adam.adam_step(
        theta,
        gradient,
        np.zeros_like(theta),
        np.zeros_like(theta),
        1,
        adam.AdamConfig(updates=7, learning_rate=0.02),
    )
    for key, value in zip(("theta", "first", "second"), expected, strict=True):
        np.testing.assert_array_equal(cv[key], value)
    before = experiment.train(circuit, theta, target, config, "cv_adam", target_id="toy", stop_after=5)
    after = experiment.train(circuit, theta, target, config, "cv_adam", target_id="toy", state=before, stop_after=6)
    current = np.asarray(before["theta"])
    gradient, _, _ = density_loss_gradient(circuit, current, target, Calibration())
    expected = adam.adam_step(
        current,
        gradient,
        np.asarray(before["first"]),
        np.asarray(before["second"]),
        6,
        adam.AdamConfig(updates=7, learning_rate=0.02),
    )
    for key, value in zip(("theta", "first", "second"), expected, strict=True):
        np.testing.assert_array_equal(after[key], value)
    assert after["trajectory_paths"] == 36


def test_checkpoint_callback_is_complete_private_state_and_final_resume_is_read_only() -> None:
    """The callback cannot alter training state; completed resume calls no simulator."""
    circuit, theta, target = small_case()
    config = small_config()
    observed = []

    def callback(state: dict[str, Any]) -> None:
        observed.append(numerical(state))
        state["first"][0] = 1e9
        state["anchor"]["theta"][0] = 1e9

    full = experiment.train(circuit, theta, target, config, "cv_adam", target_id="toy", checkpoint=callback)
    assert len(observed) == 7
    assert observed[-1] == numerical(full)
    attempt = experiment.new_attempt()
    resumed = experiment.train(circuit, theta, target, config, "cv_adam", target_id="toy", state=full, attempt=attempt)
    assert resumed == full
    assert attempt == experiment.new_attempt()


@pytest.mark.parametrize("corruption", ["anchor", "counter", "method", "target", "seed"])
def test_changed_saved_state_and_problem_are_rejected(corruption: str) -> None:
    """Check anchor integrity, logical work and full circuit/target/stream identity."""
    circuit, theta, target = small_case()
    config = small_config()
    part = experiment.train(circuit, theta, target, config, "cv_adam", target_id="toy", stop_after=3)
    method, target_id = "cv_adam", "toy"
    if corruption == "anchor":
        part["anchor"]["estimate"]["gradient"][0] += 1
    elif corruption == "counter":
        part["trajectory_paths"] += 1
    elif corruption == "method":
        method = "weighted_adam"
    elif corruption == "target":
        target = np.roll(target, 1)
    else:
        target_id = "other"
    with pytest.raises(ValueError, match=r"Invalid saved|signature changed"):
        experiment.train(circuit, theta, target, config, method, target_id=target_id, state=part)


def test_prepare_only_freezes_exactly_one_setting_and_authenticates_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparing performs no training or evaluation, and cannot overwrite settings."""
    circuit = ParameterizedCircuit(6, [ParameterizedGate("ry", (0,), index) for index in range(108)])
    starts: list[dict[str, Any]] = [
        {
            "target_id": target_id,
            "replica": 11,
            "candidate": {"circuit": pilot.circuit_record(circuit), "theta": [0.0] * 108},
        }
        for target_id in experiment.TARGETS
    ]
    starts_path = tmp_path / "starts.json"
    pilot.write_record(starts_path, {"starts": starts})
    monkeypatch.setattr(experiment.runner, "custody", lambda: {"benchmark_hashes": {"fixed": "hash"}})
    monkeypatch.setattr(experiment, "source_hashes", lambda: {"source": "hash"})

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Preparing must not evaluate or optimize")

    monkeypatch.setattr(experiment, "train", forbidden)
    monkeypatch.setattr(experiment.channels, "evaluate", forbidden)
    protocol = experiment.freeze_protocol(tmp_path / "prepared", starts_path)
    assert protocol["config"] == asdict(experiment.Config())
    assert protocol["refresh_before_updates"] == list(range(1, 151, 5))
    assert protocol["methods"] == list(experiment.METHODS)
    assert len(protocol["starts"]) == 4
    assert len(protocol["training_seeds"][experiment.TARGETS[0]]) == 150
    before = (tmp_path / "prepared/PROTOCOL.json").read_bytes()
    assert experiment.freeze_protocol(tmp_path / "prepared", starts_path) == protocol
    assert (tmp_path / "prepared/PROTOCOL.json").read_bytes() == before
    experiment.verify_protocol(protocol)
    starts[0]["candidate"]["theta"][0] += 0.1
    pilot.write_record(starts_path, {"starts": starts})
    with pytest.raises(ValueError, match="custody mismatch"):
        experiment.verify_protocol(protocol)


def toy_job(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Return a small numerical wrapper test without production benchmark access."""
    circuit, theta, target = small_case()
    monkeypatch.setattr(experiment, "verify_protocol", lambda _protocol: None)
    monkeypatch.setattr(experiment, "load_target", lambda *_args: SimpleNamespace(state_vector_copy=target.copy))
    return {
        "protocol": {"config": asdict(small_config())},
        "source": {
            "target_id": "toy",
            "replica": 11,
            "candidate": {"circuit": pilot.circuit_record(circuit), "theta": theta.tolist()},
        },
        "method": "cv_adam",
        "root": str(root),
    }


def test_job_complete_resume_preserves_all_files_and_csv_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Actual q2 twelve-channel scoring is independent and completed resume is inert."""
    job = toy_job(tmp_path, monkeypatch)
    result = experiment.execute_job(job)
    assert result["status"] == "complete"
    assert set(result["fidelities"]) == set(experiment.channels.NOISE_IDS)
    assert result["final_forward_evaluations"] == 12
    assert result["physical_attempt_totals"]["evaluation_forward_calls"] == 12
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    assert experiment.execute_job(job) == result
    assert before == {path: path.read_bytes() for path in tmp_path.iterdir()}
    experiment.write_scoreboard(tmp_path, [result])
    csv = (tmp_path / "SCOREBOARD.csv").read_bytes()
    experiment.write_scoreboard(tmp_path, [json.loads(json.dumps(result, sort_keys=True))])
    assert (tmp_path / "SCOREBOARD.csv").read_bytes() == csv


def test_interrupted_unsaved_updates_are_charged_and_resume_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A discarded update is replayed from the durable anchor without disappearing."""
    job = toy_job(tmp_path / "resumed", monkeypatch)
    full_job = {**job, "root": str(tmp_path / "full")}
    full = experiment.execute_job(full_job)
    original_step = experiment.adam.adam_step

    def interrupt(
        theta: np.ndarray,
        gradient: np.ndarray,
        first: np.ndarray,
        second: np.ndarray,
        step: int,
        config: adam.AdamConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if step == 4:
            raise KeyboardInterrupt
        return original_step(theta, gradient, first, second, step, config)

    monkeypatch.setattr(experiment.adam, "adam_step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        experiment.execute_job(job)
    prior = cast("dict[str, Any]", pilot.read_record(tmp_path / "resumed/PARTIAL.json"))
    assert prior["state"]["iterations"] == 2
    monkeypatch.setattr(experiment.adam, "adam_step", original_step)
    resumed = experiment.execute_job(job)
    assert numerical(resumed["state"]) == numerical(full["state"])
    assert resumed["fidelities"] == full["fidelities"]
    assert resumed["physical_attempt_totals"]["trajectory_paths"] == 54
    assert resumed["state"]["trajectory_paths"] == 42
    attempts = cast("list[dict[str, Any]]", pilot.read_record(tmp_path / "resumed/ATTEMPTS.json")["attempts"])
    assert [row["status"] for row in attempts] == ["interrupted", "complete"]
    assert attempts[1]["resumed_iteration"] == 2


def test_failures_persist_with_attempted_numerical_costs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Numerical failures remain failed cells and are never silently restarted."""
    job = toy_job(tmp_path, monkeypatch)

    def failure(*_args: object, **_kwargs: object) -> None:
        msg = "Injected nonfinite gradient"
        raise FloatingPointError(msg)

    monkeypatch.setattr(experiment.variance, "build_snapshot", failure)
    result = experiment.execute_job(job)
    assert result["status"] == "failed"
    assert result["physical_attempt_totals"]["exact_anchor_calls"] == 1
    assert result["physical_attempt_totals"]["diagnostic_forward_calls"] == 1
    assert experiment.execute_job(job) == result
    experiment.write_scoreboard(tmp_path, [result])
    assert len((tmp_path / "SCOREBOARD.csv").read_text().splitlines()) == 13


def test_failed_final_evaluation_is_charged_and_never_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Training success cannot hide a failed final-channel propagation."""
    job = toy_job(tmp_path, monkeypatch)

    def failure(*_args: object, **_kwargs: object) -> float:
        msg = "Injected final evaluation failure"
        raise FloatingPointError(msg)

    monkeypatch.setattr(experiment.channels, "evaluate", failure)
    result = experiment.execute_job(job)
    assert result["status"] == "failed"
    assert result["physical_attempt_totals"]["trajectory_paths"] == 42
    assert result["physical_attempt_totals"]["evaluation_forward_calls"] == 1
    assert result["physical_attempt_totals"]["evaluation_cpu_seconds"] >= 0


def test_completed_result_authenticates_its_physical_attempt_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rewritten resource record cannot silently alter a completed result's cost."""
    job = toy_job(tmp_path, monkeypatch)
    experiment.execute_job(job)
    path = tmp_path / "ATTEMPTS.json"
    ledger = cast("dict[str, Any]", pilot.read_record(path))
    ledger["attempts"][0]["counters"]["trajectory_paths"] = 0
    pilot.write_record(path, ledger)
    with pytest.raises(ValueError, match="physical-attempt ledger changed"):
        experiment.execute_job(job)


def test_unknown_killed_attempt_requires_its_missing_resource_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unfinished process invocation cannot be treated as zero physical work."""
    job = toy_job(tmp_path, monkeypatch)
    pilot.write_record(tmp_path / "ATTEMPTS.json", {"attempts": [{"status": "running"}]})
    with pytest.raises(ValueError, match="unaccounted killed"):
        experiment.execute_job(job)


@pytest.mark.parametrize(
    "kwargs", [{"updates": 0}, {"correction_paths": 1}, {"refresh_every": True}, {"learning_rate": np.nan}]
)
def test_invalid_configuration(kwargs: dict[str, Any]) -> None:
    """Reject invalid budgets and nonfinite step sizes before any simulation."""
    with pytest.raises(ValueError, match="Invalid SVRG"):
        experiment.Config(**kwargs)
