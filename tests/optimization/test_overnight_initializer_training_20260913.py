# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact small-system optimizer resume and compiled deployable checkpoint clocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import initializer_training as training
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def source() -> dict[str, Any]:
    """Return a q2 physical fixture using independent nonzero scalar parameters."""
    circuit = ParameterizedCircuit(
        2, [ParameterizedGate("ry", (0,), 0), ParameterizedGate("rx", (1,), 1), ParameterizedGate("rxx", (0, 1), 2)]
    )
    return {"circuit": pilot.circuit_record(circuit), "theta": [0.1, -0.2, 0.3], "initializer_cpu_seconds": 7.0}


def target() -> np.ndarray:
    """Return a nontrivial complex q2 target for both exact noise components."""
    vector = np.array([1.0, 0.7j, -0.3, 0.2 - 0.5j])
    return vector / np.linalg.norm(vector)


@pytest.mark.parametrize("method", ["adam", "lbfgs"])
def test_full_optimizer_resume_and_every_deployed_checkpoint_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Interrupting a compiled checkpoint preserves theta, moments and secant histories exactly."""
    config = training.Config(joint_cap=8, checkpoint_every=1, history=3)
    full = training.train(tmp_path / "full", source(), target(), method, config)
    original_write = pilot.write_record

    def interrupt(path: Path, record: dict[str, Any]) -> None:
        original_write(path, record)
        if path.name == "STATE.json" and record["optimizer"]["iterations"] == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(pilot, "write_record", interrupt)
    with pytest.raises(KeyboardInterrupt):
        training.train(tmp_path / "resume", source(), target(), method, config)
    checkpoint = cast("dict[str, Any]", pilot.read_record(tmp_path / "resume/STATE.json"))
    assert checkpoint["optimizer"]["iterations"] == 1
    monkeypatch.setattr(pilot, "write_record", original_write)
    resumed = training.train(tmp_path / "resume", source(), target(), method, config)
    assert resumed["optimizer"] == full["optimizer"]
    assert resumed["individual_channel_gradient_calls"] <= 16
    assert resumed["physical_completed_channel_calls"] >= resumed["individual_channel_gradient_calls"]
    assert resumed["refinement_cpu_seconds"] == pytest.approx(
        resumed["training_cpu_seconds"] + resumed["deployment_cpu_seconds"]
    )
    assert resumed["deployment_cpu_seconds"] > 0
    for point in resumed["curve"]:
        assert "theta" not in point
        assert "circuit" not in point
        record = cast(
            "dict[str, Any]", pilot.read_record(tmp_path / "resume/deployments" / f"{point['deployment']}.json")
        )
        assert record["signature"] == point["deployment"]
        assert record["endpoint"].get("state_audit_error", 0) < 1e-10
    assert all(
        b["refinement_cpu_seconds"] >= a["refinement_cpu_seconds"]
        for a, b in zip(resumed["curve"], resumed["curve"][1:], strict=False)
    )
    before = {p: p.read_bytes() for p in (tmp_path / "resume").rglob("*.json")}
    assert pilot.digest(training.train(tmp_path / "resume", source(), target(), method, config)) == pilot.digest(
        resumed
    )
    assert before == {p: p.read_bytes() for p in (tmp_path / "resume").rglob("*.json")}


def test_constituent_budget_and_sparse_final_checkpoint_are_exact(tmp_path: Path) -> None:
    """Adam uses the final budgeted gradient and deploys its final post-update theta."""
    result = training.train(tmp_path, source(), target(), "adam", training.Config(joint_cap=5, checkpoint_every=2))
    assert result["joint_calls"] == 5
    assert result["individual_channel_gradient_calls"] == result["physical_completed_channel_calls"] == 10
    assert [point["iterations"] for point in result["curve"]] == [0, 2, 4, 5]
    deployed = cast(
        "dict[str, Any]", pilot.read_record(tmp_path / "deployments" / f"{result['final_deployment']}.json")
    )["endpoint"]
    actual = training.initializer_policy.state_vector(deployed["circuit"], np.asarray(deployed["theta"]))
    original = training.initializer_policy.state_vector(source()["circuit"], np.asarray(result["optimizer"]["theta"]))
    assert abs(np.vdot(actual, original)) ** 2 == pytest.approx(1, abs=1e-12)


def test_changed_training_budget_or_initial_source_cannot_resume(tmp_path: Path) -> None:
    """A new budget or architecture requires its own fixed scientific job."""
    config = training.Config(joint_cap=2)
    training.train(tmp_path, source(), target(), "adam", config)
    with pytest.raises(ValueError, match="identity changed"):
        training.train(tmp_path, source(), target(), "adam", training.Config(joint_cap=3))
    changed = source()
    changed["theta"][0] += 0.01
    with pytest.raises(ValueError, match="identity changed"):
        training.train(tmp_path, changed, target(), "adam", config)


def test_nonfinite_component_failure_retains_physical_attempt_and_no_final_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed exact component is charged without presenting a completed optimizer endpoint."""

    def fail(*_a: object, **_k: object) -> tuple[np.ndarray, float, float]:
        msg = "injected component failure"
        raise FloatingPointError(msg)

    monkeypatch.setattr(training.ensemble_experiment, "component_gradient", fail)
    with pytest.raises(FloatingPointError):
        training.train(tmp_path, source(), target(), "adam", training.Config(joint_cap=2))
    attempts = cast("dict[str, Any]", pilot.read_record(tmp_path / "ATTEMPTS.json"))["attempts"]
    assert attempts[0]["attempted_channel_calls"] == 1
    assert attempts[0]["completed_channel_calls"] == 0
    assert attempts[0]["status"] == "failed"
    assert attempts[0]["training_cpu_seconds"] > 0
    assert not (tmp_path / "TRAINING.json").exists()
