# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Prepare-only custody, physical accounting and exact q2 synthesis-arm resume."""

from __future__ import annotations

import copy
import csv
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import synthesis_experiment as experiment
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def small_target() -> np.ndarray:
    """Return a deterministic complex two-qubit engineering target."""
    target = np.array([0.1 + 0.3j, 0.7 - 0.1j, -0.3 + 0.4j, 0.2 + 0.8j])
    return target / np.linalg.norm(target)


@pytest.fixture
def small_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use only q2 numerical work and fake custody in this temporary test scope."""
    monkeypatch.setattr(experiment.runner, "custody", lambda: {"benchmark_hashes": {"immutable": "fixed"}})
    monkeypatch.setattr(experiment.runner, "source_hashes", lambda: {"numerical": "fixed"})

    def loader(*_args: object) -> SimpleNamespace:
        return SimpleNamespace(state_vector_copy=small_target)

    monkeypatch.setattr(experiment.runner, "load_target", loader)
    monkeypatch.setattr(experiment, "load_target", loader)


def arm_job(root: Path, method: str) -> dict[str, Any]:
    """Return physical q2 synthesis with deliberately displaced parameters."""
    candidate = experiment.synthesis.synthesize(small_target())
    config = {"method": method, "iterations": 4, "checkpoint_every": 1, "rate": 0.02}
    if method == "lbfgs":
        config.update({"evaluations": 30, "history": 3})
    return {
        "phase": "development",
        "label": f"test_{method}",
        "target_id": experiment.TARGETS[0],
        "replica": 11,
        "circuit": candidate["circuit"],
        "initial": (np.asarray(candidate["theta"]) + 0.03).tolist(),
        "config": config,
        "seed": 117,
        "path": str(root / "arm.json"),
        "source_hash": pilot.digest(experiment.runner.source_hashes()),
    }


@pytest.mark.usefixtures("small_runtime")
@pytest.mark.parametrize("method", ["adam", "lbfgs"])
def test_full_optimizer_interruption_resume_and_idempotent_cached_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Resume moments/secant histories and count interrupted attempts physically."""
    full_job = arm_job(tmp_path / "full", method)
    full = experiment.run_arm(full_job)
    partial_job = copy.deepcopy(full_job)
    partial_job["path"] = str(tmp_path / "resume/arm.json")
    original_write = pilot.write_record

    def interrupted_write(path: Path, record: dict[str, Any]) -> None:
        original_write(path, record)
        if path.name == "arm.checkpoint.json" and record["state"]["iterations"] == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(pilot, "write_record", interrupted_write)
    with pytest.raises(KeyboardInterrupt):
        experiment.run_arm(partial_job)
    saved = cast("dict[str, Any]", pilot.read_record(tmp_path / "resume/arm.checkpoint.json")["state"])
    assert saved["iterations"] == 1
    if method == "lbfgs":
        assert saved["optimizer"]["s_history"]
    else:
        assert np.any(np.asarray(saved["first"]) != 0)
        assert np.any(np.asarray(saved["second"]) != 0)
    monkeypatch.setattr(pilot, "write_record", original_write)
    resumed = experiment.run_arm(partial_job)
    for key in ("theta", "first", "second", "counts", "iterations", "exact_training_evaluations", "direction_passes"):
        assert resumed["state"][key] == full["state"][key]
    if method == "lbfgs":
        assert resumed["state"]["optimizer"] == full["state"]["optimizer"]
    assert resumed["fidelities"] == full["fidelities"]
    assert set(resumed["fidelities"]) == set(experiment.channels.NOISE_IDS)
    attempts = cast("list[dict[str, Any]]", pilot.read_record(tmp_path / "resume/arm.attempts.json")["attempts"])
    assert len(attempts) == 2
    assert all(a["cpu_seconds"] > 0 for a in attempts)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert pilot.digest(experiment.run_arm(partial_job)) == pilot.digest(resumed)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.usefixtures("small_runtime")
def test_numeric_failure_retains_checkpoint_and_every_profile_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unfavorable numerical outcome remains an explicit terminal failed arm."""
    job = arm_job(tmp_path, "adam")
    original_train = cast("Callable[..., dict[str, Any]]", experiment.runner.train)

    def fail_after_checkpoint(*args: object, **kwargs: object) -> dict[str, Any]:
        original_train(*args, **kwargs)
        msg = "injected final failure"
        raise FloatingPointError(msg)

    monkeypatch.setattr(experiment.runner, "train", fail_after_checkpoint)
    failed = experiment.run_arm(job)
    assert failed["status"] == "failed"
    assert failed["state"]["iterations"] == 4
    assert failed["error_type"] == "FloatingPointError"
    arm = experiment.arm_summary(failed, {"construction_cpu_seconds": 0.25}, 0.5)
    experiment.write_scoreboard(
        tmp_path, [{"target_id": experiment.TARGETS[0], "replica": 11, "arms": {"failed": arm}}]
    )
    with (tmp_path / "SCOREBOARD.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert all(row["status"] == "failed" and not row["fidelity"] for row in rows)
    assert all(row["physical_one_qubit"] == "15" and row["physical_two_qubit"] == "3" for row in rows)
    monkeypatch.setattr(experiment.runner, "train", lambda *_a, **_k: pytest.fail("failed arm must stay frozen"))
    assert pilot.digest(experiment.run_arm(job)) == pilot.digest(failed)
    changed = copy.deepcopy(job)
    changed["config"]["iterations"] += 1
    with pytest.raises(ValueError, match="failure configuration changed"):
        experiment.run_arm(changed)
    monkeypatch.setattr(experiment.runner, "source_hashes", lambda: {"numerical": "changed"})
    with pytest.raises(ValueError, match="fingerprint changed"):
        experiment.run_arm(job)


@pytest.mark.usefixtures("small_runtime")
def test_custody_failures_abort_instead_of_becoming_numerical_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integrity failure cannot be downgraded to an ordinary method failure."""
    job = arm_job(tmp_path, "adam")

    def reject() -> dict[str, Any]:
        msg = "Benchmark custody mismatch"
        raise ValueError(msg)

    monkeypatch.setattr(experiment.runner, "custody", reject)
    with pytest.raises(ValueError, match="custody mismatch"):
        experiment.run_arm(job)
    assert not list(tmp_path.glob("*.json"))


def fake_references(root: Path) -> None:
    """Write temporary authenticated 108-gate records without any simulation."""
    circuit = ParameterizedCircuit(6, [ParameterizedGate("ry", (i % 6,), i) for i in range(108)])
    for target_id in experiment.TARGETS:
        for method in experiment.REFERENCES:
            path = root / "focused_curvature" / f"{target_id}__11__{method}.json"
            job = {
                "phase": "development",
                "target_id": target_id,
                "replica": 11,
                "label": method,
                "circuit": pilot.circuit_record(circuit),
                "path": str(path),
            }
            pilot.write_record(
                path,
                {
                    "status": "complete",
                    "job": job,
                    "signature": pilot.digest({k: v for k, v in job.items() if k != "path"}),
                    "state": {"theta": [0.1] * 108, "training_cpu_seconds": 1.0},
                    "fidelities": dict.fromkeys(experiment.channels.NOISE_IDS, 0.5),
                    "ballarin_counts": {"one_qubit": 108, "two_qubit": 0, "depth": 18},
                    "evaluation_cpu_seconds": 0.1,
                },
            )


@pytest.mark.usefixtures("small_runtime")
def test_prepare_only_authenticates_every_unmatched_reference_without_scientific_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Freezing neither constructs states nor scores or trains any target."""
    source = tmp_path / "source"
    fake_references(source)
    monkeypatch.setattr(experiment.runner, "OUTPUT", source)
    monkeypatch.setattr(experiment, "numerical_sources", lambda: {"driver": "fixed"})
    monkeypatch.setattr(experiment.synthesis, "synthesize", lambda *_a: pytest.fail("prepare must not construct"))
    monkeypatch.setattr(experiment.runner, "train", lambda *_a, **_k: pytest.fail("prepare must not train"))
    monkeypatch.setattr(experiment.channels, "evaluate", lambda *_a: pytest.fail("prepare must not score"))
    output = tmp_path / "prepared"
    assert experiment.main(["--output", str(output), "--prepare-only"]) == 0
    protocol = cast("dict[str, Any]", pilot.read_record(output / "PROTOCOL.json"))
    assert len(protocol["references"]) == 16
    experiment.verify_protocol(protocol)
    before = (output / "PROTOCOL.json").read_bytes()
    experiment.freeze_protocol(output, json.loads(experiment.SPEC_PATH.read_text()))
    assert before == (output / "PROTOCOL.json").read_bytes()
    assert [p.name for p in output.iterdir()] == ["PROTOCOL.json"]
    monkeypatch.setattr(experiment, "numerical_sources", lambda: {"driver": "changed"})
    with pytest.raises(ValueError, match="source identity changed"):
        experiment.verify_protocol(protocol)
    with pytest.raises(ValueError, match="Frozen synthesis protocol differs"):
        experiment.freeze_protocol(output, json.loads(experiment.SPEC_PATH.read_text()))
    monkeypatch.setattr(experiment, "numerical_sources", lambda: {"driver": "fixed"})
    ref = protocol["references"][0]
    ref_path = source / "focused_curvature" / f"{ref['target_id']}__11__{ref['method']}.json"
    record = cast("dict[str, Any]", pilot.read_record(ref_path))
    reference_arm = experiment.arm_summary(record, None, None)
    assert reference_arm["comparison"] == "UNMATCHED_different_construction_and_cost"
    assert reference_arm["construction_cpu_seconds"] is None
    assert reference_arm["construction_and_refinement_cpu_seconds"] is None
    assert reference_arm["physical_counts"]["one_qubit"] == 108
    record["fidelities"]["noiseless"] = 0.1
    pilot.write_record(ref_path, record)
    with pytest.raises(ValueError, match="reference checksum changed"):
        experiment.verify_protocol(protocol)


@pytest.mark.usefixtures("small_runtime")
def test_case_builds_once_and_charges_full_physical_construction_to_fresh_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three arms independently start from identical 18-parameter q2 synthesis."""
    monkeypatch.setattr(experiment, "verify_protocol", lambda *_a: None)
    methods = {
        "unchanged": {"method": "unchanged", "iterations": 0},
        "adam": {"method": "adam", "iterations": 2, "checkpoint_every": 1},
        "lbfgs": {"method": "lbfgs", "iterations": 2, "evaluations": 6, "history": 3},
    }
    protocol = {"spec": {"methods": methods}, "references": []}
    job = {"protocol": protocol, "target_id": experiment.TARGETS[0], "replica": 11, "root": str(tmp_path)}
    original_train = cast("Callable[..., dict[str, Any]]", experiment.runner.train)
    starts = []

    def observed_train(*args: object, **kwargs: object) -> dict[str, Any]:
        starts.append((cast("np.ndarray", args[1]).copy(), kwargs.get("state")))
        return original_train(*args, **kwargs)

    monkeypatch.setattr(experiment.runner, "train", observed_train)
    result = experiment.execute_case(job)
    assert result["status"] == "complete"
    assert len(starts) == 3
    assert all(prior is None for _, prior in starts)
    for theta, _ in starts:
        np.testing.assert_array_equal(theta, result["construction"]["candidate"]["theta"])
    cost = result["construction"]["candidate"]["construction_cpu_seconds"]
    for arm in result["arms"].values():
        assert arm["construction_cpu_seconds"] == cost
        assert arm["construction_and_refinement_cpu_seconds"] == cost + arm["state"]["training_cpu_seconds"]
        assert arm["physical_counts"]["one_qubit"] == 15
        assert arm["physical_counts"]["two_qubit"] == 3
        assert set(arm["fidelities"]) == set(experiment.channels.NOISE_IDS)
    assert result["arms"]["unchanged"]["state"]["exact_training_evaluations"] == 0
    assert result["arms"]["adam"]["state"]["exact_training_evaluations"] == 2
    assert result["arms"]["lbfgs"]["state"]["exact_training_evaluations"] <= 6
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        experiment.synthesis, "synthesize", lambda *_a: pytest.fail("completed construction must remain unchanged")
    )
    assert pilot.digest(experiment.execute_case(job)) == pilot.digest(result)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    # A hard process kill can leave untimed work. Preserve that uncertainty;
    # never silently replace its physical cost by zero on a later resume.
    attempts_path = tmp_path / "adam.attempts.json"
    attempts = cast("dict[str, Any]", pilot.read_record(attempts_path))
    attempts["attempts"].append({"status": "running", "cpu_seconds": None, "wall_seconds": None})
    pilot.write_record(attempts_path, attempts)
    (tmp_path / "RESULT.json").unlink()
    resumed = experiment.execute_case(job)
    assert resumed["arms"]["adam"]["physical_invocation_cpu_seconds"] is None
    assert resumed["arms"]["adam"]["physical_invocation_timing_complete"] is False
    assert resumed["arms"]["adam"]["state"] == result["arms"]["adam"]["state"]


@pytest.mark.usefixtures("small_runtime")
def test_construction_failure_keeps_all_three_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Failure before any circuit exists still preserves every predeclared row."""
    monkeypatch.setattr(experiment, "verify_protocol", lambda *_a: None)

    def broken(_target: np.ndarray) -> dict[str, Any]:
        msg = "injected compiler failure"
        raise ValueError(msg)

    monkeypatch.setattr(experiment.synthesis, "synthesize", broken)
    protocol = {"spec": {"methods": experiment.METHODS}, "references": []}
    result = experiment.execute_case({
        "protocol": protocol,
        "target_id": experiment.TARGETS[0],
        "replica": 11,
        "root": str(tmp_path),
    })
    assert result["status"] == "partial_failure"
    assert set(result["arms"]) == set(experiment.METHODS)
    experiment.write_scoreboard(tmp_path, [result])
    with (tmp_path / "SCOREBOARD.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 36
    assert all(row["status"] == "failed" and not row["fidelity"] for row in rows)


@pytest.mark.parametrize(
    ("key", "value"), [("targets", ["confirmation"]), ("replicas", [29]), ("methods", {}), ("phase", "confirmation")]
)
def test_scientific_panel_changes_are_rejected(key: str, value: object) -> None:
    """Confirmation targets or budget changes need a separate protocol."""
    spec = json.loads(experiment.SPEC_PATH.read_text())
    spec[key] = value
    with pytest.raises(ValueError, match="fixed development panel and budgets"):
        experiment.validate_spec(spec)
