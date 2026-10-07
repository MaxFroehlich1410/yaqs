# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Small engineering fixtures for the fixed known-zero compiler diagnostic panel."""

from __future__ import annotations

import csv
import hashlib
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import state_preparation_compilation_experiment as experiment
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def forbidden(*_args: object, **_kwargs: object) -> None:
    """Reject accidental scientific execution in a preparation or cache test."""
    pytest.fail("Scientific work is forbidden in this read-only fixture")


def source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a complete synthetic q6 artifact graph without scientific evaluation."""
    initial_root, root = tmp_path / "initial", tmp_path / "composition"
    circuit = ParameterizedCircuit(
        6,
        [
            ParameterizedGate("rz", (5,), 1, logical_gate_id=12, native_gate_id=32),
            ParameterizedGate("rx", (1,), 0, angle_scale=-0.7, noise_enabled=False),
        ],
    )
    endpoint = {"circuit": pilot.circuit_record(circuit), "theta": [0.4, 0.3]}
    inputs = {}
    for target in experiment.TARGETS:
        path = initial_root / f"{target}__11" / "INITIALIZERS.json"
        pilot.write_record(path, {"legacy": endpoint, "conditional": endpoint})
        inputs[str(path.resolve())] = experiment.file_hash(path)
    original: dict[str, Any] = {
        "initializer_root": str(initial_root),
        "input_files": inputs,
        "starts": [{"target_id": target, "replica": 11, "endpoint": endpoint} for target in experiment.TARGETS],
    }
    pilot.write_record(root / "PROTOCOL.json", original)
    checksums = {}
    for source in original["starts"]:
        target = source["target_id"]
        case_root = root / f"{target}__11"
        key = pilot.digest(endpoint["circuit"])
        path = case_root / "circuits" / f"{key}.json"
        pilot.write_record(path, endpoint["circuit"])
        case: dict[str, Any] = {
            "signature": pilot.digest([original, source]),
            "status": "complete",
            "target_id": target,
            "replica": 11,
            "dependencies": {str(path.relative_to(case_root)): experiment.file_hash(path)},
            "arms": {
                method: {"status": "complete", "final": {"circuit_hash": key, "theta": endpoint["theta"]}}
                for method in experiment.composition.PRIMARY
            },
        }
        pilot.write_record(case_root / "RESULT.json", case)
        checksums[target] = pilot.digest(case)
    pilot.write_record(
        root / "SUMMARY.json",
        {
            "status": "complete",
            "protocol_checksum": pilot.digest(original),
            "case_checksums": checksums,
        },
    )

    def verify(protocol: dict[str, Any]) -> None:
        if any(
            experiment.file_hash(experiment.Path(path)) != checksum
            for path, checksum in protocol["input_files"].items()
        ):
            msg = "Source parent inputs changed."
            raise ValueError(msg)

    monkeypatch.setattr(experiment.composition, "verify", verify)
    monkeypatch.setattr(experiment, "sources", lambda: {"fixture": "fixed"})
    monkeypatch.setattr(experiment.runner, "custody", lambda: {"benchmark_hashes": {"fixture": "unchanged"}})
    monkeypatch.setattr(
        experiment.composition.initializer_experiment.construction, "environment", lambda: {"fixture": True}
    )
    target = np.zeros(64, dtype=complex)
    target[0] = 1
    monkeypatch.setattr(experiment, "load_target", lambda *_args: SimpleNamespace(state_vector_copy=target.copy))
    return root


def test_prepare_fixed24_sources_is_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preparation authenticates source choices and IDs without compiling or scoring."""
    root = source_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(experiment.compiler, "compile_endpoint", forbidden)
    monkeypatch.setattr(experiment.measured, "score_point", forbidden)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*.json")}
    output = tmp_path / "diagnostic"
    protocol = experiment.prepare(output, root)
    assert len(protocol["pairs"]) == 24
    assert {pair["label"] for pair in protocol["pairs"]} == set(experiment.LABELS)
    assert {pair["target_id"] for pair in protocol["pairs"]} == set(experiment.TARGETS)
    assert protocol["noise_ids"] == list(experiment.channels.NOISE_IDS)
    assert protocol["tolerance"] == pytest.approx(1e-12, rel=0, abs=0)
    assert protocol["pairs"][0]["endpoint"]["circuit"]["gates"][0]["native_gate_id"] == 32
    experiment.verify(protocol)
    assert experiment.prepare(output, root) == protocol
    assert before == {path: experiment.Path(path).read_bytes() for path in before}


@pytest.mark.parametrize("mutation", ["case", "summary", "circuit", "initializer", "missing"])
def test_changed_source_graph_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    """Incomplete panels and every externally supplied artifact remain authenticated."""
    root = source_fixture(tmp_path, monkeypatch)
    if mutation == "summary":
        path = root / "SUMMARY.json"
        value = experiment.record(path)
        value["status"] = "partial_failure"
    elif mutation == "initializer":
        path = tmp_path / "initial" / f"{experiment.TARGETS[0]}__11" / "INITIALIZERS.json"
        value = experiment.record(path)
        value["conditional"]["theta"][0] += 0.1
    elif mutation == "circuit":
        path = next((root / f"{experiment.TARGETS[0]}__11" / "circuits").glob("*.json"))
        value = experiment.record(path)
        value["gates"][0]["native_gate_id"] += 1
    else:
        path = root / f"{experiment.TARGETS[0]}__11" / "RESULT.json"
        if mutation == "missing":
            path.unlink()
            with pytest.raises(FileNotFoundError):
                experiment.source_panel(root)
            return
        value = experiment.record(path)
        value["status"] = "partial_failure"
    pilot.write_record(path, value)
    with pytest.raises(ValueError, match=r"changed|incomplete|complete composition"):
        experiment.source_panel(root)


@pytest.mark.parametrize("field", ["tolerance", "labels", "pairs", "sources", "replicas", "noise_ids"])
def test_frozen_protocol_changes_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    """No threshold, source endpoint, panel or implementation tuning is permitted."""
    root = source_fixture(tmp_path, monkeypatch)
    protocol = experiment.prepare(tmp_path / "out", root)
    changed = deepcopy(protocol)
    if field == "pairs":
        changed[field][0]["endpoint"]["theta"][0] += 0.2
    else:
        changed[field] = None
    with pytest.raises(ValueError, match="custody changed"):
        experiment.verify(changed)


def pair_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Return a small q2 pair whose loader cannot access scientific target states."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("rz", (1,), 0, logical_gate_id=21, native_gate_id=81),
            ParameterizedGate("ry", (0,), 1),
            ParameterizedGate("rxx", (1, 0), 2, angle_scale=-0.5),
        ],
    )
    theta = np.array([0.4, 0.3, 0.2])
    target = np.array([1, 0.2 + 0.3j, -0.4j, 0.6], dtype=complex)
    target /= np.linalg.norm(target)
    source = {
        "target_id": "engineering",
        "replica": 11,
        "label": "legacy_initializer",
        "endpoint": {"circuit": pilot.circuit_record(circuit), "theta": theta.tolist()},
        "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
    }
    monkeypatch.setattr(experiment, "load_target", lambda *_args: SimpleNamespace(state_vector_copy=target.copy))
    monkeypatch.setattr(experiment, "verify", lambda _protocol: None)
    return {"protocol": {"pairs": [source]}, "source": source, "root": str(tmp_path / "case")}


def test_pair_exact24_evaluations_and_immutable_cached_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A physical source and compiled circuit get separate complete all12 evaluations."""
    job = pair_fixture(tmp_path, monkeypatch)
    result = experiment.run_pair(job)
    assert result["status"] == "complete"
    assert result["compilation"]["value"]["source_to_output"] == [None, 0, 1]
    assert result["compilation"]["value"]["independent_audit"]["passed"]
    assert result["compiler_attempts"] == result["compiler_completed_attempts"] == 1
    assert sum(score["physical_attempted_channel_calls"] for score in result["scoring"].values()) == 24
    for noise in experiment.channels.NOISE_IDS:
        assert (
            result["fidelity_delta"][noise]
            == result["after"]["fidelities"][noise] - result["before"]["fidelities"][noise]
        )
    nested_cpu = result["compilation"]["cpu_seconds"] + sum(
        score["cpu_seconds"] for score in result["scoring"].values()
    )
    assert result["physical_invocation"]["cpu_seconds"] >= nested_cpu
    assert any(name.startswith("before/scores/") and name.endswith("RESULT.json") for name in result["dependencies"])
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*.json")}
    monkeypatch.setattr(experiment.compiler, "compile_endpoint", forbidden)
    monkeypatch.setattr(experiment.measured, "score_point", forbidden)
    assert pilot.digest(experiment.run_pair(job)) == pilot.digest(result)
    assert before == {str(path): path.read_bytes() for path in tmp_path.rglob("*.json")}


@pytest.mark.parametrize("stage", ["compiler", "score"])
def test_caught_failure_resume_preserves_replayed_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """Failures retain complete CPU and exact replayed channel call counts."""
    job = pair_fixture(tmp_path, monkeypatch)
    owner, name = (
        (experiment.compiler, "compile_endpoint") if stage == "compiler" else (experiment.channels, "evaluate")
    )
    original = getattr(owner, name)
    calls = 0

    def interrupt(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == (1 if stage == "compiler" else 3):
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, interrupt)
    with pytest.raises(KeyboardInterrupt):
        experiment.run_pair(job)
    monkeypatch.setattr(owner, name, original)
    result = experiment.run_pair(job)
    ledger = experiment.record(experiment.Path(job["root"]) / "CASE_ATTEMPTS.json")
    assert [entry["status"] for entry in ledger["attempts"]] == ["interrupted", "complete"]
    assert result["physical_invocation"]["cpu_seconds"] == sum(entry["cpu_seconds"] for entry in ledger["attempts"])
    if stage == "compiler":
        assert result["compiler_attempts"] == 2
        assert result["failed_compiler_gate_work"].startswith("unknown partial count")
    else:
        assert result["scoring"]["before"]["physical_attempted_channel_calls"] == 15
        assert result["scoring"]["before"]["physical_completed_channel_calls"] == 14


def test_unmeasured_process_death_blocks_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A running marker cannot be silently assigned zero elapsed work."""
    job = pair_fixture(tmp_path, monkeypatch)
    signature = pilot.digest([job["protocol"], job["source"]])
    pilot.write_record(
        experiment.Path(job["root"]) / "CASE_ATTEMPTS.json",
        {
            "signature": signature,
            "attempts": [{"status": "running", "cpu_seconds": None, "wall_seconds": None}],
        },
    )
    with pytest.raises(ValueError, match="killed invocation"):
        experiment.run_pair(job)


@pytest.mark.parametrize("name", ["COMPILATION.json", "COMPILATION.attempts.json", "CASE_ATTEMPTS.json"])
def test_completed_artifact_tamper_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Completed cached results authenticate nested compilation and invocation records."""
    job = pair_fixture(tmp_path, monkeypatch)
    experiment.run_pair(job)
    path = experiment.Path(job["root"]) / name
    value = experiment.record(path)
    value["changed"] = True
    pilot.write_record(path, value)
    with pytest.raises(ValueError, match="durable dependency"):
        experiment.run_pair(job)


def test_complete_and_missing_summary_keep_all_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Incomplete panels cannot acquire a favorable selected-subset aggregate."""
    job = pair_fixture(tmp_path, monkeypatch)
    result = experiment.run_pair(job)
    output = tmp_path / "reports"
    monkeypatch.setattr(experiment, "TARGETS", ("engineering",))
    monkeypatch.setattr(experiment, "LABELS", ("legacy_initializer",))
    summary = experiment.write_summary(output, job["protocol"], [result])
    assert summary["status"] == "complete"
    assert summary["complete_panel_mean_noisy_delta"] == result["mean_noisy_delta"]
    with (output / "scoreboard.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 24
    assert {row["endpoint"] for row in rows} == {"before", "after"}
    summary = experiment.write_summary(output, job["protocol"], [])
    assert summary["complete_panel_mean_noisy_delta"] is None
    assert summary["status"] == "partial_failure"
    assert summary["missing_pairs"] == [experiment.pair_key(job["source"])]
    with (output / "scoreboard.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 24
    assert all(row["status"] == "missing_or_failed" and not row["fidelity"] for row in rows)


def test_failed_pair_retains_explicit_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed worker is recorded without deleting its target from the panel."""
    job = pair_fixture(tmp_path, monkeypatch)
    job["source"]["target_sha256"] = "wrong"
    result = experiment.attempt_pair(job)
    assert result["status"] == "failed"
    assert result["target_id"] == job["source"]["target_id"]
    assert "target vector changed" in result["failure"]
    assert experiment.record(experiment.Path(job["root"]) / "FAILURE.json") == result


@pytest.mark.parametrize("field", ["before", "fidelity_delta", "physical_invocation"])
def test_top_level_result_cells_are_bound_to_child_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Changing only a top-level fidelity or cost cannot bypass child-file custody."""
    job = pair_fixture(tmp_path, monkeypatch)
    result = experiment.run_pair(job)
    if field == "before":
        result[field]["fidelities"]["noiseless"] += 0.1
    elif field == "fidelity_delta":
        result[field]["noiseless"] += 0.1
    else:
        result[field]["cpu_seconds"] = 0.0
    pilot.write_record(experiment.Path(job["root"]) / "RESULT.json", result)
    with pytest.raises(ValueError, match=r"cells differ|deltas or inclusive cost"):
        experiment.run_pair(job)


def test_duplicate_summary_records_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replicating a favorable row cannot alter the equal-pair aggregate."""
    job = pair_fixture(tmp_path, monkeypatch)
    result = experiment.run_pair(job)
    with pytest.raises(ValueError, match="duplicate or unregistered"):
        experiment.write_summary(tmp_path / "reports", job["protocol"], [result, result])


def test_omitted_dependency_does_not_bypass_tamper_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact complete child map is required, not just valid listed entries."""
    job = pair_fixture(tmp_path, monkeypatch)
    result = experiment.run_pair(job)
    result["dependencies"].pop("COMPILATION.attempts.json")
    pilot.write_record(experiment.Path(job["root"]) / "RESULT.json", result)
    with pytest.raises(ValueError, match="durable dependency"):
        experiment.run_pair(job)


def test_failed_completed_cache_check_does_not_mutate_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Even failed custody verification leaves an existing completed case untouched."""
    job = pair_fixture(tmp_path, monkeypatch)
    result = experiment.run_pair(job)
    result["dependencies"].clear()
    pilot.write_record(experiment.Path(job["root"]) / "RESULT.json", result)
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*.json")}
    failure = experiment.attempt_pair(job)
    assert failure["status"] == "failed"
    assert before == {str(path): path.read_bytes() for path in tmp_path.rglob("*.json")}
