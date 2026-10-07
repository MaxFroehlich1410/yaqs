# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Development batch custody, compact all-case reports and small state audits."""

from __future__ import annotations

import copy
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import construction_experiment as experiment
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate


def endpoint() -> dict[str, Any]:
    """Return a q2 engineering endpoint with nontrivial complete physical preparation."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), 0),
            ParameterizedGate("ry", (1,), 1),
            ParameterizedGate("rxx", (0, 1), 2),
        ],
    )
    return {
        "circuit": pilot.circuit_record(circuit),
        "theta": [0.7, -0.3, 0.8],
        "physical_counts": {"one_qubit": 2, "two_qubit": 1, "depth": 2},
        "parameters": 3,
        "updates": 570,
        "construction_cpu_seconds": 2.5,
        "construction_wall_seconds": 3.5,
        "timing_complete": True,
        "diagnostics": {"observed_final_forward_peak_bond": 2},
    }


@pytest.fixture
def frozen_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Return mocked prepared sources while keeping wrapper custody and Qiskit audits real."""
    monkeypatch.setattr(experiment, "sources", lambda: {"numerical": "fixed"})
    monkeypatch.setattr(experiment.construction, "environment", lambda: {"runtime": "fixed"})
    monkeypatch.setattr(experiment.runner, "custody", lambda: {"benchmark_hashes": {"immutable": "fixed"}})
    monkeypatch.setattr(experiment.construction, "verify_manifest", lambda _m: None)
    loaded = []

    def loader(n: int, target_id: str) -> SimpleNamespace:
        assert n == 6
        assert target_id in experiment.TARGETS
        loaded.append(target_id)
        return SimpleNamespace(state_vector_copy=lambda: np.ones(64, dtype=np.complex128) / 8)

    def archive(root: Path, target_id: str, replica: int) -> dict[str, Any]:
        assert root == experiment.ARCHIVE_ROOT
        assert replica == 11
        return {
            **endpoint(),
            "path": str(root / "H25_lr0p7" / target_id / "seed_11/result.json"),
            "checksum": target_id,
        }

    def prepare(
        root: Path, vector: np.ndarray, target_id: str, replica: int, *, archive: dict[str, Any]
    ) -> dict[str, Any]:
        assert vector.shape == (64,)
        record: dict[str, Any] = {
            "target_id": target_id,
            "replica": replica,
            "config": asdict(experiment.construction.Config()),
            "archive_reference": archive,
        }
        pilot.write_record(root / "PROTOCOL.json", record)
        return record

    monkeypatch.setattr(experiment, "load_target", loader)
    monkeypatch.setattr(experiment.construction, "archive_reference", archive)
    monkeypatch.setattr(experiment.construction, "prepare", prepare)
    monkeypatch.setattr(experiment.construction, "reproduce", lambda *_a: pytest.fail("prepare constructed"))
    protocol = experiment.prepare_batch(tmp_path)
    assert loaded == list(experiment.TARGETS)
    return protocol


def job(protocol: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """Return one existing member of an authenticated batch."""
    return {"protocol": protocol, "case": protocol["cases"][index]}


def test_preparation_freezes_all_four_before_any_constructor_run(tmp_path: Path, frozen_batch: dict[str, Any]) -> None:
    """Every development identity and archive link is frozen before scientific work."""
    assert len(frozen_batch["cases"]) == 4
    assert frozen_batch["targets"] == list(experiment.TARGETS)
    assert frozen_batch["replicas"] == [11]
    assert len(list(tmp_path.rglob("PROTOCOL.json"))) == 4
    assert not list(tmp_path.rglob("RESULT.json"))
    assert not list(tmp_path.rglob("AUDIT.json"))
    experiment.verify_batch(frozen_batch)
    assert pilot.digest(pilot.read_record(tmp_path / "BATCH_PROTOCOL.json")) == pilot.digest(frozen_batch)


@pytest.mark.parametrize("change", ["target", "replica", "sources", "archive", "missing"])
def test_batch_custody_rejects_heldout_or_reconfigured_panel(frozen_batch: dict[str, Any], change: str) -> None:
    """A held-out identifier, changed seed or dropped family cannot enter a prepared batch."""
    altered = copy.deepcopy(frozen_batch)
    if change == "target":
        altered["cases"][0]["target_id"] = "haar_fresh_1"
    elif change == "replica":
        altered["cases"][0]["replica"] = 29
    elif change == "sources":
        altered["sources"] = {"numerical": "changed"}
    elif change == "archive":
        altered["archive_root"] = "/different/archive"
    else:
        altered["cases"].pop()
    with pytest.raises(ValueError, match="batch panel"):
        experiment.verify_batch(altered)


def test_case_manifests_cannot_change_after_batch_preparation(frozen_batch: dict[str, Any]) -> None:
    """A valid checksum envelope still cannot replace a case inside its frozen parent."""
    path = Path(frozen_batch["cases"][0]["root"]) / "PROTOCOL.json"
    record = cast("dict[str, Any]", pilot.read_record(path))
    record["archive_reference"]["theta"][0] += 0.01
    pilot.write_record(path, record)
    with pytest.raises(ValueError, match="case or archive identity changed"):
        experiment.verify_batch(frozen_batch)


def test_independent_full_preparation_audit_detects_parameter_and_state_changes() -> None:
    """The audit starts from zero and distinguishes matching states from matching parameters."""
    original = endpoint()
    same = experiment.audit_state(original, original)
    assert same["matches"]
    assert same["ideal_state_overlap"] == pytest.approx(1, abs=1e-12)
    periodic = copy.deepcopy(original)
    periodic["theta"][0] += 2 * np.pi
    audit = experiment.audit_state(periodic, original)
    assert audit["ideal_state_overlap"] == pytest.approx(1, abs=1e-12)
    assert audit["phase_aligned_state_max_abs_error"] < 1e-12
    assert not audit["matches"]
    changed = copy.deepcopy(original)
    changed["theta"][0] += 0.1
    audit = experiment.audit_state(changed, original)
    assert not audit["matches"]
    assert audit["ideal_state_overlap"] < 0.999


def test_completed_case_caches_only_after_validating_adapter_state(
    tmp_path: Path, frozen_batch: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Audits reuse immutable completed endpoints while the adapter checks full checkpoint custody."""
    calls = []

    def reproduce(root: Path, _manifest: dict[str, Any]) -> dict[str, Any]:
        calls.append(root)
        return endpoint()

    monkeypatch.setattr(experiment.construction, "reproduce", reproduce)
    result = experiment.run_case(job(frozen_batch))
    assert result["status"] == "complete"
    assert result["construction_cpu_seconds"] == pytest.approx(2.5)
    assert result["historical_cpu_seconds"] is None
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    monkeypatch.setattr(experiment, "audit_state", lambda *_a: pytest.fail("cached audit recomputed"))
    cached = experiment.run_case(job(frozen_batch))
    assert pilot.digest(cached) == pilot.digest(result)
    assert len(calls) == 2
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    changed = endpoint()
    changed["theta"][0] += 0.1
    monkeypatch.setattr(experiment.construction, "reproduce", lambda *_a: changed)
    with pytest.raises(ValueError, match="audit identity changed"):
        experiment.run_case(job(frozen_batch))


def test_retryable_numerical_failure_retains_cost_and_resumes_same_constructor(
    tmp_path: Path, frozen_batch: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A numerical failure retains its attempt cost and retries only the same frozen case."""

    def fail(root: Path, _manifest: dict[str, Any]) -> dict[str, Any]:
        pilot.write_record(root / "ATTEMPTS.json", {"attempts": [{"cpu_seconds": 0.7, "wall_seconds": 0.9}]})
        msg = "injected numerical failure after checkpoint"
        raise FloatingPointError(msg)

    monkeypatch.setattr(experiment.construction, "reproduce", fail)
    failed = experiment.run_case(job(frozen_batch))
    assert failed["status"] == "failed"
    assert failed["construction_cpu_seconds"] == pytest.approx(0.7)
    summary = experiment.write_summary(tmp_path, frozen_batch, [failed])
    assert len(summary["rows"]) == 4
    assert [r["status"] for r in summary["rows"]] == ["failed", "pending", "pending", "pending"]
    completed = endpoint()
    completed["construction_cpu_seconds"] = 3.2
    monkeypatch.setattr(experiment.construction, "reproduce", lambda *_a: completed)
    resumed = experiment.run_case(job(frozen_batch))
    assert resumed["status"] == "complete"
    assert resumed["construction_cpu_seconds"] == pytest.approx(3.2)
    assert len(list(tmp_path.rglob("FAILURES.json"))) == 1


def test_integrity_failure_and_keyboard_interrupt_are_not_downgraded_to_numerical_rows(
    tmp_path: Path, frozen_batch: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Custody failures abort; interrupts leave the underlying adapter resumable."""

    def integrity(*_args: object) -> dict[str, Any]:
        msg = "checkpoint identity changed"
        raise ValueError(msg)

    monkeypatch.setattr(experiment.construction, "reproduce", integrity)
    with pytest.raises(ValueError, match="checkpoint identity"):
        experiment.run_case(job(frozen_batch))

    def interrupt(*_args: object) -> dict[str, Any]:
        raise KeyboardInterrupt

    monkeypatch.setattr(experiment.construction, "reproduce", interrupt)
    with pytest.raises(KeyboardInterrupt):
        experiment.run_case(job(frozen_batch))
    assert not list(tmp_path.rglob("FAILURES.json"))


def test_summary_retains_every_case_and_mismatch_without_large_endpoint_copies(
    tmp_path: Path, frozen_batch: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The aggregate includes physical counts and costs, never repeated circuits or histories."""
    records = []
    for index in range(4):
        result = endpoint()
        result["theta"][0] += 0.1 if index == 2 else 0
        monkeypatch.setattr(experiment.construction, "reproduce", lambda *_a, value=result: value)
        records.append(experiment.run_case(job(frozen_batch, index)))
    summary = experiment.write_summary(tmp_path, frozen_batch, records)
    assert summary["status"] == "partial_failure"
    assert summary["total_construction_cpu_seconds"] == pytest.approx(10)
    assert not summary["all_archive_reproductions_match"]
    assert summary["rows"][2]["status"] == "mismatch"
    assert all("circuit" not in row and "theta" not in row for row in summary["rows"])
    with (tmp_path / "SCOREBOARD.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert all(row["parameters"] == "3" and row["updates"] == "570" for row in rows)


@pytest.mark.parametrize("workers", [0, 5])
def test_invalid_worker_cap_rejected_before_preparation(tmp_path: Path, workers: int) -> None:
    """No automatic parallelism or oversized worker request is permitted."""
    with pytest.raises(SystemExit):
        experiment.main(["--output", str(tmp_path), "--prepare-only", "--workers", str(workers)])
    assert not list(tmp_path.iterdir())


def test_run_cli_reports_first_case_time_and_complete_all_target_summary(
    tmp_path: Path,
    frozen_batch: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The execution CLI honors a bounded pool and reports the first observed completion."""
    monkeypatch.setattr(experiment, "ProcessPoolExecutor", ThreadPoolExecutor)

    def reproduce(_root: Path, _manifest: dict[str, Any]) -> dict[str, Any]:
        assert len(list(tmp_path.rglob("PROTOCOL.json"))) == 4
        return endpoint()

    monkeypatch.setattr(experiment.construction, "reproduce", reproduce)
    assert experiment.main(["--output", str(tmp_path), "--run", "--workers", "2"]) == 0
    summary = pilot.read_record(tmp_path / "SUMMARY.json")
    assert summary["status"] == "complete"
    assert summary["all_archive_reproductions_match"]
    assert len(list(tmp_path.rglob("AUDIT.json"))) == 4
    output = capsys.readouterr().out
    assert "First case completed after" in output
    assert "ignoring in-flight progress" in output
    assert pilot.digest(pilot.read_record(tmp_path / "BATCH_PROTOCOL.json")) == pilot.digest(frozen_batch)


def test_duplicate_summary_cases_cannot_hide_missing_targets(tmp_path: Path, frozen_batch: dict[str, Any]) -> None:
    """Completeness cannot be manufactured by repeating a successful target."""
    row = {"target_id": experiment.TARGETS[0], "status": "complete"}
    with pytest.raises(ValueError, match="unique members"):
        experiment.write_summary(tmp_path, frozen_batch, [row, row])
