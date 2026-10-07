# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Synthetic fixed-path memory diagnostic agreement, cost separation and durable custody."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import mps_memory_experiment as diagnostic
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import KrotovNoiseMap, ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("qubits", [1, 2, 3])
@pytest.mark.parametrize("factor", [False, True])
def test_same_path_memory_measurement_matches_dense_without_updates(
    tmp_path: Path, qubits: int, *, factor: bool
) -> None:
    """The actual public MPS adjoint and dense oracle agree at unchanged q1-q3 inputs."""
    circuit = ParameterizedCircuit(
        qubits, [ParameterizedGate("ry", (0,), param_index=0), ParameterizedGate("rz", (qubits - 1,), param_index=1)]
    )
    theta = np.array([0.3, -0.2])
    target_mps = MPS(qubits, state="zeros")
    target = target_mps.to_vec()
    noise = (
        KrotovNoiseMap(operators=((np.array([[0, 1], [1, 0]], dtype=np.complex128), (0,)),))
        if factor
        else KrotovNoiseMap()
    )
    attempt = diagnostic.old.Attempt(tmp_path, "tiny")
    report = diagnostic.measure(circuit, theta, target, target_mps, [noise, KrotovNoiseMap()], attempt)
    attempt.finish(None)
    assert report["accepted"]
    assert report["work_counts"]["optimizer_updates"] == 0
    assert report["analytic_no_factor_path"] is not factor
    assert report["operator_count"] == int(factor)
    np.testing.assert_array_equal(theta, np.array([0.3, -0.2]))
    assert report["memory"]["forward_tape"]["tensor_references"] == 5 * qubits
    assert (
        report["memory"]["retained_union"]["complete_backing_payload_bytes"]
        >= report["memory"]["forward_tape"]["complete_backing_payload_bytes"]
    )
    stages = diagnostic.old.costs(attempt.record)["phases"]
    for name in ("mps_gradient", "dense_gradient", "retained_memory_audit", "forward_norm_audit"):
        assert stages[name]["completed"] == 1
        assert stages[name]["known_cpu_seconds"] >= 0
    assert report["peak_rss"]["bytes"] > 0
    assert report["prior_overlap_gate"] == "failed; unchanged"


def tiny_measurement(attempt: diagnostic.old.Attempt) -> dict[str, Any]:
    """Return a tiny independently timed result for custody-only fixtures.

    Returns:
        A complete diagnostic result without invoking a simulator.
    """
    return attempt.call("retained_memory_audit", lambda: {"accepted": True, "tensor_bytes": 64})


def test_completed_cache_is_byte_preserving_and_rejects_resigned_acceptance(tmp_path: Path) -> None:
    """No cached numerical work repeats, and public acceptance remains bound to raw measurements."""
    result = diagnostic.run_endpoint(tmp_path, "tiny", tiny_measurement)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    def forbidden(attempt: diagnostic.old.Attempt) -> dict[str, Any]:
        del attempt
        pytest.fail("completed memory work repeated")

    assert diagnostic.run_endpoint(tmp_path, "tiny", forbidden) == result
    assert {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    result["accepted"] = False
    pilot.write_record(tmp_path / "RESULT.json", result)
    pilot.write_record(
        tmp_path / "COMPLETED.json", {"signature": result["signature"], "result_checksum": pilot.digest(result)}
    )
    with pytest.raises(ValueError, match="raw measurements"):
        diagnostic.run_endpoint(tmp_path, "tiny", forbidden)


def test_failed_attempt_costs_survive_resume_and_unknown_kill_blocks(tmp_path: Path) -> None:
    """A caught failure is paid again on replay; a null hard-kill marker cannot become free."""

    def failed(attempt: diagnostic.old.Attempt) -> dict[str, Any]:
        tiny_measurement(attempt)
        msg = "fixture failure"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="fixture"):
        diagnostic.run_endpoint(tmp_path, "tiny", failed)
    result = diagnostic.run_endpoint(tmp_path, "tiny", tiny_measurement)
    ledger = diagnostic.old.read(tmp_path / "ATTEMPTS.json")
    assert [a["status"] for a in ledger["attempts"]] == ["failed", "complete"]
    assert result["costs"]["cpu_seconds"] == sum(a["cpu_seconds"] for a in ledger["attempts"])
    ledger["attempts"].append({"status": "running", "cpu_seconds": None, "wall_seconds": None})
    pilot.write_record(tmp_path / "ATTEMPTS.json", ledger)
    with pytest.raises(ValueError, match="Unknown memory"):
        diagnostic.run_endpoint(tmp_path, "tiny", tiny_measurement)


@pytest.mark.parametrize("new_file", ["new_work.bin", "nested/RESULT.json", "nested/COMPLETED.json"])
def test_changed_child_or_omitted_input_cannot_be_adopted(tmp_path: Path, new_file: str) -> None:
    """The stored full child registry catches unindexed files, and frozen inputs reject changed bytes."""
    diagnostic.run_endpoint(tmp_path, "tiny", tiny_measurement)
    new_path = tmp_path / new_file
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_bytes(b"paid work")
    with pytest.raises(ValueError, match="registry"):
        diagnostic.run_endpoint(tmp_path, "tiny", tiny_measurement)
    path = tmp_path / "source.json"
    pilot.write_record(path, {"source": 1})
    files = {str(path): diagnostic.old.file_hash(path)}
    diagnostic.authenticate_files(files)
    pilot.write_record(path, {"source": 2})
    with pytest.raises(ValueError, match="custody"):
        diagnostic.authenticate_files(files)


def test_prepare_only_and_incomplete_summary_never_run_measurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preparation freezes supplied source metadata; missing endpoints cannot pass the complete gate."""
    panel = {"entries": [{"target_id": "synthetic_q2", "replica": 11, "label": "original108"}], "files": {}}
    monkeypatch.setattr(diagnostic, "inputs", lambda *_args: panel)
    monkeypatch.setattr(diagnostic, "measure", lambda *_args: pytest.fail("prepare invoked a gradient"))
    protocol = diagnostic.prepare(tmp_path / "out", tmp_path / "old", tmp_path / "physical", tmp_path / "vectors")
    summary = diagnostic.summarize(tmp_path / "out", protocol)
    assert summary["status"] == "incomplete_or_failed"
    assert not summary["accepted"]
    assert summary["cases"][0]["costs"] is None


def test_ordered_old_batch_identity_is_checked_before_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed old map hash is rejected before selecting path zero or computing any new gradient."""
    target = MPS(2, state="zeros")
    vector = target.to_vec()
    digest = diagnostic.hashlib.sha256(vector.tobytes()).hexdigest()
    vector_path, target_path = tmp_path / "vector.json", tmp_path / "target.json"
    pilot.write_record(
        vector_path, {"target": [[float(v.real), float(v.imag)] for v in vector], "target_vector_sha256": digest}
    )
    pilot.write_record(
        target_path,
        {
            "target_sha256": digest,
            "tensors": [{"real": t.real.tolist(), "imag": t.imag.tolist()} for t in target.tensors],
        },
    )
    circuit = ParameterizedCircuit(2, [ParameterizedGate("ry", (0,), param_index=0)])
    entry = {
        "target_id": "synthetic_q2",
        "target_vector_file": str(vector_path),
        "target_mps_file": str(target_path),
        "endpoint": {"circuit": pilot.circuit_record(circuit), "theta": [0.3]},
        "first_step": {"map_sha256": "wrong", "weights": [0.125] * 8},
    }
    monkeypatch.setattr(diagnostic, "measure", lambda *_args: pytest.fail("changed maps reached gradient"))
    attempt = diagnostic.old.Attempt(tmp_path / "run", "maps")
    with pytest.raises(ValueError, match="path order"):
        diagnostic.compute(entry, attempt)
    attempt.finish(ValueError("wrong maps"))
