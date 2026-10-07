# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check observational SVD tracing, exact capture, and bounded memory diagnostics."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from experiments.state_preparation_competitors import ballarin_memory as memory
from experiments.state_preparation_competitors import ballarin_memory_backward as backward
from experiments.state_preparation_competitors.bounded_probe import run_bounded
from experiments.state_preparation_competitors.iqsp_comb_backend import (
    CombLayout,
    CombTensorNetworkObjective,
    comb_target_from_vector,
)
from experiments.state_preparation_competitors.iqsp_tn import iqsp_tn_ansatz
from experiments.state_preparation_competitors.vendor.ballarin_comb.svd import SVDGESDD


def test_trace_preserves_values_derivatives_and_restores(tmp_path: Path) -> None:
    """The observer returns identical factors and derivatives without retaining tensors."""
    original = torch.linalg.svd
    rng = np.random.default_rng(88)
    matrix = torch.tensor(rng.normal(size=(8, 4)) + 1j * rng.normal(size=(8, 4)), requires_grad=True)
    u, s, v = SVDGESDD.apply(matrix, torch.tensor(1e-12), None)
    expected = torch.autograd.grad(((u * s) @ v.mH).real.sum(), matrix)[0]
    recorder = memory.SVDRecorder(tmp_path)
    with recorder.active():
        u, s, v = SVDGESDD.apply(matrix, torch.tensor(1e-12), None)
        actual = torch.autograd.grad(((u * s) @ v.mH).real.sum(), matrix)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.linalg.svd is original
    assert recorder.records[0]["input_bytes"] == 8 * 4 * 16
    assert recorder.records[0]["full_matrices"] is False
    assert [json.loads(line)["event"] for line in recorder.trace.read_text().splitlines()] == ["begin", "end"]
    with pytest.raises(RuntimeError), recorder.active():
        torch.linalg.svd(torch.ones(2))
    assert torch.linalg.svd is original
    with pytest.raises(FileExistsError):
        memory.SVDRecorder(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        memory.SVDRecorder(tmp_path, 0)


def test_exact_matrix_capture_and_isolated_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Noncontiguous complex input is captured losslessly and checked before replay."""
    monkeypatch.setattr(memory.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3))
    capture, isolated = tmp_path / "capture", tmp_path / "isolated"
    capture.mkdir()
    isolated.mkdir()
    matrix = torch.arange(24, dtype=torch.float64).reshape(4, 6).T.to(torch.complex128).conj()
    recorder = memory.SVDRecorder(capture, 1)
    with pytest.raises(memory.MatrixCapturedError), recorder.active():
        torch.linalg.svd(matrix, full_matrices=False)
    np.testing.assert_array_equal(np.load(capture / "matrix.npy"), matrix.resolve_conj().numpy())
    result = memory.isolated_probe(isolated, capture)
    assert result["singular_values_finite"]
    assert result["singular_count"] == 4
    assert result["input_bytes"] == 24 * 16
    with (capture / "matrix.npy").open("ab") as stream:
        stream.write(b"broken")
    with pytest.raises(ValueError, match="hash"):
        memory.isolated_probe(isolated, capture)
    monkeypatch.setattr(memory.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="reserve"):
        recorder.capture(matrix, {"index": 1, "input_bytes": 384})


def test_phase_samples_and_disk_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase samples retain peaks and low disk halts only the supervised worker."""
    monkeypatch.setattr(memory.shutil, "disk_usage", lambda _: SimpleNamespace(free=1000))
    observer = memory.PhaseObserver(tmp_path, minimum_free_bytes=100)
    memory.write_receipt(tmp_path / "phase.json", {"pid": os.getpid(), "stage": "svd", "index": 2})
    assert observer(os.getpid(), 10, 20) is None
    assert observer(os.getpid(), 30, 15) is None
    assert observer.peaks["svd:2"] == {"samples": 2, "rss_bytes": 30, "physical_bytes": 20}
    monkeypatch.setattr(memory.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    result = run_bounded([sys.executable, "-c", "import time; time.sleep(5)"], tmp_path / "stop.log", observer=observer)
    assert result["status"] == "disk_limit"
    assert result["exit_code"] != 0
    with pytest.raises(TypeError, match="callable"):
        # Exercise runtime rejection of an intentionally ill-typed callback.
        run_bounded([sys.executable], tmp_path / "invalid.log", observer=3)  # ty: ignore[invalid-argument-type]
    result = memory.supervise(tmp_path, [sys.executable, "-c", "import time; time.sleep(5)"])
    assert result["status"] == "disk_limit"
    assert json.loads((tmp_path / "phase_peaks.json").read_text())["minimum_observed_free_bytes"] == 1


def test_small_forward_parity_and_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The traced IBM circuit matches ordinary no-grad fidelity and can stop at an exact split."""
    monkeypatch.setattr(memory.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3))
    monkeypatch.setattr(
        memory,
        "tci_target_bank",
        lambda targets, **_: (
            (comb_target_from_vector(targets[0], CombLayout(6)),),
            {"unit_test_target_oracle": True},
        ),
    )
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps({"num_qubits": 6, "depth": 1, "final_theta": [0.1] * iqsp_tn_ansatz(6, 1).num_parameters})
    )
    engine, theta, metadata = memory.prepare_forward(checkpoint, "default")
    original = CombTensorNetworkObjective(engine.template, engine.comb_target, "IBM", options=engine.options)
    expected = original.fidelity(theta)
    assert engine.storage_snapshot() == {}
    assert engine.fidelity(theta) == expected
    assert engine.observed_state is not None
    assert engine.observed_state() is None
    full, capture, invalid = (tmp_path / name for name in ("full", "capture", "invalid"))
    for path in (full, capture, invalid):
        path.mkdir()
    result = memory.forward_probe(full, checkpoint, "default")
    assert result["fidelity"] == pytest.approx(expected, abs=0)
    assert result["theta_sha256"] == metadata["theta_sha256"]
    assert result["diagnostics"]["svd_splits"] == result["calls"]
    first = json.loads((full / "svd.jsonl").read_text().splitlines()[0])
    assert first["state_unique_backing_bytes"] >= first["state_logical_bytes"] > 0
    result = memory.forward_probe(capture, checkpoint, "tighter", 2)
    assert result["status"] == "matrix_captured"
    assert result["gradient_computed"] is False
    with pytest.raises(ValueError, match="not reached"):
        memory.forward_probe(invalid, checkpoint, "default", 9999)
    with pytest.raises(ValueError, match="compression"):
        memory.prepare_forward(checkpoint, "invalid")


def test_memory_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatch keeps worker arguments separate from the supervisor's options."""
    calls = []
    monkeypatch.setattr(memory, "forward_probe", lambda *args: calls.append(args) or {"status": "completed"})
    memory.main(["forward", "--output", str(tmp_path), "--checkpoint", "frozen.json", "--compression", "tighter"])
    assert calls[-1][1:] == (Path("frozen.json"), "tighter", None)
    monkeypatch.setattr(memory, "isolated_probe", lambda *args: calls.append(args) or {})
    memory.main(["isolated", "--output", str(tmp_path), "--capture", "saved"])
    assert calls[-1][-1] == Path("saved")
    monkeypatch.setattr(memory, "supervise", lambda *args: calls.append(args) or {"status": "memory_limit"})
    with pytest.raises(SystemExit):
        memory.main(["supervise", "--output", str(tmp_path), "--", "python", "worker.py", "--output", "child"])
    assert calls[-1][1] == ["python", "worker.py", "--output", "child"]
    with pytest.raises(SystemExit):
        memory.main(["forward", "--output", str(tmp_path)])
    with pytest.raises(SystemExit):
        memory.main(["forward", "--output", str(tmp_path), "--unknown"])


def test_isolated_backward_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The isolated author derivative runs with dense cotangents and authentic input."""
    monkeypatch.setattr(memory.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3))
    capture, output = tmp_path / "capture", tmp_path / "backward"
    capture.mkdir()
    output.mkdir()
    matrix = torch.tensor(np.random.default_rng(711).normal(size=(8, 4)), dtype=torch.complex128)
    recorder = memory.SVDRecorder(capture, 1)
    with pytest.raises(memory.MatrixCapturedError), recorder.active():
        torch.linalg.svd(matrix, full_matrices=False)
    result = backward.backward_probe(output, capture)
    assert result["gradient_finite"]
    assert result["gradient_bytes"] == matrix.numel() * matrix.element_size()
    assert result["cotangent_bytes"] == result["factor_bytes"]
    assert result["circuit_gradient_computed"] is False
    called = []
    monkeypatch.setattr(backward, "backward_probe", lambda *args: called.append(args) or {})
    backward.main(["--output", str(output), "--capture", str(capture)])
    assert called[-1] == (output, capture)


def test_isolated_backward_rejects_corrupt_matrix(tmp_path: Path) -> None:
    """A stale capture receipt cannot be used for a different derivative input."""
    (tmp_path / "capture.json").write_text(json.dumps({"matrix_sha256": "invalid"}))
    np.save(tmp_path / "matrix.npy", np.eye(2))
    with pytest.raises(ValueError, match="hash"):
        backward.backward_probe(tmp_path, tmp_path)


def test_tensor_storage_counts_shared_backing_once() -> None:
    """Short overlapping views count their shared full allocation only once."""
    base = torch.zeros(100, dtype=torch.complex128)
    assert memory.tensor_storage([base[:2], base[2:5]]) == {
        "state_logical_bytes": 5 * 16,
        "state_unique_backing_bytes": 100 * 16,
    }
