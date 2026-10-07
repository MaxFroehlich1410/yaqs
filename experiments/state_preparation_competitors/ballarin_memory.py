# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Observe unchanged Ballarin forward SVDs and replay one exact input in isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import weakref
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import torch

from benchmarks.state_preparation.targets import load_target
from experiments.state_preparation_competitors.bounded_probe import run_bounded
from experiments.state_preparation_competitors.iqsp_comb_backend import CombOptions, CombTensorNetworkObjective
from experiments.state_preparation_competitors.iqsp_tci import tci_target_bank
from experiments.state_preparation_competitors.iqsp_tn import iqsp_tn_ansatz, logical_parameterization
from experiments.state_preparation_competitors.iqsp_tn_backend import CircuitTensorGates

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from numpy.typing import NDArray

    from experiments.state_preparation_competitors.vendor.ballarin_comb.combdevice import CombTNDevice
    from experiments.state_preparation_competitors.vendor.ballarin_comb.noisy_combdevice import NoisyCombTNDevice


def write_receipt(path: Path, value: dict) -> None:
    """Atomically save JSON so an interrupted call retains its last complete phase."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    """Hash an archived matrix without loading a second copy into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MatrixCapturedError(Exception):
    """Stop a prefix replay deliberately after preserving the requested SVD input."""


def tensor_storage(tensors: Sequence[torch.Tensor]) -> dict[str, int]:
    """Count logical tensor bytes and unique backing buffers without retaining views."""
    storage = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes() for tensor in tensors}
    return {
        "state_logical_bytes": sum(t.numel() * t.element_size() for t in tensors),
        "state_unique_backing_bytes": sum(storage.values()),
    }


class ObservedForwardObjective(CombTensorNetworkObjective):
    """Expose state-buffer sizes through a weak reference without extending its life."""

    observed_state: weakref.ReferenceType[CombTNDevice | NoisyCombTNDevice] | None = None

    def _new_state(self, *, reset_diagnostics: bool = True) -> CombTNDevice | NoisyCombTNDevice:
        """Create the unchanged state and remember only a weak observational reference."""
        state = super()._new_state(reset_diagnostics=reset_diagnostics)
        self.observed_state = weakref.ref(state)
        return state

    def storage_snapshot(self) -> dict[str, int]:
        """Return current state storage if it is still alive in the numerical call."""
        state = self.observed_state() if self.observed_state is not None else None
        return tensor_storage(state.states) if state is not None else {}


class SVDRecorder:
    """Tag each unchanged SVD call; optionally capture one exact input and stop."""

    def __init__(
        self,
        output: Path,
        capture_index: int | None = None,
        state_memory: Callable[[], dict[str, int]] | None = None,
    ) -> None:
        """Require a fresh trace file and a positive one-based capture index."""
        if capture_index is not None and (type(capture_index) is not int or capture_index < 1):
            msg = "capture_index must be a positive integer."
            raise ValueError(msg)
        self.output = output
        self.capture_index = capture_index
        self.state_memory = state_memory
        self.records: list[dict[str, Any]] = []
        self.original = torch.linalg.svd
        self.trace = output / "svd.jsonl"
        self.trace.touch(exist_ok=False)

    def phase(self, stage: str, **data: object) -> None:
        """Expose the active worker phase to the independent process monitor."""
        write_receipt(self.output / "phase.json", {"pid": os.getpid(), "stage": stage, **data})

    def event(self, kind: str, record: dict) -> None:
        """Append one small receipt without keeping any tensor alive."""
        with self.trace.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": kind, **record}) + "\n")

    def capture(self, matrix: torch.Tensor, record: dict) -> None:
        """Archive a single exact matrix only when a one-GiB disk reserve remains."""
        self.phase("capture", index=record["index"])
        if shutil.disk_usage(self.output).free < record["input_bytes"] + 1024**3:
            msg = "Insufficient free disk for the matrix plus a one-GiB reserve."
            raise OSError(msg)
        destination = self.output / "matrix.npy"
        np.save(destination, matrix.detach().cpu().resolve_conj().resolve_neg().numpy(), allow_pickle=False)
        write_receipt(
            self.output / "capture.json",
            {**record, "matrix_sha256": file_hash(destination), "file_bytes": destination.stat().st_size},
        )
        raise MatrixCapturedError

    def __call__(
        self, matrix: torch.Tensor, *, full_matrices: bool = True, driver: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the original SVD with identical arguments and return its original output."""
        record: dict[str, Any] = {
            "index": len(self.records) + 1,
            "shape": list(matrix.shape),
            "strides": list(matrix.stride()),
            "dtype": str(matrix.dtype),
            "input_bytes": matrix.numel() * matrix.element_size(),
            "backing_bytes": matrix.untyped_storage().nbytes(),
            "full_matrices": full_matrices,
            "driver": driver,
        }
        if self.state_memory is not None:
            record.update(self.state_memory())
        self.records.append(record)
        self.event("begin", record)
        if record["index"] == self.capture_index:
            self.capture(matrix, record)
        self.phase("svd", index=record["index"], shape=record["shape"])
        start = time.perf_counter()
        result = self.original(matrix, full_matrices=full_matrices, driver=driver)
        record.update(
            seconds=time.perf_counter() - start,
            output_bytes=sum(t.numel() * t.element_size() for t in result),
            singular_sha256=hashlib.sha256(result[1].detach().cpu().numpy().tobytes()).hexdigest(),
        )
        self.event("end", record)
        self.phase("outside_svd")
        return result

    @contextmanager
    def active(self) -> Iterator[SVDRecorder]:
        """Restore the numerical function even if a capture or a numerical error exits."""
        self.phase("outside_svd")
        with patch.object(torch.linalg, "svd", self):
            yield self


class PhaseObserver:
    """Attribute sampled process memory to call IDs and stop on low disk space."""

    def __init__(self, output: Path, minimum_free_bytes: int = 512 * 1024**2) -> None:
        """Initialize scalar summaries only; the monitor never reads worker tensors."""
        self.output = output
        self.minimum_free_bytes = minimum_free_bytes
        self.peaks: dict[str, dict[str, int]] = {}
        self.minimum_observed_free_bytes = shutil.disk_usage(output).free

    def __call__(self, pid: int, rss: int, physical: int) -> str | None:
        """Sample the phase around a memory read; fast calls can escape this sampling."""
        phase_file = self.output / "phase.json"
        phase = json.loads(phase_file.read_text()) if phase_file.exists() else {}
        stage = phase.get("stage", "startup") if phase.get("pid", pid) == pid else "startup"
        key = f"{stage}:{phase.get('index', 0)}"
        row = self.peaks.setdefault(key, {"samples": 0, "rss_bytes": 0, "physical_bytes": 0})
        row["samples"] += 1
        row["rss_bytes"] = max(row["rss_bytes"], rss)
        row["physical_bytes"] = max(row["physical_bytes"], physical)
        free = shutil.disk_usage(self.output).free
        self.minimum_observed_free_bytes = min(self.minimum_observed_free_bytes, free)
        return "disk_limit" if free < self.minimum_free_bytes else None


def prepare_forward(checkpoint: Path, compression: str) -> tuple[ObservedForwardObjective, NDArray, dict]:
    """Rebuild the frozen Gaussian circuit and TCI target using the audit's settings."""
    if compression not in {"default", "tighter"}:
        msg = "compression must be default or tighter."
        raise ValueError(msg)
    source = json.loads(checkpoint.read_text(encoding="utf-8"))
    n, depth = source["num_qubits"], source["depth"]
    target = load_target(n, "gaussian_mu0p5_sigma0p1").state_vector
    targets, construction = tci_target_bank([target], gaussian_coordinates=[1.0])
    circuit = CircuitTensorGates(iqsp_tn_ansatz(n, depth)).bind(source["final_theta"])
    template, theta, lowering = logical_parameterization(circuit)
    settings: dict[str, Any] = {"max_tensor_elements": 2**26, "checkpoint_interval": 1}
    if compression == "tighter":
        settings.update(closed_bond=128, environment_bond=32, closed_tolerance=1e-11, environment_tolerance=1e-14)
    options = CombOptions(**settings)
    return (
        ObservedForwardObjective(template, targets[0], "IBM", options=options),
        theta,
        {
            "num_qubits": n,
            "depth": depth,
            "compression": asdict(options),
            "hardware": "IBM",
            "noise_basis": "logical",
            "checkpoint_sha256": file_hash(checkpoint),
            "theta_sha256": hashlib.sha256(theta.tobytes()).hexdigest(),
            "target_construction": construction,
            "logical_lowering": lowering,
        },
    )


def forward_probe(output: Path, checkpoint: Path, compression: str, capture_index: int | None = None) -> dict:
    """Measure forward fidelity with no gradient tape, optionally stopping at one SVD."""
    objective, theta, metadata = prepare_forward(checkpoint, compression)
    write_receipt(output / "inputs.json", metadata)
    recorder = SVDRecorder(output, capture_index, objective.storage_snapshot)
    start = time.perf_counter()
    try:
        with recorder.active():
            fidelity = objective.fidelity(theta)
    except MatrixCapturedError:
        result = {**metadata, "status": "matrix_captured", "forward_complete": False}
    else:
        if capture_index is not None:
            msg = "Requested SVD index was not reached."
            raise ValueError(msg)
        result = {**metadata, "status": "completed", "fidelity": fidelity, "diagnostics": objective.last_diagnostics}
    result.update(wall_seconds=time.perf_counter() - start, calls=len(recorder.records), gradient_computed=False)
    write_receipt(output / "result.json", result)
    return result


def isolated_probe(output: Path, capture: Path) -> dict:
    """Replay the captured matrix in a fresh worker with the same thin SVD call."""
    metadata = json.loads((capture / "capture.json").read_text())
    matrix_path = capture / "matrix.npy"
    if file_hash(matrix_path) != metadata["matrix_sha256"]:
        msg = "Captured matrix hash does not match."
        raise ValueError(msg)
    matrix = torch.from_numpy(np.load(matrix_path, allow_pickle=False))
    if list(matrix.shape) != metadata["shape"] or str(matrix.dtype) != metadata["dtype"]:
        msg = "Captured matrix shape or dtype does not match."
        raise ValueError(msg)
    recorder = SVDRecorder(output)
    with torch.no_grad(), recorder.active():
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=metadata["full_matrices"])
    result = {
        "source_capture": str(capture),
        "source_matrix_sha256": metadata["matrix_sha256"],
        "input_bytes": matrix.numel() * matrix.element_size(),
        "factor_bytes": sum(t.numel() * t.element_size() for t in (u, singular, vh)),
        "singular_values_finite": bool(torch.isfinite(singular).all()),
        "singular_count": singular.numel(),
        "gradient_computed": False,
        "source_strides": metadata["strides"],
        "loaded_strides": list(matrix.stride()),
        "layout_note": "NPY preserves values and dtype; loading may use a different memory layout.",
    }
    write_receipt(output / "result.json", result)
    return result


def supervise(output: Path, command: list[str]) -> dict:
    """Use the existing process guard plus phase sampling and a disk-space floor."""
    observer = PhaseObserver(output)
    result = run_bounded(command, output / "worker.log", observer=observer)
    write_receipt(
        output / "phase_peaks.json",
        {
            "peaks": observer.peaks,
            "sampling_seconds": 0.25,
            "minimum_observed_free_bytes": observer.minimum_observed_free_bytes,
            "minimum_free_bytes": observer.minimum_free_bytes,
            "caveat": "Coarse sampled total process memory, not exact allocation ownership or an incremental SVD cost.",
        },
    )
    return result


def main(argv: list[str] | None = None) -> None:
    """Run a single diagnostic worker or supervise its explicit argument vector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("forward", "isolated", "supervise"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--compression", choices=("default", "tighter"), default="default")
    parser.add_argument("--capture-index", type=int)
    parser.add_argument("--capture", type=Path)
    args, command = parser.parse_known_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "supervise":
        result = supervise(args.output, command[1:] if command[:1] == ["--"] else command)
    elif command:
        parser.error("Unexpected arguments")
    elif args.mode == "forward" and args.checkpoint is not None:
        result = forward_probe(args.output, args.checkpoint, args.compression, args.capture_index)
    elif args.mode == "isolated" and args.capture is not None:
        result = isolated_probe(args.output, args.capture)
    else:
        parser.error("Forward requires --checkpoint; isolated requires --capture")
    print(json.dumps(result), flush=True)
    if result.get("status", "completed") not in {"completed", "matrix_captured"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
