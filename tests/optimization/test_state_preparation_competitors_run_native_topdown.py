# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Native pilot budgets, provenance, disjoint validation and locked selection."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from qiskit import QuantumCircuit

from experiments.state_preparation_competitors import run_native_topdown as study
from experiments.state_preparation_competitors.common import Candidate
from experiments.state_preparation_pilot import circuit_record
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def test_parser_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain frozen budgets and reject unsupported numerical environments."""
    args = study.make_parser().parse_args(["--output", str(tmp_path)])
    monkeypatch.setattr(study.importlib.metadata, "version", lambda name: study.PINNED[name])
    manifest = study.build_manifest(args)
    assert manifest["targets"] == list(study.TARGETS)
    assert manifest["protocol"]["validation_paths"] == 32768
    assert manifest["protocol"]["patience"] == 3
    assert manifest["protocol"]["fine_tuning"]["num_trajectories"] == 100
    assert "final iterate" in manifest["noisy_optimizer"]
    monkeypatch.setattr(study.importlib.metadata, "version", lambda _name: "invalid")
    with pytest.raises(ValueError, match="pinned"):
        study.build_manifest(args)


@pytest.mark.parametrize("targets", [[], [study.TARGETS[0]] * 2, ["missing"]])
def test_bad_targets(tmp_path: Path, targets: list[str]) -> None:
    """Unsupported target lists cannot create an output campaign."""
    args = study.make_parser().parse_args(["--output", str(tmp_path)])
    args.targets = targets
    with pytest.raises(ValueError, match="unique"):
        study.build_manifest(args)


def test_reference_record_hash(tmp_path: Path) -> None:
    """Changed reusable checkpoints cannot silently alter reproducibility."""
    path = tmp_path / "record.json"
    study.storage.write_json(path, {"theta": [1.0]})
    receipt = {"path": str(path), "sha256": study.storage.digest(path)}
    assert study.reference_record(receipt) == {"theta": [1.0]}
    study.storage.write_json(path, {})
    with pytest.raises(ValueError, match="changed"):
        study.reference_record(receipt)


def test_validation_unconditional_native_bank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate on an independent role stream with saved raw jump codes."""
    monkeypatch.setitem(study.depth_search.PROTOCOL, "validation_paths", 2048)
    circuit = QuantumCircuit(2)
    circuit.sx(0)
    circuit.cx(0, 1)
    target = np.array([1, 0, 0, 0], dtype=complex)
    candidate = Candidate(circuit, {"preserve_native_schedule": True})
    score = study.validate_candidate(tmp_path, candidate, target, study.TARGETS[0], 1)
    assert score["entropy"][-1] == 91000
    assert score["sampled_paths"] == 2048
    assert score["jump_paths"] > 0
    assert score["no_jump_paths"] > 0
    assert score["bank_sha256"] == study.storage.digest(tmp_path / "validation.npz")
    assert score["resources"]["compiled_operations"] == {"sx": 1, "cx": 1}
    with np.load(tmp_path / "validation.npz") as saved:
        codes = saved["codes"]
    assert codes.shape == (2048, 2)
    assert score["raw_bank_sha256"] == hashlib.sha256(codes.astype(np.uint8).tobytes()).hexdigest()


def test_checkpoint_reuse_and_extension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Noise-independent caching cannot cap the new native depth search."""
    calls = []
    built: dict[str, Any] = {"depth": 1, "policy": "lbfgs", "theta": [0.2]}
    pruned: dict[str, Any] = {"depth": 1, "pruning": {"variant": "B"}}
    monkeypatch.setattr(
        study.reference,
        "authenticated_checkpoint",
        lambda _case, stem, _artifacts: built if "construction" in stem else pruned,
    )
    artifacts = {"search/construction_01.json": "a", "search/depth_01_B_pruned.json": "b"}
    got = study.checkpoint_inputs(tmp_path, artifacts, tmp_path, np.ones(2), 1, np.empty(0))
    assert got[:2] == (built, pruned)

    def construct(
        root: Path, _target: np.ndarray, seed: int, depth: int, previous: np.ndarray, policy: str
    ) -> dict[str, Any]:
        """Return a saved construction while recording lineage-extension arguments."""
        calls.append((seed, depth, previous.tolist(), policy))
        study.storage.write_json(root / "construction_01.json", built)
        return built

    def prune(root: Path, _target: np.ndarray, construction: dict[str, Any], variant: str) -> dict[str, Any]:
        """Return a saved B pruning after checking the fresh construction input."""
        assert construction == built
        assert variant == "B"
        study.storage.write_json(root / "depth_01_B_pruned.json", pruned)
        return pruned

    monkeypatch.setattr(study.batch, "construct", construct)
    monkeypatch.setattr(study.depth_search, "prune_checkpoint", prune)
    got = study.checkpoint_inputs(tmp_path, {}, tmp_path, np.ones(2), 1, np.array([0.7]))
    assert got[:2] == (built, pruned)
    assert calls == [(300, 1, [0.7], "lbfgs")]
    assert study.reference_record(got[2]["construction"]) == built
    built["policy"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        study.checkpoint_inputs(tmp_path, artifacts, tmp_path, np.ones(2), 1, np.empty(0))


def test_run_depth_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A resumed depth reads its immutable result and rejects modified banks."""

    class Native:
        """Provide the compiled-input boundary without invoking an optimizer."""

        @staticmethod
        def bind(_theta: object) -> QuantumCircuit:
            """Return a resolved empty single-qubit circuit."""
            return QuantumCircuit(1)

    target = np.array([1, 0], dtype=complex)
    pruned = {"depth": 1, "theta": [], "circuit": {"num_qubits": 1, "num_params": 0, "gates": []}}
    metadata = {"status": "complete", "preserve_native_schedule": True, "initial_theta": [], "final_theta": []}
    monkeypatch.setattr(study.native_topdown, "train_candidate", lambda *_args: Candidate(QuantumCircuit(1), metadata))
    monkeypatch.setattr(study.native_topdown, "compile_training", lambda *_args: Native())
    monkeypatch.setitem(study.depth_search.PROTOCOL, "validation_paths", 2)
    first = study.run_depth(tmp_path, pruned, target, study.TARGETS[0], {"fixture": "same"})
    monkeypatch.setattr(study.native_topdown, "train_candidate", lambda *_args: pytest.fail("reran training"))
    assert study.run_depth(tmp_path, pruned, target, study.TARGETS[0], {"fixture": "same"}) == first
    with pytest.raises(ValueError, match="changed"):
        study.run_depth(tmp_path, pruned, target, study.TARGETS[0], {"fixture": "other"})
    (tmp_path / "validation.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        study.run_depth(tmp_path, pruned, target, study.TARGETS[0], {"fixture": "same"})


def test_run_depth_zero_jump_bank_is_failed_without_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero-jump training bank preserves initial parameters without resampling."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("rz", (0,), param_index=0)])
    pruned = {"depth": 1, "theta": [0.7], "circuit": circuit_record(circuit)}
    target = np.array([1, 0], dtype=complex)
    calls = []

    def train(*_args: object) -> Candidate:
        """Record the sole training attempt and reject its unconditional bank.

        Raises:
            RuntimeError: The sampled training bank contains no jumps.
        """
        calls.append(1)
        msg = "The sampled bank has no jumps; keep the failed bank."
        raise RuntimeError(msg)

    monkeypatch.setattr(study.native_topdown, "train_candidate", train)
    monkeypatch.setitem(study.depth_search.PROTOCOL, "validation_paths", 2)
    first = study.run_depth(tmp_path, pruned, target, study.TARGETS[0], {"fixture": "same"})
    metadata = first["candidate"]
    assert calls == [1]
    assert metadata["status"] == "no_jump_training_bank"
    assert metadata["training"] is None
    assert metadata["initial_theta"] == metadata["final_theta"] == [0.7]
    assert metadata["reason"] == "The sampled bank has no jumps; keep the failed bank."
    assert metadata["preserve_native_schedule"] is True
    _, circuits = study.storage.load_bundle(tmp_path)
    assert circuits[0] == circuits[1]
    assert circuits[0].data[0].operation.params == pytest.approx([0.7])
    assert study.run_depth(tmp_path, pruned, target, study.TARGETS[0], {"fixture": "same"}) == first
    assert calls == [1]


def test_run_depth_unrelated_training_error_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected optimizer failures cannot be mislabeled as a zero-jump bank."""
    pruned = {"depth": 1, "theta": [], "circuit": {"num_qubits": 1, "num_params": 0, "gates": []}}

    def train(*_args: object) -> Candidate:
        """Raise an unrelated numerical failure at the training boundary.

        Raises:
            RuntimeError: The optimizer encountered a nonfinite gradient.
        """
        msg = "Optimizer produced a nonfinite gradient."
        raise RuntimeError(msg)

    monkeypatch.setattr(study.native_topdown, "train_candidate", train)
    monkeypatch.setattr(study.native_topdown, "compile_training", lambda *_args: pytest.fail("compiled after failure"))
    monkeypatch.setattr(study, "validate_candidate", lambda *_args: pytest.fail("validated after failure"))
    with pytest.raises(RuntimeError, match="nonfinite gradient"):
        study.run_depth(tmp_path, pruned, np.array([1, 0], dtype=complex), study.TARGETS[0], {})
    assert not (tmp_path / "seal.json").exists()


def test_run_target_lock_before_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exact scores cannot select depth or halt growth; all visited costs count."""
    target_id = study.TARGETS[0]
    source = tmp_path / "archive/pipeline/lbfgs" / f"{target_id}__seed300"
    study.storage.write_json(source / "complete.json", {"artifacts": {}})
    monkeypatch.setattr(study.reference, "frozen_reference", lambda *_args: None)
    logical = {"num_qubits": 6, "num_params": 0, "gates": []}
    input_path = tmp_path / "pruned.json"
    study.storage.write_json(input_path, {"circuit": logical})
    receipt = {"path": str(input_path), "sha256": study.storage.digest(input_path)}

    def inputs(
        _source_case: Path,
        _artifacts: dict[str, str],
        _output: Path,
        _target: np.ndarray,
        depth: int,
        _previous: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Return fixed-cost construction and pruning records at each visited depth."""
        return (
            {"theta": [], "cpu_seconds": 1.0},
            {"depth": depth, "circuit": logical, "pruning": {"cpu_seconds": 2.0}},
            {"pruning": receipt},
        )

    monkeypatch.setattr(study, "checkpoint_inputs", inputs)

    def run_depth(
        folder: Path, pruned: dict[str, Any], _target: np.ndarray, _target_id: str, sources: dict[str, Any]
    ) -> dict[str, Any]:
        """Return declining validation scores with a constant charged training cost."""
        depth = pruned["depth"]
        record = {
            "sources": sources,
            "candidate": {
                "status": "complete",
                "cpu_seconds": 3.0,
                "preserve_native_schedule": True,
                "final_theta": [],
                "training": {"noise_banks": [{}]},
            },
            "validation": {"fidelity": 0.9 - depth * 0.01, "standard_error": 0.0, "jump_paths": 1, "cpu_seconds": 4.0},
        }
        study.storage.save_bundle(folder, record, [QuantumCircuit(6), QuantumCircuit(6)])
        return record

    monkeypatch.setattr(study, "run_depth", run_depth)

    class Native:
        """Expose the native-binding interface for the independent final-stage check."""

        @staticmethod
        def bind(_theta: object) -> QuantumCircuit:
            """Return the synthetic six-qubit compiled input."""
            return QuantumCircuit(6)

    monkeypatch.setattr(study.native_topdown, "compile_training", lambda *_args: Native())
    exact_calls = []

    def exact(_candidate: object, _hardware: str) -> dict[str, float]:
        """Return improving test scores only after the validation winner is locked."""
        lock = json.loads((tmp_path / target_id / "selection_locked.json").read_text())
        assert lock["selected_depth"] == 1
        assert len(lock["decisions"]) == 4
        exact_calls.append(1)
        return {"fidelity": len(exact_calls) / 10}

    monkeypatch.setattr(study.evaluation, "exact_score", exact)
    result = study.run_target(tmp_path, tmp_path / "archive", target_id)
    assert result["selected_depth"] == 1
    assert result["selected"]["exact"]["fidelity"] == pytest.approx(0.1)
    assert result["build_train_cpu_seconds"] == pytest.approx(24.0)
    assert result["total_search_cpu_seconds"] == pytest.approx(40.0)
    assert len(exact_calls) == 8


def test_run_manifest_guard_and_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Campaign entry point preserves sources and refuses different resumes."""
    args = study.make_parser().parse_args(["--output", str(tmp_path)])
    monkeypatch.setattr(study, "build_manifest", lambda _args: {"sources": {}, "test": 1})
    monkeypatch.setattr(study, "run_target", lambda *_args: {"selected_depth": 1})
    result = study.run(args)
    assert len(result["results"]) == 3
    assert (tmp_path / "seal.json").exists()
    monkeypatch.setattr(study, "build_manifest", lambda _args: {"sources": {}, "test": 2})
    with pytest.raises(ValueError, match="manifest"):
        study.run(args)
    invoked = []
    monkeypatch.setattr(study, "run", lambda args: invoked.append(args.output))
    study.main(["--output", str(tmp_path)])
    assert invoked == [tmp_path]
