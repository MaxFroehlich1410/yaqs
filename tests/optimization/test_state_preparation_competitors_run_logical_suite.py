# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Fixed logical-suite budgets, authenticated reuse and isolated execution."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from qiskit import QuantumCircuit

from experiments.state_preparation_competitors import run_logical_suite as study
from experiments.state_preparation_competitors.common import Candidate

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterator
    from pathlib import Path


def _args(output: Path) -> argparse.Namespace:
    """Return the suite's unchanged numerical and assessment budgets."""
    return study.make_parser().parse_args(["--output", str(output)])


def _candidate(method: str, depth: int = 2, qubits: int = 2) -> Candidate:
    """Build a tiny source circuit with a retained zero-angle gate.

    Returns:
        Source circuit and minimal matching training metadata.
    """
    circuit = QuantumCircuit(qubits)
    circuit.ry(depth / 10, 0)
    circuit.rz(0, 0)
    return Candidate(
        circuit,
        {
            "method": method,
            "candidate_id": f"depth_{depth:02d}",
            "status": "complete",
            "seed": 300,
            "hardware": "IBM",
            "depth": depth,
            "noise_basis": "logical",
            "training": {"noiseless_steps": 500, "noisy_steps": 50, "feedback": "exact_density"},
        },
    )


def _sealed_group(root: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Create a valid old group whose exact winner differs from its locked winner.

    Returns:
        Summary group and the matching normalized target.
    """
    target = np.array([1, 0, 0, 0], dtype=complex)
    target_id, method = study.pilot.TARGETS[0], "robuststate"
    relative = f"{target_id}/{method}/300/IBM"
    construction_path = "construction/" + relative
    candidates = [_candidate(method, depth) for depth in (2, 4)]
    circuits = [item.circuit for item in candidates]
    study.storage.save_bundle(
        root / construction_path,
        {"candidates": [item.metadata for item in candidates], "construction_cpu_seconds": 12.5},
        circuits,
    )
    construction_hash = study.storage.digest(root / construction_path / "seal.json")
    records = [
        {
            "method": {
                "candidate_id": item.metadata["candidate_id"],
                "candidate_index": index,
                "construction_reference": {"path": construction_path, "seal_sha256": construction_hash},
            },
            "validation": {"fidelity": 0.9 - index / 10},
        }
        for index, item in enumerate(candidates)
    ]
    locked = {
        "context": {
            "identity": {"target_id": target_id, "method": method, "seed": 300, "hardware": "IBM"},
            "noise_basis": "logical",
            "validation_paths": 8192,
            "construction_receipt_sha256": construction_hash,
            "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        },
        "records": records,
        "selected_index": 0,
    }
    evaluation = root / "evaluation" / relative
    study.storage.save_bundle(evaluation / "selection_locked", locked, circuits)
    final = {
        **locked,
        "selection_lock_sha256": study.storage.digest(evaluation / "selection_locked/seal.json"),
        "final_scores": [
            {"candidate_id": item.metadata["candidate_id"], "exact": {"fidelity": value}}
            for item, value in zip(candidates, (0.3, 0.99), strict=True)
        ],
        "status": "complete",
    }
    study.storage.save_bundle(evaluation / "final", final, circuits)
    group = {
        **final,
        "construction_cache": construction_path,
        "build_train_cpu_seconds": 12.5,
        "new_build_train_cpu_seconds": 12.5,
        "training_reused": False,
    }
    study.storage.write_json(root / "manifest.json", {"original": "immutable"})
    study.storage.write_json(root / "summary.json", {"groups": [group]})
    return group, target


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("targets", []),
        ("targets", ["unknown"]),
        ("targets", ["tfim_para", "tfim_para"]),
        ("methods", []),
        ("methods", ["aqce"]),
        ("methods", ["nacl", "nacl"]),
        ("validation_paths", 100),
        ("validation_paths", 8192.0),
        ("validation_paths", True),
        ("workers", 0),
        ("workers", 4),
        ("workers", True),
    ],
)
def test_invalid_config_has_no_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: object
) -> None:
    """Reject altered scope or budgets before reading archives or making output."""
    args = _args(tmp_path / "out")
    setattr(args, key, value)
    monkeypatch.setattr(study, "authenticate_pilot", lambda _args: pytest.fail("archive reached"))
    with pytest.raises(ValueError, match=r"unique|8192|workers"):
        study.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize("source", ["pilot_root", "reference_root"])
def test_output_cannot_overwrite_source_archive(tmp_path: Path, source: str) -> None:
    """The suite cannot use either immutable input archive as its output root."""
    args = _args(tmp_path)
    setattr(args, source, tmp_path)
    with pytest.raises(ValueError, match="separate"):
        study.validate_config(args)


def test_default_scope_and_nomad_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """New NACL work requires its actual upstream package in the manifest."""
    args = _args(tmp_path / "out")
    args.pilot_root = tmp_path / "old"
    study.storage.write_json(args.pilot_root / "manifest.json", {})
    study.storage.write_json(args.pilot_root / "summary.json", {})
    versions = {**study.PINNED, "PyNomadBBO": "4.6.0"}
    monkeypatch.setattr(study.importlib.metadata, "version", versions.__getitem__)
    manifest = study.build_manifest(args)
    assert args.targets == list(study.TARGET_IDS)
    assert args.methods == list(study.METHODS)
    assert args.validation_paths == 8192
    assert args.workers == 3
    assert manifest["packages"] == versions
    assert manifest["optimizer_seed"] == 300
    assert manifest["hardware"] == "IBM"
    assert manifest["numeric_threads_per_worker"] == 1
    assert "output" not in manifest["config"]
    assert manifest["sources"]["experiments/state_preparation_competitors/run_logical_suite.py"]
    versions["PyNomadBBO"] = "wrong"
    with pytest.raises(ValueError, match="pins"):
        study.build_manifest(args)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (
            "nacl",
            {"search_steps": (0, 10, 50), "inner_evaluations": 100, "restarts": 1, "max_native_gates": 300},
        ),
        (
            "fixed_vqsp",
            {"depths": (2, 4), "construction_steps": 200, "noisy_steps": 200, "training_paths": 100},
        ),
        (
            "layerwise",
            {
                "depths": (1, 2, 3, 4),
                "construction_steps": 200,
                "online_steps": 100,
                "noisy_steps": 200,
                "noisy_online_steps": 100,
                "training_paths": 100,
            },
        ),
    ],
)
def test_new_factories_preserve_every_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, expected: dict[str, Any]
) -> None:
    """Missing controls and NACL receive exactly the original pilot settings."""
    target = np.array([1, 0, 0, 0], dtype=complex)
    candidates = [_candidate(method)]
    calls = []

    def factory(vector: np.ndarray, **kwargs: object) -> list[Candidate]:
        np.testing.assert_array_equal(vector, target)
        calls.append(kwargs)
        return candidates

    module = study.nacl if method == "nacl" else study.controls
    monkeypatch.setattr(module, method + "_candidates", factory)
    assert study.construct(_args(tmp_path), "tfim_para", method, target) is candidates
    assert calls == [{"hardware": "IBM", "seed": 300, **expected}]


@pytest.mark.parametrize("method", ["robuststate", "fontana"])
def test_paper_factory_dispatch_reuses_original_logical_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Keep the original logical noise-basis and paper optimizer dispatch."""
    args, target, calls = _args(tmp_path), np.array([1, 0], dtype=complex), []
    candidates = [_candidate(method)]

    def factory(*values: object) -> list[Candidate]:
        calls.append(values)
        return candidates

    monkeypatch.setattr(study.pilot, "construct", factory)
    assert study.construct(args, "tfim_para", method, target) is candidates
    assert calls == [(args, "tfim_para", method, target)]


def test_frozen_baseline_charges_full_history_without_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An imported baseline keeps its circuit and includes all discarded search work."""
    candidate = _candidate("current_b_lbfgs_frozen")
    candidate.metadata["archived_search_cost"] = {
        "construction_cpu_seconds": 2.0,
        "pruning_cpu_seconds": 0.5,
        "fine_tuning_cpu_seconds": 3.0,
        "validation_cpu_seconds": 0.75,
    }
    calls = []

    def frozen(*values: object) -> Candidate:
        calls.append(values)
        return candidate

    monkeypatch.setattr(study.reference, "frozen_reference", frozen)
    args, target = _args(tmp_path), np.array([1, 0], dtype=complex)
    result = study.construct(args, "tfim_para", "current_b_lbfgs_frozen", target)
    assert result == [candidate]
    assert calls == [(args.reference_root, "tfim_para", 300, "IBM", target)]
    assert candidate.metadata["historical_build_train_cpu_seconds"] == pytest.approx(5.5)
    assert candidate.metadata["historical_selection_cpu_seconds"] == pytest.approx(0.75)
    assert candidate.metadata["training_reused"] is True
    assert candidate.circuit.count_ops()["rz"] == 1
    with pytest.raises(ValueError, match="Unsupported"):
        study.construct(args, "tfim_para", "aqce", target)


def test_source_guards_reject_changes_and_unsafe_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Authenticating an old source cannot silently accept edits or path traversal."""
    monkeypatch.setattr(study.storage, "ROOT", tmp_path)
    source = tmp_path / "model.py"
    source.write_text("x = 1\n")
    manifest = {"sources": {"model.py": study.storage.digest(source)}}
    study.check_sources(manifest)
    source.write_text("x = 2\n")
    with pytest.raises(ValueError, match="source changed"):
        study.check_sources(manifest)
    with pytest.raises(ValueError, match="missing source"):
        study.check_sources({"sources": {}})
    with pytest.raises(ValueError, match="Unsupported"):
        study.check_sources(manifest, allowed={"model.py"})
    with pytest.raises(ValueError, match=r"Unsafe|relative|escape|traversal|archive"):
        study.check_sources({"sources": {"../model.py": "wrong"}})


def test_authenticated_group_never_reevaluates_or_reselects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain a validation winner even when its sealed exact score is worse."""
    group, target = _sealed_group(tmp_path / "old")
    before = copy.deepcopy(group)
    monkeypatch.setattr(study.pilot, "evaluate_group", lambda *_args, **_kwargs: pytest.fail("reevaluated"))
    monkeypatch.setattr(study.storage, "select_candidate", lambda *_args: pytest.fail("reselected"))
    # Test the sealed-group boundary directly, without fabricating the complete 17-group campaign.
    result = study._authenticate_group(tmp_path / "old", group, _args(tmp_path / "new"), target)  # noqa: SLF001
    assert group == before
    assert result["selected_index"] == 0
    assert result["final_scores"] == group["final_scores"]
    assert result["final_scores"][1]["exact"]["fidelity"] == pytest.approx(0.99)
    assert result["reused_evaluation"] is True
    assert result["training_reused"] is True
    assert result["new_build_train_cpu_seconds"] == 0
    assert result["build_train_cpu_seconds"] == pytest.approx(12.5)
    assert result["source_output"] == str((tmp_path / "old").resolve())


@pytest.mark.parametrize("field", ["selected_index", "build_train_cpu_seconds", "construction_cache"])
def test_group_authentication_rejects_summary_tampering(tmp_path: Path, field: str) -> None:
    """Unsealed summary fields must agree with authenticated source receipts."""
    group, target = _sealed_group(tmp_path / "old")
    group[field] = "changed" if field == "construction_cache" else 1
    with pytest.raises(ValueError, match=r"changed|differs"):
        # Exercise the private integrity boundary with a deliberately tampered summary.
        study._authenticate_group(tmp_path / "old", group, _args(tmp_path / "new"), target)  # noqa: SLF001


def test_group_authentication_rejects_changed_qpy(tmp_path: Path) -> None:
    """Construction circuit bytes are authenticated before accepting cached results."""
    group, target = _sealed_group(tmp_path / "old")
    folder = tmp_path / "old" / group["construction_cache"]
    (folder / "circuits.qpy").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        # Exercise the private integrity boundary before it can load invalid QPY bytes.
        study._authenticate_group(tmp_path / "old", group, _args(tmp_path / "new"), target)  # noqa: SLF001


def _nested_source(tmp_path: Path) -> tuple[dict[str, Any], QuantumCircuit, Path]:
    """Create two sealed provenance levels with one original binary history.

    Returns:
        Outer candidate metadata, its original circuit, and the history archive.
    """
    source_name = "experiments/state_preparation_competitors/controls.py"
    manifest = {"sources": {source_name: study.storage.digest(study.storage.ROOT / source_name)}}
    candidate = _candidate("nacl")
    metadata = {
        **candidate.metadata,
        "diagnostic": {"records": [{"theta": [0.1], "fidelity": 0.4}, {"theta": [0.2], "fidelity": 0.5}]},
    }
    for level in ("leaf", "middle"):
        root = tmp_path / level
        study.storage.write_json(root / "manifest.json", manifest)
        folder = root / "construction/source"
        study.storage.save_bundle(folder, {"candidates": [metadata]}, [candidate.circuit])
        metadata = {
            **{key: candidate.metadata[key] for key in ("method", "candidate_id", "status")},
            "source_candidate": {
                "output": str(root.resolve()),
                "construction_path": "construction/source",
                "candidate_index": 0,
                "manifest_sha256": study.storage.digest(root / "manifest.json"),
                "construction_seal_sha256": study.storage.digest(folder / "seal.json"),
                "unrelated_changed_sources": [],
            },
        }
    return metadata, candidate.circuit, tmp_path / "leaf/construction/source/histories.npz"


def test_recursive_provenance_preserves_single_history_copy(tmp_path: Path) -> None:
    """Traverse nested cache references without loading an optimizer or copying arrays."""
    metadata, circuit, archive = _nested_source(tmp_path)
    assert archive.exists()
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    study.authenticate_source_candidate(metadata, circuit)
    after = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after
    assert len(list(tmp_path.rglob("histories.npz"))) == 1


@pytest.mark.parametrize("tamper", ["manifest", "seal", "circuit", "index", "identity", "path", "cycle"])
def test_recursive_provenance_rejects_changed_ancestry(tmp_path: Path, tamper: str) -> None:
    """Reject altered nested bytes, circuit identities, indices and unsafe ancestry."""
    metadata, circuit, _archive = _nested_source(tmp_path)
    pointer = metadata["source_candidate"]
    seen = frozenset()
    if tamper == "manifest":
        study.storage.write_json(tmp_path / "leaf/manifest.json", {})
    elif tamper == "seal":
        study.storage.write_json(tmp_path / "middle/construction/source/seal.json", {})
    elif tamper == "circuit":
        circuit.x(0)
    elif tamper == "index":
        pointer["candidate_index"] = 1
    elif tamper == "identity":
        metadata["status"] = "failed"
    elif tamper == "path":
        pointer["construction_path"] = "../outside"
    else:
        seen = frozenset({
            (str((tmp_path / "middle").resolve()), str((tmp_path / "middle/construction/source").resolve()), 0)
        })
    with pytest.raises(ValueError, match=r"changed|differs|index|identity|Circular|Unsafe|relative|escape|archive"):
        study.authenticate_source_candidate(metadata, circuit, seen=seen)


def test_target_resume_keeps_circuits_and_never_repeats_training_or_assessment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sealed new-target construction, validation and exact results resume intact."""
    args = _args(tmp_path / "out")
    args.methods = ["robuststate"]
    calls = {"train": 0, "validation": 0, "exact": 0}

    def factory(*_args: object) -> list[Candidate]:
        calls["train"] += 1
        return [_candidate("robuststate", depth, 6) for depth in (2, 4)]

    def validation(circuit: QuantumCircuit, *_args: object) -> dict[str, Any]:
        calls["validation"] += 1
        assert circuit.count_ops().get("rz") == 1
        return {"fidelity": 0.9 - float(circuit.data[0].operation.params[0]) / 10, "jump_paths": 1}

    def exact(_circuit: QuantumCircuit, *_args: object) -> dict[str, Any]:
        calls["exact"] += 1
        assert (args.output / "evaluation/tfim_para/robuststate/300/IBM/selection_locked/seal.json").exists()
        return {"fidelity": 0.8}

    monkeypatch.setattr(study, "construct", factory)
    monkeypatch.setattr(study.pilot.logical_evaluation, "score_validation", validation)
    monkeypatch.setattr(study.pilot.logical_evaluation, "score_exact", exact)
    first = study.run_target(args, "tfim_para")
    assert calls == {"train": 1, "validation": 2, "exact": 2}
    second = study.run_target(args, "tfim_para")
    assert calls == {"train": 1, "validation": 2, "exact": 2}
    assert first["groups"] == second["groups"]
    assert second["groups"][0]["new_build_train_cpu_seconds"] > 0


def test_resumed_baseline_reauthenticates_before_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An altered baseline archive is rejected before any assessment can run."""
    args = _args(tmp_path / "out")
    args.methods = ["current_b_lbfgs_frozen"]
    saved = _candidate("current_b_lbfgs_frozen", qubits=6)
    saved.metadata.update({
        "theta_sha256": "old",
        "archive_checkpoint_sha256": "checkpoint",
        "archive_selection_sha256": "selection",
    })
    fresh = Candidate(saved.circuit, {**saved.metadata, "theta_sha256": "changed"})
    monkeypatch.setattr(study.storage, "prepare_candidates", lambda *_args: ([saved], {}))
    monkeypatch.setattr(study, "construct", lambda *_args: [fresh])
    monkeypatch.setattr(
        study.pilot, "evaluate_group", lambda *_args, **_kwargs: pytest.fail("assessed changed archive")
    )
    with pytest.raises(ValueError, match="incumbent source differs"):
        study.run_target(args, "tfim_para")


@pytest.mark.parametrize("target_id", study.TARGET_IDS)
def test_fontana_excludes_only_haar_fixtures(target_id: str) -> None:
    """Both hardcoded random-MPS fixtures are real and remain valid competitors."""
    target = study.load_target_collection().load(6, target_id).state_vector_copy()
    if target_id.startswith("haar_"):
        with pytest.raises(ValueError, match="real target"):
            study.fontana.real_target(target)
    else:
        aligned = study.fontana.real_target(target)
        assert np.max(abs(aligned.imag)) == 0


def test_old_targets_never_enter_worker_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only missing targets execute, even when workers finish out of order."""
    args = _args(tmp_path / "out")
    args.targets = [study.pilot.TARGETS[0], "tfim_para", "tfim_ferro"]
    args.methods = ["robuststate"]
    old, _ = _sealed_group(tmp_path / "old")
    old.update({"new_build_train_cpu_seconds": 0.0, "source_output": str(tmp_path / "old")})
    monkeypatch.setattr(study, "authenticate_pilot", lambda _args: {(study.pilot.TARGETS[0], "robuststate"): old})
    monkeypatch.setattr(study, "build_manifest", lambda _args: {"schema": study.SCHEMA, "sources": {"stub": "hash"}})
    monkeypatch.setattr(study, "check_sources", lambda *_args, **_kwargs: None)
    seen = []

    def completed(_factory: object, jobs: list[str], *, workers: int) -> Iterator[tuple[int, dict[str, Any]]]:
        seen.append((jobs, workers))
        for index, target_id in reversed(list(enumerate(jobs))):
            group = copy.deepcopy(old)
            group["context"]["identity"]["target_id"] = target_id
            group["build_train_cpu_seconds"] = 4.0
            group["new_build_train_cpu_seconds"] = 4.0
            yield index, {"groups": [group], "exclusions": [], "cpu_seconds": 5.0, "wall_seconds": 6.0}

    monkeypatch.setattr(study.execution, "completed_jobs", completed)
    result = study.run(args)
    assert seen == [(["tfim_para", "tfim_ferro"], 3)]
    assert [group["context"]["identity"]["target_id"] for group in result["groups"]] == args.targets
    assert result["timing"]["full_grid_build_train_cpu_seconds"] == pytest.approx(20.5)
    assert result["timing"]["new_campaign_build_train_cpu_seconds"] == 8
    assert result["timing"]["worker_cpu_seconds"] == 10
    assert json.loads((args.output / "summary.json").read_text())["groups"] == result["groups"]


def test_completed_invocations_preserved_and_sources_rechecked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume retains prior timing receipts and checks frozen sources before completion."""
    args = _args(tmp_path / "out")
    args.targets = [study.pilot.TARGETS[0]]
    args.methods = ["robuststate"]
    group, _target = _sealed_group(tmp_path / "old")
    group["new_build_train_cpu_seconds"] = 0.0
    monkeypatch.setattr(study, "authenticate_pilot", lambda _args: {(args.targets[0], "robuststate"): group})
    manifest = {"schema": study.SCHEMA, "sources": {"stub": "hash"}}
    monkeypatch.setattr(study, "build_manifest", lambda _args: manifest)
    checks = []

    def check(value: dict[str, Any]) -> None:
        checks.append(value)
        assert json.loads((args.output / "summary.json").read_text())["status"] == "running"

    monkeypatch.setattr(study, "check_sources", check)
    first = study.run(args)
    second = study.run(args)
    assert checks == [manifest, manifest]
    assert first["status"] == second["status"] == "complete"
    assert len(first["invocation_history"]) == 1
    assert len(second["invocation_history"]) == 2
    assert second["invocation_history"][0] == first["invocation_history"][0]
    assert second["groups"] == first["groups"]


def test_changed_sources_cannot_mark_campaign_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A source edit during work leaves an incomplete receipt and raises explicitly."""
    args = _args(tmp_path / "out")
    args.targets = [study.pilot.TARGETS[0]]
    args.methods = ["robuststate"]
    monkeypatch.setattr(study, "authenticate_pilot", lambda _args: {})
    monkeypatch.setattr(study, "build_manifest", lambda _args: {"schema": study.SCHEMA})

    def changed(_manifest: dict[str, Any]) -> None:
        msg = "Authenticated source changed"
        raise ValueError(msg)

    monkeypatch.setattr(study, "check_sources", changed)
    with pytest.raises(ValueError, match="source changed"):
        study.run(args)
    assert json.loads((args.output / "summary.json").read_text())["status"] == "running"
