# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Logical-pilot provenance, cache boundaries, selection locks and budget tests."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from qiskit import QuantumCircuit

from experiments.state_preparation_competitors import run_logical as study
from experiments.state_preparation_competitors.common import Candidate

if TYPE_CHECKING:
    import argparse
    from pathlib import Path


def _args(output: Path) -> argparse.Namespace:
    """Make a small assessment request without changing training budgets.

    Returns:
        Parsed fixed-budget pilot arguments.
    """
    return study.make_parser().parse_args(["--output", str(output), "--validation-paths", "8"])


def _candidate(method: str, identifier: str, angle: float = 0.1) -> Candidate:
    """Return an elementary source circuit whose compilation changes its schedule."""
    circuit = QuantumCircuit(2)
    circuit.ry(angle, 0)
    circuit.rxx(0.2, 0, 1)
    return Candidate(circuit, {"method": method, "candidate_id": identifier, "status": "complete", "seed": 300})


def _identity(method: str) -> dict[str, Any]:
    """Return a fixed pilot identity independent of optional optimizer packages."""
    return {"target_id": study.TARGETS[0], "method": method, "seed": 300, "hardware": "IBM"}


def _rewrite_manifest(folder: Path, manifest: dict[str, Any]) -> None:
    """Keep the outer checksum consistent while testing semantic manifest guards."""
    study.storage.write_json(folder / "manifest.json", manifest)
    study.storage.write_json(
        folder / "summary.json", {"manifest_sha256": study.storage.digest(folder / "manifest.json")}
    )


def _cached(
    tmp_path: Path, method: str = "fixed_vqsp", target_id: str = study.TARGETS[0]
) -> tuple[argparse.Namespace, Path, np.ndarray]:
    """Create sealed synthetic cache artifacts with a real source fingerprint set.

    Returns:
        New campaign arguments, original method root, and synthetic target.
    """
    args = _args(tmp_path / "new")
    args.gaussian_source = tmp_path / "gaussian"
    args.other_source = tmp_path / "other"
    folder = (args.gaussian_source if target_id == study.TARGETS[0] else args.other_source / target_id) / method
    target = np.array([1, 0, 0, 0], dtype=np.complex128)
    depths = [2, 4] if method == "fixed_vqsp" else [1, 2, 3, 4]
    config = {
        "targets": [target_id],
        "methods": [method],
        "hardware": ["IBM"],
        "seeds": [300],
        "depths": depths,
        "construction_steps": 200,
        "noisy_steps": 200,
        "online_steps": 100,
        "noisy_online_steps": 100,
        "training_paths": 100,
        "nacl_steps": [0, 10, 50],
        "nacl_evaluations": 100,
        "nacl_restarts": 1,
        "nacl_max_gates": 300,
    }
    points = [0, 10, 50] if method == "nacl" else [1] if method == "current_b_lbfgs_frozen" else depths
    candidates = []
    for point in points:
        identifier = f"steps_{point:05d}" if method == "nacl" else f"depth_{point:02d}"
        candidate = _candidate(method, identifier)
        if method == "nacl":
            candidate = Candidate(QuantumCircuit(2), candidate.metadata)
            candidate.circuit.rz(0.2, 0)
            candidate.circuit.sx(0)
            candidate.circuit.cx(0, 1)
        candidate.metadata.update({
            "depth": point,
            "hardware": "IBM",
            "construction": {"steps": 200},
            "noisy_training": {"steps": 200},
            "training_bank": {"sampled_paths": 100},
            "search_steps_per_restart": point,
            "inner_evaluations": 100,
            "restarts": 1,
            "max_native_gates": 300,
            "diagnostic": {"records": [{"theta": [0.1], "fidelity": 0.4}, {"theta": [0.2], "fidelity": 0.5}]},
            "theta_sha256": "theta",
            "archive_checkpoint_sha256": "checkpoint",
            "archive_selection_sha256": "selection",
        })
        candidates.append(candidate)
    relative = f"construction/{target_id}/{method}/300/IBM"
    study.storage.save_bundle(
        folder / relative,
        {
            "construction_cpu_seconds": 12.5,
            "construction_wall_seconds": 13.0,
            "candidates": [candidate.metadata for candidate in candidates],
        },
        [candidate.circuit for candidate in candidates],
    )
    study.storage.save_bundle(
        folder / f"evaluation/{target_id}/{method}/300/IBM/final",
        {
            "context": {
                "identity": {**_identity(method), "target_id": target_id},
                "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
                "construction_receipt_sha256": study.storage.digest(folder / relative / "seal.json"),
            },
        },
        [],
    )
    manifest = {"config": config, "sources": study.build_manifest(args)["sources"]}
    _rewrite_manifest(folder, manifest)
    return args, folder, target


def _evaluation(
    folder: Path, candidates: list[Candidate], *, target: np.ndarray | None = None, validation_paths: int = 8
) -> dict[str, Any]:
    """Run the real logical assessment and persistence boundary on tiny circuits.

    Returns:
        Locked selection and final logical scores.
    """
    return study.evaluate_group(
        folder,
        candidates,
        np.array([1, 0, 0, 0], dtype=np.complex128) if target is None else target,
        identity=_identity(candidates[0].metadata["method"]),
        validation_paths=validation_paths,
        construction={"construction_cpu_seconds": 1.0, "candidates": []},
        construction_path="construction/source",
        construction_hash="construction-hash",
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("targets", []),
        ("targets", ["unknown"]),
        ("targets", [study.TARGETS[0]] * 2),
        ("methods", []),
        ("methods", ["aqce"]),
        ("methods", ["nacl", "nacl"]),
        ("validation_paths", 1),
        ("validation_paths", 0),
        ("validation_paths", True),
        ("validation_paths", 8.0),
    ],
)
def test_invalid_config_fails_before_manifest_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: object
) -> None:
    """Unsupported jobs do not inspect caches, write output or invoke a solver."""
    args = _args(tmp_path / "out")
    setattr(args, key, value)
    monkeypatch.setattr(study, "build_manifest", lambda _args: pytest.fail("manifest reached"))
    with pytest.raises(ValueError, match=r"supported|Validation"):
        study.run(args)
    assert not args.output.exists()


def test_manifest_records_fixed_budgets_source_hashes_and_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Logical placement, immutable budgets and numerical versions are reviewable."""
    args = _args(tmp_path / "out")
    versions = dict(study.PINNED)
    monkeypatch.setattr(study.importlib.metadata, "version", versions.__getitem__)
    result = study.build_manifest(args)
    assert result["hardware"] == "IBM"
    assert result["optimizer_seed"] == 300
    assert result["packages"] == study.PINNED
    assert "output" not in result["config"]
    assert result["config"]["validation_paths"] == 8
    assert set(result["budgets"]) == set(study.METHODS)
    assert "500" in result["budgets"]["robuststate"]
    assert "M=100" in result["budgets"]["fixed_vqsp"]
    assert "resource accounting only" in result["compiler_role"]
    for name in (
        "benchmarks/state_preparation_target_states.json",
        "experiments/state_preparation_competitors/LOGICAL_NOISE.md",
        "experiments/state_preparation_competitors/run_logical.py",
    ):
        assert result["sources"][name] == study.storage.digest(study.storage.ROOT / name)
    versions["qiskit"] = "wrong"
    with pytest.raises(ValueError, match="pinned"):
        study.build_manifest(args)


@pytest.mark.parametrize("method", ["fixed_vqsp", "layerwise", "nacl"])
@pytest.mark.parametrize("target_id", [study.TARGETS[0], "haar_random_1"])
def test_reuse_authenticates_source_circuits_and_references_histories_once(
    tmp_path: Path, method: str, target_id: str
) -> None:
    """Both cache roots preserve source schedules and avoid copying historical arrays."""
    args, folder, target = _cached(tmp_path, method, target_id)
    original, circuits = study.storage.load_bundle(folder / f"construction/{target_id}/{method}/300/IBM")
    candidates = study.reuse_candidates(args, target_id, method, target)
    assert len(candidates) == len(circuits)
    for index, candidate in enumerate(candidates):
        assert candidate.circuit == circuits[index]
        assert candidate.metadata["historical_build_train_cpu_seconds"] == pytest.approx(12.5)
        assert candidate.metadata["historical_selection_cpu_seconds"] == 0
        assert candidate.metadata["training_reused"] is True
        assert candidate.metadata["source_candidate"]["candidate_index"] == index
        assert "diagnostic" not in candidate.metadata
    study.storage.save_bundle(
        args.output / "construction", {"candidates": [c.metadata for c in candidates]}, [c.circuit for c in candidates]
    )
    assert not (args.output / "construction/histories.npz").exists()
    assert original["candidates"][0]["diagnostic"]["parameter_history"]["file"] == "histories.npz"
    assert (folder / f"construction/{target_id}/{method}/300/IBM/histories.npz").exists()


def test_reuse_rejects_manifest_checksum_and_training_source_tampering(tmp_path: Path) -> None:
    """Checksums and training-source fingerprints are required even for sealed caches."""
    args, folder, target = _cached(tmp_path)
    manifest = json.loads((folder / "manifest.json").read_text())
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        study.reuse_candidates(args, study.TARGETS[0], "fixed_vqsp", target)
    manifest["sources"]["experiments/state_preparation_competitors/controls.py"] = "wrong"
    _rewrite_manifest(folder, manifest)
    with pytest.raises(ValueError, match="training source changed"):
        study.reuse_candidates(args, study.TARGETS[0], "fixed_vqsp", target)


def test_reuse_records_unrelated_source_changes_without_retraining(tmp_path: Path) -> None:
    """An unrelated paper adapter change is disclosed without invalidating old controls."""
    args, folder, target = _cached(tmp_path)
    manifest = json.loads((folder / "manifest.json").read_text())
    changed = "experiments/state_preparation_competitors/fontana.py"
    manifest["sources"][changed] = "old-paper-adapter"
    _rewrite_manifest(folder, manifest)
    candidates = study.reuse_candidates(args, study.TARGETS[0], "fixed_vqsp", target)
    assert candidates[0].metadata["source_candidate"]["unrelated_changed_sources"] == [changed]


@pytest.mark.parametrize(
    ("method", "key", "value"),
    [
        ("fixed_vqsp", "depths", [1, 2]),
        ("fixed_vqsp", "construction_steps", 40),
        ("fixed_vqsp", "training_paths", 3),
        ("layerwise", "noisy_online_steps", 0),
        ("nacl", "nacl_steps", [0, 2]),
        ("nacl", "nacl_evaluations", 8),
        ("nacl", "nacl_restarts", 2),
        ("nacl", "nacl_max_gates", 999),
    ],
)
def test_reuse_rejects_changed_declared_budget(tmp_path: Path, method: str, key: str, value: object) -> None:
    """A cache from another optimization budget cannot silently enter the pilot."""
    args, folder, target = _cached(tmp_path, method)
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["config"][key] = value
    _rewrite_manifest(folder, manifest)
    with pytest.raises(ValueError, match="budget"):
        study.reuse_candidates(args, study.TARGETS[0], method, target)


@pytest.mark.parametrize("field", ["target_sha256", "identity", "construction_receipt_sha256"])
def test_reuse_rejects_resealed_evaluation_context_tampering(tmp_path: Path, field: str) -> None:
    """Semantic target and cache identity checks supplement binary bundle integrity."""
    args, folder, target = _cached(tmp_path)
    final = folder / f"evaluation/{study.TARGETS[0]}/fixed_vqsp/300/IBM/final"
    record, circuits = study.storage.load_bundle(final)
    record["context"][field] = "wrong"
    study.storage.save_bundle(final, record, circuits)
    with pytest.raises(ValueError, match="target, identity or receipt"):
        study.reuse_candidates(args, study.TARGETS[0], "fixed_vqsp", target)


@pytest.mark.parametrize("kind", ["method", "seed", "missing_depth", "depth"])
def test_reuse_rejects_candidate_metadata_or_grid_tampering(tmp_path: Path, kind: str) -> None:
    """A valid seal cannot turn an incomplete or relabeled grid into the stated experiment."""
    args, folder, target = _cached(tmp_path)
    construction = folder / f"construction/{study.TARGETS[0]}/fixed_vqsp/300/IBM"
    record, circuits = study.storage.load_bundle(construction)
    if kind == "missing_depth":
        record["candidates"].pop()
        circuits.pop()
    else:
        record["candidates"][0][kind] = "wrong" if kind == "method" else 999
    study.storage.save_bundle(construction, record, circuits)
    final = folder / f"evaluation/{study.TARGETS[0]}/fixed_vqsp/300/IBM/final"
    old, saved = study.storage.load_bundle(final)
    old["context"]["construction_receipt_sha256"] = study.storage.digest(construction / "seal.json")
    study.storage.save_bundle(final, old, saved)
    with pytest.raises(ValueError, match=r"candidate|Candidate|grid|identity"):
        study.reuse_candidates(args, study.TARGETS[0], "fixed_vqsp", target)


def test_reuse_frozen_incumbent_charges_original_training_and_validation_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading the archived winner is not reported as its original optimization cost."""
    args, folder, target = _cached(tmp_path, "current_b_lbfgs_frozen")
    record, circuits = study.storage.load_bundle(
        folder / f"construction/{study.TARGETS[0]}/current_b_lbfgs_frozen/300/IBM"
    )
    fresh = Candidate(
        circuits[0],
        {
            **record["candidates"][0],
            "archived_search_cost": {
                "construction_cpu_seconds": 2.0,
                "pruning_cpu_seconds": 0.5,
                "fine_tuning_cpu_seconds": 3.0,
                "validation_cpu_seconds": 0.2,
            },
        },
    )
    monkeypatch.setattr(study.reference, "frozen_reference", lambda *_a: fresh)
    candidates = study.reuse_candidates(args, study.TARGETS[0], "current_b_lbfgs_frozen", target)
    assert candidates[0].metadata["historical_build_train_cpu_seconds"] == pytest.approx(5.5)
    assert candidates[0].metadata["historical_selection_cpu_seconds"] == pytest.approx(0.2)
    fresh.metadata["theta_sha256"] = "different"
    with pytest.raises(ValueError, match="baseline differs"):
        study.reuse_candidates(args, study.TARGETS[0], "current_b_lbfgs_frozen", target)


@pytest.mark.parametrize("method", ["robuststate", "fontana"])
def test_fresh_dispatch_keeps_paper_budgets_and_original_gate_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Switching channel placement must not silently change optimizer or iteration budgets."""
    calls = []
    expected = [_candidate(method, "depth_02")]

    def factory(target: np.ndarray, **kwargs: object) -> list[Candidate]:
        np.testing.assert_array_equal(target, [1, 0, 0, 0])
        calls.append(kwargs)
        return expected

    module = study.robuststate if method == "robuststate" else study.fontana
    monkeypatch.setattr(module, f"{method}_candidates", factory)
    assert study.construct(_args(tmp_path), study.TARGETS[0], method, np.array([1, 0, 0, 0])) is expected
    assert calls == [
        {
            "hardware": "IBM",
            "seed": 300,
            "depths": (2, 4),
            "noise_basis": "logical",
            "noiseless_steps" if method == "robuststate" else "construction_steps": 500
            if method == "robuststate"
            else 200,
            "noisy_steps": 50 if method == "robuststate" else 200,
        }
    ]


def test_reused_dispatch_never_calls_fresh_optimizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every reused method follows the authenticated-cache path."""
    args = _args(tmp_path)
    target = np.array([1, 0], dtype=np.complex128)
    calls = []
    expected = []

    def reuse(*values: object) -> list[Candidate]:
        calls.append(values)
        return expected

    monkeypatch.setattr(study, "reuse_candidates", reuse)
    for method in study.REUSED:
        assert study.construct(args, study.TARGETS[0], method, target) is expected
        assert calls[-1][2] == method
    with pytest.raises(ValueError, match="Unsupported"):
        study.construct(args, study.TARGETS[0], "aqce", target)


def test_selection_is_locked_before_exact_and_keeps_logical_circuits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact test scores cannot select a different winner or use resource-only compilation."""
    candidates = [_candidate("fontana", "a", 0.1), _candidate("fontana", "b", 0.2)]
    original = [candidate.circuit.copy() for candidate in candidates]
    validation_calls = []
    exact_calls = []

    def validation(circuit: QuantumCircuit, _target: np.ndarray, hardware: str, count: int, entropy: list[int]) -> dict:
        assert hardware == "IBM"
        assert count == 8
        index = original.index(circuit)
        validation_calls.append(entropy)
        return {"fidelity": [0.9, 0.8][index], "jump_paths": 1, "standard_error": 0.1}

    def exact(circuit: QuantumCircuit, _target: np.ndarray, hardware: str) -> dict:
        assert hardware == "IBM"
        lock, saved = study.storage.load_bundle(tmp_path / "selection_locked")
        assert lock["selected_index"] == 0
        assert saved == original
        index = original.index(circuit)
        exact_calls.append(index)
        return {"fidelity": [0.3, 0.99][index]}

    monkeypatch.setattr(study.logical_evaluation, "score_validation", validation)
    monkeypatch.setattr(study.logical_evaluation, "score_exact", exact)
    result = _evaluation(tmp_path, candidates)
    assert result["selected_index"] == 0
    assert result["final_scores"][1]["exact"]["fidelity"] == pytest.approx(0.99)
    assert exact_calls == [0, 1]
    assert len(validation_calls) == 2
    assert validation_calls[0] != validation_calls[1]
    assert candidates[0].circuit == original[0]
    assert result["records"][0]["logical_resources"]["logical_operations"] == {"ry": 1, "rxx": 1}
    assert "rxx" not in result["records"][0]["compilation"]["compiled_operations"]
    assert "resources only" in result["records"][0]["compilation"]["role"]
    assert _evaluation(tmp_path, candidates) == result
    assert len(exact_calls) == len(validation_calls) == 2


@pytest.mark.parametrize(
    ("empty", "status", "eligible"), [(False, "complete", False), (True, "complete", True), (True, "failed", False)]
)
def test_missing_jump_coverage_is_not_resampled_and_failed_candidates_are_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, empty: bool, status: str, eligible: bool
) -> None:
    """Zero-jump banks fail coverage except when the original circuit has no windows."""
    candidate = _candidate("nacl", "one")
    if empty:
        candidate = Candidate(QuantumCircuit(2), candidate.metadata)
    candidate.metadata["status"] = status
    calls = []

    def validation(*_args: object) -> dict:
        calls.append(1)
        return {"fidelity": 1.0, "jump_paths": 0}

    monkeypatch.setattr(study.logical_evaluation, "score_validation", validation)
    result = _evaluation(tmp_path, [candidate])
    assert calls == [1]
    assert result["records"][0]["eligible"] is eligible
    assert result["selected_index"] == (0 if eligible else None)
    assert result["status"] == ("complete" if eligible else "no_eligible_candidate")


def test_resumed_selection_rejects_changed_context(tmp_path: Path) -> None:
    """A completed assessment cannot silently change validation sample count."""
    candidate = Candidate(QuantumCircuit(2), {"method": "nacl", "candidate_id": "zero", "status": "complete"})
    _evaluation(tmp_path, [candidate])
    with pytest.raises(ValueError, match="context changed"):
        _evaluation(tmp_path, [candidate], validation_paths=9)


@pytest.mark.parametrize("field", ["selection_lock_sha256", "records", "selected_index", "context", "final_scores"])
def test_resumed_final_rejects_semantic_lock_mismatch(tmp_path: Path, field: str) -> None:
    """Final receipts must preserve the locked identity, winner and candidate mapping."""
    candidate = Candidate(QuantumCircuit(2), {"method": "nacl", "candidate_id": "zero", "status": "complete"})
    _evaluation(tmp_path, [candidate])
    result, circuits = study.storage.load_bundle(tmp_path / "final")
    result[field] = None if field == "selected_index" else [] if field in {"records", "final_scores"} else "wrong"
    study.storage.save_bundle(tmp_path / "final", result, circuits)
    with pytest.raises(ValueError, match=r"selection lock|final result"):
        _evaluation(tmp_path, [candidate])


def test_campaign_exclusions_cost_accounting_and_resume_without_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real target restrictions and storage work for both reused and fresh candidates."""
    args = _args(tmp_path / "out")
    args.methods = ["nacl", "fontana"]
    args.targets = [study.TARGETS[0], "haar_random_1"]
    monkeypatch.setattr(study, "build_manifest", lambda _args: {"schema": study.SCHEMA})
    calls = []

    def factory(_args: argparse.Namespace, target_id: str, method: str, target: np.ndarray) -> list[Candidate]:
        calls.append((target_id, method))
        metadata: dict[str, Any] = {"method": method, "candidate_id": "zero", "status": "complete"}
        if method == "nacl":
            metadata.update({
                "training_reused": True,
                "historical_build_train_cpu_seconds": 12.0,
                "historical_selection_cpu_seconds": 0.5,
            })
        return [Candidate(QuantumCircuit(target.size.bit_length() - 1), metadata)]

    monkeypatch.setattr(study, "construct", factory)
    result = study.run(args)
    assert len(calls) == len(result["groups"]) == 3
    assert len(result["exclusions"]) == 1
    assert result["exclusions"][0]["target"] == "haar_random_1"
    assert result["exclusions"][0]["method"] == "fontana"
    for group in result["groups"]:
        if group["training_reused"]:
            assert group["new_build_train_cpu_seconds"] == 0
            assert group["build_train_cpu_seconds"] == 12
            assert group["historical_selection_cpu_seconds"] == pytest.approx(0.5)
        else:
            assert group["build_train_cpu_seconds"] == group["new_build_train_cpu_seconds"]
    assert study.run(args) == result
    assert len(calls) == 3
    assert json.loads((args.output / "summary.json").read_text()) == result
    monkeypatch.setattr(study, "build_manifest", lambda _args: {"schema": "changed"})
    with pytest.raises(ValueError, match="manifest changed"):
        study.run(args)


def test_foreign_output_and_cli_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI honors explicit scope and never overwrites an unrelated directory."""
    args = _args(tmp_path)
    (tmp_path / "foreign.txt").write_text("preserve")
    monkeypatch.setattr(study, "build_manifest", lambda _args: {})
    with pytest.raises(ValueError, match="nonempty"):
        study.run(args)
    calls = []
    monkeypatch.setattr(study, "run", calls.append)
    study.main(["--output", str(tmp_path), "--methods", "robuststate", "--targets", "tfim_critical"])
    assert calls[0].methods == ["robuststate"]
    assert calls[0].targets == ["tfim_critical"]
    assert calls[0].validation_paths == 8192
