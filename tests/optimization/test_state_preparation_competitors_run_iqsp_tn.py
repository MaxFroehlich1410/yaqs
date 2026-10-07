# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""IQSP tensor-network runner provenance, channel, accounting and selection tests."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from qiskit import QuantumCircuit

pytest.importorskip("torch")
pytest.importorskip("opt_einsum")

from experiments.state_preparation_competitors import run_iqsp_tn as study
from experiments.state_preparation_competitors.common import Candidate

if TYPE_CHECKING:
    import argparse
    from pathlib import Path


def _args(output: Path, *, all_targets: bool = False) -> argparse.Namespace:
    """Return a reduced-grid runner request using genuine independent assessment."""
    arguments = ["--output", str(output), "--depths", "2", "--interpolation-steps", "2", "--construction-steps", "1"]
    if not all_targets:
        arguments.extend(["--targets", study.TARGET_IDS[0]])
    return study.make_parser().parse_args(arguments)


def _candidate(depth: int = 2, *, num_qubits: int = 6) -> Candidate:
    """Return a deterministic elementary circuit with binary-packable training history."""
    circuit = QuantumCircuit(num_qubits)
    circuit.h(range(num_qubits))
    circuit.ry(0.1 * depth, 0)
    circuit.rxx(0.2, 0, 1)
    return Candidate(
        circuit,
        {
            "method": study.METHOD,
            "candidate_id": f"depth_{depth:02d}",
            "status": "complete",
            "depth": depth,
            "hardware": "IBM",
            "noise_basis": "logical",
            "construction": {
                "records": [{"theta": [0.0, 0.0], "fidelity": 0.1}, {"theta": [0.1, 0.2], "fidelity": 0.2}],
                "cpu_seconds": 0.125,
            },
        },
    )


def _stub_training(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace only optimization, preserving the real fixture and assessment engines.

    Returns:
        Mutable ledger containing each actual training dispatch.
    """
    calls = []

    def factory(
        target: np.ndarray,
        *,
        target_id: str,
        depths: tuple[int, ...],
        interpolation_steps: int,
        construction_steps: int,
        noisy_steps: int,
        hardware: str,
        learning_rate: float,
    ) -> list[Candidate]:
        calls.append({
            "target": target.copy(),
            "target_id": target_id,
            "depths": depths,
            "interpolation_steps": interpolation_steps,
            "construction_steps": construction_steps,
            "noisy_steps": noisy_steps,
            "hardware": hardware,
            "learning_rate": learning_rate,
        })
        return [_candidate(depth) for depth in depths]

    monkeypatch.setattr(study.iqsp_tn, "iqsp_tn_candidates", factory)
    return calls


def _construction(args: argparse.Namespace) -> Path:
    """Return the first requested target's construction bundle."""
    return args.output / "construction" / args.targets[0] / study.METHOD / "300/IBM"


def _assessment(args: argparse.Namespace, stage: str = "selection_locked") -> Path:
    """Return the first requested target's independent assessment bundle."""
    return args.output / "evaluation" / args.targets[0] / study.METHOD / "300/IBM" / stage


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("targets", []),
        ("targets", ["unknown"]),
        ("targets", [study.TARGET_IDS[0]] * 2),
        ("depths", []),
        ("depths", [0]),
        ("depths", [2, 2]),
        ("depths", [True]),
        ("depths", [2.0]),
        ("interpolation_steps", 0),
        ("interpolation_steps", True),
        ("construction_steps", -1),
        ("construction_steps", 1.5),
        ("noisy_steps", -1),
        ("noisy_steps", True),
        ("validation_paths", 1),
        ("validation_paths", 2.0),
        ("learning_rate", 0),
        ("learning_rate", -0.1),
        ("learning_rate", float("nan")),
        ("learning_rate", float("inf")),
        ("learning_rate", True),
        ("learning_rate", "0.01"),
    ],
)
def test_invalid_config_fails_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    """Malformed requests never load dependencies, touch output or start optimization."""
    args = _args(tmp_path / "out")
    setattr(args, field, value)
    monkeypatch.setattr(study, "build_manifest", lambda _: pytest.fail("manifest reached"))
    with pytest.raises(ValueError, match=field):
        study.run(args)
    assert not args.output.exists()


def test_defaults_and_manifest_make_reproduction_scope_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All fixtures, fixed channel, all-stage costs and generalized paths are declared."""
    args = study.make_parser().parse_args(["--output", str(tmp_path / "out")])
    assert args.targets == list(study.TARGET_IDS)
    assert args.depths == [2, 4]
    assert (args.interpolation_steps, args.construction_steps, args.noisy_steps) == (20, 200, 10000)
    assert args.learning_rate == pytest.approx(0.01)
    assert args.validation_paths == 8192
    manifest = study.build_manifest(args)
    assert manifest["packages"] == study.PINNED
    assert manifest["noise_basis"] == "logical"
    assert manifest["validation_seed"] == 300
    assert "optimizer_seed" not in manifest
    assert "output" not in manifest["config"]
    assert "dry_run" not in manifest["config"]
    assert "every depth" in manifest["runtime_policy"]
    assert manifest["target_protocols"][study.TARGET_IDS[0]] == "published Gaussian exponential path"
    assert all("generalized" in manifest["target_protocols"][target] for target in study.TARGET_IDS[1:])
    assert "resource accounting only" in manifest["compiler_role"]
    assert "independent validation" in manifest["assessment"]
    assert manifest["two_qubit_cutoff"]["absolute_rotation_angle_at_most"] == pytest.approx(1e-4)
    assert "post-cutoff" in manifest["phase_schedules"]["final_assessment"]
    for name in (
        "benchmarks/state_preparation_target_states.json",
        "experiments/state_preparation_competitors/run_iqsp_tn.py",
        "experiments/state_preparation_competitors/logical_evaluation.py",
        "experiments/state_preparation_competitors/requirements-iqsp-tn.txt",
        "experiments/state_preparation_competitors/IQSP_TN.md",
        "experiments/state_preparation_competitors/vendor/iqsp_tensor/LICENSE",
        "experiments/state_preparation_competitors/vendor/iqsp_tensor/PROVENANCE.md",
    ):
        assert manifest["sources"][name] == study.storage.digest(study.storage.ROOT / name)
    versions = {**study.PINNED, "torch": "wrong"}
    monkeypatch.setattr(study.importlib.metadata, "version", versions.__getitem__)
    with pytest.raises(ValueError, match="pinned"):
        study.build_manifest(args)


def test_dry_run_only_writes_a_resumable_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Planning authenticates the environment without loading fixtures or launching training."""
    args = _args(tmp_path / "out", all_targets=True)
    args.dry_run = True
    calls = _stub_training(monkeypatch)
    monkeypatch.setattr(study, "load_target_collection", lambda: pytest.fail("target loading reached"))
    result = study.run(args)
    assert result["status"] == "planned"
    assert (result["groups"], result["candidates"]) == (9, 9)
    assert sorted(path.name for path in args.output.iterdir()) == ["manifest.json", "plan.json"]
    assert not calls
    assert study.run(args) == result


def test_all_fixture_training_calls_and_real_logical_assessment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nine real fixtures use tensor-training dispatch, independent jumps and locked exact scores."""
    args = _args(tmp_path / "out", all_targets=True)
    calls = _stub_training(monkeypatch)
    actual_exact = study.pilot.logical_evaluation.score_exact

    def check_lock(circuit: QuantumCircuit, target: np.ndarray, hardware: str) -> dict[str, Any]:
        current = calls[-1]["target_id"]
        folder = args.output / "evaluation" / current / study.METHOD / "300/IBM/selection_locked"
        assert (folder / "seal.json").is_file()
        return actual_exact(circuit, target, hardware)

    monkeypatch.setattr(study.pilot.logical_evaluation, "score_exact", check_lock)
    result = study.run(args)
    assert result["status"] == "complete"
    assert len(result["groups"]) == len(calls) == 9
    assert result["exclusions"] == []
    assert result["timing"]["build_train_cpu_seconds"] == pytest.approx(
        sum(group["build_train_cpu_seconds"] for group in result["groups"])
    )
    assert result["timing"]["new_build_train_cpu_seconds"] == result["timing"]["build_train_cpu_seconds"]
    for call, group in zip(calls, result["groups"], strict=True):
        assert call["target"].shape == (64,)
        assert call["hardware"] == "IBM"
        assert call["depths"] == (2,)
        assert call["interpolation_steps"] == 2
        assert call["construction_steps"] == 1
        assert call["noisy_steps"] == 10000
        assert call["learning_rate"] == pytest.approx(0.01)
        assert "seed" not in call
        assert group["selected_index"] == 0
        assert group["context"]["identity"]["method"] == study.METHOD
        assert group["context"]["noise_basis"] == "logical"
        assert group["records"][0]["validation"]["jump_paths"] > 0
        assert group["records"][0]["validation"]["sampled_paths"] == 8192
        assert group["records"][0]["compilation"]["role"] == "resources only; no native noise evaluation"
        assert 0 <= group["final_scores"][0]["exact"]["fidelity"] <= 1
        assert not group["training_reused"]
        method = group["records"][0]["method"]
        metadata = study.storage.load_candidate_metadata(args.output, method)
        folder = args.output / group["construction_cache"]
        history = metadata["construction"]["parameter_history"]
        np.testing.assert_array_equal(study.storage.load_parameter_history(folder, history), [[0, 0], [0.1, 0.2]])
    assert len(list(args.output.rglob("histories.npz"))) == 9
    assert '"theta"' not in (args.output / "summary.json").read_text()


def test_resume_never_retrains_or_reselects_and_keeps_full_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sealed training and assessment resume without reducing the reported historical budget."""
    args = _args(tmp_path / "out")
    _stub_training(monkeypatch)
    first = study.run(args)
    before = {path: study.storage.digest(path) for path in args.output.rglob("*") if path.is_file()}
    monkeypatch.setattr(study.iqsp_tn, "iqsp_tn_candidates", lambda *_args, **_kwargs: pytest.fail("retraining"))
    monkeypatch.setattr(study.pilot.logical_evaluation, "score_validation", lambda *_args: pytest.fail("revalidation"))
    monkeypatch.setattr(study.pilot.logical_evaluation, "score_exact", lambda *_args: pytest.fail("retesting"))
    second = study.run(args)
    assert second["groups"][0]["training_reused"]
    assert second["timing"]["new_build_train_cpu_seconds"] == 0
    assert second["timing"]["build_train_cpu_seconds"] == first["timing"]["build_train_cpu_seconds"]
    for path, expected in before.items():
        if path.name != "summary.json":
            assert study.storage.digest(path) == expected


@pytest.mark.parametrize("stage", ["root", "construction", "selection_locked", "final"])
def test_partial_nonempty_outputs_are_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """Unsealed nonempty outputs require explicit inspection instead of silent replacement."""
    args = _args(tmp_path / "out")
    calls = _stub_training(monkeypatch)
    if stage == "root":
        folder = args.output
    else:
        args.dry_run = True
        study.run(args)
        args.dry_run = False
        folder = _construction(args) if stage == "construction" else _assessment(args, stage)
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "partial.txt"
    marker.write_text("do not overwrite")
    with pytest.raises(ValueError, match="nonempty"):
        study.run(args)
    assert not calls
    assert marker.read_text() == "do not overwrite"


@pytest.mark.parametrize(
    "field", ["depths", "interpolation_steps", "construction_steps", "noisy_steps", "learning_rate"]
)
def test_changed_budgets_reject_manifest_resume(tmp_path: Path, field: str) -> None:
    """No changed optimizer configuration can reuse a previous plan or checkpoint."""
    args = _args(tmp_path / "out")
    args.dry_run = True
    study.run(args)
    value = [4] if field == "depths" else getattr(args, field) + 1
    setattr(args, field, value)
    with pytest.raises(ValueError, match="manifest changed"):
        study.run(args)


@pytest.mark.parametrize("stage", ["construction", "selection_locked", "final"])
def test_corrupt_sealed_bundles_fail_before_new_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """Corrupted training, selection and final circuits cannot be used for resumed results."""
    args = _args(tmp_path / "out")
    calls = _stub_training(monkeypatch)
    study.run(args)
    folder = _construction(args) if stage == "construction" else _assessment(args, stage)
    (folder / "circuits.qpy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        study.run(args)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "change", ["method", "candidate_id", "depth", "noise_basis", "hardware", "context", "register"]
)
def test_resealed_wrong_candidate_metadata_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Integrity seals supplement semantic target, grid and channel authentication."""
    args = _args(tmp_path / "out")
    _stub_training(monkeypatch)
    study.run(args)
    folder = _construction(args)
    record, circuits = study.storage.load_bundle(folder)
    if change == "register":
        circuits[0] = QuantumCircuit(2)
    elif change == "context":
        record["candidates"][0]["runner_context"]["target_id"] = "wrong"
    else:
        record["candidates"][0][change] = "wrong"
    study.storage.save_bundle(folder, record, circuits)
    with pytest.raises(ValueError, match="candidate identity"):
        study.run(args)


@pytest.mark.parametrize("change", ["circuits", "candidate_id", "candidate_index", "grid"])
def test_resealed_assessment_must_match_original_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Resealed assessment circuits and identities cannot drift from authenticated training."""
    args = _args(tmp_path / "out")
    _stub_training(monkeypatch)
    study.run(args)
    folder = _assessment(args)
    record, circuits = study.storage.load_bundle(folder)
    if change == "circuits":
        circuits[0].ry(0.5, 0)
    elif change == "grid":
        record["records"] = []
    else:
        record["records"][0]["method"][change] = "wrong"
    study.storage.save_bundle(folder, record, circuits)
    with pytest.raises(ValueError, match="assessment"):
        study.run(args)


def test_candidate_grid_opaque_blocks_and_incomplete_status_fail(tmp_path: Path) -> None:
    """Malformed trainer results are rejected before saving or scoring an endpoint."""
    args = _args(tmp_path / "out")
    target = np.eye(1, 64, dtype=np.complex128).ravel()
    candidate = _candidate()
    candidate.metadata["runner_context"] = study.candidate_context(args, args.targets[0], target)
    with pytest.raises(ValueError, match="grid size"):
        study.authenticate_candidates(args, args.targets[0], target, [])
    candidate.metadata["status"] = "failed"
    with pytest.raises(ValueError, match="identity"):
        study.authenticate_candidates(args, args.targets[0], target, [candidate])
    candidate.metadata["status"] = "complete"
    candidate.circuit.unitary(np.eye(4), [0, 1])
    with pytest.raises(ValueError, match="opaque"):
        study.authenticate_candidates(args, args.targets[0], target, [candidate])


def test_source_changes_and_deletions_are_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Source authentication catches both edits and missing fixture files."""
    monkeypatch.setattr(study.storage, "ROOT", tmp_path)
    source = tmp_path / "fixture.json"
    source.write_text("original")
    manifest = {"sources": {source.name: study.storage.digest(source)}}
    study.check_sources(manifest)
    source.write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        study.check_sources(manifest)
    source.unlink()
    with pytest.raises(ValueError, match="source changed"):
        study.check_sources(manifest)


def test_no_jump_validation_remains_ineligible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A validation bank without jumps cannot silently certify a nonempty candidate."""
    args = _args(tmp_path / "out")
    _stub_training(monkeypatch)
    actual = study.pilot.logical_evaluation.score_validation

    def without_jumps(
        circuit: QuantumCircuit, target: np.ndarray, hardware: str, validation_paths: int, entropy: list[int]
    ) -> dict[str, Any]:
        result = actual(circuit, target, hardware, validation_paths, entropy)
        result["jump_paths"] = 0
        return result

    monkeypatch.setattr(study.pilot.logical_evaluation, "score_validation", without_jumps)
    result = study.run(args)
    assert result["status"] == "incomplete"
    assert result["groups"][0]["selected_index"] is None
    assert result["groups"][0]["status"] == "no_eligible_candidate"


def test_dry_run_can_start_later_with_same_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run is an execution choice, not a scientific change requiring a new directory."""
    args = _args(tmp_path / "out")
    args.dry_run = True
    study.run(args)
    manifest = json.loads((args.output / "manifest.json").read_text())
    args.dry_run = False
    _stub_training(monkeypatch)
    study.run(args)
    assert json.loads((args.output / "manifest.json").read_text()) == manifest


def test_real_tensor_training_compacts_histories_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuine Gaussian training phase passes through storage and independent assessment."""
    args = _args(tmp_path / "real-tensor-training")
    args.depths = [1]
    args.interpolation_steps = 1
    args.noisy_steps = 1
    first = study.run(args)
    assert first["status"] == "complete"
    group = first["groups"][0]
    metadata = study.storage.load_candidate_metadata(args.output, group["records"][0]["method"])
    assert metadata["interpolation_scope"] == "published_gaussian_path"
    assert len(metadata["stages"]) == 1
    assert metadata["noise_basis"] == "logical"
    assert metadata["noisy"]["steps"] == 1
    assert metadata["postprocessing"]["final_noisy_fidelity"] == pytest.approx(
        group["final_scores"][0]["exact"]["fidelity"], abs=1e-12
    )
    assert metadata["postprocessing"]["output_windows"] == group["records"][0]["logical_resources"]["source_gate_count"]
    assert metadata["cpu_seconds"] <= group["build_train_cpu_seconds"]
    folder = args.output / group["construction_cache"]
    for phase in (metadata["stages"][0]["optimization"], metadata["noisy"]):
        history = study.storage.load_parameter_history(folder, phase["parameter_history"])
        assert history.shape[0] == len(phase["records"]) == 2
        assert np.all(np.isfinite(history))
        np.testing.assert_array_equal(history[0], phase["initial_theta"])
        np.testing.assert_array_equal(history[-1], phase["final_theta"])
        assert all("theta" not in record for record in phase["records"])
    assert len(list(args.output.rglob("histories.npz"))) == 1
    assert group["records"][0]["validation"]["jump_paths"] > 0
    monkeypatch.setattr(study.iqsp_tn, "iqsp_tn_candidates", lambda *_args, **_kwargs: pytest.fail("retraining"))
    monkeypatch.setattr(study.pilot.logical_evaluation, "score_validation", lambda *_args: pytest.fail("revalidation"))
    monkeypatch.setattr(study.pilot.logical_evaluation, "score_exact", lambda *_args: pytest.fail("retesting"))
    second = study.run(args)
    assert second["groups"][0]["training_reused"]
    assert second["timing"]["new_build_train_cpu_seconds"] == 0
    assert second["timing"]["build_train_cpu_seconds"] == first["timing"]["build_train_cpu_seconds"]
    assert second["groups"][0]["final_scores"] == group["final_scores"]


def test_main_dispatches_cli_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The module entry point forwards the parsed execution mode and explicit budgets."""
    captured = []
    monkeypatch.setattr(study, "run", lambda args: captured.append(copy.deepcopy(vars(args))))
    study.main(["--output", str(tmp_path / "out"), "--dry-run", "--noisy-steps", "0"])
    assert captured[0]["dry_run"]
    assert captured[0]["noisy_steps"] == 0
