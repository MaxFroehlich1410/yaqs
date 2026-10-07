# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Lossless single-copy histories, authenticated references, and portable resume."""

from __future__ import annotations

import copy
import json
import shutil
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import pytest
from qiskit import QuantumCircuit

from experiments.state_preparation_competitors import histories
from experiments.state_preparation_competitors import run as study
from experiments.state_preparation_competitors.common import Candidate

if TYPE_CHECKING:
    from pathlib import Path


def rich_candidate(identifier: str = "first", *, empty_parameters: bool = False) -> Candidate:
    """Build a toy candidate retaining realistic scalar and parameter diagnostics.

    Returns:
        An empty six-qubit circuit with two complete optimizer phase records.
    """
    initial = [] if empty_parameters else [-0.0, 0.0, np.nextafter(1.0, 2.0), -np.pi]
    final = [] if empty_parameters else [0.25, -0.125, np.nextafter(1.0, 0.0), np.pi]
    phase = {
        "initial_theta": initial,
        "best_theta": final,
        "final_theta": final,
        "fidelity_trace": [0.4, 0.9],
        "records": [
            {"iteration": 0, "theta": initial, "fidelity": 0.4, "cumulative_cpu_seconds": 0.2},
            {"iteration": 1, "theta": final, "fidelity": 0.9, "cumulative_cpu_seconds": 0.5},
        ],
        "cpu_seconds": 0.5,
        "requested_steps": 1,
    }
    metadata = {
        "candidate_id": identifier,
        "method": "aqce",
        "status": "complete",
        "seed": 300,
        "initial_theta": initial,
        "theta": final,
        "construction": phase,
        "noisy_training": copy.deepcopy(phase),
        "training_bank": {"sampled_paths": 100, "raw_sha256": "a" * 64, "counter": 5300},
    }
    return Candidate(QuantumCircuit(6), metadata)


def construction_bundle(root: Path) -> tuple[Path, list[Candidate], dict[str, Any]]:
    """Save two candidates and return a portable reference to the first one.

    Returns:
        Construction folder, original candidates, and first-candidate reference.
    """
    folder = root / "construction/toy/aqce/0/shared"
    grid = [rich_candidate(), rich_candidate("second")]
    study.prepare_candidates(folder, lambda: grid)
    method = {
        **{key: grid[0].metadata[key] for key in ("candidate_id", "method", "status")},
        "candidate_index": 0,
        "construction_reference": {
            "path": str(folder.relative_to(root)),
            "seal_sha256": study.digest(folder / "seal.json"),
        },
    }
    return folder, grid, method


def reseal_record(folder: Path, record: dict[str, Any]) -> None:
    """Authenticate an intentionally malformed record to exercise schema checks."""
    study.write_json(folder / "record.json", record)
    seal = json.loads((folder / "seal.json").read_text())
    seal["record.json"] = study.digest(folder / "record.json")
    study.write_json(folder / "seal.json", seal)


def fail_if_called(*_args: object, **_kwargs: object) -> NoReturn:
    """Fail if portable resume repeats scientific work already saved and sealed."""
    pytest.fail("A completed campaign must resume without numerical recomputation.")


@pytest.mark.parametrize("empty_parameters", [False, True])
def test_bundle_histories_preserve_exact_bits_all_scalar_curves_and_endpoints(
    tmp_path: Path, *, empty_parameters: bool
) -> None:
    """Packing removes only repeated theta vectors and preserves negative-zero bits."""
    candidate = rich_candidate(empty_parameters=empty_parameters)
    original = copy.deepcopy(candidate.metadata)
    returned, receipt = study.prepare_candidates(tmp_path, lambda: [candidate])
    assert candidate.metadata == original
    compact = returned[0].metadata
    assert compact == receipt["candidates"][0]
    reconstructed = copy.deepcopy(compact)
    for phase_name in ("construction", "noisy_training"):
        phase = reconstructed[phase_name]
        descriptor = phase.pop("parameter_history")
        matrix = study.load_parameter_history(tmp_path, descriptor)
        expected = np.asarray([row["theta"] for row in original[phase_name]["records"]], dtype="<f8")
        assert matrix.dtype == np.dtype("<f8")
        assert matrix.shape == (2, 0 if empty_parameters else 4)
        assert matrix.tobytes() == expected.tobytes()
        for row, values in zip(phase["records"], matrix, strict=True):
            assert "theta" not in row
            row["theta"] = values.tolist()
    assert reconstructed == original
    assert len(list(tmp_path.glob("*.npz"))) == 1
    seal = json.loads((tmp_path / "seal.json").read_text())
    assert set(seal) == {"record.json", "circuits.qpy", "histories.npz"}
    assert seal["histories.npz"] == study.digest(tmp_path / "histories.npz")
    cached, cached_receipt = study.prepare_candidates(tmp_path, fail_if_called)
    assert cached_receipt == receipt
    assert cached[0].metadata == compact
    assert cached[0].circuit == candidate.circuit


@pytest.mark.parametrize("mutation", ["append", "remove"])
def test_history_corruption_stops_bundle_resume_before_factory(tmp_path: Path, mutation: str) -> None:
    """An unchanged JSON receipt cannot hide a changed or missing binary payload."""
    folder, _, _ = construction_bundle(tmp_path)
    archive = folder / "histories.npz"
    if mutation == "append":
        archive.write_bytes(archive.read_bytes() + b"corrupt")
    else:
        archive.unlink()
    with pytest.raises(ValueError, match="integrity failure"):
        study.prepare_candidates(folder, fail_if_called)


def test_history_descriptor_requires_a_sealed_archive(tmp_path: Path) -> None:
    """Removing only the NPZ receipt does not turn a compact bundle into legacy JSON."""
    folder, _, _ = construction_bundle(tmp_path)
    seal = json.loads((folder / "seal.json").read_text())
    del seal["histories.npz"]
    study.write_json(folder / "seal.json", seal)
    with pytest.raises(ValueError, match="descriptors and seal disagree"):
        study.load_bundle(folder)


def test_resealed_descriptor_must_match_binary_matrix_shape(tmp_path: Path) -> None:
    """Hashes and valid JSON alone cannot authorize differently shaped numerical data."""
    folder, _, _ = construction_bundle(tmp_path)
    record = json.loads((folder / "record.json").read_text())
    record["candidates"][0]["construction"]["parameter_history"]["shape"][1] += 1
    reseal_record(folder, record)
    with pytest.raises(ValueError, match="shape"):
        study.load_bundle(folder)


def test_history_reader_rejects_descriptor_from_another_phase(tmp_path: Path) -> None:
    """A request for a nonauthenticated matrix is rejected even with valid dimensions."""
    folder, _, method = construction_bundle(tmp_path)
    metadata = study.load_candidate_metadata(tmp_path, method)
    descriptor = metadata["construction"]["parameter_history"]
    descriptor["key"] = "history_9999"
    with pytest.raises(ValueError, match="not part of the authenticated bundle"):
        study.load_parameter_history(folder, descriptor)


@pytest.mark.parametrize("index", [-1, 2, True, "0"])
def test_candidate_reference_requires_an_actual_in_range_integer(tmp_path: Path, index: object) -> None:
    """Negative, boolean, string and past-end indexing cannot select a different history."""
    _, _, method = construction_bundle(tmp_path)
    method["candidate_index"] = index
    with pytest.raises(ValueError, match="invalid candidate index"):
        study.load_candidate_metadata(tmp_path, method)


@pytest.mark.parametrize("field", ["candidate_id", "method", "status"])
def test_candidate_reference_binds_candidate_identity(tmp_path: Path, field: str) -> None:
    """Resolved metadata must match all identifying fields stored with final scores."""
    _, _, method = construction_bundle(tmp_path)
    method[field] = "different"
    with pytest.raises(ValueError, match="candidate identity"):
        study.load_candidate_metadata(tmp_path, method)


def test_permuted_candidate_index_cannot_attach_the_wrong_history(tmp_path: Path) -> None:
    """Two valid matrix bundles cannot be silently exchanged by changing a row index."""
    _, _, method = construction_bundle(tmp_path)
    method["candidate_index"] = 1
    with pytest.raises(ValueError, match="candidate identity"):
        study.load_candidate_metadata(tmp_path, method)


def test_candidate_reference_binds_the_construction_seal(tmp_path: Path) -> None:
    """A candidate reference cannot be redirected to newly resealed construction data."""
    folder, _, method = construction_bundle(tmp_path)
    record = json.loads((folder / "record.json").read_text())
    record["candidates"][0]["seed"] = 400
    reseal_record(folder, record)
    with pytest.raises(ValueError, match="Construction reference integrity"):
        study.load_candidate_metadata(tmp_path, method)


@pytest.mark.parametrize("escape", ["absolute", "parent", "symlink"])
def test_candidate_reference_rejects_paths_outside_campaign(tmp_path: Path, escape: str) -> None:
    """Valid hashes never authorize an absolute, traversing, or escaping symbolic link."""
    root = tmp_path / "campaign"
    folder, _, method = construction_bundle(root)
    if escape == "absolute":
        method["construction_reference"]["path"] = str(folder)
    elif escape == "parent":
        method["construction_reference"]["path"] = "../campaign/construction/toy/aqce/0/shared"
    else:
        outside = tmp_path / "outside"
        shutil.copytree(folder, outside)
        (root / "escape").symlink_to(outside, target_is_directory=True)
        method["construction_reference"]["path"] = "escape"
    with pytest.raises(ValueError, match="within the campaign"):
        study.load_candidate_metadata(root, method)


def test_campaign_stores_histories_once_and_portably_resumes_both_hardware_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selection, final and summary use one shared authenticated construction archive."""
    root = tmp_path / "original"
    args = study.make_parser().parse_args([
        "--output",
        str(root),
        "--targets",
        study.TARGET_IDS[0],
        "--methods",
        "aqce",
        "--hardware",
        "IBM",
        "Google",
        "--validation-paths",
        "8",
    ])
    grid = [rich_candidate(), rich_candidate("second")]
    calls = 0

    def construct(*_args: object) -> list[Candidate]:
        nonlocal calls
        calls += 1
        return grid

    monkeypatch.setattr(study, "construct", construct)
    monkeypatch.setattr(study, "build_manifest", lambda _args: {"schema": study.SCHEMA})
    summary = study.run(args)
    assert calls == 1
    assert len(summary["groups"]) == 2
    assert len(list(root.rglob("histories.npz"))) == 1
    assert json.loads((root / "summary.json").read_text()) == summary
    for group in summary["groups"]:
        assert "candidates" not in group["construction"]
        assert group["construction"]["construction_cpu_seconds"] >= 0
        assert group["construction"]["construction_wall_seconds"] >= 0
        for row in group["records"]:
            method = row["method"]
            assert set(method) == {"candidate_id", "method", "status", "candidate_index", "construction_reference"}
            resolved = study.load_candidate_metadata(root, method)
            assert resolved["seed"] == 300
            assert resolved["training_bank"] == grid[method["candidate_index"]].metadata["training_bank"]
    for path in [root / "summary.json", *root.glob("evaluation/**/record.json")]:
        saved = json.loads(path.read_text())
        assert histories.history_descriptors(saved) == []
        assert '"theta"' not in path.read_text()
        assert '"fidelity_trace"' not in path.read_text()
    moved = tmp_path / "moved"
    shutil.move(str(root), moved)
    args.output = moved
    for row in summary["groups"][0]["records"]:
        resolved = study.load_candidate_metadata(moved, row["method"])
        folder = moved / row["method"]["construction_reference"]["path"]
        values = study.load_parameter_history(folder, resolved["construction"]["parameter_history"])
        assert values.shape == (2, 4)
        assert np.signbit(values[0, 0])
    monkeypatch.setattr(study, "construct", fail_if_called)
    monkeypatch.setattr(study, "select_candidate", fail_if_called)
    for name in ("compile_candidate", "sample_validation", "validation_score", "exact_score"):
        monkeypatch.setattr(study.evaluation, name, fail_if_called)
    assert study.run(args) == summary
