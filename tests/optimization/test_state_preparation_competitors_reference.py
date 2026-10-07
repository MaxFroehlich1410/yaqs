# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Receipt integrity and unchanged parameter selection for the frozen incumbent."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH, load_target_collection
from experiments import topdown_depth_search, topdown_optimizer_stability
from experiments.legacy_topdown_weighted import file_hash, load_circuit
from experiments.state_preparation_competitors import reference as study
from experiments.state_preparation_competitors.common import from_parameterized
from experiments.topdown_batch_refinement import protocol
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path

TARGET_ID = "haar_random_1"


def write(path: Path, value: dict) -> None:
    """Write a synthetic test receipt without depending on production outputs."""
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def seal(case: Path, complete: dict) -> None:
    """Regenerate consistent outer receipts when testing semantic validation."""
    complete["selection_sha256"] = file_hash(case / "selection_locked.json")
    topdown_optimizer_stability.save_complete(case, complete)


def rewrite_checkpoint(case: Path, name: str, record: dict) -> None:
    """Rehash an intentionally modified checkpoint to exercise deeper checks."""
    path = case / "search" / f"{name}.json"
    write(path, record)
    write(path.with_suffix(".sha256.json"), {path.name: file_hash(path)})
    seal(case, json.loads((case / "complete.json").read_text()))


@pytest.fixture
def archive(tmp_path: Path) -> tuple[Path, Path, np.ndarray]:
    """Create a tiny archive using the real fixture and original receipt writers.

    Returns:
        Archive root, its single case directory, and the exact fixture target.
    """
    root = tmp_path / "archive"
    case = root / "pipeline" / "lbfgs" / f"{TARGET_ID}__seed300"
    case.mkdir(parents=True)
    fixture = load_target_collection().load(6, TARGET_ID)
    target = fixture.state_vector_copy()
    np.save(case / "target.npy", target)
    write(
        root / "manifest.json",
        {
            "protocol": protocol(),
            "fixture_sha256": file_hash(DEFAULT_TARGET_PATH),
            "target_ids": [TARGET_ID],
            "optimizer_seeds": [300],
            "sources": {name: file_hash(study.REPOSITORY / name) for name in study.REQUIRED_SOURCES},
        },
    )
    circuit = ParameterizedCircuit(
        6,
        [
            ParameterizedGate("rx", (0,), 0),
            ParameterizedGate("rz", (4,), 1, angle_scale=-0.7, angle_offset=0.2),
            ParameterizedGate("rxx", (1, 5), 2),
        ],
        3,
    )
    branches, results = [], []
    for hardware, depth in (("IBM", 2), ("Google", 1)):
        theta = np.array([0.4, -0.5, 0.6]) * depth
        topdown_depth_search.save_checkpoint(
            case / "search",
            f"construction_{depth:02d}",
            circuit,
            theta,
            {
                "policy": "lbfgs",
                "depth": depth,
            },
        )
        topdown_depth_search.save_checkpoint(
            case / "search",
            f"depth_{depth:02d}_B_pruned",
            circuit,
            theta,
            {
                "depth": depth,
                "pruning": {"variant": "B"},
            },
        )
        bank = f"depth_{depth:02d}_B_{hardware}.validation.npz"
        np.savez_compressed(case / "search" / bank, codes=np.zeros((4, 3), dtype=np.uint8))
        theta += np.array([0.1, 0.2, -0.3])
        training = {
            "steps": 200,
            "online_steps": 100,
            "rate": 0.2,
            "decay": 0.0,
            "seed": 5300,
            "crn": True,
            "noisy": True,
            "hardware": hardware,
            "num_trajectories": 100,
            "compress_no_jump": True,
        }
        topdown_depth_search.save_checkpoint(
            case / "search",
            f"depth_{depth:02d}_B_{hardware}",
            circuit,
            theta,
            {
                "depth": depth,
                "variant": "B",
                "hardware": hardware,
                "status": "complete",
                "training_paths": 100,
                "training": training,
                "validation": {
                    "bank_file": bank,
                    "bank_sha256": file_hash(case / "search" / bank),
                    "fidelity": 0.2,
                    "entropy": [20260914, 6, 4, 300, 0, 0, depth, 90000],
                },
            },
        )
        ideal = float(abs(np.vdot(target, Statevector.from_instruction(from_parameterized(circuit, theta)).data)) ** 2)
        branch = {
            "variant": "B",
            "hardware": hardware,
            "selected_depth": depth,
            "stop_reason": "patience",
            "decisions": [],
        }
        branches.append(branch)
        results.append({
            **branch,
            "selected": {
                "variant": "B",
                "hardware": hardware,
                "depth": depth,
                "status": "complete",
                "ideal": {"fidelity": ideal},
                "exact": {"fidelity": 0.01},
            },
            # These tempting test scores are irrelevant to loading the saved B winner.
            "evaluations": [{"depth": 15, "exact": {"fidelity": 1.0}}],
            "search_cost": {
                "construction_cpu_seconds": 1.0,
                "pruning_cpu_seconds": 0.2,
                "fine_tuning_cpu_seconds": 2.0,
                "validation_cpu_seconds": 0.3,
                "total_search_cpu_seconds": 3.5,
                "candidates": 5,
            },
        })
    branches.append({"variant": "D", "hardware": "IBM", "selected_depth": 15})
    results.append({"variant": "D", "hardware": "IBM", "selected_depth": 15, "selected": {"exact": {"fidelity": 1.0}}})
    identity = {"target_id": TARGET_ID, "optimizer_seed": 300, "policy": "lbfgs"}
    write(case / "selection_locked.json", {**identity, "branches": branches})
    seal(case, {**identity, "target_generation_seed": fixture.seed, "results": results})
    return root, case, target


@pytest.mark.parametrize(("hardware", "depth"), [("IBM", 2), ("Google", 1)])
def test_exact_archived_parameters_and_selection_are_preserved(
    archive: tuple[Path, Path, np.ndarray], hardware: str, depth: int
) -> None:
    """Do not choose a better-looking D/test endpoint or modify the saved angles."""
    root, case, target = archive
    before = {str(path): file_hash(path) for path in root.rglob("*") if path.is_file()}
    actual = study.frozen_reference(root, TARGET_ID, 300, hardware, target)
    theta = np.array([0.4, -0.5, 0.6]) * depth + np.array([0.1, 0.2, -0.3])
    expected = QuantumCircuit(6)
    expected.rx(theta[0], 0)
    expected.rz(-0.7 * theta[1] + 0.2, 4)
    expected.rxx(theta[2], 1, 5)
    np.testing.assert_array_equal(Operator(actual.circuit).data, Operator(expected).data)
    assert actual.metadata["selected_depth"] == depth
    assert actual.metadata["method"] == "current_b_lbfgs_frozen"
    assert actual.metadata["archived_search_cost"]["total_search_cpu_seconds"] == pytest.approx(3.5)
    assert actual.metadata["new_training_cpu_seconds"] == 0
    assert not actual.metadata["reselected_under_native_noise"]
    assert not actual.metadata["new_noise_fine_tuning"]
    assert not actual.metadata["deterministic"]
    assert actual.metadata["archive_selection_sha256"] == file_hash(case / "selection_locked.json")
    assert actual.metadata["load_cpu_seconds"] >= 0
    assert actual.metadata["cpu_seconds"] == actual.metadata["load_cpu_seconds"]
    json.dumps(actual.metadata, allow_nan=False)
    assert before == {str(path): file_hash(path) for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize(
    ("target_id", "seed", "hardware"),
    [
        ("../haar_random_1", 300, "IBM"),
        (TARGET_ID, 42, "IBM"),
        (TARGET_ID, True, "IBM"),
        (TARGET_ID, 300.0, "IBM"),
        (TARGET_ID, 300, "unknown"),
    ],
)
def test_invalid_identifiers_fail_before_accessing_archive(
    archive: tuple[Path, Path, np.ndarray], target_id: str, seed: int, hardware: str
) -> None:
    """Neither path traversal nor unsupported optimizer/profile identities are accepted."""
    root, _, target = archive
    with pytest.raises(ValueError, match="fixed target"):
        study.frozen_reference(root, target_id, seed, hardware, target)


def test_target_identity_includes_fixture_amplitudes_and_phase(archive: tuple[Path, Path, np.ndarray]) -> None:
    """A normalized state with different amplitudes or fixture phase is not the same fixture."""
    root, _, target = archive
    for changed in (np.roll(target, 1), np.exp(0.3j) * target):
        with pytest.raises(ValueError, match="exact six-qubit target fixture"):
            study.frozen_reference(root, TARGET_ID, 300, "IBM", changed)


@pytest.mark.parametrize("member", ["target.npy", "selection_locked.json", "search/depth_02_B_IBM.json"])
def test_tampered_artifacts_fail_outer_checksums(archive: tuple[Path, Path, np.ndarray], member: str) -> None:
    """Receipt validation covers target, decision, and parameter bytes before reuse."""
    root, case, target = archive
    path = case / member
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="artifacts changed"):
        study.frozen_reference(root, TARGET_ID, 300, "IBM", target)


@pytest.mark.parametrize("change", ["fixture", "protocol", "source", "missing_source"])
def test_manifest_identity_and_frozen_sources_are_checked(archive: tuple[Path, Path, np.ndarray], change: str) -> None:
    """A different method protocol or source cannot masquerade as the incumbent."""
    root, _, target = archive
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "fixture":
        manifest["fixture_sha256"] = "0" * 64
    elif change == "protocol":
        manifest["protocol"]["fine_tuning"]["num_trajectories"] = 400
    elif change == "source":
        manifest["sources"][study.REQUIRED_SOURCES[0]] = "0" * 64
    else:
        manifest["sources"].pop(study.REQUIRED_SOURCES[0])
    write(path, manifest)
    with pytest.raises(ValueError, match=r"manifest|source checksum"):
        study.frozen_reference(root, TARGET_ID, 300, "IBM", target)


@pytest.mark.parametrize("member", ["../outside.json", "/outside.json", ""])
def test_receipt_paths_cannot_escape_case(archive: tuple[Path, Path, np.ndarray], member: str) -> None:
    """Invalid receipt paths are rejected before the legacy loader reads them."""
    root, case, target = archive
    complete = json.loads((case / "complete.json").read_text())
    complete["artifacts"][member] = "0" * 64
    write(case / "complete.json", complete)
    with pytest.raises(ValueError, match="Archive member"):
        study.frozen_reference(root, TARGET_ID, 300, "IBM", target)


def test_symlink_escape_is_rejected(archive: tuple[Path, Path, np.ndarray]) -> None:
    """Relative spelling must not allow following an artifact outside its receipt."""
    root, case, target = archive
    outside = root / "outside.json"
    write(outside, {})
    (case / "escaped.json").symlink_to(outside)
    complete = json.loads((case / "complete.json").read_text())
    complete["artifacts"]["escaped.json"] = file_hash(outside)
    write(case / "complete.json", complete)
    with pytest.raises(ValueError, match="escapes"):
        study.frozen_reference(root, TARGET_ID, 300, "IBM", target)


@pytest.mark.parametrize("change", ["missing_B", "duplicate_B", "policy", "depth", "selection_hash"])
def test_locked_selection_cannot_be_substituted(archive: tuple[Path, Path, np.ndarray], change: str) -> None:
    """Require matching policy and B branch; never fall back to D or another depth."""
    root, case, target = archive
    selection = json.loads((case / "selection_locked.json").read_text())
    complete = json.loads((case / "complete.json").read_text())
    if change == "missing_B":
        selection["branches"] = [branch for branch in selection["branches"] if branch["variant"] != "B"]
    elif change == "duplicate_B":
        selection["branches"].append(selection["branches"][0])
    elif change == "policy":
        selection["policy"] = "best_only"
    elif change == "depth":
        selection["branches"][0]["selected_depth"] = 1
    write(case / "selection_locked.json", selection)
    seal(case, complete)
    if change == "selection_hash":
        complete = json.loads((case / "complete.json").read_text())
        complete["selection_sha256"] = "0" * 64
        write(case / "complete.json", complete)
    with pytest.raises(ValueError, match=r"branch|selection|endpoint"):
        study.frozen_reference(root, TARGET_ID, 300, "IBM", target)


def test_checkpoint_requires_its_exact_original_sidecar(archive: tuple[Path, Path, np.ndarray]) -> None:
    """An empty rehashed sidecar must not bypass the original checksum mechanism."""
    root, case, target = archive
    write(case / "search/depth_02_B_IBM.sha256.json", {})
    seal(case, json.loads((case / "complete.json").read_text()))
    with pytest.raises(ValueError, match="sidecar"):
        study.frozen_reference(root, TARGET_ID, 300, "IBM", target)


@pytest.mark.parametrize("change", ["parameters", "D_pruning", "training_paths", "construction_policy"])
def test_rehashed_checkpoint_must_replay_the_same_method(archive: tuple[Path, Path, np.ndarray], change: str) -> None:
    """Internal consistency checks catch substitutions beyond simple byte damage."""
    root, case, target = archive
    name = "depth_02_B_IBM"
    if change == "D_pruning":
        name = "depth_02_B_pruned"
    elif change == "construction_policy":
        name = "construction_02"
    record = json.loads((case / "search" / f"{name}.json").read_text())
    if change == "parameters":
        record["theta"][0] += 0.5
    elif change == "D_pruning":
        record["pruning"]["variant"] = "D"
    elif change == "construction_policy":
        record["policy"] = "best_only"
    else:
        record["training_paths"] = 400
    rewrite_checkpoint(case, name, record)
    with pytest.raises(ValueError, match=r"procedure|ideal-fidelity replay"):
        study.frozen_reference(root, TARGET_ID, 300, "IBM", target)


def test_original_search_cost_is_not_replaced_by_selected_candidate_cost(
    archive: tuple[Path, Path, np.ndarray],
) -> None:
    """All old search stages must sum correctly; a shortened or corrupt cost is rejected."""
    root, case, target = archive
    complete = json.loads((case / "complete.json").read_text())
    complete["results"][0]["search_cost"]["total_search_cpu_seconds"] = 0.3
    write(case / "complete.json", complete)
    with pytest.raises(ValueError, match="search cost"):
        study.frozen_reference(root, TARGET_ID, 300, "IBM", target)


@pytest.mark.skipif(not study.DEFAULT_REFERENCE_ROOT.is_dir(), reason="Optional production-archive parity check")
@pytest.mark.parametrize("hardware", ["IBM", "Google"])
def test_real_archive_selected_circuit_matches_every_saved_parameter(hardware: str) -> None:
    """Use real archive data when present; portable synthetic tests cover CI."""
    target = load_target_collection().load(6, TARGET_ID).state_vector_copy()
    actual = study.frozen_reference(study.DEFAULT_REFERENCE_ROOT, TARGET_ID, 400, hardware, target)
    case = study.DEFAULT_REFERENCE_ROOT / "pipeline/lbfgs" / f"{TARGET_ID}__seed400"
    depth = actual.metadata["selected_depth"]
    record = topdown_depth_search.read_checkpoint(case / "search", f"depth_{depth:02d}_B_{hardware}")
    assert record is not None
    expected = from_parameterized(load_circuit(record), np.array(record["theta"]))
    assert actual.circuit == expected
    np.testing.assert_array_equal(Operator(actual.circuit).data, Operator(expected).data)


def test_source_snapshots_survive_checkout_changes_and_reject_tampering(
    archive: tuple[Path, Path, np.ndarray], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Frozen source hashes describe archived bytes, even as development continues."""
    root, _, target = archive
    checkout = tmp_path / "changed_checkout"
    for name in study.REQUIRED_SOURCES:
        snapshot = root / "source" / f"{name}.txt"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((study.REPOSITORY / name).read_bytes())
        changed = checkout / name
        changed.parent.mkdir(parents=True, exist_ok=True)
        changed.write_text("# Changed after the experiment.\n")
    monkeypatch.setattr(study, "REPOSITORY", checkout)
    result = study.frozen_reference(root, TARGET_ID, 300, "IBM", target)
    assert result.metadata["selected_depth"] == 2
    snapshot.write_text("# Corrupted archived source.\n")
    with pytest.raises(ValueError, match="source checksum"):
        study.frozen_reference(root, TARGET_ID, 300, "IBM", target)
