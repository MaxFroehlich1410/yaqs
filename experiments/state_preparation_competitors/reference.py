# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Load the archived B + L-BFGS incumbent without changing its selection.

The archived winner was selected under noise on logical rotation gates. It is
loaded unchanged for a separate compiled-native-noise deployment comparison.
Hash receipts establish archive integrity, not a digital signature of authorship.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit.quantum_info import Statevector

from benchmarks.state_preparation.constants import TARGET_IDS
from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH, load_target_collection
from experiments import topdown_depth_search, topdown_optimizer_stability
from experiments.legacy_topdown_q6 import REPOSITORY
from experiments.legacy_topdown_weighted import file_hash, load_circuit
from experiments.state_preparation_competitors.common import Candidate, from_parameterized, validate_target
from experiments.topdown_batch_refinement import protocol

if TYPE_CHECKING:
    from numpy.typing import NDArray

DEFAULT_REFERENCE_ROOT = REPOSITORY / "output/topdown-batch-refinement-q6-20260919"
REQUIRED_SOURCES = (
    "experiments/topdown_batch_refinement.py",
    "experiments/topdown_batch_refinement_policies.py",
    "experiments/topdown_depth_search.py",
    "experiments/legacy_topdown_weighted.py",
)


def archive_member(root: Path, name: str) -> Path:
    """Resolve a receipt member while preventing absolute paths and escapes.

    Args:
        root: The directory whose artifacts a receipt authenticates.
        name: A nonempty relative path stored in that receipt.

    Returns:
        A resolved path inside the receipt directory.

    Raises:
        ValueError: If the path is absolute, empty, or escapes through traversal
            or a symbolic link.
    """
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        msg = "Archive members must be nonempty relative paths."
        raise ValueError(msg)
    member = (root / name).resolve()
    if not member.is_relative_to(root.resolve()) or member == root.resolve():
        msg = "Archive member escapes its receipt directory."
        raise ValueError(msg)
    return member


def authenticated_checkpoint(case: Path, name: str, artifacts: dict[str, str]) -> dict[str, Any]:
    """Require both the case receipt and the original checkpoint sidecar.

    Args:
        case: Completed target/seed case directory.
        name: Checkpoint stem within its ``search`` subdirectory.
        artifacts: Already verified completion-receipt digests.

    Returns:
        The checkpoint decoded by the archive's original loader.

    Raises:
        ValueError: If a required receipt member or checkpoint is missing, or
            its original checksum sidecar does not match the completion receipt.
    """
    record_name = f"search/{name}.json"
    marker_name = f"search/{name}.sha256.json"
    if record_name not in artifacts or marker_name not in artifacts:
        msg = "Selected archive checkpoint or its checksum sidecar is unauthenticated."
        raise ValueError(msg)
    marker = json.loads(archive_member(case, marker_name).read_text())
    if marker != {f"{name}.json": artifacts[record_name]}:
        msg = "Checkpoint checksum sidecar disagrees with the completion receipt."
        raise ValueError(msg)
    record = topdown_depth_search.read_checkpoint(case / "search", name)
    if record is None:
        msg = "Selected archive checkpoint is missing."
        raise ValueError(msg)
    return record


def selected_branch(branches: list[dict[str, Any]], hardware: str) -> dict[str, Any]:
    """Find exactly one archived B branch without ranking any test fidelity.

    Args:
        branches: Branch records from the selection or completion receipt.
        hardware: The requested historical noise profile.

    Returns:
        The matching B branch as originally saved.

    Raises:
        ValueError: If the B branch is absent or duplicated.
    """
    matches = [branch for branch in branches if branch.get("variant") == "B" and branch.get("hardware") == hardware]
    if len(matches) != 1:
        msg = "Expected exactly one archived B branch for the requested hardware."
        raise ValueError(msg)
    return matches[0]


def frozen_reference(root: Path, target_id: str, seed: int, hardware: str, target: NDArray[np.complex128]) -> Candidate:
    """Authenticate and return the exact archived B + L-BFGS selected circuit.

    Args:
        root: Root of the completed batch-refinement archive.
        target_id: One of the nine fixed six-qubit benchmark targets.
        seed: Archived optimizer seed 300 or 400.
        hardware: Historical IBM or Google synthetic noise profile.
        target: Exact fixture amplitudes, including their fixed global phase.

    Returns:
        Uncompiled archived circuit, with old search cost separate from new load
        cost. Neither its depth, its gates, nor its parameters are optimized.

    Raises:
        ValueError: If requested identifiers, fixture, receipts, protocol,
            checkpoint identity, or independent ideal fidelity do not match.
        OSError: If an archive file is missing or unreadable.
    """
    started = time.process_time()
    value = validate_target(target)
    if (
        target_id not in TARGET_IDS
        or type(seed) is not int
        or seed not in {300, 400}
        or hardware not in {"IBM", "Google"}
    ):
        msg = "Expected a fixed target, optimizer seed 300/400, and IBM/Google hardware."
        raise ValueError(msg)
    fixture = load_target_collection()
    fixture_record = fixture.load(6, target_id)
    if not np.array_equal(value, fixture_record.state_vector_copy()):
        msg = "Requested target differs from the exact six-qubit target fixture."
        raise ValueError(msg)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("fixture_sha256") != file_hash(DEFAULT_TARGET_PATH)
        or manifest.get("protocol") != protocol()
        or target_id not in manifest.get("target_ids", [])
        or seed not in manifest.get("optimizer_seeds", [])
    ):
        msg = "Reference manifest does not match the frozen fixture, protocol, or requested case."
        raise ValueError(msg)
    sources = manifest.get("sources", {})
    if not all(name in sources for name in REQUIRED_SOURCES):
        msg = "Reference manifest is missing required frozen source hashes."
        raise ValueError(msg)
    for name, digest in sources.items():
        # Completed studies freeze their source bytes alongside the receipts.
        # Authenticate those bytes so later development cannot invalidate an
        # unchanged result. Older archives without snapshots use the checkout.
        current = archive_member(REPOSITORY, name)
        snapshot = archive_member(root, f"source/{name}.txt")
        source = snapshot if snapshot.exists() else current
        if file_hash(source) != digest:
            msg = f"Frozen reference source checksum mismatch: {name}"
            raise ValueError(msg)
    case = archive_member(root, f"pipeline/lbfgs/{target_id}__seed{seed}")
    complete_path = case / "complete.json"
    receipt = json.loads(complete_path.read_text())
    artifacts = receipt.get("artifacts", {})
    if not {"target.npy", "selection_locked.json"}.issubset(artifacts):
        msg = "Reference completion receipt must authenticate target and locked selection."
        raise ValueError(msg)
    for name in artifacts:
        archive_member(case, name)
    # Reuse the archive's existing all-artifact authentication machinery.
    complete = topdown_optimizer_stability.load_complete(case)
    if complete is None:
        msg = "Reference case is incomplete."
        raise ValueError(msg)
    saved_target = np.load(case / "target.npy", allow_pickle=False)
    if not np.array_equal(saved_target, value) or complete.get("target_generation_seed") != fixture_record.seed:
        msg = "Archived target does not match the requested target fixture."
        raise ValueError(msg)
    selection_path = case / "selection_locked.json"
    selection = json.loads(selection_path.read_text())
    identity = {"target_id": target_id, "optimizer_seed": seed, "policy": "lbfgs"}
    if any(record.get(key) != expected for record in (complete, selection) for key, expected in identity.items()):
        msg = "Archived case or selection is not the requested L-BFGS target/seed."
        raise ValueError(msg)
    selection_hash = file_hash(selection_path)
    if complete.get("selection_sha256") != selection_hash:
        msg = "Locked selection checksum mismatch."
        raise ValueError(msg)
    branch = selected_branch(selection["branches"], hardware)
    result = selected_branch(complete["results"], hardware)
    depth = branch.get("selected_depth")
    selected = result.get("selected")
    if (
        type(depth) is not int
        or not 1 <= depth <= manifest["protocol"]["depth_cap"]
        or any(result.get(key) != entry for key, entry in branch.items())
        or not isinstance(selected, dict)
        or any(
            selected.get(key) != entry
            for key, entry in {"depth": depth, "variant": "B", "hardware": hardware, "status": "complete"}.items()
        )
    ):
        msg = "Archived B selection and completed endpoint disagree or are incomplete."
        raise ValueError(msg)
    name = f"depth_{depth:02d}_B_{hardware}"
    record = authenticated_checkpoint(case, name, artifacts)
    construction = authenticated_checkpoint(case, f"construction_{depth:02d}", artifacts)
    pruned = authenticated_checkpoint(case, f"depth_{depth:02d}_B_pruned", artifacts)
    expected_training = {
        "steps": 200,
        "online_steps": 100,
        "rate": 0.2,
        "decay": 0.0,
        "seed": seed + 5000,
        "crn": True,
        "noisy": True,
        "hardware": hardware,
        "num_trajectories": 100,
        "compress_no_jump": True,
    }
    if (
        construction.get("policy") != "lbfgs"
        or construction.get("depth") != depth
        or pruned.get("depth") != depth
        or pruned.get("pruning", {}).get("variant") != "B"
        or record.get("circuit") != pruned.get("circuit")
        or any(
            record.get(key) != entry
            for key, entry in {
                "depth": depth,
                "variant": "B",
                "hardware": hardware,
                "status": "complete",
                "training_paths": 100,
            }.items()
        )
        or any(record.get("training", {}).get(key) != entry for key, entry in expected_training.items())
    ):
        msg = "Selected checkpoint does not preserve the archived B/L-BFGS/M100 procedure."
        raise ValueError(msg)
    validation = record["validation"]
    bank_name = f"search/{validation['bank_file']}"
    if bank_name not in artifacts or validation["bank_sha256"] != artifacts[bank_name]:
        msg = "Archived selected validation bank is unauthenticated."
        raise ValueError(msg)
    theta = np.array(record["theta"], dtype=np.float64)
    circuit = from_parameterized(load_circuit(record), theta)
    ideal = float(abs(np.vdot(value, Statevector.from_instruction(circuit).data)) ** 2)
    if not np.isfinite(ideal) or abs(ideal - selected["ideal"]["fidelity"]) > 1e-9:
        msg = "Archived selected parameters fail independent ideal-fidelity replay."
        raise ValueError(msg)
    cost = result["search_cost"]
    cost_fields = (
        "construction_cpu_seconds",
        "pruning_cpu_seconds",
        "fine_tuning_cpu_seconds",
        "validation_cpu_seconds",
    )
    if any(
        not np.isfinite(cost[key]) or cost[key] < 0 for key in (*cost_fields, "total_search_cpu_seconds")
    ) or not np.isclose(sum(cost[key] for key in cost_fields), cost["total_search_cpu_seconds"], atol=1e-8, rtol=1e-10):
        msg = "Archived search cost is invalid or internally inconsistent."
        raise ValueError(msg)
    load_cpu = time.process_time() - started
    return Candidate(
        circuit,
        {
            "method": "current_b_lbfgs_frozen",
            "candidate_id": f"archive_depth_{depth:02d}",
            "status": "complete",
            "target_id": target_id,
            "seed": seed,
            "hardware": hardware,
            "deterministic": False,
            "frozen_archived_endpoint": True,
            "selected_depth": depth,
            "ideal_fidelity": ideal,
            "archive_root": str(root.resolve()),
            "archive_manifest_sha256": file_hash(manifest_path),
            "archive_complete_sha256": file_hash(complete_path),
            "archive_selection_sha256": selection_hash,
            "archive_checkpoint_sha256": artifacts[f"search/{name}.json"],
            "archive_target_sha256": artifacts["target.npy"],
            "archive_fixture_sha256": manifest["fixture_sha256"],
            "archived_artifacts_checked": len(artifacts),
            "archived_sources_checked": len(sources),
            "theta_sha256": hashlib.sha256(theta.tobytes()).hexdigest(),
            "archived_selection": {
                "criterion": manifest["protocol"]["selection"],
                "stop_reason": branch["stop_reason"],
                "validation_fidelity": validation["fidelity"],
                "validation_entropy": validation["entropy"],
                "validation_bank_sha256": validation["bank_sha256"],
                "noise_placement": "logical rotation gates; original at-most-one-Pauli-jump windows",
            },
            "archived_search_cost": cost,
            "archived_search_cost_scope": (
                "standalone branch including discarded candidates; shared work may recur across branches; "
                "excludes exact evaluation and compilation"
            ),
            "cpu_seconds": load_cpu,
            "load_cpu_seconds": load_cpu,
            "new_training_cpu_seconds": 0.0,
            "reselected_under_native_noise": False,
            "new_noise_fine_tuning": False,
        },
    )
