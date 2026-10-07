# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check authenticated growth, unchanged updates, costs and safe branch recovery."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from experiments import topdown_haar_matrix_continuation as study

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def target() -> np.ndarray:
    """Return a finite complex target for inexpensive exact optimizer comparisons."""
    return np.array([1, 4j, -2, 3j]) / np.sqrt(30)


@pytest.fixture
def parent(tmp_path: Path, target: np.ndarray) -> Path:
    """Create a genuine completed matrix arm and its declared supervisor receipt.

    Returns:
        The completed parent job folder.
    """
    root = tmp_path / "original"
    job_id = "haar_random_1__near_zero__seed300"
    folder = root / "jobs" / job_id
    budget = study.base.Budget(online_steps=1, batch_steps=3, max_depth=2, target_fidelity=1)
    study.matrix.run(folder, target, "haar_random_1", "near_zero", 300, budget)
    command = [sys.executable, "-m", "experiments.topdown_initialization_matrix", "--action", "run"]
    job = {"job_id": job_id, "target_id": "haar_random_1", "arm": "near_zero", "seed": 300, "command": command}
    study.support.write_json(
        root / "plan.json",
        {
            "sources": study.matrix.sources(),
            "jobs": [job],
            "budget": asdict(budget),
            "target_sha256": {"haar_random_1": study.base.capacity.topdown_mps.target_digest(target)},
            "fixture_sha256": study.support.digest(study.base.DEFAULT_TARGET_PATH),
            "environment": {"OPENBLAS_NUM_THREADS": "1"},
        },
    )
    study.support.write_json(
        root / "logs" / (job_id + ".process.json"),
        {"status": "completed", "exit_code": 0, "command": command, "wall_seconds": 2.0},
    )
    study.support.write_json(root / "claims" / job_id / "launch.json", {"command": ["supervisor", "--", *command]})
    return folder


def test_sources_include_original_numerics() -> None:
    """The branch includes all inherited hashes and its own driver."""
    hashes = study.sources()
    assert hashes.items() >= study.matrix.sources().items()
    assert "experiments/topdown_haar_matrix_continuation.py" in hashes
    assert "src/mqt/yaqs/optimization/statevector.py" in hashes


def test_authenticate_private_parent_replay(parent: Path, target: np.ndarray) -> None:
    """Only completed, replayed depth endpoints become independent branch inputs."""
    manifest, budget, theta, refs = study.authenticate(parent, target)
    original = study.support.load_checkpoint(parent / "depth_002/construction")
    assert original is not None
    assert manifest["arm"] == "near_zero"
    assert budget.max_depth == refs["start_depth"] == 2
    assert refs["initial_fidelity"] == original[0]["fidelity"]
    np.testing.assert_array_equal(theta, original[1]["theta"])
    theta[0] += 1
    replayed = study.support.load_checkpoint(parent / "depth_002/construction")
    assert replayed is not None
    np.testing.assert_array_equal(replayed[1]["theta"], original[1]["theta"])


@pytest.mark.parametrize("kind", ["source", "target", "receipt", "claim", "marker", "previous", "final", "best"])
def test_bad_parent_rejected(parent: Path, target: np.ndarray, kind: str) -> None:
    """Changed provenance, unsuccessful exits and damaged arrays cannot restart."""
    root = parent.parent.parent
    if kind == "source":
        path = parent / "manifest.json"
        record = json.loads(path.read_text())
        record["sources"] = {}
        study.support.write_json(path, record)
    elif kind == "target":
        target = target[::-1]
    elif kind in {"receipt", "claim"}:
        path = (
            root
            / ("logs" if kind == "receipt" else "claims")
            / (parent.name + ".process.json" if kind == "receipt" else parent.name + "/launch.json")
        )
        record = json.loads(path.read_text())
        if kind == "receipt":
            record["exit_code"] = 1
        else:
            record["command"][-1] = "different"
        study.support.write_json(path, record)
    elif kind == "marker":
        study.support.write_json(parent / "matrix-complete.json", {})
    elif kind in {"previous", "final"}:
        (parent / ("depth_001/construction/arrays.npz" if kind == "previous" else "final/arrays.npz")).write_bytes(
            b"corrupt"
        )
    else:
        loaded = study.support.load_checkpoint(parent / "final")
        assert loaded is not None
        record, arrays = loaded
        record["best_fidelity"] += 0.1
        study.support.save_checkpoint(parent / "final", record, arrays)
        study.support.write_json(
            parent / "matrix-complete.json",
            {
                "manifest_sha256": study.support.digest(parent / "matrix-manifest.json"),
                "final_seal_sha256": study.support.digest(parent / "final/complete.json"),
            },
        )
    with pytest.raises(ValueError, match=r"provenance|completion|checksum|endpoint"):
        study.authenticate(parent, target)


def test_continuation_exact_updates_and_inherited_inputs(parent: Path, target: np.ndarray, tmp_path: Path) -> None:
    """Every new angle and continuous optimizer update matches the old procedure."""
    _, old, previous, refs = study.authenticate(parent, target)
    budget = replace(old, max_depth=4, max_cpu_seconds=1000)
    branch = tmp_path / "branch"
    result = study.run(branch, parent, target, 4, 1000)
    for depth in (3, 4):
        circuit = study.base.capacity.continuation.brickwall(2, depth, initial_single_qubit_layer=True)
        previous = study.base.capacity.continuation.legacy.extend(previous, circuit.num_params, 1300 + depth, 0.001)
        expected, arrays = study.base.train_depth(tmp_path / f"expected_{depth}", target, previous, depth, budget)
        actual = study.support.load_checkpoint(branch / f"depth_{depth:03d}/construction")
        assert actual is not None
        np.testing.assert_array_equal(actual[1]["theta"], arrays["theta"])
        np.testing.assert_array_equal(actual[1]["fidelity"], arrays["fidelity"])
        np.testing.assert_array_equal(actual[1][actual[0]["training"]["initial_theta"]["key"]], previous)
        assert actual[0]["fidelity"] == expected["fidelity"]
        updates = json.loads(actual[1]["optimizer_records_utf8"].tobytes())
        reference = json.loads(arrays["optimizer_records_utf8"].tobytes())
        for a, b in zip(updates, reference, strict=True):
            for key in ("phase", "fidelity", "accepted", "trial_rates", "history_pairs", "curvature_dot"):
                assert a[key] == b[key]
        previous = arrays["theta"]
    assert result["stop_reason"] == "depth_budget"
    assert [r["depth"] for r in result["completed_depths"]] == [3, 4]
    assert result["historical_cpu_seconds"] == refs["historical_cpu_seconds"]
    assert result["new_cpu_seconds"] == pytest.approx(
        result["setup_cpu_seconds"] + sum(r["stage_cpu_seconds"] for r in result["completed_depths"])
    )
    assert result["total_preparation_cpu_seconds"] == pytest.approx(
        result["historical_cpu_seconds"] + result["new_cpu_seconds"]
    )


@pytest.mark.parametrize(("depth", "cpu"), [(2, 100), (True, 100), (3, 0), (3, np.inf), (3, True)])
def test_invalid_budget_writes_no_output(
    parent: Path, target: np.ndarray, tmp_path: Path, depth: int, cpu: float
) -> None:
    """Reject exhausted or invalid branch limits before producing a manifest."""
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="budget"):
        study.run(output, parent, target, depth, cpu)
    assert not output.exists()


@pytest.mark.parametrize("where", ["same", "inside", "outside"])
def test_separate_output_required(parent: Path, target: np.ndarray, where: str) -> None:
    """Neither nesting direction can overwrite the frozen campaign."""
    path = parent if where == "same" else parent / "branch" if where == "inside" else parent.parent
    with pytest.raises(ValueError, match="separate"):
        study.run(path, parent, target, 3)
    with pytest.raises(ValueError, match="separate"):
        study.prepare_plan(path, parent, 30)


def test_sealed_resume_and_changed_limits(
    parent: Path, target: np.ndarray, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completed calls replay without training, while different budgets are refused."""
    output = tmp_path / "branch"
    final = study.run(output, parent, target, 3, 1000)
    monkeypatch.setattr(study.base.capacity.policies, "train_noiseless", lambda *_a, **_k: pytest.fail("retrained"))
    assert study.run(output, parent, target, 3, 1000) == final
    with pytest.raises(ValueError, match="manifest"):
        study.run(output, parent, target, 4, 1000)


def test_interrupted_batch_refuses_live_restart(
    parent: Path, target: np.ndarray, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observational live angles never substitute for optimizer curvature history."""
    output = tmp_path / "branch"
    monkeypatch.setattr(
        study.base.capacity.policies,
        "train_noiseless",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        study.run(output, parent, target, 3)
    with pytest.raises(ValueError, match="Incomplete continuous"):
        study.run(output, parent, target, 3)


def test_new_cpu_budget_is_separate_from_history(parent: Path, target: np.ndarray, tmp_path: Path) -> None:
    """New limits stop between completed depths and preserve the parent best."""
    result = study.run(tmp_path / "branch", parent, target, 4, 1e-12)
    assert result["status"] == "budget_exhausted"
    assert result["stop_reason"] == "cpu_budget"
    assert len(result["completed_depths"]) == 1
    assert result["best_fidelity"] >= result["initial_fidelity"]


def test_plan_requires_all_eighteen_cases(parent: Path, tmp_path: Path) -> None:
    """A partial or duplicate parent matrix cannot silently shrink the study."""
    with pytest.raises(ValueError, match="eighteen"):
        study.prepare_plan(tmp_path / "plan", parent.parent.parent)


def test_complete_non_launching_plan(
    parent: Path, tmp_path: Path, target: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Commands carry the declared limits and all paired identifiers exactly once."""
    root = parent.parent.parent
    plan = json.loads((root / "plan.json").read_text())
    plan["jobs"] = [
        {"job_id": f"{t}__{a}__seed{s}", "target_id": t, "arm": a, "seed": s}
        for t in study.TARGETS
        for a in study.matrix.ARMS
        for s in study.base.SEEDS
    ]
    study.support.write_json(root / "plan.json", plan)

    class Fixture:
        @staticmethod
        def state_vector_copy() -> np.ndarray:
            return target.copy()

    monkeypatch.setattr(study.matrix, "load_target", lambda *_a: Fixture())
    monkeypatch.setattr(study, "authenticate", lambda *_a: ({}, study.base.Budget(), np.zeros(1), {"start_depth": 16}))
    output = tmp_path / "plan"
    result = study.prepare_plan(output, root)
    assert len(result["jobs"]) == 18
    assert not result["launched"]
    assert not (output / "jobs").exists()
    assert result["deployment_guards"] == {"wall_seconds_per_job": 43200, "memory_gib_per_job": 16}
    assert all(j["budget"]["online_steps"] == 100 and j["budget"]["batch_steps"] == 2000 for j in result["jobs"])
    assert all(j["budget"]["max_depth"] == 30 and j["budget"]["max_cpu_seconds"] == 32400 for j in result["jobs"])
    assert all(j["command"][-4:] == ["--max-depth", "30", "--max-cpu-seconds", "32400.0"] for j in result["jobs"])
    assert study.prepare_plan(output, root) == result
    with pytest.raises(ValueError, match="plan"):
        study.prepare_plan(output, root, 31)


def test_cli_plan_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Command-line default prepares a plan without invoking numerical training."""
    calls = []
    monkeypatch.setattr(study, "prepare_plan", lambda *args: calls.append(args) or {"launched": False})
    monkeypatch.setattr(study, "run", lambda *_a: pytest.fail("launched"))
    study.main(["--parent", str(tmp_path / "original"), "--output", str(tmp_path / "new")])
    assert calls[0][2:] == (30, 32400)
