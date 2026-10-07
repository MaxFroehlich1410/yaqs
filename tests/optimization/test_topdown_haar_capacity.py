# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact optimizer parity, isolated branches, target stopping, and provenance."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from experiments import topdown_haar_capacity as study

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Create a small sealed statevector endpoint in the real parent schema.

    Returns:
        Parent directory, normalized target, and construction configuration.
    """
    monkeypatch.setattr(study.continuation, "sources", dict)
    target = np.array([1, 2j, -3, 4j]) / np.sqrt(30)
    config = study.topdown_mps.Config(backend="statevector", qubits=2, construction_steps=4, construction_online=2)
    path = tmp_path / "parent"
    record = study.continuation.construct(path / "depth_02/construction", target, np.empty(0), config, 2)
    study.support.write_json(
        path / "manifest.json",
        {
            "version": study.continuation.VERSION,
            "execution_backend": "statevector",
            "construction_config": asdict(config),
            "target_id": "test",
            "target_sha256": study.topdown_mps.target_digest(target),
            "sources": {},
        },
    )
    study.support.save_checkpoint(
        path / "final",
        {
            "best_depth": 2,
            "best_ideal_fidelity": record["fidelity"],
            "new_cpu_seconds": record["cpu_seconds"],
            "historical_construction_cpu_seconds": 0.0,
        },
        {},
    )
    return path, target, config


@pytest.mark.parametrize(
    "change",
    [
        {"target_fidelity": 0},
        {"target_fidelity": 1.1},
        {"target_fidelity": np.nan},
        {"target_fidelity": True},
        {"fixed_steps": 0},
        {"fixed_steps": True},
        {"max_depth": 2},
        {"progress_every": 0},
        {"growth_cpu_seconds": np.inf},
        {"growth_cpu_seconds": -1},
    ],
)
def test_bad_budgets(change: dict) -> None:
    """Invalid resource limits and thresholds cannot silently change the question."""
    with pytest.raises(ValueError, match="budget"):
        replace(study.Budget(), **change).validate(2)


def test_observations_preserve_continuous_lbfgs_and_trace(parent: tuple, tmp_path: Path) -> None:
    """Monitoring leaves every accepted point and the entire line search unchanged."""
    path, target, _ = parent
    _, record, arrays, _ = study.authenticate(path, target, "test")
    circuit = study.support.restore_circuit(record["circuit"])
    expected, trace = study.policies.train_noiseless(
        circuit, arrays["theta"], target, policy="lbfgs", steps=30, online=0
    )
    engine = study.ProgressBackend(circuit, target, tmp_path / "observed", 3)
    actual, observed = study.policies.train_noiseless(
        circuit, arrays["theta"], target, policy="lbfgs", steps=30, online=0, engine=engine
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(observed["fidelity_trace"], trace["fidelity_trace"])
    for old, new in zip(trace["records"], observed["records"], strict=True):
        for key in ("accepted", "history_pairs", "trial_rates", "trial_losses", "curvature_dot"):
            assert old[key] == new[key]
    live = study.support.load_checkpoint(tmp_path / "observed/live")
    assert live is not None
    assert live[0]["completed_updates"] == 29
    assert live[0]["fidelity"] == trace["records"][29]["before_fidelity"]
    assert study.StatevectorBackend(circuit, target).metrics(live[1]["theta"])[1] == live[0]["fidelity"]


def test_fixed_branch_exact_parity_compact_storage_and_parent_immutability(parent: tuple, tmp_path: Path) -> None:
    """Longer refinement changes only the update count and resets history only once."""
    path, target, _ = parent
    before = {str(p): p.read_bytes() for p in path.rglob("*") if p.is_file()}
    _, record, arrays, _ = study.authenticate(path, target, "test")
    circuit = study.support.restore_circuit(record["circuit"])
    expected, trace = study.policies.train_noiseless(
        circuit, arrays["theta"], target, policy="lbfgs", steps=9, online=0
    )
    output = tmp_path / "fixed"
    result = study.run(output, path, target, "test", "fixed", study.Budget(fixed_steps=9, target_fidelity=1))
    saved = study.support.load_checkpoint(output / "training")
    assert saved is not None
    np.testing.assert_array_equal(saved[1]["theta"], expected)
    np.testing.assert_array_equal(saved[1]["fidelity"], trace["fidelity_trace"])
    records = json.loads(saved[1]["optimizer_records_utf8"].tobytes())
    assert len(records) == 9
    assert records[-1]["history_pairs"] == trace["records"][-1]["history_pairs"]
    assert saved[0]["training"]["online_sweeps"] == 0
    assert isinstance(saved[0]["training"]["returned_theta"], dict)
    assert result["stop_reason"] == "update_budget"
    assert result["status"] == "budget_exhausted"
    assert {str(p): p.read_bytes() for p in path.rglob("*") if p.is_file()} == before


def test_growth_exact_warm_starts_and_original_per_depth_budget(parent: tuple, tmp_path: Path) -> None:
    """New depths carry their own parameters and use the unchanged construction call."""
    path, target, config = parent
    _, _, arrays, _ = study.authenticate(path, target, "test")
    result = study.run(tmp_path / "grow", path, target, "test", "grow", study.Budget(max_depth=4, target_fidelity=1))
    previous = arrays["theta"]
    for depth in (3, 4):
        expected = study.continuation.construct(tmp_path / f"reference_{depth}", target, previous, config, depth)
        saved = study.support.load_checkpoint(tmp_path / "grow" / f"depth_{depth:03d}/construction")
        reference = study.support.load_checkpoint(tmp_path / f"reference_{depth}")
        assert saved is not None
        assert reference is not None
        np.testing.assert_array_equal(saved[1]["theta"], reference[1]["theta"])
        np.testing.assert_array_equal(saved[1]["fidelity"], reference[1]["fidelity"])
        assert saved[0]["fidelity"] == expected["fidelity"]
        previous = saved[1]["theta"]
    assert result["stop_reason"] == "depth_budget"
    assert len(result["completed_depths"]) == 2


@pytest.mark.parametrize("arm", ["fixed", "grow"])
def test_finished_resume_and_corruption(
    parent: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    """Completed branches authenticate selected arrays and never train a second time."""
    path, target, _ = parent
    output, budget = tmp_path / arm, study.Budget(fixed_steps=3, max_depth=3, target_fidelity=1)
    expected = study.run(output, path, target, "test", arm, budget)
    monkeypatch.setattr(study.policies, "train_noiseless", lambda *_a, **_k: pytest.fail("retrained"))
    assert study.run(output, path, target, "test", arm, budget) == expected
    with pytest.raises(ValueError, match="manifest"):
        study.run(output, path, target, "test", arm, replace(budget, fixed_steps=4))
    selected = output / expected["selected_checkpoint"]
    (selected / "arrays.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        study.run(output, path, target, "test", arm, budget)


def test_threshold_stops_fixed_branch_at_first_hit(parent: tuple, tmp_path: Path) -> None:
    """Threshold attainment is recorded separately from exhaustion of the update budget."""
    path, target, _ = parent
    _, record, arrays, _ = study.authenticate(path, target, "test")
    circuit = study.support.restore_circuit(record["circuit"])
    _, trace = study.policies.train_noiseless(circuit, arrays["theta"], target, policy="lbfgs", steps=5, online=0)
    threshold = (record["fidelity"] + trace["fidelity_trace"][0]) / 2
    result = study.run(tmp_path / "hit", path, target, "test", "fixed", study.Budget(target_fidelity=threshold))
    assert result["status"] == "target_reached"
    assert result["completed_updates"] == 1
    assert result["first_target_update"] == 1


def test_growth_target_and_cpu_stopping(parent: tuple, tmp_path: Path) -> None:
    """A successful depth and a resource-limited depth have distinct receipts."""
    path, target, config = parent
    _, record, arrays, _ = study.authenticate(path, target, "test")
    next_record = study.continuation.construct(tmp_path / "reference", target, arrays["theta"], config, 3)
    threshold = (record["fidelity"] + next_record["fidelity"]) / 2
    success = study.run(tmp_path / "success", path, target, "test", "grow", study.Budget(target_fidelity=threshold))
    assert success["stop_reason"] == "target_reached"
    assert success["first_target_depth"] == 3
    limited = study.run(
        tmp_path / "limited", path, target, "test", "grow", study.Budget(target_fidelity=1, growth_cpu_seconds=1e-12)
    )
    assert limited["stop_reason"] == "cpu_budget"
    assert limited["status"] == "budget_exhausted"
    assert len(limited["completed_depths"]) == 1


def test_already_successful_parent_needs_no_training(
    parent: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A satisfied target does not spend the remaining declared budget."""
    path, target, _ = parent
    monkeypatch.setattr(study.policies, "train_noiseless", lambda *_a, **_k: pytest.fail("unnecessary training"))
    result = study.run(tmp_path / "done", path, target, "test", "fixed", study.Budget(target_fidelity=1e-12))
    assert result["stop_reason"] == "target_reached_at_parent"
    assert result["new_cpu_seconds"] == 0


def test_provenance_and_path_rejection(parent: tuple) -> None:
    """A changed source, target, or output location cannot contaminate the control."""
    path, target, _ = parent
    with pytest.raises(ValueError, match="separate"):
        study.run(path / "child", path, target, "test", "fixed", study.Budget())
    with pytest.raises(ValueError, match="provenance"):
        study.authenticate(path, target[::-1].copy(), "test")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["sources"] = {"changed": "hash"}
    study.support.write_json(path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="provenance"):
        study.authenticate(path, target, "test")


def test_packing_rejects_false_fidelity(parent: tuple, tmp_path: Path) -> None:
    """A valid hash cannot substitute for numerical endpoint verification."""
    path, target, _ = parent
    _, record, arrays, _ = study.authenticate(path, target, "test")
    circuit = study.support.restore_circuit(record["circuit"])
    theta, trace = study.policies.train_noiseless(circuit, arrays["theta"], target, policy="lbfgs", steps=2, online=0)
    bad = copy.deepcopy(trace)
    bad["best_fidelity"] += 0.01
    with pytest.raises(ValueError, match="replay"):
        study.save_training(tmp_path / "invalid", circuit, target, theta, bad)


def test_cli_dispatches_only_haar(parent: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The public entry point cannot silently include Gaussian or change the backend."""
    calls = []
    monkeypatch.setattr(study, "run", lambda *args: calls.append(args) or {})
    study.main(["--parent", str(parent[0]), "--output", str(tmp_path / "run"), "--arm", "grow"])
    assert calls[0][2].shape == (4096,)
    assert calls[0][3:5] == ("haar_random_1", "grow")
    assert calls[0][5] == study.Budget()


def test_interrupted_growth_reuses_completed_depth(
    parent: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restarting an interrupted chain preserves the sealed predecessor and its cost."""
    path, target, _ = parent
    output, budget = tmp_path / "growth", study.Budget(max_depth=4, target_fidelity=1)
    original = study.continuation.construct
    called = []

    def interrupt(
        folder: Path, goal: np.ndarray, previous: np.ndarray, config: study.topdown_mps.Config, depth: int
    ) -> dict:
        called.append(depth)
        if depth == 4:
            msg = "simulated interruption"
            raise RuntimeError(msg)
        return original(folder, goal, previous, config, depth)

    monkeypatch.setattr(study.continuation, "construct", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        study.run(output, path, target, "test", "grow", budget)
    checkpoint = output / "depth_003/construction"
    seal = study.support.digest(checkpoint / "complete.json")
    prior = study.support.load_checkpoint(checkpoint)
    assert prior is not None
    monkeypatch.setattr(study.continuation, "construct", original)
    result = study.run(output, path, target, "test", "grow", budget)
    assert called == [3, 4]
    assert study.support.digest(checkpoint / "complete.json") == seal
    assert result["completed_depths"][0]["cpu_seconds"] == prior[0]["cpu_seconds"]


def test_parent_replay_and_missing_seal(parent: tuple) -> None:
    """A source-matching parent must also have an intact and numerically valid endpoint."""
    path, target, _ = parent
    final = study.support.load_checkpoint(path / "final")
    assert final is not None
    final[0]["best_ideal_fidelity"] += 0.01
    study.support.save_checkpoint(path / "final", final[0], final[1])
    with pytest.raises(ValueError, match="replay"):
        study.authenticate(path, target, "test")
    (path / "depth_02/construction/complete.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        study.authenticate(path, target, "test")
