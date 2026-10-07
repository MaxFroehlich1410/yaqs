# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check refinement parity, sealed parent selection, accounting and safe recovery."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from experiments import topdown_haar_refinement as study

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Construct two small sealed growth stages with the production schema.

    Returns:
        Growth directory and its complex normalized target.
    """
    monkeypatch.setattr(study.capacity, "sources", dict)
    target = np.array([1, 2j, -3, 4j]) / np.sqrt(30)
    config = study.capacity.topdown_mps.Config(
        backend="statevector", qubits=2, construction_steps=4, construction_online=2
    )
    root, rows, previous = tmp_path / "parent", [], np.empty(0)
    for depth in (2, 3):
        folder = root / f"depth_{depth:03d}/construction"
        record = study.capacity.continuation.construct(folder, target, previous, config, depth)
        saved = study.support.load_checkpoint(folder)
        assert saved is not None
        previous = saved[1]["theta"]
        rows.append({
            "depth": depth,
            "fidelity": record["fidelity"],
            "checkpoint": str(folder.relative_to(root)),
            "cpu_seconds": float(depth),
        })
    study.support.write_json(
        root / "manifest.json",
        {
            "version": study.capacity.VERSION,
            "arm": "grow",
            "construction_config": asdict(config),
            "target_id": "test",
            "target_sha256": study.capacity.topdown_mps.target_digest(target),
            "sources": {},
            "parent": {"historical_construction_cpu_seconds": 10.0, "historical_continuation_cpu_seconds": 20.0},
        },
    )
    study.support.save_checkpoint(root / "final", {"completed_depths": rows}, {})
    return root, target


def test_selects_requested_depth_and_only_its_historical_cost(parent: tuple) -> None:
    """Later growth cannot leak parameters or CPU costs into an earlier refinement."""
    root, target = parent
    record, arrays, refs = study.authenticate(root, target, "test", 2)
    selected = study.support.load_checkpoint(root / "depth_002/construction")
    assert selected is not None
    assert record["depth"] == 2
    np.testing.assert_array_equal(arrays["theta"], selected[1]["theta"])
    assert refs["historical_cpu_seconds"] == pytest.approx(32.0)
    assert refs["growth_to_selected_cpu_seconds"] == pytest.approx(2.0)
    assert study.authenticate(root, target, "test", 3)[2]["historical_cpu_seconds"] == pytest.approx(35.0)


def test_continuous_refinement_parity_and_parent_immutability(parent: tuple, tmp_path: Path) -> None:
    """Instrumentation and checkpoint reuse preserve every optimizer step exactly."""
    root, target = parent
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    record, arrays, _ = study.authenticate(root, target, "test", 2)
    circuit = study.support.restore_circuit(record["circuit"])
    expected, trace = study.capacity.policies.train_noiseless(
        circuit, arrays["theta"], target, policy="lbfgs", steps=6, online=0, stop=1
    )
    output = tmp_path / "refined"
    result = study.run(output, root, target, "test", 2, study.capacity.Budget(fixed_steps=6, target_fidelity=1))
    saved = study.support.load_checkpoint(output / "training")
    assert saved is not None
    np.testing.assert_array_equal(saved[1]["theta"], expected)
    np.testing.assert_array_equal(saved[1]["fidelity"], trace["fidelity_trace"])
    updates = json.loads(saved[1]["optimizer_records_utf8"].tobytes())
    for actual, reference in zip(updates, trace["records"], strict=True):
        for key in ("fidelity", "history_pairs", "trial_rates", "accepted", "curvature_dot"):
            assert actual[key] == reference[key]
    assert saved[0]["training"]["online_sweeps"] == 0
    assert result["total_training_cpu_seconds"] == pytest.approx(32.0 + result["new_cpu_seconds"])
    assert result["statevector_replay_error"] < 1e-12
    assert result["gradient_norm"] >= 0
    assert result["status"] == "budget_exhausted"
    assert {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_threshold_crossing_is_exact_and_includes_initial_state() -> None:
    """Rounded displays and later crossings cannot shift the first success time."""
    records = [
        {"iteration": i, "fidelity": f, "cumulative_cpu_seconds": i * 2, "cumulative_wall_seconds": i * 3}
        for i, f in enumerate((0.9899999999, 0.99001, 0.995, 0.9991), 1)
    ]
    assert study.threshold_crossing(0.98, records, 0.99) == {
        "update": 2,
        "fidelity": 0.99001,
        "cpu_seconds": 4,
        "wall_seconds": 6,
    }
    hit = study.threshold_crossing(0.98, records, 0.999)
    initial = study.threshold_crossing(0.995, records, 0.99)
    assert hit is not None
    assert initial is not None
    assert hit["update"] == 4
    assert initial["update"] == 0
    assert study.threshold_crossing(0.98, records, 1) is None


def test_first_target_hit_stops_training(parent: tuple, tmp_path: Path) -> None:
    """Threshold success terminates the continuous call at the first accepted hit."""
    root, target = parent
    record, arrays, _ = study.authenticate(root, target, "test", 2)
    circuit = study.support.restore_circuit(record["circuit"])
    _, trace = study.capacity.policies.train_noiseless(
        circuit, arrays["theta"], target, policy="lbfgs", steps=2, online=0
    )
    threshold = (record["fidelity"] + trace["fidelity_trace"][0]) / 2
    result = study.run(tmp_path / "hit", root, target, "test", 2, study.capacity.Budget(target_fidelity=threshold))
    assert result["completed_updates"] == 1
    assert result["status"] == "target_reached"
    assert result["threshold_crossings"][str(threshold)]["update"] == 1


def test_resume_authenticates_results_without_retraining(
    parent: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only completed training is reusable; altered budgets or arrays are rejected."""
    root, target = parent
    output, budget = tmp_path / "run", study.capacity.Budget(fixed_steps=2, target_fidelity=1)
    result = study.run(output, root, target, "test", 2, budget)
    monkeypatch.setattr(study.capacity.policies, "train_noiseless", lambda *_a, **_k: pytest.fail("retrained"))
    assert study.run(output, root, target, "test", 2, budget) == result
    with pytest.raises(ValueError, match="manifest"):
        study.run(output, root, target, "test", 2, replace(budget, fixed_steps=3))
    (output / "training/arrays.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        study.run(output, root, target, "test", 2, budget)


def test_interrupted_call_does_not_restart_silently(
    parent: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live parameter snapshots must not masquerade as resumable L-BFGS history."""
    root, target = parent
    output, budget = tmp_path / "interrupted", study.capacity.Budget(fixed_steps=3)

    def interrupt(*_args: object, **_kwargs: object) -> None:
        msg = "interrupted"
        raise RuntimeError(msg)

    monkeypatch.setattr(study.capacity, "fixed", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        study.run(output, root, target, "test", 2, budget)
    with pytest.raises(ValueError, match="Incomplete continuous call"):
        study.run(output, root, target, "test", 2, budget)


@pytest.mark.parametrize(("field", "value"), [("arm", "fixed"), ("sources", {"bad": "hash"}), ("target_id", "wrong")])
def test_parent_provenance_rejection(parent: tuple, field: str, value: object) -> None:
    """Unmatched targets and numerical implementations cannot be silently reused."""
    root, target = parent
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    study.support.write_json(path, manifest)
    with pytest.raises(ValueError, match="provenance"):
        study.authenticate(root, target, "test", 2)


def test_missing_corrupt_or_mismatched_checkpoints(parent: tuple) -> None:
    """Require sealed stages, exact topology, and numerical replay, not hashes alone."""
    root, target = parent
    with pytest.raises(ValueError, match="separate"):
        study.run(root / "nested", root, target, "test", 2, study.capacity.Budget())
    with pytest.raises(ValueError, match="sealed"):
        study.authenticate(root, target, "test", 4)
    folder = root / "depth_002/construction"
    saved = study.support.load_checkpoint(folder)
    assert saved is not None
    record, arrays = saved
    record["fidelity"] += 0.01
    study.support.save_checkpoint(folder, record, arrays)
    with pytest.raises(ValueError, match="replay"):
        study.authenticate(root, target, "test", 2)


def test_cli_freezes_two_depths_and_agreed_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The launch interface fixes the target, backend, thresholds and update count."""
    calls = []
    monkeypatch.setattr(study, "run", lambda *args: calls.append(args) or {})
    for depth in (66, 87):
        study.main(["--parent", str(tmp_path), "--output", str(tmp_path / "new"), "--depth", str(depth)])
    assert [call[4] for call in calls] == [66, 87]
    assert calls[0][2].shape == (4096,)
    assert calls[0][3] == "haar_random_1"
    assert calls[0][5] == study.capacity.Budget(fixed_steps=2000, target_fidelity=0.999, progress_every=50)
    with pytest.raises(SystemExit):
        study.main(["--parent", str(tmp_path), "--output", str(tmp_path / "new"), "--depth", "128"])
