# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Ideal stopping, original optimizer parity, provenance and interruption recovery."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH
from experiments import topdown_noiseless_continuation as runner

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Build a small authentic ideal parent using the unchanged optimizer.

    Returns:
        Parent path, normalized target, and original construction settings.
    """
    monkeypatch.setattr(runner.topdown_mps, "source_hashes", dict)
    config = runner.topdown_mps.Config(qubits=2, max_depth=1, construction_steps=5, construction_online=2)
    target = np.array([1, 2j, 3, 4j]) / np.sqrt(30)
    folder = tmp_path / "parent"
    manifest = {
        "config": asdict(config),
        "target_id": "test",
        "sources": {},
        "target_sha256": runner.topdown_mps.target_digest(target),
        "fixture_sha256": runner.support.digest(DEFAULT_TARGET_PATH),
    }
    runner.support.write_json(folder / "manifest.json", manifest)
    runner.construct(folder / "depth_01/construction", target, np.empty(0), config, 1)
    return folder, target, config


@pytest.mark.parametrize("score", [np.nan, np.inf, -0.01, 1.01])
def test_nonphysical_score_rejected(score: float) -> None:
    """Invalid scores cannot reset patience or replace the best checkpoint."""
    tracker = runner.IdealPatience(runner.GrowthRule(), 0.2, 0.2, 4)
    with pytest.raises(ValueError, match="Nonphysical"):
        tracker.observe(5, score)


def test_relative_progress_and_high_fidelity_infidelity() -> None:
    """Tiny overlaps can improve materially; near one, infidelity sets the scale."""
    tracker = runner.IdealPatience(runner.GrowthRule(), 5e-8, 5e-8, 4)
    decision = tracker.observe(5, 5.1e-8)
    assert decision["material_improvement"]
    assert decision["required_gain"] == pytest.approx(5e-10)
    tracker = runner.IdealPatience(runner.GrowthRule(), 0.999, 0.999, 4)
    assert tracker.observe(5, 0.9991)["material_improvement"]
    tracker = runner.IdealPatience(runner.GrowthRule(), 0, 0, 4)
    assert not tracker.observe(5, 1e-13)["material_improvement"]
    assert tracker.observe(6, 2e-12)["material_improvement"]


def test_accumulated_gain_best_checkpoint_and_three_stale_steps() -> None:
    """Small improvements accumulate, ties keep earlier winners, decreases count."""
    tracker = runner.IdealPatience(runner.GrowthRule(), 0.1, 0.1, 4)
    assert not tracker.observe(5, 0.1006)["material_improvement"]
    assert tracker.best_depth == 5
    assert tracker.observe(6, 0.1012)["material_improvement"]
    for depth, value in ((7, 0.1012), (8, 0.09), (9, 0.1)):
        tracker.observe(depth, value)
    assert tracker.stale == 3
    assert tracker.best_depth == 6


@pytest.mark.parametrize(
    "change",
    [
        {"max_depth": 4},
        {"max_depth": True},
        {"patience": 0},
        {"relative_gain": 0},
        {"relative_gain": float("nan")},
        {"absolute_gain": -1},
        {"max_cpu_seconds": float("inf")},
    ],
)
def test_invalid_rule(change: dict) -> None:
    """Do not silently reinterpret malformed budgets as convergence."""
    with pytest.raises(ValueError, match="budget"):
        replace(runner.GrowthRule(), **change).validate(4)


@pytest.mark.parametrize("backend", ["mps", "statevector"])
def test_continuation_matches_dense_optimizer_and_keeps_binary_history(
    parent: tuple, tmp_path: Path, backend: str
) -> None:
    """New depths use identical seeds, warm parameters, budgets and best-iterate logic."""
    folder, target, config = parent
    saved = runner.support.load_checkpoint(folder / "depth_01/construction")
    assert saved is not None
    circuit = runner.brickwall(2, 2, initial_single_qubit_layer=True)
    initial = runner.legacy.extend(saved[1]["theta"], circuit.num_params, config.seed + 1002, 0.001)
    expected, trace = runner.policies.train_noiseless(
        circuit,
        initial,
        target,
        policy="lbfgs",
        steps=config.construction_steps,
        online=config.construction_online,
        engine=runner.StatevectorBackend(circuit, target),
    )
    output = tmp_path / "continued"
    result = runner.run_case(output, folder, target, "test", runner.GrowthRule(max_depth=2), 1, backend=backend)
    actual = runner.support.load_checkpoint(output / "depth_02/construction")
    assert actual is not None
    np.testing.assert_allclose(actual[1]["theta"], expected, atol=1e-10)
    np.testing.assert_array_equal(actual[1]["initial"], initial)
    assert actual[0]["statevector_replay_error"] < 1e-12
    assert actual[0]["backend"] == backend
    assert ("statevector_mps_error" in actual[0]) == (backend == "mps")
    np.testing.assert_allclose(actual[1]["fidelity"], trace["fidelity_trace"], atol=1e-12)
    assert actual[0]["training"]["best_theta"] == {"file": "arrays.npz", "key": "best_theta"}
    assert actual[1]["fidelity"].shape == (5,)
    assert result["status"] == "budget_exhausted"
    assert result["stop_reason"] == "depth_cap"
    assert not result["noise_or_pruning_performed"]


def test_resume_skips_training_and_authenticates_every_checkpoint(
    parent: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finished resumes are exact and refuse changed settings or corrupted arrays."""
    folder, target, _ = parent
    output, rule = tmp_path / "continued", runner.GrowthRule(max_depth=2)
    expected = runner.run_case(output, folder, target, "test", rule, 1)
    monkeypatch.setattr(runner, "construct", lambda *_: pytest.fail("retrained a sealed checkpoint"))
    assert runner.run_case(output, folder, target, "test", rule, 1) == expected
    with pytest.raises(ValueError, match="manifest"):
        runner.run_case(output, folder, target, "test", replace(rule, patience=4), 1)
    (output / "depth_02/construction/arrays.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        runner.run_case(output, folder, target, "test", rule, 1)


def test_interruption_preserves_completed_depth(parent: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupted extension restarts from its prior best saved parameters."""
    folder, target, _ = parent
    original = runner.construct

    def interrupt(
        path: Path, target: np.ndarray, previous: np.ndarray, config: runner.topdown_mps.Config, depth: int
    ) -> dict:
        if depth == 3:
            msg = "interrupted"
            raise RuntimeError(msg)
        return original(path, target, previous, config, depth)

    output, rule = tmp_path / "continued", runner.GrowthRule(max_depth=3)
    monkeypatch.setattr(runner, "construct", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.run_case(output, folder, target, "test", rule, 1)
    seal = runner.support.digest(output / "depth_02/construction/complete.json")
    monkeypatch.setattr(runner, "construct", original)
    result = runner.run_case(output, folder, target, "test", rule, 1)
    assert len(result["decisions"]) == 2
    assert runner.support.digest(output / "depth_02/construction/complete.json") == seal


def test_cpu_budget_and_ideal_patience_are_separate(
    parent: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No pruning or noisy calls occur; compute exhaustion is not convergence."""
    folder, target, _ = parent
    monkeypatch.setattr(runner.legacy, "train", lambda *_a, **_k: pytest.fail("noisy training called"))
    monkeypatch.setattr(runner.support, "prune_b", lambda *_a: pytest.fail("pruning called"))
    result = runner.run_case(tmp_path / "budget", folder, target, "test", runner.GrowthRule(max_cpu_seconds=1e-12), 1)
    assert result["stop_reason"] == "cpu_budget"
    assert len(result["decisions"]) == 1
    result = runner.run_case(tmp_path / "stale", folder, target, "test", runner.GrowthRule(relative_gain=0.99), 1)
    assert result["stop_reason"] == "ideal_patience"
    assert result["status"] == "stagnated"
    assert result["decisions"][-1]["stale"] == 3


def test_parent_integrity_and_output_separation(parent: tuple, tmp_path: Path) -> None:
    """Reject altered parent scores, different targets and writes under the original."""
    folder, target, _ = parent
    with pytest.raises(ValueError, match="separate"):
        runner.run_case(folder / "new", folder, target, "test", runner.GrowthRule(), 1)
    with pytest.raises(ValueError, match="Parent target"):
        runner.run_case(tmp_path / "new", folder, target[::-1].copy(), "test", runner.GrowthRule(), 1)
    with pytest.raises(ValueError, match="Missing"):
        runner.parent_checkpoints(folder, target, "test", 2)
    saved = runner.support.load_checkpoint(folder / "depth_01/construction")
    assert saved is not None
    record, arrays = saved
    record["training"]["best_fidelity"] += 0.1
    runner.support.save_checkpoint(folder / "depth_01/construction", record, arrays)
    with pytest.raises(ValueError, match="replay"):
        runner.parent_checkpoints(folder, target, "test", 1)


@pytest.mark.parametrize("backend", ["mps", "statevector"])
def test_cli_excludes_tfim_and_dispatches_selected_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """The public driver cannot accidentally launch TFIM or noisy competitors."""
    with pytest.raises(SystemExit):
        runner.main([
            "--parent-campaign",
            str(tmp_path),
            "--output",
            str(tmp_path / "new"),
            "--target",
            "tfim_critical",
        ])
    called = []
    monkeypatch.setattr(runner, "run_case", lambda *args, **kwargs: called.append((args, kwargs)) or {"status": "test"})
    argv = ["--parent-campaign", str(tmp_path), "--output", str(tmp_path / "new"), "--target", runner.TARGETS[0]]
    if backend != "mps":
        argv.extend(["--backend", backend])
    runner.main(argv)
    assert len(called) == 1
    args, kwargs = called[0]
    assert args[2].shape == (4096,)
    assert args[3] == runner.TARGETS[0]
    assert args[4] == runner.GrowthRule()
    assert kwargs == {"backend": backend}
    assert runner.sources()["experiments/topdown_noiseless_continuation.py"]


def test_final_result_tampering_is_rejected(parent: tuple, tmp_path: Path) -> None:
    """A resealed but inconsistent final summary cannot silently replace history."""
    folder, target, _ = parent
    output, rule = tmp_path / "continued", runner.GrowthRule(max_depth=2)
    runner.run_case(output, folder, target, "test", rule, 1)
    saved = runner.support.load_checkpoint(output / "final")
    assert saved is not None
    record = saved[0]
    record["best_depth"] = 99
    runner.support.save_checkpoint(output / "final", record, {})
    with pytest.raises(ValueError, match="final result"):
        runner.run_case(output, folder, target, "test", rule, 1)
    assert json.loads((output / "manifest.json").read_text())["rule"]["max_depth"] == 2


def test_statevector_chain_never_uses_mps_and_preserves_parent(
    parent: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train all new depths in vectors, carry their own best angles and seal the backend."""
    folder, target, config = parent
    original = {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    monkeypatch.setattr(runner.MPS, "from_statevector", lambda *_: pytest.fail("statevector run entered MPS"))
    output, rule = tmp_path / "vectors", runner.GrowthRule(max_depth=3)
    result = runner.run_case(output, folder, target, "test", rule, 1, backend="statevector")
    assert result["execution_backend"] == "statevector"
    second = runner.support.load_checkpoint(output / "depth_02/construction")
    third = runner.support.load_checkpoint(output / "depth_03/construction")
    assert second is not None
    assert third is not None
    np.testing.assert_array_equal(third[1]["initial"][: len(second[1]["theta"])], second[1]["theta"])
    expected = runner.legacy.extend(second[1]["theta"], len(third[1]["theta"]), config.seed + 1003, 0.001)
    np.testing.assert_array_equal(third[1]["initial"], expected)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["parent_backend"] == "mps"
    assert manifest["execution_backend"] == "statevector"
    assert manifest["construction_config"] == asdict(replace(config, backend="statevector"))
    assert runner.run_case(output, folder, target, "test", rule, 1, backend="statevector") == result
    with pytest.raises(ValueError, match="manifest"):
        runner.run_case(output, folder, target, "test", rule, 1, backend="mps")
    assert original == {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def test_bad_backend_is_rejected_before_output(tmp_path: Path) -> None:
    """An unknown backend cannot create an ambiguously labelled experiment."""
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="backend"):
        runner.run_case(output, tmp_path / "parent", np.ones(4) / 2, "test", runner.GrowthRule(), backend="unknown")
    assert not output.exists()
