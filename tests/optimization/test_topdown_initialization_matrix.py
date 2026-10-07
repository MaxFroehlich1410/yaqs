# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Validate complete target coverage, overlap sampling and unchanged updates."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from experiments import topdown_initialization_matrix as study

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def target() -> np.ndarray:
    """Return a normalized two-qubit complex target for inexpensive checks."""
    return np.array([1, 4j, -2, 3j]) / np.sqrt(30)


def test_fixture_coverage_and_sources() -> None:
    """Nine hardcoded targets and both old and new numerical sources are covered."""
    assert study.target_ids() == (
        "gaussian_mu0p5_sigma0p1",
        "tfim_ferro",
        "tfim_critical",
        "tfim_para",
        "haar_random_1",
        "haar_random_2",
        "haar_random_3",
        "random_mps_bond2",
        "random_mps_bond3",
    )
    assert set(study.sources()) == set(study.base.sources()) | {"experiments/topdown_initialization_matrix.py"}


@pytest.mark.parametrize(("count", "seed"), [(1, 2), (True, 2), (2.5, 2), (2, -1), (2, True), (2, 2**32)])
def test_invalid_overlap_arguments(target: np.ndarray, count: int, seed: int) -> None:
    """Reject malformed sample counts and random seeds before constructing vectors."""
    with pytest.raises(ValueError, match="samples"):
        study.overlap_samples(target, count, seed)


def test_product_overlap_matches_independent_kron(target: np.ndarray) -> None:
    """Stored angles reconstruct every little-endian sample independently."""
    before = target.copy()
    record, arrays = study.overlap_samples(target, 17, 42)
    expected = []
    for polar, azimuth in zip(arrays["polar"], arrays["azimuth"], strict=True):
        states = [
            np.array([np.cos(t / 2), np.exp(1j * p) * np.sin(t / 2)]) for t, p in zip(polar, azimuth, strict=True)
        ]
        expected.append(abs(np.vdot(target, np.kron(states[1], states[0]))) ** 2)
    np.testing.assert_allclose(arrays["fidelity"], expected, atol=1e-14)
    np.testing.assert_array_equal(target, before)
    again = study.overlap_samples(target, 17, 42)
    assert record == again[0]
    for key, value in arrays.items():
        np.testing.assert_array_equal(value, again[1][key])
    assert record["theoretical_mean"] == pytest.approx(0.25)
    assert record["training_seeds"] == [300, 400, 500]
    assert record["sample_mean"] == pytest.approx(np.mean(expected))
    assert record["fraction_above_zero"] == pytest.approx(np.mean(np.array(expected) > abs(target[0]) ** 2))


def test_haar_single_qubit_statistics_and_no_universal_gain() -> None:
    """The known first moment and an optimal zero-state counterexample are preserved."""
    record, arrays = study.overlap_samples(np.array([1, 0], dtype=complex), 5000)
    assert record["sample_mean"] == pytest.approx(0.5, abs=0.02)
    assert arrays["fidelity"].var() == pytest.approx(1 / 12, abs=0.005)
    assert record["zero_state_fidelity"] == 1
    assert record["fraction_above_zero"] == 0


def test_diagnostic_seals_resume_and_reject_changed_inputs(target: np.ndarray, tmp_path: Path) -> None:
    """Diagnostic samples are reproducible, compact and protected from input mixing."""
    result = study.diagnose(tmp_path, target, "test", 20)
    assert study.diagnose(tmp_path, target, "test", 20) == result
    saved = study.support.load_checkpoint(tmp_path)
    assert saved is not None
    assert saved[1]["fidelity"].shape == (20,)
    assert "diagnostic only" in result["scope"]
    with pytest.raises(ValueError, match="diagnostic inputs"):
        study.diagnose(tmp_path, target, "test", 21)
    (tmp_path / "arrays.npz").write_bytes(b"bad seal")
    with pytest.raises(ValueError, match="checksum"):
        study.diagnose(tmp_path, target, "test", 20)


def test_plan_all_targets_and_explicit_budgets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every fresh target has two independent arms, with no planning-side training."""
    monkeypatch.setattr(study.base, "run_case", lambda *_a, **_k: pytest.fail("training during planning"))
    plan = study.prepare_plan(tmp_path, study.base.Budget())
    assert len(plan["jobs"]) == 54
    assert len(plan["diagnostic_jobs"]) == 9
    assert plan["references"] == []
    assert len({j["job_id"] for j in plan["jobs"]}) == 54
    assert {(j["target_id"], j["arm"], j["seed"]) for j in plan["jobs"]} == {
        (t, a, s) for t in study.target_ids() for a in study.ARMS for s in study.base.SEEDS
    }
    assert not (tmp_path / "jobs").exists()
    assert plan["launched"] is False
    assert study.prepare_plan(tmp_path, study.base.Budget()) == plan
    assert plan["overlap_samples"] == 1000
    for job in plan["jobs"]:
        command = job["command"]
        assert command[command.index("--batch-steps") + 1] == "2000"
        assert command[command.index("--target-id") + 1] == job["target_id"]
    with pytest.raises(ValueError, match="matrix plan differs"):
        study.prepare_plan(tmp_path, replace(study.base.Budget(), batch_steps=1999))
    with pytest.raises(ValueError, match="samples"):
        study.prepare_plan(tmp_path, study.base.Budget(), count=1)


def test_plan_omits_only_authenticated_reference_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exactly the six paired random-MPS runs are reused, yielding 48 new jobs."""
    references = [
        {"target_id": study.base.TARGET_ID, "arm": arm, "seed": seed} for seed in study.base.SEEDS for arm in study.ARMS
    ]
    calls = []
    monkeypatch.setattr(study, "authenticate_reference", lambda *args: calls.append(args) or references)
    plan = study.prepare_plan(tmp_path / "matrix", study.base.Budget(), tmp_path / "old")
    assert len(plan["jobs"]) == 48
    assert len(plan["references"]) == 6
    assert all(j["target_id"] != study.base.TARGET_ID for j in plan["jobs"])
    assert calls[0][0] == tmp_path / "old"
    assert calls[0][1].shape == (4096,)


def test_wrapper_preserves_every_training_parameter(target: np.ndarray, tmp_path: Path) -> None:
    """The new matrix wrapper exactly matches its already-tested numerical driver."""
    budget = study.base.Budget(online_steps=2, batch_steps=5, max_depth=2, target_fidelity=1)
    reference = study.base.run_case(tmp_path / "old", target, "test", "random_product", 300, budget)
    result = study.run(tmp_path / "new", target, "test", "random_product", 300, budget)
    assert result["best_fidelity"] == reference["best_fidelity"]
    for depth in (1, 2):
        a = study.support.load_checkpoint(tmp_path / "old" / f"depth_{depth:03d}/construction")
        b = study.support.load_checkpoint(tmp_path / "new" / f"depth_{depth:03d}/construction")
        assert a is not None
        assert b is not None
        np.testing.assert_array_equal(a[1]["theta"], b[1]["theta"])
        np.testing.assert_array_equal(a[1]["fidelity"], b[1]["fidelity"])
    assert (tmp_path / "new/matrix-complete.json").exists()
    assert study.run(tmp_path / "new", target, "test", "random_product", 300, budget) == result
    with pytest.raises(ValueError, match="worker provenance"):
        study.run(tmp_path / "new", target, "test", "random_product", 300, replace(budget, batch_steps=6))
    with pytest.raises(ValueError, match="Unknown matrix"):
        study.run(tmp_path / "invalid", target, "test", "largest_basis", 300, budget)


def test_reference_authentication_checks_exits_and_preserves_files(tmp_path: Path) -> None:
    """Original complete runs remain read-only and require authenticated receipts."""
    target = np.array([1, 0, 0, 0], dtype=complex)
    budget = study.base.Budget(online_steps=0, batch_steps=2, max_depth=1, target_fidelity=0.0001)
    for seed in study.base.SEEDS:
        for arm in study.ARMS:
            folder = tmp_path / "jobs" / f"{arm}__seed{seed}"
            study.base.run_case(folder, target, study.base.TARGET_ID, arm, seed, budget)
            command = ["python", "worker", arm, str(seed)]
            study.support.write_json(
                tmp_path / "logs" / f"{folder.name}.process.json",
                {
                    "status": "completed",
                    "exit_code": 0,
                    "command": command,
                    "wall_seconds": 1.0,
                },
            )
            study.support.write_json(
                tmp_path / "claims" / folder.name / "launch.json",
                {
                    "command": ["supervisor", "--", *command],
                },
            )
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    references = study.authenticate_reference(tmp_path, target, budget)
    assert len(references) == 6
    assert {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    with pytest.raises(ValueError, match="reference differs"):
        study.authenticate_reference(tmp_path, target, replace(budget, batch_steps=3))
    path = tmp_path / "logs/near_zero__seed300.process.json"
    receipt = json.loads(path.read_text())
    receipt["exit_code"] = 1
    study.support.write_json(path, receipt)
    with pytest.raises(ValueError, match="reference differs"):
        study.authenticate_reference(tmp_path, target, budget)


def test_cli_requires_explicit_target_and_runs_requested_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI dispatch carries target, seed and budgets without silently launching a matrix."""
    with pytest.raises(SystemExit):
        study.main(["--action", "run", "--output", str(tmp_path)])
    calls = []
    monkeypatch.setattr(study, "run", lambda *args: calls.append(args) or {})
    study.main([
        "--action",
        "run",
        "--output",
        str(tmp_path),
        "--target-id",
        "tfim_para",
        "--arm",
        "near_zero",
        "--seed",
        "400",
    ])
    assert calls[0][1].shape == (4096,)
    assert calls[0][2:5] == ("tfim_para", "near_zero", 400)
    assert calls[0][5] == study.base.Budget()
    calls.clear()
    monkeypatch.setattr(study, "diagnose", lambda *args: calls.append(args) or {})
    study.main([
        "--action",
        "overlaps",
        "--output",
        str(tmp_path),
        "--target-id",
        "haar_random_2",
        "--overlap-samples",
        "31",
    ])
    assert calls[0][2:] == ("haar_random_2", 31)
