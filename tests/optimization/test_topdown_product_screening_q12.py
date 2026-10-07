# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Validate selected-input ideal training, reference coverage and sealed recovery."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from experiments import topdown_product_screening_q12 as study

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def target() -> np.ndarray:
    """Return a small normalized complex state for exact integration checks."""
    return np.array([1, 4j, -2, 3j]) / np.sqrt(30)


def reference_file(path: Path) -> Path:
    """Write complete audited-reference identities for a nonlaunching plan test.

    Returns:
        The reference file path.
    """
    study.support.write_json(
        path,
        {
            "runs": [
                {"target_id": t, "arm": a, "seed": s, "authenticated": True, "target_fidelity": 0.99}
                for t in study.matrix.target_ids()
                for a in study.matrix.ARMS
                for s in study.base.SEEDS
            ]
        },
    )
    return path


def test_near_zero_matches_original_at_every_depth(target: np.ndarray, tmp_path: Path) -> None:
    """The new orchestration preserves all actual optimizer iterates and seeded growth."""
    budget = study.base.Budget(online_steps=2, batch_steps=5, max_depth=2, target_fidelity=1)
    old = study.base.run_case(tmp_path / "old", target, "test", "near_zero", 300, budget)
    new = study.run_case(tmp_path / "new", target, "test", 300, budget, arm="near_zero")
    assert new["best_fidelity"] == old["best_fidelity"]
    assert new["best_depth"] == old["best_depth"]
    assert new["stop_reason"] == old["stop_reason"]
    for depth in (1, 2):
        a = study.support.load_checkpoint(tmp_path / "old" / f"depth_{depth:03d}/construction")
        b = study.support.load_checkpoint(tmp_path / "new" / f"depth_{depth:03d}/construction")
        assert a is not None
        assert b is not None
        for key in ("theta", "fidelity", "initial_theta"):
            np.testing.assert_array_equal(a[1][key], b[1][key])


def test_screened_input_and_exact_update_parity(target: np.ndarray, tmp_path: Path) -> None:
    """Selected angles feed the unchanged hybrid call and previous best extension."""
    budget = study.base.Budget(online_steps=2, batch_steps=7, max_depth=2, target_fidelity=1)
    metadata, start = study.screening.initial_parameters(target, 300, "screened_product", 31, 4)
    circuit = study.base.capacity.continuation.brickwall(2, 1, initial_single_qubit_layer=True)
    expected, trace = study.base.capacity.policies.train_noiseless(
        circuit, start["theta"], target, policy="lbfgs", steps=9, online=2, rate=0.7, stop=1
    )
    result = study.run_case(tmp_path, target, "test", 300, budget, count=31, batch_size=4)
    initial = study.support.load_checkpoint(tmp_path / "initialization")
    first = study.support.load_checkpoint(tmp_path / "depth_001/construction")
    second = study.support.load_checkpoint(tmp_path / "depth_002/construction")
    assert initial is not None
    assert first is not None
    assert second is not None
    assert initial[0]["selected_index"] == metadata["selected_index"]
    np.testing.assert_array_equal(initial[1]["theta"], start["theta"])
    np.testing.assert_array_equal(first[1]["theta"], expected)
    np.testing.assert_array_equal(first[1]["fidelity"], trace["fidelity_trace"])
    extension = study.base.capacity.continuation.legacy.extend(expected, 24, 1302, 0.001)
    np.testing.assert_array_equal(second[1]["initial_theta"], extension)
    assert result["total_cpu_seconds"] == pytest.approx(
        initial[0]["cpu_seconds"] + sum(r["stage_cpu_seconds"] for r in result["completed_depths"])
    )
    assert result["initialization_seal_sha256"] == study.support.digest(tmp_path / "initialization/complete.json")
    assert result["noise_or_pruning_performed"] is False


def test_stopping_and_budget_exhaustion_are_distinct(target: np.ndarray, tmp_path: Path) -> None:
    """The first success stops immediately; finite resource limits are not convergence."""
    success = study.run_case(
        tmp_path / "success", target, "test", 300, replace(study.DEFAULT_BUDGET, target_fidelity=0.001)
    )
    assert success["stop_reason"] == "target_reached"
    assert success["best_depth"] == 1
    assert success["completed_depths"][0]["updates"] == 0
    limited = study.run_case(
        tmp_path / "limit",
        target,
        "test",
        300,
        study.base.Budget(online_steps=0, batch_steps=1, target_fidelity=1, max_cpu_seconds=1e-12),
    )
    assert limited["stop_reason"] == "cpu_budget"
    assert limited["status"] == "budget_exhausted"
    assert limited["best_depth"] == 1


def test_completed_resume_does_not_resample_or_retrain(
    target: np.ndarray, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completed sealed stages survive resume; altered inputs and corruption fail."""
    budget = study.base.Budget(online_steps=0, batch_steps=2, max_depth=1, target_fidelity=1)
    first = study.run_case(tmp_path, target, "test", 300, budget, count=17)
    monkeypatch.setattr(study.screening, "initial_parameters", lambda *_a, **_k: pytest.fail("resampled"))
    monkeypatch.setattr(study.base.capacity.policies, "train_noiseless", lambda *_a, **_k: pytest.fail("retrained"))
    assert study.run_case(tmp_path, target, "test", 300, budget, count=17) == first
    with pytest.raises(ValueError, match="manifest differs"):
        study.run_case(tmp_path, target, "test", 300, budget, count=18)
    (tmp_path / "initialization/arrays.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        study.run_case(tmp_path, target, "test", 300, budget, count=17)


def test_incomplete_call_cannot_restart_silently(target: np.ndarray, tmp_path: Path) -> None:
    """A live snapshot lacks curvature history and cannot resume a continuous call."""
    study.support.write_json(tmp_path / "depth_001/started.json", {"incomplete": True})
    with pytest.raises(ValueError, match="Incomplete continuous"):
        study.run_case(tmp_path, target, "test", 300, study.DEFAULT_BUDGET)


def test_prepared_selection_authenticates_target_and_pool(target: np.ndarray, tmp_path: Path) -> None:
    """An initialization sealed before dispatch cannot be mixed with other inputs."""
    study.prepare_initialization(tmp_path, target, 300, "screened_product", 31, 4)
    with pytest.raises(ValueError, match="provenance differs"):
        study.prepare_initialization(tmp_path, target, 400, "screened_product", 31, 4)
    with pytest.raises(ValueError, match="provenance differs"):
        study.prepare_initialization(tmp_path, target[::-1].copy(), 300, "screened_product", 31, 4)
    with pytest.raises(ValueError, match="provenance differs"):
        study.prepare_initialization(tmp_path, target, 300, "screened_product", 32, 4)


@pytest.mark.parametrize("target_id", ["gaussian_mu0p5_sigma0p1", "tfim_critical", "haar_random_1"])
def test_twelve_qubit_selection_matches_independent_product_vectors(target_id: str) -> None:
    """All 1,001 scores agree with independent little-endian tensor products."""
    target = study.screening.load_target(12, target_id).state_vector_copy()
    record, arrays = study.screening.initial_parameters(target, 300, "screened_product")
    expected = []
    for polar, azimuth in zip(arrays["polar"], arrays["azimuth"], strict=True):
        product = np.array([1 + 0j])
        for theta, phi in zip(polar, azimuth, strict=True):
            product = np.kron([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], product)
        expected.append(abs(np.vdot(target, product)) ** 2)
    np.testing.assert_allclose(arrays["fidelity"], expected, atol=1e-13, rtol=1e-11)
    assert record["selected_index"] == int(np.argmax(expected))
    assert record["product_state_fidelity"] == pytest.approx(max(expected), abs=1e-12)


@pytest.mark.parametrize("change", [{"count": 0}, {"count": True}, {"batch_size": 0}, {"arm": "unknown"}])
def test_invalid_selection_settings(target: np.ndarray, tmp_path: Path, change: dict) -> None:
    """Malformed sample controls fail before creating training output."""
    with pytest.raises(ValueError, match="screening"):
        study.run_case(tmp_path, target, "test", 300, study.DEFAULT_BUDGET, **change)
    assert not (tmp_path / "manifest.json").exists()


def test_plan_has_only_27_new_runs_and_all_54_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Planning keeps both historical controls and never launches or trains them."""
    monkeypatch.setattr(study, "run_case", lambda *_a, **_k: pytest.fail("training during planning"))
    reference = reference_file(tmp_path / "references.json")
    plan = study.prepare_plan(tmp_path / "new", study.DEFAULT_BUDGET, reference)
    assert len(plan["jobs"]) == 27
    assert len(plan["references"]["runs"]) == 54
    assert plan["launched"] is False
    assert not (tmp_path / "new/jobs").exists()
    assert len({j["job_id"] for j in plan["jobs"]}) == 27
    assert all(j["arm"] == "screened_product" for j in plan["jobs"])
    assert plan["references"]["sha256"] == study.support.digest(reference)
    for job in plan["jobs"]:
        command = job["command"]
        assert command[command.index("--target-fidelity") + 1] == "0.99"
        assert command[command.index("--max-depth") + 1] == "30"
        assert command[command.index("--batch-steps") + 1] == "2000"
    assert study.prepare_plan(tmp_path / "new", study.DEFAULT_BUDGET, reference) == plan
    with pytest.raises(ValueError, match="matrix plan differs"):
        study.prepare_plan(tmp_path / "new", study.DEFAULT_BUDGET, reference, count=999)


@pytest.mark.parametrize("change", [{"target_fidelity": 0.999}, {"max_depth": 16}, {"batch_steps": 100}])
def test_plan_rejects_unmatched_optimizer_limits(tmp_path: Path, change: dict) -> None:
    """The declared comparison cannot quietly change the threshold or optimizer."""
    with pytest.raises(ValueError, match="matched"):
        study.prepare_plan(
            tmp_path / "new", replace(study.DEFAULT_BUDGET, **change), reference_file(tmp_path / "ref.json")
        )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unauthenticated", "wrong_threshold"])
def test_reference_coverage_is_complete_and_authenticated(tmp_path: Path, mutation: str) -> None:
    """Incomplete, repeated or unauthenticated historical rows cannot form controls."""
    path = reference_file(tmp_path / "ref.json")
    record = json.loads(path.read_text())
    if mutation == "missing":
        record["runs"].pop()
    elif mutation == "duplicate":
        record["runs"][0] = record["runs"][1]
    elif mutation == "unauthenticated":
        record["runs"][0]["authenticated"] = False
    else:
        record["runs"][0]["target_fidelity"] = 0.999
    study.support.write_json(path, record)
    with pytest.raises(ValueError, match="54 authenticated"):
        study.prepare_plan(tmp_path / "new", study.DEFAULT_BUDGET, path)


def test_cli_requires_explicit_run_and_reference_inputs(tmp_path: Path) -> None:
    """Incomplete launch requests fail; the default action is only planning."""
    with pytest.raises(SystemExit):
        study.main(["--output", str(tmp_path)])
    with pytest.raises(SystemExit):
        study.main(["--action", "run", "--output", str(tmp_path)])
    assert "experiments/topdown_product_screening_q12.py" in study.sources()
