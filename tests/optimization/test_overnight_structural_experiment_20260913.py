# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Branch resets, full-call fairness, immutable prepare and exact resume tests."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import structural_experiment as experiment
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def small_case() -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Return a compressible one-qubit preparation with physical noise windows."""
    circuit = ParameterizedCircuit(
        1,
        [
            ParameterizedGate("ry", (0,), 0, logical_gate_id=0),
            ParameterizedGate("ry", (0,), 1, logical_gate_id=1),
            ParameterizedGate("rz", (0,), 2, logical_gate_id=2),
        ],
    )
    return circuit, np.array([0.1, 0.2, 0.3]), np.array([1, 1j]) / np.sqrt(2)


def test_historical_reference_excludes_checkpoint_crossing_budget() -> None:
    """A better later historical endpoint cannot enter a fixed-budget reference."""
    circuit, theta, _ = small_case()
    state = {"circuit": pilot.circuit_record(circuit), "theta": theta.tolist()}
    later = {"circuit": state["circuit"], "theta": (theta + 0.3).tolist()}
    refined = {
        "state": later,
        "fidelity": 0.95,
        "curve": [
            {"state": state, "fidelity": 0.8, "cpu_seconds": 9.0},
            {"state": later, "fidelity": 0.95, "cpu_seconds": 10.01},
        ],
    }
    selected = experiment.historical_endpoint(refined, 10.0)
    assert selected["theta"] == state["theta"]
    assert selected["archived_local_fidelity"] == pytest.approx(0.8)
    assert selected["historical_checkpoint_cpu_seconds"] <= selected["historical_cpu_budget_seconds"]
    with pytest.raises(ValueError, match="precedes the first"):
        experiment.historical_endpoint(refined, 8.0)


def small_config() -> experiment.Config:
    """Return bounded toy budgets retaining the same paired protocol semantics."""
    return experiment.Config(
        full_evaluations=45,
        warmup_evaluations=4,
        max_rounds=2,
        shortlist=1,
        repair_evaluations=4,
        protected_initial_gates=0,
        checkpoint_every=2,
    )


def test_pair_charges_every_branch_and_distinguishes_gradient_and_score_work(tmp_path: Path) -> None:
    """Search comparisons include rejected alternatives and actual scoring passes."""
    circuit, theta, target = small_case()
    config = small_config()
    result = experiment.run_pair(circuit, theta, target, config, tmp_path)
    structural, fixed = result["structural_lbfgs"], result["fixed_lbfgs"]
    assert structural["total_channel_calls"] <= config.full_evaluations
    assert fixed["total_channel_calls"] <= config.full_evaluations
    assert (
        structural["total_channel_calls"]
        == structural["joint_gradient_calls"] + structural["forward_score_calls"] + structural["deletion_ranking_calls"]
    )
    assert (
        structural["density_direction_passes"]
        == 2 * (structural["joint_gradient_calls"] + structural["deletion_ranking_calls"])
        + structural["forward_score_calls"]
    )
    assert fixed["density_direction_passes"] == 2 * fixed["total_channel_calls"]
    assert result["physical_training_cpu_seconds"] == pytest.approx(
        structural["cpu_seconds"] + fixed["cpu_seconds"] - result["shared_warmup_cpu_seconds"]
    )
    assert result["common_cpu_seconds"] == min(structural["cpu_seconds"], fixed["cpu_seconds"])
    for method, selected in result["common_cpu_last_available"].items():
        if selected is not None:
            possible = [p for p in result[method]["curve"] if p["cpu_seconds"] <= result["common_cpu_seconds"]]
            assert selected == possible[-1]
    records = [
        cast("dict[str, Any]", pilot.read_record(path))
        for path in (tmp_path / "branches").glob("repair_*.json")
        if ".partial." not in path.name and ".attempts." not in path.name
    ]
    assert records
    assert all(row["memory_rule"] == "fresh_curvature" for row in records)
    assert all(row["curve"][0]["iterations"] == 0 for row in records)
    assert pilot.read_record(tmp_path / "branches/fixed.json")["memory_rule"] == "continue_shared_warmup_curvature"
    assert pilot.read_record(tmp_path / "branches/polish.json")["memory_rule"] == "fresh_curvature"


def test_complete_pair_resume_preserves_every_artifact_and_numerical_endpoint(tmp_path: Path) -> None:
    """Completed warmup, repairs, search and polish are read-only on resume."""
    circuit, theta, target = small_case()
    config = small_config()
    first = experiment.run_pair(circuit, theta, target, config, tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    second = experiment.run_pair(circuit, theta, target, config, tmp_path)
    assert first == second
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_branch_partial_resume_keeps_full_lbfgs_curvature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulated accepted-boundary interruption reproduces uninterrupted parameters."""
    circuit, theta, target = small_case()
    config = small_config()
    full = experiment.BranchStore(tmp_path / "full", target, config).optimize("repair", circuit, theta, 25)
    original_write = pilot.write_record

    def interrupted_write(path: Path, record: dict[str, Any]) -> None:
        original_write(path, record)
        if path.name.endswith("partial.json") and record["optimizer"]["iterations"] == 2:
            raise KeyboardInterrupt

    store = experiment.BranchStore(tmp_path / "resume", target, config)
    monkeypatch.setattr(pilot, "write_record", interrupted_write)
    with pytest.raises(KeyboardInterrupt):
        store.optimize("repair", circuit, theta, 25)
    monkeypatch.setattr(pilot, "write_record", original_write)
    resumed = store.optimize("repair", circuit, theta, 25)
    assert resumed["optimizer"] == full["optimizer"]
    assert resumed["curve"][-1]["theta"] == full["curve"][-1]["theta"]
    assert resumed["curve"][2]["iterations"] == 2


def test_branch_custody_rejects_changed_parameters_and_cap(tmp_path: Path) -> None:
    """A retained branch cannot be silently reused for a different problem."""
    circuit, theta, target = small_case()
    store = experiment.BranchStore(tmp_path, target, small_config())
    store.optimize("repair", circuit, theta, 10)
    with pytest.raises(ValueError, match="signature changed"):
        store.optimize("repair", circuit, theta + 0.01, 10)
    with pytest.raises(ValueError, match="signature changed"):
        store.optimize("repair", circuit, theta, 11)


def test_prepare_only_freezes_authentication_without_any_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Protocol preparation fixes source vectors/budgets and never simulates."""
    source = tmp_path / "sources"
    source.mkdir()
    circuit = ParameterizedCircuit(6, [ParameterizedGate("ry", (i % 6,), i, logical_gate_id=i) for i in range(108)])
    starts = [
        {
            "target_id": target,
            "replica": 11,
            "candidate": {"circuit": pilot.circuit_record(circuit), "theta": [0.1] * 108},
        }
        for target in experiment.DEVELOPMENT_TARGETS
    ]
    pilot.write_record(source / "starts.json", {"starts": starts})
    historical = tmp_path / "output/statevector-adjoint-restructuring-20260912"
    historical.mkdir(parents=True)
    pilot.write_record(historical / "result.json", {"jobs": []})
    monkeypatch.setattr(experiment.runner, "OUTPUT", source)
    monkeypatch.setattr(experiment.runner, "REPO", tmp_path)
    monkeypatch.setattr(experiment.runner, "custody", lambda: {"benchmark_hashes": {"definition": "frozen"}})
    monkeypatch.setattr(experiment, "numerical_sources", lambda: {"driver": "fixed"})
    monkeypatch.setattr(experiment, "run_pair", lambda *_a, **_k: pytest.fail("prepare must not train"))
    output = tmp_path / "prepared"
    assert experiment.main(["--output", str(output), "--prepare-only"]) == 0
    encoded = (output / "PROTOCOL.json").read_bytes()
    protocol = pilot.read_record(output / "PROTOCOL.json")
    assert protocol["config"] == asdict(experiment.Config())
    experiment.verify_protocol(protocol)
    assert experiment.main(["--output", str(output), "--prepare-only"]) == 0
    assert (output / "PROTOCOL.json").read_bytes() == encoded
    with pytest.raises(ValueError, match="Frozen structural protocol differs"):
        experiment.freeze_protocol(output, config=experiment.Config(repair_evaluations=50))
    changed = copy.deepcopy(protocol)
    changed["source_hashes"] = {"driver": "different"}
    with pytest.raises(ValueError, match="identity changed"):
        experiment.verify_protocol(changed)


@pytest.mark.parametrize(
    "values",
    [{"full_evaluations": 10}, {"repair_evaluations": 0}, {"max_rounds": True}, {"protected_initial_gates": -1}],
)
def test_invalid_protocol_budgets(values: dict[str, Any]) -> None:
    """Insufficient or malformed caps are rejected before numerical work."""
    with pytest.raises(ValueError, match=r"budget|configuration"):
        experiment.Config(**values)


def test_scoreboard_order_survives_json_key_sorting_and_retains_failures(tmp_path: Path) -> None:
    """Cold/resumed summaries use identical row order and never omit failed cells."""
    arm = {
        "comparison": "matched",
        "fidelities": dict.fromkeys(experiment.channels.NOISE_IDS, 0.75),
        "logical_counts": {"one_qubit": 1, "two_qubit": 0, "depth": 1},
        "ballarin_counts": {"one_qubit": 1, "two_qubit": 0, "depth": 1},
        "theta": [0.3],
        "evaluation_cpu_seconds": 0.01,
    }
    records = [
        {
            "target_id": "good",
            "replica": 11,
            "status": "complete",
            "arms": {"structural_lbfgs": arm, "fixed_lbfgs": arm},
        },
        {"target_id": "bad", "replica": 11, "status": "failed", "failure": "Numerical error", "arms": {}},
    ]
    experiment.write_scoreboard(tmp_path, records)
    before = (tmp_path / "SCOREBOARD.csv").read_bytes()
    experiment.write_scoreboard(tmp_path, json.loads(json.dumps(records, sort_keys=True)))
    assert (tmp_path / "SCOREBOARD.csv").read_bytes() == before
    assert before.decode().count("Numerical error") == 24


def test_failed_case_is_saved_and_never_silently_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Failed runs remain authenticated records on repeated execution."""
    monkeypatch.setattr(experiment, "verify_protocol", lambda _: None)

    def fail(_job: dict[str, Any]) -> dict[str, Any]:
        msg = "Forced optimizer failure."
        raise FloatingPointError(msg)

    monkeypatch.setattr(experiment, "_execute_case", fail)
    job = {"protocol": {"version": "test"}, "source": {"target_id": "toy", "replica": 11}, "root": str(tmp_path)}
    first = experiment.execute_case(job)
    assert first["status"] == "failed"
    encoded = (tmp_path / "FAILED.json").read_bytes()
    assert experiment.execute_case(job) == first
    assert (tmp_path / "FAILED.json").read_bytes() == encoded


def test_unsaved_interrupted_calls_are_visible_in_physical_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay from a durable checkpoint records the discarded pre-interruption work."""
    circuit, theta, target = small_case()
    store = experiment.BranchStore(tmp_path, target, small_config())
    original = experiment.density_loss_gradient
    count = 0

    def interrupted(
        candidate: ParameterizedCircuit,
        parameters: np.ndarray,
        vector: np.ndarray,
        calibration: experiment.CalibratedLocalDepolarizingProvider,
    ) -> tuple[np.ndarray, float, float]:
        nonlocal count
        value = original(candidate, parameters, vector, calibration)
        count += 1
        if count == 3:
            raise KeyboardInterrupt
        return value

    monkeypatch.setattr(experiment, "density_loss_gradient", interrupted)
    with pytest.raises(KeyboardInterrupt):
        store.optimize("repair", circuit, theta, 25)
    monkeypatch.setattr(experiment, "density_loss_gradient", original)
    resumed = store.optimize("repair", circuit, theta, 25)
    assert resumed["interruption_replayed_joint_calls"] > 0
    assert (
        resumed["physical_joint_gradient_calls"]
        == resumed["optimizer"]["evaluations"] + resumed["interruption_replayed_joint_calls"]
    )
    attempts = cast("dict[str, Any]", pilot.read_record(tmp_path / "repair.attempts.json"))["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["objective_calls"] == 3
    assert not attempts[0]["completed"]
    assert attempts[1]["completed"]
