# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Frozen controls, identical fixed inputs and selection isolation for batch refinement."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from experiments import topdown_batch_refinement as study
from experiments.topdown_optimizer_policies import train_noiseless as old_train
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def test_protocol_preserves_downstream_and_names_strong_baselines() -> None:
    """Only noiseless batch refinement varies; both earlier winning arms remain controls."""
    frozen = study.depth_search.protocol_for(100)
    actual = study.protocol()
    for key in ("fine_tuning", "pruning", "validation_paths", "validation_entropy", "depth_cap", "patience"):
        assert actual[key] == frozen[key]
    assert actual["policies"] == ["best_only", "best_backtracking", "bb_backtracking", "lbfgs"]
    assert actual["primary_baselines"] == {"B": "best_backtracking", "D": "best_only"}
    assert actual["control_policies"] == ["best_only", "best_backtracking"]
    actual["fine_tuning"]["num_trajectories"] = 999
    assert study.depth_search.protocol_for(100)["fine_tuning"]["num_trajectories"] == 100


def test_manifest_rejects_changed_reference_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner cannot silently redefine the already frozen reference implementation."""
    repository, reference = tmp_path / "repo", tmp_path / "reference"
    repository.mkdir()
    reference.mkdir()
    source = repository / "numerics.py"
    source.write_text("pass\n")
    expected = study.file_hash(source)
    (reference / "manifest.json").write_text(json.dumps({"sources": {"numerics.py": expected}}))
    source.write_text("changed = True\n")
    monkeypatch.setattr(study, "REPOSITORY", repository)
    with pytest.raises(ValueError, match="Reference simulation source changed"):
        study.make_manifest(reference, [], [])


@pytest.mark.parametrize("policy", study.CONTROL_POLICIES)
def test_control_construction_matches_archived_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    """Explicit saved initial parameters take precedence over newly extended parents."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), 0)], 1)
    target = np.array([0.6, 0.8])
    initial = np.array([0.13])
    monkeypatch.setattr(study, "brickwall_for", lambda _target, _depth: circuit)
    theta, trace = old_train(circuit, initial, target, policy=policy)
    actual = study.construct(tmp_path, target, 300, 2, np.array([-99]), policy, initial_theta=initial)
    np.testing.assert_array_equal(actual["theta"], theta)
    np.testing.assert_array_equal(actual["initial_theta"], initial)
    np.testing.assert_array_equal(actual["training"]["fidelity_trace"], trace["fidelity_trace"])
    assert actual["ideal_fidelity"] == actual["training"]["returned_fidelity"]
    assert study.construct(tmp_path, target, 300, 2, np.array([99]), policy, initial_theta=initial) == actual
    with pytest.raises(ValueError, match="different policy or initial"):
        study.construct(tmp_path, target, 300, 2, np.array([99]), policy, initial_theta=initial + 0.1)
    with pytest.raises(ValueError, match="Initial parameter shape"):
        study.construct(tmp_path, target, 300, 2, np.empty(0), policy, initial_theta=np.empty(0))


def test_fixed_replay_uses_exact_saved_initials_for_all_four_policies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither reference endpoints nor other policies' outputs become fixed replay inputs."""
    output, reference = tmp_path / "output", tmp_path / "reference"
    source = reference / "pipeline/best_backtracking/fixture__seed300/search"
    target = np.array([1, 1j]) / np.sqrt(2)
    circuits = {
        depth: ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), index) for index in range(depth)], depth)
        for depth in (1, 2)
    }
    monkeypatch.setattr(study, "brickwall_for", lambda _target, depth: circuits[depth])
    collection = SimpleNamespace(load=lambda _qubits, _target: SimpleNamespace(state_vector_copy=target.copy))
    monkeypatch.setattr(study, "load_target_collection", lambda: collection)

    def fake_train(
        circuit: ParameterizedCircuit, initial: np.ndarray, _target: np.ndarray, *, policy: str, **_kwargs: object
    ) -> tuple[np.ndarray, dict]:
        returned = initial + (0.2 if policy == "best_backtracking" else 0.1)
        fidelity = study.stability.independent_fidelity(circuit, returned, target)
        return returned, {
            "steps": 200,
            "fidelity_trace": [0.5] * 200,
            "initial_fidelity": 0.5,
            "online_final_fidelity": 0.5,
            "online_final_theta": (initial + 0.01).tolist(),
            "best_fidelity": 0.5,
            "final_fidelity": 0.5,
            "returned_fidelity": fidelity,
            "best_iteration": 100,
            "gradient_evaluations": 100,
            "objective_evaluations": 201,
            "rejected_trials": 0,
            "failed_line_searches": 0,
            "final_theta": returned.tolist(),
        }

    monkeypatch.setattr(study, "train_noiseless", fake_train)
    originals = {1: np.array([7.3]), 2: np.array([-3.7, 4.8])}
    for depth, initial in originals.items():
        circuit = circuits[depth]
        theta, trace = fake_train(circuit, initial, target, policy="best_backtracking")
        study.depth_search.save_checkpoint(
            source,
            f"construction_{depth:02d}",
            circuit,
            theta,
            {"depth": depth, "initial_theta": initial.tolist(), "training": trace},
        )
    result = study.run_fixed((str(output), str(reference), "fixture", 300))
    assert len(result["records"]) == 8
    assert result["reference_policy"] == "best_backtracking"
    for policy in study.POLICIES:
        for depth, initial in originals.items():
            record = study.depth_search.read_checkpoint(
                output / "fixed/fixture__seed300" / policy, f"construction_{depth:02d}"
            )
            assert record is not None
            np.testing.assert_array_equal(record["initial_theta"], initial)
    assert max(item["online_prefix_max_error"] for item in result["records"]) == 0
    assert max(item["independent_returned_fidelity_error"] for item in result["records"]) == 0


@pytest.mark.parametrize("policy", study.POLICIES)
def test_pipeline_preserves_branches_and_verifies_each_control_against_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    """Policy-specific control parity and validation selection precede every exact score."""
    target = np.array([1, 1j]) / np.sqrt(2)
    target_record = SimpleNamespace(state_vector_copy=target.copy, seed=42)
    monkeypatch.setattr(
        study, "load_target_collection", lambda: SimpleNamespace(load=lambda _qubits, _target: target_record)
    )
    parents, candidate_calls, evaluation_calls, parity_calls = [], [], [], []

    def fake_construct(
        _root: Path, _target: np.ndarray, _seed: int, depth: int, previous: np.ndarray, active_policy: str
    ) -> dict:
        assert active_policy == policy
        parents.append(previous.copy())
        theta = [float(previous[0]) + 0.2] if len(previous) else [0.2]
        return {
            "depth": depth,
            "policy": policy,
            "theta": theta,
            "cpu_seconds": 0.1,
            "ideal_fidelity": 0.8,
            "training": {
                "steps": 200,
                "initial_fidelity": 0.7,
                "online_final_fidelity": 0.8,
                "best_fidelity": 0.8,
                "final_fidelity": 0.2,
                "returned_fidelity": 0.8,
                "best_iteration": 100,
                "gradient_evaluations": 100,
                "objective_evaluations": 201,
                "rejected_trials": 0,
                "failed_line_searches": 0,
            },
        }

    def fake_prune(_root: Path, _target: np.ndarray, built: dict, variant: str) -> dict:
        return {
            "depth": built["depth"],
            "theta": [-999.0],
            "pruning": {"cpu_seconds": 0.2, "repair_updates": 10 if variant == "D" else 0},
        }

    def fake_candidate(
        _root: Path,
        _target: np.ndarray,
        _seed: int,
        _target_index: int,
        pruned: dict,
        variant: str,
        hardware: str,
        *,
        baseline: bool,
        training_paths: int,
    ) -> dict:
        candidate_calls.append((variant, hardware, baseline, training_paths))
        return {
            "depth": pruned["depth"],
            "variant": variant,
            "hardware": hardware,
            "status": "complete",
            "training": {"steps": 200},
            "training_cpu_seconds": 0.3,
            "validation_cpu_seconds": 0.4,
            "validation": {"fidelity": 0.9 - 0.01 * pruned["depth"], "standard_error": 0.001},
        }

    def fake_verify(reference: Path, _root: Path, locked: dict, _constructions: list, _candidates: dict) -> dict:
        parity_calls.append(reference)
        assert locked["policy"] == policy
        return {"selection_identical": True}

    def fake_evaluate(root: Path, item: dict, _target: np.ndarray) -> dict:
        locked = json.loads((root.parent / "selection_locked.json").read_text())
        assert all(branch["selected_depth"] == 1 for branch in locked["branches"])
        evaluation_calls.append(item["depth"])
        return {"depth": item["depth"]}

    monkeypatch.setattr(study, "construct", fake_construct)
    monkeypatch.setattr(study.depth_search, "prune_checkpoint", fake_prune)
    monkeypatch.setattr(study.depth_search, "candidate", fake_candidate)
    monkeypatch.setattr(study.depth_search, "evaluate_endpoint", fake_evaluate)
    monkeypatch.setattr(study.stability, "verify_pipeline", fake_verify)
    reference = tmp_path / "reference"
    result = study.run_pipeline((str(tmp_path), str(reference), "fixture", 300, 0, policy))
    assert len(result["constructions"]) == 4
    assert len(candidate_calls) == len(evaluation_calls) == 16
    assert all(not baseline and count == 100 for _, _, baseline, count in candidate_calls)
    assert len(parents[0]) == 0
    for depth, previous in enumerate(parents[1:], start=1):
        np.testing.assert_allclose(previous, [0.2 * depth])
    assert len(result["results"]) == 4
    assert all(branch["selected"]["depth"] == 1 for branch in result["results"])
    if policy in study.CONTROL_POLICIES:
        assert parity_calls == [reference / "pipeline" / policy]
        assert result["reference_parity"]["selection_identical"]
    else:
        assert parity_calls == []
        assert result["reference_parity"] is None
