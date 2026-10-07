# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent scoring, lineage, frozen controls and authentication for stability runs."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from experiments import topdown_optimizer_stability as study
from experiments.legacy_topdown_dense import train
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate, StatevectorBackend

if TYPE_CHECKING:
    from pathlib import Path


def fixture() -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Exercise reversed gate sites, shared parameters, affine angles and complex targets.

    Returns:
        The circuit, its shared parameter vector and a normalized complex target.
    """
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (1,), 0, angle_scale=1.7, angle_offset=0.3),
            ParameterizedGate("rxx", (1, 0), 1),
            ParameterizedGate("rz", (0,), 0, angle_scale=-0.4),
        ],
        2,
    )
    return circuit, np.array([0.3, -0.7]), np.array([1, 2j, -3, 4j]) / np.sqrt(30)


def test_independent_fidelity_and_gradient_sign() -> None:
    """Qiskit primitive gates verify both the backend state and its loss derivative."""
    circuit, theta, target = fixture()
    assert study.independent_fidelity(circuit, theta, target) == pytest.approx(
        StatevectorBackend(circuit, target).metrics(theta)[1], abs=1e-14
    )
    result = study.gradient_check(circuit, theta, target, 100)
    assert len(result["checks"]) == 6
    assert max(item["absolute_error"] for item in result["checks"]) < 1e-8


def test_protocol_keeps_noisy_fine_tuning_and_d_repair_frozen() -> None:
    """The construction-only ablation cannot silently change the downstream algorithm."""
    frozen = study.depth_search.protocol_for(100)
    actual = study.protocol()
    for key in ("fine_tuning", "pruning", "validation_paths", "validation_entropy", "depth_cap", "patience"):
        assert actual[key] == frozen[key]
    assert actual["policies"] == list(study.POLICIES)
    assert "baseline" not in actual
    actual["fine_tuning"]["num_trajectories"] = 999
    assert study.depth_search.protocol_for(100)["fine_tuning"]["num_trajectories"] == 100


def test_original_construction_matches_legacy_and_records_initial_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original arm has the same arithmetic and archives the exact replay input."""
    circuit, _, target = fixture()
    monkeypatch.setattr(study, "brickwall_for", lambda _target, _depth: circuit)
    previous = np.array([0.31])
    expected_initial = study.extend(previous, 2, 1302, 0.001)
    theta, trace = train(circuit, expected_initial, target, seed=3300, **study.depth_search.PROTOCOL["construction"])
    actual = study.construct(tmp_path, target, 300, 2, previous, "original_final")
    np.testing.assert_array_equal(actual["theta"], theta)
    np.testing.assert_array_equal(actual["initial_theta"], expected_initial)
    np.testing.assert_array_equal(actual["training"]["fidelity_trace"], trace["fidelity_trace"])
    assert actual["ideal_fidelity"] == actual["training"]["returned_fidelity"]
    np.testing.assert_array_equal(previous, [0.31])
    with pytest.raises(ValueError, match="different optimizer policy"):
        study.construct(tmp_path, target, 300, 2, previous, "best_only")


def test_parity_rejects_changed_noise_banks_and_traces() -> None:
    """A matching terminal parameter vector alone is insufficient for control parity."""
    actual = {
        "circuit": {},
        "theta": [0.2],
        "training": {"fidelity_trace": [0.7], "noise_banks": [1]},
        "status": "complete",
    }
    expected = json.loads(json.dumps(actual))
    assert study.assert_parity(actual, expected)["theta_max_error"] == 0
    expected["training"]["noise_banks"] = [2]
    with pytest.raises(AssertionError):
        study.assert_parity(actual, expected)
    expected["training"]["noise_banks"] = [1]
    expected["training"]["fidelity_trace"] = [0.6]
    with pytest.raises(AssertionError):
        study.assert_parity(actual, expected)
    with pytest.raises(ValueError, match="shapes differ"):
        study.maximum_error([1], [])


def test_completed_job_rejects_modified_artifact(tmp_path: Path) -> None:
    """Resume refuses a checkpoint modified after its completion marker was written."""
    assert study.load_complete(tmp_path) is None
    artifact = tmp_path / "record.json"
    artifact.write_text('{"value": 1}')
    complete = study.save_complete(tmp_path, {"target_id": "fixture"})
    assert study.load_complete(tmp_path) == complete
    artifact.write_text('{"value": 2}')
    with pytest.raises(ValueError, match="artifacts changed"):
        study.load_complete(tmp_path)


def test_fixed_replay_uses_archived_parents_for_each_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Returning different best parameters cannot contaminate the next fixed input."""
    output, reference = tmp_path / "output", tmp_path / "reference"
    source = reference / "fixture__seed300" / "search"
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
        final = initial + 0.2
        returned = final if policy == "original_final" else initial + 0.1
        fidelity = study.independent_fidelity(circuit, returned, target)
        return returned, {
            "steps": 1,
            "fidelity_trace": [0.5],
            "initial_fidelity": 0.5,
            "online_final_fidelity": 0.5,
            "best_fidelity": 0.5,
            "final_fidelity": 0.5,
            "returned_fidelity": fidelity,
            "best_iteration": 0,
            "gradient_evaluations": 1,
            "objective_evaluations": 2,
            "rejected_trials": 0,
            "failed_line_searches": 0,
            "final_theta": final.tolist(),
        }

    monkeypatch.setattr(study, "train_noiseless", fake_train)
    previous = np.empty(0)
    for depth in (1, 2):
        circuit = circuits[depth]
        initial = study.extend(previous, depth, 1300 + depth, 0.05 if depth == 1 else 0.001)
        theta, trace = fake_train(circuit, initial, target, policy="original_final")
        study.depth_search.save_checkpoint(
            source, f"construction_{depth:02d}", circuit, theta, {"depth": depth, "training": trace}
        )
        previous = theta
    result = study.run_fixed((str(output), str(reference), "fixture", 300))
    assert len(result["records"]) == 6
    first = json.loads((source / "construction_01.json").read_text())
    for policy in study.POLICIES:
        record = study.depth_search.read_checkpoint(output / "fixed/fixture__seed300" / policy, "construction_02")
        assert record is not None
        assert record["initial_theta"][0] == first["theta"][0]
    assert max(item["independent_returned_fidelity_error"] for item in result["records"]) == 0


def test_pipeline_grows_returned_parameters_and_locks_before_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All four downstream branches share growth, retain M=100 and lock validation winners."""
    target = np.array([1, 1j]) / np.sqrt(2)
    target_record = SimpleNamespace(state_vector_copy=target.copy, seed=42)
    monkeypatch.setattr(
        study, "load_target_collection", lambda: SimpleNamespace(load=lambda _qubits, _target: target_record)
    )
    parents, pruning_calls, candidate_calls, evaluation_calls = [], [], [], []

    def fake_construct(
        _root: Path, _target: np.ndarray, _seed: int, depth: int, previous: np.ndarray, policy: str
    ) -> dict:
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
        pruning_calls.append((built["depth"], variant))
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

    def fake_evaluate(root: Path, item: dict, _target: np.ndarray) -> dict:
        locked = json.loads((root.parent / "selection_locked.json").read_text())
        assert all(branch["selected_depth"] == 1 for branch in locked["branches"])
        evaluation_calls.append(item["depth"])
        return {"depth": item["depth"]}

    monkeypatch.setattr(study, "construct", fake_construct)
    monkeypatch.setattr(study.depth_search, "prune_checkpoint", fake_prune)
    monkeypatch.setattr(study.depth_search, "candidate", fake_candidate)
    monkeypatch.setattr(study.depth_search, "evaluate_endpoint", fake_evaluate)
    result = study.run_pipeline((str(tmp_path), str(tmp_path / "unused"), "fixture", 300, 0, "best_only"))
    assert len(result["constructions"]) == 4
    assert len(pruning_calls) == 8
    assert len(candidate_calls) == len(evaluation_calls) == 16
    assert all(not baseline and count == 100 for _, _, baseline, count in candidate_calls)
    assert len(parents[0]) == 0
    for depth, previous in enumerate(parents[1:], start=1):
        np.testing.assert_allclose(previous, [0.2 * depth])
    assert len(result["results"]) == 4
    assert all(branch["selected"]["depth"] == 1 for branch in result["results"])
    assert all(branch["search_cost"]["construction_updates"] == 800 for branch in result["results"])
    assert result["reference_parity"] is None
    assert "baseline" not in result["results"][0]
