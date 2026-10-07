# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent q2 checks for custodied overnight training and exact resume."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider, exact_density_fidelity
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import runner
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def engineering_case() -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Return a small complex target and mixed-gate circuit with shared parameters."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), 0),
            ParameterizedGate("rx", (1,), 1),
            ParameterizedGate("rxx", (1, 0), 2),
            ParameterizedGate("rz", (0,), 0, angle_scale=-0.5, angle_offset=0.2),
        ],
    )
    return circuit, np.array([0.2, -0.3, 0.4]), np.array([1, 2j, -3, 4j]) / np.sqrt(30)


@pytest.mark.parametrize(
    ("method", "iterations", "interrupt"), [("adam", 7, 3), ("weighted_adam", 7, 3), ("adaptive_online", 27, 26)]
)
def test_exact_resume_preserves_parameters_moments_and_counters(method: str, iterations: int, interrupt: int) -> None:
    """A checkpoint interruption does not reset optimizer state or repeat updates."""
    circuit, initial, target = engineering_case()
    before = initial.copy()
    config = {"method": method, "iterations": iterations, "checkpoint_every": 1, "paths": 4}
    complete = runner.train(circuit, initial, target, config, 1729)
    saved: list[dict[str, Any]] = []

    def checkpoint(state: dict[str, Any]) -> None:
        """Record a complete state then simulate process interruption.

        Raises:
            RuntimeError: At the chosen engineering interruption boundary.
        """
        saved.append(copy.deepcopy(state))
        if state["iterations"] == interrupt:
            msg = "Engineering interruption."
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="Engineering interruption"):
        runner.train(circuit, initial, target, config, 1729, checkpoint=checkpoint)
    source = copy.deepcopy(saved[-1])
    resumed = runner.train(circuit, initial, target, config, 1729, state=saved[-1])
    assert saved[-1] == source
    np.testing.assert_array_equal(initial, before)
    for key in (
        "theta",
        "first",
        "second",
        "counts",
        "iterations",
        "exact_training_evaluations",
        "trajectory_paths",
        "direction_passes",
        "stop_reason",
    ):
        assert resumed[key] == complete[key]
    for actual, reference in zip(resumed["curve"], complete["curve"], strict=True):
        for key in ("iteration", "theta", "fidelity", "exact_training_evaluations", "trajectory_paths"):
            assert actual[key] == reference[key]
    if method == "adaptive_online":
        # Parameter zero occurs twice per sweep; the two adaptive sweeps after
        # the 25-sweep prefix must preserve occurrence counters across resume.
        assert resumed["counts"] == [4, 2, 2]
        assert resumed["direction_passes"] == 3 * 4 * iterations
    elif method == "weighted_adam":
        assert resumed["trajectory_paths"] == 4 * iterations
        assert resumed["direction_passes"] == 2 * 4 * iterations
    else:
        assert resumed["exact_training_evaluations"] == iterations
        assert resumed["direction_passes"] == 2 * iterations


def test_lbfgs_runner_resume_preserves_secant_history_and_full_optimizer_state() -> None:
    """Runner resume supplies the original initial vector and preserves curvature."""
    circuit, initial, _ = engineering_case()
    target = np.array([1, 0, 0, 0], dtype=complex)
    config = {"method": "lbfgs", "iterations": 8, "evaluations": 60, "history": 3}
    complete = runner.train(circuit, initial, target, config, 1729)
    saved = []

    def checkpoint(state: dict[str, Any]) -> None:
        """Retain a checkpoint with nonempty curvature history then interrupt.

        Raises:
            RuntimeError: Once two accepted curvature updates have completed.
        """
        if state["iterations"] == 2:
            saved.append(copy.deepcopy(state))
            msg = "L-BFGS interruption."
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="L-BFGS interruption"):
        runner.train(circuit, initial, target, config, 1729, checkpoint=checkpoint)
    assert len(saved[0]["optimizer"]["s_history"]) == 2
    assert saved[0]["theta"] != initial.tolist()
    unchanged_state = copy.deepcopy(saved[0])
    resumed = runner.train(circuit, initial, target, config, 1729, state=saved[0])
    assert saved[0] == unchanged_state
    assert resumed["optimizer"] == complete["optimizer"]
    assert resumed["stop_reason"] == complete["stop_reason"]
    assert resumed["theta"] == complete["theta"]
    assert resumed["exact_training_evaluations"] == complete["exact_training_evaluations"]
    assert resumed["direction_passes"] == 2 * complete["exact_training_evaluations"]
    assert resumed["trajectory_paths"] == 0
    for actual, reference in zip(resumed["curve"], complete["curve"], strict=True):
        assert actual["theta"] == reference["theta"]
        assert actual["iteration"] == reference["iteration"]
        assert actual["fidelity"] == reference["fidelity"]
    independent = exact_density_fidelity(circuit, np.asarray(resumed["theta"]), target, ExactLocalPauliProvider())
    assert resumed["curve"][-1]["fidelity"] == pytest.approx(independent, abs=1e-13)


def test_weighted_methods_receive_identical_fresh_physical_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optimizer choice does not change maps, weights, or paired update seeds."""
    circuit, initial, target = engineering_case()
    observed = []
    original = runner.FactorSampler.ensemble

    def ensemble(sampler: runner.FactorSampler, count: int, seed: int) -> tuple:
        """Return the real physical maps after retaining their exact contents."""
        maps, weights = original(sampler, count, seed)
        content = tuple(
            tuple(tuple((matrix.tobytes(), sites) for matrix, sites in gate.operators) for gate in path)
            for path in maps
        )
        observed.append((count, seed, content, weights.tolist()))
        return maps, weights

    monkeypatch.setattr(runner.FactorSampler, "ensemble", ensemble)
    states = [
        runner.train(circuit, initial, target, {"method": method, "iterations": 4, "paths": 6}, 99)
        for method in ("weighted_adam", "adaptive_online")
    ]
    assert observed[:4] == observed[4:]
    assert len({entry[1] for entry in observed[:4]}) == 4
    assert [state["trajectory_paths"] for state in states] == [24, 24]
    assert [state["direction_passes"] for state in states] == [48, 72]
    assert all(state["exact_training_evaluations"] == 0 for state in states)


@pytest.mark.parametrize("method", ["adam", "noiseless_adam", "homotopy"])
def test_final_iterate_curve_scores_actual_saved_parameters(method: str) -> None:
    """Curves use each saved physical endpoint and include the final uneven block."""
    circuit, initial, target = engineering_case()
    checkpoints = []
    state = runner.train(
        circuit,
        initial,
        target,
        {"method": method, "iterations": 5, "checkpoint_every": 2},
        99,
        checkpoint=lambda saved: checkpoints.append(copy.deepcopy(saved)),
    )
    assert [point["iteration"] for point in state["curve"]] == [0, 2, 4, 5]
    assert [saved["iterations"] for saved in checkpoints] == [2, 4, 5]
    assert state["curve"][-1]["theta"] == state["theta"]
    assert state["exact_training_evaluations"] == 5
    for point in state["curve"]:
        independent = exact_density_fidelity(circuit, np.asarray(point["theta"]), target, ExactLocalPauliProvider())
        assert point["fidelity"] == pytest.approx(independent, abs=1e-13)
    json.dumps(state, allow_nan=False)


def test_unchanged_control_is_physical_zero_update_endpoint() -> None:
    """The unchanged circuit is evaluated while consuming no training work."""
    circuit, initial, target = engineering_case()
    state = runner.train(circuit, initial, target, {"method": "unchanged", "iterations": 0}, 11)
    np.testing.assert_array_equal(state["theta"], initial)
    assert len(state["curve"]) == 1
    assert state["trajectory_paths"] == state["direction_passes"] == state["exact_training_evaluations"] == 0
    assert state["training_cpu_seconds"] == 0
    assert state["curve"][0]["fidelity"] == pytest.approx(
        exact_density_fidelity(circuit, initial, target, ExactLocalPauliProvider()), abs=1e-13
    )


def test_dependency_depth_and_gate_count_include_fixed_and_disabled_gates() -> None:
    """Physical resource counts follow qubit dependencies, not list length."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("h", (0,), noise_enabled=False),
            ParameterizedGate("x", (1,)),
            ParameterizedGate("cx", (0, 1)),
            ParameterizedGate("rz", (1,), 0),
            ParameterizedGate("ry", (0,), 1),
        ],
    )
    assert runner.counts(circuit) == {"one_qubit": 4, "two_qubit": 1, "depth": 3}
    assert runner.counts(ParameterizedCircuit(2, [])) == {"one_qubit": 0, "two_qubit": 0, "depth": 0}


def test_seed_domains_are_deterministic_paired_and_disjoint() -> None:
    """Final testing and checkpoint selection never reuse a training stream."""
    phases = ("development_training", "development_diagnostics", "confirmation_training", "final_evaluation")
    seeds = [
        runner.seed_for(phase, "engineering", replica, step, trajectory)
        for phase in phases
        for replica in (11, 29)
        for step in range(3)
        for trajectory in range(4)
    ]
    assert len(set(seeds)) == len(seeds)
    assert runner.seed_for("final_evaluation", "engineering", 11, 2, 3) == runner.seed_for(
        "final_evaluation", "engineering", 11, 2, 3
    )


def test_custody_rejects_changed_bytes_using_only_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The immutable fixture is never edited by this custody regression test."""
    repo, root = tmp_path / "repo", tmp_path / "research"
    repo.mkdir()
    root.mkdir()
    fixture = repo / "engineering_fixture.json"
    fixture.write_text('{"target": [1, 0]}\n')
    manifest = {"benchmark_hashes": {fixture.name: hashlib.sha256(fixture.read_bytes()).hexdigest()}}
    (root / "MANIFEST.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "REPO", repo)
    assert runner.custody() == manifest
    fixture.write_text('{"target": [0, 1]}\n')
    with pytest.raises(ValueError, match="Benchmark custody mismatch"):
        runner.custody()


def test_run_job_changed_iteration_budget_rejects_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A new optimizer budget cannot be silently resumed under an old signature."""
    circuit, initial, target = engineering_case()
    monkeypatch.setattr(runner, "custody", dict)
    monkeypatch.setattr(runner, "load_target", lambda *_: SimpleNamespace(state_vector_copy=target.copy))
    path = tmp_path / "engineering.json"
    job: dict[str, Any] = {
        "path": str(path),
        "config": {"method": "adam", "iterations": 3},
        "target_id": "engineering",
        "circuit": pilot.circuit_record(circuit),
        "initial": initial.tolist(),
        "seed": 11,
    }
    old_job = copy.deepcopy(job)
    old_job["config"]["iterations"] = 2
    pilot.write_record(
        path.with_suffix(".checkpoint.json"),
        {
            "signature": pilot.digest({k: v for k, v in old_job.items() if k != "path"}),
            "state": {},
        },
    )
    with pytest.raises(ValueError, match="Checkpoint configuration mismatch"):
        runner.run_job(job)


def test_completed_run_changed_config_rejected_before_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Completed output is never relabeled as a result for a changed method."""
    monkeypatch.setattr(runner, "custody", dict)
    path = tmp_path / "engineering.json"
    job = {"path": str(path), "config": {"method": "adam", "iterations": 2}}
    pilot.write_record(path, {"signature": "not-the-job-signature", "status": "complete"})
    with pytest.raises(ValueError, match="Completed run configuration mismatch"):
        runner.run_job(job)


def test_incomplete_checksumming_cannot_masquerade_as_completed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid checksum authenticates bytes but does not prove completion."""
    monkeypatch.setattr(runner, "custody", dict)
    path = tmp_path / "engineering.json"
    job = {"path": str(path), "config": {"method": "adam", "iterations": 2}}
    signature = pilot.digest({k: v for k, v in job.items() if k != "path"})
    pilot.write_record(path, {"signature": signature, "status": "failed"})
    with pytest.raises(ValueError, match="Completed run configuration mismatch"):
        runner.run_job(job)


def test_source_change_blocks_even_completed_cached_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A retained endpoint cannot bypass the frozen numerical-source identity."""
    monkeypatch.setattr(runner, "custody", dict)
    monkeypatch.setattr(runner, "source_hashes", lambda: {"engineering.py": "changed"})
    path = tmp_path / "engineering.json"
    job = {
        "path": str(path),
        "config": {"method": "adam", "iterations": 2},
        "source_hash": pilot.digest({"engineering.py": "original"}),
    }
    signature = pilot.digest({k: v for k, v in job.items() if k != "path"})
    pilot.write_record(path, {"signature": signature, "status": "complete"})
    with pytest.raises(ValueError, match="Numerical source fingerprint changed"):
        runner.run_job(job)


def test_q2_completed_job_fanout_and_authenticated_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A saved final circuit supplies all channels and resume preserves its bytes."""
    circuit, initial, target = engineering_case()
    monkeypatch.setattr(runner, "custody", dict)
    monkeypatch.setattr(runner, "load_target", lambda *_: SimpleNamespace(state_vector_copy=target.copy))
    path = tmp_path / "engineering.json"
    job = {
        "path": str(path),
        "config": {"method": "adam", "iterations": 2, "checkpoint_every": 1},
        "target_id": "engineering",
        "circuit": pilot.circuit_record(circuit),
        "initial": initial.tolist(),
        "seed": 11,
    }
    result = runner.run_job(job)
    assert result["status"] == "complete"
    assert result["state"]["iterations"] == 2
    assert set(result["fidelities"]) == set(runner.channels.NOISE_IDS)
    assert result["logical_counts"] == runner.counts(circuit)
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    assert pilot.digest(runner.run_job(job)) == pilot.digest(result)
    assert before == {p: p.read_bytes() for p in tmp_path.iterdir()}
    checkpoint = cast("dict[str, Any]", pilot.read_record(path.with_suffix(".checkpoint.json")))
    assert checkpoint["state"]["theta"] == result["state"]["theta"]
    assert checkpoint["signature"] == result["signature"]


@pytest.mark.parametrize("method", ["unknown", "ADAM", ""])
def test_unknown_method_is_rejected(method: str) -> None:
    """A misspelled algorithm cannot silently execute exact Adam."""
    circuit, initial, target = engineering_case()
    with pytest.raises(ValueError, match="Unknown refinement method"):
        runner.train(circuit, initial, target, {"method": method, "iterations": 1}, 11)


@pytest.mark.parametrize(
    "config",
    [
        {"method": "adam", "iterations": -1},
        {"method": "adam", "iterations": True},
        {"method": "adam", "iterations": 2, "checkpoint_every": 0},
        {"method": "adam", "iterations": 2, "checkpoint_every": 1.5},
        {"method": "unchanged", "iterations": 1},
    ],
)
def test_invalid_training_budget_rejected(config: dict[str, Any]) -> None:
    """Reject invalid schedules before consuming numerical work."""
    circuit, initial, target = engineering_case()
    with pytest.raises(ValueError, match=r"Invalid refinement budget|Unchanged control"):
        runner.train(circuit, initial, target, config, 11)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("second", [-1, 0, 0]),
        ("counts", [0, -1, 0]),
        ("counts", [0.5, 0, 0]),
        ("counts", [True, 0, 0]),
        ("first", [0, 1]),
        ("iterations", 8),
    ],
)
def test_invalid_resume_state_rejected(key: str, value: object) -> None:
    """Malformed adaptive state cannot be converted into an apparently valid resume."""
    circuit, initial, target = engineering_case()
    config = {"method": "adam", "iterations": 2}
    saved = runner.train(circuit, initial, target, config, 11)
    saved[key] = value
    with pytest.raises(ValueError, match=r"Invalid saved optimizer state|Saved occurrence counters"):
        runner.train(circuit, initial, target, config, 11, state=saved)
