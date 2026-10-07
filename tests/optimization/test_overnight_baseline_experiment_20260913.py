# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Strong baseline state preservation, epoch accounting and final-iterate fairness."""

from __future__ import annotations

import copy
from dataclasses import asdict
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import baseline_experiment as experiment
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def small_case() -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Return a six-gate q2 preparation with visible deletion/noise effects."""
    gates = [
        ParameterizedGate(name, sites, index, logical_gate_id=index)
        for index, (name, sites) in enumerate((
            ("ry", (0,)),
            ("rz", (1,)),
            ("rxx", (0, 1)),
            ("ry", (1,)),
            ("rzz", (0, 1)),
            ("rz", (0,)),
        ))
    ]
    target = np.array([0.7 + 0.1j, 0.2 - 0.3j, -0.1 + 0.4j, 0.2 + 0.5j])
    return ParameterizedCircuit(2, gates), np.array([0.3, 0.4, 0.2, -0.2, 0.1, 0.5]), target / np.linalg.norm(target)


def small_config() -> experiment.Config:
    """Return a bounded schedule with an actual post-25 adaptive tail."""
    return experiment.Config(
        total_updates=60, warmup=30, rounds=2, shortlist=1, repair=5, checkpoint_every=5, trajectories=4, protected=2
    )


@pytest.mark.parametrize("method", ["weighted_adam", "adaptive_online"])
def test_stochastic_piece_exactly_reproduces_historical_kernel_and_epochs(method: str) -> None:
    """The stronger repository baselines keep their sampling/update semantics."""
    circuit, theta, target = small_case()
    initial = experiment.initial_state(circuit, theta)
    config = small_config()
    old_method = "adam" if method == "weighted_adam" else method
    legacy_config = experiment.legacy.Config(trajectories=config.trajectories)
    warm = experiment.legacy.train_block(initial, target, old_method, legacy_config, 117, 0, 30)
    current_warm = experiment.train_piece(initial, target, method, 30, 0, 117, config)
    assert current_warm["state"] == warm["state"]
    assert current_warm["trace"] == warm["trace"]
    expected = experiment.legacy.train_block(warm["state"], target, old_method, legacy_config, 117, 100, 6)
    actual = experiment.train_piece(warm["state"], target, method, 6, 100, 117, config)
    for key in ("state", "trace", "trajectory_passes", "no_factor_probability"):
        assert actual[key] == expected[key]
    assert actual["fidelity"] == pytest.approx(expected["fidelity"], abs=1e-12)
    assert [row["epoch"] for row in actual["trace"]] == list(range(101, 107))
    assert [row["step"] for row in actual["trace"]] == list(range(31, 37))
    if method == "adaptive_online":
        assert all(row["phase"] == "adaptive_online" for row in actual["trace"])
        assert actual["state"]["counts"] == [11] * 6


@pytest.mark.parametrize("method", list(experiment.METHODS))
def test_block_partial_resume_preserves_all_moments_counters_and_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Accepted-piece interruption reproduces the complete final scientific state."""
    circuit, theta, target = small_case()
    config = small_config()
    warm = experiment.train_piece(experiment.initial_state(circuit, theta), target, method, 30, 0, 117, config)
    full = experiment.BlockStore(tmp_path / "full", target, method, 117, config).run("tail", warm["state"], 11, 100)
    original_write = pilot.write_record

    def interrupted_write(path: Path, record: dict[str, Any]) -> None:
        original_write(path, record)
        if path.name == "tail.partial.json" and record["completed"] == 5:
            raise KeyboardInterrupt

    store = experiment.BlockStore(tmp_path / "resume", target, method, 117, config)
    monkeypatch.setattr(pilot, "write_record", interrupted_write)
    with pytest.raises(KeyboardInterrupt):
        store.run("tail", warm["state"], 11, 100)
    monkeypatch.setattr(pilot, "write_record", original_write)
    resumed = store.run("tail", warm["state"], 11, 100)
    assert pilot.digest(resumed["state"]) == pilot.digest(full["state"])
    assert resumed["trace"] == full["trace"]
    assert resumed["fidelity"] == full["fidelity"]
    assert resumed["epoch_end"] == 111
    assert resumed["attempted_or_reserved_updates"] == 11
    assert resumed["exact_score_calls"] == 3
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert pilot.digest(store.run("tail", warm["state"], 11, 100)) == pilot.digest(resumed)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="block identity changed"):
        store.run("tail", warm["state"], 12, 100)


def test_refiner_projects_parent_memory_and_pairs_round_epochs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleted candidates inherit parent memory, never a previous sibling's update."""
    circuit, theta, target = small_case()
    config = small_config()
    store = experiment.BlockStore(tmp_path, target, "adaptive_online", 117, config)
    warm = store.run("warm", experiment.initial_state(circuit, theta), 30, 0)
    refiner = experiment.MemoryRefiner(store, warm)
    calls = []
    original_run = store.run

    def observed(name: str, start: dict[str, Any], updates: int, epoch: int) -> dict[str, Any]:
        calls.append((copy.deepcopy(start), epoch))
        return original_run(name, start, updates, epoch)

    monkeypatch.setattr(store, "run", observed)
    retained = refiner(circuit, np.asarray(warm["state"]["theta"]), 5)
    deleted = experiment.legacy.deleted_state(warm["state"], 2)
    refiner(pilot.restore_circuit(deleted["circuit"]), np.asarray(deleted["theta"]), 5)
    assert calls[0][1] == calls[1][1] == 30
    assert calls[0][0]["updates"] == calls[1][0]["updates"] == 30
    assert calls[1][0]["counts"] == [5] * 5
    assert calls[1][0]["second"] == [v for i, v in enumerate(warm["state"]["second"]) if i != 2]
    refiner(circuit, retained.theta, 5)
    assert calls[2][1] == 35
    assert calls[2][0]["updates"] == 35
    assert calls[2][0]["counts"] == [10] * 6
    # Rejection may keep the old state while its attempted noise epochs remain consumed.
    assert refiner.selected_state(warm["state"]["circuit"], warm["state"]["theta"])["updates"] == 30
    assert refiner.epoch == 40


@pytest.mark.parametrize("method", list(experiment.METHODS))
def test_structural_work_charges_rejected_repairs_and_only_exports_final_iterates(tmp_path: Path, method: str) -> None:
    """Every branch consumes the cap; final state is the polishing trajectory endpoint."""
    circuit, theta, target = small_case()
    config = small_config()
    result = experiment.run_method(circuit, theta, target, method, 117, config, tmp_path)
    arm = result["arms"][f"structural_{method}"]
    assert arm["nominal_charged_updates"] == config.total_updates
    assert arm["work"]["attempted_or_reserved_updates"] == config.total_updates
    polish = pilot.read_record(tmp_path / "blocks/polish.json")
    assert pilot.digest(arm["state"]) == pilot.digest(polish["state"])
    assert arm["theta"] == arm["curve"][-1]["theta"]
    assert arm["fidelity"] == arm["curve"][-1]["fidelity"]
    search = cast("dict[str, Any]", pilot.read_record(tmp_path / "SEARCH.json"))
    assert search["refinement_evaluations"] == sum(b["evaluations"] for r in search["rounds"] for b in r["branches"])
    assert all(g["logical_gate_id"] in {0, 1} for g in arm["circuit"]["gates"][:2])
    if method == "exact_adam":
        assert arm["work"]["exact_gradient_calls"] == config.total_updates
        assert arm["work"]["trajectory_paths"] == 0
    else:
        assert arm["work"]["trajectory_paths"] == config.trajectories * config.total_updates
        assert arm["work"]["trajectory_direction_passes"] == config.trajectories * config.total_updates * (
            2 if method == "weighted_adam" else 3
        )
    if method == "adaptive_online":
        fixed = result["arms"]["fixed_adaptive_online"]
        assert fixed["state"]["updates"] == config.total_updates
        assert fixed["state"]["counts"] == [config.total_updates - 25] * 6
        assert fixed["theta"] == fixed["curve"][-1]["theta"]
        assert result["shared_warmup_cpu_seconds"] > 0
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    replay = experiment.run_method(circuit, theta, target, method, 117, config, tmp_path)
    assert pilot.digest(replay) == pilot.digest(result)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_common_cpu_selection_never_uses_a_better_earlier_or_future_checkpoint() -> None:
    """A regressed last available state remains the exported final iterate."""
    curve = [
        {"cpu_seconds": 1.0, "fidelity": 0.95},
        {"cpu_seconds": 2.0, "fidelity": 0.8},
        {"cpu_seconds": 3.0, "fidelity": 0.99},
    ]
    assert experiment.last_available(curve, 2.5) == curve[1]
    assert experiment.last_available(curve, 2.0) == curve[1]
    assert experiment.last_available(curve, 0.5) is None


def test_common_cpu_panel_audits_one_final_circuit_per_method(monkeypatch: pytest.MonkeyPatch) -> None:
    """All six arms use the same cutoff and missing cases disable a matched claim."""
    names = (*experiment.LABELS, "structural_lbfgs", "fixed_lbfgs")
    arms: dict[str, dict[str, Any]] = {
        name: {
            "status": "complete",
            "cpu_seconds": 2.5 + i,
            "curve": [
                {"cpu_seconds": 1.0, "fidelity": 0.9, "theta": [1]},
                {"cpu_seconds": 2.0, "fidelity": 0.8, "theta": [2]},
                {"cpu_seconds": 3.0, "fidelity": 0.99, "theta": [3]},
            ],
        }
        for i, name in enumerate(names)
    }
    monkeypatch.setattr(experiment, "evaluate_endpoint", lambda arm, _target: copy.deepcopy(arm))
    matched = experiment.common_cpu_panel(arms, np.array([1, 0]))
    assert matched["cpu_seconds"] == pytest.approx(2.5)
    assert all(arm["theta"] == [2] for arm in matched["arms"].values())
    arms[experiment.LABELS[0]]["status"] = "failed"
    assert experiment.common_cpu_panel(arms, np.array([1, 0]))["status"] == "unavailable_incomplete_panel"


def test_prepare_only_matches_frozen_starts_without_reading_endpoint_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forthcoming L-BFGS run can be preregistered without conditioning on its outcomes."""
    source = tmp_path / "source"
    reference_root = tmp_path / "reference"
    gates = [ParameterizedGate("ry", (i % 6,), i, logical_gate_id=i) for i in range(78)]
    gates.extend(ParameterizedGate("rzz", (i % 5, i % 5 + 1), i, logical_gate_id=i) for i in range(78, 108))
    circuit = ParameterizedCircuit(6, gates)
    starts = [
        {
            "target_id": target,
            "replica": 11,
            "candidate": {"circuit": pilot.circuit_record(circuit), "theta": [0.1] * 108},
        }
        for target in experiment.TARGETS
    ]
    pilot.write_record(source / "starts.json", {"starts": starts})
    reference: dict[str, Any] = {
        "phase": "development",
        "starts": starts,
        "config": {
            "full_evaluations": 1500,
            "warmup_evaluations": 150,
            "max_rounds": 15,
            "shortlist": 2,
            "repair_evaluations": 25,
            "history": 10,
            "protected_initial_gates": 18,
        },
    }
    pilot.write_record(reference_root / "PROTOCOL.json", reference)
    monkeypatch.setattr(experiment.runner, "OUTPUT", source)
    monkeypatch.setattr(experiment.runner, "custody", lambda: {"benchmark_hashes": {"fixture": "frozen"}})
    monkeypatch.setattr(experiment, "numerical_sources", lambda: {"driver": "frozen"})
    monkeypatch.setattr(experiment, "run_method", lambda *_a, **_k: pytest.fail("prepare must never train"))
    monkeypatch.setattr(experiment.channels, "evaluate", lambda *_a: pytest.fail("prepare must never evaluate"))
    output = tmp_path / "prepared"
    protocol = experiment.freeze_protocol(output, reference_root)
    assert protocol["config"] == asdict(experiment.Config())
    experiment.verify_protocol(protocol)
    before = (output / "PROTOCOL.json").read_bytes()
    experiment.freeze_protocol(output, reference_root)
    assert before == (output / "PROTOCOL.json").read_bytes()
    assert [p.name for p in reference_root.iterdir()] == ["PROTOCOL.json"]
    monkeypatch.setattr(experiment, "numerical_sources", lambda: {"driver": "changed"})
    with pytest.raises(ValueError, match="identity changed"):
        experiment.verify_protocol(protocol)


def test_failed_repair_reservation_is_charged_and_never_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed shortlisted branch does not become free search work."""
    circuit, theta, target = small_case()
    original_run = experiment.BlockStore.run

    def failing_run(
        self: experiment.BlockStore, name: str, start: dict[str, Any], updates: int, epoch: int
    ) -> dict[str, Any]:
        if name == "repair_002":
            msg = "injected numerical repair failure"
            raise FloatingPointError(msg)
        return original_run(self, name, start, updates, epoch)

    monkeypatch.setattr(experiment.BlockStore, "run", failing_run)
    result = experiment.run_method(circuit, theta, target, "weighted_adam", 117, small_config(), tmp_path)
    arm = result["arms"]["structural_weighted_adam"]
    assert arm["failed_branches"] == 1
    assert arm["work"]["reserved_failed_updates"] == 5
    assert arm["work"]["attempted_or_reserved_updates"] == 60
    assert arm["work"]["trajectory_paths"] == 240
    search = cast("dict[str, Any]", pilot.read_record(tmp_path / "SEARCH.json"))
    assert search["rounds"][0]["selected_branch"] != 1


@pytest.mark.parametrize("method", list(experiment.METHODS))
def test_optimizer_diagnostics_do_not_use_slow_qiskit_oracle(method: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Checkpoint scoring uses the same fast exact channel map for every optimizer."""
    circuit, theta, target = small_case()
    monkeypatch.setattr(
        experiment.legacy, "fidelity", lambda *_a: pytest.fail("optimizer checkpoint must use fast scorer")
    )
    result = experiment.train_piece(experiment.initial_state(circuit, theta), target, method, 2, 0, 117, small_config())
    assert 0 < result["fidelity"] < 1


def test_interrupted_search_replays_identical_selection_and_retains_extra_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cached branch memory restores the same final endpoint after a midround interruption."""
    circuit, theta, target = small_case()
    config = small_config()
    full = experiment.run_method(circuit, theta, target, "adaptive_online", 117, config, tmp_path / "full")
    original_write = pilot.write_record

    def interrupted_write(path: Path, record: dict[str, Any]) -> None:
        original_write(path, record)
        if path.name == "repair_002.partial.json" and record["completed"] == 5:
            raise KeyboardInterrupt

    root = tmp_path / "resume"
    monkeypatch.setattr(pilot, "write_record", interrupted_write)
    with pytest.raises(KeyboardInterrupt):
        experiment.run_method(circuit, theta, target, "adaptive_online", 117, config, root)
    monkeypatch.setattr(pilot, "write_record", original_write)
    resumed = experiment.run_method(circuit, theta, target, "adaptive_online", 117, config, root)
    for label in ("structural_adaptive_online", "fixed_adaptive_online"):
        assert pilot.digest(resumed["arms"][label]["state"]) == pilot.digest(full["arms"][label]["state"])
        assert resumed["arms"][label]["fidelity"] == full["arms"][label]["fidelity"]
    search = cast("dict[str, Any]", pilot.read_record(root / "SEARCH.json"))
    attempts = cast("list[dict[str, Any]]", pilot.read_record(root / "SEARCH_ATTEMPTS.json")["attempts"])
    assert len(attempts) == 2
    assert search["physical_cpu_seconds"] == sum(a["cpu_seconds"] for a in attempts)
    assert search["exact_search_counts_complete"] is False


def test_q2_case_all_profiles_matched_cpu_and_completed_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the entire public case driver without any q6 numerical work."""
    circuit, theta, target = small_case()
    source = {
        "target_id": experiment.TARGETS[0],
        "replica": 11,
        "candidate": {"circuit": pilot.circuit_record(circuit), "theta": theta.tolist()},
    }
    reference_root = tmp_path / "reference"
    reference_protocol = {"scientific_test": "fixed"}
    monkeypatch.setattr(experiment, "verify_protocol", lambda _p: reference_protocol)
    monkeypatch.setattr(experiment, "load_target", lambda *_a: SimpleNamespace(state_vector_copy=target.copy))
    state = experiment.initial_state(circuit, theta)
    score = experiment.legacy.fidelity(state, target)
    point = experiment.endpoint(state, score, 0.0)
    reference_arm = experiment.evaluate_endpoint(
        {**point, "status": "complete", "cpu_seconds": 100.0, "curve": [point]}, target
    )
    reference_arm["logical_counts"] = reference_arm["physical_counts"]
    reference: dict[str, Any] = {
        "status": "complete",
        "signature": pilot.digest({"protocol": reference_protocol, "source": source}),
        "arms": {label: copy.deepcopy(reference_arm) for label in ("structural_lbfgs", "fixed_lbfgs")},
    }
    pilot.write_record(reference_root / f"{source['target_id']}__11/RESULT.json", reference)
    protocol = {"starts": [source], "reference_root": str(reference_root), "config": asdict(small_config())}
    root = tmp_path / "case"
    job = {"protocol": protocol, "source": source, "root": str(root)}
    result = experiment.execute_case(job)
    assert result["status"] == "complete"
    assert set(result["arms"]) == {*experiment.LABELS, "structural_lbfgs", "fixed_lbfgs"}
    assert result["common_cpu"]["status"] == "complete"
    for panel in (result["arms"], result["common_cpu"]["arms"]):
        assert all(set(arm["fidelities"]) == set(experiment.channels.NOISE_IDS) for arm in panel.values())
        assert all(arm["physical_counts"]["one_qubit"] >= 2 for arm in panel.values())
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert pilot.digest(experiment.execute_case(job)) == pilot.digest(result)
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    experiment.write_scoreboard(root, [result])
    assert len((root / "SCOREBOARD.csv").read_text().splitlines()) == 145


@pytest.mark.parametrize(
    "values", [{"total_updates": 20}, {"warmup": True}, {"repair": 0}, {"trajectories": 1}, {"protected": -1}]
)
def test_invalid_budget_rejected(values: dict[str, Any]) -> None:
    """No malformed or insufficient budget reaches numerical execution."""
    with pytest.raises(ValueError, match="Baseline budgets"):
        experiment.Config(**values)
