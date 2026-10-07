# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Small-system strong-baseline accounting, exact resume and provenance checks."""

from __future__ import annotations

import copy
import csv
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import (
    channels,
    trajectory_mixture,
)
from experiments.overnight_state_preparation_20260913_0108 import (
    mixture_baseline_experiment as experiment,
)
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import StatevectorBackend


def problem() -> tuple[ParameterizedCircuit, NDArray, NDArray]:
    """Return a q2 native scalar circuit with stable origins and a complex target."""
    gates = [
        ParameterizedGate(
            name,
            sites,
            index,
            angle_scale=(-1) ** index * 0.8,
            angle_offset=0.03,
            logical_gate_id=index,
            native_gate_id=f"native{index}",
            noise_enabled=index != 0,
        )
        for index, (name, sites) in enumerate([
            ("ry", (0,)),
            ("rx", (1,)),
            ("rxx", (1, 0)),
            ("rz", (0,)),
            ("ryy", (0, 1)),
            ("ry", (1,)),
        ])
    ]
    target = np.array([1, 2j, -3, 4j]) / np.sqrt(30)
    return ParameterizedCircuit(2, gates), np.linspace(-0.7, 1.1, len(gates)), target


def config(**changes: int) -> experiment.Config:
    """Return tiny engineering budgets with one symmetric candidate-repair round."""
    values = {
        "total_updates": 14,
        "warmup": 2,
        "rounds": 2,
        "shortlist": 1,
        "repair": 2,
        "protected": 1,
        "paths_per_channel": 2,
        "checkpoint_every": 2,
        "online_warmup": 2,
    }
    values.update(changes)
    return experiment.Config(**values)


def test_four_arms_all_channels_costs_compact_curves_and_byte_preserving_resume(tmp_path: Path) -> None:
    """Final and common-CPU results retain every arm, paid branch and physical condition."""
    circuit, theta, target = problem()
    result = experiment.run_case(tmp_path, circuit, theta, target, config())
    assert result["status"] == "complete"
    assert set(result["arms"]) == set(experiment.METHODS)
    store = experiment.Store(tmp_path, target, config(), 1)
    for label, arm in result["arms"].items():
        assert arm["nominal_charged_updates"] == 14
        assert arm["attempted_updates"] == arm["completed_updates"] == 14
        assert arm["completed_trajectory_paths"] == 56
        assert arm["completed_trajectory_direction_passes"] == (112 if label.endswith("weighted_adam") else 168)
        assert arm["physical_work"]["training_cpu_seconds"] == pytest.approx(arm["cpu_seconds"], abs=1e-12)
        assert arm["physical_work"]["completed_updates"] == 14
        assert arm["physical_work"]["accounting_status"] == "known"
        assert (
            arm["density_direction_passes"]
            == 2 * arm["exact_ranking_channel_calls"] + arm["exact_forward_channel_calls"]
        )
        assert arm["exact_gradient_channel_calls"] == 0
        state = arm["state"]
        final = store.restore(state["circuit_hash"])
        assert set(arm["fidelities"]) == set(channels.NOISE_IDS)
        for noise, fidelity in arm["fidelities"].items():
            assert fidelity == pytest.approx(
                channels.evaluate(final, np.asarray(state["theta"]), target, noise), abs=2e-14
            )
        assert all(set(point) == {"circuit_hash", "theta", "cpu_seconds", "updates"} for point in arm["curve"])
        assert all(
            left["cpu_seconds"] <= right["cpu_seconds"]
            for left, right in zip(arm["curve"], arm["curve"][1:], strict=False)
        )
        for row in arm["rounds"]:
            assert len(row["branches"]) == 2
            assert all(branch["origin"] is None or branch["origin"] >= 1 for branch in row["branches"])
            if row["selected_branch"] not in {None, 0}:
                choice = row["branches"][row["selected_branch"]]
                assert (
                    choice["fidelity"]
                    > max(row["incoming_fidelity"], row["branches"][0]["fidelity"]) + config().acceptance_tolerance
                )
        available = [point for point in arm["curve"] if point["cpu_seconds"] <= result["common_cpu"]["cpu_seconds"]]
        assert result["common_cpu"]["arms"][label]["point"] == available[-1]
    experiment.write_scoreboard(tmp_path, [result])
    rows = list(csv.DictReader((tmp_path / "SCOREBOARD.csv").read_text().splitlines()))
    assert len(rows) == 108
    assert {row["panel"] for row in rows} == {"full_cap", "common_cpu", "unchanged"}
    context = tmp_path / "engineering__11" / "REFERENCES.json"
    context.parent.mkdir()
    pilot.write_record(
        context,
        {
            "case_checksum": pilot.digest(result),
            "historical_unmatched": {
                "noiseless_adam": {
                    "comparison": "unmatched_historical_fixed_1100_individual_channel_call_cap",
                    "record": {"final_score": result["unchanged_start"]},
                }
            },
        },
    )
    experiment.write_scoreboard(tmp_path, [result])
    rows = list(csv.DictReader((tmp_path / "SCOREBOARD.csv").read_text().splitlines()))
    assert len(rows) == 120
    assert all(row["method"] == "noiseless_adam" for row in rows if row["panel"] == "historical_unmatched")
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with patch.object(trajectory_mixture.MixtureSampler, "ensemble", side_effect=AssertionError("cached")):
        resumed = experiment.run_case(tmp_path, circuit, theta, target, config())
    assert resumed == result
    assert before == {
        str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("method", ["weighted_adam", "adaptive_online"])
def test_stage_exact_resume_and_replayed_work(tmp_path: Path, method: str) -> None:
    """Discarded post-checkpoint work is charged and the complete final optimizer memory agrees."""
    circuit, theta, target = problem()
    uninterrupted = experiment.Store(tmp_path / "reference", target, config(), 71)
    expected = uninterrupted.train("stage", uninterrupted.initial(circuit, theta), 9, 13, method)
    store = experiment.Store(tmp_path / "resume", target, config(), 71)
    start = store.initial(circuit, theta)
    calls = 0
    gradient = trajectory_mixture.MixtureBatch.gradient
    adaptive = trajectory_mixture.MixtureBatch.adaptive_online_update

    def break_gradient(
        batch: trajectory_mixture.MixtureBatch, engine: StatevectorBackend, parameters: NDArray
    ) -> tuple[NDArray, float, float]:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise KeyboardInterrupt
        return gradient(batch, engine, parameters)

    def break_adaptive(
        batch: trajectory_mixture.MixtureBatch,
        engine: StatevectorBackend,
        parameters: NDArray,
        rate: float,
        moments: NDArray,
        counts: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return adaptive(batch, engine, parameters, rate, moments, counts)

    function, replacement = (
        ("gradient", break_gradient) if method == "weighted_adam" else ("adaptive_online_update", break_adaptive)
    )
    with patch.object(trajectory_mixture.MixtureBatch, function, new=replacement), pytest.raises(KeyboardInterrupt):
        store.train("stage", start, 9, 13, method)
    partial = pilot.read_record(tmp_path / "resume/stages/stage/PARTIAL.json")
    assert partial["completed"] == 2
    assert {"theta", "first", "second", "counts", "updates", "origins", "circuit_hash"} == set(
        cast("dict", partial["state"])
    )
    actual = store.train("stage", start, 9, 13, method)
    assert actual["state"] == expected["state"]
    assert actual["trace"] == expected["trace"]
    assert actual["completed_updates"] == 10
    assert actual["attempted_updates"] == 11
    assert actual["replayed_completed_updates"] == 1
    assert actual["attempted_trajectory_paths"] == actual["completed_trajectory_paths"] + 4
    assert actual["cpu_seconds"] > 0
    attempts = pilot.read_record(tmp_path / "resume/stages/stage/ATTEMPTS.json")
    assert len(cast("list", attempts["attempts"])) == 2
    with patch.object(trajectory_mixture.MixtureSampler, "ensemble", side_effect=AssertionError("cached")):
        assert store.train("stage", start, 9, 13, method) == actual


def test_surviving_optimizer_memory_phase_counts_and_full_metadata(tmp_path: Path) -> None:
    """Deletion compacts by original parameter order without resetting age or online phase."""
    circuit, theta, target = problem()
    store = experiment.Store(tmp_path, target, config(), 1)
    assert pilot.digest(pilot.circuit_record(store.restore(store.circuit(circuit)))) == pilot.digest(
        pilot.circuit_record(circuit)
    )
    state = store.initial(circuit, theta)
    state.update(first=list(np.arange(6) + 0.1), second=list(np.arange(6) + 0.2), counts=list(range(6)), updates=19)
    original = copy.deepcopy(state)
    removed = store.deleted(state, 3)
    assert state == original
    assert removed["updates"] == 19
    assert removed["origins"] == [0, 1, 2, 4, 5]
    for key in ("theta", "first", "second", "counts"):
        assert removed[key] == [value for index, value in enumerate(original[key]) if index != 3]
    assert [gate.native_gate_id for gate in store.restore(removed["circuit_hash"]).gates] == [
        f"native{index}" for index in (0, 1, 2, 4, 5)
    ]
    fresh = store.train("phase", store.initial(circuit, theta), 6, 0, "adaptive_online")
    assert [row["phase"] for row in fresh["trace"]] == ["online", "online", *["adaptive_online"] * 4]
    assert fresh["state"]["counts"] == [4] * 6
    assert fresh["state"]["first"] == [0] * 6


def test_parameter_permutation_and_sparse_logical_ids_do_not_change_physical_origins(tmp_path: Path) -> None:
    """Real starts have reordered parameters and sparse construction IDs; moment compaction uses parameter indices."""
    source, theta, target = problem()
    permutation = [2, 0, 1, 5, 3, 4]
    identifiers = [2, 0, 1, 105, 103, 104]
    for gate, parameter, logical_id in zip(source.gates, permutation, identifiers, strict=True):
        gate.param_index, gate.logical_gate_id = parameter, logical_id
    circuit = ParameterizedCircuit(2, source.gates)
    store = experiment.Store(tmp_path, target, config(), 1)
    state = store.initial(circuit, theta)
    state.update(first=[0.1, 1.1, 2.1, 3.1, 4.1, 5.1], second=[1, 2, 3, 4, 5, 6], counts=[2, 3, 4, 5, 6, 7], updates=41)
    assert state["origins"] == list(range(6))
    result = store.deleted(state, 3)
    assert result["origins"] == [0, 1, 2, 4, 5]
    for key in ("theta", "first", "second", "counts"):
        assert result[key] == state[key][:5]
    assert result["updates"] == 41
    retained = store.restore(result["circuit_hash"])
    assert [gate.logical_gate_id for gate in retained.gates] == [2, 0, 1, 103, 104]
    assert [gate.param_index for gate in retained.gates] == [2, 0, 1, 3, 4]


def test_fixed_and_structural_warmup_pairing_and_repair_epoch_policy(tmp_path: Path) -> None:
    """Both optimizers get the same physical maps; sibling repair epochs are paired and fresh."""
    circuit, theta, target = problem()
    store = experiment.Store(tmp_path, target, config(rounds=1), 83)
    initial = store.initial(circuit, theta)
    fixed = store.train("fixed", initial, 6, 0, "weighted_adam")
    warm = store.train("warm", initial, 2, 0, "weighted_adam")
    online = store.train("online", initial, 2, 0, "adaptive_online")
    assert fixed["trace"][:2] == warm["trace"]
    assert [row["ensemble_digest"] for row in warm["trace"]] == [row["ensemble_digest"] for row in online["trace"]]
    result = experiment.train_method(store, circuit, theta, "structural_mean2_weighted_adam")
    assert result["rounds"]
    row = result["rounds"][0]
    for branch in row["branches"]:
        stage = pilot.read_record(tmp_path / "stages" / branch["name"] / "TRAINING.json")
        assert [point["stream_index"] for point in cast("list[dict[str, Any]]", stage["trace"])] == [2, 3]
        assert cast("dict", stage["state"])["updates"] == 4


def test_search_resume_reuses_completed_branches_and_exact_decisions(tmp_path: Path) -> None:
    """An interruption between candidate repairs cannot restart moments or select another endpoint."""
    circuit, theta, target = problem()
    label = "structural_mean2_adaptive_online"
    expected = experiment.train_method(
        experiment.Store(tmp_path / "reference", target, config(), 9), circuit, theta, label
    )
    store = experiment.Store(tmp_path / "resume", target, config(), 9)
    train = store.train

    def interrupted(name: str, state: dict[str, Any], cap: int, epoch: int, method: str) -> dict[str, Any]:
        if "_delete" in name:
            raise KeyboardInterrupt
        return train(name, state, cap, epoch, method)

    with patch.object(store, "train", side_effect=interrupted), pytest.raises(KeyboardInterrupt):
        experiment.train_method(store, circuit, theta, label)
    retained = tmp_path / f"resume/branches/{label}_round0_retain.json"
    before = retained.read_bytes()
    actual = experiment.train_method(store, circuit, theta, label)
    assert retained.read_bytes() == before
    assert actual["state"] == expected["state"]
    assert actual["nominal_charged_updates"] == expected["nominal_charged_updates"]
    assert actual["completed_updates"] == expected["completed_updates"]
    assert [row["selected_branch"] for row in actual["rounds"]] == [
        row["selected_branch"] for row in expected["rounds"]
    ]


def test_failed_branch_is_charged_and_not_retried_on_resume(tmp_path: Path) -> None:
    """A numerical failure consumes its full repair allowance and retains actual attempted work."""
    circuit, theta, target = problem()
    store = experiment.Store(tmp_path, target, config(), 13)
    state = store.initial(circuit, theta)
    with patch.object(trajectory_mixture.MixtureBatch, "gradient", side_effect=FloatingPointError("engineered")):
        branch, ledger = experiment.repair_branch(store, state, "candidate", 2, "weighted_adam", 1, 1)
    assert branch["status"] == "failed"
    assert ledger[0]["reserved_updates"] == 2
    assert ledger[0]["attempted_updates"] == 1
    assert ledger[0]["completed_updates"] == 0
    with patch.object(store, "train", side_effect=AssertionError("must retain failure")):
        assert experiment.repair_branch(store, state, "candidate", 2, "weighted_adam", 1, 1) == (branch, ledger)


def test_failed_score_keeps_completed_repair_and_partial_density_work(tmp_path: Path) -> None:
    """A failed second profile is visible in the branch ledger and physical CPU."""
    circuit, theta, target = problem()
    store = experiment.Store(tmp_path, target, config(), 13)
    state = store.initial(circuit, theta)
    evaluate = channels.evaluate

    def fail_second(
        local_circuit: ParameterizedCircuit, parameters: NDArray, local_target: NDArray, noise: str
    ) -> float:
        if noise == "dephasing_1s_all":
            msg = "engineered profile failure"
            raise FloatingPointError(msg)
        return evaluate(local_circuit, parameters, local_target, noise)

    with patch.object(channels, "evaluate", side_effect=fail_second):
        branch, ledger = experiment.repair_branch(
            store, state, "structural_mean2_weighted_adam_failure", 2, "weighted_adam", 1, 1
        )
    assert branch["status"] == "failed"
    assert len(ledger) == 2
    assert ledger[0]["completed_updates"] == 2
    assert ledger[1]["physical_attempted_channel_calls"] == 2
    assert ledger[1]["physical_completed_channel_calls"] == 1
    physical = experiment.physical_work(tmp_path, "structural_mean2_weighted_adam")
    assert physical["training_cpu_seconds"] == pytest.approx(sum(row["cpu_seconds"] for row in ledger))
    assert physical["completed_density_channel_calls"] == 1


def test_failed_arm_retains_remaining_controls_and_physical_failure(tmp_path: Path) -> None:
    """One failed weighted call does not remove the other three baseline arms."""
    circuit, theta, target = problem()
    original = trajectory_mixture.MixtureBatch.gradient
    calls = 0

    def fail_once(
        batch: trajectory_mixture.MixtureBatch, engine: StatevectorBackend, parameters: NDArray
    ) -> tuple[NDArray, float, float]:
        nonlocal calls
        calls += 1
        if calls == 1:
            msg = "engineered first arm failure"
            raise FloatingPointError(msg)
        return original(batch, engine, parameters)

    with patch.object(trajectory_mixture.MixtureBatch, "gradient", new=fail_once):
        result = experiment.run_case(tmp_path, circuit, theta, target, config())
    assert result["status"] == "complete_with_failed_arms"
    assert result["common_cpu"]["status"] == "unavailable_incomplete_panel"
    assert result["arms"][experiment.METHODS[0]]["physical_work"]["attempted_updates"] == 1
    assert result["arms"][experiment.METHODS[0]]["physical_work"]["completed_updates"] == 0
    assert all(result["arms"][label]["status"] == "complete" for label in experiment.METHODS[1:])


def test_killed_marker_and_tampered_stage_are_rejected(tmp_path: Path) -> None:
    """An unobserved death cannot be reported as zero-cost resumed work."""
    circuit, theta, target = problem()
    store = experiment.Store(tmp_path, target, config(), 1)
    start = store.initial(circuit, theta)
    store.train("stage", start, 3, 0, "weighted_adam")
    path = tmp_path / "stages/stage/ATTEMPTS.json"
    record = cast("dict[str, Any]", pilot.read_record(path))
    record["attempts"][-1]["status"] = "running"
    pilot.write_record(path, record)
    with pytest.raises(ValueError, match="external resource ledger"):
        store.train("stage", start, 3, 0, "weighted_adam")
    record["attempts"][-1]["status"] = "complete"
    record["attempts"][-1]["cpu_seconds"] += 1
    pilot.write_record(path, record)
    with pytest.raises(ValueError, match="physical attempt ledger"):
        store.train("stage", start, 3, 0, "weighted_adam")


def test_common_cpu_uses_latest_available_not_largest_fidelity(tmp_path: Path) -> None:
    """CPU selection is independent of score and rescoring occurs outside training."""
    circuit, theta, target = problem()
    store = experiment.Store(tmp_path, target, config(), 1)
    state = store.initial(circuit, theta)
    points = [
        {"circuit_hash": state["circuit_hash"], "theta": (theta + shift).tolist(), "cpu_seconds": cpu}
        for shift, cpu in [(0.0, 0.0), (0.2, 1.0), (0.3, 2.0)]
    ]
    arms = {
        "slow": {"status": "complete", "cpu_seconds": 5.0, "curve": points},
        "fast": {"status": "complete", "cpu_seconds": 1.5, "curve": points[:2]},
    }
    result = experiment.common_cpu_panel(store, arms)
    assert result["cpu_seconds"] == pytest.approx(1.5)
    assert result["arms"]["slow"]["point"] == points[1]
    assert result["arms"]["fast"]["point"] == points[1]
    assert result["arms"]["slow"]["evaluation"]["individual_channel_calls"] == 12


@pytest.mark.parametrize(
    "field", ["total_updates", "warmup", "repair", "shortlist", "paths_per_channel", "protected", "learning_rate"]
)
def test_invalid_configuration(field: str) -> None:
    """Reject malformed scientific settings before creating records or samples."""
    values = asdict(config())
    values[field] = -1
    with pytest.raises(ValueError, match="budget or numerical"):
        experiment.Config(**values)


def test_near_ties_keep_memory_and_deletion_must_beat_retained_repair() -> None:
    """The original retain score enters the deletion threshold even when its improvement is sub-tolerance."""
    branches: list[dict[str, Any]] = [
        {"status": "complete", "fidelity": 0.80005},
        {"status": "complete", "fidelity": 0.80012},
    ]
    assert experiment.select_branch(branches, 0.8, 0.0001) is None
    branches[1]["fidelity"] = 0.8002
    assert experiment.select_branch(branches, 0.8, 0.0001) == 1
    branches[0] = {"status": "failed"}
    assert experiment.select_branch(branches, 0.8, 0.0001) == 1


def test_stage_cpu_includes_sampling_optimizer_and_checkpoints_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A controlled clock verifies the same checkpoint inclusion and final-ledger exclusion as reference stages."""
    circuit, theta, target = problem()
    store = experiment.Store(tmp_path, target, config(), 1)
    start = store.initial(circuit, theta)
    clock = [0.0]
    write = pilot.write_record
    gradient = trajectory_mixture.MixtureBatch.gradient

    def measured_gradient(
        batch: trajectory_mixture.MixtureBatch, engine: StatevectorBackend, parameters: NDArray
    ) -> tuple[NDArray, float, float]:
        clock[0] += 3
        return gradient(batch, engine, parameters)

    def measured_write(path: Path, record: dict[str, object]) -> None:
        write(path, record)
        clock[0] += 5 if path.name == "PARTIAL.json" else 7

    monkeypatch.setattr(experiment.time, "process_time", lambda: clock[0])
    with (
        patch.object(trajectory_mixture.MixtureBatch, "gradient", new=measured_gradient),
        patch.object(pilot, "write_record", side_effect=measured_write),
    ):
        result = store.train("clock", start, 4, 0, "weighted_adam")
    assert result["cpu_seconds"] == pytest.approx(22)
    assert [point["cpu_seconds"] for point in result["curve"]] == [0, 3, 6, 14, 17, 22]
    assert result["cpu_scope"] == experiment.CPU_SCOPE


def test_objective_matched_reference_circuit_bank_is_imported_read_only(tmp_path: Path) -> None:
    """The common-CPU panel includes all three independent exact reference methods."""
    circuit, theta, target = problem()
    root = tmp_path / "reference"
    bank = experiment.Store(root, target, config(), 1)
    key = bank.circuit(circuit)
    arm: dict[str, Any] = {
        "status": "complete",
        "circuit_hash": key,
        "theta": theta.tolist(),
        "cpu_seconds": 0.0,
        "curve": [{"circuit_hash": key, "theta": theta.tolist(), "cpu_seconds": 0.0}],
        "fidelities": {noise: channels.evaluate(circuit, theta, target, noise) for noise in channels.NOISE_IDS},
    }
    reference: dict[str, Any] = {"arms": {name: copy.deepcopy(arm) for name in experiment.reference.METHODS}}
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    result = experiment.run_case(
        tmp_path / "new", circuit, theta, target, config(), reference_case=reference, reference_root=root
    )
    assert result["status"] == "complete"
    assert set(result["arms"]) == {*experiment.METHODS, *experiment.reference.METHODS}
    assert result["common_cpu"]["cpu_seconds"] == pytest.approx(0)
    assert set(result["common_cpu"]["arms"]) == set(result["arms"])
    assert before == {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_prepare_only_and_changed_sources_or_embedded_starts_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Protocol creation is numerical-work-free and authenticates the complete declared original panel."""
    starts_root, reference_root = tmp_path / "starts", tmp_path / "reference"
    starts_root.mkdir()
    reference_root.mkdir()
    gates = [
        ParameterizedGate(
            "rx" if index < 78 else "rxx", (index % 6,) if index < 78 else (0, 1), index, logical_gate_id=index
        )
        for index in range(108)
    ]
    circuit = ParameterizedCircuit(6, gates)
    starts = [
        {
            "target_id": name,
            "replica": 11,
            "candidate": {"circuit": pilot.circuit_record(circuit), "theta": [0.1] * 108},
        }
        for name in experiment.TARGETS
    ]
    pilot.write_record(starts_root / "starts.json", {"starts": starts})
    ref: dict[str, Any] = {
        "version": experiment.reference.VERSION,
        "config": asdict(experiment.reference.Config()),
        "phase": "development",
        "noise_ids": list(experiment.ensemble_deletion.MEAN2),
        "starts": starts,
    }
    pilot.write_record(reference_root / "PROTOCOL.json", ref)
    monkeypatch.setattr(experiment.runner, "OUTPUT", starts_root)
    monkeypatch.setattr(experiment.runner, "custody", lambda: {"benchmark_hashes": {"fixed": "hash"}})
    monkeypatch.setattr(experiment, "numerical_sources", lambda: {"source": "hash"})
    with patch.object(experiment, "StatevectorBackend", side_effect=AssertionError("prepare only")):
        protocol = experiment.freeze_protocol(tmp_path / "protocol", reference_root)
    assert experiment.verify_protocol(protocol) == pilot.read_record(reference_root / "PROTOCOL.json")
    assert protocol["config"]["paths_per_channel"] == 32
    assert protocol["config"]["total_updates"] == 1500
    assert protocol["config"]["online_warmup"] == 25
    before = (tmp_path / "protocol/PROTOCOL.json").read_bytes()
    assert experiment.freeze_protocol(tmp_path / "protocol", reference_root) == protocol
    assert (tmp_path / "protocol/PROTOCOL.json").read_bytes() == before
    changed = copy.deepcopy(protocol)
    changed["starts"][0]["candidate"]["theta"][0] = 0.2
    with pytest.raises(ValueError, match="custody changed"):
        experiment.verify_protocol(changed)
    monkeypatch.setattr(experiment, "numerical_sources", lambda: {"source": "changed"})
    with pytest.raises(ValueError, match="custody changed"):
        experiment.verify_protocol(protocol)


def test_unknown_reference_cpu_blocks_common_panel(tmp_path: Path) -> None:
    """Unobserved process death cannot enter a numerical common-CPU comparison."""
    _, _, target = problem()
    arms = {
        "unknown": {"status": "complete", "physical_work": {"accounting_status": "unknown_requires_external_ledger"}}
    }
    result = experiment.common_cpu_panel(experiment.Store(tmp_path, target, config(), 1), arms)
    assert result["status"] == "unavailable_unknown_physical_cpu"
    assert result["arms"] == {}
