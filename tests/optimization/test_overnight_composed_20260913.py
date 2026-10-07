# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent q2 composition, resource-availability and artifact-custody checks."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import composed_experiment as experiment
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def input_case() -> tuple[dict[str, Any], np.ndarray]:
    """Return a complex synthetic target and a compiled scalar physical initializer."""
    gates = [
        ParameterizedGate("rx", (0,), 0, logical_gate_id=17, native_gate_id=91),
        ParameterizedGate("ry", (1,), 1, logical_gate_id=5, native_gate_id=92),
        ParameterizedGate("rxx", (1, 0), 2, logical_gate_id=88, native_gate_id=93),
        ParameterizedGate("rz", (0,), 3, logical_gate_id=2, native_gate_id=94),
    ]
    raw = {"circuit": pilot.circuit_record(ParameterizedCircuit(2, gates, 4)), "theta": [0.2, -0.3, 0.7, 0.4]}
    compiled = experiment.initializer_policy.compile_source(raw)
    source = {"circuit": compiled["circuit"], "theta": compiled["theta"], "initializer_cpu_seconds": 7.0}
    target = np.array([0.2j, 0.6, -0.3 + 0.1j, 0.4])
    return source, target / np.linalg.norm(target)


def small_configs() -> tuple[experiment.exact.Config, experiment.stochastic.Config]:
    """Return budgets funding one symmetric search round and polishing per optimizer."""
    return (
        experiment.exact.Config(
            channel_evaluations=20,
            warmup_evaluations=1,
            max_rounds=1,
            shortlist=1,
            repair_evaluations=1,
            protected_initial_gates=0,
            checkpoint_every=1,
        ),
        experiment.stochastic.Config(
            total_updates=8,
            warmup=1,
            rounds=1,
            shortlist=1,
            repair=1,
            paths_per_channel=2,
            protected=0,
            checkpoint_every=1,
            online_warmup=1,
        ),
    )


def test_complete_physical_metadata_survives_bank_reload(tmp_path: Path) -> None:
    """Native compiler IDs and ordered supports must survive exact branch restoration."""
    source, target = input_case()
    store = experiment.CompleteStore(tmp_path, target, small_configs()[0])
    restored = experiment.restore_physical(source["circuit"])
    assert pilot.circuit_record(restored) == source["circuit"]
    key = store.circuit(restored)
    assert pilot.circuit_record(store.restore(key)) == source["circuit"]
    record = pilot.read_record(tmp_path / "circuits" / f"{key}.json")
    record["num_qubits"] = 3
    pilot.write_record(tmp_path / "circuits" / f"{key}.json", record)
    with pytest.raises(ValueError, match="content identity"):
        store.restore(key)


def test_seven_arm_real_q2_integration_and_exact_cached_resume(tmp_path: Path) -> None:
    """Every optimizer gets the same source and all twelve exact endpoint conditions."""
    source, target = input_case()
    exact, stochastic = small_configs()
    result = experiment.execute(tmp_path, source, target, exact, stochastic, 8231)
    assert result["status"] == "complete"
    assert set(result["arms"]) == set(experiment.METHODS)
    for name, arm in result["arms"].items():
        assert set(arm["scores"]["fidelities"]) == set(experiment.channels.NOISE_IDS)
        assert arm["total_cpu_seconds"] == pytest.approx(7 + arm["refinement_cpu_seconds"])
        if name != "unchanged":
            assert arm["final"]["compiled_final"]
            assert arm["final"]["cpu_seconds"] >= arm["training_cpu_seconds"]
            assert arm["curve"][0]["theta"] == source["theta"]
            assert arm["dependencies"]
    for panel, names in (("common_primary", experiment.PRIMARY), ("common_all_trained", experiment.TRAINED)):
        assert result[panel]["status"] == "complete"
        assert set(result[panel]["arms"]) == set(names)
        assert result[panel]["total_cpu_seconds"] == pytest.approx(7 + result[panel]["refinement_cpu_seconds"])
        assert all(
            p["point"]["cpu_seconds"] <= result[panel]["refinement_cpu_seconds"] for p in result[panel]["arms"].values()
        )
    replay = experiment.execute(tmp_path, source, target, exact, stochastic, 8231)
    assert pilot.digest(replay) == pilot.digest(result)
    path = next((tmp_path / "stages").rglob("ATTEMPTS.json"))
    changed = pilot.read_record(path)
    changed["tampered"] = True
    pilot.write_record(path, changed)
    with pytest.raises(ValueError, match="arm identity"):
        experiment.execute(tmp_path, source, target, exact, stochastic, 8231)


def test_common_cpu_uses_last_available_raw_point_and_complete_initializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An excellent later compiled endpoint is not borrowed before compilation finishes."""
    source, target = input_case()
    store = experiment.CompleteStore(tmp_path, target, small_configs()[0])
    monkeypatch.setattr(experiment, "score", lambda _store, point: {"selected": point["theta"]})
    arms = {
        "a": {
            "status": "complete",
            "refinement_cpu_seconds": 4,
            "curve": [
                {"theta": [0.9], "cpu_seconds": 0},
                {"theta": [0.1], "cpu_seconds": 2},
                {"theta": [0.99], "cpu_seconds": 4, "compiled_final": True},
            ],
        },
        "b": {"status": "complete", "refinement_cpu_seconds": 3, "curve": [{"theta": [0.5], "cpu_seconds": 3}]},
    }
    result = experiment.common_panel(store, arms, ("a", "b"), source["initializer_cpu_seconds"])
    assert result["total_cpu_seconds"] == 10
    assert result["arms"]["a"]["scores"]["selected"] == [0.1]
    arms["b"]["status"] = "failed"
    assert experiment.common_panel(store, arms, ("a", "b"), 7)["status"] == "unavailable_failed_arm"


def test_compiler_cost_precedes_deployment_and_preserves_ideal_state(tmp_path: Path) -> None:
    """The final compiler audit preserves the prepared physical state and is charged."""
    source, target = input_case()
    store = experiment.CompleteStore(tmp_path, target, small_configs()[0])
    point = {
        "theta": source["theta"],
        "circuit_hash": store.circuit(experiment.restore_physical(source["circuit"])),
        "cpu_seconds": 5.0,
    }
    deployed = experiment.compile_final(store, point)
    assert deployed["compiler_cpu_seconds"] > 0
    assert deployed["cpu_seconds"] == 5 + deployed["compiler_cpu_seconds"]
    before = experiment.initializer_policy.state_vector(source["circuit"], np.asarray(source["theta"]))
    after = experiment.initializer_policy.state_vector(
        pilot.circuit_record(store.restore(deployed["circuit_hash"])), np.asarray(deployed["theta"])
    )
    assert abs(np.vdot(before, after)) ** 2 == pytest.approx(1, abs=1e-12)
    assert experiment.compile_final(store, point) == deployed


def test_caught_attempt_is_charged_on_retry_and_unknown_death_blocks(tmp_path: Path) -> None:
    """Lost work remains in cumulative physical CPU, and unknown death never becomes zero."""
    path = tmp_path / "operation.json"

    def interrupted() -> dict[str, Any]:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        experiment.tracked_call(path, ["frozen"], interrupted)
    attempts = cast("dict[str, Any]", pilot.read_record(tmp_path / "operation.attempts.json"))
    assert attempts["attempts"][0]["status"] == "interrupted"
    result = experiment.tracked_call(path, ["frozen"], lambda: {"value": 3})
    attempts = cast("dict[str, Any]", pilot.read_record(tmp_path / "operation.attempts.json"))
    assert result["cpu_seconds"] == sum(a["cpu_seconds"] for a in attempts["attempts"])
    assert len(attempts["attempts"]) == 2
    attempts["attempts"].append({"status": "running", "cpu_seconds": None})
    pilot.write_record(tmp_path / "operation.attempts.json", attempts)
    with pytest.raises(ValueError, match="unmeasured killed"):
        experiment.tracked_call(path, ["frozen"], lambda: {"value": 4})


@pytest.mark.parametrize("change", ["order", "protected", "too_many_deletions"])
def test_invalid_composition_opportunities_rejected(tmp_path: Path, change: str) -> None:
    """No arm omission, inherited protected subset, or empty scalar domain is allowed."""
    source, target = input_case()
    exact, stochastic = small_configs()
    order = experiment.TRAINED
    if change == "order":
        order = order[:-1]
    elif change == "protected":
        exact = experiment.exact.Config(protected_initial_gates=1)
    else:
        source = copy.deepcopy(source)
        source["circuit"]["gates"] = source["circuit"]["gates"][:1]
        source["circuit"]["num_params"] = 1
        source["theta"] = source["theta"][:1]
    with pytest.raises(ValueError, match=r"unique trained arm|empty protected set|retain at least"):
        experiment.execute(tmp_path, source, target, exact, stochastic, 3, order=order)


def test_numerical_arm_failure_remains_visible_without_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A retained failed arm invalidates its common panel without dropping the case."""
    source, target = input_case()
    original = experiment.train_deploy

    def fail_one(
        store: experiment.CompleteStore,
        sampled: experiment.stochastic.Store,
        source: dict[str, Any],
        method: str,
        signature: str,
        initial: dict[str, Any],
    ) -> dict[str, Any]:
        if method == experiment.PRIMARY[0]:
            msg = "injected failure"
            raise FloatingPointError(msg)
        return original(store, sampled, source, method, signature, initial)

    monkeypatch.setattr(experiment, "train_deploy", fail_one)
    result = experiment.execute(tmp_path, source, target, *small_configs(), 2)
    assert result["status"] == "partial_failure"
    assert len(result["arms"]) == 7
    assert result["common_primary"]["status"] == "unavailable_failed_arm"
    monkeypatch.setattr(experiment, "train_deploy", lambda *_a, **_k: pytest.fail("cached arm retried"))
    assert experiment.execute(tmp_path, source, target, *small_configs(), 2)["status"] == "partial_failure"


@pytest.fixture
def input_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a source-complete synthetic panel without constructing benchmark targets."""
    source, _ = input_case()
    root = tmp_path / "initializers"
    protocol: dict[str, Any] = {
        "starts": [{"target_id": t, "replica": 11} for t in experiment.TARGETS],
        "environment": {"fixed": True},
    }
    pilot.write_record(root / "PROTOCOL.json", protocol)
    checksums = {}
    for original in protocol["starts"]:
        case_root = root / f"{original['target_id']}__11"
        built: dict[str, Any] = {"conditional": source}
        pilot.write_record(case_root / "INITIALIZERS.json", built)
        case: dict[str, Any] = {
            "signature": pilot.digest({"protocol": protocol, "source": original}),
            "status": "complete",
            "initializer_checksum": pilot.digest(built),
            "dependencies": {},
        }
        pilot.write_record(case_root / "RESULT.json", case)
        checksums[original["target_id"]] = pilot.digest(case)
    pilot.write_record(
        root / "SUMMARY.json",
        {"status": "complete", "protocol_checksum": pilot.digest(protocol), "case_checksums": checksums},
    )
    monkeypatch.setattr(experiment.initializer_experiment, "verify", lambda _p: None)
    monkeypatch.setattr(experiment, "sources", lambda: {"code": "fixed"})
    monkeypatch.setattr(experiment.runner, "custody", lambda: {"benchmark_hashes": {"fixture": "fixed"}})
    monkeypatch.setattr(experiment.initializer_experiment.construction, "environment", lambda: {"fixed": True})
    monkeypatch.setattr(experiment.initializer_experiment, "run_case", lambda *_a: pytest.fail("prepare ran science"))
    monkeypatch.setattr(experiment, "load_target", lambda *_a: pytest.fail("prepare loaded target"))
    return root


@pytest.mark.parametrize("change", ["theta", "cost", "input_inventory", "method", "budget", "target"])
def test_protocol_binds_complete_input_endpoints_and_configuration(
    tmp_path: Path, input_panel: Path, change: str
) -> None:
    """Valid envelopes cannot substitute starts, omit source files or change numerical rules."""
    protocol = experiment.prepare(tmp_path / "prepared", input_panel)
    assert len(protocol["input_files"]) == 10
    experiment.verify(protocol)
    changed = copy.deepcopy(protocol)
    if change == "theta":
        changed["starts"][0]["endpoint"]["theta"][0] += 0.1
    elif change == "cost":
        changed["starts"][0]["endpoint"]["initializer_cpu_seconds"] = 0.0
    elif change == "input_inventory":
        changed["input_files"] = {}
    elif change == "method":
        changed["methods"].pop()
    elif change == "budget":
        changed["exact_config"]["repair_evaluations"] = 50
    else:
        changed["starts"].pop()
    with pytest.raises(ValueError, match="frozen inputs"):
        experiment.verify(changed)


def test_missing_or_changed_initializer_case_fails_without_execution(tmp_path: Path, input_panel: Path) -> None:
    """A stale complete summary cannot turn prepare-only into a scientific rerun."""
    case = input_panel / f"{experiment.TARGETS[0]}__11" / "RESULT.json"
    case.unlink()
    with pytest.raises(FileNotFoundError):
        experiment.prepare(tmp_path / "prepared", input_panel)
    assert not (tmp_path / "prepared" / "PROTOCOL.json").exists()


def test_case_invocation_overhead_records_failed_and_replayed_attempts(tmp_path: Path) -> None:
    """An inclusive clock is a separate physical scope, not added to nested numerical CPU."""

    def fail() -> dict[str, Any]:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        experiment.measured_execute(tmp_path, "frozen", fail)
    result = experiment.measured_execute(tmp_path, "frozen", lambda: {"status": "complete"})
    ledger = cast("dict[str, Any]", pilot.read_record(tmp_path / "CASE_ATTEMPTS.json"))
    assert len(ledger["attempts"]) == 2
    assert result["physical_invocation"]["cpu_seconds"] == sum(a["cpu_seconds"] for a in ledger["attempts"])
    assert result["physical_invocation"]["attempts_checksum"] == pilot.digest(ledger)
    ledger["attempts"].append({"status": "running"})
    pilot.write_record(tmp_path / "CASE_ATTEMPTS.json", ledger)
    with pytest.raises(ValueError, match="killed invocation"):
        experiment.measured_execute(tmp_path, "frozen", lambda: {"status": "complete"})
