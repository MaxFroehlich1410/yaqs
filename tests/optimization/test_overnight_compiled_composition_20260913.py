# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test compiler opportunity, chronological availability and nested scientific custody."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import (
    compiled_composition as kernel,
)
from experiments.overnight_state_preparation_20260913_0108 import (
    compiled_composition_experiment as driver,
)
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def problem() -> tuple[dict[str, Any], np.ndarray]:
    """Return a complete scalar initializer with an initial phase and nontrivial q2 target."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("rz", (0,), 4, native_gate_id=91, logical_gate_id=3),
            ParameterizedGate("ry", (0,), 0, native_gate_id=92, logical_gate_id=8),
            ParameterizedGate("rx", (1,), 1, native_gate_id=93, logical_gate_id=11),
            ParameterizedGate("rxx", (1, 0), 2, native_gate_id=94, logical_gate_id=7),
            ParameterizedGate("rz", (0,), 3, native_gate_id=95, logical_gate_id=14),
            ParameterizedGate("ry", (1,), 5, native_gate_id=96, logical_gate_id=2),
        ],
    )
    generic = kernel.base.initializer_policy.compile_source({
        "circuit": pilot.circuit_record(circuit),
        "theta": [0.3, 0.2, 0.7, -0.1, 0.8, -0.3],
    })
    source = {"circuit": generic["circuit"], "theta": generic["theta"], "initializer_cpu_seconds": 7.0}
    target = np.array([0.2j, 0.6, -0.3 + 0.1j, 0.4])
    return source, target / np.linalg.norm(target)


def configs() -> tuple[kernel.base.exact.Config, kernel.base.stochastic.Config]:
    """Return tiny real configurations retaining symmetric branch opportunities."""
    return (
        kernel.base.exact.Config(
            channel_evaluations=20,
            warmup_evaluations=1,
            max_rounds=1,
            shortlist=1,
            repair_evaluations=1,
            protected_initial_gates=0,
            checkpoint_every=1,
        ),
        kernel.base.stochastic.Config(
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


@pytest.fixture
def complete(tmp_path: Path) -> tuple[Path, dict[str, Any], np.ndarray, dict[str, Any]]:
    """Return a completed small seven-role physical experiment."""
    source, target = problem()
    result = kernel.execute(tmp_path, source, target, *configs(), 42)
    return tmp_path, source, target, result


def test_all_roles_share_compiled_start_and_receive_both_final_compilers(
    complete: tuple[Path, dict[str, Any], np.ndarray, dict[str, Any]],
) -> None:
    """Initial compilation is shared physically and charged exactly once to every arm clock."""
    root, source, target, result = complete
    assert result["status"] == "complete"
    assert set(result["arms"]) == set(kernel.METHODS)
    assert result["initial"]["removed_gates"] > 0
    initial_cpu = result["initial"]["cpu_seconds"]
    assert initial_cpu > 0
    for method, compact in result["arms"].items():
        assert "curve" not in compact
        arm = kernel.read(root / compact["arm_file"])
        assert compact["arm_checksum"] == pilot.digest(arm)
        assert arm["curve"][0]["theta"] == source["theta"]
        assert arm["curve"][0]["cpu_seconds"] == 0
        assert arm["curve"][1] == result["initial"]
        assert arm["curve"][-2]["stage"] == "generic_final"
        assert arm["curve"][-1]["stage"] == "state_preparation_final"
        assert set(arm["scores"]["fidelities"]) == set(kernel.base.channels.NOISE_IDS)
        assert arm["refinement_cpu_seconds"] == pytest.approx(
            initial_cpu
            + arm["training_cpu_seconds"]
            + arm["generic_final_compiler_cpu_seconds"]
            + arm["state_final_compiler_cpu_seconds"],
            abs=1e-14,
        )
        assert arm["total_cpu_seconds"] == pytest.approx(7 + arm["refinement_cpu_seconds"])
        if method != "unchanged":
            receipt = kernel.read(root / arm["training_receipt"])
            assert "curve" not in receipt["training"]
            assert "training" not in arm
    for panel in ("common_primary", "common_all_trained"):
        value = result[panel]
        assert value["status"] == "complete"
        assert set(value["arms"]) == set(kernel.METHODS)
        assert all(p["point"]["cpu_seconds"] <= value["refinement_cpu_seconds"] for p in value["arms"].values())
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    assert kernel.execute(root, source, target, *configs(), 42) == result
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize("field", ["theta", "cost", "score", "selection"])
def test_resigned_compact_execution_cannot_change_child_results(
    complete: tuple[Path, dict[str, Any], np.ndarray, dict[str, Any]],
    field: str,
) -> None:
    """A matching signature and newly valid JSON checksum cannot authorize changed public values."""
    root, source, target, result = complete
    changed = copy.deepcopy(result)
    arm = changed["arms"][kernel.CANDIDATE]
    if field == "theta":
        arm["final"]["theta"][0] += 0.1
    elif field == "cost":
        arm["total_cpu_seconds"] += 1
    elif field == "score":
        arm["scores"]["fidelities"]["noiseless"] = 0
    else:
        changed["common_primary"]["arms"][kernel.CANDIDATE]["point"]["theta"][0] += 0.1
    pilot.write_record(root / "EXECUTION.json", changed)
    with pytest.raises(ValueError, match=r"compact arm|selected points"):
        kernel.execute(root, source, target, *configs(), 42)


@pytest.mark.parametrize("field", ["theta", "cost", "score", "curve"])
def test_resigned_full_arm_is_bound_to_training_compiler_and_score_receipts(
    complete: tuple[Path, dict[str, Any], np.ndarray, dict[str, Any]],
    field: str,
) -> None:
    """Even an updated outer artifact inventory cannot bless a modified arm-level scientific result."""
    root, source, target, result = complete
    path = root / result["arms"][kernel.CANDIDATE]["arm_file"]
    arm = kernel.read(path)
    if field == "theta":
        arm["final"]["theta"][0] += 0.1
    elif field == "cost":
        arm["training_cpu_seconds"] += 1
    elif field == "score":
        arm["scores"]["fidelities"]["noiseless"] = 0
    else:
        arm["curve"][2]["cpu_seconds"] += 0.1
    pilot.write_record(path, arm)
    result["dependencies"] = kernel.artifacts(root)
    result["arms"][kernel.CANDIDATE] = kernel.compact_arm(arm, path, root)
    pilot.write_record(root / "EXECUTION.json", result)
    with pytest.raises(ValueError, match=r"theta, curve|arm score"):
        kernel.execute(root, source, target, *configs(), 42)


def test_same_primary_cutoff_retains_raw_and_short_finished_controls(
    complete: tuple[Path, dict[str, Any], np.ndarray, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every control shares the structural cutoff; no later compiler output is borrowed early."""
    root, _, target, _ = complete
    store = kernel.base.CompleteStore(root, target, configs()[0])
    monkeypatch.setattr(kernel.base, "score", lambda _store, point: {"selected": point["theta"]})
    arms: dict[str, Any] = {
        method: {
            "status": "complete",
            "refinement_cpu_seconds": 5,
            "curve": [
                {"theta": [0], "cpu_seconds": 0},
                {"theta": [1], "cpu_seconds": 2},
                {"theta": [2], "cpu_seconds": 5},
            ],
        }
        for method in kernel.METHODS
    }
    arms[kernel.PRIMARY[1]]["refinement_cpu_seconds"] = 3
    arms[kernel.PRIMARY[1]]["curve"][-1]["cpu_seconds"] = 3
    arms["fixed_mean2_lbfgs"]["refinement_cpu_seconds"] = 1
    arms["fixed_mean2_lbfgs"]["curve"] = [{"theta": [3], "cpu_seconds": 1}]
    value = kernel.common_panel(store, arms, kernel.PRIMARY, 7)
    assert value["refinement_cpu_seconds"] == 3
    assert value["total_cpu_seconds"] == 10
    assert value["arms"][kernel.CANDIDATE]["scores"]["selected"] == [1]
    assert value["arms"]["fixed_mean2_lbfgs"]["scores"]["selected"] == [3]


def test_caught_case_interruption_retains_prior_arms_and_unknown_kill_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume reuses completed work and keeps every measured failed invocation."""
    source, target = problem()
    original = kernel.train_arm
    calls = 0

    def interrupted(*args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(kernel, "train_arm", interrupted)
    with pytest.raises(KeyboardInterrupt):
        kernel.execute(tmp_path, source, target, *configs(), 42)
    first = tmp_path / "arms" / f"{kernel.TRAINED[0]}.json"
    before = first.read_bytes()
    monkeypatch.setattr(kernel, "train_arm", original)
    result = kernel.execute(tmp_path, source, target, *configs(), 42)
    assert result["status"] == "complete"
    assert first.read_bytes() == before
    attempts = kernel.read(tmp_path / "CASE_ATTEMPTS.json")["attempts"]
    assert [a["status"] for a in attempts] == ["interrupted", "complete"]
    assert result["physical_invocation"]["cpu_seconds"] == sum(a["cpu_seconds"] for a in attempts)


def test_numerical_failure_stays_visible_and_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed role never becomes a smaller accepted panel and is not silently retried."""
    source, target = problem()
    original = kernel.train_arm

    def fail_one(*args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        if args[-1] == kernel.CANDIDATE:
            msg = "injected numerical failure"
            raise FloatingPointError(msg)
        return original(*args, **kwargs)

    monkeypatch.setattr(kernel, "train_arm", fail_one)
    result = kernel.execute(tmp_path, source, target, *configs(), 42)
    assert result["status"] == "partial_failure"
    assert result["common_primary"]["status"] == "unavailable_failed_arm"
    monkeypatch.setattr(kernel, "train_arm", lambda *_a, **_k: pytest.fail("failed role retried"))
    assert kernel.execute(tmp_path, source, target, *configs(), 42) == result


def test_promotion_requires_strongest_control_and_complete_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Beating fixed B* is insufficient if another aggregate control is stronger."""
    targets = ("a", "b", "c", "d")
    monkeypatch.setattr(driver, "TARGETS", targets)
    records: list[dict[str, Any]] = []
    for target in targets:
        scores = {
            name: {"scores": {"fidelities": dict.fromkeys(kernel.base.channels.NOISE_IDS, 0.9)}}
            for name in kernel.METHODS
        }
        for noise in kernel.base.channels.NOISE_IDS:
            scores[kernel.BASELINE]["scores"]["fidelities"][noise] = 0.89
            scores["fixed_mean2_lbfgs"]["scores"]["fidelities"][noise] = 0.91
        records.append({
            "target_id": target,
            "status": "complete",
            "common_primary": {"status": "complete", "arms": scores},
        })
    result = driver.analyze(records)
    assert not result["promoted"]
    assert result["strongest_control"] == "fixed_mean2_lbfgs"
    assert driver.analyze(records[:-1])["status"] == "incomplete"


def test_prepare_is_read_only_and_binds_source_arrays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Protocol reconstruction rejects edited sources and never invokes the initializer/compiler during prepare."""
    source, _ = problem()
    starts = [{"target_id": target, "replica": 11, "endpoint": source} for target in driver.TARGETS]
    monkeypatch.setattr(kernel.base, "input_panel", lambda _root: (starts, {"source": "hash"}, {"environment": {}}))
    monkeypatch.setattr(kernel.compiler, "compile_endpoint", lambda *_a: pytest.fail("prepare ran compiler"))
    monkeypatch.setattr(driver, "load_target", lambda *_a: pytest.fail("prepare loaded target"))
    protocol = driver.prepare(tmp_path, tmp_path / "source")
    driver.verify(protocol)
    changed = copy.deepcopy(protocol)
    changed["starts"][0]["endpoint"]["theta"][0] += 0.1
    with pytest.raises(ValueError, match="custody changed"):
        driver.verify(changed)
    changed = copy.deepcopy(protocol)
    changed["input_files"] = {}
    with pytest.raises(ValueError, match="custody changed"):
        driver.verify(changed)
    assert driver.prepare(tmp_path, tmp_path / "source") == protocol


def test_omitted_artifact_and_unknown_case_kill_fail_closed(
    complete: tuple[Path, dict[str, Any], np.ndarray, dict[str, Any]],
) -> None:
    """A top-level signature cannot omit a changed stage or hide an extra unfinished invocation."""
    root, source, target, result = complete
    result["dependencies"].pop(next(iter(result["dependencies"])))
    pilot.write_record(root / "EXECUTION.json", result)
    with pytest.raises(ValueError, match="complete artifact index"):
        kernel.execute(root, source, target, *configs(), 42)
    attempts = kernel.read(root / "CASE_ATTEMPTS.json")
    attempts["attempts"].append({"status": "running", "cpu_seconds": None, "wall_seconds": None})
    pilot.write_record(root / "CASE_ATTEMPTS.json", attempts)
    result["dependencies"] = kernel.artifacts(root)
    pilot.write_record(root / "EXECUTION.json", result)
    with pytest.raises(ValueError, match="clocks differ"):
        kernel.execute(root, source, target, *configs(), 42)


def test_driver_retains_early_numeric_case_failure_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An initial compiler failure leaves a complete failed-case receipt and does not abort other cases."""
    source, _ = problem()
    entry = {"target_id": driver.TARGETS[0], "replica": 11, "endpoint": source}
    protocol: dict[str, Any] = {"starts": [entry]}
    job: dict[str, Any] = {"protocol": protocol, "source": entry, "root": str(tmp_path)}
    monkeypatch.setattr(driver, "verify", lambda _p: None)

    def failed(_root: Path, _protocol: dict[str, Any], _source: dict[str, Any]) -> dict[str, Any]:
        msg = "injected early numerical failure"
        raise FloatingPointError(msg)

    monkeypatch.setattr(driver, "run_kernel", failed)
    result = driver.run_case(job)
    assert result["status"] == "failed"
    monkeypatch.setattr(driver, "run_kernel", lambda *_a: pytest.fail("failed case retried"))
    assert driver.run_case(job) == result
    result["failure"] = "changed summary"
    pilot.write_record(tmp_path / "RESULT.json", result)
    with pytest.raises(ValueError, match="failed case"):
        driver.run_case(job)
