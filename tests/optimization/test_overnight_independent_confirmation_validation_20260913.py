# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Meaningful independent replay, complete-panel and durable-cost verifier tests."""

from __future__ import annotations

import copy
import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import channels
from experiments.overnight_state_preparation_20260913_0108 import independent_confirmation_validation as driver
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sealed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Provide synthetic sealed-API stand-ins; the independent oracle remains real.

    Returns:
        Parent root and mutable stand-ins for sealed records and saved input.
    """
    for name in driver.parent_api.THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(driver.parent_api.construction, "environment", lambda: {"test": "synthetic"})
    forecast = tmp_path / "forecast.json"
    pilot.write_record(forecast, {"scope": "synthetic test planning receipt"})
    monkeypatch.setattr(driver, "FORECAST_PATH", forecast)
    root = tmp_path / "parent"
    circuit = ParameterizedCircuit(
        2, [ParameterizedGate("ry", (0,), param_index=0), ParameterizedGate("rxx", (1, 0), param_index=1)]
    )
    theta = [0.4, -0.3]
    target = np.array([1, 1j, -0.4, 0.2j], dtype=np.complex128)
    target /= np.linalg.norm(target)
    spec = dict(pilot.circuit_record(circuit))
    chash = pilot.digest(spec)
    case = {
        "target_id": "synthetic",
        "replica": 11,
        "num_qubits": 2,
        "target_vector_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
    }
    parent: dict[str, Any] = {"phase": "engineering", "cases": [case], "source_hashes": {}}
    scores = {
        noise: channels.evaluate(circuit, np.asarray(theta), target, noise) for noise in driver.oracle.PROFILE_IDS
    }
    point = {"circuit_hash": chash, "theta": theta, "cpu_seconds": 1.0}
    record: dict[str, Any] = {
        "status": "complete",
        "panels": {
            panel: {
                "arms": {
                    name: {"point": copy.deepcopy(point), "scores": {"fidelities": dict(scores)}}
                    for name in driver.parent_api.METHODS
                }
            }
            for panel in driver.PANELS
        },
    }
    summary: dict[str, Any] = {
        "status": "complete",
        "cases": [{"target_id": "synthetic", "replica": 11, "checksum": pilot.digest(record)}],
    }
    child = root / "synthetic__11"
    pilot.write_record(root / "PROTOCOL.json", parent)
    pilot.write_record(root / "SUMMARY.json", summary)
    pilot.write_record(root / "DESIGN.json", {"test": True})
    pilot.write_record(child / "RESULT.json", record)
    pilot.write_record(child / "execution/COMPLETED.json", {"test": True})
    pairs = [[float(z.real), float(z.imag)] for z in target]
    loaded: dict[str, Any] = {"value": {"target": pairs}}
    pilot.write_record(child / "execution/input/RESULT.json", loaded)
    pilot.write_record(child / f"execution/training/work/circuits/{chash}.json", spec)
    monkeypatch.setattr(driver.parent_api, "verify_parent", lambda *_a: None)
    monkeypatch.setattr(driver.parent_api, "read_case", lambda *_a: copy.deepcopy(record))
    monkeypatch.setattr(driver.parent_api, "read_phase", lambda *_a: copy.deepcopy(loaded))
    monkeypatch.setattr(driver.parent_api, "load_target", lambda *_a: pytest.fail("target loader called"))
    monkeypatch.setattr(channels, "evaluate", lambda *_a: pytest.fail("YAQS score called during independent audit"))
    return root, record, loaded


def test_real_oracle_all_rows_dedup_and_byte_stable_cache(
    tmp_path: Path, sealed: tuple[Path, dict[str, Any], dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """All 252 synthetic role/profile rows reuse only twelve identical real oracle calls."""
    parent, _, _ = sealed
    root = tmp_path / "audit"
    protocol = driver.prepare(root, parent, engineering=True)
    assert protocol["role_count"] == 21
    assert protocol["row_count"] == 252
    assert protocol["unique_calls"] == 12
    assert not (root / "endpoints").exists()
    result = driver.run(root, protocol)
    assert result["status"] == "complete"
    assert result["accepted"]
    assert len(result["rows"]) == 252
    assert max(row["absolute_error"] for row in result["rows"]) < 1e-12
    assert result["costs"]["cpu_seconds"] == pytest.approx(
        protocol["preparation_costs"]["cpu_seconds"] + result["run_costs"]["cpu_seconds"]
    )
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(driver.oracle, "evaluate_profile", lambda *_a: pytest.fail("cached oracle repeated"))
    assert driver.run(root, protocol) == result
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize("changed", ["method", "profile", "vector", "case"])
def test_incomplete_or_changed_parent_aborts_before_any_score(
    tmp_path: Path, sealed: tuple[Path, dict[str, Any], dict[str, Any]], monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    """The original complete role product and saved vector bytes cannot be silently reduced."""
    parent, record, loaded = sealed
    if changed == "method":
        record["panels"]["primary"]["arms"].pop(driver.parent_api.METHODS[0])
    elif changed == "profile":
        record["panels"]["primary"]["arms"][driver.parent_api.METHODS[0]]["scores"]["fidelities"].pop("noiseless")
    elif changed == "vector":
        loaded["value"]["target"][0][0] += 0.1
    else:
        value = driver.read(parent / "PROTOCOL.json")
        value["cases"] *= 2
        pilot.write_record(parent / "PROTOCOL.json", value)
    summary = driver.read(parent / "SUMMARY.json")
    summary["cases"][0]["checksum"] = pilot.digest(record)
    pilot.write_record(parent / "SUMMARY.json", summary)
    monkeypatch.setattr(driver.oracle, "evaluate_profile", lambda *_a: pytest.fail("incomplete panel reached oracle"))
    with pytest.raises(ValueError, match=r"required|differ|duplicated|Every role"):
        driver.prepare(tmp_path / "audit", parent, engineering=True)
    assert not (tmp_path / "audit/PROTOCOL.json").exists()


def test_disagreement_and_bad_density_remain_full_failed_gate(
    tmp_path: Path, sealed: tuple[Path, dict[str, Any], dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A numerical mismatch or nonphysical density never drops its repeated comparison rows."""
    parent, _, _ = sealed
    root = tmp_path / "audit"
    protocol = driver.prepare(root, parent, engineering=True)
    original = driver.oracle.evaluate_profile

    def wrong(spec: dict[str, Any], theta: list[float], target: np.ndarray, noise: str) -> dict[str, Any]:
        value = original(spec, theta, target, noise)
        if noise == "noiseless":
            value["fidelity"] += 1e-4
        if noise == "ballarin_coupled":
            value["physicality"]["minimum_eigenvalue"] = -1e-4
        return value

    monkeypatch.setattr(driver.oracle, "evaluate_profile", wrong)
    result = driver.run(root, protocol)
    assert result["status"] == "complete"
    assert not result["accepted"]
    assert len(result["rows"]) == 252
    assert sum(not row["accepted"] for row in result["rows"]) == 42


def test_caught_interruption_reuses_calls_and_retains_inclusive_costs(
    tmp_path: Path, sealed: tuple[Path, dict[str, Any], dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One durable completed profile survives an interruption; all caught work stays paid."""
    parent, _, _ = sealed
    root = tmp_path / "audit"
    protocol = driver.prepare(root, parent, engineering=True)
    original = driver.oracle.evaluate_profile
    calls = []

    def interrupted(spec: dict[str, Any], theta: list[float], target: np.ndarray, noise: str) -> dict[str, Any]:
        calls.append(noise)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return original(spec, theta, target, noise)

    monkeypatch.setattr(driver.oracle, "evaluate_profile", interrupted)
    with pytest.raises(KeyboardInterrupt):
        driver.run(root, protocol)
    result = driver.run(root, protocol)
    assert result["accepted"]
    assert calls.count("noiseless") == 1
    assert len(calls) == 13
    attempts = driver.read(root / "RUN_ATTEMPTS.json")
    assert [a["status"] for a in attempts["attempts"]] == ["interrupted", "complete"]
    assert result["run_costs"] == driver.costs(attempts)


def test_terminal_oracle_failure_is_paid_and_never_retried(
    tmp_path: Path, sealed: tuple[Path, dict[str, Any], dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An oracle exception remains a failed profile in every role while other calls continue."""
    parent, _, _ = sealed
    root = tmp_path / "audit"
    protocol = driver.prepare(root, parent, engineering=True)
    original = driver.oracle.evaluate_profile

    def failed(spec: dict[str, Any], theta: list[float], target: np.ndarray, noise: str) -> dict[str, Any]:
        if noise == "noiseless":
            msg = "fixture failure"
            raise FloatingPointError(msg)
        return original(spec, theta, target, noise)

    monkeypatch.setattr(driver.oracle, "evaluate_profile", failed)
    result = driver.run(root, protocol)
    assert not result["accepted"]
    assert result["unique_calls_failed"] == 1
    assert sum(row["status"] == "failed" for row in result["rows"]) == 21
    assert driver.run(root, protocol) == result


def test_admission_cap_stops_new_calls_without_shrinking_rows(
    tmp_path: Path, sealed: tuple[Path, dict[str, Any], dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One admitted slow call may overrun; no later profile gets free work or vanishes."""
    parent, _, _ = sealed
    root = tmp_path / "audit"
    protocol = driver.prepare(root, parent, engineering=True)
    clock = [0.0]
    monkeypatch.setattr(driver.time, "process_time", lambda: clock[0])
    original = driver.oracle.evaluate_profile

    def slow(spec: dict[str, Any], theta: list[float], target: np.ndarray, noise: str) -> dict[str, Any]:
        value = original(spec, theta, target, noise)
        clock[0] += 601
        return value

    monkeypatch.setattr(driver.oracle, "evaluate_profile", slow)
    result = driver.run(root, protocol)
    assert result["status"] == "incomplete_budget"
    assert not result["accepted"]
    assert result["unique_calls_terminal"] == 1
    assert len(result["rows"]) == 252
    assert sum(row["status"] == "missing" for row in result["rows"]) == 231
    assert result["cpu_overrun_seconds"] >= 1


def test_unknown_kill_and_resigned_public_or_nested_tamper_fail_closed(
    tmp_path: Path, sealed: tuple[Path, dict[str, Any], dict[str, Any]]
) -> None:
    """A running marker cannot become free; public re-signing cannot change raw evidence."""
    parent, _, _ = sealed
    root = tmp_path / "killed"
    protocol = driver.prepare(root, parent, engineering=True)
    driver.Attempt(root / "RUN_ATTEMPTS.json", [driver.VERSION, pilot.digest(protocol)])
    with pytest.raises(ValueError, match="hard-kill"):
        driver.run(root, protocol)
    other = tmp_path / "complete"
    protocol = driver.prepare(other, parent, engineering=True)
    result = driver.run(other, protocol)
    result["rows"][0]["independent_fidelity"] += 0.1
    driver.seal(other, result)
    with pytest.raises(ValueError, match="rows differ"):
        driver.run(other, protocol)
    pilot.write_record(other / "extra/RESULT.json", {"unknown": True})
    with pytest.raises(ValueError, match="inventory"):
        driver.run(other, protocol)


@pytest.mark.parametrize("field", ["expected", "role", "source"])
def test_resigned_frozen_input_cannot_redefine_sealed_parent(
    tmp_path: Path, sealed: tuple[Path, dict[str, Any], dict[str, Any]], monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Re-signing both new input and protocol cannot rewrite independently sealed expectations."""
    parent, _, _ = sealed
    root = tmp_path / "audit"
    protocol = driver.prepare(root, parent, engineering=True)
    inputs = driver.read(root / "INPUTS.json")
    if field == "expected":
        inputs["roles"][0]["expected_fidelities"]["noiseless"] += 0.01
    elif field == "role":
        inputs["roles"].pop()
        protocol["role_count"] -= 1
        protocol["row_count"] -= 12
    else:
        protocol["source_hashes"] = {}
    pilot.write_record(root / "INPUTS.json", inputs)
    protocol["inputs_checksum"] = pilot.digest(inputs)
    protocol["input_file_sha256"] = driver.file_hash(root / "INPUTS.json")
    pilot.write_record(root / "PROTOCOL.json", protocol)
    monkeypatch.setattr(driver.oracle, "evaluate_profile", lambda *_a: pytest.fail("changed frozen rows scored"))
    with pytest.raises(ValueError, match=r"sealed parent|inventory"):
        driver.run(root, protocol)


def test_endpoint_cost_tamper_and_unknown_status_are_rejected(
    tmp_path: Path, sealed: tuple[Path, dict[str, Any], dict[str, Any]]
) -> None:
    """Nested endpoint annotations cannot rewrite physical cost after outer re-signing."""
    parent, _, _ = sealed
    root = tmp_path / "audit"
    protocol = driver.prepare(root, parent, engineering=True)
    result = driver.run(root, protocol)
    endpoint = next((root / "endpoints").iterdir())
    value = driver.read(endpoint / "RESULT.json")
    value["costs"]["cpu_seconds"] += 1
    driver.seal(endpoint, value)
    driver.seal(root, result)
    with pytest.raises(ValueError, match="Endpoint public costs"):
        driver.run(root, protocol)
    with pytest.raises(ValueError, match="status"):
        driver.costs({"attempts": [{"status": "imagined", "cpu_seconds": 0, "wall_seconds": 0}]})


@pytest.mark.parametrize("field", ["noise_id", "version", "status"])
def test_resigned_raw_call_cannot_change_oracle_identity(
    tmp_path: Path, sealed: tuple[Path, dict[str, Any], dict[str, Any]], field: str
) -> None:
    """Even freshly sealed raw/public call copies must match their requested identity."""
    parent, _, _ = sealed
    root = tmp_path / "audit"
    protocol = driver.prepare(root, parent, engineering=True)
    result = driver.run(root, protocol)
    endpoint = next((root / "endpoints").iterdir())
    call = endpoint / "calls/noiseless"
    raw = driver.read(call / "MEASUREMENT.json")
    if field == "status":
        raw["status"] = "invented"
    else:
        raw["oracle"][field] = "invented"
    pilot.write_record(call / "MEASUREMENT.json", raw)
    public = driver.read(call / "RESULT.json")
    public.update(raw)
    driver.seal(call, public)
    driver.seal(endpoint, driver.read(endpoint / "RESULT.json"))
    driver.seal(root, result)
    with pytest.raises(ValueError, match=r"oracle|attempt status"):
        driver.run(root, protocol)


def test_calibration_and_cached_run_identity_are_not_free_annotations(
    tmp_path: Path, sealed: tuple[Path, dict[str, Any], dict[str, Any]]
) -> None:
    """Public calibration rates and invocation identities remain bound after completion."""
    parent, _, _ = sealed
    root = tmp_path / "audit"
    protocol = driver.prepare(root, parent, engineering=True)
    edited = copy.deepcopy(protocol)
    edited["calibrations"] = {}
    pilot.write_record(root / "PROTOCOL.json", edited)
    with pytest.raises(ValueError, match="calibration"):
        driver.run(root, edited)
    pilot.write_record(root / "PROTOCOL.json", protocol)
    other = tmp_path / "complete"
    protocol = driver.prepare(other, parent, engineering=True)
    result = driver.run(other, protocol)
    attempts = driver.read(other / "RUN_ATTEMPTS.json")
    attempts["signature"] = "changed"
    pilot.write_record(other / "RUN_ATTEMPTS.json", attempts)
    driver.seal(other, result)
    with pytest.raises(ValueError, match="Run attempt identity"):
        driver.run(other, protocol)
