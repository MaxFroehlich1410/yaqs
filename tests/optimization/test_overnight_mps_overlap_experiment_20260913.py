# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exercise paired weighted MPS replay, complete Adam resume and artifact custody."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import mps_overlap_experiment as experiment
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from collections.abc import Callable


def problem() -> tuple[dict[str, Any], np.ndarray]:
    """Return two distinct small physical circuits and a complex normalized target."""
    gates = [
        ParameterizedGate("ry", (0,), param_index=0),
        ParameterizedGate("rx", (1,), param_index=1),
        ParameterizedGate("rxx", (1, 0), param_index=2),
        ParameterizedGate("rz", (0,), param_index=3, angle_scale=-0.7, angle_offset=0.1),
    ]
    original = {"circuit": pilot.circuit_record(ParameterizedCircuit(2, gates)), "theta": [0.4, -0.9, 0.3, 0.6]}
    short = {"circuit": pilot.circuit_record(ParameterizedCircuit(2, gates[:3])), "theta": [0.4, -0.9, 0.3]}
    rng = np.random.default_rng(901)
    target = rng.normal(size=4) + 1j * rng.normal(size=4)
    return {
        "target_id": "synthetic_complex",
        "replica": 11,
        "endpoints": dict(zip(experiment.ENDPOINTS, (original, short), strict=True)),
    }, target / np.linalg.norm(target)


def states(root: Path) -> dict[str, dict[str, Any]]:
    """Read the complete current paired optimizer states.

    Returns:
        The full dense and MPS states for each physical endpoint.
    """
    return {label: experiment.read(root / label / "STATE.json") for label in experiment.ENDPOINTS}


def test_actual_q2_replay_updates_raw_audits_costs_and_source_immutability(tmp_path: Path) -> None:
    """Eight weighted paths and three full Adam updates agree through both real backends."""
    case, target = problem()
    original, original_target = copy.deepcopy(case), target.copy()
    result = experiment.run_case(tmp_path, case, "engineering", lambda: target)
    assert result["status"] == "complete"
    assert result["accepted"]
    assert case == original
    np.testing.assert_array_equal(target, original_target)
    assert result["costs"]["timing_complete"]
    assert result["costs"]["cpu_seconds"] > 0
    phases = result["costs"]["phases"]
    assert phases["target_factorization"]["completed"] == 1
    assert phases["dense_gradient"]["completed"] == phases["mps_gradient"]["completed"] == 6
    assert phases["dense_path_metrics"]["completed"] == 6
    assert phases["sampling"]["completed"] == 6
    assert sum(phase["known_cpu_seconds"] for phase in phases.values()) <= result["costs"]["cpu_seconds"]
    for label in experiment.ENDPOINTS:
        assert result["costs"]["endpoints"][label]["dense_gradient"]["completed"] == 3
        assert result["costs"]["endpoints"][label]["mps_gradient"]["completed"] == 3
    for label, state in states(tmp_path).items():
        assert state["dense"]["iterations"] == state["dense"]["evaluations"] == 3
        assert state["mps"]["iterations"] == state["mps"]["evaluations"] == 3
        assert any(value != 0 for value in state["dense"]["first_moment"])
        assert len(state["steps"]) == 3
        for key in ("theta", "first_moment", "second_moment"):
            np.testing.assert_allclose(state["dense"][key], state["mps"][key], atol=1e-10, rtol=0)
        for index, step in enumerate(state["steps"]):
            assert step["accepted"]
            assert step["update_index"] == index
            assert step["sampling"]["update_index"] == index
            assert step["sampling"]["root_seed"] == experiment.runner.seed_for("mps_overlap", case["target_id"], 11)
            assert len(step["weights"]) == 8
            assert sum(step["weights"][:4]) == pytest.approx(0.5)
            assert sum(step["weights"][4:]) == pytest.approx(0.5)
            assert step["weights"][0] != step["weights"][1]
            assert max(step["errors"].values()) < 1e-10
            assert step["audit"]["maximum_internal_bond"] <= 2
            assert step["audit"]["discarded_squared_singular_values"] == 0
            assert step["dense_gradient_resources"]["paths"] == 8
            assert step["dense_gradient_resources"]["directional_passes"] == 16
            assert step["dense_diagnostic_paths"] == 8
            assert "splits" not in step["audit"]
            audit_path = tmp_path / label / step["raw_audit_file"]
            raw = json.loads(gzip.decompress(audit_path.read_bytes()))
            assert raw["splits"]
            assert len(raw["splits"]) == raw["truncation_calls"]
            assert {key: value for key, value in raw.items() if key != "splits"} == step["audit"]
    cached = experiment.run_case(tmp_path, case, "engineering", lambda: pytest.fail("cached run loaded target"))
    assert cached == result


@pytest.mark.parametrize("after_write", [False, True])
def test_interruption_exactly_resumes_full_moments_and_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, after_write: bool
) -> None:
    """Replay only uncommitted work, preserving all full states and charging it physically."""
    case, target = problem()
    full_root, split_root = tmp_path / "full", tmp_path / "split"
    experiment.run_case(full_root, case, "engineering", lambda: target)
    real_write = pilot.write_record
    interrupted = False

    def write(path: Path, record: dict[str, Any]) -> None:
        nonlocal interrupted
        trigger = path == split_root / "original108" / "STATE.json" and not interrupted
        if trigger:
            interrupted = True
            if after_write:
                real_write(path, record)
            raise KeyboardInterrupt
        real_write(path, record)

    monkeypatch.setattr(pilot, "write_record", write)
    with pytest.raises(KeyboardInterrupt):
        experiment.run_case(split_root, case, "engineering", lambda: target)
    ledger = experiment.read(split_root / "ATTEMPTS.json")
    assert ledger["attempts"][0]["status"] == "interrupted"
    assert ledger["attempts"][0]["cpu_seconds"] > 0
    monkeypatch.setattr(experiment.MPS, "from_statevector", lambda *_: pytest.fail("target was refactorized"))
    resumed = experiment.run_case(split_root, case, "engineering", lambda: target)
    assert resumed["accepted"]
    for label in experiment.ENDPOINTS:
        full, split = states(full_root)[label], states(split_root)[label]
        assert split["dense"] == full["dense"]
        assert split["mps"] == full["mps"]
        for left, right in zip(split["steps"], full["steps"], strict=True):
            assert left["map_sha256"] == right["map_sha256"]
            assert left["weights"] == right["weights"]
            assert left["dense"] == right["dense"]
            assert left["mps"] == right["mps"]
    phases = resumed["costs"]["phases"]
    assert phases["target_factorization"]["completed"] == 1
    assert phases["mps_gradient"]["completed"] == (6 if after_write else 7)
    assert phases["dense_gradient"]["completed"] == (6 if after_write else 7)


def test_failed_gradient_is_retained_then_resumes_with_charged_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing second MPS call retains CPU and does not invent zero-cost completion."""
    case, target = problem()
    gradient = cast("Callable[..., experiment.mps_bridge.GradientResult]", experiment.mps_bridge.gradient)
    calls = 0

    def fail(*args: object, **kwargs: object) -> experiment.mps_bridge.GradientResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            msg = "injected MPS failure"
            raise RuntimeError(msg)
        return gradient(*args, **kwargs)

    monkeypatch.setattr(experiment.mps_bridge, "gradient", fail)
    failed = experiment.run_case(tmp_path, case, "engineering", lambda: target)
    assert failed["status"] == "failed"
    assert not failed["accepted"]
    assert failed["costs"]["cpu_seconds"] > 0
    phase = failed["costs"]["phases"]["mps_gradient"]
    assert phase["attempted"] == 2
    assert phase["completed"] == 1
    monkeypatch.setattr(experiment.mps_bridge, "gradient", gradient)
    final = experiment.run_case(tmp_path, case, "engineering", lambda: target)
    assert final["accepted"]
    assert final["costs"]["phases"]["mps_gradient"]["attempted"] == 7
    assert final["costs"]["phases"]["mps_gradient"]["completed"] == 6
    assert final["costs"]["cpu_seconds"] > failed["costs"]["cpu_seconds"]


def test_uncatchable_kill_marker_does_not_become_zero_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resuming an externally killed attempt keeps total CPU explicitly unavailable."""
    case, target = problem()
    real = experiment.mps_bridge.gradient

    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(experiment.mps_bridge, "gradient", interrupt)
    with pytest.raises(KeyboardInterrupt):
        experiment.run_case(tmp_path, case, "engineering", lambda: target)
    path = tmp_path / "ATTEMPTS.json"
    ledger = experiment.read(path)
    ledger["attempts"][-1].update(status="running", cpu_seconds=None, wall_seconds=None)
    ledger["attempts"][-1]["stages"][-1].update(status="running", cpu_seconds=None, wall_seconds=None)
    pilot.write_record(path, ledger)
    monkeypatch.setattr(experiment.mps_bridge, "gradient", real)
    final = experiment.run_case(tmp_path, case, "engineering", lambda: target)
    assert final["status"] == "complete"
    assert final["costs"]["cpu_seconds"] is None
    assert final["costs"]["wall_seconds"] is None
    assert not final["costs"]["timing_complete"]
    assert final["costs"]["known_cpu_seconds"] > 0
    assert final["costs"]["unknown_attempts"] == 1
    assert not final["costs"]["phases"]["mps_gradient"]["timing_complete"]


@pytest.mark.parametrize("changed", ["counter", "shape", "negative_moment"])
def test_resume_rejects_malformed_current_optimizer_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    """A checked envelope does not validate fractional counters or malformed moments."""
    case, target = problem()
    real_write = pilot.write_record

    def interrupt(path: Path, record: dict[str, Any]) -> None:
        real_write(path, record)
        if path == tmp_path / "original108" / "STATE.json":
            raise KeyboardInterrupt

    monkeypatch.setattr(pilot, "write_record", interrupt)
    with pytest.raises(KeyboardInterrupt):
        experiment.run_case(tmp_path, case, "engineering", lambda: target)
    monkeypatch.setattr(pilot, "write_record", real_write)
    path = tmp_path / "original108" / "STATE.json"
    saved = experiment.read(path)
    if changed == "counter":
        saved["dense"]["evaluations"] = 1.5
    elif changed == "shape":
        saved["dense"]["theta"].pop()
    else:
        saved["mps"]["second_moment"][0] = -1
    real_write(path, saved)
    with pytest.raises(ValueError, match="malformed"):
        experiment.run_case(tmp_path, case, "engineering", lambda: target)


@pytest.mark.parametrize("defect", ["gradient", "path", "tail", "bond"])
def test_disagreement_is_a_retained_negative_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    """Numerical disagreement, discarded weight and excessive bonds each fail the gate."""
    case, target = problem()
    gradient = cast("Callable[..., experiment.mps_bridge.GradientResult]", experiment.mps_bridge.gradient)

    def altered(*args: object, **kwargs: object) -> experiment.mps_bridge.GradientResult:
        result = gradient(*args, **kwargs)
        if defect == "gradient":
            result.gradient[0] += 1e-5
        elif defect == "path":
            result.trajectory_fidelities[0] += 1e-5
        elif defect == "tail":
            result.audit["discarded_squared_singular_values"] = 1e-30
        else:
            result.audit["maximum_internal_bond"] = 3
        return result

    monkeypatch.setattr(experiment.mps_bridge, "gradient", altered)
    result = experiment.run_case(tmp_path, case, "engineering", lambda: target)
    assert result["status"] == "complete"
    assert not result["accepted"]
    assert all(len(state["steps"]) == 3 for state in states(tmp_path).values())
    assert all(not arm["accepted"] for arm in result["endpoints"].values())


def mock_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Provide checked temporary endpoint sources without loading any q6 state.

    Returns:
        Four metadata-only sources with independent immutable identities.
    """
    case, _ = problem()
    source = tmp_path / "source.json"
    pilot.write_record(source, {"identity": "engineering"})
    panel = [
        {
            **copy.deepcopy(case),
            "target_id": target,
            "endpoints_checksum": pilot.digest(case["endpoints"]),
            "files": {str(source): experiment.file_hash(source)},
        }
        for target in experiment.TARGETS
    ]
    monkeypatch.setattr(experiment, "endpoints", lambda _: copy.deepcopy(panel))
    monkeypatch.setattr(experiment, "sources", lambda: {"kernel": "frozen"})
    monkeypatch.setattr(experiment.construction, "environment", lambda: {"threads": 1})
    monkeypatch.setattr(experiment.runner, "custody", lambda: {"benchmark_hashes": {"fixed": "hash"}})
    return panel


def test_prepare_only_freezes_before_any_simulation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preparation retains all eight sources while invoking neither target input nor kernels."""
    mock_panel(tmp_path, monkeypatch)
    monkeypatch.setattr(experiment, "load_target", lambda *_: pytest.fail("preparation loaded target"))
    monkeypatch.setattr(experiment.mps_bridge, "gradient", lambda *_: pytest.fail("preparation simulated"))
    assert experiment.main(["--output", str(tmp_path / "out"), "--prepare-only"]) == 0
    protocol = experiment.read(tmp_path / "out" / "PROTOCOL.json")
    assert len(protocol["cases"]) == 4
    assert protocol["settings"]["paths_per_channel"] == 4
    assert protocol["settings"]["updates"] == 3
    experiment.verify(protocol)
    assert experiment.prepare(tmp_path / "out", tmp_path) == protocol
    summary = experiment.summarize(tmp_path / "out", protocol)
    assert not summary["accepted"]
    assert all(row["status"] == "pending" for row in summary["cases"].values())
    assert len((tmp_path / "out" / "SCOREBOARD.csv").read_text().splitlines()) == 9


@pytest.mark.parametrize("changed", ["source", "benchmark", "kernel", "panel", "theta", "budget", "environment"])
def test_frozen_custody_rejects_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str) -> None:
    """No source edit, changed target panel, budget or numerical environment may be reused."""
    panel = mock_panel(tmp_path, monkeypatch)
    protocol = experiment.prepare(tmp_path / "out", tmp_path)
    if changed == "source":
        pilot.write_record(Path(next(iter(panel[0]["files"]))), {"identity": "changed"})
    elif changed == "benchmark":
        monkeypatch.setattr(experiment.runner, "custody", lambda: {"benchmark_hashes": {"fixed": "changed"}})
    elif changed == "kernel":
        monkeypatch.setattr(experiment, "sources", lambda: {"kernel": "changed"})
    elif changed == "panel":
        protocol["cases"][0]["target_id"] = "tfim_ferro"
    elif changed == "theta":
        protocol["cases"][0]["endpoints"]["original108"]["theta"][0] += 0.1
    elif changed == "budget":
        protocol["settings"]["updates"] += 1
    else:
        protocol["environment"] = {"threads": 4}
    with pytest.raises(ValueError, match=r"custody|source"):
        experiment.verify(protocol)


def test_completed_audit_dependency_tamper_is_detected(tmp_path: Path) -> None:
    """A modified compressed raw audit cannot survive a completed-case fast path."""
    case, target = problem()
    experiment.run_case(tmp_path, case, "engineering", lambda: target)
    path = tmp_path / "original108" / "step1.audit.json.gz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="dependency"):
        experiment.run_case(tmp_path, case, "engineering", lambda: target)


def archived_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create metadata-only q6 archives without loading or evolving a q6 state.

    Returns:
        The temporary structural artifact root matching the exact source schema.
    """
    monkeypatch.setattr(experiment.runner, "OUTPUT", tmp_path)
    monkeypatch.setattr(experiment.runner, "custody", lambda: {"benchmark_hashes": {"fixed": "hash"}})
    circuit = ParameterizedCircuit(
        6, [ParameterizedGate("rz", (index % 6,), param_index=index) for index in range(108)]
    )
    candidate = {"circuit": pilot.circuit_record(circuit), "theta": np.linspace(-0.3, 0.4, 108).tolist()}
    starts = [{"target_id": target, "replica": 11, "candidate": candidate} for target in experiment.TARGETS]
    pilot.write_record(tmp_path / "starts.json", {"starts": starts})
    root = tmp_path / "structural"
    protocol: dict[str, Any] = {
        "phase": "development",
        "targets": list(experiment.TARGETS),
        "replica": 11,
        "starts": starts,
        "source_starts_sha256": experiment.file_hash(tmp_path / "starts.json"),
        "noise_ids": list(experiment.trajectory_mixture.NOISE_IDS),
        "benchmark_hashes": {"fixed": "hash"},
        "config": {"repair_evaluations": 25, "channel_evaluations": 3000},
    }
    pilot.write_record(root / "PROTOCOL.json", protocol)
    checksums = []
    for target in experiment.TARGETS:
        case_root = root / f"{target}__11"
        circuit_hash = pilot.digest(candidate["circuit"])
        pilot.write_record(case_root / "circuits" / f"{circuit_hash}.json", candidate["circuit"])
        result: dict[str, Any] = {
            "status": "complete",
            "target_id": target,
            "replica": 11,
            "protocol_checksum": pilot.digest(protocol),
            "arms": {
                "structural_mean2_lbfgs": {
                    "status": "complete",
                    "circuit_hash": circuit_hash,
                    "theta": candidate["theta"],
                    "curve": [{"theta": candidate["theta"]}],
                }
            },
        }
        pilot.write_record(case_root / "RESULT.json", result)
        checksums.append(pilot.digest(result))
    pilot.write_record(
        root / "RESULT.json", {"status": "complete", "protocol_checksum": pilot.digest(protocol), "results": checksums}
    )
    return root


def test_archive_loader_authenticates_all_final_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Endpoint freezing authenticates result, circuit-bank and source-protocol checksums."""
    root = archived_panel(tmp_path, monkeypatch)
    monkeypatch.setattr(experiment, "load_target", lambda *_: pytest.fail("source audit loaded target"))
    panel = experiment.endpoints(root)
    assert len(panel) == 4
    for case in panel:
        assert case["replica"] == 11
        assert case["endpoints_checksum"] == pilot.digest(case["endpoints"])
        assert set(case["endpoints"]) == set(experiment.ENDPOINTS)
        assert len(case["files"]) == 5


@pytest.mark.parametrize("changed", ["budget", "case", "circuit", "final_theta", "panel"])
def test_archive_loader_rejects_unmatched_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    """A valid JSON envelope cannot substitute another endpoint or scientific source panel."""
    root = archived_panel(tmp_path, monkeypatch)
    target = experiment.TARGETS[0]
    case_path = root / f"{target}__11" / "RESULT.json"
    if changed in {"budget", "panel"}:
        path = root / "PROTOCOL.json"
        record = experiment.read(path)
        if changed == "budget":
            record["config"]["channel_evaluations"] = 100
        else:
            record["targets"][0] = "tfim_ferro"
    elif changed == "circuit":
        case = experiment.read(case_path)
        path = case_path.parent / "circuits" / f"{case['arms']['structural_mean2_lbfgs']['circuit_hash']}.json"
        record = experiment.read(path)
        record["gates"][0]["angle_offset"] += 0.1
    else:
        path = case_path
        record = experiment.read(path)
        if changed == "case":
            record["target_id"] = "tfim_ferro"
        else:
            record["arms"]["structural_mean2_lbfgs"]["theta"][0] += 0.1
    pilot.write_record(path, record)
    with pytest.raises(ValueError, match=r"panel|identity"):
        experiment.endpoints(root)


@pytest.mark.parametrize("workers", ["0", "5"])
def test_cli_rejects_unallocated_worker_counts(workers: str) -> None:
    """The bounded CLI never launches an unbounded worker pool."""
    with pytest.raises(SystemExit, match="2"):
        experiment.main(["--prepare-only", "--workers", workers])
