# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Development-only panel custody and real small-system lazy reproduction audits."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import initializer_policy
from experiments.overnight_state_preparation_20260913_0108 import lazy_initializer as lazy
from experiments.overnight_state_preparation_20260913_0108 import lazy_initializer_experiment as experiment
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def source() -> dict[str, Any]:
    """Return a nontrivial q2 scalar preparation for independent state checks."""
    circuit = ParameterizedCircuit(
        2,
        [ParameterizedGate("ry", (0,), 0), ParameterizedGate("ry", (1,), 1), ParameterizedGate("rxx", (0, 1), 2)],
    )
    return {
        "circuit": pilot.circuit_record(circuit),
        "theta": [0.7, -0.4, 0.6],
        "construction_cpu_seconds": 0.01,
        "timing_complete": True,
    }


@pytest.fixture
def metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep actual numerical kernels while replacing only slow metadata collection."""
    monkeypatch.setattr(experiment, "sources", lambda: {"wrapper": "fixed"})
    monkeypatch.setattr(lazy, "sources", lambda: {"kernel": "fixed"})
    monkeypatch.setattr(lazy.construction, "environment", lambda: {"runtime": "fixed"})
    monkeypatch.setattr(lazy.runner, "custody", lambda: {"benchmark_hashes": {"immutable": "fixed"}})


@pytest.fixture
def panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata: None) -> dict[str, Any]:
    """Return four metadata-only q6 cases without loading any scientific vector."""
    del metadata
    expected = [
        {
            "target_id": target,
            "replica": 11,
            "endpoint": source(),
            "ideal_tt_overlap": 1.0,
            "rank_two_physical_overlap": 1.0,
            "rank_two_selected": True,
            "initializer_checksum": target,
        }
        for target in experiment.TARGETS
    ]
    monkeypatch.setattr(experiment, "references", lambda _: (copy.deepcopy(expected), {"old": "fixed"}))
    loaded = []

    def loader(n: int, target_id: str) -> SimpleNamespace:
        assert n == 6
        assert target_id in experiment.TARGETS
        loaded.append(target_id)
        return SimpleNamespace(state_vector_copy=lambda: np.ones(64, dtype=np.complex128) / 8)

    monkeypatch.setattr(experiment, "load_target", loader)
    monkeypatch.setattr(experiment, "construct", lambda *_a: pytest.fail("prepare invoked constructor"))
    monkeypatch.setattr(lazy, "probe", lambda *_a: pytest.fail("prepare invoked probe"))
    protocol = experiment.prepare(tmp_path, tmp_path / "old")
    assert loaded == list(experiment.TARGETS)
    return protocol


def test_preparation_freezes_all_cases_before_probe_or_constructor(tmp_path: Path, panel: dict[str, Any]) -> None:
    """Every fixed source and target identity is prepared before any numerical branch."""
    assert len(panel["cases"]) == 4
    assert len(list(tmp_path.rglob("PROTOCOL.json"))) == 5
    assert not list(tmp_path.rglob("RESULT.json"))
    assert not list(tmp_path.rglob("PROBE.json"))
    experiment.verify(tmp_path, panel)
    assert experiment.prepare(tmp_path, tmp_path / "old") == panel


@pytest.mark.parametrize("change", ["sources", "target", "replica", "missing", "reference", "child"])
def test_batch_custody_rejects_altered_complete_panel(tmp_path: Path, panel: dict[str, Any], change: str) -> None:
    """Neither a held-out identity nor a rewritten reference/child can enter an existing batch."""
    altered = copy.deepcopy(panel)
    if change == "sources":
        altered["sources"]["wrapper"] = "changed"
    elif change == "target":
        altered["cases"][0]["target_id"] = "tfim_ferro"
    elif change == "replica":
        altered["cases"][0]["replica"] = 29
    elif change == "missing":
        altered["cases"].pop()
    elif change == "reference":
        altered["cases"][0]["reference"]["endpoint"]["theta"][0] += 0.1
    else:
        path = tmp_path / f"{experiment.TARGETS[0]}__11/initializer/PROTOCOL.json"
        record = lazy.read(path)
        record["constructor"]["config"]["source_updates"] += 1
        pilot.write_record(path, record)
    with pytest.raises(ValueError, match="custody"):
        experiment.verify(tmp_path, altered)


def test_audit_requires_identical_parameters_even_for_same_physical_state() -> None:
    """An exact periodic parameter shift passes state equivalence but fails recipe equality."""
    record: dict[str, Any] = {
        "endpoint": source(),
        "ideal_tt_overlap": 0.8,
        "rank_two_physical_overlap": 1.0,
        "rank_two_selected": False,
    }
    assert experiment.audit(record, record)["matches"]
    shifted = copy.deepcopy(record)
    shifted["endpoint"]["theta"][0] += 2 * np.pi
    checked = experiment.audit(shifted, record)
    assert not checked["matches"]
    assert checked["circuit_equal"]
    assert not checked["theta_equal"]
    assert checked["phase_aligned_max_state_error"] < 1e-12
    shifted["endpoint"]["theta"][0] += 0.1
    assert experiment.audit(shifted, record)["phase_aligned_max_state_error"] > 0.01


@pytest.fixture
def small_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata: None) -> dict[str, Any]:
    """Return a synthetic q2 job with the real lazy numerical kernel and durable adapter."""
    del metadata
    vector = initializer_policy.state_vector(source()["circuit"], np.asarray(source()["theta"]))
    expected = initializer_policy.build_initializers(vector, source())
    case = {
        "target_id": "synthetic_case",
        "replica": 11,
        "reference": {
            "endpoint": expected["conditional"],
            "ideal_tt_overlap": expected["ideal_tt_overlap"],
            "rank_two_physical_overlap": expected["rank_two_physical_overlap"],
            "rank_two_selected": expected["rank_two_selected"],
        },
    }
    root = tmp_path / "synthetic_case__11"
    lazy.prepare(root / "initializer", vector, "synthetic_case", 11, phase="engineering", constructor={"test": True})
    protocol = {"cases": [case]}
    monkeypatch.setattr(experiment, "verify", lambda *_a: None)
    monkeypatch.setattr(experiment, "load_target", lambda *_a: SimpleNamespace(state_vector_copy=vector.copy))
    monkeypatch.setattr(experiment, "construct", lambda *_a: pytest.fail("selected rank two constructed legacy"))
    return {"root": str(tmp_path), "protocol": protocol, "case": case}


def test_real_small_case_audit_is_compact_and_authenticates_cached_dependencies(
    tmp_path: Path, small_job: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real synthesis/compilation matches the old eager helper and saves no duplicate histories."""
    result = experiment.run_case(small_job)
    assert result["accepted"]
    assert result["audit"]["theta_equal"]
    assert result["audit"]["lazy_branch_verified"]
    assert result["initializer_cpu_seconds"] > 0
    assert result["nested_legacy_cpu_seconds"] is None
    assert result["physical_invocation"]["cpu_seconds"] >= result["initializer_cpu_seconds"]
    assert "endpoint" not in result
    assert "synthesis" not in result
    monkeypatch.setattr(lazy, "probe", lambda *_a: pytest.fail("completed case replayed"))
    assert experiment.run_case(small_job) == result
    path = tmp_path / "synthetic_case__11/initializer/PROBE.json"
    record = lazy.read(path)
    record["ideal_tt_overlap"] -= 0.01
    pilot.write_record(path, record)
    with pytest.raises(ValueError, match="dependency"):
        experiment.run_case(small_job)


def test_interrupted_audit_reuses_completed_initializer_and_charges_both_invocations(
    tmp_path: Path, small_job: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interruption after numerical initialization preserves its complete state and full elapsed work."""
    real_audit = experiment.audit

    def interrupt(*_args: object) -> dict[str, Any]:
        raise KeyboardInterrupt

    monkeypatch.setattr(experiment, "audit", interrupt)
    with pytest.raises(KeyboardInterrupt):
        experiment.run_case(small_job)
    initial = lazy.read(tmp_path / "synthetic_case__11/initializer/RESULT.json")
    monkeypatch.setattr(experiment, "audit", real_audit)
    monkeypatch.setattr(lazy, "probe", lambda *_a: pytest.fail("completed initialization replayed"))
    result = experiment.run_case(small_job)
    ledger = lazy.read(tmp_path / "synthetic_case__11/CASE_ATTEMPTS.json")
    assert [a["status"] for a in ledger["attempts"]] == ["interrupted", "complete"]
    assert all(a["cpu_seconds"] > 0 for a in ledger["attempts"])
    assert result["physical_invocation"]["cpu_seconds"] == sum(a["cpu_seconds"] for a in ledger["attempts"])
    assert result["initializer_cpu_seconds"] == initial["initializer_cpu_seconds"]


def test_unknown_killed_case_blocks_before_loading_target(
    tmp_path: Path, small_job: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing elapsed work cannot be silently replaced by a fresh free invocation."""
    ledger: dict[str, Any] = {
        "signature": pilot.digest([small_job["protocol"], small_job["case"]]),
        "attempts": [{"status": "running", "cpu_seconds": None, "wall_seconds": None}],
    }
    pilot.write_record(tmp_path / "synthetic_case__11/CASE_ATTEMPTS.json", ledger)
    monkeypatch.setattr(experiment, "load_target", lambda *_a: pytest.fail("unknown work loaded target"))
    with pytest.raises(ValueError, match="Unknown"):
        experiment.run_case(small_job)


def test_numerical_failure_is_durable_and_never_omitted_from_panel(
    tmp_path: Path, panel: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Summary restart retains failed and pending rows with unknown costs represented as null."""
    case = panel["cases"][0]

    def fail(*_args: object) -> dict[str, Any]:
        msg = "deliberate arithmetic failure"
        raise FloatingPointError(msg)

    monkeypatch.setattr(experiment, "run_case", fail)
    result = experiment.attempt_case({"root": str(tmp_path), "protocol": panel, "case": case})
    assert result["status"] == "failed"
    assert result["physical_invocation"]["cpu_seconds"] is None
    summary = experiment.summarize(tmp_path, panel, {})
    assert not summary["accepted"]
    assert len(summary["cases"]) == 4
    assert [r["status"] for r in summary["cases"]] == ["failed", "pending", "pending", "pending"]
    assert summary["cases"][0]["failure"].endswith("deliberate arithmetic failure")


@pytest.mark.parametrize("workers", ["0", "5"])
def test_cli_rejects_unallocated_worker_counts(workers: str) -> None:
    """The development-only CLI has no unrestricted pool or confirmation switch."""
    with pytest.raises(SystemExit, match="2"):
        experiment.main(["--prepare-only", "--workers", workers])


@pytest.mark.parametrize("field", ["audit", "cost", "branch", "initializer", "new_child"])
def test_completion_receipt_binds_all_top_level_fields_and_complete_child_map(
    tmp_path: Path, small_job: dict[str, Any], field: str
) -> None:
    """Resigned outer JSON edits and additional unregistered child evidence are rejected."""
    result = experiment.run_case(small_job)
    root = tmp_path / "synthetic_case__11"
    if field == "audit":
        result["audit"]["phase_aligned_max_state_error"] += 0.1
    elif field == "cost":
        result["initializer_cpu_seconds"] += 1
    elif field == "branch":
        result["rank_two_selected"] = False
    elif field == "initializer":
        result["initializer_checksum"] = "unlinked"
    else:
        pilot.write_record(root / "initializer/UNREGISTERED.json", {"unexpected": True})
    pilot.write_record(root / "RESULT.json", result)
    with pytest.raises(ValueError, match="dependency registry"):
        experiment.run_case(small_job)
