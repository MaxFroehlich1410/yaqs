# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Resource-ledger provenance and nonadditive accounting, with tiny signed fixtures."""

# Private adapters require direct synthetic fixtures to test interrupted resource records.
# ruff: noqa: SLF001

from __future__ import annotations

import hashlib
import shutil
from typing import TYPE_CHECKING, Any

import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import resource_accounting as accounting

if TYPE_CHECKING:
    from pathlib import Path


def signed(path: Path, record: dict[str, Any]) -> str:
    """Write a tiny fixture.

    Returns:
        Authenticated content checksum.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pilot.write_record(path, record)
    return pilot.digest(record)


@pytest.fixture
def panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make a complete one-case original-runner panel with real source checksums.

    Returns:
        Root containing the frozen protocol and signed records.
    """
    source = tmp_path / "driver.py"
    source.write_text("# numerical source\n")
    monkeypatch.setattr(accounting.runner, "REPO", tmp_path)
    monkeypatch.setattr(accounting.runner, "custody", lambda: {"benchmark_hashes": {"benchmark": "fixed"}})
    root = tmp_path / "wave1"
    protocol: dict[str, Any] = {
        "protocol": {
            "name": "wave1",
            "targets": ["tiny"],
            "replicas": [11],
            "methods": {"unchanged": {"method": "unchanged"}},
        },
        "benchmark_hashes": {"benchmark": "fixed"},
        "source_hashes": {"driver.py": hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    checksum = signed(root / "PROTOCOL.json", protocol)
    case: dict[str, Any] = {
        "status": "complete",
        "signature": "physical-run-one",
        "job": {
            "target_id": "tiny",
            "replica": 11,
            "label": "unchanged",
            "config": {"method": "unchanged"},
            "runtime_checksum": checksum,
        },
        "state": {
            "direction_passes": 0,
            "training_cpu_seconds": 0.0,
            "training_wall_seconds": 0.0,
            "exact_training_evaluations": 0,
            "trajectory_paths": 0,
            "iterations": 0,
            "curve": [{}],
            "diagnostic_cpu_seconds": 0.2,
        },
        "evaluation_cpu_seconds": 0.3,
        "invocation_wall_seconds": 0.6,
        "peak_rss_bytes": 100,
    }
    case_checksum = signed(root / "tiny__11__unchanged.json", case)
    signed(root / "RESULT.json", {"status": "complete", "runtime_checksum": checksum, "runs": [case_checksum]})
    return root


def test_copied_panel_and_annotation_do_not_duplicate_physical_work(panel: Path) -> None:
    """An identical archive copy adds proofs and labels, not another execution."""
    copied = panel.parent / "copy"
    shutil.copytree(panel, copied)
    (copied / "ENGINEERING_NOTE.json").write_text('{"note":"annotations only"}')
    result = accounting.collect([panel, copied, panel])
    assert len(result["physical_events"]) == 3
    assert result["measured_totals"]["cpu_seconds"]["known_subtotal"] == pytest.approx(0.5)
    assert result["measured_totals"]["exact_forward_calls_completed"]["known_subtotal"] == 13
    assert result["measured_totals"]["exact_gradient_calls_completed"]["known_subtotal"] == 1
    assert result["complete_development_cpu_seconds"] is None
    assert result["maximum_reported_worker_peak_rss_bytes"] == 100
    assert all(len(event["panels"]) == 2 for event in result["physical_events"])


def test_measured_subtotals_do_not_impute_unknown_times() -> None:
    """Missing historical CPU stays null even when another execution was measured."""
    assert accounting.sum_known([0, 2, None]) == {"known_subtotal": 2, "total": None, "unknown_count": 1}
    ledger = accounting.Ledger()
    ledger.add(
        "one", "panel", "optimization", {"cpu_seconds": 2, "peak_rss_bytes": 80}, [], label="a", scope="measured"
    )
    ledger.add(
        "two", "panel", "optimization", {"cpu_seconds": None, "peak_rss_bytes": 90}, [], label="b", scope="untimed"
    )
    ledger.observe("panel", "reference", "reused_reference", {"cpu_seconds": 100}, [])
    ledger.observe("panel", "wrapper", "covering_invocation", {"cpu_seconds": 200}, [])
    result = ledger.result()
    assert result["measured_totals"]["cpu_seconds"] == {"known_subtotal": 2, "total": None, "unknown_count": 1}
    assert result["maximum_reported_worker_peak_rss_bytes"] == 90


def test_distinct_durable_attempts_count_even_with_equal_timings() -> None:
    """Equal times are not evidence that two independent attempt identities were reused."""
    ledger = accounting.Ledger()
    for identity in ("run/attempt0", "run/attempt1", "run/attempt0"):
        ledger.add(identity, "panel", "optimization", {"cpu_seconds": 1}, [], label=identity, scope="attempt")
    assert ledger.result()["measured_totals"]["cpu_seconds"]["known_subtotal"] == 2


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "unknown"])
def test_invalid_resource_measurements_rejected(value: object) -> None:
    """Invalid numeric values cannot masquerade as known scientific costs."""
    with pytest.raises(ValueError, match="nonnegative finite"):
        accounting.Ledger().add("id", "panel", "optimization", {"cpu_seconds": value}, [], label="case", scope="test")


def test_covering_timer_residual_is_nonoverlapping() -> None:
    """Outer timing is partitioned, not added wholesale to its inner components."""
    assert accounting.residual(4, [1, 2]) == 1
    assert accounting.residual(4, [None]) is None
    with pytest.raises(ValueError, match="exceed"):
        accounting.residual(1, [2])


def test_source_bytes_and_record_checksums_are_authenticated(panel: Path) -> None:
    """Changed source and changed unsigned content both fail closed."""
    (panel.parent / "driver.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="source hash mismatch"):
        accounting.collect([panel])
    (panel.parent / "driver.py").write_text("# numerical source\n")
    path = panel / "tiny__11__unchanged.json"
    path.write_text(path.read_text().replace('"evaluation_cpu_seconds": 0.3', '"evaluation_cpu_seconds": 10.3'))
    with pytest.raises(ValueError, match="checksum"):
        accounting.collect([panel])


def test_missing_whole_case_or_completion_entry_is_visible(panel: Path) -> None:
    """Missing methods/cases never produce a smaller apparently complete compute total."""
    path = panel / "RESULT.json"
    completion = accounting.read(path)
    completion["runs"] = []
    signed(path, completion)
    with pytest.raises(ValueError, match="every expected case"):
        accounting.collect([panel])


def test_unsupported_schema_is_not_silently_discovered_as_zero(panel: Path) -> None:
    """A future panel requires an explicit adapter before it can enter the ledger."""
    path = panel / "PROTOCOL.json"
    protocol = accounting.read(path)
    protocol["version"] = "future.unsupported.v2"
    signed(path, protocol)
    with pytest.raises(ValueError, match="Unsupported"):
        accounting.collect([panel])


def test_wrong_case_method_and_stale_completion_hash_rejected(panel: Path) -> None:
    """Individually signed but mismatched records cannot enter another panel's ledger."""
    path = panel / "tiny__11__unchanged.json"
    case = accounting.read(path)
    case["job"]["label"] = "other"
    signed(path, case)
    with pytest.raises(ValueError, match="case method"):
        accounting.collect([panel])
    case["job"]["label"] = "unchanged"
    case["evaluation_cpu_seconds"] += 1
    signed(path, case)
    with pytest.raises(ValueError, match="case hashes"):
        accounting.collect([panel])


def test_constructor_internal_diagnostic_is_not_added_again(tmp_path: Path) -> None:
    """The MPS timer includes internal diagnostics; fresh external audit is additional."""
    root = tmp_path / "construction" / "tiny__11"
    attempts: dict[str, Any] = {
        "signature": "construction-one",
        "attempts": [{"status": "complete", "cpu_seconds": 10, "wall_seconds": 11}],
    }
    checksum = signed(root / "ATTEMPTS.json", attempts)
    construction: dict[str, Any] = {"updates": 570, "attempts_checksum": checksum, "diagnostic_cpu_seconds": 2}
    checksum = signed(root / "RESULT.json", construction)
    audit: dict[str, Any] = {
        "signature": "audit-one",
        "construction_checksum": checksum,
        "final_forward_peak_bond": 8,
        "state_audit_cpu_seconds": 1,
        "wrapper_cpu_seconds": 12,
        "construction_cpu_seconds": 10,
        "wrapper_wall_seconds": 14,
        "construction_wall_seconds": 11,
        "historical_cpu_seconds": None,
        "archive_checksum": "archive-one",
    }
    signed(root / "AUDIT.json", audit)
    ledger = accounting.Ledger()
    accounting._construction(root, audit, ledger, ledger.proof(root / "AUDIT.json", audit))
    result = ledger.result()
    assert result["measured_totals"]["cpu_seconds"]["known_subtotal"] == 12
    assert result["measured_totals"]["mps_optimizer_updates"]["known_subtotal"] == 570
    assert result["measured_totals"]["wall_seconds"]["known_subtotal"] == 14
    assert result["nonadditive_observations"][-1]["costs"]["historical_cpu_seconds"] is None


def test_failed_density_attempt_retains_attempts_and_completed_directions(tmp_path: Path) -> None:
    """A failure after one of two channels costs CPU and one completed gradient."""
    ledger = accounting.Ledger()
    attempts: dict[str, Any] = {
        "signature": "oracle",
        "attempts": [
            {
                "status": "failed",
                "attempted_channel_calls": 2,
                "completed_channel_calls": 1,
                "cpu_seconds": 3,
                "wall_seconds": 4,
            },
            {
                "status": "running",
                "attempted_channel_calls": 0,
                "completed_channel_calls": 0,
                "cpu_seconds": None,
                "wall_seconds": None,
            },
        ],
    }
    accounting._density_attempt(
        tmp_path / "panel" / "tiny__11", {"signature": "case"}, "attempts", attempts, "proof", ledger, gradient=True
    )
    result = ledger.result()
    assert result["measured_totals"]["exact_gradient_calls_attempted"]["known_subtotal"] == 2
    assert result["measured_totals"]["exact_gradient_calls_completed"]["known_subtotal"] == 1
    assert result["measured_totals"]["density_direction_passes"]["known_subtotal"] == 2
    assert result["measured_totals"]["cpu_seconds"]["total"] is None
    assert result["measured_totals"]["cpu_seconds"]["known_subtotal"] == 3


def test_shared_warmup_and_rejected_repair_are_counted_once(tmp_path: Path) -> None:
    """Shared warmup is removed once; the whole-search timer includes rejected repairs."""
    root = tmp_path / "structural" / "tiny__11"
    warm: dict[str, Any] = {"physical_joint_gradient_calls": 2}
    signed(root / "branches/warmup.json", warm)
    for name, cpu, calls in (("warmup", 2, 2), ("fixed", 3, 3), ("repair_000", 4, 4), ("polish", 1, 1)):
        signed(
            root / f"branches/{name}.attempts.json",
            {"attempts": [{"completed": True, "cpu_seconds": cpu, "wall_seconds": cpu, "objective_calls": calls}]},
        )
    signed(root / "SEARCH_ATTEMPTS.json", {"attempts": [{"cpu_seconds": 5, "wall_seconds": 5}]})
    arm: dict[str, Any] = {
        "physical_joint_gradient_calls": 7,
        "forward_score_calls": 1,
        "deletion_ranking_calls": 1,
        "evaluation_cpu_seconds": 0.1,
        "fidelities": dict.fromkeys(range(12), 0.5),
        "cpu_seconds": 8,
    }
    fixed: dict[str, Any] = {**arm, "physical_joint_gradient_calls": 5, "cpu_seconds": 5}
    case: dict[str, Any] = {
        "signature": "case",
        "arms": {"structural_lbfgs": arm, "fixed_lbfgs": fixed},
        "pair": {"physical_training_cpu_seconds": 11, "shared_warmup_cpu_seconds": 2, "shared_warmup_calls": 2},
    }
    ledger = accounting.Ledger()
    accounting._structural(root, case, ledger, "caseproof")
    result = ledger.result()
    optimization = next(event for event in result["physical_events"] if event["category"] == "optimization")
    assert optimization["measurements"]["cpu_seconds"] == 11
    assert optimization["measurements"]["wall_seconds"] == 11
    assert optimization["measurements"]["exact_gradient_calls_completed"] == 10
    assert optimization["measurements"]["density_direction_passes"] == 23
    assert result["measured_totals"]["cpu_seconds"]["known_subtotal"] == pytest.approx(11.2)


def test_both_estimator_executions_charge_one_snapshot_each(tmp_path: Path) -> None:
    """A typing-only rerun repeats physical work, while each run reuses its own anchor."""
    ledger = accounting.Ledger()
    for revision in ("original", "typed"):
        root = tmp_path / revision
        protocol: dict[str, Any] = {"targets": ["tiny"], "distances": [0, 0.2], "replicates": 2, "revision": revision}
        runtime = pilot.digest(protocol)
        anchor: dict[str, Any] = {
            "runtime_hash": runtime,
            "snapshot": {"estimate": {"cpu_seconds": 1, "wall_seconds": 1}},
            "shared_backend_setup_cpu_seconds": 0.5,
        }
        anchor_checksum = signed(root / "tiny__anchor.json", anchor)
        settings = []
        for index, distance in enumerate(protocol["distances"]):
            estimate = {
                "cv": {"cpu_seconds": 0.1, "wall_seconds": 0.2, "paths": 4, "directional_passes": 8},
                "weighted": {"cpu_seconds": 0.2, "wall_seconds": 0.3, "paths": 4, "directional_passes": 8},
            }
            setting: dict[str, Any] = {
                "runtime_hash": runtime,
                "anchor_checksum": anchor_checksum,
                "distance": distance,
                "target_id": "tiny",
                "state": {
                    "signature": f"setting{index}",
                    "oracle_cpu_seconds": 0 if index == 0 else 2,
                    "oracle_exact_evaluations": index,
                    "replications": [estimate, estimate],
                },
            }
            settings.append({"run_checksum": signed(root / f"tiny__distance_{index}.json", setting)})
        accounting._estimator(root, protocol, {"anchor_checksums": [anchor_checksum], "settings": settings}, ledger)
    result = ledger.result()
    assert result["measured_totals"]["cpu_seconds"]["known_subtotal"] == pytest.approx(9.4)
    assert result["measured_totals"]["exact_gradient_calls_completed"]["known_subtotal"] == 4
    assert result["measured_totals"]["trajectory_direction_passes"]["known_subtotal"] == 128
    assert all(
        not observation["costs"]["independent_replication"] for observation in result["nonadditive_observations"]
    )


def test_original_estimator_accepts_only_exact_frozen_source_bytes(panel: Path) -> None:
    """The exception for an annotation-only old driver is an exact archived-byte match."""
    protocol: dict[str, Any] = {
        "schema": "yaqs.overnight.exact_anchor_diagnostic.v1",
        "benchmark_hashes": {"benchmark": "fixed"},
        "source_hashes": {"estimator_diagnostic.py": hashlib.sha256(b"# frozen old driver\n").hexdigest()},
    }
    signed(panel / "PROTOCOL.json", protocol)
    signed(panel / "RESULT.json", {"runtime_hash": pilot.digest(protocol), "status": "complete"})
    snapshot = panel / "frozen_driver.py"
    snapshot.write_text("# frozen old driver\n")
    kind, _, _ = accounting.authenticate(panel, accounting.Ledger())
    assert kind == "estimator"
    snapshot.write_text("# different snapshot\n")
    with pytest.raises(ValueError, match="source hash mismatch"):
        accounting.authenticate(panel, accounting.Ledger())


def test_cached_endpoint_references_do_not_repeat_a_physical_score(tmp_path: Path) -> None:
    """Final and CPU-selected views may both point at one exact score-cache execution."""
    root = tmp_path / "ensemble" / "tiny__11"
    score: dict[str, Any] = {"signature": "score-one", "cpu_seconds": 2, "forward_channel_evaluations": 12}
    checksum = signed(root / "scores/one.json", score)
    record: dict[str, Any] = {"post_training_scoring": {"checksums": [checksum]}}
    ledger = accounting.Ledger()
    accounting._score_cache(root, record, ledger, geometry=False)
    accounting._score_cache(root, record, ledger, geometry=False)
    result = ledger.result()
    assert result["measured_totals"]["cpu_seconds"]["known_subtotal"] == 2
    assert result["measured_totals"]["exact_forward_calls_completed"]["known_subtotal"] == 12
    signed(root / "scores/two.json", {**score, "signature": "unindexed"})
    with pytest.raises(ValueError, match="Unindexed score-cache"):
        accounting._score_cache(root, record, ledger, geometry=False)


def test_initializer_reference_charge_is_not_new_constructor_work(tmp_path: Path) -> None:
    """Declared old construction and unavailable common-total points remain separate."""
    root = tmp_path / "initializer" / "tiny__11"
    attempts: dict[str, Any] = {"signature": "build", "attempts": [{"status": "complete", "cpu_seconds": 5}]}
    checksum = signed(root / "INITIALIZER_ATTEMPTS.json", attempts)
    initial: dict[str, Any] = {
        "signature": "build",
        "attempts_checksum": checksum,
        "physical_build_wall_seconds": 6,
        "costs": {
            "rank_two_synthesis_cpu_seconds": 0.2,
            "rank_two_compilation_and_audit_cpu_seconds": 0.3,
            "legacy_compilation_and_audit_cpu_seconds": 0.4,
            "overlap_and_lowering_audit_cpu_seconds": 0.1,
            "legacy_construction_cpu_seconds": 1000,
        },
    }
    initial_checksum = signed(root / "INITIALIZERS.json", initial)
    training_attempts: dict[str, Any] = {
        "signature": "training",
        "attempts": [
            {
                "status": "complete",
                "attempted_channel_calls": 2,
                "completed_channel_calls": 2,
                "training_cpu_seconds": 2,
                "deployment_cpu_seconds": 1,
                "wall_seconds": 4,
            }
        ],
    }
    checksum = signed(root / "legacy_lbfgs/ATTEMPTS.json", training_attempts)
    checksum = signed(root / "legacy_lbfgs/TRAINING.json", {"signature": "training", "attempts_checksum": checksum})
    score: dict[str, Any] = {
        "signature": "score",
        "status": "complete",
        "evaluation_cpu_seconds": 0.5,
        "fidelities": dict.fromkeys((str(i) for i in range(12)), 0.5),
    }
    signed(root / "scores/score.json", score)
    arm: dict[str, Any] = {
        "training_checksum": checksum,
        "initializer_cpu_seconds": 1000.4,
        "training_cpu_seconds": 2,
        "deployment_cpu_seconds": 1,
        "refinement_cpu_seconds": 3,
        "total_cpu_seconds": 1003.4,
    }
    case: dict[str, Any] = {
        "signature": "case",
        "initializer_checksum": initial_checksum,
        "arms": {"legacy_lbfgs": arm, "legacy_unchanged": {**arm, "training_checksum": None}},
        "common_total_cpu": {
            "status": "complete",
            "points": [
                {
                    "fraction": 1,
                    "cpu_seconds": 10,
                    "all_available": False,
                    "arms": {"legacy_lbfgs": {"status": "initializer_not_yet_available"}},
                }
            ],
        },
        "dependencies": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*.json")
        },
    }
    ledger = accounting.Ledger()
    accounting._initializer(root, case, ledger, "caseproof")
    result = ledger.result()
    assert result["measured_totals"]["cpu_seconds"]["known_subtotal"] == pytest.approx(8.5)
    assert result["by_category_cpu_seconds"]["construction"]["known_subtotal"] == pytest.approx(0.2)
    availability = next(
        item for item in result["nonadditive_observations"] if item["kind"] == "initializer_build_and_charges"
    )
    assert not availability["costs"]["common_total_availability"][0]["all_available"]
    assert result["measured_totals"]["exact_gradient_calls_completed"]["known_subtotal"] == 2


def hash_children(root: Path) -> dict[str, str]:
    """Return raw checksums for a small fixture's signed child index."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.json")}


def exact_attempt(cpu: float, calls: int) -> dict[str, Any]:
    """Return a measured complete exact-channel attempt fixture."""
    return {
        "status": "complete",
        "cpu_seconds": cpu,
        "wall_seconds": cpu + 0.1,
        "attempted_channel_calls": calls,
        "completed_channel_calls": calls,
    }


def timed_attempts(root: Path, name: str, cpu: float, *, calls: int = 0) -> dict[str, Any]:
    """Write a timer with identity independent of its copied filesystem root.

    Returns:
        The saved durable attempt record.
    """
    payload = {"signature": name, "attempts": [exact_attempt(cpu, calls)]}
    signed(root / name, payload)
    return payload


def test_common_control_reused_training_is_not_fresh_cost(tmp_path: Path) -> None:
    """New score work and wrapper sum once, while large source procedure charges stay references."""
    timed_attempts(tmp_path, "operations/audit_one.attempts.json", 0.2, calls=12)
    outer = timed_attempts(tmp_path, "RESULT.attempts.json", 0.5)
    record: dict[str, Any] = {
        "signature": "common",
        "attempts_checksum": pilot.digest(outer),
        "cpu_seconds": 0.5,
        "wall_seconds": 0.6,
        "value": {
            "dependencies": {k: v for k, v in hash_children(tmp_path).items() if "/" in k},
            "total_cpu_seconds": 999,
            "source_case_checksum": "archive",
        },
    }
    ledger = accounting.Ledger()
    accounting._common_control(tmp_path, record, ledger, "case")
    accounting._common_control(tmp_path, record, ledger, "case")
    result = ledger.result()
    assert result["measured_totals"]["cpu_seconds"]["known_subtotal"] == pytest.approx(0.5)
    assert result["measured_totals"]["exact_forward_calls_completed"]["known_subtotal"] == 12
    assert result["measured_totals"]["exact_gradient_calls_completed"]["known_subtotal"] == 0
    record["cpu_seconds"] += 1
    with pytest.raises(ValueError, match="wrapper duration"):
        accounting._common_control(tmp_path, record, accounting.Ledger(), "case")


def test_state_compiler_pair_counts_both_scores_and_nested_audit_once(tmp_path: Path) -> None:
    """Both scores are new even when training is reused; compiler audit already belongs to its timer."""
    for side in ("before", "after"):
        timed_attempts(tmp_path, f"{side}/scores/endpoint/ATTEMPTS.json", 0.1, calls=12)
    timed_attempts(tmp_path, "COMPILATION.attempts.json", 0.3)
    outer = timed_attempts(tmp_path, "CASE_ATTEMPTS.json", 0.9)
    record: dict[str, Any] = {
        "signature": "compiler",
        "dependencies": hash_children(tmp_path),
        "physical_invocation": {"cpu_seconds": 0.9, "wall_seconds": 1.0, "attempts_checksum": pilot.digest(outer)},
    }
    ledger = accounting.Ledger()
    accounting._state_compilation(tmp_path, record, ledger, "case")
    result = ledger.result()
    assert result["measured_totals"]["cpu_seconds"]["known_subtotal"] == pytest.approx(0.9)
    assert result["measured_totals"]["exact_forward_calls_completed"]["known_subtotal"] == 24
    assert result["by_category_cpu_seconds"]["compilation"]["known_subtotal"] == pytest.approx(0.3)
    timed_attempts(tmp_path, "new/ATTEMPTS.json", 1)
    with pytest.raises(ValueError, match="omits a durable"):
        accounting._state_compilation(tmp_path, record, accounting.Ledger(), "case")


def lazy_fixture(root: Path, *, fallback: bool) -> dict[str, Any]:
    """Construct a signed lazy branch with fresh fallback only when executed.

    Returns:
        The saved lazy initializer result.
    """
    legacy_checksum = None
    if fallback:
        attempted = timed_attempts(root, "legacy/ATTEMPTS.json", 5)
        legacy_checksum = signed(
            root / "legacy/RESULT.json",
            {
                "updates": 570,
                "attempts_checksum": pilot.digest(attempted),
            },
        )
    payload = {
        "signature": "lazy",
        "attempts": [
            {"status": "complete", "cpu_seconds": 6.0, "wall_seconds": 7.0, "dependencies": hash_children(root)}
        ],
    }
    signed(root / "ATTEMPTS.json", payload)
    result: dict[str, Any] = {
        "signature": "lazy",
        "initializer_cpu_seconds": 6.0,
        "initializer_wall_seconds": 7.0,
        "attempts_checksum": pilot.digest(payload),
        "legacy_receipt_checksum": legacy_checksum,
        "rank_two_selected": not fallback,
        "probe_cpu_seconds": 0.2,
        "discarded_probe_cpu_seconds": 0.2 if fallback else 0.0,
        "nested_legacy_cpu_seconds": 5 if fallback else None,
    }
    signed(root / "RESULT.json", result)
    return result


@pytest.mark.parametrize("fallback", [False, True])
def test_lazy_probe_and_fresh_fallback_are_partitioned_not_doubled(tmp_path: Path, *, fallback: bool) -> None:
    """The entire lazy call costs six seconds in either branch; discarded probes remain included."""
    lazy_fixture(tmp_path, fallback=fallback)
    ledger = accounting.Ledger()
    accounting._lazy_costs(tmp_path, ledger)
    accounting._lazy_costs(tmp_path, ledger)
    result = ledger.result()
    assert result["measured_totals"]["cpu_seconds"]["known_subtotal"] == 6
    assert result["measured_totals"]["mps_optimizer_updates"]["known_subtotal"] == (570 if fallback else 0)
    assert result["nonadditive_observations"][-1]["costs"]["discarded_probe_cpu_seconds"] == pytest.approx(
        0.2 if fallback else 0.0
    )


def test_compiled_initial_shared_compiler_charged_once_physically(tmp_path: Path) -> None:
    """Both compiler families count physically once despite repeated seven-arm procedure charges."""
    timed_attempts(tmp_path, "state_compilations/shared.attempts.json", 0.2)
    timed_attempts(tmp_path, "compilations/final.attempts.json", 0.1)
    outer = timed_attempts(tmp_path, "CASE_ATTEMPTS.json", 0.8)
    execution: dict[str, Any] = {
        "signature": "compiled",
        "dependencies": hash_children(tmp_path),
        "initializer_cpu_seconds": 37,
        "physical_invocation": {"attempts_checksum": pilot.digest(outer)},
        "arms": {
            str(index): {"initializer_cpu_seconds": 37, "refinement_cpu_seconds": 1, "total_cpu_seconds": 38}
            for index in range(7)
        },
    }
    signed(tmp_path / "EXECUTION.json", execution)
    record = {**execution, "execution_checksum": pilot.digest(execution), "signature": "driver"}
    ledger = accounting.Ledger()
    accounting._compiled_resource(tmp_path, record, ledger, "case")
    result = ledger.result()
    assert result["measured_totals"]["cpu_seconds"]["known_subtotal"] == pytest.approx(0.8)
    assert result["by_category_cpu_seconds"]["compilation"]["known_subtotal"] == pytest.approx(0.3)
    record["initializer_cpu_seconds"] = 0
    with pytest.raises(ValueError, match="public resource values"):
        accounting._compiled_resource(tmp_path, record, accounting.Ledger(), "case")


def sealed_phase(root: Path, value: dict[str, Any], cpu: float) -> dict[str, Any]:
    """Seal a parent phase around existing children without numerical execution.

    Returns:
        The saved sealed phase result.
    """
    payload = {
        "signature": root.name,
        "attempts": [{"status": "complete", "cpu_seconds": cpu, "wall_seconds": cpu + 1, "files": hash_children(root)}],
    }
    result = {
        "signature": root.name,
        "value": value,
        "cpu_seconds": cpu,
        "wall_seconds": cpu + 1,
        "attempts_checksum": pilot.digest(payload),
    }
    signed(root / "PHASE.json", {"signature": root.name})
    signed(root / "ATTEMPTS.json", payload)
    signed(
        root / "COMPLETED.json",
        {"signature": root.name, "result_checksum": pilot.digest(result), "attempts_checksum": pilot.digest(payload)},
    )
    signed(root / "RESULT.json", result)
    return result


def test_confirmation_phase_seal_binds_costs_and_extra_nonjson_files(tmp_path: Path) -> None:
    """Re-signing a public phase or appending an unindexed binary cannot change paid work."""
    result = sealed_phase(tmp_path, {"score": 0.9}, 2)
    assert accounting._phase_resource(tmp_path, accounting.Ledger())[0] == result
    changed = {**result, "cpu_seconds": 1}
    signed(tmp_path / "RESULT.json", changed)
    with pytest.raises(ValueError, match="phase resource"):
        accounting._phase_resource(tmp_path, accounting.Ledger())
    signed(tmp_path / "RESULT.json", result)
    (tmp_path / "unindexed.bin").write_bytes(b"not zero-cost work")
    with pytest.raises(ValueError, match="phase resource"):
        accounting._phase_resource(tmp_path, accounting.Ledger())


def test_confirmation_nested_case_phases_count_initialization_once(tmp_path: Path) -> None:
    """Preparation charges are references inside arms; full case CPU is partitioned exactly."""
    execution = tmp_path / "execution"
    loaded = sealed_phase(execution / "input", {}, 1)
    lazy_fixture(execution / "initializer/work", fallback=True)
    initialized = sealed_phase(execution / "initializer", {}, 7)
    work = execution / "training/work"
    timed_attempts(work, "stages/lbfgs/ATTEMPTS.json", 2, calls=10)
    arm = {"training_cpu_seconds": 2, "refinement_cpu_seconds": 2, "total_cpu_seconds": 11}
    signed(work / "arms/lbfgs.json", arm)
    trained = sealed_phase(
        execution / "training",
        {
            "initializer_cpu_seconds": 9,
            "arm_receipts": {"lbfgs": {"path": "arms/lbfgs.json", "checksum": pilot.digest(arm)}},
        },
        4,
    )
    value: dict[str, Any] = {
        "initializer_cpu_seconds": 9,
        "amortized_preparation_cpu_seconds": 1,
        "peak_rss": {
            "raw_value": 256,
            "raw_unit": "KiB",
            "bytes": 262144,
            "platform": "linux",
            "scope": "worker lifetime, not incremental",
        },
    }
    physical = sealed_phase(execution, value, 14)
    record = {**value, "physical_invocation": {"checksum": pilot.digest(physical)}}
    ledger = accounting.Ledger()
    accounting._confirmation_resource(tmp_path, record, ledger, "case")
    result = ledger.result()
    assert loaded["cpu_seconds"] + initialized["cpu_seconds"] + trained["cpu_seconds"] == 12
    assert result["measured_totals"]["cpu_seconds"]["known_subtotal"] == pytest.approx(14)
    assert result["measured_totals"]["exact_gradient_calls_completed"]["known_subtotal"] == 10
    assert result["measured_totals"]["mps_optimizer_updates"]["known_subtotal"] == 570
    assert result["maximum_reported_worker_peak_rss_bytes"] == 262144
    assert all(event["measurements"]["peak_rss_bytes"] is None for event in result["physical_events"])
    assert result["nonadditive_observations"][-1]["kind"] == "worker_lifetime_peak_rss"


def test_new_panel_incomplete_grid_cannot_emit_subset_total(tmp_path: Path) -> None:
    """A complete status with a missing compiler pair is rejected before any event is added."""
    protocol = {
        "pairs": [{"target_id": "tiny", "replica": 11, "label": "initial"}],
        "targets": ["tiny"],
        "replicas": [11],
        "labels": ["initial", "final"],
    }
    with pytest.raises(ValueError, match="omits a frozen source pair"):
        accounting._collect_new_panel(
            tmp_path,
            "state_compilation",
            protocol,
            {"status": "complete", "case_checksums": {"one": "a"}},
            accounting.Ledger(),
        )


def memory_resource_fixture(root: Path, *, interrupted: bool = False) -> dict[str, Any]:
    """Create a complete one-path audit with a failed numerical gate and measured replay costs.

    Returns:
        A signed endpoint resource receipt with no actual simulator execution.
    """
    phases = [
        ("mps_gradient", 0.7),
        ("dense_gradient", 0.2),
        ("retained_memory_audit", 0.1),
        ("forward_norm_audit", 0.2),
    ]
    stages = [
        {"phase": phase, "status": "complete", "cpu_seconds": cpu, "wall_seconds": cpu + 0.05} for phase, cpu in phases
    ]
    attempts: list[dict[str, Any]] = []
    if interrupted:
        attempts.append({
            "status": "failed",
            "cpu_seconds": 0.4,
            "wall_seconds": 0.5,
            "stages": [
                {"phase": "mps_gradient", "status": "failed_or_interrupted", "cpu_seconds": 0.3, "wall_seconds": 0.35}
            ],
        })
    attempts.append({"status": "complete", "cpu_seconds": 1.5, "wall_seconds": 2.0, "stages": stages})
    payload = {"signature": "memory-measurement", "attempts": attempts}
    signed(root / "ATTEMPTS.json", payload)
    raw: dict[str, Any] = {
        "accepted": False,
        "memory": {
            "forward_tape": {"complete_backing_payload_bytes": 64},
            "exact_target": {"complete_backing_payload_bytes": 16},
            "retained_union": {"complete_backing_payload_bytes": 80},
        },
        "dense_oracle_target_bytes": 32,
        "analytic_no_factor_path": True,
        "selected_path_index": 0,
        "prior_overlap_gate": "failed; unchanged",
        "peak_rss": {"raw_value": 8, "raw_unit": "KiB", "bytes": 8192, "platform": "linux", "scope": "worker lifetime"},
    }
    signed(root / "MEASUREMENT.json", raw)
    record: dict[str, Any] = {
        "version": "overnight.one_path_retained_mps_memory.v1",
        "signature": payload["signature"],
        "status": "complete",
        "accepted": False,
        "measurement": raw,
        "measurement_checksum": pilot.digest(raw),
        "costs": accounting.overlap.costs(payload),
        "dependencies": hash_children(root),
    }
    signed(root / "COMPLETED.json", {"signature": record["signature"], "result_checksum": pilot.digest(record)})
    signed(root / "RESULT.json", record)
    return record


@pytest.mark.parametrize("interrupted", [False, True])
def test_memory_resource_stages_partition_full_cost_and_keep_storage_nonadditive(
    tmp_path: Path, *, interrupted: bool
) -> None:
    """Failed gates and partial gradients retain CPU; tape bytes never masquerade as worker RSS."""
    record = memory_resource_fixture(tmp_path, interrupted=interrupted)
    ledger = accounting.Ledger()
    accounting._mps_memory_resource(tmp_path, record, ledger, "case")
    accounting._mps_memory_resource(tmp_path, record, ledger, "case")
    result = ledger.result()
    assert result["measured_totals"]["cpu_seconds"]["known_subtotal"] == pytest.approx(1.9 if interrupted else 1.5)
    assert result["measured_totals"]["trajectory_paths"]["known_subtotal"] == 2
    assert result["measured_totals"]["attempted_trajectory_paths"]["known_subtotal"] == (3 if interrupted else 2)
    assert result["measured_totals"]["trajectory_direction_passes"]["known_subtotal"] == 4
    assert result["measured_totals"]["trajectory_paths"]["total"] is None
    assert result["maximum_reported_worker_peak_rss_bytes"] == 8192
    assert all(event["measurements"]["peak_rss_bytes"] is None for event in result["physical_events"])
    storage = next(row for row in result["nonadditive_observations"] if row["kind"] == "retained_mps_tensor_storage")
    assert storage["costs"]["retained_union"]["complete_backing_payload_bytes"] == 80
    assert storage["costs"]["accepted"] is False


@pytest.mark.parametrize("damage", ["lost_cost", "unknown_stage", "nested_result", "resigned_acceptance"])
def test_memory_resource_missing_cost_or_changed_child_fails_closed(tmp_path: Path, damage: str) -> None:
    """No missing duration, unsupported work or re-signed public acceptance yields a smaller ledger."""
    record = memory_resource_fixture(tmp_path)
    if damage in {"lost_cost", "unknown_stage"}:
        payload = accounting.read(tmp_path / "ATTEMPTS.json")
        if damage == "lost_cost":
            payload["attempts"][0]["stages"][0]["cpu_seconds"] = None
        else:
            payload["attempts"][0]["stages"][0]["phase"] = "new_uninstrumented_pass"
        signed(tmp_path / "ATTEMPTS.json", payload)
        record["costs"] = accounting.overlap.costs(payload)
        record["dependencies"]["ATTEMPTS.json"] = hashlib.sha256((tmp_path / "ATTEMPTS.json").read_bytes()).hexdigest()
    elif damage == "nested_result":
        signed(tmp_path / "nested/RESULT.json", {"paid": "work"})
    else:
        record["accepted"] = True
    signed(tmp_path / "COMPLETED.json", {"signature": record["signature"], "result_checksum": pilot.digest(record)})
    signed(tmp_path / "RESULT.json", record)
    with pytest.raises(ValueError, match=r"Memory resource|Unknown memory"):
        accounting._mps_memory_resource(tmp_path, record, accounting.Ledger(), "case")


def test_memory_resource_panel_cannot_drop_one_of_eight_endpoints(tmp_path: Path) -> None:
    """A complete summary over a smaller or duplicated source set cannot hide missing diagnostic work."""
    entries = [
        {"target_id": target, "replica": 11, "label": label}
        for target in accounting.overlap.TARGETS
        for label in accounting.overlap.ENDPOINTS
    ]
    protocol: dict[str, Any] = {"input_panel": {"entries": entries[:-1], "files": {}}}
    with pytest.raises(ValueError, match="all eight"):
        accounting._collect_new_panel(
            tmp_path, "mps_memory", protocol, {"cases": [{"checksum": str(i)} for i in range(7)]}, accounting.Ledger()
        )


def test_complete_eight_memory_endpoints_collect_once_with_failed_gate(panel: Path) -> None:
    """The full known schema includes every endpoint even when numerical acceptance fails."""
    source_protocol = accounting.read(panel / "PROTOCOL.json")
    root = panel.parent / "memory-complete"
    entries: list[dict[str, Any]] = [
        {"target_id": target, "replica": 11, "label": label}
        for target in accounting.overlap.TARGETS
        for label in accounting.overlap.ENDPOINTS
    ]
    protocol: dict[str, Any] = {
        "version": "overnight.one_path_retained_mps_memory.v1",
        "benchmark_hashes": source_protocol["benchmark_hashes"],
        "source_hashes": source_protocol["source_hashes"],
        "input_panel": {"entries": entries, "files": {}},
    }
    signed(root / "PROTOCOL.json", protocol)
    checksums = []
    for entry in entries:
        child = root / f"{entry['target_id']}__11" / entry["label"]
        record = memory_resource_fixture(child)
        record["signature"] = pilot.digest([protocol["version"], [protocol, entry]])
        attempts = accounting.read(child / "ATTEMPTS.json")
        attempts["signature"] = record["signature"]
        signed(child / "ATTEMPTS.json", attempts)
        record["dependencies"]["ATTEMPTS.json"] = hashlib.sha256((child / "ATTEMPTS.json").read_bytes()).hexdigest()
        signed(child / "COMPLETED.json", {"signature": record["signature"], "result_checksum": pilot.digest(record)})
        checksums.append(signed(child / "RESULT.json", record))
    signed(
        root / "SUMMARY.json",
        {
            "status": "complete",
            "accepted": False,
            "protocol_checksum": pilot.digest(protocol),
            "cases": [{"checksum": value} for value in checksums],
        },
    )
    result = accounting.collect([root, root])
    assert result["measured_totals"]["cpu_seconds"]["known_subtotal"] == pytest.approx(12)
    assert result["measured_totals"]["trajectory_paths"]["known_subtotal"] == 16
    assert len(result["physical_events"]) == 40
    assert result["maximum_reported_worker_peak_rss_bytes"] == 8192


def test_long_wrapper_and_score_are_disjoint_and_copied_once(tmp_path: Path) -> None:
    """Long training, wrapper overhead and a shared score sum to measured physical work once."""
    root = tmp_path / "long" / "tiny__11"
    signed(root / "mean2_lbfgs" / "ATTEMPTS.json", {"attempts": [exact_attempt(2, 10)]})
    signed(root / "mean2_lbfgs" / "INVOCATIONS.json", {"attempts": [exact_attempt(2.5, 10)]})
    signed(root / "scores" / "one" / "ATTEMPTS.json", {"attempts": [exact_attempt(0.4, 12)]})
    record: dict[str, Any] = {
        "signature": "long-physical",
        "arms": {"mean2_lbfgs": {"training": {"cpu_seconds": 2}}},
        "artifact_hashes": hash_children(root),
    }
    ledger = accounting.Ledger()
    accounting._long_ensemble(root, record, ledger, "parent")
    accounting._long_ensemble(root, record, ledger, "parent")
    result = ledger.result()
    assert result["measured_totals"]["cpu_seconds"]["known_subtotal"] == pytest.approx(2.9)
    assert result["measured_totals"]["exact_gradient_calls_completed"]["known_subtotal"] == 10
    assert result["measured_totals"]["exact_forward_calls_completed"]["known_subtotal"] == 12
    assert result["measured_totals"]["density_direction_passes"]["known_subtotal"] == 32


def test_composition_covers_wrappers_without_recharging_initialization(tmp_path: Path) -> None:
    """An inclusive composed case partitions exact/stochastic/score/compiler work with reused constructor charges."""
    root = tmp_path / "composition" / "tiny__11"
    signed(root / "stages" / "exact" / "ATTEMPTS.json", {"attempts": [exact_attempt(2, 10)]})
    signed(root / "stages" / "exact" / "INVOCATIONS.json", {"attempts": [exact_attempt(2.2, 10)]})
    signed(
        root / "stages" / "weighted" / "ATTEMPTS.json",
        {
            "attempts": [
                {
                    "status": "complete",
                    "cpu_seconds": 3,
                    "wall_seconds": 3.1,
                    "attempted_trajectory_paths": 80,
                    "completed_trajectory_paths": 80,
                    "attempted_trajectory_direction_passes": 160,
                    "completed_trajectory_direction_passes": 160,
                    "attempted_local_operator_applications": 400,
                    "completed_local_operator_applications": 400,
                }
            ]
        },
    )
    signed(root / "operations" / "audit_one.attempts.json", {"attempts": [exact_attempt(1, 12)]})
    signed(root / "operations" / "round0_ranking.attempts.json", {"attempts": [exact_attempt(0.5, 2)]})
    signed(root / "compilations" / "one.attempts.json", {"attempts": [exact_attempt(0.2, 0)]})
    outer = {"attempts": [{"status": "complete", "cpu_seconds": 8, "wall_seconds": 9}]}
    signed(root / "CASE_ATTEMPTS.json", outer)
    record: dict[str, Any] = {
        "signature": "composition-physical",
        "dependencies": hash_children(root),
        "physical_invocation": {"attempts_checksum": pilot.digest(outer)},
        "source": {"initializer_cpu_seconds": 37},
        "arms": {"weighted": {"initializer_cpu_seconds": 37, "training_cpu_seconds": 3, "total_cpu_seconds": 40}},
    }
    ledger = accounting.Ledger()
    accounting._composition(root, record, ledger, "parent")
    result = ledger.result()
    assert result["measured_totals"]["cpu_seconds"]["known_subtotal"] == 8
    assert result["measured_totals"]["wall_seconds"]["known_subtotal"] == 9
    assert result["measured_totals"]["ranking_channel_calls"]["known_subtotal"] == 2
    assert result["measured_totals"]["trajectory_direction_passes"]["known_subtotal"] == 160
    assert result["by_category_cpu_seconds"]["bookkeeping"]["known_subtotal"] == pytest.approx(1.3)
    assert result["complete_development_cpu_seconds"] is None


@pytest.mark.parametrize("physical", [False, True])
def test_mps_failed_gate_and_followup_both_charge_actual_replays(tmp_path: Path, *, physical: bool) -> None:
    """Failed numerical acceptance retains measured gradients, forward paths and wrapper work."""
    root = tmp_path / ("physical" if physical else "mps") / "tiny__11"
    stages = [
        {"phase": "dense_gradient", "status": "complete", "cpu_seconds": 1, "wall_seconds": 1.1},
        {
            "phase": "mps_same_theta_gradient" if physical else "mps_gradient",
            "status": "complete",
            "cpu_seconds": 2,
            "wall_seconds": 2.1,
        },
        {
            "phase": "physical_mps_dense" if physical else "dense_path_metrics",
            "status": "complete",
            "cpu_seconds": 0.4,
            "wall_seconds": 0.5,
        },
        {"phase": "mps_gradient", "status": "failed_or_interrupted", "cpu_seconds": 0.2, "wall_seconds": 0.3},
    ]
    signed(
        root / "ATTEMPTS.json",
        {
            "attempts": [
                {
                    "status": "complete",
                    "cpu_seconds": 4,
                    "wall_seconds": 5,
                    "stages": stages,
                }
            ]
        },
    )
    record: dict[str, Any] = {"signature": "mps-audit-physical", "accepted": False, "dependencies": hash_children(root)}
    settings: dict[str, Any] = {"paths_per_channel": 4}
    protocol = {"settings": {"prior_settings": settings} if physical else settings}
    ledger = accounting.Ledger()
    accounting._mps_diagnostic(root, record, ledger, "parent", protocol, physical=physical)
    result = ledger.result()
    assert result["measured_totals"]["cpu_seconds"]["known_subtotal"] == 4
    assert result["measured_totals"]["wall_seconds"]["known_subtotal"] == 5
    assert result["measured_totals"]["trajectory_paths"]["known_subtotal"] == 24
    assert result["measured_totals"]["trajectory_direction_passes"]["known_subtotal"] == 40
    assert result["measured_totals"]["trajectory_paths"]["total"] is None
    assert result["nonadditive_observations"][-1]["costs"]["accepted"] is False


def test_new_child_index_authenticates_bytes_and_unknown_attempts(tmp_path: Path) -> None:
    """Neither changed raw work nor an unindexed extra attempt disappears from the ledger."""
    signed(tmp_path / "ATTEMPTS.json", {"attempts": [exact_attempt(1, 1)]})
    record = {"dependencies": hash_children(tmp_path)}
    assert accounting._authenticated_children(tmp_path, record, "dependencies") == {"ATTEMPTS.json"}
    signed(tmp_path / "new" / "ATTEMPTS.json", {"attempts": [exact_attempt(1, 1)]})
    with pytest.raises(ValueError, match="omits a durable"):
        accounting._authenticated_children(tmp_path, record, "dependencies")
    record["dependencies"] = hash_children(tmp_path)
    signed(tmp_path / "ATTEMPTS.json", {"attempts": [exact_attempt(2, 1)]})
    with pytest.raises(ValueError, match="bytes changed"):
        accounting._authenticated_children(tmp_path, record, "dependencies")


@pytest.mark.parametrize("cases", [[], [{"target_id": "tiny", "replica": 11}] * 2])
def test_physical_followup_cannot_reduce_grid_to_empty_or_duplicate(
    tmp_path: Path,
    cases: list[dict[str, Any]],
) -> None:
    """An empty or duplicated diagnostic panel cannot count as a complete zero-cost audit."""
    with pytest.raises(ValueError, match="Empty or duplicated"):
        accounting.expected_cases(tmp_path, "mps_physical", {"input_panel": {"cases": cases}})


def test_mps_summary_without_status_accounts_failed_complete_panel(panel: Path) -> None:
    """Known diagnostic summaries use complete cases rather than numerical acceptance as completion."""
    protocol = accounting.read(panel / "PROTOCOL.json")
    protocol.update(
        version="overnight.mps_overlap_development.v1",
        targets=["tiny"],
        replicas=[11],
        settings={"paths_per_channel": 4},
    )
    checksum = signed(panel / "PROTOCOL.json", protocol)
    root = panel / "tiny__11"
    signed(
        root / "ATTEMPTS.json",
        {
            "attempts": [
                {
                    "status": "complete",
                    "cpu_seconds": 1,
                    "wall_seconds": 2,
                    "stages": [
                        {"phase": "dense_gradient", "status": "complete", "cpu_seconds": 0.5, "wall_seconds": 0.6}
                    ],
                }
            ]
        },
    )
    case: dict[str, Any] = {
        "target_id": "tiny",
        "replica": 11,
        "signature": "mps-one",
        "status": "complete",
        "accepted": False,
        "endpoints": {"original108": {}, "structural_mean2_lbfgs": {}},
        "dependencies": hash_children(root),
    }
    case_checksum = signed(root / "RESULT.json", case)
    signed(
        panel / "SUMMARY.json",
        {
            "accepted": False,
            "protocol_checksum": checksum,
            "cases": {"tiny": {"status": "complete", "checksum": case_checksum}},
        },
    )
    result = accounting.collect([panel])
    assert result["measured_totals"]["cpu_seconds"]["known_subtotal"] == 1
    assert result["measured_totals"]["trajectory_direction_passes"]["known_subtotal"] == 16


def verification_fixture(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, call_count: int = 12, failed: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build every frozen role with one shared endpoint and independently timed raw calls.

    Returns:
        Protocol and input records; saved-parent reconstruction is the only stub.
    """
    verifier = accounting.verifier
    parent_root = root.parent / "parent"
    parent: dict[str, Any] = {"benchmark_hashes": {"benchmark": "fixed"}}
    parent_checksum = signed(parent_root / "PROTOCOL.json", parent)
    source = root.parent / "source.py"
    source.write_text("# independent oracle fixture\n")
    sources = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    roles = [
        {
            "target_id": target,
            "replica": replica,
            "panel": panel_name,
            "method": method,
            "endpoint_id": "shared",
            "expected_fidelities": dict.fromkeys(verifier.oracle.PROFILE_IDS, 0.8),
            "case_checksum": "sealed-case",
            "point_checksum": "sealed-point",
        }
        for target in verifier.parent_api.TARGETS
        for replica in verifier.parent_api.REPLICAS
        for panel_name in verifier.PANELS
        for method in verifier.parent_api.METHODS
    ]
    inputs: dict[str, Any] = {
        "parent_root": str(parent_root),
        "parent_checksum": parent_checksum,
        "source_hashes": sources,
        "engineering": False,
        "roles": roles,
        "endpoints": {"shared": {"target_vector_sha256": "same-target", "theta": [0.1]}},
    }
    signed(root / "INPUTS.json", inputs)
    monkeypatch.setattr(verifier, "input_panel", lambda _root: inputs)
    monkeypatch.setattr(accounting.runner, "custody", lambda: {"benchmark_hashes": {"benchmark": "fixed"}})

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Resource accounting must never execute an oracle or calibration.")

    monkeypatch.setattr(verifier.oracle, "evaluate_profile", forbidden)
    monkeypatch.setattr(verifier.oracle, "calibration_profile", forbidden)
    preparation: dict[str, Any] = {
        "signature": pilot.digest([verifier.VERSION, str(parent_root), False]),
        "attempts": [
            {"status": "failed", "cpu_seconds": 1.0, "wall_seconds": 1.0},
            {"status": "complete", "cpu_seconds": 1.0, "wall_seconds": 1.0},
        ],
    }
    prep_checksum = signed(root / "PREPARATION_ATTEMPTS.json", preparation)
    protocol: dict[str, Any] = {
        "version": verifier.VERSION,
        "source_hashes": sources,
        "inputs_checksum": pilot.digest(inputs),
        "input_file_sha256": verifier.file_hash(root / "INPUTS.json"),
        "preparation_attempts_checksum": prep_checksum,
        "preparation_costs": verifier.costs(preparation),
        "engineering": False,
        "profile_ids": list(verifier.oracle.PROFILE_IDS),
        "role_count": 210,
        "row_count": 2520,
        "unique_endpoints": 1,
        "unique_calls": 12,
        "fidelity_tolerance": 1e-10,
        "physicality_tolerance": 1e-10,
        "cpu_admission_cap_seconds": 600,
    }
    checksum = signed(root / "PROTOCOL.json", protocol)
    signed(
        root / "RUN_ATTEMPTS.json",
        {
            "signature": pilot.digest([verifier.VERSION, checksum]),
            "attempts": [
                {"status": "interrupted", "cpu_seconds": 1.0, "wall_seconds": 1.0},
                {"status": "complete", "cpu_seconds": 4.0, "wall_seconds": 4.0},
            ],
        },
    )
    endpoint_root = root / "endpoints" / "shared"
    signed(
        endpoint_root / "ATTEMPTS.json",
        {
            "signature": pilot.digest([checksum, "shared"]),
            "attempts": [{"status": "complete", "cpu_seconds": 4.0, "wall_seconds": 4.0}],
        },
    )
    for index, noise in enumerate(verifier.oracle.PROFILE_IDS[:call_count]):
        path = endpoint_root / "calls" / noise
        identity = [checksum, "shared", noise]
        terminal = "failed" if failed and index == 0 else "complete"
        attempts = [{"status": terminal, "cpu_seconds": 0.1, "wall_seconds": 0.1}]
        if index == 0:
            attempts.insert(0, {"status": "interrupted", "cpu_seconds": 0.2, "wall_seconds": 0.2})
        signed(path / "ATTEMPTS.json", {"signature": pilot.digest(identity), "attempts": attempts})
        measurement: dict[str, Any] = (
            {"status": "failed", "failure": "known numerical failure"}
            if terminal == "failed"
            else {
                "status": "complete",
                "oracle": {
                    "version": verifier.oracle.VERSION,
                    "noise_id": noise,
                    "fidelity": 0.8,
                    "physicality": {
                        "trace_real": 1.0,
                        "trace_imag": 0.0,
                        "hermitian_max_abs": 0.0,
                        "minimum_eigenvalue": 0.0,
                    },
                },
            }
        )
        signed(path / "MEASUREMENT.json", measurement)
        verifier.seal(path, verifier.call_result(path, identity))
    if call_count == 12:
        verifier.seal(endpoint_root, verifier.endpoint_result(endpoint_root, [checksum, "shared"]))
    verifier.seal(root, verifier.summarize(root, protocol, inputs))
    return protocol, inputs


def test_independent_verification_counts_inclusive_work_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """2520 logical rows share 12 physical calls; interrupted replay and failed prep remain paid."""
    root = tmp_path / "verification"
    verification_fixture(root, monkeypatch)
    before = hash_children(root)
    result = accounting.collect([root, root])
    assert hash_children(root) == before
    totals = result["measured_totals"]
    assert totals["cpu_seconds"]["known_subtotal"] == pytest.approx(7)
    assert totals["wall_seconds"]["known_subtotal"] == pytest.approx(7)
    assert totals["exact_forward_calls_attempted"]["known_subtotal"] == 13
    assert totals["exact_forward_calls_completed"]["known_subtotal"] == 12
    assert totals["density_direction_passes"]["known_subtotal"] == 12
    assert totals["density_direction_passes"]["total"] is None
    assert result["maximum_reported_worker_peak_rss_bytes"] is None
    assert result["complete_development_cpu_seconds"] is None
    assert len(result["physical_events"]) == 16


@pytest.mark.parametrize(("call_count", "failed"), [(1, False), (12, True)])
def test_independent_incomplete_or_failed_rows_keep_paid_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call_count: int, *, failed: bool
) -> None:
    """Budget omissions and terminal numerical failures never disappear from rows or compute."""
    root = tmp_path / "verification"
    verification_fixture(root, monkeypatch, call_count=call_count, failed=failed)
    result = accounting.collect([root])
    outcome = result["nonadditive_observations"][-1]["costs"]
    assert outcome["expected_rows"] == 2520
    assert outcome["accepted"] is False
    assert outcome["unique_calls_terminal"] == call_count
    assert outcome["unique_calls_failed"] == int(failed)
    assert result["measured_totals"]["cpu_seconds"]["known_subtotal"] == pytest.approx(7)
    assert result["measured_totals"]["exact_forward_calls_completed"]["known_subtotal"] == call_count - int(failed)


@pytest.mark.parametrize("mutation", ["rows", "unknown_attempt", "lost_cost", "wrong_identity", "negative_residual"])
def test_independent_raw_or_resigned_public_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Fresh checksums cannot erase missing roles, extra work, raw unknown cost or invalid clocks."""
    root = tmp_path / "verification"
    verification_fixture(root, monkeypatch)
    verifier = accounting.verifier
    result = accounting.read(root / "RESULT.json")
    if mutation == "rows":
        result["rows"].pop()
    elif mutation == "unknown_attempt":
        signed(root / "extra" / "ATTEMPTS.json", {"attempts": [exact_attempt(1, 1)]})
    else:
        path = root / "RUN_ATTEMPTS.json"
        attempts = accounting.read(path)
        if mutation == "lost_cost":
            attempts["attempts"][0]["cpu_seconds"] = None
        elif mutation == "wrong_identity":
            attempts["signature"] = "different-physical-run"
        else:
            attempts["attempts"] = [{"status": "complete", "cpu_seconds": 0.1, "wall_seconds": 0.1}]
            result.update(
                run_costs={"cpu_seconds": 0.1, "wall_seconds": 0.1}, costs={"cpu_seconds": 2.1, "wall_seconds": 2.1}
            )
        signed(path, attempts)
    verifier.seal(root, result)
    with pytest.raises(ValueError, match=r"2520-row|unindexed|finite scalar|identity changed|covering timer"):
        accounting.collect([root])
