# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Read-only accounting of durable development work and overlapping procedure costs.

Physical events are counted once. Full-procedure charges, shared references and
covering invocation timers are separate observations, never additional work.
Missing timings are null; sums of available measurements are named subtotals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import independent_confirmation_validation as verifier
from experiments.overnight_state_preparation_20260913_0108 import mps_overlap_experiment as overlap
from experiments.overnight_state_preparation_20260913_0108 import runner

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

VERSION = "overnight.development_resource_accounting.v1"
KINDS = {
    "overnight.exact_structural_lbfgs.v1": "structural",
    "overnight.rank_two_development_panel.v1": "synthesis",
    "yaqs.overnight.svrg_adam.v1": "svrg",
    "overnight.ensemble_objective_comparison.v1": "ensemble",
    "overnight.strong_structural_baselines.v1": "baseline",
    "overnight.pure_geometry_comparison.v1": "geometry",
    "overnight.physical_compilation_panel.v1": "compilation",
    "overnight.ensemble_structural.v1": "ensemble_structural",
    "yaqs.overnight.exact_anchor_diagnostic.v1": "estimator",
    "overnight.legacy_construction_development_batch.v1": "construction",
    "overnight.entropic_risk.v1": "risk",
    "overnight.mean2_stochastic_baselines.v1": "mixture",
    "overnight.initializer_resource_policy_development.v1": "initializer",
    "overnight.long_ensemble_convergence.v1": "long_ensemble",
    "overnight.conditional_structural_composition.v1": "composition",
    "overnight.mps_overlap_development.v1": "mps_overlap",
    "overnight.mps_physical_followup.v1": "mps_physical",
    "overnight.common_control_audit.v1": "common_control",
    "overnight.zero_state_eigenstate_panel.v1": "state_compilation",
    "overnight.lazy_initializer_development_reproduction.v1": "lazy_initializer",
    "overnight.compiled_conditional_composition_development.v1": "compiled_composition",
    "overnight.complete_best_candidate_confirmation.v1": "confirmation",
    "overnight.one_path_retained_mps_memory.v1": "mps_memory",
    "overnight.independent_sealed_confirmation_validation.v1": "independent_validation",
}
NEW_KINDS = {
    "common_control",
    "state_compilation",
    "lazy_initializer",
    "compiled_composition",
    "confirmation",
    "mps_memory",
    "independent_validation",
}
FIELDS = (
    "cpu_seconds",
    "wall_seconds",
    "peak_rss_bytes",
    "exact_gradient_calls_attempted",
    "exact_gradient_calls_completed",
    "exact_forward_calls_attempted",
    "exact_forward_calls_completed",
    "ranking_channel_calls",
    "density_direction_passes",
    "trajectory_paths",
    "trajectory_direction_passes",
    "attempted_trajectory_paths",
    "attempted_trajectory_direction_passes",
    "local_operator_applications_completed",
    "local_operator_applications_attempted",
    "coordinate_sweeps",
    "metric_row_applications",
    "mps_optimizer_updates",
)


def read(path: Path) -> dict[str, Any]:
    """Return one authenticated record; callers release each large case before the next."""
    return cast("dict[str, Any]", pilot.read_record(path))


def sum_known(values: Iterable[float | int | None]) -> dict[str, Any]:
    """Return a measured subtotal, nullable complete sum and explicit missing count."""
    items = list(values)
    known = [value for value in items if value is not None]
    subtotal = math.fsum(known)
    return {
        "known_subtotal": subtotal,
        "total": subtotal if len(known) == len(items) else None,
        "unknown_count": len(items) - len(known),
    }


class Ledger:
    """Collect distinct physical measurements plus nonadditive procedure/reference views."""

    def __init__(self) -> None:
        """Initialize a compact proof index and empty physical event map."""
        self.events: dict[str, dict[str, Any]] = {}
        self.observations: list[dict[str, Any]] = []
        self.proofs: dict[str, dict[str, Any]] = {}
        self.panels: list[dict[str, Any]] = []
        self.panel_override: str | None = None

    def proof(self, path: Path, payload: dict[str, Any]) -> str:
        """Index a checked record without retaining its circuits, arrays or full curves."""
        key = str(path.resolve())
        digest = pilot.digest(payload)
        if key in self.proofs and self.proofs[key]["content_checksum"] != digest:
            msg = f"Resource source changed during accounting: {path}."
            raise ValueError(msg)
        self.proofs[key] = {"content_checksum": digest, "bytes": path.stat().st_size}
        return key

    def add(
        self,
        operation: str,
        panel: str,
        category: str,
        costs: dict[str, Any],
        proofs: Sequence[str],
        *,
        label: str,
        scope: str,
        count_quality: str = "measured",
    ) -> str:
        """Add one physical operation, deduplicating copied or annotation-only reports.

        Raises:
            ValueError: If a numeric measurement is negative, nonfinite or boolean.
        """
        panel = self.panel_override or panel
        normalized = {key: costs.get(key) for key in FIELDS}
        for value in normalized.values():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            ):
                msg = "Resource measurements must be nonnegative finite numbers or explicit null."
                raise ValueError(msg)
        identity = pilot.digest({"operation": operation, "category": category, "measurements": normalized})
        if identity in self.events:
            event = self.events[identity]
            event["panels"] = sorted(set(event["panels"]) | {panel})
            event["proofs"] = sorted(set(event["proofs"]) | set(proofs))
            event["labels"] = sorted(set(event["labels"]) | {label})
        else:
            self.events[identity] = {
                "id": identity,
                "operation": operation,
                "category": category,
                "measurements": normalized,
                "panels": [panel],
                "labels": [label],
                "proofs": sorted(proofs),
                "scope": scope,
                "count_quality": count_quality,
            }
        return identity

    def observe(self, panel: str, label: str, kind: str, costs: dict[str, Any], proofs: Sequence[str]) -> None:
        """Record an explicitly nonadditive charge, reference, or covering timer."""
        self.observations.append({
            "panel": self.panel_override or panel,
            "label": label,
            "kind": kind,
            "physical_additive": False,
            "costs": costs,
            "proofs": list(proofs),
        })

    def result(self) -> dict[str, Any]:
        """Return available measured totals without inventing missing scientific or engineering time."""
        events = sorted(
            self.events.values(), key=lambda row: (row["panels"][0], row["category"], row["labels"][0], row["id"])
        )
        groups = {
            category: [event for event in events if event["category"] == category]
            for category in sorted({event["category"] for event in events})
        }
        totals = {
            key: sum_known(event["measurements"][key] for event in events) for key in FIELDS if key != "peak_rss_bytes"
        }
        peaks = [
            event["measurements"]["peak_rss_bytes"]
            for event in events
            if event["measurements"]["peak_rss_bytes"] is not None
        ]
        peaks.extend(item["costs"]["bytes"] for item in self.observations if item["kind"] == "worker_lifetime_peak_rss")
        return {
            "version": VERSION,
            "status": "audited_available_measurements",
            "physical_events": events,
            "nonadditive_observations": self.observations,
            "source_proofs": self.proofs,
            "panels": self.panels,
            "measured_totals": totals,
            "by_category_cpu_seconds": {
                category: sum_known(event["measurements"]["cpu_seconds"] for event in members)
                for category, members in groups.items()
            },
            "maximum_reported_worker_peak_rss_bytes": max(peaks) if peaks else None,
            "peak_memory_scope": (
                "maximum observed process high-water mark; not sum, incremental job memory or concurrent-machine peak"
            ),
            "complete_development_cpu_seconds": None,
            "unmeasured_engineering": {
                "cpu_seconds": None,
                "wall_seconds": None,
                "scope": (
                    "shell, listing, literature, editing, lint, tests, exports and "
                    "unrecorded failed attempts; never inferred from elapsed clock"
                ),
            },
            "wall_scope": "sum of per-operation elapsed durations is worker-time, not parallel panel makespan",
        }


def _kind(protocol: dict[str, Any]) -> str:
    """Recognize only explicit frozen schema versions and the two original runner panels."""
    tag = protocol.get("version", protocol.get("schema"))
    if tag in KINDS:
        return KINDS[tag]
    if tag is None and protocol.get("protocol", {}).get("name") in {"wave1", "focused_curvature"}:
        return "runner"
    msg = f"Unsupported resource-accounting panel schema: {tag!r}."
    raise ValueError(msg)


def authenticate(root: Path, ledger: Ledger) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Authenticate protocol, benchmark and actual source bytes, including the old estimator snapshot."""
    protocol_path = root / ("BATCH_PROTOCOL.json" if (root / "BATCH_PROTOCOL.json").exists() else "PROTOCOL.json")
    protocol = read(protocol_path)
    kind = _kind(protocol)
    sources = protocol.get("source_hashes", protocol.get("sources"))
    benchmarks = protocol.get("benchmark_hashes")
    if kind == "independent_validation":
        inputs = read(root / "INPUTS.json")
        parent = read(Path(inputs["parent_root"]) / "PROTOCOL.json")
        if pilot.digest(parent) != inputs["parent_checksum"]:
            msg = "Independent validation parent identity changed."
            raise ValueError(msg)
        benchmarks = parent["benchmark_hashes"]
    if not isinstance(sources, dict) or not sources or benchmarks != runner.custody()["benchmark_hashes"]:
        msg = "Resource protocol lacks authenticated source and benchmark fingerprints."
        raise ValueError(msg)
    source_proofs = {}
    for name, expected in sources.items():
        path = runner.REPO / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            snapshot = root / "frozen_driver.py"
            if (
                kind != "estimator"
                or Path(name).name != "estimator_diagnostic.py"
                or not snapshot.is_file()
                or hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected
            ):
                msg = f"Resource numerical source hash mismatch: {name}."
                raise ValueError(msg)
            path = snapshot
        source_proofs[name] = {"sha256": expected, "authenticated_path": str(path.resolve())}
    completion_path = root / (
        "RESULT.json"
        if kind == "independent_validation"
        else "SUMMARY.json"
        if kind in {"construction", "initializer", "composition", "mps_overlap", "mps_physical"} | NEW_KINDS
        else "RESULT.json"
    )
    completion = read(completion_path)
    link = (
        "runtime_hash"
        if kind == "estimator"
        else "runtime_checksum"
        if kind == "runner"
        else "parent_checksum"
        if kind == "confirmation"
        else "protocol_checksum"
    )
    if completion.get(link) != pilot.digest(protocol):
        msg = "Resource completion ledger does not authenticate its frozen protocol."
        raise ValueError(msg)
    ledger.proof(protocol_path, protocol)
    ledger.proof(completion_path, completion)
    ledger.panels.append({
        "root": str(root.resolve()),
        "kind": kind,
        "protocol_checksum": pilot.digest(protocol),
        "completion_checksum": pilot.digest(completion),
        "completion_status": completion.get("status"),
        "source_hash_proofs": source_proofs,
    })
    return kind, protocol, completion


def _runner_costs(record: dict[str, Any], ledger: Ledger, panel: str, proof: str, label: str) -> None:
    """Account the original runner's separate optimization, diagnostic and final-audit timers."""
    state, method = record["state"], record["job"]["config"]["method"]
    operation = record["signature"]
    stochastic = method in {"weighted_adam", "adaptive_online"}
    directions = state["direction_passes"]
    ledger.add(
        operation,
        panel,
        "optimization",
        {
            "cpu_seconds": state["training_cpu_seconds"],
            "wall_seconds": state["training_wall_seconds"],
            "exact_gradient_calls_completed": state["exact_training_evaluations"],
            "trajectory_paths": state["trajectory_paths"],
            "trajectory_direction_passes": directions if stochastic else 0,
            "density_direction_passes": 0 if stochastic else directions,
            "coordinate_sweeps": state["iterations"] if method == "coordinate" else 0,
            "peak_rss_bytes": record.get("peak_rss_bytes"),
        },
        [proof],
        label=label,
        scope=(
            "saved optimization timer excludes checkpoint I/O; discarded uncheckpointed attempts were not instrumented"
        ),
        count_quality="retained completed work; attempted/replay counts unavailable",
    )
    diagnostic_calls = 1 if method in {"lbfgs", "coordinate", "unchanged"} else len(state["curve"])
    ledger.add(
        operation,
        panel,
        "diagnostic",
        {
            "cpu_seconds": state["diagnostic_cpu_seconds"],
            "exact_gradient_calls_completed": diagnostic_calls,
            "density_direction_passes": 2 * diagnostic_calls,
        },
        [proof],
        label=label,
        scope=(
            "initial and required checkpoint exact density-loss-gradient diagnostics; cached "
            "loss annotations add no calls"
        ),
        count_quality="derived from authenticated runner control flow and retained curve",
    )
    ledger.add(
        operation,
        panel,
        "evaluation",
        {
            "cpu_seconds": record["evaluation_cpu_seconds"],
            "exact_forward_calls_completed": 13,
            "density_direction_passes": 13,
        },
        [proof],
        label=label,
        scope="all12 endpoint channels plus separate independent local Qiskit audit",
        count_quality="derived from authenticated run_job",
    )
    ledger.observe(
        panel,
        label,
        "covering_invocation",
        {"wall_seconds": record.get("invocation_wall_seconds"), "cpu_seconds": None},
        [proof],
    )
    ledger.observe(
        panel,
        label,
        "full_procedure_charge",
        {"refinement_cpu_seconds": state["training_cpu_seconds"], "historical_construction_cpu_seconds": None},
        [proof],
    )


def _estimator(root: Path, protocol: dict[str, Any], completion: dict[str, Any], ledger: Ledger) -> None:
    """Charge both actual diagnostic executions without treating the rerun as replication."""
    panel = root.name
    seen_anchors, seen_settings = set(), set()
    for target in protocol["targets"]:
        path = root / f"{target}__anchor.json"
        anchor = read(path)
        proof = ledger.proof(path, anchor)
        seen_anchors.add(pilot.digest(anchor))
        if anchor["runtime_hash"] != pilot.digest(protocol):
            msg = "Estimator anchor protocol changed."
            raise ValueError(msg)
        estimate = anchor["snapshot"]["estimate"]
        operation = pilot.digest([anchor["runtime_hash"], target, "anchor"])
        ledger.add(
            operation,
            panel,
            "diagnostic",
            {
                "cpu_seconds": estimate["cpu_seconds"],
                "wall_seconds": estimate["wall_seconds"],
                "exact_gradient_calls_completed": 1,
                "density_direction_passes": 2,
            },
            [proof],
            label=target + "/snapshot",
            scope="one exact anchor physically executed; all setting/replica uses reuse it",
        )
        ledger.add(
            operation + "/setup",
            panel,
            "setup",
            {"cpu_seconds": anchor["shared_backend_setup_cpu_seconds"]},
            [proof],
            label=target,
            scope="shared backend setup physically once",
        )
        for index, distance in enumerate(protocol["distances"]):
            path = root / f"{target}__distance_{index}.json"
            record = read(path)
            proof = ledger.proof(path, record)
            seen_settings.add(pilot.digest(record))
            if (
                record["runtime_hash"] != pilot.digest(protocol)
                or record["anchor_checksum"] != pilot.digest(anchor)
                or record["distance"] != distance
                or record["target_id"] != target
                or len(record["state"]["replications"]) != protocol["replicates"]
            ):
                msg = "Estimator setting, anchor or replicate panel is incomplete or changed."
                raise ValueError(msg)
            state = record["state"]
            operation = pilot.digest([record["runtime_hash"], state["signature"]])
            ledger.add(
                operation + "/oracle",
                panel,
                "diagnostic",
                {
                    "cpu_seconds": state["oracle_cpu_seconds"],
                    "exact_gradient_calls_completed": state["oracle_exact_evaluations"],
                    "density_direction_passes": 2 * state["oracle_exact_evaluations"],
                },
                [proof],
                label=f"{target}/{distance}/oracle",
                scope="zero at anchor means exact cached reuse, not missing timing",
            )
            for method in ("cv", "weighted"):
                rows = [replicate[method] for replicate in state["replications"]]
                ledger.add(
                    operation + "/" + method,
                    panel,
                    "diagnostic",
                    {
                        "cpu_seconds": sum(row["cpu_seconds"] for row in rows),
                        "wall_seconds": sum(row["wall_seconds"] for row in rows),
                        "trajectory_paths": sum(row["paths"] for row in rows),
                        "trajectory_direction_passes": sum(row["directional_passes"] for row in rows),
                        "exact_gradient_calls_completed": 0,
                    },
                    [proof],
                    label=f"{target}/{distance}/{method}",
                    scope="all paired stochastic diagnostic estimates, including zero-distance and unused settings",
                )
    if seen_anchors != set(completion["anchor_checksums"]) or seen_settings != {
        row["run_checksum"] for row in completion["settings"]
    }:
        msg = "Estimator completion ledger omits or changes a complete setting."
        raise ValueError(msg)
    ledger.observe(
        panel,
        "same original scientific streams",
        "rerun_interpretation",
        {
            "independent_replication": False,
            "physical_execution_charged": True,
            "reason": "both original and typing-only driver reruns are real work; copied summaries are not work",
        },
        [str((root / "PROTOCOL.json").resolve())],
    )


def checked(path: Path, ledger: Ledger, expected: str | None = None) -> tuple[dict[str, Any], str]:
    """Read a signed child and, when provided, enforce its parent's content checksum."""
    record = read(path)
    if expected is not None and pilot.digest(record) != expected:
        msg = f"Resource child checksum disagrees with its parent: {path}."
        raise ValueError(msg)
    return record, ledger.proof(path, record)


def _sum(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    """Sum a fully measured field, preserving any unknown attempt."""
    return sum_known(row.get(key) for row in rows)["total"]


def residual(total: float | None, parts: Sequence[float | None]) -> float | None:
    """Subtract nonoverlapping inner timers, rejecting material scope inconsistencies."""
    if total is None or any(part is None for part in parts):
        return None
    value = total - math.fsum(cast("Sequence[float]", parts))
    if value < -1e-7:
        msg = "Inner resource timers exceed their covering timer."
        raise ValueError(msg)
    return max(0.0, value)


def _attempts(path: Path, ledger: Ledger, expected: str | None = None) -> tuple[dict[str, Any], str]:
    """Authenticate every durable attempt, retaining failed or unfinished entries."""
    payload, proof = checked(path, ledger, expected)
    if not isinstance(payload.get("attempts"), list) or not payload["attempts"]:
        msg = f"Missing durable resource attempts: {path}."
        raise ValueError(msg)
    return payload, proof


def _declare(
    ledger: Ledger, panel: str, label: str, arm: dict[str, Any], proof: str, *, reference: bool = False
) -> None:
    """Keep procedure charges and reused training entirely outside physical sums."""
    ledger.observe(
        panel,
        label,
        "reused_reference" if reference else "full_procedure_charge",
        {
            "refinement_cpu_seconds": arm.get("cpu_seconds", arm.get("training_cpu_seconds")),
            "historical_construction_cpu_seconds": None,
            "reference_only_no_fresh_training": reference,
        },
        [proof],
    )


def _endpoint(
    ledger: Ledger, panel: str, operation: str, label: str, arm: dict[str, Any], proof: str, calls: int
) -> None:
    """Charge a separately executed endpoint evaluation, never its copied reference."""
    completed = arm.get("status", "complete") == "complete" and len(arm.get("fidelities", {})) == 12
    ledger.add(
        operation,
        panel,
        "evaluation",
        {
            "cpu_seconds": arm.get("evaluation_cpu_seconds"),
            "exact_forward_calls_completed": calls if completed else None,
            "density_direction_passes": calls if completed else None,
        },
        [proof],
        label=label,
        scope=f"fresh all12 evaluation; {calls - 12} additional independent local audit calls",
        count_quality="derived from verified successful driver flow; failures retain unknown partial counts",
    )


def _svrg(root: Path, record: dict[str, Any], ledger: Ledger, proof: str) -> None:
    """Partition invocation timers and charge all saved stochastic and exact attempts."""
    payload, ap = _attempts(root / "ATTEMPTS.json", ledger, record["attempts_checksum"])
    for index, attempt in enumerate(payload["attempts"]):
        counters = attempt["counters"]
        operation = pilot.digest([record["signature"], payload["signature"], index, attempt])
        done = attempt["status"] == "complete"
        label = root.name + f"/attempt{index}"
        ledger.add(
            operation,
            root.parent.name,
            "optimization",
            {
                "cpu_seconds": counters["training_cpu_seconds"],
                "wall_seconds": counters["training_wall_seconds"],
                "exact_gradient_calls_attempted": counters["exact_gradient_calls"],
                "exact_gradient_calls_completed": counters["exact_gradient_calls"] if done else None,
                "trajectory_paths": counters["trajectory_paths"] if done else None,
                "trajectory_direction_passes": counters["trajectory_direction_passes"] if done else None,
                "density_direction_passes": counters["density_gradient_direction_passes"] if done else None,
                "peak_rss_bytes": record.get("peak_rss_bytes"),
            },
            [proof, ap],
            label=label,
            scope="includes persistent exact anchor refresh and every correction/base propagation",
            count_quality="pre-call attempted counters; completed only when whole attempt completed",
        )
        for category, prefix in (("diagnostic", "diagnostic"), ("evaluation", "evaluation")):
            ledger.add(
                operation,
                root.parent.name,
                category,
                {
                    "cpu_seconds": counters[prefix + "_cpu_seconds"],
                    "exact_forward_calls_attempted": counters[prefix + "_forward_calls"],
                    "exact_forward_calls_completed": counters[prefix + "_forward_calls"] if done else None,
                    "density_direction_passes": counters[prefix + "_forward_calls"] if done else None,
                },
                [proof, ap],
                label=label,
                scope="separate exact forward scores including discarded attempt work",
            )
        ledger.add(
            operation,
            root.parent.name,
            "bookkeeping",
            {
                "cpu_seconds": residual(
                    attempt.get("invocation_cpu_seconds_including_io"),
                    [
                        counters["training_cpu_seconds"],
                        counters["diagnostic_cpu_seconds"],
                        counters["evaluation_cpu_seconds"],
                    ],
                ),
            },
            [ap],
            label=label,
            scope="invocation CPU minus disjoint optimization, diagnostic and evaluation timers",
        )
        ledger.observe(
            root.parent.name,
            label,
            "covering_invocation",
            {
                "cpu_seconds": attempt.get("invocation_cpu_seconds_including_io"),
                "wall_seconds": attempt.get("invocation_wall_seconds_including_io"),
                "reserved_trajectory_paths": counters["trajectory_paths"],
                "exact_anchor_calls": counters["exact_anchor_calls"],
            },
            [ap],
        )


def _score_cache(root: Path, record: dict[str, Any], ledger: Ledger, *, geometry: bool) -> None:
    """Count unique cached parameter evaluations once across endpoint and CPU views."""
    summary = record["scoring_ledger" if geometry else "post_training_scoring"]
    expected, found = set(summary["checksums"]), set()
    for path in sorted((root / "scores").glob("*.json")):
        score, proof = checked(path, ledger)
        digest = pilot.digest(score)
        if digest not in expected:
            msg = f"Unindexed score-cache resource record: {path}."
            raise ValueError(msg)
        found.add(digest)
        ledger.add(
            score["signature"],
            root.parent.name,
            "evaluation",
            {
                "cpu_seconds": score["cpu_seconds"],
                "exact_forward_calls_attempted": score.get("physical_attempted_forward_calls"),
                "exact_forward_calls_completed": score.get(
                    "physical_completed_forward_calls", score["forward_channel_evaluations"]
                ),
                "density_direction_passes": score.get(
                    "physical_completed_forward_calls", score["forward_channel_evaluations"]
                ),
            },
            [proof],
            label=root.name + "/" + path.stem,
            scope=(
                "unique score-cache execution reused by final and common-CPU points; includes "
                "recorded failed channel attempts"
            ),
        )
    if expected != found:
        msg = "Resource score-cache panel is incomplete."
        raise ValueError(msg)


def _ensemble(root: Path, record: dict[str, Any], ledger: Ledger, proof: str, *, geometry: bool = False) -> None:
    """Read optimization attempts and cached scores for exact ensemble or geometry panels."""
    for method, arm in sorted(record["arms"].items()):
        train = arm["training"]
        attempts, ap = _attempts(root / method / "ATTEMPTS.json", ledger, train["attempts_checksum"])
        if attempts["signature"] != train["signature"]:
            msg = "Optimization attempt identity differs from final state."
            raise ValueError(msg)
        for index, attempt in enumerate(attempts["attempts"]):
            before, after = (
                ("channel_attempted", "channel_completed")
                if geometry
                else ("attempted_channel_calls", "completed_channel_calls")
            )
            ledger.add(
                pilot.digest([train["signature"], index, attempt]),
                root.parent.name,
                "optimization",
                {
                    "cpu_seconds": attempt.get("cpu_seconds"),
                    "wall_seconds": attempt.get("wall_seconds"),
                    "exact_gradient_calls_attempted": attempt[before],
                    "exact_gradient_calls_completed": attempt[after],
                    "density_direction_passes": 2 * attempt[after],
                    "metric_row_applications": attempt.get("statevector_row_operator_applications", 0),
                    "trajectory_paths": 0,
                    "trajectory_direction_passes": 0,
                },
                [proof, ap],
                label=f"{root.name}/{method}/attempt{index}",
                scope=(
                    "all optimization attempts including failed/replayed calls and checkpoint I/O; "
                    "metric and solve timers are nested"
                ),
            )
            if geometry:
                ledger.observe(
                    root.parent.name,
                    method,
                    "nested_metric_breakdown",
                    {
                        key: value
                        for key, value in attempt.items()
                        if key.startswith(("metric_", "solve_", "channel_"))
                        or key in {"linear_solves", "symmetric_eigensolves", "statevector_row_operator_applications"}
                    },
                    [ap],
                )
        _declare(ledger, root.parent.name, root.name + "/" + method, train, proof)
    _score_cache(root, record, ledger, geometry=geometry)
    if not geometry:
        diagnostic, dp = checked(root / "START_GRADIENT_DIAGNOSTIC.json", ledger)
        ledger.add(
            diagnostic["signature"],
            root.parent.name,
            "diagnostic",
            {
                "cpu_seconds": diagnostic["cpu_seconds"],
                "exact_gradient_calls_completed": 11,
                "density_direction_passes": 22,
            },
            [dp],
            label=root.name,
            scope="all11 initial exact gradient diagnostic, including unused objective comparisons",
        )


def _structural(root: Path, record: dict[str, Any], ledger: Ledger, proof: str) -> None:
    """Account a shared local warmup once and preserve whole-search replay timers."""
    pair, arms = record["pair"], record["arms"]
    warm, wp = checked(root / "branches" / "warmup.json", ledger)
    attempt_proofs = [proof, wp]
    all_complete = True
    wall_parts = []
    for path in sorted((root / "branches").glob("*.attempts.json")):
        attempts, ap = _attempts(path, ledger)
        attempt_proofs.append(ap)
        all_complete = all_complete and all(row["completed"] for row in attempts["attempts"])
        if path.stem.removesuffix(".attempts") in {"warmup", "fixed", "polish"}:
            wall_parts.append(_sum(attempts["attempts"], "wall_seconds"))
    search, sp = _attempts(root / "SEARCH_ATTEMPTS.json", ledger)
    attempt_proofs.append(sp)
    wall_parts.append(_sum(search["attempts"], "wall_seconds"))
    all_complete = all_complete and len(search["attempts"]) == 1
    structural, fixed = arms["structural_lbfgs"], arms["fixed_lbfgs"]
    gradients = (
        structural["physical_joint_gradient_calls"]
        + fixed["physical_joint_gradient_calls"]
        - warm["physical_joint_gradient_calls"]
    )
    forward, ranking = structural["forward_score_calls"], structural["deletion_ranking_calls"]
    ledger.add(
        record["signature"],
        root.parent.name,
        "optimization",
        {
            "cpu_seconds": pair["physical_training_cpu_seconds"],
            "wall_seconds": sum_known(wall_parts)["total"],
            "exact_gradient_calls_attempted": gradients,
            "exact_gradient_calls_completed": gradients if all_complete else None,
            "exact_forward_calls_completed": forward if all_complete else None,
            "ranking_channel_calls": ranking if all_complete else None,
            "density_direction_passes": 2 * (gradients + ranking) + forward if all_complete else None,
            "trajectory_paths": 0,
            "trajectory_direction_passes": 0,
        },
        attempt_proofs,
        label=root.name,
        scope=(
            "structural plus fixed tail plus one shared warmup; whole-search timer replaces "
            "nested repair timers, including replay/cache I/O"
        ),
        count_quality=(
            "gradient attempted counters; search counts from retained ledger, exact only absent failed attempts"
        ),
    )
    for method, arm in sorted(arms.items()):
        _endpoint(
            ledger, root.parent.name, record["signature"] + "/" + method, root.name + "/" + method, arm, proof, 12
        )
        _declare(
            ledger,
            root.parent.name,
            root.name + "/" + method,
            arm,
            proof,
            reference=method not in {"structural_lbfgs", "fixed_lbfgs"},
        )
    ledger.observe(
        root.parent.name,
        root.name,
        "shared_warmup_charge",
        {
            "physical_executions": 1,
            "charged_arms": 2,
            "cpu_seconds": pair["shared_warmup_cpu_seconds"],
            "joint_gradient_calls": pair["shared_warmup_calls"],
        },
        [wp, proof],
    )


def _ensemble_structural(root: Path, record: dict[str, Any], ledger: Ledger, proof: str) -> None:
    """Count primitive stage/operation attempts, with separate invocation overhead."""
    indexed: dict[str, str] = {}
    for method, arm in sorted(record["arms"].items()):
        indexed.update(arm["physical_work"]["raw_ledgers"])
        _declare(ledger, root.parent.name, root.name + "/" + method, arm, proof)
    actual = {str(path.relative_to(root)) for path in (root / "operations").glob("*.attempts.json")}
    actual |= {str(path.relative_to(root)) for path in (root / "stages").glob("*/ATTEMPTS.json")}
    actual |= {str(path.relative_to(root)) for path in (root / "stages").glob("*/INVOCATIONS.json")}
    if set(indexed) != actual:
        msg = "Ensemble-structural resource ledger omits a physical stage or operation."
        raise ValueError(msg)
    for relative, expected in sorted(indexed.items()):
        if relative.endswith("INVOCATIONS.json"):
            continue
        path = root / relative
        payload, ap = _attempts(path, ledger, expected)
        stage = relative.startswith("stages/")
        ranking = path.name.endswith("_ranking.attempts.json")
        evaluation = path.name.startswith(("final_", "common_"))
        for index, attempt in enumerate(payload["attempts"]):
            done = attempt.get("status") != "running"
            attempted = attempt.get("attempted_channel_calls") if done else None
            completed = attempt.get("completed_channel_calls") if done else None
            costs = {
                "cpu_seconds": attempt.get("cpu_seconds") if done else None,
                "wall_seconds": attempt.get("wall_seconds") if done else None,
                "exact_gradient_calls_attempted": attempted if stage else 0,
                "exact_gradient_calls_completed": completed if stage else 0,
                "exact_forward_calls_attempted": attempted if not stage and not ranking else 0,
                "exact_forward_calls_completed": completed if not stage and not ranking else 0,
                "ranking_channel_calls": completed if ranking else 0,
                "density_direction_passes": (2 if stage or ranking else 1) * completed
                if completed is not None
                else None,
                "trajectory_paths": 0,
                "trajectory_direction_passes": 0,
            }
            ledger.add(
                pilot.digest([record["signature"], relative, index, attempt]),
                root.parent.name,
                "evaluation" if evaluation else "optimization",
                costs,
                [proof, ap],
                label=root.name + "/" + relative + f"/{index}",
                scope=(
                    "durable primitive attempt, including failed branches and interrupted work; "
                    "completed directions only"
                ),
            )
        if stage:
            marker_path = path.parent / "INVOCATIONS.json"
            marker, mp = _attempts(marker_path, ledger, indexed[str(marker_path.relative_to(root))])
            ledger.add(
                record["signature"] + "/" + relative,
                root.parent.name,
                "bookkeeping",
                {
                    "cpu_seconds": residual(
                        _sum(marker["attempts"], "cpu_seconds"), [_sum(payload["attempts"], "cpu_seconds")]
                    ),
                    "wall_seconds": residual(
                        _sum(marker["attempts"], "wall_seconds"), [_sum(payload["attempts"], "wall_seconds")]
                    ),
                },
                [ap, mp],
                label=root.name + "/" + relative,
                scope="outer stage invocation minus its optimizer attempt timers, including final checkpoint I/O",
            )


def _baseline(root: Path, record: dict[str, Any], ledger: Ledger, proof: str) -> None:
    """Keep fresh strong-baseline training distinct from archived exact-LBFGS references."""
    methods = (
        "structural_exact_adam",
        "structural_weighted_adam",
        "structural_adaptive_online",
        "fixed_adaptive_online",
    )
    arms = record["arms"]
    warm, wp = checked(root / "adaptive_online" / "blocks" / "warmup.json", ledger)
    proofs, walls = [proof, wp], []
    complete = True
    for method in ("exact_adam", "weighted_adam", "adaptive_online"):
        base = root / method
        initial, ip = checked(base / "INITIAL.json", ledger)
        proofs.append(ip)
        # INITIAL exposes CPU only, so complete optimization wall remains unknown.
        for path in sorted((base / "blocks").glob("*.attempts.json")):
            attempts, ap = _attempts(path, ledger)
            proofs.append(ap)
            complete = complete and all(row["completed"] for row in attempts["attempts"])
            if path.name in {"warmup.attempts.json", "polish.attempts.json", "fixed_tail.attempts.json"}:
                walls.append(_sum(attempts["attempts"], "wall_seconds"))
        search, sp = _attempts(base / "SEARCH_ATTEMPTS.json", ledger)
        proofs.append(sp)
        complete = complete and all(row["completed"] for row in search["attempts"])
        walls.append(_sum(search["attempts"], "wall_seconds"))
        ledger.observe(
            root.parent.name,
            root.name + "/" + method,
            "initial_score",
            {
                "cpu_seconds": initial["cpu_seconds"],
                "exact_forward_calls": 1,
            },
            [ip],
        )
    work = {
        key: sum(arms[method]["work"].get(key, 0) for method in methods)
        for key in (
            "exact_gradient_calls",
            "exact_score_calls",
            "trajectory_paths",
            "trajectory_direction_passes",
            "deletion_ranking_calls",
            "density_direction_passes",
        )
    }
    for key in ("exact_gradient_calls", "exact_score_calls", "trajectory_paths", "trajectory_direction_passes"):
        work[key] -= warm[key]
    work["exact_score_calls"] -= 1
    work["density_direction_passes"] -= 2 * warm["exact_gradient_calls"] + warm["exact_score_calls"] + 1
    ledger.add(
        record["signature"],
        root.parent.name,
        "optimization",
        {
            "cpu_seconds": sum(arms[method]["cpu_seconds"] for method in methods)
            - record["shared_adaptive_warmup_cpu_seconds"],
            "exact_gradient_calls_attempted": work["exact_gradient_calls"] if complete else None,
            "exact_gradient_calls_completed": work["exact_gradient_calls"] if complete else None,
            "exact_forward_calls_completed": work["exact_score_calls"] if complete else None,
            "ranking_channel_calls": work["deletion_ranking_calls"] if complete else None,
            "trajectory_paths": work["trajectory_paths"] if complete else None,
            "trajectory_direction_passes": work["trajectory_direction_passes"] if complete else None,
            "density_direction_passes": work["density_direction_passes"] if complete else None,
        },
        proofs,
        label=root.name,
        scope="four fresh arms, one shared adaptive warmup; whole-search CPU replaces nested branch CPU",
        count_quality=(
            "complete paths are exact; any failed block converts reserved work to a separate upper-bound declaration"
        ),
    )
    ledger.observe(
        root.parent.name,
        root.name,
        "reserved_work_and_partial_wall",
        {
            **work,
            "count_interpretation": "exact" if complete else "attempted_or_reserved_upper_bound",
            "known_partial_wall_seconds": sum_known(walls)["known_subtotal"],
            "wall_seconds": None,
            "shared_warmup_cpu_seconds": record["shared_adaptive_warmup_cpu_seconds"],
        },
        proofs,
    )
    for method, arm in sorted(arms.items()):
        if method in methods:
            _endpoint(
                ledger,
                root.parent.name,
                record["signature"] + "/final/" + method,
                root.name + "/" + method,
                arm,
                proof,
                13,
            )
        _declare(ledger, root.parent.name, root.name + "/" + method, arm, proof, reference=method not in methods)
    for method, arm in sorted(record["common_cpu"]["arms"].items()):
        _endpoint(
            ledger,
            root.parent.name,
            record["signature"] + "/common/" + method,
            root.name + "/common/" + method,
            arm,
            proof,
            13,
        )
    for method, arm in sorted(record["historical_unmatched"].items()):
        _declare(ledger, root.parent.name, root.name + "/historical/" + method, arm, proof, reference=True)


def _synthesis(root: Path, record: dict[str, Any], ledger: Ledger, proof: str, protocol: dict[str, Any]) -> None:
    """Count one physical synthesis and fresh runner attempts, excluding reused references."""
    construction, cp = checked(root / "CONSTRUCTION.json", ledger)
    ledger.add(
        construction["signature"],
        root.parent.name,
        "construction",
        {
            "cpu_seconds": construction["physical_cpu_seconds"],
            "wall_seconds": construction["physical_wall_seconds"],
        },
        [cp],
        label=root.name,
        scope="one physical rank-two synthesis shared by three declared procedures",
    )
    methods = set(protocol["spec"]["methods"])
    for method, arm in sorted(record["arms"].items()):
        label = root.name + "/" + method
        if method not in methods:
            _declare(ledger, root.parent.name, label, arm, proof, reference=True)
            continue
        raw, rp = checked(root / (method + ".json"), ledger, arm["record_checksum"])
        _runner_costs(raw, ledger, root.parent.name, rp, label)
        attempts, ap = _attempts(root / (method + ".attempts.json"), ledger)
        inner = [
            raw["state"]["training_cpu_seconds"],
            raw["state"]["diagnostic_cpu_seconds"],
            raw["evaluation_cpu_seconds"],
        ]
        ledger.add(
            raw["signature"],
            root.parent.name,
            "bookkeeping",
            {
                "cpu_seconds": residual(_sum(attempts["attempts"], "cpu_seconds"), inner),
            },
            [rp, ap],
            label=label,
            scope=(
                "all outer invocation attempts minus retained runner timers; includes I/O and any "
                "discarded replay work, whose call counts are unknown"
            ),
        )
        ledger.observe(
            root.parent.name,
            label,
            "covering_invocation",
            {
                "cpu_seconds": _sum(attempts["attempts"], "cpu_seconds"),
                "wall_seconds": _sum(attempts["attempts"], "wall_seconds"),
                "construction_cpu_seconds_charged": construction["physical_cpu_seconds"],
            },
            [ap, cp],
        )


def _compilation(root: Path, record: dict[str, Any], ledger: Ledger, proof: str) -> None:
    """Charge new compilation and paired evaluations without charging source training again."""
    operation, panel = record["signature"], root.name
    label = record["target_id"] + "/" + record["source_label"]
    if record["status"] != "complete":
        ledger.add(
            operation,
            panel,
            "compilation",
            {
                "cpu_seconds": record.get("invocation_cpu_seconds"),
                "wall_seconds": record.get("invocation_wall_seconds"),
            },
            [proof],
            label=label,
            scope=(
                "failed compilation invocation retained; unavailable internal breakdown and channel counts remain null"
            ),
        )
        return
    compile_cpu = record["compilation"]["compilation_cpu_seconds"]
    ledger.add(
        operation,
        panel,
        "compilation",
        {"cpu_seconds": compile_cpu},
        [proof],
        label=label,
        scope="physical gate rewrites only; no variational refinement",
    )
    ledger.add(
        operation,
        panel,
        "diagnostic",
        {"cpu_seconds": record["ideal_audit_cpu_seconds"]},
        [proof],
        label=label,
        scope="two independent ideal statevector preparations; not density-channel or trajectory gradients",
    )
    for method, arm in sorted(record["arms"].items()):
        _endpoint(ledger, panel, operation + "/" + method, label + "/" + method, arm, proof, 13)
    ledger.add(
        operation,
        panel,
        "bookkeeping",
        {
            "cpu_seconds": residual(
                record["invocation_cpu_seconds"],
                [
                    compile_cpu,
                    record["ideal_audit_cpu_seconds"],
                    *(arm["evaluation_cpu_seconds"] for arm in record["arms"].values()),
                ],
            ),
        },
        [proof],
        label=label,
        scope="measured invocation minus compiler, audit and two endpoint evaluation timers",
    )
    ledger.observe(
        panel,
        label,
        "covering_invocation",
        {
            "cpu_seconds": record["invocation_cpu_seconds"],
            "wall_seconds": record["invocation_wall_seconds"],
        },
        [proof],
    )
    ledger.observe(panel, label, "reused_reference", record["source_costs"], [proof])


def _construction(root: Path, record: dict[str, Any], ledger: Ledger, proof: str) -> None:
    """Charge fresh MPS reproduction once; keep original historical CPU unknown."""
    result, rp = checked(root / "RESULT.json", ledger, record["construction_checksum"])
    payload, ap = _attempts(root / "ATTEMPTS.json", ledger, result["attempts_checksum"])
    for index, attempt in enumerate(payload["attempts"]):
        ledger.add(
            pilot.digest([payload["signature"], index, attempt]),
            root.parent.name,
            "construction",
            {
                "cpu_seconds": attempt.get("cpu_seconds"),
                "wall_seconds": attempt.get("wall_seconds"),
                "mps_optimizer_updates": result["updates"]
                if len(payload["attempts"]) == 1 and attempt["status"] == "complete"
                else None,
            },
            [proof, rp, ap],
            label=root.name + f"/attempt{index}",
            scope=(
                "physical full MPS construction including its internal final bond/fidelity "
                "diagnostic and every caught attempt"
            ),
        )
    ledger.observe(
        root.parent.name,
        root.name,
        "nested_construction_diagnostic",
        {
            "cpu_seconds": result["diagnostic_cpu_seconds"],
            "final_forward_peak_bond": record["final_forward_peak_bond"],
            "historical_construction_cpu_seconds": None,
        },
        [proof, rp],
    )
    ledger.add(
        record["signature"],
        root.parent.name,
        "diagnostic",
        {
            "cpu_seconds": record["state_audit_cpu_seconds"],
        },
        [proof],
        label=root.name,
        scope="new independent state comparison against archived endpoint",
    )
    ledger.add(
        record["signature"],
        root.parent.name,
        "bookkeeping",
        {
            "cpu_seconds": residual(
                record["wrapper_cpu_seconds"], [record["construction_cpu_seconds"], record["state_audit_cpu_seconds"]]
            ),
            "wall_seconds": residual(record["wrapper_wall_seconds"], [record["construction_wall_seconds"]]),
        },
        [proof],
        label=root.name,
        scope="wrapper outside construction; wall includes audit whose standalone wall was not measured",
    )
    ledger.observe(
        root.parent.name,
        root.name,
        "historical_constructor_reference",
        {
            "historical_cpu_seconds": record["historical_cpu_seconds"],
            "fresh_reproduction_cpu_seconds": record["construction_cpu_seconds"],
            "reference_archive_checksum": record["archive_checksum"],
            "rule": (
                "new measured reproduction is a possible declared constructor charge, "
                "never an imputed old physical execution"
            ),
        },
        [proof],
    )


def _density_attempt(
    root: Path,
    case: dict[str, Any],
    relative: str,
    payload: dict[str, Any],
    proof: str,
    ledger: Ledger,
    *,
    gradient: bool = False,
    ranking: bool = False,
    evaluation: bool = False,
) -> None:
    """Account durable exact-channel attempts with honest completed propagation counts."""
    for index, attempt in enumerate(payload["attempts"]):
        known = attempt.get("status") != "running"
        completed = attempt.get("completed_channel_calls") if known else None
        attempted = attempt.get("attempted_channel_calls") if known else None
        ledger.add(
            pilot.digest([case["signature"], relative, index, attempt]),
            root.parent.name,
            "evaluation" if evaluation else "optimization",
            {
                "cpu_seconds": attempt.get("cpu_seconds") if known else None,
                "wall_seconds": attempt.get("wall_seconds") if known else None,
                "exact_gradient_calls_attempted": attempted if gradient else 0,
                "exact_gradient_calls_completed": completed if gradient else 0,
                "exact_forward_calls_attempted": attempted if not gradient and not ranking else 0,
                "exact_forward_calls_completed": completed if not gradient and not ranking else 0,
                "ranking_channel_calls": completed if ranking else 0,
                "density_direction_passes": (2 if gradient or ranking else 1) * completed
                if completed is not None
                else None,
            },
            [proof],
            label=root.name + "/" + relative + f"/{index}",
            scope="each durable exact-channel attempt; failed and interrupted calls retained separately",
        )


def _risk(root: Path, record: dict[str, Any], ledger: Ledger, proof: str) -> None:
    """Charge all six retired/promoted risk controls and their separate common-CPU evaluations."""
    inner_cpu, inner_wall = [], []
    for method, arm in sorted(record["arms"].items()):
        relative = f"training/{method}/ATTEMPTS.json"
        attempts, ap = _attempts(root / relative, ledger, arm["attempts_checksum"])
        _density_attempt(root, record, relative, attempts, ap, ledger, gradient=True)
        inner_cpu.append(_sum(attempts["attempts"], "cpu_seconds"))
        inner_wall.append(_sum(attempts["attempts"], "wall_seconds"))
        _declare(ledger, root.parent.name, root.name + "/" + method, arm, proof)
    expected = record["evaluation_attempts"]
    if set(expected) != {str(path.relative_to(root)) for path in (root / "operations").glob("*.attempts.json")}:
        msg = "Risk evaluation resource index is incomplete."
        raise ValueError(msg)
    for relative, parent in sorted(expected.items()):
        attempts, ap = _attempts(root / relative, ledger, pilot.digest(parent))
        _density_attempt(root, record, relative, attempts, ap, ledger, evaluation=True)
        inner_cpu.append(_sum(attempts["attempts"], "cpu_seconds"))
        inner_wall.append(_sum(attempts["attempts"], "wall_seconds"))
    wrapper, wp = _attempts(root / "CASE_ATTEMPTS.json", ledger, record["case_attempts_checksum"])
    ledger.add(
        record["signature"],
        root.parent.name,
        "bookkeeping",
        {
            "cpu_seconds": residual(_sum(wrapper["attempts"], "cpu_seconds"), inner_cpu),
            "wall_seconds": residual(_sum(wrapper["attempts"], "wall_seconds"), inner_wall),
        },
        [proof, wp],
        label=root.name,
        scope="all case invocations minus nonoverlapping training and score attempts",
    )


def _mixture(root: Path, record: dict[str, Any], ledger: Ledger, proof: str, protocol: dict[str, Any]) -> None:
    """Account four new stochastic arms, every repair, and fresh scoring of three reused controls."""
    artifact_hashes = record["artifact_hashes"]
    for relative, expected in artifact_hashes.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            msg = f"Mixture resource child bytes changed: {relative}."
            raise ValueError(msg)
    for path in sorted((root / "stages").glob("*/ATTEMPTS.json")):
        relative = str(path.relative_to(root))
        if relative not in artifact_hashes:
            msg = "Mixture resource index omits a stochastic attempt."
            raise ValueError(msg)
        attempts, ap = _attempts(path, ledger)
        for index, attempt in enumerate(attempts["attempts"]):
            known = attempt["status"] != "running"
            mapping = {
                "trajectory_paths": "completed_trajectory_paths",
                "trajectory_direction_passes": "completed_trajectory_direction_passes",
                "attempted_trajectory_paths": "attempted_trajectory_paths",
                "attempted_trajectory_direction_passes": "attempted_trajectory_direction_passes",
                "local_operator_applications_completed": "completed_local_operator_applications",
                "local_operator_applications_attempted": "attempted_local_operator_applications",
                "cpu_seconds": "cpu_seconds",
                "wall_seconds": "wall_seconds",
            }
            ledger.add(
                pilot.digest([record["signature"], relative, index, attempt]),
                root.parent.name,
                "optimization",
                {key: attempt[value] if known else None for key, value in mapping.items()},
                [proof, ap],
                label=root.name + "/" + relative + f"/{index}",
                scope="one weighted-mixture or adaptive-online stage; all failed/replayed updates and repairs",
            )
            ledger.observe(
                root.parent.name,
                relative,
                "stochastic_updates",
                {key: attempt[key] for key in ("attempted_updates", "completed_updates", "sampling_attempts")},
                [ap],
            )
    for path in sorted((root / "operations").glob("*.attempts.json")):
        relative = str(path.relative_to(root))
        if relative not in artifact_hashes:
            msg = "Mixture resource index omits an exact operation."
            raise ValueError(msg)
        attempts, ap = _attempts(path, ledger)
        _density_attempt(
            root,
            record,
            relative,
            attempts,
            ap,
            ledger,
            ranking=path.name.endswith("_ranking.attempts.json"),
            evaluation=path.name.startswith(("final_", "common_", "unchanged_start")),
        )
    for method, arm in sorted(record["arms"].items()):
        _declare(
            ledger, root.parent.name, root.name + "/" + method, arm, proof, reference=method not in protocol["methods"]
        )
    ledger.observe(
        root.parent.name,
        root.name,
        "unmeasured_case_overhead",
        {
            "cpu_seconds": None,
            "wall_seconds": None,
            "scope": "outer case bookkeeping is untimed; primitive stages and operations include inner I/O",
        },
        [proof],
    )


def _initializer(root: Path, record: dict[str, Any], ledger: Ledger, proof: str) -> None:
    """Charge a fresh shared initializer probe, new refinement/deployment, and unique scores."""
    for relative, expected in record["dependencies"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            msg = f"Initializer resource child changed: {relative}."
            raise ValueError(msg)
    initial, ip = checked(root / "INITIALIZERS.json", ledger, record["initializer_checksum"])
    attempts, ap = _attempts(root / "INITIALIZER_ATTEMPTS.json", ledger, initial["attempts_checksum"])
    costs = initial["costs"]
    parts = {
        "construction": costs["rank_two_synthesis_cpu_seconds"],
        "compilation": costs["rank_two_compilation_and_audit_cpu_seconds"]
        + costs["legacy_compilation_and_audit_cpu_seconds"],
        "diagnostic": costs["overlap_and_lowering_audit_cpu_seconds"],
    }
    for category, cpu in parts.items():
        ledger.add(
            initial["signature"],
            root.parent.name,
            category,
            {"cpu_seconds": cpu},
            [ip, ap],
            label=root.name + "/shared_initializer",
            scope="one physically shared initializer build; archived legacy construction is excluded",
        )
    ledger.add(
        initial["signature"],
        root.parent.name,
        "bookkeeping",
        {
            "cpu_seconds": residual(_sum(attempts["attempts"], "cpu_seconds"), list(parts.values())),
        },
        [ip, ap],
        label=root.name + "/shared_initializer",
        scope="initializer attempts minus final measured build components; retains prior interrupted setup work",
    )
    ledger.observe(
        root.parent.name,
        root.name,
        "initializer_build_and_charges",
        {
            "physical_build_wall_seconds": initial["physical_build_wall_seconds"],
            **costs,
            "legacy_construction_is_reused": True,
            "common_total_cpu_status": record["common_total_cpu"].get("status"),
            "common_total_availability": [
                {
                    "fraction": point["fraction"],
                    "cpu_seconds": point["cpu_seconds"],
                    "all_available": point["all_available"],
                    "arms": {name: value["status"] for name, value in point["arms"].items()},
                }
                for point in record["common_total_cpu"].get("points", [])
            ],
        },
        [proof, ip],
    )
    for method, arm in sorted(record["arms"].items()):
        ledger.observe(
            root.parent.name,
            root.name + "/" + method,
            "full_procedure_charge",
            {
                key: arm[key]
                for key in (
                    "initializer_cpu_seconds",
                    "training_cpu_seconds",
                    "deployment_cpu_seconds",
                    "refinement_cpu_seconds",
                    "total_cpu_seconds",
                )
            },
            [proof],
        )
        if arm["training_checksum"] is None:
            continue
        training, tp = checked(root / method / "TRAINING.json", ledger, arm["training_checksum"])
        attempts, ap = _attempts(root / method / "ATTEMPTS.json", ledger, training["attempts_checksum"])
        for index, attempt in enumerate(attempts["attempts"]):
            operation = pilot.digest([training["signature"], index, attempt])
            completed = attempt["completed_channel_calls"] if attempt["status"] != "running" else None
            ledger.add(
                operation,
                root.parent.name,
                "optimization",
                {
                    "cpu_seconds": attempt["training_cpu_seconds"],
                    "exact_gradient_calls_attempted": attempt["attempted_channel_calls"]
                    if attempt["status"] != "running"
                    else None,
                    "exact_gradient_calls_completed": completed,
                    "density_direction_passes": 2 * completed if completed is not None else None,
                },
                [tp, ap],
                label=root.name + "/" + method + f"/{index}",
                scope="mean2 individual-channel gradients, excluding separately timed deployment compilation",
            )
            ledger.add(
                operation,
                root.parent.name,
                "compilation",
                {
                    "cpu_seconds": attempt["deployment_cpu_seconds"],
                },
                [tp, ap],
                label=root.name + "/" + method + f"/{index}",
                scope="executed accepted-point deployment compilation and ideal audits; cached reuse is free",
            )
            ledger.observe(
                root.parent.name,
                method,
                "covering_refinement_wall",
                {
                    "wall_seconds": attempt["wall_seconds"],
                },
                [ap],
            )
    for path in sorted((root / "scores").glob("*.json")):
        if str(path.relative_to(root)) not in record["dependencies"]:
            msg = "Initializer resource index omits a cached evaluation."
            raise ValueError(msg)
        score, sp = checked(path, ledger)
        _endpoint(
            ledger,
            root.parent.name,
            record["signature"] + "/" + score["signature"],
            root.name + "/score/" + path.stem,
            score,
            sp,
            13,
        )


def _authenticated_children(root: Path, record: dict[str, Any], key: str) -> set[str]:
    """Return all indexed physical attempt files after authenticating raw child hashes."""
    indexed = record[key]
    for relative, expected in indexed.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            msg = f"Resource child bytes changed: {relative}."
            raise ValueError(msg)
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*.json")
        if "ATTEMPT" in path.name or path.name.endswith(".attempts.json") or path.name == "INVOCATIONS.json"
    }
    if not actual.issubset(indexed):
        msg = "Resource dependency index omits a durable physical attempt."
        raise ValueError(msg)
    return actual


def _stochastic_attempts(
    root: Path,
    record: dict[str, Any],
    relative: str,
    payload: dict[str, Any],
    ledger: Ledger,
    proof: str,
) -> None:
    """Charge each weighted/adaptive stage, including repairs that were rejected."""
    mapping = {
        "trajectory_paths": "completed_trajectory_paths",
        "trajectory_direction_passes": "completed_trajectory_direction_passes",
        "attempted_trajectory_paths": "attempted_trajectory_paths",
        "attempted_trajectory_direction_passes": "attempted_trajectory_direction_passes",
        "local_operator_applications_completed": "completed_local_operator_applications",
        "local_operator_applications_attempted": "attempted_local_operator_applications",
        "cpu_seconds": "cpu_seconds",
        "wall_seconds": "wall_seconds",
    }
    for index, attempt in enumerate(payload["attempts"]):
        ledger.add(
            pilot.digest([record["signature"], relative, index, attempt]),
            root.parent.name,
            "optimization",
            {key: attempt.get(value) if attempt["status"] != "running" else None for key, value in mapping.items()},
            [proof],
            label=root.name + "/" + relative + f"/{index}",
            scope="durable weighted/adaptive trajectory work; failed/repeated and rejected repairs included",
        )


def _long_ensemble(root: Path, record: dict[str, Any], ledger: Ledger, proof: str) -> None:
    """Partition long optimizer invocations and charge unique new all12 score attempts."""
    actual = _authenticated_children(root, record, "artifact_hashes")
    expected = {f"{method}/{name}.json" for method in record["arms"] for name in ("ATTEMPTS", "INVOCATIONS")}
    expected |= {str(p.relative_to(root)) for p in root.glob("scores/*/ATTEMPTS.json")}
    if actual != expected:
        msg = "Unknown or missing long-ensemble physical attempt schema."
        raise ValueError(msg)
    for method, arm in sorted(record["arms"].items()):
        relative = f"{method}/ATTEMPTS.json"
        attempts, ap = _attempts(root / relative, ledger)
        _density_attempt(root, record, relative, attempts, ap, ledger, gradient=True)
        wrapper, wp = _attempts(root / method / "INVOCATIONS.json", ledger)
        ledger.add(
            record["signature"] + "/" + method,
            root.parent.name,
            "bookkeeping",
            {
                key: residual(_sum(wrapper["attempts"], key), [_sum(attempts["attempts"], key)])
                for key in ("cpu_seconds", "wall_seconds")
            },
            [ap, wp],
            label=root.name + "/" + method,
            scope="long-run wrapper minus its nested primitive optimizer; no reused archive training",
        )
        _declare(ledger, root.parent.name, root.name + "/" + method, arm.get("training", arm), proof)
    for path in sorted((root / "scores").glob("*/ATTEMPTS.json")):
        attempts, ap = _attempts(path, ledger)
        _density_attempt(root, record, str(path.relative_to(root)), attempts, ap, ledger, evaluation=True)
    ledger.observe(
        root.parent.name,
        root.name,
        "unmeasured_case_overhead",
        {
            "cpu_seconds": None,
            "wall_seconds": None,
            "scope": "target input, circuit restore, outer case assembly/custody and result serialization are untimed",
        },
        [proof],
    )


def _composition(
    root: Path,
    record: dict[str, Any],
    ledger: Ledger,
    proof: str,
    *,
    compiled: bool = False,
    wrapper_path: Path | None = None,
    initializer_reused: bool = True,
) -> None:
    """Partition the inclusive composed case around disjoint training, scoring and compiler work."""
    actual = _authenticated_children(root, record, "dependencies")
    patterns = (
        "stages/*/ATTEMPTS.json",
        "stages/*/INVOCATIONS.json",
        "stages/*/INVOCATION_RESULT.attempts.json",
        "operations/*.attempts.json",
        "compilations/*.attempts.json",
        "CASE_ATTEMPTS.json",
    )
    if compiled:
        patterns += ("state_compilations/*.attempts.json",)
    expected = {str(p.relative_to(root)) for pattern in patterns for p in root.glob(pattern)}
    if actual != expected:
        msg = "Unknown composition physical attempt schema."
        raise ValueError(msg)
    parts: dict[str, list[float | None]] = {key: [] for key in ("cpu_seconds", "wall_seconds")}
    for path in sorted((root / "stages").glob("*/ATTEMPTS.json")):
        payload, ap = _attempts(path, ledger)
        relative = str(path.relative_to(root))
        if "completed_trajectory_paths" in payload["attempts"][0]:
            _stochastic_attempts(root, record, relative, payload, ledger, ap)
        elif "completed_channel_calls" in payload["attempts"][0]:
            _density_attempt(root, record, relative, payload, ap, ledger, gradient=True)
        else:
            msg = "Unknown composition primitive stage accounting schema."
            raise ValueError(msg)
        for key, values in parts.items():
            values.append(_sum(payload["attempts"], key))
    for path in sorted((root / "operations").glob("*.attempts.json")):
        payload, ap = _attempts(path, ledger)
        _density_attempt(
            root,
            record,
            str(path.relative_to(root)),
            payload,
            ap,
            ledger,
            ranking=path.name.endswith("_ranking.attempts.json"),
            evaluation=path.name.startswith("audit_"),
        )
        for key, values in parts.items():
            values.append(_sum(payload["attempts"], key))
    compilation_paths = list((root / "compilations").glob("*.attempts.json"))
    if compiled:
        compilation_paths.extend((root / "state_compilations").glob("*.attempts.json"))
    for path in sorted(compilation_paths):
        payload, ap = _attempts(path, ledger)
        for index, attempt in enumerate(payload["attempts"]):
            ledger.add(
                pilot.digest([record["signature"], str(path.relative_to(root)), index, attempt]),
                root.parent.name,
                "compilation",
                {key: attempt.get(key) for key in parts},
                [ap],
                label=root.name + "/" + path.stem + f"/{index}",
                scope="fresh final deployment compiler and ideal-state audit, including failed attempts",
            )
        for key, values in parts.items():
            values.append(_sum(payload["attempts"], key))
    wrapper, wp = _attempts(
        wrapper_path or root / "CASE_ATTEMPTS.json", ledger, record["physical_invocation"]["attempts_checksum"]
    )
    ledger.add(
        record["signature"],
        root.parent.name,
        "bookkeeping",
        {key: residual(_sum(wrapper["attempts"], key), parts[key]) for key in parts},
        [proof, wp],
        label=root.name,
        scope="inclusive case minus primitive, score and compiler attempts; includes inner wrappers",
    )
    covered = [*root.glob("stages/*/INVOCATIONS.json"), *root.glob("stages/*/INVOCATION_RESULT.attempts.json")]
    for path in sorted(covered):
        payload, cp = _attempts(path, ledger)
        ledger.observe(
            root.parent.name,
            str(path.relative_to(root)),
            "covering_stage_invocation",
            {key: _sum(payload["attempts"], key) for key in parts},
            [cp],
        )
    for method, arm in sorted(record["arms"].items()):
        ledger.observe(
            root.parent.name,
            root.name + "/" + method,
            "full_procedure_charge",
            {
                key: arm.get(key)
                for key in (
                    "initializer_cpu_seconds",
                    "training_cpu_seconds",
                    "refinement_cpu_seconds",
                    "total_cpu_seconds",
                )
            },
            [proof],
        )
    ledger.observe(
        root.parent.name,
        root.name,
        "reused_initializer" if initializer_reused else "separately_charged_initializer",
        {
            "declared_initializer_cpu_seconds": (
                record["initializer_cpu_seconds"] if compiled else record["source"]["initializer_cpu_seconds"]
            ),
            "fresh_physical_construction_cpu_seconds": 0.0,
            "scope": "initializer physical work is accounted separately once; this is only its declared arm charge",
        },
        [proof],
    )


def _mps_diagnostic(
    root: Path,
    record: dict[str, Any],
    ledger: Ledger,
    proof: str,
    protocol: dict[str, Any],
    *,
    physical: bool,
) -> None:
    """Partition inclusive replay attempts; original gate failure does not erase physical work."""
    actual = _authenticated_children(root, record, "dependencies")
    if actual != {"ATTEMPTS.json"}:
        msg = "Unknown MPS diagnostic physical attempt schema."
        raise ValueError(msg)
    payload, ap = _attempts(root / "ATTEMPTS.json", ledger)
    settings = protocol["settings"]["prior_settings"] if physical else protocol["settings"]
    paths = 2 * settings["paths_per_channel"]
    gradients = {"dense_gradient", "mps_gradient", "mps_same_theta_gradient"}
    forwards = {"dense_path_metrics", "dense_preupdate_path_metrics"}
    forwards |= {f"physical_{backend}_{source}" for backend in ("dense", "mps") for source in ("dense", "mps")}
    setup = {
        "target_input",
        "target_factorization",
        "target_factorization_audit",
        "target_mps_input",
        "dense_setup",
        "sampler_setup",
        "sampling",
        "dense_adam_update",
        "mps_adam_update",
        "prior_state_input",
        "scalar_adam_replay",
        "circuit_restore",
    }
    for index, attempt in enumerate(payload["attempts"]):
        for stage_index, stage in enumerate(attempt["stages"]):
            phase = stage["phase"]
            if phase not in gradients | forwards | setup:
                msg = "Unknown MPS resource stage; completed path work cannot be inferred."
                raise ValueError(msg)
            numerical = phase in gradients | forwards
            complete = stage["status"] == "complete"
            planned = paths if numerical else 0
            done_paths = planned if complete else None
            directions = 2 if phase in gradients else 1
            ledger.add(
                pilot.digest([record["signature"], "ATTEMPTS.json", index, stage_index, stage]),
                root.parent.name,
                "diagnostic",
                {
                    "cpu_seconds": stage.get("cpu_seconds"),
                    "wall_seconds": stage.get("wall_seconds"),
                    "trajectory_paths": done_paths,
                    "trajectory_direction_passes": directions * done_paths if done_paths is not None else None,
                    "attempted_trajectory_paths": planned if complete else None,
                    "attempted_trajectory_direction_passes": directions * planned if complete else None,
                    "mps_optimizer_updates": int(phase == "mps_adam_update") if complete else None,
                },
                [proof, ap],
                label=f"{root.name}/{index}/{stage_index}/{phase}",
                scope="recorded replay stage; completed paths from frozen schedule; failed partial paths unknown",
                count_quality="source-derived completed path directions; CPU/wall measured",
            )
        ledger.add(
            pilot.digest([record["signature"], "wrapper", index, attempt]),
            root.parent.name,
            "bookkeeping",
            {
                key: residual(attempt.get(key), [stage.get(key) for stage in attempt["stages"]])
                for key in ("cpu_seconds", "wall_seconds")
            },
            [proof, ap],
            label=f"{root.name}/{index}",
            scope="inclusive diagnostic attempt minus disjoint stages; factorization and replay counted once",
        )
    ledger.observe(
        root.parent.name,
        root.name,
        "diagnostic_acceptance_and_memory",
        {
            "accepted": record["accepted"],
            "prior_gate": record.get("prior_gate", "original acceptance unchanged"),
            "process_peak_rss_bytes": None,
            "scope": "tensor storage is not process peak; diagnostic is not independent optimization replication",
        },
        [proof],
    )


def _json_inventory(root: Path, excluded: Sequence[str] = ()) -> dict[str, str]:
    """Hash a complete child registry without loading numerical arrays into memory."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.json"))
        if str(path.relative_to(root)) not in excluded
    }


def _timed_costs(payload: dict[str, Any]) -> dict[str, float | None]:
    """Keep incomplete durations null, including any durable unknown running work."""
    return {key: _sum(payload["attempts"], key) for key in ("cpu_seconds", "wall_seconds")}


def _partition_timer(
    root: Path,
    record: dict[str, Any],
    ledger: Ledger,
    path: Path,
    parts: Sequence[dict[str, Any]],
    *,
    category: str = "bookkeeping",
    expected: str | None = None,
) -> dict[str, float | None]:
    """Charge only the portion of an inclusive durable timer outside its nested work."""
    payload, proof = _attempts(path, ledger, expected)
    totals = _timed_costs(payload)
    ledger.add(
        pilot.digest([payload["signature"], str(path.relative_to(root)), payload["attempts"]]),
        root.parent.name,
        category,
        {key: residual(total, [part.get(key) for part in parts]) for key, total in totals.items()},
        [proof],
        label=root.name + "/" + str(path.relative_to(root)),
        scope="inclusive durable attempts minus authenticated disjoint nested measurements; all replay remains charged",
    )
    ledger.observe(root.parent.name, root.name, "inclusive_invocation", totals, [proof])
    if (
        "physical_invocation" in record
        and path.name == "CASE_ATTEMPTS.json"
        and any(record["physical_invocation"].get(key) != value for key, value in totals.items())
    ):
        msg = "Inclusive case resource fields differ from their durable attempts."
        raise ValueError(msg)
    return totals


def _common_control(root: Path, record: dict[str, Any], ledger: Ledger, proof: str) -> None:
    """Charge new fixed-control scores and wrapper only; the four primary scores are reused."""
    value = record["value"]
    indexed = {name: digest for name, digest in _json_inventory(root).items() if "/" in name}
    if indexed != value["dependencies"]:
        msg = "Common-control resource child registry changed."
        raise ValueError(msg)
    actual = {name for name in indexed if "ATTEMPT" in Path(name).name or name.endswith(".attempts.json")}
    expected = {str(p.relative_to(root)) for p in root.glob("operations/audit_*.attempts.json")}
    if actual != expected:
        msg = "Unknown common-control physical work schema."
        raise ValueError(msg)
    parts = []
    for path in sorted(root.glob("operations/audit_*.attempts.json")):
        payload, ap = _attempts(path, ledger)
        _density_attempt(root, record, str(path.relative_to(root)), payload, ap, ledger, evaluation=True)
        parts.append(_timed_costs(payload))
    payload, ap = _attempts(root / "RESULT.attempts.json", ledger, record["attempts_checksum"])
    if any(record[key] != value for key, value in _timed_costs(payload).items()):
        msg = "Common-control wrapper duration differs from its attempt ledger."
        raise ValueError(msg)
    _partition_timer(root, record, ledger, root / "RESULT.attempts.json", parts)
    ledger.observe(
        root.parent.name,
        root.name,
        "reference_only_developed_arms",
        {
            "fresh_training_cpu_seconds": 0.0,
            "declared_total_cpu_seconds": value["total_cpu_seconds"],
            "source_case_checksum": value["source_case_checksum"],
            "scope": "source optimization, initializers and four primary endpoint evaluations were already executed",
        },
        [proof, ap],
    )


def _state_compilation(root: Path, record: dict[str, Any], ledger: Ledger, proof: str) -> None:
    """Partition both fresh endpoint evaluations and the compiler inside each pair invocation."""
    actual = _authenticated_children(root, record, "dependencies")
    expected = {"CASE_ATTEMPTS.json", "COMPILATION.attempts.json"}
    expected |= {
        str(p.relative_to(root)) for side in ("before", "after") for p in root.glob(f"{side}/scores/*/ATTEMPTS.json")
    }
    if actual != expected or record["dependencies"] != _json_inventory(root, ("RESULT.json",)):
        msg = "Unknown or changed state-preparation compiler resource schema."
        raise ValueError(msg)
    parts = []
    for path in sorted(root.glob("*/scores/*/ATTEMPTS.json")):
        payload, ap = _attempts(path, ledger)
        _density_attempt(root, record, str(path.relative_to(root)), payload, ap, ledger, evaluation=True)
        parts.append(_timed_costs(payload))
    parts.append(_partition_timer(root, {}, ledger, root / "COMPILATION.attempts.json", [], category="compilation"))
    _partition_timer(
        root,
        record,
        ledger,
        root / "CASE_ATTEMPTS.json",
        parts,
        expected=record["physical_invocation"]["attempts_checksum"],
    )
    ledger.observe(
        root.parent.name,
        root.name,
        "reference_only_source_training",
        {
            "cpu_seconds": 0.0,
            "scope": "both scores are new, but supplied source construction/training is reused",
        },
        [proof],
    )


def _lazy_costs(root: Path, ledger: Ledger) -> dict[str, float | None]:
    """Count an actual lazy initializer once, including discarded probes and fresh fallback."""
    result, proof = checked(root / "RESULT.json", ledger)
    payload, ap = _attempts(root / "ATTEMPTS.json", ledger, result["attempts_checksum"])
    if any(result[f"initializer_{key}"] != value for key, value in _timed_costs(payload).items()):
        msg = "Lazy initializer cost differs from its complete attempt ledger."
        raise ValueError(msg)
    indexed = _json_inventory(root, ("RESULT.json", "ATTEMPTS.json"))
    if payload["attempts"][-1]["dependencies"] != indexed:
        msg = "Lazy initializer resource registry changed."
        raise ValueError(msg)
    nested = []
    legacy_path = root / "legacy/ATTEMPTS.json"
    if legacy_path.exists():
        legacy, lp = checked(root / "legacy/RESULT.json", ledger, result["legacy_receipt_checksum"])
        attempts, tp = _attempts(legacy_path, ledger, legacy["attempts_checksum"])
        for index, attempt in enumerate(attempts["attempts"]):
            ledger.add(
                pilot.digest([attempts["signature"], index, attempt]),
                root.parent.name,
                "construction",
                {
                    **{key: attempt.get(key) for key in ("cpu_seconds", "wall_seconds")},
                    "mps_optimizer_updates": legacy["updates"]
                    if len(attempts["attempts"]) == 1 and attempt["status"] == "complete"
                    else None,
                },
                [lp, tp],
                label=root.name + f"/fresh_legacy/{index}",
                scope="fresh fallback MPS construction; its internal diagnostic is included, not additive",
            )
        nested.append(_timed_costs(attempts))
    if bool(legacy_path.exists()) == result["rank_two_selected"]:
        msg = "Lazy constructor presence disagrees with the recorded conditional branch."
        raise ValueError(msg)
    actual = {name for name in indexed if "ATTEMPT" in Path(name).name or name.endswith(".attempts.json")}
    if actual != ({"legacy/ATTEMPTS.json"} if legacy_path.exists() else set()):
        msg = "Unknown lazy initializer nested attempt schema."
        raise ValueError(msg)
    totals = _partition_timer(root, {}, ledger, root / "ATTEMPTS.json", nested, category="construction")
    ledger.observe(
        root.parent.name,
        root.name,
        "nested_lazy_probe",
        {
            "cpu_seconds": result["probe_cpu_seconds"],
            "discarded_probe_cpu_seconds": result["discarded_probe_cpu_seconds"],
            "nested_legacy_cpu_seconds": result["nested_legacy_cpu_seconds"],
            "scope": "included in the lazy initializer attempts; no historical construction CPU imputed",
        },
        [proof, ap],
    )
    return totals


def _lazy_reproduction(root: Path, record: dict[str, Any], ledger: Ledger, proof: str) -> None:
    """Charge fresh lazy construction, new equivalence audit and known target-input overhead."""
    if record["dependencies"] != _json_inventory(root, ("RESULT.json", "COMPLETED.json")):
        msg = "Lazy reproduction resource child registry changed."
        raise ValueError(msg)
    sealed, _ = checked(root / "COMPLETED.json", ledger)
    if sealed != {"signature": record["signature"], "result_checksum": pilot.digest(record)}:
        msg = "Lazy reproduction final fields differ from their completion receipt."
        raise ValueError(msg)
    initializer = _lazy_costs(root / "initializer", ledger)
    audit = record["audit"]["cpu_seconds"]
    ledger.add(
        record["signature"],
        root.parent.name,
        "diagnostic",
        {"cpu_seconds": audit},
        [proof],
        label=root.name,
        scope="fresh independent lazy initializer endpoint and physical-state equivalence audit",
    )
    _partition_timer(
        root,
        record,
        ledger,
        root / "CASE_ATTEMPTS.json",
        [initializer, {"cpu_seconds": audit, "wall_seconds": 0.0}],
        expected=record["physical_invocation"]["attempts_checksum"],
    )
    ledger.add(
        record["signature"],
        root.parent.name,
        "setup",
        {"cpu_seconds": record["target_input_cpu_seconds"]},
        [proof],
        label=root.name,
        scope="separately timed target input outside lazy reproduction invocation; earlier interrupted loads unknown",
    )


def _compiled_resource(root: Path, record: dict[str, Any], ledger: Ledger, proof: str) -> None:
    """Use the compact kernel receipt to charge both compilers and all original stage work."""
    execution, ep = checked(root / "EXECUTION.json", ledger, record["execution_checksum"])
    allowed = {
        "signature",
        "version",
        "execution_signature",
        "execution_checksum",
        "protocol_checksum",
        "target_id",
        "replica",
    }
    if any(record.get(key) != value for key, value in execution.items() if key not in allowed):
        msg = "Compiled development public resource values differ from the kernel receipt."
        raise ValueError(msg)
    if execution["dependencies"] != _json_inventory(root, ("RESULT.json", "EXECUTION.json")):
        msg = "Compiled resource receipt omits or changes a physical child."
        raise ValueError(msg)
    _composition(root, execution, ledger, ep, compiled=True)
    ledger.observe(
        root.parent.name, root.name, "wrapper_receipt", {"kernel_checksum": pilot.digest(execution)}, [proof]
    )


def _phase_resource(root: Path, ledger: Ledger) -> tuple[dict[str, Any], str]:
    """Authenticate a parent phase using its completed result seal and exact file inventory."""
    result, proof = checked(root / "RESULT.json", ledger)
    manifest, _ = checked(root / "PHASE.json", ledger)
    sealed, _ = checked(root / "COMPLETED.json", ledger)
    attempts, _ = _attempts(root / "ATTEMPTS.json", ledger, result["attempts_checksum"])
    excluded = {root / name for name in ("PHASE.json", "RESULT.json", "COMPLETED.json", "ATTEMPTS.json")}
    files = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p not in excluded
    }
    if (
        manifest != {"signature": result["signature"]}
        or sealed
        != {
            "signature": result["signature"],
            "result_checksum": pilot.digest(result),
            "attempts_checksum": pilot.digest(attempts),
        }
        or attempts["signature"] != result["signature"]
        or attempts["attempts"][-1]["files"] != files
        or any(result[key] != value for key, value in _timed_costs(attempts).items())
        or any(a["status"] not in {"complete", "failed", "interrupted"} for a in attempts["attempts"])
    ):
        msg = "Confirmation phase resource fields, sealed result or full child inventory changed."
        raise ValueError(msg)
    return result, proof


def _confirmation_resource(root: Path, record: dict[str, Any], ledger: Ledger, proof: str) -> None:
    """Partition fresh confirmation input, initialization and training under one inclusive case."""
    execution = root / "execution"
    physical, pp = _phase_resource(execution, ledger)
    if record["physical_invocation"]["checksum"] != pilot.digest(physical) or any(
        record.get(key) != value for key, value in physical["value"].items()
    ):
        msg = "Confirmation outer fields differ from the complete physical phase."
        raise ValueError(msg)
    phases = {}
    for name in ("input", "initializer", "training"):
        phases[name], _ = _phase_resource(execution / name, ledger)
    _partition_timer(execution, {}, ledger, execution / "input/ATTEMPTS.json", [], category="setup")
    lazy_cost = _lazy_costs(execution / "initializer/work", ledger)
    _partition_timer(
        execution, {}, ledger, execution / "initializer/ATTEMPTS.json", [lazy_cost], category="construction"
    )
    work = execution / "training/work"
    trained = phases["training"]
    if (work / "EXECUTION.json").exists():
        kernel_result, kp = checked(work / "EXECUTION.json", ledger)
        if kernel_result["dependencies"] != _json_inventory(work, ("RESULT.json", "EXECUTION.json")):
            msg = "Confirmation compiled kernel inventory changed."
            raise ValueError(msg)
        _composition(work, kernel_result, ledger, kp, compiled=True, initializer_reused=False)
        _partition_timer(
            execution, {}, ledger, execution / "training/ATTEMPTS.json", [kernel_result["physical_invocation"]]
        )
    else:
        arms = {
            name: checked(work / receipt["path"], ledger, receipt["checksum"])[0]
            for name, receipt in trained["value"]["arm_receipts"].items()
        }
        synthetic: dict[str, Any] = {
            "signature": trained["signature"],
            "dependencies": _json_inventory(work),
            "arms": arms,
            "source": {"initializer_cpu_seconds": trained["value"]["initializer_cpu_seconds"]},
            "physical_invocation": {"attempts_checksum": trained["attempts_checksum"]},
        }
        _composition(
            work, synthetic, ledger, pp, wrapper_path=execution / "training/ATTEMPTS.json", initializer_reused=False
        )
    _partition_timer(execution, {}, ledger, execution / "ATTEMPTS.json", list(phases.values()))
    ledger.observe(
        root.parent.name,
        root.name,
        "complete_confirmation_procedure_clocks",
        {
            "initializer_cpu_seconds": record["initializer_cpu_seconds"],
            "amortized_preparation_cpu_seconds": record["amortized_preparation_cpu_seconds"],
            "scope": "declared per-arm clock only; physical preparation and shared initializer counted once",
        },
        [proof],
    )
    memory = record.get("peak_rss")
    if memory is not None:
        unit = {"darwin": "bytes", "linux": "KiB"}.get(memory.get("platform"))
        raw = memory.get("raw_value")
        if (
            unit is None
            or memory.get("raw_unit") != unit
            or isinstance(raw, bool)
            or not isinstance(raw, (float, int))
            or not math.isfinite(raw)
            or raw < 0
            or memory.get("bytes") != int(raw * (1 if unit == "bytes" else 1024))
        ):
            msg = "Confirmation peak RSS raw units or normalized bytes changed."
            raise ValueError(msg)
        ledger.observe(root.parent.name, root.name, "worker_lifetime_peak_rss", memory, [proof])


def _mps_memory_resource(root: Path, record: dict[str, Any], ledger: Ledger, proof: str) -> None:
    """Partition one fixed-path memory diagnostic without treating tensor payload as process RSS."""
    raw, rp = checked(root / "MEASUREMENT.json", ledger, record["measurement_checksum"])
    attempts, ap = _attempts(root / "ATTEMPTS.json", ledger)
    sealed, _ = checked(root / "COMPLETED.json", ledger)
    files = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and path not in {root / "RESULT.json", root / "COMPLETED.json"}
    }
    all_stages = [stage for attempt in attempts["attempts"] for stage in attempt["stages"]]
    if (
        record["dependencies"] != files
        or record["measurement"] != raw
        or record["accepted"] != raw["accepted"]
        or record["costs"] != overlap.costs(attempts)
        or record["signature"] != attempts["signature"]
        or sealed != {"signature": record["signature"], "result_checksum": pilot.digest(record)}
        or any(attempt["status"] not in {"complete", "failed", "interrupted"} for attempt in attempts["attempts"])
        or any(
            row.get(key) is None
            or isinstance(row[key], bool)
            or not isinstance(row[key], (float, int))
            or not math.isfinite(row[key])
            or row[key] < 0
            for row in [*attempts["attempts"], *all_stages]
            for key in ("cpu_seconds", "wall_seconds")
        )
    ):
        msg = "Memory resource endpoint has changed raw records, acceptance, inventory or missing durable costs."
        raise ValueError(msg)
    gradients = {"mps_gradient", "dense_gradient"}
    known = gradients | {
        "saved_target_input",
        "saved_target_mps_input",
        "circuit_restore",
        "sampler_setup",
        "map_identity_regeneration",
        "saved_target_equivalence",
        "retained_memory_audit",
        "dense_setup",
        "forward_norm_audit",
    }
    for index, stage in enumerate(all_stages):
        name = stage["phase"]
        if name not in known or stage["status"] not in {"complete", "failed_or_interrupted"}:
            msg = "Unknown memory diagnostic stage cannot be silently assigned zero cost."
            raise ValueError(msg)
        numerical = name in gradients
        paths = 1 if stage["status"] == "complete" else None
        ledger.add(
            pilot.digest([attempts["signature"], index, stage]),
            root.parent.name,
            "diagnostic",
            {
                "cpu_seconds": stage["cpu_seconds"],
                "wall_seconds": stage["wall_seconds"],
                "trajectory_paths": paths if numerical else 0,
                "trajectory_direction_passes": 2 * paths
                if numerical and paths is not None
                else None
                if numerical
                else 0,
                "attempted_trajectory_paths": int(numerical),
                "attempted_trajectory_direction_passes": 2 * int(numerical),
                "mps_optimizer_updates": 0,
            },
            [ap],
            label=root.name + f"/{index}/{name}",
            scope="one fixed-path MPS/dense gradient or separate measured memory/validation stage; no optimizer update",
        )
    _partition_timer(root, {}, ledger, root / "ATTEMPTS.json", all_stages)
    ledger.observe(
        root.parent.name,
        root.name,
        "retained_mps_tensor_storage",
        {
            "forward_tape": raw["memory"]["forward_tape"],
            "exact_target": raw["memory"]["exact_target"],
            "retained_union": raw["memory"]["retained_union"],
            "dense_oracle_target_bytes": raw["dense_oracle_target_bytes"],
            "analytic_no_factor_path": raw["analytic_no_factor_path"],
            "selected_path_index": raw["selected_path_index"],
            "scope": "one simultaneously retained forward tape plus exact MPS target; not peak or whole-batch memory",
            "prior_overlap_gate": raw["prior_overlap_gate"],
            "accepted": raw["accepted"],
        },
        [proof, rp],
    )
    memory = raw["peak_rss"]
    unit = {"darwin": "bytes", "linux": "KiB"}.get(memory["platform"])
    raw_value = memory["raw_value"]
    if (
        unit is None
        or memory["raw_unit"] != unit
        or isinstance(raw_value, bool)
        or not isinstance(raw_value, (float, int))
        or not math.isfinite(raw_value)
        or raw_value < 0
        or memory["bytes"] != int(raw_value * (1 if unit == "bytes" else 1024))
    ):
        msg = "Memory worker RSS units differ from their normalized bytes."
        raise ValueError(msg)
    ledger.observe(root.parent.name, root.name, "worker_lifetime_peak_rss", memory, [rp])


def _independent_validation(root: Path, protocol: dict[str, Any], record: dict[str, Any], ledger: Ledger) -> None:
    """Charge independent numerical replay once while retaining every failed/missing row."""
    inputs, ip = checked(root / "INPUTS.json", ledger, protocol["inputs_checksum"])
    if verifier.file_hash(root / "INPUTS.json") != protocol["input_file_sha256"]:
        msg = "Independent verification input bytes changed."
        raise ValueError(msg)
    expected = {
        (target, replica, panel, method)
        for target in verifier.parent_api.TARGETS
        for replica in verifier.parent_api.REPLICAS
        for panel in verifier.PANELS
        for method in verifier.parent_api.METHODS
    }
    actual = [(row["target_id"], row["replica"], row["panel"], row["method"]) for row in inputs["roles"]]
    if (
        protocol["engineering"]
        or len(actual) != 210
        or set(actual) != expected
        or protocol["row_count"] != 2520
        or len(record["rows"]) != 2520
        or protocol["role_count"] != 210
        or protocol["profile_ids"] != list(verifier.oracle.PROFILE_IDS)
        or protocol["unique_endpoints"] != len(inputs["endpoints"])
        or protocol["unique_calls"] != 12 * len(inputs["endpoints"])
        or protocol["source_hashes"] != inputs["source_hashes"]
        or record["status"] not in {"complete", "incomplete_budget"}
        or verifier.input_panel(Path(inputs["parent_root"])) != inputs
    ):
        msg = "Independent resource ledger requires the complete frozen 2520-row panel."
        raise ValueError(msg)
    allowed_attempts = {root / name for name in ("PREPARATION_ATTEMPTS.json", "RUN_ATTEMPTS.json")}
    for key in inputs["endpoints"]:
        endpoint_root = root / "endpoints" / key
        allowed_attempts.add(endpoint_root / "ATTEMPTS.json")
        allowed_attempts.update(endpoint_root / "calls" / noise / "ATTEMPTS.json" for noise in protocol["profile_ids"])
    if any(path not in allowed_attempts for path in root.rglob("*ATTEMPTS.json")):
        msg = "Independent resource ledger contains unindexed durable attempts."
        raise ValueError(msg)
    if verifier.cached(root) != record or verifier.summarize(root, protocol, inputs) != record:
        msg = "Independent public rows or costs differ from sealed raw calls."
        raise ValueError(msg)
    prep, pp = checked(root / "PREPARATION_ATTEMPTS.json", ledger, protocol["preparation_attempts_checksum"])
    running, rp = _attempts(root / "RUN_ATTEMPTS.json", ledger)
    if (
        prep["signature"] != pilot.digest([verifier.VERSION, inputs["parent_root"], False])
        or running["signature"] != pilot.digest([verifier.VERSION, pilot.digest(protocol)])
        or verifier.costs(prep) != protocol["preparation_costs"]
    ):
        msg = "Independent preparation/run identities or measured costs changed."
        raise ValueError(msg)
    for index, attempt in enumerate(prep["attempts"]):
        ledger.add(
            pilot.digest([prep["signature"], index, attempt]),
            root.name,
            "diagnostic",
            {
                "cpu_seconds": attempt["cpu_seconds"],
                "wall_seconds": attempt["wall_seconds"],
                "exact_forward_calls_attempted": 0,
                "exact_forward_calls_completed": 0,
                "density_direction_passes": 0,
            },
            [pp],
            label=f"preparation/{index}",
            scope="Independent-verifier preparation, saved-input custody and target-free calibration; no training.",
        )
    child_costs = []
    attempted, completed = 0, 0
    for path in sorted((root / "endpoints").glob("*/calls/*/ATTEMPTS.json")):
        attempts, ap = _attempts(path, ledger)
        if attempts["signature"] != pilot.digest([pilot.digest(protocol), path.parents[2].name, path.parent.name]):
            msg = "Independent raw call identity changed."
            raise ValueError(msg)
        child_costs.append(verifier.costs(attempts))
        for index, attempt in enumerate(attempts["attempts"]):
            success = attempt["status"] == "complete"
            attempted += 1
            completed += int(success)
            ledger.add(
                pilot.digest([attempts["signature"], index, attempt]),
                root.name,
                "diagnostic",
                {
                    "cpu_seconds": attempt["cpu_seconds"],
                    "wall_seconds": attempt["wall_seconds"],
                    "exact_forward_calls_attempted": 1,
                    "exact_forward_calls_completed": int(success),
                    "density_direction_passes": 1 if success else None,
                },
                [ap],
                label=str(path.parent.relative_to(root)) + f"/{index}",
                scope=(
                    "One independently implemented density-channel endpoint oracle admission; "
                    "failed partial forward work is not a known full direction count."
                ),
            )
    run_costs = verifier.costs(running)
    ledger.add(
        pilot.digest([running["signature"], running]),
        root.name,
        "diagnostic",
        {
            **{
                key: residual(run_costs[key], [row[key] for row in child_costs])
                for key in ("cpu_seconds", "wall_seconds")
            },
            "exact_forward_calls_attempted": 0,
            "exact_forward_calls_completed": 0,
            "density_direction_passes": 0,
        },
        [rp, ip],
        label="inclusive_run_residual",
        scope=(
            "Inclusive run less measured profile calls: custody, endpoint bookkeeping, I/O and full row reduction; "
            "no reference training is recharged."
        ),
    )
    for path in sorted((root / "endpoints").glob("*/ATTEMPTS.json")):
        attempts, ap = _attempts(path, ledger)
        if attempts["signature"] != pilot.digest([pilot.digest(protocol), path.parent.name]):
            msg = "Independent endpoint attempt identity changed."
            raise ValueError(msg)
        ledger.observe(root.name, path.parent.name, "covering_endpoint_invocation", verifier.costs(attempts), [ap])
    ledger.observe(
        root.name,
        "all_roles",
        "independent_verification_outcome",
        {
            "status": record["status"],
            "accepted": record["accepted"],
            "expected_rows": 2520,
            "unique_calls_expected": record["unique_calls_expected"],
            "unique_calls_terminal": record["unique_calls_terminal"],
            "unique_calls_failed": record["unique_calls_failed"],
            "oracle_calls_attempted_including_replay": attempted,
            "oracle_calls_completed_including_replay": completed,
            "cpu_overrun_seconds": record["cpu_overrun_seconds"],
            "original_training_is_reused": True,
            "peak_rss_bytes": None,
            "excluded_cpu_seconds": None,
            "excluded_scope": (
                "final receipt publication and later cache-only authentication; "
                "full engineering time remains unmeasured"
            ),
        },
        [ip, rp],
    )


def _collect_new_panel(
    root: Path, kind: str, protocol: dict[str, Any], completion: dict[str, Any], ledger: Ledger
) -> None:
    """Require complete explicit new-schema grids and dispatch read-only physical accounting."""
    if kind == "independent_validation":
        _independent_validation(root, protocol, completion, ledger)
        return
    if any(
        not Path(name).is_file() or hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected
        for name, expected in protocol.get("input_files", {}).items()
    ):
        msg = "New resource panel reused input source bytes changed."
        raise ValueError(msg)
    if kind == "mps_memory":
        selected = protocol["input_panel"]["entries"]
        entries = [(entry["target_id"], entry["replica"], entry["label"]) for entry in selected]
        checksums = [row["checksum"] for row in completion["cases"]]
        if len(entries) != 8 or set(entries) != {
            (target, 11, label) for target in overlap.TARGETS for label in overlap.ENDPOINTS
        }:
            msg = "Memory resource panel requires all eight fixed development endpoints."
            raise ValueError(msg)
        if any(
            hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected
            for name, expected in protocol["input_panel"]["files"].items()
        ):
            msg = "Memory resource frozen input bytes changed."
            raise ValueError(msg)
    elif kind == "common_control":
        entries = [(entry["target_id"], 11, None) for entry in protocol["input_panel"]["cases"]]
        checksums = completion["cases"]
    elif kind == "state_compilation":
        entries = [(entry["target_id"], entry["replica"], entry["label"]) for entry in protocol["pairs"]]
        checksums = list(completion["case_checksums"].values())
        if set(entries) != {
            (t, r, label) for t in protocol["targets"] for r in protocol["replicas"] for label in protocol["labels"]
        }:
            msg = "State compiler resource panel omits a frozen source pair."
            raise ValueError(msg)
    else:
        entries = [(t, r, None) for t in protocol["targets"] for r in protocol["replicas"]]
        checksums = (
            [row["checksum"] for row in completion["cases"]]
            if kind in {"lazy_initializer", "confirmation"}
            else list(completion["case_checksums"].values())
        )
    complete = (
        all(row["status"] == "complete" for row in completion["cases"])
        if kind == "lazy_initializer"
        else completion.get("status") == "complete"
    )
    if (
        not complete
        or not entries
        or len(set(entries)) != len(entries)
        or len(checksums) != len(entries)
        or len(set(checksums)) != len(checksums)
    ):
        msg = "New resource panel is incomplete, failed or duplicated; no favorable subset total emitted."
        raise ValueError(msg)
    if kind == "confirmation":
        design, _ = checked(root / "DESIGN.json", ledger, protocol["design_checksum"])
        if any(protocol.get(key) != value for key, value in design.items()):
            msg = "Confirmation resource protocol differs from its frozen design."
            raise ValueError(msg)
        prepared, _ = _phase_resource(root / "preparation", ledger)
        if (
            pilot.digest(prepared) != protocol["preparation_checksum"]
            or prepared["cpu_seconds"] != protocol["preparation_cpu_seconds"]
        ):
            msg = "Confirmation batch preparation resource receipt changed."
            raise ValueError(msg)
        _partition_timer(root, {}, ledger, root / "preparation/ATTEMPTS.json", [], category="setup")
    found = set()
    adapters = {
        "common_control": _common_control,
        "state_compilation": _state_compilation,
        "lazy_initializer": _lazy_reproduction,
        "compiled_composition": _compiled_resource,
        "confirmation": _confirmation_resource,
        "mps_memory": _mps_memory_resource,
    }
    for target, replica, label in entries:
        child = root / f"{target}__{replica}"
        if label is not None:
            child /= label
        record, proof = checked(child / "RESULT.json", ledger)
        metadata = record["value"] if kind == "common_control" else record
        if kind == "mps_memory":
            source = next(
                entry
                for entry in protocol["input_panel"]["entries"]
                if (entry["target_id"], entry["replica"], entry["label"]) == (target, replica, label)
            )
            if record["signature"] != pilot.digest([protocol["version"], [protocol, source]]):
                msg = "Memory resource endpoint differs from its frozen source identity."
                raise ValueError(msg)
            metadata = {"target_id": target, "replica": replica, "label": label, "status": record["status"]}
        if (
            metadata.get("target_id") != target
            or metadata.get("replica", 11) != replica
            or metadata.get("status") != "complete"
            or (label is not None and metadata.get("label") != label)
        ):
            msg = "New resource case identity, completion or source label differs from its panel."
            raise ValueError(msg)
        if kind == "compiled_composition" and (
            record.get("protocol_checksum") != pilot.digest(protocol)
            or set(record.get("arms", {})) != set(protocol["methods"])
        ):
            msg = "Compiled resource case protocol or complete method grid changed."
            raise ValueError(msg)
        if kind == "confirmation" and (
            record.get("parent_checksum") != pilot.digest(protocol)
            or set(record.get("panels", {})) != {"primary", "all_trained_sensitivity", "full_cap"}
            or any(
                panel.get("status") != "complete" or set(panel.get("arms", {})) != set(protocol["arms"])
                for panel in record["panels"].values()
            )
        ):
            msg = "Confirmation resource case parent or complete fair-control panel changed."
            raise ValueError(msg)
        adapters[kind](child, record, ledger, proof)
        found.add(pilot.digest(record))
    if found != set(checksums):
        msg = "New resource completion hashes do not cover every authenticated case."
        raise ValueError(msg)


def expected_cases(root: Path, kind: str, protocol: dict[str, Any]) -> list[tuple[Path, str, int, str | None]]:
    """Derive the complete case grid from each known frozen panel schema."""
    if kind == "estimator":
        return []
    spec = protocol["protocol"] if kind == "runner" else protocol["spec"] if kind == "synthesis" else protocol
    if kind == "mps_physical":
        selected = protocol["input_panel"]["cases"]
        if not selected or len({(c["target_id"], c["replica"]) for c in selected}) != len(selected):
            msg = "Empty or duplicated MPS physical resource case grid."
            raise ValueError(msg)
        return [
            (root / f"{c['target_id']}__{c['replica']}" / "RESULT.json", c["target_id"], c["replica"], None)
            for c in selected
        ]
    targets = spec["targets"]
    replicas = spec.get("replicas", [spec.get("replica", 11)])
    methods: Sequence[str | None] = list(spec["methods"]) if kind in {"runner", "svrg"} else [None]
    if kind == "compilation":
        methods = spec["labels"]
    result = []
    for target in targets:
        for replica in replicas:
            for method in methods:
                name = f"{target}__{replica}" + (f"__{method}" if method is not None else "")
                path = (
                    root / (name + ".json")
                    if kind in {"runner", "compilation"}
                    else root / name / ("AUDIT.json" if kind == "construction" else "RESULT.json")
                )
                result.append((path, target, replica, method))
    if not result or len({item[0] for item in result}) != len(result):
        msg = "Empty or duplicated resource protocol case grid."
        raise ValueError(msg)
    return result


def _case_methods(kind: str, protocol: dict[str, Any]) -> set[str] | None:
    """Require every fresh and declared reference arm, never silently pool a subset."""
    if kind in {"ensemble", "geometry", "long_ensemble"}:
        return set(protocol["arms"])
    if kind in {"ensemble_structural", "risk", "initializer", "composition"}:
        return set(protocol["methods"])
    if kind == "mixture":
        return set(protocol["methods"]) | {"structural_mean2_lbfgs", "structural_mean2_adam", "fixed_mean2_lbfgs"}
    if kind == "structural":
        return {
            "structural_lbfgs",
            "fixed_lbfgs",
            "unchanged_reference",
            "historical_structural_adam",
            "historical_structural_adaptive_online",
        }
    if kind == "baseline":
        return {
            "structural_exact_adam",
            "structural_weighted_adam",
            "structural_adaptive_online",
            "fixed_adaptive_online",
            "structural_lbfgs",
            "fixed_lbfgs",
        }
    if kind == "synthesis":
        return set(protocol["spec"]["methods"]) | {
            "reference_" + name for name in protocol["spec"]["reference_methods"]
        }
    if kind == "compilation":
        return {"compiled", "uncompiled"}
    return None


def collect_panel(root: Path, ledger: Ledger) -> None:
    """Authenticate and account one panel, releasing each case before loading the next.

    Raises:
        ValueError: If schema, checksum, grid, method coverage or failure accounting is incomplete.
    """
    kind, protocol, completion = authenticate(root, ledger)
    if kind in NEW_KINDS:
        previous = ledger.panel_override
        ledger.panel_override = root.name
        try:
            _collect_new_panel(root, kind, protocol, completion, ledger)
        finally:
            ledger.panel_override = previous
        return
    complete = completion.get("status") in {"complete", "complete_with_failures"}
    if kind in {"mps_overlap", "mps_physical"}:
        complete = bool(completion["cases"]) and all(c["status"] == "complete" for c in completion["cases"].values())
    if not complete:
        msg = f"Resource panel is not complete: {root}."
        raise ValueError(msg)
    if kind == "estimator":
        _estimator(root, protocol, completion, ledger)
        return
    index_name = (
        "runs"
        if kind in {"runner", "svrg"}
        else "cases"
        if kind in {"compilation", "baseline", "mixture"}
        else "case_checksums"
        if kind in {"construction", "initializer", "composition"}
        else "cases"
        if kind in {"mps_overlap", "mps_physical"}
        else "results"
    )
    index = completion[index_name]
    expected_checksums = (
        [case["checksum"] for case in index.values()]
        if kind in {"mps_overlap", "mps_physical"}
        else list(index.values())
        if isinstance(index, dict)
        else index
    )
    cases = expected_cases(root, kind, protocol)
    if len(expected_checksums) != len(cases) or len(set(expected_checksums)) != len(expected_checksums):
        msg = "Resource completion index does not cover every expected case exactly once."
        raise ValueError(msg)
    found = set()
    for path, target, replica, method in cases:
        record, proof = checked(path, ledger)
        found.add(pilot.digest(record))
        metadata = record["job"] if kind == "runner" else record
        if metadata.get("target_id") != target or metadata.get("replica") != replica:
            msg = f"Resource case identity differs from its frozen panel: {path}."
            raise ValueError(msg)
        if kind in {"runner", "svrg"} and metadata.get("label" if kind == "runner" else "method") != method:
            msg = "Resource case method differs from its protocol."
            raise ValueError(msg)
        required = _case_methods(kind, protocol)
        if required is not None and set(record.get("arms", {})) != required:
            msg = f"Resource panel has missing or unexpected methods: {path}."
            raise ValueError(msg)
        if record.get("status") != "complete" and kind != "compilation":
            msg = f"Incomplete resource case requires explicit failed-case adapter; no partial total emitted: {path}."
            raise ValueError(msg)
        if kind in {"mps_overlap", "mps_physical"} and set(record.get("endpoints", {})) != {
            "original108",
            "structural_mean2_lbfgs",
        }:
            msg = "MPS resource case omits a fixed endpoint."
            raise ValueError(msg)
        if kind == "long_ensemble":
            _long_ensemble(path.parent, record, ledger, proof)
        elif kind == "composition":
            _composition(path.parent, record, ledger, proof)
        elif kind in {"mps_overlap", "mps_physical"}:
            _mps_diagnostic(path.parent, record, ledger, proof, protocol, physical=kind == "mps_physical")
        elif kind == "runner":
            if record["job"]["runtime_checksum"] != pilot.digest(protocol):
                msg = "Runner resource case refers to another protocol."
                raise ValueError(msg)
            _runner_costs(record, ledger, root.name, proof, path.stem)
        elif kind == "svrg":
            _svrg(path.parent, record, ledger, proof)
        elif kind in {"ensemble", "geometry"}:
            _ensemble(path.parent, record, ledger, proof, geometry=kind == "geometry")
        elif kind == "structural":
            _structural(path.parent, record, ledger, proof)
        elif kind == "baseline":
            _baseline(path.parent, record, ledger, proof)
        elif kind == "ensemble_structural":
            _ensemble_structural(path.parent, record, ledger, proof)
        elif kind == "synthesis":
            _synthesis(path.parent, record, ledger, proof, protocol)
        elif kind == "compilation":
            _compilation(root, record, ledger, proof)
        elif kind == "risk":
            _risk(path.parent, record, ledger, proof)
        elif kind == "mixture":
            _mixture(path.parent, record, ledger, proof, protocol)
        elif kind == "initializer":
            _initializer(path.parent, record, ledger, proof)
        elif kind == "construction":
            manifests = {(item["target_id"], item["replica"]): item for item in protocol["cases"]}
            checked(path.parent / "PROTOCOL.json", ledger, manifests[target, replica]["manifest_checksum"])
            _construction(path.parent, record, ledger, proof)
        del record, metadata
    if found != set(expected_checksums):
        msg = "Resource case hashes disagree with the complete panel index."
        raise ValueError(msg)


def collect(roots: Sequence[Path]) -> dict[str, Any]:
    """Return a consolidated authenticated ledger; copied panels do not add physical work."""
    if not roots:
        msg = "At least one explicit or discovered frozen panel is required."
        raise ValueError(msg)
    ledger = Ledger()
    for root in sorted({path.resolve() for path in roots}):
        collect_panel(root, ledger)
    return ledger.result()


def main(argv: Sequence[str] | None = None) -> int:
    """Audit frozen artifacts without simulation and write a signed JSON ledger plus flat event CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, action="append", default=[])
    parser.add_argument("--discover", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    roots = args.panel
    if args.discover is not None:
        roots.extend(
            path
            for path in args.discover.iterdir()
            if path.is_dir() and ((path / "PROTOCOL.json").exists() or (path / "BATCH_PROTOCOL.json").exists())
        )
    cpu, wall = time.process_time(), time.perf_counter()
    result = collect(roots)
    result["accounting_audit"] = {
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall,
        "scope": (
            "this read-only source and JSON accounting pass; engineering, excluded from scientific physical event sums"
        ),
    }
    result["accounting_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    pilot.write_record(args.output / "LEDGER.json", result)
    with (args.output / "EVENTS.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("id", "category", "panels", "labels", *FIELDS, "count_quality", "scope")
        )
        writer.writeheader()
        for event in result["physical_events"]:
            writer.writerow({
                "id": event["id"],
                "category": event["category"],
                "panels": ";".join(event["panels"]),
                "labels": ";".join(event["labels"]),
                **event["measurements"],
                "count_quality": event["count_quality"],
                "scope": event["scope"],
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
