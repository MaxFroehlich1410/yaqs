# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Analyze saved logical-noise selections without running or selecting circuits.

CPU comparisons charge each complete candidate grid once, including discarded
candidates. Compiled resources are reported separately from logical noise gates.
Missing or failed endpoints remain missing; they never become zero fidelity.
"""

from __future__ import annotations

import argparse
import json
import math
import operator
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

OURS = "current_b_lbfgs_frozen"
CONTROLS = frozenset({"fixed_vqsp", "layerwise"})
EXTERNAL_METHODS = frozenset({OURS, "nacl", "robuststate", "fontana"})
PRIOR_TARGETS = frozenset({"gaussian_mu0p5_sigma0p1", "tfim_critical", "haar_random_1"})
EPSILON = 1e-12


def _number(value: object) -> float | None:
    """Return a finite scalar, preserving unavailable or invalid values as null."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _sum(values: Sequence[Any]) -> float | None:
    """Sum a complete numeric ledger without treating missing entries as zero."""
    numbers = [_number(value) for value in values]
    return math.fsum(value for value in numbers if value is not None) if None not in numbers else None


def _costs(group: dict[str, Any]) -> dict[str, Any]:
    """Extract one group's historical and current costs independently of grid size."""
    reused_evaluation = bool(group.get("reused_evaluation"))
    build = _number(group.get("build_train_cpu_seconds"))
    new_build = 0.0 if reused_evaluation else _number(group.get("new_build_train_cpu_seconds"))
    if new_build is None and not reused_evaluation:
        new_build = 0.0 if group.get("training_reused") else build
    grid = {
        "compilation": _sum([row.get("compilation", {}).get("compilation_cpu_seconds") for row in group["records"]]),
        "validation": _sum([row.get("validation", {}).get("cpu_seconds") for row in group["records"]]),
        "final": _sum([row.get("cpu_seconds") for row in group["final_scores"]]),
    }
    construction = _number(group.get("construction", {}).get("construction_cpu_seconds"))
    historical_selection = _number(group.get("historical_selection_cpu_seconds", 0.0))
    return {
        "build_train_cpu_seconds": build,
        "new_build_train_cpu_seconds": new_build,
        "reused_build_train_cpu_seconds": build - new_build if build is not None and new_build is not None else None,
        "historical_selection_cpu_seconds": historical_selection,
        "recorded_load_or_construction_cpu_seconds": construction,
        "current_load_or_construction_cpu_seconds": 0.0 if reused_evaluation else construction,
        **{f"grid_{name}_cpu_seconds": cost for name, cost in grid.items()},
        **{f"current_{name}_cpu_seconds": 0.0 if reused_evaluation else cost for name, cost in grid.items()},
        "construction_and_selection_cpu_seconds": _sum([
            build,
            historical_selection,
            grid["compilation"],
            grid["validation"],
        ]),
    }


def _rows(group: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join exact diagnostics by candidate identity while preserving the locked index.

    Raises:
        ValueError: If identities, candidate scores or a selected index are inconsistent.
    """
    identity = group["context"]["identity"]
    selected = group["selected_index"]
    records = group["records"]
    if selected is not None and (type(selected) is not int or not 0 <= selected < len(records)):
        msg = "Saved selected_index must identify a recorded candidate or be null."
        raise ValueError(msg)
    scores = {score["candidate_id"]: score for score in group["final_scores"]}
    ids = [record["method"]["candidate_id"] for record in records]
    if len(scores) != len(group["final_scores"]) or len(set(ids)) != len(ids) or set(scores) != set(ids):
        msg = "Candidate and exact-score identities must be unique and match."
        raise ValueError(msg)
    costs = _costs(group)
    phase = group.get(
        "phase", "prior_three_target_pilot" if identity["target_id"] in PRIOR_TARGETS else "six_target_extension"
    )
    shared = {
        "target": identity["target_id"],
        "method": identity["method"],
        "seed": identity.get("seed"),
        "hardware": identity.get("hardware"),
        "phase": phase,
        "training_reused": bool(group.get("training_reused")),
        "reused_evaluation": bool(group.get("reused_evaluation")),
        "source_output": group.get("source_output"),
        **costs,
    }
    rows = []
    for index, record in enumerate(records):
        method = record["method"]
        if method["method"] != identity["method"]:
            msg = "Candidate method differs from its group identity."
            raise ValueError(msg)
        score = scores[method["candidate_id"]]
        logical, compiled, validation = record["logical_resources"], record["compilation"], record["validation"]
        rows.append({
            **shared,
            "candidate_id": method["candidate_id"],
            "candidate_index": index,
            "selected": index == selected,
            "status": method.get("status"),
            "eligible": bool(record.get("eligible", False)),
            "exact_fidelity": _number(score.get("exact", {}).get("fidelity")),
            "ideal_fidelity": _number(logical.get("ideal_fidelity")),
            **{
                key: logical.get(key)
                for key in (
                    "logical_one_qubit",
                    "logical_two_qubit",
                    "logical_two_qubit_depth",
                    "logical_total_depth",
                    "logical_operations",
                )
            },
            "cx": compiled.get("compiled_two_qubit"),
            "cx_depth": compiled.get("compiled_two_qubit_depth"),
            "compiled_one_qubit": compiled.get("compiled_one_qubit"),
            "compiled_total_depth": compiled.get("compiled_total_depth"),
            "validation_fidelity": _number(validation.get("fidelity")),
            "validation_standard_error": _number(validation.get("standard_error")),
            "validation_jump_paths": validation.get("jump_paths"),
            "validation_paths": validation.get("sampled_paths"),
            "validation_coverage": record.get("validation_coverage"),
            "validation_cpu_seconds": _number(validation.get("cpu_seconds")),
            "final_cpu_seconds": _number(score.get("cpu_seconds")),
            "source_references": {
                "source_output": group.get("source_output"),
                "construction_cache": group.get("construction_cache"),
                "construction_reference": method.get("construction_reference"),
                "construction_receipt_sha256": group["context"].get("construction_receipt_sha256"),
                "selection_lock_sha256": group.get("selection_lock_sha256"),
                "target_sha256": group["context"].get("target_sha256"),
            },
        })
    chosen = rows[selected] if selected is not None else None
    summary = {
        **shared,
        "status": group.get("status"),
        "candidate_count": len(rows),
        "selected_index": selected,
        "selected_candidate_id": chosen["candidate_id"] if chosen else None,
        "selected_exact_fidelity": chosen["exact_fidelity"] if chosen else None,
        "selection_scope": group.get("selection_scope"),
        "selection_rule": group.get("selection_rule"),
    }
    return rows, summary


def _usable(row: dict[str, Any]) -> bool:
    """Require a successfully selected endpoint with available exact fidelity."""
    return row["selected"] and row["eligible"] and row["status"] == "complete" and row["exact_fidelity"] is not None


def dominates(left: dict[str, Any], right: dict[str, Any], *, include_resources: bool = False) -> bool:
    """Compare fidelity, full-grid build CPU and optionally logical entangling costs.

    Args:
        left: Proposed dominating endpoint, using analysis-row field names.
        right: Endpoint tested for domination.
        include_resources: Also minimize logical two-qubit count and depth.

    Returns:
        True only when every available objective is no worse and at least one
        is strictly better beyond a relative tolerance of ``1e-12``. Missing
        or nonfinite objectives prevent a dominance claim.
    """
    objectives = [("exact_fidelity", -1), ("build_train_cpu_seconds", 1)]
    if include_resources:
        objectives.extend([("logical_two_qubit", 1), ("logical_two_qubit_depth", 1)])
    strict = False
    for key, direction in objectives:
        a, b = _number(left.get(key)), _number(right.get(key))
        if a is None or b is None:
            return False
        tolerance = EPSILON * max(abs(a), abs(b))
        difference = direction * (a - b)
        if difference > tolerance:
            return False
        strict |= difference < -tolerance
    return strict


def _pareto(rows: list[dict[str, Any]], *, include_resources: bool) -> dict[str, Any]:
    """Report nondominated selected methods and all auditable dominators."""
    objectives = ["exact_fidelity", "build_train_cpu_seconds"]
    if include_resources:
        objectives.extend(["logical_two_qubit", "logical_two_qubit_depth"])
    usable = [row for row in rows if _usable(row) and all(_number(row.get(key)) is not None for key in objectives)]
    dominated = {
        row["method"]: [
            other["method"]
            for other in usable
            if other is not row and dominates(other, row, include_resources=include_resources)
        ]
        for row in usable
    }
    return {
        "objectives": objectives,
        "relative_epsilon": EPSILON,
        "frontier_methods": [method for method, others in dominated.items() if not others],
        "dominated_by": dominated,
        "excluded_methods": [row["method"] for row in rows if row not in usable],
    }


def _aggregates(groups: list[dict[str, Any]], rows: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    """Summarize coverage and costs once per group, never once per candidate."""
    result = {}
    for method in methods:
        method_groups = [group for group in groups if group["method"] == method]
        selected = [row for row in rows if row["method"] == method and _usable(row)]
        fidelities = [row["exact_fidelity"] for row in selected]
        result[method] = {
            "group_count": len(method_groups),
            "candidate_count": sum(group["candidate_count"] for group in method_groups),
            "selected_count": len(selected),
            "targets": [row["target"] for row in selected],
            "failed_or_missing_selection_targets": [
                group["target"] for group in method_groups if group["target"] not in {row["target"] for row in selected}
            ],
            "mean_exact_fidelity": statistics.mean(fidelities) if fidelities else None,
            "median_exact_fidelity": statistics.median(fidelities) if fidelities else None,
            "total_build_train_cpu_seconds": _sum([group["build_train_cpu_seconds"] for group in method_groups]),
            "total_new_build_train_cpu_seconds": _sum([
                group["new_build_train_cpu_seconds"] for group in method_groups
            ]),
            "total_construction_and_selection_cpu_seconds": _sum([
                group["construction_and_selection_cpu_seconds"] for group in method_groups
            ]),
        }
    return result


def _paired(rows: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    """Compare ours with each method on their shared successful target set."""
    ours = {row["target"]: row for row in rows if row["method"] == OURS and _usable(row)}
    comparisons = {}
    for method in methods:
        if method == OURS:
            continue
        others = {row["target"]: row for row in rows if row["method"] == method and _usable(row)}
        pairs: list[dict[str, Any]] = []
        for target in ours.keys() & others.keys():
            a, b = ours[target], others[target]
            gap = a["exact_fidelity"] - b["exact_fidelity"]
            tolerance = EPSILON * max(abs(a["exact_fidelity"]), abs(b["exact_fidelity"]))
            outcome = "tie" if abs(gap) <= tolerance else "win" if gap > 0 else "loss"
            a_cpu, b_cpu = a["build_train_cpu_seconds"], b["build_train_cpu_seconds"]
            pairs.append({
                "target": target,
                "ours_candidate_id": a["candidate_id"],
                "other_candidate_id": b["candidate_id"],
                "ours_exact_fidelity": a["exact_fidelity"],
                "other_exact_fidelity": b["exact_fidelity"],
                "ours_minus_other_pp": 100 * gap,
                "outcome_for_ours": outcome,
                "ours_minus_other_build_cpu_seconds": a_cpu - b_cpu
                if a_cpu is not None and b_cpu is not None
                else None,
            })
        pairs.sort(key=operator.itemgetter("target"))
        gaps = [pair["ours_minus_other_pp"] for pair in pairs]
        comparisons[method] = {
            "common_target_count": len(pairs),
            "targets": [pair["target"] for pair in pairs],
            "ours_only_targets": sorted(ours.keys() - others.keys()),
            "other_only_targets": sorted(others.keys() - ours.keys()),
            **{
                name: sum(pair["outcome_for_ours"] == outcome for pair in pairs)
                for name, outcome in (("wins", "win"), ("ties", "tie"), ("losses", "loss"))
            },
            "mean_gap_pp": statistics.mean(gaps) if gaps else None,
            "median_gap_pp": statistics.median(gaps) if gaps else None,
            "per_target": pairs,
        }
    return comparisons


def build_analysis(summary: dict[str, Any]) -> dict[str, Any]:
    """Derive reproducible tables from saved matched-logical group receipts.

    Args:
        summary: Existing logical-pilot or extended-suite summary. Additional
            execution metadata is permitted and never influences selection.

    Returns:
        JSON-compatible candidate/selected rows, coverage, cost ledger, paired
        comparisons and Pareto frontiers. Exact maxima are diagnostic only.

    Raises:
        ValueError: If group identities are duplicated or candidate receipts disagree.
    """
    candidates, groups = [], []
    identities = set()
    for group in summary["groups"]:
        identity = group["context"]["identity"]
        key = (identity["target_id"], identity["method"])
        if key in identities:
            msg = "Each target/method must occur once; mixed seeds or profiles need separate analyses."
            raise ValueError(msg)
        identities.add(key)
        rows, costs = _rows(group)
        candidates.extend(rows)
        groups.append(costs)
    exclusions = summary.get("exclusions", [])
    methods = list(dict.fromkeys([group["method"] for group in groups] + [item["method"] for item in exclusions]))
    targets = list(dict.fromkeys([group["target"] for group in groups] + [item["target"] for item in exclusions]))
    selected = [row for row in candidates if row["selected"]]
    disagreements = []
    for chosen in selected:
        if not _usable(chosen):
            continue
        alternatives = [
            row
            for row in candidates
            if row["target"] == chosen["target"]
            and row["method"] == chosen["method"]
            and row["eligible"]
            and row["status"] == "complete"
            and row["exact_fidelity"] is not None
        ]
        maximum = max(alternatives, key=operator.itemgetter("exact_fidelity"))
        if maximum["exact_fidelity"] > chosen["exact_fidelity"] + EPSILON * abs(maximum["exact_fidelity"]):
            disagreements.append({
                "target": chosen["target"],
                "method": chosen["method"],
                "selected_candidate_id": chosen["candidate_id"],
                "diagnostic_exact_max_candidate_id": maximum["candidate_id"],
                "exact_max_minus_selected_pp": 100 * (maximum["exact_fidelity"] - chosen["exact_fidelity"]),
                "selection_changed": False,
            })
    pareto = {}
    for target in targets:
        endpoints = [row for row in selected if row["target"] == target]
        pareto[target] = {
            label: {
                "fidelity_cpu": _pareto(subset, include_resources=False),
                "fidelity_cpu_logical_resources": _pareto(subset, include_resources=True),
            }
            for label, subset in (
                ("external_only", [row for row in endpoints if row["method"] in EXTERNAL_METHODS]),
                ("including_controls", endpoints),
            )
        }
    cost_keys = [key for key in groups[0] if key.endswith("cpu_seconds")] if groups else []
    return {
        "schema": "yaqs.logical_suite_analysis.v1",
        "source_schema": summary.get("schema"),
        "source_manifest_sha256": summary.get("manifest_sha256"),
        "selection_policy": "Retain saved selected_index; exact test scores never reselect an endpoint.",
        "cost_policy": (
            "Full group build/train CPU, including discarded candidates; sum group_costs once, not candidate_rows."
        ),
        "resource_policy": "Logical gates define noise; compiled CX counts and depths are resource-only diagnostics.",
        "coverage": {
            "targets": targets,
            "target_count": len(targets),
            "methods": methods,
            "group_count": len(groups),
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "successful_selected_count": sum(_usable(row) for row in selected),
            "exclusion_count": len(exclusions),
            "reused_evaluation_group_count": sum(group["reused_evaluation"] for group in groups),
            "new_training_group_count": sum((group["new_build_train_cpu_seconds"] or 0) > 0 for group in groups),
            "phase_groups": {
                phase: sum(group["phase"] == phase for group in groups)
                for phase in sorted({group["phase"] for group in groups})
            },
        },
        "exclusions": exclusions,
        "candidate_rows": candidates,
        "selected_rows": selected,
        "group_costs": groups,
        "cost_totals": {key: _sum([group[key] for group in groups]) for key in cost_keys},
        "aggregates": _aggregates(groups, selected, methods),
        "paired_comparisons": _paired(selected, methods),
        "pareto": pareto,
        "exact_max_diagnostics": disagreements,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Write an analysis artifact from a summary without changing experiment data.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Raises:
        ValueError: If the output would overwrite the input summary.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.summary.with_name("analysis.json")
    if output.resolve() == args.summary.resolve():
        msg = "Analysis output must differ from the input summary."
        raise ValueError(msg)
    result = build_analysis(json.loads(args.summary.read_text()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
