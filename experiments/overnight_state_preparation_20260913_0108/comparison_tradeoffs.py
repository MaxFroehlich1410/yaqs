# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compute descriptive physical-resource frontiers from authenticated endpoints.

Callers must authenticate their scientific source records before constructing
points. This module never loads targets, selects optimizer checkpoints or treats
different development panels as matched experiments.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from experiments.overnight_state_preparation_20260913_0108 import channels

if TYPE_CHECKING:
    from collections.abc import Sequence

VERSION = "overnight.descriptive_resource_frontiers.v1"
RESOURCES = ("one_qubit", "two_qubit", "parameters", "depth", "cpu_seconds")
OBJECTIVES = ("one_qubit", "two_qubit", "cpu_seconds")


def endpoint(
    target: str, replica: int, method: str, score: dict[str, Any], cpu_seconds: float | None
) -> dict[str, Any]:
    """Form one all-channel endpoint without silently replacing unknown resources.

    Args:
        target: Immutable target identity from an authenticated source.
        replica: Predeclared training replica.
        method: Complete method identity within one comparison panel.
        score: Authenticated all12 score with physical and Ballarin counts.
        cpu_seconds: Charged deployable procedure CPU, excluding diagnostic scoring.

    Returns:
        One descriptive endpoint using the equal mean of all eleven noisy scores.

    Raises:
        ValueError: If a fidelity, identity or known resource is invalid.
    """
    fidelities = score["fidelities"]
    if (
        not target
        or not method
        or type(replica) is not int
        or set(fidelities) != set(channels.NOISE_IDS)
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not -1e-10 <= v <= 1 + 1e-10
            for v in fidelities.values()
        )
    ):
        msg = "Tradeoff endpoints require valid identities and all twelve finite exact fidelities."
        raise ValueError(msg)
    result = {
        "target_id": target,
        "replica": replica,
        "method": method,
        "fidelity": statistics.mean(fidelities[n] for n in channels.NOISE_IDS if n != "noiseless"),
        "noiseless_fidelity": fidelities["noiseless"],
        **{key: score["physical_counts"].get(key) for key in ("one_qubit", "two_qubit", "depth")},
        "parameters": score.get("parameters"),
        "cpu_seconds": cpu_seconds,
        "ballarin_counts": score.get("ballarin_counts"),
    }
    for key in RESOURCES:
        value = result[key]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (key != "cpu_seconds" and int(value) != value)
        ):
            msg = "Known endpoint resources must be finite nonnegative counts or CPU seconds."
            raise ValueError(msg)
    return result


def frontier(points: Sequence[dict[str, Any]], resources: Sequence[str] = OBJECTIVES) -> dict[str, Any]:
    """Return strict Pareto dominance at recorded precision, preserving exact ties.

    Fidelity is maximized and every supplied resource is minimized. No tolerance
    can turn a real fidelity loss into dominance. Unknown dimensions or missing
    measurements make the whole requested frontier unavailable.
    """
    if not points or not resources or len(set(resources)) != len(resources) or set(resources) - set(RESOURCES):
        msg = "A frontier needs endpoints and distinct supported resource dimensions."
        raise ValueError(msg)
    if len({p["method"] for p in points}) != len(points):
        msg = "Frontier method identities must be distinct within one matched panel."
        raise ValueError(msg)
    missing = [p["method"] for p in points if any(p.get(k) is None for k in resources)]
    if missing:
        return {"status": "unknown_resources", "missing_methods": missing, "nondominated": None}
    for point in points:
        if (
            any(
                isinstance(point[k], bool)
                or not isinstance(point[k], (int, float))
                or not math.isfinite(point[k])
                or point[k] < (-1e-10 if k == "fidelity" else 0)
                for k in ("fidelity", *resources)
            )
            or point["fidelity"] > 1 + 1e-10
        ):
            msg = "Frontier values must be finite physical fidelities and nonnegative resources."
            raise ValueError(msg)
    dominated_by = {}
    for point in points:
        dominant = []
        for alternative in points:
            if alternative["method"] == point["method"]:
                continue
            weak = alternative["fidelity"] >= point["fidelity"] and all(alternative[k] <= point[k] for k in resources)
            strict = alternative["fidelity"] > point["fidelity"] or any(alternative[k] < point[k] for k in resources)
            if weak and strict:
                dominant.append(alternative["method"])
        dominated_by[point["method"]] = dominant
    return {
        "status": "complete",
        "maximize": ["fidelity"],
        "minimize": list(resources),
        "dominance_precision": "strict comparison of recorded values; exact ties remain nondominated",
        "nondominated": [p["method"] for p in points if not dominated_by[p["method"]]],
        "dominated_by": dominated_by,
    }


def summarize(
    points: Sequence[dict[str, Any]], targets: Sequence[str], replicas: Sequence[int], methods: Sequence[str]
) -> dict[str, Any]:
    """Build per-case and target-balanced frontiers only for the full declared panel.

    Arithmetic resource means describe these fixed cases. They are neither a
    guarantee for every target nor a frontier over all possible algorithms.
    """
    axes = (targets, replicas, methods)
    if any(not a or len(set(a)) != len(a) for a in axes):
        msg = "Each declared tradeoff axis must be nonempty and contain distinct identities."
        raise ValueError(msg)
    actual = {(p["target_id"], p["replica"], p["method"]): p for p in points}
    expected = {(t, r, m) for t in targets for r in replicas for m in methods}
    if len(actual) != len(points) or set(actual) != expected:
        return {
            "status": "incomplete_panel",
            "expected_points": len(expected),
            "received_points": len(points),
            "missing": [list(key) for key in sorted(expected - set(actual))],
            "extra": [list(key) for key in sorted(set(actual) - expected)],
            "aggregate": None,
        }
    cases = {}
    for target in targets:
        for replica in replicas:
            selected = [actual[target, replica, method] for method in methods]
            cases[f"{target}__{replica}"] = {
                "physical_resource_frontier": frontier(selected),
                "two_qubit_frontier": frontier(selected, ("two_qubit",)),
            }
    averages = []
    for method in methods:
        values: dict[str, list[float | None]] = defaultdict(list)
        for key in ("fidelity", "noiseless_fidelity", *RESOURCES):
            for target in targets:
                per_replica = [actual[target, replica, method][key] for replica in replicas]
                values[key].append(None if any(v is None for v in per_replica) else statistics.mean(per_replica))
        averages.append({
            "method": method,
            **{key: None if any(v is None for v in rows) else statistics.mean(rows) for key, rows in values.items()},
        })
    return {
        "version": VERSION,
        "status": "complete",
        "weighting": "replicas equally within each target, then targets equally; eleven noisy profiles equally",
        "resource_scope": "physical pre-Ballarin circuit counts; charged deployable procedure CPU; evaluation separate",
        "comparison_limit": "descriptive fixed-panel frontier, not a statistical dominance or scalability claim",
        "cases": cases,
        "aggregate": {
            "points": averages,
            "physical_resource_frontier": frontier(averages),
            "two_qubit_frontier": frontier(averages, ("two_qubit",)),
        },
    }
