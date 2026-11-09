#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
# SPDX-License-Identifier: MIT
# Licensed under the MIT License

"""Summarize fallback frequency, event accuracy, and matched convergence checks."""

from __future__ import annotations

import csv
import itertools
import json
from collections import Counter
from pathlib import Path

import fallback_pilot as pilot
import numpy as np

STAGES = ("scan", "frame", "infeasible", "no_minimizer", "solver_failure", "gauge_failure", "unresolved")


def pooled_rates(arms):
    """Pool numerators and denominators; use pre-correction accuracy for selection."""
    calls = sum(a["counts"]["scalar_calls"] for a in arms)
    attempts = sum(a["counts"]["fallback_attempts"] for a in arms)
    qualified = sum(a["qualified_calls"] for a in arms)
    qualified_events = Counter(e for a in arms for e in a["qualified_events"])
    all_events = Counter(e["stage"] for a in arms for e in a["events"])
    if sum(all_events.values()) != attempts or sum(qualified_events.values()) > qualified:
        raise ValueError("Event counts do not agree with correction counters")
    return {
        "trajectories": len(arms),
        "calls": calls,
        "fallback_attempts": attempts,
        "rate": attempts / calls if calls else None,
        "stages": {s: all_events[s] for s in STAGES},
        "qualified_calls": qualified,
        "qualified_fallback_attempts": sum(qualified_events.values()),
        "qualified_rate": sum(qualified_events.values()) / qualified if qualified else None,
        "qualified_stages": {s: qualified_events[s] for s in STAGES},
    }


def physical_group(result):
    """Match physics and actual prepared states independently of dt and cap."""
    case = result["case"].copy()
    for field in ("dt", "cap"):
        case.pop(field)
    return json.dumps({"case": case, "initial": result["initial_vector_sha256"]}, sort_keys=True)


def compare_traces(left, right, tolerance):
    """Compare observables at common physical times, without interpolating states."""
    lrows = {round(t["time"], 10): t for t in left["trace"]}
    rrows = {round(t["time"], 10): t for t in right["trace"]}
    common = sorted(lrows.keys() & rrows.keys())
    error = max(
        (float(np.max(np.abs(np.array(lrows[t]["observables"]) - rrows[t]["observables"]))) for t in common),
        default=None,
    )
    full_coverage = len(common) == min(len(lrows), len(rrows)) and max(lrows) == max(rrows)
    # Accuracy is checked at all recorded endpoints in both arms, not only the
    # common coarse grid, and remains distinct from pairwise observable agreement.
    accurate = all(t["within_thresholds"] for t in [*left["trace"], *right["trace"]])
    return {
        "common_times": len(common),
        "full_coarse_grid_coverage": full_coverage,
        "max_observable_difference": error,
        "observable_agreement": full_coverage and error is not None and error <= tolerance,
        "both_within_reference_thresholds": accurate,
        "max_reference_infidelities": [max(t["infidelity"] for t in a["trace"]) for a in (left, right)],
    }


def convergence_pairs(results, tolerance):
    """Report adjacent dt and cap comparisons only within identical physical setups."""
    rows = []
    for left, right in itertools.combinations(results, 2):
        if physical_group(left) != physical_group(right):
            continue
        a, b = left["case"], right["case"]
        axis = (
            "dt"
            if a["cap"] == b["cap"] and a["dt"] != b["dt"]
            else ("cap" if a["dt"] == b["dt"] and a["cap"] != b["cap"] else None)
        )
        if axis is None:
            continue
        other = "cap" if axis == "dt" else "dt"
        lower, upper = sorted((a[axis], b[axis]))
        if any(
            physical_group(r) == physical_group(left)
            and r["case"][other] == a[other]
            and lower < r["case"][axis] < upper
            for r in results
        ):
            continue
        # Independently estimated normalization may differ by roundoff, but never
        # compare trajectories with materially different Hamiltonian scales.
        if not np.isclose(left["hamiltonian"]["scale"], right["hamiltonian"]["scale"], rtol=1e-11, atol=0):
            raise ValueError("Matched cases have inconsistent Hamiltonian normalization")
        for arm in left["arms"]:
            counterpart = next(v for v in right["arms"] if v["variant"] == arm["variant"])
            rows.append({
                "left": left["key"],
                "right": right["key"],
                "axis": axis,
                "variant": arm["variant"],
                **compare_traces(arm, counterpart, tolerance),
            })
    return rows


def report(output):
    """Write JSON and CSV reports; disclose missing cases and avoid early conclusions."""
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["signature"]["mode"] != "run":
        raise ValueError("Use the audit.json report for an audit directory")
    plan = manifest["signature"]["plan"]
    results = [json.loads(p.read_text()) for p in sorted(output.glob("cases/*/result.json"))]
    expected = {s["key"]: s for s in plan["studies"]}
    for r in results:
        if (
            r.get("status") != "complete"
            or r["key"] not in expected
            or r["specification"] != expected[r["key"]]
            or r["thresholds"] != plan["thresholds"]
        ):
            raise ValueError("Report contains an incomplete or incompatible case")
    groups = []
    for study in ("convergence", "larger"):
        for variant in pilot.VARIANTS:
            arms = [
                a for r in results if r["specification"]["study"] == study for a in r["arms"] if a["variant"] == variant
            ]
            groups.append({"study": study, "variant": variant, **pooled_rates(arms)})
    events = [
        {
            "key": r["key"],
            "model": r["case"]["model"],
            "length": r["case"]["length"],
            "cap": r["case"]["cap"],
            "dt": r["case"]["dt"],
            "variant": a["variant"],
            "time": e["physical_time"],
            "stage": e["stage"],
            **{f"{where}_{k}": v for where in ("pre", "post") for k, v in e[f"accuracy_{where}"].items()},
        }
        for r in results
        for a in r["arms"]
        for e in a["events"]
    ]
    probes = [
        {"key": r["key"], "variant": a["variant"], **{k: v for k, v in p.items() if k != "observables"}}
        for r in results
        for a in r["arms"]
        for p in a["probes"]
    ]
    missing = sorted(expected.keys() - {r["key"] for r in results})
    summary = {
        "thresholds": plan["thresholds"],
        "completed_cases": len(results),
        "planned_cases": len(expected),
        "missing_cases": missing,
        "complete": not missing,
        "groups": groups,
        "events": events,
        "probes": probes,
        "convergence": convergence_pairs(results, plan["thresholds"]["observable"]),
        "interpretation": "Rates use correction calls. Qualified calls satisfy both pre-correction error thresholds. "
        "Observable agreement is reported separately from reference accuracy; neither proves asymptotic convergence. "
        "A zero qualified denominator is undefined, not evidence of zero fallback probability.",
    }
    pilot.write_json(output / "accuracy_summary.json", summary)
    for filename, rows in (("accuracy_events.csv", events), ("accuracy_probes.csv", probes)):
        with (output / filename).open("w", newline="") as stream:
            fields = list(rows[0]) if rows else ["key", "variant", "time"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Accuracy-controlled fallback study",
        "",
        f"Completed {len(results)} / {len(expected)} planned cases.",
        "",
        "Thresholds: infidelity <= {infidelity:g}; maximum spin-observable error <= {observable:g}.".format(
            **plan["thresholds"]
        ),
        "",
        "| Study | Arm | Calls | Fallbacks | Qualified calls | Qualified fallbacks |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {g['study']} | {g['variant']} | {g['calls']} | {g['fallback_attempts']} | {g['qualified_calls']} | {g['qualified_fallback_attempts']} |"
        for g in groups
    ]
    lines += [
        "",
        "Scan recovery, full-frame recovery, infeasibility, and numerical failures are separate in accuracy_summary.json.",
        "Event errors are in accuracy_events.csv; errors at the original event times are in accuracy_probes.csv.",
        "",
        summary["interpretation"],
        "",
    ]
    (output / "ACCURACY_REPORT.md").write_text("\n".join(lines))
    return summary
