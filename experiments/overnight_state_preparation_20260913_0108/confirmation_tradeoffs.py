# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Read-only complete confirmation resource analysis and descriptive figures.

All selected endpoints come from the frozen parent's sealed case receipts.
No target vectors, numerical caches, optimizers or checkpoint selectors run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import best_candidate as parent
from experiments.overnight_state_preparation_20260913_0108 import comparison_tradeoffs as tradeoffs
from experiments.overnight_state_preparation_20260913_0108 import development_pareto_plot as style

if TYPE_CHECKING:
    from collections.abc import Sequence

VERSION = "overnight.complete_confirmation_tradeoffs.v1"
PANELS = ("primary", "all_trained_sensitivity", "full_cap")
METRICS = ("fidelity", "noiseless_fidelity", *tradeoffs.RESOURCES)
NOISY = tuple(n for n in parent.base.channels.NOISE_IDS if n != "noiseless")
NOISE_LABELS = {
    "ballarin_coupled": "Ballarin",
    "dephasing_1s_1q": "Dph 1s/1q",
    "dephasing_1s_2q": "Dph 1s/2q",
    "dephasing_1s_all": "Dph 1s/all",
    "dephasing_2s_2q": "Dph 2s/2q",
    "dephasing_1s2s_all": "Dph 1+2/all",
    "depolarizing_1s_1q": "Dep 1s/1q",
    "depolarizing_1s_2q": "Dep 1s/2q",
    "depolarizing_1s_all": "Dep 1s/all",
    "depolarizing_2s_2q": "Dep 2s/2q",
    "depolarizing_1s2s_all": "Dep 1+2/all",
}
FAMILIES = {
    "tfim_ferro": "TFIM",
    "tfim_para": "TFIM",
    "haar_random_2": "Haar",
    "haar_random_3": "Haar",
    "random_mps_bond3": "MPS",
}


def file_hash(path: Path) -> str:
    """Return the byte digest of a source or publication artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_cost(value: object) -> float:
    """Require measured CPU without converting an unknown or invalid value to zero.

    Raises:
        ValueError: If CPU is unknown, negative, nonfinite or boolean.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        msg = "Confirmation resource analysis requires known finite nonnegative CPU."
        raise ValueError(msg)
    return float(value)


def point(case: dict[str, Any], panel: str, method: str) -> dict[str, Any]:
    """Keep the actual deployable point cost and all profile scores separately from its cutoff."""
    selected = case["panels"][panel]
    arm = selected["arms"][method]
    cpu = finite_cost(case["initializer_cpu_seconds"]) + finite_cost(arm["point"]["cpu_seconds"])
    if panel == "full_cap" and cpu != finite_cost(arm["total_cpu_seconds"]):
        msg = "Full-cap endpoint CPU differs from its initializer plus chronological point."
        raise ValueError(msg)
    cutoff = None if panel == "full_cap" else finite_cost(selected["total_cpu_seconds"])
    if cutoff is not None and cpu > cutoff + 1e-10:
        msg = "A selected confirmation point exceeds its fixed common cutoff."
        raise ValueError(msg)
    result = tradeoffs.endpoint(case["target_id"], case["replica"], method, arm["scores"], cpu)
    if any(result[key] is None for key in tradeoffs.RESOURCES):
        msg = "All confirmation endpoint resources must be measured."
        raise ValueError(msg)
    return {
        **result,
        "panel": panel,
        "cutoff_cpu_seconds": cutoff,
        "refinement_cpu_seconds": arm["point"]["cpu_seconds"],
        "initializer_cpu_seconds": case["initializer_cpu_seconds"],
        "fidelities": arm["scores"]["fidelities"],
        "evaluation_checksum": arm["scores"]["evaluation_checksum"],
        "circuit_hash": arm["point"]["circuit_hash"],
        "parameter_checksum": pilot.digest(arm["point"]["theta"]),
    }


def average(points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Average an already balanced target/replica block, retaining every profile."""
    return {
        **{key: statistics.mean(p[key] for p in points) for key in METRICS},
        "fidelities": {
            noise: statistics.mean(p["fidelities"][noise] for p in points) for noise in parent.base.channels.NOISE_IDS
        },
    }


def difference(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Describe paired fidelity/resource changes and the ratio of mean CPU costs."""
    return {
        "candidate": {key: candidate[key] for key in METRICS},
        "baseline": {key: baseline[key] for key in METRICS},
        "delta": {key: candidate[key] - baseline[key] for key in METRICS},
        "cpu_ratio": candidate["cpu_seconds"] / baseline["cpu_seconds"] if baseline["cpu_seconds"] else None,
        "profile_deltas": {
            n: candidate["fidelities"][n] - baseline["fidelities"][n] for n in parent.base.channels.NOISE_IDS
        },
    }


def analyze(protocol: dict[str, Any], records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Retain the complete product and compute every paired target/family comparison.

    Raises:
        ValueError: If a case, role, score or resource is missing or invalid.
    """
    targets, replicas = protocol["targets"], protocol["replicas"]
    required = {(t, r) for t in targets for r in replicas}
    actual = {(r["target_id"], r["replica"]) for r in records}
    if actual != required or len(records) != len(required) or any(r["status"] != "complete" for r in records):
        msg = "Confirmation tradeoffs require the complete distinct successful target/replica panel."
        raise ValueError(msg)
    points = []
    for record in records:
        for panel in PANELS:
            selected = record["panels"][panel]
            if selected["status"] != "complete" or set(selected["arms"]) != set(parent.METHODS):
                msg = "Confirmation tradeoffs require every role in all three endpoint panels."
                raise ValueError(msg)
            points.extend(point(record, panel, method) for method in parent.METHODS)
    panels: dict[str, Any] = {}
    families = {
        family: [t for t in targets if FAMILIES.get(t, t) == family]
        for family in dict.fromkeys(FAMILIES.get(t, t) for t in targets)
    }
    for panel in PANELS:
        rows = [row for row in points if row["panel"] == panel]
        lookup = {(p["target_id"], p["replica"], p["method"]): p for p in rows}
        by_target = {t: {m: average([lookup[t, r, m] for r in replicas]) for m in parent.METHODS} for t in targets}
        by_family = {
            family: {m: average([by_target[t][m] for t in members]) for m in parent.METHODS}
            for family, members in families.items()
        }
        means = {m: average([by_target[t][m] for t in targets]) for m in parent.METHODS}
        comparisons = {}
        for baseline in parent.METHODS:
            if baseline == parent.CANDIDATE:
                continue
            comparisons[baseline] = {
                "aggregate": difference(means[parent.CANDIDATE], means[baseline]),
                "by_case": {
                    f"{t}__{r}": difference(lookup[t, r, parent.CANDIDATE], lookup[t, r, baseline])
                    for t in targets
                    for r in replicas
                },
                "by_target": {t: difference(by_target[t][parent.CANDIDATE], by_target[t][baseline]) for t in targets},
                "by_family": {f: difference(by_family[f][parent.CANDIDATE], by_family[f][baseline]) for f in families},
            }
        descriptive = tradeoffs.summarize(rows, targets, replicas, parent.METHODS)
        panels[panel] = {
            "descriptive": descriptive,
            "by_target": by_target,
            "by_family": by_family,
            "target_frontiers": {
                t: tradeoffs.frontier([{"method": m, **by_target[t][m]} for m in parent.METHODS]) for t in targets
            },
            "family_frontiers": {
                f: tradeoffs.frontier([{"method": m, **by_family[f][m]} for m in parent.METHODS]) for f in families
            },
            "comparisons": comparisons,
            "strongest_control": max(
                (m for m in parent.METHODS if m != parent.CANDIDATE), key=lambda m: means[m]["fidelity"]
            ),
        }
    return {
        "version": VERSION,
        "status": "complete",
        "targets": targets,
        "replicas": replicas,
        "methods": list(parent.METHODS),
        "points": points,
        "panels": panels,
        "frozen_gate": parent.analyze(records, targets, replicas),
        "weighting": (
            "eleven noisy profiles equally; replicas equally within target; targets equally; "
            "family means within family only"
        ),
        "resource_scope": (
            "actual deployable point CPU including initializer; gate counts pre-Ballarin; "
            "final evaluation and physical case overhead separate"
        ),
        "physical_invocations": {f"{r['target_id']}__{r['replica']}": r["physical_invocation"] for r in records},
        "actual_experiment_cpu_seconds": finite_cost(protocol["preparation_cpu_seconds"])
        + sum(finite_cost(r["physical_invocation"]["cpu_seconds"]) for r in records),
        "comparison_limit": (
            "descriptive finite-panel resources; full-cap and sensitivity cannot rescue the fixed primary gate"
        ),
    }


def authenticate(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Authenticate fixed parent and complete cases using only read-only public APIs.

    Raises:
        ValueError: If the source is not the complete confirmation parent or its summary differs.
    """
    protocol = parent.read(root / "PROTOCOL.json")
    parent.verify_parent(root, protocol)
    if (
        protocol["phase"] != "confirmation"
        or protocol["targets"] != list(parent.TARGETS)
        or protocol["replicas"] != list(parent.REPLICAS)
    ):
        msg = "Scientific tradeoff export requires the entire fixed confirmation panel."
        raise ValueError(msg)
    records = [parent.read_case(root, protocol, case) for case in protocol["cases"]]
    summary = parent.read(root / "SUMMARY.json")
    expected_cases = [
        {
            "target_id": r["target_id"],
            "replica": r["replica"],
            "status": r["status"],
            "checksum": pilot.digest(r),
            "failure": r.get("failure"),
            "physical_cpu_seconds": r["physical_invocation"]["cpu_seconds"],
        }
        for r in records
    ]
    expected_summary = {
        "version": parent.VERSION,
        "parent_checksum": pilot.digest(protocol),
        "configuration_sha256": protocol["configuration_sha256"],
        "status": "complete",
        "analysis": parent.analyze(records, protocol["targets"], protocol["replicas"]),
        "cases": expected_cases,
        "preparation_cpu_seconds": protocol["preparation_cpu_seconds"],
        "actual_complete_experiment_cpu_seconds": protocol["preparation_cpu_seconds"]
        + sum(r["physical_invocation"]["cpu_seconds"] for r in records),
    }
    if pilot.digest(summary) != pilot.digest(expected_summary):
        msg = "Confirmation SUMMARY differs from the full authenticated case panel."
        raise ValueError(msg)
    paths = [root / name for name in ("DESIGN.json", "PROTOCOL.json", "SUMMARY.json")]
    for case in protocol["cases"]:
        child = root / f"{case['target_id']}__{case['replica']}"
        paths.extend(
            child / name
            for name in (
                "CASE.json",
                "RESULT.json",
                "execution/PHASE.json",
                "execution/ATTEMPTS.json",
                "execution/RESULT.json",
                "execution/COMPLETED.json",
            )
        )
    return protocol, records, {str(p.resolve()): file_hash(p) for p in paths}


def heatmap_arrays(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return paired target-by-noise percentage-point deltas against both required controls."""
    panel = data["panels"]["primary"]
    return [
        {
            "baseline": baseline,
            "role": role,
            "targets": data["targets"],
            "noise_ids": list(NOISY),
            "delta_pp": [
                [100 * panel["comparisons"][baseline]["by_target"][t]["profile_deltas"][n] for n in NOISY]
                for t in data["targets"]
            ],
        }
        for role, baseline in (
            ("Fixed primary baseline", parent.BASELINE),
            ("Strongest aggregate fair control", panel["strongest_control"]),
        )
    ]


def scatter_figure(data: dict[str, Any]) -> Figure:
    """Show all seven primary method means with four-objective nondominated rings."""
    aggregate = data["panels"]["primary"]["descriptive"]["aggregate"]
    points = {p["method"]: p for p in aggregate["points"]}
    result = Figure(figsize=(13, 6.5), dpi=160)
    axes = result.subplots(1, 2)
    result.subplots_adjust(left=0.075, right=0.98, top=0.69, bottom=0.22, wspace=0.18)
    passed = data["frozen_gate"]["promoted"]
    result.suptitle(
        f"Confirmation at primary common CPU · fixed gate {'passed' if passed else 'failed'}",
        x=0.075,
        y=0.97,
        ha="left",
        fontsize=17,
    )
    handles = [
        Line2D([], [], linestyle="none", marker=marker, color=color, label=f"{i + 1}  {label}")
        for i, (label, marker, color) in enumerate(zip(style.LABELS, style.MARKERS, style.COLORS, strict=True))
    ]
    result.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.07, 0.89), ncol=3, frameon=False)
    offsets = ((-24, 15), (10, -16), (22, 14), (-24, -3), (23, -14), (8, -12), (8, 8))
    for axis, resource in zip(axes, ("cpu_seconds", "two_qubit"), strict=True):
        for i, method in enumerate(parent.METHODS):
            x, y = points[method][resource], 100 * points[method]["fidelity"]
            if method in aggregate["physical_resource_frontier"]["nondominated"]:
                axis.scatter(x, y, s=210, facecolors="none", edgecolors="#222222", linewidths=1.1)
            axis.scatter(x, y, s=66, marker=style.MARKERS[i], color=style.COLORS[i], zorder=3)
            axis.annotate(
                str(i + 1),
                (x, y),
                xytext=offsets[i] if resource == "cpu_seconds" else (8, 7 if i % 2 == 0 else -12),
                textcoords="offset points",
                fontsize=9,
                arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.5}
                if resource == "cpu_seconds" and i < 5
                else None,
            )
        axis.set_xlabel("Mean deployable procedure CPU (s)" if resource == "cpu_seconds" else "Mean two-qubit gates")
        axis.set_ylabel("Mean noisy fidelity (%)")
        axis.grid(visible=True, color="#E4E4E4", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(x=0.15, y=0.16)
    result.text(
        0.075,
        0.065,
        "Five targets · two replicas each · eleven noisy profiles equally weighted. "
        "Physical gate counts precede Ballarin lowering.\n"
        "Ring: nondominated in fidelity, one-qubit gates, two-qubit gates and CPU; no projected frontier line.\n"
        "CPU includes initialization; final evaluation is separate. Full-cap results cannot rescue the primary gate.",
        fontsize=9,
        linespacing=1.5,
    )
    return result


def heatmap_figure(arrays: Sequence[dict[str, Any]]) -> Figure:
    """Use one symmetric percentage-point color scale for both complete paired heatmaps."""
    result = Figure(figsize=(14, 8), dpi=160)
    axes = result.subplots(2, 1)
    result.subplots_adjust(left=0.14, right=0.90, top=0.86, bottom=0.14, hspace=0.52)
    result.suptitle(
        "Confirmation: candidate minus baseline by target and noise", x=0.14, y=0.97, ha="left", fontsize=16
    )
    limit = max(0.01, *(abs(float(value)) for a in arrays for row in a["delta_pp"] for value in row))
    short_noise = [NOISE_LABELS[noise] for noise in NOISY]
    image = None
    for axis, array in zip(axes, arrays, strict=True):
        values = np.asarray(array["delta_pp"])
        image = axis.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_title(
            array["role"] + ": " + array["baseline"].removeprefix("structural_mean2_").replace("_", " "),
            loc="left",
            fontsize=12,
            pad=10,
        )
        axis.set_xticks(range(len(NOISY)), short_noise, rotation=25, ha="right", fontsize=9)
        axis.set_yticks(range(len(array["targets"])), [t.replace("_", " ") for t in array["targets"]], fontsize=9)
        for y, row in enumerate(values):
            for x, value in enumerate(row):
                axis.text(
                    x,
                    y,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) > 0.58 * limit else "black",
                )
    color_axis = result.add_axes((0.925, 0.25, 0.016, 0.49))
    assert image is not None
    result.colorbar(image, cax=color_axis, label="Fidelity difference (percentage points)")
    result.text(
        0.14,
        0.035,
        "Primary common-CPU endpoints; each cell averages the paired replicas 11 and 29.\n"
        "All five targets and eleven noisy profiles retained. Baselines fixed globally, never selected per cell.",
        fontsize=10,
    )
    return result


def export(root: Path, prefix: Path) -> dict[str, Any]:
    """Authenticate once and export complete data/CSV plus primary scatter and paired heatmaps.

    Raises:
        FileExistsError: If any publication path already exists.
        ValueError: If a source changes while the read-only analysis is produced.
    """
    paths = {
        name: prefix.with_name(prefix.name + name)
        for name in (".json", ".csv", "_scatter.png", "_scatter.svg", "_heatmaps.png", "_heatmaps.svg")
    }
    if any(path.exists() for path in paths.values()):
        msg = "Confirmation tradeoff outputs already exist; use a new prefix."
        raise FileExistsError(msg)
    cpu, wall = time.process_time(), time.perf_counter()
    protocol, records, sources = authenticate(root)
    data = analyze(protocol, records)
    arrays = heatmap_arrays(data)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": VERSION}):
        for label, figure in (("scatter", scatter_figure(data)), ("heatmaps", heatmap_figure(arrays))):
            for extension in ("png", "svg"):
                figure.savefig(
                    paths[f"_{label}.{extension}"],
                    dpi=200,
                    metadata={"Creator": VERSION, "Date": None} if extension == "svg" else {"Software": VERSION},
                )
    rows = []
    for point_row in data["points"]:
        for noise in parent.base.channels.NOISE_IDS:
            counts = point_row["ballarin_counts"] if noise == "ballarin_coupled" else point_row
            rows.append({
                **{
                    k: point_row[k]
                    for k in (
                        "target_id",
                        "replica",
                        "method",
                        "panel",
                        "cpu_seconds",
                        "cutoff_cpu_seconds",
                        "parameters",
                        "evaluation_checksum",
                        "circuit_hash",
                    )
                },
                "noise_id": noise,
                "fidelity": point_row["fidelities"][noise],
                "mean_noisy_fidelity": point_row["fidelity"],
                "noiseless_fidelity": point_row["noiseless_fidelity"],
                **{k: counts[k] for k in ("one_qubit", "two_qubit", "depth")},
            })
    with paths[".csv"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if any(file_hash(Path(path)) != checksum for path, checksum in sources.items()):
        msg = "Confirmation sources changed during publication."
        raise ValueError(msg)
    data.update(
        parent_checksum=pilot.digest(protocol),
        configuration_sha256=protocol["configuration_sha256"],
        source_files=sources,
        scientific_source_hashes=protocol["source_hashes"],
        benchmark_hashes=protocol["benchmark_hashes"],
        source_hashes={
            str(Path(module.__file__).resolve()): file_hash(Path(module.__file__))
            for module in (parent, tradeoffs, style)
        },
        plotter_source_sha256=file_hash(Path(__file__)),
        heatmap_arrays=arrays,
        analysis_cpu_seconds=time.process_time() - cpu,
        analysis_wall_seconds=time.perf_counter() - wall,
        files={str(path.resolve()): file_hash(path) for name, path in paths.items() if name != ".json"},
    )
    pilot.write_record(paths[".json"], data)
    return data


def main(argv: Sequence[str] | None = None) -> int:
    """Export an explicit immutable complete confirmation root without numerical work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.source_root, args.output_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
