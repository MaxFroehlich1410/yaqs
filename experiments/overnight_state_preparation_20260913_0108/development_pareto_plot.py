# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Render authenticated development endpoints without selecting scientific results.

The source is a sealed, already authenticated development analysis. This module
checks its source bytes and recomputes its descriptive summaries from stored
endpoint arrays. It never loads target vectors, circuits or numerical trainers.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import comparison_tradeoffs as tradeoffs

if TYPE_CHECKING:
    from collections.abc import Sequence

VERSION = "overnight.development_pareto_plot.v1"
INPUT_VERSION = "overnight.development_pareto_inputs.v1"
EXPECTED_INPUT_CHECKSUM = "449a5d08016a8147b3ce704c05b13e42dc25892edb2a20fbe7de281276e175a2"
TARGETS = ("gaussian_mu0p5_sigma0p1", "tfim_critical", "haar_random_1", "random_mps_bond2")
METHODS = (
    "structural_mean2_lbfgs",
    "structural_mean2_adam",
    "structural_mean2_weighted_adam",
    "structural_mean2_adaptive_online",
    "fixed_mean2_lbfgs",
    "fixed_noiseless_adam",
    "unchanged",
)
LABELS = (
    "Structural L-BFGS",
    "Structural exact Adam",
    "Structural weighted Adam",
    "Structural adaptive online",
    "Fixed mean-two L-BFGS",
    "Fixed noiseless Adam",
    "Unchanged initializer",
)
RECIPES = ("original", "state_preparation_compiled")
ROW_TITLES = {
    "original": "Original recipe · selected for confirmation",
    "state_preparation_compiled": "With state preparation compiler · not promoted",
}
STRATA = ("primary", "all_trained_sensitivity", "full_cap")
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#555555", "#A76525", "#222222")
MARKERS = ("D", "s", "^", "v", "P", "X", "o")


def file_hash(path: Path) -> str:
    """Return a byte checksum without parsing the referenced scientific artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_points(points: list[dict[str, Any]]) -> None:
    """Reject nonphysical or unknown values before plotting or recomputing means.

    Raises:
        ValueError: If an endpoint has nonfinite scores or missing physical costs.
    """
    for point in points:
        for key in ("fidelity", "noiseless_fidelity", *tradeoffs.RESOURCES):
            value = point[key]
            score = key in {"fidelity", "noiseless_fidelity"}
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value < (-1e-10 if score else 0)
                or (score and value > 1 + 1e-10)
                or (not score and key != "cpu_seconds" and value != int(value))
            ):
                msg = "Plot endpoints require finite physical fidelities, integer counts and known nonnegative CPU."
                raise ValueError(msg)


def authenticate(path: Path, expected_checksum: str | None = None) -> dict[str, Any]:
    """Verify the sealed input, every recorded source and all complete aggregates.

    This verifies the existing evidence snapshot. It does not re-execute a
    numerical source or claim that changing and re-signing all evidence would
    preserve its original provenance.

    Raises:
        ValueError: If source bytes, panel coverage or recorded analysis differ.
    """
    record = cast("dict[str, Any]", pilot.read_record(path))
    expected = {f"{recipe}/{stratum}" for recipe in RECIPES for stratum in STRATA}
    if (
        record["version"] != INPUT_VERSION
        or (expected_checksum is not None and pilot.digest(record) != expected_checksum)
        or record["analysis_source_sha256"] != file_hash(Path(tradeoffs.__file__))
        or set(record["panels"]) != expected
    ):
        msg = "Development plot input has a changed schema, analysis source or complete six-panel selection."
        raise ValueError(msg)
    merged: dict[str, str] = {}
    for field in ("source_files", "original_evidence_files"):
        mapping = record[field]
        if not isinstance(mapping, dict) or not mapping:
            msg = "Development plot input is missing its nonempty authenticated source inventory."
            raise ValueError(msg)
        for name, checksum in mapping.items():
            source = Path(name)
            if (
                not source.is_absolute()
                or not source.is_file()
                or file_hash(source) != checksum
                or (name in merged and merged[name] != checksum)
            ):
                msg = f"Development plot source bytes changed: {name}."
                raise ValueError(msg)
            merged[name] = checksum
    for key in sorted(expected):
        panel = record["panels"][key]
        validate_points(panel["points"])
        recomputed = tradeoffs.summarize(panel["points"], TARGETS, (11,), METHODS)
        if recomputed["status"] != "complete" or pilot.digest(recomputed) != pilot.digest(panel["analysis"]):
            msg = f"Development plot aggregate differs from its complete raw endpoint panel: {key}."
            raise ValueError(msg)
    return record


def plotted_arrays(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return exact primary-panel coordinates and the four-objective ring flags."""
    arrays = []
    for recipe in RECIPES:
        aggregate = record["panels"][recipe + "/primary"]["analysis"]["aggregate"]
        lookup = {point["method"]: point for point in aggregate["points"]}
        nondominated = set(aggregate["physical_resource_frontier"]["nondominated"])
        arrays.extend(
            {
                "recipe": recipe,
                "panel": "primary",
                "x_resource": axis,
                "methods": list(METHODS),
                "x": [lookup[name][axis] for name in METHODS],
                "fidelity": [lookup[name]["fidelity"] for name in METHODS],
                "y_percent": [100 * lookup[name]["fidelity"] for name in METHODS],
                "four_objective_nondominated": [name in nondominated for name in METHODS],
                "projected_frontier_line": False,
            }
            for axis in ("cpu_seconds", "two_qubit")
        )
    return arrays


def figure(arrays: list[dict[str, Any]]) -> Figure:
    """Build two recipe rows and two resource columns with all seven method points."""
    with mpl.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": VERSION}):
        result = Figure(figsize=(13, 9.3), dpi=160)
        axes = result.subplots(2, 2, sharey=True, squeeze=False)
        result.subplots_adjust(left=0.075, right=0.98, top=0.74, bottom=0.19, wspace=0.17, hspace=0.42)
        result.suptitle("Development: fidelity and physical resources", x=0.075, y=0.98, ha="left", fontsize=18)
        result.text(
            0.075,
            0.93,
            "Primary common-CPU checkpoints · four targets · seed 11 · eleven noisy profiles equally weighted",
            fontsize=10,
            color="#444444",
        )
        labels = [
            Line2D([], [], marker=marker, linestyle="none", color=color, markersize=7, label=f"{i + 1}  {label}")
            for i, (label, color, marker) in enumerate(zip(LABELS, COLORS, MARKERS, strict=True))
        ]
        labels.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color="#222222",
                markerfacecolor="none",
                markersize=10,
                label="Ring: nondominated in all four objectives",
            )
        )
        result.legend(handles=labels, loc="upper left", bbox_to_anchor=(0.07, 0.90), ncol=3, frameon=False)
        offsets = ((8, 7), (8, -13), (-14, -12), (-14, 8), (8, 0), (8, -11), (8, 7))
        cpu_offsets = ((-24, 14), (16, -18), (23, 15), (-24, -6), (23, -9), (8, -11), (8, 7))
        for index, values in enumerate(arrays):
            axis = axes[index // 2, index % 2]
            for i, (x, y, outlined) in enumerate(
                zip(values["x"], values["y_percent"], values["four_objective_nondominated"], strict=True)
            ):
                if outlined:
                    axis.scatter(x, y, s=210, marker="o", facecolors="none", edgecolors="#222222", linewidths=1.1)
                axis.scatter(x, y, s=66, marker=MARKERS[i], color=COLORS[i], zorder=3)
                cpu_cluster = values["x_resource"] == "cpu_seconds" and i < 5
                axis.annotate(
                    str(i + 1),
                    (x, y),
                    xytext=cpu_offsets[i] if cpu_cluster else offsets[i],
                    textcoords="offset points",
                    fontsize=9,
                    arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.5} if cpu_cluster else None,
                )
            axis.set_title(ROW_TITLES[values["recipe"]], loc="left", fontsize=12, pad=12)
            axis.set_xlabel(
                "Mean deployable procedure CPU (s)" if values["x_resource"] == "cpu_seconds" else "Mean two-qubit gates"
            )
            if index % 2 == 0:
                axis.set_ylabel("Mean noisy fidelity (%)")
            axis.grid(visible=True, color="#E4E4E4", linewidth=0.6)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.margins(x=0.11, y=0.16)
        result.text(
            0.075,
            0.04,
            "Ring objectives: maximize fidelity; minimize one-qubit gates, two-qubit gates and CPU. "
            "No projected frontier line.\n"
            "CPU includes the declared initializer charge; diagnostic evaluation is separate. "
            "Gate counts precede Ballarin lowering.\n"
            "Rows are separate development comparisons, not a causal compiler effect. "
            "Fidelity axes show the observed range.",
            fontsize=9,
            color="#444444",
            linespacing=1.5,
            va="bottom",
        )
    return result


def export(source: Path, prefix: Path, *, expected_checksum: str = EXPECTED_INPUT_CHECKSUM) -> dict[str, Any]:
    """Authenticate, render PNG/SVG, and seal exact plotted arrays and file hashes.

    Raises:
        FileExistsError: If an output would overwrite existing scientific evidence.
        ValueError: If an input changes during rendering.
    """
    paths = {suffix: prefix.with_suffix("." + suffix) for suffix in ("png", "svg", "json")}
    if any(path.exists() for path in paths.values()):
        msg = "Plot outputs already exist; select a new output prefix."
        raise FileExistsError(msg)
    source = source.resolve()
    source_sha = file_hash(source)
    record = authenticate(source, expected_checksum)
    arrays = plotted_arrays(record)
    result = figure(arrays)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"svg.hashsalt": VERSION}):
        result.savefig(paths["png"], dpi=200, metadata={"Software": VERSION})
        result.savefig(paths["svg"], metadata={"Date": None, "Creator": VERSION})
    if file_hash(source) != source_sha:
        msg = "Immutable development plot input changed during rendering."
        raise ValueError(msg)
    payload = {
        "version": VERSION,
        "source_path": str(source),
        "source_sha256": source_sha,
        "source_checksum": pilot.digest(record),
        "plotter_source_sha256": file_hash(Path(__file__)),
        "analysis_source_sha256": file_hash(Path(tradeoffs.__file__)),
        "matplotlib_version": mpl.__version__,
        "comparison_scope": record["comparison_scope"],
        "cpu_scope": record["cpu_scope"],
        "physical_gate_scope": record["physical_gate_scope"],
        "development_selection_labels": ROW_TITLES,
        "source_files": record["source_files"],
        "original_evidence_files": record["original_evidence_files"],
        "plotted_arrays": arrays,
        "files": {str(path.resolve()): file_hash(path) for suffix, path in paths.items() if suffix != "json"},
    }
    pilot.write_record(paths["json"], payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Render an explicit immutable development input to a new output prefix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.input, args.output_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
