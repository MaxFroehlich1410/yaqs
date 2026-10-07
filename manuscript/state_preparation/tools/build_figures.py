# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Build manuscript figures and tables from portable, immutable endpoint data.

Run ``python tools/build_figures.py`` from the manuscript directory. The script
requires only NumPy and Matplotlib in addition to the Python standard library.
It neither imports the experiment implementation nor accesses training archives.
"""

from __future__ import annotations

import argparse
import json
import math
import operator
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TOPDOWN = "current_b_lbfgs_frozen"
STYLES = {
    TOPDOWN: {"color": "#0072B2", "marker": "*", "size": 64, "filled": True},
    "nacl": {"color": "#777777", "marker": "o", "size": 18, "filled": True},
    "robuststate": {"color": "#D55E00", "marker": "D", "size": 23, "filled": True},
    "fontana": {"color": "#009E73", "marker": "v", "size": 26, "filled": True},
    "fixed_vqsp": {"color": "#CC79A7", "marker": "s", "size": 23, "filled": False},
    "layerwise": {"color": "#9B7100", "marker": "^", "size": 25, "filled": False},
}
SHORT_TARGETS = {
    "gaussian_mu0p5_sigma0p1": "Gaussian",
    "tfim_ferro": "TFIM ferro.",
    "tfim_critical": "TFIM critical",
    "tfim_para": "TFIM para.",
    "haar_random_1": "Haar 1",
    "haar_random_2": "Haar 2",
    "haar_random_3": "Haar 3",
    "random_mps_bond2": "MPS bond 2",
    "random_mps_bond3": "MPS bond 3",
}


def load_data(path: Path) -> dict[str, Any]:
    """Load portable endpoint data and reject invalid numerical records.

    Args:
        path: Packaged JSON input file.

    Returns:
        Parsed manuscript data with unique method and target identities.

    Raises:
        ValueError: If the schema or an endpoint is inconsistent.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "yaqs.state_preparation_manuscript.v1":
        msg = "Unsupported manuscript data schema."
        raise ValueError(msg)
    for key in ("logical_endpoints", "native_endpoints"):
        identities = set()
        for row in data[key]:
            identity = (row["target"], row["method"])
            fidelity, cpu = row["exact_fidelity"], row["build_train_cpu_seconds"]
            if (
                identity in identities
                or not math.isfinite(fidelity)
                or not 0 <= fidelity <= 1
                or not math.isfinite(cpu)
                or cpu <= 0
            ):
                msg = f"Invalid or duplicate endpoint in {key}: {identity}."
                raise ValueError(msg)
            identities.add(identity)
    return data


def paired_gaps(data: dict[str, Any], comparator: str) -> list[tuple[str, float]]:
    """Calculate percentage-point gaps on the common supported target subset.

    Args:
        data: Portable manuscript data.
        comparator: Method identifier compared with Top-down.

    Returns:
        Target and Top-down-minus-comparator gap in the declared target order.
    """
    rows = {(row["target"], row["method"]): row for row in data["logical_endpoints"]}
    return [
        (target, 100 * (rows[target, TOPDOWN]["exact_fidelity"] - rows[target, comparator]["exact_fidelity"]))
        for target in data["target_order"]
        if (target, TOPDOWN) in rows and (target, comparator) in rows
    ]


def pareto_frontier(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find nondominated fidelity and complete construction/training CPU endpoints.

    Args:
        rows: Selected endpoints for a single target, including controls.

    Returns:
        Nondominated endpoints ordered by increasing complete training CPU.
        Comparisons use a relative tolerance of 1e-12, as in the archived analysis.
    """
    frontier = []
    for row in rows:
        dominated = False
        for other in rows:
            differences = [
                (other["exact_fidelity"] - row["exact_fidelity"], "exact_fidelity"),
                (row["build_train_cpu_seconds"] - other["build_train_cpu_seconds"], "build_train_cpu_seconds"),
            ]
            comparisons = [(difference, 1e-12 * max(abs(row[key]), abs(other[key]))) for difference, key in differences]
            if all(gap >= -tol for gap, tol in comparisons) and any(gap > tol for gap, tol in comparisons):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=operator.itemgetter("build_train_cpu_seconds"))


def configure_style() -> None:
    """Set consistent typography, vector fonts and a noninteractive backend."""
    plt.switch_backend("Agg")
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def save_figure(figure: Figure, output: Path, stem: str) -> None:
    """Save a vector PDF and a high-resolution PNG with identical geometry.

    Args:
        figure: Fully laid-out Matplotlib figure.
        output: Directory for figure files.
        stem: Output basename without an extension.
    """
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(output / f"{stem}.png", dpi=360)
    plt.close(figure)


def build_fidelity(data: dict[str, Any], output: Path) -> None:
    """Plot all logical endpoints and per-target differences from RobustState.

    Args:
        data: Portable manuscript data.
        output: Figure output directory.
    """
    methods, targets = data["method_order"], data["target_order"]
    rows = {(row["target"], row["method"]): row for row in data["logical_endpoints"]}
    values = np.array([
        [100 * rows[target, method]["exact_fidelity"] if (target, method) in rows else np.nan for method in methods]
        for target in targets
    ])
    figure = plt.figure(figsize=(7.05, 3.65))
    left = figure.add_axes((0.155, 0.23, 0.51, 0.665))
    right = figure.add_axes((0.727, 0.23, 0.25, 0.665), sharey=left)
    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad("#eeeeee")
    heatmap = left.imshow(values, cmap=cmap, vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    left.set_title("(a) Selected logical fidelity (%)", loc="left", pad=9)
    for i, row in enumerate(values):
        best = float(np.nanmax(row))
        for j, value in enumerate(row):
            left.text(
                j,
                i,
                "—" if np.isnan(value) else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.1,
                color="white" if not np.isnan(value) and value < 60 else "#171717",
                fontweight="bold" if value == best else "normal",
            )
    left.set_yticks(range(len(targets)), [SHORT_TARGETS[target] for target in targets])
    short_methods = ["Top-down", "NACL", "RobustState", "Fontana", "Fixed VQSP", "Layerwise"]
    left.set_xticks(range(len(methods)), short_methods, rotation=38, ha="right", rotation_mode="anchor")
    left.tick_params(axis="both", length=0, pad=4)
    left.set_xticks(np.arange(-0.5, len(methods), 1), minor=True)
    left.set_yticks(np.arange(-0.5, len(targets), 1), minor=True)
    left.grid(which="minor", color="white", linewidth=0.35)
    left.tick_params(which="minor", length=0)
    for spine in left.spines.values():
        spine.set_visible(False)
    right.set_title(r"(b) Top-down $-$ RobustState", loc="left", pad=9, fontsize=8.4)
    gaps = dict(paired_gaps(data, "robuststate"))
    for index, target in enumerate(targets):
        gap = gaps[target]
        color = STYLES[TOPDOWN if gap > 0 else "robuststate"]["color"]
        right.plot([0, gap], [index, index], color=color, linewidth=1.1)
        right.scatter([gap], [index], color=color, s=18, edgecolors="white", linewidths=0.35, zorder=3)
    right.axvline(0, color="#555555", linewidth=0.7)
    right.set_xlim(-1.7, 4.5)
    right.set_xticks([-1, 0, 1, 2, 3, 4])
    right.set_xlabel("Fidelity difference (pp)", labelpad=5)
    right.tick_params(axis="y", which="both", left=False, labelleft=False)
    right.grid(axis="x", color="#e4e4e4", linewidth=0.45)
    right.set_axisbelow(True)
    for side in ("top", "left", "right"):
        right.spines[side].set_visible(False)
    colorbar_axis = figure.add_axes((0.22, 0.064, 0.36, 0.026))
    colorbar = figure.colorbar(heatmap, cax=colorbar_axis, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    colorbar.ax.tick_params(labelsize=7, length=2, pad=2)
    save_figure(figure, output, "fidelity")


def build_pareto(data: dict[str, Any], output: Path) -> None:
    """Plot fidelity versus complete construction/training CPU for every target.

    Args:
        data: Portable manuscript data.
        output: Figure output directory.
    """
    figure, axes = plt.subplots(3, 3, figsize=(7.05, 5.05), sharex=True, sharey=True)
    figure.subplots_adjust(left=0.077, right=0.988, bottom=0.205, top=0.952, wspace=0.14, hspace=0.37)
    handles = {}
    for index, (target, axis) in enumerate(zip(data["target_order"], axes.flat, strict=True)):
        rows = [row for row in data["logical_endpoints"] if row["target"] == target]
        frontier = pareto_frontier(rows)
        axis.plot(
            [row["build_train_cpu_seconds"] for row in frontier],
            [100 * row["exact_fidelity"] for row in frontier],
            color="#a7a7a7",
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            zorder=1,
        )
        for row in rows:
            method = row["method"]
            style = STYLES[method]
            handle = axis.scatter(
                row["build_train_cpu_seconds"],
                100 * row["exact_fidelity"],
                s=style["size"],
                marker=style["marker"],
                edgecolors=style["color"],
                facecolors=style["color"] if style["filled"] else "white",
                linewidths=0.8,
                zorder=4 if method == TOPDOWN else 3,
            )
            handles[method] = handle
        axis.set_title(f"({chr(97 + index)}) {SHORT_TARGETS[target]}", loc="left", pad=4, fontsize=8.4)
        axis.set_xscale("log")
        axis.set_xlim(2.5, 110)
        axis.set_ylim(0, 102)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.xaxis.set_major_locator(FixedLocator([3, 10, 30, 100]))
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.grid(color="#e6e6e6", linewidth=0.4)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=7)
        axis.tick_params(which="minor", bottom=False)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    figure.supylabel("Exact logical noisy fidelity (%)", x=0.009, y=0.58, fontsize=9)
    figure.supxlabel("Complete construction and training CPU (s; logarithmic scale)", y=0.125, fontsize=9)
    labels = [
        data["method_labels"][method] + (" (control)" if method in {"fixed_vqsp", "layerwise"} else "")
        for method in data["method_order"]
    ]
    figure.legend(
        [handles[method] for method in data["method_order"]],
        labels,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.008),
        ncol=3,
        columnspacing=1.5,
        handletextpad=0.5,
        frameon=False,
        fontsize=8.2,
    )
    save_figure(figure, output, "pareto")


def _box(axis: Axes, x: float, y: float, width: float, height: float, text: str, *, face: str = "white") -> None:
    """Draw an aligned workflow box with centered multiline text.

    Args:
        axis: Diagram axes.
        x: Left box coordinate.
        y: Bottom box coordinate.
        width: Box width.
        height: Box height.
        text: Multiline box label.
        face: Box background color.
    """
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.045",
            facecolor=face,
            edgecolor="#66747d",
            linewidth=0.7,
            zorder=3,
        )
    )
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=8.2, linespacing=1.3, zorder=4)


def _arrow(axis: Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    """Draw a directed workflow edge.

    Args:
        axis: Diagram axes.
        start: Edge start in diagram coordinates.
        end: Edge end in diagram coordinates.
    """
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.85,
            color="#405566",
            zorder=2,
        )
    )


def build_workflow(output: Path) -> None:
    """Draw parent growth, separate candidate refinement and locked assessment.

    Args:
        output: Figure output directory.
    """
    figure = plt.figure(figsize=(7.05, 3.78))
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, 7.05)
    axis.set_ylim(0, 3.78)
    axis.axis("off")
    for y, height, face in [(2.47, 1.25, "#eef4f8"), (0.96, 1.25, "#f4f5f4")]:
        axis.add_patch(
            FancyBboxPatch(
                (0.10, y),
                6.84,
                height,
                boxstyle="round,pad=0.01,rounding_size=0.055",
                facecolor=face,
                edgecolor="none",
            )
        )
    axis.text(0.24, 3.51, "(a) Unpruned noiseless parent", fontweight="bold", fontsize=9)
    _box(axis, 0.24, 2.66, 1.91, 0.56, "Add a brickwall cycle\nretain trained parameters")
    _box(axis, 2.52, 2.66, 2.01, 0.56, "Noiseless optimization\n100 online + 100 L-BFGS")
    _box(axis, 4.90, 2.66, 1.79, 0.56, "Restore best iterate\nsave unpruned parent")
    _arrow(axis, (2.17, 2.94), (2.49, 2.94))
    _arrow(axis, (4.55, 2.94), (4.87, 2.94))
    axis.plot([5.795, 5.795, 1.195], [3.24, 3.37, 3.37], linewidth=0.85, color="#405566")
    _arrow(axis, (1.195, 3.37), (1.195, 3.24))
    axis.text(4.12, 3.44, "next depth, if continuing", ha="center", fontsize=7.6)
    axis.plot([5.795, 5.795, 1.77], [2.64, 2.34, 2.34], linewidth=0.85, color="#405566")
    _arrow(axis, (1.77, 2.34), (1.77, 1.88))
    axis.text(3.38, 2.40, "Separate copy at every visited depth", ha="center", fontsize=8)
    axis.text(0.24, 2.04, "(b) Pruned candidate", fontweight="bold", fontsize=9)
    _box(axis, 0.24, 1.24, 1.69, 0.62, "Pruning B\nscore gates once\ndelete together", face="#ffffff")
    _box(axis, 2.25, 1.24, 2.07, 0.62, "Noisy refinement\nfrozen M = 100 histories\n100 online + 100 cross")
    _box(axis, 4.65, 1.24, 2.04, 0.62, "Depth validation\n32,768 fresh trajectories\npatience 3; depth cap 16")
    _arrow(axis, (1.95, 1.55), (2.22, 1.55))
    _arrow(axis, (4.34, 1.55), (4.62, 1.55))
    axis.text(1.085, 1.06, r"Delete if $\Delta_i < 0.01$", ha="center", fontsize=7.9)
    axis.text(3.285, 1.06, "Keep final noisy iterate", ha="center", fontsize=7.9)
    _arrow(axis, (5.67, 1.22), (5.67, 0.71))
    _box(axis, 3.72, 0.13, 2.97, 0.56, "Lock selected depth\nexact density-matrix fidelity", face="#eef4f8")
    axis.text(
        0.24,
        0.45,
        "Latest comparison retains the archive depth winner.\nIts 8,192-path validation is diagnostic.",
        ha="left",
        va="center",
        fontsize=8,
        linespacing=1.5,
    )
    save_figure(figure, output, "workflow")


def native_table_rows(data: dict[str, Any]) -> list[list[str]]:
    """Format native rows without mixing them with logical-model endpoint scores.

    Args:
        data: Portable manuscript data.

    Returns:
        Rows containing target, method, fidelity, full CPU, CX count and CX depth.
    """
    native = {(row["target"], row["method"]): row for row in data["native_endpoints"]}
    rows = []
    for target in data["target_order"]:
        for method in data["method_order"]:
            if (target, method) not in native:
                continue
            row = native[target, method]
            rows.append([
                data["target_labels"][target],
                data["method_labels"][method],
                f"{100 * row['exact_fidelity']:.3f}",
                f"{row['build_train_cpu_seconds']:.2f}",
                str(row["cx"]),
                str(row["cx_depth"]),
            ])
    return rows


def fidelity_table_rows(data: dict[str, Any]) -> list[list[str]]:
    """Format all selected logical fidelities with explicit unsupported entries.

    Args:
        data: Portable manuscript data.

    Returns:
        Target rows followed by each method's fidelity in percent.
    """
    selected = {(row["target"], row["method"]): row for row in data["logical_endpoints"]}
    rows = []
    for target in data["target_order"]:
        row = [data["target_labels"][target]]
        for method in data["method_order"]:
            endpoint = selected.get((target, method))
            row.append(r"\textemdash" if endpoint is None else f"{100 * endpoint['exact_fidelity']:.3f}")
        rows.append(row)
    return rows


def _latex_rows(rows: Sequence[Sequence[str]], *, group_by_target: bool = False) -> str:
    """Join table cells and optionally separate target groups.

    Args:
        rows: Preformatted LaTeX table cells.
        group_by_target: Insert rules and suppress repeated target labels.

    Returns:
        LaTeX table body preserving the numeric precision in the supplied cells.
    """
    result, previous = [], None
    for row in rows:
        cells = list(row)
        if group_by_target:
            if cells[0] == previous:
                cells[0] = ""
            elif previous is not None:
                result.append(r"\midrule")
            previous = row[0]
        result.append(" & ".join(cells) + r" \\")
    return "\n".join(result)


def build_tables(data: dict[str, Any], output: Path) -> None:
    """Generate the two main tables and supplementary endpoint tables.

    Args:
        data: Portable manuscript data.
        output: Directory for standalone LaTeX table environments.
    """
    output.mkdir(parents=True, exist_ok=True)
    protocol = r"""% Created by tools/build_figures.py from the frozen experimental protocol.
\begin{table*}[t]
\caption{\textbf{Frozen logical-comparison protocol.}
Candidate policies, training budgets, and feedback used for the six-qubit comparison at optimizer seed 300.
Each budget is charged across the complete search.
These are finite implementation budgets, not equal-compute or converged-publication comparisons.
Top-down uses its archived depth selection; all other methods select by 8,192 independent trajectories.
Final scores integrate the logical channel exactly. The two controls are internal comparisons.}
\label{tab:protocol}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.14}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.13\textwidth}
>{\raggedright\arraybackslash}p{0.21\textwidth}
>{\raggedright\arraybackslash}p{0.48\textwidth}
>{\raggedright\arraybackslash}p{0.10\textwidth}@{}}
\toprule
Method & Candidate policy & Training budget and feedback & Targets \\
\midrule
Top-down & Every visited depth; patience 3, cap 16; pruning B &
Noiseless: 100 online Krotov sweeps ($\eta=0.7$), then 100 L-BFGS updates; restore best iterate.
Noisy: 100 online and 100 cross-trajectory updates ($\eta=0.2$), $M=100$ frozen histories; keep final iterate.
& All 9 \\
NACL~\cite{Cincio2021} & Structural checkpoints 0, 10, 50; one restart; 300-gate cap &
Up to 100 inner OrthoMADS objective evaluations per structural proposal; noisy-fidelity simulator objective.
& All 9 \\
RobustState~\cite{RobustState2024} & Depths 2 and 4 &
500 ideal and 50 noisy Adam updates ($\eta=0.005$); exact-density feedback adaptation. & All 9 \\
Fontana~\cite{Fontana2021} & Depths 2 and 4 &
200 ideal and 200 noisy BFGS iterations; RY/CX ansatz with simulator feedback. & 6 real \\
Fixed VQSP (control) & Depths 2 and 4; independent initializations &
200 ideal and 200 noisy L-BFGS-B iterations; $M=100$ weighted frozen histories for noisy feedback. & All 9 \\
Layerwise (control) & Depths 1--4; unpruned growth &
Noiseless: 100 online Krotov and 100 L-BFGS updates; restore best parent.
Noisy: 100 online and 100 cross-trajectory updates per depth; $M=100$ weighted frozen histories. & All 9 \\
\bottomrule
\end{tabular}
\end{table*}
"""
    (output / "protocol.tex").write_text(protocol)
    native = (
        r"""% Created by tools/build_figures.py from data/results.json.
\begin{table*}[t]
\caption{\textbf{Matched native-noise comparison.}
Exact density-matrix fidelity of the selected circuits after training and assessment with the synthetic Pauli channel
after RZ/SX/X/CX operations. CPU includes all construction and training candidates, including reused work.
Top-down uses 32,768 validation trajectories per depth; competitors retain their locked 8,192-trajectory selections.
Fontana excludes complex Haar states. Fontana dominates Top-down on Gaussian, RobustState dominates it on Haar 1,
and TFIM critical retains a fidelity/CPU trade-off.}
\label{tab:native}
\centering
\small
\setlength{\tabcolsep}{8pt}
\begin{tabular}{llrrrr}
\toprule
Target & Method & Fidelity (\%) & CPU (s) & CX & CX depth \\
\midrule
"""
        + _latex_rows(native_table_rows(data), group_by_target=True)
        + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    )
    (output / "native_results.tex").write_text(native)
    header = "Target & " + " & ".join(data["method_labels"][method] for method in data["method_order"]) + r" \\"
    fidelity = (
        r"""% Created by tools/build_figures.py from data/results.json.
\begin{table*}[t]
\caption{\textbf{All selected logical fidelities.}
Exact fidelity in percent for every supported method--target pair at six qubits and seed 300.
Values correspond to locked selections, not post-hoc exact maxima.
Fontana's three unsupported complex targets are marked by dashes. Fixed VQSP and layerwise are internal controls.}
\label{tab:full-fidelity}
\centering
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrrrr}
\toprule
"""
        + header
        + "\n"
        + r"\midrule"
        + "\n"
        + _latex_rows(fidelity_table_rows(data))
        + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    )
    (output / "full_fidelity.tex").write_text(fidelity)
    resources = []
    selected = {(row["target"], row["method"]): row for row in data["logical_endpoints"]}
    for target in data["target_order"]:
        cells = [data["target_labels"][target]]
        for method in (TOPDOWN, "robuststate"):
            row = selected[target, method]
            cells.extend([
                str(row["logical_two_qubit"]),
                str(row["logical_two_qubit_depth"]),
                str(row["cx"]),
                str(row["cx_depth"]),
            ])
        resources.append(cells)
    resource_table = (
        r"""% Created by tools/build_figures.py from data/results.json.
\begin{table*}[t]
\caption{\textbf{Selected circuit resources.}
Top-down and RobustState circuits from the logical comparison.
Logical two-qubit counts (2Q) and depths ($d_{\mathrm{2Q}}$) describe the gates that define the logical noise windows.
Compiled CX counts and depths ($d_{\mathrm{CX}}$) are resource diagnostics;
no native-noise scores are inferred from them.
Resources and complete training CPU for all 51 endpoints are retained in the accompanying data.}
\label{tab:resources}
\centering
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lrrrrrrrr}
\toprule
& \multicolumn{4}{c}{Top-down} & \multicolumn{4}{c}{RobustState} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Target & 2Q & $d_{\mathrm{2Q}}$ & CX & $d_{\mathrm{CX}}$
& 2Q & $d_{\mathrm{2Q}}$ & CX & $d_{\mathrm{CX}}$ \\
\midrule
"""
        + _latex_rows(resources)
        + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    )
    (output / "resources.tex").write_text(resource_table)


def main(argv: Sequence[str] | None = None) -> None:
    """Build figures and tables from the packaged data file.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PACKAGE_ROOT / "data" / "results.json")
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT)
    args = parser.parse_args(argv)
    data = load_data(args.data)
    configure_style()
    build_workflow(args.output_root / "figures")
    build_fidelity(data, args.output_root / "figures")
    build_pareto(data, args.output_root / "figures")
    build_tables(data, args.output_root / "tables")
    sys.stdout.write("Built 3 vector figures, 3 PNG previews and 4 LaTeX tables from packaged data.\n")


if __name__ == "__main__":
    main()
