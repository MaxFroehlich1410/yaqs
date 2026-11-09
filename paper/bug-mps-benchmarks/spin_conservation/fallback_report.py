#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
# SPDX-License-Identifier: MIT
# Licensed under the MIT License

"""Summarize a fallback experiment as tables, a report, and scientific figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def collect(output: Path) -> tuple[list[dict], list[dict]]:
    """Load successful arm records and group recovery counts by model and cap."""
    rows, buckets = [], defaultdict(list)
    for path in sorted((output / "cases").glob("*/result.json")):
        result = json.loads(path.read_text())
        for arm in result["arms"]:
            row = {"key": result["key"], **result["case"], "variant": arm["variant"], **arm["counts"]}
            row.update({
                key: arm[key]
                for key in [
                    "scalar_rejections",
                    "fallback_frequency",
                    "fallback_frequency_when_needed",
                    "wall_seconds",
                    "evolution_seconds",
                    "correction_seconds",
                    "fallback_seconds",
                    "max_energy_drift",
                    "max_pre_correction_drift",
                    "max_post_correction_drift",
                    "final_infidelity",
                    "peak_bond",
                ]
            })
            rows.append(row)
            if arm["variant"] == "fallback":
                buckets[(result["case"]["model"], result["case"]["cap"])].append((arm, result["case"]))
    groups = []
    for (model, cap), arms in sorted(buckets.items()):
        counts = Counter()
        reasons = Counter()
        for arm, _ in arms:
            counts.update(arm["counts"])
            reasons.update(event["reason"] for event in arm["events"])
        needed = counts["scalar_calls"] - counts["already_satisfied"]
        groups.append({
            "model": model,
            "cap": cap,
            "runs": len(arms),
            "eventful_runs": sum(arm["counts"]["fallback_attempts"] > 0 for arm, _ in arms),
            **dict(counts),
            "needed_calls": needed,
            "rate": counts["fallback_attempts"] / counts["scalar_calls"] if counts["scalar_calls"] else None,
            "conditional_rate": counts["fallback_attempts"] / needed if needed else None,
            "reasons": dict(reasons),
        })
    return rows, groups


def write_csv(path: Path, rows: list[dict]) -> None:
    """Export machine-readable flat tables."""
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def make_plots(output: Path, rows: list[dict], groups: list[dict]) -> None:
    """Save annotated frequency heatmaps and matched accuracy comparisons."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(group["model"] for group in groups))
    caps = sorted({group["cap"] for group in groups})
    lookup = {(group["model"], group["cap"]): group for group in groups}
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.55 * len(models) + 1.5)), constrained_layout=True)
    for ax, field, label in zip(
        axes, ["rate", "conditional_rate"], ["All correction calls", "Calls requiring correction"], strict=True
    ):
        data = np.array([[100 * (lookup.get((model, cap), {}).get(field) or 0) for cap in caps] for model in models])
        mesh = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=max(1, float(data.max())))
        for i, model in enumerate(models):
            for j, cap in enumerate(caps):
                group = lookup.get((model, cap))
                if group is None:
                    ax.text(j, i, "not run", ha="center", va="center", fontsize=8)
                    continue
                denominator = group["scalar_calls"] if field == "rate" else group["needed_calls"]
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.2f}%\n{group['fallback_attempts']}/{denominator}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if data[i, j] > 0.6 * max(1, float(data.max())) else "black",
                )
        ax.set(
            xticks=range(len(caps)),
            xticklabels=caps,
            yticks=range(len(models)),
            yticklabels=models,
            xlabel="Bond cap",
            title=label,
        )
        fig.colorbar(mesh, ax=ax, label="Fallback frequency (%)")
    fig.suptitle("Observed fallback use — pooled call counts")
    fig.savefig(output / "fallback_frequency.png", dpi=180)
    fig.savefig(output / "fallback_frequency.pdf")
    plt.close(fig)

    by_case = defaultdict(dict)
    for row in rows:
        by_case[row["key"]][row["variant"]] = row
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    colors = {model: plt.get_cmap("tab20")(i % 20) for i, model in enumerate(models)}
    for model in models:
        selected = [arms for arms in by_case.values() if arms["fallback"]["model"] == model]
        for ax, metric in zip(axes, ["max_energy_drift", "final_infidelity"], strict=True):
            x = [max(1e-16, arms["scalar"][metric]) for arms in selected]
            y = [max(1e-16, arms["fallback"][metric]) for arms in selected]
            sizes = [45 if arms["fallback"]["fallback_attempts"] else 15 for arms in selected]
            ax.scatter(x, y, label=model, s=sizes, alpha=0.7, color=colors[model])
    for ax, title in zip(axes, ["Maximum energy drift", "Final infidelity"], strict=True):
        ax.set(xscale="log", yscale="log", xlabel="Scalar correction", ylabel="With fallback", title=title)
        limits = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(limits, limits, "k--", linewidth=0.8)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Matched trajectories; larger markers indicate fallback use")
    fig.savefig(output / "accuracy_comparison.png", dpi=180)
    fig.savefig(output / "accuracy_comparison.pdf")
    plt.close(fig)


def report(output: Path, *, plots: bool = True) -> dict:
    """Write summaries with explicit denominators and completion status."""
    rows, groups = collect(output)
    manifest = json.loads((output / "manifest.json").read_text())
    expected = len(manifest["signature"]["cases"])
    complete = len(rows) // 3
    totals = Counter()
    for row in rows:
        if row["variant"] == "fallback":
            totals.update({
                key: row[key]
                for key in [
                    "scalar_calls",
                    "already_satisfied",
                    "fallback_attempts",
                    "scan_successes",
                    "frame_successes",
                    "fallback_failures",
                    "infeasible",
                    "solver_failures",
                    "no_minimizer",
                    "gauge_failures",
                ]
            })
    summary = {
        "expected_cases": expected,
        "complete_cases": complete,
        "arms": len(rows),
        "fallback_totals": dict(totals),
        "eventful_runs": sum(row["fallback_attempts"] > 0 for row in rows if row["variant"] == "fallback"),
        "groups": groups,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(output / "runs.csv", rows)
    # Preserve all experimental factors in a one-row-per-run table, and provide
    # pooled marginal counts to expose timestep, state and size effects.
    slices = []
    for factor in ["model", "length", "cap", "initial", "dt"]:
        for value in sorted({row[factor] for row in rows}):
            chosen = [row for row in rows if row["variant"] == "fallback" and row[factor] == value]
            counts = {key: sum(row[key] for row in chosen) for key in totals}
            slices.append({
                "factor": factor,
                "value": value,
                "runs": len(chosen),
                "eventful_runs": sum(row["fallback_attempts"] > 0 for row in chosen),
                **counts,
                "rate": counts["fallback_attempts"] / counts["scalar_calls"] if counts["scalar_calls"] else None,
            })
    write_csv(output / "slices.csv", slices)
    write_csv(
        output / "groups.csv", [{key: value for key, value in group.items() if key != "reasons"} for group in groups]
    )
    lines = [
        "# Small-system fallback experiment",
        "",
        f"Completed **{complete}/{expected} configurations**, with **{len(rows)} trajectories**.",
        "",
        "Each configuration has uncorrected, scalar-only, and fallback-enabled arms, starting from identical "
        "prepared tensors. Energy and reference dynamics use an independently assembled full-state Hamiltonian.",
        "",
        "## Fallback frequency",
        "",
        "Only fallback-enabled arms enter this table. Rates pool counts rather than average run percentages. "
        "Time steps within a trajectory are correlated; these are descriptive frequencies for the chosen grid.",
        "",
        "| Model | Bond cap | Events / calls | Rate | Events / needed calls | Scan | Frame | Infeasible | Other failure |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in groups:
        lines.append(
            f"| {group['model']} | {group['cap']} | {group['fallback_attempts']}/{group['scalar_calls']} | "
            f"{100 * group['rate']:.3f}% | {group['fallback_attempts']}/{group['needed_calls']} | "
            f"{group['scan_successes']} | {group['frame_successes']} | {group['infeasible']} | "
            f"{group['fallback_failures'] - group['infeasible']} |"
        )
    lines += [
        "",
        f"Total: **{totals['fallback_attempts']} fallback entries in {totals['scalar_calls']} calls**; "
        f"{totals['scan_successes']} scan recoveries, {totals['frame_successes']} full-frame recoveries, "
        f"{totals['fallback_failures']} unresolved.",
        "",
        "## Accuracy and cost",
        "",
        "| Variant | Median maximum energy drift | Worst maximum drift | Median final infidelity | Evolution seconds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for variant in ["uncorrected", "scalar", "fallback"]:
        selected = [row for row in rows if row["variant"] == variant]
        if selected:
            lines.append(
                f"| {variant} | {np.median([r['max_energy_drift'] for r in selected]):.3e} | "
                f"{max(r['max_energy_drift'] for r in selected):.3e} | "
                f"{np.median([r['final_infidelity'] for r in selected]):.3e} | "
                f"{sum(r['evolution_seconds'] for r in selected):.2f} |"
            )
    lines += [
        "",
        "Energy drift is sampled after each complete timestep. Before/after-correction maxima are "
        "reported separately in runs.csv. Infidelity is normalized and phase independent. "
        "Timings exclude measured observer work but are not calibrated overhead estimates: process contention, "
        "warmup and trajectory divergence affect them.",
        "",
        "## Reproducibility",
        "",
        "- `manifest.json`: full grid, dependency versions, base commit, and SHA-256 hashes of executed code.",
        "- `source.tar.gz`: the executed source, including local modifications.",
        "- `cases/*/result.json`: time traces, counters, and event metadata.",
        "- `cases/*/fallback/event-*.npz`: every pre-fallback state, including center and orientation.",
        "- `cases/*/replay.json` and `replay_summary.json`: replay verification when available.",
        "- `runs.csv`, `groups.csv`, `slices.csv`, `summary.json`: reusable numeric results.",
        "",
        "Initial states and Hamiltonian normalization are specified in manifest.json and each result. "
        "Product states use seeded 1e-10 padding; random entangled states use seeded complex tensors. "
        "References use the actual prepared tensors. Initial chi defaults to min(cap, 4), or is fixed "
        "by --initial-chi. Unpadded-state infidelity applies only to product states. "
        "The two time steps are compared at equal final time, so pooled rates weight the smaller step more heavily. "
        "Use runs.csv for separate time-step and initial-state comparisons.",
        "",
    ]
    replay_path = output / "replay_summary.json"
    if replay_path.exists():
        replay = json.loads(replay_path.read_text())
        lines += [f"Replay checks: **{replay['passed']}/{replay['events']} passed**, {replay['failed']} failed.", ""]
    if plots and groups:
        make_plots(output, rows, groups)
        lines += [
            "![Fallback frequency](fallback_frequency.png)",
            "",
            "![Accuracy comparison](accuracy_comparison.png)",
            "",
        ]
    (output / "report.md").write_text("\n".join(lines))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    report(args.output, plots=not args.no_plots)
