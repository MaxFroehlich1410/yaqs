#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
# SPDX-License-Identifier: MIT
# Licensed under the MIT License

"""Derive all manuscript tables and figures from the same completed trajectories."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import paper_reproduction as run

mpl.use("Agg")
import matplotlib.pyplot as plt

# Transcribed from Tables I-II of the attached September 27 manuscript.
# Order: relative final energy drift, maximum final total-spin drift, final S2 drift, infidelity.
PUBLISHED = {
    "bug-none-none": [6.0e-5, 6.2e-7, 8.0e-2, 6.72e-2],
    "bug-joint4-none": [4.4e-15, 3.5e-14, 1.0e-1, 6.83e-2],
    "tdvp-none-none": [1.1e-4, 1.6e-13, 1.6e-1, 7.03e-2],
    "tdvp-joint4-none": [4.7e-15, 8.7e-17, 1.6e-1, 7.04e-2],
    "bug-joint4-full": [2.8e-15, 2.3e-14, 8.6e-2, 6.78e-2],
    "bug-jointS2-none": [6.2e-15, 3.2e-1, 1.8e-15, 1.53e-1],
    "bug-jointS2-full": [2.5e-15, 3.5e-1, 3.2e-14, 1.85e-1],
    "bug-joint5-none": [4.9e-4, 8.5e-4, 2.1e-2, 1.26e-1],
    "bug-joint5-full": [4.9e-15, 7.0e-14, 2.1e-14, 1.25e-1],
}
PUBLISHED_OBSERVABLES = {
    "bug-none-none": [8.7e-4, 1.22e-2, 2.1e-3, 6.0e-5],
    "bug-joint4-full": [8.7e-4, 1.23e-2, 2.0e-3, 2.5e-4],
    "bug-joint5-full": [5.8e-2, 1.23e-2, 2.4e-3, 1.1e-3],
}
METRICS = ("energy_relative", "total_spin", "S2", "infidelity")
FAMILIES = ("local_spin", "bond_energy", "corr_zz", "staggered")


def write_csv(path, rows) -> None:
    """Write a flat machine-readable table with stable column names."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(result):
    """Return final and full-trajectory maxima without mixing the two."""
    initial = result["initial_values"]
    series = {
        "energy_relative": [abs(r["values"]["H"] - initial["H"]) / (abs(initial["H"]) or 1) for r in result["trace"]],
        "total_spin": [max(abs(r["values"][s] - initial[s]) for s in ("Sx", "Sy", "Sz")) for r in result["trace"]],
        "S2": [abs(r["values"]["S2"] - initial["S2"]) for r in result["trace"]],
        "infidelity": [r["infidelity"] for r in result["trace"]],
        "spin_observable_error": [r["spin_observable_error"] for r in result["trace"]],
        **{family: [r["family_errors"][family] for r in result["trace"]] for family in FAMILIES},
    }
    return {
        "arm": result["arm"]["id"],
        **{f"final_{k}": v[-1] for k, v in series.items()},
        **{f"max_{k}": max(v) for k, v in series.items()},
        "peak_bond": max(r["bond"] for r in result["trace"]),
    }


def comparison(identifier, name, actual, published, digits):
    """Compare printed precision, treating round-off values as a numerical floor."""
    tolerance = 1e-12 if published < 1e-12 else 0.5 * 10 ** (np.floor(np.log10(abs(published))) - digits + 1)
    return {
        "arm": identifier,
        "metric": name,
        "published": published,
        "reproduced": actual,
        "absolute_difference": abs(actual - published),
        "comparison_tolerance": float(tolerance),
        "within_printed_precision_or_roundoff": bool(abs(actual - published) <= tolerance),
    }


def validate(output):
    """Reject missing arms, changed references, incorrect clocks, or broken counters."""
    output = Path(output)
    plan = run.read(output / "manifest.json")["plan"]
    reference = run.read(output / "reference" / "data.json")
    if run.digest(output / "reference" / "vectors.npy") != reference["vectors_sha256"]:
        msg = "Reference array changed"
        raise ValueError(msg)
    results = {}
    for arm in plan["arms"]:
        folder = output / "arms" / arm["id"]
        done = run.read(folder / "done.json")
        if run.digest(folder / "result.json") != done["result_sha256"]:
            msg = "Trajectory checksum mismatch"
            raise ValueError(msg)
        result = run.read(folder / "result.json")
        p, rows, counts = result["protocol"], result["trace"], result["counts"]
        if result["status"] != "complete" or result["arm"] != arm or p != plan["protocol"]:
            msg = "Incomplete or mismatched trajectory"
            raise ValueError(msg)
        if (
            result["initial_sha256"] != reference["initial_sha256"]
            or result["reference_sha256"] != reference["vectors_sha256"]
        ):
            msg = "Different initial state or reference"
            raise ValueError(msg)
        if len(rows) != p["steps"] + 1 or not np.allclose(
            [r["time"] for r in rows], np.arange(p["steps"] + 1) * p["dt"], atol=1e-13, rtol=0
        ):
            msg = "Incomplete time grid"
            raise ValueError(msg)
        if max(r["bond"] for r in rows) > p["cap"]:
            msg = "Bond cap violation"
            raise ValueError(msg)
        expected = (2 if arm["integrator"] == "bug" else 1) * p["steps"] if arm["variant"] != "none" else 0
        if result["diagnostics"]["schedule_calls"] != expected:
            msg = "Unexpected number of correction schedules"
            raise ValueError(msg)
        if arm["variant"].startswith("joint"):
            expected *= p["length"] if arm["sweep"] == "full" else 1
            if counts["joint_calls"] != expected or counts["fallback_attempts"]:
                msg = "Joint/scalar dispatcher mismatch"
                raise ValueError(msg)
        elif counts["scalar_calls"] != expected:
            msg = "Scalar call denominator mismatch"
            raise ValueError(msg)
        if counts["fallback_attempts"] != len(result["diagnostics"]["events"]):
            msg = "Incomplete event records"
            raise ValueError(msg)
        results[arm["id"]] = result
    return plan, results


def scalar_plot(results, output) -> None:
    """Compare scalar ablations for both integrators using the same saved data."""
    fig, axes = plt.subplots(3, 2, figsize=(8.0, 6.8), sharex=True)
    colors = ("#303030", "#d55e00", "#0072b2")
    for col, integrator in enumerate(("bug", "tdvp")):
        keys = [f"{integrator}-none-none", f"{integrator}-pinH-none-fallback-0", f"{integrator}-pinH-none-fallback-1"]
        for key, label, color in zip(keys, ("Uncorrected", "Scalar only", "With fallback"), colors, strict=True):
            r = results[key]
            times = [row["time"] for row in r["trace"]]
            energy = [
                abs(row["values"]["H"] - r["initial_values"]["H"]) / max(1, abs(r["initial_values"]["H"]))
                for row in r["trace"]
            ]
            arrays = [
                energy,
                [row["infidelity"] for row in r["trace"]],
                [row["spin_observable_error"] for row in r["trace"]],
            ]
            for ax, values in zip(axes[:, col], arrays, strict=True):
                ax.semilogy(times, np.maximum(values, 1e-17), color=color, label=label, lw=1.1)
            if r["arm"]["fallback"]:
                for event in r["diagnostics"]["events"]:
                    axes[0, col].axvline(event["time"], color=color, alpha=0.18, lw=0.7)
        axes[0, col].set_title("BUG" if integrator == "bug" else "Two-site TDVP")
        axes[0, col].axhline(r["protocol"]["conserve_tol"], color="0.5", linestyle=":", lw=0.8)
        axes[-1, col].set_xlabel("Time")
    for row, label in enumerate(("Scaled energy error", "Infidelity", "Maximum spin-observable error")):
        axes[row, 0].set_ylabel(label)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"scalar_ablation.{suffix}", dpi=220)
    plt.close(fig)


def report(output):
    """Write tables, figures, validation, and a comparison with published numbers."""
    output = Path(output)
    plan, results = validate(output)
    folder = output / "report"
    folder.mkdir(exist_ok=True)
    measured = {key: metrics(value) for key, value in results.items()}
    comparisons = []
    production_protocol = plan["protocol"] == run.make_plan()["protocol"]
    if production_protocol:
        for key, published in PUBLISHED.items():
            for name, value in zip(METRICS, published, strict=True):
                comparisons.append(
                    comparison(key, name, measured[key][f"final_{name}"], value, 3 if name == "infidelity" else 2)
                )
        for key, published in PUBLISHED_OBSERVABLES.items():
            for name, value in zip(FAMILIES, published, strict=True):
                comparisons.append(
                    comparison(key, name, measured[key][f"final_{name}"], value, 3 if name == "bond_energy" else 2)
                )
    write_csv(folder / "trajectory_metrics.csv", list(measured.values()))
    if comparisons:
        write_csv(folder / "published_comparison.csv", comparisons)
    counts = []
    for key, result in results.items():
        d, c = result["diagnostics"], result["counts"]
        counts.append({
            "arm": key,
            **c,
            "schedule_calls": d["schedule_calls"],
            "schedule_unresolved": d["schedule_unresolved"],
            "qualified_calls": d["qualified_calls"] if result["arm"]["variant"] == "pinH" else None,
            "qualified_fallbacks": d["qualified_fallbacks"] if result["arm"]["variant"] == "pinH" else None,
            "fallback_rate": c["fallback_attempts"] / c["scalar_calls"] if c["scalar_calls"] else None,
            "qualified_rate": d["qualified_fallbacks"] / d["qualified_calls"] if d["qualified_calls"] else None,
            "max_pre_scaled_residual": d["max_pre_residual"],
            "max_post_scaled_residual": d["max_post_residual"],
        })
    write_csv(folder / "correction_counts.csv", counts)
    # Legacy format lets the unchanged original plotting script reproduce Figure 1.
    legacy = {
        "protocol": plan["protocol"],
        "initial_values": next(iter(results.values()))["initial_values"],
        "arms": [],
    }
    for key in plan["figure_1"]:
        result = results[key]
        legacy["arms"].append({
            **result["arm"],
            "times": [r["time"] for r in result["trace"]],
            "drift": {
                name: [r["values"][name] - result["initial_values"][name] for r in result["trace"]]
                for name in result["initial_values"]
            },
        })
    run.pilot.write_json(folder / "l16_trajectory.json", legacy)
    subprocess.run(
        [
            sys.executable,
            str(run.HERE / "plot_trajectory.py"),
            "--input",
            str(folder / "l16_trajectory.json"),
            "--output-dir",
            str(folder),
        ],
        check=True,
    )
    scalar_plot(results, folder)
    lines = [
        "# Manuscript reproduction",
        "",
        f"Completed {len(results)} unique trajectories and one shared reference.",
        "",
        "The original Hamiltonian is unscaled. Scalar fallback does not modify the joint solver.",
        "",
        "All tables and curves below use the same saved full-step trajectories. Final errors and trajectory maxima are separate.",
        "",
    ]
    for title, keys, columns in (
        ("Table I", plan["table_I"], METRICS),
        ("Table II", plan["table_II"], METRICS),
        ("Table III", plan["table_III"], FAMILIES),
    ):
        lines += [f"## {title}", "", "| Arm | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
        lines += [
            "| " + key + " | " + " | ".join(f"{measured[key]['final_' + c]:.5e}" for c in columns) + " |"
            for key in keys
        ]
        lines.append("")
        write_csv(
            folder / f"{title.lower().replace(' ', '_')}.csv",
            [{"arm": key, **{c: measured[key]["final_" + c] for c in columns}} for key in keys],
        )
    mismatches = [c for c in comparisons if not c["within_printed_precision_or_roundoff"]]
    lines += [
        "## Verification and limitations",
        "",
        f"Printed-table comparisons outside the declared tolerance: {len(mismatches)}/{len(comparisons)}.",
        "Round-off entries are compared at an absolute floor of 1e-12; other entries use half the last printed digit.",
        "A changed result is reported, not automatically accepted as an improvement.",
        "Original numeric Figure 1 data are not supplied: the figure is regenerated from the documented protocol, not certified against digitized pixels.",
        "Joint unresolved-center counts and unresolved complete schedules are distinct. A later center can resolve an earlier failure.",
        "Zero accuracy-qualified denominators are undefined. Joint arms are not assigned scalar fallback probabilities.",
        "Timing includes tracking; estimated evolution time is not a clean overhead benchmark.",
        "The completed dimer study is reused separately, retaining its normalization, reference validation and source archive.",
        "These small fixed-seed experiments do not establish general fallback probabilities or accuracy improvements.",
        "",
        "![Reproduced Figure 1](trajectory.png)",
        "",
        "![Scalar ablations](scalar_ablation.png)",
        "",
    ]
    if not production_protocol:
        lines.insert(2, "**SMOKE/CALIBRATION PROTOCOL: not manuscript production results.**")
    (folder / "REPORT.md").write_text("\n".join(lines))
    run.pilot.write_json(
        folder / "validation.json",
        {
            "complete": True,
            "trajectories": len(results),
            "production_protocol": production_protocol,
            "published_comparisons": len(comparisons),
            "outside_printed_tolerance": mismatches,
            "initial_sha256": next(iter(results.values()))["initial_sha256"],
        },
    )
    return measured
