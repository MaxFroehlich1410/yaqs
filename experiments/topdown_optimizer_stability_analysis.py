# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Audit and compare the controlled noiseless optimizer stability experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from experiments.legacy_pruning_analysis import markdown_table
from experiments.legacy_topdown_weighted import file_hash, load_circuit
from experiments.topdown_depth_analysis import LABELS
from experiments.topdown_depth_search import HARDWARE, PROTOCOL, REPOSITORY, VARIANTS, Patience, sample_validation

POLICIES = ("original_final", "best_only", "best_backtracking")
SEEDS = (300, 400)
TOLERANCE = 1e-10
COST_KEYS = (
    "construction_cpu_seconds",
    "pruning_cpu_seconds",
    "fine_tuning_cpu_seconds",
    "validation_cpu_seconds",
    "total_search_cpu_seconds",
)
RESOURCE_KEYS = ("logical_two_qubit", "compiled_two_qubit", "compiled_two_qubit_depth")


def mean(values: list[float]) -> float | None:
    """Return a finite mean, or null when no observations are available."""
    return float(np.mean(values)) if values else None


def ratio(numerator: float, denominator: float) -> float | None:
    """Return a ratio only when the denominator is positive."""
    return numerator / denominator if denominator > 0 else None


def search_rows(cases: list[dict]) -> list[dict]:
    """Flatten all searches, preserving failed selections and rejecting duplicates."""
    rows, identities = [], set()
    for case in cases:
        for result in case["results"]:
            key = (case["policy"], case["target_id"], case["optimizer_seed"], result["variant"], result["hardware"])
            if key in identities:
                msg = f"Duplicate search identity: {key}"
                raise ValueError(msg)
            identities.add(key)
            selected = result["selected"]
            rows.append({
                "policy": case["policy"],
                "target_id": case["target_id"],
                "seed": case["optimizer_seed"],
                "variant": result["variant"],
                "hardware": result["hardware"],
                "fidelity": selected["exact"]["fidelity"] if selected else None,
                "ideal_fidelity": selected["ideal"]["fidelity"] if selected else None,
                "selected_depth": result["selected_depth"],
                "stop_depth": len(result["decisions"]),
                "stop_reason": result["stop_reason"],
                "resources": selected["resources"] if selected else None,
                "cost": result["search_cost"],
            })
    return rows


def paired_summary(
    rows: list[dict],
    policy: str,
    reference: str,
    variant: str,
    hardware: str,
    expected: list[tuple[str, int]],
) -> dict:
    """Compare identical target/seed pairs with explicit absent and failed outcomes."""
    lookup = {
        (row["policy"], row["target_id"], row["seed"]): row
        for row in rows
        if row["variant"] == variant and row["hardware"] == hardware
    }
    pairs, missing = [], []
    for target, seed in expected:
        new, old = lookup.get((policy, target, seed)), lookup.get((reference, target, seed))
        if new is None or old is None or new["fidelity"] is None or old["fidelity"] is None:
            missing.append({
                "target_id": target,
                "seed": seed,
                "policy_status": "absent" if new is None else "failed" if new["fidelity"] is None else "complete",
                "reference_status": "absent" if old is None else "failed" if old["fidelity"] is None else "complete",
            })
        else:
            pairs.append((new, old))
    differences = [new["fidelity"] - old["fidelity"] for new, old in pairs]
    new_cost = {key: sum(new["cost"][key] for new, _ in pairs) for key in COST_KEYS}
    old_cost = {key: sum(old["cost"][key] for _, old in pairs) for key in COST_KEYS}
    return {
        "policy": policy,
        "reference": reference,
        "variant": variant,
        "hardware": hardware,
        "planned_pairs": len(expected),
        "complete_pairs": len(pairs),
        "missing_pairs": missing,
        "mean_percent": mean([100 * new["fidelity"] for new, _ in pairs]),
        "reference_mean_percent": mean([100 * old["fidelity"] for _, old in pairs]),
        "gain_pp": mean([100 * value for value in differences]),
        "median_gain_pp": float(np.median(differences) * 100) if differences else None,
        "minimum_gain_pp": min(differences) * 100 if differences else None,
        "maximum_gain_pp": max(differences) * 100 if differences else None,
        "wins": sum(value > TOLERANCE for value in differences),
        "losses": sum(value < -TOLERANCE for value in differences),
        "ties": sum(abs(value) <= TOLERANCE for value in differences),
        "cost_totals": new_cost,
        "reference_cost_totals": old_cost,
        "cost_ratios": {key: ratio(new_cost[key], old_cost[key]) for key in COST_KEYS},
        "pairs": [
            {"target_id": new["target_id"], "seed": new["seed"], "gain_pp": 100 * (new["fidelity"] - old["fidelity"])}
            for new, old in pairs
        ],
    }


def policy_summary(rows: list[dict], policy: str, variant: str, hardware: str) -> dict:
    """Charge failed candidates and retain resource denominators for valid endpoints."""
    current = [row for row in rows if (row["policy"], row["variant"], row["hardware"]) == (policy, variant, hardware)]
    valid = [row for row in current if row["fidelity"] is not None]
    return {
        "policy": policy,
        "variant": variant,
        "hardware": hardware,
        "recorded_searches": len(current),
        "complete_searches": len(valid),
        "mean_percent": mean([100 * row["fidelity"] for row in valid]),
        "mean_selected_depth": mean([row["selected_depth"] for row in valid]),
        "mean_stop_depth": mean([row["stop_depth"] for row in current]),
        "depth_cap_hits": sum(row["stop_reason"] == "depth_cap" for row in current),
        "failed_candidates": sum(row["cost"]["failed_candidates"] for row in current),
        "cost_totals": {key: sum(row["cost"][key] for row in current) for key in COST_KEYS},
        "resources": {key: mean([row["resources"][key] for row in valid]) for key in RESOURCE_KEYS},
    }


def fixed_rows(cases: list[dict]) -> list[dict]:
    """Flatten fixed-start construction records without conflating depth warm starts."""
    rows, seen = [], set()
    for case in cases:
        for record in case["records"]:
            key = (case["target_id"], case["optimizer_seed"], record["depth"], record["policy"])
            if key in seen:
                msg = f"Duplicate fixed-start identity: {key}"
                raise ValueError(msg)
            seen.add(key)
            rows.append({"target_id": case["target_id"], "seed": case["optimizer_seed"], **record})
    return rows


def fixed_summary(rows: list[dict]) -> list[dict]:
    """Summarize actual endpoint protection separately from raw optimizer stability."""
    control = {(row["target_id"], row["seed"], row["depth"]): row for row in rows if row["policy"] == POLICIES[0]}
    groups = []
    for policy in POLICIES:
        current = [row for row in rows if row["policy"] == policy]
        paired = [
            (row, control[row["target_id"], row["seed"], row["depth"]])
            for row in current
            if (row["target_id"], row["seed"], row["depth"]) in control
        ]
        groups.append({
            "policy": policy,
            "records": len(current),
            "paired_records": len(paired),
            "raw_peak_to_final_drops_over_10pp": sum(
                row["best_fidelity"] - row["final_fidelity"] > 0.1 for row in current
            ),
            "returned_peak_losses_over_10pp": sum(
                row["best_fidelity"] - row["returned_fidelity"] > 0.1 for row in current
            ),
            "mean_final_percent": mean([100 * row["final_fidelity"] for row in current]),
            "mean_returned_percent": mean([100 * row["returned_fidelity"] for row in current]),
            "mean_returned_gain_pp": mean([
                100 * (row["returned_fidelity"] - old["returned_fidelity"]) for row, old in paired
            ]),
            "cpu_seconds": sum(row["cpu_seconds"] for row in current),
            "paired_cpu_ratio": ratio(
                sum(row["cpu_seconds"] for row, _ in paired), sum(old["cpu_seconds"] for _, old in paired)
            ),
            "objective_evaluations": sum(row["objective_evaluations"] for row in current),
            "gradient_evaluations": sum(row["gradient_evaluations"] for row in current),
            "rejected_trials": sum(row["rejected_trials"] for row in current),
            "failed_searches": sum(row["failed_searches"] for row in current),
            "maximum_original_trace_error": max(
                (row["original_trace_max_error"] for row in current if row.get("original_trace_max_error") is not None),
                default=None,
            ),
            "maximum_original_theta_error": max(
                (row["original_theta_max_error"] for row in current if row.get("original_theta_max_error") is not None),
                default=None,
            ),
        })
    return groups


def audit_case(root: Path, case: dict) -> dict:
    """Authenticate saved artifacts, trajectory weights, validation seeds and decisions."""
    if any(file_hash(root / name) != digest for name, digest in case["artifacts"].items()):
        msg = f"Artifact hash mismatch in {root}"
        raise ValueError(msg)
    assert json.loads((root / "complete.json").read_text()) == case
    if file_hash(root / "selection_locked.json") != case["selection_sha256"]:
        msg = "Selection lock was changed."
        raise ValueError(msg)
    locked = json.loads((root / "selection_locked.json").read_text())
    totals = dict.fromkeys(
        (
            "training_banks",
            "training_paths",
            "retained_paths",
            "jump_paths",
            "validation_banks",
            "validation_paths",
            "validation_jump_paths",
            "failed_training_banks",
            "failed_validation_banks",
        ),
        0,
    )
    for result, branch in zip(case["results"], locked["branches"], strict=True):
        for key in ("variant", "hardware", "selected_depth", "stop_reason", "decisions"):
            assert result[key] == branch[key]
        tracker = Patience()
        for decision in result["decisions"]:
            depth = decision["depth"]
            stem = f"depth_{depth:02d}_{result['variant']}_{result['hardware']}"
            record = json.loads((root / "search" / (stem + ".json")).read_text())
            assert record["depth"] == depth
            circuit = load_circuit(record)
            training = record["training"]
            if training is not None:
                assert training["steps"] == 200
                assert training["online_steps"] == 100
                assert training["seed"] == case["optimizer_seed"] + 5000
                assert training["num_trajectories"] == 100
                assert training["compress_no_jump"]
                assert len(training["noise_banks"]) == 1
                bank = training["noise_banks"][0]
                assert bank["sampled_paths"] == 100
                assert len(bank["events"]) == 100
                jumps = [len(events) for events in bank["events"]]
                assert jumps == bank["raw_jump_counts"]
                assert (
                    bank["raw_sha256"]
                    == hashlib.sha256(json.dumps(bank["events"], sort_keys=True).encode()).hexdigest()
                )
                assert bank["no_jump_paths"] == jumps.count(0)
                assert bank["jump_paths"] == sum(value > 0 for value in jumps)
                multiplicities = [1] * bank["jump_paths"] + ([bank["no_jump_paths"]] if bank["no_jump_paths"] else [])
                assert sorted(bank["multiplicities"]) == sorted(multiplicities)
                assert sum(bank["multiplicities"]) == 100
                assert len(multiplicities) == bank["retained_paths"]
                np.testing.assert_allclose(bank["weights"], np.array(bank["multiplicities"]) / 100, atol=1e-15, rtol=0)
                assert not circuit.gates or bank["jump_paths"] > 0
                totals["training_banks"] += 1
                totals["training_paths"] += 100
                totals["retained_paths"] += bank["retained_paths"]
                totals["jump_paths"] += bank["jump_paths"]
            else:
                assert record["status"] == "no_jump_training_bank"
                totals["failed_training_banks"] += 1
            validation = record["validation"]
            if validation is not None:
                entropy = validation["entropy"]
                expected = [
                    20260914,
                    6,
                    list(LABELS).index(case["target_id"]),
                    case["optimizer_seed"],
                    VARIANTS.index(result["variant"]),
                    HARDWARE.index(result["hardware"]),
                    depth,
                    90000,
                ]
                assert entropy == expected
                path = root / "search" / validation["bank_file"]
                assert file_hash(path) == validation["bank_sha256"]
                codes = np.load(path)["codes"]
                assert len(codes) == PROTOCOL["validation_paths"] == validation["sampled_paths"]
                np.testing.assert_array_equal(
                    codes, sample_validation(circuit, result["hardware"], len(codes), entropy)
                )
                jumps = int(np.any(codes != 0, axis=1).sum())
                assert jumps == validation["jump_paths"]
                assert len(codes) - jumps == validation["no_jump_paths"]
                assert not circuit.gates or jumps > 0 or record["status"] == "no_jump_validation_bank"
                totals["validation_banks"] += 1
                totals["validation_paths"] += len(codes)
                totals["validation_jump_paths"] += jumps
                totals["failed_validation_banks"] += int(record["status"] == "no_jump_validation_bank")
            valid = record["status"] == "complete"
            score, se = (validation["fidelity"], validation["standard_error"]) if valid else (None, 0.0)
            replay = tracker.observe(depth, score, se)
            assert decision["score"] == score
            assert decision["standard_error"] == se
            assert all(decision[key] == value for key, value in replay.items())
        assert tracker.best_depth == result["selected_depth"]
        selected = result["selected"]
        assert selected is None or selected["depth"] == tracker.best_depth
        for evaluation in result["evaluations"]:
            stem = f"depth_{evaluation['depth']:02d}_{result['variant']}_{result['hardware']}"
            assert evaluation == json.loads((root / "search" / (stem + ".evaluation.json")).read_text())
        assert selected is None or selected in result["evaluations"]
    return {"policy": case["policy"], "target_id": case["target_id"], "seed": case["optimizer_seed"], **totals}


def audit_fixed(root: Path, case: dict) -> dict:
    """Verify saved endpoint policies and batch monotonicity on archived fixed inputs."""
    assert json.loads((root / "complete.json").read_text()) == case
    assert all(file_hash(root / name) == digest for name, digest in case["artifacts"].items())
    originals = {}
    protected, monotone_steps, trace_parity = 0, 0, 0
    for summary in case["records"]:
        saved = json.loads((root / summary["checkpoint_file"]).read_text())
        training = saved["training"]
        trace = [training["initial_fidelity"], *training["fidelity_trace"]]
        assert training["best_fidelity"] == max(trace)
        assert training["final_fidelity"] == trace[-1]
        assert training["returned_fidelity"] == saved["ideal_fidelity"] == summary["returned_fidelity"]
        assert saved["theta"] == training["returned_theta"]
        assert training["steps"] == 200
        assert training["online_steps"] == 100
        assert training["gradient_evaluations"] == 100
        if summary["policy"] == "original_final":
            assert training["returned_fidelity"] == trace[-1]
            assert summary["original_theta_max_error"] < 1e-12
            assert summary["original_trace_max_error"] < 1e-12
            originals[summary["depth"]] = saved
        else:
            assert training["returned_fidelity"] == max(trace)
            protected += 1
        if summary["policy"] == "best_only":
            original = originals[summary["depth"]]
            assert saved["initial_theta"] == original["initial_theta"]
            assert training["fidelity_trace"] == original["training"]["fidelity_trace"]
            assert training["final_theta"] == original["theta"]
            trace_parity += 1
        if summary["policy"] == "best_backtracking":
            for step in training["records"]:
                if step["phase"] == "batch":
                    assert step["fidelity"] >= step["before_fidelity"] - 1e-12
                    monotone_steps += 1
        assert summary["independent_returned_fidelity_error"] < 1e-10
    gradient_errors = [item["absolute_error"] for check in case["gradient_checks"] for item in check["checks"]]
    assert all(error < 1e-7 for error in gradient_errors)
    return {
        "target_id": case["target_id"],
        "seed": case["optimizer_seed"],
        "protected_endpoints": protected,
        "best_only_trace_parity": trace_parity,
        "monotone_backtracking_batch_steps": monotone_steps,
        "finite_difference_checks": len(gradient_errors),
        "maximum_gradient_error": max(gradient_errors, default=None),
    }


def number(value: float | None, digits: int = 3) -> str:
    """Format unavailable values explicitly rather than as zero or NaN."""
    return "missing" if value is None else f"{value:.{digits}f}"


def overview(output: Path, data: dict) -> None:
    """Export scientific figures showing fidelity gains and computational costs."""
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"original_final": "#666666", "best_only": "#0072B2", "best_backtracking": "#D55E00"}
    for policy in POLICIES:
        rows = [row for row in data["fixed_rows"] if row["policy"] == policy]
        axes[0].plot(sorted(100 * row["returned_fidelity"] for row in rows), label=policy, color=colors[policy])
    axes[0].set(xlabel="Fixed-start case (sorted within policy)", ylabel="Returned ideal fidelity (%)")
    axes[0].legend(fontsize=8)
    for offset, policy in enumerate(POLICIES[1:]):
        groups = [row for row in data["paired"] if row["policy"] == policy and row["reference"] == POLICIES[0]]
        for index, row in enumerate(groups):
            if row["gain_pp"] is not None:
                axes[1].bar(
                    index + (offset - 0.5) * 0.35,
                    row["gain_pp"],
                    width=0.35,
                    color=colors[policy],
                    label=policy if index == 0 else None,
                )
    axes[1].set_xticks(range(4), [f"{h}\n{v}" for h in HARDWARE for v in VARIANTS])
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_ylabel("Paired noisy fidelity gain (pp)")
    axes[1].legend(fontsize=8)
    for row in data["policies"]:
        if row["mean_percent"] is not None:
            cpu = row["cost_totals"]["total_search_cpu_seconds"] / row["recorded_searches"]
            axes[2].scatter(
                cpu, row["mean_percent"], color=colors[row["policy"]], marker="o" if row["variant"] == "B" else "s"
            )
            axes[2].annotate(f"{row['hardware'][0]}-{row['variant']}", (cpu, row["mean_percent"]), fontsize=7)
    axes[2].set(xlabel="Mean standalone search CPU (s)", ylabel="Exact selected noisy fidelity (%)")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"stability_overview.{suffix}", dpi=180)
    plt.close(figure)


def construction_trace_rows(output: Path) -> list[dict]:
    """Load complete illustrative triplets, including the pre-update initial state."""
    examples = []
    for target, seed, depth in (("tfim_ferro", 400, 3), ("haar_random_3", 400, 10)):
        root = output / "fixed" / f"{target}__seed{seed}"
        paths = {policy: root / policy / f"construction_{depth:02d}.json" for policy in POLICIES}
        if not all(path.exists() for path in paths.values()):
            continue
        training = {policy: json.loads(path.read_text())["training"] for policy, path in paths.items()}
        original, retained, backtracking = (training[policy] for policy in POLICIES)
        assert original["fidelity_trace"] == retained["fidelity_trace"]
        assert original["initial_fidelity"] == retained["initial_fidelity"]
        examples.append({
            "target_id": target,
            "seed": seed,
            "depth": depth,
            "original": [original["initial_fidelity"], *original["fidelity_trace"]],
            "backtracking": [backtracking["initial_fidelity"], *backtracking["fidelity_trace"]],
            "retained_iteration": retained["best_iteration"],
            "retained_fidelity": retained["returned_fidelity"],
            "online_steps": original["online_steps"],
        })
    return examples


def trace_figure(output: Path, examples: list[dict]) -> None:
    """Plot fixed-input optimization traces and the retained best original checkpoint."""
    if not examples:
        return
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, len(examples), figsize=(6.5 * len(examples), 4.5), squeeze=False, sharey=True)
    for axis, record in zip(axes.flat, examples, strict=True):
        axis.plot(
            100 * np.array(record["original"]),
            color="#777777",
            linewidth=1.4,
            label="Original / best-only raw trajectory",
        )
        axis.plot(
            100 * np.array(record["backtracking"]), color="#D55E00", linewidth=1.5, label="Backtracking batch updates"
        )
        axis.scatter(
            record["retained_iteration"],
            100 * record["retained_fidelity"],
            marker="*",
            s=150,
            color="#0072B2",
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
            label="Best-only returned checkpoint",
        )
        axis.axvline(
            record["online_steps"], linestyle="--", color="black", linewidth=0.8, label="Online-to-batch switch"
        )
        axis.set(
            title=f"{LABELS[record['target_id']]} — seed {record['seed']}, depth {record['depth']}",
            xlabel="Optimization update (0 = initial state)",
            ylabel="Exact construction ideal fidelity (%)",
            ylim=(-2, 103),
        )
        axis.legend(loc="lower left", fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"stability_traces.{suffix}", dpi=180)
    plt.close(figure)


def report(data: dict) -> str:
    """Render a complete comparison with pairing, failed outcomes and all endpoint rows."""
    lines = [
        "# Noiseless optimizer stability comparison",
        "",
        (
            "Only construction optimization changes. Pruning B/D, D's 5+5 fixed-rate repair, noisy "
            "fine-tuning with 100 compressed weighted histories, independent validation, depth cap and "
            "patience remain fixed. Exact noisy density-matrix fidelity is evaluated after locking selection."
        ),
        "",
        (
            "The fixed-start screen reuses the archived input at each depth. The pipeline instead grows "
            "from each policy's returned parameters, so later inputs and visited depths can differ."
        ),
        "",
        "## Fixed-start construction",
        "",
    ]
    lines.append(
        markdown_table(
            [
                "Policy",
                "Cases",
                "Raw drops >10 pp",
                "Returned losses >10 pp",
                "Returned F (%)",
                "Gain (pp)",
                "CPU ratio",
                "Loss calls",
                "Gradient calls",
                "Rejected trials",
                "Failed searches",
            ],
            [
                [
                    g["policy"],
                    str(g["records"]),
                    str(g["raw_peak_to_final_drops_over_10pp"]),
                    str(g["returned_peak_losses_over_10pp"]),
                    number(g["mean_returned_percent"]),
                    number(g["mean_returned_gain_pp"]),
                    number(g["paired_cpu_ratio"]),
                    str(g["objective_evaluations"]),
                    str(g["gradient_evaluations"]),
                    str(g["rejected_trials"]),
                    str(g["failed_searches"]),
                ]
                for g in data["fixed"]
            ],
        )
    )
    lines.extend([
        "",
        (
            "Loss calls are standalone forward evaluations, including initial and trial evaluations. "
            "Batch gradient calls also produce an objective; online sweeps require forward/backward propagation. "
            "These counters are not interchangeable units of compute. Fixed-start optimizer CPU includes "
            "optimizer diagnostics, but excludes separate Qiskit checks and finite-difference auditing. "
            "Best retention protects the returned state but does not make its raw trajectory monotone."
        ),
        "",
        "### Recorded failure examples",
        "",
    ])
    for group in data["fixed"]:
        if group["maximum_original_trace_error"] is not None:
            lines.insert(
                -3,
                (
                    f"{group['policy']} maximum raw fidelity-trace error against the original: "
                    f"{group['maximum_original_trace_error']:.3e}."
                ),
            )
    highlight = [
        r
        for r in data["fixed_rows"]
        if (r["target_id"], r["seed"], r["depth"]) in {("tfim_ferro", 400, 3), ("haar_random_3", 400, 10)}
    ]
    lines.append(
        markdown_table(
            ["Target", "Seed", "Depth", "Policy", "Online F (%)", "Peak F (%)", "Final F (%)", "Returned F (%)"],
            [
                [
                    r["target_id"],
                    str(r["seed"]),
                    str(r["depth"]),
                    r["policy"],
                    number(100 * r["online_final_fidelity"]),
                    number(100 * r["best_fidelity"]),
                    number(100 * r["final_fidelity"]),
                    number(100 * r["returned_fidelity"]),
                ]
                for r in highlight
            ],
        )
    )
    if data.get("trace_examples"):
        lines.extend(["", "![Fixed-input ideal-fidelity optimization traces](stability_traces.png)", ""])
    lines.extend([
        "",
        "## Paired pipeline comparisons",
        "",
        (
            "Each group plans 18 target/seed pairs. Wins/losses use a fidelity tolerance of 1e-10. "
            "Missing or failed selections are listed explicitly and never replaced with zero. "
            "CPU ratios divide summed paired CPU totals."
        ),
        "",
    ])
    lines.append(
        markdown_table(
            ["Policy", "Reference", "Noise", "Pruning", "Pairs", "F (%)", "Gain (pp)", "W/L/T", "CPU ratio"],
            [
                [
                    g["policy"],
                    g["reference"],
                    g["hardware"],
                    g["variant"],
                    f"{g['complete_pairs']}/{g['planned_pairs']}",
                    number(g["mean_percent"]),
                    number(g["gain_pp"]),
                    f"{g['wins']}/{g['losses']}/{g['ties']}",
                    number(g["cost_ratios"]["total_search_cpu_seconds"]),
                ]
                for g in data["paired"]
            ],
        )
    )
    missing = [
        {"policy": g["policy"], "reference": g["reference"], "variant": g["variant"], "hardware": g["hardware"], **r}
        for g in data["paired"]
        for r in g["missing_pairs"]
    ]
    lines.extend([
        "",
        "Missing pairs: " + (json.dumps(missing) if missing else "none."),
        "",
        "## Resources and complete search costs",
        "",
        (
            "Each standalone branch is charged all its construction, pruning, fine-tuning and validation work, "
            "including failed candidates. Shared construction is charged separately to each standalone search. "
            "Retrospective exact evaluation and compilation are excluded from search CPU. "
            "Resources describe the selected circuit."
        ),
        "",
    ])
    lines.append(
        markdown_table(
            [
                "Policy",
                "Noise",
                "Pruning",
                "Valid",
                "2Q",
                "Compiled CX",
                "CX depth",
                "Selected depth",
                "Cap hits",
                "Failed candidates",
                "Construct s",
                "Prune s",
                "Fine-tune s",
                "Validate s",
                "Total s",
            ],
            [
                [
                    g["policy"],
                    g["hardware"],
                    g["variant"],
                    f"{g['complete_searches']}/{g['recorded_searches']}",
                    *[number(g["resources"][key], 2) for key in RESOURCE_KEYS],
                    number(g["mean_selected_depth"], 2),
                    str(g["depth_cap_hits"]),
                    str(g["failed_candidates"]),
                    *[number(g["cost_totals"][key], 1) for key in COST_KEYS],
                ]
                for g in data["policies"]
            ],
        )
    )
    for hardware in HARDWARE:
        lines.extend([
            "",
            f"## Per-target exact fidelity: {hardware}",
            "",
            "Means below require both seeds (300 and 400).",
            "",
        ])
        table = []
        for target in LABELS:
            row = [LABELS[target]]
            for policy in POLICIES:
                for variant in VARIANTS:
                    current = [
                        r
                        for r in data["rows"]
                        if (r["target_id"], r["hardware"], r["policy"], r["variant"])
                        == (target, hardware, policy, variant)
                        and r["fidelity"] is not None
                    ]
                    row.append(
                        number(mean([100 * r["fidelity"] for r in current]))
                        if len(current) == 2
                        else f"missing ({len(current)}/2)"
                    )
            table.append(row)
        lines.append(markdown_table(["Target", *[f"{p} / {v}" for p in POLICIES for v in VARIANTS]], table))
    lines.extend([
        "",
        "## Audit and interpretation limits",
        "",
        (
            f"Authenticated {len(data['audits'])} pipeline cases. Artifact hashes, locked validation decisions, "
            "100-history multiplicities and weights, raw jump events, and independently seeded validation "
            "banks were checked. The audit regenerates validation maps; final exact fidelity sums the model's "
            "noise outcomes and needs no trajectory test sample."
        ),
        "",
        (
            f"Fixed-start checks: {sum(r['best_only_trace_parity'] for r in data.get('fixed_audits', []))} "
            "best-only traces match their original trajectories exactly; "
            f"{sum(r['monotone_backtracking_batch_steps'] for r in data.get('fixed_audits', []))} "
            "backtracking batch steps preserve fidelity within 1e-12. "
            f"Independent finite-difference directional checks: "
            f"{sum(r['finite_difference_checks'] for r in data.get('fixed_audits', []))}."
        ),
        "",
        (
            "A construction improvement need not improve the full pipeline: simultaneous B deletions can "
            "accumulate loss, "
            "D repair remains fixed-rate and can lose fidelity, and deeper circuits expose more noisy gates. "
            "The comparisons include only two optimizer seeds per fixed target and are descriptive. "
            "Depths and branches are correlated, and the target fixtures have already informed method development. "
            "No confidence claim across unseen targets follows. A line search costs extra objective evaluations; "
            "these runs are update-budget matched, not CPU matched. Compiled resources are reference CX counts, "
            "while noise remains on logical gates. Noisy cross-trajectory optimization is unchanged."
        ),
        "",
        "![Stability, noisy fidelity gains and costs](stability_overview.png)",
        "",
        "## All pipeline endpoints",
        "",
    ])
    lines.append(
        markdown_table(
            ["Target", "Seed", "Policy", "Noise", "Pruning", "F (%)", "Depth", "CX", "CX depth", "CPU s", "Stop"],
            [
                [
                    r["target_id"],
                    str(r["seed"]),
                    r["policy"],
                    r["hardware"],
                    r["variant"],
                    number(100 * r["fidelity"] if r["fidelity"] is not None else None),
                    str(r["selected_depth"]),
                    str(r["resources"]["compiled_two_qubit"]) if r["resources"] else "missing",
                    str(r["resources"]["compiled_two_qubit_depth"]) if r["resources"] else "missing",
                    number(r["cost"]["total_search_cpu_seconds"], 1),
                    r["stop_reason"],
                ]
                for r in data["rows"]
            ],
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Verify completed artifacts and export JSON, Markdown and scientific figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["sources"].items():
        assert file_hash(REPOSITORY / name) == digest, name
    for name, digest in manifest["inputs"].items():
        assert file_hash(Path(name)) == digest, name
    cases = json.loads((output / "results.json").read_text())
    fixed_cases = json.loads((output / "fixed_results.json").read_text())
    rows = search_rows(cases)
    fixed = fixed_rows(fixed_cases)
    expected = [(target, seed) for target in LABELS for seed in SEEDS]
    data: dict[str, Any] = {
        "planned_searches": 216,
        "recorded_searches": len(rows),
        "planned_fixed_optimizations": 372,
        "recorded_fixed_optimizations": len(fixed),
        "policies": [
            policy_summary(rows, policy, variant, hardware)
            for policy in POLICIES
            for hardware in HARDWARE
            for variant in VARIANTS
        ],
        "paired": [
            paired_summary(rows, policy, reference, variant, hardware, expected)
            for policy, reference in (
                (POLICIES[1], POLICIES[0]),
                (POLICIES[2], POLICIES[0]),
                (POLICIES[2], POLICIES[1]),
            )
            for hardware in HARDWARE
            for variant in VARIANTS
        ],
        "fixed": fixed_summary(fixed),
        "fixed_rows": fixed,
        "trace_examples": construction_trace_rows(output),
        "fixed_audits": [
            audit_fixed(output / "fixed" / f"{case['target_id']}__seed{case['optimizer_seed']}", case)
            for case in fixed_cases
        ],
        "rows": rows,
        "audits": [
            audit_case(
                output / "pipeline" / case["policy"] / f"{case['target_id']}__seed{case['optimizer_seed']}", case
            )
            for case in cases
        ],
    }
    (output / "comparison.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    overview(output, data)
    trace_figure(output, data["trace_examples"])
    (output / "COMPARISON.md").write_text(report(data))
    (output / "analysis_source.py.txt").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"searches": len(rows), "fixed": len(fixed), "paired": data["paired"]}, indent=2))


if __name__ == "__main__":
    main()
