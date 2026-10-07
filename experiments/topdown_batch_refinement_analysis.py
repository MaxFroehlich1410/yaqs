# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Audit BB/L-BFGS batch refinement against the current pruning-specific controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import itertools

import matplotlib.pyplot as plt

from experiments import topdown_optimizer_stability_analysis as previous
from experiments.legacy_pruning_analysis import markdown_table
from experiments.legacy_topdown_weighted import file_hash

POLICIES = ("best_only", "best_backtracking", "bb_backtracking", "lbfgs")
CONTROLS = POLICIES[:2]
NEW_POLICIES = POLICIES[2:]
PRIMARY_BASELINES = {"B": "best_backtracking", "D": "best_only"}
COLORS = {"best_only": "#777777", "best_backtracking": "#0072B2", "bb_backtracking": "#D55E00", "lbfgs": "#009E73"}


def paired_comparisons(rows: list[dict], expected: list[tuple[str, int]]) -> list[dict]:
    """Compare each new policy with both controls and label the prescribed primary one."""
    return [
        {
            **previous.paired_summary(rows, policy, reference, variant, hardware, expected),
            "primary": reference == PRIMARY_BASELINES[variant],
        }
        for policy in NEW_POLICIES
        for reference in CONTROLS
        for hardware in previous.HARDWARE
        for variant in previous.VARIANTS
    ]


def checkpoint_at_budget(curve: list[dict], budget: float) -> dict:
    """Return the last completed batch checkpoint within a CPU budget, without interpolation."""
    if not curve or not np.isfinite(budget) or budget < 0:
        msg = "A nonempty curve and finite nonnegative CPU budget are required."
        raise ValueError(msg)
    times = [point["batch_cpu_seconds"] for point in curve]
    if any(not np.isfinite(time) or time < 0 for time in times) or any(
        right < left for left, right in itertools.pairwise(times)
    ):
        msg = "CPU checkpoints must be finite, nonnegative and ordered."
        raise ValueError(msg)
    eligible = [point for point in curve if point["batch_cpu_seconds"] <= budget]
    if not eligible:
        msg = "The CPU curve must include the common online prefix at zero batch CPU."
        raise ValueError(msg)
    return dict(eligible[-1])


def fixed_comparisons(rows: list[dict], expected: list[tuple[str, int, int]]) -> list[dict]:
    """Pair fixed-input endpoints and compare measured batch time with held checkpoints."""
    lookup = {(r["policy"], r["target_id"], r["seed"], r["depth"]): r for r in rows}
    groups = []
    for policy in NEW_POLICIES:
        for reference in CONTROLS:
            pairs, missing = [], []
            for identity in expected:
                new, old = lookup.get((policy, *identity)), lookup.get((reference, *identity))
                if new is None or old is None:
                    missing.append({"target_id": identity[0], "seed": identity[1], "depth": identity[2]})
                    continue
                budget = old["batch_curve"][-1]["batch_cpu_seconds"]
                matched = checkpoint_at_budget(new["batch_curve"], budget)
                pairs.append({
                    "target_id": identity[0],
                    "seed": identity[1],
                    "depth": identity[2],
                    "endpoint_gain_pp": 100 * (new["returned_fidelity"] - old["returned_fidelity"]),
                    "matched_gain_pp": 100 * (matched["best_fidelity"] - old["returned_fidelity"]),
                    "matched_fidelity": matched["best_fidelity"],
                    "reference_fidelity": old["returned_fidelity"],
                    "reference_batch_cpu_seconds": budget,
                    "matched_batch_cpu_seconds": matched["batch_cpu_seconds"],
                    "matched_iteration": matched["iteration"],
                    "matched_objective_evaluations": matched["objective_evaluations"],
                    "matched_gradient_evaluations": matched["gradient_evaluations"],
                    "new_batch_finished_within_budget": new["batch_curve"][-1]["batch_cpu_seconds"] <= budget,
                    "new_cpu_seconds": new["cpu_seconds"],
                    "reference_cpu_seconds": old["cpu_seconds"],
                })
            groups.append({
                "policy": policy,
                "reference": reference,
                "planned_pairs": len(expected),
                "complete_pairs": len(pairs),
                "missing_pairs": missing,
                "endpoint_gain_pp": previous.mean([p["endpoint_gain_pp"] for p in pairs]),
                "matched_gain_pp": previous.mean([p["matched_gain_pp"] for p in pairs]),
                "matched_wins": sum(p["matched_gain_pp"] > 100 * previous.TOLERANCE for p in pairs),
                "matched_losses": sum(p["matched_gain_pp"] < -100 * previous.TOLERANCE for p in pairs),
                "matched_ties": sum(abs(p["matched_gain_pp"]) <= 100 * previous.TOLERANCE for p in pairs),
                "finished_within_budget": sum(p["new_batch_finished_within_budget"] for p in pairs),
                "optimizer_cpu_ratio": previous.ratio(
                    sum(p["new_cpu_seconds"] for p in pairs), sum(p["reference_cpu_seconds"] for p in pairs)
                ),
                "budget_cpu_seconds": sum(p["reference_batch_cpu_seconds"] for p in pairs),
                "matched_cpu_seconds": sum(p["matched_batch_cpu_seconds"] for p in pairs),
                "pairs": pairs,
            })
    return groups


def fixed_summary(rows: list[dict]) -> list[dict]:
    """Retain endpoint, computational-cost and numerical-safeguard diagnostics by policy."""
    summaries = []
    for policy in POLICIES:
        selected = [r for r in rows if r["policy"] == policy]
        summaries.append({
            "policy": policy,
            "records": len(selected),
            "mean_returned_percent": previous.mean([100 * r["returned_fidelity"] for r in selected]),
            "mean_raw_final_percent": previous.mean([100 * r["final_fidelity"] for r in selected]),
            "raw_losses_over_10pp": sum(r["best_fidelity"] - r["final_fidelity"] > 0.1 for r in selected),
            "returned_losses_over_10pp": sum(r["best_fidelity"] - r["returned_fidelity"] > 0.1 for r in selected),
            **{
                key: sum(r[key] for r in selected)
                for key in (
                    "cpu_seconds",
                    "objective_evaluations",
                    "gradient_evaluations",
                    "rejected_trials",
                    "failed_searches",
                    "curvature_skips",
                    "direction_resets",
                    "bb_fallbacks",
                    "bb_bound_hits",
                )
            },
        })
    return summaries


def read_fixed(output: Path, cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """Authenticate fixed inputs, shared online prefixes, best retention and stable batch steps."""
    rows, audits = [], []
    identities = set()
    for case in cases:
        root = output / "fixed" / f"{case['target_id']}__seed{case['optimizer_seed']}"
        assert json.loads((root / "complete.json").read_text()) == case
        assert all(file_hash(root / name) == digest for name, digest in case["artifacts"].items())
        common = {}
        counts = {"records": 0, "monotone_batch_steps": 0, "reference_checks": 0}
        for summary in case["records"]:
            identity = (case["target_id"], case["optimizer_seed"], summary["depth"], summary["policy"])
            if identity in identities:
                msg = f"Duplicate fixed record: {identity}"
                raise ValueError(msg)
            identities.add(identity)
            checkpoint = json.loads((root / summary["checkpoint_file"]).read_text())
            training = checkpoint["training"]
            trace = [training["initial_fidelity"], *training["fidelity_trace"]]
            assert training["best_fidelity"] == max(trace)
            assert training["returned_fidelity"] == max(trace) == summary["returned_fidelity"]
            assert training["returned_theta"] == checkpoint["theta"]
            assert checkpoint["ideal_fidelity"] == summary["returned_fidelity"]
            prefix = (checkpoint["initial_theta"], trace[:101], training["online_final_theta"])
            if summary["depth"] not in common:
                common[summary["depth"]] = prefix
            else:
                assert common[summary["depth"]] == prefix
            assert summary["online_prefix_max_error"] <= 1e-12
            assert summary["independent_returned_fidelity_error"] < 1e-10
            if summary["policy"] == "best_backtracking":
                assert summary["reference_theta_max_error"] <= 1e-12
                assert summary["reference_trace_max_error"] <= 1e-12
                counts["reference_checks"] += 1
            if summary["policy"] != "best_only":
                for step in training["records"]:
                    if step["phase"] == "batch":
                        assert step["fidelity"] >= step["before_fidelity"] - 1e-12
                        counts["monotone_batch_steps"] += 1
            curve = batch_curve(training)
            checkpoint_at_budget(curve, curve[-1]["batch_cpu_seconds"])
            rows.append({
                "target_id": case["target_id"],
                "seed": case["optimizer_seed"],
                **summary,
                "curvature_skips": training["curvature_skips"],
                "direction_resets": training["direction_resets"],
                "bb_fallbacks": training["bb_fallbacks"],
                "bb_bound_hits": training["bb_bound_hits"],
                "batch_curve": curve,
                "raw_trace": trace,
            })
            counts["records"] += 1
        audits.append({"target_id": case["target_id"], "seed": case["optimizer_seed"], **counts})
    return rows, audits


def batch_curve(training: dict) -> list[dict]:
    """Extract completed batch checkpoints with a shared online-prefix checkpoint at time zero."""
    online = training["online_steps"]
    prefix = [training["initial_fidelity"], *training["fidelity_trace"][:online]]
    best = max(prefix)
    prefix_records = [record for record in training["records"] if record["phase"] == "online"]
    prefix_objectives = prefix_records[-1]["cumulative_objective_evaluations"] if prefix_records else 1
    prefix_gradients = prefix_records[-1]["cumulative_gradient_evaluations"] if prefix_records else 0
    curve = [
        {
            "iteration": online,
            "batch_cpu_seconds": 0.0,
            "best_fidelity": best,
            "objective_evaluations": 0,
            "gradient_evaluations": 0,
        }
    ]
    for record in training["records"]:
        if record["phase"] != "batch":
            continue
        best = max(best, record["fidelity"])
        curve.append({
            "iteration": record["iteration"],
            "batch_cpu_seconds": record["cumulative_cpu_seconds"] - training["batch_start_cpu_seconds"],
            "best_fidelity": best,
            "objective_evaluations": record["cumulative_objective_evaluations"] - prefix_objectives,
            "gradient_evaluations": record["cumulative_gradient_evaluations"] - prefix_gradients,
        })
    return curve


def trace_figures(output: Path, rows: list[dict]) -> None:
    """Show raw ideal fidelity versus updates and retained fidelity versus completed batch CPU."""
    examples = [("tfim_ferro", 400, 3), ("haar_random_3", 400, 10)]
    groups = [[r for r in rows if (r["target_id"], r["seed"], r["depth"]) == identity] for identity in examples]
    populated = [(identity, group) for identity, group in zip(examples, groups, strict=True) if group]
    if not populated:
        return
    figure, axes = plt.subplots(2, len(populated), figsize=(7 * len(populated), 8), squeeze=False)
    for column, (identity, group) in enumerate(populated):
        for row in group:
            axes[0, column].plot(100 * np.array(row["raw_trace"]), color=COLORS[row["policy"]], label=row["policy"])
            axes[1, column].step(
                [p["batch_cpu_seconds"] for p in row["batch_curve"]],
                [100 * p["best_fidelity"] for p in row["batch_curve"]],
                where="post",
                color=COLORS[row["policy"]],
                label=row["policy"],
            )
        axes[0, column].axvline(100, color="black", linestyle="--", linewidth=0.7)
        axes[0, column].set(
            title=f"{previous.LABELS[identity[0]]}, seed {identity[1]}, depth {identity[2]}",
            xlabel="Optimization update (0 = initial)",
            ylabel="Raw ideal fidelity (%)",
        )
        axes[1, column].set(xlabel="Completed batch CPU (s)", ylabel="Best retained ideal fidelity (%)")
        axes[0, column].legend(fontsize=8)
        axes[1, column].legend(fontsize=8)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(output / f"batch_refinement_traces.{extension}", dpi=180)
    plt.close(figure)


def overview(output: Path, data: dict) -> None:
    """Export downstream paired gains and the measured fidelity/cost tradeoff."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for offset, policy in enumerate(NEW_POLICIES):
        groups = sorted(
            [r for r in data["paired"] if r["policy"] == policy and r["primary"]],
            key=lambda row: (previous.HARDWARE.index(row["hardware"]), previous.VARIANTS.index(row["variant"])),
        )
        for index, group in enumerate(groups):
            if group["gain_pp"] is not None:
                axes[0].bar(
                    index + (offset - 0.5) * 0.35,
                    group["gain_pp"],
                    width=0.35,
                    color=COLORS[policy],
                    label=policy if index == 0 else None,
                )
    axes[0].set_xticks(range(4), [f"{h}\n{v}" for h in previous.HARDWARE for v in previous.VARIANTS])
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Exact noisy fidelity gain over current baseline (pp)")
    axes[0].legend()
    for group in data["policies"]:
        if group["mean_percent"] is not None:
            cpu = group["cost_totals"]["total_search_cpu_seconds"] / group["recorded_searches"]
            axes[1].scatter(
                cpu,
                group["mean_percent"],
                color=COLORS[group["policy"]],
                marker="o" if group["variant"] == "B" else "s",
            )
            axes[1].annotate(f"{group['hardware'][0]}-{group['variant']}", (cpu, group["mean_percent"]), fontsize=7)
    for policy in POLICIES:
        axes[1].scatter([], [], color=COLORS[policy], label=policy)
    axes[1].legend(fontsize=8)
    axes[1].set(xlabel="Mean standalone full-search CPU (s)", ylabel="Exact selected noisy fidelity (%)")
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(output / f"batch_refinement_overview.{extension}", dpi=180)
    plt.close(figure)


def report(data: dict) -> str:
    """Render the completed comparisons while keeping failures, cost and reference scope explicit."""
    number = previous.number
    lines = [
        "# Batch refinement: BB and L-BFGS against current controls",
        "",
        (
            "The primary construction baseline is **best_backtracking for B** and **best_only for D**. "
            "Both strong controls are also reported separately. Only the noiseless construction batch phase changes; "
            "the common 100 online updates, pruning/repair, M=100 noisy fine-tuning, independent validation and "
            "optional depth-selection protocol remain fixed. Final fidelity is exact density-matrix evaluation after "
            "locking the validation choice."
        ),
        "",
        (
            f"Planned: {data['planned_fixed_optimizations']} fixed-input optimizations and {data['planned_searches']} "
            f"full searches. Recorded: {len(data['fixed_rows'])} fixed-input optimizations "
            f"and {len(data['rows'])} searches."
        ),
        "",
        "## Fixed construction inputs",
        "",
        (
            "Every policy starts from the same archived best_backtracking input at each depth. These frozen inputs "
            "differ from an adaptive pipeline's later warm starts. All policies retain their best observed parameters "
            "including iteration zero. Counts below sum actual objective and gradient calls, rejected trials and "
            "curvature safeguards. Gradient calls include a loss evaluation; standalone loss calls are additional."
        ),
        "",
    ]
    lines.append(
        markdown_table(
            [
                "Policy",
                "Inputs",
                "Returned F (%)",
                "Raw loss >10pp",
                "CPU s",
                "Loss calls",
                "Gradient calls",
                "Rejected",
                "Failed searches",
                "Curvature skips",
                "Direction resets",
                "BB fallbacks",
                "BB bound hits",
            ],
            [
                [
                    g["policy"],
                    str(g["records"]),
                    number(g["mean_returned_percent"]),
                    str(g["raw_losses_over_10pp"]),
                    number(g["cpu_seconds"], 2),
                    str(g["objective_evaluations"]),
                    str(g["gradient_evaluations"]),
                    str(g["rejected_trials"]),
                    str(g["failed_searches"]),
                    str(g["curvature_skips"]),
                    str(g["direction_resets"]),
                    str(g["bb_fallbacks"]),
                    str(g["bb_bound_hits"]),
                ]
                for g in data["fixed"]
            ],
        )
    )
    lines.extend([
        "",
        "## Fixed-input comparisons at the control's batch CPU budget",
        "",
        (
            "The equal-update endpoint is not compute matched. For a descriptive CPU-matched comparison, "
            "each control's measured batch CPU is the budget; the alternative uses its last fully completed "
            "batch checkpoint within that budget, retaining the best fidelity reached by then. No interpolation "
            "or partially finished iteration is used. The common online prefix is available at batch CPU zero. "
            "Methods finishing all requested updates early hold their final retained state. Batch clocks subtract "
            "the recorded phase start from each completed-step timestamp, excluding final return serialization. "
            "These timing-dependent comparisons are post hoc, not new optimizations; small timing differences "
            "are not precise speedups. Full optimizer CPU above includes the online phase and optimizer diagnostics, "
            "but excludes independent Qiskit checks."
        ),
        "",
    ])
    lines.append(
        markdown_table(
            [
                "Policy",
                "Reference",
                "Pairs",
                "Equal-update gain (pp)",
                "CPU-budget gain (pp)",
                "Budget W/L/T",
                "Optimizer CPU ratio",
                "Finished within budget",
            ],
            [
                [
                    g["policy"],
                    g["reference"],
                    f"{g['complete_pairs']}/{g['planned_pairs']}",
                    number(g["endpoint_gain_pp"]),
                    number(g["matched_gain_pp"]),
                    f"{g['matched_wins']}/{g['matched_losses']}/{g['matched_ties']}",
                    number(g["optimizer_cpu_ratio"]),
                    str(g["finished_within_budget"]),
                ]
                for g in data["fixed_paired"]
            ],
        )
    )
    lines.extend([
        "",
        "![Raw construction traces and completed batch CPU curves](batch_refinement_traces.png)",
        "",
        (
            "These trace examples use saved initial vectors from the latest best_backtracking-grown pipeline. "
            "Despite identical target/seed/depth labels, their inputs differ from the earlier instability study's "
            "original-grown inputs; the historical 99.542% to 7.037% collapse is not being replayed here."
        ),
        "",
    ])
    for primary, heading in (
        (True, "Primary full-pipeline comparisons"),
        (False, "Secondary strong-control comparisons"),
    ):
        groups = [g for g in data["paired"] if g["primary"] == primary]
        lines.extend([
            f"## {heading}",
            "",
            (
                "Pairs match target, optimizer seed, pruning and noise profile. "
                "Fidelity differences are percentage points; win/loss tolerance is 1e-10 in fidelity units. "
                "CPU ratios divide summed paired full-search costs."
            ),
            "",
        ])
        lines.extend((
            markdown_table(
                [
                    "Policy",
                    "Reference",
                    "Noise",
                    "Pruning",
                    "Pairs",
                    "F (%)",
                    "Gain (pp)",
                    "Median (pp)",
                    "W/L/T",
                    "CPU ratio",
                    "Worst (pp)",
                ],
                [
                    [
                        g["policy"],
                        g["reference"],
                        g["hardware"],
                        g["variant"],
                        f"{g['complete_pairs']}/{g['planned_pairs']}",
                        number(g["mean_percent"]),
                        number(g["gain_pp"]),
                        number(g["median_gain_pp"]),
                        f"{g['wins']}/{g['losses']}/{g['ties']}",
                        number(g["cost_ratios"]["total_search_cpu_seconds"]),
                        number(g["minimum_gain_pp"]),
                    ]
                    for g in groups
                ],
            ),
            "",
        ))
    lines.extend([
        "## Resources and complete search costs",
        "",
        (
            "All visited candidates, including discarded and failed ones, are charged. Construction shared across "
            "pruning/noise branches, and pruning shared across noise profiles, is charged in full to each hypothetical "
            "standalone search. Summing these branch costs duplicates that shared work and is neither actual suite CPU "
            "nor parallel suite wall time. Final exact evaluation and compilation are excluded. "
            "Selected-circuit resource means use valid "
            "endpoints only. Missing outcomes are never replaced with zero."
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
                "CX",
                "CX depth",
                "Selected depth",
                "Cap hits",
                "Failures",
                "Construct s",
                "Prune s",
                "Fine-tune s",
                "Validation s",
                "Total s",
            ],
            [
                [
                    g["policy"],
                    g["hardware"],
                    g["variant"],
                    f"{g['complete_searches']}/{g['recorded_searches']}",
                    *[number(g["resources"][key], 2) for key in previous.RESOURCE_KEYS],
                    number(g["mean_selected_depth"], 2),
                    str(g["depth_cap_hits"]),
                    str(g["failed_candidates"]),
                    *[number(g["cost_totals"][key], 1) for key in previous.COST_KEYS],
                ]
                for g in data["policies"]
            ],
        )
    )
    for hardware in previous.HARDWARE:
        lines.extend(["", f"## Per-target exact fidelity: {hardware}", ""])
        table = []
        for target in data["target_ids"]:
            cells = [previous.LABELS[target]]
            for policy in POLICIES:
                for variant in previous.VARIANTS:
                    current = [
                        r
                        for r in data["rows"]
                        if (r["policy"], r["target_id"], r["variant"], r["hardware"])
                        == (policy, target, variant, hardware)
                        and r["fidelity"] is not None
                    ]
                    cells.append(
                        number(previous.mean([100 * r["fidelity"] for r in current]))
                        if len(current) == len(data["seeds"])
                        else f"missing ({len(current)}/{len(data['seeds'])})"
                    )
            table.append(cells)
        lines.append(markdown_table(["Target", *[f"{p} / {v}" for p in POLICIES for v in previous.VARIANTS]], table))
    lines.extend([
        "",
        "## Audits and limits",
        "",
        (
            f"Authenticated {len(data['audits'])} pipeline and {len(data['fixed_audits'])} fixed-input cases. "
            "Checks cover source/input/artifact hashes, shared online prefixes, best retention, batch monotonicity "
            "for safeguarded methods, independent Qiskit endpoint checks, controls reproducing their archives, "
            "100-history multiplicities and jumps, separate validation RNG, validation-map regeneration and "
            "locked stopping/selection decisions. Final exact scores do not select the winning depth."
        ),
        "",
        (
            "Only construction changed. D repair and noisy cross-trajectory updates still use their prior policies. "
            "The full searches have equal nominal update limits but can visit different depths and cost different "
            "amounts. CPU matching above applies only to fixed-input batch refinement, not the adaptive pipeline. "
            f"Previously used development targets, {len(data['seeds'])} optimizer seeds, "
            "correlated depths and shared branches "
            "do not establish statistical significance or generalization. Compiled CX resources remain separate "
            "from the historical logical-noise profiles; these are not calibrated hardware predictions."
        ),
        "",
        "Missing pipeline pairs: "
        + json.dumps([
            {
                "policy": g["policy"],
                "reference": g["reference"],
                "variant": g["variant"],
                "hardware": g["hardware"],
                **r,
            }
            for g in data["paired"]
            for r in g["missing_pairs"]
        ]),
        "",
        "![Full-pipeline fidelity and computation](batch_refinement_overview.png)",
        "",
        "## All pipeline endpoints",
        "",
    ])
    lines.append(
        markdown_table(
            [
                "Target",
                "Seed",
                "Policy",
                "Noise",
                "Pruning",
                "F (%)",
                "Selected depth",
                "CX",
                "CX depth",
                "CPU s",
                "Stop",
            ],
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
    """Audit all available records and export comparisons and scientific figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["sources"].items():
        assert file_hash(previous.REPOSITORY / name) == digest, name
    for name, digest in manifest["inputs"].items():
        assert file_hash(Path(name)) == digest, name
    assert manifest["primary_baselines"] == PRIMARY_BASELINES
    cases = json.loads((output / "results.json").read_text())
    fixed_cases = json.loads((output / "fixed_results.json").read_text())
    expected = [(target, seed) for target in manifest["target_ids"] for seed in manifest["optimizer_seeds"]]
    rows = previous.search_rows(cases)
    fixed, fixed_audits = read_fixed(output, fixed_cases)
    for case in cases:
        if case["policy"] in CONTROLS:
            parity = case["reference_parity"]
            assert parity["maximum_parameter_error"] <= 1e-12
            assert parity["maximum_trace_error"] <= 1e-12
            assert parity["selection_identical"]
            assert parity["training_banks_identical"]
    data = {
        "target_ids": manifest["target_ids"],
        "seeds": manifest["optimizer_seeds"],
        "primary_baselines": PRIMARY_BASELINES,
        "planned_searches": len(expected) * len(POLICIES) * len(previous.VARIANTS) * len(previous.HARDWARE),
        "planned_fixed_optimizations": len(manifest["fixed_inputs"]) * len(POLICIES),
        "rows": rows,
        "fixed_rows": fixed,
        "fixed_audits": fixed_audits,
        "fixed": fixed_summary(fixed),
        "fixed_paired": fixed_comparisons(
            fixed, [(item["target_id"], item["optimizer_seed"], item["depth"]) for item in manifest["fixed_inputs"]]
        ),
        "policies": [
            previous.policy_summary(rows, policy, variant, hardware)
            for policy in POLICIES
            for hardware in previous.HARDWARE
            for variant in previous.VARIANTS
        ],
        "paired": paired_comparisons(rows, expected),
        "audits": [
            previous.audit_case(
                output / "pipeline" / case["policy"] / f"{case['target_id']}__seed{case['optimizer_seed']}", case
            )
            for case in cases
        ],
        "reference_parity": [
            {
                "policy": case["policy"],
                "target_id": case["target_id"],
                "seed": case["optimizer_seed"],
                "parity": case["reference_parity"],
            }
            for case in cases
            if case["policy"] in CONTROLS
        ],
    }
    (output / "comparison.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    trace_figures(output, fixed)
    overview(output, data)
    (output / "COMPARISON.md").write_text(report(data))
    (output / "analysis_source.py.txt").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"fixed": data["fixed"], "primary": [g for g in data["paired"] if g["primary"]]}, indent=2))


if __name__ == "__main__":
    main()
