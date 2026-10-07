# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Audit locked depth searches and report paired fidelity, resources and all search work."""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from qiskit.quantum_info import Statevector

from benchmarks.state_preparation.execution import completed_jobs, parse_workers
from experiments.legacy_pruning_analysis import markdown_table
from experiments.legacy_topdown_dense import PRIMITIVES, LegacyNoise
from experiments.legacy_topdown_weighted import file_hash, load_circuit
from experiments.topdown_depth_search import (
    HARDWARE,
    PROTOCOL,
    REPOSITORY,
    VARIANTS,
    Patience,
    noise_outcomes,
    protocol_for,
    read_checkpoint,
    sample_validation,
    validation_score,
)
from mqt.yaqs.optimization import KrotovTJMOptions, StatevectorBackend

LABELS = {
    "gaussian_mu0p5_sigma0p1": "Gaussian",
    "tfim_ferro": "TFIM ferromagnetic",
    "tfim_critical": "TFIM critical",
    "tfim_para": "TFIM paramagnetic",
    "haar_random_1": "Haar 1",
    "haar_random_2": "Haar 2",
    "haar_random_3": "Haar 3",
    "random_mps_bond2": "Random MPS, bond 2",
    "random_mps_bond3": "Random MPS, bond 3",
}


def qiskit_path_fidelity(record: dict, target: np.ndarray, row: np.ndarray) -> float:
    """Replay one raw validation history through independent Qiskit state evolution."""
    circuit, theta = load_circuit(record), np.array(record["theta"])
    state = Statevector.from_label("0" * circuit.num_qubits)
    for gate, code in zip(circuit.gates, row, strict=True):
        angle = gate.angle_scale * theta[gate.param_index] + gate.angle_offset
        state = state.evolve(PRIMITIVES[gate.name](angle), qargs=list(gate.sites))
        if code:
            _, operators = noise_outcomes(record["hardware"], tuple(sorted(gate.sites)))
            operator, sites = operators[int(code) - 1]
            state = state.evolve(operator, qargs=list(reversed(sites)))
    return float(abs(np.vdot(target, state.data)) ** 2)


def audit_case(job: tuple[str, str, int]) -> dict:
    """Authenticate all artifacts and replay every validation bank and stopping decision."""
    output, folder, training_paths = job
    root = Path(output) / folder
    complete = json.loads((root / "complete.json").read_text())
    assert all(file_hash(root / name) == digest for name, digest in complete["artifacts"].items())
    assert file_hash(root / "selection_locked.json") == complete["selection_sha256"]
    target = np.load(root / "target.npy")
    counts: dict = {
        "case": folder,
        "training_banks": 0,
        "training_sampled": 0,
        "failed_training_sampled": 0,
        "training_retained": 0,
        "training_nonempty_banks_with_jumps": 0,
        "empty_candidates": 0,
        "coverage_failures": 0,
        "validation_banks": 0,
        "validation_sampled": 0,
        "validation_retained": 0,
        "validation_nonempty_banks_with_jumps": 0,
        "replayed_decisions": 0,
        "max_validation_replay_error": 0.0,
        "max_qiskit_path_mean_error": 0.0,
        "max_vector_matrix_error": 0.0,
        "max_trace_error": 0.0,
        "max_hermiticity_error": 0.0,
        "max_compiled_unitary_error": 0.0,
        "minimum_eigenvalue": 1.0,
    }
    for result in complete["results"]:
        tracker = Patience()
        for decision in result["decisions"]:
            depth, variant, hardware = decision["depth"], result["variant"], result["hardware"]
            name = f"depth_{depth:02d}_{variant}_{hardware}"
            record = read_checkpoint(root / "search", name)
            assert record is not None
            counts["replayed_decisions"] += 1
            circuit, theta = load_circuit(record), np.array(record["theta"])
            if not circuit.gates:
                counts["empty_candidates"] += 1
            if record["training"]:
                bank = record["training"]["noise_banks"][0]
                assert len(record["training"]["noise_banks"]) == 1
                assert record["training"]["steps"] == 200
                events = bank["events"]
                jumps = [len(row) for row in events]
                assert len(events) == bank["sampled_paths"] == training_paths
                assert bank["raw_sha256"] == hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest()
                assert bank["raw_jump_counts"] == jumps
                assert jumps.count(0) == bank["no_jump_paths"]
                assert sum(value > 0 for value in jumps) == bank["jump_paths"]
                assert sum(bank["multiplicities"]) == training_paths
                assert bank["retained_paths"] == bank["jump_paths"] + int(bank["no_jump_paths"] > 0)
                np.testing.assert_allclose(
                    bank["weights"], np.array(bank["multiplicities"]) / training_paths, atol=1e-15
                )
                assert sorted(bank["multiplicities"]) == sorted(
                    [1] * bank["jump_paths"] + ([bank["no_jump_paths"]] if bank["no_jump_paths"] else [])
                )
                counts["training_banks"] += 1
                counts["training_sampled"] += training_paths
                counts["training_retained"] += bank["retained_paths"]
                if circuit.gates:
                    assert bank["jump_paths"] > 0
                    counts["training_nonempty_banks_with_jumps"] += 1
            else:
                assert record["status"] == "no_jump_training_bank"
                maps = StatevectorBackend(circuit, target).sample_noise_maps(
                    theta,
                    KrotovTJMOptions(num_trajectories=training_paths),
                    LegacyNoise(hardware),
                    iteration=complete["optimizer_seed"] + 5000,
                )
                assert len(maps) == training_paths
                assert not any(noise.operators for row in maps for noise in row)
                counts["failed_training_sampled"] += training_paths
            if record["validation"]:
                saved = record["validation"]
                bank_path = root / "search" / saved["bank_file"]
                assert file_hash(bank_path) == saved["bank_sha256"]
                codes = np.load(bank_path)["codes"]
                np.testing.assert_array_equal(codes, sample_validation(circuit, hardware, len(codes), saved["entropy"]))
                actual = validation_score(circuit, theta, target, hardware, codes)
                for key in ("fidelity", "standard_error", "analytic_no_jump_probability", "norm_max_error"):
                    error = abs(actual[key] - saved[key])
                    counts["max_validation_replay_error"] = max(counts["max_validation_replay_error"], error)
                    assert error < 1e-12
                for key in ("sampled_paths", "retained_paths", "jump_paths", "no_jump_paths"):
                    assert actual[key] == saved[key]
                assert actual["norm_max_error"] < 1e-10
                counts["validation_banks"] += 1
                counts["validation_sampled"] += len(codes)
                counts["validation_retained"] += saved["retained_paths"]
                if circuit.gates:
                    assert saved["jump_paths"] > 0
                    counts["validation_nonempty_banks_with_jumps"] += 1
                if depth == result["selected_depth"]:
                    jumps = np.any(codes != 0, axis=1)
                    indices = [*np.flatnonzero(~jumps)[:1], *np.flatnonzero(jumps)[:7]]
                    if len(indices) == 1:
                        indices.append(indices[0])
                    expected = np.mean([qiskit_path_fidelity(record, target, codes[index]) for index in indices])
                    sampled = validation_score(circuit, theta, target, hardware, codes[indices])["fidelity"]
                    error = abs(sampled - expected)
                    counts["max_qiskit_path_mean_error"] = max(counts["max_qiskit_path_mean_error"], error)
                    assert error < 1e-10
            valid = record["status"] == "complete"
            if not valid:
                counts["coverage_failures"] += 1
            score = record["validation"]["fidelity"] if valid else None
            se = record["validation"]["standard_error"] if valid else 0.0
            replay = tracker.observe(depth, score, se)
            assert all(replay[key] == decision[key] for key in replay)
            assert score == decision["score"]
            assert se == decision["standard_error"]
            if depth < len(result["decisions"]):
                assert tracker.stale < PROTOCOL["patience"]
        assert tracker.best_depth == result["selected_depth"]
        assert result["stop_reason"] == ("patience" if tracker.stale >= PROTOCOL["patience"] else "depth_cap")
        for evaluation in [result["baseline"], *result["evaluations"]]:
            counts["max_vector_matrix_error"] = max(
                counts["max_vector_matrix_error"], evaluation["vector_matrix_error"]
            )
            counts["max_compiled_unitary_error"] = max(
                counts["max_compiled_unitary_error"],
                evaluation["resources"]["logical_unitary_max_error_up_to_global_phase"],
            )
            for law in ("ideal", "exact", "lindblad"):
                counts["max_trace_error"] = max(counts["max_trace_error"], evaluation[law]["trace_error"])
                counts["max_hermiticity_error"] = max(
                    counts["max_hermiticity_error"], evaluation[law]["hermiticity_error"]
                )
                counts["minimum_eigenvalue"] = min(counts["minimum_eigenvalue"], evaluation[law]["minimum_eigenvalue"])
    return counts


def pareto_indices(points: np.ndarray) -> list[int]:
    """Return nondominated rows when all supplied objectives are minimized."""
    return [
        index
        for index, point in enumerate(points)
        if not any(np.all(other <= point) and np.any(other < point) for other in points)
    ]


def collect_rows(output: Path, cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """Extract every visited candidate and every paired search without imputing failures."""
    searches, candidates = [], []
    for case in cases:
        root = output / (case["target_id"] + "__seed" + str(case["optimizer_seed"]))
        for result in case["results"]:
            identity = {
                "target_id": case["target_id"],
                "seed": case["optimizer_seed"],
                "variant": result["variant"],
                "hardware": result["hardware"],
            }
            current = []
            for evaluation in result["evaluations"]:
                depth = evaluation["depth"]
                name = f"depth_{depth:02d}_{result['variant']}_{result['hardware']}"
                saved = read_checkpoint(root / "search", name)
                built = read_checkpoint(root / "search", f"construction_{depth:02d}")
                pruned = read_checkpoint(root / "search", f"depth_{depth:02d}_{result['variant']}_pruned")
                assert saved is not None
                assert built is not None
                assert pruned is not None
                bank, validation = saved["training"]["noise_banks"][0], saved["validation"]
                current.append({
                    **identity,
                    "depth": depth,
                    "selected": depth == result["selected_depth"],
                    "construction_ideal": built["ideal_fidelity"],
                    "construction_peak_ideal": max(built["training"]["fidelity_trace"]),
                    "construction_online_end_ideal": built["training"]["fidelity_trace"][99],
                    "pruned_ideal": pruned["pruning"]["final_noiseless_fidelity"],
                    "final_ideal": evaluation["ideal"]["fidelity"],
                    "fidelity": evaluation["exact"]["fidelity"],
                    "lindblad_fidelity": evaluation["lindblad"]["fidelity"],
                    "validation": validation["fidelity"],
                    "validation_se": validation["standard_error"],
                    "training_fidelity": saved["training"]["fidelity_trace"][-1],
                    "training_retained": bank["retained_paths"],
                    "training_jumps": bank["jump_paths"],
                    "training_no_jump": bank["no_jump_paths"],
                    "validation_jumps": validation["jump_paths"],
                    "resources": evaluation["resources"],
                    "construction_cpu": built["cpu_seconds"],
                    "pruning_cpu": pruned["pruning"]["cpu_seconds"],
                    "fine_tuning_cpu": saved["training_cpu_seconds"],
                    "validation_cpu": saved["validation_cpu_seconds"],
                })
            points = np.array([
                [
                    item["resources"]["logical_two_qubit"],
                    item["resources"]["compiled_two_qubit_depth"],
                    -item["fidelity"],
                ]
                for item in current
            ])
            frontier = pareto_indices(points)
            full = np.array([[*point, -item["final_ideal"]] for point, item in zip(points, current, strict=True)])
            frontier4 = pareto_indices(full)
            for index, item in enumerate(current):
                item["pareto_noisy_resources"] = index in frontier
                item["pareto_four_objectives"] = index in frontier4
            candidates.extend(current)
            selected = next((item for item in current if item["selected"]), None)
            baseline = result["baseline"]
            searches.append({
                **identity,
                "selected": selected,
                "baseline_fidelity": baseline["exact"]["fidelity"],
                "baseline_ideal": baseline["ideal"]["fidelity"],
                "baseline_depth": baseline["depth"],
                "baseline_resources": baseline["resources"],
                "baseline_cost": result["baseline_cost"],
                "search_cost": result["search_cost"],
                "stop_reason": result["stop_reason"],
                "stop_depth": len(result["decisions"]),
                "gain_pp": 100 * (selected["fidelity"] - baseline["exact"]["fidelity"]) if selected else None,
                "selection_regret_pp": 100 * (max(item["fidelity"] for item in current) - selected["fidelity"])
                if selected
                else None,
                "retrospective_best_depth": max(current, key=operator.itemgetter("fidelity"))["depth"]
                if current
                else None,
            })
    return searches, candidates


def aggregate(searches: list[dict]) -> list[dict]:
    """Calculate paired descriptive statistics and measured full-search cost ratios."""
    groups = []
    for hardware in HARDWARE:
        for variant in VARIANTS:
            rows = [row for row in searches if row["hardware"] == hardware and row["variant"] == variant]
            complete = [row for row in rows if row["selected"]]
            gains = [row["gain_pp"] for row in complete]
            base_cpu = np.mean([row["baseline_cost"]["total_search_cpu_seconds"] for row in complete])
            new_cpu = np.mean([row["search_cost"]["total_search_cpu_seconds"] for row in complete])
            groups.append({
                "hardware": hardware,
                "variant": variant,
                "planned": len(rows),
                "complete": len(complete),
                "baseline_mean_percent": float(np.mean([100 * row["baseline_fidelity"] for row in complete])),
                "new_mean_percent": float(np.mean([100 * row["selected"]["fidelity"] for row in complete])),
                "gain_pp": float(np.mean(gains)),
                "median_gain_pp": float(np.median(gains)),
                "wins": sum(gain > 1e-8 for gain in gains),
                "losses": sum(gain < -1e-8 for gain in gains),
                "ties": sum(abs(gain) <= 1e-8 for gain in gains),
                "baseline_at_least_90": sum(row["baseline_fidelity"] >= 0.9 for row in complete),
                "new_at_least_90": sum(row["selected"]["fidelity"] >= 0.9 for row in complete),
                "baseline_cpu_mean": float(base_cpu),
                "search_cpu_mean": float(new_cpu),
                "cpu_ratio": float(new_cpu / base_cpu),
                "mean_selected_depth": float(np.mean([row["selected"]["depth"] for row in complete])),
                "mean_stop_depth": float(np.mean([row["stop_depth"] for row in rows])),
                "depth_cap_cases": sum(row["stop_reason"] == "depth_cap" for row in rows),
                "candidate_failures": sum(row["search_cost"]["failed_candidates"] for row in rows),
                "selected_before_ideal_90": sum(row["selected"]["construction_ideal"] < 0.9 for row in complete),
                "mean_selection_regret_pp": float(np.mean([row["selection_regret_pp"] for row in complete])),
                "max_selection_regret_pp": max(row["selection_regret_pp"] for row in complete),
                "exact_winner_mismatches": sum(
                    row["selected"]["depth"] != row["retrospective_best_depth"] for row in complete
                ),
                "baseline_resources": {
                    key: float(np.mean([row["baseline_resources"][key] for row in complete]))
                    for key in ("logical_two_qubit", "compiled_two_qubit", "compiled_two_qubit_depth")
                },
                "new_resources": {
                    key: float(np.mean([row["selected"]["resources"][key] for row in complete]))
                    for key in ("logical_two_qubit", "compiled_two_qubit", "compiled_two_qubit_depth")
                },
                "cost_breakdown": {
                    key: float(np.mean([row["search_cost"][key] for row in rows]))
                    for key in (
                        "construction_cpu_seconds",
                        "pruning_cpu_seconds",
                        "fine_tuning_cpu_seconds",
                        "validation_cpu_seconds",
                    )
                },
            })
    return groups


def figures(output: Path, candidates: list[dict], groups: list[dict]) -> None:
    """Export static depth-response and cost/fidelity figures for the scientific report."""
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"B": "#0072B2", "D": "#D55E00"}
    for hardware in HARDWARE:
        figure, axes = plt.subplots(3, 3, figsize=(14, 11), sharey=True)
        for axis, target in zip(axes.flat, LABELS, strict=True):
            for variant in VARIANTS:
                for seed, style in ((300, "-"), (400, "--")):
                    rows = sorted(
                        [
                            row
                            for row in candidates
                            if (row["hardware"], row["target_id"], row["variant"], row["seed"])
                            == (hardware, target, variant, seed)
                        ],
                        key=operator.itemgetter("depth"),
                    )
                    axis.plot(
                        [row["depth"] for row in rows],
                        [100 * row["fidelity"] for row in rows],
                        style,
                        marker="o",
                        markersize=3,
                        color=colors[variant],
                        label=f"{variant}, seed {seed}",
                    )
                    selected = next((row for row in rows if row["selected"]), None)
                    if selected:
                        axis.scatter(
                            selected["depth"],
                            100 * selected["fidelity"],
                            marker="*",
                            s=140,
                            color=colors[variant],
                            edgecolor="black",
                            linewidth=0.5,
                            zorder=5,
                        )
            axis.set_title(LABELS[target])
            axis.set_xlabel("Construction depth index")
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.set_ylabel("Exact final noisy fidelity (%)")
            axis.set_ylim(0, 102)
            axis.grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.97))
        figure.suptitle(
            f"{hardware} logical noise profile: every fine-tuned depth candidate\n"
            "Stars mark independently validation-selected circuits; exact curves are retrospective",
            y=1.01,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        figure.savefig(output / f"depth_curves_{hardware}.png", dpi=170, bbox_inches="tight")
        figure.savefig(output / f"depth_curves_{hardware}.pdf", bbox_inches="tight")
        plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, hardware in zip(axes, HARDWARE, strict=True):
        for group in [group for group in groups if group["hardware"] == hardware]:
            variant = group["variant"]
            axis.plot(
                [group["baseline_cpu_mean"], group["search_cpu_mean"]],
                [group["baseline_mean_percent"], group["new_mean_percent"]],
                color=colors[variant],
                alpha=0.5,
            )
            for endpoint, marker in (("baseline", "o"), ("new", "*")):
                cpu = group["baseline_cpu_mean"] if endpoint == "baseline" else group["search_cpu_mean"]
                fidelity = group[endpoint + "_mean_percent"]
                axis.scatter(cpu, fidelity, color=colors[variant], marker=marker, s=90)
                axis.annotate(
                    f"{variant} {'90% cutoff' if endpoint == 'baseline' else 'depth search'}",
                    (cpu, fidelity),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
        axis.set_xscale("log")
        axis.set_title(hardware)
        axis.set_xlabel("Mean full search CPU seconds (log scale)")
        axis.set_ylabel("Mean exact noisy fidelity (%)")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "cost_fidelity.png", dpi=180, bbox_inches="tight")
    figure.savefig(output / "cost_fidelity.pdf", bbox_inches="tight")
    plt.close(figure)


def report(output: Path, searches: list[dict], groups: list[dict], audits: list[dict]) -> str:
    """Generate the detailed reproducible measurement report and all paired endpoints."""
    training_paths = json.loads((output / "manifest.json").read_text())["protocol"]["fine_tuning"]["num_trajectories"]
    text = [
        "# Fine-tuning every depth candidate: six-qubit results",
        "",
        (
            "All nine fixture targets, two optimizer seeds (300/400), pruning B/D "
            "and two historical logical noise profiles. "
            "Training uses 64-amplitude vectors; final scoring uses 64-by-64 density matrices. "
            "Random-MPS targets are loaded as dense vectors; these runs do not use MPS simulation."
        ),
        "",
        "## Protocol and comparison",
        "",
        (
            "The baseline is the original first-90%-ideal construction, "
            f"followed by B or D and {training_paths} weighted training histories. "
            "It was rerun for current CPU timings, with archived construction/pruned starting parameters "
            "and common training-bank prefixes verified. Endpoint equality is required only at M=100. "
            "The new method removes the 90% cutoff both inside each depth's optimization and between extensions. "
            "Each visited depth gets 200 noiseless updates, separate B/D pruning, "
            f"200 noisy hybrid updates on {training_paths} frozen histories, "
            "and 32,768 independent validation histories. Growth always uses the unpruned noiseless parent."
        ),
        "",
        (
            "Patience is three extensions without improving the last material validation record by "
            "0.001 + 2√(SE²_current + SE²_record), capped at depth index 16. "
            "Each depth index adds one even/odd brickwall cycle, 45 scalar gates at six qubits. "
            "The highest validation score wins, including small improvements that do not reset patience. "
            "All decisions were saved before any exact density assessment. "
            "The uncertainty margin is a heuristic, not a formal sequential confidence guarantee."
        ),
        "",
        (
            "The existing learning rates, final-iterate policy, pruning threshold and repair behavior are unchanged. "
            "Removing the threshold also changes how long noiseless training continues at early depths; "
            "this comparison therefore does not isolate candidate selection from extra optimization work."
        ),
        "",
        "## Overall fidelity",
        "",
        markdown_table(
            [
                "Profile",
                "Pruning",
                "Pairs",
                "90% baseline",
                "Depth search",
                "Gain (pp)",
                "Wins/ties/losses",
                "≥90%: old → new",
            ],
            [
                [
                    g["hardware"],
                    g["variant"],
                    str(g["complete"]),
                    f"{g['baseline_mean_percent']:.3f}%",
                    f"{g['new_mean_percent']:.3f}%",
                    f"{g['gain_pp']:+.3f}",
                    f"{g['wins']}/{g['ties']}/{g['losses']}",
                    f"{g['baseline_at_least_90']} → {g['new_at_least_90']}",
                ]
                for g in groups
            ],
        ),
        "",
        (
            "Percentages are arithmetic means over the completed, paired target/seed cases. "
            "Each target has equal weight because every target has the same two seeds. "
            "Wins use exact model expectations, not significance tests."
        ),
        "",
        "## Per-target fidelity",
        "",
        "Each cell averages seeds 300 and 400; pp denotes percentage points.",
        "",
    ]
    for hardware in HARDWARE:
        table = []
        for target, label in LABELS.items():
            cells = [label]
            for variant in VARIANTS:
                subset = [
                    row
                    for row in searches
                    if (row["hardware"], row["target_id"], row["variant"]) == (hardware, target, variant)
                    and row["selected"]
                ]
                cells.extend([
                    f"{100 * np.mean([row['baseline_fidelity'] for row in subset]):.2f}%",
                    f"{100 * np.mean([row['selected']['fidelity'] for row in subset]):.2f}%",
                    f"{np.mean([row['gain_pp'] for row in subset]):+.2f}",
                ])
            table.append(cells)
        text.extend([
            f"### {hardware}",
            "",
            markdown_table(["Target", "B old", "B search", "B Δ pp", "D old", "D search", "D Δ pp"], table),
            "",
        ])
    text.extend([
        "## Full search cost and selected circuit resources",
        "",
        (
            "CPU includes all visited construction, pruning/repair, noisy training and validation, "
            "including discarded candidates. Each standalone search is charged its full construction/pruning prefix "
            "even where the suite shares it across branches. Ratios are ratios of mean CPU on paired cases. "
            "Exact assessment, reference compilation and checkpoint JSON bookkeeping "
            "are outside the search CPU metric; validation bank generation, compression and persistence are included. "
            "Actual suite wall time is reported separately."
        ),
        "",
        markdown_table(
            [
                "Profile",
                "Method",
                "Old CPU s",
                "Search CPU s",
                "Ratio",
                "Logical 2Q old → new",
                "Compiled CX old → new",
                "2Q depth old → new",
            ],
            [
                [
                    g["hardware"],
                    g["variant"],
                    f"{g['baseline_cpu_mean']:.2f}",
                    f"{g['search_cpu_mean']:.2f}",
                    f"{g['cpu_ratio']:.2f}x",
                    *[
                        f"{g['baseline_resources'][key]:.2f} → {g['new_resources'][key]:.2f}"
                        for key in ("logical_two_qubit", "compiled_two_qubit", "compiled_two_qubit_depth")
                    ],
                ]
                for g in groups
            ],
        ),
        "",
        markdown_table(
            [
                "Profile",
                "Method",
                "Construction s",
                "Pruning/repair s",
                "Fine-tuning s",
                "Validation s",
                "Mean candidates",
            ],
            [
                [
                    g["hardware"],
                    g["variant"],
                    *[f"{value:.2f}" for value in g["cost_breakdown"].values()],
                    f"{g['mean_stop_depth']:.2f}",
                ]
                for g in groups
            ],
        ),
        "",
        "![Full search cost versus fidelity](cost_fidelity.png)",
        "",
        (
            "Compiled resources use the unchanged fixed CX reference: line connectivity, identity layout, "
            "rz/sx/x/cx basis, optimization level 1 and transpiler seed 1729. "
            "The noise model acts on the logical circuit, not the compiled CX circuit. "
            "These are consistent comparative resource counts, not predictions for a native device."
        ),
        "",
        "## Depth selection and validation accuracy",
        "",
        markdown_table(
            [
                "Profile",
                "Method",
                "Mean selected depth",
                "Mean stopped depth",
                "At cap",
                "Selected ideal <90%",
                "Exact-rank mismatches",
                "Mean/max regret pp",
            ],
            [
                [
                    g["hardware"],
                    g["variant"],
                    f"{g['mean_selected_depth']:.2f}",
                    f"{g['mean_stop_depth']:.2f}",
                    str(g["depth_cap_cases"]),
                    str(g["selected_before_ideal_90"]),
                    str(g["exact_winner_mismatches"]),
                    f"{g['mean_selection_regret_pp']:.4f} / {g['max_selection_regret_pp']:.4f}",
                ]
                for g in groups
            ],
        ),
        "",
        (
            "Here 'selected ideal <90%' refers to the unpruned noiseless parent, before pruning or noisy fine-tuning. "
            "Regret compares the validation-selected circuit with the best exact fidelity "
            "among already visited candidates only. It does not estimate the best unvisited deeper circuit. "
            "Exact scoring is retrospective and never changes the winner. "
            "The validation and exact scores are under the same known model; this is independent numerical evaluation, "
            "not evidence of generalization to another noise model or hardware."
        ),
        "",
        "![IBM depth curves](depth_curves_IBM.png)",
        "",
        "![Google depth curves](depth_curves_Google.png)",
        "",
        (
            "The exported `candidates.json` contains every complete candidate with "
            "(logical 2Q count, compiled 2Q depth, ideal fidelity, exact noisy fidelity), "
            "and both the three-objective resource/noisy-fidelity frontier and the full four-objective frontier. "
            "Frontiers are computed only after selection; ideal fidelity is not a selection constraint."
        ),
        "",
        "## Trajectory coverage and audit",
        "",
    ])
    sums = {
        key: sum(audit[key] for audit in audits)
        for key in (
            "training_banks",
            "training_sampled",
            "failed_training_sampled",
            "training_retained",
            "training_nonempty_banks_with_jumps",
            "empty_candidates",
            "coverage_failures",
            "validation_banks",
            "validation_sampled",
            "validation_retained",
            "validation_nonempty_banks_with_jumps",
            "replayed_decisions",
        )
    }
    text.extend([
        (
            f"Visited {sums['replayed_decisions']} candidates: {sums['training_banks']} completed training banks, "
            f"{sums['coverage_failures']} coverage failures, and {sums['empty_candidates']} empty circuit candidates. "
            f"Every one of the {sums['training_nonempty_banks_with_jumps']} "
            "completed nonempty training banks contained jumps."
        ),
        "",
        (
            f"Completed training banks sampled {sums['training_sampled']:,} raw histories "
            f"and retained {sums['training_retained']:,} "
            f"representatives ({100 * sums['training_retained'] / sums['training_sampled']:.2f}% of the raw count). "
            "These are path counts, not a measured wall-time speedup; "
            "cross-trajectory operations also depend on batch size."
        ),
        "",
        (
            f"The failed training banks account for a further {sums['failed_training_sampled']:,} raw histories, "
            f"giving {sums['training_sampled'] + sums['failed_training_sampled']:,} training histories in total. "
            "The audit regenerates these failed banks with their original seeds and confirms they contain no jumps. "
            "Their sampling cost remains in the search totals."
        ),
        "",
        (
            f"Validation sampled {sums['validation_sampled']:,} independent histories and retained "
            f"{sums['validation_retained']:,} representatives. All "
            f"{sums['validation_nonempty_banks_with_jumps']} nonempty validation banks contained jumps. "
            "No bank was replaced or conditioned on containing a jump. "
            "Final density matrices integrate all modeled outcomes."
        ),
        "",
        (
            "The audit checks every artifact checksum, training multiplicity and jump count, "
            "regenerates and re-evaluates every raw validation bank, replays every stopping decision, "
            "and independently replays no-jump and jump paths from selected candidates through Qiskit."
        ),
        "",
        markdown_table(
            ["Check", "Maximum absolute error"],
            [
                [key, f"{max(audit[key] for audit in audits):.3e}"]
                for key in (
                    "max_validation_replay_error",
                    "max_qiskit_path_mean_error",
                    "max_vector_matrix_error",
                    "max_trace_error",
                    "max_hermiticity_error",
                    "max_compiled_unitary_error",
                )
            ],
        ),
        "",
        "## Every paired endpoint",
        "",
        (
            "All final fidelities below are exact sampler-law expectations. Validation errors and full phase timings "
            "are in `searches.json`; failed intermediate candidates "
            "remain in their original checkpoints and cost totals."
        ),
        "",
        markdown_table(
            [
                "Target",
                "Seed",
                "Profile",
                "Prune",
                "Old F %",
                "New F %",
                "Δ pp",
                "Old depth",
                "Selected / stopped",
                "CPU ratio",
            ],
            [
                [
                    LABELS[row["target_id"]],
                    str(row["seed"]),
                    row["hardware"],
                    row["variant"],
                    f"{100 * row['baseline_fidelity']:.3f}",
                    f"{100 * row['selected']['fidelity']:.3f}" if row["selected"] else "FAILED",
                    f"{row['gain_pp']:+.3f}" if row["selected"] else "—",
                    str(row["baseline_depth"]),
                    f"{row['selected']['depth'] if row['selected'] else '—'} / {row['stop_depth']}",
                    format(
                        row["search_cost"]["total_search_cpu_seconds"]
                        / row["baseline_cost"]["total_search_cpu_seconds"],
                        ".2f",
                    )
                    + "x",
                ]
                for row in searches
            ],
        ),
        "",
        "## Limits",
        "",
        (
            "Two optimizer seeds per fixed target are a descriptive comparison. "
            "The target fixture has already informed "
            "method development, so this is not an untouched target generalization benchmark. "
            "The larger search has a larger optimization/validation budget; there is no compute-matched control here. "
            "Three-step patience may miss later improvements. "
            "Fixed-rate batch updates and D's repair can lower fidelity, "
            "and simultaneous B pruning has no cumulative fidelity guard. Those policies were deliberately retained. "
            "An additional stabilization or training-budget study would be a separate intervention."
        ),
        "",
    ])
    wall = json.loads((output / "wall_time.json").read_text())
    text.extend([
        (
            f"Recorded parallel suite wall time: {wall['seconds_this_invocation']:.2f} seconds. "
            "This includes baseline replay and retrospective evaluation, and excludes the separate audit/report run."
        ),
        "",
    ])
    if (output / "recovery.json").exists():
        recovery = json.loads((output / "recovery.json").read_text())
        text.extend([
            "### Checkpoint serialization recovery",
            "",
            (
                "The initial invocation stopped when NumPy's Boolean type could not be serialized "
                "in the selection record. The only driver correction converts that value to a Python Boolean. "
                "Source comparison verifies no numerical or protocol change. "
                f"Recovered {len(recovery['reused_files'])} authenticated checkpoint/bank files; "
                "saved training and validation work was reused without drawing replacement banks. "
                "`recovery.json` records the original manifest and every reused artifact hash."
            ),
            "",
            (
                f"The initial invocation used approximately {recovery['prior_active_wall_seconds_approx']:.2f} "
                "additional wall seconds, estimated from execution/log timestamps. "
                f"Combined active execution time was approximately "
                f"{recovery['prior_active_wall_seconds_approx'] + wall['seconds_this_invocation']:.2f} seconds. "
                "This excludes debugging time. Per-search CPU totals already include the reused stages once."
            ),
            "",
        ])
    return "\n".join(text)


def main() -> None:
    """Audit the completed frozen experiment, export machine-readable facts and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=parse_workers, default=4)
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text())
    training_paths = manifest["protocol"]["fine_tuning"]["num_trajectories"]
    assert manifest["protocol"] == protocol_for(training_paths)
    assert all(file_hash(REPOSITORY / name) == digest for name, digest in manifest["sources"].items())
    cases = json.loads((args.output / "results.json").read_text())
    assert len(cases) == 18
    jobs = [
        (str(args.output), case["target_id"] + "__seed" + str(case["optimizer_seed"]), training_paths) for case in cases
    ]
    audits = dict(completed_jobs(audit_case, jobs, workers=args.workers))
    ordered = [audits[index] for index in sorted(audits)]
    searches, candidates = collect_rows(args.output, cases)
    groups = aggregate(searches)
    for name, data in (("audit", ordered), ("searches", searches), ("candidates", candidates), ("summary", groups)):
        (args.output / (name + ".json")).write_text(
            json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    figures(args.output, candidates, groups)
    (args.output / "REPORT.md").write_text(report(args.output, searches, groups, ordered), encoding="utf-8")
    (args.output / "analysis_source.py.txt").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
