# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independently audit and report the frozen four-setting pruning experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Statevector

from benchmarks.state_preparation.execution import completed_jobs, parse_workers
from benchmarks.state_preparation.pilot_components import delete_gates
from experiments.legacy_pruning_ablation import HARDWARE, PROTOCOL, VARIANTS
from experiments.legacy_topdown_dense import PRIMITIVES
from experiments.legacy_topdown_q6 import REPOSITORY, Checkpoints, write_json
from experiments.legacy_topdown_weighted import file_hash

if TYPE_CHECKING:
    from mqt.yaqs.optimization import ParameterizedCircuit

LABELS = {"A": "Current saliency", "B": "Exact, simultaneous", "C": "Exact, sequential", "D": "Sequential + repair"}


def read_compiled_qasm(path: Path) -> QuantumCircuit:
    """Read Qiskit's exported QASM with its legacy qelib1 instruction definitions."""
    return qasm2.loads(path.read_text(encoding="utf-8"), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def qiskit_fidelity(
    circuit: ParameterizedCircuit, theta: np.ndarray, target: np.ndarray, skip: int | None = None
) -> float:
    """Independently simulate a circuit, optionally physically omitting one gate."""
    quantum = QuantumCircuit(circuit.num_qubits)
    for index, gate in enumerate(circuit.gates):
        if index == skip:
            continue
        angle = gate.angle_offset + (gate.angle_scale * theta[gate.param_index] if gate.param_index is not None else 0)
        quantum.append(PRIMITIVES[gate.name](angle), list(gate.sites))
    state = Statevector.from_instruction(quantum).data
    return float(abs(np.vdot(target, state)) ** 2)


def audit_target(job: tuple[str, str]) -> dict:
    """Audit every recorded deletion, endpoint checksum and training noise bank."""
    output, folder = job
    root = Path(output)
    manifest = json.loads((root / "manifest.json").read_text())
    case_root = root / folder
    completed = json.loads((case_root / "complete.json").read_text())
    assert all(file_hash(case_root / name) == digest for name, digest in completed["artifacts"].items())
    target = np.load(case_root / "target.npy")
    source = Checkpoints(Path(manifest["source_root"]) / folder, target, resume=True)
    store = Checkpoints(case_root, target, resume=True)
    original, parameters, _ = source.load("unpruned_initial")
    exact_before = qiskit_fidelity(original, parameters, target)
    first_scores = store.load("B_initial")[2]["training"]["iterations"][0]["scores"]
    errors = [
        abs(exact_before - qiskit_fidelity(original, parameters, target, i) - score)
        for i, score in enumerate(first_scores)
    ]
    selected_checks = 0
    for variant in VARIANTS:
        circuit, theta = original, parameters.copy()
        origins = list(range(len(circuit.gates)))
        _, expected, saved = store.load(variant + "_initial")
        trace = saved["training"]
        repair_updates = 0
        for entry in trace["iterations"]:
            assert entry["origins"] == origins
            scores = np.array(entry["scores"])
            eligible = np.flatnonzero(scores < PROTOCOL["cutoff"])
            selected = eligible.tolist() if variant in {"A", "B"} else [int(np.argmin(scores))] if len(eligible) else []
            assert entry["removed_origins"] == [origins[index] for index in selected]
            before = qiskit_fidelity(circuit, theta, target)
            errors.append(abs(before - entry["fidelity_before"]))
            if not selected:
                continue
            if variant in {"C", "D"}:
                after = qiskit_fidelity(circuit, theta, target, selected[0])
                errors.append(abs(before - after - scores[selected[0]]))
                selected_checks += 1
            circuit, theta = delete_gates(circuit, theta, selected)
            origins = [origin for index, origin in enumerate(origins) if index not in set(selected)]
            errors.append(abs(qiskit_fidelity(circuit, theta, target) - entry["fidelity_after_deletion"]))
            if variant == "D" and circuit.gates:
                repair = entry["repair"]
                assert repair["steps"] == 10
                assert repair["online_steps"] == 5
                assert repair["rate"] == PROTOCOL["repair"]["rate"]
                assert not repair["noisy"]
                repair_updates += repair["steps"]
            else:
                np.testing.assert_array_equal(theta, entry["theta_after_repair"])
            theta = np.asarray(entry["theta_after_repair"])
            errors.append(abs(qiskit_fidelity(circuit, theta, target) - entry["fidelity_after_repair"]))
        np.testing.assert_array_equal(theta, expected)
        assert trace["repair_updates"] == repair_updates
        assert origins == trace["remaining_origins"]
        assert len(circuit.gates) == trace["final_gates"]
    assert max(errors) < 1e-10
    matrix_errors, trace_errors, hermiticity, eigenvalues, compile_errors = [], [], [], [], []
    banks, jump_banks, sampled, retained, empty, failures, checkpoints = 0, 0, 0, 0, 0, 0, 0
    for path in case_root.glob("*.sha256.json"):
        _, _, record = store.load(path.name.removesuffix(".sha256.json"))
        checkpoints += 1
        matrix_errors.append(record["vector_matrix_fidelity_error"])
        for score in [record["noiseless"], *[s for hardware in record["noise"].values() for s in hardware.values()]]:
            trace_errors.append(score["trace_error"])
            hermiticity.append(score["hermiticity_error"])
            eigenvalues.append(score["minimum_eigenvalue"])
    for result in completed["results"]:
        name = result["variant"] + "_" + result["hardware"]
        resources = result["resources"]
        quantum = read_compiled_qasm(case_root / (name + ".qasm"))
        assert quantum.count_ops().get("cx", 0) == resources["compiled_two_qubit"]
        assert quantum.depth(lambda item: item.operation.num_qubits == 2) == resources["compiled_two_qubit_depth"]
        compile_errors.append(resources["logical_unitary_max_error_up_to_global_phase"])
        if result["status"] != "complete":
            failure = json.loads((case_root / (name + ".failure.json")).read_text())
            assert failure["sampled"] == 100
            assert failure["jump_paths"] == 0
            failures += 1
            continue
        _, _, record = store.load(name)
        bank = record["training"]["noise_banks"][0]
        events = bank["events"]
        counts = [len(row) for row in events]
        assert bank["raw_sha256"] == hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest()
        assert counts == bank["raw_jump_counts"]
        assert counts.count(0) == bank["no_jump_paths"]
        assert sum(n > 0 for n in counts) == bank["jump_paths"]
        assert len(events) == bank["sampled_paths"] == 100
        assert sum(bank["multiplicities"]) == 100
        np.testing.assert_allclose(bank["weights"], np.array(bank["multiplicities"]) / 100, atol=1e-15)
        assert sorted(bank["multiplicities"]) == sorted(
            [1] * bank["jump_paths"] + ([bank["no_jump_paths"]] if bank["no_jump_paths"] else [])
        )
        if result["gates"]:
            assert bank["jump_paths"] > 0
            jump_banks += 1
        else:
            empty += 1
        for row in events:
            for event in row:
                assert 0 <= event["gate"] < result["gates"]
                for operator in event["operators"]:
                    matrix = np.array(operator["real"]) + 1j * np.array(operator["imag"])
                    np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(len(matrix)), atol=1e-12)
        banks += 1
        sampled += bank["sampled_paths"]
        retained += bank["retained_paths"]
    return {
        "case": folder,
        "initial_candidates_checked_with_qiskit": len(first_scores),
        "sequential_deletions_checked_with_qiskit": selected_checks,
        "maximum_deletion_or_repair_fidelity_error": max(errors),
        "checkpoints": checkpoints,
        "maximum_vector_matrix_fidelity_error": max(matrix_errors),
        "maximum_trace_error": max(trace_errors),
        "maximum_hermiticity_error": max(hermiticity),
        "minimum_eigenvalue": min(eigenvalues),
        "maximum_compiled_unitary_error": max(compile_errors),
        "completed_banks": banks,
        "nonempty_banks_containing_jumps": jump_banks,
        "empty_circuit_conditions": empty,
        "coverage_failures": failures,
        "sampled_histories": sampled,
        "retained_histories": retained,
    }


def flatten(cases: list[dict]) -> list[dict]:
    """Attach common target, seed, pruning and cost fields to every condition."""
    rows = []
    for case in cases:
        for result in case["results"]:
            pruning = case["pruning"][result["variant"]]
            rows.append({
                **result,
                "target_id": case["target_id"],
                "optimizer_seed": case["optimizer_seed"],
                "initial_gates": case["initial_gates"],
                "pruning_cpu_seconds": pruning["cpu_seconds"],
                "score_calls": pruning["score_calls"],
                "scored_candidates": pruning["scored_candidates"],
                "repair_updates": pruning["repair_updates"],
                "repair_parameter_updates": pruning["repair_parameter_updates"],
                "optimization_cpu_seconds": pruning["cpu_seconds"] + result["fine_tuning_cpu_seconds"]
                if result["status"] == "complete"
                else None,
            })
    return rows


def paired(rows: list[dict], hardware: str, before: str, after: str) -> dict:
    """Compare only identical completed target/seed cases, without imputing failures."""
    first = {
        (r["target_id"], r["optimizer_seed"]): r
        for r in rows
        if r["hardware"] == hardware and r["variant"] == before and r["status"] == "complete"
    }
    second = {
        (r["target_id"], r["optimizer_seed"]): r
        for r in rows
        if r["hardware"] == hardware and r["variant"] == after and r["status"] == "complete"
    }
    keys = sorted(first.keys() & second.keys())
    deltas = [100 * (second[k]["fidelity"] - first[k]["fidelity"]) for k in keys]
    return {
        "hardware": hardware,
        "before": before,
        "after": after,
        "pairs": len(keys),
        "mean_before_percent": float(np.mean([100 * first[k]["fidelity"] for k in keys])) if keys else None,
        "mean_after_percent": float(np.mean([100 * second[k]["fidelity"] for k in keys])) if keys else None,
        "mean_delta_pp": float(np.mean(deltas)) if keys else None,
        "median_delta_pp": float(np.median(deltas)) if keys else None,
        "wins": sum(d > 1e-8 for d in deltas),
        "ties": sum(abs(d) <= 1e-8 for d in deltas),
        "losses": sum(d < -1e-8 for d in deltas),
        "cases": [{"target_id": k[0], "seed": k[1], "delta_pp": delta} for k, delta in zip(keys, deltas, strict=True)],
    }


def summarize(rows: list[dict]) -> dict:
    """Aggregate descriptive means and matched comparisons with explicit denominators."""
    groups = []
    for hardware in HARDWARE:
        for variant in VARIANTS:
            subset = [r for r in rows if r["hardware"] == hardware and r["variant"] == variant]
            complete = [r for r in subset if r["status"] == "complete"]
            groups.append({
                "hardware": hardware,
                "variant": variant,
                "planned": len(subset),
                "complete": len(complete),
                "coverage_failures": len(subset) - len(complete),
                "empty": sum(r["gates"] == 0 for r in subset),
                "mean_fidelity_percent": float(np.mean([100 * r["fidelity"] for r in complete])) if complete else None,
                "median_fidelity_percent": float(np.median([100 * r["fidelity"] for r in complete]))
                if complete
                else None,
                "at_least_90_percent": sum(r["fidelity"] >= 0.9 for r in complete),
                "mean_noiseless_after_pruning_percent": float(
                    np.mean([100 * r["fidelity_after_pruning"] for r in subset])
                ),
                "mean_gates": float(np.mean([r["gates"] for r in subset])),
                "mean_logical_1q": float(np.mean([r["resources"]["logical_one_qubit"] for r in subset])),
                "mean_logical_2q": float(np.mean([r["resources"]["logical_two_qubit"] for r in subset])),
                "mean_compiled_2q": float(np.mean([r["resources"]["compiled_two_qubit"] for r in subset])),
                "mean_compiled_2q_depth": float(np.mean([r["resources"]["compiled_two_qubit_depth"] for r in subset])),
                "mean_pruning_cpu_seconds": float(np.mean([r["pruning_cpu_seconds"] for r in subset])),
                "mean_repair_updates": float(np.mean([r["repair_updates"] for r in subset])),
                "mean_fine_tuning_cpu_seconds_completed": float(
                    np.mean([r["fine_tuning_cpu_seconds"] for r in complete])
                )
                if complete
                else None,
                "mean_optimization_cpu_seconds_completed": float(
                    np.mean([r["optimization_cpu_seconds"] for r in complete])
                )
                if complete
                else None,
            })
    return {
        "groups": groups,
        "paired": [
            paired(rows, hardware, before, after)
            for hardware in HARDWARE
            for before, after in (("A", "B"), ("B", "C"), ("C", "D"), ("B", "D"), ("A", "C"), ("A", "D"))
        ],
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format a simple complete Markdown table."""
    return "\n".join("| " + " | ".join(row) + " |" for row in [headers, ["---"] * len(headers), *rows])


def write_report(root: Path, cases: list[dict], rows: list[dict], summary: dict) -> None:
    """Write per-condition, per-target and paired descriptive comparison tables."""
    body = [
        "# Four-setting q6 pruning comparison",
        "",
        (
            "All nine fixture targets, optimizer seeds 300/400, unchanged 100-map fine-tuning. Final "
            "fidelity is exact density-matrix fidelity under the historical logical sampler law."
        ),
        "",
        "## Setting-level results",
        "",
        (
            "Means use each setting's completed subset. Missing coverage trials have no final fidelity; "
            "empty circuits remain included. Use the paired table below for comparisons on identical "
            "cases."
        ),
        "",
    ]
    body.append(
        markdown_table(
            ["Noise", "Setting", "Complete / 18", "Empty", "Mean F (%)", "Median F (%)", "F >= 90%"],
            [
                [
                    g["hardware"],
                    g["variant"],
                    f"{g['complete']} / {g['planned']}",
                    str(g["empty"]),
                    f"{g['mean_fidelity_percent']:.4f}",
                    f"{g['median_fidelity_percent']:.4f}",
                    str(g["at_least_90_percent"]),
                ]
                for g in summary["groups"]
            ],
        )
    )
    body.extend([
        "",
        "## Paired changes",
        "",
        (
            "Positive changes favor the later setting. Ties use a 1e-8 percentage-point tolerance. Two "
            "optimizer seeds are a small development sample; these are not population confidence "
            "intervals."
        ),
        "",
        markdown_table(
            [
                "Noise",
                "Comparison",
                "Pairs",
                "Before (%)",
                "After (%)",
                "Mean change (pp)",
                "Median change (pp)",
                "Win / tie / loss",
            ],
            [
                [
                    g["hardware"],
                    g["before"] + " → " + g["after"],
                    str(g["pairs"]),
                    f"{g['mean_before_percent']:.4f}",
                    f"{g['mean_after_percent']:.4f}",
                    f"{g['mean_delta_pp']:+.4f}",
                    f"{g['median_delta_pp']:+.4f}",
                    f"{g['wins']} / {g['ties']} / {g['losses']}",
                ]
                for g in summary["paired"]
            ],
        ),
    ])
    for hardware in HARDWARE:
        table = []
        for case in cases[::2]:
            target = case["target_id"]
            values = []
            for variant in VARIANTS:
                selection = [
                    r
                    for r in rows
                    if r["target_id"] == target
                    and r["hardware"] == hardware
                    and r["variant"] == variant
                    and r["status"] == "complete"
                ]
                values.append(
                    f"{100 * np.mean([r['fidelity'] for r in selection]):.4f} ({len(selection)}/2)"
                    if selection
                    else "missing (0/2)"
                )
            table.append([target, *values])
        body.extend([
            "",
            "## " + hardware + " per-target means",
            "",
            (
                "Parentheses give completed optimizer seeds. Compare differing denominators using the "
                "full condition table."
            ),
            "",
            markdown_table(["Target", "A (%)", "B (%)", "C (%)", "D (%)"], table),
        ])
    body.extend([
        "",
        "## Circuit resources",
        "",
        (
            "Means cover all 18 cases, including empty circuits and the post-pruning circuit for failed "
            "fine-tuning trials. Compilation uses one fixed CX reference on a line (level 1); these are "
            "not native Heron/Willow counts. Noisy fidelity was scored on the logical circuit."
        ),
        "",
        markdown_table(
            [
                "Noise",
                "Setting",
                "Logical gates",
                "Logical 1q",
                "Logical 2q",
                "Compiled CX",
                "Compiled 2q depth",
                "Noiseless F after pruning (%)",
            ],
            [
                [
                    g["hardware"],
                    g["variant"],
                    f"{g['mean_gates']:.2f}",
                    f"{g['mean_logical_1q']:.2f}",
                    f"{g['mean_logical_2q']:.2f}",
                    f"{g['mean_compiled_2q']:.2f}",
                    f"{g['mean_compiled_2q_depth']:.2f}",
                    f"{g['mean_noiseless_after_pruning_percent']:.4f}",
                ]
                for g in summary["groups"]
            ],
        ),
        "",
        "## Optimization cost",
        "",
        (
            "CPU seconds exclude shared construction, checkpoint scoring and compilation. Pruning "
            "averages all cases; fine-tuning and total averages use completed cases. D has additional "
            "repair work and is not compute-matched to C."
        ),
        "",
        markdown_table(
            ["Noise", "Setting", "Pruning CPU s", "Repair updates", "Fine-tuning CPU s", "Total CPU s"],
            [
                [
                    g["hardware"],
                    g["variant"],
                    f"{g['mean_pruning_cpu_seconds']:.3f}",
                    f"{g['mean_repair_updates']:.1f}",
                    f"{g['mean_fine_tuning_cpu_seconds_completed']:.3f}",
                    f"{g['mean_optimization_cpu_seconds_completed']:.3f}",
                ]
                for g in summary["groups"]
            ],
        ),
        "",
        "## Audit and artifacts",
        "",
        (
            "See [the numerical audit](validation/audit.json), [all conditions](all_conditions.md), "
            "[machine-readable rows](conditions.json), [paired summary](summary.json), and [frozen "
            "protocol](../../experiments/legacy_pruning_ablation_protocol.md). Each target directory "
            "contains checkpoints, candidate scores, repair traces, training jump histories and compiled "
            "QASM."
        ),
        "",
        "![Fidelity and circuit size](comparison.png)",
    ])
    (root / "comparison.md").write_text("\n".join(body) + "\n", encoding="utf-8")
    table = []
    for row in rows:
        resources = row["resources"]
        table.append([
            row["target_id"],
            str(row["optimizer_seed"]),
            row["hardware"],
            row["variant"],
            row["status"],
            f"{100 * row['fidelity']:.6f}" if row["fidelity"] is not None else "missing",
            str(row["gates"]),
            str(resources["logical_two_qubit"]),
            str(resources["compiled_two_qubit"]),
            str(resources["compiled_two_qubit_depth"]),
            str(row["jump_paths"]),
            str(row["retained_paths"]),
            str(row["repair_updates"]),
            f"{row['optimization_cpu_seconds']:.4f}" if row["optimization_cpu_seconds"] is not None else "missing",
        ])
    (root / "all_conditions.md").write_text(
        "# All 144 conditions\n\n"
        + markdown_table(
            [
                "Target",
                "Seed",
                "Noise",
                "Setting",
                "Status",
                "F (%)",
                "Gates",
                "Logical 2q",
                "CX",
                "2q depth",
                "Jump paths",
                "Retained paths",
                "Repair updates",
                "CPU s",
            ],
            table,
        )
        + "\n",
        encoding="utf-8",
    )


def plot_results(root: Path, rows: list[dict]) -> None:
    """Plot every completed condition without implying seed-level confidence intervals."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    colors = {"A": "#797979", "B": "#db9222", "C": "#2575b8", "D": "#13806d"}
    for axis, hardware in zip(axes, HARDWARE, strict=True):
        for variant in VARIANTS:
            complete = [
                r for r in rows if r["hardware"] == hardware and r["variant"] == variant and r["status"] == "complete"
            ]
            axis.scatter(
                [r["resources"]["compiled_two_qubit"] for r in complete],
                [100 * r["fidelity"] for r in complete],
                color=colors[variant],
                alpha=0.65,
                s=32,
                label=f"{variant}: {LABELS[variant]} ({len(complete)}/18)",
            )
        axis.set_title(hardware + " logical noise profile")
        axis.set_xlabel("Compiled two-qubit gates (common CX reference)")
        axis.set_ylim(-2, 102)
        axis.grid(alpha=0.18)
        axis.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Exact final noisy fidelity (%)")
    fig.suptitle("Six-qubit pruning comparison · 100 weighted training histories")
    fig.savefig(root / "comparison.png", dpi=180)
    fig.savefig(root / "comparison.pdf")
    plt.close(fig)


def main() -> None:
    """Audit all completed jobs and write reproducible comparison artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=parse_workers, default=4)
    args = parser.parse_args()
    root = args.output
    manifest = json.loads((root / "manifest.json").read_text())
    assert all(file_hash(REPOSITORY / name) == digest for name, digest in manifest["sources"].items())
    assert all(file_hash(Path(manifest["source_root"]) / name) == digest for name, digest in manifest["inputs"].items())
    cases = json.loads((root / "results.json").read_text())
    assert len(cases) == 18
    assert sum(len(c["results"]) for c in cases) == 144
    validation = root / "validation"
    validation.mkdir(exist_ok=True)
    jobs = [(str(root.resolve()), c["target_id"] + "__seed" + str(c["optimizer_seed"])) for c in cases]
    audits = {}
    for index, audit in completed_jobs(audit_target, jobs, workers=args.workers):
        audits[index] = audit
        write_json(
            validation / "audit.json",
            {"sources_unchanged": True, "inputs_unchanged": True, "cases": [audits[i] for i in sorted(audits)]},
        )
        print("AUDITED", audit["case"], flush=True)
    rows = flatten(cases)
    summary = summarize(rows)
    write_json(root / "conditions.json", rows)
    write_json(root / "summary.json", summary)
    write_report(root, cases, rows, summary)
    plot_results(root, rows)
    write_json(
        validation / "analysis_source_sha256.json",
        {str(Path(__file__).relative_to(REPOSITORY)): file_hash(Path(__file__))},
    )
    write_json(
        root / "artifact_sha256.json",
        {
            str(p.relative_to(root)): file_hash(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "artifact_sha256.json"
        },
    )


if __name__ == "__main__":
    main()
