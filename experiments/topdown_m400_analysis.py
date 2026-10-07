# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compare the isolated M=400 rerun with authenticated M=100 depth-search results."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from experiments.legacy_pruning_analysis import markdown_table
from experiments.legacy_topdown_q6 import write_json
from experiments.legacy_topdown_weighted import file_hash
from experiments.topdown_depth_analysis import LABELS, collect_rows
from experiments.topdown_depth_search import HARDWARE, VARIANTS, read_checkpoint


def identity(row: dict) -> tuple:
    """Identify a paired target, optimizer seed, pruning method and noise profile."""
    return tuple(row[key] for key in ("target_id", "seed", "variant", "hardware"))


def pair_rows(old: list[dict], new: list[dict]) -> list[dict]:
    """Pair complete search and control results without substituting missing outcomes."""
    before, after = {identity(row): row for row in old}, {identity(row): row for row in new}
    if len(before) != len(old) or len(after) != len(new) or before.keys() != after.keys():
        msg = "Comparisons require unique, identical experimental identities."
        raise ValueError(msg)
    rows = []
    for key, first in before.items():
        second = after[key]
        for mode in ("depth_search", "single_circuit"):
            selected = mode == "depth_search"
            a, b = (first["selected"], second["selected"]) if selected else (first, second)
            cost_key = "search_cost" if selected else "baseline_cost"
            fidelity_key = "fidelity" if selected else "baseline_fidelity"
            depth_key = "depth" if selected else "baseline_depth"
            resource_key = "resources" if selected else "baseline_resources"
            old_cpu = first[cost_key]["total_search_cpu_seconds"]
            new_cpu = second[cost_key]["total_search_cpu_seconds"]
            rows.append({
                **dict(zip(("target_id", "seed", "variant", "hardware"), key, strict=True)),
                "mode": mode,
                "old_fidelity": a[fidelity_key] if a else None,
                "new_fidelity": b[fidelity_key] if b else None,
                "gain_pp": 100 * (b[fidelity_key] - a[fidelity_key]) if a and b else None,
                "old_cpu": old_cpu,
                "new_cpu": new_cpu,
                "old_depth": a[depth_key] if a else None,
                "new_depth": b[depth_key] if b else None,
                "old_resources": a[resource_key] if a else None,
                "new_resources": b[resource_key] if b else None,
                "old_stop_depth": first["stop_depth"] if selected else first["baseline_depth"],
                "new_stop_depth": second["stop_depth"] if selected else second["baseline_depth"],
            })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Use paired means and ratios of total CPU, retaining failure denominators."""
    groups = []
    resources = ("logical_two_qubit", "compiled_two_qubit", "compiled_two_qubit_depth")
    for mode in ("depth_search", "single_circuit"):
        for hardware in HARDWARE:
            for variant in VARIANTS:
                subset = [r for r in rows if (r["mode"], r["hardware"], r["variant"]) == (mode, hardware, variant)]
                complete = [r for r in subset if r["gain_pp"] is not None]
                if not complete:
                    continue
                gains = np.array([r["gain_pp"] for r in complete])
                groups.append({
                    "mode": mode,
                    "hardware": hardware,
                    "variant": variant,
                    "planned": len(subset),
                    "complete": len(complete),
                    "old_percent": float(100 * np.mean([r["old_fidelity"] for r in complete])),
                    "new_percent": float(100 * np.mean([r["new_fidelity"] for r in complete])),
                    "gain_pp": float(gains.mean()),
                    "median_gain_pp": float(np.median(gains)),
                    "wins": int(np.count_nonzero(gains > 1e-9)),
                    "losses": int(np.count_nonzero(gains < -1e-9)),
                    "ties": int(np.count_nonzero(abs(gains) <= 1e-9)),
                    "old_cpu_mean": float(np.mean([r["old_cpu"] for r in subset])),
                    "new_cpu_mean": float(np.mean([r["new_cpu"] for r in subset])),
                    "cpu_ratio": sum(r["new_cpu"] for r in subset) / sum(r["old_cpu"] for r in subset),
                    "changed_selected_depths": sum(r["old_depth"] != r["new_depth"] for r in complete),
                    "changed_stop_depths": sum(r["old_stop_depth"] != r["new_stop_depth"] for r in subset),
                    **{
                        f"{side}_{key}": float(np.mean([r[f"{side}_resources"][key] for r in complete]))
                        for side in ("old", "new")
                        for key in resources
                    },
                })
    return groups


def audit_comparison(reference: Path, output: Path) -> dict:
    """Verify one changed setting, identical starts, bank prefixes and validation draws."""
    old = json.loads((reference / "manifest.json").read_text())
    new = json.loads((output / "manifest.json").read_text())
    normalized = deepcopy(new["protocol"])
    assert normalized["fine_tuning"]["num_trajectories"] == 400
    normalized["fine_tuning"]["num_trajectories"] = 100
    normalized["pruning"]["fine_tuning"]["num_trajectories"] = 100
    assert normalized == old["protocol"]
    for key in ("target_ids", "optimizer_seeds", "fixture_sha256", "inputs", "python", "packages"):
        assert old[key] == new[key], key
    changed = [name for name, digest in old["sources"].items() if new["sources"].get(name) != digest]
    assert changed == ["experiments/topdown_depth_search.py"]
    for root, manifest in ((reference, old), (output, new)):
        assert all(
            file_hash(root / "source" / (name + ".txt")) == digest for name, digest in manifest["sources"].items()
        )
    counts: dict = {
        "only_training_budget_changed": True,
        "changed_existing_sources": changed,
        "identical_construction_checkpoints": 0,
        "identical_pruned_checkpoints": 0,
        "identical_100_history_prefixes": 0,
        "identical_validation_banks": 0,
        "old_coverage_failures_resolved": 0,
        "baseline_banks_checked": 0,
    }
    for current in output.glob("*/complete.json"):
        old_root, new_root = reference / current.parent.name, current.parent
        for root in (old_root, new_root):
            complete = json.loads((root / "complete.json").read_text())
            assert all(file_hash(root / name) == digest for name, digest in complete["artifacts"].items())
        for stage in ("baseline", "search"):
            for path in (new_root / stage).glob("*.sha256.json"):
                name = path.name.removesuffix(".sha256.json")
                a = read_checkpoint(old_root / stage, name)
                b = read_checkpoint(new_root / stage, name)
                assert b is not None
                if stage == "baseline" and b.get("training") and "hardware" in b:
                    bank = b["training"]["noise_banks"][0]
                    assert bank["sampled_paths"] == sum(bank["multiplicities"]) == 400
                    assert bank["jump_paths"] > 0
                    np.testing.assert_allclose(bank["weights"], np.array(bank["multiplicities"]) / 400)
                    counts["baseline_banks_checked"] += 1
                if a is None:
                    continue
                assert a["circuit"] == b["circuit"]
                if name.startswith("construction_") or name.endswith("_pruned"):
                    np.testing.assert_array_equal(a["theta"], b["theta"])
                    key = (
                        "identical_construction_checkpoints"
                        if name.startswith("construction_")
                        else "identical_pruned_checkpoints"
                    )
                    counts[key] += 1
                    continue
                if a["training"] and b["training"]:
                    assert a["training"]["noise_banks"][0]["events"] == b["training"]["noise_banks"][0]["events"][:100]
                    counts["identical_100_history_prefixes"] += 1
                elif a["status"] == "no_jump_training_bank" and b["training"]:
                    assert not any(b["training"]["noise_banks"][0]["events"][:100])
                    counts["old_coverage_failures_resolved"] += 1
                if a["validation"] and b["validation"]:
                    np.testing.assert_array_equal(
                        np.load(old_root / stage / a["validation"]["bank_file"])["codes"],
                        np.load(new_root / stage / b["validation"]["bank_file"])["codes"],
                    )
                    counts["identical_validation_banks"] += 1
    return counts


def report(rows: list[dict], groups: list[dict], matched: list[dict], audit: dict) -> str:
    """Write final-pipeline and shared-depth comparisons with explicit limitations."""
    text = [
        "# M=400 versus M=100: six-qubit top-down state preparation",
        "",
        (
            "The sole scientific change is 400 instead of 100 raw frozen training histories. "
            "All nine targets, two optimizer seeds, pruning B/D, logical noise profiles, optimizer settings, "
            "validation streams and selection rules remain fixed. Both the optional depth search and "
            "the default first-90% single-circuit pipeline were rerun from noiseless starting circuits."
        ),
        "",
        (
            "Final fidelity is the exact 64-by-64 density-matrix average under the implemented gate-window "
            "Pauli law, evaluated after selection. It includes every modeled jump outcome and has no Monte Carlo "
            "test error. Training uses dense 64-amplitude statevectors, including random-MPS targets."
        ),
        "",
    ]
    for mode, title in (
        ("depth_search", "Optional depth search"),
        ("single_circuit", "Default single-circuit pipeline"),
    ):
        text.extend([
            "## " + title,
            "",
            markdown_table(
                ["Profile", "Pruning", "M=100", "M=400", "Change (pp)", "Wins/losses/ties", "CPU ratio"],
                [
                    [
                        g["hardware"],
                        g["variant"],
                        f"{g['old_percent']:.3f}%",
                        f"{g['new_percent']:.3f}%",
                        f"{g['gain_pp']:+.3f}",
                        f"{g['wins']}/{g['losses']}/{g['ties']}",
                        f"{g['cpu_ratio']:.2f}x",
                    ]
                    for g in groups
                    if g["mode"] == mode
                ],
            ),
            "",
            (
                "Each row averages 18 matched target/seed conditions. CPU includes all visited construction, "
                "pruning/repair, fine-tuning and validation, including discarded candidates; shared work is "
                "charged fully to each standalone pipeline. Exact evaluation and compilation are excluded."
            ),
            "",
        ])
        for hardware in HARDWARE:
            table = []
            for target, label in LABELS.items():
                cells = [label]
                for variant in VARIANTS:
                    subset = [
                        r
                        for r in rows
                        if (r["mode"], r["hardware"], r["variant"], r["target_id"]) == (mode, hardware, variant, target)
                    ]
                    complete = [r for r in subset if r["gain_pp"] is not None]
                    if len(complete) != len(subset):
                        cells.extend(["FAILED", "FAILED", "FAILED"])
                    else:
                        cells.extend([
                            f"{100 * np.mean([r['old_fidelity'] for r in complete]):.2f}%",
                            f"{100 * np.mean([r['new_fidelity'] for r in complete]):.2f}%",
                            f"{np.mean([r['gain_pp'] for r in complete]):+.2f}",
                        ])
                table.append(cells)
            text.extend([
                "### " + hardware,
                "",
                markdown_table(
                    ["Target", "B: M=100", "B: M=400", "B: change pp", "D: M=100", "D: M=400", "D: change pp"],
                    table,
                ),
                "",
            ])
    text.extend([
        "## Same-depth fine-tuning comparison",
        "",
        (
            "These rows use only candidates completed at the same depth in both runs, before final depth "
            "selection. Their constructed/pruned starting parameters and first 100 training histories "
            "are identical. Candidates are correlated within searches; counts are descriptive, not "
            "independent statistical replicates."
        ),
        "",
        markdown_table(
            [
                "Profile",
                "Pruning",
                "Matched candidates",
                "Mean change pp",
                "Median change pp",
                "Fine-tuning CPU ratio",
                "Mean retained: 100 → 400",
            ],
            [
                [
                    hardware,
                    variant,
                    str(len(subset)),
                    f"{np.mean([r['gain_pp'] for r in subset]):+.3f}",
                    f"{np.median([r['gain_pp'] for r in subset]):+.3f}",
                    f"{sum(r['new_cpu'] for r in subset) / sum(r['old_cpu'] for r in subset):.2f}x",
                    (
                        f"{np.mean([r['old_retained'] for r in subset]):.1f} → "
                        f"{np.mean([r['new_retained'] for r in subset]):.1f}"
                    ),
                ]
                for hardware in HARDWARE
                for variant in VARIANTS
                if (subset := [r for r in matched if (r["hardware"], r["variant"]) == (hardware, variant)])
            ],
        ),
        "",
        "## Selected circuit resources and search decisions",
        "",
        markdown_table(
            [
                "Profile",
                "Pruning",
                "Logical 2Q: 100 → 400",
                "Compiled CX: 100 → 400",
                "Compiled 2Q depth: 100 → 400",
                "Changed selected/stop depths",
            ],
            [
                [
                    g["hardware"],
                    g["variant"],
                    *[
                        f"{g['old_' + key]:.2f} → {g['new_' + key]:.2f}"
                        for key in ("logical_two_qubit", "compiled_two_qubit", "compiled_two_qubit_depth")
                    ],
                    f"{g['changed_selected_depths']}/18 selected; {g['changed_stop_depths']}/18 stopped",
                ]
                for g in groups
                if g["mode"] == "depth_search"
            ],
        ),
        "",
        (
            "Compiled CX counts use the unchanged reference compiler; noisy fidelity is evaluated on the "
            "logical gate circuit, not the compiled circuit."
        ),
        "",
        "## Comparison audit",
        "",
        markdown_table(["Check", "Result"], [[key, str(value)] for key, value in audit.items()]),
        "",
        "## Interpretation limits",
        "",
        (
            "More histories improve representation of the training noise distribution, but do not guarantee "
            "better optimization. The implemented cross-trajectory signal is not established as the gradient "
            "of the sampled fidelity. The noiseless optimizer still retains the final iterate and can lose "
            "better intermediate solutions. Neither behavior was changed."
        ),
        "",
        (
            "Two optimizer seeds on previously used targets do not establish statistical significance or "
            "generalization to fresh training banks. Exact final scoring removes test sampling error, "
            "not variability of training. CPU comparisons use the archived M=100 run and the new M=400 run "
            "on the same machine; runtime conditions can differ. Greater sample counts can change selected "
            "depths and total search work even though the stopping rule is unchanged."
        ),
        "",
        "## All paired endpoints",
        "",
        markdown_table(
            ["Mode", "Target", "Seed", "Profile", "Pruning", "M=100 %", "M=400 %", "Change pp", "Depth: 100 → 400"],
            [
                [
                    r["mode"],
                    r["target_id"],
                    str(r["seed"]),
                    r["hardware"],
                    r["variant"],
                    f"{100 * r['old_fidelity']:.4f}" if r["old_fidelity"] is not None else "FAILED",
                    f"{100 * r['new_fidelity']:.4f}" if r["new_fidelity"] is not None else "FAILED",
                    f"{r['gain_pp']:+.4f}" if r["gain_pp"] is not None else "FAILED",
                    f"{r['old_depth']} → {r['new_depth']}",
                ]
                for r in rows
            ],
        ),
        "",
    ])
    return "\n".join(text)


def main() -> None:
    """Authenticate both runs and export all paired comparisons and their audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit = audit_comparison(args.reference, args.output)
    old, old_candidates = collect_rows(args.reference, json.loads((args.reference / "results.json").read_text()))
    new, new_candidates = collect_rows(args.output, json.loads((args.output / "results.json").read_text()))
    rows = pair_rows(old, new)
    groups = summarize(rows)
    old_index = {(*identity(r), r["depth"]): r for r in old_candidates}
    matched = []
    for current in new_candidates:
        previous = old_index.get((*identity(current), current["depth"]))
        if previous is not None:
            matched.append({
                **{key: current[key] for key in ("target_id", "seed", "variant", "hardware", "depth")},
                "gain_pp": 100 * (current["fidelity"] - previous["fidelity"]),
                "old_fidelity": previous["fidelity"],
                "new_fidelity": current["fidelity"],
                "old_cpu": previous["fine_tuning_cpu"],
                "new_cpu": current["fine_tuning_cpu"],
                "old_retained": previous["training_retained"],
                "new_retained": current["training_retained"],
            })
    for name, data in (
        ("comparison_audit", audit),
        ("paired_budgets", rows),
        ("budget_summary", groups),
        ("matched_depths", matched),
    ):
        write_json(args.output / (name + ".json"), data)
    (args.output / "M_COMPARISON.md").write_text(report(rows, groups, matched, audit), encoding="utf-8")
    (args.output / "budget_analysis_source.py.txt").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
