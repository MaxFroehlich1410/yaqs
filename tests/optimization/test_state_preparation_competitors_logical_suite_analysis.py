# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Selection-preserving summaries, paired coverage and scientific cost accounting."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import pytest

from experiments.state_preparation_competitors import logical_suite_analysis as analysis

if TYPE_CHECKING:
    from pathlib import Path


def _group(
    target: str,
    method: str,
    fidelities: tuple[float | None, ...] = (0.9,),
    *,
    selected: int | None = 0,
    cpu: float = 10.0,
    gates: int = 4,
    depth: int = 3,
) -> dict[str, Any]:
    """Build a minimal complete receipt with independently specified test scores.

    Returns:
        A self-contained group using the logical-runner summary schema.
    """
    return {
        "context": {
            "identity": {"target_id": target, "method": method, "seed": 300, "hardware": "IBM"},
            "construction_receipt_sha256": "construction-hash",
            "target_sha256": "target-hash",
        },
        "selected_index": selected,
        "status": "complete" if selected is not None else "no_eligible_candidate",
        "construction_cache": f"construction/{target}/{method}/300/IBM",
        "selection_lock_sha256": "lock-hash",
        "selection_scope": "declared candidate grid",
        "selection_rule": "validation only",
        "training_reused": False,
        "build_train_cpu_seconds": cpu,
        "new_build_train_cpu_seconds": cpu,
        "historical_selection_cpu_seconds": 0.5,
        "construction": {"construction_cpu_seconds": cpu},
        "records": [
            {
                "method": {
                    "method": method,
                    "candidate_id": f"depth_{index + 1}",
                    "status": "complete",
                    "construction_reference": {"path": "input", "seal_sha256": "construction-hash"},
                },
                "logical_resources": {
                    "ideal_fidelity": 0.99,
                    "logical_one_qubit": 8,
                    "logical_two_qubit": gates,
                    "logical_two_qubit_depth": depth,
                    "logical_total_depth": depth + 5,
                    "logical_operations": {"ry": 8, "rxx": gates},
                },
                "compilation": {
                    "compiled_two_qubit": 2 * gates,
                    "compiled_two_qubit_depth": 2 * depth,
                    "compiled_one_qubit": 20,
                    "compiled_total_depth": 2 * depth + 10,
                    "compilation_cpu_seconds": 0.25,
                },
                "validation": {
                    "fidelity": 0.95 - index * 0.01,
                    "standard_error": 0.01,
                    "jump_paths": 20 + index,
                    "sampled_paths": 8192,
                    "cpu_seconds": 1.0,
                },
                "eligible": selected is not None,
                "validation_coverage": "complete",
            }
            for index in range(len(fidelities))
        ],
        "final_scores": [
            {
                "candidate_id": f"depth_{index + 1}",
                "exact": {"fidelity": fidelity},
                "cpu_seconds": 0.5,
            }
            for index, fidelity in enumerate(fidelities)
        ],
    }


def test_validation_selection_survives_higher_exact_alternative() -> None:
    """The TFIM depth-two endpoint survives a better posthoc depth-four score."""
    group = _group("tfim_critical", "robuststate", (0.9216812176816171, 0.9222492685981827))
    group["records"][0]["method"]["candidate_id"] = "depth_2"
    group["records"][1]["method"]["candidate_id"] = "depth_4"
    group["final_scores"][0]["candidate_id"] = "depth_2"
    group["final_scores"][1]["candidate_id"] = "depth_4"
    group["final_scores"].reverse()
    summary = {"groups": [group], "execution_metadata": {"unrelated": True}}
    original = copy.deepcopy(summary)
    result = analysis.build_analysis(summary)
    chosen = result["selected_rows"][0]
    assert chosen["candidate_id"] == "depth_2"
    assert chosen["exact_fidelity"] == pytest.approx(0.9216812176816171)
    assert result["aggregates"]["robuststate"]["mean_exact_fidelity"] == chosen["exact_fidelity"]
    assert result["exact_max_diagnostics"][0] == {
        "target": "tfim_critical",
        "method": "robuststate",
        "selected_candidate_id": "depth_2",
        "diagnostic_exact_max_candidate_id": "depth_4",
        "exact_max_minus_selected_pp": pytest.approx(0.05680509165656478),
        "selection_changed": False,
    }
    assert summary == original


def test_group_costs_not_multiplied_by_candidate_count() -> None:
    """Historical and new group costs are charged once even for unequal grids."""
    reused = _group("tfim_critical", analysis.OURS, (0.9, 0.8, 0.7), cpu=12.0)
    reused.update({"reused_evaluation": True, "source_output": "prior/output"})
    fresh = _group("tfim_ferro", analysis.OURS, (0.8, 0.7), cpu=8.0)
    result = analysis.build_analysis({"groups": [reused, fresh]})
    totals = result["cost_totals"]
    assert totals["build_train_cpu_seconds"] == pytest.approx(20.0)
    assert totals["new_build_train_cpu_seconds"] == pytest.approx(8.0)
    assert totals["reused_build_train_cpu_seconds"] == pytest.approx(12.0)
    assert totals["grid_validation_cpu_seconds"] == pytest.approx(5.0)
    assert totals["current_validation_cpu_seconds"] == pytest.approx(2.0)
    assert totals["current_final_cpu_seconds"] == pytest.approx(1.0)
    assert totals["current_load_or_construction_cpu_seconds"] == pytest.approx(8.0)
    assert totals["construction_and_selection_cpu_seconds"] == pytest.approx(27.25)
    assert result["aggregates"][analysis.OURS]["total_build_train_cpu_seconds"] == pytest.approx(20.0)
    assert result["coverage"]["candidate_count"] == 5
    assert result["coverage"]["new_training_group_count"] == 1
    assert result["coverage"]["reused_evaluation_group_count"] == 1
    assert result["coverage"]["phase_groups"] == {"prior_three_target_pilot": 1, "six_target_extension": 1}
    row = result["candidate_rows"][0]
    assert row["source_output"] == "prior/output"
    assert row["source_references"]["selection_lock_sha256"] == "lock-hash"
    assert row["logical_two_qubit"] == 4
    assert row["cx"] == 8
    assert row["logical_two_qubit_depth"] == 3
    assert row["cx_depth"] == 6
    assert row["validation_jump_paths"] == 20


def test_asymmetric_coverage_and_paired_gaps() -> None:
    """Fontana's real-only mean is compared on common targets, with ties explicit."""
    groups = [
        _group("a", analysis.OURS, (0.9,)),
        _group("b", analysis.OURS, (0.8,)),
        _group("c", analysis.OURS, (0.6,)),
        _group("d", analysis.OURS, (0.7,)),
        _group("a", "fontana", (0.8,)),
        _group("b", "fontana", (0.8 + 1e-14,)),
        _group("c", "fontana", (0.7,)),
        _group("e", "fontana", (0.99,)),
    ]
    result = analysis.build_analysis({
        "groups": groups,
        "exclusions": [{"target": "d", "method": "fontana", "reason": "complex"}],
    })
    paired = result["paired_comparisons"]["fontana"]
    assert paired["common_target_count"] == 3
    assert (paired["wins"], paired["ties"], paired["losses"]) == (1, 1, 1)
    assert paired["mean_gap_pp"] == pytest.approx(0.0, abs=1e-11)
    assert paired["ours_only_targets"] == ["d"]
    assert paired["other_only_targets"] == ["e"]
    assert result["aggregates"]["fontana"]["mean_exact_fidelity"] == pytest.approx(0.8225)
    assert result["coverage"]["target_count"] == 5
    assert result["coverage"]["exclusion_count"] == 1


def test_pareto_control_and_resource_objectives_are_separate() -> None:
    """A faster control may dominate two objectives without dominating resources."""
    groups = [
        _group("a", analysis.OURS, (0.9,), cpu=10.0, gates=2, depth=2),
        _group("a", "robuststate", (0.8,), cpu=11.0, gates=3, depth=3),
        _group("a", "fixed_vqsp", (0.95,), cpu=9.0, gates=10, depth=10),
    ]
    result = analysis.build_analysis({"groups": groups})
    pareto = result["pareto"]["a"]
    assert pareto["external_only"]["fidelity_cpu"]["frontier_methods"] == [analysis.OURS]
    assert pareto["including_controls"]["fidelity_cpu"]["frontier_methods"] == ["fixed_vqsp"]
    resources = pareto["including_controls"]["fidelity_cpu_logical_resources"]
    assert resources["frontier_methods"] == [analysis.OURS, "fixed_vqsp"]
    assert resources["dominated_by"]["robuststate"] == [analysis.OURS]
    assert resources["relative_epsilon"] == pytest.approx(1e-12)


@pytest.mark.parametrize(
    ("fidelity", "cpu", "expected"),
    [(0.9, 10.0, False), (0.9 + 1e-14, 10.0, False), (0.91, 10.0, True), (0.91, 11.0, False)],
)
def test_relative_dominance_requires_strict_improvement(fidelity: float, cpu: float, *, expected: bool) -> None:
    """Numerical ties cannot create spurious domination or undo a real tradeoff."""
    assert (
        analysis.dominates(
            {"exact_fidelity": fidelity, "build_train_cpu_seconds": cpu},
            {"exact_fidelity": 0.9, "build_train_cpu_seconds": 10.0},
        )
        is expected
    )


@pytest.mark.parametrize("unavailable", [None, float("nan"), float("inf"), True, "unknown"])
def test_missing_fidelity_never_becomes_zero(unavailable: object) -> None:
    """Missing diagnostics remain null and cannot enter means or frontiers."""
    group = _group("a", analysis.OURS)
    group["final_scores"][0]["exact"]["fidelity"] = unavailable
    result = analysis.build_analysis({"groups": [group]})
    assert result["selected_rows"][0]["exact_fidelity"] is None
    assert result["aggregates"][analysis.OURS]["mean_exact_fidelity"] is None
    assert result["coverage"]["successful_selected_count"] == 0
    assert result["pareto"]["a"]["external_only"]["fidelity_cpu"]["excluded_methods"] == [analysis.OURS]
    json.dumps(result, allow_nan=False)


def test_failed_and_absent_selection_preserve_missingness() -> None:
    """No eligible endpoint and failed selected records are explicit failures."""
    absent = _group("a", analysis.OURS, (0.2,), selected=None)
    failed = _group("b", "robuststate", (0.3,))
    failed["records"][0]["method"]["status"] = "failed"
    result = analysis.build_analysis({"groups": [absent, failed]})
    assert result["group_costs"][0]["selected_exact_fidelity"] is None
    assert len(result["selected_rows"]) == 1
    assert result["coverage"]["successful_selected_count"] == 0
    assert result["aggregates"][analysis.OURS]["failed_or_missing_selection_targets"] == ["a"]
    assert result["paired_comparisons"]["robuststate"]["common_target_count"] == 0
    assert result["paired_comparisons"]["robuststate"]["mean_gap_pp"] is None


def test_missing_costs_not_undercounted_and_missing_resources_not_ranked() -> None:
    """Incomplete timing ledgers remain null instead of implying free computation."""
    group = _group("a", analysis.OURS)
    del group["records"][0]["validation"]["cpu_seconds"]
    del group["records"][0]["logical_resources"]["logical_two_qubit_depth"]
    result = analysis.build_analysis({"groups": [group]})
    assert result["cost_totals"]["grid_validation_cpu_seconds"] is None
    assert result["cost_totals"]["construction_and_selection_cpu_seconds"] is None
    assert result["pareto"]["a"]["external_only"]["fidelity_cpu_logical_resources"]["excluded_methods"] == [
        analysis.OURS
    ]
    assert not analysis.dominates({"exact_fidelity": 0.9}, {"exact_fidelity": 0.8})


@pytest.mark.parametrize("selected", [-1, 1, True, "0"])
def test_invalid_selection_rejected(selected: object) -> None:
    """Corrupt lock indices fail instead of selecting a different endpoint."""
    group = _group("a", analysis.OURS)
    group["selected_index"] = selected
    with pytest.raises(ValueError, match="selected_index"):
        analysis.build_analysis({"groups": [group]})


@pytest.mark.parametrize(
    "mutation", ["duplicate_group", "duplicate_score", "missing_score", "wrong_method", "duplicate_candidate"]
)
def test_inconsistent_receipt_identities_rejected(mutation: str) -> None:
    """Every exact score must identify exactly one candidate in exactly one group."""
    group = _group("a", analysis.OURS, (0.9, 0.8))
    groups = [group]
    if mutation == "duplicate_group":
        groups.append(copy.deepcopy(group))
    elif mutation == "duplicate_score":
        group["final_scores"].append(copy.deepcopy(group["final_scores"][0]))
    elif mutation == "missing_score":
        group["final_scores"].pop()
    elif mutation == "wrong_method":
        group["records"][0]["method"]["method"] = "fontana"
    else:
        group["records"][1]["method"]["candidate_id"] = "depth_1"
    with pytest.raises(ValueError, match=r"(once|identities|method differs)"):
        analysis.build_analysis({"groups": groups})


def test_full_nine_target_coverage_and_grid_counts() -> None:
    """The full suite distinguishes six Fontana targets from nine general targets."""
    targets = [
        "gaussian_mu0p5_sigma0p1",
        "tfim_ferro",
        "tfim_critical",
        "tfim_para",
        "haar_random_1",
        "haar_random_2",
        "haar_random_3",
        "random_mps_bond2",
        "random_mps_bond3",
    ]
    grids = {analysis.OURS: 1, "nacl": 3, "robuststate": 2, "fontana": 2, "fixed_vqsp": 2, "layerwise": 4}
    groups, exclusions = [], []
    for target in targets:
        for method, count in grids.items():
            if method == "fontana" and target.startswith("haar"):
                exclusions.append({"target": target, "method": method, "reason": "complex"})
                continue
            group = _group(target, method, (0.9,) * count)
            if target in analysis.PRIOR_TARGETS:
                group["reused_evaluation"] = True
            elif method == analysis.OURS:
                group["training_reused"] = True
                group["new_build_train_cpu_seconds"] = 0.0
            groups.append(group)
    result = analysis.build_analysis({"groups": groups, "exclusions": exclusions})
    coverage = result["coverage"]
    assert coverage["group_count"] == coverage["selected_count"] == 51
    assert coverage["candidate_count"] == 120
    assert coverage["new_training_group_count"] == 28
    assert coverage["reused_evaluation_group_count"] == 17
    assert result["paired_comparisons"]["fontana"]["common_target_count"] == 6
    assert result["paired_comparisons"]["robuststate"]["common_target_count"] == 9
    assert result["aggregates"]["fontana"]["selected_count"] == 6
    assert result["aggregates"][analysis.OURS]["selected_count"] == 9
    assert result["cost_totals"]["new_build_train_cpu_seconds"] == pytest.approx(280.0)


def test_legacy_new_training_cost_fallback() -> None:
    """An older summary's reuse flag determines the missing current-training field."""
    fresh = _group("a", analysis.OURS)
    reused = _group("b", analysis.OURS)
    del fresh["new_build_train_cpu_seconds"]
    del reused["new_build_train_cpu_seconds"]
    reused["training_reused"] = True
    result = analysis.build_analysis({"groups": [fresh, reused]})
    assert result["cost_totals"]["new_build_train_cpu_seconds"] == pytest.approx(10.0)
    assert result["cost_totals"]["reused_build_train_cpu_seconds"] == pytest.approx(10.0)


def test_empty_and_exclusion_only_summaries() -> None:
    """An empty or entirely excluded subset has no fabricated aggregate fidelity."""
    empty = analysis.build_analysis({"groups": []})
    assert empty["coverage"]["group_count"] == 0
    assert empty["cost_totals"] == {}
    excluded = analysis.build_analysis({
        "groups": [],
        "exclusions": [{"target": "haar_random_1", "method": "fontana", "reason": "complex"}],
    })
    assert excluded["aggregates"]["fontana"]["selected_count"] == 0
    assert excluded["aggregates"]["fontana"]["median_exact_fidelity"] is None


def test_cli_writes_artifact_without_overwriting_input(tmp_path: Path) -> None:
    """The CLI writes reproducible JSON and refuses to overwrite its source."""
    source = tmp_path / "summary.json"
    source.write_text(json.dumps({"groups": [_group("a", analysis.OURS)]}))
    original = source.read_bytes()
    analysis.main([str(source)])
    result = json.loads((tmp_path / "analysis.json").read_text())
    assert result["schema"] == "yaqs.logical_suite_analysis.v1"
    assert source.read_bytes() == original
    explicit = tmp_path / "nested/result.json"
    analysis.main([str(source), "--output", str(explicit)])
    assert json.loads(explicit.read_text()) == result
    with pytest.raises(ValueError, match="differ"):
        analysis.main([str(source), "--output", str(source)])
