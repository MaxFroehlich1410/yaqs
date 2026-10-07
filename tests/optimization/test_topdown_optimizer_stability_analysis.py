# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Controlled optimizer reporting preserves pairing, failures and real costs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from experiments import topdown_optimizer_stability_analysis as analysis

if TYPE_CHECKING:
    from pathlib import Path


def row(policy: str, seed: int, fidelity: float | None, cpu: float) -> dict:
    """Build one endpoint with an explicit failed-candidate cost.

    Returns:
        A synthetic pipeline endpoint with complete accounting fields.
    """
    return {
        "policy": policy,
        "target_id": "target",
        "seed": seed,
        "variant": "B",
        "hardware": "IBM",
        "fidelity": fidelity,
        "resources": dict.fromkeys(analysis.RESOURCE_KEYS, 10) if fidelity is not None else None,
        "selected_depth": 4 if fidelity is not None else None,
        "stop_depth": 7,
        "stop_reason": "depth_cap" if seed == 400 else "patience",
        "cost": {**dict.fromkeys(analysis.COST_KEYS, cpu), "failed_candidates": 1},
    }


def test_pairing_preserves_negative_outcomes_and_summed_cpu_ratio() -> None:
    """One large improvement cannot conceal one regression or unequal denominators."""
    rows = [
        row("original_final", 300, 0.8, 2),
        row("original_final", 400, 0.95, 4),
        row("best_only", 300, 0.9, 8),
        row("best_only", 400, 0.85, 20),
    ]
    result = analysis.paired_summary(
        rows, "best_only", "original_final", "B", "IBM", [("target", 300), ("target", 400)]
    )
    assert result["complete_pairs"] == result["planned_pairs"] == 2
    assert result["gain_pp"] == pytest.approx(0)
    assert result["mean_percent"] == pytest.approx(87.5)
    assert result["wins"] == result["losses"] == 1
    assert result["cost_ratios"]["total_search_cpu_seconds"] == pytest.approx(28 / 6)
    assert not result["missing_pairs"]


def test_missing_and_failed_pairs_are_distinct_and_failed_costs_are_retained() -> None:
    """Only valid pairs enter gains; every attempted search contributes to total costs."""
    rows = [row("original_final", 300, 0.8, 2), row("original_final", 400, 0.9, 4), row("best_only", 300, None, 8)]
    paired = analysis.paired_summary(
        rows, "best_only", "original_final", "B", "IBM", [("target", 300), ("target", 400)]
    )
    assert paired["complete_pairs"] == 0
    assert paired["mean_percent"] is paired["gain_pp"] is None
    assert [item["policy_status"] for item in paired["missing_pairs"]] == ["failed", "absent"]
    assert paired["cost_ratios"]["total_search_cpu_seconds"] is None
    totals = analysis.policy_summary(rows, "best_only", "B", "IBM")
    assert totals["recorded_searches"] == 1
    assert totals["complete_searches"] == 0
    assert totals["cost_totals"]["total_search_cpu_seconds"] == 8
    assert totals["failed_candidates"] == 1
    assert totals["resources"]["compiled_two_qubit"] is None


def test_win_threshold_is_in_fidelity_units() -> None:
    """A sub-tolerance change remains a tie after conversion to percentage points."""
    rows = [row("original_final", 300, 0.8, 1), row("best_only", 300, 0.8 + 5e-11, 1)]
    paired = analysis.paired_summary(rows, "best_only", "original_final", "B", "IBM", [("target", 300)])
    assert paired["ties"] == 1
    assert paired["wins"] == paired["losses"] == 0


def test_flattening_rejects_duplicate_search_identities() -> None:
    """Repeated rows cannot silently overstate the number of independent pairs."""
    result = {
        "variant": "B",
        "hardware": "IBM",
        "selected": None,
        "selected_depth": None,
        "decisions": [],
        "stop_reason": "patience",
        "search_cost": {},
    }
    case = {"target_id": "target", "optimizer_seed": 300, "policy": "original_final", "results": [result]}
    with pytest.raises(ValueError, match="Duplicate search"):
        analysis.search_rows([case, case])


def test_best_retention_does_not_claim_raw_trajectory_stability() -> None:
    """Identical collapsing traces differ only in their safely returned checkpoint."""
    base = {
        "target_id": "target",
        "seed": 400,
        "depth": 3,
        "best_fidelity": 0.995,
        "final_fidelity": 0.07,
        "cpu_seconds": 1,
        "objective_evaluations": 201,
        "gradient_evaluations": 100,
        "rejected_trials": 0,
        "failed_searches": 0,
        "original_trace_max_error": 0,
        "original_theta_max_error": 0,
    }
    rows = [
        {**base, "policy": "original_final", "returned_fidelity": 0.07},
        {**base, "policy": "best_only", "returned_fidelity": 0.995},
    ]
    original, retained, _ = analysis.fixed_summary(rows)
    assert original["raw_peak_to_final_drops_over_10pp"] == retained["raw_peak_to_final_drops_over_10pp"] == 1
    assert original["returned_peak_losses_over_10pp"] == 1
    assert retained["returned_peak_losses_over_10pp"] == 0
    assert retained["mean_returned_gain_pp"] == pytest.approx(92.5)
    assert retained["objective_evaluations"] == original["objective_evaluations"]


def test_fixed_records_reject_duplicate_policy_depth_pairs() -> None:
    """A repeated fixed-start record cannot hide a missing depth/policy combination."""
    case = {"target_id": "target", "optimizer_seed": 300, "records": [{"depth": 1, "policy": "best_only"}]}
    with pytest.raises(ValueError, match="Duplicate fixed-start"):
        analysis.fixed_rows([case, case])


def test_report_keeps_markdown_tables_and_labels_incomplete_target_means() -> None:
    """An incomplete report stays readable and never averages one seed as two."""
    data = {"fixed": [], "fixed_rows": [], "paired": [], "policies": [], "rows": [], "audits": []}
    rendered = analysis.report(data)
    assert "| Target |" in rendered
    assert "missing (0/2)" in rendered
    assert "|\n " not in rendered
    assert "two optimizer seeds" in rendered


def test_trace_extraction_keeps_initial_state_and_best_iterate(tmp_path: Path) -> None:
    """The retained marker points to the actual common raw peak, not the final update."""
    assert not analysis.construction_trace_rows(tmp_path)
    analysis.trace_figure(tmp_path, [])
    assert not (tmp_path / "stability_traces.png").exists()
    for policy in analysis.POLICIES:
        folder = tmp_path / "fixed" / "tfim_ferro__seed400" / policy
        folder.mkdir(parents=True)
        training = {
            "initial_fidelity": 0.5,
            "fidelity_trace": [0.9, 0.99, 0.1],
            "best_iteration": 2,
            "returned_fidelity": 0.99 if policy == "best_only" else 0.1,
            "online_steps": 1,
        }
        if policy == "best_backtracking":
            training["fidelity_trace"] = [0.9, 0.99, 0.999]
        (folder / "construction_03.json").write_text(json.dumps({"training": training}))
    extracted = analysis.construction_trace_rows(tmp_path)
    assert len(extracted) == 1  # The unavailable Haar example is skipped.
    assert extracted[0]["original"] == [0.5, 0.9, 0.99, 0.1]
    assert extracted[0]["backtracking"] == [0.5, 0.9, 0.99, 0.999]
    assert extracted[0]["retained_iteration"] == 2
    assert extracted[0]["retained_fidelity"] == extracted[0]["original"][2]
    analysis.trace_figure(tmp_path, extracted)
    assert (tmp_path / "stability_traces.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "stability_traces.pdf").read_bytes().startswith(b"%PDF")
