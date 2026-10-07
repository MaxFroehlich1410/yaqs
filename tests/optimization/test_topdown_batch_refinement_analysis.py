# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Batch-refinement reports use the prescribed controls and honest CPU checkpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from experiments import topdown_batch_refinement_analysis as analysis

if TYPE_CHECKING:
    from pathlib import Path


def curve() -> list[dict]:
    """Return a retained-fidelity curve with deliberately uneven completed-step times.

    Returns:
        Three completed checkpoints with explicit objective and gradient work.
    """
    return [
        {
            "iteration": 100 + i,
            "batch_cpu_seconds": time,
            "best_fidelity": fidelity,
            "objective_evaluations": 3 * i,
            "gradient_evaluations": i,
        }
        for i, (time, fidelity) in enumerate(((0.0, 0.8), (2.0, 0.9), (5.0, 0.95)))
    ]


def test_cpu_budget_uses_only_last_completed_checkpoint_without_interpolation() -> None:
    """A budget inside an expensive step cannot receive its future fidelity improvement."""
    points = curve()
    assert analysis.checkpoint_at_budget(points, 0)["best_fidelity"] == pytest.approx(0.8)
    assert analysis.checkpoint_at_budget(points, 1.999)["iteration"] == 100
    assert analysis.checkpoint_at_budget(points, 2)["iteration"] == 101
    held = analysis.checkpoint_at_budget(points, 4.99)
    assert held["best_fidelity"] == pytest.approx(0.9)
    assert held["objective_evaluations"] == 3
    assert analysis.checkpoint_at_budget(points, 8)["best_fidelity"] == pytest.approx(0.95)
    held["best_fidelity"] = 0
    assert points[1]["best_fidelity"] == pytest.approx(0.9)


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf")])
def test_cpu_budget_rejects_invalid_budgets(budget: float) -> None:
    """Undefined clocks and negative budgets cannot silently select a checkpoint."""
    with pytest.raises(ValueError, match="nonnegative"):
        analysis.checkpoint_at_budget(curve(), budget)


def test_cpu_budget_rejects_reordered_or_missing_zero_checkpoints() -> None:
    """Timing corruption and an absent initial state fail explicitly."""
    with pytest.raises(ValueError, match="ordered"):
        analysis.checkpoint_at_budget(curve()[::-1], 6)
    with pytest.raises(ValueError, match="zero batch CPU"):
        analysis.checkpoint_at_budget(curve()[1:], 1)


def test_batch_curve_retains_best_common_prefix_and_subtracts_prefix_costs() -> None:
    """A decreasing online endpoint cannot erase the free retained initial checkpoint."""
    training = {
        "online_steps": 1,
        "initial_fidelity": 0.95,
        "fidelity_trace": [0.8, 0.9, 0.98],
        "batch_start_cpu_seconds": 10,
        "records": [
            {"phase": "online", "cumulative_objective_evaluations": 2, "cumulative_gradient_evaluations": 0},
            {
                "phase": "batch",
                "iteration": 2,
                "fidelity": 0.9,
                "cumulative_cpu_seconds": 12,
                "cumulative_objective_evaluations": 5,
                "cumulative_gradient_evaluations": 1,
            },
            {
                "phase": "batch",
                "iteration": 3,
                "fidelity": 0.98,
                "cumulative_cpu_seconds": 15,
                "cumulative_objective_evaluations": 6,
                "cumulative_gradient_evaluations": 3,
            },
        ],
    }
    points = analysis.batch_curve(training)
    assert [p["batch_cpu_seconds"] for p in points] == [0, 2, 5]
    assert [p["best_fidelity"] for p in points] == [0.95, 0.95, 0.98]
    assert points[-1]["objective_evaluations"] == 4
    assert points[-1]["gradient_evaluations"] == 3


def test_fixed_pairing_keeps_missing_panel_inputs_and_negative_cpu_matched_gains() -> None:
    """A better final endpoint can still be worse within the control's measured budget."""
    base = {"target_id": "target", "seed": 300, "depth": 3, "batch_curve": curve(), "cpu_seconds": 10}
    alternative = curve()
    alternative[1]["batch_cpu_seconds"] = 6
    alternative[2]["batch_cpu_seconds"] = 10
    rows = [
        {**base, "policy": "best_backtracking", "returned_fidelity": 0.95},
        {**base, "policy": "bb_backtracking", "returned_fidelity": 0.99, "cpu_seconds": 20, "batch_curve": alternative},
    ]
    groups = analysis.fixed_comparisons(rows, [("target", 300, 3), ("missing", 400, 4)])
    selected = next(g for g in groups if (g["policy"], g["reference"]) == ("bb_backtracking", "best_backtracking"))
    assert selected["planned_pairs"] == 2
    assert selected["complete_pairs"] == 1
    assert selected["endpoint_gain_pp"] == pytest.approx(4)
    assert selected["matched_gain_pp"] == pytest.approx(-15)
    assert selected["matched_losses"] == 1
    assert selected["optimizer_cpu_ratio"] == 2
    assert selected["pairs"][0]["matched_iteration"] == 100
    assert selected["missing_pairs"] == [{"target_id": "missing", "seed": 400, "depth": 4}]


def test_primary_baselines_are_pruning_specific_and_secondary_controls_remain() -> None:
    """New methods compare with current B/D choices, never the unstable original-final arm."""
    groups = analysis.paired_comparisons([], [("target", 300)])
    assert len(groups) == 16
    primary = [g for g in groups if g["primary"]]
    assert len(primary) == 8
    assert all(g["reference"] == ("best_backtracking" if g["variant"] == "B" else "best_only") for g in primary)
    assert {g["reference"] for g in groups} == {"best_only", "best_backtracking"}
    assert all(g["planned_pairs"] == 1 and g["complete_pairs"] == 0 for g in groups)


def test_trace_figure_exports_completed_cpu_steps(tmp_path: Path) -> None:
    """Available examples export both scientific formats; absent examples need no placeholders."""
    analysis.trace_figures(tmp_path, [])
    assert not (tmp_path / "batch_refinement_traces.png").exists()
    row = {
        "target_id": "tfim_ferro",
        "seed": 400,
        "depth": 3,
        "policy": "lbfgs",
        "raw_trace": [0.5, 0.9, 0.95],
        "batch_curve": curve(),
    }
    analysis.trace_figures(tmp_path, [row])
    assert (tmp_path / "batch_refinement_traces.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "batch_refinement_traces.pdf").read_bytes().startswith(b"%PDF")


def test_report_uses_manifest_denominators_and_prints_incomplete_rows() -> None:
    """Partial scientific reports cannot present missing means as complete measurements."""
    data = {
        "planned_fixed_optimizations": 4,
        "planned_searches": 16,
        "fixed_rows": [],
        "rows": [],
        "fixed": analysis.fixed_summary([]),
        "fixed_paired": analysis.fixed_comparisons([], [("tfim_ferro", 300, 3)]),
        "paired": analysis.paired_comparisons([], [("tfim_ferro", 300)]),
        "policies": [],
        "target_ids": ["tfim_ferro"],
        "seeds": [300],
        "audits": [],
        "fixed_audits": [],
    }
    text = analysis.report(data)
    assert "Planned: 4 fixed-input optimizations and 16 full searches" in text
    assert "missing (0/1)" in text
    assert "| Policy | Reference |" in text
    assert "best_backtracking for B" in text
    assert "1 optimizer seeds" in text
    assert "their inputs differ from the earlier instability study's" in text
    assert "99.542% to 7.037% collapse is not being replayed here" in text
    assert "duplicates that shared work" in text
    assert "neither actual suite CPU nor parallel suite wall time" in text
    assert "Final exact evaluation and compilation are excluded" in text
