# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Regression checks for paired training-budget comparison and cost accounting."""

from __future__ import annotations

from copy import deepcopy

import pytest

from experiments import topdown_m400_analysis as analysis


def search_row(seed: int, fidelity: float, cpu: float) -> dict:
    """Build an endpoint whose baseline and searched circuit remain distinguishable.

    Returns:
        A synthetic search result with a separate single-circuit control.
    """
    resources = {"logical_two_qubit": 8, "compiled_two_qubit": 16, "compiled_two_qubit_depth": 10}
    return {
        "target_id": "example",
        "seed": seed,
        "variant": "B",
        "hardware": "IBM",
        "selected": {"fidelity": fidelity, "depth": 3, "resources": resources},
        "search_cost": {"total_search_cpu_seconds": cpu},
        "stop_depth": 6,
        "baseline_fidelity": fidelity - 0.1,
        "baseline_depth": 1,
        "baseline_resources": resources,
        "baseline_cost": {"total_search_cpu_seconds": cpu / 2},
    }


def test_budget_comparison_pairs_identities_not_input_order() -> None:
    """Reordering completed jobs cannot mix seeds or confuse control and search."""
    old = [search_row(300, 0.8, 2), search_row(400, 0.9, 8)]
    new = [search_row(400, 0.85, 16), search_row(300, 0.86, 6)]
    rows = analysis.pair_rows(old, new)
    groups = analysis.summarize(rows)
    assert len(rows) == 4
    assert len(groups) == 2
    for group in groups:
        assert group["gain_pp"] == pytest.approx(0.5)
        assert group["cpu_ratio"] == pytest.approx(22 / 10)
        assert group["wins"] == group["losses"] == 1
    search = next(group for group in groups if group["mode"] == "depth_search")
    control = next(group for group in groups if group["mode"] == "single_circuit")
    assert search["old_percent"] == pytest.approx(85)
    assert control["old_percent"] == pytest.approx(75)


def test_budget_comparison_rejects_unmatched_and_duplicate_conditions() -> None:
    """Missing seeds and duplicate rows cannot silently shrink paired denominators."""
    rows = [search_row(300, 0.8, 2), search_row(400, 0.9, 8)]
    with pytest.raises(ValueError, match="unique, identical"):
        analysis.pair_rows(rows, rows[:1])
    with pytest.raises(ValueError, match="unique, identical"):
        analysis.pair_rows(rows + rows[:1], rows)


def test_budget_comparison_keeps_failed_search_costs_and_separate_baseline() -> None:
    """A missing winner remains a failure and its compute stays in aggregate cost."""
    old = [search_row(300, 0.8, 2), search_row(400, 0.9, 8)]
    new = deepcopy(old)
    new[1]["selected"] = None
    new[1]["search_cost"]["total_search_cpu_seconds"] = 18
    groups = analysis.summarize(analysis.pair_rows(old, new))
    search = next(group for group in groups if group["mode"] == "depth_search")
    control = next(group for group in groups if group["mode"] == "single_circuit")
    assert search["planned"] == 2
    assert search["complete"] == 1
    assert search["cpu_ratio"] == 2
    assert control["complete"] == 2
