# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Verify fixed-panel Pareto semantics, exact ties and resource incompleteness."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from experiments.overnight_state_preparation_20260913_0108 import comparison_tradeoffs as tradeoffs


def point(
    method: str = "a",
    fidelity: float = 0.9,
    gates: int = 2,
    cpu: float | None = 3.0,
    target: str = "t",
    replica: int = 11,
) -> dict[str, Any]:
    """Return a score whose noiseless value must not enter its noisy mean."""
    score = {
        "fidelities": {n: 1.0 if n == "noiseless" else fidelity for n in tradeoffs.channels.NOISE_IDS},
        "physical_counts": {"one_qubit": 4, "two_qubit": gates, "depth": 4},
        "ballarin_counts": {"one_qubit": 8, "two_qubit": 8, "depth": 9},
        "parameters": 6,
    }
    return tradeoffs.endpoint(target, replica, method, score, cpu)


def test_endpoint_preserves_all_channel_mean_and_physical_counts() -> None:
    """Use eleven equal noisy conditions, separate ideal fidelity and pre-Ballarin counts."""
    row = point()
    assert row["fidelity"] == pytest.approx(0.9)
    assert row["noiseless_fidelity"] == pytest.approx(1.0)
    assert row["two_qubit"] == 2
    assert row["ballarin_counts"]["two_qubit"] == 8


def test_nonuniform_profiles_receive_equal_profile_weight() -> None:
    """A singleton angle profile cannot receive the weight of either five-profile noise family."""
    values = {
        noise: 1.0 if noise == "noiseless" else 0.1 if noise == "ballarin_coupled" else 0.7
        for noise in tradeoffs.channels.NOISE_IDS
    }
    score = {
        "fidelities": values,
        "physical_counts": {"one_qubit": 4, "two_qubit": 2, "depth": 4},
        "parameters": 6,
    }
    row = tradeoffs.endpoint("t", 11, "a", score, 3.0)
    assert row["fidelity"] == pytest.approx((0.1 + 10 * 0.7) / 11)
    del values["ballarin_coupled"]
    with pytest.raises(ValueError, match="twelve"):
        tradeoffs.endpoint("t", 11, "a", score, 3.0)


def test_competing_objectives_and_exact_ties() -> None:
    """An expensive gain remains a tradeoff; dominated points lose and identical ties survive."""
    rows = [point("a"), point("tie"), point("expensive", 0.91, 3, 4), point("inferior", 0.89, 3, 4)]
    result = tradeoffs.frontier(rows)
    assert result["nondominated"] == ["a", "tie", "expensive"]
    assert result["dominated_by"]["inferior"] == ["a", "tie", "expensive"]


def test_small_real_loss_is_not_rounded_into_dominance() -> None:
    """A tiny recorded fidelity loss is retained even when resources improve."""
    result = tradeoffs.frontier([point("a"), point("tiny_loss", 0.9 - 1e-13, 1, 1)])
    assert result["nondominated"] == ["a", "tiny_loss"]


def test_physical_roundoff_is_preserved_without_clipping() -> None:
    """The evaluator's documented roundoff band is retained at both physical boundaries."""
    rows = [point("near_zero", -1e-12), point("near_one", 1 + 1e-12)]
    assert tradeoffs.frontier(rows)["nondominated"] == ["near_one"]
    assert rows[0]["fidelity"] < 0
    assert rows[1]["fidelity"] > 1


def test_unknown_cpu_does_not_become_free_or_drop_method() -> None:
    """Unknown resource coverage blocks only a frontier needing that resource."""
    rows = [point(), point("b", cpu=None)]
    assert tradeoffs.frontier(rows)["status"] == "unknown_resources"
    assert tradeoffs.frontier(rows, ("two_qubit",))["status"] == "complete"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_resources_rejected(value: float) -> None:
    """Nonfinite, negative or boolean timings cannot masquerade as measured work."""
    with pytest.raises(ValueError, match="resources"):
        point(cpu=value)


@pytest.mark.parametrize("resources", [(), ("no_such_resource",), ("cpu_seconds", "cpu_seconds")])
def test_invalid_objective_definitions(resources: tuple[str, ...]) -> None:
    """Only distinct supported resource dimensions define a frontier."""
    with pytest.raises(ValueError, match="dimensions"):
        tradeoffs.frontier([point()], resources)


def test_duplicate_method_frontier_rejected() -> None:
    """Two endpoints with the same method identity cannot hide a comparison."""
    with pytest.raises(ValueError, match="distinct"):
        tradeoffs.frontier([point(), point()])


def test_full_target_replica_product_and_balanced_aggregate() -> None:
    """Aggregate every declared target and replica without taking only favorable cases."""
    rows = [
        point(target=t, replica=r, fidelity=f)
        for t, r, f in [("t", 11, 0.8), ("t", 29, 0.9), ("u", 11, 0.6), ("u", 29, 0.7)]
    ]
    result = tradeoffs.summarize(rows, ["t", "u"], [11, 29], ["a"])
    assert result["status"] == "complete"
    assert result["aggregate"]["points"][0]["fidelity"] == pytest.approx(0.75)
    assert len(result["cases"]) == 4
    missing = tradeoffs.summarize(rows[:-1], ["t", "u"], [11, 29], ["a"])
    assert missing["status"] == "incomplete_panel"
    assert missing["aggregate"] is None


def test_duplicate_case_and_unknown_aggregate_are_explicit() -> None:
    """A duplicate invalidates the complete panel; null CPU propagates through resource means."""
    rows = [point(), point(replica=29, cpu=None)]
    assert tradeoffs.summarize([*rows, rows[0]], ["t"], [11, 29], ["a"])["status"] == "incomplete_panel"
    result = tradeoffs.summarize(rows, ["t"], [11, 29], ["a"])
    assert result["aggregate"]["points"][0]["cpu_seconds"] is None
    assert result["aggregate"]["physical_resource_frontier"]["status"] == "unknown_resources"


def test_frontier_is_read_only_and_rejects_nonfinite_direct_input() -> None:
    """Scientific endpoints stay unchanged, and a direct invalid numeric point cannot win."""
    rows = [point(), point("b", 0.8)]
    previous = copy.deepcopy(rows)
    tradeoffs.frontier(rows)
    assert rows == previous
    rows[0]["fidelity"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        tradeoffs.frontier(rows)
