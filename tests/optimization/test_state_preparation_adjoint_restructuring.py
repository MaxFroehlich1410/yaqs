# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared-score verification barrier, unchanged search and budgeted continuation."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from experiments import state_preparation_adjoint_restructuring as study
from experiments import state_preparation_noisy_restructuring as base
from experiments import state_preparation_pilot as pilot
from experiments import state_preparation_trajectory_comparison as comparison
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def synthetic_case(name: str, replica: int) -> dict[str, Any]:
    """Return an engineering-only case with protected and eligible gates."""
    gates = [ParameterizedGate("ry", (i % 2,), param_index=i, logical_gate_id=i) for i in range(6)]
    gates.extend([
        ParameterizedGate("rxx", (1, 0), param_index=7, logical_gate_id=70),
        ParameterizedGate("rz", (0,), param_index=6, logical_gate_id=60),
        ParameterizedGate("rx", (1,), param_index=8, logical_gate_id=80),
    ])
    target = np.array([1, 2j, -3, 4j]) / np.sqrt(30)
    case = {
        "target": comparison.target_record(name, "test", target),
        "replica": replica,
        "circuit": pilot.circuit_record(ParameterizedCircuit(2, gates)),
        "initial": np.linspace(-0.3, 0.5, 9).tolist(),
    }
    case["expected_fidelity"] = base.fidelity(base.initial_state(case), target)
    return case


def small_config() -> base.Config:
    """Return small legacy-source budgets, inherited unchanged by the new run."""
    return base.Config(warmup=4, online_updates=2, rounds=2, repair=2, trajectories=4, control_max_updates=40)


def test_ranker_injection_preserves_all_repair_branches_and_moments() -> None:
    """Changing only scoring produces identical accepted and discarded optimizer states."""
    case, config = synthetic_case("test", 11), small_config()
    target = comparison.target_vector(case["target"])
    warm = base.train_block(base.initial_state(case), target, "adaptive_online", config, 123, 0, 4)
    ledger = [{"role": "warmup", "cpu_seconds": 0.01, "wall_seconds": 0.01}]
    prefix = {**warm, "ledger": ledger, "curve": [base.point(warm["state"], warm["fidelity"], ledger)]}
    exhaustive = base.restructure(prefix, target, "adaptive_online", config, 123)
    shared = base.restructure(prefix, target, "adaptive_online", config, 123, ranker=study.rank_deletions)
    study.verify_search(exhaustive, shared)
    assert shared["state"] == exhaustive["state"]
    changed = copy.deepcopy(shared)
    changed["rounds"][0]["candidates"][1]["state"]["second"][0] += 1
    with pytest.raises(ValueError, match="repair branch"):
        study.verify_search(exhaustive, changed)


def test_score_comparison_rejects_bad_values_and_changed_rankings() -> None:
    """A numerical score or shortlist discrepancy must block scientific continuation."""
    rows = [{"index": i, "origin": i + 6, "arity": 1, "fidelity": 0.7 - i * 0.1} for i in range(2)]
    assert study.compare_scores(rows, copy.deepcopy(rows)) == 0
    with pytest.raises(ValueError, match="ranking"):
        study.compare_scores(rows, rows[::-1])
    for value in (np.nan, 0.8):
        changed = copy.deepcopy(rows)
        changed[0]["fidelity"] = value
        with pytest.raises(ValueError, match="scores"):
            study.compare_scores(rows, changed)


def test_continuation_preserves_state_and_skips_already_consumed_noise_epochs() -> None:
    """Rejected structural rounds do not cause post-pruning training to replay noise."""
    case, config = synthetic_case("test", 11), small_config()
    target = comparison.target_vector(case["target"])
    warm = base.train_block(base.initial_state(case), target, "adaptive_online", config, 123, 0, 4)
    compact = base.deleted_state(warm["state"], 6)
    score = base.fidelity(compact, target)
    ledger = [{"role": "search", "cpu_seconds": 0.01, "wall_seconds": 0.01}]
    prefix = {"state": compact, "fidelity": score, "ledger": ledger, "curve": [base.point(compact, score, ledger)]}
    control = base.fixed_control(
        prefix,
        target,
        "adaptive_online",
        replace(config, control_max_updates=6),
        123,
        100,
        epoch_start=20,
        role="post_prune_refinement",
    )
    direct = base.train_block(compact, target, "adaptive_online", config, 123, 20, 2)
    assert control["blocks"][0]["state"] == direct["state"]
    assert [t["epoch"] for t in control["blocks"][0]["trace"]] == [21, 22]
    assert control["ledger"][-1]["role"] == "post_prune_refinement"
    assert control["last_state"]["counts"] == [4] * 8
    for epoch in (-1, 3, True):
        with pytest.raises(ValueError, match="epochs"):
            base.fixed_control(prefix, target, "adaptive_online", config, 123, 100, epoch_start=epoch)


def test_historical_authentication_verification_barrier_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All source cases verify before optimization, and completed resume preserves bytes."""
    cases = [synthetic_case("a", 11), synthetic_case("b", 29)]
    monkeypatch.setattr(base, "load_cases", lambda _: ({"engineering": "source"}, cases))
    old_root, root = tmp_path / "old", tmp_path / "new"
    original_charge = base.charged

    def padded_charge(function: Callable[[], object], ledger: list[dict[str, Any]], role: str) -> object:
        """Return real work with synthetic search cost, avoiding millisecond-budget timing flakiness."""
        value = original_charge(function, ledger, role)
        if role == "deletion_scoring":
            ledger[-1]["cpu_seconds"] += 0.25
        return value

    # This tests budget inheritance and replay, not a speed assertion on tiny
    # circuits. Padding only the engineering source makes cold worker costs safe.
    with monkeypatch.context() as source_cost:
        source_cost.setattr(base, "charged", padded_charge)
        base.run(old_root, tmp_path, config=small_config(), workers=1)
    old_bytes = {p: p.read_bytes() for p in old_root.rglob("*") if p.is_file()}
    verification = study.run(root, old_root, workers=1, stage="verify")
    assert verification["status"] == "passed"
    assert not (root / "optimization").exists()
    result = study.run(root, old_root, workers=2, resume=True)
    assert len(result["rows"]) == 4
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert study.run(root, old_root, workers=1, resume=True) == result
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert old_bytes == {p: p.read_bytes() for p in old_root.rglob("*") if p.is_file()}
    for row in result["jobs"]:
        record = cast("dict[str, Any]", pilot.read_record(root / row["path"]))
        previous = cast(
            "dict[str, Any]", pilot.read_record(old_root / f"{row['target_id']}/seed_{row['replica']}.json")
        )
        for method in base.METHODS:
            study.verify_search(previous["structural"][method], record["searches"][method])
            ledger = record["refined"][method]["ledger"]
            assert ledger[: len(record["searches"][method]["ledger"])] == record["searches"][method]["ledger"]
        assert record["budget_seconds"] == max(base.cpu_cost(a["ledger"]) for a in previous["structural"].values())
    path = root / verification["jobs"][0]["path"]
    record = pilot.read_record(path)
    record["status"] = "failed"
    pilot.write_record(path, record)
    with pytest.raises(ValueError, match="Verification checkpoint"):
        study.run(root, old_root, workers=1, resume=True)


def test_wrong_source_score_blocks_all_optimization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed exhaustive-score check must not create any optimization case."""
    cases = [synthetic_case("a", 11)]
    monkeypatch.setattr(base, "load_cases", lambda _: ({"engineering": "source"}, cases))
    old_root, root = tmp_path / "old", tmp_path / "new"
    base.run(old_root, tmp_path, config=small_config(), workers=1)
    original = study.rank_deletions

    def incorrect(state: dict[str, Any], target: np.ndarray) -> list[dict[str, Any]]:
        rows = original(state, target)
        rows[0]["fidelity"] += 0.01
        return rows

    monkeypatch.setattr(study, "rank_deletions", incorrect)
    with pytest.raises(ValueError, match="scores"):
        study.run(root, old_root, workers=1)
    assert not (root / "optimization").exists()
