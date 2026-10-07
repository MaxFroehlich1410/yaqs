# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Paired statistics and conservative competitiveness-screen regression tests."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments import state_preparation_trajectory_analysis as analysis
from experiments import state_preparation_trajectory_comparison as experiment
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def panel() -> list[dict[str, object]]:
    """Return a complete balanced synthetic panel with known fidelity/cost gaps."""
    return [
        {
            "method": {"kind": kind},
            "family": family,
            "target_id": f"{family}_{index}",
            "replica": replica,
            "final_fidelity": 0.8 + (0.01 if kind == "trajectory_hybrid" else 0),
            "frozen_fidelity": 0.7,
            "final_noiseless_fidelity": 0.9,
            "cpu_seconds": 11.0 if kind == "trajectory_hybrid" else 10.0,
            "wall_seconds": 20.0,
        }
        for kind in experiment.KINDS
        for family in experiment.FAMILIES
        for index in (1, 2)
        for replica in (11, 29)
    ]


def test_balanced_paired_statistics_and_fidelity_screen() -> None:
    """A one-pp gain at 10% extra cost satisfies the predeclared fidelity screen."""
    result = analysis.paired_summary(panel())
    assert cast("dict", result["methods"])["trajectory_hybrid"]["fidelity"] == pytest.approx(0.81)
    assert result["highest_mean_fidelity_method"] == "trajectory_hybrid"
    assert result["strongest_reference"] != "trajectory_hybrid"
    for comparison in cast("dict", result["hybrid_comparisons"]).values():
        assert comparison["mean_delta_pp"] == pytest.approx(1)
        assert comparison["median_delta_pp"] == pytest.approx(1)
        assert comparison["winning_pairs"] == 16
        assert comparison["hybrid_over_comparator_cpu"] == pytest.approx(1.1)
        assert comparison["fidelity_screen"] is True
        assert comparison["speed_screen"] is False


def test_family_regression_cannot_be_hidden_by_overall_gain() -> None:
    """Failure in one target family blocks a cross-family competitiveness claim."""
    rows = panel()
    for row in rows:
        if cast("dict", row["method"])["kind"] == "trajectory_hybrid" and row["family"] == "haar":
            row["final_fidelity"] = 0.799
    result = analysis.paired_summary(rows)
    for comparison in cast("dict", result["hybrid_comparisons"]).values():
        assert comparison["mean_delta_pp"] > 0.5
        assert comparison["fidelity_screen"] is False
        assert comparison["family_delta_pp"]["haar"] < 0


def test_speed_screen_requires_accuracy_and_cost_together() -> None:
    """A near-match at half the cost passes, but a larger fidelity loss does not."""
    rows = panel()
    for row in rows:
        if cast("dict", row["method"])["kind"] == "trajectory_hybrid":
            row["final_fidelity"] = 0.799
            row["cpu_seconds"] = 5.0
    result = analysis.paired_summary(rows)
    assert result["speed_screen_against_strongest_reference"] is True
    for row in rows:
        if cast("dict", row["method"])["kind"] == "trajectory_hybrid":
            row["final_fidelity"] = 0.795
    assert analysis.paired_summary(rows)["speed_screen_against_strongest_reference"] is False


@pytest.mark.parametrize("damage", ["missing", "duplicate", "replica", "unpaired", "nan", "cost"])
def test_invalid_or_unpaired_panels_are_rejected(damage: str) -> None:
    """Do not silently drop failures, overweight a replica, or join unequal starts."""
    rows = panel()
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows[0] = rows[1]
    elif damage == "replica":
        rows[0]["replica"] = 44
    elif damage == "unpaired":
        rows[0]["frozen_fidelity"] = 0.6
    elif damage == "nan":
        rows[0]["final_fidelity"] = float("nan")
    else:
        rows[0]["cpu_seconds"] = 0
    with pytest.raises(ValueError, match=r"Expected|Methods"):
        analysis.paired_summary(rows)


def audit_fixture(root: Path) -> dict[str, object]:
    """Return a synthetic no-motion panel for testing the complete audit pipeline."""
    vector = np.array([np.sqrt(0.3), np.sqrt(0.7)], dtype=complex)
    targets = [
        experiment.target_record(f"{family}_{index}", family, vector)
        for family in experiment.FAMILIES
        for index in (1, 2)
    ]
    development: list[dict[str, object]] = [
        {
            "method": asdict(method),
            "family": family,
            "target_id": target,
            "replica": 11,
            "final_fidelity": 0.8 + method.rate / 100,
        }
        for method in experiment.development_methods()
        for target, family in zip(pilot.TARGETS, experiment.FAMILIES, strict=True)
    ]
    selection = experiment.nominate(development)
    manifest: dict[str, object] = {
        "targets": targets,
        "config": asdict(experiment.Config()),
        "runtime_fingerprint": "test",
    }
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)])
    theta = np.array([0.4])
    start = {"circuit": pilot.circuit_record(circuit), "theta": theta.tolist()}
    endpoint = experiment.exact.evaluate(circuit, theta, vector, 0)
    rows = []
    for target in targets:
        for replica in (11, 29):
            pilot.write_record(root / "starts" / str(target["target_id"]) / f"seed_{replica}" / "start.json", start)
            for kind in experiment.KINDS:
                method = experiment.Method(**cast("dict", selection["selected"])[kind])
                path = f"confirmation/{target['target_id']}/seed_{replica}/{method.label}.json"
                trace = [
                    {
                        "phase": method.phase_rate(step, experiment.Config())[0],
                        "rate": method.phase_rate(step, experiment.Config())[1],
                        "cpu_seconds": 1.0,
                        "wall_seconds": 2.0,
                    }
                    for step in range(1, 51)
                ]
                state = {
                    "status": "complete",
                    "config": manifest["config"],
                    "method": asdict(method),
                    "completed_updates": 50,
                    "trace": trace,
                    "theta": theta.tolist(),
                    "start_checksum": pilot.digest(start),
                    "target_checksum": pilot.digest(target),
                    "seed": experiment.seed_for("confirmation", str(target["target_id"]), replica),
                    "evaluations": [{**endpoint, "update": step} for step in (0, 10, 25, 50)],
                }
                pilot.write_record(root / path, state)
                rows.append({
                    "target_id": target["target_id"],
                    "family": target["family"],
                    "replica": replica,
                    "path": path,
                    "checksum": pilot.digest(state),
                    "method": asdict(method),
                    "final_fidelity": endpoint["exact_noisy_fidelity"],
                    "frozen_fidelity": endpoint["exact_noisy_fidelity"],
                    "final_noiseless_fidelity": endpoint["noiseless_fidelity"],
                    "cpu_seconds": 50.0,
                    "wall_seconds": 100.0,
                })
    result: dict[str, object] = {
        "status": "complete",
        "manifest_checksum": pilot.digest(manifest),
        "selection_checksum": pilot.digest(selection),
        "development": development,
        "confirmation": rows,
    }
    for name, record in (("manifest", manifest), ("selection", selection), ("result", result)):
        pilot.write_record(root / f"{name}.json", record)
    return result


def test_independent_audit_recomputes_all_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit all 640 fidelities, then detect a plausible but incorrect result summary."""
    monkeypatch.setattr(experiment, "runtime_fingerprint", lambda: "test")
    result = audit_fixture(tmp_path)
    audited = analysis.audit(tmp_path)
    assert audited["independent_fidelity_checks"] == 640
    assert cast("float", audited["max_fidelity_error"]) < 1e-12
    assert audited["confirmation_runs"] == 80
    cast("list[dict]", result["confirmation"])[0]["final_fidelity"] = 0.5
    pilot.write_record(tmp_path / "result.json", result)
    with pytest.raises(ValueError, match="Summary row"):
        analysis.audit(tmp_path)


def test_audit_rejects_changed_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not validate a scientific run against silently changed source code."""
    audit_fixture(tmp_path)
    monkeypatch.setattr(experiment, "runtime_fingerprint", lambda: "changed")
    with pytest.raises(ValueError, match="provenance"):
        analysis.audit(tmp_path)
