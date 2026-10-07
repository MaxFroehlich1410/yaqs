# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Chronological common-cutoff tests independent of any benchmark outcome."""

from __future__ import annotations

import copy
from dataclasses import asdict
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import common_control_audit as audit
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def arms() -> dict[str, Any]:
    """Return a late compiler point and an already completed shorter control."""
    return {
        "candidate": {
            "status": "complete",
            "curve": [
                {"cpu_seconds": 0, "theta": [0.9]},
                {"cpu_seconds": 2, "theta": [0.1]},
                {"cpu_seconds": 4, "theta": [0.99], "compiled_final": True},
            ],
        },
        "short_control": {
            "status": "complete",
            "curve": [{"cpu_seconds": 0, "theta": [0.8]}, {"cpu_seconds": 1, "theta": [0.2]}],
        },
    }


def test_select_last_available_without_quality_or_retroactive_compilation() -> None:
    """High-looking later values never override chronological availability."""
    data = arms()
    before = copy.deepcopy(data)
    selected = audit.select_points(data, tuple(data), 3)
    assert selected["candidate"]["theta"] == [0.1]
    assert "compiled_final" not in selected["candidate"]
    assert selected["short_control"]["theta"] == [0.2]
    assert data == before


def test_equal_timestamps_use_last_actual_point() -> None:
    """A completed compiler point is available exactly at its recorded time."""
    data = arms()
    data["candidate"]["curve"].insert(2, {"cpu_seconds": 4, "theta": [0.3]})
    selected = audit.select_points(data, tuple(data), 4)
    assert selected["candidate"]["compiled_final"]


@pytest.mark.parametrize("cutoff", [-1, float("nan"), float("inf")])
def test_invalid_cutoff_rejected(cutoff: float) -> None:
    """Unavailable or nonphysical time budgets cannot produce a comparison."""
    with pytest.raises(ValueError, match="finite nonnegative"):
        audit.select_points(arms(), ("candidate",), cutoff)


@pytest.mark.parametrize("fault", ["failed", "missing", "empty", "unordered", "nonfinite", "negative", "too_late"])
def test_no_failed_or_unavailable_control_is_omitted(fault: str) -> None:
    """One bad control invalidates the whole requested comparison."""
    data = arms()
    if fault == "failed":
        data["short_control"]["status"] = "failed"
    elif fault == "missing":
        del data["short_control"]
    elif fault == "empty":
        data["short_control"]["curve"] = []
    elif fault == "unordered":
        data["short_control"]["curve"].reverse()
    elif fault == "nonfinite":
        data["short_control"]["curve"][0]["cpu_seconds"] = float("nan")
    elif fault == "negative":
        data["short_control"]["curve"][0]["cpu_seconds"] = -1
    else:
        data["short_control"]["curve"] = [{"cpu_seconds": 9, "theta": [0.1]}]
    with pytest.raises(ValueError, match=r"chronological curve|no deployable"):
        audit.select_points(data, ("candidate", "short_control"), 3)


@pytest.mark.parametrize("methods", [(), ("candidate", "candidate")])
def test_method_set_must_be_nonempty_and_distinct(methods: tuple[str, ...]) -> None:
    """Duplicated controls cannot receive extra aggregate weight."""
    with pytest.raises(ValueError, match="distinct"):
        audit.select_points(arms(), methods, 3)


@pytest.mark.parametrize("tamper", ["child", "inventory", "control_score", "primary_score", "time"])
def test_fresh_q2_controls_cached_resume_and_source_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    """New all12 scores use full metadata and never mutate the source circuit bank."""
    source_root, output = tmp_path / "source", tmp_path / "audit"
    case_root = source_root / "synthetic_control__11"
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), 1, native_gate_id=92),
            ParameterizedGate("rxx", (1, 0), 0, logical_gate_id=19, native_gate_id=43),
        ],
        2,
    )
    theta = np.array([0.37, -0.21])
    target = np.array([0.1j, 0.4, -0.2, 0.3j])
    target /= np.linalg.norm(target)
    config = audit.composed.exact.Config()
    store = audit.composed.CompleteStore(case_root, target, config)
    key = store.circuit(circuit)
    point = {"circuit_hash": key, "theta": theta.tolist(), "cpu_seconds": 2.0}
    source_score = audit.composed.score(store, point)
    case: dict[str, Any] = {
        "common_primary": {
            "refinement_cpu_seconds": 3.0,
            "total_cpu_seconds": 10.0,
            "arms": {name: {"point": point, "scores": source_score} for name in audit.composed.PRIMARY},
        },
        "arms": {"unchanged": {"scores": source_score}},
    }
    pilot.write_record(case_root / "RESULT.json", case)
    pilot.write_record(source_root / "PROTOCOL.json", {"exact_config": asdict(config)})
    original_bytes = {str(path): path.read_bytes() for path in source_root.rglob("*.json")}
    entry = {
        "target_id": "synthetic_control",
        "checksum": pilot.digest(case),
        "points": dict.fromkeys(audit.composed.TRAINED, point),
    }
    monkeypatch.setattr(audit, "load_target", lambda *_args: SimpleNamespace(state_vector_copy=target.copy))
    result = audit.score_case(output, source_root, entry, {"test": "q2"})
    assert set(result["value"]["scores"]) == set(audit.composed.TRAINED)
    for name in ("fixed_mean2_lbfgs", "fixed_noiseless_adam"):
        assert result["value"]["scores"][name]["fidelities"] == source_score["fidelities"]
    assert result["cpu_seconds"] > 0
    assert audit.score_case(output, source_root, entry, {"test": "q2"}) == result
    assert {str(path): path.read_bytes() for path in source_root.rglob("*.json")} == original_bytes
    case["tampered"] = True
    pilot.write_record(case_root / "RESULT.json", case)
    with pytest.raises(ValueError, match="source case changed"):
        audit.score_case(output, source_root, entry, {"test": "q2"})
    del case["tampered"]
    pilot.write_record(case_root / "RESULT.json", case)
    path = next((output / "operations").glob("*.json")) if tamper == "child" else output / "RESULT.json"
    altered = audit.read(path)
    if tamper == "child":
        altered["tampered"] = True
    elif tamper == "inventory":
        altered["value"]["dependencies"].popitem()
    elif tamper == "time":
        altered["value"]["total_cpu_seconds"] = 0
    else:
        name = "fixed_mean2_lbfgs" if tamper == "control_score" else audit.composed.PRIMARY[0]
        altered["value"]["scores"][name]["mean_noisy_fidelity"] = 0.999
    pilot.write_record(path, altered)
    with pytest.raises(ValueError, match=r"audit dependency changed|authenticated receipts|cutoff or source"):
        audit.score_case(output, source_root, entry, {"test": "q2"})


def summary_cases() -> list[dict[str, Any]]:
    """Return three synthetic complete targets where fixed L-BFGS is the strongest control."""
    records = []
    for target in ("a", "b", "c"):
        scores = {
            name: {"fidelities": dict.fromkeys(audit.composed.channels.NOISE_IDS, 0.90)}
            for name in audit.composed.TRAINED
        }
        scores["structural_mean2_lbfgs"]["fidelities"] = dict.fromkeys(audit.composed.channels.NOISE_IDS, 0.922)
        scores["fixed_mean2_lbfgs"]["fidelities"] = dict.fromkeys(audit.composed.channels.NOISE_IDS, 0.914)
        records.append({
            "value": {
                "target_id": target,
                "scores": scores,
                "source_unchanged_fidelities": dict.fromkeys(audit.composed.channels.NOISE_IDS, 0.91),
            }
        })
    return records


def test_summary_uses_strongest_whole_panel_control_and_all_target_signs() -> None:
    """A fixed control can outrank weighted Adam, and concentrated gains cannot pass."""
    rows = summary_cases()
    summary = audit.summarize(rows, ("a", "b", "c"))
    assert summary["strongest_control"] == "fixed_mean2_lbfgs"
    assert summary["same_rule_passes_against_strongest"]
    assert summary["paired_comparisons"]["fixed_mean2_lbfgs"]["mean_delta_pp"] == pytest.approx(0.8)
    assert "unchanged" in summary["paired_comparisons"]
    for row, value in zip(rows, (0.99, 0.90, 0.90), strict=True):
        row["value"]["scores"]["structural_mean2_lbfgs"]["fidelities"] = dict.fromkeys(
            audit.composed.channels.NOISE_IDS, value
        )
    summary = audit.summarize(rows, ("a", "b", "c"))
    assert summary["paired_comparisons"]["fixed_mean2_lbfgs"]["mean_delta_pp"] > 0.5
    assert not summary["same_rule_passes_against_strongest"]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "channel", "nonfinite"])
def test_summary_requires_every_case_and_every_channel(fault: str) -> None:
    """Missing or invalid outcomes never produce a reduced-panel headline."""
    rows = summary_cases()
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "channel":
        rows[0]["value"]["scores"]["fixed_mean2_lbfgs"]["fidelities"].pop("noiseless")
    else:
        rows[0]["value"]["source_unchanged_fidelities"]["noiseless"] = float("nan")
    with pytest.raises(ValueError, match=r"entire distinct target panel|all twelve finite"):
        audit.summarize(rows, ("a", "b", "c"))
