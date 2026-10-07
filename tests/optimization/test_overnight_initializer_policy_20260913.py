# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Geometry-only policy selection, physical preparation and complete probe charges."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import initializer_policy as policy
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate


def legacy_source(n: int) -> dict[str, Any]:
    """Return a small physical legacy fixture with known construction cost."""
    circuit = ParameterizedCircuit(n, [ParameterizedGate("ry", (q,), q) for q in range(n)])
    return {
        "circuit": pilot.circuit_record(circuit),
        "theta": [0.2] * n,
        "construction_cpu_seconds": 10.0,
        "timing_complete": True,
    }


@pytest.mark.parametrize("n", [1, 2, 3])
def test_exact_rank_two_targets_select_physical_compiled_rank_two_and_preserve_inputs(n: int) -> None:
    """All q1-q3 states admit bond two and retain exact preparation after compilation."""
    rng = np.random.default_rng(37 + n)
    target = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    target /= np.linalg.norm(target)
    legacy = legacy_source(n)
    before, vector = copy.deepcopy(legacy), target.copy()
    result = policy.build_initializers(target, legacy)
    assert result["rank_two_selected"]
    assert result["conditional"]["selected_source"] == "rank_two"
    selected = result["conditional"]
    physical = policy.state_vector(selected["circuit"], np.asarray(selected["theta"]))
    assert abs(np.vdot(target, physical)) ** 2 == pytest.approx(1, abs=1e-12)
    assert result["conditional"]["initializer_cpu_seconds"] == pytest.approx(
        result["costs"]["rank_two_probe_cpu_seconds"]
    )
    assert result["legacy"]["initializer_cpu_seconds"] > 10
    assert result["costs"]["discarded_probe_cpu_seconds"] == pytest.approx(0)
    assert legacy == before
    np.testing.assert_array_equal(target, vector)


def test_synthetic_q4_rank_loss_uses_legacy_but_still_compiles_and_charges_full_probe() -> None:
    """A synthetic full-rank central cut falsifies unconditional rank-two selection."""
    target = np.zeros(16, dtype=np.complex128)
    target[[0, 5, 10, 15]] = 0.5
    result = policy.build_initializers(target, legacy_source(4))
    assert result["ideal_tt_overlap"] < 0.99
    assert not result["rank_two_selected"]
    assert result["conditional"]["selected_source"] == "legacy"
    assert result["conditional"]["circuit"] == result["legacy"]["circuit"]
    assert result["conditional"]["theta"] == result["legacy"]["theta"]
    costs = result["costs"]
    assert costs["rank_two_synthesis_cpu_seconds"] > 0
    assert costs["overlap_and_lowering_audit_cpu_seconds"] > 0
    assert costs["rank_two_compilation_and_audit_cpu_seconds"] > 0
    assert costs["discarded_probe_cpu_seconds"] == costs["rank_two_probe_cpu_seconds"]
    assert result["conditional"]["initializer_cpu_seconds"] == pytest.approx(
        result["legacy"]["initializer_cpu_seconds"] + costs["rank_two_probe_cpu_seconds"]
    )


def test_threshold_has_no_tolerance_band_or_target_identity_input() -> None:
    """The one fixed threshold includes equality and excludes the immediately lower float."""
    assert policy.use_rank_two(0.99)
    assert not policy.use_rank_two(float(np.nextafter(0.99, 0)))
    assert policy.use_rank_two(float(np.nextafter(0.99, 1)))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_overlap_cannot_silently_trigger_fallback(value: float) -> None:
    """Invalid geometric information fails instead of changing the policy branch."""
    with pytest.raises(ValueError, match="physical fidelity"):
        policy.use_rank_two(value)


@pytest.mark.parametrize(
    "change", [{"construction_cpu_seconds": None}, {"construction_cpu_seconds": -1}, {"timing_complete": False}]
)
def test_unknown_or_incomplete_archive_cost_cannot_be_used(change: dict[str, Any]) -> None:
    """Historical wall time cannot masquerade as a fresh complete construction CPU cost."""
    legacy = {**legacy_source(2), **change}
    with pytest.raises(ValueError, match="measured legacy construction CPU"):
        policy.build_initializers(np.array([1, 0, 0, 0], dtype=np.complex128), legacy)


def test_false_synthesis_overlap_metadata_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The selector recomputes ideal overlap instead of trusting a claimed fidelity."""
    synthesize = policy.synthesis.synthesize

    def corrupted(target: np.ndarray) -> dict[str, Any]:
        result = synthesize(target)
        result["compression"]["approximation_fidelity"] = 0.5
        return result

    monkeypatch.setattr(policy.synthesis, "synthesize", corrupted)
    with pytest.raises(FloatingPointError, match="overlap or physical lowering"):
        policy.build_initializers(np.array([1, 0], dtype=np.complex128), legacy_source(1))


def test_source_compiler_preserves_native_mapping_and_rejects_data_maps() -> None:
    """Input metadata cannot disappear while resolving a frozen scalar endpoint."""
    legacy = legacy_source(2)
    legacy["circuit"]["gates"][0]["native_gate_id"] = "physical_origin"
    result = policy.compile_source(legacy)
    assert result["output_mapping"][0]["source_native_gate_ids"] == ["physical_origin"]
    legacy["circuit"]["gates"][0]["data_map"] = "forbidden"
    with pytest.raises(ValueError, match="without data maps"):
        policy.compile_source(legacy)
