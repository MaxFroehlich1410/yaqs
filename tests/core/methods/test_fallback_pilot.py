# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Numerical and persistence contracts of the small-system experiment runner."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.yaqs import track_conservation
from mqt.yaqs.core.methods import bug, conservation

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture
def pilot(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the manuscript runner from its source directory.

    Returns:
        Experiment module.
    """
    folder = Path(__file__).resolve().parents[3] / "paper" / "bug-mps-benchmarks" / "spin_conservation"
    monkeypatch.syspath_prepend(str(folder))
    return importlib.import_module("fallback_pilot")


def test_matched_triple_and_reference(pilot: ModuleType, tmp_path: Path) -> None:
    """Real evolution obeys cap, call-count and independent-reference contracts."""
    case = pilot.Case("tfim", 4, 2, "random_product", 0.1, total_time=0.2)
    result = pilot.run_case(case, str(tmp_path))
    assert result["status"] == "complete", result
    assert result["matrix_max_error"] < 1e-14
    assert result["reference_max_energy_drift"] < 1e-12
    for arm in result["arms"]:
        assert arm["peak_bond"] <= 2
        assert len(arm["times"]) == len(arm["energies"]) == 3
        assert arm["counts"]["scalar_calls"] == (0 if arm["variant"] == "uncorrected" else 4)
        assert arm["energies"][0] == pytest.approx(result["target_energy"], abs=1e-13)
        assert arm["evolution_seconds"] >= 0
    path = tmp_path / "cases" / case.key / "result.json"
    assert json.loads(path.read_text()) == result


@pytest.mark.parametrize("reflected", [False, True])
def test_saved_event_replays_and_preserves_gauge(pilot: ModuleType, tmp_path: Path, *, reflected: bool) -> None:
    """An infeasible target yields a faithfully saved and independently replayable event."""
    case = pilot.Case("tfim", 4, 1, "random_product", 0.1)
    state = pilot.fx.initial_state(4, "random_product", chi=1)
    mpo = pilot.fx.hamiltonian_mpo(4, "tfim")
    dense = pilot.fx.sparse_hamiltonian(4, "tfim")
    if reflected:
        state.flip_network()
        mpo = mpo.reflected()
    original = [tensor.copy() for tensor in state.tensors]
    previous_hook = bug.apply_conservation
    with (
        track_conservation() as stats,
        pilot.observe_corrections(stats, tmp_path, dense, case, "fallback") as observation,
    ):
        observation["step"] = 1
        assert not bug.apply_conservation(state, mpo, conservation.EnergyTarget(100.0, 1e-13))
    assert bug.apply_conservation is previous_hook
    assert len(observation["events"]) == stats.fallback_attempts == 1
    restored = pilot.load_snapshot(tmp_path / observation["events"][0]["snapshot"])
    assert restored.flipped == reflected
    for actual, expected in zip(restored.tensors, original, strict=True):
        np.testing.assert_array_equal(actual, expected)
    replay = importlib.import_module("fallback_replay")
    audit = replay.replay_event(tmp_path, pilot.asdict(case), observation["events"][0])
    assert audit["valid"], audit
    assert audit["checks"]["fallback"]["counts"]["infeasible"] == 1


def test_resume_rejects_different_protocol(pilot: ModuleType, tmp_path: Path) -> None:
    """Resume cannot silently mix parameters or rerun completed trajectories."""
    case = pilot.Case("xxx", 4, 1, "domain_wall", 0.1, total_time=0.1)
    first = pilot.execute([case], tmp_path, workers=1)
    assert first[0]["status"] == "complete", first
    second = pilot.execute([case], tmp_path, workers=1)
    assert first == second
    with pytest.raises(ValueError, match="different grid"):
        pilot.execute([pilot.replace(case, dt=0.05)], tmp_path, workers=1)


def test_full_frame_recovery_snapshot_replays(pilot: ModuleType, tmp_path: Path) -> None:
    """Replay a successful full-frame recovery from an eigenstate with zero gradient."""
    case = pilot.Case("xxx", 4, 1, "plus", 0.1)
    state = pilot.fx.initial_state(4, "plus", chi=1)
    mpo = pilot.fx.hamiltonian_mpo(4, "xxx")
    dense = pilot.fx.sparse_hamiltonian(4, "xxx")
    with (
        track_conservation() as stats,
        pilot.observe_corrections(stats, tmp_path, dense, case, "fallback") as observation,
    ):
        observation["step"] = 1
        assert bug.apply_conservation(state, mpo, conservation.EnergyTarget(0.0, 1e-13))
    assert stats.frame_successes == 1
    replay = importlib.import_module("fallback_replay")
    audit = replay.replay_event(tmp_path, pilot.asdict(case), observation["events"][0])
    assert audit["valid"], audit
    assert audit["checks"]["fallback"]["changed"]


def test_report_uses_pooled_denominators(pilot: ModuleType, tmp_path: Path) -> None:
    """Frequency weights actual calls, not an unweighted mean of run percentages."""
    module = importlib.import_module("fallback_report")
    for index, (calls, attempts) in enumerate([(100, 10), (10, 5)]):
        case = pilot.Case("tfim", 4, 1, "random_product", 0.1 + index)
        arm: dict[str, object] = dict.fromkeys(
            [
                "scalar_rejections",
                "fallback_frequency",
                "fallback_frequency_when_needed",
                "wall_seconds",
                "evolution_seconds",
                "correction_seconds",
                "fallback_seconds",
                "max_energy_drift",
                "max_pre_correction_drift",
                "max_post_correction_drift",
                "final_infidelity",
                "peak_bond",
            ],
            0,
        )
        arm.update({
            "variant": "fallback",
            "events": [],
            "counts": {
                "scalar_calls": calls,
                "already_satisfied": 0,
                "fallback_attempts": attempts,
                "scan_successes": attempts,
                "frame_successes": 0,
                "fallback_failures": 0,
            },
        })
        folder = tmp_path / "cases" / case.key
        folder.mkdir(parents=True)
        pilot.write_json(folder / "result.json", {"case": pilot.asdict(case), "key": case.key, "arms": [arm]})
    _, groups = module.collect(tmp_path)
    assert groups[0]["rate"] == pytest.approx(15 / 110)
    assert groups[0]["runs"] == 2


@pytest.mark.parametrize("length", [4, 6])
def test_stress_hamiltonians_and_normalization(pilot: ModuleType, length: int) -> None:
    """Every stress model has matching Hermitian representations at the requested scale."""
    models = pilot.models
    assert len(models.STRESS_MODELS) == 17
    for name in models.STRESS_MODELS:
        mpo, dense, metadata = models.hamiltonians(length, name, 1.0)
        actual = dense.toarray()
        np.testing.assert_allclose(mpo.to_matrix_mps_order(), actual, atol=2e-14)
        np.testing.assert_allclose(actual, actual.conj().T, atol=2e-14)
        eigenvalues = np.linalg.eigvalsh(actual)
        assert (eigenvalues[-1] - eigenvalues[0]) / length == pytest.approx(1.0, abs=2e-14)
        assert metadata["scale"] > 0
        assert metadata["normalization"] == "spectral_width_per_site"


def test_stress_known_limits(pilot: ModuleType) -> None:
    """Model conventions reproduce the manuscript's HS, XXX, and TFIM limits."""
    models = pilot.models
    length = 6
    limits = [
        (models.Model("power", alpha=2, delta=1), "hs"),
        (models.Model("j1j2", j2=0), "xxx"),
        (models.Model("tilted_ising", hz=0), "tfim"),
    ]
    for model, legacy in limits:
        pairs, fields = models.coefficients(length, model)
        independent = pilot.fx.sparse_hamiltonian(length, legacy).toarray()
        np.testing.assert_allclose(models.bit_matrix(pairs, fields).toarray(), independent, atol=2e-14)
        np.testing.assert_allclose(models.pair_mpo(pairs, fields).to_matrix_mps_order(), independent, atol=2e-14)


def test_normalized_entangled_run_and_fixed_initial_state(pilot: ModuleType, tmp_path: Path) -> None:
    """Seeded physical input and energy unit stay fixed as the evolution cap changes."""
    results = []
    for cap in [2, 4, 8]:
        case = pilot.Case(
            "power_a1.5_d2",
            4,
            cap,
            "random_entangled",
            0.1,
            total_time=0.1,
            initial_chi=2,
            width_per_site=1.0,
        )
        result = pilot.run_case(case, str(tmp_path))
        assert result["status"] == "complete", result
        assert result["initial_infidelity_to_unpadded"] is None
        assert result["hamiltonian"]["spectral_width_per_site"] == pytest.approx(1.0)
        results.append(result)
    assert len({result["initial_vector_sha256"] for result in results}) == 1
    assert len({result["target_energy"] for result in results}) == 1


def test_normalized_event_replay(pilot: ModuleType, tmp_path: Path) -> None:
    """Replay reconstructs exactly the scaled stress operator, including reflection."""
    case = pilot.Case("j1j2_0.5", 4, 1, "random_product", 0.1, width_per_site=1.0)
    state = pilot.fx.initial_state(4, "random_product", chi=1)
    mpo, dense, _ = pilot.models.hamiltonians(4, case.model, case.width_per_site)
    state.flip_network()
    mpo = mpo.reflected()
    with (
        track_conservation() as stats,
        pilot.observe_corrections(stats, tmp_path, dense, case, "fallback") as observation,
    ):
        assert not bug.apply_conservation(state, mpo, conservation.EnergyTarget(100.0, 1e-13))
    assert observation["events"][0]["stage"] == "infeasible"
    replay = importlib.import_module("fallback_replay")
    assert replay.replay_event(tmp_path, pilot.asdict(case), observation["events"][0])["valid"]


def test_invalid_initial_cap_is_rejected(pilot: ModuleType, tmp_path: Path) -> None:
    """A fixed initial rank cannot quietly exceed an evolution cap."""
    case = pilot.Case("xxx", 4, 2, "random_entangled", 0.1, total_time=0.1, initial_chi=4)
    result = pilot.run_case(case, str(tmp_path))
    assert result["status"] == "error"
    assert "Initial bond dimension" in result["traceback"]


def test_parallel_rank_growth_at_cap_eight(pilot: ModuleType, tmp_path: Path) -> None:
    """Spawned workers retain single-thread kernels while the MPS grows to rank eight."""
    cases = [
        pilot.Case(model, 8, 8, "random_entangled", 0.1, total_time=0.2, initial_chi=2, width_per_site=1.0)
        for model in ["power_a1_d2", "tilted_ising_1"]
    ]
    results = pilot.execute(cases, tmp_path, workers=2)
    assert all(result["status"] == "complete" for result in results), results
    assert all(arm["peak_bond"] == 8 for result in results for arm in result["arms"])
