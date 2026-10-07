# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Full matched refinement, frozen replay and paired held-out truncation checks."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from experiments import legacy_topdown_dense as legacy
from experiments import topdown_backend_support as support
from experiments import topdown_mps_truncation as study
from experiments.topdown_backend_probe import replay_training_bank
from experiments.topdown_depth_search import sample_validation
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import KrotovTruncation, MPSBackend, StatevectorBackend
from tests.optimization.test_mps_backend import maps_for
from tests.optimization.test_statevector import case

if TYPE_CHECKING:
    from pathlib import Path

    from mqt.yaqs.optimization.backends import PreparationBackend


def test_fixed_refinement_reproduces_existing_weighted_policy() -> None:
    """Frozen maps, both cross weights, phase switch and final iterate match legacy."""
    circuit, theta, target = case(3)
    final, trace = legacy.train(
        circuit,
        theta,
        target,
        steps=6,
        online=3,
        rate=0.2,
        seed=5300,
        noisy=True,
        num_trajectories=100,
        compress_no_jump=True,
        record_noise_banks=True,
    )
    maps, weights = replay_training_bank(circuit, trace["noise_banks"][0])
    saved = theta.copy()
    actual = study.fine_tune_fixed(StatevectorBackend(circuit, target), theta, maps, weights, steps=6, online=3)
    np.testing.assert_array_equal(theta, saved)
    np.testing.assert_array_equal(actual["parameters"][-1], final)
    np.testing.assert_array_equal(actual["reported_fidelity"], trace["fidelity_trace"])
    np.testing.assert_array_equal(actual["parameters"][0], theta)


@pytest.mark.parametrize("qubits", [3, 6])
@pytest.mark.parametrize("settings", [{}, {"max_bond_dim": 8}, {"svd_threshold": 1e-14}])
def test_multistep_exact_and_sufficient_bonds_match(qubits: int, settings: dict) -> None:
    """Complex targets and explicit jumps agree across the online/batch boundary."""
    circuit, theta, target = case(qubits)
    maps, weights = maps_for(circuit), np.array([0.9, 0.07, 0.03])
    expected = study.fine_tune_fixed(StatevectorBackend(circuit, target), theta, maps, weights, steps=4, online=2)
    engine = MPSBackend(circuit, target, truncation=KrotovTruncation(**settings))
    actual = study.fine_tune_fixed(engine, theta, maps, weights, steps=4, online=2)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-11, rtol=0)
    assert engine.target_compression_infidelity < 1e-12


def test_aggressive_cap_is_measured_by_exact_assessment() -> None:
    """Approximate signals can change learned angles; target stays uncompressed."""
    circuit, theta, target = case(6)
    maps, weights = maps_for(circuit), np.array([0.9, 0.07, 0.03])
    exact = StatevectorBackend(circuit, target)
    capped = MPSBackend(circuit, target, truncation=KrotovTruncation(max_bond_dim=1))
    a = study.fine_tune_fixed(exact, theta, maps, weights, steps=4, online=2)
    b = study.fine_tune_fixed(capped, theta, maps, weights, steps=4, online=2)
    assert np.max(abs(a["parameters"] - b["parameters"])) > 1e-7
    assert capped.target_compression_infidelity < 1e-12
    assert capped.diagnostics.record()["cap_reductions"] > 0
    assert np.isfinite(exact.metrics(b["parameters"][-1], maps, weights=weights)[1])


@pytest.mark.parametrize("options", [{"steps": -1}, {"online": 201}, {"rate": 0}, {"rate": float("nan")}])
def test_invalid_refinement_options(options: dict) -> None:
    """Reject budgets before modifying parameters."""
    circuit, theta, target = case(2)
    with pytest.raises(ValueError, match="budget"):
        study.fine_tune_fixed(StatevectorBackend(circuit, target), theta, maps_for(circuit), np.ones(3) / 3, **options)


def test_nonfinite_refinement_fails() -> None:
    """A corrupt numerical backend cannot emit a successful receipt."""
    engine = SimpleNamespace(
        online_update=lambda *_a, **_kw: (np.array([np.nan]), None), metrics=lambda *_a, **_kw: (np.nan, np.nan)
    )
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        study.fine_tune_fixed(cast("PreparationBackend", engine), np.zeros(1), [], np.ones(1), steps=1, online=1)


def test_weighted_paired_statistics_equal_expanded_raw_sample() -> None:
    """Uncertainty uses raw multiplicities and paired differences, not effective K."""
    a, b, counts = np.array([0.9, 0.1, 0.4]), np.array([0.8, 0.3, 0.4]), np.array([90, 7, 3])
    raw = np.repeat(a - b, counts)
    actual = study.weighted_statistics(a - b, counts)
    assert actual["mean"] == pytest.approx(np.mean(raw))
    assert actual["standard_error"] == pytest.approx(np.std(raw, ddof=1) / np.sqrt(len(raw)))
    assert study.weighted_statistics(a - a, counts)["ci95_normal"] == [0, 0]


@pytest.mark.parametrize(
    ("values", "counts"), [([1], [1]), ([1, 2], [1, 0]), ([1, 2], [1.0, 2.0]), ([np.nan], [2]), ([1], [2, 3])]
)
def test_invalid_weighted_statistics(values: list, counts: list) -> None:
    """Reject malformed observations and misleading compressed sample counts."""
    with pytest.raises(ValueError, match="observations"):
        study.weighted_statistics(np.array(values), np.array(counts))


def test_distinct_code_assessment_matches_streamed_raw_bank() -> None:
    """Exact duplicate compression preserves mean and raw-sample standard error."""
    circuit, theta, target = case(3)
    engine = StatevectorBackend(circuit, target)
    codes = sample_validation(circuit, "IBM", 2000, [918, 73])
    assert np.any(codes)
    unique, counts = np.unique(codes, axis=0, return_counts=True)
    actual = study.weighted_statistics(study.assessment_values(engine, theta, unique), counts)
    expected = support.score_codes(engine, theta, "IBM", codes)
    assert actual["mean"] == pytest.approx(expected["fidelity"], abs=1e-14)
    assert actual["standard_error"] == pytest.approx(expected["standard_error"], abs=1e-14)
    for bad in (np.full_like(codes, 255), codes[:, :-1], codes.astype(float), -codes.astype(int)):
        with pytest.raises(ValueError, match="assessment"):
            study.assessment_values(engine, theta, bad)
    invalid = codes.copy()
    invalid[:, 4] = 1  # This H gate has explicitly disabled noise.
    with pytest.raises(ValueError, match="jump code"):
        study.assessment_values(engine, theta, invalid)
    broken = SimpleNamespace(circuit=circuit, metrics=lambda *_a, **_kw: (0, 1.1))
    with pytest.raises(ValueError, match="Nonphysical"):
        study.assessment_values(cast("PreparationBackend", broken), theta, codes[:1])


def test_study_cli_freezes_replays_and_summarizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tiny complete study authenticates inputs, runs full stages and seals arrays."""
    circuit, theta, target = case(2)
    _, trace = legacy.train(
        circuit,
        theta,
        target,
        steps=0,
        online=0,
        rate=0.2,
        seed=5300,
        noisy=True,
        num_trajectories=100,
        compress_no_jump=True,
        record_noise_banks=True,
    )
    trace.update(steps=200, online_steps=100)
    source, root = tmp_path / "source", tmp_path / "output"
    support.write_json(
        source / "manifest.json",
        {
            "config": {"qubits": 2},
            "target_id": "test",
            "target_sha256": study.target_digest(target),
            "fixture_sha256": support.digest(study.ROOT / "benchmarks/state_preparation_target_states.json"),
        },
    )
    support.save_checkpoint(source / "depth_01/pruning", {"circuit": support.circuit_record(circuit)}, {"theta": theta})
    support.save_checkpoint(source / "depth_01/refinement", {"training": trace}, {})
    monkeypatch.setattr(
        study,
        "load_target_collection",
        lambda: SimpleNamespace(load=lambda *_a: SimpleNamespace(state_vector_copy=target.copy)),
    )
    monkeypatch.setattr(study, "source_hashes", dict)
    monkeypatch.setattr(study, "VARIANTS", {"statevector": None, "mps_exact": {}})
    study.main(["--root", str(root), "--prepare", str(source), "--depths", "1"])
    with pytest.raises(FileExistsError, match="fresh"):
        study.prepare(root, source, [1])

    # CLI worker dispatch is exercised directly; no uncontrolled subprocesses in tests.
    def execute(command: list[str], **kwargs: object) -> None:
        assert kwargs["cwd"] == study.ROOT
        study.main(command[3:])

    monkeypatch.setattr(study.subprocess, "run", execute)
    study.main(["--root", str(root), "--run-all"])
    study.main(["--root", str(root), "--run-all"])
    study.main(["--root", str(root), "--summarize"])
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"]
    for row in summary["rows"]:
        assert abs(row["paired_difference"]["mean"]) < 1e-12
        assert row["validation_jump_paths"] > 0
        assert row["final_parameter_max_error"] < 1e-11
    with pytest.raises(ValueError, match="outside"):
        study.run_worker(root, 9, "statevector")
    protocol = json.loads((root / "protocol.json").read_text())
    protocol["sources"]["experiments/topdown_mps_truncation.py"] = "invalid"
    support.write_json(root / "protocol.json", protocol)
    with pytest.raises(ValueError, match="sources changed"):
        study.run_worker(root, 1, "statevector")


@pytest.mark.parametrize("cap", [1, 2, 4, 8])
def test_centered_local_truncation_matches_dense_schmidt_oracle(cap: int) -> None:
    """Gauge-only QR is exact; a centered identity split has the optimal SVD tail."""
    _, _, target = case(6)
    u_matrix, singular, vh_matrix = np.linalg.svd(target.reshape(8, 8), full_matrices=False)
    expected = ((u_matrix[:, :cap] * singular[:cap]) @ vh_matrix[:cap]).reshape(-1)
    state = MPS.from_statevector(target)
    state.set_canonical_form(2)
    np.testing.assert_allclose(state.to_vec(), target, atol=1e-12, rtol=0)
    state.apply_local_operator(np.eye(4, dtype=np.complex128), (2, 3), max_bond_dim=cap, min_bond_dim=1)
    np.testing.assert_allclose(state.to_vec(), expected, atol=1e-12, rtol=0)
    assert np.linalg.norm(state.to_vec() - target) ** 2 == pytest.approx(np.sum(singular[cap:] ** 2), abs=1e-12)
