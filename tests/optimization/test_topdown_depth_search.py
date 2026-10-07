# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Depth-selection independence, raw-sample validation, lineage and replay checks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit.quantum_info import DensityMatrix, Operator, Statevector, SuperOp

from experiments import topdown_depth_search as study
from experiments.legacy_topdown_dense import PRIMITIVES, noise_channel
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate, StatevectorBackend

if TYPE_CHECKING:
    from pathlib import Path


def small_circuit() -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Return a nontrivial entangled two-qubit validation fixture."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), 0),
            ParameterizedGate("rxx", (1, 0), 1),
            ParameterizedGate("rz", (1,), 2),
        ],
        3,
    )
    theta = np.array([0.7, -0.6, 1.1])
    target = np.array([1, 2j, -3, 4j], dtype=complex) / np.sqrt(30)
    return circuit, theta, target


def test_patience_tracks_plateaus_accumulated_gains_and_actual_winner() -> None:
    """Tiny improvements can accumulate; winner selection does not use the tolerance."""
    tracker = study.Patience()
    assert tracker.observe(1, 0.9)["material_improvement"]
    assert tracker.observe(2, 0.9005)["stale"] == 1
    assert tracker.observe(3, 0.9009)["stale"] == 2
    assert tracker.observe(4, 0.9012)["material_improvement"]
    assert tracker.observe(5, 0.9013)["stale"] == 1
    assert tracker.observe(6, 0.9013)["stale"] == 2
    assert tracker.observe(7, 0.899)["stale"] == 3
    assert json.loads(json.dumps(tracker.observe(8, 0.899)))["material_improvement"] is False
    assert tracker.best_depth == 5
    assert tracker.record_score == pytest.approx(0.9012)


def test_patience_uncertainty_failures_and_input_validation() -> None:
    """Use independent standard errors and retain no winner for all-failed searches."""
    tracker = study.Patience()
    tracker.observe(1, 0.8, 0.002)
    assert not tracker.observe(2, 0.805, 0.002)["material_improvement"]
    assert tracker.best_depth == 2
    tracker.observe(3, None)
    assert tracker.observe(4, None)["stale"] == 3
    empty = study.Patience()
    for depth in range(1, 4):
        empty.observe(depth, None)
    assert empty.best_depth is None
    assert empty.stale == 3
    with pytest.raises(ValueError, match="finite"):
        tracker.observe(5, float("nan"))


@pytest.mark.parametrize("hardware", study.HARDWARE)
def test_validation_probabilities_match_independent_exact_channels(hardware: str) -> None:
    """The categorical validator represents precisely the existing sampler-law channel."""
    for sites in ((0,), (0, 1)):
        probabilities, operators = study.noise_outcomes(hardware, sites)
        identity = DensityMatrix.from_label("0" * len(sites))
        mixed = probabilities[0] * identity.data
        channel = probabilities[0] * SuperOp(Operator(np.eye(2 ** len(sites)))).data
        for probability, (operator, acted) in zip(probabilities[1:], operators, strict=True):
            embedded = Operator.from_label("I" * len(sites)).compose(Operator(operator), qargs=list(reversed(acted)))
            channel += probability * SuperOp(embedded).data
            mixed += probability * identity.evolve(operator, qargs=list(reversed(acted))).data
        np.testing.assert_allclose(channel, noise_channel(hardware, len(sites)).data, atol=1e-15)
        np.testing.assert_allclose(mixed, identity.evolve(noise_channel(hardware, len(sites))).data, atol=1e-15)


@pytest.mark.parametrize("hardware", study.HARDWARE)
def test_compressed_validation_matches_explicit_qiskit_paths_and_se(hardware: str) -> None:
    """No-jump multiplicities preserve both mean and raw-sample uncertainty."""
    circuit, theta, target = small_circuit()
    codes = np.array([[0, 0, 0]] * 7 + [[1, 0, 0], [0, 2, 1], [3, 7, 0], [0, 8, 2]], dtype=np.uint8)
    result = study.validation_score(circuit, theta, target, hardware, codes)
    values = []
    for row in codes:
        state = Statevector.from_label("00")
        for gate, code in zip(circuit.gates, row, strict=True):
            angle = gate.angle_scale * theta[gate.param_index] + gate.angle_offset
            state = state.evolve(PRIMITIVES[gate.name](angle), qargs=list(gate.sites))
            if code:
                _, operators = study.noise_outcomes(hardware, tuple(sorted(gate.sites)))
                operator, sites = operators[code - 1]
                state = state.evolve(operator, qargs=list(reversed(sites)))
        values.append(abs(np.vdot(target, state.data)) ** 2)
    assert result["fidelity"] == pytest.approx(np.mean(values), abs=1e-14)
    assert result["standard_error"] == pytest.approx(np.std(values, ddof=1) / np.sqrt(len(values)), abs=1e-14)
    assert result["retained_paths"] == 5
    assert result["no_jump_paths"] == 7
    assert result["jump_paths"] == 4
    assert result["norm_max_error"] < 1e-14


def test_validation_empty_bank_no_jump_limit_and_reproducible_streams() -> None:
    """Deterministic noiseless multiplicities do not invent jumps or statistical error."""
    circuit, theta, target = small_circuit()
    bank = np.zeros((100, 3), dtype=np.uint8)
    score = study.validation_score(circuit, theta, target, "IBM", bank)
    assert score["fidelity"] == pytest.approx(StatevectorBackend(circuit, target).metrics(theta)[1])
    assert score["retained_paths"] == 1
    assert score["standard_error"] == 0
    first = study.sample_validation(circuit, "IBM", 32768, [90000, 2])
    np.testing.assert_array_equal(first, study.sample_validation(circuit, "IBM", 32768, [90000, 2]))
    assert np.any(first)
    assert not np.array_equal(first, study.sample_validation(circuit, "IBM", 32768, [90000, 3]))
    empty = ParameterizedCircuit(2, [], 0)
    empty_score = study.validation_score(empty, np.empty(0), target, "IBM", np.zeros((100, 0), dtype=np.uint8))
    assert empty_score["fidelity"] == pytest.approx(abs(target[0]) ** 2)
    assert empty_score["jump_paths"] == 0
    with pytest.raises(ValueError, match="shape"):
        study.validation_score(circuit, theta, target, "IBM", np.zeros((1, 3), dtype=np.uint8))


def test_checkpoint_checksums_and_evaluation_lock(tmp_path: Path) -> None:
    """Exact scores are unavailable to selection and modified cached parameters fail."""
    circuit, theta, _ = small_circuit()
    saved = study.save_checkpoint(tmp_path, "example", circuit, theta, {"depth": 1})
    assert study.read_checkpoint(tmp_path, "example") == saved
    with pytest.raises(RuntimeError, match="locked selection"):
        study.evaluate_endpoint(tmp_path / "search", {}, np.ones(4) / 2)
    (tmp_path / "example.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        study.read_checkpoint(tmp_path, "example")


def test_construction_removes_only_stop_and_preserves_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Growth receives unmodified old parameters and full hybrid budgets beyond 90%."""
    target = np.ones(64, dtype=complex) / 8
    first = study.brickwall_for(target, 1)
    previous = np.arange(first.num_params, dtype=float) / 100
    expected = previous.copy()
    calls = []

    def fake_train(_circuit: ParameterizedCircuit, theta: np.ndarray, _target: np.ndarray, **kwargs: object) -> tuple:
        calls.append((theta.copy(), kwargs))
        return theta + 0.1, {"fidelity_trace": [0.999], "steps": 200}

    monkeypatch.setattr(study, "train", fake_train)
    study.construct(tmp_path, target, 300, 2, previous, baseline=False)
    np.testing.assert_array_equal(previous, expected)
    np.testing.assert_array_equal(calls[0][0][: len(previous)], expected)
    assert calls[0][1] == {"seed": 3300, "stop": 2.0, "steps": 200, "online": 100, "rate": 0.7}
    study.construct(tmp_path, target, 300, 2, previous, baseline=False)
    assert len(calls) == 1


@pytest.mark.parametrize("training_paths", [100, 400])
def test_candidate_preserves_actual_fine_tuning_and_separate_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, training_paths: int
) -> None:
    """A real candidate reuses exactly the old optimizer, then independently validates."""
    target = np.ones(64, dtype=complex) / 8
    circuit = study.brickwall_for(target, 1)
    initial = np.random.default_rng(4).normal(0, 0.1, circuit.num_params)
    pruned = study.save_checkpoint(tmp_path, "pruned", circuit, initial, {"depth": 1})
    monkeypatch.setitem(study.PROTOCOL, "validation_paths", 1024)
    result = study.candidate(
        tmp_path, target, 300, 0, pruned, "B", "IBM", baseline=False, training_paths=training_paths
    )
    expected, trace = study.train(
        circuit, initial, target, seed=5300, hardware="IBM", **study.protocol_for(training_paths)["fine_tuning"]
    )
    np.testing.assert_array_equal(result["theta"], expected)
    np.testing.assert_array_equal(pruned["theta"], initial)
    assert result["training"]["noise_banks"] == trace["noise_banks"]
    assert result["validation"]["sampled_paths"] == 1024
    assert result["validation"]["jump_paths"] > 0
    assert result["validation"]["entropy"][-1] == 90000
    assert "exact" not in result
    assert (
        study.candidate(tmp_path, target, 300, 0, pruned, "B", "IBM", baseline=False, training_paths=training_paths)
        == result
    )
    with pytest.raises(ValueError, match="different training-history count"):
        study.candidate(tmp_path, target, 300, 0, pruned, "B", "IBM", baseline=False, training_paths=401)
    prefix_settings: dict = {**study.PROTOCOL["fine_tuning"], "steps": 1}
    _, original = study.train(circuit, initial, target, seed=5300, hardware="IBM", **prefix_settings)
    assert trace["noise_banks"][0]["events"][:100] == original["noise_banks"][0]["events"]
    assert trace["noise_banks"][0]["sampled_paths"] == training_paths
    stored = json.loads((tmp_path / "depth_01_B_IBM.json").read_text())
    assert stored == result


def test_training_budget_is_the_only_protocol_change() -> None:
    """A larger bank preserves optimizer, pruning, validation and selection settings."""
    larger = study.protocol_for(400)
    assert larger["fine_tuning"]["num_trajectories"] == 400
    larger["fine_tuning"]["num_trajectories"] = 100
    larger["pruning"]["fine_tuning"]["num_trajectories"] = 100
    assert larger == study.PROTOCOL == study.protocol_for()
    assert study.PROTOCOL["fine_tuning"]["num_trajectories"] == 100
    for invalid in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            study.protocol_for(invalid)


def test_training_coverage_failure_is_not_resampled(tmp_path: Path) -> None:
    """The existing zero-jump policy propagates to the new candidate search."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), 0)], 1)
    target = np.array([1, 0], dtype=complex)
    pruned = study.save_checkpoint(tmp_path, "pruned", circuit, np.array([0.2]), {"depth": 1})
    result = study.candidate(tmp_path, target, 300, 0, pruned, "B", "IBM", baseline=False)
    assert result["status"] == "no_jump_training_bank"
    assert result["training"] is None
    assert result["validation"] is None
    assert result["theta"] == [0.2]


def test_cost_counts_discarded_candidates_and_failures() -> None:
    """Search totals include all construction, repair, training and validation work."""
    construction = [{"cpu_seconds": 2, "training": {"steps": 200}}] * 3
    pruning = [{"pruning": {"cpu_seconds": 3, "repair_updates": 10}}] * 3
    candidates: list[dict] = [
        {"training_cpu_seconds": 5, "validation_cpu_seconds": 7, "status": "complete", "training": {"steps": 200}}
    ] * 2
    candidates.append({
        "training_cpu_seconds": 1,
        "validation_cpu_seconds": 0,
        "status": "no_jump_training_bank",
        "training": None,
    })
    costs = study.summarize_cost(construction, pruning, candidates)
    assert costs["total_search_cpu_seconds"] == 40
    assert costs["fine_tuning_updates"] == 400
    assert costs["repair_updates"] == 30
    assert costs["failed_candidates"] == 1
