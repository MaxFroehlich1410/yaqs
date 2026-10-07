# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent pruning oracles, input isolation and sealed recovery contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from experiments import topdown_pruning_screen as study
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate, StatevectorBackend

if TYPE_CHECKING:
    from pathlib import Path


def case(qubits: int = 3) -> tuple:
    """Construct complex targets, affine angles and long-range correlated noise.

    Returns:
        Circuit, independent parameters and normalized target.
    """
    specs = [("ry", (0,)), ("rz", (0,)), ("rxx", (qubits - 1, 0)), ("rx", (0,)), ("rzz", (0, qubits - 1)), ("ry", (1,))]
    circuit = ParameterizedCircuit(
        qubits,
        [
            ParameterizedGate(name, sites, param_index=i, angle_scale=-0.7, angle_offset=0.1, noise_enabled=i != 1)
            for i, (name, sites) in enumerate(specs)
        ],
    )
    rng = np.random.default_rng(82)
    target = rng.normal(size=2**qubits) + 1j * rng.normal(size=2**qubits)
    return circuit, rng.normal(size=len(specs)), target / np.linalg.norm(target)


@pytest.mark.parametrize("qubits", [2, 3, 6])
def test_exact_noisy_adjoint_matches_every_independent_density_deletion(qubits: int) -> None:
    """Each counterfactual removes the entire gate window, including correlated noise."""
    circuit, theta, target = case(qubits)
    scores, baseline = study.noisy_deletions(circuit, theta, target)
    assert baseline == pytest.approx(
        study.legacy.matrix_score(circuit, theta, target, "IBM", "tjm")["fidelity"], abs=1e-12
    )
    for index, score in enumerate(scores):
        reduced, retained = study.delete_gates(circuit, theta, [index])
        oracle = study.legacy.matrix_score(reduced, retained, target, "IBM", "tjm")
        assert score == pytest.approx(oracle["fidelity"], abs=1e-12)


def test_pair_adjoint_matches_independent_joint_deletions() -> None:
    """Pair supports and locality are frozen; scores include the intervening gates."""
    circuit, theta, target = case()
    pairs, scores, baseline = study.pair_deletions(circuit, theta, target)
    assert pairs == [[0, 1], [0, 3], [1, 3], [2, 4]]
    assert baseline == pytest.approx(study.legacy.matrix_score(circuit, theta, target, "IBM", "noiseless")["fidelity"])
    for pair, score in zip(pairs, scores, strict=True):
        reduced, retained = study.delete_gates(circuit, theta, pair)
        assert score == pytest.approx(
            study.legacy.matrix_score(reduced, retained, target, "IBM", "noiseless")["fidelity"], abs=1e-12
        )


@pytest.mark.parametrize("method", study.METHODS)
def test_methods_are_independent_and_only_repair_arm_optimizes(method: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """No caller mutation or hidden training; repair retains the best ideal parameters."""
    circuit = ParameterizedCircuit(2, [ParameterizedGate("ry", (0,), param_index=i) for i in range(3)])
    theta = np.array([0.06, 0.07, 0.08])
    target = np.array([1, 0, 0, 0], dtype=complex)
    original = study.support.circuit_record(circuit)
    calls = []
    train = study.policies.train_noiseless

    def repair(circuit: ParameterizedCircuit, theta: np.ndarray, target: np.ndarray, **options: object) -> tuple:
        calls.append(options)
        settings: dict = dict(options)
        result, trace = train(circuit, theta, target, **settings)
        assert (
            StatevectorBackend(circuit, target).metrics(result)[1]
            >= StatevectorBackend(circuit, target).metrics(theta)[1] - 1e-12
        )
        return result, trace

    monkeypatch.setattr(study.policies, "train_noiseless", repair)
    reduced, retained, trace = study.prune(circuit, theta, target, method)
    assert study.support.circuit_record(circuit) == original
    np.testing.assert_array_equal(theta, [0.06, 0.07, 0.08])
    assert bool(calls) == (method == "ideal_repair")
    assert all(settings == study.PROTOCOL["repair"] for settings in calls)
    assert len(reduced.gates) == len(retained)
    if method != "ideal_repair":
        np.testing.assert_array_equal(retained, theta[trace["remaining_origins"]])
        assert trace["repair_updates"] == 0
    if method == "sequential_noisy":
        last = study.noisy_deletions(reduced, retained, target)[1]
        assert last >= study.noisy_deletions(circuit, theta, target)[1] - 1e-12


def test_b_exact_incumbent_parity_and_simultaneous_interactions() -> None:
    """B retains the original one-pass rule; sequential rescoring can stop earlier."""
    circuit = ParameterizedCircuit(2, [ParameterizedGate("ry", (0,), param_index=i) for i in range(3)])
    theta = np.array([0.5, -0.5, 0.0])
    target = np.array([1, 0, 0, 0], dtype=complex)
    expected, values, _ = study.support.prune_b(StatevectorBackend(circuit, target), theta)
    reduced, retained, _ = study.prune(circuit, theta, target, "B")
    assert study.support.circuit_record(expected) == study.support.circuit_record(reduced)
    np.testing.assert_array_equal(values, retained)
    circuit, theta, target = case()
    expected, values, _ = study.support.prune_b(StatevectorBackend(circuit, target), theta)
    reduced, retained, _ = study.prune(circuit, theta, target, "B")
    assert study.support.circuit_record(expected) == study.support.circuit_record(reduced)
    np.testing.assert_array_equal(values, retained)


def test_magnitude_uses_effective_periodic_angle_and_input_validation() -> None:
    """Offsets/scales and full rotations count correctly; unsupported inputs fail."""
    circuit = ParameterizedCircuit(2, [ParameterizedGate("ry", (0,), param_index=0, angle_scale=2, angle_offset=0.1)])
    target = np.array([1, 0, 0, 0], dtype=complex)
    reduced, _, _ = study.prune(circuit, np.array([np.pi]), target, "magnitude")
    assert not reduced.gates
    with pytest.raises(ValueError, match="independent"):
        study.prune(circuit, np.array([np.nan]), target, "B")
    with pytest.raises(ValueError, match="Unknown"):
        study.prune(circuit, np.zeros(1), target, "hybrid")
    with pytest.raises(ValueError, match="six qubits"):
        study.noisy_deletions(ParameterizedCircuit(7, []), np.empty(0), np.ones(128) / np.sqrt(128))


def test_sealed_method_recovery_and_cross_method_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume assessment without pruning again; reject changed inputs and damaged seals."""
    circuit, theta, target = case(2)
    folder = tmp_path / "input"
    folder.mkdir()
    (folder / "input.json").write_text("{}")
    built = {"circuit": study.support.circuit_record(circuit)}
    receipt = {"target_id": "synthetic", "seed": 300, "selected_depth": 1}
    monkeypatch.setattr(study.parents, "authenticate_input", lambda _: (receipt, built, {"theta": theta}, target))
    result_folder = tmp_path / "result"
    result = study.run_method(result_folder, folder, "magnitude")
    assert result["new_construction_updates"] == result["noisy_fine_tuning_updates"] == result["repair_updates"] == 0
    assert result["ideal_replay_error"] < 1e-12
    monkeypatch.setattr(study, "prune", lambda *_: pytest.fail("Cached pruning must be reused."))
    assert study.run_method(result_folder, folder, "magnitude") == result
    (result_folder / "final/complete.json").unlink()
    assert study.run_method(result_folder, folder, "magnitude")["ideal_pruned"] == result["ideal_pruned"]
    with pytest.raises(ValueError, match="Changed method"):
        study.run_method(result_folder, folder, "pair")
    (result_folder / "pruning/arrays.npz").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        study.run_method(result_folder, folder, "magnitude")
