# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact deletion, cumulative damage, repair, resource and frozen-policy checks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from benchmarks.state_preparation.pilot_components import delete_gates
from experiments import legacy_pruning_ablation as study
from experiments.legacy_topdown_dense import matrix_score, prune
from experiments.legacy_topdown_q6 import brickwall_for
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate, StatevectorBackend

if TYPE_CHECKING:
    from pathlib import Path


def rotations(angles: list[float]) -> tuple[ParameterizedCircuit, np.ndarray]:
    """Build consecutive independent Y rotations on one qubit.

    Returns:
        The circuit and its independent scalar parameters.
    """
    return ParameterizedCircuit(
        1, [ParameterizedGate("ry", (0,), i) for i in range(len(angles))], len(angles)
    ), np.array(angles)


def test_stationary_essential_gate_and_periodic_invariance() -> None:
    """A perfect rotation has zero gradient but unit removal damage."""
    circuit, theta = rotations([np.pi])
    target = np.array([0, 1], dtype=complex)
    assert abs(StatevectorBackend(circuit, target).gradient(theta)[0][0]) < 1e-15
    removed, fidelity = study.deletion_fidelities(circuit, theta, target)
    assert fidelity == pytest.approx(1)
    assert removed[0] == pytest.approx(0)
    for angle in (0.8, 0.8 + 2 * np.pi):
        scores, fidelity = study.deletion_fidelities(circuit, np.array([angle]), target)
        assert fidelity - scores[0] == pytest.approx(np.sin(0.4) ** 2)


@pytest.mark.parametrize("seed", [19, 41, 77])
def test_shared_adjoint_equals_exhaustive_qiskit_deletion(seed: int) -> None:
    """Check every finite removal using an independent quantum library."""
    rng = np.random.default_rng(seed)
    circuit = ParameterizedCircuit(
        3,
        [
            ParameterizedGate("ry", (2,), 0),
            ParameterizedGate("rxx", (2, 0), 1, angle_scale=-0.7, angle_offset=0.3),
            ParameterizedGate("rz", (1,), None, angle_offset=0.7),
            ParameterizedGate("ryy", (1, 0), 2),
            ParameterizedGate("rx", (0,), 0),
            ParameterizedGate("rzz", (0, 2), 3),
        ],
        4,
    )
    theta = rng.normal(size=4)
    target = rng.normal(size=8) + 1j * rng.normal(size=8)
    target /= np.linalg.norm(target)
    removed, fidelity = study.deletion_fidelities(circuit, theta, target)
    assert fidelity == pytest.approx(matrix_score(circuit, theta, target, "IBM", "noiseless")["fidelity"], abs=1e-13)
    for index, score in enumerate(removed):
        reduced, values = delete_gates(circuit, theta, [index])
        assert score == pytest.approx(matrix_score(reduced, values, target, "IBM", "noiseless")["fidelity"], abs=1e-13)


def test_score_uses_one_matrix_per_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scorer shares circuit propagation across candidates."""
    circuit, theta = rotations([0.2, 0.3, 0.4, 0.5])
    original = ParameterizedCircuit.gate_matrix
    calls = []

    def counted(self: ParameterizedCircuit, gate: ParameterizedGate, values: np.ndarray) -> tuple:
        calls.append(gate.name)
        return original(self, gate, values)

    monkeypatch.setattr(ParameterizedCircuit, "gate_matrix", counted)
    study.deletion_fidelities(circuit, theta, np.array([1, 0], dtype=complex))
    assert len(calls) == len(circuit.gates)


@pytest.mark.parametrize("permutation", [[0, 1, 2], [2, 0, 1]])
def test_baseline_retains_literal_pruning(permutation: list[int]) -> None:
    """A reproduces the old cutoff and parameter order exactly."""
    circuit, theta = rotations([0.3, 0.9, -0.2])
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), i) for i in permutation], 3)
    target = np.array([1, 0], dtype=complex)
    impact = abs(theta * StatevectorBackend(circuit, target).gradient(theta)[0])
    expected, values = prune(circuit, theta, impact)
    actual, updated, trace = study.prune_variant(circuit, theta, target, "A")
    assert actual.gates == expected.gates
    np.testing.assert_array_equal(updated, values)
    assert trace["repair_updates"] == 0
    assert trace["score_calls"] == 1


def test_simultaneous_damage_sequential_rescoring_and_repair() -> None:
    """B compounds small losses; C stops; D repairs before rescoring."""
    circuit, theta = rotations([0.18, 0.18])
    target = np.array([np.cos(0.18), np.sin(0.18)], dtype=complex)
    outputs = {variant: study.prune_variant(circuit, theta, target, variant) for variant in ("B", "C", "D")}
    assert not outputs["B"][0].gates
    assert outputs["B"][2]["final_noiseless_fidelity"] < 0.97
    assert len(outputs["C"][0].gates) == len(outputs["D"][0].gates) == 1
    assert outputs["C"][2]["final_noiseless_fidelity"] > 0.99
    assert outputs["D"][2]["final_noiseless_fidelity"] > outputs["C"][2]["final_noiseless_fidelity"]
    assert outputs["D"][2]["repair_updates"] == 10
    assert outputs["D"][2]["repair_parameter_updates"] == 10
    repair = outputs["D"][2]["iterations"][0]["repair"]
    assert repair["online_steps"] == 5
    assert repair["rate"] == pytest.approx(0.7)
    assert not repair["noisy"]
    assert outputs["C"][2]["iterations"][-1]["scores"][0] >= 0.01


def test_negative_damage_is_eligible_and_empty_circuits_are_supported() -> None:
    """Deletion may improve fidelity; removing the last gate needs no repair."""
    circuit, theta = rotations([0.4])
    target = np.array([1, 0], dtype=complex)
    for variant in ("B", "C", "D"):
        reduced, values, trace = study.prune_variant(circuit, theta, target, variant)
        assert not reduced.gates
        assert not len(values)
        assert trace["iterations"][0]["scores"][0] < 0
        assert trace["repair_updates"] == 0
        scores, fidelity = study.deletion_fidelities(reduced, values, target)
        assert not len(scores)
        assert fidelity == pytest.approx(1)


def test_cutoff_equality_is_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strictly below 0.01 is the predeclared removal condition."""
    circuit, theta = rotations([0.1])
    target = np.array([1, 0], dtype=complex)
    monkeypatch.setattr(study, "deletion_fidelities", lambda *_args: (np.array([0.0]), 0.01))
    for variant in ("B", "C", "D"):
        reduced, _, trace = study.prune_variant(circuit, theta, target, variant)
        assert len(reduced.gates) == 1
        assert not trace["iterations"][0]["removed_origins"]


def test_reference_compilation_preserves_unitary_and_parallel_depth() -> None:
    """Disjoint CX decompositions run in parallel and empty depth is zero."""
    circuit = ParameterizedCircuit(4, [ParameterizedGate("rxx", (0, 1), 0), ParameterizedGate("ryy", (2, 3), 1)], 2)
    _, metrics = study.compile_resources(circuit, np.array([0.37, 0.61]))
    assert metrics["logical_two_qubit"] == 2
    assert metrics["logical_two_qubit_depth"] == 1
    assert metrics["compiled_two_qubit"] == 4
    assert metrics["compiled_two_qubit_depth"] == 2
    assert metrics["logical_unitary_max_error_up_to_global_phase"] < 1e-12
    _, empty = study.compile_resources(ParameterizedCircuit(4, [], 0), np.empty(0))
    assert empty["compiled_two_qubit"] == empty["compiled_two_qubit_depth"] == 0


def test_fine_tuning_matches_frozen_training_and_keeps_coverage_failure(tmp_path: Path) -> None:
    """Actual q6 maps, cross updates and independent density scores remain fixed."""
    target = Statevector.from_instruction(QuantumCircuit(6)).data
    circuit = brickwall_for(target, 1)
    theta = np.random.default_rng(2).normal(0, 0.1, circuit.num_params)
    store = study.Checkpoints(tmp_path / "case", target)
    store.save("B_initial", circuit, theta, {})
    result = study.fine_tune(store, "B", "IBM", 300)
    assert result["status"] == "complete"
    assert result["jump_paths"] > 0
    _, final, record = store.load("B_IBM")
    expected, old = study.train(circuit, theta, target, hardware="IBM", seed=5300, **study.PROTOCOL["fine_tuning"])
    np.testing.assert_array_equal(final, expected)
    assert record["training"]["noise_banks"] == old["noise_banks"]
    assert study.fine_tune(store, "B", "IBM", 300)["fidelity"] == result["fidelity"]
    short, values = rotations([0.2])
    failure = study.Checkpoints(tmp_path / "failure", np.array([1, 0], dtype=complex))
    failure.save("B_initial", short, values, {})
    missing = study.fine_tune(failure, "B", "IBM", 300)
    assert missing["status"] == "no_jump_bank"
    assert missing["fidelity"] is None
    assert json.loads((failure.root / "B_IBM.failure.json").read_text())["jump_paths"] == 0
    assert not failure.exists("B_IBM")


def test_invalid_policy_and_parameters_fail() -> None:
    """Reject ambiguous policies and malformed parameter vectors."""
    circuit, theta = rotations([0.2])
    target = np.array([1, 0], dtype=complex)
    with pytest.raises(ValueError, match="Unknown pruning"):
        study.prune_variant(circuit, theta, target, "E")
    with pytest.raises(ValueError, match="finite real parameters"):
        study.deletion_fidelities(circuit, np.array([np.nan]), target)


def test_complete_resume_authenticates_artifacts(tmp_path: Path) -> None:
    """A completed job cannot silently reuse modified results."""
    target_id = "gaussian_mu0p5_sigma0p1"
    folder = target_id + "__seed300"
    source = study.Checkpoints(
        tmp_path / "source" / folder, study.load_target_collection().load(6, target_id).state_vector_copy()
    )
    root = tmp_path / "output" / folder
    root.mkdir(parents=True)
    artifact = root / "artifact.json"
    artifact.write_text("{}")
    saved = {"artifacts": {artifact.name: study.file_hash(artifact)}}
    study.write_json(root / "complete.json", saved)
    job = (str(root.parent), str(source.root.parent), target_id, 300)
    assert study.run_target(job) == saved
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="artifacts changed"):
        study.run_target(job)
