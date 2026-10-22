# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Small-system regression checks for the frozen B/G scaling pilot and its telemetry."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from benchmarks.state_preparation import load_target
from experiments import topdown_backend_support as storage
from experiments import topdown_guarded_pruning as guarded
from experiments import topdown_mpo_evaluation as mpo
from experiments import topdown_structured_q20 as pilot
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import KrotovNoiseMap, MPSBackend, ParameterizedCircuit, ParameterizedGate
from mqt.yaqs.optimization import create_brickwall_matrix_product_disentangler_parameterized_circuit as brickwall

if TYPE_CHECKING:
    from pathlib import Path


def test_host_assignments_share_parents_without_duplicating_targets() -> None:
    """Both policies belong to each unique target job; hosts have disjoint target ownership."""
    first, second = pilot.HOST_TARGETS.values()
    assert not set(first) & set(second)
    assert set(first) | set(second) == set(pilot.TARGETS)
    assert len(pilot.TARGETS) == 6
    assert pilot.STAGES.count("prepare") == 1
    assert set(pilot.STAGES) >= {"B", "G", "B_selected_mpo", "G_selected_mpo"}


def test_compact_targets_reproduce_fixed_amplitudes_and_phase() -> None:
    """Representation changes have a verified absolute vector error against the fixed target."""
    for identifier in pilot.TARGETS:
        vector = load_target(6, identifier).state_vector
        target, record = pilot.compact_target(vector, 1e-13)
        np.testing.assert_allclose(target.to_vec(), vector, atol=1e-12)
        assert record["verified_full_vector_error_norm"] < 1e-12
        assert record["pure_target_fidelity_error_bound"] == 2 * record["verified_full_vector_error_norm"]


def test_mps_guarded_pruning_preserves_the_dense_policy() -> None:
    """The optional backend changes representation without changing G's deletion decisions."""
    circuit = brickwall(3, 1, initial_single_qubit_layer=True)
    rng = np.random.default_rng(8)
    theta = rng.normal(size=circuit.num_params)
    vector = rng.normal(size=8) + 1j * rng.normal(size=8)
    vector /= np.linalg.norm(vector)
    dense_circuit, dense_theta, dense = guarded.prune(circuit, theta, vector)
    mps_circuit, mps_theta, mps = guarded.prune(
        circuit, theta, MPS.from_statevector(vector), backend_factory=MPSBackend
    )
    assert storage.circuit_record(dense_circuit) == storage.circuit_record(mps_circuit)
    np.testing.assert_equal(dense_theta, mps_theta)
    assert mps["final_fidelity"] == pytest.approx(dense["final_fidelity"], abs=1e-12)


def test_mpo_progress_contains_each_bond_and_does_not_change_fidelity() -> None:
    """Per-gate observations survive expensive steps without changing the contraction."""
    circuit = ParameterizedCircuit(
        2, [ParameterizedGate("ry", (0,), param_index=0), ParameterizedGate("rxx", (0, 1), param_index=1)]
    )
    theta = np.array([0.3, 0.8])
    events = []
    observed = mpo.evaluate(circuit, theta, MPS(2), progress=events.append)
    original = mpo.evaluate(circuit, theta, MPS(2))
    assert observed["fidelity"] == original["fidelity"]
    assert [entry["event"] for entry in events] == ["before_gate", "after_gate"] * 2
    assert len(observed["diagnostics"]["bond_growth"]) == len(circuit.gates)
    assert events[-1]["maximum_bond"] == observed["diagnostics"]["maximum_bond"]


def test_shared_preparation_smoke(tmp_path: Path) -> None:
    """The pilot can seal one shared parent and a verified target on a small fixture."""
    settings = replace(pilot.Settings(), qubits=6, max_depth=1, ideal_online=1, ideal_batch=1, product_candidates=2)
    pilot.prepare(tmp_path, "gaussian_mu0p5_sigma0p1", settings)
    saved = storage.load_checkpoint(tmp_path / "parent")
    assert saved is not None
    assert saved[0]["shared_policies"] == ["B", "G"]
    assert len(saved[0]["depths"]) == 1
    assert pilot.get_target(tmp_path)[1]["approximation"]["verified_full_vector_error_norm"] < 1e-12


def test_both_branches_and_mpo_endpoints_preserve_the_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both policies train on frozen histories and assess retained endpoints without changing shared preparation."""
    circuit = ParameterizedCircuit(
        2, [ParameterizedGate("ry", (0,), param_index=0), ParameterizedGate("rxx", (0, 1), param_index=1)]
    )
    theta = np.array([0.8, 1.1])
    quantum = QuantumCircuit(2)
    quantum.ry(theta[0], 0)
    quantum.rxx(theta[1], 0, 1)
    target = MPS.from_statevector(Statevector.from_instruction(quantum).data)
    storage.save_checkpoint(
        tmp_path / "target",
        {"qubits": 2, "approximation": {"pure_target_fidelity_error_bound": 0.0}},
        {f"site_{i}": t for i, t in enumerate(target.tensors)},
    )
    storage.save_checkpoint(tmp_path / "parent", {"circuit": storage.circuit_record(circuit)}, {"theta": theta})
    parent_seal = storage.digest(tmp_path / "parent/complete.json")

    def bank(
        self: MPSBackend, _theta: object, _options: object, _provider: object, **_kwargs: object
    ) -> list[tuple[KrotovNoiseMap, ...]]:
        rows = [tuple(KrotovNoiseMap(operators=()) for _ in self.circuit.gates) for _ in range(2)]
        rows[1] = (KrotovNoiseMap(operators=((np.array([[0, 1], [1, 0]], dtype=complex), (0,)),)), *rows[1][1:])
        return rows

    monkeypatch.setattr(MPSBackend, "sample_noise_maps", bank)
    settings = replace(pilot.Settings(), qubits=2, noisy_online=1, b_batch=1, training_paths=2)
    for policy in ("B", "G"):
        pilot.train_branch(tmp_path, policy, settings)
        selected = storage.load_checkpoint(tmp_path / policy / "selection")
        assert selected is not None
        assert selected[0]["selected_step"] >= 0
        pilot.assess(tmp_path, policy, "pruned", settings)
        pilot.assess(tmp_path, policy, "selected", settings)
        report = storage.load_checkpoint(tmp_path / policy / "mpo-selected")
        assert report is not None
        assert report[0]["diagnostics"]["exact_up_to_roundoff"]
        assert (tmp_path / policy / "mpo-selected/bond-growth.jsonl").exists()
    assert storage.digest(tmp_path / "parent/complete.json") == parent_seal


def test_stopping_rules_and_frozen_source_validation(tmp_path: Path) -> None:
    """G's stopping decisions and source integrity are independent of final assessment."""
    assert pilot.continuation_stop(0.9, 0.89, 0)[0] == "decline"
    assert pilot.continuation_stop(0.9, 0.90000001, 4)[0] == "plateau"
    assert pilot.continuation_stop(0.9, 0.91, 4) == (None, 0)
    (tmp_path / "plan.json").write_text(json.dumps({"version": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        pilot.verify(tmp_path)
    with pytest.raises(ValueError, match="settings"):
        replace(pilot.Settings(), training_paths=0)


def test_target_setup_and_mpo_norms_never_build_four_bond_transfer_tensors() -> None:
    """Large-target normalization uses the existing streaming two-index contraction."""
    vector = load_target(6, "tfim_ferro").state_vector
    with patch.object(MPS, "scalar_product", side_effect=AssertionError("Quartic transfer allocation")):
        target, diagnostics = pilot.compact_target(vector, 5e-13)
        report = mpo.evaluate(ParameterizedCircuit(6, []), np.empty(0), target)
    assert diagnostics["verified_full_vector_error_norm"] < 1e-12
    assert report["diagnostics"]["exact_up_to_roundoff"]
    assert pilot.THREAD_ENV["RAYON_NUM_THREADS"] == "1"
