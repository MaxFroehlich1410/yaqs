# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Fixed-bank convergence diagnostics, exact replay and independent probe output."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from experiments import legacy_topdown_dense as legacy
from experiments import topdown_backend_probe as probe
from experiments import topdown_backend_support as support
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import KrotovNoiseMap, MPSBackend, ParameterizedCircuit, ParameterizedGate
from mqt.yaqs.optimization.mps_backend import mps_overlap
from tests.optimization.test_mps_backend import maps_for
from tests.optimization.test_statevector import case

if TYPE_CHECKING:
    from pathlib import Path

    from mqt.yaqs.optimization.trajectories import Estimator


def test_probe_checks_exact_cross_parity_and_bond_convergence() -> None:
    """A sufficient cap recovers dense signals and finite deletion scores."""
    circuit, theta, target = case(3)
    report = probe.compare_backends(
        circuit, theta, target, maps_for(circuit), np.array([0.9, 0.07, 0.03]), bond_caps=(1, 2, 4)
    )
    assert report["exact_parity_passed"]
    assert report["runs"][-1]["cross_signal_max_error"] < 1e-10
    assert report["runs"][1]["diagnostics"]["cap_reductions"] > 0
    assert report["runs"][1]["diagnostics"]["discarded_weight_sum"] > 0
    assert report["runs"][0]["resources"]["traced_peak_bytes"] > 0
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match="caps"):
        probe.compare_backends(circuit, theta, target, maps_for(circuit), np.array([0.9, 0.07, 0.03]), bond_caps=(0,))


def test_empty_probe() -> None:
    """An empty pruned circuit has well-defined zero signal/deletion errors."""
    circuit = ParameterizedCircuit(2, [])
    target = np.array([1, 0, 0, 0], dtype=complex)
    report = probe.compare_backends(circuit, np.empty(0), target, [()], np.ones(1), bond_caps=())
    assert report["exact_parity_passed"]


def test_recorded_bank_replay_and_corruption_rejection() -> None:
    """Reconstruction reproduces empirical weights without calling a sampler."""
    circuit, theta, target = case(2)
    _, trace = legacy.train(
        circuit,
        theta,
        target,
        steps=0,
        rate=0.2,
        seed=5300,
        noisy=True,
        num_trajectories=100,
        compress_no_jump=True,
        record_noise_banks=True,
    )
    bank = trace["noise_banks"][0]
    maps, weights = probe.replay_training_bank(circuit, bank)
    assert len(maps) == bank["retained_paths"]
    np.testing.assert_array_equal(weights, bank["weights"])
    corrupt = copy.deepcopy(bank)
    corrupt["raw_sha256"] = "invalid"
    with pytest.raises(ValueError, match="checksum"):
        probe.replay_training_bank(circuit, corrupt)
    corrupt = copy.deepcopy(bank)
    corrupt["multiplicities"][0] += 1
    with pytest.raises(ValueError, match="multiplicities"):
        probe.replay_training_bank(circuit, corrupt)


def test_two_index_overlap_matches_dense_complex_states() -> None:
    """No four-bond transfer tensors are needed even for arbitrary complex states."""
    rng = np.random.default_rng(713)
    a, b = rng.normal(size=(2, 64)) + 1j * rng.normal(size=(2, 64))
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    left, right = MPS.from_statevector(a), MPS.from_statevector(b)
    assert mps_overlap(left, right) == pytest.approx(np.vdot(a, b), abs=1e-12)
    with pytest.raises(ValueError, match="register"):
        mps_overlap(MPS(2), MPS(3))


def test_probe_cli_references_sealed_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The standalone diagnostic reads original maps and writes a separate report."""
    circuit = ParameterizedCircuit(2, [ParameterizedGate("ry", (0,), param_index=0)])
    target = np.array([1, 0, 0, 0], dtype=complex)
    theta, trace = legacy.train(
        circuit,
        np.array([0.4]),
        target,
        steps=0,
        rate=0.2,
        seed=5300,
        noisy=True,
        num_trajectories=100,
        compress_no_jump=True,
        record_noise_banks=True,
    )
    support.write_json(tmp_path / "manifest.json", {"target_id": "test", "config": {"qubits": 2}})
    support.save_checkpoint(tmp_path / "final", {"selected_depth": 1}, {})
    support.save_checkpoint(tmp_path / "depth_01/pruning", {"circuit": support.circuit_record(circuit)}, {})
    support.save_checkpoint(tmp_path / "depth_01/refinement", {"training": trace}, {"theta": theta})
    record = SimpleNamespace(state_vector_copy=target.copy)
    monkeypatch.setattr(probe, "load_target_collection", lambda: SimpleNamespace(load=lambda *_a: record))
    output = tmp_path / "probe.json"
    probe.main(["--case-root", str(tmp_path), "--output", str(output), "--bond-caps", "2"])
    report = json.loads(output.read_text())
    assert report["exact_parity_passed"]
    assert report["source_checkpoint_sha256"] == support.digest(tmp_path / "depth_01/refinement/complete.json")


def test_malformed_maps_and_parameters_leave_no_tape() -> None:
    """Rejected replay and failed numerical calls cannot leak a retained tape."""
    c, theta, target = case(2)
    engine = MPSBackend(c, target)
    for bad in (np.full_like(theta, np.nan), theta[:-1], theta.astype(complex)):
        with pytest.raises(ValueError, match="parameters"):
            engine.gradient(bad)
    with pytest.raises(ValueError, match="estimator"):
        engine.gradient(theta, estimator=cast("Estimator", "bad"))
    with pytest.raises(ValueError, match="ensemble"):
        engine.metrics(theta, [])
    invalid = list(maps_for(c)[0])
    invalid[0] = KrotovNoiseMap(normalized=True)
    with pytest.raises(NotImplementedError, match=r"normalization|Normalized"):
        engine.gradient(theta, [invalid])
    with pytest.raises(ValueError, match="Step size"):
        engine.online_update(theta, -1)
    assert engine.retained_state_count == 0
