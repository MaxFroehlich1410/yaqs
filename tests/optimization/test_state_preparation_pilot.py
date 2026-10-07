# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent channel, pruning, gradient, and resumable multi-target pilot tests."""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Pauli, Statevector, SuperOp, average_gate_fidelity
from scipy.linalg import expm

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.pilot_components import (
    ExactLocalPauliProvider,
    delete_gates,
    exact_density_fidelity,
    pruning_scores,
    select_deletions,
    with_origins,
)
from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_pilot as pilot
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    GateNoiseContext,
    KrotovTJMOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    ParameterizedGate,
    noisy_state_preparation_contribution,
    noisy_state_preparation_metrics,
    state_preparation_metrics,
)

if TYPE_CHECKING:
    from pathlib import Path


class FixedDraws:
    """Deterministic uniforms to enumerate every sampler branch exactly."""

    def __init__(self, bits: tuple[int, ...]) -> None:
        """Convert branch indicators into deterministic uniform draws."""
        self.draws = iter(0.0 if bit else 1.0 for bit in bits)

    def random(self) -> float:
        """Return the next predetermined branch draw."""
        return next(self.draws)


@pytest.mark.parametrize("arity", [1, 2])
def test_exact_factor_sampler_matches_independent_lindblad_exponential(arity: int) -> None:
    """Enumerate the actual provider and recover the calibrated exact channel."""
    provider = ExactLocalPauliProvider()
    strength = provider.calibration.one_qubit_strength if arity == 1 else provider.calibration.two_qubit_strength
    probability = -np.expm1(-2 * strength) / 2
    dimension = 2**arity
    channel = np.zeros((dimension**2, dimension**2), dtype=complex)
    context = GateNoiseContext(0, "rzz" if arity == 2 else "rz", tuple(range(arity)), arity, 0.3, 0, 0, 0)
    for bits in itertools.product((0, 1), repeat=3 * arity):
        instruction = provider(context, cast("np.random.Generator", FixedDraws(bits)))
        unitary = np.eye(dimension, dtype=complex)
        for local in instruction.operators:
            full = Operator(np.eye(dimension)).compose(Operator(local.matrix), qargs=list(local.sites)).data
            unitary = full @ unitary
        weight = probability ** sum(bits) * (1 - probability) ** (len(bits) - sum(bits))
        channel += weight * np.kron(unitary.conj(), unitary)
    generator = np.zeros_like(channel)
    for site in range(arity):
        for label in "XYZ":
            pauli = Operator(np.eye(dimension)).compose(Operator(Pauli(label)), qargs=[site]).data
            generator += strength * (np.kron(pauli.conj(), pauli) - np.eye(dimension**2))
    np.testing.assert_allclose(channel, expm(generator), atol=1e-12, rtol=0)
    error = provider.calibration.one_qubit_error if arity == 1 else provider.calibration.two_qubit_error
    assert 1 - average_gate_fidelity(SuperOp(channel)) == pytest.approx(error, abs=1e-12)


def _dense(circuit: ParameterizedCircuit, theta: np.ndarray, removed: tuple[int, ...] = ()) -> np.ndarray:
    """Return an independent dense state, optionally skipping original gates."""
    state = Statevector.from_int(0, 2**circuit.num_qubits)
    for index, gate in enumerate(circuit.gates):
        if index not in removed:
            matrix, sites = circuit.gate_matrix(gate, theta)
            state = state.evolve(Operator(matrix), qargs=list(reversed(sites)))
    return state.data


def test_pruning_quotas_protected_origins_and_dense_parameter_remapping() -> None:
    """Six simultaneous deletion rounds reach the compact resource counts exactly."""
    circuit = pilot.pilot_circuit(4)
    assert circuit.num_params == 198
    theta = np.random.default_rng(20).normal(0, 0.2, circuit.num_params)
    target = load_target(6, pilot.TARGETS[0]).state_vector_copy()
    for _ in range(6):
        scores = pruning_scores(circuit, theta, target, "M")
        selected = select_deletions(scores, 6)
        assert sum(len(circuit.gates[index].sites) == 1 for index in selected) == 10
        assert sum(len(circuit.gates[index].sites) == 2 for index in selected) == 5
        expected = _dense(circuit, theta, tuple(selected))
        circuit, theta = delete_gates(circuit, theta, selected)
        np.testing.assert_allclose(_dense(circuit, theta), expected, atol=1e-11, rtol=0)
        assert set(range(18)).issubset({g.logical_gate_id for g in circuit.gates})
    assert circuit.num_params == 108
    assert sum(len(g.sites) == 1 for g in circuit.gates) == 78
    assert sum(len(g.sites) == 2 for g in circuit.gates) == 30
    restored = pilot.restore_circuit(json.loads(json.dumps(pilot.circuit_record(circuit))))
    np.testing.assert_array_equal(_dense(restored, theta), _dense(circuit, theta))


def test_deletion_scores_match_dense_loss_with_angle_offsets_and_scales() -> None:
    """Damage is actual unitary removal, not a zero-parameter approximation."""
    circuit = pilot.pilot_circuit(1)
    circuit.gates[0] = replace(circuit.gates[0], angle_scale=-1.7, angle_offset=0.4)
    theta = np.random.default_rng(21).normal(0, 0.2, circuit.num_params)
    target = load_target(6, "haar_random_1").state_vector_copy()
    scores = pruning_scores(circuit, theta, target, "D")
    original_fidelity = abs(np.vdot(target, _dense(circuit, theta))) ** 2
    for row in scores:
        removed_fidelity = abs(np.vdot(target, _dense(circuit, theta, (int(row["index"]),)))) ** 2
        assert row["score"] == pytest.approx(original_fidelity - removed_fidelity, abs=1e-10)
    assert select_deletions(scores, 6) == select_deletions(list(reversed(scores)), 6)


@pytest.mark.parametrize("target_id", pilot.TARGETS)
def test_noiseless_density_oracle_matches_mps_for_every_pilot_family(target_id: str) -> None:
    """The evaluation-only oracle follows the exact BMPD little-endian convention."""
    circuit = pilot.pilot_circuit(1)
    theta = np.random.default_rng(22).normal(0, 0.3, circuit.num_params)
    target = load_target(6, target_id).state_vector_copy()
    zero = ExactLocalPauliProvider(CalibratedLocalDepolarizingProvider(0.0, 0.0))
    expected = state_preparation_metrics(circuit, theta, target)[1]
    assert exact_density_fidelity(circuit, theta, target, zero) == pytest.approx(expected, abs=1e-10)


def test_deleted_zero_angle_gate_loses_its_noise_window() -> None:
    """A retained zero angle remains noisy; deleting it removes the channel."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("rz", (0,), param_index=0)])
    target = np.array([1, 0], dtype=complex)
    provider = ExactLocalPauliProvider()
    noisy = exact_density_fidelity(circuit, np.array([0.0]), target, provider)
    deleted, theta = delete_gates(circuit, np.array([0.0]), [0])
    assert noisy < 1
    assert exact_density_fidelity(deleted, theta, target, provider) == 1


def test_independent_exact_noise_gradient_matches_fixed_map_finite_differences() -> None:
    """The pure-trajectory training direction differentiates its sampled loss."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), param_index=0, angle_scale=-1.5, angle_offset=0.2),
            ParameterizedGate("rxx", (0, 1), param_index=1),
            ParameterizedGate("rz", (1,), param_index=2),
        ],
    )
    rng = np.random.default_rng(23)
    theta = rng.normal(size=3)
    target = rng.normal(size=4) + 1j * rng.normal(size=4)
    target /= np.linalg.norm(target)
    provider = ExactLocalPauliProvider(CalibratedLocalDepolarizingProvider(0.15, 0.3))
    options = KrotovTJMOptions(num_trajectories=8, random_seed=24, trajectory_update="independent")
    gradient, _loss, _fidelity, paths = noisy_state_preparation_contribution(
        circuit, theta, target, None, options, MPS(2), KrotovTruncation(), noise_provider=provider
    )
    maps = [path.noise_maps for path in paths]
    for index in range(3):
        plus, minus = theta.copy(), theta.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        upper = noisy_state_preparation_metrics(circuit, plus, target, None, options, fixed_noise_maps=maps)[0]
        lower = noisy_state_preparation_metrics(circuit, minus, target, None, options, fixed_noise_maps=maps)[0]
        assert gradient[index] == pytest.approx((upper - lower) / 2e-6, abs=2e-6)


def test_invalid_pruning_and_configuration_are_rejected() -> None:
    """Malformed requests cannot silently alter the target panel or circuit."""
    with pytest.raises(ValueError, match="budgets"):
        pilot.PilotConfig(training_trajectories=0)
    with pytest.raises(ValueError, match="seeds"):
        pilot.PilotConfig(seeds=(11, 11))
    circuit = pilot.pilot_circuit(1)
    with pytest.raises(ValueError, match="deletion"):
        delete_gates(circuit, np.zeros(63), [0, 0])
    with pytest.raises(ValueError, match="unique scalar"):
        with_origins(ParameterizedCircuit(1, [ParameterizedGate("x", (0,))]))


def test_interrupted_noisy_block_resumes_exactly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial checkpoint resumes the same sampled updates without duplication."""
    circuit = ParameterizedCircuit(6, [ParameterizedGate("ry", (0,), param_index=0)])
    theta = np.array([0.2])
    target = load_target(6, pilot.TARGETS[0]).state_vector_copy()
    write_record = pilot.write_record

    def interrupted(path: Path, record: dict[str, object]) -> None:
        """Simulate an interruption after the first checkpoint was persisted.

        Raises:
            RuntimeError: Immediately after persisting the checkpoint.
        """
        write_record(path, record)
        msg = "Simulated interruption"
        raise RuntimeError(msg)

    checkpoint = tmp_path / "interrupted.json"
    with monkeypatch.context() as patch:
        patch.setattr(pilot, "write_record", interrupted)
        with pytest.raises(RuntimeError, match="Simulated interruption"):
            pilot.optimize(checkpoint, circuit, theta, target, updates=20, rate=0.1, seed=30, trajectories=2)
    assert pilot.read_record(checkpoint)["completed_updates"] == 10
    resumed = pilot.optimize(checkpoint, circuit, theta, target, updates=20, rate=0.1, seed=30, trajectories=2)
    fresh = pilot.optimize(
        tmp_path / "fresh.json", circuit, theta, target, updates=20, rate=0.1, seed=30, trajectories=2
    )
    assert resumed["theta"] == fresh["theta"]
    assert resumed["trace"] == fresh["trace"]
    assert resumed["completed_updates"] == 20
    with pytest.raises(ValueError, match="configuration mismatch"):
        pilot.optimize(checkpoint, circuit, theta, target, updates=20, rate=0.2, seed=30, trajectories=2)


def test_corrupt_artifact_and_invalid_parallel_request_are_rejected(tmp_path: Path) -> None:
    """Corrupt JSON is rejected and invalid workers do not create output."""
    checkpoint = tmp_path / "checkpoint.json"
    pilot.write_record(checkpoint, {"theta": [1.0]})
    payload = json.loads(checkpoint.read_text())
    payload["theta"] = [2.0]
    checkpoint.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum mismatch"):
        pilot.read_record(checkpoint)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="workers"):
        pilot.run_pilot(output, workers=0)
    assert not output.exists()


def test_reduced_multitarget_pilot_parallel_equivalence_and_resume(tmp_path: Path) -> None:
    """All four target families complete both stages identically across workers."""
    config = pilot.PilotConfig(
        source_updates=1, repair_updates=1, refinement_updates=1, training_trajectories=1, seeds=(11,)
    )
    serial = pilot.run_pilot(tmp_path / "serial", config=config, workers=1)
    parallel = pilot.run_pilot(tmp_path / "parallel", config=config, workers=2)
    assert serial["nominee"] == parallel["nominee"]
    assert serial["construction"] == parallel["construction"]
    assert serial["refinement"] == parallel["refinement"]
    assert serial["completed_target_seed_jobs"] == 4
    for target_id in pilot.TARGETS:
        root = tmp_path / "parallel" / target_id / "seed_11"
        construction = pilot.read_record(root / "construction.json")
        assert construction["construction_updates_per_route"] == 10
        rows = cast("list[dict[str, object]]", pilot.read_record(root / "refinement.json")["candidates"])
        assert len(rows) == 6
        for index in range(0, 6, 2):
            assert rows[index]["initial_theta_checksum"] == rows[index + 1]["initial_theta_checksum"]
    assert pilot.run_pilot(tmp_path / "parallel", config=config, workers=1, resume=True) == parallel
    with pytest.raises(ValueError, match="identical manifest"):
        pilot.run_pilot(tmp_path / "parallel", config=replace(config, root_seed=25), resume=True)
    with pytest.raises(ValueError, match="explicit resume"):
        pilot.run_pilot(tmp_path / "parallel", config=config)
