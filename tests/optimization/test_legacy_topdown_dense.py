# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent matrix, pruning, and historical optimizer verification."""

from __future__ import annotations

import json
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit.quantum_info import DensityMatrix

from experiments import legacy_topdown_dense
from experiments.legacy_topdown_dense import (
    LegacyNoise,
    local_model,
    matrix_score,
    noise_channel,
    prune,
    target_state,
    train,
)
from mqt.yaqs.optimization import (
    KrotovNoiseMap,
    KrotovOptions,
    KrotovTJMOptions,
    ParameterizedCircuit,
    ParameterizedGate,
    StatevectorBackend,
    create_brickwall_matrix_product_disentangler_parameterized_circuit,
    train_krotov_noisy_state_preparation_batch,
)

if TYPE_CHECKING:
    from pathlib import Path


def case() -> tuple:
    """Return a nontrivial two-qubit circuit with signed affine angles."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), param_index=0),
            ParameterizedGate("rz", (1,), param_index=1, angle_scale=-1.0),
            ParameterizedGate("rxx", (0, 1), param_index=2),
            ParameterizedGate("ry", (1,), param_index=3),
        ],
    )
    return circuit, np.array([0.4, -0.7, 0.3, 0.9]), target_state(19, "haar", 2)


def test_qiskit_noiseless_and_parameter_shift() -> None:
    """Independent Qiskit propagation and parameter shifts agree with the adjoint."""
    circuit, theta, target = case()
    engine = StatevectorBackend(circuit, target)
    gradient, _, fidelity = engine.gradient(theta)
    assert matrix_score(circuit, theta, target, "IBM", "noiseless")["fidelity"] == pytest.approx(fidelity, abs=1e-13)
    for index in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[index] += np.pi / 2
        minus[index] -= np.pi / 2
        shifted = (engine.metrics(plus)[1] - engine.metrics(minus)[1]) / 2
        assert gradient[index] == pytest.approx(-shifted, abs=1e-13)


@pytest.mark.parametrize("crn", [False, True])
def test_historical_batch_matches_mps(*, crn: bool) -> None:
    """Replay seed schedules and cross updates match the production MPS loop."""
    circuit, theta, target = case()
    provider = LegacyNoise()
    options = KrotovTJMOptions(num_trajectories=3, trajectory_update="cross", use_crn=crn)
    engine = StatevectorBackend(circuit, target)
    seed = next(
        value
        for value in range(1000)
        if any(m.operators for row in engine.sample_noise_maps(theta, options, provider, iteration=value) for m in row)
    )
    result = train_krotov_noisy_state_preparation_batch(
        circuit,
        target,
        local_model("IBM", (0, 1)),
        KrotovTJMOptions(num_trajectories=3, trajectory_update="cross", use_crn=crn),
        initial_theta=theta,
        options=KrotovOptions(max_iterations=3, batch_step_size=0.2, batch_schedule="exp", batch_decay=0.01, seed=seed),
    )
    actual, trace = train(circuit, theta, target, steps=3, rate=0.2, decay=0.01, seed=seed, noisy=True, crn=crn)
    np.testing.assert_allclose(actual, result.theta, atol=1e-12)
    np.testing.assert_allclose(
        trace["fidelity_trace"], np.asarray(result.trace["fidelity"][1:], dtype=float), atol=1e-12
    )


def test_literal_pruning_boundary_and_stationary_failure() -> None:
    """The cutoff deletes complete gates/noise windows, including all at a stationary point."""
    circuit, theta, target = case()
    pruned, values = prune(circuit, theta, np.array([0.0, 0.01, 0.009, 0.02]))
    assert pruned.num_params == 2
    assert [gate.param_index for gate in pruned.gates] == [0, 1]
    np.testing.assert_array_equal(values, theta[[1, 3]])
    empty, values = prune(circuit, theta, np.zeros(4))
    assert not empty.gates
    assert values.size == 0
    assert matrix_score(empty, values, target, "IBM", "tjm")["fidelity"] == pytest.approx(abs(target[0]) ** 2)


@pytest.mark.parametrize(("hardware", "arity"), product(("IBM", "Google"), (1, 2)))
def test_channel_cptp_and_sampling(hardware: str, arity: int) -> None:
    """Analytic local channels agree with the trajectory law within sampling error."""
    for law in ("tjm", "lindblad"):
        assert noise_channel(hardware, arity, law).is_cptp(atol=1e-12)
    # Exact identity probability is independent of the state or gate angle.
    target = np.zeros(2**arity, dtype=complex)
    target[0] = 1
    rho = DensityMatrix(target).evolve(noise_channel(hardware, arity))
    one, two = (3e-4, 3e-3) if hardware == "IBM" else (6e-4, 5e-3)
    total = one if arity == 1 else 2 * one + two
    nontrivial = 2 * one / 3 if arity == 1 else 4 * one / 3 + two * (0.5 if hardware == "IBM" else 2 / 3)
    expected = 1 + np.expm1(-total) * nontrivial / total
    assert rho.data[0, 0].real == pytest.approx(expected, abs=1e-14)


def test_hybrid_stops_at_threshold() -> None:
    """The literal arm checks the fidelity after each update, before pruning."""
    circuit, theta, target = case()
    result, trace = train(circuit, theta, target, steps=10, online=5, rate=0.1, seed=1, stop=0)
    assert trace["steps"] == 1
    assert trace["online_steps"] == 1
    assert np.all(np.isfinite(result))


def test_exact_enumerated_channel_and_gradient() -> None:
    """Enumerating every jump at one gate matches independent noisy matrix differentiation."""
    circuit, theta, target = case()
    for index, gate in enumerate(circuit.gates):
        gate.noise_enabled = index == 2
    model = local_model("IBM", (0, 1))
    rates = np.array([float(process["strength"]) for process in model.processes])
    total = rates.sum()
    weights = np.concatenate(([np.exp(-total)], -np.expm1(-total) * rates / total))
    rows = [[KrotovNoiseMap() for _ in circuit.gates]]
    for process in model.processes:
        matrix = process["matrix"] if "matrix" in process else np.kron(*process["factors"])
        row = [KrotovNoiseMap() for _ in circuit.gates]
        row[2] = KrotovNoiseMap(operators=((matrix, tuple(process["sites"])),))
        rows.append(row)
    gradient, _, fidelity = StatevectorBackend(circuit, target).gradient(theta, rows, weights=weights)
    assert fidelity == pytest.approx(matrix_score(circuit, theta, target, "IBM", "tjm")["fidelity"], abs=1e-13)
    for index in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        difference = (
            matrix_score(circuit, minus, target, "IBM", "tjm")["fidelity"]
            - matrix_score(circuit, plus, target, "IBM", "tjm")["fidelity"]
        ) / 2e-5
        assert gradient[index] == pytest.approx(difference, abs=1e-10)


def test_baseline_workflow_retains_reproducible_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real workflow on two qubits and reload its independently scored output."""
    monkeypatch.setattr(legacy_topdown_dense, "target_state", lambda seed: target_state(seed, qubits=2))
    monkeypatch.setattr(
        legacy_topdown_dense,
        "brickwall",
        lambda depth: create_brickwall_matrix_product_disentangler_parameterized_circuit(
            2, depth, initial_single_qubit_layer=True
        ),
    )
    legacy_topdown_dense.archived_baselines(tmp_path)
    root = tmp_path / "baseline_settings_tfim_400"
    target = np.load(root / "target.npy")
    for path in root.glob("*.json"):
        record = json.loads(path.read_text())
        gates = []
        for gate in record["circuit"]["gates"]:
            gate["sites"] = tuple(gate["sites"])
            gates.append(ParameterizedGate(**gate))
        circuit = ParameterizedCircuit(2, gates, record["circuit"]["num_params"])
        theta = np.load(root / (path.stem + ".theta.npy"))
        assert StatevectorBackend(circuit, target).metrics(theta)[1] == pytest.approx(
            record["noiseless"]["fidelity"], abs=1e-12
        )
    assert len(list(root.glob("*.json"))) == 3
