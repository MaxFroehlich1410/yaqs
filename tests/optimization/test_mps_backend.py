# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent dense oracles for weighted MPS training, pruning and diagnostics."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from experiments import legacy_topdown_dense as legacy
from experiments import topdown_batch_refinement_policies as policies
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.linalg.svd_utils import observe_truncations, truncate
from mqt.yaqs.optimization import (
    KrotovNoiseMap,
    KrotovTJMOptions,
    KrotovTruncation,
    MPSBackend,
    ParameterizedCircuit,
    ParameterizedGate,
    StatevectorBackend,
    TrajectorySampler,
    create_backend,
)
from tests.optimization.test_statevector import case

if TYPE_CHECKING:
    from mqt.yaqs.optimization.trajectories import Estimator


def maps_for(circuit: ParameterizedCircuit) -> tuple:
    """Provide actual jumps on an ordered noise support.

    Returns:
        Identity, single-site and two-site replay paths.
    """
    empty = tuple(KrotovNoiseMap() for _ in circuit.gates)
    one, two = list(empty), list(empty)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    if circuit.gates:
        one[0] = KrotovNoiseMap(operators=((x, (0,)),))
        two[-1] = KrotovNoiseMap(operators=((np.kron(x, y), (0, circuit.num_qubits - 1)),))
    return empty, tuple(one), tuple(two)


@pytest.mark.parametrize("qubits", [2, 3, 6])
@pytest.mark.parametrize("estimator", ["independent", "cross"])
def test_weighted_mps_matches_dense_for_multiple_updates(qubits: int, estimator: Estimator) -> None:
    """Unequal weights preserve complex shared-coordinate batch and online signals."""
    circuit, theta, target = case(qubits)
    maps = maps_for(circuit)
    weights = np.array([0.9, 0.07, 0.03])
    dense, mps = StatevectorBackend(circuit, target), MPSBackend(circuit, target)
    expected, actual = theta.copy(), theta.copy()
    for _ in range(2):
        g, loss, fidelity = dense.gradient(expected, maps, estimator=estimator, weights=weights)
        mg, ml, mf = mps.gradient(actual, maps, estimator=estimator, weights=weights)
        np.testing.assert_allclose(mg, g, atol=3e-11, rtol=0)
        np.testing.assert_allclose([ml, mf], [loss, fidelity], atol=3e-12)
        expected -= 0.2 * g
        actual -= 0.2 * mg
        expected, signal = dense.online_update(expected, 0.2, maps, estimator=estimator, weights=weights)
        actual, msignal = mps.online_update(actual, 0.2, maps, estimator=estimator, weights=weights)
        np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=0)
        np.testing.assert_allclose(msignal, signal, atol=3e-11, rtol=0)
    assert mps.retained_state_count == 0
    assert mps.diagnostics.discarded_weight_sum == 0
    assert mps.diagnostics.maximum_retained_tensor_bytes > 0


def test_compressed_weighted_cross_equals_one_hundred_explicit_paths() -> None:
    """Weights apply to both pair indices and to every sequential gate update."""
    circuit, theta, target = case(2)
    maps = maps_for(circuit)
    raw = [maps[0]] * 90 + [maps[1]] * 7 + [maps[2]] * 3
    sampler = TrajectorySampler(circuit)
    compressed, weights = sampler.compress_no_jump_maps(raw)
    assert len(compressed) == 11
    assert weights[0] == pytest.approx(0.9)
    dense, mps = StatevectorBackend(circuit, target), MPSBackend(circuit, target)
    a, b = theta.copy(), theta.copy()
    for _ in range(3):
        g, loss, fidelity = dense.gradient(a, raw, estimator="cross")
        mg, ml, mf = mps.gradient(b, compressed, estimator="cross", weights=weights)
        np.testing.assert_allclose(mg, g, atol=1e-12)
        np.testing.assert_allclose([ml, mf], [loss, fidelity], atol=1e-12)
        a -= 0.2 * g
        b -= 0.2 * mg
        a, _ = dense.online_update(a, 0.2, raw, estimator="cross")
        b, _ = mps.online_update(b, 0.2, compressed, estimator="cross", weights=weights)
        np.testing.assert_allclose(a, b, atol=2e-12)
    # A separate all-MPS replication check catches backend-specific weighting errors.
    small = [maps[0]] * 9 + [maps[1]]
    reduced, mass = sampler.compress_no_jump_maps(small)
    np.testing.assert_allclose(
        mps.online_update(theta, 0.2, small, estimator="cross")[0],
        mps.online_update(theta, 0.2, reduced, estimator="cross", weights=mass)[0],
        atol=1e-12,
    )


def test_independent_gradient_finite_difference_and_deletions() -> None:
    """True adjoints and finite removal amplitudes agree with independent replays."""
    circuit, theta, target = case(3)
    engine = MPSBackend(circuit, target)
    maps, weights = maps_for(circuit), np.array([0.6, 0.1, 0.3])
    gradient = engine.gradient(theta, maps, weights=weights)[0]
    for k in range(len(theta)):
        delta = np.eye(len(theta))[k] * 1e-6
        fd = (
            engine.metrics(theta + delta, maps, weights=weights)[0]
            - engine.metrics(theta - delta, maps, weights=weights)[0]
        ) / 2e-6
        assert gradient[k] == pytest.approx(fd, abs=1e-8)
    scores, fidelity = engine.deletion_fidelities(theta)
    dense_scores, dense_fidelity = StatevectorBackend(circuit, target).deletion_fidelities(theta)
    np.testing.assert_allclose(scores, dense_scores, atol=1e-12)
    assert fidelity == pytest.approx(dense_fidelity, abs=1e-12)
    for k, score in enumerate(scores):
        deleted = ParameterizedCircuit(3, [g for i, g in enumerate(circuit.gates) if i != k], circuit.num_params)
        assert score == pytest.approx(StatevectorBackend(deleted, target).metrics(theta)[1], abs=1e-12)


@pytest.mark.parametrize("estimator", ["independent", "cross"])
def test_zero_weights_and_input_immutability(estimator: Estimator) -> None:
    """Zero mass suppresses a path and neither maps nor tensors are mutated."""
    circuit, theta, target = case()
    source = MPS.from_statevector(target)
    before = copy.deepcopy(source)
    engine = MPSBackend(circuit, source)
    maps = maps_for(circuit)
    original = theta.copy()
    weights = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        engine.gradient(theta, maps, weights=weights, estimator=estimator)[0],
        engine.gradient(theta, [maps[0]], estimator=estimator)[0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        engine.online_update(theta, 0.3, maps, weights=weights, estimator=estimator)[0],
        engine.online_update(theta, 0.3, [maps[0]], estimator=estimator)[0],
        atol=1e-12,
    )
    np.testing.assert_array_equal(theta, original)
    for a, b in zip(before.tensors, source.tensors, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("weights", [[0.5, 0.5], [-1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [np.nan, 0.0, 1.0], [1j, 0.0, 1.0]])
def test_bad_weights_fail_before_propagation(weights: list, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject malformed probabilities rather than normalizing or ignoring them."""
    c, theta, target = case()
    engine = MPSBackend(c, target)
    monkeypatch.setattr(engine, "_forward", lambda *_a, **_k: pytest.fail("propagated invalid input"))
    for method in [engine.metrics, engine.gradient]:
        with pytest.raises(ValueError, match="weights"):
            method(theta, maps_for(c), weights=np.array(weights))
    with pytest.raises(ValueError, match="weights"):
        engine.online_update(theta, 0.2, maps_for(c), weights=np.array(weights))


def test_sampling_is_state_free_and_matches_archived_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exactly preserve existing Pauli seed streams and no-jump multiplicities."""
    c, theta, target = case()
    dense = StatevectorBackend(c, target)
    options = KrotovTJMOptions(num_trajectories=100)
    expected = dense.sample_noise_maps(theta, options, legacy.LegacyNoise(), iteration=5300)
    monkeypatch.setattr(MPS, "__init__", lambda *_a, **_k: pytest.fail("MPS sampling allocation"))
    monkeypatch.setattr(StatevectorBackend, "__init__", lambda *_a, **_k: pytest.fail("dense sampling allocation"))
    actual = TrajectorySampler(c).sample_noise_maps(theta, options, legacy.LegacyNoise(), iteration=5300)
    for a, b in zip(actual, expected, strict=True):
        for x, y in zip(a, b, strict=True):
            assert len(x.operators) == len(y.operators)
            for (u, sites), (v, other) in zip(x.operators, y.operators, strict=True):
                assert sites == other
                np.testing.assert_array_equal(u, v)


def test_noiseless_lbfgs_and_noisy_training_share_existing_policies() -> None:
    """Only the representation changes for construction and frozen fine-tuning."""
    c, theta, target = case(2)
    expected, old = policies.train_noiseless(c, theta, target, policy="lbfgs", steps=8, online=3)
    actual, new = policies.train_noiseless(
        c, theta, target, policy="lbfgs", steps=8, online=3, engine=MPSBackend(c, target)
    )
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    np.testing.assert_allclose(new["fidelity_trace"], old["fidelity_trace"], atol=1e-10)
    assert new["returned_fidelity"] >= new["final_fidelity"]
    settings: dict[str, Any] = {
        "steps": 4,
        "online": 2,
        "rate": 0.2,
        "seed": 5300,
        "noisy": True,
        "num_trajectories": 100,
        "compress_no_jump": True,
        "record_noise_banks": True,
    }
    expected, old = legacy.train(c, theta, target, **settings)
    actual, new = legacy.train(c, theta, target, engine=MPSBackend(c, target), **settings)
    np.testing.assert_allclose(actual, expected, atol=1e-11)
    np.testing.assert_allclose(new["fidelity_trace"], old["fidelity_trace"], atol=1e-12)
    assert old["noise_banks"] == new["noise_banks"]


def test_twelve_qubit_mps_never_densifies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Construction and noise replay can run with a tensor-only target."""
    c = ParameterizedCircuit(
        12, [ParameterizedGate("ry", (0,), param_index=0), ParameterizedGate("rxx", (0, 1), param_index=1)]
    )
    target = MPS(12)
    monkeypatch.setattr(MPS, "to_vec", lambda *_a: pytest.fail("densified target"))
    monkeypatch.setattr(MPS, "from_statevector", lambda *_a: pytest.fail("dense target conversion"))
    engine = create_backend(c, target, backend="mps")
    theta = np.array([0.1, 0.2])
    engine.gradient(theta, maps_for(c), estimator="cross", weights=np.array([0.9, 0.07, 0.03]))
    engine.online_update(theta, 0.2)
    engine.deletion_fidelities(theta)
    assert engine.metrics(theta)[1] > 0.9


def test_truncation_reports_actual_loss_and_converges() -> None:
    """A forced bond cap loses Bell entanglement; a sufficient cap recovers exact results."""
    c = ParameterizedCircuit(2, [ParameterizedGate("h", (0,)), ParameterizedGate("cx", (0, 1))])
    target = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    truncated = MPSBackend(c, target, truncation=KrotovTruncation(max_bond_dim=1))
    exact = MPSBackend(c, target, truncation=KrotovTruncation(max_bond_dim=2))
    assert truncated.metrics(np.array([]))[1] < 0.6
    assert exact.metrics(np.array([]))[1] == pytest.approx(1)
    assert truncated.diagnostics.discarded_weight_sum > 0
    assert truncated.diagnostics.maximum_forward_norm_error > 0.1
    assert exact.diagnostics.discarded_weight_sum == 0
    compressed = MPSBackend(c, target, truncation=KrotovTruncation(max_bond_dim=1), compress_target=True)
    assert compressed.target_compression_infidelity == pytest.approx(0.5)


def test_observer_nested_exception_and_thread_isolation() -> None:
    """Diagnostics do not monkeypatch numerical functions or leak across contexts."""
    outer, inner = [], []
    values = np.array([0.8, 0.6])
    with observe_truncations(outer.append):
        assert truncate(values, mode="discarded_weight", threshold=0, max_bond_dim=1) == 1

        def fail() -> None:
            """Exercise exception restoration.

            Raises:
                RuntimeError: Always, after the observed split.
            """
            with observe_truncations(inner.append):
                truncate(values, mode="relative", threshold=0.9)
                msg = "test"
                raise RuntimeError(msg)

        with pytest.raises(RuntimeError):
            fail()
        with ThreadPoolExecutor(1) as pool:
            pool.submit(truncate, values, mode="relative", threshold=0).result()
        truncate(values, mode="relative", threshold=0)
    truncate(values, mode="relative", threshold=0)
    assert len(outer) == 2
    assert len(inner) == 1
    assert outer[0].discarded_weight == pytest.approx(0.36)
    assert outer[0].capped


def test_empty_circuit_and_invalid_factory() -> None:
    """Empty circuits remain valid and incompatible factory options fail explicitly."""
    c = ParameterizedCircuit(2, [])
    engine = MPSBackend(c, MPS(2))
    assert engine.metrics(np.array([])) == (0.0, 1.0)
    assert engine.gradient(np.array([]))[0].size == 0
    assert engine.online_update(np.array([]), 0)[0].size == 0
    assert engine.deletion_fidelities(np.array([]))[0].size == 0
    with pytest.raises(ValueError, match="backend"):
        create_backend(c, MPS(2), backend="statevector")
    with pytest.raises(ValueError, match="truncation"):
        MPSBackend(c, MPS(2), truncation=KrotovTruncation(max_bond_dim=0))


@pytest.mark.parametrize("bad", [np.array([2, 0, 0, 0]), np.array([np.nan, 0, 0, 0]), np.array([1, 0])])
def test_invalid_targets(bad: np.ndarray) -> None:
    """Reject wrong norms, nonfinite amplitudes and mismatched dimensions."""
    with pytest.raises(ValueError, match=r"normalized|finite"):
        MPSBackend(ParameterizedCircuit(2, []), bad)


def test_best_initial_parameters_survive_a_damaging_mps_sweep() -> None:
    """Restoration covers the initial state and does not silently change the step policy."""
    c = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)])
    target = np.array([1, 0], dtype=complex)
    theta = np.array([0.3])
    returned, trace = policies.train_noiseless(
        c, theta, target, policy="lbfgs", steps=1, online=1, rate=50.0, engine=MPSBackend(c, target)
    )
    np.testing.assert_array_equal(returned, theta)
    assert trace["returned_fidelity"] > trace["final_fidelity"]
    assert trace["best_iteration"] == 0
