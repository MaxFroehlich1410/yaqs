# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compare weighted MPS replay to dense gradients and measure true discarded weights."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace

import numpy as np
import pytest

from experiments.overnight_state_preparation_20260913_0108 import mps_bridge, trajectory_mixture
from mqt.yaqs.core import linalg
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KrotovNoiseMap,
    KrotovTruncation,
    ParameterizedCircuit,
    ParameterizedGate,
    StatevectorBackend,
)


def problem(qubits: int) -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Return a complex shared-parameter case with reversed and distant sites."""
    c = ParameterizedCircuit(
        qubits,
        [
            ParameterizedGate("ry", (0,), param_index=1, angle_scale=-0.7, angle_offset=0.2),
            ParameterizedGate("rx", (qubits - 1,), param_index=0),
            ParameterizedGate("rxx", (qubits - 1, 0), param_index=2),
            ParameterizedGate("rz", (0,), param_index=1, angle_scale=1.3),
            ParameterizedGate("h", (1,), noise_enabled=False),
        ],
    )
    rng = np.random.default_rng(8)
    target = rng.normal(size=2**qubits) + 1j * rng.normal(size=2**qubits)
    return c, np.array([0.6, -0.9, 0.4]), target / np.linalg.norm(target)


@pytest.mark.parametrize("qubits", [2, 3])
@pytest.mark.parametrize("target_mps", [False, True])
def test_weighted_fixed_map_adjoint_matches_dense_and_finite_difference(qubits: int, *, target_mps: bool) -> None:
    """Unequal channel/path weights and signed shared occurrences retain the true gradient."""
    c, theta, target = problem(qubits)
    batch = trajectory_mixture.MixtureSampler(c).ensemble(seed=17, update_index=3, paths_per_channel=3)
    engine = StatevectorBackend(c, target)
    dense, loss, fidelity = batch.gradient(engine, theta)
    actual = mps_bridge.gradient(
        c, theta, MPS.from_statevector(target) if target_mps else target, batch.maps, batch.weights
    )
    np.testing.assert_allclose(actual.gradient, dense, atol=3e-12, rtol=0)
    assert actual.loss == pytest.approx(loss, abs=3e-12)
    assert actual.fidelity == pytest.approx(fidelity, abs=3e-12)
    for j in range(len(theta)):
        step = np.eye(len(theta))[j] * 1e-6
        finite = (
            engine.metrics(theta + step, batch.maps, weights=batch.weights)[0]
            - engine.metrics(theta - step, batch.maps, weights=batch.weights)[0]
        ) / 2e-6
        assert actual.gradient[j] == pytest.approx(finite, abs=2e-9)
    assert actual.audit["trajectory_paths"] == 6
    assert actual.audit["directional_paths"] == 12
    assert actual.audit["discarded_squared_singular_values"] == pytest.approx(0.0, abs=1e-30)
    assert actual.audit["maximum_internal_bond"] <= 2 ** (qubits // 2)
    assert actual.audit["forward_final_max_norm_error"] < 1e-12
    assert actual.cpu_seconds > 0


def test_actual_bonds_exclude_physical_axes() -> None:
    """A product state has bond one even though every physical dimension is two."""
    state = MPS(3, state="zeros")
    audit = mps_bridge.BondAudit()
    audit.observe(state)
    assert mps_bridge.bonds(state) == [1, 1]
    assert audit.maximum_internal_bond == 1


def test_svd_meter_records_actual_tail_and_restores_functions() -> None:
    """Passive wrapping records discarded squared values without modifying the keep decision."""
    raw_truncate, raw_apply = linalg.truncate, MPS.apply_local_operator
    audit = mps_bridge.BondAudit()
    with ExitStack() as stack:
        audit.instrument(stack)
        keep = linalg.truncate(np.array([1.0, 0.3, 0.2]), mode="discarded_weight", threshold=0, max_bond_dim=1)
        assert keep == 1
    assert linalg.truncate is raw_truncate
    assert MPS.apply_local_operator is raw_apply
    assert audit.discarded_squared_singular_values == pytest.approx(0.13)
    assert audit.hard_cap_reductions == 1
    assert audit.splits[0]["available"] == 3


def test_truncated_gradient_reports_loss_of_weight_and_norm() -> None:
    """A forced product approximation of a Bell preparation cannot claim zero truncation."""
    c = ParameterizedCircuit(2, [ParameterizedGate("ry", (0,), param_index=0), ParameterizedGate("cx", (0, 1))])
    theta = np.array([np.pi / 2])
    target = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    maps = ((KrotovNoiseMap(), KrotovNoiseMap()),)
    weights = np.array([1.0])
    exact = mps_bridge.gradient(c, theta, target, maps, weights)
    truncated = mps_bridge.gradient(c, theta, target, maps, weights, truncation=KrotovTruncation(max_bond_dim=1))
    assert exact.fidelity == pytest.approx(1)
    assert truncated.fidelity < exact.fidelity
    assert truncated.audit["discarded_squared_singular_values"] > 0
    assert truncated.audit["forward_final_max_norm_error"] > 0
    assert truncated.audit["hard_cap_reductions"] > 0


@pytest.mark.parametrize("bad", ["negative", "not_normalized", "nan", "length", "normalized_map", "nonunitary"])
def test_rejects_invalid_weighted_replay(bad: str) -> None:
    """Unsupported maps and weights cannot silently change the physical objective."""
    c, theta, target = problem(2)
    maps = ((KrotovNoiseMap(),) * len(c.gates),)
    weights = np.array([1.0])
    if bad == "negative":
        weights = np.array([-1.0])
    elif bad == "not_normalized":
        weights = np.array([0.5])
    elif bad == "nan":
        weights = np.array([np.nan])
    elif bad == "length":
        maps = (maps[0][:-1],)
    elif bad == "normalized_map":
        maps = ((replace(maps[0][0], normalized=True), *maps[0][1:]),)
    else:
        maps = ((KrotovNoiseMap(operators=((np.eye(2) * 2, (0,)),)), *maps[0][1:]),)
    with pytest.raises(ValueError, match=r"weights|maps|operators"):
        mps_bridge.gradient(c, theta, target, maps, weights)


def test_instrumentation_is_restored_after_numerical_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupted adjoint leaves the public numerical functions intact."""
    c, theta, target = problem(2)
    raw_truncate, raw_apply = linalg.truncate, MPS.apply_local_operator

    def fail(*_args: object, **_kwargs: object) -> None:
        msg = "injected numerical failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(mps_bridge, "noisy_state_preparation_contribution", fail)
    with pytest.raises(RuntimeError, match="injected"):
        mps_bridge.gradient(c, theta, target, ((KrotovNoiseMap(),) * len(c.gates),), np.array([1.0]))
    assert linalg.truncate is raw_truncate
    assert MPS.apply_local_operator is raw_apply
