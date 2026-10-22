# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Dense independent checks for optional canonical epsilon-compressed updates."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization.canonical_mps_backend import CanonicalMPSBackend
from mqt.yaqs.optimization.mps_backend import MPSBackend
from mqt.yaqs.optimization.statevector import StatevectorBackend
from tests.optimization.test_mps_backend import maps_for
from tests.optimization.test_statevector import case

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.optimization.trajectories import Estimator


@pytest.mark.parametrize("qubits", [2, 3, 6])
@pytest.mark.parametrize("estimator", ["independent", "cross"])
@pytest.mark.parametrize("relative_cutoff", [float(np.finfo(float).eps), 1e-8])
def test_complete_updates_match_dense(qubits: int, estimator: Estimator, relative_cutoff: float) -> None:
    """Preserve weighted gradients, sequential/shared updates and long-range noise."""
    circuit, theta, target = case(qubits)
    maps = maps_for(circuit)
    weights = np.array([0.8, 0.13, 0.07])
    dense = StatevectorBackend(circuit, target)
    compressed = CanonicalMPSBackend(circuit, target, relative_cutoff=relative_cutoff)
    original = theta.copy()
    for _ in range(2):
        actual = compressed.gradient(theta, maps, estimator=estimator, weights=weights)
        expected = dense.gradient(theta, maps, estimator=estimator, weights=weights)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b, atol=5e-12, rtol=0)
        updated, signal = compressed.online_update(theta, 0.2, maps, estimator=estimator, weights=weights)
        reference, expected_signal = dense.online_update(theta, 0.2, maps, estimator=estimator, weights=weights)
        np.testing.assert_allclose(updated, reference, atol=5e-12, rtol=0)
        np.testing.assert_allclose(signal, expected_signal, atol=5e-12, rtol=0)
        assert compressed.retained_state_count == 0
        theta = updated
    np.testing.assert_array_equal(original, case(qubits)[1])
    assert compressed.diagnostics.cap_reductions == 0


def test_snapshot_backing_arrays_and_phase_are_preserved() -> None:
    """QR center moves cannot mutate earlier tape entries or caller-owned states."""
    circuit, theta, target = case(6)
    initial = MPS.from_statevector(target * np.exp(0.7j))
    before = initial.to_vec().copy()

    class CheckedBackend(CanonicalMPSBackend):
        def _apply(self, state: MPS, matrix: NDArray, sites: tuple[int, ...], role: str) -> None:
            """Check that every propagation preserves the old tensor backing arrays."""
            references = list(state.tensors)
            old_values = [a.copy() for a in references]
            super()._apply(state, matrix, sites, role)
            for old, expected in zip(references, old_values, strict=True):
                np.testing.assert_array_equal(old, expected)

    engine = CheckedBackend(circuit, initial, initial)
    engine.online_update(theta, 0.2, maps_for(circuit), estimator="cross")
    np.testing.assert_allclose(copy.deepcopy(engine.initial).to_vec(), before, atol=1e-13)
    np.testing.assert_array_equal(initial.to_vec(), before)


def test_deletions_and_metrics_preserve_untruncated_results() -> None:
    """Pruning scores and sampled fidelities remain consistent at small widths."""
    circuit, theta, target = case(6)
    exact = MPSBackend(circuit, target)
    compressed = CanonicalMPSBackend(circuit, target)
    for actual, expected in zip(compressed.deletion_fidelities(theta), exact.deletion_fidelities(theta), strict=True):
        np.testing.assert_allclose(actual, expected, atol=5e-12, rtol=0)
    np.testing.assert_allclose(
        compressed.metrics(theta, maps_for(circuit)), exact.metrics(theta, maps_for(circuit)), atol=5e-12
    )


def test_roundoff_directions_are_removed_without_renormalization() -> None:
    """Remove padded GHZ directions while preserving amplitudes and complex phase."""
    circuit, _, _ = case(8)
    target = np.zeros(256, dtype=complex)
    target[0], target[-1] = np.sqrt(0.3), np.sqrt(0.7) * np.exp(0.8j)
    padded = MPS.from_statevector(target)
    before = copy.deepcopy(padded).to_vec()
    assert max(t.shape[2] for t in padded.tensors) > 2
    compressed = CanonicalMPSBackend(circuit, padded)
    assert max(t.shape[2] for t in compressed.target.tensors) == 2
    np.testing.assert_allclose(copy.deepcopy(compressed.target).to_vec(), before, atol=1e-13)
    assert compressed.diagnostics.cap_reductions == 0
    assert compressed.diagnostics.discarded_weight_sum < 1e-25


def test_stronger_propagation_preserves_input_precision_and_is_instance_local() -> None:
    """Retain small target/input components while truncating only propagated copies."""

    class Probe(CanonicalMPSBackend):
        def propagate_target(self) -> MPS:
            """Replay an identity split on a private target copy.

            Returns:
                Propagated state with the selected instance's tolerance.
            """
            propagated = copy.deepcopy(self.target)
            self._apply(propagated, np.eye(4, dtype=complex), (0, 1), "adjoint")
            return propagated

    circuit, _, _ = case(2)
    target = np.array([1, 0, 0, 1e-10j], dtype=complex)
    strong = Probe(circuit, target, target, relative_cutoff=1e-8)
    epsilon = Probe(circuit, target, target)
    for original in (strong.target, strong.initial, epsilon.target):
        np.testing.assert_allclose(copy.deepcopy(original).to_vec(), target, atol=1e-15, rtol=0)
    propagated = strong.propagate_target()
    assert max(t.shape[2] for t in propagated.tensors) == 1
    assert abs(propagated.to_vec()[-1]) < 1e-15
    propagated = epsilon.propagate_target()
    np.testing.assert_allclose(propagated.to_vec(), target, atol=1e-15, rtol=0)
    np.testing.assert_allclose(copy.deepcopy(strong.target).to_vec(), target, atol=1e-15, rtol=0)


@pytest.mark.parametrize("cutoff", [0.0, -1.0, 1.0, float("nan"), float("inf"), True])
def test_invalid_propagation_cutoffs_are_rejected(cutoff: float) -> None:
    """Reject invalid tolerances before canonicalizing or propagating any state."""
    circuit, _, target = case(2)
    with pytest.raises(ValueError, match="relative_cutoff must"):
        CanonicalMPSBackend(circuit, target, relative_cutoff=cutoff)
