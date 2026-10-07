# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Dense and scalar references for reusable mixed-state MPS contractions."""

# Private snapshot/cache checks avoid adding public APIs solely for tests.
# ruff: noqa: SLF001

from __future__ import annotations

import copy
import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import KrotovTruncation, MPSBackend, ParameterizedCircuit, ParameterizedGate
from mqt.yaqs.optimization import mps_backend as backend_module
from mqt.yaqs.optimization.mps_contractions import MixedEnvironments, advance_left, advance_right, overlap
from tests.optimization.test_mps_backend import maps_for
from tests.optimization.test_statevector import case

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.optimization.trajectories import Estimator, NoiseMaps


def random_state(rng: np.random.Generator, bonds: list[int]) -> MPS:
    """Build a deliberately noncanonical, unnormalized complex tensor chain.

    Returns:
        An MPS with the requested unequal bond dimensions.
    """
    tensors = [(rng.normal(size=(2, a, b)) + 1j * rng.normal(size=(2, a, b))) / 3 for a, b in itertools.pairwise(bonds)]
    return MPS(len(bonds) - 1, tensors=tensors)


def scalar_overlap(a: MPS, b: MPS) -> complex:
    """Retain the original scalar contraction order as an independent oracle.

    Returns:
        Full-chain overlap without reuse or pair batching.
    """
    environment = np.ones((1, 1), dtype=complex)
    for bra, ket in zip(a.tensors, b.tensors, strict=True):
        environment = np.einsum("ab,pac,pbd->cd", environment, bra.conj(), ket, optimize=True)
    return complex(environment[0, 0])


class ScalarMPSBackend(MPSBackend):
    """Original scalar-pair signal, including derivative SVD behavior."""

    def _signal(
        self, gate: ParameterizedGate, outputs: list[MPS], costates: list[MPS], estimator: Estimator, weights: NDArray
    ) -> float:
        """Evaluate the historical implementation independently.

        Returns:
            The weighted scalar reference signal.
        """
        matrix, sites = self.circuit.derivative_operator(gate)
        signal = 0.0
        for s, output in enumerate(outputs):
            tangent = copy.deepcopy(output)
            self._apply(tangent, matrix, sites, "derivative")
            if estimator == "independent":
                signal += 2 * scalar_overlap(costates[s], tangent).real
            else:
                for r, costate in enumerate(costates):
                    signal -= (
                        2
                        * weights[r]
                        * weights[s]
                        * (scalar_overlap(costate, tangent) * scalar_overlap(output, costate)).real
                    )
        return float(gate.angle_scale * signal)


def test_batched_steps_match_explicit_tensor_contractions() -> None:
    """Non-square bonds and arbitrary environments agree in either direction."""
    rng = np.random.default_rng(615)
    a = rng.normal(size=(3, 2, 4, 5)) + 1j * rng.normal(size=(3, 2, 4, 5))
    b = rng.normal(size=(3, 2, 2, 3)) + 1j * rng.normal(size=(3, 2, 2, 3))
    left = rng.normal(size=(3, 4, 2)) + 1j * rng.normal(size=(3, 4, 2))
    right = rng.normal(size=(3, 5, 3)) + 1j * rng.normal(size=(3, 5, 3))
    np.testing.assert_allclose(advance_left(left, a, b), np.einsum("xab,xpac,xpbd->xcd", left, a.conj(), b))
    np.testing.assert_allclose(advance_right(right, a, b), np.einsum("xcd,xpac,xpbd->xab", right, a.conj(), b))


def test_canonical_form_does_not_make_cross_environments_identity() -> None:
    """Identical centers of orthogonal product states must still overlap to zero."""
    a, b = MPS(2), MPS(2)
    b.tensors[0] = np.array([0, 1], dtype=complex).reshape(2, 1, 1)
    assert np.vdot(a.tensors[1], b.tensors[1]) == 1
    assert overlap(a, b) == 0
    plain, tangent = MixedEnvironments().local_overlaps([a], [b], [b], (1, 1))
    np.testing.assert_array_equal(plain, [[0]])
    np.testing.assert_array_equal(tangent, [[0]])
    with pytest.raises(ValueError, match="equal register"):
        overlap(a, MPS(3))


@pytest.mark.parametrize("cache_bytes", [0, 128, 2**20])
@pytest.mark.parametrize("max_pairs", [1, 4, 64])
@pytest.mark.parametrize("interval", [(0, 0), (2, 3), (0, 4)])
def test_mixed_environments_match_dense_and_invalidate_locally(
    cache_bytes: int, max_pairs: int, interval: tuple[int, int]
) -> None:
    """Padding, chunking, local edits and cache eviction preserve complex overlaps."""
    rng = np.random.default_rng(890)
    bras = [random_state(rng, bonds) for bonds in ([1, 2, 3, 4, 2, 1], [1, 1, 2, 2, 1, 1])]
    kets = [random_state(rng, bonds) for bonds in ([1, 2, 4, 3, 2, 1], [1, 1, 1, 1, 1, 1], [1, 2, 2, 2, 2, 1])]
    cache = MixedEnvironments(max_cache_bytes=cache_bytes, max_pairs=max_pairs)
    first, last = interval
    for iteration in range(3):
        if iteration:
            # Change a site outside/inside the previous local interval, including
            # a bra-only change: both prefix and suffix invalidation are needed.
            site = 4 if iteration == 1 else 0
            bras[0].tensors[site] = np.multiply(bras[0].tensors[site], 0.8 + 0.3j)
            kets[-1].tensors[site] = np.multiply(kets[-1].tensors[site], 0.9 - 0.1j)
        tangents = [backend_module._snapshot(state) for state in kets]
        for tangent in tangents:
            for site in range(first, last + 1):
                tangent.tensors[site] = np.multiply(tangent.tensors[site], 0.2 - 0.6j)
        # Keep to_vec's view replacements away from the cached states, so only
        # the explicitly edited sites invalidate their surrounding environments.
        left_vectors = [copy.deepcopy(state).to_vec() for state in bras]
        right_vectors = [copy.deepcopy(state).to_vec() for state in kets]
        tangent_vectors = [copy.deepcopy(state).to_vec() for state in tangents]
        expected = np.array([[np.vdot(a, b) for b in right_vectors] for a in left_vectors])
        derivative = np.array([[np.vdot(a, b) for b in tangent_vectors] for a in left_vectors])
        for _ in range(2):
            plain, actual = cache.local_overlaps(bras, kets, tangents, interval)
            np.testing.assert_allclose(plain, expected, atol=1e-12)
            np.testing.assert_allclose(actual, derivative, atol=1e-12)
        assert cache.cache_bytes <= cache_bytes
        assert cache.peak_cache_bytes <= cache_bytes
    assert overlap(bras[0], kets[0]) == pytest.approx(expected[0, 0], abs=1e-12)
    if cache_bytes == 2**20 and interval != (0, 4):
        assert cache.environment_reuses > 0
    # Switching ensemble layout cannot reuse the old pair ordering.
    _, diagonal = cache.local_overlaps(bras, kets[:2], tangents[:2], interval, diagonal=True)
    np.testing.assert_allclose(diagonal, np.diag(derivative[:, :2]), atol=1e-12)
    cache.clear()
    assert not cache.snapshots
    assert not cache.environments
    assert cache.cache_bytes == 0


def test_unchanged_environments_reduce_repeated_chain_work() -> None:
    """A repeated local insertion contracts its local block, not both full chains."""
    states = [MPS(12), MPS(12, state="x+")]
    cache = MixedEnvironments()
    cache.local_overlaps(states, states, states, (5, 5))
    before = cache.site_contractions
    cache.local_overlaps(states, states, states, (5, 5))
    assert cache.site_contractions - before == 2
    assert cache.environment_reuses >= 2


@pytest.mark.parametrize(("max_cache_bytes", "max_pairs"), [(-1, 4), (32, 0)])
def test_invalid_cache_settings(max_cache_bytes: int, max_pairs: int) -> None:
    """Invalid cache bounds fail rather than bypassing chunk limits."""
    with pytest.raises(ValueError, match="limits"):
        MixedEnvironments(max_cache_bytes=max_cache_bytes, max_pairs=max_pairs)


def test_invalid_overlap_ensembles() -> None:
    """Reject incompatible ensembles and site intervals before contraction."""
    cache, state = MixedEnvironments(), MPS(2)
    for a, b, t, interval, diagonal in (
        ([], [state], [state], (0, 0), False),
        ([state], [state], [], (0, 0), False),
        ([state], [state, state], [state, state], (0, 0), True),
        ([state], [MPS(3)], [MPS(3)], (0, 0), False),
        ([state], [state], [state], (0, 2), False),
    ):
        with pytest.raises(ValueError, match="Invalid overlap"):
            cache.local_overlaps(a, b, t, interval, diagonal=diagonal)


@pytest.mark.parametrize("cap", [None, 1, 2])
@pytest.mark.parametrize("estimator", ["independent", "cross"])
def test_repeated_updates_match_original_scalar_mps_with_truncation(
    cap: int | None, estimator: Estimator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep derivative SVDs, long-range jumps and weighted update ordering unchanged."""
    circuit, theta, target = case(4)
    maps, weights = maps_for(circuit), np.array([0.83, 0.0, 0.17])
    reference = ScalarMPSBackend(circuit, target, truncation=KrotovTruncation(max_bond_dim=cap))
    actual = MPSBackend(circuit, target, truncation=KrotovTruncation(max_bond_dim=cap))
    a, b = theta.copy(), theta.copy()
    for _ in range(2):
        with monkeypatch.context() as old:
            old.setattr(backend_module, "_snapshot", copy.deepcopy)
            gradient, loss, fidelity = reference.gradient(a, maps, estimator=estimator, weights=weights)
            a -= 0.2 * gradient
            a, signal = reference.online_update(a, 0.2, maps, estimator=estimator, weights=weights)
        g, actual_loss, actual_fidelity = actual.gradient(b, maps, estimator=estimator, weights=weights)
        b -= 0.2 * g
        b, s = actual.online_update(b, 0.2, maps, estimator=estimator, weights=weights)
        np.testing.assert_allclose(g, gradient, atol=1e-11)
        np.testing.assert_allclose([actual_loss, actual_fidelity], [loss, fidelity], atol=1e-12)
        np.testing.assert_allclose(s, signal, atol=1e-11)
        np.testing.assert_allclose(b, a, atol=1e-11)
        assert actual.retained_state_count == 0
        assert not actual._overlaps.snapshots
        assert not actual._overlaps.environments
    assert actual.diagnostics.overlap_site_contractions > 0


def test_snapshot_local_gate_replacement_does_not_mutate_tape() -> None:
    """Shared arrays remain untouched for one-site, adjacent and swap-chain gates."""
    circuit, _, target = case(4)
    engine = MPSBackend(circuit, target)
    state = copy.deepcopy(engine.target)
    snapshot = backend_module._snapshot(state)
    before = [tensor.copy() for tensor in state.tensors]
    for tensor, saved in zip(state.tensors, snapshot.tensors, strict=True):
        assert tensor is saved
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    for matrix, sites in ((x, (1,)), (np.kron(x, x), (0, 1)), (np.kron(x, x), (3, 0))):
        engine._apply(state, matrix, sites, "forward")
    for tensor, saved in zip(snapshot.tensors, before, strict=True):
        np.testing.assert_array_equal(tensor, saved)


def test_cross_online_skips_unused_initial_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cross costates need no final overlaps; this shortcut cannot affect independent ones."""
    circuit, theta, target = case(3)
    engine = MPSBackend(circuit, target)
    original = engine._forward
    calls = []

    def counted(theta: NDArray, maps: NoiseMaps, *, retain: bool) -> tuple:
        """Count the retained initial forward pass.

        Returns:
            The original propagation result.
        """
        calls.append(1)
        return original(theta, maps, retain=retain)

    monkeypatch.setattr(engine, "_forward", counted)
    engine.online_update(theta, 0.2, maps_for(circuit), estimator="cross")
    assert not calls
    engine.online_update(theta, 0.2, maps_for(circuit), estimator="independent")
    assert len(calls) == 1


@pytest.mark.parametrize("sites", [(1,), (0, 1), (3, 0)])
@pytest.mark.parametrize("cap", [None, 1])
def test_shared_local_inputs_reuse_the_same_gate_kernel(sites: tuple[int, ...], cap: int | None) -> None:
    """Reuse exact local inputs even when untouched sites and trajectory weights differ."""
    circuit, _, target = case(4)
    engine = MPSBackend(circuit, target, truncation=KrotovTruncation(max_bond_dim=cap))
    states = [backend_module._snapshot(engine.target) for _ in range(4)]
    if sites == (1,):
        states[1].tensors[3] = np.multiply(states[1].tensors[3], 0.2j)
    distinct = copy.deepcopy(states[2].tensors[min(sites)])
    states[2].tensors[min(sites)] = distinct
    expected = copy.deepcopy(states)
    rng = np.random.default_rng(151)
    matrix = rng.normal(size=(2 ** len(sites), 2 ** len(sites))) + 0.3j
    for state in expected:
        engine._apply(state, matrix, sites, "derivative")
    before = engine.diagnostics.operator_applications
    engine._apply_all(states, matrix, sites, "derivative")
    assert engine.diagnostics.operator_applications - before == 2
    assert engine.diagnostics.reused_local_applications == 2
    # to_vec flips the network twice and replaces tensor views, so inspect
    # sharing before invoking that independent dense oracle.
    assert states[0].tensors[min(sites)] is states[3].tensors[min(sites)]
    for actual, reference in zip(states, expected, strict=True):
        np.testing.assert_allclose(actual.to_vec(), reference.to_vec(), atol=1e-12)


def test_cross_online_records_returned_state_norm_loss() -> None:
    """Skipping the preliminary propagation must not hide truncated norm loss."""
    circuit = ParameterizedCircuit(2, [ParameterizedGate("h", (0,)), ParameterizedGate("cx", (0, 1))])
    engine = MPSBackend(circuit, MPS(2), truncation=KrotovTruncation(max_bond_dim=1))
    engine.online_update(np.array([]), 0.2, estimator="cross")
    assert engine.diagnostics.maximum_forward_norm_error > 0.1


def test_failed_signal_releases_cached_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exceptions cannot retain a previous optimizer call's environments or tape."""
    circuit, theta, target = case(3)
    engine = MPSBackend(circuit, target)
    original = engine._signal

    def fail(
        gate: ParameterizedGate, outputs: list[MPS], costates: list[MPS], estimator: Estimator, weights: NDArray
    ) -> float:
        """Populate environments before an artificial failure.

        Raises:
            RuntimeError: After evaluating a real signal.
        """
        original(gate, outputs, costates, estimator, weights)
        msg = "signal interrupted"
        raise RuntimeError(msg)

    monkeypatch.setattr(engine, "_signal", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        engine.gradient(theta, maps_for(circuit), estimator="cross")
    assert not engine._overlaps.snapshots
    assert not engine._overlaps.environments
    assert engine.retained_state_count == 0
