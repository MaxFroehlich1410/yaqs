# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent numerical checks of both recovery stages and their run counters."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.sparse.linalg import ArpackNoConvergence

from mqt.yaqs import AnalogSimParams, EvolutionMode, Hamiltonian, Observable, Simulator, State, track_conservation
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.methods import conservation_fallback as recovery
from mqt.yaqs.core.methods.conservation import (
    ConservationTargets,
    EnergyTarget,
    apply_conservation,
    expectation_value,
    restore_invariant,
    restore_invariant_at_center,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def one_site(matrix: NDArray, vector: NDArray) -> tuple[MPS, MPO]:
    """Build a canonical one-site problem without MPO truncation.

    Returns:
        State and operator.
    """
    state = MPS(1, tensors=[np.asarray(vector, dtype=complex).reshape(-1, 1, 1)], physical_dimensions=[len(vector)])
    state.set_center(0)
    mpo = MPO()
    mpo.custom([np.asarray(matrix, dtype=complex).reshape(len(vector), len(vector), 1, 1)], transpose=False)
    return state, mpo


def product_problem(*, eigenstate: bool) -> tuple[MPS, MPO]:
    """Place a nontrivial constraint at the far end of a product chain.

    Returns:
        State centered at zero and an observable acting at site two.
    """
    local = np.array([1.0, 0.0]) if eigenstate else np.sqrt([0.8, 0.2])
    state = MPS(
        3,
        tensors=[np.array([1, 0], dtype=complex).reshape(2, 1, 1) for _ in range(2)]
        + [local.astype(complex).reshape(2, 1, 1)],
    )
    state.set_center(0)
    mpo = MPO()
    mpo.custom(
        [np.eye(2, dtype=complex).reshape(2, 2, 1, 1) for _ in range(2)]
        + [np.diag([1, -1]).astype(complex).reshape(2, 2, 1, 1)],
        transpose=False,
    )
    return state, mpo


@pytest.mark.parametrize("reflected", [False, True])
def test_center_scan_restores_target_and_entry_gauge(*, reflected: bool) -> None:
    """A direction absent at the compression endpoint exists at another site."""
    state, mpo = product_problem(eigenstate=False)
    if reflected:
        state.flip_network()
        mpo = mpo.reflected()
    entry = state.orthogonality_center
    before = [tensor.shape for tensor in state.tensors]
    with track_conservation() as stats:
        assert restore_invariant(state, mpo, 0.0, tol=1e-13)
    assert abs(expectation_value(state, mpo)) < 1e-13
    assert state.orthogonality_center == entry
    assert state.flipped == reflected
    assert [tensor.shape for tensor in state.tensors] == before
    assert stats.fallback_attempts == stats.scan_successes == 1
    assert stats.scan_sites == 2
    assert stats.frame_attempts == stats.fallback_failures == 0


@pytest.mark.parametrize("reflected", [False, True])
def test_frame_scan_skips_infeasible_frames_and_solves_only_once(*, reflected: bool) -> None:
    """An eigenstate needs a full frame, available only at the last site."""
    state, mpo = product_problem(eigenstate=True)
    if reflected:
        state.flip_network()
        mpo = mpo.reflected()
    entry = state.orthogonality_center
    shapes = [tensor.shape for tensor in state.tensors]
    with track_conservation() as stats:
        assert restore_invariant(state, mpo, 0.0, tol=1e-13)
    assert abs(expectation_value(state, mpo)) < 1e-13
    assert state.orthogonality_center == entry
    assert state.flipped == reflected
    assert [tensor.shape for tensor in state.tensors] == shapes
    assert stats.frames_checked == 3
    assert stats.frame_attempts == stats.frame_successes == stats.fallback_successes == 1
    assert stats.scan_successes == 0


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_negative_discriminant_matches_independent_dense_minimum(sign: float) -> None:
    """Check feasibility, accuracy, stationarity and the global minimum separately."""
    rng = np.random.default_rng(29)
    q, _ = np.linalg.qr(rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7)))
    eigenvalues = sign * np.array([-1.0, 0.0, 1.0, -0.8, 0.3, 0.6, 0.7])
    coefficients = np.sqrt([0.1, 0.8, 0.1, 0, 0, 0, 0]).astype(complex)
    matrix = (q * eigenvalues) @ q.conj().T
    c = q @ coefficients
    target = sign * 0.9
    state, mpo = one_site(matrix, c)
    assert not restore_invariant_at_center(state, mpo, target, tol=1e-13)
    with track_conservation() as stats:
        assert restore_invariant(state, mpo, target, tol=1e-13)
    actual = state.to_vec()
    # The dense spectral oracle does not use the production Krylov/linear solvers.
    b = -sign * (eigenvalues - target)
    pole = -1 / b.min()

    def secular(multiplier: float) -> float:
        return float(np.sum(b * np.abs(coefficients / (1 + multiplier * b)) ** 2))

    multiplier = brentq(secular, 0, pole * (1 - 1e-8), xtol=1e-14)
    expected = q @ (coefficients / (1 + multiplier * b))
    np.testing.assert_allclose(actual, expected, atol=3e-12)
    assert abs(expectation_value(state, mpo) - target) < 1e-13
    assert stats.frame_successes == 1


def test_singular_endpoint_adds_a_missing_eigenspace() -> None:
    """C alone spans no correcting direction; the global minimizer is at the pole."""
    matrix = np.diag([-1.0, 1.0, 2.0, 3.0, 4.0])
    c = np.array([0, 1, 0, 0, 0], dtype=complex)
    state, mpo = one_site(matrix, c)
    with track_conservation() as stats:
        assert restore_invariant(state, mpo, 0.0, tol=1e-13)
    x = state.to_vec()
    assert abs(expectation_value(state, mpo)) < 1e-13
    assert np.linalg.norm(x - c) ** 2 == pytest.approx(0.5, abs=1e-12)
    np.testing.assert_allclose((np.eye(5) + matrix) @ x, c, atol=1e-12)
    assert stats.frame_successes == 1


def test_semidefinite_endpoint_projects_onto_complete_degenerate_kernel() -> None:
    """The boundary solution keeps all kernel components, not one Ritz vector."""
    matrix = np.diag([0.0, 0.0, 1.0, 2.0, 3.0])
    c = np.array([0.3, 0.4j, 0.2, 0.5, 0.6], dtype=complex)
    status, frame = recovery.prepare_frame(lambda x: matrix @ x, c, 0.0, tol=1e-14)
    assert status == "feasible"
    assert frame is not None
    status, x = recovery.minimum_displacement(frame, tol=1e-13)
    assert status == "feasible"
    assert x is not None
    np.testing.assert_allclose(x, [0.3, 0.4j, 0, 0, 0], atol=1e-12)


def test_boundary_without_nonzero_minimizer_is_reported_and_rolled_back() -> None:
    """The nonzero-constrained minimization can have an unattained infimum."""
    state, mpo = one_site(np.diag([0.0, 1.0, 2.0]), np.array([0, 1, 0]))
    original = state.tensors[0].copy()
    with track_conservation() as stats:
        assert not restore_invariant(state, mpo, 0.0, tol=1e-13)
    np.testing.assert_array_equal(state.tensors[0], original)
    assert stats.no_minimizer == stats.fallback_failures == 1


def test_infeasible_target_retains_tensors_bit_for_bit() -> None:
    """A target outside every frame is not confused with solver failure."""
    state, _ = product_problem(eigenstate=False)
    mpo = MPO.identity(3)
    original = deepcopy(state)
    with track_conservation() as stats:
        assert not restore_invariant(state, mpo, 2.0, tol=1e-13)
    for actual, expected in zip(state.tensors, original.tensors, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert state.orthogonality_center == original.orthogonality_center
    assert stats.infeasible == stats.fallback_failures == 1
    assert stats.frame_attempts == stats.solver_failures == 0


def test_nonconverged_spectrum_is_not_claimed_infeasible(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unconverged extremum cannot certify absence of a feasible frame."""
    state, mpo = one_site(np.diag([-1.0, 1.0, 2.0, 3.0]), np.array([0, 1, 0, 0]))

    def fail(*_args: object, **_kwargs: object) -> None:
        msg = "test exhaustion"
        raise ArpackNoConvergence(msg, np.array([]), np.empty((4, 0)))

    monkeypatch.setattr(recovery, "eigsh", fail)
    with track_conservation() as stats:
        assert not restore_invariant(state, mpo, 0.0, tol=1e-13)
    assert stats.solver_failures == 1
    assert stats.infeasible == stats.frame_attempts == 0


def test_fallback_can_be_disabled_and_satisfied_targets_do_not_count() -> None:
    """Count recovery only on an actual failed scalar direction."""
    state, mpo = one_site(np.diag([-1.0, 1.0]), np.array([0, 1]))
    with track_conservation() as stats:
        assert not apply_conservation(state, mpo, EnergyTarget(0.0, 1e-13, fallback=False))
        assert not restore_invariant(state, mpo, 1.0, tol=1e-13)
    assert stats.scalar_calls == 2
    assert stats.already_satisfied == 1
    assert stats.fallback_attempts == 0


def test_unresolved_joint_constraints_are_counted_separately() -> None:
    """A coupled failure must not silently apply the single-constraint fallback."""
    state, mpo = one_site(np.diag([-1.0, 1.0]), np.array([0, 1]))
    targets = ConservationTargets(("H", "I"), (mpo, MPO.identity(1)), (0.0, 2.0), 1e-13)
    with track_conservation() as stats:
        assert not apply_conservation(state, mpo, targets)
    assert stats.joint_calls == stats.joint_unresolved == 1
    assert stats.fallback_attempts == stats.scalar_calls == 0


@pytest.mark.parametrize("mode", [EvolutionMode.BUG, EvolutionMode.TDVP])
def test_simulator_returns_independent_run_counters(mode: EvolutionMode, monkeypatch: pytest.MonkeyPatch) -> None:
    """A forced rejected target exercises actual integrators and Result plumbing."""
    state, mpo = product_problem(eigenstate=True)
    monkeypatch.setattr("mqt.yaqs.analog.analog_tjm.capture_conservation_target", lambda *_: EnergyTarget(0.0, 1e-13))
    params = AnalogSimParams(
        elapsed_time=0.1,
        dt=0.1,
        evolution_mode=mode,
        max_bond_dim=1,
        conserve_energy=True,
        get_state=True,
    )
    sim = Simulator(show_progress=False, parallel=False)
    first = sim.run(State.from_mps(state), Hamiltonian.from_mpo(mpo), params)
    snapshot = first.conservation_stats.to_dict()
    second = sim.run(State.from_mps(state), Hamiltonian.from_mpo(mpo), params)
    assert first.conservation_stats is not second.conservation_stats
    assert snapshot == first.conservation_stats.to_dict() == second.conservation_stats.to_dict()
    assert snapshot["fallback_attempts"] == snapshot["fallback_successes"] == 1
    assert first.output_state is not None
    assert abs(expectation_value(first.output_state.mps, mpo)) < 1e-12
    assert params.conserve_fallback is True


@pytest.mark.parametrize("parallel", [False, True])
def test_ensemble_aggregates_worker_counters(*, parallel: bool) -> None:
    """Counts survive process boundaries and are summed, not averaged."""
    params = AnalogSimParams(
        elapsed_time=0.1,
        dt=0.1,
        evolution_mode=EvolutionMode.BUG,
        conserve_energy=True,
        observables=[Observable("z", sites=0)],
    )
    result = Simulator(show_progress=False, parallel=parallel, max_workers=2).run(
        [State(3, initial="zeros"), State(3, initial="ones")], Hamiltonian.ising(3, 1.0, 0.5), params
    )
    assert result.conservation_stats.scalar_calls == 4


def test_tracking_scope_resets_after_exception() -> None:
    """Collectors cannot leak into subsequent runs.

    Raises:
        RuntimeError: Deliberately raised to exercise context cleanup.
    """
    with track_conservation() as outer:
        with track_conservation(), pytest.raises(RuntimeError):
            raise RuntimeError
        assert outer.fallback_attempts == 0
    with track_conservation() as next_run:
        assert next_run.fallback_attempts == 0


def test_fast_path_never_uses_the_spectral_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful original corrections retain the inexpensive path."""
    state, mpo = one_site(np.diag([-1.0, 1.0]), np.sqrt([0.6, 0.4]))

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("The fast path must not enter a spectral solve")

    monkeypatch.setattr("mqt.yaqs.core.methods.conservation.prepare_frame", forbidden)
    with track_conservation() as stats:
        assert restore_invariant(state, mpo, 0.0, tol=1e-13)
    assert stats.scalar_successes == 1
    assert stats.fallback_attempts == 0


def test_program_collects_counts_over_segments() -> None:
    """A program run reports all correction calls in its outer result."""
    params = AnalogSimParams(
        elapsed_time=0.1,
        dt=0.1,
        evolution_mode=EvolutionMode.BUG,
        conserve_energy=True,
    )
    hamiltonian = Hamiltonian.ising(3, 1.0, 0.5)
    result = Simulator(show_progress=False, parallel=False).run(
        State(3, initial="zeros"), [(hamiltonian, params), (hamiltonian, params)], get_state=True
    )
    assert result.conservation_stats.scalar_calls == 4
