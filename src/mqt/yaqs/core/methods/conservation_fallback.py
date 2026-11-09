# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Minimum-distance projection onto a single Hermitian Rayleigh level set.

The secular equation is solved on the interval where ``I + lambda B`` is
positive semidefinite. This selects a global minimum of the displacement, rather
than an arbitrary stationary point. Large frames use only operator actions,
an extremal Krylov eigensolve, and shifted iterative linear solves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, LinearOperator, cg, eigsh, lsmr

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

FrameStatus = Literal["feasible", "infeasible", "solver_failure", "no_minimizer"]
_EPS = np.finfo(float).eps


class _HermitianOperator(LinearOperator):
    """Hermitian operator backed by a matrix-free action."""

    def __init__(self, size: int, action: Callable[[NDArray[np.complex128]], NDArray[np.complex128]]) -> None:
        self.action = action
        super().__init__(dtype=np.dtype(np.complex128), shape=(size, size))

    def _matvec(self, x: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return self.action(np.asarray(x).reshape(-1))

    def _rmatvec(self, x: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return self._matvec(x)


@dataclass
class FrameProblem:
    """Signed constraint operator and its converged smallest eigenpair.

    The sign is chosen so that the initial constraint residual is positive.
    ``spectral_tol`` is used only for feasibility; accepted states must also pass
    the original Rayleigh tolerance in the caller.
    """

    operator: LinearOperator
    center: NDArray[np.complex128]
    minimum: float
    vector: NDArray[np.complex128]
    spectral_tol: float


def prepare_frame(
    apply: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
    center: NDArray[np.complex128],
    target: float,
    *,
    tol: float,
) -> tuple[FrameStatus, FrameProblem | None]:
    """Test the target-facing spectral endpoint without assuming a cyclic start.

    Args:
        apply: Effective Hermitian operator acting on center-shaped tensors.
        center: Current center tensor.
        target: Requested Rayleigh quotient.
        tol: Absolute spectral tolerance.

    Returns:
        Status and the reusable spectral information for a feasible frame.
        A nonconverged eigensolve is never reported as infeasibility.
    """
    shape = center.shape
    c = center.ravel().copy()
    value = float(np.vdot(c, apply(center).ravel()).real / np.vdot(c, c).real)
    sign = 1.0 if value > target else -1.0

    def matvec(vector: NDArray[np.complex128]) -> NDArray[np.complex128]:
        vector = np.asarray(vector).reshape(-1)
        return sign * (apply(vector.reshape(shape)).ravel() - target * vector)

    operator = _HermitianOperator(c.size, matvec)
    # ARPACK's complex path cannot handle k >= N - 1. Only these tiny frames
    # are materialized; there is no full Hilbert-space operator construction.
    if c.size <= 3:
        matrix = np.column_stack([matvec(column) for column in np.eye(c.size, dtype=np.complex128)])
        values, vectors = np.linalg.eigh((matrix + matrix.conj().T) / 2)
        minimum, vector = float(values[0]), vectors[:, 0]
    else:
        # Starting with C alone misses other eigenspaces when C is an eigenstate.
        # An independent deterministic complex vector explores those directions
        # without consuming the simulation's random stream.
        rng = np.random.default_rng(719)
        extra = rng.standard_normal(c.size) + 1j * rng.standard_normal(c.size)
        unit = c / np.linalg.norm(c)
        extra -= unit * np.vdot(unit, extra)
        start = unit + extra / np.linalg.norm(extra)
        try:
            values, vectors = eigsh(operator, k=1, which="SA", v0=start, tol=8 * _EPS, maxiter=max(300, 10 * c.size))
        except (ArpackNoConvergence, ArpackError):
            return "solver_failure", None
        minimum, vector = float(values[0]), vectors[:, 0]
    residual = np.linalg.norm(matvec(vector) - minimum * vector)
    spectral_tol = max(tol, 64 * _EPS * max(1.0, abs(minimum), abs(value), abs(target)))
    if not np.isfinite(minimum) or residual > spectral_tol:
        return "solver_failure", None
    if minimum > spectral_tol:
        return "infeasible", None
    return "feasible", FrameProblem(operator, c, minimum, vector, spectral_tol)


def minimum_displacement(problem: FrameProblem, *, tol: float) -> tuple[FrameStatus, NDArray[np.complex128] | None]:
    """Solve the secular equation, including singular-endpoint solutions.

    Args:
        problem: Feasible frame returned by :func:`prepare_frame`.
        tol: Absolute tolerance of the Rayleigh constraint.

    Returns:
        Status and the least-displacing nonzero tensor, flattened. At a
        semidefinite boundary with zero projection of C onto the nullspace,
        the nonzero-constrained problem has an unattained infimum; this is
        reported as ``no_minimizer``, not replaced by an arbitrary state.
    """
    operator, c = problem.operator, problem.center
    size = c.size
    norm = float(np.linalg.norm(c))
    minimum = problem.minimum
    linear_tol = 8 * _EPS
    maxiter = max(300, 10 * size)

    def constraint(x: NDArray[np.complex128]) -> float:
        return float(np.vdot(x, operator @ x).real)

    def acceptable(x: NDArray[np.complex128]) -> bool:
        squared = float(np.vdot(x, x).real)
        return bool(
            np.all(np.isfinite(x)) and squared > (64 * _EPS * norm) ** 2 and abs(constraint(x)) <= tol * squared
        )

    if minimum >= -problem.spectral_tol:
        # At a spectral boundary feasibility forces X into ker(B). Compute the
        # orthogonal projection, including a degenerate extremal eigenspace.
        removed = lsmr(operator, operator @ c, atol=linear_tol, btol=linear_tol, maxiter=maxiter)[0]
        candidate = np.asarray(c - removed, dtype=np.complex128)
        if np.linalg.norm(candidate) <= 64 * _EPS * norm:
            return "no_minimizer", None
        return ("feasible", candidate) if acceptable(candidate) else ("solver_failure", None)

    pole = -1.0 / minimum

    def shifted(multiplier: float) -> LinearOperator:
        def matvec(x: NDArray[np.complex128]) -> NDArray[np.complex128]:
            return x + multiplier * (operator @ x)

        return _HermitianOperator(size, matvec)

    # If C is orthogonal to the extremal eigenspace, the minimum can occur
    # exactly at the pole. LSMR gives the minimum-norm particular solution;
    # the missing extremal component then enforces the quadratic constraint.
    endpoint_operator = shifted(pole)
    particular = np.asarray(
        lsmr(endpoint_operator, c, atol=linear_tol, btol=linear_tol, maxiter=maxiter)[0], dtype=np.complex128
    )
    endpoint_residual = float(np.linalg.norm(endpoint_operator @ particular - c))
    if endpoint_residual <= 128 * _EPS * norm:
        particular -= problem.vector * np.vdot(problem.vector, particular)
        value = constraint(particular)
        if value >= 0.0:
            candidate = particular + np.sqrt(value / -minimum) * problem.vector
            if acceptable(candidate):
                return "feasible", candidate

    # On the positive-definite branch phi(lambda) is monotone decreasing.
    # Approach the pole geometrically to bracket a regular root.
    lower = 0.0
    upper = 0.0
    for power in range(1, 49):
        upper = pole * (1.0 - 2.0**-power)
        candidate, info = cg(shifted(upper), c, rtol=linear_tol, atol=0.0, maxiter=maxiter)
        if info != 0:
            return "solver_failure", None
        if acceptable(candidate):
            return "feasible", candidate
        if constraint(candidate) < 0:
            break
        lower = upper
    else:
        return "solver_failure", None

    for _ in range(100):
        midpoint = (lower + upper) / 2
        candidate, info = cg(shifted(midpoint), c, rtol=linear_tol, atol=0.0, maxiter=maxiter)
        if info != 0:
            return "solver_failure", None
        if acceptable(candidate):
            return "feasible", candidate
        if constraint(candidate) > 0:
            lower = midpoint
        else:
            upper = midpoint
    return "solver_failure", None
