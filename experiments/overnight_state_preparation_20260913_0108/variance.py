# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Anchor-gradient control variates for the fixed local Pauli channel.

Adapted from Johnson and Zhang, NIPS 2013, ``Accelerating Stochastic Gradient
Descent using Predictive Variance Reduction``. This module diagnoses estimators;
it does not claim a new optimizer or change final benchmark noise semantics.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from benchmarks.state_preparation.exact_density import density_loss_gradient
from benchmarks.state_preparation.weighted_sampling import FactorSampler
from experiments.state_preparation_pilot import circuit_record

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
    from mqt.yaqs.optimization import StatevectorBackend
    from mqt.yaqs.optimization.statevector import NoiseMaps


@dataclass(frozen=True)
class Estimate:
    """One gradient estimate with explicit measured and algorithmic costs.

    Attributes:
        gradient: Estimated gradient of channel-averaged infidelity.
        loss: Control-variate loss estimate; it need not lie in [0, 1].
        paths: Number of propagated statevector paths, counting both parameters.
        directional_passes: Forward plus adjoint trajectory-direction passes.
        exact_channel_evaluations: Exact gradient/fidelity oracle calls.
        cpu_seconds: Process CPU for setup, sampling and numerical work.
        wall_seconds: Elapsed wall time for the same interval.
        seed: First per-path seed, or None for deterministic computation.
        bias_status: Conditional bias description, excluding numerical error.
    """

    gradient: NDArray[np.float64]
    loss: float
    paths: int
    directional_passes: int
    exact_channel_evaluations: int
    cpu_seconds: float
    wall_seconds: float
    seed: int | None
    bias_status: str


@dataclass(frozen=True)
class Snapshot:
    """Frozen anchor, its expectation estimate, and complete one-time cost.

    Arrays are private copies. Reusing a sampled snapshot retains its random
    error throughout the epoch; correction resampling does not remove it.
    """

    theta: NDArray[np.float64]
    estimate: Estimate
    problem_signature: str


def _signature(engine: StatevectorBackend, calibration: CalibratedLocalDepolarizingProvider) -> str:
    """Identify the complete fixed problem used by a snapshot."""
    digest = hashlib.sha256(json.dumps(circuit_record(engine.circuit), sort_keys=True).encode())
    digest.update(engine.target.tobytes())
    digest.update(engine.initial.tobytes())
    digest.update(calibration.content_checksum.encode())
    return digest.hexdigest()


def _validate_problem(engine: StatevectorBackend) -> None:
    """Enforce the benchmark's all-zero initial state.

    Raises:
        ValueError: If the supplied backend has a different initial state.
    """
    zero = np.zeros_like(engine.initial)
    zero[0] = 1
    if not np.array_equal(engine.initial, zero):
        msg = "Anchor estimation requires the benchmark all-zero initial state."
        raise ValueError(msg)


def _validate_sampling(paths: int, seed: int) -> None:
    """Require an explicit valid weighted trajectory budget.

    Raises:
        ValueError: If the path count or seed is invalid.
    """
    if type(paths) is not int or paths < 2 or type(seed) is not int or seed < 0:
        msg = "Weighted paths must be an integer >= 2 and seed a nonnegative integer."
        raise ValueError(msg)


def weighted_estimate(
    engine: StatevectorBackend,
    theta: NDArray[np.float64],
    calibration: CalibratedLocalDepolarizingProvider,
    *,
    paths: int,
    seed: int,
) -> Estimate:
    """Return the ordinary weighted estimator with full per-call cost.

    For an equal-direction comparison with a control variate using R paths,
    use paths=2*R here. Snapshot construction remains additional charged work.
    """
    _validate_problem(engine)
    _validate_sampling(paths, seed)
    cpu, wall = time.process_time(), time.perf_counter()
    sampler = FactorSampler(engine.circuit, calibration)
    maps, weights = sampler.ensemble(paths, seed)
    gradient, loss, _ = engine.gradient(theta, maps, weights=weights)
    return Estimate(
        gradient=gradient,
        loss=loss,
        paths=paths,
        directional_passes=2 * paths,
        exact_channel_evaluations=0,
        cpu_seconds=time.process_time() - cpu,
        wall_seconds=time.perf_counter() - wall,
        seed=seed,
        bias_status="unbiased_for_fixed_parameters",
    )


def build_snapshot(
    engine: StatevectorBackend,
    theta: NDArray[np.float64],
    calibration: CalibratedLocalDepolarizingProvider,
    *,
    snapshot_paths: int = 0,
    seed: int = 0,
) -> Snapshot:
    """Create an exact anchor or a separately sampled weighted snapshot.

    Args:
        engine: Fixed zero-input circuit and target.
        theta: Anchor parameters, copied and retained read-only.
        calibration: Fixed parameter-independent calibrated local channel.
        snapshot_paths: Zero selects the q1-q6 exact oracle; >=2 samples paths.
        seed: Independent snapshot seed when snapshot_paths is positive.

    Returns:
        Authenticated immutable snapshot and separately charged creation cost.

    Raises:
        ValueError: If the snapshot budget is invalid.
    """
    _validate_problem(engine)
    if type(snapshot_paths) is not int or snapshot_paths < 0 or snapshot_paths == 1:
        msg = "Snapshot paths must be zero or an integer >= 2."
        raise ValueError(msg)
    if snapshot_paths:
        estimate = weighted_estimate(engine, theta, calibration, paths=snapshot_paths, seed=seed)
    else:
        cpu, wall = time.process_time(), time.perf_counter()
        gradient, loss, _ = density_loss_gradient(engine.circuit, theta, engine.target, calibration)
        estimate = Estimate(
            gradient=gradient,
            loss=loss,
            paths=0,
            directional_passes=0,
            exact_channel_evaluations=1,
            cpu_seconds=time.process_time() - cpu,
            wall_seconds=time.perf_counter() - wall,
            seed=None,
            bias_status="exact_channel_expectation",
        )
    anchor = np.array(theta, dtype=float, copy=True)
    anchor.setflags(write=False)
    estimate.gradient.setflags(write=False)
    return Snapshot(anchor, estimate, _signature(engine, calibration))


def control_variate_from_maps(
    engine: StatevectorBackend,
    theta: NDArray[np.float64],
    calibration: CalibratedLocalDepolarizingProvider,
    snapshot: Snapshot,
    maps: NoiseMaps,
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Evaluate anchor + paired current-minus-anchor gradients on common maps.

    An exact anchor yields an unbiased estimator for parameter-independent maps.
    A sampled anchor gives conditional bias equal to its frozen estimation error.
    The caller must supply fresh maps independent of sampled anchor randomness.

    Returns:
        Gradient and loss control variates, preserving signed corrections.

    Raises:
        ValueError: If the snapshot belongs to another circuit/target/channel.
    """
    if snapshot.problem_signature != _signature(engine, calibration):
        msg = "Snapshot problem signature differs from circuit, target, input or calibration."
        raise ValueError(msg)
    current_gradient, current_loss, _ = engine.gradient(theta, maps, weights=weights)
    anchor_gradient, anchor_loss, _ = engine.gradient(snapshot.theta, maps, weights=weights)
    return (
        snapshot.estimate.gradient + (current_gradient - anchor_gradient),
        snapshot.estimate.loss + (current_loss - anchor_loss),
    )


def control_variate_estimate(
    engine: StatevectorBackend,
    theta: NDArray[np.float64],
    calibration: CalibratedLocalDepolarizingProvider,
    snapshot: Snapshot,
    *,
    paths: int,
    seed: int,
) -> Estimate:
    """Draw one independent correction batch and account for both gradients.

    Snapshot cost is intentionally separate: charge it once per refresh and
    report amortized cost only with the actual number of corrections performed.

    Returns:
        One paired control-variate estimate and its cost.

    Raises:
        ValueError: If correction seeds overlap the sampled snapshot seed range.
    """
    _validate_problem(engine)
    _validate_sampling(paths, seed)
    previous = snapshot.estimate
    if previous.seed is not None and max(seed, previous.seed) < min(
        seed + paths - 1, previous.seed + previous.paths - 1
    ):
        msg = "Snapshot and correction per-path seed ranges must be disjoint."
        raise ValueError(msg)
    cpu, wall = time.process_time(), time.perf_counter()
    sampler = FactorSampler(engine.circuit, calibration)
    maps, weights = sampler.ensemble(paths, seed)
    gradient, loss = control_variate_from_maps(engine, theta, calibration, snapshot, maps, weights)
    return Estimate(
        gradient=gradient,
        loss=loss,
        paths=2 * paths,
        directional_passes=4 * paths,
        exact_channel_evaluations=0,
        cpu_seconds=time.process_time() - cpu,
        wall_seconds=time.perf_counter() - wall,
        seed=seed,
        bias_status="unbiased_given_exact_anchor"
        if previous.exact_channel_evaluations
        else "conditional_bias_equals_frozen_snapshot_error",
    )
