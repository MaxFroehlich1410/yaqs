# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Known entropic risk over two complete fixed-channel expected infidelities.

Risk is applied across calibrated channel expectations, never across selected
noise trajectories. The channel adjoints are the unchanged ensemble oracle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from experiments.overnight_state_preparation_20260913_0108 import ensemble_objective

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

TRAINING_NOISE_IDS = ("depolarizing_1s_all", "dephasing_1s_all")
TEMPERATURES = (0.02, 0.1)


@dataclass(frozen=True)
class Result:
    """Retain the exact scalar objective, adjoint and complete component work."""

    loss: float
    gradient: NDArray[np.float64]
    weights: NDArray[np.float64]
    component_losses: NDArray[np.float64]
    temperature: float | None
    individual_channel_calls: int
    density_direction_passes: int
    cpu_seconds: float
    wall_seconds: float


def validate_temperature(temperature: float | None) -> None:
    """Require positive finite risk temperature, or explicit None for the mean.

    Raises:
        ValueError: If the supplied temperature is not a positive finite scalar.
    """
    if temperature is not None and (
        isinstance(temperature, (bool, np.bool_))
        or not isinstance(temperature, (float, int, np.floating, np.integer))
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        msg = "Risk temperature must be finite and positive; None explicitly selects the mean."
        raise ValueError(msg)


def aggregate(
    losses: NDArray[np.float64], gradients: NDArray[np.float64], temperature: float | None
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Return stable log-mean-exp infidelity, its gradient and softmax weights.

    At temperature None, return the equally weighted mean control. A maximum
    shift prevents exponential overflow, while expm1/log1p preserve the mean
    limit at very large temperature. Harmless negative-exponent underflow makes
    the corresponding softmax weight zero at extremely small temperature.

    Args:
        losses: Nonempty real finite complete-channel infidelities.
        gradients: One real finite gradient row per channel; zero columns allowed.
        temperature: Positive finite risk scale, or None for the mean control.

    Returns:
        Scalar loss, independent read-only gradient and read-only channel weights.

    Raises:
        ValueError: If arrays are malformed, nonfinite or outside infidelity bounds.
    """
    validate_temperature(temperature)
    if (
        np.iscomplexobj(losses)
        or np.iscomplexobj(gradients)
        or losses.ndim != 1
        or not losses.size
        or gradients.ndim != 2
        or gradients.shape[0] != losses.size
        or not np.all(np.isfinite(losses))
        or not np.all(np.isfinite(gradients))
        or np.any(losses < -1e-10)
        or np.any(losses > 1 + 1e-10)
    ):
        msg = "Risk aggregation requires finite real infidelities and matching gradient rows."
        raise ValueError(msg)
    if temperature is None:
        value = float(np.mean(losses))
        weights = np.full(losses.size, 1 / losses.size)
    else:
        maximum = float(np.max(losses))
        with np.errstate(over="ignore", under="ignore"):
            shifted = (losses - maximum) / temperature
            exponentials = np.exp(shifted)
            correction = np.log1p(np.mean(np.expm1(shifted)))
        value = maximum + temperature * float(correction)
        weights = exponentials / np.sum(exponentials)
    gradient = np.asarray(weights @ gradients, dtype=float)
    gradient.setflags(write=False)
    weights.setflags(write=False)
    return value, gradient, weights


def loss_gradient(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    *,
    temperature: float | None = 0.02,
    progress: Callable[[str, str], None] | None = None,
) -> Result:
    """Return the exact two-channel entropic-risk gradient with explicit work.

    Args:
        circuit: Same physical circuit, prepared from zero under both channels.
        theta: Shared scalar parameters; not modified.
        target: Common normalized pure target.
        temperature: Positive finite scale or None for the mean control.
        progress: Optional (channel, start/complete) callback for interrupted costs.

    Returns:
        Exact objective and component diagnostics; two channel calls, four directions.

    Raises:
        ValueError: If temperature or the unchanged component oracle domain is invalid.
    """
    validate_temperature(temperature)
    cpu, wall = time.process_time(), time.perf_counter()
    gradients, losses = [], []
    for noise in TRAINING_NOISE_IDS:
        if progress is not None:
            progress(noise, "start")
        gradient, loss, _ = ensemble_objective.loss_gradient(circuit, theta, target, noise)
        if progress is not None:
            progress(noise, "complete")
        gradients.append(gradient)
        losses.append(loss)
    components = np.asarray(losses, dtype=float)
    value, gradient, weights = aggregate(components, np.asarray(gradients), temperature)
    components.setflags(write=False)
    return Result(
        value, gradient, weights, components, temperature, 2, 4, time.process_time() - cpu, time.perf_counter() - wall
    )
