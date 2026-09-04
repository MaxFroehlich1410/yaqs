# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Small adapter boundary for state-preparation optimization methods."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from mqt.yaqs.optimization import (
    KrotovOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    train_krotov_state_preparation_batch,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    """Method-independent optimization budget and numerical settings."""

    iterations: int = 100
    step_size: float = 0.2
    optimizer_seed: int = 12
    max_bond_dimension: int | None = None
    svd_threshold: float = 0.0

    def __post_init__(self) -> None:
        """Validate settings shared by benchmark method adapters.

        Raises:
            ValueError: If a budget, seed, or numerical setting is invalid.
        """
        if self.iterations < 0:
            msg = "iterations must be nonnegative."
            raise ValueError(msg)
        if not np.isfinite(self.step_size) or self.step_size <= 0.0:
            msg = "step_size must be positive and finite."
            raise ValueError(msg)
        if self.optimizer_seed < 0:
            msg = "optimizer_seed must be nonnegative."
            raise ValueError(msg)
        if self.max_bond_dimension is not None and self.max_bond_dimension < 1:
            msg = "max_bond_dimension must be positive or None."
            raise ValueError(msg)
        if not np.isfinite(self.svd_threshold) or self.svd_threshold < 0.0:
            msg = "svd_threshold must be finite and nonnegative."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """Final parameters and serializable metadata returned by a method."""

    theta: NDArray[np.float64]
    trace: Mapping[str, tuple[float | int | str, ...]]
    wall_time_seconds: float


class StatePreparationMethod(Protocol):
    """Interface implemented by every benchmark optimization method."""

    method_id: str

    def train(
        self,
        circuit: ParameterizedCircuit,
        target: NDArray[np.complex128],
        initial_theta: NDArray[np.float64],
        settings: TrainingSettings,
    ) -> TrainingOutcome:
        """Optimize one target and return the final parameter vector."""


@dataclass(frozen=True, slots=True)
class KrotovBatchMethod:
    """Reference full-batch noiseless Krotov-inspired optimizer adapter."""

    method_id: str = "krotov_batch"

    def train(
        self,
        circuit: ParameterizedCircuit,
        target: NDArray[np.complex128],
        initial_theta: NDArray[np.float64],
        settings: TrainingSettings,
    ) -> TrainingOutcome:
        """Return the result of the reference optimizer."""
        truncation = KrotovTruncation(
            max_bond_dim=settings.max_bond_dimension,
            svd_threshold=settings.svd_threshold,
        )
        options = KrotovOptions(
            max_iterations=settings.iterations,
            batch_step_size=settings.step_size,
            batch_schedule="constant",
            seed=settings.optimizer_seed,
            truncation=truncation,
        )
        started = time.perf_counter()
        result = train_krotov_state_preparation_batch(
            circuit,
            target,
            initial_theta=np.asarray(initial_theta, dtype=np.float64),
            options=options,
        )
        elapsed = time.perf_counter() - started
        trace = {key: tuple(values) for key, values in result.trace.items()}
        trace["method"] = (self.method_id,)
        return TrainingOutcome(result.theta.copy(), trace, elapsed)


DEFAULT_METHODS: dict[str, StatePreparationMethod] = {"krotov_batch": KrotovBatchMethod()}

__all__ = [
    "DEFAULT_METHODS",
    "KrotovBatchMethod",
    "StatePreparationMethod",
    "TrainingOutcome",
    "TrainingSettings",
]
