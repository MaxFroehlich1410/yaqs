# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Common numerical interface for the unchanged state-preparation optimizers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..core.data_structures.mps import MPS
from .mps_backend import MPSBackend
from .statevector import StatevectorBackend

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .gate_noise import GateNoiseProvider
    from .krotov import KrotovTJMOptions, KrotovTruncation
    from .parameterized_circuit import ParameterizedCircuit
    from .trajectories import Estimator, NoiseMaps


class PreparationBackend(Protocol):
    """Operations required by construction, pruning, refinement and assessment."""

    circuit: ParameterizedCircuit

    def metrics(
        self, theta: NDArray, maps: NoiseMaps | None = None, *, weights: NDArray | None = None
    ) -> tuple[float, float]:
        """Return weighted loss and fidelity."""
        ...

    def gradient(
        self,
        theta: NDArray,
        maps: NoiseMaps | None = None,
        *,
        estimator: Estimator = "independent",
        weights: NDArray | None = None,
    ) -> tuple[NDArray, float, float]:
        """Return weighted signal, loss and fidelity."""
        ...

    def online_update(
        self,
        theta: NDArray,
        step_size: float,
        maps: NoiseMaps | None = None,
        *,
        estimator: Estimator = "independent",
        weights: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Return a sequential stale-adjoint sweep and accumulated signals."""
        ...

    def deletion_fidelities(self, theta: NDArray) -> tuple[NDArray, float]:
        """Return individual gate-deletion fidelities and the original fidelity."""
        ...

    def sample_noise_maps(
        self, theta: NDArray, options: KrotovTJMOptions, provider: GateNoiseProvider, *, iteration: int = 0
    ) -> NoiseMaps:
        """Sample state-independent histories without propagation."""
        ...

    def compress_no_jump_maps(self, maps: NoiseMaps) -> tuple[NoiseMaps, NDArray]:
        """Merge identical empty histories with empirical multiplicities."""
        ...


def create_backend(
    circuit: ParameterizedCircuit,
    target: MPS | NDArray,
    *,
    backend: str = "statevector",
    truncation: KrotovTruncation | None = None,
    compress_target: bool = False,
) -> PreparationBackend:
    """Construct an explicit backend without silently densifying MPS targets.

    Returns:
        An engine implementing the common optimizer interface.

    Raises:
        ValueError: If the backend or representation-specific options conflict.
    """
    if backend == "mps":
        return MPSBackend(circuit, target, truncation=truncation, compress_target=compress_target)
    if backend != "statevector" or isinstance(target, MPS) or truncation is not None or compress_target:
        msg = "Invalid backend or MPS-only options supplied to a statevector backend."
        raise ValueError(msg)
    return StatevectorBackend(circuit, target)
