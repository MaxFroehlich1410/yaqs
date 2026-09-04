# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Circuit and finite-trajectory statistics for benchmark results."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mqt.yaqs.optimization import ParameterizedCircuit

    from .ballarin import BallarinCircuitMaterialization

_NORMAL_95_PERCENT_QUANTILE = 1.959963984540054


@dataclass(frozen=True, slots=True)
class GateCounts:
    """Dependency depth and arity/name counts for one circuit."""

    depth: int
    one_qubit: int
    two_qubit: int
    by_name: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CircuitStatistics:
    """Logical and optional Ballarin-native circuit statistics."""

    num_parameters: int
    logical: GateCounts
    native: GateCounts | None = None
    pre_pruning_rzz_count: int = 0
    pruned_rzz_count: int = 0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "logical": self.logical.to_dict(),
            "native": None if self.native is None else self.native.to_dict(),
            "num_parameters": self.num_parameters,
            "pre_pruning_rzz_count": self.pre_pruning_rzz_count,
            "pruned_rzz_count": self.pruned_rzz_count,
        }


@dataclass(frozen=True, slots=True)
class FidelityEstimate:
    """Mean fidelity and finite-sample uncertainty."""

    count: int
    mean: float
    standard_deviation: float | None
    standard_error: float | None
    confidence_interval_lower: float | None
    confidence_interval_upper: float | None
    values: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def gate_counts(circuit: ParameterizedCircuit) -> GateCounts:
    """Return gate and dependency-layer counts for a circuit.

    Raises:
        ValueError: If a circuit gate does not act on one or two sites.
    """
    site_depths = [0] * circuit.num_qubits
    by_name: Counter[str] = Counter()
    one_qubit = 0
    two_qubit = 0
    for gate in circuit.gates:
        arity = len(gate.sites)
        if arity not in {1, 2}:
            msg = f"Cannot count unsupported {arity}-site gate {gate.name!r}."
            raise ValueError(msg)
        layer = 1 + max(site_depths[site] for site in gate.sites)
        for site in gate.sites:
            site_depths[site] = layer
        one_qubit += arity == 1
        two_qubit += arity == 2
        by_name[gate.name] += 1
    return GateCounts(max(site_depths, default=0), one_qubit, two_qubit, dict(sorted(by_name.items())))


def circuit_statistics(
    logical_circuit: ParameterizedCircuit,
    native: BallarinCircuitMaterialization | None = None,
) -> CircuitStatistics:
    """Return logical and optional native circuit statistics."""
    return CircuitStatistics(
        num_parameters=logical_circuit.num_params,
        logical=gate_counts(logical_circuit),
        native=None if native is None else gate_counts(native.circuit),
        pre_pruning_rzz_count=0 if native is None else native.pre_pruning_rzz_count,
        pruned_rzz_count=0 if native is None else native.pruned_rzz_count,
    )


def fidelity_estimate(values: tuple[float, ...] | list[float]) -> FidelityEstimate:
    """Summarize raw trajectory fidelities with a clipped normal 95% interval.

    Returns:
        Mean, sample uncertainty, clipped confidence interval, and raw values.

    Raises:
        ValueError: If no values are supplied or a value is non-finite or lies
            materially outside the fidelity interval.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        msg = "At least one scalar trajectory fidelity is required."
        raise ValueError(msg)
    if not np.all(np.isfinite(array)) or np.any(array < -1e-9) or np.any(array > 1.0 + 1e-9):
        msg = "Trajectory fidelities must be finite values in [0, 1]."
        raise ValueError(msg)
    array = np.clip(array, 0.0, 1.0)
    mean = float(np.mean(array))
    if array.size == 1:
        return FidelityEstimate(1, mean, None, None, None, None, (float(array[0]),))
    standard_deviation = float(np.std(array, ddof=1))
    standard_error = standard_deviation / math.sqrt(array.size)
    radius = _NORMAL_95_PERCENT_QUANTILE * standard_error
    return FidelityEstimate(
        count=int(array.size),
        mean=mean,
        standard_deviation=standard_deviation,
        standard_error=standard_error,
        confidence_interval_lower=max(0.0, mean - radius),
        confidence_interval_upper=min(1.0, mean + radius),
        values=tuple(float(value) for value in array),
    )


__all__ = [
    "CircuitStatistics",
    "FidelityEstimate",
    "GateCounts",
    "circuit_statistics",
    "fidelity_estimate",
    "gate_counts",
]
