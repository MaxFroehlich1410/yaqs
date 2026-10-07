# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact conditional sampling for the fixed state-independent local Pauli channel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from benchmarks.state_preparation.pilot_components import PAULIS
from mqt.yaqs.optimization.krotov import KrotovNoiseMap

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
    from mqt.yaqs.optimization import ParameterizedCircuit
    from mqt.yaqs.optimization.statevector import NoiseMaps


class FactorSampler:
    """Condition independent Pauli factors on at least one selected factor.

    Only the calibrated, parameter-independent, one-/two-qubit local Pauli
    channel is supported, with a unit noise window after enabled gates.
    This experimental sampler does not replace any production provider.
    """

    def __init__(self, circuit: ParameterizedCircuit, calibration: CalibratedLocalDepolarizingProvider) -> None:
        """Precompute factor probabilities and the conditional first-event CDF.

        Raises:
            ValueError: If noise-enabled gates are data-dependent or not one-/two-qubit.
        """
        self.gate_count = len(circuit.gates)
        self.factors: list[tuple[int, int, str]] = []
        probabilities = []
        for index, gate in enumerate(circuit.gates):
            if not gate.noise_enabled:
                continue
            if len(gate.sites) not in {1, 2} or gate.data_map is not None:
                msg = "Only fixed one-/two-qubit local Pauli noise is supported."
                raise ValueError(msg)
            strength = calibration.one_qubit_strength if len(gate.sites) == 1 else calibration.two_qubit_strength
            probability = -np.expm1(-2 * strength) / 2
            for site in gate.sites:
                for label in "XYZ":
                    self.factors.append((index, site, label))
                    probabilities.append(probability)
        self.probabilities = np.asarray(probabilities, dtype=float)
        logs = np.log1p(-self.probabilities)
        self.no_factor_probability = float(np.exp(logs.sum()))
        self.event_probability = float(-np.expm1(logs.sum()))
        first = self.probabilities * np.exp(np.cumsum(logs) - logs)
        self.first_cdf = np.cumsum(first)
        if self.event_probability > 0:
            self.first_cdf /= self.first_cdf[-1]

    def draw(self, count: int, seed: int) -> NDArray[np.bool_]:
        """Draw conditional event masks with trajectory-index-stable seeds.

        The first selected factor has mass p_j prod_{i<j}(1-p_i)/(1-p0).
        Later factors keep their original independent Bernoulli probabilities.
        There is no rejection loop and no approximation to multi-factor events.

        Returns:
            Conditional factor masks in original gate/site/XYZ order.

        Raises:
            ValueError: If count/seed is invalid or the conditioning event has zero probability.
        """
        if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
            msg = "Count and seed must be positive/nonnegative integers."
            raise ValueError(msg)
        if self.event_probability <= 0:
            msg = "Cannot condition on a zero-probability event."
            raise ValueError(msg)
        masks = np.zeros((count, len(self.factors)), dtype=bool)
        for index in range(count):
            rng = np.random.default_rng(seed + index)
            first = int(np.searchsorted(self.first_cdf, rng.random(), side="right"))
            masks[index, first] = True
            masks[index, first + 1 :] = rng.random(len(self.factors) - first - 1) < self.probabilities[first + 1 :]
        return masks

    def maps(self, masks: NDArray[np.bool_]) -> NoiseMaps:
        """Translate nonempty conditional masks to replayable unitary maps.

        Returns:
            Per-path, per-gate unitary replay maps, retaining all selected factors.

        Raises:
            ValueError: If masks have invalid shape/type or include a no-factor path.
        """
        if (
            masks.ndim != 2
            or masks.shape[1] != len(self.factors)
            or masks.dtype != bool
            or not len(masks)
            or not np.all(masks.any(axis=1))
        ):
            msg = "Expected nonempty Boolean conditional masks with at least one selected factor each."
            raise ValueError(msg)
        result = []
        for mask in masks:
            operators: list[list[tuple[NDArray[np.complex128], tuple[int, ...]]]] = [[] for _ in range(self.gate_count)]
            for selected in np.flatnonzero(mask):
                gate, site, label = self.factors[selected]
                operators[gate].append((PAULIS[label], (site,)))
            result.append(
                tuple(KrotovNoiseMap(operators=tuple(ops), source_gate_index=k) for k, ops in enumerate(operators))
            )
        return tuple(result)

    def ensemble(self, count: int, seed: int) -> tuple[NoiseMaps, NDArray[np.float64]]:
        """Return one deterministic path plus count-1 conditional paths and weights.

        The zero-noise limit retains the requested path budget with zero-weight
        duplicate paths. Weights are fixed probabilities, independent of theta.

        Returns:
            Replay maps and normalized weights, with the deterministic path first.

        Raises:
            ValueError: If fewer than two paths or an invalid seed are requested.
        """
        if type(count) is not int or count < 2 or type(seed) is not int or seed < 0:
            msg = "Weighted sampling requires two total paths and a nonnegative integer seed."
            raise ValueError(msg)
        zero = (KrotovNoiseMap(),) * self.gate_count
        weights = np.full(count, self.event_probability / (count - 1))
        weights[0] = self.no_factor_probability
        conditional = (zero,) * (count - 1) if self.event_probability == 0 else self.maps(self.draw(count - 1, seed))
        return (zero, *conditional), weights
