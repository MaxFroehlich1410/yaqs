# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Known stratified Pauli sampling for an equal mixture of two fixed channels.

The deterministic no-factor stratum follows the repository's FactorSampler;
the dephasing adaptation uses the unchanged benchmark calibration. Concatenating
two weighted ensembles implements linearity of expectation, not a new estimator
principle. See MIXTURE_PROTOCOL.md for attribution and the frozen path budget.
"""

from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.weighted_sampling import FactorSampler
from experiments.overnight_state_preparation_20260913_0108 import channels
from experiments.state_preparation_pilot import circuit_record, digest

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit, StatevectorBackend
    from mqt.yaqs.optimization.statevector import NoiseMaps

NOISE_IDS = ("depolarizing_1s_all", "dephasing_1s_all")
PATHS_PER_CHANNEL = 32
COUNTER_LIMIT = 2**32
Namespaces = Literal["training", "development_diagnostics", "final_evaluation"]
Modes = Literal["gradient", "online", "adaptive_online", "metrics"]


def _validate_circuit(circuit: ParameterizedCircuit) -> None:
    """Require a scalar, finite, fixed circuit before assigning noise factors.

    Raises:
        ValueError: If the register, support, angle map, or parameter index is invalid.
    """
    if type(circuit.num_qubits) is not int or circuit.num_qubits < 1:
        msg = "Expected a positive integer qubit count."
        raise ValueError(msg)
    for gate in circuit.gates:
        if (
            len(gate.sites) not in {1, 2}
            or len(set(gate.sites)) != len(gate.sites)
            or any(type(site) is not int or not 0 <= site < circuit.num_qubits for site in gate.sites)
            or gate.data_map is not None
            or not np.isrealobj([gate.angle_scale, gate.angle_offset])
            or not np.all(np.isfinite([gate.angle_scale, gate.angle_offset]))
            or type(gate.noise_enabled) is not bool
            or (
                gate.param_index is not None
                and (type(gate.param_index) is not int or not 0 <= gate.param_index < circuit.num_params)
            )
        ):
            msg = "Expected finite scalar one-/two-site gates with valid fixed metadata."
            raise ValueError(msg)


class DephasingFactorSampler(FactorSampler):
    """Reuse the exact conditional factor algorithm with calibrated local Z factors.

    Initialization replaces the entire factor law: every field consumed by the
    inherited public draw/maps/ensemble methods is defined here. No depolarizing
    constructor or TJM provider is called. The operator order is original gate
    order, then the original site's order within that gate.
    """

    def __init__(self, circuit: ParameterizedCircuit) -> None:
        """Construct the dephasing Bernoulli law from canonical window coefficients."""
        _validate_circuit(circuit)
        self.gate_count = len(circuit.gates)
        self.factors: list[tuple[int, int, str]] = []
        probabilities = []
        for index, gate in enumerate(circuit.gates):
            if not gate.noise_enabled:
                continue
            coefficient = channels.window(NOISE_IDS[1], len(gate.sites))["coefficient"]
            probability = float(-np.expm1(-2 * coefficient) / 2)
            for site in gate.sites:
                self.factors.append((index, site, "Z"))
                probabilities.append(probability)
        self.probabilities = np.asarray(probabilities, dtype=float)
        logs = np.log1p(-self.probabilities)
        self.no_factor_probability = float(np.exp(logs.sum()))
        self.event_probability = float(-np.expm1(logs.sum()))
        first = self.probabilities * np.exp(np.cumsum(logs) - logs)
        self.first_cdf = np.cumsum(first)
        if self.event_probability > 0:
            self.first_cdf /= self.first_cdf[-1]


def stream_seed(seed: int, update_index: int, channel_index: int, namespace: Namespaces = "training") -> int:
    """Allocate disjoint integer seed intervals, independent of the path budget.

    SHA-256 separates the root seed and purpose; within that digest, a channel
    gets 2**64 integer seeds, an update 2**32, and path r uses base+r through
    FactorSampler.draw. Different update/channel intervals cannot overlap. Hash
    collision resistance separates root seeds and purposes. Both optimizers
    must receive the same root seed and update index for paired opportunities.

    Returns:
        Base seed for conditional paths of one channel at one update.

    Raises:
        ValueError: If a seed, counter, channel, or purpose is unsupported.
    """
    if (
        type(seed) is not int
        or seed < 0
        or type(update_index) is not int
        or not 0 <= update_index < COUNTER_LIMIT
        or type(channel_index) is not int
        or channel_index not in {0, 1}
        or namespace not in {"training", "development_diagnostics", "final_evaluation"}
    ):
        msg = "Expected a nonnegative root seed, bounded update counter, channel 0/1 and explicit namespace."
        raise ValueError(msg)
    prefix = f"overnight.mean2.weighted.v1:{namespace}:{seed}".encode()
    root = int.from_bytes(hashlib.sha256(prefix).digest(), "big")
    return (root << 65) + (channel_index << 64) + (update_index << 32)


@dataclass(frozen=True)
class MixtureBatch:
    """Ordered replay maps and fixed weights for the same mean-two objective.

    The two no-factor paths stay separate and are both charged. Public methods
    deliberately pass maps AND weights to the existing independent backend.
    Sampling times are recorded; callers must time the enclosing numerical call
    and optimizer/checkpoint work separately. Online updates use stale adjoints
    and are not claimed to be unbiased finite-step gradient updates.
    """

    maps: NoiseMaps
    weights: NDArray[np.float64]
    record: dict[str, Any]
    circuit_signature: str
    gate_count: int
    parameter_occurrences: int

    def _validate_engine(self, engine: StatevectorBackend) -> None:
        """Reject stale circuit maps and changes to the benchmark input state.

        Raises:
            ValueError: If the circuit differs or preparation does not start at zero.
        """
        zero = np.zeros_like(engine.initial)
        zero[0] = 1
        if digest(circuit_record(engine.circuit)) != self.circuit_signature or not np.array_equal(engine.initial, zero):
            msg = "Mixture replay requires the recorded circuit and all-zero initial state."
            raise ValueError(msg)

    def gradient(
        self, engine: StatevectorBackend, theta: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float, float]:
        """Return the weighted independent gradient, sampled loss and fidelity."""
        self._validate_engine(engine)
        return engine.gradient(theta, self.maps, weights=self.weights)

    def online_update(
        self, engine: StatevectorBackend, theta: NDArray[np.float64], step_size: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply the existing online sweep to exactly the same weighted objective."""
        self._validate_engine(engine)
        return engine.online_update(theta, step_size, self.maps, weights=self.weights)

    def adaptive_online_update(
        self,
        engine: StatevectorBackend,
        theta: NDArray[np.float64],
        step_size: float,
        second_moment: NDArray[np.float64],
        counts: NDArray[np.int64],
        *,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        """Apply existing adaptive online scaling after averaging weighted signals."""
        self._validate_engine(engine)
        return engine.adaptive_online_update(
            theta, step_size, second_moment, counts, self.maps, weights=self.weights, beta2=beta2, epsilon=epsilon
        )

    def resource_counts(self, mode: Modes) -> dict[str, int]:
        """Count work for one successful existing backend call, including zero weights.

        Circuit applications include each tangent and online gate reevaluation.
        Noise applications count every selected factor in every directional
        pass, including factor products with a physically cancelling result.
        These counts describe the current StatevectorBackend implementation;
        measured CPU, failed attempts and any extra metrics calls remain charged
        separately by the experiment driver.

        Returns:
            Path, directional, gate/tangent and Pauli-factor application counts.

        Raises:
            ValueError: If the operation is not one of the existing supported calls.
        """
        if mode not in {"gradient", "online", "adaptive_online", "metrics"}:
            msg = "Unknown mixture backend operation."
            raise ValueError(msg)
        directions = 1 if mode == "metrics" else 2 if mode == "gradient" else 3
        tangents = 0 if mode == "metrics" else self.parameter_occurrences
        reevaluations = self.parameter_occurrences if directions == 3 else 0
        paths = len(self.maps)
        factors = sum(len(gate.operators) for path in self.maps for gate in path)
        return {
            "paths": paths,
            "directional_passes": directions * paths,
            "circuit_gate_applications": (directions * self.gate_count + reevaluations) * paths,
            "tangent_applications": tangents * paths,
            "noise_factor_applications": directions * factors,
            "total_local_operator_applications": (
                (directions * self.gate_count + reevaluations + tangents) * paths + directions * factors
            ),
            "exact_channel_gradient_calls": 0,
        }


class MixtureSampler:
    """Sample equal-weight depolarizing/dephasing channels on one fixed structure."""

    def __init__(self, circuit: ParameterizedCircuit) -> None:
        """Freeze complete gate metadata and construct both calibrated factor laws.

        Raises:
            ValueError: If the circuit is invalid or the fixed calibrations disagree.
        """
        cpu, wall = time.process_time(), time.perf_counter()
        _validate_circuit(circuit)
        circuit = copy.deepcopy(circuit)
        self.circuit_signature = digest(circuit_record(circuit))
        self.gate_count = len(circuit.gates)
        self.parameter_occurrences = sum(gate.param_index is not None for gate in circuit.gates)
        calibration = CalibratedLocalDepolarizingProvider()
        self.windows = {
            noise_id: {str(arity): copy.deepcopy(channels.window(noise_id, arity)) for arity in (1, 2)}
            for noise_id in NOISE_IDS
        }
        for arity, coefficient in ((1, calibration.one_qubit_strength), (2, calibration.two_qubit_strength)):
            if not np.isclose(coefficient, self.windows[NOISE_IDS[0]][str(arity)]["coefficient"], rtol=0, atol=2e-14):
                msg = "Local depolarizing calibration disagrees with the canonical fixed channel."
                raise ValueError(msg)
        self.samplers = (FactorSampler(circuit, calibration), DephasingFactorSampler(circuit))
        self.setup_cpu_seconds = time.process_time() - cpu
        self.setup_wall_seconds = time.perf_counter() - wall

    def ensemble(
        self,
        paths_per_channel: int = PATHS_PER_CHANNEL,
        *,
        seed: int,
        update_index: int,
        namespace: Namespaces = "training",
    ) -> MixtureBatch:
        """Return two deterministic strata plus fresh conditional trajectories.

        Args:
            paths_per_channel: Total paths for EACH channel; preregistered default 32.
            seed: Root seed shared between paired optimizer arms.
            update_index: Fresh stream counter, persisted across exact resume.
            namespace: Disjoint training, diagnostic, or evaluation purpose.

        Returns:
            Depolarizing paths followed by dephasing paths, with each weight halved.

        Raises:
            ValueError: If counts, seeds, counters or the namespace are invalid.
        """
        cpu, wall = time.process_time(), time.perf_counter()
        if type(paths_per_channel) is not int or not 2 <= paths_per_channel < COUNTER_LIMIT:
            msg = "Each channel needs an integer path count in [2, 2**32)."
            raise ValueError(msg)
        seeds = [stream_seed(seed, update_index, index, namespace) for index in range(2)]
        draws = [sampler.ensemble(paths_per_channel, base) for sampler, base in zip(self.samplers, seeds, strict=True)]
        maps = tuple(path for draw, _ in draws for path in draw)
        weights = np.concatenate([probabilities / 2 for _, probabilities in draws])
        weights.flags["W"] = False
        record: dict[str, Any] = {
            "version": "exact_factor_mean2.v1",
            "execution_channel": "exact_independent_pauli_factors",
            "calibration_convention": "calibrated_exact",
            "circuit_signature": self.circuit_signature,
            "channel_order": list(NOISE_IDS),
            "channel_weights": [0.5, 0.5],
            "paths_per_channel": paths_per_channel,
            "total_paths": len(maps),
            "deterministic_strata": 2,
            "conditional_paths": 2 * (paths_per_channel - 1),
            "root_seed": seed,
            "update_index": update_index,
            "namespace": namespace,
            # Decimal strings preserve the full integer seeds through non-Python JSON readers.
            "channel_base_seeds": [str(value) for value in seeds],
            "factor_counts": [len(sampler.factors) for sampler in self.samplers],
            "no_factor_probabilities": [sampler.no_factor_probability for sampler in self.samplers],
            "conditional_event_probabilities": [sampler.event_probability for sampler in self.samplers],
            "sampling_cpu_seconds": time.process_time() - cpu,
            "sampling_wall_seconds": time.perf_counter() - wall,
            "bias_status": "gradient_unbiased_at_fixed_parameters; online_finite_step_not_claimed_unbiased",
        }
        return MixtureBatch(maps, weights, record, self.circuit_signature, self.gate_count, self.parameter_occurrences)
