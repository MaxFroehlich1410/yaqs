# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact local noise, small-system evaluation, and pruning for the q6 pilot."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator, Pauli

from mqt.yaqs.optimization import (
    GateNoiseContext,
    LocalOperator,
    ParameterizedCircuit,
    RandomUnitaryInstruction,
    state_preparation_metrics,
)

from .calibration import CalibratedLocalDepolarizingProvider

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

PAULIS = {label: Pauli(label).to_matrix() for label in "XYZ"}
PAULI_OPERATORS = {label: Operator(matrix) for label, matrix in PAULIS.items()}


@dataclass(frozen=True)
class ExactLocalPauliProvider:
    """Sample each commuting Pauli-Lindblad factor exactly in a unit window."""

    calibration: CalibratedLocalDepolarizingProvider = field(default_factory=CalibratedLocalDepolarizingProvider)

    def scientific_dict(self) -> dict[str, object]:
        """Return exact sampler provenance, distinct from the legacy provider."""
        return {
            **self.calibration.scientific_dict(),
            "version": "pilot.exact_local_pauli_factors.v1",
            "execution_channel": "exact_commuting_pauli_factors",
            "sampling_rule": "independent Bernoulli((1-exp(-2*a))/2) per site and X,Y,Z",
        }

    def __call__(self, context: GateNoiseContext, rng: np.random.Generator, /) -> RandomUnitaryInstruction:
        """Return state-independent exact factor outcomes in site/XYZ order."""
        strength = self.calibration.one_qubit_strength if context.arity == 1 else self.calibration.two_qubit_strength
        probability = -math.expm1(-2 * strength) / 2
        operators = []
        labels = []
        for site in context.sites:
            for label in "XYZ":
                selected = rng.random() < probability
                labels.append(f"{site}:{label}:{int(selected)}")
                if selected:
                    operators.append(LocalOperator(PAULIS[label], (site,), label=label))
        return RandomUnitaryInstruction(
            tuple(operators), channel_id="pilot.exact_local_pauli_factors.v1", outcome_labels=tuple(labels)
        )


def exact_density_fidelity(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    provider: ExactLocalPauliProvider,
) -> float:
    """Evaluate the exact channel, without trajectory sampling or training.

    Args:
        circuit: Scalar gate circuit, including its retained noise windows.
        theta: Parameters in the circuit's current compact indexing.
        target: Normalized little-endian target vector.
        provider: Calibrated local Pauli-Lindblad channel.

    Returns:
        The exact channel-averaged preparation fidelity.

    Raises:
        ValueError: If the system exceeds the pilot's small-system limit.
        FloatingPointError: If density-matrix or fidelity checks fail.
    """
    if not 1 <= circuit.num_qubits <= 6:
        msg = "The pilot density oracle is limited to at most six qubits."
        raise ValueError(msg)
    rho = DensityMatrix.from_label("0" * circuit.num_qubits)
    for gate in circuit.gates:
        matrix, sites = circuit.gate_matrix(gate, theta)
        rho = rho.evolve(Operator(matrix), qargs=list(reversed(sites)))
        if not gate.noise_enabled:
            continue
        calibration = provider.calibration
        strength = calibration.one_qubit_strength if len(sites) == 1 else calibration.two_qubit_strength
        # Independent exact local depolarization: a different derivation from
        # the sampler's Bernoulli factors, but the same Lindblad exponential.
        weight = -math.expm1(-4 * strength) / 4
        for site in sites:
            noisy = (1 - 3 * weight) * rho.data
            for pauli in PAULI_OPERATORS.values():
                noisy += weight * rho.evolve(pauli, qargs=[site]).data
            rho = DensityMatrix(noisy)
    fidelity = float(np.vdot(target, rho.data @ target).real)
    if (
        not np.all(np.isfinite(rho.data))
        or abs(np.trace(rho.data) - 1) > 1e-10
        or np.max(np.abs(rho.data - rho.data.conj().T)) > 1e-10
        or np.linalg.eigvalsh(rho.data).min() < -1e-10
        or not -1e-10 <= fidelity <= 1 + 1e-10
    ):
        msg = "Exact-channel density or fidelity validation failed."
        raise FloatingPointError(msg)
    return fidelity


def with_origins(circuit: ParameterizedCircuit) -> ParameterizedCircuit:
    """Assign immutable original-parameter IDs to the pilot's scalar gates.

    Returns:
        A gate-equivalent circuit with stable provenance.

    Raises:
        ValueError: If parameters are shared, fixed, or data-dependent.
    """
    indices = [gate.param_index for gate in circuit.gates]
    if (
        sorted(index for index in indices if index is not None) != list(range(circuit.num_params))
        or len(indices) != circuit.num_params
        or any(gate.data_map is not None or gate.angle_scale == 0 for gate in circuit.gates)
    ):
        msg = "Pilot pruning requires unique scalar parameters and no data maps."
        raise ValueError(msg)
    return ParameterizedCircuit(
        circuit.num_qubits,
        [replace(gate, logical_gate_id=gate.param_index) for gate in circuit.gates],
        circuit.num_params,
    )


def delete_gates(
    circuit: ParameterizedCircuit, theta: NDArray[np.float64], remove: Sequence[int]
) -> tuple[ParameterizedCircuit, NDArray[np.float64]]:
    """Remove complete gates/noise windows and compact parameters in old order.

    Returns:
        The retained circuit and matching compact parameter vector.

    Raises:
        ValueError: If the deletion indices are invalid or duplicated.
    """
    removed = set(remove)
    if len(removed) != len(remove) or any(
        type(index) is not int or not 0 <= index < len(circuit.gates) for index in remove
    ):
        msg = "Invalid gate deletion indices."
        raise ValueError(msg)
    retained = [gate for index, gate in enumerate(circuit.gates) if index not in removed]
    parameters = sorted({gate.param_index for gate in retained if gate.param_index is not None})
    remap = {old: new for new, old in enumerate(parameters)}
    gates = [
        replace(gate, param_index=None if gate.param_index is None else remap[gate.param_index]) for gate in retained
    ]
    return ParameterizedCircuit(circuit.num_qubits, gates, len(parameters)), theta[parameters].copy()


def pruning_scores(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    method: Literal["M", "D"],
) -> list[dict[str, float | int]]:
    """Score eligible gates by magnitude or actual noiseless deletion damage.

    Returns:
        Scores with stable origins, arities, and current gate indices.

    Raises:
        ValueError: If the method or original gate IDs are unsupported.
    """
    if method not in {"M", "D"}:
        msg = "Unknown pilot pruning method."
        raise ValueError(msg)
    baseline_loss = state_preparation_metrics(circuit, theta, target)[0] if method == "D" else 0.0
    scores = []
    for index, gate in enumerate(circuit.gates):
        origin = gate.logical_gate_id
        if type(origin) is not int:
            msg = "Pruning needs immutable integer original-parameter IDs."
            raise ValueError(msg)
        if origin < 3 * circuit.num_qubits:
            continue
        angle = circuit.angle(gate, theta, np.array([], dtype=np.float64))
        magnitude = abs((angle + math.pi) % (2 * math.pi) - math.pi)
        if method == "D":
            deleted, compact = delete_gates(circuit, theta, [index])
            score = state_preparation_metrics(deleted, compact, target)[0] - baseline_loss
        else:
            score = magnitude
        scores.append({
            "index": index,
            "origin": origin,
            "arity": len(gate.sites),
            "score": score,
            "magnitude": magnitude,
        })
    return scores


def select_deletions(scores: list[dict[str, float | int]], num_qubits: int) -> list[int]:
    """Choose the same fixed-arity quota for every pilot pruning round.

    Returns:
        Current gate indices selected simultaneously from the old circuit.

    Raises:
        ValueError: If there are too few eligible gates to meet a quota.
    """
    chosen = []
    for arity, quota in ((1, 2 * (num_qubits - 1)), (2, num_qubits - 1)):
        eligible = [row for row in scores if row["arity"] == arity]
        eligible.sort(key=lambda row: (round(row["score"] / 1e-12), row["magnitude"], row["origin"]))
        if len(eligible) < quota:
            msg = "Insufficient eligible gates for pruning quota."
            raise ValueError(msg)
        chosen.extend(int(row["index"]) for row in eligible[:quota])
    return sorted(chosen)
