# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact density adjoints for the fixed benchmark's eleven noisy conditions.

The mean objective is a known ensemble-control strategy, adapted to the frozen
benchmark channel set. Ballarin's channel derivative is included; pruning and
angle wrapping make its objective only piecewise differentiable. This module
does not alter the normative evaluator or materializer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from benchmarks.state_preparation.ballarin import (
    RZZ_PRUNING_THRESHOLD,
    ballarin_epsilon,
    ballarin_pauli_probability,
    materialize_ballarin_circuit,
)

# Isolated experiments intentionally reuse these validated local array kernels.
# The pinned repository lint uses PLC2701 for these two private imports.
from benchmarks.state_preparation.exact_density import _conjugate, _left_apply  # noqa: PLC2701
from experiments.overnight_state_preparation_20260913_0108 import channels

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

NOISY_IDS = channels.NOISE_IDS[1:]
BOUNDARY_ATOL = 2e-12


@dataclass(frozen=True)
class _Layer:
    """Retain one exact forward layer and its parameter-dependent channel tangent."""

    gate: ParameterizedGate
    unitary: NDArray[np.complex128]
    sites: tuple[int, ...]
    before_noise: NDArray[np.complex128]
    probabilities: tuple[float, ...] | None
    derivative_probabilities: tuple[float, ...] | None


def _validate(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    noise_id: str,
) -> None:
    """Validate the fixed small-system scalar-angle adjoint domain.

    Raises:
        ValueError: If inputs are outside the supported numerical domain.
    """
    n = circuit.num_qubits
    if (
        type(n) is not int
        or not 1 <= n <= 6
        or noise_id not in channels.NOISE_IDS
        or theta.shape != (circuit.num_params,)
        or not np.isrealobj(theta)
        or not np.all(np.isfinite(theta))
        or target.shape != (2**n,)
        or not np.all(np.isfinite(target))
        or abs(np.vdot(target, target) - 1) > 1e-10
        or any(
            gate.data_map is not None
            or not np.isrealobj(gate.angle_scale)
            or not np.isrealobj(gate.angle_offset)
            or not np.isrealobj(gate.fixed_params)
            or not np.isfinite(gate.angle_scale)
            or not np.isfinite(gate.angle_offset)
            or not np.all(np.isfinite(gate.fixed_params))
            for gate in circuit.gates
        )
    ):
        msg = "Exact ensemble adjoint requires a supported q1-q6 circuit, finite theta, and normalized target."
        raise ValueError(msg)


def _ballarin_probabilities(
    angle: float, gate: ParameterizedGate
) -> tuple[tuple[float, ...], tuple[float, ...] | None]:
    """Return one local factor and its full parameter derivative.

    Returns:
        Four local probabilities and, for a trainable gate, their tangents.
        Both sites use this factor. Keeping factors separate avoids caching
        angle-dependent 16-by-16 superoperators for every optimizer evaluation.
    """
    r = ballarin_pauli_probability(angle)
    local = np.array([1 - 3 * r, r, r, r])
    probabilities = tuple(float(value) for value in local)
    if gate.param_index is None or gate.angle_scale == 0:
        return probabilities, None
    # Differentiate r=(1-sqrt(1-5*epsilon/4))/3; the product rule is applied below.
    r_prime = 5 * 1.43e-3 * np.sign(angle) / (24 * math.sqrt(1 - 1.25 * ballarin_epsilon(angle)))
    local_derivative = gate.angle_scale * r_prime * np.array([-3, 1, 1, 1])
    return probabilities, tuple(float(value) for value in local_derivative)


def _apply_noise(
    matrix: NDArray[np.complex128], sites: tuple[int, ...], probabilities: tuple[float, ...], n: int
) -> NDArray[np.complex128]:
    """Apply a complete fixed window or the two local Ballarin factors.

    Returns:
        The propagated density or observable matrix.
    """
    if len(sites) == 2 and len(probabilities) == 4:
        for site in sites:
            matrix = channels.apply_channel(matrix, (site,), probabilities, n)
        return matrix
    return channels.apply_channel(matrix, sites, probabilities, n)


def _ballarin_tangent(layer: _Layer, n: int) -> NDArray[np.complex128]:
    """Differentiate both local factors using the product rule.

    Returns:
        E_second(dE_first(rho)) + dE_second(E_first(rho)), including angle scale.
    """
    probabilities = layer.probabilities
    derivative = layer.derivative_probabilities
    assert probabilities is not None
    assert derivative is not None
    first, second = layer.sites
    first_term = channels.apply_channel(layer.before_noise, (first,), derivative, n)
    first_term = channels.apply_channel(first_term, (second,), probabilities, n)
    second_term = channels.apply_channel(layer.before_noise, (first,), probabilities, n)
    second_term = channels.apply_channel(second_term, (second,), derivative, n)
    return first_term + second_term


def _check_ballarin_boundary(angle: float, gate: ParameterizedGate) -> None:
    """Reject trainable angles near pruning discontinuities or wrap cusps.

    The materializer includes a 1e-12 comparison tolerance around the pruning
    threshold. A conservative 2e-12 exclusion band covers that boundary. Fixed
    and zero-scale gates may lie there because their theta derivative is zero.

    Raises:
        ValueError: If an angle with nonzero parameter scale is on a boundary.
    """
    if gate.param_index is None or gate.angle_scale == 0:
        return
    if abs(abs(angle) - RZZ_PRUNING_THRESHOLD) <= BOUNDARY_ATOL or abs(abs(angle) - math.pi) <= BOUNDARY_ATOL:
        msg = "Ballarin gradient is unsupported at a pruning boundary or canonical-angle cusp."
        raise ValueError(msg)


def loss_gradient(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    noise_id: str,
) -> tuple[NDArray[np.float64], float, float]:
    """Differentiate exact infidelity under one frozen benchmark channel.

    Args:
        circuit: Logical one-to-six-qubit circuit with supported scalar gates.
        theta: Finite parameters; sharing, offsets and signed scales are supported.
        target: Normalized target in the repository's little-endian convention.
        noise_id: One of the twelve benchmark conditions, including noiseless.

    Returns:
        Loss gradient, infidelity, and exact fidelity. Ballarin two-qubit gates
        must be RXX, RYY, or RZZ, matching the unchanged native materializer.
        Pruned gates in the interior of its zero-angle region have derivative
        zero. Each profile requires one forward/backward density pass, stores
        O(L*4**n) entries, and performs no stochastic trajectory sampling.

    Raises:
        ValueError: If inputs, lowering, or a Ballarin derivative are unsupported.
        FloatingPointError: If density physicality or gradient finiteness fails.

    Notes:
        Ballarin's local depolarization commutes through its noiseless local
        basis changes. Its native channel therefore equals the product channel
        applied after the corresponding logical Pauli rotation. The materializer
        is authoritative for canonical angles and deletion. It forces retained
        two-qubit noise on even when the logical noise flag is false; one-qubit
        gates are noiseless. Constant profiles respect the original noise flags.
        Probability derivatives include both local channels and angle_scale.
    """
    _validate(circuit, theta, target, noise_id)
    n = circuit.num_qubits
    mappings = materialize_ballarin_circuit(circuit, theta).mapping if noise_id == "ballarin_coupled" else None
    rho = np.zeros((2**n, 2**n), dtype=complex)
    rho[0, 0] = 1
    tape: list[_Layer] = []
    for index, gate in enumerate(circuit.gates):
        unitary, sites = circuit.gate_matrix(gate, theta)
        probabilities = None
        derivative_probabilities = None
        if mappings is not None:
            if len(sites) == 2:
                angle = mappings[index].canonical_angle
                assert angle is not None
                _check_ballarin_boundary(angle, gate)
                if mappings[index].pruned:
                    continue
                bound = replace(gate, param_index=None, angle_offset=angle)
                unitary, sites = circuit.gate_matrix(bound, theta)
                probabilities, derivative_probabilities = _ballarin_probabilities(angle, gate)
        elif noise_id != "noiseless" and gate.noise_enabled:
            probabilities = tuple(channels.window(noise_id, len(sites))["probabilities"])
        before_noise = _conjugate(rho, unitary, sites, n)
        tape.append(_Layer(gate, unitary, sites, before_noise, probabilities, derivative_probabilities))
        rho = before_noise if probabilities is None else _apply_noise(before_noise, sites, probabilities, n)
    fidelity = float(np.vdot(target, rho @ target).real)
    if (
        not np.all(np.isfinite(rho))
        or not -1e-10 <= fidelity <= 1 + 1e-10
        or abs(np.trace(rho) - 1) > 1e-10
        or np.max(abs(rho - rho.conj().T)) > 1e-10
        or np.linalg.eigvalsh(rho).min() < -1e-10
    ):
        msg = "Exact ensemble propagation failed density physicality checks."
        raise FloatingPointError(msg)
    costate = np.outer(target, target.conj())
    gradient = np.zeros(circuit.num_params)
    for layer in reversed(tape):
        gate = layer.gate
        if gate.param_index is not None and layer.derivative_probabilities is not None:
            tangent = _ballarin_tangent(layer, n)
            gradient[gate.param_index] -= float(np.vdot(costate, tangent).real)
        if layer.probabilities is not None:
            costate = _apply_noise(costate, layer.sites, layer.probabilities, n)
        if gate.param_index is not None and gate.angle_scale != 0:
            derivative, sites = circuit.derivative_operator(gate)
            left = _left_apply(layer.before_noise, derivative, sites, n)
            tangent = gate.angle_scale * (left + left.conj().T)
            gradient[gate.param_index] -= float(np.vdot(costate, tangent).real)
        costate = _conjugate(costate, layer.unitary.conj().T, layer.sites, n)
    if not np.all(np.isfinite(gradient)):
        msg = "Exact ensemble gradient contains nonfinite entries."
        raise FloatingPointError(msg)
    return gradient, 1 - fidelity, fidelity


def mean_loss_gradient(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    noise_ids: Sequence[str] = NOISY_IDS,
) -> tuple[NDArray[np.float64], float, dict[str, float]]:
    """Return the equally weighted mean infidelity gradient across unique profiles.

    Args:
        circuit: Same physical logical circuit for every profile.
        theta: Shared parameter vector, not modified in place.
        target: Common normalized pure target.
        noise_ids: Unique profiles; defaults to all eleven noisy conditions.

    Returns:
        Mean loss gradient, mean infidelity, and separate exact fidelities.
        Work is the sum of one exact density gradient call per profile. The
        default excludes noiseless from the mean, as required by the benchmark.

    Raises:
        ValueError: If the profile list is empty, duplicate, or unsupported.
    """
    if (
        not noise_ids
        or len(set(noise_ids)) != len(noise_ids)
        or any(key not in channels.NOISE_IDS for key in noise_ids)
    ):
        msg = "The ensemble requires nonempty unique supported channel identities."
        raise ValueError(msg)
    gradient = np.zeros(circuit.num_params)
    loss = 0.0
    fidelities: dict[str, float] = {}
    for noise_id in noise_ids:
        current_gradient, current_loss, fidelity = loss_gradient(circuit, theta, target, noise_id)
        gradient += current_gradient
        loss += current_loss
        fidelities[noise_id] = fidelity
    return gradient / len(noise_ids), loss / len(noise_ids), fidelities
