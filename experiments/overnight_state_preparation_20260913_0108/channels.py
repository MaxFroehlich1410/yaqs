# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact fixed-benchmark channel evaluation, with explicit calibration identity.

The normative Pauli-transfer calibration equation is inverted separately for
all operator supports. Legacy one-jump execution remains explicitly separate.
Ballarin evaluation uses the repository's unchanged native materialization.
"""

from __future__ import annotations

from functools import cache
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

from benchmarks.state_preparation.ballarin import ballarin_pauli_probability, materialize_ballarin_circuit

# The isolated evaluator intentionally reuses the validated local array kernel.
from benchmarks.state_preparation.exact_density import _conjugate  # noqa: PLC2701
from benchmarks.state_preparation.noise import standard_noise_profile

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

PAULIS = (
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.diag([1, -1]).astype(complex),
)
NOISE_IDS = (
    "noiseless",
    "ballarin_coupled",
    "dephasing_1s_1q",
    "dephasing_1s_2q",
    "dephasing_1s_all",
    "dephasing_2s_2q",
    "dephasing_1s2s_all",
    "depolarizing_1s_1q",
    "depolarizing_1s_2q",
    "depolarizing_1s_all",
    "depolarizing_2s_2q",
    "depolarizing_1s2s_all",
)


@cache
def local_paulis(arity: int) -> tuple[NDArray[np.complex128], ...]:
    """Return tensor-product Paulis in lexicographic IXYZ order."""
    if arity == 1:
        return PAULIS
    if arity == 2:
        return tuple(np.kron(a, b) for a in PAULIS for b in PAULIS)
    msg = "Only one- and two-qubit physical operations are supported."
    raise ValueError(msg)


@cache
def window(noise_id: str, arity: int, convention: str = "calibrated_exact") -> dict:
    """Resolve a fixed Pauli window from the benchmark's calibration equation.

    Args:
        noise_id: One standard dephasing or depolarizing profile.
        arity: Gate support, either one or two.
        convention: Calibrated exact Lindblad or legacy finite one-jump map.

    Returns:
        Generators, coefficient, complete-channel error and Pauli probabilities.

    Raises:
        ValueError: If the convention, profile or support is invalid.
    """
    matrices = local_paulis(arity)
    profile = standard_noise_profile(noise_id)
    if convention not in {"calibrated_exact", "legacy_one_jump"}:
        msg = "Unknown channel convention."
        raise ValueError(msg)
    codes = tuple(product(range(4), repeat=arity))
    axes = (3,) if profile.family == "dephasing" else (1, 2, 3)
    applies = (
        profile.placement == "all_gates"
        or (profile.placement == "single_qubit_gates" and arity == 1)
        or (profile.placement == "multi_qubit_gates" and arity == 2)
    )
    active = []
    if applies:
        for i, code in enumerate(codes):
            nonzero = tuple(x for x in code if x != 0)
            single = profile.support in {"single_site", "single_site_two_site"} and len(nonzero) == 1
            pair = profile.support in {"two_site", "single_site_two_site"} and arity == 2 and len(nonzero) == 2
            if (single or pair) and all(x in axes for x in nonzero):
                active.append(i)
    dimension = 2**arity
    probability = np.zeros(4**arity)
    probability[0] = 1
    coefficient = 0.0
    target_error = 0.00064 if arity == 1 else 0.0051
    if active:
        # Character chi(P,Q)=+1 for commuting, -1 for anticommuting Paulis.
        character = np.array([[float(np.trace(p @ q @ p @ q).real / dimension) for q in matrices] for p in matrices])
        counts = np.sum(character[active] < 0, axis=0)
        if convention == "calibrated_exact":

            def error(strength: float) -> float:
                return float((dimension**2 - np.exp(-2 * strength * counts).sum()) / (dimension * (dimension + 1)))

            coefficient = brentq(lambda a: error(a) - target_error, 0.0, 1.0, xtol=1e-15)
            eigenvalues = np.exp(-2 * coefficient * counts)
            probability = character @ eigenvalues / dimension**2
            probability[np.abs(probability) < 1e-15] = 0
        else:
            coefficient = target_error  # Frozen legacy per-operator coefficients.
            probability[0] = np.exp(-len(active) * coefficient)
            probability[active] = -np.expm1(-len(active) * coefficient) / len(active)
    actual_error = dimension / (dimension + 1) * (1 - probability[0])
    if np.min(probability) < -1e-13 or abs(probability.sum() - 1) > 1e-12:
        msg = "Invalid Pauli-channel probability decomposition."
        raise FloatingPointError(msg)
    return {
        "noise_id": noise_id,
        "arity": arity,
        "convention": convention,
        "generators": ["".join("IXYZ"[x] for x in codes[i]) for i in active],
        "coefficient": coefficient,
        "target_error": target_error if active else 0.0,
        "actual_error": float(actual_error),
        "probabilities": probability.tolist(),
    }


@cache
def superoperator(probabilities: tuple[float, ...]) -> NDArray[np.complex128]:
    """Return the local row-vectorized channel; never a full-system operator."""
    arity = 1 if len(probabilities) == 4 else 2
    return sum(
        (w * np.kron(p, p.conj()) for w, p in zip(probabilities, local_paulis(arity), strict=True)),
        start=np.zeros((4**arity, 4**arity), dtype=complex),
    )


def apply_channel(
    rho: NDArray[np.complex128],
    sites: tuple[int, ...],
    probabilities: tuple[float, ...],
    num_qubits: int,
) -> NDArray[np.complex128]:
    """Apply a local Pauli channel to arbitrary density or observable matrices."""
    k = len(sites)
    axes = tuple(num_qubits - 1 - q for q in sites) + tuple(2 * num_qubits - 1 - q for q in sites)
    front = tuple(range(2 * k))
    moved = np.moveaxis(rho.reshape((2,) * (2 * num_qubits)), axes, front)
    result = (superoperator(probabilities) @ moved.reshape(4**k, -1)).reshape(moved.shape)
    return np.moveaxis(result, front, axes).reshape(rho.shape)


def evaluate(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    noise_id: str,
    *,
    convention: str = "calibrated_exact",
) -> float:
    """Evaluate exact noisy preparation fidelity of the same physical circuit.

    Ballarin rotations are lowered and pruned by the frozen implementation.
    All other conditions retain the already supported one-/two-qubit gates.

    Raises:
        ValueError: If circuit size, state, parameters or channel is invalid.
        FloatingPointError: If propagation violates physicality checks.
    """
    n = circuit.num_qubits
    if (
        type(n) is not int
        or not 1 <= n <= 6
        or not np.isrealobj(theta)
        or convention not in {"calibrated_exact", "legacy_one_jump"}
        or noise_id not in NOISE_IDS
        or theta.shape != (circuit.num_params,)
        or not np.all(np.isfinite(theta))
        or target.shape != (2**n,)
        or abs(np.vdot(target, target) - 1) > 1e-10
        or not np.all(np.isfinite(target))
    ):
        msg = "Expected a supported q1-q6 circuit, finite theta and normalized target."
        raise ValueError(msg)
    if noise_id == "ballarin_coupled":
        circuit = materialize_ballarin_circuit(circuit, theta).circuit
        theta = np.empty(0)
    rho = np.zeros((2**n, 2**n), dtype=complex)
    rho[0, 0] = 1
    for gate in circuit.gates:
        operator, sites, angle = circuit.gate_matrix_and_angle(gate, theta)
        rho = _conjugate(rho, operator, sites, n)
        if not gate.noise_enabled or noise_id == "noiseless":
            continue
        if noise_id == "ballarin_coupled":
            assert angle is not None
            r = ballarin_pauli_probability(float(angle))
            probabilities = (1 - 3 * r, r, r, r)
            for site in sites:
                rho = apply_channel(rho, (site,), probabilities, n)
        else:
            probabilities = tuple(window(noise_id, len(sites), convention)["probabilities"])
            rho = apply_channel(rho, sites, probabilities, n)
    fidelity = float(np.vdot(target, rho @ target).real)
    if (
        not np.isfinite(fidelity)
        or not -1e-10 <= fidelity <= 1 + 1e-10
        or abs(np.trace(rho) - 1) > 1e-10
        or np.max(abs(rho - rho.conj().T)) > 1e-10
        or np.linalg.eigvalsh(rho).min() < -1e-10
    ):
        msg = "Exact evaluation failed density physicality checks."
        raise FloatingPointError(msg)
    return fidelity
