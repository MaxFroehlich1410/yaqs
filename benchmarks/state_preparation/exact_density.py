# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Small-system exact adjoint for parameter-independent local depolarization.

This diagnostic deliberately does not extend the production trajectory backend.
Matrices use little-endian basis indices; local gate matrices use the circuit's
ascending-site tensor-factor order. No full-system unitary or Liouvillian is built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

    from .calibration import CalibratedLocalDepolarizingProvider


def _left_apply(
    matrix: NDArray[np.complex128], operator: NDArray[np.complex128], sites: tuple[int, ...], num_qubits: int
) -> NDArray[np.complex128]:
    """Apply a local operator to matrix row indices.

    Returns:
        The matrix product with the locally embedded operator on the left.
    """
    axes = tuple(num_qubits - 1 - site for site in sites)
    front = tuple(range(len(sites)))
    tensor = np.moveaxis(matrix.reshape((2,) * num_qubits + (2**num_qubits,)), axes, front)
    transformed = (operator @ tensor.reshape(2 ** len(sites), -1)).reshape(tensor.shape)
    return np.moveaxis(transformed, front, axes).reshape(matrix.shape)


def _conjugate(
    matrix: NDArray[np.complex128], operator: NDArray[np.complex128], sites: tuple[int, ...], num_qubits: int
) -> NDArray[np.complex128]:
    """Conjugate a matrix by a locally embedded operator.

    Returns:
        ``U matrix U.adjoint`` without constructing a full-system U.
    """
    left = _left_apply(matrix, operator, sites, num_qubits)
    return _left_apply(left.conj().T, operator, sites, num_qubits).conj().T


def _noise(
    matrix: NDArray[np.complex128], sites: tuple[int, ...], strength: float, num_qubits: int
) -> NDArray[np.complex128]:
    """Apply the exact, self-adjoint local Pauli-Lindblad channel.

    Each site map is lambda*A + (1-lambda)*I_site/2 tensor Tr_site(A),
    lambda=exp(-4*a). It acts on density matrices and costates alike.

    Returns:
        The channel output, or the unchanged matrix at zero strength.
    """
    if strength == 0:
        return matrix
    decay = np.exp(-4 * strength)
    mixed_weight = -np.expm1(-4 * strength) / 2
    for site in sites:
        axes = (num_qubits - 1 - site, 2 * num_qubits - 1 - site)
        tensor = np.moveaxis(matrix.reshape((2,) * (2 * num_qubits)), axes, (0, 1))
        reduced = tensor[0, 0] + tensor[1, 1]
        output = decay * tensor
        output[0, 0] += mixed_weight * reduced
        output[1, 1] += mixed_weight * reduced
        matrix = np.moveaxis(output, (0, 1), axes).reshape(matrix.shape)
    return matrix


def density_loss_gradient(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    calibration: CalibratedLocalDepolarizingProvider,
) -> tuple[NDArray[np.float64], float, float]:
    """Differentiate exact target infidelity through unitary/channel layers.

    Args:
        circuit: One-to-six-qubit circuit, with no data-dependent gates.
        theta: Finite real parameter vector, including shared parameters.
        target: Normalized finite complex target vector in little-endian order.
        calibration: Constant local Pauli-Lindblad rates. Zero errors give the
            matched noiseless objective; angle/state-dependent channels are not
            supported by this adjoint.

    Returns:
        Exact loss gradient, infidelity, and pure-target state fidelity.

    Raises:
        ValueError: If the system, parameters, or target are invalid.
        FloatingPointError: If propagation violates numerical physicality checks.
    """
    n = circuit.num_qubits
    if (
        type(n) is not int
        or not 1 <= n <= 6
        or theta.shape != (circuit.num_params,)
        or not np.isrealobj(theta)
        or not np.all(np.isfinite(theta))
        or target.shape != (2**n,)
        or not np.all(np.isfinite(target))
        or abs(np.vdot(target, target) - 1) > 1e-10
        or any(gate.data_map is not None or not np.isfinite(gate.angle_scale) for gate in circuit.gates)
    ):
        msg = "Exact density adjoint requires 1-6 qubits, finite parameters, and a normalized target."
        raise ValueError(msg)
    rho = np.zeros((2**n, 2**n), dtype=np.complex128)
    rho[0, 0] = 1
    tape = []
    for gate in circuit.gates:
        unitary, sites = circuit.gate_matrix(gate, theta)
        before_noise = _conjugate(rho, unitary, sites, n)
        strength = calibration.one_qubit_strength if len(sites) == 1 else calibration.two_qubit_strength
        if not gate.noise_enabled:
            strength = 0.0
        tape.append((unitary, sites, before_noise, strength))
        rho = _noise(before_noise, sites, strength, n)
    fidelity = float(np.vdot(target, rho @ target).real)
    if (
        not np.all(np.isfinite(rho))
        or abs(np.trace(rho) - 1) > 1e-10
        or np.max(np.abs(rho - rho.conj().T)) > 1e-10
        or np.linalg.eigvalsh(rho).min() < -1e-10
        or not -1e-10 <= fidelity <= 1 + 1e-10
    ):
        msg = "Exact density propagation failed finite/trace/Hermiticity/positivity checks."
        raise FloatingPointError(msg)
    # Propagate the fidelity observable; negate its derivative for infidelity.
    costate = np.outer(target, target.conj())
    gradient = np.zeros(circuit.num_params, dtype=np.float64)
    for gate, (unitary, sites, before_noise, strength) in reversed(list(zip(circuit.gates, tape, strict=True))):
        costate = _noise(costate, sites, strength, n)
        if gate.param_index is not None:
            derivative, derivative_sites = circuit.derivative_operator(gate)
            left = _left_apply(before_noise, derivative, derivative_sites, n)
            tangent = gate.angle_scale * (left + left.conj().T)
            gradient[gate.param_index] -= float(np.vdot(costate, tangent).real)
        costate = _conjugate(costate, unitary.conj().T, sites, n)
    if not np.all(np.isfinite(gradient)):
        msg = "Exact density gradient contains nonfinite entries."
        raise FloatingPointError(msg)
    return gradient, 1 - fidelity, fidelity


def density_deletion_fidelities(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    calibration: CalibratedLocalDepolarizingProvider,
) -> tuple[NDArray[np.float64], float]:
    """Score every complete single-gate/channel deletion with shared adjoint tapes.

    For gate k, the forward state rho_before excludes that gate, and B_after
    includes only its unchanged suffix. The exact deletion fidelity is
    Tr(B_after rho_before). Only after scoring is B_after propagated through
    the current channel adjoint and then the unitary adjoint. Removing a gate
    therefore removes its noise window too, not just its ideal rotation.

    Args:
        circuit: One-to-six-qubit scalar-gate circuit, including fixed or shared
            parameters. Gate order and all other noise windows stay unchanged.
        theta: Finite real parameters in the original circuit's parameter order.
        target: Normalized complex target in little-endian basis ordering.
        calibration: Constant local Pauli-Lindblad strengths; disabled windows
            are respected. Rescheduling, routing and state-dependent noise are
            not supported by this model.

    Returns:
        Single-deletion fidelities in gate order and the undeleted fidelity.
        One forward and one backward circuit pass use O(L * 4**n) stored density
        entries, without a full-system unitary or Liouvillian. This is not a
        trajectory or tensor-network memory-scaling claim.

    Raises:
        ValueError: If inputs are outside the small-system scalar-gate domain.
        FloatingPointError: If density or fidelity physicality checks fail.
    """
    n = circuit.num_qubits
    if (
        type(n) is not int
        or not 1 <= n <= 6
        or theta.shape != (circuit.num_params,)
        or not np.isrealobj(theta)
        or not np.all(np.isfinite(theta))
        or target.shape != (2**n,)
        or not np.all(np.isfinite(target))
        or abs(np.vdot(target, target) - 1) > 1e-10
        or any(gate.data_map is not None or not np.isfinite(gate.angle_scale) for gate in circuit.gates)
    ):
        msg = "Exact deletion adjoint requires 1-6 qubits, finite parameters, and a normalized target."
        raise ValueError(msg)
    rho = np.zeros((2**n, 2**n), dtype=np.complex128)
    rho[0, 0] = 1
    tape = []
    for gate in circuit.gates:
        unitary, sites = circuit.gate_matrix(gate, theta)
        strength = calibration.one_qubit_strength if len(sites) == 1 else calibration.two_qubit_strength
        if not gate.noise_enabled:
            strength = 0.0
        tape.append((rho, unitary, sites, strength))
        rho = _noise(_conjugate(rho, unitary, sites, n), sites, strength, n)
    fidelity = float(np.vdot(target, rho @ target).real)
    if (
        not np.all(np.isfinite(rho))
        or abs(np.trace(rho) - 1) > 1e-10
        or np.max(np.abs(rho - rho.conj().T)) > 1e-10
        or np.linalg.eigvalsh(rho).min() < -1e-10
        or not -1e-10 <= fidelity <= 1 + 1e-10
    ):
        msg = "Exact deletion forward density failed physicality checks."
        raise FloatingPointError(msg)
    costate = np.outer(target, target.conj())
    scores = np.empty(len(tape), dtype=np.float64)
    for index in range(len(tape) - 1, -1, -1):
        before, unitary, sites, strength = tape[index]
        overlap = np.vdot(costate, before)
        if not np.isfinite(overlap) or abs(overlap.imag) > 1e-10 or not -1e-10 <= overlap.real <= 1 + 1e-10:
            msg = "Exact deletion score failed finite/reality/fidelity checks."
            raise FloatingPointError(msg)
        scores[index] = overlap.real
        costate = _conjugate(_noise(costate, sites, strength, n), unitary.conj().T, sites, n)
    return scores, fidelity
