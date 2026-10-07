# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact noisy Rotosolve using local density contractions and reused suffixes.

This adapts Nakanishi, Fujii, and Todo, arXiv:1903.12166, and Ostaszewski,
Grant, and Benedetti, arXiv:1905.09692. The optimization rule is known;
the density-adjoint implementation is an experiment-specific adaptation.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider

# This isolated experiment deliberately shares the already validated local kernels.
from benchmarks.state_preparation.exact_density import _conjugate, _noise  # noqa: PLC2701

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

_SUPPORTED = frozenset({"rx", "ry", "rz", "p", "rxx", "ryy", "rzz", "cp"})


def _validate(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    calibration: CalibratedLocalDepolarizingProvider,
) -> None:
    """Validate the domain in which one-coordinate sinusoidal fitting is exact.

    Raises:
        ValueError: If inputs or the channel violate the supported domain.
    """
    n = circuit.num_qubits
    indices = [gate.param_index for gate in circuit.gates if gate.param_index is not None]
    if (
        type(calibration) is not CalibratedLocalDepolarizingProvider
        or type(n) is not int
        or not 1 <= n <= 6
        or theta.shape != (circuit.num_params,)
        or not np.isrealobj(theta)
        or not np.all(np.isfinite(theta))
        or target.shape != (2**n,)
        or not np.all(np.isfinite(target))
        or abs(np.vdot(target, target) - 1) > 1e-10
        or len(set(indices)) != len(indices)
        or any(type(index) is not int or not 0 <= index < circuit.num_params for index in indices)
        or any(
            gate.data_map is not None
            or not np.isfinite(gate.angle_offset)
            or not np.isfinite(gate.angle_scale)
            or (gate.param_index is not None and (gate.angle_scale == 0 or gate.name not in _SUPPORTED))
            for gate in circuit.gates
        )
    ):
        msg = "Rotosolve requires 1-6 qubits, finite inputs, unique nonzero-scale parameters, and constant local noise."
        raise ValueError(msg)


def _overlap(observable: NDArray[np.complex128], density: NDArray[np.complex128]) -> float:
    """Return a physically valid real expectation of a fidelity observable.

    Raises:
        FloatingPointError: If the contraction is nonfinite or nonphysical.
    """
    value = np.vdot(observable, density)
    if not np.isfinite(value) or abs(value.imag) > 1e-10 or not -1e-10 <= value.real <= 1 + 1e-10:
        msg = "Rotosolve fidelity contraction is nonfinite or nonphysical."
        raise FloatingPointError(msg)
    return float(value.real)


def sweep(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    calibration: CalibratedLocalDepolarizingProvider,
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Perform one forward sweep of exact coordinate fidelity maximizations.

    For each supported single-occurrence gate, fixed surrounding channels make
    fidelity ``a + b*cos(delta) + c*sin(delta)`` in its physical angle shift.
    Three local contractions recover these coefficients; ``atan2(c, b)`` gives
    a global coordinate maximizer. The prefix incorporates all earlier updates
    and the suffix contains the unchanged later gates. A retained zero-angle
    gate keeps its noise window. Inputs and circuit structure are unchanged.

    Args:
        circuit: Circuit of one- and two-qubit gates with unique parameters.
        theta: Finite real parameter vector. Nonunit scales and offsets work.
        target: Normalized pure target in little-endian order.
        calibration: Exact local Pauli-Lindblad calibration, including zero
            noise. Ballarin, data-dependent gates, and shared parameters fail.

    Returns:
        Updated parameter vector and JSON-compatible diagnostics. Work counts
        describe local matrix operations, not hardware shots. Stored tape bytes
        count retained arrays and are not a measured process peak memory value.

    Raises:
        ValueError: If inputs violate the supported domain.
        FloatingPointError: If propagation or finite parameter updates fail.
    """
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    _validate(circuit, theta, target, calibration)
    n = circuit.num_qubits
    parameters = np.array(theta, dtype=np.float64, copy=True)
    costate = np.outer(target, target.conj())
    tape = []
    for gate in reversed(circuit.gates):
        unitary, sites = circuit.gate_matrix(gate, parameters)
        strength = calibration.one_qubit_strength if len(sites) == 1 else calibration.two_qubit_strength
        if not gate.noise_enabled:
            strength = 0.0
        before_noise = _noise(costate, sites, strength, n)
        tape.append((before_noise, unitary, sites, strength))
        costate = _conjugate(before_noise, unitary.conj().T, sites, n)
    tape.reverse()
    rho = np.zeros((2**n, 2**n), dtype=np.complex128)
    rho[0, 0] = 1
    initial = _overlap(costate, rho)
    coordinate_scores = []
    accepted = 0
    for gate, (observable, unitary, sites, strength) in zip(circuit.gates, tape, strict=True):
        before_noise = _conjugate(rho, unitary, sites, n)
        if gate.param_index is not None:
            old_score = _overlap(observable, before_noise)
            angle = circuit.angle(gate, parameters, None)
            shifted = []
            for delta in (np.pi / 2, -np.pi / 2):
                probe = replace(gate, param_index=None, angle_offset=angle + delta)
                matrix, _ = circuit.gate_matrix(probe, parameters)
                shifted.append(_overlap(observable, _conjugate(rho, matrix, sites, n)))
            constant = (shifted[0] + shifted[1]) / 2
            cosine = old_score - constant
            sine = (shifted[0] - shifted[1]) / 2
            delta = float(np.arctan2(sine, cosine)) if np.hypot(cosine, sine) > 1e-14 else 0.0
            candidate = parameters[gate.param_index] + delta / gate.angle_scale
            if not np.isfinite(candidate):
                msg = "Rotosolve update exceeds the finite parameter range."
                raise FloatingPointError(msg)
            probe = parameters.copy()
            probe[gate.param_index] = candidate
            matrix, _ = circuit.gate_matrix(gate, probe)
            candidate_density = _conjugate(rho, matrix, sites, n)
            new_score = _overlap(observable, candidate_density)
            # Keep the old point if finite precision causes a downward change.
            if new_score >= old_score:
                parameters = probe
                before_noise = candidate_density
                accepted += int(delta != 0)
            else:
                new_score = old_score
            coordinate_scores.append([gate.param_index, old_score, new_score])
        rho = _noise(before_noise, sites, strength, n)
    final = _overlap(np.outer(target, target.conj()), rho)
    if (
        abs(np.trace(rho) - 1) > 1e-10
        or np.max(np.abs(rho - rho.conj().T)) > 1e-10
        or np.linalg.eigvalsh(rho).min() < -1e-10
        or final < initial - 1e-10
    ):
        msg = "Rotosolve propagation failed physicality or monotonicity checks."
        raise FloatingPointError(msg)
    count = len(coordinate_scores)
    gates = len(circuit.gates)
    diagnostics: dict[str, object] = {
        "initial_fidelity": initial,
        "final_fidelity": final,
        "coordinate_updates": count,
        "nonzero_accepted_updates": accepted,
        "coordinate_scores": coordinate_scores,
        "density_conjugations": 2 * gates + 3 * count,
        "noise_kernel_calls": 2 * gates,
        "local_fidelity_contractions": 4 * count + 2,
        "circuit_direction_passes": 2,
        "training_trajectories": 0,
        "stored_tape_bytes": sum(observable.nbytes + unitary.nbytes for observable, unitary, _, _ in tape),
        "gate_count": gates,
        "num_parameters": circuit.num_params,
        "training_cpu_seconds": time.process_time() - cpu_start,
        "training_wall_seconds": time.perf_counter() - wall_start,
        "channel": "exact_parameter_independent_local_pauli_lindblad",
    }
    return parameters, diagnostics
