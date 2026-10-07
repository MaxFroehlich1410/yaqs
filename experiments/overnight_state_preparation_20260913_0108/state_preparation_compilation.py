# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Known-input eigenstate simplification, valid only for preparation from zero.

This is a small dense implementation of known constant/eigenstate propagation,
not a unitary-equivalence compiler or a new optimizer. A noisy prefix need not
be the ideal prefix used to justify removal; noisy fidelity is never promised.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from experiments import state_preparation_pilot as pilot
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

# This isolated diagnostic reuses only the validated local state-application kernel.
# PLC2701 is the private-import rule in the repository's pinned Ruff version.
from mqt.yaqs.optimization.statevector import _apply  # noqa: PLC2701

if TYPE_CHECKING:
    from numpy.typing import NDArray

VERSION = "overnight.zero_state_eigenstate_compilation.v1"
EIGENSTATE_ATOL = 1e-12
PRIMITIVES = frozenset({"rx", "ry", "rz", "rxx", "ryy", "rzz"})


def validate(circuit: ParameterizedCircuit, theta: NDArray) -> NDArray[np.float64]:
    """Return finite real parameters for supported unique-scalar q1-q6 gates.

    Raises:
        ValueError: If a circuit, gate or parameter vector is unsupported.
    """
    values = np.asarray(theta)
    if (
        type(circuit.num_qubits) is not int
        or not 1 <= circuit.num_qubits <= 6
        or type(circuit.num_params) is not int
        or circuit.num_params != len(circuit.gates)
        or values.shape != (circuit.num_params,)
        or not np.isrealobj(values)
        or not np.all(np.isfinite(values))
    ):
        msg = "Expected a finite real unique-scalar q1-q6 preparation circuit."
        raise ValueError(msg)
    indices = []
    for gate in circuit.gates:
        arity = 1 if gate.name in {"rx", "ry", "rz"} else 2
        if (
            gate.name not in PRIMITIVES
            or gate.data_map is not None
            or gate.fixed_params
            or type(gate.param_index) is not int
            or not 0 <= gate.param_index < circuit.num_params
            or len(gate.sites) != arity
            or len(set(gate.sites)) != arity
            or any(type(site) is not int or not 0 <= site < circuit.num_qubits for site in gate.sites)
            or type(gate.noise_enabled) is not bool
            or not np.isrealobj(gate.angle_scale)
            or not np.isrealobj(gate.angle_offset)
            or not np.isfinite(gate.angle_scale)
            or not np.isfinite(gate.angle_offset)
        ):
            msg = (
                "Only independent scalar Pauli rotations with finite affine angles and complete metadata are supported."
            )
            raise ValueError(msg)
        indices.append(gate.param_index)
    if sorted(indices) != list(range(circuit.num_params)):
        msg = "Shared or unused parameters are unsupported by the state-preparation compiler."
        raise ValueError(msg)
    return values.astype(float, copy=True)


def restore(record: dict[str, Any]) -> ParameterizedCircuit:
    """Restore every scalar gate field, including original logical and native IDs.

    Raises:
        ValueError: If a serialized source contains a data map or unsupported metadata.
    """
    if any(gate.get("data_map") is not None for gate in record["gates"]):
        msg = "Serialized state-preparation gates cannot contain data maps."
        raise ValueError(msg)
    gates = []
    for gate in record["gates"]:
        options: dict[str, Any] = {
            **gate,
            "sites": tuple(gate["sites"]),
            "fixed_params": tuple(gate.get("fixed_params", ())),
        }
        gates.append(ParameterizedGate(**options))
    return ParameterizedCircuit(record["num_qubits"], gates, record["num_params"])


def global_phase_residual(before: NDArray, after: NDArray) -> tuple[complex, float]:
    """Return the best global phase and its stable direct state-vector residual.

    The returned phase multiplies ``before`` to approximate ``after``. A direct
    norm avoids subtracting nearly equal overlaps when the tolerance is 1e-12.

    Raises:
        ValueError: If vectors are nonfinite, mismatched or not normalized.
    """
    if (
        before.ndim != 1
        or before.shape != after.shape
        or not before.size
        or not np.all(np.isfinite(before))
        or not np.all(np.isfinite(after))
        or abs(np.vdot(before, before) - 1) > 1e-10
        or abs(np.vdot(after, after) - 1) > 1e-10
    ):
        msg = "Phase comparison requires equally shaped, finite normalized vectors."
        raise ValueError(msg)
    overlap = complex(np.vdot(before, after))
    phase = overlap / abs(overlap) if overlap else 1.0 + 0.0j
    return phase, float(np.linalg.norm(after - phase * before))


def qiskit_state(circuit: ParameterizedCircuit, theta: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Build the independent Qiskit state without YAQS matrices or application kernels."""
    reference = QuantumCircuit(circuit.num_qubits)
    for gate in circuit.gates:
        index = cast("int", gate.param_index)
        angle = float(gate.angle_scale) * float(theta[index]) + float(gate.angle_offset)
        getattr(reference, gate.name)(angle, *gate.sites)
    return np.asarray(Statevector.from_instruction(reference).data, dtype=complex)


def audit(
    original: ParameterizedCircuit,
    original_theta: NDArray[np.float64],
    compiled: ParameterizedCircuit,
    compiled_theta: NDArray[np.float64],
    phase: complex,
    error_bound: float,
) -> dict[str, Any]:
    """Independently verify the claimed state and accumulated global phase.

    Raises:
        FloatingPointError: If either state or the claimed error/phase audit fails.
    """
    cpu, wall = time.process_time(), time.perf_counter()
    before, after = qiskit_state(original, original_theta), qiskit_state(compiled, compiled_theta)
    _, aligned = global_phase_residual(after, before)
    recorded = float(np.linalg.norm(before - phase * after))
    normalization = max(abs(float(np.vdot(before, before).real) - 1), abs(float(np.vdot(after, after).real) - 1))
    if not np.isfinite(recorded) or recorded > error_bound or normalization > 1e-10:
        msg = "Independent Qiskit state-preparation equivalence or phase audit failed."
        raise FloatingPointError(msg)
    overlap = abs(np.vdot(before, after)) / (np.linalg.norm(before) * np.linalg.norm(after))
    return {
        "passed": True,
        "phase_aligned_l2_error": aligned,
        "recorded_phase_l2_error": recorded,
        "ideal_state_infidelity": max(0.0, float(1 - overlap**2)),
        "max_normalization_error": normalization,
        "qiskit_state_preparations": 2,
        "qiskit_gate_applications": len(original.gates) + len(compiled.gates),
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall,
    }


def compile_endpoint(circuit: ParameterizedCircuit, theta: NDArray) -> dict[str, Any]:
    """Remove only global-phase gates on the current retained ideal prefix from zero.

    Gate order, resolved affine angles, noise flags and native/logical identities
    survive unchanged. Compact parameter order follows surviving original scalar
    indices. The triangle-inequality residual sum bounds ideal state error, with
    a separately reported conservative floating-point allowance. This numerical
    allowance is audited independently; it is not an interval-arithmetic proof.

    Raises:
        ValueError: If an angle overflows the finite supported endpoint domain.
    """
    cpu, wall = time.process_time(), time.perf_counter()
    values = validate(circuit, theta)
    state = np.zeros(2**circuit.num_qubits, dtype=complex)
    state[0] = 1
    retained = []
    events = []
    phases = []
    accumulated_phase = 1.0 + 0.0j
    residuals = []
    for index, gate in enumerate(circuit.gates):
        angle = circuit.angle(gate, values, None)
        if not np.isfinite(angle):
            msg = "Resolved endpoint angle is nonfinite."
            raise ValueError(msg)
        matrix, sites = circuit.gate_matrix(gate, values)
        after = _apply(state[None], matrix, sites, circuit.num_qubits)[0]
        phase, residual = global_phase_residual(state, after)
        removed = residual <= EIGENSTATE_ATOL
        events.append({
            "source_gate_index": index,
            "source_parameter_index": gate.param_index,
            "removed": removed,
            "resolved_angle": angle,
            "phase": [phase.real, phase.imag],
            "phase_aligned_l2_residual": residual,
            "noise_enabled": gate.noise_enabled,
        })
        if removed:
            residuals.append(residual)
            phases.append(math.atan2(phase.imag, phase.real))
            accumulated_phase *= phase
            accumulated_phase /= abs(accumulated_phase)
        else:
            state = after
            retained.append(index)
    parameters = sorted(cast("int", circuit.gates[index].param_index) for index in retained)
    parameter_map = {old: new for new, old in enumerate(parameters)}
    output = [
        replace(circuit.gates[index], param_index=parameter_map[cast("int", circuit.gates[index].param_index)])
        for index in retained
    ]
    compiled = ParameterizedCircuit(circuit.num_qubits, output, num_params=len(parameters))
    new_theta = values[parameters]
    source_to_output = {source: new for new, source in enumerate(retained)}
    residual_bound = math.fsum(residuals)
    roundoff = float(128 * np.finfo(float).eps * (len(circuit.gates) + 1) * 2**circuit.num_qubits)
    error_bound = residual_bound + roundoff
    compilation_cpu, compilation_wall = time.process_time() - cpu, time.perf_counter() - wall
    independent = audit(circuit, values, compiled, new_theta, accumulated_phase, error_bound)
    input_windows = sum(gate.noise_enabled for gate in circuit.gates)
    output_windows = sum(gate.noise_enabled for gate in output)
    return {
        "version": VERSION,
        "source_checksum": pilot.digest({"circuit": pilot.circuit_record(circuit), "theta": values.tolist()}),
        "circuit": pilot.circuit_record(compiled),
        "theta": new_theta.tolist(),
        "source_to_output": [source_to_output.get(index) for index in range(len(circuit.gates))],
        "parameter_to_output": [parameter_map.get(index) for index in range(circuit.num_params)],
        "retained_source_gate_indices": retained,
        "retained_source_parameter_indices": parameters,
        "output_mapping": [
            {
                "output_gate_index": new,
                "source_gate_index": old,
                "source_parameter_index": circuit.gates[old].param_index,
                "output_parameter_index": output[new].param_index,
                "logical_gate_id": output[new].logical_gate_id,
                "native_gate_id": output[new].native_gate_id,
            }
            for new, old in enumerate(retained)
        ],
        "events": events,
        "eigenstate_tolerance": EIGENSTATE_ATOL,
        "omitted_global_phase": math.atan2(accumulated_phase.imag, accumulated_phase.real),
        "omitted_global_phase_factor": [accumulated_phase.real, accumulated_phase.imag],
        "summed_removed_phase_radians": math.fsum(phases),
        "summed_removed_l2_residuals": residual_bound,
        "floating_point_allowance": roundoff,
        "ideal_state_error_bound": error_bound,
        "input_gates": len(circuit.gates),
        "output_gates": len(output),
        "input_noise_windows": input_windows,
        "output_noise_windows": output_windows,
        "removed_noise_windows": input_windows - output_windows,
        "local_statevector_gate_applications": len(circuit.gates),
        "input_state": "all_zero_only",
        "parameter_semantics": "surviving original affine parameters compacted in original parameter order",
        "noisy_equivalence_claim": False,
        "independent_audit": independent,
        "compilation_cpu_seconds": compilation_cpu,
        "compilation_wall_seconds": compilation_wall,
        "total_cpu_seconds": time.process_time() - cpu,
        "total_wall_seconds": time.perf_counter() - wall,
    }
