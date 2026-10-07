# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compile pruning-B circuits before weighted native-noise refinement.

Compilation changes noise windows and may reorder trainable gate occurrences.
The historical online/batch update formulas and budgets are reused, with online
sweeps following the new native order. Every original scalar parameter must
still occur exactly once; shared or merged coordinates are rejected to avoid
changing the update frequency. The returned circuit preserves its trained native
schedule, including fixed gates and zero-angle rotations.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Operator, Statevector

from experiments import legacy_topdown_dense as legacy
from experiments.state_preparation_competitors.common import ROTATIONS, Candidate, from_parameterized, validate_target
from experiments.state_preparation_competitors.evaluation import COMPILER, NATIVE_GATES
from experiments.state_preparation_competitors.noisy_core import nonnegative_integer, real_parameters
from mqt.yaqs.optimization import ParameterizedCircuit, StatevectorBackend

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from qiskit.circuit import Parameter


@dataclass(frozen=True)
class TrainingCircuit:
    """Fixed symbolic native schedule with original-coordinate backend and target."""

    template: QuantumCircuit
    parameters: tuple[Parameter, ...]
    circuit: ParameterizedCircuit
    target: NDArray[np.complex128]
    compilation: dict[str, Any]

    def bind(self, theta: ArrayLike) -> QuantumCircuit:
        """Bind original coordinates without compiling or removing zero gates."""
        values = real_parameters(theta, len(self.parameters))
        return self.template.assign_parameters(dict(zip(self.parameters, values, strict=True)))


def audit_ideal_equivalence(
    original: ParameterizedCircuit, compiled: TrainingCircuit, theta: ArrayLike
) -> dict[str, float]:
    """Compare full logical/native unitaries up to phase and backend fidelity.

    Args:
        original: Original scalar-Pauli circuit before native compilation.
        compiled: Fixed compiled template and native statevector backend circuit.
        theta: Original parameter coordinates at the audit point.

    Returns:
        Maximum phase-aligned unitary error and backend fidelity disagreement.

    Raises:
        ValueError: If parameters are invalid or ideal equivalence fails.
    """
    values = real_parameters(theta, original.num_params)
    logical = Operator(from_parameterized(original, values)).data
    bound = compiled.bind(values)
    native = Operator(bound).data
    overlap = np.vdot(native, logical) / logical.shape[0]
    phase = overlap / abs(overlap) if abs(overlap) else 1.0
    error = float(np.max(abs(logical - phase * native)))
    expected = float(abs(np.vdot(compiled.target, Statevector.from_instruction(bound).data)) ** 2)
    actual = StatevectorBackend(compiled.circuit, compiled.target).metrics(values)[1]
    fidelity_error = float(abs(actual - expected))
    if not np.isfinite(error + fidelity_error) or max(error, fidelity_error) > 1e-10:
        msg = "Native compilation failed ideal unitary or statevector fidelity equivalence."
        raise ValueError(msg)
    return {"phase_aligned_unitary_max_error": error, "backend_fidelity_error": fidelity_error}


def compile_training(circuit: ParameterizedCircuit, target: ArrayLike) -> TrainingCircuit:
    """Compile original scalar rotations once, preserving their parameter order.

    Args:
        circuit: Nearest-neighbor q1-q6 scalar-Pauli circuit with all noise enabled.
        target: Normalized target amplitudes in Qiskit little-endian order.

    Returns:
        Native template, exact original coordinate order, backend and audit receipt.

    Raises:
        ValueError: If the source or compiled coordinates, gates or target are unsupported.
        TypeError: If the source is not a ParameterizedCircuit.
    """
    if not isinstance(circuit, ParameterizedCircuit):
        msg = "Expected a ParameterizedCircuit of scalar Pauli rotations."
        raise TypeError(msg)
    value = validate_target(target)
    if value.size != 2**circuit.num_qubits:
        msg = "Source register and target must match."
        raise ValueError(msg)
    counts = np.zeros(circuit.num_params, dtype=int)
    parameters = tuple(ParameterVector("original", circuit.num_params))
    symbolic = QuantumCircuit(circuit.num_qubits)
    for gate in circuit.gates:
        if (
            gate.name not in ROTATIONS
            or gate.data_map is not None
            or not gate.noise_enabled
            or not np.isfinite(gate.angle_scale)
            or not np.isfinite(gate.angle_offset)
            or (gate.param_index is not None and gate.angle_scale == 0)
            or (len(gate.sites) == 2 and abs(gate.sites[0] - gate.sites[1]) != 1)
        ):
            msg = "Require finite scalar Pauli rotations, all noise enabled, on nearest-neighbor supports."
            raise ValueError(msg)
        angle = gate.angle_offset
        if gate.param_index is not None:
            counts[gate.param_index] += 1
            angle += gate.angle_scale * parameters[gate.param_index]
        symbolic.append(ROTATIONS[gate.name](angle), list(gate.sites))
    if np.any(counts != 1):
        msg = "Each original parameter must occur exactly once; missing or shared coordinates are unsupported."
        raise ValueError(msg)
    cpu, wall = time.process_time(), time.perf_counter()
    template = transpile(symbolic, **COMPILER)
    if template.layout is not None or any(item.operation.name not in NATIVE_GATES for item in template.data):
        msg = "Expected an unrouted RZ/SX/X/CX native template."
        raise ValueError(msg)
    if set(template.parameters) != set(parameters):
        msg = "Native compilation removed an original parameter."
        raise ValueError(msg)
    native = ParameterizedCircuit.from_qiskit(template, list(parameters))
    native_counts = np.zeros(circuit.num_params, dtype=int)
    for gate in native.gates:
        if gate.param_index is not None:
            native_counts[gate.param_index] += 1
    if np.any(native_counts != 1):
        msg = "Each compiled parameter must occur exactly once to preserve update frequency."
        raise ValueError(msg)
    receipt: dict[str, Any] = {
        "compiler": copy.deepcopy(COMPILER),
        "source_gate_count": len(circuit.gates),
        "native_gate_count": len(native.gates),
        "native_operations": dict(template.count_ops()),
        "parameter_count": len(parameters),
        "parameter_order": [str(parameter) for parameter in parameters],
        "parameter_occurrences": native_counts.tolist(),
        "original_parameter_occurrence_order": [
            gate.param_index for gate in circuit.gates if gate.param_index is not None
        ],
        "native_parameter_occurrence_order": [
            gate.param_index for gate in native.gates if gate.param_index is not None
        ],
        "online_order": "compiled native order; may differ from logical chronological order",
        "backend_global_phase": "ignored; state-preparation fidelity is invariant under global phase",
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall,
    }
    result = TrainingCircuit(template, parameters, native, value, receipt)
    receipt["ideal_audit"] = audit_ideal_equivalence(circuit, result, np.linspace(-0.37, 0.53, circuit.num_params))
    return result


def train_candidate(
    circuit: ParameterizedCircuit,
    initial: ArrayLike,
    target: ArrayLike,
    *,
    hardware: str = "IBM",
    seed: int = 5300,
    steps: int = 200,
    online: int = 100,
    rate: float = 0.2,
    num_trajectories: int = 100,
    require_jump_coverage: bool = True,
) -> Candidate:
    """Run historical noisy Krotov update formulas in compiled native gate order.

    Args:
        circuit: Original pruned circuit before noisy optimization.
        initial: Noiseless/pruned starting point in its original coordinates.
        target: Normalized target in little-endian order.
        hardware: Historical IBM or Google synthetic Pauli noise profile.
        seed: Frozen training bank counter; default matches optimizer seed 300.
        steps: Total updates, default 200.
        online: Initial sequential online sweeps, default 100.
        rate: Fixed step size for both phases, default 0.2.
        num_trajectories: Raw unconditional bank size before no-jump compression.
        require_jump_coverage: Stop explicitly if a nonempty circuit's bank has no jumps.

    Returns:
        Final-iterate native circuit with its original-coordinate parameters,
        unmodified training trace, noise-bank events and compilation audit.

    Raises:
        ValueError: If budgets, model, parameters or compilation are invalid.
        RuntimeError: If required jump coverage is absent; no resampling occurs.
        TypeError: If the source is not a ParameterizedCircuit.
    """
    for number, name in ((seed, "seed"), (steps, "steps"), (online, "online"), (num_trajectories, "num_trajectories")):
        nonnegative_integer(number, name)
    if (
        hardware not in {"IBM", "Google"}
        or online > steps
        or num_trajectories < 1
        or isinstance(rate, bool)
        or not np.isfinite(rate)
        or rate < 0
        or type(require_jump_coverage) is not bool
    ):
        msg = "Require IBM/Google, online <= steps, positive trajectories, finite nonnegative rate and bool coverage."
        raise ValueError(msg)
    cpu, wall = time.process_time(), time.perf_counter()
    if not isinstance(circuit, ParameterizedCircuit):
        msg = "Expected a ParameterizedCircuit of scalar Pauli rotations."
        raise TypeError(msg)
    theta = real_parameters(initial, circuit.num_params)
    compiled = compile_training(circuit, target)
    initial_audit = audit_ideal_equivalence(circuit, compiled, theta)
    training_cpu = time.process_time()
    final, training = legacy.train(
        compiled.circuit,
        theta,
        compiled.target,
        steps=steps,
        rate=rate,
        seed=seed,
        decay=0.0,
        noisy=True,
        crn=True,
        online=online,
        stop=2.0,
        hardware=hardware,
        num_trajectories=num_trajectories,
        compress_no_jump=True,
        record_noise_banks=True,
        require_jump_coverage=require_jump_coverage,
    )
    training_seconds = time.process_time() - training_cpu
    final = real_parameters(final, compiled.circuit.num_params)
    return Candidate(
        compiled.bind(final),
        {
            "method": "current_b_native_refinement",
            "candidate_id": "native_refinement",
            "status": "complete",
            "hardware": hardware,
            "seed": seed,
            "noise_basis": "native",
            "preserve_native_schedule": True,
            "training_noise": "historical arity-dependent Pauli channel after each fixed compiled native gate",
            "optimizer": "experiments.legacy_topdown_dense.train",
            "retention": "final_iterate_historical_noisy_policy",
            "gradient_estimator": "historical cross-trajectory signal; not asserted to be the sample-loss gradient",
            "initial_theta": theta.tolist(),
            "final_theta": final.tolist(),
            "theta": final.tolist(),
            "compilation": compiled.compilation,
            "initial_ideal_audit": initial_audit,
            "training": training,
            "training_cpu_seconds": training_seconds,
            "training_fidelity": training["fidelity_trace"][-1] if training["fidelity_trace"] else None,
            "ideal_fidelity": StatevectorBackend(compiled.circuit, compiled.target).metrics(final)[1],
            "cpu_seconds": time.process_time() - cpu,
            "wall_seconds": time.perf_counter() - wall,
        },
    )
