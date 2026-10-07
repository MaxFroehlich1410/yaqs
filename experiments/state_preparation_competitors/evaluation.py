# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Common compiled circuits, independent validation, and exact final assessment.

The historical synthetic Pauli coefficients are now applied after every native
RZ/SX/X/CX operation, including RZ. This is a new deployment evaluation model,
not a reinterpretation of the archived logical-rotation fidelities or a device
calibration. Routing is charged, and output permutations are accounted for.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.quantum_info import DensityMatrix, Operator, Pauli, Statevector
from qiskit.transpiler import CouplingMap

from experiments.legacy_topdown_dense import noise_channel
from experiments.state_preparation_competitors.common import Candidate, validate_target
from experiments.state_preparation_competitors.exact_synthesis import repair_opaque_two_qubit_gates
from mqt.yaqs.optimization.statevector import _apply  # noqa: PLC2701  # Reuse tested kernels.

if TYPE_CHECKING:
    from numpy.typing import NDArray

NATIVE_GATES = ("rz", "sx", "x", "cx")
HARDWARE = ("IBM", "Google")
COMPILER: dict[str, Any] = {
    "basis_gates": list(NATIVE_GATES),
    "optimization_level": 1,
    "seed_transpiler": 1729,
    "approximation_degree": 1.0,
    "routing_method": "basic",
}


@dataclass(frozen=True)
class CompiledCandidate:
    """A physical circuit, its permuted target, and a compilation receipt."""

    circuit: QuantumCircuit
    target: NDArray[np.complex128]
    metadata: dict[str, Any]


def validate_circuit(circuit: QuantumCircuit, *, native: bool = False, max_qubits: int = 6) -> None:
    """Require a resolved unitary circuit without extra qubits or classical control.

    Args:
        circuit: Circuit operating on a one- to six-qubit register.
        max_qubits: Explicit register limit; legacy callers remain capped at six.
        native: Additionally require the common nearest-neighbor native basis.

    Raises:
        ValueError: If the circuit register, parameters, or unitary are invalid.
        TypeError: If an instruction is not a unitary gate.
    """
    if (
        type(max_qubits) is not int
        or not 1 <= max_qubits <= 12
        or not 1 <= circuit.num_qubits <= max_qubits
        or circuit.num_clbits
        or circuit.parameters
        or not np.isfinite(float(circuit.global_phase))
    ):
        msg = "Expected a resolved unitary q1-q6 circuit without classical bits."
        raise ValueError(msg)
    for instruction in circuit.data:
        operation = instruction.operation
        if not isinstance(operation, Gate):
            msg = "Measurement, reset, barriers, and classical control are not preparation gates."
            raise TypeError(msg)
        sites = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if native and (
            operation.name not in NATIVE_GATES
            or (len(sites) == 2 and abs(sites[0] - sites[1]) != 1)
            or len(sites) not in {1, 2}
        ):
            msg = "Expected the nearest-neighbor RZ/SX/X/CX basis."
            raise ValueError(msg)
        matrix = Operator(operation).data
        if not np.all(np.isfinite(matrix)) or not np.allclose(
            matrix.conj().T @ matrix, np.eye(matrix.shape[0]), atol=1e-10, rtol=0
        ):
            msg = "Preparation operations must be finite and unitary."
            raise ValueError(msg)


def physical_target(target: NDArray[np.complex128], permutation: list[int]) -> NDArray[np.complex128]:
    """Map logical target amplitudes to the compiler's final physical wire order.

    Args:
        target: Logical little-endian target amplitudes.
        permutation: Physical output index of each logical qubit.

    Returns:
        The same state with relabeled output wires; no physical SWAP is added.

    Raises:
        ValueError: If the permutation does not match the target register.
    """
    value = validate_target(target)
    qubits = value.size.bit_length() - 1
    if (
        len(permutation) != qubits
        or any(type(site) is not int for site in permutation)
        or sorted(permutation) != list(range(qubits))
    ):
        msg = "Output permutation must contain each target qubit exactly once."
        raise ValueError(msg)
    logical = np.arange(value.size)
    physical = sum(((logical >> site) & 1) << destination for site, destination in enumerate(permutation))
    result = np.empty_like(value)
    result[physical] = value
    return result


def compile_candidate(candidate: Candidate, target: NDArray[np.complex128]) -> CompiledCandidate:
    """Compile every algorithm identically and audit the complete ideal unitary.

    Args:
        candidate: Original circuit and method-specific receipt.
        target: Target in the original logical wire order.

    Returns:
        Native circuit, physical target and explicit routing/resource statistics.

    Raises:
        ValueError: If register size, unitary equivalence, or layout is invalid.
    """
    target = validate_target(target)
    source = candidate.circuit
    validate_circuit(source)
    if target.size != 2**source.num_qubits or source.layout is not None:
        msg = "Source circuit must match the target and have no previous transpiler layout."
        raise ValueError(msg)
    cpu, wall = time.process_time(), time.perf_counter()
    expected = Operator(source).data
    compilation_input, fallback_count = source, 0
    initial_error = 0.0
    for attempt in range(2):
        compiled = transpile(
            compilation_input,
            coupling_map=CouplingMap.from_line(source.num_qubits) if source.num_qubits > 1 else None,
            initial_layout=list(range(source.num_qubits)),
            **COMPILER,
        )
        validate_circuit(compiled, native=True)
        if compiled.num_qubits != source.num_qubits:
            msg = "Compilation may not introduce ancillary qubits."
            raise ValueError(msg)
        actual = Operator.from_circuit(compiled).data
        overlap = np.vdot(expected, actual)
        phase = overlap / abs(overlap) if abs(overlap) else 1
        error = float(np.max(abs(actual - phase * expected)))
        if attempt == 0:
            initial_error = error
        if error <= 1e-9:
            break
        if attempt == 1:
            msg = "Compilation failed the independent ideal-unitary equivalence check."
            raise ValueError(msg)
        compilation_input, fallback_count = repair_opaque_two_qubit_gates(source, COMPILER)
    permutation = (
        compiled.layout.final_index_layout() if compiled.layout is not None else list(range(source.num_qubits))
    )
    mapped_target = physical_target(target, permutation)
    ideal = float(abs(np.vdot(mapped_target, Statevector.from_instruction(compiled).data)) ** 2)
    original_ideal = float(abs(np.vdot(target, Statevector.from_instruction(source).data)) ** 2)
    if abs(ideal - original_ideal) > 1e-9:
        msg = "Compiler output permutation changed target-state fidelity."
        raise ValueError(msg)
    two_qubit_depth = compiled.depth(lambda instruction: len(instruction.qubits) == 2)
    return CompiledCandidate(
        compiled,
        mapped_target,
        {
            "compiler": COMPILER.copy(),
            "initial_layout": list(range(source.num_qubits)),
            "logical_to_physical_output": permutation,
            "initial_ideal_unitary_error": initial_error,
            "ideal_unitary_error": error,
            "exact_synthesis_fallback_count": fallback_count,
            "exact_synthesis_fallback_policy": "repair inaccurate opaque SU4 only after failed full-unitary audit",
            "ideal_fidelity": ideal,
            "source_gate_count": len(source.data),
            "compiled_one_qubit": sum(len(item.qubits) == 1 for item in compiled.data),
            "compiled_two_qubit": sum(len(item.qubits) == 2 for item in compiled.data),
            "compiled_two_qubit_depth": two_qubit_depth,
            "compiled_total_depth": compiled.depth(),
            "compiled_operations": dict(compiled.count_ops()),
            "compilation_cpu_seconds": time.process_time() - cpu,
            "compilation_wall_seconds": time.perf_counter() - wall,
            "noise_placement": "after every compiled operation, including RZ; no idle or measurement noise",
        },
    )


def preserve_native_candidate(candidate: Candidate, target: NDArray[np.complex128]) -> CompiledCandidate:
    """Audit an already-native trained schedule without changing its noise windows.

    Noisy competitors compile their symbolic ansatz before optimization. Applying
    angle-dependent cancellation afterward would evaluate a different channel.
    Their explicit receipt requests this path; every gate and output wire stays
    fixed. The same native gate alphabet, line topology and evaluation channels
    apply as in ``compile_candidate``.

    Args:
        candidate: Resolved nearest-neighbor native circuit and explicit receipt.
        target: Logical target; routing/layout changes are forbidden here.

    Returns:
        A private circuit/target with the usual physical resource fields.

    Raises:
        ValueError: If the native schedule receipt, target or layout is invalid.
    """
    cpu, wall = time.process_time(), time.perf_counter()
    source, target = candidate.circuit, validate_target(target)
    validate_circuit(source, native=True)
    if (
        candidate.metadata.get("preserve_native_schedule") is not True
        or source.layout is not None
        or target.size != 2**source.num_qubits
    ):
        msg = "A frozen native candidate needs an explicit schedule receipt and an unchanged output layout."
        raise ValueError(msg)
    ideal = float(abs(np.vdot(target, Statevector.from_instruction(source).data)) ** 2)
    return CompiledCandidate(
        source.copy(),
        target,
        {
            "compiler": COMPILER.copy(),
            "compiler_role": "shared policy reference; native schedule audited without transformation",
            "compilation_stage": "native schedule frozen before noisy training; audit only at evaluation",
            "initial_layout": list(range(source.num_qubits)),
            "logical_to_physical_output": list(range(source.num_qubits)),
            "initial_ideal_unitary_error": 0.0,
            "ideal_unitary_error": 0.0,
            "exact_synthesis_fallback_count": 0,
            "ideal_fidelity": ideal,
            "source_gate_count": len(source.data),
            "compiled_one_qubit": sum(len(item.qubits) == 1 for item in source.data),
            "compiled_two_qubit": sum(len(item.qubits) == 2 for item in source.data),
            "compiled_two_qubit_depth": source.depth(lambda instruction: len(instruction.qubits) == 2),
            "compiled_total_depth": source.depth(),
            "compiled_operations": dict(source.count_ops()),
            "compilation_cpu_seconds": time.process_time() - cpu,
            "compilation_wall_seconds": time.perf_counter() - wall,
            "noise_placement": "after every compiled operation, including RZ; no idle or measurement noise",
        },
    )


@lru_cache(maxsize=None, typed=True)
def pauli_distribution(hardware: str, arity: int) -> tuple[NDArray[np.float64], tuple[NDArray[np.complex128], ...]]:
    """Return the literal one-jump channel in Qiskit local basis ordering.

    Args:
        hardware: Historical synthetic profile name, IBM or Google.
        arity: One or two acted-on qubits.

    Returns:
        Immutable categorical probabilities and corresponding local Pauli matrices.

    Raises:
        ValueError: If a profile or gate support is unsupported.
    """
    if hardware not in HARDWARE or type(arity) is not int or arity not in {1, 2}:
        msg = "Expected an IBM/Google profile and one- or two-qubit gate."
        raise ValueError(msg)
    one, two = (3e-4, 3e-3) if hardware == "IBM" else (6e-4, 5e-3)
    labels = ["X", "Y", "Z"]
    if arity == 2:
        labels = [axis + "I" for axis in labels] + ["I" + axis for axis in labels]
    rates = [one / 3] * len(labels)
    if arity == 2:
        correlated = ["XX", "ZZ"] if hardware == "IBM" else ["XX", "YY", "ZZ"]
        labels += correlated
        rates += [two / len(correlated)] * len(correlated)
    rate = np.array(rates)
    total = float(rate.sum())
    probabilities = np.concatenate(([np.exp(-total)], -np.expm1(-total) * rate / total))
    matrices = tuple(Pauli(label).to_matrix() for label in ["I" * arity, *labels])
    probabilities.setflags(write=False)
    for matrix in matrices:
        matrix.setflags(write=False)
    return probabilities, matrices


def sample_validation(circuit: QuantumCircuit, hardware: str, count: int, entropy: list[int]) -> NDArray[np.uint8]:
    """Sample unconditional fixed native-gate histories independently of training.

    Args:
        circuit: Compiled nearest-neighbor native circuit.
        hardware: Synthetic Pauli profile.
        count: At least two raw validation trajectories.
        entropy: Predeclared validation stream identifier.

    Returns:
        One categorical Pauli code for every raw trajectory and native gate.

    Raises:
        ValueError: If the circuit, count, profile, or entropy is invalid.
    """
    validate_circuit(circuit, native=True)
    pauli_distribution(hardware, 1)
    if type(count) is not int or count < 2 or not entropy or any(type(x) is not int or x < 0 for x in entropy):
        msg = "Validation needs at least two paths and nonnegative integer stream entropy."
        raise ValueError(msg)
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    codes = np.zeros((count, len(circuit.data)), dtype=np.uint8)
    for index, item in enumerate(circuit.data):
        probabilities, _ = pauli_distribution(hardware, len(item.qubits))
        codes[:, index] = rng.choice(len(probabilities), size=count, p=probabilities)
    return codes


def validation_score(candidate: CompiledCandidate, hardware: str, codes: NDArray[np.uint8]) -> dict[str, Any]:
    """Evaluate frozen paths with exact empirical no-jump duplicate compression.

    Args:
        candidate: Compiled circuit with target already mapped to physical wires.
        hardware: Synthetic Pauli profile.
        codes: Unconditional path bank; no resampling or forcing jumps is allowed.

    Returns:
        Mean, raw-sample standard error, jump coverage and bank hash.

    Raises:
        ValueError: If codes, register size, circuit or profile are invalid.
    """
    circuit, target = candidate.circuit, validate_target(candidate.target)
    validate_circuit(circuit, native=True)
    pauli_distribution(hardware, 1)
    codes = np.asarray(codes)
    if (
        codes.ndim != 2
        or codes.shape[1] != len(circuit.data)
        or len(codes) < 2
        or codes.dtype.kind not in "iu"
        or np.any(codes < 0)
        or target.size != 2**circuit.num_qubits
    ):
        msg = "Validation bank must contain integer gate codes matching the circuit and target."
        raise ValueError(msg)
    has_jump = np.any(codes != 0, axis=1)
    indices = np.flatnonzero(has_jump).tolist()
    multiplicities = [1] * len(indices)
    no_jump = int(np.count_nonzero(~has_jump))
    if no_jump:
        indices.insert(0, int(np.flatnonzero(~has_jump)[0]))
        multiplicities.insert(0, no_jump)
    retained = codes[indices]
    states = np.zeros((len(retained), target.size), dtype=np.complex128)
    states[:, 0] = 1
    analytic_no_jump = 1.0
    for index, item in enumerate(circuit.data):
        sites = tuple(circuit.find_bit(qubit).index for qubit in item.qubits)
        probabilities, matrices = pauli_distribution(hardware, len(sites))
        if np.any(codes[:, index] >= len(probabilities)):
            msg = "Invalid Pauli code for the gate support."
            raise ValueError(msg)
        analytic_no_jump *= float(probabilities[0])
        # _apply uses most-significant-first local indices; Qiskit uses qargs[0] as LSB.
        states = _apply(states, Operator(item.operation).data, tuple(reversed(sites)), circuit.num_qubits)
        for code in range(1, len(matrices)):
            selected = np.flatnonzero(retained[:, index] == code)
            if selected.size:
                states[selected] = _apply(states[selected], matrices[code], tuple(reversed(sites)), circuit.num_qubits)
    fidelities = abs(states @ target.conj()) ** 2
    weights = np.array(multiplicities, dtype=float) / len(codes)
    mean = float(weights @ fidelities)
    return {
        "fidelity": mean,
        "standard_error": float(np.sqrt(weights @ (fidelities - mean) ** 2 / (len(codes) - 1))),
        "sampled_paths": len(codes),
        "no_jump_paths": no_jump,
        "jump_paths": int(has_jump.sum()),
        "retained_paths": len(retained),
        "analytic_no_jump_probability": analytic_no_jump,
        "norm_max_error": float(np.max(abs(np.sum(abs(states) ** 2, axis=1) - 1))),
        "bank_sha256": hashlib.sha256(codes.astype(np.uint8).tobytes()).hexdigest(),
    }


def exact_score(candidate: CompiledCandidate, hardware: str, *, law: str = "tjm") -> dict[str, float]:
    """Integrate all modeled noise outcomes after a candidate selection is locked.

    Args:
        candidate: Physical compiled circuit and output-permuted target.
        hardware: Synthetic Pauli profile.
        law: Matching one-jump law, full Lindblad sensitivity, or noiseless check.

    Returns:
        Target fidelity and density-matrix validity diagnostics, without MC error.

    Raises:
        ValueError: If the model, circuit, target or final density is invalid.
    """
    circuit, target = candidate.circuit, validate_target(candidate.target)
    validate_circuit(circuit, native=True)
    pauli_distribution(hardware, 1)
    if target.size != 2**circuit.num_qubits or law not in {"tjm", "lindblad", "noiseless"}:
        msg = "Expected a matching target and tjm/lindblad/noiseless evaluation law."
        raise ValueError(msg)
    rho = DensityMatrix.from_label("0" * circuit.num_qubits)
    for item in circuit.data:
        sites = [circuit.find_bit(qubit).index for qubit in item.qubits]
        rho = rho.evolve(item.operation, qargs=sites)
        if law != "noiseless":
            rho = rho.evolve(noise_channel(hardware, len(sites), law), qargs=sites)
    matrix = rho.data
    result = {
        "fidelity": float(np.vdot(target, matrix @ target).real),
        "trace_error": float(abs(np.trace(matrix) - 1)),
        "hermiticity_error": float(np.max(abs(matrix - matrix.conj().T))),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(matrix).min()),
    }
    if result["trace_error"] > 1e-9 or result["hermiticity_error"] > 1e-9 or result["minimum_eigenvalue"] < -1e-9:
        msg = "Exact channel propagation produced an invalid density matrix."
        raise ValueError(msg)
    return result
