# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Gaussian-only dense specialization of interpolative quantum state preparation.

The interpolation and Adam schedule follow arXiv:2511.15674. This local adapter
uses exact six-qubit amplitudes instead of approximate tensor interpolation,
a universal KAK SU(4) parameterization, and common-native final training noise.
It is neither author code nor a reproduction of the paper's H2 experiment.
"""

from __future__ import annotations

import itertools
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Operator, Statevector

from experiments.state_preparation_competitors.common import Candidate, validate_target
from experiments.state_preparation_competitors.evaluation import compile_candidate
from experiments.state_preparation_competitors.noisy_core import DenseObjective, nonnegative_integer, real_parameters
from experiments.state_preparation_exact_adam import AdamConfig, adam_step

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

GAUSSIAN_TARGET_ID = "gaussian_mu0p5_sigma0p1"


def gaussian_state(qubits: int, interpolation: float) -> NDArray[np.complex128]:
    """Evaluate the fixture's square-root Gaussian family on its quantics grid.

    The amplitudes are proportional to exp[-lambda*(x-0.5)^2/(4*0.1^2)].
    Thus lambda=0 is uniform and lambda=1 matches the stored Gaussian fixture;
    the paper's amplitude covariance Sigma is 2*0.1^2 for this target.

    Returns:
        Exact normalized amplitudes in the fixture's little-endian bit order.

    Raises:
        ValueError: If qubits or interpolation is outside its supported domain.
    """
    if (
        type(qubits) is not int
        or not 1 <= qubits <= 12
        or isinstance(interpolation, (bool, np.bool_))
        or not isinstance(interpolation, (int, float, np.integer, np.floating))
        or not np.isfinite(interpolation)
        or not 0 <= interpolation <= 1
    ):
        msg = "Require q1-q12 and a real interpolation coordinate in [0,1]."
        raise ValueError(msg)
    indices = np.arange(2**qubits)
    x = sum(((indices >> site) & 1) * 2.0 ** (-site - 1) for site in range(qubits))
    amplitudes = np.exp(-float(interpolation) * (x - 0.5) ** 2 / 0.04)
    return np.asarray(amplitudes / np.linalg.norm(amplitudes), dtype=np.complex128)


def validate_gaussian_target(target: ArrayLike, *, max_qubits: int = 6) -> NDArray[np.complex128]:
    """Require an existing Gaussian fixture, with an explicit twelve-qubit opt-in.

    Returns:
        A copied target aligned to the canonical positive Gaussian phase.

    Raises:
        ValueError: If an arbitrary, complex, wrong-width, or other-family state is supplied.
    """
    value = validate_target(target, max_qubits=max_qubits)
    if value.size not in {64, 4096}:
        msg = "IQSP subset accepts only the six-qubit Gaussian fixture or its explicit twelve-qubit extension."
        raise ValueError(msg)
    reference = gaussian_state(value.size.bit_length() - 1, 1.0)
    overlap = np.vdot(reference, value)
    aligned = value * np.exp(-1j * np.angle(overlap))
    if np.max(abs(aligned - reference)) > 1e-10:
        msg = "IQSP subset requires gaussian_mu0p5_sigma0p1, not an arbitrary target homotopy."
        raise ValueError(msg)
    return aligned.copy()


def iqsp_ansatz(qubits: int, depth: int) -> QuantumCircuit:
    """Build the one-dimensional H-plus-SU(4)-brickwork specialization of Fig. 6.

    Each SU(4) is represented by local ZYZ rotations before and after commuting
    XX/YY/ZZ rotations, using fifteen real coordinates. This is a universal
    KAK family chosen explicitly here; the paper specifies generic SU(4) gates.
    Zero coordinates give an exact uniform state after the initial Hadamards.

    Returns:
        A nearest-neighbor circuit with 15*(n-1)*depth parameters.

    Raises:
        ValueError: If qubits or depth is invalid.
    """
    nonnegative_integer(depth, "depth")
    if type(qubits) is not int or not 2 <= qubits <= 6 or depth < 1:
        msg = "IQSP ansatz requires q2-q6 and positive depth."
        raise ValueError(msg)
    circuit = QuantumCircuit(qubits)
    circuit.h(range(qubits))
    parameters = ParameterVector("iqsp", 15 * (qubits - 1) * depth)
    cursor = 0
    for _ in range(depth):
        for parity in (0, 1):
            for first in range(parity, qubits - 1, 2):
                pair = (first, first + 1)
                for site in pair:
                    circuit.rz(parameters[cursor], site)
                    circuit.ry(parameters[cursor + 1], site)
                    circuit.rz(parameters[cursor + 2], site)
                    cursor += 3
                circuit.rxx(parameters[cursor], *pair)
                circuit.ryy(parameters[cursor + 1], *pair)
                circuit.rzz(parameters[cursor + 2], *pair)
                cursor += 3
                for site in pair:
                    circuit.rz(parameters[cursor], site)
                    circuit.ry(parameters[cursor + 1], site)
                    circuit.rz(parameters[cursor + 2], site)
                    cursor += 3
    return circuit


def compile_su4_endpoint(
    qubits: int, depth: int, theta: ArrayLike, target: ArrayLike
) -> tuple[QuantumCircuit, NDArray[np.float64], dict[str, Any]]:
    """Compile resolved learned SU(4)s, then free each native RZ angle for tuning.

    Grouping each learned block into its exact unitary lets the existing common
    compiler synthesize a generic two-qubit unitary with at most three CX gates.
    The final noisy phase therefore does not inherit a six-CX KAK expansion or
    the constraints of its original rotation coordinates.

    Returns:
        Native template, its exact initial RZ angles, and transition diagnostics.

    Raises:
        ValueError: If parameters, target or the circuit transition is invalid.
    """
    started = time.process_time()
    original = iqsp_ansatz(qubits, depth)
    values = real_parameters(theta, len(original.parameters))
    bound = original.assign_parameters(dict(zip(original.parameters, values, strict=True)))
    source = QuantumCircuit(qubits)
    source.h(range(qubits))
    block_count = (qubits - 1) * depth
    for block_index in range(block_count):
        operations = bound.data[qubits + 15 * block_index : qubits + 15 * (block_index + 1)]
        sites = sorted({bound.find_bit(q).index for item in operations for q in item.qubits})
        block = QuantumCircuit(2)
        for item in operations:
            block.append(item.operation, [sites.index(bound.find_bit(q).index) for q in item.qubits])
        source.unitary(Operator(block).data, sites, label="learned_SU4")
    grouping_error = float(np.max(abs(Operator(bound).data - Operator(source).data)))
    if grouping_error > 1e-10:
        msg = "Grouping learned SU4 blocks changed the ideal circuit."
        raise ValueError(msg)
    compiled = compile_candidate(Candidate(source, {"method": "iqsp"}), validate_target(target))
    if compiled.metadata["logical_to_physical_output"] != list(range(qubits)):
        msg = "Nearest-neighbor IQSP compilation unexpectedly changed the wire order."
        raise ValueError(msg)
    parameters = ParameterVector("native_rz", int(compiled.circuit.count_ops().get("rz", 0)))
    native = QuantumCircuit(qubits)
    native.global_phase = compiled.circuit.global_phase
    angles = []
    for item in compiled.circuit.data:
        sites = [compiled.circuit.find_bit(q).index for q in item.qubits]
        if item.operation.name == "rz":
            native.rz(parameters[len(angles)], sites[0])
            angles.append(float(item.operation.params[0]))
        else:
            native.append(item.operation, sites)
    initial = np.array(angles, dtype=np.float64)
    recovered = native.assign_parameters(dict(zip(native.parameters, initial, strict=True)))
    reparameterization_error = float(np.max(abs(Operator(recovered).data - Operator(compiled.circuit).data)))
    if reparameterization_error > 1e-10:
        msg = "Native RZ reparameterization changed the compiled ideal circuit."
        raise ValueError(msg)
    return (
        native,
        initial,
        {
            "su4_blocks": block_count,
            "original_parameter_count": len(original.parameters),
            "native_parameter_count": len(parameters),
            "grouping_unitary_error": grouping_error,
            "native_reparameterization_error": reparameterization_error,
            "compiled": compiled.metadata,
            "noisy_parameterization": "independent RZ angles on fixed compiled SX/X/CX schedule",
            "cpu_seconds": time.process_time() - started,
        },
    )


def optimize_adam(
    objective: DenseObjective, initial: ArrayLike, *, steps: int, learning_rate: float = 0.01
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Reuse standard bias-corrected Adam with fresh moments for one IQSP phase.

    Unlike a new algorithm, this only orchestrates the repository's existing
    tested Adam update. The final iterate warm-starts the next phase as in the
    paper; the best observed iterate is recorded for diagnosis but not returned.

    Returns:
        Final angles and every evaluated iterate, including phase endpoints.

    Raises:
        ValueError: If the update budget, learning rate or initial vector is invalid.
        FloatingPointError: If an oracle or update becomes nonfinite.
    """
    nonnegative_integer(steps, "steps")
    config = AdamConfig(updates=max(1, steps), learning_rate=learning_rate)
    theta = real_parameters(initial, len(objective.parameters))
    cpu, wall = time.process_time(), time.perf_counter()
    initial_theta = theta.copy()
    first, second = np.zeros_like(theta), np.zeros_like(theta)
    best_theta = theta.copy()
    best_loss = float("inf")
    best_iteration = 0
    records = []
    for step in range(steps + 1):
        loss, gradient = objective.loss_gradient(theta)
        if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
            msg = "Adam objective/gradient must be finite."
            raise FloatingPointError(msg)
        if loss < best_loss:
            best_theta, best_loss, best_iteration = theta.copy(), float(loss), step
        records.append({
            "iteration": step,
            "theta": theta.tolist(),
            "fidelity": 1 - float(loss),
            "gradient_norm": float(np.linalg.norm(gradient)),
            "cumulative_objective_evaluations": step + 1,
            "cumulative_gradient_evaluations": step + 1,
            "cumulative_cpu_seconds": time.process_time() - cpu,
        })
        if step < steps:
            theta, first, second = adam_step(theta, gradient, first, second, step + 1, config)
    return theta, {
        "optimizer": "existing standard bias-corrected Adam",
        "update_source": "experiments/state_preparation_exact_adam.py:adam_step",
        "retention": "final_iterate; best recorded only",
        "moments": "reset at each interpolation stage and final noisy phase",
        "learning_rate": config.learning_rate,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "epsilon": config.epsilon,
        "steps": steps,
        "initial_theta": initial_theta.tolist(),
        "initial_fidelity": records[0]["fidelity"],
        "final_theta": theta.tolist(),
        "final_fidelity": records[-1]["fidelity"],
        "returned_fidelity": records[-1]["fidelity"],
        "best_theta": best_theta.tolist(),
        "best_fidelity": 1 - best_loss,
        "best_iteration": best_iteration,
        "objective_evaluations": steps + 1,
        "gradient_evaluations": steps + 1,
        "first_moment": first.tolist(),
        "second_moment": second.tolist(),
        "records": records,
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall,
    }


def iqsp_candidates(
    target: ArrayLike,
    *,
    hardware: str,
    seed: int,
    depths: tuple[int, ...],
    construction_steps: int = 200,
    noisy_steps: int = 200,
    interpolation_steps: int = 20,
    learning_rate: float = 0.01,
) -> list[Candidate]:
    """Follow the Gaussian family, then tune only the last target with native noise.

    Construction steps are Adam updates PER nonzero lambda stage, not a total
    across all stages. Defaults cost 20*200 ideal updates plus 200 noisy updates
    per depth. Candidate depths restart independently at the exact uniform state.
    Seed is validated and recorded but deliberately unused: this identity-start
    algorithm is deterministic and must not receive fictitious seed replicas.

    Returns:
        Gaussian-only candidates with all interpolation and final training traces.

    Raises:
        ValueError: If the target, profile, depth grid or budgets are invalid.
    """
    vector = validate_gaussian_target(target)
    for value, name in (
        (seed, "seed"),
        (construction_steps, "construction_steps"),
        (noisy_steps, "noisy_steps"),
        (interpolation_steps, "interpolation_steps"),
    ):
        nonnegative_integer(value, name)
    AdamConfig(learning_rate=learning_rate)
    if hardware not in {"IBM", "Google"} or interpolation_steps < 1 or not isinstance(depths, tuple) or not depths:
        msg = "Require IBM/Google, positive interpolation_steps, and a nonempty positive depth tuple."
        raise ValueError(msg)
    if any(type(depth) is not int or depth < 1 for depth in depths) or tuple(sorted(set(depths))) != depths:
        msg = "Depths must be distinct increasing positive integers."
        raise ValueError(msg)
    start = time.process_time()
    candidates = []
    path = [gaussian_state(6, k / interpolation_steps) for k in range(interpolation_steps + 1)]
    path_distances = [float(np.linalg.norm(right - left)) for left, right in itertools.pairwise(path)]
    for depth in depths:
        cpu = time.process_time()
        template = iqsp_ansatz(6, depth)
        theta = np.zeros(len(template.parameters), dtype=np.float64)
        stages = []
        for index, intermediate in enumerate(path[1:], start=1):
            ideal = DenseObjective(template, intermediate, None, compile_native=False)
            theta, trace = optimize_adam(ideal, theta, steps=construction_steps, learning_rate=learning_rate)
            stages.append({"lambda": index / interpolation_steps, **trace})
        ideal_theta = theta.copy()
        native, native_initial, transition = compile_su4_endpoint(6, depth, theta, vector)
        noisy = DenseObjective(native, vector, hardware, compile_native=False)
        final, refinement = optimize_adam(noisy, native_initial, steps=noisy_steps, learning_rate=learning_rate)
        circuit = noisy.bind(final)
        candidates.append(
            Candidate(
                circuit,
                {
                    "method": "iqsp",
                    "candidate_id": f"depth_{depth:02d}",
                    "status": "complete",
                    "depth": depth,
                    "seed": seed,
                    "seed_used": False,
                    "deterministic": True,
                    "hardware": hardware,
                    "target_id": GAUSSIAN_TARGET_ID,
                    "paper": "arXiv:2511.15674v1, Eq.6 and Fig.6",
                    "implementation_scope": "dense Gaussian specialization; KAK SU4; common-native final noise",
                    "target_representation": "exact 64 amplitudes; no TCI approximation or scalable-runtime claim",
                    "amplitude_covariance": 0.02,
                    "initialization": "H on every qubit; all SU4 parameters zero, independent for each depth",
                    "initial_theta": np.zeros(len(template.parameters)).tolist(),
                    "preserve_native_schedule": True,
                    "training_noise": "only final target; exact common-native synthetic Pauli channel, not H2",
                    "training_trajectories": None,
                    "interpolation_steps": interpolation_steps,
                    "lambda_grid": [k / interpolation_steps for k in range(interpolation_steps + 1)],
                    "adjacent_target_distances": path_distances,
                    "construction_steps_per_stage": construction_steps,
                    "construction_steps_total": construction_steps * interpolation_steps,
                    "construction_theta": ideal_theta.tolist(),
                    "native_initial_theta": native_initial.tolist(),
                    "native_transition": transition,
                    "theta": final.tolist(),
                    "interpolation": stages,
                    "noisy_training": refinement,
                    "ideal_fidelity": float(abs(np.vdot(vector, Statevector.from_instruction(circuit).data)) ** 2),
                    "training_fidelity": refinement["returned_fidelity"],
                    "cpu_seconds": time.process_time() - cpu,
                    "cumulative_cpu_seconds": time.process_time() - start,
                },
            )
        )
    return candidates
