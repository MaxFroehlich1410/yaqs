# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Real-state Fontana ansatz with BFGS and explicitly adapted native noise.

This is a local extension through twelve qubits of Fig. 6 of arXiv:2011.01125, not author
code or a reproduction of its product-channel-after-entangling-layer model.
"""

from __future__ import annotations

import time
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from experiments.state_preparation_competitors.checkpointed_dense import make_objective
from experiments.state_preparation_competitors.common import Candidate, validate_target
from experiments.state_preparation_competitors.noisy_core import DenseObjective, nonnegative_integer, real_parameters

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from scipy.optimize import OptimizeResult


def real_target(target: ArrayLike) -> NDArray[np.complex128]:
    """Remove one global phase and require a real normalized state.

    Returns:
        Real amplitudes with the largest-magnitude amplitude positive.

    Raises:
        ValueError: If the target is invalid or genuinely complex.
    """
    vector = validate_target(target, max_qubits=12)
    aligned = vector * np.exp(-1j * np.angle(vector[np.argmax(abs(vector))]))
    if np.max(abs(aligned.imag)) > 1e-10:
        msg = "Fontana RY/CX ansatz requires a real target up to one global phase."
        raise ValueError(msg)
    return aligned.real.astype(np.complex128)


def fontana_ansatz(qubits: int, depth: int) -> QuantumCircuit:
    """Extend the paper's RY/even-CX/odd-CX layer to a q1-q12 line.

    Returns:
        Exactly n*depth RY parameters and (n-1)*depth CX gates.

    Raises:
        ValueError: If register size or positive depth is invalid.
    """
    nonnegative_integer(depth, "depth")
    if type(qubits) is not int or not 1 <= qubits <= 12 or depth < 1:
        msg = "Fontana ansatz requires q1-q12 and positive depth."
        raise ValueError(msg)
    circuit = QuantumCircuit(qubits)
    parameters = ParameterVector("fontana", qubits * depth)
    for layer in range(depth):
        for site in range(qubits):
            circuit.ry(parameters[layer * qubits + site], site)
        for parity in (0, 1):
            for site in range(parity, qubits - 1, 2):
                circuit.cx(site, site + 1)
    return circuit


class _BudgetReachedError(RuntimeError):
    """Stop before a joint objective/gradient call exceeds its declared budget."""


def optimize_bfgs(
    objective: DenseObjective, initial: ArrayLike, *, steps: int, max_evaluations: int | None = None
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Run original SciPy BFGS and retain the best accepted iterate including zero.

    Args:
        objective: Exact ideal or noisy density objective on a fixed schedule.
        initial: Finite real angles in the original template parameter order.
        steps: Maximum accepted SciPy iterations.
        max_evaluations: Hard joint oracle-call limit; default 1+20*steps.

    Returns:
        Best accepted angles and a complete parameter/fidelity/cost trace.

    Raises:
        ValueError: If a budget or initial vector is invalid.
        FloatingPointError: If the oracle returns nonfinite output.
    """
    nonnegative_integer(steps, "steps")
    cap = 1 + 20 * steps if max_evaluations is None else max_evaluations
    if type(cap) is not int or cap < 1:
        msg = "max_evaluations must be a positive integer."
        raise ValueError(msg)
    cpu, wall = time.process_time(), time.perf_counter()
    theta = real_parameters(initial, len(objective.parameters))
    calls = 0
    last_theta = None
    last_loss = 0.0
    last_gradient = np.empty_like(theta)

    def oracle(values: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        nonlocal calls, last_theta, last_loss, last_gradient
        if last_theta is not None and np.array_equal(values, last_theta):
            return last_loss, last_gradient.copy()
        if calls >= cap:
            raise _BudgetReachedError
        loss, gradient = objective.loss_gradient(values)
        calls += 1
        if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
            msg = "BFGS objective/gradient must be finite."
            raise FloatingPointError(msg)
        last_theta, last_loss, last_gradient = values.copy(), float(loss), gradient.copy()
        return last_loss, last_gradient.copy()

    initial_loss, _ = oracle(theta)
    best_theta, final_theta = theta.copy(), theta.copy()
    best_loss = final_loss = initial_loss
    best_iteration = accepted = 0
    records = []

    def record() -> None:
        records.append({
            "iteration": accepted,
            "theta": final_theta.tolist(),
            "fidelity": 1 - final_loss,
            "cumulative_objective_evaluations": calls,
            "cumulative_gradient_evaluations": calls,
            "cumulative_cpu_seconds": time.process_time() - cpu,
        })

    record()

    def callback(intermediate_result: OptimizeResult) -> None:
        nonlocal accepted, best_theta, final_theta, best_loss, final_loss, best_iteration
        accepted += 1
        final_theta, final_loss = np.asarray(intermediate_result.x).copy(), float(intermediate_result.fun)
        if final_loss < best_loss:
            best_theta, best_loss, best_iteration = final_theta.copy(), final_loss, accepted
        record()

    result = None
    reason = "zero_steps"
    if steps:
        try:
            result = minimize(
                oracle, theta, method="BFGS", jac=True, callback=callback, options={"maxiter": steps, "gtol": 1e-8}
            )
            reason = str(result.message)
        except _BudgetReachedError:
            reason = "evaluation_budget"
    return best_theta, {
        "optimizer": "scipy.optimize.minimize:BFGS",
        "scipy_version": version("scipy"),
        "retention": "best_accepted_including_initial",
        "requested_steps": steps,
        "max_evaluations": cap,
        "gtol": 1e-8,
        "steps": accepted,
        "initial_theta": theta.tolist(),
        "initial_fidelity": 1 - initial_loss,
        "final_theta": final_theta.tolist(),
        "final_fidelity": 1 - final_loss,
        "best_theta": best_theta.tolist(),
        "best_fidelity": 1 - best_loss,
        "best_iteration": best_iteration,
        "returned_fidelity": 1 - best_loss,
        "objective_evaluations": calls,
        "gradient_evaluations": calls,
        "stop_reason": reason,
        "scipy_success": None if result is None else bool(result.success),
        "records": records,
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall,
    }


def fontana_candidates(
    target: ArrayLike,
    *,
    hardware: str,
    seed: int,
    depths: tuple[int, ...],
    construction_steps: int = 200,
    noisy_steps: int = 200,
    noise_basis: str = "native",
) -> list[Candidate]:
    """Build real-only RY/CX circuits, then ideal and noisy BFGS at each depth.

    Every depth initializes independently with Uniform[-pi, pi] angles. The
    noiseless endpoint warm-starts noisy optimization on the identical schedule.
    ``noise_basis="native"`` applies the common IBM/Google synthetic channel
    after each compiled native gate; ``"logical"`` applies it after each
    original RY/CX gate without compilation. Both differ from the paper's
    channels and placement. Training exactly integrates the declared channel.

    Returns:
        Independent depth candidates with all phases and costs recorded.

    Raises:
        ValueError: If the target, profile, noise basis, seed or depth grid is invalid.
    """
    if noise_basis not in {"native", "logical"}:
        msg = "noise_basis must be native or logical."
        raise ValueError(msg)
    vector = real_target(target).astype(np.complex128)
    nonnegative_integer(seed, "seed")
    nonnegative_integer(construction_steps, "construction_steps")
    nonnegative_integer(noisy_steps, "noisy_steps")
    if hardware not in {"IBM", "Google"} or not isinstance(depths, tuple) or not depths:
        msg = "Require IBM/Google and a nonempty tuple of positive depths."
        raise ValueError(msg)
    if any(type(depth) is not int or depth < 1 for depth in depths) or tuple(sorted(set(depths))) != depths:
        msg = "Depths must be distinct increasing positive integers."
        raise ValueError(msg)
    qubits = vector.size.bit_length() - 1
    start = time.process_time()
    candidates = []
    for depth in depths:
        cpu = time.process_time()
        template = fontana_ansatz(qubits, depth)
        entropy = [seed, 201101125, depth]
        initial = np.random.default_rng(np.random.SeedSequence(entropy)).uniform(
            -np.pi, np.pi, len(template.parameters)
        )
        ideal = make_objective(template, vector, None, compile_native=noise_basis == "native")
        theta, construction = optimize_bfgs(ideal, initial, steps=construction_steps)
        noisy = make_objective(template, vector, hardware, compile_native=noise_basis == "native")
        final, refinement = optimize_bfgs(noisy, theta, steps=noisy_steps)
        circuit = noisy.bind(final)
        metadata = {
            "method": "fontana",
            "execution_backend": "exact_recursive_checkpoints" if qubits > 6 else "exact_dense_tape",
            "candidate_id": f"depth_{depth:02d}",
            "status": "complete",
            "depth": depth,
            "seed": seed,
            "seed_used": True,
            "deterministic": False,
            "hardware": hardware,
            "paper": "arXiv:2011.01125v3, Sec. IV/Fig. 6",
            "implementation_scope": "local real-state ansatz extension with SciPy BFGS and common-native noise",
            "target_restriction": "real up to global phase",
            "initialization": "independent Uniform[-pi,pi] per depth",
            "initialization_entropy": entropy,
            "preserve_native_schedule": True,
            "training_noise": "exact common-native synthetic Pauli channel; not paper product-layer noise",
            "training_trajectories": None,
            "initial_theta": initial.tolist(),
            "construction_theta": theta.tolist(),
            "theta": final.tolist(),
            "construction": construction,
            "noisy_training": refinement,
            "ideal_fidelity": float(abs(np.vdot(vector, Statevector.from_instruction(circuit).data)) ** 2),
            "training_fidelity": refinement["returned_fidelity"],
            "cpu_seconds": time.process_time() - cpu,
            "cumulative_cpu_seconds": time.process_time() - start,
        }
        if noise_basis == "logical":
            metadata.update({
                "noise_basis": "logical",
                "preserve_native_schedule": False,
                "implementation_scope": "local real-state ansatz extension with SciPy BFGS and logical-gate noise",
                "training_noise": (
                    "exact synthetic Pauli channel after each original one- or two-qubit gate; "
                    "no native compilation; not paper product-layer noise"
                ),
            })
        candidates.append(Candidate(circuit, metadata))
    return candidates
