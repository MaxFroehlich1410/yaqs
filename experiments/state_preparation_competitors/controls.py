# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Conventional fixed-ansatz VQSP and an unpruned layerwise internal control.

SciPy supplies the conventional optimizer. The layerwise control reuses the
current construction and noisy-refinement implementations without pruning.
Neither control implements operator-pool ADAPT-VQE.
"""

from __future__ import annotations

import hashlib
import json
import time
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import minimize

from experiments.legacy_topdown_dense import LegacyNoise, extend, train
from experiments.legacy_topdown_q6 import brickwall_for
from experiments.state_preparation_competitors.common import Candidate, from_parameterized, validate_target
from experiments.state_preparation_pilot import circuit_record
from experiments.topdown_batch_refinement_policies import train_noiseless
from mqt.yaqs.optimization import KrotovTJMOptions, StatevectorBackend

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.optimize import OptimizeResult

    from mqt.yaqs.optimization import ParameterizedCircuit
    from mqt.yaqs.optimization.statevector import NoiseMaps

LBFGSB_OPTIONS = {"maxcor": 10, "maxls": 20, "ftol": 1e-12, "gtol": 1e-8}


class _EvaluationBudgetError(RuntimeError):
    """Stop before an oracle call would exceed the declared hard cap."""


def _nonnegative_integer(value: int, name: str) -> None:
    """Reject negative and noninteger budgets or seeds.

    Raises:
        ValueError: If the value is not a nonnegative built-in integer.
    """
    if type(value) is not int or value < 0:
        msg = f"{name} must be a nonnegative integer."
        raise ValueError(msg)


def optimize_lbfgsb(
    circuit: ParameterizedCircuit,
    initial: NDArray[np.float64],
    target: NDArray[np.complex128],
    *,
    steps: int,
    maps: NoiseMaps | None = None,
    weights: NDArray[np.float64] | None = None,
    max_evaluations: int | None = None,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Run SciPy L-BFGS-B on a true, optionally weighted, frozen objective.

    Args:
        circuit: Fixed topology with unwrapped, unbounded scalar angles.
        initial: Initial parameters; never changed in place.
        target: Normalized target vector in repository basis order.
        steps: Maximum accepted SciPy iterations; convergence may stop earlier.
        maps: Fixed parameter-independent unitary histories, or noiseless.
        weights: Empirical history weights, including any no-jump compression.
        max_evaluations: Hard joint loss/gradient call cap. Defaults to one
            initial call plus twenty calls per allowed iteration.

    Returns:
        Best actual accepted parameters, including iteration zero, and complete
        endpoint/call diagnostics. Rejected line-search probes cannot become
        returned iterates. Every gradient call also computes the objective.
        Zero-update or parameter-free problems use one metrics-only call.

    Raises:
        ValueError: If budgets or backend inputs are invalid.
        FloatingPointError: If the exact oracle returns a nonfinite value.
    """
    _nonnegative_integer(steps, "steps")
    cap = 1 + LBFGSB_OPTIONS["maxls"] * steps if max_evaluations is None else max_evaluations
    if type(cap) is not int or cap < 1:
        msg = "max_evaluations must be a positive integer."
        raise ValueError(msg)
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    engine = StatevectorBackend(circuit, target)
    theta = np.asarray(initial).copy()
    gradient_calls = 0
    objective_calls = 0
    accepted = 0
    last_theta: NDArray | None = None
    last_gradient: NDArray | None = None
    last_loss = 0.0
    records: list[dict[str, Any]] = []

    def objective(values: NDArray) -> tuple[float, NDArray]:
        nonlocal objective_calls, gradient_calls, last_theta, last_gradient, last_loss
        if last_theta is not None and np.array_equal(values, last_theta):
            assert last_gradient is not None
            return last_loss, last_gradient.copy()
        if objective_calls >= cap:
            raise _EvaluationBudgetError
        gradient, loss, fidelity = engine.gradient(values, maps, estimator="independent", weights=weights)
        objective_calls += 1
        gradient_calls += 1
        if not np.all(np.isfinite(gradient)) or not np.isfinite(loss) or not np.isfinite(fidelity):
            msg = "The L-BFGS-B objective or gradient is nonfinite."
            raise FloatingPointError(msg)
        last_theta, last_gradient, last_loss = values.copy(), gradient.copy(), float(loss)
        return last_loss, last_gradient.copy()

    if steps == 0 or circuit.num_params == 0:
        initial_loss, initial_fidelity = engine.metrics(theta, maps, weights=weights)
        objective_calls = 1
    else:
        initial_loss, _ = objective(theta)
        initial_fidelity = 1 - initial_loss
    best_theta, final_theta = theta.copy(), theta.copy()
    best_fidelity, final_fidelity = float(initial_fidelity), float(initial_fidelity)
    best_iteration = 0

    def record() -> None:
        records.append({
            "iteration": accepted,
            "fidelity": final_fidelity,
            "theta": final_theta.tolist(),
            "cumulative_cpu_seconds": time.process_time() - cpu_start,
            "cumulative_objective_evaluations": objective_calls,
            "cumulative_gradient_evaluations": gradient_calls,
        })

    record()

    def callback(intermediate_result: OptimizeResult) -> None:
        nonlocal accepted, final_theta, final_fidelity, best_theta, best_fidelity, best_iteration
        accepted += 1
        final_theta = np.asarray(intermediate_result.x).copy()
        final_fidelity = 1 - float(intermediate_result.fun)
        if not np.all(np.isfinite(final_theta)) or not np.isfinite(final_fidelity):
            msg = "L-BFGS-B returned a nonfinite accepted iterate."
            raise FloatingPointError(msg)
        if final_fidelity > best_fidelity:
            best_theta, best_fidelity, best_iteration = final_theta.copy(), final_fidelity, accepted
        record()

    result = None
    stop_reason = "zero_steps" if steps == 0 else "no_parameters"
    if steps and circuit.num_params:
        try:
            result = minimize(
                objective,
                theta,
                method="L-BFGS-B",
                jac=True,
                callback=callback,
                options={**LBFGSB_OPTIONS, "maxiter": steps, "maxfun": cap},
            )
            stop_reason = str(result.message)
        except _EvaluationBudgetError:
            stop_reason = "evaluation_budget"
    return best_theta, {
        "optimizer": "scipy.optimize.minimize:L-BFGS-B",
        "scipy_version": version("scipy"),
        "options": {**LBFGSB_OPTIONS, "maxiter": steps, "maxfun": cap},
        "estimator": "independent",
        "retention": "best_accepted_including_initial",
        "initial_theta": theta.tolist(),
        "initial_fidelity": float(initial_fidelity),
        "final_theta": final_theta.tolist(),
        "final_fidelity": final_fidelity,
        "best_theta": best_theta.tolist(),
        "best_fidelity": best_fidelity,
        "best_iteration": best_iteration,
        "returned_fidelity": best_fidelity,
        "steps": accepted,
        "requested_steps": steps,
        "objective_evaluations": objective_calls,
        "gradient_evaluations": gradient_calls,
        "standalone_objective_evaluations": objective_calls - gradient_calls,
        "nonaccepted_probe_evaluations": max(0, objective_calls - 1 - accepted),
        "stop_reason": stop_reason,
        "scipy_success": None if result is None else bool(result.success),
        "scipy_status": None if result is None else int(result.status),
        "scipy_function_evaluations": None if result is None else int(result.nfev),
        "records": records,
        "cpu_seconds": time.process_time() - cpu_start,
        "wall_seconds": time.perf_counter() - wall_start,
    }


def frozen_training_bank(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    *,
    seed: int,
    hardware: str,
    training_paths: int,
) -> tuple[NoiseMaps, NDArray[np.float64], dict[str, Any]]:
    """Sample once and merge only empirical duplicate no-jump histories.

    Args:
        circuit: Physical primitive circuit with its original noise flags.
        theta: Parameters used to construct the parameter-independent maps.
        target: Normalized target vector.
        seed: Legacy sampler iteration key; no rejection sampling is performed.
        hardware: Exactly ``IBM`` or ``Google`` for the current archived law.
        training_paths: Positive number of unconditional raw histories.

    Returns:
        Compressed maps, empirical weights, and raw event/hash/coverage records.
        ``coverage_ok`` is false for a noisy circuit whose bank has no jumps.
        Parameter-free empty circuits are valid with a single weight-one path.

    Raises:
        ValueError: If hardware, sampling budget, or seed is invalid.
    """
    _nonnegative_integer(seed, "seed")
    if hardware not in {"IBM", "Google"} or type(training_paths) is not int or training_paths < 1:
        msg = "Require IBM or Google and a positive integer training_paths."
        raise ValueError(msg)
    cpu, wall = time.process_time(), time.perf_counter()
    engine = StatevectorBackend(circuit, target)
    raw = engine.sample_noise_maps(
        theta,
        KrotovTJMOptions(num_trajectories=training_paths, trajectory_update="cross", use_crn=True),
        LegacyNoise(hardware),
        iteration=seed,
    )
    maps, weights = engine.compress_no_jump_maps(raw)
    jumps = [sum(bool(noise.operators) for noise in row) for row in raw]
    events = [
        [
            {
                "gate": k,
                "operators": [
                    {"sites": list(sites), "real": matrix.real.tolist(), "imag": matrix.imag.tolist()}
                    for matrix, sites in noise.operators
                ],
            }
            for k, noise in enumerate(row)
            if noise.operators
        ]
        for row in raw
    ]
    return (
        maps,
        weights,
        {
            "counter": seed,
            "sampler_calls": 1,
            "sampled_paths": len(raw),
            "retained_paths": len(maps),
            "jump_paths": sum(count > 0 for count in jumps),
            "no_jump_paths": jumps.count(0),
            "raw_jump_counts": jumps,
            "weights": weights.tolist(),
            "multiplicities": np.rint(weights * len(raw)).astype(int).tolist(),
            "coverage_ok": bool(any(jumps) or not any(gate.noise_enabled for gate in circuit.gates)),
            "events": events,
            "raw_sha256": hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest(),
            "cpu_seconds": time.process_time() - cpu,
            "wall_seconds": time.perf_counter() - wall,
        },
    )


def _inputs(
    target: NDArray[np.complex128],
    depths: tuple[int, ...],
    seed: int,
    hardware: str,
    budgets: tuple[int, ...],
    paths: int,
) -> NDArray[np.complex128]:
    """Validate common controls before creating any candidate or sampling noise.

    Returns:
        A private normalized target vector.

    Raises:
        ValueError: If depths, hardware, seeds, or budgets are invalid.
    """
    vector = validate_target(target)
    if (
        not isinstance(depths, tuple)
        or not depths
        or any(type(depth) is not int or depth < 0 for depth in depths)
        or tuple(sorted(set(depths))) != depths
    ):
        msg = "depths must be a nonempty strictly increasing tuple of nonnegative integers."
        raise ValueError(msg)
    _nonnegative_integer(seed, "seed")
    for budget in budgets:
        _nonnegative_integer(budget, "update budget")
    if seed + 1000 + max(depths) >= 2**32:
        msg = "Initialization seeds must fit the legacy RandomState range."
        raise ValueError(msg)
    if hardware not in {"IBM", "Google"} or type(paths) is not int or paths < 1:
        msg = "Require IBM or Google and a positive integer training_paths."
        raise ValueError(msg)
    return vector


def _skipped_noisy_training() -> dict[str, Any]:
    """Return an explicit failed-coverage endpoint without pretending to train."""
    return {
        "status": "no_jump_coverage",
        "steps": 0,
        "objective_evaluations": 0,
        "gradient_evaluations": 0,
        "cpu_seconds": 0.0,
        "returned_fidelity": None,
    }


def fixed_vqsp_candidates(
    target: NDArray[np.complex128],
    *,
    depths: tuple[int, ...],
    seed: int,
    hardware: str,
    construction_steps: int = 200,
    noisy_steps: int = 200,
    training_paths: int = 100,
) -> list[Candidate]:
    """Train independent fixed-depth circuits with noiseless then noisy L-BFGS-B.

    Args:
        target: Normalized target state vector.
        depths: Strictly increasing, nonnegative BMPD depths; zero is product-only.
        seed: Shared replication seed; each depth independently uses seed+1000+d.
        hardware: Current ``IBM`` or ``Google`` Pauli-jump profile.
        construction_steps: Maximum noiseless SciPy iterations per candidate.
        noisy_steps: Maximum noisy SciPy iterations per candidate.
        training_paths: Unconditional frozen histories before empirical compression.

    Returns:
        Bound physical circuits with JSON provenance and charged work. Every
        depth initializes all angles from N(0,0.05**2), with no parent warm start.
        Missing jump coverage returns an explicit ineligible status and the
        constructed circuit; it does not trigger resampling or noisy updates.
    """
    vector = _inputs(target, depths, seed, hardware, (construction_steps, noisy_steps), training_paths)
    start = time.process_time()
    cumulative_construction = cumulative_noisy = 0.0
    candidates = []
    for depth in depths:
        cpu = time.process_time()
        circuit = brickwall_for(vector, depth)
        initial = extend(None, circuit.num_params, seed + 1000 + depth, 0.05)
        built, construction = optimize_lbfgsb(circuit, initial, vector, steps=construction_steps)
        maps, weights, bank = frozen_training_bank(
            circuit, built, vector, seed=seed + 5000, hardware=hardware, training_paths=training_paths
        )
        if bank["coverage_ok"]:
            theta, noisy = optimize_lbfgsb(circuit, built, vector, steps=noisy_steps, maps=maps, weights=weights)
        else:
            theta, noisy = built.copy(), _skipped_noisy_training()
        physical = from_parameterized(circuit, theta)
        cumulative_construction += construction["cpu_seconds"]
        cumulative_noisy += bank["cpu_seconds"] + noisy["cpu_seconds"]
        metadata = {
            "method": "fixed_vqsp",
            "candidate_id": f"depth_{depth:02d}",
            "depth": depth,
            "seed": seed,
            "hardware": hardware,
            "status": "complete" if bank["coverage_ok"] else "no_jump_coverage",
            "deterministic": False,
            "construction_lineage": [depth],
            "parent_depth": None,
            "initialization": "independent Normal(0,0.05**2) at every depth",
            "initialization_scale": 0.05,
            "initialization_seed": seed + 1000 + depth,
            "parameterized_circuit": circuit_record(circuit),
            "initial_theta": initial.tolist(),
            "construction_theta": built.tolist(),
            "theta": theta.tolist(),
            "construction": construction,
            "training_bank": bank,
            "noisy_training": noisy,
            "construction_cpu_seconds": construction["cpu_seconds"],
            "noisy_cpu_seconds": bank["cpu_seconds"] + noisy["cpu_seconds"],
            "cumulative_construction_cpu_seconds": cumulative_construction,
            "cumulative_noisy_cpu_seconds": cumulative_noisy,
            "cpu_seconds": time.process_time() - cpu,
            "cumulative_cpu_seconds": time.process_time() - start,
        }
        candidates.append(Candidate(physical, metadata))
    return candidates


def _legacy_refinement(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    bank: dict[str, Any],
    maps: NoiseMaps,
    weights: NDArray[np.float64],
    *,
    steps: int,
    online: int,
    hardware: str,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Reuse unchanged noisy hybrid training after a charged coverage precheck.

    Returns:
        Its original final iterate and augmented work diagnostics. The legacy
        routine regenerates the identical bank because it cannot accept maps;
        this deterministic replay is charged and its event hash is verified.

    Raises:
        RuntimeError: If deterministic replay unexpectedly changes the bank.
    """
    cpu, wall = time.process_time(), time.perf_counter()
    initial_fidelity = StatevectorBackend(circuit, target).metrics(theta, maps, weights=weights)[1]
    final, trace = train(
        circuit,
        theta,
        target,
        steps=steps,
        online=online,
        rate=0.2,
        seed=bank["counter"],
        noisy=True,
        crn=True,
        hardware=hardware,
        num_trajectories=bank["sampled_paths"],
        compress_no_jump=True,
        record_noise_banks=True,
        require_jump_coverage=True,
    )
    if trace["noise_banks"][0]["raw_sha256"] != bank["raw_sha256"]:
        msg = "The unchanged legacy routine replayed a different training bank."
        raise RuntimeError(msg)
    gradients = trace["steps"] - trace["online_steps"]
    trace.update({
        "optimizer": "unchanged_legacy_online_then_batch_cross",
        "retention": "final_unchanged_legacy_policy",
        "initial_fidelity": initial_fidelity,
        "returned_fidelity": trace["fidelity_trace"][-1] if trace["fidelity_trace"] else initial_fidelity,
        "objective_evaluations": 1 + trace["steps"] + gradients,
        "standalone_objective_evaluations": 1 + trace["steps"],
        "gradient_evaluations": gradients,
        "online_sweeps": trace["online_steps"],
        "sampler_calls": 1,
        "coverage_precheck_replayed": True,
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall,
    })
    return final, trace


def layerwise_candidates(
    target: NDArray[np.complex128],
    *,
    depths: tuple[int, ...],
    seed: int,
    hardware: str,
    construction_steps: int = 200,
    online_steps: int = 100,
    noisy_steps: int = 200,
    noisy_online_steps: int = 100,
    training_paths: int = 100,
) -> list[Candidate]:
    """Grow unpruned BMPD using current construction and noisy fine-tuning.

    Args:
        target: Normalized target vector.
        depths: Exactly the contiguous positive depths (1, ..., maximum).
        seed: Current replication seed and unchanged construction/noise schedule.
        hardware: Current ``IBM`` or ``Google`` Pauli-jump profile.
        construction_steps: Total attempted construction updates per depth.
        online_steps: Initial online construction sweeps, followed by L-BFGS.
        noisy_steps: Unchanged noisy fine-tuning updates per candidate.
        noisy_online_steps: Initial online noisy sweeps before cross batch updates.
        training_paths: Unconditional fixed histories before empirical compression.

    Returns:
        Every depth candidate, including explicit failed-coverage records. Each
        extension inherits the best unpruned noiseless parent, never the noisy
        endpoint. This control does no pruning, stopping, or candidate selection.

    Raises:
        ValueError: If depths omit a required construction ancestor.
    """
    vector = _inputs(
        target,
        depths,
        seed,
        hardware,
        (construction_steps, online_steps, noisy_steps, noisy_online_steps),
        training_paths,
    )
    if depths != tuple(range(1, max(depths) + 1)):
        msg = "Layerwise depths must be contiguous from one through the maximum."
        raise ValueError(msg)
    start = time.process_time()
    cumulative_construction = cumulative_noisy = 0.0
    previous = np.empty(0, dtype=float)
    candidates = []
    for depth in depths:
        cpu = time.process_time()
        circuit = brickwall_for(vector, depth)
        initial = extend(previous, circuit.num_params, seed + 1000 + depth, 0.05 if depth == 1 else 0.001)
        built, construction = train_noiseless(
            circuit, initial, vector, policy="lbfgs", steps=construction_steps, online=online_steps, rate=0.7, stop=2.0
        )
        previous = built.copy()
        maps, weights, bank = frozen_training_bank(
            circuit, built, vector, seed=seed + 5000, hardware=hardware, training_paths=training_paths
        )
        if bank["coverage_ok"]:
            theta, noisy = _legacy_refinement(
                circuit,
                built,
                vector,
                bank,
                maps,
                weights,
                steps=noisy_steps,
                online=noisy_online_steps,
                hardware=hardware,
            )
        else:
            theta, noisy = built.copy(), _skipped_noisy_training()
        physical = from_parameterized(circuit, theta)
        cumulative_construction += construction["cpu_seconds"]
        cumulative_noisy += bank["cpu_seconds"] + noisy["cpu_seconds"]
        metadata = {
            "method": "layerwise",
            "candidate_id": f"depth_{depth:02d}",
            "depth": depth,
            "seed": seed,
            "hardware": hardware,
            "status": "complete" if bank["coverage_ok"] else "no_jump_coverage",
            "deterministic": False,
            "construction_lineage": list(range(1, depth + 1)),
            "parent_depth": depth - 1 if depth > 1 else None,
            "initialization": "depth1 Normal(0,0.05**2); later best noiseless parent plus Normal(0,0.001**2)",
            "initialization_scale": 0.05 if depth == 1 else 0.001,
            "initialization_seed": seed + 1000 + depth,
            "parameterized_circuit": circuit_record(circuit),
            "initial_theta": initial.tolist(),
            "construction_theta": built.tolist(),
            "theta": theta.tolist(),
            "construction": construction,
            "training_bank": bank,
            "noisy_training": noisy,
            "construction_cpu_seconds": construction["cpu_seconds"],
            "noisy_cpu_seconds": bank["cpu_seconds"] + noisy["cpu_seconds"],
            "cumulative_construction_cpu_seconds": cumulative_construction,
            "cumulative_noisy_cpu_seconds": cumulative_noisy,
            "cpu_seconds": time.process_time() - cpu,
            "cumulative_cpu_seconds": time.process_time() - start,
        }
        candidates.append(Candidate(physical, metadata))
    return candidates
