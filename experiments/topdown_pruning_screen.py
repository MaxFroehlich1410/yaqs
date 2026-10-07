# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compare standalone pruning arms on identical authenticated q6 checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.state_preparation.pilot_components import delete_gates
from experiments import legacy_topdown_dense as legacy
from experiments import topdown_backend_support as support
from experiments import topdown_batch_refinement_policies as policies
from experiments import topdown_depth_search as depth
from experiments import topdown_unpruned_finetuning as parents
from mqt.yaqs.optimization import StatevectorBackend

# Reuse the tested local contraction, with independent density and pair adjoints.
from mqt.yaqs.optimization.statevector import _apply  # noqa: PLC2701 -- Reuse the tested numerical kernel.

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

VERSION = "topdown.pruning_screen.v1"
METHODS = (
    "B",
    "sequential_ideal",
    "magnitude",
    "sequential_noisy",
    "pair",
    "resource_weighted",
    "ideal_repair",
    "B_0025",
    "B_005",
    "B_015",
    "B_02",
)
CUTOFFS = {"B": 0.01, "B_0025": 0.0025, "B_005": 0.005, "B_015": 0.015, "B_02": 0.02}
PROTOCOL: dict = {
    "version": VERSION,
    "methods": list(METHODS),
    "primary_fidelity": "ideal fidelity at end of pruning; ideal_repair includes its own noiseless repairs",
    "B": "simultaneously delete all signed exact ideal damages strictly below 0.01",
    "sequential_ideal": "recompute signed ideal damages after each minimum-damage deletion; cutoff 0.01",
    "magnitude": "simultaneously delete effective rotation angles with wrapped magnitude strictly below 0.2 radians",
    "sequential_noisy": "exact IBM damage; delete gate and noise sequentially only if damage < -1e-12",
    "pair": "sequential joint ideal damage for same-support pairs at distance <=3; cutoff 0.01 per pair",
    "resource_weighted": "sequential ideal damage / fixed cost (1 for 1q, 2 for 2q); normalized cutoff 0.01",
    "ideal_repair": "sequential ideal cutoff 0.01; after each deletion, ten safeguarded noiseless L-BFGS updates",
    "B_thresholds": CUTOFFS,
    "repair": {"policy": "lbfgs", "steps": 10, "online": 0, "rate": 0.7, "stop": 2.0},
    "composition": "none; every method starts from the same full original circuit and angles",
    "new_construction_updates": 0,
    "noisy_fine_tuning_updates": 0,
    "hardware": "IBM logical noise for secondary exact assessment only",
    "resources": "logical total/2q counts and fixed-reference compiled 2q counts/depth; cost weights are estimates",
    "ties": "original order; pair candidates lexicographic; no target-dependent hyperparameters",
    "selection": "report every frozen method; no target-wise hybrid or fallback method",
}


def conjugate(rho: NDArray, operator: NDArray, sites: tuple[int, ...], qubits: int) -> NDArray:
    """Apply U rho U-dagger using local contractions.

    Returns:
        The new density matrix or adjoint effect operator.
    """
    left = _apply(rho.T, operator, sites, qubits).T
    return _apply(left, operator.conj(), sites, qubits)


@cache
def outcomes(hardware: str, sites: tuple[int, ...]) -> tuple:
    """Cache state-independent Hermitian Pauli outcomes at one gate window.

    Returns:
        Identity probability and nonidentity probabilities/operators.
    """
    probabilities, operators = depth.noise_outcomes(hardware, sites)
    return probabilities, operators


def noise(rho: NDArray, sites: tuple[int, ...], qubits: int, hardware: str) -> NDArray:
    """Apply the exact self-adjoint Pauli channel used by the incumbent.

    Returns:
        A fresh density matrix or adjoint effect.
    """
    probabilities, operators = outcomes(hardware, sites)
    result = probabilities[0] * rho
    for probability, (operator, acted) in zip(probabilities[1:], operators, strict=True):
        result += probability * conjugate(rho, operator, acted, qubits)
    return result


def noisy_deletions(circuit: ParameterizedCircuit, theta: NDArray, target: NDArray) -> tuple[NDArray, float]:
    """Score deletion of both gate and noise using exact forward/backward density tapes.

    Returns:
        Counterfactual fidelities for every deletion and original noisy fidelity.

    Raises:
        ValueError: If the exact-density small-system limit is exceeded.
    """
    if circuit.num_qubits > 6:
        msg = "Exact noisy deletion scoring is limited to six qubits."
        raise ValueError(msg)
    rho = np.zeros((2**circuit.num_qubits,) * 2, dtype=np.complex128)
    rho[0, 0] = 1
    tape = []
    for gate in circuit.gates:
        unitary, sites = circuit.gate_matrix(gate, theta)
        tape.append((rho, unitary, sites, gate.noise_enabled))
        rho = conjugate(rho, unitary, sites, circuit.num_qubits)
        if gate.noise_enabled:
            rho = noise(rho, sites, circuit.num_qubits, "IBM")
    effect = np.outer(target, target.conj())
    baseline = float(np.vdot(effect, rho).real)
    removed = np.empty(len(tape))
    for index in range(len(tape) - 1, -1, -1):
        prefix, unitary, sites, enabled = tape[index]
        removed[index] = np.vdot(effect, prefix).real
        if enabled:
            effect = noise(effect, sites, circuit.num_qubits, "IBM")
        effect = conjugate(effect, unitary.conj().T, sites, circuit.num_qubits)
    return removed, baseline


def pair_deletions(circuit: ParameterizedCircuit, theta: NDArray, target: NDArray) -> tuple[list, NDArray, float]:
    """Evaluate same-support nearby pairs with exact shared ideal adjoints.

    Returns:
        Candidate gate-index pairs, removed fidelities and baseline ideal fidelity.
    """
    state = np.zeros((1, 2**circuit.num_qubits), dtype=np.complex128)
    state[0, 0] = 1
    tape = []
    for gate in circuit.gates:
        unitary, sites = circuit.gate_matrix(gate, theta)
        tape.append((state, unitary, sites))
        state = _apply(state, unitary, sites, circuit.num_qubits)
    baseline = float(abs(np.vdot(target, state[0])) ** 2)
    suffixes: list = [None] * len(tape)
    suffix = target[None, :]
    for index in range(len(tape) - 1, -1, -1):
        suffixes[index] = suffix
        _, unitary, sites = tape[index]
        suffix = _apply(suffix, unitary.conj().T, sites, circuit.num_qubits)
    pairs, scores = [], []
    for first, gate in enumerate(circuit.gates):
        for last in range(first + 1, min(first + 4, len(tape))):
            if tuple(sorted(gate.sites)) != tuple(sorted(circuit.gates[last].sites)):
                continue
            state = tape[first][0]
            for middle in range(first + 1, last):
                _, unitary, sites = tape[middle]
                state = _apply(state, unitary, sites, circuit.num_qubits)
            pairs.append([first, last])
            scores.append(float(abs(np.vdot(suffixes[last], state)) ** 2))
    return pairs, np.asarray(scores), baseline


def prune(circuit: ParameterizedCircuit, theta: NDArray, target: NDArray, method: str) -> tuple:
    """Execute exactly one standalone strategy, preserving the caller's inputs.

    Returns:
        Reduced circuit, retained/repaired parameters and full decision trace.

    Raises:
        ValueError: If method or circuit parameters do not satisfy the frozen panel.
    """
    if method not in METHODS:
        msg = "Unknown standalone pruning method."
        raise ValueError(msg)
    indices = [gate.param_index for gate in circuit.gates]
    if (
        any(index is None for index in indices)
        or sorted(index for index in indices if index is not None) != list(range(circuit.num_params))
        or theta.shape != (circuit.num_params,)
        or not np.all(np.isfinite(theta))
    ):
        msg = "Require one independent scalar parameter per gate."
        raise ValueError(msg)
    circuit = support.restore_circuit(support.circuit_record(circuit))
    theta = theta.copy()
    origins = list(range(len(circuit.gates)))
    trace: dict = {"method": method, "iterations": [], "scored_candidates": 0, "repair_updates": 0}
    while circuit.gates:
        pairs = []
        engine = StatevectorBackend(circuit, target)
        if method == "magnitude":
            scores = np.array([
                abs((gate.angle_offset + gate.angle_scale * theta[gate.param_index] + np.pi) % (2 * np.pi) - np.pi)
                for gate in circuit.gates
            ])
            baseline = engine.metrics(theta)[1]
            selected = np.flatnonzero(scores < 0.2).tolist()
        else:
            if method == "sequential_noisy":
                removed, baseline = noisy_deletions(circuit, theta, target)
                cutoff = -1e-12
            elif method == "pair":
                pairs, removed, baseline = pair_deletions(circuit, theta, target)
                cutoff = 0.01
            else:
                removed, baseline = engine.deletion_fidelities(theta)
                cutoff = CUTOFFS.get(method, 0.01)
            scores = baseline - removed
            if method == "resource_weighted":
                scores /= np.array([1 if len(gate.sites) == 1 else 2 for gate in circuit.gates])
            eligible = np.flatnonzero(scores < cutoff)
            if method in CUTOFFS:
                selected = eligible.tolist()
            elif len(eligible):
                index = int(np.argmin(scores))
                selected = pairs[index] if method == "pair" else [index]
            else:
                selected = []
        trace["scored_candidates"] += len(scores)
        entry = {
            "score_baseline": baseline,
            "scores": scores.tolist(),
            "removed_origins": [origins[index] for index in selected],
        }
        if method == "pair":
            entry["pair_origins"] = [[origins[i], origins[j]] for i, j in pairs]
        trace["iterations"].append(entry)
        if not selected:
            break
        circuit, theta = delete_gates(circuit, theta, selected)
        selected_set = set(selected)
        origins = [origin for index, origin in enumerate(origins) if index not in selected_set]
        entry["ideal_after_deletion"] = StatevectorBackend(circuit, target).metrics(theta)[1]
        if method == "ideal_repair" and circuit.gates:
            theta, repair = policies.train_noiseless(circuit, theta, target, **PROTOCOL["repair"])
            entry["repair"] = repair
            trace["repair_updates"] += repair["steps"]
        entry["ideal_after_repair"] = StatevectorBackend(circuit, target).metrics(theta)[1]
        if method in CUTOFFS or method == "magnitude":
            break
    trace["remaining_origins"] = origins
    return circuit, theta, trace


def prepare(parent: Path, output: Path) -> dict:
    """Freeze all 27 shared inputs and the independent method panel.

    Returns:
        A plan containing 27 checkpoints and 297 independent jobs.

    Raises:
        ValueError: If output exists or the authenticated parent panel is incomplete.
    """
    if output.exists():
        msg = "Prepare requires a fresh output directory."
        raise ValueError(msg)
    old = json.loads((parent / "plan.json").read_text())
    cases = {(job["target_id"], job["seed"]) for job in old["jobs"]}
    if len(old["jobs"]) != 27 or len(cases) != 27 or len({case[0] for case in cases}) != 9:
        msg = "Require all 27 original checkpoint cases."
        raise ValueError(msg)
    for job in old["jobs"]:
        folder = parent / "inputs" / job["job_id"]
        parents.authenticate_input(folder)
        if support.digest(folder / "input.json") != old["inputs"][job["job_id"]]:
            msg = "Parent input receipt changed."
            raise ValueError(msg)
    shutil.copytree(parent / "inputs", output / "inputs")
    plan = {"protocol": PROTOCOL, "jobs": old["jobs"], "inputs": old["inputs"], "comparisons": 27 * len(METHODS)}
    support.write_json(output / "plan.json", plan)
    return plan


def run_method(folder: Path, input_folder: Path, method: str) -> dict:
    """Apply one method from its unpruned parent and independently assess its output.

    Returns:
        Authenticated primary ideal score, secondary noisy score, resources and cost.

    Raises:
        ValueError: If cached provenance changes, replay fails or B is not reproduced.
    """
    receipt, built, arrays, target = parents.authenticate_input(input_folder)
    identity = {"protocol": PROTOCOL, "method": method, "input_sha256": support.digest(input_folder / "input.json")}
    manifest = folder / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        msg = "Changed method or input; use a fresh output directory."
        raise ValueError(msg)
    support.write_json(manifest, identity)
    final = support.load_checkpoint(folder / "final")
    if final is not None:
        parents.required(folder / "pruning")
        if final[0]["pruning_seal_sha256"] != support.digest(folder / "pruning/complete.json"):
            msg = "Pruning result changed after assessment."
            raise ValueError(msg)
        return final[0]
    saved = support.load_checkpoint(folder / "pruning")
    if saved is None:
        cpu, wall = time.process_time(), time.perf_counter()
        circuit = support.restore_circuit(built["circuit"])
        reduced, theta, trace = prune(circuit, arrays["theta"], target, method)
        support.save_checkpoint(
            folder / "pruning",
            {
                "circuit": support.circuit_record(reduced),
                "trace": trace,
                "identity": identity,
                "cpu_seconds": time.process_time() - cpu,
                "wall_seconds": time.perf_counter() - wall,
            },
            {"theta": theta},
        )
        saved = parents.required(folder / "pruning")
    if saved[0]["identity"] != identity:
        msg = "Cached pruning refers to different inputs."
        raise ValueError(msg)
    circuit = support.restore_circuit(saved[0]["circuit"])
    theta = saved[1]["theta"]
    cpu = time.process_time()
    ideal = legacy.matrix_score(circuit, theta, target, "IBM", "noiseless")
    noisy = legacy.matrix_score(circuit, theta, target, "IBM", "tjm")
    replay_error = abs(StatevectorBackend(circuit, target).metrics(theta)[1] - ideal["fidelity"])
    if replay_error > 1e-10:
        msg = "Independent ideal fidelity replay failed."
        raise ValueError(msg)
    if method == "B":
        original, original_arrays = parents.required(input_folder / "parent_pruning")
        if original["circuit"] != saved[0]["circuit"] or not np.array_equal(original_arrays["theta"], theta):
            msg = "B does not reproduce the original circuit and parameters."
            raise ValueError(msg)
    result = {
        "version": VERSION,
        "target_id": receipt["target_id"],
        "seed": receipt["seed"],
        "selected_depth": receipt["selected_depth"],
        "method": method,
        "status": "complete",
        "input_sha256": identity["input_sha256"],
        "pruning_seal_sha256": support.digest(folder / "pruning/complete.json"),
        "ideal_pruned": ideal,
        "noisy_pruned": noisy,
        "ideal_replay_error": replay_error,
        "logical_total_gates": len(circuit.gates),
        "pruning_cpu_seconds": saved[0]["cpu_seconds"],
        "pruning_wall_seconds": saved[0]["wall_seconds"],
        "repair_updates": saved[0]["trace"]["repair_updates"],
        "scored_candidates": saved[0]["trace"]["scored_candidates"],
        "new_construction_updates": 0,
        "noisy_fine_tuning_updates": 0,
        "assessment_cpu_seconds": time.process_time() - cpu,
    }
    cpu = time.process_time()
    result["resources"] = support.compile_resources(circuit, theta)
    result["compilation_cpu_seconds"] = time.process_time() - cpu
    return support.save_checkpoint(folder / "final", result, {})


def main() -> None:
    """Prepare the panel or execute its independent methods for one checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "run"), required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=(*METHODS, "all"), default="all")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.action == "prepare" and args.parent is not None:
            result = prepare(args.parent, args.output)
            print(json.dumps({"workers": len(result["jobs"]), "comparisons": result["comparisons"]}))
        elif args.action == "run" and args.input is not None:
            for method in METHODS if args.method == "all" else (args.method,):
                result = run_method(args.output / method, args.input, method)
                print(
                    json.dumps({
                        "method": method,
                        "status": result["status"],
                        "ideal": result["ideal_pruned"]["fidelity"],
                    }),
                    flush=True,
                )
        else:
            parser.error("Prepare requires --parent; run requires --input.")


if __name__ == "__main__":
    main()
