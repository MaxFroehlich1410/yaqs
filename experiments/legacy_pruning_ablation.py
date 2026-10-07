# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Four controlled pruning variants from the frozen eighteen q6 checkpoints.

A retains the literal saliency cutoff; B changes only its score to exact
noiseless gate-removal damage; C makes removal sequential; D adds ten noiseless
repair updates per deletion. All endpoints retain the existing 100-history
weighted cross-trajectory fine-tuning and independent Qiskit density scoring.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap

from benchmarks.state_preparation.execution import completed_jobs, parse_workers, resolve_workers
from benchmarks.state_preparation.pilot_components import delete_gates
from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH, load_target_collection
from experiments.legacy_topdown_dense import PRIMITIVES, LegacyNoise, prune, train
from experiments.legacy_topdown_q6 import REPOSITORY, SEEDS, Checkpoints, write_json
from experiments.legacy_topdown_weighted import file_hash
from mqt.yaqs.optimization import KrotovTJMOptions, StatevectorBackend

# Share the tested local application kernel; scoring itself is independently
# checked against exhaustive Qiskit deletions, including reversed site order.
from mqt.yaqs.optimization.statevector import _apply  # noqa: PLC2701

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from qiskit.circuit import CircuitInstruction

    from mqt.yaqs.optimization import ParameterizedCircuit

VARIANTS = ("A", "B", "C", "D")
HARDWARE = ("IBM", "Google")
SOURCE = REPOSITORY / "output/legacy-topdown-q6-weighted100-20260914"
PROTOCOL: dict = {
    "version": 1,
    "variants": {
        "A": "absolute theta times loss gradient; simultaneous deletion",
        "B": "signed exact noiseless deletion damage; simultaneous deletion",
        "C": "signed exact noiseless deletion damage; sequential rescoring",
        "D": "C plus ten noiseless hybrid repair updates after each deletion",
    },
    "cutoff": 0.01,
    "deletion_condition": "strictly less than cutoff; stable original-order tie breaking",
    "stop": "no remaining candidate below cutoff, or empty circuit; no cumulative fidelity guard",
    "repair": {"steps": 10, "online": 5, "rate": 0.7, "decay": 0.0, "seed": 42},
    "fine_tuning": {
        "steps": 200,
        "online": 100,
        "rate": 0.2,
        "decay": 0.0,
        "noisy": True,
        "crn": True,
        "num_trajectories": 100,
        "compress_no_jump": True,
        "record_noise_banks": True,
        "require_jump_coverage": True,
    },
    "fine_tuning_seed": "optimizer seed + 5000; same formula, not aligned gates across different circuits",
    "noise": "unchanged LegacyNoise IBM/Google at-most-one-Pauli-jump gate windows",
    "test": "fixed final iterate; exact Qiskit density matrix under sampler law; no test-based selection",
    "compiler": {
        "basis_gates": ["rz", "sx", "x", "cx"],
        "optimization_level": 1,
        "seed_transpiler": 1729,
        "approximation_degree": 1.0,
    },
    "compiler_topology": "bidirectional six-qubit line; identity initial layout; common reference, not a device model",
}


def deletion_fidelities(
    circuit: ParameterizedCircuit, theta: NDArray, target: NDArray
) -> tuple[NDArray[np.float64], float]:
    """Score complete single-gate removals with one forward/backward vector tape.

    The suffix target adjoint excludes the candidate gate; its overlap with
    the prefix gives the exact removed-circuit amplitude. This is noiseless
    scoring, supports fixed/shared scalar gates, and stores O(L * 2**n) entries.
    """
    engine = StatevectorBackend(circuit, target)
    if theta.shape != (circuit.num_params,) or not np.isrealobj(theta) or not np.all(np.isfinite(theta)):
        msg = "Expected finite real parameters matching the circuit."
        raise ValueError(msg)
    state = engine.initial[None, :]
    tape = []
    for gate in circuit.gates:
        unitary, sites = circuit.gate_matrix(gate, theta)
        tape.append((state, unitary, sites))
        state = _apply(state, unitary, sites, circuit.num_qubits)
    fidelity = float(abs(np.vdot(target, state[0])) ** 2)
    suffix = engine.target[None, :]
    removed = np.empty(len(tape))
    for index in range(len(tape) - 1, -1, -1):
        prefix, unitary, sites = tape[index]
        removed[index] = abs(np.vdot(suffix, prefix)) ** 2
        suffix = _apply(suffix, unitary.conj().T, sites, circuit.num_qubits)
    return removed, fidelity


def prune_variant(
    circuit: ParameterizedCircuit, theta: NDArray, target: NDArray, variant: str
) -> tuple[ParameterizedCircuit, NDArray, dict]:
    """Apply one frozen pruning policy, recording every score and repair decision."""
    if variant not in VARIANTS:
        msg = "Unknown pruning variant."
        raise ValueError(msg)
    indices = [gate.param_index for gate in circuit.gates]
    if any(index is None for index in indices) or sorted(indices) != list(range(circuit.num_params)):
        msg = "This ablation requires one independent parameter per gate."
        raise ValueError(msg)
    start, cpu = time.perf_counter(), time.process_time()
    theta = theta.copy()
    origins = list(range(len(circuit.gates)))
    trace: dict = {
        "variant": variant,
        "initial_gates": len(origins),
        "iterations": [],
        "score_calls": 0,
        "scored_candidates": 0,
        "score_local_vector_applications": 0,
        "repair_updates": 0,
        "repair_parameter_updates": 0,
        "repair_local_vector_applications_including_metrics": 0,
    }
    while circuit.gates:
        if variant == "A":
            gradient, _, baseline = StatevectorBackend(circuit, target).gradient(theta)
            parameter_scores = np.abs(theta * gradient)
            scores = np.array([parameter_scores[gate.param_index] for gate in circuit.gates])
            trace["score_local_vector_applications"] += 3 * len(circuit.gates)
        else:
            removed_fidelities, baseline = deletion_fidelities(circuit, theta, target)
            scores = baseline - removed_fidelities
            trace["score_local_vector_applications"] += 2 * len(circuit.gates)
        trace["score_calls"] += 1
        trace["scored_candidates"] += len(scores)
        eligible = np.flatnonzero(scores < PROTOCOL["cutoff"])
        selected = eligible.tolist() if variant in {"A", "B"} else [int(np.argmin(scores))] if len(eligible) else []
        entry = {
            "origins": origins.copy(),
            "fidelity_before": baseline,
            "scores": scores.tolist(),
            "removed_origins": [origins[index] for index in selected],
        }
        trace["iterations"].append(entry)
        if not selected:
            break
        if variant == "A":
            circuit, theta = prune(circuit, theta, parameter_scores)
        else:
            circuit, theta = delete_gates(circuit, theta, selected)
        origins = [origin for index, origin in enumerate(origins) if index not in set(selected)]
        entry["fidelity_after_deletion"] = StatevectorBackend(circuit, target).metrics(theta)[1]
        if variant == "D" and circuit.gates:
            theta, repair = train(circuit, theta, target, **PROTOCOL["repair"])
            entry["repair"] = repair
            trace["repair_updates"] += repair["steps"]
            trace["repair_parameter_updates"] += repair["steps"] * circuit.num_params
            # Five online sweeps: 6L applications each, including metric replay.
            # Five batch updates: 4L each, including metric replay.
            trace["repair_local_vector_applications_including_metrics"] += 50 * len(circuit.gates)
        entry["theta_after_repair"] = theta.tolist()
        entry["fidelity_after_repair"] = StatevectorBackend(circuit, target).metrics(theta)[1]
        if variant in {"A", "B"}:
            break
    trace.update({
        "remaining_origins": origins,
        "final_gates": len(circuit.gates),
        "final_noiseless_fidelity": StatevectorBackend(circuit, target).metrics(theta)[1],
        "wall_seconds": time.perf_counter() - start,
        "cpu_seconds": time.process_time() - cpu,
    })
    return circuit, theta, trace


def compile_resources(circuit: ParameterizedCircuit, theta: NDArray) -> tuple[QuantumCircuit, dict]:
    """Compile a numerical circuit to one fixed CX reference and verify its unitary."""
    source = QuantumCircuit(circuit.num_qubits)
    for gate in circuit.gates:
        angle = gate.angle_offset + (gate.angle_scale * theta[gate.param_index] if gate.param_index is not None else 0)
        source.append(PRIMITIVES[gate.name](angle), list(gate.sites))
    start = time.perf_counter()
    compiled = transpile(
        source,
        coupling_map=CouplingMap.from_line(circuit.num_qubits),
        initial_layout=list(range(circuit.num_qubits)),
        **PROTOCOL["compiler"],
    )
    expected, actual = Operator(source).data, Operator.from_circuit(compiled).data
    overlap = np.vdot(expected, actual)
    phase = overlap / abs(overlap)
    residual = float(np.max(np.abs(actual - phase * expected)))
    assert residual < 1e-10

    def two(item: CircuitInstruction) -> bool:
        """Select only two-qubit operations for dependency-aware depth."""
        return item.operation.num_qubits == 2

    return compiled, {
        "logical_one_qubit": sum(len(g.sites) == 1 for g in circuit.gates),
        "logical_two_qubit": sum(len(g.sites) == 2 for g in circuit.gates),
        "logical_two_qubit_depth": source.depth(two),
        "compiled_one_qubit": sum(item.operation.num_qubits == 1 for item in compiled.data),
        "compiled_two_qubit": sum(two(item) for item in compiled.data),
        "compiled_two_qubit_depth": compiled.depth(two),
        "compiled_total_depth": compiled.depth(),
        "compiled_operations": dict(compiled.count_ops()),
        "logical_unitary_max_error_up_to_global_phase": residual,
        "wall_seconds": time.perf_counter() - start,
    }


def verify_baseline(store: Checkpoints, source: Checkpoints, hardware: str, result: dict) -> dict:
    """Check A reproduces the prior cutoff, frozen maps and final parameters."""
    circuit, theta, before = store.load("A_initial")
    previous, expected, saved = source.load("cutoff_initial")
    assert circuit.num_params == previous.num_params
    assert before["circuit"] == saved["circuit"]
    np.testing.assert_array_equal(theta, expected)
    if result["status"] == "no_jump_bank":
        assert (source.root / f"cutoff_{hardware}_100.failure.json").exists()
        return {"initial_parameters_exact": True, "coverage_failure_reproduced": True}
    _, final, current = store.load("A_" + hardware)
    _, expected_final, old = source.load("cutoff_" + hardware + "_100")
    np.testing.assert_allclose(final, expected_final, atol=1e-12, rtol=0)
    assert current["training"]["noise_banks"] == old["training"]["noise_banks"]
    return {
        "initial_parameters_exact": True,
        "final_parameter_max_error": float(np.max(np.abs(final - expected_final))) if len(final) else 0.0,
        "noise_banks_exact": True,
    }


def fine_tune(store: Checkpoints, variant: str, hardware: str, seed: int) -> dict:
    """Run the unchanged weighted fine-tuning, preserving coverage failures."""
    circuit, initial, before = store.load(variant + "_initial")
    name = variant + "_" + hardware
    result = {
        "variant": variant,
        "hardware": hardware,
        "gates": len(circuit.gates),
        "fidelity_after_pruning": before["noiseless"]["fidelity"],
        "noisy_fidelity_before_fine_tuning": before["noise"][hardware]["tjm"]["fidelity"],
    }
    if not store.exists(name) and not (store.root / (name + ".failure.json")).exists():
        print(store.root.name, name, "fine-tuning", len(circuit.gates), "gates", flush=True)
        cpu = time.process_time()
        try:
            theta, trace = train(
                circuit, initial, store.target, hardware=hardware, seed=seed + 5000, **PROTOCOL["fine_tuning"]
            )
        except RuntimeError as error:
            if "sampled bank has no jumps" not in str(error):
                raise
            # Reproduce the same rejected-by-coverage bank for the audit; never
            # draw an alternative seed or condition the sample on a jump.
            maps = StatevectorBackend(circuit, store.target).sample_noise_maps(
                initial, KrotovTJMOptions(num_trajectories=100), LegacyNoise(hardware), iteration=seed + 5000
            )
            assert len(maps) == 100
            assert not any(m.operators for row in maps for m in row)
            write_json(store.root / (name + ".failure.json"), {"reason": str(error), "sampled": 100, "jump_paths": 0})
        else:
            trace["cpu_seconds"] = time.process_time() - cpu
            store.save(name, circuit, theta, trace)
    if store.exists(name):
        _, theta, final = store.load(name)
        training = final["training"]
        bank = training["noise_banks"][0]
        assert len(training["noise_banks"]) == 1
        assert bank["sampled_paths"] == 100
        assert sum(bank["multiplicities"]) == 100
        assert bank["retained_paths"] == bank["jump_paths"] + int(bank["no_jump_paths"] > 0)
        assert not circuit.gates or bank["jump_paths"] > 0
        result.update({
            "status": "complete",
            "fidelity": final["noise"][hardware]["tjm"]["fidelity"],
            "fidelity_full_lindblad": final["noise"][hardware]["lindblad"]["fidelity"],
            "fidelity_noiseless_final": final["noiseless"]["fidelity"],
            "training_fidelity": training["fidelity_trace"][-1],
            "fine_tuning_cpu_seconds": training["cpu_seconds"],
            "fine_tuning_wall_seconds": training["wall_seconds"],
            "retained_paths": bank["retained_paths"],
            "jump_paths": bank["jump_paths"],
            "no_jump_paths": bank["no_jump_paths"],
            "fine_tuning_parameter_updates": training["steps"] * circuit.num_params,
            "fine_tuning_ideal_local_vector_applications": 1000 * len(circuit.gates) * bank["retained_paths"],
        })
    else:
        theta = initial
        result.update({"status": "no_jump_bank", "fidelity": None, "jump_paths": 0, "retained_paths": 1})
    compiled, resources = compile_resources(circuit, theta)
    result["resources"] = resources
    result["resource_endpoint"] = "final" if store.exists(name) else "post_pruning_no_fine_tuning"
    (store.root / (name + ".qasm")).write_text(qasm2.dumps(compiled) + "\n", encoding="utf-8")
    write_json(store.root / (name + ".resources.json"), resources)
    return result


def run_target(job: tuple[str, str, str, int]) -> dict:
    """Run four policies from the same authenticated target/seed checkpoint."""
    output, input_root, target_id, seed = job
    target = load_target_collection().load(6, target_id)
    folder = target_id + "__seed" + str(seed)
    source = Checkpoints(Path(input_root) / folder, target.state_vector_copy(), resume=True)
    root = Path(output) / folder
    if (root / "complete.json").exists():
        saved = json.loads((root / "complete.json").read_text())
        if any(file_hash(root / path) != digest for path, digest in saved["artifacts"].items()):
            msg = "Completed output artifacts changed."
            raise ValueError(msg)
        return saved
    store = Checkpoints(root, target.state_vector_copy(), resume=True)
    initial, theta, before = source.load("unpruned_initial")
    assert before["noiseless"]["fidelity"] >= 0.9
    results, pruning = [], {}
    for variant in VARIANTS:
        if not store.exists(variant + "_initial"):
            print(folder, "pruning", variant, flush=True)
            circuit, values, trace = prune_variant(initial, theta, store.target, variant)
            store.save(variant + "_initial", circuit, values, trace)
        pruning[variant] = store.load(variant + "_initial")[2]["training"]
        for hardware in HARDWARE:
            result = fine_tune(store, variant, hardware, seed)
            if variant == "A":
                result["historical_parity"] = verify_baseline(store, source, hardware, result)
            results.append(result)
    saved = {
        "target_id": target_id,
        "optimizer_seed": seed,
        "target_generation_seed": target.seed,
        "initial_gates": len(initial.gates),
        "initial_noiseless_fidelity": before["noiseless"]["fidelity"],
        "shared_construction_updates": sum(t["steps"] for t in before["training"]),
        "shared_construction_cpu_seconds": None,
        "pruning": pruning,
        "results": results,
        "artifacts": {p.name: file_hash(p) for p in sorted(root.iterdir()) if p.is_file()},
    }
    write_json(root / "complete.json", saved)
    print("COMPLETE", folder, [(r["variant"], r["hardware"], r["status"]) for r in results], flush=True)
    return saved


def make_manifest(source: Path) -> dict:
    """Authenticate all historical inputs, code and fixture before starting jobs."""
    collection = load_target_collection()
    targets = collection.select(num_qubits=(6,))
    assert len(targets) == 9
    old = json.loads((source / "manifest.json").read_text())
    assert old["fixture_sha256"] == file_hash(DEFAULT_TARGET_PATH)
    assert all(file_hash(REPOSITORY / name) == digest for name, digest in old["sources"].items())
    assert old["packages"] == {name: version(name) for name in old["packages"]}
    inputs = {"manifest.json": file_hash(source / "manifest.json")}
    for target in targets:
        for seed in SEEDS:
            folder = source / (target.target_id + "__seed" + str(seed))
            completed = json.loads((folder / "complete.json").read_text())
            for name, digest in completed["artifacts"].items():
                assert file_hash(folder / name) == digest
                if name.startswith(("unpruned_initial", "cutoff_", "target.npy")):
                    inputs[str((folder / name).relative_to(source))] = digest
            inputs[str((folder / "complete.json").relative_to(source))] = file_hash(folder / "complete.json")
    sources = [
        Path(__file__),
        REPOSITORY / "benchmarks/state_preparation/pilot_components.py",
        *[REPOSITORY / name for name in old["sources"]],
        REPOSITORY / "src/mqt/yaqs/optimization/parameterized_circuit.py",
        REPOSITORY / "src/mqt/yaqs/optimization/gate_noise.py",
        REPOSITORY / "src/mqt/yaqs/optimization/krotov.py",
    ]
    return {
        "protocol": PROTOCOL,
        "target_ids": [target.target_id for target in targets],
        "optimizer_seeds": list(SEEDS),
        "fixture_sha256": file_hash(DEFAULT_TARGET_PATH),
        "source_root": str(source.resolve()),
        "inputs": inputs,
        "sources": {str(p.relative_to(REPOSITORY)): file_hash(p) for p in sources},
        "python": platform.python_version(),
        "packages": old["packages"],
    }


def main() -> None:
    """Execute the complete 4 x 9 x 2 x 2 matrix with authenticated resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--workers", type=parse_workers, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = make_manifest(args.source)
    if args.output.exists():
        if not args.resume or json.loads((args.output / "manifest.json").read_text()) != manifest:
            msg = "Use a fresh output directory, or --resume with identical inputs, code and environment."
            raise ValueError(msg)
    else:
        args.output.mkdir(parents=True)
        write_json(args.output / "manifest.json", manifest)
        for name in manifest["sources"]:
            destination = args.output / "source" / (name + ".txt")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPOSITORY / name).read_bytes())
    jobs = [
        (str(args.output.resolve()), str(args.source.resolve()), target_id, seed)
        for target_id in manifest["target_ids"]
        for seed in SEEDS
    ]
    write_json(
        args.output / "execution.json",
        {"jobs": len(jobs), "conditions": 144, "workers": resolve_workers(args.workers, len(jobs))},
    )
    results = {}
    for index, result in completed_jobs(run_target, jobs, workers=args.workers):
        results[index] = result
        write_json(args.output / "results.json", [results[index] for index in sorted(results)])


if __name__ == "__main__":
    main()
