# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Extend the frozen legacy 3/100-history comparison to all nine q6 fixture targets.

Only target identity and register size extend the previous experiment. The
archived growth/rank-pruning and literal hybrid/0.01-cutoff routes retain their
budgets, noise coefficients, seed schedules, and CRN policies. This is a noisy
fine-tuning extension, distinct from the base benchmark's noise-profile matrix.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from benchmarks.state_preparation.execution import completed_jobs, parse_workers, resolve_workers
from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH, load_target_collection
from experiments.legacy_topdown_dense import Recorder, extend, prune, train
from experiments.legacy_topdown_weighted import file_hash, load_circuit
from mqt.yaqs.optimization import StatevectorBackend, create_brickwall_matrix_product_disentangler_parameterized_circuit

if TYPE_CHECKING:
    from mqt.yaqs.optimization import ParameterizedCircuit

REPOSITORY = Path(__file__).resolve().parents[1]
SEEDS = (300, 400)
PROTOCOL = {
    "version": 1,
    "qubits": 6,
    "seeds": list(SEEDS),
    "trajectories": [3, 100],
    "archived_depths": [4, 8, 12, 16],
    "archived_rates": [5.0, 2.0, 0.5, 0.2],
    "archived_steps_per_depth": 150,
    "archived_decay": 0.01,
    "pruning_drop_cap": 150,
    "pruning_gate_floor": 150,
    "repair_steps": 40,
    "repair_rate": 0.2,
    "literal_depth_cap": 16,
    "literal_steps_per_depth": 200,
    "literal_online_steps": 100,
    "literal_rate": 0.7,
    "literal_threshold": 0.9,
    "cutoff": 0.01,
    "noise": "unchanged LegacyNoise IBM/Google Pauli coefficients and at-most-one-jump gate windows",
    "selection": "fixed final iteration; independent exact density scoring; no selection using test scores",
}


def write_json(path: Path, value: object) -> None:
    """Atomically persist finite JSON in an existing output directory."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class Checkpoints(Recorder):
    """Add parameter/circuit checksums to independently scored legacy checkpoints."""

    def save(self, name: str, circuit: ParameterizedCircuit, theta: np.ndarray, trace: dict | list) -> None:
        """Save independent exact scores and hashes for safe partial-job resume."""
        super().save(name, circuit, theta, trace)
        paths = [self.root / (name + suffix) for suffix in (".json", ".theta.npy")]
        write_json(self.root / (name + ".sha256.json"), {path.name: file_hash(path) for path in paths})

    def exists(self, name: str) -> bool:
        """Return whether a complete authenticated checkpoint is present."""
        return (self.root / (name + ".sha256.json")).exists()

    def load(self, name: str) -> tuple:
        """Verify and load a saved circuit, parameter vector, and score record.

        Raises:
            ValueError: If a checkpoint was modified or corrupted.
        """
        hashes = json.loads((self.root / (name + ".sha256.json")).read_text())
        if any(file_hash(self.root / filename) != digest for filename, digest in hashes.items()):
            msg = f"Checkpoint checksum mismatch: {self.root / name}"
            raise ValueError(msg)
        record = json.loads((self.root / (name + ".json")).read_text())
        return load_circuit(record), np.load(self.root / (name + ".theta.npy")), record


def brickwall_for(target: np.ndarray, depth: int) -> ParameterizedCircuit:
    """Retain the original BMPD architecture with the target's register size."""
    qubits = len(target).bit_length() - 1
    return create_brickwall_matrix_product_disentangler_parameterized_circuit(
        qubits, depth, initial_single_qubit_layer=True
    )


def construct_archived(store: Checkpoints, seed: int) -> None:
    """Construct the archived VQA, layerwise, and rank-pruned initial checkpoints."""
    target = store.target
    if not store.exists("fixed_vqa"):
        circuit = brickwall_for(target, 4)
        theta = extend(None, circuit.num_params, seed + 1, 0.1)
        theta, trace = train(
            circuit,
            theta,
            target,
            steps=300,
            rate=0.2,
            decay=0.01,
            seed=seed + 2,
            noisy=True,
            record_noise_banks=True,
        )
        store.save("fixed_vqa", circuit, theta, trace)
    if not store.exists("layerwise_initial"):
        theta, traces = np.empty(0, dtype=float), []
        for depth in (1, 2):
            circuit = brickwall_for(target, depth)
            theta = extend(
                theta,
                circuit.num_params,
                seed + 100 if depth == 1 else seed + 200 + depth,
                0.05 if depth == 1 else 0.001,
            )
            theta, trace = train(circuit, theta, target, steps=100, rate=1.0, seed=seed + 300)
            traces.append(trace)
        store.save("layerwise_initial", circuit, theta, traces)
    if not store.exists("topdown_initial"):
        if store.exists("topdown_before_pruning"):
            circuit, theta, _ = store.load("topdown_before_pruning")
        else:
            theta, traces = np.empty(0, dtype=float), []
            for depth, rate in zip((4, 8, 12, 16), (5.0, 2.0, 0.5, 0.2), strict=True):
                circuit = brickwall_for(target, depth)
                theta = extend(
                    theta,
                    circuit.num_params,
                    seed + 1000 if depth == 4 else seed + 2000 + depth,
                    0.05 if depth == 4 else 0.001,
                )
                theta, trace = train(circuit, theta, target, steps=150, rate=rate, decay=0.01, seed=seed + 3000)
                traces.append(trace)
                print(store.root.name, "archived depth", depth, trace["fidelity_trace"][-1], flush=True)
            store.save("topdown_before_pruning", circuit, theta, traces)
        traces = []
        while circuit.num_params > 150:
            impact = np.abs(theta * StatevectorBackend(circuit, target).gradient(theta)[0])
            circuit, theta = prune(circuit, theta, impact, keep=max(150, circuit.num_params - 150))
            if circuit.num_params > 150:
                theta, trace = train(circuit, theta, target, steps=40, rate=0.2, seed=42)
                traces.append(trace)
        store.save("topdown_initial", circuit, theta, traces)


def construct_literal(store: Checkpoints, seed: int) -> bool:
    """Grow one layer at a time, stop at 90%, and retain the literal 0.01 cutoff."""
    target = store.target
    if store.exists("unpruned_initial"):
        circuit, theta, saved = store.load("unpruned_initial")
        fidelity = saved["noiseless"]["fidelity"]
        reached = saved["training"][-1]["fidelity_trace"][-1] >= 0.9
    else:
        theta, traces = np.empty(0, dtype=float), []
        for depth in range(1, 17):
            circuit = brickwall_for(target, depth)
            theta = extend(theta, circuit.num_params, seed + 1000 + depth, 0.05 if depth == 1 else 0.001)
            theta, trace = train(circuit, theta, target, steps=200, online=100, rate=0.7, seed=seed + 3000, stop=0.9)
            traces.append(trace)
            fidelity = trace["fidelity_trace"][-1]
            print(store.root.name, "literal depth", depth, fidelity, flush=True)
            if fidelity >= 0.9:
                break
        store.save("unpruned_initial", circuit, theta, traces)
        reached = fidelity >= 0.9
    if not reached:
        write_json(store.root / "literal_failure.json", {"reason": "90% not reached by depth 16", "fidelity": fidelity})
        return False
    if not store.exists("cutoff_initial"):
        impact = np.abs(theta * StatevectorBackend(circuit, target).gradient(theta)[0])
        pruned, values = prune(circuit, theta, impact)
        store.save("cutoff_initial", pruned, values, {"cutoff": 0.01, "impacts": impact.tolist()})
    return True


def pair(store: Checkpoints, name: str, hardware: str, seed: int) -> dict:
    """Compare the same initial circuit with three and 100 weighted histories.

    A no-jump coverage failure is recorded as an unsuccessful 100-path trial;
    its bank is never replaced by a more favorable seed or forced jump.
    """
    circuit, initial, before = store.load(name + "_initial")
    run = partial(
        train,
        circuit,
        initial,
        store.target,
        steps=100 if name == "layerwise" else 200,
        online=100 if name in {"cutoff", "unpruned"} else 0,
        rate=0.2 if name in {"cutoff", "unpruned"} else 0.1,
        seed=seed + (400 if name == "layerwise" else 5000),
        crn=name != "layerwise",
        noisy=True,
        hardware=hardware,
        record_noise_banks=True,
    )
    baseline_name, weighted_name = name + "_" + hardware + "_3", name + "_" + hardware + "_100"
    if not store.exists(baseline_name):
        theta, trace = run()
        store.save(baseline_name, circuit, theta, trace)
    _, _, baseline = store.load(baseline_name)
    result = {
        "method": name,
        "hardware": hardware,
        "gates": len(circuit.gates),
        "one_qubit_gates": before["gate_counts"]["one"],
        "two_qubit_gates": before["gate_counts"]["two"],
        "fidelity_before": before["noise"][hardware]["tjm"]["fidelity"],
        "fidelity_3": baseline["noise"][hardware]["tjm"]["fidelity"],
        "baseline_checkpoint": baseline_name,
        "weighted_checkpoint": weighted_name,
    }
    if not store.exists(weighted_name):
        print(store.root.name, "fine-tuning 100", name, hardware, len(circuit.gates), "gates", flush=True)
        try:
            theta, trace = run(
                num_trajectories=100,
                compress_no_jump=True,
                require_jump_coverage=True,
            )
        except RuntimeError as error:
            if "sampled bank has no jumps" not in str(error):
                raise
            result.update({"status": "no_jump_bank", "reason": str(error), "fidelity_100": None})
            write_json(store.root / (weighted_name + ".failure.json"), result)
            return result
        store.save(weighted_name, circuit, theta, trace)
    _, _, weighted = store.load(weighted_name)
    assert before["circuit"] == baseline["circuit"] == weighted["circuit"]
    np.testing.assert_array_equal(initial, store.load(name + "_initial")[1])
    first, second = baseline["training"], weighted["training"]
    for key in ("steps", "online_steps", "rate", "decay", "seed", "crn", "hardware"):
        assert first[key] == second[key]
    for old, new in zip(first["noise_banks"], second["noise_banks"], strict=True):
        assert old["events"] == new["events"][:3]
        assert old["first_three_sha256"] == new["first_three_sha256"]
        assert new["sampled_paths"] == 100
        assert new["retained_paths"] == new["jump_paths"] + int(new["no_jump_paths"] > 0)
        assert sum(new["multiplicities"]) == 100
        assert abs(sum(new["weights"]) - 1) < 1e-12
        assert not circuit.gates or new["jump_paths"] > 0
    result.update({
        "status": "complete",
        "fidelity_100": weighted["noise"][hardware]["tjm"]["fidelity"],
        "noiseless_3": baseline["noiseless"]["fidelity"],
        "noiseless_100": weighted["noiseless"]["fidelity"],
        "training_fidelity_3": first["fidelity_trace"][-1],
        "training_fidelity_100": second["fidelity_trace"][-1],
        "seconds_3": first["wall_seconds"],
        "seconds_100": second["wall_seconds"],
        "banks": len(second["noise_banks"]),
        "retained_paths": [bank["retained_paths"] for bank in second["noise_banks"]],
        "jump_paths": [bank["jump_paths"] for bank in second["noise_banks"]],
        "nested_histories_verified": True,
    })
    return result


def run_target(job: tuple[str, str, int]) -> dict:
    """Run one fixture target/optimizer seed with isolated, resumable checkpoints."""
    output, target_id, seed = job
    target = load_target_collection().load(6, target_id)
    root = Path(output) / (target_id + "__seed" + str(seed))
    if (root / "complete.json").exists():
        saved = json.loads((root / "complete.json").read_text())
        if any(file_hash(root / path) != digest for path, digest in saved["artifacts"].items()):
            msg = f"Completed target artifacts changed: {root}"
            raise ValueError(msg)
        return saved
    store = Checkpoints(root, target.state_vector_copy(), resume=True)
    start = time.perf_counter()
    construct_archived(store, seed)
    results = [pair(store, name, "IBM", seed) for name in ("topdown", "layerwise")]
    reached = construct_literal(store, seed)
    for name in ("cutoff", "unpruned"):
        results.extend(
            pair(store, name, hardware, seed)
            if reached
            else {
                "method": name,
                "hardware": hardware,
                "status": "construction_failed",
                "reason": "90% not reached by depth 16",
                "fidelity_100": None,
            }
            for hardware in ("IBM", "Google")
        )
    _, _, reference = store.load("fixed_vqa")
    saved = {
        "target_id": target_id,
        "optimizer_seed": seed,
        "target_generation_seed": target.seed,
        "qubits": 6,
        "results": results,
        "fixed_vqa": reference,
        "wall_seconds_this_invocation": time.perf_counter() - start,
        "artifacts": {path.name: file_hash(path) for path in sorted(root.iterdir()) if path.is_file()},
    }
    write_json(root / "complete.json", saved)
    print("COMPLETE", target_id, seed, [(r["method"], r["hardware"], r["status"]) for r in results], flush=True)
    return saved


def main() -> None:
    """Execute all nine q6 targets and both fixed optimizer seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=parse_workers, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    collection = load_target_collection()
    targets = collection.select(num_qubits=(6,))
    assert len(targets) == 9
    sources = [
        Path(__file__),
        REPOSITORY / "experiments/legacy_topdown_dense.py",
        REPOSITORY / "experiments/legacy_topdown_weighted.py",
        REPOSITORY / "src/mqt/yaqs/optimization/statevector.py",
        REPOSITORY / "benchmarks/state_preparation/targets.py",
        REPOSITORY / "benchmarks/state_preparation/execution.py",
    ]
    manifest = {
        "protocol": PROTOCOL,
        "fixture_checksum": collection.checksum,
        "fixture_sha256": file_hash(DEFAULT_TARGET_PATH),
        "target_ids": [target.target_id for target in targets],
        "sources": {str(path.relative_to(REPOSITORY)): file_hash(path) for path in sources},
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "qiskit", "threadpoolctl")},
    }
    if args.output.exists():
        if not args.resume or json.loads((args.output / "manifest.json").read_text()) != manifest:
            msg = "Use a fresh output directory or --resume with identical sources, fixture, and environment."
            raise ValueError(msg)
    else:
        args.output.mkdir(parents=True)
        write_json(args.output / "manifest.json", manifest)
        for path in sources:
            (args.output / (path.name + ".txt")).write_bytes(path.read_bytes())
        (args.output / "target_fixture.json").write_bytes(DEFAULT_TARGET_PATH.read_bytes())
    jobs = [(str(args.output.resolve()), target.target_id, seed) for target in targets for seed in SEEDS]
    write_json(args.output / "execution.json", {"workers": resolve_workers(args.workers, len(jobs)), "jobs": len(jobs)})
    results = {}
    for index, result in completed_jobs(run_target, jobs, workers=args.workers):
        results[index] = result
        write_json(args.output / "results.json", [results[index] for index in sorted(results)])


if __name__ == "__main__":
    main()
