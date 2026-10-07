# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Fine-tune every depth candidate with frozen B/D pruning and weighted histories.

Selection sees only independent trajectory validation. Exact density scores and
compiled resource counts are produced after all depth decisions are locked.
The old 90% pipeline is rerun separately for paired timing and parity checks.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qiskit import qasm2

from benchmarks.state_preparation.execution import completed_jobs, parse_workers, resolve_workers
from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH, load_target_collection
from experiments.legacy_pruning_ablation import HARDWARE, compile_resources, prune_variant
from experiments.legacy_pruning_ablation import PROTOCOL as PRUNING_PROTOCOL
from experiments.legacy_topdown_dense import extend, local_model, matrix_score, train
from experiments.legacy_topdown_q6 import REPOSITORY, SEEDS, Checkpoints, brickwall_for, write_json
from experiments.legacy_topdown_weighted import file_hash, load_circuit
from mqt.yaqs.optimization import StatevectorBackend

# Reuse only the tested local state application, not the training RNG stream.
from mqt.yaqs.optimization.statevector import _apply  # noqa: PLC2701 -- Reuse the tested numerical kernel.

if TYPE_CHECKING:
    from mqt.yaqs.optimization import ParameterizedCircuit

VARIANTS = ("B", "D")
BASELINE_ROOT = REPOSITORY / "output/legacy-pruning-ablation-q6-20260914"
CONSTRUCTION_ROOT = REPOSITORY / "output/legacy-topdown-q6-weighted100-20260914"
PROTOCOL: dict = {
    "version": 1,
    "variants": list(VARIANTS),
    "hardware": list(HARDWARE),
    "depth_cap": 16,
    "construction": {"steps": 200, "online": 100, "rate": 0.7},
    "construction_endpoint": "final iterate; stop=2 disables fidelity threshold; no best-iterate rollback",
    "initialization": "unchanged RandomState(seed+1000+depth), sigma=.05 initially and .001 subsequently",
    "growth_parent": "unpruned noiseless checkpoint, shared across active pruning/noise branches",
    "patience": 3,
    "minimum_gain": 0.001,
    "uncertainty_multiplier": 2.0,
    "patience_reset": "validation exceeds last material record by .001 + 2*sqrt(SE_current^2+SE_record^2)",
    "selection": "maximum validation fidelity among successful visited candidates; earliest depth on exact tie",
    "failed_candidate": "count as no improvement; never replace a zero-jump training/validation bank",
    "validation_paths": 32768,
    "validation_entropy": "[20260914, 6, target index, optimizer seed, variant index, hardware index, depth, 90000]",
    "validation_law": "same independent at-most-one-Pauli-jump windows; unconditional categorical draws",
    "fine_tuning": PRUNING_PROTOCOL["fine_tuning"],
    "fine_tuning_seed": "optimizer seed + 5000 at every depth, as before; gates are not aligned across circuits",
    "pruning": PRUNING_PROTOCOL,
    "baseline": "rerun first-90% construction and B/D, checking archived parameters and noise-bank parity",
    "test": "Qiskit exact sampler-law density scores after selection lock; also report full Lindblad sensitivity",
    "cost": "sum of construction, pruning/repair, training and validation CPU including discarded candidates",
}


def protocol_for(training_paths: int = 100) -> dict:
    """Change only the raw noisy-training sample count in the frozen protocol."""
    if type(training_paths) is not int or training_paths < 1:
        msg = "Training histories must be a positive integer."
        raise ValueError(msg)
    protocol = deepcopy(PROTOCOL)
    protocol["fine_tuning"]["num_trajectories"] = training_paths
    protocol["pruning"]["fine_tuning"]["num_trajectories"] = training_paths
    return protocol


@dataclass
class Patience:
    """Track a material record separately from the maximum-score winner."""

    stale: int = 0
    record_score: float | None = None
    record_se: float = 0.0
    best_score: float | None = None
    best_depth: int | None = None

    def observe(self, depth: int, score: float | None, se: float = 0.0) -> dict:
        """Consume validation only; failures advance patience without a winner."""
        if score is not None and (not np.isfinite(score) or not np.isfinite(se) or se < 0):
            msg = "Validation scores and nonnegative standard errors must be finite."
            raise ValueError(msg)
        threshold = PROTOCOL["minimum_gain"] + PROTOCOL["uncertainty_multiplier"] * np.hypot(se, self.record_se)
        improved = score is not None and (self.record_score is None or score > self.record_score + threshold)
        if improved:
            self.record_score, self.record_se, self.stale = score, se, 0
        else:
            self.stale += 1
        if score is not None and (self.best_score is None or score > self.best_score):
            self.best_score, self.best_depth = score, depth
        return {**asdict(self), "material_improvement": bool(improved), "required_gain": float(threshold)}


def save_checkpoint(root: Path, name: str, circuit: ParameterizedCircuit, theta: np.ndarray, data: dict) -> dict:
    """Persist parameters and audit metadata without evaluating noisy test fidelity."""
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "circuit": {
            "num_qubits": circuit.num_qubits,
            "num_params": circuit.num_params,
            "gates": [{**asdict(gate), "sites": list(gate.sites)} for gate in circuit.gates],
        },
        "theta": theta.tolist(),
        **data,
    }
    path = root / (name + ".json")
    write_json(path, record)
    write_json(root / (name + ".sha256.json"), {path.name: file_hash(path)})
    return json.loads(path.read_text())


def read_checkpoint(root: Path, name: str) -> dict | None:
    """Load a complete authenticated checkpoint, or report a missing stage."""
    marker = root / (name + ".sha256.json")
    if not marker.exists():
        return None
    hashes = json.loads(marker.read_text())
    if any(file_hash(root / path) != digest for path, digest in hashes.items()):
        msg = "Checkpoint checksum mismatch."
        raise ValueError(msg)
    return json.loads((root / (name + ".json")).read_text())


def noise_outcomes(hardware: str, sites: tuple[int, ...]) -> tuple[np.ndarray, list]:
    """Expose scalar Pauli probabilities and operators in the existing model order."""
    processes = local_model(hardware, sites).processes
    rates = np.array([process["strength"] for process in processes], dtype=float)
    total = rates.sum()
    probabilities = np.concatenate(([np.exp(-total)], -np.expm1(-total) * rates / total))
    operators = []
    for process in processes:
        matrix = process["matrix"] if "matrix" in process else np.kron(*process["factors"])
        operators.append((np.asarray(matrix, dtype=complex), tuple(process["sites"])))
    return probabilities, operators


def sample_validation(circuit: ParameterizedCircuit, hardware: str, count: int, entropy: list[int]) -> np.ndarray:
    """Draw a complete independent unconditional bank before vector propagation."""
    if count < 2:
        msg = "Validation requires at least two raw histories."
        raise ValueError(msg)
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    codes = np.zeros((count, len(circuit.gates)), dtype=np.uint8)
    for index, gate in enumerate(circuit.gates):
        if gate.noise_enabled:
            probabilities, _ = noise_outcomes(hardware, gate.sites)
            codes[:, index] = rng.choice(len(probabilities), size=count, p=probabilities)
    return codes


def validation_score(
    circuit: ParameterizedCircuit, theta: np.ndarray, target: np.ndarray, hardware: str, codes: np.ndarray
) -> dict:
    """Evaluate a bank, compressing no-jump duplicates and preserving raw-sample SE."""
    if codes.ndim != 2 or codes.shape[1] != len(circuit.gates) or len(codes) < 2:
        msg = "Validation bank shape must match the circuit and contain at least two histories."
        raise ValueError(msg)
    has_jumps = np.any(codes != 0, axis=1)
    retained_indices = np.flatnonzero(has_jumps).tolist()
    no_jump_count = int(np.count_nonzero(~has_jumps))
    multiplicities = [1] * len(retained_indices)
    if no_jump_count:
        retained_indices.insert(0, int(np.flatnonzero(~has_jumps)[0]))
        multiplicities.insert(0, no_jump_count)
    retained_codes = codes[retained_indices]
    states = np.zeros((len(retained_codes), 2**circuit.num_qubits), dtype=complex)
    states[:, 0] = 1
    analytic_no_jump = 1.0
    for index, gate in enumerate(circuit.gates):
        unitary, sites = circuit.gate_matrix(gate, theta)
        states = _apply(states, unitary, sites, circuit.num_qubits)
        probabilities, operators = noise_outcomes(hardware, sites)
        if gate.noise_enabled:
            analytic_no_jump *= probabilities[0]
        column = retained_codes[:, index]
        if np.any(column >= len(probabilities)) or (not gate.noise_enabled and np.any(column)):
            msg = "Invalid jump code for this gate."
            raise ValueError(msg)
        for code, (operator, acted_sites) in enumerate(operators, start=1):
            selected = np.flatnonzero(column == code)
            if len(selected):
                states[selected] = _apply(states[selected], operator, acted_sites, circuit.num_qubits)
    fidelities = abs(states @ target.conj()) ** 2
    weights = np.array(multiplicities) / len(codes)
    mean = float(weights @ fidelities)
    se = float(np.sqrt(weights @ (fidelities - mean) ** 2 / (len(codes) - 1)))
    return {
        "fidelity": mean,
        "standard_error": se,
        "sampled_paths": len(codes),
        "no_jump_paths": no_jump_count,
        "jump_paths": int(has_jumps.sum()),
        "retained_paths": len(retained_codes),
        "analytic_no_jump_probability": float(analytic_no_jump),
        "norm_max_error": float(np.max(abs(np.sum(abs(states) ** 2, axis=1) - 1))),
    }


def construct(root: Path, target: np.ndarray, seed: int, depth: int, previous: np.ndarray, *, baseline: bool) -> dict:
    """Grow from the noiseless parameter lineage, retaining the existing optimizer."""
    name = f"construction_{depth:02d}"
    saved = read_checkpoint(root, name)
    if saved is not None:
        return saved
    cpu, wall = time.process_time(), time.perf_counter()
    circuit = brickwall_for(target, depth)
    initial = extend(previous, circuit.num_params, seed + 1000 + depth, 0.05 if depth == 1 else 0.001)
    theta, trace = train(
        circuit, initial, target, seed=seed + 3000, stop=0.9 if baseline else 2.0, **PROTOCOL["construction"]
    )
    return save_checkpoint(
        root,
        name,
        circuit,
        theta,
        {
            "depth": depth,
            "ideal_fidelity": trace["fidelity_trace"][-1],
            "training": trace,
            "cpu_seconds": time.process_time() - cpu,
            "wall_seconds": time.perf_counter() - wall,
        },
    )


def prune_checkpoint(root: Path, target: np.ndarray, construction: dict, variant: str) -> dict:
    """Prune a copy, shared by hardware branches but never fed back into growth."""
    name = f"depth_{construction['depth']:02d}_{variant}_pruned"
    saved = read_checkpoint(root, name)
    if saved is not None:
        return saved
    circuit, theta, trace = prune_variant(load_circuit(construction), np.array(construction["theta"]), target, variant)
    return save_checkpoint(root, name, circuit, theta, {"depth": construction["depth"], "pruning": trace})


def candidate(
    root: Path,
    target: np.ndarray,
    seed: int,
    target_index: int,
    pruned: dict,
    variant: str,
    hardware: str,
    *,
    baseline: bool,
    training_paths: int = 100,
) -> dict:
    """Fine-tune one depth and validate using a separate frozen sample bank."""
    depth = pruned["depth"]
    name = f"depth_{depth:02d}_{variant}_{hardware}"
    saved = read_checkpoint(root, name)
    if saved is not None:
        if saved.get("training_paths", 100) != training_paths:
            msg = "Cached candidate has a different training-history count."
            raise ValueError(msg)
        validation = saved.get("validation")
        if validation and file_hash(root / validation["bank_file"]) != validation["bank_sha256"]:
            msg = "Validation bank checksum mismatch."
            raise ValueError(msg)
        return saved
    circuit, initial = load_circuit(pruned), np.array(pruned["theta"])
    cpu, wall = time.process_time(), time.perf_counter()
    try:
        theta, trace = train(
            circuit,
            initial,
            target,
            seed=seed + 5000,
            hardware=hardware,
            **protocol_for(training_paths)["fine_tuning"],
        )
    except RuntimeError as error:
        if "sampled bank has no jumps" not in str(error):
            raise
        return save_checkpoint(
            root,
            name,
            circuit,
            initial,
            {
                "depth": depth,
                "training_paths": training_paths,
                "variant": variant,
                "hardware": hardware,
                "status": "no_jump_training_bank",
                "reason": str(error),
                "training_cpu_seconds": time.process_time() - cpu,
                "training_wall_seconds": time.perf_counter() - wall,
                "validation_cpu_seconds": 0.0,
                "validation_wall_seconds": 0.0,
                "training": None,
                "validation": None,
            },
        )
    data = {
        "depth": depth,
        "training_paths": training_paths,
        "variant": variant,
        "hardware": hardware,
        "status": "complete",
        "training": trace,
        "training_cpu_seconds": time.process_time() - cpu,
        "training_wall_seconds": time.perf_counter() - wall,
        "validation_cpu_seconds": 0.0,
        "validation_wall_seconds": 0.0,
        "validation": None,
    }
    if not baseline:
        cpu, wall = time.process_time(), time.perf_counter()
        entropy = [20260914, 6, target_index, seed, VARIANTS.index(variant), HARDWARE.index(hardware), depth, 90000]
        codes = sample_validation(circuit, hardware, PROTOCOL["validation_paths"], entropy)
        validation = validation_score(circuit, theta, target, hardware, codes)
        bank_file = name + ".validation.npz"
        np.savez_compressed(root / bank_file, codes=codes)
        validation.update({"entropy": entropy, "bank_file": bank_file, "bank_sha256": file_hash(root / bank_file)})
        data.update({
            "validation": validation,
            "validation_cpu_seconds": time.process_time() - cpu,
            "validation_wall_seconds": time.perf_counter() - wall,
        })
        if circuit.gates and validation["jump_paths"] == 0:
            data["status"] = "no_jump_validation_bank"
    return save_checkpoint(root, name, circuit, theta, data)


def verify_baseline(
    root: Path,
    target: np.ndarray,
    folder: str,
    construction: dict,
    candidates: list[dict],
    prunings: dict,
    training_paths: int = 100,
) -> dict:
    """Check fixed starts and common bank prefixes; require endpoint parity at M=100."""
    source = Checkpoints(CONSTRUCTION_ROOT / folder, target, resume=True)
    _, theta, old = source.load("unpruned_initial")
    np.testing.assert_allclose(construction["theta"], theta, atol=1e-12, rtol=0)
    assert construction["circuit"] == old["circuit"]
    archive = Checkpoints(BASELINE_ROOT / folder, target, resume=True)
    errors = []
    prefixes = []
    for item in candidates:
        _, initial, pruned = archive.load(item["variant"] + "_initial")
        assert prunings[item["variant"]]["circuit"] == pruned["circuit"]
        np.testing.assert_array_equal(prunings[item["variant"]]["theta"], initial)
        name = item["variant"] + "_" + item["hardware"]
        _, expected, old = archive.load(name)
        assert item["status"] == "complete"
        assert item["circuit"] == old["circuit"]
        common = min(training_paths, 100)
        assert (
            item["training"]["noise_banks"][0]["events"][:common]
            == old["training"]["noise_banks"][0]["events"][:common]
        )
        prefixes.append(common)
        if training_paths == 100:
            np.testing.assert_allclose(item["theta"], expected, atol=1e-12, rtol=0)
            assert item["training"]["noise_banks"] == old["training"]["noise_banks"]
            errors.append(float(np.max(abs(np.array(item["theta"]) - expected))) if len(expected) else 0.0)
    result = {
        "maximum_parameter_error": max(errors) if errors else None,
        "all_noise_banks_identical": training_paths == 100,
        "all_pruned_starting_parameters_identical": True,
        "common_prefix_histories": min(prefixes),
        "conditions": len(candidates),
    }
    write_json(root / "baseline_parity.json", result)
    return result


def summarize_cost(constructions: list[dict], prunings: list[dict], candidates: list[dict]) -> dict:
    """Charge all visited candidates to each standalone search, without shared-work discounts."""
    costs = {
        "construction_cpu_seconds": sum(item["cpu_seconds"] for item in constructions),
        "pruning_cpu_seconds": sum(item["pruning"]["cpu_seconds"] for item in prunings),
        "fine_tuning_cpu_seconds": sum(item["training_cpu_seconds"] for item in candidates),
        "validation_cpu_seconds": sum(item["validation_cpu_seconds"] for item in candidates),
    }
    return {
        **costs,
        "total_search_cpu_seconds": sum(costs.values()),
        "candidates": len(candidates),
        "construction_updates": sum(item["training"]["steps"] for item in constructions),
        "repair_updates": sum(item["pruning"]["repair_updates"] for item in prunings),
        "fine_tuning_updates": sum(item["training"]["steps"] for item in candidates if item["training"]),
        "failed_candidates": sum(item["status"] != "complete" for item in candidates),
    }


def evaluate_endpoint(root: Path, item: dict, target: np.ndarray) -> dict:
    """Score and compile a fixed candidate only after the selection artifact exists."""
    if not (root.parent / "selection_locked.json").exists():
        msg = "Exact evaluation requires locked selection."
        raise RuntimeError(msg)
    circuit, theta = load_circuit(item), np.array(item["theta"])
    cpu = time.process_time()
    result = {
        "depth": item["depth"],
        "variant": item["variant"],
        "hardware": item["hardware"],
        "status": item["status"],
        "ideal": matrix_score(circuit, theta, target, item["hardware"], "noiseless"),
        "exact": matrix_score(circuit, theta, target, item["hardware"], "tjm"),
        "lindblad": matrix_score(circuit, theta, target, item["hardware"], "lindblad"),
    }
    result["evaluation_cpu_seconds"] = time.process_time() - cpu
    dense = StatevectorBackend(circuit, target).metrics(theta)[1]
    result["vector_matrix_error"] = abs(dense - result["ideal"]["fidelity"])
    assert result["vector_matrix_error"] < 1e-10
    cpu = time.process_time()
    compiled, result["resources"] = compile_resources(circuit, theta)
    result["compilation_cpu_seconds"] = time.process_time() - cpu
    name = f"depth_{item['depth']:02d}_{item['variant']}_{item['hardware']}"
    (root / (name + ".qasm")).write_text(qasm2.dumps(compiled) + "\n", encoding="utf-8")
    write_json(root / (name + ".evaluation.json"), result)
    return result


def run_target(job: tuple[str, str, int, int, int]) -> dict:
    """Run one target/seed with shared construction and four separately stopped branches."""
    output, target_id, seed, target_index, training_paths = job
    target_record = load_target_collection().load(6, target_id)
    target = target_record.state_vector_copy()
    folder = target_id + "__seed" + str(seed)
    root = Path(output) / folder
    if (root / "complete.json").exists():
        saved = json.loads((root / "complete.json").read_text())
        if any(file_hash(root / name) != digest for name, digest in saved["artifacts"].items()):
            msg = "Completed search artifacts changed."
            raise ValueError(msg)
        return saved
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "target.npy", target)
    baseline_root, search_root = root / "baseline", root / "search"
    baseline_root.mkdir(exist_ok=True)
    search_root.mkdir(exist_ok=True)
    baseline_construction, baseline_prunings, baseline_candidates = [], {}, []
    previous = np.empty(0)
    for depth in range(1, PROTOCOL["depth_cap"] + 1):
        built = construct(baseline_root, target, seed, depth, previous, baseline=True)
        previous = np.array(built["theta"])
        baseline_construction.append(built)
        if built["ideal_fidelity"] >= 0.9:
            break
    for variant in VARIANTS:
        pruned = prune_checkpoint(baseline_root, target, built, variant)
        baseline_prunings[variant] = pruned
        baseline_candidates.extend(
            candidate(
                baseline_root,
                target,
                seed,
                target_index,
                pruned,
                variant,
                hardware,
                baseline=True,
                training_paths=training_paths,
            )
            for hardware in HARDWARE
        )
    parity = verify_baseline(root, target, folder, built, baseline_candidates, baseline_prunings, training_paths)
    print(folder, "BASELINE VERIFIED", flush=True)
    branches = {(variant, hardware): Patience() for variant in VARIANTS for hardware in HARDWARE}
    constructions, prunings, candidates, decisions = (
        [],
        {},
        {key: [] for key in branches},
        {key: [] for key in branches},
    )
    previous = np.empty(0)
    for depth in range(1, PROTOCOL["depth_cap"] + 1):
        active = [key for key, tracker in branches.items() if tracker.stale < PROTOCOL["patience"]]
        if not active:
            break
        built = construct(search_root, target, seed, depth, previous, baseline=False)
        previous = np.array(built["theta"])
        constructions.append(built)
        for variant in VARIANTS:
            if not any(key[0] == variant for key in active):
                continue
            pruned = prune_checkpoint(search_root, target, built, variant)
            prunings[depth, variant] = pruned
            for hardware in HARDWARE:
                key = variant, hardware
                if key not in active:
                    continue
                item = candidate(
                    search_root,
                    target,
                    seed,
                    target_index,
                    pruned,
                    variant,
                    hardware,
                    baseline=False,
                    training_paths=training_paths,
                )
                candidates[key].append(item)
                score = item["validation"]["fidelity"] if item["status"] == "complete" else None
                se = item["validation"]["standard_error"] if score is not None else 0.0
                decision = branches[key].observe(depth, score, se)
                decisions[key].append({"depth": depth, "score": score, "standard_error": se, **decision})
                print(
                    folder,
                    variant,
                    hardware,
                    "depth",
                    depth,
                    "ideal",
                    round(built["ideal_fidelity"], 5),
                    "validation",
                    score,
                    "stale",
                    decision["stale"],
                    flush=True,
                )
    locked: dict = {
        "target_id": target_id,
        "optimizer_seed": seed,
        "branches": [
            {
                "variant": variant,
                "hardware": hardware,
                "selected_depth": tracker.best_depth,
                "stop_reason": "patience" if tracker.stale >= PROTOCOL["patience"] else "depth_cap",
                "decisions": decisions[variant, hardware],
            }
            for (variant, hardware), tracker in branches.items()
        ],
    }
    write_json(root / "selection_locked.json", locked)
    selection_hash = file_hash(root / "selection_locked.json")
    results = []
    for branch in locked["branches"]:
        variant, hardware = str(branch["variant"]), str(branch["hardware"])
        visited = candidates[variant, hardware]
        baseline = next(
            item for item in baseline_candidates if (item["variant"], item["hardware"]) == (variant, hardware)
        )
        baseline_evaluation = evaluate_endpoint(baseline_root, baseline, target)
        # Post-selection diagnostics evaluate every visited endpoint, without
        # changing the locked winner, even if the exact ranking is different.
        evaluations = [evaluate_endpoint(search_root, item, target) for item in visited if item["status"] == "complete"]
        selected = next((item for item in evaluations if item["depth"] == branch["selected_depth"]), None)
        results.append({
            **branch,
            "selected": selected,
            "baseline": baseline_evaluation,
            "evaluations": evaluations,
            "search_cost": summarize_cost(
                constructions[: len(visited)], [prunings[item["depth"], variant] for item in visited], visited
            ),
            "baseline_cost": summarize_cost(baseline_construction, [baseline_prunings[variant]], [baseline]),
        })
    assert selection_hash == file_hash(root / "selection_locked.json")
    result = {
        "target_id": target_id,
        "optimizer_seed": seed,
        "target_generation_seed": target_record.seed,
        "baseline_parity": parity,
        "selection_sha256": selection_hash,
        "results": results,
        "artifacts": {
            str(path.relative_to(root)): file_hash(path) for path in sorted(root.rglob("*")) if path.is_file()
        },
    }
    write_json(root / "complete.json", result)
    print(
        "COMPLETE",
        folder,
        [(item["variant"], item["hardware"], item["selected_depth"]) for item in results],
        flush=True,
    )
    return result


def make_manifest(training_paths: int = 100) -> dict:
    """Freeze the full protocol, fixture, dependencies and prior B/D inputs."""
    sources = [
        Path(__file__),
        REPOSITORY / "experiments/topdown_depth_search_protocol.md",
        REPOSITORY / "experiments/topdown_m400_protocol.md",
        *[REPOSITORY / name for name in json.loads((BASELINE_ROOT / "manifest.json").read_text())["sources"]],
    ]
    inputs = {}
    for root in (BASELINE_ROOT, CONSTRUCTION_ROOT):
        for path in root.glob("*/complete.json"):
            for name, digest in json.loads(path.read_text())["artifacts"].items():
                if name.startswith(("unpruned_initial", "B_", "D_", "target.npy")):
                    assert file_hash(path.parent / name) == digest
                    inputs[str((path.parent / name).relative_to(REPOSITORY))] = digest
    return {
        "protocol": protocol_for(training_paths),
        "target_ids": [target.target_id for target in load_target_collection().select(num_qubits=(6,))],
        "optimizer_seeds": list(SEEDS),
        "fixture_sha256": file_hash(DEFAULT_TARGET_PATH),
        "inputs": inputs,
        "sources": {str(path.relative_to(REPOSITORY)): file_hash(path) for path in sources},
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "qiskit")},
    }


def main() -> None:
    """Execute all nine q6 targets, two seeds and both B/D noise-profile branches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=parse_workers, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--training-paths", type=int, default=100)
    args = parser.parse_args()
    manifest = make_manifest(args.training_paths)
    if args.output.exists():
        if not args.resume or json.loads((args.output / "manifest.json").read_text()) != manifest:
            msg = "Use a new output directory or resume with identical protocol, sources and inputs."
            raise ValueError(msg)
    else:
        args.output.mkdir(parents=True)
        write_json(args.output / "manifest.json", manifest)
        for name in manifest["sources"]:
            destination = args.output / "source" / (name + ".txt")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPOSITORY / name).read_bytes())
    jobs = [
        (str(args.output.resolve()), target_id, seed, index, args.training_paths)
        for index, target_id in enumerate(manifest["target_ids"])
        for seed in SEEDS
    ]
    start = time.perf_counter()
    write_json(
        args.output / "execution.json",
        {"jobs": len(jobs), "searches": len(jobs) * 4, "workers": resolve_workers(args.workers, len(jobs))},
    )
    results = {}
    for index, result in completed_jobs(run_target, jobs, workers=args.workers):
        results[index] = result
        write_json(args.output / "results.json", [results[index] for index in sorted(results)])
    write_json(args.output / "wall_time.json", {"seconds_this_invocation": time.perf_counter() - start})


if __name__ == "__main__":
    main()
