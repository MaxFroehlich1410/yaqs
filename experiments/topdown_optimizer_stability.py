# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Replay noiseless instability, then compare construction policies with frozen B/D.

Fixed-depth jobs reconstruct every archived input independently of the new
optimizer's output. Pipeline jobs instead grow from each policy's returned
parameters. Noisy fine-tuning, pruning repair, validation and selection retain
the previous protocol. Exact noisy evaluation starts only after selection lock.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qiskit.quantum_info import Statevector

from benchmarks.state_preparation.execution import completed_jobs, parse_workers, resolve_workers
from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH, load_target_collection
from experiments import topdown_depth_search as depth_search
from experiments.legacy_topdown_dense import PRIMITIVES, extend
from experiments.legacy_topdown_q6 import REPOSITORY, SEEDS, brickwall_for, write_json
from experiments.legacy_topdown_weighted import file_hash, load_circuit
from experiments.topdown_optimizer_policies import ARMIJO, MAX_BACKTRACKS, POLICIES, train_noiseless
from mqt.yaqs.optimization import StatevectorBackend

if TYPE_CHECKING:
    from mqt.yaqs.optimization import ParameterizedCircuit

REFERENCE_ROOT = REPOSITORY / "output/topdown-depth-search-q6-20260914-v2"
GRADIENT_CASES = {
    ("tfim_ferro", 400, 3),
    ("haar_random_3", 400, 10),
    ("gaussian_mu0p5_sigma0p1", 300, 1),
    ("tfim_para", 300, 1),
}


def protocol() -> dict:
    """Describe the only permitted construction changes and both comparison stages."""
    result = deepcopy(depth_search.protocol_for(100))
    result.pop("baseline")
    result.update({
        "version": 1,
        "policies": list(POLICIES),
        "construction_endpoint": "policy-dependent: final, best, or best with batch Armijo backtracking",
        "armijo": ARMIJO,
        "max_backtracks": MAX_BACKTRACKS,
        "backtracking_initial_rate": 0.7,
        "backtracking_factor": 0.5,
        "fixed_inputs": "every archived search construction, grown from the preceding ARCHIVED final parameters",
        "pipeline_inputs": "growth from each policy's returned parameters; same deterministic new-layer seeds",
        "pruning_repair": "unchanged legacy final-iterate fixed-step optimizer, including D repair",
        "noisy_training": "unchanged legacy final-iterate cross-trajectory optimizer, M=100",
        "reference": "original_final repeats optional every-depth search; first-90% baseline is not rerun",
        "gradient_checks": sorted([list(item) for item in GRADIENT_CASES]),
        "gradient_check": "online-final checkpoint; central Qiskit finite differences at h=1e-5,1e-6 in 3 directions",
    })
    return result


def independent_fidelity(circuit: ParameterizedCircuit, theta: np.ndarray, target: np.ndarray) -> float:
    """Propagate primitive Qiskit gates without using the optimizer's gate matrices."""
    state = Statevector.from_label("0" * circuit.num_qubits)
    for gate in circuit.gates:
        angle = gate.angle_offset + (gate.angle_scale * theta[gate.param_index] if gate.param_index is not None else 0)
        state = state.evolve(PRIMITIVES[gate.name](angle), qargs=list(gate.sites))
    assert abs(float(np.linalg.norm(state.data)) - 1) < 1e-10
    return float(abs(np.vdot(target, state.data)) ** 2)


def maximum_error(actual: list | np.ndarray, expected: list | np.ndarray) -> float:
    """Check equal array shapes before measuring a maximum absolute difference."""
    actual_array, expected_array = np.asarray(actual), np.asarray(expected)
    if actual_array.shape != expected_array.shape:
        msg = "Reference array shapes differ."
        raise ValueError(msg)
    return float(np.max(abs(actual_array - expected_array))) if actual_array.size else 0.0


def assert_parity(actual: dict, expected: dict, *, construction: bool = False) -> dict:
    """Require unchanged circuit parameters, objective traces and sampled noise banks."""
    assert actual["circuit"] == expected["circuit"]
    theta_error = maximum_error(actual["theta"], expected["theta"])
    assert theta_error <= 1e-12
    trace_error = 0.0
    if actual.get("training") is not None:
        trace_error = maximum_error(actual["training"]["fidelity_trace"], expected["training"]["fidelity_trace"])
        assert trace_error <= 1e-12
        if not construction:
            assert actual["training"]["noise_banks"] == expected["training"]["noise_banks"]
    if "status" in actual:
        assert actual["status"] == expected["status"]
    return {"theta_max_error": theta_error, "trace_max_error": trace_error}


def training_summary(record: dict, checkpoint_file: str) -> dict:
    """Expose policy diagnostics without duplicating the complete parameter trace."""
    training = record["training"]
    return {
        "depth": record["depth"],
        "policy": record["policy"],
        **{
            key: training[key]
            for key in (
                "initial_fidelity",
                "online_final_fidelity",
                "best_fidelity",
                "final_fidelity",
                "returned_fidelity",
                "best_iteration",
                "gradient_evaluations",
                "objective_evaluations",
                "rejected_trials",
            )
        },
        "failed_searches": training["failed_line_searches"],
        "cpu_seconds": record["cpu_seconds"],
        "checkpoint_file": checkpoint_file,
    }


def construct(root: Path, target: np.ndarray, seed: int, depth: int, previous: np.ndarray, policy: str) -> dict:
    """Grow one circuit using the selected noiseless construction optimizer only."""
    name = f"construction_{depth:02d}"
    saved = depth_search.read_checkpoint(root, name)
    if saved is not None:
        if saved["policy"] != policy:
            msg = "Cached construction has a different optimizer policy."
            raise ValueError(msg)
        return saved
    circuit = brickwall_for(target, depth)
    initial = extend(previous, circuit.num_params, seed + 1000 + depth, 0.05 if depth == 1 else 0.001)
    cpu, wall = time.process_time(), time.perf_counter()
    theta, trace = train_noiseless(
        circuit, initial, target, policy=policy, stop=2.0, **depth_search.PROTOCOL["construction"]
    )
    return depth_search.save_checkpoint(
        root,
        name,
        circuit,
        theta,
        {
            "depth": depth,
            "policy": policy,
            "initial_theta": initial.tolist(),
            "ideal_fidelity": trace["returned_fidelity"],
            "training": trace,
            "cpu_seconds": time.process_time() - cpu,
            "wall_seconds": time.perf_counter() - wall,
        },
    )


def load_complete(root: Path) -> dict | None:
    """Reuse a completed job only while every saved artifact remains authenticated."""
    path = root / "complete.json"
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    for name, digest in saved["artifacts"].items():
        if file_hash(root / name) != digest:
            msg = "Completed optimizer-stability artifacts changed."
            raise ValueError(msg)
    return saved


def save_complete(root: Path, result: dict) -> dict:
    """Authenticate all job artifacts before writing its completion marker."""
    result["artifacts"] = {
        str(path.relative_to(root)): file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "complete.json"
    }
    write_json(root / "complete.json", result)
    return result


def gradient_check(circuit: ParameterizedCircuit, theta: np.ndarray, target: np.ndarray, seed: int) -> dict:
    """Audit the exact noiseless loss gradient at the online-to-batch transition."""
    gradient, loss, fidelity = StatevectorBackend(circuit, target).gradient(theta, estimator="independent")
    rng = np.random.default_rng(seed)
    directions = [gradient.copy(), rng.normal(size=len(theta)), rng.normal(size=len(theta))]
    checks = []
    for index, direction in enumerate(directions):
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        normalized = direction / norm
        analytic = float(gradient @ normalized)
        for step in (1e-5, 1e-6):
            difference = (
                independent_fidelity(circuit, theta - step * normalized, target)
                - independent_fidelity(circuit, theta + step * normalized, target)
            ) / (2 * step)
            error = abs(analytic - difference)
            assert error < 1e-7
            checks.append({
                "direction": index,
                "step": step,
                "analytic": analytic,
                "finite_difference": difference,
                "absolute_error": error,
            })
    return {
        "theta": theta.tolist(),
        "loss": loss,
        "fidelity": fidelity,
        "gradient_norm": float(np.linalg.norm(gradient)),
        "checks": checks,
    }


def run_fixed(job: tuple[str, str, str, int]) -> dict:
    """Replay every historical depth from identical archived parents under all policies."""
    output, reference, target_id, seed = job
    folder = f"{target_id}__seed{seed}"
    root, source = Path(output) / "fixed" / folder, Path(reference) / folder / "search"
    saved = load_complete(root)
    if saved is not None:
        return saved
    root.mkdir(parents=True, exist_ok=True)
    target = load_target_collection().load(6, target_id).state_vector_copy()
    np.save(root / "target.npy", target)
    records, gradient_checks = [], []
    previous = np.empty(0)
    for path in sorted(source.glob("construction_??.json")):
        archive = depth_search.read_checkpoint(source, path.stem)
        assert archive is not None
        depth = archive["depth"]
        original = None
        for policy in POLICIES:
            record = construct(root / policy, target, seed, depth, previous, policy)
            summary = training_summary(record, f"{policy}/{path.name}")
            theta_error, trace_error = None, None
            if policy == "original_final":
                parity = assert_parity(record, archive, construction=True)
                theta_error, trace_error = parity["theta_max_error"], parity["trace_max_error"]
                original = record
            elif policy == "best_only":
                assert original is not None
                trace_error = maximum_error(
                    record["training"]["fidelity_trace"], original["training"]["fidelity_trace"]
                )
                assert trace_error == 0
                assert maximum_error(record["training"]["final_theta"], original["theta"]) == 0
            independent = independent_fidelity(load_circuit(record), np.array(record["theta"]), target)
            error = abs(independent - record["ideal_fidelity"])
            assert error < 1e-10
            records.append({
                **summary,
                "original_theta_max_error": theta_error,
                "original_trace_max_error": trace_error,
                "independent_returned_fidelity_error": error,
            })
        if (target_id, seed, depth) in GRADIENT_CASES:
            assert original is not None
            gradient_checks.append({
                "depth": depth,
                **gradient_check(
                    load_circuit(original), np.array(original["training"]["online_final_theta"]), target, seed + depth
                ),
            })
        # Every new replay uses the historical parent, never the newly returned best.
        previous = np.array(archive["theta"])
        print(
            "FIXED",
            folder,
            depth,
            [(item["policy"], round(item["returned_fidelity"], 6)) for item in records[-3:]],
            flush=True,
        )
    if not records:
        msg = "No archived construction checkpoints found."
        raise ValueError(msg)
    return save_complete(
        root, {"target_id": target_id, "optimizer_seed": seed, "records": records, "gradient_checks": gradient_checks}
    )


def verify_pipeline(reference: Path, root: Path, locked: dict, constructions: list[dict], candidates: dict) -> dict:
    """Require original-arm parameter, trajectory-bank and depth-selection parity."""
    errors = []
    source = reference / root.name
    old_locked = json.loads((source / "selection_locked.json").read_text())
    assert locked["branches"] == old_locked["branches"]
    for record in constructions:
        name = f"construction_{record['depth']:02d}"
        expected = depth_search.read_checkpoint(source / "search", name)
        assert expected is not None
        errors.append(assert_parity(record, expected, construction=True))
    bank_count = 0
    pruning_count = 0
    for path in sorted((root / "search").glob("depth_*_pruned.json")):
        record = depth_search.read_checkpoint(root / "search", path.stem)
        expected = depth_search.read_checkpoint(source / "search", path.stem)
        assert record is not None
        assert expected is not None
        errors.append(assert_parity(record, expected))
        pruning_count += 1
    for items in candidates.values():
        for record in items:
            name = f"depth_{record['depth']:02d}_{record['variant']}_{record['hardware']}"
            expected = depth_search.read_checkpoint(source / "search", name)
            assert expected is not None
            errors.append(assert_parity(record, expected))
            if record["validation"] is not None:
                actual_validation = {key: value for key, value in record["validation"].items() if key != "bank_sha256"}
                old_validation = {key: value for key, value in expected["validation"].items() if key != "bank_sha256"}
                assert actual_validation == old_validation
                with (
                    np.load(root / "search" / record["validation"]["bank_file"]) as actual,
                    np.load(source / "search" / expected["validation"]["bank_file"]) as old,
                ):
                    np.testing.assert_array_equal(actual["codes"], old["codes"])
                bank_count += 1
    return {
        "maximum_parameter_error": max(item["theta_max_error"] for item in errors),
        "maximum_trace_error": max(item["trace_max_error"] for item in errors),
        "construction_checkpoints": len(constructions),
        "pruned_checkpoints": pruning_count,
        "validation_banks": bank_count,
        "selection_identical": True,
        "training_banks_identical": True,
    }


def run_pipeline(job: tuple[str, str, str, int, int, str]) -> dict:
    """Run four separately stopped B/D-profile branches for one construction policy."""
    output, reference, target_id, seed, target_index, policy = job
    folder = f"{target_id}__seed{seed}"
    root = Path(output) / "pipeline" / policy / folder
    saved = load_complete(root)
    if saved is not None:
        return saved
    root.mkdir(parents=True, exist_ok=True)
    target_record = load_target_collection().load(6, target_id)
    target = target_record.state_vector_copy()
    np.save(root / "target.npy", target)
    search = root / "search"
    search.mkdir(exist_ok=True)
    branches = {
        (variant, hardware): depth_search.Patience()
        for variant in depth_search.VARIANTS
        for hardware in depth_search.HARDWARE
    }
    constructions, prunings = [], {}
    candidates, decisions = {key: [] for key in branches}, {key: [] for key in branches}
    previous = np.empty(0)
    for depth in range(1, depth_search.PROTOCOL["depth_cap"] + 1):
        active = [key for key, tracker in branches.items() if tracker.stale < depth_search.PROTOCOL["patience"]]
        if not active:
            break
        built = construct(search, target, seed, depth, previous, policy)
        previous = np.array(built["theta"])
        constructions.append(built)
        for variant in depth_search.VARIANTS:
            if not any(key[0] == variant for key in active):
                continue
            pruned = depth_search.prune_checkpoint(search, target, built, variant)
            prunings[depth, variant] = pruned
            for hardware in depth_search.HARDWARE:
                key = variant, hardware
                if key not in active:
                    continue
                item = depth_search.candidate(
                    search, target, seed, target_index, pruned, variant, hardware, baseline=False, training_paths=100
                )
                candidates[key].append(item)
                score = item["validation"]["fidelity"] if item["status"] == "complete" else None
                se = item["validation"]["standard_error"] if score is not None else 0.0
                decision = branches[key].observe(depth, score, se)
                decisions[key].append({"depth": depth, "score": score, "standard_error": se, **decision})
        print(
            "PIPELINE",
            policy,
            folder,
            "depth",
            depth,
            "ideal",
            round(built["ideal_fidelity"], 6),
            "active",
            len(active),
            flush=True,
        )
    locked = {
        "target_id": target_id,
        "optimizer_seed": seed,
        "policy": policy,
        "branches": [
            {
                "variant": variant,
                "hardware": hardware,
                "selected_depth": tracker.best_depth,
                "stop_reason": "patience" if tracker.stale >= depth_search.PROTOCOL["patience"] else "depth_cap",
                "decisions": decisions[variant, hardware],
            }
            for (variant, hardware), tracker in branches.items()
        ],
    }
    write_json(root / "selection_locked.json", locked)
    selection_hash = file_hash(root / "selection_locked.json")
    parity = (
        verify_pipeline(Path(reference), root, locked, constructions, candidates)
        if policy == "original_final"
        else None
    )
    results = []
    for branch in locked["branches"]:
        variant, hardware = str(branch["variant"]), str(branch["hardware"])
        visited = candidates[variant, hardware]
        evaluations = [
            depth_search.evaluate_endpoint(search, item, target) for item in visited if item["status"] == "complete"
        ]
        selected = next((item for item in evaluations if item["depth"] == branch["selected_depth"]), None)
        results.append({
            **branch,
            "selected": selected,
            "evaluations": evaluations,
            "search_cost": depth_search.summarize_cost(
                constructions[: len(visited)], [prunings[item["depth"], variant] for item in visited], visited
            ),
        })
    assert selection_hash == file_hash(root / "selection_locked.json")
    result = save_complete(
        root,
        {
            "target_id": target_id,
            "optimizer_seed": seed,
            "policy": policy,
            "target_generation_seed": target_record.seed,
            "reference_parity": parity,
            "selection_sha256": selection_hash,
            "results": results,
            "constructions": [
                training_summary(item, f"search/construction_{item['depth']:02d}.json") for item in constructions
            ],
        },
    )
    print(
        "COMPLETE",
        policy,
        folder,
        [(item["variant"], item["hardware"], item["selected_depth"]) for item in results],
        flush=True,
    )
    return result


def make_manifest(reference: Path, target_ids: list[str], seeds: list[int]) -> dict:
    """Freeze source dependencies, target fixture and authenticated reference outputs."""
    old = json.loads((reference / "manifest.json").read_text())
    sources = {REPOSITORY / name for name in old["sources"]}
    sources.update((
        Path(__file__),
        REPOSITORY / "experiments/topdown_optimizer_policies.py",
        REPOSITORY / "experiments/topdown_optimizer_stability_protocol.md",
    ))
    inputs = {str(reference / "manifest.json"): file_hash(reference / "manifest.json")}
    for target_id in target_ids:
        for seed in seeds:
            root = reference / f"{target_id}__seed{seed}"
            complete = load_complete(root)
            if complete is None:
                msg = "A selected reference job is incomplete."
                raise ValueError(msg)
            inputs[str(root / "complete.json")] = file_hash(root / "complete.json")
            for name, digest in complete["artifacts"].items():
                if name.startswith("search/") or name in {"target.npy", "selection_locked.json"}:
                    inputs[str(root / name)] = digest
    return {
        "protocol": protocol(),
        "target_ids": target_ids,
        "optimizer_seeds": seeds,
        "reference": str(reference),
        "fixture_sha256": file_hash(DEFAULT_TARGET_PATH),
        "inputs": inputs,
        "sources": {str(path.relative_to(REPOSITORY)): file_hash(path) for path in sorted(sources)},
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "qiskit")},
    }


def main() -> None:
    """Execute fixed-input diagnostics and the full three-policy B/D comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--workers", type=parse_workers, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("fixed", "pipeline", "all"), default="all")
    parser.add_argument("--target", action="append")
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    args = parser.parse_args()
    targets = [item.target_id for item in load_target_collection().select(num_qubits=(6,))]
    if args.target and any(target not in targets for target in args.target):
        parser.error("Unknown six-qubit target ID.")
    selected = [target for target in targets if not args.target or target in args.target]
    seeds = [seed for seed in SEEDS if not args.seed or seed in args.seed]
    reference = args.reference.resolve()
    manifest = make_manifest(reference, selected, seeds)
    if args.output.exists():
        if not args.resume or json.loads((args.output / "manifest.json").read_text()) != manifest:
            msg = "Use a new output directory or resume with identical sources, protocol and reference inputs."
            raise ValueError(msg)
    else:
        args.output.mkdir(parents=True)
        write_json(args.output / "manifest.json", manifest)
        for name in manifest["sources"]:
            destination = args.output / "source" / (name + ".txt")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPOSITORY / name).read_bytes())
    start = time.perf_counter()
    output = str(args.output.resolve())
    stages = ("fixed", "pipeline") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "fixed":
            fixed_jobs = [(output, str(reference), target, seed) for target in selected for seed in seeds]
            job_count, filename = len(fixed_jobs), "fixed_results.json"
            iterator = completed_jobs(run_fixed, fixed_jobs, workers=args.workers)
        else:
            pipeline_jobs = [
                (output, str(reference), target, seed, targets.index(target), policy)
                for policy in POLICIES
                for target in selected
                for seed in seeds
            ]
            job_count, filename = len(pipeline_jobs), "results.json"
            iterator = completed_jobs(run_pipeline, pipeline_jobs, workers=args.workers)
        write_json(
            args.output / (stage + "_execution.json"),
            {"jobs": job_count, "workers": resolve_workers(args.workers, job_count)},
        )
        results = {}
        for index, result in iterator:
            results[index] = result
            write_json(args.output / filename, [results[index] for index in sorted(results)])
    write_json(args.output / (args.stage + "_wall_time.json"), {"seconds_this_invocation": time.perf_counter() - start})


if __name__ == "__main__":
    main()
