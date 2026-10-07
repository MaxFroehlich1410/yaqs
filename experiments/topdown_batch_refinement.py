# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compare exact noiseless batch refinements against both current B/D controls.

All four optimizers share the same one-hundred-update online prefix. Fixed
replays use the archived best-backtracking pipeline's saved initial parameters;
the full pipelines grow from each policy's returned parameters. Downstream
pruning, D repair, M=100 noisy fine-tuning, validation and stopping remain frozen.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np

from benchmarks.state_preparation.execution import completed_jobs, parse_workers, resolve_workers
from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH, load_target_collection
from experiments import topdown_depth_search as depth_search
from experiments import topdown_optimizer_stability as stability
from experiments.legacy_topdown_dense import extend
from experiments.legacy_topdown_q6 import REPOSITORY, SEEDS, brickwall_for, write_json
from experiments.legacy_topdown_weighted import file_hash, load_circuit
from experiments.topdown_batch_refinement_policies import (
    ARMIJO,
    BB_MAX_RATE,
    BB_MIN_RATE,
    CURVATURE_TOL,
    LBFGS_HISTORY,
    MAX_BACKTRACKS,
    POLICIES,
    train_noiseless,
)

REFERENCE_ROOT = REPOSITORY / "output/topdown-optimizer-stability-q6-20260919"
CONTROL_POLICIES = ("best_only", "best_backtracking")
PRIMARY_BASELINES = {"B": "best_backtracking", "D": "best_only"}
FIXED_REFERENCE_POLICY = "best_backtracking"


def protocol() -> dict:
    """Freeze the current B/D recipe while replacing only the noiseless batch policy."""
    result = deepcopy(depth_search.protocol_for(100))
    result.pop("baseline")
    result.update({
        "version": 1,
        "policies": list(POLICIES),
        "control_policies": list(CONTROL_POLICIES),
        "primary_baselines": PRIMARY_BASELINES.copy(),
        "construction_endpoint": "best iterate, including the initial parameters, for every policy",
        "construction_changes": "batch phase only; all arms use identical 100-step online prefixes",
        "fixed_inputs": "saved initial_theta of every archived best_backtracking pipeline construction",
        "fixed_reference_policy": FIXED_REFERENCE_POLICY,
        "pipeline_inputs": "each policy's returned parameters plus unchanged deterministic layer initialization",
        "pruning_repair": "unchanged legacy final-iterate fixed-step optimizer, including D repair",
        "noisy_training": "unchanged legacy final-iterate cross-trajectory optimizer, M=100",
        "reference": "fresh best_only and best_backtracking runs must reproduce their respective archived pipelines",
        "gradient_checks": "the shared exact noiseless gradient was independently verified in the preceding study",
        "batch_refinement": {
            "armijo": ARMIJO,
            "maximum_backtracks": MAX_BACKTRACKS,
            "backtracking_factor": 0.5,
            "curvature_tolerance": CURVATURE_TOL,
            "bb_minimum_initial_rate": BB_MIN_RATE,
            "bb_maximum_initial_rate": BB_MAX_RATE,
            "lbfgs_history": LBFGS_HISTORY,
            "control_initial_rate": 0.7,
            "bb_first_rate": 0.7,
            "lbfgs_first_rate": 0.7,
            "lbfgs_initial_rate_with_history": 1.0,
            "lbfgs_inverse_hessian_scale_bounds": [1e-8, 1e8],
            "bb_invalid_curvature_fallback": "previous accepted step, or 0.7 without one; clip to BB bounds",
            "angle_wrapping": False,
            "gradient_tolerance_early_exit": False,
        },
    })
    return result


def construct(
    root: Path,
    target: np.ndarray,
    seed: int,
    depth: int,
    previous: np.ndarray,
    policy: str,
    *,
    initial_theta: np.ndarray | None = None,
) -> dict:
    """Optimize either a saved exact input or a newly extended policy-specific parent."""
    circuit = brickwall_for(target, depth)
    initial = (
        extend(previous, circuit.num_params, seed + 1000 + depth, 0.05 if depth == 1 else 0.001)
        if initial_theta is None
        else initial_theta.copy()
    )
    if initial.shape != (circuit.num_params,):
        msg = "Initial parameter shape does not match the archived circuit."
        raise ValueError(msg)
    name = f"construction_{depth:02d}"
    saved = depth_search.read_checkpoint(root, name)
    if saved is not None:
        if saved["policy"] != policy or not np.array_equal(saved["initial_theta"], initial):
            msg = "Cached construction has a different policy or initial parameter vector."
            raise ValueError(msg)
        return saved
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


def run_fixed(job: tuple[str, str, str, int]) -> dict:
    """Compare all policies on the same saved best-backtracking construction inputs."""
    output, reference, target_id, seed = job
    folder = f"{target_id}__seed{seed}"
    root = Path(output) / "fixed" / folder
    source = Path(reference) / "pipeline" / FIXED_REFERENCE_POLICY / folder / "search"
    saved = stability.load_complete(root)
    if saved is not None:
        return saved
    root.mkdir(parents=True, exist_ok=True)
    target = load_target_collection().load(6, target_id).state_vector_copy()
    np.save(root / "target.npy", target)
    records: list[dict] = []
    for path in sorted(source.glob("construction_??.json")):
        archive = depth_search.read_checkpoint(source, path.stem)
        assert archive is not None
        initial = np.array(archive["initial_theta"])
        for policy in POLICIES:
            record = construct(
                root / policy, target, seed, archive["depth"], np.empty(0), policy, initial_theta=initial
            )
            assert record["circuit"] == archive["circuit"]
            np.testing.assert_array_equal(record["initial_theta"], initial)
            prefix_error = stability.maximum_error(
                record["training"]["fidelity_trace"][:100], archive["training"]["fidelity_trace"][:100]
            )
            assert prefix_error == 0
            np.testing.assert_array_equal(
                record["training"]["online_final_theta"], archive["training"]["online_final_theta"]
            )
            theta_error, trace_error = None, None
            if policy == FIXED_REFERENCE_POLICY:
                parity = stability.assert_parity(record, archive, construction=True)
                theta_error, trace_error = parity["theta_max_error"], parity["trace_max_error"]
            fidelity = stability.independent_fidelity(load_circuit(record), np.array(record["theta"]), target)
            independent_error = abs(fidelity - record["ideal_fidelity"])
            assert independent_error < 1e-10
            records.append({
                **stability.training_summary(record, f"{policy}/{path.name}"),
                "reference_theta_max_error": theta_error,
                "reference_trace_max_error": trace_error,
                "original_theta_max_error": theta_error,
                "original_trace_max_error": trace_error,
                "online_prefix_max_error": prefix_error,
                "independent_returned_fidelity_error": independent_error,
            })
        print(
            "FIXED",
            folder,
            archive["depth"],
            [(item["policy"], round(item["returned_fidelity"], 6)) for item in records[-len(POLICIES) :]],
            flush=True,
        )
    if not records:
        msg = "No best-backtracking reference construction checkpoints found."
        raise ValueError(msg)
    return stability.save_complete(
        root,
        {
            "target_id": target_id,
            "optimizer_seed": seed,
            "reference_policy": FIXED_REFERENCE_POLICY,
            "records": records,
            "gradient_checks": [],
        },
    )


def make_manifest(reference: Path, target_ids: list[str], seeds: list[int]) -> dict:
    """Authenticate both strong controls and every fixed input before running any job."""
    old = json.loads((reference / "manifest.json").read_text())
    for name, digest in old["sources"].items():
        if file_hash(REPOSITORY / name) != digest:
            msg = f"Reference simulation source changed: {name}"
            raise ValueError(msg)
    sources = {REPOSITORY / name for name in old["sources"]}
    sources.update((
        Path(__file__),
        REPOSITORY / "experiments/topdown_batch_refinement_policies.py",
        REPOSITORY / "experiments/topdown_batch_refinement_protocol.md",
        REPOSITORY / "experiments/overnight_state_preparation_20260913_0108/optimizers.py",
    ))
    inputs = {str(reference / "manifest.json"): file_hash(reference / "manifest.json")}
    fixed_inputs = []
    for policy in CONTROL_POLICIES:
        for target_id in target_ids:
            for seed in seeds:
                root = reference / "pipeline" / policy / f"{target_id}__seed{seed}"
                complete = stability.load_complete(root)
                if complete is None:
                    msg = "A selected reference control job is incomplete."
                    raise ValueError(msg)
                inputs[str(root / "complete.json")] = file_hash(root / "complete.json")
                for name, digest in complete["artifacts"].items():
                    if name.startswith("search/") or name in {"target.npy", "selection_locked.json"}:
                        inputs[str(root / name)] = digest
                if policy == FIXED_REFERENCE_POLICY:
                    for path in sorted((root / "search").glob("construction_??.json")):
                        record = depth_search.read_checkpoint(root / "search", path.stem)
                        assert record is not None
                        fixed_inputs.append({
                            "target_id": target_id,
                            "optimizer_seed": seed,
                            "depth": record["depth"],
                            "path": str(path),
                            "sha256": file_hash(path),
                        })
    return {
        "protocol": protocol(),
        "primary_baselines": PRIMARY_BASELINES.copy(),
        "control_policies": list(CONTROL_POLICIES),
        "target_ids": target_ids,
        "optimizer_seeds": seeds,
        "reference": str(reference),
        "fixture_sha256": file_hash(DEFAULT_TARGET_PATH),
        "fixed_inputs": fixed_inputs,
        "inputs": inputs,
        "sources": {str(path.relative_to(REPOSITORY)): file_hash(path) for path in sorted(sources)},
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "qiskit")},
    }


def run_pipeline(job: tuple[str, str, str, int, int, str]) -> dict:
    """Run four separately stopped B/D-profile branches for one construction policy."""
    output, reference, target_id, seed, target_index, policy = job
    folder = f"{target_id}__seed{seed}"
    root = Path(output) / "pipeline" / policy / folder
    saved = stability.load_complete(root)
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
        stability.verify_pipeline(Path(reference) / "pipeline" / policy, root, locked, constructions, candidates)
        if policy in CONTROL_POLICIES
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
    result = stability.save_complete(
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
                stability.training_summary(item, f"search/construction_{item['depth']:02d}.json")
                for item in constructions
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


def main() -> None:
    """Execute fixed-input diagnostics and the full four-policy B/D comparison."""
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
