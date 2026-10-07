# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Noisy online/cross-batch refinement of authenticated sequential-ideal circuits."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from experiments import legacy_topdown_dense as legacy
from experiments import topdown_backend_support as support
from experiments import topdown_pruning_screen as screen
from experiments import topdown_unpruned_finetuning as parents
from mqt.yaqs.optimization import StatevectorBackend

VERSION = "topdown.sequential_noisy_finetuning.v1"
PROTOCOL: dict = {
    "version": VERSION,
    "parent_method": "sequential_ideal",
    "online_updates": 100,
    "batch_updates": 200,
    "total_updates": 300,
    "online_rate": 0.2,
    "batch_rate": 0.2,
    "decay": 0.0,
    "estimator": "unchanged noisy cross-trajectory update in both phases",
    "hardware": "IBM",
    "sampled_histories": 100,
    "noise_bank": "one fixed bank shared by both phases; no-jump multiplicity weights preserved",
    "coverage": "fail without resampling if enabled noise has no sampled jumps",
    "training_seed": "original optimizer seed + 5000",
    "evaluation": "independent exact Qiskit density at initial, online and final endpoints",
    "returned_iterate": "final iterate after all 300 updates; no exact-test-based selection",
    "new_construction_updates": 0,
    "new_pruning_updates": 0,
}


def authenticate_input(folder: Path) -> tuple:
    """Validate the complete lineage from the original parent to sequential pruning.

    Returns:
        Input receipt, sealed sequential-pruning record/arrays, and target vector.

    Raises:
        ValueError: If bytes, methods, identity or the ideal endpoint differ.
    """
    receipt = json.loads((folder / "input.json").read_text())
    if receipt["version"] != VERSION:
        msg = "Sequential refinement input version differs."
        raise ValueError(msg)
    for name, digest in receipt["files"].items():
        member = (folder / name).resolve()
        if not member.is_relative_to(folder.resolve()) or support.digest(member) != digest:
            msg = "Sequential refinement input checksum mismatch."
            raise ValueError(msg)
    parent, _, _, target = parents.authenticate_input(folder / "parent")
    pruned, arrays = parents.required(folder / "sequential_pruning")
    assessed, _ = parents.required(folder / "sequential_assessment")
    parent_hash = support.digest(folder / "parent/input.json")
    identity = {"protocol": screen.PROTOCOL, "method": "sequential_ideal", "input_sha256": parent_hash}
    if (
        pruned["identity"] != identity
        or assessed["method"] != "sequential_ideal"
        or assessed["input_sha256"] != parent_hash
        or pruned["trace"]["repair_updates"] != 0
        or assessed["pruning_seal_sha256"] != support.digest(folder / "sequential_pruning/complete.json")
        or any(parent[k] != receipt[k] or assessed[k] != receipt[k] for k in ["target_id", "seed", "selected_depth"])
    ):
        msg = "Sequential parent identity or sealed lineage differs."
        raise ValueError(msg)
    circuit = support.restore_circuit(pruned["circuit"])
    if (
        circuit.num_qubits != 6
        or abs(StatevectorBackend(circuit, target).metrics(arrays["theta"])[1] - assessed["ideal_pruned"]["fidelity"])
        > 1e-10
    ):
        msg = "Sequential ideal checkpoint fails independent replay."
        raise ValueError(msg)
    return receipt, pruned, arrays, target


def prepare(parent: Path, output: Path) -> dict:
    """Freeze all nine three-seed sequential-pruning checkpoints without new training.

    Returns:
        The complete 27-job plan and input receipt hashes.

    Raises:
        ValueError: If output exists or the checkpoint panel is incomplete.
    """
    if output.exists():
        msg = "Prepare requires a fresh output directory."
        raise ValueError(msg)
    panel = json.loads((parent / "plan.json").read_text())
    jobs = panel["jobs"]
    targets = {j["target_id"] for j in jobs}
    if (
        len(jobs) != 27
        or len(targets) != 9
        or any({j["seed"] for j in jobs if j["target_id"] == target} != {300, 400, 500} for target in targets)
    ):
        msg = "Require all nine targets and seeds 300/400/500."
        raise ValueError(msg)
    for job in jobs:
        identity = job["job_id"]
        if support.digest(parent / "inputs" / identity / "input.json") != panel["inputs"][identity]:
            msg = "Pruning-screen input receipt differs from its frozen plan."
            raise ValueError(msg)
        folder = output / "inputs" / identity
        source = parent / "retrieved/jobs" / identity / "sequential_ideal"
        for src, dst in [
            (parent / "inputs" / identity, "parent"),
            (source / "pruning", "sequential_pruning"),
            (source / "final", "sequential_assessment"),
        ]:
            shutil.copytree(src, folder / dst)
        receipt = {
            "version": VERSION,
            "target_id": job["target_id"],
            "seed": job["seed"],
            "selected_depth": job["depth"],
            "files": {str(p.relative_to(folder)): support.digest(p) for p in sorted(folder.rglob("*")) if p.is_file()},
        }
        support.write_json(folder / "input.json", receipt)
        authenticate_input(folder)
    plan = {
        "protocol": PROTOCOL,
        "jobs": jobs,
        "inputs": {j["job_id"]: support.digest(output / "inputs" / j["job_id"] / "input.json") for j in jobs},
        "comparison": "historical B has 100 online +100 cross batch; pruning and batch budget differ",
    }
    support.write_json(output / "plan.json", plan)
    return plan


def run_case(folder: Path, input_folder: Path) -> dict:
    """Resume online and batch stages, then independently assess the final noisy endpoint.

    Returns:
        Sealed endpoint scores, stage budgets, resource counts and historical B comparison.

    Raises:
        ValueError: If provenance changes, a bank differs or an endpoint fails replay.
    """
    receipt, pruned, arrays, target = authenticate_input(input_folder)
    identity = {"protocol": PROTOCOL, "input_sha256": support.digest(input_folder / "input.json")}
    manifest = folder / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        msg = "Resume protocol or input differs; use a fresh output directory."
        raise ValueError(msg)
    support.write_json(manifest, identity)
    cached = support.load_checkpoint(folder / "final")
    if cached is not None:
        if cached[0]["input_sha256"] != identity["input_sha256"] or cached[0]["protocol"] != PROTOCOL:
            msg = "Cached final result identity differs."
            raise ValueError(msg)
        for stage in ["online", "batch"]:
            parents.required(folder / stage)
            if cached[0][stage + "_seal_sha256"] != support.digest(folder / stage / "complete.json"):
                msg = "Cached final refers to changed training stages."
                raise ValueError(msg)
        return cached[0]
    circuit = support.restore_circuit(pruned["circuit"])
    initial = arrays["theta"]
    engine = StatevectorBackend(circuit, target)
    seed = receipt["seed"] + 5000
    online = support.load_checkpoint(folder / "online")
    if online is None:
        cpu = time.process_time()
        theta, training = legacy.train(
            circuit,
            initial,
            target,
            steps=PROTOCOL["online_updates"],
            online=PROTOCOL["online_updates"],
            rate=PROTOCOL["online_rate"],
            seed=seed,
            noisy=True,
            crn=True,
            hardware="IBM",
            num_trajectories=PROTOCOL["sampled_histories"],
            compress_no_jump=True,
            record_noise_banks=True,
            require_jump_coverage=True,
            engine=engine,
        )
        support.save_checkpoint(
            folder / "online",
            {"identity": identity, "training": training, "cpu_seconds": time.process_time() - cpu},
            {"initial": initial, "theta": theta},
        )
        online = parents.required(folder / "online")
    if online[0]["identity"] != identity or not np.array_equal(online[1]["initial"], initial):
        msg = "Cached online stage input differs."
        raise ValueError(msg)
    digest = online[0]["training"]["noise_banks"][0]["raw_sha256"]
    refined = support.load_checkpoint(folder / "batch")
    if refined is None:
        cpu = time.process_time()
        theta, training = legacy.train(
            circuit,
            online[1]["theta"],
            target,
            steps=PROTOCOL["batch_updates"],
            online=0,
            rate=PROTOCOL["batch_rate"],
            seed=seed,
            noisy=True,
            crn=True,
            hardware="IBM",
            num_trajectories=PROTOCOL["sampled_histories"],
            compress_no_jump=True,
            record_noise_banks=True,
            require_jump_coverage=True,
            engine=engine,
        )
        if training["noise_banks"][0]["raw_sha256"] != digest:
            msg = "Batch noise bank differs from the online bank."
            raise ValueError(msg)
        support.save_checkpoint(
            folder / "batch",
            {
                "identity": identity,
                "noise_bank_sha256": digest,
                "training": training,
                "cpu_seconds": time.process_time() - cpu,
            },
            {"initial": online[1]["theta"], "theta": theta},
        )
        refined = parents.required(folder / "batch")
    if (
        refined[0]["identity"] != identity
        or refined[0]["noise_bank_sha256"] != digest
        or refined[0]["training"]["noise_banks"][0]["raw_sha256"] != digest
        or not np.array_equal(refined[1]["initial"], online[1]["theta"])
    ):
        msg = "Cached batch stage input or noise bank differs."
        raise ValueError(msg)
    assessed, _ = parents.required(input_folder / "sequential_assessment")
    incumbent, _ = parents.required(input_folder / "parent/parent_final")
    cpu = time.process_time()
    before = legacy.matrix_score(circuit, initial, target, "IBM", "tjm")
    online_score = legacy.matrix_score(circuit, online[1]["theta"], target, "IBM", "tjm")
    after = legacy.matrix_score(circuit, refined[1]["theta"], target, "IBM", "tjm")
    ideal = legacy.matrix_score(circuit, refined[1]["theta"], target, "IBM", "noiseless")
    error = abs(before["fidelity"] - assessed["noisy_pruned"]["fidelity"])
    if error > 1e-10 or abs(ideal["fidelity"] - engine.metrics(refined[1]["theta"])[1]) > 1e-10:
        msg = "Exact endpoint replay failed."
        raise ValueError(msg)
    old = incumbent["endpoints"]["noisy_after_fine_tuning"]
    result = {
        **identity,
        "status": "complete",
        "target_id": receipt["target_id"],
        "seed": receipt["seed"],
        "selected_depth": receipt["selected_depth"],
        "online_updates": online[0]["training"]["online_steps"],
        "batch_updates": refined[0]["training"]["steps"],
        "noise_bank_sha256": digest,
        "online_seal_sha256": support.digest(folder / "online/complete.json"),
        "batch_seal_sha256": support.digest(folder / "batch/complete.json"),
        "noisy_before": before,
        "noisy_after_online": online_score,
        "noisy_after_batch": after,
        "ideal_after": ideal,
        "historical_B_noisy_after": old,
        "change_vs_historical_B_pp": 100 * (after["fidelity"] - old["fidelity"]),
        "change_from_sequential_parent_pp": 100 * (after["fidelity"] - before["fidelity"]),
        "maximum_parent_replay_error": error,
        "logical_total_gates": len(circuit.gates),
        "new_construction_updates": 0,
        "new_pruning_updates": 0,
        "online_cpu_seconds": online[0]["cpu_seconds"],
        "batch_cpu_seconds": refined[0]["cpu_seconds"],
        "assessment_cpu_seconds": time.process_time() - cpu,
    }
    result["resources"] = support.compile_resources(circuit, refined[1]["theta"])
    return support.save_checkpoint(folder / "final", result, {})


def main() -> None:
    """Prepare the 27 checkpoints or run one independent noisy refinement job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "run"), required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.action == "prepare" and args.parent is not None:
            print(json.dumps({"jobs": len(prepare(args.parent, args.output)["jobs"])}))
        elif args.action == "run" and args.input is not None:
            result = run_case(args.output, args.input)
            print(json.dumps({"status": result["status"], "noisy_fidelity": result["noisy_after_batch"]["fidelity"]}))
        else:
            parser.error("Prepare requires --parent; run requires --input.")


if __name__ == "__main__":
    main()
