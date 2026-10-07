# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Fine-tune authenticated unpruned parents at the incumbent's selected depth."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH, load_target
from experiments import legacy_pruning_ablation as pruning
from experiments import legacy_topdown_dense as legacy
from experiments import topdown_backend_support as support
from mqt.yaqs.optimization import StatevectorBackend

if TYPE_CHECKING:
    from numpy.typing import NDArray

VERSION = "topdown.unpruned_finetuning.v1"
ROOT = Path(__file__).resolve().parents[1]


def required(folder: Path) -> tuple[dict, dict[str, NDArray]]:
    """Authenticate a committed stage, rejecting absent or partial parents.

    Returns:
        Metadata and private copies of the saved arrays.

    Raises:
        ValueError: If the stage is absent or its seal fails authentication.
    """
    saved = support.load_checkpoint(folder)
    if saved is None:
        msg = f"Missing committed parent stage: {folder}"
        raise ValueError(msg)
    return saved


def authenticate_input(folder: Path) -> tuple[dict, dict, dict[str, NDArray], NDArray]:
    """Verify all copied parent bytes and the exact fixture before using angles.

    Returns:
        Input receipt, construction record/arrays and the normalized target.

    Raises:
        ValueError: If identity, bytes, settings or target differ from the receipt.
    """
    receipt = json.loads((folder / "input.json").read_text())
    if receipt["version"] != VERSION or receipt["fixture_sha256"] != support.digest(DEFAULT_TARGET_PATH):
        msg = "Input version or target fixture differs."
        raise ValueError(msg)
    for name, digest in receipt["files"].items():
        member = (folder / name).resolve()
        if not member.is_relative_to(folder.resolve()) or support.digest(member) != digest:
            msg = "Parent input checksum mismatch."
            raise ValueError(msg)
    final, _ = required(folder / "parent_final")
    built, arrays = required(folder / "construction")
    manifest = json.loads((folder / "parent_manifest.json").read_text())
    for name, digest in manifest.get("sources", {}).items():
        if support.digest(ROOT / name) != digest:
            msg = f"Frozen parent numerical source differs: {name}"
            raise ValueError(msg)
    config = manifest["config"]
    if (
        final["target_id"] != receipt["target_id"]
        or final["seed"] != receipt["seed"]
        or final["selected_depth"] != receipt["selected_depth"]
        or final["arm"] != "near_zero"
        or final["status"] != "complete"
        or config["seed"] != receipt["seed"]
        or config["qubits"] != 6
        or config["hardware"] != "IBM"
        or config["backend"] != "statevector"
        or config["noisy_steps"] != 200
        or config["noisy_online"] != 100
        or config["training_paths"] != 100
        or manifest["fixture_sha256"] != receipt["fixture_sha256"]
    ):
        msg = "Parent identity or frozen fine-tuning settings differ."
        raise ValueError(msg)
    target = load_target(6, receipt["target_id"]).state_vector_copy()
    circuit = support.restore_circuit(built["circuit"])
    score = StatevectorBackend(circuit, target).metrics(arrays["theta"])[1]
    if abs(score - final["endpoints"]["ideal_before_pruning"]) > 1e-10:
        msg = "Unpruned parent fails independent ideal replay."
        raise ValueError(msg)
    return receipt, built, arrays, target


def prepare(parent: Path, output: Path) -> dict:
    """Copy the 27 selected-depth parents, preserving originals and their seals.

    Returns:
        Frozen job plan with separate historical and new-work scope.

    Raises:
        ValueError: If output exists or any of the nine three-seed panels is missing.
    """
    if output.exists():
        msg = "Prepare requires a fresh output directory."
        raise ValueError(msg)
    jobs = []
    for case in sorted(parent.glob("*__near_zero__seed*")):
        final, _ = required(case / "final")
        training, _ = required(case / "training/final")
        depth = final["selected_depth"]
        selection = case / "training/selection_locked.json"
        if (
            final["selection_sha256"] != support.digest(selection)
            or final["training_seal_sha256"] != support.digest(case / "training/final/complete.json")
            or training["selected_depth"] != depth
        ):
            msg = "Parent selection or downstream seal differs."
            raise ValueError(msg)
        identity = f"{final['target_id']}__seed{final['seed']}"
        folder = output / "inputs" / identity
        for source, destination in (
            (case / "final", "parent_final"),
            (case / "training/final", "parent_training_final"),
            (case / "training" / f"depth_{depth:02d}" / "construction", "construction"),
            (case / "training" / f"depth_{depth:02d}" / "pruning", "parent_pruning"),
            (case / "training" / f"depth_{depth:02d}" / "refinement", "parent_refinement"),
        ):
            required(source)
            shutil.copytree(source, folder / destination)
        shutil.copyfile(selection, folder / "parent_selection.json")
        shutil.copyfile(case / "training/manifest.json", folder / "parent_manifest.json")
        receipt = {
            "version": VERSION,
            "target_id": final["target_id"],
            "seed": final["seed"],
            "selected_depth": depth,
            "original_parent": str(case.resolve()),
            "fixture_sha256": support.digest(DEFAULT_TARGET_PATH),
            "files": {str(p.relative_to(folder)): support.digest(p) for p in sorted(folder.rglob("*")) if p.is_file()},
        }
        support.write_json(folder / "input.json", receipt)
        authenticate_input(folder)
        jobs.append({"job_id": identity, "target_id": final["target_id"], "seed": final["seed"], "depth": depth})
    if (
        len(jobs) != 27
        or len({job["target_id"] for job in jobs}) != 9
        or any(
            {job["seed"] for job in jobs if job["target_id"] == target} != {300, 400, 500}
            for target in {job["target_id"] for job in jobs}
        )
    ):
        msg = "Require all nine targets with seeds 300/400/500."
        raise ValueError(msg)
    plan = {
        "version": VERSION,
        "jobs": jobs,
        "inputs": {job["job_id"]: support.digest(output / "inputs" / job["job_id"] / "input.json") for job in jobs},
        "fine_tuning": pruning.PROTOCOL["fine_tuning"],
        "hardware": "IBM",
        "selection": "incumbent selected depth fixed before this ablation; no depth reselection",
        "comparison": "same unpruned construction parent; change only omission of pruning B",
        "new_construction_updates": 0,
        "evaluation": "exact independent Qiskit density channel; final iterate, no endpoint selection",
        "training_seed": "original optimizer seed + 5000; different circuits have different gate windows",
    }
    support.write_json(output / "plan.json", plan)
    return plan


def run_case(folder: Path, input_folder: Path) -> dict:
    """Resume unchanged noisy updates, then independently assess both endpoints.

    Returns:
        Sealed matched comparison including coverage failures and separate costs.

    Raises:
        ValueError: If input provenance, cached results or archived replay differ.
    """
    receipt, built, arrays, target = authenticate_input(input_folder)
    identity = {"version": VERSION, "input_sha256": support.digest(input_folder / "input.json")}
    manifest = folder / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        msg = "Resume input differs; use a fresh output directory."
        raise ValueError(msg)
    support.write_json(manifest, identity)
    saved = support.load_checkpoint(folder / "final")
    if saved is not None:
        required(folder / "refinement")
        if saved[0]["input_sha256"] != identity["input_sha256"] or saved[0]["refinement_seal_sha256"] != support.digest(
            folder / "refinement/complete.json"
        ):
            msg = "Cached final result refers to changed inputs or refinement."
            raise ValueError(msg)
        return saved[0]
    circuit = support.restore_circuit(built["circuit"])
    initial = arrays["theta"]
    refined = support.load_checkpoint(folder / "refinement")
    if refined is None:
        cpu, wall = time.process_time(), time.perf_counter()
        status, reason = "complete", None
        try:
            theta, training = legacy.train(
                circuit, initial, target, seed=receipt["seed"] + 5000, hardware="IBM", **pruning.PROTOCOL["fine_tuning"]
            )
        except RuntimeError as error:
            if "sampled bank has no jumps" not in str(error):
                raise
            theta, training, status, reason = initial.copy(), {}, "no_jump_training_bank", str(error)
        support.save_checkpoint(
            folder / "refinement",
            {
                **identity,
                "status": status,
                "reason": reason,
                "training": training,
                "cpu_seconds": time.process_time() - cpu,
                "wall_seconds": time.perf_counter() - wall,
            },
            {"theta": theta, "initial": initial},
        )
        refined = required(folder / "refinement")
    if refined[0]["input_sha256"] != identity["input_sha256"] or not np.array_equal(refined[1]["initial"], initial):
        msg = "Cached refinement input differs."
        raise ValueError(msg)
    cpu, wall = time.process_time(), time.perf_counter()
    parent, _ = required(input_folder / "parent_final")
    pruned, pruned_arrays = required(input_folder / "parent_pruning")
    incumbent, incumbent_arrays = required(input_folder / "parent_refinement")
    pruned_circuit = support.restore_circuit(pruned["circuit"])
    before = legacy.matrix_score(circuit, initial, target, "IBM", "tjm")
    ideal = legacy.matrix_score(circuit, initial, target, "IBM", "noiseless")["fidelity"]
    baseline_before = legacy.matrix_score(pruned_circuit, pruned_arrays["theta"], target, "IBM", "tjm")
    baseline_after = legacy.matrix_score(pruned_circuit, incumbent_arrays["theta"], target, "IBM", "tjm")
    replay_errors = [
        abs(ideal - parent["endpoints"]["ideal_before_pruning"]),
        abs(baseline_before["fidelity"] - parent["endpoints"]["noisy_before_fine_tuning"]["fidelity"]),
        abs(baseline_after["fidelity"] - parent["endpoints"]["noisy_after_fine_tuning"]["fidelity"]),
    ]
    if max(replay_errors) > 1e-10:
        msg = "Archived endpoint fails independent density replay."
        raise ValueError(msg)
    after = (
        legacy.matrix_score(circuit, refined[1]["theta"], target, "IBM", "tjm")
        if refined[0]["status"] == "complete"
        else None
    )
    result = {
        **identity,
        "target_id": receipt["target_id"],
        "seed": receipt["seed"],
        "selected_depth": receipt["selected_depth"],
        "status": refined[0]["status"],
        "reason": refined[0]["reason"],
        "refinement_seal_sha256": support.digest(folder / "refinement/complete.json"),
        "ideal_unpruned_before": ideal,
        "noisy_unpruned_before": before,
        "noisy_unpruned_after": after,
        "incumbent_endpoints": parent["endpoints"],
        "maximum_parent_replay_error": max(replay_errors),
        "change_vs_pruned_pp": None if after is None else 100 * (after["fidelity"] - baseline_after["fidelity"]),
        "unpruned_logical_gates": len(circuit.gates),
        "unpruned_logical_two_qubit_gates": sum(len(g.sites) == 2 for g in circuit.gates),
        "pruned_resources": parent["resources"],
        "new_construction_updates": 0,
        "fine_tuning_cpu_seconds": refined[0]["cpu_seconds"],
        "fine_tuning_wall_seconds": refined[0]["wall_seconds"],
        "assessment_cpu_seconds": time.process_time() - cpu,
        "assessment_wall_seconds": time.perf_counter() - wall,
        "historical_pruned_refinement_cpu_seconds": incumbent["cpu_seconds"],
        "historical_construction_cpu_seconds": built["cpu_seconds"],
    }
    cpu = time.process_time()
    result["unpruned_resources"] = support.compile_resources(circuit, refined[1]["theta"])
    result["resource_compilation_cpu_seconds"] = time.process_time() - cpu
    return support.save_checkpoint(folder / "final", result, {})


def main() -> None:
    """Prepare a new panel or execute one authenticated fixed-depth case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.action == "prepare" and args.parent is not None:
            result = prepare(args.parent, args.output)
            print(json.dumps({"jobs": len(result["jobs"]), "new_construction_updates": 0}))
        elif args.action == "run" and args.input is not None:
            result = run_case(args.output, args.input)
            print(json.dumps({key: result[key] for key in ("target_id", "seed", "status", "change_vs_pruned_pp")}))
        else:
            parser.error("Prepare requires --parent; run requires --input.")


if __name__ == "__main__":
    main()
