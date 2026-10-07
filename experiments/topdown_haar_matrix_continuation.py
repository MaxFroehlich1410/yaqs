# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Continue each completed Haar initialization arm without changing its updates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

from threadpoolctl import threadpool_limits

from experiments import topdown_initialization_matrix as matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray

VERSION = "topdown.haar_matrix_continuation.v1"
TARGETS = ("haar_random_1", "haar_random_2", "haar_random_3")
MAX_DEPTH = 30
MAX_CPU_SECONDS = 32400.0
base = matrix.base
support = matrix.support


def sources() -> dict[str, str]:
    """Fingerprint this branch and the original numerical implementation."""
    return {
        **matrix.sources(),
        str(Path(__file__).relative_to(base.capacity.topdown_mps.ROOT)): support.digest(Path(__file__)),
    }


def authenticate(parent: Path, target: NDArray) -> tuple:
    """Check final seals, supervisor completion, inherited input and exact replay.

    Returns:
        Parent manifest, original budget, private parameters and cost references.

    Raises:
        ValueError: If the parent identity, numerical source or checkpoint differs.
    """
    parent = parent.resolve()
    root = parent.parent.parent
    manifest = json.loads((parent / "manifest.json").read_text())
    wrapped = json.loads((parent / "matrix-manifest.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    budget = base.Budget(**manifest["budget"])
    seed, arm, target_id = manifest["seed"], manifest["arm"], manifest["target_id"]
    budget.validate(seed)
    target, qubits = base.validate_target(target)
    job_id = f"{target_id}__{arm}__seed{seed}"
    jobs = [job for job in plan["jobs"] if job["job_id"] == job_id]
    if (
        parent.parent.name != "jobs"
        or parent.name != job_id
        or len(jobs) != 1
        or manifest["version"] != base.VERSION
        or wrapped["version"] != matrix.VERSION
        or plan["sources"] != matrix.sources()
        or manifest["sources"] != base.sources()
        or wrapped["sources"] != matrix.sources()
        or manifest["fixture_sha256"] != support.digest(base.DEFAULT_TARGET_PATH)
        or plan["fixture_sha256"] != manifest["fixture_sha256"]
        or any(item[key] != manifest[key] for item in (wrapped, jobs[0]) for key in ("target_id", "arm", "seed"))
        or plan["budget"] != manifest["budget"]
        or wrapped["budget"] != manifest["budget"]
        or manifest["target_sha256"] != base.capacity.topdown_mps.target_digest(target)
        or wrapped["target_sha256"] != manifest["target_sha256"]
        or plan["target_sha256"][target_id] != manifest["target_sha256"]
    ):
        msg = "Parent identity, target, budget or frozen numerical provenance differs."
        raise ValueError(msg)
    receipt_path = root / "logs" / (job_id + ".process.json")
    receipt = json.loads(receipt_path.read_text())
    claim = json.loads((root / "claims" / job_id / "launch.json").read_text())
    if (
        receipt["status"] != "completed"
        or receipt["exit_code"] != 0
        or receipt["command"] != jobs[0]["command"]
        or receipt["command"] != claim["command"][claim["command"].index("--") + 1 :]
        or json.loads((parent / "matrix-complete.json").read_text())
        != {
            "manifest_sha256": support.digest(parent / "matrix-manifest.json"),
            "final_seal_sha256": support.digest(parent / "final/complete.json"),
        }
    ):
        msg = "Parent lacks authenticated successful supervisor completion."
        raise ValueError(msg)
    saved = support.load_checkpoint(parent / "final")
    if saved is None:
        msg = "Parent final checkpoint is incomplete."
        raise ValueError(msg)
    final, _ = saved
    start_depth = budget.max_depth
    checkpoint = f"depth_{start_depth:03d}/construction"
    rows = final["completed_depths"]
    if (
        final["stop_reason"] != "depth_budget"
        or final["status"] != "budget_exhausted"
        or final["best_depth"] != start_depth
        or final["selected_checkpoint"] != checkpoint
        or [row["depth"] for row in rows] != list(range(1, start_depth + 1))
        or any(final[key] != manifest[key] for key in ("target_id", "arm", "seed"))
        or final["target_fidelity"] != budget.target_fidelity
        or final["best_fidelity"] >= budget.target_fidelity
        or final["noise_or_pruning_performed"]
        or rows[-1]["checkpoint"] != checkpoint
        or support.digest(parent / checkpoint / "complete.json") != rows[-1]["checkpoint_seal_sha256"]
        or abs(final["total_cpu_seconds"] - final["setup_cpu_seconds"] - sum(row["stage_cpu_seconds"] for row in rows))
        > 1e-8
    ):
        msg = "Require the completed depth-budget endpoint and consistent parent costs."
        raise ValueError(msg)
    if start_depth == 1:
        previous, _ = base.initial_parameters(target, arm, seed)
    else:
        before = support.load_checkpoint(parent / rows[-2]["checkpoint"])
        if (
            before is None
            or support.digest(parent / rows[-2]["checkpoint"] / "complete.json") != rows[-2]["checkpoint_seal_sha256"]
        ):
            msg = "Previous parent depth checkpoint is incomplete."
            raise ValueError(msg)
        count = base.capacity.continuation.brickwall(qubits, start_depth, initial_single_qubit_layer=True).num_params
        previous = base.capacity.continuation.legacy.extend(before[1]["theta"], count, seed + 1000 + start_depth, 0.001)
    selected = base.check_depth(parent / f"depth_{start_depth:03d}", target, previous, start_depth)
    if selected is None or abs(selected[0]["fidelity"] - final["best_fidelity"]) > 1e-9:
        msg = "Parent selected endpoint fails exact statevector replay."
        raise ValueError(msg)
    references = {
        "folder": str(parent),
        "job_id": job_id,
        "start_depth": start_depth,
        "initial_fidelity": final["best_fidelity"],
        "historical_cpu_seconds": final["total_cpu_seconds"],
        "historical_wall_seconds": receipt["wall_seconds"],
        "manifest_sha256": support.digest(parent / "manifest.json"),
        "matrix_manifest_sha256": support.digest(parent / "matrix-manifest.json"),
        "final_seal_sha256": support.digest(parent / "final/complete.json"),
        "checkpoint": checkpoint,
        "checkpoint_seal_sha256": support.digest(parent / checkpoint / "complete.json"),
        "supervisor_receipt_sha256": support.digest(receipt_path),
    }
    return manifest, budget, selected[1]["theta"].copy(), references


def run(
    output: Path, parent: Path, target: NDArray, max_depth: int = MAX_DEPTH, max_cpu_seconds: float = MAX_CPU_SECONDS
) -> dict:
    """Extend from the sealed parent, reusing only completed continuous calls.

    Returns:
        Sealed best endpoint, separate historical/new costs and stopping reason.

    Raises:
        ValueError: If paths overlap, budgets/provenance change or recovery is unsafe.
    """
    output, parent = output.resolve(), parent.resolve()
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        msg = "Continuation output must be separate from its parent."
        raise ValueError(msg)
    started = time.process_time()
    original, old_budget, previous, references = authenticate(parent, target)
    budget = replace(old_budget, max_depth=max_depth, max_cpu_seconds=max_cpu_seconds)
    seed, start_depth = original["seed"], references["start_depth"]
    budget.validate(seed)
    if max_depth <= start_depth:
        msg = "Continuation depth budget must exceed the parent depth."
        raise ValueError(msg)
    target, qubits = base.validate_target(target)
    manifest = {
        "version": VERSION,
        "target_id": original["target_id"],
        "arm": original["arm"],
        "seed": seed,
        "target_sha256": original["target_sha256"],
        "budget": asdict(budget),
        "parent": references,
        "sources": sources(),
        "optimizer": "unchanged inherited hybrid call; curvature resets at each new depth",
        "extension": "unchanged RandomState(seed+1000+depth), scale=0.001",
        "patience": "disabled in this bounded capacity continuation",
    }
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        msg = "Continuation manifest differs; use a fresh directory."
        raise ValueError(msg)
    support.write_json(path, manifest)
    setup = output / "setup.json"
    if not setup.exists():
        support.write_json(setup, {"cpu_seconds": time.process_time() - started})
    setup_cpu = json.loads(setup.read_text())["cpu_seconds"]
    best, best_depth, selected = references["initial_fidelity"], start_depth, "parent"
    elapsed, rows, stop = setup_cpu, [], "depth_budget"
    for depth in range(start_depth + 1, max_depth + 1):
        circuit = base.capacity.continuation.brickwall(qubits, depth, initial_single_qubit_layer=True)
        previous = base.capacity.continuation.legacy.extend(previous, circuit.num_params, seed + 1000 + depth, 0.001)
        folder = output / f"depth_{depth:03d}"
        support.write_json(
            output / "progress.json",
            {
                "status": "running",
                "depth": depth,
                "best_fidelity": best,
                "new_cpu_seconds": elapsed,
                "completed_depths": rows,
            },
        )
        record, arrays = base.train_depth(folder, target, previous, depth, budget)
        trace, previous = record["training"], arrays["theta"]
        elapsed += trace["stage_cpu_seconds"]
        if record["fidelity"] > best:
            best, best_depth, selected = record["fidelity"], depth, f"depth_{depth:03d}/construction"
        rows.append({
            "depth": depth,
            "fidelity": record["fidelity"],
            "updates": trace["steps"],
            "online_sweeps": trace["online_sweeps"],
            "batch_updates": trace["steps"] - trace["online_sweeps"],
            "gradient_norm": trace["gradient_norm"],
            "stage_cpu_seconds": trace["stage_cpu_seconds"],
            "stage_wall_seconds": trace["stage_wall_seconds"],
            "threshold_crossings": trace["threshold_crossings"],
            "checkpoint": f"depth_{depth:03d}/construction",
            "checkpoint_seal_sha256": support.digest(folder / "construction/complete.json"),
        })
        print(json.dumps(rows[-1]), flush=True)
        if sources() != manifest["sources"]:
            msg = "Continuation numerical sources changed during execution."
            raise ValueError(msg)
        if best >= budget.target_fidelity:
            stop = "target_reached"
            break
        if elapsed >= budget.max_cpu_seconds:
            stop = "cpu_budget"
            break
    result = {
        "target_id": original["target_id"],
        "arm": original["arm"],
        "seed": seed,
        "status": "target_reached" if stop == "target_reached" else "budget_exhausted",
        "stop_reason": stop,
        "best_depth": best_depth,
        "best_fidelity": best,
        "selected_checkpoint": selected,
        "initial_fidelity": references["initial_fidelity"],
        "target_fidelity": budget.target_fidelity,
        "completed_depths": rows,
        "new_cpu_seconds": elapsed,
        "setup_cpu_seconds": setup_cpu,
        "historical_cpu_seconds": references["historical_cpu_seconds"],
        "total_preparation_cpu_seconds": references["historical_cpu_seconds"] + elapsed,
        "logical_two_qubit_gates": 3 * (qubits - 1) * best_depth,
        "num_parameters": 3 * qubits + 9 * (qubits - 1) * best_depth,
        "noise_or_pruning_performed": False,
        "cpu_accounting_scope": (
            "new parent authentication plus preserved training-stage costs; historical costs reported separately"
        ),
    }
    saved = support.load_checkpoint(output / "final")
    if saved is not None:
        if {k: v for k, v in saved[0].items() if k != "arrays"} != result:
            msg = "Sealed final result differs from the completed continuation."
            raise ValueError(msg)
        result = saved[0]
    else:
        result = support.save_checkpoint(output / "final", result, {})
    support.write_json(output / "progress.json", result)
    return result


def prepare_plan(
    output: Path, parent: Path, max_depth: int = MAX_DEPTH, max_cpu_seconds: float = MAX_CPU_SECONDS
) -> dict:
    """Authenticate all eighteen starts and prepare explicit non-launching commands.

    Returns:
        Frozen job commands, numerical sources, parent references and resource guards.

    Raises:
        ValueError: If coverage, settings, source snapshots or an existing plan differ.
    """
    output, parent = output.resolve(), parent.resolve()
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        msg = "Continuation output must be separate from its parent."
        raise ValueError(msg)
    plan = json.loads((parent / "plan.json").read_text())
    selected = [job for job in plan["jobs"] if job["target_id"] in TARGETS]
    expected = {(target, arm, seed) for target in TARGETS for arm in matrix.ARMS for seed in base.SEEDS}
    if len(selected) != len(expected) or {(j["target_id"], j["arm"], j["seed"]) for j in selected} != expected:
        msg = "Require all eighteen original Haar initialization cases."
        raise ValueError(msg)
    jobs = []
    for job in selected:
        folder = parent / "jobs" / job["job_id"]
        target = matrix.load_target(12, job["target_id"]).state_vector_copy()
        _, old, _, references = authenticate(folder, target)
        budget = replace(old, max_depth=max_depth, max_cpu_seconds=max_cpu_seconds)
        budget.validate(job["seed"])
        if references["start_depth"] != 16 or max_depth <= 16:
            msg = "The matrix continuation requires depth-16 parents and a larger limit."
            raise ValueError(msg)
        command = [
            sys.executable,
            "-m",
            "experiments.topdown_haar_matrix_continuation",
            "--action",
            "run",
            "--parent",
            str(folder.resolve()),
            "--output",
            str((output / "jobs" / job["job_id"]).resolve()),
            "--max-depth",
            str(max_depth),
            "--max-cpu-seconds",
            str(max_cpu_seconds),
        ]
        jobs.append({
            **{k: job[k] for k in ("job_id", "target_id", "arm", "seed")},
            "parent": references,
            "budget": asdict(budget),
            "command": command,
        })
    result = {
        "version": VERSION,
        "parent": str(parent.resolve()),
        "sources": sources(),
        "jobs": jobs,
        "environment": plan["environment"],
        "deployment_guards": {"wall_seconds_per_job": 43200, "memory_gib_per_job": 16},
        "launched": False,
    }
    path = output / "plan.json"
    if path.exists() and json.loads(path.read_text()) != result:
        msg = "Continuation plan differs; use a fresh directory."
        raise ValueError(msg)
    support.write_json(path, result)
    return result


def main(argv: list[str] | None = None) -> None:
    """Prepare the matrix or explicitly run one authenticated continuation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("plan", "run"), default="plan")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    parser.add_argument("--max-cpu-seconds", type=float, default=MAX_CPU_SECONDS)
    args = parser.parse_args(argv)
    with threadpool_limits(limits=1):
        if args.action == "plan":
            result = prepare_plan(args.output, args.parent, args.max_depth, args.max_cpu_seconds)
        else:
            manifest = json.loads((args.parent / "manifest.json").read_text())
            if (
                manifest["target_id"] not in TARGETS
                or manifest["arm"] not in matrix.ARMS
                or manifest["seed"] not in base.SEEDS
            ):
                parser.error("Require an original Haar target, initializer and seed.")
            target = matrix.load_target(12, manifest["target_id"]).state_vector_copy()
            result = run(args.output, args.parent, target, args.max_depth, args.max_cpu_seconds)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
