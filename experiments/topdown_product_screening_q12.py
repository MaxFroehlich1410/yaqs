# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Screen product starts before unchanged longer twelve-qubit ideal training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

from threadpoolctl import threadpool_limits

from benchmarks.state_preparation.targets import load_target
from experiments import topdown_initialization_matrix as matrix
from experiments import topdown_product_screening as screening

if TYPE_CHECKING:
    from numpy.typing import NDArray

VERSION = "topdown.product_screening_q12.v1"
base = screening.base
support = screening.support
DEFAULT_BUDGET = replace(base.Budget(), max_depth=30, max_cpu_seconds=50400.0)


def sources() -> dict[str, str]:
    """Fingerprint selection, orchestration and the unchanged numerical updates."""
    return {
        **screening.sources(),
        **matrix.sources(),
        str(Path(__file__).relative_to(base.capacity.topdown_mps.ROOT)): support.digest(Path(__file__)),
    }


def prepare_initialization(output: Path, target: NDArray, seed: int, arm: str, count: int, batch_size: int) -> tuple:
    """Seal sampling once and authenticate its target, stream and source on reuse.

    Returns:
        The initialized parameter record and private binary arrays.

    Raises:
        ValueError: If cached selection inputs or provenance differ.
    """
    provenance = {
        "target_sha256": screening.runner.target_digest(target),
        "arm": arm,
        "seed": seed,
        "count": count,
        "batch_size": batch_size,
        "sources": sources(),
    }
    initial = support.load_checkpoint(output / "initialization")
    if initial is None:
        metadata, arrays = screening.initial_parameters(target, seed, arm, count, batch_size)
        support.save_checkpoint(output / "initialization", {**metadata, "provenance": provenance}, arrays)
        initial = support.load_checkpoint(output / "initialization")
    assert initial is not None
    if initial[0]["provenance"] != provenance:
        msg = "Stored screened initialization provenance differs."
        raise ValueError(msg)
    return initial


def run_case(
    output: Path,
    target: NDArray,
    target_id: str,
    seed: int,
    budget: base.Budget,
    *,
    count: int = 1000,
    batch_size: int = 64,
    arm: str = "screened_product",
) -> dict:
    """Grow from a sealed selected product using the original continuous depth call.

    Returns:
        Best ideal circuit, per-depth references and sampling-inclusive CPU cost.

    Raises:
        ValueError: If cached inputs, sources or completed results differ.
    """
    budget.validate(seed)
    target, qubits = base.validate_target(target)
    if (
        arm not in screening.ARMS
        or type(count) is not int
        or count < 1
        or type(batch_size) is not int
        or batch_size < 1
    ):
        msg = "Invalid screening arm or sample/batch counts."
        raise ValueError(msg)
    output = output.resolve()
    manifest = {
        "version": VERSION,
        "target_id": target_id,
        "qubits": qubits,
        "target_sha256": screening.runner.target_digest(target),
        "fixture_sha256": support.digest(base.DEFAULT_TARGET_PATH),
        "arm": arm,
        "seed": seed,
        "budget": asdict(budget),
        "count": count,
        "batch_size": batch_size,
        "sources": sources(),
        "noise_or_pruning": False,
        "patience": "disabled for this bounded initialization comparison",
        "extension": "RandomState(seed+1000+depth); scale=0.001 after depth one",
    }
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        msg = "Screened ideal manifest differs; use a fresh output directory."
        raise ValueError(msg)
    support.write_json(path, manifest)
    initial = prepare_initialization(output, target, seed, arm, count, batch_size)
    rows, previous = [], initial[1]["theta"]
    elapsed = initial[0]["cpu_seconds"]
    best, best_depth, selected = -1.0, 0, None
    stop = "depth_budget"
    for depth in range(1, budget.max_depth + 1):
        folder = output / f"depth_{depth:03d}"
        if depth > 1:
            circuit = base.capacity.continuation.brickwall(qubits, depth, initial_single_qubit_layer=True)
            previous = base.capacity.continuation.legacy.extend(
                previous, circuit.num_params, seed + 1000 + depth, 0.001
            )
        support.write_json(
            output / "progress.json",
            {
                "status": "running",
                "target_id": target_id,
                "arm": arm,
                "seed": seed,
                "depth": depth,
                "depth_progress": str((folder / "progress.json").relative_to(output)),
                "completed_depths": rows,
            },
        )
        record, arrays = base.train_depth(folder, target, previous, depth, budget)
        trace = record["training"]
        previous = arrays["theta"]
        elapsed += trace["stage_cpu_seconds"]
        if record["fidelity"] > best:
            best, best_depth, selected = record["fidelity"], depth, str((folder / "construction").relative_to(output))
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
            "checkpoint": str((folder / "construction").relative_to(output)),
            "checkpoint_seal_sha256": support.digest(folder / "construction/complete.json"),
        })
        print(json.dumps(rows[-1]), flush=True)
        if sources() != manifest["sources"]:
            msg = "Numerical sources changed during screened ideal training."
            raise ValueError(msg)
        if best >= budget.target_fidelity:
            stop = "target_reached"
            break
        if elapsed >= budget.max_cpu_seconds:
            stop = "cpu_budget"
            break
    result = {
        "version": VERSION,
        "target_id": target_id,
        "arm": arm,
        "seed": seed,
        "best_fidelity": best,
        "best_depth": best_depth,
        "selected_checkpoint": selected,
        "stop_reason": stop,
        "status": "target_reached" if stop == "target_reached" else "budget_exhausted",
        "target_fidelity": budget.target_fidelity,
        "total_cpu_seconds": elapsed,
        "initialization_cpu_seconds": initial[0]["cpu_seconds"],
        "initialization_wall_seconds": initial[0]["wall_seconds"],
        "initialization_seal_sha256": support.digest(output / "initialization/complete.json"),
        "num_parameters": 3 * qubits + 9 * (qubits - 1) * best_depth,
        "logical_two_qubit_gates": 3 * (qubits - 1) * best_depth,
        "depth_note": "entangling blocks per bond; not compiled two-qubit depth",
        "completed_depths": rows,
        "noise_or_pruning_performed": False,
        "cpu_accounting_scope": "original sampling plus training/progress/endpoint replay; excludes imports and seals",
        "optimizer_history": "reset once per depth; retained throughout each call; live is observational",
    }
    existing = support.load_checkpoint(output / "final")
    if existing is not None:
        if {k: v for k, v in existing[0].items() if k != "arrays"} != result:
            msg = "Sealed screened result differs from authenticated completed stages."
            raise ValueError(msg)
        support.write_json(output / "progress.json", existing[0])
        return existing[0]
    support.write_json(output / "progress.json", result)
    return support.save_checkpoint(output / "final", result, {})


def prepare_plan(output: Path, budget: base.Budget, references: Path, count: int = 1000, batch_size: int = 64) -> dict:
    """Plan only 27 screened runs and reference the 54 original unscreened results.

    Returns:
        Explicit commands, source hashes, matched targets and audited reference file.

    Raises:
        ValueError: If reference coverage, numerical settings or an existing plan differ.
    """
    budget.validate(base.SEEDS[-1])
    if (
        not math.isclose(budget.target_fidelity, 0.99, rel_tol=0, abs_tol=1e-15)
        or budget.max_depth != 30
        or budget.online_steps != 100
        or budget.batch_steps != 2000
        or type(count) is not int
        or count < 1
        or type(batch_size) is not int
        or batch_size < 1
    ):
        msg = "Require matched 99%, depth-30, 100-online/2000-batch settings and positive sampling counts."
        raise ValueError(msg)
    reference = json.loads(references.read_text(encoding="utf-8"))
    expected = {(t, a, s) for t in matrix.target_ids() for a in matrix.ARMS for s in base.SEEDS}
    rows = reference["runs"]
    if (
        len(rows) != len(expected)
        or {(r["target_id"], r["arm"], r["seed"]) for r in rows} != expected
        or any(
            not r["authenticated"] or not math.isclose(r["target_fidelity"], 0.99, rel_tol=0, abs_tol=1e-15)
            for r in rows
        )
    ):
        msg = "Require all 54 authenticated matching near-zero/single-product references."
        raise ValueError(msg)
    jobs = []
    for target_id in matrix.target_ids():
        for seed in base.SEEDS:
            job_id = f"{target_id}__screened_product__seed{seed}"
            command = [
                sys.executable,
                "-m",
                "experiments.topdown_product_screening_q12",
                "--action",
                "run",
                "--target-id",
                target_id,
                "--seed",
                str(seed),
                "--output",
                str(output.resolve() / "jobs" / job_id),
                "--count",
                str(count),
                "--batch-size",
                str(batch_size),
            ]
            for name, value in asdict(budget).items():
                command.extend(["--" + name.replace("_", "-"), str(value)])
            jobs.append({
                "job_id": job_id,
                "target_id": target_id,
                "arm": "screened_product",
                "seed": seed,
                "command": command,
            })
    plan = {
        "version": VERSION,
        "qubits": 12,
        "budget": asdict(budget),
        "count": count,
        "batch_size": batch_size,
        "sources": sources(),
        "fixture_sha256": support.digest(base.DEFAULT_TARGET_PATH),
        "target_sha256": {
            t: screening.runner.target_digest(load_target(12, t).state_vector_copy()) for t in matrix.target_ids()
        },
        "references": {"file": str(references.resolve()), "sha256": support.digest(references), "runs": rows},
        "jobs": jobs,
        "launched": False,
        "environment": dict.fromkeys(
            (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ),
            "1",
        ),
        "deployment_guards": {"wall_seconds_per_job": 57600, "memory_gib_per_job": 16},
    }
    path = output / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        msg = "Screened matrix plan differs; use a fresh directory."
        raise ValueError(msg)
    support.write_json(path, plan)
    return plan


def main(argv: list[str] | None = None) -> None:
    """Plan without training by default; run only an explicitly named screened case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("plan", "run"), default="plan")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--target-id", choices=matrix.target_ids())
    parser.add_argument("--seed", type=int, choices=base.SEEDS)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    for name, value in asdict(DEFAULT_BUDGET).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args(argv)
    budget = base.Budget(**{name: getattr(args, name) for name in asdict(DEFAULT_BUDGET)})
    with threadpool_limits(limits=1):
        if args.action == "plan":
            if args.references is None:
                parser.error("plan requires the authenticated --references file")
            result = prepare_plan(args.output, budget, args.references, args.count, args.batch_size)
            print(
                json.dumps({
                    "new_runs": len(result["jobs"]),
                    "reused_runs": len(result["references"]["runs"]),
                    "launched": False,
                })
            )
        else:
            if args.target_id is None or args.seed is None:
                parser.error("run requires --target-id and --seed")
            target = load_target(12, args.target_id).state_vector_copy()
            print(
                json.dumps(
                    run_case(
                        args.output,
                        target,
                        args.target_id,
                        args.seed,
                        budget,
                        count=args.count,
                        batch_size=args.batch_size,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
