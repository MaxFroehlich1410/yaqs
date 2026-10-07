# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compare existing initializations across every twelve-qubit benchmark target."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.state_preparation.targets import load_target, load_target_collection
from experiments import topdown_random_mps_initialization as base

if TYPE_CHECKING:
    from numpy.typing import NDArray

VERSION = "topdown.initialization_matrix.v1"
ARMS = ("near_zero", "random_product")
DIAGNOSTIC_SEED = 730001
support = base.support


def sources() -> dict[str, str]:
    """Fingerprint the wrapper and unchanged numerical implementation."""
    return {
        **base.sources(),
        str(Path(__file__).relative_to(base.capacity.topdown_mps.ROOT)): support.digest(Path(__file__)),
    }


def target_ids() -> tuple[str, ...]:
    """Read every twelve-qubit target identifier from the benchmark fixture."""
    return tuple(t.target_id for t in load_target_collection().select(num_qubits=(12,)))


def overlap_samples(target: NDArray, count: int = 1000, seed: int = DIAGNOSTIC_SEED) -> tuple[dict, dict]:
    """Measure product-state overlaps without selecting any training start.

    Returns:
        Distribution statistics and compact samples, polar angles and phases.

    Raises:
        ValueError: If the sample count, seed or normalized target is invalid.
    """
    target, qubits = base.validate_target(target)
    if type(count) is not int or count < 2 or type(seed) is not int or not 0 <= seed < 2**32:
        msg = "Require at least two overlap samples and a valid integer seed."
        raise ValueError(msg)
    draw = np.random.default_rng(seed).random((count, 2, qubits))
    probability, phase = draw[:, 0], 2 * np.pi * draw[:, 1]
    vectors = np.ones((count, 1), dtype=np.complex128)
    for site in range(qubits):
        local = np.column_stack((
            np.sqrt(1 - probability[:, site]),
            np.exp(1j * phase[:, site]) * np.sqrt(probability[:, site]),
        ))
        vectors = (local[:, :, None] * vectors[:, None, :]).reshape(count, -1)
    fidelity = np.abs(vectors @ target.conj()) ** 2
    zero = float(abs(target[0]) ** 2)
    return {
        "count": count,
        "seed": seed,
        "qubits": qubits,
        "scope": "product layer before entangling gates; diagnostic only, no start selection",
        "theoretical_mean": 2.0**-qubits,
        "sample_mean": float(fidelity.mean()),
        "sample_mean_standard_error": float(fidelity.std(ddof=1) / np.sqrt(count)),
        "sample_median": float(np.median(fidelity)),
        "quantiles": {str(q): float(np.quantile(fidelity, q)) for q in (0.05, 0.25, 0.75, 0.95)},
        "zero_state_fidelity": zero,
        "fraction_above_zero": float(np.mean(fidelity > zero)),
        "training_seeds": list(base.SEEDS),
        "training_seed_selection": "predetermined; diagnostic samples never select optimizer starts",
    }, {"fidelity": fidelity, "polar": np.arccos(1 - 2 * probability), "azimuth": phase}


def diagnose(output: Path, target: NDArray, target_id: str, count: int = 1000) -> dict:
    """Save or authenticate a separate, deterministic overlap-distribution audit.

    Returns:
        The sealed statistics with sample arrays stored once.

    Raises:
        ValueError: If a saved diagnostic has different inputs or provenance.
    """
    fingerprint = sources()
    record, arrays = overlap_samples(target, count)
    expected = {
        **record,
        "version": VERSION,
        "target_id": target_id,
        "target_sha256": base.capacity.topdown_mps.target_digest(target),
        "sources": fingerprint,
    }
    saved = support.load_checkpoint(output)
    if saved is not None:
        if {k: v for k, v in saved[0].items() if k != "arrays"} != expected or any(
            not np.array_equal(saved[1][k], v) for k, v in arrays.items()
        ):
            msg = "Overlap diagnostic inputs or provenance differ."
            raise ValueError(msg)
        return saved[0]
    return support.save_checkpoint(output, expected, arrays)


def authenticate_reference(parent: Path, target: NDArray, budget: base.Budget) -> list[dict]:
    """Authenticate six existing random-MPS runs without retraining or changing them.

    Returns:
        Six references to complete matched histories and successful supervisor exits.

    Raises:
        ValueError: If any required source, setting, seal, trajectory or exit differs.
    """
    parent = parent.resolve()
    rows = []
    for seed in base.SEEDS:
        for arm in ARMS:
            folder = parent / "jobs" / f"{arm}__seed{seed}"
            manifest = json.loads((folder / "manifest.json").read_text())
            receipt = json.loads((parent / "logs" / f"{folder.name}.process.json").read_text())
            claim = json.loads((parent / "claims" / folder.name / "launch.json").read_text())
            if (
                manifest["version"] != base.VERSION
                or manifest["target_id"] != base.TARGET_ID
                or manifest["target_sha256"] != base.capacity.topdown_mps.target_digest(target)
                or manifest["fixture_sha256"] != support.digest(base.DEFAULT_TARGET_PATH)
                or manifest["sources"] != base.sources()
                or manifest["budget"] != asdict(budget)
                or manifest["arm"] != arm
                or manifest["seed"] != seed
                or receipt["status"] != "completed"
                or receipt["exit_code"] != 0
                or receipt["command"] != claim["command"][claim["command"].index("--") + 1 :]
            ):
                msg = "Existing random-MPS reference differs or lacks successful completion."
                raise ValueError(msg)
            completed = support.load_checkpoint(folder / "final")
            initial = support.load_checkpoint(folder / "initialization")
            if completed is None or initial is None:
                msg = "Reference checkpoints are incomplete."
                raise ValueError(msg)
            final, _ = completed
            previous, metadata = base.initial_parameters(target, arm, seed)
            if not np.array_equal(previous, initial[1]["theta"]) or any(
                initial[0][k] != v for k, v in metadata.items()
            ):
                msg = "Reference initial parameters differ; authenticate on the original numerical platform."
                raise ValueError(msg)
            cpu = initial[0]["setup_cpu_seconds"]
            for depth, row in enumerate(final["completed_depths"], 1):
                circuit = base.capacity.continuation.brickwall(
                    len(target).bit_length() - 1, depth, initial_single_qubit_layer=True
                )
                if depth > 1:
                    previous = base.capacity.continuation.legacy.extend(
                        previous, circuit.num_params, seed + 1000 + depth, 0.001
                    )
                checked = base.check_depth(folder / f"depth_{depth:03d}", target, previous, depth)
                if (
                    checked is None
                    or support.digest(folder / row["checkpoint"] / "complete.json") != row["checkpoint_seal_sha256"]
                    or row["depth"] != depth
                    or row["fidelity"] != checked[0]["fidelity"]
                ):
                    msg = "Reference depth history is incomplete or inconsistent."
                    raise ValueError(msg)
                previous = checked[1]["theta"]
                cpu += row["stage_cpu_seconds"]
            selected = support.load_checkpoint(folder / final["selected_checkpoint"])
            if (
                selected is None
                or abs(cpu - final["total_cpu_seconds"]) > 1e-8
                or selected[0]["fidelity"] != final["best_fidelity"]
                or final["stop_reason"] != "target_reached"
                or final["best_fidelity"] < budget.target_fidelity
            ):
                msg = "Reference final summary fails authentication."
                raise ValueError(msg)
            rows.append({
                "target_id": base.TARGET_ID,
                "arm": arm,
                "seed": seed,
                "job_id": f"{base.TARGET_ID}__{arm}__seed{seed}",
                "folder": str(folder),
                "final_seal_sha256": support.digest(folder / "final/complete.json"),
                "manifest_sha256": support.digest(folder / "manifest.json"),
                "supervisor_receipt_sha256": support.digest(parent / "logs" / f"{folder.name}.process.json"),
                "fidelity": final["best_fidelity"],
                "depth": final["best_depth"],
                "cpu_seconds": final["total_cpu_seconds"],
                "wall_seconds": receipt["wall_seconds"],
            })
    return rows


def prepare_plan(output: Path, budget: base.Budget, reuse: Path | None = None, count: int = 1000) -> dict:
    """Prepare the complete matrix, authenticating references before omitting jobs.

    Returns:
        Explicit independent commands and immutable reference provenance.

    Raises:
        ValueError: If an existing plan or reused experiment differs.
    """
    budget.validate(base.SEEDS[-1])
    if type(count) is not int or count < 2:
        msg = "Require at least two overlap samples."
        raise ValueError(msg)
    references = (
        []
        if reuse is None
        else authenticate_reference(reuse, load_target(12, base.TARGET_ID).state_vector_copy(), budget)
    )
    targets = target_ids()
    omitted = {(r["target_id"], r["arm"], r["seed"]) for r in references}
    jobs, diagnostic_jobs, targets_hashes = [], [], {}
    for target_id in targets:
        target = load_target(12, target_id).state_vector_copy()
        targets_hashes[target_id] = base.capacity.topdown_mps.target_digest(target)
        diagnostic_jobs.append({
            "target_id": target_id,
            "command": [
                sys.executable,
                "-m",
                "experiments.topdown_initialization_matrix",
                "--action",
                "overlaps",
                "--target-id",
                target_id,
                "--output",
                str(output.resolve() / "overlaps" / target_id),
                "--overlap-samples",
                str(count),
            ],
        })
        for seed in base.SEEDS:
            for arm in ARMS:
                if (target_id, arm, seed) in omitted:
                    continue
                job_id = f"{target_id}__{arm}__seed{seed}"
                command = [
                    sys.executable,
                    "-m",
                    "experiments.topdown_initialization_matrix",
                    "--action",
                    "run",
                    "--target-id",
                    target_id,
                    "--arm",
                    arm,
                    "--seed",
                    str(seed),
                    "--output",
                    str(output.resolve() / "jobs" / job_id),
                ]
                for name, value in asdict(budget).items():
                    command.extend(["--" + name.replace("_", "-"), str(value)])
                jobs.append({"job_id": job_id, "target_id": target_id, "arm": arm, "seed": seed, "command": command})
    plan = {
        "version": VERSION,
        "qubits": 12,
        "target_ids": targets,
        "target_sha256": targets_hashes,
        "fixture_sha256": support.digest(base.DEFAULT_TARGET_PATH),
        "sources": sources(),
        "budget": asdict(budget),
        "jobs": jobs,
        "references": references,
        "diagnostic_jobs": diagnostic_jobs,
        "overlap_samples": count,
        "overlap_seed": DIAGNOSTIC_SEED,
        "overlap_selection": "diagnostic only; fixed optimizer seeds are not selected from samples",
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
        "deployment_guards": {"wall_seconds_per_job": 28800, "memory_gib_per_job": 16},
        "launched": False,
    }
    plan = json.loads(json.dumps(plan))
    path = output / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        msg = "Existing initialization matrix plan differs."
        raise ValueError(msg)
    support.write_json(path, plan)
    return plan


def run(output: Path, target: NDArray, target_id: str, arm: str, seed: int, budget: base.Budget) -> dict:
    """Delegate all numerical updates to the tested driver and seal wrapper provenance.

    Returns:
        The unchanged numerical driver's sealed final result.

    Raises:
        ValueError: If wrapper settings or numerical source provenance differ.
    """
    if arm not in ARMS:
        msg = "Unknown matrix initialization arm."
        raise ValueError(msg)
    budget.validate(seed)
    target, _ = base.validate_target(target)
    receipt = {
        "version": VERSION,
        "target_id": target_id,
        "target_sha256": base.capacity.topdown_mps.target_digest(target),
        "arm": arm,
        "seed": seed,
        "budget": asdict(budget),
        "sources": sources(),
        "scope": "unchanged noiseless driver; no noise, pruning or seed screening",
    }
    path = output / "matrix-manifest.json"
    if path.exists() and json.loads(path.read_text()) != receipt:
        msg = "Existing matrix worker provenance differs."
        raise ValueError(msg)
    support.write_json(path, receipt)
    result = base.run_case(output, target, target_id, arm, seed, budget)
    if sources() != receipt["sources"]:
        msg = "Matrix numerical sources changed during training."
        raise ValueError(msg)
    support.write_json(
        output / "matrix-complete.json",
        {
            "manifest_sha256": support.digest(path),
            "final_seal_sha256": support.digest(output / "final/complete.json"),
        },
    )
    return result


def main(argv: list[str] | None = None) -> None:
    """Prepare explicit jobs, or run one target/arm or one separate overlap audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("plan", "run", "overlaps"), default="plan")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-id", choices=target_ids())
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=base.SEEDS)
    parser.add_argument("--reuse-random-mps", type=Path)
    parser.add_argument("--overlap-samples", type=int, default=1000)
    for name, value in asdict(base.Budget()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args(argv)
    budget = base.Budget(**{name: getattr(args, name) for name in asdict(base.Budget())})
    with threadpool_limits(limits=1):
        if args.action == "plan":
            result = prepare_plan(args.output, budget, args.reuse_random_mps, args.overlap_samples)
            print(json.dumps({"jobs": len(result["jobs"]), "reused": len(result["references"]), "launched": False}))
        else:
            if args.target_id is None or (args.action == "run" and (args.arm is None or args.seed is None)):
                parser.error("An explicit target is required; training also requires arm and seed.")
            target = load_target(12, args.target_id).state_vector_copy()
            started = time.perf_counter()
            if args.action == "overlaps":
                result = diagnose(args.output, target, args.target_id, args.overlap_samples)
            else:
                result = run(args.output, target, args.target_id, args.arm, args.seed, budget)
            print(
                json.dumps({
                    "action": args.action,
                    "elapsed_wall_seconds": time.perf_counter() - started,
                    "result": result,
                }),
                flush=True,
            )


if __name__ == "__main__":
    main()
