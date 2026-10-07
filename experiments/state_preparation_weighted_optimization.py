# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Matched ordinary/weighted trajectory optimization on frozen development circuits."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.execution import WorkerSetting, completed_jobs, parse_workers, resolve_workers
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider
from benchmarks.state_preparation.weighted_sampling import FactorSampler
from experiments import state_preparation_pilot as pilot
from experiments import state_preparation_sampling_budget as budget
from experiments import state_preparation_trajectory_diagnostics as development

if TYPE_CHECKING:
    from collections.abc import Sequence

VERSION = "yaqs.weighted_trajectory_optimization.v1"


def run_job(job: development.Job) -> dict[str, Any]:
    """Use the shared optimizer, charging both engine and sampler setup to training."""
    row = development.run_job(job)
    record = cast("dict[str, Any]", pilot.read_record(job.root / row["path"]))
    setup_cpu = sum(s["cpu_seconds"] for s in record["setup"])
    setup_wall = sum(s["wall_seconds"] for s in record["setup"])
    curve = budget.curve(record)
    for point in curve:
        point["training_cpu_seconds"] += setup_cpu
    return {
        **row,
        "sampling": job.sampling,
        "trajectories": job.config.trajectories,
        "setup_cpu_seconds": setup_cpu,
        "setup_wall_seconds": setup_wall,
        "training_cpu_seconds": row["training_cpu_seconds"] + setup_cpu,
        "training_wall_seconds": row["training_wall_seconds"] + setup_wall,
        "evaluation_wall_seconds": sum(e["wall_seconds"] for e in record["evaluations"]),
        "curve": curve,
    }


def report(rows: list[dict[str, Any]]) -> str:
    """Report fixed endpoints, with equal target/replica weights and no winner selection."""
    lines = [
        "# Weighted trajectory optimization: development screen",
        "",
        "Four old development targets, two noise replicas per target; not held-out evidence.",
        "Training includes setup, sampling and updates; exact scoring and artifact logging are separate.",
        "",
        "| Sampling | Method | R | Final exact fidelity (%) | Training CPU (s) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for sampling, variant, count in sorted({(r["sampling"], r["variant"], r["trajectories"]) for r in rows}):
        group = [r for r in rows if (r["sampling"], r["variant"], r["trajectories"]) == (sampling, variant, count)]
        lines.append(
            f"| {sampling} | {variant} | {count} | {100 * np.mean([r['final_fidelity'] for r in group]):.6f} | "
            f"{np.mean([r['training_cpu_seconds'] for r in group]):.6f} |"
        )
    return "\n".join(lines) + "\n"


def run(
    root: Path,
    source_root: Path,
    *,
    config: development.Config | None = None,
    counts: tuple[int, ...] = (32, 128),
    workers: WorkerSetting = "auto",
    resume: bool = False,
    variants: tuple[development.Variant, ...] | None = None,
    provenance: dict[str, str] | None = None,
    samplers: tuple[str, ...] = ("ordinary", "weighted"),
) -> dict[str, Any]:
    """Run the predeclared paired screen; ordinary controls are rerun, never retuned."""
    config = development.Config() if config is None else config
    methods = budget.methods(config) if variants is None else variants
    if (
        not methods
        or len({v.name for v in methods}) != len(methods)
        or any(v.online_updates > config.updates for v in methods)
        or not samplers
        or len(set(samplers)) != len(samplers)
        or not set(samplers) <= {"ordinary", "weighted"}
    ):
        msg = "Invalid experiment methods or sampling policies."
        raise ValueError(msg)
    if not counts or len(set(counts)) != len(counts) or any(type(r) is not int or r < 2 for r in counts):
        msg = "Expected distinct budgets with at least two total paths."
        raise ValueError(msg)
    if not 0 < config.online_updates < config.updates:
        msg = "Hybrid schedule needs nonempty online and batch phases."
        raise ValueError(msg)
    resolve_workers(workers, 1)
    source, cases = development.load_cases(source_root)
    manifest = json.loads(
        json.dumps({
            "version": VERSION,
            "config": asdict(config),
            "counts": counts,
            "methods": [asdict(v) for v in methods],
            "samplers": samplers,
            "extension": provenance,
            "source_root": str(source_root.resolve()),
            "source": source,
            "cases": cases,
            "runtime": budget.runtime(),
            "drivers": {
                name: pilot.digest(Path(module.__file__).read_text(encoding="utf-8"))
                for name, module in (("budget", budget),)
            },
            "driver": pilot.digest(Path(__file__).read_text(encoding="utf-8")),
            "noise": ExactLocalPauliProvider().scientific_dict(),
            "no_factor_probabilities": {
                c["target"]["target_id"]: FactorSampler(
                    pilot.restore_circuit(c["circuit"]), CalibratedLocalDepolarizingProvider()
                ).no_factor_probability
                for c in cases
            },
            "path_budget": "R ordinary paths versus one deterministic plus R-1 conditional paths",
            "pairing": "Same training seed, epoch and maps across optimizers within each sampler and budget",
        })
    )
    if root.exists():
        if not resume or not (root / "manifest.json").exists() or pilot.read_record(root / "manifest.json") != manifest:
            msg = "Existing output requires explicit resume with an identical manifest."
            raise ValueError(msg)
    else:
        if resume:
            msg = "Cannot resume missing output."
            raise ValueError(msg)
        pilot.write_record(root / "manifest.json", manifest)
    identity = pilot.digest(manifest)
    jobs = [
        development.Job(
            root / f"R{count}",
            case,
            replace(config, trajectories=count),
            identity,
            variant=variant,
            replica=replica,
            sampling=sampling,
        )
        for case in cases
        for count in counts
        for sampling in samplers
        for variant in methods
        for replica in config.replicas
    ]
    # Fixed, method-neutral interleaving reduces systematic timing/order effects.
    order = np.random.default_rng(config.root_seed).permutation(len(jobs))
    completed = {}
    for index, row in completed_jobs(run_job, [jobs[i] for i in order], workers=workers):
        row["path"] = str(Path(f"R{row['trajectories']}") / row["path"])
        completed[int(order[index])] = row
        print(
            f"weighted optimization {len(completed)}/{len(jobs)}: "
            f"{row['target_id']} {row['sampling']} {row['variant']} R={row['trajectories']}",
            flush=True,
        )
    result: dict[str, Any] = {
        "status": "complete",
        "manifest_checksum": identity,
        "rows": [completed[i] for i in range(len(jobs))],
    }
    path = root / "result.json"
    if path.exists() and pilot.read_record(path) != result:
        msg = "Completed result differs from authenticated job records."
        raise ValueError(msg)
    pilot.write_record(path, result)
    (root / "summary.md").write_text(report(result["rows"]), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Launch only the fixed matched development experiment, with explicit resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=parse_workers, default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run(args.output_root, args.source_root, workers=args.workers, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
