# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Paired development-only optimizer/trajectory-budget screen using existing kernels."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from benchmarks.state_preparation.execution import WorkerSetting, completed_jobs, parse_workers, resolve_workers
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider
from experiments import state_preparation_pilot as pilot
from experiments import state_preparation_trajectory_comparison as comparison
from experiments import state_preparation_trajectory_diagnostics as development

if TYPE_CHECKING:
    from collections.abc import Sequence

VERSION = "yaqs.statevector_sampling_budget.v1"


def runtime() -> dict[str, str]:
    """Return the historical development runtime without modifying its sources."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "qiskit": version("qiskit"),
        "code": pilot.digest([
            comparison.runtime_fingerprint(),
            Path(development.__file__).read_text(encoding="utf-8"),
        ]),
    }


def methods(config: development.Config) -> tuple[development.Variant, ...]:
    """Return the three primary arms, keeping the previously selected Adam rate."""
    return (
        development.Variant("all_online", config.updates),
        development.Variant("hybrid_batch_0p3", config.online_updates, rate=0.3),
        development.Variant("trajectory_adam_0p02", 0, tail="adam", rate=0.02),
    )


def reuse_tail(
    root: Path,
    source_root: Path,
    provenance: dict[str, Any],
    cases: list[dict[str, Any]],
    config: development.Config,
) -> tuple[dict[str, Any], dict[tuple[str, str, int], dict[str, Any]]]:
    """Authenticate compatible completed tail records without writing to their root."""
    manifest = pilot.read_record(root / "manifest.json")
    result = cast("dict[str, Any]", pilot.read_record(root / "result.json"))
    expected = json.loads(
        json.dumps({
            "version": development.VERSION,
            "study": "tail",
            "variants": [asdict(v) for v in development.variants("tail", config)],
            "config": asdict(config),
            "include_cross": False,
            "backend": "statevector",
            "cases": cases,
            "source_root": str(source_root.resolve()),
            "source": provenance,
            "noise": ExactLocalPauliProvider().scientific_dict(),
            "runtime": runtime(),
        })
    )
    if manifest != expected or result["status"] != "complete" or result["manifest_checksum"] != pilot.digest(manifest):
        msg = "Tail reuse requires matching complete sources, configuration and runtime."
        raise ValueError(msg)
    identity = pilot.digest(manifest)
    expected_jobs = len(cases) * len(config.replicas) * 4
    if len(result["rows"]) != expected_jobs or len({row["path"] for row in result["rows"]}) != expected_jobs:
        msg = "Invalid tail result inventory."
        raise ValueError(msg)
    selected = {}
    for case in cases:
        for variant in methods(config)[:2]:
            for replica in config.replicas:
                job = development.Job(root, case, config, identity, variant=variant, replica=replica)
                record = pilot.read_record(job.path)
                if record["status"] != "complete" or record["completed"] != config.updates:
                    msg = "Cannot reuse an unfinished tail checkpoint."
                    raise ValueError(msg)
                # Completed-only optimizer validation is read-only; never resume old work here.
                row = development.run_job(job)
                if row not in result["rows"]:
                    msg = "Tail row/checkpoint checksum mismatch."
                    raise ValueError(msg)
                key = (case["target"]["target_id"], variant.name, replica)
                selected[key] = {"root": str(root.resolve()), "row": row}
    return {"root": str(root.resolve()), "manifest": identity, "result": pilot.digest(result)}, selected


def curve(record: dict[str, Any], *, exact: bool = False) -> list[dict[str, float | int]]:
    """Pair scored iterates with cumulative training CPU; never select best iterates."""
    cumulative = [0.0]
    for step in record["trace"]:
        cost = step["cpu_seconds"] if exact else step["sampling"]["cpu_seconds"] + step["update_cost"]["cpu_seconds"]
        cumulative.append(cumulative[-1] + cost)
    return [
        {
            "update": evaluation["update"] if exact else evaluation["step"],
            "fidelity": evaluation["exact_noisy_fidelity"] if exact else evaluation["fidelity"],
            "training_cpu_seconds": cumulative[evaluation["update"] if exact else evaluation["step"]],
        }
        for evaluation in record["evaluations"]
    ]


@dataclass(frozen=True)
class Job:
    """A new refinement, read-only reused refinement, or unique deterministic reference."""

    root: Path
    case: dict[str, Any]
    config: development.Config
    provenance: str
    variant: development.Variant | None = None
    replica: int | None = None
    reused: dict[str, Any] | None = None


def run_job(job: Job) -> dict[str, Any]:
    """Execute a private job and normalize exact/trajectory result accounting."""
    target = job.case["target"]
    row: dict[str, Any]
    if job.variant is None:
        path = job.root / "exact" / target["target_id"] / "adam_0p02.json"
        record = cast(
            "dict[str, Any]",
            comparison.optimize(
                path,
                {"circuit": job.case["circuit"], "theta": job.case["initial"]},
                target,
                comparison.Method("exact_adam", 0.02),
                comparison.Config(updates=job.config.updates, switch=job.config.online_updates, backend="statevector"),
                job.provenance,
                0,
            ),
        )
        if (
            abs(record["evaluations"][0]["exact_noisy_fidelity"] - job.case["checkpoints"][0]["exact_noisy_fidelity"])
            > 1e-10
        ):
            msg = "Exact reference did not reproduce the shared start."
            raise ValueError(msg)
        row = {
            "variant": "exact_adam_0p02",
            "replica": None,
            "trajectories": None,
            "path": str(path.relative_to(job.root)),
            "checksum": pilot.digest(record),
            "curve": curve(record, exact=True),
            "training_cpu_seconds": sum(r["cpu_seconds"] for r in record["trace"]),
            "training_wall_seconds": sum(r["wall_seconds"] for r in record["trace"]),
            "evaluation_cpu_seconds": None,
            "evaluation_wall_seconds": sum(e["evaluation_wall_seconds"] for e in record["evaluations"]),
        }
    else:
        assert job.replica is not None
        if job.reused is not None:
            root = Path(job.reused["root"])
            row = dict(job.reused["row"])
        else:
            root = job.root / f"R{job.config.trajectories}"
            row = development.run_job(
                development.Job(root, job.case, job.config, job.provenance, variant=job.variant, replica=job.replica)
            )
        record = pilot.read_record(root / row["path"])
        if row["checksum"] != pilot.digest(record):
            msg = "Refinement result/checkpoint mismatch."
            raise ValueError(msg)
        row.update({
            "path": str((root / row["path"]).relative_to(job.root)) if job.reused is None else row["path"],
            "trajectories": job.config.trajectories,
            "curve": curve(record),
        })
    return {
        **row,
        "target_id": target["target_id"],
        "family": target["family"],
        "reused": job.reused is not None,
        "artifact_root": job.reused["root"] if job.reused is not None else str(job.root.resolve()),
        "initial_fidelity": row["curve"][0]["fidelity"],
        "final_fidelity": row["curve"][-1]["fidelity"],
    }


def report(rows: list[dict[str, Any]]) -> str:
    """Report endpoint and common-update CPU summaries without promoting a winner."""
    lines = [
        "# Optimizer versus trajectory-budget development screen",
        "",
        "Old development targets only. Reused rows retain their original timings.",
        "Exact Adam is a deterministic reference, not a global optimum or a trajectory replica.",
        "",
        "| Method | R | Cases | Mean final fidelity (%) | Mean train CPU (s) | Reused |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    keys = sorted({(row["variant"], row["trajectories"] or 0) for row in rows})
    for name, count in keys:
        subset = [r for r in rows if (r["variant"], r["trajectories"] or 0) == (name, count)]
        lines.append(
            f"| {name} | {count or 'exact'} | {len(subset)} | "
            f"{100 * np.mean([r['final_fidelity'] for r in subset]):.6f} | "
            f"{np.mean([r['training_cpu_seconds'] for r in subset]):.6f} | {sum(r['reused'] for r in subset)} |"
        )
    lines += ["", "Full fidelity-versus-training-CPU curves are retained in result.json; exact scoring is excluded."]
    return "\n".join(lines) + "\n"


def run(
    root: Path,
    source_root: Path,
    *,
    tail_root: Path | None = None,
    config: development.Config | None = None,
    counts: tuple[int, ...] = (32, 128),
    workers: WorkerSetting = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    """Freeze a compact panel; never construct or nominate fresh targets."""
    config = development.Config() if config is None else config
    if not counts or len(set(counts)) != len(counts) or any(type(r) is not int or r < 1 for r in counts):
        msg = "Invalid trajectory counts."
        raise ValueError(msg)
    if config.online_updates >= config.updates:
        msg = "A reference schedule needs a nonempty tail."
        raise ValueError(msg)
    resolve_workers(workers, 1)
    source, cases = development.load_cases(source_root)
    reuse_provenance, reused = (
        (None, {}) if tail_root is None else reuse_tail(tail_root, source_root, source, cases, config)
    )
    manifest = json.loads(
        json.dumps({
            "version": VERSION,
            "config": asdict(config),
            "counts": counts,
            "methods": [asdict(v) for v in methods(config)],
            "matched_rate_control": asdict(development.Variant("hybrid_batch_0p7", config.online_updates, rate=0.7)),
            "control_trajectories": config.trajectories,
            "source_root": str(source_root.resolve()),
            "source": source,
            "cases": cases,
            "reuse": reuse_provenance,
            "runtime": runtime(),
            "driver": pilot.digest(Path(__file__).read_text(encoding="utf-8")),
            "noise": ExactLocalPauliProvider().scientific_dict(),
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
    jobs = []
    for case in cases:
        for count in counts:
            for variant in methods(config):
                for replica in config.replicas:
                    previous = (
                        reused.get((case["target"]["target_id"], variant.name, replica))
                        if count == config.trajectories
                        else None
                    )
                    jobs.append(
                        Job(root, case, replace(config, trajectories=count), identity, variant, replica, previous)
                    )
        jobs.extend(
            Job(
                root,
                case,
                config,
                identity,
                development.Variant("hybrid_batch_0p7", config.online_updates, rate=0.7),
                replica,
            )
            for replica in config.replicas
        )
        jobs.append(Job(root, case, config, identity))
    completed = {}
    for index, row in completed_jobs(run_job, jobs, workers=workers):
        completed[index] = row
        print(
            f"budget {len(completed)}/{len(jobs)}: {row['target_id']} {row['variant']} R={row['trajectories']}",
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
    """Run the fixed scientific screen, optionally reusing compatible old tail work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--tail-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=parse_workers, default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run(args.output_root, args.source_root, tail_root=args.tail_root, workers=args.workers, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
