# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Freeze and run four development cases of the new compiler-before-training composition."""

from __future__ import annotations

import argparse
import csv
import operator
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import compiled_composition as kernel

if TYPE_CHECKING:
    from collections.abc import Sequence

VERSION = "overnight.compiled_conditional_composition_development.v1"
TARGETS = kernel.base.TARGETS


def sources() -> dict[str, str]:
    """Freeze the complete old primitives, compiler and this new composition recipe."""
    result = kernel.base.sources()
    result.update(kernel.common_control_audit.sources())
    for name in (
        "compiled_composition.py",
        "compiled_composition_experiment.py",
        "COMPILED_COMPOSITION_PROTOCOL.md",
        "state_preparation_compilation.py",
        "STATE_PREPARATION_COMPILATION_PROTOCOL.md",
    ):
        path = Path(__file__).with_name(name)
        result[str(path.relative_to(kernel.base.runner.REPO))] = kernel.file_hash(path)
    return result


def protocol_record(initializer_root: Path) -> dict[str, Any]:
    """Authenticate the entire existing initializer panel without starting numerical work."""
    starts, files, original = kernel.base.input_panel(initializer_root)
    return {
        "version": VERSION,
        "phase": "development",
        "targets": list(TARGETS),
        "replicas": [11],
        "methods": list(kernel.METHODS),
        "primary": list(kernel.PRIMARY),
        "baseline": kernel.BASELINE,
        "candidate": kernel.CANDIDATE,
        "starts": starts,
        "input_files": files,
        "initializer_root": str(initializer_root.resolve()),
        "sources": sources(),
        "exact_config": asdict(kernel.base.exact.Config(protected_initial_gates=0)),
        "stochastic_config": asdict(kernel.base.stochastic.Config(protected=0)),
        "environment": original["environment"],
        "benchmark_hashes": kernel.base.runner.custody()["benchmark_hashes"],
        "seed_namespace": "compiled_composition_training",
        "primary_cutoff": "minimum four structural final refinement CPU; score all seven at same cutoff",
        "promotion": {"mean_delta_pp": 0.5, "target_wins": 4, "fixed_and_strongest_controls_required": True},
    }


def prepare(root: Path, initializer_root: Path) -> dict[str, Any]:
    """Persist a read-only freeze; refuse replacement of a different existing protocol."""
    protocol = protocol_record(initializer_root)
    path = root / "PROTOCOL.json"
    if path.exists() and pilot.digest(kernel.read(path)) != pilot.digest(protocol):
        msg = "Compiled-composition protocol changed; use a new output root."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return kernel.read(path)


def verify(protocol: dict[str, Any]) -> None:
    """Reconstruct the full expected input/source freeze, including all source arrays and costs."""
    if pilot.digest(protocol) != pilot.digest(protocol_record(Path(protocol["initializer_root"]))):
        msg = "Compiled-composition protocol or initializer custody changed."
        raise ValueError(msg)


def run_kernel(root: Path, protocol: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Supply only an authenticated development target to the public numerical kernel."""
    offset = TARGETS.index(source["target_id"]) % len(kernel.TRAINED)
    return kernel.execute(
        root,
        source["endpoint"],
        load_target(6, source["target_id"]).state_vector_copy(),
        kernel.base.exact.Config(**protocol["exact_config"]),
        kernel.base.stochastic.Config(**protocol["stochastic_config"]),
        kernel.base.runner.seed_for(protocol["seed_namespace"], source["target_id"], 11),
        order=(*kernel.TRAINED[offset:], *kernel.TRAINED[:offset]),
    )


def run_case(job: dict[str, Any]) -> dict[str, Any]:
    """Execute supplied development inputs, binding compact outer fields to the kernel receipt."""
    protocol, source, root = job["protocol"], job["source"], Path(job["root"])
    verify(protocol)
    if source not in protocol["starts"]:
        msg = "Compiled-composition case is outside the complete development panel."
        raise ValueError(msg)
    path = root / "RESULT.json"
    signature = pilot.digest([protocol, source])
    if path.exists() and kernel.read(path)["status"] == "failed":
        result = kernel.read(path)
        failure = kernel.read(root / "CASE_FAILURE.json")
        expected = {**failure, "case_failure_checksum": pilot.digest(failure), "dependencies": kernel.artifacts(root)}
        if result != expected or result["signature"] != signature:
            msg = "Retained failed case or its durable artifact inventory changed."
            raise ValueError(msg)
        return result
    try:
        execution = run_kernel(root, protocol, source)
    except (ArithmeticError, RuntimeError, MemoryError) as error:
        failure: dict[str, Any] = {
            "version": VERSION,
            "status": "failed",
            "signature": signature,
            "protocol_checksum": pilot.digest(protocol),
            "target_id": source["target_id"],
            "replica": 11,
            "failure": f"{type(error).__name__}: {error}",
            "arms": {},
            "common_primary": {"status": "unavailable_failed_case", "arms": {}},
            "common_all_trained": {"status": "unavailable_failed_case", "arms": {}},
            "initializer_cpu_seconds": source["endpoint"]["initializer_cpu_seconds"],
            "physical_cost_scope": "durable CASE_ATTEMPTS and operations; unavailable partial counts stay unknown",
        }
        pilot.write_record(root / "CASE_FAILURE.json", failure)
        failed: dict[str, Any] = {
            **failure,
            "case_failure_checksum": pilot.digest(failure),
            "dependencies": kernel.artifacts(root),
        }
        pilot.write_record(path, failed)
        return kernel.read(path)
    record: dict[str, Any] = {
        **execution,
        "execution_signature": execution["signature"],
        "execution_checksum": pilot.digest(execution),
        "signature": pilot.digest([protocol, source]),
        "protocol_checksum": pilot.digest(protocol),
        "target_id": source["target_id"],
        "replica": 11,
        "version": VERSION,
    }
    path = root / "RESULT.json"
    if path.exists() and kernel.read(path) != record:
        msg = "Cached outer result differs from its authenticated kernel receipt."
        raise ValueError(msg)
    verify(protocol)
    if not path.exists():
        pilot.write_record(path, record)
    return kernel.read(path)


def analyze(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Apply the fixed full-panel strongest-control rule without pooling failures or missing roles."""
    cases = {record["target_id"]: record for record in records}
    if len(cases) != len(records) or set(cases) != set(TARGETS):
        return {"status": "incomplete", "promoted": False, "missing_targets": sorted(set(TARGETS) - set(cases))}
    if any(case["status"] != "complete" or case["common_primary"]["status"] != "complete" for case in cases.values()):
        return {"status": "failed_case", "promoted": False}
    adapted = [
        {
            "value": {
                "target_id": target,
                "scores": {name: row["scores"] for name, row in case["common_primary"]["arms"].items()},
                "source_unchanged_fidelities": case["common_primary"]["arms"]["unchanged"]["scores"]["fidelities"],
            }
        }
        for target, case in cases.items()
    ]
    comparison = kernel.common_control_audit.summarize(adapted, TARGETS)
    fixed = comparison["paired_comparisons"][kernel.BASELINE]
    return {
        **comparison,
        "status": "complete",
        "baseline": kernel.BASELINE,
        "promoted": comparison["same_rule_passes_against_strongest"]
        and fixed["mean_delta_pp"] >= 0.5
        and fixed["target_wins"] == len(TARGETS),
        "comparison": "all seven new recipe roles at the four-structural common CPU cutoff; development only",
        "retirement": "retain earlier candidate if this new recipe fails; no retrospective replacement",
    }


def summarize(root: Path, protocol: dict[str, Any], records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Write explicit coverage and deterministic all12 rows without copying scientific curves."""
    rows = []
    for record in sorted(records, key=operator.itemgetter("target_id")):
        panels = {"full_cap": record["arms"]}
        panels.update({name: record[name]["arms"] for name in ("common_primary", "common_all_trained")})
        for panel, arms in panels.items():
            for method in kernel.METHODS:
                arm = arms.get(method, {})
                for noise in kernel.base.channels.NOISE_IDS:
                    fidelity = arm.get("scores", {}).get("fidelities", {}).get(noise)
                    rows.append({
                        "panel": panel,
                        "target_id": record["target_id"],
                        "replica": 11,
                        "method": method,
                        "noise_id": noise,
                        "status": arm.get("status", "complete" if fidelity is not None else "missing"),
                        "fidelity": fidelity,
                        "fidelity_percent": None if fidelity is None else 100 * fidelity,
                        "convention": "calibrated_exact",
                        "exactness": "exact_calibrated_channel",
                        **{
                            key: arm.get("scores", {}).get("physical_counts", {}).get(key)
                            for key in ("one_qubit", "two_qubit", "depth")
                        },
                        "parameters": arm.get("scores", {}).get("parameters"),
                        "total_cpu_seconds": arm.get("total_cpu_seconds")
                        if panel == "full_cap"
                        else record["initializer_cpu_seconds"] + arm["point"]["cpu_seconds"]
                        if "point" in arm
                        else None,
                        "cutoff_total_cpu_seconds": None
                        if panel == "full_cap"
                        else record[panel].get("total_cpu_seconds"),
                    })
    analysis = analyze(records)
    result: dict[str, Any] = {
        "version": VERSION,
        "protocol_checksum": pilot.digest(protocol),
        "analysis": analysis,
        "status": "complete"
        if analysis["status"] == "complete" and all(r["status"] == "complete" for r in records)
        else "partial_failure"
        if len(records) == len(TARGETS)
        else "running",
        "case_checksums": {r["target_id"]: pilot.digest(r) for r in records},
    }
    pilot.write_record(root / "SUMMARY.json", result)
    with (root / "SCOREBOARD.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare without science or run only the frozen four-target development panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=kernel.base.runner.OUTPUT / "compiled-composition")
    parser.add_argument("--initializer-root", type=Path, default=kernel.base.runner.OUTPUT / "initializer-policy")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    if args.prepare_only:
        prepare(args.output, args.initializer_root)
        return 0
    if any(
        os.environ.get(name) != "1"
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    ):
        parser.error("Use explicitly single-thread numerical workers.")
    protocol = kernel.read(args.output / "PROTOCOL.json")
    verify(protocol)
    jobs = [
        {"protocol": protocol, "source": source, "root": str(args.output / f"{source['target_id']}__11")}
        for source in protocol["starts"]
    ]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(run_case, jobs):
            records.append(record)
            summarize(args.output, protocol, records)
            print(record["target_id"], record["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
