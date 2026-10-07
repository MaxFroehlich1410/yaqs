# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Prepare and explicitly execute the fixed physical rank-two development panel.

The prepare-only mode freezes sources and authenticated unmatched references;
it never constructs, scores or trains a target. Numerical execution requires
--run. Complete runner checkpoints retain independent Adam/L-BFGS memories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import operator
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import channels, runner, synthesis

if TYPE_CHECKING:
    from collections.abc import Sequence

VERSION = "overnight.rank_two_development_panel.v1"
SPEC_PATH = Path(__file__).with_name("synthesis_wave1.json")
TARGETS = ("gaussian_mu0p5_sigma0p1", "tfim_critical", "haar_random_1", "random_mps_bond2")
REFERENCES = ("exact_adam_600", "lbfgs_600", "weighted_adam_600", "noiseless_adam_600")
METHODS = {
    "rank2_unchanged": {"method": "unchanged", "iterations": 0},
    "rank2_exact_adam_600": {"method": "adam", "iterations": 600, "rate": 0.02, "checkpoint_every": 1},
    "rank2_lbfgs_600": {"method": "lbfgs", "iterations": 599, "evaluations": 600, "history": 10},
}


def numerical_sources() -> dict[str, str]:
    """Extend the runner fingerprint without modifying its frozen dependency set."""
    result = runner.source_hashes()
    for name in ("synthesis", "synthesis_experiment"):
        path = Path(__file__).with_name(f"{name}.py")
        result[str(path.relative_to(runner.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def validate_spec(spec: dict[str, Any]) -> None:
    """Reject changes to the preregistered scientific panel or numerical policy."""
    if (
        spec.get("name") != "synthesis-wave1"
        or spec.get("phase") != "development"
        or spec.get("targets") != list(TARGETS)
        or spec.get("replicas") != [11]
        or spec.get("methods") != METHODS
        or spec.get("reference_panel") != "focused_curvature"
        or spec.get("reference_methods") != list(REFERENCES)
    ):
        msg = "Synthesis wave one requires its fixed development panel and budgets."
        raise ValueError(msg)


def _reference(path: Path, target_id: str, method: str) -> dict[str, Any]:
    """Authenticate a complete 108-gate endpoint and all twelve saved scores."""
    record = cast("dict[str, Any]", pilot.read_record(path))
    job = record["job"]
    circuit = pilot.restore_circuit(job["circuit"])
    if (
        record.get("status") != "complete"
        or job["target_id"] != target_id
        or job["replica"] != 11
        or job["label"] != method
        or job["phase"] != "development"
        or circuit.num_qubits != 6
        or len(circuit.gates) != 108
        or record["signature"] != pilot.digest({k: v for k, v in job.items() if k != "path"})
        or set(record["fidelities"]) != set(channels.NOISE_IDS)
        or any(not np.isfinite(v) or not -1e-10 <= v <= 1 + 1e-10 for v in record["fidelities"].values())
    ):
        msg = "Unmatched focused-curvature reference is incomplete or has the wrong identity."
        raise ValueError(msg)
    return record


def freeze_protocol(output: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Freeze numerical sources and reference custody without scientific work."""
    validate_spec(spec)
    manifest = runner.custody()
    references = []
    for target_id in TARGETS:
        for method in REFERENCES:
            path = runner.OUTPUT / "focused_curvature" / f"{target_id}__11__{method}.json"
            record = _reference(path, target_id, method)
            references.append({
                "target_id": target_id,
                "method": method,
                "path": str(path),
                "checksum": pilot.digest(record),
            })
    protocol = {
        "version": VERSION,
        "spec": spec,
        "benchmark_hashes": manifest["benchmark_hashes"],
        "source_hashes": numerical_sources(),
        "references": references,
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "PROTOCOL.json"
    if path.exists() and pilot.digest(pilot.read_record(path)) != pilot.digest(protocol):
        msg = "Frozen synthesis protocol differs; use a new output directory."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return protocol


def verify_protocol(protocol: dict[str, Any]) -> None:
    """Check scientific identity before any cached read or new numerical work."""
    validate_spec(protocol["spec"])
    manifest = runner.custody()
    if (
        protocol["version"] != VERSION
        or protocol["benchmark_hashes"] != manifest["benchmark_hashes"]
        or protocol["source_hashes"] != numerical_sources()
        or len(protocol["references"]) != len(TARGETS) * len(REFERENCES)
        or {(r["target_id"], r["method"]) for r in protocol["references"]}
        != {(target, method) for target in TARGETS for method in REFERENCES}
    ):
        msg = "Synthesis protocol custody or numerical source identity changed."
        raise ValueError(msg)
    for ref in protocol["references"]:
        if pilot.digest(_reference(Path(ref["path"]), ref["target_id"], ref["method"])) != ref["checksum"]:
            msg = "Unmatched focused-curvature reference checksum changed."
            raise ValueError(msg)


def run_arm(job: dict[str, Any]) -> dict[str, Any]:
    """Use runner.train through its authenticated runner, retaining failure rows.

    Adam checkpoints every update; L-BFGS checkpoints each accepted boundary.
    Interrupted attempts are measured even when a partial line search must be
    replayed. Numerical endpoint resume is exact; physical attempt CPU includes
    all replay and I/O, separately from checkpoint-accounted refinement CPU.
    """
    runner.custody()
    if job.get("source_hash") is not None and job["source_hash"] != pilot.digest(runner.source_hashes()):
        msg = "Numerical source fingerprint changed after protocol freeze."
        raise ValueError(msg)
    path = Path(job["path"])
    signature = pilot.digest({k: v for k, v in job.items() if k != "path"})
    failure_path, attempts_path = path.with_suffix(".failure.json"), path.with_suffix(".attempts.json")
    if failure_path.exists():
        failed = cast("dict[str, Any]", pilot.read_record(failure_path))
        if failed["signature"] != signature or failed.get("status") != "failed":
            msg = "Retained synthesis failure configuration changed."
            raise ValueError(msg)
        return failed
    if path.exists():
        return runner.run_job(job)
    attempts: dict[str, Any] = (
        cast("dict[str, Any]", pilot.read_record(attempts_path))
        if attempts_path.exists()
        else {"signature": signature, "attempts": []}
    )
    if attempts["signature"] != signature:
        msg = "Synthesis attempt configuration changed."
        raise ValueError(msg)
    cpu, wall = time.process_time(), time.perf_counter()
    attempt: dict[str, Any] = {"status": "running", "cpu_seconds": None, "wall_seconds": None}
    attempts["attempts"].append(attempt)
    pilot.write_record(attempts_path, attempts)
    try:
        return runner.run_job(job)
    except (ArithmeticError, RuntimeError) as error:
        runner.custody()
        checkpoint = path.with_suffix(".checkpoint.json")
        saved = cast("dict[str, Any]", pilot.read_record(checkpoint)) if checkpoint.exists() else None
        if saved is not None and saved["signature"] != signature:
            msg = "Failed synthesis checkpoint configuration changed."
            raise ValueError(msg) from error
        failed = {
            "signature": signature,
            "status": "failed",
            "job": job,
            "error_type": type(error).__name__,
            "error": str(error),
            "state": None if saved is None else saved["state"],
        }
        pilot.write_record(failure_path, failed)
        return failed
    finally:
        attempt.update({
            "status": "timed",
            "cpu_seconds": time.process_time() - cpu,
            "wall_seconds": time.perf_counter() - wall,
        })
        pilot.write_record(attempts_path, attempts)


def arm_summary(
    record: dict[str, Any], construction: dict[str, Any] | None, attempt_cpu: float | None
) -> dict[str, Any]:
    """Expose complete states and distinguish charged from physically shared work."""
    state = record.get("state")
    construction_cpu = None if construction is None else construction["construction_cpu_seconds"]
    return {
        "status": record["status"],
        "record_checksum": pilot.digest(record),
        "circuit": record["job"]["circuit"],
        "theta": None if state is None else state["theta"],
        "state": state,
        "fidelities": record.get("fidelities", {}),
        "physical_counts": runner.counts(pilot.restore_circuit(record["job"]["circuit"])),
        "ballarin_counts": record.get("ballarin_counts"),
        "construction_cpu_seconds": construction_cpu,
        "construction_and_refinement_cpu_seconds": None
        if construction_cpu is None or state is None
        else construction_cpu + state["training_cpu_seconds"],
        "physical_invocation_cpu_seconds": attempt_cpu,
        "evaluation_cpu_seconds": record.get("evaluation_cpu_seconds"),
        "comparison": "independent_rank_two_arm"
        if construction is not None
        else "UNMATCHED_different_construction_and_cost",
        "error_type": record.get("error_type"),
        "error": record.get("error"),
    }


def execute_case(job: dict[str, Any]) -> dict[str, Any]:
    """Construct once, independently refine three arms and preserve every case."""
    protocol, target_id, replica = job["protocol"], job["target_id"], job["replica"]
    verify_protocol(protocol)
    if target_id not in TARGETS or replica != 11:
        msg = "Only the frozen development cases may execute."
        raise ValueError(msg)
    root = Path(job["root"])
    signature = pilot.digest({"protocol": protocol, "target_id": target_id, "replica": replica})
    path = root / "RESULT.json"
    if path.exists():
        result = cast("dict[str, Any]", pilot.read_record(path))
        if result["signature"] != signature or result["status"] not in {"complete", "partial_failure"}:
            msg = "Completed synthesis case configuration changed."
            raise ValueError(msg)
        return result
    root.mkdir(parents=True, exist_ok=True)
    construction_path = root / "CONSTRUCTION.json"
    built: dict[str, Any]
    if construction_path.exists():
        built = cast("dict[str, Any]", pilot.read_record(construction_path))
        if built["signature"] != signature or built.get("status") not in {"complete", "failed"}:
            msg = "Retained synthesis construction configuration changed."
            raise ValueError(msg)
    else:
        target = load_target(6, target_id).state_vector_copy()
        cpu, wall = time.process_time(), time.perf_counter()
        try:
            built = {"signature": signature, "status": "complete", "candidate": synthesis.synthesize(target)}
        except (ArithmeticError, RuntimeError, ValueError) as error:
            built = {
                "signature": signature,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        built["physical_cpu_seconds"] = time.process_time() - cpu
        built["physical_wall_seconds"] = time.perf_counter() - wall
        verify_protocol(protocol)
        pilot.write_record(construction_path, built)
    arms = {}
    for label, config in protocol["spec"]["methods"].items():
        verify_protocol(protocol)
        if built["status"] != "complete":
            arms[label] = {
                "status": "failed",
                "error_type": built["error_type"],
                "error": built["error"],
                "fidelities": {},
                "comparison": "independent_rank_two_arm",
                "construction_cpu_seconds": built["physical_cpu_seconds"],
            }
            continue
        candidate = built["candidate"]
        arm_job = {
            "phase": "development",
            "label": label,
            "config": config,
            "target_id": target_id,
            "replica": replica,
            "circuit": candidate["circuit"],
            "initial": candidate["theta"],
            "construction_checksum": pilot.digest(built),
            "protocol_checksum": pilot.digest(protocol),
            "source_hash": pilot.digest(runner.source_hashes()),
            "seed": runner.seed_for("development_training", target_id, replica),
            "path": str(root / f"{label}.json"),
        }
        record = run_arm(arm_job)
        attempts_path = Path(arm_job["path"]).with_suffix(".attempts.json")
        attempts = cast("dict[str, Any]", pilot.read_record(attempts_path)) if attempts_path.exists() else None
        timing_complete = attempts is not None and all(a["cpu_seconds"] is not None for a in attempts["attempts"])
        attempt_cpu = sum(a["cpu_seconds"] for a in attempts["attempts"]) if timing_complete else None
        arms[label] = arm_summary(record, candidate, attempt_cpu)
        arms[label]["physical_invocation_timing_complete"] = timing_complete
    for ref in protocol["references"]:
        if ref["target_id"] == target_id:
            record = _reference(Path(ref["path"]), target_id, ref["method"])
            if pilot.digest(record) != ref["checksum"]:
                msg = "Unmatched reference changed during synthesis execution."
                raise ValueError(msg)
            arms[f"reference_{ref['method']}"] = arm_summary(record, None, None)
    result = {
        "signature": signature,
        "status": "complete" if all(a["status"] == "complete" for a in arms.values()) else "partial_failure",
        "target_id": target_id,
        "replica": replica,
        "construction": built,
        "arms": arms,
        "physical_construction_cpu_seconds": built["physical_cpu_seconds"],
        "shared_work_rule": "construction executed once and charged fully to every rank-two arm",
    }
    verify_protocol(protocol)
    pilot.write_record(path, result)
    return result


def write_scoreboard(root: Path, records: list[dict[str, Any]]) -> None:
    """Include every target, arm and profile, including explicit failed rows."""
    rows = []
    for result in sorted(records, key=operator.itemgetter("target_id", "replica")):
        for label, arm in sorted(result["arms"].items()):
            for noise in channels.NOISE_IDS:
                physical = arm.get("physical_counts") or {}
                lowered = (arm.get("ballarin_counts") or {}) if noise == "ballarin_coupled" else physical
                state = arm.get("state") or {}
                rows.append({
                    "target_id": result["target_id"],
                    "replica": result["replica"],
                    "method": label,
                    "noise_id": noise,
                    "status": arm["status"],
                    "fidelity": arm["fidelities"].get(noise),
                    "comparison": arm["comparison"],
                    "physical_one_qubit": physical.get("one_qubit"),
                    "physical_two_qubit": physical.get("two_qubit"),
                    "lowered_one_qubit": lowered.get("one_qubit"),
                    "lowered_two_qubit": lowered.get("two_qubit"),
                    "physical_depth": physical.get("depth"),
                    "lowered_depth": lowered.get("depth"),
                    "parameters": len(arm["theta"]) if arm.get("theta") is not None else None,
                    "construction_cpu_seconds": arm.get("construction_cpu_seconds"),
                    "construction_and_refinement_cpu_seconds": arm.get("construction_and_refinement_cpu_seconds"),
                    "physical_invocation_cpu_seconds": arm.get("physical_invocation_cpu_seconds"),
                    "physical_invocation_timing_complete": arm.get("physical_invocation_timing_complete"),
                    "training_cpu_seconds": state.get("training_cpu_seconds"),
                    "diagnostic_cpu_seconds": state.get("diagnostic_cpu_seconds"),
                    "exact_training_evaluations": state.get("exact_training_evaluations"),
                    "direction_passes": state.get("direction_passes"),
                    "iterations": state.get("iterations"),
                    "evaluation_cpu_seconds": arm.get("evaluation_cpu_seconds"),
                    "error_type": arm.get("error_type"),
                    "error": arm.get("error"),
                    "record_checksum": arm.get("record_checksum"),
                })
    if rows:
        with (root / "SCOREBOARD.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare only by default; scientific execution requires an explicit mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=runner.OUTPUT / "synthesis-wave1")
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four workers.")
    if args.prepare_only:
        freeze_protocol(args.output, json.loads(args.spec.read_text()))
        print(f"Prepared only: {args.output / 'PROTOCOL.json'}", flush=True)
        return 0
    protocol = cast("dict[str, Any]", pilot.read_record(args.output / "PROTOCOL.json"))
    verify_protocol(protocol)
    records = []
    jobs = [
        {"protocol": protocol, "target_id": target, "replica": 11, "root": str(args.output / f"{target}__11")}
        for target in TARGETS
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(execute_case, jobs):
            records.append(record)
            write_scoreboard(args.output, records)
            print(record["target_id"], record["status"], flush=True)
    complete = all(record["status"] == "complete" for record in records)
    pilot.write_record(
        args.output / "RESULT.json",
        {
            "status": "complete" if complete else "partial_failure",
            "protocol_checksum": pilot.digest(protocol),
            "results": [pilot.digest(r) for r in records],
        },
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
