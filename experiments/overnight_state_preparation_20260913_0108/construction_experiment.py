# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Freeze and reproduce the four archived development constructions, replica 11."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import construction, runner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

VERSION = "overnight.legacy_construction_development_batch.v1"
TARGETS = tuple(construction.TARGETS)
ARCHIVE_ROOT = Path(
    "/Users/maximilianfrohlich/Desktop/yaqs-experiment-results/q6-hybrid-construction-extension-20260906"
)


def sources() -> dict[str, str]:
    """Return adapter numerical custody plus the complete batch-wrapper source."""
    hashes = construction.sources()
    hashes[str(Path(__file__).relative_to(runner.REPO))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashes


def prepare_batch(root: Path) -> dict[str, Any]:
    """Return a complete four-case frozen batch without constructing or scoring states."""
    cases = []
    for target_id in TARGETS:
        archive = construction.archive_reference(ARCHIVE_ROOT, target_id, 11)
        vector = load_target(6, target_id).state_vector_copy()
        case_root = root / f"{target_id}__11"
        manifest = construction.prepare(case_root, vector, target_id, 11, archive=archive)
        cases.append({
            "target_id": target_id,
            "replica": 11,
            "root": str(case_root),
            "manifest_checksum": pilot.digest(manifest),
        })
    protocol: dict[str, Any] = {
        "version": VERSION,
        "phase": "development",
        "targets": list(TARGETS),
        "replicas": [11],
        "archive_root": str(ARCHIVE_ROOT),
        "sources": sources(),
        "environment": construction.environment(),
        "benchmark_hashes": runner.custody()["benchmark_hashes"],
        "cases": cases,
        "rule": "All four independently reconstructed 570-update final endpoints; no checkpoint selection or retuning.",
        "reproduction_tolerance": {"parameter_max_abs": 1e-11, "ideal_state_infidelity": 1e-10, "state_max_abs": 1e-10},
    }
    verify_batch(protocol)
    path = root / "BATCH_PROTOCOL.json"
    if path.exists() and pilot.digest(pilot.read_record(path)) != pilot.digest(protocol):
        msg = "Frozen construction batch differs; use a new output root."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return protocol


def verify_batch(protocol: dict[str, Any]) -> None:
    """Authenticate the complete development panel and every prepared case before work."""
    if (
        protocol["version"] != VERSION
        or protocol["phase"] != "development"
        or protocol["targets"] != list(TARGETS)
        or protocol["replicas"] != [11]
        or protocol["archive_root"] != str(ARCHIVE_ROOT)
        or protocol["sources"] != sources()
        or protocol["environment"] != construction.environment()
        or protocol["benchmark_hashes"] != runner.custody()["benchmark_hashes"]
        or len(protocol["cases"]) != len(TARGETS)
        or {(c["target_id"], c["replica"]) for c in protocol["cases"]} != {(t, 11) for t in TARGETS}
        or protocol["reproduction_tolerance"]
        != {"parameter_max_abs": 1e-11, "ideal_state_infidelity": 1e-10, "state_max_abs": 1e-10}
    ):
        msg = "Construction batch panel, sources or benchmark custody changed."
        raise ValueError(msg)
    for case in protocol["cases"]:
        manifest = cast("dict[str, Any]", pilot.read_record(Path(case["root"]) / "PROTOCOL.json"))
        if (
            pilot.digest(manifest) != case["manifest_checksum"]
            or manifest["target_id"] != case["target_id"]
            or manifest["replica"] != 11
            or manifest["config"] != asdict(construction.Config())
            or manifest["archive_reference"]["path"]
            != str(ARCHIVE_ROOT / "H25_lr0p7" / case["target_id"] / "seed_11/result.json")
        ):
            msg = "Prepared construction case or archive identity changed."
            raise ValueError(msg)
        construction.verify_manifest(manifest)


def prepared_state(record: dict[str, Any], theta: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Return an independent Qiskit state after the entire physical circuit from zero."""
    circuit = pilot.restore_circuit(record)
    reference = QuantumCircuit(circuit.num_qubits)
    for gate in circuit.gates:
        getattr(reference, gate.name)(circuit.angle(gate, theta, None), *gate.sites)
    return Statevector.from_instruction(reference).data


def audit_state(result: dict[str, Any], archive: dict[str, Any]) -> dict[str, Any]:
    """Return full circuit, parameter and phase-insensitive prepared-state equivalence."""
    actual = np.asarray(result["theta"], dtype=np.float64)
    expected = np.asarray(archive["theta"], dtype=np.float64)
    parameter_error = float(np.max(abs(actual - expected))) if actual.shape == expected.shape else None
    left, right = prepared_state(result["circuit"], actual), prepared_state(archive["circuit"], expected)
    inner = np.vdot(right, left)
    phase = inner / abs(inner) if abs(inner) else 1.0
    overlap = float(abs(inner) ** 2)
    error = float(np.max(abs(left - phase * right)))
    if not np.isfinite(overlap) or not np.isfinite(error):
        msg = "Independent construction state audit is nonfinite."
        raise FloatingPointError(msg)
    circuit_equal = pilot.digest(result["circuit"]) == pilot.digest(archive["circuit"])
    return {
        "circuit_equal": circuit_equal,
        "parameter_max_abs_error": parameter_error,
        "ideal_state_overlap": overlap,
        "phase_aligned_state_max_abs_error": error,
        "matches": circuit_equal
        and parameter_error is not None
        and parameter_error <= 1e-11
        and abs(overlap - 1) <= 1e-10
        and error <= 1e-10,
    }


def failed_case(root: Path, signature: str, target_id: str, error: Exception) -> dict[str, Any]:
    """Retain retryable numerical failures and all measured construction attempts."""
    path = root / "FAILURES.json"
    history: dict[str, Any] = (
        cast("dict[str, Any]", pilot.read_record(path)) if path.exists() else {"signature": signature, "failures": []}
    )
    if history["signature"] != signature:
        msg = "Construction batch failure identity changed."
        raise ValueError(msg)
    attempts_path = root / "ATTEMPTS.json"
    attempts = cast("dict[str, Any]", pilot.read_record(attempts_path))["attempts"] if attempts_path.exists() else []
    timed = bool(attempts) and all(a["cpu_seconds"] is not None and a["wall_seconds"] is not None for a in attempts)
    failure = {
        "status": "failed",
        "signature": signature,
        "target_id": target_id,
        "replica": 11,
        "failure": f"{type(error).__name__}: {error}",
        "matches": False,
        "construction_cpu_seconds": sum(a["cpu_seconds"] for a in attempts) if timed else None,
        "construction_wall_seconds": sum(a["wall_seconds"] for a in attempts) if timed else None,
        "historical_cpu_seconds": None,
    }
    history["failures"].append(failure)
    pilot.write_record(path, history)
    return failure


def run_case(job: dict[str, Any]) -> dict[str, Any]:
    """Resume one adapter job and retain its independent final-state reproduction audit."""
    protocol, case = job["protocol"], job["case"]
    verify_batch(protocol)
    if pilot.digest(case) not in {pilot.digest(c) for c in protocol["cases"]}:
        msg = "Construction job is outside its frozen batch."
        raise ValueError(msg)
    root = Path(case["root"])
    manifest = cast("dict[str, Any]", pilot.read_record(root / "PROTOCOL.json"))
    signature = pilot.digest({"protocol": protocol, "case": case})
    cpu, wall = time.process_time(), time.perf_counter()
    try:
        result = construction.reproduce(root, manifest)
    except (ArithmeticError, RuntimeError) as error:
        verify_batch(protocol)
        return failed_case(root, signature, case["target_id"], error)
    path = root / "AUDIT.json"
    if path.exists():
        record = cast("dict[str, Any]", pilot.read_record(path))
        if record["signature"] != signature or record["construction_checksum"] != pilot.digest(result):
            msg = "Completed construction audit identity changed."
            raise ValueError(msg)
        return record
    started = time.process_time()
    try:
        audit = audit_state(result, manifest["archive_reference"])
    except (ArithmeticError, RuntimeError) as error:
        verify_batch(protocol)
        return failed_case(root, signature, case["target_id"], error)
    record = {
        **audit,
        "status": "complete" if audit["matches"] else "mismatch",
        "signature": signature,
        "target_id": case["target_id"],
        "replica": 11,
        "construction_checksum": pilot.digest(result),
        "construction_path": str(root / "RESULT.json"),
        "archive_checksum": manifest["archive_reference"]["checksum"],
        "physical_counts": result["physical_counts"],
        "parameters": result["parameters"],
        "updates": result["updates"],
        "construction_cpu_seconds": result["construction_cpu_seconds"],
        "construction_wall_seconds": result["construction_wall_seconds"],
        "construction_timing_complete": result["timing_complete"],
        "historical_cpu_seconds": None,
        "state_audit_cpu_seconds": time.process_time() - started,
        "wrapper_cpu_seconds": time.process_time() - cpu,
        "wrapper_wall_seconds": time.perf_counter() - wall,
        "final_forward_peak_bond": result["diagnostics"]["observed_final_forward_peak_bond"],
    }
    verify_batch(protocol)
    pilot.write_record(path, record)
    return record


def write_summary(root: Path, protocol: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain every available case, explicit pending cases, mismatches and failures."""
    by_target = {record["target_id"]: record for record in records}
    if len(by_target) != len(records) or set(by_target) - set(TARGETS):
        msg = "Construction summary requires unique members of the fixed development panel."
        raise ValueError(msg)
    rows = []
    for target_id in TARGETS:
        record = by_target.get(target_id, {"status": "pending"})
        rows.append({
            "target_id": target_id,
            "replica": 11,
            **{
                key: record.get(key)
                for key in (
                    "status",
                    "matches",
                    "circuit_equal",
                    "parameter_max_abs_error",
                    "ideal_state_overlap",
                    "phase_aligned_state_max_abs_error",
                    "construction_cpu_seconds",
                    "construction_wall_seconds",
                    "historical_cpu_seconds",
                    "state_audit_cpu_seconds",
                    "failure",
                )
            },
            "physical_one_qubit": record.get("physical_counts", {}).get("one_qubit"),
            "physical_two_qubit": record.get("physical_counts", {}).get("two_qubit"),
            "parameters": record.get("parameters"),
            "updates": record.get("updates"),
        })
    complete = len(records) == len(TARGETS)
    timed = complete and all(r.get("construction_cpu_seconds") is not None for r in records)
    result = {
        "status": "complete"
        if complete and all(r["status"] == "complete" for r in records)
        else "partial_failure"
        if complete
        else "running",
        "protocol_checksum": pilot.digest(protocol),
        "rows": rows,
        "total_construction_cpu_seconds": sum(r["construction_cpu_seconds"] for r in records) if timed else None,
        "all_archive_reproductions_match": complete and all(r.get("matches", False) for r in records),
        "case_checksums": {r["target_id"]: pilot.digest(r) for r in records},
    }
    pilot.write_record(root / "SUMMARY.json", result)
    with (root / "SCOREBOARD.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare all four cases or execute them using the caller's explicit worker cap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=runner.OUTPUT / "construction-reproduction")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four workers; no automatic worker expansion.")
    if args.prepare_only:
        prepare_batch(args.output)
        print(f"Prepared all four development cases: {args.output / 'BATCH_PROTOCOL.json'}", flush=True)
        return 0
    protocol = cast("dict[str, Any]", pilot.read_record(args.output / "BATCH_PROTOCOL.json"))
    verify_batch(protocol)
    records = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, {"protocol": protocol, "case": case}) for case in protocol["cases"]]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            summary = write_summary(args.output, protocol, records)
            print(
                record["target_id"],
                record["status"],
                "construction_cpu",
                record["construction_cpu_seconds"],
                flush=True,
            )
            if len(records) == 1:
                elapsed = time.perf_counter() - started
                estimate = elapsed * math.ceil((len(TARGETS) - 1) / args.workers)
                print(
                    f"First case completed after {elapsed:.1f}s; rough remaining {estimate:.1f}s "
                    "assuming equal case time and ignoring in-flight progress.",
                    flush=True,
                )
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
