# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Four-case development reproduction of the lazy conditional initializer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import (
    composed_experiment,
    construction,
    initializer_policy,
    lazy_initializer,
    runner,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

VERSION = "overnight.lazy_initializer_development_reproduction.v1"
TARGETS = tuple(construction.TARGETS)


def sources() -> dict[str, str]:
    """Freeze all lazy numerics, source authentication code and this development wrapper."""
    result = composed_experiment.sources()
    result.update(lazy_initializer.sources())
    path = Path(__file__)
    result[str(path.relative_to(runner.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def references(root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Authenticate every original conditional initializer without rebuilding any source."""
    starts, files, _ = composed_experiment.input_panel(root)
    result = []
    for start in starts:
        built = lazy_initializer.read(root / f"{start['target_id']}__11" / "INITIALIZERS.json")
        result.append({
            "target_id": start["target_id"],
            "replica": 11,
            "endpoint": start["endpoint"],
            "ideal_tt_overlap": built["ideal_tt_overlap"],
            "rank_two_physical_overlap": built["rank_two_physical_overlap"],
            "rank_two_selected": built["rank_two_selected"],
            "initializer_checksum": pilot.digest(built),
        })
    return result, files


def prepare(root: Path, initializer_root: Path) -> dict[str, Any]:
    """Prepare all four normalized development inputs before any numerical construction."""
    expected, files = references(initializer_root)
    constructor = lazy_initializer.constructor_identity("development")
    assert constructor is not None
    cases = []
    for source in expected:
        target = load_target(6, source["target_id"]).state_vector_copy()
        path = root / f"{source['target_id']}__11" / "initializer"
        manifest = lazy_initializer.prepare(
            path,
            target,
            source["target_id"],
            11,
            phase="development",
            constructor=constructor,
        )
        cases.append({
            "target_id": source["target_id"],
            "replica": 11,
            "reference": source,
            "manifest_checksum": pilot.digest(manifest),
        })
    protocol = {
        "version": VERSION,
        "phase": "development",
        "targets": list(TARGETS),
        "replicas": [11],
        "cases": cases,
        "sources": sources(),
        "input_files": files,
        "initializer_root": str(initializer_root.resolve()),
        "environment": construction.environment(),
        "benchmark_hashes": runner.custody()["benchmark_hashes"],
        "acceptance": (
            "all four complete; exact circuit and theta equality; overlap errors <=1e-10; "
            "physical state error <=1e-10; constructor only below threshold"
        ),
    }
    path = root / "PROTOCOL.json"
    if path.exists() and pilot.digest(lazy_initializer.read(path)) != pilot.digest(protocol):
        msg = "Lazy development reproduction changed; use a new root."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return lazy_initializer.read(path)


def verify(root: Path, protocol: dict[str, Any]) -> None:
    """Reject changed complete source panels, code, environment or child manifests."""
    expected, files = references(Path(protocol["initializer_root"]))
    if (
        protocol["version"] != VERSION
        or protocol["phase"] != "development"
        or protocol["targets"] != list(TARGETS)
        or protocol["replicas"] != [11]
        or protocol["sources"] != sources()
        or protocol["input_files"] != files
        or protocol["environment"] != construction.environment()
        or protocol["benchmark_hashes"] != runner.custody()["benchmark_hashes"]
        or len(protocol["cases"]) != 4
        or pilot.digest([case["reference"] for case in protocol["cases"]]) != pilot.digest(expected)
        or {(c["target_id"], c["replica"]) for c in protocol["cases"]} != {(t, 11) for t in TARGETS}
        or any(
            pilot.digest(lazy_initializer.read(root / f"{c['target_id']}__11" / "initializer/PROTOCOL.json"))
            != c["manifest_checksum"]
            for c in protocol["cases"]
        )
        or pilot.digest(lazy_initializer.read(root / "PROTOCOL.json")) != pilot.digest(protocol)
    ):
        msg = "Lazy development reproduction source, case or numerical custody changed."
        raise ValueError(msg)


def audit(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Compare exact scalar endpoints and independent full physical preparation states."""
    cpu, wall = time.process_time(), time.perf_counter()
    new, old = actual["endpoint"], expected["endpoint"]
    circuit_equal = pilot.digest(new["circuit"]) == pilot.digest(old["circuit"])
    theta_equal = np.array_equal(np.asarray(new["theta"]), np.asarray(old["theta"]))
    before = initializer_policy.state_vector(old["circuit"], np.asarray(old["theta"]))
    after = initializer_policy.state_vector(new["circuit"], np.asarray(new["theta"]))
    overlap = complex(np.vdot(before, after))
    phase = overlap / abs(overlap) if overlap else 1.0 + 0j
    error = float(np.max(np.abs(after - phase * before)))
    overlap_error = abs(actual["ideal_tt_overlap"] - expected["ideal_tt_overlap"])
    lowering_error = abs(actual["rank_two_physical_overlap"] - expected["rank_two_physical_overlap"])
    return {
        "matches": circuit_equal
        and theta_equal
        and error <= 1e-10
        and overlap_error <= 1e-10
        and lowering_error <= 1e-10
        and actual["rank_two_selected"] == expected["rank_two_selected"],
        "circuit_equal": circuit_equal,
        "theta_equal": theta_equal,
        "phase_aligned_max_state_error": error,
        "tt_overlap_error": overlap_error,
        "lowering_overlap_error": lowering_error,
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall,
    }


def construct(root: Path, vector: NDArray[np.complex128], target_id: str) -> dict[str, Any]:
    """Run the exact already-reproduced development constructor only when requested."""
    manifest = construction.prepare(root, vector, target_id, 11)
    return construction.reproduce(root, manifest)


def dependencies(root: Path) -> dict[str, str]:
    """Require the complete current child registry, including unexpected file additions."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.json"))
        if p not in {root / "RESULT.json", root / "COMPLETED.json"}
    }


def cached_result(root: Path, signature: str) -> dict[str, Any]:
    """Bind every final audit/cost field to its completion receipt and exact child registry."""
    result, sealed = lazy_initializer.read(root / "RESULT.json"), lazy_initializer.read(root / "COMPLETED.json")
    child = lazy_initializer.read(root / "initializer/RESULT.json")
    ledger = lazy_initializer.read(root / "CASE_ATTEMPTS.json")
    if (
        result["signature"] != signature
        or sealed != {"signature": signature, "result_checksum": pilot.digest(result)}
        or result["dependencies"] != dependencies(root)
        or result["initializer_checksum"] != pilot.digest(child)
        or result["physical_invocation"]["attempts_checksum"] != pilot.digest(ledger)
        or any(
            result["physical_invocation"][f"{kind}_seconds"] != sum(a[f"{kind}_seconds"] for a in ledger["attempts"])
            for kind in ("cpu", "wall")
        )
    ):
        msg = "Lazy reproduction endpoint, audit, cost or complete dependency registry changed."
        raise ValueError(msg)
    return result


def run_case(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one frozen development equivalence case and retain explicit failures."""
    batch_root, protocol, case = Path(job["root"]), job["protocol"], job["case"]
    verify(batch_root, protocol)
    if case not in protocol["cases"]:
        msg = "Lazy initializer case is outside the complete development panel."
        raise ValueError(msg)
    root = batch_root / f"{case['target_id']}__11"
    path = root / "RESULT.json"
    signature = pilot.digest([protocol, case])
    ledger_path = root / "CASE_ATTEMPTS.json"
    if ledger_path.exists():
        ledger = lazy_initializer.read(ledger_path)
        if ledger["signature"] != signature or any(
            a["status"] not in {"complete", "failed", "interrupted"}
            or any(a.get(k) is None or not np.isfinite(a[k]) or a[k] < 0 for k in ("cpu_seconds", "wall_seconds"))
            for a in ledger["attempts"]
        ):
            msg = "Unknown lazy reproduction work or changed case blocks resume."
            raise ValueError(msg)
    if path.exists():
        return cached_result(root, signature)
    input_cpu = time.process_time()
    target = load_target(6, case["target_id"]).state_vector_copy()
    input_cpu = time.process_time() - input_cpu
    manifest = lazy_initializer.read(root / "initializer/PROTOCOL.json")

    def work() -> dict[str, Any]:
        actual = lazy_initializer.run(
            root / "initializer", manifest, target, lambda child: construct(child, target, case["target_id"])
        )
        checked = audit(actual, case["reference"])
        has_legacy = (root / "initializer/legacy").exists()
        checked["lazy_branch_verified"] = has_legacy is not actual["rank_two_selected"]
        return {
            "status": "complete",
            "accepted": checked["matches"] and checked["lazy_branch_verified"],
            "initializer_checksum": pilot.digest(actual),
            "audit": checked,
            "rank_two_selected": actual["rank_two_selected"],
            "ideal_tt_overlap": actual["ideal_tt_overlap"],
            "initializer_cpu_seconds": actual["initializer_cpu_seconds"],
            "initializer_wall_seconds": actual["initializer_wall_seconds"],
            "nested_legacy_cpu_seconds": actual["nested_legacy_cpu_seconds"],
            "physical_counts": actual["endpoint"]["physical_counts"],
        }

    result = composed_experiment.measured_execute(root, signature, work)
    verify(batch_root, protocol)
    result.update(signature=signature, target_id=case["target_id"], replica=11, target_input_cpu_seconds=input_cpu)
    result["dependencies"] = dependencies(root)
    pilot.write_record(root / "COMPLETED.json", {"signature": signature, "result_checksum": pilot.digest(result)})
    pilot.write_record(path, result)
    return result


def attempt_case(job: dict[str, Any]) -> dict[str, Any]:
    """Return failed numerical cases while custody violations still abort immediately."""
    try:
        return run_case(job)
    except (ArithmeticError, RuntimeError) as error:
        root = Path(job["root"]) / f"{job['case']['target_id']}__11"
        ledger_path = root / "CASE_ATTEMPTS.json"
        attempts = lazy_initializer.read(ledger_path)["attempts"] if ledger_path.exists() else []
        complete = bool(attempts) and all(
            a["status"] in {"complete", "failed", "interrupted"}
            and all(a.get(k) is not None and np.isfinite(a[k]) and a[k] >= 0 for k in ("cpu_seconds", "wall_seconds"))
            for a in attempts
        )
        result: dict[str, Any] = {
            "status": "failed",
            "accepted": False,
            "target_id": job["case"]["target_id"],
            "replica": 11,
            "failure": f"{type(error).__name__}: {error}",
            "physical_invocation": {
                "cpu_seconds": sum(a["cpu_seconds"] for a in attempts) if complete else None,
                "wall_seconds": sum(a["wall_seconds"] for a in attempts) if complete else None,
                "timing_complete": complete,
            },
        }
        pilot.write_record(root / "FAILURE.json", result)
        return result


def summarize(root: Path, protocol: dict[str, Any], failures: dict[str, Any]) -> dict[str, Any]:
    """Retain all four expected cases without averaging away a missing result."""
    rows = []
    for target in TARGETS:
        path = root / f"{target}__11" / "RESULT.json"
        failed_path = path.with_name("FAILURE.json")
        record = (
            lazy_initializer.read(path)
            if path.exists()
            else failures.get(
                target,
                lazy_initializer.read(failed_path)
                if failed_path.exists()
                else {"status": "pending", "accepted": False},
            )
        )
        rows.append({
            "target_id": target,
            "status": record["status"],
            "accepted": record["accepted"],
            "failure": record.get("failure"),
            "initializer_cpu_seconds": record.get("initializer_cpu_seconds"),
            "physical_invocation_cpu_seconds": record.get("physical_invocation", {}).get("cpu_seconds"),
            "nested_legacy_cpu_seconds": record.get("nested_legacy_cpu_seconds"),
            "rank_two_selected": record.get("rank_two_selected"),
            "checksum": pilot.digest(record) if path.exists() else None,
        })
    summary: dict[str, Any] = {
        "protocol_checksum": pilot.digest(protocol),
        "accepted": all(row["status"] == "complete" and row["accepted"] for row in rows),
        "cases": rows,
    }
    pilot.write_record(root / "SUMMARY.json", summary)
    with (root / "SCOREBOARD.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare all four inputs or execute their previously frozen bounded reproduction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=runner.OUTPUT / "lazy-initializer-reproduction")
    parser.add_argument("--initializer-root", type=Path, default=runner.OUTPUT / "initializer-policy")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four explicitly allocated single-thread workers.")
    if args.prepare_only:
        prepare(args.output, args.initializer_root)
        return 0
    protocol = lazy_initializer.read(args.output / "PROTOCOL.json")
    verify(args.output, protocol)
    failures: dict[str, Any] = {}
    summarize(args.output, protocol, failures)
    jobs = [{"root": str(args.output), "protocol": protocol, "case": case} for case in protocol["cases"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(attempt_case, jobs):
            if result["status"] != "complete":
                failures[result["target_id"]] = result
            summarize(args.output, protocol, failures)
            print(result["target_id"], result["status"], result["accepted"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
