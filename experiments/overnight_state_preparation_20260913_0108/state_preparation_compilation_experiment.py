# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Authenticate and audit the fixed development panel of known-zero compilations."""

from __future__ import annotations

import argparse
import csv
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import (
    channels,
    runner,
)
from experiments.overnight_state_preparation_20260913_0108 import (
    composed_experiment as composition,
)
from experiments.overnight_state_preparation_20260913_0108 import (
    long_ensemble_experiment as measured,
)
from experiments.overnight_state_preparation_20260913_0108 import (
    state_preparation_compilation as compiler,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

VERSION = "overnight.zero_state_eigenstate_panel.v1"
TARGETS = tuple(composition.TARGETS)
LABELS = ("legacy_initializer", "conditional_initializer", *composition.PRIMARY)
CPU_SCOPE = "inclusive pair invocation, excluding source verification and final result/attempt writes"


def record(path: Path) -> dict[str, Any]:
    """Read a required JSON object without invoking any scientific operation."""
    return cast("dict[str, Any]", pilot.read_record(path))


def file_hash(path: Path) -> str:
    """Return a raw artifact checksum."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependencies_valid(root: Path, dependencies: dict[str, str]) -> bool:
    """Authenticate descendant files without permitting path traversal."""
    return all(
        (root / name).resolve().is_relative_to(root.resolve()) and file_hash(root / name) == digest
        for name, digest in dependencies.items()
    )


def artifact_dependencies(root: Path) -> dict[str, str]:
    """Return every pair child JSON checksum, including all nested score results."""
    return {
        str(path.relative_to(root)): file_hash(path) for path in root.rglob("*.json") if path != root / "RESULT.json"
    }


def source_panel(composition_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Read all24 authenticated final endpoints, never compiling or scoring them.

    Raises:
        ValueError: If any required source, dependency, metadata or panel identity changed.
    """
    original = record(composition_root / "PROTOCOL.json")
    composition.verify(original)
    summary = record(composition_root / "SUMMARY.json")
    if summary["status"] != "complete" or summary["protocol_checksum"] != pilot.digest(original):
        msg = "The complete composition panel is required before compiler preparation."
        raise ValueError(msg)
    initializer_root = Path(original["initializer_root"])
    pairs, files = [], dict(original["input_files"])
    for source in original["starts"]:
        target = source["target_id"]
        case_root = composition_root / f"{target}__11"
        case_path = case_root / "RESULT.json"
        case = record(case_path)
        if (
            case["signature"] != pilot.digest([original, source])
            or case["status"] != "complete"
            or case["target_id"] != target
            or case["replica"] != 11
            or summary["case_checksums"][target] != pilot.digest(case)
            or not dependencies_valid(case_root, case["dependencies"])
        ):
            msg = "A composition source case is incomplete or differs from its authenticated dependencies."
            raise ValueError(msg)
        initializers = record(initializer_root / f"{target}__11" / "INITIALIZERS.json")
        endpoints = {f"{name}_initializer": initializers[name] for name in ("legacy", "conditional")}
        for method in composition.PRIMARY:
            arm = case["arms"][method]
            if arm["status"] != "complete":
                msg = "Every structural final endpoint must be complete."
                raise ValueError(msg)
            point = arm["final"]
            circuit_path = case_root / "circuits" / f"{point['circuit_hash']}.json"
            circuit = record(circuit_path)
            if pilot.digest(circuit) != point["circuit_hash"]:
                msg = "The physical source circuit differs from its content identity."
                raise ValueError(msg)
            endpoints[method] = {"circuit": circuit, "theta": point["theta"]}
        for label in LABELS:
            endpoint = {key: endpoints[label][key] for key in ("circuit", "theta")}
            circuit = compiler.restore(endpoint["circuit"])
            compiler.validate(circuit, np.asarray(endpoint["theta"]))
            if circuit.num_qubits != 6:
                msg = "The scientific compiler panel contains only fixed q6 development sources."
                raise ValueError(msg)
            pairs.append({
                "target_id": target,
                "replica": 11,
                "label": label,
                "endpoint": endpoint,
                "source_case_checksum": pilot.digest(case),
                "target_sha256": hashlib.sha256(load_target(6, target).state_vector_copy().tobytes()).hexdigest(),
            })
        files[str(case_path.resolve())] = file_hash(case_path)
        for name, digest in case["dependencies"].items():
            files[str((case_root / name).resolve())] = digest
    for name in ("PROTOCOL.json", "SUMMARY.json"):
        path = composition_root / name
        files[str(path.resolve())] = file_hash(path)
    if len(pairs) != 24 or {(p["target_id"], p["label"], p["replica"]) for p in pairs} != {
        (target, label, 11) for target in TARGETS for label in LABELS
    }:
        msg = "The compiler diagnostic requires exactly all24 fixed source pairs."
        raise ValueError(msg)
    return pairs, files


def sources() -> dict[str, str]:
    """Pin compiler, exact score/attempt helpers and every authenticated source primitive."""
    result = {**composition.sources(), **measured.numerical_sources()}
    for name in (
        "state_preparation_compilation.py",
        "state_preparation_compilation_experiment.py",
        "STATE_PREPARATION_COMPILATION_PROTOCOL.md",
    ):
        path = Path(__file__).with_name(name)
        result[str(path.relative_to(runner.REPO))] = file_hash(path)
    return result


def prepare(output: Path, composition_root: Path) -> dict[str, Any]:
    """Freeze complete source records before any scientific compiler call."""
    pairs, files = source_panel(composition_root)
    protocol = {
        "version": VERSION,
        "phase": "development",
        "composition_root": str(composition_root.resolve()),
        "pairs": pairs,
        "input_files": files,
        "sources": sources(),
        "benchmark_hashes": runner.custody()["benchmark_hashes"],
        "environment": composition.initializer_experiment.construction.environment(),
        "targets": list(TARGETS),
        "labels": list(LABELS),
        "replicas": [11],
        "noise_ids": list(channels.NOISE_IDS),
        "tolerance": compiler.EIGENSTATE_ATOL,
        "source_selection": "all fixed method final endpoints; no quality selection",
        "cpu_scope": CPU_SCOPE,
    }
    path = output / "PROTOCOL.json"
    if path.exists() and pilot.digest(record(path)) != pilot.digest(protocol):
        msg = "Frozen compiler inputs changed; use a new output root."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return protocol


def verify(protocol: dict[str, Any]) -> None:
    """Reject changes to source records, target identities, code or fixed semantics."""
    pairs, files = source_panel(Path(protocol["composition_root"]))
    if (
        protocol["version"] != VERSION
        or protocol["phase"] != "development"
        or pilot.digest(protocol["pairs"]) != pilot.digest(pairs)
        or protocol["input_files"] != files
        or protocol["sources"] != sources()
        or protocol["benchmark_hashes"] != runner.custody()["benchmark_hashes"]
        or protocol["environment"] != composition.initializer_experiment.construction.environment()
        or protocol["targets"] != list(TARGETS)
        or protocol["labels"] != list(LABELS)
        or protocol["replicas"] != [11]
        or protocol["noise_ids"] != list(channels.NOISE_IDS)
        or protocol["tolerance"] != compiler.EIGENSTATE_ATOL
        or protocol["source_selection"] != "all fixed method final endpoints; no quality selection"
        or protocol["cpu_scope"] != CPU_SCOPE
    ):
        msg = "The frozen compiler protocol or source custody changed."
        raise ValueError(msg)


def execute_pair(root: Path, source: dict[str, Any]) -> dict[str, Any]:
    """Compile one endpoint, then independently evaluate both physical circuits."""
    endpoint = source["endpoint"]
    circuit, theta = compiler.restore(endpoint["circuit"]), np.asarray(endpoint["theta"])
    target = load_target(circuit.num_qubits, source["target_id"]).state_vector_copy()
    if hashlib.sha256(target.tobytes()).hexdigest() != source["target_sha256"]:
        msg = "The compiler target vector changed."
        raise ValueError(msg)
    compilation = composition.tracked_call(
        root / "COMPILATION.json",
        [VERSION, source],
        lambda: compiler.compile_endpoint(circuit, theta),
    )
    result = compilation["value"]
    compiled = compiler.restore(result["circuit"])
    before = measured.score_point(root / "before", circuit, theta, target)
    after = measured.score_point(root / "after", compiled, np.asarray(result["theta"]), target)
    delta = {noise: after["fidelities"][noise] - before["fidelities"][noise] for noise in channels.NOISE_IDS}
    attempts = record(root / "COMPILATION.attempts.json")["attempts"]
    return {
        "status": "complete",
        "source": source,
        "compilation": compilation,
        "before": before,
        "after": after,
        "fidelity_delta": delta,
        "mean_noisy_delta": float(np.mean([delta[noise] for noise in channels.NOISE_IDS[1:]])),
        "compiler_attempts": len(attempts),
        "compiler_completed_attempts": sum(a["status"] == "complete" for a in attempts),
        "failed_compiler_gate_work": "unknown partial count; all observed CPU charged"
        if any(a["status"] != "complete" for a in attempts)
        else "none",
        "scoring": {side: measured.physical_scoring(root / side) for side in ("before", "after")},
        "accounting": "inclusive invocation contains compiler and scoring costs; do not add nested CPU twice",
    }


def run_pair(job: dict[str, Any]) -> dict[str, Any]:
    """Run or authenticate one signed pair, retaining all replay and child artifacts."""
    protocol, source, root = job["protocol"], job["source"], Path(job["root"])
    verify(protocol)
    if pilot.digest(source) not in {pilot.digest(pair) for pair in protocol["pairs"]}:
        msg = "The compiler source lies outside the fixed panel."
        raise ValueError(msg)
    signature = pilot.digest([protocol, source])
    path = root / "RESULT.json"
    if path.exists():
        saved = record(path)
        if saved["signature"] != signature or saved["dependencies"] != artifact_dependencies(root):
            msg = "A compiler result or durable dependency changed."
            raise ValueError(msg)
        authenticate_result(root, source, saved)
        return saved
    result = composition.measured_execute(root, signature, lambda: execute_pair(root, source))
    verify(protocol)
    result.update(signature=signature, target_id=source["target_id"], replica=11, label=source["label"])
    result["dependencies"] = artifact_dependencies(root)
    pilot.write_record(path, result)
    return result


def authenticate_result(root: Path, source: dict[str, Any], result: dict[str, Any]) -> None:
    """Bind cached result cells to their immutable compiler, score and cost records."""
    compiled = record(root / "COMPILATION.json")
    compiler_attempts = record(root / "COMPILATION.attempts.json")["attempts"]
    if (
        pilot.digest(result["source"]) != pilot.digest(source)
        or pilot.digest(result["compilation"]) != pilot.digest(compiled)
        or compiled["signature"] != pilot.digest([VERSION, source])
        or result["status"] != "complete"
        or result["compiler_attempts"] != len(compiler_attempts)
        or result["compiler_completed_attempts"] != sum(a["status"] == "complete" for a in compiler_attempts)
    ):
        msg = "Cached compiler source or compilation result changed."
        raise ValueError(msg)
    endpoints = {"before": source["endpoint"], "after": compiled["value"]}
    for side, endpoint in endpoints.items():
        signature = pilot.digest({
            "version": measured.VERSION,
            "circuit": endpoint["circuit"],
            "theta": endpoint["theta"],
            "target": source["target_sha256"],
        })
        scored = record(root / side / "scores" / signature / "RESULT.json")
        if pilot.digest(result[side]) != pilot.digest(scored) or result["scoring"][side] != measured.physical_scoring(
            root / side
        ):
            msg = "Cached compiler fidelity cells differ from their score artifacts."
            raise ValueError(msg)
    delta = {
        noise: result["after"]["fidelities"][noise] - result["before"]["fidelities"][noise]
        for noise in channels.NOISE_IDS
    }
    ledger = record(root / "CASE_ATTEMPTS.json")
    if (
        result["fidelity_delta"] != delta
        or pilot.digest(result["mean_noisy_delta"])
        != pilot.digest(float(np.mean([delta[noise] for noise in channels.NOISE_IDS[1:]])))
        or result["physical_invocation"]["attempts_checksum"] != pilot.digest(ledger)
        or result["physical_invocation"]["cpu_seconds"] != sum(a["cpu_seconds"] for a in ledger["attempts"])
    ):
        msg = "Cached compiler deltas or inclusive cost changed."
        raise ValueError(msg)


def pair_key(source: dict[str, Any]) -> str:
    """Return the unique fixed identity used for cases and missing-result rows."""
    return f"{source['target_id']}__11/{source['label']}"


def write_summary(root: Path, protocol: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve all pairs and channels, with no quality-selected or omitted cases."""
    lookup = {pair_key(result): result for result in results}
    if len(lookup) != len(results) or not set(lookup) <= {pair_key(pair) for pair in protocol["pairs"]}:
        msg = "The compiler summary contains duplicate or unregistered source identities."
        raise ValueError(msg)
    rows = []
    for pair in protocol["pairs"]:
        result = lookup.get(pair_key(pair))
        for noise in channels.NOISE_IDS:
            for side in ("before", "after"):
                scored = result[side] if result else {}
                counts = scored.get("ballarin_counts" if noise == "ballarin_coupled" else "logical_counts", {})
                rows.append({
                    "target_id": pair["target_id"],
                    "replica": 11,
                    "label": pair["label"],
                    "endpoint": side,
                    "noise_id": noise,
                    "status": "complete" if result else "missing_or_failed",
                    "fidelity": scored.get("fidelities", {}).get(noise),
                    "fidelity_delta": result["fidelity_delta"][noise] if result else None,
                    "one_qubit": counts.get("one_qubit"),
                    "two_qubit": counts.get("two_qubit"),
                    "depth": counts.get("depth"),
                    "parameters": len(scored.get("theta", [])) if result else None,
                    "compiler_cpu_seconds": result["compilation"]["cpu_seconds"] if result else None,
                    "evaluation_cpu_seconds": scored.get("cpu_seconds"),
                    "inclusive_pair_cpu_seconds": result["physical_invocation"]["cpu_seconds"] if result else None,
                })
    complete = len(lookup) == len(protocol["pairs"])
    summary: dict[str, Any] = {
        "version": VERSION,
        "status": "complete" if complete else "partial_failure",
        "protocol_checksum": pilot.digest(protocol),
        "case_checksums": {key: pilot.digest(value) for key, value in lookup.items()},
        "missing_pairs": [pair_key(pair) for pair in protocol["pairs"] if pair_key(pair) not in lookup],
        "complete_panel_mean_noisy_delta": float(np.mean([r["mean_noisy_delta"] for r in results]))
        if complete
        else None,
        "label_mean_noisy_delta": {
            label: float(np.mean([r["mean_noisy_delta"] for r in results if r["label"] == label])) if complete else None
            for label in LABELS
        },
        "target_mean_noisy_delta": {
            target: float(np.mean([r["mean_noisy_delta"] for r in results if r["target_id"] == target]))
            if complete
            else None
            for target in TARGETS
        },
        "cpu_scope": CPU_SCOPE,
        "no_optimizer_novelty_claim": True,
    }
    mean_delta = summary["complete_panel_mean_noisy_delta"]
    summary["broad_adoption_retired"] = mean_delta < -1e-10 if mean_delta is not None else None
    summary["negative_label_means"] = [
        label for label, value in summary["label_mean_noisy_delta"].items() if value is not None and value < 0
    ]
    summary["negative_target_means"] = [
        target for target, value in summary["target_mean_noisy_delta"].items() if value is not None and value < 0
    ]
    pilot.write_record(root / "SUMMARY.json", summary)
    with (root / "scoreboard.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def attempt_pair(job: dict[str, Any]) -> dict[str, Any]:
    """Return a complete result or preserve the explicit failed source identity."""
    try:
        return run_pair(job)
    except Exception as error:
        source = job["source"]
        failure = {
            "status": "failed",
            "target_id": source["target_id"],
            "label": source["label"],
            "replica": 11,
            "failure": f"{type(error).__name__}: {error}",
        }
        # Completed case folders are immutable, including after a failed custody check.
        if not (Path(job["root"]) / "RESULT.json").exists():
            pilot.write_record(Path(job["root"]) / "FAILURE.json", failure)
        return failure


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare or run the fixed development panel with at most four workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--composition-root", type=Path)
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 5))
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        if args.composition_root is None:
            parser.error("--composition-root is required for preparation")
        prepare(args.output, args.composition_root)
        return 0
    protocol = record(args.output / "PROTOCOL.json")
    verify(protocol)
    jobs: list[dict[str, Any]] = [
        {"protocol": protocol, "source": pair, "root": str(args.output / pair_key(pair))} for pair in protocol["pairs"]
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        attempted = list(pool.map(attempt_pair, jobs))
    results = [result for result in attempted if result["status"] == "complete"]
    failures: dict[str, Any] = {pair_key(result): result for result in attempted if result["status"] != "complete"}
    pilot.write_record(args.output / "FAILURES.json", failures)
    write_summary(args.output, protocol, results)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
