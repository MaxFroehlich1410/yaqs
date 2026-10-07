# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Six-arm development test of conditional physical rank-two initialization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import (
    compilation_experiment,
    construction,
    construction_experiment,
    initializer_policy,
    initializer_training,
    runner,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

VERSION = "overnight.initializer_resource_policy_development.v1"
TARGETS = tuple(construction.TARGETS)
METHODS = tuple(
    f"{source}_{method}" for source in ("legacy", "conditional") for method in ("unchanged", "lbfgs", "adam")
)
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def sources() -> dict[str, str]:
    """Return every frozen source used by construction custody, policy and refinement."""
    result = construction_experiment.sources()
    result.update(runner.source_hashes())
    for name in (
        "initializer_experiment.py",
        "initializer_policy.py",
        "initializer_training.py",
        "INITIALIZER_POLICY_PROTOCOL.md",
        "synthesis.py",
        "compilation.py",
        "compilation_experiment.py",
        "ensemble_experiment.py",
        "ensemble_objective.py",
    ):
        path = Path(__file__).with_name(name)
        result[str(path.relative_to(runner.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def fresh_sources(root: Path) -> list[dict[str, Any]]:
    """Require the complete fresh construction audit without rerunning any construction."""
    protocol = cast("dict[str, Any]", pilot.read_record(root / "BATCH_PROTOCOL.json"))
    summary = cast("dict[str, Any]", pilot.read_record(root / "SUMMARY.json"))
    construction_experiment.verify_batch(protocol)
    if (
        summary["status"] != "complete"
        or not summary["all_archive_reproductions_match"]
        or summary["protocol_checksum"] != pilot.digest(protocol)
    ):
        msg = "Initializer freezing requires the complete fresh construction reproduction."
        raise ValueError(msg)
    result = []
    for case in protocol["cases"]:
        case_root = Path(case["root"])
        final_path, audit_path = case_root / "RESULT.json", case_root / "AUDIT.json"
        if not final_path.exists() or not audit_path.exists():
            msg = "Fresh construction endpoint and audit must exist before initializer freezing."
            raise ValueError(msg)
        audit = cast("dict[str, Any]", pilot.read_record(audit_path))
        manifest = cast("dict[str, Any]", pilot.read_record(case_root / "PROTOCOL.json"))
        # The endpoint exists: this call only authenticates completed blocks and attempt costs.
        built = construction.reproduce(case_root, manifest)
        physical_circuit = pilot.restore_circuit(built["circuit"])
        if (
            built["status"] != "complete"
            or audit["status"] != "complete"
            or not audit["matches"]
            or summary["case_checksums"][case["target_id"]] != pilot.digest(audit)
            or audit["construction_checksum"] != pilot.digest(built)
            or audit["signature"] != pilot.digest({"protocol": protocol, "case": case})
            or not built["timing_complete"]
            or built["construction_cpu_seconds"] is None
            or not np.isfinite(built["construction_cpu_seconds"])
            or built["construction_cpu_seconds"] <= 0
            or not built["archive_reproduction"]["matches"]
            or built["physical_counts"]["one_qubit"] != 78
            or built["physical_counts"]["two_qubit"] != 30
            or built["parameters"] != 108
            or built["updates"] != 570
            or physical_circuit.num_qubits != 6
            or physical_circuit.num_params != 108
            or len(physical_circuit.gates) != 108
            or runner.counts(physical_circuit) != built["physical_counts"]
            or audit["construction_cpu_seconds"] != built["construction_cpu_seconds"]
        ):
            msg = "Fresh legacy construction identity, reproduction or measured cost is incomplete."
            raise ValueError(msg)
        result.append({
            "target_id": case["target_id"],
            "replica": 11,
            "circuit": built["circuit"],
            "theta": built["theta"],
            "construction_cpu_seconds": built["construction_cpu_seconds"],
            "timing_complete": True,
            "files": {
                str(path): pilot.digest(pilot.read_record(path))
                for path in (
                    final_path,
                    audit_path,
                    case_root / "PROTOCOL.json",
                    root / "BATCH_PROTOCOL.json",
                    root / "SUMMARY.json",
                )
            },
        })
    if len(result) != 4 or {s["target_id"] for s in result} != set(TARGETS):
        msg = "Fresh construction panel must retain every fixed development target."
        raise ValueError(msg)
    return result


def prepare(root: Path, construction_root: Path) -> dict[str, Any]:
    """Freeze authenticated fresh legacy sources without synthesizing or training."""
    protocol = {
        "version": VERSION,
        "phase": "development",
        "targets": list(TARGETS),
        "replicas": [11],
        "methods": list(METHODS),
        "threshold": 0.99,
        "config": asdict(initializer_training.Config()),
        "sources": sources(),
        "environment": construction.environment(),
        "benchmark_hashes": runner.custody()["benchmark_hashes"],
        "starts": fresh_sources(construction_root),
    }
    verify(protocol)
    path = root / "PROTOCOL.json"
    if path.exists() and pilot.digest(pilot.read_record(path)) != pilot.digest(protocol):
        msg = "Frozen initializer protocol differs; use a new output root."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return protocol


def verify(protocol: dict[str, Any]) -> None:
    """Reject source, benchmark, panel or numerical changes before any cached result."""
    if (
        protocol["version"] != VERSION
        or protocol["phase"] != "development"
        or protocol["targets"] != list(TARGETS)
        or protocol["replicas"] != [11]
        or protocol["methods"] != list(METHODS)
        or protocol["threshold"] != initializer_policy.OVERLAP_THRESHOLD
        or protocol["config"] != asdict(initializer_training.Config())
        or protocol["sources"] != sources()
        or protocol["environment"] != construction.environment()
        or protocol["benchmark_hashes"] != runner.custody()["benchmark_hashes"]
        or len(protocol["starts"]) != 4
        or {(s["target_id"], s["replica"]) for s in protocol["starts"]} != {(t, 11) for t in TARGETS}
    ):
        msg = "Initializer protocol or numerical custody changed."
        raise ValueError(msg)
    for source in protocol["starts"]:
        if (
            not source["timing_complete"]
            or source["construction_cpu_seconds"] is None
            or source["construction_cpu_seconds"] <= 0
        ):
            msg = "Initializer source has no complete measured construction cost."
            raise ValueError(msg)
        if any(pilot.digest(pilot.read_record(Path(path))) != expected for path, expected in source["files"].items()):
            msg = "Fresh construction source changed after initializer preparation."
            raise ValueError(msg)


def build(root: Path, source: dict[str, Any], target: NDArray[np.complex128], signature: str) -> dict[str, Any]:
    """Freeze one shared initializer build and retain every interrupted setup attempt."""
    signature = pilot.digest([signature, hashlib.sha256(target.tobytes()).hexdigest()])
    path, attempt_path = root / "INITIALIZERS.json", root / "INITIALIZER_ATTEMPTS.json"
    if path.exists():
        saved = cast("dict[str, Any]", pilot.read_record(path))
        if saved["signature"] != signature or saved["attempts_checksum"] != pilot.digest(
            pilot.read_record(attempt_path)
        ):
            msg = "Initializer build or setup-attempt identity changed."
            raise ValueError(msg)
        return saved
    attempts: dict[str, Any] = (
        cast("dict[str, Any]", pilot.read_record(attempt_path))
        if attempt_path.exists()
        else {"signature": signature, "attempts": []}
    )
    if attempts["signature"] != signature:
        msg = "Initializer setup signature changed."
        raise ValueError(msg)
    timed = all(a["cpu_seconds"] is not None for a in attempts["attempts"])
    extra_cpu = sum(a["cpu_seconds"] for a in attempts["attempts"]) if timed else None
    attempt: dict[str, Any] = {"status": "running", "cpu_seconds": None}
    attempts["attempts"].append(attempt)
    pilot.write_record(attempt_path, attempts)
    cpu = time.process_time()
    try:
        result = initializer_policy.build_initializers(target, source)
        attempt["status"] = "complete"
    except BaseException as error:
        attempt["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        attempt["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        attempt["cpu_seconds"] = time.process_time() - cpu
        pilot.write_record(attempt_path, attempts)
    for policy in ("legacy", "conditional"):
        result[policy]["initializer_cpu_seconds"] = (
            result[policy]["initializer_cpu_seconds"] + extra_cpu if extra_cpu is not None else None
        )
    result.update({
        "signature": signature,
        "attempts_checksum": pilot.digest(attempts),
        "shared_interrupted_setup_cpu_seconds": extra_cpu,
        "setup_timing_complete": timed,
    })
    pilot.write_record(path, result)
    return result


def score(root: Path, endpoint: dict[str, Any], target: NDArray[np.complex128]) -> dict[str, Any]:
    """Cache all twelve exact final-channel audits for a compiled physical endpoint."""
    signature = pilot.digest({
        "circuit": endpoint["circuit"],
        "theta": endpoint["theta"],
        "target": hashlib.sha256(target.tobytes()).hexdigest(),
    })
    path = root / "scores" / f"{signature}.json"
    if path.exists():
        saved = cast("dict[str, Any]", pilot.read_record(path))
        if saved["signature"] != signature:
            msg = "Initializer endpoint score identity changed."
            raise ValueError(msg)
        return saved
    result = compilation_experiment.score(
        pilot.restore_circuit(endpoint["circuit"]), np.asarray(endpoint["theta"]), target
    )
    result.update({
        "signature": signature,
        "mean_noisy_fidelity": float(
            np.mean([result["fidelities"][n] for n in compilation_experiment.channels.NOISE_IDS[1:]])
        ),
    })
    pilot.write_record(path, result)
    return result


def select_point(curve: list[dict[str, Any]], offset: float, budget: float) -> dict[str, Any] | None:
    """Return only an available deployed endpoint, never an interpolated or best state."""
    return next((p for p in reversed(curve) if offset + p["refinement_cpu_seconds"] <= budget), None)


def common_panel(
    root: Path,
    trained: dict[str, dict[str, Any]],
    initials: dict[str, Any],
    target: NDArray[np.complex128],
    *,
    total: bool,
) -> dict[str, Any]:
    """Compare four trained arms on total-resource or separately labeled refinement clocks."""
    if len(trained) != 4 or any(not t["timing_complete"] for t in trained.values()):
        return {"status": "incomplete_training_or_timing", "points": []}
    offsets = {name: initials[name.split("_")[0]]["initializer_cpu_seconds"] if total else 0.0 for name in trained}
    if any(value is None for value in offsets.values()):
        return {"status": "unknown_initializer_timing", "points": []}
    common = min(offsets[name] + t["refinement_cpu_seconds"] for name, t in trained.items())
    points = []
    for fraction in FRACTIONS:
        selected = {}
        for name, training in trained.items():
            point = select_point(training["curve"], offsets[name], fraction * common)
            if point is None:
                selected[name] = {"status": "initializer_not_yet_available"}
                continue
            path = root / name / "deployments" / f"{point['deployment']}.json"
            endpoint = cast("dict[str, Any]", pilot.read_record(path))["endpoint"]
            selected[name] = {"status": "available", "point": point, "scores": score(root, endpoint, target)}
        points.append({
            "fraction": fraction,
            "cpu_seconds": fraction * common,
            "all_available": all(p["status"] == "available" for p in selected.values()),
            "arms": selected,
        })
    return {
        "status": "complete",
        "clock": "total_construction_and_refinement" if total else "refinement_only_including_deployment",
        "common_cpu_seconds": common,
        "points": points,
    }


def execute(
    root: Path,
    source: dict[str, Any],
    target: NDArray[np.complex128],
    signature: str,
    config: initializer_training.Config,
) -> dict[str, Any]:
    """Build both policies once and execute every independent fixed-objective arm."""
    initials = build(root, source, target, signature)
    arms, trained = {}, {}

    def refine(arm_root: Path, start: dict[str, Any], method: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if method == "unchanged":
            return None, score(root, start, target)
        training = initializer_training.train(arm_root, start, target, method, config)
        endpoint = cast(
            "dict[str, Any]", pilot.read_record(arm_root / "deployments" / f"{training['final_deployment']}.json")
        )["endpoint"]
        return training, score(root, endpoint, target)

    for name in METHODS:
        policy, method = name.split("_")
        start = initials[policy]
        arm_root = root / name
        failure_path = arm_root / "FAILED.json"
        failure_signature = pilot.digest([signature, name, start])
        if failure_path.exists():
            failed = cast("dict[str, Any]", pilot.read_record(failure_path))
            if failed["signature"] != failure_signature:
                msg = "Retained initializer arm failure identity changed."
                raise ValueError(msg)
            arms[name] = failed
            continue
        try:
            training, scores = refine(arm_root, start, method)
        except (ArithmeticError, RuntimeError) as error:
            arms[name] = {
                "status": "failed",
                "signature": failure_signature,
                "failure": f"{type(error).__name__}: {error}",
            }
            pilot.write_record(failure_path, arms[name])
            continue
        if training is not None:
            trained[name] = training
        refinement = 0.0 if training is None else training["refinement_cpu_seconds"]
        total = (
            start["initializer_cpu_seconds"] + refinement
            if start["initializer_cpu_seconds"] is not None and refinement is not None
            else None
        )
        arms[name] = {
            "status": "complete",
            "scores": scores,
            "selected_source": start["selected_source"],
            "initializer_cpu_seconds": start["initializer_cpu_seconds"],
            "refinement_cpu_seconds": refinement,
            "total_cpu_seconds": total,
            "training_checksum": None if training is None else pilot.digest(training),
            "individual_channel_gradient_calls": 0
            if training is None
            else training["individual_channel_gradient_calls"],
            "training_cpu_seconds": 0.0 if training is None else training["training_cpu_seconds"],
            "deployment_cpu_seconds": 0.0 if training is None else training["deployment_cpu_seconds"],
        }
    return {
        "status": "complete" if all(a["status"] == "complete" for a in arms.values()) else "partial_failure",
        "signature": signature,
        "target_id": source["target_id"],
        "replica": 11,
        "initializer_checksum": pilot.digest(initials),
        "ideal_tt_overlap": initials["ideal_tt_overlap"],
        "rank_two_selected": initials["rank_two_selected"],
        "raw_legacy_counts": initials["raw_legacy_counts"],
        "raw_rank_two_counts": initials["raw_rank_two_counts"],
        "initializer_costs": initials["costs"],
        "arms": arms,
        "common_total_cpu": common_panel(root, trained, initials, target, total=True),
        "common_refinement_cpu": common_panel(root, trained, initials, target, total=False),
    }


def run_case(job: dict[str, Any]) -> dict[str, Any]:
    """Run or authenticate one source-complete development case without family exclusion."""
    protocol, source, root = job["protocol"], job["source"], Path(job["root"])
    verify(protocol)
    if pilot.digest(source) not in {pilot.digest(s) for s in protocol["starts"]}:
        msg = "Initializer case is outside its fixed development panel."
        raise ValueError(msg)
    signature = pilot.digest({"protocol": protocol, "source": source})
    path = root / "RESULT.json"
    if path.exists():
        record = cast("dict[str, Any]", pilot.read_record(path))
        if record["signature"] != signature:
            msg = "Initializer completed case identity changed."
            raise ValueError(msg)
        if any(
            hashlib.sha256((root / name).read_bytes()).hexdigest() != expected
            for name, expected in record["dependencies"].items()
        ):
            msg = "Completed initializer checkpoint, score or setup artifact changed."
            raise ValueError(msg)
        return record
    target = load_target(6, source["target_id"]).state_vector_copy()
    try:
        result = execute(root, source, target, signature, initializer_training.Config(**protocol["config"]))
    except (ArithmeticError, RuntimeError) as error:
        result = {
            "status": "failed",
            "signature": signature,
            "target_id": source["target_id"],
            "replica": 11,
            "failure": f"{type(error).__name__}: {error}",
            "arms": {name: {"status": "failed"} for name in METHODS},
        }
    verify(protocol)
    result["dependencies"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.json") if p != path
    }
    pilot.write_record(path, result)
    return result


def write_summary(root: Path, protocol: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain every target, arm and condition with compact pointers to full case audits."""
    by_target = {r["target_id"]: r for r in records}
    if len(by_target) != len(records) or set(by_target) - set(TARGETS):
        msg = "Initializer summary requires unique members of the fixed target panel."
        raise ValueError(msg)
    rows = []
    for target_id in TARGETS:
        case = by_target.get(target_id, {})
        for name in METHODS:
            arm = case.get("arms", {}).get(name, {"status": "pending"})
            scores = arm.get("scores", {})
            for noise in compilation_experiment.channels.NOISE_IDS:
                physical = scores.get("physical_counts", {})
                lower = scores.get("ballarin_counts", {}) if noise == "ballarin_coupled" else physical
                rows.append({
                    "target_id": target_id,
                    "replica": 11,
                    "method": name,
                    "noise_id": noise,
                    "status": arm["status"],
                    "fidelity": scores.get("fidelities", {}).get(noise),
                    "ideal_tt_overlap": case.get("ideal_tt_overlap"),
                    "rank_two_selected": case.get("rank_two_selected"),
                    "selected_source": arm.get("selected_source"),
                    "physical_one_qubit": physical.get("one_qubit"),
                    "physical_two_qubit": physical.get("two_qubit"),
                    "lowered_one_qubit": lower.get("one_qubit"),
                    "lowered_two_qubit": lower.get("two_qubit"),
                    "initializer_cpu_seconds": arm.get("initializer_cpu_seconds"),
                    "training_cpu_seconds": arm.get("training_cpu_seconds"),
                    "deployment_cpu_seconds": arm.get("deployment_cpu_seconds"),
                    "total_cpu_seconds": arm.get("total_cpu_seconds"),
                    "individual_channel_gradient_calls": arm.get("individual_channel_gradient_calls"),
                    "failure": arm.get("failure", case.get("failure")),
                })
    complete = len(records) == 4
    means = {}
    for name in METHODS:
        arms = [r["arms"][name] for r in records]
        means[name] = (
            float(np.mean([a["scores"]["mean_noisy_fidelity"] for a in arms]))
            if complete and all(a["status"] == "complete" for a in arms)
            else None
        )
    summary: dict[str, Any] = {
        "status": "complete"
        if complete and all(r["status"] == "complete" for r in records)
        else "partial_failure"
        if complete
        else "running",
        "protocol_checksum": pilot.digest(protocol),
        "case_checksums": {r["target_id"]: pilot.digest(r) for r in records},
        "equal_target_all_noisy_means": means,
        "selection": {
            t: {"ideal_tt_overlap": r.get("ideal_tt_overlap"), "rank_two_selected": r.get("rank_two_selected")}
            for t, r in by_target.items()
        },
    }
    pilot.write_record(root / "SUMMARY.json", summary)
    with (root / "SCOREBOARD.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Freeze sources or explicitly execute the six-arm development resource-policy screen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=runner.OUTPUT / "initializer-policy")
    parser.add_argument("--construction-root", type=Path, default=runner.OUTPUT / "construction-reproduction")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four explicitly allocated workers.")
    if args.prepare_only:
        prepare(args.output, args.construction_root)
        print(f"Prepared only: {args.output / 'PROTOCOL.json'}", flush=True)
        return 0
    protocol = cast("dict[str, Any]", pilot.read_record(args.output / "PROTOCOL.json"))
    verify(protocol)
    jobs = [
        {"protocol": protocol, "source": source, "root": str(args.output / f"{source['target_id']}__11")}
        for source in protocol["starts"]
    ]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(run_case, jobs):
            records.append(record)
            print(record["target_id"], record["status"], flush=True)
            write_summary(args.output, protocol, records)
    return 0 if all(r["status"] == "complete" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
