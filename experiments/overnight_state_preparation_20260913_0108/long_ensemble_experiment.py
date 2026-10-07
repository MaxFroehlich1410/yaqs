# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Prepare or run the fixed four-arm full-family convergence diagnostic.

The numerical optimizer is the frozen ensemble training primitive. New durable
invocation markers reject an unobserved process death, while caught interruption
work and complete optimizer history remain recoverable by that primitive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from benchmarks.state_preparation.ballarin import materialize_ballarin_circuit
from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import channels, ensemble_objective, runner
from experiments.overnight_state_preparation_20260913_0108 import ensemble_experiment as base

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

VERSION = "overnight.long_ensemble_convergence.v1"
TARGETS = base.TARGETS
CPU_SCOPE = (
    "frozen train_arm optimization CPU including checkpoint writes, failed line searches and observed replay; "
    "wrapper, final/attempt record writes, construction, imports, custody and all12 scoring excluded; "
    "wrapper/scoring CPU reported separately; same inclusion basis for all four arms"
)


@dataclass(frozen=True)
class Config(base.Config):
    """Use the frozen primitive defaults with the single declared longer cap."""

    channel_evaluations: int = 6050


def arm_specs(config: Config) -> dict[str, dict[str, Any]]:
    """Return four optimizer/objective combinations with equal component caps."""
    return {
        f"{objective}_{optimizer}": {
            "optimizer": optimizer,
            "noise_ids": list(profiles),
            "joint_cap": config.channel_evaluations // len(profiles),
        }
        for optimizer in ("lbfgs", "adam")
        for objective, profiles in (
            ("mean2", ("depolarizing_1s_all", "dephasing_1s_all")),
            ("mean11", ensemble_objective.NOISY_IDS),
        )
    }


def numerical_sources() -> dict[str, str]:
    """Return the frozen primitive dependencies and this new driver/protocol."""
    result = base.numerical_sources()
    for name in ("long_ensemble_experiment.py", "LONG_ENSEMBLE_PROTOCOL.md"):
        path = Path(__file__).with_name(name)
        result[str(path.relative_to(runner.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def invocation_ledger(path: Path, signature: str) -> dict[str, Any]:
    """Return a compatible observed ledger, rejecting unknown process-death cost.

    Raises:
        ValueError: If an identity changed or a running invocation never closed.
    """
    ledger: dict[str, Any] = (
        cast("dict[str, Any]", pilot.read_record(path)) if path.exists() else {"signature": signature, "attempts": []}
    )
    if ledger["signature"] != signature or any(row["status"] == "running" for row in ledger["attempts"]):
        msg = "Invocation identity changed or unknown kill cost remains; this case cannot be resumed or compared."
        raise ValueError(msg)
    return ledger


def train_arm(
    root: Path,
    circuit: ParameterizedCircuit,
    initial: NDArray,
    target: NDArray,
    spec: dict[str, Any],
    config: Config,
) -> dict[str, Any]:
    """Wrap the unchanged training primitive with durable process-death detection.

    Raises:
        ValueError: If a cached primitive result lacks an authenticated attempt ledger.
    """
    signature = pilot.digest({
        "version": VERSION,
        "circuit": pilot.circuit_record(circuit),
        "theta": initial.tolist(),
        "target": hashlib.sha256(target.tobytes()).hexdigest(),
        "spec": spec,
        "config": asdict(config),
    })
    path = root / "INVOCATIONS.json"
    ledger = invocation_ledger(path, signature)
    final_path = root / "TRAINING.json"
    if final_path.exists():
        if not ledger["attempts"]:
            msg = "Cached training lacks its invocation ledger."
            raise ValueError(msg)
        result = base.train_arm(root, circuit, initial, target, spec, config)
    else:
        entry: dict[str, Any] = {"status": "running", "cpu_seconds": None, "wall_seconds": None}
        ledger["attempts"].append(entry)
        pilot.write_record(path, ledger)
        cpu, wall = time.process_time(), time.perf_counter()
        try:
            result = base.train_arm(root, circuit, initial, target, spec, config)
            entry["status"] = "complete"
        except BaseException as error:
            entry.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed")
            entry["failure"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            entry.update(cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - wall)
            pilot.write_record(path, ledger)
    attempts = pilot.read_record(root / "ATTEMPTS.json")
    if result["attempts_checksum"] != pilot.digest(attempts):
        msg = "Primitive training attempt ledger changed."
        raise ValueError(msg)
    invocation_cpu = sum(row["cpu_seconds"] for row in ledger["attempts"])
    return {
        **result,
        "invocations_checksum": pilot.digest(ledger),
        "invocation_cpu_seconds": invocation_cpu,
        "invocation_overhead_cpu_seconds": invocation_cpu - result["cpu_seconds"],
        "cpu_scope": CPU_SCOPE,
    }


def score_point(root: Path, circuit: ParameterizedCircuit, theta: NDArray, target: NDArray) -> dict[str, Any]:
    """Cache all12 exact scores and record failed/replayed forward component work.

    Raises:
        ValueError: If a cached score or its physical attempt ledger changed.
    """
    signature = pilot.digest({
        "version": VERSION,
        "circuit": pilot.circuit_record(circuit),
        "theta": theta.tolist(),
        "target": hashlib.sha256(target.tobytes()).hexdigest(),
    })
    folder = root / "scores" / signature
    path, attempts_path = folder / "RESULT.json", folder / "ATTEMPTS.json"
    ledger = invocation_ledger(attempts_path, signature)
    if path.exists():
        result = cast("dict[str, Any]", pilot.read_record(path))
        if result["signature"] != signature or result["attempts_checksum"] != pilot.digest(ledger):
            msg = "Cached long-ensemble score or attempt ledger changed."
            raise ValueError(msg)
        return result
    entry: dict[str, Any] = {
        "status": "running",
        "attempted_channel_calls": 0,
        "completed_channel_calls": 0,
        "cpu_seconds": None,
        "wall_seconds": None,
    }
    ledger["attempts"].append(entry)
    pilot.write_record(attempts_path, ledger)
    cpu, wall = time.process_time(), time.perf_counter()
    fidelities = {}
    try:
        for noise in channels.NOISE_IDS:
            entry["attempted_channel_calls"] += 1
            fidelities[noise] = channels.evaluate(circuit, theta, target, noise)
            entry["completed_channel_calls"] += 1
        logical, physical = runner.counts(circuit), runner.counts(materialize_ballarin_circuit(circuit, theta).circuit)
        entry["status"] = "complete"
    except BaseException as error:
        entry.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed")
        entry["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        entry.update(cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - wall)
        pilot.write_record(attempts_path, ledger)
    result = {
        "signature": signature,
        "status": "complete",
        "theta": theta.tolist(),
        "fidelities": fidelities,
        "mean_noisy_fidelity": float(np.mean([fidelities[key] for key in ensemble_objective.NOISY_IDS])),
        "logical_counts": logical,
        "ballarin_counts": physical,
        "forward_channel_evaluations": 12,
        "physical_attempted_channel_calls": sum(row["attempted_channel_calls"] for row in ledger["attempts"]),
        "physical_completed_channel_calls": sum(row["completed_channel_calls"] for row in ledger["attempts"]),
        "density_direction_passes": sum(row["completed_channel_calls"] for row in ledger["attempts"]),
        "cpu_seconds": sum(row["cpu_seconds"] for row in ledger["attempts"]),
        "wall_seconds": sum(row["wall_seconds"] for row in ledger["attempts"]),
        "attempts_checksum": pilot.digest(ledger),
    }
    pilot.write_record(path, result)
    return result


def physical_scoring(root: Path) -> dict[str, Any]:
    """Return all attempted scoring work, including points without a final result."""
    paths = sorted((root / "scores").glob("*/ATTEMPTS.json"))
    ledgers = [cast("dict[str, Any]", pilot.read_record(path)) for path in paths]
    attempts = [attempt for ledger in ledgers for attempt in ledger["attempts"]]
    return {
        "attempted_parameter_points": len(paths),
        "complete_parameter_points": len(list((root / "scores").glob("*/RESULT.json"))),
        "physical_attempted_channel_calls": sum(row["attempted_channel_calls"] for row in attempts),
        "physical_completed_channel_calls": sum(row["completed_channel_calls"] for row in attempts),
        "density_direction_passes": sum(row["completed_channel_calls"] for row in attempts),
        "cpu_seconds": sum(row["cpu_seconds"] or 0 for row in attempts),
        "unknown_kill_cost": any(row["status"] == "running" for row in attempts),
        "checksums": [pilot.digest(ledger) for ledger in ledgers],
    }


def physical_training(root: Path) -> dict[str, Any]:
    """Return observed training work even when no complete optimizer result exists."""
    path, invocation_path = root / "ATTEMPTS.json", root / "INVOCATIONS.json"
    attempts = cast("dict[str, Any]", pilot.read_record(path))["attempts"] if path.exists() else []
    invocations = (
        cast("dict[str, Any]", pilot.read_record(invocation_path))["attempts"] if invocation_path.exists() else []
    )
    return {
        "physical_attempted_channel_calls": sum(row["attempted_channel_calls"] for row in attempts),
        "physical_completed_channel_calls": sum(row["completed_channel_calls"] for row in attempts),
        "physical_completed_direction_passes": 2 * sum(row["completed_channel_calls"] for row in attempts),
        "known_training_cpu_seconds": sum(row["cpu_seconds"] for row in attempts),
        "known_invocation_cpu_seconds": sum(row["cpu_seconds"] or 0 for row in invocations),
        "unknown_kill_cost": any(row["status"] == "running" for row in invocations),
    }


def run_case(
    root: Path, source: dict[str, Any], config: Config, *, order_index: int = 0, protocol_checksum: str | None = None
) -> dict[str, Any]:
    """Run the complete fixed-start panel with final and chronological CPU scores.

    Raises:
        ValueError: If a completed case or any durable artifact has changed.
    """
    circuit = pilot.restore_circuit(source["candidate"]["circuit"])
    initial = np.asarray(source["candidate"]["theta"], dtype=float)
    target = load_target(circuit.num_qubits, source["target_id"]).state_vector_copy()
    signature = pilot.digest({
        "version": VERSION,
        "source": source,
        "config": asdict(config),
        "target": hashlib.sha256(target.tobytes()).hexdigest(),
        "order_index": order_index,
        "protocol_checksum": protocol_checksum,
    })
    path = root / "RESULT.json"
    if path.exists():
        result = cast("dict[str, Any]", pilot.read_record(path))
        if result["signature"] != signature or any(
            hashlib.sha256((root / name).read_bytes()).hexdigest() != value
            for name, value in result["artifact_hashes"].items()
        ):
            msg = "Completed long-ensemble case or durable artifact changed."
            raise ValueError(msg)
        return result
    specs = arm_specs(config)
    names = list(specs)
    offset = order_index % len(names)
    arms: dict[str, Any] = {}
    for name in (*names[offset:], *names[:offset]):
        failed_path = root / name / "FAILED.json"
        arm_signature = pilot.digest([signature, specs[name]])
        if failed_path.exists():
            arms[name] = pilot.read_record(failed_path)
            if arms[name]["signature"] != arm_signature:
                msg = "Retained long-ensemble arm failure changed."
                raise ValueError(msg)
            arms[name]["physical_work"] = physical_training(root / name)
            continue
        arm: dict[str, Any] = {}
        try:
            arm["training"] = train_arm(root / name, circuit, initial, target, specs[name], config)
            arm["final_score"] = score_point(root, circuit, np.asarray(arm["training"]["theta"]), target)
            arm["status"] = "complete"
        except (ArithmeticError, RuntimeError, ValueError) as error:
            arm.update(status="failed", failure=f"{type(error).__name__}: {error}", signature=arm_signature)
            pilot.write_record(failed_path, arm)
        arm["physical_work"] = physical_training(root / name)
        arms[name] = arm
    comparison: dict[str, Any] = {"status": "incomplete", "common_cpu_seconds": None, "points": []}
    if all(arm["status"] == "complete" for arm in arms.values()):
        common, points = base.common_cpu_points({name: arm["training"] for name, arm in arms.items()})
        complete = True
        for point in points:
            point["scores"] = {}
            for name, value in point["selected"].items():
                try:
                    score = score_point(root, circuit, np.asarray(value["theta"]), target)
                except (ArithmeticError, RuntimeError, ValueError) as error:
                    score = {"status": "failed", "failure": f"{type(error).__name__}: {error}"}
                    complete = False
                point["scores"][name] = score
        comparison = {
            "status": "complete" if complete else "incomplete",
            "common_cpu_seconds": common,
            "points": points,
        }
    result = {
        "signature": signature,
        "target_id": source["target_id"],
        "replica": source["replica"],
        "protocol_checksum": protocol_checksum,
        "status": "complete" if comparison["status"] == "complete" else "complete_with_failed_arms",
        "arms": arms,
        "common_cpu_comparison": comparison,
        "post_training_scoring": physical_scoring(root),
        "cpu_scope": CPU_SCOPE,
        "artifact_hashes": {
            str(child.relative_to(root)): hashlib.sha256(child.read_bytes()).hexdigest()
            for child in sorted(root.rglob("*.json"))
            if child != path
        },
    }
    pilot.write_record(path, result)
    return result


def protocol_record() -> dict[str, Any]:
    """Return the complete scientific freeze from authenticated original starts.

    Raises:
        ValueError: If the full original 78+30 gate development panel is absent.
    """
    config = Config()
    starts_path = runner.OUTPUT / "starts.json"
    selected = [
        source
        for source in cast("list[dict[str, Any]]", pilot.read_record(starts_path)["starts"])
        if source["target_id"] in TARGETS and source["replica"] == 11
    ]
    if len(selected) != len(TARGETS) or {source["target_id"] for source in selected} != set(TARGETS):
        msg = "Long-ensemble science requires all four original development seed11 starts."
        raise ValueError(msg)
    for source in selected:
        circuit = pilot.restore_circuit(source["candidate"]["circuit"])
        counts = runner.counts(circuit)
        if (
            circuit.num_qubits != 6
            or counts["one_qubit"] != 78
            or counts["two_qubit"] != 30
            or sorted(gate.param_index for gate in circuit.gates) != list(range(108))
            or len(source["candidate"]["theta"]) != 108
        ):
            msg = "Long-ensemble science requires the original q6 78+30 gate scalar construction."
            raise ValueError(msg)
    return {
        "version": VERSION,
        "phase": "development",
        "targets": list(TARGETS),
        "replica": 11,
        "config": asdict(config),
        "arms": arm_specs(config),
        "starts": selected,
        "target_sha256": {
            name: hashlib.sha256(load_target(6, name).state_vector_copy().tobytes()).hexdigest() for name in TARGETS
        },
        "benchmark_hashes": runner.custody()["benchmark_hashes"],
        "source_hashes": numerical_sources(),
        "source_starts_sha256": hashlib.sha256(starts_path.read_bytes()).hexdigest(),
        "cpu_scope": CPU_SCOPE,
        "common_cpu_fractions": list(base.CPU_FRACTIONS),
        "endpoint_rule": "final accepted L-BFGS iterate or final Adam update; latest available by CPU, never quality",
        "promotion": "mean11 >=0.5pp all11 mean over best matched mean2 at common CPU, positive on all four targets",
        "resource_alternative": "all11 mean within0.1pp of best mean2 using <=0.8 its CPU on every target",
        "retirement": "otherwise retire this full-family 6050-call budget; no additional tuning in this diagnostic",
        "hypothesis": "the earlier mean11 deficit may reflect only100 joint evaluations rather than objective conflict",
        "randomness": "deterministic exact density; matched original construction replica11 only",
        "order": "rotate all four arms by frozen target index; no shared learned state",
        "construction_cost": "existing shared original construction; excluded from every refinement CPU curve",
    }


def freeze_protocol(output: Path) -> dict[str, Any]:
    """Write the immutable prepare-only protocol without training or scoring.

    Raises:
        ValueError: If an existing frozen protocol differs.
    """
    protocol = protocol_record()
    path = output / "PROTOCOL.json"
    if path.exists() and pilot.read_record(path) != protocol:
        msg = "Frozen long-ensemble protocol changed; use a new output root."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return protocol


def verify_protocol(protocol: dict[str, Any]) -> None:
    """Authenticate every frozen setting, source, target and starting parameter.

    Raises:
        ValueError: If any frozen protocol field differs from its original input.
    """
    if protocol != protocol_record():
        msg = "Long-ensemble protocol custody or frozen settings changed."
        raise ValueError(msg)


def execute_case(job: dict[str, Any]) -> dict[str, Any]:
    """Execute exactly one authenticated development construction.

    Raises:
        ValueError: If a requested case or execution order is outside the freeze.
    """
    protocol, source = job["protocol"], job["source"]
    verify_protocol(protocol)
    if source not in protocol["starts"] or job["order_index"] != protocol["targets"].index(source["target_id"]):
        msg = "Requested long-ensemble case is outside its frozen construction/order panel."
        raise ValueError(msg)
    result = run_case(
        Path(job["root"]),
        source,
        Config(**protocol["config"]),
        order_index=job["order_index"],
        protocol_checksum=pilot.digest(protocol),
    )
    verify_protocol(protocol)
    return result


def write_scoreboard(output: Path, records: list[dict[str, Any]]) -> None:
    """Write separate full-cap and common-CPU all12 cells, including failed pairs."""
    rows = []
    for record in records:
        comparison = record["common_cpu_comparison"]
        final_point = comparison["points"][-1] if comparison["points"] else None
        for method, arm in record["arms"].items():
            training = arm.get("training", {})
            for panel in ("full_cap", "common_cpu"):
                score = (
                    arm.get("final_score", {})
                    if panel == "full_cap"
                    else (final_point["scores"][method] if final_point is not None else {})
                )
                status = arm["status"] if panel == "full_cap" else score.get("status", "incomplete")
                rows.extend(
                    {
                        "target_id": record["target_id"],
                        "replica": record["replica"],
                        "panel": panel,
                        "method": method,
                        "noise_id": noise,
                        "status": status,
                        "failure": arm.get("failure") if panel == "full_cap" else score.get("failure"),
                        "fidelity": score.get("fidelities", {}).get(noise) if status == "complete" else None,
                        "comparison": "equal_individual_channel_cap"
                        if panel == "full_cap"
                        else "common_cpu_last_available",
                        "training_cpu_seconds": training.get("cpu_seconds"),
                        "common_cpu_seconds": comparison["common_cpu_seconds"] if panel == "common_cpu" else None,
                        "selected_cpu_seconds": final_point["selected"][method]["cpu_seconds"]
                        if panel == "common_cpu" and final_point is not None
                        else training.get("cpu_seconds"),
                        "joint_calls": training.get("joint_gradient_calls"),
                        "individual_channel_calls": training.get("individual_channel_gradient_calls"),
                        "physical_attempted_channel_calls": arm["physical_work"]["physical_attempted_channel_calls"],
                        "physical_completed_channel_calls": arm["physical_work"]["physical_completed_channel_calls"],
                        "physical_completed_direction_passes": arm["physical_work"][
                            "physical_completed_direction_passes"
                        ],
                        "known_training_cpu_seconds": arm["physical_work"]["known_training_cpu_seconds"],
                        "unknown_kill_cost": arm["physical_work"]["unknown_kill_cost"],
                        "density_direction_passes": training.get("density_direction_passes"),
                        "replayed_completed_channel_calls": training.get("replayed_completed_channel_calls"),
                        "invocation_cpu_seconds": training.get("invocation_cpu_seconds"),
                        "cpu_scope": CPU_SCOPE,
                        "case_checksum": pilot.digest(record),
                    }
                    for noise in channels.NOISE_IDS
                )
    with (output / "SCOREBOARD.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare first, then run only the four fixed arms with one to four workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four workers with single-thread BLAS limits.")
    if args.prepare_only:
        freeze_protocol(args.output)
        return 0
    protocol = cast("dict[str, Any]", pilot.read_record(args.output / "PROTOCOL.json"))
    verify_protocol(protocol)
    sources = {source["target_id"]: source for source in protocol["starts"]}
    jobs = [
        {
            "protocol": protocol,
            "source": sources[target],
            "order_index": index,
            "root": str(args.output / f"{target}__11"),
        }
        for index, target in enumerate(protocol["targets"])
    ]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(execute_case, jobs):
            records.append(result)
            write_scoreboard(args.output, records)
            print(result["target_id"], result["status"], flush=True)
    pilot.write_record(
        args.output / "RESULT.json",
        {
            "status": "complete"
            if all(result["status"] == "complete" for result in records)
            else "complete_with_failed_arms",
            "protocol_checksum": pilot.digest(protocol),
            "results": [pilot.digest(result) for result in records],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
