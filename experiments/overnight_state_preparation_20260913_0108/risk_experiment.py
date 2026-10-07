# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Frozen six-arm entropic-risk comparison with exact resume and complete costs."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import operator
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from benchmarks.state_preparation.ballarin import materialize_ballarin_circuit
from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_exact_adam as adam
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import (
    channels,
    ensemble_experiment,
    ensemble_structural_experiment,
    optimizers,
    risk_objective,
    runner,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

VERSION = "overnight.entropic_risk.v1"
TARGETS = ensemble_experiment.TARGETS
METHODS = ("risk020_lbfgs", "risk100_lbfgs", "risk020_adam", "risk100_adam", "mean2_lbfgs", "mean2_adam")


@dataclass(frozen=True)
class Config:
    """Declare equal two-channel budgets and unchanged optimizer settings."""

    channel_evaluations: int = 1100
    history: int = 10
    checkpoint_every: int = 10
    adam_learning_rate: float = 0.02

    def __post_init__(self) -> None:
        """Reject nonintegral caps and invalid optimizer constants.

        Raises:
            ValueError: If the budget cannot comprise complete paired channel calls.
        """
        if (
            any(
                type(value) is not int or value < 1
                for value in (self.channel_evaluations, self.history, self.checkpoint_every)
            )
            or self.channel_evaluations % 2
            or isinstance(self.adam_learning_rate, bool)
            or not np.isfinite(self.adam_learning_rate)
            or self.adam_learning_rate <= 0
        ):
            msg = "Risk screen requires positive even channel cap and valid optimizer settings."
            raise ValueError(msg)


def specification(method: str) -> dict[str, Any]:
    """Return one declared optimizer/temperature identity.

    Raises:
        ValueError: If a caller requests an undeclared method or temperature.
    """
    if method not in METHODS:
        msg = "Unsupported frozen risk method."
        raise ValueError(msg)
    return {
        "optimizer": method.rsplit("_", 1)[1],
        "temperature": 0.02 if method.startswith("risk020") else 0.1 if method.startswith("risk100") else None,
        "noise_ids": list(risk_objective.TRAINING_NOISE_IDS),
    }


def attempt_totals(attempts: dict[str, Any]) -> dict[str, Any]:
    """Return attempted/completed work, including failed components and replay."""
    known = all(row["status"] != "running" for row in attempts["attempts"])
    rows = [row for row in attempts["attempts"] if row["status"] != "running"]
    result: dict[str, Any] = {
        "accounting_status": "known" if known else "unknown_requires_external_ledger",
        **{
            key: sum(row[key] for row in rows)
            for key in (
                "attempted_joint_calls",
                "completed_joint_calls",
                "attempted_channel_calls",
                "completed_channel_calls",
                "cpu_seconds",
                "wall_seconds",
            )
        },
    }
    result["completed_density_direction_passes"] = 2 * result["completed_channel_calls"]
    return result


def train(
    root: Path,
    circuit: ParameterizedCircuit,
    initial: NDArray[np.float64],
    target: NDArray[np.complex128],
    method: str,
    config: Config,
) -> dict[str, Any]:
    """Train or resume exactly, preserving full optimizer memory and raw attempts."""
    spec = specification(method)
    signature = pilot.digest({
        "version": VERSION,
        "specification": spec,
        "circuit": pilot.circuit_record(circuit),
        "initial": initial.tolist(),
        "target": hashlib.sha256(target.tobytes()).hexdigest(),
        "config": asdict(config),
    })
    root.mkdir(parents=True, exist_ok=True)
    final_path, partial_path, attempts_path = (
        root / name for name in ("TRAINING.json", "PARTIAL.json", "ATTEMPTS.json")
    )
    attempts: dict[str, Any] = (
        cast("dict[str, Any]", pilot.read_record(attempts_path))
        if attempts_path.exists()
        else {"signature": signature, "attempts": []}
    )
    if attempts["signature"] != signature or (attempts["attempts"] and attempts["attempts"][-1]["status"] == "running"):
        msg = "Risk identity changed or killed work requires an external resource ledger."
        raise ValueError(msg)
    if final_path.exists():
        final = cast("dict[str, Any]", pilot.read_record(final_path))
        if final["signature"] != signature or final["attempts_checksum"] != pilot.digest(attempts):
            msg = "Completed risk training or physical attempt ledger changed."
            raise ValueError(msg)
        return final
    prior = cast("dict[str, Any]", pilot.read_record(partial_path)) if partial_path.exists() else None
    if prior is not None and prior["signature"] != signature:
        msg = "Risk checkpoint objective or original initialization changed."
        raise ValueError(msg)
    prior_cost = attempt_totals(attempts)
    curve = (
        copy.deepcopy(prior["curve"])
        if prior is not None
        else [{"theta": initial.tolist(), "cpu_seconds": 0.0, "joint_evaluations": 0, "individual_channel_calls": 0}]
    )
    last_oracle = copy.deepcopy(prior["last_oracle"]) if prior is not None else None
    entry: dict[str, Any] = {
        "status": "running",
        "attempted_joint_calls": 0,
        "completed_joint_calls": 0,
        "attempted_channel_calls": 0,
        "completed_channel_calls": 0,
    }
    attempts["attempts"].append(entry)
    pilot.write_record(attempts_path, attempts)
    cpu, wall = time.process_time(), time.perf_counter()

    def progress(_noise: str, phase: str) -> None:
        entry["attempted_channel_calls" if phase == "start" else "completed_channel_calls"] += 1

    def objective(theta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        nonlocal last_oracle
        entry["attempted_joint_calls"] += 1
        result = risk_objective.loss_gradient(
            circuit, theta, target, temperature=spec["temperature"], progress=progress
        )
        entry["completed_joint_calls"] += 1
        last_oracle = {
            "theta": theta.tolist(),
            "loss": result.loss,
            "weights": result.weights.tolist(),
            "component_losses": result.component_losses.tolist(),
        }
        return result.loss, result.gradient

    def point(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "theta": state["theta"],
            "cpu_seconds": prior_cost["cpu_seconds"] + time.process_time() - cpu,
            "joint_evaluations": state["evaluations"],
            "individual_channel_calls": 2 * state["evaluations"],
        }

    def save(state: dict[str, Any]) -> None:
        curve.append(point(state))
        if state["iterations"] % config.checkpoint_every == 0:
            pilot.write_record(
                partial_path, {"signature": signature, "optimizer": state, "curve": curve, "last_oracle": last_oracle}
            )

    try:
        if spec["optimizer"] == "lbfgs":
            state = optimizers.minimize(
                objective,
                initial,
                max_evaluations=config.channel_evaluations // 2,
                max_iterations=config.channel_evaluations // 2,
                history=config.history,
                state=None if prior is None else prior["optimizer"],
                callback=save,
            )
        else:
            state = ensemble_experiment.minimize_adam(
                objective,
                initial,
                adam.AdamConfig(updates=config.channel_evaluations // 2, learning_rate=config.adam_learning_rate),
                state=None if prior is None else prior["optimizer"],
                callback=save,
            )
        entry["status"] = "complete"
    except BaseException as error:
        entry.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            failure=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        entry.update(cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - wall)
        pilot.write_record(attempts_path, attempts)
    physical = attempt_totals(attempts)
    curve.append({**point(state), "cpu_seconds": physical["cpu_seconds"]})
    result: dict[str, Any] = {
        "signature": signature,
        "status": "complete",
        "method": method,
        "specification": spec,
        "optimizer": state,
        "theta": state["theta"],
        "curve": curve,
        "last_oracle": last_oracle,
        "joint_gradient_calls": state["evaluations"],
        "individual_channel_calls": 2 * state["evaluations"],
        "density_direction_passes": 4 * state["evaluations"],
        "physical_work": physical,
        "cpu_seconds": physical["cpu_seconds"],
        "wall_seconds": physical["wall_seconds"],
        "attempts_checksum": pilot.digest(attempts),
        "endpoint": "final accepted L-BFGS iterate or final Adam update; last oracle may be a rejected trial",
    }
    pilot.write_record(final_path, result)
    return result


def metrics(fidelities: dict[str, float]) -> dict[str, Any]:
    """Return all-eleven mean and predeclared tail metrics, keeping training scope distinct.

    Raises:
        ValueError: If the complete calibrated endpoint is missing or nonfinite.
    """
    if set(fidelities) != set(channels.NOISE_IDS) or any(
        not np.isfinite(value) or not -1e-10 <= value <= 1 + 1e-10 for value in fidelities.values()
    ):
        msg = "Risk endpoint metrics require all twelve finite exact fidelities."
        raise ValueError(msg)
    noisy = [fidelities[noise] for noise in channels.NOISE_IDS if noise != "noiseless"]
    families = {
        "N1_angle": fidelities["ballarin_coupled"],
        "N2_dephasing": float(
            np.mean([value for noise, value in fidelities.items() if noise.startswith("dephasing_")])
        ),
        "N3_depolarizing": float(
            np.mean([value for noise, value in fidelities.items() if noise.startswith("depolarizing_")])
        ),
    }
    return {
        "mean11": float(np.mean(noisy)),
        "worst11": min(noisy),
        "lowest3_mean": float(np.mean(sorted(noisy)[:3])),
        "training_mean2": float(np.mean([fidelities[noise] for noise in risk_objective.TRAINING_NOISE_IDS])),
        "noiseless": fidelities["noiseless"],
        "noise_families": families,
        "worst_family": min(families, key=lambda name: families[name]),
        "worst_family_mean": min(families.values()),
    }


def run_case(
    root: Path,
    circuit: ParameterizedCircuit,
    initial: NDArray[np.float64],
    target: NDArray[np.complex128],
    config: Config,
    *,
    target_id: str = "engineering",
    replica: int = 11,
    protocol_checksum: str | None = None,
) -> dict[str, Any]:
    """Run all six endpoints and CPU-only selected points with separate exact scores."""
    signature = pilot.digest({
        "version": VERSION,
        "circuit": pilot.circuit_record(circuit),
        "theta": initial.tolist(),
        "target": hashlib.sha256(target.tobytes()).hexdigest(),
        "config": asdict(config),
        "target_id": target_id,
        "replica": replica,
        "protocol_checksum": protocol_checksum,
    })
    root.mkdir(parents=True, exist_ok=True)
    path, attempt_path = root / "RESULT.json", root / "CASE_ATTEMPTS.json"
    attempts: dict[str, Any] = (
        cast("dict[str, Any]", pilot.read_record(attempt_path))
        if attempt_path.exists()
        else {"signature": signature, "attempts": []}
    )
    if attempts["signature"] != signature or (attempts["attempts"] and attempts["attempts"][-1]["status"] == "running"):
        msg = "Risk case identity changed or killed work requires an external resource ledger."
        raise ValueError(msg)
    if path.exists():
        cached = cast("dict[str, Any]", pilot.read_record(path))
        if cached["signature"] != signature or cached["case_attempts_checksum"] != pilot.digest(attempts):
            msg = "Completed risk case identity or invocation ledger changed."
            raise ValueError(msg)
        return cached
    entry: dict[str, Any] = {"status": "running"}
    attempts["attempts"].append(entry)
    pilot.write_record(attempt_path, attempts)
    cpu, wall = time.process_time(), time.perf_counter()
    try:
        result = _case_endpoints(root, circuit, initial, target, config)
        entry["status"] = result["status"]
    except BaseException as error:
        entry.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            failure=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        entry.update(cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - wall)
        pilot.write_record(attempt_path, attempts)
    result.update(
        signature=signature,
        target_id=target_id,
        replica=replica,
        protocol_checksum=protocol_checksum,
        case_attempts_checksum=pilot.digest(attempts),
        case_invocation_cpu_seconds=sum(row["cpu_seconds"] for row in attempts["attempts"]),
        case_invocation_wall_seconds=sum(row["wall_seconds"] for row in attempts["attempts"]),
    )
    pilot.write_record(path, result)
    return result


def _case_endpoints(
    root: Path,
    circuit: ParameterizedCircuit,
    initial: NDArray[np.float64],
    target: NDArray[np.complex128],
    config: Config,
) -> dict[str, Any]:
    """Retain every arm failure and independently score complete final/CPU endpoints."""
    store = ensemble_structural_experiment.Store(root, target, ensemble_structural_experiment.Config())
    circuit_hash = store.circuit(circuit)
    arms: dict[str, Any] = {}
    for method in METHODS:
        arm: dict[str, Any] = {}
        try:
            arm = train(root / "training" / method, circuit, initial, target, method, config)
            score = store.operation(f"final_{method}", circuit, np.asarray(arm["theta"]), profiles=channels.NOISE_IDS)
            arm.update(
                fidelities=score["value"]["fidelities"],
                metrics=metrics(score["value"]["fidelities"]),
                evaluation=score,
                ballarin_counts=runner.counts(materialize_ballarin_circuit(circuit, np.asarray(arm["theta"])).circuit),
            )
        except (ArithmeticError, RuntimeError, ValueError) as error:
            arm.update(status="failed", failure=f"{type(error).__name__}: {error}")
            ledger = root / "training" / method / "ATTEMPTS.json"
            if ledger.exists():
                arm["physical_work"] = attempt_totals(cast("dict[str, Any]", pilot.read_record(ledger)))
        arms[method] = arm
    common: dict[str, Any] | None = None
    if all(arm["status"] == "complete" for arm in arms.values()):
        cutoff = min(arm["cpu_seconds"] for arm in arms.values())
        points: dict[str, Any] = {}
        for method, arm in arms.items():
            selected = [point for point in arm["curve"] if point["cpu_seconds"] <= cutoff][-1]
            try:
                score = store.operation(
                    f"common_{method}", circuit, np.asarray(selected["theta"]), profiles=channels.NOISE_IDS
                )
                points[method] = {
                    "status": "complete",
                    "point": selected,
                    "fidelities": score["value"]["fidelities"],
                    "metrics": metrics(score["value"]["fidelities"]),
                    "evaluation": score,
                }
            except (ArithmeticError, RuntimeError, ValueError) as error:
                points[method] = {"status": "failed", "point": selected, "failure": f"{type(error).__name__}: {error}"}
        common = {"cpu_seconds": cutoff, "methods": points, "selection": "last available iterate by CPU only"}
    evaluation_attempts = {
        str(path.relative_to(root)): cast("dict[str, Any]", pilot.read_record(path))
        for path in sorted((root / "operations").glob("*.attempts.json"))
    }
    complete = common is not None and all(point["status"] == "complete" for point in common["methods"].values())
    known_training = all(arm.get("physical_work", {}).get("accounting_status") == "known" for arm in arms.values())
    measured_training = sum(arm.get("physical_work", {}).get("cpu_seconds", 0) for arm in arms.values())
    known_evaluation = all(
        attempt["status"] != "running" for ledger in evaluation_attempts.values() for attempt in ledger["attempts"]
    )
    measured_evaluation = sum(
        attempt["cpu_seconds"]
        for ledger in evaluation_attempts.values()
        for attempt in ledger["attempts"]
        if attempt["status"] != "running"
    )
    return {
        "status": "complete" if complete else "complete_with_failures",
        "circuit_hash": circuit_hash,
        "logical_counts": runner.counts(circuit),
        "parameters": circuit.num_params,
        "arms": arms,
        "common_cpu": common,
        "evaluation_attempts": evaluation_attempts,
        "physical_training_cpu_seconds": measured_training if known_training else None,
        "known_training_cpu_seconds": measured_training,
        "physical_evaluation_cpu_seconds": measured_evaluation if known_evaluation else None,
        "known_evaluation_cpu_seconds": measured_evaluation,
        "physical_accounting_status": "known"
        if known_training and known_evaluation
        else "unknown_requires_external_ledger",
    }


def numerical_sources() -> dict[str, str]:
    """Freeze all actual numerical and cache dependencies without editing them."""
    sources = ensemble_structural_experiment.numerical_sources()
    for name in ("risk_objective.py", "risk_experiment.py", "RISK_PROTOCOL.md"):
        path = Path(__file__).with_name(name)
        sources[str(path.relative_to(runner.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def freeze_protocol(output: Path) -> dict[str, Any]:
    """Prepare only the fixed original four-target panel and six declared arms."""
    source = runner.OUTPUT / "starts.json"
    starts = cast("list[dict[str, Any]]", pilot.read_record(source)["starts"])
    selected = [row for row in starts if row["target_id"] in TARGETS and row["replica"] == 11]
    if len(selected) != 4 or {row["target_id"] for row in selected} != set(TARGETS):
        msg = "Risk screen requires all four original development starts."
        raise ValueError(msg)
    for row in selected:
        circuit = pilot.restore_circuit(row["candidate"]["circuit"])
        if circuit.num_qubits != 6 or len(circuit.gates) != 108:
            msg = "Risk screen requires original 108-gate q6 circuits."
            raise ValueError(msg)
    protocol: dict[str, Any] = {
        "version": VERSION,
        "phase": "development",
        "targets": list(TARGETS),
        "replica": 11,
        "methods": {method: specification(method) for method in METHODS},
        "config": asdict(Config()),
        "starts": selected,
        "source_starts_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_hashes": numerical_sources(),
        "benchmark_hashes": runner.custody()["benchmark_hashes"],
        "endpoint": "final iterate",
        "primary": "equal mean of all eleven noisy conditions, equally across targets",
        "tail": "minimum all11 and mean of three lowest all11 per target",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "PROTOCOL.json"
    if path.exists() and pilot.read_record(path) != protocol:
        msg = "Frozen risk protocol differs."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return protocol


def verify_protocol(protocol: dict[str, Any]) -> None:
    """Reject changes to starts, settings, components, sources or benchmark custody."""
    source = runner.OUTPUT / "starts.json"
    starts = cast("list[dict[str, Any]]", pilot.read_record(source)["starts"])
    expected = [row for row in starts if row["target_id"] in TARGETS and row["replica"] == 11]
    if (
        protocol["version"] != VERSION
        or protocol["targets"] != list(TARGETS)
        or protocol["replica"] != 11
        or protocol["methods"] != {method: specification(method) for method in METHODS}
        or protocol["config"] != asdict(Config())
        or protocol["starts"] != expected
        or protocol["source_starts_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest()
        or protocol["source_hashes"] != numerical_sources()
        or protocol["benchmark_hashes"] != runner.custody()["benchmark_hashes"]
    ):
        msg = "Risk protocol source, benchmark, settings or starts custody mismatch."
        raise ValueError(msg)


def execute_case(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one authenticated case only after coordinator allocation."""
    protocol, source = job["protocol"], job["source"]
    verify_protocol(protocol)
    result = run_case(
        Path(job["root"]),
        pilot.restore_circuit(source["candidate"]["circuit"]),
        np.asarray(source["candidate"]["theta"]),
        load_target(6, source["target_id"]).state_vector_copy(),
        Config(**protocol["config"]),
        target_id=source["target_id"],
        replica=source["replica"],
        protocol_checksum=pilot.digest(protocol),
    )
    verify_protocol(protocol)
    return result


def write_scoreboard(output: Path, records: list[dict[str, Any]]) -> None:
    """Write stable final-endpoint rows with all declared methods and failures."""
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=operator.itemgetter("target_id", "replica")):
        for method in METHODS:
            arm = record["arms"][method]
            for noise in channels.NOISE_IDS:
                row: dict[str, Any] = {
                    "target_id": record["target_id"],
                    "replica": record["replica"],
                    "method": method,
                    "noise_id": noise,
                    "status": arm["status"],
                    "failure": arm.get("failure"),
                    "convention": "calibrated_exact",
                    "exactness": "exact_calibrated_channel",
                    "case_checksum": pilot.digest(record),
                }
                if arm["status"] == "complete":
                    row.update(
                        fidelity=arm["fidelities"][noise],
                        temperature=arm["specification"]["temperature"],
                        individual_channel_calls=arm["individual_channel_calls"],
                        direction_passes=arm["density_direction_passes"],
                        training_cpu_seconds=arm["cpu_seconds"],
                        training_wall_seconds=arm["wall_seconds"],
                        physical_attempted_channel_calls=arm["physical_work"]["attempted_channel_calls"],
                        physical_completed_channel_calls=arm["physical_work"]["completed_channel_calls"],
                        evaluation_cpu_seconds=arm["evaluation"]["cpu_seconds"],
                        logical_one_qubit=record["logical_counts"]["one_qubit"],
                        logical_two_qubit=record["logical_counts"]["two_qubit"],
                        logical_depth=record["logical_counts"]["depth"],
                        ballarin_one_qubit=arm["ballarin_counts"]["one_qubit"],
                        ballarin_two_qubit=arm["ballarin_counts"]["two_qubit"],
                        ballarin_depth=arm["ballarin_counts"]["depth"],
                        parameters=record["parameters"],
                    )
                rows.append(row)
    with (output / "SCOREBOARD.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare only or execute an immutable six-arm protocol with bounded workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four single-thread numerical workers.")
    if args.prepare_only:
        freeze_protocol(args.output)
        return 0
    protocol = cast("dict[str, Any]", pilot.read_record(args.output / "PROTOCOL.json"))
    verify_protocol(protocol)
    jobs = [
        {"protocol": protocol, "source": source, "root": str(args.output / f"{source['target_id']}__11")}
        for source in protocol["starts"]
    ]
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(execute_case, jobs):
            records.append(record)
            write_scoreboard(args.output, records)
            print(record["target_id"], record["status"], flush=True)
    pilot.write_record(
        args.output / "RESULT.json",
        {
            "status": "complete"
            if all(record["status"] == "complete" for record in records)
            else "complete_with_failures",
            "protocol_checksum": pilot.digest(protocol),
            "results": [pilot.digest(record) for record in records],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
