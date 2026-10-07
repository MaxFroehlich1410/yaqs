# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Allocated development driver for the frozen four-arm pure-geometry protocol.

All arms use one outer process-CPU timing scope including intermediate
checkpoints. Temporary process-local metering calls the unchanged geometry
kernels and restores every callable, even on interruption. Target workers run
arms serially; concurrent threads within a worker are unsupported.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import numpy as np

from benchmarks.state_preparation.ballarin import materialize_ballarin_circuit
from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_exact_adam as adam
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import channels, geometry, optimizers, runner

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

VERSION = "overnight.pure_geometry_comparison.v1"
TARGETS = ("gaussian_mu0p5_sigma0p1", "tfim_critical", "haar_random_1", "random_mps_bond2")
ARMS = ("geometry_0p001", "geometry_0p01", "local_lbfgs", "local_adam")
CPU_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Config:
    """Scientific defaults; smaller engineering budgets do not prepare a panel."""

    evaluations: int = 300
    checkpoint_every: int = 10

    def __post_init__(self) -> None:
        """Reject invalid caps or checkpoint intervals without adding tunable settings."""
        if (
            type(self.evaluations) is not int
            or not 1 <= self.evaluations <= 300
            or type(self.checkpoint_every) is not int
            or self.checkpoint_every < 1
        ):
            msg = "Geometry experiment needs a 1-300 call cap and positive checkpoint interval."
            raise ValueError(msg)


def numerical_sources() -> dict[str, str]:
    """Fingerprint the unchanged numerical dependencies and frozen geometry protocol."""
    result = runner.source_hashes()
    for name in ("geometry.py", "geometry_experiment.py", "GEOMETRY_PROTOCOL.md"):
        path = Path(__file__).with_name(name)
        result[str(path.relative_to(runner.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class Meter:
    """Measure physical attempted/completed component work, including replayed calls."""

    def __init__(self) -> None:
        """Retain original callables before installing any process-local wrapper."""
        self.raw_channel = geometry.density_loss_gradient
        self.raw_metric = geometry.pure_state_metric
        self.raw_solve = geometry.precondition
        self.values: dict[str, Any] = {
            f"{kind}_{field}": 0.0 if field == "cpu_seconds" else 0
            for kind in ("channel", "metric", "solve")
            for field in ("attempted", "completed", "cpu_seconds")
        }
        self.values.update(statevector_row_operator_applications=0, linear_solves=0, symmetric_eigensolves=0)

    def channel(
        self,
        circuit: ParameterizedCircuit,
        theta: NDArray,
        target: NDArray,
        calibration: CalibratedLocalDepolarizingProvider,
    ) -> tuple[NDArray, float, float]:
        """Call the original true noisy gradient and account for success or failure."""
        cpu = time.process_time()
        self.values["channel_attempted"] += 1
        try:
            result = self.raw_channel(circuit, theta, target, calibration)
            self.values["channel_completed"] += 1
            return result
        finally:
            self.values["channel_cpu_seconds"] += time.process_time() - cpu

    def metric(self, circuit: ParameterizedCircuit, theta: NDArray) -> tuple[NDArray, dict[str, Any]]:
        """Call the unchanged ideal metric and retain its exact row-operation count."""
        cpu = time.process_time()
        self.values["metric_attempted"] += 1
        try:
            result, metadata = self.raw_metric(circuit, theta)
            self.values["metric_completed"] += 1
            self.values["statevector_row_operator_applications"] += metadata["statevector_row_operator_applications"]
            return result, metadata
        finally:
            self.values["metric_cpu_seconds"] += time.process_time() - cpu

    def solve(self, gradient: NDArray, metric: NDArray, damping: float) -> tuple[NDArray, dict[str, Any]]:
        """Call the original damped solve, including its spectral validation cost."""
        cpu = time.process_time()
        self.values["solve_attempted"] += 1
        try:
            result, metadata = self.raw_solve(gradient, metric, damping)
            self.values["solve_completed"] += 1
            self.values["linear_solves"] += metadata["linear_solves"]
            self.values["symmetric_eigensolves"] += metadata["symmetric_eigensolves"]
            return result, metadata
        finally:
            self.values["solve_cpu_seconds"] += time.process_time() - cpu


def adam_chunk(
    objective: Callable[[NDArray], tuple[float, NDArray]],
    initial: NDArray,
    cap: int,
    until: int,
    state: dict[str, Any] | None,
    callback: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Advance standard Adam to a durable call boundary without resetting moments."""
    config = adam.AdamConfig(updates=cap, learning_rate=0.02)
    current: dict[str, Any] = (
        copy.deepcopy(state)
        if state is not None
        else {
            "version": "overnight.geometry_adam.v1",
            "config": asdict(config),
            "initial": initial.tolist(),
            "theta": initial.tolist(),
            "first_moment": np.zeros_like(initial).tolist(),
            "second_moment": np.zeros_like(initial).tolist(),
            "iterations": 0,
            "evaluations": 0,
            "status": "running",
            "last_pre_update_loss": None,
        }
    )
    if (
        current["version"] != "overnight.geometry_adam.v1"
        or current["config"] != asdict(config)
        or not np.array_equal(np.asarray(current["initial"]), initial)
        or type(current["evaluations"]) is not int
        or current["iterations"] != current["evaluations"]
        or not 0 <= current["evaluations"] <= until <= cap
    ):
        msg = "Adam chunk checkpoint or cumulative budget is incompatible."
        raise ValueError(msg)
    theta, first, second = (np.asarray(current[key], dtype=float) for key in ("theta", "first_moment", "second_moment"))
    for step in range(current["evaluations"] + 1, until + 1):
        loss, gradient = objective(theta.copy())
        if not np.isfinite(loss):
            msg = "Adam true channel objective is nonfinite."
            raise FloatingPointError(msg)
        theta, first, second = adam.adam_step(theta, gradient, first, second, step, config)
        current.update(
            theta=theta.tolist(),
            first_moment=first.tolist(),
            second_moment=second.tolist(),
            iterations=step,
            evaluations=step,
            last_pre_update_loss=loss,
            status="running",
        )
        callback(copy.deepcopy(current))
    current["status"] = "evaluation_limit"
    return current


def train_arm(
    root: Path,
    circuit: ParameterizedCircuit,
    initial: NDArray,
    target: NDArray,
    method: str,
    config: Config,
) -> dict[str, Any]:
    """Train from original parameters with shared CPU scope and complete attempt accounting.

    Chunk caps preserve pending L-BFGS/geometric searches and checkpoint every
    ten completed budgeted calls, rather than waiting for an accepted step.
    All parameter selection uses accepted states, never rejected trial points.
    """
    if method not in ARMS:
        msg = "Unknown geometry experiment arm."
        raise ValueError(msg)
    signature = pilot.digest({
        "version": VERSION,
        "method": method,
        "config": asdict(config),
        "circuit": pilot.circuit_record(circuit),
        "initial": initial.tolist(),
        "target": hashlib.sha256(target.tobytes()).hexdigest(),
    })
    root.mkdir(parents=True, exist_ok=True)
    final_path, partial_path, attempts_path = (
        root / name for name in ("TRAINING.json", "PARTIAL.json", "ATTEMPTS.json")
    )
    if final_path.exists():
        final = cast("dict[str, Any]", pilot.read_record(final_path))
        if final["signature"] != signature:
            msg = "Completed geometry training identity changed."
            raise ValueError(msg)
        return final
    prior = cast("dict[str, Any]", pilot.read_record(partial_path)) if partial_path.exists() else None
    attempts: dict[str, Any] = (
        cast("dict[str, Any]", pilot.read_record(attempts_path))
        if attempts_path.exists()
        else {
            "signature": signature,
            "attempts": [],
        }
    )
    if (prior is not None and prior["signature"] != signature) or attempts["signature"] != signature:
        msg = "Geometry partial checkpoint or attempt-ledger identity changed."
        raise ValueError(msg)
    current: dict[str, Any] | None = copy.deepcopy(prior["optimizer"]) if prior is not None else None
    curve: list[dict[str, Any]] = (
        copy.deepcopy(prior["curve"])
        if prior is not None
        else [
            {
                "theta": initial.tolist(),
                "evaluations": 0,
                "iterations": 0,
                "cpu_seconds": 0.0,
                "local_fidelity": None,
            }
        ]
    )
    prior_cpu = sum(row["cpu_seconds"] for row in attempts["attempts"])
    meter = Meter()
    cpu, wall = time.process_time(), time.perf_counter()

    def objective(theta: NDArray) -> tuple[float, NDArray]:
        try:
            gradient, loss, _ = meter.channel(circuit, theta, target, CalibratedLocalDepolarizingProvider())
        except FloatingPointError:
            return float("inf"), np.full_like(theta, np.nan)
        return loss, gradient

    def available(state: dict[str, Any]) -> None:
        curve.append({
            "theta": state["theta"],
            "evaluations": state["evaluations"],
            "iterations": state["iterations"],
            "cpu_seconds": prior_cpu + time.process_time() - cpu,
            "local_fidelity": None if method == "local_adam" else 1 - state["loss"],
        })

    def advance(until: int, state: dict[str, Any] | None) -> dict[str, Any]:
        if method.startswith("geometry"):
            return geometry.minimize(
                circuit,
                initial,
                target,
                damping=0.001 if method == "geometry_0p001" else 0.01,
                max_evaluations=until,
                state=state,
                callback=available,
            )
        if method == "local_lbfgs":
            return optimizers.minimize(
                objective,
                initial,
                max_evaluations=until,
                max_iterations=config.evaluations,
                history=10,
                state=state,
                callback=available,
            )
        return adam_chunk(objective, initial, config.evaluations, until, state, available)

    def optimize_chunks() -> dict[str, Any]:
        nonlocal current
        with ExitStack() as stack:
            if method.startswith("geometry"):
                stack.enter_context(patch.object(geometry, "density_loss_gradient", meter.channel))
                stack.enter_context(patch.object(geometry, "pure_state_metric", meter.metric))
                stack.enter_context(patch.object(geometry, "precondition", meter.solve))
            while current is None or current["evaluations"] < config.evaluations:
                if current is not None and current["status"] in {"converged", "line_search_failed"}:
                    break
                until = (
                    1
                    if current is None
                    else min(
                        config.evaluations,
                        (current["evaluations"] // config.checkpoint_every + 1) * config.checkpoint_every,
                    )
                )
                current = advance(until, current)
                available(current)
                pilot.write_record(partial_path, {"signature": signature, "optimizer": current, "curve": curve})
        assert current is not None
        return current

    completed, failure = False, None
    try:
        current = optimize_chunks()
        completed = True
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        attempts["attempts"].append({
            **meter.values,
            "cpu_seconds": time.process_time() - cpu,
            "wall_seconds": time.perf_counter() - wall,
            "completed": completed,
            "failure": failure,
        })
        pilot.write_record(attempts_path, attempts)
    assert current is not None
    physical: dict[str, Any] = {key: sum(row[key] for row in attempts["attempts"]) for key in meter.values}
    outer_cpu = sum(row["cpu_seconds"] for row in attempts["attempts"])
    curve.append({
        "theta": current["theta"],
        "evaluations": current["evaluations"],
        "iterations": current["iterations"],
        "cpu_seconds": outer_cpu,
        "local_fidelity": None if method == "local_adam" else 1 - current["loss"],
    })
    metric_calls = current.get("metric_evaluations", 0)
    result: dict[str, Any] = {
        "signature": signature,
        "status": "complete",
        "method": method,
        "optimizer": current,
        "theta": current["theta"],
        "curve": curve,
        "cpu_seconds": outer_cpu,
        "wall_seconds": sum(row["wall_seconds"] for row in attempts["attempts"]),
        "true_channel_gradient_calls": current["evaluations"],
        "budgeted_density_direction_passes": 2 * current["evaluations"],
        "metric_evaluations": metric_calls,
        "logical_metric_row_applications": current.get("statevector_row_operator_applications", 0),
        "physical": physical,
        "physical_completed_density_direction_passes": 2 * physical["channel_completed"],
        "replayed_attempted_gradient_calls": physical["channel_attempted"] - current["evaluations"],
        "replayed_completed_metric_calls": physical["metric_completed"] - metric_calls,
        "attempts_checksum": pilot.digest(attempts),
        "training_trajectories": 0,
        "cpu_scope": (
            "same outer optimization timer for every arm, including metrics, solves and intermediate checkpoints"
        ),
    }
    pilot.write_record(final_path, result)
    return result


def score_point(root: Path, circuit: ParameterizedCircuit, theta: NDArray, target: NDArray) -> dict[str, Any]:
    """Score all twelve profiles with durable per-profile checkpoints and failure costs."""
    signature = pilot.digest({
        "circuit": pilot.circuit_record(circuit),
        "theta": theta.tolist(),
        "target": hashlib.sha256(target.tobytes()).hexdigest(),
    })
    path = root / "scores" / f"{signature}.json"
    result: dict[str, Any] = (
        cast("dict[str, Any]", pilot.read_record(path))
        if path.exists()
        else {
            "signature": signature,
            "theta": theta.tolist(),
            "fidelities": {},
            "attempts": [],
            "status": "running",
        }
    )
    if result["signature"] != signature:
        msg = "Geometry score-cache identity changed."
        raise ValueError(msg)
    if result["status"] == "complete":
        return result
    for noise_id in channels.NOISE_IDS:
        if noise_id in result["fidelities"]:
            continue
        cpu, finished, failure = time.process_time(), False, None
        try:
            result["fidelities"][noise_id] = channels.evaluate(circuit, theta, target, noise_id)
            finished = True
        except BaseException as error:
            failure = f"{type(error).__name__}: {error}"
            raise
        finally:
            result["attempts"].append({
                "noise_id": noise_id,
                "completed": finished,
                "failure": failure,
                "cpu_seconds": time.process_time() - cpu,
            })
            pilot.write_record(path, result)
    result.update({
        "status": "complete",
        "mean_noisy_fidelity": float(np.mean([result["fidelities"][key] for key in channels.NOISE_IDS[1:]])),
        "forward_channel_evaluations": len(result["fidelities"]),
        "physical_attempted_forward_calls": len(result["attempts"]),
        "physical_completed_forward_calls": sum(row["completed"] for row in result["attempts"]),
        "cpu_seconds": sum(row["cpu_seconds"] for row in result["attempts"]),
        "logical_counts": runner.counts(circuit),
        "ballarin_counts": runner.counts(materialize_ballarin_circuit(circuit, theta).circuit),
    })
    pilot.write_record(path, result)
    return result


def common_cpu_points(training: dict[str, dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    """Select chronological last available accepted parameters before viewing scores."""
    common = min(value["cpu_seconds"] for value in training.values())
    points = [
        {
            "fraction": fraction,
            "cpu_budget_seconds": fraction * common,
            "selected": {
                name: next(point for point in reversed(value["curve"]) if point["cpu_seconds"] <= fraction * common)
                for name, value in training.items()
            },
        }
        for fraction in CPU_FRACTIONS
    ]
    return common, points


def failed_training_cost(root: Path) -> dict[str, Any]:
    """Expose unfinished physical work in the failed arm's scientific record."""
    path = root / "ATTEMPTS.json"
    if not path.exists():
        return {"cpu_seconds": 0.0, "physical": {}, "attempts_checksum": None}
    attempts = cast("dict[str, Any]", pilot.read_record(path))
    keys = Meter().values
    return {
        "cpu_seconds": sum(row["cpu_seconds"] for row in attempts["attempts"]),
        "physical": {key: sum(row[key] for row in attempts["attempts"]) for key in keys},
        "attempts_checksum": pilot.digest(attempts),
    }


def run_case(root: Path, source: dict[str, Any], config: Config, *, order_index: int = 0) -> dict[str, Any]:
    """Run the four original-start arms and keep failures instead of dropping targets."""
    root.mkdir(parents=True, exist_ok=True)
    circuit = pilot.restore_circuit(source["candidate"]["circuit"])
    initial = np.asarray(source["candidate"]["theta"], dtype=float)
    target = load_target(circuit.num_qubits, source["target_id"]).state_vector_copy()
    methods = ARMS[order_index % len(ARMS) :] + ARMS[: order_index % len(ARMS)]
    arms: dict[str, dict[str, Any]] = {}
    for method in methods:
        failure_signature = pilot.digest({"source": source, "method": method, "config": asdict(config)})
        failed_path = root / method / "FAILED.json"
        if failed_path.exists():
            arms[method] = cast("dict[str, Any]", pilot.read_record(failed_path))
            if arms[method]["signature"] != failure_signature:
                msg = "Retained failed geometry arm identity changed."
                raise ValueError(msg)
            continue
        try:
            training = train_arm(root / method, circuit, initial, target, method, config)
            arms[method] = {"status": "complete", "training": training}
        except (ArithmeticError, RuntimeError, ValueError) as error:
            arms[method] = {
                "status": "failed",
                "signature": failure_signature,
                "failure": f"{type(error).__name__}: {error}",
                "partial_training_cost": failed_training_cost(root / method),
            }
            pilot.write_record(failed_path, arms[method])
    for arm in arms.values():
        if arm["status"] == "complete":
            arm["final_score"] = score_point(root, circuit, np.asarray(arm["training"]["theta"]), target)
    comparison = None
    if all(arm["status"] == "complete" for arm in arms.values()):
        common, points = common_cpu_points({method: arm["training"] for method, arm in arms.items()})
        for point in points:
            point["scores"] = {
                method: score_point(root, circuit, np.asarray(value["theta"]), target)
                for method, value in point["selected"].items()
            }
        comparison = {"common_cpu_seconds": common, "points": points}
    scores = [cast("dict[str, Any]", pilot.read_record(path)) for path in sorted((root / "scores").glob("*.json"))]
    return {
        "status": "complete" if comparison is not None else "complete_with_failed_arms",
        "target_id": source["target_id"],
        "replica": source["replica"],
        "arms": arms,
        "common_cpu_comparison": comparison,
        "scoring_ledger": {
            "unique_parameter_points": len(scores),
            "forward_channel_evaluations": sum(len(row["fidelities"]) for row in scores),
            "physical_attempted_forward_calls": sum(len(row["attempts"]) for row in scores),
            "physical_completed_forward_calls": sum(
                sum(attempt["completed"] for attempt in row["attempts"]) for row in scores
            ),
            "cpu_seconds": sum(sum(attempt["cpu_seconds"] for attempt in row["attempts"]) for row in scores),
            "checksums": [pilot.digest(row) for row in scores],
        },
    }


def freeze_protocol(output: Path) -> dict[str, Any]:
    """Prepare only the complete four-target scientific protocol and source identities."""
    manifest = runner.custody()
    starts_path = runner.OUTPUT / "starts.json"
    starts = cast("list[dict[str, Any]]", pilot.read_record(starts_path)["starts"])
    selected = [row for row in starts if row["target_id"] in TARGETS and row["replica"] == 11]
    if len(selected) != len(TARGETS) or {row["target_id"] for row in selected} != set(TARGETS):
        msg = "The four frozen geometry development sources at replica 11 are required."
        raise ValueError(msg)
    for source in selected:
        circuit = pilot.restore_circuit(source["candidate"]["circuit"])
        if circuit.num_qubits != 6 or len(circuit.gates) != 108:
            msg = "Scientific geometry starts must be original q6 108-gate circuits."
            raise ValueError(msg)
    protocol: dict[str, Any] = {
        "version": VERSION,
        "phase": "development",
        "targets": list(TARGETS),
        "replica": 11,
        "config": asdict(Config()),
        "arms": list(ARMS),
        "starts": selected,
        "benchmark_hashes": manifest["benchmark_hashes"],
        "source_hashes": numerical_sources(),
        "source_starts_sha256": hashlib.sha256(starts_path.read_bytes()).hexdigest(),
        "dampings": list(geometry.DAMPINGS),
        "geometry_version": geometry.VERSION,
        "lbfgs_version": optimizers.VERSION,
        "lbfgs_history": 10,
        "lbfgs_max_iterations": 300,
        "adam_config": asdict(adam.AdamConfig(updates=300, learning_rate=0.02)),
        "geometry_constants": {key: getattr(geometry, key) for key in ("ARMIJO", "BACKTRACK", "MAX_TRIALS", "GTOL")},
        "lbfgs_constants": {
            key: getattr(optimizers, key) for key in ("ARMIJO", "BACKTRACK", "MAX_TRIALS", "GTOL", "CURVATURE_TOL")
        },
        "training_objective": "unchanged exact calibrated local-depolarizing infidelity for all four arms",
        "primary": "equal target mean over all eleven noisy final conditions; noiseless separate",
        "endpoint_rule": (
            "final accepted geometry/L-BFGS iterate or final Adam update; no quality-based checkpoint selection"
        ),
        "cpu_scope": "identical outer optimization+intermediate-checkpoint process CPU for every arm",
        "excluded_cpu": "imports, custody, final/attempt-ledger writes and separately charged scoring",
        "common_cpu_fractions": list(CPU_FRACTIONS),
        "common_cpu_rule": (
            "latest available accepted parameters at fractions of shortest arm CPU, then exact all12 rescoring"
        ),
        "checkpoint_rule": (
            "initial one-call boundary then cumulative ten-call chunks, retaining complete pending search and moments"
        ),
        "metering": (
            "temporary process-local wrappers call unchanged kernels; serial arms, no concurrent threads per worker"
        ),
        "failure_accounting": (
            "attempted/completed gradients, metrics and solves; caught interruption CPU and replay retained"
        ),
        "uncatchable_interruptions": "SIGKILL/host failure require external resource-ledger reconciliation",
        "order": "rotate four-arm order by frozen target index",
        "randomness": "none; deterministic exact numerics",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "PROTOCOL.json"
    if path.exists() and pilot.read_record(path) != protocol:
        msg = "Frozen geometry protocol changed; use a new output directory."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return protocol


def verify_protocol(protocol: dict[str, Any]) -> None:
    """Authenticate benchmark, starting records and every imported numerical source."""
    if (
        protocol["version"] != VERSION
        or protocol["benchmark_hashes"] != runner.custody()["benchmark_hashes"]
        or protocol["source_hashes"] != numerical_sources()
        or protocol["source_starts_sha256"] != hashlib.sha256((runner.OUTPUT / "starts.json").read_bytes()).hexdigest()
    ):
        msg = "Geometry protocol custody or numerical source identity changed."
        raise ValueError(msg)


def execute_case(job: dict[str, Any]) -> dict[str, Any]:
    """Run an authenticated target, preserving complete failures and immutable results."""
    protocol, root = job["protocol"], Path(job["root"])
    verify_protocol(protocol)
    signature = pilot.digest({"protocol": protocol, "source": job["source"], "order_index": job["order_index"]})
    path = root / "RESULT.json"
    if path.exists():
        result = cast("dict[str, Any]", pilot.read_record(path))
        if result["signature"] != signature:
            msg = "Completed geometry case identity changed."
            raise ValueError(msg)
        return result
    try:
        result = run_case(root, job["source"], Config(**protocol["config"]), order_index=job["order_index"])
    except (ArithmeticError, RuntimeError, ValueError) as error:
        failure = f"{type(error).__name__}: {error}"
        result = {
            "status": "failed",
            "target_id": job["source"]["target_id"],
            "replica": job["source"]["replica"],
            "failure": failure,
            "arms": {method: {"status": "failed", "failure": failure} for method in ARMS},
        }
    result["signature"] = signature
    verify_protocol(protocol)
    pilot.write_record(path, result)
    return result


def write_scoreboard(output: Path, records: list[dict[str, Any]]) -> None:
    """Retain every target/method/profile, with missing endpoints explicitly failed."""
    rows: list[dict[str, Any]] = []
    for record in records:
        for method, arm in record["arms"].items():
            for noise_id in channels.NOISE_IDS:
                row: dict[str, Any] = {
                    "target_id": record["target_id"],
                    "replica": record["replica"],
                    "method": method,
                    "noise_id": noise_id,
                    "status": arm["status"],
                    "failure": arm.get("failure"),
                }
                if arm["status"] == "complete":
                    training, score = arm["training"], arm["final_score"]
                    row.update(
                        fidelity=score["fidelities"][noise_id],
                        mean_noisy_fidelity=score["mean_noisy_fidelity"],
                        training_cpu_seconds=training["cpu_seconds"],
                        channel_gradient_calls=training["true_channel_gradient_calls"],
                        metric_evaluations=training["metric_evaluations"],
                        physical_metric_cpu_seconds=training["physical"]["metric_cpu_seconds"],
                        physical_solve_cpu_seconds=training["physical"]["solve_cpu_seconds"],
                        physical_attempted_gradients=training["physical"]["channel_attempted"],
                        physical_row_applications=training["physical"]["statevector_row_operator_applications"],
                        optimizer_status=training["optimizer"]["status"],
                    )
                elif arm.get("partial_training_cost"):
                    partial = arm["partial_training_cost"]
                    row.update(
                        training_cpu_seconds=partial["cpu_seconds"],
                        physical_attempted_gradients=partial["physical"].get("channel_attempted", 0),
                        physical_metric_cpu_seconds=partial["physical"].get("metric_cpu_seconds", 0),
                        physical_solve_cpu_seconds=partial["physical"].get("solve_cpu_seconds", 0),
                    )
                rows.append(row)
    with (output / "SCOREBOARD.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare only, or explicitly execute the allocated fixed four-arm panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four serial target workers with single-thread BLAS limits.")
    if args.prepare_only:
        freeze_protocol(args.output)
        print(f"Prepared only: {args.output / 'PROTOCOL.json'}", flush=True)
        return 0
    protocol = cast("dict[str, Any]", pilot.read_record(args.output / "PROTOCOL.json"))
    verify_protocol(protocol)
    starts = {row["target_id"]: row for row in protocol["starts"]}
    jobs = [
        {
            "protocol": protocol,
            "source": starts[target_id],
            "order_index": index,
            "root": str(args.output / f"{target_id}__11"),
        }
        for index, target_id in enumerate(protocol["targets"])
    ]
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(execute_case, jobs):
            records.append(result)
            write_scoreboard(args.output, records)
            print(result["target_id"], result["status"], flush=True)
    pilot.write_record(
        args.output / "RESULT.json",
        {
            "status": "complete"
            if all(row["status"] == "complete" for row in records)
            else "complete_with_failed_arms",
            "protocol_checksum": pilot.digest(protocol),
            "results": [pilot.digest(row) for row in records],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
