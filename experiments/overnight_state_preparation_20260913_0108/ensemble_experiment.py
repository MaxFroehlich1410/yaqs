# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Preregistered fixed-structure ensemble-objective development comparison.

Prepare only before execution. Every arm starts from the original frozen
parameters and receives the same cap in individual channel-gradient calls.
Final iterate selection and post-training common-CPU rescoring are explicit.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from benchmarks.state_preparation.ballarin import materialize_ballarin_circuit
from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_exact_adam as adam
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import channels, ensemble_objective, optimizers, runner

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

VERSION = "overnight.ensemble_objective_comparison.v1"
TARGETS = ("gaussian_mu0p5_sigma0p1", "tfim_critical", "haar_random_1", "random_mps_bond2")
CPU_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Config:
    """Fixed optimizer budgets; smaller caps are allowed only in engineering tests."""

    channel_evaluations: int = 1100
    history: int = 10
    checkpoint_every: int = 10
    adam_learning_rate: float = 0.02

    def __post_init__(self) -> None:
        """Reject nonintegral or unequal per-profile budgets."""
        if (
            type(self.channel_evaluations) is not int
            or self.channel_evaluations < 22
            or self.channel_evaluations % 22
            or type(self.history) is not int
            or self.history < 1
            or type(self.checkpoint_every) is not int
            or self.checkpoint_every < 1
            or isinstance(self.adam_learning_rate, bool)
            or not np.isfinite(self.adam_learning_rate)
            or self.adam_learning_rate <= 0
        ):
            msg = "Ensemble budgets must be positive, with the channel cap divisible by 22."
            raise ValueError(msg)


def arm_specs(config: Config) -> dict[str, dict[str, Any]]:
    """Return the complete six-arm comparison with equal individual-channel caps."""
    definitions = (
        ("local_lbfgs", "lbfgs", ("depolarizing_1s_all",)),
        ("mean2_lbfgs", "lbfgs", ("depolarizing_1s_all", "dephasing_1s_all")),
        ("mean11_lbfgs", "lbfgs", ensemble_objective.NOISY_IDS),
        ("local_adam", "adam", ("depolarizing_1s_all",)),
        ("noiseless_adam", "adam", ("noiseless",)),
        ("noiseless_lbfgs", "lbfgs", ("noiseless",)),
    )
    return {
        name: {
            "optimizer": method,
            "noise_ids": list(profiles),
            "joint_cap": config.channel_evaluations // len(profiles),
        }
        for name, method, profiles in definitions
    }


def numerical_sources() -> dict[str, str]:
    """Fingerprint the driver, immutable protocol prose, and imported numerics."""
    result = runner.source_hashes()
    for name in ("ensemble_objective.py", "ensemble_experiment.py", "ENSEMBLE_PROTOCOL.md"):
        path = Path(__file__).with_name(name)
        result[str(path.relative_to(runner.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def component_gradient(
    circuit: ParameterizedCircuit, theta: NDArray, target: NDArray, noise_id: str, *, fast: bool
) -> tuple[NDArray, float, float]:
    """Use the existing fast adjoint for strong local and noiseless baseline arms."""
    if fast and noise_id in {"noiseless", "depolarizing_1s_all"}:
        calibration = (
            CalibratedLocalDepolarizingProvider(0, 0)
            if noise_id == "noiseless"
            else CalibratedLocalDepolarizingProvider()
        )
        return density_loss_gradient(circuit, theta, target, calibration)
    return ensemble_objective.loss_gradient(circuit, theta, target, noise_id)


def minimize_adam(
    objective: Callable[[NDArray], tuple[float, NDArray]],
    initial: NDArray,
    config: adam.AdamConfig,
    *,
    state: dict[str, Any] | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run standard Adam with complete moment persistence and final-iterate output.

    The final update uses the last budgeted gradient. Its loss is deliberately
    not inferred from the pre-update loss; final scoring is separate work.
    """
    current: dict[str, Any] = (
        copy.deepcopy(state)
        if state is not None
        else {
            "version": "overnight.ensemble_adam.v1",
            "initial": initial.tolist(),
            "config": asdict(config),
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
        current["version"] != "overnight.ensemble_adam.v1"
        or current["config"] != asdict(config)
        or not np.array_equal(np.asarray(current["initial"]), initial)
        or type(current["iterations"]) is not int
        or not 0 <= current["iterations"] <= config.updates
        or current["iterations"] != current["evaluations"]
    ):
        msg = "Adam checkpoint is incompatible with the frozen initial state or configuration."
        raise ValueError(msg)
    theta, first, second = (np.asarray(current[key], dtype=float) for key in ("theta", "first_moment", "second_moment"))
    if any(
        value.shape != initial.shape or not np.all(np.isfinite(value)) for value in (theta, first, second)
    ) or np.any(second < 0):
        msg = "Adam checkpoint contains malformed numerical arrays."
        raise ValueError(msg)
    for step in range(current["iterations"] + 1, config.updates + 1):
        loss, gradient = objective(theta.copy())
        if not np.isfinite(loss):
            msg = "Adam objective is nonfinite."
            raise FloatingPointError(msg)
        theta, first, second = adam.adam_step(theta, gradient, first, second, step, config)
        current.update(
            theta=theta.tolist(),
            first_moment=first.tolist(),
            second_moment=second.tolist(),
            iterations=step,
            evaluations=step,
            last_pre_update_loss=loss,
        )
        if callback is not None:
            callback(copy.deepcopy(current))
    current["status"] = "evaluation_limit"
    return current


def train_arm(
    root: Path,
    circuit: ParameterizedCircuit,
    initial: NDArray,
    target: NDArray,
    spec: dict[str, Any],
    config: Config,
) -> dict[str, Any]:
    """Train or exactly resume one arm while charging replayed interruption work.

    Checkpoints retain all optimizer state and every available accepted iterate.
    Caught interruptions preserve attempted and completed component-call counts
    in a separate attempt ledger. SIGKILL/host failure requires external resource
    accounting; the finally block cannot observe uncatchable process termination.
    """
    signature = pilot.digest({
        "version": VERSION,
        "circuit": pilot.circuit_record(circuit),
        "initial": initial.tolist(),
        "target": hashlib.sha256(target.tobytes()).hexdigest(),
        "spec": spec,
        "config": asdict(config),
    })
    root.mkdir(parents=True, exist_ok=True)
    final_path, partial_path, attempts_path = (
        root / name for name in ("TRAINING.json", "PARTIAL.json", "ATTEMPTS.json")
    )
    if final_path.exists():
        final = cast("dict[str, Any]", pilot.read_record(final_path))
        if final["signature"] != signature:
            msg = "Completed ensemble training signature changed."
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
        msg = "Partial ensemble state or attempt-ledger identity changed."
        raise ValueError(msg)
    prior_cpu = sum(row["cpu_seconds"] for row in attempts["attempts"])
    prior_wall = sum(row["wall_seconds"] for row in attempts["attempts"])
    curve = (
        copy.deepcopy(prior["curve"])
        if prior is not None
        else [
            {
                "theta": initial.tolist(),
                "iterations": 0,
                "joint_evaluations": 0,
                "individual_channel_evaluations": 0,
                "cpu_seconds": 0.0,
                "training_loss": None,
            }
        ]
    )
    cpu, wall = time.process_time(), time.perf_counter()
    counters = {
        "attempted_joint_calls": 0,
        "completed_joint_calls": 0,
        "attempted_channel_calls": 0,
        "completed_channel_calls": 0,
    }
    profiles = spec["noise_ids"]

    def objective(theta: NDArray) -> tuple[float, NDArray]:
        counters["attempted_joint_calls"] += 1
        gradient, loss = np.zeros_like(theta), 0.0
        for noise_id in profiles:
            counters["attempted_channel_calls"] += 1
            local_gradient, local_loss, _ = component_gradient(
                circuit, theta, target, noise_id, fast=len(profiles) == 1
            )
            counters["completed_channel_calls"] += 1
            gradient += local_gradient
            loss += local_loss
        counters["completed_joint_calls"] += 1
        return loss / len(profiles), gradient / len(profiles)

    def point(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "theta": state["theta"],
            "iterations": state["iterations"],
            "joint_evaluations": state["evaluations"],
            "individual_channel_evaluations": len(profiles) * state["evaluations"],
            "cpu_seconds": prior_cpu + time.process_time() - cpu,
            "training_loss": state["loss"] if spec["optimizer"] == "lbfgs" else None,
        }

    def save(state: dict[str, Any]) -> None:
        curve.append(point(state))
        if state["iterations"] % config.checkpoint_every == 0:
            pilot.write_record(partial_path, {"signature": signature, "optimizer": state, "curve": curve})

    def optimize() -> dict[str, Any]:
        if spec["optimizer"] == "lbfgs":
            return optimizers.minimize(
                objective,
                initial,
                max_evaluations=spec["joint_cap"],
                max_iterations=spec["joint_cap"],
                history=config.history,
                state=None if prior is None else prior["optimizer"],
                callback=save,
            )
        if spec["optimizer"] == "adam":
            return minimize_adam(
                objective,
                initial,
                adam.AdamConfig(updates=spec["joint_cap"], learning_rate=config.adam_learning_rate),
                state=None if prior is None else prior["optimizer"],
                callback=save,
            )
        msg = "Unsupported ensemble optimizer."
        raise ValueError(msg)

    completed, failure = False, None
    try:
        state = optimize()
        completed = True
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        attempts["attempts"].append({
            **counters,
            "cpu_seconds": time.process_time() - cpu,
            "wall_seconds": time.perf_counter() - wall,
            "completed": completed,
            "failure": failure,
        })
        pilot.write_record(attempts_path, attempts)
    final_cpu = sum(row["cpu_seconds"] for row in attempts["attempts"])
    curve.append({**point(state), "cpu_seconds": final_cpu})
    logical_calls = len(profiles) * state["evaluations"]
    physical_calls = sum(row["completed_channel_calls"] for row in attempts["attempts"])
    result = {
        "signature": signature,
        "status": "complete",
        "optimizer": state,
        "spec": spec,
        "curve": curve,
        "theta": state["theta"],
        "joint_gradient_calls": state["evaluations"],
        "individual_channel_gradient_calls": logical_calls,
        "density_direction_passes": 2 * logical_calls,
        "physical_completed_channel_calls": physical_calls,
        "physical_attempted_channel_calls": sum(row["attempted_channel_calls"] for row in attempts["attempts"]),
        "physical_completed_direction_passes": 2 * physical_calls,
        "replayed_completed_channel_calls": physical_calls - logical_calls,
        "cpu_seconds": final_cpu,
        "wall_seconds": prior_wall + attempts["attempts"][-1]["wall_seconds"],
        "attempts_checksum": pilot.digest(attempts),
        "training_trajectories": 0,
        "endpoint_rule": "final accepted L-BFGS iterate or final Adam update; no best-checkpoint selection",
    }
    pilot.write_record(final_path, result)
    return result


def gradient_cosines(root: Path, circuit: ParameterizedCircuit, theta: NDArray, target: NDArray) -> dict[str, Any]:
    """Measure exact channel-gradient alignment at the common frozen starting point."""
    signature = pilot.digest({
        "circuit": pilot.circuit_record(circuit),
        "theta": theta.tolist(),
        "target": hashlib.sha256(target.tobytes()).hexdigest(),
    })
    path = root / "START_GRADIENT_DIAGNOSTIC.json"
    if path.exists():
        result = cast("dict[str, Any]", pilot.read_record(path))
        if result["signature"] != signature:
            msg = "Gradient diagnostic starting identity changed."
            raise ValueError(msg)
        return result
    cpu = time.process_time()
    gradients = {
        key: component_gradient(circuit, theta, target, key, fast=True)[0] for key in ensemble_objective.NOISY_IDS
    }
    norms = {key: float(np.linalg.norm(value)) for key, value in gradients.items()}
    cosines = {
        first: {
            second: float(np.clip(np.dot(value, gradients[second]) / (norms[first] * norms[second]), -1, 1))
            if norms[first] * norms[second] > 0
            else None
            for second in gradients
        }
        for first, value in gradients.items()
    }
    result: dict[str, Any] = {
        "signature": signature,
        "point": "shared_frozen_start",
        "noise_ids": list(gradients),
        "gradients": {key: value.tolist() for key, value in gradients.items()},
        "gradient_norms": norms,
        "cosines": cosines,
        "individual_channel_gradient_calls": len(gradients),
        "density_direction_passes": 2 * len(gradients),
        "cpu_seconds": time.process_time() - cpu,
        "interpretation": "negative cosine indicates local first-order objective conflict, not global impossibility",
    }
    pilot.write_record(path, result)
    return result


def score_point(root: Path, circuit: ParameterizedCircuit, theta: NDArray, target: NDArray) -> dict[str, Any]:
    """Cache all twelve exact scores without using them for endpoint selection."""
    signature = pilot.digest({
        "circuit": pilot.circuit_record(circuit),
        "theta": theta.tolist(),
        "target": hashlib.sha256(target.tobytes()).hexdigest(),
    })
    path = root / "scores" / f"{signature}.json"
    if path.exists():
        saved = cast("dict[str, Any]", pilot.read_record(path))
        if saved["signature"] != signature:
            msg = "Cached ensemble score identity changed."
            raise ValueError(msg)
        return saved
    cpu = time.process_time()
    fidelities = {key: channels.evaluate(circuit, theta, target, key) for key in channels.NOISE_IDS}
    result = {
        "signature": signature,
        "theta": theta.tolist(),
        "fidelities": fidelities,
        "mean_noisy_fidelity": float(np.mean([fidelities[key] for key in ensemble_objective.NOISY_IDS])),
        "logical_counts": runner.counts(circuit),
        "ballarin_counts": runner.counts(materialize_ballarin_circuit(circuit, theta).circuit),
        "cpu_seconds": time.process_time() - cpu,
        "forward_channel_evaluations": len(channels.NOISE_IDS),
        "density_direction_passes": len(channels.NOISE_IDS),
    }
    pilot.write_record(path, result)
    return result


def common_cpu_points(arms: dict[str, dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    """Select latest available iterates at common CPU fractions without score access."""
    common = min(arm["cpu_seconds"] for arm in arms.values())
    points = []
    for fraction in CPU_FRACTIONS:
        budget = fraction * common
        selected = {
            name: next(point for point in reversed(arm["curve"]) if point["cpu_seconds"] <= budget)
            for name, arm in arms.items()
        }
        points.append({"fraction": fraction, "cpu_budget_seconds": budget, "selected": selected})
    return common, points


def run_case(root: Path, source: dict[str, Any], config: Config, *, order_index: int = 0) -> dict[str, Any]:
    """Execute six fixed-start arms, retaining failures and separate diagnostic work."""
    circuit = pilot.restore_circuit(source["candidate"]["circuit"])
    initial = np.asarray(source["candidate"]["theta"], dtype=float)
    target = load_target(circuit.num_qubits, source["target_id"]).state_vector_copy()
    root.mkdir(parents=True, exist_ok=True)
    diagnostic = gradient_cosines(root, circuit, initial, target)
    specs = arm_specs(config)
    names = list(specs)
    names = names[order_index % len(names) :] + names[: order_index % len(names)]
    arms: dict[str, dict[str, Any]] = {}
    for name in names:
        failure_signature = pilot.digest({"source": source, "spec": specs[name], "config": asdict(config)})
        failed_path = root / name / "FAILED.json"
        if failed_path.exists():
            arms[name] = cast("dict[str, Any]", pilot.read_record(failed_path))
            if arms[name]["signature"] != failure_signature:
                msg = "Retained failed-arm identity changed."
                raise ValueError(msg)
            continue
        try:
            training = train_arm(root / name, circuit, initial, target, specs[name], config)
            score = score_point(root, circuit, np.asarray(training["theta"]), target)
            arms[name] = {"status": "complete", "training": training, "final_score": score}
        except (ArithmeticError, RuntimeError, ValueError) as error:
            arms[name] = {
                "signature": failure_signature,
                "status": "failed",
                "failure": f"{type(error).__name__}: {error}",
                "spec": specs[name],
            }
            pilot.write_record(failed_path, arms[name])
    comparison = None
    if all(arm["status"] == "complete" for arm in arms.values()):
        common, points = common_cpu_points({name: arm["training"] for name, arm in arms.items()})
        for point in points:
            point["scores"] = {
                name: score_point(root, circuit, np.asarray(value["theta"]), target)
                for name, value in point["selected"].items()
            }
        comparison = {"common_cpu_seconds": common, "points": points}
    scoring_records = [pilot.read_record(path) for path in sorted((root / "scores").glob("*.json"))]
    return {
        "status": "complete" if comparison is not None else "complete_with_failed_arms",
        "target_id": source["target_id"],
        "replica": source["replica"],
        "arms": arms,
        "gradient_diagnostic": diagnostic,
        "common_cpu_comparison": comparison,
        "post_training_scoring": {
            "unique_parameter_points": len(scoring_records),
            "forward_channel_evaluations": sum(
                cast("int", row["forward_channel_evaluations"]) for row in scoring_records
            ),
            "density_direction_passes": sum(cast("int", row["density_direction_passes"]) for row in scoring_records),
            "cpu_seconds": sum(cast("float", row["cpu_seconds"]) for row in scoring_records),
            "checksums": [pilot.digest(row) for row in scoring_records],
        },
    }


def freeze_protocol(output: Path) -> dict[str, Any]:
    """Prepare the immutable scientific protocol and authenticate all original starts."""
    config = Config()
    manifest = runner.custody()
    starts_path = runner.OUTPUT / "starts.json"
    sources = cast("list[dict[str, Any]]", pilot.read_record(starts_path)["starts"])
    selected = [row for row in sources if row["target_id"] in TARGETS and row["replica"] == 11]
    if len(selected) != len(TARGETS) or {row["target_id"] for row in selected} != set(TARGETS):
        msg = "The four development starts at replica 11 are required."
        raise ValueError(msg)
    for source in selected:
        circuit = pilot.restore_circuit(source["candidate"]["circuit"])
        if circuit.num_qubits != 6 or len(circuit.gates) != 108:
            msg = "Scientific ensemble starts must be the original q6 108-gate circuits."
            raise ValueError(msg)
    protocol = {
        "version": VERSION,
        "phase": "development",
        "targets": list(TARGETS),
        "replica": 11,
        "config": asdict(config),
        "arms": arm_specs(config),
        "starts": selected,
        "benchmark_hashes": manifest["benchmark_hashes"],
        "source_hashes": numerical_sources(),
        "source_starts_sha256": hashlib.sha256(starts_path.read_bytes()).hexdigest(),
        "optimizer_version": optimizers.VERSION,
        "optimizer_constants": {
            key: getattr(optimizers, key) for key in ("ARMIJO", "BACKTRACK", "MAX_TRIALS", "GTOL", "CURVATURE_TOL")
        },
        "adam_constants": asdict(
            adam.AdamConfig(updates=config.channel_evaluations, learning_rate=config.adam_learning_rate)
        ),
        "lbfgs_iteration_cap": "joint-call cap; all remaining optimizer numerical defaults unchanged",
        "endpoint_rule": "final accepted L-BFGS iterate or final Adam update; no best-checkpoint selection",
        "primary": "equal target mean across all eleven unchanged noisy benchmark conditions",
        "secondary": "local depolarizing, dephasing, Ballarin, and noiseless scores separately",
        "common_cpu_fractions": list(CPU_FRACTIONS),
        "common_cpu_rule": "latest available accepted iterate at each fraction of the shortest completed arm CPU",
        "rescoring": (
            "all12 exact conditions after selection by CPU only; cached by exact parameters and charged separately"
        ),
        "gradient_diagnostic": (
            "all11 exact per-profile gradients and pairwise cosines at the same frozen starting theta"
        ),
        "order": "rotate the six-arm order by target index in the frozen target list",
        "budget_rule": (
            "1100 individual channel-gradient calls per arm; unused budget on convergence is reported, not recycled"
        ),
        "randomness": "deterministic exact density training and scoring; no diagnostic or final trajectory randomness",
        "failure_rule": (
            "keep numerical failures and partial ledgers; do not silently remove targets or retry hyperparameters"
        ),
        "cpu_scope": (
            "optimization including intermediate checkpoints; final/attempt ledger writes, imports, custody, "
            "diagnostics and rescoring excluded and reported separately where applicable"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "PROTOCOL.json"
    if path.exists() and pilot.read_record(path) != protocol:
        msg = "Frozen ensemble protocol differs; a changed experiment needs a new output directory."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return protocol


def verify_protocol(protocol: dict[str, Any]) -> None:
    """Authenticate benchmark, start, optimizer, and numerical-source custody."""
    if (
        protocol["version"] != VERSION
        or protocol["benchmark_hashes"] != runner.custody()["benchmark_hashes"]
        or protocol["source_hashes"] != numerical_sources()
        or protocol["source_starts_sha256"] != hashlib.sha256((runner.OUTPUT / "starts.json").read_bytes()).hexdigest()
    ):
        msg = "Ensemble protocol custody or numerical source identity changed."
        raise ValueError(msg)


def execute_case(job: dict[str, Any]) -> dict[str, Any]:
    """Execute an authenticated target or reuse its exact completed artifact."""
    protocol, root = job["protocol"], Path(job["root"])
    verify_protocol(protocol)
    signature = pilot.digest({"protocol": protocol, "source": job["source"], "order_index": job["order_index"]})
    path = root / "RESULT.json"
    if path.exists():
        result = cast("dict[str, Any]", pilot.read_record(path))
        if result["signature"] != signature:
            msg = "Completed ensemble target signature changed."
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
            "arms": {name: {"status": "failed", "failure": failure} for name in protocol["arms"]},
        }
    result["signature"] = signature
    verify_protocol(protocol)
    pilot.write_record(path, result)
    return result


def write_scoreboard(output: Path, records: list[dict[str, Any]]) -> None:
    """Write all final profile scores and explicit failed-arm placeholders."""
    rows = []
    for record in records:
        for method, arm in record["arms"].items():
            for noise_id in channels.NOISE_IDS:
                row = {
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
                        joint_calls=training["joint_gradient_calls"],
                        individual_channel_calls=training["individual_channel_gradient_calls"],
                        direction_passes=training["density_direction_passes"],
                        physical_completed_channel_calls=training["physical_completed_channel_calls"],
                        optimizer_status=training["optimizer"]["status"],
                    )
                rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with (output / "SCOREBOARD.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare only, or explicitly run the frozen six-arm serial-per-target panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four workers, each with single-thread BLAS limits.")
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
            if all(row["status"] == "complete" for row in records)
            else "complete_with_failed_arms",
            "protocol_checksum": pilot.digest(protocol),
            "results": [pilot.digest(row) for row in records],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
