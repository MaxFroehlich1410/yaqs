# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Experiment 1: matched exact-noisy/noiseless Adam from frozen hybrid starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import resource
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from benchmarks.state_preparation.execution import WorkerSetting, completed_jobs, parse_workers, resolve_workers
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider, exact_density_fidelity
from benchmarks.state_preparation.targets import load_target, load_target_collection
from experiments import state_preparation_pilot as pilot

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

VERSION = "yaqs.exact_adam_headroom.v1"
SOURCE_VERSION = "yaqs.compression_hybrid_construction_extension.v1"
SOURCE_SETTING = "H25_lr0p7"
SOURCE_ROUTE = f"{SOURCE_SETTING}:cold_final"
SEEDS = (11, 29)
ARMS = ("noiseless", "exact_noisy")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdamConfig:
    """Frozen scientific configuration; reduced updates are engineering-only."""

    updates: int = 100
    learning_rate: float = 0.005
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        """Reject invalid Adam hyperparameters."""
        values = (self.learning_rate, self.beta1, self.beta2, self.epsilon)
        if (
            type(self.updates) is not int
            or self.updates < 1
            or any(isinstance(value, bool) or not np.isfinite(value) for value in values)
            or self.learning_rate <= 0
            or self.epsilon <= 0
            or not 0 <= self.beta1 < 1
            or not 0 <= self.beta2 < 1
        ):
            msg = "Invalid Adam configuration."
            raise ValueError(msg)


def adam_step(
    theta: NDArray[np.float64],
    gradient: NDArray[np.float64],
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    step: int,
    config: AdamConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return one standard bias-corrected Adam update without mutating inputs."""
    arrays = (theta, gradient, first, second)
    if (
        type(step) is not int
        or step < 1
        or theta.ndim != 1
        or any(array.shape != theta.shape or not np.all(np.isfinite(array)) for array in arrays)
        or np.any(second < 0)
    ):
        msg = "Invalid Adam state or gradient."
        raise ValueError(msg)
    first = config.beta1 * first + (1 - config.beta1) * gradient
    second = config.beta2 * second + (1 - config.beta2) * gradient**2
    corrected_first = first / (1 - config.beta1**step)
    corrected_second = second / (1 - config.beta2**step)
    updated = theta - config.learning_rate * corrected_first / (np.sqrt(corrected_second) + config.epsilon)
    if not all(np.all(np.isfinite(array)) for array in (updated, first, second)):
        msg = "Nonfinite Adam update."
        raise FloatingPointError(msg)
    return updated, first, second


def evaluate(
    circuit: ParameterizedCircuit, theta: NDArray[np.float64], target: NDArray[np.complex128], update: int
) -> dict[str, object]:
    """Evaluate both objectives independently of the adjoint training kernel."""
    zero = ExactLocalPauliProvider(CalibratedLocalDepolarizingProvider(0.0, 0.0))
    return {
        "update": update,
        "theta": theta.tolist(),
        "noiseless_fidelity": exact_density_fidelity(circuit, theta, target, zero),
        "exact_noisy_fidelity": exact_density_fidelity(circuit, theta, target, ExactLocalPauliProvider()),
    }


def optimize(
    path: Path,
    circuit: ParameterizedCircuit,
    initial: NDArray[np.float64],
    target: NDArray[np.complex128],
    arm: str,
    config: AdamConfig,
    provenance: str,
) -> dict[str, object]:
    """Run one resumable Adam arm, preserving moments and bias-correction time."""
    if arm not in ARMS:
        msg = "Unknown exact Adam arm."
        raise ValueError(msg)
    calibration = (
        CalibratedLocalDepolarizingProvider() if arm == "exact_noisy" else CalibratedLocalDepolarizingProvider(0, 0)
    )
    signature = pilot.digest({
        "version": VERSION,
        "provenance": provenance,
        "config": asdict(config),
        "arm": arm,
        "circuit": pilot.circuit_record(circuit),
        "initial": initial.tolist(),
        "target": hashlib.sha256(target.tobytes()).hexdigest(),
        "noise": calibration.scientific_dict(),
    })
    state: dict[str, object] = {
        "signature": signature,
        "arm": arm,
        "completed_updates": 0,
        "theta": initial.tolist(),
        "first_moment": np.zeros_like(initial).tolist(),
        "second_moment": np.zeros_like(initial).tolist(),
        "trace": [],
        "evaluations": [],
        "training_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "invocation_seconds": 0.0,
    }
    if path.exists():
        state = pilot.read_record(path)
        if state["signature"] != signature:
            msg = "Exact Adam checkpoint configuration mismatch."
            raise ValueError(msg)
    completed = cast("int", state["completed_updates"])
    trace = cast("list[dict[str, object]]", state["trace"])
    evaluations = cast("list[dict[str, object]]", state["evaluations"])
    arrays = [np.asarray(state[key], dtype=np.float64) for key in ("theta", "first_moment", "second_moment")]
    if (
        type(completed) is not int
        or not 0 <= completed <= config.updates
        or len(trace) != completed
        or [row["update"] for row in trace] != list(range(1, completed + 1))
        or any(array.shape != initial.shape or not np.all(np.isfinite(array)) for array in arrays)
        or np.any(arrays[2] < 0)
        or (completed > 0 and (not evaluations or evaluations[-1]["update"] != completed))
    ):
        msg = "Invalid exact Adam checkpoint progress or moments."
        raise ValueError(msg)
    if state.get("status") == "complete":
        if completed != config.updates:
            msg = "Prematurely completed Adam checkpoint."
            raise ValueError(msg)
        return state
    theta, first, second = arrays
    started = time.perf_counter()
    previous_seconds = cast("float", state["invocation_seconds"])
    if not evaluations:
        evaluation_started = time.perf_counter()
        evaluations.append(evaluate(circuit, theta, target, 0))
        state["evaluation_seconds"] = time.perf_counter() - evaluation_started
    for step in range(completed + 1, config.updates + 1):
        training_started = time.perf_counter()
        gradient, loss, fidelity = density_loss_gradient(circuit, theta, target, calibration)
        updated, first, second = adam_step(theta, gradient, first, second, step, config)
        trace.append({
            "update": step,
            "pre_update_loss": loss,
            "pre_update_fidelity": fidelity,
            "gradient_norm": float(np.linalg.norm(gradient)),
            "update_norm": float(np.linalg.norm(updated - theta)),
            "learning_rate": config.learning_rate,
        })
        theta = updated
        state["training_seconds"] = cast("float", state["training_seconds"]) + time.perf_counter() - training_started
        if step % 10 == 0 or step == config.updates:
            evaluation_started = time.perf_counter()
            evaluations.append(evaluate(circuit, theta, target, step))
            if step == config.updates:
                gradient, loss, fidelity = density_loss_gradient(circuit, theta, target, calibration)
                oracle = cast("float", evaluations[-1][f"{arm}_fidelity"])
                if abs(fidelity - oracle) > 1e-10:
                    msg = "Training kernel disagrees with independent endpoint evaluation."
                    raise FloatingPointError(msg)
                state.update({"status": "complete", "final_gradient_norm": float(np.linalg.norm(gradient))})
            state["evaluation_seconds"] = (
                cast("float", state["evaluation_seconds"]) + time.perf_counter() - evaluation_started
            )
            state.update({
                "completed_updates": step,
                "theta": theta.tolist(),
                "first_moment": first.tolist(),
                "second_moment": second.tolist(),
                "invocation_seconds": previous_seconds + time.perf_counter() - started,
            })
            pilot.write_record(path, state)
    return state


def load_starts(source_root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Authenticate the completed extension and select only its eight final starts."""
    manifest = pilot.read_record(source_root / "manifest.json")
    result = pilot.read_record(source_root / "result.json")
    original_manifest = pilot.read_record(Path(cast("str", manifest["source_root"])) / "manifest.json")
    if (
        manifest["version"] != SOURCE_VERSION
        or result["status"] != "complete"
        or result["manifest_checksum"] != pilot.digest(manifest)
        or manifest["source_manifest_checksum"] != pilot.digest(original_manifest)
        or original_manifest["target_fixture_checksum"] != load_target_collection().checksum
        or manifest["targets"] != list(pilot.TARGETS)
        or manifest["config"] != json.loads(json.dumps(asdict(pilot.PilotConfig())))
        or manifest["noise"] != ExactLocalPauliProvider().scientific_dict()
    ):
        msg = "Source must be the completed, compatible q6 hybrid construction extension."
        raise ValueError(msg)
    starts: list[dict[str, object]] = []
    source_checksums = {}
    for target_id in pilot.TARGETS:
        for seed in SEEDS:
            relative = f"{SOURCE_SETTING}/{target_id}/seed_{seed}/result.json"
            record = pilot.read_record(source_root / relative)
            candidates = [
                row for row in cast("list[dict[str, object]]", record["candidates"]) if row["route"] == SOURCE_ROUTE
            ]
            if (
                len(candidates) != 1
                or record["target_id"] != target_id
                or record["replica"] != seed
                or record["setting"] != SOURCE_SETTING
                or record["schedule"] != {"online_updates": 25, "online_rate": 0.7}
                or record["include_warm"] is not False
            ):
                msg = f"Invalid selected source record: {relative}."
                raise ValueError(msg)
            candidate = candidates[0]
            circuit = pilot.restore_circuit(cast("dict[str, object]", candidate["circuit"]))
            if (
                circuit.num_qubits != 6
                or circuit.num_params != 108
                or len(circuit.gates) != 108
                or sum(len(g.sites) == 2 for g in circuit.gates) != 30
                or sorted(g.param_index for g in circuit.gates if g.param_index is not None) != list(range(108))
                or any(not g.noise_enabled for g in circuit.gates)
            ):
                msg = "Selected start does not have the frozen compact gate/parameter budget."
                raise ValueError(msg)
            source_checksums[relative] = pilot.digest(record)
            starts.append({"target_id": target_id, "replica": seed, "candidate": candidate})
    return {
        "source_manifest_checksum": pilot.digest(manifest),
        "source_result_checksum": pilot.digest(result),
        "original_manifest_checksum": pilot.digest(original_manifest),
        "selected_source_checksums": source_checksums,
    }, starts


def preflight(start: dict[str, object]) -> dict[str, object]:
    """Reproduce both fidelities and independently check directional derivatives."""
    candidate = cast("dict[str, object]", start["candidate"])
    circuit = pilot.restore_circuit(cast("dict[str, object]", candidate["circuit"]))
    theta = np.asarray(candidate["theta"], dtype=np.float64)
    target = load_target(6, cast("str", start["target_id"])).state_vector_copy()
    evaluation = evaluate(circuit, theta, target, 0)
    direction = np.random.default_rng(2026090601 + cast("int", start["replica"])).normal(size=theta.size)
    direction /= np.linalg.norm(direction)
    checks = {}
    for arm in ARMS:
        calibration = (
            CalibratedLocalDepolarizingProvider() if arm == "exact_noisy" else CalibratedLocalDepolarizingProvider(0, 0)
        )
        provider = ExactLocalPauliProvider(calibration)
        gradient, _loss, fidelity = density_loss_gradient(circuit, theta, target, calibration)
        upper = 1 - exact_density_fidelity(circuit, theta + 1e-5 * direction, target, provider)
        lower = 1 - exact_density_fidelity(circuit, theta - 1e-5 * direction, target, provider)
        error = abs(float(gradient @ direction) - (upper - lower) / 2e-5)
        if (
            error > 2e-7
            or abs(fidelity - cast("float", evaluation[f"{arm}_fidelity"])) > 1e-10
            or abs(fidelity - cast("float", candidate[f"{arm}_fidelity"])) > 1e-10
        ):
            msg = "Source fidelity or exact directional-gradient preflight failed."
            raise FloatingPointError(msg)
        checks[arm] = {"directional_gradient_error": error, "fidelity": fidelity}
    return checks


@dataclass(frozen=True)
class Job:
    """One independent target/seed pair, with both Adam arms from the same start."""

    root: Path
    start: dict[str, object]
    config: AdamConfig
    provenance: str


def run_job(job: Job) -> dict[str, object]:
    """Run both paired continuations and retain independent reference evaluations."""
    target_id = cast("str", job.start["target_id"])
    replica = cast("int", job.start["replica"])
    directory = job.root / target_id / f"seed_{replica}"
    source = cast("dict[str, object]", job.start["candidate"])
    circuit = pilot.restore_circuit(cast("dict[str, object]", source["circuit"]))
    initial = np.asarray(source["theta"], dtype=np.float64)
    target = load_target(6, target_id).state_vector_copy()
    pilot.write_record(directory / "start.json", job.start)
    candidates = [{**evaluate(circuit, initial, target, 0), "route": "frozen"}]
    arm_checksums = {}
    for arm in ARMS:
        state = optimize(directory / f"{arm}.json", circuit, initial.copy(), target, arm, job.config, job.provenance)
        endpoint = cast("list[dict[str, object]]", state["evaluations"])[-1]
        candidates.append({**endpoint, "route": arm})
        arm_checksums[arm] = pilot.digest(state)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result: dict[str, object] = {
        "status": "complete",
        "target_id": target_id,
        "replica": replica,
        "start_checksum": pilot.digest(job.start),
        "initial_theta_checksum": pilot.digest(initial.tolist()),
        "circuit_checksum": pilot.digest(source["circuit"]),
        "arm_checksums": arm_checksums,
        "candidates": candidates,
        "worker_pid": os.getpid(),
        "process_high_water_rss_bytes": int(rss if platform.system() == "Darwin" else rss * 1024),
    }
    pilot.write_record(directory / "result.json", result)
    return result


def run_experiment(
    root: Path,
    source_root: Path,
    *,
    config: AdamConfig | None = None,
    workers: WorkerSetting = "auto",
    resume: bool = False,
) -> dict[str, object]:
    """Authenticate inputs, finish all preflights, and run the paired experiment."""
    config = AdamConfig() if config is None else config
    provenance, starts = load_starts(source_root)
    count = resolve_workers(workers, len(starts))
    manifest: dict[str, object] = {
        "version": VERSION,
        "config": asdict(config),
        "source_root": str(source_root.resolve()),
        "provenance": provenance,
        "target_fixture_checksum": load_target_collection().checksum,
        "runtime_fingerprint": pilot.digest({
            "runtime": pilot.source_fingerprint(),
            "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": version("scipy"),
            "qiskit": version("qiskit"),
        },
        "noise": ExactLocalPauliProvider().scientific_dict(),
        "selection": "final iterate; no tuning; descriptive equal-target mean; no held-out claims",
    }
    if root.exists():
        if not resume or not (root / "manifest.json").exists() or pilot.read_record(root / "manifest.json") != manifest:
            msg = "Existing output requires explicit resume and an identical manifest."
            raise ValueError(msg)
    else:
        if resume:
            msg = "Cannot resume nonexistent output."
            raise ValueError(msg)
        root.mkdir(parents=True)
        pilot.write_record(root / "manifest.json", manifest)
    if (root / "result.json").exists():
        saved = pilot.read_record(root / "result.json")
        if saved["status"] != "complete" or saved["manifest_checksum"] != pilot.digest(manifest):
            msg = "Completed result does not match the experiment manifest."
            raise ValueError(msg)
        checksums = cast("dict[str, str]", saved["record_checksums"])
        for start in starts:
            relative = f"{start['target_id']}/seed_{start['replica']}/result.json"
            record = pilot.read_record(root / relative)
            if pilot.digest(record) != checksums[relative] or record["start_checksum"] != pilot.digest(start):
                msg = "Completed result does not match its start/job checksums."
                raise ValueError(msg)
            arm_checksums = cast("dict[str, str]", record["arm_checksums"])
            for arm in ARMS:
                state = pilot.read_record((root / relative).parent / f"{arm}.json")
                if pilot.digest(state) != arm_checksums[arm]:
                    msg = "Completed result does not match its Adam checkpoints."
                    raise ValueError(msg)
        return saved
    started = time.perf_counter()
    checks = []
    for index, result in completed_jobs(preflight, starts, workers=workers):
        checks.append({"target_id": starts[index]["target_id"], "replica": starts[index]["replica"], "checks": result})
    checks.sort(key=lambda row: (str(row["target_id"]), cast("int", row["replica"])))
    pilot.write_record(root / "preflight.json", {"status": "passed", "checks": checks})
    print("All eight source and exact-gradient preflights passed.", flush=True)
    jobs = [Job(root, start, config, pilot.digest(manifest)) for start in starts]
    records = {}
    for index, result in completed_jobs(run_job, jobs, workers=workers):
        records[index] = result
        print(
            f"Exact Adam {len(records)}/{len(jobs)} pairs: {result['target_id']} seed {result['replica']}", flush=True
        )
    ordered = [records[index] for index in range(len(jobs))]
    aggregate = pilot.aggregate(ordered)
    differences = []
    for row in ordered:
        candidates = {item["route"]: item for item in cast("list[dict[str, object]]", row["candidates"])}
        frozen, zero, noisy = [cast("float", candidates[arm]["exact_noisy_fidelity"]) for arm in ("frozen", *ARMS)]
        differences.append({
            "target_id": row["target_id"],
            "replica": row["replica"],
            "noisy_minus_noiseless": noisy - zero,
            "noisy_minus_frozen": noisy - frozen,
            "noiseless_minus_frozen": zero - frozen,
        })
    result: dict[str, object] = {
        "version": VERSION,
        "status": "complete",
        "manifest_checksum": pilot.digest(manifest),
        "completed_pairs": len(jobs),
        "completed_continuations": len(jobs) * 2,
        "gradient_updates": len(jobs) * 2 * config.updates,
        "aggregate": aggregate,
        "paired_differences": differences,
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "workers": count,
        "record_checksums": {
            f"{row['target_id']}/seed_{row['replica']}/result.json": pilot.digest(row) for row in ordered
        },
    }
    pilot.write_record(root / "result.json", result)
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    """Run only Experiment 1 with its frozen scientific configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=parse_workers, default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(arguments)
    try:
        result = run_experiment(args.output_root, args.source_root, workers=args.workers, resume=args.resume)
    except Exception:
        LOGGER.exception("Exact Adam experiment stopped; existing checkpoints retained")
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
