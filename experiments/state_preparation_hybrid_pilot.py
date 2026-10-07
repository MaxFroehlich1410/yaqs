# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Matched cold-construction and warm-refinement hybrid Krotov pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from benchmarks.state_preparation.execution import WorkerSetting, completed_jobs, parse_workers, resolve_workers
from benchmarks.state_preparation.pilot_components import (
    ExactLocalPauliProvider,
    delete_gates,
    pruning_scores,
    select_deletions,
)
from benchmarks.state_preparation.targets import load_target, load_target_collection
from experiments import state_preparation_pilot as pilot
from mqt.yaqs.optimization import (
    KrotovOptions,
    KrotovResult,
    KrotovTJMOptions,
    KrotovTruncation,
    StatevectorBackend,
    train_krotov_noisy_state_preparation_hybrid,
    train_krotov_state_preparation_hybrid,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

VERSION = "yaqs.compression_hybrid_screen.v1"
CONSTRUCTION_VERSION = "yaqs.compression_hybrid_construction_extension.v1"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Schedule:
    """Finite online phase, followed by the unchanged constant-rate batch tail."""

    online_updates: int = 0
    online_rate: float = 0.1

    def __post_init__(self) -> None:
        """Validate the schedule before any output is created.

        Raises:
            ValueError: If the online budget or learning rate is invalid.
        """
        if (
            type(self.online_updates) is not int
            or self.online_updates < 0
            or isinstance(self.online_rate, bool)
            or not np.isfinite(self.online_rate)
            or self.online_rate <= 0
        ):
            msg = "Invalid hybrid schedule."
            raise ValueError(msg)

    @property
    def label(self) -> str:
        """Stable directory/setting label, with one canonical batch control."""
        if self.online_updates == 0:
            return "B"
        return f"H{self.online_updates}_lr{self.online_rate:g}".replace(".", "p")


SCHEDULES = (Schedule(), *tuple(Schedule(steps, rate) for steps in (5, 10, 20) for rate in (0.1, 0.2, 0.5)))
EXTENSION_SCHEDULES = tuple(Schedule(steps, rate) for steps in (25, 30) for rate in (0.7, 0.9))


def statevector_chunk(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    options: KrotovOptions,
    trajectories: int,
    seed: int,
) -> KrotovResult:
    """Run a constant-rate chunk with the same global noise counters as MPS training."""
    engine = StatevectorBackend(circuit, target)
    sampling = KrotovTJMOptions(num_trajectories=max(1, trajectories), random_seed=seed)
    provider = ExactLocalPauliProvider()
    trace: dict[str, list[float | int | str]] = {
        key: [] for key in ("step", "phase", "loss", "fidelity", "step_size", "gradient_norm", "update_norm")
    }
    for step in range(options.max_iterations + 1):
        before = theta.copy()
        maps = (
            engine.sample_noise_maps(theta, sampling, provider, iteration=options.seed + step if step else 0)
            if trajectories
            else None
        )
        online = 0 < step <= options.switch_iteration
        rate = (options.online_step_size if online else options.batch_step_size) if step else 0.0
        signal = np.zeros_like(theta)
        if online:
            theta, signal = engine.online_update(theta, rate, maps)
        elif step:
            signal, _, _ = engine.gradient(theta, maps)
            updated = theta - rate * signal
            theta = updated
        loss, fidelity = engine.metrics(theta, maps)
        values = (
            step,
            "init" if not step else "online" if online else "batch",
            loss,
            fidelity,
            rate,
            float(np.linalg.norm(signal)),
            float(np.linalg.norm(theta - before)),
        )
        for key, value in zip(trace, values, strict=True):
            trace[key].append(value)
    return KrotovResult(theta=theta, bias=0.0, trace=trace)


def optimize_block(
    path: Path,
    circuit: ParameterizedCircuit,
    initial: NDArray[np.float64],
    target: NDArray[np.complex128],
    *,
    updates: int,
    batch_rate: float,
    schedule: Schedule,
    seed: int,
    trajectories: int = 0,
    checkpoint_every: int = 10,
    backend: str = "mps",
) -> dict[str, object]:
    """Run public hybrid training in exactly resumable constant-rate chunks.

    Noise iteration counters are global within this block, including after a
    checkpoint resume. Sampling is refreshed every update, never every chunk.

    Returns:
        Final parameters, post-update traces, phase counts, and elapsed time.

    Raises:
        ValueError: If a budget or retained checkpoint is incompatible.
        FloatingPointError: If parameters or training diagnostics become invalid.
    """
    if (
        type(updates) is not int
        or updates < 1
        or type(checkpoint_every) is not int
        or checkpoint_every < 1
        or type(trajectories) is not int
        or trajectories < 0
        or type(seed) is not int
        or seed < 0
        or not 0 <= schedule.online_updates < updates
        or not np.isfinite(batch_rate)
        or batch_rate <= 0
        or backend not in {"mps", "statevector"}
    ):
        msg = "Invalid hybrid block budget; a nonempty batch tail is required."
        raise ValueError(msg)
    signature = pilot.digest({
        "version": VERSION,
        "circuit": pilot.circuit_record(circuit),
        "initial": initial.tolist(),
        "target": hashlib.sha256(target.tobytes()).hexdigest(),
        "updates": updates,
        "batch_rate": batch_rate,
        "schedule": asdict(schedule),
        "seed": seed,
        "trajectories": trajectories,
        "provider": ExactLocalPauliProvider().scientific_dict(),
        "backend": backend,
    })
    state: dict[str, object] = {
        "signature": signature,
        "theta": initial.tolist(),
        "completed_updates": 0,
        "trace": [],
        "wall_seconds": 0.0,
        "executed_chunks": 0,
    }
    if path.exists():
        state = pilot.read_record(path)
        if state["signature"] != signature:
            msg = f"Hybrid checkpoint configuration mismatch: {path}."
            raise ValueError(msg)
    completed = cast("int", state["completed_updates"])
    trace = cast("list[dict[str, object]]", state["trace"])
    if type(completed) is not int or not 0 <= completed <= updates or len(trace) != completed:
        msg = "Invalid hybrid checkpoint progress."
        raise ValueError(msg)
    theta = np.asarray(state["theta"], dtype=np.float64)
    prior_seconds = cast("float", state["wall_seconds"])
    started = time.perf_counter()
    while completed < updates:
        count = min(checkpoint_every, updates - completed)
        online = max(0, min(count, schedule.online_updates - completed))
        options = KrotovOptions(
            max_iterations=count,
            switch_iteration=online,
            online_step_size=schedule.online_rate,
            batch_step_size=batch_rate,
            online_schedule="constant",
            batch_schedule="constant",
            seed=completed,
            truncation=KrotovTruncation(),
        )
        if backend == "statevector":
            trained = statevector_chunk(circuit, theta, target, options, trajectories, seed)
        elif trajectories:
            trained = train_krotov_noisy_state_preparation_hybrid(
                circuit,
                target,
                None,
                KrotovTJMOptions(
                    num_trajectories=trajectories, random_seed=seed, trajectory_update="independent", use_crn=False
                ),
                initial_theta=theta,
                options=options,
                noise_provider=ExactLocalPauliProvider(),
            )
        else:
            trained = train_krotov_state_preparation_hybrid(circuit, target, initial_theta=theta, options=options)
        if not np.all(np.isfinite(trained.theta)):
            msg = "Hybrid parameters are not finite."
            raise FloatingPointError(msg)
        for index in range(1, count + 1):
            row: dict[str, object] = {key: values[index] for key, values in trained.trace.items()}
            row["step"] = completed + index
            for key in ("loss", "fidelity", "step_size", "gradient_norm", "update_norm"):
                if not np.isfinite(cast("float", row[key])):
                    msg = f"Nonfinite hybrid diagnostic: {key}."
                    raise FloatingPointError(msg)
            if not -1e-10 <= cast("float", row["fidelity"]) <= 1 + 1e-10:
                msg = "Hybrid training fidelity is outside physical bounds."
                raise FloatingPointError(msg)
            trace.append(row)
        theta = trained.theta.copy()
        completed += count
        state.update({
            "theta": theta.tolist(),
            "completed_updates": completed,
            "trace": trace,
            "wall_seconds": prior_seconds + time.perf_counter() - started,
            "executed_chunks": cast("int", state["executed_chunks"]) + 1,
        })
        pilot.write_record(path, state)
    return state


@dataclass(frozen=True)
class HybridJob:
    """A paired target/seed/schedule job with private checkpoint ownership."""

    root: Path
    source_root: Path
    config: pilot.PilotConfig
    target_id: str
    replica: int
    schedule: Schedule
    include_warm: bool = True

    @property
    def directory(self) -> Path:
        """Directory unique to the schedule, target, and replica."""
        return self.root / self.schedule.label / self.target_id / f"seed_{self.replica}"

    def train(
        self,
        name: str,
        circuit: ParameterizedCircuit,
        theta: NDArray[np.float64],
        target: NDArray[np.complex128],
        updates: int,
        rate: float,
        *,
        hybrid: bool = False,
        noisy: bool = False,
    ) -> NDArray[np.float64]:
        """Train a block while preserving the original pilot's paired seed keys.

        Returns:
            The final block parameters.
        """
        seed_name = name.removeprefix("warm_")
        result = optimize_block(
            self.directory / "blocks" / f"{name}.json",
            circuit,
            theta,
            target,
            updates=updates,
            batch_rate=rate,
            schedule=self.schedule if hybrid else Schedule(),
            seed=pilot.pilot_seed(self.config, "training", self.target_id, self.replica, seed_name),
            trajectories=self.config.training_trajectories if noisy else 0,
        )
        return np.asarray(result["theta"], dtype=np.float64)


def run_job(job: HybridJob) -> dict[str, object]:
    """Compare early hybrid construction and same-start hybrid refinement.

    Returns:
        Two construction endpoints, optional warm endpoints, and source checksums.

    Raises:
        ValueError: If the batch control fails to reproduce the original pilot.
    """
    path = job.directory / "result.json"
    if path.exists():
        return pilot.read_record(path)
    started = time.perf_counter()
    source = job.source_root / job.target_id / f"seed_{job.replica}"
    construction = pilot.read_record(source / "construction.json")
    previous_refinement = pilot.read_record(source / "refinement.json")
    original = {row["route"]: row for row in cast("list[dict[str, object]]", construction["candidates"])}
    original.update({row["route"]: row for row in cast("list[dict[str, object]]", previous_refinement["candidates"])})
    target = load_target(6, job.target_id).state_vector_copy()
    master = np.random.default_rng(pilot.pilot_seed(job.config, "initialization", job.target_id, job.replica)).normal(
        size=pilot.pilot_circuit(4).num_params
    )
    theta = 0.1 * master[: pilot.pilot_circuit(1).num_params]
    for depth in range(1, 5):
        circuit = pilot.pilot_circuit(depth)
        if theta.size < circuit.num_params:
            theta = np.concatenate((theta, 0.001 * master[theta.size : circuit.num_params]))
        theta = job.train(f"source_{depth}", circuit, theta, target, job.config.source_updates, 0.2, hybrid=True)
    events = []
    for index in range(6):
        before = pilot.state_preparation_metrics(circuit, theta, target)[1]
        scores = pruning_scores(circuit, theta, target, "M")
        removed = select_deletions(scores, 6)
        origins = [circuit.gates[k].logical_gate_id for k in removed]
        circuit, theta = delete_gates(circuit, theta, removed)
        after = pilot.state_preparation_metrics(circuit, theta, target)[1]
        theta = job.train(f"M_repair_{index}", circuit, theta, target, job.config.repair_updates, 0.1)
        events.append({
            "round": index + 1,
            "scores": scores,
            "removed_origins": origins,
            "before_fidelity": before,
            "after_deletion_fidelity": after,
            "after_repair_fidelity": pilot.state_preparation_metrics(circuit, theta, target)[1],
        })
    candidates = [pilot.candidate(circuit, theta, target, "cold_construct")]
    theta = job.train("refine_M_0", circuit, theta, target, job.config.refinement_updates, 0.1)
    candidates.append(pilot.candidate(circuit, theta, target, "cold_final"))
    if job.include_warm:
        warm = original["M"]
        warm_circuit = pilot.restore_circuit(cast("dict[str, object]", warm["circuit"]))
        warm_theta = np.asarray(warm["theta"], dtype=np.float64)
        for arm in ("0", "N"):
            theta = job.train(
                f"warm_refine_M_{arm}",
                warm_circuit,
                warm_theta.copy(),
                target,
                job.config.refinement_updates,
                0.1,
                hybrid=True,
                noisy=arm == "N",
            )
            candidate = pilot.candidate(warm_circuit, theta, target, f"warm_{arm}")
            candidate["initial_theta_checksum"] = pilot.digest(warm_theta.tolist())
            candidates.append(candidate)
    reproduction = {}
    if job.schedule.online_updates == 0:
        reference_keys = ("M", "M_0", "M_0", "M_N") if job.include_warm else ("M", "M_0")
        for candidate, key in zip(candidates, reference_keys, strict=True):
            reference = original[key]
            deviation = float(np.max(np.abs(np.asarray(candidate["theta"]) - np.asarray(reference["theta"]))))
            if (
                deviation > 1e-11
                or pilot.digest(candidate["circuit"]) != pilot.digest(reference["circuit"])
                or abs(
                    cast("float", candidate["exact_noisy_fidelity"]) - cast("float", reference["exact_noisy_fidelity"])
                )
                > 1e-11
            ):
                msg = f"Batch control did not reproduce {job.target_id}/{job.replica}/{key}."
                raise ValueError(msg)
            reproduction[cast("str", candidate["route"])] = deviation
    for candidate in candidates:
        candidate["route"] = f"{job.schedule.label}:{candidate['route']}"
    result: dict[str, object] = {
        "target_id": job.target_id,
        "replica": job.replica,
        "setting": job.schedule.label,
        "schedule": asdict(job.schedule),
        "include_warm": job.include_warm,
        "candidates": candidates,
        "pruning_events": events,
        "batch_reproduction_max_parameter_errors": reproduction,
        "source_construction_checksum": pilot.digest(construction),
        "source_refinement_checksum": pilot.digest(previous_refinement),
        "wall_seconds": time.perf_counter() - started,
        "worker_pid": os.getpid(),
    }
    pilot.write_record(path, result)
    return result


def run_screen(
    root: Path,
    source_root: Path,
    *,
    config: pilot.PilotConfig | None = None,
    schedules: tuple[Schedule, ...] = SCHEDULES,
    workers: WorkerSetting = "auto",
    resume: bool = False,
    include_warm: bool = True,
) -> dict[str, object]:
    """Run the fully specified hybrid screen without overwriting the batch pilot.

    Returns:
        All target-balanced descriptive endpoint aggregates, without selection.

    Raises:
        ValueError: If source, schedules, or resume metadata are incompatible.
    """
    config = pilot.PilotConfig() if config is None else config
    resolve_workers(workers, len(pilot.TARGETS) * len(config.seeds) * len(schedules))
    if (
        not schedules
        or (include_warm and schedules[0] != Schedule())
        or len({s.label for s in schedules}) != len(schedules)
        or any(
            s.online_updates
            >= (min(config.source_updates, config.refinement_updates) if include_warm else config.source_updates)
            for s in schedules
        )
    ):
        msg = "Schedules must be unique and retain batch tails; the full screen must begin with batch."
        raise ValueError(msg)
    source_manifest = pilot.read_record(source_root / "manifest.json")
    source_result = pilot.read_record(source_root / "result.json")
    source_config = json.loads(json.dumps(asdict(config)))
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "qiskit": version("qiskit"),
    }
    if (
        source_result["status"] != "complete"
        or source_result["nominee"] != "M"
        or source_result["manifest_checksum"] != pilot.digest(source_manifest)
        or source_manifest["version"] != pilot.VERSION
        or source_manifest["config"] != source_config
        or source_manifest["targets"] != list(pilot.TARGETS)
        or source_manifest["target_fixture_checksum"] != load_target_collection().checksum
        or source_manifest["noise"] != ExactLocalPauliProvider().scientific_dict()
        or source_manifest["source_fingerprint"] != pilot.source_fingerprint()
        or source_manifest["environment"] != environment
    ):
        msg = "Hybrid source must be the completed compatible magnitude-pruning pilot."
        raise ValueError(msg)
    sources = {}
    for target in pilot.TARGETS:
        for seed in config.seeds:
            for name in ("construction.json", "refinement.json"):
                relative = Path(target) / f"seed_{seed}" / name
                sources[str(relative)] = pilot.digest(pilot.read_record(source_root / relative))
    manifest = {
        "version": VERSION if include_warm else CONSTRUCTION_VERSION,
        "config": source_config,
        "schedules": [asdict(s) for s in schedules],
        "targets": list(pilot.TARGETS),
        "source_manifest_checksum": pilot.digest(source_manifest),
        "source_records": sources,
        "source_root": str(source_root.resolve()),
        "source_fingerprint": pilot.digest({
            "pilot": pilot.source_fingerprint(),
            "hybrid": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }),
        "environment": environment,
        "noise": ExactLocalPauliProvider().scientific_dict(),
        "cold": "hybrid at each growth stage; batch-only pruning repair and final continuation",
        "warm": "identical original M construction checkpoint for all schedules; 0/N paired arms"
        if include_warm
        else "omitted",
        "selection": "none; all final iterates and all settings reported",
        "sampling": (
            "independent R=8 by default; refreshed each update; paired across warm schedules"
            if include_warm
            else "noiseless MPS training; exact noisy endpoint evaluation only"
        ),
    }
    if root.exists():
        if not resume or not (root / "manifest.json").exists() or pilot.read_record(root / "manifest.json") != manifest:
            msg = "Existing hybrid output requires explicit resume with an identical manifest."
            raise ValueError(msg)
    else:
        if resume:
            msg = "Cannot resume nonexistent hybrid output."
            raise ValueError(msg)
        root.mkdir(parents=True)
        pilot.write_record(root / "manifest.json", manifest)
    if (root / "result.json").exists():
        return pilot.read_record(root / "result.json")
    jobs = [
        HybridJob(root, source_root, config, target, seed, setting, include_warm=include_warm)
        for setting in schedules
        for target in pilot.TARGETS
        for seed in config.seeds
    ]
    started = time.perf_counter()
    results = {}
    for index, record in completed_jobs(run_job, jobs, workers=workers):
        results[index] = record
        print(
            f"hybrid screen {len(results)}/{len(jobs)}: {jobs[index].schedule.label} "
            f"{jobs[index].target_id} seed {jobs[index].replica}",
            flush=True,
        )
    ordered = [results[index] for index in range(len(jobs))]
    result: dict[str, object] = {
        "status": "complete",
        "version": manifest["version"],
        "scope": "exploratory; no held-out confirmation or selected winner",
        "manifest_checksum": pilot.digest(manifest),
        "completed_jobs": len(jobs),
        "aggregate": pilot.aggregate(ordered),
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "artifacts": [str(job.directory.relative_to(root)) for job in jobs],
        "worker_pids": sorted({cast("int", item["worker_pid"]) for item in ordered}),
    }
    pilot.write_record(root / "result.json", result)
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the default ten-setting matched hybrid screen.

    Returns:
        Zero on completion, two on a retained and reported failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--workers", type=parse_workers, default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--construction-extension",
        action="store_true",
        help="Only add 25/30 online updates at rates 0.7/0.9; omit warm refinements and old settings.",
    )
    args = parser.parse_args(arguments)
    try:
        result = run_screen(
            args.output_root,
            args.source_root,
            workers=args.workers,
            resume=args.resume,
            schedules=EXTENSION_SCHEDULES if args.construction_extension else SCHEDULES,
            include_warm=not args.construction_extension,
        )
    except Exception:
        LOGGER.exception("Hybrid screen stopped; checkpoints retained for diagnosis and compatible resume")
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
