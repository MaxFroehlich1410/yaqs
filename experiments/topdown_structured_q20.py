# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared-preparation B/G twenty-qubit pilot with bounded, instrumented MPO assessment."""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from benchmarks.state_preparation import load_target
from experiments import legacy_topdown_dense as legacy
from experiments import topdown_backend_support as storage
from experiments import topdown_batch_refinement_policies as policies
from experiments import topdown_guarded_pruning as guarded
from experiments import topdown_mpo_evaluation as mpo
from experiments import topdown_product_screening as screening
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import KrotovTJMOptions, MPSBackend
from mqt.yaqs.optimization import create_brickwall_matrix_product_disentangler_parameterized_circuit as brickwall
from mqt.yaqs.optimization.mps_backend import mps_overlap

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit
    from mqt.yaqs.optimization.trajectories import NoiseMaps

VERSION = "topdown.structured_q20_mpo.v2"
GIB = 1024**3
TARGETS = (
    "gaussian_mu0p5_sigma0p1",
    "tfim_ferro",
    "tfim_critical",
    "tfim_para",
    "random_mps_bond2",
    "random_mps_bond3",
)
HOST_TARGETS = {"leonhard-13": TARGETS[::2], "leonhard-16": TARGETS[1::2]}
STAGES = ("prepare", "B", "B_pruned_mpo", "B_selected_mpo", "G", "G_pruned_mpo", "G_selected_mpo")
THREAD_ENV = dict.fromkeys(
    (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "QISKIT_NUM_PROCS",
    ),
    "1",
)


@dataclass(frozen=True)
class Settings:
    """Frozen pilot settings, with small widths/budgets available only for regression tests."""

    qubits: int = 20
    seed: int = 300
    max_depth: int = 30
    ideal_online: int = 100
    ideal_batch: int = 2000
    ideal_fidelity: float = 0.99
    preparation_cpu_seconds: float = 14 * 3600
    product_candidates: int = 1000
    product_batch_size: int = 8
    training_paths: int = 100
    noisy_online: int = 100
    b_batch: int = 100
    rate: float = 0.2
    hardware: str = "IBM"
    target_split_tolerance: float = 5e-13
    target_vector_error_limit: float = 1e-12
    mpo_workspace_bytes: int = 32 * GIB

    def __post_init__(self) -> None:
        """Reject malformed pilot budgets before any numerical work.

        Raises:
            ValueError: If the profile, counts, or numerical limits are invalid.
        """
        integers = (
            self.qubits,
            self.max_depth,
            self.ideal_online,
            self.ideal_batch,
            self.product_candidates,
            self.product_batch_size,
            self.training_paths,
            self.noisy_online,
            self.b_batch,
            self.mpo_workspace_bytes,
        )
        if (
            any(type(value) is not int or value < 1 for value in integers)
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**32 - self.max_depth - 1001
            or self.hardware != "IBM"
            or not 0 < self.ideal_fidelity < 1
            or any(
                not np.isfinite(value) or value <= 0
                for value in (
                    self.rate,
                    self.preparation_cpu_seconds,
                    self.target_split_tolerance,
                    self.target_vector_error_limit,
                )
            )
        ):
            msg = "Invalid structured scaling pilot settings."
            raise ValueError(msg)


def resources() -> dict:
    """Return process CPU and actual OS-reported lifetime peak RSS."""
    return {
        "process_cpu_seconds": time.process_time(),
        "wall_unix": time.time(),
        "process_peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        ),
    }


def compact_target(vector: NDArray, tolerance: float) -> tuple[MPS, dict]:
    """Factor a fixed target with a bounded absolute TT-SVD tail budget.

    Returns:
        Normalized MPS and full-vector verification diagnostics.
    """
    count = vector.size.bit_length() - 1
    current = np.asarray(vector).reshape((2,) * count)
    tensors = []
    left = 1
    discarded = 0.0
    for _ in range(count - 1):
        u, singular, vh = np.linalg.svd(current.reshape(2 * left, -1), full_matrices=False)
        tails = np.cumsum(singular[::-1] ** 2)[::-1]
        allowed = np.flatnonzero(tails <= (tolerance / (count - 1)) ** 2)
        keep = max(1, int(allowed[0])) if len(allowed) else len(singular)
        discarded += float(np.linalg.norm(singular[keep:]))
        tensors.append(u[:, :keep].reshape(left, 2, keep).transpose(1, 0, 2))
        current = singular[:keep, None] * vh[:keep]
        left = keep
    tensors.append(current.reshape(left, 2, 1).transpose(1, 0, 2))
    target = MPS(count, tensors=[tensor.transpose(0, 2, 1).copy() for tensor in reversed(tensors)])
    target.tensors[0] /= np.sqrt(mps_overlap(target, target).real)
    error = float(np.linalg.norm(target.to_vec() - vector))
    return target, {
        "absolute_split_tail_budget_sum": tolerance,
        "discarded_norm_sum": discarded,
        "verified_full_vector_error_norm": error,
        "pure_target_fidelity_error_bound": 2 * error,
        "maximum_bond": max(max(t.shape[1:]) for t in target.tensors),
        "phase_aligned": False,
    }


def get_target(folder: Path) -> tuple[MPS, dict]:
    """Load the shared, sealed target representation without dense reconstruction."""
    saved = storage.load_checkpoint(folder / "target")
    if saved is None:
        msg = "Shared target representation is missing."
        raise ValueError(msg)
    record, arrays = saved
    return MPS(record["qubits"], tensors=[arrays[f"site_{site}"] for site in range(record["qubits"])]), record


class ProgressBackend(MPSBackend):
    """Keep a live resource/fidelity record while retaining the original numerical kernels."""

    def __init__(self, circuit: ParameterizedCircuit, target: MPS, path: Path) -> None:
        """Bind progress reporting to a private stage directory."""
        super().__init__(circuit, target)
        self.path = path
        self.calls = 0

    def metrics(
        self, theta: NDArray, maps: NoiseMaps | None = None, *, weights: NDArray | None = None
    ) -> tuple[float, float]:
        """Record completed objective calls without changing their result."""
        result = super().metrics(theta, maps, weights=weights)
        self.calls += 1
        storage.write_json(
            self.path / "progress.json",
            {"calls": self.calls, "fidelity": result[1], **resources(), "diagnostics": self.diagnostics.record()},
        )
        return result


def prepare(folder: Path, target_id: str, settings: Settings) -> None:
    """Prepare one ideal parent shared by both policies and seal its target representation."""
    cpu_start = time.process_time()
    original = load_target(settings.qubits, target_id)
    dense = original.state_vector
    if target_id.startswith("random_mps_") and settings.qubits >= 20:
        target = original.to_mps()
        error = float(np.linalg.norm(target.to_vec() - dense))
        approximation: dict = {
            "verified_full_vector_error_norm": error,
            "pure_target_fidelity_error_bound": 2 * error,
            "representation": "native random MPS",
            "maximum_bond": max(max(t.shape[1:]) for t in target.tensors),
        }
    else:
        target, approximation = compact_target(dense, settings.target_split_tolerance)
    if approximation["verified_full_vector_error_norm"] > settings.target_vector_error_limit:
        msg = "Target representation exceeds its fixed vector-error limit."
        raise ValueError(msg)
    storage.save_checkpoint(
        folder / "target",
        {"qubits": settings.qubits, "target_id": target_id, "approximation": approximation},
        {f"site_{site}": tensor for site, tensor in enumerate(target.tensors)},
    )
    metadata, arrays = screening.initial_parameters(
        dense, settings.seed, "screened_product", settings.product_candidates, settings.product_batch_size
    )
    storage.save_checkpoint(folder / "initialization", metadata, arrays)
    optimize_preparation(folder, target, arrays["theta"], settings, started_cpu=cpu_start)


def optimize_preparation(
    folder: Path,
    target: MPS,
    theta: NDArray,
    settings: Settings,
    *,
    backend_factory: Callable[[ParameterizedCircuit, MPS, Path], MPSBackend] = ProgressBackend,
    started_cpu: float | None = None,
) -> None:
    """Optimize an initialized native target using the existing depth/retention policy.

    Args:
        folder: Private job checkpoint directory.
        target: Normalized target MPS, never expanded to a vector here.
        theta: Complete depth-one initialization from the selected product state.
        settings: Existing preparation budgets and stopping thresholds.
        backend_factory: Explicit opt-in propagation backend; archived defaults stay exact.
        started_cpu: Earlier setup start so the archived preparation budget includes setup.
    """
    cpu_start = time.process_time() if started_cpu is None else started_cpu
    best, selected = -1.0, None
    rows = []
    for depth in range(1, settings.max_depth + 1):
        stage = folder / "preparation" / f"depth_{depth:03d}"
        circuit = brickwall(settings.qubits, depth, initial_single_qubit_layer=True)
        if depth > 1:
            theta = legacy.extend(theta, circuit.num_params, settings.seed + 1000 + depth, 0.001)
        backend = backend_factory(circuit, target, stage)
        theta, trace = policies.train_noiseless(
            circuit,
            theta,
            target,
            policy="lbfgs",
            steps=settings.ideal_online + settings.ideal_batch,
            online=settings.ideal_online,
            rate=0.7,
            stop=settings.ideal_fidelity,
            engine=backend,
        )
        fidelity = backend.metrics(theta)[1]
        stored_arrays = {"theta": theta}
        for key in (
            "initial_theta",
            "online_final_theta",
            "best_theta",
            "final_theta",
            "returned_theta",
            "fidelity_trace",
        ):
            stored_arrays[key] = np.asarray(trace.pop(key))
        storage.save_checkpoint(
            stage,
            {
                "depth": depth,
                "fidelity": fidelity,
                "training": trace,
                "circuit": storage.circuit_record(circuit),
                "resources": resources(),
            },
            stored_arrays,
        )
        rows.append({"depth": depth, "fidelity": fidelity, "checkpoint": str(stage.relative_to(folder))})
        if fidelity > best:
            best, selected = fidelity, stage
        storage.write_json(
            folder / "preparation-progress.json", {"rows": rows, "selected": str(selected), **resources()}
        )
        if fidelity >= settings.ideal_fidelity or time.process_time() - cpu_start >= settings.preparation_cpu_seconds:
            break
    assert selected is not None
    saved = storage.load_checkpoint(selected)
    assert saved is not None
    storage.save_checkpoint(
        folder / "parent",
        {
            "circuit": saved[0]["circuit"],
            "fidelity": best,
            "shared_policies": ["B", "G"],
            "depths": rows,
            "selected_seal": storage.digest(selected / "complete.json"),
            "target_fidelity_reached": best >= settings.ideal_fidelity,
        },
        {"theta": saved[1]["theta"]},
    )


def continuation_stop(previous: float, current: float, stale: int) -> tuple[str | None, int]:
    """Apply the frozen G continuation decline/plateau rules.

    Returns:
        Stop reason and updated consecutive-small-gain counter.
    """
    gain = current - previous
    stale = stale + 1 if gain < 1e-5 else 0
    return ("decline" if gain < -1e-12 else "plateau" if stale >= 5 else None), stale


def train_branch(
    folder: Path,
    policy: str,
    settings: Settings,
    *,
    backend_factory: Callable[[ParameterizedCircuit, MPS, Path], MPSBackend] = ProgressBackend,
    pruning_backend_factory: Callable[..., MPSBackend] = MPSBackend,
) -> None:
    """Prune the shared parent and run the policy's unchanged fixed-bank updates."""
    target, _ = get_target(folder)
    parent = storage.load_checkpoint(folder / "parent")
    assert parent is not None
    circuit = storage.restore_circuit(parent[0]["circuit"])
    initial = parent[1]["theta"]
    if policy == "B":
        circuit, theta, trace = storage.prune_b(pruning_backend_factory(circuit, target), initial)
    else:
        circuit, theta, trace = guarded.prune(circuit, initial, target, backend_factory=pruning_backend_factory)
    branch = folder / policy
    storage.save_checkpoint(
        branch / "pruning",
        {
            "policy": policy,
            "circuit": storage.circuit_record(circuit),
            "pruning": trace,
            "parent_seal": storage.digest(folder / "parent/complete.json"),
            **resources(),
        },
        {"theta": theta},
    )
    backend = backend_factory(circuit, target, branch)
    options = KrotovTJMOptions(num_trajectories=settings.training_paths, trajectory_update="cross", use_crn=True)
    raw = backend.sample_noise_maps(
        theta, options, legacy.LegacyNoise(settings.hardware), iteration=settings.seed + 5000
    )
    maps, weights = backend.compress_no_jump_maps(raw)
    jumps = [sum(bool(g.operators) for g in row) for row in raw]
    if any(g.noise_enabled for g in circuit.gates) and not any(jumps):
        msg = "Training bank contains no jumps; stopped without resampling."
        raise RuntimeError(msg)
    events = [
        [
            [
                {"sites": list(sites), "real": operator.real.tolist(), "imag": operator.imag.tolist()}
                for operator, sites in gate.operators
            ]
            for gate in row
        ]
        for row in raw
    ]
    storage.write_json(
        branch / "noise-bank.json",
        {
            "raw_paths": settings.training_paths,
            "jump_counts": jumps,
            "weights": weights.tolist(),
            "retained_paths": len(maps),
            "raw_events": events,
        },
    )
    best = previous = float(backend.metrics(theta, maps, weights=weights)[1])
    best_step = 0
    history = []
    kernel_cpu = initial_cpu = extra_cpu = 0.0
    stale = 0
    stop = "fixed_budget"
    step = 0
    while True:
        checkpoint = branch / "iterates" / f"step_{step:05d}"
        storage.save_checkpoint(
            checkpoint, {"step": step, "fidelity": previous, "kernel_cpu_seconds": kernel_cpu}, {"theta": theta}
        )
        history.append({"step": step, "fidelity": previous, "kernel_cpu_seconds": kernel_cpu})
        if previous > best:
            best, best_step = previous, step
        storage.write_json(
            branch / "training-progress.json",
            {
                "step": step,
                "fidelity": previous,
                "best_fidelity": best,
                "best_step": best_step,
                "initial_online_cpu_seconds": initial_cpu,
                "extra_online_cpu_seconds": extra_cpu,
                **resources(),
            },
        )
        if policy == "B" and step >= settings.noisy_online + settings.b_batch:
            break
        if policy == "G" and step >= settings.noisy_online and extra_cpu >= initial_cpu:
            stop = "extra_cpu_budget"
            break
        started = time.process_time()
        if policy == "G" or step < settings.noisy_online:
            theta, _ = backend.online_update(theta, settings.rate, maps, estimator="cross", weights=weights)
        else:
            signal_value, _, _ = backend.gradient(theta, maps, estimator="cross", weights=weights)
            theta -= settings.rate * signal_value
        value = float(backend.metrics(theta, maps, weights=weights)[1])
        elapsed = time.process_time() - started
        kernel_cpu += elapsed
        if step < settings.noisy_online:
            initial_cpu += elapsed
        else:
            extra_cpu += elapsed
        step += 1
        if policy == "G" and step > settings.noisy_online:
            stop_reason, stale = continuation_stop(previous, value, stale)
            if stop_reason:
                stop = stop_reason
                # Persist the complete stopping update, including its charged CPU.
                storage.save_checkpoint(
                    branch / "iterates" / f"step_{step:05d}",
                    {"step": step, "fidelity": value, "kernel_cpu_seconds": kernel_cpu},
                    {"theta": theta},
                )
                history.append({"step": step, "fidelity": value, "kernel_cpu_seconds": kernel_cpu})
                if value > best:
                    best, best_step = value, step
                break
        previous = value
    selected = branch / "iterates" / f"step_{best_step:05d}"
    payload = storage.load_checkpoint(selected)
    assert payload is not None
    storage.save_checkpoint(
        branch / "selection",
        {
            "policy": policy,
            "selected_step": best_step,
            "training_fidelity": best,
            "kernel_cpu_seconds": kernel_cpu,
            "initial_online_cpu_seconds": initial_cpu,
            "extra_online_cpu_seconds": extra_cpu,
            "stop": stop,
            "history": history,
            "training_bank_sha256": storage.digest(branch / "noise-bank.json"),
            "pruning_seal": storage.digest(branch / "pruning/complete.json"),
        },
        {"theta": payload[1]["theta"]},
    )


def assess(folder: Path, policy: str, endpoint: str, settings: Settings) -> None:
    """Run an untruncated final MPO with a durable per-gate bond/resource history."""
    target, representation = get_target(folder)
    pruned = storage.load_checkpoint(folder / policy / "pruning")
    saved = storage.load_checkpoint(folder / policy / ("pruning" if endpoint == "pruned" else "selection"))
    assert pruned is not None
    assert saved is not None
    path = folder / policy / f"mpo-{endpoint}"
    path.mkdir(parents=True, exist_ok=True)
    with (path / "bond-growth.jsonl").open("x", encoding="utf-8") as stream:

        def progress(event: dict) -> None:
            event.update(resources())
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()

        report = mpo.evaluate(
            storage.restore_circuit(pruned[0]["circuit"]),
            saved[1]["theta"],
            target,
            settings.hardware,
            mpo.MPOSettings(max_workspace_bytes=settings.mpo_workspace_bytes),
            progress=progress,
        )
    report.update(
        target_representation=representation["approximation"],
        process_resources=resources(),
        circuit_seal=storage.digest(folder / policy / "pruning/complete.json"),
        endpoint_seal=storage.digest(
            folder / policy / ("pruning" if endpoint == "pruned" else "selection") / "complete.json"
        ),
    )
    target_error = representation["approximation"]["pure_target_fidelity_error_bound"]
    report["fixed_target_absolute_error_bound_excluding_roundoff"] = (
        target_error + report["diagnostics"]["fidelity_absolute_error_bound_excluding_roundoff"]
    )
    storage.save_checkpoint(path, report, {})


def verify(root: Path) -> dict:
    """Authenticate the frozen source/target snapshot before launching numerical work.

    Returns:
        The frozen pilot plan.

    Raises:
        ValueError: If any source or target file changed.
    """
    plan = json.loads((root / "plan.json").read_text())
    source = Path(__file__).resolve().parents[1]
    if plan["version"] != VERSION:
        msg = "Pilot protocol version differs."
        raise ValueError(msg)
    for name, checksum in plan["files"].items():
        if storage.digest(source / name) != checksum:
            msg = f"Frozen source/target changed: {name}."
            raise ValueError(msg)
    return plan


def execute_stage(folder: Path, target_id: str, stage: str, settings: Settings) -> None:
    """Dispatch one stage using a single numerical thread."""
    with threadpool_limits(limits=1):
        if stage == "prepare":
            prepare(folder, target_id, settings)
        elif stage in {"B", "G"}:
            train_branch(folder, stage, settings)
        else:
            policy, endpoint, _ = stage.split("_")
            assess(folder, policy, endpoint, settings)


def leaf(root: Path, target_id: str, stage: str) -> None:
    """Execute a single fresh-process stage and retain success/failure resource receipts."""
    plan = verify(root)
    settings = Settings(**plan["settings"])
    folder = root / "jobs" / target_id
    started = time.perf_counter()
    result: dict = {"target_id": target_id, "stage": stage, "host": socket.gethostname(), "pid": os.getpid()}
    try:
        execute_stage(folder, target_id, stage, settings)
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result.update(resources(), stage_wall_seconds=time.perf_counter() - started)
        storage.write_json(folder / "receipts" / f"{stage}.json", result)


def run_stage(
    root: Path,
    target_id: str,
    stage: str,
    host: str,
    plan: dict,
    *,
    leaf_module: str = "experiments.topdown_structured_q20",
) -> tuple[int, dict]:
    """Run and supervise one owned leaf, leaving every unrelated process untouched."""
    command = [
        sys.executable,
        "-u",
        "-m",
        leaf_module,
        "leaf",
        "--root",
        str(root),
        "--target-id",
        target_id,
        "--stage",
        stage,
    ]
    directory = root / "jobs" / target_id / "supervision"
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    peak = 0
    reason = None
    with (directory / f"{stage}.log").open("x", encoding="utf-8") as log:
        # The executable is the current interpreter; the sealed plan supplies argv without a shell.
        child = subprocess.Popen(  # noqa: S603
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **THREAD_ENV},
            start_new_session=True,
        )
        process = psutil.Process(child.pid)
        created = process.create_time()
        while child.poll() is None:
            try:
                rss = sum(p.memory_info().rss for p in [process, *process.children(recursive=True)] if p.is_running())
            except psutil.NoSuchProcess:
                rss = 0
            peak = max(peak, rss)
            elapsed = time.time() - started
            storage.write_json(
                root / f"status-{host}.json",
                {
                    "status": "running",
                    "target_id": target_id,
                    "stage": stage,
                    "pid": child.pid,
                    "rss_bytes": rss,
                    "sampled_tree_peak_rss_bytes": peak,
                    "wall_seconds": elapsed,
                },
            )
            if rss > plan["memory_guard_gib"] * GIB or elapsed > plan["stage_wall_seconds"]:
                reason = "memory_guard" if rss > plan["memory_guard_gib"] * GIB else "wall_guard"
                if process.is_running() and process.create_time() == created and os.getpgid(child.pid) == child.pid:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(1)
        code = child.wait()
    storage.write_json(
        directory / f"{stage}.json",
        {
            "status": "complete" if code == 0 else "failed",
            "returncode": code,
            "guard_reason": reason,
            "sampled_tree_peak_rss_bytes": peak,
            "wall_seconds": time.time() - started,
            "command": command,
            "pid": child.pid,
            "created": created,
        },
    )
    return code, {"sampled_peak_rss_bytes": peak, "wall_seconds": time.time() - started}


def supervise(root: Path, host: str) -> None:
    """Sequentially run this host's disjoint targets, preserving failures and guarding only owned leaves."""
    plan = verify(root)
    if host not in HOST_TARGETS or socket.gethostname().split(".")[0] != host:
        msg = "Pilot host assignment differs."
        raise ValueError(msg)
    claim = root / "claims" / host
    claim.mkdir(parents=True, exist_ok=False)
    storage.write_json(claim / "owner.json", {"pid": os.getpid(), "host": host, "created": time.time()})
    for target_id in HOST_TARGETS[host]:
        failed = set()
        for stage in STAGES:
            dependency = "prepare" if stage in {"B", "G"} else stage[0] if stage != "prepare" else None
            if dependency in failed:
                failed.add(stage)
                storage.write_json(
                    root / "jobs" / target_id / "supervision" / f"{stage}.json",
                    {"status": "skipped", "dependency": dependency},
                )
                continue
            # Other campaigns are never stopped or preempted to make room.
            while psutil.virtual_memory().available < (plan["memory_guard_gib"] + 64) * GIB:
                storage.write_json(
                    root / f"status-{host}.json",
                    {"status": "waiting_for_memory", "target_id": target_id, "stage": stage},
                )
                time.sleep(30)
            code, _ = run_stage(root, target_id, stage, host, plan)
            if code:
                failed.add(stage)
        storage.write_json(
            root / "jobs" / target_id / "finished.json",
            {"status": "complete" if not failed else "partial", "failed_or_skipped_stages": sorted(failed)},
        )
    storage.write_json(root / f"status-{host}.json", {"status": "complete", "host": host})


def main(argv: list[str] | None = None) -> None:
    """Check or run an explicitly frozen scaling pilot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "leaf", "supervise"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", choices=tuple(HOST_TARGETS))
    parser.add_argument("--target-id", choices=TARGETS)
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args(argv)
    if args.action == "check":
        print(
            json.dumps({"version": VERSION, "verified": bool(verify(args.root)), "host": socket.gethostname()}),
            flush=True,
        )
    elif args.action == "leaf":
        if args.target_id is None or args.stage is None:
            parser.error("leaf requires target and stage")
        leaf(args.root, args.target_id, args.stage)
    else:
        supervise(args.root, args.host)


if __name__ == "__main__":
    main()
