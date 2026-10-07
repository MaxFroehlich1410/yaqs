# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Resumable four-target compression-first pilot; not the full research study."""

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
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from benchmarks.state_preparation.execution import WorkerSetting, completed_jobs, parse_workers, resolve_workers
from benchmarks.state_preparation.pilot_components import (
    ExactLocalPauliProvider,
    delete_gates,
    exact_density_fidelity,
    pruning_scores,
    select_deletions,
    with_origins,
)
from benchmarks.state_preparation.targets import load_target, load_target_collection
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KrotovTJMOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    ParameterizedGate,
    create_brickwall_matrix_product_disentangler_parameterized_circuit,
    noisy_state_preparation_contribution,
    state_preparation_contribution,
    state_preparation_metrics,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

VERSION = "yaqs.compression_first_pilot.v1"
TARGETS = ("gaussian_mu0p5_sigma0p1", "tfim_critical", "haar_random_1", "random_mps_bond2")
ROUTES = ("L2", "L4", "M", "D")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PilotConfig:
    """Fixed scientific budgets; smaller values are for engineering tests only."""

    source_updates: int = 100
    repair_updates: int = 20
    refinement_updates: int = 50
    training_trajectories: int = 8
    seeds: tuple[int, ...] = (11, 29)
    root_seed: int = 2026090520

    def __post_init__(self) -> None:
        """Reject invalid budgets before creating artifacts.

        Raises:
            ValueError: If counts or seeds are invalid.
        """
        counts = (self.source_updates, self.repair_updates, self.refinement_updates, self.training_trajectories)
        if any(type(value) is not int or value < 1 for value in counts):
            msg = "Pilot budgets must be positive integers."
            raise ValueError(msg)
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(type(seed) is not int or seed < 0 for seed in (*self.seeds, self.root_seed))
        ):
            msg = "Pilot seeds must be unique nonnegative integers."
            raise ValueError(msg)


def digest(value: object) -> str:
    """Return the SHA-256 of canonical, finite JSON data."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def pilot_seed(config: PilotConfig, *identity: object) -> int:
    """Derive a seed from scientific identity, never execution ordering."""
    return int(digest([VERSION, config.root_seed, *identity])[:16], 16)


def write_record(path: Path, record: dict[str, object]) -> None:
    """Atomically store a checksummed JSON artifact in a job-owned directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**record, "content_checksum": digest(record)}
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_record(path: Path) -> dict[str, object]:
    """Read and verify a stored artifact.

    Returns:
        The verified record, excluding the checksum envelope.

    Raises:
        ValueError: If the JSON record or its checksum is invalid.
        TypeError: If the artifact is not a JSON object.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Invalid artifact {path}."
        raise TypeError(msg)
    checksum = payload.pop("content_checksum", None)
    if checksum != digest(payload):
        msg = f"Artifact checksum mismatch: {path}."
        raise ValueError(msg)
    return cast("dict[str, object]", payload)


def circuit_record(circuit: ParameterizedCircuit) -> dict[str, object]:
    """Return JSON-safe scalar-circuit parameters and immutable gate origins."""
    return {
        "num_qubits": circuit.num_qubits,
        "num_params": circuit.num_params,
        "gates": [asdict(g) for g in circuit.gates],
    }


def restore_circuit(record: dict[str, object]) -> ParameterizedCircuit:
    """Restore a pilot scalar circuit without changing its gate order.

    Returns:
        The recorded parameterized circuit.
    """
    gates = [
        ParameterizedGate(
            name=cast("str", value["name"]),
            sites=tuple(cast("list[int]", value["sites"])),
            param_index=cast("int", value["param_index"]),
            angle_scale=cast("float", value["angle_scale"]),
            angle_offset=cast("float", value["angle_offset"]),
            logical_gate_id=cast("int", value["logical_gate_id"]),
            noise_enabled=cast("bool", value["noise_enabled"]),
        )
        for value in cast("list[dict[str, object]]", record["gates"])
    ]
    return ParameterizedCircuit(cast("int", record["num_qubits"]), gates, cast("int", record["num_params"]))


def pilot_circuit(depth: int) -> ParameterizedCircuit:
    """Build the canonical q6 BMPD adjoint with protected initial-layer origins."""
    return with_origins(
        create_brickwall_matrix_product_disentangler_parameterized_circuit(6, depth, initial_single_qubit_layer=True)
    )


def optimize(
    path: Path,
    circuit: ParameterizedCircuit,
    initial_theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    *,
    updates: int,
    rate: float,
    seed: int,
    trajectories: int = 0,
) -> dict[str, object]:
    """Run a fixed-budget batch block, saving exact progress every ten updates.

    Returns:
        Final parameters, complete pre-update trace, work counters, and timing.

    Raises:
        ValueError: If retained block settings differ from the requested block.
        FloatingPointError: If trajectory norms, parameters, or gradients fail.
    """
    signature = digest({
        "circuit": circuit_record(circuit),
        "initial_theta": initial_theta.tolist(),
        "target": hashlib.sha256(target.tobytes()).hexdigest(),
        "updates": updates,
        "rate": rate,
        "seed": seed,
        "trajectories": trajectories,
        "provider": ExactLocalPauliProvider().scientific_dict(),
    })
    state: dict[str, object] = {
        "signature": signature,
        "completed_updates": 0,
        "theta": initial_theta.tolist(),
        "trace": [],
        "wall_seconds": 0.0,
        "maximum_norm_drift": 0.0,
    }
    if path.exists():
        state = read_record(path)
        if state["signature"] != signature:
            msg = f"Checkpoint configuration mismatch: {path}."
            raise ValueError(msg)
    completed = cast("int", state["completed_updates"])
    theta = np.asarray(state["theta"], dtype=np.float64).copy()
    trace = cast("list[dict[str, object]]", state["trace"])
    prior_seconds = cast("float", state["wall_seconds"])
    started = time.perf_counter()
    truncation = KrotovTruncation()
    provider = ExactLocalPauliProvider()
    maximum_drift = cast("float", state["maximum_norm_drift"])
    for step in range(completed, updates):
        if trajectories:
            gradient, loss, fidelity, paths = noisy_state_preparation_contribution(
                circuit,
                theta,
                target,
                None,
                KrotovTJMOptions(num_trajectories=trajectories, random_seed=seed, trajectory_update="independent"),
                MPS(6),
                truncation,
                iteration=step + 1,
                noise_provider=provider,
            )
            for trajectory in paths:
                vector = trajectory.states[-1].to_vec()
                drift = abs(float(np.vdot(vector, vector).real) - 1)
                maximum_drift = max(maximum_drift, drift)
                if not np.all(np.isfinite(vector)) or drift > 1e-10:
                    msg = "Pilot trajectory failed its finite/norm checks."
                    raise FloatingPointError(msg)
        else:
            gradient, loss, fidelity = state_preparation_contribution(circuit, theta, target, MPS(6), truncation)
        if not np.all(np.isfinite(gradient)) or not np.isfinite(loss):
            msg = "Pilot gradient or loss is not finite."
            raise FloatingPointError(msg)
        theta -= rate * gradient
        if not np.all(np.isfinite(theta)):
            msg = "Pilot parameters are not finite."
            raise FloatingPointError(msg)
        trace.append({
            "update": step + 1,
            "pre_update_loss": loss,
            "pre_update_fidelity": fidelity,
            "gradient_norm": float(np.linalg.norm(gradient)),
            "learning_rate": rate,
            "update_norm": float(rate * np.linalg.norm(gradient)),
        })
        state.update({
            "theta": theta.tolist(),
            "completed_updates": step + 1,
            "trace": trace,
            "wall_seconds": prior_seconds + time.perf_counter() - started,
            "maximum_norm_drift": maximum_drift,
        })
        if (step + 1) % 10 == 0 or step + 1 == updates:
            write_record(path, state)
    return state


def candidate(
    circuit: ParameterizedCircuit, theta: NDArray[np.float64], target: NDArray[np.complex128], route: str
) -> dict[str, object]:
    """Freeze parameters before evaluating both noiseless and exact noisy fidelity."""
    started = time.perf_counter()
    return {
        "route": route,
        "circuit": circuit_record(circuit),
        "theta": theta.tolist(),
        "one_qubit_gates": sum(len(g.sites) == 1 for g in circuit.gates),
        "two_qubit_gates": sum(len(g.sites) == 2 for g in circuit.gates),
        "noiseless_fidelity": state_preparation_metrics(circuit, theta, target)[1],
        "exact_noisy_fidelity": exact_density_fidelity(circuit, theta, target, ExactLocalPauliProvider()),
        "evaluation_seconds": time.perf_counter() - started,
    }


@dataclass(frozen=True)
class PilotJob:
    """One independently seeded target/replica task and its owned directory."""

    root: Path
    config: PilotConfig
    target_id: str
    replica: int
    nominee: str = ""

    @property
    def directory(self) -> Path:
        """Owned target/replica artifact directory."""
        return self.root / self.target_id / f"seed_{self.replica}"


def _train_block(
    job: PilotJob,
    name: str,
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    updates: int,
    rate: float,
    *,
    noisy: bool = False,
) -> NDArray[np.float64]:
    """Run or restore one block under a role-separated reproducible seed.

    Returns:
        The trained parameter vector.
    """
    result = optimize(
        job.directory / "blocks" / f"{name}.json",
        circuit,
        theta,
        target,
        updates=updates,
        rate=rate,
        seed=pilot_seed(job.config, "training", job.target_id, job.replica, name),
        trajectories=job.config.training_trajectories if noisy else 0,
    )
    return np.asarray(result["theta"], dtype=np.float64)


def _prune(
    job: PilotJob,
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    route: Literal["M", "D"],
) -> dict[str, object]:
    """Delete six fixed-quota batches, repairing after every round.

    Returns:
        The final depth-2-resource candidate and all scored deletion decisions.
    """
    events = []
    for round_index in range(6):
        scoring_started = time.perf_counter()
        scores = pruning_scores(circuit, theta, target, route)
        scoring_seconds = time.perf_counter() - scoring_started
        indices = select_deletions(scores, circuit.num_qubits)
        removed_origins = [circuit.gates[index].logical_gate_id for index in indices]
        before = state_preparation_metrics(circuit, theta, target)[1]
        circuit, theta = delete_gates(circuit, theta, indices)
        after = state_preparation_metrics(circuit, theta, target)[1]
        theta = _train_block(
            job, f"{route}_repair_{round_index}", circuit, theta, target, job.config.repair_updates, 0.1
        )
        events.append({
            "round": round_index + 1,
            "scoring_seconds": scoring_seconds,
            "candidate_deletion_evaluations": len(scores) if route == "D" else 0,
            "scores": scores,
            "removed_origins": removed_origins,
            "before_fidelity": before,
            "after_deletion_fidelity": after,
            "after_repair_fidelity": state_preparation_metrics(circuit, theta, target)[1],
        })
    result = candidate(circuit, theta, target, route)
    result["pruning_events"] = events
    return result


def run_construction(job: PilotJob) -> dict[str, object]:
    """Run all four construction routes from a shared layerwise source.

    Returns:
        The complete construction results, or their previously completed record.
    """
    path = job.directory / "construction.json"
    if path.exists():
        return read_record(path)
    started = time.perf_counter()
    target = load_target(6, job.target_id).state_vector_copy()
    master = np.random.default_rng(pilot_seed(job.config, "initialization", job.target_id, job.replica)).normal(
        size=pilot_circuit(4).num_params
    )
    theta = 0.1 * master[: pilot_circuit(1).num_params]
    depth_two = theta.copy()
    for depth in range(1, 5):
        circuit = pilot_circuit(depth)
        if theta.size < circuit.num_params:
            theta = np.concatenate((theta, 0.001 * master[theta.size : circuit.num_params]))
        theta = _train_block(job, f"source_{depth}", circuit, theta, target, job.config.source_updates, 0.2)
        if depth == 2:
            depth_two = theta.copy()
    deep = theta.copy()
    compact = pilot_circuit(2)
    depth_two = _train_block(job, "L2_continue", compact, depth_two, target, 2 * job.config.source_updates, 0.2)
    depth_two = _train_block(job, "L2_control", compact, depth_two, target, 6 * job.config.repair_updates, 0.1)
    depth_four = _train_block(job, "L4_control", circuit, deep, target, 6 * job.config.repair_updates, 0.1)
    candidates = [
        candidate(compact, depth_two, target, "L2"),
        candidate(circuit, depth_four, target, "L4"),
        _prune(job, circuit, deep.copy(), target, "M"),
        _prune(job, circuit, deep.copy(), target, "D"),
    ]
    result: dict[str, object] = {
        "target_id": job.target_id,
        "replica": job.replica,
        "candidates": candidates,
        "construction_updates_per_route": 4 * job.config.source_updates + 6 * job.config.repair_updates,
        "wall_seconds": time.perf_counter() - started,
        "worker_pid": os.getpid(),
    }
    write_record(path, result)
    return result


def choose_nominee(constructions: list[dict[str, object]]) -> str:
    """Select one pruning rule globally using the balanced development panel.

    Returns:
        M or D; exact ties at 1e-12 prefer M as the cheaper scoring method.
    """
    scores = {route: [] for route in ("M", "D")}
    for construction in constructions:
        for item in cast("list[dict[str, object]]", construction["candidates"]):
            route = cast("str", item["route"])
            if route in scores:
                scores[route].append(cast("float", item["exact_noisy_fidelity"]))
    return "D" if np.mean(scores["D"]) > np.mean(scores["M"]) + 1e-12 else "M"


def run_refinement(job: PilotJob) -> dict[str, object]:
    """Fork fixed noiseless/noisy refinements from exactly the same candidates.

    Returns:
        All fixed-budget refinement endpoints with their pre-refinement controls.

    Raises:
        ValueError: If a retained result used another global pruning nominee.
    """
    path = job.directory / "refinement.json"
    if path.exists():
        retained = read_record(path)
        if retained["nominee"] != job.nominee:
            msg = "Refinement nominee differs from retained results."
            raise ValueError(msg)
        return retained
    target = load_target(6, job.target_id).state_vector_copy()
    construction = read_record(job.directory / "construction.json")
    started = time.perf_counter()
    outputs = []
    for item in cast("list[dict[str, object]]", construction["candidates"]):
        route = cast("str", item["route"])
        if route not in {"L2", "L4", job.nominee}:
            continue
        circuit = restore_circuit(cast("dict[str, object]", item["circuit"]))
        initial = np.asarray(item["theta"], dtype=np.float64)
        for arm in ("0", "N"):
            theta = _train_block(
                job,
                f"refine_{route}_{arm}",
                circuit,
                initial.copy(),
                target,
                job.config.refinement_updates,
                0.1,
                noisy=arm == "N",
            )
            result = candidate(circuit, theta, target, f"{route}_{arm}")
            result["initial_theta_checksum"] = digest(initial.tolist())
            result["initial_exact_noisy_fidelity"] = item["exact_noisy_fidelity"]
            outputs.append(result)
    result: dict[str, object] = {
        "target_id": job.target_id,
        "replica": job.replica,
        "nominee": job.nominee,
        "candidates": outputs,
        "wall_seconds": time.perf_counter() - started,
        "worker_pid": os.getpid(),
    }
    write_record(path, result)
    return result


def aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    """Summarize equal-weight target means without treating trajectories as seeds.

    Returns:
        Per-target and equally weighted aggregate fidelities for each route.
    """
    collected: dict[str, dict[str, list[float]]] = {}
    for record in records:
        target = cast("str", record["target_id"])
        for item in cast("list[dict[str, object]]", record["candidates"]):
            route = cast("str", item["route"])
            collected.setdefault(route, {}).setdefault(target, []).append(cast("float", item["exact_noisy_fidelity"]))
    result = {}
    for route, targets in collected.items():
        means = {target: float(np.mean(values)) for target, values in targets.items()}
        result[route] = {
            "target_means": means,
            "mean": float(np.mean(list(means.values()))),
            "worst": min(means.values()),
        }
    return result


def source_fingerprint() -> str:
    """Hash runtime Python sources, including uncommitted implementation changes."""
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve()]
    for directory in (root / "benchmarks/state_preparation", root / "src/mqt/yaqs"):
        paths.extend(directory.rglob("*.py"))
    return digest({
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)
    })


def run_pilot(
    root: Path, *, config: PilotConfig | None = None, workers: WorkerSetting = "auto", resume: bool = False
) -> dict[str, object]:
    """Run or exactly resume the two-stage exploratory pilot.

    Returns:
        The final descriptive results and artifact paths.

    Raises:
        ValueError: If an existing directory lacks explicit compatible resume.
    """
    config = PilotConfig() if config is None else config
    resolve_workers(workers, len(TARGETS) * len(config.seeds))
    manifest = {
        "version": VERSION,
        "config": json.loads(json.dumps(asdict(config))),
        "targets": list(TARGETS),
        "target_fixture_checksum": load_target_collection().checksum,
        "source_fingerprint": source_fingerprint(),
        "noise": ExactLocalPauliProvider().scientific_dict(),
        "selection": "fixed final iterates; global M/D nomination after complete phase 1",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": version("scipy"),
            "qiskit": version("qiskit"),
        },
    }
    if root.exists():
        if not resume or not (root / "manifest.json").is_file() or read_record(root / "manifest.json") != manifest:
            msg = "Existing pilot directory requires explicit resume with an identical manifest."
            raise ValueError(msg)
    else:
        if resume:
            msg = "Cannot resume a pilot directory that does not exist."
            raise ValueError(msg)
        root.mkdir(parents=True)
        write_record(root / "manifest.json", manifest)
    if (root / "result.json").exists():
        return read_record(root / "result.json")
    started = time.perf_counter()
    jobs = [PilotJob(root, config, target, seed) for target in TARGETS for seed in config.seeds]
    constructions_by_index = {}
    for index, result in completed_jobs(run_construction, jobs, workers=workers):
        constructions_by_index[index] = result
        print(
            f"construction {len(constructions_by_index)}/{len(jobs)}: "
            f"{jobs[index].target_id} seed {jobs[index].replica}",
            flush=True,
        )
    constructions = [constructions_by_index[index] for index in range(len(jobs))]
    nominee = choose_nominee(constructions)
    write_record(root / "nomination.json", {"nominee": nominee, "aggregate": aggregate(constructions)})
    print(f"Global pruning nominee: {nominee}; starting paired refinements", flush=True)
    refinement_jobs = [PilotJob(root, config, job.target_id, job.replica, nominee) for job in jobs]
    refinements_by_index = {}
    for index, result in completed_jobs(run_refinement, refinement_jobs, workers=workers):
        refinements_by_index[index] = result
        print(
            f"refinement {len(refinements_by_index)}/{len(jobs)}: {jobs[index].target_id} seed {jobs[index].replica}",
            flush=True,
        )
    refinements = [refinements_by_index[index] for index in range(len(jobs))]
    result: dict[str, object] = {
        "status": "complete",
        "version": VERSION,
        "scope": "exploratory four-family screening; not held-out confirmation",
        "manifest_checksum": digest(manifest),
        "nominee": nominee,
        "construction": aggregate(constructions),
        "refinement": aggregate(refinements),
        "completed_target_seed_jobs": len(jobs),
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "construction_worker_pids": sorted({cast("int", item["worker_pid"]) for item in constructions}),
        "refinement_worker_pids": sorted({cast("int", item["worker_pid"]) for item in refinements}),
        "artifacts": [str(job.directory.relative_to(root)) for job in jobs],
    }
    write_record(root / "result.json", result)
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the fixed-budget pilot CLI, preserving failed and completed artifacts.

    Returns:
        Zero on completion and two on an execution/configuration error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=parse_workers, default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(arguments)
    try:
        result = run_pilot(args.output_root, workers=args.workers, resume=args.resume)
    except Exception:
        LOGGER.exception("Pilot stopped; retained checkpoints can be resumed after diagnosing the failure")
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
