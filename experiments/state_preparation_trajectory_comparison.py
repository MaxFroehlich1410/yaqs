# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Experiment 2: fresh-target, paired trajectory hybrid/Adam comparison."""

from __future__ import annotations

import argparse
import hashlib
import logging
import platform
import resource
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

# Deliberately reuse the canonical generator internals without expanding its public API.
from benchmarks.generate_state_preparation_targets import (
    TfimSpec,
    _dense_haar_random_state,  # noqa: PLC2701
    _normalize_and_fix_global_phase,  # noqa: PLC2701
    _paper_quantics_grid,  # noqa: PLC2701
    _random_mps_state,  # noqa: PLC2701
    _state_to_json_vector,  # noqa: PLC2701
    _tfim_ground_state,  # noqa: PLC2701
    json_vector_to_state,
)
from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from benchmarks.state_preparation.execution import WorkerSetting, completed_jobs, parse_workers, resolve_workers
from benchmarks.state_preparation.pilot_components import (
    ExactLocalPauliProvider,
    delete_gates,
    pruning_scores,
    select_deletions,
)
from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_exact_adam as exact
from experiments import state_preparation_hybrid_pilot as hybrid
from experiments import state_preparation_pilot as pilot
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KrotovTJMOptions,
    KrotovTruncation,
    StatevectorBackend,
    noisy_state_preparation_contribution,
)

# Benchmark-only access excludes diagnostic forwards; tested against public trainers.
from mqt.yaqs.optimization.krotov import (
    _noisy_state_preparation_online_update,  # noqa: PLC2701
    _state_preparation_online_update,  # noqa: PLC2701
)
from mqt.yaqs.optimization.krotov import state_preparation_contribution as noiseless_gradient

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

VERSION = "yaqs.trajectory_competitiveness.v1"
KINDS = ("noiseless_hybrid", "exact_adam", "trajectory_adam", "trajectory_hybrid", "trajectory_batch")
FAMILIES = ("gaussian", "tfim", "haar", "mps")
ROOT_SEED = 2026091001


@dataclass(frozen=True)
class Config:
    """Fixed refinement budget; reduced values are only for engineering tests."""

    updates: int = 50
    switch: int = 25
    trajectories: int = 32
    backend: str = "mps"

    def __post_init__(self) -> None:
        """Reject malformed budgets before creating artifacts."""
        if (
            any(type(value) is not int for value in (self.updates, self.switch, self.trajectories))
            or not 0 <= self.switch < self.updates
            or self.trajectories < 1
            or self.backend not in {"mps", "statevector"}
        ):
            msg = "Invalid trajectory comparison budget."
            raise ValueError(msg)

    @property
    def evaluation_steps(self) -> tuple[int, ...]:
        """Return the diagnostic checkpoints within the configured budget."""
        return tuple(sorted({0, self.updates, self.switch, min(10, self.updates)}))


@dataclass(frozen=True)
class Method:
    """One optimizer and its prefix/tail learning rates."""

    kind: str
    rate: float
    tail_rate: float = 0.1

    def __post_init__(self) -> None:
        """Reject unknown methods and nonpositive or nonfinite rates."""
        if self.kind not in KINDS or any(
            isinstance(value, bool) or not np.isfinite(value) or value <= 0 for value in (self.rate, self.tail_rate)
        ):
            msg = "Invalid comparison method."
            raise ValueError(msg)

    @property
    def label(self) -> str:
        """Return an unambiguous directory label including both rates."""
        return f"{self.kind}_r{self.rate:g}_t{self.tail_rate:g}".replace(".", "p")

    def phase_rate(self, step: int, config: Config) -> tuple[str, float]:
        """Return the finite online/prefix schedule at a one-based update."""
        if not 1 <= step <= config.updates:
            msg = "Update outside configured budget."
            raise ValueError(msg)
        if self.kind.endswith("adam"):
            return "adam", self.rate
        prefix = step <= config.switch
        return ("online" if prefix and self.kind.endswith("hybrid") else "batch"), (
            self.rate if prefix else self.tail_rate
        )


def seed_for(role: str, target_id: str, replica: int) -> int:
    """Use disjoint seed blocks; omit method/rate to pair within each case."""
    return int(pilot.digest([VERSION, ROOT_SEED, role, target_id, replica])[:16], 16) << 32


def target_record(target_id: str, family: str, state: NDArray[np.complex128], **metadata: object) -> dict[str, object]:
    """Store the exact little-endian vector with its mathematical specification."""
    return {"target_id": target_id, "family": family, "vector": _state_to_json_vector(state), "metadata": metadata}


def target_vector(record: dict[str, object]) -> NDArray[np.complex128]:
    """Restore a normalized target from a JSON-safe panel record."""
    return json_vector_to_state(cast("list[list[float]]", record["vector"]))


def fresh_targets() -> list[dict[str, object]]:
    """Generate the eight predeclared confirmation states without changing fixtures."""
    records = []
    for index, (mean, sigma) in enumerate(((0.35, 0.08), (0.65, 0.14)), 1):
        amplitude = np.exp(-((_paper_quantics_grid(6) - mean) ** 2) / (4 * sigma**2)).astype(complex)
        records.append(
            target_record(
                f"gaussian_fresh_{index}",
                "gaussian",
                _normalize_and_fix_global_phase(amplitude),
                mean=mean,
                sigma=sigma,
            )
        )
    for index, field in enumerate((0.7, 1.3), 1):
        spec = TfimSpec(f"tfim_fresh_{index}", "fresh", 9200 + index, field)
        state, energy, couplings, fields = _tfim_ground_state(6, spec)
        records.append(
            target_record(
                spec.target_id,
                "tfim",
                state,
                specification=asdict(spec),
                energy=energy,
                couplings=couplings.tolist(),
                fields=fields.tolist(),
                boundary="open",
            )
        )
    for index in (1, 2):
        records.extend((
            target_record(f"haar_fresh_{index}", "haar", _dense_haar_random_state(6, 9100 + index), seed=9100 + index),
            target_record(
                f"mps_fresh_{index}",
                "mps",
                _random_mps_state(6, 9000 + index, 2)[0],
                seed=9000 + index,
                bond_dimension=2,
            ),
        ))
    return records


def optimizer_step(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    target_mps: MPS | None,
    moments: tuple[NDArray[np.float64], NDArray[np.float64]],
    method: Method,
    config: Config,
    seed: int,
    step: int,
    *,
    dense: StatevectorBackend | None = None,
) -> tuple[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    """Reuse production update kernels, excluding extra diagnostic forwards."""
    phase, rate = method.phase_rate(step, config)
    truncation = KrotovTruncation()
    options = KrotovTJMOptions(
        num_trajectories=config.trajectories, random_seed=seed, trajectory_update="independent", use_crn=False
    )
    provider = ExactLocalPauliProvider()
    if config.backend == "statevector" and method.kind != "exact_adam":
        engine = dense if dense is not None else StatevectorBackend(circuit, target)
        maps = (
            None
            if method.kind == "noiseless_hybrid"
            else engine.sample_noise_maps(theta, options, provider, iteration=step)
        )
        if phase == "online":
            updated, signal = engine.online_update(theta, rate, maps)
            return updated, moments, signal
        signal, _, _ = engine.gradient(theta, maps)
        if phase == "adam":
            updated, first, second = exact.adam_step(
                theta, signal, *moments, step, exact.AdamConfig(updates=config.updates, learning_rate=rate)
            )
            return updated, (first, second), signal
        updated = theta - rate * signal
        return updated, moments, signal
    if method.kind == "exact_adam":
        signal, _, _ = density_loss_gradient(circuit, theta, target, CalibratedLocalDepolarizingProvider())
        updated, first, second = exact.adam_step(
            theta, signal, *moments, step, exact.AdamConfig(updates=config.updates, learning_rate=rate)
        )
        return updated, (first, second), signal
    assert target_mps is not None
    initial = MPS(circuit.num_qubits)
    if phase == "online":
        if method.kind == "noiseless_hybrid":
            updated, signal = _state_preparation_online_update(circuit, theta, target_mps, rate, initial, truncation)
        else:
            updated, signal = _noisy_state_preparation_online_update(
                circuit, theta, target_mps, None, options, rate, initial, truncation, step, noise_provider=provider
            )
        return updated, moments, signal
    if method.kind == "noiseless_hybrid":
        signal, _, _ = noiseless_gradient(circuit, theta, target_mps, initial, truncation)
    else:
        signal, _, _, _ = noisy_state_preparation_contribution(
            circuit, theta, target_mps, None, options, initial, truncation, iteration=step, noise_provider=provider
        )
    if phase == "adam":
        updated, first, second = exact.adam_step(
            theta, signal, *moments, step, exact.AdamConfig(updates=config.updates, learning_rate=rate)
        )
        return updated, (first, second), signal
    updated = theta - rate * signal
    return updated, moments, signal


def optimize(
    path: Path,
    start: dict[str, object],
    target: dict[str, object],
    method: Method,
    config: Config,
    provenance: str,
    seed: int,
    *,
    stop_after: int | None = None,
) -> dict[str, object]:
    """Run a resumable refinement, retaining moments, streams and exact checkpoints."""
    limit = config.updates if stop_after is None else stop_after
    if type(limit) is not int or not 0 <= limit <= config.updates:
        msg = "Invalid stop checkpoint."
        raise ValueError(msg)
    signature = pilot.digest({
        "start": start,
        "target": target,
        "method": asdict(method),
        "config": asdict(config),
        "provenance": provenance,
        "seed": seed,
        "version": VERSION,
    })
    circuit = pilot.restore_circuit(cast("dict[str, object]", start["circuit"]))
    initial = np.asarray(start["theta"], dtype=float)
    vector = target_vector(target)
    state: dict[str, object] = {
        "signature": signature,
        "method": asdict(method),
        "config": asdict(config),
        "seed": seed,
        "start_checksum": pilot.digest(start),
        "target_checksum": pilot.digest(target),
        "theta": initial.tolist(),
        "first_moment": np.zeros_like(initial).tolist(),
        "second_moment": np.zeros_like(initial).tolist(),
        "completed_updates": 0,
        "trace": [],
        "evaluations": [],
        "status": "running",
    }
    if path.exists():
        state = pilot.read_record(path)
        if state["signature"] != signature:
            msg = "Comparison checkpoint configuration mismatch."
            raise ValueError(msg)
    completed = cast("int", state["completed_updates"])
    trace = cast("list[dict[str, object]]", state["trace"])
    evaluations = cast("list[dict[str, object]]", state["evaluations"])
    theta, first, second = (np.asarray(state[key], dtype=float) for key in ("theta", "first_moment", "second_moment"))
    if (
        type(completed) is not int
        or not 0 <= completed <= config.updates
        or [row["step"] for row in trace] != list(range(1, completed + 1))
        or any(value.shape != initial.shape or not np.all(np.isfinite(value)) for value in (theta, first, second))
        or np.any(second < 0)
        or (
            completed
            and [row["update"] for row in evaluations] != [s for s in config.evaluation_steps if s <= completed]
        )
        or (state["status"] == "complete" and completed != config.updates)
    ):
        msg = "Invalid comparison checkpoint progress."
        raise ValueError(msg)
    if state["status"] == "complete":
        return state
    target_mps = MPS.from_statevector(vector) if config.backend == "mps" and method.kind != "exact_adam" else None
    dense = (
        StatevectorBackend(circuit, vector) if config.backend == "statevector" and method.kind != "exact_adam" else None
    )

    def evaluate(step: int) -> None:
        started = time.perf_counter()
        record = exact.evaluate(circuit, theta, vector, step)
        record["evaluation_wall_seconds"] = time.perf_counter() - started
        evaluations.append(record)

    if not evaluations:
        evaluate(0)
    for step in range(completed + 1, limit + 1):
        wall, cpu = time.perf_counter(), time.process_time()
        updated, (first, second), signal = optimizer_step(
            circuit, theta, vector, target_mps, (first, second), method, config, seed, step, dense=dense
        )
        cpu_seconds, wall_seconds = time.process_time() - cpu, time.perf_counter() - wall
        if not np.all(np.isfinite(updated)) or not np.all(np.isfinite(signal)):
            msg = "Nonfinite comparison update."
            raise FloatingPointError(msg)
        phase, rate = method.phase_rate(step, config)
        trace.append({
            "step": step,
            "phase": phase,
            "rate": rate,
            "signal_norm": float(np.linalg.norm(signal)),
            "update_norm": float(np.linalg.norm(updated - theta)),
            "cpu_seconds": cpu_seconds,
            "wall_seconds": wall_seconds,
        })
        theta = updated
        if step in config.evaluation_steps:
            evaluate(step)
        if step % 10 == 0 or step in config.evaluation_steps or step == limit:
            state.update({
                "theta": theta.tolist(),
                "first_moment": first.tolist(),
                "second_moment": second.tolist(),
                "completed_updates": step,
                "status": "complete" if step == config.updates else "running",
                "trace": trace,
                "evaluations": evaluations,
            })
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            state["process_high_water_rss_bytes"] = int(rss if platform.system() == "Darwin" else rss * 1024)
            pilot.write_record(path, state)
    if limit == 0:
        pilot.write_record(path, state)
    return state


@dataclass(frozen=True)
class RefinementJob:
    """One private, paired refinement case and optimizer configuration."""

    root: Path
    stage: str
    start: dict[str, object]
    target: dict[str, object]
    replica: int
    method: Method
    config: Config
    provenance: str

    @property
    def path(self) -> Path:
        """Return the job-owned checkpoint path."""
        return (
            self.root
            / self.stage
            / str(self.target["target_id"])
            / f"seed_{self.replica}"
            / f"{self.method.label}.json"
        )


def run_refinement(job: RefinementJob) -> dict[str, object]:
    """Run one optimizer and return an authenticated compact endpoint summary."""
    state = optimize(
        job.path,
        job.start,
        job.target,
        job.method,
        job.config,
        job.provenance,
        seed_for(job.stage, str(job.target["target_id"]), job.replica),
    )
    evaluations = cast("list[dict[str, object]]", state["evaluations"])
    trace = cast("list[dict[str, object]]", state["trace"])
    return {
        "target_id": job.target["target_id"],
        "family": job.target["family"],
        "replica": job.replica,
        "method": asdict(job.method),
        "path": str(job.path.relative_to(job.root)),
        "checksum": pilot.digest(state),
        "frozen_fidelity": evaluations[0]["exact_noisy_fidelity"],
        "final_fidelity": evaluations[-1]["exact_noisy_fidelity"],
        "final_noiseless_fidelity": evaluations[-1]["noiseless_fidelity"],
        "cpu_seconds": sum(cast("float", row["cpu_seconds"]) for row in trace),
        "wall_seconds": sum(cast("float", row["wall_seconds"]) for row in trace),
    }


def development_methods() -> tuple[Method, ...]:
    """Return the equal three-configuration development grids."""
    return tuple(
        Method(kind, rate)
        for kind in KINDS
        for rate in ((0.001, 0.005, 0.02) if kind.endswith("adam") else (0.1, 0.3, 0.7))
    )


def nominate(rows: list[dict[str, object]]) -> dict[str, object]:
    """Select globally on a complete development panel; retain matched batch rates."""
    scores = []
    for method in development_methods():
        cases = [r for r in rows if r["method"] == asdict(method)]
        expected_cases = {(target, family, 11) for target, family in zip(pilot.TARGETS, FAMILIES, strict=True)}
        if len(cases) != 4 or {(r["target_id"], r["family"], r["replica"]) for r in cases} != expected_cases:
            msg = "Nomination requires the complete four-family development grid."
            raise ValueError(msg)
        score = float(np.mean([r["final_fidelity"] for r in cases]))
        if not np.isfinite(score):
            msg = "Nonfinite development score."
            raise ValueError(msg)
        scores.append({"method": asdict(method), "mean_final_fidelity": score})
    if len(rows) != len(scores) * 4:
        msg = "Unexpected development records."
        raise ValueError(msg)
    chosen = {}
    for kind in KINDS:
        candidates = [r for r in scores if cast("dict[str, object]", r["method"])["kind"] == kind]
        winner = min(
            candidates,
            key=lambda r: (-cast("float", r["mean_final_fidelity"]), cast("dict[str, float]", r["method"])["rate"]),
        )
        chosen[kind] = winner["method"]
    batch_winner = chosen["trajectory_batch"]
    matched = {**cast("dict[str, object]", chosen["trajectory_hybrid"]), "kind": "trajectory_batch"}
    chosen["trajectory_batch"] = matched
    return {
        "selected": chosen,
        "independent_batch_development_winner": batch_winner,
        "scores": scores,
        "development_records_checksum": pilot.digest(rows),
        "selection": "global; batch rate matched to noisy hybrid",
    }


@dataclass(frozen=True)
class ConstructionJob:
    """One fresh target/replica; construction is shared by every refinement."""

    root: Path
    target: dict[str, object]
    replica: int
    provenance: str
    backend: str = "mps"


def construct(job: ConstructionJob) -> dict[str, object]:
    """Apply the frozen 570-update noiseless growth/pruning recipe once per case."""
    target_id = str(job.target["target_id"])
    directory = job.root / "starts" / target_id / f"seed_{job.replica}"
    signature = pilot.digest({
        "target": job.target,
        "replica": job.replica,
        "provenance": job.provenance,
        "backend": job.backend,
    })
    path = directory / "start.json"
    if path.exists():
        record = pilot.read_record(path)
        if record["signature"] != signature:
            msg = "Construction checkpoint configuration mismatch."
            raise ValueError(msg)
        for name, checksum in cast("dict[str, str]", record["blocks"]).items():
            if pilot.digest(pilot.read_record(directory / name)) != checksum:
                msg = "Construction block checksum mismatch."
                raise ValueError(msg)
        return record
    vector = target_vector(job.target)
    master = np.random.default_rng(seed_for("construction", target_id, job.replica)).normal(
        size=pilot.pilot_circuit(4).num_params
    )
    theta = 0.1 * master[: pilot.pilot_circuit(1).num_params]
    blocks: dict[str, str] = {}

    def train(name: str, updates: int, rate: float, schedule: hybrid.Schedule) -> NDArray[np.float64]:
        block = hybrid.optimize_block(
            directory / name,
            circuit,
            theta,
            vector,
            updates=updates,
            batch_rate=rate,
            schedule=schedule,
            seed=seed_for("construction", target_id, job.replica),
            backend=job.backend,
        )
        blocks[name] = pilot.digest(block)
        return np.asarray(block["theta"], dtype=float)

    for depth in range(1, 5):
        circuit = pilot.pilot_circuit(depth)
        if theta.size < circuit.num_params:
            theta = np.concatenate((theta, 0.001 * master[theta.size : circuit.num_params]))
        theta = train(f"growth_{depth}.json", 100, 0.2, hybrid.Schedule(25, 0.7))
    events = []
    for index in range(6):
        scores = pruning_scores(circuit, theta, vector, "M")
        removed = select_deletions(scores, 6)
        events.append({
            "round": index + 1,
            "scores": scores,
            "removed_origins": [circuit.gates[k].logical_gate_id for k in removed],
        })
        circuit, theta = delete_gates(circuit, theta, removed)
        theta = train(f"repair_{index}.json", 20, 0.1, hybrid.Schedule())
    theta = train("final.json", 50, 0.1, hybrid.Schedule())
    if circuit.num_params != 108 or len(circuit.gates) != 108 or sum(len(g.sites) == 2 for g in circuit.gates) != 30:
        msg = "Incorrect compact construction budget."
        raise ValueError(msg)
    record = {
        "signature": signature,
        "target_id": target_id,
        "replica": job.replica,
        "circuit": pilot.circuit_record(circuit),
        "theta": theta.tolist(),
        "pruning_events": events,
        "blocks": blocks,
        "evaluation": exact.evaluate(circuit, theta, vector, 0),
        "shared_construction_wall_seconds": sum(
            cast("float", pilot.read_record(directory / name)["wall_seconds"]) for name in blocks
        ),
    }
    pilot.write_record(path, record)
    return record


def run_jobs(jobs: list[RefinementJob], workers: WorkerSetting) -> list[dict[str, object]]:
    """Run a deterministic ordered set of independent refinements."""
    results = {}
    for index, row in completed_jobs(run_refinement, jobs, workers=workers):
        results[index] = row
        print(
            f"{jobs[index].stage} {len(results)}/{len(jobs)}: {row['target_id']} "
            f"seed {row['replica']} {jobs[index].method.label}",
            flush=True,
        )
    return [results[index] for index in range(len(jobs))]


def runtime_fingerprint() -> str:
    """Fingerprint all scientific code, including the reused target generator."""
    root = Path(__file__).resolve().parents[1]
    extra = (
        Path(__file__),
        Path(exact.__file__),
        Path(hybrid.__file__),
        root / "benchmarks/generate_state_preparation_targets.py",
    )
    return pilot.digest({
        "runtime": pilot.source_fingerprint(),
        "extra": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in extra},
    })


def run_experiment(
    root: Path,
    source_root: Path,
    *,
    workers: WorkerSetting = 8,
    resume: bool = False,
    stage: str = "all",
    backend: str = "mps",
) -> dict[str, object]:
    """Freeze development selection before training any fresh-target starts."""
    if stage not in {"all", "development"}:
        msg = "Unknown experiment stage."
        raise ValueError(msg)
    resolve_workers(workers, 80)
    provenance, old_starts = exact.load_starts(source_root)
    config = Config(backend=backend)
    panel = fresh_targets()
    manifest: dict[str, object] = {
        "version": VERSION,
        "config": asdict(config),
        "source_root": str(source_root.resolve()),
        "source_provenance": provenance,
        "runtime_fingerprint": runtime_fingerprint(),
        "targets": panel,
        "replicas": [11, 29],
        "noise": ExactLocalPauliProvider().scientific_dict(),
        "development_grid": [asdict(m) for m in development_methods()],
        "root_seed": ROOT_SEED,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": version("scipy"),
            "qiskit": version("qiskit"),
        },
    }
    if root.exists():
        if not resume or not (root / "manifest.json").exists() or pilot.read_record(root / "manifest.json") != manifest:
            msg = "Existing output requires explicit resume with an identical manifest."
            raise ValueError(msg)
    else:
        if resume:
            msg = "Cannot resume nonexistent experiment."
            raise ValueError(msg)
        root.mkdir(parents=True)
        pilot.write_record(root / "manifest.json", manifest)
    identity = pilot.digest(manifest)
    development_jobs = []
    for old in old_starts:
        if old["replica"] != 11:
            continue
        target_id = str(old["target_id"])
        family = FAMILIES[pilot.TARGETS.index(target_id)]
        target = target_record(target_id, family, load_target(6, target_id).state_vector_copy(), role="development")
        start = cast("dict[str, object]", old["candidate"])
        development_jobs.extend(
            RefinementJob(root, "development", start, target, 11, method, config, identity)
            for method in development_methods()
        )
    # A cached selection never bypasses authentication of the development endpoints.
    development = run_jobs(development_jobs, workers)
    selection = nominate(development)
    selection_path = root / "selection.json"
    if selection_path.exists():
        if pilot.read_record(selection_path) != selection:
            msg = "Frozen selection differs from authenticated development results."
            raise ValueError(msg)
    else:
        pilot.write_record(selection_path, selection)
    if stage == "development":
        return selection
    selected = [Method(**cast("dict", record)) for record in cast("dict[str, dict]", selection["selected"]).values()]
    selected.sort(key=lambda method: KINDS.index(method.kind))
    confirmation_identity = pilot.digest({"manifest": identity, "selection": pilot.digest(selection)})
    construction_jobs = [
        ConstructionJob(root, target, replica, confirmation_identity, backend)
        for target in panel
        for replica in (11, 29)
    ]
    starts = {}
    for index, record in completed_jobs(construct, construction_jobs, workers=workers):
        starts[index] = record
        print(f"construction {len(starts)}/16: {record['target_id']} seed {record['replica']}", flush=True)
    jobs = [
        RefinementJob(
            root, "confirmation", starts[index], job.target, job.replica, method, config, confirmation_identity
        )
        for index, job in enumerate(construction_jobs)
        for method in selected
    ]
    confirmation = run_jobs(jobs, workers)
    result: dict[str, object] = {
        "status": "complete",
        "manifest_checksum": identity,
        "selection_checksum": pilot.digest(selection),
        "development": development,
        "confirmation": confirmation,
        "start_checksums": [pilot.digest(starts[index]) for index in range(len(construction_jobs))],
    }
    result_path = root / "result.json"
    if result_path.exists():
        if pilot.read_record(result_path) != result:
            msg = "Completed result does not match its authenticated checkpoints."
            raise ValueError(msg)
    else:
        pilot.write_record(result_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the frozen experiment without modifying prior artifact directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=parse_workers, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("all", "development"), default="all")
    parser.add_argument("--backend", choices=("mps", "statevector"), default="mps")
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    result = run_experiment(
        arguments.output_root,
        arguments.source_root,
        workers=arguments.workers,
        resume=arguments.resume,
        stage=arguments.stage,
        backend=arguments.backend,
    )
    print(
        f"Experiment 2 {arguments.stage} complete: {arguments.output_root} "
        f"({result.get('status', 'selection frozen')})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
