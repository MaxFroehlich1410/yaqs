# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Strong final-iterate structural Adam/online baselines with charged CPU cutoffs."""

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
from benchmarks.state_preparation.exact_density import density_loss_gradient
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider
from benchmarks.state_preparation.targets import load_target
from benchmarks.state_preparation.weighted_sampling import FactorSampler
from experiments import state_preparation_exact_adam as adam
from experiments import state_preparation_noisy_restructuring as legacy
from experiments import state_preparation_pilot as pilot
from experiments import state_preparation_trajectory_diagnostics as diagnostics
from experiments.overnight_state_preparation_20260913_0108 import channels, runner, structural
from mqt.yaqs.optimization import StatevectorBackend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

VERSION = "overnight.strong_structural_baselines.v1"
TARGETS = ("gaussian_mu0p5_sigma0p1", "tfim_critical", "haar_random_1", "random_mps_bond2")
METHODS = ("exact_adam", "weighted_adam", "adaptive_online")
LABELS = (*(f"structural_{m}" for m in METHODS), "fixed_adaptive_online")


@dataclass(frozen=True)
class Config:
    """Fixed scientific defaults; smaller positive budgets support toy tests."""

    total_updates: int = 1500
    warmup: int = 150
    rounds: int = 15
    shortlist: int = 2
    repair: int = 25
    checkpoint_every: int = 10
    trajectories: int = 32
    protected: int = 18

    def __post_init__(self) -> None:
        """Reject invalid budgets before creating artifacts or numerical work."""
        if (
            any(type(v) is not int or v < int(k != "protected") for k, v in asdict(self).items())
            or self.trajectories < 2
            or self.total_updates <= self.warmup + (self.shortlist + 1) * self.repair
        ):
            msg = "Baseline budgets must leave warmup, one balanced repair round and polishing."
            raise ValueError(msg)


def initial_state(circuit: ParameterizedCircuit, theta: NDArray[np.float64]) -> dict[str, Any]:
    """Return the historical independent moment/counter initialization."""
    return legacy.initial_state({"circuit": pilot.circuit_record(circuit), "initial": theta.tolist()})


def state_key(state: dict[str, Any]) -> str:
    """Bind a physical circuit and exact parameter vector to its optimizer memory."""
    return pilot.digest([state["circuit"], state["theta"]])


def project_state(parent: dict[str, Any], circuit: ParameterizedCircuit, theta: NDArray[np.float64]) -> dict[str, Any]:
    """Compact surviving moments exactly as the historical deletion baseline."""
    original = pilot.restore_circuit(parent["circuit"])
    if len(circuit.gates) == len(original.gates):
        projected = copy.deepcopy(parent)
    else:
        remaining = {g.logical_gate_id for g in circuit.gates}
        removed = [i for i, gate in enumerate(original.gates) if gate.logical_gate_id not in remaining]
        if len(removed) != 1:
            msg = "A baseline repair must retain or delete exactly one physical gate."
            raise ValueError(msg)
        projected = legacy.deleted_state(parent, removed[0])
    if pilot.digest(projected["circuit"]) != pilot.digest(pilot.circuit_record(circuit)) or not np.array_equal(
        projected["theta"], theta
    ):
        msg = "Repair parameters or surviving gate identities do not match their parent."
        raise ValueError(msg)
    return projected


def train_piece(
    state: dict[str, Any],
    target: NDArray[np.complex128],
    method: str,
    updates: int,
    epoch: int,
    seed: int,
    config: Config,
) -> dict[str, Any]:
    """Preserve historical stochastic updates with a fast exact endpoint score."""
    if method not in METHODS or type(updates) is not int or updates < 1:
        msg = "Unknown baseline method or invalid piece length."
        raise ValueError(msg)
    result = copy.deepcopy(state)
    circuit = pilot.restore_circuit(state["circuit"])
    theta, first, second = (np.asarray(state[k], dtype=float) for k in ("theta", "first", "second"))
    counts = np.asarray(state["counts"], dtype=int)
    calibration = ExactLocalPauliProvider().calibration
    engine = StatevectorBackend(circuit, target) if method != "exact_adam" else None
    sampler = FactorSampler(circuit, calibration) if method != "exact_adam" else None
    trace = []
    for offset in range(1, updates + 1):
        step = state["updates"] + offset
        extra = {}
        if method == "exact_adam":
            signal, _, _ = density_loss_gradient(circuit, theta, target, calibration)
            theta, first, second = adam.adam_step(
                theta, signal, first, second, step, adam.AdamConfig(learning_rate=0.02)
            )
            phase, passes = "exact_adam", 0
        else:
            assert sampler is not None
            assert engine is not None
            maps, weights = sampler.ensemble(config.trajectories, seed + 1_000_003 * (epoch + offset))
            if method == "weighted_adam":
                signal = engine.gradient(theta, maps, weights=weights)[0]
                theta, first, second = adam.adam_step(
                    theta, signal, first, second, step, adam.AdamConfig(learning_rate=0.02)
                )
                phase = "adam"
            elif step <= 25:
                theta, signal = engine.online_update(theta, 0.7, maps, weights=weights)
                phase = "online"
            else:
                theta, signal, second, counts = engine.adaptive_online_update(
                    theta, 0.02, second, counts, maps, weights=weights
                )
                phase = "adaptive_online"
            passes = config.trajectories * (2 if method == "weighted_adam" else 3)
            extra["ensemble_digest"] = diagnostics.ensemble_digest(maps, weights)
        if not all(np.all(np.isfinite(a)) for a in (theta, first, second, signal)):
            msg = "Nonfinite baseline update."
            raise FloatingPointError(msg)
        trace.append({
            "step": step,
            "epoch": epoch + offset,
            "phase": phase,
            "signal_norm": float(np.linalg.norm(signal)),
            "trajectory_passes": passes,
            **extra,
        })
    result.update({
        "theta": theta.tolist(),
        "first": first.tolist(),
        "second": second.tolist(),
        "counts": counts.tolist(),
        "updates": state["updates"] + updates,
    })
    record = {
        "state": result,
        "fidelity": channels.evaluate(circuit, theta, target, "depolarizing_1s_all"),
        "trace": trace,
        "trajectory_passes": sum(row["trajectory_passes"] for row in trace),
    }
    if sampler is not None:
        record["no_factor_probability"] = sampler.no_factor_probability
    return record


class BlockStore:
    """Store full optimizer states with deterministic epoch and checkpoint resume."""

    def __init__(self, root: Path, target: NDArray[np.complex128], method: str, seed: int, config: Config) -> None:
        """Create a private method workspace with a fixed target and noise stream."""
        self.root, self.target, self.method, self.seed, self.config = root, target.copy(), method, seed, config
        root.mkdir(parents=True, exist_ok=True)

    def run(self, name: str, start: dict[str, Any], updates: int, epoch: int) -> dict[str, Any]:
        """Return a final block iterate, preserving all rejected/replayed attempt work."""
        if (
            self.method not in METHODS
            or type(updates) is not int
            or updates < 1
            or type(epoch) is not int
            or epoch < start["updates"]
        ):
            msg = "Invalid baseline method, update count or noise epoch."
            raise ValueError(msg)
        signature = pilot.digest([
            VERSION,
            self.method,
            start,
            updates,
            epoch,
            self.seed,
            asdict(self.config),
            hashlib.sha256(self.target.tobytes()).hexdigest(),
        ])
        path, partial_path, attempts_path = (
            self.root / f"{name}{suffix}.json" for suffix in ("", ".partial", ".attempts")
        )
        if path.exists():
            record = cast("dict[str, Any]", pilot.read_record(path))
            if record["signature"] != signature or record["status"] != "complete":
                msg = "Completed baseline block identity changed."
                raise ValueError(msg)
            return record
        partial = cast("dict[str, Any]", pilot.read_record(partial_path)) if partial_path.exists() else None
        if partial is not None and partial["signature"] != signature:
            msg = "Partial baseline block identity changed."
            raise ValueError(msg)
        progress: dict[str, Any] = (
            copy.deepcopy(partial)
            if partial is not None
            else {"signature": signature, "state": copy.deepcopy(start), "completed": 0, "trace": [], "curve": []}
        )
        attempts: dict[str, Any] = (
            cast("dict[str, Any]", pilot.read_record(attempts_path))
            if attempts_path.exists()
            else {"signature": signature, "attempts": []}
        )
        if attempts["signature"] != signature:
            msg = "Baseline attempt identity changed."
            raise ValueError(msg)
        previous_cpu = sum(a["cpu_seconds"] for a in attempts["attempts"])
        previous_wall = sum(a["wall_seconds"] for a in attempts["attempts"])
        cpu, wall = time.process_time(), time.perf_counter()
        attempted_updates = score_calls = 0
        completed = False

        def advance() -> None:
            nonlocal attempted_updates, score_calls
            while progress["completed"] < updates:
                amount = min(self.config.checkpoint_every, updates - progress["completed"])
                attempted_updates += amount
                score_calls += 1
                piece = train_piece(
                    progress["state"],
                    self.target,
                    self.method,
                    amount,
                    epoch + progress["completed"],
                    self.seed,
                    self.config,
                )
                progress["state"] = piece["state"]
                progress["completed"] += amount
                progress["trace"].extend(piece["trace"])
                progress["fidelity"] = piece["fidelity"]
                progress["curve"].append({
                    "state": copy.deepcopy(piece["state"]),
                    "fidelity": piece["fidelity"],
                    "cpu_seconds": previous_cpu + time.process_time() - cpu,
                    "updates": progress["completed"],
                })
                pilot.write_record(partial_path, progress)

        try:
            advance()
            completed = True
        finally:
            attempts["attempts"].append({
                "cpu_seconds": time.process_time() - cpu,
                "wall_seconds": time.perf_counter() - wall,
                "attempted_or_reserved_updates": attempted_updates,
                "attempted_or_reserved_scores": score_calls,
                "completed": completed,
            })
            pilot.write_record(attempts_path, attempts)
        physical_updates = sum(a["attempted_or_reserved_updates"] for a in attempts["attempts"])
        endpoint_scores = sum(a["attempted_or_reserved_scores"] for a in attempts["attempts"])
        record = {
            **progress,
            "status": "complete",
            "cpu_seconds": previous_cpu + time.process_time() - cpu,
            "wall_seconds": previous_wall + time.perf_counter() - wall,
            "epoch_end": epoch + updates,
            "attempted_or_reserved_updates": physical_updates,
            "exact_gradient_calls": physical_updates if self.method == "exact_adam" else 0,
            "exact_score_calls": endpoint_scores,
            "trajectory_paths": physical_updates * self.config.trajectories if self.method != "exact_adam" else 0,
            "trajectory_direction_passes": physical_updates
            * self.config.trajectories
            * (2 if self.method == "weighted_adam" else 3)
            if self.method != "exact_adam"
            else 0,
            "work_count_rule": "exact without a failed piece; reserve its full length on interruption",
        }
        pilot.write_record(path, record)
        return record


class MemoryRefiner:
    """Adapt shared deletion search to preserved moments and independent noise epochs."""

    def __init__(self, store: BlockStore, warm: dict[str, Any]) -> None:
        """Seed the incumbent memory map from the complete warmup state."""
        self.store = store
        self.states = {state_key(warm["state"]): copy.deepcopy(warm["state"])}
        self.parent = warm["state"]
        self.epoch = warm["epoch_end"]
        self.round_epoch = self.epoch
        self.position = self.count = self.index = 0
        self.records: list[dict[str, Any]] = []

    def selected_state(self, circuit: dict[str, Any], theta: list[float]) -> dict[str, Any]:
        """Return the exact optimizer memory of the selected physical endpoint."""
        return copy.deepcopy(self.states[pilot.digest([circuit, theta])])

    def __call__(
        self, circuit: ParameterizedCircuit, theta: NDArray[np.float64], cap: int
    ) -> structural.RefinementResult:
        """Repair paired candidates without restarting phases or replaying rejected epochs."""
        if self.position == 0:
            self.parent = self.selected_state(pilot.circuit_record(circuit), theta.tolist())
            eligible = sum(
                g.logical_gate_id not in frozenset(range(self.store.config.protected)) for g in circuit.gates
            )
            self.count = 1 + min(self.store.config.shortlist, eligible)
            self.round_epoch = self.epoch
            self.epoch += cap
        start = project_state(self.parent, circuit, theta)
        self.index += 1
        try:
            record = self.store.run(f"repair_{self.index:03d}", start, cap, self.round_epoch)
            self.records.append(record)
            # Identical parameters cannot strictly improve their identical
            # physical circuit. Preserve the incumbent's older memory on ties.
            self.states.setdefault(state_key(record["state"]), copy.deepcopy(record["state"]))
            return structural.RefinementResult(
                np.asarray(record["state"]["theta"]),
                cap,
                cap,
                {
                    "artifact": f"repair_{self.index:03d}.json",
                    "state": record["state"],
                    "epoch_start": self.round_epoch,
                    "epoch_end": record["epoch_end"],
                    "cpu_seconds": record["cpu_seconds"],
                },
            )
        finally:
            self.position = (self.position + 1) % self.count


def last_available(curve: list[dict[str, Any]], cutoff: float) -> dict[str, Any] | None:
    """Return the last deployable final iterate, never the best or interpolated one."""
    available = [p for p in curve if p["cpu_seconds"] <= cutoff]
    return copy.deepcopy(available[-1]) if available else None


def endpoint(state: dict[str, Any], fidelity: float, cpu: float) -> dict[str, Any]:
    """Return a common deployable curve schema including complete optimizer state."""
    return {
        "circuit": state["circuit"],
        "theta": state["theta"],
        "state": copy.deepcopy(state),
        "fidelity": fidelity,
        "cpu_seconds": cpu,
    }


def run_method(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    method: str,
    seed: int,
    config: Config,
    root: Path,
) -> dict[str, Any]:
    """Run one structural optimizer and, for adaptive online, its extended fixed arm."""
    store = BlockStore(root / "blocks", target, method, seed, config)
    start = initial_state(circuit, theta)
    initial_path = root / "INITIAL.json"
    initial: dict[str, Any]
    if initial_path.exists():
        initial = cast("dict[str, Any]", pilot.read_record(initial_path))
    else:
        cpu = time.process_time()
        initial = {
            "fidelity": channels.evaluate(circuit, theta, target, "depolarizing_1s_all"),
            "cpu_seconds": time.process_time() - cpu,
        }
        pilot.write_record(initial_path, initial)
    warm = store.run("warmup", start, config.warmup, 0)
    prefix_cpu = initial["cpu_seconds"] + warm["cpu_seconds"]
    prefix = [endpoint(start, initial["fidelity"], initial["cpu_seconds"])]
    prefix.extend(endpoint(p["state"], p["fidelity"], initial["cpu_seconds"] + p["cpu_seconds"]) for p in warm["curve"])
    refiner = MemoryRefiner(store, warm)
    search_path = root / "SEARCH.json"
    attempts_path = root / "SEARCH_ATTEMPTS.json"
    attempts: dict[str, Any] = (
        cast("dict[str, Any]", pilot.read_record(attempts_path)) if attempts_path.exists() else {"attempts": []}
    )
    prior_search_cpu = sum(a["cpu_seconds"] for a in attempts["attempts"])
    if search_path.exists():
        search = cast("dict[str, Any]", pilot.read_record(search_path))
        # Reconstruct the state/epoch adapter without running any optimization.
        for row in search["rounds"]:
            for branch in row["branches"]:
                if branch["status"] == "complete":
                    state = branch["diagnostics"]["state"]
                    refiner.states.setdefault(state_key(state), state)
        refiner.epoch = config.warmup + len(search["rounds"]) * config.repair
    else:
        cpu = time.process_time()
        completed = False
        try:
            search = structural.adaptive_delete_refine(
                circuit,
                np.asarray(warm["state"]["theta"]),
                target,
                ExactLocalPauliProvider().calibration,
                refiner,
                config=structural.StructuralConfig(
                    max_rounds=config.rounds,
                    shortlist=config.shortlist,
                    repair_evaluations=config.repair,
                    max_refinement_evaluations=config.total_updates - config.warmup - 1,
                ),
                protected_origins=frozenset(range(config.protected)),
            )
            completed = True
        finally:
            attempts["attempts"].append({"cpu_seconds": time.process_time() - cpu, "completed": completed})
            pilot.write_record(attempts_path, attempts)
        search["physical_cpu_seconds"] = sum(a["cpu_seconds"] for a in attempts["attempts"])
        search["prior_attempt_cpu_seconds"] = prior_search_cpu
        search["exact_search_counts_complete"] = all(a["completed"] for a in attempts["attempts"])
        pilot.write_record(search_path, search)
    selected = refiner.selected_state(search["circuit"], search["theta"])
    remaining = config.total_updates - config.warmup - search["refinement_evaluations"]
    polish = store.run("polish", selected, remaining, refiner.epoch)
    curve = copy.deepcopy(prefix)
    elapsed = prefix_cpu + search["prior_attempt_cpu_seconds"] + search["ledger"][0]["cpu_seconds"]
    current = warm["state"]
    fidelity = warm["fidelity"]
    position = 1
    for row in search["rounds"]:
        block = search["ledger"][position : position + 1 + 2 * len(row["branches"])]
        position += len(block)
        elapsed += sum(entry["cpu_seconds"] for entry in block)
        if row["selected_branch"] is not None:
            chosen = row["branches"][row["selected_branch"]]
            current, fidelity = chosen["diagnostics"]["state"], chosen["fidelity"]
        curve.append(endpoint(current, fidelity, elapsed))
    before_polish = prefix_cpu + search["physical_cpu_seconds"]
    curve.extend(endpoint(p["state"], p["fidelity"], before_polish + p["cpu_seconds"]) for p in polish["curve"])
    records = [warm, polish]
    for row in search["rounds"]:
        records.extend(
            cast("dict[str, Any]", pilot.read_record(store.root / branch["diagnostics"]["artifact"]))
            for branch in row["branches"]
            if branch["status"] == "complete"
        )
    work = {
        key: sum(record[key] for record in records)
        for key in (
            "attempted_or_reserved_updates",
            "exact_gradient_calls",
            "exact_score_calls",
            "trajectory_paths",
            "trajectory_direction_passes",
        )
    }
    failed_caps = sum(
        branch["evaluation_cap"]
        for row in search["rounds"]
        for branch in row["branches"]
        if branch["status"] == "failed"
    )
    work["reserved_failed_updates"] = failed_caps
    work["attempted_or_reserved_updates"] += failed_caps
    work["exact_gradient_calls"] += failed_caps if method == "exact_adam" else 0
    work["trajectory_paths"] += failed_caps * config.trajectories if method != "exact_adam" else 0
    work["trajectory_direction_passes"] += (
        failed_caps * config.trajectories * (2 if method == "weighted_adam" else 3) if method != "exact_adam" else 0
    )
    # Failed candidate work is conservatively reserved, never dropped.
    work["exact_score_calls"] += sum(
        (branch["evaluation_cap"] + config.checkpoint_every - 1) // config.checkpoint_every
        for row in search["rounds"]
        for branch in row["branches"]
        if branch["status"] == "failed"
    )
    rankings = sum(entry["role"] in {"initial_score_validation", "deletion_ranking"} for entry in search["ledger"])
    forward = sum(entry["exact_channel_evaluations"] for entry in search["ledger"]) - rankings
    work["exact_score_calls"] += 1 + forward
    work["deletion_ranking_calls"] = rankings
    work["density_direction_passes"] = 2 * (work["exact_gradient_calls"] + rankings) + work["exact_score_calls"]
    structural_arm = {
        **endpoint(polish["state"], polish["fidelity"], before_polish + polish["cpu_seconds"]),
        "status": "complete",
        "curve": curve,
        "work": work,
        "nominal_charged_updates": config.warmup + search["refinement_evaluations"] + remaining,
        "selected_optimizer_updates": polish["state"]["updates"],
        "search_stop_reason": search["stop_reason"],
        "exact_search_counts_complete": search["exact_search_counts_complete"],
        "failed_branches": sum(branch["status"] == "failed" for row in search["rounds"] for branch in row["branches"]),
        "memory_rule": "preserved_surviving_moments_and_occurrence_counters",
    }
    arms = {f"structural_{method}": structural_arm}
    if method == "adaptive_online":
        fixed = store.run("fixed_tail", warm["state"], config.total_updates - config.warmup, config.warmup)
        fixed_work = {
            key: warm[key] + fixed[key]
            for key in (
                "attempted_or_reserved_updates",
                "exact_gradient_calls",
                "exact_score_calls",
                "trajectory_paths",
                "trajectory_direction_passes",
            )
        }
        fixed_work["exact_score_calls"] += 1
        fixed_work.update({"deletion_ranking_calls": 0, "density_direction_passes": fixed_work["exact_score_calls"]})
        arms["fixed_adaptive_online"] = {
            **endpoint(fixed["state"], fixed["fidelity"], prefix_cpu + fixed["cpu_seconds"]),
            "status": "complete",
            "curve": [
                *prefix,
                *(endpoint(p["state"], p["fidelity"], prefix_cpu + p["cpu_seconds"]) for p in fixed["curve"]),
            ],
            "work": fixed_work,
            "nominal_charged_updates": config.total_updates,
            "selected_optimizer_updates": fixed["state"]["updates"],
            "memory_rule": "continued_shared_warmup",
        }
    return {"arms": arms, "shared_warmup_cpu_seconds": prefix_cpu if method == "adaptive_online" else 0.0}


def numerical_sources() -> dict[str, str]:
    """Fingerprint all numerical dependencies and this preregistered protocol."""
    sources = runner.source_hashes()
    paths = [
        Path(__file__),
        Path(__file__).with_name("structural.py"),
        Path(__file__).with_name("BASELINE_PROTOCOL.md"),
        runner.REPO / "experiments/state_preparation_noisy_restructuring.py",
        runner.REPO / "experiments/state_preparation_trajectory_diagnostics.py",
    ]
    sources.update({
        str(path.relative_to(runner.REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    })
    return sources


def freeze_protocol(output: Path, reference_root: Path) -> dict[str, Any]:
    """Authenticate the fixed panel and L-BFGS protocol without any simulation."""
    manifest = runner.custody()
    starts_path = runner.OUTPUT / "starts.json"
    starts = cast("list[dict[str, Any]]", pilot.read_record(starts_path)["starts"])
    selected = [s for s in starts if s["target_id"] in TARGETS and s["replica"] == 11]
    reference = cast("dict[str, Any]", pilot.read_record(reference_root / "PROTOCOL.json"))
    reference_starts = [s for s in reference["starts"] if s["target_id"] in TARGETS and s["replica"] == 11]
    if (
        len(selected) != len(TARGETS)
        or {s["target_id"] for s in selected} != set(TARGETS)
        or pilot.digest(selected) != pilot.digest(reference_starts)
    ):
        msg = "Baseline and L-BFGS construction panels must be identical and complete."
        raise ValueError(msg)
    expected = {
        "full_evaluations": 1500,
        "warmup_evaluations": 150,
        "max_rounds": 15,
        "shortlist": 2,
        "repair_evaluations": 25,
        "history": 10,
        "protected_initial_gates": 18,
    }
    if reference["phase"] != "development" or any(reference["config"].get(k) != v for k, v in expected.items()):
        msg = "Expected the frozen structural L-BFGS25 comparator policy."
        raise ValueError(msg)
    for source in selected:
        circuit = pilot.restore_circuit(source["candidate"]["circuit"])
        if (
            circuit.num_qubits != 6
            or runner.counts(circuit)["one_qubit"] != 78
            or runner.counts(circuit)["two_qubit"] != 30
        ):
            msg = "Baseline source must be the physical q6 78+30 construction."
            raise ValueError(msg)
        initial_state(circuit, np.asarray(source["candidate"]["theta"]))
    protocol = {
        "version": VERSION,
        "phase": "development",
        "targets": list(TARGETS),
        "replicas": [11],
        "config": asdict(Config()),
        "starts": selected,
        "source_starts_sha256": hashlib.sha256(starts_path.read_bytes()).hexdigest(),
        "benchmark_hashes": manifest["benchmark_hashes"],
        "source_hashes": numerical_sources(),
        "reference_root": str(reference_root),
        "reference_protocol_checksum": pilot.digest(reference),
        "comparison_rule": ("last final iterate within common measured CPU; heterogeneous caps are not matched work"),
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "PROTOCOL.json"
    if path.exists() and pilot.digest(pilot.read_record(path)) != pilot.digest(protocol):
        msg = "Frozen baseline protocol differs; use a new output root."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return protocol


def verify_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """Abort before numerical work or cached results if scientific custody changed."""
    manifest = runner.custody()
    reference = cast("dict[str, Any]", pilot.read_record(Path(protocol["reference_root"]) / "PROTOCOL.json"))
    if (
        protocol["version"] != VERSION
        or protocol["phase"] != "development"
        or protocol["targets"] != list(TARGETS)
        or protocol["replicas"] != [11]
        or pilot.digest(protocol["config"]) != pilot.digest(asdict(Config()))
        or protocol["benchmark_hashes"] != manifest["benchmark_hashes"]
        or protocol["source_hashes"] != numerical_sources()
        or protocol["source_starts_sha256"] != hashlib.sha256((runner.OUTPUT / "starts.json").read_bytes()).hexdigest()
        or protocol["reference_protocol_checksum"] != pilot.digest(reference)
    ):
        msg = "Baseline protocol custody or numerical source identity changed."
        raise ValueError(msg)
    return reference


def evaluate_endpoint(arm: dict[str, Any], target: NDArray[np.complex128]) -> dict[str, Any]:
    """Audit every fixed channel on one final circuit, counting all physical gates."""
    result = copy.deepcopy(arm)
    circuit, theta = pilot.restore_circuit(arm["circuit"]), np.asarray(arm["theta"])
    cpu = time.process_time()
    result["fidelities"] = {noise: channels.evaluate(circuit, theta, target, noise) for noise in channels.NOISE_IDS}
    independent = legacy.fidelity({"circuit": arm["circuit"], "theta": arm["theta"]}, target)
    result["evaluation_cpu_seconds"] = time.process_time() - cpu
    if abs(independent - result["fidelities"]["depolarizing_1s_all"]) > 1e-10:
        msg = "Independent Qiskit final baseline audit disagrees."
        raise FloatingPointError(msg)
    if "fidelity" in arm and abs(arm["fidelity"] - result["fidelities"]["depolarizing_1s_all"]) > 1e-10:
        msg = "Final baseline iterate and independent channel audit disagree."
        raise FloatingPointError(msg)
    result["physical_counts"] = runner.counts(circuit)
    result["ballarin_counts"] = runner.counts(materialize_ballarin_circuit(circuit, theta).circuit)
    return result


def common_cpu_panel(arms: dict[str, dict[str, Any]], target: NDArray[np.complex128]) -> dict[str, Any]:
    """Evaluate only last available final iterates under one complete-panel CPU cutoff."""
    expected = {*LABELS, "structural_lbfgs", "fixed_lbfgs"}
    if set(arms) != expected or any(arm["status"] != "complete" for arm in arms.values()):
        return {"status": "unavailable_incomplete_panel", "cpu_seconds": None, "arms": {}}
    cutoff = min(arm["cpu_seconds"] for arm in arms.values())
    matched = {}
    for name, arm in arms.items():
        point = last_available(arm["curve"], cutoff)
        if point is None:
            return {"status": "unavailable_no_checkpoint", "cpu_seconds": cutoff, "arms": {}}
        matched[name] = evaluate_endpoint(point, target)
        matched[name]["comparison"] = "common_cpu_last_available_final_iterate"
        matched[name]["status"] = "complete"
    return {"status": "complete", "cpu_seconds": cutoff, "arms": matched}


def execute_case(job: dict[str, Any]) -> dict[str, Any]:
    """Run the fixed baseline panel and compare with authenticated L-BFGS endpoints."""
    protocol, source, root = job["protocol"], job["source"], Path(job["root"])
    reference_protocol = verify_protocol(protocol)
    if pilot.digest(source) not in {pilot.digest(s) for s in protocol["starts"]}:
        msg = "Baseline case is absent from the frozen construction panel."
        raise ValueError(msg)
    signature = pilot.digest([protocol, source])
    path = root / "RESULT.json"
    if path.exists():
        result = cast("dict[str, Any]", pilot.read_record(path))
        if result["signature"] != signature or result["status"] not in {"complete", "partial_failure"}:
            msg = "Completed baseline case identity changed."
            raise ValueError(msg)
        return result
    reference_path = Path(protocol["reference_root"]) / f"{source['target_id']}__11/RESULT.json"
    reference = cast("dict[str, Any]", pilot.read_record(reference_path))
    if reference["status"] != "complete" or reference["signature"] != pilot.digest({
        "protocol": reference_protocol,
        "source": source,
    }):
        msg = "Baseline comparison requires the matching completed structural L-BFGS case."
        raise ValueError(msg)
    root.mkdir(parents=True, exist_ok=True)
    target = load_target(6, source["target_id"]).state_vector_copy()
    circuit, theta = pilot.restore_circuit(source["candidate"]["circuit"]), np.asarray(source["candidate"]["theta"])
    seed = int(pilot.digest([legacy.VERSION, legacy.Config().root_seed, source["target_id"], 11])[:16], 16) << 32
    arms: dict[str, dict[str, Any]] = {}
    shared_cpu = 0.0
    record: dict[str, Any]
    for method in METHODS:
        verify_protocol(protocol)
        method_path = root / f"{method}.json"
        if method_path.exists():
            record = cast("dict[str, Any]", pilot.read_record(method_path))
            if record["signature"] != signature:
                msg = "Retained baseline method identity changed."
                raise ValueError(msg)
        else:
            try:
                record = run_method(circuit, theta, target, method, seed, Config(**protocol["config"]), root / method)
                record["status"] = "complete"
                for label, arm in record["arms"].items():
                    record["arms"][label] = evaluate_endpoint(arm, target)
                    record["arms"][label]["comparison"] = "full_cap_endpoint_different_computation"
            except (ArithmeticError, RuntimeError) as error:
                labels = [f"structural_{method}"] + (["fixed_adaptive_online"] if method == "adaptive_online" else [])
                record = {
                    "status": "failed",
                    "arms": {
                        label: {
                            "status": "failed",
                            "failure": f"{type(error).__name__}: {error}",
                            "fidelities": {},
                            "comparison": "failed_no_final_endpoint",
                        }
                        for label in labels
                    },
                    "shared_warmup_cpu_seconds": 0.0,
                }
            record["signature"] = signature
            verify_protocol(protocol)
            pilot.write_record(method_path, record)
        arms.update(record["arms"])
        shared_cpu += record["shared_warmup_cpu_seconds"]
    for label in ("structural_lbfgs", "fixed_lbfgs"):
        arm = copy.deepcopy(reference["arms"][label])
        arm["status"] = "complete"
        arm["physical_counts"] = arm["logical_counts"]
        arm["comparison"] = "full_cap_endpoint_different_computation"
        arms[label] = arm
    panel = common_cpu_panel(arms, target)
    historical = {name: arm for name, arm in reference["arms"].items() if name.startswith("historical_")}
    result = {
        "signature": signature,
        "status": "complete" if panel["status"] == "complete" else "partial_failure",
        "target_id": source["target_id"],
        "replica": 11,
        "arms": arms,
        "common_cpu": panel,
        "historical_unmatched": historical,
        "reference_case_checksum": pilot.digest(reference),
        "shared_adaptive_warmup_cpu_seconds": shared_cpu,
        "historical_rule": "already budget-filtered best-incumbent references remain unmatched",
    }
    verify_protocol(protocol)
    pilot.write_record(path, result)
    return result


def write_scoreboard(root: Path, records: list[dict[str, Any]]) -> None:
    """Retain all methods/conditions and distinguish full-cap from CPU-matched rows."""
    rows = []
    for record in sorted(records, key=operator.itemgetter("target_id")):
        for panel, arms in (
            ("full_cap", record["arms"]),
            ("common_cpu", record["common_cpu"]["arms"]),
            ("historical_unmatched", record["historical_unmatched"]),
        ):
            for method, arm in sorted(arms.items()):
                for noise in channels.NOISE_IDS:
                    physical = arm.get("physical_counts", arm.get("logical_counts", {}))
                    lowered = arm.get("ballarin_counts", {}) if noise == "ballarin_coupled" else physical
                    rows.append({
                        "target_id": record["target_id"],
                        "replica": 11,
                        "panel": panel,
                        "method": method,
                        "noise_id": noise,
                        "status": arm.get("status", "complete"),
                        "fidelity": arm["fidelities"].get(noise),
                        "comparison": arm["comparison"],
                        "charged_cpu_seconds": arm.get("cpu_seconds"),
                        "common_cpu_cutoff_seconds": record["common_cpu"]["cpu_seconds"]
                        if panel == "common_cpu"
                        else None,
                        "physical_one_qubit": physical.get("one_qubit"),
                        "physical_two_qubit": physical.get("two_qubit"),
                        "lowered_one_qubit": lowered.get("one_qubit"),
                        "lowered_two_qubit": lowered.get("two_qubit"),
                        "nominal_charged_updates": arm.get("nominal_charged_updates"),
                        "selected_optimizer_updates": arm.get("selected_optimizer_updates"),
                        "exact_gradient_calls": arm.get("work", {}).get(
                            "exact_gradient_calls", arm.get("joint_gradient_calls")
                        ),
                        "exact_score_calls": arm.get("work", {}).get(
                            "exact_score_calls", arm.get("forward_score_calls")
                        ),
                        "deletion_ranking_calls": arm.get("work", {}).get(
                            "deletion_ranking_calls", arm.get("deletion_ranking_calls")
                        ),
                        "density_direction_passes": arm.get("work", {}).get(
                            "density_direction_passes", arm.get("density_direction_passes")
                        ),
                        "trajectory_paths": arm.get("work", {}).get("trajectory_paths"),
                        "trajectory_direction_passes": arm.get("work", {}).get("trajectory_direction_passes"),
                        "evaluation_cpu_seconds": arm.get("evaluation_cpu_seconds"),
                        "failure": arm.get("failure"),
                    })
    if rows:
        with (root / "SCOREBOARD.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare a frozen panel or explicitly execute it with bounded workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=runner.OUTPUT / "baseline_strong1500")
    parser.add_argument("--reference-root", type=Path, default=runner.OUTPUT / "structural_lbfgs25")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four workers.")
    if args.prepare_only:
        freeze_protocol(args.output, args.reference_root)
        print(f"Prepared only: {args.output / 'PROTOCOL.json'}", flush=True)
        return 0
    protocol = cast("dict[str, Any]", pilot.read_record(args.output / "PROTOCOL.json"))
    verify_protocol(protocol)
    jobs = [
        {"protocol": protocol, "source": source, "root": str(args.output / f"{source['target_id']}__11")}
        for source in protocol["starts"]
    ]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(execute_case, jobs):
            records.append(result)
            write_scoreboard(args.output, records)
            print(result["target_id"], result["status"], flush=True)
    complete = all(r["status"] == "complete" for r in records)
    pilot.write_record(
        args.output / "RESULT.json",
        {
            "status": "complete" if complete else "partial_failure",
            "protocol_checksum": pilot.digest(protocol),
            "cases": [pilot.digest(r) for r in records],
        },
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
