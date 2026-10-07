# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Frozen exact-anchor SVRG Adam screen with paired weighted and exact controls.

The control variate is the known Johnson-Zhang construction, adapted to fresh
weighted Pauli trajectories and persistent Adam moments. Only the fixed local
Pauli channel is optimized. Final scoring covers all twelve benchmark settings.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import math
import operator
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from benchmarks.state_preparation.ballarin import materialize_ballarin_circuit
from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider as Calibration
from benchmarks.state_preparation.exact_density import density_loss_gradient
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider, exact_density_fidelity
from benchmarks.state_preparation.targets import load_target
from benchmarks.state_preparation.weighted_sampling import FactorSampler
from experiments import state_preparation_exact_adam as adam
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import channels, runner, variance
from mqt.yaqs.optimization import StatevectorBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit
    from mqt.yaqs.optimization.statevector import NoiseMaps

ROOT = Path(__file__).parent
REPO = ROOT.parents[1]
OUTPUT = REPO / "output/overnight-state-preparation-20260913-0108"
TARGETS = ("gaussian_mu0p5_sigma0p1", "tfim_critical", "haar_random_1", "random_mps_bond2")
METHODS = ("cv_adam", "weighted_adam", "exact_adam")
COUNTERS = (
    "exact_gradient_calls",
    "exact_anchor_calls",
    "trajectory_gradient_calls",
    "trajectory_paths",
    "trajectory_direction_passes",
    "density_gradient_direction_passes",
    "diagnostic_forward_calls",
    "diagnostic_direction_passes",
)


@dataclass(frozen=True)
class Config:
    """Predeclared production settings; reduced values are engineering tests only."""

    updates: int = 150
    correction_paths: int = 16
    refresh_every: int = 5
    learning_rate: float = 0.02
    checkpoint_every: int = 10

    def __post_init__(self) -> None:
        """Reject invalid budgets or hyperparameters.

        Raises:
            ValueError: If a budget is nonpositive or a rate is invalid.
        """
        budgets = (self.updates, self.correction_paths, self.refresh_every, self.checkpoint_every)
        if (
            any(type(value) is not int or value < 1 for value in budgets)
            or self.correction_paths < 2
            or isinstance(self.learning_rate, bool)
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            msg = "Invalid SVRG experiment configuration."
            raise ValueError(msg)


def stream_seed(target_id: str, replica: int, update: int) -> int:
    """Derive fresh update streams in a separate paired development domain."""
    return runner.seed_for("development_training_svrg_v1", target_id, replica, update)


def map_digest(maps: NoiseMaps) -> str:
    """Fingerprint every physical operator and its ordered gate/site placement."""
    digest = hashlib.sha256()
    for path in maps:
        digest.update(b"path")
        for gate in path:
            digest.update(b"gate")
            for matrix, sites in gate.operators:
                digest.update(np.asarray(sites, dtype="<i8").tobytes())
                digest.update(np.asarray(matrix, dtype="<c16").tobytes())
    return digest.hexdigest()


def encode_snapshot(snapshot: variance.Snapshot, update: int) -> dict[str, Any]:
    """Serialize the complete exact anchor, with an internal integrity checksum."""
    estimate = asdict(snapshot.estimate)
    estimate["gradient"] = snapshot.estimate.gradient.tolist()
    payload = {
        "theta": snapshot.theta.tolist(),
        "estimate": estimate,
        "problem_signature": snapshot.problem_signature,
        "created_before_update": update,
    }
    return {**payload, "checksum": pilot.digest(payload)}


def decode_snapshot(record: dict[str, Any], shape: tuple[int, ...]) -> variance.Snapshot:
    """Restore a finite immutable exact anchor without an extra oracle call.

    Raises:
        ValueError: If the anchor is corrupt, sampled, or dimensionally invalid.
    """
    payload = {key: value for key, value in record.items() if key != "checksum"}
    raw = dict(record["estimate"])
    theta, gradient = np.asarray(record["theta"], dtype=float), np.asarray(raw["gradient"], dtype=float)
    if (
        pilot.digest(payload) != record["checksum"]
        or any(array.shape != shape or not np.all(np.isfinite(array)) for array in (theta, gradient))
        or not math.isfinite(raw["loss"])
        or raw["exact_channel_evaluations"] != 1
        or raw["paths"] != 0
        or raw["seed"] is not None
        or raw["bias_status"] != "exact_channel_expectation"
    ):
        msg = "Invalid saved exact anchor."
        raise ValueError(msg)
    theta.setflags(write=False)
    gradient.setflags(write=False)
    raw["gradient"] = gradient
    return variance.Snapshot(theta, variance.Estimate(**raw), record["problem_signature"])


def new_attempt() -> dict[str, Any]:
    """Create an independent physical-attempt ledger, including discarded work."""
    return {
        **dict.fromkeys(COUNTERS, 0),
        "training_cpu_seconds": 0.0,
        "training_wall_seconds": 0.0,
        "diagnostic_cpu_seconds": 0.0,
        "evaluation_cpu_seconds": 0.0,
        "evaluation_forward_calls": 0,
    }


def _increment(state: dict[str, Any], attempt: dict[str, Any], **amounts: int) -> None:
    """Charge attempted numerical work to both logical and physical ledgers."""
    for key, amount in amounts.items():
        state[key] += amount
        attempt[key] += amount


def _signature(
    circuit: ParameterizedCircuit,
    initial: NDArray[np.float64],
    target: NDArray[np.complex128],
    config: Config,
    method: str,
    target_id: str,
    replica: int,
) -> str:
    """Authenticate all parameters, target bytes and deterministic stream identity."""
    return pilot.digest({
        "circuit": pilot.circuit_record(circuit),
        "initial": initial.tolist(),
        "target_sha256": hashlib.sha256(np.asarray(target, dtype="<c16").tobytes()).hexdigest(),
        "config": asdict(config),
        "method": method,
        "target_id": target_id,
        "replica": replica,
        "calibration": Calibration().content_checksum,
    })


def _validate_state(state: dict[str, Any], initial: NDArray[np.float64], config: Config, method: str) -> None:
    """Reject inconsistent optimizer, anchor and logical work state.

    Raises:
        ValueError: If saved arrays, counters or the anchor epoch are invalid.
    """
    iteration = state["iterations"]
    arrays = [np.asarray(state[key]) for key in ("theta", "first", "second")]
    anchors = (iteration + config.refresh_every - 1) // config.refresh_every if method == "cv_adam" else 0
    paths = 2 * config.correction_paths * iteration if method != "exact_adam" else 0
    expected = {
        "exact_gradient_calls": iteration if method == "exact_adam" else anchors,
        "exact_anchor_calls": anchors,
        "trajectory_gradient_calls": (2 if method == "cv_adam" else 1) * iteration if paths else 0,
        "trajectory_paths": paths,
        "trajectory_direction_passes": 2 * paths,
        "density_gradient_direction_passes": 2 * (iteration if method == "exact_adam" else anchors),
    }
    valid = (
        type(iteration) is int
        and 0 <= iteration <= config.updates
        and all(array.shape == initial.shape and np.all(np.isfinite(array)) for array in arrays)
        and np.all(arrays[2] >= 0)
        and len(state["audit"]) == iteration
        and all(type(state[key]) is int and state[key] == value for key, value in expected.items())
        and all(math.isfinite(state[key]) and state[key] >= 0 for key in new_attempt() if key.endswith("seconds"))
    )
    if method == "cv_adam" and iteration:
        valid = valid and state["anchor"] is not None
        if state["anchor"] is not None:
            valid = valid and state["anchor"]["created_before_update"] == 1 + (anchors - 1) * config.refresh_every
            decode_snapshot(state["anchor"], initial.shape)
    else:
        valid = valid and state["anchor"] is None
    if not valid:
        msg = "Invalid saved SVRG optimizer state or work counters."
        raise ValueError(msg)


def _point(
    circuit: ParameterizedCircuit,
    target: NDArray[np.complex128],
    state: dict[str, Any],
    attempt: dict[str, Any],
) -> None:
    """Charge a distinct exact forward diagnostic for a final-iterate CPU curve."""
    cpu = time.process_time()
    _increment(state, attempt, diagnostic_forward_calls=1, diagnostic_direction_passes=1)
    try:
        fidelity = exact_density_fidelity(circuit, np.asarray(state["theta"]), target, ExactLocalPauliProvider())
    finally:
        elapsed = time.process_time() - cpu
        state["diagnostic_cpu_seconds"] += elapsed
        attempt["diagnostic_cpu_seconds"] += elapsed
    state["curve"].append({
        "iteration": state["iterations"],
        "theta": state["theta"].copy(),
        "fidelity": fidelity,
        "training_cpu_seconds": state["training_cpu_seconds"],
        "charged_cpu_seconds": state["training_cpu_seconds"] + state["diagnostic_cpu_seconds"],
        **{key: state[key] for key in COUNTERS},
    })


def _estimate(
    engine: StatevectorBackend,
    theta: NDArray[np.float64],
    config: Config,
    method: str,
    update: int,
    seed: int,
    state: dict[str, Any],
    attempt: dict[str, Any],
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Compute one charged estimator with replayable maps and anchor age."""
    calibration = Calibration()
    audit: dict[str, Any] = {"update": update, "seed": None, "anchor_distance": None, "anchor_age": None}
    if method == "exact_adam":
        _increment(state, attempt, exact_gradient_calls=1, density_gradient_direction_passes=2)
        gradient, loss, _ = density_loss_gradient(engine.circuit, theta, engine.target, calibration)
    else:
        if method == "cv_adam" and (update - 1) % config.refresh_every == 0:
            _increment(
                state, attempt, exact_gradient_calls=1, exact_anchor_calls=1, density_gradient_direction_passes=2
            )
            state["anchor"] = encode_snapshot(variance.build_snapshot(engine, theta, calibration), update)
        paths = config.correction_paths if method == "cv_adam" else 2 * config.correction_paths
        maps, weights = FactorSampler(engine.circuit, calibration).ensemble(paths, seed)
        audit.update({
            "seed": seed,
            "map_checksum": map_digest(maps),
            "paired_prefix_checksum": map_digest(maps[: config.correction_paths]),
            "weights": weights.tolist(),
        })
        calls = 2 if method == "cv_adam" else 1
        _increment(
            state,
            attempt,
            trajectory_gradient_calls=calls,
            trajectory_paths=calls * paths,
            trajectory_direction_passes=2 * calls * paths,
        )
        if method == "cv_adam":
            snapshot = decode_snapshot(state["anchor"], theta.shape)
            gradient, loss = variance.control_variate_from_maps(engine, theta, calibration, snapshot, maps, weights)
            audit["anchor_distance"] = float(np.linalg.norm(theta - snapshot.theta))
            audit["anchor_age"] = update - state["anchor"]["created_before_update"]
        else:
            gradient, loss, _ = engine.gradient(theta, maps, weights=weights)
    audit["loss_estimate_before_update"] = loss
    audit["gradient_norm"] = float(np.linalg.norm(gradient))
    return gradient, audit


def train(
    circuit: ParameterizedCircuit,
    initial: NDArray[np.float64],
    target: NDArray[np.complex128],
    config: Config,
    method: str,
    *,
    target_id: str,
    replica: int = 11,
    state: dict[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
    stop_after: int | None = None,
    attempt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train or exactly resume from complete accepted Adam updates.

    Args:
        circuit: Fixed supported circuit with the physical all-zero input.
        initial: Original parameters, also authenticated on resume.
        target: Fixed normalized complex target vector.
        config: Complete immutable optimizer and estimator configuration.
        method: cv_adam, weighted_adam or exact_adam.
        target_id: Stream identity; scientific wrapper limits development targets.
        replica: Construction identity, shared across paired methods.
        state: Full prior optimizer, anchor, stream audit and logical counters.
        checkpoint: Receives a private complete state after every accepted update.
        stop_after: Engineering interruption boundary, outside scientific settings.
        attempt: Optional external physical-attempt counter, retained on exception.

    Returns:
        Final iterate and complete resumable state; never selects a best checkpoint.

    Raises:
        ValueError: If initial settings or resumed state identities are invalid.
    """
    if (
        method not in METHODS
        or initial.shape != (circuit.num_params,)
        or not np.isrealobj(initial)
        or not np.all(np.isfinite(initial))
        or (stop_after is not None and (type(stop_after) is not int or not 0 <= stop_after <= config.updates))
    ):
        msg = "Invalid SVRG method, initial parameters or stopping boundary."
        raise ValueError(msg)
    signature = _signature(circuit, initial, target, config, method, target_id, replica)
    state = (
        copy.deepcopy(state)
        if state is not None
        else {
            "signature": signature,
            "theta": initial.tolist(),
            "first": np.zeros_like(initial).tolist(),
            "second": np.zeros_like(initial).tolist(),
            "iterations": 0,
            "anchor": None,
            "audit": [],
            "curve": [],
            "status": "running",
            **new_attempt(),
        }
    )
    if state["signature"] != signature:
        msg = "Saved SVRG problem or protocol signature changed."
        raise ValueError(msg)
    _validate_state(state, initial, config, method)
    limit = config.updates if stop_after is None else stop_after
    if limit < state["iterations"]:
        msg = "Cannot resume before the saved accepted update."
        raise ValueError(msg)
    if state["iterations"] == config.updates:
        return state
    attempt = new_attempt() if attempt is None else attempt
    cpu, wall = time.process_time(), time.perf_counter()
    engine = StatevectorBackend(circuit, target)
    elapsed_cpu, elapsed_wall = time.process_time() - cpu, time.perf_counter() - wall
    for ledger in (state, attempt):
        ledger["training_cpu_seconds"] += elapsed_cpu
        ledger["training_wall_seconds"] += elapsed_wall
    if not state["curve"]:
        _point(circuit, target, state, attempt)
    theta, first, second = (np.asarray(state[key], dtype=float) for key in ("theta", "first", "second"))
    adam_config = adam.AdamConfig(updates=config.updates, learning_rate=config.learning_rate)
    for update in range(state["iterations"] + 1, limit + 1):
        cpu, wall = time.process_time(), time.perf_counter()
        try:
            gradient, audit = _estimate(
                engine, theta, config, method, update, stream_seed(target_id, replica, update), state, attempt
            )
            theta, first, second = adam.adam_step(theta, gradient, first, second, update, adam_config)
        finally:
            elapsed_cpu, elapsed_wall = time.process_time() - cpu, time.perf_counter() - wall
            for ledger in (state, attempt):
                ledger["training_cpu_seconds"] += elapsed_cpu
                ledger["training_wall_seconds"] += elapsed_wall
        state.update({
            "theta": theta.tolist(),
            "first": first.tolist(),
            "second": second.tolist(),
            "iterations": update,
        })
        state["audit"].append(audit)
        if update % config.checkpoint_every == 0 or update == config.updates:
            _point(circuit, target, state, attempt)
        state["status"] = "complete" if update == config.updates else "running"
        if checkpoint:
            checkpoint(copy.deepcopy(state))
    return state


def source_hashes() -> dict[str, str]:
    """Freeze owned sources and every imported numerical dependency."""
    hashes = runner.source_hashes()
    paths = [Path(__file__), ROOT / "variance.py", ROOT / "SVRG_PROTOCOL.md"]
    paths.extend((REPO / "src/mqt/yaqs").rglob("*.py"))
    hashes.update({str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    return dict(sorted(hashes.items()))


def freeze_protocol(output: Path, starts_path: Path = OUTPUT / "starts.json") -> dict[str, Any]:
    """Authenticate original starts and freeze the single production screen.

    This function performs no simulation, optimization or final evaluation.

    Raises:
        ValueError: If any source, retained protocol or production start differs.
    """
    manifest = runner.custody()
    sources = cast("list[dict[str, Any]]", pilot.read_record(starts_path)["starts"])
    selected = sorted(
        (row for row in sources if row["replica"] == 11 and row["target_id"] in TARGETS),
        key=operator.itemgetter("target_id"),
    )
    if len(selected) != len(TARGETS) or {row["target_id"] for row in selected} != set(TARGETS):
        msg = "Exactly the four original development starts at replica 11 are required."
        raise ValueError(msg)
    for row in selected:
        circuit = pilot.restore_circuit(row["candidate"]["circuit"])
        if circuit.num_qubits != 6 or len(circuit.gates) != 108:
            msg = "SVRG production requires the original q6 108-gate starts."
            raise ValueError(msg)
    config = Config()
    seeds = {target: [stream_seed(target, 11, update) for update in range(1, 151)] for target in TARGETS}
    ordered = sorted(seed for values in seeds.values() for seed in values)
    if any(right - left < 2 * config.correction_paths for left, right in itertools.pairwise(ordered)):
        msg = "Per-path training seed intervals overlap."
        raise ValueError(msg)
    protocol = {
        "schema": "yaqs.overnight.svrg_adam.v1",
        "phase": "development",
        "config": asdict(config),
        "methods": list(METHODS),
        "targets": list(TARGETS),
        "replica": 11,
        "starts": selected,
        "starts_path": str(starts_path.resolve()),
        "starts_sha256": hashlib.sha256(starts_path.read_bytes()).hexdigest(),
        "benchmark_hashes": manifest["benchmark_hashes"],
        "source_hashes": source_hashes(),
        "training_seeds": seeds,
        "refresh_before_updates": list(range(1, 151, 5)),
        "training_channel": "depolarizing_1s_all_calibrated_exact",
        "final_channels": list(channels.NOISE_IDS),
        "endpoint_rule": "final_accepted_update_150_no_best_checkpoint",
        "pairing": "same first 15 conditional maps plus deterministic path; weighted control has 31 conditional maps",
        "snapshot_cost": "30 additional exact joint gradients, fully charged to CV",
        "evaluation_randomness": "none: separate exact deterministic forward evaluations after all training",
        "timing_scope": "numerical setup, sampling, map hashing, anchor validation and Adam; excludes checkpoint I/O",
        "diagnostics": "distinct exact forward local score at initial and every 10 updates, charged to each arm",
        "checkpoint": "every 10 accepted updates; complete anchor, Adam moments, maps/checksums and counters",
        "promotion": "See SVRG_PROTOCOL.md; local mechanism and full-family performance assessed separately",
        "followup": "refresh 10 may only be separately preregistered after refresh 5 passes; no initial grid",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "PROTOCOL.json"
    if path.exists():
        if pilot.read_record(path) != protocol:
            msg = "Frozen SVRG protocol differs from current sources or starts."
            raise ValueError(msg)
    else:
        pilot.write_record(path, protocol)
    return protocol


def verify_protocol(protocol: dict[str, Any]) -> None:
    """Reject changed benchmark, numerical sources or original start artifacts.

    Raises:
        ValueError: If any authenticated source differs from the frozen protocol.
    """
    manifest = runner.custody()
    sources = cast("list[dict[str, Any]]", pilot.read_record(Path(protocol["starts_path"]))["starts"])
    selected = sorted(
        (row for row in sources if row["replica"] == 11 and row["target_id"] in TARGETS),
        key=operator.itemgetter("target_id"),
    )
    if (
        protocol["benchmark_hashes"] != manifest["benchmark_hashes"]
        or protocol["source_hashes"] != source_hashes()
        or protocol["starts_sha256"] != hashlib.sha256(Path(protocol["starts_path"]).read_bytes()).hexdigest()
        or protocol["starts"] != selected
        or protocol["config"] != asdict(Config())
        or protocol["methods"] != list(METHODS)
        or protocol["final_channels"] != list(channels.NOISE_IDS)
    ):
        msg = "SVRG benchmark, numerical source or original-start custody mismatch."
        raise ValueError(msg)


def _finished_record(
    circuit: ParameterizedCircuit,
    target: NDArray[np.complex128],
    state: dict[str, Any],
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    """Independently score and charge every final channel, including failed calls.

    Raises:
        FloatingPointError: If distinct local final scores disagree.
    """
    theta = np.asarray(state["theta"])
    fidelities = {}
    evaluation_cpu = time.process_time()
    try:
        for noise in channels.NOISE_IDS:
            telemetry["evaluation_forward_calls"] += 1
            fidelities[noise] = channels.evaluate(circuit, theta, target, noise)
    finally:
        telemetry["evaluation_cpu_seconds"] += time.process_time() - evaluation_cpu
    if abs(fidelities["depolarizing_1s_all"] - state["curve"][-1]["fidelity"]) > 1e-10:
        msg = "Distinct final fanout and diagnostic local score disagree."
        raise FloatingPointError(msg)
    return {
        "state": state,
        "fidelities": fidelities,
        "evaluation_cpu_seconds": telemetry["evaluation_cpu_seconds"],
        "final_forward_evaluations": len(channels.NOISE_IDS),
        "logical_counts": runner.counts(circuit),
        "ballarin_counts": runner.counts(materialize_ballarin_circuit(circuit, theta).circuit),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def execute_job(job: dict[str, Any]) -> dict[str, Any]:
    """Persist all complete, interrupted and failed attempts for one method.

    Completed scientific artifacts are read-only on resume. A caught interruption
    can replay unsaved updates; its actual attempted work remains in ATTEMPTS.
    An uncatchable process kill leaves an unfinished attempt and requires an
    external resource ledger before automatic resumption can proceed.

    Raises:
        ValueError: If a retained result differs or an unaccounted attempt exists.
    """
    protocol, source, method = job["protocol"], job["source"], job["method"]
    verify_protocol(protocol)
    root = Path(job["root"])
    root.mkdir(parents=True, exist_ok=True)
    signature = pilot.digest({"protocol": pilot.digest(protocol), "source": pilot.digest(source), "method": method})
    result_path, partial_path, attempts_path = (
        root / name for name in ("RESULT.json", "PARTIAL.json", "ATTEMPTS.json")
    )
    for path in (result_path, root / "FAILED.json"):
        if path.exists():
            record = cast("dict[str, Any]", pilot.read_record(path))
            if record["signature"] != signature:
                msg = "Retained SVRG result signature changed."
                raise ValueError(msg)
            if record["attempts_checksum"] != pilot.digest(pilot.read_record(attempts_path)):
                msg = "Retained SVRG physical-attempt ledger changed."
                raise ValueError(msg)
            return record
    prior = None
    if partial_path.exists():
        record = pilot.read_record(partial_path)
        if record["signature"] != signature:
            msg = "Retained SVRG checkpoint signature changed."
            raise ValueError(msg)
        prior = cast("dict[str, Any]", record["state"])
    attempts = (
        cast("list[dict[str, Any]]", pilot.read_record(attempts_path)["attempts"]) if attempts_path.exists() else []
    )
    if attempts and attempts[-1]["status"] == "running":
        msg = "An unaccounted killed SVRG attempt needs an external resource ledger before resume."
        raise ValueError(msg)
    telemetry = new_attempt()
    invocation: dict[str, Any] = {
        "status": "running",
        "counters": telemetry,
        "resumed_iteration": 0 if prior is None else prior["iterations"],
    }
    attempts.append(invocation)
    pilot.write_record(attempts_path, {"signature": signature, "attempts": attempts})
    circuit = pilot.restore_circuit(source["candidate"]["circuit"])
    initial = np.asarray(source["candidate"]["theta"], dtype=float)
    target = load_target(6, source["target_id"]).state_vector_copy()
    config = Config(**protocol["config"])

    def checkpoint(state: dict[str, Any]) -> None:
        if state["iterations"] % config.checkpoint_every == 0 or state["status"] == "complete":
            pilot.write_record(partial_path, {"signature": signature, "state": state})

    started_cpu, started_wall = time.process_time(), time.perf_counter()
    try:
        state = train(
            circuit,
            initial,
            target,
            config,
            method,
            target_id=source["target_id"],
            replica=source["replica"],
            state=prior,
            checkpoint=checkpoint,
            attempt=telemetry,
        )
        record = {
            "signature": signature,
            "status": "complete",
            "target_id": source["target_id"],
            "replica": source["replica"],
            "method": method,
            **_finished_record(circuit, target, state, telemetry),
        }
        invocation["status"] = "complete"
    except KeyboardInterrupt:
        invocation["status"] = "interrupted"
        raise
    except Exception as error:
        invocation["status"] = "failed"
        record = {
            "signature": signature,
            "status": "failed",
            "target_id": source["target_id"],
            "replica": source["replica"],
            "method": method,
            "failure": f"{type(error).__name__}: {error}",
        }
    finally:
        invocation["invocation_cpu_seconds_including_io"] = time.process_time() - started_cpu
        invocation["invocation_wall_seconds_including_io"] = time.perf_counter() - started_wall
        pilot.write_record(attempts_path, {"signature": signature, "attempts": attempts})
    record["physical_attempt_totals"] = {key: sum(row["counters"][key] for row in attempts) for key in telemetry}
    record["attempts_checksum"] = pilot.digest({"signature": signature, "attempts": attempts})
    verify_protocol(protocol)
    pilot.write_record(result_path if record["status"] == "complete" else root / "FAILED.json", record)
    return record


def write_scoreboard(output: Path, records: list[dict[str, Any]]) -> None:
    """Write every condition deterministically, retaining failed cells."""
    rows = []
    for result in sorted(records, key=operator.itemgetter("target_id", "replica", "method")):
        for noise in channels.NOISE_IDS:
            row = {key: result[key] for key in ("target_id", "replica", "method", "status")}
            row.update({"noise_id": noise, "failure": result.get("failure"), "run_checksum": pilot.digest(result)})
            if result["status"] == "complete":
                state = result["state"]
                logical = result["logical_counts"]
                lowered = result["ballarin_counts"] if noise == "ballarin_coupled" else logical
                row.update({
                    "fidelity": result["fidelities"][noise],
                    "noiseless_fidelity": result["fidelities"]["noiseless"],
                    "convention": "calibrated_exact",
                    "exactness": "deterministic_channel_expectation",
                    "logical_one_qubit": logical["one_qubit"],
                    "logical_two_qubit": logical["two_qubit"],
                    "lowered_one_qubit": lowered["one_qubit"],
                    "lowered_two_qubit": lowered["two_qubit"],
                    "logical_depth": logical["depth"],
                    "lowered_depth": lowered["depth"],
                    "parameters": len(state["theta"]),
                    **{
                        key: state[key]
                        for key in (*COUNTERS, "iterations", "training_cpu_seconds", "diagnostic_cpu_seconds")
                    },
                    "evaluation_cpu_seconds": result["evaluation_cpu_seconds"],
                    "physical_attempt_cpu_seconds": result["physical_attempt_totals"]["training_cpu_seconds"],
                })
            rows.append(row)
    if rows:
        leading = ("target_id", "replica", "method", "noise_id", "status")
        fields = (*leading, *sorted({key for row in rows for key in row} - set(leading)))
        with (output / "SCOREBOARD.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare only, or execute an existing immutable protocol with bounded workers."""
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
        print(f"Prepared only: {args.output / 'PROTOCOL.json'}", flush=True)
        return 0
    protocol = cast("dict[str, Any]", pilot.read_record(args.output / "PROTOCOL.json"))
    verify_protocol(protocol)
    jobs = [
        {
            "protocol": protocol,
            "source": source,
            "method": method,
            "root": str(args.output / f"{source['target_id']}__11__{method}"),
        }
        for source in protocol["starts"]
        for method in METHODS
    ]
    np.random.default_rng(runner.seed_for("development_diagnostics", "svrg_job_order", 11)).shuffle(jobs)
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(execute_job, jobs):
            records.append(record)
            write_scoreboard(args.output, records)
            print(record["target_id"], record["method"], record["status"], flush=True)
    records.sort(key=operator.itemgetter("target_id", "replica", "method"))
    pilot.write_record(
        args.output / "RESULT.json",
        {
            "status": "complete" if all(record["status"] == "complete" for record in records) else "contains_failures",
            "protocol_checksum": pilot.digest(protocol),
            "runs": [pilot.digest(record) for record in records],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
