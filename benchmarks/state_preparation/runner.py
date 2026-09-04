# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Minimal train-once/evaluate-many state-preparation benchmark runner."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from mqt.yaqs.optimization import (
    KrotovTJMOptions,
    ParameterizedCircuit,
    create_brickwall_matrix_product_disentangler_parameterized_circuit,
    noisy_state_preparation_metrics,
    state_preparation_metrics,
)

from .ballarin import BallarinNoiseProvider, materialize_ballarin_circuit
from .constants import (
    BALLARIN_NOISE_ID,
    MINIMUM_NOISE_IDS,
    NOISE_IDS,
    NOISELESS_NOISE_ID,
    STANDARD_NOISE_IDS,
    SUPPORTED_QUBIT_COUNTS,
    TARGET_IDS,
)
from .methods import DEFAULT_METHODS, StatePreparationMethod, TrainingOutcome, TrainingSettings
from .noise import StandardNoiseProvider
from .statistics import circuit_statistics, fidelity_estimate
from .targets import TargetCollection, TargetRecord, load_target_collection

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

RUNNER_FORMAT = "yaqs.state_preparation_benchmark.v1"
RESULTS_NAME = "results.jsonl"
MANIFEST_NAME = "manifest.json"
CHECKPOINT_DIRECTORY = "checkpoints"
_UINT64_MODULUS = 2**64


class BenchmarkError(RuntimeError):
    """Raised when a benchmark configuration or retained artifact is invalid."""


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Complete, directly executable benchmark configuration."""

    preset: str
    output_directory: Path
    num_qubits: tuple[int, ...]
    target_ids: tuple[str, ...]
    noise_ids: tuple[str, ...]
    method_ids: tuple[str, ...] = ("krotov_batch",)
    bmpd_depths: tuple[int, ...] = (1,)
    initialization_seeds: tuple[int, ...] = (11,)
    optimizer_iterations: int = 100
    optimizer_step_size: float = 0.2
    test_trajectories: int = 100
    tjm_dt: float = 1.0
    initial_single_qubit_layer: bool = True
    resume: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        """Validate identifiers, budgets, and mutually exclusive actions.

        Raises:
            ValueError: If any identifier, budget, seed, or action is invalid.
        """
        if not self.preset or self.preset != self.preset.strip():
            msg = "preset must be a nonempty identifier."
            raise ValueError(msg)
        _validate_selection(self.num_qubits, SUPPORTED_QUBIT_COUNTS, "num_qubits")
        _validate_selection(self.target_ids, TARGET_IDS, "target_ids")
        _validate_selection(self.noise_ids, NOISE_IDS, "noise_ids")
        if not self.method_ids or len(set(self.method_ids)) != len(self.method_ids):
            msg = "method_ids must be nonempty and unique."
            raise ValueError(msg)
        if any(not method_id or method_id != method_id.strip() for method_id in self.method_ids):
            msg = "method_ids must contain nonempty identifiers without surrounding whitespace."
            raise ValueError(msg)
        if not self.bmpd_depths or len(set(self.bmpd_depths)) != len(self.bmpd_depths):
            msg = "bmpd_depths must be nonempty and unique."
            raise ValueError(msg)
        if any(type(depth) is not int or depth < 0 for depth in self.bmpd_depths):
            msg = "bmpd_depths must contain nonnegative integers."
            raise ValueError(msg)
        if not self.initialization_seeds or len(set(self.initialization_seeds)) != len(self.initialization_seeds):
            msg = "initialization_seeds must be nonempty and unique."
            raise ValueError(msg)
        if any(type(seed) is not int or seed < 0 or seed >= _UINT64_MODULUS for seed in self.initialization_seeds):
            msg = "initialization_seeds must contain unsigned 64-bit integers."
            raise ValueError(msg)
        if type(self.optimizer_iterations) is not int or self.optimizer_iterations < 0:
            msg = "optimizer_iterations must be a nonnegative integer."
            raise ValueError(msg)
        if not np.isfinite(self.optimizer_step_size) or self.optimizer_step_size <= 0.0:
            msg = "optimizer_step_size must be positive and finite."
            raise ValueError(msg)
        if type(self.test_trajectories) is not int or self.test_trajectories < 1:
            msg = "test_trajectories must be a positive integer."
            raise ValueError(msg)
        if not np.isfinite(self.tjm_dt) or self.tjm_dt <= 0.0:
            msg = "tjm_dt must be positive and finite."
            raise ValueError(msg)
        if self.resume and self.overwrite:
            msg = "resume and overwrite are mutually exclusive."
            raise ValueError(msg)

    @property
    def result_cells(self) -> int:
        """Return the number of result rows in the expanded matrix."""
        return (
            len(self.num_qubits)
            * len(self.target_ids)
            * len(self.noise_ids)
            * len(self.method_ids)
            * len(self.bmpd_depths)
            * len(self.initialization_seeds)
        )

    @property
    def training_runs(self) -> int:
        """Return the number of train-once optimization jobs."""
        return self.result_cells // len(self.noise_ids)

    def manifest_dict(self) -> dict[str, object]:
        """Return the reproducibility-relevant configuration."""
        return {
            "bmpd_depths": list(self.bmpd_depths),
            "initial_single_qubit_layer": self.initial_single_qubit_layer,
            "initialization_seeds": list(self.initialization_seeds),
            "method_ids": list(self.method_ids),
            "noise_ids": list(self.noise_ids),
            "num_qubits": list(self.num_qubits),
            "optimizer_iterations": self.optimizer_iterations,
            "optimizer_step_size": self.optimizer_step_size,
            "preset": self.preset,
            "target_ids": list(self.target_ids),
            "test_trajectories": self.test_trajectories,
            "tjm_dt": self.tjm_dt,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Execution counts and output paths from one runner invocation."""

    expected_rows: int
    successful_rows: int
    failed_rows: int
    skipped_rows: int
    trained_runs: int
    reused_checkpoints: int
    results_path: Path


def _validate_selection(
    values: tuple[int | str, ...],
    supported: tuple[int | str, ...],
    name: str,
) -> None:
    """Validate one ordered, nonempty, duplicate-free identifier selection.

    Raises:
        ValueError: If the selection is empty, duplicated, or unsupported.
    """
    if not values or len(set(values)) != len(values):
        msg = f"{name} must be nonempty and unique."
        raise ValueError(msg)
    unsupported = [value for value in values if value not in supported]
    if unsupported:
        msg = f"{name} contains unsupported values {unsupported!r}."
        raise ValueError(msg)


def preset_config(preset: str, output_directory: Path | None = None) -> BenchmarkConfig:
    """Return one frozen smoke, minimum, or full configuration.

    Raises:
        ValueError: If the preset is unknown.
    """
    output = Path("state_preparation_results") / preset if output_directory is None else Path(output_directory)
    if preset == "smoke":
        return BenchmarkConfig(
            preset=preset,
            output_directory=output,
            num_qubits=(6,),
            target_ids=(TARGET_IDS[0],),
            noise_ids=NOISE_IDS,
            bmpd_depths=(0,),
            optimizer_iterations=0,
            test_trajectories=2,
        )
    if preset == "minimum":
        return BenchmarkConfig(
            preset=preset,
            output_directory=output,
            num_qubits=SUPPORTED_QUBIT_COUNTS,
            target_ids=TARGET_IDS,
            noise_ids=MINIMUM_NOISE_IDS,
        )
    if preset == "full":
        return BenchmarkConfig(
            preset=preset,
            output_directory=output,
            num_qubits=SUPPORTED_QUBIT_COUNTS,
            target_ids=TARGET_IDS,
            noise_ids=NOISE_IDS,
        )
    msg = f"Unknown preset {preset!r}; expected smoke, minimum, or full."
    raise ValueError(msg)


def _canonical_json(value: object) -> str:
    """Return a canonical serialization of JSON-compatible data."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _checksum(value: object) -> str:
    """Return a checksum of canonical JSON data."""
    return f"sha256:{hashlib.sha256(_canonical_json(value).encode()).hexdigest()}"


def derive_seed(root: int, *components: object) -> int:
    """Return an execution-order-independent unsigned 64-bit seed."""
    digest = hashlib.sha256(_canonical_json([root, *components]).encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _run_identity(
    config: BenchmarkConfig,
    method_id: str,
    target: TargetRecord,
    depth: int,
    initialization_seed: int,
    noise_id: str,
) -> dict[str, object]:
    """Return all fields that uniquely identify one result row."""
    return {
        "bmpd_depth": depth,
        "initial_single_qubit_layer": config.initial_single_qubit_layer,
        "initialization_seed": initialization_seed,
        "method": method_id,
        "noise_id": noise_id,
        "num_qubits": target.num_qubits,
        "optimizer_iterations": config.optimizer_iterations,
        "optimizer_step_size": config.optimizer_step_size,
        "target_id": target.target_id,
        "test_trajectories": 0 if noise_id == NOISELESS_NOISE_ID else config.test_trajectories,
        "tjm_dt": config.tjm_dt,
    }


def _training_identity(
    config: BenchmarkConfig,
    method_id: str,
    target: TargetRecord,
    depth: int,
    initialization_seed: int,
) -> dict[str, object]:
    """Return all fields that uniquely identify one shared training run."""
    identity = _run_identity(config, method_id, target, depth, initialization_seed, NOISELESS_NOISE_ID)
    identity.pop("noise_id")
    identity.pop("test_trajectories")
    identity.pop("tjm_dt")
    return identity


def _atomic_write(path: Path, text: str) -> None:
    """Replace a UTF-8 text file atomically within its directory."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


class _ResultStore:
    """Small atomic JSONL store keyed by deterministic run identifier."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[str, dict[str, object]] = {}
        if not path.exists():
            return
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                msg = f"Invalid JSON in {path} at line {line_number}."
                raise BenchmarkError(msg) from error
            if not isinstance(row, dict) or not isinstance(row.get("run_id"), str):
                msg = f"Invalid result record in {path} at line {line_number}."
                raise BenchmarkError(msg)
            run_id = cast("str", row["run_id"])
            if run_id in self.rows:
                msg = f"Duplicate run_id {run_id!r} in {path}."
                raise BenchmarkError(msg)
            self.rows[run_id] = cast("dict[str, object]", row)

    def upsert(self, row: dict[str, object]) -> None:
        """Persist one new or replacement row atomically."""
        run_id = cast("str", row["run_id"])
        self.rows[run_id] = row
        encoded = "".join(f"{json.dumps(value, sort_keys=True, allow_nan=False)}\n" for value in self.rows.values())
        _atomic_write(self.path, encoded)


def _prepare_output(config: BenchmarkConfig, targets: TargetCollection) -> _ResultStore:
    """Return a store after creating or validating owned output files.

    Raises:
        BenchmarkError: If retained output is invalid or incompatible.
    """
    output = config.output_directory
    manifest_path = output / MANIFEST_NAME
    results_path = output / RESULTS_NAME
    checkpoints = output / CHECKPOINT_DIRECTORY
    owned_output_exists = manifest_path.exists() or results_path.exists() or checkpoints.exists()
    if config.overwrite:
        for path in (manifest_path, results_path):
            path.unlink(missing_ok=True)
        if checkpoints.exists():
            shutil.rmtree(checkpoints)
    elif config.resume and not manifest_path.exists():
        msg = f"Output {output} has no retained benchmark manifest to resume."
        raise BenchmarkError(msg)
    elif owned_output_exists and not config.resume:
        msg = f"Output {output} already contains benchmark-owned artifacts; pass resume or overwrite."
        raise BenchmarkError(msg)

    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(exist_ok=True)
    manifest = {
        "config": config.manifest_dict(),
        "format": RUNNER_FORMAT,
        "result_cells": config.result_cells,
        "target_fixture_checksum": targets.checksum,
        "training_runs": config.training_runs,
    }
    if manifest_path.exists():
        try:
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            msg = f"Could not read retained manifest {manifest_path}."
            raise BenchmarkError(msg) from error
        if stored != manifest:
            msg = f"Retained manifest {manifest_path} does not match the requested benchmark."
            raise BenchmarkError(msg)
    else:
        _atomic_write(manifest_path, f"{json.dumps(manifest, indent=2, sort_keys=True)}\n")
    return _ResultStore(results_path)


def _checkpoint_path(config: BenchmarkConfig, identity: Mapping[str, object]) -> Path:
    """Return the content-addressed checkpoint path for one training run."""
    return config.output_directory / CHECKPOINT_DIRECTORY / f"{_checksum(identity).removeprefix('sha256:')}.npz"


def _save_checkpoint(path: Path, identity: Mapping[str, object], outcome: TrainingOutcome) -> None:
    """Write a validated optimizer checkpoint."""
    metadata = {
        "format": RUNNER_FORMAT,
        "identity": dict(identity),
        "trace": {key: list(values) for key, values in outcome.trace.items()},
        "wall_time_seconds": outcome.wall_time_seconds,
    }
    np.savez_compressed(path, theta=outcome.theta, metadata=np.asarray(_canonical_json(metadata)))


def _load_checkpoint(
    path: Path,
    identity: Mapping[str, object],
    num_parameters: int,
) -> TrainingOutcome | None:
    """Return a matching checkpoint, or None when it does not exist.

    Raises:
        BenchmarkError: If the retained checkpoint is invalid or incompatible.
    """
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            theta = np.asarray(payload["theta"], dtype=np.float64)
            metadata = json.loads(str(payload["metadata"].item()))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        msg = f"Could not load benchmark checkpoint {path}."
        raise BenchmarkError(msg) from error
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != RUNNER_FORMAT
        or metadata.get("identity") != dict(identity)
        or theta.shape != (num_parameters,)
        or not np.all(np.isfinite(theta))
    ):
        msg = f"Benchmark checkpoint {path} does not match its training identity."
        raise BenchmarkError(msg)
    raw_trace = metadata.get("trace", {})
    if not isinstance(raw_trace, dict) or any(not isinstance(value, list) for value in raw_trace.values()):
        msg = f"Benchmark checkpoint {path} contains an invalid optimizer trace."
        raise BenchmarkError(msg)
    trace = {str(key): tuple(cast("list[float | int | str]", value)) for key, value in raw_trace.items()}
    wall_time = metadata.get("wall_time_seconds")
    if not isinstance(wall_time, (int, float)) or not np.isfinite(wall_time) or wall_time < 0.0:
        msg = f"Benchmark checkpoint {path} contains an invalid wall time."
        raise BenchmarkError(msg)
    return TrainingOutcome(theta, trace, float(wall_time))


def _evaluate_noise(
    config: BenchmarkConfig,
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: TargetRecord,
    noise_id: str,
    test_seed: int,
) -> tuple[float, tuple[float, ...], dict[str, object]]:
    """Return metrics for one noiseless or noisy condition.

    Raises:
        ValueError: If the noise identifier is unsupported.
    """
    if noise_id == NOISELESS_NOISE_ID:
        _loss, fidelity = state_preparation_metrics(circuit, theta, target.state_vector)
        return fidelity, (), circuit_statistics(circuit).to_dict()

    options = KrotovTJMOptions(
        num_trajectories=config.test_trajectories,
        random_seed=test_seed,
        dt=config.tjm_dt,
        apply_noise_to="all",
    )
    if noise_id == BALLARIN_NOISE_ID:
        materialized = materialize_ballarin_circuit(circuit, theta)
        _loss, mean, values = noisy_state_preparation_metrics(
            materialized.circuit,
            np.empty(0, dtype=np.float64),
            target.state_vector,
            None,
            options,
            noise_provider=BallarinNoiseProvider(),
        )
        return mean, tuple(values), circuit_statistics(circuit, materialized).to_dict()
    if noise_id not in STANDARD_NOISE_IDS:
        msg = f"Unsupported noise identifier {noise_id!r}."
        raise ValueError(msg)
    _loss, mean, values = noisy_state_preparation_metrics(
        circuit,
        theta,
        target.state_vector,
        None,
        options,
        noise_provider=StandardNoiseProvider(noise_id),
    )
    return mean, tuple(values), circuit_statistics(circuit).to_dict()


def _success_row(
    *,
    config: BenchmarkConfig,
    identity: dict[str, object],
    target: TargetRecord,
    outcome: TrainingOutcome,
    train_fidelity: float,
    noiseless_fidelity: float,
    noisy_fidelity: float,
    trajectory_fidelities: tuple[float, ...],
    test_seed: int | None,
    circuit_data: dict[str, object],
    checkpoint_path: Path,
    evaluation_wall_time: float,
) -> dict[str, object]:
    """Return one complete successful result record."""
    noise_id = cast("str", identity["noise_id"])
    estimate = None if noise_id == NOISELESS_NOISE_ID else fidelity_estimate(trajectory_fidelities)
    logical_counts = cast("dict[str, object]", circuit_data["logical"])
    native_counts = cast("dict[str, object] | None", circuit_data["native"])
    confidence_interval = None
    if estimate is not None and estimate.confidence_interval_lower is not None:
        confidence_interval = {
            "level": 0.95,
            "lower": estimate.confidence_interval_lower,
            "upper": estimate.confidence_interval_upper,
        }
    optimizer_budget = {
        "iterations": config.optimizer_iterations,
        "optimizer_seed": (cast("int", identity["initialization_seed"]) + 1) % _UINT64_MODULUS,
        "schedule": "constant",
        "step_size": config.optimizer_step_size,
    }
    return {
        **identity,
        "ansatz": {
            "bmpd_depth": identity["bmpd_depth"],
            "brickwall_layers": 2 * cast("int", identity["bmpd_depth"]),
            "initial_single_qubit_layer": config.initial_single_qubit_layer,
        },
        "checkpoint": str(checkpoint_path.relative_to(config.output_directory)),
        "circuit_statistics": circuit_data,
        "confidence_interval": confidence_interval,
        "evaluation_wall_time_seconds": evaluation_wall_time,
        "format": RUNNER_FORMAT,
        "initialization": {
            "distribution": "normal",
            "mean": 0.0,
            "seed": identity["initialization_seed"],
            "standard_deviation": 0.1,
        },
        "noise_seed": test_seed,
        "noisy_fidelity_standard_deviation": None if estimate is None else estimate.standard_deviation,
        "noisy_fidelity_standard_error": None if estimate is None else estimate.standard_error,
        "notes": [],
        "num_parameters": circuit_data["num_parameters"],
        "logical_gate_counts": logical_counts,
        "native_gate_counts": native_counts,
        "optimizer": optimizer_budget,
        "optimizer_budget": optimizer_budget,
        "run_id": _checksum(identity),
        "status": "success",
        "target_generation_seed": target.seed,
        "test_noiseless_fidelity": noiseless_fidelity,
        "test_noisy_fidelity": noisy_fidelity,
        "test_trajectories_or_shots": 0 if noise_id == NOISELESS_NOISE_ID else config.test_trajectories,
        "train_fidelity": train_fidelity,
        "train_trajectories_or_shots": 0,
        "training_wall_time_seconds": outcome.wall_time_seconds,
        "trajectory_fidelities": list(trajectory_fidelities),
        "wall_time_seconds": outcome.wall_time_seconds + evaluation_wall_time,
    }


def _failure_row(identity: dict[str, object], error: Exception, phase: str) -> dict[str, object]:
    """Return a stable failed result record."""
    return {
        **identity,
        "error": f"{type(error).__name__}: {error}",
        "failure_phase": phase,
        "format": RUNNER_FORMAT,
        "run_id": _checksum(identity),
        "status": "failed",
    }


def _train_or_load(
    config: BenchmarkConfig,
    method: StatePreparationMethod,
    circuit: ParameterizedCircuit,
    target: TargetRecord,
    initialization_seed: int,
    training_identity: Mapping[str, object],
    checkpoint_path: Path,
) -> tuple[TrainingOutcome, bool, float]:
    """Return a trained or restored outcome, reuse flag, and fidelity.

    Raises:
        BenchmarkError: If a method returns an invalid parameter vector.
    """
    outcome = _load_checkpoint(checkpoint_path, training_identity, circuit.num_params)
    reused = outcome is not None
    if outcome is None:
        rng = np.random.default_rng(initialization_seed)
        initial_theta = rng.normal(0.0, 0.1, size=circuit.num_params)
        outcome = method.train(
            circuit,
            target.state_vector,
            initial_theta,
            TrainingSettings(
                iterations=config.optimizer_iterations,
                step_size=config.optimizer_step_size,
                optimizer_seed=(initialization_seed + 1) % _UINT64_MODULUS,
            ),
        )
        if outcome.theta.shape != (circuit.num_params,) or not np.all(np.isfinite(outcome.theta)):
            msg = f"Method {method.method_id!r} returned an invalid parameter vector."
            raise BenchmarkError(msg)
        _save_checkpoint(checkpoint_path, training_identity, outcome)
    _loss, train_fidelity = state_preparation_metrics(circuit, outcome.theta, target.state_vector)
    return outcome, reused, train_fidelity


def run_benchmark(
    config: BenchmarkConfig,
    *,
    methods: Mapping[str, StatePreparationMethod] | None = None,
    targets: TargetCollection | None = None,
    fail_fast: bool = False,
) -> BenchmarkSummary:
    """Execute a benchmark configuration with shared training across noise rows.

    Args:
        config: Fully resolved benchmark configuration.
        methods: Optional adapter registry. This is the extension point for
            algorithms other than the bundled Krotov reference.
        targets: Optional preloaded target collection.
        fail_fast: Whether to re-raise the first training or evaluation error.

    Returns:
        Execution summary and result path.

    Raises:
        BenchmarkError: If method registration or retained artifacts are invalid.
    """
    method_registry = DEFAULT_METHODS if methods is None else dict(methods)
    missing_methods = [method_id for method_id in config.method_ids if method_id not in method_registry]
    if missing_methods:
        msg = f"No method adapters registered for {missing_methods!r}."
        raise BenchmarkError(msg)
    target_collection = load_target_collection() if targets is None else targets
    store = _prepare_output(config, target_collection)
    selected_targets = target_collection.select(num_qubits=config.num_qubits, target_ids=config.target_ids)
    trained_runs = 0
    reused_checkpoints = 0
    skipped_rows = 0

    training_cells = itertools.product(
        config.method_ids,
        config.bmpd_depths,
        config.initialization_seeds,
        selected_targets,
    )
    for method_id, depth, initialization_seed, target in training_cells:
        method = method_registry[method_id]
        if method.method_id != method_id:
            msg = f"Method registry key {method_id!r} does not match adapter id {method.method_id!r}."
            raise BenchmarkError(msg)
        identities = {
            noise_id: _run_identity(config, method_id, target, depth, initialization_seed, noise_id)
            for noise_id in config.noise_ids
        }
        complete = {
            noise_id
            for noise_id, identity in identities.items()
            if (retained := store.rows.get(_checksum(identity))) is not None and retained.get("status") == "success"
        }
        skipped_rows += len(complete)
        pending = [noise_id for noise_id in config.noise_ids if noise_id not in complete]
        if not pending:
            continue

        circuit = create_brickwall_matrix_product_disentangler_parameterized_circuit(
            target.num_qubits,
            depth,
            initial_single_qubit_layer=config.initial_single_qubit_layer,
        )
        training_identity = _training_identity(config, method_id, target, depth, initialization_seed)
        checkpoint_path = _checkpoint_path(config, training_identity)
        try:
            outcome, reused, train_fidelity = _train_or_load(
                config,
                method,
                circuit,
                target,
                initialization_seed,
                training_identity,
                checkpoint_path,
            )
        except Exception as error:
            for noise_id in pending:
                store.upsert(_failure_row(identities[noise_id], error, "training"))
            if fail_fast:
                raise
            continue
        trained_runs += not reused
        reused_checkpoints += reused

        for noise_id in pending:
            identity = identities[noise_id]
            test_seed = None
            if noise_id != NOISELESS_NOISE_ID:
                root_seed = (initialization_seed + 2) % _UINT64_MODULUS
                test_seed = derive_seed(root_seed, "final-test", noise_id)
            started = time.perf_counter()
            try:
                noisy_fidelity, values, circuit_data = _evaluate_noise(
                    config,
                    circuit,
                    outcome.theta,
                    target,
                    noise_id,
                    0 if test_seed is None else test_seed,
                )
                row = _success_row(
                    config=config,
                    identity=identity,
                    target=target,
                    outcome=outcome,
                    train_fidelity=train_fidelity,
                    noiseless_fidelity=train_fidelity,
                    noisy_fidelity=noisy_fidelity,
                    trajectory_fidelities=values,
                    test_seed=test_seed,
                    circuit_data=circuit_data,
                    checkpoint_path=checkpoint_path,
                    evaluation_wall_time=time.perf_counter() - started,
                )
            except Exception as error:
                row = _failure_row(identity, error, "evaluation")
                if fail_fast:
                    store.upsert(row)
                    raise
            store.upsert(row)

    selected_run_ids = {
        _checksum(_run_identity(config, method_id, target, depth, seed, noise_id))
        for method_id in config.method_ids
        for depth in config.bmpd_depths
        for seed in config.initialization_seeds
        for target in selected_targets
        for noise_id in config.noise_ids
    }
    selected_rows = [row for run_id, row in store.rows.items() if run_id in selected_run_ids]
    successful = sum(row.get("status") == "success" for row in selected_rows)
    failed = sum(row.get("status") == "failed" for row in selected_rows)
    return BenchmarkSummary(
        expected_rows=config.result_cells,
        successful_rows=successful,
        failed_rows=failed,
        skipped_rows=skipped_rows,
        trained_runs=trained_runs,
        reused_checkpoints=reused_checkpoints,
        results_path=config.output_directory / RESULTS_NAME,
    )


def _argument_parser() -> argparse.ArgumentParser:
    """Return the benchmark command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "minimum", "full"), default="minimum")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-qubits", action="append", type=int)
    parser.add_argument("--target-id", action="append")
    parser.add_argument("--noise-id", action="append")
    parser.add_argument("--method", dest="method_ids", action="append")
    parser.add_argument("--bmpd-depth", action="append", type=int)
    parser.add_argument("--initialization-seed", action="append", type=int)
    parser.add_argument("--optimizer-iterations", type=int)
    parser.add_argument("--optimizer-step-size", type=float)
    parser.add_argument("--test-trajectories", type=int)
    parser.add_argument("--tjm-dt", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def resolve_arguments(arguments: Sequence[str] | None = None) -> tuple[BenchmarkConfig, bool, bool]:
    """Return command-line arguments resolved over one frozen preset."""
    args = _argument_parser().parse_args(arguments)
    config = preset_config(args.preset, args.output_dir)
    replacements: dict[str, object] = {"overwrite": args.overwrite, "resume": args.resume}
    for argument_name, field_name in (
        ("num_qubits", "num_qubits"),
        ("target_id", "target_ids"),
        ("noise_id", "noise_ids"),
        ("method_ids", "method_ids"),
        ("bmpd_depth", "bmpd_depths"),
        ("initialization_seed", "initialization_seeds"),
    ):
        value = getattr(args, argument_name)
        if value is not None:
            replacements[field_name] = tuple(value)
    for name in ("optimizer_iterations", "optimizer_step_size", "test_trajectories", "tjm_dt"):
        value = getattr(args, name)
        if value is not None:
            replacements[name] = value
    return replace(config, **replacements), bool(args.dry_run), bool(args.fail_fast)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the benchmark command-line interface.

    Returns:
        Zero when every selected cell succeeds, otherwise one.
    """
    config, dry_run, fail_fast = resolve_arguments(arguments)
    if dry_run:
        sys.stdout.write(
            f"{json.dumps({'config': config.manifest_dict(), 'result_cells': config.result_cells}, indent=2)}\n"
        )
        return 0
    summary = run_benchmark(config, fail_fast=fail_fast)
    summary_payload = {
        "expected_rows": summary.expected_rows,
        "failed_rows": summary.failed_rows,
        "results_path": str(summary.results_path),
        "reused_checkpoints": summary.reused_checkpoints,
        "skipped_rows": summary.skipped_rows,
        "successful_rows": summary.successful_rows,
        "trained_runs": summary.trained_runs,
    }
    sys.stdout.write(f"{json.dumps(summary_payload, indent=2)}\n")
    return 1 if summary.failed_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BenchmarkConfig",
    "BenchmarkError",
    "BenchmarkSummary",
    "derive_seed",
    "main",
    "preset_config",
    "resolve_arguments",
    "run_benchmark",
]
