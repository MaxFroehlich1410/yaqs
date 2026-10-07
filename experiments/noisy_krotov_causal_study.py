# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Canonical q6 causal study of noisy Krotov trajectory-sampling policies.

The experiment first creates one deterministic layerwise noiseless checkpoint.
Every replicate and every arm starts from that exact checkpoint.  Four controls
and treatments are then compared:

* the frozen pretrained circuit;
* an equally long noiseless continuation;
* noisy cross-trajectory updates using one fixed CRN ensemble;
* noisy cross-trajectory updates with periodically refreshed CRN ensembles; and
* noisy cross-trajectory updates with a fresh ensemble at every update.

Checkpoint selection uses a fixed validation ensemble that is disjoint from
both training and final evaluation.  Final evaluation uses fresh batches of
maps, with each batch replayed across all arms to produce paired differences.
The canonical standard-noise profile contains only Pauli jump operators, whose
jump probabilities are state independent, so this replay is an unbiased common-
random-number comparison for every circuit in the batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from functools import partial
from importlib.metadata import version
from pathlib import Path
from statistics import fmean, stdev
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from scipy.stats import t as student_t

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.execution import WorkerSetting, completed_jobs, parse_workers, resolve_workers
from benchmarks.state_preparation.noise import StandardNoiseProvider
from benchmarks.state_preparation.targets import load_target, load_target_collection
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KrotovFixedMapEnsemble,
    KrotovMapSamplingPolicy,
    KrotovMapSchedule,
    KrotovTJMOptions,
    KrotovTruncation,
    derive_krotov_trajectory_seed,
    noisy_state_preparation_contribution,
    noisy_state_preparation_metrics,
    sample_krotov_fixed_map_ensemble,
    state_preparation_contribution,
    state_preparation_metrics,
)
from mqt.yaqs.optimization.parameterized_circuit import (
    ParameterizedCircuit,
    create_brickwall_matrix_product_disentangler_parameterized_circuit,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import ArrayLike, NDArray


SCHEMA_VERSION = "yaqs.noisy_krotov.causal_study.v1"
CONFIG_NAME = "config.json"
RESULT_NAME = "causal_study_result.json"
PRETRAINING_DIRECTORY = "pretraining"
PRETRAINING_STATE_NAME = "state.npz"
PRETRAINED_CHECKPOINT_NAME = "pretrained_checkpoint.npz"
ARMS: tuple[str, ...] = (
    "frozen",
    "noiseless_continuation",
    "crn_fixed",
    "crn_refresh",
    "resampled",
)
NOISY_ARMS: tuple[str, ...] = ("crn_fixed", "crn_refresh", "resampled")
Arm = Literal["frozen", "noiseless_continuation", "crn_fixed", "crn_refresh", "resampled"]


class CausalStudyError(RuntimeError):
    """Raised when a retained study artifact is incomplete or inconsistent."""


def _canonical_json(value: object) -> str:
    """Serialize JSON-native content deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _checksum(value: object) -> str:
    """Return a SHA-256 checksum of canonical JSON content."""
    return f"sha256:{hashlib.sha256(_canonical_json(value).encode()).hexdigest()}"


def _array_checksum(value: NDArray[np.float64] | NDArray[np.complex128]) -> str:
    """Return a platform-independent checksum of one numeric vector."""
    dtype = "<c16" if np.iscomplexobj(value) else "<f8"
    payload = np.ascontiguousarray(value, dtype=dtype).tobytes()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _seal(value: Mapping[str, object]) -> dict[str, object]:
    """Return a detached JSON mapping carrying its content checksum."""
    payload = dict(value)
    payload["content_checksum"] = _checksum(payload)
    return payload


def _load_json(path: Path, *, sealed: bool = True) -> dict[str, object]:
    """Load one JSON object and optionally verify its checksum."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        msg = f"{path} must contain a JSON object."
        raise TypeError(msg)
    result = cast("dict[str, object]", value)
    if sealed:
        supplied = result.pop("content_checksum", None)
        computed = _checksum(result)
        result["content_checksum"] = supplied
        if supplied != computed:
            msg = f"{path} has an invalid content checksum."
            raise ValueError(msg)
    return result


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Atomically replace one file with exact bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: object) -> None:
    """Atomically write canonical JSON."""
    _atomic_bytes(path, f"{_canonical_json(value)}\n".encode())


def _atomic_npz(path: Path, **arrays: ArrayLike) -> None:
    """Atomically write a compressed NumPy archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, allow_pickle=False, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True, slots=True)
class StudyConfig:
    """Complete scientific configuration of the first canonical study."""

    qubits: int = 6
    target_id: str = "tfim_critical"
    depths: tuple[int, ...] = (1, 2, 3, 4)
    initialization_seed: int = 17
    initialization_scale: float = 0.1
    growth_initialization: float = 0.001
    pretraining_iterations_per_depth: int = 100
    pretraining_learning_rate: float = 0.2
    finetune_updates: int = 200
    finetune_learning_rate: float = 0.2
    finetune_learning_rate_decay: float = 0.01
    training_trajectories: int = 3
    refresh_interval: int = 5
    validation_interval: int = 5
    validation_trajectories: int = 12
    test_trajectories: int = 500
    test_batch_size: int = 50
    replicates: int = 3
    base_seed: int = 20260904
    noise_id: str = "depolarizing_1s_all"
    noise_calibration: Literal["legacy", "hardware_average_error"] = "legacy"

    def __post_init__(self) -> None:
        """Reject configurations outside the intended exact q6 study."""
        if self.qubits != 6 or self.target_id != "tfim_critical":
            msg = "The causal study intentionally supports only the canonical q6 tfim_critical target."
            raise ValueError(msg)
        if not self.depths or tuple(sorted(set(self.depths))) != self.depths or self.depths[-1] != 4:
            msg = "depths must be a strictly increasing tuple ending at depth four."
            raise ValueError(msg)
        integer_fields = (
            "initialization_seed",
            "pretraining_iterations_per_depth",
            "finetune_updates",
            "training_trajectories",
            "refresh_interval",
            "validation_interval",
            "validation_trajectories",
            "test_trajectories",
            "test_batch_size",
            "replicates",
            "base_seed",
        )
        for name in integer_fields:
            value = getattr(self, name)
            minimum = 0 if name == "initialization_seed" else 1
            if type(value) is not int or value < minimum or value >= 2**64:
                msg = f"{name} must be an integer in [{minimum}, 2**64)."
                raise ValueError(msg)
        if self.refresh_interval > self.finetune_updates or self.validation_interval > self.finetune_updates:
            msg = "refresh_interval and validation_interval cannot exceed finetune_updates."
            raise ValueError(msg)
        if self.test_batch_size > self.test_trajectories:
            msg = "test_batch_size cannot exceed test_trajectories."
            raise ValueError(msg)
        finite_fields = (
            "initialization_scale",
            "growth_initialization",
            "pretraining_learning_rate",
            "finetune_learning_rate",
            "finetune_learning_rate_decay",
        )
        if any(not math.isfinite(getattr(self, name)) for name in finite_fields):
            msg = "All floating-point settings must be finite."
            raise ValueError(msg)
        if self.initialization_scale <= 0.0 or self.pretraining_learning_rate <= 0.0:
            msg = "Initialization scale and pretraining learning rate must be positive."
            raise ValueError(msg)
        if self.finetune_learning_rate <= 0.0 or self.finetune_learning_rate_decay < 0.0:
            msg = "Fine-tuning learning rate must be positive and its decay nonnegative."
            raise ValueError(msg)
        StandardNoiseProvider(self.noise_id)
        if self.noise_calibration not in {"legacy", "hardware_average_error"}:
            msg = "Unknown noise calibration."
            raise ValueError(msg)
        if self.noise_calibration != "legacy" and self.noise_id != "depolarizing_1s_all":
            msg = "Hardware calibration currently supports only depolarizing_1s_all."
            raise ValueError(msg)

    def noise_provider(self) -> StandardNoiseProvider | CalibratedLocalDepolarizingProvider:
        """Resolve legacy or separately identified hardware-calibrated noise."""
        if self.noise_calibration == "hardware_average_error":
            return CalibratedLocalDepolarizingProvider()
        return StandardNoiseProvider(self.noise_id)

    def scientific_dict(self) -> dict[str, object]:
        """Return a JSON-native description of all scientific settings."""
        value = asdict(self)
        value["depths"] = list(self.depths)
        if self.noise_calibration == "legacy":
            # Preserve all existing legacy configuration checksums and seeds.
            value.pop("noise_calibration")
        else:
            value["noise_calibration_details"] = CalibratedLocalDepolarizingProvider().scientific_dict()
        return {"schema_version": SCHEMA_VERSION, **value}


def derive_study_seed(base_seed: int, role: str, replicate_index: int) -> int:
    """Derive an execution-order-independent uint64 seed for one study stream."""
    if type(base_seed) is not int or not 0 <= base_seed < 2**64:
        msg = "base_seed must be a uint64 integer."
        raise ValueError(msg)
    if type(replicate_index) is not int or not 0 <= replicate_index < 2**64:
        msg = "replicate_index must be a uint64 integer."
        raise ValueError(msg)
    if not role:
        msg = "role must be a nonempty string."
        raise ValueError(msg)
    digest = hashlib.sha256(_canonical_json([SCHEMA_VERSION, base_seed, role, replicate_index]).encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def create_circuit(qubits: int, depth: int) -> ParameterizedCircuit:
    """Create the canonical BMPD circuit at one layerwise depth."""
    return create_brickwall_matrix_product_disentangler_parameterized_circuit(
        qubits,
        depth,
        initial_single_qubit_layer=True,
    )


def _circuit_checksum(circuit: ParameterizedCircuit) -> str:
    """Hash the complete logical gate sequence used by the study."""
    return _checksum({
        "qubits": circuit.num_qubits,
        "parameters": circuit.num_params,
        "gates": [
            {
                "name": gate.name,
                "sites": list(gate.sites),
                "parameter": gate.param_index,
                "scale": gate.angle_scale,
                "offset": gate.angle_offset,
                "fixed": list(gate.fixed_params),
                "noise_enabled": gate.noise_enabled,
            }
            for gate in circuit.gates
        ],
    })


def _truncation() -> KrotovTruncation:
    """Return the exact MPS truncation configuration."""
    return KrotovTruncation(
        max_bond_dim=None,
        svd_threshold=0.0,
        trunc_mode="discarded_weight",
        min_bond_dim=1,
    )


def _trajectory_options(count: int, *, cross: bool) -> KrotovTJMOptions:
    """Create canonical all-gate TJM options."""
    return KrotovTJMOptions(
        num_trajectories=count,
        random_seed=0,
        dt=1.0,
        apply_noise_to="all",
        trajectory_update="cross" if cross else "independent",
        differentiate_jump_normalization=False,
        use_crn=False,
    )


def _prepare_root(output_root: Path, config_document: Mapping[str, object], *, resume: bool) -> Path:
    """Create a new output root or verify an existing one for exact resume."""
    root = output_root.resolve()
    if resume:
        if not root.is_dir():
            msg = "--resume requires an existing output directory."
            raise ValueError(msg)
        if _load_json(root / CONFIG_NAME) != dict(config_document):
            msg = "The active configuration differs from the retained run."
            raise ValueError(msg)
        return root
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        msg = "A new run requires a missing or empty output directory."
        raise ValueError(msg)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / CONFIG_NAME, config_document)
    return root


def _save_pretraining_state(
    path: Path,
    theta: NDArray[np.float64],
    completed_depths: int,
    stages: Sequence[Mapping[str, object]],
    elapsed_seconds: float,
) -> None:
    """Save one layer-boundary pretraining checkpoint."""
    _atomic_npz(
        path,
        theta=theta,
        completed_depths=np.asarray(completed_depths, dtype=np.int64),
        stages_json=_canonical_json(list(stages)).encode(),
        elapsed_seconds=np.asarray(elapsed_seconds, dtype=np.float64),
    )


def _load_pretraining_state(
    path: Path,
) -> tuple[NDArray[np.float64], int, list[dict[str, object]], float]:
    """Load one validated layer-boundary pretraining checkpoint."""
    with np.load(path, allow_pickle=False) as archive:
        theta = np.asarray(archive["theta"], dtype=np.float64).copy()
        completed = int(archive["completed_depths"])
        stages = cast("list[dict[str, object]]", json.loads(bytes(archive["stages_json"]).decode()))
        elapsed = float(archive["elapsed_seconds"])
    if completed != len(stages) or not np.all(np.isfinite(theta)):
        msg = "The retained pretraining state is inconsistent."
        raise CausalStudyError(msg)
    return theta, completed, stages, elapsed


def pretrain(
    root: Path,
    target: NDArray[np.complex128],
    config: StudyConfig,
) -> tuple[NDArray[np.float64], list[dict[str, object]], float]:
    """Create or resume the one shared layerwise noiseless checkpoint."""
    directory = root / PRETRAINING_DIRECTORY
    state_path = directory / PRETRAINING_STATE_NAME
    checkpoint_path = directory / PRETRAINED_CHECKPOINT_NAME
    theta: NDArray[np.float64]
    stages: list[dict[str, object]]
    if checkpoint_path.is_file():
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            theta = np.asarray(archive["theta"], dtype=np.float64).copy()
            stages = cast("list[dict[str, object]]", json.loads(bytes(archive["stages_json"]).decode()))
            elapsed = float(archive["elapsed_seconds"])
        return theta, stages, elapsed

    if state_path.is_file():
        theta, completed, stages, prior_elapsed = _load_pretraining_state(state_path)
    else:
        first_circuit = create_circuit(config.qubits, config.depths[0])
        theta = np.asarray(
            np.random.default_rng(config.initialization_seed).normal(
                0.0,
                config.initialization_scale,
                size=first_circuit.num_params,
            ),
            dtype=np.float64,
        )
        completed = 0
        stages = []
        prior_elapsed = 0.0
        _save_pretraining_state(state_path, theta, completed, stages, prior_elapsed)

    started = time.perf_counter()
    truncation = _truncation()
    for depth_index in range(completed, len(config.depths)):
        depth = config.depths[depth_index]
        circuit = create_circuit(config.qubits, depth)
        if theta.shape != (circuit.num_params,):
            if theta.size >= circuit.num_params:
                msg = "Layerwise circuit growth did not increase the parameter count."
                raise CausalStudyError(msg)
            expanded = np.full(circuit.num_params, config.growth_initialization, dtype=np.float64)
            expanded[: theta.size] = theta
            theta = expanded
        initial_fidelity = state_preparation_metrics(circuit, theta, target, truncation=truncation)[1]
        gradient_norms: list[float] = []
        for _ in range(config.pretraining_iterations_per_depth):
            signal, _loss, _fidelity = state_preparation_contribution(
                circuit,
                theta,
                target,
                MPS(config.qubits),
                truncation,
            )
            gradient_norms.append(float(np.linalg.norm(signal)))
            theta -= config.pretraining_learning_rate * signal
        final_fidelity = state_preparation_metrics(circuit, theta, target, truncation=truncation)[1]
        stages.append({
            "depth": depth,
            "parameter_count": circuit.num_params,
            "iterations": config.pretraining_iterations_per_depth,
            "initial_fidelity": initial_fidelity,
            "final_fidelity": final_fidelity,
            "final_signal_norm": gradient_norms[-1],
            "maximum_signal_norm": max(gradient_norms),
        })
        elapsed = prior_elapsed + time.perf_counter() - started
        _save_pretraining_state(state_path, theta, depth_index + 1, stages, elapsed)
        print(f"pretraining depth {depth}: fidelity={final_fidelity:.8f}", flush=True)

    elapsed = prior_elapsed + time.perf_counter() - started
    _atomic_npz(
        checkpoint_path,
        theta=theta,
        stages_json=_canonical_json(stages).encode(),
        elapsed_seconds=np.asarray(elapsed, dtype=np.float64),
    )
    return theta, stages, elapsed


def _ensemble_path(directory: Path, ensemble_index: int) -> Path:
    """Return the stable path for one fixed-map ensemble."""
    return directory / f"ensemble_{ensemble_index:04d}.json"


def _load_or_sample_ensemble(
    path: Path,
    *,
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    provider: StandardNoiseProvider | CalibratedLocalDepolarizingProvider,
    options: KrotovTJMOptions,
    truncation: KrotovTruncation,
    role: Literal["training_trajectory", "checkpoint_validation", "pilot_evaluation"],
    resolved_seed: int,
    stage_index: int,
    stage_id: str,
    configuration_checksum: str,
    circuit_checksum: str,
    ensemble_index: int,
    refresh_index: int,
    global_iteration_start: int,
) -> KrotovFixedMapEnsemble:
    """Load an immutable ensemble or sample and persist it exactly once."""
    if path.is_file():
        return KrotovFixedMapEnsemble.from_json(path.read_text(encoding="utf-8"))
    ensemble = sample_krotov_fixed_map_ensemble(
        circuit,
        theta,
        MPS(circuit.num_qubits),
        truncation,
        provider,
        options,
        role=role,
        resolved_seed=resolved_seed,
        stage_index=stage_index,
        stage_id=stage_id,
        stage_configuration_checksum=configuration_checksum,
        circuit_checksum=circuit_checksum,
        provider_checksum=provider.content_checksum,
        ensemble_index=ensemble_index,
        refresh_index=refresh_index,
        global_iteration_start=global_iteration_start,
    )
    _atomic_bytes(path, f"{ensemble.to_json()}\n".encode())
    return ensemble


def _validation_ensemble(
    replicate_root: Path,
    circuit: ParameterizedCircuit,
    shared_theta: NDArray[np.float64],
    config: StudyConfig,
    configuration_checksum: str,
) -> KrotovFixedMapEnsemble:
    """Return the replicate's fixed, training-disjoint selection ensemble."""
    return _load_or_sample_ensemble(
        replicate_root / "validation_ensemble.json",
        circuit=circuit,
        theta=shared_theta,
        provider=config.noise_provider(),
        options=_trajectory_options(config.validation_trajectories, cross=False),
        truncation=_truncation(),
        role="checkpoint_validation",
        resolved_seed=derive_study_seed(config.base_seed, "checkpoint_validation", int(replicate_root.name[-3:])),
        stage_index=1,
        stage_id="checkpoint_validation",
        configuration_checksum=configuration_checksum,
        circuit_checksum=_circuit_checksum(circuit),
        ensemble_index=0,
        refresh_index=0,
        global_iteration_start=0,
    )


def _validation_fidelity(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    config: StudyConfig,
    ensemble: KrotovFixedMapEnsemble,
) -> float:
    """Evaluate one checkpoint on the fixed validation ensemble."""
    return noisy_state_preparation_metrics(
        circuit,
        theta,
        target,
        None,
        _trajectory_options(config.validation_trajectories, cross=False),
        truncation=_truncation(),
        fixed_noise_maps=ensemble.replay_maps(),
        noise_provider=config.noise_provider(),
    )[1]


def _save_arm_state(
    path: Path,
    *,
    theta: NDArray[np.float64],
    selected_theta: NDArray[np.float64],
    completed_updates: int,
    selected_iteration: int,
    selected_validation_fidelity: float,
    trace: Sequence[Mapping[str, object]],
    elapsed_seconds: float,
) -> None:
    """Persist one validation-boundary fine-tuning state."""
    _atomic_npz(
        path,
        theta=theta,
        selected_theta=selected_theta,
        completed_updates=np.asarray(completed_updates, dtype=np.int64),
        selected_iteration=np.asarray(selected_iteration, dtype=np.int64),
        selected_validation_fidelity=np.asarray(selected_validation_fidelity, dtype=np.float64),
        trace_json=_canonical_json(list(trace)).encode(),
        elapsed_seconds=np.asarray(elapsed_seconds, dtype=np.float64),
    )


def _load_arm_state(
    path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, int, float, list[dict[str, object]], float]:
    """Load one validation-boundary fine-tuning state."""
    with np.load(path, allow_pickle=False) as archive:
        theta = np.asarray(archive["theta"], dtype=np.float64).copy()
        selected_theta = np.asarray(archive["selected_theta"], dtype=np.float64).copy()
        completed = int(archive["completed_updates"])
        selected_iteration = int(archive["selected_iteration"])
        selected_fidelity = float(archive["selected_validation_fidelity"])
        trace = cast("list[dict[str, object]]", json.loads(bytes(archive["trace_json"]).decode()))
        elapsed = float(archive["elapsed_seconds"])
    if len(theta) != len(selected_theta) or not np.all(np.isfinite(theta)) or not np.all(np.isfinite(selected_theta)):
        msg = f"The retained arm state {path} is inconsistent."
        raise CausalStudyError(msg)
    return theta, selected_theta, completed, selected_iteration, selected_fidelity, trace, elapsed


def training_schedule(arm: Arm, config: StudyConfig) -> KrotovMapSchedule:
    """Return the exact fixed, refreshed, or resampled schedule for a noisy arm."""
    policy: KrotovMapSamplingPolicy
    if arm == "crn_fixed":
        policy = "crn_fixed"
    elif arm == "crn_refresh":
        policy = "crn_refresh"
    elif arm == "resampled":
        policy = "resampled"
    else:
        msg = f"Arm {arm!r} does not use noisy trajectory ensembles."
        raise ValueError(msg)
    return KrotovMapSchedule(
        policy=policy,
        refresh_interval=config.refresh_interval if policy == "crn_refresh" else None,
    )


def _training_ensemble(
    arm_root: Path,
    arm: Arm,
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    config: StudyConfig,
    configuration_checksum: str,
    replicate_index: int,
    update_index: int,
) -> KrotovFixedMapEnsemble:
    """Return the policy-selected training ensemble for one noisy update."""
    schedule = training_schedule(arm, config)
    policy = schedule.policy
    point = schedule.point(update_index)
    return _load_or_sample_ensemble(
        _ensemble_path(arm_root / "training_ensembles", point.ensemble_index),
        circuit=circuit,
        theta=theta,
        provider=config.noise_provider(),
        options=_trajectory_options(config.training_trajectories, cross=True),
        truncation=_truncation(),
        role="training_trajectory",
        resolved_seed=derive_study_seed(config.base_seed, "training_trajectory", replicate_index),
        stage_index=0,
        stage_id="noisy_finetune",
        configuration_checksum=configuration_checksum,
        circuit_checksum=_circuit_checksum(circuit),
        ensemble_index=point.ensemble_index,
        refresh_index=point.refresh_index,
        global_iteration_start=(
            point.ensemble_index * config.refresh_interval if policy == "crn_refresh" else point.global_iteration
        ),
    )


def train_arm(
    replicate_root: Path,
    arm: Arm,
    shared_theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    config: StudyConfig,
    configuration_checksum: str,
    replicate_index: int,
    validation_ensemble: KrotovFixedMapEnsemble,
) -> dict[str, object]:
    """Train or resume one arm and return its selected checkpoint metadata."""
    arm_root = replicate_root / "arms" / arm
    state_path = arm_root / "state.npz"
    selected_path = arm_root / "selected_checkpoint.npz"
    circuit = create_circuit(config.qubits, config.depths[-1])
    truncation = _truncation()
    provider = config.noise_provider()
    initial_validation = _validation_fidelity(circuit, shared_theta, target, config, validation_ensemble)
    theta: NDArray[np.float64]
    selected_theta: NDArray[np.float64]
    trace: list[dict[str, object]]
    if state_path.is_file():
        theta, selected_theta, completed, selected_iteration, selected_fidelity, trace, prior_elapsed = _load_arm_state(
            state_path
        )
    else:
        theta = shared_theta.copy()
        selected_theta = shared_theta.copy()
        completed = 0
        selected_iteration = 0
        selected_fidelity = initial_validation
        prior_elapsed = 0.0
        trace = [
            {
                "iteration": 0,
                "learning_rate": 0.0,
                "training_fidelity": None,
                "signal_norm": 0.0,
                "update_norm": 0.0,
                "validation_fidelity": initial_validation,
            }
        ]
        _save_arm_state(
            state_path,
            theta=theta,
            selected_theta=selected_theta,
            completed_updates=completed,
            selected_iteration=selected_iteration,
            selected_validation_fidelity=selected_fidelity,
            trace=trace,
            elapsed_seconds=prior_elapsed,
        )

    target_updates = 0 if arm == "frozen" else config.finetune_updates
    started = time.perf_counter()
    while completed < target_updates:
        update_index = completed
        learning_rate = config.finetune_learning_rate * math.exp(-config.finetune_learning_rate_decay * update_index)
        theta_before = theta.copy()
        ensemble: KrotovFixedMapEnsemble | None = None
        if arm == "noiseless_continuation":
            signal, _training_loss, training_fidelity = state_preparation_contribution(
                circuit,
                theta,
                target,
                MPS(config.qubits),
                truncation,
            )
        else:
            ensemble = _training_ensemble(
                arm_root,
                arm,
                circuit,
                theta,
                config,
                configuration_checksum,
                replicate_index,
                update_index,
            )
            signal, _training_loss, training_fidelity, _trajectories = noisy_state_preparation_contribution(
                circuit,
                theta,
                target,
                None,
                _trajectory_options(config.training_trajectories, cross=True),
                MPS(config.qubits),
                truncation,
                fixed_noise_maps=ensemble.replay_maps(),
                noise_provider=provider,
            )
        theta -= learning_rate * signal
        completed += 1
        validation_fidelity: float | None = None
        if completed % config.validation_interval == 0 or completed == target_updates:
            validation_fidelity = _validation_fidelity(circuit, theta, target, config, validation_ensemble)
            if validation_fidelity > selected_fidelity:
                selected_theta = theta.copy()
                selected_iteration = completed
                selected_fidelity = validation_fidelity
        trace.append({
            "iteration": completed,
            "learning_rate": learning_rate,
            "training_fidelity": training_fidelity,
            "signal_norm": float(np.linalg.norm(signal)),
            "update_norm": float(np.linalg.norm(theta - theta_before)),
            "training_ensemble_index": None if ensemble is None else ensemble.ensemble_index,
            "training_ensemble_checksum": None if ensemble is None else ensemble.content_checksum,
            "validation_fidelity": validation_fidelity,
        })
        if validation_fidelity is not None:
            elapsed = prior_elapsed + time.perf_counter() - started
            _save_arm_state(
                state_path,
                theta=theta,
                selected_theta=selected_theta,
                completed_updates=completed,
                selected_iteration=selected_iteration,
                selected_validation_fidelity=selected_fidelity,
                trace=trace,
                elapsed_seconds=elapsed,
            )
            print(
                f"replicate {replicate_index} {arm} update {completed}: "
                f"validation={validation_fidelity:.8f} best={selected_fidelity:.8f}",
                flush=True,
            )

    elapsed = prior_elapsed + time.perf_counter() - started
    _atomic_npz(
        selected_path,
        theta=selected_theta,
        final_theta=theta,
        selected_iteration=np.asarray(selected_iteration, dtype=np.int64),
        selected_validation_fidelity=np.asarray(selected_fidelity, dtype=np.float64),
        elapsed_seconds=np.asarray(elapsed, dtype=np.float64),
    )
    return {
        "arm": arm,
        "completed_updates": target_updates,
        "initial_theta_checksum": _array_checksum(shared_theta),
        "selected_theta_checksum": _array_checksum(selected_theta),
        "selected_iteration": selected_iteration,
        "selected_validation_fidelity": selected_fidelity,
        "final_validation_fidelity": next(
            cast("float", row["validation_fidelity"])
            for row in reversed(trace)
            if row["validation_fidelity"] is not None
        ),
        "checkpoint": str(selected_path.relative_to(replicate_root)),
        "wall_time_seconds": elapsed,
        "trace": trace,
    }


def _selected_theta(replicate_root: Path, arm: str) -> NDArray[np.float64]:
    """Load one arm's validation-selected parameter vector."""
    path = replicate_root / "arms" / arm / "selected_checkpoint.npz"
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["theta"], dtype=np.float64).copy()


def _event_count(ensemble: KrotovFixedMapEnsemble) -> list[int]:
    """Count nonidentity gate events in each trajectory of an ensemble."""
    document = ensemble.to_dict()
    trajectories = cast("Sequence[Sequence[Mapping[str, object]]]", document["trajectory_maps"])
    return [sum(noise_map["is_identity"] is False for noise_map in trajectory) for trajectory in trajectories]


def _sample_summary(values: Sequence[float]) -> dict[str, object]:
    """Return a small-sample summary with a two-sided 95% Student-t interval."""
    count = len(values)
    if count == 0:
        msg = "Cannot summarize an empty sample."
        raise ValueError(msg)
    mean = fmean(values)
    sample_std = stdev(values) if count > 1 else 0.0
    standard_error = sample_std / math.sqrt(count)
    half_width = float(student_t.ppf(0.975, count - 1)) * standard_error if count > 1 else None
    return {
        "count": count,
        "mean": mean,
        "sample_standard_deviation": sample_std,
        "standard_error": standard_error,
        "confidence_interval_95_student_t": None if half_width is None else [mean - half_width, mean + half_width],
    }


def evaluate_replicate(
    replicate_root: Path,
    shared_theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    config: StudyConfig,
    configuration_checksum: str,
    replicate_index: int,
) -> dict[str, object]:
    """Evaluate all selected arms on common fresh, checkpointed map batches."""
    circuit = create_circuit(config.qubits, config.depths[-1])
    circuit_checksum = _circuit_checksum(circuit)
    truncation = _truncation()
    provider = config.noise_provider()
    selected = {arm: _selected_theta(replicate_root, arm) for arm in ARMS}
    rows: list[dict[str, object]] = []
    batch_count = math.ceil(config.test_trajectories / config.test_batch_size)
    evaluation_root = replicate_root / "evaluation"
    for batch_index in range(batch_count):
        result_path = evaluation_root / f"batch_{batch_index:04d}.json"
        if result_path.is_file():
            rows.extend(cast("list[dict[str, object]]", _load_json(result_path)["rows"]))
            continue
        start = batch_index * config.test_batch_size
        count = min(config.test_batch_size, config.test_trajectories - start)
        options = _trajectory_options(count, cross=False)
        ensemble = _load_or_sample_ensemble(
            evaluation_root / f"ensemble_{batch_index:04d}.json",
            circuit=circuit,
            theta=shared_theta,
            provider=provider,
            options=options,
            truncation=truncation,
            role="pilot_evaluation",
            resolved_seed=derive_study_seed(config.base_seed, "pilot_evaluation", replicate_index),
            stage_index=2,
            stage_id="paired_heldout_evaluation",
            configuration_checksum=configuration_checksum,
            circuit_checksum=circuit_checksum,
            ensemble_index=batch_index,
            refresh_index=batch_index,
            global_iteration_start=start,
        )
        maps = ensemble.replay_maps()
        values = {
            arm: noisy_state_preparation_metrics(
                circuit,
                theta,
                target,
                None,
                options,
                truncation=truncation,
                fixed_noise_maps=maps,
                noise_provider=provider,
            )[2]
            for arm, theta in selected.items()
        }
        event_counts = _event_count(ensemble)
        batch_rows: list[dict[str, object]] = []
        evaluation_seed = derive_study_seed(config.base_seed, "pilot_evaluation", replicate_index)
        for local_index in range(count):
            fidelities = {arm: values[arm][local_index] for arm in ARMS}
            batch_rows.append({
                "replicate": replicate_index,
                "trajectory_index": start + local_index,
                "trajectory_seed": derive_krotov_trajectory_seed(
                    role="pilot_evaluation",
                    resolved_seed=evaluation_seed,
                    stage_index=2,
                    ensemble_index=batch_index,
                    trajectory_index=local_index,
                    refresh_index=batch_index,
                ),
                "batch_index": batch_index,
                "event_count": event_counts[local_index],
                "fidelities": fidelities,
                "paired_differences_from_frozen": {
                    arm: fidelities[arm] - fidelities["frozen"] for arm in ARMS if arm != "frozen"
                },
            })
        batch_document = _seal({
            "configuration_checksum": configuration_checksum,
            "ensemble_checksum": ensemble.content_checksum,
            "rows": batch_rows,
        })
        _atomic_json(result_path, batch_document)
        rows.extend(batch_rows)
        print(f"replicate {replicate_index} evaluation batch {batch_index + 1}/{batch_count}", flush=True)

    rows.sort(key=lambda row: cast("int", row["trajectory_index"]))
    if len(rows) != config.test_trajectories:
        msg = f"Replicate {replicate_index} retained {len(rows)} evaluation rows, expected {config.test_trajectories}."
        raise CausalStudyError(msg)
    arm_values = {
        arm: [cast("float", cast("Mapping[str, object]", row["fidelities"])[arm]) for row in rows] for arm in ARMS
    }
    comparisons = {
        f"{left}_minus_{right}": [
            left_value - right_value
            for left_value, right_value in zip(arm_values[left], arm_values[right], strict=True)
        ]
        for left, right in (
            ("noiseless_continuation", "frozen"),
            ("crn_fixed", "frozen"),
            ("crn_refresh", "frozen"),
            ("resampled", "frozen"),
            ("crn_refresh", "crn_fixed"),
            ("crn_refresh", "resampled"),
            ("crn_refresh", "noiseless_continuation"),
        )
    }
    return {
        "replicate": replicate_index,
        "same_maps_used_for_all_arms": True,
        "trajectory_count": len(rows),
        "arm_summaries": {arm: _sample_summary(values) for arm, values in arm_values.items()},
        "paired_effect_summaries": {name: _sample_summary(values) for name, values in comparisons.items()},
        "noiseless_fidelities": {
            arm: state_preparation_metrics(circuit, theta, target, truncation=truncation)[1]
            for arm, theta in selected.items()
        },
        "rows": rows,
    }


def _ensemble_trajectory_seeds(path: Path) -> set[int]:
    """Derive every trajectory seed represented by one ensemble artifact."""
    ensemble = KrotovFixedMapEnsemble.from_json(path.read_text(encoding="utf-8"))
    return {
        derive_krotov_trajectory_seed(
            role=ensemble.role,
            resolved_seed=ensemble.resolved_seed,
            stage_index=ensemble.stage_index,
            ensemble_index=ensemble.ensemble_index,
            trajectory_index=trajectory_index,
            refresh_index=ensemble.refresh_index,
        )
        for trajectory_index in range(len(ensemble.replay_maps()))
    }


def verify_seed_separation(root: Path) -> dict[str, object]:
    """Verify that training, validation, and final-test trajectory streams are disjoint."""
    training_paths = sorted(root.glob("replicate_*/arms/*/training_ensembles/ensemble_*.json"))
    validation_paths = sorted(root.glob("replicate_*/validation_ensemble.json"))
    test_paths = sorted(root.glob("replicate_*/evaluation/ensemble_*.json"))
    training = set().union(*(_ensemble_trajectory_seeds(path) for path in training_paths)) if training_paths else set()
    validation = set().union(*(_ensemble_trajectory_seeds(path) for path in validation_paths))
    test = set().union(*(_ensemble_trajectory_seeds(path) for path in test_paths))
    pairwise_disjoint = not (training & validation or training & test or validation & test)
    if not pairwise_disjoint:
        msg = "Training, validation, and final-test trajectory seeds are not pairwise disjoint."
        raise CausalStudyError(msg)
    return {
        "pairwise_disjoint": True,
        "unique_training_seeds": len(training),
        "unique_validation_seeds": len(validation),
        "unique_test_seeds": len(test),
    }


def _aggregate(replicates: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate arm means and paired effects using replicate as the sampling unit."""
    arm_means = {
        arm: [
            cast(
                "float",
                cast("Mapping[str, object]", cast("Mapping[str, object]", item["arm_summaries"])[arm])["mean"],
            )
            for item in replicates
        ]
        for arm in ARMS
    }
    effect_names = tuple(cast("Mapping[str, object]", replicates[0]["paired_effect_summaries"]).keys())
    effect_means = {
        name: [
            cast(
                "float",
                cast(
                    "Mapping[str, object]",
                    cast("Mapping[str, object]", item["paired_effect_summaries"])[name],
                )["mean"],
            )
            for item in replicates
        ]
        for name in effect_names
    }
    return {
        "sampling_unit": "independent noisy-training replicate",
        "arm_summaries": {arm: _sample_summary(values) for arm, values in arm_means.items()},
        "paired_effect_summaries": {
            name: {**_sample_summary(values), "positive_replicates": sum(value > 0.0 for value in values)}
            for name, values in effect_means.items()
        },
        "replicate_arm_means": arm_means,
        "replicate_paired_effect_means": effect_means,
    }


def _run_replicate(
    replicate_index: int,
    root: Path,
    shared_theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    config: StudyConfig,
    configuration_checksum: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Execute one independent replicate in its own artifact directory."""
    replicate_root = root / f"replicate_{replicate_index:03d}"
    circuit = create_circuit(config.qubits, config.depths[-1])
    validation = _validation_ensemble(replicate_root, circuit, shared_theta, config, configuration_checksum)
    arms = [
        train_arm(
            replicate_root,
            cast("Arm", arm),
            shared_theta,
            target,
            config,
            configuration_checksum,
            replicate_index,
            validation,
        )
        for arm in ARMS
    ]
    initial_checksums = {cast("str", arm["initial_theta_checksum"]) for arm in arms}
    if initial_checksums != {_array_checksum(shared_theta)}:
        msg = f"Replicate {replicate_index} arms did not share the pretrained checkpoint."
        raise CausalStudyError(msg)
    training: dict[str, object] = {
        "replicate": replicate_index,
        "validation_ensemble_checksum": validation.content_checksum,
        "arms": arms,
    }
    evaluation = evaluate_replicate(
        replicate_root, shared_theta, target, config, configuration_checksum, replicate_index
    )
    _atomic_json(replicate_root / "replicate_result.json", _seal(evaluation))
    return training, evaluation


def run_study(
    output_root: Path,
    *,
    config: StudyConfig | None = None,
    resume: bool = False,
    workers: WorkerSetting = 1,
) -> dict[str, object]:
    """Run or exactly resume the canonical five-arm causal study."""
    config = StudyConfig() if config is None else config
    effective_workers = resolve_workers(workers, config.replicates)
    target_collection = load_target_collection()
    target_record = load_target(config.qubits, config.target_id)
    target = target_record.state_vector_copy()
    circuit = create_circuit(config.qubits, config.depths[-1])
    provider = config.noise_provider()
    config_document = _seal({
        **config.scientific_dict(),
        "target_fixture_checksum": target_collection.checksum,
        "target_vector_checksum": _array_checksum(target),
        "circuit_checksum": _circuit_checksum(circuit),
        "noise_provider_checksum": provider.content_checksum,
    })
    root = _prepare_root(output_root, config_document, resume=resume)
    result_path = root / RESULT_NAME
    if result_path.is_file():
        return _load_json(result_path)

    started = time.perf_counter()
    shared_theta, pretraining_stages, pretraining_seconds = pretrain(root, target, config)
    if shared_theta.shape != (circuit.num_params,):
        msg = f"Pretraining returned {shared_theta.shape}, expected ({circuit.num_params},)."
        raise CausalStudyError(msg)
    replicate_results: list[dict[str, object]] = []
    training_results: list[dict[str, object]] = []
    configuration_checksum = cast("str", config_document["content_checksum"])
    execute_replicate = partial(
        _run_replicate,
        root=root,
        shared_theta=shared_theta,
        target=target,
        config=config,
        configuration_checksum=configuration_checksum,
    )
    completed_replicates = [
        result
        for _index, result in sorted(completed_jobs(execute_replicate, list(range(config.replicates)), workers=workers))
    ]
    for training, evaluation in completed_replicates:
        training_results.append(training)
        replicate_results.append(evaluation)

    seed_audit = verify_seed_separation(root)
    result = _seal({
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "scope": "exploratory simulation",
        "configuration_checksum": configuration_checksum,
        "runtime": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "workers": effective_workers,
            "numerical_threads_per_worker": 1,
        },
        "design": {
            "arms": list(ARMS),
            "shared_pretrained_checkpoint": True,
            "validation_used_only_for_checkpoint_selection": True,
            "final_test_used_only_after_selection": True,
            "same_final_maps_used_for_all_arms_within_replicate": True,
            "paired_replay_is_unbiased_for_state_independent_pauli_jump_probabilities": True,
        },
        "pretraining": {
            "checkpoint": f"{PRETRAINING_DIRECTORY}/{PRETRAINED_CHECKPOINT_NAME}",
            "theta_checksum": _array_checksum(shared_theta),
            "stages": pretraining_stages,
            "wall_time_seconds": pretraining_seconds,
        },
        "training": training_results,
        "replicates": replicate_results,
        "aggregate": _aggregate(replicate_results),
        "seed_audit": seed_audit,
        "process_wall_time_seconds": time.perf_counter() - started,
    })
    _atomic_json(result_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    defaults = StudyConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--noise-calibration", choices=("legacy", "hardware_average_error"), default="legacy")
    parser.add_argument("--workers", type=parse_workers, default="auto")
    parser.add_argument(
        "--pretraining-iterations-per-depth",
        type=int,
        default=defaults.pretraining_iterations_per_depth,
    )
    parser.add_argument("--finetune-updates", type=int, default=defaults.finetune_updates)
    parser.add_argument("--training-trajectories", type=int, default=defaults.training_trajectories)
    parser.add_argument("--refresh-interval", type=int, default=defaults.refresh_interval)
    parser.add_argument("--validation-interval", type=int, default=defaults.validation_interval)
    parser.add_argument("--validation-trajectories", type=int, default=defaults.validation_trajectories)
    parser.add_argument("--test-trajectories", type=int, default=defaults.test_trajectories)
    parser.add_argument("--test-batch-size", type=int, default=defaults.test_batch_size)
    parser.add_argument("--replicates", type=int, default=defaults.replicates)
    parser.add_argument("--base-seed", type=int, default=defaults.base_seed)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the study and report its retained result path."""
    arguments = _parser().parse_args(argv)
    try:
        result = run_study(
            arguments.output_root,
            config=StudyConfig(
                noise_calibration=arguments.noise_calibration,
                pretraining_iterations_per_depth=arguments.pretraining_iterations_per_depth,
                finetune_updates=arguments.finetune_updates,
                training_trajectories=arguments.training_trajectories,
                refresh_interval=arguments.refresh_interval,
                validation_interval=arguments.validation_interval,
                validation_trajectories=arguments.validation_trajectories,
                test_trajectories=arguments.test_trajectories,
                test_batch_size=arguments.test_batch_size,
                replicates=arguments.replicates,
                base_seed=arguments.base_seed,
            ),
            resume=arguments.resume,
            workers=arguments.workers,
        )
    except Exception as error:
        print(f"Experiment failed: {error}", file=sys.stderr)
        return 2
    print(
        _canonical_json({
            "status": result["status"],
            "result": str(arguments.output_root / RESULT_NAME),
            "content_checksum": result["content_checksum"],
        })
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
