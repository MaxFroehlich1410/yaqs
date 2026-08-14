# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Minimal q8 noisy-Krotov experiment with three trajectories refreshed every five updates.

This runner intentionally contains only the scientific procedure and ordinary
reproducibility metadata. It does not depend on the Phase-II work-package,
preregistration, custody, source-locking, or benchmark-comparison machinery.
"""

# The command boundary reports failures directly and the compact experiment
# helpers have obvious return/error contracts.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from statistics import fmean, stdev
from typing import TYPE_CHECKING, cast

import numpy as np

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization import (
    GateNoiseContext,
    KrotovFixedMapEnsemble,
    KrotovTJMOptions,
    KrotovTruncation,
    TJMNoiseInstruction,
    derive_krotov_trajectory_seed,
    noisy_state_preparation_contribution,
    noisy_state_preparation_metrics,
    sample_krotov_fixed_map_ensemble,
    state_preparation_metrics,
)
from mqt.yaqs.optimization.parameterized_circuit import (
    ParameterizedCircuit,
    create_brickwall_matrix_product_disentangler_parameterized_circuit,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import ArrayLike, NDArray


SCHEMA_VERSION = "yaqs.noisy_krotov.minimal_refresh_study.v1"
CONFIG_NAME = "config.json"
RESULT_NAME = "refresh_study_result.json"
TRAINING_STATE_NAME = "training_state.npz"
TRAINING_TRACE_NAME = "training_trace.json"
FINAL_CHECKPOINT_NAME = "refresh5_checkpoint.npz"
INPUT_DIRECTORY = "inputs"
ENSEMBLE_DIRECTORY = "training_ensembles"
EVALUATION_DIRECTORY = "evaluation_batches"
PRETRAINED_CHECKPOINT_NAME = "pretrained_depth4_checkpoint.npz"
HISTORICAL_CHECKPOINT_NAME = "historical_fixed3_post_checkpoint.npz"
HISTORICAL_NOISE_ID = "ibm_inspired_pauli_legacy_v1"
HISTORICAL_NOISE_VERSION = "yaqs.state_preparation.noise.v1"
TARGET_VECTOR_CHECKSUM = "sha256:ff693beaa54cc47b10f7ec18148f2b90597cd6e8c9aa1b4eed71a2b64fb5965a"
DEFAULT_TRAINING_SEED = 17580007066253930890


class MinimalStudyError(RuntimeError):
    """Raised when the compact experiment cannot continue safely."""


def _canonical_json(value: object) -> str:
    """Serialize one JSON-native value deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _checksum(value: object) -> str:
    """Hash one JSON-native value."""
    return f"sha256:{hashlib.sha256(_canonical_json(value).encode()).hexdigest()}"


def _seal(value: Mapping[str, object]) -> dict[str, object]:
    """Return a detached mapping with a content checksum."""
    payload = dict(value)
    payload["content_checksum"] = _checksum(payload)
    return payload


def _load_json(path: Path, *, sealed: bool = True) -> dict[str, object]:
    """Load one JSON object and optionally verify its checksum."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        msg = f"{path} must contain a JSON object."
        raise TypeError(msg)
    if sealed:
        supplied = value.pop("content_checksum", None)
        computed = _checksum(value)
        if supplied != computed:
            msg = f"{path} has an invalid content checksum."
            raise ValueError(msg)
        value["content_checksum"] = supplied
    return cast("dict[str, object]", value)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Replace one file atomically."""
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
    """Write one canonical JSON document atomically."""
    _atomic_bytes(path, f"{_canonical_json(value)}\n".encode())


def _atomic_npz(path: Path, **arrays: ArrayLike) -> None:
    """Write one compressed NumPy archive atomically."""
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


def _file_checksum(path: Path) -> str:
    """Hash the exact bytes of one input or output file."""
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


@dataclass(frozen=True, slots=True)
class StudyConfig:
    """Complete scientific configuration for one refresh study."""

    updates: int = 200
    trajectories: int = 3
    refresh_interval: int = 5
    learning_rate: float = 0.2
    learning_rate_decay: float = 0.01
    training_seed: int = DEFAULT_TRAINING_SEED
    evaluation_trajectories: int = 2000
    evaluation_seed: int = 0
    evaluation_batch_size: int = 250
    qubits: int = 8
    depth: int = 4
    target_seed: int = 100

    def __post_init__(self) -> None:
        """Validate the narrow experiment configuration."""
        for name in ("updates", "trajectories", "refresh_interval", "evaluation_trajectories", "evaluation_batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                msg = f"{name} must be a positive integer."
                raise ValueError(msg)
        if self.refresh_interval > self.updates:
            msg = "refresh_interval cannot exceed updates."
            raise ValueError(msg)
        if self.evaluation_batch_size > self.evaluation_trajectories:
            msg = "evaluation_batch_size cannot exceed evaluation_trajectories."
            raise ValueError(msg)
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            msg = "learning_rate must be positive and finite."
            raise ValueError(msg)
        if not math.isfinite(self.learning_rate_decay) or self.learning_rate_decay < 0.0:
            msg = "learning_rate_decay must be nonnegative and finite."
            raise ValueError(msg)
        for name in ("training_seed", "evaluation_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**64:
                msg = f"{name} must be a uint64 integer."
                raise ValueError(msg)
        if (self.qubits, self.depth, self.target_seed) != (8, 4, 100):
            msg = "This minimal runner intentionally supports only the q8/depth4/target-seed100 experiment."
            raise ValueError(msg)

    @property
    def ensemble_count(self) -> int:
        """Number of training ensembles used by the refresh schedule."""
        return math.ceil(self.updates / self.refresh_interval)

    def scientific_dict(self) -> dict[str, object]:
        """Return the configuration without runtime or input paths."""
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


def disordered_tfim_ground_state(qubits: int = 8, seed: int = 100) -> NDArray[np.complex128]:
    """Reconstruct the historical disordered-TFIM ground state."""
    rng = np.random.RandomState(seed)
    couplings = rng.uniform(0.8, 1.2, size=qubits - 1)
    fields = rng.uniform(0.8, 1.2, size=qubits)
    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    hamiltonian = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)

    def kron_chain(operators: Sequence[NDArray[np.complex128]]) -> NDArray[np.complex128]:
        value = operators[0]
        for operator in operators[1:]:
            value = np.asarray(np.kron(value, operator), dtype=np.complex128)
        return np.asarray(value, dtype=np.complex128)

    for site, coupling in enumerate(couplings):
        operators = [identity] * qubits
        operators[site] = pauli_z
        operators[site + 1] = pauli_z
        hamiltonian -= coupling * kron_chain(operators)
    for site, field_value in enumerate(fields):
        operators = [identity] * qubits
        operators[site] = pauli_x
        hamiltonian -= field_value * kron_chain(operators)
    _eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    vector = np.asarray(eigenvectors[:, 0], dtype=np.complex128)
    if (qubits, seed) == (8, 100):
        checksum = f"sha256:{hashlib.sha256(np.ascontiguousarray(vector, dtype='<c16').tobytes()).hexdigest()}"
        if checksum != TARGET_VECTOR_CHECKSUM:
            msg = "The reconstructed target differs from the retained seed-100 reference."
            raise MinimalStudyError(msg)
    return vector


@dataclass(frozen=True, slots=True)
class HistoricalNoiseProvider:
    """The historical logical TJM noise profile used by the experiment."""

    noise_id: str = field(default=HISTORICAL_NOISE_ID, init=False)
    one_qubit_strength: float = field(default=3.0e-4 / 3.0, init=False)
    two_qubit_strength: float = field(default=1.5e-3, init=False)

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete fixed profile."""
        return _checksum({
            "noise_id": self.noise_id,
            "noise_definition_version": HISTORICAL_NOISE_VERSION,
            "one_qubit_strength": self.one_qubit_strength,
            "two_qubit_strength": self.two_qubit_strength,
        })

    def __call__(self, context: GateNoiseContext, rng: np.random.Generator, /) -> TJMNoiseInstruction:
        """Construct a fresh gate-local noise instruction."""
        del rng
        processes: list[dict[str, object]] = []
        for site in context.sites:
            processes.extend(
                {"name": f"pauli_{label}", "sites": [site], "strength": self.one_qubit_strength}
                for label in ("x", "y", "z")
            )
        if context.arity == 2 and abs(context.sites[0] - context.sites[1]) == 1:
            processes.extend(
                {
                    "name": f"crosstalk_{label}",
                    "sites": list(context.sites),
                    "strength": self.two_qubit_strength,
                }
                for label in ("xx", "zz")
            )
        model = NoiseModel(processes)
        for process in model.processes:
            if "matrix" in process:
                process["matrix"] = np.array(process["matrix"], dtype=np.complex128, copy=True)
            if "factors" in process:
                process["factors"] = tuple(
                    np.array(factor, dtype=np.complex128, copy=True)
                    for factor in cast("Sequence[object]", process["factors"])
                )
        return TJMNoiseInstruction(model, channel_id=self.noise_id)


def create_circuit() -> ParameterizedCircuit:
    """Create the exact q8 depth-four logical circuit."""
    return create_brickwall_matrix_product_disentangler_parameterized_circuit(
        8,
        4,
        initial_single_qubit_layer=True,
    )


def _circuit_checksum(circuit: ParameterizedCircuit) -> str:
    """Hash the logical gate sequence used by this experiment."""
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


def _load_theta(path: Path) -> NDArray[np.float64]:
    """Load a parameter vector from a minimal or legacy checkpoint."""
    with np.load(path, allow_pickle=False) as archive:
        key = "theta" if "theta" in archive.files else "selected_theta"
        theta = np.asarray(archive[key], dtype=np.float64).copy()
    if theta.shape != (276,) or not np.all(np.isfinite(theta)):
        msg = f"{path} does not contain a finite 276-parameter circuit."
        raise ValueError(msg)
    return theta


def _save_checkpoint(path: Path, theta: NDArray[np.float64], metadata: Mapping[str, object]) -> None:
    """Save one compact parameter checkpoint."""
    _atomic_npz(path, theta=np.asarray(theta, dtype=np.float64), metadata_json=_canonical_json(dict(metadata)).encode())


def _prepare_root(output_root: Path, bundle_root: Path | None, *, resume: bool) -> tuple[Path, dict[str, object]]:
    """Create or verify the simple self-contained output directory."""
    root = output_root.resolve()
    if resume:
        if not root.is_dir():
            msg = "--resume requires an existing output directory."
            raise ValueError(msg)
        return root, _load_json(root / CONFIG_NAME)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        msg = "A new run requires a missing or empty output directory."
        raise ValueError(msg)
    if bundle_root is None:
        msg = "--bundle-root is required for a new run."
        raise ValueError(msg)
    source = bundle_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    inputs = root / INPUT_DIRECTORY
    inputs.mkdir()
    for name in (PRETRAINED_CHECKPOINT_NAME, HISTORICAL_CHECKPOINT_NAME):
        shutil.copyfile(source / name, inputs / name)
    return root, {}


def _training_options(circuit: ParameterizedCircuit, config: StudyConfig) -> KrotovTJMOptions:
    """Create the exact three-trajectory cross-update options."""
    noisy_indices = tuple(index for index, gate in enumerate(circuit.gates) if gate.is_trainable and gate.noise_enabled)
    return KrotovTJMOptions(
        num_trajectories=config.trajectories,
        random_seed=config.training_seed,
        dt=1.0,
        apply_noise_to="all",
        noisy_gate_indices=noisy_indices,
        trajectory_update="cross",
        differentiate_jump_normalization=False,
        use_crn=False,
    )


def _load_training_state(path: Path) -> tuple[NDArray[np.float64], int, list[dict[str, object]], float]:
    """Load one atomic refresh-boundary checkpoint."""
    with np.load(path, allow_pickle=False) as archive:
        theta = np.asarray(archive["theta"], dtype=np.float64).copy()
        completed = int(archive["completed_updates"])
        trace = cast("list[dict[str, object]]", json.loads(bytes(archive["trace_json"]).decode()))
        elapsed = float(archive["elapsed_seconds"])
    if len(trace) != completed + 1:
        msg = "Training checkpoint trace length does not match completed updates."
        raise MinimalStudyError(msg)
    return theta, completed, trace, elapsed


def _save_training_state(
    path: Path,
    theta: NDArray[np.float64],
    completed: int,
    trace: Sequence[Mapping[str, object]],
    elapsed: float,
) -> None:
    """Persist one atomic refresh-boundary checkpoint."""
    _atomic_npz(
        path,
        theta=np.asarray(theta, dtype=np.float64),
        completed_updates=np.asarray(completed, dtype=np.int64),
        trace_json=_canonical_json(list(trace)).encode(),
        elapsed_seconds=np.asarray(elapsed, dtype=np.float64),
    )


def _ensemble(
    root: Path,
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    provider: HistoricalNoiseProvider,
    options: KrotovTJMOptions,
    truncation: KrotovTruncation,
    config: StudyConfig,
    config_checksum: str,
    circuit_checksum: str,
    ensemble_index: int,
) -> KrotovFixedMapEnsemble:
    """Load or sample one refresh-window ensemble."""
    path = root / ENSEMBLE_DIRECTORY / f"ensemble_{ensemble_index:03d}.json"
    if path.is_file():
        return KrotovFixedMapEnsemble.from_json(path.read_text(encoding="utf-8"))
    ensemble = sample_krotov_fixed_map_ensemble(
        circuit,
        theta,
        MPS(config.qubits),
        truncation,
        provider,
        options,
        role="training_trajectory",
        resolved_seed=config.training_seed,
        stage_index=0,
        stage_id="refresh5_finetune",
        stage_configuration_checksum=config_checksum,
        circuit_checksum=circuit_checksum,
        provider_checksum=provider.content_checksum,
        ensemble_index=ensemble_index,
        refresh_index=ensemble_index,
        global_iteration_start=ensemble_index * config.refresh_interval,
    )
    _atomic_bytes(path, f"{ensemble.to_json()}\n".encode())
    return ensemble


def train(
    root: Path,
    pretrained: NDArray[np.float64],
    target: NDArray[np.complex128],
    config: StudyConfig,
    config_checksum: str,
) -> tuple[NDArray[np.float64], list[dict[str, object]], float]:
    """Train directly, checkpointing after every refresh window."""
    state_path = root / TRAINING_STATE_NAME
    circuit = create_circuit()
    provider = HistoricalNoiseProvider()
    truncation = KrotovTruncation(max_bond_dim=None, svd_threshold=0.0, trunc_mode="discarded_weight", min_bond_dim=1)
    options = _training_options(circuit, config)
    circuit_checksum = _circuit_checksum(circuit)
    theta: NDArray[np.float64]
    trace: list[dict[str, object]]
    if state_path.is_file():
        theta, completed, trace, prior_elapsed = _load_training_state(state_path)
    else:
        theta = pretrained.copy()
        completed = 0
        prior_elapsed = 0.0
        initial_ensemble = _ensemble(
            root,
            circuit,
            theta,
            provider,
            options,
            truncation,
            config,
            config_checksum,
            circuit_checksum,
            0,
        )
        initial_loss, initial_fidelity, _ = noisy_state_preparation_metrics(
            circuit,
            theta,
            target,
            None,
            options,
            initial_state=MPS(config.qubits),
            truncation=truncation,
            fixed_noise_maps=initial_ensemble.replay_maps(),
            noise_provider=provider,
        )
        initial_trace_row: dict[str, object] = {
            "iteration": 0,
            "ensemble_index": 0,
            "ensemble_checksum": initial_ensemble.content_checksum,
            "ensemble_nonidentity_events": initial_ensemble.nonidentity_event_count,
            "learning_rate": 0.0,
            "loss": initial_loss,
            "fidelity": initial_fidelity,
            "update_norm": 0.0,
            "signal_norm": 0.0,
        }
        trace = [initial_trace_row]
        _save_training_state(state_path, theta, completed, trace, prior_elapsed)

    started = time.perf_counter()
    while completed < config.updates:
        ensemble_index = completed // config.refresh_interval
        ensemble = _ensemble(
            root,
            circuit,
            theta,
            provider,
            options,
            truncation,
            config,
            config_checksum,
            circuit_checksum,
            ensemble_index,
        )
        window_end = min((ensemble_index + 1) * config.refresh_interval, config.updates)
        maps = ensemble.replay_maps()
        for update_index in range(completed, window_end):
            learning_rate = config.learning_rate * math.exp(-config.learning_rate_decay * update_index)
            theta_before = theta.copy()
            signal, _pre_loss, _pre_fidelity, _ = noisy_state_preparation_contribution(
                circuit,
                theta,
                target,
                None,
                options,
                MPS(config.qubits),
                truncation,
                fixed_noise_maps=maps,
                noise_provider=provider,
            )
            theta -= learning_rate * signal
            loss, fidelity, _ = noisy_state_preparation_metrics(
                circuit,
                theta,
                target,
                None,
                options,
                initial_state=MPS(config.qubits),
                truncation=truncation,
                fixed_noise_maps=maps,
                noise_provider=provider,
            )
            trace.append({
                "iteration": update_index + 1,
                "ensemble_index": ensemble_index,
                "ensemble_checksum": ensemble.content_checksum,
                "ensemble_nonidentity_events": ensemble.nonidentity_event_count,
                "learning_rate": learning_rate,
                "loss": loss,
                "fidelity": fidelity,
                "update_norm": float(np.linalg.norm(theta - theta_before)),
                "signal_norm": float(np.linalg.norm(signal)),
            })
        completed = window_end
        elapsed = prior_elapsed + time.perf_counter() - started
        _save_training_state(state_path, theta, completed, trace, elapsed)
    elapsed = prior_elapsed + time.perf_counter() - started
    _atomic_json(root / TRAINING_TRACE_NAME, trace)
    _save_checkpoint(
        root / FINAL_CHECKPOINT_NAME,
        theta,
        {"completed_updates": completed, "configuration_checksum": config_checksum},
    )
    return theta, trace, elapsed


def _event_details(trajectory_maps: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Extract compact nonidentity-event metadata."""
    return [
        {
            "gate_index": gate_index,
            "channel_id": noise_map["channel_id"],
            "jump_process_index": noise_map["jump_process_index"],
            "outcome_labels": noise_map["outcome_labels"],
        }
        for gate_index, noise_map in enumerate(trajectory_maps)
        if noise_map["is_identity"] is False
    ]


def _summary(values: Sequence[float]) -> dict[str, float]:
    """Summarize one finite sample."""
    sample_std = stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": fmean(values),
        "sample_standard_deviation": sample_std,
        "standard_error": sample_std / math.sqrt(len(values)),
    }


def _group_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize paired circuit fidelities."""
    pretrained = [cast("float", row["pretrained"]) for row in rows]
    historical = [cast("float", row["historical_fixed3"]) for row in rows]
    refreshed = [cast("float", row["refresh5"]) for row in rows]
    differences = [new - old for new, old in zip(refreshed, historical, strict=True)]
    return {
        "count": len(rows),
        "pretrained": _summary(pretrained),
        "historical_fixed3": _summary(historical),
        "refresh5": _summary(refreshed),
        "refresh5_minus_historical_fixed3": _summary(differences),
        "refresh5_improvement_fraction": sum(value > 0.0 for value in differences) / len(rows),
        "historical_fixed3_fidelity_below_0p01_count": sum(value < 0.01 for value in historical),
        "refresh5_fidelity_below_0p01_count": sum(value < 0.01 for value in refreshed),
    }


def _training_seeds(config: StudyConfig) -> set[int]:
    """Derive every training trajectory seed."""
    return {
        derive_krotov_trajectory_seed(
            role="training_trajectory",
            resolved_seed=config.training_seed,
            stage_index=0,
            ensemble_index=ensemble_index,
            trajectory_index=trajectory_index,
            refresh_index=ensemble_index,
        )
        for ensemble_index in range(config.ensemble_count)
        for trajectory_index in range(config.trajectories)
    }


def evaluate(
    root: Path,
    pretrained: NDArray[np.float64],
    historical: NDArray[np.float64],
    refreshed: NDArray[np.float64],
    target: NDArray[np.complex128],
    config: StudyConfig,
    config_checksum: str,
) -> tuple[dict[str, object], float]:
    """Evaluate all three circuits on common fresh maps with batch checkpoints."""
    started = time.perf_counter()
    circuit = create_circuit()
    provider = HistoricalNoiseProvider()
    circuit_checksum = _circuit_checksum(circuit)
    truncation = KrotovTruncation(max_bond_dim=None, svd_threshold=0.0, trunc_mode="discarded_weight", min_bond_dim=1)
    noisy_indices = tuple(index for index, gate in enumerate(circuit.gates) if gate.is_trainable and gate.noise_enabled)
    rows: list[dict[str, object]] = []
    batches: list[dict[str, object]] = []
    evaluation_seeds: set[int] = set()
    batch_count = math.ceil(config.evaluation_trajectories / config.evaluation_batch_size)
    for batch_index in range(batch_count):
        batch_path = root / EVALUATION_DIRECTORY / f"batch_{batch_index:03d}.json"
        if batch_path.is_file():
            batch = _load_json(batch_path)
            batch_rows = cast("list[dict[str, object]]", batch["rows"])
            rows.extend(batch_rows)
            batches.append(cast("dict[str, object]", batch["summary"]))
            evaluation_seeds.update(cast("int", row["trajectory_seed"]) for row in batch_rows)
            continue
        start = batch_index * config.evaluation_batch_size
        count = min(config.evaluation_batch_size, config.evaluation_trajectories - start)
        options = KrotovTJMOptions(
            num_trajectories=count,
            random_seed=config.evaluation_seed,
            dt=1.0,
            apply_noise_to="all",
            noisy_gate_indices=noisy_indices,
            trajectory_update="independent",
            differentiate_jump_normalization=False,
            use_crn=False,
        )
        ensemble = sample_krotov_fixed_map_ensemble(
            circuit,
            pretrained,
            None,
            truncation,
            provider,
            options,
            role="pilot_evaluation",
            resolved_seed=config.evaluation_seed,
            stage_index=1,
            stage_id="refresh5_heldout_evaluation",
            stage_configuration_checksum=config_checksum,
            circuit_checksum=circuit_checksum,
            provider_checksum=provider.content_checksum,
            ensemble_index=batch_index,
            refresh_index=batch_index,
            global_iteration_start=start,
        )
        maps = ensemble.replay_maps()
        pretrained_values = noisy_state_preparation_metrics(
            circuit,
            pretrained,
            target,
            None,
            options,
            truncation=truncation,
            fixed_noise_maps=maps,
            noise_provider=provider,
        )[2]
        historical_values = noisy_state_preparation_metrics(
            circuit,
            historical,
            target,
            None,
            options,
            truncation=truncation,
            fixed_noise_maps=maps,
            noise_provider=provider,
        )[2]
        refreshed_values = noisy_state_preparation_metrics(
            circuit,
            refreshed,
            target,
            None,
            options,
            truncation=truncation,
            fixed_noise_maps=maps,
            noise_provider=provider,
        )[2]
        trajectory_maps = cast(
            "Sequence[Sequence[Mapping[str, object]]]",
            ensemble.to_dict()["trajectory_maps"],
        )
        batch_rows: list[dict[str, object]] = []
        for local_index, (pre_value, old_value, new_value, trajectory) in enumerate(
            zip(pretrained_values, historical_values, refreshed_values, trajectory_maps, strict=True)
        ):
            seed = derive_krotov_trajectory_seed(
                role="pilot_evaluation",
                resolved_seed=config.evaluation_seed,
                stage_index=1,
                ensemble_index=batch_index,
                trajectory_index=local_index,
                refresh_index=batch_index,
            )
            events = _event_details(trajectory)
            batch_rows.append({
                "trajectory_index": start + local_index,
                "trajectory_seed": seed,
                "batch_index": batch_index,
                "event_count": len(events),
                "events": events,
                "pretrained": pre_value,
                "historical_fixed3": old_value,
                "refresh5": new_value,
                "refresh5_minus_historical_fixed3": new_value - old_value,
            })
        summary: dict[str, object] = {
            "batch_index": batch_index,
            "trajectory_start": start,
            "trajectory_count": count,
            "ensemble_checksum": ensemble.content_checksum,
            "nonidentity_event_count": ensemble.nonidentity_event_count,
        }
        batch_document = _seal({"configuration_checksum": config_checksum, "summary": summary, "rows": batch_rows})
        _atomic_json(batch_path, batch_document)
        rows.extend(batch_rows)
        batches.append(summary)
        evaluation_seeds.update(cast("int", row["trajectory_seed"]) for row in batch_rows)

    rows.sort(key=lambda row: cast("int", row["trajectory_index"]))
    training_seeds = _training_seeds(config)
    if training_seeds.intersection(evaluation_seeds):
        msg = "Training and evaluation trajectory seeds overlap."
        raise MinimalStudyError(msg)
    zero_event = [row for row in rows if row["event_count"] == 0]
    event_affected = [row for row in rows if cast("int", row["event_count"]) > 0]
    result: dict[str, object] = {
        "same_maps_used_for_all_circuits": True,
        "training_and_evaluation_seeds_disjoint": True,
        "batches": batches,
        "noiseless_fidelities": {
            "pretrained": state_preparation_metrics(circuit, pretrained, target, truncation=truncation)[1],
            "historical_fixed3": state_preparation_metrics(circuit, historical, target, truncation=truncation)[1],
            "refresh5": state_preparation_metrics(circuit, refreshed, target, truncation=truncation)[1],
        },
        "groups": {
            "all": _group_summary(rows),
            "zero_event": _group_summary(zero_event) if zero_event else None,
            "event_affected": _group_summary(event_affected) if event_affected else None,
        },
        "rows": rows,
    }
    return result, time.perf_counter() - started


def run_study(
    output_root: Path,
    *,
    bundle_root: Path | None,
    config: StudyConfig | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Run or resume the compact experiment."""
    config = StudyConfig() if config is None else config
    root, retained_config = _prepare_root(output_root, bundle_root, resume=resume)
    inputs = root / INPUT_DIRECTORY
    pretrained_path = inputs / PRETRAINED_CHECKPOINT_NAME
    historical_path = inputs / HISTORICAL_CHECKPOINT_NAME
    config_payload = {
        **config.scientific_dict(),
        "pretrained_checkpoint_checksum": _file_checksum(pretrained_path),
        "historical_checkpoint_checksum": _file_checksum(historical_path),
    }
    config_document = _seal(config_payload)
    if resume:
        if retained_config != config_document:
            msg = "The active configuration or input checkpoints differ from the retained run."
            raise ValueError(msg)
    else:
        _atomic_json(root / CONFIG_NAME, config_document)
    result_path = root / RESULT_NAME
    if result_path.is_file():
        return _load_json(result_path)

    pretrained = _load_theta(pretrained_path)
    historical = _load_theta(historical_path)
    target = disordered_tfim_ground_state(config.qubits, config.target_seed)
    started = time.perf_counter()
    refreshed, trace, training_seconds = train(
        root,
        pretrained,
        target,
        config,
        cast("str", config_document["content_checksum"]),
    )
    evaluation, evaluation_seconds = evaluate(
        root,
        pretrained,
        historical,
        refreshed,
        target,
        config,
        cast("str", config_document["content_checksum"]),
    )
    ensemble_files = sorted((root / ENSEMBLE_DIRECTORY).glob("ensemble_*.json"))
    ensembles = [_load_json(path, sealed=False) for path in ensemble_files]
    result = _seal({
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "scope": "exploratory simulation",
        "configuration_checksum": config_document["content_checksum"],
        "runtime": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        },
        "training": {
            "completed_updates": config.updates,
            "ensemble_count": len(ensembles),
            "unique_trajectory_count": len(ensembles) * config.trajectories,
            "nonidentity_event_count": sum(cast("int", ensemble["nonidentity_event_count"]) for ensemble in ensembles),
            "initial_fidelity": trace[0]["fidelity"],
            "final_training_fidelity": trace[-1]["fidelity"],
            "checkpoint": FINAL_CHECKPOINT_NAME,
            "checkpoint_checksum": _file_checksum(root / FINAL_CHECKPOINT_NAME),
            "wall_time_seconds": training_seconds,
        },
        "evaluation": evaluation,
        "evaluation_wall_time_seconds": evaluation_seconds,
        "process_wall_time_seconds": time.perf_counter() - started,
    })
    _atomic_json(result_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    """Build the small command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--refresh-interval", type=int, default=5)
    parser.add_argument("--training-seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--evaluation-trajectories", type=int, default=2000)
    parser.add_argument("--evaluation-seed", type=int, default=0)
    parser.add_argument("--evaluation-batch-size", type=int, default=250)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run or resume the minimal experiment."""
    arguments = _parser().parse_args(argv)
    try:
        result = run_study(
            arguments.output_root,
            bundle_root=arguments.bundle_root,
            config=StudyConfig(
                updates=arguments.updates,
                trajectories=arguments.trajectories,
                refresh_interval=arguments.refresh_interval,
                training_seed=arguments.training_seed,
                evaluation_trajectories=arguments.evaluation_trajectories,
                evaluation_seed=arguments.evaluation_seed,
                evaluation_batch_size=arguments.evaluation_batch_size,
            ),
            resume=arguments.resume,
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
