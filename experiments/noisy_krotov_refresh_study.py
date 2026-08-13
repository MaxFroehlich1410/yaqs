# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run an exploratory q8 Krotov study with three trajectories refreshed every five updates.

The module has two explicit commands. ``export-pretrained`` extracts the retained
depth-four noiseless checkpoint from a completed single-target calibration into
a portable, checksum-bound bundle. ``run`` imports that bundle as a verified
external checkpoint, executes only the noisy fine-tuning stage, and compares the
refreshed result with the retained historical fixed-three result on one common
fresh held-out ensemble.

Every produced artifact is exploratory and not confirmatory evidence.
"""

# The CLI reports terminal failures, Git is invoked only with fixed read-only
# arguments, and direct execution needs the repository root on sys.path.
# ruff: file-ignore[blind-except, docstring-missing-returns, docstring-missing-exception, module-import-not-at-top-of-file, subprocess-without-shell-equals-true, start-process-with-partial-path, print]

from __future__ import annotations

import argparse
import contextlib
import hashlib
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from statistics import fmean, stdev
from typing import TYPE_CHECKING, cast

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from benchmarks.state_preparation.noise import create_historical_fixed_rate_noise_provider
from benchmarks.state_preparation.phase2.artifact_codecs import StageParameterCheckpoint
from benchmarks.state_preparation.phase2.artifacts import Phase2ArtifactStore
from benchmarks.state_preparation.phase2.canonical import (
    canonical_checksum,
    canonical_json,
    load_canonical_json_object,
    seal_mapping,
    thaw_json,
)
from benchmarks.state_preparation.phase2.execution import Phase2PipelineExecutor, PipelineExecutionFailure
from benchmarks.state_preparation.phase2.layerwise_bmpd import create_bmpd_circuit_binding
from benchmarks.state_preparation.phase2.legacy_targets import load_legacy_target_collection
from benchmarks.state_preparation.phase2.noisy_krotov import (
    NoisyKrotovStageExecution,
    NoisyKrotovStageFailure,
    execute_fixed_rate_krotov_stage,
)
from benchmarks.state_preparation.phase2.pipeline import (
    ExternalCheckpointRef,
    TrainingPipelineConfig,
    TrainingPipelineResult,
    TrainingPipelineTemplate,
    TrainingStageResult,
    TrainingStageTemplate,
)
from benchmarks.state_preparation.phase2.resumability import capture_resumability_fingerprint
from benchmarks.state_preparation.reporting import atomic_write_bytes
from mqt.yaqs.optimization import (
    KrotovTJMOptions,
    KrotovTruncation,
    derive_krotov_trajectory_seed,
    noisy_state_preparation_metrics,
    sample_krotov_fixed_map_ensemble,
    state_preparation_metrics,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    import numpy as np
    from numpy.typing import NDArray

    from benchmarks.state_preparation.phase2.legacy_targets import LegacyMaterializedTarget
    from benchmarks.state_preparation.phase2.pipeline import TrainingStageConfig
    from benchmarks.state_preparation.phase2.resumability import ResumabilityFingerprint


PRETRAINED_BUNDLE_SCHEMA_VERSION = "yaqs.noisy_krotov.pretrained_q8_bundle.v1"
REFRESH_STUDY_SCHEMA_VERSION = "yaqs.noisy_krotov.refresh5_study.v1"
PRETRAINED_MANIFEST_NAME = "pretrained_circuit.json"
PRETRAINED_CHECKPOINT_NAME = "pretrained_depth4_checkpoint.npz"
HISTORICAL_POST_CHECKPOINT_NAME = "historical_fixed3_post_checkpoint.npz"
STUDY_CONFIG_NAME = "refresh_study_config.json"
STUDY_RESULT_NAME = "refresh_study_result.json"
STUDY_NOTICE_NAME = "EXPLORATORY_NOT_CONFIRMATORY.txt"
STUDY_STORE_NAME = "exploratory_refresh5_phase2_store"
STUDY_INPUT_DIRECTORY = "inputs"
EVIDENCE_CLASSIFICATION = "exploratory"
EVIDENCE_WARNING = "not confirmatory evidence"
DEFAULT_FINETUNE_UPDATES = 200
DEFAULT_TRAINING_TRAJECTORIES = 3
DEFAULT_REFRESH_INTERVAL = 5
DEFAULT_EVALUATION_TRAJECTORIES = 2000
DEFAULT_EVALUATION_SEED = 0
DEFAULT_EVALUATION_BATCH_SIZE = 250
_DEPENDENCIES = ("mqt.yaqs", "numpy", "scipy")
_LOCKFILES = (Path("pyproject.toml"), Path("uv.lock"))
_SEALED_INPUTS = (
    Path("benchmarks/state_preparation/phase2/data/initial_preregistration_v1.json"),
    Path("benchmarks/state_preparation/phase2/data/legacy_evidence_audit_v1.json"),
    Path("benchmarks/state_preparation/phase2/data/legacy_tfim_targets_v1.json"),
)


class RefreshStudyExecutionError(RuntimeError):
    """Raised when the exploratory refresh study cannot complete safely."""


@dataclass(frozen=True, slots=True)
class RefreshStudyConfig:
    """Scientific configuration of one fine-tuning-only refresh study."""

    finetune_updates: int = DEFAULT_FINETUNE_UPDATES
    training_trajectories: int = DEFAULT_TRAINING_TRAJECTORIES
    refresh_interval: int = DEFAULT_REFRESH_INTERVAL
    training_repetition: int = 0
    evaluation_trajectories: int = DEFAULT_EVALUATION_TRAJECTORIES
    evaluation_seed: int = DEFAULT_EVALUATION_SEED
    evaluation_batch_size: int = DEFAULT_EVALUATION_BATCH_SIZE

    def __post_init__(self) -> None:
        """Validate the deliberately narrow exploratory treatment."""
        for name in (
            "finetune_updates",
            "training_trajectories",
            "refresh_interval",
            "evaluation_trajectories",
            "evaluation_batch_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                msg = f"{name} must be a positive integer."
                raise ValueError(msg)
        if self.refresh_interval > self.finetune_updates:
            msg = "refresh_interval cannot exceed finetune_updates."
            raise ValueError(msg)
        if type(self.training_repetition) is not int or self.training_repetition < 0:
            msg = "training_repetition must be a nonnegative integer."
            raise ValueError(msg)
        if type(self.evaluation_seed) is not int or self.evaluation_seed < 0:
            msg = "evaluation_seed must be a nonnegative integer."
            raise ValueError(msg)
        if self.evaluation_batch_size > self.evaluation_trajectories:
            msg = "evaluation_batch_size cannot exceed evaluation_trajectories."
            raise ValueError(msg)

    @property
    def expected_training_ensemble_count(self) -> int:
        """Exact number of periodically refreshed ensembles."""
        return math.ceil(self.finetune_updates / self.refresh_interval)

    @property
    def expected_unique_training_trajectories(self) -> int:
        """Number of distinct trajectory coordinates seen during training."""
        return self.expected_training_ensemble_count * self.training_trajectories

    def to_dict(self, *, pretrained_bundle_checksum: str) -> dict[str, object]:
        """Return a checksum-sealed configuration document."""
        return seal_mapping({
            "schema_version": REFRESH_STUDY_SCHEMA_VERSION,
            "evidence_classification": EVIDENCE_CLASSIFICATION,
            "evidence_warning": EVIDENCE_WARNING,
            "method_id": "layerwise_bmpd_cross_crn_refresh_q8_exploratory",
            "method_variant": "q8_historical_noise_periodic_full_refresh",
            "pretrained_bundle_checksum": pretrained_bundle_checksum,
            "finetune_updates": self.finetune_updates,
            "training_trajectories_per_update": self.training_trajectories,
            "training_sampling_policy": "crn_refresh",
            "crn_refresh_interval": self.refresh_interval,
            "training_repetition": self.training_repetition,
            "expected_training_ensemble_count": self.expected_training_ensemble_count,
            "expected_unique_training_trajectories": self.expected_unique_training_trajectories,
            "evaluation_trajectories": self.evaluation_trajectories,
            "evaluation_seed": self.evaluation_seed,
            "evaluation_batch_size": self.evaluation_batch_size,
            "evaluation_pairing": "same_fresh_maps_for_pre_fixed3_and_refresh5",
        })


def _git(repository: Path, *arguments: str) -> str:
    """Return stripped output from one read-only Git query."""
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _safe_new_root(output_root: Path, repository: Path) -> Path:
    """Create one empty external output directory without overwriting data."""
    if output_root.is_symlink():
        msg = "The output root must not be a symbolic link."
        raise ValueError(msg)
    root = output_root.resolve()
    source = repository.resolve()
    if root == source or source in root.parents:
        msg = "The output directory must be outside the repository."
        raise ValueError(msg)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        msg = "Existing output is never overwritten."
        raise ValueError(msg)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_study_root(output_root: Path, repository: Path, *, resume: bool) -> Path:
    """Open or create the external refresh-study root."""
    if output_root.is_symlink():
        msg = "The output root must not be a symbolic link."
        raise ValueError(msg)
    root = output_root.resolve()
    source = repository.resolve()
    if root == source or source in root.parents:
        msg = "The output directory must be outside the repository."
        raise ValueError(msg)
    nonempty = root.exists() and root.is_dir() and any(root.iterdir())
    if resume and not nonempty:
        msg = "--resume requires an existing nonempty study output."
        raise ValueError(msg)
    if not resume and (root.exists() and (not root.is_dir() or nonempty)):
        msg = "Existing study output is never overwritten; use --resume for an exact continuation."
        raise ValueError(msg)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_json(path: Path) -> dict[str, object]:
    """Load one canonical JSON object as a detached mapping."""
    return cast("dict[str, object]", thaw_json(load_canonical_json_object(path.read_text(encoding="utf-8"))))


def _path_below(root: Path, relative: object, *, name: str) -> Path:
    """Resolve one retained relative artifact path beneath its declared root."""
    if type(relative) is not str or Path(relative).is_absolute():
        msg = f"{name} must be a relative path."
        raise ValueError(msg)
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        msg = f"{name} escapes its declared artifact root."
        raise ValueError(msg)
    return candidate


def _sum_work(stage_results: Sequence[TrainingStageResult]) -> dict[str, int]:
    """Sum the exact normalized-work keys across completed stages."""
    keys = tuple(stage_results[0].normalized_work)
    return {key: sum(cast("int", result.normalized_work[key]) for result in stage_results) for key in keys}


def _reconstruct_calibration_result(calibration_root: Path) -> tuple[dict[str, object], TrainingPipelineResult]:
    """Reconstruct and verify the retained calibration's producer result."""
    root = calibration_root.resolve()
    result = _load_json(root / "calibration_result.json")
    if result.get("status") != "success" or result.get("terminal_status") != "completed":
        msg = "The source calibration did not complete successfully."
        raise ValueError(msg)
    store_name = result.get("artifact_store")
    store_root = _path_below(root, store_name, name="artifact_store")
    pipeline = TrainingPipelineConfig.from_dict(_load_json(store_root / "pipeline.json"))
    stage_results: list[TrainingStageResult] = []
    for line in (store_root / "stage_results.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = cast("dict[str, object]", thaw_json(load_canonical_json_object(line)))
            stage_results.append(TrainingStageResult.from_dict(row["stage_result"]))
    if len(stage_results) != 5:
        msg = "The reusable q8 calibration must contain exactly five completed stages."
        raise ValueError(msg)
    final = stage_results[-1]
    producer = TrainingPipelineResult(
        config=pipeline,
        stage_results=tuple(stage_results),
        final_checkpoint_path=final.produced_checkpoint_path,
        final_checkpoint_checksum=final.produced_checkpoint_checksum,
        final_checkpoint_provenance_checksum=final.checkpoint_provenance_checksum,
        wall_time_seconds=sum(stage.wall_time_seconds for stage in stage_results),
        peak_memory_bytes=max(stage.peak_memory_bytes for stage in stage_results),
        normalized_work=_sum_work(stage_results),
    )
    manifest = _load_json(store_root / "manifest.json")
    if manifest.get("completed_pipeline_result_checksum") != producer.content_checksum:
        msg = "The reconstructed producer result does not match the retained store manifest."
        raise ValueError(msg)
    return result, producer


def _verified_checkpoint_bytes(
    calibration_root: Path,
    relative_path: str,
    *,
    expected_checksum: str,
) -> bytes:
    """Read and strictly decode one calibration checkpoint."""
    path = _path_below(calibration_root.resolve(), relative_path, name="checkpoint_path")
    payload = path.read_bytes()
    StageParameterCheckpoint.from_bytes(payload, expected_checksum=expected_checksum)
    return payload


def export_pretrained_bundle(calibration_root: Path, output_root: Path) -> dict[str, object]:
    """Export a portable noiseless depth-four checkpoint and historical comparator."""
    root = _safe_new_root(output_root, _REPOSITORY_ROOT)
    source_root = calibration_root.resolve()
    calibration, producer = _reconstruct_calibration_result(source_root)
    reference = ExternalCheckpointRef.from_pipeline_result(producer, 3)
    pre = cast("Mapping[str, object]", calibration["pre_finetune_parameters"])
    post = cast("Mapping[str, object]", calibration["post_finetune_parameters"])
    pre_bytes = _verified_checkpoint_bytes(
        source_root,
        cast("str", pre["checkpoint_path"]),
        expected_checksum=reference.produced_checkpoint_checksum,
    )
    post_bytes = _verified_checkpoint_bytes(
        source_root,
        cast("str", post["checkpoint_path"]),
        expected_checksum=cast("str", post["checkpoint_checksum"]),
    )
    atomic_write_bytes(root / PRETRAINED_CHECKPOINT_NAME, pre_bytes)
    atomic_write_bytes(root / HISTORICAL_POST_CHECKPOINT_NAME, post_bytes)
    bundle = seal_mapping({
        "schema_version": PRETRAINED_BUNDLE_SCHEMA_VERSION,
        "evidence_classification": EVIDENCE_CLASSIFICATION,
        "evidence_warning": EVIDENCE_WARNING,
        "source_calibration_result_checksum": calibration["content_checksum"],
        "source_git_commit": calibration["git_commit"],
        "target_identity": calibration["target_identity"],
        "pretrained_checkpoint": {
            "path": PRETRAINED_CHECKPOINT_NAME,
            "checkpoint_checksum": reference.produced_checkpoint_checksum,
            "checkpoint_provenance_checksum": reference.checkpoint_provenance_checksum,
            "output_topology_id": reference.output_topology_id,
            "output_parameter_count": reference.output_parameter_count,
        },
        "historical_fixed3_post_checkpoint": {
            "path": HISTORICAL_POST_CHECKPOINT_NAME,
            "checkpoint_checksum": post["checkpoint_checksum"],
        },
        "external_checkpoint_ref": reference.to_dict(),
    })
    atomic_write_bytes(root / PRETRAINED_MANIFEST_NAME, f"{canonical_json(bundle)}\n".encode())
    return bundle


def _load_pretrained_bundle(root: Path) -> dict[str, object]:
    """Load and verify one exported pretrained-circuit bundle."""
    bundle = _load_json(root / PRETRAINED_MANIFEST_NAME)
    if bundle.get("schema_version") != PRETRAINED_BUNDLE_SCHEMA_VERSION:
        msg = "The pretrained bundle has an unsupported schema version."
        raise ValueError(msg)
    checksum = bundle.get("content_checksum")
    if checksum != canonical_checksum({key: value for key, value in bundle.items() if key != "content_checksum"}):
        msg = "The pretrained bundle content checksum is invalid."
        raise ValueError(msg)
    reference = ExternalCheckpointRef.from_dict(bundle["external_checkpoint_ref"])
    pre = cast("Mapping[str, object]", bundle["pretrained_checkpoint"])
    post = cast("Mapping[str, object]", bundle["historical_fixed3_post_checkpoint"])
    _verified_checkpoint_bytes(
        root,
        cast("str", pre["path"]),
        expected_checksum=reference.produced_checkpoint_checksum,
    )
    _verified_checkpoint_bytes(
        root,
        cast("str", post["path"]),
        expected_checksum=cast("str", post["checkpoint_checksum"]),
    )
    return bundle


def _copy_bundle_inputs(source_root: Path, study_root: Path) -> dict[str, object]:
    """Ingest the portable bundle into the study output before optimization."""
    _load_pretrained_bundle(source_root)
    inputs = study_root / STUDY_INPUT_DIRECTORY
    inputs.mkdir(parents=True, exist_ok=False)
    for name in (PRETRAINED_CHECKPOINT_NAME, HISTORICAL_POST_CHECKPOINT_NAME, PRETRAINED_MANIFEST_NAME):
        atomic_write_bytes(inputs / name, (source_root / name).read_bytes())
    return _load_pretrained_bundle(inputs)


def _build_refresh_pipeline(
    config: RefreshStudyConfig,
    bundle: Mapping[str, object],
    *,
    input_checkpoint_path: str,
) -> TrainingPipelineConfig:
    """Build a one-stage q8 treatment that consumes the reusable checkpoint."""
    reference = ExternalCheckpointRef.from_dict(bundle["external_checkpoint_ref"])
    producer = reference.producer_result.config
    historical_final = producer.template.stages[-1]
    policy = cast("dict[str, object]", thaw_json(historical_final.stage_policy))
    policy.update({
        "stage_index": 0,
        "stage_id": "refresh5_finetune",
        "parameter_transfer_rule": "load_checkpoint",
        "iteration_budget": config.finetune_updates,
        "trajectory_count": config.training_trajectories,
        "trajectory_update": "cross",
        "sampling_policy": "crn_refresh",
        "crn_refresh_interval": config.refresh_interval,
    })
    stage = TrainingStageTemplate(
        stage_policy=policy,
        seed_bindings={
            "initialization": None,
            "optimizer": f"refresh5_optimizer_repetition_{config.training_repetition}",
            "training": f"refresh5_training_repetition_{config.training_repetition}",
            "checkpoint_validation": None,
        },
    )
    template = TrainingPipelineTemplate(
        template_id="q8_cross_crn_refresh5_exploratory",
        preregistration_checksum=producer.template.preregistration_checksum,
        target_scope_id="legacy_reproduction",
        ansatz_family="bmpd_brickwall",
        method_id="layerwise_bmpd_cross_crn_refresh_q8_exploratory",
        method_version="q8-refresh5-exploratory-v1",
        resource_stratum_id="legacy_historical_unmatched",
        stages=(stage,),
        seed_domains=producer.template.seed_domains,
        final_materialization_policy=producer.template.final_materialization_policy,
    )
    return template.resolve(
        target_namespace=producer.target_namespace,
        target_manifest=None,
        target_instance_id=producer.target_instance_id,
        target_population_manifest_checksum=producer.target_population_manifest_checksum,
        target_instance_spec_checksum=producer.target_instance_spec_checksum,
        target_family_id=producer.target_family_id,
        target_stratum_id=producer.target_stratum_id,
        qubit_count=producer.qubit_count,
        optimization_block_id=producer.optimization_block_id,
        optimization_seed=producer.optimization_seed,
        data_role=producer.data_role,
        input_checkpoint_path=input_checkpoint_path,
        input_checkpoint_ref=reference,
    )


@dataclass(frozen=True, slots=True)
class _RefreshDepth4Runner:
    """Execute one depth-four stage from a verified external predecessor."""

    pipeline: TrainingPipelineConfig
    target: LegacyMaterializedTarget

    def __call__(
        self,
        stage: TrainingStageConfig,
        predecessor_parameters: NDArray[np.float64] | None,
    ) -> NoisyKrotovStageExecution | NoisyKrotovStageFailure:
        """Run the configured refresh stage without mutating the filesystem."""
        if stage != self.pipeline.stages[0] or predecessor_parameters is None:
            msg = "The refresh stage requires its exact configured pretrained checkpoint."
            raise ValueError(msg)
        binding = create_bmpd_circuit_binding(8, 4)
        return execute_fixed_rate_krotov_stage(stage, binding, self.target, predecessor_parameters)

    @staticmethod
    def circuit_statistics(stage: TrainingStageConfig) -> Mapping[str, object]:
        """Return the exact q8 depth-four logical statistics."""
        binding = create_bmpd_circuit_binding(8, 4)
        return {
            "topology_id": stage.output_topology_id,
            "parameter_count": stage.output_parameter_count,
            "qubit_count": 8,
            "bmpd_depth": 4,
            "logical_gate_count": len(binding.circuit.gates),
            "trainable_gate_count": binding.circuit.num_params,
        }


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    """Temporarily bind the artifact store's portable input root."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _capture_fingerprint(
    repository: Path,
    store_root: Path,
    pipeline: TrainingPipelineConfig,
    commit: str,
    dependencies: Mapping[str, str],
) -> ResumabilityFingerprint:
    """Capture tracked optimizer sources plus this explicit experiment runner."""
    tracked = _git(
        repository,
        "ls-files",
        "-z",
        "--",
        "benchmarks/state_preparation",
        "src/mqt/yaqs",
    )
    sources = [Path(item) for item in tracked.split("\0") if item.endswith(".py")]
    runner = Path(__file__).resolve().relative_to(repository.resolve())
    if runner not in sources:
        sources.append(runner)
    return capture_resumability_fingerprint(
        repository,
        output_root=store_root,
        starting_commit=commit,
        pipeline_prefix_id=pipeline.prefix_id(0),
        dependency_versions=dependencies,
        execution_source_paths=tuple(sources),
        lockfile_paths=_LOCKFILES,
        sealed_input_paths=_SEALED_INPUTS,
    )


def _summary(values: Sequence[float]) -> dict[str, float]:
    """Return finite sample summaries without hiding raw observations."""
    sample_std = stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": fmean(values),
        "sample_standard_deviation": sample_std,
        "standard_error": sample_std / len(values) ** 0.5,
    }


def _event_details(trajectory_maps: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Extract compact realized nonidentity-event metadata from one trajectory."""
    events: list[dict[str, object]] = []
    for gate_index, noise_map in enumerate(trajectory_maps):
        if noise_map["is_identity"] is False:
            events.append({
                "gate_index": gate_index,
                "source_gate_index": noise_map["source_gate_index"],
                "channel_id": noise_map["channel_id"],
                "jump_process_index": noise_map["jump_process_index"],
                "outcome_labels": noise_map["outcome_labels"],
            })
    return events


def _group_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize a paired subset of pre, historical, and refreshed outcomes."""
    pre = [cast("float", row["pretrained"]) for row in rows]
    fixed = [cast("float", row["historical_fixed3"]) for row in rows]
    refreshed = [cast("float", row["refresh5"]) for row in rows]
    refresh_minus_fixed = [new - old for new, old in zip(refreshed, fixed, strict=True)]
    return {
        "count": len(rows),
        "pretrained": _summary(pre),
        "historical_fixed3": _summary(fixed),
        "refresh5": _summary(refreshed),
        "refresh5_minus_historical_fixed3": _summary(refresh_minus_fixed),
        "refresh5_improvement_fraction": sum(value > 0.0 for value in refresh_minus_fixed) / len(rows),
        "historical_fixed3_fidelity_below_0p01_count": sum(value < 0.01 for value in fixed),
        "refresh5_fidelity_below_0p01_count": sum(value < 0.01 for value in refreshed),
    }


def _training_seed_set(stage: TrainingStageConfig) -> set[int]:
    """Derive every trajectory seed used by the periodic refresh schedule."""
    assert stage.training_seed is not None
    assert stage.crn_refresh_interval is not None
    count = math.ceil(stage.iteration_budget / stage.crn_refresh_interval)
    return {
        derive_krotov_trajectory_seed(
            role="training_trajectory",
            resolved_seed=stage.training_seed,
            stage_index=stage.stage_index,
            ensemble_index=ensemble_index,
            trajectory_index=trajectory_index,
            refresh_index=ensemble_index,
        )
        for ensemble_index in range(count)
        for trajectory_index in range(stage.trajectory_count)
    }


def _evaluate(
    config: RefreshStudyConfig,
    pipeline: TrainingPipelineConfig,
    target: LegacyMaterializedTarget,
    pretrained: NDArray[np.float64],
    historical_fixed: NDArray[np.float64],
    refreshed: NDArray[np.float64],
    configuration_checksum: str,
) -> dict[str, object]:
    """Run one batched, fresh, common-map comparison of all three circuits."""
    binding = create_bmpd_circuit_binding(8, 4)
    circuit = binding.circuit
    stage = pipeline.stages[0]
    provider = create_historical_fixed_rate_noise_provider()
    truncation = KrotovTruncation(
        max_bond_dim=stage.max_bond_dimension,
        svd_threshold=stage.svd_threshold,
        trunc_mode=stage.truncation_mode,
        min_bond_dim=stage.min_bond_dimension,
    )
    target_vector = target.state_vector_copy()
    rows: list[dict[str, object]] = []
    batches: list[dict[str, object]] = []
    evaluation_seeds: set[int] = set()
    batch_count = math.ceil(config.evaluation_trajectories / config.evaluation_batch_size)
    for batch_index in range(batch_count):
        start = batch_index * config.evaluation_batch_size
        count = min(config.evaluation_batch_size, config.evaluation_trajectories - start)
        options = KrotovTJMOptions(
            num_trajectories=count,
            random_seed=config.evaluation_seed,
            dt=1.0,
            apply_noise_to="all",
            noisy_gate_indices=binding.noisy_gate_indices,
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
            stage_configuration_checksum=configuration_checksum,
            circuit_checksum=binding.content_checksum,
            provider_checksum=provider.content_checksum,
            ensemble_index=batch_index,
            refresh_index=batch_index,
            global_iteration_start=start,
        )
        maps = ensemble.replay_maps()
        pre_values = noisy_state_preparation_metrics(
            circuit,
            pretrained,
            target_vector,
            None,
            options,
            truncation=truncation,
            fixed_noise_maps=maps,
            noise_provider=provider,
        )[2]
        fixed_values = noisy_state_preparation_metrics(
            circuit,
            historical_fixed,
            target_vector,
            None,
            options,
            truncation=truncation,
            fixed_noise_maps=maps,
            noise_provider=provider,
        )[2]
        refresh_values = noisy_state_preparation_metrics(
            circuit,
            refreshed,
            target_vector,
            None,
            options,
            truncation=truncation,
            fixed_noise_maps=maps,
            noise_provider=provider,
        )[2]
        ensemble_data = cast("Mapping[str, object]", ensemble.to_dict())
        trajectory_maps = cast("Sequence[Sequence[Mapping[str, object]]]", ensemble_data["trajectory_maps"])
        batch_seeds = [
            derive_krotov_trajectory_seed(
                role="pilot_evaluation",
                resolved_seed=config.evaluation_seed,
                stage_index=1,
                ensemble_index=batch_index,
                trajectory_index=index,
                refresh_index=batch_index,
            )
            for index in range(count)
        ]
        if evaluation_seeds.intersection(batch_seeds):
            msg = "Evaluation batches reused a trajectory seed."
            raise RefreshStudyExecutionError(msg)
        evaluation_seeds.update(batch_seeds)
        for local_index, (pre_value, fixed_value, refresh_value, maps_for_trajectory, seed) in enumerate(
            zip(pre_values, fixed_values, refresh_values, trajectory_maps, batch_seeds, strict=True)
        ):
            events = _event_details(maps_for_trajectory)
            rows.append({
                "trajectory_index": start + local_index,
                "trajectory_seed": seed,
                "batch_index": batch_index,
                "event_count": len(events),
                "events": events,
                "pretrained": pre_value,
                "historical_fixed3": fixed_value,
                "refresh5": refresh_value,
                "historical_fixed3_minus_pretrained": fixed_value - pre_value,
                "refresh5_minus_pretrained": refresh_value - pre_value,
                "refresh5_minus_historical_fixed3": refresh_value - fixed_value,
            })
        batches.append({
            "batch_index": batch_index,
            "trajectory_start": start,
            "trajectory_count": count,
            "ensemble_id": ensemble.ensemble_id,
            "ensemble_checksum": ensemble.content_checksum,
            "nonidentity_event_count": ensemble.nonidentity_event_count,
        })

    training_seeds = _training_seed_set(stage)
    if training_seeds.intersection(evaluation_seeds):
        msg = "Training and evaluation trajectory seed domains overlap."
        raise RefreshStudyExecutionError(msg)
    zero_event = [row for row in rows if row["event_count"] == 0]
    event_affected = [row for row in rows if cast("int", row["event_count"]) > 0]
    groups = {
        "all": _group_summary(rows),
        "zero_event": _group_summary(zero_event) if zero_event else None,
        "event_affected": _group_summary(event_affected) if event_affected else None,
    }
    return {
        "ensemble_role": "pilot_evaluation",
        "same_maps_used_for_all_three_circuits": True,
        "training_and_evaluation_seed_domains_disjoint": True,
        "training_trajectory_seed_count": len(training_seeds),
        "evaluation_trajectory_seed_count": len(evaluation_seeds),
        "ensemble_batches": batches,
        "noiseless_fidelities": {
            "pretrained": state_preparation_metrics(circuit, pretrained, target_vector, truncation=truncation)[1],
            "historical_fixed3": state_preparation_metrics(
                circuit, historical_fixed, target_vector, truncation=truncation
            )[1],
            "refresh5": state_preparation_metrics(circuit, refreshed, target_vector, truncation=truncation)[1],
        },
        "group_summaries": groups,
        "paired_trajectory_fidelities": rows,
        "normalized_evaluation_work": {
            "objective_evaluations": 3 * batch_count + 3,
            "gradient_evaluations": 0,
            "training_trajectories": 0,
            "checkpoint_validation_trajectories": 0,
            "test_trajectories": 3 * config.evaluation_trajectories,
            "trajectory_gate_applications": 4 * config.evaluation_trajectories * len(circuit.gates),
        },
    }


def _training_schedule_summary(
    store: Phase2ArtifactStore,
    store_root: Path,
    config: RefreshStudyConfig,
) -> dict[str, object]:
    """Summarize every persisted refreshed training ensemble."""
    artifact = store.stage_artifacts[0]
    refs = [ref for ref in artifact.fixed_map_artifacts if ref.role == "training_trajectory"]
    rows = []
    for ref in refs:
        ensemble = _load_json(store_root / ref.path)
        rows.append({
            "ensemble_id": ensemble["ensemble_id"],
            "ensemble_checksum": ensemble["content_checksum"],
            "ensemble_index": ensemble["ensemble_index"],
            "refresh_index": ensemble["refresh_index"],
            "global_iteration_start": ensemble["global_iteration_start"],
            "trajectory_count": ensemble["trajectory_count"],
            "nonidentity_event_count": ensemble["nonidentity_event_count"],
        })
    rows.sort(key=lambda row: cast("int", row["ensemble_index"]))
    if len(rows) != config.expected_training_ensemble_count:
        msg = "Persisted training ensembles do not match the configured refresh schedule."
        raise RefreshStudyExecutionError(msg)
    starts = [cast("int", row["global_iteration_start"]) for row in rows]
    if starts != list(range(0, config.finetune_updates, config.refresh_interval)):
        msg = "Persisted training refresh boundaries do not match the configured cadence."
        raise RefreshStudyExecutionError(msg)
    return {
        "ensemble_count": len(rows),
        "unique_trajectory_coordinate_count": len(rows) * config.training_trajectories,
        "nonidentity_event_count": sum(cast("int", row["nonidentity_event_count"]) for row in rows),
        "ensembles": rows,
    }


def _checkpoint_from_inputs(root: Path, bundle: Mapping[str, object], key: str) -> StageParameterCheckpoint:
    """Load one verified checkpoint copied into the self-contained study input."""
    record = cast("Mapping[str, object]", bundle[key])
    payload = (root / STUDY_INPUT_DIRECTORY / cast("str", record["path"])).read_bytes()
    return StageParameterCheckpoint.from_bytes(
        payload,
        expected_checksum=cast("str", record["checkpoint_checksum"]),
    )


def run_refresh_study(
    output_root: Path,
    *,
    pretrained_root: Path | None,
    config: RefreshStudyConfig | None = None,
    resume: bool = False,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Execute or exactly resume the fine-tuning-only refresh study."""
    config = RefreshStudyConfig() if config is None else config
    repository = _REPOSITORY_ROOT if repository_root is None else repository_root.resolve()
    root = _safe_study_root(output_root, repository, resume=resume)
    inputs = root / STUDY_INPUT_DIRECTORY
    if resume:
        bundle = _load_pretrained_bundle(inputs)
    else:
        if pretrained_root is None:
            msg = "A pretrained bundle is required for a new refresh study."
            raise ValueError(msg)
        bundle = _copy_bundle_inputs(pretrained_root.resolve(), root)
        notice = f"{EVIDENCE_CLASSIFICATION}: every file below this directory is {EVIDENCE_WARNING}.\n"
        atomic_write_bytes(root / STUDY_NOTICE_NAME, notice.encode())
    bundle_checksum = cast("str", bundle["content_checksum"])
    config_document = config.to_dict(pretrained_bundle_checksum=bundle_checksum)
    config_path = root / STUDY_CONFIG_NAME
    if resume:
        retained_config = _load_json(config_path)
        if retained_config != config_document:
            msg = "The active refresh-study configuration differs from the retained configuration."
            raise ValueError(msg)
    else:
        atomic_write_bytes(config_path, f"{canonical_json(config_document)}\n".encode())

    input_root = root.parent
    input_checkpoint_path = str((root / STUDY_INPUT_DIRECTORY / PRETRAINED_CHECKPOINT_NAME).relative_to(input_root))
    pipeline = _build_refresh_pipeline(config, bundle, input_checkpoint_path=input_checkpoint_path)
    commit = _git(repository, "rev-parse", "HEAD")
    dependencies = {**{name: version(name) for name in _DEPENDENCIES}, "python": platform.python_version()}
    store_root = root / STUDY_STORE_NAME
    fingerprint = _capture_fingerprint(repository, store_root, pipeline, commit, dependencies)
    with _working_directory(input_root):
        store = Phase2ArtifactStore(store_root, pipeline, fingerprint, resume=resume)
    result_path = root / STUDY_RESULT_NAME
    if resume and result_path.is_file():
        return _load_json(result_path)

    target_identity = cast("Mapping[str, object]", bundle["target_identity"])
    target = load_legacy_target_collection().target(cast("str", target_identity["target_instance_id"]))
    runner = _RefreshDepth4Runner(pipeline, target)
    started = time.perf_counter()
    training = Phase2PipelineExecutor(store).execute(runner, circuit_statistics=runner.circuit_statistics)
    if isinstance(training, PipelineExecutionFailure):
        msg = f"Refresh training failed during {training.failure.stage_id}: {training.failure.message}"
        raise RefreshStudyExecutionError(msg)
    training_schedule = _training_schedule_summary(store, store_root, config)
    pretrained = _checkpoint_from_inputs(root, bundle, "pretrained_checkpoint").selected_theta
    historical_fixed = _checkpoint_from_inputs(
        root,
        bundle,
        "historical_fixed3_post_checkpoint",
    ).selected_theta
    refreshed = store.load_final_parameters()
    evaluation_started = time.perf_counter()
    evaluation = _evaluate(
        config,
        pipeline,
        target,
        pretrained,
        historical_fixed,
        refreshed,
        cast("str", config_document["content_checksum"]),
    )
    evaluation_wall_time = time.perf_counter() - evaluation_started
    stage_artifact = store.stage_artifacts[0].stage_result
    result = seal_mapping({
        "schema_version": REFRESH_STUDY_SCHEMA_VERSION,
        "status": "success",
        "terminal_status": "completed",
        "evidence_classification": EVIDENCE_CLASSIFICATION,
        "evidence_warning": EVIDENCE_WARNING,
        "configuration_checksum": config_document["content_checksum"],
        "pretrained_bundle_checksum": bundle_checksum,
        "git_commit": commit,
        "runner_source_checksum": f"sha256:{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
        "dependency_versions": dependencies,
        "target_identity": target.identity_dict(),
        "pipeline_configuration_checksum": pipeline.configuration_checksum,
        "runtime_fingerprint_checksum": fingerprint.content_checksum,
        "artifact_store": STUDY_STORE_NAME,
        "pretrained_checkpoint": bundle["pretrained_checkpoint"],
        "historical_fixed3_post_checkpoint": bundle["historical_fixed3_post_checkpoint"],
        "refresh5_post_checkpoint": {
            "path": f"{STUDY_STORE_NAME}/{stage_artifact.produced_checkpoint_path}",
            "checkpoint_checksum": stage_artifact.produced_checkpoint_checksum,
        },
        "optimizer_trace": {
            "path": f"{STUDY_STORE_NAME}/{stage_artifact.optimizer_trace_path}",
            "checksum": stage_artifact.optimizer_trace_checksum,
        },
        "training_schedule": training_schedule,
        "paired_fresh_evaluation": evaluation,
        "training_wall_time_seconds": training.wall_time_seconds,
        "evaluation_wall_time_seconds": evaluation_wall_time,
        "total_wall_time_seconds": time.perf_counter() - started,
        "peak_training_memory_bytes": training.peak_memory_bytes,
        "normalized_training_work": dict(training.normalized_work),
    })
    atomic_write_bytes(result_path, f"{canonical_json(result)}\n".encode())
    return result


def _parser() -> argparse.ArgumentParser:
    """Build the two-command exploratory interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-pretrained", help="Export the reusable depth-four checkpoint bundle.")
    export.add_argument("--calibration-root", type=Path, required=True)
    export.add_argument("--output-root", type=Path, required=True)
    run = commands.add_parser("run", help="Run fine-tuning only from a reusable checkpoint.")
    run.add_argument("--pretrained-root", type=Path)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--execute", action="store_true", help="Required acknowledgement for q8 optimization.")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--finetune-updates", type=int, default=DEFAULT_FINETUNE_UPDATES)
    run.add_argument("--training-trajectories", type=int, default=DEFAULT_TRAINING_TRAJECTORIES)
    run.add_argument("--refresh-interval", type=int, default=DEFAULT_REFRESH_INTERVAL)
    run.add_argument("--training-repetition", type=int, default=0)
    run.add_argument("--evaluation-trajectories", type=int, default=DEFAULT_EVALUATION_TRAJECTORIES)
    run.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    run.add_argument("--evaluation-batch-size", type=int, default=DEFAULT_EVALUATION_BATCH_SIZE)
    return parser


def _export_command(arguments: argparse.Namespace) -> int:
    """Execute the non-numerical pretrained-bundle export command."""
    try:
        result = export_pretrained_bundle(arguments.calibration_root, arguments.output_root)
    except Exception as error:
        print(f"Pretrained export failed: {error}", file=sys.stderr)
        return 2
    print(
        canonical_json({
            "status": "success",
            "pretrained_manifest": str(arguments.output_root / PRETRAINED_MANIFEST_NAME),
            "content_checksum": result["content_checksum"],
        })
    )
    return 0


def _run_command(arguments: argparse.Namespace) -> int:
    """Execute the opt-in numerical refresh-study command."""
    if not arguments.execute:
        print("Refresh study not started: pass --execute to acknowledge the q8 run.", file=sys.stderr)
        return 2
    try:
        result = run_refresh_study(
            arguments.output_root,
            pretrained_root=arguments.pretrained_root,
            config=RefreshStudyConfig(
                finetune_updates=arguments.finetune_updates,
                training_trajectories=arguments.training_trajectories,
                refresh_interval=arguments.refresh_interval,
                training_repetition=arguments.training_repetition,
                evaluation_trajectories=arguments.evaluation_trajectories,
                evaluation_seed=arguments.evaluation_seed,
                evaluation_batch_size=arguments.evaluation_batch_size,
            ),
            resume=arguments.resume,
        )
    except Exception as error:
        print(f"Refresh study failed: {error}", file=sys.stderr)
        return 2
    print(
        canonical_json({
            "status": result["status"],
            "terminal_status": result["terminal_status"],
            "result_path": str(arguments.output_root / STUDY_RESULT_NAME),
            "content_checksum": result["content_checksum"],
        })
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Export the reusable circuit or execute the opt-in refresh study."""
    arguments = _parser().parse_args(argv)
    return _export_command(arguments) if arguments.command == "export-pretrained" else _run_command(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
