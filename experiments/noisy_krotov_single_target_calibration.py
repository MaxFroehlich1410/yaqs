# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run one exploratory q8 timing and held-out-fidelity calibration.

The default configuration executes the exact five-stage WP19 training sequence
for legacy target seed 100.  It then compares the depth-four parameters from
immediately before and after noisy fine-tuning on one fresh paired ensemble.
All output is exploratory and is not confirmatory evidence.
"""

# The CLI reports terminal failures, and Git is invoked only with fixed
# read-only arguments assembled in this module.
# Direct execution adds the repository root before importing benchmark modules.
# ruff: noqa: BLE001, DOC201, DOC501, E402, S603, S607, T201

from __future__ import annotations

import argparse
import hashlib
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
from benchmarks.state_preparation.phase2.artifacts import Phase2ArtifactStore
from benchmarks.state_preparation.phase2.canonical import (
    canonical_json,
    load_canonical_json_object,
    seal_mapping,
    thaw_json,
)
from benchmarks.state_preparation.phase2.execution import Phase2PipelineExecutor, PipelineExecutionFailure
from benchmarks.state_preparation.phase2.layerwise_bmpd import (
    LAYERWISE_BMPD_FINETUNE_ITERATIONS,
    LAYERWISE_BMPD_GROWTH_ITERATIONS,
    LEGACY_EVALUATION_SEED,
    LEGACY_EVALUATION_TRAJECTORY_COUNT,
    LayerwiseBMPDStageRunner,
    create_bmpd_circuit_binding,
    resolve_layerwise_bmpd_crn_legacy_v1_pipeline,
)
from benchmarks.state_preparation.phase2.legacy_targets import load_legacy_target_collection
from benchmarks.state_preparation.phase2.pipeline import TrainingPipelineTemplate, TrainingStageTemplate
from benchmarks.state_preparation.phase2.resumability import capture_resumability_fingerprint
from benchmarks.state_preparation.reporting import atomic_write_bytes
from mqt.yaqs.optimization import (
    KrotovTJMOptions,
    KrotovTruncation,
    derive_legacy_krotov_trajectory_seed,
    noisy_state_preparation_metrics,
    sample_krotov_fixed_map_ensemble,
    state_preparation_metrics,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from benchmarks.state_preparation.phase2.legacy_targets import LegacyMaterializedTarget
    from benchmarks.state_preparation.phase2.pipeline import TrainingPipelineConfig, TrainingPipelineResult
    from benchmarks.state_preparation.phase2.resumability import ResumabilityFingerprint


CALIBRATION_SCHEMA_VERSION = "yaqs.noisy_krotov.single_target_calibration.v1"
CALIBRATION_CONFIG_NAME = "calibration_config.json"
CALIBRATION_RESULT_NAME = "calibration_result.json"
CALIBRATION_NOTICE_NAME = "EXPLORATORY_NOT_CONFIRMATORY.txt"
CALIBRATION_STORE_NAME = "exploratory_not_confirmatory_phase2_store"
EVIDENCE_CLASSIFICATION = "exploratory"
EVIDENCE_WARNING = "not confirmatory evidence"
DEFAULT_TARGET_SEED = 100
_DEPENDENCIES = ("mqt.yaqs", "numpy", "scipy")
_LOCKFILES = (Path("pyproject.toml"), Path("uv.lock"))
_SEALED_INPUTS = (
    Path("benchmarks/state_preparation/phase2/data/initial_preregistration_v1.json"),
    Path("benchmarks/state_preparation/phase2/data/legacy_evidence_audit_v1.json"),
    Path("benchmarks/state_preparation/phase2/data/legacy_tfim_targets_v1.json"),
)


class CalibrationExecutionError(RuntimeError):
    """Raised when a configured calibration stage fails numerically."""


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Explicit scientific configuration for one exploratory calibration."""

    target_seed: int = DEFAULT_TARGET_SEED
    growth_updates: int = LAYERWISE_BMPD_GROWTH_ITERATIONS
    finetune_updates: int = LAYERWISE_BMPD_FINETUNE_ITERATIONS
    evaluation_trajectories: int = LEGACY_EVALUATION_TRAJECTORY_COUNT
    evaluation_seed: int = LEGACY_EVALUATION_SEED

    def __post_init__(self) -> None:
        """Validate the deliberately narrow q8 configuration."""
        if self.target_seed not in {100, 200, 300, 400, 500}:
            msg = "target_seed must identify one audited legacy q8 target."
            raise ValueError(msg)
        for name in ("growth_updates", "finetune_updates", "evaluation_trajectories"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                msg = f"{name} must be a positive integer."
                raise ValueError(msg)
        if type(self.evaluation_seed) is not int or self.evaluation_seed < 0:
            msg = "evaluation_seed must be a nonnegative integer."
            raise ValueError(msg)

    @property
    def exact_historical_training_sequence(self) -> bool:
        """Whether stage budgets equal the frozen WP19 training sequence."""
        return (
            self.growth_updates == LAYERWISE_BMPD_GROWTH_ITERATIONS
            and self.finetune_updates == LAYERWISE_BMPD_FINETUNE_ITERATIONS
        )

    def to_dict(self) -> dict[str, object]:
        """Return a checksum-sealed canonical configuration."""
        return seal_mapping({
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "evidence_classification": EVIDENCE_CLASSIFICATION,
            "evidence_warning": EVIDENCE_WARNING,
            "target_seed": self.target_seed,
            "qubit_count": 8,
            "growth_depths": [1, 2, 3, 4],
            "growth_updates_per_depth": self.growth_updates,
            "finetune_updates": self.finetune_updates,
            "training_trajectories": 3,
            "evaluation_trajectories": self.evaluation_trajectories,
            "evaluation_seed": self.evaluation_seed,
            "evaluation_pairing": "same_fresh_heldout_maps_for_pre_and_post",
            "exact_historical_training_sequence": self.exact_historical_training_sequence,
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


def _resolved_pipeline(config: CalibrationConfig) -> TrainingPipelineConfig:
    """Resolve the exact historical pipeline or an explicitly bounded variant."""
    historical = resolve_layerwise_bmpd_crn_legacy_v1_pipeline(config.target_seed)
    if config.exact_historical_training_sequence:
        return historical
    stages = tuple(
        TrainingStageTemplate(
            stage_policy={
                **cast("dict[str, object]", thaw_json(stage.stage_policy)),
                "iteration_budget": (
                    config.finetune_updates if stage.stage_id == "final_finetune" else config.growth_updates
                ),
            },
            seed_bindings=stage.seed_bindings,
        )
        for stage in historical.template.stages
    )
    template = TrainingPipelineTemplate(
        template_id="layerwise_bmpd_crn_legacy_v1_exploratory_calibration",
        preregistration_checksum=historical.template.preregistration_checksum,
        target_scope_id=historical.template.target_scope_id,
        ansatz_family=historical.template.ansatz_family,
        method_id=historical.method_id,
        method_version="1-exploratory-calibration",
        resource_stratum_id=historical.template.resource_stratum_id,
        stages=stages,
        seed_domains=historical.template.seed_domains,
        final_materialization_policy=historical.template.final_materialization_policy,
    )
    return template.resolve(
        target_namespace="legacy_reproduction",
        target_manifest=None,
        target_instance_id=historical.target_instance_id,
        target_population_manifest_checksum=historical.target_population_manifest_checksum,
        target_instance_spec_checksum=historical.target_instance_spec_checksum,
        target_family_id=historical.target_family_id,
        target_stratum_id=historical.target_stratum_id,
        qubit_count=historical.qubit_count,
        optimization_block_id=historical.optimization_block_id,
        optimization_seed=historical.optimization_seed,
        data_role="secondary_benchmark",
    )


def _safe_output_root(output_root: Path, repository: Path, *, resume: bool) -> Path:
    """Require an external, dedicated, non-overwritten output directory."""
    if output_root.is_symlink():
        msg = "The calibration output root must not be a symbolic link."
        raise ValueError(msg)
    root = output_root.resolve()
    source = repository.resolve()
    if root == source or source in root.parents:
        msg = "The calibration output directory must be outside the repository."
        raise ValueError(msg)
    if root.exists() and not root.is_dir():
        msg = "The calibration output root must be a directory."
        raise ValueError(msg)
    nonempty = root.exists() and any(root.iterdir())
    if resume and not nonempty:
        msg = "--resume requires an existing calibration output."
        raise ValueError(msg)
    if not resume and nonempty:
        msg = "Existing calibration output is never overwritten; use --resume for an exact continuation."
        raise ValueError(msg)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _capture_fingerprint(
    repository: Path,
    store_root: Path,
    pipeline: TrainingPipelineConfig,
    commit: str,
    dependencies: Mapping[str, str],
) -> ResumabilityFingerprint:
    """Capture the tracked scientific source and environment identity."""
    tracked = _git(
        repository,
        "ls-files",
        "-z",
        "--",
        "benchmarks/state_preparation",
        "src/mqt/yaqs",
        "experiments/noisy_krotov_single_target_calibration.py",
    )
    sources = tuple(Path(item) for item in tracked.split("\0") if item.endswith(".py"))
    return capture_resumability_fingerprint(
        repository,
        output_root=store_root,
        starting_commit=commit,
        pipeline_prefix_id=pipeline.prefix_id(len(pipeline.stages) - 1),
        dependency_versions=dependencies,
        execution_source_paths=sources,
        lockfile_paths=_LOCKFILES,
        sealed_input_paths=_SEALED_INPUTS,
    )


def _summary(values: Sequence[float]) -> dict[str, float]:
    """Return finite sample summaries without hiding the raw values."""
    mean = fmean(values)
    sample_std = stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "sample_standard_deviation": sample_std,
        "standard_error": sample_std / len(values) ** 0.5,
    }


def _paired_evaluation(
    config: CalibrationConfig,
    pipeline: TrainingPipelineConfig,
    training: TrainingPipelineResult,
    store: Phase2ArtifactStore,
    target: LegacyMaterializedTarget,
) -> dict[str, object]:
    """Evaluate pre/post parameters on one disjoint paired held-out ensemble."""
    binding = create_bmpd_circuit_binding(8, 4)
    circuit = binding.circuit
    pre = store.load_stage_checkpoint(3).selected_theta
    post = store.load_final_parameters()
    stage = pipeline.stages[-1]
    training_seed = cast("int", stage.training_seed)
    training_trajectory_seeds = {
        derive_legacy_krotov_trajectory_seed(optimizer_iteration_seed=training_seed, trajectory_index=index)
        for index in range(stage.trajectory_count)
    }
    evaluation_trajectory_seeds = tuple(
        derive_legacy_krotov_trajectory_seed(
            optimizer_iteration_seed=config.evaluation_seed,
            trajectory_index=index,
        )
        for index in range(config.evaluation_trajectories)
    )
    if training_trajectory_seeds.intersection(evaluation_trajectory_seeds):
        msg = "Fresh evaluation trajectory seeds overlap the fixed training CRN domain."
        raise ValueError(msg)

    provider = create_historical_fixed_rate_noise_provider()
    provider_checksum = provider.content_checksum
    truncation = KrotovTruncation(
        max_bond_dim=stage.max_bond_dimension,
        svd_threshold=stage.svd_threshold,
        trunc_mode=stage.truncation_mode,
        min_bond_dim=stage.min_bond_dimension,
    )
    options = KrotovTJMOptions(
        num_trajectories=config.evaluation_trajectories,
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
        pre,
        None,
        truncation,
        provider,
        options,
        role="pilot_evaluation",
        resolved_seed=config.evaluation_seed,
        stage_index=len(pipeline.stages),
        stage_id="calibration_heldout_evaluation",
        stage_configuration_checksum=cast("str", config.to_dict()["content_checksum"]),
        circuit_checksum=binding.content_checksum,
        provider_checksum=provider_checksum,
        ensemble_index=0,
        refresh_index=0,
        global_iteration_start=0,
        legacy_linear_seed=True,
        legacy_compact_replay=False,
    )
    target_vector = target.state_vector_copy()
    pre_noiseless = state_preparation_metrics(circuit, pre, target_vector, truncation=truncation)[1]
    post_noiseless = state_preparation_metrics(circuit, post, target_vector, truncation=truncation)[1]
    pre_fidelities = tuple(
        noisy_state_preparation_metrics(
            circuit,
            pre,
            target_vector,
            None,
            options,
            truncation=truncation,
            iteration=0,
            fixed_noise_maps=ensemble.replay_maps(),
            noise_provider=provider,
        )[2]
    )
    post_fidelities = tuple(
        noisy_state_preparation_metrics(
            circuit,
            post,
            target_vector,
            None,
            options,
            truncation=truncation,
            iteration=0,
            fixed_noise_maps=ensemble.replay_maps(),
            noise_provider=provider,
        )[2]
    )
    differences = tuple(
        post_value - pre_value for pre_value, post_value in zip(pre_fidelities, post_fidelities, strict=True)
    )
    gate_count = len(circuit.gates)
    evaluation_work = {
        "objective_evaluations": 4,
        "gradient_evaluations": 0,
        "training_trajectories": 0,
        "checkpoint_validation_trajectories": 0,
        "test_trajectories": 2 * config.evaluation_trajectories,
        "trajectory_gate_applications": 3 * config.evaluation_trajectories * gate_count,
    }
    total_work = {name: cast("int", training.normalized_work[name]) + value for name, value in evaluation_work.items()}
    return {
        "ensemble_role": ensemble.role,
        "ensemble_checksum": ensemble.content_checksum,
        "ensemble_resolved_seed": ensemble.resolved_seed,
        "trajectory_seeds": list(evaluation_trajectory_seeds),
        "training_and_evaluation_seed_domains_disjoint": True,
        "same_ensemble_used_for_pre_and_post": True,
        "pre_noiseless_fidelity": pre_noiseless,
        "post_noiseless_fidelity": post_noiseless,
        "paired_trajectory_fidelities": [
            {"trajectory_index": index, "pre": pre_value, "post": post_value, "difference": difference}
            for index, (pre_value, post_value, difference) in enumerate(
                zip(pre_fidelities, post_fidelities, differences, strict=True)
            )
        ],
        "pre_summary": _summary(pre_fidelities),
        "post_summary": _summary(post_fidelities),
        "difference_summary": _summary(differences),
        "normalized_evaluation_work": evaluation_work,
        "normalized_total_work": total_work,
    }


def run_calibration(
    output_root: Path,
    *,
    config: CalibrationConfig | None = None,
    resume: bool = False,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Execute or exactly resume one exploratory single-target calibration."""
    config = CalibrationConfig() if config is None else config
    repository = _REPOSITORY_ROOT if repository_root is None else repository_root.resolve()
    root = _safe_output_root(output_root, repository, resume=resume)
    config_document = config.to_dict()
    config_path = root / CALIBRATION_CONFIG_NAME
    if resume:
        stored_config = load_canonical_json_object(config_path.read_text(encoding="utf-8"))
        if thaw_json(stored_config) != thaw_json(config_document):
            msg = "The active calibration configuration differs from the retained configuration."
            raise ValueError(msg)
    else:
        atomic_write_bytes(config_path, f"{canonical_json(config_document)}\n".encode())
        notice = f"{EVIDENCE_CLASSIFICATION}: every file below this directory is {EVIDENCE_WARNING}.\n"
        atomic_write_bytes(root / CALIBRATION_NOTICE_NAME, notice.encode())

    pipeline = _resolved_pipeline(config)
    commit = _git(repository, "rev-parse", "HEAD")
    dependencies = {**{name: version(name) for name in _DEPENDENCIES}, "python": platform.python_version()}
    store_root = root / CALIBRATION_STORE_NAME
    fingerprint = _capture_fingerprint(repository, store_root, pipeline, commit, dependencies)
    store = Phase2ArtifactStore(store_root, pipeline, fingerprint, resume=resume)
    result_path = root / CALIBRATION_RESULT_NAME
    if resume and result_path.is_file():
        return cast(
            "dict[str, object]",
            thaw_json(load_canonical_json_object(result_path.read_text(encoding="utf-8"))),
        )

    target = load_legacy_target_collection().target(pipeline.target_instance_id)
    runner = LayerwiseBMPDStageRunner(pipeline, target)
    started = time.perf_counter()
    training = Phase2PipelineExecutor(store).execute(runner, circuit_statistics=runner.circuit_statistics)
    if isinstance(training, PipelineExecutionFailure):
        msg = f"Calibration training failed during {training.failure.stage_id}: {training.failure.message}"
        raise CalibrationExecutionError(msg)
    evaluation_started = time.perf_counter()
    evaluation = _paired_evaluation(config, pipeline, training, store, target)
    evaluation_wall_time = time.perf_counter() - evaluation_started
    stage_artifacts = store.stage_artifacts
    result = seal_mapping({
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "status": "success",
        "evidence_classification": EVIDENCE_CLASSIFICATION,
        "evidence_warning": EVIDENCE_WARNING,
        "configuration_checksum": config_document["content_checksum"],
        "git_commit": commit,
        "runner_source_checksum": f"sha256:{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
        "dependency_versions": dependencies,
        "target_identity": target.identity_dict(),
        "pipeline_configuration_checksum": pipeline.configuration_checksum,
        "runtime_fingerprint_checksum": fingerprint.content_checksum,
        "artifact_store": CALIBRATION_STORE_NAME,
        "stage_traces": [
            {
                "stage_index": artifact.stage_result.stage_index,
                "stage_id": artifact.stage_result.stage_id,
                "trace_path": f"{CALIBRATION_STORE_NAME}/{artifact.stage_result.optimizer_trace_path}",
                "trace_checksum": artifact.stage_result.optimizer_trace_checksum,
                "normalized_work": dict(artifact.stage_result.normalized_work),
            }
            for artifact in stage_artifacts
        ],
        "pre_finetune_parameters": {
            "checkpoint_path": (f"{CALIBRATION_STORE_NAME}/{stage_artifacts[3].stage_result.produced_checkpoint_path}"),
            "checkpoint_checksum": stage_artifacts[3].stage_result.produced_checkpoint_checksum,
        },
        "post_finetune_parameters": {
            "checkpoint_path": (f"{CALIBRATION_STORE_NAME}/{stage_artifacts[4].stage_result.produced_checkpoint_path}"),
            "checkpoint_checksum": stage_artifacts[4].stage_result.produced_checkpoint_checksum,
        },
        "paired_fresh_evaluation": evaluation,
        "training_wall_time_seconds": training.wall_time_seconds,
        "evaluation_wall_time_seconds": evaluation_wall_time,
        "total_wall_time_seconds": time.perf_counter() - started,
        "peak_training_memory_bytes": training.peak_memory_bytes,
        "terminal_status": "completed",
    })
    atomic_write_bytes(result_path, f"{canonical_json(result)}\n".encode())
    return result


def _parser() -> argparse.ArgumentParser:
    """Build the explicit exploratory command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Required acknowledgement for the q8 optimization.")
    parser.add_argument("--resume", action="store_true", help="Verify and continue the exact existing run.")
    parser.add_argument("--target-seed", type=int, default=DEFAULT_TARGET_SEED)
    parser.add_argument("--growth-updates", type=int, default=LAYERWISE_BMPD_GROWTH_ITERATIONS)
    parser.add_argument("--finetune-updates", type=int, default=LAYERWISE_BMPD_FINETUNE_ITERATIONS)
    parser.add_argument("--evaluation-trajectories", type=int, default=LEGACY_EVALUATION_TRAJECTORY_COUNT)
    parser.add_argument("--evaluation-seed", type=int, default=LEGACY_EVALUATION_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the opt-in calibration CLI."""
    arguments = _parser().parse_args(argv)
    if not arguments.execute:
        print("Calibration not started: pass --execute to acknowledge the q8 run.", file=sys.stderr)
        return 2
    try:
        result = run_calibration(
            arguments.output_root,
            config=CalibrationConfig(
                target_seed=arguments.target_seed,
                growth_updates=arguments.growth_updates,
                finetune_updates=arguments.finetune_updates,
                evaluation_trajectories=arguments.evaluation_trajectories,
                evaluation_seed=arguments.evaluation_seed,
            ),
            resume=arguments.resume,
        )
    except Exception as error:
        print(f"Calibration failed: {error}", file=sys.stderr)
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
