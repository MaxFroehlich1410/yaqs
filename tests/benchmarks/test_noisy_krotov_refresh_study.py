# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Focused tests for the fine-tuning-only q8 refresh study."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.resumability import capture_resumability_fingerprint
from experiments import noisy_krotov_refresh_study as study_module
from experiments.noisy_krotov_refresh_study import (
    HISTORICAL_POST_CHECKPOINT_NAME,
    PRETRAINED_CHECKPOINT_NAME,
    PRETRAINED_MANIFEST_NAME,
    STUDY_RESULT_NAME,
    RefreshStudyConfig,
    export_pretrained_bundle,
    run_refresh_study,
)
from experiments.noisy_krotov_single_target_calibration import CalibrationConfig, run_calibration

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pytest

    from benchmarks.state_preparation.phase2.pipeline import TrainingPipelineConfig
    from benchmarks.state_preparation.phase2.resumability import ResumabilityFingerprint


def test_refresh_configuration_counts_periodic_trajectory_exposure() -> None:
    """Eleven updates at cadence five use three ensembles and nine coordinates."""
    config = RefreshStudyConfig(
        finetune_updates=11,
        training_trajectories=3,
        refresh_interval=5,
        evaluation_trajectories=2,
        evaluation_batch_size=1,
    )

    assert config.expected_training_ensemble_count == 3
    assert config.expected_unique_training_trajectories == 9
    document = config.to_dict(pretrained_bundle_checksum="sha256:" + "1" * 64)
    assert document["training_sampling_policy"] == "crn_refresh"
    assert document["crn_refresh_interval"] == 5
    assert document["content_checksum"] == canonical_checksum({
        key: value for key, value in document.items() if key != "content_checksum"
    })


def test_pretrained_bundle_drives_bounded_refresh_and_exact_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retained depth-four checkpoint skips growth and remains provenance-bound."""
    calibration_root = tmp_path / "calibration"
    run_calibration(
        calibration_root,
        config=CalibrationConfig(
            growth_updates=1,
            finetune_updates=1,
            evaluation_trajectories=2,
        ),
    )
    bundle_root = tmp_path / "pretrained"
    bundle = export_pretrained_bundle(calibration_root, bundle_root)

    assert (bundle_root / PRETRAINED_CHECKPOINT_NAME).is_file()
    assert (bundle_root / HISTORICAL_POST_CHECKPOINT_NAME).is_file()
    assert (bundle_root / PRETRAINED_MANIFEST_NAME).is_file()
    assert cast("Mapping[str, object]", bundle["pretrained_checkpoint"])["output_parameter_count"] == 276

    # Production execution requires the new runner to exist in the pinned Git
    # commit. This test substitutes an otherwise identical tracked-source
    # fingerprint so a pre-commit worktree can exercise the numerical path.
    def development_fingerprint(
        repository: Path,
        store_root: Path,
        pipeline: TrainingPipelineConfig,
        commit: str,
        dependencies: Mapping[str, str],
    ) -> ResumabilityFingerprint:
        tracked = study_module._git(  # ruff: ignore[private-member-access]
            repository,
            "ls-files",
            "-z",
            "--",
            "benchmarks/state_preparation",
            "src/mqt/yaqs",
        )
        sources = tuple(Path(item) for item in tracked.split("\0") if item.endswith(".py"))
        return capture_resumability_fingerprint(
            repository,
            output_root=store_root,
            starting_commit=commit,
            pipeline_prefix_id=pipeline.prefix_id(0),
            dependency_versions=dependencies,
            execution_source_paths=sources,
            lockfile_paths=study_module._LOCKFILES,  # ruff: ignore[private-member-access]
            sealed_input_paths=study_module._SEALED_INPUTS,  # ruff: ignore[private-member-access]
        )

    monkeypatch.setattr(study_module, "_capture_fingerprint", development_fingerprint)
    study_root = tmp_path / "study"
    config = RefreshStudyConfig(
        finetune_updates=1,
        training_trajectories=1,
        refresh_interval=1,
        evaluation_trajectories=2,
        evaluation_batch_size=1,
    )
    result = run_refresh_study(study_root, pretrained_root=bundle_root, config=config)
    evaluation = cast("Mapping[str, object]", result["paired_fresh_evaluation"])
    schedule = cast("Mapping[str, object]", result["training_schedule"])

    assert result["status"] == "success"
    assert result["terminal_status"] == "completed"
    assert result["evidence_warning"] == "not confirmatory evidence"
    assert schedule["ensemble_count"] == 1
    assert schedule["unique_trajectory_coordinate_count"] == 1
    assert evaluation["same_maps_used_for_all_three_circuits"] is True
    assert evaluation["training_and_evaluation_seed_domains_disjoint"] is True
    rows = cast("Sequence[Mapping[str, object]]", evaluation["paired_trajectory_fidelities"])
    assert len(rows) == 2
    assert {"pretrained", "historical_fixed3", "refresh5", "events"} <= rows[0].keys()
    assert result["content_checksum"] == canonical_checksum({
        key: value for key, value in result.items() if key != "content_checksum"
    })

    retained = (study_root / STUDY_RESULT_NAME).read_bytes()
    assert run_refresh_study(study_root, pretrained_root=None, config=config, resume=True) == result
    assert (study_root / STUDY_RESULT_NAME).read_bytes() == retained
