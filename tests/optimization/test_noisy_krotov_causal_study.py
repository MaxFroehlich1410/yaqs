# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the canonical noisy state-preparation causal study."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from experiments.noisy_krotov_causal_study import (
    ARMS,
    RESULT_NAME,
    StudyConfig,
    derive_study_seed,
    run_study,
    training_schedule,
)
from mqt.yaqs.optimization import KrotovMapRole, derive_krotov_trajectory_seed

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


def test_sampling_schedules_have_the_intended_reuse_cadence() -> None:
    """The three noisy arms differ only in their ensemble reuse schedule."""
    config = StudyConfig(finetune_updates=6, refresh_interval=5)

    assert [training_schedule("crn_fixed", config).point(index).ensemble_index for index in range(6)] == [
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert [training_schedule("crn_refresh", config).point(index).ensemble_index for index in range(6)] == [
        0,
        0,
        0,
        0,
        0,
        1,
    ]
    assert [training_schedule("resampled", config).point(index).ensemble_index for index in range(6)] == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]


def test_role_separated_study_streams_produce_disjoint_trajectory_seeds() -> None:
    """Training, validation, and test roles cannot silently share a stream."""
    base_seed = 91
    roles: tuple[KrotovMapRole, ...] = (
        "training_trajectory",
        "checkpoint_validation",
        "pilot_evaluation",
    )
    roots = {role: derive_study_seed(base_seed, role, 0) for role in roles}
    seeds = {
        role: derive_krotov_trajectory_seed(
            role=role,
            resolved_seed=root,
            stage_index=0,
            ensemble_index=0,
            trajectory_index=0,
            refresh_index=0,
        )
        for role, root in roots.items()
    }

    assert len(set(roots.values())) == 3
    assert len(set(seeds.values())) == 3


def test_configuration_rejects_noncanonical_target_and_invalid_batch() -> None:
    """The runner cannot accidentally drift away from its registered q6 study."""
    with pytest.raises(ValueError, match="canonical q6"):
        StudyConfig(qubits=8)
    with pytest.raises(ValueError, match="test_batch_size"):
        StudyConfig(test_trajectories=1, test_batch_size=2)
    with pytest.raises(ValueError, match="only depolarizing_1s_all"):
        StudyConfig(noise_id="dephasing_1s_all", noise_calibration="hardware_average_error")


def test_calibration_is_explicit_and_legacy_configuration_is_preserved() -> None:
    """Legacy manifest format survives while calibrated runs cannot reuse it."""
    legacy = StudyConfig()
    calibrated = replace(legacy, noise_calibration="hardware_average_error")
    assert "noise_calibration" not in legacy.scientific_dict()
    assert (
        legacy.noise_provider().content_checksum
        == "sha256:617b7e867df529de5cd559bddb9d22f3cdd1b12f4b5eae3f19e415f4e3b45254"
    )
    assert isinstance(calibrated.noise_provider(), CalibratedLocalDepolarizingProvider)
    assert "noise_calibration_details" in calibrated.scientific_dict()


def test_bounded_study_uses_shared_start_paired_maps_and_exact_resume(tmp_path: Path) -> None:
    """A minimal run retains the causal design and resumes without mutation."""
    output = tmp_path / "study"
    config = StudyConfig(
        pretraining_iterations_per_depth=1,
        finetune_updates=1,
        refresh_interval=1,
        validation_interval=1,
        training_trajectories=1,
        validation_trajectories=1,
        test_trajectories=2,
        test_batch_size=1,
        replicates=1,
    )

    result = run_study(output, config=config)
    training = cast("Sequence[Mapping[str, object]]", result["training"])
    arm_rows = cast("Sequence[Mapping[str, object]]", training[0]["arms"])
    replicate = cast("Mapping[str, object]", cast("Sequence[object]", result["replicates"])[0])
    rows = cast("Sequence[Mapping[str, object]]", replicate["rows"])

    assert result["status"] == "success"
    assert [row["arm"] for row in arm_rows] == list(ARMS)
    assert len({row["initial_theta_checksum"] for row in arm_rows}) == 1
    assert [row["completed_updates"] for row in arm_rows] == [0, 1, 1, 1, 1]
    assert replicate["same_maps_used_for_all_arms"] is True
    assert len(rows) == 2
    assert all(set(cast("Mapping[str, object]", row["fidelities"])) == set(ARMS) for row in rows)
    assert cast("Mapping[str, object]", result["seed_audit"])["pairwise_disjoint"] is True

    first_training_ensembles = [
        next((output / "replicate_000" / "arms" / arm / "training_ensembles").glob("ensemble_*.json"))
        for arm in ("crn_fixed", "crn_refresh", "resampled")
    ]
    ensemble_checksums = {
        json.loads(path.read_text(encoding="utf-8"))["content_checksum"] for path in first_training_ensembles
    }
    assert len(ensemble_checksums) == 1

    retained = (output / RESULT_NAME).read_bytes()
    assert run_study(output, config=config, resume=True) == result
    assert (output / RESULT_NAME).read_bytes() == retained


def test_calibrated_parallel_study_matches_serial_and_rejects_legacy_resume(tmp_path: Path) -> None:
    """Process parallelism preserves streams and all numerical replicate results."""
    config = StudyConfig(
        pretraining_iterations_per_depth=1,
        finetune_updates=1,
        refresh_interval=1,
        validation_interval=1,
        training_trajectories=1,
        validation_trajectories=1,
        test_trajectories=2,
        test_batch_size=1,
        replicates=2,
        noise_calibration="hardware_average_error",
    )
    serial = run_study(tmp_path / "serial", config=config)
    parallel = run_study(tmp_path / "parallel", config=config, workers="auto")
    assert serial["configuration_checksum"] == parallel["configuration_checksum"]
    assert serial["aggregate"] == parallel["aggregate"]
    assert serial["seed_audit"] == parallel["seed_audit"]
    assert cast("Mapping[str, object]", parallel["runtime"])["numerical_threads_per_worker"] == 1
    assert run_study(tmp_path / "parallel", config=config, resume=True, workers=1) == parallel
    with pytest.raises(ValueError, match="differs"):
        run_study(tmp_path / "serial", config=replace(config, noise_calibration="legacy"), resume=True)
    with pytest.raises(ValueError, match="workers"):
        run_study(tmp_path / "invalid", config=config, workers=0)
