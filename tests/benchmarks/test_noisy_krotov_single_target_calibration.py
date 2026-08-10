# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Focused bounded test for the exploratory q8 calibration runner."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from experiments.noisy_krotov_single_target_calibration import CalibrationConfig, run_calibration

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


def test_bounded_calibration_is_fresh_paired_and_resume_deterministic(tmp_path: Path) -> None:
    """A one-update run retains paired held-out data and resumes byte-identically."""
    output_root = tmp_path / "calibration"
    config = CalibrationConfig(
        growth_updates=1,
        finetune_updates=1,
        evaluation_trajectories=2,
    )

    result = run_calibration(output_root, config=config)
    evaluation = cast("Mapping[str, object]", result["paired_fresh_evaluation"])

    assert result["status"] == "success"
    assert result["evidence_classification"] == "exploratory"
    assert result["evidence_warning"] == "not confirmatory evidence"
    assert evaluation["ensemble_role"] == "pilot_evaluation"
    assert evaluation["training_and_evaluation_seed_domains_disjoint"] is True
    assert evaluation["same_ensemble_used_for_pre_and_post"] is True
    assert len(cast("Sequence[object]", evaluation["paired_trajectory_fidelities"])) == 2
    assert len(cast("Sequence[object]", result["stage_traces"])) == 5
    assert result["content_checksum"] == canonical_checksum({
        key: value for key, value in result.items() if key != "content_checksum"
    })

    retained = (output_root / "calibration_result.json").read_bytes()
    assert run_calibration(output_root, config=config, resume=True) == result
    assert (output_root / "calibration_result.json").read_bytes() == retained
