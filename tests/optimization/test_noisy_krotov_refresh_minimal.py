# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the direct noisy-Krotov refresh experiment."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from experiments.noisy_krotov_refresh_minimal import (
    HISTORICAL_CHECKPOINT_NAME,
    PRETRAINED_CHECKPOINT_NAME,
    RESULT_NAME,
    TARGET_VECTOR_CHECKSUM,
    HistoricalNoiseProvider,
    StudyConfig,
    disordered_tfim_ground_state,
    run_study,
)
from mqt.yaqs.optimization import GateNoiseContext

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


def _context(sites: tuple[int, ...]) -> GateNoiseContext:
    """Create one minimal logical gate context.

    Returns:
        A context with the requested one- or two-qubit support.
    """
    return GateNoiseContext(
        gate_index=0,
        gate_name="test",
        sites=sites,
        arity=len(sites),
        resolved_angle=0.0,
        logical_gate_id=0,
        native_gate_id=0,
        parameter_index=0,
    )


def test_configuration_counts_refresh_windows() -> None:
    """Eleven updates at cadence five expose nine trajectory coordinates."""
    config = StudyConfig(
        updates=11,
        trajectories=3,
        refresh_interval=5,
        evaluation_trajectories=2,
        evaluation_batch_size=1,
    )

    assert config.ensemble_count == 3
    assert config.ensemble_count * config.trajectories == 9


def test_target_reconstruction_matches_retained_seed_100_reference() -> None:
    """The small generator reproduces the retained target exactly."""
    target = disordered_tfim_ground_state()
    checksum = f"sha256:{hashlib.sha256(np.ascontiguousarray(target, dtype='<c16').tobytes()).hexdigest()}"

    assert checksum == TARGET_VECTOR_CHECKSUM
    assert np.linalg.norm(target) == pytest.approx(1.0)


def test_historical_provider_has_three_local_and_two_crosstalk_processes() -> None:
    """The direct provider preserves the historical per-gate profile."""
    provider = HistoricalNoiseProvider()
    one_qubit = provider(_context((3,)), np.random.default_rng(0)).noise_model
    two_qubit = provider(_context((3, 4)), np.random.default_rng(0)).noise_model

    assert [process["name"] for process in one_qubit.processes] == ["pauli_x", "pauli_y", "pauli_z"]
    assert [process["name"] for process in two_qubit.processes] == [
        "pauli_x",
        "pauli_y",
        "pauli_z",
        "pauli_x",
        "pauli_y",
        "pauli_z",
        "crosstalk_xx",
        "crosstalk_zz",
    ]


def test_bounded_run_checkpoints_evaluation_and_resumes_exactly(tmp_path: Path) -> None:
    """A one-update run produces ordinary resumable artifacts."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    theta = np.linspace(-0.1, 0.1, 276, dtype=np.float64)
    for name in (PRETRAINED_CHECKPOINT_NAME, HISTORICAL_CHECKPOINT_NAME):
        np.savez_compressed(bundle / name, selected_theta=theta)
    output = tmp_path / "study"
    config = StudyConfig(
        updates=1,
        trajectories=1,
        refresh_interval=1,
        evaluation_trajectories=2,
        evaluation_batch_size=1,
    )

    result = run_study(output, bundle_root=bundle, config=config)
    training = cast("Mapping[str, object]", result["training"])
    evaluation = cast("Mapping[str, object]", result["evaluation"])

    assert result["status"] == "success"
    assert training["completed_updates"] == 1
    assert training["ensemble_count"] == 1
    assert evaluation["same_maps_used_for_all_circuits"] is True
    assert evaluation["training_and_evaluation_seeds_disjoint"] is True
    assert len(cast("Sequence[object]", evaluation["rows"])) == 2
    assert len(list((output / "evaluation_batches").glob("batch_*.json"))) == 2
    assert json.loads((output / RESULT_NAME).read_text(encoding="utf-8")) == result

    retained = (output / RESULT_NAME).read_bytes()
    assert run_study(output, bundle_root=None, config=config, resume=True) == result
    assert (output / RESULT_NAME).read_bytes() == retained
