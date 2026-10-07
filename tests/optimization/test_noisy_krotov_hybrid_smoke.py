# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the calibrated rerun of the small hybrid diagnostic."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from experiments.noisy_krotov_causal_study import create_circuit
from experiments.noisy_krotov_hybrid_smoke import run_hybrid_smoke

if TYPE_CHECKING:
    from pathlib import Path


def _source(root: Path) -> Path:
    """Write a deterministic source-checkpoint fixture with matching digest.

    Returns:
        The source directory.
    """
    root.mkdir()
    (root / "pretraining").mkdir()
    theta = np.random.default_rng(17).normal(0, 0.1, create_circuit(6, 4).num_params)
    np.savez_compressed(root / "pretraining" / "pretrained_checkpoint.npz", theta=theta)
    checksum = f"sha256:{hashlib.sha256(np.ascontiguousarray(theta, dtype='<f8').tobytes()).hexdigest()}"
    (root / "config.json").write_text(json.dumps({"qubits": 6, "target_id": "tfim_critical", "depths": [1, 2, 3, 4]}))
    (root / "causal_study_result.json").write_text(json.dumps({"pretraining": {"theta_checksum": checksum}}))
    return root


def test_hybrid_rerun_is_paired_deterministic_and_preserves_existing_outputs(tmp_path: Path) -> None:
    """Both phases run with calibrated noise and a fresh paired test ensemble."""
    source = _source(tmp_path / "source")
    settings = {"max_iterations": 2, "switch_iteration": 1, "training_trajectories": 1, "test_trajectories": 2}
    first = run_hybrid_smoke(tmp_path / "first", source, **settings)
    second = run_hybrid_smoke(tmp_path / "second", source, **settings)
    assert first["status"] == "success"
    assert first["configuration_checksum"] == second["configuration_checksum"]
    assert first["evaluation"] == second["evaluation"]
    training = cast("dict[str, object]", first["training"])
    assert training["phase"] == ["init", "online", "batch"]
    evaluation = cast("dict[str, object]", first["evaluation"])
    rows = cast("list[dict[str, float]]", evaluation["rows"])
    assert evaluation["same_maps_used_for_both_circuits"] is True
    assert len(rows) == 2
    assert all(row["hybrid_minus_frozen"] == pytest.approx(row["hybrid"] - row["frozen"]) for row in rows)
    retained = (tmp_path / "first" / "hybrid_smoke_result.json").read_bytes()
    with pytest.raises(ValueError, match="never overwritten"):
        run_hybrid_smoke(tmp_path / "first", source, **settings)
    assert (tmp_path / "first" / "hybrid_smoke_result.json").read_bytes() == retained


def test_hybrid_rejects_invalid_budget_and_tampered_checkpoint(tmp_path: Path) -> None:
    """Invalid input cannot create a misleading run or reuse altered parameters."""
    source = _source(tmp_path / "source")
    with pytest.raises(ValueError, match="budget"):
        run_hybrid_smoke(tmp_path / "bad", source, test_trajectories=1)
    with pytest.raises(ValueError, match="budget"):
        run_hybrid_smoke(tmp_path / "bad", source, training_trajectories=True)
    (source / "causal_study_result.json").write_text(json.dumps({"pretraining": {"theta_checksum": "incorrect"}}))
    with pytest.raises(ValueError, match="checksum"):
        run_hybrid_smoke(tmp_path / "bad", source)
    assert not (tmp_path / "bad").exists()
