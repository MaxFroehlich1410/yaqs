# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shorter-budget cross-update parity and separation from the frozen long run."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from experiments import topdown_sequential_noisy_finetuning_100 as study
from tests.optimization import test_topdown_sequential_noisy_finetuning as fixtures

if TYPE_CHECKING:
    from pathlib import Path

input_folder = fixtures.input_folder


def test_full_short_budget_matches_one_archived_call(input_folder: Path, tmp_path: Path) -> None:
    """Actual 100-plus-100 updates retain exact parameters, traces and bank weights."""
    receipt, pruned, arrays, target = study.base.authenticate_input(input_folder)
    circuit = study.base.support.restore_circuit(pruned["circuit"])
    expected, training = study.base.legacy.train(
        circuit,
        arrays["theta"],
        target,
        steps=200,
        online=100,
        rate=0.2,
        seed=receipt["seed"] + 5000,
        noisy=True,
        crn=True,
        hardware="IBM",
        num_trajectories=100,
        compress_no_jump=True,
        record_noise_banks=True,
        require_jump_coverage=True,
    )
    result = study.run_case(tmp_path / "short", input_folder)
    online, online_arrays = study.base.parents.required(tmp_path / "short/online")
    batch, batch_arrays = study.base.parents.required(tmp_path / "short/batch")
    np.testing.assert_array_equal(batch_arrays["theta"], expected)
    np.testing.assert_array_equal(batch_arrays["initial"], online_arrays["theta"])
    np.testing.assert_array_equal(online["training"]["fidelity_trace"], training["fidelity_trace"][:100])
    np.testing.assert_array_equal(batch["training"]["fidelity_trace"], training["fidelity_trace"][100:])
    assert online["training"]["noise_banks"] == batch["training"]["noise_banks"] == training["noise_banks"]
    assert result["online_updates"] == result["batch_updates"] == 100
    assert result["protocol"] == study.PROTOCOL
    assert result["noisy_after_batch"] == study.base.legacy.matrix_score(circuit, expected, target, "IBM", "tjm")
    assert study.base.PROTOCOL["batch_updates"] == 200
    assert study.run_case(tmp_path / "short", input_folder) == result


def test_long_budget_cannot_resume_short_run(input_folder: Path, tmp_path: Path) -> None:
    """The original driver must reject a sealed short-budget manifest before training."""
    folder = tmp_path / "short"
    identity = {"protocol": study.PROTOCOL, "input_sha256": study.base.support.digest(input_folder / "input.json")}
    study.base.support.write_json(folder / "manifest.json", identity)
    with pytest.raises(ValueError, match="protocol or input"):
        study.base.run_case(folder, input_folder)


def test_frozen_protocol_is_restored_after_failure() -> None:
    """A failed worker never leaves the imported frozen driver with a changed budget."""
    original = study.base.PROTOCOL

    def failing_worker() -> None:
        """Fail while the short protocol is active.

        Raises:
            RuntimeError: Always, to exercise recovery of the imported default.
        """
        with study.fixed_budget():
            assert study.base.PROTOCOL is study.PROTOCOL
            msg = "synthetic failure"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        failing_worker()
    assert study.base.PROTOCOL is original


def test_prepare_records_short_budget_and_restores_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preparation delegates to the frozen lineage checks while sealing the short budget."""
    seen = []

    def prepare(parent: Path, output: Path) -> dict:
        seen.append((parent, output, study.base.PROTOCOL))
        return {"protocol": study.base.PROTOCOL}

    monkeypatch.setattr(study.base, "prepare", prepare)
    result = study.prepare(tmp_path / "parent", tmp_path / "output")
    assert result["protocol"] == study.PROTOCOL
    assert seen == [(tmp_path / "parent", tmp_path / "output", study.PROTOCOL)]
    assert study.base.PROTOCOL["batch_updates"] == 200
