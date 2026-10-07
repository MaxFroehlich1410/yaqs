# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Self-contained fixtures for experiment source publication."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import ensemble_experiment, runner
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def isolated_overnight_custody(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind synthetic test protocols to current benchmarks without historical receipts."""
    benchmark_hashes = {
        name: hashlib.sha256((runner.REPO / name).read_bytes()).hexdigest()
        for name in (
            "benchmarks/state_preparation_benchmarks.md",
            "benchmarks/state_preparation_target_states.json",
        )
    }
    monkeypatch.setattr(runner, "custody", lambda: {"benchmark_hashes": benchmark_hashes})


@pytest.fixture
def synthetic_overnight_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_overnight_custody: None
) -> None:
    """Supply synthetic circuit metadata for preparation-only protocol checks."""
    del isolated_overnight_custody
    root = tmp_path / "synthetic-inputs"
    circuit = ParameterizedCircuit(6, [ParameterizedGate("rx", (index % 6,), index) for index in range(108)])
    starts = [
        {
            "target_id": target_id,
            "replica": 11,
            "candidate": {"circuit": pilot.circuit_record(circuit), "theta": [0.0] * circuit.num_params},
        }
        for target_id in ensemble_experiment.TARGETS
    ]
    pilot.write_record(root / "starts.json", {"starts": starts})
    monkeypatch.setattr(runner, "OUTPUT", root)
