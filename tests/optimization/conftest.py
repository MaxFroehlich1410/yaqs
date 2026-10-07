# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Self-contained fixtures for experiment source publication."""

from __future__ import annotations

import hashlib

import pytest

from experiments.overnight_state_preparation_20260913_0108 import runner


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
