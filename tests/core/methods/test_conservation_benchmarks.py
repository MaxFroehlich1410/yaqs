# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Verify that manuscript benchmark outputs retain per-arm recovery counters."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture


@pytest.mark.parametrize("script", ["l16_joint_table", "l16_trajectory", "l16_observables"])
def test_benchmark_reports_recovery(
    script: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    """Run an actual four-site short arm through each output writer."""
    folder = Path(__file__).resolve().parents[3] / "paper" / "bug-mps-benchmarks" / "spin_conservation"
    monkeypatch.syspath_prepend(str(folder))
    jt = importlib.import_module("l16_joint_table")
    module = importlib.import_module(script)
    monkeypatch.setattr(jt, "LENGTH", 4)
    monkeypatch.setattr(module, "LENGTH", 4)
    output = tmp_path / "arm.json"
    args = [script, "--total-time", "0.01", "--output", str(output)]
    if script == "l16_joint_table":
        args += ["--models", "xxx", "--caps", "1", "--variants", "pinH"]
    elif script == "l16_trajectory":
        args += ["--cap", "1", "--arms", "bug:pinH:none", "--record-every", "1"]
    else:
        args += ["--cap", "1", "--variants", "pinH"]
    monkeypatch.setattr("sys.argv", args)
    assert module.main() == 0
    data = json.loads(output.read_text())
    counts = data["arms"][0]["conservation_stats"]
    assert counts["scalar_calls"] == 2
    assert counts["fallback_attempts"] == counts["fallback_successes"] + counts["fallback_failures"]
    assert "fallback:" in capsys.readouterr().out
