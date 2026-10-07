# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Verify plotting custody, complete aggregation and displayed coordinate semantics."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import development_pareto_plot as plot

if TYPE_CHECKING:
    from collections.abc import Callable


def fixture(tmp_path: Path) -> Path:
    """Build a fully checksummed four-target descriptive input without simulation.

    Returns:
        The new sealed input path.
    """
    dependency = tmp_path / "source.txt"
    dependency.write_text("authenticated original bytes")
    points = [
        {
            "target_id": target,
            "replica": 11,
            "method": method,
            "fidelity": 0.80 + 0.01 * index + 0.001 * number,
            "noiseless_fidelity": 0.98,
            "one_qubit": 10 + index,
            "two_qubit": 4 + index,
            "parameters": 14 + 2 * index,
            "depth": 8 + index,
            "cpu_seconds": 20.0 - index,
            "ballarin_counts": {"one_qubit": 12 + index, "two_qubit": 4 + index, "depth": 10 + index},
        }
        for number, target in enumerate(plot.TARGETS)
        for index, method in enumerate(plot.METHODS)
    ]
    panel = {"points": points, "analysis": plot.tradeoffs.summarize(points, plot.TARGETS, (11,), plot.METHODS)}
    record: dict[str, Any] = {
        "version": plot.INPUT_VERSION,
        "analysis_source_sha256": plot.file_hash(Path(plot.tradeoffs.__file__)),
        "source_files": {str(dependency): plot.file_hash(dependency)},
        "original_evidence_files": {str(dependency): plot.file_hash(dependency)},
        "panels": {f"{recipe}/{stratum}": copy.deepcopy(panel) for recipe in plot.RECIPES for stratum in plot.STRATA},
        "comparison_scope": "separate development comparisons",
        "cpu_scope": "declared deployable CPU",
        "physical_gate_scope": "pre-Ballarin physical counts",
    }
    path = tmp_path / "input.json"
    pilot.write_record(path, record)
    return path


def test_complete_coordinates_and_rendered_artists(tmp_path: Path) -> None:
    """The figure includes all methods and exposes exact percentage/CPU coordinates."""
    arrays = plot.plotted_arrays(plot.authenticate(fixture(tmp_path)))
    assert len(arrays) == 4
    assert all(a["methods"] == list(plot.METHODS) and not a["projected_frontier_line"] for a in arrays)
    assert arrays[0]["x"] == [20.0 - index for index in range(7)]
    assert arrays[1]["x"] == [4 + index for index in range(7)]
    np.testing.assert_allclose(arrays[0]["y_percent"], [80.15 + index for index in range(7)])
    drawn = plot.figure(arrays)
    assert len(drawn.axes) == 4
    assert [axis.get_title(loc="left") for axis in drawn.axes] == [
        "Original recipe · selected for confirmation",
        "Original recipe · selected for confirmation",
        "With state preparation compiler · not promoted",
        "With state preparation compiler · not promoted",
    ]
    for axis, expected in zip(drawn.axes, arrays, strict=True):
        assert not axis.lines
        scatter = [item for item in axis.collections if item.get_zorder() == 3]
        assert len(scatter) == 7
        np.testing.assert_allclose(
            [np.asarray(item.get_offsets())[0] for item in scatter],
            list(zip(expected["x"], expected["y_percent"], strict=True)),
        )
        assert len(axis.collections) == 7 + sum(expected["four_objective_nondominated"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["panels"].pop("original/full_cap"),
        lambda r: r["panels"]["original/primary"]["points"].pop(),
        lambda r: r["panels"]["original/primary"]["analysis"]["aggregate"]["points"][0].update(fidelity=0.99),
        lambda r: r["panels"]["original/primary"]["analysis"]["aggregate"]["physical_resource_frontier"].update(
            nondominated=[]
        ),
        lambda r: r.update(analysis_source_sha256="0" * 64),
        lambda r: r.update(source_files={}),
        lambda r: r["panels"]["original/primary"]["points"][0].update(cpu_seconds=None),
        lambda r: r["panels"]["original/primary"]["points"][0].update(two_qubit=1.5),
        lambda r: r["panels"]["original/primary"]["points"][0].update(fidelity=True),
    ],
    ids=("missing_panel", "missing_case", "aggregate", "frontier", "source_code", "inventory", "cpu", "count", "bool"),
)
def test_resigned_inconsistent_input_rejected(tmp_path: Path, mutate: Callable[[dict[str, Any]], object]) -> None:
    """Re-signing edited metadata cannot replace complete point or source evidence."""
    source = fixture(tmp_path)
    record = dict(pilot.read_record(source))
    mutate(record)
    pilot.write_record(source, record)
    with pytest.raises(ValueError, match=r"[Dd]evelopment|[Pp]lot endpoints"):
        plot.authenticate(source)


def test_source_byte_tamper_rejected(tmp_path: Path) -> None:
    """Referenced source bytes are independently checked even with a valid input seal."""
    source = fixture(tmp_path)
    (tmp_path / "source.txt").write_text("changed")
    with pytest.raises(ValueError, match="source bytes"):
        plot.authenticate(source)


def test_resigned_coherent_replacement_rejected_by_pinned_checksum(tmp_path: Path) -> None:
    """The export's externally pinned input cannot be replaced by new coherent arrays."""
    source = fixture(tmp_path)
    record = cast("dict[str, Any]", pilot.read_record(source))
    expected = pilot.digest(record)
    for panel in record["panels"].values():
        panel["points"][0]["fidelity"] = 0.99
        panel["analysis"] = plot.tradeoffs.summarize(panel["points"], plot.TARGETS, (11,), plot.METHODS)
    pilot.write_record(source, record)
    with pytest.raises(ValueError, match="Development plot input"):
        plot.authenticate(source, expected)


def test_export_evidence_and_no_overwrite(tmp_path: Path) -> None:
    """Saved arrays, source hashes and both standard image formats are reproducible evidence."""
    source = fixture(tmp_path)
    before = source.read_bytes()
    prefix = tmp_path / "plots" / "development"
    expected = pilot.digest(pilot.read_record(source))
    result = plot.export(source, prefix, expected_checksum=expected)
    assert prefix.with_suffix(".png").read_bytes().startswith(b"\x89PNG")
    assert "<svg" in prefix.with_suffix(".svg").read_text()
    saved = cast("dict[str, Any]", pilot.read_record(prefix.with_suffix(".json")))
    assert pilot.digest(saved) == pilot.digest(result)
    assert saved["source_sha256"] == plot.file_hash(source)
    assert saved["development_selection_labels"] == {
        "original": "Original recipe · selected for confirmation",
        "state_preparation_compiled": "With state preparation compiler · not promoted",
    }
    assert saved["plotted_arrays"] == plot.plotted_arrays(plot.authenticate(source))
    assert all(plot.file_hash(Path(path)) == checksum for path, checksum in saved["files"].items())
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError, match="already exist"):
        plot.export(source, prefix, expected_checksum=expected)
