# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check complete-panel confirmation analysis without target construction or simulation."""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import confirmation_tradeoffs as report

if TYPE_CHECKING:
    from collections.abc import Callable


def fixture() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return explicit complete reporting stand-ins with unequal target/profile values.

    Returns:
        A metadata-only parent and ten synthetic case records.
    """
    protocol: dict[str, Any] = {
        "targets": list(report.parent.TARGETS),
        "replicas": [11, 29],
        "phase": "confirmation",
        "preparation_cpu_seconds": 1.0,
        "configuration_sha256": "f" * 64,
        "source_hashes": {},
        "benchmark_hashes": {},
    }
    protocol["cases"] = [{"target_id": t, "replica": r} for t in protocol["targets"] for r in protocol["replicas"]]
    records = []
    for t, target in enumerate(protocol["targets"]):
        for r, replica in enumerate(protocol["replicas"]):
            record: dict[str, Any] = {
                "target_id": target,
                "replica": replica,
                "status": "complete",
                "initializer_cpu_seconds": 2.0,
                "physical_invocation": {"cpu_seconds": 50.0, "wall_seconds": 51.0},
                "panels": {},
            }
            for p, panel in enumerate(report.PANELS):
                arms = {}
                for m, method in enumerate(report.parent.METHODS):
                    cpu = float(3 + m + (10 if panel == "full_cap" else 0))
                    fidelities = {
                        noise: 0.99
                        if noise == "noiseless"
                        else 0.70 + 0.01 * m + 0.02 * t + 0.004 * r + 0.001 * p + 0.0002 * n
                        for n, noise in enumerate(report.parent.base.channels.NOISE_IDS)
                    }
                    arms[method] = {
                        "status": "complete",
                        "point": {"theta": [0.0] * 6, "circuit_hash": "a" * 64, "cpu_seconds": cpu},
                        "scores": {
                            "fidelities": fidelities,
                            "physical_counts": {"one_qubit": 10 + m, "two_qubit": 4 + m, "depth": 6 + m},
                            "ballarin_counts": {"one_qubit": 30 + m, "two_qubit": 14 + m, "depth": 16 + m},
                            "parameters": 6,
                            "evaluation_checksum": "e" * 64,
                        },
                        "total_cpu_seconds": 2.0 + cpu,
                    }
                record["panels"][panel] = {"status": "complete", "arms": arms, "total_cpu_seconds": 12.0}
            records.append(record)
    return protocol, records


def test_complete_paired_means_resource_clocks_and_families() -> None:
    """All210 endpoints retain true point cost and profile/target weighting, including Haar."""
    protocol, records = fixture()
    result = report.analyze(protocol, records)
    assert len(result["points"]) == 210
    first = result["points"][0]
    assert first["fidelity"] == pytest.approx(0.7012)
    assert first["noiseless_fidelity"] == pytest.approx(0.99)
    assert first["cpu_seconds"] == pytest.approx(5.0)
    assert first["cutoff_cpu_seconds"] == pytest.approx(12.0)
    assert first["two_qubit"] == 4
    assert first["ballarin_counts"]["two_qubit"] == 14
    primary = result["panels"]["primary"]
    points = primary["descriptive"]["aggregate"]["points"]
    assert points[0]["fidelity"] == pytest.approx(0.7432)
    assert primary["by_family"]["TFIM"][report.parent.CANDIDATE]["fidelity"] == pytest.approx(0.7132)
    assert set(primary["by_family"]) == {"TFIM", "Haar", "MPS"}
    assert len(primary["target_frontiers"]) == 5
    assert len(primary["descriptive"]["cases"]) == 10
    assert len(primary["comparisons"]) == 6
    paired = primary["comparisons"][report.parent.BASELINE]
    assert len(paired["by_case"]) == 10
    assert len(paired["by_target"]) == 5
    assert paired["aggregate"]["delta"]["fidelity"] == pytest.approx(-0.02)
    assert paired["aggregate"]["cpu_ratio"] == pytest.approx(5 / 7)
    assert result["actual_experiment_cpu_seconds"] == pytest.approx(501.0)


def test_heatmaps_use_two_global_baselines_and_every_noise() -> None:
    """The strongest aggregate control is global; every cell uses paired replicas."""
    protocol, records = fixture()
    data = report.analyze(protocol, records)
    arrays = report.heatmap_arrays(data)
    assert [a["baseline"] for a in arrays] == [report.parent.BASELINE, "unchanged"]
    assert all(np.shape(a["delta_pp"]) == (5, 11) for a in arrays)
    np.testing.assert_allclose(arrays[0]["delta_pp"], -2.0)
    np.testing.assert_allclose(arrays[1]["delta_pp"], -6.0)
    assert "noiseless" not in arrays[0]["noise_ids"]
    figure = report.heatmap_figure(arrays)
    for axis, array in zip(figure.axes[:2], arrays, strict=True):
        np.testing.assert_array_equal(axis.images[0].get_array(), array["delta_pp"])
        assert [label.get_text() for label in axis.get_xticklabels()] == [
            "Ballarin",
            "Dph 1s/1q",
            "Dph 1s/2q",
            "Dph 1s/all",
            "Dph 2s/2q",
            "Dph 1+2/all",
            "Dep 1s/1q",
            "Dep 1s/2q",
            "Dep 1s/all",
            "Dep 2s/2q",
            "Dep 1+2/all",
        ]
    assert set(report.NOISE_LABELS) == set(report.NOISY)
    assert figure.axes[0].images[0].get_clim() == figure.axes[1].images[0].get_clim()
    scatter = report.scatter_figure(data)
    assert len(scatter.axes) == 2
    assert all(not axis.lines for axis in scatter.axes)
    assert "failed" in scatter.texts[0].get_text()
    assert any("Physical gate counts precede Ballarin lowering." in text.get_text() for text in scatter.texts)
    for axis, resource in zip(scatter.axes, ("cpu_seconds", "two_qubit"), strict=True):
        collections = [artist for artist in axis.collections if artist.get_zorder() == 3]
        expected = data["panels"]["primary"]["descriptive"]["aggregate"]["points"]
        np.testing.assert_allclose(
            [np.asarray(artist.get_offsets())[0] for artist in collections],
            [[point[resource], 100 * point["fidelity"]] for point in expected],
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows.pop(),
        lambda rows: rows.append(copy.deepcopy(rows[0])),
        lambda rows: rows[0].update(status="failed"),
        lambda rows: rows[0]["panels"]["primary"]["arms"].pop("unchanged"),
        lambda rows: rows[0].update(initializer_cpu_seconds=None),
        lambda rows: rows[0].update(initializer_cpu_seconds=True),
        lambda rows: rows[0]["panels"]["primary"]["arms"][report.parent.CANDIDATE]["point"].update(cpu_seconds=1000),
        lambda rows: rows[0]["panels"]["full_cap"]["arms"][report.parent.CANDIDATE].update(total_cpu_seconds=1000),
        lambda rows: rows[0]["panels"]["primary"]["arms"][report.parent.CANDIDATE]["scores"]["fidelities"].pop(
            "ballarin_coupled"
        ),
    ],
    ids=(
        "missing",
        "duplicate",
        "failed",
        "missing_role",
        "unknown_cpu",
        "boolean_cpu",
        "future_point",
        "wrong_total",
        "missing_noise",
    ),
)
def test_incomplete_or_invalid_panel_never_shrinks(mutate: Callable[[list[dict[str, Any]]], object]) -> None:
    """Missing targets, controls, profile scores and cost failures block the whole report."""
    protocol, records = fixture()
    mutate(records)
    with pytest.raises(ValueError, match=r"[Cc]onfirmation|[Ff]ull-cap|[Ss]elected|[Aa]ll|[Tt]radeoff"):
        report.analyze(protocol, records)


def test_export_contains_all2520_profile_cells_and_checksummed_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read-only publication retains all panels, profiles, point resources and source bytes."""
    protocol, records = fixture()
    source = tmp_path / "sealed-source.txt"
    source.write_text("immutable source")
    source_map = {str(source): report.file_hash(source)}
    monkeypatch.setattr(report, "authenticate", lambda _root: (protocol, records, source_map))
    monkeypatch.setattr(report.parent, "target_loader", lambda *_args: pytest.fail("No target loader is permitted"))
    prefix = tmp_path / "plots" / "confirmation"
    result = report.export(tmp_path, prefix)
    with prefix.with_suffix(".csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2520
    assert {r["panel"] for r in rows} == set(report.PANELS)
    assert next(r for r in rows if r["noise_id"] == "ballarin_coupled")["two_qubit"] == "14"
    assert next(r for r in rows if r["noise_id"] == "noiseless")["two_qubit"] == "4"
    saved = cast("dict[str, Any]", pilot.read_record(prefix.with_suffix(".json")))
    assert pilot.digest(saved) == pilot.digest(result)
    assert all(report.file_hash(Path(p)) == h for p, h in saved["files"].items())
    assert saved["source_files"] == source_map
    assert all(
        prefix.with_name(prefix.name + suffix).is_file()
        for suffix in ("_scatter.png", "_scatter.svg", "_heatmaps.png", "_heatmaps.svg")
    )
    with pytest.raises(FileExistsError, match="already exist"):
        report.export(tmp_path, prefix)


def test_authentication_reads_all_cases_and_rejects_resigned_summary_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The publication adapter delegates sealed case reads and binds every summary cost field."""
    protocol, records = fixture()
    calls = []
    monkeypatch.setattr(report.parent, "verify_parent", lambda _root, _protocol: calls.append("verify"))

    def case_reader(_root: Path, _protocol: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
        calls.append((case["target_id"], case["replica"]))
        return next(r for r in records if (r["target_id"], r["replica"]) == (case["target_id"], case["replica"]))

    monkeypatch.setattr(report.parent, "read_case", case_reader)
    monkeypatch.setattr(report.parent, "target_loader", lambda *_args: pytest.fail("Target access is forbidden"))
    summary: dict[str, Any] = {
        "version": report.parent.VERSION,
        "parent_checksum": pilot.digest(protocol),
        "configuration_sha256": protocol["configuration_sha256"],
        "status": "complete",
        "analysis": report.parent.analyze(records, protocol["targets"], protocol["replicas"]),
        "cases": [
            {
                "target_id": r["target_id"],
                "replica": r["replica"],
                "status": "complete",
                "checksum": pilot.digest(r),
                "failure": None,
                "physical_cpu_seconds": 50.0,
            }
            for r in records
        ],
        "preparation_cpu_seconds": 1.0,
        "actual_complete_experiment_cpu_seconds": 501.0,
    }
    pilot.write_record(tmp_path / "PROTOCOL.json", protocol)
    pilot.write_record(tmp_path / "DESIGN.json", {"synthetic": True})
    pilot.write_record(tmp_path / "SUMMARY.json", summary)
    for case in protocol["cases"]:
        child = tmp_path / f"{case['target_id']}__{case['replica']}"
        for name in (
            "CASE.json",
            "RESULT.json",
            "execution/PHASE.json",
            "execution/ATTEMPTS.json",
            "execution/RESULT.json",
            "execution/COMPLETED.json",
        ):
            pilot.write_record(child / name, {"synthetic": True})
    actual_parent, actual_records, sources = report.authenticate(tmp_path)
    assert actual_parent == protocol
    assert actual_records == records
    assert calls[0] == "verify"
    assert len(calls) == 11
    assert len(sources) == 63
    summary["actual_complete_experiment_cpu_seconds"] = 0.0
    pilot.write_record(tmp_path / "SUMMARY.json", summary)
    with pytest.raises(ValueError, match="SUMMARY differs"):
        report.authenticate(tmp_path)
