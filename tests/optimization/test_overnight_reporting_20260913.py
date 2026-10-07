# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Verify balanced aggregation, explicit failure handling and reproducible reports."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import (
    baseline_experiment,
    best_candidate,
    common_control_audit,
    composed_experiment,
    ensemble_structural_experiment,
    initializer_experiment,
    long_ensemble_experiment,
    mixture_baseline_experiment,
    reporting,
    risk_experiment,
)
from experiments.overnight_state_preparation_20260913_0108 import compiled_composition as compiled_kernel
from experiments.overnight_state_preparation_20260913_0108 import compiled_composition_experiment as compiled_driver
from experiments.overnight_state_preparation_20260913_0108 import (
    state_preparation_compilation_experiment as state_compiler,
)
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def rows() -> list[dict[str, str]]:
    """Return unequal replica counts to expose accidental weighting by runs."""
    return [
        {
            "target_id": target,
            "replica": replica,
            "method": method,
            "noise_id": noise,
            "status": "complete",
            "fidelity": str(1 if noise == "noiseless" else value + offset),
        }
        for method, offset in (("a", 0), ("b", 0.05))
        for target, replica, value in (("gaussian", "11", 0.2), ("gaussian", "29", 0.4), ("tfim", "11", 0.8))
        for noise in reporting.channels.NOISE_IDS
    ]


def test_balances_targets_before_comparing_methods() -> None:
    """Two starts on one target cannot double its contribution to the mean."""
    result = reporting.summarize(rows())
    method = result["methods"]["a"]
    assert method["mean_noisy_fidelity"] == pytest.approx(0.55)
    assert method["by_noise"]["noiseless"] == pytest.approx(1)
    assert method["worst_target"] == "gaussian"
    pair = next(row for row in result["paired_all_noisy"] if row["candidate"] == "b")
    assert pair["mean_delta_pp"] == pytest.approx(5)
    assert pair["target_wins"] == 2
    assert pair["case_wins"] == 3


@pytest.mark.parametrize("change", ["failed", "missing"])
def test_failures_or_missing_cells_invalidate_aggregate(change: str) -> None:
    """A failed difficult condition cannot disappear from a favorable average."""
    data = rows()
    if change == "failed":
        data[0]["status"] = "failed"
        data[0]["fidelity"] = ""
    else:
        data.pop(0)
    result = reporting.summarize(data)
    assert result["methods"]["a"]["status"] == "incomplete"
    assert result["methods"]["b"]["status"] == "complete"
    assert result["paired_all_noisy"] == []


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "1.1"])
def test_rejects_invalid_completed_fidelity(value: str) -> None:
    """Nonfinite completed results require an explicit failed-run record."""
    data = rows()
    data[0]["fidelity"] = value
    with pytest.raises(ValueError, match="unphysical"):
        reporting.summarize(data)


def test_duplicate_cell_rejected() -> None:
    """Repeated output paths or duplicate evaluations cannot bias aggregate weights."""
    data = rows()
    with pytest.raises(ValueError, match="Duplicate"):
        reporting.summarize([*data, data[0]])


def panel_fixture(root: Path, monkeypatch: pytest.MonkeyPatch, kind: str = "svrg", *, failed: bool = False) -> Path:
    """Return a complete checksummed toy panel with independent provenance inputs."""
    root.mkdir(parents=True)
    repository = root / "repository"
    repository.mkdir()
    benchmark = repository / "benchmark.txt"
    benchmark.write_text("fixed target and noise definition")
    source_file = repository / "numerical.py"
    source_file.write_text("# Frozen toy numerical dependency")
    hashes = {"benchmark.txt": hashlib.sha256(benchmark.read_bytes()).hexdigest()}
    sources = {"numerical.py": hashlib.sha256(source_file.read_bytes()).hexdigest()}
    manifest = root / "MANIFEST.json"
    manifest.write_text(json.dumps({"benchmark_hashes": hashes}))
    monkeypatch.setattr(reporting, "REPO", repository)
    monkeypatch.setattr(reporting, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(reporting, "_required_sources", lambda _kind: {"numerical.py"})
    targets = ["gaussian_mu0p5_sigma0p1", "tfim_critical"]
    methods = {"a": {"method": "adam", "iterations": 1}, "b": {"method": "adam", "iterations": 1}}
    starts: list[dict[str, Any]] = [
        {"target_id": target, "replica": 11, "candidate": {"theta": [0.0]}} for target in targets
    ]
    protocol: dict[str, Any] = {"benchmark_hashes": hashes, "source_hashes": sources}
    if kind == "runner":
        protocol["protocol"] = {
            "name": "wave1",
            "phase": "development",
            "targets": targets,
            "replicas": [11],
            "methods": methods,
        }
    elif kind == "synthesis":
        protocol.update(
            version="overnight.rank_two_development_panel.v1",
            spec={"targets": targets, "replicas": [11], "methods": methods, "reference_methods": ["exact_adam"]},
        )
        methods = {**methods, "reference_exact_adam": {}}
    else:
        tag = next(tag for tag, name in reporting.VERSIONS.items() if name == kind)
        protocol.update({
            "schema" if kind == "svrg" else "version": tag,
            "targets": targets,
            "replica": 11,
            "starts": starts,
        })
        protocol["arms" if kind in {"ensemble", "geometry"} else "methods"] = methods
        if kind == "structural":
            historical = root / "history"
            jobs = []
            for target in targets:
                archived: dict[str, Any] = {"refined": {"adam": {}, "adaptive_online": {}}}
                pilot.write_record(historical / f"{target}.json", archived)
                jobs.append({
                    "target_id": target,
                    "replica": 11,
                    "path": f"{target}.json",
                    "checksum": pilot.digest(archived),
                })
            history: dict[str, Any] = {"jobs": jobs}
            pilot.write_record(historical / "result.json", history)
            protocol.update(historical_root=str(historical), historical_result_checksum=pilot.digest(history))
            methods = {
                name: {}
                for name in (
                    "fixed_lbfgs",
                    "structural_lbfgs",
                    "unchanged_reference",
                    "historical_structural_adam",
                    "historical_structural_adaptive_online",
                )
            }
    if kind in {"ensemble_structural", "risk"}:
        methods = {
            key: {} for key in (risk_experiment.METHODS if kind == "risk" else ensemble_structural_experiment.METHODS)
        }
        protocol["methods"] = list(methods)
        protocol["config"] = {"frozen_budget": 10}
        for source in starts:
            source["candidate"]["circuit"] = {"num_qubits": 6, "gates": []}
        monkeypatch.setattr(reporting, "target_hash", lambda target: hashlib.sha256(target.encode()).hexdigest())
    pilot.write_record(root / "PROTOCOL.json", protocol)
    records: list[dict[str, Any]] = []
    counts = {"one_qubit": 3, "two_qubit": 1, "depth": 3}
    for index, source in enumerate(starts):
        target = source["target_id"]
        arms = {}
        for method_index, method in enumerate(methods):
            status = "failed" if failed and index == method_index == 0 else "complete"
            values = {
                noise: 1.0 if noise == "noiseless" else 0.2 + 0.1 * index + 0.05 * method_index
                for noise in reporting.channels.NOISE_IDS
            }
            state = {
                "theta": [0.0],
                "iterations": 1,
                "exact_training_evaluations": 1,
                "trajectory_paths": 0,
                "direction_passes": 2,
                "training_cpu_seconds": 1.0,
                "training_wall_seconds": 1.1,
                "diagnostic_cpu_seconds": 0.1,
                **dict.fromkeys(
                    (
                        "exact_gradient_calls",
                        "exact_anchor_calls",
                        "trajectory_gradient_calls",
                        "trajectory_direction_passes",
                        "density_gradient_direction_passes",
                        "diagnostic_forward_calls",
                        "diagnostic_direction_passes",
                    ),
                    0,
                ),
            }
            if kind in {"runner", "svrg"}:
                record: dict[str, Any] = {
                    "status": status,
                    "target_id": target,
                    "replica": 11,
                    "method": method,
                    "state": state,
                    "fidelities": values,
                    "logical_counts": counts,
                    "ballarin_counts": counts,
                    "evaluation_cpu_seconds": 0.2,
                    "peak_rss_bytes": 1024,
                    "physical_attempt_totals": {"training_cpu_seconds": 1.0},
                }
                if status == "failed":
                    record["failure"] = "injected failure"
                if kind == "runner":
                    job = {
                        "target_id": target,
                        "replica": 11,
                        "label": method,
                        "phase": "development",
                        "config": methods[method],
                        "runtime_checksum": pilot.digest(protocol),
                        "source_hash": pilot.digest(sources),
                        "seed": 42,
                    }
                    record["job"] = job
                    record["signature"] = pilot.digest(job)
                    path = root / f"{target}__11__{method}.json"
                else:
                    record["signature"] = pilot.digest({
                        "protocol": pilot.digest(protocol),
                        "source": pilot.digest(source),
                        "method": method,
                    })
                    path = root / f"{target}__11__{method}" / ("RESULT.json" if status == "complete" else "FAILED.json")
                pilot.write_record(path, record)
                records.append(record)
            elif kind in {"ensemble", "geometry"}:
                arms[method] = {
                    "status": status,
                    "failure": "injected failure" if status == "failed" else None,
                    "training": {
                        "cpu_seconds": 1.0,
                        "joint_gradient_calls": 1,
                        "individual_channel_gradient_calls": 1,
                        "density_direction_passes": 2,
                        "physical_completed_channel_calls": 1,
                        "optimizer": {"status": "evaluation_limit"},
                        "true_channel_gradient_calls": 1,
                        "metric_evaluations": 1,
                        "physical": {
                            "metric_cpu_seconds": 0.1,
                            "solve_cpu_seconds": 0.1,
                            "channel_attempted": 1,
                            "statevector_row_operator_applications": 5,
                        },
                    },
                    "final_score": {"fidelities": values, "mean_noisy_fidelity": 0.2},
                }
            else:
                arms[method] = {
                    "status": status,
                    "fidelities": values,
                    "theta": [0.0],
                    "state": state,
                    "logical_counts": counts,
                    "physical_counts": counts,
                    "ballarin_counts": counts,
                    "evaluation_cpu_seconds": 0.2,
                    "individual_channel_calls": 10,
                    "gradient_channel_calls": 6,
                    "ranking_channel_calls": 2,
                    "forward_channel_calls": 2,
                    "density_direction_passes": 18,
                    "physical_completed_channel_calls": 10,
                    "cpu_seconds": 1.0,
                    "final_evaluation": {"cpu_seconds": 0.2},
                    "specification": {"temperature": 0.2},
                    "wall_seconds": 1.1,
                    "physical_work": {"attempted_channel_calls": 10, "completed_channel_calls": 10},
                    "evaluation": {"cpu_seconds": 0.2},
                    "comparison": "unmatched_reference"
                    if "reference" in method or "historical" in method
                    else "protocol_budget_control",
                }
        if kind not in {"runner", "svrg"}:
            identity: dict[str, Any] = {"protocol": protocol, "source": source}
            if kind == "synthesis":
                identity = {"protocol": protocol, "target_id": target, "replica": 11}
            elif kind in {"ensemble", "geometry"}:
                identity["order_index"] = index
            record = {
                "target_id": target,
                "replica": 11,
                "status": "complete_with_failed_arms" if failed else "complete",
                "signature": pilot.digest(identity),
                "arms": arms,
            }
            if kind in {"ensemble_structural", "risk"}:
                record["signature"] = reporting.ensemble_structural_signature(protocol, source)
                record["logical_counts"] = counts
                record["parameters"] = 1
            pilot.write_record(root / f"{target}__11" / "RESULT.json", record)
            records.append(record)
    summary: dict[str, Any] = {
        "status": "contains_failures" if failed else "complete",
        "runtime_checksum" if kind == "runner" else "protocol_checksum": pilot.digest(protocol),
        "runs" if kind in {"runner", "svrg"} else "results": [pilot.digest(record) for record in records],
    }
    pilot.write_record(root / "RESULT.json", summary)
    path = root / "SCOREBOARD.csv"
    write_rows(path, reporting.canonical_rows(kind, records))
    return path


def write_rows(path: Path, data: list[dict[str, str]]) -> None:
    """Write one deliberately varied source CSV without touching its checksummed inputs."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Return raw cells for independent mutation of a scoreboard."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def baseline_fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return six paired toy baselines with distinct full and common-CPU values."""
    reference_path = panel_fixture(root / "reference", monkeypatch, "structural")
    reference = pilot.read_record(reference_path.parent / "PROTOCOL.json")
    output = root / "baselines"
    output.mkdir()
    protocol: dict[str, Any] = {
        "version": "overnight.strong_structural_baselines.v1",
        "benchmark_hashes": reference["benchmark_hashes"],
        "source_hashes": reference["source_hashes"],
        "targets": reference["targets"],
        "replicas": [11],
        "starts": reference["starts"],
        "reference_root": str(reference_path.parent),
        "reference_protocol_checksum": pilot.digest(reference),
    }
    pilot.write_record(output / "PROTOCOL.json", protocol)
    records = []
    for source in cast("list[dict[str, Any]]", reference["starts"]):
        target = source["target_id"]
        ref = pilot.read_record(reference_path.parent / f"{target}__11/RESULT.json")
        arms, common = {}, {}
        for method in (*baseline_experiment.LABELS, "structural_lbfgs", "fixed_lbfgs"):
            arms[method] = {
                "status": "complete",
                "fidelities": dict.fromkeys(reporting.channels.NOISE_IDS, 0.8),
                "comparison": "full_cap_endpoint_different_computation",
                "cpu_seconds": 4.0,
            }
            common[method] = {
                **arms[method],
                "fidelities": dict.fromkeys(reporting.channels.NOISE_IDS, 0.6),
                "comparison": "common_cpu_last_available_final_iterate",
                "cpu_seconds": 2.0,
            }
        ref_arms = cast("dict[str, dict[str, Any]]", ref["arms"])
        record: dict[str, Any] = {
            "target_id": target,
            "replica": 11,
            "status": "complete",
            "arms": arms,
            "common_cpu": {"arms": common, "cpu_seconds": 2.0, "status": "complete"},
            "historical_unmatched": {name: value for name, value in ref_arms.items() if name.startswith("historical_")},
            "signature": pilot.digest([protocol, source]),
            "reference_case_checksum": pilot.digest(ref),
        }
        pilot.write_record(output / f"{target}__11/RESULT.json", record)
        records.append(record)
    pilot.write_record(
        output / "RESULT.json",
        {
            "status": "complete",
            "protocol_checksum": pilot.digest(protocol),
            "cases": [pilot.digest(record) for record in records],
        },
    )
    baseline_experiment.write_scoreboard(output, records)
    return output / "SCOREBOARD.csv"


def test_baseline_strata_never_pool_full_and_common_cpu_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One physical method has separate authenticated budget comparison rows."""
    source = baseline_fixture(tmp_path, monkeypatch)
    result = reporting.collect([source], tmp_path / "overview")
    assert set(result["panels"]) == {f"baselines/{kind}" for kind in ("full_cap", "common_cpu", "historical_unmatched")}
    full = result["panels"]["baselines/full_cap"]["methods"]["structural_lbfgs [full_cap]"]
    common = result["panels"]["baselines/common_cpu"]["methods"]["structural_lbfgs [common_cpu]"]
    assert full["mean_noisy_fidelity"] == pytest.approx(0.8)
    assert common["mean_noisy_fidelity"] == pytest.approx(0.6)
    with pytest.raises(ValueError, match="distinct parent"):
        reporting.collect([source, source], tmp_path / "duplicated")


def test_missing_entire_cpu_stratum_is_visible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing matched comparison cannot masquerade as its full-budget version."""
    source = baseline_fixture(tmp_path, monkeypatch)
    write_rows(source, [row for row in read_rows(source) if row["panel"] != "common_cpu"])
    result = reporting.collect([source], tmp_path / "overview")
    summary = result["panels"]["baselines/common_cpu"]["methods"]
    assert len(summary) == 6
    assert all(method["status"] == "incomplete" for method in summary.values())


def test_changed_baseline_reference_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rewriting a referenced endpoint invalidates its downstream comparison."""
    source = baseline_fixture(tmp_path, monkeypatch)
    path = tmp_path / "reference/gaussian_mu0p5_sigma0p1__11/RESULT.json"
    reference = pilot.read_record(path)
    cast("dict[str, Any]", reference["arms"])["fixed_lbfgs"]["fidelities"]["noiseless"] = 0.9
    pilot.write_record(path, reference)
    with pytest.raises(ValueError, match="changed structural reference"):
        reporting.authenticate_panel(source)


def test_baseline_normalization_rejects_unknown_strata() -> None:
    """Unknown checkpoint policies cannot enter a matched comparison."""
    with pytest.raises(ValueError, match="Unknown baseline comparison stratum"):
        reporting.normalize_strata([{"panel": "best_checkpoint", "method": "adam"}], "baseline")


@pytest.mark.parametrize(
    "kind", ["runner", "structural", "synthesis", "svrg", "ensemble", "geometry", "ensemble_structural", "risk"]
)
def test_every_supported_schema_authenticates_complete_expected_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    """All five writer adapters derive expectations from protocol and checksummed records."""
    source = panel_fixture(tmp_path / "panel", monkeypatch, kind)
    result = reporting.collect([source], tmp_path / "overview")
    assert result["sources"]["panel"]["schema_kind"] == kind
    assert result["sources"]["panel"]["missing_csv_cells"] == 0
    assert all(method["status"] == "complete" for method in result["panels"]["panel"]["methods"].values())
    before = {path.name: path.read_bytes() for path in (tmp_path / "overview").iterdir()}
    reporting.collect([source], tmp_path / "overview")
    assert before == {path.name: path.read_bytes() for path in (tmp_path / "overview").iterdir()}


def test_collect_retains_failed_cells_and_unknown_resource_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure status must match the checksummed result; every source column is retained."""
    source = panel_fixture(tmp_path / "panel", monkeypatch, failed=True)
    data = read_rows(source)
    for row in data:
        row["arbitrary_resource_counter"] = "42"
    write_rows(source, data)
    result = reporting.collect([source], tmp_path / "overview")
    assert result["panels"]["panel"]["methods"]["a"]["status"] == "incomplete"
    assert result["panels"]["panel"]["methods"]["b"]["status"] == "complete"
    assert result["raw_rows"] == len(data)
    assert "arbitrary_resource_counter" in (tmp_path / "overview/SCOREBOARD.csv").read_text()
    assert "INCOMPLETE" in (tmp_path / "overview/SCOREBOARD.md").read_text()


@pytest.mark.parametrize("drop", ["whole_target", "whole_method", "one_cell", "all_cells"])
def test_missing_cases_and_methods_materialize_incomplete_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drop: str,
) -> None:
    """The observed CSV cannot redefine the protocol's expected comparison panel."""
    source = panel_fixture(tmp_path / "panel", monkeypatch)
    original = read_rows(source)
    data = original.copy()
    if drop == "whole_target":
        data = [row for row in data if row["target_id"] != original[0]["target_id"]]
    elif drop == "whole_method":
        data = [row for row in data if row["method"] != "a"]
    elif drop == "one_cell":
        data.pop(0)
    else:
        data = []
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(original[0]))
        writer.writeheader()
        writer.writerows(data)
    result = reporting.collect([source], tmp_path / "overview")
    assert result["rows_including_missing"] == len(original)
    assert result["raw_rows"] == len(data)
    assert result["panels"]["panel"]["methods"]["a"]["status"] == "incomplete"
    assert result["panels"]["panel"]["paired_all_noisy"] == []
    assert "missing_expected_csv_cell" in (tmp_path / "overview/SCOREBOARD.csv").read_text()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("convention", "legacy_one_jump"),
        ("exactness", "monte_carlo"),
        ("fidelity", "0.999"),
        ("status", "failed"),
        ("training_cpu_seconds", "0"),
    ],
)
def test_explicit_wrong_tags_and_altered_cells_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    """Scores, completion and known resource counts must match authenticated records."""
    source = panel_fixture(tmp_path / "panel", monkeypatch)
    data = read_rows(source)
    if field not in data[0]:
        for row in data:
            row[field] = ""
    data[0][field] = value
    write_rows(source, data)
    with pytest.raises(ValueError, match=r"convention|exactness|differs"):
        reporting.collect([source], tmp_path / "overview")


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_protocol",
        "unknown_schema",
        "wrong_protocol_checksum",
        "running_summary",
        "missing_run",
        "missing_summary",
        "source_hash",
        "source_coverage",
        "benchmark_hash",
        "run_checksum",
    ],
)
def test_missing_or_changed_provenance_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """No source CSV can manufacture provenance by carrying plausible field labels."""
    source = panel_fixture(tmp_path / "panel", monkeypatch)
    protocol_path, summary_path = source.parent / "PROTOCOL.json", source.parent / "RESULT.json"
    protocol = cast("dict[str, Any]", pilot.read_record(protocol_path))
    summary = cast("dict[str, Any]", pilot.read_record(summary_path))
    if mutation == "missing_protocol":
        protocol_path.rename(source.parent / "hidden_protocol.json")
    elif mutation == "unknown_schema":
        protocol["schema"] = "unrecognized.v1"
        pilot.write_record(protocol_path, protocol)
    elif mutation == "wrong_protocol_checksum":
        summary["protocol_checksum"] = "wrong"
        pilot.write_record(summary_path, summary)
    elif mutation == "running_summary":
        summary["status"] = "running"
        pilot.write_record(summary_path, summary)
    elif mutation == "missing_summary":
        summary_path.rename(source.parent / "hidden_summary.json")
    elif mutation == "missing_run":
        path = next(source.parent.glob("*__11__a/RESULT.json"))
        path.rename(path.with_name("hidden_run.json"))
    elif mutation == "source_hash":
        (source.parent / "repository/numerical.py").write_text("changed numerical source")
    elif mutation == "source_coverage":
        protocol["source_hashes"] = {}
        pilot.write_record(protocol_path, protocol)
    elif mutation == "benchmark_hash":
        protocol["benchmark_hashes"] = {"benchmark.txt": "wrong"}
        pilot.write_record(protocol_path, protocol)
    else:
        summary["runs"] = summary["runs"][:-1]
        pilot.write_record(summary_path, summary)
    with pytest.raises(ValueError, match=r"missing|Unknown|Panel RESULT|terminal|source|benchmark|checksums|hash"):
        reporting.collect([source], tmp_path / "overview")


def test_noise_families_follow_normative_numbering_and_keep_condition_weighting() -> None:
    """N1, N2 and N3 summaries must not replace the eleven-condition primary mean."""
    data = rows()
    for row in data:
        noise = row["noise_id"]
        row["fidelity"] = str(0.2 if noise == "ballarin_coupled" else 0.4 if noise.startswith("dephasing_") else 0.8)
    method = reporting.summarize(data)["methods"]["a"]
    assert method["by_noise_family"] == pytest.approx({
        "N1_angle_dependent": 0.2,
        "N2_dephasing": 0.4,
        "N3_depolarizing": 0.8,
    })
    assert method["mean_noisy_fidelity"] == pytest.approx((0.2 + 5 * 0.4 + 5 * 0.8) / 11)
    assert method["worst_noise_family"] == "N1_angle_dependent"
    assert method["worst_noise_family_fidelity"] == pytest.approx(0.2)
    assert "worst_target_family_noisy_fidelity" in method


def compilation_fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return independently authenticated paired endpoint sources for two targets."""
    base_path = panel_fixture(root / "base", monkeypatch)
    base = cast("dict[str, Any]", pilot.read_record(base_path.parent / "PROTOCOL.json"))
    output = root / "compiler"
    output.mkdir()
    inputs = root / "original.json"
    inputs.write_text("untouched endpoint")
    input_hashes = {str(inputs): hashlib.sha256(inputs.read_bytes()).hexdigest()}
    sources = [
        {"target_id": target, "replica": 11, "label": label, "file_hashes": input_hashes}
        for target in base["targets"]
        for label in ("first", "second")
    ]
    protocol: dict[str, Any] = {
        "version": "overnight.physical_compilation_panel.v1",
        "targets": base["targets"],
        "replicas": [11],
        "labels": ["first", "second"],
        "sources": sources,
        "benchmark_hashes": base["benchmark_hashes"],
        "source_hashes": base["source_hashes"],
    }
    pilot.write_record(output / "PROTOCOL.json", protocol)
    records = []
    for source in sources:
        record: dict[str, Any] = {
            "status": "complete",
            "target_id": source["target_id"],
            "replica": 11,
            "source_label": source["label"],
            "source_checksum": pilot.digest(source),
            "signature": pilot.digest([pilot.digest(protocol), source]),
            "source_costs": {"construction_cpu_seconds": None, "refinement_cpu_seconds": 2.0},
            "comparison": "paired_compilation_of_same_final_endpoint_no_refinement",
            "arms": {
                name: {
                    "status": "complete",
                    "fidelities": dict.fromkeys(reporting.channels.NOISE_IDS, value),
                    "physical_counts": {"one_qubit": 2, "two_qubit": 1, "depth": 3},
                }
                for name, value in (("uncompiled", 0.7), ("compiled", 0.8))
            },
        }
        pilot.write_record(output / f"{source['target_id']}__11__{source['label']}.json", record)
        records.append(record)
    pilot.write_record(
        output / "RESULT.json",
        {
            "status": "complete",
            "protocol_checksum": pilot.digest(protocol),
            "cases": [pilot.digest(record) for record in records],
        },
    )
    path = output / "SCOREBOARD.csv"
    write_rows(path, reporting.canonical_rows("compilation", records))
    return path


def test_compilation_keeps_distinct_endpoint_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compiler variants retain four method identities and all twelve conditions."""
    path = compilation_fixture(tmp_path, monkeypatch)
    data, audit = reporting.authenticate_panel(path)
    assert len(data) == audit["expected_cells"] == 96
    summary = reporting.summarize(data)
    assert set(summary["methods"]) == {
        f"{label}__{arm}" for label in ("first", "second") for arm in ("compiled", "uncompiled")
    }
    assert summary["methods"]["first__compiled"]["mean_noisy_fidelity"] == pytest.approx(0.8)


@pytest.mark.parametrize("change", ["source_file", "source_label", "checksum", "resource", "missing_pair"])
def test_compilation_provenance_and_pair_completeness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Changed source custody and resources fail; missing pairs remain incomplete."""
    path = compilation_fixture(tmp_path, monkeypatch)
    if change == "source_file":
        (tmp_path / "original.json").write_text("changed endpoint")
    elif change in {"source_label", "checksum"}:
        target = "gaussian_mu0p5_sigma0p1"
        record_path = path.parent / f"{target}__11__first.json"
        record = pilot.read_record(record_path)
        record["source_label" if change == "source_label" else "source_checksum"] = "changed"
        pilot.write_record(record_path, record)
    else:
        data = read_rows(path)
        if change == "resource":
            data[0]["physical_two_qubit"] = "999"
        else:
            data = [row for row in data if row["source"] != "second" or row["arm"] != "compiled"]
        write_rows(path, data)
    if change == "missing_pair":
        data, audit = reporting.authenticate_panel(path)
        assert audit["missing_csv_cells"] == 24
        assert reporting.summarize(data)["methods"]["second__compiled"]["status"] == "incomplete"
    else:
        with pytest.raises(ValueError, match=r"Compilation|CSV cell"):
            reporting.authenticate_panel(path)


def mixture_fixture(root: Path, monkeypatch: pytest.MonkeyPatch, *, failed: bool = False) -> Path:
    """Return independent primary/history inputs and the auxiliary mixture context."""
    primary_path = panel_fixture(root / "primary", monkeypatch, "ensemble_structural")
    primary = cast("dict[str, Any]", pilot.read_record(primary_path.parent / "PROTOCOL.json"))
    pilot.write_record(root / "starts.json", {"starts": primary["starts"]})
    monkeypatch.setattr(reporting.runner, "OUTPUT", root)
    history_root = root / "history"
    history_protocol: dict[str, Any] = {
        **primary,
        "version": "overnight.ensemble_objective_comparison.v1",
        "arms": {key: {} for key in reporting.MIXTURE_HISTORICAL_METHODS},
    }
    pilot.write_record(history_root / "PROTOCOL.json", history_protocol)
    history_records = {}
    history_files = {}
    for index, source in enumerate(primary["starts"]):
        target = source["target_id"]
        archived: dict[str, Any] = {
            "target_id": target,
            "replica": 11,
            "status": "complete",
            "signature": pilot.digest({"protocol": history_protocol, "source": source, "order_index": index}),
            "arms": {
                method: {
                    "status": "complete",
                    "final_score": {"fidelities": dict.fromkeys(reporting.channels.NOISE_IDS, 0.4)},
                }
                for method in reporting.MIXTURE_HISTORICAL_METHODS
            },
        }
        path = history_root / f"{target}__11" / "RESULT.json"
        pilot.write_record(path, archived)
        history_records[target] = archived
        history_files[str(path)] = reporting.file_sha256(path)
    pilot.write_record(
        history_root / "RESULT.json",
        {
            "status": "complete",
            "protocol_checksum": pilot.digest(history_protocol),
            "results": [pilot.digest(record) for record in history_records.values()],
        },
    )
    output = root / "mixture"
    protocol: dict[str, Any] = {
        **primary,
        "version": mixture_baseline_experiment.VERSION,
        "methods": list(mixture_baseline_experiment.METHODS),
        "reference_root": str(primary_path.parent),
        "reference_protocol_checksum": pilot.digest(primary),
        "source_starts_sha256": reporting.file_sha256(root / "starts.json"),
        "unmatched_reference_files": history_files,
        "cpu_scope": mixture_baseline_experiment.CPU_SCOPE,
    }
    pilot.write_record(output / "PROTOCOL.json", protocol)
    records = []
    for index, source in enumerate(protocol["starts"]):
        target = source["target_id"]
        primary_record = cast("dict[str, Any]", pilot.read_record(primary_path.parent / f"{target}__11/RESULT.json"))
        arms: dict[str, Any] = {
            method: {
                "status": "failed" if failed and index == position == 0 else "complete",
                "fidelities": dict.fromkeys(reporting.channels.NOISE_IDS, 0.8),
                "comparison": "full_update_cap_heterogeneous_compute",
                "cpu_seconds": 4.0,
            }
            for position, method in enumerate(mixture_baseline_experiment.METHODS)
        }
        arms.update({
            method: {
                **arm,
                "comparison": "objective_matched_deterministic_reference_heterogeneous_call_cap",
                "cpu_scope": protocol["cpu_scope"],
            }
            for method, arm in primary_record["arms"].items()
        })
        common = {
            method: {
                "status": arm["status"],
                "fidelities": dict.fromkeys(reporting.channels.NOISE_IDS, 0.6),
                "comparison": "common_cpu_last_available_final_iterate",
                "cpu_seconds": 2.0,
            }
            for method, arm in arms.items()
        }
        folder = output / f"{target}__11"
        pilot.write_record(folder / "durable.json", {"actual_count": 20})
        record: dict[str, Any] = {
            "target_id": target,
            "replica": 11,
            "protocol_checksum": pilot.digest(protocol),
            "status": "complete_with_failed_arms" if failed else "complete",
            "arms": arms,
            "common_cpu": {"arms": common, "cpu_seconds": 2.0, "status": "failed" if failed else "complete"},
            "unchanged_start": {
                "fidelities": dict.fromkeys(reporting.channels.NOISE_IDS, 0.1),
                "comparison": "unchanged_construction_reference_no_additional_training",
            },
            "reference_case_checksum": pilot.digest(primary_record),
            "artifact_hashes": {"durable.json": reporting.file_sha256(folder / "durable.json")},
            "signature": pilot.digest([
                protocol["version"],
                source["candidate"]["circuit"],
                source["candidate"]["theta"],
                reporting.target_hash(target),
                protocol["config"],
                target,
                11,
                pilot.digest(protocol),
                pilot.digest(primary_record),
            ]),
        }
        pilot.write_record(folder / "RESULT.json", record)
        historical_path = history_root / f"{target}__11/RESULT.json"
        pilot.write_record(
            folder / "REFERENCES.json",
            {
                "case_checksum": pilot.digest(record),
                "historical_unmatched": {
                    method: {
                        "comparison": "unmatched_historical_fixed_1100_individual_channel_call_cap",
                        "source_sha256": history_files[str(historical_path)],
                        "source_file": str(historical_path),
                        "record": history_records[target]["arms"][method],
                    }
                    for method in reporting.MIXTURE_HISTORICAL_METHODS
                },
            },
        )
        records.append(record)
    pilot.write_record(
        output / "RESULT.json",
        {
            "status": "complete_with_failed_arms" if failed else "complete",
            "protocol_checksum": pilot.digest(protocol),
            "cases": [pilot.digest(record) for record in records],
        },
    )
    mixture_baseline_experiment.write_scoreboard(output, records)
    return output / "SCOREBOARD.csv"


@pytest.mark.parametrize("failed", [False, True])
def test_mixture_preserves_all_four_comparison_strata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, failed: bool
) -> None:
    """Historical context survives canonical regeneration without pooling budgets."""
    path = mixture_fixture(tmp_path, monkeypatch, failed=failed)
    result = reporting.collect([path], tmp_path / "overview")
    assert set(result["panels"]) == {
        "mixture/" + panel for panel in ("full_cap", "common_cpu", "historical_unmatched", "unchanged")
    }
    assert result["sources"]["mixture"]["expected_cells"] == 480
    assert result["sources"]["mixture"]["missing_csv_cells"] == 0
    assert result["panels"]["mixture/historical_unmatched"]["methods"]["local_adam [historical_unmatched]"][
        "mean_noisy_fidelity"
    ] == pytest.approx(0.4)
    assert result["panels"]["mixture/unchanged"]["methods"]["unchanged_start [unchanged]"][
        "mean_noisy_fidelity"
    ] == pytest.approx(0.1)
    for panel, value in (("full_cap", 0.8), ("common_cpu", 0.6)):
        method = result["panels"]["mixture/" + panel]["methods"][f"fixed_mean2_weighted_adam [{panel}]"]
        assert method["status"] == ("incomplete" if failed else "complete")
        if not failed:
            assert method["mean_noisy_fidelity"] == pytest.approx(value)
    before = (tmp_path / "overview/SCOREBOARD.json").read_bytes()
    reporting.collect([path], tmp_path / "overview")
    assert (tmp_path / "overview/SCOREBOARD.json").read_bytes() == before


def test_mixture_missing_paired_cpu_arm_remains_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A full-cap result cannot replace a missing method on the matched CPU panel."""
    path = mixture_fixture(tmp_path, monkeypatch)
    write_rows(
        path,
        [
            row
            for row in read_rows(path)
            if not (row["panel"] == "common_cpu" and row["method"] == "fixed_mean2_weighted_adam")
        ],
    )
    result = reporting.collect([path], tmp_path / "overview")
    assert result["sources"]["mixture"]["missing_csv_cells"] == 24
    assert (
        result["panels"]["mixture/common_cpu"]["methods"]["fixed_mean2_weighted_adam [common_cpu]"]["status"]
        == "incomplete"
    )
    assert (
        result["panels"]["mixture/full_cap"]["methods"]["fixed_mean2_weighted_adam [full_cap]"]["status"] == "complete"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "context_checksum",
        "historical_score",
        "historical_path",
        "missing_context",
        "artifact",
        "artifact_coverage",
        "primary_case",
        "copied_arm",
        "historical_file",
        "starts",
        "stratum",
        "resource",
    ],
)
def test_mixture_provenance_tampering_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Rechecksummed auxiliary rows cannot change their immutable upstream source."""
    path = mixture_fixture(tmp_path, monkeypatch)
    folder = path.parent / "gaussian_mu0p5_sigma0p1__11"
    context_path = folder / "REFERENCES.json"
    context = cast("dict[str, Any]", pilot.read_record(context_path))
    if mutation == "missing_context":
        context_path.unlink()
    elif mutation in {"context_checksum", "historical_score", "historical_path"}:
        if mutation == "context_checksum":
            context["case_checksum"] = "changed"
        elif mutation == "historical_path":
            context["historical_unmatched"]["local_adam"]["source_file"] = "another_case.json"
        else:
            context["historical_unmatched"]["local_adam"]["record"]["final_score"]["fidelities"]["noiseless"] = 0.99
        pilot.write_record(context_path, context)
    elif mutation == "artifact":
        pilot.write_record(folder / "durable.json", {"actual_count": 0})
    elif mutation == "starts":
        pilot.write_record(tmp_path / "starts.json", {"starts": []})
    elif mutation in {"artifact_coverage", "copied_arm", "primary_case", "historical_file"}:
        source_folder = folder
        if mutation in {"primary_case", "historical_file"}:
            source_folder = tmp_path / ("primary" if mutation == "primary_case" else "history") / folder.name
        record_path = source_folder / "RESULT.json"
        record = cast("dict[str, Any]", pilot.read_record(record_path))
        if mutation == "artifact_coverage":
            record["artifact_hashes"] = {}
        else:
            method = "structural_mean2_lbfgs" if mutation == "copied_arm" else next(iter(record["arms"]))
            record["arms"][method]["status"] = "failed"
        pilot.write_record(record_path, record)
    else:
        data = read_rows(path)
        data[0]["panel" if mutation == "stratum" else "cpu_seconds"] = (
            "best_checkpoint" if mutation == "stratum" else "0"
        )
        write_rows(path, data)
    with pytest.raises(ValueError, match=r"Mixture|missing|Case|Panel|CSV|Unknown"):
        reporting.authenticate_panel(path)


def test_mixture_canonical_writer_requires_auxiliary_context() -> None:
    """Historical cells must never silently disappear from source regeneration."""
    with pytest.raises(ValueError, match="requires authenticated historical context"):
        reporting.canonical_rows("mixture_baseline", [])


@pytest.mark.parametrize("mutation", ["failed", "missing", "signature", "resource"])
def test_risk_adapter_failures_and_case_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Risk rows preserve failed controls and bind identity and charged work."""
    path = panel_fixture(tmp_path / "risk", monkeypatch, "risk", failed=mutation == "failed")
    if mutation == "signature":
        record_path = path.parent / "gaussian_mu0p5_sigma0p1__11/RESULT.json"
        record = cast("dict[str, Any]", pilot.read_record(record_path))
        record["signature"] = "changed"
        pilot.write_record(record_path, record)
    elif mutation in {"missing", "resource"}:
        data = read_rows(path)
        if mutation == "missing":
            data.pop(0)
        else:
            data[0]["individual_channel_calls"] = "0"
        write_rows(path, data)
    if mutation in {"signature", "resource"}:
        with pytest.raises(ValueError, match=r"Case identity|CSV cell"):
            reporting.authenticate_panel(path)
    else:
        result = reporting.collect([path], tmp_path / "overview")
        assert result["panels"]["risk"]["methods"][risk_experiment.METHODS[0]]["status"] == "incomplete"


def long_ensemble_fixture(root: Path, monkeypatch: pytest.MonkeyPatch, *, failed: bool = False) -> Path:
    """Return a provenance-complete four-arm fixture with distinct CPU endpoints."""
    base_path = panel_fixture(root / "base", monkeypatch, "ensemble_structural")
    base = cast("dict[str, Any]", pilot.read_record(base_path.parent / "PROTOCOL.json"))
    output = root / "long"
    protocol: dict[str, Any] = {
        **base,
        "version": long_ensemble_experiment.VERSION,
        "arms": long_ensemble_experiment.arm_specs(long_ensemble_experiment.Config()),
    }
    pilot.write_record(output / "PROTOCOL.json", protocol)
    records = []
    for index, source in enumerate(protocol["starts"]):
        target = source["target_id"]
        folder = output / f"{target}__11"
        pilot.write_record(folder / "durable.json", {"counter": 20})
        arms, scores = {}, {}
        for position, method in enumerate(protocol["arms"]):
            status = "failed" if failed and index == position == 0 else "complete"
            arms[method] = {
                "status": status,
                "training": {"cpu_seconds": 4.0, "individual_channel_gradient_calls": 20},
                "final_score": {"status": status, "fidelities": dict.fromkeys(reporting.channels.NOISE_IDS, 0.8)},
                "physical_work": {
                    "physical_attempted_channel_calls": 20,
                    "physical_completed_channel_calls": 20,
                    "physical_completed_direction_passes": 40,
                    "known_training_cpu_seconds": 4.0,
                    "unknown_kill_cost": False,
                },
            }
            scores[method] = {"status": status, "fidelities": dict.fromkeys(reporting.channels.NOISE_IDS, 0.6)}
        record: dict[str, Any] = {
            "signature": pilot.digest({
                "version": protocol["version"],
                "source": source,
                "config": protocol["config"],
                "target": reporting.target_hash(target),
                "order_index": index,
                "protocol_checksum": pilot.digest(protocol),
            }),
            "status": "complete_with_failed_arms" if failed else "complete",
            "target_id": target,
            "replica": 11,
            "arms": arms,
            "common_cpu_comparison": {
                "common_cpu_seconds": 2.0,
                "points": [{"scores": scores, "selected": {method: {"cpu_seconds": 2.0} for method in arms}}],
            },
            "artifact_hashes": {"durable.json": reporting.file_sha256(folder / "durable.json")},
        }
        pilot.write_record(folder / "RESULT.json", record)
        records.append(record)
    pilot.write_record(
        output / "RESULT.json",
        {
            "status": "complete_with_failed_arms" if failed else "complete",
            "protocol_checksum": pilot.digest(protocol),
            "results": [pilot.digest(record) for record in records],
        },
    )
    long_ensemble_experiment.write_scoreboard(output, records)
    return output / "SCOREBOARD.csv"


@pytest.mark.parametrize("failed", [False, True])
def test_long_ensemble_full_and_cpu_strata_keep_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, failed: bool
) -> None:
    """Full-cap curves cannot replace the paired four-arm common-CPU endpoint."""
    path = long_ensemble_fixture(tmp_path, monkeypatch, failed=failed)
    result = reporting.collect([path], tmp_path / "overview")
    assert set(result["panels"]) == {"long/full_cap", "long/common_cpu"}
    assert result["sources"]["long"]["expected_cells"] == 192
    for panel, value in (("full_cap", 0.8), ("common_cpu", 0.6)):
        method = result["panels"]["long/" + panel]["methods"][f"mean2_lbfgs [{panel}]"]
        assert method["status"] == ("incomplete" if failed else "complete")
        if not failed:
            assert method["mean_noisy_fidelity"] == pytest.approx(value)


@pytest.mark.parametrize("change", ["artifact", "signature", "resource", "missing_pair"])
def test_long_ensemble_provenance_and_pair_completeness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Changed durable costs are rejected and a missing CPU pair stays incomplete."""
    path = long_ensemble_fixture(tmp_path, monkeypatch)
    folder = path.parent / "gaussian_mu0p5_sigma0p1__11"
    if change == "artifact":
        pilot.write_record(folder / "durable.json", {"counter": 0})
    elif change == "signature":
        record = cast("dict[str, Any]", pilot.read_record(folder / "RESULT.json"))
        record["signature"] = "changed"
        pilot.write_record(folder / "RESULT.json", record)
    else:
        data = read_rows(path)
        if change == "resource":
            data[0]["physical_attempted_channel_calls"] = "0"
        else:
            data = [row for row in data if row["panel"] != "common_cpu" or row["method"] != "mean11_adam"]
        write_rows(path, data)
    if change == "missing_pair":
        result = reporting.collect([path], tmp_path / "overview")
        assert result["sources"]["long"]["missing_csv_cells"] == 24
        assert result["panels"]["long/common_cpu"]["methods"]["mean11_adam [common_cpu]"]["status"] == "incomplete"
        assert result["panels"]["long/full_cap"]["methods"]["mean11_adam [full_cap]"]["status"] == "complete"
    else:
        with pytest.raises(ValueError, match=r"Long-ensemble|Case identity|CSV cell"):
            reporting.authenticate_panel(path)


def initializer_fixture(root: Path, monkeypatch: pytest.MonkeyPatch, *, failed: bool = False) -> Path:
    """Return six full-endpoint methods and independently checksummed construction inputs."""
    base_path = panel_fixture(root / "base", monkeypatch, "ensemble_structural")
    base = cast("dict[str, Any]", pilot.read_record(base_path.parent / "PROTOCOL.json"))
    targets = list(initializer_experiment.TARGETS)
    environment = {"numerical_environment": "fixed"}
    construction_root = root / "construction"
    gates = [ParameterizedGate("rx", (index % 6,), param_index=index) for index in range(78)]
    gates.extend(ParameterizedGate("rzz", (0, 1), param_index=index) for index in range(78, 108))
    circuit = ParameterizedCircuit(6, gates)
    cases = []
    for target in targets:
        folder = construction_root / f"{target}__11"
        archive, archive_manifest = folder / "archive.json", folder / "archive_manifest.json"
        pilot.write_record(archive, {"original_endpoint": target})
        pilot.write_record(archive_manifest, {"original_manifest": target})
        manifest: dict[str, Any] = {
            "version": initializer_experiment.construction.VERSION,
            "config": asdict(initializer_experiment.construction.Config()),
            "target_id": target,
            "replica": 11,
            "target_vector_sha256": reporting.target_hash(target),
            "sources": base["source_hashes"],
            "environment": environment,
            "benchmark_hashes": base["benchmark_hashes"],
            "archive_reference": {
                "path": str(archive),
                "checksum": pilot.digest(pilot.read_record(archive)),
                "manifest_path": str(archive_manifest),
                "manifest_checksum": pilot.digest(pilot.read_record(archive_manifest)),
            },
        }
        pilot.write_record(folder / "PROTOCOL.json", manifest)
        pilot.write_record(folder / "ATTEMPTS.json", {"cpu_seconds": 100.0})
        pilot.write_record(folder / "blocks" / "complete.json", {"updates": 570})
        built: dict[str, Any] = {
            "signature": pilot.digest(manifest),
            "status": "complete",
            "circuit": pilot.circuit_record(circuit),
            "theta": [0.2] * 108,
            "timing_complete": True,
            "construction_cpu_seconds": 100.0,
            "physical_counts": reporting.runner.counts(circuit),
            "parameters": 108,
            "updates": 570,
            "archive_reproduction": {"matches": True},
            "attempts_checksum": pilot.digest(pilot.read_record(folder / "ATTEMPTS.json")),
            "blocks": {"complete": {"checksum": pilot.digest(pilot.read_record(folder / "blocks/complete.json"))}},
        }
        pilot.write_record(folder / "RESULT.json", built)
        cases.append({
            "target_id": target,
            "replica": 11,
            "root": str(folder),
            "manifest_checksum": pilot.digest(manifest),
        })
    batch: dict[str, Any] = {
        "version": initializer_experiment.construction_experiment.VERSION,
        "targets": targets,
        "replicas": [11],
        "phase": "development",
        "cases": cases,
        "sources": base["source_hashes"],
        "environment": environment,
        "benchmark_hashes": base["benchmark_hashes"],
    }
    pilot.write_record(construction_root / "BATCH_PROTOCOL.json", batch)
    audit_checksums = {}
    for case in cases:
        folder = construction_root / f"{case['target_id']}__11"
        audit: dict[str, Any] = {
            "signature": pilot.digest({"protocol": batch, "case": case}),
            "construction_checksum": pilot.digest(pilot.read_record(folder / "RESULT.json")),
            "construction_cpu_seconds": 100.0,
            "status": "complete",
            "matches": True,
        }
        pilot.write_record(folder / "AUDIT.json", audit)
        audit_checksums[case["target_id"]] = pilot.digest(audit)
    pilot.write_record(
        construction_root / "SUMMARY.json",
        {
            "status": "complete",
            "all_archive_reproductions_match": True,
            "protocol_checksum": pilot.digest(batch),
            "case_checksums": audit_checksums,
        },
    )
    starts = []
    for case in cases:
        folder = construction_root / f"{case['target_id']}__11"
        built = cast("dict[str, Any]", pilot.read_record(folder / "RESULT.json"))
        starts.append({
            "target_id": case["target_id"],
            "replica": 11,
            "circuit": built["circuit"],
            "theta": built["theta"],
            "construction_cpu_seconds": 100.0,
            "timing_complete": True,
            "files": {
                str(path): pilot.digest(pilot.read_record(path))
                for path in (
                    folder / "RESULT.json",
                    folder / "AUDIT.json",
                    folder / "PROTOCOL.json",
                    construction_root / "BATCH_PROTOCOL.json",
                    construction_root / "SUMMARY.json",
                )
            },
        })
    output = root / "initializer"
    protocol: dict[str, Any] = {
        "version": initializer_experiment.VERSION,
        "phase": "development",
        "targets": targets,
        "replicas": [11],
        "methods": list(initializer_experiment.METHODS),
        "threshold": 0.99,
        "config": asdict(initializer_experiment.initializer_training.Config()),
        "sources": base["source_hashes"],
        "benchmark_hashes": base["benchmark_hashes"],
        "environment": environment,
        "starts": starts,
    }
    pilot.write_record(output / "PROTOCOL.json", protocol)
    records = []
    for index, source in enumerate(starts):
        target = source["target_id"]
        folder = output / f"{target}__11"
        pilot.write_record(folder / "INITIALIZER_ATTEMPTS.json", {"attempts": [{"cpu_seconds": 1.0}]})
        initials: dict[str, Any] = {
            "attempts_checksum": pilot.digest(pilot.read_record(folder / "INITIALIZER_ATTEMPTS.json")),
            "legacy": {
                "circuit": source["circuit"],
                "theta": source["theta"],
                "initializer_cpu_seconds": source["construction_cpu_seconds"],
                "selected_source": "legacy",
            },
            "conditional": {
                "circuit": pilot.circuit_record(circuit),
                "theta": [0.2] * 108,
                "initializer_cpu_seconds": 1.0,
                "selected_source": "rank_two",
            },
        }
        pilot.write_record(folder / "INITIALIZERS.json", initials)
        arms = {}
        for position, method in enumerate(initializer_experiment.METHODS):
            score: dict[str, Any] = {
                "signature": pilot.digest([target, method]),
                "fidelities": dict.fromkeys(reporting.channels.NOISE_IDS, 0.8),
                "mean_noisy_fidelity": 0.8,
                "physical_counts": {"one_qubit": 20, "two_qubit": 10},
                "ballarin_counts": {"one_qubit": 25, "two_qubit": 10},
            }
            pilot.write_record(folder / "scores" / f"{score['signature']}.json", score)
            trained = not method.endswith("unchanged")
            training: dict[str, Any] = {"optimizer": method, "updates": 1500}
            if trained:
                pilot.write_record(folder / method / "TRAINING.json", training)
            arms[method] = {
                "status": "failed" if failed and index == position == 0 else "complete",
                "scores": score,
                "selected_source": "rank_two" if method.startswith("conditional") else "legacy",
                "initializer_cpu_seconds": 100.0 if method.startswith("legacy") else 1.0,
                "refinement_cpu_seconds": 5.0 if trained else 0.0,
                "total_cpu_seconds": (100.0 if method.startswith("legacy") else 1.0) + (5.0 if trained else 0.0),
                "training_checksum": pilot.digest(training) if trained else None,
                "individual_channel_gradient_calls": 3000 if trained else 0,
            }
        availability = {
            "status": "complete",
            "clock": "total_construction_and_refinement",
            "common_cpu_seconds": 6.0,
            "points": [
                {
                    "fraction": 1.0,
                    "cpu_seconds": 6.0,
                    "all_available": False,
                    "arms": {
                        method: {
                            "status": "initializer_not_yet_available" if method.startswith("legacy") else "available"
                        }
                        for method in arms
                        if not method.endswith("unchanged")
                    },
                }
            ],
        }
        record: dict[str, Any] = {
            "signature": pilot.digest({"protocol": protocol, "source": source}),
            "status": "partial_failure" if failed else "complete",
            "target_id": target,
            "replica": 11,
            "initializer_checksum": pilot.digest(initials),
            "ideal_tt_overlap": 0.999,
            "rank_two_selected": True,
            "arms": arms,
            "common_total_cpu": availability,
            "common_refinement_cpu": {
                "status": "complete",
                "clock": "refinement_only_including_deployment",
                "points": [],
            },
            "dependencies": {
                str(path.relative_to(folder)): reporting.file_sha256(path) for path in folder.rglob("*.json")
            },
        }
        pilot.write_record(folder / "RESULT.json", record)
        records.append(record)
    initializer_experiment.write_summary(output, protocol, records)
    return output / "SCOREBOARD.csv"


@pytest.mark.parametrize("failed", [False, True])
def test_initializer_full_rows_preserve_common_total_unavailability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, failed: bool
) -> None:
    """The source CSV stays full-endpoint-only while provenance retains missing initializers."""
    path = initializer_fixture(tmp_path, monkeypatch, failed=failed)
    result = reporting.collect([path], tmp_path / "overview")
    assert set(result["panels"]) == {"initializer"}
    assert result["sources"]["initializer"]["expected_cells"] == 288
    method = result["panels"]["initializer"]["methods"]["legacy_unchanged"]
    assert method["status"] == ("incomplete" if failed else "complete")
    availability = result["sources"]["initializer"]["common_cpu_availability"]
    assert set(availability) == set(initializer_experiment.TARGETS)
    for case in availability.values():
        total = case["common_total_cpu"]
        assert total["clock"] == "total_construction_and_refinement"
        assert not total["points"][0]["all_available"]
        assert total["points"][0]["arms"]["legacy_adam"] == "initializer_not_yet_available"


@pytest.mark.parametrize(
    "change", ["config", "source", "construction_block", "archive", "dependency", "summary", "resource", "missing_pair"]
)
def test_initializer_custody_and_case_completeness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Changed construction, score, CPU or summary provenance cannot become a comparison."""
    path = initializer_fixture(tmp_path, monkeypatch)
    target = initializer_experiment.TARGETS[0]
    construction = tmp_path / "construction" / f"{target}__11"
    if change == "config":
        protocol = cast("dict[str, Any]", pilot.read_record(path.parent / "PROTOCOL.json"))
        protocol["config"]["joint_cap"] = 1
        pilot.write_record(path.parent / "PROTOCOL.json", protocol)
    elif change in {"source", "construction_block", "archive"}:
        source_path = (
            construction
            / {"source": "RESULT.json", "construction_block": "blocks/complete.json", "archive": "archive.json"}[change]
        )
        pilot.write_record(source_path, {"changed": True})
    elif change == "dependency":
        pilot.write_record(path.parent / f"{target}__11/legacy_adam/TRAINING.json", {"changed": True})
    elif change == "summary":
        summary = cast("dict[str, Any]", pilot.read_record(path.parent / "SUMMARY.json"))
        summary["equal_target_all_noisy_means"]["legacy_adam"] = 0.99
        pilot.write_record(path.parent / "SUMMARY.json", summary)
    else:
        data = read_rows(path)
        if change == "resource":
            data[0]["initializer_cpu_seconds"] = "0"
        else:
            data = [row for row in data if row["method"] != "conditional_adam"]
        write_rows(path, data)
    if change == "missing_pair":
        result = reporting.collect([path], tmp_path / "overview")
        assert result["sources"]["initializer"]["missing_csv_cells"] == 48
        assert result["panels"]["initializer"]["methods"]["conditional_adam"]["status"] == "incomplete"
        assert result["panels"]["initializer"]["methods"]["legacy_adam"]["status"] == "complete"
    else:
        with pytest.raises(ValueError, match=r"Initializer|CSV cell"):
            reporting.authenticate_panel(path)


def composed_fixture(root: Path, monkeypatch: pytest.MonkeyPatch, *, failed: bool = False) -> Path:
    """Return a complete composed panel with three distinct authenticated strata."""
    input_path = initializer_fixture(root / "input", monkeypatch)
    original = cast("dict[str, Any]", pilot.read_record(input_path.parent / "PROTOCOL.json"))
    starts: list[dict[str, Any]] = []
    files = {input_path.parent / "PROTOCOL.json", input_path.parent / "SUMMARY.json"}
    for source in original["starts"]:
        folder = input_path.parent / f"{source['target_id']}__11"
        built = cast("dict[str, Any]", pilot.read_record(folder / "INITIALIZERS.json"))
        starts.append({"target_id": source["target_id"], "replica": 11, "endpoint": built["conditional"]})
        files.update({folder / "RESULT.json", folder / "INITIALIZERS.json"})
    protocol: dict[str, Any] = {
        "version": composed_experiment.VERSION,
        "phase": "development",
        "targets": list(composed_experiment.TARGETS),
        "replicas": [11],
        "methods": list(composed_experiment.METHODS),
        "primary": list(composed_experiment.PRIMARY),
        "exact_config": asdict(composed_experiment.exact.Config(protected_initial_gates=0)),
        "stochastic_config": asdict(composed_experiment.stochastic.Config(protected=0)),
        "sources": original["sources"],
        "benchmark_hashes": original["benchmark_hashes"],
        "environment": original["environment"],
        "starts": starts,
        "initializer_root": str(input_path.parent.resolve()),
        "input_files": {str(path.resolve()): reporting.file_sha256(path) for path in files},
    }
    output = root / "composed"
    pilot.write_record(output / "PROTOCOL.json", protocol)
    records = []
    for target_index, source in enumerate(starts):
        target = source["target_id"]
        endpoint = source["endpoint"]
        folder = output / f"{target}__11"
        signature = pilot.digest([protocol, source])
        invocation: dict[str, Any] = {
            "signature": signature,
            "attempts": [{"status": "complete", "cpu_seconds": 100.0, "wall_seconds": 101.0}],
        }
        pilot.write_record(folder / "CASE_ATTEMPTS.json", invocation)
        circuit_hash = pilot.digest(endpoint["circuit"])
        pilot.write_record(folder / "circuits" / f"{circuit_hash}.json", endpoint["circuit"])
        initial: dict[str, Any] = {"circuit_hash": circuit_hash, "theta": endpoint["theta"], "cpu_seconds": 0.0}
        checkpoint = {**initial, "cpu_seconds": 1.0}

        def scored(
            point: dict[str, Any], value: float, case_root: Path = folder, case_target: str = target
        ) -> dict[str, Any]:
            identity = {"circuit_hash": point["circuit_hash"], "theta": point["theta"]}
            operation_signature = pilot.digest({
                "version": composed_experiment.exact.VERSION,
                "circuit": point["circuit_hash"],
                "theta": point["theta"],
                "target": reporting.target_hash(case_target),
                "ranking": False,
                "profiles": list(reporting.channels.NOISE_IDS),
            })
            operation_path = case_root / "operations" / f"audit_{pilot.digest(identity)}.json"
            attempts: dict[str, Any] = {
                "signature": operation_signature,
                "attempts": [{"status": "complete", "cpu_seconds": 0.1}],
            }
            pilot.write_record(operation_path.with_suffix(".attempts.json"), attempts)
            operation: dict[str, Any] = {
                "signature": operation_signature,
                "value": {"fidelities": dict.fromkeys(reporting.channels.NOISE_IDS, value)},
                "attempts_checksum": pilot.digest(attempts),
            }
            pilot.write_record(operation_path, operation)
            return {
                "evaluation": operation,
                "fidelities": operation["value"]["fidelities"],
                "parameters": 108,
                "physical_counts": {"one_qubit": 78, "two_qubit": 30, "depth": 40},
                "ballarin_counts": {"one_qubit": 150, "two_qubit": 30, "depth": 80},
            }

        initial_score = scored(initial, 0.4)
        arms: dict[str, Any] = {}
        for position, method in enumerate(composed_experiment.TRAINED):
            parameters = list(endpoint["theta"])
            parameters[0] += position + 1.0
            duration = 4.0 if method in composed_experiment.PRIMARY else 2.0 if method == "fixed_mean2_lbfgs" else 3.0
            final = {"circuit_hash": circuit_hash, "theta": parameters, "cpu_seconds": duration}
            arm: dict[str, Any] = {
                "signature": pilot.digest([
                    protocol["version"],
                    endpoint,
                    reporting.target_hash(target),
                    protocol["exact_config"],
                    protocol["stochastic_config"],
                    reporting.runner.seed_for("composition_training", target, 11),
                    method,
                ]),
                "status": "failed" if failed and target_index == position == 0 else "complete",
                "curve": [initial, checkpoint, final],
                "final": final,
                "scores": scored(final, 0.8),
                "initializer_cpu_seconds": 1.0,
                "refinement_cpu_seconds": duration,
                "total_cpu_seconds": duration + 1.0,
            }
            arm["dependencies"] = {
                str(path.relative_to(folder)): reporting.file_sha256(path)
                for path in folder.rglob("*.json")
                if path.name not in {"CASE_ATTEMPTS.json", "RESULT.json"}
            }
            pilot.write_record(folder / "arms" / f"{method}.json", arm)
            arms[method] = arm
        arms["unchanged"] = {
            "status": "complete",
            "scores": initial_score,
            "final": initial,
            "refinement_cpu_seconds": 0.0,
            "total_cpu_seconds": 1.0,
        }
        common = {}
        for panel, methods in (
            ("common_primary", composed_experiment.PRIMARY),
            ("common_all_trained", composed_experiment.TRAINED),
        ):
            if any(arms[method]["status"] != "complete" for method in methods):
                common[panel] = {"status": "unavailable_failed_arm", "methods": list(methods), "arms": {}}
                continue
            cutoff = min(arms[method]["refinement_cpu_seconds"] for method in methods)
            selected = {}
            for method in methods:
                point = next(value for value in reversed(arms[method]["curve"]) if value["cpu_seconds"] <= cutoff)
                selected[method] = {
                    "point": point,
                    "scores": arms[method]["scores"] if point == arms[method]["final"] else initial_score,
                }
            common[panel] = {
                "status": "complete",
                "methods": list(methods),
                "refinement_cpu_seconds": cutoff,
                "total_cpu_seconds": cutoff + 1.0,
                "arms": selected,
            }
        record: dict[str, Any] = {
            "signature": signature,
            "target_id": target,
            "replica": 11,
            "source": endpoint,
            "status": "partial_failure" if failed and target_index == 0 else "complete",
            "arms": arms,
            **common,
            "physical_invocation": {
                "cpu_seconds": 100.0,
                "wall_seconds": 101.0,
                "attempts_checksum": pilot.digest(invocation),
                "scope": "inclusive scientific case",
            },
            "dependencies": {
                str(path.relative_to(folder)): reporting.file_sha256(path) for path in folder.rglob("*.json")
            },
        }
        pilot.write_record(folder / "RESULT.json", record)
        records.append(record)
    composed_experiment.write_summary(output, protocol, records)
    return output / "SCOREBOARD.csv"


@pytest.mark.parametrize("failed", [False, True])
def test_composed_adapter_preserves_all_three_complete_method_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, failed: bool
) -> None:
    """Seven full arms, four primary controls and six trained controls remain distinct."""
    path = composed_fixture(tmp_path, monkeypatch, failed=failed)
    result = reporting.collect([path], tmp_path / "overview")
    assert result["sources"]["composed"]["expected_cells"] == 816
    assert result["sources"]["composed"]["missing_csv_cells"] == 0
    assert set(result["panels"]) == {
        "composed/" + panel for panel in ("full_cap", "common_primary", "common_all_trained")
    }
    for panel, count in (("full_cap", 7), ("common_primary", 4), ("common_all_trained", 6)):
        methods = result["panels"]["composed/" + panel]["methods"]
        assert len(methods) == count
        assert methods[f"structural_mean2_lbfgs [{panel}]"]["status"] == ("incomplete" if failed else "complete")
    physical = result["sources"]["composed"]["physical_invocations"]
    assert all(value["cpu_seconds"] == pytest.approx(100.0) for value in physical.values())
    assert "nested numerical clocks are not added" in result["sources"]["composed"]["physical_cpu_aggregation"]


@pytest.mark.parametrize(
    "change",
    [
        "input",
        "protection",
        "dependency",
        "score",
        "chronology",
        "clock",
        "unknown_kill",
        "active_arm_dependency",
        "resource",
        "missing_pair",
    ],
)
def test_composed_adapter_rejects_tampering_and_keeps_missing_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Availability, conditional inputs and inclusive resource custody cannot be rewritten."""
    path = composed_fixture(tmp_path, monkeypatch)
    target = composed_experiment.TARGETS[0]
    folder = path.parent / f"{target}__11"
    record = cast("dict[str, Any]", pilot.read_record(folder / "RESULT.json"))
    if change == "input":
        pilot.write_record(tmp_path / "input/initializer" / f"{target}__11/INITIALIZERS.json", {"changed": True})
    elif change == "protection":
        protocol = cast("dict[str, Any]", pilot.read_record(path.parent / "PROTOCOL.json"))
        protocol["exact_config"]["protected_initial_gates"] = 18
        pilot.write_record(path.parent / "PROTOCOL.json", protocol)
    elif change == "dependency":
        pilot.write_record(
            folder / "circuits" / f"{record['arms']['unchanged']['final']['circuit_hash']}.json", {"changed": True}
        )
    elif change in {"score", "chronology", "clock", "unknown_kill", "active_arm_dependency"}:
        method = "structural_mean2_lbfgs"
        if change == "score":
            record["common_primary"]["arms"][method]["scores"]["fidelities"]["noiseless"] = 0.99
        elif change == "chronology":
            record["common_primary"]["arms"][method]["point"] = record["arms"][method]["curve"][0]
        elif change == "clock":
            record["physical_invocation"]["cpu_seconds"] = 200.0
        elif change == "unknown_kill":
            invocation = cast("dict[str, Any]", pilot.read_record(folder / "CASE_ATTEMPTS.json"))
            invocation["attempts"][-1]["status"] = "running"
            pilot.write_record(folder / "CASE_ATTEMPTS.json", invocation)
            record["dependencies"]["CASE_ATTEMPTS.json"] = reporting.file_sha256(folder / "CASE_ATTEMPTS.json")
        else:
            record["arms"][method]["dependencies"]["CASE_ATTEMPTS.json"] = record["dependencies"]["CASE_ATTEMPTS.json"]
            arm_path = folder / "arms" / f"{method}.json"
            pilot.write_record(arm_path, record["arms"][method])
            record["dependencies"][str(arm_path.relative_to(folder))] = reporting.file_sha256(arm_path)
        pilot.write_record(folder / "RESULT.json", record)
    else:
        data = read_rows(path)
        if change == "resource":
            data[0]["total_cpu_seconds"] = "0"
        else:
            data = [row for row in data if row["panel"] != "common_primary" or row["method"] != "structural_mean2_adam"]
        write_rows(path, data)
    if change == "missing_pair":
        result = reporting.collect([path], tmp_path / "overview")
        assert result["sources"]["composed"]["missing_csv_cells"] == 48
        assert (
            result["panels"]["composed/common_primary"]["methods"]["structural_mean2_adam [common_primary]"]["status"]
            == "incomplete"
        )
        assert (
            result["panels"]["composed/full_cap"]["methods"]["structural_mean2_adam [full_cap]"]["status"] == "complete"
        )
    else:
        with pytest.raises(ValueError, match=r"Composition|Initializer|CSV cell"):
            reporting.authenticate_panel(path)


def common_control_fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return seven source-bound controls with two separately receipted score audits."""
    parent = composed_fixture(root / "input", monkeypatch).parent
    original = cast("dict[str, Any]", pilot.read_record(parent / "PROTOCOL.json"))
    summary = pilot.read_record(parent / "SUMMARY.json")
    source_cases = [
        cast("dict[str, Any]", pilot.read_record(parent / f"{target}__11/RESULT.json"))
        for target in composed_experiment.TARGETS
    ]
    entries: list[dict[str, Any]] = [
        {
            "target_id": case["target_id"],
            "checksum": pilot.digest(case),
            "points": common_control_audit.select_points(
                case["arms"], composed_experiment.TRAINED, case["common_primary"]["refinement_cpu_seconds"]
            ),
        }
        for case in source_cases
    ]
    protocol: dict[str, Any] = {
        "version": common_control_audit.VERSION,
        "phase": "development_diagnostic",
        "sources": original["sources"],
        "benchmark_hashes": original["benchmark_hashes"],
        "environment": original["environment"],
        "input_panel": {
            "root": str(parent.resolve()),
            "protocol_checksum": pilot.digest(original),
            "summary_checksum": pilot.digest(summary),
            "cases": entries,
        },
    }
    output = root / "common-control"
    pilot.write_record(output / "PROTOCOL.json", protocol)
    records = []
    for entry, case in zip(entries, source_cases, strict=True):
        target = case["target_id"]
        folder, source_folder = output / f"{target}__11", parent / f"{target}__11"
        scores = {}
        for method in composed_experiment.TRAINED:
            if method in composed_experiment.PRIMARY:
                scores[method] = case["common_primary"]["arms"][method]["scores"]
                continue
            point = entry["points"][method]
            circuit_path = f"circuits/{point['circuit_hash']}.json"
            (folder / "circuits").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_folder / circuit_path, folder / circuit_path)
            identity = {"circuit_hash": point["circuit_hash"], "theta": point["theta"]}
            operation_path = f"operations/audit_{pilot.digest(identity)}.json"
            operation = cast("dict[str, Any]", pilot.read_record(source_folder / operation_path))
            operation["cpu_seconds"] = 0.1
            operation["wall_seconds"] = 0.11
            operation_attempts = cast(
                "dict[str, Any]", pilot.read_record((source_folder / operation_path).with_suffix(".attempts.json"))
            )
            operation_attempts["attempts"][0]["wall_seconds"] = 0.11
            operation_attempts["attempts"][0].update(attempted_channel_calls=12, completed_channel_calls=12)
            operation.update(
                individual_channel_calls=12,
                density_direction_passes=12,
                physical_attempted_channel_calls=12,
                physical_completed_channel_calls=12,
                physical_completed_direction_passes=12,
            )
            operation["attempts_checksum"] = pilot.digest(operation_attempts)
            pilot.write_record((folder / operation_path).with_suffix(".attempts.json"), operation_attempts)
            pilot.write_record(folder / operation_path, operation)
            circuit = cast("dict[str, Any]", pilot.read_record(folder / circuit_path))
            scores[method] = {
                **case["arms"][method]["scores"],
                **reporting.endpoint_resources(circuit, point["theta"]),
                "evaluation": operation,
            }
        value: dict[str, Any] = {
            "status": "complete",
            "target_id": target,
            "refinement_cpu_seconds": case["common_primary"]["refinement_cpu_seconds"],
            "total_cpu_seconds": case["common_primary"]["total_cpu_seconds"],
            "points": entry["points"],
            "source_case_checksum": entry["checksum"],
            "source_unchanged_fidelities": case["arms"]["unchanged"]["scores"]["fidelities"],
            "scores": scores,
            "dependencies": {
                str(path.relative_to(folder)): reporting.file_sha256(path) for path in folder.rglob("*.json")
            },
        }
        signature = pilot.digest(pilot.digest([protocol, entry]))
        attempts: dict[str, Any] = {
            "signature": signature,
            "attempts": [{"status": "complete", "cpu_seconds": 1.0, "wall_seconds": 1.1}],
        }
        pilot.write_record(folder / "RESULT.attempts.json", attempts)
        record: dict[str, Any] = {
            "signature": signature,
            "value": value,
            "cpu_seconds": 1.0,
            "wall_seconds": 1.1,
            "attempts_checksum": pilot.digest(attempts),
        }
        pilot.write_record(folder / "RESULT.json", record)
        records.append(record)
    pilot.write_record(
        output / "SUMMARY.json",
        {
            "status": "complete",
            "protocol_checksum": pilot.digest(protocol),
            "cases": [pilot.digest(record) for record in records],
            **common_control_audit.summarize(records, composed_experiment.TARGETS),
        },
    )
    return reporting.write_common_control_scoreboard(output)


def state_compiler_fixture(root: Path, monkeypatch: pytest.MonkeyPatch, *, failed: bool = False) -> Path:
    """Return 24 metadata-only pair receipts without executing a physical simulator."""
    parent = composed_fixture(root / "input", monkeypatch).parent
    original = cast("dict[str, Any]", pilot.read_record(parent / "PROTOCOL.json"))
    pairs, files = [], dict(original["input_files"])
    for target in state_compiler.TARGETS:
        folder = parent / f"{target}__11"
        case = cast("dict[str, Any]", pilot.read_record(folder / "RESULT.json"))
        initials = cast(
            "dict[str, Any]", pilot.read_record(parent.parent / "input/initializer" / f"{target}__11/INITIALIZERS.json")
        )
        endpoints = {f"{name}_initializer": initials[name] for name in ("legacy", "conditional")}
        for method in composed_experiment.PRIMARY:
            point = case["arms"][method]["final"]
            endpoints[method] = {
                "circuit": pilot.read_record(folder / "circuits" / f"{point['circuit_hash']}.json"),
                "theta": point["theta"],
            }
        pairs.extend(
            {
                "target_id": target,
                "replica": 11,
                "label": label,
                "endpoint": {key: endpoints[label][key] for key in ("circuit", "theta")},
                "source_case_checksum": pilot.digest(case),
                "target_sha256": reporting.target_hash(target),
            }
            for label in state_compiler.LABELS
        )
        files[str((folder / "RESULT.json").resolve())] = reporting.file_sha256(folder / "RESULT.json")
        files.update({
            str((folder / name).resolve()): fingerprint for name, fingerprint in case["dependencies"].items()
        })
    files.update({
        str((parent / name).resolve()): reporting.file_sha256(parent / name)
        for name in ("PROTOCOL.json", "SUMMARY.json")
    })
    protocol: dict[str, Any] = {
        "version": state_compiler.VERSION,
        "phase": "development",
        "composition_root": str(parent.resolve()),
        "sources": original["sources"],
        "benchmark_hashes": original["benchmark_hashes"],
        "environment": original["environment"],
        "targets": list(state_compiler.TARGETS),
        "replicas": [11],
        "labels": list(state_compiler.LABELS),
        "pairs": pairs,
        "input_files": files,
        "noise_ids": list(reporting.channels.NOISE_IDS),
        "tolerance": state_compiler.compiler.EIGENSTATE_ATOL,
        "cpu_scope": state_compiler.CPU_SCOPE,
        "source_selection": "all fixed method final endpoints; no quality selection",
    }
    output = root / "state-compiler"
    pilot.write_record(output / "PROTOCOL.json", protocol)
    records, failures = [], {}
    for index, source in enumerate(pairs):
        key = state_compiler.pair_key(source)
        folder = output / key
        if failed and index == 0:
            failure = {
                "status": "failed",
                "target_id": source["target_id"],
                "replica": 11,
                "label": source["label"],
                "failure": "fixture interruption",
            }
            failures[key] = failure
            pilot.write_record(folder / "FAILURE.json", failure)
            continue
        signature = pilot.digest([protocol, source])
        invocation: dict[str, Any] = {
            "signature": signature,
            "attempts": [{"status": "complete", "cpu_seconds": 1.0, "wall_seconds": 1.1}],
        }
        pilot.write_record(folder / "CASE_ATTEMPTS.json", invocation)
        compiler_signature = pilot.digest([state_compiler.VERSION, source])
        compiler_attempts: dict[str, Any] = {
            "signature": compiler_signature,
            "attempts": [{"status": "complete", "cpu_seconds": 0.1, "wall_seconds": 0.11}],
        }
        pilot.write_record(folder / "COMPILATION.attempts.json", compiler_attempts)
        compilation: dict[str, Any] = {
            "signature": compiler_signature,
            "value": source["endpoint"],
            "cpu_seconds": 0.1,
            "wall_seconds": 0.11,
            "attempts_checksum": pilot.digest(compiler_attempts),
        }
        pilot.write_record(folder / "COMPILATION.json", compilation)
        scores = {}
        for side in ("before", "after"):
            endpoint = source["endpoint"]
            score_signature = pilot.digest({
                "version": long_ensemble_experiment.VERSION,
                **endpoint,
                "target": source["target_sha256"],
            })
            score_folder = folder / side / "scores" / score_signature
            score_attempts: dict[str, Any] = {
                "signature": score_signature,
                "attempts": [
                    {
                        "status": "complete",
                        "cpu_seconds": 0.2,
                        "wall_seconds": 0.21,
                        "attempted_channel_calls": 12,
                        "completed_channel_calls": 12,
                    }
                ],
            }
            pilot.write_record(score_folder / "ATTEMPTS.json", score_attempts)
            resources = reporting.endpoint_resources(endpoint["circuit"], endpoint["theta"])
            score: dict[str, Any] = {
                "signature": score_signature,
                "status": "complete",
                "theta": endpoint["theta"],
                "fidelities": dict.fromkeys(reporting.channels.NOISE_IDS, 0.8),
                "logical_counts": resources["physical_counts"],
                "ballarin_counts": resources["ballarin_counts"],
                "cpu_seconds": 0.2,
                "wall_seconds": 0.21,
                "attempts_checksum": pilot.digest(score_attempts),
            }
            pilot.write_record(score_folder / "RESULT.json", score)
            scores[side] = score
        record: dict[str, Any] = {
            "signature": signature,
            "status": "complete",
            "target_id": source["target_id"],
            "replica": 11,
            "label": source["label"],
            "source": source,
            "compilation": compilation,
            **scores,
            "fidelity_delta": dict.fromkeys(reporting.channels.NOISE_IDS, 0.0),
            "mean_noisy_delta": 0.0,
            "compiler_attempts": 1,
            "compiler_completed_attempts": 1,
            "physical_invocation": {
                "cpu_seconds": 1.0,
                "wall_seconds": 1.1,
                "attempts_checksum": pilot.digest(invocation),
            },
            "scoring": {side: long_ensemble_experiment.physical_scoring(folder / side) for side in ("before", "after")},
            "dependencies": state_compiler.artifact_dependencies(folder),
        }
        pilot.write_record(folder / "RESULT.json", record)
        records.append(record)
    pilot.write_record(output / "FAILURES.json", failures)
    state_compiler.write_summary(output, protocol, records)
    return output / "scoreboard.csv"


@pytest.mark.parametrize("failed", [False, True])
def test_state_compiler_adapter_preserves_pairs_failures_and_nested_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, failed: bool
) -> None:
    """All 576 cells remain visible, including both sides of a failed pair."""
    path = state_compiler_fixture(tmp_path, monkeypatch, failed=failed)
    result = reporting.collect([path], tmp_path / "overview")
    provenance = result["sources"]["state-compiler"]
    assert provenance["expected_cells"] == 576
    assert provenance["missing_csv_cells"] == 0
    assert provenance["authenticated_runs_or_cases"] == (23 if failed else 24)
    methods = result["panels"]["state-compiler"]["methods"]
    assert len(methods) == 12
    assert methods["legacy_initializer__before"]["status"] == ("incomplete" if failed else "complete")
    assert methods["legacy_initializer__after"]["status"] == ("incomplete" if failed else "complete")
    rows, metadata = reporting.authenticate_panel(path)
    assert len(metadata["physical_invocations"]) == (23 if failed else 24)
    complete = [row for row in rows if row["status"] == "complete"]
    assert all(row["inclusive_pair_cpu_seconds"] == "1.0" for row in complete)
    assert all(row["compiler_cpu_seconds"] == "0.1" for row in complete)
    assert all(row["evaluation_cpu_seconds"] == "0.2" for row in complete)
    assert {row["one_qubit"] for row in complete} == {"78"}


def test_common_control_adapter_preserves_all_methods_at_original_primary_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Copied and fresh scores share the original cutoff without double-charging source work."""
    path = common_control_fixture(tmp_path, monkeypatch)
    before = path.read_bytes()
    assert reporting.write_common_control_scoreboard(path.parent).read_bytes() == before
    result = reporting.collect([path], tmp_path / "overview")
    assert result["sources"]["common-control"]["expected_cells"] == 336
    methods = result["panels"]["common-control/common_primary"]["methods"]
    assert len(methods) == 7
    assert all(value["status"] == "complete" for value in methods.values())
    rows, _ = reporting.authenticate_panel(path)
    assert {row["refinement_cpu_cutoff"] for row in rows} == {"4.0"}
    assert {row["total_cpu_cutoff"] for row in rows} == {"5.0"}
    assert {row["selected_cpu_seconds"] for row in rows} == {"0.0", "2.0", "3.0", "4.0"}
    assert {row["one_qubit"] for row in rows} == {"78", "150"}
    assert all(
        row["evaluation_cpu_seconds"] == ("0.1" if row["score_origin"] == "fresh_audit" else "0.0") for row in rows
    )
    assert all(row["inclusive_case_audit_cpu_seconds"] == "1.0" for row in rows)
    assert all(
        row["physical_attempted_channel_calls"] == ("12" if row["score_origin"] == "fresh_audit" else "0")
        for row in rows
    )


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "pair",
        "child",
        "score",
        "clock",
        "parameter_count",
        "summary",
        "unrecorded_failure",
        "csv_resource",
        "missing_csv_pair",
    ],
)
def test_state_compiler_adapter_rejects_tampering_and_preserves_missing_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Changed source identities, receipts and resource cells cannot pass via a refreshed summary."""
    path = state_compiler_fixture(tmp_path, monkeypatch)
    protocol = cast("dict[str, Any]", pilot.read_record(path.parent / "PROTOCOL.json"))
    source = protocol["pairs"][0]
    folder = path.parent / state_compiler.pair_key(source)
    record = cast("dict[str, Any]", pilot.read_record(folder / "RESULT.json"))
    if change == "source":
        source_path = next(iter(protocol["input_files"]))
        pilot.write_record(path.parent / source_path, {"changed": True})
    elif change == "pair":
        protocol["pairs"][0]["endpoint"]["theta"][0] += 0.2
        pilot.write_record(path.parent / "PROTOCOL.json", protocol)
    elif change == "child":
        pilot.write_record(folder / "unexpected.json", {"extra": True})
    elif change in {"score", "clock", "parameter_count"}:
        if change == "score":
            record["after"]["fidelities"]["noiseless"] = 0.99
        elif change == "clock":
            record["physical_invocation"]["wall_seconds"] = 12.0
        else:
            record["after"]["theta"] = record["after"]["theta"][:-1]
            score_signature = record["after"]["signature"]
            pilot.write_record(folder / "after/scores" / score_signature / "RESULT.json", record["after"])
            record["dependencies"] = state_compiler.artifact_dependencies(folder)
        pilot.write_record(folder / "RESULT.json", record)
        records = [
            cast("dict[str, Any]", pilot.read_record(path.parent / state_compiler.pair_key(pair) / "RESULT.json"))
            for pair in protocol["pairs"]
        ]
        state_compiler.write_summary(path.parent, protocol, records)
    elif change == "summary":
        summary = cast("dict[str, Any]", pilot.read_record(path.parent / "SUMMARY.json"))
        summary["complete_panel_mean_noisy_delta"] = 0.9
        pilot.write_record(path.parent / "SUMMARY.json", summary)
    elif change == "unrecorded_failure":
        (folder / "RESULT.json").unlink()
    else:
        with path.open(newline="") as stream:
            data = list(csv.DictReader(stream))
        if change == "csv_resource":
            data[0]["parameters"] = "1"
        else:
            data = [row for row in data if (row["target_id"], row["label"]) != (source["target_id"], source["label"])]
        write_rows(path, data)
    if change == "missing_csv_pair":
        result = reporting.collect([path], tmp_path / "overview")
        assert result["sources"]["state-compiler"]["missing_csv_cells"] == 24
        assert result["panels"]["state-compiler"]["methods"]["legacy_initializer__before"]["status"] == "incomplete"
        assert result["panels"]["state-compiler"]["methods"]["legacy_initializer__after"]["status"] == "incomplete"
        assert result["panels"]["state-compiler"]["methods"]["conditional_initializer__after"]["status"] == "complete"
    else:
        with pytest.raises(
            ValueError,
            match=(
                r"State compiler|Cached compiler|Diagnostic|Composition|Initializer|CSV cell|missing compiler|"
                r"Panel benchmark"
            ),
        ):
            reporting.authenticate_panel(path)


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "cutoff",
        "point",
        "primary_score",
        "fresh_score",
        "fresh_counts",
        "charged_counts",
        "unknown_clock",
        "child",
        "summary",
        "missing_case",
        "csv_resource",
        "missing_csv_pair",
    ],
)
def test_common_control_adapter_rejects_tampering_and_preserves_missing_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Primary and newly scored controls remain bound to their original chronological endpoint."""
    path = common_control_fixture(tmp_path, monkeypatch)
    protocol = cast("dict[str, Any]", pilot.read_record(path.parent / "PROTOCOL.json"))
    target = composed_experiment.TARGETS[0]
    folder = path.parent / f"{target}__11"
    record = cast("dict[str, Any]", pilot.read_record(folder / "RESULT.json"))
    if change == "source":
        parent_path = path.parent / protocol["input_panel"]["root"] / f"{target}__11/RESULT.json"
        parent_record = cast("dict[str, Any]", pilot.read_record(parent_path))
        parent_record["signature"] = "changed"
        pilot.write_record(parent_path, parent_record)
    elif change == "child":
        pilot.write_record(folder / "operations/unexpected.json", {"extra": True})
    elif change == "summary":
        summary = cast("dict[str, Any]", pilot.read_record(path.parent / "SUMMARY.json"))
        summary["strongest_control"] = "unchanged"
        pilot.write_record(path.parent / "SUMMARY.json", summary)
    elif change == "missing_case":
        (folder / "RESULT.json").unlink()
    elif change in {"csv_resource", "missing_csv_pair"}:
        with path.open(newline="") as stream:
            data = list(csv.DictReader(stream))
        if change == "csv_resource":
            data[0]["selected_cpu_seconds"] = "0.0"
        else:
            data = [row for row in data if (row["target_id"], row["method"]) != (target, "fixed_mean2_lbfgs")]
        write_rows(path, data)
    else:
        if change == "cutoff":
            record["value"]["refinement_cpu_seconds"] = 5.0
        elif change == "point":
            record["value"]["points"]["fixed_mean2_lbfgs"]["theta"][0] += 1.0
        elif change in {"primary_score", "fresh_score"}:
            method = "structural_mean2_lbfgs" if change == "primary_score" else "fixed_mean2_lbfgs"
            record["value"]["scores"][method]["fidelities"]["noiseless"] = 0.9
        elif change == "fresh_counts":
            record["value"]["scores"]["fixed_mean2_lbfgs"]["parameters"] = 1
        elif change == "charged_counts":
            score = record["value"]["scores"]["fixed_mean2_lbfgs"]
            score["evaluation"]["physical_attempted_channel_calls"] = 0
            point = record["value"]["points"]["fixed_mean2_lbfgs"]
            identity = {"circuit_hash": point["circuit_hash"], "theta": point["theta"]}
            operation_path = folder / "operations" / f"audit_{pilot.digest(identity)}.json"
            pilot.write_record(operation_path, score["evaluation"])
            record["value"]["dependencies"][str(operation_path.relative_to(folder))] = reporting.file_sha256(
                operation_path
            )
        else:
            attempts = cast("dict[str, Any]", pilot.read_record(folder / "RESULT.attempts.json"))
            attempts["attempts"][0]["status"] = "running"
            pilot.write_record(folder / "RESULT.attempts.json", attempts)
            record["attempts_checksum"] = pilot.digest(attempts)
        pilot.write_record(folder / "RESULT.json", record)
        records = [
            cast("dict[str, Any]", pilot.read_record(path.parent / f"{name}__11/RESULT.json"))
            for name in composed_experiment.TARGETS
        ]
        pilot.write_record(
            path.parent / "SUMMARY.json",
            {
                "status": "complete",
                "protocol_checksum": pilot.digest(protocol),
                "cases": [pilot.digest(value) for value in records],
                **common_control_audit.summarize(records, composed_experiment.TARGETS),
            },
        )
    if change == "missing_csv_pair":
        result = reporting.collect([path], tmp_path / "overview")
        assert result["sources"]["common-control"]["missing_csv_cells"] == 12
        methods = result["panels"]["common-control/common_primary"]["methods"]
        assert methods["fixed_mean2_lbfgs [common_primary]"]["status"] == "incomplete"
        assert methods["structural_mean2_lbfgs [common_primary]"]["status"] == "complete"
    else:
        with pytest.raises(ValueError, match=r"[Cc]ommon-control|Composition|Diagnostic|CSV cell|missing"):
            reporting.authenticate_panel(path)


@pytest.mark.parametrize("change", [None, "running", "negative", "nonfinite", "null", "omitted_failed_cost", "wall"])
def test_diagnostic_timer_includes_failed_replay_and_rejects_unknown_cost(tmp_path: Path, change: str | None) -> None:
    """Both CPU and elapsed time include every completed or failed invocation exactly once."""
    entries: list[dict[str, Any]] = [
        {"status": "interrupted", "cpu_seconds": 1.0, "wall_seconds": 1.5},
        {"status": "complete", "cpu_seconds": 2.0, "wall_seconds": 2.5},
    ]
    attempts: dict[str, Any] = {
        "signature": "fixed",
        "attempts": entries,
    }
    timer: dict[str, Any] = {"cpu_seconds": 3.0, "wall_seconds": 4.0}
    if change == "running":
        attempts["attempts"][0]["status"] = "running"
    elif change == "negative":
        attempts["attempts"][0]["cpu_seconds"] = -1.0
    elif change == "nonfinite":
        attempts["attempts"][0]["wall_seconds"] = float("inf")
    elif change == "null":
        attempts["attempts"][0]["cpu_seconds"] = None
    elif change == "omitted_failed_cost":
        timer["cpu_seconds"] = 2.0
    elif change == "wall":
        timer["wall_seconds"] = 2.5
    path = tmp_path / "ATTEMPTS.json"
    if change == "nonfinite":
        path.write_text(json.dumps(attempts))
        with pytest.raises(ValueError, match="Out of range float values"):
            reporting.terminal_attempts(path, timer, "fixed")
        return
    pilot.write_record(path, attempts)
    timer["attempts_checksum"] = pilot.digest(attempts)
    if change is None:
        reporting.terminal_attempts(path, timer, "fixed")
    else:
        with pytest.raises(ValueError, match="Diagnostic attempt"):
            reporting.terminal_attempts(path, timer, "fixed")


def tiny_composition_configs() -> tuple[Any, Any]:
    """Return bounded q2 settings retaining the complete optimizer-role opportunity."""
    return (
        composed_experiment.exact.Config(
            channel_evaluations=20,
            warmup_evaluations=1,
            max_rounds=1,
            shortlist=1,
            repair_evaluations=1,
            protected_initial_gates=0,
            checkpoint_every=1,
        ),
        composed_experiment.stochastic.Config(
            total_updates=8,
            warmup=1,
            rounds=1,
            shortlist=1,
            repair=1,
            paths_per_channel=2,
            protected=0,
            checkpoint_every=1,
            online_warmup=1,
        ),
    )


def compiled_reporting_fixture(root: Path, monkeypatch: pytest.MonkeyPatch, *, failed: bool = False) -> Path:
    """Return actual synthetic q2 receipts for all four development-shaped adapter cases."""
    provenance = cast(
        "dict[str, Any]", pilot.read_record(panel_fixture(root / "custody", monkeypatch).parent / "PROTOCOL.json")
    )
    if failed:
        original_function = compiled_kernel.train_arm

        def failing(
            store: composed_experiment.CompleteStore,
            sampled: composed_experiment.stochastic.Store,
            source: dict[str, Any],
            initial: dict[str, Any],
            original: dict[str, Any],
            method: str,
        ) -> dict[str, Any]:
            if method == compiled_kernel.CANDIDATE:
                msg = "synthetic retained arithmetic failure"
                raise ArithmeticError(msg)
            return original_function(store, sampled, source, initial, original, method)

        monkeypatch.setattr(compiled_kernel, "train_arm", failing)
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("rz", (0,), 0, logical_gate_id=0, native_gate_id=0),
            ParameterizedGate("ry", (0,), 1, logical_gate_id=1, native_gate_id=1),
            ParameterizedGate("rx", (1,), 2, logical_gate_id=2, native_gate_id=2),
            ParameterizedGate("rxx", (1, 0), 3, logical_gate_id=3, native_gate_id=3),
        ],
    )
    source = {"circuit": pilot.circuit_record(circuit), "theta": [0.4, 0.3, -0.2, 0.7], "initializer_cpu_seconds": 2.0}
    target = np.array([0.2j, 0.6, -0.3 + 0.1j, 0.4])
    target /= np.linalg.norm(target)
    exact, sampled = tiny_composition_configs()
    protocol = {
        "version": compiled_driver.VERSION,
        "baseline": compiled_kernel.BASELINE,
        "candidate": compiled_kernel.CANDIDATE,
        "seed_namespace": "compiled_composition_training",
        "primary_cutoff": "minimum four structural final refinement CPU; score all seven at same cutoff",
        "promotion": {"mean_delta_pp": 0.5, "target_wins": 4, "fixed_and_strongest_controls_required": True},
        "source_hashes": provenance["source_hashes"],
        "benchmark_hashes": provenance["benchmark_hashes"],
        "exact_config": asdict(exact),
        "stochastic_config": asdict(sampled),
        "targets": list(compiled_driver.TARGETS),
        "replicas": [11],
        "methods": list(compiled_kernel.METHODS),
        "starts": [{"target_id": name, "replica": 11, "endpoint": source} for name in compiled_driver.TARGETS],
    }
    output = root / "compiled"
    pilot.write_record(output / "PROTOCOL.json", protocol)
    protocol = compiled_kernel.read(output / "PROTOCOL.json")
    records = []
    for index, start in enumerate(protocol["starts"]):
        case = output / f"{start['target_id']}__11"
        order = (*compiled_kernel.TRAINED[index:], *compiled_kernel.TRAINED[:index])
        execution = compiled_kernel.execute(
            case,
            start["endpoint"],
            target,
            exact,
            sampled,
            reporting.runner.seed_for(protocol["seed_namespace"], start["target_id"], 11),
            order=order,
        )
        record = {
            **execution,
            "execution_signature": execution["signature"],
            "execution_checksum": pilot.digest(execution),
            "signature": pilot.digest([protocol, start]),
            "protocol_checksum": pilot.digest(protocol),
            "target_id": start["target_id"],
            "replica": 11,
            "version": compiled_driver.VERSION,
        }
        pilot.write_record(case / "RESULT.json", record)
        records.append(record)
    compiled_driver.summarize(output, protocol, records)
    # Existing composed-input fixtures exercise the complete upstream initializer graph;
    # here only that previously covered boundary is stubbed, never new receipt validation.
    monkeypatch.setattr(reporting, "composed_inputs", lambda _protocol: None)
    monkeypatch.setattr(reporting, "target_hash", lambda _name: hashlib.sha256(target.tobytes()).hexdigest())
    return output / "SCOREBOARD.csv"


def test_compiled_reporting_preserves_raw_counts_and_all_three_physical_strata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Source Ballarin counts remain auditable while canonical resources use actual lowering."""
    path = compiled_reporting_fixture(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in path.parent.rglob("*") if p.is_file()}
    monkeypatch.setattr(compiled_kernel, "execute", lambda *_a, **_k: pytest.fail("reporting executed a kernel"))
    rows, metadata = reporting.authenticate_panel(path)
    assert len(rows) == metadata["expected_cells"] == 1008
    assert {r["comparison_panel"] for r in rows} == {"full_cap", "common_primary", "common_all_trained"}
    assert len({r["source_method"] for r in rows}) == 7
    ballarin = [r for r in rows if r["noise_id"] == "ballarin_coupled"]
    assert any(int(r["one_qubit"]) > int(r["source_one_qubit"]) for r in ballarin)
    assert all(r["count_scope"] == "per-condition materialized physical circuit" for r in rows)
    assert all(p.read_bytes() == content for p, content in before.items())


def best_reporting_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    targets: int = 1,
    completed: int = 2,
    recipe: str = "original",
) -> Path:
    """Return real reduced-budget q2 parent receipts without reading any benchmark target."""
    provenance = cast(
        "dict[str, Any]", pilot.read_record(panel_fixture(root / "custody", monkeypatch).parent / "PROTOCOL.json")
    )
    monkeypatch.setattr(best_candidate, "sources", lambda _recipe: provenance["source_hashes"])
    monkeypatch.setattr(best_candidate.lazy, "sources", lambda: provenance["source_hashes"])
    monkeypatch.setattr(best_candidate.constructor, "constructor_spec", lambda: {"synthetic_constructor": "fixed"})
    monkeypatch.setattr(best_candidate.construction, "environment", lambda: {"runtime": "synthetic"})
    monkeypatch.setattr(
        best_candidate.base.runner, "custody", lambda: {"benchmark_hashes": provenance["benchmark_hashes"]}
    )
    monkeypatch.setattr(best_candidate, "load_target", lambda *_a: pytest.fail("synthetic fixture requested benchmark"))
    for variable in best_candidate.THREAD_VARIABLES:
        monkeypatch.setenv(variable, "1")
    vector = np.array([0.2j, 0.6, -0.3 + 0.1j, 0.4])
    vector /= np.linalg.norm(vector)
    vectors = {f"synthetic_q2_{i}": vector.copy() for i in range(targets)}
    exact, sampled = tiny_composition_configs()
    selected = best_candidate.configuration(recipe, exact, sampled, list(vectors), (11, 29))
    output = root / "confirmation"
    design = {
        "version": best_candidate.VERSION,
        "status": "frozen",
        "phase": "engineering",
        "recipe": recipe,
        "targets": list(vectors),
        "replicas": [11, 29],
        **selected,
        "configuration_sha256": pilot.digest(selected),
        "source_hashes": provenance["source_hashes"],
        "benchmark_hashes": provenance["benchmark_hashes"],
        "environment": best_candidate.construction.environment(),
        "development_evidence": {},
        "disk_plan": best_candidate.disk_plan(root, targets * 2),
    }
    pilot.write_record(output / "DESIGN.json", design)
    parent = best_candidate.prepare(output, vectors=vectors)
    for case in parent["cases"][:completed]:
        best_candidate.run_case(
            {"root": str(output), "parent": parent, "case": case},
            loader=lambda _n, target: vectors[target].copy(),
        )
    best_candidate.summarize(output, parent)
    monkeypatch.setattr(best_candidate, "target_loader", lambda *_a: pytest.fail("reporting loaded target"))
    monkeypatch.setattr(best_candidate, "run_case", lambda *_a, **_k: pytest.fail("reporting ran confirmation"))
    monkeypatch.setattr(best_candidate, "train_case", lambda *_a, **_k: pytest.fail("reporting trained"))
    monkeypatch.setattr(reporting, "target_hash", lambda *_a: pytest.fail("reporting decoded fixture targets"))
    return output / "SCOREBOARD.csv"


def test_best_reporting_authenticates_sealed_parent_without_target_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All roles, replicas, CPU clocks and per-channel counts survive portable read-only reporting."""
    path = best_reporting_fixture(tmp_path, monkeypatch, targets=5, completed=10)
    before = {p: p.read_bytes() for p in path.parent.rglob("*") if p.is_file()}
    rows, metadata = reporting.authenticate_panel(path)
    assert len(rows) == metadata["expected_cells"] == 2520
    assert {r["comparison_panel"] for r in rows} == {"primary", "all_trained_sensitivity", "full_cap"}
    assert {r["status"] for r in rows} == {"complete"}
    assert len({r["source_method"] for r in rows}) == 7
    assert metadata["actual_complete_experiment_cpu_seconds"] == pytest.approx(
        metadata["preparation_cpu_seconds"] + sum(r["cpu_seconds"] for r in metadata["physical_invocations"].values())
    )
    assert all(p.read_bytes() == content for p, content in before.items())


def refresh_compiled_summary(root: Path) -> None:
    """Re-sign outer metadata only, leaving independent numerical child receipts untouched."""
    protocol = compiled_kernel.read(root / "PROTOCOL.json")
    records = []
    for source in protocol["starts"]:
        child = root / f"{source['target_id']}__11"
        execution = compiled_kernel.read(child / "EXECUTION.json")
        execution["dependencies"] = compiled_kernel.artifacts(child)
        for method in compiled_kernel.METHODS:
            arm_path = child / "arms" / f"{method}.json"
            execution["arms"][method] = compiled_kernel.compact_arm(compiled_kernel.read(arm_path), arm_path, child)
        pilot.write_record(child / "EXECUTION.json", execution)
        record = {
            **execution,
            "execution_signature": execution["signature"],
            "execution_checksum": pilot.digest(execution),
            "signature": pilot.digest([protocol, source]),
            "protocol_checksum": pilot.digest(protocol),
            "target_id": source["target_id"],
            "replica": 11,
            "version": compiled_driver.VERSION,
        }
        pilot.write_record(child / "RESULT.json", record)
        records.append(record)
    compiled_driver.summarize(root, protocol, records)


@pytest.mark.parametrize("changed", ["theta", "score", "cost", "curve", "cutoff", "inventory", "summary", "csv"])
def test_compiled_reporting_rejects_resigned_changes_to_physical_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    """New JSON checksums cannot authorize edited endpoints, shared cutoffs, resources or missing files."""
    path = compiled_reporting_fixture(tmp_path, monkeypatch)
    child = path.parent / f"{compiled_driver.TARGETS[0]}__11"
    if changed in {"theta", "score", "cost", "curve"}:
        arm_path = child / "arms" / f"{compiled_kernel.CANDIDATE}.json"
        arm = compiled_kernel.read(arm_path)
        if changed == "theta":
            arm["final"]["theta"][0] += 0.1
        elif changed == "score":
            arm["scores"]["fidelities"]["noiseless"] = 0.0
        elif changed == "cost":
            arm["training_cpu_seconds"] += 1
        else:
            arm["curve"][2]["cpu_seconds"] += 0.01
        pilot.write_record(arm_path, arm)
        refresh_compiled_summary(path.parent)
    elif changed == "cutoff":
        execution = compiled_kernel.read(child / "EXECUTION.json")
        execution["common_primary"]["refinement_cpu_seconds"] += 1
        pilot.write_record(child / "EXECUTION.json", execution)
        refresh_compiled_summary(path.parent)
    elif changed == "inventory":
        next((child / "operations").glob("*.json")).unlink()
    elif changed == "summary":
        summary = compiled_kernel.read(path.parent / "SUMMARY.json")
        summary["analysis"]["promoted"] = not summary["analysis"]["promoted"]
        pilot.write_record(path.parent / "SUMMARY.json", summary)
    else:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["two_qubit"] = str(int(rows[0]["two_qubit"]) + 1)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with pytest.raises((ValueError, FileNotFoundError)):
        reporting.authenticate_panel(path)


def test_compiled_reporting_keeps_failed_arms_and_missing_common_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed candidate cannot remove its cells or create a common-CPU comparison."""
    path = compiled_reporting_fixture(tmp_path, monkeypatch, failed=True)
    rows, metadata = reporting.authenticate_panel(path)
    assert len(rows) == 1008
    assert metadata["result_status"] != "complete"
    candidate = [
        r for r in rows if r["source_method"] == compiled_kernel.CANDIDATE and r["comparison_panel"] == "full_cap"
    ]
    assert len(candidate) == 48
    assert all(r["status"] == "failed" and not r["fidelity"] for r in candidate)
    assert all(not r["fidelity"] for r in rows if r["comparison_panel"] != "full_cap")


def rewrite_best_phase(root: Path, value: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a re-signed synthetic outer phase to test deeper independent receipt binding."""
    result = best_candidate.read(root / "RESULT.json")
    if value is not None:
        result["value"] = value
    ledger = best_candidate.read(root / "ATTEMPTS.json")
    ignored = tuple(root / name for name in ("PHASE.json", "ATTEMPTS.json", "RESULT.json", "COMPLETED.json"))
    ledger["attempts"][-1]["files"] = best_candidate.inventory(root, excluded=ignored)
    pilot.write_record(root / "ATTEMPTS.json", ledger)
    result["attempts_checksum"] = pilot.digest(ledger)
    pilot.write_record(root / "RESULT.json", result)
    pilot.write_record(
        root / "COMPLETED.json",
        {
            "signature": result["signature"],
            "result_checksum": pilot.digest(result),
            "attempts_checksum": pilot.digest(ledger),
        },
    )
    return result


@pytest.mark.parametrize(
    "changed", ["input_cost", "initializer", "score", "point", "arm", "missing", "parent", "summary"]
)
def test_best_reporting_rejects_deep_resigned_endpoint_and_cost_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    """Sealed outer copies must still agree with every independent phase and numerical receipt."""
    path = best_reporting_fixture(tmp_path, monkeypatch)
    parent = best_candidate.read(path.parent / "PROTOCOL.json")
    case = parent["cases"][0]
    child = path.parent / f"{case['target_id']}__{case['replica']}"
    execution = child / "execution"
    physical = best_candidate.read(execution / "RESULT.json")
    value = physical["value"]
    if changed == "input_cost":
        value["phase_costs"]["input"]["wall_seconds"] += 1
    elif changed == "initializer":
        value["initializer"]["rank_two_selected"] = not value["initializer"]["rank_two_selected"]
    elif changed in {"score", "point"}:
        trained = best_candidate.read(execution / "training/RESULT.json")
        arm = trained["value"]["panels"]["primary"]["arms"][best_candidate.CANDIDATE]
        if changed == "score":
            arm["scores"]["fidelities"]["noiseless"] = 0.0
        else:
            arm["point"]["theta"][0] += 0.1
        trained = rewrite_best_phase(execution / "training", trained["value"])
        value.update(trained["value"])
        value["phase_costs"]["training_and_evaluation"]["checksum"] = pilot.digest(trained)
    elif changed == "arm":
        arm_path = execution / "training/work/arms" / f"{best_candidate.CANDIDATE}.json"
        arm = best_candidate.read(arm_path)
        arm["training_cpu_seconds"] += 1
        pilot.write_record(arm_path, arm)
        trained = rewrite_best_phase(execution / "training")
        value["phase_costs"]["training_and_evaluation"]["checksum"] = pilot.digest(trained)
    elif changed == "missing":
        (execution / "initializer/COMPLETED.json").unlink()
    elif changed == "parent":
        manifest = best_candidate.read(child / "CASE.json")
        manifest["training_seed"] += 1
        pilot.write_record(child / "CASE.json", manifest)
    else:
        summary = best_candidate.read(path.parent / "SUMMARY.json")
        summary["actual_complete_experiment_cpu_seconds"] += 1
        pilot.write_record(path.parent / "SUMMARY.json", summary)
    if changed not in {"missing", "parent", "summary"}:
        physical = rewrite_best_phase(execution, value)
        pilot.write_record(child / "RESULT.json", best_candidate.case_record(parent, case, physical))
        best_candidate.summarize(path.parent, parent)
    with pytest.raises((ValueError, FileNotFoundError)):
        reporting.authenticate_panel(path)


@pytest.mark.parametrize("status", ["pending", "failed", "unknown_cost"])
def test_best_reporting_keeps_ten_case_product_and_unavailable_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """No absent replica, failure or unknown kill cost can shrink the five-target panel."""
    path = best_reporting_fixture(tmp_path, monkeypatch, targets=5, completed=0)
    parent = best_candidate.read(path.parent / "PROTOCOL.json")
    case = parent["cases"][0]
    child = path.parent / f"{case['target_id']}__{case['replica']}"
    if status != "pending":
        record = {
            "target_id": case["target_id"],
            "replica": case["replica"],
            "status": "failed",
            "failure": "synthetic failure",
            "timing_complete": status == "failed",
            "physical_cpu_seconds": 0.4 if status == "failed" else None,
            "physical_wall_seconds": 0.5 if status == "failed" else None,
        }
        attempts: dict[str, Any] = {
            "signature": pilot.digest([pilot.digest(parent), case, "complete_physical_case"]),
            "attempts": [{"status": "failed", "cpu_seconds": 0.4, "wall_seconds": 0.5}]
            if status == "failed"
            else [{"status": "running", "cpu_seconds": None, "wall_seconds": None}],
        }
        pilot.write_record(child / "execution/ATTEMPTS.json", attempts)
        pilot.write_record(child / "FAILURE.json", record)
        best_candidate.summarize(path.parent, parent)
    rows, metadata = reporting.authenticate_panel(path)
    assert len(rows) == metadata["expected_cells"] == 2520
    assert all(not row["fidelity"] for row in rows)
    assert metadata["actual_complete_experiment_cpu_seconds"] is None
    assert not metadata["confirmation_analysis"]["promoted"]
    assert metadata["result_status"] == "incomplete_or_failed"


@pytest.mark.parametrize("changed", ["signature", "negative", "boolean", "status", "infinite"])
def test_best_reporting_rejects_invalid_failed_attempt_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    """An outer failure checksum cannot legitimate impossible costs or an unrelated attempt ledger."""
    path = best_reporting_fixture(tmp_path, monkeypatch, completed=0)
    parent = best_candidate.read(path.parent / "PROTOCOL.json")
    case = parent["cases"][0]
    child = path.parent / f"{case['target_id']}__{case['replica']}"
    attempt = {"status": "failed", "cpu_seconds": 0.4, "wall_seconds": 0.5}
    if changed == "negative":
        attempt["cpu_seconds"] = -0.4
    elif changed == "boolean":
        attempt["cpu_seconds"] = True
    elif changed == "status":
        attempt["status"] = "unknown"
    elif changed == "infinite":
        attempt["cpu_seconds"] = float("inf")
    ledger: dict[str, Any] = {
        "signature": "changed"
        if changed == "signature"
        else pilot.digest([pilot.digest(parent), case, "complete_physical_case"]),
        "attempts": [attempt],
    }
    record = {
        "target_id": case["target_id"],
        "replica": case["replica"],
        "status": "failed",
        "failure": "synthetic",
        "timing_complete": True,
        "physical_cpu_seconds": attempt["cpu_seconds"],
        "physical_wall_seconds": 0.5,
    }
    if changed == "infinite":
        # The signed JSON reader itself rejects nonfinite numbers before accounting.
        (child / "execution").mkdir()
        (child / "execution/ATTEMPTS.json").write_text(json.dumps({**ledger, "content_checksum": "invalid"}))
        record["physical_cpu_seconds"] = 0.4
    else:
        pilot.write_record(child / "execution/ATTEMPTS.json", ledger)
    pilot.write_record(child / "FAILURE.json", record)
    best_candidate.summarize(path.parent, parent)
    with pytest.raises(ValueError, match=r"attempt identity|Out of range float"):
        reporting.authenticate_panel(path)


@pytest.mark.parametrize("kind", ["compiled", "confirmation"])
def test_new_reporting_adapters_keep_missing_csv_cells_in_their_own_strata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """A missing row invalidates its method without pooling incomparable CPU endpoints."""
    path = (
        compiled_reporting_fixture(tmp_path, monkeypatch)
        if kind == "compiled"
        else best_reporting_fixture(tmp_path, monkeypatch)
    )
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    removed = rows.pop(0)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = reporting.collect([path], tmp_path / "report")
    assert len(result["panels"]) == 3
    assert result["sources"][path.parent.name]["missing_csv_cells"] == 1
    assert result["rows_including_missing"] == len(rows) + 1
    name = f"{removed['method']} [{removed['panel']}]"
    affected = result["panels"][f"{path.parent.name}/{removed['panel']}"]["methods"][name]
    assert affected["status"] == "incomplete"


def test_best_reporting_compiled_recipe_reuses_the_same_physical_kernel_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unused engineering alternative preserves both compiler clocks and all-role receipts."""
    path = best_reporting_fixture(tmp_path, monkeypatch, recipe="state_preparation_compiled")
    rows, metadata = reporting.authenticate_panel(path)
    assert metadata["expected_cells"] == len(rows) == 504
    assert all(row["status"] == "complete" for row in rows)
