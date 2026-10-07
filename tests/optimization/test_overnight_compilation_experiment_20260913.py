# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Authenticated prepare-only sources and q2 compilation comparison audits."""

from __future__ import annotations

import copy
import csv
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import compilation_experiment as experiment
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def small_source() -> dict[str, Any]:
    """Return physical preparation with one identity and one redundant rotation."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), 0, native_gate_id="first"),
            ParameterizedGate("rx", (1,), 1),
            ParameterizedGate("ry", (0,), 2),
            ParameterizedGate("rzz", (0, 1), 3),
        ],
    )
    return {
        "target_id": experiment.TARGETS[0],
        "replica": 11,
        "label": "original108",
        "circuit": pilot.circuit_record(circuit),
        "theta": [0.2, 0.0, 0.3, -0.4],
        "origin_checksum": "engineering",
        "source_costs": {"construction_cpu_seconds": 0.125, "refinement_cpu_seconds": 0.375},
    }


def small_target() -> np.ndarray:
    """Return the original ideal state so physical noise has a clear cost."""
    source = small_source()
    return experiment.ideal_state(experiment.restore_source(source["circuit"]), np.asarray(source["theta"]))


@pytest.fixture
def temporary_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restrict all numerical work to q2 while isolating custody from real files."""
    monkeypatch.setattr(experiment.runner, "custody", lambda: {"benchmark_hashes": {"immutable": "fixed"}})
    monkeypatch.setattr(experiment, "numerical_sources", lambda: {"numerical": "fixed"})
    monkeypatch.setattr(experiment, "load_target", lambda *_args: SimpleNamespace(state_vector_copy=small_target))


def fake_artifacts(root: Path) -> None:
    """Create complete source envelopes without constructing or simulating q6."""
    circuit = pilot.circuit_record(ParameterizedCircuit(6, [ParameterizedGate("ry", (i % 6,), i) for i in range(108)]))
    candidate = {"circuit": circuit, "theta": [0.1] * 108}
    starts = [{"target_id": t, "replica": 11, "candidate": candidate} for t in experiment.TARGETS]
    pilot.write_record(root / "starts.json", {"starts": starts})
    methods = {
        "rank2_unchanged": {"method": "unchanged", "iterations": 0},
        "rank2_exact_adam_600": {"method": "adam", "iterations": 600, "rate": 0.02, "checkpoint_every": 1},
    }
    synthesis_protocol: dict[str, Any] = {
        "spec": {"phase": "development", "methods": methods},
        "benchmark_hashes": {"immutable": "fixed"},
    }
    structural_protocol: dict[str, Any] = {
        "phase": "development",
        "starts": starts,
        "config": {"full_evaluations": 1500, "repair_evaluations": 25},
        "benchmark_hashes": {"immutable": "fixed"},
    }
    pilot.write_record(root / "synthesis-wave1/PROTOCOL.json", synthesis_protocol)
    pilot.write_record(root / "structural_lbfgs25/PROTOCOL.json", structural_protocol)
    for source in starts:
        target = source["target_id"]
        case = root / "synthesis-wave1" / f"{target}__11"
        construction: dict[str, Any] = {
            "status": "complete",
            "signature": pilot.digest({"protocol": synthesis_protocol, "target_id": target, "replica": 11}),
            "candidate": {**candidate, "construction_cpu_seconds": 0.25},
        }
        pilot.write_record(case / "CONSTRUCTION.json", construction)
        for label, config in methods.items():
            job = {
                "phase": "development",
                "target_id": target,
                "replica": 11,
                "label": label,
                "config": config,
                "protocol_checksum": pilot.digest(synthesis_protocol),
                "construction_checksum": pilot.digest(construction),
                "circuit": circuit,
                "initial": candidate["theta"],
                "path": str(case / f"{label}.json"),
            }
            record: dict[str, Any] = {
                "status": "complete",
                "signature": pilot.digest({k: v for k, v in job.items() if k != "path"}),
                "job": job,
                "state": {"theta": [0.2] * 108, "training_cpu_seconds": 2.0},
                "curve": [{"deliberately_large_history": "must not propagate"}],
            }
            pilot.write_record(case / f"{label}.json", record)
        pilot.write_record(
            root / "structural_lbfgs25" / f"{target}__11/RESULT.json",
            {
                "status": "complete",
                "signature": pilot.digest({"protocol": structural_protocol, "source": source}),
                "arms": {"structural_lbfgs": {**candidate, "theta": [0.3] * 108, "cpu_seconds": 3.0}},
            },
        )


def small_job(root: Path) -> dict[str, Any]:
    """Return an authenticated full-panel envelope whose selected case is q2."""
    source_path = root / "source.txt"
    source_path.write_text("frozen source\n")
    sources = []
    for target in experiment.TARGETS:
        for label in experiment.LABELS:
            source = small_source()
            source.update({
                "target_id": target,
                "label": label,
                "file_hashes": {str(source_path): experiment.file_hash(source_path)},
            })
            sources.append(source)
    protocol = {
        "version": experiment.VERSION,
        "phase": "development",
        "targets": list(experiment.TARGETS),
        "replicas": [11],
        "labels": list(experiment.LABELS),
        "compiler_version": experiment.compilation.VERSION,
        "benchmark_hashes": {"immutable": "fixed"},
        "source_hashes": {"numerical": "fixed"},
        "sources": sources,
    }
    return {"protocol": protocol, "source": sources[0], "path": str(root / "case.json")}


def test_compare_has_full_physical_preparation_all_profiles_and_source_immutability() -> None:
    """Q2 comparison evaluates the identical endpoint before and after noise removal."""
    source = small_source()
    before = copy.deepcopy(source)
    result = experiment.compare_endpoint(source, small_target())
    assert source == before
    assert result["status"] == "complete"
    assert result["ideal_overlap"] == pytest.approx(1, abs=1e-12)
    assert result["ideal_state_error"] < 1e-12
    assert result["compilation"]["removed_noise_windows"] == 2
    assert result["compilation"]["output_mapping"][0]["source_native_gate_ids"] == ["first", None]
    original, compiled = result["arms"]["uncompiled"], result["arms"]["compiled"]
    assert set(original["fidelities"]) == set(compiled["fidelities"]) == set(experiment.channels.NOISE_IDS)
    assert original["physical_counts"]["one_qubit"] == 3
    assert compiled["physical_counts"]["one_qubit"] == 1
    assert original["physical_counts"]["two_qubit"] == compiled["physical_counts"]["two_qubit"] == 1
    assert compiled["fidelities"]["depolarizing_1s_all"] > original["fidelities"]["depolarizing_1s_all"]
    assert result["compilation"]["compilation_cpu_seconds"] > 0
    assert all(arm["evaluation_cpu_seconds"] > 0 for arm in result["arms"].values())


@pytest.mark.usefixtures("temporary_runtime")
def test_prepare_authenticates_all_sources_without_science_or_history_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preparation freezes sixteen final endpoints, retains Haar and never scores."""
    fake_artifacts(tmp_path)
    monkeypatch.setattr(experiment.runner, "OUTPUT", tmp_path)
    monkeypatch.setattr(experiment.compilation, "compile_endpoint", lambda *_a: pytest.fail("prepare compiled"))
    monkeypatch.setattr(experiment.channels, "evaluate", lambda *_a: pytest.fail("prepare evaluated"))
    output = tmp_path / "prepared"
    assert experiment.main(["--prepare-only", "--output", str(output)]) == 0
    protocol = cast("dict[str, Any]", pilot.read_record(output / "PROTOCOL.json"))
    assert len(protocol["sources"]) == 16
    assert {s["target_id"] for s in protocol["sources"]} == set(experiment.TARGETS)
    for source in protocol["sources"]:
        experiment.verify_protocol(protocol, source)
        assert "curve" not in source
        assert "state" not in source
        expected = {"original108": 0.1, "rank2_unchanged": 0.2, "rank2_exact_adam_600": 0.2}
        assert source["theta"] == [expected.get(source["label"], 0.3)] * 108
        if source["label"] == "original108":
            assert source["source_costs"]["construction_cpu_seconds"] is None
    assert [p.name for p in output.iterdir()] == ["PROTOCOL.json"]
    before = (output / "PROTOCOL.json").read_bytes()
    experiment.freeze_protocol(output)
    assert (output / "PROTOCOL.json").read_bytes() == before


@pytest.mark.usefixtures("temporary_runtime")
@pytest.mark.parametrize("field", ["signature", "initial", "config", "protocol_checksum", "construction_checksum"])
def test_authenticated_rank_endpoint_cannot_be_relabelled_or_reconfigured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Even a valid envelope cannot override the source's linked construction protocol."""
    fake_artifacts(tmp_path)
    monkeypatch.setattr(experiment.runner, "OUTPUT", tmp_path)
    path = tmp_path / "synthesis-wave1" / f"{experiment.TARGETS[0]}__11/rank2_exact_adam_600.json"
    record = cast("dict[str, Any]", pilot.read_record(path))
    if field == "signature":
        record[field] = "changed"
    else:
        record["job"][field] = {"method": "adam", "iterations": 599} if field == "config" else "changed"
        record["signature"] = pilot.digest({k: v for k, v in record["job"].items() if k != "path"})
    pilot.write_record(path, record)
    with pytest.raises(ValueError, match="Rank-two endpoint identity"):
        experiment.load_sources()


@pytest.mark.usefixtures("temporary_runtime")
def test_cached_result_is_idempotent_but_requires_current_source_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached endpoint cannot bypass source or numerical custody validation."""
    job = small_job(tmp_path)
    result = experiment.execute_case(job)
    assert result["status"] == "complete"
    before = (tmp_path / "case.json").read_bytes()
    monkeypatch.setattr(experiment, "compare_endpoint", lambda *_a: pytest.fail("cached case recomputed"))
    assert pilot.digest(experiment.execute_case(job)) == pilot.digest(result)
    assert (tmp_path / "case.json").read_bytes() == before
    monkeypatch.setattr(experiment, "numerical_sources", lambda: {"numerical": "changed"})
    with pytest.raises(ValueError, match="custody or numerical identity"):
        experiment.execute_case(job)
    monkeypatch.setattr(experiment, "numerical_sources", lambda: {"numerical": "fixed"})
    (tmp_path / "source.txt").write_text("mutated source\n")
    with pytest.raises(ValueError, match="input artifact changed"):
        experiment.execute_case(job)
    assert (tmp_path / "case.json").read_bytes() == before


@pytest.mark.usefixtures("temporary_runtime")
def test_independent_ideal_audit_rejects_corruption_and_retains_all_failure_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An incorrect compiler endpoint fails visibly instead of producing noisy scores."""
    original_compile = experiment.compilation.compile_endpoint

    def corrupt(circuit: ParameterizedCircuit, theta: np.ndarray) -> dict[str, Any]:
        result = original_compile(circuit, theta)
        result["theta"][0] += 0.1
        return result

    monkeypatch.setattr(experiment.compilation, "compile_endpoint", corrupt)
    monkeypatch.setattr(experiment.channels, "evaluate", lambda *_a: pytest.fail("invalid circuit scored"))
    job = small_job(tmp_path)
    result = experiment.execute_case(job)
    assert result["status"] == "failed"
    assert "changed the independently prepared ideal state" in result["failure"]
    experiment.write_scoreboard(tmp_path, [result])
    with (tmp_path / "SCOREBOARD.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 24
    assert all(row["status"] == "failed" and not row["fidelity"] for row in rows)
    assert all(not row["paired_fidelity_change"] for row in rows)
    assert {row["noise_id"] for row in rows} == set(experiment.channels.NOISE_IDS)
    assert pilot.digest(experiment.execute_case(job)) == pilot.digest(result)


@pytest.mark.usefixtures("temporary_runtime")
def test_scoreboard_reports_paired_changes_and_separate_source_and_compile_costs(tmp_path: Path) -> None:
    """Comparison rows use only their own untouched endpoint and preserve unknown costs."""
    result = experiment.execute_case(small_job(tmp_path))
    experiment.write_scoreboard(tmp_path, [result])
    with (tmp_path / "SCOREBOARD.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 24
    for row in rows:
        baseline = result["arms"]["uncompiled"]["fidelities"][row["noise_id"]]
        assert float(row["paired_fidelity_change"]) == pytest.approx(float(row["fidelity"]) - baseline)
        assert float(row["source_construction_cpu_seconds"]) == pytest.approx(0.125)
        assert float(row["source_refinement_cpu_seconds"]) == pytest.approx(0.375)
        assert float(row["compilation_cpu_seconds"]) >= 0
    assert all(float(row["paired_fidelity_change"]) == pytest.approx(0) for row in rows if row["arm"] == "uncompiled")


@pytest.mark.parametrize("unsupported", [{"data_map": "nonserializable"}, {"fixed_params": [0.2]}])
def test_restore_rejects_metadata_that_legacy_restore_would_discard(unsupported: dict[str, Any]) -> None:
    """Serialized data maps and non-scalar parameters must never disappear silently."""
    source = small_source()["circuit"]
    source["gates"][0].update(unsupported)
    with pytest.raises(ValueError, match="no data maps"):
        experiment.restore_source(source)


@pytest.mark.usefixtures("temporary_runtime")
def test_partial_panel_and_changed_frozen_source_rejected(tmp_path: Path) -> None:
    """Neither unfavorable target removal nor a mutated endpoint fits the frozen panel."""
    job = small_job(tmp_path)
    changed = copy.deepcopy(job["source"])
    changed["theta"][0] += 0.01
    with pytest.raises(ValueError, match="absent from the frozen panel"):
        experiment.verify_protocol(job["protocol"], changed)
    job["protocol"]["sources"].pop()
    with pytest.raises(ValueError, match="custody or numerical identity"):
        experiment.verify_protocol(job["protocol"])


@pytest.mark.usefixtures("temporary_runtime")
def test_nonfinite_score_becomes_serializable_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A nonfinite evaluation cannot escape into a misleading successful artifact."""
    monkeypatch.setattr(experiment.channels, "evaluate", lambda *_a: float("nan"))
    result = experiment.execute_case(small_job(tmp_path))
    assert result["status"] == "failed"
    assert "nonfinite or invalid fidelity" in result["failure"]
    assert pilot.digest(pilot.read_record(tmp_path / "case.json")) == pilot.digest(result)


@pytest.mark.usefixtures("temporary_runtime")
def test_changed_frozen_preparation_fails_without_overwriting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A later configuration change requires a distinct protocol and output root."""
    fake_artifacts(tmp_path)
    monkeypatch.setattr(experiment.runner, "OUTPUT", tmp_path)
    output = tmp_path / "prepared"
    experiment.freeze_protocol(output)
    original = (output / "PROTOCOL.json").read_bytes()
    monkeypatch.setattr(experiment, "numerical_sources", lambda: {"numerical": "changed"})
    with pytest.raises(ValueError, match="use a new output root"):
        experiment.freeze_protocol(output)
    assert (output / "PROTOCOL.json").read_bytes() == original


@pytest.mark.usefixtures("temporary_runtime")
def test_prior_protocol_must_use_the_identical_immutable_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Source record checksums do not permit importing a different benchmark run."""
    fake_artifacts(tmp_path)
    monkeypatch.setattr(experiment.runner, "OUTPUT", tmp_path)
    path = tmp_path / "synthesis-wave1/PROTOCOL.json"
    protocol = cast("dict[str, Any]", pilot.read_record(path))
    protocol["benchmark_hashes"] = {"immutable": "changed"}
    pilot.write_record(path, protocol)
    with pytest.raises(ValueError, match="unchanged benchmark"):
        experiment.load_sources()
