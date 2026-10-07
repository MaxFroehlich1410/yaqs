# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Six-arm q2 integration and fresh-construction custody without development runs."""

from __future__ import annotations

import copy
import csv
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import initializer_experiment as experiment
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate


def source(n: int = 2) -> dict[str, Any]:
    """Return a small physical source whose known construction cost exposes availability gaps."""
    circuit = ParameterizedCircuit(n, [ParameterizedGate("ry", (q,), q) for q in range(n)])
    return {
        "circuit": pilot.circuit_record(circuit),
        "theta": [0.2] * n,
        "target_id": experiment.TARGETS[0],
        "replica": 11,
        "construction_cpu_seconds": 100.0,
        "timing_complete": True,
    }


def target() -> np.ndarray:
    """Return a normalized complex q2 engineering target."""
    vector = np.array([0.1 + 0.2j, 0.7, -0.3j, 0.4 - 0.2j])
    return vector / np.linalg.norm(vector)


@pytest.fixture
def mocked_custody(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate target/source custody to temporary test artifacts."""
    monkeypatch.setattr(experiment, "sources", lambda: {"numerical": "fixed"})
    monkeypatch.setattr(experiment.construction, "environment", lambda: {"runtime": "fixed"})
    monkeypatch.setattr(experiment.runner, "custody", lambda: {"benchmark_hashes": {"immutable": "fixed"}})


def temporary_sources(root: Path) -> list[dict[str, Any]]:
    """Return four source identities without loading or simulating benchmark targets."""
    path = root / "fresh.json"
    pilot.write_record(path, {"fresh": True})
    return [
        {**source(), "target_id": name, "files": {str(path): pilot.digest({"fresh": True})}}
        for name in experiment.TARGETS
    ]


def test_six_arm_small_integration_uses_same_objective_and_deploys_all_profiles(tmp_path: Path) -> None:
    """Both policies receive unchanged, Adam and L-BFGS arms with real q2 exact scores."""
    result = experiment.execute(
        tmp_path,
        source(),
        target(),
        "engineering",
        experiment.initializer_training.Config(joint_cap=3, checkpoint_every=1, history=2),
    )
    assert result["status"] == "complete"
    assert set(result["arms"]) == set(experiment.METHODS)
    assert result["rank_two_selected"]
    for name, arm in result["arms"].items():
        assert set(arm["scores"]["fidelities"]) == set(experiment.compilation_experiment.channels.NOISE_IDS)
        assert arm["individual_channel_gradient_calls"] <= 6
        assert arm["total_cpu_seconds"] == pytest.approx(arm["initializer_cpu_seconds"] + arm["refinement_cpu_seconds"])
        if name.endswith("adam"):
            assert arm["individual_channel_gradient_calls"] == 6
    total = result["common_total_cpu"]
    assert total["status"] == "complete"
    assert total["common_cpu_seconds"] < 100
    assert all(p["arms"]["legacy_adam"]["status"] == "initializer_not_yet_available" for p in total["points"])
    assert not any(p["all_available"] for p in total["points"])
    refinement = result["common_refinement_cpu"]
    assert refinement["clock"] == "refinement_only_including_deployment"
    assert all(p["all_available"] for p in refinement["points"])
    assert "optimizer" not in result
    assert "curve" not in result["arms"]["legacy_adam"]


@pytest.mark.usefixtures("mocked_custody")
def test_preparation_requires_fresh_sources_and_never_builds_or_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preparation freezes source costs and all four targets without running the new policy."""
    starts = temporary_sources(tmp_path)
    monkeypatch.setattr(experiment, "fresh_sources", lambda _root: starts)
    monkeypatch.setattr(
        experiment.initializer_policy, "build_initializers", lambda *_a: pytest.fail("prepare synthesized")
    )
    monkeypatch.setattr(experiment, "score", lambda *_a: pytest.fail("prepare scored"))
    root = tmp_path / "prepared"
    protocol = experiment.prepare(root, tmp_path / "fresh_construction")
    assert protocol["targets"] == list(experiment.TARGETS)
    assert protocol["config"] == asdict(experiment.initializer_training.Config())
    assert list(root.iterdir()) == [root / "PROTOCOL.json"]
    before = (root / "PROTOCOL.json").read_bytes()
    experiment.prepare(root, tmp_path / "fresh_construction")
    assert (root / "PROTOCOL.json").read_bytes() == before
    starts[0]["construction_cpu_seconds"] = None
    with pytest.raises(ValueError, match="measured construction cost"):
        experiment.verify(protocol)


@pytest.mark.usefixtures("mocked_custody")
def test_frozen_panel_source_and_code_changes_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Haar, fixed methods, exact CPU provenance and numerical hashes remain mandatory."""
    starts = temporary_sources(tmp_path)
    monkeypatch.setattr(experiment, "fresh_sources", lambda _root: starts)
    protocol = experiment.prepare(tmp_path / "prepared", tmp_path)
    changed = copy.deepcopy(protocol)
    changed["starts"].pop()
    with pytest.raises(ValueError, match="protocol or numerical custody"):
        experiment.verify(changed)
    monkeypatch.setattr(experiment, "sources", lambda: {"numerical": "changed"})
    with pytest.raises(ValueError, match="protocol or numerical custody"):
        experiment.verify(protocol)
    monkeypatch.setattr(experiment, "sources", lambda: {"numerical": "fixed"})
    pilot.write_record(tmp_path / "fresh.json", {"fresh": False})
    with pytest.raises(ValueError, match="source changed"):
        experiment.verify(protocol)


def test_total_cpu_selection_has_no_free_initializer_or_score_selection() -> None:
    """A late, worse checkpoint wins availability order; a missing initializer stays absent."""
    curve = [{"refinement_cpu_seconds": 0.0, "fidelity": 0.9}, {"refinement_cpu_seconds": 2.0, "fidelity": 0.8}]
    assert experiment.select_point(curve, 5.0, 4.999) is None
    assert experiment.select_point(curve, 5.0, 5.0) is curve[0]
    assert experiment.select_point(curve, 5.0, 7.0) is curve[1]


def test_initializer_interruption_charges_shared_overhead_and_reuses_frozen_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted policy build is reproducible and its measured work is not erased."""
    original = experiment.initializer_policy.build_initializers

    def interrupt(*_a: object) -> dict[str, Any]:
        raise KeyboardInterrupt

    monkeypatch.setattr(experiment.initializer_policy, "build_initializers", interrupt)
    with pytest.raises(KeyboardInterrupt):
        experiment.build(tmp_path, source(), target(), "fixed")
    monkeypatch.setattr(experiment.initializer_policy, "build_initializers", original)
    result = experiment.build(tmp_path, source(), target(), "fixed")
    attempts = cast("dict[str, Any]", pilot.read_record(tmp_path / "INITIALIZER_ATTEMPTS.json"))["attempts"]
    assert [a["status"] for a in attempts] == ["interrupted", "complete"]
    assert result["shared_interrupted_setup_cpu_seconds"] == pytest.approx(attempts[0]["cpu_seconds"])
    assert result["conditional"]["initializer_cpu_seconds"] == pytest.approx(
        result["costs"]["rank_two_probe_cpu_seconds"] + attempts[0]["cpu_seconds"]
    )
    monkeypatch.setattr(
        experiment.initializer_policy, "build_initializers", lambda *_a: pytest.fail("cached initializer rebuilt")
    )
    assert pilot.digest(experiment.build(tmp_path, source(), target(), "fixed")) == pilot.digest(result)


def test_numeric_arm_failure_remains_terminal_with_every_arm_and_score_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed arms remain explicit on resume and are not rescreened for a better endpoint."""
    original = experiment.initializer_training.train
    calls = []

    def fail_first(
        root: Path,
        start: dict[str, Any],
        vector: np.ndarray,
        method: str,
        config: experiment.initializer_training.Config,
    ) -> dict[str, Any]:
        calls.append(root.name)
        if root.name == "legacy_lbfgs":
            msg = "fixed numerical failure"
            raise FloatingPointError(msg)
        return original(root, start, vector, method, config)

    monkeypatch.setattr(experiment.initializer_training, "train", fail_first)
    config = experiment.initializer_training.Config(joint_cap=2, checkpoint_every=1)
    first = experiment.execute(tmp_path, source(), target(), "fixed", config)
    assert first["status"] == "partial_failure"
    assert first["arms"]["legacy_lbfgs"]["status"] == "failed"
    second = experiment.execute(tmp_path, source(), target(), "fixed", config)
    assert calls.count("legacy_lbfgs") == 1
    assert second["arms"]["legacy_lbfgs"] == first["arms"]["legacy_lbfgs"]
    summary = experiment.write_summary(tmp_path, {"fixture": True}, [second])
    assert all(value is None for value in summary["equal_target_all_noisy_means"].values())
    with (tmp_path / "SCOREBOARD.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 * 6 * 12
    assert len([r for r in rows if r["status"] == "failed"]) == 12


def fake_fresh_construction(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Create authenticated source metadata only, without a q6 vector or optimizer call."""
    cases: list[dict[str, Any]] = [
        {"target_id": name, "replica": 11, "root": str(root / f"{name}__11")} for name in experiment.TARGETS
    ]
    protocol: dict[str, Any] = {"cases": cases}
    checksums = {}
    gates = [ParameterizedGate("ry", (i % 6,), i) for i in range(78)]
    gates.extend(ParameterizedGate("rzz", (i % 5, i % 5 + 1), i + 78) for i in range(30))
    circuit = ParameterizedCircuit(6, gates)
    for case in cases:
        case_root = Path(case["root"])
        built = {
            **source(),
            "status": "complete",
            "target_id": case["target_id"],
            "circuit": pilot.circuit_record(circuit),
            "theta": [0.1] * 108,
            "physical_counts": experiment.runner.counts(circuit),
            "parameters": 108,
            "updates": 570,
            "archive_reproduction": {"matches": True},
        }
        audit: dict[str, Any] = {
            "status": "complete",
            "matches": True,
            "construction_cpu_seconds": 100.0,
            "construction_checksum": pilot.digest(built),
            "signature": pilot.digest({"protocol": protocol, "case": case}),
        }
        pilot.write_record(case_root / "PROTOCOL.json", {"target_id": case["target_id"]})
        pilot.write_record(case_root / "RESULT.json", built)
        pilot.write_record(case_root / "AUDIT.json", audit)
        checksums[case["target_id"]] = pilot.digest(audit)
    pilot.write_record(root / "BATCH_PROTOCOL.json", protocol)
    pilot.write_record(
        root / "SUMMARY.json",
        {
            "status": "complete",
            "all_archive_reproductions_match": True,
            "protocol_checksum": pilot.digest(protocol),
            "case_checksums": checksums,
        },
    )
    monkeypatch.setattr(experiment.construction_experiment, "verify_batch", lambda _p: None)
    monkeypatch.setattr(experiment.construction, "reproduce", lambda path, _m: pilot.read_record(path / "RESULT.json"))


def test_fresh_construction_authentication_rejects_missing_or_unaudited_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete matching, measured fresh source is required before scientific freezing."""
    fake_fresh_construction(tmp_path, monkeypatch)
    starts = experiment.fresh_sources(tmp_path)
    assert len(starts) == 4
    assert all(s["construction_cpu_seconds"] == 100 for s in starts)
    path = tmp_path / "SUMMARY.json"
    summary = cast("dict[str, Any]", pilot.read_record(path))
    summary["all_archive_reproductions_match"] = False
    pilot.write_record(path, summary)
    with pytest.raises(ValueError, match="complete fresh construction reproduction"):
        experiment.fresh_sources(tmp_path)
    summary["all_archive_reproductions_match"] = True
    pilot.write_record(path, summary)
    (tmp_path / f"{experiment.TARGETS[0]}__11/RESULT.json").unlink()
    monkeypatch.setattr(experiment.construction, "reproduce", lambda *_a: pytest.fail("prepare launched construction"))
    with pytest.raises(ValueError, match="must exist before"):
        experiment.fresh_sources(tmp_path)


@pytest.mark.usefixtures("mocked_custody")
def test_cached_case_authenticates_every_compiled_checkpoint_and_optimizer_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cached source validity does not hide an edited deployment or optimizer checkpoint."""
    starts = temporary_sources(tmp_path)
    monkeypatch.setattr(experiment, "fresh_sources", lambda _root: starts)
    protocol = experiment.prepare(tmp_path / "prepared", tmp_path)
    monkeypatch.setattr(experiment, "load_target", lambda *_a: SimpleNamespace(state_vector_copy=target))
    original = experiment.execute

    def bounded(
        root: Path,
        start: dict[str, Any],
        vector: np.ndarray,
        signature: str,
        _config: experiment.initializer_training.Config,
    ) -> dict[str, Any]:
        return original(
            root, start, vector, signature, experiment.initializer_training.Config(joint_cap=2, checkpoint_every=1)
        )

    monkeypatch.setattr(experiment, "execute", bounded)
    root = tmp_path / "case"
    job = {"protocol": protocol, "source": starts[0], "root": str(root)}
    record = experiment.run_case(job)
    monkeypatch.setattr(experiment, "execute", lambda *_a: pytest.fail("cached case reran"))
    assert pilot.digest(experiment.run_case(job)) == pilot.digest(record)
    path = root / "legacy_adam/TRAINING.json"
    stored = cast("dict[str, Any]", pilot.read_record(path))
    stored["training_cpu_seconds"] += 1
    pilot.write_record(path, stored)
    with pytest.raises(ValueError, match="checkpoint, score or setup artifact changed"):
        experiment.run_case(job)


def test_geometric_fallback_gives_both_optimizers_identical_legacy_training_starts(tmp_path: Path) -> None:
    """On fallback the policy changes only charged probe work, not the optimizer opportunity."""
    vector = np.zeros(16, dtype=np.complex128)
    vector[[0, 5, 10, 15]] = 0.5
    result = experiment.execute(
        tmp_path,
        source(4),
        vector,
        "synthetic_fallback",
        experiment.initializer_training.Config(joint_cap=2, checkpoint_every=1),
    )
    assert not result["rank_two_selected"]
    for method in ("lbfgs", "adam"):
        legacy = pilot.read_record(tmp_path / f"legacy_{method}/TRAINING.json")
        conditional = pilot.read_record(tmp_path / f"conditional_{method}/TRAINING.json")
        assert legacy["optimizer"] == conditional["optimizer"]
        left, right = result["arms"][f"legacy_{method}"], result["arms"][f"conditional_{method}"]
        assert left["scores"]["fidelities"] == right["scores"]["fidelities"]
        assert right["initializer_cpu_seconds"] > left["initializer_cpu_seconds"]


def test_score_cache_is_bound_to_the_target_vector(tmp_path: Path) -> None:
    """Identical compiled circuits do not share an evaluation across different targets."""
    first = experiment.score(tmp_path, source(), target())
    second = experiment.score(tmp_path, source(), np.array([1, 0, 0, 0], dtype=np.complex128))
    assert first["signature"] != second["signature"]
    assert len(list((tmp_path / "scores").iterdir())) == 2
