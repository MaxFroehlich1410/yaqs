# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Synthetic-only parity/resume tests and metadata-only parent confirmation gates."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import confirmation_construction as wrapper
from experiments.overnight_state_preparation_20260913_0108 import construction

if TYPE_CHECKING:
    from pathlib import Path


def config() -> construction.Config:
    """Return a reduced synthetic schedule with all original construction phases."""
    return construction.Config(
        mode="engineering",
        depths=2,
        source_updates=4,
        online_updates=1,
        pruning_rounds=1,
        repair_updates=2,
        final_updates=2,
        checkpoint_every=2,
    )


def target() -> np.ndarray:
    """Return a complex synthetic q2 vector independent of every benchmark target."""
    rng = np.random.default_rng(1901)
    state = rng.normal(size=4) + 1j * rng.normal(size=4)
    return state / np.linalg.norm(state)


@pytest.fixture
def custody(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use temporary custody metadata while retaining actual small-system MPS numerics."""
    source_path = tmp_path / "kernel.py"
    source_path.write_text("# artificial source identity\n")
    snapshot = {"kernel.py": hashlib.sha256(source_path.read_bytes()).hexdigest()}
    monkeypatch.setattr(wrapper, "REPO", tmp_path)
    monkeypatch.setattr(wrapper, "sources", snapshot.copy)
    monkeypatch.setattr(construction, "sources", snapshot.copy)
    monkeypatch.setattr(construction, "environment", lambda: {"numerical": "engineering"})
    monkeypatch.setattr(wrapper.runner, "custody", lambda: {"benchmark_hashes": {"immutable": "fixed"}})


def parent(tmp_path: Path) -> tuple[Path, dict[str, Any], np.ndarray]:
    """Create a complete artificial parent without a held-out vector or simulation.

    Returns:
        Parent path, metadata and a synthetic q6 vector used only for hashing.
    """
    vector = np.ones(64, dtype=np.complex128) / 8
    record: dict[str, Any] = {
        "schema": wrapper.PARENT_SCHEMA,
        "version": "engineering_metadata_only",
        "status": "frozen",
        "phase": "confirmation",
        "targets": list(wrapper.TARGETS),
        "replicas": [11, 29],
        "cases": [
            {
                "target_id": t,
                "replica": r,
                "num_qubits": 6,
                "target_vector_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
            }
            for t in wrapper.TARGETS
            for r in (11, 29)
        ],
        "algorithm": {key: {"engineering_fixture": True} for key in wrapper.ALGORITHM_FIELDS},
        "arms": {
            "candidate": {"roles": ["candidate"], "configuration": {"method": "fixture_a"}},
            "baseline": {"roles": ["primary_baseline"], "configuration": {"method": "fixture_b"}},
        },
        "budgets": {"candidate": {"updates": 1500}, "baseline": {"updates": 1500}},
        "acceptance": {
            "candidate": "candidate",
            "primary_baseline": "baseline",
            "primary_metric": "target-balanced mean of all eleven noisy conditions",
            "rule": {
                "route": "fidelity",
                "criteria": ["At least 0.005 mean gain and a positive gain on every target."],
                "complete_panel_required": True,
                "failed_case_policy": "retain_and_fail",
            },
        },
        "seed_policy": {"training": "confirmation_training", "evaluation": "exact; no final randomness"},
        "constructor": wrapper.constructor_spec(),
        "source_hashes": wrapper.sources(),
        "benchmark_hashes": wrapper.runner.custody()["benchmark_hashes"],
        "environment": construction.environment(),
    }
    path = tmp_path / "parent.json"
    pilot.write_record(path, record)
    return path, wrapper.read(path), vector


@pytest.mark.usefixtures("custody")
def test_synthetic_constructor_matches_frozen_recipe_bit_exactly(tmp_path: Path) -> None:
    """The wrapper changes custody only: endpoint parameters and block traces match."""
    vector = target()
    unchanged = vector.copy()
    old_root, new_root = tmp_path / "legacy", tmp_path / "wrapper"
    old_manifest = construction.prepare(old_root, vector, "synthetic_q2", 11, config=config())
    expected = construction.reproduce(old_root, old_manifest)
    manifest = wrapper.prepare(new_root, vector, "synthetic_q2", 11, config=config())
    actual = wrapper.run(new_root, manifest, vector)
    assert actual["phase"] == "engineering"
    assert actual["parent_checksum"] is None
    assert actual["theta"] == expected["theta"]
    assert actual["cold_construct_theta"] == expected["cold_construct_theta"]
    assert pilot.digest(actual["circuit"]) == pilot.digest(expected["circuit"])
    assert actual["pruning_events"] == expected["pruning_events"]
    assert actual["updates"] == expected["updates"] == 12
    for name in actual["blocks"]:
        new = wrapper.read(new_root / "blocks" / f"{name}.json")
        old = wrapper.read(old_root / "blocks" / f"{name}.json")
        assert new["signature"] == old["signature"]
        assert new["theta"] == old["theta"]
        assert new["trace"] == old["trace"]
        assert new["completed_updates"] == old["completed_updates"]
        assert actual["blocks"][name]["checksum"] == pilot.digest(new)
    np.testing.assert_array_equal(vector, unchanged)
    assert actual["construction_cpu_seconds"] > 0
    assert actual["construction_wall_seconds"] > 0
    assert actual["timing_complete"]
    assert manifest["initialization_seed"] == construction.seed_for("initialization", "synthetic_q2", 11)
    assert manifest["root_seed"] == 2026090520


@pytest.mark.usefixtures("custody")
@pytest.mark.parametrize("block", ["source_1", "source_2", "M_repair_0", "refine_M_0"])
def test_caught_interrupt_resumes_exactly_and_preserves_all_attempt_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, block: str
) -> None:
    """Each phase resumes from its full saved theta/trace without reseeding."""
    vector = target()
    whole, split = tmp_path / "whole", tmp_path / "split"
    full_manifest = wrapper.prepare(whole, vector, "synthetic_q2", 11, config=config())
    full = wrapper.run(whole, full_manifest, vector)
    manifest = wrapper.prepare(split, vector, "synthetic_q2", 11, config=config())
    write = pilot.write_record

    def interrupt(path: Path, record: dict[str, Any]) -> None:
        write(path, record)
        if path == split / "blocks" / f"{block}.json" and record["completed_updates"] == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(pilot, "write_record", interrupt)
    with pytest.raises(KeyboardInterrupt):
        wrapper.run(split, manifest, vector)
    monkeypatch.setattr(pilot, "write_record", write)
    result = wrapper.run(split, manifest, vector)
    assert result["theta"] == full["theta"]
    assert result["circuit"] == full["circuit"]
    assert result["pruning_events"] == full["pruning_events"]
    for name in result["blocks"]:
        actual = wrapper.read(split / "blocks" / f"{name}.json")
        expected = wrapper.read(whole / "blocks" / f"{name}.json")
        assert actual["theta"] == expected["theta"]
        assert actual["trace"] == expected["trace"]
    ledger = wrapper.read(split / "ATTEMPTS.json")
    assert [a["status"] for a in ledger["attempts"]] == ["interrupted", "complete"]
    assert all(a["cpu_seconds"] > 0 and a["wall_seconds"] > 0 for a in ledger["attempts"])
    assert result["construction_cpu_seconds"] == pytest.approx(sum(a["cpu_seconds"] for a in ledger["attempts"]))
    assert result["construction_wall_seconds"] == pytest.approx(sum(a["wall_seconds"] for a in ledger["attempts"]))
    assert ledger["attempts"][-1]["block_checksums"] == wrapper.block_hashes(split)


@pytest.mark.usefixtures("custody")
def test_no_q6_preparation_without_complete_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A q6 flag or identity cannot start preparation or reach a numerical kernel."""
    monkeypatch.setattr(construction, "run_pipeline", lambda *_: pytest.fail("q6 kernel reached"))
    vector = np.ones(64, dtype=np.complex128) / 8
    with pytest.raises(ValueError, match="parent"):
        wrapper.prepare(tmp_path / "forbidden", vector, wrapper.TARGETS[0], 11)
    assert not (tmp_path / "forbidden").exists()
    with pytest.raises(ValueError, match="synthetic"):
        wrapper.prepare(tmp_path / "engineering_bypass", vector, "synthetic_q6", 11, config=config())
    assert not (tmp_path / "engineering_bypass").exists()


@pytest.mark.usefixtures("custody")
def test_complete_parent_metadata_binds_case_without_any_q6_evolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Artificial metadata passes preparation while exact hashes bind later use."""
    path, document, vector = parent(tmp_path)
    monkeypatch.setattr(construction, "run_pipeline", lambda *_: pytest.fail("metadata preparation evolved q6"))
    checksum = pilot.digest(document)
    assert wrapper.parent_protocol(path, checksum) == document
    root = tmp_path / "case"
    manifest = wrapper.prepare(root, vector, wrapper.TARGETS[0], 11, parent_path=path, parent_checksum=checksum)
    assert manifest["config"] == asdict(construction.Config())
    assert manifest["phase"] == "confirmation"
    assert manifest["parent"] == {"path": str(path.resolve()), "checksum": checksum}
    assert list(root.iterdir()) == [root / "PROTOCOL.json"]
    assert wrapper.prepare(root, vector, wrapper.TARGETS[0], 11, parent_path=path, parent_checksum=checksum) == manifest
    document["acceptance"]["rule"]["criteria"].append("Changed after freeze")
    pilot.write_record(path, document)
    with pytest.raises(ValueError, match="checksummed parent"):
        wrapper.run(root, manifest, vector)
    assert not (root / "ATTEMPTS.json").exists()


@pytest.mark.usefixtures("custody")
@pytest.mark.parametrize(
    "field",
    [
        "initializer",
        "compiler",
        "training",
        "structure",
        "endpoint",
        "arms",
        "budgets",
        "acceptance",
        "seed_policy",
        "constructor",
        "source_hashes",
    ],
)
def test_parent_requires_every_algorithm_arm_budget_and_rule_section(tmp_path: Path, field: str) -> None:
    """Rehashing an incomplete draft does not make it a complete confirmation freeze."""
    path, document, vector = parent(tmp_path)
    if field in wrapper.ALGORITHM_FIELDS:
        document["algorithm"].pop(field)
    else:
        document.pop(field)
    pilot.write_record(path, document)
    with pytest.raises(ValueError, match=r"complete|incomplete"):
        wrapper.prepare(
            tmp_path / "case", vector, wrapper.TARGETS[0], 11, parent_path=path, parent_checksum=pilot.digest(document)
        )
    assert not (tmp_path / "case").exists()


@pytest.mark.usefixtures("custody")
@pytest.mark.parametrize(
    "change",
    [
        "draft",
        "baseline",
        "same_baseline",
        "criterion",
        "failure_policy",
        "missing_case",
        "case_hash",
        "constructor",
        "source",
        "budget_arm",
    ],
)
def test_semantically_incomplete_parent_or_changed_case_fails(tmp_path: Path, change: str) -> None:
    """Acceptance, full panel, exact constructor and vector hashes are mandatory."""
    path, document, vector = parent(tmp_path)
    if change == "draft":
        document["status"] = "draft"
    elif change == "baseline":
        document["acceptance"]["primary_baseline"] = "missing"
    elif change == "same_baseline":
        document["acceptance"]["primary_baseline"] = "candidate"
    elif change == "criterion":
        document["acceptance"]["rule"]["criteria"] = []
    elif change == "failure_policy":
        document["acceptance"]["rule"]["failed_case_policy"] = "drop"
    elif change == "missing_case":
        document["cases"].pop()
    elif change == "case_hash":
        for row in document["cases"]:
            row["target_vector_sha256"] = "0" * 64
    elif change == "constructor":
        document["constructor"]["config"]["source_updates"] = 1
    elif change == "source":
        (tmp_path / "kernel.py").write_text("# changed numerical dependency\n")
    else:
        document["budgets"].pop("baseline")
    pilot.write_record(path, document)
    with pytest.raises(ValueError, match=r"confirmation|Confirmation|parent|Parent"):
        wrapper.prepare(
            tmp_path / "case", vector, wrapper.TARGETS[0], 11, parent_path=path, parent_checksum=pilot.digest(document)
        )
    assert not (tmp_path / "case").exists()


@pytest.mark.usefixtures("custody")
@pytest.mark.parametrize("changed", ["parent_hash", "vector", "case", "source"])
def test_q6_custody_change_aborts_before_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    """Supplied arrays cannot bypass the complete parent-bound identity."""
    path, document, vector = parent(tmp_path)
    root = tmp_path / "case"
    manifest = wrapper.prepare(
        root, vector, wrapper.TARGETS[0], 11, parent_path=path, parent_checksum=pilot.digest(document)
    )
    monkeypatch.setattr(construction, "run_pipeline", lambda *_: pytest.fail("q6 kernel reached"))
    if changed == "parent_hash":
        manifest["parent"]["checksum"] = "0" * 64
    elif changed == "vector":
        vector = vector.copy()
        vector[0] *= -1
    elif changed == "case":
        manifest["replica"] = 29
    else:
        monkeypatch.setattr(wrapper, "sources", lambda: {"kernel.py": "0" * 64})
    with pytest.raises(ValueError, match=r"parent|custody|vector"):
        wrapper.run(root, manifest, vector)
    assert not (root / "ATTEMPTS.json").exists()


@pytest.mark.usefixtures("custody")
@pytest.mark.parametrize("changed", ["unknown", "missing_field", "negative", "orphan", "chunk"])
def test_unknown_or_missing_cost_provenance_blocks_numerical_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    """Uncatchable work and orphan checkpoints never acquire zero construction cost."""
    vector, root = target(), tmp_path / "case"
    manifest = wrapper.prepare(root, vector, "synthetic_q2", 11, config=config())
    write = pilot.write_record

    def interrupt(path: Path, record: dict[str, Any]) -> None:
        write(path, record)
        if path == root / "blocks" / "source_1.json":
            raise KeyboardInterrupt

    monkeypatch.setattr(pilot, "write_record", interrupt)
    with pytest.raises(KeyboardInterrupt):
        wrapper.run(root, manifest, vector)
    monkeypatch.setattr(pilot, "write_record", write)
    ledger_path = root / "ATTEMPTS.json"
    ledger = wrapper.read(ledger_path)
    if changed == "unknown":
        ledger["attempts"][0].update(status="running", cpu_seconds=None, wall_seconds=None)
    elif changed == "missing_field":
        ledger["attempts"][0].pop("cpu_seconds")
    elif changed == "negative":
        ledger["attempts"][0]["wall_seconds"] = -1
    elif changed == "orphan":
        ledger_path.unlink()
    else:
        path = root / "blocks" / "source_1.json"
        block = wrapper.read(path)
        block["theta"][0] += 0.01
        write(path, block)
    if changed not in {"orphan", "chunk"}:
        write(ledger_path, ledger)
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    monkeypatch.setattr(construction, "run_pipeline", lambda *_: pytest.fail("unknown-cost work resumed"))
    with pytest.raises(ValueError, match=r"cost|ledger|chunk"):
        wrapper.run(root, manifest, vector)
    assert before == {p: p.read_bytes() for p in root.rglob("*.json")}


@pytest.mark.usefixtures("custody")
def test_completed_result_revalidates_full_ledger_and_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed cache cannot hide modified durations or another endpoint identity."""
    vector, root = target(), tmp_path / "case"
    manifest = wrapper.prepare(root, vector, "synthetic_q2", 11, config=config())
    result = wrapper.run(root, manifest, vector)
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    monkeypatch.setattr(construction, "run_pipeline", lambda *_: pytest.fail("cached endpoint recomputed"))
    assert wrapper.run(root, manifest, vector) == result
    assert before == {p: p.read_bytes() for p in root.rglob("*.json")}
    ledger_path = root / "ATTEMPTS.json"
    ledger = wrapper.read(ledger_path)
    ledger["attempts"][0]["cpu_seconds"] += 1
    pilot.write_record(ledger_path, ledger)
    with pytest.raises(ValueError, match="cost ledger changed"):
        wrapper.run(root, manifest, vector)


@pytest.mark.usefixtures("custody")
def test_numerical_failure_before_publication_keeps_cost_and_full_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed attempt after completed blocks remains charged when publication resumes."""
    vector, root = target(), tmp_path / "case"
    manifest = wrapper.prepare(root, vector, "synthetic_q2", 11, config=config())
    pipeline = construction.run_pipeline
    failed_theta = []

    def fail(path: Path, record: dict[str, Any], state: np.ndarray, options: construction.Config) -> dict[str, Any]:
        failed_theta.extend(pipeline(path, record, state, options)["theta"])
        msg = "injected finalization failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(construction, "run_pipeline", fail)
    with pytest.raises(RuntimeError, match="injected finalization"):
        wrapper.run(root, manifest, vector)
    assert not (root / "RESULT.json").exists()
    ledger = wrapper.read(root / "ATTEMPTS.json")
    assert ledger["attempts"][0]["status"] == "failed"
    assert ledger["attempts"][0]["cpu_seconds"] > 0
    assert ledger["attempts"][0]["block_checksums"] == wrapper.block_hashes(root)
    monkeypatch.setattr(construction, "run_pipeline", pipeline)
    result = wrapper.run(root, manifest, vector)
    assert result["theta"] == failed_theta
    assert result["construction_cpu_seconds"] > ledger["attempts"][0]["cpu_seconds"]
    assert [a["status"] for a in wrapper.read(root / "ATTEMPTS.json")["attempts"]] == ["failed", "complete"]


@pytest.mark.usefixtures("custody")
def test_engineering_identity_and_invalid_input_cannot_cross_scope(tmp_path: Path) -> None:
    """Engineering overrides remain limited to normalized small synthetic vectors."""
    with pytest.raises(ValueError, match="synthetic"):
        wrapper.prepare(tmp_path / "identity", target(), "tfim_ferro", 11, config=config())
    with pytest.raises(ValueError, match="normalized"):
        wrapper.prepare(tmp_path / "norm", target() * 2, "synthetic_q2", 11, config=config())
    path, document, _ = parent(tmp_path)
    with pytest.raises(ValueError, match="Engineering"):
        wrapper.prepare(
            tmp_path / "parent",
            target(),
            "synthetic_q2",
            11,
            config=config(),
            parent_path=path,
            parent_checksum=pilot.digest(document),
        )
    root = tmp_path / "occupied"
    root.mkdir()
    (root / "user.txt").write_text("preserve\n")
    with pytest.raises(ValueError, match="nonempty"):
        wrapper.prepare(root, target(), "synthetic_q2", 11, config=config())
    assert (root / "user.txt").read_text() == "preserve\n"
