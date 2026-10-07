# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Complete-parent custody, fair-control gate and small physical confirmation engineering."""

from __future__ import annotations

import copy
import hashlib
import importlib
import shutil
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import best_candidate as candidate

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace only custody discovery, retaining all numerical kernels and protocol comparisons."""
    monkeypatch.setattr(candidate, "sources", lambda recipe: {"kernel": recipe})
    monkeypatch.setattr(candidate.lazy, "sources", lambda: {"lazy": "fixed"})
    monkeypatch.setattr(candidate.constructor, "constructor_spec", lambda: {"recipe": "fixed"})
    monkeypatch.setattr(candidate.construction, "environment", lambda: {"runtime": "fixed"})
    monkeypatch.setattr(candidate.base.runner, "custody", lambda: {"benchmark_hashes": {"fixture": "fixed"}})
    monkeypatch.setattr(candidate.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3))
    for name in candidate.THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")


def vector(n: int) -> np.ndarray:
    """Return a normalized small synthetic target with no benchmark fixture access."""
    rng = np.random.default_rng(904 + n)
    target = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return target / np.linalg.norm(target)


def engineering_design(root: Path, recipe: str = "original") -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return a complete reduced-budget synthetic design, preserving all seven roles."""
    vectors = {"synthetic_q2": vector(2)}
    exact = candidate.base.exact.Config(
        channel_evaluations=40,
        warmup_evaluations=1,
        max_rounds=1,
        shortlist=1,
        repair_evaluations=2,
        protected_initial_gates=0,
        checkpoint_every=1,
        history=2,
    )
    stochastic = candidate.base.stochastic.Config(
        total_updates=20,
        warmup=1,
        rounds=1,
        shortlist=1,
        repair=2,
        checkpoint_every=1,
        paths_per_channel=2,
        protected=0,
        online_warmup=1,
    )
    selected = candidate.configuration(recipe, exact, stochastic, list(vectors), (11, 29))
    design: dict[str, Any] = {
        "version": candidate.VERSION,
        "status": "frozen",
        "phase": "engineering",
        "recipe": recipe,
        "targets": list(vectors),
        "replicas": [11, 29],
        **selected,
        "configuration_sha256": pilot.digest(selected),
        "source_hashes": candidate.sources(recipe),
        "benchmark_hashes": candidate.base.runner.custody()["benchmark_hashes"],
        "environment": candidate.construction.environment(),
        "development_evidence": {},
        "disk_plan": candidate.disk_plan(root, 2),
    }
    pilot.write_record(root / "DESIGN.json", design)
    return candidate.read(root / "DESIGN.json"), vectors


@pytest.fixture
def prepared(tmp_path: Path, metadata: None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return an actual prepared synthetic parent without decoding any held-out data."""
    del metadata
    _, vectors = engineering_design(tmp_path)
    parent = candidate.prepare(tmp_path, vectors=vectors)
    return parent, vectors


def test_freeze_never_loads_targets_and_complete_design_precedes_hash_binding(
    tmp_path: Path, metadata: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scientific identities may be frozen without requesting any held-out vector or simulation."""
    del metadata
    monkeypatch.setattr(candidate, "development_evidence", lambda *_a: {})
    monkeypatch.setattr(candidate, "lazy_evidence", lambda *_a: {})
    monkeypatch.setattr(candidate, "load_target", lambda *_a: pytest.fail("freeze loaded a target"))
    frozen = candidate.freeze_design(tmp_path, "original", tmp_path / "old", tmp_path / "lazy")
    assert frozen["targets"] == list(candidate.TARGETS)
    assert frozen["replicas"] == [11, 29]
    assert set(frozen["arms"]) == set(candidate.METHODS)
    assert not (tmp_path / "PROTOCOL.json").exists()
    assert list(tmp_path.iterdir()) == [tmp_path / "DESIGN.json"]
    candidate.verify_design(tmp_path, frozen)


def test_no_target_loader_before_valid_design_and_no_constructor_during_prepare(
    tmp_path: Path, metadata: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing or changed design fails before loading; all case manifests precede construction."""
    del metadata
    monkeypatch.setattr(candidate, "load_target", lambda *_a: pytest.fail("invalid design loaded target"))
    with pytest.raises(FileNotFoundError):
        candidate.prepare(tmp_path)
    _, vectors = engineering_design(tmp_path)
    monkeypatch.setattr(candidate.lazy, "run", lambda *_a: pytest.fail("prepare initialized"))
    parent = candidate.prepare(tmp_path, vectors=vectors)
    assert len(parent["cases"]) == 2
    assert len(list(tmp_path.glob("*/CASE.json"))) == 2
    assert not list(tmp_path.rglob("PROBE.json"))
    candidate.verify_parent(tmp_path, parent)


@pytest.mark.parametrize(
    "changed", ["budget", "recipe", "target", "replica", "acceptance", "source", "case", "child", "parent"]
)
def test_full_parent_and_configuration_tamper_fails_before_target_access(
    tmp_path: Path,
    prepared: tuple[dict[str, Any], dict[str, np.ndarray]],
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    """Correct JSON serialization cannot legitimize a modified algorithm, panel or child identity."""
    parent, _ = prepared
    altered = copy.deepcopy(parent)
    if changed == "budget":
        altered["algorithm"]["training"]["exact"]["channel_evaluations"] += 2
    elif changed == "recipe":
        altered["recipe"] = "state_preparation_compiled"
    elif changed == "target":
        altered["targets"] = ["tfim_ferro"]
    elif changed == "replica":
        altered["replicas"] = [11]
    elif changed == "acceptance":
        altered["acceptance"]["rule"]["minimum_mean_advantage"] = 0
    elif changed == "source":
        altered["source_hashes"]["kernel"] = "changed"
    elif changed == "case":
        altered["cases"][0]["target_vector_sha256"] = "0" * 64
    elif changed == "child":
        path = tmp_path / "synthetic_q2__11/CASE.json"
        record = candidate.read(path)
        record["training_seed"] += 1
        pilot.write_record(path, record)
    else:
        record = candidate.read(tmp_path / "PROTOCOL.json")
        record["configuration_sha256"] = "changed"
        pilot.write_record(tmp_path / "PROTOCOL.json", record)
    monkeypatch.setattr(candidate, "target_loader", lambda *_a: pytest.fail("invalid parent loaded target"))
    with pytest.raises(ValueError, match=r"parent|registry"):
        candidate.verify_parent(tmp_path, altered)


def test_phase_exact_resume_retains_paid_interruption_and_never_replays_durable_work(tmp_path: Path) -> None:
    """Complete child state survives a caught interruption with both physical attempts retained."""
    state_path = tmp_path / "work/state.json"
    calls = []

    def first() -> dict[str, Any]:
        calls.append("first")
        pilot.write_record(state_path, {"theta": [0.2], "moments": [0.3]})
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        candidate.phase(tmp_path, "fixed", first)

    def resume() -> dict[str, Any]:
        calls.append("resume")
        return candidate.read(state_path)

    result = candidate.phase(tmp_path, "fixed", resume)
    assert result["value"] == {"theta": [0.2], "moments": [0.3]}
    assert calls == ["first", "resume"]
    ledger = candidate.read(tmp_path / "ATTEMPTS.json")
    assert [a["status"] for a in ledger["attempts"]] == ["interrupted", "complete"]
    assert all(a["cpu_seconds"] > 0 for a in ledger["attempts"])
    assert result["cpu_seconds"] == sum(a["cpu_seconds"] for a in ledger["attempts"])
    assert candidate.phase(tmp_path, "fixed", lambda: pytest.fail("cache replayed")) == result


@pytest.mark.parametrize("changed", ["theta", "score", "cost", "new_file", "missing_file", "unknown", "orphan"])
def test_phase_binds_all_cached_fields_exact_registry_and_known_costs(tmp_path: Path, changed: str) -> None:
    """Signatures alone never authorize edited endpoints, scores, costs or partial file inventories."""

    def work() -> dict[str, Any]:
        pilot.write_record(tmp_path / "work/raw.json", {"state": "fixed"})
        return {"theta": [0.2], "scores": {"noiseless": 0.9}}

    if changed == "orphan":
        pilot.write_record(tmp_path / "work/orphan.json", {"unmeasured": True})
    else:
        result = candidate.phase(tmp_path, "fixed", work)
        if changed == "theta":
            result["value"]["theta"][0] += 0.1
        elif changed == "score":
            result["value"]["scores"]["noiseless"] = 1.0
        elif changed == "cost":
            result["cpu_seconds"] = 0.0
        elif changed == "new_file":
            pilot.write_record(tmp_path / "work/new.json", {"unexpected": True})
        elif changed == "missing_file":
            (tmp_path / "work/raw.json").unlink()
        else:
            ledger = candidate.read(tmp_path / "ATTEMPTS.json")
            ledger["attempts"][-1].update(status="running", cpu_seconds=None, wall_seconds=None)
            pilot.write_record(tmp_path / "ATTEMPTS.json", ledger)
        pilot.write_record(tmp_path / "RESULT.json", result)
    with pytest.raises(ValueError, match=r"Confirmation|confirmation|Unknown|adopt"):
        candidate.phase(tmp_path, "fixed", lambda: pytest.fail("invalid phase ran"))


def scores(fidelity: float) -> dict[str, Any]:
    """Return a complete exact-score panel for deterministic acceptance arithmetic."""
    return {
        "fidelities": {noise: 1.0 if noise == "noiseless" else fidelity for noise in candidate.base.channels.NOISE_IDS}
    }


def complete_records() -> list[dict[str, Any]]:
    """Return all ten cases with a candidate advantage against every fair control."""
    records = []
    for target in candidate.TARGETS:
        for replica in candidate.REPLICAS:
            arms = {
                name: {"scores": scores(0.91 if name == candidate.CANDIDATE else 0.90)} for name in candidate.METHODS
            }
            records.append({
                "status": "complete",
                "target_id": target,
                "replica": replica,
                "panels": {
                    panel: {"status": "complete", "arms": copy.deepcopy(arms)}
                    for panel in ("primary", "all_trained_sensitivity", "full_cap")
                },
            })
    return records


def test_primary_gate_uses_two_replica_target_means_and_all_controls() -> None:
    """The strongest aggregate comparator is identified over all six alternatives."""
    records = complete_records()
    analysis = candidate.analyze(records, candidate.TARGETS, candidate.REPLICAS)
    assert analysis["promoted"]
    assert analysis["panels"]["primary"]["comparisons"][candidate.BASELINE]["mean_advantage"] == pytest.approx(0.01)
    for record in records:
        record["panels"]["primary"]["arms"]["fixed_mean2_lbfgs"]["scores"] = scores(0.908)
    analysis = candidate.analyze(records, candidate.TARGETS, candidate.REPLICAS)
    assert not analysis["promoted"]
    assert analysis["panels"]["primary"]["strongest_control"] == "fixed_mean2_lbfgs"
    assert analysis["panels"]["full_cap"]["passes_fixed_and_strongest_rule"]


@pytest.mark.parametrize(
    "change",
    [
        "missing_case",
        "failed_case",
        "duplicate",
        "missing_role",
        "missing_noise",
        "nonfinite",
        "negative_target",
        "one_replica_bad",
    ],
)
def test_complete_panel_gate_cannot_be_rescued_by_subset_or_full_cap(change: str) -> None:
    """Every target and replica contributes, and supporting panels cannot rescue a failed primary gate."""
    records = complete_records()
    if change == "missing_case":
        records.pop()
    elif change == "failed_case":
        records[0]["status"] = "failed"
    elif change == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif change == "missing_role":
        records[0]["panels"]["primary"]["arms"].pop("unchanged")
    elif change == "missing_noise":
        records[0]["panels"]["primary"]["arms"][candidate.CANDIDATE]["scores"]["fidelities"].pop("noiseless")
    elif change == "nonfinite":
        records[0]["panels"]["primary"]["arms"][candidate.CANDIDATE]["scores"] = scores(float("nan"))
    else:
        records[0]["panels"]["primary"]["arms"][candidate.CANDIDATE]["scores"] = scores(0.87)
        if change == "negative_target":
            records[1]["panels"]["primary"]["arms"][candidate.CANDIDATE]["scores"] = scores(0.89)
    assert not candidate.analyze(records, candidate.TARGETS, candidate.REPLICAS)["promoted"]


def test_common_cutoff_selects_shorter_final_and_every_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixed short-running controls remain deployable at the four-structural cutoff."""
    arms = {}
    for name in candidate.METHODS:
        final = 0 if name == "unchanged" else 2 if name == "fixed_mean2_lbfgs" else 10
        arms[name] = {
            "status": "complete",
            "refinement_cpu_seconds": final,
            "final": {"cpu_seconds": final, "theta": [final]},
            "curve": [{"cpu_seconds": 0, "theta": [0]}, {"cpu_seconds": final, "theta": [final]}],
        }
    monkeypatch.setattr(candidate.base, "score", lambda _store, point: {"point": point})
    monkeypatch.setattr(candidate, "compact_score", lambda scored: scored)
    panel = candidate.common_panel(cast("candidate.base.CompleteStore", None), arms, candidate.PRIMARY, 5)
    assert panel["refinement_cpu_seconds"] == 10
    assert panel["total_cpu_seconds"] == 15
    assert set(panel["arms"]) == set(candidate.METHODS)
    assert panel["arms"]["fixed_mean2_lbfgs"]["point"]["cpu_seconds"] == 2
    assert panel["arms"]["unchanged"]["point"]["cpu_seconds"] == 0


def test_disk_forecast_covers_complete_product_and_blocks_before_any_load(
    tmp_path: Path, metadata: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Storage limits cannot quietly shorten the target/replica panel."""
    del metadata
    plan = candidate.disk_plan(tmp_path, 10)
    assert plan["required_bytes"] == 10 * candidate.PER_CASE_BYTES + candidate.RESERVE_BYTES
    monkeypatch.setattr(candidate.shutil, "disk_usage", lambda _: SimpleNamespace(free=plan["required_bytes"] - 1))
    monkeypatch.setattr(candidate, "development_evidence", lambda *_a: pytest.fail("disk failure read evidence"))
    with pytest.raises(RuntimeError, match="ten-case"):
        candidate.freeze_design(tmp_path, "original", tmp_path / "old", tmp_path / "lazy")
    assert not list(tmp_path.iterdir())


def test_real_q2_all_seven_roles_complete_and_resume_exactly(
    tmp_path: Path, prepared: tuple[dict[str, Any], dict[str, np.ndarray]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual frozen exact/stochastic kernels run through the parent on synthetic q2 data."""
    parent, vectors = prepared
    case = parent["cases"][0]
    job = {"root": str(tmp_path), "parent": parent, "case": case}
    monkeypatch.setattr(candidate, "load_target", lambda *_a: pytest.fail("engineering touched benchmark"))
    result = candidate.run_case(job, loader=lambda _n, target: vectors[target].copy())
    assert result["status"] == "complete"
    assert set(result["panels"]["full_cap"]["arms"]) == set(candidate.METHODS)
    assert all(
        set(arm["scores"]["fidelities"]) == set(candidate.base.channels.NOISE_IDS)
        for arm in result["panels"]["full_cap"]["arms"].values()
    )
    assert result["initializer"]["rank_two_selected"]
    assert result["initializer"]["nested_legacy_cpu_seconds"] is None
    assert result["physical_invocation"]["cpu_seconds"] >= sum(
        row["cpu_seconds"] for row in result["phase_costs"].values()
    )
    assert "raw_training" not in result
    assert "curve" not in result
    monkeypatch.setattr(candidate, "train_case", lambda *_a: pytest.fail("completed case trained again"))
    assert candidate.run_case(job, loader=lambda *_a: pytest.fail("completed case loaded again")) == result
    summary = candidate.summarize(tmp_path, parent)
    assert not summary["analysis"]["promoted"]
    assert len(summary["cases"]) == 2
    assert summary["status"] == "incomplete_or_failed"
    assert len((tmp_path / "SCOREBOARD.csv").read_text().splitlines()) == 1 + 2 * 3 * 7 * 12


@pytest.mark.parametrize("args", [["--run", "--workers", "3"], ["--run", "--recipe", "original"], ["--freeze-design"]])
def test_cli_refuses_unallocated_workers_and_post_freeze_choice_changes(args: list[str]) -> None:
    """The public entry point never accepts a grid or new recipe at execution time."""
    with pytest.raises(SystemExit, match="2"):
        candidate.main(args)


def test_real_q2_compiled_recipe_preserves_all_roles_and_chargeable_initial_compilation(
    tmp_path: Path, metadata: None
) -> None:
    """The alternative developed kernel receives the identical lazy endpoint API."""
    del metadata
    _, vectors = engineering_design(tmp_path, "state_preparation_compiled")
    parent = candidate.prepare(tmp_path, vectors=vectors)
    result = candidate.run_case(
        {"root": str(tmp_path), "parent": parent, "case": parent["cases"][0]},
        loader=lambda _n, target: vectors[target].copy(),
    )
    assert result["status"] == "complete"
    assert set(result["panels"]["primary"]["arms"]) == set(candidate.METHODS)
    path = tmp_path / "synthetic_q2__11/execution/training/work/EXECUTION.json"
    execution = candidate.read(path)
    assert execution["initial"]["cpu_seconds"] > 0
    assert result["panels"]["full_cap"]["arms"]["unchanged"]["point"] == execution["arms"]["unchanged"]["final"]
    assert result["initializer_cpu_seconds"] == execution["initializer_cpu_seconds"]


def test_interrupted_complete_training_reuses_exact_optimizer_and_curve_files(
    tmp_path: Path, prepared: tuple[dict[str, Any], dict[str, np.ndarray]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caught publication interruption resumes every frozen optimizer state without modifying its trajectory."""
    parent, vectors = prepared
    job = {"root": str(tmp_path), "parent": parent, "case": parent["cases"][0]}
    actual = candidate.train_case

    def interrupted(
        root: Path, parent: dict[str, Any], case: dict[str, Any], source: dict[str, Any], target: np.ndarray
    ) -> dict[str, Any]:
        actual(root, parent, case, source, target)
        raise KeyboardInterrupt

    monkeypatch.setattr(candidate, "train_case", interrupted)
    with pytest.raises(KeyboardInterrupt):
        candidate.run_case(job, loader=lambda _n, target: vectors[target].copy())
    work = tmp_path / "synthetic_q2__11/execution/training/work"
    before = candidate.inventory(work)
    assert any("TRAINING.json" in name for name in before)
    monkeypatch.setattr(candidate, "train_case", actual)
    result = candidate.run_case(job, loader=lambda *_a: pytest.fail("saved input reloaded"))
    assert result["status"] == "complete"
    assert candidate.inventory(work) == before
    ledger = candidate.read(tmp_path / "synthetic_q2__11/execution/ATTEMPTS.json")
    assert [a["status"] for a in ledger["attempts"]] == ["interrupted", "complete"]
    assert result["physical_invocation"]["cpu_seconds"] == sum(a["cpu_seconds"] for a in ledger["attempts"])


@pytest.mark.parametrize("field", ["scores", "cost", "branch"])
def test_compact_outer_result_tamper_is_rejected_even_by_summary(
    tmp_path: Path, prepared: tuple[dict[str, Any], dict[str, np.ndarray]], monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Summary cannot detach a result's scores, initializer decision or total CPU from sealed evidence."""
    parent, _ = prepared
    case = parent["cases"][0]
    child = tmp_path / "synthetic_q2__11"
    raw: dict[str, Any] = {
        "status": "complete",
        "target_id": case["target_id"],
        "replica": case["replica"],
        "panels": {},
        "initializer": {"rank_two_selected": True},
    }
    physical = candidate.phase(child / "execution", [pilot.digest(parent), case, "complete_physical_case"], lambda: raw)
    record = candidate.case_record(parent, case, physical)
    if field == "scores":
        record["panels"]["changed"] = True
    elif field == "cost":
        record["physical_invocation"]["cpu_seconds"] += 1
    else:
        record["initializer"]["rank_two_selected"] = False
    pilot.write_record(child / "RESULT.json", record)
    monkeypatch.setattr(candidate, "target_loader", lambda *_a: pytest.fail("tampered summary loaded target"))
    with pytest.raises(ValueError, match="changed case"):
        candidate.summarize(tmp_path, parent)


def test_scientific_prepare_binds_all_ten_only_after_design_with_metadata_standins(
    tmp_path: Path, metadata: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No held-out state is decoded: a synthetic q6 array tests only freeze-before-loader ordering."""
    del metadata
    monkeypatch.setattr(candidate, "development_evidence", lambda *_a: {})
    monkeypatch.setattr(candidate, "lazy_evidence", lambda *_a: {})
    design = candidate.freeze_design(tmp_path, "original", tmp_path / "old", tmp_path / "lazy")
    loaded = []

    def loader(n: int, target_id: str) -> SimpleNamespace:
        assert candidate.read(tmp_path / "DESIGN.json") == design
        assert n == 6
        assert target_id in candidate.TARGETS
        loaded.append(target_id)
        return SimpleNamespace(state_vector_copy=lambda: np.ones(64, dtype=np.complex128) / 8)

    def parent_schema(path: Path, checksum: str) -> dict[str, Any]:
        parent = candidate.read(path)
        assert pilot.digest(parent) == checksum
        assert parent["schema"] == candidate.constructor.PARENT_SCHEMA
        assert len(parent["cases"]) == 10
        assert set(parent["algorithm"]) == set(candidate.constructor.ALGORITHM_FIELDS)
        return parent

    monkeypatch.setattr(candidate, "load_target", loader)
    monkeypatch.setattr(candidate.constructor, "parent_protocol", parent_schema)
    monkeypatch.setattr(candidate.lazy, "run", lambda *_a: pytest.fail("metadata-only prepare ran initializer"))
    parent = candidate.prepare(tmp_path)
    assert loaded == list(candidate.TARGETS)
    assert len(list(tmp_path.glob("*/CASE.json"))) == 10
    assert not list(tmp_path.rglob("PROBE.json"))
    candidate.verify_parent(tmp_path, parent)
    monkeypatch.setattr(candidate, "load_target", lambda *_a: pytest.fail("cached prepare reloaded"))
    assert candidate.prepare(tmp_path) == parent


def test_engineering_cached_binding_rejects_changed_vectors_and_missing_panel_members(
    tmp_path: Path, prepared: tuple[dict[str, Any], dict[str, np.ndarray]]
) -> None:
    """A synthetic shortcut cannot replace frozen input bytes on resume."""
    _, vectors = prepared
    with pytest.raises(ValueError, match="entire distinct"):
        candidate.prepare(tmp_path, vectors={})
    altered = {name: value.copy() for name, value in vectors.items()}
    altered["synthetic_q2"][0] *= -1
    with pytest.raises(ValueError, match="changed after"):
        candidate.prepare(tmp_path, vectors=altered)


@pytest.mark.parametrize(
    ("platform_name", "raw", "expected", "unit"), [("darwin", 1234, 1234, "bytes"), ("linux", 1234, 1263616, "KiB")]
)
def test_memory_units_and_worker_lifetime_scope(platform_name: str, raw: int, expected: int, unit: str) -> None:
    """Normalize peak RSS without claiming incremental per-method memory."""
    measured = candidate.peak_rss(raw, platform_name)
    assert measured["bytes"] == expected
    assert measured["raw_unit"] == unit
    assert "not incremental" in measured["scope"]


@pytest.mark.parametrize("raw", [-1.0, float("nan"), float("inf"), True])
def test_memory_rejects_invalid_measurements(raw: float) -> None:
    """Unknown or Boolean memory measurements cannot become valid resource totals."""
    with pytest.raises(ValueError, match="Peak RSS"):
        candidate.peak_rss(raw, "darwin")


@pytest.mark.parametrize("name", candidate.THREAD_VARIABLES)
def test_scientific_freeze_requires_all_four_explicit_thread_limits(
    tmp_path: Path, metadata: None, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """An omitted BLAS/MKL setting is refused before evidence reads or target loading."""
    del metadata
    monkeypatch.delenv(name)
    monkeypatch.setattr(candidate, "development_evidence", lambda *_a: pytest.fail("threads checked after evidence"))
    with pytest.raises(ValueError, match="Set OPENBLAS"):
        candidate.freeze_design(tmp_path, "original", tmp_path / "old", tmp_path / "lazy")


def test_cached_development_vector_is_linked_to_existing_constructor_hash_without_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only saved development bytes are decoded; the benchmark fixture loader remains unreachable."""
    vector = np.ones(64, dtype=np.complex128) / 8
    target_id = candidate.base.TARGETS[0]
    child: dict[str, Any] = {
        "target": [[z.real, z.imag] for z in vector],
        "target_id": target_id,
        "replica": 11,
        "target_vector_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
    }
    child_path = tmp_path / "constructor/PROTOCOL.json"
    pilot.write_record(child_path, child)
    original: dict[str, Any] = {
        "starts": [{"target_id": target_id, "replica": 11, "files": {str(child_path): pilot.digest(child)}}]
    }
    pilot.write_record(tmp_path / "PROTOCOL.json", original)
    monkeypatch.setattr(candidate, "load_target", lambda *_a: pytest.fail("cached evidence loaded fixture"))
    actual, files = candidate.cached_development_target(tmp_path, target_id)
    assert np.array_equal(actual, vector)
    assert len(files) == 2
    child["target_vector_sha256"] = "0" * 64
    pilot.write_record(child_path, child)
    original["starts"][0]["files"][str(child_path)] = pilot.digest(child)
    pilot.write_record(tmp_path / "PROTOCOL.json", original)
    with pytest.raises(ValueError, match="byte hash"):
        candidate.cached_development_target(tmp_path, target_id)


@pytest.mark.parametrize("changed", [None, "theta", "score", "cost", "execution_score"])
def test_compiled_development_evidence_uses_complete_cache_reconstruction_even_if_summary_resigned(
    tmp_path: Path, metadata: None, monkeypatch: pytest.MonkeyPatch, changed: str | None
) -> None:
    """Outer or execution field edits cannot change promotion by resigning the development summary."""
    del metadata
    live = tmp_path / "engineering"
    _, vectors = engineering_design(live, "state_preparation_compiled")
    parent = candidate.prepare(live, vectors=vectors)
    case = parent["cases"][0]
    finished = candidate.run_case(
        {"root": str(live), "parent": parent, "case": case}, loader=lambda _n, t: vectors[t].copy()
    )
    source = candidate.read(live / "synthetic_q2__11/execution/initializer/RESULT.json")["value"]["endpoint"]
    source["initializer_cpu_seconds"] = finished["initializer_cpu_seconds"]
    root = tmp_path / "development"
    child = root / "synthetic_q2__11"
    shutil.copytree(live / "synthetic_q2__11/execution/training/work", child)
    module = importlib.import_module(
        "experiments.overnight_state_preparation_20260913_0108.compiled_composition_experiment"
    )
    entry = {"target_id": "synthetic_q2", "replica": 11, "endpoint": source}
    protocol: dict[str, Any] = {
        "starts": [entry],
        "initializer_root": str(tmp_path / "saved_initializers"),
        "seed_namespace": "confirmation_training",
        "exact_config": parent["algorithm"]["training"]["exact"],
        "stochastic_config": parent["algorithm"]["training"]["stochastic"],
    }
    execution = candidate.read(child / "EXECUTION.json")
    record: dict[str, Any] = {
        **execution,
        "execution_signature": execution["signature"],
        "execution_checksum": pilot.digest(execution),
        "signature": pilot.digest([protocol, entry]),
        "protocol_checksum": pilot.digest(protocol),
        "target_id": "synthetic_q2",
        "replica": 11,
        "version": module.VERSION,
    }
    if changed == "theta":
        record["arms"][candidate.CANDIDATE]["final"]["theta"][0] += 0.1
    elif changed == "score":
        record["common_primary"]["arms"][candidate.CANDIDATE]["scores"]["fidelities"]["noiseless"] -= 0.1
    elif changed == "cost":
        record["arms"][candidate.CANDIDATE]["total_cpu_seconds"] += 1
    elif changed == "execution_score":
        execution["common_primary"]["arms"][candidate.CANDIDATE]["scores"]["fidelities"]["noiseless"] -= 0.1
        pilot.write_record(child / "EXECUTION.json", execution)
        record["execution_checksum"] = pilot.digest(execution)
    pilot.write_record(root / "PROTOCOL.json", protocol)
    pilot.write_record(child / "RESULT.json", record)
    pilot.write_record(
        root / "SUMMARY.json", {"status": "complete", "case_checksums": {"synthetic_q2": pilot.digest(record)}}
    )
    monkeypatch.setattr(candidate.base, "TARGETS", ("synthetic_q2",))
    monkeypatch.setattr(candidate, "cached_development_target", lambda *_a: (vectors["synthetic_q2"].copy(), {}))
    monkeypatch.setattr(module, "verify", lambda *_a: None)
    monkeypatch.setattr(module, "analyze", lambda *_a: {"promoted": True})
    monkeypatch.setattr(candidate, "load_target", lambda *_a: pytest.fail("evidence loaded fixture"))
    monkeypatch.setattr(candidate.base.channels, "evaluate", lambda *_a, **_k: pytest.fail("evidence evaluated state"))
    monkeypatch.setattr(
        candidate.lazy.initializer_policy, "compile_source", lambda *_a: pytest.fail("evidence compiled")
    )
    monkeypatch.setattr(
        candidate.kernel("state_preparation_compiled").compiler,
        "compile_endpoint",
        lambda *_a: pytest.fail("evidence state-compiled"),
    )
    if changed is None:
        assert candidate.development_evidence("state_preparation_compiled", root)
    else:
        with pytest.raises(ValueError, match=r"differs|differ"):
            candidate.development_evidence("state_preparation_compiled", root)


@pytest.mark.parametrize("changed", [None, "theta", "score", "cost"])
def test_original_development_record_binds_scores_points_and_costs_to_existing_receipts(
    tmp_path: Path,
    prepared: tuple[dict[str, Any], dict[str, np.ndarray]],
    monkeypatch: pytest.MonkeyPatch,
    changed: str | None,
) -> None:
    """Original control evidence is reconstructed without fresh evaluation, regardless of a resigned summary."""
    parent, vectors = prepared
    case = parent["cases"][0]
    finished = candidate.run_case(
        {"root": str(tmp_path), "parent": parent, "case": case}, loader=lambda _n, t: vectors[t].copy()
    )
    work = tmp_path / "synthetic_q2__11/execution/training/work"
    source_root, audit_root = tmp_path / "source", tmp_path / "audit"
    old_child, child = source_root / "synthetic_q2__11", audit_root / "synthetic_q2__11"
    shutil.copytree(work, old_child)
    for name in ("circuits", "operations"):
        shutil.copytree(work / name, child / name)
    config = candidate.base.exact.Config(**parent["algorithm"]["training"]["exact"])
    arms = {name: candidate.read(old_child / "arms" / f"{name}.json") for name in candidate.TRAINED}
    arms["unchanged"] = {"status": "complete", "final": finished["panels"]["full_cap"]["arms"]["unchanged"]["point"]}
    cutoff = min(arms[name]["refinement_cpu_seconds"] for name in candidate.PRIMARY)
    points = candidate.common_control_audit.select_points(arms, candidate.TRAINED, cutoff)
    source_case: dict[str, Any] = {
        "arms": arms,
        "common_primary": {"total_cpu_seconds": finished["initializer_cpu_seconds"] + cutoff},
    }
    pilot.write_record(old_child / "RESULT.json", source_case)
    pilot.write_record(
        source_root / "PROTOCOL.json",
        {
            "initializer_root": str(tmp_path / "cached_initializer"),
            "exact_config": parent["algorithm"]["training"]["exact"],
        },
    )
    pilot.write_record(source_root / "SUMMARY.json", {"case_checksum": pilot.digest(source_case)})
    entry = {"target_id": "synthetic_q2", "checksum": pilot.digest(source_case), "points": points}
    protocol = {"input_panel": {"root": str(source_root), "cases": [entry]}}
    scored = {
        name: candidate.cached_score(
            old_child if name in candidate.PRIMARY else child, vectors["synthetic_q2"], config, point
        )
        for name, point in points.items()
    }
    unchanged = candidate.cached_score(old_child, vectors["synthetic_q2"], config, arms["unchanged"]["final"])
    value = {
        "status": "complete",
        "target_id": "synthetic_q2",
        "refinement_cpu_seconds": cutoff,
        "total_cpu_seconds": source_case["common_primary"]["total_cpu_seconds"],
        "points": points,
        "source_case_checksum": pilot.digest(source_case),
        "source_unchanged_fidelities": unchanged["fidelities"],
        "scores": scored,
        "dependencies": {
            str(p.relative_to(child)): candidate.file_hash(p) for p in child.rglob("*.json") if p.parent != child
        },
    }
    signature = pilot.digest(pilot.digest([protocol, entry]))
    ledger: dict[str, Any] = {
        "signature": signature,
        "attempts": [{"status": "complete", "cpu_seconds": 0.01, "wall_seconds": 0.02}],
    }
    pilot.write_record(child / "RESULT.attempts.json", ledger)
    initial: dict[str, Any] = {
        "signature": signature,
        "attempts_checksum": pilot.digest(ledger),
        "cpu_seconds": 0.01,
        "wall_seconds": 0.02,
        "value": value,
    }
    pilot.write_record(child / "RESULT.json", initial)
    record = candidate.read(child / "RESULT.json")
    if changed == "theta":
        record["value"]["points"][candidate.CANDIDATE]["theta"][0] += 0.1
    elif changed == "score":
        record["value"]["scores"][candidate.CANDIDATE]["fidelities"]["noiseless"] -= 0.1
    elif changed == "cost":
        record["cpu_seconds"] += 1
    pilot.write_record(child / "RESULT.json", record)
    pilot.write_record(audit_root / "SUMMARY.json", {"cases": [pilot.digest(record)]})
    monkeypatch.setattr(candidate, "cached_development_target", lambda *_a: (vectors["synthetic_q2"].copy(), {}))
    monkeypatch.setattr(candidate, "load_target", lambda *_a: pytest.fail("original evidence loaded fixture"))
    monkeypatch.setattr(
        candidate.base.channels, "evaluate", lambda *_a, **_k: pytest.fail("original evidence evaluated")
    )
    if changed is None:
        assert candidate.original_record(audit_root, protocol, record)
    else:
        with pytest.raises(ValueError, match="differ from"):
            candidate.original_record(audit_root, protocol, record)


def test_read_only_reporting_apis_never_create_missing_records(
    tmp_path: Path, prepared: tuple[dict[str, Any], dict[str, np.ndarray]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reporting validates sealed evidence only and cannot repair or start an incomplete case."""
    parent, _ = prepared
    before = candidate.inventory(tmp_path)
    monkeypatch.setattr(candidate, "load_target", lambda *_a: pytest.fail("reporting loaded fixture"))
    monkeypatch.setattr(candidate, "execute_case", lambda *_a: pytest.fail("reporting executed"))
    with pytest.raises(ValueError, match="Read-only phase"):
        candidate.read_case(tmp_path, parent, parent["cases"][0])
    assert candidate.inventory(tmp_path) == before
