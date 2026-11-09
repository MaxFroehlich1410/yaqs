# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Manuscript provenance, deduplication, real solver dispatch, and queue contracts."""

from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.yaqs.analog import evolution
from mqt.yaqs.analog.analog_tjm import capture_conservation_target

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture
def reproduction(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load opt-in scripts without starting any experiments.

    Returns:
        Reproduction runner module.
    """
    folder = Path(__file__).resolve().parents[3] / "paper" / "bug-mps-benchmarks" / "spin_conservation"
    monkeypatch.syspath_prepend(str(folder))
    return importlib.import_module("paper_reproduction")


def test_plan_is_metadata_and_deduplicates(reproduction: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """The nine manuscript arms serve every table and plot, with four scalar arms."""

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Planning performed numerical work")

    monkeypatch.setattr(reproduction.fx, "initial_state", forbidden)
    plan = reproduction.make_plan()
    assert len(plan["arms"]) == len({a["id"] for a in plan["arms"]}) == 13
    assert len([a for a in plan["arms"] if a["role"] == "manuscript"]) == 9
    assert set(plan["table_III"]) <= set(plan["table_II"]) == set(plan["figure_1"])
    assert plan["protocol"]["hamiltonian_scale"] == pytest.approx(1.0)
    assert plan["protocol"]["total_time"] == 6
    assert all(a["fallback"] is None for a in plan["arms"] if a["variant"].startswith("joint"))
    with pytest.raises(ValueError, match="integral"):
        reproduction.make_plan(total_time=0.015)


def test_shared_queue_reference_barrier_and_exclusivity(reproduction: ModuleType, tmp_path: Path) -> None:
    """Concurrent workers cannot duplicate a reference or start its dependents early."""
    dispatch = importlib.import_module("paper_reproduction_dispatch")
    plan = reproduction.make_plan(length=4, cap=4, total_time=0.1, dt=0.05)
    out = tmp_path / "run"
    reproduction.prepare(plan, out)
    jobs = dispatch.queue(plan)
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: dispatch.claim(out, jobs), range(16)))
    assert [j["id"] for j in claims if j] == ["reference"]
    reproduction.pilot.write_json(out / "reference" / "done.json", {"test": True})
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: dispatch.claim(out, jobs), range(24)))
    identifiers = [j["id"] for j in claims if j]
    assert len(set(identifiers)) == len(identifiers) == 13
    snapshot = dispatch.state(out)
    assert snapshot["counts"]["complete"] == 1
    assert snapshot["counts"]["claimed_or_interrupted"] == 13
    assert snapshot["counts"]["queued"] == 0
    with pytest.raises(ValueError, match="new output"):
        reproduction.prepare(plan, out)


def test_reference_provenance_guard(reproduction: ModuleType, tmp_path: Path) -> None:
    """A changed source signature cannot silently resume a prepared study."""
    out = tmp_path / "run"
    reproduction.prepare(reproduction.make_plan(), out)
    manifest = reproduction.read(out / "manifest.json")
    manifest["source_sha256"]["changed.py"] = "bad"
    reproduction.pilot.write_json(out / "manifest.json", manifest)
    with pytest.raises(ValueError, match="source changed"):
        reproduction.verify(out)


def test_complete_small_reproduction_matches_legacy(reproduction: ModuleType, tmp_path: Path) -> None:
    """All real BUG/TDVP schedules produce shared plots/tables and match legacy evolution."""
    run = reproduction
    out = tmp_path / "run"
    plan = run.make_plan(length=4, cap=4, total_time=0.1, dt=0.05)
    run.prepare(plan, out)
    data = run.reference(out)
    assert data["shape"] == [5, 16]
    assert data["operator_action_error"] < 1e-13
    assert data["endpoint_check"] < 1e-13
    for arm in plan["arms"]:
        result = run.run_arm(out, arm["id"])
        initial = run.pilot.load_snapshot(out / "reference" / "initial.npz")
        state = deepcopy(initial)
        mpo = run.fx.hamiltonian_mpo(4, "xxx")
        params = run.configure(plan["protocol"], arm)
        target = capture_conservation_target(state, mpo, None, params)
        run.jt.install_correction(arm["sweep"])
        try:
            for _ in range(2):
                evolution.apply_unitary_evolution(state, mpo, params, conservation_target=target)
        finally:
            run.jt.install_correction("none")
        saved = run.pilot.load_snapshot(out / "arms" / arm["id"] / "final.npz")
        np.testing.assert_allclose(run.pilot.canonical_vector(saved), run.pilot.canonical_vector(state), atol=1e-13)
        counts = result["counts"]
        if arm["variant"].startswith("joint"):
            assert counts["fallback_attempts"] == 0
            assert counts["joint_calls"] == (4 if arm["integrator"] == "bug" else 2) * (
                4 if arm["sweep"] == "full" else 1
            )
        elif arm["variant"] == "pinH":
            assert counts["scalar_calls"] == (4 if arm["integrator"] == "bug" else 2)
            assert result["diagnostics"]["qualified_calls"] == counts["scalar_calls"]
            assert target is not None
            assert target.fallback == arm["fallback"]
    report = importlib.import_module("paper_reproduction_report")
    report.report(out)
    assert (out / "report" / "trajectory.pdf").exists()
    assert (out / "report" / "scalar_ablation.pdf").exists()
    validation = run.read(out / "report" / "validation.json")
    assert validation["complete"]
    assert not validation["production_protocol"]
    assert validation["published_comparisons"] == 0
    with pytest.raises(FileExistsError):
        run.run_arm(out, plan["arms"][0]["id"])
    (out / "arms" / plan["arms"][0]["id"] / "done.json").unlink()
    with pytest.raises(FileNotFoundError):
        report.validate(out)


def test_published_precision_and_roundoff(reproduction: ModuleType) -> None:
    """Comparison reports real discrepancies instead of forcing reproduction success."""
    assert reproduction.SCHEMA == 1
    report = importlib.import_module("paper_reproduction_report")
    assert report.comparison("a", "I", 0.06721, 0.0672, 3)["within_printed_precision_or_roundoff"]
    assert not report.comparison("a", "I", 0.068, 0.0672, 3)["within_printed_precision_or_roundoff"]
    assert report.comparison("a", "energy", 5e-14, 4.4e-15, 2)["within_printed_precision_or_roundoff"]


@pytest.mark.parametrize(("integrator", "expected_time"), [("bug", 0.05), ("tdvp", 0.1)])
def test_recovery_events_use_true_clock_and_preselection(
    reproduction: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, integrator: str, expected_time: float
) -> None:
    """Recovery logs preserve the half-step clock and qualify the input state."""
    run = reproduction
    plan = run.make_plan(length=4, cap=4, total_time=0.1, dt=0.1)
    arm = next(a for a in plan["arms"] if a["variant"] == "pinH" and a["integrator"] == integrator and a["fallback"])
    state = run.fx.initial_state(4, "tilted_neel")
    exact = run.fx.initial_state(4, "plus").to_vec()
    mpo = run.fx.hamiltonian_mpo(4, "xxx")
    vectors = np.array([exact, exact, exact])
    data = {"spin_values": [run.ref.Observables(4).values(exact).tolist()] * 3}
    target = run.conservation.EnergyTarget(0.0, 1e-13)
    with run.track_conservation() as stats:

        def fake_scan(*_args: object, **_kwargs: object) -> bool:
            stats.scalar_calls += 1
            stats.fallback_attempts += 1
            stats.scan_successes += 1
            return True

        monkeypatch.setattr(run.jt, "_stock_correction", fake_scan)
        with run.diagnostics(plan["protocol"], arm, stats, tmp_path, vectors, data, {}) as record:
            record["step"] = 1
            run.bug.apply_conservation(state, mpo, target)
        assert record["qualified_calls"] == 0
        assert record["qualified_fallbacks"] == 0
        event = record["events"][0]
        assert event["time"] == pytest.approx(expected_time)
        assert event["stage"] == "scan"
        assert event["bonds_preserved"]
        assert event["gauge_preserved"]
        assert not event["accuracy_pre"]["qualified"]
        assert (tmp_path / "event-00000-pre.npz").exists()
        assert (tmp_path / "event-00000-post.npz").exists()


@pytest.mark.parametrize("fail", [False, True])
def test_worker_respects_host_assignment_and_records_failure(
    reproduction: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fail: bool
) -> None:
    """A worker runs only its assigned arm and never retries a failed claim."""
    run = reproduction
    dispatch = importlib.import_module("paper_reproduction_dispatch")
    out = tmp_path / "run"
    plan = run.make_plan(length=4, cap=4, dt=0.05, total_time=0.1)
    run.prepare(plan, out)
    monkeypatch.setattr(dispatch.os, "nice", lambda _n: None)
    monkeypatch.setattr(dispatch.resources, "available_gib", lambda: 200)
    performed = []

    def reference(folder: Path) -> None:
        run.pilot.write_json(folder / "reference" / "done.json", {"test": True})

    def arm(folder: Path, identifier: str) -> None:
        performed.append(identifier)
        if fail:
            msg = "deliberate worker failure"
            raise RuntimeError(msg)
        destination = folder / "arms" / identifier
        destination.mkdir()
        run.pilot.write_json(destination / "done.json", {"test": True})

    monkeypatch.setattr(run, "reference", reference)
    monkeypatch.setattr(run, "run_arm", arm)
    selected = "bug-joint4-full"
    dispatch.worker(out, None, 96, [selected])
    assert performed == [selected]
    assert (out / "claims" / selected / "failed.json").exists() == fail
    assert dispatch.claim(out, [j for j in dispatch.queue(plan) if j["id"] == selected]) is None
    assert dispatch.state(out)["counts"]["queued"] == 12
    with pytest.raises(ValueError, match="assignment"):
        dispatch.host(out, 1, allowed=["does-not-exist"])
