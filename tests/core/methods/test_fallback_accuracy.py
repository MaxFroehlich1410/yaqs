# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Accuracy-study contracts on small synthetic fixtures, without running a study."""

from __future__ import annotations

import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.linalg import expm

from mqt.yaqs import track_conservation
from mqt.yaqs.core.methods import bug, conservation

if TYPE_CHECKING:
    from types import ModuleType

    from mqt.yaqs.core.data_structures.mps import MPS


@pytest.fixture
def accuracy(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the opt-in experiment scripts.

    Returns:
        Accuracy experiment module.
    """
    folder = Path(__file__).resolve().parents[3] / "paper" / "bug-mps-benchmarks" / "spin_conservation"
    monkeypatch.syspath_prepend(str(folder))
    return importlib.import_module("fallback_accuracy")


@pytest.mark.parametrize("name", ["power_a2_d1", "power_a1_d2", "dimer_0.2", "tfim", "xxz"])
def test_matrix_free_reference(accuracy: ModuleType, name: str) -> None:
    """MPO action, independent bit action, spectral scaling and propagation agree."""
    ref, models = accuracy.ref, accuracy.pilot.models
    pair, field, _ = ref.model_coefficients(4, name)
    dense = models.bit_matrix(pair, field).toarray()
    matrix_free = ref.SpinOperator(pair, field)
    np.testing.assert_allclose(matrix_free @ np.eye(16), dense, atol=1e-14)
    mpo, operator, metadata = ref.hamiltonians(4, name)
    assert ref.check_operators(mpo, operator) < 1e-13
    assert metadata["spectral_width_per_site"] == pytest.approx(1)
    edges = np.linalg.eigvalsh(dense)
    assert metadata["scale"] == pytest.approx(4 / (edges[-1] - edges[0]), abs=1e-13)
    v = accuracy.pilot.fx.initial_state(4, "random_product", chi=2).to_vec()
    half = ref.propagate(operator, v, 0.025)
    full = ref.propagate(operator, half, 0.025)
    np.testing.assert_allclose(full, expm(-0.05j * metadata["scale"] * dense) @ v, atol=1e-13)


def test_complex_fields_and_observables(accuracy: ModuleType) -> None:
    """Y signs and site ordering match explicit Pauli matrices, including correlations."""
    ref, models = accuracy.ref, accuracy.pilot.models
    pair = np.zeros((4, 4, 3))
    field = np.zeros((4, 3))
    field[0] = [0.2, 0.7, -0.5]
    pair[1, 3] = [-0.4, 0.8, 0.1]
    operator = ref.SpinOperator(pair, field)
    dense = models.bit_matrix(pair, field).toarray()
    np.testing.assert_allclose(operator @ np.eye(16), dense, atol=1e-14)
    assert ref.check_operators(models.pair_mpo(pair, field), operator) < 1e-13
    np.testing.assert_allclose(operator.H @ np.eye(16), dense, atol=1e-14)
    rng = np.random.default_rng(24)
    vector = rng.normal(size=16) + 1j * rng.normal(size=16)
    vector /= np.linalg.norm(vector)
    spins = [np.array([[0, 1], [1, 0]]) / 2, np.array([[0, -1j], [1j, 0]]) / 2, np.diag([1, -1]) / 2]
    expected = []
    for site in range(4):
        for spin in spins:
            factors = [spin if i == site else np.eye(2) for i in reversed(range(4))]
            matrix = np.kron(np.kron(np.kron(factors[0], factors[1]), factors[2]), factors[3])
            expected.append(np.vdot(vector, matrix @ vector).real)
    for site in range(3):
        for spin in spins:
            factors = [spin if i in {site, site + 1} else np.eye(2) for i in reversed(range(4))]
            matrix = np.kron(np.kron(np.kron(factors[0], factors[1]), factors[2]), factors[3])
            expected.append(np.vdot(vector, matrix @ vector).real)
    np.testing.assert_allclose(ref.Observables(4).values(vector), expected, atol=1e-14)


def test_singlet_preparation_and_seed(accuracy: ModuleType) -> None:
    """The dimer quench starts from the same canonical padded singlets at every cap."""
    ref = accuracy.ref
    exact = ref.singlet_product(6)
    padded = ref.singlet_product(6, seed=71, pad=True)
    repeated = ref.singlet_product(6, seed=71, pad=True)
    assert max(padded.bond_dimensions()) == 2
    assert np.linalg.norm(padded.to_vec()) == pytest.approx(1, abs=1e-14)
    for a, b in zip(padded.tensors, repeated.tensors, strict=True):
        np.testing.assert_array_equal(a, b)
    for tensor in padded.tensors[1:]:
        matrix = tensor.transpose(1, 0, 2).reshape(tensor.shape[1], -1)
        np.testing.assert_allclose(matrix @ matrix.conj().T, np.eye(matrix.shape[0]), atol=1e-14)
    assert accuracy.pilot.infidelity(exact.to_vec(), padded.to_vec()) < 1e-14
    obs = ref.Observables(6).values(exact.to_vec())
    np.testing.assert_allclose(obs[:18], 0, atol=1e-14)
    np.testing.assert_allclose(obs[18:].reshape(5, 3), [[-0.25] * 3, [0] * 3, [-0.25] * 3, [0] * 3, [-0.25] * 3])
    _, operator, _ = ref.hamiltonians(6, "dimer_0", scale=1)
    assert accuracy.pilot.energy(exact.to_vec(), operator) == pytest.approx(-2.25)


@pytest.mark.parametrize(("call", "expected"), [(1, 1.75), (2, 1.8)])
def test_event_half_step_clock(accuracy: ModuleType, call: int, expected: float) -> None:
    """Half-sweep corrections must not inherit the enclosing step's end time."""
    assert accuracy.ref.event_time(
        {"dt": 0.1, "integrator": "bug"}, {"step": 18, "call_in_step": call}
    ) == pytest.approx(expected)
    with pytest.raises(ValueError, match="position"):
        accuracy.ref.event_time({"dt": 0.1, "integrator": "bug"}, {"step": 18, "call_in_step": 3})


def source_fixture(accuracy: ModuleType, root: Path) -> Path:
    """Write two metadata-only eventful source runs, with the same prepared state.

    Returns:
        Synthetic source results directory.
    """
    root.mkdir()
    for step, dt in enumerate((0.1, 0.05), 1):
        case = accuracy.pilot.Case(
            "power_a2_d1", 4, 2, "domain_wall", dt, total_time=0.2, width_per_site=1.0, initial_chi=2
        )
        folder = root / "cases" / case.key
        (folder / "fallback").mkdir(parents=True)
        (folder / "initial.npz").write_bytes(b"metadata-only-initial")
        (folder / "fallback" / "event-00000.npz").write_bytes(b"metadata-only-event")
        event = {"index": 0, "step": step, "call_in_step": 2, "snapshot": "event-00000.npz"}
        result = {
            "case": accuracy.asdict(case),
            "status": "complete",
            "initial_vector_sha256": "same-vector",
            "hamiltonian": {"scale": 1.0},
            "arms": [{"variant": "fallback", "events": [event], "counts": {"fallback_attempts": 1}}],
        }
        accuracy.pilot.write_json(folder / "result.json", result)
    return root


def test_planning_never_computes(accuracy: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Planning deduplicates metadata and cannot trigger any numerical work."""
    source = source_fixture(accuracy, tmp_path / "source")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Planning attempted a numerical calculation")

    for name in ("hamiltonians", "propagate", "singlet_product"):
        monkeypatch.setattr(accuracy.ref, name, forbidden)
    for name in ("run_case", "audit_events", "prepare_initial"):
        monkeypatch.setattr(accuracy, name, forbidden)
    path = tmp_path / "plan.json"
    accuracy.main(["plan", "--source", str(source), "--output", str(path)])
    plan = json.loads(path.read_text())
    assert plan["summary"] == {
        "source_results": 2,
        "old_events": 2,
        "convergence_cases": 8,
        "larger_cases": 48,
        "trajectories": 168,
    }
    convergence = [s for s in plan["studies"] if s["study"] == "convergence"]
    assert {s["case"]["cap"] for s in convergence} == {2, 4}
    assert all(s["probe_times"] == [0.1] for s in convergence)
    assert len({s["initial"]["sha256"] for s in convergence}) == 1
    larger = [s for s in plan["studies"] if s["study"] == "larger"]
    assert {s["case"]["cap"] for s in larger} == {16, 32, 64}
    assert sum(s["full_rank_control"] for s in larger) == 8
    assert not list(tmp_path.rglob("manifest.json"))
    with pytest.raises(SystemExit):
        accuracy.main([])
    with pytest.raises(ValueError, match="already exists"):
        accuracy.main(["plan", "--source", str(source), "--output", str(path)])


def test_manifest_and_input_integrity(accuracy: ModuleType, tmp_path: Path) -> None:
    """Resumption pins thresholds, implementation, and original snapshot bytes."""
    source = source_fixture(accuracy, tmp_path / "source")
    plan = accuracy.make_plan(source)
    out = tmp_path / "output"
    accuracy.initialize_output(plan, out, mode="run")
    accuracy.initialize_output(plan, out, mode="run")
    changed = deepcopy(plan)
    changed["thresholds"]["infidelity"] = 0.01
    with pytest.raises(ValueError, match="Different plan"):
        accuracy.initialize_output(changed, out, mode="run")
    with pytest.raises(ValueError, match="separate output"):
        accuracy.initialize_output(plan, source / "nested", mode="run")
    (source / plan["audits"][0]["initial"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        accuracy.initialize_output(plan, tmp_path / "new-output", mode="run")


def test_matched_triple_probes_and_report(accuracy: ModuleType, tmp_path: Path) -> None:
    """Tiny unit fixture exercises actual BUG half-step hooks, references and reports."""
    case = accuracy.pilot.Case(
        "dimer_0.2", 4, 2, "singlet_product", 0.1, total_time=0.1, width_per_site=1, initial_chi=2
    )
    spec = {
        "study": "larger",
        "key": case.key,
        "case": accuracy.asdict(case),
        "initial": None,
        "scale": None,
        "probe_times": [0.05, 0.1],
        "source_cases": [],
        "full_rank_control": False,
    }
    thresholds = accuracy.ref.Accuracy()
    plan = {
        "schema": 1,
        "source_root": str(tmp_path / "unused-source"),
        "thresholds": accuracy.asdict(thresholds),
        "audits": [],
        "studies": [spec],
    }
    output = tmp_path / "output"
    before_hooks = (bug.apply_conservation, vars(bug)["_postprocess_bug_state"])
    first = accuracy.execute(plan, output)
    assert before_hooks == (bug.apply_conservation, vars(bug)["_postprocess_bug_state"])
    second = accuracy.execute(plan, output)
    assert first == second
    result = first[0]
    assert result["status"] == "complete"
    for arm in result["arms"]:
        assert arm["counts"]["scalar_calls"] == (0 if arm["variant"] == "uncorrected" else 2)
        assert [p["time"] for p in arm["probes"]] == [0.05, 0.1]
        assert len(arm["trace"]) == 2
        assert arm["reference_health"]["energy_drift"] < 1e-12
        assert max(row["bond"] for row in arm["trace"]) <= 2
    report = importlib.import_module("fallback_accuracy_report").report(output)
    assert report["complete"]
    assert report["completed_cases"] == 1
    assert (output / "accuracy_events.csv").exists()
    assert (output / "ACCURACY_REPORT.md").exists()


def test_accuracy_selection_is_pre_correction(
    accuracy: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An inaccurate input is not selected just because correction improves it."""
    case = accuracy.pilot.Case("xxx", 4, 2, "domain_wall", 0.1, width_per_site=1, initial_chi=2)
    state = accuracy.pilot.fx.initial_state(4, "domain_wall", chi=2)
    exact = accuracy.pilot.fx.initial_state(4, "plus", chi=2)
    mpo, operator, _ = accuracy.ref.hamiltonians(4, "xxx")
    clock = {"vectors": (exact.to_vec(), exact.to_vec()), "postprocess_call": 0}
    with track_conservation() as stats:

        def fake_correction(current: MPS, _mpo: object, _target: object) -> bool:
            current.tensors = deepcopy(exact.tensors)
            current.set_center(0)
            stats.scalar_calls += 1
            stats.fallback_attempts += 1
            stats.frame_successes += 1
            return True

        monkeypatch.setattr(conservation, "apply_conservation", fake_correction)
        with accuracy.accuracy_hooks(
            stats, tmp_path, operator, case, "fallback", accuracy.ref.Accuracy(), clock, []
        ) as (obs, record):
            obs["step"] = 1
            bug.apply_conservation(state, mpo, conservation.EnergyTarget(0.0, 1e-13))
    assert record["qualified_calls"] == 0
    assert record["qualified_events"] == []
    event = obs["events"][0]
    assert event["stage"] == "frame"
    assert event["physical_time"] == pytest.approx(0.05)
    assert not event["accuracy_pre"]["within_thresholds"]
    assert event["accuracy_post"]["within_thresholds"]
    assert (tmp_path / event["post_snapshot"]).exists()


def test_pooled_rates_and_missing_denominators(accuracy: ModuleType) -> None:
    """Rates use pooled call counts and preserve stage and denominator distinctions."""
    report = importlib.import_module("fallback_accuracy_report")
    arms = [
        {
            "counts": {"scalar_calls": calls, "fallback_attempts": len(stages)},
            "qualified_calls": qualified,
            "qualified_events": good,
            "events": [{"stage": stage} for stage in stages],
        }
        for calls, qualified, stages, good in [
            (100, 20, ["scan", "frame"], ["scan"]),
            (10, 2, ["infeasible"], ["infeasible"]),
        ]
    ]
    result = report.pooled_rates(arms)
    assert result["rate"] == pytest.approx(3 / 110)
    assert result["qualified_rate"] == pytest.approx(2 / 22)
    assert result["stages"]["frame"] == 1
    assert result["qualified_stages"]["frame"] == 0
    assert result["qualified_stages"]["infeasible"] == 1
    assert report.pooled_rates([])["qualified_rate"] is None
    assert accuracy.ref.Accuracy().observable == pytest.approx(1e-3)
    with pytest.raises(ValueError, match="positive"):
        accuracy.ref.Accuracy(infidelity=-1)


def test_reference_validation_and_incomplete_convergence(accuracy: ModuleType) -> None:
    """Bad references abort; matching observables alone cannot establish accuracy."""
    _, operator, _ = accuracy.ref.hamiltonians(4, "xxx", scale=1)
    v = accuracy.pilot.fx.initial_state(4, "plus", chi=2).to_vec()
    with pytest.raises(ValueError, match="reference failed"):
        accuracy.reference_health(1.01 * v, operator, 1, accuracy.pilot.energy(v, operator), 1e-9)
    report = importlib.import_module("fallback_accuracy_report")
    row = {"time": 0, "observables": [0.0], "within_thresholds": False, "infidelity": 0.5}
    a = {"trace": [row, {**row, "time": 1}]}
    b = {"trace": [row, {**row, "time": 0.5}, {**row, "time": 1}]}
    comparison = report.compare_traces(a, b, 1e-3)
    assert comparison["observable_agreement"]
    assert not comparison["both_within_reference_thresholds"]
    b["trace"].pop()
    comparison = report.compare_traces(a, b, 1e-3)
    assert not comparison["full_coarse_grid_coverage"]
    assert not comparison["observable_agreement"]


@pytest.mark.parametrize("reflected", [False, True])
def test_audit_saved_half_step_event(
    accuracy: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, reflected: bool
) -> None:
    """Audit a synthetic saved correction, including reflected gauge and true event time."""
    pilot = accuracy.pilot
    case = pilot.Case("xxx", 4, 2, "domain_wall", 0.1, total_time=0.2, width_per_site=1, initial_chi=2)
    source = tmp_path / "source"
    folder = source / "cases" / case.key
    (folder / "fallback").mkdir(parents=True)
    initial = pilot.fx.initial_state(4, "domain_wall", chi=2)
    state = pilot.fx.initial_state(4, "plus", chi=2)
    pilot.save_snapshot(folder / "initial.npz", initial)
    mpo, operator, metadata = accuracy.ref.hamiltonians(4, "xxx")
    target = pilot.energy(initial.to_vec(), operator)
    if reflected:
        state.flip_network()
        mpo = mpo.reflected()
    with (
        track_conservation() as stats,
        pilot.observe_corrections(stats, folder / "fallback", operator, case, "fallback") as obs,
    ):
        obs["step"] = 1
        bug.apply_conservation(state, mpo, conservation.EnergyTarget(target, 1e-13))
    assert stats.fallback_attempts == 1
    result = {
        "case": accuracy.asdict(case),
        "status": "complete",
        "initial_vector_sha256": "fixture",
        "hamiltonian": metadata,
        "arms": [{"variant": "fallback", "counts": stats.to_dict(), "events": obs["events"], "final_infidelity": 0.9}],
    }
    pilot.write_json(folder / "result.json", result)
    plan = accuracy.make_plan(source)
    durations = []
    original_propagate = accuracy.ref.propagate

    def propagate(operator: object, vector: np.ndarray, duration: float) -> np.ndarray:
        durations.append(duration)
        return original_propagate(operator, vector, duration)

    monkeypatch.setattr(accuracy.ref, "propagate", propagate)
    audit = accuracy.audit_events(plan, tmp_path / "audit")
    assert durations == [0.05]
    assert audit[0]["replay_valid"]
    assert audit[0]["physical_time"] == pytest.approx(0.05)
    assert audit[0]["old_step_end_time"] == pytest.approx(0.1)
    assert audit[0]["original_final_infidelity"] == pytest.approx(0.9)
    assert json.loads((tmp_path / "audit" / "audit.json").read_text())["complete"]


def test_hooks_restore_on_exception(accuracy: ModuleType, tmp_path: Path) -> None:
    """An interrupted observer cannot affect subsequent runs in the same process."""
    case = accuracy.pilot.Case("xxx", 4, 2, "domain_wall", 0.1)
    hooks = (bug.apply_conservation, vars(bug)["_postprocess_bug_state"])

    def fail() -> None:
        """Raise a fixture error.

        Raises:
            RuntimeError: Always, to exercise hook cleanup.
        """
        msg = "stop fixture"
        raise RuntimeError(msg)

    with (
        track_conservation() as stats,
        pytest.raises(RuntimeError, match="stop fixture"),
        accuracy.accuracy_hooks(stats, tmp_path, None, case, "fallback", accuracy.ref.Accuracy(), {}, []),
    ):
        fail()
    assert hooks == (bug.apply_conservation, vars(bug)["_postprocess_bug_state"])


def test_convergence_pairs_match_states_and_adjacent_parameters(accuracy: ModuleType) -> None:
    """Only adjacent one-parameter refinements of the same physical state are compared."""
    report = importlib.import_module("fallback_accuracy_report")
    rows = []
    for cap, dt, initial in [
        (2, 0.1, "same"),
        (4, 0.1, "same"),
        (8, 0.1, "same"),
        (2, 0.05, "same"),
        (2, 0.05, "different"),
    ]:
        case = accuracy.pilot.Case("xxx", 6, cap, "domain_wall", dt, total_time=0.2)
        trace = [{"time": t, "observables": [0.0], "within_thresholds": True, "infidelity": 0.0} for t in (0, 0.1, 0.2)]
        rows.append({
            "case": accuracy.asdict(case),
            "key": f"{case.key}-{initial}",
            "initial_vector_sha256": initial,
            "hamiltonian": {"scale": 1},
            "arms": [{"variant": "fallback", "trace": trace}],
        })
    pairs = report.convergence_pairs(rows, 1e-3)
    assert len(pairs) == 3
    assert sorted(p["axis"] for p in pairs) == ["cap", "cap", "dt"]
    assert all(p["observable_agreement"] and p["both_within_reference_thresholds"] for p in pairs)
    assert not any("different" in p["left"] or "different" in p["right"] for p in pairs)
    rows[1]["hamiltonian"] = {"scale": 2}
    with pytest.raises(ValueError, match="normalization"):
        report.convergence_pairs(rows, 1e-3)


def test_distributed_arms_and_exclusive_claims(accuracy: ModuleType, tmp_path: Path) -> None:
    """Independent arms reassemble the serial result and cannot be claimed twice."""
    dispatch = importlib.import_module("fallback_dispatch")
    case = accuracy.pilot.Case(
        "dimer_0.2", 4, 2, "singlet_product", 0.1, total_time=0.1, width_per_site=1, initial_chi=2
    )
    spec = {
        "study": "larger",
        "key": case.key,
        "case": accuracy.asdict(case),
        "initial": None,
        "scale": None,
        "probe_times": [0.05, 0.1],
        "source_cases": [],
        "full_rank_control": False,
    }
    thresholds = accuracy.ref.Accuracy()
    plan = {
        "schema": 1,
        "source_root": str(tmp_path / "source"),
        "thresholds": accuracy.asdict(thresholds),
        "audits": [],
        "studies": [spec],
    }
    output = tmp_path / "parallel"
    jobs = dispatch.prepare(plan, output)
    assert dispatch.verify_worker(output) == plan
    assert dispatch.prepare(plan, output) == jobs
    with ThreadPoolExecutor(max_workers=4) as executor:
        claims = list(executor.map(lambda _: dispatch.claim(output, jobs), range(6)))
    assert len({job["id"] for job in claims if job}) == 3
    assert sum(job is None for job in claims) == 3
    assert dispatch.collect(output)["completed"] == 0
    for job in jobs:
        folder = output / "jobs" / job["id"]
        accuracy.run_case(spec, plan["source_root"], folder, thresholds, variants=[job["variant"]])
        accuracy.pilot.write_json(folder / "done.json", dispatch.identity())
    assert dispatch.collect(output)["completed"] == 3
    merged = dispatch.read(output / "cases" / case.key / "result.json")
    serial = accuracy.run_case(spec, plan["source_root"], tmp_path / "serial", thresholds)
    for combined, expected in zip(merged["arms"], serial["arms"], strict=True):
        assert combined["counts"] == expected["counts"]
        for field in ("trace", "probes"):
            for actual_row, expected_row in zip(combined[field], expected[field], strict=True):
                assert actual_row.keys() == expected_row.keys()
                for key in actual_row:
                    np.testing.assert_allclose(actual_row[key], expected_row[key], atol=1e-12, rtol=1e-12)
    assert dispatch.collect(output)["completed"] == 3
    progress = dispatch.read(output / "cases" / case.key / "fallback" / "progress.json")
    assert progress["step"] == progress["steps"] == 1
    for variants in ([], ["unknown"], ["fallback", "fallback"]):
        with pytest.raises(ValueError, match="Invalid or duplicate"):
            accuracy.run_case(spec, plan["source_root"], tmp_path / "bad", thresholds, variants=variants)
    manifest = dispatch.read(output / "manifest.json")
    manifest["signature"]["runtime"]["versions"]["numpy"] = "wrong"
    accuracy.pilot.write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="dependencies differ"):
        dispatch.verify_worker(output)


def test_distributed_cpu_and_memory_admission(
    accuracy: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CPU selection avoids SMT siblings and rejects consuming reserved cores."""
    dispatch = importlib.import_module("fallback_dispatch")
    assert dispatch.accuracy is accuracy
    monkeypatch.setattr(dispatch.os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    monkeypatch.setattr(
        dispatch.subprocess,
        "check_output",
        lambda *_a, **_k: (
            "# CPU,CORE,SOCKET,NODE\n0,0,0,0\n1,1,0,0\n2,0,1,1\n3,1,1,1\n4,0,0,0\n5,1,0,0\n6,0,1,1\n7,1,1,1\n"
        ),
    )
    assert dispatch.physical_cpus() == [0, 2, 1, 3]
    monkeypatch.setattr(dispatch, "verify_worker", lambda _: {})
    with pytest.raises(ValueError, match="headroom"):
        dispatch.host(tmp_path, 4, 96)
