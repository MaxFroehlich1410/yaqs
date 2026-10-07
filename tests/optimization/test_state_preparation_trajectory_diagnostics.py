# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Dense development-study schedules, diagnostics, source audits and resume tests."""

from __future__ import annotations

from dataclasses import asdict, replace
from itertools import starmap
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider
from experiments import state_preparation_pilot as pilot
from experiments import state_preparation_trajectory_comparison as comparison
from experiments import state_preparation_trajectory_diagnostics as study
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import KrotovTJMOptions, ParameterizedCircuit, ParameterizedGate, StatevectorBackend

if TYPE_CHECKING:
    from pathlib import Path

    from mqt.yaqs.optimization.statevector import Estimator, NoiseMaps


def small_case(name: str = "small", family: str = "haar") -> dict[str, Any]:
    """Build a shared-parameter two-qubit case with complex target phases.

    Returns:
        A synthetic source case with exact reference fidelities.
    """
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), param_index=0, angle_scale=-1.3, angle_offset=0.2),
            ParameterizedGate("rx", (1,), param_index=1),
            ParameterizedGate("cx", (1, 0)),
            ParameterizedGate("rz", (0,), param_index=0, angle_scale=0.7),
        ],
    )
    rng = np.random.default_rng(42)
    target = rng.normal(size=4) + 1j * rng.normal(size=4)
    target /= np.linalg.norm(target)
    theta = np.array([0.4, -0.7])
    fidelity = density_loss_gradient(circuit, theta, target, CalibratedLocalDepolarizingProvider())[2]
    return {
        "target": comparison.target_record(name, family, target),
        "circuit": pilot.circuit_record(circuit),
        "initial": theta.tolist(),
        "checkpoints": [
            {"update": k, "theta": theta.tolist(), "exact_noisy_fidelity": fidelity} for k in (0, 2, 4, 10, 25, 50)
        ],
    }


def science(record: object) -> object:
    """Remove only timing/provenance data from a complete recursive comparison.

    Returns:
        Nested scientific fields without invocation-dependent timings.
    """
    if isinstance(record, dict):
        return {
            k: science(v) for k, v in record.items() if k not in {"cpu_seconds", "wall_seconds", "setup", "signature"}
        }
    if isinstance(record, list):
        return [science(v) for v in record]
    return record


def test_presets_and_global_epoch_boundaries() -> None:
    """Each planned screen changes only the declared sampling/update variable."""
    config = study.Config()
    tail = study.variants("tail", config)
    assert [v.rate for v in tail[:3]] == [0.1, 0.2, 0.3]
    assert [v.online_updates for v in tail] == [25, 25, 25, 50]
    assert all(v.refresh == 1 for v in tail)
    fresh, periodic, fixed = study.variants("crn", config)
    assert [fresh.epoch(k) for k in range(1, 8)] == list(range(1, 8))
    assert [periodic.epoch(k) for k in range(1, 8)] == [1, 1, 1, 1, 1, 6, 6]
    assert [fixed.epoch(k) for k in range(1, 8)] == [1] * 7
    assert [v.tail for v in study.variants("adaptive", config)] == ["batch", "momentum", "rmsprop", "adam", "adam"]
    assert study.variants("adaptive", config)[-1].online_updates == 0
    seeds = {
        study.seed_for(config, role, "a", replica, checkpoint)
        for role in ("training", "diagnostic")
        for replica in (11, 29)
        for checkpoint in (0, 25)
    }
    assert len(seeds) == 8
    for kwargs in (
        {"updates": 0},
        {"online_updates": 51},
        {"replicas": ()},
        {"diagnostic_counts": (8, 8)},
        {"diagnostic_repeats": 1},
        {"contraction": 1},
        {"armijo": float("nan")},
        {"trajectories": True},
        {"beta1": 1},
        {"epsilon": 0},
    ):
        with pytest.raises(ValueError, match="configuration"):
            study.Config(**cast("dict[str, Any]", kwargs))
    with pytest.raises(ValueError, match="variant"):
        study.Variant("../escape", 25)
    with pytest.raises(ValueError, match="variant"):
        study.Variant("bad", 25, tail="unknown")
    with pytest.raises(ValueError, match="step"):
        fixed.epoch(0)
    with pytest.raises(ValueError, match="Unknown"):
        study.variants("unknown", config)


VARIANTS = (
    study.Variant("batch", 2),
    study.Variant("all_online", 6),
    study.Variant("periodic", 2, refresh=5),
    study.Variant("fixed", 2, refresh=0),
    study.Variant("adam", 2, tail="adam", rate=0.005),
    study.Variant("pure_adam", 0, tail="adam", rate=0.005),
    study.Variant("momentum", 2, tail="momentum"),
    study.Variant("rmsprop", 2, tail="rmsprop", rate=0.005),
    study.Variant("line_search", 2, tail="backtracking", rate=8),
)


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.name)
@pytest.mark.parametrize("stop", [1, 3, 5])
def test_partial_resume_preserves_noise_phases_moments_and_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: study.Variant,
    stop: int,
) -> None:
    """Resumed science is identical across phase and five-update ensemble boundaries."""
    case = small_case()
    calibration = CalibratedLocalDepolarizingProvider(0.15, 0.3)
    provider = ExactLocalPauliProvider(calibration)
    monkeypatch.setattr(study, "ExactLocalPauliProvider", lambda: provider)
    fidelity = density_loss_gradient(
        pilot.restore_circuit(case["circuit"]),
        np.asarray(case["initial"]),
        comparison.target_vector(case["target"]),
        calibration,
    )[2]
    for checkpoint in case["checkpoints"]:
        checkpoint["exact_noisy_fidelity"] = fidelity
    config = study.Config(updates=6, online_updates=2, trajectories=3)
    path = tmp_path / "partial.json"
    partial = study.optimize(path, case, variant, config, 11, "test", stop_after=stop)
    assert partial["completed"] == stop
    resumed = study.optimize(path, case, variant, config, 11, "test")
    fresh = study.optimize(tmp_path / "fresh.json", case, variant, config, 11, "test")
    assert science(resumed) == science(fresh)
    assert study.optimize(path, case, variant, config, 11, "test") == resumed
    assert resumed["tail_step"] == 6 - variant.online_updates
    assert [r["phase"] for r in resumed["trace"][: variant.online_updates]] == ["online"] * variant.online_updates
    with pytest.raises(ValueError, match="configuration mismatch"):
        study.optimize(path, case, replace(variant, rate=0.2), config, 11, "test")


def test_interruption_and_corrupt_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Persist-before-interrupt recovery works; impossible epoch/counter records fail."""
    case, variant = small_case(), study.Variant("fixed", 2, tail="adam", refresh=0)
    config = study.Config(updates=4, online_updates=2, trajectories=2)
    write = pilot.write_record
    calls = 0

    def interrupt(path: Path, record: dict[str, object]) -> None:
        nonlocal calls
        write(path, record)
        calls += 1
        if calls == 3:
            msg = "test interruption"
            raise RuntimeError(msg)

    path = tmp_path / "partial.json"
    with monkeypatch.context() as patch:
        patch.setattr(pilot, "write_record", interrupt)
        with pytest.raises(RuntimeError, match="interruption"):
            study.optimize(path, case, variant, config, 11, "test")
    resumed = study.optimize(path, case, variant, config, 11, "test")
    assert science(resumed) == science(study.optimize(tmp_path / "full.json", case, variant, config, 11, "test"))
    resumed["trace"][1]["epoch"] = 2
    pilot.write_record(path, resumed)
    with pytest.raises(ValueError, match="progress"):
        study.optimize(path, case, variant, config, 11, "test")


def test_online_control_and_adam_tail_use_the_existing_kernels(tmp_path: Path) -> None:
    """The all-online loop and Adam tail match explicit reference compositions."""
    case = small_case()
    config = study.Config(updates=4, online_updates=2, trajectories=4)
    engine = StatevectorBackend(pilot.restore_circuit(case["circuit"]), comparison.target_vector(case["target"]))
    initial = np.asarray(case["initial"])
    seed = study.seed_for(config, "training", "small", 11)
    options = KrotovTJMOptions(num_trajectories=4, random_seed=seed)
    for variant in (study.Variant("online", 4), study.Variant("adam", 2, tail="adam", rate=0.005)):
        theta, first, second = initial.copy(), np.zeros_like(initial), np.zeros_like(initial)
        for step in range(1, 5):
            maps = engine.sample_noise_maps(initial, options, ExactLocalPauliProvider(), iteration=step)
            if step <= variant.online_updates:
                theta, _ = engine.online_update(theta, 0.7, maps)
            else:
                gradient = engine.gradient(theta, maps)[0]
                theta, first, second = study.exact.adam_step(
                    theta, gradient, first, second, step - 2, study.exact.AdamConfig(learning_rate=0.005)
                )
        actual = study.optimize(tmp_path / f"{variant.name}.json", case, variant, config, 11, "test")
        np.testing.assert_array_equal(theta, actual["theta"])
        np.testing.assert_array_equal(first, actual["first"])
        np.testing.assert_array_equal(second, actual["second"])


class Quadratic:
    """Scalar objective with explicit map-identity and evaluation-call recording."""

    def __init__(self) -> None:
        """Initialize an empty call log."""
        self.maps: list[object] = []

    def gradient(self, theta: np.ndarray, maps: object) -> tuple[np.ndarray, float, float]:
        """Return the analytic quadratic gradient and objective."""
        self.maps.append(maps)
        return 2 * theta, float(theta @ theta), 1 - float(theta @ theta)

    def metrics(self, theta: np.ndarray, maps: object) -> tuple[float, float]:
        """Return a quadratic objective and log its map identity."""
        self.maps.append(maps)
        return float(theta @ theta), 1 - float(theta @ theta)


@pytest.mark.parametrize("max_backtracks", [2, 4])
def test_backtracking_reuses_maps_counts_rejections_and_preserves_inputs(max_backtracks: int) -> None:
    """An oversized rate is rejected without losing objective-call costs or CRN."""
    engine = Quadratic()
    maps = ((),)
    theta, zero = np.ones(1), np.zeros(1)
    config = study.Config(max_backtracks=max_backtracks)
    updated, _, _, info = study.tail_update(
        cast("StatevectorBackend", engine),
        theta,
        maps,
        study.Variant("line", 0, tail="backtracking", rate=8),
        zero,
        zero,
        1,
        config,
    )
    assert all(m is maps for m in engine.maps)
    assert info["objective_evaluations"] == max_backtracks + 1
    assert len(engine.maps) == info["objective_evaluations"] + 1
    assert info["accepted"] == (max_backtracks == 4)
    np.testing.assert_array_equal(updated, [0] if info["accepted"] else [1])
    assert info["rate"] == pytest.approx(0.5 if info["accepted"] else 0)
    np.testing.assert_array_equal(theta, [1])
    np.testing.assert_array_equal(zero, [0])


@pytest.mark.parametrize("tail", ["momentum", "rmsprop"])
def test_adaptive_scaling_formula(tail: str) -> None:
    """EMA momentum and bias-corrected second moments have the documented scaling."""
    theta, first, second = np.array([0.2, -0.3]), np.array([0.1, 0.2]), np.array([0.5, 0.6])
    gradient = 2 * theta
    actual, m, v, _ = study.tail_update(
        cast("StatevectorBackend", Quadratic()),
        theta,
        ((),),
        study.Variant("tail", 0, tail=tail),
        first,
        second,
        3,
        study.Config(),
    )
    if tail == "momentum":
        expected_m = 0.9 * first + 0.1 * gradient
        np.testing.assert_allclose(actual, theta - 0.1 * expected_m)
        np.testing.assert_allclose(m, expected_m)
        np.testing.assert_array_equal(v, second)
    else:
        expected_v = 0.999 * second + 0.001 * gradient**2
        np.testing.assert_allclose(actual, theta - 0.1 * gradient / (np.sqrt(expected_v / (1 - 0.999**3)) + 1e-8))
        np.testing.assert_allclose(v, expected_v)
        np.testing.assert_array_equal(m, first)


def test_diagnostic_pairing_metrics_and_dense_only_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both estimators share each ensemble and report exact-reference error and cost."""
    case = small_case()
    config = study.Config(updates=4, online_updates=2, diagnostic_counts=(2, 4), diagnostic_repeats=2)
    engine = StatevectorBackend(pilot.restore_circuit(case["circuit"]), comparison.target_vector(case["target"]))
    theta = np.asarray(case["initial"])

    def forbid(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Dense study constructed an MPS")

    monkeypatch.setattr(MPS, "__init__", forbid)
    gradient = engine.gradient
    seen = []

    def capture(parameters: np.ndarray, maps: NoiseMaps, *, estimator: Estimator) -> tuple[np.ndarray, float, float]:
        seen.append((maps, estimator))
        return gradient(parameters, maps, estimator=estimator)

    monkeypatch.setattr(engine, "gradient", capture)
    result = study.diagnostic(engine, theta, config, "small", 25, include_cross=True)
    assert len(result["rows"]) == 8
    assert len(result["aggregate"]) == 4
    for index in range(0, 8, 2):
        assert seen[index][0] is seen[index + 1][0]
        assert seen[index][1] != seen[index + 1][1]
    for row in result["rows"]:
        expected = study.gradient_errors(np.asarray(row["gradient"]), np.asarray(result["reference_gradient"]))
        assert all(row[key] == value for key, value in expected.items())
        assert row["cpu_seconds"] == row["sampling"]["cpu_seconds"] + row["gradient_cost"]["cpu_seconds"]
    for item in result["aggregate"]:
        assert item["mse_times_cpu"] == item["mse"] * item["mean_cpu_seconds"]
    study.optimize(tmp_path / "dense.json", case, study.Variant("adam", 2, tail="adam"), config, 11, "test")
    assert study.gradient_errors(np.zeros(2), np.zeros(2))["cosine"] is None
    assert study.gradient_errors(np.ones(2), np.zeros(2))["relative_error"] is None
    assert study.nullable_mean([None, None]) is None


@pytest.mark.parametrize("stage", study.STUDIES)
def test_driver_multi_target_summary_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    """Each selected study produces only its declared jobs across all four families."""
    cases = [small_case(f"target_{i}", family) for i, family in enumerate(comparison.FAMILIES)]
    monkeypatch.setattr(study, "load_cases", lambda _path: ({"source": "test"}, cases))
    config = study.Config(
        updates=4,
        online_updates=2,
        trajectories=2,
        diagnostic_counts=(2,),
        diagnostic_repeats=2,
        checkpoints=(0, 2),
        replicas=(11,),
    )
    root = tmp_path / stage
    kwargs: dict[str, Any] = {"study": stage, "config": config, "include_cross": stage == "gradient"}
    actual = study.run(root, tmp_path, **kwargs)
    expected_jobs = 8 if stage == "gradient" else 4 * len(study.variants(stage, config))
    assert len(actual["rows"]) == expected_jobs
    assert {r["family"] for r in actual["rows"]} == set(comparison.FAMILIES)
    if stage != "gradient":
        for row in actual["rows"]:
            record = cast("dict[str, Any]", pilot.read_record(root / row["path"]))
            assert row["split_update"] == 2
            assert row["late_gain_pp"] == 100 * (
                record["evaluations"][-1]["fidelity"] - record["evaluations"][2]["fidelity"]
            )
    assert (root / "summary.md").read_text(encoding="utf-8").startswith("# Statevector development study")
    assert study.run(root, tmp_path, resume=True, **kwargs) == actual
    with pytest.raises(ValueError, match="explicit resume"):
        study.run(root, tmp_path, **kwargs)


def test_parallel_jobs_preserve_science(tmp_path: Path) -> None:
    """Worker allocation changes timings only, including fixed-CRN adaptive tails."""
    config = study.Config(updates=4, online_updates=2, trajectories=3)
    jobs = [
        study.Job(
            tmp_path / "serial",
            small_case(f"target_{i}"),
            config,
            "test",
            variant=study.Variant("fixed", 2, tail="adam", refresh=0),
            replica=i,
        )
        for i in (11, 29)
    ]
    serial = dict(study.completed_jobs(study.run_job, jobs, workers=1))
    parallel_jobs = [replace(job, root=tmp_path / "parallel") for job in jobs]
    parallel = dict(study.completed_jobs(study.run_job, parallel_jobs, workers=2))
    for index, job in enumerate(jobs):
        assert serial[index]["final_fidelity"] == parallel[index]["final_fidelity"]
        assert science(pilot.read_record(job.path)) == science(pilot.read_record(parallel_jobs[index].path))


def test_cli_defaults_and_opt_in_switches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default CLI launches diagnostics only; cross is a deliberate separate flag."""
    calls = []

    def capture(*args: object, **kwargs: object) -> dict[str, Any]:
        calls.append((args, kwargs))
        return {}

    monkeypatch.setattr(study, "run", capture)
    base = ["--source-root", str(tmp_path), "--output-root", str(tmp_path / "out")]
    assert study.main(base) == 0
    assert calls[-1][1] == {"study": "gradient", "include_cross": False, "workers": "auto", "resume": False}
    assert study.main([*base, "--include-cross", "--resume", "--workers", "2"]) == 0
    assert calls[-1][1]["include_cross"] is True
    assert calls[-1][1]["workers"] == 2


def test_source_authentication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An authentic four-target development panel is accepted; altered paths fail."""
    cases = list(starmap(small_case, zip(pilot.TARGETS, comparison.FAMILIES, strict=True)))
    starts = [
        {
            "target_id": c["target"]["target_id"],
            "replica": 11,
            "candidate": {"theta": c["initial"], "circuit": c["circuit"]},
        }
        for c in cases
    ]
    monkeypatch.setattr(study.exact, "load_starts", lambda _: ({"source": "original"}, starts))

    class Target:
        @staticmethod
        def state_vector_copy() -> np.ndarray:
            """Return the fixture vector independently of the production target file."""
            return comparison.target_vector(cases[0]["target"])

    monkeypatch.setattr(study, "load_target", lambda *_: Target())
    config = {"updates": 50, "switch": 25, "trajectories": 32}
    manifest: dict[str, Any] = {
        "version": comparison.VERSION,
        "config": config,
        "source_root": str(tmp_path),
        "source_provenance": {"source": "original"},
        "noise": ExactLocalPauliProvider().scientific_dict(),
    }
    method = asdict(comparison.Method("trajectory_hybrid", 0.7))
    pilot.write_record(tmp_path / "manifest.json", manifest)
    pilot.write_record(tmp_path / "selection.json", {})
    rows = []
    for case, source in zip(cases, starts, strict=True):
        name, family = case["target"]["target_id"], case["target"]["family"]
        target = comparison.target_record(name, family, comparison.target_vector(case["target"]), role="development")
        signature = pilot.digest({
            "start": source["candidate"],
            "target": target,
            "method": method,
            "config": config,
            "provenance": pilot.digest(manifest),
            "seed": comparison.seed_for("development", name, 11),
            "version": comparison.VERSION,
        })
        record: dict[str, Any] = {
            "signature": signature,
            "config": config,
            "method": method,
            "status": "complete",
            "completed_updates": 50,
            "trace": [{"step": k} for k in range(1, 51)],
            "evaluations": [e for e in case["checkpoints"] if e["update"] in {0, 10, 25, 50}],
            "theta": case["initial"],
        }
        path = f"development/{name}/seed_11/trajectory_hybrid_r0p7_t0p1.json"
        pilot.write_record(tmp_path / path, record)
        rows.append({
            "method": method,
            "target_id": name,
            "family": family,
            "replica": 11,
            "path": path,
            "checksum": pilot.digest(record),
        })
    result: dict[str, Any] = {
        "status": "complete",
        "manifest_checksum": pilot.digest(manifest),
        "selection_checksum": pilot.digest({}),
        "development": rows,
    }
    pilot.write_record(tmp_path / "result.json", result)
    _, loaded = study.load_cases(tmp_path)
    assert len(loaded) == 4
    assert {c["target"]["family"] for c in loaded} == set(comparison.FAMILIES)
    rows[0]["path"] = "../outside.json"
    pilot.write_record(tmp_path / "result.json", result)
    with pytest.raises(ValueError, match="audit"):
        study.load_cases(tmp_path)


def test_invalid_study_and_source_fidelity_do_not_create_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid requests and inconsistent scientific source values fail before saving results."""
    case = small_case()
    config = study.Config(updates=4, online_updates=2, trajectories=2)
    root = tmp_path / "invalid"
    with pytest.raises(ValueError, match="Unknown study"):
        study.run(root, tmp_path, study="adaptive", include_cross=True)
    assert not root.exists()
    monkeypatch.setattr(study, "load_cases", lambda _: ({}, [case]))
    with pytest.raises(ValueError, match="Missing requested"):
        study.run(root, tmp_path, config=replace(config, checkpoints=(3,)))
    assert not root.exists()
    case["checkpoints"][0]["exact_noisy_fidelity"] = -1
    with pytest.raises(ValueError, match="initial fidelity"):
        study.optimize(root / "initial.json", case, study.Variant("test", 2), config, 11, "test")
    with pytest.raises(ValueError, match="checkpoint fidelity"):
        study.run_job(study.Job(root, case, replace(config, diagnostic_counts=(2,), diagnostic_repeats=2), "test"))
    assert not root.exists()
