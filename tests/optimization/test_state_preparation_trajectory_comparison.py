# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Experiment 2 target, update-kernel, sampling, selection and resume checks."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.execution import completed_jobs
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider
from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_pilot as pilot
from experiments import state_preparation_trajectory_comparison as experiment
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KrotovOptions,
    KrotovTJMOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    ParameterizedGate,
    noisy_state_preparation_contribution,
    train_krotov_noisy_state_preparation_hybrid,
    train_krotov_state_preparation_hybrid,
)

if TYPE_CHECKING:
    from pathlib import Path


def small_case() -> tuple[dict[str, object], dict[str, object]]:
    """Return a two-qubit case with shared parameters, phases and unequal scales."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), param_index=0, angle_scale=-1.3, angle_offset=0.2),
            ParameterizedGate("rx", (1,), param_index=1),
            ParameterizedGate("rxx", (1, 0), param_index=2),
            ParameterizedGate("rz", (0,), param_index=0, angle_scale=0.8),
        ],
    )
    rng = np.random.default_rng(44)
    target = rng.normal(size=4) + 1j * rng.normal(size=4)
    target /= np.linalg.norm(target)
    return {"circuit": pilot.circuit_record(circuit), "theta": rng.normal(size=3).tolist()}, experiment.target_record(
        "small", "haar", target
    )


def test_fresh_target_panel_mathematics_and_separation() -> None:
    """Validate normalization, quantics convention, TFIM residuals and MPS ranks."""
    panel = experiment.fresh_targets()
    assert panel == experiment.fresh_targets()
    assert len(panel) == len({row["target_id"] for row in panel}) == 8
    old = [load_target(6, name).state_vector_copy() for name in pilot.TARGETS]
    for family in experiment.FAMILIES:
        assert sum(row["family"] == family for row in panel) == 2
    for row in panel:
        vector = experiment.target_vector(row)
        assert vector.shape == (64,)
        assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-14)
        assert all(abs(np.vdot(vector, previous)) < 1 - 1e-5 for previous in old)
        metadata = cast("dict", row["metadata"])
        if row["family"] == "gaussian":
            grid = np.array([sum(((k >> q) & 1) * 2 ** (-q - 1) for q in range(6)) for k in range(64)])
            expected = np.exp(-((grid - metadata["mean"]) ** 2) / (4 * metadata["sigma"] ** 2))
            np.testing.assert_allclose(vector, expected / np.linalg.norm(expected), atol=1e-14)
        elif row["family"] == "tfim":
            indices = np.arange(64)
            applied = np.zeros(64, dtype=complex)
            for q in range(5):
                applied -= np.where(((indices >> q) & 1) == ((indices >> (q + 1)) & 1), 1, -1) * vector
            for q in range(6):
                applied -= metadata["fields"][q] * vector[indices ^ (1 << q)]
            np.testing.assert_allclose(applied, metadata["energy"] * vector, atol=1e-11)
        elif row["family"] == "mps":
            assert np.max(np.abs(vector.imag)) < 1e-14
            for cut in range(1, 6):
                singular = np.linalg.svd(vector.reshape(2**cut, -1), compute_uv=False)
                assert np.count_nonzero(singular > 1e-12) <= 2


@pytest.mark.parametrize("kind", ["trajectory_hybrid", "trajectory_batch", "noiseless_hybrid"])
def test_adapters_match_public_trainer_and_preserve_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """The lean adapter changes diagnostic cost, not the optimizer or seed sequence."""
    start, target = small_case()
    circuit = pilot.restore_circuit(cast("dict", start["circuit"]))
    initial = np.asarray(start["theta"], dtype=float)
    before = initial.copy()
    vector = experiment.target_vector(target)
    provider = ExactLocalPauliProvider(CalibratedLocalDepolarizingProvider(0.15, 0.3))
    monkeypatch.setattr(experiment, "ExactLocalPauliProvider", lambda: provider)
    config = experiment.Config(updates=5, switch=2, trajectories=3)
    # Constant rates make the public batch-only trainer the exact control.
    method = experiment.Method(kind, 0.3, 0.3)
    options = KrotovOptions(
        max_iterations=5,
        switch_iteration=0 if kind == "trajectory_batch" else 2,
        online_step_size=0.3,
        batch_step_size=0.3,
    )
    if kind == "noiseless_hybrid":
        expected = train_krotov_state_preparation_hybrid(circuit, vector, initial_theta=initial, options=options)
    else:
        expected = train_krotov_noisy_state_preparation_hybrid(
            circuit,
            vector,
            None,
            KrotovTJMOptions(num_trajectories=3, random_seed=45),
            initial_theta=initial,
            options=options,
            noise_provider=provider,
        )
    actual = experiment.optimize(tmp_path / "run.json", start, target, method, config, "test", 45)
    np.testing.assert_allclose(np.asarray(actual["theta"]), expected.theta, atol=2e-14, rtol=0)
    np.testing.assert_array_equal(initial, before)
    assert start["theta"] == before.tolist()
    assert [row["update"] for row in cast("list[dict]", actual["evaluations"])] == [0, 2, 5]


@pytest.mark.parametrize("kind", experiment.KINDS)
@pytest.mark.parametrize("stop", [1, 2, 3])
@pytest.mark.parametrize("backend", ["mps", "statevector"])
def test_exact_resume_across_switch_and_adam_bias_correction(
    tmp_path: Path, kind: str, stop: int, backend: str
) -> None:
    """Resuming either side of the switch preserves every scientific update exactly."""
    start, target = small_case()
    config = experiment.Config(4, 2, 2, backend)
    method = experiment.Method(kind, 0.2)
    path = tmp_path / "resumed.json"
    partial = experiment.optimize(path, start, target, method, config, "test", 922, stop_after=stop)
    actual = experiment.optimize(path, start, target, method, config, "test", 922)
    expected = experiment.optimize(tmp_path / "full.json", start, target, method, config, "test", 922)
    for key in ("theta", "first_moment", "second_moment"):
        np.testing.assert_array_equal(actual[key], expected[key])
    for lhs, rhs in zip(cast("list[dict]", actual["trace"]), cast("list[dict]", expected["trace"]), strict=True):
        assert {k: v for k, v in lhs.items() if not k.endswith("seconds")} == {
            k: v for k, v in rhs.items() if not k.endswith("seconds")
        }
    assert cast("list[dict]", actual["trace"])[:stop] == partial["trace"]
    assert experiment.optimize(path, start, target, method, config, "test", 922) == actual
    with pytest.raises(ValueError, match="configuration mismatch"):
        experiment.optimize(path, start, target, replace(method, rate=0.01), config, "test", 922)


def test_batch_and_adam_share_sampled_gradient_with_paired_fresh_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finite-ensemble gradient has the right sign and scale under common paths."""
    start, target = small_case()
    circuit = pilot.restore_circuit(cast("dict", start["circuit"]))
    theta = np.asarray(start["theta"], dtype=float)
    original = theta.copy()
    vector = experiment.target_vector(target)
    mps = MPS.from_statevector(vector)
    provider = ExactLocalPauliProvider(CalibratedLocalDepolarizingProvider(0.15, 0.3))
    monkeypatch.setattr(experiment, "ExactLocalPauliProvider", lambda: provider)
    config = experiment.Config(4, 2, 8)
    seed = experiment.seed_for("test", "small", 11)
    moments = (np.zeros_like(theta), np.zeros_like(theta))
    batch, _, gradient = experiment.optimizer_step(
        circuit, theta, vector, mps, moments, experiment.Method("trajectory_batch", 0.2), config, seed, 1
    )
    _, _, adam_gradient = experiment.optimizer_step(
        circuit, theta, vector, mps, moments, experiment.Method("trajectory_adam", 0.2), config, seed, 1
    )
    np.testing.assert_array_equal(adam_gradient, gradient)
    np.testing.assert_array_equal(batch, theta - 0.2 * gradient)
    np.testing.assert_array_equal(theta, original)
    for index in range(theta.size):
        shift = np.eye(theta.size)[index] * 1e-6
        losses = []
        for parameters in (theta + shift, theta - shift):
            _, loss, _, _ = noisy_state_preparation_contribution(
                circuit,
                parameters,
                vector,
                None,
                KrotovTJMOptions(num_trajectories=8, random_seed=seed),
                MPS(2),
                KrotovTruncation(),
                iteration=1,
                noise_provider=provider,
            )
            losses.append(loss)
        assert gradient[index] == pytest.approx((losses[0] - losses[1]) / 2e-6, abs=2e-8)
    _, _, refreshed = experiment.optimizer_step(
        circuit, theta, vector, mps, moments, experiment.Method("trajectory_batch", 0.2), config, seed, 2
    )
    assert not np.allclose(gradient, refreshed)
    for moment in moments:
        assert np.all(moment == 0)


def test_seed_domains_and_matched_rate_schedule() -> None:
    """Distinct roles/cases have disjoint seed blocks; rates match through the switch."""
    seeds = [
        experiment.seed_for(role, target, replica)
        for role in ("construction", "development", "confirmation")
        for target in ("a", "b")
        for replica in (11, 29)
    ]
    assert len(set(seeds)) == len(seeds)
    assert all(abs(a - b) > 1_000_003 * 50 + 32 for a in seeds for b in seeds if a != b)
    config = experiment.Config()
    hybrid = experiment.Method("trajectory_hybrid", 0.7)
    batch = experiment.Method("trajectory_batch", 0.7)
    for step in range(1, 51):
        assert hybrid.phase_rate(step, config)[1] == batch.phase_rate(step, config)[1]
        assert batch.phase_rate(step, config)[0] == "batch"
    assert hybrid.phase_rate(25, config) == ("online", 0.7)
    assert hybrid.phase_rate(26, config) == ("batch", 0.1)


def development_rows() -> list[dict[str, object]]:
    """Return complete development data with different hybrid and batch nominees."""
    return [
        {
            "method": asdict(method),
            "family": family,
            "replica": 11,
            "target_id": target_id,
            "final_fidelity": 0.8 + (method.rate / 10 if method.kind != "trajectory_batch" else -method.rate / 10),
        }
        for method in experiment.development_methods()
        for target_id, family in zip(pilot.TARGETS, experiment.FAMILIES, strict=True)
    ]


def test_nomination_is_global_and_batch_is_an_ordering_ablation() -> None:
    """Selection never replaces the matched batch rate with its independent winner."""
    rows = development_rows()
    selection = experiment.nominate(rows)
    assert cast("dict", selection["independent_batch_development_winner"])["rate"] == pytest.approx(0.1)
    chosen = cast("dict", selection["selected"])
    assert chosen["trajectory_batch"]["rate"] == pytest.approx(0.7)
    assert chosen["trajectory_hybrid"]["rate"] == pytest.approx(0.7)
    assert chosen["trajectory_adam"]["rate"] == pytest.approx(0.02)
    with pytest.raises(ValueError, match="complete"):
        experiment.nominate(rows[:-1])
    rows[0]["replica"] = 29
    with pytest.raises(ValueError, match="complete"):
        experiment.nominate(rows)
    rows = development_rows()
    rows[0]["target_id"] = "gaussian_fresh_1"
    with pytest.raises(ValueError, match="complete"):
        experiment.nominate(rows)


@pytest.mark.parametrize("invalid", [(0, 0, 1), (3, 3, 1), (3, -1, 1), (3, 1, 0), (True, 0, 1)])
def test_invalid_budgets_are_rejected(invalid: tuple[int, int, int]) -> None:
    """Invalid budgets cannot create partial artifacts."""
    with pytest.raises(ValueError, match="budget"):
        experiment.Config(*invalid)


@pytest.mark.parametrize(("kind", "rate"), [("unknown", 0.1), ("trajectory_hybrid", 0), ("exact_adam", np.nan)])
def test_invalid_methods_are_rejected(kind: str, rate: float) -> None:
    """Only supported methods with finite positive rates can run."""
    with pytest.raises(ValueError, match="method"):
        experiment.Method(kind, rate)


@pytest.mark.parametrize("backend", ["mps", "statevector"])
def test_serial_parallel_refinements_match(tmp_path: Path, backend: str) -> None:
    """Worker scheduling changes timing, never parameters or sampled streams."""
    start, target = small_case()
    config = experiment.Config(3, 1, 2, backend)
    jobs = [
        experiment.RefinementJob(
            tmp_path, "serial", start, target, replica, experiment.Method("trajectory_adam", 0.005), config, "test"
        )
        for replica in (11, 29)
    ]
    serial = dict(completed_jobs(experiment.run_refinement, jobs, workers=1))
    parallel_jobs = [replace(job, root=tmp_path / "parallel") for job in jobs]
    parallel = dict(completed_jobs(experiment.run_refinement, parallel_jobs, workers=2))
    for index in serial:
        assert serial[index]["final_fidelity"] == parallel[index]["final_fidelity"]
        lhs = pilot.read_record(jobs[index].path)
        rhs = pilot.read_record(parallel_jobs[index].path)
        assert lhs["theta"] == rhs["theta"]
        assert lhs["first_moment"] == rhs["first_moment"]


@pytest.mark.parametrize("backend", ["mps", "statevector"])
def test_construction_budget_pruning_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    """Exercise real growth/pruning plumbing with cheap deterministic training blocks."""
    calls = []

    def train(
        path: Path, circuit: ParameterizedCircuit, initial: np.ndarray, _target: np.ndarray, **kwargs: object
    ) -> dict[str, object]:
        assert kwargs["backend"] == backend
        calls.append((circuit.num_params, kwargs["updates"], kwargs["batch_rate"], kwargs["schedule"]))
        record = {"theta": initial.tolist(), "wall_seconds": 1.0}
        pilot.write_record(path, record)
        return record

    monkeypatch.setattr(experiment.hybrid, "optimize_block", train)
    target = experiment.fresh_targets()[0]
    job = experiment.ConstructionJob(tmp_path, target, 11, "test", backend)
    result = experiment.construct(job)
    circuit = pilot.restore_circuit(cast("dict", result["circuit"]))
    assert circuit.num_params == len(circuit.gates) == 108
    assert sum(len(g.sites) == 2 for g in circuit.gates) == 30
    assert len({g.logical_gate_id for g in circuit.gates}) == 108
    assert sum(int(call[1]) for call in calls) == 570
    assert len(calls) == 11
    assert [call[3] for call in calls[:4]] == [experiment.hybrid.Schedule(25, 0.7)] * 4
    assert [call[3] for call in calls[4:]] == [experiment.hybrid.Schedule()] * 7
    assert pilot.digest(experiment.construct(job)) == pilot.digest(result)
    assert len(calls) == 11
    path = tmp_path / "starts" / str(target["target_id"]) / "seed_11" / "growth_1.json"
    pilot.write_record(path, {"theta": [0], "wall_seconds": 1.0})
    with pytest.raises(ValueError, match="block checksum"):
        experiment.construct(job)


@pytest.mark.parametrize("backend", ["mps", "statevector"])
def test_orchestrator_nomination_barrier_and_completed_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """No fresh construction runs before a frozen, authenticated old-target nomination."""
    start, _ = small_case()
    old = [{"target_id": name, "replica": 11, "candidate": start} for name in pilot.TARGETS]
    monkeypatch.setattr(experiment.exact, "load_starts", lambda _: ({"fixture": "test"}, old))
    construction_calls = []
    root = tmp_path / "run"

    def run_jobs(jobs: list[experiment.RefinementJob], _workers: object) -> list[dict[str, object]]:
        assert all(job.config.backend == backend for job in jobs)
        if jobs[0].stage == "development":
            assert len(jobs) == 60
            return development_rows()
        assert len(jobs) == 80
        assert {job.target["target_id"] for job in jobs}.isdisjoint(pilot.TARGETS)
        return [{"path": str(job.path.relative_to(root))} for job in jobs]

    def construct(job: experiment.ConstructionJob) -> dict[str, object]:
        assert job.backend == backend
        assert (root / "selection.json").exists()
        construction_calls.append((job.target["target_id"], job.replica))
        return {**start, "target_id": job.target["target_id"], "replica": job.replica}

    monkeypatch.setattr(experiment, "run_jobs", run_jobs)
    monkeypatch.setattr(experiment, "construct", construct)
    selection = experiment.run_experiment(root, tmp_path, workers=1, stage="development", backend=backend)
    assert not construction_calls
    assert pilot.read_record(root / "selection.json") == selection
    result = experiment.run_experiment(root, tmp_path, workers=1, resume=True, backend=backend)
    assert len(construction_calls) == 16
    assert result["status"] == "complete"
    assert experiment.run_experiment(root, tmp_path, workers=1, resume=True, backend=backend) == result
    with pytest.raises(ValueError, match="explicit resume"):
        experiment.run_experiment(root, tmp_path, workers=1, backend=backend)
    selection["selection"] = "modified"
    pilot.write_record(root / "selection.json", selection)
    with pytest.raises(ValueError, match="Frozen selection"):
        experiment.run_experiment(root, tmp_path, workers=1, resume=True, backend=backend)


def test_cli_passes_statevector_choice_without_running_experiments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The explicit backend flag reaches orchestration without changing other options."""
    calls = []

    def run(root: Path, source: Path, **kwargs: object) -> dict[str, object]:
        calls.append((root, source, kwargs))
        return {"status": "complete"}

    monkeypatch.setattr(experiment, "run_experiment", run)
    assert (
        experiment.main([
            "--source-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "output"),
            "--backend",
            "statevector",
            "--workers",
            "1",
            "--stage",
            "development",
        ])
        == 0
    )
    assert calls == [
        (
            tmp_path / "output",
            tmp_path,
            {
                "backend": "statevector",
                "workers": 1,
                "stage": "development",
                "resume": False,
            },
        )
    ]
    assert not (tmp_path / "output").exists()


def test_corrupt_checkpoint_progress_is_rejected(tmp_path: Path) -> None:
    """A checksummed but internally inconsistent checkpoint must not be resumed."""
    start, target = small_case()
    config = experiment.Config(4, 2, 2)
    method = experiment.Method("trajectory_adam", 0.005)
    path = tmp_path / "bad.json"
    record = experiment.optimize(path, start, target, method, config, "test", 1, stop_after=2)
    record["completed_updates"] = 3
    pilot.write_record(path, record)
    with pytest.raises(ValueError, match="progress"):
        experiment.optimize(path, start, target, method, config, "test", 1)


@pytest.mark.parametrize("kind", experiment.KINDS)
def test_statevector_refinements_match_mps_without_constructing_mps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Every refinement supports dense execution, not just dense target input."""
    start, target = small_case()
    method = experiment.Method(kind, 0.3)
    config = experiment.Config(5, 2, 4)
    provider = ExactLocalPauliProvider(CalibratedLocalDepolarizingProvider(0.15, 0.3))
    monkeypatch.setattr(experiment, "ExactLocalPauliProvider", lambda: provider)
    expected = experiment.optimize(tmp_path / "mps.json", start, target, method, config, "test", 43)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Dense refinement constructed an MPS")

    monkeypatch.setattr(MPS, "__init__", forbidden)
    actual = experiment.optimize(
        tmp_path / "dense.json", start, target, method, replace(config, backend="statevector"), "test", 43
    )
    np.testing.assert_allclose(
        np.asarray(actual["theta"], dtype=float), np.asarray(expected["theta"], dtype=float), atol=2e-12
    )
    for key in ("first_moment", "second_moment"):
        np.testing.assert_allclose(
            np.asarray(actual[key], dtype=float), np.asarray(expected[key], dtype=float), atol=2e-12
        )
    with pytest.raises(ValueError, match="configuration mismatch"):
        experiment.optimize(tmp_path / "dense.json", start, target, method, config, "test", 43)
    with pytest.raises(ValueError, match="budget"):
        experiment.Config(backend="unsupported")
