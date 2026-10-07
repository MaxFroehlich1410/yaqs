# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Small-system checks for uniform CPU accounting and full geometry-driver resume."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import channels, geometry, optimizers
from experiments.overnight_state_preparation_20260913_0108 import geometry_experiment as experiment
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def _problem() -> tuple[ParameterizedCircuit, NDArray, NDArray]:
    """Return a q2 fixture with nontrivial curvature and every allowed noise profile."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("rx", (0,), param_index=0),
            ParameterizedGate("ry", (1,), param_index=1),
            ParameterizedGate("rxx", (1, 0), param_index=2),
            ParameterizedGate("rzz", (0, 1), param_index=3),
        ],
    )
    theta = np.array([0.3, -0.7, 0.8, -0.2])
    target = np.array([1, 0.4j, -0.6, 0.2 - 0.9j], dtype=complex)
    target /= np.linalg.norm(target)
    return circuit, theta, target


def _decisions(state: dict[str, Any]) -> dict[str, Any]:
    """Extract mathematical path data independently of measured timing.

    Returns:
        Parameters, moments/curvature, counters, and pending line-search state.
    """
    keys = (
        "theta",
        "gradient",
        "loss",
        "evaluations",
        "iterations",
        "status",
        "pending",
        "s_history",
        "y_history",
        "first_moment",
        "second_moment",
        "metric_evaluations",
        "rejected_evaluations",
        "nonfinite_evaluations",
    )
    return {key: state[key] for key in keys if key in state}


@pytest.mark.parametrize("method", experiment.ARMS)
def test_each_arm_counts_physical_work_and_completed_resume_is_read_only(tmp_path: Path, method: str) -> None:
    """Uniform wrappers retain all real work and restore the original geometry callables."""
    circuit, initial, target = _problem()
    original_functions = (geometry.density_loss_gradient, geometry.pure_state_metric, geometry.precondition)
    result = experiment.train_arm(tmp_path, circuit, initial, target, method, experiment.Config(17, 4))
    assert (geometry.density_loss_gradient, geometry.pure_state_metric, geometry.precondition) == original_functions
    assert result["true_channel_gradient_calls"] <= 17
    assert (
        result["physical"]["channel_attempted"]
        == result["physical"]["channel_completed"]
        == result["true_channel_gradient_calls"]
    )
    assert result["budgeted_density_direction_passes"] == 2 * result["true_channel_gradient_calls"]
    assert result["replayed_attempted_gradient_calls"] == 0
    assert result["replayed_completed_metric_calls"] == 0
    assert result["physical"]["metric_completed"] == result["metric_evaluations"]
    assert result["curve"][-1]["theta"] == result["theta"]
    assert np.min(np.diff([point["cpu_seconds"] for point in result["curve"]])) >= 0
    if method.startswith("geometry"):
        assert result["metric_evaluations"] > 0
        assert result["physical"]["linear_solves"] == result["metric_evaluations"]
        assert result["physical"]["statevector_row_operator_applications"] == result["logical_metric_row_applications"]
    else:
        assert result["metric_evaluations"] == 0
        assert result["physical"]["metric_cpu_seconds"] == result["physical"]["solve_cpu_seconds"] == 0
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.json")}
    assert experiment.train_arm(tmp_path, circuit, initial, target, method, experiment.Config(17, 4)) == result
    assert {path.name: path.read_bytes() for path in tmp_path.glob("*.json")} == before


@pytest.mark.parametrize("method", experiment.ARMS[:-1])
def test_durable_chunk_boundaries_do_not_change_uninterrupted_optimizer_path(tmp_path: Path, method: str) -> None:
    """Chunked Armijo searches preserve exact pending directions and curvature memory."""
    circuit, initial, target = _problem()
    result = experiment.train_arm(tmp_path, circuit, initial, target, method, experiment.Config(17, 4))
    if method.startswith("geometry"):
        expected = geometry.minimize(
            circuit, initial, target, damping=0.001 if method == "geometry_0p001" else 0.01, max_evaluations=17
        )
    else:

        def objective(theta: NDArray) -> tuple[float, NDArray]:
            gradient, loss, _ = density_loss_gradient(circuit, theta, target, CalibratedLocalDepolarizingProvider())
            return loss, gradient

        expected = optimizers.minimize(objective, initial, max_evaluations=17, max_iterations=17, history=10)
    assert _decisions(result["optimizer"]) == _decisions(expected)


@pytest.mark.parametrize("method", experiment.ARMS)
def test_interrupted_attempt_is_replayed_and_fully_charged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    """Metering persists calls between durable checkpoints and restores wrappers on errors."""
    circuit, initial, target = _problem()
    config = experiment.Config(17, 4)
    expected = experiment.train_arm(tmp_path / "expected", circuit, initial, target, method, config)
    original = geometry.density_loss_gradient
    calls = 0

    def interrupted(
        candidate: ParameterizedCircuit,
        theta: NDArray,
        desired: NDArray,
        calibration: CalibratedLocalDepolarizingProvider,
    ) -> tuple[NDArray, float, float]:
        nonlocal calls
        calls += 1
        if calls == 6:
            raise InterruptedError
        return original(candidate, theta, desired, calibration)

    monkeypatch.setattr(geometry, "density_loss_gradient", interrupted)
    root = tmp_path / "interrupted"
    with pytest.raises(InterruptedError):
        experiment.train_arm(root, circuit, initial, target, method, config)
    assert geometry.density_loss_gradient is interrupted
    attempts = cast("dict[str, Any]", pilot.read_record(root / "ATTEMPTS.json"))
    assert attempts["attempts"][0]["channel_attempted"] == 6
    assert attempts["attempts"][0]["channel_completed"] == 5
    monkeypatch.setattr(geometry, "density_loss_gradient", original)
    resumed = experiment.train_arm(root, circuit, initial, target, method, config)
    assert _decisions(resumed["optimizer"]) == _decisions(expected["optimizer"])
    assert resumed["replayed_attempted_gradient_calls"] == 2
    assert resumed["physical"]["channel_completed"] == resumed["true_channel_gradient_calls"] + 1
    assert resumed["physical"]["metric_completed"] >= resumed["metric_evaluations"]
    attempts = cast("dict[str, Any]", pilot.read_record(root / "ATTEMPTS.json"))
    assert resumed["cpu_seconds"] == sum(row["cpu_seconds"] for row in attempts["attempts"])


@pytest.mark.parametrize("method", experiment.ARMS)
def test_outer_cpu_scope_includes_checkpoint_work_for_every_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    """A controlled CPU clock proves identical inclusion of checkpoint/metric/solve time."""
    circuit, initial, target = _problem()
    clock = {"cpu": 0.0, "writes": 0}
    monkeypatch.setattr(experiment.time, "process_time", lambda: clock["cpu"])
    original_writer = pilot.write_record

    def write(path: Path, record: dict[str, object]) -> None:
        if path.name == "PARTIAL.json":
            clock["cpu"] += 7
            clock["writes"] += 1
        original_writer(path, record)

    def channel(
        _circuit: object, theta: NDArray, _target: object, _calibration: object
    ) -> tuple[NDArray, float, float]:
        clock["cpu"] += 2
        loss = float(theta @ theta / 2)
        return theta.copy(), loss, 1 - loss

    def metric(_circuit: object, theta: NDArray) -> tuple[NDArray, dict[str, Any]]:
        clock["cpu"] += 3
        return np.eye(theta.size), {"cpu_seconds": 3.0, "statevector_row_operator_applications": 13}

    def solve(gradient: NDArray, _metric: NDArray, _damping: float) -> tuple[NDArray, dict[str, Any]]:
        clock["cpu"] += 5
        return -gradient, {
            "cpu_seconds": 5.0,
            "slope": -float(gradient @ gradient),
            "linear_solves": 1,
            "symmetric_eigensolves": 1,
        }

    monkeypatch.setattr(pilot, "write_record", write)
    monkeypatch.setattr(geometry, "density_loss_gradient", channel)
    monkeypatch.setattr(geometry, "pure_state_metric", metric)
    monkeypatch.setattr(geometry, "precondition", solve)
    result = experiment.train_arm(tmp_path, circuit, initial, target, method, experiment.Config(3, 2))
    physical = result["physical"]
    component_cpu = sum(physical[f"{name}_cpu_seconds"] for name in ("channel", "metric", "solve"))
    assert result["cpu_seconds"] == component_cpu + 7 * clock["writes"]
    assert physical["channel_cpu_seconds"] == 2 * physical["channel_attempted"]


def test_score_cache_resumes_each_profile_and_keeps_failed_attempt_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupted fanout reuses finished profiles and retries only the missing condition."""
    circuit, initial, target = _problem()
    original = channels.evaluate
    calls = 0

    def evaluate(candidate: ParameterizedCircuit, theta: NDArray, desired: NDArray, noise_id: str) -> float:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise InterruptedError
        return original(candidate, theta, desired, noise_id)

    monkeypatch.setattr(channels, "evaluate", evaluate)
    with pytest.raises(InterruptedError):
        experiment.score_point(tmp_path, circuit, initial, target)
    monkeypatch.setattr(channels, "evaluate", original)
    result = experiment.score_point(tmp_path, circuit, initial, target)
    assert result["forward_channel_evaluations"] == result["physical_completed_forward_calls"] == 12
    assert result["physical_attempted_forward_calls"] == 13
    assert set(result["fidelities"]) == set(channels.NOISE_IDS)
    assert result["cpu_seconds"] == sum(row["cpu_seconds"] for row in result["attempts"])
    assert experiment.score_point(tmp_path, circuit, initial, target) == result


def test_common_cpu_selects_chronologically_latest_parameters_without_scores() -> None:
    """A later worse iterate is still selected because no quality-based selection is allowed."""
    curve = [
        {"theta": [index], "cpu_seconds": float(index), "local_fidelity": fidelity}
        for index, fidelity in enumerate([0.1, 0.9, 0.2])
    ]
    common, points = experiment.common_cpu_points({
        "first": {"cpu_seconds": 2, "curve": curve},
        "second": {"cpu_seconds": 3, "curve": curve},
    })
    assert common == 2
    assert points[-1]["selected"]["first"]["theta"] == [2]
    assert points[1]["selected"]["first"]["theta"] == [0]


def test_q2_four_arm_panel_has_all_scores_and_byte_stable_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run a tiny complete panel while keeping physical metric costs separate from gradient caps."""
    circuit, theta, target = _problem()
    source = {
        "target_id": "fixture",
        "replica": 11,
        "candidate": {"circuit": pilot.circuit_record(circuit), "theta": theta.tolist()},
    }
    monkeypatch.setattr(experiment, "load_target", lambda *_args: SimpleNamespace(state_vector_copy=target.copy))
    result = experiment.run_case(tmp_path, source, experiment.Config(9, 4))
    assert result["status"] == "complete"
    assert len(result["arms"]) == 4
    for arm in result["arms"].values():
        assert set(arm["final_score"]["fidelities"]) == set(channels.NOISE_IDS)
        assert arm["training"]["true_channel_gradient_calls"] <= 9
    assert len(result["common_cpu_comparison"]["points"]) == 5
    scoring = result["scoring_ledger"]
    assert scoring["forward_channel_evaluations"] == 12 * scoring["unique_parameter_points"]
    assert scoring["physical_attempted_forward_calls"] == scoring["physical_completed_forward_calls"]
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*.json")}
    assert experiment.run_case(tmp_path, source, experiment.Config(9, 4)) == result
    assert {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*.json")} == before
    experiment.write_scoreboard(tmp_path, [result])
    assert len((tmp_path / "SCOREBOARD.csv").read_text().splitlines()) == 49


def test_prepare_only_authenticates_original108_gate_sources_and_freezes_all_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scientific preparation checks metadata and custody without numerical training."""
    circuit = ParameterizedCircuit(6, [ParameterizedGate("rx", (0,), param_index=0) for _ in range(108)])
    sources: list[dict[str, Any]] = [
        {"target_id": key, "replica": 11, "candidate": {"circuit": pilot.circuit_record(circuit), "theta": [0.3]}}
        for key in experiment.TARGETS
    ]
    pilot.write_record(tmp_path / "starts.json", {"starts": sources})
    monkeypatch.setattr(experiment.runner, "OUTPUT", tmp_path)
    monkeypatch.setattr(experiment.runner, "custody", lambda: {"benchmark_hashes": {"benchmark": "frozen"}})
    monkeypatch.setattr(experiment, "numerical_sources", lambda: {"source": "frozen"})
    output = tmp_path / "prepared"
    assert experiment.main(["--output", str(output), "--prepare-only"]) == 0
    assert len(list(output.iterdir())) == 1
    protocol = cast("dict[str, Any]", pilot.read_record(output / "PROTOCOL.json"))
    assert protocol["dampings"] == [0.001, 0.01]
    assert protocol["config"]["evaluations"] == 300
    assert protocol["lbfgs_history"] == 10
    assert protocol["lbfgs_max_iterations"] == 300
    assert protocol["adam_config"]["learning_rate"] == pytest.approx(0.02)
    assert experiment.freeze_protocol(output) == protocol
    experiment.verify_protocol(protocol)
    sources[0]["candidate"]["theta"] = [0.4]
    pilot.write_record(tmp_path / "starts.json", {"starts": sources})
    with pytest.raises(ValueError, match="custody or numerical source identity changed"):
        experiment.verify_protocol(protocol)


def test_case_failure_produces_all48_failed_scoreboard_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An incomplete target cannot disappear from the reported comparison."""
    monkeypatch.setattr(experiment, "verify_protocol", lambda _protocol: None)

    def fail(*_args: object, **_kwargs: object) -> None:
        msg = "Injected geometry failure"
        raise ValueError(msg)

    monkeypatch.setattr(experiment, "run_case", fail)
    job = {
        "protocol": {"config": {"evaluations": 300, "checkpoint_every": 10}},
        "source": {"target_id": "fixture", "replica": 11},
        "root": str(tmp_path / "case"),
        "order_index": 0,
    }
    result = experiment.execute_case(job)
    assert result["status"] == "failed"
    assert len(result["arms"]) == 4
    assert experiment.execute_case(job) == result
    experiment.write_scoreboard(tmp_path, [result])
    text = (tmp_path / "SCOREBOARD.csv").read_text()
    assert len(text.splitlines()) == 49
    assert text.count("Injected geometry failure") == 48


def test_failed_metric_arms_keep_their_attempt_cost_while_baselines_finish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsuccessful metric is charged and does not prevent independent controls."""
    circuit, theta, target = _problem()
    source = {
        "target_id": "fixture",
        "replica": 11,
        "candidate": {"circuit": pilot.circuit_record(circuit), "theta": theta.tolist()},
    }
    monkeypatch.setattr(experiment, "load_target", lambda *_args: SimpleNamespace(state_vector_copy=target.copy))

    def fail(_circuit: ParameterizedCircuit, _theta: NDArray) -> tuple[NDArray, dict[str, Any]]:
        msg = "Injected metric failure"
        raise ValueError(msg)

    monkeypatch.setattr(geometry, "pure_state_metric", fail)
    result = experiment.run_case(tmp_path, source, experiment.Config(5, 2))
    assert result["status"] == "complete_with_failed_arms"
    assert result["common_cpu_comparison"] is None
    for method in experiment.ARMS[:2]:
        arm = result["arms"][method]
        assert arm["status"] == "failed"
        physical = arm["partial_training_cost"]["physical"]
        assert physical["metric_attempted"] == 1
        assert physical["metric_completed"] == 0
        assert physical["channel_completed"] == 1
    for method in experiment.ARMS[2:]:
        assert result["arms"][method]["status"] == "complete"
