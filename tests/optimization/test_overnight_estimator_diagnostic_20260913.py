# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Small deterministic checks for estimator protocol, metrics, and resume."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import estimator_diagnostic as diagnostic
from experiments.overnight_state_preparation_20260913_0108 import variance
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate, StatevectorBackend

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def _problem() -> tuple[StatevectorBackend, NDArray[np.float64], CalibratedLocalDepolarizingProvider]:
    """Build a q2 fixture with asymmetric target and noncommuting gates.

    Returns:
        Backend, trainable parameters, and a test calibration.
    """
    gates = [
        ParameterizedGate("rx", (0,), param_index=0),
        ParameterizedGate("ry", (1,), param_index=1),
        ParameterizedGate("rzz", (1, 0), param_index=2),
        ParameterizedGate("cp", (0, 1), param_index=3),
    ]
    target = np.array([1, 0.2j, -0.7, 0.3 - 0.8j], dtype=complex)
    target /= np.linalg.norm(target)
    engine = StatevectorBackend(ParameterizedCircuit(2, gates), target)
    return engine, np.array([0.3, -0.7, 0.8, 0.4]), CalibratedLocalDepolarizingProvider(0.04, 0.08)


def test_anchor_is_exact_and_directional_work_matches() -> None:
    """Paired corrections cancel exactly at the anchor; their cost still counts."""
    engine, theta, calibration = _problem()
    snapshot = variance.build_snapshot(engine, theta, calibration)
    state = diagnostic.diagnose_setting(engine, theta, calibration, snapshot, seeds=[101, 301, 501], paths=4)
    assert state["exact_by_construction"]
    assert state["oracle_exact_evaluations"] == 0
    assert state["oracle_cpu_seconds"] == 0
    assert state["summary"]["cv"]["mse"] == 0
    assert state["summary"]["weighted"]["mse"] > 0
    assert state["summary"]["equal_directional_work"]
    for row in state["replications"]:
        np.testing.assert_array_equal(row["cv"]["gradient"], snapshot.estimate.gradient)
        assert row["cv"]["paths"] == row["weighted"]["paths"] == 8
        assert row["cv"]["directional_passes"] == row["weighted"]["directional_passes"] == 16
        assert row["cv"]["exact_channel_evaluations"] == row["weighted"]["exact_channel_evaluations"] == 0


def test_shifted_oracle_metrics_and_raw_gradients_are_reproducible() -> None:
    """Compare records against direct estimator APIs and independently computed norms."""
    engine, theta, calibration = _problem()
    snapshot = variance.build_snapshot(engine, theta, calibration)
    shifted = theta + 0.2 * diagnostic.direction(theta, 77)
    state = diagnostic.diagnose_setting(engine, shifted, calibration, snapshot, seeds=[101, 301], paths=4)
    oracle, loss, _ = density_loss_gradient(engine.circuit, shifted, engine.target, calibration)
    assert not state["exact_by_construction"]
    assert state["oracle_exact_evaluations"] == 1
    assert state["oracle_loss"] == loss
    np.testing.assert_array_equal(state["oracle_gradient"], oracle)
    assert np.linalg.norm(shifted - theta) == pytest.approx(0.2)
    for row in state["replications"]:
        direct = variance.weighted_estimate(engine, shifted, calibration, paths=8, seed=row["seed"])
        np.testing.assert_array_equal(row["weighted"]["gradient"], direct.gradient)
        error = np.linalg.norm(direct.gradient - oracle)
        assert row["weighted"]["squared_error"] == pytest.approx(error**2)
        assert row["weighted"]["relative_error"] == pytest.approx(error / np.linalg.norm(oracle))
        assert row["weighted"]["cosine"] == pytest.approx(
            np.dot(direct.gradient, oracle) / (np.linalg.norm(direct.gradient) * np.linalg.norm(oracle))
        )


def test_resume_preserves_completed_rows_and_complete_resume_does_no_numerical_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A paired-replication checkpoint resumes only missing deterministic estimates."""
    engine, theta, calibration = _problem()
    snapshot = variance.build_snapshot(engine, theta, calibration)
    shifted = theta + 0.05 * diagnostic.direction(theta, 81)
    saved: list[dict[str, Any]] = []

    def interrupt(state: dict[str, Any]) -> None:
        saved.append(copy.deepcopy(state))
        if len(state["replications"]) == 2:
            raise InterruptedError

    with pytest.raises(InterruptedError):
        diagnostic.diagnose_setting(
            engine, shifted, calibration, snapshot, seeds=[100, 200, 300], paths=4, checkpoint=interrupt
        )
    prior = saved[-1]
    state = diagnostic.diagnose_setting(
        engine, shifted, calibration, snapshot, seeds=[100, 200, 300], paths=4, prior=prior
    )
    assert state["replications"][:2] == prior["replications"]
    assert state["oracle_cpu_seconds"] == prior["oracle_cpu_seconds"]
    assert len(prior["replications"]) == 2
    expected = variance.control_variate_estimate(engine, shifted, calibration, snapshot, paths=4, seed=300)
    np.testing.assert_array_equal(state["replications"][2]["cv"]["gradient"], expected.gradient)

    def fail(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Completed resume must perform no numerical estimator or oracle calls.")

    monkeypatch.setattr(variance, "control_variate_estimate", fail)
    monkeypatch.setattr(variance, "weighted_estimate", fail)
    monkeypatch.setattr(diagnostic, "density_loss_gradient", fail)
    resumed = diagnostic.diagnose_setting(
        engine, shifted, calibration, snapshot, seeds=[100, 200, 300], paths=4, prior=state
    )
    assert resumed == state


def test_amortization_uses_strict_cost_weighted_mse_inequality() -> None:
    """A tie at three reuses needs four; lower MSE need not imply lower CPU."""
    cv = {
        "gradient": [2.0],
        "gradient_norm": 2.0,
        "squared_error": 1.0,
        "cosine": 1.0,
        "cpu_seconds": 2.0,
        "wall_seconds": 2.1,
        "directional_passes": 16,
        "paths": 8,
    }
    weighted = {
        "gradient": [3.0],
        "gradient_norm": 3.0,
        "squared_error": 4.0,
        "cosine": 1.0,
        "cpu_seconds": 1.0,
        "wall_seconds": 1.1,
        "directional_passes": 16,
        "paths": 8,
    }
    rows = [{"cv": cv, "weighted": weighted}] * 2
    result = diagnostic.summarize(rows, np.ones(1), 6)
    assert result["snapshot_mse_cpu_break_even_reuses"] == 4
    assert result["snapshot_cpu_only_break_even_reuses"] is None
    assert result["cv_mse_times_cpu_with_snapshot_at_setting_reuses"] == 5
    assert result["cv_over_weighted_mse_ratio"] == pytest.approx(0.25)
    assert result["cv"]["mse_standard_error"] == 0
    assert result["cv"]["mse_times_cpu"] == 2
    weighted["squared_error"] = 1
    assert diagnostic.summarize(rows, np.ones(1), 6)["snapshot_mse_cpu_break_even_reuses"] is None
    assert diagnostic.summarize(rows, np.zeros(1), 6)["cv"]["relative_rmse"] is None


@pytest.mark.parametrize("invalid", ["paths_bool", "paths_zero", "overlap", "empty", "sampled_anchor", "prior_seed"])
def test_invalid_diagnostic_settings_fail(invalid: str) -> None:
    """Reject budgets, dependence between replications, and incompatible checkpoints."""
    engine, theta, calibration = _problem()
    snapshot = variance.build_snapshot(engine, theta, calibration)
    seeds = [100, 200]
    paths = 4
    prior = None
    if invalid == "paths_bool":
        paths = True
    elif invalid == "paths_zero":
        paths = 0
    elif invalid == "overlap":
        seeds = [100, 104]
    elif invalid == "empty":
        seeds = []
    elif invalid == "sampled_anchor":
        snapshot = variance.build_snapshot(engine, theta, calibration, snapshot_paths=4)
    else:
        prior = diagnostic.diagnose_setting(engine, theta, calibration, snapshot, seeds=seeds, paths=paths)
        prior["replications"][0]["seed"] = 101
    with pytest.raises(ValueError, match=r"paths|overlap|exact snapshot|seeds changed"):
        diagnostic.diagnose_setting(engine, theta, calibration, snapshot, seeds=seeds, paths=paths, prior=prior)


@pytest.mark.parametrize("invalid", [np.array([]), np.array([np.nan]), np.array([1j]), np.ones((2, 1))])
def test_invalid_directions_fail(invalid: NDArray[np.float64]) -> None:
    """A direction is defined only for finite real nonempty vectors."""
    with pytest.raises(ValueError, match="nonempty finite real"):
        diagnostic.direction(invalid, 10)


def _frozen_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Supply checksummed q2 starts without touching any scientific artifact.

    Returns:
        Path to the isolated source record.
    """
    engine, theta, _ = _problem()
    sources = [
        {
            "target_id": target_id,
            "replica": 11,
            "candidate": {"circuit": pilot.circuit_record(engine.circuit), "theta": theta.tolist()},
        }
        for target_id in diagnostic.TARGETS
    ]
    path = tmp_path / "starts.json"
    pilot.write_record(path, {"starts": sources})
    monkeypatch.setattr(diagnostic, "_custody", lambda: {"benchmark_hashes": {"fixture": "frozen"}})
    monkeypatch.setattr(diagnostic, "_source_hashes", lambda: {"implementation": "frozen"})
    monkeypatch.setattr(diagnostic, "load_target", lambda *_: SimpleNamespace(state_vector_copy=engine.target.copy))
    return path


def test_prepare_preregisters_all_development_settings_and_separates_randomness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production defaults freeze four distances, 32 repeats, and equal work."""
    starts = _frozen_sources(tmp_path, monkeypatch)
    output = tmp_path / "diagnostic"
    assert diagnostic.main(["--output-dir", str(output), "--starts", str(starts), "--prepare-only"]) == 0
    protocol = pilot.read_record(output / "PROTOCOL.json")
    assert protocol["targets"] == list(diagnostic.TARGETS)
    assert protocol["distances"] == [0, 0.05, 0.2, 1]
    assert protocol["phase"] == "development_diagnostics"
    assert protocol["cv_paths"] == 32
    assert protocol["weighted_paths"] == 64
    assert protocol["directional_passes_per_estimate"] == 128
    seeds = cast("dict[str, list[list[int]]]", protocol["correction_seeds"])
    flattened = sorted(seed for groups in seeds.values() for group in groups for seed in group)
    assert len(flattened) == 4 * 4 * 32
    assert min(np.diff(flattened)) >= 64
    assert len(list(output.iterdir())) == 1
    assert diagnostic.prepare(output, starts) == protocol
    monkeypatch.setattr(diagnostic, "_source_hashes", lambda: {"implementation": "changed"})
    with pytest.raises(ValueError, match="source hashes changed"):
        diagnostic.prepare(output, starts)


def test_serial_q2_panel_records_checksums_costs_and_exact_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise complete orchestration using a tiny panel and frozen timing records."""
    monkeypatch.setattr(diagnostic, "TARGETS", ("test_target",))
    monkeypatch.setattr(diagnostic, "DISTANCES", (0.0, 0.05))
    monkeypatch.setattr(diagnostic, "REPLICATES", 2)
    monkeypatch.setattr(diagnostic, "CV_PATHS", 4)
    starts = _frozen_sources(tmp_path, monkeypatch)
    output = tmp_path / "diagnostic"
    protocol = diagnostic.prepare(output, starts)
    result = diagnostic.run(output, protocol)
    assert result["ledger"]["exact_snapshot_evaluations"] == 1
    assert result["ledger"]["oracle_exact_evaluations"] == 1
    assert result["ledger"]["directional_passes_per_method"] == 64
    assert result["ledger"]["snapshot_actual_reuses_per_target"] == 4
    assert len(result["settings"]) == 2
    before = {path.name: path.read_bytes() for path in output.glob("*.json")}
    assert len(before) == 5
    assert diagnostic.run(output, protocol) == result
    assert {path.name: path.read_bytes() for path in output.glob("*.json")} == before
    retained = pilot.read_record(output / "test_target__distance_1.json")
    assert pilot.digest(retained) == result["settings"][1]["run_checksum"]
    (output / "test_target__distance_1.json").write_text('{"content_checksum":"invalid"}')
    with pytest.raises(ValueError, match="checksum"):
        diagnostic.run(output, protocol)
