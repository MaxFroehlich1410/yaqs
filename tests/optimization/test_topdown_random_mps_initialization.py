# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Validate paired starts, exact optimizer parity, stopping and sealed recovery."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from experiments import topdown_random_mps_initialization as study

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def target() -> np.ndarray:
    """Return a normalized complex target with a unique largest basis component."""
    return np.array([1, 4j, -2, 3j]) / np.sqrt(30)


@pytest.mark.parametrize(
    "change",
    [
        {"target_fidelity": 0},
        {"target_fidelity": 1.1},
        {"target_fidelity": True},
        {"target_fidelity": np.nan},
        {"online_steps": -1},
        {"online_steps": True},
        {"batch_steps": 0},
        {"batch_steps": 1.5},
        {"max_depth": 0},
        {"max_depth": "sixteen"},
        {"max_cpu_seconds": np.inf},
        {"max_cpu_seconds": True},
        {"progress_every": 0},
    ],
)
def test_invalid_budget(change: dict) -> None:
    """Invalid limits fail before writing a study manifest."""
    with pytest.raises(ValueError, match="budget"):
        replace(study.Budget(), **change).validate(300)


@pytest.mark.parametrize("seed", [-1, True, 2**32, 0.5])
def test_invalid_seed(seed: int) -> None:
    """Reject Boolean, out-of-range and noninteger random seeds."""
    with pytest.raises(ValueError, match="seed"):
        study.Budget().validate(seed)


@pytest.mark.parametrize(
    "value",
    [np.zeros(4), np.ones(3) / np.sqrt(3), np.array([1, np.nan]), np.ones((2, 2)) / 2, np.array([1])],
)
def test_invalid_target(value: np.ndarray) -> None:
    """Unnormalized or malformed amplitudes cannot define a preparation run."""
    with pytest.raises(ValueError, match="target"):
        study.validate_target(value)


def test_target_copy_and_source_fingerprint(target: np.ndarray) -> None:
    """Inputs stay private and provenance covers this driver and reused numerics."""
    copied, qubits = study.validate_target(target)
    copied[0] = 0
    assert target[0] != 0
    assert qubits == 2
    hashes = study.sources()
    assert hashes["experiments/topdown_random_mps_initialization.py"] == study.support.digest(
        study.capacity.topdown_mps.ROOT / "experiments/topdown_random_mps_initialization.py"
    )
    assert hashes["src/mqt/yaqs/optimization/statevector.py"]


def test_matched_initialization_and_historical_near_zero(target: np.ndarray) -> None:
    """Only the first U3 layer changes between arms; old RandomState draws survive."""
    circuit = study.capacity.continuation.brickwall(2, 1, initial_single_qubit_layer=True)
    values = [study.initial_parameters(target, arm, 300)[0] for arm in study.ARMS]
    expected = np.random.RandomState(1301).randn(circuit.num_params) * 0.05
    np.testing.assert_array_equal(values[0], expected)
    for theta in values[1:]:
        np.testing.assert_array_equal(theta[6:], expected[6:])
        assert not np.array_equal(theta[:6], expected[:6])
    for arm, theta in zip(study.ARMS, values, strict=True):
        np.testing.assert_array_equal(study.initial_parameters(target, arm, 300)[0], theta)
    with pytest.raises(ValueError, match="Unknown"):
        study.initial_parameters(target, "unknown", 300)


def test_largest_basis_is_little_endian_and_uses_no_extra_gates(target: np.ndarray) -> None:
    """The U3 prefix prepares basis index one, rather than the reversed bit order."""
    theta, meta = study.initial_parameters(target, "largest_basis", 300)
    np.testing.assert_allclose(theta[:6], [np.pi, 0, 0, 0, 0, 0], atol=0)
    assert meta["largest_basis_index"] == 1
    assert meta["product_state_fidelity"] == pytest.approx(16 / 30, abs=1e-14)
    circuit = study.capacity.continuation.brickwall(2, 1, initial_single_qubit_layer=True)
    zero_entanglers = theta.copy()
    zero_entanglers[6:] = 0
    basis = np.array([0, 1, 0, 0], dtype=complex)
    assert study.StatevectorBackend(circuit, basis).metrics(zero_entanglers)[1] == pytest.approx(1, abs=1e-14)
    tied = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
    assert study.initial_parameters(tied, "largest_basis", 300)[1]["largest_basis_index"] == 0


def test_random_product_matches_independent_tensor_product(target: np.ndarray) -> None:
    """Independent amplitudes validate uniform Bloch-state angles and ordering."""
    theta, meta = study.initial_parameters(target, "random_product", 400)
    rng = np.random.default_rng(np.random.SeedSequence([400, 2381]))
    polar = np.arccos(1 - 2 * rng.random(2))
    azimuth = rng.uniform(0, 2 * np.pi, 2)
    qubit_states = [
        np.array([np.cos(t / 2), np.exp(1j * p) * np.sin(t / 2)]) for t, p in zip(polar, azimuth, strict=True)
    ]
    product = np.kron(qubit_states[1], qubit_states[0])
    assert meta["product_state_fidelity"] == pytest.approx(abs(np.vdot(target, product)) ** 2, abs=1e-14)
    circuit = study.capacity.continuation.brickwall(2, 1, initial_single_qubit_layer=True)
    theta[6:] = 0
    assert study.StatevectorBackend(circuit, product).metrics(theta)[1] == pytest.approx(1, abs=1e-14)


def test_longer_hybrid_call_has_exact_optimizer_parity(target: np.ndarray, tmp_path: Path) -> None:
    """Monitoring preserves every update, true gradient and continuous curvature pair."""
    initial, _ = study.initial_parameters(target, "random_product", 300)
    circuit = study.capacity.continuation.brickwall(2, 1, initial_single_qubit_layer=True)
    budget = study.Budget(online_steps=2, batch_steps=7, target_fidelity=1, progress_every=2)
    expected, trace = study.capacity.policies.train_noiseless(
        circuit, initial, target, policy="lbfgs", steps=9, online=2, rate=0.7, stop=1
    )
    record, arrays = study.train_depth(tmp_path / "depth", target, initial, 1, budget)
    np.testing.assert_array_equal(arrays["theta"], expected)
    np.testing.assert_array_equal(arrays["fidelity"], trace["fidelity_trace"])
    updates = json.loads(arrays["optimizer_records_utf8"].tobytes())
    for actual, reference in zip(updates, trace["records"], strict=True):
        for key in ("phase", "fidelity", "history_pairs", "accepted", "trial_rates", "curvature_dot"):
            assert actual[key] == reference[key]
    assert record["training"]["online_sweeps"] == 2
    assert len(updates) == 9
    assert record["training"]["statevector_replay_error"] < 1e-14
    assert record["training"]["stage_cpu_seconds"] >= record["training"]["cpu_seconds"]
    assert study.support.load_checkpoint(tmp_path / "depth/live") is not None


def test_growth_carries_best_angles_and_paired_new_draws(target: np.ndarray, tmp_path: Path) -> None:
    """Longer optimization uses the same architecture and previous best parent."""
    budget = study.Budget(online_steps=1, batch_steps=3, max_depth=2, target_fidelity=1)
    result = study.run_case(tmp_path, target, "test", "near_zero", 300, budget)
    first = study.support.load_checkpoint(tmp_path / "depth_001/construction")
    second = study.support.load_checkpoint(tmp_path / "depth_002/construction")
    assert first is not None
    assert second is not None
    expected = study.capacity.continuation.legacy.extend(first[1]["theta"], 24, 1302, 0.001)
    np.testing.assert_array_equal(second[1][second[0]["training"]["initial_theta"]["key"]], expected)
    assert result["status"] == "budget_exhausted"
    assert result["stop_reason"] == "depth_budget"
    assert [r["depth"] for r in result["completed_depths"]] == [1, 2]
    assert all(r["batch_updates"] == 3 for r in result["completed_depths"])
    assert result["total_cpu_seconds"] == pytest.approx(
        result["setup_cpu_seconds"] + sum(r["stage_cpu_seconds"] for r in result["completed_depths"])
    )
    selected = study.support.load_checkpoint(tmp_path / result["selected_checkpoint"])
    assert selected is not None
    assert selected[0]["fidelity"] == result["best_fidelity"]


def test_initial_success_and_between_depth_cpu_budget(target: np.ndarray, tmp_path: Path) -> None:
    """Success needs no update; resource exhaustion is distinct from convergence."""
    reached = study.run_case(
        tmp_path / "success", target, "test", "largest_basis", 300, study.Budget(target_fidelity=0.01)
    )
    assert reached["status"] == "target_reached"
    assert reached["completed_depths"][0]["updates"] == 0
    assert reached["completed_depths"][0]["threshold_crossings"]["0.01"]["update"] == 0
    limited = study.run_case(
        tmp_path / "limited",
        target,
        "test",
        "near_zero",
        300,
        study.Budget(online_steps=0, batch_steps=1, target_fidelity=1, max_cpu_seconds=1e-12),
    )
    assert limited["stop_reason"] == "cpu_budget"
    assert limited["status"] == "budget_exhausted"
    assert len(limited["completed_depths"]) == 1


def test_first_accepted_threshold_stops_inside_depth(target: np.ndarray, tmp_path: Path) -> None:
    """Stopping uses exact accepted scores before display rounding."""
    initial, meta = study.initial_parameters(target, "random_product", 300)
    circuit = study.capacity.continuation.brickwall(2, 1, initial_single_qubit_layer=True)
    _, trace = study.capacity.policies.train_noiseless(circuit, initial, target, policy="lbfgs", steps=1, online=0)
    threshold = (meta["initial_circuit_fidelity"] + trace["fidelity_trace"][0]) / 2
    assert threshold > meta["initial_circuit_fidelity"]
    result = study.run_case(
        tmp_path,
        target,
        "test",
        "random_product",
        300,
        study.Budget(online_steps=0, batch_steps=5, target_fidelity=threshold),
    )
    assert result["stop_reason"] == "target_reached"
    assert result["completed_depths"][0]["updates"] == 1
    assert result["completed_depths"][0]["threshold_crossings"][str(threshold)]["update"] == 1


def test_resume_and_provenance_rejection(target: np.ndarray, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Finished histories are reused; changed settings or seals cannot be mixed."""
    budget = study.Budget(online_steps=0, batch_steps=2, max_depth=1, target_fidelity=1)
    result = study.run_case(tmp_path, target, "test", "near_zero", 300, budget)
    monkeypatch.setattr(study.capacity.policies, "train_noiseless", lambda *_a, **_k: pytest.fail("retrained"))
    assert study.run_case(tmp_path, target, "test", "near_zero", 300, budget) == result
    assert json.loads((tmp_path / "progress.json").read_text())["status"] == "budget_exhausted"
    with pytest.raises(ValueError, match="manifest"):
        study.run_case(tmp_path, target, "test", "near_zero", 300, replace(budget, batch_steps=3))
    with pytest.raises(ValueError, match="manifest"):
        study.run_case(tmp_path, target[::-1], "test", "near_zero", 300, budget)
    (tmp_path / "depth_001/construction/arrays.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        study.run_case(tmp_path, target, "test", "near_zero", 300, budget)


def test_interrupted_call_is_not_silently_restarted(
    target: np.ndarray, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live parameter snapshot cannot recover continuous L-BFGS history."""

    def interrupt(*_args: object, **_kwargs: object) -> None:
        msg = "interrupted"
        raise RuntimeError(msg)

    monkeypatch.setattr(study.capacity.policies, "train_noiseless", interrupt)
    budget = study.Budget(online_steps=0, batch_steps=2, max_depth=1)
    with pytest.raises(RuntimeError, match="interrupted"):
        study.run_case(tmp_path, target, "test", "near_zero", 300, budget)
    with pytest.raises(ValueError, match="Incomplete continuous"):
        study.run_case(tmp_path, target, "test", "near_zero", 300, budget)


def test_bad_start_or_completed_replay_is_rejected(target: np.ndarray, tmp_path: Path) -> None:
    """Seals are necessary but do not replace mathematical input/replay checks."""
    initial, _ = study.initial_parameters(target, "near_zero", 300)
    folder = tmp_path / "depth"
    study.train_depth(folder, target, initial, 1, study.Budget(online_steps=0, batch_steps=2))
    changed = initial.copy()
    changed[0] += 0.1
    with pytest.raises(ValueError, match="input"):
        study.check_depth(folder, target, changed, 1)
    saved = study.support.load_checkpoint(folder / "construction")
    assert saved is not None
    record, arrays = saved
    record["fidelity"] += 0.01
    study.support.save_checkpoint(folder / "construction", record, arrays)
    with pytest.raises(ValueError, match="replay"):
        study.check_depth(folder, target, initial, 1)


def test_source_change_during_run_rejected(target: np.ndarray, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A study cannot finish under a numerical implementation changed mid-call."""
    answers = iter([{"before": "hash"}, {"after": "hash"}])
    monkeypatch.setattr(study, "sources", lambda: next(answers))
    with pytest.raises(ValueError, match="sources changed"):
        study.run_case(tmp_path, target, "test", "near_zero", 300, study.Budget(online_steps=0, batch_steps=1))


def test_plan_and_cli_freeze_nine_independent_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Planning has no training side effects and commands carry every budget field."""
    monkeypatch.setattr(study, "run_case", lambda *_a, **_k: pytest.fail("launched during planning"))
    study.main(["--output", str(tmp_path)])
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert len(plan["jobs"]) == 9
    assert {(j["arm"], j["seed"]) for j in plan["jobs"]} == {(a, s) for a in study.ARMS for s in study.SEEDS}
    assert plan["budget"]["batch_steps"] == 2000
    assert plan["budget"]["online_steps"] == 100
    assert plan["budget"]["target_fidelity"] == pytest.approx(0.99)
    assert plan["launched"] is False
    assert study.prepare_plan(tmp_path, study.Budget()) == plan
    with pytest.raises(ValueError, match="matrix plan differs"):
        study.prepare_plan(tmp_path, study.Budget(batch_steps=2001))
    assert not (tmp_path / "jobs").exists()
    assert all(v == "1" for v in plan["environment"].values())
    for job in plan["jobs"]:
        command = job["command"]
        assert command[command.index("--action") + 1] == "run"
        assert command[command.index("--batch-steps") + 1] == "2000"
    with pytest.raises(SystemExit):
        study.main(["--action", "run", "--output", str(tmp_path)])
    calls = []
    monkeypatch.setattr(study, "run_case", lambda *args: calls.append(args) or {})
    study.main(["--action", "run", "--output", str(tmp_path), "--arm", "largest_basis", "--seed", "300"])
    assert calls[0][1].shape == (4096,)
    assert calls[0][2:5] == (study.TARGET_ID, "largest_basis", 300)
    assert calls[0][5] == study.Budget()
