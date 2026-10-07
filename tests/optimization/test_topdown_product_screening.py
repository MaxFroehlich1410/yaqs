# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Validate product selection, physical preparation and unchanged noisy updates."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from experiments import topdown_product_screening as study
from mqt.yaqs.optimization import StatevectorBackend
from tests.optimization.test_topdown_mps import small_config

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def target() -> np.ndarray:
    """Return a normalized complex two-qubit state with asymmetric amplitudes."""
    return np.array([1, 4j, -2, 3j]) / np.sqrt(30)


@pytest.mark.parametrize(
    ("count", "seed", "batch_size"),
    [(0, 1, 2), (True, 1, 2), (1.5, 1, 2), (2, -1, 2), (2, True, 2), (2, 2**32, 2), (2, 1, 0), (2, 1, 2.5)],
)
def test_invalid_sampling_arguments(target: np.ndarray, count: int, seed: int, batch_size: int) -> None:
    """Reject malformed sampling settings before allocating candidate vectors."""
    with pytest.raises(ValueError, match="sample/batch"):
        study.screen_products(target, count, seed, batch_size)


@pytest.mark.parametrize("target", [np.array([1, 1]), np.zeros(4), np.array([1, np.nan]), np.ones((2, 2))])
def test_invalid_target(target: np.ndarray) -> None:
    """Reject nonnormalized, nonfinite or nonvector target states."""
    with pytest.raises(ValueError, match="normalized"):
        study.screen_products(target)


def test_selection_matches_independent_little_endian_products(target: np.ndarray) -> None:
    """Every score and the winner agree with explicit normalized Kronecker states."""
    before = target.copy()
    record, arrays = study.screen_products(target, count=37, seed=300, batch_size=7)
    scores = []
    for polar, azimuth in zip(arrays["polar"], arrays["azimuth"], strict=True):
        single = [
            np.array([np.cos(t / 2), np.exp(1j * p) * np.sin(t / 2)]) for t, p in zip(polar, azimuth, strict=True)
        ]
        vector = np.kron(single[1], single[0])
        assert np.linalg.norm(vector) == pytest.approx(1)
        scores.append(abs(np.vdot(target, vector)) ** 2)
    np.testing.assert_allclose(arrays["fidelity"], scores, atol=1e-14)
    np.testing.assert_array_equal(target, before)
    assert record["candidate_count"] == 38
    assert record["selected_index"] == int(np.argmax(scores))
    assert record["selected_product_fidelity"] == pytest.approx(max(scores))
    assert record["selected_product_fidelity"] >= record["zero_state_fidelity"]
    assert record["seed_entropy"] == [300, 730002]


def test_zero_state_is_retained_and_other_targets_select_random_products() -> None:
    """An already optimal zero start is protected, while an orthogonal start improves."""
    zero = np.array([1, 0, 0, 0], dtype=complex)
    record, _ = study.screen_products(zero)
    assert record["selected_index"] == 0
    assert record["selected_source"] == "zero"
    assert record["selected_product_fidelity"] == 1
    other, _ = study.screen_products(np.array([0, 0, 0, 1], dtype=complex))
    assert other["selected_source"] == "random_product"
    assert other["selected_product_fidelity"] > 0


def test_streaming_reproducibility_and_separate_pools(target: np.ndarray) -> None:
    """Batch size does not change random candidates; independent seeds change pools."""
    one = study.screen_products(target, 53, 300, 1)
    batch = study.screen_products(target, 53, 300, 23)
    again = study.screen_products(target, 53, 300, 23)
    different = study.screen_products(target, 53, 400, 23)
    for name in ("polar", "azimuth"):
        np.testing.assert_array_equal(one[1][name], batch[1][name])
        np.testing.assert_array_equal(batch[1][name], again[1][name])
        assert not np.array_equal(batch[1][name], different[1][name])
    np.testing.assert_allclose(one[1]["fidelity"], batch[1]["fidelity"], atol=1e-14)
    assert one[0]["selected_index"] == batch[0]["selected_index"]
    assert batch[0] == again[0]


def test_selected_state_replays_without_changing_entangling_parameters(target: np.ndarray) -> None:
    """Existing U3 gates encode the winner, preserving all paired remaining angles."""
    expected, _ = study.base.initial_parameters(target, "near_zero", 300)
    record, arrays = study.initial_parameters(target, 300, "screened_product", 1000)
    np.testing.assert_array_equal(arrays["theta"][6:], expected[6:])
    assert arrays["theta"].shape == expected.shape
    assert record["product_state_fidelity"] == pytest.approx(record["selected_product_fidelity"], abs=1e-12)
    assert record["initial_circuit_fidelity"] >= 0
    for arm in ("near_zero", "random_product"):
        _, actual = study.initial_parameters(target, 300, arm)
        old, _ = study.base.initial_parameters(target, arm, 300)
        np.testing.assert_array_equal(actual["theta"], old)
    with pytest.raises(ValueError, match="Unknown"):
        study.initial_parameters(target, 300, "bad")


def test_initialization_seals_resume_without_resampling(
    target: np.ndarray, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Commit selection once; altered inputs and corrupt arrays are rejected."""
    config = small_config("statevector")
    first = study.prepare_initialization(tmp_path, target, config, "screened_product", 23)
    monkeypatch.setattr(study, "initial_parameters", lambda *_a, **_k: pytest.fail("resampled"))
    saved = study.prepare_initialization(tmp_path, target, config, "screened_product", 23)
    assert first[0] == saved[0]
    np.testing.assert_array_equal(first[1]["theta"], saved[1]["theta"])
    with pytest.raises(ValueError, match="manifest differs"):
        study.prepare_initialization(tmp_path, target, config, "screened_product", 24)
    (tmp_path / "initialization/arrays.npz").write_bytes(b"bad checksum")
    with pytest.raises(ValueError, match="checksum"):
        study.prepare_initialization(tmp_path, target, config, "screened_product", 23)


@pytest.mark.parametrize("backend", ["statevector", "mps"])
def test_default_initial_input_reproduces_existing_full_pipeline(
    target: np.ndarray, tmp_path: Path, backend: str
) -> None:
    """Explicit near-zero parameters produce exactly the unchanged default updates."""
    config = replace(small_config(backend), training_paths=4096)
    old = study.runner.run_case(tmp_path / "old", target, config, target_id="test")
    _, arrays = study.initial_parameters(target, config.seed, "near_zero")
    explicit = study.runner.run_case(
        tmp_path / "explicit", target, config, target_id="test", initial_theta=arrays["theta"]
    )
    assert explicit["fidelity"] == old["fidelity"]
    changed = arrays["theta"].copy()
    changed[0] += 0.1
    with pytest.raises(ValueError, match="manifest"):
        study.runner.run_case(tmp_path / "explicit", target, config, target_id="test", initial_theta=changed)
    for stage in ("construction", "pruning", "refinement", "validation"):
        a = study.support.load_checkpoint(tmp_path / "old/depth_01" / stage)
        b = study.support.load_checkpoint(tmp_path / "explicit/depth_01" / stage)
        assert a is not None
        assert b is not None
        for key in a[1]:
            np.testing.assert_array_equal(a[1][key], b[1][key])


def test_full_screened_pipeline_uses_selected_angles_and_exact_noise(
    target: np.ndarray, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selected state feeds real construction, pruning, weighted noise and testing."""
    config = replace(small_config("statevector"), training_paths=4096)
    result = study.run_case(tmp_path, target, config, target_id="test", count=31)
    initial = study.support.load_checkpoint(tmp_path / "initialization")
    built = study.support.load_checkpoint(tmp_path / "training/depth_01/construction")
    assert initial is not None
    assert built is not None
    np.testing.assert_array_equal(initial[1]["theta"], built[1]["initial"])
    assert result["selected_depth"] == 1
    assert result["accounted_stage_cpu_seconds"] >= result["initialization_cpu_seconds"]
    assert result["endpoints"]["noisy_after_fine_tuning"]["fidelity"] >= 0
    val = study.support.load_checkpoint(tmp_path / "training/depth_01/validation")
    test = study.support.load_checkpoint(tmp_path / "training/test_00")
    assert val is not None
    assert test is not None
    assert val[0]["entropy"] != test[0]["entropy"]
    assert test[0]["jump_paths"] > 0
    monkeypatch.setattr(study.runner, "run_case", lambda *_a, **_k: pytest.fail("retrained"))
    assert study.run_case(tmp_path, target, config, target_id="test", count=31) == result
    with pytest.raises(ValueError, match="identity"):
        study.run_case(tmp_path, target, config, target_id="changed", count=31)


def test_initial_override_only_changes_first_depth(target: np.ndarray, tmp_path: Path) -> None:
    """Later depths inherit their trained parent with the original extension stream."""
    config = replace(small_config("statevector"), max_depth=2, training_paths=4096)
    result = study.run_case(tmp_path, target, config, target_id="test", count=13)
    assert result["status"] in {"complete", "test_jump_coverage_failure"}
    first = study.support.load_checkpoint(tmp_path / "training/depth_01/construction")
    second = study.support.load_checkpoint(tmp_path / "training/depth_02/construction")
    assert first is not None
    assert second is not None
    circuit = study.support.restore_circuit(second[0]["circuit"])
    expected = study.runner.legacy.extend(first[1]["theta"], circuit.num_params, config.seed + 1002, 0.001)
    np.testing.assert_array_equal(second[1]["initial"], expected)
    assert StatevectorBackend(circuit, target).metrics(second[1]["theta"])[1] == pytest.approx(
        second[0]["training"]["best_fidelity"]
    )


def test_no_jump_training_failure_is_recorded_without_resampling(target: np.ndarray, tmp_path: Path) -> None:
    """A genuinely zero-jump sampled training bank remains a failed candidate."""
    result = study.run_case(tmp_path, target, small_config("statevector"), target_id="test", count=31)
    refined = study.support.load_checkpoint(tmp_path / "training/depth_01/refinement")
    assert refined is not None
    assert refined[0]["status"] == "no_jump_training_bank"
    assert result["status"] == "no_valid_candidate"
    assert result["selected_depth"] is None
    assert result["endpoints"] == {}
    assert study.run_case(tmp_path, target, small_config("statevector"), target_id="test", count=31) == result


@pytest.mark.parametrize("values", [np.zeros(3), np.full(15, np.nan), np.ones(15, dtype=complex)])
def test_invalid_explicit_angles_before_output(target: np.ndarray, tmp_path: Path, values: np.ndarray) -> None:
    """Malformed initial parameter inputs cannot create a training manifest."""
    with pytest.raises(ValueError, match="Explicit initial"):
        study.runner.run_case(
            tmp_path / "invalid", target, small_config("statevector"), target_id="test", initial_theta=values
        )
    assert not (tmp_path / "invalid").exists()


def test_plan_matches_all_nine_targets_without_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Planning exposes 54 matched jobs with the baseline budgets and no launches."""
    config = study.runner.Config(backend="statevector")
    monkeypatch.setattr(study, "run_case", lambda *_a, **_k: pytest.fail("training while planning"))
    plan = study.prepare_plan(tmp_path, config)
    assert len(plan["jobs"]) == 54
    assert len({j["target_id"] for j in plan["jobs"]}) == 9
    assert {j["arm"] for j in plan["jobs"]} == {"near_zero", "screened_product"}
    assert {j["seed"] for j in plan["jobs"]} == {300, 400, 500}
    assert plan["launched"] is False
    assert len(plan["target_sha256"]) == 9
    assert not (tmp_path / "jobs").exists()
    for job in plan["jobs"]:
        command = job["command"]
        assert command[command.index("--construction-steps") + 1] == "200"
        assert command[command.index("--count") + 1] == "1000"
        assert command[command.index("--training-paths") + 1] == "100"
    assert study.prepare_plan(tmp_path, config) == plan
    with pytest.raises(ValueError, match="plan differs"):
        study.prepare_plan(tmp_path, config, 999)
    with pytest.raises(ValueError, match="six-qubit"):
        study.prepare_plan(tmp_path / "q12", replace(config, qubits=12))
    changed = study.prepare_plan(tmp_path / "explicit", replace(config, training_paths=123))
    assert all(j["command"][j["command"].index("--training-paths") + 1] == "123" for j in changed["jobs"])


def test_cli_requires_explicit_target_seed_and_has_nonlaunch_default(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Screening and training require explicit identities; default only plans."""
    with pytest.raises(SystemExit):
        study.main(["--action", "screen", "--output", str(tmp_path)])
    study.main(["--output", str(tmp_path)])
    assert json.loads(capsys.readouterr().out)["new_runs"] == 54
