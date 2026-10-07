# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check matched parent replay, unchanged noisy updates and guarded recovery."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from experiments import topdown_unpruned_finetuning as study
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def input_folder(tmp_path: Path) -> Path:
    """Seal a six-qubit complex-rotation parent and its pruned incumbent.

    Returns:
        A minimal authenticated input with independent exact endpoint scores.
    """
    folder = tmp_path / "input"
    target_id = "gaussian_mu0p5_sigma0p1"
    target = study.load_target(6, target_id).state_vector_copy()
    circuit = ParameterizedCircuit(
        6,
        [ParameterizedGate("ry", (0,), param_index=0), ParameterizedGate("rxx", (0, 1), param_index=1)],
    )
    theta = np.array([1.1, 0.8])
    pruned = ParameterizedCircuit(6, circuit.gates[:1], num_params=1)
    incumbent = np.array([0.4])
    score = study.legacy.matrix_score
    endpoints = {
        "ideal_before_pruning": score(circuit, theta, target, "IBM", "noiseless")["fidelity"],
        "noisy_before_fine_tuning": score(pruned, theta[:1], target, "IBM", "tjm"),
        "noisy_after_fine_tuning": score(pruned, incumbent, target, "IBM", "tjm"),
    }
    study.support.save_checkpoint(
        folder / "parent_final",
        {
            "target_id": target_id,
            "seed": 300,
            "selected_depth": 1,
            "arm": "near_zero",
            "status": "complete",
            "endpoints": endpoints,
            "resources": {"logical_two_qubit": 0},
        },
        {},
    )
    study.support.save_checkpoint(
        folder / "construction",
        {"circuit": study.support.circuit_record(circuit), "cpu_seconds": 7.0},
        {"theta": theta},
    )
    study.support.save_checkpoint(
        folder / "parent_pruning", {"circuit": study.support.circuit_record(pruned)}, {"theta": theta[:1]}
    )
    study.support.save_checkpoint(folder / "parent_refinement", {"cpu_seconds": 4.0}, {"theta": incumbent})
    study.support.write_json(
        folder / "parent_manifest.json",
        {
            "fixture_sha256": study.support.digest(study.DEFAULT_TARGET_PATH),
            "config": {
                "qubits": 6,
                "seed": 300,
                "hardware": "IBM",
                "backend": "statevector",
                "noisy_steps": 200,
                "noisy_online": 100,
                "training_paths": 100,
            },
        },
    )
    study.support.write_json(
        folder / "input.json",
        {
            "version": study.VERSION,
            "target_id": target_id,
            "seed": 300,
            "selected_depth": 1,
            "fixture_sha256": study.support.digest(study.DEFAULT_TARGET_PATH),
            "files": {str(p.relative_to(folder)): study.support.digest(p) for p in folder.rglob("*") if p.is_file()},
        },
    )
    return folder


def test_parent_corruption_and_missing_seal_are_rejected(input_folder: Path) -> None:
    """Never start training from altered angles or an uncommitted checkpoint."""
    study.authenticate_input(input_folder)
    (input_folder / "construction/arrays.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        study.authenticate_input(input_folder)
    with pytest.raises(ValueError, match="Missing committed"):
        study.required(input_folder / "absent")


def test_real_noisy_update_parity_and_resume(
    input_folder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wrapper passes the full frozen recipe and preserves unpruned coordinates."""
    _, built, arrays, target = study.authenticate_input(input_folder)
    circuit = study.support.restore_circuit(built["circuit"])
    original = study.legacy.train
    # A short numerical execution tests integration without repeating the full experiment.
    options: dict = {**study.pruning.PROTOCOL["fine_tuning"], "steps": 6, "online": 3, "require_jump_coverage": False}
    expected, trace = original(circuit, arrays["theta"], target, seed=5300, hardware="IBM", **options)
    calls = []

    def shortened(circuit: ParameterizedCircuit, theta: np.ndarray, target: np.ndarray, **settings: dict) -> tuple:
        calls.append(settings)
        assert settings == {**study.pruning.PROTOCOL["fine_tuning"], "seed": 5300, "hardware": "IBM"}
        assert len(circuit.gates) == 2
        np.testing.assert_array_equal(theta, arrays["theta"])
        return original(circuit, theta, target, seed=5300, hardware="IBM", **options)

    monkeypatch.setattr(study.legacy, "train", shortened)
    folder = tmp_path / "result"
    result = study.run_case(folder, input_folder)
    saved, values = study.required(folder / "refinement")
    np.testing.assert_array_equal(values["theta"], expected)
    assert saved["training"]["noise_banks"] == trace["noise_banks"]
    assert result["maximum_parent_replay_error"] < 1e-12
    assert result["new_construction_updates"] == 0
    assert result["historical_construction_cpu_seconds"] == 7
    assert result["noisy_unpruned_after"]["fidelity"] == pytest.approx(
        study.legacy.matrix_score(circuit, expected, target, "IBM", "tjm")["fidelity"], abs=1e-12
    )
    assert study.run_case(folder, input_folder) == result
    assert len(calls) == 1
    # Recover assessment after an interrupted final write, without repeating training.
    (folder / "final/complete.json").unlink()
    resumed = study.run_case(folder, input_folder)
    assert resumed["noisy_unpruned_after"] == result["noisy_unpruned_after"]
    assert len(calls) == 1
    (folder / "refinement/arrays.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        study.run_case(folder, input_folder)


def test_no_jump_failure_is_saved_without_resampling(
    input_folder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coverage failure stays missing and cannot become a successful zero-update arm."""
    calls = []

    def fail(*_args: object, **settings: object) -> tuple:
        calls.append(settings)
        msg = "The sampled bank has no jumps; stopped without rejecting/resampling it or changing its weights."
        raise RuntimeError(msg)

    monkeypatch.setattr(study.legacy, "train", fail)
    folder = tmp_path / "result"
    result = study.run_case(folder, input_folder)
    assert result["status"] == "no_jump_training_bank"
    assert result["noisy_unpruned_after"] is None
    assert result["change_vs_pruned_pp"] is None
    study.run_case(folder, input_folder)
    assert len(calls) == 1


def test_changed_receipt_cannot_resume(input_folder: Path, tmp_path: Path) -> None:
    """A changed plan cannot reuse the previous result directory."""
    folder = tmp_path / "result"
    study.support.write_json(folder / "manifest.json", {"version": study.VERSION, "input_sha256": "other"})
    with pytest.raises(ValueError, match="Resume input differs"):
        study.run_case(folder, input_folder)
    path = input_folder / "input.json"
    receipt = json.loads(path.read_text())
    receipt["seed"] = 400
    study.support.write_json(path, receipt)
    with pytest.raises(ValueError, match="identity"):
        study.authenticate_input(input_folder)


def test_prepare_requires_complete_panel_and_new_output(tmp_path: Path) -> None:
    """A partial panel or an occupied output is rejected before dispatch."""
    with pytest.raises(ValueError, match="all nine"):
        study.prepare(tmp_path / "missing", tmp_path / "new")
    with pytest.raises(ValueError, match="fresh"):
        study.prepare(tmp_path / "missing", tmp_path)
