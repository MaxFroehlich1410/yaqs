# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Authentication, phase equivalence and recovery for saved q12 ideal endpoints."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from experiments import topdown_q12_noisy_continuation as study
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate, StatevectorBackend

if TYPE_CHECKING:
    from pathlib import Path


def simple() -> tuple:
    """Build a small entangled case for direct numerical phase comparisons.

    Returns:
        Circuit, angles and normalized complex target.
    """
    circuit = ParameterizedCircuit(
        2, [ParameterizedGate("ry", (0,), param_index=0), ParameterizedGate("rxx", (0, 1), param_index=1)]
    )
    return circuit, np.array([1.1, 0.8]), np.array([1, 1j, 1, 2j], dtype=complex) / np.sqrt(7)


def test_split_phases_match_unchanged_continuous_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sealing the online boundary leaves the original shared-bank endpoint unchanged."""
    monkeypatch.setitem(study.PROTOCOL, "online_updates", 3)
    monkeypatch.setitem(study.PROTOCOL, "batch_updates", 4)
    monkeypatch.setitem(study.PROTOCOL, "training_paths", 2000)
    circuit, theta, target = simple()
    expected, trace = study.legacy.train(
        circuit,
        theta,
        target,
        steps=7,
        online=3,
        rate=0.2,
        seed=5300,
        noisy=True,
        crn=True,
        hardware="IBM",
        num_trajectories=2000,
        compress_no_jump=True,
        record_noise_banks=True,
        require_jump_coverage=True,
        engine=StatevectorBackend(circuit, target),
    )
    identity = {"input_sha256": "example", "protocol": study.PROTOCOL}
    online = study.train_phase(tmp_path / "online", circuit, theta, target, identity, 300, online=True)
    batch = study.train_phase(tmp_path / "batch", circuit, online[1]["theta"], target, identity, 300, online=False)
    np.testing.assert_array_equal(batch[1]["theta"], expected)
    assert online[0]["training"]["noise_banks"][0]["raw_sha256"] == trace["noise_banks"][0]["raw_sha256"]
    assert batch[0]["training"]["noise_banks"] == online[0]["training"]["noise_banks"]
    monkeypatch.setattr(study.legacy, "train", lambda *_a, **_k: pytest.fail("Completed stages must be reused"))
    restored = study.train_phase(tmp_path / "batch", circuit, online[1]["theta"], target, identity, 300, online=False)
    np.testing.assert_array_equal(restored[1]["theta"], expected)
    with pytest.raises(ValueError, match="Cached phase"):
        study.train_phase(tmp_path / "batch", circuit, theta, target, identity, 300, online=False)
    with pytest.raises(ValueError, match="Cached phase"):
        study.train_phase(tmp_path / "batch", circuit, online[1]["theta"], target, identity, 300, online=True)


def test_no_jump_bank_does_not_retry_or_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty sampled bank remains a declared failure, never a resampling opportunity."""
    circuit, theta, target = simple()
    calls = []

    def failure(*_args: object, **_kwargs: object) -> None:
        calls.append(1)
        msg = "The sampled bank has no jumps"
        raise RuntimeError(msg)

    monkeypatch.setattr(study.legacy, "train", failure)
    with pytest.raises(RuntimeError, match="no jumps"):
        study.train_phase(tmp_path / "online", circuit, theta, target, {}, 300, online=True)
    assert calls == [1]
    assert study.support.load_checkpoint(tmp_path / "online") is None


def test_input_replay_source_and_payload_authentication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid seal cannot mask a changed target, source identity or endpoint score."""
    circuit = ParameterizedCircuit(12, [ParameterizedGate("ry", (0,), param_index=0)])
    target = np.zeros(4096, dtype=complex)
    target[0] = 1
    theta = np.array([0.2])
    folder = tmp_path / "inputs/example"
    fidelity = StatevectorBackend(circuit, target).metrics(theta)[1]
    study.support.save_checkpoint(
        folder / "construction", {"training": {"circuit": study.support.circuit_record(circuit)}}, {"theta": theta}
    )
    receipt = {
        "version": study.VERSION,
        "target_id": "tfim_critical",
        "ideal_fidelity": fidelity,
        "construction_seal_sha256": study.support.digest(folder / "construction/complete.json"),
    }
    study.support.write_json(folder / "input.json", receipt)
    monkeypatch.setattr(study, "sources", lambda: {"source": "original"})
    monkeypatch.setattr(study, "load_target", lambda *_args: SimpleNamespace(state_vector_copy=target.copy))

    def plan() -> None:
        study.support.write_json(
            tmp_path / "plan.json",
            {
                "protocol": study.PROTOCOL,
                "sources": study.sources(),
                "inputs": {"example": study.support.digest(folder / "input.json")},
            },
        )

    plan()
    study.authenticate_input(folder)
    receipt["ideal_fidelity"] = 0.5
    study.support.write_json(folder / "input.json", receipt)
    plan()
    with pytest.raises(ValueError, match="ideal replay"):
        study.authenticate_input(folder)
    receipt["ideal_fidelity"] = fidelity
    study.support.write_json(folder / "input.json", receipt)
    plan()
    monkeypatch.setattr(study, "sources", lambda: {"source": "changed"})
    with pytest.raises(ValueError, match="numerical source"):
        study.authenticate_input(folder)


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_prepare_rejects_incomplete_panel(tmp_path: Path, mutation: str) -> None:
    """A launch plan must cover every target and optimizer seed exactly once."""
    rows = [{"target_id": target, "seed": seed} for target in study.TARGET_IDS for seed in (300, 400, 500)]
    if mutation == "missing":
        rows.pop()
    else:
        rows[-1] = rows[0]
    reference = tmp_path / "references.json"
    reference.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="all nine targets"):
        study.prepare(reference, tmp_path / "campaign")


def test_cli_requires_explicit_inputs(tmp_path: Path) -> None:
    """Neither preparation nor numerical execution accepts implicit parent inputs."""
    with pytest.raises(SystemExit):
        study.main(["--action", "prepare", "--output", str(tmp_path)])
    with pytest.raises(SystemExit):
        study.main(["--action", "run", "--output", str(tmp_path)])


def test_complete_continuation_reuses_all_seals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The full pipeline scores fresh paired banks and authenticates completed recovery."""
    monkeypatch.setitem(study.PROTOCOL, "online_updates", 3)
    monkeypatch.setitem(study.PROTOCOL, "batch_updates", 4)
    monkeypatch.setitem(study.PROTOCOL, "training_paths", 2000)
    monkeypatch.setitem(study.PROTOCOL, "final_batches", 2)
    monkeypatch.setitem(study.PROTOCOL, "final_paths_per_batch", 2048)
    circuit, theta, target = simple()
    prepared = QuantumCircuit(2)
    prepared.ry(theta[0], 0)
    prepared.rxx(theta[1], 0, 1)
    target = Statevector.from_instruction(prepared).data
    receipt = {"target_id": "tfim_critical", "seed": 300}
    identity = {"protocol": study.PROTOCOL, "input_sha256": "example", "sources": {}}
    monkeypatch.setattr(study, "authenticate_input", lambda _folder: (receipt, circuit, theta, target, identity))
    folder = tmp_path / "run"
    result = study.run_case(folder, tmp_path / "input")
    assert result["status"] == "complete"
    assert result["training_cpu_seconds"] >= 0
    assert result["fidelity"] == result["noisy_refined"]["fidelity"]
    for index in range(2):
        record, arrays = study.required(folder / f"test_{index:02d}")
        assert len(arrays["codes"]) == 2048
        assert record["before"]["bank_sha256"] == record["after"]["bank_sha256"]
        assert record["entropy"][-1] == index
    monkeypatch.setattr(study.legacy, "train", lambda *_a, **_k: pytest.fail("No repeat training"))
    assert study.run_case(folder, tmp_path / "input") == result
    (folder / "batch/arrays.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        study.run_case(folder, tmp_path / "input")


@pytest.mark.parametrize("mutation", [None, "receipt", "fidelity", "seal"])
def test_original_parent_requires_completion_and_exact_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str | None
) -> None:
    """Original parent failures cannot pass merely because a final JSON file exists."""
    circuit = ParameterizedCircuit(12, [ParameterizedGate("ry", (0,), param_index=0)])
    target = np.zeros(4096, dtype=complex)
    target[0] = 1
    theta = np.array([0.2])
    parent = tmp_path / "jobs/tfim_critical__screened_product__seed300"
    checkpoint = parent / "depth_001/construction"
    fidelity = StatevectorBackend(circuit, target).metrics(theta)[1]
    study.support.save_checkpoint(
        checkpoint,
        {
            "fidelity": fidelity,
            "training": {"circuit": study.support.circuit_record(circuit), "depth": 1},
        },
        {"theta": theta},
    )
    study.support.write_json(
        parent / "manifest.json",
        {
            "target_id": "tfim_critical",
            "seed": 300,
            "arm": "screened_product",
            "sources": {},
            "target_sha256": study.topdown_mps.target_digest(target),
            "fixture_sha256": study.support.digest(study.DEFAULT_TARGET_PATH),
        },
    )
    study.support.save_checkpoint(
        parent / "final",
        {
            "target_id": "tfim_critical",
            "seed": 300,
            "noise_or_pruning_performed": False,
            "best_depth": 1,
            "best_fidelity": 0.5 if mutation == "fidelity" else fidelity,
            "selected_checkpoint": "depth_001/construction",
            "arm": "screened_product",
            "total_cpu_seconds": 12.0,
            "completed_depths": [
                {
                    "depth": 1,
                    "checkpoint": "depth_001/construction",
                    "checkpoint_seal_sha256": "wrong"
                    if mutation == "seal"
                    else study.support.digest(checkpoint / "complete.json"),
                }
            ],
        },
        {},
    )
    command = ["python", "run", str(parent)]
    study.support.write_json(tmp_path / "plan.json", {"jobs": [{"job_id": parent.name, "command": command}]})
    study.support.write_json(tmp_path / "claims" / parent.name / "launch.json", {"command": ["guard", "--", *command]})
    study.support.write_json(
        tmp_path / "logs" / f"{parent.name}.process.json",
        {
            "command": command,
            "status": "error" if mutation == "receipt" else "completed",
            "exit_code": 0,
        },
    )
    monkeypatch.setattr(study, "load_target", lambda *_args: SimpleNamespace(state_vector_copy=target.copy))
    if mutation is None:
        provenance, _record, arrays = study.authenticate_parent(parent, "tfim_critical", 300)
        assert provenance["ideal_fidelity"] == pytest.approx(fidelity)
        assert provenance["historical_preparation_cpu_seconds"] == pytest.approx(12.0)
        np.testing.assert_array_equal(arrays["theta"], theta)
    else:
        with pytest.raises(ValueError, match="Parent"):
            study.authenticate_parent(parent, "tfim_critical", 300)
