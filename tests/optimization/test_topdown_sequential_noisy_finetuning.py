# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Archived cross-update parity, shared bank, lineage checks and stage recovery."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from experiments import topdown_sequential_noisy_finetuning as study
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def input_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Seal an entangling six-qubit input while mocking original-parent authentication.

    Returns:
        Sealed sequential checkpoint for real short noisy integration tests.
    """
    circuit = ParameterizedCircuit(
        6,
        [
            ParameterizedGate(name, sites, param_index=i, angle_scale=-0.7, angle_offset=0.2)
            for i, (name, sites) in enumerate([
                ("ry", (0,)),
                ("rxx", (0, 1)),
                ("rz", (1,)),
                ("ry", (1,)),
                ("rzz", (0, 1)),
                ("rx", (0,)),
                ("ryy", (0, 1)),
                ("rxx", (0, 1)),
            ])
        ],
    )
    theta = np.linspace(-0.7, 0.8, 8)
    target = np.zeros(64, dtype=complex)
    target[:4] = np.array([1, 2j, -3, 4j]) / np.sqrt(30)
    folder = tmp_path / "input"
    parent = folder / "parent"
    parent.mkdir(parents=True)
    (parent / "input.json").write_text("{}")
    receipt = {"target_id": "synthetic", "seed": 300, "selected_depth": 1}
    monkeypatch.setattr(study.parents, "authenticate_input", lambda _: (receipt, {}, {}, target))
    identity = {
        "protocol": study.screen.PROTOCOL,
        "method": "sequential_ideal",
        "input_sha256": study.support.digest(parent / "input.json"),
    }
    study.support.save_checkpoint(
        folder / "sequential_pruning",
        {"identity": identity, "circuit": study.support.circuit_record(circuit), "trace": {"repair_updates": 0}},
        {"theta": theta},
    )
    ideal = study.legacy.matrix_score(circuit, theta, target, "IBM", "noiseless")
    noisy = study.legacy.matrix_score(circuit, theta, target, "IBM", "tjm")
    study.support.save_checkpoint(
        folder / "sequential_assessment",
        {
            **receipt,
            "method": "sequential_ideal",
            "input_sha256": identity["input_sha256"],
            "pruning_seal_sha256": study.support.digest(folder / "sequential_pruning/complete.json"),
            "ideal_pruned": ideal,
            "noisy_pruned": noisy,
        },
        {},
    )
    study.support.save_checkpoint(parent / "parent_final", {"endpoints": {"noisy_after_fine_tuning": noisy}}, {})
    study.support.write_json(
        folder / "input.json",
        {
            "version": study.VERSION,
            **receipt,
            "files": {str(p.relative_to(folder)): study.support.digest(p) for p in folder.rglob("*") if p.is_file()},
        },
    )
    return folder


def test_split_phases_match_one_archived_call_and_recover(
    input_folder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both phases retain cross updates, weights, all traces, exact endpoints and bank identity."""
    monkeypatch.setattr(
        study, "PROTOCOL", {**study.PROTOCOL, "online_updates": 3, "batch_updates": 5, "total_updates": 8}
    )
    receipt, pruned, arrays, target = study.authenticate_input(input_folder)
    circuit = study.support.restore_circuit(pruned["circuit"])
    expected, old = study.legacy.train(
        circuit,
        arrays["theta"],
        target,
        steps=8,
        online=3,
        rate=0.2,
        seed=receipt["seed"] + 5000,
        noisy=True,
        crn=True,
        hardware="IBM",
        num_trajectories=100,
        compress_no_jump=True,
        record_noise_banks=True,
        require_jump_coverage=True,
    )
    folder = tmp_path / "result"
    result = study.run_case(folder, input_folder)
    online, online_arrays = study.parents.required(folder / "online")
    batch, values = study.parents.required(folder / "batch")
    np.testing.assert_array_equal(values["theta"], expected)
    np.testing.assert_array_equal(batch["training"]["fidelity_trace"], old["fidelity_trace"][3:])
    np.testing.assert_array_equal(online["training"]["fidelity_trace"], old["fidelity_trace"][:3])
    np.testing.assert_array_equal(values["initial"], online_arrays["theta"])
    assert online["training"]["noise_banks"] == batch["training"]["noise_banks"] == old["noise_banks"]
    assert result["noise_bank_sha256"] == old["noise_banks"][0]["raw_sha256"]
    assert result["online_updates"] == 3
    assert result["batch_updates"] == 5
    assert result["new_construction_updates"] == result["new_pruning_updates"] == 0
    assert result["maximum_parent_replay_error"] < 1e-12
    assert result["noisy_after_batch"] == study.legacy.matrix_score(circuit, expected, target, "IBM", "tjm")
    monkeypatch.setattr(study.legacy, "train", lambda *_args, **_kwargs: pytest.fail("Do not repeat sealed training."))
    assert study.run_case(folder, input_folder) == result
    (folder / "final/complete.json").unlink()
    assert study.run_case(folder, input_folder)["noisy_after_batch"] == result["noisy_after_batch"]
    (folder / "batch/arrays.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        study.run_case(folder, input_folder)


def test_changed_input_and_protocol_are_rejected(input_folder: Path, tmp_path: Path) -> None:
    """Reject incompatible recovery identities and changed checkpoint bytes."""
    folder = tmp_path / "result"
    study.support.write_json(folder / "manifest.json", {"wrong": "protocol"})
    with pytest.raises(ValueError, match="protocol or input"):
        study.run_case(folder, input_folder)
    (input_folder / "sequential_pruning/arrays.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        study.authenticate_input(input_folder)


def test_wrong_pruning_method_is_rejected_even_with_valid_seals(input_folder: Path) -> None:
    """Input authentication must establish the sequential lineage, not only valid bytes."""
    record, arrays = study.parents.required(input_folder / "sequential_pruning")
    record["identity"]["method"] = "B"
    study.support.save_checkpoint(input_folder / "sequential_pruning", record, arrays)
    receipt = json.loads((input_folder / "input.json").read_text())
    receipt["files"] = {name: study.support.digest(input_folder / name) for name in receipt["files"]}
    study.support.write_json(input_folder / "input.json", receipt)
    with pytest.raises(ValueError, match="lineage"):
        study.authenticate_input(input_folder)


def test_prepare_requires_complete_panel_and_fresh_output(tmp_path: Path) -> None:
    """Incomplete targets or accidental overwrite never produce a launchable panel."""
    parent = tmp_path / "parent"
    study.support.write_json(parent / "plan.json", {"jobs": []})
    with pytest.raises(ValueError, match="nine targets"):
        study.prepare(parent, tmp_path / "output")
    with pytest.raises(ValueError, match="fresh"):
        study.prepare(parent, parent)


def test_prepare_copies_all_three_seeds_and_authenticates_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Copy the full 27-case panel without constructing, pruning or optimizing again."""
    parent = tmp_path / "parent"
    jobs, inputs = [], {}
    for target in range(9):
        for seed in [300, 400, 500]:
            identity = f"target{target}__seed{seed}"
            jobs.append({"job_id": identity, "target_id": f"target{target}", "seed": seed, "depth": 2})
            study.support.write_json(parent / "inputs" / identity / "input.json", {"case": identity})
            inputs[identity] = study.support.digest(parent / "inputs" / identity / "input.json")
            for stage in ["pruning", "final"]:
                study.support.write_json(
                    parent / "retrieved/jobs" / identity / "sequential_ideal" / stage / "record.json",
                    {"case": identity, "stage": stage},
                )
    study.support.write_json(parent / "plan.json", {"jobs": jobs, "inputs": inputs})
    seen = []
    monkeypatch.setattr(study, "authenticate_input", lambda folder: seen.append(folder.name))
    output = tmp_path / "output"
    plan = study.prepare(parent, output)
    assert len(plan["jobs"]) == len(seen) == 27
    for job in jobs:
        folder = output / "inputs" / job["job_id"]
        assert plan["inputs"][job["job_id"]] == study.support.digest(folder / "input.json")
        receipt = json.loads((folder / "input.json").read_text())
        assert receipt["selected_depth"] == 2
        assert all(study.support.digest(folder / name) == value for name, value in receipt["files"].items())
