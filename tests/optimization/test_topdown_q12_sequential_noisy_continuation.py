# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Sequential q12 deletion, protocol isolation and unchanged phase recovery."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from experiments import topdown_q12_sequential_noisy_continuation as study
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate, StatevectorBackend

if TYPE_CHECKING:
    from pathlib import Path


def test_twelve_qubit_rescoring_prevents_compounded_deletions() -> None:
    """Two individually cheap deletions need rescoring to preserve the second gate."""
    circuit = ParameterizedCircuit(12, [ParameterizedGate("ry", (0,), param_index=i) for i in range(2)])
    theta = np.array([0.18, 0.18])
    qc = QuantumCircuit(12)
    qc.ry(0.36, 0)
    target = Statevector.from_instruction(qc).data
    engine = StatevectorBackend(circuit, target)
    simultaneous, _, _ = study.base.support.prune_b(engine, theta)
    sequential, values, trace = study.sequential_prune(engine, theta)
    assert not simultaneous.gates
    assert len(sequential.gates) == 1
    assert trace["remaining_origins"] == [1]
    assert trace["repair_updates"] == 0
    assert trace["iterations"][-1]["scores"][0] >= 0.01
    for entry in trace["iterations"]:
        origins = entry["origins"]

        def fidelity(retained: list[int]) -> float:
            """Independently replay literal retained rotations using Qiskit.

            Returns:
                Exact ideal fidelity for the retained gate indices.
            """
            reference = QuantumCircuit(12)
            for index in retained:
                reference.ry(theta[index], 0)
            return float(abs(np.vdot(target, Statevector.from_instruction(reference).data)) ** 2)

        baseline = fidelity(origins)
        expected = [baseline - fidelity([j for j in origins if j != origin]) for origin in origins]
        np.testing.assert_allclose(entry["scores"], expected, atol=1e-14)
    np.testing.assert_array_equal(values, theta[1:])
    np.testing.assert_array_equal(theta, [0.18, 0.18])
    assert len(circuit.gates) == 2


def test_policy_restores_original_workflow_after_failure() -> None:
    """A sequential failure cannot change the simultaneous protocol or pruning hook."""
    original = study.base.VERSION, study.base.PROTOCOL, study.base.sources, study.base.support.prune_b

    def fail() -> None:
        """Exercise restoration after the imported workflow raises.

        Raises:
            RuntimeError: Always, while the sequential policy is active.
        """
        with study.sequential_policy():
            assert study.base.PROTOCOL is study.PROTOCOL
            assert study.base.support.prune_b is study.sequential_prune
            assert study.base.sources is study.sources
            msg = "synthetic"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="synthetic"):
        fail()
    assert (study.base.VERSION, study.base.PROTOCOL, study.base.sources, study.base.support.prune_b) == original


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "haar"])
def test_prepare_requires_exact_structured_panel(tmp_path: Path, mutation: str) -> None:
    """Incomplete, duplicated and Haar-containing panels are rejected before writes."""
    rows = [{"target_id": target, "seed": seed} for target in study.TARGET_IDS for seed in (300, 400, 500)]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0]
    else:
        rows[-1]["target_id"] = "haar_random_1"
    reference = tmp_path / "references.json"
    reference.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="six structured"):
        study.prepare(reference, tmp_path / "campaign")
    assert not (tmp_path / "campaign").exists()


def test_prepare_seals_and_replays_all_structured_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful preparation preserves parent angles and seals a separate C identity."""
    circuit = ParameterizedCircuit(12, [ParameterizedGate("ry", (0,), param_index=0)])
    theta = np.array([0.5])
    qc = QuantumCircuit(12)
    qc.ry(0.5, 0)
    target = Statevector.from_instruction(qc).data
    stage = {"training": {"circuit": study.base.support.circuit_record(circuit)}}

    def parent(folder: Path, target_id: str, seed: int) -> tuple:
        """Stand in for an already authenticated noiseless parent.

        Returns:
            Parent identity, selected stage and its private angles.
        """
        return (
            {"parent": str(folder), "target_id": target_id, "seed": seed, "ideal_fidelity": 1.0},
            stage,
            {"theta": theta},
        )

    monkeypatch.setattr(study.base, "authenticate_parent", parent)
    monkeypatch.setattr(study.base, "load_target", lambda *_: SimpleNamespace(state_vector_copy=target.copy))
    monkeypatch.setattr(study, "sources", lambda: {"numerical_source": "frozen"})
    rows = [
        {"target_id": t, "seed": s, "parent": str(tmp_path / "parent"), "host": "leonhard-13"}
        for t in study.TARGET_IDS
        for s in (300, 400, 500)
    ]
    references = tmp_path / "references.json"
    references.write_text(json.dumps(rows))
    original = study.base.PROTOCOL
    plan = study.prepare(references, tmp_path / "campaign")
    assert len(plan["jobs"]) == 18
    assert plan["protocol"] == study.PROTOCOL
    assert study.base.PROTOCOL is original
    for job in plan["jobs"]:
        folder = tmp_path / "campaign/inputs" / job["job_id"]
        receipt = json.loads((folder / "input.json").read_text())
        assert receipt["version"] == study.VERSION
        assert receipt["seed"] == job["seed"]
        _, arrays = study.base.required(folder / "construction")
        np.testing.assert_array_equal(arrays["theta"], theta)


def test_pipeline_preserves_shared_bank_and_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Actual noisy phases inherit C parameters, share their bank and reuse sealed stages."""
    for key, value in {
        "online_updates": 3,
        "batch_updates": 4,
        "training_paths": 2000,
        "final_batches": 2,
        "final_paths_per_batch": 512,
    }.items():
        monkeypatch.setitem(study.PROTOCOL, key, value)
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), param_index=0),
            ParameterizedGate("ry", (0,), param_index=1),
            ParameterizedGate("rxx", (0, 1), param_index=2),
        ],
    )
    theta = np.array([0.18, 0.18, 0.8])
    qc = QuantumCircuit(2)
    qc.ry(0.36, 0)
    qc.rxx(0.8, 1, 0)
    target = Statevector.from_instruction(qc).data
    parent = {"target_id": study.TARGET_IDS[0], "seed": 300}
    identity = {"protocol": study.PROTOCOL, "input_sha256": "example"}
    monkeypatch.setattr(
        study.base,
        "authenticate_input",
        lambda _: (parent, circuit, theta, target, {**identity, "protocol": study.base.PROTOCOL}),
    )
    folder = tmp_path / "job"
    result = study.run_case(folder, tmp_path / "input")
    pruned, pa = study.base.required(folder / "pruning")
    online, oa = study.base.required(folder / "online")
    batch, batch_arrays = study.base.required(folder / "batch")
    assert pruned["pruning"]["variant"] == "C"
    assert pruned["pruning"]["repair_updates"] == 0
    np.testing.assert_array_equal(oa["initial"], pa["theta"])
    np.testing.assert_array_equal(batch_arrays["initial"], oa["theta"])
    assert online["training"]["noise_banks"] == batch["training"]["noise_banks"]
    reduced = study.base.support.restore_circuit(pruned["circuit"])
    expected, _ = study.base.legacy.train(
        reduced,
        pa["theta"],
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
        engine=StatevectorBackend(reduced, target),
    )
    np.testing.assert_array_equal(batch_arrays["theta"], expected)
    monkeypatch.setattr(study.base.legacy, "train", lambda *_a, **_k: pytest.fail("Sealed phases must be reused"))
    assert study.run_case(folder, tmp_path / "input") == result
    with pytest.raises(ValueError, match="Run identity"):
        study.base.run_case(folder, tmp_path / "input")


def test_cli_requires_explicit_input(tmp_path: Path) -> None:
    """The adapter has no implicit launch or parent selection."""
    for action in ("prepare", "run"):
        with pytest.raises(SystemExit):
            study.main(["--action", action, "--output", str(tmp_path)])
