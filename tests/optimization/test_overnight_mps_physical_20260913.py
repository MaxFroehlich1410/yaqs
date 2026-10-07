# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test physical equivalence, replay custody and honest diagnostic interruption costs."""

from __future__ import annotations

import gzip
import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import (
    mps_physical_experiment as audit,
)
from experiments.overnight_state_preparation_20260913_0108 import (
    trajectory_mixture,
)
from mqt.yaqs.optimization import KrotovNoiseMap, ParameterizedCircuit, ParameterizedGate, StatevectorBackend

if TYPE_CHECKING:
    from pathlib import Path

    from mqt.yaqs.core.data_structures.mps import MPS


def problem() -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Return a phase coordinate and an asymmetric reversed-control entangling circuit."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("rz", (0,), param_index=0),
            ParameterizedGate("ry", (0,), param_index=1),
            ParameterizedGate("rx", (1,), param_index=2),
            ParameterizedGate("cx", (1, 0)),
            ParameterizedGate("rz", (1,), param_index=3),
        ],
    )
    target = np.array([1, 2j, -0.5, 0.7j])
    return circuit, np.array([0.8, 0.5, -0.3, 0.2]), target / np.linalg.norm(target)


def saved_case(root: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Return small saved inputs with explicit phase-gradient rounding residuals."""
    circuit, theta, target = problem()
    sampler = trajectory_mixture.MixtureSampler(circuit)
    engine = StatevectorBackend(circuit, target)
    endpoints: dict[str, Any] = {}
    for label in audit.old.ENDPOINTS:
        endpoint: dict[str, Any] = {"circuit": pilot.circuit_record(circuit), "theta": theta.tolist()}
        states = {b: audit.old.initial_state(theta.tolist()) for b in ("dense", "mps")}
        rows = []
        for index in range(3):
            batch = sampler.ensemble(
                4,
                seed=audit.runner.seed_for("mps_overlap", "tiny", 11),
                update_index=index,
                namespace="development_diagnostics",
            )
            gradients = {b: batch.gradient(engine, np.asarray(states[b]["theta"]))[0] for b in states}
            gradients["mps"][0] += 7e-16
            rows.append({
                "update_index": index,
                "accepted": False,
                "map_sha256": audit.old.maps_hash(batch),
                "weights": batch.weights.tolist(),
                **{b: {"gradient": gradient.tolist()} for b, gradient in gradients.items()},
            })
            states = {b: audit.old.update(states[b], gradients[b]) for b in states}
        record: dict[str, Any] = {**states, "steps": rows, "accepted": False}
        pilot.write_record(root / label / "STATE.json", record)
        endpoints[label] = endpoint
    return {"target_id": "tiny", "replica": 11, "endpoints": endpoints}, target


def raw(path: Path) -> dict[str, Any]:
    """Return full compressed engineering output without modifying it."""
    return json.loads(gzip.decompress(path.read_bytes()))


def numerical(value: object) -> object:
    """Return deterministic replay data excluding only runtime measurements."""
    if isinstance(value, dict):
        mapping = cast("dict[str, Any]", value)
        return {k: numerical(v) for k, v in mapping.items() if "seconds" not in k}
    if isinstance(value, list):
        return [numerical(item) for item in value]
    return value


def test_dense_and_mps_order_agree_with_independent_circuit() -> None:
    """Asymmetric two-site noise and reversed CX detect a wrong qargs convention."""
    circuit, theta, target = problem()
    batch = trajectory_mixture.MixtureSampler(circuit).ensemble(2, seed=3, update_index=0)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    noise = KrotovNoiseMap(operators=((np.kron(x, np.eye(2)), (0, 1)),))
    row = (KrotovNoiseMap(),) * 4 + (noise,)
    batch = replace(batch, maps=(row,), weights=np.array([1.0]))
    expected = QuantumCircuit(2)
    expected.rz(theta[0], 0)
    expected.ry(theta[1], 0)
    expected.rx(theta[2], 1)
    expected.cx(1, 0)
    expected.rz(theta[3], 1)
    expected.x(0)
    expected_vector = np.asarray(Statevector.from_instruction(expected).data)
    for tensor in (False, True):
        vectors, record = audit.forward_states(circuit, theta, batch, tensor=tensor)
        np.testing.assert_allclose(vectors[0], expected_vector, atol=1e-14, rtol=0)
        assert record["maximum_norm_error"] < 1e-14
    engine = StatevectorBackend(circuit, target)
    assert engine.metrics(theta, batch.maps, weights=batch.weights)[1] == pytest.approx(
        abs(np.vdot(target, expected_vector)) ** 2,
        abs=1e-14,
    )


def test_phase_alignment_does_not_hide_norm_or_non_gauge_drift() -> None:
    """A scalar phase is removable; a physical rotation and norm change remain visible."""
    left = np.array([1, 0], dtype=complex)
    phase = np.exp(0.71j) * left
    aligned = audit.state_comparison(left, phase, left)
    assert aligned["phase_aligned_distance"] < 1e-15
    assert aligned["unaligned_distance"] > 0.5
    moved = np.array([np.cos(0.1), np.sin(0.1)], dtype=complex)
    assert audit.state_comparison(left, moved, left)["phase_aligned_distance"] > 0.09
    assert audit.state_comparison(left, phase * 1.001, left)["maximum_norm_error"] > 0.002


def test_gauge_bound_counts_shared_non_gauge_occurrences_and_noise_support() -> None:
    """A phase occurrence cannot erase a shared physical rotation or prior noise support."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("rz", (0,), param_index=0),
            ParameterizedGate("ry", (0,), param_index=0, angle_scale=-2),
            ParameterizedGate("rz", (1,), param_index=1),
        ],
    )
    batch = trajectory_mixture.MixtureSampler(circuit).ensemble(2, seed=9, update_index=0)
    batch = replace(batch, maps=((KrotovNoiseMap(),) * 3,), weights=np.array([1.0]))
    left, right = np.array([0.2, 0.3]), np.array([0.21, 0.32])
    bound = audit.parameter_bound(circuit, left, right, batch)
    assert [row["phase"] for row in bound["terms"]] == [True, False, True]
    assert bound["bound"] == pytest.approx(0.01)
    vectors = [audit.forward_states(circuit, t, batch, tensor=False)[0][0] for t in (left, right)]
    assert (
        0.009 < audit.state_comparison(vectors[0], vectors[1], vectors[0])["phase_aligned_distance"] <= bound["bound"]
    )
    noise = KrotovNoiseMap(operators=((np.array([[0, 1], [1, 0]], dtype=complex), (1,)),))
    changed = replace(batch, maps=((noise, KrotovNoiseMap(), KrotovNoiseMap()),))
    assert audit.parameter_bound(circuit, left, right, changed)["terms"][2]["phase"] is False


def test_path_order_and_weights_must_match_exactly() -> None:
    """The same set or distribution of trajectories is insufficient for paired checks."""
    circuit, _, _ = problem()
    batch = trajectory_mixture.MixtureSampler(circuit).ensemble(4, seed=23, update_index=2)
    previous = {"map_sha256": audit.old.maps_hash(batch), "weights": batch.weights.tolist()}
    audit.authenticate_batch(batch, previous)
    for changed in (
        replace(batch, maps=batch.maps[::-1]),
        replace(batch, weights=batch.weights[::-1].copy()),
    ):
        with pytest.raises(ValueError, match="path order"):
            audit.authenticate_batch(changed, previous)


def test_replay_first_step_bound_and_saved_state_custody(tmp_path: Path) -> None:
    """Tiny phase gradients create large theta drift while satisfying the derived Adam bound."""
    case, _ = saved_case(tmp_path)
    label = audit.old.ENDPOINTS[0]
    saved = audit.old.read(tmp_path / label / "STATE.json")
    rows = audit.replay(case["endpoints"][label], saved)
    assert all(row["adam_bound_accepted"] for row in rows)
    assert rows[0]["first_step_bound_accepted"]
    assert abs(rows[0]["theta_delta"][0]) > audit.TOLERANCE
    saved["dense"]["theta"][0] += 1e-12
    with pytest.raises(ValueError, match="bit-for-bit"):
        audit.replay(case["endpoints"][label], saved)


def test_physical_gate_rejects_non_gauge_state_change() -> None:
    """The new test cannot turn an arbitrary physical discrepancy into an accepted phase."""
    vector = np.array([1, 0], dtype=complex)
    wrong = np.array([np.cos(1e-3), np.sin(1e-3)], dtype=complex)
    values = {key: [vector] for key in ("dense_dense", "dense_mps", "mps_dense", "mps_mps")}
    values["mps_mps"] = [wrong]
    result = audit.physical_comparisons(values, vector, 0)
    assert result["same_dense_theta"]["accepted"]
    assert not result["same_mps_theta"]["accepted"]
    assert not result["different_cross"]["accepted"]


def test_exact_resume_charges_failed_work_and_preserves_prior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed step survives interruption, prior failures survive, and no training is rerun."""
    prior, resumed, uninterrupted = (tmp_path / name for name in ("prior", "resumed", "uninterrupted"))
    case, target = saved_case(prior)
    prior_bytes = {path: path.read_bytes() for path in prior.rglob("*") if path.is_file()}
    original = audit.audit_step
    calls = 0

    def interrupted(
        circuit: ParameterizedCircuit,
        target: np.ndarray,
        tensor_target: MPS,
        batch: trajectory_mixture.MixtureBatch,
        previous: dict[str, Any],
        replayed: dict[str, Any],
        attempt: audit.old.Attempt,
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            msg = "injected interruption after a durable completed step"
            raise RuntimeError(msg)
        return original(circuit, target, tensor_target, batch, previous, replayed, attempt)

    monkeypatch.setattr(audit, "audit_step", interrupted)
    with pytest.raises(RuntimeError, match="injected interruption"):
        audit.run_case(resumed, prior, case, "q2", lambda: target)
    first = resumed / audit.old.ENDPOINTS[0] / "step1.json"
    first_bytes = first.read_bytes()
    monkeypatch.setattr(audit, "audit_step", original)
    result = audit.run_case(resumed, prior, case, "q2", lambda: target)
    full = audit.run_case(uninterrupted, prior, case, "q2", lambda: target)
    assert result["accepted"]
    assert full["accepted"]
    assert first.read_bytes() == first_bytes
    assert result["prior_gate"] == "failed; unchanged"
    attempts = audit.old.read(resumed / "ATTEMPTS.json")["attempts"]
    assert len(attempts) == 2
    assert [a["status"] for a in attempts] == ["failed", "complete"]
    assert all(a["cpu_seconds"] > 0 for a in attempts)
    for label in audit.old.ENDPOINTS:
        for step in range(1, 4):
            name = f"{label}/step{step}.raw.json.gz"
            assert numerical(raw(resumed / name)) == numerical(raw(uninterrupted / name))
    assert all(path.read_bytes() == content for path, content in prior_bytes.items())
    complete_bytes = {p: p.read_bytes() for p in resumed.rglob("*") if p.is_file()}
    assert audit.run_case(resumed, prior, case, "q2", lambda: target) == result
    assert all(p.read_bytes() == value for p, value in complete_bytes.items())
    with (resumed / audit.old.ENDPOINTS[0] / "step1.raw.json.gz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="dependency changed"):
        audit.run_case(resumed, prior, case, "q2", lambda: target)


def test_unknown_hard_kill_cost_blocks_resume(tmp_path: Path) -> None:
    """An unfinished durable attempt is unknown physical work, never a zero-cost retry."""
    audit.open_attempt(tmp_path, "identity")
    before = (tmp_path / "ATTEMPTS.json").read_bytes()
    with pytest.raises(ValueError, match="Unknown interrupted attempt cost"):
        audit.open_attempt(tmp_path, "identity")
    assert (tmp_path / "ATTEMPTS.json").read_bytes() == before


def test_changed_completed_step_identity_is_rejected(tmp_path: Path) -> None:
    """Signed step identity is checked even when its raw output checksum still matches."""
    path = tmp_path / "step.json"
    raw_path = path.with_suffix(".raw.json.gz")
    raw_path.write_bytes(b"test")
    pilot.write_record(path, {"signature": "old", "raw_sha256": audit.old.file_hash(raw_path)})
    with pytest.raises(ValueError, match="Completed physical step"):
        audit.persist_step(path, "changed", dict)


def prior_panel(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Return a signed q2 prior fixture while replacing only the fixed q6 panel validator."""
    case_root = root / "tiny__11"
    case, _ = saved_case(case_root)
    protocol: dict[str, Any] = {"cases": [case], "engineering": "q2 custody fixture"}
    monkeypatch.setattr(audit.old, "TARGETS", ("tiny",))
    monkeypatch.setattr(audit.old, "verify", lambda _protocol: None)
    endpoints: dict[str, Any] = {}
    for label in audit.old.ENDPOINTS:
        path = case_root / label / "STATE.json"
        state = audit.old.read(path)
        for index, row in enumerate(state["steps"]):
            raw_path = path.parent / f"step{index + 1}.audit.json.gz"
            raw_path.write_bytes(gzip.compress(b'{"fixture":true}', mtime=0))
            row.update(raw_audit_file=raw_path.name, raw_audit_sha256=audit.old.file_hash(raw_path))
        pilot.write_record(path, state)
        endpoints[label] = {"accepted": False, "state_checksum": pilot.digest(state)}
    for name in ("TARGET_MPS.json", "ATTEMPTS.json"):
        pilot.write_record(case_root / name, {"engineering": "q2 fixture"})
    result: dict[str, Any] = {
        "signature": pilot.digest([pilot.digest(protocol), case, audit.old.VERSION, audit.old.settings()]),
        "status": "complete",
        "accepted": False,
        "endpoints": endpoints,
        "dependencies": {
            str(p.relative_to(case_root)): audit.old.file_hash(p) for p in case_root.rglob("*") if p.is_file()
        },
    }
    pilot.write_record(case_root / "RESULT.json", result)
    pilot.write_record(root / "PROTOCOL.json", protocol)
    pilot.write_record(
        root / "SUMMARY.json",
        {
            "version": audit.old.VERSION,
            "protocol_checksum": pilot.digest(protocol),
            "accepted": False,
            "cases": {"tiny": {"checksum": pilot.digest(result)}},
            "resource_accounting_complete": True,
        },
    )
    return result


def test_prior_manifest_requires_all_raw_files_and_keeps_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing split custody or an altered failure flag cannot masquerade as an authorized input."""
    prior_panel(tmp_path, monkeypatch)
    panel = audit.input_panel(tmp_path)
    assert len(panel["files"]) == 13
    summary_path = tmp_path / "SUMMARY.json"
    summary = audit.old.read(summary_path)
    summary["accepted"] = True
    pilot.write_record(summary_path, summary)
    with pytest.raises(ValueError, match="failed prior panel"):
        audit.input_panel(tmp_path)
    summary["accepted"] = False
    pilot.write_record(summary_path, summary)
    path = tmp_path / "tiny__11" / audit.old.ENDPOINTS[0] / "step1.audit.json.gz"
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="compressed split audit"):
        audit.input_panel(tmp_path)


@pytest.mark.usefixtures("isolated_overnight_custody")
def test_protocol_cannot_omit_input_manifest_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Freshly signed metadata does not authorize dropping expected prior records."""
    prior = tmp_path / "prior"
    prior_panel(prior, monkeypatch)
    protocol = audit.prepare(tmp_path / "new", prior)
    audit.verify(protocol)
    protocol["input_panel"]["files"].pop(next(iter(protocol["input_panel"]["files"])))
    with pytest.raises(ValueError, match="input custody changed"):
        audit.verify(protocol)
