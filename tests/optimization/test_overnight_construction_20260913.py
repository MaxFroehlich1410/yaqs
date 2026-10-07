# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Small-system legacy construction, exact resume, custody and MPS audits."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from experiments import state_preparation_hybrid_pilot as hybrid
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import construction
from mqt.yaqs.optimization import (
    KrotovOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    ParameterizedGate,
    train_krotov_state_preparation_hybrid,
)

if TYPE_CHECKING:
    from pathlib import Path


def engineering_config() -> construction.Config:
    """Return a cheap schedule that switches phase inside its first chunk."""
    return construction.Config(
        mode="engineering",
        depths=2,
        source_updates=4,
        online_updates=1,
        pruning_rounds=1,
        repair_updates=2,
        final_updates=2,
        checkpoint_every=2,
    )


def target_for(n: int) -> np.ndarray:
    """Return a deterministic complex synthetic state, never a benchmark loader."""
    rng = np.random.default_rng(20260913 + n)
    target = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return target / np.linalg.norm(target)


@pytest.fixture
def custody(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep test manifests local while preserving the real optimizer implementation."""
    monkeypatch.setattr(construction, "sources", lambda: {"numerical": "fixed"})
    monkeypatch.setattr(construction, "environment", lambda: {"runtime": "fixed"})
    monkeypatch.setattr(construction.runner, "custody", lambda: {"benchmark_hashes": {"immutable": "fixed"}})


def direct_legacy(n: int) -> tuple[dict[str, Any], list[float], list[list[int]]]:
    """Return an uninterrupted public-API implementation of the old source recipe."""
    config = engineering_config()
    target, name = target_for(n), f"synthetic_q{n}"
    legacy_seed = pilot.pilot_seed(pilot.PilotConfig(), "initialization", name, 11)
    master = np.random.default_rng(legacy_seed).normal(size=construction.circuit_at_depth(n, config.depths).num_params)
    theta = 0.1 * master[: construction.circuit_at_depth(n, 1).num_params]

    def train(updates: int, rate: float, online: int) -> np.ndarray:
        options = KrotovOptions(
            max_iterations=updates,
            switch_iteration=online,
            online_step_size=0.7,
            batch_step_size=rate,
            online_schedule="constant",
            batch_schedule="constant",
            truncation=KrotovTruncation(),
        )
        return train_krotov_state_preparation_hybrid(circuit, target, initial_theta=theta, options=options).theta

    for depth in range(1, config.depths + 1):
        circuit = construction.circuit_at_depth(n, depth)
        if theta.size < circuit.num_params:
            theta = np.concatenate((theta, 0.001 * master[theta.size : circuit.num_params]))
        theta = train(config.source_updates, 0.2, config.online_updates)
    deletions = []
    for _ in range(config.pruning_rounds):
        scores = construction.pruning_scores(circuit, theta, target, "M")
        removed = construction.select_deletions(scores, n)
        deletions.append([cast("int", circuit.gates[k].logical_gate_id) for k in removed])
        circuit, theta = construction.delete_gates(circuit, theta, removed)
        theta = train(config.repair_updates, 0.1, 0)
    theta = train(config.final_updates, 0.1, 0)
    return cast("dict[str, Any]", pilot.circuit_record(circuit)), theta.tolist(), deletions


@pytest.mark.usefixtures("custody")
@pytest.mark.parametrize("n", [1, 2, 3])
def test_complete_adapter_matches_uninterrupted_legacy_and_full_qiskit_preparation(tmp_path: Path, n: int) -> None:
    """Chunking, layer growth and magnitude deletion preserve the exact old endpoint."""
    target = target_for(n)
    before = target.copy()
    manifest = construction.prepare(tmp_path, target, f"synthetic_q{n}", 11, config=engineering_config())
    result = construction.reproduce(tmp_path, manifest)
    circuit, theta, deletions = direct_legacy(n)
    assert pilot.digest(result["circuit"]) == pilot.digest(circuit)
    np.testing.assert_allclose(result["theta"], theta, atol=2e-13, rtol=0)
    assert [e["removed_origins"] for e in result["pruning_events"]] == deletions
    assert result["updates"] == 12
    assert all(g["noise_enabled"] for g in result["circuit"]["gates"])
    assert set(range(3 * n)) <= {g["logical_gate_id"] for g in result["circuit"]["gates"]}
    np.testing.assert_array_equal(target, before)
    restored = pilot.restore_circuit(result["circuit"])
    reference = QuantumCircuit(n)
    for gate in restored.gates:
        getattr(reference, gate.name)(restored.angle(gate, np.asarray(theta), None), *gate.sites)
    fidelity = abs(np.vdot(target, Statevector.from_instruction(reference).data)) ** 2
    assert result["diagnostics"]["fidelity"] == pytest.approx(fidelity, abs=1e-12)
    assert result["diagnostics"]["norm_squared"] == pytest.approx(1, abs=1e-12)
    assert result["diagnostics"]["observed_final_forward_peak_bond"] <= 2 ** (n // 2)
    assert result["diagnostics"]["training_peak_bond"] is None
    assert result["construction_cpu_seconds"] > result["diagnostic_cpu_seconds"] > 0
    assert result["timing_complete"]


@pytest.mark.usefixtures("custody")
@pytest.mark.parametrize("block_name", ["source_1", "source_2", "M_repair_0", "refine_M_0"])
def test_interrupted_block_resume_preserves_parameters_origins_phase_and_charges_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, block_name: str
) -> None:
    """An interruption after a real checkpoint resumes every growth/pruning phase exactly."""
    full_root, resumed_root = tmp_path / "full", tmp_path / "resume"
    config = engineering_config()
    first = construction.prepare(full_root, target_for(2), "synthetic_q2", 11, config=config)
    full = construction.reproduce(full_root, first)
    second = construction.prepare(resumed_root, target_for(2), "synthetic_q2", 11, config=config)
    original_write = pilot.write_record

    def interrupt(path: Path, record: dict[str, Any]) -> None:
        original_write(path, record)
        if path.name == f"{block_name}.json" and record["completed_updates"] == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(pilot, "write_record", interrupt)
    with pytest.raises(KeyboardInterrupt):
        construction.reproduce(resumed_root, second)
    monkeypatch.setattr(pilot, "write_record", original_write)
    resumed = construction.reproduce(resumed_root, second)
    assert resumed["theta"] == full["theta"]
    assert pilot.digest(resumed["circuit"]) == pilot.digest(full["circuit"])
    assert resumed["pruning_events"] == full["pruning_events"]
    for name in resumed["blocks"]:
        actual = pilot.read_record(resumed_root / "blocks" / f"{name}.json")
        expected = pilot.read_record(full_root / "blocks" / f"{name}.json")
        assert actual["theta"] == expected["theta"]
        assert actual["trace"] == expected["trace"]
        assert actual["completed_updates"] == expected["completed_updates"]
    attempts = cast("list[dict[str, Any]]", pilot.read_record(resumed_root / "ATTEMPTS.json")["attempts"])
    assert [a["status"] for a in attempts] == ["interrupted", "complete"]
    assert all(a["cpu_seconds"] > 0 for a in attempts)
    assert resumed["construction_cpu_seconds"] == pytest.approx(sum(a["cpu_seconds"] for a in attempts))


@pytest.mark.usefixtures("custody")
def test_prepare_only_rejects_confirmation_without_loading_or_simulating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even an explicit dense file cannot use the adapter to access confirmation."""
    monkeypatch.setattr(hybrid, "optimize_block", lambda *_a, **_k: pytest.fail("prepare optimized"))
    monkeypatch.setattr(construction, "mps_diagnostics", lambda *_a: pytest.fail("prepare simulated"))
    manifest = construction.prepare(tmp_path, target_for(2), "synthetic_q2", 11, config=engineering_config())
    assert list(tmp_path.iterdir()) == [tmp_path / "PROTOCOL.json"]
    assert manifest["truncation"] == asdict(KrotovTruncation()) == construction.TRUNCATION
    assert manifest["initialization_seed"] == pilot.pilot_seed(
        pilot.PilotConfig(), "initialization", "synthetic_q2", 11
    )
    before = (tmp_path / "PROTOCOL.json").read_bytes()
    construction.prepare(tmp_path, target_for(2), "synthetic_q2", 11, config=engineering_config())
    assert (tmp_path / "PROTOCOL.json").read_bytes() == before
    with pytest.raises(ValueError, match="normalized development"):
        construction.prepare(tmp_path / "forbidden", np.ones(64, dtype=np.complex128) / 8, "gaussian_fresh_1", 11)
    assert not (tmp_path / "forbidden").exists()


@pytest.mark.usefixtures("custody")
def test_cached_result_custody_and_block_timing_authentication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cached endpoints do not bypass code, block or physical-cost custody."""
    manifest = construction.prepare(tmp_path, target_for(1), "synthetic_q1", 11, config=engineering_config())
    result = construction.reproduce(tmp_path, manifest)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    monkeypatch.setattr(construction, "run_pipeline", lambda *_a: pytest.fail("cached construction ran"))
    assert pilot.digest(construction.reproduce(tmp_path, manifest)) == pilot.digest(result)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    monkeypatch.setattr(construction, "sources", lambda: {"numerical": "changed"})
    with pytest.raises(ValueError, match="custody changed"):
        construction.reproduce(tmp_path, manifest)
    monkeypatch.setattr(construction, "sources", lambda: {"numerical": "fixed"})
    path = tmp_path / "ATTEMPTS.json"
    attempts = cast("dict[str, Any]", pilot.read_record(path))
    attempts["attempts"][0]["cpu_seconds"] = 0.0
    pilot.write_record(path, attempts)
    with pytest.raises(ValueError, match="timing ledger changed"):
        construction.reproduce(tmp_path, manifest)
    path.write_bytes(before[path])
    path = tmp_path / "blocks/source_1.json"
    block = cast("dict[str, Any]", pilot.read_record(path))
    block["theta"][0] += 0.01
    pilot.write_record(path, block)
    with pytest.raises(ValueError, match="block changed"):
        construction.reproduce(tmp_path, manifest)


@pytest.mark.usefixtures("custody")
def test_hard_killed_attempt_leaves_physical_total_cost_unknown(tmp_path: Path) -> None:
    """An unobserved elapsed duration cannot be imputed as zero on resume."""
    manifest = construction.prepare(tmp_path, target_for(1), "synthetic_q1", 11, config=engineering_config())
    pilot.write_record(
        tmp_path / "ATTEMPTS.json",
        {
            "signature": pilot.digest(manifest),
            "attempts": [{"status": "running", "cpu_seconds": None, "wall_seconds": None}],
        },
    )
    result = construction.reproduce(tmp_path, manifest)
    assert not result["timing_complete"]
    assert result["construction_cpu_seconds"] is None
    assert result["construction_wall_seconds"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"source_updates": 99},
        {"backend": "statevector"},
        {"online_updates": 30},
        {"checkpoint_every": 5},
        {"mode": "confirmation"},
        {"depths": True},
    ],
)
def test_scientific_recipe_cannot_change_silently(change: dict[str, Any]) -> None:
    """Any schedule/backend change requires separately named development work."""
    with pytest.raises(ValueError, match="frozen legacy"):
        construction.Config(**change)


@pytest.mark.parametrize("bad", [np.zeros(4), np.ones(3), np.array([1, np.nan]), np.ones((2, 2)) / 2])
def test_invalid_target_vector_rejected_before_artifacts(tmp_path: Path, bad: np.ndarray) -> None:
    """Inputs cannot be normalized silently or interpreted in a different basis shape."""
    with pytest.raises(ValueError, match="normalized development"):
        construction.prepare(tmp_path, bad, "synthetic_q2", 11, config=engineering_config())
    assert not list(tmp_path.iterdir())


def test_legacy_q6_architecture_seed_and_origin_recipe_without_simulation() -> None:
    """Full physical starts and adjoint parameter ordering are structural, not scientific tests."""
    for depth, total in enumerate((63, 108, 153, 198), start=1):
        circuit = construction.circuit_at_depth(6, depth)
        assert pilot.digest(pilot.circuit_record(circuit)) == pilot.digest(
            pilot.circuit_record(pilot.pilot_circuit(depth))
        )
        assert len(circuit.gates) == circuit.num_params == total
        assert all(g.logical_gate_id == g.param_index for g in circuit.gates)
        assert all(g.noise_enabled for g in circuit.gates)
    seed = construction.seed_for("initialization", pilot.TARGETS[0], 11)
    assert seed == pilot.pilot_seed(pilot.PilotConfig(), "initialization", pilot.TARGETS[0], 11)
    newer = (
        int(
            pilot.digest(["yaqs.trajectory_competitiveness.v1", 2026091001, "construction", pilot.TARGETS[0], 11])[:16],
            16,
        )
        << 32
    )
    assert seed != newer


@pytest.mark.usefixtures("custody")
def test_changed_input_endpoint_requires_new_root(tmp_path: Path) -> None:
    """A target, identity or seed cannot silently inherit existing construction state."""
    construction.prepare(tmp_path, target_for(2), "synthetic_q2", 11, config=engineering_config())
    with pytest.raises(ValueError, match="inputs differ"):
        construction.prepare(tmp_path, target_for(2), "synthetic_q2", 29, config=engineering_config())


@pytest.mark.usefixtures("custody")
@pytest.mark.parametrize("changed", [False, True])
def test_archive_comparison_retains_matching_and_mismatching_endpoints(tmp_path: Path, *, changed: bool) -> None:
    """An archive mismatch is an explicit result, never a reason to retune or overwrite."""
    circuit, theta, _ = direct_legacy(1)
    theta[0] += 0.01 if changed else 0
    archived: dict[str, Any] = {"circuit": circuit, "theta": theta}
    path, manifest_path = tmp_path / "reference.json", tmp_path / "reference_manifest.json"
    pilot.write_record(path, archived)
    pilot.write_record(manifest_path, {"historical": "fixture"})
    archive = {
        **archived,
        "target_id": "synthetic_q1",
        "replica": 11,
        "path": str(path),
        "checksum": pilot.digest(archived),
        "manifest_path": str(manifest_path),
        "manifest_checksum": pilot.digest({"historical": "fixture"}),
    }
    root = tmp_path / "run"
    manifest = construction.prepare(
        root, target_for(1), "synthetic_q1", 11, config=engineering_config(), archive=archive
    )
    result = construction.reproduce(root, manifest)
    assert result["status"] == "complete"
    assert result["archive_reproduction"]["matches"] is not changed
    assert result["archive_reproduction"]["historical_cpu_seconds"] is None
    assert result["archive_reproduction"]["parameter_max_abs_error"] == pytest.approx(0.01 if changed else 0, abs=1e-12)
    archived["theta"][0] += 0.02
    pilot.write_record(path, archived)
    with pytest.raises(ValueError, match="archive reference changed"):
        construction.reproduce(root, manifest)


def test_archive_import_authenticates_recipe_and_preserves_unknown_cpu_without_simulation(tmp_path: Path) -> None:
    """Archive audit reads scalar metadata only, including every initial physical gate."""
    gates = [ParameterizedGate("ry", (i % 6,), i) for i in range(78)]
    gates.extend(ParameterizedGate("rzz", (i % 5, i % 5 + 1), i + 78) for i in range(30))
    target_id = construction.TARGETS[0]
    path = tmp_path / "H25_lr0p7" / target_id / "seed_11/result.json"
    manifest: dict[str, Any] = {
        "version": hybrid.CONSTRUCTION_VERSION,
        "config": asdict(pilot.PilotConfig()),
        "source_fingerprint": "historical",
        "source_records": {
            f"{target_id}/seed_11/construction.json": "prior_construct",
            f"{target_id}/seed_11/refinement.json": "prior_refine",
        },
    }
    record: dict[str, Any] = {
        "target_id": target_id,
        "replica": 11,
        "setting": "H25_lr0p7",
        "schedule": {"online_updates": 25, "online_rate": 0.7},
        "wall_seconds": 12.5,
        "source_construction_checksum": "prior_construct",
        "source_refinement_checksum": "prior_refine",
        "candidates": [
            {
                "route": "H25_lr0p7:cold_final",
                "circuit": pilot.circuit_record(ParameterizedCircuit(6, gates)),
                "theta": [0.1] * 108,
            }
        ],
    }
    pilot.write_record(tmp_path / "manifest.json", manifest)
    pilot.write_record(path, record)
    reference = construction.archive_reference(tmp_path, target_id, 11)
    assert reference["historical_cpu_seconds"] is None
    assert reference["historical_wall_seconds"] == pytest.approx(12.5)
    assert reference["checksum"] == pilot.digest(record)
    record["schedule"]["online_rate"] = 0.9
    pilot.write_record(path, record)
    with pytest.raises(ValueError, match="identity differs"):
        construction.archive_reference(tmp_path, target_id, 11)
    with pytest.raises(ValueError, match="unchanged development panel"):
        construction.archive_reference(tmp_path, "haar_fresh_1", 11)
