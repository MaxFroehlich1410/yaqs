# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Noisy finite deletion, optimizer-state retention, exact acceptance and cost controls."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider
from benchmarks.state_preparation.weighted_sampling import FactorSampler
from experiments import state_preparation_exact_adam as exact
from experiments import state_preparation_noisy_restructuring as study
from experiments import state_preparation_pilot as pilot
from experiments import state_preparation_trajectory_comparison as comparison
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate, StatevectorBackend

if TYPE_CHECKING:
    from pathlib import Path


def small_case(name: str = "synthetic", replica: int = 11) -> dict[str, Any]:
    """Return a test-only unique-parameter circuit with nonchronological indexing."""
    gates = [ParameterizedGate("ry", (i % 2,), param_index=i, logical_gate_id=i) for i in range(6)]
    gates += [
        ParameterizedGate("rxx", (1, 0), param_index=7, logical_gate_id=70),
        ParameterizedGate("rz", (1,), param_index=6, logical_gate_id=60),
        ParameterizedGate("rx", (0,), param_index=8, logical_gate_id=80),
    ]
    circuit = ParameterizedCircuit(2, gates)
    target = np.array([1, 2j, -3, 4j]) / np.sqrt(30)
    case = {
        "target": comparison.target_record(name, "test", target),
        "replica": replica,
        "circuit": pilot.circuit_record(circuit),
        "initial": np.linspace(-0.3, 0.5, 9).tolist(),
    }
    case["expected_fidelity"] = study.fidelity(study.initial_state(case), target)
    return case


def config() -> study.Config:
    """Return small budgets whose hard control cap makes testing inexpensive."""
    return study.Config(warmup=4, online_updates=2, rounds=2, repair=2, trajectories=4, control_max_updates=8)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("warmup", 0),
        ("warmup", True),
        ("online_updates", 50),
        ("online_updates", -1),
        ("rounds", 0),
        ("shortlist", 0),
        ("repair", 0),
        ("trajectories", 1),
        ("control_max_updates", 49),
        ("root_seed", -1),
        ("rate", np.nan),
        ("online_rate", False),
        ("acceptance_tolerance", 0),
    ],
)
def test_invalid_config(field: str, value: object) -> None:
    """Malformed configurations fail before creating experiment artifacts."""
    with pytest.raises(ValueError, match="configuration"):
        replace(study.Config(), **{field: value})


def test_deletion_compacts_moments_in_parameter_not_gate_order() -> None:
    """Deleted parameter 7 must not shift parameter 6's moment or immutable origin."""
    state = study.initial_state(small_case())
    state.update({"first": list(range(9)), "second": list(range(10, 19)), "counts": list(range(20, 29)), "updates": 30})
    before = copy.deepcopy(state)
    compact = study.deleted_state(state, 6)
    keep = [0, 1, 2, 3, 4, 5, 6, 8]
    for key in ("theta", "first", "second", "counts"):
        assert compact[key] == [state[key][i] for i in keep]
    assert compact["updates"] == 30
    assert compact["circuit"]["gates"][6]["logical_gate_id"] == 60
    assert compact["circuit"]["gates"][7]["param_index"] == 7
    assert state == before
    with pytest.raises(ValueError, match="deletion"):
        study.deleted_state(state, 10)


def test_zero_angle_deletion_removes_noise_and_ranking_protects_product_layer() -> None:
    """An identity unitary still incurs noise unless its entire gate is removed."""
    circuit = ParameterizedCircuit(
        1, [ParameterizedGate("rz", (0,), param_index=i, logical_gate_id=i) for i in range(5)]
    )
    target = np.array([1, 0], dtype=complex)
    state = study.initial_state({"circuit": pilot.circuit_record(circuit), "initial": [0] * 5})
    before = study.fidelity(state, target)
    ranked = study.rank_deletions(state, target)
    assert [row["origin"] for row in ranked] == [3, 4]
    assert all(row["fidelity"] > before for row in ranked)
    full = FactorSampler(circuit, ExactLocalPauliProvider().calibration)
    compact = study.deleted_state(state, 3)
    sampler = FactorSampler(pilot.restore_circuit(compact["circuit"]), ExactLocalPauliProvider().calibration)
    assert sampler.no_factor_probability > full.no_factor_probability
    block = study.train_block(compact, target, "adam", config(), 123, 0, 1)
    assert block["no_factor_probability"] == sampler.no_factor_probability


@pytest.mark.parametrize("method", study.METHODS)
def test_block_wiring_matches_shared_kernels_and_continues_after_deletion(method: str) -> None:
    """Independent manual replay agrees with the driver, including the online switch."""
    case, cfg = small_case(), config()
    target = comparison.target_vector(case["target"])
    state = study.initial_state(case)
    before = copy.deepcopy(state)
    engine = StatevectorBackend(pilot.restore_circuit(state["circuit"]), target)
    sampler = FactorSampler(engine.circuit, ExactLocalPauliProvider().calibration)
    theta = np.array(state["theta"])
    first, second, counts = np.zeros(9), np.zeros(9), np.zeros(9, dtype=int)
    for step in range(1, 5):
        maps, weights = sampler.ensemble(4, 123 + 1_000_003 * step)
        if method == "adam":
            gradient = engine.gradient(theta, maps, weights=weights)[0]
            theta, first, second = exact.adam_step(
                theta, gradient, first, second, step, exact.AdamConfig(learning_rate=0.02)
            )
        elif step <= 2:
            theta, _ = engine.online_update(theta, 0.7, maps, weights=weights)
        else:
            theta, _, second, counts = engine.adaptive_online_update(theta, 0.02, second, counts, maps, weights=weights)
    block = study.train_block(state, target, method, cfg, 123, 0, 4)
    for key, expected in (("theta", theta), ("first", first), ("second", second), ("counts", counts)):
        np.testing.assert_array_equal(block["state"][key], expected)
    assert state == before
    assert block["trajectory_passes"] == 4 * 4 * (2 if method == "adam" else 3)
    compact = study.deleted_state(block["state"], 6)
    repaired = study.train_block(compact, target, method, cfg, 123, 20, 2)
    assert repaired["state"]["updates"] == 6
    assert [t["epoch"] for t in repaired["trace"]] == [21, 22]
    assert all(t["phase"] == method for t in repaired["trace"])
    assert repaired["state"]["counts"] == ([4] * 8 if method == "adaptive_online" else [0] * 8)


def test_selection_requires_improvement_over_repaired_unchanged_control() -> None:
    """A deletion beating the input but not unchanged repair must not be accepted."""
    rows = [{"fidelity": score} for score in (0.8, 0.9, 0.85, 0.9 + 1e-12)]
    assert study.choose_candidate(rows, 1e-10) == 1
    assert study.choose_candidate([{"fidelity": 0.8}] * 4, 1e-10) == 0
    rows[-1]["fidelity"] = 0.91
    assert study.choose_candidate(rows, 1e-10) == 3


def test_cpu_cutoff_excludes_future_and_safety_cap_is_explicit() -> None:
    """Availability follows completed checkpoints, not interpolation or future best scores."""
    arm = {"curve": [{"cpu_seconds": 1, "fidelity": 0.5}, {"cpu_seconds": 2, "fidelity": 0.9}]}
    assert study.at_budget(arm, 1.5)["fidelity"] == pytest.approx(0.5)
    assert study.at_budget(arm, 2)["fidelity"] == pytest.approx(0.9)
    with pytest.raises(ValueError, match="precedes"):
        study.at_budget(arm, 0)
    case, cfg = small_case(), config()
    target = comparison.target_vector(case["target"])
    warm = study.train_block(study.initial_state(case), target, "adam", cfg, 123, 0, 4)
    ledger = [{"role": "warmup", "cpu_seconds": 1.0, "wall_seconds": 1.0}]
    prefix = {**warm, "ledger": ledger, "curve": [study.point(warm["state"], warm["fidelity"], ledger)]}
    control = study.fixed_control(prefix, target, "adam", cfg, 123, 1e9)
    assert control["covered_budget"] is False
    assert control["last_state"]["updates"] == 8
    assert len(control["blocks"]) == 2
    assert control["curve"][-1]["completed_updates"] == 8
    assert control["fidelity"] == max([warm["fidelity"], *(b["fidelity"] for b in control["blocks"])])


def test_all_rejected_work_is_charged_and_round_noise_is_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """A flat exact objective retains input but pays for every trial and refreshes epochs."""
    case, cfg = small_case(), config()
    target = comparison.target_vector(case["target"])
    state = study.initial_state(case)
    state["updates"] = 4
    monkeypatch.setattr(study, "fidelity", lambda *_: 0.5)
    ledger = [{"role": "warmup", "cpu_seconds": 1.0, "wall_seconds": 1.0}]
    prefix = {"state": state, "fidelity": 0.5, "ledger": ledger, "curve": [study.point(state, 0.5, ledger)]}
    arm = study.restructure(prefix, target, "adam", cfg, 123)
    assert arm["state"] == state
    assert all(r["selected"] == 0 for r in arm["rounds"])
    assert len(arm["ledger"]) == 1 + 2 * 4
    assert study.cpu_cost(arm["ledger"]) > 1
    assert sum(r["role"] == "deletion_repair" for r in arm["ledger"]) == 4
    assert arm["rounds"][0]["candidates"][1]["trace"][0]["epoch"] == 5
    assert arm["rounds"][1]["candidates"][1]["trace"][0]["epoch"] == 7


def test_case_authentication_resume_parallel_science_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completed case resume is byte-preserving; structure decisions are timing-independent."""
    cases = [small_case("a", 11), small_case("b", 29)]
    monkeypatch.setattr(study, "load_cases", lambda _: ({"source": "engineering"}, cases))
    cfg = config()
    root = tmp_path / "serial"
    result = study.run(root, tmp_path, config=cfg, workers=1)
    assert len(result["rows"]) == 4
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert study.run(root, tmp_path, config=cfg, workers=2, resume=True) == result
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    other_root = tmp_path / "parallel"
    other = study.run(other_root, tmp_path, config=cfg, workers=2)
    for a, b in zip(result["jobs"], other["jobs"], strict=True):
        left = cast("dict[str, Any]", pilot.read_record(root / a["path"]))
        right = cast("dict[str, Any]", pilot.read_record(other_root / b["path"]))
        assert pilot.digest(left) == a["checksum"]
        assert [t["ensemble_digest"] for t in left["prefixes"]["adam"]["trace"]] == [
            t["ensemble_digest"] for t in left["prefixes"]["adaptive_online"]["trace"]
        ]
        for method in study.METHODS:
            assert left["structural"][method]["state"] == right["structural"][method]["state"]
            arm = left["structural"][method]
            assert arm["curve"][-1]["cpu_seconds"] == pytest.approx(study.cpu_cost(arm["ledger"]))
            assert len(arm["ledger"]) == 2 + cfg.rounds * (2 + cfg.shortlist)
            assert len(arm["state"]["circuit"]["gates"]) >= 9 - cfg.rounds
            assert [r["selected"] for r in arm["rounds"]] == [
                r["selected"] for r in right["structural"][method]["rounds"]
            ]
    with pytest.raises(ValueError, match="identical manifest"):
        study.run(root, tmp_path, config=replace(cfg, rate=0.03), resume=True)
    path = root / result["jobs"][0]["path"]
    record = pilot.read_record(path)
    record["signature"] = "tampered"
    pilot.write_record(path, record)
    with pytest.raises(ValueError, match="signature"):
        study.run(root, tmp_path, config=cfg, resume=True, workers=1)


def test_source_initial_state_rejects_shared_parameters() -> None:
    """Deletion-specific state mapping must reject unsupported shared parameter layouts."""
    case = small_case()
    case["circuit"]["gates"][-1]["param_index"] = 0
    with pytest.raises(ValueError, match="unique"):
        study.initial_state(case)
