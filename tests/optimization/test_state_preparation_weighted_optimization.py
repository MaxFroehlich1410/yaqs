# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Weighted optimizer pairing, kernel reuse, phase-safe resume and accounting."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from benchmarks.state_preparation.weighted_sampling import FactorSampler
from experiments import state_preparation_exact_adam as exact
from experiments import state_preparation_pilot as pilot
from experiments import state_preparation_trajectory_comparison as comparison
from experiments import state_preparation_trajectory_diagnostics as development
from experiments import state_preparation_weighted_optimization as study
from experiments.state_preparation_weighted_sampling import weighted_gradient
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate, StatevectorBackend

if TYPE_CHECKING:
    from pathlib import Path


def small_case(name: str = "small") -> dict[str, Any]:
    """Build a frozen complex-target source with an independently scored warm start.

    Returns:
        A synthetic two-qubit development case.
    """
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), param_index=0),
            ParameterizedGate("rxx", (1, 0), param_index=1),
            ParameterizedGate("rz", (0,), param_index=0, angle_scale=-0.7),
        ],
    )
    target = np.array([1, 2j, -3, 4j]) / np.sqrt(30)
    theta = np.array([0.4, -0.7])
    fidelity = density_loss_gradient(circuit, theta, target, CalibratedLocalDepolarizingProvider())[2]
    return {
        "target": comparison.target_record(name, "haar", target),
        "circuit": pilot.circuit_record(circuit),
        "initial": theta.tolist(),
        "checkpoints": [{"update": 0, "theta": theta.tolist(), "exact_noisy_fidelity": fidelity}],
    }


@pytest.mark.parametrize("tail", ["batch", "adam", "momentum", "rmsprop", "backtracking", "online_rmsprop"])
@pytest.mark.parametrize("refresh", [0, 1, 3])
@pytest.mark.parametrize("stop", [1, 3])
def test_weighted_resume_preserves_joint_updates_moments_and_streams(
    tmp_path: Path, tail: str, refresh: int, stop: int
) -> None:
    """Interrupted weighted runs reproduce both phases, including cached-ensemble replay."""
    case = small_case()
    config = development.Config(updates=5, online_updates=2, trajectories=4)
    variant = development.Variant("hybrid", 2, tail=tail, rate=0.02, refresh=refresh)
    path = tmp_path / "partial.json"
    development.optimize(path, case, variant, config, 11, "test", sampling="weighted", stop_after=stop)
    resumed = development.optimize(path, case, variant, config, 11, "test", sampling="weighted")
    fresh = development.optimize(tmp_path / "fresh.json", case, variant, config, 11, "test", sampling="weighted")
    for key in ("theta", "first", "second", "online_counts", "tail_step", "completed"):
        assert resumed[key] == fresh[key]
    for a, b in zip(resumed["trace"], fresh["trace"], strict=True):
        assert {k: v for k, v in a.items() if k not in {"sampling", "update_cost"}} == {
            k: v for k, v in b.items() if k not in {"sampling", "update_cost"}
        }
        assert len(a["path_weights"]) == 4
        assert a["event_free_trajectories"] == 1
    before = path.read_bytes()
    assert development.optimize(path, case, variant, config, 11, "test", sampling="weighted") == resumed
    assert before == path.read_bytes()
    assert resumed["online_counts"] == ([6, 3] if tail == "online_rmsprop" else [0, 0])
    with pytest.raises(ValueError, match="configuration mismatch"):
        development.optimize(path, case, variant, config, 11, "test", sampling="ordinary")


def test_weighted_adam_matches_separate_stratum_estimator(tmp_path: Path) -> None:
    """Verify actual optimizer wiring with the independently implemented diagnostic helper."""
    case = small_case()
    config = development.Config(updates=4, online_updates=2, trajectories=8)
    engine = StatevectorBackend(pilot.restore_circuit(case["circuit"]), comparison.target_vector(case["target"]))
    theta = np.array(case["initial"])
    sampler = FactorSampler(engine.circuit, CalibratedLocalDepolarizingProvider())
    seed = development.seed_for(config, "training", "small", 11)
    first, second = np.zeros_like(theta), np.zeros_like(theta)
    for step in range(1, 5):
        maps = sampler.maps(sampler.draw(7, seed + 1_000_003 * step))
        gradient, _ = weighted_gradient(engine, theta, sampler, maps)
        theta, first, second = exact.adam_step(
            theta, gradient, first, second, step, exact.AdamConfig(learning_rate=0.02)
        )
    result = development.optimize(
        tmp_path / "adam.json",
        case,
        development.Variant("adam", 0, tail="adam", rate=0.02),
        config,
        11,
        "test",
        sampling="weighted",
    )
    np.testing.assert_allclose(result["theta"], theta, atol=1e-14)
    np.testing.assert_allclose(result["first"], first, atol=1e-14)
    np.testing.assert_allclose(result["second"], second, atol=1e-14)


def test_matched_screen_parallel_pairing_costs_and_authenticated_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All arms use identical maps/weights within a sampler, budget, target and noise seed."""
    cases = [small_case("a"), small_case("b")]
    monkeypatch.setattr(development, "load_cases", lambda _: ({"source": "test"}, cases))
    config = development.Config(updates=3, online_updates=2, replicas=(11,))
    root = tmp_path / "serial"
    result = study.run(root, tmp_path, config=config, counts=(2, 4), workers=1)
    assert len(result["rows"]) == 24
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert study.run(root, tmp_path, config=config, counts=(2, 4), workers=2, resume=True) == result
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    parallel = tmp_path / "parallel"
    other = study.run(parallel, tmp_path, config=config, counts=(2, 4), workers=2)
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for a, b in zip(result["rows"], other["rows"], strict=True):
        left = cast("dict[str, Any]", pilot.read_record(root / a["path"]))
        right = cast("dict[str, Any]", pilot.read_record(parallel / b["path"]))
        assert left["theta"] == right["theta"]
        assert a["checksum"] == pilot.digest(left)
        assert a["training_cpu_seconds"] == pytest.approx(
            sum(s["cpu_seconds"] for s in left["setup"])
            + sum(t["sampling"]["cpu_seconds"] + t["update_cost"]["cpu_seconds"] for t in left["trace"])
        )
        assert a["curve"][-1]["training_cpu_seconds"] == pytest.approx(a["training_cpu_seconds"])
        assert a["curve"][-1]["fidelity"] == a["final_fidelity"]
        expected_passes = (
            3 * 3 if a["variant"] == "all_online" else 3 * 2 if a["variant"] == "trajectory_adam_0p02" else 2 * 3 + 2
        ) * a["trajectories"]
        assert a["trajectory_passes"] == expected_passes
        groups.setdefault((a["target_id"], a["sampling"], a["trajectories"]), []).append(left)
    for records in groups.values():
        streams = [[t["ensemble_digest"] for t in r["trace"]] for r in records]
        assert streams[0] == streams[1] == streams[2]
        assert [e["fidelity"] for e in records[0]["evaluations"][:3]] == [
            e["fidelity"] for e in records[1]["evaluations"][:3]
        ]
    first_path = root / result["rows"][0]["path"]
    record = pilot.read_record(first_path)
    record["signature"] = "corrupted"
    pilot.write_record(first_path, record)
    with pytest.raises(ValueError, match="configuration mismatch"):
        study.run(root, tmp_path, config=config, counts=(2, 4), resume=True)


def test_guards_cli_and_runtime_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid budgets and changed source code cannot silently mix scientific runs."""
    for counts in [(), (1,), (2, 2), (True,)]:
        with pytest.raises(ValueError, match="two total paths"):
            study.run(tmp_path, tmp_path, counts=counts)
    with pytest.raises(ValueError, match="nonempty"):
        study.run(tmp_path, tmp_path, config=development.Config(online_updates=0))
    case = small_case()
    config = development.Config(updates=2, online_updates=1, trajectories=2, replicas=(11,))
    for sampling in ["bad", "weighted"]:
        with pytest.raises(ValueError, match="sampling policy"):
            development.optimize(
                tmp_path / "bad.json",
                case,
                development.Variant("adam", 0),
                replace(config, trajectories=1),
                11,
                "test",
                sampling=sampling,
            )
    monkeypatch.setattr(development, "load_cases", lambda _: ({}, [case]))
    root = tmp_path / "run"
    study.run(root, tmp_path, config=config, counts=(2,), workers=1)
    monkeypatch.setattr(study.budget, "runtime", lambda: {"code": "changed"})
    with pytest.raises(ValueError, match="identical manifest"):
        study.run(root, tmp_path, config=config, counts=(2,), workers=1, resume=True)
    calls: list[dict[str, Any]] = []

    def capture(_root: Path, _source: Path, **kwargs: object) -> dict[str, Any]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(study, "run", capture)
    assert study.main(["--source-root", str(tmp_path), "--output-root", str(root), "--workers", "2", "--resume"]) == 0
    assert calls == [{"workers": 2, "resume": True}]
