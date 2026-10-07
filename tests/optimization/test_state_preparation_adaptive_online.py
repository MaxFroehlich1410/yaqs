# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Adaptive-online experiment pairing, phase transitions, budgets and resume."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from experiments import state_preparation_adaptive_online as study
from experiments import state_preparation_pilot as pilot
from experiments import state_preparation_trajectory_comparison as comparison
from experiments import state_preparation_trajectory_diagnostics as development
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def small_case(name: str) -> dict[str, Any]:
    """Create a shared-parameter synthetic source for engineering tests.

    Returns:
        An authenticated-shape case with an exact initial fidelity.
    """
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), param_index=0),
            ParameterizedGate("rxx", (1, 0), param_index=1),
            ParameterizedGate("rz", (1,), param_index=0, angle_scale=-0.7),
        ],
    )
    theta = np.array([0.4, -0.7])
    target = np.array([1, 2j, -3, 4j]) / np.sqrt(30)
    fidelity = density_loss_gradient(circuit, theta, target, CalibratedLocalDepolarizingProvider())[2]
    return {
        "target": comparison.target_record(name, "haar", target),
        "circuit": pilot.circuit_record(circuit),
        "initial": theta.tolist(),
        "checkpoints": [{"update": 0, "theta": theta.tolist(), "exact_noisy_fidelity": fidelity}],
    }


def test_presets_are_exactly_the_agreed_eight_methods() -> None:
    """Adam and adaptive-online get the same rate screen, with two fixed controls."""
    variants = study.variants(development.Config())
    assert len(variants) == 8
    assert len({v.name for v in variants}) == 8
    assert [v.rate for v in variants[2:5]] == [0.005, 0.01, 0.02]
    assert [v.rate for v in variants[5:]] == [0.005, 0.01, 0.02]
    assert [v.online_updates for v in variants] == [50, 25, 0, 0, 0, 25, 25, 25]
    assert [v.tail for v in variants[5:]] == ["online_rmsprop"] * 3
    assert all(v.refresh == 1 for v in variants)
    with pytest.raises(ValueError, match="gate-sequential updater"):
        development.tail_update(
            cast("Any", None), np.zeros(1), (), variants[-1], np.zeros(1), np.zeros(1), 1, development.Config()
        )


def test_screen_parallel_determinism_pairing_and_accounting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """All eight methods share streams; adaptive tails retain the online pass budget."""
    cases = [small_case("a"), small_case("b")]
    monkeypatch.setattr(development, "load_cases", lambda _: ({"source": "test"}, cases))
    config = development.Config(updates=4, online_updates=2, trajectories=4, replicas=(11,))
    root = tmp_path / "serial"
    result = study.run(root, tmp_path, config=config, workers=1)
    assert len(result["rows"]) == 16
    manifest = cast("dict[str, Any]", pilot.read_record(root / "manifest.json"))
    assert manifest["samplers"] == ["weighted"]
    assert manifest["counts"] == [4]
    assert manifest["extension"]["version"] == study.VERSION
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert study.run(root, tmp_path, config=config, workers=2, resume=True) == result
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    parallel = tmp_path / "parallel"
    other = study.run(parallel, tmp_path, config=config, workers=2)
    groups: dict[str, list[dict[str, Any]]] = {}
    for left, right in zip(result["rows"], other["rows"], strict=True):
        record = cast("dict[str, Any]", pilot.read_record(root / left["path"]))
        second = cast("dict[str, Any]", pilot.read_record(parallel / right["path"]))
        assert record["theta"] == second["theta"]
        assert record["second"] == second["second"]
        assert record["online_counts"] == second["online_counts"]
        assert left["checksum"] == pilot.digest(record)
        assert left["curve"][-1]["training_cpu_seconds"] == pytest.approx(left["training_cpu_seconds"])
        assert left["final_fidelity"] == left["curve"][-1]["fidelity"]
        adaptive = left["variant"].startswith("adaptive_online")
        assert record["online_counts"] == ([4, 2] if adaptive else [0, 0])
        expected = (
            12
            if adaptive or left["variant"] == "all_online"
            else 10
            if left["variant"].startswith("krotov_adam")
            else 8
        )
        assert left["trajectory_passes"] == 4 * expected
        assert all(t["objective_evaluations"] == 0 for t in record["trace"])
        if adaptive:
            assert [t["phase"] for t in record["trace"]] == ["online", "online", "online_rmsprop", "online_rmsprop"]
            assert record["first"] == [0.0, 0.0]
        groups.setdefault(left["target_id"], []).append(record)
    for group in groups.values():
        assert all(
            [t["ensemble_digest"] for t in r["trace"]] == [t["ensemble_digest"] for t in group[0]["trace"]]
            for r in group
        )
        for index in (1, 5, 6, 7):
            assert [e["fidelity"] for e in group[index]["evaluations"][:3]] == [
                e["fidelity"] for e in group[0]["evaluations"][:3]
            ]
    with pytest.raises(ValueError, match="identical manifest"):
        study.run(root, tmp_path, config=replace(config, beta2=0.9), resume=True)


def test_switch_resets_adaptation_and_rejects_bad_counts(tmp_path: Path) -> None:
    """No moments leak from the unscaled prefix; saved counters must match occurrences."""
    case = small_case("a")
    config = development.Config(updates=4, online_updates=2, trajectories=4)
    variant = study.variants(config)[5]
    path = tmp_path / "run.json"
    prefix = development.optimize(path, case, variant, config, 11, "test", sampling="weighted", stop_after=2)
    assert prefix["second"] == [0, 0]
    assert prefix["online_counts"] == [0, 0]
    completed = development.optimize(path, case, variant, config, 11, "test", sampling="weighted")
    assert completed["online_counts"] == [4, 2]
    for bad in ([2, 2], [4.0, 2.0], [4, -2], [4], [True, 2]):
        changed = {**completed, "online_counts": bad}
        pilot.write_record(path, changed)
        with pytest.raises(ValueError, match="progress"):
            development.optimize(path, case, variant, config, 11, "test", sampling="weighted")


def test_custom_screen_guards_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom method identities and samplers must be unambiguous and CLI remains fixed."""
    duplicate = development.Variant("same", 25)
    for kwargs in (
        {"variants": ()},
        {"variants": (duplicate, duplicate)},
        {"samplers": ()},
        {"samplers": ("weighted", "weighted")},
        {"samplers": ("unknown",)},
    ):
        with pytest.raises(ValueError, match="methods or sampling"):
            study.weighted.run(tmp_path, tmp_path, **cast("dict[str, Any]", kwargs))
    calls: list[dict[str, Any]] = []

    def capture(_root: Path, _source: Path, **kwargs: object) -> dict[str, Any]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(study, "run", capture)
    assert (
        study.main([
            "--source-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "out"),
            "--workers",
            "2",
            "--resume",
        ])
        == 0
    )
    assert calls == [{"workers": 2, "resume": True}]
