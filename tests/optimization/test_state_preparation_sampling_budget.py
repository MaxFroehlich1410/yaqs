# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Budget-screen source reuse, exact references, paired execution and resume checks."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from experiments import state_preparation_exact_adam as adam
from experiments import state_preparation_pilot as pilot
from experiments import state_preparation_sampling_budget as budget
from experiments import state_preparation_trajectory_comparison as comparison
from experiments import state_preparation_trajectory_diagnostics as development
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def case(name: str = "small") -> dict[str, Any]:
    """Construct a small complex-target development case.

    Returns:
        Shared starting circuit, vector and exact checkpoint fidelity.
    """
    circuit = ParameterizedCircuit(
        2,
        [ParameterizedGate("ry", (0,), param_index=0), ParameterizedGate("rxx", (1, 0), param_index=1)],
    )
    target = np.array([1, 2j, -3, 4j]) / np.sqrt(30)
    initial = np.array([0.4, -0.7])
    fidelity = density_loss_gradient(circuit, initial, target, CalibratedLocalDepolarizingProvider())[2]
    return {
        "target": comparison.target_record(name, "haar", target),
        "circuit": pilot.circuit_record(circuit),
        "initial": initial.tolist(),
        "checkpoints": [
            {"update": k, "theta": initial.tolist(), "exact_noisy_fidelity": fidelity} for k in (0, 25, 50)
        ],
    }


def fingerprints(root: Path) -> dict[str, bytes]:
    """Retain all artifact bytes to detect unintended rewrites during reuse.

    Returns:
        Relative artifact paths and their original contents.
    """
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_screen_serial_parallel_reuse_and_completed_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Old work is read-only, parallel science matches serial, and exact cases are unique."""
    cases = [case("a"), case("b")]
    source = {"source": "synthetic"}
    monkeypatch.setattr(development, "load_cases", lambda _: (source, cases))
    config = development.Config(updates=4, online_updates=2, trajectories=2)
    tail, serial, parallel = (tmp_path / name for name in ("tail", "serial", "parallel"))
    development.run(tail, tmp_path, study="tail", config=config)
    saved = fingerprints(tail)
    result = budget.run(serial, tmp_path, tail_root=tail, config=config, counts=(2, 4), workers=1)
    assert fingerprints(tail) == saved
    assert len(result["rows"]) == 30
    assert sum(row["reused"] for row in result["rows"]) == 8
    exact_rows = [row for row in result["rows"] if row["trajectories"] is None]
    assert len(exact_rows) == 2
    assert all(row["replica"] is None for row in exact_rows)
    before = fingerprints(serial)
    assert budget.run(serial, tmp_path, tail_root=tail, config=config, counts=(2, 4), workers=2, resume=True) == result
    assert fingerprints(serial) == before
    other = budget.run(parallel, tmp_path, config=config, counts=(2, 4), workers=2)
    for first, second in zip(result["rows"], other["rows"], strict=True):
        assert first["final_fidelity"] == second["final_fidelity"]
        assert [r["fidelity"] for r in first["curve"]] == [r["fidelity"] for r in second["curve"]]
        record = (
            pilot.read_record(_first_path := serial / first["path"])
            if not first["reused"]
            else pilot.read_record(tail / first["path"])
        )
        assert pilot.digest(record) == first["checksum"]
        assert all(
            b["training_cpu_seconds"] >= a["training_cpu_seconds"]
            for a, b in zip(first["curve"], first["curve"][1:], strict=False)
        )
    with pytest.raises(ValueError, match="identical manifest"):
        budget.run(serial, tmp_path, tail_root=tail, config=config, counts=(2, 8), resume=True)


def test_exact_reference_is_existing_adam_kernel(tmp_path: Path) -> None:
    """Exact-only training preserves the established moments and independent scoring."""
    item = case()
    config = development.Config(updates=4, online_updates=2)
    row = budget.run_job(budget.Job(tmp_path, item, config, "test"))
    theta = np.array(item["initial"])
    first = np.zeros_like(theta)
    second = np.zeros_like(theta)
    circuit, target = pilot.restore_circuit(item["circuit"]), comparison.target_vector(item["target"])
    for step in range(1, 5):
        gradient = density_loss_gradient(circuit, theta, target, CalibratedLocalDepolarizingProvider())[0]
        theta, first, second = adam.adam_step(theta, gradient, first, second, step, adam.AdamConfig(learning_rate=0.02))
    record = pilot.read_record(tmp_path / row["path"])
    np.testing.assert_array_equal(record["theta"], theta)
    np.testing.assert_array_equal(record["first_moment"], first)
    assert row["training_cpu_seconds"] == row["curve"][-1]["training_cpu_seconds"]
    assert row["evaluation_cpu_seconds"] is None
    assert row["final_fidelity"] == pytest.approx(
        density_loss_gradient(circuit, theta, target, CalibratedLocalDepolarizingProvider())[2], abs=1e-12
    )


@pytest.mark.parametrize("change", ["runtime", "row", "incomplete", "config"])
def test_reuse_rejects_incompatible_or_damaged_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Never silently reuse stale software, mismatched records or unfinished work."""
    cases, source = [case()], {"source": "test"}
    config = development.Config(updates=4, online_updates=2, trajectories=2)
    monkeypatch.setattr(development, "load_cases", lambda _: (source, cases))
    root = tmp_path / "tail"
    result = development.run(root, tmp_path, study="tail", config=config)
    if change == "runtime":
        original = budget.runtime()
        monkeypatch.setattr(budget, "runtime", lambda: {**original, "code": "changed"})
    elif change == "row":
        result["rows"][-1]["checksum"] = "corrupt"
        pilot.write_record(root / "result.json", result)
    elif change == "incomplete":
        path = root / result["rows"][-1]["path"]
        record = pilot.read_record(path)
        record["status"] = "running"
        pilot.write_record(path, record)
    else:
        config = replace(config, root_seed=8)
    before = fingerprints(root)
    with pytest.raises(ValueError, match=r"Tail|tail"):
        budget.reuse_tail(root, tmp_path, source, cases, config)
    assert before == fingerprints(root)


@pytest.mark.parametrize("counts", [(), (2, 2), (0,), (-1,), (True,)])
def test_invalid_counts_rejected_before_output(tmp_path: Path, counts: tuple[int, ...]) -> None:
    """Malformed trajectory grids do not create scientific artifacts."""
    with pytest.raises(ValueError, match="counts"):
        budget.run(tmp_path / "new", tmp_path, counts=counts)
    assert not (tmp_path / "new").exists()


def test_cli_and_defaults() -> None:
    """Scientific rates are explicit and the command-line interface is discoverable."""
    config = development.Config()
    methods = budget.methods(config)
    assert [(m.online_updates, m.tail, m.rate) for m in methods] == [
        (50, "batch", 0.1),
        (25, "batch", 0.3),
        (0, "adam", 0.02),
    ]
    with pytest.raises(SystemExit) as error:
        budget.main(["--help"])
    assert error.value.code == 0
