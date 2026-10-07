# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Hybrid phase boundaries, paired sampling, restarts, and multi-target tests."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider
from experiments import state_preparation_hybrid_pilot as hybrid
from experiments import state_preparation_pilot as pilot
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KrotovOptions,
    KrotovTJMOptions,
    ParameterizedCircuit,
    ParameterizedGate,
    train_krotov_noisy_state_preparation_hybrid,
    train_krotov_state_preparation_hybrid,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from numpy.typing import NDArray


def small_problem() -> tuple[ParameterizedCircuit, NDArray[np.float64], NDArray[np.complex128]]:
    """Return an interacting circuit with nontrivial parameter scales and phases."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), param_index=0, angle_scale=-1.3, angle_offset=0.2),
            ParameterizedGate("rx", (1,), param_index=1),
            ParameterizedGate("rxx", (0, 1), param_index=2),
            ParameterizedGate("rz", (0,), param_index=3),
        ],
    )
    rng = np.random.default_rng(44)
    target = rng.normal(size=4) + 1j * rng.normal(size=4)
    target /= np.linalg.norm(target)
    return circuit, rng.normal(size=4), target


@pytest.mark.parametrize("trajectories", [0, 3])
def test_statevector_blocks_match_mps_and_preserve_chunk_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trajectories: int
) -> None:
    """Dense growth/repair chunks preserve phase, global sampling counters and inputs."""
    circuit, initial, target = small_problem()
    before = initial.copy()
    provider = ExactLocalPauliProvider(CalibratedLocalDepolarizingProvider(0.15, 0.3))
    monkeypatch.setattr(hybrid, "ExactLocalPauliProvider", lambda: provider)
    kwargs: dict[str, Any] = {
        "updates": 8,
        "batch_rate": 0.1,
        "schedule": hybrid.Schedule(3, 0.7),
        "seed": 432,
        "trajectories": trajectories,
    }
    mps = hybrid.optimize_block(tmp_path / "mps.json", circuit, initial, target, **kwargs)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Dense construction/repair constructed an MPS")

    monkeypatch.setattr(MPS, "__init__", forbidden)
    dense = hybrid.optimize_block(
        tmp_path / "dense.json", circuit, initial, target, backend="statevector", checkpoint_every=2, **kwargs
    )
    whole = hybrid.optimize_block(
        tmp_path / "whole.json", circuit, initial, target, backend="statevector", checkpoint_every=8, **kwargs
    )
    np.testing.assert_allclose(
        np.asarray(dense["theta"], dtype=float), np.asarray(mps["theta"], dtype=float), atol=2e-12
    )
    np.testing.assert_array_equal(dense["theta"], whole["theta"])
    np.testing.assert_array_equal(initial, before)
    assert dense["trace"] == whole["trace"]
    assert (
        hybrid.optimize_block(tmp_path / "dense.json", circuit, initial, target, backend="statevector", **kwargs)
        == dense
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        hybrid.optimize_block(tmp_path / "dense.json", circuit, initial, target, **kwargs)


@pytest.mark.parametrize(
    ("switch", "rate"), [(5, 0.5), (10, 0.5), (20, 0.5), (25, 0.7), (25, 0.9), (30, 0.7), (30, 0.9)]
)
@pytest.mark.parametrize("trajectories", [0, 2])
def test_chunked_hybrid_matches_public_full_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, switch: int, rate: float, trajectories: int
) -> None:
    """Chunks preserve the full online prefix, batch tail, and noisy seed counters."""
    circuit, initial, target = small_problem()
    provider = ExactLocalPauliProvider(CalibratedLocalDepolarizingProvider(0.15, 0.3))
    monkeypatch.setattr(hybrid, "ExactLocalPauliProvider", lambda: provider)
    updates = max(23, switch + 3)
    options = KrotovOptions(max_iterations=updates, switch_iteration=switch, online_step_size=rate, batch_step_size=0.1)
    if trajectories:
        reference = train_krotov_noisy_state_preparation_hybrid(
            circuit,
            target,
            None,
            KrotovTJMOptions(num_trajectories=trajectories, random_seed=45),
            initial_theta=initial,
            options=options,
            noise_provider=provider,
        )
    else:
        reference = train_krotov_state_preparation_hybrid(circuit, target, initial_theta=initial, options=options)
    actual = hybrid.optimize_block(
        tmp_path / "block.json",
        circuit,
        initial,
        target,
        updates=updates,
        batch_rate=0.1,
        schedule=hybrid.Schedule(switch, rate),
        seed=45,
        trajectories=trajectories,
        checkpoint_every=7,
    )
    np.testing.assert_array_equal(actual["theta"], reference.theta)
    rows = cast("list[dict[str, object]]", actual["trace"])
    assert len(rows) == updates
    assert [row["phase"] for row in rows] == ["online"] * switch + ["batch"] * (updates - switch)
    assert [row["step_size"] for row in rows] == [rate] * switch + [0.1] * (updates - switch)
    for key, values in reference.trace.items():
        assert [row[key] for row in rows] == values[1:]


@pytest.mark.parametrize("stop_after", [1, 2])
@pytest.mark.parametrize("backend", ["mps", "statevector"])
def test_resume_before_and_after_hybrid_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_after: int, backend: str
) -> None:
    """Partial noisy runs resume identically on either side of the phase boundary."""
    circuit, initial, target = small_problem()
    write_record = pilot.write_record
    calls = 0

    def interrupted(path: Path, record: dict[str, object]) -> None:
        """Persist progress, then simulate a process interruption.

        Raises:
            RuntimeError: At the selected checkpoint.
        """
        nonlocal calls
        write_record(path, record)
        calls += 1
        if calls == stop_after:
            msg = "Simulated interruption"
            raise RuntimeError(msg)

    path = tmp_path / "interrupted.json"
    with monkeypatch.context() as patch:
        patch.setattr(pilot, "write_record", interrupted)
        with pytest.raises(RuntimeError, match="Simulated interruption"):
            hybrid.optimize_block(
                path,
                circuit,
                initial,
                target,
                updates=12,
                batch_rate=0.1,
                schedule=hybrid.Schedule(5, 0.5),
                seed=46,
                trajectories=2,
                checkpoint_every=3,
                backend=backend,
            )
    assert pilot.read_record(path)["completed_updates"] == 3 * stop_after
    resumed = hybrid.optimize_block(
        path,
        circuit,
        initial,
        target,
        updates=12,
        batch_rate=0.1,
        schedule=hybrid.Schedule(5, 0.5),
        seed=46,
        trajectories=2,
        checkpoint_every=4,
        backend=backend,
    )
    fresh = hybrid.optimize_block(
        tmp_path / "fresh.json",
        circuit,
        initial,
        target,
        updates=12,
        batch_rate=0.1,
        schedule=hybrid.Schedule(5, 0.5),
        seed=46,
        trajectories=2,
        checkpoint_every=10,
        backend=backend,
    )
    assert resumed["theta"] == fresh["theta"]
    assert resumed["trace"] == fresh["trace"]
    with pytest.raises(ValueError, match="configuration mismatch"):
        hybrid.optimize_block(
            path,
            circuit,
            initial,
            target,
            updates=12,
            batch_rate=0.1,
            schedule=hybrid.Schedule(5, 0.2),
            seed=46,
            trajectories=2,
        )


def test_invalid_schedules_and_budgets_do_not_create_output(tmp_path: Path) -> None:
    """Unsupported schedule grids or zero budgets fail before artifact creation."""
    for steps, rate in ((-1, 0.1), (True, 0.1), (5, 0), (5, float("nan"))):
        with pytest.raises(ValueError, match="schedule"):
            hybrid.Schedule(steps, rate)
    circuit, initial, target = small_problem()
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="budget"):
        hybrid.optimize_block(
            output, circuit, initial, target, updates=5, batch_rate=0.1, schedule=hybrid.Schedule(5, 0.1), seed=0
        )
    with pytest.raises(ValueError, match="Schedules"):
        hybrid.run_screen(output, tmp_path / "absent", schedules=(hybrid.Schedule(), hybrid.Schedule()))
    with pytest.raises(ValueError, match="Schedules"):
        hybrid.run_screen(
            output,
            tmp_path / "absent",
            schedules=(hybrid.Schedule(100, 0.9),),
            include_warm=False,
        )
    # The short final refinement is batch-only, so it does not limit the
    # online prefix in the longer source-growth blocks.
    with pytest.raises(FileNotFoundError):
        hybrid.run_screen(
            output,
            tmp_path / "absent",
            config=pilot.PilotConfig(refinement_updates=1),
            schedules=(hybrid.Schedule(30, 0.9),),
            include_warm=False,
        )
    assert not output.exists()


def test_multitarget_hybrid_parallel_reproduction_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both scientific strata run on all four targets and reproduce batch controls."""
    config = pilot.PilotConfig(
        source_updates=2, repair_updates=1, refinement_updates=2, training_trajectories=1, seeds=(11,)
    )
    source = tmp_path / "source"
    # This fixture exercises the M-checkpoint consumer, not the separate
    # nomination rule. Tiny engineering budgets otherwise nominate D.
    with monkeypatch.context() as patch:
        patch.setattr(pilot, "choose_nominee", lambda _constructions: "M")
        pilot.run_pilot(source, config=config, workers=1)
    schedules = (hybrid.Schedule(), hybrid.Schedule(1, 0.5))
    serial = hybrid.run_screen(tmp_path / "serial", source, config=config, schedules=schedules, workers=1)
    parallel = hybrid.run_screen(tmp_path / "parallel", source, config=config, schedules=schedules, workers=2)
    assert serial["aggregate"] == parallel["aggregate"]
    assert parallel["completed_jobs"] == 8
    for target in pilot.TARGETS:
        for setting in schedules:
            record = pilot.read_record(tmp_path / "parallel" / setting.label / target / "seed_11" / "result.json")
            candidates = cast("list[dict[str, object]]", record["candidates"])
            assert len(candidates) == 4
            assert len(cast("list[object]", record["pruning_events"])) == 6
            assert candidates[2]["initial_theta_checksum"] == candidates[3]["initial_theta_checksum"]
            if setting.online_updates == 0:
                errors = cast("dict[str, float]", record["batch_reproduction_max_parameter_errors"])
                assert max(errors.values()) <= 1e-11
    assert (
        hybrid.run_screen(tmp_path / "parallel", source, config=config, schedules=schedules, workers=1, resume=True)
        == parallel
    )
    with pytest.raises(ValueError, match="explicit resume"):
        hybrid.run_screen(tmp_path / "parallel", source, config=config, schedules=schedules)
    with pytest.raises(ValueError, match="identical manifest"):
        hybrid.run_screen(
            tmp_path / "parallel",
            source,
            config=config,
            schedules=(hybrid.Schedule(), hybrid.Schedule(1, 0.2)),
            resume=True,
        )
    with pytest.raises(ValueError, match="compatible"):
        hybrid.run_screen(tmp_path / "changed", source, config=replace(config, root_seed=47), schedules=schedules)
    extension_root = tmp_path / "extension"
    extension = hybrid.run_screen(
        extension_root, source, config=config, schedules=schedules, workers=2, include_warm=False
    )
    assert extension["version"] == hybrid.CONSTRUCTION_VERSION
    aggregate = cast("dict[str, object]", serial["aggregate"])
    assert extension["aggregate"] == {key: value for key, value in aggregate.items() if ":cold_" in key}
    for target in pilot.TARGETS:
        for setting in schedules:
            directory = extension_root / setting.label / target / "seed_11"
            record = pilot.read_record(directory / "result.json")
            assert record["include_warm"] is False
            assert len(cast("list[object]", record["candidates"])) == 2
            blocks = list((directory / "blocks").glob("*.json"))
            assert len(blocks) == 11
            assert sum(cast("int", pilot.read_record(block)["completed_updates"]) for block in blocks) == 16
    assert (
        hybrid.run_screen(
            extension_root, source, config=config, schedules=schedules, workers=1, include_warm=False, resume=True
        )
        == extension
    )
    with pytest.raises(ValueError, match="identical manifest"):
        hybrid.run_screen(extension_root, source, config=config, schedules=schedules, resume=True)
    # Construction-only screens may add settings without rerunning batch.
    without_batch = hybrid.run_screen(
        tmp_path / "without_batch",
        source,
        config=config,
        schedules=(hybrid.Schedule(1, 0.9),),
        workers=1,
        include_warm=False,
    )
    assert without_batch["completed_jobs"] == 4


def test_extension_cli_runs_only_requested_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The explicit extension selects exactly four new construction-only arms."""
    calls = []

    def capture(root: Path, source_root: Path, **kwargs: object) -> dict[str, object]:
        """Capture CLI dispatch without executing a scientific run.

        Returns:
            A minimal successful result for the CLI.
        """
        calls.append((root, source_root, kwargs))
        return {"status": "complete"}

    monkeypatch.setattr(hybrid, "run_screen", capture)
    assert (
        hybrid.main([
            "--source-root",
            str(tmp_path / "source"),
            "--output-root",
            str(tmp_path / "output"),
            "--construction-extension",
            "--workers",
            "2",
            "--resume",
        ])
        == 0
    )
    options = calls[0][2]
    assert options["schedules"] == (
        hybrid.Schedule(25, 0.7),
        hybrid.Schedule(25, 0.9),
        hybrid.Schedule(30, 0.7),
        hybrid.Schedule(30, 0.9),
    )
    assert options["include_warm"] is False
    assert options["workers"] == 2
    assert options["resume"] is True
