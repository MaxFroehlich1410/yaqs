# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the minimal state-preparation benchmark runner."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import pytest

from benchmarks.state_preparation import TrainingOutcome, TrainingSettings, load_target_collection
from benchmarks.state_preparation.runner import (
    BenchmarkConfig,
    BenchmarkError,
    derive_seed,
    main,
    preset_config,
    resolve_arguments,
    run_benchmark,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from benchmarks.state_preparation.methods import StatePreparationMethod
    from mqt.yaqs.optimization import ParameterizedCircuit


@dataclass
class CountingIdentityMethod:
    """Test adapter that preserves initialization and counts training calls."""

    method_id: str = "identity_test"
    calls: int = 0

    def train(
        self,
        circuit: ParameterizedCircuit,
        target: NDArray[np.complex128],
        initial_theta: NDArray[np.float64],
        settings: TrainingSettings,
    ) -> TrainingOutcome:
        """Return unchanged parameters with a minimal serializable trace."""
        assert target.shape == (2**circuit.num_qubits,)
        self.calls += 1
        return TrainingOutcome(
            theta=initial_theta.copy(),
            trace={"step": (0,), "method": (self.method_id,), "optimizer_seed": (settings.optimizer_seed,)},
            wall_time_seconds=0.0,
        )


def _config(output: Path, *, resume: bool = False) -> BenchmarkConfig:
    """Return a bounded configuration exercising all evaluation paths."""
    return BenchmarkConfig(
        preset="test",
        output_directory=output,
        num_qubits=(6,),
        target_ids=("gaussian_mu0p5_sigma0p1",),
        noise_ids=("noiseless", "ballarin_coupled", "dephasing_2s_2q"),
        method_ids=("identity_test",),
        bmpd_depths=(1,),
        initialization_seeds=(11,),
        optimizer_iterations=0,
        test_trajectories=2,
        resume=resume,
    )


def _rows(path: Path) -> list[dict[str, object]]:
    """Return decoded result rows."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_presets_have_documented_matrix_sizes() -> None:
    """Smoke, minimum, and full presets expand to 12, 108, and 216 rows."""
    assert preset_config("smoke").result_cells == 12
    assert preset_config("minimum").result_cells == 108
    assert preset_config("minimum").training_runs == 18
    assert preset_config("full").result_cells == 216
    with pytest.raises(ValueError, match="Unknown preset"):
        preset_config("unknown")


def test_seed_derivation_is_stable_and_domain_separated() -> None:
    """Evaluation seeds do not depend on execution order or collide by role."""
    assert derive_seed(13, "final-test", "ballarin_coupled") == derive_seed(13, "final-test", "ballarin_coupled")
    assert derive_seed(13, "final-test", "ballarin_coupled") != derive_seed(13, "final-test", "dephasing_2s_2q")


def test_runner_trains_once_and_emits_complete_noiseless_standard_and_ballarin_rows(tmp_path: Path) -> None:
    """One optimization fans out into complete, independently seeded result rows."""
    method = CountingIdentityMethod()
    config = _config(tmp_path / "benchmark")

    summary = run_benchmark(
        config,
        methods={method.method_id: method},
        targets=load_target_collection(),
        fail_fast=True,
    )
    rows = _rows(summary.results_path)

    assert method.calls == 1
    assert summary.successful_rows == 3
    assert summary.failed_rows == 0
    assert summary.trained_runs == 1
    assert {row["noise_id"] for row in rows} == set(config.noise_ids)
    assert len({row["checkpoint"] for row in rows}) == 1
    assert all(row["test_noiseless_fidelity"] == row["train_fidelity"] for row in rows)

    noiseless = next(row for row in rows if row["noise_id"] == "noiseless")
    ballarin = next(row for row in rows if row["noise_id"] == "ballarin_coupled")
    standard = next(row for row in rows if row["noise_id"] == "dephasing_2s_2q")
    ballarin_statistics = cast("dict[str, object]", ballarin["circuit_statistics"])
    standard_statistics = cast("dict[str, object]", standard["circuit_statistics"])
    standard_interval = cast("dict[str, float]", standard["confidence_interval"])
    assert noiseless["noise_seed"] is None
    assert noiseless["test_trajectories_or_shots"] == 0
    assert noiseless["confidence_interval"] is None
    assert noiseless["noisy_fidelity_standard_error"] is None
    assert ballarin_statistics["native"] is not None
    assert standard_statistics["native"] is None
    assert len(cast("list[float]", ballarin["trajectory_fidelities"])) == 2
    assert standard_interval["level"] == pytest.approx(0.95)
    assert standard["noisy_fidelity_standard_error"] is not None


def test_resume_skips_successes_and_reuses_checkpoint_for_a_missing_row(tmp_path: Path) -> None:
    """Resume does not retrain and deterministically reconstructs a missing result."""
    method = CountingIdentityMethod()
    config = _config(tmp_path / "benchmark")
    first = run_benchmark(config, methods={method.method_id: method}, fail_fast=True)
    retained = _rows(first.results_path)
    first.results_path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in retained if row["noise_id"] != "ballarin_coupled"),
        encoding="utf-8",
    )

    resumed = run_benchmark(replace(config, resume=True), methods={method.method_id: method}, fail_fast=True)

    assert method.calls == 1
    assert resumed.successful_rows == 3
    assert resumed.skipped_rows == 2
    assert resumed.reused_checkpoints == 1
    assert len(_rows(resumed.results_path)) == 3


def test_resume_rejects_manifest_mismatch(tmp_path: Path) -> None:
    """Changing a scientific setting cannot reuse an incompatible output directory."""
    method = CountingIdentityMethod()
    config = _config(tmp_path / "benchmark")
    run_benchmark(config, methods={method.method_id: method}, fail_fast=True)

    with pytest.raises(BenchmarkError, match="does not match"):
        run_benchmark(
            replace(config, resume=True, test_trajectories=3),
            methods={method.method_id: method},
            fail_fast=True,
        )


def test_overwrite_replaces_only_owned_artifacts(tmp_path: Path) -> None:
    """Overwrite restarts the benchmark while preserving unrelated output files."""
    method = CountingIdentityMethod()
    config = _config(tmp_path / "benchmark")
    run_benchmark(config, methods={method.method_id: method}, fail_fast=True)
    sentinel = config.output_directory / "keep-me.txt"
    sentinel.write_text("unrelated\n", encoding="utf-8")

    summary = run_benchmark(replace(config, overwrite=True), methods={method.method_id: method}, fail_fast=True)

    assert summary.successful_rows == 3
    assert summary.trained_runs == 1
    assert sentinel.read_text(encoding="utf-8") == "unrelated\n"


def test_resume_requires_a_retained_manifest(tmp_path: Path) -> None:
    """Orphaned result data cannot be adopted as a compatible run."""
    config = _config(tmp_path / "benchmark", resume=True)
    config.output_directory.mkdir()
    (config.output_directory / "results.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(BenchmarkError, match="no retained benchmark manifest"):
        run_benchmark(config, methods={"identity_test": CountingIdentityMethod()})


def test_existing_owned_artifact_requires_an_explicit_output_policy(tmp_path: Path) -> None:
    """A partial prior output cannot be replaced without resume or overwrite."""
    config = _config(tmp_path / "benchmark")
    config.output_directory.mkdir()
    (config.output_directory / "results.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(BenchmarkError, match="benchmark-owned artifacts"):
        run_benchmark(config, methods={"identity_test": CountingIdentityMethod()})


def test_default_krotov_adapter_completes_a_bounded_noiseless_run(tmp_path: Path) -> None:
    """The bundled method adapter is executable without custom wiring."""
    config = BenchmarkConfig(
        preset="test",
        output_directory=tmp_path / "krotov",
        num_qubits=(6,),
        target_ids=("gaussian_mu0p5_sigma0p1",),
        noise_ids=("noiseless",),
        bmpd_depths=(0,),
        optimizer_iterations=0,
        test_trajectories=1,
    )

    summary = run_benchmark(config, fail_fast=True)

    assert summary.successful_rows == 1
    assert summary.failed_rows == 0
    assert summary.trained_runs == 1


def test_argument_resolution_and_dry_run_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI overrides remain typed and dry-run creates no output."""
    output = tmp_path / "dry-run"
    arguments = [
        "--preset",
        "minimum",
        "--output-dir",
        str(output),
        "--num-qubits",
        "6",
        "--target-id",
        "tfim_critical",
        "--noise-id",
        "noiseless",
        "--optimizer-iterations",
        "0",
        "--dry-run",
    ]
    config, dry_run, fail_fast = resolve_arguments(arguments)

    assert config.result_cells == 1
    assert dry_run
    assert not fail_fast
    assert main(arguments) == 0
    assert '"result_cells": 1' in capsys.readouterr().out
    assert not output.exists()


def test_missing_method_adapter_is_rejected_before_output(tmp_path: Path) -> None:
    """An unspecified competitor cannot silently fall back to Krotov."""
    config = replace(_config(tmp_path / "benchmark"), method_ids=("missing",))

    with pytest.raises(BenchmarkError, match="No method adapters"):
        run_benchmark(config)
    assert not config.output_directory.exists()


def test_method_protocol_is_runtime_compatible() -> None:
    """The custom adapter exposes the complete structural method contract."""
    method: StatePreparationMethod = CountingIdentityMethod()
    assert method.method_id == "identity_test"
