# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Empirical compression, preserved random streams, and paired checkpoint reruns."""

from __future__ import annotations

import json
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pytest

from experiments.legacy_topdown_dense import LegacyNoise, Recorder, target_state, train
from experiments.legacy_topdown_weighted import compression_diagnostic, load_circuit, run_case
from mqt.yaqs.optimization import KrotovTJMOptions, ParameterizedCircuit, ParameterizedGate, StatevectorBackend

if TYPE_CHECKING:
    from pathlib import Path


def problem() -> tuple:
    """Return two-qubit shared-angle gates with nonzero jump probability."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("rxx", (0, 1), param_index=0),
            ParameterizedGate("ry", (1,), param_index=1, angle_scale=-0.5),
        ]
        * 6,
    )
    return circuit, np.array([0.4, -0.8]), target_state(15, "haar", 2)


@pytest.mark.parametrize("crn", [False, True])
def test_compressed_training_matches_raw_hybrid_and_preserves_stream(*, crn: bool) -> None:
    """Five real updates match full sampling, including frozen and refreshed banks."""
    circuit, theta, target = problem()
    run = partial(train, circuit, theta, target, steps=5, online=3, rate=0.1, seed=300, noisy=True, crn=crn)
    original = theta.copy()
    full, full_trace = run(num_trajectories=100, record_noise_banks=True)
    short, short_trace = run(num_trajectories=100, compress_no_jump=True, record_noise_banks=True)
    _, old_trace = run(record_noise_banks=True)
    np.testing.assert_array_equal(theta, original)
    np.testing.assert_allclose(full, short, atol=1e-12)
    np.testing.assert_allclose(full_trace["fidelity_trace"], short_trace["fidelity_trace"], atol=1e-12)
    assert len(short_trace["noise_banks"]) == (1 if crn else 5)
    assert sum(bank["jump_paths"] for bank in short_trace["noise_banks"]) > 0
    for raw, compressed, three in zip(
        full_trace["noise_banks"], short_trace["noise_banks"], old_trace["noise_banks"], strict=True
    ):
        assert raw["raw_sha256"] == compressed["raw_sha256"]
        assert three["first_three_sha256"] == compressed["first_three_sha256"]
        assert compressed["sampled_paths"] == 100
        assert compressed["retained_paths"] == 1 + compressed["jump_paths"]
        assert sum(compressed["multiplicities"]) == 100
        assert sum(compressed["weights"]) == pytest.approx(1)
        assert compressed["no_jump_paths"] == compressed["raw_jump_counts"].count(0)


def test_no_jump_bank_is_inspected_without_rejection_sampling() -> None:
    """An empty sampled bank is reported or stops explicitly; empty circuits remain valid."""
    circuit, theta, target = problem()
    engine = StatevectorBackend(circuit, target)
    seed = next(
        counter
        for counter in range(100)
        if not any(
            noise.operators
            for row in engine.sample_noise_maps(
                theta, KrotovTJMOptions(num_trajectories=1), LegacyNoise(), iteration=counter
            )
            for noise in row
        )
    )
    run = partial(
        train,
        circuit,
        theta,
        target,
        steps=1,
        rate=0.1,
        seed=seed,
        noisy=True,
        num_trajectories=1,
        compress_no_jump=True,
    )
    _, trace = run()
    assert trace["noise_banks"][0]["jump_paths"] == 0
    with pytest.raises(RuntimeError, match="stopped without rejecting/resampling"):
        run(require_jump_coverage=True)
    empty = ParameterizedCircuit(2, [])
    result, trace = train(
        empty,
        np.array([]),
        target,
        steps=2,
        online=1,
        rate=0.1,
        seed=seed,
        noisy=True,
        num_trajectories=100,
        compress_no_jump=True,
        require_jump_coverage=True,
    )
    assert result.size == 0
    assert trace["noise_banks"][0]["multiplicities"] == [100]
    assert trace["noise_banks"][0]["weights"] == [1.0]


def test_paired_rerun_exact_scores_and_resume(tmp_path: Path) -> None:
    """Saved circuits and original three-path parameters survive the complete rerun."""
    circuit, theta, target = problem()
    engine = StatevectorBackend(circuit, target)
    seed = next(
        counter
        for counter in range(100)
        if any(
            noise.operators
            for row in engine.sample_noise_maps(
                theta, KrotovTJMOptions(num_trajectories=100), LegacyNoise(), iteration=counter
            )
            for noise in row
        )
    )
    source = tmp_path / "source"
    recorder = Recorder(source, target)
    recorder.save("before", circuit, theta, {})
    old_theta, old_trace = train(circuit, theta, target, steps=3, online=1, rate=0.1, seed=seed, noisy=True)
    recorder.save("final", circuit, old_theta, old_trace)
    output = tmp_path / "rerun"
    actual = run_case(source / "before.json", source / "final.json", output)
    assert actual["first_three_histories_preserved"]
    assert actual["jump_paths"][0] > 0
    assert 0 < actual["fidelity_hundred"] < 1
    saved = json.loads((output / "source__final" / "weighted100.json").read_text())
    assert saved["circuit"] == json.loads((source / "before.json").read_text())["circuit"]
    assert load_circuit(saved).num_params == circuit.num_params
    assert saved["vector_matrix_fidelity_error"] < 1e-12
    assert run_case(source / "before.json", source / "final.json", output) == actual
    Recorder(source, target, resume=True)
    with pytest.raises(AssertionError):
        Recorder(source, np.roll(target, 1), resume=True)
    # Re-enter a partially complete case and verify the recorded baseline is reusable.
    (output / "source__final" / "comparison.json").unlink()
    resumed = run_case(source / "before.json", source / "final.json", output)
    assert resumed["fidelity_hundred"] == pytest.approx(actual["fidelity_hundred"], abs=1e-13)
    diagnostic = compression_diagnostic(circuit, theta, target, seed)
    assert diagnostic["sampled"] == 100
    assert diagnostic["retained"] < 100
    assert max(diagnostic["errors"].values()) < 1e-12
