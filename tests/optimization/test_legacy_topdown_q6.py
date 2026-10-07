# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Frozen workflow parity, full fixture selection, and paired q6 resume checks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from benchmarks.state_preparation.targets import load_target_collection
from experiments import legacy_topdown_dense as legacy
from experiments import legacy_topdown_q6 as study

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def shorten_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply the same two-update engineering budget to both compared workflows."""
    original = cast("Callable[..., tuple]", legacy.train)

    def fast(*args: object, steps: int, **kwargs: object) -> tuple:
        return original(*args, steps=min(steps, 2), **kwargs)

    monkeypatch.setattr(legacy, "train", fast)
    monkeypatch.setattr(study, "train", fast)


def test_all_nine_q6_fixture_states_and_frozen_budget() -> None:
    """Select exact fixture amplitudes, including the five previously absent targets."""
    collection = load_target_collection()
    targets = collection.select(num_qubits=(6,))
    assert [target.target_id for target in targets] == [
        "gaussian_mu0p5_sigma0p1",
        "tfim_ferro",
        "tfim_critical",
        "tfim_para",
        "haar_random_1",
        "haar_random_2",
        "haar_random_3",
        "random_mps_bond2",
        "random_mps_bond3",
    ]
    for target in targets:
        assert target.num_qubits == 6
        assert target.state_vector.shape == (64,)
        assert study.brickwall_for(target.state_vector, 2).num_params == 108
        assert study.brickwall_for(target.state_vector, 4).num_params == 198
    assert study.SEEDS == (300, 400)
    assert study.PROTOCOL["literal_threshold"] == pytest.approx(0.9)
    assert study.PROTOCOL["cutoff"] == pytest.approx(0.01)
    assert study.PROTOCOL["pruning_gate_floor"] == 150


def test_archived_construction_matches_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare exact parameters to the old workflow, including rank-pruning repair."""
    shorten_training(monkeypatch)
    target = np.eye(16, dtype=complex)[0]
    monkeypatch.setattr(legacy, "target_state", lambda _seed: target)
    monkeypatch.setattr(legacy, "brickwall", lambda depth: study.brickwall_for(target, depth))
    legacy.archived_example(tmp_path / "old", 300)
    store = study.Checkpoints(tmp_path / "new", target)
    study.construct_archived(store, 300)
    for new, old in [
        ("fixed_vqa", "fixed_vqa"),
        ("layerwise_initial", "layerwise_before_noise"),
        ("topdown_before_pruning", "topdown_before_pruning"),
        ("topdown_initial", "topdown_before_noise"),
    ]:
        _, theta, record = store.load(new)
        original = tmp_path / "old" / "archived_tfim_300" / old
        np.testing.assert_array_equal(theta, np.load(original.with_suffix(".theta.npy")))
        assert record["circuit"] == json.loads(original.with_suffix(".json").read_text())["circuit"]
    assert store.load("topdown_initial")[0].num_params == 150
    study.construct_archived(store, 300)


def test_literal_construction_and_empty_cutoff_match_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve layer-one growth, stopping and deletion of the full empty cutoff."""
    shorten_training(monkeypatch)
    target = np.eye(4, dtype=complex)[0]
    monkeypatch.setattr(legacy, "target_state", lambda _seed, _family: target)
    monkeypatch.setattr(legacy, "brickwall", lambda depth: study.brickwall_for(target, depth))
    legacy.literal_example(tmp_path / "old", "tfim", 300)
    store = study.Checkpoints(tmp_path / "new", target)
    assert study.construct_literal(store, 300)
    for new, old in [("unpruned_initial", "before_pruning"), ("cutoff_initial", "cutoff_before_noise")]:
        _, theta, record = store.load(new)
        original = tmp_path / "old" / "literal_tfim_300" / old
        np.testing.assert_array_equal(theta, np.load(original.with_suffix(".theta.npy")))
        assert record["circuit"] == json.loads(original.with_suffix(".json").read_text())["circuit"]
    assert not store.load("cutoff_initial")[0].gates
    assert study.construct_literal(store, 300)
    result = study.pair(store, "cutoff", "IBM", 300)
    assert result["status"] == "complete"
    assert result["retained_paths"] == [1]
    assert result["jump_paths"] == [0]
    assert result["fidelity_100"] == pytest.approx(result["fidelity_3"], abs=1e-14)


def test_actual_q6_pairs_and_authenticated_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real q6 sampling and matrix scoring exercise both frozen and refreshed banks."""
    shorten_training(monkeypatch)
    monkeypatch.setattr(study, "construct_literal", lambda _store, _seed: False)
    job = (str(tmp_path), "gaussian_mu0p5_sigma0p1", 300)
    result = study.run_target(job)
    assert len(result["results"]) == 6
    for row in result["results"][:2]:
        assert row["status"] == "complete"
        assert row["nested_histories_verified"]
        assert min(row["jump_paths"]) > 0
        assert 0 <= row["fidelity_100"] <= 1
    assert all(row["status"] == "construction_failed" for row in result["results"][2:])
    assert study.run_target(job) == result
    root = tmp_path / "gaussian_mu0p5_sigma0p1__seed300"
    (root / "topdown_initial.theta.npy").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="artifacts changed"):
        study.run_target(job)
    with pytest.raises(ValueError, match="checksum mismatch"):
        study.Checkpoints(root, load_target_collection().load(6, job[1]).state_vector_copy(), resume=True).load(
            "topdown_initial"
        )


def test_coverage_failure_is_recorded_without_forcing_jumps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed coverage trial remains a failure and does not draw another bank."""
    target = np.eye(4, dtype=complex)[0]
    store = study.Checkpoints(tmp_path / "failure", target)
    circuit = study.brickwall_for(target, 1)
    theta = np.zeros(circuit.num_params)
    store.save("topdown_initial", circuit, theta, {})
    store.save("topdown_IBM_3", circuit, theta, {})
    calls = []

    def fail(*_args: object, **kwargs: object) -> tuple:
        calls.append(kwargs)
        msg = "The sampled bank has no jumps; stopped without rejecting/resampling it or changing its weights."
        raise RuntimeError(msg)

    monkeypatch.setattr(study, "train", fail)
    result = study.pair(store, "topdown", "IBM", 300)
    assert result["status"] == "no_jump_bank"
    assert result["fidelity_100"] is None
    assert len(calls) == 1
    assert calls[0]["num_trajectories"] == 100
    assert not store.exists("topdown_IBM_100")


def test_missed_threshold_skips_literal_pruning(tmp_path: Path) -> None:
    """A depth-cap failure cannot masquerade as a successfully pruned method."""
    target = np.eye(4, dtype=complex)[3]
    store = study.Checkpoints(tmp_path / "threshold", target)
    circuit = study.brickwall_for(target, 1)
    store.save("unpruned_initial", circuit, np.zeros(circuit.num_params), [{"fidelity_trace": [0.1]}])
    assert not study.construct_literal(store, 300)
    assert not store.exists("cutoff_initial")
