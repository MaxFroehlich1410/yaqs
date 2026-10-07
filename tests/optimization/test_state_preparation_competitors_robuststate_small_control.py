# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check matched control starts, independent assessment and compact receipts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmarks.generate_state_preparation_targets import json_vector_to_state
from experiments.state_preparation_competitors import robuststate_small_control as study
from experiments.state_preparation_competitors.logical_evaluation import score_exact
from experiments.state_preparation_competitors.run import load_bundle, load_parameter_history

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("target_id", study.TARGETS)
def test_small_control_matched_starts_and_independent_scores(tmp_path: Path, target_id: str) -> None:
    """Exercise real four-qubit sampled updates and sealed round trips."""
    output = tmp_path / target_id
    summary = study.run_case(output, target_id, 300, noiseless_steps=2, noisy_steps=2)
    record, circuits = load_bundle(output / "bundle")
    manifest = json.loads((output / "manifest.json").read_text())
    target = json_vector_to_state(manifest["target"]["state_vector"])
    assert manifest["cnot_blocks"] == 12
    assert manifest["bases"] == 81
    assert manifest["shots_per_basis"] == 1024
    assert record["summary"] == summary
    assert len(circuits) == 3
    histories = []
    for name, arm in record["arms"].items():
        training = arm["training"]
        history = load_parameter_history(output / "bundle", training["parameter_history"])
        histories.append(history)
        assert history.shape == (5, 72)
        assert all("theta" not in row for row in training["records"])
        np.testing.assert_array_equal(history[-1], training["final_theta"])
        assert training["measurement_calls"] == (2 if name == "pauli_shadows" else 0)
        assert summary["arms"][name]["shots"] == (2 * 81 * 1024 if name == "pauli_shadows" else 0)
        independent = score_exact(circuits[arm["circuit_index"]], target, "IBM")["fidelity"]
        assert summary["arms"][name]["final_noisy_fidelity"] == pytest.approx(independent, abs=1e-14)
        assert summary["arms"][name]["change_percentage_points"] == pytest.approx(
            100 * (independent - summary["warm_noisy_fidelity"])
        )
    np.testing.assert_array_equal(histories[0][:3], histories[1][:3])
    assert not np.array_equal(histories[0][3:], histories[1][3:])


def test_control_preserves_existing_outputs_and_rejects_unknown_target(tmp_path: Path) -> None:
    """Never overwrite an existing run or silently choose another target."""
    with pytest.raises(ValueError, match="new output"):
        study.run_case(tmp_path, study.TARGETS[0], 300)
    with pytest.raises(ValueError, match="supported"):
        study.run_case(tmp_path / "new", "unknown", 300)
    assert not (tmp_path / "new").exists()


def test_cli_keeps_full_published_schedule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The public CLI cannot silently shorten either training phase."""
    calls = []
    monkeypatch.setattr(study, "run_case", lambda *args, **kwargs: calls.append((args, kwargs)))
    study.main(["--output", str(tmp_path / "run"), "--target-id", study.TARGETS[0], "--seed", "400"])
    assert calls == [((tmp_path / "run", study.TARGETS[0], 400), {})]
