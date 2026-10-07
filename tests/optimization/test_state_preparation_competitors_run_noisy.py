# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Scope, provenance, storage, cost accounting and dispatch of noisy-paper runs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit import QuantumCircuit

from benchmarks.state_preparation.targets import load_target_collection
from experiments.state_preparation_competitors import run_noisy as study
from experiments.state_preparation_competitors.common import Candidate

if TYPE_CHECKING:
    import argparse
    from pathlib import Path


def _args(output: Path) -> argparse.Namespace:
    """Return a tiny explicit run that still exercises every required phase."""
    return study.make_parser().parse_args([
        "--output",
        str(output),
        "--methods",
        *study.METHODS,
        "--targets",
        "gaussian_mu0p5_sigma0p1",
        "haar_random_1",
        "--seeds",
        "3",
        "4",
        "--depths",
        "1",
        "--nacl-steps",
        "0",
        "1",
        "--construction-steps",
        "1",
        "--noisy-steps",
        "1",
        "--validation-paths",
        "8",
    ])


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("targets", []),
        ("methods", ["nacl", "nacl"]),
        ("hardware", ["bad"]),
        ("seeds", [-1]),
        ("seeds", [True]),
        ("depths", [0]),
        ("depths", [2, 1]),
        ("nacl_steps", [1, 0]),
        ("nacl_steps", [1.5]),
        ("construction_steps", -1),
        ("noisy_steps", True),
        ("nacl_evaluations", 0),
        ("nacl_restarts", 0),
        ("nacl_max_gates", 0),
        ("nacl_max_gates", 4),
        ("seeds", [2**32]),
        ("shots_per_basis", 0),
        ("iqsp_interpolation_steps", 0),
        ("validation_paths", 1),
        ("shadow_bases", 0),
        ("shadow_bases", 730),
        ("robust_feedback", "wrong"),
    ],
)
def test_config_rejects_invalid_requests(tmp_path: Path, key: str, value: object) -> None:
    """Invalid jobs fail before writing anything or starting a numerical solver."""
    args = _args(tmp_path)
    setattr(args, key, value)
    with pytest.raises(ValueError, match=r"must|invalid|Unknown|Unsupported|requires"):
        study.run(args)
    assert not list(tmp_path.iterdir())


def test_real_and_gaussian_subset_eligibility() -> None:
    """Subset methods never silently replace a complex target or invent a homotopy."""
    collection = load_target_collection()
    supported = {method: [] for method in study.METHODS}
    for identifier in study.TARGET_IDS:
        target = collection.load(6, identifier).state_vector_copy()
        for method in study.METHODS:
            if study.eligibility(method, target) is None:
                supported[method].append(identifier)
    assert len(supported["nacl"]) == len(supported["robuststate"]) == 9
    assert len(supported["fontana"]) == 6
    assert supported["iqsp"] == ["gaussian_mu0p5_sigma0p1"]
    with pytest.raises(ValueError, match="Unknown"):
        study.eligibility("bad", np.array([1, 0]))


@pytest.mark.parametrize("method", study.METHODS)
def test_dispatch_passes_separate_published_policies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """One control cannot accidentally receive another method's optimizer or budgets."""
    args = _args(tmp_path)
    target = load_target_collection().load(6, "gaussian_mu0p5_sigma0p1").state_vector_copy()
    calls = []
    expected = [Candidate(QuantumCircuit(6), {})]

    def factory(value: np.ndarray, **kwargs: object) -> list[Candidate]:
        np.testing.assert_array_equal(value, target)
        calls.append(kwargs)
        return expected

    for module, name in (
        (study.nacl, "nacl_candidates"),
        (study.robuststate, "robuststate_candidates"),
        (study.fontana, "fontana_candidates"),
        (study.iqsp, "iqsp_candidates"),
    ):
        monkeypatch.setattr(module, name, factory)
    assert study.construct(args, method, target, 123, "Google") is expected
    assert len(calls) == 1
    receipt = calls[0]
    assert receipt["seed"] == 123
    assert receipt["hardware"] == "Google"
    if method == "nacl":
        assert receipt["search_steps"] == (0, 1)
        assert receipt["inner_evaluations"] == 100
        assert "noisy_steps" not in receipt
    else:
        assert receipt["depths"] == (1,)
        assert receipt["noisy_steps"] == 1
        assert receipt["noiseless_steps" if method == "robuststate" else "construction_steps"] == 1
    if method == "iqsp":
        assert receipt["interpolation_steps"] == 20
    if method == "robuststate":
        assert receipt["feedback"] == "exact_density"
        assert receipt["shots_per_basis"] == 1024


def test_dispatch_defaults_and_rejects_ineligible_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default warm and noisy budgets remain method-specific and unsupported input fails."""
    args = _args(tmp_path)
    args.construction_steps = args.noisy_steps = None
    target = load_target_collection().load(6, "gaussian_mu0p5_sigma0p1").state_vector_copy()
    calls = []
    monkeypatch.setattr(study.robuststate, "robuststate_candidates", lambda _target, **kw: calls.append(kw))
    study.construct(args, "robuststate", target, 3, "IBM")
    assert calls[0]["noiseless_steps"] == 500
    assert calls[0]["noisy_steps"] == 50
    with pytest.raises(ValueError, match="real target"):
        study.construct(args, "fontana", np.array([1, 1j]) / np.sqrt(2), 3, "IBM")


def test_manifest_pins_selected_dependencies_and_parameters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Source hashes include the algorithms and fixed fixtures, and pins are enforced."""
    args = _args(tmp_path)
    versions = {**study.PINNED, "PyNomadBBO": "4.6.0"}
    monkeypatch.setattr(study.importlib.metadata, "version", versions.__getitem__)
    result = study.build_manifest(args)
    assert result["packages"] == versions
    assert "benchmarks/state_preparation_target_states.json" in result["sources"]
    assert "experiments/state_preparation_competitors/noisy_core.py" in result["sources"]
    assert result["config"]["robust_feedback"] == "exact_density"
    args.methods = ["robuststate"]
    assert "PyNomadBBO" not in study.build_manifest(args)["packages"]
    versions["qiskit"] = "wrong"
    with pytest.raises(ValueError, match="qiskit"):
        study.build_manifest(args)


def test_campaign_records_exclusions_and_resumes_without_retraining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use real storage/evaluation to verify all seeds, hardware, exclusions and replay."""
    args = _args(tmp_path)
    monkeypatch.setattr(study, "build_manifest", lambda _args: {"schema": study.SCHEMA})
    calls = []

    def factory(
        _args: argparse.Namespace, method: str, target: np.ndarray, seed: int, hardware: str
    ) -> list[Candidate]:
        calls.append((method, seed, hardware))
        return [
            Candidate(
                QuantumCircuit(target.size.bit_length() - 1),
                {"method": method, "candidate_id": "zero", "status": "complete", "preserve_native_schedule": True},
            )
        ]

    monkeypatch.setattr(study, "construct", factory)
    result = study.run(args)
    assert len(result["groups"]) == len(calls) == 22
    assert len(result["exclusions"]) == 2
    assert {x["method"] for x in result["exclusions"]} == {"fontana", "iqsp"}
    assert all(x["target_id"] == "haar_random_1" for x in result["exclusions"])
    assert {(seed, hardware) for method, seed, hardware in calls if method == "iqsp"} == {(0, "IBM"), (0, "Google")}
    for group in result["groups"]:
        assert group["status"] == "complete"
        assert group["construction"]["construction_cpu_seconds"] >= 0
        assert group["selection_lock_sha256"]
        assert group["records"][0]["compilation"]["compiled_two_qubit"] == 0
    assert json.loads((tmp_path / "summary.json").read_text()) == result
    resumed = study.run(args)
    assert resumed == result
    assert len(calls) == 22
    monkeypatch.setattr(study, "build_manifest", lambda _args: {"schema": "changed"})
    with pytest.raises(ValueError, match="manifest changed"):
        study.run(args)


def test_nonempty_directory_and_main_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI neither overwrites foreign output nor bypasses argument validation."""
    args = _args(tmp_path)
    (tmp_path / "foreign.txt").write_text("preserve")
    monkeypatch.setattr(study, "build_manifest", lambda _args: {})
    with pytest.raises(ValueError, match="nonempty"):
        study.run(args)
    calls = []
    monkeypatch.setattr(study, "run", calls.append)
    study.main(["--output", str(tmp_path), "--methods", "robuststate"])
    assert calls[0].methods == ["robuststate"]


def test_actual_six_qubit_training_storage_and_final_score(tmp_path: Path) -> None:
    """Run real complex-state training through the CLI's complete persistence path."""
    args = _args(tmp_path)
    args.methods = ["robuststate"]
    args.targets = ["haar_random_1"]
    args.hardware = ["IBM"]
    args.seeds = [3]
    args.validation_paths = 512
    result = study.run(args)
    assert len(result["groups"]) == 1
    group = result["groups"][0]
    assert group["status"] == "complete"
    assert group["records"][0]["validation"]["jump_paths"] > 0
    metadata = study.storage.load_candidate_metadata(tmp_path, group["records"][0]["method"])
    assert metadata["training"]["gradient_evaluations"] == 2
    assert "parameter_history" in metadata["training"]
    assert group["final_scores"][0]["exact"]["fidelity"] == pytest.approx(metadata["noisy_fidelity"], abs=1e-12)
    assert group["records"][0]["compilation"]["compilation_cpu_seconds"] >= 0
    assert study.run(args) == result
