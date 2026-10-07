# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent amplitude, gradient, noise, and failure checks for the TCI bridge."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from qiskit import QuantumCircuit

pytest.importorskip("torch")

from benchmarks.state_preparation.constants import TARGET_IDS
from benchmarks.state_preparation.targets import load_target
from experiments.state_preparation_competitors import iqsp_tci as bridge
from experiments.state_preparation_competitors.iqsp import GAUSSIAN_TARGET_ID, gaussian_state
from experiments.state_preparation_competitors.iqsp_comb_backend import CombOptions, CombTensorNetworkObjective
from experiments.state_preparation_competitors.iqsp_tn import iqsp_tn_ansatz, iqsp_tn_candidates
from experiments.state_preparation_competitors.logical_evaluation import score_exact
from experiments.state_preparation_competitors.q12_readiness import (
    StatevectorAuditObjective,
    audit_gaussian_initialization,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture(scope="module")
def require_tci() -> None:
    """Integration checks require the separately installed, pinned Julia project."""
    executable = shutil.which("julia")
    if executable is None:
        pytest.skip("Optional Julia TCI environment is not installed")
    result = subprocess.run(  # noqa: S603 -- fixed local dependency probe, no shell or external data
        [
            executable,
            "--startup-file=no",
            f"--project={bridge.TCI_PROJECT}",
            "-e",
            "using TensorCrossInterpolation, JSON3",
        ],
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode:
        pytest.skip("Instantiate the optional tci_julia environment to run TCI integration checks")


@pytest.mark.usefixtures("require_tci")
@pytest.mark.parametrize("qubits", [6, 12])
def test_every_gaussian_interpolation_target(qubits: int) -> None:
    """All 21 path coordinates retain the exact benchmark state and bit ordering."""
    coordinates = [j / 20 for j in range(21)]
    targets, receipt = bridge.tci_target_bank(
        [gaussian_state(qubits, x) for x in coordinates], gaussian_coordinates=coordinates
    )
    assert len(targets) == 21
    assert all(t.source == bridge.TCI_SOURCE for t in targets)
    assert receipt["oracle"] == "analytic_gaussian"
    assert max(t["vector_error"] for t in receipt["targets"]) < 1e-10
    assert max(max(t["bond_dimensions"]) for t in receipt["targets"]) <= 16
    assert receipt["cpu_seconds"] >= receipt["child_cpu_seconds"] > 0
    assert set(receipt["sources"]) == {"targets.jl", "Project.toml", "Manifest.toml"}
    if qubits == 12:
        assert max(t["unique_function_evaluations"] for t in receipt["targets"]) < 4096
        assert max(t["stored_elements"] for t in receipt["targets"]) < 4096
    json.dumps(receipt, allow_nan=False)


@pytest.mark.usefixtures("require_tci")
def test_complex_sparse_and_constant_targets() -> None:
    """Complex phases and zeros work without replacing TCI by dense SVD."""
    rng = np.random.default_rng(55)
    random = rng.normal(size=64) + 1j * rng.normal(size=64)
    random /= np.linalg.norm(random)
    sparse = np.zeros(64, dtype=complex)
    sparse[37] = 1j
    bell = np.zeros(64, dtype=complex)
    bell[[1, 62]] = [1 / np.sqrt(2), 1j / np.sqrt(2)]
    _, receipt = bridge.tci_target_bank([random, sparse, bell, np.full(64, 1j / 8)])
    assert receipt["oracle"] == "dense_fixture_index_oracle"
    assert max(t["vector_error"] for t in receipt["targets"]) < 1e-12
    assert receipt["targets"][1]["initial_pivot"] == [2, 1, 2, 1, 1, 2]


@pytest.mark.usefixtures("require_tci")
def test_all_nine_six_qubit_fixture_endpoints() -> None:
    """Every existing benchmark endpoint is reproduced, including Haar phases."""
    _, receipt = bridge.tci_target_bank([load_target(6, name).state_vector for name in TARGET_IDS])
    assert len(receipt["targets"]) == 9
    assert max(t["vector_error"] for t in receipt["targets"]) < 1e-10


@pytest.mark.usefixtures("require_tci")
def test_seed_replay_and_independent_fidelity_gradient() -> None:
    """TCI targets agree with a Qiskit statevector oracle at nondegenerate gates."""
    reference = gaussian_state(6, 1)
    first, receipt = bridge.tci_target_bank([reference], gaussian_coordinates=[1.0])
    _, replay = bridge.tci_target_bank([reference], gaussian_coordinates=[1.0])
    assert receipt["targets"][0]["tensor_sha256"] == replay["targets"][0]["tensor_sha256"]
    circuit = iqsp_tn_ansatz(6, 1)
    theta = np.random.default_rng(3).normal(size=circuit.num_parameters) * 0.1
    objective = CombTensorNetworkObjective(circuit, first[0])
    actual_loss, actual_gradient = objective.loss_gradient(theta)
    loss, gradient = StatevectorAuditObjective(circuit).loss_gradient(theta, reference)
    assert actual_loss == pytest.approx(loss, abs=1e-10)
    np.testing.assert_allclose(actual_gradient, gradient, atol=2e-8, rtol=1e-6)
    noisy = QuantumCircuit(6)
    noisy.h(range(6))
    noisy.ry(0.4, 2)
    noisy.rzz(0.7, 2, 3)
    noisy.cx(4, 5)
    noisy_objective = CombTensorNetworkObjective(noisy, first[0], "IBM")
    assert noisy_objective.fidelity([]) == pytest.approx(score_exact(noisy, reference, "IBM")["fidelity"], abs=1e-9)


@pytest.mark.usefixtures("require_tci")
def test_tci_startup_audit_preserves_origin_caveat() -> None:
    """The new target source receives the same audit without substituting gradients."""
    receipt = audit_gaussian_initialization(num_qubits=6, steps=20, tci_options=bridge.TCIOptions())
    assert receipt["passed"]
    assert receipt["records"][0]["relative_gradient_error"] > 0.9
    assert receipt["tensor_diagnostics"]["target_source"] == bridge.TCI_SOURCE
    assert not receipt["gradient_fallback"]
    with pytest.raises(ValueError, match="target_tolerance"):
        audit_gaussian_initialization(options=CombOptions(target_tolerance=1e-28), tci_options=bridge.TCIOptions())


@pytest.mark.usefixtures("require_tci")
def test_twelve_qubit_candidate_uses_tci_in_every_phase() -> None:
    """Default compressed training uses TCI and retains the original update policy."""
    candidate = iqsp_tn_candidates(
        gaussian_state(12, 1),
        target_id=GAUSSIAN_TARGET_ID,
        depths=(1,),
        interpolation_steps=2,
        construction_steps=1,
        noisy_steps=1,
        comb_options=CombOptions(checkpoint_interval=8),
    )[0]
    metadata = candidate.metadata
    assert metadata["target_construction"] == "tci"
    assert metadata["hardware"] == "IBM"
    for stage in metadata["stages"]:
        assert stage["tensor_network"]["target_source"] == bridge.TCI_SOURCE
    assert metadata["noisy_tensor_network"]["target_source"] == bridge.TCI_SOURCE
    assert metadata["postprocessing"]["final_tensor_network"]["target_source"] == bridge.TCI_SOURCE
    assert metadata["noisy"]["retention"].startswith("final_iterate")
    np.testing.assert_array_equal(
        metadata["stages"][0]["optimization"]["final_theta"],
        metadata["stages"][1]["optimization"]["initial_theta"],
    )
    assert metadata["cpu_seconds"] > metadata["target_construction_receipt"]["cpu_seconds"]
    json.dumps(metadata, allow_nan=False)


@pytest.mark.usefixtures("require_tci")
def test_insufficient_rank_rejected_without_fallback() -> None:
    """An inaccurate target cannot enter the competitor fidelity ranking."""
    with pytest.raises(ValueError, match="no fallback"):
        bridge.tci_target_bank([gaussian_state(6, 1)], gaussian_coordinates=[1], options=bridge.TCIOptions(max_bond=1))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_bond", 0),
        ("max_bond", True),
        ("max_iterations", -1),
        ("seed", -1),
        ("seed", True),
        ("tolerance", np.nan),
        ("tolerance", 1),
        ("max_vector_error", 0),
        ("max_vector_error", 1),
        ("timeout_seconds", np.inf),
        ("timeout_seconds", True),
    ],
)
def test_invalid_options(field: str, value: float) -> None:
    """Settings fail before starting an external process."""
    invalid: dict[str, Any] = {field: value}
    with pytest.raises(ValueError, match="must"):
        bridge.TCIOptions(**invalid)


@pytest.mark.parametrize(
    "action",
    [
        lambda: bridge.tci_target_bank([]),
        lambda: bridge.tci_target_bank([np.ones(4) / 2, np.ones(8) / np.sqrt(8)]),
        lambda: bridge.tci_target_bank([np.ones(4) / 2], gaussian_coordinates=[]),
        lambda: bridge.tci_target_bank([np.ones(4) / 2], gaussian_coordinates=[np.nan]),
        lambda: bridge.tci_target_bank([np.ones(64) / 8], gaussian_coordinates=[1]),
        lambda: iqsp_tn_candidates(np.ones(4) / 2, target_id="test", target_construction="unknown"),
        lambda: iqsp_tn_candidates(np.ones(4) / 2, target_id="test", comb_options=CombOptions(target_tolerance=1e-28)),
    ],
    ids=["empty", "registers", "count", "nan", "wrong-gaussian", "mode", "svd-tolerance"],
)
def test_invalid_bank_inputs(action: Callable[[], object]) -> None:
    """Invalid and inconsistent inputs never change the represented target."""
    with pytest.raises(ValueError, match=r"Require|must|Gaussian|TCI"):
        action()


def test_dependency_timeout_and_failure_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing dependencies and bounded execution fail explicitly."""
    target = np.ones(4) / 2
    monkeypatch.setattr(bridge.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="requires Julia"):
        bridge.tci_target_bank([target])
    monkeypatch.setattr(bridge.shutil, "which", lambda _name: "/fake/julia")

    def timeout(*_args: object, **_kwargs: object) -> None:
        msg = "julia"
        raise subprocess.TimeoutExpired(msg, 1)

    monkeypatch.setattr(bridge.subprocess, "run", timeout)
    with pytest.raises(TimeoutError, match="time budget"):
        bridge.tci_target_bank([target])
    monkeypatch.setattr(bridge.subprocess, "run", lambda *_a, **_kw: subprocess.CompletedProcess([], 1, "", "missing"))
    with pytest.raises(RuntimeError, match="missing"):
        bridge.tci_target_bank([target])
    bad = json.dumps({"schema": 2, "library_version": "0.9.19", "targets": []})
    monkeypatch.setattr(bridge.subprocess, "run", lambda *_a, **_kw: subprocess.CompletedProcess([], 0, bad, ""))
    with pytest.raises(RuntimeError, match="mismatch"):
        bridge.tci_target_bank([target])


def test_optional_unix_accounting_preserves_legacy_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the existing adapter must not require Unix-only TCI accounting."""
    monkeypatch.setattr(bridge, "resource", None)
    target = np.ones(4) / 2
    with pytest.raises(RuntimeError, match="requires Unix"):
        bridge.tci_target_bank([target])
    candidate = iqsp_tn_candidates(
        target, target_id="uniform", depths=(1,), interpolation_steps=1, construction_steps=0, noisy_steps=0
    )[0]
    assert candidate.metadata["target_construction"] == "legacy_exact"
