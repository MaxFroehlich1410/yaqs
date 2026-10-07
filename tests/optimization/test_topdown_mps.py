# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Full method parity, streamed assessment, resource checks and safe resume."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, NoReturn

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from experiments import legacy_pruning_ablation as old_pruning
from experiments import topdown_backend_support as support
from experiments import topdown_depth_search as old_depth
from experiments import topdown_mps as runner
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import MPSBackend, ParameterizedCircuit, ParameterizedGate, StatevectorBackend
from tests.optimization.test_statevector import case

if TYPE_CHECKING:
    from pathlib import Path

    from qiskit.circuit import Operation
    from qiskit.quantum_info.operators.base_operator import BaseOperator


def simple() -> tuple:
    """Return an entangled two-qubit preparation with independent coordinates.

    Returns:
        Circuit, angles and normalized complex target.
    """
    circuit = ParameterizedCircuit(
        2, [ParameterizedGate("ry", (0,), param_index=0), ParameterizedGate("rxx", (0, 1), param_index=1)]
    )
    return circuit, np.array([1.1, 0.8]), np.array([1, 1j, 1, 2j], dtype=complex) / np.sqrt(7)


@pytest.mark.parametrize("hardware", ["IBM", "Google"])
def test_streamed_validation_matches_raw_multiplicity_variance(hardware: str) -> None:
    """Streamed MPS scores equal the archived independent dense assessment law."""
    c, theta, target = simple()
    codes = old_depth.sample_validation(c, hardware, 2048, [91, 2, 3])
    assert np.any(codes)
    expected = old_depth.validation_score(c, theta, target, hardware, codes)
    for backend in [MPSBackend(c, target), StatevectorBackend(c, target)]:
        actual = support.score_codes(backend, theta, hardware, codes)
        assert actual["fidelity"] == pytest.approx(expected["fidelity"], abs=1e-12)
        assert actual["standard_error"] == pytest.approx(expected["standard_error"], abs=1e-12)
        assert actual["retained_paths"] == expected["retained_paths"]
        assert actual["jump_paths"] == expected["jump_paths"]


def test_assessment_edge_cases_and_coverage() -> None:
    """Zero jumps are reported without forcing events or replacing the bank."""
    c, theta, target = simple()
    engine = MPSBackend(c, target)
    result = support.score_codes(engine, theta, "IBM", np.zeros((20, 2), dtype=np.uint8))
    assert result["status"] == "no_jump_bank"
    assert result["retained_paths"] == 1
    assert result["standard_error"] == 0
    for codes in [
        np.zeros((1, 2), dtype=int),
        np.zeros((10, 3), dtype=int),
        np.ones((2, 2)) * 0.5,
        np.full((2, 2), 255, dtype=int),
    ]:
        with pytest.raises(ValueError, match="assessment"):
            support.score_codes(engine, theta, "IBM", codes)
    empty = ParameterizedCircuit(2, [])
    assert (
        support.score_codes(MPSBackend(empty, target), np.array([]), "IBM", np.empty((2, 0), dtype=int))["status"]
        == "complete"
    )


def test_pruning_b_matches_frozen_implementation_and_roundtrip() -> None:
    """Signed scores, threshold and simultaneous deletion match archived B exactly."""
    c, theta, target = simple()
    expected, angles, old = old_pruning.prune_variant(c, theta, target, "B")
    for backend in [MPSBackend(c, target), StatevectorBackend(c, target)]:
        pruned, actual, trace = support.prune_b(backend, theta)
        assert support.circuit_record(pruned) == support.circuit_record(expected)
        np.testing.assert_array_equal(actual, angles)
        np.testing.assert_allclose(trace["scores"], old["iterations"][0]["scores"], atol=1e-12)
        assert support.circuit_record(
            support.restore_circuit(support.circuit_record(pruned))
        ) == support.circuit_record(pruned)
    shared, theta, target = case()
    with pytest.raises(ValueError, match="independent"):
        support.prune_b(MPSBackend(shared, target), theta)


def test_pruning_cutoff_and_permuted_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep equality and preserve gate-to-parameter mappings after pruning."""
    c, theta, target = simple()
    c.gates[0].param_index, c.gates[1].param_index = 1, 0
    engine = StatevectorBackend(c, target)
    monkeypatch.setattr(engine, "deletion_fidelities", lambda _: (np.array([-0.01, 0.02]), 0.0))
    pruned, values, trace = support.prune_b(engine, theta)
    assert len(pruned.gates) == 1
    assert pruned.gates[0].name == "ry"
    np.testing.assert_array_equal(values, [theta[1]])
    assert trace["removed_gate_indices"] == [1]


def test_resource_counts_match_old_compilation_without_full_unitaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local replay verifies preparation and preserves compiled two-qubit counts."""
    c, theta, _ = simple()
    _, expected = old_pruning.compile_resources(c, theta)
    original = Operator.__init__

    def local_only(
        self: Operator,
        data: QuantumCircuit | Operation | BaseOperator | np.ndarray,
        input_dims: tuple | None = None,
        output_dims: tuple | None = None,
    ) -> None:
        if isinstance(data, QuantumCircuit):
            pytest.fail("full circuit unitary allocated")
        original(self, data, input_dims, output_dims)

    monkeypatch.setattr(Operator, "__init__", local_only)
    actual = support.compile_resources(c, theta)
    for key in ("compiled_one_qubit", "compiled_two_qubit", "compiled_two_qubit_depth", "logical_two_qubit_depth"):
        assert actual[key] == expected[key]
    twelve = ParameterizedCircuit(
        12, [ParameterizedGate("ry", (1,), param_index=0), ParameterizedGate("rxx", (1, 2), param_index=1)]
    )
    result = support.compile_resources(twelve, theta)
    assert result["phase_aligned_squared_state_distance"] < 1e-12


def test_checkpoint_atomicity_corruption_and_binary_references(tmp_path: Path) -> None:
    """Incomplete stages restart; completed payloads are authenticated on load."""
    assert support.load_checkpoint(tmp_path) is None
    data = {"theta": np.array([0.1, 0.2]), "codes": np.array([[0, 1]], dtype=np.uint8)}
    support.save_checkpoint(tmp_path, {"seed": 42}, data)
    saved = support.load_checkpoint(tmp_path)
    assert saved is not None
    record, arrays = saved
    assert record["arrays"]["theta"]["file"] == "arrays.npz"
    np.testing.assert_array_equal(arrays["codes"], data["codes"])
    (tmp_path / "arrays.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        support.load_checkpoint(tmp_path)


def small_config(backend: str = "mps") -> runner.Config:
    """Use short explicit test budgets while preserving the scientific policies.

    Returns:
        One-depth configuration with sufficient independent jump coverage.
    """
    return runner.Config(
        backend=backend,
        qubits=2,
        seed=301,
        max_depth=1,
        construction_steps=8,
        construction_online=3,
        noisy_steps=4,
        noisy_online=2,
        validation_paths=4096,
        final_paths_per_batch=4096,
        final_batches=2,
    )


def test_full_pipeline_backend_parity_independence_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Construction through final selection agrees across exact representations."""
    _, _, target = simple()
    # Keep hashing deterministic while allowing tests to inject failures below.
    monkeypatch.setattr(runner, "source_hashes", lambda: {"test-source": "stable"})
    dense_root, mps_root = tmp_path / "dense", tmp_path / "mps"
    expected = runner.run_case(dense_root, target, small_config("statevector"), target_id="test")
    actual = runner.run_case(mps_root, target, small_config(), target_id="test")
    assert actual["selected_depth"] == expected["selected_depth"] == 1
    assert actual["fidelity"] == pytest.approx(expected["fidelity"], abs=1e-9)
    assert actual["resources"]["compiled_two_qubit"] == expected["resources"]["compiled_two_qubit"]
    for stage in ("construction", "pruning", "refinement"):
        a, b = (
            support.load_checkpoint(dense_root / "depth_01" / stage),
            support.load_checkpoint(mps_root / "depth_01" / stage),
        )
        assert a is not None
        assert b is not None
        np.testing.assert_allclose(a[1]["theta"], b[1]["theta"], atol=1e-8)
    saved_val = support.load_checkpoint(mps_root / "depth_01/validation")
    saved_test0 = support.load_checkpoint(mps_root / "test_00")
    saved_test1 = support.load_checkpoint(mps_root / "test_01")
    assert saved_val is not None
    assert saved_test0 is not None
    assert saved_test1 is not None
    val, test0, test1 = saved_val[0], saved_test0[0], saved_test1[0]
    assert val["entropy"] != test0["entropy"] != test1["entropy"]
    assert test0["jump_paths"] > 0
    assert test1["jump_paths"] > 0
    monkeypatch.setattr(runner, "_search", lambda *_a: pytest.fail("retrained a complete run"))
    assert runner.run_case(mps_root, target, small_config(), target_id="test") == actual
    with pytest.raises(ValueError, match="manifest"):
        runner.run_case(mps_root, target, replace(small_config(), max_bond_dim=4), target_id="test")
    (mps_root / "selection_locked.json").write_text("{}")
    with pytest.raises(ValueError, match="selection"):
        runner.run_case(mps_root, target, small_config(), target_id="test")


def test_interrupted_stage_resumes_without_repeating_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sealed construction survives a refinement failure and retains identical seeds."""
    _, _, target = simple()
    monkeypatch.setattr(runner, "source_hashes", dict)
    original = runner.legacy.train

    def fail(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "interrupted"
        raise RuntimeError(msg)

    monkeypatch.setattr(runner.legacy, "train", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.run_case(tmp_path, target, small_config(), target_id="test")
    before = support.digest(tmp_path / "depth_01/construction/record.json")
    monkeypatch.setattr(runner.legacy, "train", original)
    monkeypatch.setattr(runner.policies, "train_noiseless", lambda *_a, **_k: pytest.fail("repeated construction"))
    result = runner.run_case(tmp_path, target, small_config(), target_id="test")
    assert result["selected_depth"] == 1
    assert support.digest(tmp_path / "depth_01/construction/record.json") == before


@pytest.mark.parametrize(
    "changes",
    [
        {"backend": "bad"},
        {"qubits": 0},
        {"max_bond_dim": 0},
        {"svd_threshold": -1},
        {"validation_paths": 1},
        {"seed": -1},
        {"noisy_online": 300},
        {"qubits": 12, "exact_density_check": True},
        {"backend": "statevector", "max_bond_dim": 4},
    ],
)
def test_invalid_configuration(changes: dict) -> None:
    """Reject invalid or conflicting settings before allocating simulation states."""
    with pytest.raises(ValueError, match=r"budget|bond|backend|Truncation|histories|oracle"):
        replace(runner.Config(), **changes).validate()


def test_cli_help_and_bad_target_size(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Expose backend and truncation options and reject inconsistent register sizes."""
    with pytest.raises(SystemExit) as caught:
        runner.main(["--help"])
    assert caught.value.code == 0
    assert "--max-bond-dim" in capsys.readouterr().out
    with pytest.raises(ValueError, match="Target"):
        runner.run_case(tmp_path, np.array([1, 0], dtype=complex), small_config(), target_id="bad")


def test_twelve_qubit_pipeline_uses_tensor_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The full orchestration supports q12 without a dense target or full operator."""
    monkeypatch.setattr(runner, "source_hashes", dict)
    monkeypatch.setattr(MPS, "to_vec", lambda *_a: pytest.fail("densified"))
    monkeypatch.setattr(MPS, "from_statevector", lambda *_a: pytest.fail("dense conversion"))
    config = replace(
        small_config(),
        qubits=12,
        construction_steps=0,
        construction_online=0,
        noisy_steps=0,
        noisy_online=0,
        validation_paths=2,
        final_paths_per_batch=2,
    )
    result = runner.run_case(tmp_path, MPS(12), config, target_id="zero")
    assert result["status"] == "complete"
    assert result["fidelity"] == pytest.approx(1)
    assert result["resources"]["logical_two_qubit"] == 0


def test_no_jump_failure_advances_patience_without_resampling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Three failed candidates stop depth growth and cannot become a selected result."""
    _, _, target = simple()
    attempts = []

    def no_jump(*_args: object, **kwargs: object) -> None:
        """Record one attempted bank and report its coverage failure.

        Raises:
            RuntimeError: The bank has no jumps.
        """
        attempts.append(kwargs["seed"])
        msg = "The sampled bank has no jumps"
        raise RuntimeError(msg)

    monkeypatch.setattr(runner, "source_hashes", dict)
    monkeypatch.setattr(runner.legacy, "train", no_jump)
    config = replace(small_config(), max_depth=5, construction_steps=1, construction_online=1)
    result = runner.run_case(tmp_path, target, config, target_id="test")
    assert result["status"] == "no_valid_candidate"
    assert result["stop_reason"] == "patience"
    assert len(result["candidates"]) == len(attempts) == 3
    assert attempts == [config.seed + 5000] * 3


def test_exact_oracle_and_memory_profile_are_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional density cross-checks are reported after independent selection."""
    _, _, target = simple()
    monkeypatch.setattr(runner, "source_hashes", dict)
    config = replace(small_config(), exact_density_check=True, profile_memory=True, max_bond_dim=2)
    result = runner.run_case(tmp_path, target, config, target_id="test")
    assert abs(result["fidelity"] - result["exact_density_oracle"]["fidelity"]) < 0.02
    assert result["invocation_resources"]["traced_peak_bytes"] > 0
    assert result["invocation_resources"]["worker_lifetime_peak_rss_bytes"] > 0


def test_cli_real_fixture_writes_authenticated_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Actual fixture loading, qubit selection and command-line wiring work together."""
    monkeypatch.setattr(runner, "source_hashes", dict)
    target_id = "gaussian_mu0p5_sigma0p1"
    runner.main([
        "--output",
        str(tmp_path),
        "--targets",
        target_id,
        "--qubits",
        "6",
        "--max-depth",
        "1",
        "--construction-steps",
        "0",
        "--construction-online",
        "0",
        "--noisy-steps",
        "0",
        "--noisy-online",
        "0",
        "--validation-paths",
        "2",
        "--final-paths-per-batch",
        "2",
        "--final-batches",
        "1",
    ])
    saved = support.load_checkpoint(tmp_path / target_id / "final")
    assert saved is not None
    assert saved[0]["status"] in {"complete", "no_valid_candidate", "test_jump_coverage_failure"}
    assert (tmp_path / target_id / "manifest.json").exists()


def test_binary_checkpoints_are_deterministic_numeric_and_portable(tmp_path: Path) -> None:
    """Round-trip noise bytes and complex parameters without pickle or timestamps."""
    arrays = {
        "codes": np.array([[0, 15], [7, 0]], dtype=np.uint8),
        "parameters": np.array([1 + 2j, 0.0]),
        "scalar": np.array(0.25),
        "empty": np.empty(0),
    }
    support.save_checkpoint(tmp_path / "a", {"status": "complete"}, arrays)
    support.save_checkpoint(tmp_path / "b", {"status": "complete"}, dict(reversed(list(arrays.items()))))
    assert support.digest(tmp_path / "a/arrays.npz") == support.digest(tmp_path / "b/arrays.npz")
    restored = support.load_checkpoint(tmp_path / "a")
    assert restored is not None
    assert set(restored[1]) == set(arrays)
    for name, expected in arrays.items():
        np.testing.assert_array_equal(restored[1][name], expected)
        assert restored[1][name].dtype == expected.dtype
    for unsafe in [
        {"../escape": np.zeros(1)},
        {"objects": np.array([object()])},
        {"invalid": np.array([np.nan])},
    ]:
        with pytest.raises(ValueError, match="finite numeric"):
            support.save_checkpoint(tmp_path / "bad", {}, unsafe)
    assert not (tmp_path / "bad").exists()
