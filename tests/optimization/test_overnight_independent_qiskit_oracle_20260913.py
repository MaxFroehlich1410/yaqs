# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent calibration, physicality and q1-q3 oracle comparisons."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from qiskit.quantum_info import DensityMatrix

from benchmarks.state_preparation import ballarin, noise
from experiments.overnight_state_preparation_20260913_0108 import channels
from experiments.overnight_state_preparation_20260913_0108 import independent_qiskit_oracle as oracle
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from numpy.typing import NDArray


def problem(n: int) -> tuple[dict[str, Any], NDArray[np.float64], NDArray[np.complex128]]:
    """Return a synthetic problem with reversed sites and fixed/shared affine angles."""
    gates: list[dict[str, Any]] = [
        {"name": "ry", "sites": [0], "param_index": 0, "angle_scale": -1.3, "angle_offset": 0.21},
        {"name": "rz", "sites": [n - 1], "param_index": 1, "noise_enabled": False},
        {"name": "rx", "sites": [0], "param_index": None, "angle_offset": -0.48},
    ]
    if n > 1:
        gates.extend(
            {"name": name, "sites": [n - 1, 0], "param_index": index % 2, "angle_offset": 0.31 * index}
            for index, name in enumerate(("rxx", "ryy", "rzz"))
        )
    for i, gate in enumerate(gates):
        gate.update(logical_gate_id=f"origin-{i}", native_gate_id=i, fixed_params=[], data_map=None)
    rng = np.random.default_rng(129 + n)
    target = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return {"num_qubits": n, "num_params": 2, "gates": gates}, np.array([0.413, -7.12]), target / np.linalg.norm(target)


def reference(spec: dict[str, Any], theta: NDArray, target: NDArray, profile: str) -> float:
    """Return the existing evaluator's score only in a comparison test."""
    gates = []
    for gate in spec["gates"]:
        fields: dict[str, Any] = {**gate, "sites": tuple(gate["sites"])}
        gates.append(ParameterizedGate(**fields))
    return channels.evaluate(
        ParameterizedCircuit(spec["num_qubits"], gates, spec["num_params"]), theta, target, profile
    )


def test_no_shared_numerical_dependencies_even_when_the_selected_oracle_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail every selected numerical helper, matrix builder and canonical profile mapper."""
    spec, theta, target = problem(3)
    expected = {name: reference(spec, theta, target, name) for name in channels.NOISE_IDS}

    def prohibited(*_args: object, **_kwargs: object) -> None:
        pytest.fail("The independent oracle called a selected numerical implementation.")

    for name in ("evaluate", "window", "superoperator", "apply_channel", "local_paulis"):
        monkeypatch.setattr(channels, name, prohibited)
    for name in ("gate_matrix", "gate_matrix_and_angle", "angle"):
        monkeypatch.setattr(ParameterizedCircuit, name, prohibited)
    monkeypatch.setattr(ballarin, "materialize_ballarin_circuit", prohibited)
    monkeypatch.setattr(ballarin, "ballarin_pauli_probability", prohibited)
    monkeypatch.setattr(noise, "standard_noise_profile", prohibited)
    for name in oracle.PROFILE_IDS:
        actual = oracle.evaluate_profile(spec, theta, target, name)
        assert actual["fidelity"] == pytest.approx(expected[name], abs=3e-13)
    imported = [
        node.module
        for node in ast.walk(ast.parse(inspect.getsource(oracle)))
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    assert not any(name.startswith(("mqt", "benchmarks", "experiments")) for name in imported)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("profile", oracle.PROFILE_IDS)
def test_all_profiles_match_existing_evaluator_without_mutating_inputs(n: int, profile: str) -> None:
    """Independent gate classes, channel conventions and sites reproduce synthetic scores."""
    spec, theta, target = problem(n)
    original = copy.deepcopy(spec), theta.copy(), target.copy()
    actual = oracle.evaluate_profile(spec, theta, target, profile)
    assert actual["fidelity"] == pytest.approx(reference(spec, theta, target, profile), abs=3e-13)
    assert actual["physicality"]["trace_error"] < 1e-13
    assert actual["physicality"]["hermitian_max_abs"] < 1e-13
    assert actual["physicality"]["minimum_eigenvalue"] >= -1e-13
    assert actual["physicality"]["norm_error"] < 1e-13
    assert actual["cpu_seconds"] >= 0
    assert actual["wall_seconds"] >= 0
    assert spec == original[0]
    np.testing.assert_array_equal(theta, original[1])
    np.testing.assert_array_equal(target, original[2])
    json.dumps(actual, allow_nan=False)


@pytest.mark.parametrize("arity", [1, 2])
@pytest.mark.parametrize("profile", oracle.STANDARD_PROFILES)
def test_complete_channel_calibration_and_literal_operator_sets(profile: str, arity: int) -> None:
    """Verify independent average-error inversion, including inactive identity channels."""
    row = oracle.calibration_profile(profile, arity)
    expected = channels.window(profile, arity)
    assert set(row["generators"]) == set(expected["generators"])
    assert row["actual_error"] == pytest.approx(expected["target_error"], abs=2e-13)
    assert row["coefficient"] == pytest.approx(expected["coefficient"], abs=2e-14)
    assert row["minimum_choi_eigenvalue"] >= -1e-12
    row["generators"].append("mutated")
    assert "mutated" not in oracle.calibration_profile(profile, arity)["generators"]


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("profile", oracle.PROFILE_IDS)
def test_empty_circuit_always_starts_from_physical_zero(n: int, profile: str) -> None:
    """Targets do not become free input states, and empty circuits have no noise windows."""
    target = np.zeros(2**n, dtype=complex)
    target[-1] = 1
    row = oracle.evaluate_profile({"num_qubits": n, "num_params": 0, "gates": []}, [], target, profile)
    assert row["fidelity"] == 0
    assert row["channel_windows"] == 0
    assert row["physical_counts"] == row["evaluated_counts"] == {"one_qubit": 0, "two_qubit": 0, "total": 0}


def test_little_endian_qubit_labels_and_zero_angle_noise_are_physical() -> None:
    """Qubit two addresses amplitude four; an identity rotation still carries its channel."""
    spec = {"num_qubits": 3, "num_params": 1, "gates": [{"name": "ry", "sites": [2], "param_index": 0}]}
    target = np.zeros(8, dtype=complex)
    target[4] = 1
    assert oracle.evaluate_profile(spec, [math.pi], target, "noiseless")["fidelity"] == pytest.approx(1, abs=1e-14)
    target[4], target[0] = 0, 1
    assert oracle.evaluate_profile(spec, [0], target, "depolarizing_1s_1q")["fidelity"] == pytest.approx(1 - 0.00064)
    spec["gates"][0]["noise_enabled"] = False
    assert oracle.evaluate_profile(spec, [0], target, "depolarizing_1s_1q")["fidelity"] == 1


@pytest.mark.parametrize("name", ["rxx", "ryy", "rzz"])
@pytest.mark.parametrize("angle", [0, 1e-4, -1e-4, 1e-4 + 0.5e-12, 1e-4 + 2e-12, -7.2, 8.1])
def test_ballarin_lowering_pruning_flags_and_native_counts(name: str, angle: float) -> None:
    """Independently transcribed lowering reproduces the canonical threshold and basis factors."""
    gate = {"name": name, "sites": [1, 0], "param_index": 0, "noise_enabled": False}
    spec = {"num_qubits": 2, "num_params": 1, "gates": [gate]}
    target = np.array([1, 0, 0, 0], dtype=complex)
    row = oracle.evaluate_profile(spec, [angle], target, "ballarin_coupled")
    assert row["fidelity"] == pytest.approx(reference(spec, np.array([angle]), target, "ballarin_coupled"), abs=2e-13)
    wrapped = (angle + math.pi) % (2 * math.pi) - math.pi
    pruned = abs(wrapped) <= 1e-4 or math.isclose(abs(wrapped), 1e-4, rel_tol=0, abs_tol=1e-12)
    assert row["ballarin_pruned_two_qubit"] == int(pruned)
    assert row["evaluated_counts"]["two_qubit"] == int(not pruned)
    assert row["evaluated_counts"]["one_qubit"] == (4 if not pruned and name != "rzz" else 0)
    assert row["channel_windows"] == 2 * int(not pruned)


def test_ballarin_rzz_probability_formula_is_independent_and_two_site() -> None:
    """On zero, RZZ contributes only phase and loss comes exclusively from both channels."""
    spec = {"num_qubits": 2, "num_params": 0, "gates": [{"name": "rzz", "sites": [0, 1], "angle_offset": -0.61}]}
    row = oracle.evaluate_profile(spec, [], [1, 0, 0, 0], "ballarin_coupled")
    error = 2.1e-4 + 1.43e-3 * 0.61
    probability = (1 - math.sqrt(1 - 1.25 * error)) / 3
    assert row["fidelity"] == pytest.approx((1 - 2 * probability) ** 2, abs=1e-14)


@pytest.mark.parametrize(
    "bad",
    [
        "theta",
        "complex_theta",
        "boolean_theta",
        "target",
        "profile",
        "arity",
        "qubits",
        "data",
        "unitary",
        "index",
        "flag",
        "gate",
        "overflow",
    ],
)
def test_invalid_domain_fails_before_scoring(bad: str) -> None:
    """Malformed scalar metadata and targets must not silently change the physical model."""
    spec, theta, target = problem(2)
    profile = "noiseless"
    if bad == "theta":
        theta[0] = np.nan
    elif bad == "complex_theta":
        theta = np.array([1j, 0])
    elif bad == "boolean_theta":
        theta = np.array([True, False])
    elif bad == "target":
        target *= 2
    elif bad == "profile":
        profile = "legacy_one_jump"
    elif bad == "arity":
        spec["gates"][0]["sites"] = [0, 1]
    elif bad == "qubits":
        spec["num_qubits"] = True
    elif bad == "data":
        spec["gates"][0]["data_map"] = "map"
    elif bad == "unitary":
        spec["gates"][0]["name"] = "unitary"
    elif bad == "index":
        spec["gates"][0]["param_index"] = -1
    elif bad == "flag":
        spec["gates"][0]["noise_enabled"] = 1
    elif bad == "gate":
        spec["gates"][0] = None
    else:
        spec["gates"][0]["angle_scale"] = 1e308
        theta[0] = 1e308
    with pytest.raises((ValueError, TypeError), match=r"Oracle|oracle|angle"):
        oracle.evaluate_profile(spec, theta, target, profile)


@pytest.mark.parametrize("defect", ["trace", "hermitian", "positive", "nonfinite"])
def test_corrupted_density_is_rejected(monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    """A numerical propagation defect cannot become a successful fidelity receipt."""
    matrix = np.diag([1.0, 0.0]).astype(complex)
    if defect == "trace":
        matrix *= 0.8
    elif defect == "hermitian":
        matrix[0, 1] = 0.1j
    elif defect == "positive":
        matrix[0, 0], matrix[1, 1] = 1.1, -0.1
    else:
        matrix[0, 0] = np.nan
    monkeypatch.setattr(DensityMatrix, "evolve", lambda *_args, **_kwargs: DensityMatrix(matrix))
    spec = {"num_qubits": 1, "num_params": 0, "gates": [{"name": "ry", "sites": [0]}]}
    with pytest.raises(FloatingPointError):
        oracle.evaluate_profile(spec, [], [1, 0], "noiseless")


def test_minimum_eigenvalue_uses_hermitian_part_and_preserves_raw_residual(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sub-tolerance asymmetric roundoff must not silently select one matrix triangle."""
    matrix = np.array([[0.5, 2e-11j], [0, 0.5]], dtype=complex)
    monkeypatch.setattr(DensityMatrix, "evolve", lambda *_args, **_kwargs: DensityMatrix(matrix))
    spec = {"num_qubits": 1, "num_params": 0, "gates": [{"name": "ry", "sites": [0]}]}
    row = oracle.evaluate_profile(spec, [], [1, 0], "noiseless")
    assert row["physicality"]["minimum_eigenvalue"] == pytest.approx(0.5 - 1e-11, abs=1e-15)
    assert row["physicality"]["hermitian_max_abs"] == pytest.approx(2e-11, abs=1e-25)
    assert row["minimum_eigenvalue_scope"].startswith("Hermitian part")
