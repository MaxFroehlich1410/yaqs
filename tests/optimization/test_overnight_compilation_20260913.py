# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent full-unitary, physical-noise and dependency compiler checks."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import channels, compilation
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from numpy.typing import NDArray


def qiskit_unitary(circuit: ParameterizedCircuit, theta: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Return the independent full operator of all original physical rotations."""
    reference = QuantumCircuit(circuit.num_qubits)
    for gate in circuit.gates:
        getattr(reference, gate.name)(circuit.angle(gate, theta, None), *gate.sites)
    return Operator(reference).data


def assert_unitary_equivalent(
    circuit: ParameterizedCircuit, theta: NDArray[np.float64], result: dict[str, Any]
) -> None:
    """Check every input basis vector, including the recorded omitted global sign."""
    compiled = pilot.restore_circuit(result["circuit"])
    original = qiskit_unitary(circuit, theta)
    output = np.exp(1j * result["omitted_global_phase"]) * qiskit_unitary(compiled, np.asarray(result["theta"]))
    np.testing.assert_allclose(output, original, atol=max(2e-12, result["ideal_operator_error_bound"]), rtol=0)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("seed", [11, 29])
def test_random_shared_scaled_endpoint_preserves_full_unitary_and_inputs(n: int, seed: int) -> None:
    """Random affine/shared and fixed endpoint rotations retain their full operator."""
    rng = np.random.default_rng(seed)
    theta = np.array([0.0, 2 * np.pi, -np.pi, 0.173, -0.419])
    gates = []
    for index in range(60):
        arity = int(rng.integers(1, min(n, 2) + 1))
        sites = tuple(int(q) for q in rng.choice(n, arity, replace=False))
        axis = str(rng.choice(["x", "y", "z"]))
        gates.append(
            ParameterizedGate(
                "r" + axis * arity,
                sites,
                None if index % 7 == 0 else int(rng.integers(5)),
                angle_scale=float(rng.choice([-2, -1, 0.5, 1, 2])),
                angle_offset=float(rng.choice([0, 0.271, np.pi, -2 * np.pi])),
                logical_gate_id=index,
                native_gate_id=f"native_{index}",
                noise_enabled=index % 11 != 0,
            )
        )
    circuit = ParameterizedCircuit(n, gates, num_params=5)
    before_circuit, before_theta = copy.deepcopy(pilot.circuit_record(circuit)), theta.copy()
    result = compilation.compile_endpoint(circuit, theta)
    assert_unitary_equivalent(circuit, theta, result)
    np.testing.assert_array_equal(theta, before_theta)
    assert pilot.circuit_record(circuit) == before_circuit
    compiled = pilot.restore_circuit(result["circuit"])
    assert [g.param_index for g in compiled.gates] == list(range(compiled.num_params))
    assert all(g.angle_scale == 1 and g.angle_offset == 0 for g in compiled.gates)
    represented = [i for row in result["output_mapping"] for i in row["source_gate_indices"]]
    removed = [i for event in result["events"] if event["kind"] == "remove_identity" for i in event["sources"]]
    assert sorted([*represented, *removed]) == list(range(len(circuit.gates)))
    assert len(result["source_to_output"]) == len(circuit.gates)
    assert result["removed_noise_windows"] == result["input_noise_windows"] - result["output_noise_windows"]
    second = compilation.compile_endpoint(circuit, theta)
    for key in result.keys() - {"compilation_cpu_seconds", "compilation_wall_seconds"}:
        assert result[key] == second[key]


@pytest.mark.parametrize("turns", [-3, -2, -1, 0, 1, 2, 3])
def test_periodic_identity_removal_preserves_global_sign(turns: int) -> None:
    """Two-pi Pauli rotations disappear physically with the correct global sign."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), 0)])
    theta = np.array([math.tau * turns])
    result = compilation.compile_endpoint(circuit, theta)
    assert result["output_gates"] == 0
    assert result["theta"] == []
    assert result["source_to_output"] == [None]
    assert result["discarded_periods"] == turns
    assert_unitary_equivalent(circuit, theta, result)


def test_disjoint_fusion_compacts_shared_scales_offsets_and_native_mapping() -> None:
    """Only the frozen resolved endpoint is reparameterized into independent angles."""
    circuit = ParameterizedCircuit(
        3,
        [
            ParameterizedGate(
                "rxx", (2, 0), 0, angle_scale=2, angle_offset=0.3, logical_gate_id="left", native_gate_id="a"
            ),
            ParameterizedGate("ry", (1,), None, angle_offset=0.4, logical_gate_id="middle"),
            ParameterizedGate(
                "rxx", (2, 0), 0, angle_scale=-1, angle_offset=0.2, logical_gate_id="right", native_gate_id="b"
            ),
        ],
    )
    theta = np.array([0.7])
    result = compilation.compile_endpoint(circuit, theta)
    assert result["theta"] == pytest.approx([1.2, 0.4])
    assert result["source_to_output"] == [0, 1, 0]
    assert result["output_mapping"][0]["source_parameter_indices"] == [0, 0]
    assert result["output_mapping"][0]["source_native_gate_ids"] == ["a", "b"]
    assert result["events"][-1]["commuted_past_sources"] == [1]
    assert_unitary_equivalent(circuit, theta, result)


def test_fused_identity_removes_both_windows_after_disjoint_commutation() -> None:
    """A cancelling pair is removed, and its disjoint neighbor keeps its own window."""
    circuit = ParameterizedCircuit(
        2, [ParameterizedGate("rz", (0,), 0), ParameterizedGate("ry", (1,), 1), ParameterizedGate("rz", (0,), 2)]
    )
    theta = np.array([0.3, 0.4, -0.3])
    result = compilation.compile_endpoint(circuit, theta)
    assert result["source_to_output"] == [None, 0, None]
    assert result["output_gates"] == 1
    assert result["removed_noise_windows"] == 2
    assert_unitary_equivalent(circuit, theta, result)


@pytest.mark.parametrize(
    "middle",
    [
        ParameterizedGate("rx", (0,), 1),
        ParameterizedGate("rzz", (0, 1), 1),
        ParameterizedGate("rx", (1,), 1, noise_enabled=False),
    ],
)
def test_overlap_and_retained_noise_boundaries_prevent_fusion(middle: ParameterizedGate) -> None:
    """No overlapping dependency or disjoint noise-flag boundary is crossed."""
    circuit = ParameterizedCircuit(2, [ParameterizedGate("rz", (0,), 0), middle, ParameterizedGate("rz", (0,), 2)])
    theta = np.array([0.3, 0.4, 0.5])
    result = compilation.compile_endpoint(circuit, theta)
    assert result["output_gates"] == 3
    assert not any(event["kind"] == "fuse" for event in result["events"])
    assert_unitary_equivalent(circuit, theta, result)


def test_different_generator_and_ordered_site_support_remain_unfused() -> None:
    """Even commuting overlapping Pauli products require a distinct compiler rule."""
    circuit = ParameterizedCircuit(
        2,
        [ParameterizedGate("rxx", (0, 1), 0), ParameterizedGate("ryy", (0, 1), 1), ParameterizedGate("rxx", (1, 0), 2)],
    )
    theta = np.array([0.3, 0.4, 0.5])
    result = compilation.compile_endpoint(circuit, theta)
    assert result["output_gates"] == 3
    assert_unitary_equivalent(circuit, theta, result)


@pytest.mark.parametrize("theta", [np.array([2 * np.pi, 0.0]), np.array([0.3, 0.4])])
def test_physical_noise_windows_are_removed_instead_of_merely_zeroing_angles(theta: np.ndarray) -> None:
    """Identity removal and fusion both improve the retained all-zero noisy state."""
    circuit = ParameterizedCircuit(2, [ParameterizedGate("rz", (0,), 0), ParameterizedGate("rz", (0,), 1)])
    result = compilation.compile_endpoint(circuit, theta)
    compiled = pilot.restore_circuit(result["circuit"])
    target = np.array([1, 0, 0, 0], dtype=complex)
    original = channels.evaluate(circuit, theta, target, "depolarizing_1s_all")
    simplified = channels.evaluate(compiled, np.asarray(result["theta"]), target, "depolarizing_1s_all")
    assert simplified > original + 1e-4
    assert result["removed_noise_windows"] > 0
    assert all(g.noise_enabled for g in compiled.gates)


def test_numerical_identity_threshold_is_explicit_and_error_bounded() -> None:
    """Discarded residuals are reported; angles outside the threshold remain physical."""
    circuit = ParameterizedCircuit(2, [ParameterizedGate("rx", (0,), 0), ParameterizedGate("ry", (1,), 1)])
    theta = np.array([0.9e-12, 1.1e-12])
    result = compilation.compile_endpoint(circuit, theta)
    assert result["output_gates"] == 1
    assert result["ideal_operator_error_bound"] >= abs(theta[0]) / 2
    assert_unitary_equivalent(circuit, theta, result)


@pytest.mark.parametrize(
    "gate",
    [
        ParameterizedGate("cx", (0, 1)),
        ParameterizedGate("p", (0,), 0),
        ParameterizedGate("rz", (0,), 0, data_map=lambda x: float(x[0])),
        ParameterizedGate("ry", (0, 1), 0),
        ParameterizedGate("rz", (0,), 0, angle_scale=np.nan),
        ParameterizedGate("ry", (0,), param_index=False),
        ParameterizedGate("ry", (0,), 0, fixed_params=(0.1,)),
    ],
)
def test_unsupported_gate_or_parameterization_rejected(gate: ParameterizedGate) -> None:
    """Unsupported matrices/data maps cannot silently become a different endpoint."""
    circuit = ParameterizedCircuit(2, [gate], num_params=1)
    with pytest.raises(ValueError, match="supported Pauli rotations"):
        compilation.compile_endpoint(circuit, np.array([0.1]))


@pytest.mark.parametrize("theta", [np.array([np.nan]), np.array([0.1j]), np.array([]), np.array([[0.1]])])
def test_malformed_parameter_vector_rejected(theta: np.ndarray) -> None:
    """No nonfinite or wrongly shaped parameter vector reaches simplification."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), 0)])
    with pytest.raises(ValueError, match="finite real endpoint"):
        compilation.compile_endpoint(circuit, theta)


def test_unreliable_huge_angle_reduction_rejected() -> None:
    """Modulo reduction cannot silently discard argument-reduction error."""
    with pytest.raises(ValueError, match="reliable modulo"):
        compilation.reduced_angle(1e20)
