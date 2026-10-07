# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent state, phase, metadata and noisy-channel compiler diagnostics."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator, Pauli

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import channels
from experiments.overnight_state_preparation_20260913_0108 import state_preparation_compilation as compiler
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from numpy.typing import NDArray


def checked(circuit: ParameterizedCircuit, theta: NDArray) -> dict[str, Any]:
    """Return a compilation after checking its preserved inputs and independent phase."""
    before, values = copy.deepcopy(pilot.circuit_record(circuit)), theta.copy()
    result = compiler.compile_endpoint(circuit, theta)
    compiled = compiler.restore(result["circuit"])
    np.testing.assert_array_equal(theta, values)
    assert pilot.circuit_record(circuit) == before
    source = compiler.qiskit_state(circuit, theta)
    output = compiler.qiskit_state(compiled, np.asarray(result["theta"]))
    phase = complex(*result["omitted_global_phase_factor"])
    assert np.linalg.norm(source - phase * output) <= result["ideal_state_error_bound"]
    assert result["independent_audit"]["passed"]
    assert result["input_state"] == "all_zero_only"
    assert not result["noisy_equivalence_claim"]
    return result


@pytest.mark.parametrize("phase", [-2.9, -0.2, 0, 0.7, math.pi])
def test_complex_global_phase_residual_is_stable_at_machine_precision(phase: float) -> None:
    """Direct vector subtraction resolves tiny deviations hidden by overlap rounding."""
    before = np.array([1 + 2j, -3j, 0.7, -0.3 + 0.8j])
    before /= np.linalg.norm(before)
    factor, residual = compiler.global_phase_residual(before, np.exp(1j * phase) * before)
    assert factor == pytest.approx(np.exp(1j * phase))
    assert residual < 1e-15
    state = np.array([1, 0], dtype=complex)
    after = np.array([math.cos(9e-13), math.sin(9e-13)], dtype=complex)
    _, tiny = compiler.global_phase_residual(state, after)
    assert tiny == pytest.approx(9e-13, rel=1e-12)
    assert abs(np.vdot(state, after)) == 1


def test_original_style_initial_rz_coordinates_are_removed_with_permuted_indices() -> None:
    """A synthetic q6 product layer has precisely the six unobservable RZ coordinates."""
    gates = [
        ParameterizedGate(
            name,
            (site,),
            3 * site + index,
            logical_gate_id=f"logical_{site}_{index}",
            native_gate_id=100 + 3 * site + index,
        )
        for site in range(6)
        for name, index in (("rz", 2), ("ry", 1), ("rz", 0))
    ]
    circuit = ParameterizedCircuit(6, gates)
    theta = np.linspace(0.21, 0.77, 18)
    result = checked(circuit, theta)
    removed = [event["source_parameter_index"] for event in result["events"] if event["removed"]]
    assert removed == [2, 5, 8, 11, 14, 17]
    assert result["output_gates"] == 12
    retained_parameters = [index for index in range(18) if index not in removed]
    assert result["retained_source_parameter_indices"] == retained_parameters
    assert result["theta"] == theta[retained_parameters].tolist()
    for mapping, gate in zip(result["output_mapping"], compiler.restore(result["circuit"]).gates, strict=True):
        original = circuit.gates[mapping["source_gate_index"]]
        assert gate.logical_gate_id == original.logical_gate_id
        assert gate.native_gate_id == original.native_gate_id
        assert gate.sites == original.sites
        assert gate.angle_scale == original.angle_scale
        assert gate.angle_offset == original.angle_offset


def test_complex_y_eigenstate_gate_drops_but_non_eigenstate_rotation_remains() -> None:
    """Eigenstate propagation recognizes a complex prefix, beyond computational constants."""
    circuit = ParameterizedCircuit(
        1, [ParameterizedGate("rx", (0,), 0), ParameterizedGate("ry", (0,), 1), ParameterizedGate("rz", (0,), 2)]
    )
    result = checked(circuit, np.array([-math.pi / 2, 0.37, 0.23]))
    assert result["source_to_output"] == [0, None, 1]
    assert complex(*result["omitted_global_phase_factor"]) == pytest.approx(np.exp(-0.185j))


def test_nonadjacent_reversed_two_site_eigenstate_and_metadata_are_preserved() -> None:
    """A reversed XX on a nonadjacent product eigenstate disappears without reordering neighbors."""
    gates = [
        ParameterizedGate("ry", (2,), 3, angle_scale=-2, angle_offset=0.1, native_gate_id="left"),
        ParameterizedGate("ry", (0,), 2, native_gate_id="right"),
        ParameterizedGate("rxx", (2, 0), 0, noise_enabled=False, native_gate_id="removable"),
        ParameterizedGate("ryy", (2, 0), 1, logical_gate_id="retained", native_gate_id="joint", angle_scale=0.5),
    ]
    theta = np.array([0.4, 0.6, math.pi / 2, (0.1 - math.pi / 2) / 2])
    result = checked(ParameterizedCircuit(3, gates), theta)
    assert result["source_to_output"] == [0, 1, None, 2]
    restored = compiler.restore(result["circuit"])
    assert restored.gates[-1].sites == (2, 0)
    assert restored.gates[-1].native_gate_id == "joint"
    assert restored.gates[-1].angle_scale == pytest.approx(0.5)
    assert result["removed_noise_windows"] == 0
    assert [gate.param_index for gate in restored.gates] == [2, 1, 0]


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("seed", [11, 29])
def test_random_affine_circuits_keep_full_zero_state_with_independent_audit(n: int, seed: int) -> None:
    """Generic complex preparations retain every noneigenstate rotation and map surviving parameters."""
    rng = np.random.default_rng(seed)
    gates = []
    theta = rng.normal(size=24)
    permutation = rng.permutation(24)
    for index in range(24):
        arity = int(rng.integers(1, min(2, n) + 1))
        axis = str(rng.choice(["x", "y", "z"]))
        sites = tuple(int(site) for site in rng.choice(n, arity, replace=False))
        gates.append(
            ParameterizedGate(
                "r" + axis * arity,
                sites,
                int(permutation[index]),
                angle_scale=float(rng.choice([-2, 0.5, 1])),
                angle_offset=float(rng.normal()),
                logical_gate_id=index * 3,
                native_gate_id=f"native_{index}",
                noise_enabled=index % 3 != 0,
            )
        )
    result = checked(ParameterizedCircuit(n, gates), theta)
    assert all(
        event["phase_aligned_l2_residual"] <= compiler.EIGENSTATE_ATOL for event in result["events"] if event["removed"]
    )
    assert all(
        event["phase_aligned_l2_residual"] > compiler.EIGENSTATE_ATOL
        for event in result["events"]
        if not event["removed"]
    )


def test_small_residual_tolerance_and_accumulated_error_budget() -> None:
    """Many individually negligible rotations contribute their full cumulative error bound."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("rx", (0,), index) for index in range(9)])
    theta = np.array([1.8e-12] * 8 + [2.2e-12])
    result = checked(circuit, theta)
    assert result["retained_source_gate_indices"] == [8]
    assert result["summed_removed_l2_residuals"] == pytest.approx(8 * 9e-13, rel=1e-12)
    assert result["ideal_state_error_bound"] >= 8 * 9e-13
    assert result["independent_audit"]["recorded_phase_l2_error"] > 7e-12


def test_periodic_and_continuous_removed_phases_accumulate_with_empty_output() -> None:
    """The compiled all-zero circuit records complex phase rather than assuming sign-only equivalence."""
    circuit = ParameterizedCircuit(
        1, [ParameterizedGate("rx", (0,), 0), ParameterizedGate("rz", (0,), 1), ParameterizedGate("ry", (0,), 2)]
    )
    result = checked(circuit, np.array([2 * math.pi, 0.63, -4 * math.pi]))
    assert result["theta"] == []
    assert result["circuit"]["gates"] == []
    assert result["parameter_to_output"] == [None, None, None]
    assert complex(*result["omitted_global_phase_factor"]) == pytest.approx(-np.exp(-0.315j))


def test_zero_state_equivalence_does_not_claim_full_unitary_equivalence() -> None:
    """An RZ removed on zero still changes the relative phase on an arbitrary input."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("rz", (0,), 0)])
    result = checked(circuit, np.array([0.7]))
    original = QuantumCircuit(1)
    original.rz(0.7, 0)
    phase = complex(*result["omitted_global_phase_factor"])
    assert np.linalg.norm(Operator(original).data - phase * np.eye(2)) > 0.5


def qiskit_noisy(circuit: ParameterizedCircuit, theta: NDArray, target: NDArray) -> float:
    """Return fidelity using independent Qiskit gates and local depolarizing mixtures."""
    calibration = CalibratedLocalDepolarizingProvider()
    rho = DensityMatrix.from_label("0" * circuit.num_qubits)
    for gate in circuit.gates:
        step = QuantumCircuit(circuit.num_qubits)
        angle = gate.angle_scale * float(theta[cast("int", gate.param_index)]) + gate.angle_offset
        getattr(step, gate.name)(angle, *gate.sites)
        rho = rho.evolve(step)
        if gate.noise_enabled:
            strength = calibration.one_qubit_strength if len(gate.sites) == 1 else calibration.two_qubit_strength
            weight = -math.expm1(-4 * strength) / 4
            for site in gate.sites:
                data = (1 - 3 * weight) * rho.data
                for label in "XYZ":
                    data += weight * rho.evolve(Operator(Pauli(label)), qargs=[site]).data
                rho = DensityMatrix(data)
    return float(np.vdot(target, rho.data @ target).real)


def test_compiled_noisy_final_fidelity_matches_independent_qiskit_for_both_circuits() -> None:
    """Physically removed eigenstate windows change noisy fidelity and both versions remain auditable."""
    circuit = ParameterizedCircuit(
        2, [ParameterizedGate("rz", (1,), 0), ParameterizedGate("ry", (0,), 1), ParameterizedGate("rxx", (1, 0), 2)]
    )
    theta = np.array([0.5, 0.7, 0.4])
    result = checked(circuit, theta)
    target = compiler.qiskit_state(circuit, theta)
    compiled = compiler.restore(result["circuit"])
    values = np.asarray(result["theta"])
    before = channels.evaluate(circuit, theta, target, "depolarizing_1s_all")
    after = channels.evaluate(compiled, values, target, "depolarizing_1s_all")
    assert before == pytest.approx(qiskit_noisy(circuit, theta, target), abs=2e-13)
    assert after == pytest.approx(qiskit_noisy(compiled, values, target), abs=2e-13)
    assert abs(before - after) > 1e-5
    assert result["removed_noise_windows"] == 1


@pytest.mark.parametrize(
    "kind", ["shared", "unused", "fixed", "data", "unsupported", "complex", "nonfinite", "arity", "qubits"]
)
def test_unsupported_parameter_domains_are_rejected(kind: str) -> None:
    """Unsupported families must not lose parameters or metadata through silent conversion."""
    gate = ParameterizedGate("ry", (0,), 0)
    circuit = ParameterizedCircuit(2, [gate])
    theta = np.array([0.4])
    if kind == "shared":
        circuit.gates.append(ParameterizedGate("rz", (1,), 0))
    elif kind == "unused":
        circuit.num_params = 2
        theta = np.array([0.4, 0.7])
    elif kind == "fixed":
        gate.param_index = None
    elif kind == "data":
        gate.data_map = lambda x: float(x[0])
    elif kind == "unsupported":
        gate.name = "p"
    elif kind == "complex":
        theta = theta.astype(complex)
    elif kind == "nonfinite":
        gate.angle_scale = np.nan
    elif kind == "arity":
        gate.name = "rzz"
    else:
        circuit.num_qubits = 7
    with pytest.raises(ValueError, match=r"unique-scalar|independent scalar|Shared"):
        compiler.compile_endpoint(circuit, theta)


def test_independent_audit_detects_an_incorrect_local_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken prefix propagator cannot certify its own incorrect gate removals."""
    monkeypatch.setattr(compiler, "_apply", lambda states, *_args: states.copy())
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), 0)])
    with pytest.raises(FloatingPointError, match="Independent Qiskit"):
        compiler.compile_endpoint(circuit, np.array([0.4]))
