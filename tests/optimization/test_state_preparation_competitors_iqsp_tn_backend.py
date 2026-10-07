# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Dense independent oracles for the pure/purified IQSP tensor networks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter, ParameterVector
from qiskit.quantum_info import DensityMatrix, Operator, Statevector
from scipy.linalg import expm

pytest.importorskip("torch")
pytest.importorskip("opt_einsum")
import torch

from experiments.legacy_topdown_dense import noise_channel
from experiments.state_preparation_competitors import iqsp_tn_backend as backend
from experiments.state_preparation_competitors.evaluation import pauli_distribution

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def _target(qubits: int, seed: int = 94) -> NDArray[np.complex128]:
    """Create reproducible asymmetric complex amplitudes to expose wire reversal.

    Returns:
        A normalized complex vector.
    """
    rng = np.random.default_rng(seed)
    target = rng.normal(size=2**qubits) + 1j * rng.normal(size=2**qubits)
    return target / np.linalg.norm(target)


def _oracle(circuit: QuantumCircuit, hardware: str | None) -> NDArray[np.complex128]:
    """Propagate only in this test with independent Qiskit superoperators.

    Returns:
        The independently evaluated density matrix.
    """
    density = DensityMatrix.from_label("0" * circuit.num_qubits)
    for item in circuit.data:
        sites = [circuit.find_bit(qubit).index for qubit in item.qubits]
        density = density.evolve(Operator(item.operation), qargs=sites)
        if hardware:
            density = density.evolve(noise_channel(hardware, len(sites)), qargs=sites)
    return density.data


@pytest.mark.parametrize("qubits", [1, 2, 3, 6])
def test_exact_target_mps_reconstruction(qubits: int) -> None:
    """TT-SVD preserves asymmetric complex amplitudes and Qiskit endianness."""
    target = _target(qubits)
    tensors, receipt = backend.target_mps(target)
    tensor = tensors[0].numpy()
    for following in tensors[1:]:
        tensor = np.tensordot(tensor, following.numpy(), axes=(-1, 0))
    actual = tensor.reshape((2,) * qubits).transpose(tuple(reversed(range(qubits)))).reshape(-1)
    np.testing.assert_allclose(actual, target, atol=2e-15)
    assert receipt["truncation"] is False
    assert receipt["discarded_squared_singular_values"] == 0


@pytest.mark.parametrize("hardware", [None, "IBM", "Google"])
@pytest.mark.parametrize("qubits", [1, 2, 3, 6])
def test_tensor_fidelity_gradient_and_trace(hardware: str | None, qubits: int) -> None:
    """Local purification exactly matches independently applied 1Q/2Q channels."""
    theta = ParameterVector("x", 3)
    circuit = QuantumCircuit(qubits)
    circuit.h(qubits - 1)
    circuit.ry(2 * theta[0] + 0.3, 0)
    circuit.rx(theta[1] * theta[2], qubits - 1)
    circuit.rz(theta[0] - theta[2], 0)
    if qubits > 1:
        circuit.cx(qubits - 1, 0)
        circuit.rxx(theta[2], 0, qubits - 1)
        circuit.ryy(theta[0], qubits - 1, 0)
        circuit.rzz(theta[1], 0, qubits - 1)
    target = _target(qubits)
    objective = backend.TensorNetworkObjective(circuit, target, hardware)
    values = np.array([0.43, -0.72, 0.28])
    expected = _oracle(objective.bind(values), hardware)
    loss, gradient = objective.loss_gradient(values)
    fidelity = float(np.vdot(target, expected @ target).real)
    assert objective.fidelity(values) == pytest.approx(fidelity, abs=3e-14)
    assert loss == pytest.approx(1 - fidelity, abs=3e-14)
    delta = 1e-5
    difference = []
    for axis in np.eye(3):
        plus = _oracle(objective.bind(values + delta * axis), hardware)
        minus = _oracle(objective.bind(values - delta * axis), hardware)
        difference.append(-np.vdot(target, (plus - minus) @ target).real / (2 * delta))
    np.testing.assert_allclose(gradient, difference, atol=2e-10, rtol=3e-7)
    assert np.linalg.eigvalsh(expected).min() > -1e-13
    receipt = objective.diagnostics(values)
    assert receipt["trace_real"] == pytest.approx(1, abs=2e-14)
    assert receipt["trace_imag"] == pytest.approx(0, abs=2e-14)
    assert receipt["positive_semidefinite_by_construction"]
    assert receipt["global_state_or_density_materialized"] is False
    assert receipt["logical_noise_windows"] == (len(circuit.data) if hardware else 0)
    assert receipt["largest_intermediate_elements"] > 0


@pytest.mark.parametrize("at_origin", [True, False])
def test_su4_matches_author_exponential_and_dense_jacobian(*, at_origin: bool) -> None:
    """The SU4 identity origin retains nonzero first-order entangling directions."""
    parameters = ParameterVector("s", 15)
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.append(Gate("su4", 2, list(parameters)), [1, 0])
    objective = backend.TensorNetworkObjective(circuit, _target(2))
    values = np.zeros(15) if at_origin else np.random.default_rng(82).normal(scale=0.1, size=15)
    paulis = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
    generator = sum(values[index - 1] * np.kron(paulis[index // 4], paulis[index % 4]) for index in range(1, 16))
    actual = backend.su4_matrix(torch.tensor(values)).detach().numpy()
    np.testing.assert_allclose(actual, expm(1j * generator), atol=1e-14)
    np.testing.assert_allclose(actual.conj().T @ actual, np.eye(4), atol=1e-14)
    assert np.linalg.det(actual) == pytest.approx(1, abs=1e-14)
    _, gradient = objective.loss_gradient(values)
    finite_difference = []
    target = _target(2)
    delta = 1e-5
    for axis in np.eye(15):
        plus = Statevector.from_instruction(objective.bind(values + delta * axis)).data
        minus = Statevector.from_instruction(objective.bind(values - delta * axis)).data
        finite_difference.append(-(abs(np.vdot(target, plus)) ** 2 - abs(np.vdot(target, minus)) ** 2) / (2 * delta))
    np.testing.assert_allclose(gradient, finite_difference, atol=3e-10, rtol=3e-7)
    if at_origin:
        assert np.linalg.norm(gradient[[4, 5, 6, 8, 9, 10, 12, 13, 14]]) > 0.01


@pytest.mark.parametrize("hardware", ["IBM", "Google"])
@pytest.mark.parametrize("qubits", [1, 2])
def test_zero_rotation_is_still_a_noise_window(hardware: str, qubits: int) -> None:
    """Zero angles are not simplified away and correlated two-site channels remain joint."""
    circuit = QuantumCircuit(qubits)
    if qubits == 1:
        circuit.rz(0, 0)
    else:
        circuit.rzz(0, 1, 0)
    target = np.eye(2**qubits, dtype=complex)[0]
    objective = backend.TensorNetworkObjective(circuit, target, hardware)
    assert objective.fidelity([]) < 1
    assert len(objective.bind([]).data) == 1
    rho = _oracle(objective.bind([]), hardware)
    assert objective.fidelity([]) == pytest.approx(rho[0, 0].real, abs=1e-15)
    probabilities, matrices = pauli_distribution(hardware, qubits)
    completeness = sum(
        probability * matrix.conj().T @ matrix for probability, matrix in zip(probabilities, matrices, strict=True)
    )
    np.testing.assert_allclose(completeness, np.eye(2**qubits), atol=1e-15)
    _, gradient = objective.loss_gradient([])
    assert gradient.size == 0


@pytest.mark.parametrize("hardware", [None, "IBM"])
def test_empty_circuit_and_global_phase(hardware: str | None) -> None:
    """Empty schedules have unit trace and global phase cannot affect fidelity."""
    circuit = QuantumCircuit(3)
    circuit.global_phase = 0.72
    target = _target(3)
    objective = backend.TensorNetworkObjective(circuit, target, hardware)
    assert objective.fidelity([]) == pytest.approx(abs(target[0]) ** 2)
    assert objective.diagnostics([])["trace_real"] == pytest.approx(1)
    assert float(objective.bind([]).global_phase) == pytest.approx(0.72)


@pytest.mark.parametrize("bad", [[1, 0, 0], [np.nan, 0], [0.2, 0.2], np.eye(2)])
def test_target_rejection(bad: ArrayLike) -> None:
    """Malformed, nonfinite, and unnormalized target fixtures fail before planning."""
    with pytest.raises(ValueError, match="normalized"):
        backend.target_mps(bad)


@pytest.mark.parametrize("bad", [[0, 0], [np.nan], [1j]])
def test_parameter_rejection(bad: ArrayLike) -> None:
    """All public evaluators reject mismatched or nonreal coefficient arrays."""
    circuit = QuantumCircuit(1)
    circuit.rx(Parameter("p"), 0)
    objective = backend.TensorNetworkObjective(circuit, [1, 0])
    for function in [objective.fidelity, objective.loss_gradient, objective.bind, objective.diagnostics]:
        with pytest.raises(ValueError, match="finite real vector"):
            function(bad)


@pytest.mark.parametrize(
    "case",
    ["hardware", "register", "classical", "global", "measurement", "three", "symbolic", "su4", "su4_noise", "rotation"],
)
def test_invalid_circuits(case: str) -> None:
    """Unsupported schedules fail explicitly instead of silently changing algorithms."""
    circuit = QuantumCircuit(2)
    target = _target(2)
    hardware = None
    if case == "hardware":
        hardware = "invalid"
    elif case == "register":
        target = _target(1)
    elif case == "classical":
        circuit = QuantumCircuit(2, 1)
    elif case == "global":
        circuit.global_phase = Parameter("g")
    elif case == "measurement":
        circuit.reset(0)
    elif case == "three":
        circuit = QuantumCircuit(3)
        target = _target(3)
        circuit.ccx(0, 1, 2)
    elif case == "symbolic":
        circuit.u(Parameter("p"), 0, 0, 0)
    elif case == "su4":
        circuit.append(Gate("su4", 2, [0]), [0, 1])
    elif case == "su4_noise":
        circuit.append(Gate("su4", 2, [0] * 15), [0, 1])
        hardware = "IBM"
    elif case == "rotation":
        circuit.append(Gate("rxx", 1, [0]), [0])
    with pytest.raises(ValueError, match=r"Expected|Use|Only|Parameterized|Custom|Pauli"):
        backend.TensorNetworkObjective(circuit, target, hardware)


@pytest.mark.parametrize(
    "bad", [torch.zeros(14), torch.ones(15, dtype=torch.complex128), torch.full((15,), float("nan"))]
)
def test_invalid_su4_coefficients(bad: torch.Tensor) -> None:
    """The attributed SU4 kernel enforces its real coefficient domain."""
    with pytest.raises(ValueError, match="fifteen finite real"):
        backend.su4_matrix(bad)


@pytest.mark.parametrize("bad", [0, -1, 1.2, True])
def test_invalid_memory_limits(bad: object) -> None:
    """Resource limits cannot accidentally accept booleans or nonintegral sizes."""
    with pytest.raises(ValueError, match="positive integer"):
        backend.TensorNetworkOptions(max_intermediate_elements=cast("int", bad))


def test_contraction_memory_guard() -> None:
    """A too-small contraction ceiling fails rather than switching to dense simulation."""
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 2)
    with pytest.raises(ValueError, match="exceeds"):
        backend.TensorNetworkObjective(circuit, _target(3), "IBM", options=backend.TensorNetworkOptions(1))
    with pytest.raises(ValueError, match="greedy"):
        backend.TensorNetworkOptions(optimizer=cast("Literal['greedy']", "unknown"))


@pytest.mark.parametrize("optimizer", ["auto", "auto-hq"])
def test_alternative_contraction_planners(optimizer: Literal["auto", "auto-hq"]) -> None:
    """Path planning changes contraction order without changing the channel."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    objective = backend.TensorNetworkObjective(
        circuit, _target(2), "IBM", options=backend.TensorNetworkOptions(optimizer=optimizer)
    )
    rho = _oracle(circuit, "IBM")
    assert objective.fidelity([]) == pytest.approx(np.vdot(_target(2), rho @ _target(2)).real)


def test_six_qubit_four_cycle_noisy_readiness() -> None:
    """The actual 306-window schedule remains bounded and differentiable at six qubits."""
    parameters = ParameterVector("depth4", 300)
    circuit = QuantumCircuit(6)
    circuit.h(range(6))
    offset = 0
    for _ in range(4):
        for parity in (0, 1):
            for first in range(parity, 5, 2):
                for site in (first, first + 1):
                    circuit.rz(parameters[offset], site)
                    circuit.ry(parameters[offset + 1], site)
                    circuit.rz(parameters[offset + 2], site)
                    offset += 3
                circuit.rxx(parameters[offset], first, first + 1)
                circuit.ryy(parameters[offset + 1], first, first + 1)
                circuit.rzz(parameters[offset + 2], first, first + 1)
                offset += 3
                for site in (first, first + 1):
                    circuit.rz(parameters[offset], site)
                    circuit.ry(parameters[offset + 1], site)
                    circuit.rz(parameters[offset + 2], site)
                    offset += 3
    target = _target(6)
    objective = backend.TensorNetworkObjective(circuit, target, "IBM")
    theta = np.random.default_rng(73).normal(scale=0.13, size=300)
    loss, gradient = objective.loss_gradient(theta)
    expected = _oracle(objective.bind(theta), "IBM")
    assert loss == pytest.approx(1 - np.vdot(target, expected @ target).real, abs=1e-13)
    assert np.isfinite(gradient).all()
    for position in [0, 87, 299]:
        displacement = np.eye(300)[position] * 1e-5
        plus = _oracle(objective.bind(theta + displacement), "IBM")
        minus = _oracle(objective.bind(theta - displacement), "IBM")
        expected_gradient = -np.vdot(target, (plus - minus) @ target).real / 2e-5
        assert gradient[position] == pytest.approx(expected_gradient, abs=3e-10)
    receipt = objective.diagnostics(theta)
    assert receipt["logical_noise_windows"] == 306
    assert receipt["trace_real"] == pytest.approx(1, abs=2e-13)
    assert receipt["largest_intermediate_elements"] <= 2**16


def test_target_and_template_are_owned_copies() -> None:
    """Caller mutations cannot change the fixed objective after path planning."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    target = _target(1)
    objective = backend.TensorNetworkObjective(circuit, target, "IBM")
    before = objective.fidelity([])
    target[:] = 0
    circuit.x(0)
    assert objective.fidelity([]) == before
    assert len(objective.bind([]).data) == 1
