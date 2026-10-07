# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent dense oracles and resource regressions for the compressed comb."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import DensityMatrix, Operator, Statevector

pytest.importorskip("torch")
import opt_einsum as oe
import torch

from benchmarks.state_preparation.ballarin import ballarin_pauli_probability
from benchmarks.state_preparation.targets import load_target
from experiments.legacy_topdown_dense import noise_channel
from experiments.state_preparation_competitors.iqsp_comb_backend import (
    CombLayout,
    CombOptions,
    CombTarget,
    CombTensorNetworkObjective,
    comb_target_from_vector,
)
from experiments.state_preparation_competitors.iqsp_tn import iqsp_tn_ansatz, iqsp_tn_candidates, optimize_tn_adam
from experiments.state_preparation_competitors.iqsp_tn_backend import TensorNetworkObjective
from experiments.state_preparation_competitors.vendor.ballarin_comb.combdevice import CombTNDevice
from experiments.state_preparation_competitors.vendor.ballarin_comb.svd import SVDGESDD

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _random_target(n: int, seed: int = 817) -> NDArray[np.complex128]:
    """Return asymmetric complex amplitudes for endianness and conjugation checks."""
    rng = np.random.default_rng(seed)
    state = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return state / np.linalg.norm(state)


def _dense_target(target: CombTarget) -> NDArray[np.complex128]:
    """Independently contract all pure target tensors, only for small test fixtures.

    Returns:
        Target amplitudes in Qiskit ordering.
    """
    n = target.layout.num_qubits
    edges = {edge: n + j for j, edge in enumerate(target.layout.edges)}
    operands = []
    boundary = n + len(edges)
    for site, tensor in enumerate(target.tensors):
        labels = []
        for axis in range(tensor.ndim):
            if axis == 1:
                labels.append(site)
                continue
            matching = [
                edge
                for edge in edges
                if site in edge and target.layout.bond_axis(site, edge[1] if site == edge[0] else edge[0]) == axis
            ]
            if matching:
                labels.append(edges[matching[0]])
            else:
                labels.append(boundary)
                boundary += 1
        operands.extend([tensor.numpy(), labels])
    return oe.contract(*operands, list(reversed(range(n)))).reshape(-1)


def _oracle(circuit: QuantumCircuit, hardware: str | None) -> NDArray[np.complex128]:
    """Return an independent Qiskit density evolution, including paper H2 noise."""
    density = DensityMatrix.from_label("0" * circuit.num_qubits)
    paulis = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
    for item in circuit.data:
        sites = [circuit.find_bit(q).index for q in item.qubits]
        density = density.evolve(Operator(item.operation), qargs=sites)
        if hardware in {"IBM", "Google"}:
            density = density.evolve(noise_channel(hardware, len(sites)), qargs=sites)
        elif hardware == "H2" and len(sites) == 2:
            r = ballarin_pauli_probability(float(item.operation.params[0]))
            for site in sites:
                data = (1 - 3 * r) * density.data + sum(
                    r * density.evolve(Operator(p), qargs=[site]).data for p in paulis
                )
                density = DensityMatrix(data)
    return density.data


@pytest.mark.parametrize("layout", [CombLayout(2), CombLayout(4), CombLayout(2, 2), CombLayout(3, 2), CombLayout(2, 3)])
def test_target_factorization_and_isometry_transport(layout: CombLayout) -> None:
    """A genuine branching comb preserves complex amplitudes under center shifts."""
    vector = _random_target(layout.num_qubits)
    target = comb_target_from_vector(vector, layout)
    np.testing.assert_allclose(_dense_target(target), vector, atol=3e-15)
    state = CombTNDevice(layout.sites_per_tooth, layout.teeth)
    state.states = list(target.tensors)
    state.iso_center = (layout.teeth - 1) * layout.sites_per_tooth
    for site in [0, layout.num_qubits - 1, 1, 0]:
        state.iso_towards(site)
        moved = CombTarget(layout, tuple(state.states))
        np.testing.assert_allclose(_dense_target(moved), vector, atol=5e-15)


@pytest.mark.parametrize("hardware", [None, "IBM", "Google", "H2"])
@pytest.mark.parametrize("layout", [CombLayout(2), CombLayout(2, 2)])
def test_fidelity_and_gradient_against_independent_density(layout: CombLayout, hardware: str | None) -> None:
    """Pure, correlated logical, and H2 channels match density and finite differences."""
    n = layout.num_qubits
    p = ParameterVector("t", 3)
    circuit = QuantumCircuit(n)
    for j in range(n):
        circuit.ry(0.24 + j * 0.31, j)
    circuit.rx(p[0], n - 1)
    for j, (a, b) in enumerate(layout.edges):
        circuit.rzz(p[1] + j * 0.13, b, a)
    circuit.ry(2 * p[2] - p[0], 0)
    vector = _random_target(n)
    target = comb_target_from_vector(vector, layout)
    objective = CombTensorNetworkObjective(
        circuit,
        target,
        hardware,
        options=CombOptions(closed_bond=64, environment_bond=32, closed_tolerance=1e-15, environment_tolerance=1e-15),
    )
    theta = np.array([0.31, 0.42, -0.17])
    rho = _oracle(objective.bind(theta), hardware)
    loss, gradient = objective.loss_gradient(theta)
    assert loss == pytest.approx(1 - np.vdot(vector, rho @ vector).real, abs=2e-12)
    delta = 1e-5
    expected = []
    for axis in np.eye(3):
        difference = _oracle(objective.bind(theta + delta * axis), hardware) - _oracle(
            objective.bind(theta - delta * axis), hardware
        )
        expected.append(-np.vdot(vector, difference @ vector).real / (2 * delta))
    np.testing.assert_allclose(gradient, expected, atol=3e-7, rtol=3e-5)
    receipt = objective.diagnostics(theta)
    assert receipt["trace"] == pytest.approx(1, abs=1e-12)
    assert receipt["global_state_or_density_materialized"] is False
    assert np.linalg.eigvalsh(rho).min() >= -1e-13


def test_backbone_gate_with_nontrivial_purification_regression() -> None:
    """Upstream root-root permutation must preserve physical/environment ordering."""
    layout = CombLayout(3, 2)
    vector = _random_target(6)
    circuit = QuantumCircuit(6)
    circuit.h(range(6))
    circuit.ry(0.456, 1)
    circuit.rxx(0.237, 0, 1)
    circuit.cx(1, 0)
    circuit.rxx(0.237, 0, 3)
    circuit.cx(3, 0)
    options = CombOptions(closed_bond=128, environment_bond=64, closed_tolerance=1e-15, environment_tolerance=1e-15)
    objective = CombTensorNetworkObjective(circuit, comb_target_from_vector(vector, layout), "IBM", options=options)
    expected = _oracle(circuit, "IBM")
    assert objective.fidelity([]) == pytest.approx(np.vdot(vector, expected @ vector).real, abs=1e-12)


@pytest.mark.parametrize("angle", [0.0, -0.7, 2 * np.pi + 0.21])
def test_h2_zero_angles_signed_angles_and_noiseless_single_qubit_gates(angle: float) -> None:
    """The paper channel keeps the constant RZZ error and noises root sites too."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.rzz(angle, 0, 1)
    vector = Statevector.from_instruction(circuit).data
    objective = CombTensorNetworkObjective(circuit, comb_target_from_vector(vector, CombLayout(2)), "H2")
    rho = _oracle(circuit, "H2")
    assert objective.fidelity([]) == pytest.approx(np.vdot(vector, rho @ vector).real, abs=1e-13)
    assert objective.fidelity([]) < 1


def test_caps_and_tolerances_actually_compress_and_remain_normalized() -> None:
    """Compression reduces storage and converges to an independent exact circuit."""
    layout = CombLayout(2, 2)
    vector = _random_target(4)
    circuit = QuantumCircuit(4)
    for j in range(4):
        circuit.ry(0.23 + j * 0.45, j)
    for a, b in layout.edges:
        circuit.rxx(0.37, a, b)
        circuit.ry(0.16, b)
    target = comb_target_from_vector(vector, layout)
    rho = _oracle(circuit, "IBM")
    exact = np.vdot(vector, rho @ vector).real
    receipts, errors = [], []
    for cap, tolerance in [(1, 1e-2), (4, 1e-8), (64, 1e-15)]:
        objective = CombTensorNetworkObjective(
            circuit,
            target,
            "IBM",
            options=CombOptions(
                closed_bond=cap, environment_bond=cap, closed_tolerance=tolerance, environment_tolerance=tolerance
            ),
        )
        errors.append(abs(objective.fidelity([]) - exact))
        receipt = objective.diagnostics([])
        receipts.append(receipt)
        assert receipt["maximum_closed_bond"] <= cap
        assert receipt["maximum_environment_bond"] <= cap
        assert receipt["trace"] == pytest.approx(1, abs=1e-11)
        assert 0 <= objective.fidelity([]) <= 1 + 1e-12
    assert receipts[0]["stored_tensor_elements"] < receipts[-1]["stored_tensor_elements"]
    assert receipts[0]["sum_local_relative_discarded_weight"] > 0
    assert errors[-1] < 1e-11
    assert errors[-1] < errors[0]


def test_twelve_qubit_noisy_comb_against_analytic_product_channel() -> None:
    """Twelve-qubit noisy values/gradients use bounded comb storage, not density tapes."""
    layout = CombLayout(4, 3)
    vector = np.zeros(4096, dtype=complex)
    vector[0] = 1
    target = comb_target_from_vector(vector, layout)
    p = Parameter("angle")
    circuit = QuantumCircuit(12)
    for a, b in layout.edges:
        circuit.rzz(p, a, b)
    objective = CombTensorNetworkObjective(
        circuit, target, "H2", options=CombOptions(closed_bond=8, environment_bond=4)
    )
    degrees = np.bincount(np.array(layout.edges).ravel(), minlength=12)

    def expected(angle: float) -> float:
        """Return survival after repeated independent single-site depolarization."""
        r = ballarin_pauli_probability(angle)
        return float(np.prod((1 + (1 - 4 * r) ** degrees) / 2))

    loss, gradient = objective.loss_gradient([0.3])
    assert 1 - loss == pytest.approx(expected(0.3), abs=2e-10)
    delta = 1e-5
    derivative = -(expected(0.3 + delta) - expected(0.3 - delta)) / (2 * delta)
    assert gradient[0] == pytest.approx(derivative, abs=2e-6)
    receipt = objective.diagnostics([0.3])
    assert receipt["stored_tensor_elements"] < 4096
    assert receipt["largest_local_tensor_elements"] < 4096


def test_su4_adam_training_and_replay() -> None:
    """The original Adam update can train and replay a compressed objective."""
    vector = np.array([1, 0, 0, 1]) / np.sqrt(2)
    objective = CombTensorNetworkObjective(iqsp_tn_ansatz(2, 1), comb_target_from_vector(vector, CombLayout(2)))
    # Away from rank-changing identity points: same explicitly supplied seed.
    initial = np.random.default_rng(41).normal(scale=0.1, size=15)
    final, receipt = optimize_tn_adam(objective, initial, steps=80)
    assert receipt["final_fidelity"] > 0.99
    assert objective.fidelity(final) == pytest.approx(receipt["final_fidelity"], abs=1e-12)
    dense = Statevector.from_instruction(objective.bind(final)).data
    assert abs(np.vdot(vector, dense)) ** 2 == pytest.approx(receipt["final_fidelity"], abs=1e-12)


@pytest.mark.parametrize("hardware", [None, "IBM", "H2"])
def test_empty_circuit_and_global_phase(hardware: str | None) -> None:
    """No operations leave the zero state independent of global phase."""
    circuit = QuantumCircuit(2, global_phase=0.23)
    vector = _random_target(2)
    objective = CombTensorNetworkObjective(circuit, comb_target_from_vector(vector, CombLayout(2)), hardware)
    loss, gradient = objective.loss_gradient([])
    assert 1 - loss == pytest.approx(abs(vector[0]) ** 2)
    assert gradient.size == 0
    assert float(objective.bind([]).global_phase) == pytest.approx(0.23)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"closed_bond": 0},
        {"environment_bond": True},
        {"closed_tolerance": -1},
        {"environment_tolerance": np.nan},
        {"environment_tolerance": 1},
        {"max_tensor_elements": 1.5},
    ],
)
def test_invalid_compression_options(kwargs: dict) -> None:
    """Invalid rank/tolerance settings fail before tensor allocation."""
    with pytest.raises(ValueError, match="must"):
        CombOptions(**kwargs)


@pytest.mark.parametrize("size", [0, 1, True, 2.5])
def test_invalid_layout(size: int) -> None:
    """A tooth has at least two sites in the reused author implementation."""
    with pytest.raises(ValueError, match="must"):
        CombLayout(size)
    with pytest.raises(ValueError, match="must"):
        CombLayout(2, 0)


def test_invalid_inputs_and_memory_guard() -> None:
    """Wrong targets, unsupported topology, and oversized tensors cannot silently run."""
    layout = CombLayout(2, 2)
    target = comb_target_from_vector(_random_target(4), layout)
    with pytest.raises(ValueError, match="normalized"):
        comb_target_from_vector(np.ones(16), layout)
    with pytest.raises(ValueError, match="normalized"):
        CombTarget(layout, tuple(t * 2 for t in target.tensors))
    with pytest.raises(ValueError, match="count"):
        CombTarget(layout, target.tensors[:-1])
    circuit = QuantumCircuit(4)
    circuit.cx(1, 3)
    with pytest.raises(ValueError, match="edge"):
        CombTensorNetworkObjective(circuit, target)
    circuit = QuantumCircuit(4)
    circuit.cx(0, 2)
    with pytest.raises(ValueError, match="RZZ"):
        CombTensorNetworkObjective(circuit, target, "H2")
    objective = CombTensorNetworkObjective(circuit, target, options=CombOptions(max_tensor_elements=1))
    with pytest.raises(ValueError, match="max_tensor"):
        objective.fidelity([])
    for bad in [[1], [np.nan], [1j]]:
        with pytest.raises(ValueError, match="finite real"):
            objective.fidelity(bad)


def test_two_phase_iqsp_backend_selection_preserves_method_and_noise() -> None:
    """The existing candidate pipeline can use compressed ideal and noisy stages."""
    target = np.array([1, 0, 0, 1]) / np.sqrt(2)
    candidate = iqsp_tn_candidates(
        target,
        target_id="comb_backend_test",
        depths=(1,),
        interpolation_steps=2,
        construction_steps=2,
        noisy_steps=2,
        comb_options=CombOptions(),
        target_construction="dense_svd",
    )[0]
    assert candidate.metadata["noisy_tensor_network"]["backend"] == "ballarin_locally_purified_comb"
    assert candidate.metadata["stages"][0]["tensor_network"]["backend"] == "ballarin_compressed_comb"
    assert candidate.metadata["noisy"]["optimizer"] == "existing standard bias-corrected Adam"
    assert candidate.metadata["noisy"]["retention"].startswith("final_iterate")
    rho = _oracle(candidate.circuit, "IBM")
    expected = np.vdot(target, rho @ target).real
    actual = candidate.metadata["postprocessing"]["final_noisy_fidelity"]
    assert actual == pytest.approx(expected, abs=2e-7)


@pytest.mark.parametrize("cap", [1, 2])
def test_truncated_gradients_match_finite_differences_away_from_rank_changes(cap: int) -> None:
    """Active state/purification truncations have the expected local derivatives."""
    layout = CombLayout(2, 2)
    p = ParameterVector("p", 2)
    circuit = QuantumCircuit(4)
    circuit.ry(0.43, 0)
    circuit.ry(0.82, 2)
    circuit.rxx(p[0], 0, 2)
    circuit.ry(p[1], 1)
    circuit.rzz(0.34, 0, 1)
    target = comb_target_from_vector(_random_target(4), layout)
    objective = CombTensorNetworkObjective(
        circuit, target, "IBM", options=CombOptions(closed_bond=cap, environment_bond=cap)
    )
    theta = np.array([0.46, -0.28])
    _, gradient = objective.loss_gradient(theta)
    delta = 1e-5
    difference = [
        -(objective.fidelity(theta + delta * axis) - objective.fidelity(theta - delta * axis)) / (2 * delta)
        for axis in np.eye(2)
    ]
    np.testing.assert_allclose(gradient, difference, atol=2e-7, rtol=2e-5)


def test_author_identity_rank_change_is_explicitly_not_exact_gradient_parity() -> None:
    """Record the upstream identity-origin limitation instead of hiding it with jitter."""
    target = np.array([1, 0, 0, 1]) / np.sqrt(2)
    circuit = iqsp_tn_ansatz(2, 1)
    compressed = CombTensorNetworkObjective(circuit, comb_target_from_vector(target, CombLayout(2)))
    exact = TensorNetworkObjective(circuit, target)
    zero = np.zeros(15)
    loss, gradient = compressed.loss_gradient(zero)
    exact_loss, exact_gradient = exact.loss_gradient(zero)
    assert loss == pytest.approx(exact_loss, abs=1e-14)
    assert np.linalg.norm(gradient) < 1e-12
    assert np.linalg.norm(exact_gradient) > 0.1
    assert "rank changes" in compressed.diagnostics(zero)["gradient"]


def test_twelve_qubit_noisy_adam_refinement_matches_analytic_fidelity() -> None:
    """A complete short noisy refinement runs without a global density matrix."""
    layout = CombLayout(6, 2)
    vector = np.zeros(4096, dtype=complex)
    vector[0] = 1
    target = comb_target_from_vector(vector, layout)
    circuit = QuantumCircuit(12)
    parameter = Parameter("angle")
    for a, b in layout.edges:
        circuit.rzz(parameter, a, b)
    objective = CombTensorNetworkObjective(circuit, target, "H2")
    theta, history = optimize_tn_adam(objective, [0.4], steps=4)
    assert history["final_fidelity"] > history["initial_fidelity"]
    degrees = np.bincount(np.array(layout.edges).ravel(), minlength=12)
    for record in history["records"]:
        r = ballarin_pauli_probability(record["theta"][0])
        expected = np.prod((1 + (1 - 4 * r) ** degrees) / 2)
        assert record["fidelity"] == pytest.approx(expected, abs=2e-10)
    assert objective.diagnostics(theta)["trace"] == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("hardware", [None, "IBM", "H2"])
@pytest.mark.parametrize("interval", [1, 2, 4, 8])
def test_autograd_checkpoint_recomputation_preserves_loss_gradient_and_diagnostics(
    hardware: str | None, interval: int
) -> None:
    """Recomputation must preserve the original SVD derivative and all truncations."""
    target = comb_target_from_vector(_random_target(4), CombLayout(4))
    circuit = QuantumCircuit(4)
    parameters = ParameterVector("c", 4)
    for repeat in range(2):
        for site in range(4):
            circuit.ry(parameters[site] * (repeat + 1), site)
        for site in range(3):
            circuit.rzz(parameters[site] + 0.13, site, site + 1)
    theta = np.array([0.19, -0.28, 0.48, 0.73])
    plain = CombTensorNetworkObjective(circuit, target, hardware, options=CombOptions())
    recomputed = CombTensorNetworkObjective(
        circuit, target, hardware, options=CombOptions(checkpoint_interval=interval)
    )
    expected_loss, expected_gradient = plain.loss_gradient(theta)
    loss, gradient = recomputed.loss_gradient(theta)
    assert loss == pytest.approx(expected_loss, abs=2e-14)
    np.testing.assert_allclose(gradient, expected_gradient, atol=2e-12, rtol=2e-10)
    assert recomputed.last_diagnostics == plain.last_diagnostics
    assert recomputed.fidelity(theta) == pytest.approx(1 - loss, abs=2e-14)
    assert recomputed.last_diagnostics == plain.last_diagnostics


@pytest.mark.parametrize("interval", [-1, True, 0.5])
def test_checkpoint_interval_validation(interval: int) -> None:
    """Invalid checkpoint controls fail before allocating any tensors."""
    with pytest.raises(ValueError, match="checkpoint_interval"):
        CombOptions(checkpoint_interval=interval)


@pytest.mark.parametrize("tolerance", [1e-28, 1e-30])
def test_gaussian_numerical_null_target_directions(tolerance: float) -> None:
    """Removing numerical null directions retains the twelve-qubit fixture to roundoff."""
    vector = load_target(12, "gaussian_mu0p5_sigma0p1").state_vector
    target = comb_target_from_vector(vector, CombLayout(12), relative_tolerance=tolerance)
    actual = _dense_target(target)
    assert np.linalg.norm(actual - vector) < 8e-14
    assert max(max(t.shape) for t in target.tensors) < 16
    assert np.linalg.norm(actual) == pytest.approx(1, abs=3e-15)
    assert "rtol" in target.source


@pytest.mark.parametrize("layout", [CombLayout(4), CombLayout(2, 2)])
def test_target_compression_error_and_normalization(layout: CombLayout) -> None:
    """Optional target approximation carries the usual sequential SVD error bound."""
    vector = _random_target(4)
    tolerance = 0.02
    target = comb_target_from_vector(vector, layout, relative_tolerance=tolerance)
    actual = _dense_target(target)
    assert np.linalg.norm(actual) == pytest.approx(1, abs=2e-14)
    assert np.linalg.norm(actual - vector) <= np.sqrt(2 * (layout.num_qubits - 1) * tolerance) + 1e-14
    with pytest.raises(ValueError, match="target_tolerance"):
        comb_target_from_vector(vector, layout, relative_tolerance=-1)


@pytest.mark.parametrize("shape", [(64, 4), (4, 64)])
def test_svd_backward_avoids_large_square_projectors(shape: tuple[int, int], monkeypatch: pytest.MonkeyPatch) -> None:
    """The rectangular SVD derivative stays exact without quadratic null-space storage."""
    rng = np.random.default_rng(8)
    value = torch.tensor(rng.normal(size=shape) + 1j * rng.normal(size=shape), requires_grad=True)
    upstream = torch.tensor(rng.normal(size=shape) + 1j * rng.normal(size=shape))
    original = torch.Tensor.__matmul__

    def guarded(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.ndim == right.ndim == 2:
            assert (left.shape[0], right.shape[1]) != (64, 64), "Full orthogonal projector was materialized"
        return original(left, right)

    monkeypatch.setattr(torch.Tensor, "__matmul__", guarded)
    u, singular, v = SVDGESDD.apply(value, torch.tensor(1e-12), None)
    reconstructed = (u * singular) @ v.conj().T
    loss = (upstream.conj() * reconstructed).sum().real
    gradient = torch.autograd.grad(loss, value)[0]
    np.testing.assert_allclose(gradient.detach().numpy(), upstream.numpy(), atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("case", range(8))
def test_svd_backward_preserves_published_derivative(case: int) -> None:
    """Memory changes preserve golden real/complex, rectangular and degenerate derivatives.

    These values were recorded before workspace changes. The companion JSON
    identifies that source and every case. Finite differences would not be
    the right oracle for the author's regularized degenerate/null subspaces.
    """
    with np.load(Path(__file__).with_name("data") / "ballarin_svd_backward_reference.npz") as reference:
        tensors = {
            name: torch.tensor(reference[f"case{case}_{name}"]) for name in ("u", "s", "v", "cutoff", "gu", "gs", "gv")
        }
        context = SimpleNamespace(
            saved_tensors=tuple(tensors[name] for name in ("u", "s", "v", "cutoff")), diagnostics=None
        )
        actual = SVDGESDD.backward(context, tensors["gu"], tensors["gs"], tensors["gv"])[0]
        np.testing.assert_allclose(actual.numpy(), reference[f"case{case}_expected"], atol=1e-12, rtol=2e-13)
