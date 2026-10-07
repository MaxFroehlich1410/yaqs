# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent generator and Qiskit checks for the complete evaluation fan-out."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from qiskit.quantum_info import DensityMatrix, Operator, Pauli
from scipy.linalg import expm

from benchmarks.state_preparation.ballarin import ballarin_pauli_probability, materialize_ballarin_circuit
from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from experiments.overnight_state_preparation_20260913_0108 import channels
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Literal specification transcribed from the immutable benchmark tables. It is
# deliberately independent of standard_noise_profile and channels.window.
# Pauli operator names are mathematical symbols, not spelling candidates.
# spellchecker:off  # noqa: ERA001
_LOCAL_TWO = ("IX", "IY", "IZ", "XI", "YI", "ZI")
_CORRELATED_TWO = ("XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ")
_ALL_TWO = ("IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ")
_EXPECTED_WINDOWS = {
    "dephasing_1s_1q": (("Z",), ()),
    "dephasing_1s_2q": ((), ("IZ", "ZI")),
    "dephasing_1s_all": (("Z",), ("IZ", "ZI")),
    "dephasing_2s_2q": ((), ("ZZ",)),
    "dephasing_1s2s_all": (("Z",), ("IZ", "ZI", "ZZ")),
    "depolarizing_1s_1q": (("X", "Y", "Z"), ()),
    "depolarizing_1s_2q": ((), _LOCAL_TWO),
    "depolarizing_1s_all": (("X", "Y", "Z"), _LOCAL_TWO),
    "depolarizing_2s_2q": ((), _CORRELATED_TWO),
    "depolarizing_1s2s_all": (("X", "Y", "Z"), _ALL_TWO),
}
# spellchecker:on  # noqa: ERA001


@pytest.mark.parametrize("noise_id", _EXPECTED_WINDOWS)
@pytest.mark.parametrize("arity", [1, 2])
def test_exact_window_matches_independent_generator(noise_id: str, arity: int) -> None:
    """Calibrated map agrees with dense Lindblad exponentiation and RB error."""
    resolved = channels.window(noise_id, arity)
    dimension = 2**arity
    expected_labels = _EXPECTED_WINDOWS[noise_id][arity - 1]
    assert resolved["generators"] == list(expected_labels)
    target_error = (0.00064 if arity == 1 else 0.0051) if expected_labels else 0.0
    assert resolved["target_error"] == pytest.approx(target_error, abs=1e-15)
    generator = np.zeros((dimension**2, dimension**2), dtype=complex)
    for name in expected_labels:
        p = Pauli(name).to_matrix()
        generator += resolved["coefficient"] * (np.kron(p, p.conj()) - np.eye(dimension**2))
    expected = expm(generator)
    actual = channels.superoperator(tuple(resolved["probabilities"]))
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=0)
    error = (dimension**2 - np.trace(actual).real) / (dimension * (dimension + 1))
    assert error == pytest.approx(target_error, abs=2e-13)
    if not expected_labels:
        np.testing.assert_array_equal(resolved["probabilities"], np.eye(1, 4**arity)[0])
        np.testing.assert_array_equal(actual, np.eye(dimension**2))


@pytest.mark.parametrize("noise_id", _EXPECTED_WINDOWS)
@pytest.mark.parametrize("arity", [1, 2])
def test_legacy_is_separate_finite_one_jump(noise_id: str, arity: int) -> None:
    """Legacy preserves its coefficients and excludes multiple jump cancellation."""
    row = channels.window(noise_id, arity, "legacy_one_jump")
    labels = _EXPECTED_WINDOWS[noise_id][arity - 1]
    assert row["generators"] == list(labels)
    dimension = 2**arity
    coefficient = 0.00064 if arity == 1 else 0.0051
    expected = np.eye(dimension**2, dtype=complex)
    if labels:
        assert row["coefficient"] == pytest.approx(coefficient, rel=0, abs=0)
        identity_probability = np.exp(-coefficient * len(labels))
        assert row["actual_error"] == pytest.approx(dimension / (dimension + 1) * (1 - identity_probability))
        expected *= identity_probability
        for label in labels:
            p = Pauli(label).to_matrix()
            expected += (1 - identity_probability) / len(labels) * np.kron(p, p.conj())
    else:
        assert row["coefficient"] == pytest.approx(0, abs=0)
    np.testing.assert_allclose(channels.superoperator(tuple(row["probabilities"])), expected, atol=1e-14)


def test_complete_noise_identifier_matrix_is_unchanged() -> None:
    """Ensure the complete twelve-condition matrix cannot silently lose a cell."""
    assert ("noiseless", "ballarin_coupled", *_EXPECTED_WINDOWS) == channels.NOISE_IDS


@pytest.mark.parametrize("arity", [1, 2])
def test_pauli_order_matches_independent_qiskit(arity: int) -> None:
    """Verify the probability indices against independently constructed Paulis."""
    expected = [Pauli("".join(label)).to_matrix() for label in product("IXYZ", repeat=arity)]
    np.testing.assert_array_equal(channels.local_paulis(arity), expected)


def test_local_application_reversed_nonadjacent_qiskit() -> None:
    """Joint ket/bra axis placement agrees with independently embedded Paulis."""
    rng = np.random.default_rng(921)
    state = rng.normal(size=16) + 1j * rng.normal(size=16)
    state /= np.linalg.norm(state)
    rho = DensityMatrix(state)
    probability = tuple(channels.window("depolarizing_2s_2q", 2)["probabilities"])
    sites = (3, 0)
    expected = sum(
        (
            p * rho.evolve(Operator(op), qargs=list(reversed(sites))).data
            for p, op in zip(probability, channels.local_paulis(2), strict=True)
        ),
        start=np.zeros((16, 16), complex),
    )
    np.testing.assert_allclose(channels.apply_channel(rho.data, sites, probability, 4), expected, atol=1e-14)


def example() -> tuple[ParameterizedCircuit, NDArray[np.float64], NDArray[np.complex128]]:
    """Return a nontrivial circuit including disabled noise and signed rotations."""
    circuit = ParameterizedCircuit(
        3,
        [
            ParameterizedGate("ry", (0,), param_index=0),
            ParameterizedGate("rx", (2,), param_index=1),
            ParameterizedGate("rxx", (2, 0), param_index=2),
            ParameterizedGate("ryy", (0, 1), param_index=3),
            ParameterizedGate("rzz", (1, 2), param_index=4, noise_enabled=False),
        ],
        num_params=5,
    )
    theta = np.array([0.4, 1.1, -0.7, 0.9, 0.8])
    target = np.arange(1, 9).astype(complex)
    target /= np.linalg.norm(target)
    return circuit, theta, target


@pytest.mark.parametrize("noise_id", _EXPECTED_WINDOWS)
def test_profile_placement_matches_full_qiskit_reference(noise_id: str) -> None:
    """Apply explicitly specified process sets only after their matching gates."""
    circuit, theta, target = example()
    rho = DensityMatrix.from_label("000")
    for gate in circuit.gates:
        operator, sites = circuit.gate_matrix(gate, theta)
        qargs = list(reversed(sites))
        rho = rho.evolve(Operator(operator), qargs=qargs)
        labels = _EXPECTED_WINDOWS[noise_id][len(sites) - 1] if gate.noise_enabled else ()
        coefficient = channels.window(noise_id, len(sites))["coefficient"]
        probability = -np.expm1(-2 * coefficient) / 2
        for label in labels:
            transformed = rho.evolve(Operator(Pauli(label).to_matrix()), qargs=qargs)
            rho = DensityMatrix((1 - probability) * rho.data + probability * transformed.data)
    expected = float(np.vdot(target, rho.data @ target).real)
    assert channels.evaluate(circuit, theta, target, noise_id) == pytest.approx(expected, abs=1e-13)


def test_local_depolarization_matches_existing_density_oracle() -> None:
    """Full six-local-Pauli calibration reproduces the existing exact adjoint."""
    circuit, theta, target = example()
    expected = density_loss_gradient(circuit, theta, target, CalibratedLocalDepolarizingProvider())[2]
    assert channels.evaluate(circuit, theta, target, "depolarizing_1s_all") == pytest.approx(expected, abs=1e-13)
    expected = density_loss_gradient(circuit, theta, target, CalibratedLocalDepolarizingProvider(0, 0))[2]
    assert channels.evaluate(circuit, theta, target, "noiseless") == pytest.approx(expected, abs=1e-13)


def test_ballarin_matches_materialized_qiskit_sum() -> None:
    """Angle-dependent probabilities apply to lowered native RZZ gates only."""
    circuit, theta, target = example()
    native = materialize_ballarin_circuit(circuit, theta).circuit
    rho = DensityMatrix.from_label("000")
    for gate in native.gates:
        operator, sites, angle = native.gate_matrix_and_angle(gate, np.empty(0))
        rho = rho.evolve(Operator(operator), qargs=list(reversed(sites)))
        if gate.noise_enabled:
            assert angle is not None
            r = ballarin_pauli_probability(angle)
            for site in sites:
                value = (1 - 3 * r) * rho.data
                for p in channels.PAULIS[1:]:
                    value += r * rho.evolve(Operator(p), qargs=[site]).data
                rho = DensityMatrix(value)
    expected = float(np.vdot(target, rho.data @ target).real)
    assert channels.evaluate(circuit, theta, target, "ballarin_coupled") == pytest.approx(expected, abs=1e-13)


def test_ballarin_pruning_is_physical_deletion() -> None:
    """An entangler below the immutable cutoff incurs no channel window."""
    circuit = ParameterizedCircuit(2, [ParameterizedGate("rxx", (0, 1), param_index=0)], num_params=1)
    target = np.array([1, 0, 0, 0], dtype=complex)
    assert channels.evaluate(circuit, np.array([1e-5]), target, "ballarin_coupled") == 1
    assert channels.evaluate(circuit, np.array([0.2]), target, "ballarin_coupled") < 1


def test_invalid_profiles_and_inputs() -> None:
    """Invalid channel choices and nonfinite states fail before simulation."""
    with pytest.raises(ValueError, match="convention"):
        channels.window("depolarizing_1s_all", 1, "incorrect")
    circuit, theta, target = example()
    theta[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        channels.evaluate(circuit, theta, target, "noiseless")


@pytest.mark.parametrize("noise_id", ["noiseless", "ballarin_coupled", "depolarizing_1s_all"])
def test_unknown_convention_rejected_before_channel_dispatch(noise_id: str) -> None:
    """Even channels without a standard window must authenticate the convention."""
    circuit, theta, target = example()
    with pytest.raises(ValueError, match="q1-q6"):
        channels.evaluate(circuit, theta, target, noise_id, convention="incorrect")


@pytest.mark.parametrize("noise_id", ["noiseless", "ballarin_coupled", "depolarizing_1s_all"])
def test_complex_parameters_are_not_silently_cast(noise_id: str) -> None:
    """Reject imaginary control components before a NumPy float conversion."""
    circuit, theta, target = example()
    invalid = theta.astype(complex)
    invalid[0] += 0.1j
    with pytest.raises(ValueError, match="finite"):
        channels.evaluate(circuit, invalid, target, noise_id)


@pytest.mark.parametrize("num_qubits", [True, 3.0, 0, 7])
def test_invalid_qubit_count_is_rejected(num_qubits: object) -> None:
    """Require an integer small-system size before shape or allocation logic."""
    circuit, theta, target = example()
    circuit.num_qubits = cast("int", num_qubits)
    with pytest.raises(ValueError, match="q1-q6"):
        channels.evaluate(circuit, theta, target, "noiseless")
