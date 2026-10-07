# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent finite-difference checks for all fixed ensemble conditions."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from experiments.overnight_state_preparation_20260913_0108 import channels
from experiments.overnight_state_preparation_20260913_0108.ensemble_objective import loss_gradient, mean_loss_gradient
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _problem(*, ballarin_compatible: bool = False) -> tuple[ParameterizedCircuit, NDArray, NDArray]:
    """Build a q3 problem with reversed sites, sharing, offsets and disabled gates.

    Returns:
        Circuit, real theta, and a normalized complex target.
    """
    names = ["rx", "ry", "rz", "p", "rxx", "ryy", "rzz"]
    if not ballarin_compatible:
        names.append("cp")
    gates = [ParameterizedGate("h", (2,)), ParameterizedGate("x", (0,))]
    gates.extend(
        ParameterizedGate(
            name,
            (2, 0) if index >= 4 else (index % 3,),
            param_index=index % 5,
            angle_scale=(-1) ** index * (0.4 + 0.2 * index),
            angle_offset=0.23 * index,
            noise_enabled=index % 3 != 1,
        )
        for index, name in enumerate(names)
    )
    rng = np.random.default_rng(15)
    target = rng.normal(size=8) + 1j * rng.normal(size=8)
    target /= np.linalg.norm(target)
    return ParameterizedCircuit(3, gates, num_params=6), rng.normal(size=6), target


@pytest.mark.parametrize("noise_id", channels.NOISE_IDS)
def test_all_profiles_match_unchanged_evaluator_and_finite_differences(noise_id: str) -> None:
    """Use the native materializer/evaluator independently for every shifted score."""
    circuit, theta, target = _problem(ballarin_compatible=noise_id == "ballarin_coupled")
    original_theta, original_target, original_gates = theta.copy(), target.copy(), copy.deepcopy(circuit.gates)
    gradient, loss, fidelity = loss_gradient(circuit, theta, target, noise_id)
    assert fidelity == pytest.approx(channels.evaluate(circuit, theta, target, noise_id), abs=3e-13)
    assert loss == pytest.approx(1 - fidelity, abs=1e-15)
    finite_difference = []
    for index in range(circuit.num_params):
        upper, lower = theta.copy(), theta.copy()
        upper[index] += 1e-6
        lower[index] -= 1e-6
        finite_difference.append(
            (channels.evaluate(circuit, lower, target, noise_id) - channels.evaluate(circuit, upper, target, noise_id))
            / 2e-6
        )
    np.testing.assert_allclose(gradient, finite_difference, atol=2e-9, rtol=2e-7)
    assert gradient[-1] == 0
    np.testing.assert_array_equal(theta, original_theta)
    np.testing.assert_array_equal(target, original_target)
    assert circuit.gates == original_gates


@pytest.mark.parametrize("name", ["rxx", "ryy", "rzz"])
@pytest.mark.parametrize("angle", [-7.2, -0.7, 0.4, 8.1])
def test_ballarin_signed_wrapped_angles_and_basis_covariance(name: str, angle: float) -> None:
    """Local depolarization after each logical rotation matches native lowering."""
    gate = ParameterizedGate(name, (1, 0), param_index=0, angle_scale=-1.7, angle_offset=0.3, noise_enabled=False)
    circuit = ParameterizedCircuit(
        2, [ParameterizedGate("h", (0,)), ParameterizedGate("ry", (1,), angle_offset=0.7), gate]
    )
    theta = np.array([(angle - 0.3) / -1.7])
    target = np.array([1, 0.3j, -0.2, 0.1 + 0.2j], dtype=complex)
    target /= np.linalg.norm(target)
    gradient, _, fidelity = loss_gradient(circuit, theta, target, "ballarin_coupled")
    assert fidelity == pytest.approx(channels.evaluate(circuit, theta, target, "ballarin_coupled"), abs=2e-13)
    expected = (
        channels.evaluate(circuit, theta - 1e-6, target, "ballarin_coupled")
        - channels.evaluate(circuit, theta + 1e-6, target, "ballarin_coupled")
    ) / 2e-6
    assert gradient[0] == pytest.approx(expected, abs=5e-10, rel=2e-7)


@pytest.mark.parametrize("angle", [-0.8, 0.6])
@pytest.mark.parametrize("scale", [-1.7, 0.4])
def test_ballarin_pure_probability_derivative_includes_both_sites(angle: float, scale: float) -> None:
    """RZZ changes only global phase on |00>; every loss derivative comes from noise."""
    circuit = ParameterizedCircuit(2, [ParameterizedGate("rzz", (1, 0), param_index=0, angle_scale=scale)])
    theta = np.array([angle / scale])
    target = np.array([1, 0, 0, 0], dtype=complex)
    gradient, _, fidelity = loss_gradient(circuit, theta, target, "ballarin_coupled")
    epsilon = 2.1e-4 + 1.43e-3 * abs(angle)
    r = (1 - math.sqrt(1 - 1.25 * epsilon)) / 3
    r_prime = 5 * 1.43e-3 * np.sign(angle) / (24 * math.sqrt(1 - 1.25 * epsilon))
    assert fidelity == pytest.approx((1 - 2 * r) ** 2, abs=2e-14)
    assert gradient[0] == pytest.approx(4 * (1 - 2 * r) * r_prime * scale, abs=1e-13)
    assert abs(gradient[0]) > 1e-5


@pytest.mark.parametrize("angle", [0.0, 1e-5, -9e-5])
@pytest.mark.parametrize("name", ["rxx", "ryy", "rzz"])
def test_ballarin_pruned_region_is_exactly_flat(name: str, angle: float) -> None:
    """Interior deletion removes the ideal gate, its basis changes, and both noise sites."""
    circuit = ParameterizedCircuit(2, [ParameterizedGate(name, (1, 0), param_index=0)])
    target = np.array([1, 0, 0, 0], dtype=complex)
    gradient, loss, fidelity = loss_gradient(circuit, np.array([angle]), target, "ballarin_coupled")
    assert fidelity == 1
    assert loss == 0
    assert gradient[0] == 0


@pytest.mark.parametrize("angle", [1e-4, -1e-4, 1e-4 + 0.5e-12, math.pi, -math.pi, 3 * math.pi])
def test_ballarin_nondifferentiable_boundaries_are_explicitly_rejected(angle: float) -> None:
    """Do not silently supply a derivative across pruning jumps or angle-wrap cusps."""
    circuit = ParameterizedCircuit(2, [ParameterizedGate("rzz", (0, 1), param_index=0)])
    target = np.array([1, 0, 0, 0], dtype=complex)
    with pytest.raises(ValueError, match="pruning boundary or canonical-angle cusp"):
        loss_gradient(circuit, np.array([angle]), target, "ballarin_coupled")


@pytest.mark.parametrize("angle", [1e-4, -1e-4, math.pi])
def test_ballarin_zero_scale_boundary_is_constant_and_allowed(angle: float) -> None:
    """A parameter with zero physical angle scale cannot cross the boundary."""
    circuit = ParameterizedCircuit(
        2, [ParameterizedGate("rzz", (0, 1), param_index=0, angle_scale=0, angle_offset=angle)]
    )
    target = np.array([1, 0, 0, 0], dtype=complex)
    theta = np.array([0.2])
    gradient, _, fidelity = loss_gradient(circuit, theta, target, "ballarin_coupled")
    assert gradient[0] == 0
    assert fidelity == pytest.approx(channels.evaluate(circuit, theta, target, "ballarin_coupled"), abs=1e-14)


def test_disabled_windows_follow_each_normative_profile_convention() -> None:
    """Constant channels respect flags; Ballarin forces retained two-qubit noise."""
    circuit = ParameterizedCircuit(2, [ParameterizedGate("rzz", (0, 1), param_index=0, noise_enabled=False)])
    theta = np.array([0.5])
    target = np.array([1, 0, 0, 0], dtype=complex)
    for noise_id in channels.NOISE_IDS[2:]:
        gradient, _, fidelity = loss_gradient(circuit, theta, target, noise_id)
        assert fidelity == pytest.approx(1, abs=1e-14)
        np.testing.assert_allclose(gradient, 0, atol=1e-15)
    gradient, _, fidelity = loss_gradient(circuit, theta, target, "ballarin_coupled")
    assert fidelity < 1
    assert gradient[0] > 0


def test_ballarin_single_qubit_gates_and_empty_circuit_are_noiseless() -> None:
    """Retain every single-qubit ideal gate without imposing a local noise window."""
    for gates in ([], [ParameterizedGate("rx", (0,), param_index=0), ParameterizedGate("p", (0,), param_index=0)]):
        circuit = ParameterizedCircuit(1, gates, num_params=1)
        theta = np.array([0.5])
        target = np.array([1, 0.2j], dtype=complex)
        target /= np.linalg.norm(target)
        gradient, loss, fidelity = loss_gradient(circuit, theta, target, "ballarin_coupled")
        expected_gradient, expected_loss, expected_fidelity = loss_gradient(circuit, theta, target, "noiseless")
        np.testing.assert_allclose(gradient, expected_gradient, atol=1e-14)
        assert loss == pytest.approx(expected_loss, abs=1e-14)
        assert fidelity == pytest.approx(expected_fidelity, abs=1e-14)


def test_noiseless_and_local_depolarizing_match_existing_exact_adjoint() -> None:
    """Independent existing adjoint agrees in its two overlapping special cases."""
    circuit, theta, target = _problem()
    for noise_id, calibration in (
        ("noiseless", CalibratedLocalDepolarizingProvider(0, 0)),
        ("depolarizing_1s_all", CalibratedLocalDepolarizingProvider()),
    ):
        gradient, loss, fidelity = loss_gradient(circuit, theta, target, noise_id)
        expected_gradient, expected_loss, expected_fidelity = density_loss_gradient(circuit, theta, target, calibration)
        np.testing.assert_allclose(gradient, expected_gradient, atol=1e-13)
        assert loss == pytest.approx(expected_loss, abs=1e-13)
        assert fidelity == pytest.approx(expected_fidelity, abs=1e-13)


def test_mean_objective_is_the_equal_mean_of_all_eleven_noisy_conditions() -> None:
    """Verify the default profile identities, mean score, and finite-difference gradient."""
    circuit, theta, target = _problem(ballarin_compatible=True)
    gradient, loss, fidelities = mean_loss_gradient(circuit, theta, target)
    assert tuple(fidelities) == channels.NOISE_IDS[1:]
    assert "noiseless" not in fidelities
    expected_fidelities = {key: channels.evaluate(circuit, theta, target, key) for key in channels.NOISE_IDS[1:]}
    np.testing.assert_allclose(list(fidelities.values()), list(expected_fidelities.values()), atol=3e-13)
    assert loss == pytest.approx(1 - np.mean(list(expected_fidelities.values())), abs=3e-13)
    expected = np.zeros(circuit.num_params)
    for index in range(circuit.num_params):
        plus, minus = theta.copy(), theta.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        expected[index] = np.mean([
            (channels.evaluate(circuit, minus, target, key) - channels.evaluate(circuit, plus, target, key)) / 2e-6
            for key in channels.NOISE_IDS[1:]
        ])
    np.testing.assert_allclose(gradient, expected, atol=2e-9, rtol=2e-7)


@pytest.mark.parametrize("noise_ids", [(), ("noiseless", "noiseless"), ("unknown",)])
def test_mean_rejects_empty_duplicate_or_unknown_profiles(noise_ids: tuple[str, ...]) -> None:
    """Avoid an accidental weighted objective from duplicated profile identities."""
    circuit, theta, target = _problem()
    with pytest.raises(ValueError, match="nonempty unique supported"):
        mean_loss_gradient(circuit, theta, target, noise_ids)


@pytest.mark.parametrize(
    "invalid",
    [
        "unknown",
        "complex",
        "nan",
        "target",
        "offset",
        "scale",
        "complex_scale",
        "complex_offset",
        "data",
        "unsupported_two_qubit",
    ],
)
def test_invalid_domains_are_rejected(invalid: str) -> None:
    """Invalid inputs and unsupported Ballarin lowering fail before an optimization step."""
    circuit, theta, target = _problem(ballarin_compatible=invalid != "unsupported_two_qubit")
    noise_id = "ballarin_coupled"
    if invalid == "unknown":
        noise_id = "unknown"
    elif invalid == "complex":
        theta = theta.astype(complex)
    elif invalid == "nan":
        theta[0] = np.nan
    elif invalid == "target":
        target *= 0.5
    elif invalid == "offset":
        circuit.gates[-1].angle_offset = np.inf
    elif invalid == "scale":
        circuit.gates[-1].angle_scale = np.nan
    elif invalid == "complex_scale":
        circuit.gates[-1].angle_scale = cast("float", 1j)
    elif invalid == "complex_offset":
        circuit.gates[-1].angle_offset = cast("float", 1j)
    elif invalid == "data":
        circuit.gates[-1].data_map = lambda _x: 0.1
    with pytest.raises(ValueError, match=r"requires a supported|has no exact Ballarin"):
        loss_gradient(circuit, theta, target, noise_id)
