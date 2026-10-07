# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent full-circuit checks for the overnight exact coordinate sweep."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider, exact_density_fidelity
from experiments.overnight_state_preparation_20260913_0108.coordinate import sweep
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate


def _problem(seed: int) -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Create all supported generators with reversed sites and nonunit scales.

    Returns:
        Nontrivial circuit, starting parameters, and random normalized target.
    """
    names = ["rx", "ry", "rz", "p", "rxx", "ryy", "rzz", "cp"]
    gates = [ParameterizedGate("h", (0,)), ParameterizedGate("cx", (0, 2))]
    gates.extend(
        ParameterizedGate(
            name,
            (2, 0) if index >= 4 else (index % 3,),
            param_index=index,
            angle_scale=(-1) ** index * (0.4 + 0.2 * index),
            angle_offset=0.23 * index,
            noise_enabled=index != 3,
        )
        for index, name in enumerate(names)
    )
    rng = np.random.default_rng(seed)
    target = rng.normal(size=8) + 1j * rng.normal(size=8)
    target /= np.linalg.norm(target)
    return ParameterizedCircuit(3, gates), rng.normal(size=8), target


@pytest.mark.parametrize("errors", [(0.0, 0.0), (0.00064, 0.0051), (0.13, 0.24)])
@pytest.mark.parametrize("seed", [17, 29])
def test_sweep_matches_independent_full_circuit_coordinate_optimization(errors: tuple[float, float], seed: int) -> None:
    """Verify every sequential update against independently propagated channels."""
    circuit, theta, target = _problem(seed)
    original = theta.copy()
    gate_snapshot = [vars(gate).copy() for gate in circuit.gates]
    calibration = CalibratedLocalDepolarizingProvider(*errors)
    provider = ExactLocalPauliProvider(calibration)
    expected = theta.copy()
    reference_scores = []
    for gate in circuit.gates:
        index = gate.param_index
        if index is None:
            continue
        current = exact_density_fidelity(circuit, expected, target, provider)
        plus = expected.copy()
        minus = expected.copy()
        plus[index] += np.pi / (2 * gate.angle_scale)
        minus[index] -= np.pi / (2 * gate.angle_scale)
        upper = exact_density_fidelity(circuit, plus, target, provider)
        lower = exact_density_fidelity(circuit, minus, target, provider)
        delta = np.arctan2((upper - lower) / 2, current - (upper + lower) / 2)
        expected[index] += delta / gate.angle_scale
        new_score = exact_density_fidelity(circuit, expected, target, provider)
        reference_scores.append([index, current, new_score])
    result, diagnostics = sweep(circuit, theta, target, calibration)
    np.testing.assert_allclose(result, expected, atol=2e-11, rtol=0)
    np.testing.assert_allclose(
        cast("list[list[float]]", diagnostics["coordinate_scores"]), reference_scores, atol=2e-12, rtol=0
    )
    assert diagnostics["final_fidelity"] == pytest.approx(
        exact_density_fidelity(circuit, result, target, provider), abs=2e-12
    )
    assert diagnostics["initial_fidelity"] == pytest.approx(
        exact_density_fidelity(circuit, theta, target, provider), abs=2e-12
    )
    np.testing.assert_array_equal(theta, original)
    assert [vars(gate) for gate in circuit.gates] == gate_snapshot
    assert diagnostics["density_conjugations"] == 2 * len(circuit.gates) + 3 * circuit.num_params
    assert diagnostics["noise_kernel_calls"] == 2 * len(circuit.gates)
    assert diagnostics["training_trajectories"] == 0
    assert cast("float", diagnostics["training_cpu_seconds"]) >= 0


def test_retained_identity_and_flat_coordinate_keep_noise() -> None:
    """Retained zero rotations are physical noisy windows; flat angles stay put."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("rz", (0,), param_index=0)])
    theta = np.zeros(1)
    target = np.array([1, 0], dtype=complex)
    calibration = CalibratedLocalDepolarizingProvider()
    result, diagnostics = sweep(circuit, theta, target, calibration)
    np.testing.assert_array_equal(result, theta)
    assert diagnostics["nonzero_accepted_updates"] == 0
    assert 0 < cast("float", diagnostics["final_fidelity"]) < 1
    assert diagnostics["final_fidelity"] == pytest.approx(
        exact_density_fidelity(circuit, theta, target, ExactLocalPauliProvider()), abs=1e-12
    )


@pytest.mark.parametrize("with_gate", [False, True])
def test_fixed_circuits_and_unused_parameters_are_supported(*, with_gate: bool) -> None:
    """No trainable occurrences leave unused parameter values untouched."""
    gates = [ParameterizedGate("h", (0,), noise_enabled=False)] if with_gate else []
    circuit = ParameterizedCircuit(1, gates, num_params=1)
    theta = np.array([0.23])
    target = np.array([1, 0], dtype=complex)
    result, diagnostics = sweep(circuit, theta, target, CalibratedLocalDepolarizingProvider())
    np.testing.assert_array_equal(result, theta)
    assert diagnostics["coordinate_updates"] == 0
    assert diagnostics["final_fidelity"] == pytest.approx(0.5 if with_gate else 1)


@pytest.mark.parametrize(
    "invalid",
    [
        "shared",
        "zero_scale",
        "data",
        "offset_nan",
        "theta_shape",
        "theta_complex",
        "theta_nan",
        "target_norm",
        "target_shape",
        "target_nan",
        "too_large",
        "wrong_channel",
    ],
)
def test_rejects_inputs_that_invalidate_coordinate_model(invalid: str) -> None:
    """Fail before updating for unsupported channels or malformed gate mappings."""
    circuit, theta, target = _problem(15)
    calibration = CalibratedLocalDepolarizingProvider()
    if invalid == "shared":
        circuit.gates[-1].param_index = 0
    elif invalid == "zero_scale":
        circuit.gates[-1].angle_scale = 0
    elif invalid == "data":
        circuit.gates[-1].data_map = lambda _x: 0.1
    elif invalid == "offset_nan":
        circuit.gates[-1].angle_offset = np.nan
    elif invalid == "theta_shape":
        theta = theta[:-1]
    elif invalid == "theta_complex":
        theta = theta.astype(complex)
    elif invalid == "theta_nan":
        theta[0] = np.nan
    elif invalid == "target_norm":
        target *= 2
    elif invalid == "target_shape":
        target = target[:-1]
    elif invalid == "target_nan":
        target[0] = np.nan
    elif invalid == "too_large":
        circuit.num_qubits = 7
    else:
        calibration = cast("CalibratedLocalDepolarizingProvider", object())
    with pytest.raises(ValueError, match="Rotosolve requires"):
        sweep(circuit, theta, target, calibration)


def test_multiple_sweeps_monotonically_improve_true_channel() -> None:
    """Rebuilding suffixes each sweep preserves exact noisy improvement."""
    circuit, theta, target = _problem(13)
    calibration = CalibratedLocalDepolarizingProvider()
    provider = ExactLocalPauliProvider(calibration)
    previous = exact_density_fidelity(circuit, theta, target, provider)
    for _ in range(4):
        theta, diagnostics = sweep(circuit, theta, target, calibration)
        observed = exact_density_fidelity(circuit, theta, target, provider)
        assert observed >= previous - 1e-12
        assert observed == pytest.approx(diagnostics["final_fidelity"], abs=1e-12)
        previous = observed
