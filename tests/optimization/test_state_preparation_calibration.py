# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent channel checks for hardware-error-calibrated local noise."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit.quantum_info import Pauli, SuperOp, average_gate_fidelity
from scipy.linalg import expm

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.noise import StandardNoiseProvider
from mqt.yaqs.optimization import GateNoiseContext, TJMNoiseInstruction


@pytest.mark.parametrize("one_error", [0.0, 1e-10, 0.0001, 0.00035, 0.00064])
@pytest.mark.parametrize("two_error", [0.0, 1e-10, 0.001, 0.0033, 0.0051, 0.01])
def test_calibration_matches_dense_lindblad_channels(one_error: float, two_error: float) -> None:
    """Exponentiate independent dense generators and recover requested errors."""
    provider = CalibratedLocalDepolarizingProvider(one_error, two_error)
    for arity, target, strength, labels in (
        (1, one_error, provider.one_qubit_strength, ["X", "Y", "Z"]),
        (2, two_error, provider.two_qubit_strength, [f"{p}I" for p in "XYZ"] + [f"I{p}" for p in "XYZ"]),
    ):
        dim = 2**arity
        generator = np.zeros((dim**2, dim**2), dtype=complex)
        for label in labels:
            matrix = Pauli(label).to_matrix()
            generator += strength * (np.kron(matrix.conj(), matrix) - np.eye(dim**2))
        actual = 1 - average_gate_fidelity(SuperOp(expm(generator)))
        assert actual == pytest.approx(target, abs=1e-13)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), True])
def test_invalid_calibration_targets_are_rejected(bad: float) -> None:
    """Rates cannot be generated from nonphysical or Boolean error inputs."""
    with pytest.raises(ValueError, match="one_qubit_error"):
        CalibratedLocalDepolarizingProvider(one_qubit_error=bad)
    with pytest.raises(ValueError, match="two_qubit_error"):
        CalibratedLocalDepolarizingProvider(two_qubit_error=bad)


def test_finite_rate_limits_are_rejected() -> None:
    """Completely depolarized limits require infinite integrated rates."""
    with pytest.raises(ValueError, match="one_qubit_error"):
        CalibratedLocalDepolarizingProvider(one_qubit_error=0.5)
    with pytest.raises(ValueError, match="two_qubit_error"):
        CalibratedLocalDepolarizingProvider(two_qubit_error=0.75)


def test_provider_retains_placement_but_has_distinct_calibration_identity() -> None:
    """Only coefficients change; recorded finite-step errors remain explicit."""
    provider = CalibratedLocalDepolarizingProvider()
    legacy = StandardNoiseProvider("depolarizing_1s_all")
    assert provider.content_checksum != legacy.content_checksum
    assert provider.content_checksum == CalibratedLocalDepolarizingProvider().content_checksum
    assert provider.content_checksum != CalibratedLocalDepolarizingProvider(two_qubit_error=0.0033).content_checksum
    metadata = provider.scientific_dict()
    for sites in ((2,), (1, 4)):
        arity = len(sites)
        context = GateNoiseContext(0, "rz" if arity == 1 else "rzz", sites, arity, 0.1, 0, 0, None)
        instruction = provider(context, np.random.default_rng(0))
        legacy_instruction = legacy(context, np.random.default_rng(0))
        assert isinstance(legacy_instruction, TJMNoiseInstruction)
        assert instruction.channel_id != legacy_instruction.channel_id
        processes = instruction.noise_model.processes
        assert len(processes) == 3 * arity
        assert [p["sites"] for p in processes] == [p["sites"] for p in legacy_instruction.noise_model.processes]
        assert [p["name"] for p in processes] == [p["name"] for p in legacy_instruction.noise_model.processes]
        strength = provider.one_qubit_strength if arity == 1 else provider.two_qubit_strength
        assert all(p["strength"] == strength for p in processes)
        error = -(2**arity) / (2**arity + 1) * math.expm1(-len(processes) * strength)
        key = "one_jump_one_qubit_error" if arity == 1 else "one_jump_two_qubit_error"
        assert metadata[key] == pytest.approx(error, abs=1e-15)


def test_reference_coefficients_match_benchmark_derivation() -> None:
    """The documented reference pair is not copied into every generator."""
    provider = CalibratedLocalDepolarizingProvider()
    assert provider.one_qubit_strength == pytest.approx(3.2020497493061084e-4, abs=1e-15)
    assert provider.two_qubit_strength == pytest.approx(1.0664702651014474e-3, abs=1e-15)
