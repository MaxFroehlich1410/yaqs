# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for frozen standard and Ballarin benchmark noise models."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest

from benchmarks.state_preparation import (
    NOISE_IDS,
    BallarinNoiseProvider,
    StandardNoiseProvider,
    ballarin_epsilon,
    ballarin_pauli_probability,
    canonicalize_angle,
    materialize_ballarin_circuit,
    standard_noise_profile,
)
from benchmarks.state_preparation.constants import MULTI_QUBIT_STRENGTH, ONE_QUBIT_STRENGTH
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import GateNoiseContext, ParameterizedCircuit, ParameterizedGate, TJMNoiseInstruction


def _context(arity: int, *, angle: float | None = None) -> GateNoiseContext:
    """Return a valid one- or two-qubit gate context."""
    sites = (0,) if arity == 1 else (0, 1)
    return GateNoiseContext(
        gate_index=0,
        gate_name="rz" if arity == 1 else "rzz",
        sites=sites,
        arity=arity,
        resolved_angle=angle,
        logical_gate_id=0,
        native_gate_id=0,
        parameter_index=None,
    )


def _process_names(provider: StandardNoiseProvider, arity: int) -> tuple[str, ...] | None:
    """Return process names generated for one gate arity."""
    instruction = provider(_context(arity), np.random.default_rng(0))
    if instruction is None:
        return None
    assert isinstance(instruction, TJMNoiseInstruction)
    model = instruction.noise_model
    expected_strength = ONE_QUBIT_STRENGTH if arity == 1 else MULTI_QUBIT_STRENGTH
    assert all(process["strength"] == expected_strength for process in model.processes)
    return tuple(cast("str", process["name"]) for process in model.processes)


@pytest.mark.parametrize(
    ("noise_id", "one_qubit", "two_qubit"),
    [
        ("dephasing_1s_1q", ("pauli_z",), None),
        ("dephasing_1s_2q", None, ("pauli_z", "pauli_z")),
        ("dephasing_1s_all", ("pauli_z",), ("pauli_z", "pauli_z")),
        ("dephasing_2s_2q", None, ("crosstalk_zz",)),
        ("dephasing_1s2s_all", ("pauli_z",), ("pauli_z", "pauli_z", "crosstalk_zz")),
        ("depolarizing_1s_1q", ("pauli_x", "pauli_y", "pauli_z"), None),
        (
            "depolarizing_1s_2q",
            None,
            ("pauli_x", "pauli_y", "pauli_z", "pauli_x", "pauli_y", "pauli_z"),
        ),
        (
            "depolarizing_1s_all",
            ("pauli_x", "pauli_y", "pauli_z"),
            ("pauli_x", "pauli_y", "pauli_z", "pauli_x", "pauli_y", "pauli_z"),
        ),
        (
            "depolarizing_2s_2q",
            None,
            tuple(f"crosstalk_{left}{right}" for left in "xyz" for right in "xyz"),
        ),
        (
            "depolarizing_1s2s_all",
            ("pauli_x", "pauli_y", "pauli_z"),
            (
                "pauli_x",
                "pauli_y",
                "pauli_z",
                "pauli_x",
                "pauli_y",
                "pauli_z",
                *(f"crosstalk_{left}{right}" for left in "xyz" for right in "xyz"),
            ),
        ),
    ],
)
def test_standard_profiles_have_exact_process_sets_and_placement(
    noise_id: str,
    one_qubit: tuple[str, ...] | None,
    two_qubit: tuple[str, ...] | None,
) -> None:
    """Every standard identifier resolves to its frozen per-operator model."""
    provider = StandardNoiseProvider(noise_id)

    assert _process_names(provider, 1) == one_qubit
    assert _process_names(provider, 2) == two_qubit
    assert provider.content_checksum == standard_noise_profile(noise_id).content_checksum


def test_standard_profile_identifiers_are_complete_and_reject_unknown_values() -> None:
    """The registry covers all ten non-Ballarin noisy identifiers."""
    assert len(NOISE_IDS) == 12
    with pytest.raises(ValueError, match="Unknown standard-noise"):
        StandardNoiseProvider("unknown")


def _circuit_state(circuit: ParameterizedCircuit, theta: np.ndarray) -> np.ndarray:
    """Return the dense state prepared by a small test circuit."""
    state = MPS(circuit.num_qubits)
    for gate in circuit.gates:
        matrix, sites = circuit.gate_matrix(gate, theta)
        state.apply_local_operator(matrix, sites, threshold=0.0)
    return state.to_vec()


def test_ballarin_angle_fit_and_canonicalization() -> None:
    """The implementation follows the frozen signed-angle convention."""
    assert canonicalize_angle(math.pi) == pytest.approx(-math.pi)
    assert canonicalize_angle(-math.pi) == pytest.approx(-math.pi)
    assert canonicalize_angle(5 * math.pi / 2) == pytest.approx(math.pi / 2)
    expected_epsilon = 2.1e-4 + 1.43e-3 * math.pi / 2
    assert ballarin_epsilon(5 * math.pi / 2) == pytest.approx(expected_epsilon)
    assert ballarin_pauli_probability(math.pi / 2) == pytest.approx(
        (1.0 - math.sqrt(1.0 - 5.0 * expected_epsilon / 4.0)) / 3.0
    )
    with pytest.raises(ValueError, match="finite"):
        canonicalize_angle(float("nan"))


def test_ballarin_materialization_is_unitarily_equivalent_up_to_global_phase() -> None:
    """RXX, RYY, and RZZ lowerings preserve the bound logical circuit."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("rxx", (0, 1), param_index=0),
            ParameterizedGate("ryy", (0, 1), param_index=1),
            ParameterizedGate("rzz", (0, 1), param_index=2),
            ParameterizedGate("rz", (0,), param_index=3),
        ],
        num_params=4,
    )
    theta = np.array([0.31, -0.47, 2 * math.pi + 0.22, 0.19])
    materialized = materialize_ballarin_circuit(circuit, theta)

    logical_state = _circuit_state(circuit, theta)
    native_state = _circuit_state(materialized.circuit, np.empty(0))
    assert abs(np.vdot(logical_state, native_state)) ** 2 == pytest.approx(1.0, abs=1e-12)
    assert materialized.pre_pruning_rzz_count == 3
    assert materialized.pruned_rzz_count == 0
    assert all(gate.name not in {"rxx", "ryy"} for gate in materialized.circuit.gates)
    assert all(not gate.noise_enabled for gate in materialized.circuit.gates if gate.name != "rzz")


def test_ballarin_pruning_removes_the_complete_basis_change_round_trip() -> None:
    """A near-zero RXX contributes no native gate or residual basis change."""
    circuit = ParameterizedCircuit(2, [ParameterizedGate("rxx", (0, 1), param_index=0)], num_params=1)

    materialized = materialize_ballarin_circuit(circuit, np.array([1e-4]))

    assert materialized.circuit.gates == []
    assert materialized.pruned_rzz_count == 1
    assert materialized.mapping[0].pruned
    assert materialized.mapping[0].native_gate_indices == ()


def test_ballarin_provider_matches_two_independent_categorical_draws() -> None:
    """The provider consumes exactly one independent local draw per gate qubit."""
    angle = 0.7
    probability = ballarin_pauli_probability(angle)
    probabilities = (1.0 - 3.0 * probability, probability, probability, probability)
    reference_rng = np.random.default_rng(91)
    expected = tuple("IXYZ"[int(reference_rng.choice(4, p=probabilities))] for _ in range(2))

    instruction = BallarinNoiseProvider()(_context(2, angle=angle), np.random.default_rng(91))

    assert instruction.outcome_labels == expected
    assert tuple(operator.label for operator in instruction.operators) == tuple(
        label for label in expected if label != "I"
    )


def test_ballarin_provider_rejects_non_native_context() -> None:
    """The exact categorical channel cannot be attached to an unresolved gate."""
    with pytest.raises(ValueError, match="resolved two-qubit RZZ"):
        BallarinNoiseProvider()(_context(1), np.random.default_rng(0))
