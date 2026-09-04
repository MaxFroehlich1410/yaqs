# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact Ballarin/Quantinuum channel and native circuit materialization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs.optimization import (
    GateNoiseContext,
    LocalOperator,
    ParameterizedCircuit,
    ParameterizedGate,
    RandomUnitaryInstruction,
)

from .constants import BALLARIN_NOISE_ID

if TYPE_CHECKING:
    from numpy.typing import NDArray

RZZ_PRUNING_THRESHOLD = 1e-4
_PRUNING_COMPARISON_ATOL = 1e-12
_PAULI_LABELS = ("I", "X", "Y", "Z")
_PAULIS = (
    np.eye(2, dtype=np.complex128),
    np.array([[0, 1], [1, 0]], dtype=np.complex128),
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    np.array([[1, 0], [0, -1]], dtype=np.complex128),
)


def canonicalize_angle(angle: float) -> float:
    """Return a finite rotation angle in the half-open interval [-pi, pi).

    Raises:
        ValueError: If the input angle is not finite.
    """
    value = float(angle)
    if not math.isfinite(value):
        msg = f"Rotation angle must be finite, got {angle!r}."
        raise ValueError(msg)
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def ballarin_epsilon(angle: float) -> float:
    """Return the fitted native-gate error for an angle in radians."""
    return 2.1e-4 + 1.43e-3 * abs(canonicalize_angle(angle))


def ballarin_pauli_probability(angle: float) -> float:
    """Return the probability of each non-identity local Pauli branch.

    Raises:
        ValueError: If the fitted error lies outside the probability domain.
    """
    epsilon = ballarin_epsilon(angle)
    radicand = 1.0 - 5.0 * epsilon / 4.0
    if radicand < 0.0:
        msg = f"Ballarin error fit is outside its probability domain for angle {angle!r}."
        raise ValueError(msg)
    return (1.0 - math.sqrt(radicand)) / 3.0


@dataclass(frozen=True, slots=True)
class NativeGateMapping:
    """Provenance for one logical gate after native materialization."""

    logical_gate_index: int
    native_gate_indices: tuple[int, ...]
    original_angle: float | None
    canonical_angle: float | None
    pruned: bool


@dataclass(frozen=True, slots=True)
class BallarinCircuitMaterialization:
    """A bound native circuit and its pruning provenance."""

    circuit: ParameterizedCircuit
    mapping: tuple[NativeGateMapping, ...]
    pre_pruning_rzz_count: int
    pruned_rzz_count: int


def _native_gate(
    name: str,
    sites: tuple[int, ...],
    *,
    logical_gate_id: int | str,
    native_gate_id: int,
    angle: float | None = None,
    fixed_params: tuple[float, ...] = (),
    noise_enabled: bool = False,
) -> ParameterizedGate:
    """Return one bound native gate."""
    return ParameterizedGate(
        name=name,
        sites=sites,
        angle_offset=0.0 if angle is None else angle,
        fixed_params=fixed_params,
        logical_gate_id=logical_gate_id,
        native_gate_id=native_gate_id,
        noise_enabled=noise_enabled,
    )


def _append_gate(
    gates: list[ParameterizedGate],
    name: str,
    sites: tuple[int, ...],
    *,
    logical_gate_id: int | str,
    angle: float | None = None,
    fixed_params: tuple[float, ...] = (),
    noise_enabled: bool = False,
) -> int:
    """Return the stable index of one newly appended native gate."""
    index = len(gates)
    gates.append(
        _native_gate(
            name,
            sites,
            logical_gate_id=logical_gate_id,
            native_gate_id=index,
            angle=angle,
            fixed_params=fixed_params,
            noise_enabled=noise_enabled,
        )
    )
    return index


def _append_rxx_lowering(
    gates: list[ParameterizedGate],
    sites: tuple[int, int],
    angle: float,
    logical_gate_id: int | str,
) -> tuple[int, ...]:
    """Return native indices after appending H-H-RZZ-H-H for RXX."""
    first, second = sites
    return (
        _append_gate(gates, "h", (first,), logical_gate_id=logical_gate_id),
        _append_gate(gates, "h", (second,), logical_gate_id=logical_gate_id),
        _append_gate(
            gates,
            "rzz",
            sites,
            logical_gate_id=logical_gate_id,
            angle=angle,
            noise_enabled=True,
        ),
        _append_gate(gates, "h", (second,), logical_gate_id=logical_gate_id),
        _append_gate(gates, "h", (first,), logical_gate_id=logical_gate_id),
    )


def _append_ryy_lowering(
    gates: list[ParameterizedGate],
    sites: tuple[int, int],
    angle: float,
    logical_gate_id: int | str,
) -> tuple[int, ...]:
    """Return native indices after appending the RX lowering of RYY."""
    first, second = sites
    return (
        _append_gate(gates, "rx", (first,), logical_gate_id=logical_gate_id, angle=math.pi / 2),
        _append_gate(gates, "rx", (second,), logical_gate_id=logical_gate_id, angle=math.pi / 2),
        _append_gate(
            gates,
            "rzz",
            sites,
            logical_gate_id=logical_gate_id,
            angle=angle,
            noise_enabled=True,
        ),
        _append_gate(gates, "rx", (second,), logical_gate_id=logical_gate_id, angle=-math.pi / 2),
        _append_gate(gates, "rx", (first,), logical_gate_id=logical_gate_id, angle=-math.pi / 2),
    )


def materialize_ballarin_circuit(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
) -> BallarinCircuitMaterialization:
    """Bind, lower, canonicalize, and prune a logical benchmark circuit.

    One-qubit gates are bound and retained as noiseless operations. RXX and RYY
    rotations are expressed through exact noiseless basis changes around one
    RZZ. Native RZZ angles are canonicalized to [-pi, pi); rotations at or
    below the benchmark threshold are removed together with their basis-change
    round trip.

    Returns:
        Bound native circuit and per-logical-gate provenance.

    Raises:
        ValueError: If a logical two-qubit gate has no benchmark lowering.
    """
    values = np.asarray(theta, dtype=np.float64)
    if values.shape != (circuit.num_params,):
        msg = f"Expected {circuit.num_params} parameters, got shape {values.shape}."
        raise ValueError(msg)
    native_gates: list[ParameterizedGate] = []
    mappings: list[NativeGateMapping] = []
    pre_pruning = 0
    pruned = 0

    for logical_index, gate in enumerate(circuit.gates):
        logical_id = gate.logical_gate_id if gate.logical_gate_id is not None else logical_index
        matrix, sorted_sites, resolved_angle = circuit.gate_matrix_and_angle(gate, values)
        del matrix
        if len(sorted_sites) == 1:
            index = _append_gate(
                native_gates,
                gate.name,
                sorted_sites,
                logical_gate_id=logical_id,
                angle=resolved_angle,
                fixed_params=tuple(gate.fixed_params),
            )
            mappings.append(
                NativeGateMapping(
                    logical_gate_index=logical_index,
                    native_gate_indices=(index,),
                    original_angle=resolved_angle,
                    canonical_angle=resolved_angle,
                    pruned=False,
                )
            )
            continue

        if gate.name not in {"rxx", "ryy", "rzz"} or resolved_angle is None:
            msg = f"Two-qubit gate {gate.name!r} has no exact Ballarin RZZ lowering."
            raise ValueError(msg)
        pre_pruning += 1
        angle = canonicalize_angle(resolved_angle)
        if abs(angle) <= RZZ_PRUNING_THRESHOLD or math.isclose(
            abs(angle),
            RZZ_PRUNING_THRESHOLD,
            rel_tol=0.0,
            abs_tol=_PRUNING_COMPARISON_ATOL,
        ):
            pruned += 1
            mappings.append(
                NativeGateMapping(
                    logical_gate_index=logical_index,
                    native_gate_indices=(),
                    original_angle=resolved_angle,
                    canonical_angle=angle,
                    pruned=True,
                )
            )
            continue
        sites = (sorted_sites[0], sorted_sites[1])
        if gate.name == "rxx":
            indices = _append_rxx_lowering(native_gates, sites, angle, logical_id)
        elif gate.name == "ryy":
            indices = _append_ryy_lowering(native_gates, sites, angle, logical_id)
        else:
            indices = (
                _append_gate(
                    native_gates,
                    "rzz",
                    sites,
                    logical_gate_id=logical_id,
                    angle=angle,
                    noise_enabled=True,
                ),
            )
        mappings.append(
            NativeGateMapping(
                logical_gate_index=logical_index,
                native_gate_indices=indices,
                original_angle=resolved_angle,
                canonical_angle=angle,
                pruned=False,
            )
        )

    return BallarinCircuitMaterialization(
        circuit=ParameterizedCircuit(circuit.num_qubits, native_gates, num_params=0),
        mapping=tuple(mappings),
        pre_pruning_rzz_count=pre_pruning,
        pruned_rzz_count=pruned,
    )


@dataclass(frozen=True, slots=True)
class BallarinNoiseProvider:
    """Sample the exact product of two local Ballarin Pauli channels."""

    noise_id: str = field(default=BALLARIN_NOISE_ID, init=False)

    def __call__(self, context: GateNoiseContext, rng: np.random.Generator, /) -> RandomUnitaryInstruction:
        """Return one independent local Pauli branch on each RZZ qubit.

        Raises:
            ValueError: If the context is not a resolved two-qubit RZZ gate.
        """
        if context.gate_name != "rzz" or context.arity != 2 or context.resolved_angle is None:
            msg = "Ballarin noise can only be applied after a resolved two-qubit RZZ gate."
            raise ValueError(msg)
        probability = ballarin_pauli_probability(context.resolved_angle)
        probabilities = (1.0 - 3.0 * probability, probability, probability, probability)
        sampled = tuple(int(rng.choice(4, p=probabilities)) for _site in context.sites)
        operators = tuple(
            LocalOperator(_PAULIS[index], (site,), label=_PAULI_LABELS[index])
            for site, index in zip(context.sites, sampled, strict=True)
            if index != 0
        )
        labels = tuple(_PAULI_LABELS[index] for index in sampled)
        return RandomUnitaryInstruction(operators, channel_id=self.noise_id, outcome_labels=labels)


__all__ = [
    "RZZ_PRUNING_THRESHOLD",
    "BallarinCircuitMaterialization",
    "BallarinNoiseProvider",
    "NativeGateMapping",
    "ballarin_epsilon",
    "ballarin_pauli_probability",
    "canonicalize_angle",
    "materialize_ballarin_circuit",
]
