# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Representation-independent Pauli sampling, replay validation and multiplicities."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from .gate_noise import (
    CompositeGateNoiseInstruction,
    GateNoiseContext,
    LocalOperator,
    RandomUnitaryInstruction,
    TJMNoiseInstruction,
    validate_gate_noise_instruction,
)
from .krotov import KrotovNoiseMap

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .gate_noise import GateNoiseProvider
    from .krotov import KrotovTJMOptions
    from .parameterized_circuit import ParameterizedCircuit

NoiseMaps: TypeAlias = Sequence[Sequence[KrotovNoiseMap]]
Estimator: TypeAlias = Literal["independent", "cross"]


def _instruction_operators(
    instruction: RandomUnitaryInstruction | TJMNoiseInstruction | CompositeGateNoiseInstruction | None,
    rng: np.random.Generator,
    dt: float,
) -> tuple[tuple[NDArray[np.complex128], tuple[int, ...]], ...]:
    """Resolve unitary provider instructions, including the legacy one-jump model.

    Returns:
        Ordered physical unitaries; scalar Pauli no-jump drift cancels on normalization.

    Raises:
        NotImplementedError: If a TJM process is not a Pauli unitary jump.
    """
    if instruction is None:
        return ()
    if isinstance(instruction, RandomUnitaryInstruction):
        return tuple((op.matrix, op.sites) for op in instruction.operators)
    if isinstance(instruction, CompositeGateNoiseInstruction):
        return tuple(op for child in instruction.instructions for op in _instruction_operators(child, rng, dt))
    processes = instruction.noise_model.processes
    strengths = np.asarray([float(process["strength"]) for process in processes])
    if not len(strengths) or np.all(np.isclose(strengths, 0.0)):
        return ()
    operators = []
    for process, strength in zip(processes, strengths, strict=True):
        if strength <= 0:
            operators.append(None)
            continue
        if not str(process["name"]).startswith(("pauli_", "crosstalk_", "longrange_crosstalk_")):
            msg = "Statevector TJM sampling supports only Pauli unitary jumps."
            raise NotImplementedError(msg)
        matrix = process["matrix"] if "matrix" in process else np.kron(process["factors"][0], process["factors"][1])
        local = LocalOperator(np.asarray(matrix, dtype=complex), tuple(process["sites"]))
        operators.append((local.matrix, local.sites))
    weights = np.maximum(strengths, 0.0)
    total = float(weights.sum())
    if rng.random() >= -np.expm1(-dt * total):
        return ()
    selected = operators[int(rng.choice(len(processes), p=weights / total))]
    assert selected is not None
    return (selected,)


class TrajectorySampler:
    """Sample and compress histories without allocating a quantum state.

    All probabilities and maps must be independent of the optimized parameters.
    The legacy trajectory seeds and operator order are preserved exactly.
    """

    def __init__(self, circuit: ParameterizedCircuit) -> None:
        """Copy a valid, data-free circuit.

        Raises:
            ValueError: If a gate depends on data or has invalid affine angles.
        """
        if circuit.num_qubits < 1 or any(
            gate.data_map is not None or not np.isfinite(gate.angle_scale) or not np.isfinite(gate.angle_offset)
            for gate in circuit.gates
        ):
            msg = "Preparation requires a nonempty, data-free qubit register."
            raise ValueError(msg)
        self.circuit = copy.deepcopy(circuit)

    def _parameters(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Validate and copy gate parameters.

        Returns:
            A private real parameter vector.

        Raises:
            ValueError: If parameters have the wrong shape or nonfinite/complex values.
        """
        value = np.asarray(theta)
        if value.shape != (self.circuit.num_params,) or not np.isrealobj(value) or not np.all(np.isfinite(value)):
            msg = "Expected finite real parameters matching circuit.num_params."
            raise ValueError(msg)
        return value.astype(np.float64, copy=True)

    def sample_noise_maps(
        self,
        theta: NDArray[np.float64],
        options: KrotovTJMOptions,
        provider: GateNoiseProvider,
        *,
        iteration: int = 0,
    ) -> tuple[tuple[KrotovNoiseMap, ...], ...]:
        """Sample state-independent maps with the production trajectory seed formula.

        Args:
            theta: Parameters used only to resolve provider gate metadata.
            options: Trajectory count, base seed, time window, and gate mask.
                ``use_crn`` is managed by the caller by retaining returned maps.
            provider: Parameter-independent random-unitary or Pauli-TJM provider.
            iteration: Nonnegative global noise counter; no hidden RNG state.

        Returns:
            One replay-map sequence per trajectory. Masks affect sampling only;
            subsequently supplied maps are replayed exactly as in the MPS backend.

        Raises:
            ValueError: If a counter, count, or seed is invalid.
        """
        theta = self._parameters(theta)
        if (
            type(iteration) is not int
            or iteration < 0
            or type(options.num_trajectories) is not int
            or options.num_trajectories < 1
            or type(options.random_seed) is not int
            or options.random_seed < 0
        ):
            msg = "Trajectory count and nonnegative seed/counter must be integers."
            raise ValueError(msg)
        contexts = []
        for k, gate in enumerate(self.circuit.gates):
            _, sites, angle = self.circuit.gate_matrix_and_angle(gate, theta)
            enabled = (
                gate.noise_enabled
                and (options.apply_noise_to == "all" or len(sites) == 2)
                and (options.noisy_gate_indices is None or k in options.noisy_gate_indices)
            )
            contexts.append(
                GateNoiseContext(
                    k,
                    gate.name,
                    sites,
                    len(sites),
                    angle,
                    gate.logical_gate_id if gate.logical_gate_id is not None else k,
                    gate.native_gate_id if gate.native_gate_id is not None else k,
                    gate.param_index,
                )
                if enabled
                else None
            )
        result = []
        for index in range(options.num_trajectories):
            rng = np.random.default_rng(options.random_seed + 1_000_003 * iteration + index)
            maps = []
            for context in contexts:
                if context is None:
                    maps.append(KrotovNoiseMap())
                    continue
                instruction = validate_gate_noise_instruction(provider(context, rng), context)
                operators = _instruction_operators(instruction, rng, options.dt)
                maps.append(
                    KrotovNoiseMap(
                        operators=operators,
                        source_gate_index=context.gate_index,
                        resolved_native_angle=context.resolved_angle,
                        channel_id=None if instruction is None else instruction.channel_id,
                        outcome_labels=instruction.outcome_labels
                        if isinstance(instruction, RandomUnitaryInstruction)
                        else (),
                    )
                )
            result.append(tuple(maps))
        return tuple(result)

    def _maps(self, maps: NoiseMaps | None) -> NoiseMaps:
        """Validate physical replay operators before any propagation.

        Returns:
            The validated ensemble, or one noiseless trajectory.

        Raises:
            ValueError: If maps are empty, malformed, nonunitary, or out of bounds.
            NotImplementedError: If a map requests nonlinear normalization.
        """
        if maps is None:
            return ((KrotovNoiseMap(),) * len(self.circuit.gates),)
        if not maps or any(len(row) != len(self.circuit.gates) for row in maps):
            msg = "Expected a nonempty ensemble with one map per circuit gate."
            raise ValueError(msg)
        for row in maps:
            for noise_map in row:
                if noise_map.normalized or noise_map.normalization_checkpoints:
                    msg = "Normalized nonunitary replay is unsupported by the statevector adjoint."
                    raise NotImplementedError(msg)
                for matrix, sites in noise_map.operators:
                    LocalOperator(matrix, sites)
                    if max(sites) >= self.circuit.num_qubits:
                        msg = "Noise operator site outside the statevector register."
                        raise ValueError(msg)
        return maps

    def compress_no_jump_maps(self, maps: NoiseMaps) -> tuple[NoiseMaps, NDArray[np.float64]]:
        """Merge empty operator histories with their empirical probability mass.

        Every input history has equal weight. Only histories with no operators
        at any gate are merged; jump histories, including cancelling jumps,
        remain separate. The representative occupies the first empty history's
        position. Recompute its forward and backward states at each update.

        Args:
            maps: Already sampled, parameter-independent unitary histories.

        Returns:
            Retained histories and multiplicity weights summing to one.
        """
        maps = self._maps(maps)
        retained = []
        counts = []
        representative = None
        for row in maps:
            empty = not any(noise_map.operators for noise_map in row)
            if empty and representative is not None:
                counts[representative] += 1
            else:
                if empty:
                    representative = len(retained)
                retained.append(tuple(row))
                counts.append(1)
        return tuple(retained), np.asarray(counts, dtype=np.float64) / len(maps)

    @staticmethod
    def _weights(weights: NDArray[np.float64] | None, count: int) -> NDArray[np.float64] | None:
        """Validate fixed probability weights without silently renormalizing them.

        Returns:
            A private probability vector, or None for the historical sample mean.

        Raises:
            ValueError: If weights are invalid.
        """
        if weights is None:
            return None
        value = np.asarray(weights)
        if (
            value.shape != (count,)
            or not np.isrealobj(value)
            or value.dtype.kind not in "fiu"
            or not np.all(np.isfinite(value))
            or np.any(value < 0)
            or abs(float(value.sum()) - 1) > 1e-12
        ):
            msg = "Trajectory weights must be finite nonnegative probabilities summing to one."
            raise ValueError(msg)
        return value.astype(np.float64, copy=True)
