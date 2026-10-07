# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Small-system statevector adjoints for state preparation and unitary trajectories.

Vectors use little-endian basis indices and local matrices use ascending-site
tensor factors, as in ParameterizedCircuit. No full-system matrices or MPS are
constructed. Sampling, gradient estimation, and parameter updates are separate:
callers can replay exactly the same maps for CRN, refresh, or step acceptance.
Noise distributions must be independent of the state and parameters. General
normalized nonunitary jumps and derivatives of noise probabilities are unsupported.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from .trajectories import Estimator, NoiseMaps, TrajectorySampler

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from .parameterized_circuit import ParameterizedCircuit, ParameterizedGate


def _apply(
    states: NDArray[np.complex128], operator: NDArray[np.complex128], sites: tuple[int, ...], qubits: int
) -> NDArray[np.complex128]:
    """Apply a local matrix to a batch of row statevectors.

    Returns:
        A new batch, without constructing an embedded full-system operator.
    """
    axes = tuple(1 + qubits - 1 - site for site in sites)
    destination = tuple(range(1, 1 + len(sites)))
    tensor = np.moveaxis(states.reshape((len(states),) + (2,) * qubits), axes, destination)
    transformed = np.einsum("ab,rbc->rac", operator, tensor.reshape(len(states), 2 ** len(sites), -1))
    return np.moveaxis(transformed.reshape(tensor.shape), destination, axes).reshape(states.shape)


class StatevectorBackend(TrajectorySampler):
    """Shared dense trajectory engine for a fixed state-preparation circuit.

    Args:
        circuit: Data-free one-/two-qubit gate circuit, copied defensively.
        target: Normalized finite target vector in little-endian order.
        initial_state: Normalized input vector; defaults to the all-zero state.

    Raises:
        ValueError: If a vector or circuit is invalid or data-dependent.

    Notes:
        The independent batch signal differentiates the sample mean infidelity.
        Cross mode reproduces the existing R-by-R trajectory expansion; it is
        not asserted to differentiate that same finite-sample loss. Online mode
        holds old adjoints fixed while updating each gate's forward prefix.
        Optional fixed probability weights apply to independent metrics,
        gradients and every online gate update. They must sum to one and must
        not depend on parameters; no probability derivatives are included.
        Cross mode weights both trajectory indices, preserving explicit path
        replication without claiming a gradient of the finite-sample loss.
    """

    def __init__(
        self,
        circuit: ParameterizedCircuit,
        target: NDArray[np.complex128],
        initial_state: NDArray[np.complex128] | None = None,
    ) -> None:
        """Create a private circuit and normalized input/target vectors.

        Raises:
            ValueError: If the circuit has invalid angles or depends on input data.
        """
        if circuit.num_qubits < 1 or any(
            gate.data_map is not None or not np.isfinite(gate.angle_scale) or not np.isfinite(gate.angle_offset)
            for gate in circuit.gates
        ):
            msg = "Statevector preparation requires a nonempty, data-free qubit register."
            raise ValueError(msg)
        self.circuit = copy.deepcopy(circuit)
        self.target = self._vector(target)
        if initial_state is None:
            initial_state = np.zeros(2**circuit.num_qubits, dtype=complex)
            initial_state[0] = 1
        self.initial = self._vector(initial_state)

    def _vector(self, value: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Validate and copy a normalized vector.

        Returns:
            A private complex128 vector.

        Raises:
            ValueError: If dimension, finiteness, or normalization is invalid.
        """
        vector = np.asarray(value, dtype=np.complex128)
        if (
            vector.shape != (2**self.circuit.num_qubits,)
            or not np.all(np.isfinite(vector))
            or abs(np.vdot(vector, vector) - 1) > 1e-10
        ):
            msg = "Expected a finite normalized statevector of dimension 2**num_qubits."
            raise ValueError(msg)
        return vector.copy()

    def _noise(
        self, states: NDArray[np.complex128], maps: NoiseMaps, gate: int, *, adjoint: bool = False
    ) -> NDArray[np.complex128]:
        """Apply each trajectory's local map or its adjoint.

        Returns:
            A new batch of states/costates.
        """
        result = states.copy()
        for r, row in enumerate(maps):
            operators = reversed(row[gate].operators) if adjoint else row[gate].operators
            for matrix, sites in operators:
                result[r : r + 1] = _apply(
                    result[r : r + 1], matrix.conj().T if adjoint else matrix, sites, self.circuit.num_qubits
                )
        return result

    def _forward(
        self, theta: NDArray[np.float64], maps: NoiseMaps
    ) -> tuple[list[NDArray[np.complex128]], NDArray[np.complex128]]:
        """Propagate the ensemble and retain pre-noise gate outputs.

        Returns:
            Gate outputs and final states, each batched over trajectories.
        """
        states = np.repeat(self.initial[None, :], len(maps), axis=0)
        outputs = []
        for k, gate in enumerate(self.circuit.gates):
            unitary, sites = self.circuit.gate_matrix(gate, theta)
            states = _apply(states, unitary, sites, self.circuit.num_qubits)
            outputs.append(states)
            states = self._noise(states, maps, k)
        return outputs, states

    def _costates(
        self,
        theta: NDArray[np.float64],
        maps: NoiseMaps,
        final: NDArray[np.complex128],
        estimator: Estimator,
        weights: NDArray[np.float64] | None = None,
    ) -> list[NDArray[np.complex128]]:
        """Propagate old-parameter adjoints, applying independent weights exactly once.

        Returns:
            Costates after noise pullback at each gate.

        Raises:
            ValueError: If the requested estimator is unknown.
        """
        if estimator not in {"independent", "cross"}:
            msg = "Unknown trajectory estimator."
            raise ValueError(msg)
        chi = np.repeat(self.target[None, :], len(maps), axis=0)
        if estimator == "independent":
            overlaps = -(final @ self.target.conj())[:, None]
            chi *= overlaps / len(maps) if weights is None else overlaps * weights[:, None]
        backwards = []
        for k in reversed(range(len(self.circuit.gates))):
            chi = self._noise(chi, maps, k, adjoint=True)
            backwards.append(chi)
            unitary, sites = self.circuit.gate_matrix(self.circuit.gates[k], theta)
            chi = _apply(chi, unitary.conj().T, sites, self.circuit.num_qubits)
        return list(reversed(backwards))

    def _signal(
        self,
        gate: ParameterizedGate,
        outputs: NDArray[np.complex128],
        chi: NDArray[np.complex128],
        estimator: Estimator,
        weights: NDArray[np.float64] | None = None,
    ) -> float:
        """Average trajectory contributions before a parameter update.

        Returns:
            The independent loss-gradient or cross-expansion scalar signal.
        """
        derivative, sites = self.circuit.derivative_operator(gate)
        tangent = _apply(outputs, derivative, sites, self.circuit.num_qubits)
        if estimator == "independent":
            return gate.angle_scale * 2 * float(np.vdot(chi, tangent).real)
        products = (chi.conj() @ tangent.T) * (outputs.conj() @ chi.T).T
        if weights is not None:
            return -gate.angle_scale * 2 * float((products * weights[:, None] * weights[None, :]).sum().real)
        return -gate.angle_scale * 2 * float(products.sum().real) / len(outputs) ** 2

    def metrics(
        self,
        theta: NDArray[np.float64],
        maps: NoiseMaps | None = None,
        *,
        weights: NDArray[np.float64] | None = None,
    ) -> tuple[float, float]:
        """Evaluate a sample-average loss and fidelity, or the noiseless objective.

        Returns:
            Infidelity and mean individual-trajectory fidelity.
        """
        maps = self._maps(maps)
        weights = self._weights(weights, len(maps))
        _, final = self._forward(self._parameters(theta), maps)
        fidelities = np.abs(final @ self.target.conj()) ** 2
        fidelity = float(np.mean(fidelities) if weights is None else weights @ fidelities)
        return 1 - fidelity, fidelity

    def gradient(
        self,
        theta: NDArray[np.float64],
        maps: NoiseMaps | None = None,
        *,
        estimator: Estimator = "independent",
        weights: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], float, float]:
        """Compute the batch signal without changing parameters, maps, or targets.

        Returns:
            Signal, infidelity, and mean fidelity at the supplied parameters.
        """
        theta, maps = self._parameters(theta), self._maps(maps)
        weights = self._weights(weights, len(maps))
        outputs, final = self._forward(theta, maps)
        costates = self._costates(theta, maps, final, estimator, weights)
        signal = np.zeros_like(theta)
        for gate, output, chi in zip(self.circuit.gates, outputs, costates, strict=True):
            if gate.param_index is not None:
                signal[gate.param_index] += self._signal(gate, output, chi, estimator, weights)
        fidelities = np.abs(final @ self.target.conj()) ** 2
        fidelity = float(np.mean(fidelities) if weights is None else weights @ fidelities)
        return signal, 1 - fidelity, fidelity

    def deletion_fidelities(self, theta: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
        """Score complete noiseless gate removals with one prefix/suffix pass.

        Returns:
            Individual deleted-circuit fidelities and the original fidelity.
        """
        theta = self._parameters(theta)
        state = self.initial[None, :]
        tape = []
        for gate in self.circuit.gates:
            matrix, sites = self.circuit.gate_matrix(gate, theta)
            tape.append((state, matrix, sites))
            state = _apply(state, matrix, sites, self.circuit.num_qubits)
        fidelity = float(abs(np.vdot(self.target, state[0])) ** 2)
        suffix = self.target[None, :]
        removed = np.empty(len(tape))
        for k in reversed(range(len(tape))):
            prefix, matrix, sites = tape[k]
            removed[k] = abs(np.vdot(suffix, prefix)) ** 2
            suffix = _apply(suffix, matrix.conj().T, sites, self.circuit.num_qubits)
        return removed, fidelity

    def online_update(
        self,
        theta: NDArray[np.float64],
        step_size: float,
        maps: NoiseMaps | None = None,
        *,
        estimator: Estimator = "independent",
        weights: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Perform one gate-sequential stale-adjoint sweep with ensemble averaging.

        Returns:
            New parameters and accumulated update signals. Shared parameters
            are updated at each occurrence, matching the existing MPS method.

        Raises:
            ValueError: If the step size is negative or nonfinite.
        """
        if isinstance(step_size, bool) or not np.isfinite(step_size) or step_size < 0:
            msg = "Step size must be finite and nonnegative."
            raise ValueError(msg)
        return self._online_sweep(theta, maps, estimator, weights, lambda _index, local: step_size * local)

    def adaptive_online_update(
        self,
        theta: NDArray[np.float64],
        step_size: float,
        second_moment: NDArray[np.float64],
        counts: NDArray[np.int64],
        maps: NoiseMaps | None = None,
        *,
        weights: NDArray[np.float64] | None = None,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        """Scale every independent online gate signal by its running second moment.

        For each parameter occurrence: v <- beta2*v + (1-beta2)*signal**2,
        count <- count+1, then delta <- rate*signal/(sqrt(v/(1-beta2**count))+eps).
        The signal is summed over weighted trajectories before updating moments.
        Old adjoints are retained, but every forward prefix uses updated gates.
        Counts are per occurrence, so shared parameters have correct bias correction.
        No first moment, extra batch-gradient pass or loss-based acceptance is used.

        Args:
            theta: Current circuit parameters.
            step_size: Nonnegative adaptive base rate.
            second_moment: Nonnegative per-parameter uncorrected squared-signal EMA.
            counts: Nonnegative integer occurrence counters, initially zero.
            maps: Physical replay ensemble, or the deterministic noiseless path.
            weights: Fixed trajectory probabilities; default is the sample mean.
            beta2: Squared-signal EMA decay in [0, 1).
            epsilon: Positive stabilizer outside the square root.

        Returns:
            New parameters, accumulated raw signals, moments and occurrence counts.
            Caller-owned arrays are never modified.

        Raises:
            ValueError: If rates, moments, counts or decay/stabilizer are invalid.
        """
        if (
            any(isinstance(x, bool) or not np.isfinite(x) for x in (step_size, beta2, epsilon))
            or step_size < 0
            or not 0 <= beta2 < 1
            or epsilon <= 0
        ):
            msg = "Invalid adaptive online rate, decay or stabilizer."
            raise ValueError(msg)
        second = self._parameters(second_moment)
        counter = np.asarray(counts)
        if (
            np.any(second < 0)
            or counter.shape != second.shape
            or counter.dtype.kind not in "iu"
            or np.any(counter < 0)
            or np.any(counter > np.iinfo(np.int64).max - len(self.circuit.gates) - 1)
        ):
            msg = "Expected nonnegative adaptive moments and integer occurrence counts."
            raise ValueError(msg)
        counter = counter.astype(np.int64, copy=True)

        def displacement(index: int, local: float) -> float:
            """Update the local second moment.

            Returns:
                The bias-corrected parameter displacement.
            """
            second[index] = beta2 * second[index] + (1 - beta2) * local**2
            counter[index] += 1
            return float(step_size * local / (np.sqrt(second[index] / (1 - beta2 ** counter[index])) + epsilon))

        updated, signal = self._online_sweep(theta, maps, "independent", weights, displacement)
        return updated, signal, second, counter

    def _online_sweep(
        self,
        theta: NDArray[np.float64],
        maps: NoiseMaps | None,
        estimator: Estimator,
        weights: NDArray[np.float64] | None,
        displacement: Callable[[int, float], float],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Share forward/costate replay and updated prefixes across online update rules.

        Returns:
            Updated parameters and summed raw local signals.
        """
        theta, maps = self._parameters(theta), self._maps(maps)
        weights = self._weights(weights, len(maps))
        _, final = self._forward(theta, maps)
        costates = self._costates(theta, maps, final, estimator, weights)
        updated = theta.copy()
        signal = np.zeros_like(theta)
        current = np.repeat(self.initial[None, :], len(maps), axis=0)
        for k, gate in enumerate(self.circuit.gates):
            unitary, sites = self.circuit.gate_matrix(gate, updated)
            output = _apply(current, unitary, sites, self.circuit.num_qubits)
            if gate.param_index is not None:
                local = self._signal(gate, output, costates[k], estimator, weights)
                signal[gate.param_index] += local
                updated[gate.param_index] -= displacement(gate.param_index, local)
                unitary, sites = self.circuit.gate_matrix(gate, updated)
                output = _apply(current, unitary, sites, self.circuit.num_qubits)
            current = self._noise(output, maps, k)
        return updated, signal
