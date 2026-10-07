# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact twelve-qubit objectives with logarithmic density checkpoint storage.

Only execution changes: recursive recomputation replaces the gate-by-gate tape.
The original channel, analytic derivatives, optimizers and iterate policies stay
unchanged. No inverse noise channel or stochastic gradient approximation is used.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit.quantum_info import Operator

from benchmarks.state_preparation.exact_density import _conjugate, _left_apply  # noqa: PLC2701  # Reuse tested kernels.
from experiments.state_preparation_competitors.evaluation import pauli_distribution
from experiments.state_preparation_competitors.noisy_core import DenseObjective, _evaluate, real_parameters
from mqt.yaqs.optimization.statevector import _apply  # noqa: PLC2701  # Reuse tested kernels.

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter


class CheckpointedDenseObjective(DenseObjective):
    """Exact mixed-state adjoint and pure-state VJP, bounded to twelve qubits."""

    def __init__(
        self, template: QuantumCircuit, target: ArrayLike, hardware: str | None = "IBM", *, compile_native: bool = True
    ) -> None:
        """Reuse the original schedule validation and local derivative definitions."""
        super().__init__(template, target, hardware, compile_native=compile_native, max_qubits=12)
        self.last_checkpoint_count = 0
        self.last_forward_gate_count = 0
        self._local_channels = {}
        if hardware is not None:
            for arity in (1, 2):
                probabilities, paulis = pauli_distribution(hardware, arity)
                self._local_channels[arity] = sum(
                    weight * np.kron(pauli, pauli.conj()) for weight, pauli in zip(probabilities, paulis, strict=True)
                )

    def _channel(
        self, matrix: NDArray[np.complex128], sites: tuple[int, ...], *, noisy: bool
    ) -> NDArray[np.complex128]:
        """Apply the same Pauli mixture in one small local superoperator contraction."""
        if not noisy or self.hardware is None:
            return matrix
        # Row-major vectorization puts all ket bits before all bra bits.
        support = tuple(self.num_qubits + site for site in sites) + sites
        return _apply(matrix.reshape(1, -1), self._local_channels[len(sites)], support, 2 * self.num_qubits).reshape(
            matrix.shape
        )

    def _schedule(self, theta: ArrayLike) -> tuple[list[NDArray[np.complex128]], dict[Parameter, float]]:
        """Bind small local matrices once per objective evaluation."""
        values = real_parameters(theta, len(self.parameters))
        bindings = dict(zip(self.parameters, map(float, values), strict=True))
        return [Operator(item.operation).data for item in self.bind(values).data], bindings

    def _pure_forward(self, gates: list[NDArray[np.complex128]]) -> list[NDArray[np.complex128]]:
        """Keep inexpensive statevectors for the noiseless reverse pass."""
        state = np.zeros(self.target.size, dtype=np.complex128)
        state[0] = 1
        states = [state]
        for window, gate in zip(self.windows, gates, strict=True):
            state = _apply(state[None, :], gate, window.sites, self.num_qubits)[0]
            states.append(state)
        return states

    def _pure_adjoint(
        self,
        gates: list[NDArray[np.complex128]],
        bindings: dict[Parameter, float],
        states: list[NDArray[np.complex128]],
        costate: NDArray[np.complex128],
    ) -> NDArray[np.float64]:
        """Differentiate a real pure-state objective with its ket costate."""
        gradient = np.zeros(len(self.parameters))
        for index in reversed(range(len(gates))):
            window = self.windows[index]
            if window.generator is not None:
                tangent = _apply(states[index + 1][None, :], -0.5j * window.generator, window.sites, self.num_qubits)[0]
                derivative = 2 * float(np.vdot(costate, tangent).real)
                for parameter, expression in window.derivatives:
                    gradient[parameter] += _evaluate(expression, bindings) * derivative
            costate = _apply(costate[None, :], gates[index].conj().T, window.sites, self.num_qubits)[0]
        return gradient

    def _advance(
        self, rho: NDArray[np.complex128], gates: list[NDArray[np.complex128]], start: int, end: int
    ) -> NDArray[np.complex128]:
        """Recompute a density interval without retaining its intermediates."""
        for index in range(start, end):
            window = self.windows[index]
            rho = self._channel(_conjugate(rho, gates[index], window.sites, self.num_qubits), window.sites, noisy=True)
            self.last_forward_gate_count += 1
        return rho

    def density(self, theta: ArrayLike, *, noisy: bool = True) -> NDArray[np.complex128]:
        """Propagate the exact density with no density tape."""
        gates, _ = self._schedule(theta)
        if not noisy or self.hardware is None:
            vector = self._pure_forward(gates)[-1]
            return np.outer(vector, vector.conj())
        rho = np.zeros((self.target.size, self.target.size), dtype=np.complex128)
        rho[0, 0] = 1
        self.last_forward_gate_count = 0
        rho = self._advance(rho, gates, 0, len(gates))
        if not np.isfinite(rho).all() or abs(np.trace(rho) - 1) > 1e-9:
            msg = "Exact density propagation failed finiteness or trace checks."
            raise FloatingPointError(msg)
        return rho

    def loss_gradient(self, theta: ArrayLike) -> tuple[float, NDArray[np.float64]]:
        """Return the original exact objective using a recursive checkpoint tree."""
        gates, bindings = self._schedule(theta)
        self.last_checkpoint_count = 0
        self.last_forward_gate_count = 0
        if self.hardware is None:
            states = self._pure_forward(gates)
            overlap = np.vdot(self.target, states[-1])
            return 1 - float(abs(overlap) ** 2), self._pure_adjoint(gates, bindings, states, -overlap * self.target)
        initial = np.zeros((self.target.size, self.target.size), dtype=np.complex128)
        initial[0, 0] = 1
        final = self._advance(initial, gates, 0, len(gates))
        loss = 1 - float(np.vdot(self.target, final @ self.target).real)
        del final
        gradient = np.zeros(len(self.parameters))

        def reverse(
            start: int, end: int, rho: NDArray[np.complex128], costate: NDArray[np.complex128], level: int
        ) -> NDArray[np.complex128]:
            self.last_checkpoint_count = max(self.last_checkpoint_count, level)
            if start == end:
                return costate
            if end - start > 1:
                middle = (start + end) // 2
                checkpoint = self._advance(rho, gates, start, middle)
                # Do not keep the incoming costate alive in the parent frame.
                outgoing = reverse(middle, end, checkpoint, costate, level + 1)
                del checkpoint, costate
                return reverse(start, middle, rho, outgoing, level)
            window = self.windows[start]
            costate = self._channel(costate, window.sites, noisy=True)
            if window.generator is not None:
                rotated = _conjugate(rho, gates[start], window.sites, self.num_qubits)
                tangent = _left_apply(rotated, -0.5j * window.generator, window.sites, self.num_qubits)
                derivative = 2 * float(np.vdot(costate, tangent).real)
                for parameter, expression in window.derivatives:
                    gradient[parameter] += _evaluate(expression, bindings) * derivative
            return _conjugate(costate, gates[start].conj().T, window.sites, self.num_qubits)

        reverse(0, len(gates), initial, -np.outer(self.target, self.target.conj()), 1)
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            msg = "Checkpointed objective or gradient is nonfinite."
            raise FloatingPointError(msg)
        return loss, gradient

    def density_vjp(self, theta: ArrayLike, costate: ArrayLike) -> NDArray[np.float64]:
        """Compute the exact ideal-density VJP using only ket propagation."""
        matrix = np.asarray(costate, dtype=np.complex128)
        if matrix.shape != (self.target.size, self.target.size) or not np.isfinite(matrix).all():
            msg = "Costate must be a finite Hermitian matrix matching the target."
            raise ValueError(msg)
        # Chunk the check to avoid another full matrix temporary at twelve qubits.
        for start in range(0, len(matrix), 64):
            if not np.allclose(matrix[start : start + 64], matrix[:, start : start + 64].conj().T, atol=1e-10, rtol=0):
                msg = "Costate must be a finite Hermitian matrix matching the target."
                raise ValueError(msg)
        gates, bindings = self._schedule(theta)
        states = self._pure_forward(gates)
        return self._pure_adjoint(gates, bindings, states, matrix @ states[-1])

    def diagnostics(self) -> dict[str, Any]:
        """Report storage accounting; the estimate is not an operating-system limit."""
        levels = math.ceil(math.log2(max(1, len(self.windows)))) + 1
        return {
            "backend": "exact_recursive_density_checkpoints_with_statevector_ideal_vjp",
            "num_qubits": self.num_qubits,
            "windows": len(self.windows),
            "density_bytes": self.target.size**2 * 16,
            "checkpoint_depth_bound": levels,
            "last_checkpoint_count": self.last_checkpoint_count,
            "last_forward_gate_count": self.last_forward_gate_count,
            "gradient_approximation": False,
        }


def make_objective(
    template: QuantumCircuit, target: ArrayLike, hardware: str | None = "IBM", *, compile_native: bool = True
) -> DenseObjective:
    """Preserve the six-qubit engine and opt larger circuits into bounded storage."""
    backend = DenseObjective if template.num_qubits <= 6 else CheckpointedDenseObjective
    return backend(template, target, hardware, compile_native=compile_native)
