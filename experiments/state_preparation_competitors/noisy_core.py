# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact small-system objectives shared by paper-based noisy competitors.

Symbolic compilation happens once, before training. Gradients differentiate the
same native gate/channel schedule that is returned for evaluation. The channel
is the existing benchmark's synthetic, parameter-independent Pauli mixture.
This module supplies numerical machinery, not an alternative training policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate, ParameterExpression
from qiskit.quantum_info import Operator, Pauli

# Reuse the independently tested local dense kernels; do not construct full-system unitaries.
from benchmarks.state_preparation.exact_density import _conjugate, _left_apply  # noqa: PLC2701  # Reuse tested kernels.
from experiments.state_preparation_competitors.common import validate_target
from experiments.state_preparation_competitors.evaluation import COMPILER, pauli_distribution

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from qiskit.circuit import Parameter

_GENERATORS = {
    name: Pauli(label).to_matrix()
    for name, label in {"rx": "X", "ry": "Y", "rz": "Z", "rxx": "XX", "ryy": "YY", "rzz": "ZZ"}.items()
}


def real_parameters(theta: ArrayLike, size: int) -> NDArray[np.float64]:
    """Copy a finite real parameter vector of the specified size.

    Raises:
        ValueError: If dimensions, type or finiteness are invalid.
    """
    values = np.asarray(theta)
    if values.shape != (size,) or values.dtype.kind not in "fiu" or not np.all(np.isfinite(values)):
        msg = "Expected a finite real parameter vector matching the template."
        raise ValueError(msg)
    return values.astype(np.float64, copy=True)


def nonnegative_integer(value: int, name: str) -> None:
    """Validate integer counts without silently accepting booleans.

    Raises:
        ValueError: If value is not a nonnegative built-in integer.
    """
    if type(value) is not int or value < 0:
        msg = f"{name} must be a nonnegative integer."
        raise ValueError(msg)


def _evaluate(expression: float | ParameterExpression, bindings: dict[Parameter, float]) -> float:
    """Resolve one real symbolic expression without unknown bindings."""
    if isinstance(expression, ParameterExpression):
        return float(expression.bind({p: bindings[p] for p in expression.parameters}))
    return float(expression)


@dataclass(frozen=True)
class _Window:
    """One local rotation/fixed gate with derivatives in original coordinates."""

    operation: Gate
    sites: tuple[int, ...]
    generator: NDArray[np.complex128] | None
    derivatives: tuple[tuple[int, float | ParameterExpression], ...]


class DenseObjective:
    """Exact noisy fidelity and ideal-state vector-Jacobian products for q1-q6.

    Args:
        template: Symbolic unitary circuit on a linear nearest-neighbor register.
        target: Normalized complex amplitudes in Qiskit little-endian order.
        hardware: Existing IBM/Google synthetic profile, or None for no noise.
        compile_native: Compile symbolically once with the shared RZ/SX/X/CX
            compiler settings, without routing. False allows scalar Pauli rotations and fixed local gates,
            with one noise window per supplied operation; callers must disclose
            that distinct logical-gate model.
        max_qubits: Explicit register limit for bounded-storage subclasses;
            ordinary dense-tape callers retain the six-qubit default.

    Raises:
        ValueError: If dimensions, parameters, topology or gates are unsupported.
        TypeError: If a nonunitary instruction is supplied.
    """

    def __init__(
        self,
        template: QuantumCircuit,
        target: ArrayLike,
        hardware: str | None = "IBM",
        *,
        compile_native: bool = True,
        max_qubits: int = 6,
    ) -> None:
        """Validate and compile a private fixed-topology training objective."""
        self.target = validate_target(target, max_qubits=max_qubits)
        n = template.num_qubits
        if template.num_clbits or template.layout is not None or self.target.size != 2**n:
            msg = "Template must match the target without classical bits or transpiler layout."
            raise ValueError(msg)
        self.num_qubits = n
        self.hardware = hardware
        if hardware is not None:
            pauli_distribution(hardware, 1)
        self.parameters = tuple(template.parameters)
        for item in template.data:
            if not isinstance(item.operation, Gate):
                msg = "Only unitary gate instructions are supported."
                raise TypeError(msg)
            sites = tuple(template.find_bit(q).index for q in item.qubits)
            if len(sites) not in {1, 2} or (len(sites) == 2 and abs(sites[0] - sites[1]) != 1):
                msg = "Use one-qubit gates or nearest-neighbor two-qubit gates."
                raise ValueError(msg)
        compiled = transpile(template, **COMPILER) if compile_native else template.copy()
        # Strip the compilation receipt; no routing or output permutation occurs.
        self.template = QuantumCircuit(n)
        self.template.compose(compiled, inplace=True)
        self.windows: list[_Window] = []
        for item in self.template.data:
            operation = item.operation
            if not isinstance(operation, Gate):
                msg = "The compiled template contains a nonunitary instruction."
                raise TypeError(msg)
            sites = tuple(reversed([self.template.find_bit(q).index for q in item.qubits]))
            symbolic = any(isinstance(x, ParameterExpression) and x.parameters for x in operation.params)
            derivatives = []
            generator = None
            if symbolic:
                if operation.name not in _GENERATORS or len(operation.params) != 1:
                    msg = "Parameterized gates must be scalar Pauli rotations after compilation."
                    raise ValueError(msg)
                expression = operation.params[0]
                generator = _GENERATORS[operation.name]
                for index, parameter in enumerate(self.parameters):
                    if parameter in expression.parameters:
                        derivatives.append((index, expression.gradient(parameter)))
            else:
                matrix = Operator(operation).data
                if not np.all(np.isfinite(matrix)) or not np.allclose(
                    matrix.conj().T @ matrix, np.eye(matrix.shape[0]), atol=1e-10, rtol=0
                ):
                    msg = "Fixed gates must be finite and unitary."
                    raise ValueError(msg)
            self.windows.append(_Window(operation, sites, generator, tuple(derivatives)))

    def bind(self, theta: ArrayLike) -> QuantumCircuit:
        """Return the exact trained gate schedule without cancellation or recompilation."""
        values = real_parameters(theta, len(self.parameters))
        bindings = dict(zip(self.parameters, values, strict=True))
        return self.template.assign_parameters(bindings, strict=False)

    def _channel(
        self, matrix: NDArray[np.complex128], sites: tuple[int, ...], *, noisy: bool
    ) -> NDArray[np.complex128]:
        """Apply the existing self-adjoint Pauli mixture to states or costates."""
        if not noisy or self.hardware is None:
            return matrix
        probabilities, paulis = pauli_distribution(self.hardware, len(sites))
        result = probabilities[0] * matrix
        for weight, pauli in zip(probabilities[1:], paulis[1:], strict=True):
            result += weight * _conjugate(matrix, pauli, sites, self.num_qubits)
        return result

    def _forward(
        self, theta: ArrayLike, *, noisy: bool, tape: bool
    ) -> tuple[
        NDArray[np.complex128],
        list[tuple[_Window, NDArray[np.complex128], NDArray[np.complex128]]],
        dict[Parameter, float],
    ]:
        """Propagate the state and optionally retain local adjoint intermediates."""
        values = real_parameters(theta, len(self.parameters))
        bindings = dict(zip(self.parameters, map(float, values), strict=True))
        bound = self.bind(values)
        rho = np.zeros((self.target.size, self.target.size), dtype=np.complex128)
        rho[0, 0] = 1
        history: list[tuple[_Window, NDArray[np.complex128], NDArray[np.complex128]]] = []
        for window, item in zip(self.windows, bound.data, strict=True):
            unitary = Operator(item.operation).data
            rotated = _conjugate(rho, unitary, window.sites, self.num_qubits)
            if tape:
                history.append((window, unitary, rotated))
            rho = self._channel(rotated, window.sites, noisy=noisy)
        if not np.all(np.isfinite(rho)) or abs(np.trace(rho) - 1) > 1e-9 or np.max(abs(rho - rho.conj().T)) > 1e-9:
            msg = "Density propagation failed finiteness, trace or Hermiticity checks."
            raise FloatingPointError(msg)
        return rho, history, bindings

    def density(self, theta: ArrayLike, *, noisy: bool = True) -> NDArray[np.complex128]:
        """Return the exact density output, including all modeled noise histories."""
        return self._forward(theta, noisy=noisy, tape=False)[0]

    def ideal_density(self, theta: ArrayLike) -> NDArray[np.complex128]:
        """Return the noiseless density output for a simulator surrogate gradient."""
        return self.density(theta, noisy=False)

    def fidelity(self, theta: ArrayLike) -> float:
        """Return noisy pure-target fidelity on the fixed native schedule."""
        rho = self.density(theta)
        return float(np.vdot(self.target, rho @ self.target).real)

    def _adjoint(
        self,
        history: list[tuple[_Window, NDArray[np.complex128], NDArray[np.complex128]]],
        bindings: dict[Parameter, float],
        costate: NDArray[np.complex128],
        *,
        noisy: bool,
    ) -> NDArray[np.float64]:
        """Contract the local analytic rotation derivatives with a Hermitian costate."""
        gradient = np.zeros(len(self.parameters), dtype=np.float64)
        for window, unitary, rotated in reversed(history):
            costate = self._channel(costate, window.sites, noisy=noisy)
            if window.generator is not None:
                left = _left_apply(rotated, -0.5j * window.generator, window.sites, self.num_qubits)
                derivative = float(np.vdot(costate, left + left.conj().T).real)
                for index, expression in window.derivatives:
                    gradient[index] += _evaluate(expression, bindings) * derivative
            costate = _conjugate(costate, unitary.conj().T, window.sites, self.num_qubits)
        if not np.all(np.isfinite(gradient)):
            msg = "The density derivative is nonfinite."
            raise FloatingPointError(msg)
        return gradient

    def loss_gradient(self, theta: ArrayLike) -> tuple[float, NDArray[np.float64]]:
        """Return exact noisy infidelity and its analytic adjoint gradient."""
        rho, tape, bindings = self._forward(theta, noisy=True, tape=True)
        projector = np.outer(self.target, self.target.conj())
        loss = 1 - float(np.vdot(projector, rho).real)
        return loss, self._adjoint(tape, bindings, -projector, noisy=True)

    def density_vjp(self, theta: ArrayLike, costate: ArrayLike) -> NDArray[np.float64]:
        """Differentiate Tr(costate rho_ideal), holding the supplied costate fixed.

        This is the noiseless simulator Jacobian contraction required by
        RobustState. It intentionally does not differentiate the noisy output
        used to construct its upstream loss derivative.

        Raises:
            ValueError: If costate is not a finite Hermitian matrix of the right size.
        """
        matrix = np.asarray(costate, dtype=np.complex128)
        if (
            matrix.shape != (self.target.size, self.target.size)
            or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=0)
        ):
            msg = "Costate must be a finite Hermitian matrix matching the target."
            raise ValueError(msg)
        _, tape, bindings = self._forward(theta, noisy=False, tape=True)
        return self._adjoint(tape, bindings, matrix.copy(), noisy=False)
