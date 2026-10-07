# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact pure and locally purified circuit TN objectives for IQSP.

Each gate is a local tensor. A noisy logical window is a Kraus tensor with a
local environment leg shared by ket and bra. Final fidelity contracts this
network against an exact target MPS. No global simulated vector/density output
is constructed, and no decomposition occurs on a trainable parameter path.
This uncompressed specialization deliberately makes no large-system claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import opt_einsum as oe
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter, ParameterExpression
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Pauli

from experiments.state_preparation_competitors.common import validate_target
from experiments.state_preparation_competitors.evaluation import pauli_distribution
from experiments.state_preparation_competitors.vendor.iqsp_tensor.sun import su4_matrix

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

_ROTATIONS = {"rx": "X", "ry": "Y", "rz": "Z", "rxx": "XX", "ryy": "YY", "rzz": "ZZ"}


@dataclass(frozen=True)
class TensorNetworkOptions:
    """Exact-contraction resource policy; there is no silent rank truncation."""

    max_intermediate_elements: int = 2**24
    optimizer: Literal["greedy", "auto", "auto-hq"] = "greedy"

    def __post_init__(self) -> None:
        """Reject nonintegral memory limits and unsupported path optimizers."""
        if type(self.max_intermediate_elements) is not int or self.max_intermediate_elements < 1:
            msg = "max_intermediate_elements must be a positive integer."
            raise ValueError(msg)
        if self.optimizer not in {"greedy", "auto", "auto-hq"}:
            msg = "Use greedy, auto, or auto-hq tensor contraction planning."
            raise ValueError(msg)


def target_mps(target: ArrayLike) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """Factor a normalized q1-q6 target exactly in ascending Qiskit qubit order.

    Args:
        target: Little-endian fixture amplitudes, already normalized.

    Returns:
        Three-index MPS tensors (left, physical, right) and an exactness receipt.
    """
    vector = validate_target(target)
    qubits = vector.size.bit_length() - 1
    remainder = vector.reshape((2,) * qubits).transpose(tuple(reversed(range(qubits))))
    tensors = []
    left = 1
    for _ in range(qubits - 1):
        matrix = remainder.reshape(left * 2, -1)
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        rank = singular.size  # Keep zero singular values: this is exact, not approximate.
        tensors.append(torch.tensor(u.reshape(left, 2, rank), dtype=torch.complex128))
        remainder = singular[:, None] * vh
        left = rank
    tensors.append(torch.tensor(remainder.reshape(left, 2, 1), dtype=torch.complex128))
    return tuple(tensors), {
        "representation": "exact_target_mps",
        "qubit_order": "q0_to_qn_minus_1",
        "bond_dimensions": [tensor.shape[2] for tensor in tensors[:-1]],
        "discarded_squared_singular_values": 0.0,
        "truncation": False,
    }


def tensor_angle(
    expression: float | ParameterExpression, theta: torch.Tensor, parameters: tuple[Parameter, ...]
) -> torch.Tensor:
    """Evaluate a Qiskit expression with its exact first-order symbolic derivative."""
    if isinstance(expression, Parameter):
        return theta[parameters.index(expression)]
    if not isinstance(expression, ParameterExpression) or not expression.parameters:
        return torch.tensor(float(expression), dtype=torch.float64)
    positions = {parameter: parameters.index(parameter) for parameter in expression.parameters}
    bindings = {parameter: float(theta[index].detach()) for parameter, index in positions.items()}
    result = torch.tensor(float(expression.bind(bindings)), dtype=torch.float64)
    for parameter, index in positions.items():
        derivative = expression.gradient(parameter)
        if isinstance(derivative, ParameterExpression) and derivative.parameters:
            derivative = derivative.bind({p: bindings[p] for p in derivative.parameters})
        result += float(derivative) * (theta[index] - theta[index].detach())
    return result


class _Network:
    """A scalar tensor contraction with a checked, cached index path."""

    def __init__(self, terms: list[torch.Tensor], indices: list[list[int]], options: TensorNetworkOptions) -> None:
        """Plan from shapes only, preserving a fixed contraction across iterates."""
        inputs = ["".join(oe.get_symbol(index) for index in term) for term in indices]
        equation = ",".join(inputs) + "->"
        shapes = [tuple(term.shape) for term in terms]
        path, info = oe.contract_path(equation, *shapes, shapes=True, optimize=options.optimizer)
        self.largest_intermediate_elements = int(info.largest_intermediate)
        if self.largest_intermediate_elements > options.max_intermediate_elements:
            msg = "Tensor contraction exceeds max_intermediate_elements; no dense fallback is permitted."
            raise ValueError(msg)
        self.optimized_flops = int(info.opt_cost)
        self.expression = oe.contract_expression(equation, *shapes, optimize=path)

    def contract(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """Evaluate with Torch operations and retain the autograd graph."""
        return self.expression(*tensors, backend="torch")


class CircuitTensorGates:
    """Shared differentiable local gates, with no global simulation allocation."""

    def __init__(self, template: QuantumCircuit, hardware: str | None = None) -> None:
        """Validate and cache a fixed local-gate schedule and its noise channels."""
        if not isinstance(template, QuantumCircuit) or template.num_clbits or template.layout is not None:
            msg = "Expected an uncompiled circuit without classical bits."
            raise ValueError(msg)
        if isinstance(template.global_phase, ParameterExpression) or not np.isfinite(float(template.global_phase)):
            msg = "Use a finite constant global phase."
            raise ValueError(msg)
        if hardware is not None:
            pauli_distribution(hardware, 1)
        self.template = template.copy()
        self.parameters = tuple(template.parameters)
        self.hardware = hardware
        self._static: dict[int, torch.Tensor] = {}
        self._generators: dict[int, torch.Tensor] = {}
        self._kraus: dict[int, torch.Tensor] = {}
        self._validate_operations()

    def _validate_operations(self) -> None:
        """Cache local matrices while rejecting unresolved unsupported gates."""
        for index, item in enumerate(self.template.data):
            operation = item.operation
            if not isinstance(operation, Gate) or len(item.qubits) not in {1, 2}:
                msg = "Only one- or two-qubit unitary gates are supported."
                raise ValueError(msg)
            if operation.name == "su4":
                if len(operation.params) != 15 or len(item.qubits) != 2 or self.hardware is not None:
                    msg = "Custom su4 requires fifteen parameters and is restricted to the noiseless phase."
                    raise ValueError(msg)
            elif operation.name in _ROTATIONS:
                if len(operation.params) != 1 or len(item.qubits) != len(_ROTATIONS[operation.name]):
                    msg = "Pauli rotations require one angle and their declared arity."
                    raise ValueError(msg)
                self._generators[index] = torch.tensor(Pauli(_ROTATIONS[operation.name]).to_matrix())
            else:
                if any(isinstance(p, ParameterExpression) and p.parameters for p in operation.params):
                    msg = "Parameterized gates must be SU4 or scalar Pauli rotations."
                    raise ValueError(msg)
                matrix = np.asarray(Operator(operation).data)
                if not np.isfinite(matrix).all() or not np.allclose(matrix.conj().T @ matrix, np.eye(len(matrix))):
                    msg = "Every fixed gate must be finite and unitary."
                    raise ValueError(msg)
                self._static[index] = torch.tensor(matrix, dtype=torch.complex128)
            if self.hardware is not None:
                arity = len(item.qubits)
                probabilities, matrices = pauli_distribution(self.hardware, arity)
                array = np.stack(
                    [np.sqrt(p) * matrix for p, matrix in zip(probabilities, matrices, strict=True)], axis=-1
                )
                self._kraus[index] = torch.tensor(array.reshape((2,) * (2 * arity) + (-1,)))

    def _matrix(self, index: int, theta: torch.Tensor) -> torch.Tensor:
        """Build a differentiable local gate in Qiskit's local matrix ordering."""
        if index in self._static:
            return self._static[index]
        operation = self.template.data[index].operation
        angles = [tensor_angle(expression, theta, self.parameters) for expression in operation.params]
        if operation.name == "su4":
            return su4_matrix(torch.stack(angles))
        generator = self._generators[index]
        return torch.cos(angles[0] / 2) * torch.eye(len(generator), dtype=torch.complex128) - (
            1j * torch.sin(angles[0] / 2) * generator
        )

    def _theta(self, theta: ArrayLike, *, gradients: bool = False) -> torch.Tensor:
        """Require finite real coefficients with exactly the symbolic parameter count."""
        values = np.asarray(theta)
        if values.shape != (len(self.parameters),) or np.iscomplexobj(values) or not np.isfinite(values).all():
            msg = "Expected a finite real vector matching circuit.parameters."
            raise ValueError(msg)
        return torch.tensor(values, dtype=torch.float64, requires_grad=gradients)

    def bind(self, theta: ArrayLike) -> QuantumCircuit:
        """Resolve parameters while preserving every logical window, even at zero."""
        values = self._theta(theta)
        bindings = dict(zip(self.parameters, values.numpy(), strict=True))
        bound = self.template.assign_parameters(bindings)
        result = QuantumCircuit(bound.num_qubits)
        result.global_phase = bound.global_phase
        for index, item in enumerate(bound.data):
            sites = [bound.find_bit(qubit).index for qubit in item.qubits]
            operation = (
                UnitaryGate(self._matrix(index, values).detach().numpy(), label="su4")
                if item.operation.name == "su4"
                else item.operation
            )
            result.append(operation, sites)
        return result


class TensorNetworkObjective(CircuitTensorGates):
    """Deterministic noisy infidelity using a locally purified circuit network.

    Args:
        template: Fixed 1/2-qubit gate schedule; custom su4 is ideal-only.
        target: Normalized fixture vector converted once to an exact MPS.
        hardware: None for pure-state TN, or IBM/Google for logical Pauli noise.
        options: Contraction planner and checked intermediate-memory ceiling.
    """

    def __init__(
        self,
        template: QuantumCircuit,
        target: ArrayLike,
        hardware: str | None = None,
        *,
        options: TensorNetworkOptions | None = None,
    ) -> None:
        """Validate a fixed schedule and precompute its scalar contraction path."""
        self.target, self.target_metadata = target_mps(target)
        super().__init__(template, hardware)
        self.options = options or TensorNetworkOptions()
        if template.num_qubits != len(self.target):
            msg = "Expected an uncompiled circuit matching the target, without classical bits."
            raise ValueError(msg)
        zeros = torch.zeros(len(self.parameters), dtype=torch.float64)
        terms, indices = self._network_tensors(zeros, trace=False)
        self._objective_network = _Network(terms, indices, self.options)
        terms, indices = self._network_tensors(zeros, trace=True)
        self._trace_network = _Network(terms, indices, self.options)

    def _network_tensors(self, theta: torch.Tensor, *, trace: bool) -> tuple[list[torch.Tensor], list[list[int]]]:
        """Build ket/bra circuit tensors sharing only the local environment legs."""
        doubled = self.hardware is not None or trace
        terms: list[torch.Tensor] = []
        indices: list[list[int]] = []
        qubits = self.template.num_qubits
        frontier = [list(range(qubits))]
        counter = qubits
        if doubled:
            frontier.append(list(range(counter, counter + qubits)))
            counter += qubits
        for wires in frontier:
            for wire in wires:
                terms.append(torch.tensor([1, 0], dtype=torch.complex128))
                indices.append([wire])
        for index, item in enumerate(self.template.data):
            sites = [self.template.find_bit(qubit).index for qubit in reversed(item.qubits)]
            matrix = self._matrix(index, theta).reshape((2,) * (2 * len(sites)))
            for copy, wires in enumerate(frontier):
                output = list(range(counter, counter + len(sites)))
                counter += len(sites)
                terms.append(matrix if copy == 0 else matrix.conj())
                indices.append(output + [wires[site] for site in sites])
                for site, label in zip(sites, output, strict=True):
                    wires[site] = label
            if self.hardware is not None:
                environment = counter
                counter += 1
                kraus = self._kraus[index]
                for copy, wires in enumerate(frontier):
                    output = list(range(counter, counter + len(sites)))
                    counter += len(sites)
                    terms.append(kraus if copy == 0 else kraus.conj())
                    indices.append(output + [wires[site] for site in sites] + [environment])
                    for site, label in zip(sites, output, strict=True):
                        wires[site] = label
        if trace:
            for site in range(qubits):
                terms.append(torch.eye(2, dtype=torch.complex128))
                indices.append([frontier[0][site], frontier[1][site]])
        else:
            for copy, wires in enumerate(frontier):
                bonds = list(range(counter, counter + qubits + 1))
                counter += qubits + 1
                terms.extend([torch.ones(1, dtype=torch.complex128), torch.ones(1, dtype=torch.complex128)])
                indices.extend([[bonds[0]], [bonds[-1]]])
                for site, tensor in enumerate(self.target):
                    terms.append(tensor.conj() if copy == 0 else tensor)
                    indices.append([bonds[site], wires[site], bonds[site + 1]])
        return terms, indices

    def _fidelity_tensor(self, theta: torch.Tensor) -> torch.Tensor:
        """Contract the target projection, retaining differentiability."""
        terms, _ = self._network_tensors(theta, trace=False)
        value = self._objective_network.contract(terms)
        return value.abs().square() if self.hardware is None else value.real

    def loss_gradient(self, theta: ArrayLike) -> tuple[float, NDArray[np.float64]]:
        """Return infidelity and exact first derivatives of the untruncated network."""
        values = self._theta(theta, gradients=True)
        loss = 1 - self._fidelity_tensor(values)
        if self.parameters:
            gradient = torch.autograd.grad(loss, values)[0].detach().numpy()
        else:
            gradient = np.empty(0, dtype=np.float64)
        result = float(loss.detach())
        if not np.isfinite(result) or not np.isfinite(gradient).all():
            msg = "Nonfinite tensor-network objective or gradient."
            raise FloatingPointError(msg)
        return result, gradient

    def fidelity(self, theta: ArrayLike) -> float:
        """Return the same deterministic TN fidelity without gradient bookkeeping."""
        with torch.no_grad():
            return float(self._fidelity_tensor(self._theta(theta)))

    def diagnostics(self, theta: ArrayLike) -> dict[str, Any]:
        """Return trace, PSD-by-construction guarantee and measured TN resource bounds."""
        with torch.no_grad():
            terms, _ = self._network_tensors(self._theta(theta), trace=True)
            trace = self._trace_network.contract(terms)
        return {
            "backend": "exact_pure_circuit_tn" if self.hardware is None else "exact_locally_purified_circuit_tn",
            "hardware": self.hardware,
            "noise_placement": "after_every_logical_gate_including_zero_angles" if self.hardware else "none",
            "trace_real": float(trace.real),
            "trace_imag": float(trace.imag),
            "positive_semidefinite_by_construction": True,
            "trace_preserving_local_kraus": True,
            "state_truncation": False,
            "parameter_path_svd": False,
            "target": self.target_metadata,
            "largest_intermediate_elements": self._objective_network.largest_intermediate_elements,
            "trace_largest_intermediate_elements": self._trace_network.largest_intermediate_elements,
            "estimated_contraction_flops": self._objective_network.optimized_flops,
            "logical_noise_windows": len(self.template.data) if self.hardware else 0,
            "global_state_or_density_materialized": False,
            "large_system_scaling_claim": False,
        }
