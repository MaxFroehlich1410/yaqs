# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Deterministic forward MPO assessment of frozen logical noisy circuits.

Local physical indices are (ket, bra), site 0 is the least significant qubit,
and channels follow the historical at-most-one-Pauli-jump window exactly.
No dense state, density, or global unitary is built by the evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe
from threadpoolctl import threadpool_limits

from benchmarks.state_preparation import load_target
from experiments import topdown_backend_support as support
from experiments import topdown_depth_search as depth
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.optimization.mps_backend import mps_overlap

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.mps import MPS
    from mqt.yaqs.optimization import ParameterizedCircuit

VERSION = "topdown.forward_mpo_evaluation.v1"


@dataclass(frozen=True)
class MPOSettings:
    """Explicit approximation and per-operation workspace limits.

    Attributes:
        max_bond_dim: Optional bond cap. None retains all dimensions by default.
        svd_atol: Absolute Frobenius tail-norm budget per canonical split.
        max_workspace_bytes: Conservative estimated workspace limit per split.
    """

    max_bond_dim: int | None = None
    svd_atol: float = 0.0
    max_workspace_bytes: int = 1024**3

    def __post_init__(self) -> None:
        """Reject invalid approximation settings and workspace limits.

        Raises:
            ValueError: If a setting is invalid.
        """
        if (
            (self.max_bond_dim is not None and (type(self.max_bond_dim) is not int or self.max_bond_dim < 1))
            or not np.isfinite(self.svd_atol)
            or self.svd_atol < 0
            or type(self.max_workspace_bytes) is not int
            or self.max_workspace_bytes < 1
        ):
            msg = "Invalid MPO bond, SVD tolerance, or workspace limit."
            raise ValueError(msg)


DEFAULT_SETTINGS = MPOSettings()


def _conjugation_map(operator: NDArray, arity: int) -> NDArray:
    """Return a local conjugation map in interleaved site-(ket,bra) order."""
    tensor = np.einsum("ac,bd->abcd", operator, operator.conj())
    if arity == 1:
        return tensor.reshape(4, 4)
    return tensor.reshape((2,) * 8).transpose(0, 2, 1, 3, 4, 6, 5, 7).reshape(16, 16)


def logical_channel(hardware: str, sites: tuple[int, ...]) -> NDArray:
    """Sum the existing logical sampler's Pauli outcomes exactly.

    Returns:
        A 4-by-4 or 16-by-16 local density-channel matrix.

    Raises:
        ValueError: If the hardware profile or gate support is unsupported.
    """
    if hardware not in {"IBM", "Google", "noiseless"} or len(sites) not in {1, 2}:
        msg = "Expected IBM, Google, or noiseless profile and one/two-site gates."
        raise ValueError(msg)
    dimension = 4 ** len(sites)
    if hardware == "noiseless":
        return np.eye(dimension, dtype=complex)
    probabilities, outcomes = depth.noise_outcomes(hardware, sites)
    channel = probabilities[0] * np.eye(dimension, dtype=complex)
    for probability, (operator, acted_on) in zip(probabilities[1:], outcomes, strict=True):
        embedded = operator
        if len(sites) == 2 and len(acted_on) == 1:
            embedded = np.kron(operator, np.eye(2)) if acted_on[0] == sites[0] else np.kron(np.eye(2), operator)
        channel += probability * _conjugation_map(embedded, len(sites))
    return channel


class ForwardMPO:
    """Forward density MPO with canonical splits and no trajectory sampling."""

    def __init__(self, num_qubits: int, settings: MPOSettings = DEFAULT_SETTINGS) -> None:
        """Initialize |0><0| as a bond-one MPO.

        Raises:
            ValueError: If the register is empty.
        """
        if type(num_qubits) is not int or num_qubits < 1:
            msg = "MPO assessment requires a nonempty qubit register."
            raise ValueError(msg)
        self.settings = settings
        self.tensors = [np.array([1, 0, 0, 0], dtype=complex).reshape(4, 1, 1) for _ in range(num_qubits)]
        self.center = 0
        self.maximum_bond = 1
        self.peak_tensor_bytes = 16 * 4 * num_qubits
        self.maximum_estimated_workspace_bytes = 0
        self.bond_history: list[dict] = []
        self.splits = 0
        self.truncated_splits = 0
        self.discarded_frobenius_norm_sum = 0.0

    def _guard(self, elements: int, *, factor: int = 1) -> None:
        estimate = factor * 16 * elements + sum(t.nbytes for t in self.tensors)
        self.maximum_estimated_workspace_bytes = max(self.maximum_estimated_workspace_bytes, estimate)
        if estimate > self.settings.max_workspace_bytes:
            msg = f"MPO estimated workspace {estimate} exceeds limit {self.settings.max_workspace_bytes}."
            raise MemoryError(msg)

    def _move_center(self, site: int) -> None:
        while self.center < site:
            index = self.center
            tensor = self.tensors[index]
            physical, left, right = tensor.shape
            self._guard(tensor.size, factor=8)
            q, r = np.linalg.qr(tensor.transpose(1, 0, 2).reshape(left * physical, right), mode="reduced")
            self.tensors[index] = q.reshape(left, physical, q.shape[1]).transpose(1, 0, 2)
            self.tensors[index + 1] = np.einsum("ab,pbc->pac", r, self.tensors[index + 1])
            self.center += 1
        while self.center > site:
            index = self.center
            tensor = self.tensors[index]
            physical, left, right = tensor.shape
            self._guard(tensor.size, factor=8)
            q, r = np.linalg.qr(tensor.transpose(0, 2, 1).reshape(physical * right, left), mode="reduced")
            self.tensors[index] = q.T.reshape(q.shape[1], physical, right).transpose(1, 0, 2)
            self.tensors[index - 1] = np.einsum("pab,bc->pac", self.tensors[index - 1], r.T)
            self.center -= 1

    def apply(self, channel: NDArray, sites: tuple[int, ...]) -> None:
        """Apply one complete local gate/noise map and canonicalize its split.

        Args:
            channel: Local superoperator in site-(ket,bra) physical order.
            sites: One site or an ascending nearest-neighbor pair.

        Raises:
            ValueError: If the map or support is invalid.
        """
        if (
            len(sites) not in {1, 2}
            or any(site < 0 or site >= len(self.tensors) for site in sites)
            or (len(sites) == 2 and sites[1] != sites[0] + 1)
            or channel.shape != (4 ** len(sites),) * 2
            or not np.all(np.isfinite(channel))
        ):
            msg = "MPO evaluation supports finite one-site or nearest-neighbor two-site channels."
            raise ValueError(msg)
        site = sites[0]
        self._move_center(site)
        if len(sites) == 1:
            self.tensors[site] = np.einsum("pq,qab->pab", channel, self.tensors[site])
        else:
            left, right = self.tensors[site : site + 2]
            self._guard(16 * left.shape[1] * right.shape[2], factor=12)
            pair = np.einsum("pab,qbc->pqac", left, right)
            pair = (channel @ pair.reshape(16, -1)).reshape(pair.shape)
            matrix = pair.transpose(2, 0, 1, 3).reshape(4 * left.shape[1], 4 * right.shape[2])
            u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
            keep = len(singular)
            if self.settings.svd_atol:
                tails = np.cumsum(singular[::-1] ** 2)[::-1]
                candidates = np.flatnonzero(tails <= self.settings.svd_atol**2)
                if len(candidates):
                    keep = max(1, int(candidates[0]))
            if self.settings.max_bond_dim is not None:
                keep = min(keep, self.settings.max_bond_dim)
            discarded = float(np.linalg.norm(singular[keep:]))
            self.discarded_frobenius_norm_sum += discarded
            self.truncated_splits += int(discarded > 0)
            self.splits += 1
            self.tensors[site] = u[:, :keep].reshape(left.shape[1], 4, keep).transpose(1, 0, 2)
            self.tensors[site + 1] = (
                (singular[:keep, None] * vh[:keep]).reshape(keep, 4, right.shape[2]).transpose(1, 0, 2)
            )
            self.center = site + 1
        self.maximum_bond = max(self.maximum_bond, *(max(t.shape[1:]) for t in self.tensors))
        self.peak_tensor_bytes = max(self.peak_tensor_bytes, sum(t.nbytes for t in self.tensors))

    def to_mpo(self) -> MPO:
        """Return the retained operator using YAQS's (ket,bra,left,right) layout."""
        state = MPO()
        state.custom([tensor.reshape(2, 2, *tensor.shape[1:]).copy() for tensor in self.tensors], transpose=False)
        return state

    def trace(self) -> complex:
        """Contract the trace without materializing the density matrix."""
        environment = np.ones(1, dtype=complex)
        for tensor in self.tensors:
            environment = np.matmul(environment, tensor[0] + tensor[3])
        return complex(environment[0])

    def fidelity(self, target: MPS) -> float:
        """Contract <target|rho|target> directly with a normalized pure MPS.

        Raises:
            ValueError: If the target tensors or normalization are invalid.
            FloatingPointError: If the fidelity has a significant imaginary part.
        """
        if target.length != len(self.tensors) or any(t.shape[0] != 2 for t in target.tensors):
            msg = "Target MPS must have the same qubit register."
            raise ValueError(msg)
        if not np.isclose(mps_overlap(target, target), 1.0, rtol=0, atol=1e-10):
            msg = "Target MPS must be normalized."
            raise ValueError(msg)
        environment = np.ones((1, 1, 1), dtype=complex)
        for tensor, ket in zip(self.tensors, target.tensors, strict=True):
            self._guard(ket.shape[2] ** 2 * tensor.shape[2], factor=8)
            environment = oe.contract(
                "abc,pad,pqbe,qcf->def",
                environment,
                ket.conj(),
                tensor.reshape(2, 2, *tensor.shape[1:]),
                ket,
                optimize="greedy",
            )
        value = complex(environment[0, 0, 0])
        if abs(value.imag) > 1e-9:
            msg = f"MPO fidelity is complex: {value}."
            raise FloatingPointError(msg)
        return float(value.real)

    def diagnostics(self) -> dict:
        """Return truncation, storage, trace, and conservative fidelity-error diagnostics."""
        trace = self.trace()
        return {
            "settings": asdict(self.settings),
            "bond_growth": self.bond_history,
            "maximum_bond": self.maximum_bond,
            "peak_retained_tensor_bytes": self.peak_tensor_bytes,
            "maximum_estimated_workspace_bytes": self.maximum_estimated_workspace_bytes,
            "two_site_splits": self.splits,
            "truncated_splits": self.truncated_splits,
            "discarded_frobenius_norm_sum": self.discarded_frobenius_norm_sum,
            "fidelity_absolute_error_bound_excluding_roundoff": self.discarded_frobenius_norm_sum,
            "exact_up_to_roundoff": self.truncated_splits == 0,
            "trace_real": trace.real,
            "trace_imag": trace.imag,
            "trace_error": abs(trace - 1),
            "renormalized": False,
            "positivity": "not enforced after SVD truncation",
        }


def propagate(
    circuit: ParameterizedCircuit,
    theta: NDArray,
    hardware: str,
    settings: MPOSettings = DEFAULT_SETTINGS,
    *,
    progress: Callable[[dict], None] | None = None,
) -> ForwardMPO:
    """Propagate every original gate and enabled logical noise window once.

    Returns:
        Final density MPO and its numerical diagnostics.

    Raises:
        ValueError: If parameters, profile, gate unitarity, or circuit supports are invalid.
    """
    if (
        hardware not in {"IBM", "Google", "noiseless"}
        or theta.shape != (circuit.num_params,)
        or not np.isrealobj(theta)
        or not np.all(np.isfinite(theta))
    ):
        msg = "Invalid MPO profile or circuit parameters."
        raise ValueError(msg)
    if any(gate.data_map is not None for gate in circuit.gates):
        msg = "MPO final assessment requires data-free gates."
        raise ValueError(msg)
    state = ForwardMPO(circuit.num_qubits, settings)
    start = time.perf_counter()
    history = []
    for index, gate in enumerate(circuit.gates):
        entry = {
            "gate_index": index,
            "gate_name": gate.name,
            "sites": list(gate.sites),
            "elapsed_wall_seconds": time.perf_counter() - start,
            "bond_dimensions": [t.shape[2] for t in state.tensors[:-1]],
        }
        if progress is not None:
            progress({**entry, "event": "before_gate"})
        unitary, sites = circuit.gate_matrix(gate, theta)
        if not np.allclose(unitary.conj().T @ unitary, np.eye(len(unitary)), rtol=0, atol=1e-12):
            msg = "Final circuit gates must be unitary."
            raise ValueError(msg)
        channel = _conjugation_map(unitary, len(sites))
        if gate.noise_enabled:
            channel = logical_channel(hardware, sites) @ channel
        state.apply(channel, sites)
        entry.update(
            elapsed_wall_seconds=time.perf_counter() - start,
            bond_dimensions=[t.shape[2] for t in state.tensors[:-1]],
            retained_tensor_bytes=sum(t.nbytes for t in state.tensors),
            maximum_bond=state.maximum_bond,
        )
        history.append(entry)
        if progress is not None:
            progress({**entry, "event": "after_gate"})
    state.bond_history = history
    return state


def evaluate(
    circuit: ParameterizedCircuit,
    theta: NDArray,
    target: MPS,
    hardware: str = "IBM",
    settings: MPOSettings = DEFAULT_SETTINGS,
    *,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Return deterministic pure-target noisy fidelity for one frozen circuit."""
    start = time.perf_counter()
    state = propagate(circuit, theta, hardware, settings, progress=progress)
    fidelity = state.fidelity(target)
    diagnostics = state.diagnostics()
    bound = diagnostics["fidelity_absolute_error_bound_excluding_roundoff"]
    if not -bound - 1e-9 <= fidelity <= 1 + bound + 1e-9:
        msg = "MPO fidelity is inconsistent with its truncation-error bound."
        raise FloatingPointError(msg)
    return {
        "version": VERSION,
        "hardware": hardware,
        "noise_law": "original logical at-most-one-Pauli-jump channel after each enabled gate",
        "num_qubits": circuit.num_qubits,
        "gate_count": len(circuit.gates),
        "fidelity": fidelity,
        "diagnostics": diagnostics,
        "wall_seconds": time.perf_counter() - start,
    }


def main(argv: list[str] | None = None) -> None:
    """Assess a sealed frozen parameter checkpoint into a separate output file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--circuit-checkpoint", type=Path, help="Sealed circuit checkpoint, if stored separately.")
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--hardware", choices=("IBM", "Google", "noiseless"), default="IBM")
    parser.add_argument("--max-bond-dim", type=int)
    parser.add_argument("--svd-atol", type=float, default=0.0)
    parser.add_argument("--max-workspace-mib", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("assessment output already exists; use a fresh path")
    saved = support.load_checkpoint(args.checkpoint)
    source = support.load_checkpoint(args.circuit_checkpoint or args.checkpoint)
    if saved is None or source is None:
        parser.error("assessment requires sealed parameter and circuit checkpoints")
    _record, arrays = saved
    circuit_record = source[0].get("circuit", source[0].get("training", {}).get("circuit"))
    if circuit_record is None:
        parser.error("circuit not found; supply --circuit-checkpoint")
    circuit = support.restore_circuit(circuit_record)
    target = load_target(circuit.num_qubits, args.target_id)
    with threadpool_limits(limits=1):
        report = evaluate(
            circuit,
            arrays["theta"],
            target.to_mps(),
            args.hardware,
            MPOSettings(args.max_bond_dim, args.svd_atol, args.max_workspace_mib * 1024**2),
        )
    report.update(
        target_id=args.target_id,
        target_sha256=hashlib.sha256(target.state_vector.tobytes()).hexdigest(),
        parameter_checkpoint_sha256=support.digest(args.checkpoint / "complete.json"),
        circuit_checkpoint_sha256=support.digest((args.circuit_checkpoint or args.checkpoint) / "complete.json"),
        evaluator_source_sha256=support.digest(Path(__file__)),
    )
    support.write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "fidelity": report["fidelity"]}), flush=True)


if __name__ == "__main__":
    main()
