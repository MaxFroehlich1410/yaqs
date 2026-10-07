# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Thin, version-pinned adapter for QCLib's official BAA low-rank preparation."""

from __future__ import annotations

import hashlib
import time
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit.quantum_info import Operator, Statevector

from experiments.state_preparation_competitors.common import Candidate, validate_target

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from qiskit import QuantumCircuit

QCLIB_VERSION = "0.1.15"
QISKIT_VERSION = "2.4.1"
QCLIB_WHEEL_SHA256 = "773b41cefc7154d5466f55777c71ad5d3813e117f7f4637d957b982c48e4922a"
UPSTREAM_HASHES = {
    "qclib.state_preparation.baa_lowrank": "87e44268d6278dd4531f8b2ed604dd8f7c62702477a221162ecb7dcb5bf60ab6",
    "qclib.state_preparation.lowrank": "78ec56f3072685ad0d6b34026118074407e61766747ea4abab79a929c0d592d8",
    "qclib.state_preparation.util.baa": "7aa209416caf4f4cef209ff76b56d13a37253045a1e9aa5c72f460ea758c1dbd",
}
FIDELITY_TOLERANCE = 1e-9
DECOMPOSITION_REPETITIONS = 20


def _loss_grid(fidelity_losses: tuple[float, ...]) -> tuple[float, ...]:
    """Validate the requested loss bounds and include the exact endpoint once."""
    if not isinstance(fidelity_losses, tuple) or not fidelity_losses:
        msg = "fidelity_losses must be a nonempty tuple of finite bounds in [0, 1]."
        raise ValueError(msg)
    values = []
    for loss in fidelity_losses:
        if isinstance(loss, (bool, np.bool_)) or not isinstance(loss, (int, float, np.integer, np.floating)):
            msg = "Fidelity-loss bounds must be real numbers, not booleans."
            raise TypeError(msg)
        value = float(loss)
        if not np.isfinite(value) or not 0 <= value <= 1:
            msg = "Fidelity-loss bounds must be finite and lie in [0, 1]."
            raise ValueError(msg)
        values.append(value)
    return tuple(sorted({0.0, *values}))


def _load_upstream() -> type[Any]:
    """Load the unchanged official implementation only at the audited versions."""
    for package, expected in (("qclib", QCLIB_VERSION), ("qiskit", QISKIT_VERSION)):
        try:
            actual = version(package)
        except PackageNotFoundError as error:
            msg = "Install experiments/state_preparation_competitors/requirements.txt in an isolated uv environment."
            raise RuntimeError(msg) from error
        if actual != expected:
            msg = f"This adapter requires {package}=={expected}; found {actual}."
            raise RuntimeError(msg)
    try:
        modules = {name: import_module(name) for name in UPSTREAM_HASHES}
    except ImportError as error:
        msg = "QCLib imports require the pinned optional dependencies, including qiskit-aer; use requirements.txt."
        raise RuntimeError(msg) from error
    for name, module in modules.items():
        if (
            module.__file__ is None
            or hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != UPSTREAM_HASHES[name]
        ):
            msg = f"The installed QCLib source differs from the audited release: {name}."
            raise RuntimeError(msg)
    return modules["qclib.state_preparation.baa_lowrank"].BaaLowRankInitialize


def _verify_candidate(circuit: QuantumCircuit, target: NDArray[np.complex128], loss: float) -> dict:
    """Independently check unitary output, state ordering and the requested loss bound."""
    qubits = target.size.bit_length() - 1
    if circuit.num_qubits != qubits or circuit.num_ancillas or circuit.num_clbits or circuit.parameters:
        msg = "BAA returned ancillas, classical bits, unresolved parameters, or the wrong number of qubits."
        raise RuntimeError(msg)
    if any(item.operation.num_qubits > 2 for item in circuit.data):
        msg = "BAA gate decomposition left an opaque gate with more than two qubits."
        raise RuntimeError(msg)
    try:
        unitary = Operator(circuit)
        state = np.asarray(Statevector.from_instruction(circuit).data, dtype=np.complex128)
    except Exception as error:
        msg = "BAA output could not be independently verified as a unitary preparation circuit."
        raise RuntimeError(msg) from error
    if not unitary.is_unitary(atol=FIDELITY_TOLERANCE, rtol=0):
        msg = "BAA returned a nonunitary circuit."
        raise RuntimeError(msg)
    norm_error = float(abs(np.vdot(state, state) - 1))
    raw_fidelity = float(abs(np.vdot(target, state)) ** 2)
    if not np.isfinite(raw_fidelity) or norm_error > FIDELITY_TOLERANCE:
        msg = "BAA output has invalid amplitudes or normalization."
        raise RuntimeError(msg)
    if 1 - raw_fidelity > loss + FIDELITY_TOLERANCE:
        msg = "BAA output violates the requested fidelity-loss bound."
        raise RuntimeError(msg)
    fidelity = float(np.clip(raw_fidelity, 0, 1))
    return {
        "ideal_fidelity": fidelity,
        "actual_squared_fidelity": fidelity,
        "raw_squared_fidelity": raw_fidelity,
        "actual_fidelity_loss": 1 - fidelity,
        "state_norm_error": norm_error,
        "fidelity_bound_tolerance": FIDELITY_TOLERANCE,
    }


def baa_candidates(
    target: ArrayLike, *, fidelity_losses: tuple[float, ...], strategy: str = "brute_force", use_low_rank: bool = True
) -> list[Candidate]:
    """Generate exact and bounded approximate circuits using official QCLib BAA-LRSP.

    Args:
        target: Normalized complex amplitudes of a one- to six-qubit state.
        fidelity_losses: Finite squared-fidelity loss bounds in [0, 1]. Bounds
            are sorted and deduplicated; zero is always included explicitly.
        strategy: Official BAA search strategy, either ``brute_force`` or ``greedy``.
        use_low_rank: Whether upstream also considers Schmidt ranks above one.

    Returns:
        Deterministic candidates in increasing bound order, with independently
        measured ideal squared fidelity and upstream approximation metadata.
        Individual and cumulative CPU include setup, official synthesis, exact
        gate decomposition and verification; these costs are also separated.

    Raises:
        TypeError: If a loss bound is not a real numeric value.
        ValueError: If the target, fidelity-loss grid or options are invalid.
        RuntimeError: If dependencies differ from the audited pins or an output
            violates the qubit, unitarity or squared-fidelity requirements.
    """
    started, wall_started = time.process_time(), time.perf_counter()
    vector = validate_target(target)
    bounds = _loss_grid(fidelity_losses)
    if strategy not in {"brute_force", "greedy"}:
        msg = "Supported BAA strategies are 'brute_force' and 'greedy'."
        raise ValueError(msg)
    if type(use_low_rank) is not bool:
        msg = "use_low_rank must be a bool."
        raise ValueError(msg)
    initializer = _load_upstream()
    setup_cpu = time.process_time() - started
    previous_cpu = started
    candidates = []
    for index, bound in enumerate(bounds):
        synthesis_start = time.process_time()
        options = {
            "max_fidelity_loss": bound,
            "strategy": strategy,
            "use_low_rank": use_low_rank,
            "max_combination_size": 0,
            "iso_scheme": "ccd",
            "unitary_scheme": "qsd",
        }
        gate = initializer(vector.copy(), opt_params=options)
        # This only expands upstream/Qiskit gate definitions: no new optimizer,
        # routing, gate cancellation, approximation or qubit permutation.
        circuit = gate.definition.decompose(reps=DECOMPOSITION_REPETITIONS)
        synthesis_cpu = time.process_time() - synthesis_start
        verification_start = time.process_time()
        verified = _verify_candidate(circuit, vector, bound)
        verification_cpu = time.process_time() - verification_start
        node = gate.node
        estimated_loss = float(node.total_fidelity_loss)
        if not np.isfinite(estimated_loss):
            msg = "QCLib returned a nonfinite approximation estimate."
            raise RuntimeError(msg)
        ended = time.process_time()
        metadata = {
            "method": "baa_lrsp",
            "candidate_id": f"baa_{index:02d}_loss_{bound:.12g}",
            "status": "complete",
            "deterministic": True,
            "qclib_version": QCLIB_VERSION,
            "qiskit_version": QISKIT_VERSION,
            "upstream_wheel_sha256": QCLIB_WHEEL_SHA256,
            "upstream_source_sha256": UPSTREAM_HASHES.copy(),
            "requested_fidelity_loss": bound,
            "upstream_options": options,
            "upstream_fidelity_loss_estimate": estimated_loss,
            "upstream_saved_cnot_estimate": int(node.total_saved_cnots),
            "upstream_subsystem_qubits": [list(qubits) for qubits in node.qubits],
            "upstream_schmidt_ranks": [int(rank) for rank in node.ranks],
            "decomposition_repetitions": DECOMPOSITION_REPETITIONS,
            **verified,
            "setup_cpu_seconds": setup_cpu if index == 0 else 0.0,
            "synthesis_cpu_seconds": synthesis_cpu,
            "verification_cpu_seconds": verification_cpu,
            "cpu_seconds": ended - previous_cpu,
            "cumulative_cpu_seconds": ended - started,
            "cumulative_wall_seconds": time.perf_counter() - wall_started,
        }
        candidates.append(Candidate(circuit, metadata))
        previous_cpu = ended
    return candidates
