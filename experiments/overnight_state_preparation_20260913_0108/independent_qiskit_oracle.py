# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent scalar-circuit oracle using only Qiskit and SciPy numerics.

There are no YAQS or benchmark implementation imports. Standard channels are
matrix exponentials of literal local Liouvillians in Qiskit's convention,
calibrated through Qiskit's average_gate_fidelity. This intentionally avoids
the selected evaluator's Pauli-transfer inversion and local contraction code.
The caller owns input authentication, output persistence and inclusive timing.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit.circuit.library import HGate, RXGate, RXXGate, RYGate, RYYGate, RZGate, RZZGate
from qiskit.quantum_info import Choi, DensityMatrix, Kraus, Operator, Pauli, SuperOp, average_gate_fidelity
from scipy.linalg import expm

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

VERSION = "overnight.independent_qiskit_oracle.v1"
PHYSICAL_TOLERANCE = 1e-10
PRIMITIVES = {"rx": RXGate, "ry": RYGate, "rz": RZGate, "rxx": RXXGate, "ryy": RYYGate, "rzz": RZZGate}

# These literal sets transcribe the immutable benchmark; no profile builder is
# imported. Pauli labels are mathematical symbols rather than prose words.
# spellchecker:off  # noqa: ERA001
LOCAL_TWO = ("XI", "YI", "ZI", "IX", "IY", "IZ")
CORRELATED_TWO = ("XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ")
STANDARD_PROFILES = {
    "dephasing_1s_1q": (("Z",), ()),
    "dephasing_1s_2q": ((), ("ZI", "IZ")),
    "dephasing_1s_all": (("Z",), ("ZI", "IZ")),
    "dephasing_2s_2q": ((), ("ZZ",)),
    "dephasing_1s2s_all": (("Z",), ("ZI", "IZ", "ZZ")),
    "depolarizing_1s_1q": (("X", "Y", "Z"), ()),
    "depolarizing_1s_2q": ((), LOCAL_TWO),
    "depolarizing_1s_all": (("X", "Y", "Z"), LOCAL_TWO),
    "depolarizing_2s_2q": ((), CORRELATED_TWO),
    "depolarizing_1s2s_all": (("X", "Y", "Z"), (*LOCAL_TWO, *CORRELATED_TWO)),
}
# spellchecker:on  # noqa: ERA001
PROFILE_IDS = ("noiseless", "ballarin_coupled", *STANDARD_PROFILES)


@dataclass(frozen=True)
class _Calibrated:
    """A private cached channel and its independently derived calibration."""

    channel: SuperOp
    metadata: dict[str, Any]


@cache
def _standard_channel(noise_id: str, arity: int) -> _Calibrated:
    """Construct a local matrix exponential and independently invert its error."""
    if noise_id not in STANDARD_PROFILES or type(arity) is not int or arity not in {1, 2}:
        msg = "Expected a standard profile and integer arity one or two."
        raise ValueError(msg)
    labels = STANDARD_PROFILES[noise_id][arity - 1]
    identity = SuperOp(Operator(np.eye(2**arity))).data
    generator = sum((SuperOp(Pauli(label)).data - identity for label in labels), start=np.zeros_like(identity))
    target_error = (0.00064 if arity == 1 else 0.0051) if labels else 0.0

    def channel_at(strength: float) -> SuperOp:
        return SuperOp(expm(strength * generator))

    def error_at(strength: float) -> float:
        return float(1 - average_gate_fidelity(channel_at(strength), require_cp=False, require_tp=False))

    strength = 0.0
    if labels:
        low, high = 0.0, 1.0
        if not error_at(low) <= target_error <= error_at(high):
            msg = "Independent gate-error calibration has no bracket."
            raise FloatingPointError(msg)
        # Fixed bisection is independent of the selected Brent/Pauli-character
        # implementation. Eighty steps exceed binary64 resolution of this root.
        for _ in range(80):
            middle = (low + high) / 2
            if error_at(middle) < target_error:
                low = middle
            else:
                high = middle
        strength = (low + high) / 2
    channel = channel_at(strength)
    actual_error = float(1 - average_gate_fidelity(channel, require_cp=True, require_tp=True))
    minimum_choi_eigenvalue = float(np.linalg.eigvalsh(Choi(channel).data).min())
    if not channel.is_cptp(atol=1e-12, rtol=0) or abs(actual_error - target_error) > 2e-13:
        msg = "Independent calibrated local channel failed CPTP or gate-error validation."
        raise FloatingPointError(msg)
    return _Calibrated(
        channel,
        {
            "noise_id": noise_id,
            "arity": arity,
            "generators": list(labels),
            "coefficient": strength,
            "target_error": target_error,
            "actual_error": actual_error,
            "minimum_choi_eigenvalue": minimum_choi_eigenvalue,
            "calibration": "Qiskit_average_gate_fidelity_of_SciPy_expm_Liouvillian;80_bisections",
            "vectorization": "Qiskit_SuperOp_column_vectorization",
        },
    )


def calibration_profile(noise_id: str, arity: int) -> dict[str, Any]:
    """Return a detached standard-profile calibration receipt without target work."""
    return copy.deepcopy(_standard_channel(noise_id, arity).metadata)


def _inputs(
    circuit_spec: dict[str, Any], theta: ArrayLike, target_vector: ArrayLike, noise_id: str
) -> tuple[int, list[tuple[str, tuple[int, ...], float, bool]], NDArray[np.complex128]]:
    """Validate the scalar JSON domain without constructing a YAQS circuit."""
    n, count = circuit_spec.get("num_qubits"), circuit_spec.get("num_params")
    raw_theta = np.asarray(theta)
    target = np.asarray(target_vector, dtype=np.complex128)
    gates = circuit_spec.get("gates")
    if (
        type(n) is not int
        or not 1 <= n <= 6
        or type(count) is not int
        or count < 0
        or not isinstance(gates, list)
        or raw_theta.dtype.kind not in "iuf"
        or raw_theta.shape != (count,)
        or not np.all(np.isfinite(raw_theta))
        or target.shape != (2**n,)
        or not np.all(np.isfinite(target))
        or abs(np.linalg.norm(target) - 1) > PHYSICAL_TOLERANCE
        or noise_id not in PROFILE_IDS
    ):
        msg = "Oracle requires q1-q6, finite real parameters, normalized target and a fixed profile."
        raise ValueError(msg)
    resolved = []
    for gate in gates:
        if not isinstance(gate, dict):
            msg = "Each oracle circuit gate must be a scalar metadata object."
            raise TypeError(msg)
        name, sites, index = gate.get("name"), gate.get("sites"), gate.get("param_index")
        scale, offset = gate.get("angle_scale", 1.0), gate.get("angle_offset", 0.0)
        enabled = gate.get("noise_enabled", True)
        if (
            not isinstance(name, str)
            or name not in PRIMITIVES
            or not isinstance(sites, (list, tuple))
            or len(sites) != (1 if len(name) == 2 else 2)
            or any(type(site) is not int or not 0 <= site < n for site in sites)
            or len(set(sites)) != len(sites)
            or (index is not None and (type(index) is not int or not 0 <= index < count))
            or type(enabled) is not bool
            or gate.get("data_map") is not None
            or gate.get("fixed_params", ()) not in ([], ())
            or any(type(value) not in {int, float} or not math.isfinite(value) for value in (scale, offset))
        ):
            msg = "Oracle supports only finite scalar Pauli rotations with fixed one-/two-site metadata."
            raise ValueError(msg)
        angle = float(offset + (0.0 if index is None else scale * float(raw_theta[index])))
        if not math.isfinite(angle):
            msg = "Resolved physical angle is nonfinite."
            raise ValueError(msg)
        resolved.append((name, tuple(sites), angle, enabled))
    return n, resolved, target


def _ballarin_gate(
    rho: DensityMatrix, name: str, sites: tuple[int, ...], angle: float
) -> tuple[DensityMatrix, int, int]:
    """Apply literal native lowering and two independent Kraus channels.

    Returns:
        Density matrix, added basis-gate count, and retained native RZZ count.
    """
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    if abs(angle) <= 1e-4 or math.isclose(abs(angle), 1e-4, rel_tol=0.0, abs_tol=1e-12):
        return rho, 0, 0
    if name == "rxx":
        before, after = HGate(), HGate()
    else:
        before, after = RXGate(math.pi / 2), RXGate(-math.pi / 2)
    if name != "rzz":
        for site in sites:
            rho = rho.evolve(before, qargs=[site])
    rho = rho.evolve(RZZGate(angle), qargs=list(sites))
    epsilon = 2.1e-4 + 1.43e-3 * abs(angle)
    probability = (1 - math.sqrt(1 - 1.25 * epsilon)) / 3
    # Qiskit's constructor accepts a list of matrices; its annotation omits the
    # documented multi-operator list form (an ndarray means a single operator).
    kraus_data: Any = [
        math.sqrt(weight) * Pauli(label).to_matrix()
        for label, weight in zip("IXYZ", (1 - 3 * probability, probability, probability, probability), strict=True)
    ]
    kraus = Kraus(kraus_data)
    for site in sites:
        rho = rho.evolve(kraus, qargs=[site])
    if name != "rzz":
        for site in reversed(sites):
            rho = rho.evolve(after, qargs=[site])
    return rho, 4 if name != "rzz" else 0, 1


def evaluate_profile(
    circuit_spec: dict[str, Any], theta: ArrayLike, target_vector: ArrayLike, noise_id: str
) -> dict[str, Any]:
    """Independently score one physical scalar circuit from the all-zero state.

    Args:
        circuit_spec: Stored scalar circuit JSON, including every noise flag.
        theta: Real finite current parameters; fixed and shared angles are allowed.
        target_vector: Explicit normalized little-endian target; never loaded here.
        noise_id: One of the twelve immutable profile identifiers.

    Returns:
        Plain fidelity, physicality, calibration and physical/native count receipt.
        Timings include validation and any cold local calibration. They are nested
        diagnostics; the caller must charge its complete invocation separately.

    Raises:
        ValueError: If the input is outside the supported scalar domain.
        TypeError: If a gate is not a scalar metadata object.
        FloatingPointError: If a channel or final density fails physicality checks.
    """
    cpu, wall = time.process_time(), time.perf_counter()
    n, gates, target = _inputs(circuit_spec, theta, target_vector, noise_id)
    rho = DensityMatrix.from_label("0" * n)
    one = sum(len(sites) == 1 for _, sites, _, _ in gates)
    physical = {"one_qubit": one, "two_qubit": len(gates) - one, "total": len(gates)}
    evaluated = dict(physical)
    windows, pruned, basis = 0, 0, 0
    calibration = {}
    if noise_id in STANDARD_PROFILES:
        calibration = {str(arity): calibration_profile(noise_id, arity) for arity in (1, 2)}
    for name, sites, angle, enabled in gates:
        if noise_id == "ballarin_coupled" and len(sites) == 2:
            rho, added_basis, retained = _ballarin_gate(rho, name, sites, angle)
            windows += 2 * retained
            pruned += 1 - retained
            basis += added_basis
            continue
        rho = rho.evolve(PRIMITIVES[name](angle), qargs=list(sites))
        if noise_id in STANDARD_PROFILES and enabled and calibration[str(len(sites))]["generators"]:
            rho = rho.evolve(_standard_channel(noise_id, len(sites)).channel, qargs=list(sites))
            windows += 1
    if noise_id == "ballarin_coupled":
        evaluated = {"one_qubit": one + basis, "two_qubit": physical["two_qubit"] - pruned}
        evaluated["total"] = sum(evaluated.values())
    data = rho.data
    if not np.all(np.isfinite(data)):
        msg = "Independent final density contains nonfinite entries."
        raise FloatingPointError(msg)
    trace = complex(np.trace(data))
    fidelity = float(np.vdot(target, data @ target).real)
    physicality = {
        "trace_real": trace.real,
        "trace_imag": trace.imag,
        "trace_error": abs(trace - 1),
        "hermitian_max_abs": float(np.max(abs(data - data.conj().T))),
        "minimum_eigenvalue": float(np.linalg.eigvalsh((data + data.conj().T) / 2).min()),
        "norm_error": float(abs(np.linalg.norm(target) - 1)),
    }
    if (
        not all(math.isfinite(value) for value in (*physicality.values(), fidelity))
        or physicality["trace_error"] > PHYSICAL_TOLERANCE
        or physicality["hermitian_max_abs"] > PHYSICAL_TOLERANCE
        or physicality["minimum_eigenvalue"] < -PHYSICAL_TOLERANCE
        or not -PHYSICAL_TOLERANCE <= fidelity <= 1 + PHYSICAL_TOLERANCE
    ):
        msg = "Independent final density failed finite, trace, Hermitian, positive or fidelity checks."
        raise FloatingPointError(msg)
    return {
        "version": VERSION,
        "noise_id": noise_id,
        "fidelity": fidelity,
        "physicality": physicality,
        "physical_counts": physical,
        "evaluated_counts": evaluated,
        "channel_windows": windows,
        "channel_windows_scope": "full-support standard channel; two local Kraus channels per retained Ballarin RZZ",
        "ballarin_pruned_two_qubit": pruned,
        "ballarin_added_basis_gates": basis,
        "calibration": calibration,
        "exactness": "independent_Qiskit_density_SciPy_expm;floating_point;no_Monte_Carlo",
        "norm_error_scope": "input target vector Euclidean norm minus one in absolute value",
        "minimum_eigenvalue_scope": "Hermitian part (rho+rho.adjoint)/2; raw Hermitian residual checked separately",
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall,
    }
