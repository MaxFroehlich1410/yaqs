# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent Qiskit density validation of the final-circuit MPO evaluator.

Dense construction is deliberately confined to this diagnostic and at most
12 qubits. No optimization, sampling, or twenty-qubit experiment is performed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from threadpoolctl import threadpool_limits

from benchmarks.state_preparation import load_target
from experiments import legacy_topdown_dense as legacy
from experiments import topdown_backend_support as support
from experiments import topdown_mpo_evaluation as evaluator
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.mps import MPS


ANGLE_GATES = frozenset({"rx", "ry", "rz", "rxx", "ryy", "rzz", "p", "cp"})
FIXED_GATES = frozenset({"h", "x", "y", "z", "cx", "cy", "cz", "swap", "id"})


def reference_density(circuit: ParameterizedCircuit, theta: NDArray, hardware: str) -> NDArray:
    """Propagate a dense density using independent Qiskit gates and channel matrices.

    Returns:
        Exact channel-averaged density in little-endian order.

    Raises:
        ValueError: If the diagnostic width, gates, or profile is unsupported.
    """
    if not 1 <= circuit.num_qubits <= 12 or hardware not in {"IBM", "Google", "noiseless"}:
        msg = "Dense reference requires one to twelve qubits and a declared profile."
        raise ValueError(msg)
    density = DensityMatrix.from_label("0" * circuit.num_qubits)
    for gate in circuit.gates:
        quantum_circuit = QuantumCircuit(circuit.num_qubits)
        sites = list(gate.sites)
        if gate.data_map is not None:
            msg = "Reference requires data-free gates."
            raise ValueError(msg)
        if gate.name in ANGLE_GATES:
            angle = gate.angle_offset + (
                gate.angle_scale * theta[gate.param_index] if gate.param_index is not None else 0
            )
            getattr(quantum_circuit, gate.name)(angle, *sites)
        elif gate.name in FIXED_GATES:
            getattr(quantum_circuit, gate.name)(*sites)
        else:
            msg = f"Unsupported independent reference gate: {gate.name}."
            raise ValueError(msg)
        density = density.evolve(quantum_circuit)
        if hardware != "noiseless" and gate.noise_enabled:
            density = density.evolve(legacy.noise_channel(hardware, len(sites), "tjm"), qargs=sites)
    return np.asarray(density.data)


def dense_mpo(state: evaluator.ForwardMPO) -> NDArray:
    """Densify a small validation MPO and explicitly restore little-endian order.

    Returns:
        Density matrix with amplitude index bit i corresponding to site i.

    Raises:
        ValueError: If dense conversion is requested above twelve qubits.
    """
    count = len(state.tensors)
    if count > 12:
        msg = "Dense MPO conversion is confined to at most twelve validation qubits."
        raise ValueError(msg)
    matrix = state.to_mpo().to_matrix()
    permutation = (*reversed(range(count)), *reversed(range(count, 2 * count)))
    return matrix.reshape((2,) * (2 * count)).transpose(permutation).reshape(2**count, 2**count)


def compare_density(
    circuit: ParameterizedCircuit,
    theta: NDArray,
    target: MPS,
    hardware: str,
    settings: evaluator.MPOSettings = evaluator.DEFAULT_SETTINGS,
) -> dict:
    """Compare full density, fidelity, trace, and Hermiticity against independent evolution.

    Returns:
        Numerical validation result, including actual compression diagnostics.
    """
    if circuit.num_qubits > 12:
        msg = "Dense validation is limited to twelve qubits."
        raise ValueError(msg)
    start = time.perf_counter()
    state = evaluator.propagate(circuit, theta, hardware, settings)
    fidelity = state.fidelity(target)
    diagnostics = state.diagnostics()
    print(
        f"MPO q{circuit.num_qubits} {hardware}: bond {diagnostics['maximum_bond']}; starting density reference",
        flush=True,
    )
    reference = reference_density(circuit, theta, hardware)
    vector = target.to_vec()
    expected = float(np.vdot(vector, reference @ vector).real)
    matrix = dense_mpo(state)
    frobenius_error = float(np.linalg.norm(matrix - reference))
    hermiticity_error = float(np.max(np.abs(matrix - matrix.conj().T)))
    reference_trace_error = float(abs(np.trace(reference) - 1))
    bound = diagnostics["fidelity_absolute_error_bound_excluding_roundoff"]
    tolerance = 1e-10
    passed = (
        frobenius_error <= bound + tolerance
        and abs(fidelity - expected) <= bound + tolerance
        and reference_trace_error < tolerance
        and diagnostics["trace_error"] <= np.sqrt(2**circuit.num_qubits) * bound + tolerance
        and hermiticity_error <= 2 * bound + tolerance
    )
    return {
        "num_qubits": circuit.num_qubits,
        "gate_count": len(circuit.gates),
        "hardware": hardware,
        "settings": as_settings(settings),
        "mpo_fidelity": fidelity,
        "density_fidelity": expected,
        "fidelity_absolute_error": abs(fidelity - expected),
        "density_frobenius_error": frobenius_error,
        "mpo_hermiticity_max_error": hermiticity_error,
        "reference_trace_error": reference_trace_error,
        "reference_minimum_eigenvalue": float(np.linalg.eigvalsh(reference).min()) if circuit.num_qubits <= 6 else None,
        "positivity_validation": "full spectrum at q6; CPTP reference and full-density comparison at q12",
        "diagnostics": diagnostics,
        "passed": passed,
        "roundoff_allowance": tolerance,
        "wall_seconds": time.perf_counter() - start,
    }


def as_settings(settings: evaluator.MPOSettings) -> dict:
    """Return JSON-compatible numerical settings."""
    return {
        "max_bond_dim": settings.max_bond_dim,
        "svd_atol": settings.svd_atol,
        "max_workspace_bytes": settings.max_workspace_bytes,
    }


def validation_circuit(count: int) -> tuple[ParameterizedCircuit, NDArray]:
    """Build a seeded register-wide circuit covering phases, correlated noise, and reversed gates.

    Returns:
        Fixed circuit and parameters, with zero angles and disabled noise included.
    """
    gates = [ParameterizedGate("ry", (site,), param_index=site) for site in range(count)]
    gates.extend(
        ParameterizedGate(("rxx", "ryy", "rzz")[site % 3], (site, site + 1), param_index=count + site)
        for site in range(count - 1)
    )
    gates.extend([
        ParameterizedGate("rx", (count - 1,), param_index=0, angle_scale=-0.7, angle_offset=0.2),
        ParameterizedGate("rz", (0,), angle_offset=0, noise_enabled=True),
        ParameterizedGate("cx", (count - 1, count - 2), noise_enabled=False),
        ParameterizedGate("h", (0,), noise_enabled=False),
        ParameterizedGate("rxx", (1, 0), angle_offset=0.13),
    ])
    return ParameterizedCircuit(count, gates), np.random.default_rng(20261019 + count).normal(size=2 * count - 1)


def main(argv: list[str] | None = None) -> None:
    """Run reproducible q6/q12 density checks or assess a supplied frozen checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-qubits", type=int, action="append", choices=(6, 12))
    parser.add_argument("--hardware", action="append", choices=("IBM", "Google", "noiseless"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--circuit-checkpoint", type=Path)
    parser.add_argument("--target-id", default="tfim_critical")
    parser.add_argument("--max-bond-dim", type=int)
    parser.add_argument("--svd-atol", type=float, default=0)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("validation output already exists; use a fresh path")
    source_hashes = {
        "evaluator_source_sha256": support.digest(Path(evaluator.__file__)),
        "reference_source_sha256": support.digest(Path(__file__)),
    }
    rows = []
    provenance = {}
    with threadpool_limits(limits=1):
        if args.checkpoint:
            parameters = support.load_checkpoint(args.checkpoint)
            circuit_saved = support.load_checkpoint(args.circuit_checkpoint or args.checkpoint)
            if parameters is None or circuit_saved is None:
                parser.error("validation requires sealed checkpoints")
            record = circuit_saved[0]
            circuit = support.restore_circuit(record.get("circuit", record.get("training", {}).get("circuit")))
            cases = [(circuit, parameters[1]["theta"])]
            provenance = {
                "parameters_seal": support.digest(args.checkpoint / "complete.json"),
                "circuit_seal": support.digest((args.circuit_checkpoint or args.checkpoint) / "complete.json"),
            }
        else:
            cases = [validation_circuit(count) for count in (args.num_qubits or (6, 12))]
        profiles = args.hardware or ("IBM", "Google", "noiseless")
        expected_cases = len(cases) * len(profiles)
        for circuit, theta in cases:
            if not 1 <= circuit.num_qubits <= 12:
                parser.error("dense validation is limited to twelve qubits, before any target factorization")
            target = load_target(circuit.num_qubits, args.target_id).to_mps()
            for hardware in profiles:
                result = compare_density(
                    circuit, theta, target, hardware, evaluator.MPOSettings(args.max_bond_dim, args.svd_atol)
                )
                rows.append(result)
                print(
                    json.dumps({
                        "qubits": circuit.num_qubits,
                        "hardware": hardware,
                        "passed": result["passed"],
                        "error": result["fidelity_absolute_error"],
                    }),
                    flush=True,
                )
                support.write_json(
                    args.output,
                    {
                        "version": evaluator.VERSION,
                        "target_id": args.target_id,
                        "provenance": provenance,
                        "rows": rows,
                        "status": "complete" if len(rows) == expected_cases else "incomplete",
                        "expected_cases": expected_cases,
                        "completed_cases": len(rows),
                        "passed": len(rows) == expected_cases and all(row["passed"] for row in rows),
                        **source_hashes,
                    },
                )
    if not all(row["passed"] for row in rows):
        msg = "MPO/density validation failed."
        raise RuntimeError(msg)


if __name__ == "__main__":
    main()
