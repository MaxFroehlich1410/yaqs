# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Adapt published AQCE to Qiskit using pinned MNISQ tensor/gate primitives.

Local orchestration follows Algorithms 1-5 of the original published paper;
the separately licensed upstream resource supplies unchanged tensor primitives.
One sweep comprises forward and backward passes. No noise training is added.
See AQCE_PROVENANCE.md for attribution and reproducibility limitations.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import itertools
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from experiments.state_preparation_competitors import aqce_schedule
from experiments.state_preparation_competitors.common import Candidate, validate_target

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import ArrayLike

UPSTREAM_COMMIT = "c0b6d860477c4dcfb2737e6e6c996a4854858d5d"
UPSTREAM_URL = "https://github.com/FujiiLabCollaboration/MNISQ-quantum-circuit-dataset"
ALGORITHM = "aqce_published_2024_algorithms_1_to_5"
UPSTREAM_DIRECTORY = Path(__file__).parent / "vendor" / "mnisq_aqce"
UPSTREAM_HASHES = {
    "AQCE_from_python.py.txt": "2ca009f6001ead4dd50ff5e875704329ce76384547ba1929cf538f457895a57b",
    "LICENSE.txt": "2003829cf643d61b00a76088dd31bb11b8029546f24c56b39fd492adb9281483",
}


def load_upstream() -> ModuleType:
    """Verify and load the pinned, separately licensed AQCE Python resource.

    Returns:
        The unmodified upstream module, requiring NumPy and Qulacs.

    Raises:
        RuntimeError: If a vendored file differs from the recorded upstream.
        ImportError: If the source cannot be loaded or Qulacs is unavailable.
    """
    for filename, expected in UPSTREAM_HASHES.items():
        if hashlib.sha256((UPSTREAM_DIRECTORY / filename).read_bytes()).hexdigest() != expected:
            msg = f"AQCE upstream integrity check failed for {filename}"
            raise RuntimeError(msg)
    source = UPSTREAM_DIRECTORY / "AQCE_from_python.py.txt"
    loader = importlib.machinery.SourceFileLoader("_yaqs_mnisq_aqce", str(source))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        msg = "Cannot load the pinned MNISQ AQCE source"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def to_qiskit(gates: list[Any], num_qubits: int) -> QuantumCircuit:
    """Preserve Qulacs gate order and little-endian target ordering in Qiskit.

    Args:
        gates: Upstream Qulacs gates in execution order.
        num_qubits: Number of qubits in the target state.

    Returns:
        A circuit containing the original full two-qubit unitaries.
    """
    circuit = QuantumCircuit(num_qubits)
    for gate in gates:
        circuit.unitary(np.asarray(gate.get_matrix()), gate.get_target_index_list(), label="AQCE")
    return circuit


def aqce_candidates(
    target: ArrayLike,
    *,
    gate_budgets: tuple[int, ...],
    sweeps: int = 100,
    initial_gates: int = 1,
    growth_increment: int = 1,
    seed: int = 300,
    connectivity: str = "all_to_all",
) -> list[Candidate]:
    """Run published AQCE independently at each requested gate budget.

    Initialize M0 gates once, then alternate full forward/backward sweeps and
    identity-block growth. Growth updates only the new gates backwards against
    the residual target. Each SVD update searches every qubit pair. Independent
    budget restarts and their full CPU cost are reported explicitly.

    Args:
        target: A normalized finite complex statevector of one to six qubits.
        gate_budgets: Nonempty, strictly increasing nonnegative gate counts.
        sweeps: N full forward/backward sweeps after initialization and growth.
        initial_gates: M0 for positive budgets; zero budget remains an empty control.
        growth_increment: Positive delta M. Each positive budget must be at least
            M0 and reachable exactly using this increment.
        seed: Recorded for the common interface; this deterministic method
            does not use randomness and should not receive seed replicas.
        connectivity: Only ``all_to_all`` is supported by this benchmark.
            Hardware routing belongs to the common compilation stage.

    Returns:
        Independently constructed circuit candidates and timing/provenance.

    Raises:
        ValueError: If controls are invalid, the target fails validation, or a
            positive two-qubit gate budget is requested for one qubit.
    """
    setup_start = time.process_time()
    vector = validate_target(target)
    if (
        not isinstance(gate_budgets, tuple)
        or not gate_budgets
        or any(isinstance(budget, bool) or not isinstance(budget, int) or budget < 0 for budget in gate_budgets)
        or any(right <= left for left, right in itertools.pairwise(gate_budgets))
    ):
        msg = "gate_budgets must be a nonempty, strictly increasing tuple of nonnegative integers"
        raise ValueError(msg)
    for label, value in (
        ("sweeps", sweeps),
        ("seed", seed),
        ("initial_gates", initial_gates),
        ("growth_increment", growth_increment),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            msg = f"{label} must be a nonnegative integer"
            raise ValueError(msg)
    if growth_increment == 0 or any(
        budget and (budget < initial_gates or (budget - initial_gates) % growth_increment) for budget in gate_budgets
    ):
        msg = "AQCE positive budgets must be reachable from initial_gates with a positive growth_increment"
        raise ValueError(msg)
    if connectivity != "all_to_all":
        msg = "This AQCE benchmark supports only all_to_all connectivity"
        raise ValueError(msg)
    num_qubits = vector.size.bit_length() - 1
    if num_qubits == 1 and gate_budgets[-1] > 0:
        msg = "AQCE requires at least two qubits for a positive two-qubit gate budget"
        raise ValueError(msg)
    upstream = load_upstream()
    setup_cpu = time.process_time() - setup_start
    cumulative_cpu = 0.0
    candidates = []
    for budget in gate_budgets:
        start = time.process_time()
        state = upstream.QuantumState(num_qubits)
        state.load(vector)
        actual_initial = initial_gates if budget else 0
        gates = aqce_schedule.encode(
            upstream,
            state,
            initial_gates=actual_initial,
            growth_increment=growth_increment,
            max_gates=budget,
            sweeps=sweeps,
        )
        circuit = to_qiskit(gates, num_qubits)
        fidelity = float(abs(np.vdot(vector, Statevector.from_instruction(circuit).data)) ** 2)
        attributed_setup_cpu = setup_cpu if not candidates else 0.0
        cpu = time.process_time() - start + attributed_setup_cpu
        cumulative_cpu += cpu
        sweep_updates = 2 * sweeps * sum(range(actual_initial, budget + 1, growth_increment)) if budget else 0
        growth_updates = budget - actual_initial
        metadata = {
            "method": "aqce",
            "algorithm": ALGORITHM,
            "paper_doi": "10.1103/PhysRevResearch.6.043008",
            "schedule_source_sha256": hashlib.sha256(Path(aqce_schedule.__file__).read_bytes()).hexdigest(),
            "candidate_id": f"gates_{budget:03d}",
            "status": "complete",
            "deterministic": True,
            "seed": seed,
            "seed_used": False,
            "gate_budget": budget,
            "logical_two_qubit_gates": len(gates),
            "ideal_fidelity": fidelity,
            "connectivity": connectivity,
            "sweeps": sweeps,
            "sweep_definition": "forward_then_backward",
            "initial_gates": actual_initial,
            "requested_initial_gates": initial_gates,
            "initialization": "algorithm_4_density_eigenvectors_once",
            "growth_increment": growth_increment,
            "growth": "algorithm_5_identity_block_backward_on_residual",
            "budget_reuse": "independent_restarts",
            "scheduled_initialization_steps": actual_initial,
            "scheduled_sweep_updates": sweep_updates,
            "scheduled_growth_updates": growth_updates,
            "scheduled_unitary_updates": sweep_updates + growth_updates,
            "setup_cpu_seconds": attributed_setup_cpu,
            "cpu_seconds": cpu,
            "cumulative_cpu_seconds": cumulative_cpu,
            "cpu_scope": "setup_once_plus_all_budget_restarts_including_conversion_and_ideal_fidelity",
            "upstream_url": UPSTREAM_URL,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_reuse": "unchanged_evaluate_F_evaluate_Rho_and_qulacs_primitives",
            "upstream_source_sha256": UPSTREAM_HASHES["AQCE_from_python.py.txt"],
            "upstream_license": "CC-BY-SA-4.0",
            "qulacs_version": importlib.metadata.version("qulacs"),
            "noise_fine_tuning": False,
        }
        candidates.append(Candidate(circuit=circuit, metadata=metadata))
    return candidates
