# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Original logical gate windows, endian conventions, validation and exact checks."""

from __future__ import annotations

import hashlib
import itertools
from typing import TYPE_CHECKING, NoReturn, cast

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Statevector

from experiments.legacy_topdown_dense import matrix_score
from experiments.state_preparation_competitors import evaluation
from experiments.state_preparation_competitors import logical_evaluation as study
from experiments.state_preparation_competitors.common import Candidate, from_parameterized
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from numpy.typing import NDArray


def asymmetric_circuit() -> QuantumCircuit:
    """Return complex amplitudes and reversed asymmetric two-qubit gate supports."""
    circuit = QuantumCircuit(2, global_phase=0.37)
    circuit.rx(0.61, 1)
    circuit.cx(1, 0)
    circuit.rzx(-0.29, 1, 0)
    circuit.ry(0.43, 0)
    return circuit


def path_fidelity(circuit: QuantumCircuit, target: NDArray, hardware: str, codes: tuple[int, ...]) -> float:
    """Evaluate one complete path independently with Qiskit's ordered qargs API.

    Returns:
        Target fidelity after every original gate and the specified Pauli errors.
    """
    state = Statevector.from_label("0" * circuit.num_qubits)
    for item, code in zip(circuit.data, codes, strict=True):
        sites = [circuit.find_bit(qubit).index for qubit in item.qubits]
        _, paulis = evaluation.pauli_distribution(hardware, len(sites))
        state = state.evolve(item.operation, qargs=sites).evolve(Operator(paulis[code]), qargs=sites)
    return float(abs(np.vdot(target, state.data)) ** 2)


@pytest.mark.parametrize(("hardware", "rate"), [("IBM", 3e-4), ("Google", 6e-4)])
def test_single_gate_exact_fidelity_matches_analytic_depolarization(hardware: str, rate: float) -> None:
    """X and Y errors flip a prepared computational basis state; Z preserves it."""
    circuit = QuantumCircuit(1)
    circuit.x(0)
    target = np.array([0, 1], dtype=complex)
    result = study.score_exact(circuit, target, hardware)
    assert result["fidelity"] == pytest.approx(1 - 2 / 3 * (-np.expm1(-rate)), abs=1e-15)
    assert result["trace_error"] < 1e-15
    assert result["hermiticity_error"] < 1e-15
    assert result["minimum_eigenvalue"] >= 0


@pytest.mark.parametrize("hardware", evaluation.HARDWARE)
def test_exact_logical_model_matches_legacy_shared_parameter_rotations(hardware: str) -> None:
    """Archive parity includes offsets, negative scales and complex non-palindromic targets."""
    gates = [
        ParameterizedGate("rx", (2,), 0, angle_scale=-0.7, angle_offset=0.21),
        ParameterizedGate("ry", (0,), 1),
        ParameterizedGate("rz", (1,), None, angle_offset=-0.4),
        ParameterizedGate("rxx", (2, 1), 0),
        ParameterizedGate("ryy", (1, 0), 2, angle_scale=0.6),
        ParameterizedGate("rzz", (0, 1), 1, angle_offset=0.13),
    ]
    circuit = ParameterizedCircuit(3, gates, 3)
    theta = np.array([0.8, -0.37, 1.1])
    target = np.array([1, 2j, -3, 4 + 1j, -2j, 1 - 2j, 0.3j, 0.5])
    target /= np.linalg.norm(target)
    logical = from_parameterized(circuit, theta)
    expected = matrix_score(circuit, theta, target, hardware, "tjm")
    actual = study.score_exact(logical, target, hardware)
    assert actual == pytest.approx(expected, abs=1e-14)
    assert study.logical_resources(logical, target)["ideal_fidelity"] == pytest.approx(
        matrix_score(circuit, theta, target, hardware, "noiseless")["fidelity"], abs=1e-14
    )


@pytest.mark.parametrize("hardware", evaluation.HARDWARE)
def test_exact_score_matches_exhaustive_asymmetric_qiskit_paths(hardware: str) -> None:
    """Enumerate every categorical outcome independently of the density superoperator."""
    circuit = QuantumCircuit(2)
    circuit.rx(0.61, 1)
    circuit.rzx(-0.29, 1, 0)
    target = np.array([1, 2j, -3, 4j]) / np.sqrt(30)
    distributions = [evaluation.pauli_distribution(hardware, len(item.qubits))[0] for item in circuit.data]
    expected = 0.0
    for codes in itertools.product(*(range(len(p)) for p in distributions)):
        probability = float(np.prod([p[code] for p, code in zip(distributions, codes, strict=True)]))
        expected += probability * path_fidelity(circuit, target, hardware, codes)
    assert study.score_exact(circuit, target, hardware)["fidelity"] == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("hardware", evaluation.HARDWARE)
def test_empirical_compression_matches_raw_paths_and_standard_error(hardware: str) -> None:
    """Five duplicate no-jump paths retain weight five and do not inflate the sample SE."""
    circuit = asymmetric_circuit()
    target = np.array([1, 2j, -3, 4j]) / np.sqrt(30)
    codes = np.array([[0, 0, 0, 0]] * 5 + [[1, 0, 0, 0], [0, 2, 3, 1], [0, 8, 0, 2]], dtype=np.uint8)
    original = codes.copy()
    result = study.validation_from_paths(circuit, target, hardware, codes)
    raw = [path_fidelity(circuit, target, hardware, tuple(row)) for row in codes]
    assert result["fidelity"] == pytest.approx(np.mean(raw), abs=1e-14)
    assert result["standard_error"] == pytest.approx(np.std(raw, ddof=1) / np.sqrt(len(raw)), abs=1e-14)
    assert result["no_jump_paths"] == 5
    assert result["jump_paths"] == 3
    assert result["retained_paths"] == 4
    assert result["norm_max_error"] < 1e-14
    assert result["bank_sha256"] == hashlib.sha256(codes.tobytes()).hexdigest()
    expected_no_jump = np.prod([
        evaluation.pauli_distribution(hardware, len(item.qubits))[0][0] for item in circuit.data
    ])
    assert result["analytic_no_jump_probability"] == pytest.approx(expected_no_jump)
    np.testing.assert_array_equal(codes, original)


@pytest.mark.parametrize("hardware", evaluation.HARDWARE)
def test_sampling_replays_independent_streams_and_covers_actual_jumps(hardware: str) -> None:
    """A fixed validation stream replays, while a role-separated stream has a different bank."""
    circuit = asymmetric_circuit()
    target = Statevector.from_instruction(circuit).data
    entropy = [20260920, 6, 300, 90000]
    codes = study.sample_paths(circuit, hardware, 8192, entropy)
    np.testing.assert_array_equal(codes, study.sample_paths(circuit, hardware, 8192, entropy))
    assert not np.array_equal(codes, study.sample_paths(circuit, hardware, 8192, [20260920, 6, 300, 89999]))
    result = study.score_validation(circuit, target, hardware, 8192, entropy)
    assert result["bank_sha256"] == hashlib.sha256(codes.tobytes()).hexdigest()
    assert result["entropy"] == entropy
    assert result["entropy"] is not entropy
    assert 0 < result["jump_paths"] < 8192
    assert result["sampled_paths"] == 8192
    assert 0 <= result["fidelity"] <= 1
    exact = study.score_exact(circuit, target, hardware)["fidelity"]
    assert abs(result["fidelity"] - exact) < 6 * result["standard_error"]


def test_validation_never_calls_exact_final_assessment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Construction/selection can use validation without exposing the sealed final oracle."""

    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        pytest.fail("Validation must not compute exact final fidelity.")

    monkeypatch.setattr(study, "score_exact", forbidden)
    circuit = QuantumCircuit(1)
    circuit.ry(0.3, 0)
    result = study.score_validation(circuit, np.array([1, 0]), "IBM", 100, [5])
    assert "exact" not in result
    assert result["sampled_paths"] == 100


def test_empty_and_zero_angle_schedules_preserve_physical_noise_windows() -> None:
    """Zero angles and cancelling gates remain noisy; only a truly empty circuit has none."""
    target = np.array([1, 0, 0, 0], dtype=complex)
    empty = QuantumCircuit(2)
    result = study.score_validation(empty, target, "IBM", 8192, [7])
    assert result["fidelity"] == 1
    assert result["standard_error"] == 0
    assert result["jump_paths"] == 0
    assert result["retained_paths"] == 1
    assert result["analytic_no_jump_probability"] == 1
    assert study.score_exact(empty, target, "IBM")["fidelity"] == 1
    circuit = QuantumCircuit(2)
    circuit.rxx(0, 1, 0)
    circuit.cx(1, 0)
    circuit.cx(1, 0)
    circuit.id(0)
    before = circuit.copy()
    resources = study.logical_resources(circuit, target)
    assert resources["ideal_fidelity"] == 1
    assert resources["source_gate_count"] == 4
    assert resources["logical_one_qubit"] == 1
    assert resources["logical_two_qubit"] == 3
    assert resources["logical_two_qubit_depth"] == 3
    assert resources["logical_total_depth"] == 4
    assert resources["logical_operations"] == {"cx": 2, "rxx": 1, "id": 1}
    assert resources["compilation_performed"] is False
    assert resources["noise_basis"] == "logical"
    assert study.score_exact(circuit, target, "IBM")["fidelity"] < 1
    no_jump = study.validation_from_paths(circuit, target, "IBM", np.zeros((100, 4), dtype=np.uint8))
    assert no_jump["jump_paths"] == 0
    assert no_jump["retained_paths"] == 1
    assert no_jump["fidelity"] == 1
    assert circuit == before


def test_all_jump_bank_keeps_each_path_without_inserting_a_no_jump_trajectory() -> None:
    """Compression cannot synthesize an absent event or replace empirical proportions."""
    circuit = QuantumCircuit(1)
    circuit.x(0)
    result = study.validation_from_paths(circuit, np.array([0, 1]), "IBM", np.array([[1], [2], [3]]))
    assert result["no_jump_paths"] == 0
    assert result["retained_paths"] == 3
    assert result["jump_paths"] == 3
    assert result["fidelity"] == pytest.approx(1 / 3)
    assert result["standard_error"] == pytest.approx(1 / 3)


def test_compiled_native_noise_differs_from_the_same_ideal_logical_circuit() -> None:
    """Native decomposition preserves ideal states but creates additional noise windows."""
    circuit = QuantumCircuit(2)
    circuit.ry(0.41, 0)
    circuit.rxx(1.13, 1, 0)
    target = Statevector.from_instruction(circuit).data
    native = evaluation.compile_candidate(Candidate(circuit, {}), target)
    logical = study.logical_resources(circuit, target)
    assert logical["ideal_fidelity"] == pytest.approx(native.metadata["ideal_fidelity"], abs=1e-14)
    assert logical["logical_two_qubit"] == 1
    assert native.metadata["compiled_two_qubit"] == 2
    assert study.score_exact(circuit, target, "IBM")["fidelity"] > evaluation.exact_score(native, "IBM")["fidelity"]


@pytest.mark.parametrize("gate", sorted(study.ONE_QUBIT_GATES | study.TWO_QUBIT_GATES))
def test_supported_elementary_gate_alphabet_preserves_each_window(gate: str) -> None:
    """All advertised scalar rotations and fixed gates have explicit one/two-site channels."""
    circuit = QuantumCircuit(2)
    args: list[int | float] = [0.23] if gate.startswith("r") else []
    args += [0] if gate in study.ONE_QUBIT_GATES else [1, 0]
    getattr(circuit, gate)(*args)
    target = Statevector.from_instruction(circuit).data
    assert study.logical_resources(circuit, target)["source_gate_count"] == 1
    assert 0 <= study.score_exact(circuit, target, "IBM")["fidelity"] <= 1 + 1e-14


@pytest.mark.parametrize(
    ("paths", "entropy"),
    [(1, [1]), (True, [1]), (2.1, [1]), (2, []), (2, [-1]), (2, [True]), (2, [1.1]), (2, None), (2, (1,))],
)
def test_invalid_sample_counts_and_entropy_are_rejected(paths: int, entropy: list[int]) -> None:
    """The stream declaration is explicit and never silently coerces malformed seed data."""
    with pytest.raises(ValueError, match="Validation needs"):
        study.sample_paths(QuantumCircuit(1), "IBM", paths, entropy)


@pytest.mark.parametrize(
    "codes",
    [
        np.zeros((2, 2), dtype=int),
        np.zeros((1, 1), dtype=int),
        np.zeros(2, dtype=int),
        np.array([[-1], [0]]),
        np.array([[0.1], [0.0]]),
        np.array([[False], [True]]),
        np.array([[4], [0]]),
        np.array([[256], [0]], dtype=np.uint64),
    ],
)
def test_malformed_or_out_of_range_path_banks_fail_closed(codes: NDArray) -> None:
    """Codes are validated before hashing to uint8, avoiding wraparound of invalid values."""
    circuit = QuantumCircuit(1)
    circuit.ry(0.3, 0)
    with pytest.raises(ValueError, match=r"bank|Pauli code"):
        study.validation_from_paths(circuit, np.array([1, 0]), "IBM", codes)


def test_invalid_circuit_gate_layout_target_and_hardware_inputs() -> None:
    """No routing, ancillary register, opaque block or nonunitary instruction can enter the study."""
    with pytest.raises(TypeError, match="Qiskit logical circuit"):
        study.validate_logical_circuit(cast("QuantumCircuit", None))
    invalid = QuantumCircuit(1)
    invalid.reset(0)
    with pytest.raises(TypeError, match="Measurement, reset"):
        study.validate_logical_circuit(invalid)
    three = QuantumCircuit(3)
    three.ccx(0, 1, 2)
    nonlocal_gate = QuantumCircuit(3)
    nonlocal_gate.cx(0, 2)
    opaque = QuantumCircuit(1)
    opaque.unitary(np.eye(2), [0])
    for circuit in (three, nonlocal_gate, opaque):
        with pytest.raises(ValueError, match="elementary logical"):
            study.validate_logical_circuit(circuit)
    parameterized = QuantumCircuit(1)
    parameterized.ry(Parameter("angle"), 0)
    nonunitary = QuantumCircuit(1)
    nonunitary.append(UnitaryGate(np.diag([1, 2]), check_input=False), [0])
    classical = QuantumCircuit(1, 1)
    for circuit in (parameterized, nonunitary, classical, QuantumCircuit(0), QuantumCircuit(13)):
        with pytest.raises(ValueError, match=r"resolved unitary|finite and unitary"):
            study.validate_logical_circuit(circuit)
    logical = QuantumCircuit(2)
    logical.cx(0, 1)
    layout = transpile(logical, basis_gates=["cx", "rz", "sx", "x"], initial_layout=[0, 1])
    with pytest.raises(ValueError, match="transpiler layout"):
        study.validate_logical_circuit(layout)
    with pytest.raises(ValueError, match="register size"):
        study.logical_resources(logical, np.array([1, 0]))
    with pytest.raises(ValueError, match="normalized finite"):
        study.score_exact(QuantumCircuit(1), np.array([1, 1]), "IBM")
    with pytest.raises(ValueError, match="IBM/Google profile"):
        study.score_exact(QuantumCircuit(1), np.array([1, 0]), "unknown")
    with pytest.raises(ValueError, match="IBM/Google profile"):
        study.sample_paths(QuantumCircuit(1), "unknown", 2, [1])
    with pytest.raises(ValueError, match="IBM/Google profile"):
        study.validation_from_paths(QuantumCircuit(1), np.array([1, 0]), "unknown", np.zeros((2, 0), dtype=int))
