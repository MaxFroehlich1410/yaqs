# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent layout, channel and trajectory checks for every preparation method."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Statevector, SuperOp

from experiments.legacy_topdown_dense import noise_channel
from experiments.state_preparation_competitors import evaluation as study
from experiments.state_preparation_competitors.common import Candidate


def test_compile_routes_nonlocal_gates_and_preserves_complex_target() -> None:
    """A real SWAP changes output order; evaluating an unpermuted target would be wrong."""
    source = QuantumCircuit(3)
    source.h(0)
    source.s(0)
    source.ry(0.31, 1)
    source.cx(0, 2)
    target = Statevector.from_instruction(source).data
    saved = source.copy()
    result = study.compile_candidate(Candidate(source, {}), target)
    assert source == saved
    assert result.metadata["logical_to_physical_output"] != [0, 1, 2]
    assert result.metadata["compiled_two_qubit"] >= 4
    assert result.metadata["compiled_two_qubit_depth"] >= 4
    assert result.metadata["ideal_unitary_error"] < 1e-12
    assert result.metadata["ideal_fidelity"] == pytest.approx(1)
    assert study.exact_score(result, "IBM", law="noiseless")["fidelity"] == pytest.approx(1)
    raw_state = Statevector.from_instruction(result.circuit).data
    assert abs(np.vdot(target, raw_state)) ** 2 < 0.99
    replay = study.compile_candidate(Candidate(source, {}), target)
    assert result.circuit == replay.circuit
    np.testing.assert_array_equal(result.target, replay.target)


def test_compilation_of_arbitrary_unitary_and_single_qubit() -> None:
    """Universal blocks get charged as native operations rather than opaque noise windows."""
    rng = np.random.default_rng(32)
    matrix = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    source = QuantumCircuit(2)
    source.unitary(matrix, [1, 0])
    target = Statevector.from_instruction(source).data
    result = study.compile_candidate(Candidate(source, {}), target)
    assert result.metadata["compiled_two_qubit"] == 3
    assert study.exact_score(result, "Google")["fidelity"] < 1
    one = QuantumCircuit(1)
    one.h(0)
    one.s(0)
    result = study.compile_candidate(Candidate(one, {}), np.array([1, 1j]) / np.sqrt(2))
    assert result.metadata["ideal_fidelity"] == pytest.approx(1)


def test_physical_target_all_permutations_match_basis_relabeling() -> None:
    """Check every three-wire permutation on every computational basis state."""
    for permutation in itertools.permutations(range(3)):
        for original in range(8):
            target = np.eye(8, dtype=complex)[original]
            output = study.physical_target(target, list(permutation))
            bits = [int(x) for x in f"{original:03b}"[::-1]]
            mapped = [bits[permutation.index(i)] for i in range(3)]
            expected = int("".join(str(x) for x in reversed(mapped)), 2)
            np.testing.assert_array_equal(output, np.eye(8)[expected])
    for invalid in ([0], [0, 0], [0, 2], [0, True]):
        with pytest.raises(ValueError, match="permutation"):
            study.physical_target(np.array([1, 0, 0, 0]), invalid)


@pytest.mark.parametrize("hardware", study.HARDWARE)
@pytest.mark.parametrize("arity", [1, 2])
def test_native_pauli_mixture_equals_independent_legacy_channel(hardware: str, arity: int) -> None:
    """The new gate placement keeps precisely the existing channel coefficients/law."""
    probabilities, matrices = study.pauli_distribution(hardware, arity)
    reconstructed = sum(p * SuperOp(Operator(matrix)).data for p, matrix in zip(probabilities, matrices, strict=True))
    np.testing.assert_allclose(reconstructed, noise_channel(hardware, arity).data, atol=1e-15)
    assert probabilities.sum() == pytest.approx(1)
    assert not probabilities.flags["WRITEABLE"]
    assert all(not matrix.flags["WRITEABLE"] for matrix in matrices)


@pytest.mark.parametrize("hardware", study.HARDWARE)
def test_compressed_bank_matches_independent_qiskit_trajectory_enumeration(hardware: str) -> None:
    """CNOT orientation and complex phases are checked against Qiskit path-by-path."""
    circuit = QuantumCircuit(2)
    circuit.sx(1)
    circuit.rz(0.4, 0)
    circuit.cx(1, 0)
    circuit.sx(0)
    target = np.array([1, 2j, -3, 4j]) / np.sqrt(30)
    candidate = study.CompiledCandidate(circuit, target, {})
    codes = np.array([[0, 0, 0, 0]] * 5 + [[1, 0, 0, 0], [0, 2, 3, 1], [0, 0, 8, 0]], dtype=np.uint8)
    actual = study.validation_score(candidate, hardware, codes)
    values = []
    for row in codes:
        state = Statevector.from_label("00")
        for item, code in zip(circuit.data, row, strict=True):
            sites = [circuit.find_bit(qubit).index for qubit in item.qubits]
            state = state.evolve(item.operation, qargs=sites)
            _, paulis = study.pauli_distribution(hardware, len(sites))
            state = state.evolve(paulis[code], qargs=sites)
        values.append(abs(np.vdot(target, state.data)) ** 2)
    assert actual["fidelity"] == pytest.approx(np.mean(values), abs=1e-14)
    assert actual["standard_error"] == pytest.approx(np.std(values, ddof=1) / np.sqrt(len(values)), abs=1e-14)
    assert actual["jump_paths"] == 3
    assert actual["retained_paths"] == 4
    assert actual["norm_max_error"] < 1e-14
    bank = study.sample_validation(circuit, hardware, 8192, [20260920, 1, 90000])
    np.testing.assert_array_equal(bank, study.sample_validation(circuit, hardware, 8192, [20260920, 1, 90000]))
    assert not np.array_equal(bank, study.sample_validation(circuit, hardware, 8192, [20260920, 2, 90000]))
    sampled = study.validation_score(candidate, hardware, bank)
    exact = study.exact_score(candidate, hardware)
    assert sampled["jump_paths"] > 0
    assert abs(sampled["fidelity"] - exact["fidelity"]) < 6 * sampled["standard_error"]
    assert study.exact_score(candidate, hardware, law="lindblad")["trace_error"] < 1e-12


def test_empty_circuit_has_exact_fidelity_and_no_invented_jumps() -> None:
    """No-noise circuits are valid endpoints and do not require synthetic jump events."""
    candidate = study.compile_candidate(Candidate(QuantumCircuit(2), {}), np.array([1, 1j, 0, 0]) / np.sqrt(2))
    bank = study.sample_validation(candidate.circuit, "IBM", 100, [77])
    result = study.validation_score(candidate, "IBM", bank)
    assert result["fidelity"] == pytest.approx(0.5)
    assert result["standard_error"] == 0
    assert result["jump_paths"] == 0
    assert result["retained_paths"] == 1
    assert result["analytic_no_jump_probability"] == 1
    assert study.exact_score(candidate, "IBM")["fidelity"] == pytest.approx(0.5)


def test_invalid_circuits_profiles_banks_and_registers_fail_closed() -> None:
    """Nonunitary operations, invalid labels and malformed codes cannot enter a ranking."""
    target = np.array([1, 0], dtype=complex)
    reset = QuantumCircuit(1)
    reset.reset(0)
    measured = QuantumCircuit(1, 1)
    measured.measure(0, 0)
    parameterized = QuantumCircuit(1)
    parameterized.rx(Parameter("angle"), 0)
    nonunitary = QuantumCircuit(1)
    nonunitary.append(UnitaryGate(np.diag([1, 2]), check_input=False), [0])
    with pytest.raises(TypeError, match="Measurement, reset"):
        study.validate_circuit(reset)
    for invalid in (measured, parameterized, nonunitary, QuantumCircuit(7)):
        with pytest.raises(ValueError, match=r"resolved unitary|finite and unitary"):
            study.validate_circuit(invalid)
    native = QuantumCircuit(1)
    native.sx(0)
    candidate = study.CompiledCandidate(native, target, {})
    nonlocal_cx = QuantumCircuit(3)
    nonlocal_cx.cx(0, 2)
    with pytest.raises(ValueError, match="nearest-neighbor"):
        study.validate_circuit(nonlocal_cx, native=True)
    with pytest.raises(ValueError, match="target"):
        study.compile_candidate(Candidate(QuantumCircuit(2), {}), target)
    compiled = study.compile_candidate(Candidate(native, {}), target)
    with pytest.raises(ValueError, match="previous transpiler"):
        study.compile_candidate(Candidate(compiled.circuit, {}), target)
    for hardware, arity in (("unknown", 1), ("IBM", 3), ("Google", True)):
        with pytest.raises(ValueError, match="profile"):
            study.pauli_distribution(hardware, arity)
    for count, entropy in ((1, [1]), (True, [1]), (10, []), (10, [-1])):
        with pytest.raises(ValueError, match="Validation needs"):
            study.sample_validation(native, "IBM", count, entropy)
    for invalid in (
        np.zeros((2, 2), dtype=int),
        np.zeros((1, 1), dtype=int),
        np.array([[-1], [0]]),
        np.array([[0.1], [0.0]]),
        np.array([[4], [0]]),
    ):
        with pytest.raises(ValueError, match=r"codes|bank|histories|Pauli"):
            study.validation_score(candidate, "IBM", invalid)
    wrong_target = study.CompiledCandidate(native, np.array([1, 0, 0, 0]), {})
    with pytest.raises(ValueError, match="target"):
        study.validation_score(wrong_target, "IBM", np.zeros((2, 1), dtype=int))
    with pytest.raises(ValueError, match="evaluation law"):
        study.exact_score(candidate, "IBM", law="invented")


def test_frozen_native_schedule_keeps_zero_angle_noise_windows() -> None:
    """Do not cancel trained gates and thereby change the optimized noisy channel."""
    source = QuantumCircuit(2)
    source.sx(0)
    source.rz(0, 0)
    source.cx(0, 1)
    source.cx(0, 1)
    target = Statevector.from_instruction(source).data
    candidate = Candidate(source, {"preserve_native_schedule": True})
    frozen = study.preserve_native_candidate(candidate, target)
    assert frozen.circuit == source
    assert frozen.circuit is not source
    assert frozen.metadata["compiled_two_qubit"] == 2
    assert frozen.metadata["compiled_one_qubit"] == 2
    assert frozen.metadata["compiled_two_qubit_depth"] == 2
    assert frozen.metadata["ideal_fidelity"] == pytest.approx(1)
    rewritten = study.compile_candidate(candidate, target)
    assert study.exact_score(frozen, "IBM")["fidelity"] < study.exact_score(rewritten, "IBM")["fidelity"]
    for invalid in (Candidate(source, {}), Candidate(source, {"preserve_native_schedule": 1})):
        with pytest.raises(ValueError, match="receipt"):
            study.preserve_native_candidate(invalid, target)
    with pytest.raises(ValueError, match="layout"):
        study.preserve_native_candidate(candidate, np.array([1, 0]))
    nonnative = QuantumCircuit(1)
    nonnative.ry(0.5, 0)
    with pytest.raises(ValueError, match=r"native basis|RZ/SX/X/CX"):
        study.preserve_native_candidate(Candidate(nonnative, {"preserve_native_schedule": True}), np.array([1, 0]))
