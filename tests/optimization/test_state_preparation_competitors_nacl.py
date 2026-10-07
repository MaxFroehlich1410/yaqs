# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent circuit, noise, annealing and actual OrthoMADS NACL checks."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector, random_statevector

from experiments.state_preparation_competitors import nacl as study
from experiments.state_preparation_competitors.evaluation import CompiledCandidate, exact_score

if TYPE_CHECKING:
    from numpy.typing import NDArray

HAS_NOMAD = importlib.util.find_spec("PyNomad") is not None
requires_nomad = pytest.mark.skipif(not HAS_NOMAD, reason="Optional official PyNomadBBO dependency is unavailable")


def matrix(gates: tuple[study.NativeGate, ...], num_qubits: int) -> Operator:
    """Resolve a native gate sequence into Qiskit's independent full matrix.

    Returns:
        Full-register unitary operator.
    """
    circuit = QuantumCircuit(num_qubits)
    for gate in gates:
        if gate.name == "rz":
            circuit.rz(gate.angle, gate.sites[0])
        elif gate.name == "sx":
            circuit.sx(gate.sites[0])
        else:
            circuit.cx(gate.sites[0], gate.sites[1])
    return Operator(circuit)


@pytest.mark.parametrize(
    ("name", "sites", "angle"),
    [
        ("h", (0,), 0),
        ("sx", (0,), 1),
        ("rz", (0,), float("nan")),
        ("rz", (0,), True),
        ("rz", (True,), 0),
        ("rz", (), 0),
        ("rz", (0, 1), 0),
        ("cx", (0, 2), 0),
        ("cx", (0, 0), 0),
        ("cx", (-1, 0), 0),
    ],
)
def test_native_gate_rejects_invalid_values(name: str, sites: tuple[int, ...], angle: float) -> None:
    """Unsupported gates, nonlocal supports and nonfinite angles fail explicitly."""
    with pytest.raises(ValueError, match="finite RZ/SX"):
        study.NativeGate(name, sites, angle)


def test_parameter_order_and_resolution_preserve_complex_native_circuit() -> None:
    """Multiple RZs retain both execution order and little-endian wire order."""
    gates = (
        *tuple(
            gate
            for index in range(12)
            for gate in (study.NativeGate("rz", (index % 3,), index / 7), study.NativeGate("sx", (index % 3,)))
        ),
        study.NativeGate("cx", (2, 1)),
        study.NativeGate("cx", (0, 1)),
    )
    template, theta = study.native_template(gates, 3)
    np.testing.assert_allclose(Operator(template.assign_parameters(theta)).data, matrix(gates, 3).data, atol=1e-14)
    assert [parameter.name for parameter in template.parameters] == [f"rz_{index:06d}" for index in range(12)]
    changed = study.resolve_angles(gates, theta + 0.19)
    assert [gate.angle for gate in changed if gate.name == "rz"] == pytest.approx(theta + 0.19)
    assert [gate for gate in changed if gate.name != "rz"] == [gate for gate in gates if gate.name != "rz"]
    assert not study.native_template((), 1)[1].size


@pytest.mark.parametrize("num_qubits", [0, 7, True])
def test_template_rejects_invalid_register(num_qubits: int) -> None:
    """Gate templates cannot silently construct a different-size benchmark."""
    with pytest.raises(ValueError, match="one- to six-qubit"):
        study.native_template((), num_qubits)


def test_template_rejects_gate_outside_register() -> None:
    """A native support must fit its supplied register."""
    with pytest.raises(ValueError, match="one- to six-qubit"):
        study.native_template((study.NativeGate("sx", (2,)),), 2)


@pytest.mark.parametrize("theta", [[0, 1], [float("nan")], [1j], [[0]], [True], ["0"]])
def test_resolution_rejects_malformed_parameters(theta: list[Any]) -> None:
    """Finite scalar angles must match exactly the number of RZ operations."""
    with pytest.raises(ValueError, match="one finite real"):
        study.resolve_angles((study.NativeGate("rz", (0,)),), theta)


def test_every_recognized_commutation_rule_matches_full_matrix_oracle() -> None:
    """Scan intersecting native supports; false positives would corrupt training."""
    gates = [study.NativeGate(name, (site,), 0.37 if name == "rz" else 0) for site in range(3) for name in ("rz", "sx")]
    gates += [study.NativeGate("cx", sites) for sites in ((0, 1), (1, 0), (1, 2), (2, 1))]
    for first in gates:
        for second in gates:
            assert study.commute(first, second) == study.commute(second, first)
            if study.commute(first, second):
                np.testing.assert_allclose(matrix((first, second), 3).data, matrix((second, first), 3).data, atol=1e-14)
    assert not study.commute(study.NativeGate("rz", (1,), 0.37), study.NativeGate("cx", (0, 1)))
    assert not study.commute(study.NativeGate("sx", (0,)), study.NativeGate("cx", (0, 1)))


def test_simplification_cancels_only_exact_identities_through_commuting_gates() -> None:
    """Cancel CX/SX and combine RZ while preserving state up to global phase."""
    gates = (
        study.NativeGate("cx", (1, 0)),
        study.NativeGate("rz", (1,), 0.31),
        study.NativeGate("cx", (1, 0)),
        study.NativeGate("rz", (2,), 2 * math.pi),
        *[study.NativeGate("sx", (0,)) for _ in range(4)],
        study.NativeGate("rz", (1,), -0.31),
        study.NativeGate("rz", (0,), 0.22),
        study.NativeGate("rz", (0,), 0.12),
    )
    actual = study.simplify_gates(gates)
    assert len(actual) == 1
    assert actual[0].angle == pytest.approx(0.34)
    assert matrix(actual, 3).equiv(matrix(gates, 3))
    # A noncommuting intervening rotation must prevent CX cancellation.
    protected = (study.NativeGate("cx", (0, 1)), study.NativeGate("rz", (1,), 0.2), study.NativeGate("cx", (0, 1)))
    assert study.simplify_gates(protected) == protected
    # No arbitrary near-zero threshold is permitted by this reproduction.
    tiny = (study.NativeGate("rz", (0,), 1e-15),)
    assert study.simplify_gates(tiny) == tiny


def test_random_simplifications_match_independent_unitaries() -> None:
    """Random long sequences catch interactions between cancellation rules."""
    rng = np.random.default_rng(61)
    pool = [study.NativeGate("rz", (q,), a) for q in range(3) for a in (-0.3, 0, 0.3)]
    pool += [study.NativeGate("sx", (q,)) for q in range(3)]
    pool += [study.NativeGate("cx", pair) for pair in ((0, 1), (1, 0), (1, 2), (2, 1))]
    for _ in range(12):
        gates = tuple(pool[int(index)] for index in rng.integers(len(pool), size=35))
        simplified = study.simplify_gates(gates)
        assert len(simplified) <= len(gates)
        assert matrix(gates, 3).equiv(matrix(simplified, 3), atol=1e-13)


@pytest.mark.parametrize("two_qubit", [False, True])
def test_inserted_block_is_identity_and_parameters_can_leave_identity(*, two_qubit: bool) -> None:
    """Use a full unitary oracle, not fidelity only on the zero state."""
    for seed in range(8):
        block = study.identity_block(3, np.random.default_rng(seed), two_qubit=two_qubit)
        assert matrix(block, 3).equiv(Operator(np.eye(8)), atol=1e-14)
        _, theta = study.native_template(block, 3)
        theta[len(theta) // 2] += 0.37
        perturbed = study.resolve_angles(block, theta)
        assert not matrix(perturbed, 3).equiv(Operator(np.eye(8)))


@pytest.mark.parametrize(("num_qubits", "two_qubit"), [(0, False), (7, False), (1, True), (True, False)])
def test_identity_block_rejects_invalid_support(num_qubits: int, *, two_qubit: bool) -> None:
    """Two-qubit identity proposals require an actual adjacent pair."""
    with pytest.raises(ValueError, match="support must fit"):
        study.identity_block(num_qubits, np.random.default_rng(0), two_qubit=two_qubit)


def test_structural_proposals_cover_published_move_types_and_resource_cap() -> None:
    """All move types are available, seeded, and preserve untouched operations."""
    original = (study.NativeGate("sx", (0,)), study.NativeGate("rz", (0,), 0.2), study.NativeGate("cx", (0, 1)))
    seen = set()
    for seed in range(100):
        proposal, record = study.propose_structure(original, 2, np.random.default_rng(seed), max_native_gates=8)
        replay, replay_record = study.propose_structure(original, 2, np.random.default_rng(seed), max_native_gates=8)
        assert proposal == replay
        assert record == replay_record
        seen.add(record["move"])
        assert len(proposal) <= 8
        if record["move"].startswith("insert"):
            assert matrix(proposal, 2).equiv(matrix(original, 2))
        elif record["move"] == "remove_gate":
            assert len(proposal) == len(original) - 1
        else:
            assert sorted(map(repr, proposal)) == sorted(map(repr, original))
    assert seen == {"insert_1q_identity", "insert_2q_identity", "remove_gate", "shuffle_adjacent"}
    assert study.propose_structure((), 1, np.random.default_rng(0), max_native_gates=0) == (
        (),
        {"move": "no_available_move"},
    )
    for cap in (-1, True, 2):
        with pytest.raises(ValueError, match="max_native_gates"):
            study.propose_structure(original, 2, np.random.default_rng(0), max_native_gates=cap)


def test_annealing_matches_exponential_rule_including_zero_temperature() -> None:
    """The rule must sometimes accept worsening moves, not just greedy descent."""
    assert study.anneal_accept(0.3, 0.2, 0, 0.99)
    assert study.anneal_accept(0.3, 0.3, 0, 0.99)
    assert not study.anneal_accept(0.3, 0.4, 0, 0)
    probability = math.exp(-(0.4 - 0.3) / 0.02)
    assert study.anneal_accept(0.3, 0.4, 0.02, probability / 2)
    assert not study.anneal_accept(0.3, 0.4, 0.02, probability * 2)
    assert study.anneal_accept(0.3, 0.4, 0.02, 0)
    assert not study.anneal_accept(0, 1, 1e-200, 0.5)


@pytest.mark.parametrize("arguments", [(0, float("nan"), 1, 0.5), (0, 1, -1, 0.5), (0, 1, 1, -0.1), (0, 1, 1, 1)])
def test_annealing_rejects_invalid_values(arguments: tuple[float, ...]) -> None:
    """Invalid random variates must not silently bias the search."""
    with pytest.raises(ValueError, match="Annealing requires"):
        study.anneal_accept(*arguments)


def test_loader_fails_without_dependency_or_on_wrong_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """A different optimizer or package version is never silently substituted."""

    def missing(_name: str) -> str:
        """Simulate an optional package absent from this environment.

        Raises:
            PackageNotFoundError: Always, as the test's missing dependency.
        """
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(study.importlib.metadata, "version", missing)
    with pytest.raises(ImportError, match="optional official"):
        study.load_nomad()
    monkeypatch.setattr(study.importlib.metadata, "version", lambda _name: "0.0.0")
    with pytest.raises(ImportError, match=r"found 0\.0\.0"):
        study.load_nomad()


@requires_nomad
def test_actual_orthomads_minimizes_independent_quadratic_and_replays() -> None:
    """Invoke the installed C++ optimizer rather than a mocked replacement."""

    def objective(theta: NDArray[np.float64]) -> float:
        """Return a quadratic whose analytic minimizer is known independently."""
        return float((theta[0] - 0.73) ** 2 + 2 * (theta[1] + 0.31) ** 2)

    first = study.ortho_mads(objective, [0, 0], evaluations=120, seed=73)
    replay = study.ortho_mads(objective, [0, 0], evaluations=120, seed=73)
    assert first.loss < 1e-4
    assert first.loss == pytest.approx(objective(first.theta))
    np.testing.assert_array_equal(first.theta, replay.theta)
    assert first.receipt == replay.receipt
    assert first.receipt["evaluations"] == first.receipt["upstream_evaluations"] <= 120
    assert first.loss == min(record["loss"] for record in first.receipt["records"])
    assert "DIRECTION_TYPE ORTHO 2N" in first.receipt["parameters"]
    assert "QUAD_MODEL_SEARCH false" in first.receipt["parameters"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial": []},
        {"initial": [1j]},
        {"initial": [float("inf")]},
        {"evaluations": 0},
        {"seed": 2**31},
        {"seed": True},
        {"initial_frame_size": 0},
        {"min_frame_size": 1},
    ],
)
def test_orthomads_rejects_invalid_controls_before_native_call(kwargs: dict[str, Any]) -> None:
    """Native seed overflow is rejected in Python, avoiding the upstream parser."""
    arguments: dict[str, Any] = {"initial": [0], "evaluations": 10, "seed": 1, **kwargs}
    with pytest.raises(ValueError, match="OrthoMADS needs"):
        study.ortho_mads(lambda _theta: 0.0, **arguments)


@requires_nomad
def test_actual_optimizer_prepares_arbitrary_complex_one_qubit_state() -> None:
    """The variational identity block spans complex amplitudes under real MADS."""
    target = np.array([np.sqrt(0.3), np.exp(0.7j) * np.sqrt(0.7)])
    initial = study.identity_block(1, np.random.default_rng(9), two_qubit=False)
    optimized, loss, receipt = study.optimize_structure(
        initial, target, None, evaluations=120, seed=7, initial_frame_size=0.5, min_frame_size=1e-6
    )
    state = matrix(optimized, 1).data[:, 0]
    assert loss < 1e-4
    assert loss == pytest.approx(1 - abs(np.vdot(target, state)) ** 2, abs=1e-13)
    assert receipt["optimizer"] == "official_PyNomadBBO_OrthoMADS"


@requires_nomad
@pytest.mark.parametrize("hardware", ["IBM", "Google"])
def test_six_qubit_complex_pipeline_matches_independent_exact_noisy_scoring(hardware: str) -> None:
    """Run real search, restarts, simplification and independent final scoring."""
    target = random_statevector(64, seed=991).data
    untouched = target.copy()
    candidates = study.nacl_candidates(
        target,
        hardware=hardware,
        seed=400,
        search_steps=(0, 2, 4),
        inner_evaluations=5,
        restarts=2,
        simplify_every=2,
    )
    cumulative = 0.0
    total_evaluations = 0
    prior_fidelity = -1.0
    for index, candidate in enumerate(candidates):
        metadata = candidate.metadata
        reference = exact_score(CompiledCandidate(candidate.circuit, target, {}), hardware)
        assert metadata["training_fidelity"] == pytest.approx(reference["fidelity"], abs=2e-13)
        assert metadata["ideal_fidelity"] == pytest.approx(
            abs(np.vdot(target, Statevector(candidate.circuit).data)) ** 2
        )
        assert metadata["training_fidelity"] >= prior_fidelity - 1e-14
        prior_fidelity = metadata["training_fidelity"]
        cumulative += metadata["cpu_seconds"]
        assert metadata["cumulative_cpu_seconds"] == pytest.approx(cumulative)
        total_evaluations += metadata["interval_objective_evaluations"]
        assert metadata["cumulative_objective_evaluations"] == total_evaluations
        assert metadata["preserve_native_schedule"]
        assert metadata["inner_seed_distribution"] == "uniform_integer_0_through_65535_from_restart_generator"
        for record in metadata["records"]:
            for run in record["inner_runs"]:
                assert 0 <= run["seed"] < 2**16
        assert metadata["previous_checkpoint"] == (
            None if index == 0 else candidates[index - 1].metadata["candidate_id"]
        )
        assert {record["step"] for record in metadata["records"]} == (
            set() if index == 0 else {2 * index - 1, 2 * index}
        )
        json.dumps(metadata, allow_nan=False)
    np.testing.assert_array_equal(target, untouched)


@requires_nomad
def test_checkpoints_do_not_change_seeded_search_and_histories_are_not_duplicated() -> None:
    """Splitting the same search into checkpoint intervals preserves every move."""
    target = random_statevector(4, seed=700).data
    split = study.nacl_candidates(target, hardware="IBM", seed=82, search_steps=(0, 1, 3), inner_evaluations=5)
    single = study.nacl_candidates(target, hardware="IBM", seed=82, search_steps=(3,), inner_evaluations=5)[0]
    assert [record for candidate in split for record in candidate.metadata["records"]] == single.metadata["records"]
    np.testing.assert_array_equal(Operator(split[-1].circuit).data, Operator(single.circuit).data)
    assert split[-1].metadata["cumulative_objective_evaluations"] == single.metadata["cumulative_objective_evaluations"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hardware": "unsupported"},
        {"seed": -1},
        {"search_steps": ()},
        {"search_steps": (1, 1)},
        {"search_steps": (2, 1)},
        {"search_steps": (False,)},
        {"restarts": 0},
        {"inner_evaluations": 0},
        {"max_native_gates": 4},
        {"initial_temperature": float("nan")},
        {"cooling": 0},
        {"min_frame_size": 1},
        {"simplify_every": -1},
    ],
)
def test_pipeline_rejects_invalid_controls_before_optimizer_load(kwargs: dict[str, Any]) -> None:
    """Malformed budgets and noise controls never launch native optimization."""
    arguments: dict[str, Any] = {"hardware": "IBM", **kwargs}
    with pytest.raises(ValueError, match="Invalid NACL"):
        study.nacl_candidates([1, 0], **arguments)


@requires_nomad
def test_actual_orthomads_frame_convergence_uses_returned_feasible_incumbent() -> None:
    """NOMAD 4.6.0 may mislabel a successful frame stop with a negative flag."""
    result = study.ortho_mads(
        lambda theta: float((theta[0] - 0.73) ** 2),
        [0.0],
        evaluations=50,
        seed=7,
        initial_frame_size=0.5,
        min_frame_size=0.1,
    )
    assert result.loss < 0.01
    assert "Min frame size reached" in result.receipt["stop_reason"]
    assert result.receipt["evaluations"] < 50


def test_parameter_free_structure_uses_exact_noisy_value_without_optimizer() -> None:
    """A fixed native circuit has no angles to pass to an unrelated optimizer."""
    target = np.array([1, 1j]) / np.sqrt(2)
    gates = (study.NativeGate("sx", (0,)),)
    optimized, loss, receipt = study.optimize_structure(
        gates, target, "Google", evaluations=10, seed=7, initial_frame_size=0.5, min_frame_size=1e-6
    )
    circuit, theta = study.native_template(gates, 1)
    reference = exact_score(CompiledCandidate(circuit.assign_parameters(theta), target, {}), "Google")
    assert optimized == gates
    assert loss == pytest.approx(1 - reference["fidelity"])
    assert receipt["evaluations"] == 1


@requires_nomad
def test_actual_optimizer_propagates_nonfinite_callback_failure() -> None:
    """Failed blackbox evaluations must not become a fabricated fidelity."""
    with pytest.raises(ValueError, match="nonfinite loss"):
        study.ortho_mads(lambda _theta: math.nan, [0.0], evaluations=3, seed=1)
