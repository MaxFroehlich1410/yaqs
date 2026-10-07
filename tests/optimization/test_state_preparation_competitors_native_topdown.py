# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Native schedule, noise, parameter and unchanged-update regression oracles."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, Statevector, random_statevector

from experiments import legacy_topdown_dense as legacy
from experiments.state_preparation_competitors import evaluation
from experiments.state_preparation_competitors import native_topdown as study
from experiments.state_preparation_competitors.common import from_parameterized
from experiments.state_preparation_competitors.noisy_core import DenseObjective
from mqt.yaqs.optimization import KrotovNoiseMap, ParameterizedCircuit, ParameterizedGate, StatevectorBackend

if TYPE_CHECKING:
    from mqt.yaqs.optimization import KrotovTJMOptions
    from mqt.yaqs.optimization.statevector import NoiseMaps


def problem() -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Use reversed parameter/support order, signed scales and constant offsets.

    Returns:
        Affine circuit, original parameter vector and normalized complex target.
    """
    circuit = ParameterizedCircuit(
        3,
        [
            ParameterizedGate("rx", (2,), angle_offset=np.pi),
            ParameterizedGate("ry", (0,), param_index=2, angle_scale=-1.3, angle_offset=0.2),
            ParameterizedGate("rxx", (1, 0), param_index=0, angle_scale=0.7, angle_offset=-0.4),
            ParameterizedGate("rz", (1,), param_index=1, angle_scale=1.2, angle_offset=0.3),
        ],
        num_params=3,
    )
    return circuit, np.array([0.4, -0.7, 0.9]), random_statevector(8, seed=931).data


def injected_paths(circuit: ParameterizedCircuit) -> NoiseMaps:
    """Construct explicit noise events on the native schedule.

    Returns:
        Ninety no-jump paths, seven local jumps and three two-site jumps.
    """
    empty = tuple(KrotovNoiseMap() for _ in circuit.gates)
    one = list(empty)
    two = list(empty)
    index_one = next(index for index, gate in enumerate(circuit.gates) if len(gate.sites) == 1)
    index_two = next(index for index, gate in enumerate(circuit.gates) if len(gate.sites) == 2)
    one[index_one] = KrotovNoiseMap(operators=((Pauli("Y").to_matrix(), circuit.gates[index_one].sites),))
    sites = tuple(sorted(circuit.gates[index_two].sites))
    two[index_two] = KrotovNoiseMap(operators=((Pauli("XZ").to_matrix(), sites),))
    return (empty,) * 90 + (tuple(one),) * 7 + (tuple(two),) * 3


def trajectory_oracle(circuit: QuantumCircuit, target: np.ndarray, maps: NoiseMaps) -> float:
    """Independently propagate Qiskit states, respecting backend local-MSB maps.

    Returns:
        Empirical mean target fidelity of the supplied explicit paths.
    """
    fidelities = []
    for row in maps:
        state = Statevector.from_label("0" * circuit.num_qubits)
        for item, noise in zip(circuit.data, row, strict=True):
            sites = [circuit.find_bit(qubit).index for qubit in item.qubits]
            state = state.evolve(item.operation, qargs=sites)
            for matrix, support in noise.operators:
                state = state.evolve(Operator(matrix), qargs=list(reversed(support)))
        fidelities.append(abs(np.vdot(target, state.data)) ** 2)
    return float(np.mean(fidelities))


def test_affine_coordinates_and_native_unitary_match_original_at_random_points() -> None:
    """Compilation preserves original index order and all ideal transformations."""
    original, _, target = problem()
    compiled = study.compile_training(original, target)
    assert compiled.compilation["compiler"] == evaluation.COMPILER
    assert [str(parameter) for parameter in compiled.parameters] == ["original[0]", "original[1]", "original[2]"]
    assert compiled.compilation["parameter_occurrences"] == [1, 1, 1]
    assert set(compiled.template.count_ops()) == {"rz", "sx", "x", "cx"}
    assert compiled.circuit.num_params == original.num_params
    assert all(gate.noise_enabled for gate in compiled.circuit.gates)
    assert compiled.template.layout is None
    for theta in np.random.default_rng(901).normal(size=(4, 3)):
        diagnostics = study.audit_ideal_equivalence(original, compiled, theta)
        assert max(diagnostics.values()) < 1e-12
        expected = Statevector.from_instruction(from_parameterized(original, theta)).data
        assert StatevectorBackend(compiled.circuit, expected).metrics(theta)[1] == pytest.approx(1, abs=1e-12)
    assert compiled.bind([0, 0, 0]).count_ops() == compiled.bind([0.3, 0.5, -0.4]).count_ops()
    json.dumps(compiled.compilation, allow_nan=False)


@pytest.mark.parametrize("gate_name", ["rx", "ry", "rz", "rxx", "ryy", "rzz"])
def test_every_source_rotation_compiles_without_reordering_its_affine_angle(gate_name: str) -> None:
    """All pruning-B primitive axes and reversed pair supports have exact conversion."""
    sites = (1, 0) if len(gate_name) == 3 else (1,)
    original = ParameterizedCircuit(
        2, [ParameterizedGate(gate_name, sites, param_index=0, angle_scale=-0.9, angle_offset=0.2)]
    )
    target = random_statevector(4, seed=43).data
    compiled = study.compile_training(original, target)
    theta = np.array([0.53])
    assert max(study.audit_ideal_equivalence(original, compiled, theta).values()) < 1e-12
    native = [gate for gate in compiled.circuit.gates if gate.param_index is not None]
    assert len(native) == 1
    assert native[0].angle_scale == pytest.approx(-0.9)


def test_empty_and_fixed_only_source_circuits_remain_valid() -> None:
    """A valid zero-parameter schedule needs no artificial optimizer coordinates."""
    for gates in ([], [ParameterizedGate("rx", (0,), angle_offset=np.pi)]):
        original = ParameterizedCircuit(1, gates, num_params=0)
        compiled = study.compile_training(original, [1, 0])
        assert compiled.parameters == ()
        assert compiled.circuit.num_params == 0
        assert max(study.audit_ideal_equivalence(original, compiled, []).values()) < 1e-12


@pytest.mark.parametrize("hardware", ["IBM", "Google"])
def test_native_density_and_adjoint_match_independent_final_channel(hardware: str) -> None:
    """The noisy native objective exactly matches the independent deployment oracle."""
    original, theta, target = problem()
    compiled = study.compile_training(original, target)
    objective = DenseObjective(compiled.template, target, hardware, compile_native=False)
    bound = compiled.bind(theta)
    candidate = evaluation.CompiledCandidate(bound, target, {})
    expected = evaluation.exact_score(candidate, hardware)["fidelity"]
    loss, gradient = objective.loss_gradient(theta)
    assert 1 - loss == pytest.approx(expected, abs=1e-13)
    delta = 1e-5
    finite_difference = []
    for direction in np.eye(len(theta)):
        plus = evaluation.CompiledCandidate(compiled.bind(theta + delta * direction), target, {})
        minus = evaluation.CompiledCandidate(compiled.bind(theta - delta * direction), target, {})
        finite_difference.append(
            (evaluation.exact_score(minus, hardware)["fidelity"] - evaluation.exact_score(plus, hardware)["fidelity"])
            / (2 * delta)
        )
    np.testing.assert_allclose(gradient, finite_difference, atol=2e-10, rtol=0)


def test_raw_compressed_cross_signal_and_online_updates_match_with_injected_jumps() -> None:
    """Both trajectory indices retain multiplicity under every native online update."""
    original, theta, target = problem()
    compiled = study.compile_training(original, target)
    engine = StatevectorBackend(compiled.circuit, target)
    raw = injected_paths(compiled.circuit)
    compressed, weights = engine.compress_no_jump_maps(raw)
    assert len(compressed) == 11
    assert weights[0] == pytest.approx(0.9)
    dense, short = theta.copy(), theta.copy()
    for step in range(4):
        full_signal, full_loss, full_fidelity = engine.gradient(dense, raw, estimator="cross")
        reduced, loss, fidelity = engine.gradient(short, compressed, estimator="cross", weights=weights)
        np.testing.assert_allclose(full_signal, reduced, atol=2e-13, rtol=0)
        assert (loss, fidelity) == pytest.approx((full_loss, full_fidelity), abs=2e-13)
        if step < 2:
            dense, _ = engine.online_update(dense, 0.2, raw, estimator="cross")
            short, _ = engine.online_update(short, 0.2, compressed, estimator="cross", weights=weights)
        else:
            dense -= 0.2 * full_signal
            short -= 0.2 * reduced
        np.testing.assert_allclose(dense, short, atol=2e-13, rtol=0)


def test_independent_trajectory_gradient_matches_qiskit_finite_differences() -> None:
    """Only the independent estimator is asserted to differentiate sample loss."""
    original, theta, target = problem()
    compiled = study.compile_training(original, target)
    engine = StatevectorBackend(compiled.circuit, target)
    maps = injected_paths(compiled.circuit)
    signal, _, fidelity = engine.gradient(theta, maps, estimator="independent")
    assert fidelity == pytest.approx(trajectory_oracle(compiled.bind(theta), target, maps), abs=1e-13)
    delta = 1e-5
    numerical = [
        (
            trajectory_oracle(compiled.bind(theta - delta * direction), target, maps)
            - trajectory_oracle(compiled.bind(theta + delta * direction), target, maps)
        )
        / (2 * delta)
        for direction in np.eye(len(theta))
    ]
    np.testing.assert_allclose(signal, numerical, atol=2e-10, rtol=0)


def test_training_is_identical_to_direct_historical_train_with_native_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrapper retains exact online/batch updates and final-iterate policy."""
    original, theta, target = problem()
    before = theta.copy()
    compiled = study.compile_training(original, target)

    def sample(
        self: StatevectorBackend, _theta: np.ndarray, options: KrotovTJMOptions, *_args: object, **_kwargs: object
    ) -> NoiseMaps:
        assert options.num_trajectories == 100
        return injected_paths(self.circuit)

    monkeypatch.setattr(StatevectorBackend, "sample_noise_maps", sample)
    expected, trace = legacy.train(
        compiled.circuit,
        theta,
        target,
        steps=4,
        online=2,
        rate=0.2,
        seed=5300,
        noisy=True,
        crn=True,
        decay=0.0,
        stop=2.0,
        hardware="IBM",
        num_trajectories=100,
        compress_no_jump=True,
        record_noise_banks=True,
        require_jump_coverage=True,
    )
    actual = study.train_candidate(original, theta, target, steps=4, online=2)
    np.testing.assert_array_equal(theta, before)
    np.testing.assert_array_equal(actual.metadata["final_theta"], expected)
    assert actual.circuit == compiled.bind(expected)
    assert actual.metadata["training"]["fidelity_trace"] == trace["fidelity_trace"]
    assert actual.metadata["training"]["noise_banks"] == trace["noise_banks"]
    assert actual.metadata["preserve_native_schedule"] is True
    assert actual.metadata["retention"] == "final_iterate_historical_noisy_policy"
    bank = actual.metadata["training"]["noise_banks"][0]
    assert (bank["sampled_paths"], bank["jump_paths"], bank["retained_paths"]) == (100, 10, 11)
    assert bank["weights"][0] == pytest.approx(0.9)
    assert len(bank["raw_sha256"]) == 64
    assert actual.metadata["training_cpu_seconds"] > 0
    assert actual.metadata["cpu_seconds"] >= actual.metadata["training_cpu_seconds"]
    json.dumps(actual.metadata, allow_nan=False)


def test_missing_jumps_stop_without_resampling(monkeypatch: pytest.MonkeyPatch) -> None:
    """A nonempty native circuit never silently becomes a noiseless fine-tuning arm."""
    original, theta, target = problem()
    calls = 0

    def empty(
        self: StatevectorBackend, _theta: np.ndarray, options: KrotovTJMOptions, *_args: object, **_kwargs: object
    ) -> NoiseMaps:
        nonlocal calls
        calls += 1
        return tuple(tuple(KrotovNoiseMap() for _ in self.circuit.gates) for _ in range(options.num_trajectories))

    monkeypatch.setattr(StatevectorBackend, "sample_noise_maps", empty)
    with pytest.raises(RuntimeError, match="no jumps"):
        study.train_candidate(original, theta, target, steps=2, online=1)
    assert calls == 1


@pytest.mark.parametrize(
    "options",
    [
        {"steps": -1},
        {"steps": True},
        {"online": -1},
        {"steps": 1, "online": 2},
        {"seed": -1},
        {"num_trajectories": 0},
        {"num_trajectories": True},
        {"rate": -0.1},
        {"rate": float("nan")},
        {"rate": True},
        {"hardware": "unknown"},
        {"require_jump_coverage": 1},
    ],
)
def test_training_rejects_invalid_settings(options: dict[str, Any]) -> None:
    """Invalid policies cannot be mistaken for the declared historical defaults."""
    original, theta, target = problem()
    with pytest.raises(ValueError, match=r"nonnegative|Require"):
        study.train_candidate(original, theta, target, **options)


@pytest.mark.parametrize(
    ("gates", "count"),
    [
        ([ParameterizedGate("ry", (0,), param_index=0)], 2),
        ([ParameterizedGate("ry", (0,), param_index=0), ParameterizedGate("rz", (0,), param_index=0)], 1),
        ([ParameterizedGate("ry", (0,), param_index=0, angle_scale=0)], 1),
        ([ParameterizedGate("ry", (0,), param_index=0, noise_enabled=False)], 1),
        ([ParameterizedGate("ry", (0,), param_index=0, data_map=lambda _x: 0)], 1),
        ([ParameterizedGate("cx", (0, 1))], 0),
        ([ParameterizedGate("rxx", (0, 2), param_index=0)], 1),
    ],
)
def test_unsupported_original_coordinates_and_gates_are_rejected(gates: list[ParameterizedGate], count: int) -> None:
    """Missing, shared, disabled, data-driven or unsupported source gates fail explicitly."""
    original = ParameterizedCircuit(3, gates, count)
    with pytest.raises(ValueError, match=r"exactly once|removed|Require"):
        study.compile_training(original, random_statevector(8, seed=8).data)


def test_compiler_merged_coordinates_are_rejected_without_changing_online_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adjacent independently controlled RZ gates cannot silently become one update."""
    source = ParameterizedCircuit(
        1, [ParameterizedGate("rz", (0,), param_index=0), ParameterizedGate("rz", (0,), param_index=1)]
    )

    def merged(template: QuantumCircuit, **_kwargs: object) -> QuantumCircuit:
        result = QuantumCircuit(1)
        first, second = template.parameters
        result.rz(first + second, 0)
        return result

    monkeypatch.setattr(study, "transpile", merged)
    with pytest.raises(ValueError, match="exactly one free parameter"):
        study.compile_training(source, [1, 0])


def test_compiler_duplicate_parameter_occurrences_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed compiler must not create multiple online updates per coordinate."""
    original = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)])

    def duplicate(source: QuantumCircuit, **_kwargs: object) -> QuantumCircuit:
        result = QuantumCircuit(1)
        p = next(iter(source.parameters))
        result.rz(p / 2, 0)
        result.rz(p / 2, 0)
        return result

    monkeypatch.setattr(study, "transpile", duplicate)
    with pytest.raises(ValueError, match="compiled parameter must occur exactly once"):
        study.compile_training(original, [1, 0])


def test_audit_detects_wrong_unitary_even_when_source_fidelity_is_degenerate() -> None:
    """Full-unitary audit catches errors invisible on one all-zero input state."""
    original = ParameterizedCircuit(1, [ParameterizedGate("rz", (0,), param_index=0)])
    compiled = study.compile_training(original, [1, 0])
    wrong = compiled.template.copy()
    wrong.x(0)
    with pytest.raises(ValueError, match="equivalence"):
        study.audit_ideal_equivalence(original, replace(compiled, template=wrong), [0.3])


def test_bad_target_and_parameters_fail_before_training() -> None:
    """Validate target register and original coordinate vectors without renormalization."""
    original, theta, target = problem()
    with pytest.raises(ValueError, match="match"):
        study.compile_training(original, [1, 0])
    with pytest.raises(ValueError, match="normalized"):
        study.compile_training(original, target * 2)
    compiled = study.compile_training(original, target)
    for value in ([], [1j, 0, 0], [float("nan"), 0, 0]):
        with pytest.raises(ValueError, match="parameter vector"):
            compiled.bind(value)
    np.testing.assert_array_equal(theta, np.array([0.4, -0.7, 0.9]))
