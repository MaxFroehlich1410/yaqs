# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent oracle checks and scientific identities of the VQSP controls."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector
from scipy.optimize import OptimizeResult

from experiments.legacy_topdown_dense import LegacyNoise, extend, train
from experiments.state_preparation_competitors import controls as study
from experiments.state_preparation_competitors.common import from_parameterized
from experiments.state_preparation_pilot import restore_circuit
from experiments.topdown_batch_refinement_policies import train_noiseless
from mqt.yaqs.optimization import (
    KrotovNoiseMap,
    KrotovTJMOptions,
    ParameterizedCircuit,
    ParameterizedGate,
    StatevectorBackend,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def problem() -> tuple[ParameterizedCircuit, np.ndarray, np.ndarray]:
    """Return complex, entangling, shared-parameter gates with appreciable noise."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), 0),
            ParameterizedGate("rxx", (0, 1), 1, angle_scale=-0.7, angle_offset=0.2),
            ParameterizedGate("rz", (1,), 0, angle_scale=0.3),
        ]
        * 6,
        2,
    )
    return circuit, np.array([0.4, -0.8]), np.array([1, 2j, -3, 4j], dtype=complex) / np.sqrt(30)


def test_empirical_compression_matches_raw_gradient_and_independent_qiskit_oracle() -> None:
    """No-jump merging preserves exact finite-sample loss and its true derivative."""
    circuit, theta, target = problem()
    engine = StatevectorBackend(circuit, target)
    maps, weights, bank = study.frozen_training_bank(
        circuit, theta, target, seed=300, hardware="Google", training_paths=100
    )
    raw = engine.sample_noise_maps(theta, KrotovTJMOptions(num_trajectories=100), LegacyNoise("Google"), iteration=300)
    weighted = engine.gradient(theta, maps, weights=weights, estimator="independent")
    uncompressed = engine.gradient(theta, raw, estimator="independent")
    np.testing.assert_allclose(weighted[0], uncompressed[0], atol=2e-14)
    np.testing.assert_allclose(weighted[1:], uncompressed[1:], atol=2e-14)
    assert 0 < bank["jump_paths"] < 100
    assert bank["retained_paths"] == 1 + bank["jump_paths"]
    assert sum(bank["multiplicities"]) == 100
    assert sum(weights) == pytest.approx(1)
    assert bank["raw_jump_counts"].count(0) == bank["no_jump_paths"]
    physical = from_parameterized(circuit, theta)
    values = []
    for row in maps:
        state = Statevector.from_label("00")
        for instruction, noise in zip(physical.data, row, strict=True):
            sites = [physical.find_bit(qubit).index for qubit in instruction.qubits]
            state = state.evolve(instruction.operation, qargs=sites)
            for matrix, support in noise.operators:
                state = state.evolve(Operator(matrix), qargs=list(support))
        values.append(abs(np.vdot(target, state.data)) ** 2)
    assert weights @ values == pytest.approx(weighted[2], abs=2e-14)
    _, _, repeated = study.frozen_training_bank(
        circuit, theta + 1, target, seed=300, hardware="Google", training_paths=100
    )
    assert bank["raw_sha256"] == repeated["raw_sha256"]
    json.dumps(bank, allow_nan=False)


def test_scipy_callback_receives_true_weighted_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe the actual objective adapter with central differences at frozen maps."""
    circuit, initial, target = problem()
    maps, weights, _ = study.frozen_training_bank(
        circuit, initial, target, seed=300, hardware="Google", training_paths=100
    )
    backend = StatevectorBackend(circuit, target)
    observed = {}

    def inspect(objective: Callable, theta: np.ndarray, **kwargs: object) -> OptimizeResult:
        observed.update(kwargs)
        loss, gradient = objective(theta)
        for index in range(len(theta)):
            delta = np.eye(len(theta))[index] * 1e-6
            finite_difference = (objective(theta + delta)[0] - objective(theta - delta)[0]) / 2e-6
            assert gradient[index] == pytest.approx(finite_difference, abs=2e-9)
        assert loss == pytest.approx(backend.metrics(theta, maps, weights=weights)[0], abs=1e-14)
        return OptimizeResult(x=theta, fun=loss, success=False, status=1, message="test", nfev=5)

    monkeypatch.setattr(study, "minimize", inspect)
    returned, trace = study.optimize_lbfgsb(circuit, initial, target, steps=2, maps=maps, weights=weights)
    np.testing.assert_array_equal(returned, initial)
    assert observed["method"] == "L-BFGS-B"
    assert observed["jac"] is True
    assert trace["estimator"] == "independent"
    assert trace["objective_evaluations"] == trace["gradient_evaluations"] == 5


def test_real_scipy_converges_and_reports_every_accepted_checkpoint() -> None:
    """Exercise the installed optimizer and independently materialize its endpoint."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), 0)], 1)
    initial, target = np.array([0.2]), np.array([0, 1], dtype=complex)
    returned, trace = study.optimize_lbfgsb(circuit, initial, target, steps=12)
    fidelity = abs(np.vdot(target, Statevector.from_instruction(from_parameterized(circuit, returned)).data)) ** 2
    assert fidelity > 1 - 1e-10
    assert trace["returned_fidelity"] == pytest.approx(fidelity, abs=1e-14)
    assert len(trace["records"]) == trace["steps"] + 1
    assert 0 < trace["steps"] <= 12
    assert trace["gradient_evaluations"] == trace["objective_evaluations"] <= trace["options"]["maxfun"]
    assert trace["scipy_function_evaluations"] == trace["objective_evaluations"]
    assert trace["scipy_success"]
    assert np.all(np.diff([row["fidelity"] for row in trace["records"]]) >= -1e-14)
    assert np.all(np.diff([row["cumulative_cpu_seconds"] for row in trace["records"]]) >= 0)
    np.testing.assert_array_equal(initial, [0.2])
    json.dumps(trace, allow_nan=False)


def test_best_retention_includes_initial_and_excludes_unaccepted_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only initial and accepted states can win; returned vectors match their scores."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), 0)], 1)
    initial, target = np.array([2.9]), np.array([0, 1], dtype=complex)

    def regression(objective: Callable, _theta: np.ndarray, **kwargs: object) -> OptimizeResult:
        objective(np.array([np.pi]))  # Better, but never accepted.
        worse = np.array([0.0])
        loss, _ = objective(worse)
        callback = cast("Callable", kwargs["callback"])
        callback(OptimizeResult(x=worse, fun=loss))
        return OptimizeResult(x=worse, fun=loss, success=False, status=2, message="synthetic regression", nfev=3)

    monkeypatch.setattr(study, "minimize", regression)
    returned, trace = study.optimize_lbfgsb(circuit, initial, target, steps=2)
    np.testing.assert_array_equal(returned, initial)
    np.testing.assert_array_equal(trace["final_theta"], [0])
    assert trace["best_iteration"] == 0
    assert trace["returned_fidelity"] == pytest.approx(StatevectorBackend(circuit, target).metrics(initial)[1])
    assert trace["returned_fidelity"] < 1
    assert trace["final_fidelity"] < trace["initial_fidelity"]
    assert trace["objective_evaluations"] == 3
    assert trace["nonaccepted_probe_evaluations"] == 1


def test_joint_evaluation_budget_is_hard_even_inside_scipy_line_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """SciPy's delayed maxfun check cannot overshoot the declared oracle budget."""
    circuit, initial, target = problem()

    def trials(objective: Callable, theta: np.ndarray, **_kwargs: object) -> None:
        objective(theta + 0.1)
        objective(theta + 0.2)

    monkeypatch.setattr(study, "minimize", trials)
    returned, trace = study.optimize_lbfgsb(circuit, initial, target, steps=10, max_evaluations=2)
    np.testing.assert_array_equal(returned, initial)
    assert trace["stop_reason"] == "evaluation_budget"
    assert trace["objective_evaluations"] == trace["gradient_evaluations"] == 2
    assert trace["steps"] == 0
    assert trace["scipy_status"] is None


@pytest.mark.parametrize("empty", [False, True])
def test_zero_update_and_parameter_free_controls_do_not_invoke_scipy(
    monkeypatch: pytest.MonkeyPatch, *, empty: bool
) -> None:
    """Degenerate inputs have one well-defined metrics call and preserve parameters."""
    circuit, theta, target = problem()
    if empty:
        circuit, theta = ParameterizedCircuit(2, [], 0), np.empty(0)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("No optimizer is needed for a zero-update or parameter-free problem.")

    monkeypatch.setattr(study, "minimize", forbidden)
    returned, trace = study.optimize_lbfgsb(circuit, theta, target, steps=4 if empty else 0)
    np.testing.assert_array_equal(returned, theta)
    assert trace["objective_evaluations"] == trace["standalone_objective_evaluations"] == 1
    assert trace["gradient_evaluations"] == trace["steps"] == 0
    assert trace["returned_fidelity"] == pytest.approx(StatevectorBackend(circuit, target).metrics(theta)[1])
    assert trace["stop_reason"] == ("no_parameters" if empty else "zero_steps")


def test_fixed_depths_are_independent_and_metadata_scores_describe_returned_circuits() -> None:
    """No construction state leaks from a shallower independently optimized circuit."""
    _, _, target = problem()
    candidates = study.fixed_vqsp_candidates(
        target, depths=(0, 2), seed=300, hardware="Google", construction_steps=3, noisy_steps=3, training_paths=1000
    )
    for candidate, depth in zip(candidates, (0, 2), strict=True):
        info = candidate.metadata
        circuit = restore_circuit(info["parameterized_circuit"])
        expected = extend(None, circuit.num_params, 1300 + depth, 0.05)
        np.testing.assert_array_equal(info["initial_theta"], expected)
        assert info["parent_depth"] is None
        assert info["construction_lineage"] == [depth]
        assert info["status"] == ("complete" if info["training_bank"]["coverage_ok"] else "no_jump_coverage")
        assert info["method"] == "fixed_vqsp"
        assert info["candidate_id"] == f"depth_{depth:02d}"
        if info["status"] == "complete":
            assert info["noisy_training"]["returned_fidelity"] >= info["noisy_training"]["initial_fidelity"]
        else:
            assert info["noisy_training"]["returned_fidelity"] is None
        theta = np.array(info["theta"])
        state = Statevector.from_instruction(candidate.circuit).data
        assert StatevectorBackend(circuit, state).metrics(theta)[1] == pytest.approx(1, abs=2e-13)
        assert len(candidate.circuit.data) == len(circuit.gates)
        assert info["cpu_seconds"] >= info["construction_cpu_seconds"] + info["noisy_cpu_seconds"]
        json.dumps(info, allow_nan=False)
    assert candidates[1].metadata["status"] == "complete"
    assert candidates[1].metadata["cumulative_cpu_seconds"] >= sum(c.metadata["cpu_seconds"] for c in candidates)
    alone = study.fixed_vqsp_candidates(
        target, depths=(2,), seed=300, hardware="Google", construction_steps=3, noisy_steps=3, training_paths=1000
    )[0]
    np.testing.assert_array_equal(alone.metadata["theta"], candidates[1].metadata["theta"])


def test_layerwise_exactly_reuses_current_optimizers_and_unpruned_noiseless_parent() -> None:
    """Growth must retain every gate and inherit construction rather than noisy parameters."""
    _, _, target = problem()
    candidates = study.layerwise_candidates(
        target,
        depths=(1, 2),
        seed=300,
        hardware="Google",
        construction_steps=3,
        online_steps=1,
        noisy_steps=3,
        noisy_online_steps=1,
        training_paths=1000,
    )
    previous = np.empty(0)
    for depth, candidate in enumerate(candidates, start=1):
        info = candidate.metadata
        circuit = restore_circuit(info["parameterized_circuit"])
        initial = extend(previous, circuit.num_params, 1300 + depth, 0.05 if depth == 1 else 0.001)
        built, _ = train_noiseless(circuit, initial, target, policy="lbfgs", steps=3, online=1, rate=0.7)
        final, expected = train(
            circuit,
            built,
            target,
            steps=3,
            online=1,
            rate=0.2,
            seed=5300,
            noisy=True,
            hardware="Google",
            num_trajectories=1000,
            compress_no_jump=True,
            record_noise_banks=True,
            require_jump_coverage=True,
        )
        np.testing.assert_array_equal(info["initial_theta"], initial)
        np.testing.assert_array_equal(info["construction_theta"], built)
        np.testing.assert_array_equal(info["theta"], final)
        np.testing.assert_array_equal(info["noisy_training"]["fidelity_trace"], expected["fidelity_trace"])
        assert info["training_bank"]["raw_sha256"] == expected["noise_banks"][0]["raw_sha256"]
        assert info["noisy_training"]["sampler_calls"] == info["training_bank"]["sampler_calls"] == 1
        assert info["noisy_training"]["gradient_evaluations"] == 2
        assert info["noisy_training"]["objective_evaluations"] == 6
        assert info["noisy_training"]["online_sweeps"] == 1
        assert info["construction_lineage"] == list(range(1, depth + 1))
        assert info["parent_depth"] == (depth - 1 if depth > 1 else None)
        assert len(candidate.circuit.data) == 6 + 9 * depth
        previous = built
        json.dumps(info, allow_nan=False)
    first, second = (candidate.metadata for candidate in candidates)
    assert not np.array_equal(first["theta"], first["construction_theta"])
    np.testing.assert_array_equal(second["initial_theta"][: len(previous) - 9], first["construction_theta"])


@pytest.mark.parametrize("method", [study.fixed_vqsp_candidates, study.layerwise_candidates])
def test_missing_jumps_are_recorded_without_resampling_or_noisy_training(
    monkeypatch: pytest.MonkeyPatch, method: Callable
) -> None:
    """Coverage failure cannot silently become a completed noiseless fine-tuning arm."""
    _, _, target = problem()
    calls = 0

    def no_jumps(
        self: StatevectorBackend, _theta: np.ndarray, options: KrotovTJMOptions, *_args: object, **_kwargs: object
    ) -> tuple:
        nonlocal calls
        calls += 1
        return tuple(tuple(KrotovNoiseMap() for _ in self.circuit.gates) for _ in range(options.num_trajectories))

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("A failed coverage bank must not trigger noisy optimization.")

    monkeypatch.setattr(StatevectorBackend, "sample_noise_maps", no_jumps)
    monkeypatch.setattr(study, "train", forbidden)
    if method is study.fixed_vqsp_candidates:
        original = study.optimize_lbfgsb

        def guard(
            circuit: ParameterizedCircuit,
            initial: np.ndarray,
            target: np.ndarray,
            *,
            steps: int,
            maps: object = None,
            **_kwargs: object,
        ) -> tuple:
            assert maps is None
            return original(circuit, initial, target, steps=steps)

        monkeypatch.setattr(study, "optimize_lbfgsb", guard)
    candidate = method(target, depths=(1,), seed=1, hardware="IBM", construction_steps=0, noisy_steps=3)[0]
    info = candidate.metadata
    assert calls == 1
    assert info["status"] == "no_jump_coverage"
    assert info["training_bank"]["sampled_paths"] == 100
    assert info["training_bank"]["retained_paths"] == 1
    assert info["training_bank"]["weights"] == [1.0]
    assert info["noisy_training"]["returned_fidelity"] is None
    np.testing.assert_array_equal(info["theta"], info["construction_theta"])


def test_empty_circuit_bank_is_valid_and_uses_empirical_unit_weight() -> None:
    """An empty circuit has no physical noise window, so zero jumps are expected."""
    _, _, target = problem()
    maps, weights, bank = study.frozen_training_bank(
        ParameterizedCircuit(2, [], 0), np.empty(0), target, seed=1, hardware="IBM", training_paths=100
    )
    assert len(maps) == 1
    np.testing.assert_array_equal(weights, [1])
    assert bank["coverage_ok"]
    assert bank["multiplicities"] == [100]


@pytest.mark.parametrize("method", [study.fixed_vqsp_candidates, study.layerwise_candidates])
def test_zero_updates_still_report_the_same_valid_noisy_initial_point(method: Callable) -> None:
    """A zero budget measures the unchanged start rather than inventing an endpoint."""
    _, _, target = problem()
    candidate = method(
        target, depths=(1,), seed=300, hardware="Google", construction_steps=0, noisy_steps=0, training_paths=100
    )[0]
    info = candidate.metadata
    assert info["status"] == "complete"
    np.testing.assert_array_equal(info["theta"], info["initial_theta"])
    np.testing.assert_array_equal(info["construction_theta"], info["initial_theta"])
    assert info["construction"]["steps"] == info["noisy_training"]["steps"] == 0
    assert info["noisy_training"]["objective_evaluations"] == 1
    assert info["noisy_training"]["gradient_evaluations"] == 0
    assert info["noisy_training"]["returned_fidelity"] == pytest.approx(info["noisy_training"]["initial_fidelity"])


def test_legacy_bank_replay_mismatch_fails_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not silently continue if the reused routine changes the inspected bank."""
    _, _, target = problem()
    original = study.train

    def changed_bank(*args: object, **kwargs: object) -> tuple:
        runner = cast("Callable", original)
        theta, trace = runner(*args, **kwargs)
        trace["noise_banks"][0]["raw_sha256"] = "changed"
        return theta, trace

    monkeypatch.setattr(study, "train", changed_bank)
    with pytest.raises(RuntimeError, match="different training bank"):
        study.layerwise_candidates(
            target, depths=(1,), seed=300, hardware="Google", construction_steps=0, noisy_steps=0, training_paths=100
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"depths": ()},
        {"depths": (1, 1)},
        {"depths": (2, 1)},
        {"depths": (-1,)},
        {"depths": [1]},
        {"depths": (True,)},
        {"seed": -1},
        {"seed": 2**32},
        {"seed": True},
        {"construction_steps": -1},
        {"noisy_steps": 1.5},
        {"training_paths": 0},
        {"training_paths": True},
        {"hardware": "ibm"},
    ],
)
@pytest.mark.parametrize("method", [study.fixed_vqsp_candidates, study.layerwise_candidates])
def test_candidate_inputs_fail_closed(method: Callable, kwargs: dict) -> None:
    """Reject ambiguous seeds, incomplete grids and unknown physical noise profiles."""
    _, _, target = problem()
    arguments = {"depths": (1,), "seed": 300, "hardware": "IBM", "construction_steps": 0, "noisy_steps": 0}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=r"integer|depths|IBM|range"):
        method(target, **arguments)


@pytest.mark.parametrize("depths", [(0,), (2,), (1, 3)])
def test_layerwise_rejects_missing_ancestors(depths: tuple[int, ...]) -> None:
    """Every requested layerwise circuit must have its depth-one construction chain."""
    _, _, target = problem()
    with pytest.raises(ValueError, match="contiguous"):
        study.layerwise_candidates(target, depths=depths, seed=1, hardware="IBM")


@pytest.mark.parametrize("kwargs", [{"steps": -1}, {"steps": True}, {"steps": 2.5}, {"max_evaluations": 0}])
def test_optimizer_rejects_invalid_budgets(kwargs: dict) -> None:
    """The SciPy wrapper cannot accept fractional or nonpositive evaluation caps."""
    circuit, initial, target = problem()
    arguments: dict[str, Any] = {"steps": 2, **kwargs}
    with pytest.raises(ValueError, match="integer"):
        study.optimize_lbfgsb(circuit, initial, target, **arguments)


def test_optimizer_fails_on_nonfinite_true_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid numerical evaluations cannot be returned as successful candidates."""
    circuit, initial, target = problem()
    monkeypatch.setattr(StatevectorBackend, "gradient", lambda *_args, **_kwargs: (initial * np.nan, 0.5, 0.5))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        study.optimize_lbfgsb(circuit, initial, target, steps=1)
