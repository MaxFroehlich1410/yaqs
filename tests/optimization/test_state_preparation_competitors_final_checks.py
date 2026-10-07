# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check diagnostic algebra, preserved updates, receipts and resource guards."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import random_statevector

pytest.importorskip("torch")
import torch

from experiments.state_preparation_competitors import q12_final_checks as checks
from experiments.state_preparation_competitors.bounded_probe import main as bounded_main
from experiments.state_preparation_competitors.bounded_probe import process_memory, run_bounded
from experiments.state_preparation_competitors.checkpointed_dense import CheckpointedDenseObjective
from experiments.state_preparation_competitors.iqsp_comb_backend import (
    CombLayout,
    CombOptions,
    CombTensorNetworkObjective,
    comb_target_from_vector,
)
from experiments.state_preparation_competitors.robuststate import (
    robuststate_loss_gradient,
    topology_aware_ansatz,
    train_robuststate,
)


def test_pure_hs_algebra_and_adam_path() -> None:
    """The ket identity reproduces the existing density-based loss and every update."""
    target = random_statevector(8, seed=82).data
    circuit = topology_aware_ansatz(3, 2)
    engine = CheckpointedDenseObjective(circuit, target, None, compile_native=False)
    theta = np.random.default_rng(5).normal(size=circuit.num_parameters)
    expected_loss, expected_gradient = robuststate_loss_gradient(engine, theta, noisy=False)
    loss, gradient = checks.pure_hilbert_schmidt_gradient(engine, theta)
    assert loss == pytest.approx(expected_loss, abs=2e-14)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-14)
    arrays, receipt = checks.warm_robuststate(engine, theta, steps=12)
    final, original = train_robuststate(engine, theta, seed=1, noiseless_steps=12, noisy_steps=0)
    np.testing.assert_allclose(arrays["theta"], final, atol=2e-13)
    np.testing.assert_allclose(arrays["parameter_history"], [r["theta"] for r in original["records"]], atol=2e-13)
    assert receipt["retention"] == "final_iterate"
    assert arrays["first"].shape == theta.shape
    with pytest.raises(ValueError, match="noiseless"):
        checks.pure_hilbert_schmidt_gradient(CheckpointedDenseObjective(circuit, target), theta)


def test_zero_loss_and_invalid_pure_fidelity(monkeypatch: pytest.MonkeyPatch) -> None:
    """At zero loss, preserve the published zero norm subgradient convention."""
    circuit = topology_aware_ansatz(2, 1)
    engine = CheckpointedDenseObjective(circuit, [1, 0, 0, 0], None, compile_native=False)
    loss, gradient = checks.pure_hilbert_schmidt_gradient(engine, np.zeros(6))
    assert loss == 0
    np.testing.assert_array_equal(gradient, np.zeros(6))
    monkeypatch.setattr(engine, "loss_gradient", lambda _theta: (-0.1, np.ones(6)))
    with pytest.raises(ValueError, match="infidelity"):
        checks.pure_hilbert_schmidt_gradient(engine, np.zeros(6))


def test_gradient_and_sampling_statistics() -> None:
    """Signs, scale differences, zero norms, and uncertainty are explicit."""
    assert checks.gradient_comparison([1, 0], [2, 0])["relative_error"] == 1
    assert checks.gradient_comparison([1, 0], [-1, 0])["cosine"] == -1
    assert checks.gradient_comparison([0, 0], [0, 0])["cosine"] is None
    stats = checks.mean_interval([-1, 0, 1])
    assert stats["mean"] == 0
    assert stats["positive_count"] == 1
    assert stats["mean_ci95"][0] < 0 < stats["mean_ci95"][1]
    assert checks.mean_interval([2, 2, 2])["mean_ci95"] == [2, 2]
    with pytest.raises(ValueError, match="finite"):
        checks.gradient_comparison([1], [np.nan])
    with pytest.raises(ValueError, match="two finite"):
        checks.mean_interval([1])


def test_ballarin_probe_receipt_and_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A small real noisy probe preserves the frozen parameters and uses its target bank."""
    circuit = checks.iqsp_tn_ansatz(6, 1)
    source = tmp_path / "checkpoint.json"
    source.write_text(json.dumps({"num_qubits": 6, "depth": 1, "final_theta": [0.1] * circuit.num_parameters}))
    monkeypatch.setattr(
        checks,
        "tci_target_bank",
        lambda targets, **_kw: (
            (comb_target_from_vector(targets[0], CombLayout(6)),),
            {"child_cpu_seconds": 0, "test_oracle": True},
        ),
    )
    for compression in ("default", "tighter"):
        result = checks.ballarin_probe(tmp_path, depth=1, compression=compression, checkpoint=source, num_qubits=6)
        assert result["compression"]["checkpoint_interval"] == 1
        assert 0 <= result["fidelity"] <= 1
        assert (tmp_path / f"ballarin_depth1_{compression}.npz").exists()
    first = json.loads((tmp_path / "ballarin_depth1_default.json").read_text())
    second = json.loads((tmp_path / "ballarin_depth1_tighter.json").read_text())
    assert first["theta_sha256"] == second["theta_sha256"]
    original_gradient = np.load(tmp_path / "ballarin_depth1_tighter.npz")["gradient"]
    retry = checks.ballarin_probe(
        tmp_path,
        depth=1,
        compression="tighter",
        checkpoint=source,
        num_qubits=6,
        max_tensor_elements=2**26,
        checkpoint_interval=2,
        compact_checkpoints=True,
    )
    assert retry["compression"]["max_tensor_elements"] == 2**26
    assert retry["compression"]["checkpoint_interval"] == 2
    assert retry["compact_checkpoints"] is True
    assert retry["fidelity"] == pytest.approx(second["fidelity"], abs=1e-14)
    np.testing.assert_allclose(
        np.load(tmp_path / "ballarin_depth1_tighter.npz")["gradient"], original_gradient, atol=1e-12
    )
    with pytest.raises(ValueError, match="compression"):
        checks.ballarin_probe(tmp_path, depth=1, compression="bad", checkpoint=source)
    with pytest.raises(ValueError, match="Checkpoint"):
        checks.ballarin_probe(tmp_path, depth=2, compression="default", checkpoint=source)


def test_small_complete_feedback_probe(tmp_path: Path) -> None:
    """Real feedback, preserved moments, held-fixed repeats and final files work together."""
    result = checks.robuststate_probe(tmp_path, target_id="haar_random_1", num_qubits=6, trials=2)
    assert result["warm"]["steps"] == 500
    assert result["shots_per_update"] == 40960
    assert len(result["records"]) == 2
    assert all(row["independent_noisy_fidelity_after"] is not None for row in result["records"])
    assert len({row["seed"] for row in result["records"]}) == 2
    archive = np.load(tmp_path / "robuststate_haar_random_1_feedback.npz")
    assert archive["gradients"].shape[0] == 2
    assert (tmp_path / "robuststate_haar_random_1.json").exists()
    with pytest.raises(ValueError, match="two"):
        checks.robuststate_probe(tmp_path, target_id="haar_random_1", trials=1)


def test_declared_basis_budget_preserves_warm_parameters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only sample count and its seed change; ideal parameters and Adam moments agree."""
    sampler = Mock(wraps=checks.classical_shadow_feedback)
    monkeypatch.setattr(checks, "classical_shadow_feedback", sampler)
    default_dir, larger_dir = tmp_path / "default", tmp_path / "larger"
    default_dir.mkdir()
    larger_dir.mkdir()
    default = checks.robuststate_probe(default_dir, target_id="haar_random_1", num_qubits=6, trials=2)
    larger = checks.robuststate_probe(larger_dir, target_id="haar_random_1", num_qubits=6, trials=2, num_bases=80)
    assert [call.kwargs["num_bases"] for call in sampler.call_args_list] == [40, 40, 80, 80]
    assert larger["bases"] == 80
    assert larger["shots_per_update"] == 80 * 1024
    for key in ("theta_sha256", "first_moment_sha256", "second_moment_sha256", "initialization_seed"):
        assert default[key] == larger[key]
    expected_seed = int(np.random.SeedSequence([default["measurement_seed"], 120040, 0]).generate_state(1)[0])
    assert default["records"][0]["seed"] == expected_seed
    assert default["records"][0]["seed"] != larger["records"][0]["seed"]
    for invalid in (0, True, 1.5, 730):
        invalid_options: dict[str, Any] = {"num_bases": invalid}
        with pytest.raises(ValueError, match="num_bases"):
            checks.robuststate_probe(tmp_path, target_id="haar_random_1", num_qubits=6, **invalid_options)


def test_probe_command_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI dispatch never silently chooses a different checkpoint or method."""
    with pytest.raises(SystemExit):
        checks.main(["ballarin", "--output", str(tmp_path)])
    called = []
    monkeypatch.setattr(checks, "robuststate_probe", lambda *_a, **kw: called.append(kw))
    checks.main(["robuststate", "--output", str(tmp_path), "--trials", "2", "--num-bases", "400", "--trial-start", "8"])
    assert called[0]["trials"] == 2
    assert called[0]["num_bases"] == 400
    assert called[0]["trial_start"] == 8
    monkeypatch.setattr(checks, "ballarin_probe", lambda *_a, **kw: called.append(kw))
    checks.main([
        "ballarin",
        "--output",
        str(tmp_path),
        "--checkpoint",
        "test.json",
        "--max-tensor-elements",
        "67108864",
        "--checkpoint-interval",
        "2",
        "--compact-checkpoints",
    ])
    assert called[1]["checkpoint"] == Path("test.json")
    assert called[1]["max_tensor_elements"] == 2**26
    assert called[1]["checkpoint_interval"] == 2
    assert called[1]["compact_checkpoints"] is True


def test_feedback_shards_preserve_global_trial_seeds(tmp_path: Path) -> None:
    """A separate trial range reproduces the same feedback and frozen Adam update."""
    full, shard = tmp_path / "full", tmp_path / "shard"
    full.mkdir()
    shard.mkdir()
    baseline = checks.robuststate_probe(full, target_id="haar_random_1", num_qubits=6, trials=3)
    subset = checks.robuststate_probe(shard, target_id="haar_random_1", num_qubits=6, trials=2, trial_start=1)
    assert subset["trial_start"] == 1
    assert [row["trial"] for row in subset["records"]] == [1, 2]
    assert [row["seed"] for row in subset["records"]] == [row["seed"] for row in baseline["records"]][1:]
    first = np.load(full / "robuststate_haar_random_1_feedback.npz")
    second = np.load(shard / "robuststate_haar_random_1_feedback.npz")
    for key in ("gradients", "updates"):
        np.testing.assert_array_equal(second[key], first[key][1:])
    for key in ("exact_gradient", "exact_update"):
        np.testing.assert_array_equal(second[key], first[key])
    assert subset["records"][0]["independent_noisy_fidelity_after"] is not None
    assert subset["records"][1]["independent_noisy_fidelity_after"] is None
    with pytest.raises(ValueError, match="trial_start"):
        checks.robuststate_probe(tmp_path, target_id="haar_random_1", trial_start=-1)


@pytest.mark.parametrize("hardware", [None, "IBM", "H2"])
def test_compact_checkpoint_storage_and_gradient(hardware: str | None) -> None:
    """Owning compact storage removes large backing views without changing derivatives."""
    target = comb_target_from_vector(random_statevector(16, seed=81).data, CombLayout(4))
    circuit = QuantumCircuit(4)
    parameters = ParameterVector("compact", 7)
    for repeat in range(3):
        for site in range(4):
            circuit.ry(parameters[site] * (repeat + 1), site)
        for site in range(3):
            circuit.rzz(parameters[site + 4], site, site + 1)
    theta = np.random.default_rng(91).normal(size=7)
    options = CombOptions(closed_bond=2, environment_bond=2, checkpoint_interval=1)
    original = CombTensorNetworkObjective(circuit, target, hardware, options=options)
    compact = checks.CompactCheckpointObjective(circuit, target, hardware, options=options)
    expected_loss, expected_gradient = original.loss_gradient(theta)
    loss, gradient = compact.loss_gradient(theta)
    assert loss == pytest.approx(expected_loss, abs=2e-14)
    np.testing.assert_allclose(gradient, expected_gradient, atol=2e-12, rtol=2e-10)
    assert compact.last_diagnostics["trace"] == pytest.approx(original.last_diagnostics["trace"], abs=2e-14, rel=0)
    assert {k: v for k, v in compact.last_diagnostics.items() if k != "trace"} == {
        k: v for k, v in original.last_diagnostics.items() if k != "trace"
    }
    with torch.no_grad():
        # Inspect the actual saved boundary tensors; public fidelity hides ownership.
        state = compact._new_state()  # noqa: SLF001 -- white-box checkpoint storage regression
        values = torch.as_tensor(theta, dtype=torch.float64)
        tensors = compact._checkpoint_segment(  # noqa: SLF001 -- inspect retained checkpoint storage
            0, len(circuit.data), 0, (1,) * 4, values, *state.states
        )
        unowned = original._checkpoint_segment(  # noqa: SLF001 -- compare original boundary storage
            0, len(circuit.data), 0, (1,) * 4, values, *state.states
        )
    assert all(t.untyped_storage().nbytes() == t.numel() * t.element_size() for t in tensors)
    assert sum(t.untyped_storage().nbytes() for t in tensors) <= sum(t.untyped_storage().nbytes() for t in unowned)
    if hardware == "IBM":
        assert sum(t.untyped_storage().nbytes() for t in tensors) < sum(t.untyped_storage().nbytes() for t in unowned)


def test_process_guard_success_timeout_and_memory(tmp_path: Path) -> None:
    """The guard terminates only its child, retains logs, and distinguishes failures."""
    assert process_memory(os.getpid())[0] > 0
    command = [sys.executable, "-c", "print('ok')"]
    result = run_bounded(command, tmp_path / "ok.log")
    assert result["status"] == "completed"
    assert (tmp_path / "ok.log").read_text().strip() == "ok"
    sleeper = [sys.executable, "-c", "import time; time.sleep(2)"]
    assert run_bounded(sleeper, tmp_path / "time.log", seconds=0.02)["status"] == "time_limit"
    assert run_bounded(sleeper, tmp_path / "memory.log", memory_bytes=1)["status"] == "memory_limit"
    with pytest.raises(ValueError, match="Require"):
        run_bounded([], tmp_path / "invalid.log")
    bounded_main(["--log", str(tmp_path / "cli.log"), "--", *command])
    with pytest.raises(SystemExit):
        bounded_main(["--log", str(tmp_path / "fail.log"), "--", sys.executable, "-c", "raise SystemExit(1)"])
