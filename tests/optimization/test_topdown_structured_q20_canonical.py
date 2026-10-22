# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Sealed fixed-bank continuation checks without modifying an original campaign."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from experiments import topdown_backend_support as storage
from experiments import topdown_structured_q20 as pilot
from experiments import topdown_structured_q20_canonical as continuation
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import KrotovNoiseMap, MPSBackend, ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture
def completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, pilot.Settings]:
    """Create a complete small B branch using the same weighted replay machinery.

    Returns:
        Original branch folder and its fixed update settings.
    """
    folder = tmp_path / "original"
    circuit = ParameterizedCircuit(
        2, [ParameterizedGate("ry", (0,), param_index=0), ParameterizedGate("rxx", (0, 1), param_index=1)]
    )
    theta = np.array([0.8, 1.1])
    target = MPS.from_statevector(np.array([1, 2j, 3, 4j]) / np.sqrt(30))
    storage.save_checkpoint(
        folder / "target",
        {"qubits": 2, "approximation": {"pure_target_fidelity_error_bound": 0.0}},
        {f"site_{i}": t for i, t in enumerate(target.tensors)},
    )
    storage.save_checkpoint(folder / "parent", {"circuit": storage.circuit_record(circuit)}, {"theta": theta})

    def bank(self: MPSBackend, *_args: object, **_kwargs: object) -> list[tuple]:
        """Use a fixed no-jump/one-jump pair without resampling.

        Returns:
            Two physical unitary replay paths.
        """
        empty = tuple(KrotovNoiseMap() for _ in self.circuit.gates)
        jump = list(empty)
        jump[0] = KrotovNoiseMap(operators=((np.array([[0, 1], [1, 0]], dtype=complex), (0,)),))
        return [empty, tuple(jump)]

    monkeypatch.setattr(MPSBackend, "sample_noise_maps", bank)
    settings = replace(pilot.Settings(), qubits=2, noisy_online=2, b_batch=2, training_paths=2)
    pilot.train_branch(folder, "B", settings, backend_factory=continuation.CanonicalProgressBackend)
    return folder, settings


def partial(original: Path, destination: Path) -> None:
    """Copy only sealed inputs and the first two completed updates."""
    for name in ("target", "parent", "B/pruning"):
        shutil.copytree(original / name, destination / name)
    shutil.copyfile(original / "B/noise-bank.json", destination / "B/noise-bank.json")
    for step in range(3):
        name = f"B/iterates/step_{step:05d}"
        shutil.copytree(original / name, destination / name)


@pytest.mark.parametrize("relative_cutoff", [float(np.finfo(float).eps), 1e-8])
def test_resume_preserves_complete_update_sequence_and_history(
    completed: tuple, tmp_path: Path, relative_cutoff: float
) -> None:
    """Resume across the online/batch boundary without repeating updates or sampling."""
    original, settings = completed
    folder = tmp_path / "continuation"
    partial(original, folder)
    old_seals = {p: p.read_bytes() for p in (folder / "B/iterates").glob("*/complete.json")}
    old_bank = (folder / "B/noise-bank.json").read_bytes()
    continuation.resume_b(folder, settings, relative_cutoff=relative_cutoff)
    actual = storage.load_checkpoint(folder / "B/selection")
    expected = storage.load_checkpoint(original / "B/selection")
    assert actual is not None
    assert expected is not None
    np.testing.assert_allclose(actual[1]["theta"], expected[1]["theta"], atol=1e-12)
    assert [row["step"] for row in actual[0]["history"]] == list(range(5))
    assert actual[0]["resume_step"] == 2
    assert actual[0]["historical_kernel_cpu_seconds"] == actual[0]["history"][2]["kernel_cpu_seconds"]
    started = json.loads((folder / "B/continuation-started.json").read_text())
    assert started["resume_step"] == 2
    assert started["remaining_updates"] == 2
    assert started["relative_cutoff"] == relative_cutoff
    assert started["input_relative_cutoff"] == np.finfo(float).eps
    for step in (3, 4):
        saved = storage.load_checkpoint(folder / f"B/iterates/step_{step:05d}")
        assert saved is not None
        assert saved[0]["relative_cutoff"] == relative_cutoff
        assert saved[0]["input_relative_cutoff"] == np.finfo(float).eps
    for path, seal in old_seals.items():
        assert path.read_bytes() == seal
    assert (folder / "B/noise-bank.json").read_bytes() == old_bank
    with pytest.raises(ValueError, match="selection already exists"):
        continuation.resume_b(folder, settings)


def test_historical_best_is_retained(completed: tuple, tmp_path: Path) -> None:
    """An earlier retained optimum stays eligible after the representation change."""
    original, settings = completed
    folder = tmp_path / "continuation"
    partial(original, folder)
    checkpoint = folder / "B/iterates/step_00000"
    saved = storage.load_checkpoint(checkpoint)
    assert saved is not None
    record = {key: value for key, value in saved[0].items() if key != "arrays"}
    storage.save_checkpoint(checkpoint, {**record, "fidelity": 1.0}, saved[1])
    continuation.resume_b(folder, settings)
    selection = storage.load_checkpoint(folder / "B/selection")
    assert selection is not None
    assert selection[0]["selected_step"] == 0
    np.testing.assert_array_equal(selection[1]["theta"], saved[1]["theta"])


def test_resume_rejects_missing_history(completed: tuple, tmp_path: Path) -> None:
    """Missing original checkpoints cannot be silently skipped."""
    original, settings = completed
    folder = tmp_path / "continuation"
    partial(original, folder)
    shutil.rmtree(folder / "B/iterates/step_00001")
    with pytest.raises(ValueError, match="non-contiguous"):
        continuation.resume_b(folder, settings)


def test_replay_rejects_changed_weights(completed: tuple, tmp_path: Path) -> None:
    """Compressed trajectory probabilities must match the saved physical paths."""
    original, settings = completed
    folder = tmp_path / "continuation"
    partial(original, folder)
    path = folder / "B/noise-bank.json"
    bank = json.loads(path.read_text())
    bank["weights"] = [0.9, 0.1]
    path.write_text(json.dumps(bank))
    with pytest.raises(ValueError, match="replay bank differs"):
        continuation.resume_b(folder, settings)


def test_original_root_requires_explicit_new_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An original campaign cannot accidentally select the approximate backend."""
    monkeypatch.setattr(pilot, "verify", lambda _root: {})
    with pytest.raises(ValueError, match="not sealed"):
        continuation.verify(tmp_path)


@pytest.mark.parametrize("fail_b", [False, True])
def test_controller_claim_and_stage_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fail_b: bool
) -> None:
    """Replace one seat only, skip preparation and never retry failed training."""
    plan = {
        "memory_guard_gib": 96,
        "canonical_continuation": {
            "targets": ["tfim_critical"],
            "host": "testhost",
            "admission_available_gib": 256,
        },
    }
    monkeypatch.setattr(continuation, "verify", lambda _root: plan)
    monkeypatch.setattr(continuation.socket, "gethostname", lambda: "testhost")
    monkeypatch.setattr(continuation.psutil, "virtual_memory", lambda: SimpleNamespace(available=1024 * pilot.GIB))
    calls = []

    def run(_root: Path, _target: str, stage: str, _host: str, _plan: dict, **kwargs: str) -> tuple:
        """Observe the frozen module dispatch without running a server process.

        Returns:
            A synthetic exit code and resource receipt.
        """
        calls.append(stage)
        assert kwargs["leaf_module"] == continuation.MODULE
        return (1 if fail_b and stage == "B" else 0), {}

    monkeypatch.setattr(pilot, "run_stage", run)
    continuation.supervise(tmp_path, "tfim_critical", "testhost")
    assert "prepare" not in calls
    assert calls.count("B") == 1
    assert ("B_selected_mpo" in calls) is not fail_b
    assert "G" in calls
    assert "G_selected_mpo" in calls
    with pytest.raises(FileExistsError):
        continuation.supervise(tmp_path, "tfim_critical", "testhost")


def test_continuation_inputs_are_authenticated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing an original bank or checkpoint after sealing invalidates the root."""
    source = tmp_path / "original-bank.json"
    source.write_text("{}\n")
    plan = {
        "canonical_continuation": {
            "version": continuation.VERSION,
            "relative_cutoff": float(np.finfo(float).eps),
            "inputs": {source.name: storage.digest(source)},
        }
    }
    monkeypatch.setattr(pilot, "verify", lambda _root: plan)
    assert continuation.verify(tmp_path) is plan
    source.write_text("[]\n")
    with pytest.raises(ValueError, match="Continuation input changed"):
        continuation.verify(tmp_path)


@pytest.mark.parametrize("input_cutoff", [float(np.finfo(float).eps), 1e-8])
def test_explicit_truncation_rule_preserves_target_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, input_cutoff: float
) -> None:
    """A stronger propagation rule cannot silently loosen the sealed input rule."""
    source = tmp_path / "input.json"
    source.write_text("{}\n")
    plan = {
        "canonical_continuation": {
            "version": continuation.TRUNCATED_VERSION,
            "relative_cutoff": 1e-8,
            "input_relative_cutoff": input_cutoff,
            "inputs": {source.name: storage.digest(source)},
        }
    }
    monkeypatch.setattr(pilot, "verify", lambda _root: plan)
    if input_cutoff == np.finfo(float).eps:
        assert continuation.verify(tmp_path) is plan
    else:
        with pytest.raises(ValueError, match="preserved input precision"):
            continuation.verify(tmp_path)


def test_dispatch_uses_cutoff_for_g_and_keeps_mpo_assessment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """G inherits propagation precision while final MPO dispatch remains independent."""
    calls = []

    def train(
        folder: Path,
        policy: str,
        settings: pilot.Settings,
        *,
        backend_factory: Callable[..., continuation.CanonicalProgressBackend],
    ) -> None:
        """Capture the actual backend constructed for the G stage."""
        circuit = ParameterizedCircuit(2, [ParameterizedGate("rxx", (0, 1), param_index=0)])
        engine = backend_factory(circuit, MPS(2), folder / policy)
        calls.append((policy, engine.relative_cutoff, settings))

    monkeypatch.setattr(pilot, "train_branch", train)
    monkeypatch.setattr(pilot, "execute_stage", lambda *args: calls.append(args))
    settings = pilot.Settings()
    continuation.execute_stage(tmp_path, "tfim_critical", "G", settings, relative_cutoff=1e-8)
    continuation.execute_stage(tmp_path, "tfim_critical", "B_selected_mpo", settings, relative_cutoff=1e-8)
    assert calls == [("G", 1e-8, settings), (tmp_path, "tfim_critical", "B_selected_mpo", settings)]
