# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exercise native preparation, real B/G updates, cutoff behavior and final assessment."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from experiments import topdown_backend_support as storage
from experiments import topdown_mpo_validation as validation
from experiments import topdown_native_mps as native
from experiments import topdown_structured_q20 as pilot
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import KrotovNoiseMap, MPSBackend, ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def test_required_cutoff_actually_discards_small_propagated_directions(tmp_path: Path) -> None:
    """A tiny Schmidt component survives input loading but is removed by a real gate replay."""
    target = MPS.from_statevector(np.array([1, 0, 0, 1e-10j], dtype=complex))
    circuit = ParameterizedCircuit(2, [ParameterizedGate("rxx", (0, 1), param_index=0)])
    engine = native.backend(circuit, target, tmp_path)
    assert engine.relative_cutoff == pytest.approx(1e-8, rel=0, abs=0)
    np.testing.assert_allclose(engine.target.to_vec(), target.to_vec(), atol=1e-15, rtol=0)
    engine.metrics(np.array([2e-10]))
    assert engine.diagnostics.discarded_weight_sum > 1e-23
    assert engine.diagnostics.cap_reductions == 0
    record = json.loads((tmp_path / "progress.json").read_text())
    assert record["relative_cutoff"] == pytest.approx(1e-8, rel=0, abs=0)
    assert record["input_relative_cutoff"] == np.finfo(float).eps


def test_real_native_pipeline_prunes_and_updates_with_the_required_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use real optimizers and fixed weighted noise, then independently compare full final densities."""
    root = tmp_path / "pilot"
    settings = replace(
        pilot.Settings(),
        qubits=2,
        max_depth=1,
        ideal_online=3,
        ideal_batch=10,
        noisy_online=1,
        b_batch=1,
        training_paths=2,
        product_candidates=17,
    )
    native.create(root, settings)
    expected = native.verify(root)[2].to_vec()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        """Production stages must not expand either target or propagated states."""
        pytest.fail("Unexpected dense conversion in native pilot")

    pruning_engines = []
    original = native.CanonicalMPSBackend

    class ObservedPruning(original):
        """Observe actual pruning engines while executing their original numerical kernels."""

        def __init__(
            self,
            circuit: ParameterizedCircuit,
            target: MPS | NDArray[np.complex128],
            initial_state: MPS | NDArray[np.complex128] | None = None,
            *,
            relative_cutoff: float = 1e-8,
        ) -> None:
            super().__init__(circuit, target, initial_state, relative_cutoff=relative_cutoff)
            pruning_engines.append(self.relative_cutoff)

    def bank(self: MPSBackend, *_args: object, **_kwargs: object) -> list[tuple]:
        """Return the same physical no-jump/one-jump pair without adaptive resampling."""
        empty = tuple(KrotovNoiseMap() for _ in self.circuit.gates)
        assert empty, "The test must exercise a retained noisy circuit"
        jump = list(empty)
        jump[0] = KrotovNoiseMap(operators=((np.array([[0, 1], [1, 0]], dtype=complex), (0,)),))
        return [empty, tuple(jump)]

    monkeypatch.setattr(native, "CanonicalMPSBackend", ObservedPruning)
    monkeypatch.setattr(MPSBackend, "sample_noise_maps", bank)
    monkeypatch.setattr(MPS, "to_vec", forbidden)
    monkeypatch.setattr(MPS, "from_statevector", forbidden)
    for stage in pilot.STAGES:
        native.run_stage(root, stage)
    monkeypatch.undo()
    assert pruning_engines
    assert set(pruning_engines) == {1e-8}
    for policy in ("B", "G"):
        pruning = storage.load_checkpoint(root / f"job/{policy}/pruning")
        selection = storage.load_checkpoint(root / f"job/{policy}/selection")
        final = storage.load_checkpoint(root / f"job/{policy}/mpo-selected")
        assert pruning is not None
        assert selection is not None
        assert final is not None
        assert len(selection[0]["history"]) >= 2
        assert final[0]["diagnostics"]["exact_up_to_roundoff"]
        assert final[0]["diagnostics"]["settings"]["svd_atol"] == 0
        assert final[0]["diagnostics"]["settings"]["max_bond_dim"] is None
        progress = json.loads((root / f"job/{policy}/progress.json").read_text())
        assert progress["relative_cutoff"] == pytest.approx(1e-8, rel=0, abs=0)
        circuit = storage.restore_circuit(pruning[0]["circuit"])
        # The Qiskit/dense reference is used only by this small independent oracle.
        independent = validation.compare_density(circuit, selection[1]["theta"], MPS.from_statevector(expected), "IBM")
        assert independent["passed"]
        assert final[0]["fidelity"] == pytest.approx(independent["density_fidelity"], abs=1e-12)
    with pytest.raises(FileExistsError):
        native.run_stage(root, "prepare")


def test_thirty_qubit_prepare_and_complete_noisy_update_without_dense_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute real 30-qubit preparation and a weighted noisy update with dense APIs trapped."""
    root = tmp_path / "q30"
    settings = replace(pilot.Settings(), qubits=30, max_depth=1, ideal_online=1, ideal_batch=1, product_candidates=17)
    native.create(root, settings)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        """Reject exponential target conversions in the full-width smoke test."""
        pytest.fail("Dense expansion in 30-qubit optimization")

    monkeypatch.setattr(MPS, "to_vec", forbidden)
    monkeypatch.setattr(MPS, "from_statevector", forbidden)
    native.run_stage(root, "prepare")
    saved = storage.load_checkpoint(root / "job/parent")
    assert saved is not None
    assert saved[0]["circuit"]["num_qubits"] == 30
    target = native.verify(root)[2]
    circuit = storage.restore_circuit(saved[0]["circuit"])
    engine = native.backend(circuit, target, root / "probe")
    empty = tuple(KrotovNoiseMap() for _ in circuit.gates)
    noisy = list(empty)
    noisy[0] = KrotovNoiseMap(operators=((np.array([[0, 1], [1, 0]], dtype=complex), (0,)),))
    jump = tuple(noisy)
    theta = saved[1]["theta"]
    updated, signal = engine.online_update(theta, 0.2, [empty, jump], estimator="cross", weights=np.array([0.7, 0.3]))
    assert np.isfinite(updated).all()
    assert np.isfinite(signal).all()
    assert np.linalg.norm(signal) > 0
    engine.metrics(updated, [empty, jump], weights=np.array([0.7, 0.3]))
    assert engine.relative_cutoff == pytest.approx(1e-8, rel=0, abs=0)
    assert max(t.shape[2] for t in engine.target.tensors) <= 2
    assert engine.diagnostics.maximum_retained_tensor_bytes < 10_000_000


@pytest.mark.parametrize("change", ["cutoff", "target", "sources", "width", "seed"])
def test_sealed_pilot_rejects_changed_identity(tmp_path: Path, change: str) -> None:
    """A changed cutoff, width, numerical source or target cannot silently resume a pilot."""
    root = tmp_path / "pilot"
    native.create(root, replace(pilot.Settings(), qubits=2))
    path = root / "plan.json"
    plan = json.loads(path.read_text())
    if change == "cutoff":
        plan["precision"]["relative_cutoff"] = 1e-6
    elif change == "target":
        plan["target_manifest_sha256"] = "bad"
    elif change == "sources":
        plan["sources"] = {}
    elif change == "width":
        plan["settings"]["qubits"] = 3
    else:
        plan["settings"]["seed"] = 400
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match=r"identity|width"):
        native.verify(root)
