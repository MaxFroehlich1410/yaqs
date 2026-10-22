# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent dense oracles and bounded 30-qubit checks for native targets."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmarks.state_preparation import load_target, mps_targets
from benchmarks.state_preparation.binary_targets import file_sha256
from experiments import topdown_product_screening as screening
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import StatevectorBackend
from mqt.yaqs.optimization.mps_backend import mps_overlap

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("bond", [2, 3])
def test_native_family_matches_independent_archived_dense_fixture(bond: int) -> None:
    """Recover the same physical state and little-endian order, allowing the declared phase."""
    expected = load_target(6, f"random_mps_bond{bond}").state_vector
    actual = mps_targets.random_mps_target(6, bond, 5000 + bond).to_vec()
    phase = np.vdot(expected, actual)
    np.testing.assert_allclose(actual, phase * expected, atol=2e-14, rtol=0)
    assert abs(phase) == pytest.approx(1, abs=1e-14)
    assert actual[0].real > 0


def test_complex_mps_screening_matches_dense_products_and_parameter_stream() -> None:
    """Conjugation, basis order, all candidate scores and entangling angles match dense oracles."""
    vector = np.array([1, 4j, -2, 3j]) / np.sqrt(30)
    target = MPS.from_statevector(vector)
    before = [t.copy() for t in target.tensors]
    dense, dense_arrays = screening.screen_products(vector, 37, 300, 7)
    native, native_arrays = screening.screen_products_mps(target, 37, 300, 7)
    for name in ("polar", "azimuth"):
        np.testing.assert_array_equal(native_arrays[name], dense_arrays[name])
    np.testing.assert_allclose(native_arrays["fidelity"], dense_arrays["fidelity"], atol=1e-14, rtol=0)
    assert native["selected_index"] == dense["selected_index"]
    for batch in (1, 13):
        _, other = screening.screen_products_mps(target, 37, 300, batch)
        np.testing.assert_allclose(other["fidelity"], native_arrays["fidelity"], atol=1e-14, rtol=0)
    _, other_seed = screening.screen_products_mps(target, 37, 400, 7)
    assert not np.array_equal(other_seed["polar"], native_arrays["polar"])
    dense, dense_arrays = screening.initial_parameters(vector, 300, "screened_product", 37, 7)
    native, native_arrays = screening.initial_parameters_mps(target, 300, 37, 7)
    np.testing.assert_allclose(native_arrays["theta"], dense_arrays["theta"], atol=2e-14, rtol=0)
    assert native["initial_circuit_fidelity"] == pytest.approx(dense["initial_circuit_fidelity"], abs=2e-14)
    for actual, expected in zip(target.tensors, before, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_actual_thirty_qubit_generation_storage_and_screening_are_tensor_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tripwires prohibit dense expansion while exercising a real 30-qubit target and 1000 starts."""

    def forbidden(*_args: object, **_kwargs: object) -> None:
        """Fail if any exponential state conversion or dense backend is reached."""
        pytest.fail("Dense expansion of the thirty-qubit target")

    monkeypatch.setattr(MPS, "to_vec", forbidden)
    monkeypatch.setattr(MPS, "from_statevector", forbidden)
    monkeypatch.setattr(MPS, "scalar_product", forbidden)
    monkeypatch.setattr(StatevectorBackend, "__init__", forbidden)
    manifest = tmp_path / "target.json"
    record = mps_targets.write_mps_target(manifest)
    target, loaded = mps_targets.load_mps_target(manifest)
    assert loaded == record
    assert target.length == 30
    assert len(target.tensors) == 30
    assert max(max(t.shape[1:]) for t in target.tensors) == 2
    assert sum(t.nbytes for t in target.tensors) == 3712
    assert manifest.with_suffix(".npz").stat().st_size < 20_000
    assert mps_overlap(target, target) == pytest.approx(1, abs=1e-13)
    metadata, arrays = screening.initial_parameters_mps(target, 300, 1000, 8)
    assert arrays["polar"].shape == (1001, 30)
    assert metadata["selected_product_fidelity"] >= metadata["zero_state_fidelity"]
    assert np.isfinite(metadata["initial_circuit_fidelity"])


def test_target_seed_reproducibility_and_private_loads(tmp_path: Path) -> None:
    """Seeds determine tensors, repeated loads own their arrays, and existing files are preserved."""
    a = mps_targets.random_mps_target(6, 2, 5002)
    b = mps_targets.random_mps_target(6, 2, 5002)
    c = mps_targets.random_mps_target(6, 2, 5003)
    for left, right in zip(a.tensors, b.tensors, strict=True):
        np.testing.assert_array_equal(left, right)
    assert abs(mps_overlap(a, c)) < 0.99
    path = tmp_path / "target.json"
    mps_targets.write_mps_target(path, 6)
    loaded, _ = mps_targets.load_mps_target(path)
    loaded.tensors[0][:] = 0
    fresh, _ = mps_targets.load_mps_target(path)
    assert mps_overlap(fresh, fresh) == pytest.approx(1)
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        mps_targets.write_mps_target(path, 6, seed=42)
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("corruption", ["hash", "shape", "dtype", "nan", "norm", "phase", "keys", "escape", "seed"])
def test_corrupted_targets_fail_even_when_payload_hash_is_updated(tmp_path: Path, corruption: str) -> None:
    """Authentication alone cannot make a malformed MPS scientifically valid."""
    path = tmp_path / "target.json"
    record = mps_targets.write_mps_target(path, 4)
    array = path.with_suffix(".npz")
    with np.load(array, allow_pickle=False) as archive:
        tensors = {name: archive[name] for name in archive.files}
    if corruption == "shape":
        tensors["site_1"] = tensors["site_1"][:, :1, :]
    elif corruption == "dtype":
        tensors["site_1"] = tensors["site_1"].real
    elif corruption == "nan":
        tensors["site_1"][0, 0, 0] = np.nan
    elif corruption == "norm":
        tensors["site_0"] *= 2
    elif corruption == "phase":
        tensors["site_0"] *= -1
    elif corruption == "keys":
        del tensors["site_1"]
    elif corruption == "seed":
        record["seed"] = 5003
    elif corruption == "escape":
        record["array"]["file"] = "../outside.npz"
    else:
        tensors["site_0"] *= 1.01
    np.savez(array, **tensors)
    if corruption != "hash":
        record["array"]["sha256"] = file_sha256(array)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=r"checksum|tensor|normalization|escapes"):
        mps_targets.load_mps_target(path)


@pytest.mark.parametrize(("qubits", "bond", "seed"), [(True, 2, 1), (1, 2, 1), (30, 0, 1), (30, True, 1), (30, 2, -1)])
def test_invalid_target_definitions(qubits: int, bond: int, seed: int) -> None:
    """Reject malformed dimensions/seeds before allocating tensors."""
    with pytest.raises(ValueError, match="Require"):
        mps_targets.random_mps_target(qubits, bond, seed)


def test_zero_start_and_invalid_screening_inputs() -> None:
    """Preserve the optimal zero candidate and reject invalid numerical target data."""
    target = MPS(30)
    record, _ = screening.screen_products_mps(target, 37)
    assert record["selected_index"] == 0
    assert record["selected_product_fidelity"] == 1
    with pytest.raises(ValueError, match="sample/batch"):
        screening.screen_products_mps(target, count=True)
    target.tensors[0] *= 2
    with pytest.raises(ValueError, match="normalized"):
        screening.screen_products_mps(target)
