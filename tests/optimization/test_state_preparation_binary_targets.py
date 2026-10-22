# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Regression tests for binary targets without large CI allocations."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np
import pytest

from benchmarks import generate_state_preparation_targets as generation
from benchmarks.state_preparation.binary_targets import (
    LEGACY_SHA256,
    binary_main,
    check_binary_targets,
    file_sha256,
    generate_binary_targets,
    tfim_diagnostics,
)
from benchmarks.state_preparation.targets import (
    DEFAULT_TARGET_PATH,
    TargetCollection,
    load_target,
    load_target_collection,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def binary_manifest(tmp_path: Path) -> Path:
    """Generate all nine small binary targets for format and numerical regression tests.

    Returns:
        Path to the complete binary manifest.
    """
    manifest = tmp_path / "manifest.json"
    generate_binary_targets(manifest, (6,), generation.TARGET_IDS)
    return manifest


def test_binary_generation_preserves_the_original_definitions(binary_manifest: Path) -> None:
    """All binary families reproduce the legacy targets, phases, and basis order."""
    collection = load_target_collection(binary_manifest)
    assert len(collection.records) == 9
    legacy = load_target_collection()
    for target in collection.records:
        np.testing.assert_allclose(target.state_vector, legacy.load(*target.key).state_vector, atol=1e-10)
        assert isinstance(target.state_vector, np.memmap)
        assert not target.state_vector.flags.writeable
        assert target.state_vector.dtype.str == "<c16"
        copied = target.state_vector_copy()
        copied[0] = 0
        assert not np.array_equal(copied, target.state_vector)
    check_binary_targets(binary_manifest, (6,), generation.TARGET_IDS)
    assert file_sha256(DEFAULT_TARGET_PATH) == LEGACY_SHA256


def test_binary_metadata_loading_is_lazy(binary_manifest: Path) -> None:
    """Selection of one target never opens unrelated numerical arrays."""
    with patch("numpy.load", side_effect=AssertionError("Unexpected eager array load")):
        collection = load_target_collection(binary_manifest)
        assert len(collection.select()) == 9
        assert collection.load(6, "haar_random_1").seed == 4001


@pytest.mark.parametrize("bond", [2, 3])
def test_native_random_mps_avoids_dense_factorization(binary_manifest: Path, bond: int) -> None:
    """Native tensors reproduce the stored state at the original small bond dimension."""
    target = load_target(6, f"random_mps_bond{bond}", binary_manifest)
    with patch("benchmarks.state_preparation.targets.validated_vector", side_effect=AssertionError("Dense load")):
        mps = target.to_mps()
    assert max(max(tensor.shape[1:]) for tensor in mps.tensors) == bond
    np.testing.assert_allclose(mps.to_vec(), target.state_vector, atol=1e-13)


def test_binary_arrays_are_authenticated_on_access(binary_manifest: Path) -> None:
    """Corruption fails on access even after the metadata collection was loaded."""
    target = load_target(6, "haar_random_1", binary_manifest)
    payload = binary_manifest.parent / "q6" / "haar_random_1.npy"
    with payload.open("r+b") as stream:
        stream.seek(-1, 2)
        last = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([last[0] ^ 1]))
    with pytest.raises(ValueError, match="checksum mismatch"):
        _ = target.state_vector


@pytest.mark.parametrize("corruption", ["shape", "dtype", "norm", "phase", "finite"])
def test_binary_loader_rejects_invalid_arrays_even_with_updated_hash(binary_manifest: Path, corruption: str) -> None:
    """A matching hash alone cannot authorize malformed scientific payloads."""
    document = json.loads(binary_manifest.read_text(encoding="utf-8"))
    record = document["targets"][0]
    path = binary_manifest.parent / record["array"]["file"]
    state = np.load(path)
    if corruption == "shape":
        state = state[:-1]
    elif corruption == "dtype":
        state = state.real.copy()
    elif corruption == "norm":
        state *= 2
    elif corruption == "phase":
        state *= -1
    else:
        state[0] = np.nan
    np.save(path, state, allow_pickle=False)
    record["array"]["sha256"] = file_sha256(path)
    record["array"]["size_bytes"] = path.stat().st_size
    binary_manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=r"dtype or shape|norm|phase|Non-finite"):
        _ = load_target(6, generation.TARGET_IDS[0], binary_manifest).state_vector


@pytest.mark.parametrize("corruption", ["escape", "missing", "duplicate", "seed", "basis"])
def test_binary_loader_rejects_invalid_manifest(binary_manifest: Path, corruption: str) -> None:
    """Incomplete, duplicate, redefined, or externally referenced fixtures are rejected."""
    document = json.loads(binary_manifest.read_text(encoding="utf-8"))
    if corruption == "escape":
        document["targets"][0]["array"]["file"] = "../outside.npy"
    elif corruption == "missing":
        document["targets"].pop()
    elif corruption == "duplicate":
        document["targets"].append(document["targets"][0])
    elif corruption == "seed":
        document["targets"][4]["seed"] = 42
    else:
        document["basis_order"] = "big_endian"
    binary_manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=r"escapes|incomplete|duplicate|seed|convention"):
        load_target_collection(binary_manifest)


@pytest.mark.parametrize("num_qubits", [6, 12])
@pytest.mark.parametrize("spec", generation.TFIM_SPECS)
def test_even_sector_tfim_matches_full_ground_state(num_qubits: int, spec: generation.TfimSpec) -> None:
    """Symmetry reduction preserves the full Hamiltonian's ground state and energy."""
    legacy = load_target(num_qubits, spec.target_id)
    expected = legacy.state_vector
    expected_energy = cast("float", legacy.parameters["ground_energy"])
    metadata, state = next(
        generation.iter_target_arrays((num_qubits,), target_ids=(spec.target_id,), scalable_tfim=True)
    )
    energy = cast("dict", metadata["parameters"])["ground_energy"]
    np.testing.assert_allclose(state, expected, atol=1e-12)
    np.testing.assert_allclose(energy, expected_energy, atol=1e-12)
    diagnostics = tfim_diagnostics(state, num_qubits, spec.h_over_j, energy)
    assert diagnostics["eigenpair_residual_norm"] < 1e-10
    assert diagnostics["global_x_even_max_error"] == 0
    with pytest.raises(ValueError, match="TFIM target failed"):
        tfim_diagnostics(state, num_qubits, spec.h_over_j, energy + 0.01)


def test_binary_cli_is_explicit_and_cannot_replace_legacy_fixture(tmp_path: Path) -> None:
    """New generation rejects legacy widths and cannot overwrite the original JSON."""
    with pytest.raises(ValueError, match="qubit counts"):
        binary_main(tmp_path / "manifest.json", (6,), None, check=False)
    with pytest.raises(ValueError, match="frozen"):
        binary_main(DEFAULT_TARGET_PATH, (20,), None, check=False)
    with pytest.raises(SystemExit):
        generation.main(["--num-qubits", "20"])


def test_binary_cli_dispatches_selected_targets(tmp_path: Path) -> None:
    """Width/target selections reach the binary writer without touching the legacy generator."""
    destination = tmp_path / "manifest.json"
    with patch("benchmarks.state_preparation.binary_targets.generate_binary_targets") as generate:
        assert (
            generation.main([
                "--binary",
                "--output",
                str(destination),
                "--num-qubits",
                "20",
                "--target-id",
                "haar_random_1",
            ])
            == 0
        )
    generate.assert_called_once_with(destination, (20,), ("haar_random_1",))


def test_binary_generation_retains_other_completed_targets(binary_manifest: Path) -> None:
    """Selective regeneration preserves every unselected target and its checksum."""
    before = json.loads(binary_manifest.read_text(encoding="utf-8"))
    generate_binary_targets(binary_manifest, (6,), ("haar_random_2",))
    after = json.loads(binary_manifest.read_text(encoding="utf-8"))
    assert before == after


@pytest.mark.parametrize(
    ("identifier", "parameter", "value"),
    [
        ("gaussian_mu0p5_sigma0p1", "mean", 0.6),
        ("tfim_ferro", "h_over_j", 0.7),
        ("random_mps_bond2", "max_bond_dimension", 4),
    ],
)
def test_binary_metadata_cannot_redefine_a_family(
    binary_manifest: Path, identifier: str, parameter: str, value: float
) -> None:
    """Metadata validation preserves the same physical targets across system sizes."""
    document = json.loads(binary_manifest.read_text(encoding="utf-8"))
    record = next(record for record in document["targets"] if record["target_id"] == identifier)
    record["parameters"][parameter] = value
    binary_manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="scientific definition"):
        load_target_collection(binary_manifest)


def test_native_mps_payload_is_authenticated(binary_manifest: Path) -> None:
    """Direct tensor loading does not bypass fixture integrity checks."""
    record = load_target(6, "random_mps_bond2", binary_manifest)
    path = binary_manifest.parent / "q6" / "random_mps_bond2.mps.npz"
    path.write_bytes(b"corrupted archive")
    with pytest.raises(ValueError, match="Native MPS checksum mismatch"):
        record.to_mps()


def test_even_sector_operator_matches_full_hamiltonian() -> None:
    """The folded highest-bit flip and boundary interaction match explicit full-space action."""
    num_qubits = 5
    dim = 2**num_qubits
    field = 0.5
    rng = np.random.default_rng(7)
    reduced = rng.standard_normal(dim // 2)
    state = np.concatenate((reduced, reduced[::-1])) / np.sqrt(2.0)
    indices = np.arange(dim)
    diagonal = np.zeros(dim)
    for site in range(num_qubits - 1):
        unequal = ((indices >> site) ^ (indices >> (site + 1))) & 1
        diagonal += 2 * unequal - 1
    full_action = diagonal * state
    for site in range(num_qubits):
        full_action -= field * state[indices ^ (1 << site)]
    reduced_action = generation.EvenTfimLinearOperator(num_qubits, field) @ reduced
    np.testing.assert_allclose(reduced_action, np.sqrt(2.0) * full_action[: dim // 2], atol=1e-13)


def test_large_target_routing_preserves_legacy_defaults(binary_manifest: Path) -> None:
    """Explicit large-width lookups and mixed collections keep the default eighteen targets."""
    legacy = load_target_collection()
    # Only identities are needed for routing; reuse tiny vectors to avoid large test allocations.
    records = tuple(replace(record, num_qubits=count) for count in (20, 24) for record in legacy.records[:9])
    binary = TargetCollection("sha256:" + "0" * 64, records, {**legacy.metadata, "qubit_counts": (20, 24)})
    with (
        patch("benchmarks.state_preparation.targets.DEFAULT_BINARY_TARGET_PATH", binary_manifest),
        patch("benchmarks.state_preparation.targets._parse_binary_collection", return_value=binary),
    ):
        assert load_target(20, "haar_random_1").num_qubits == 20
        assert len(load_target_collection(num_qubits=(20, 24)).records) == 18
        combined = load_target_collection(num_qubits=(6, 12, 20, 24))
        assert len(combined.records) == 36
        assert len(combined.select(num_qubits=(20,))) == 9
        assert load_target_collection().checksum == legacy.checksum
        assert len(load_target_collection().records) == 18
