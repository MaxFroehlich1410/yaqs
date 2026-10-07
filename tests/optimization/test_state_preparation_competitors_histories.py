# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Lossless, deterministic, and fail-closed storage of parameter histories."""

from __future__ import annotations

import copy
import io
import zipfile
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from experiments.state_preparation_competitors.histories import (
    history_descriptors,
    load_history_array,
    pack_histories,
    write_history_archive,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_pack_preserves_curves_receipts_order_and_inputs() -> None:
    """Only per-iteration theta fields move; every scientific diagnostic survives."""
    record: dict[str, Any] = {
        "seed": 300,
        "theta": [0.5, -0.5],
        "construction": {
            "initial_theta": [0.0, -0.0],
            "best_theta": [0.5, -0.5],
            "final_theta": [0.4, -0.4],
            "records": [
                {"theta": [0.0, -0.0], "iteration": 0, "fidelity": 0.1, "cumulative_cpu_seconds": 0.3},
                {"theta": [0.5, -0.5], "iteration": 1, "fidelity": 0.9, "cumulative_cpu_seconds": 0.4},
                {"theta": [0.4, -0.4], "iteration": 2, "fidelity": 0.8, "cumulative_cpu_seconds": 0.6},
            ],
        },
        "noisy_training": {"records": [{"theta": [0.1, 0.2], "iteration": 0, "fidelity": 0.7}]},
    }
    original = copy.deepcopy(record)
    compact, arrays = pack_histories(record)
    assert record == original
    assert list(compact) == list(record)
    assert compact["seed"] == 300
    assert compact["theta"] == original["theta"]
    for name in ("initial_theta", "best_theta", "final_theta"):
        assert compact["construction"][name] == original["construction"][name]
    for optimizer in ("construction", "noisy_training"):
        actual = compact[optimizer]
        expected = original[optimizer]
        descriptor = actual["parameter_history"]
        assert descriptor["file"] == "histories.npz"
        assert descriptor["dtype"] == "<f8"
        assert descriptor["shape"] == [len(expected["records"]), 2]
        assert actual["records"] == [
            {key: value for key, value in row.items() if key != "theta"} for row in expected["records"]
        ]
        np.testing.assert_array_equal(arrays[descriptor["key"]], [row["theta"] for row in expected["records"]])
    assert np.signbit(arrays["history_0000"][0, 1])
    compact["theta"][0] = 20
    compact["construction"]["records"][0]["fidelity"] = 20
    arrays["history_0000"][0, 0] = 20
    assert record == original


def test_pack_naming_is_independent_of_dictionary_insertion_order() -> None:
    """Canonical archive keys do not depend on unrelated receipt insertion order."""
    early = {"records": [{"theta": [1, 2]}]}
    late = {"records": [{"theta": [3, 4]}]}
    first, arrays = pack_histories({"z": late, "a": early})
    second, other_arrays = pack_histories({"a": early, "z": late})
    assert first == second
    assert list(first) == ["z", "a"]
    assert list(second) == ["a", "z"]
    for key, array in arrays.items():
        np.testing.assert_array_equal(array, other_arrays[key])
    assert first["a"]["parameter_history"]["key"] == "history_0000"


def test_pack_nested_records_and_zero_parameter_histories(tmp_path: Path) -> None:
    """Nested optimizer receipts and pruned empty circuits retain all iterations."""
    record = {"candidates": [{"optimizer": {"records": [{"theta": [], "child": {"records": [{"theta": [2.0]}]}}]}}]}
    compact, arrays = pack_histories(record)
    assert arrays["history_0000"].shape == (1, 0)
    assert arrays["history_0001"].shape == (1, 1)
    assert len(history_descriptors(compact)) == 2
    path = tmp_path / "histories.npz"
    write_history_archive(path, arrays)
    assert load_history_array(path, "history_0000", [1, 0]).shape == (1, 0)


@pytest.mark.parametrize("record", [{}, {"records": []}, {"records": [{"fidelity": 0.5}]}, {"records": "unrelated"}])
def test_pack_without_theta_histories_is_an_independent_copy(record: dict[str, Any]) -> None:
    """Nonparameter diagnostics do not create an archive or disappear from JSON."""
    compact, arrays = pack_histories(record)
    assert compact == record
    assert compact is not record
    assert not arrays
    assert not history_descriptors(compact)


@pytest.mark.parametrize(
    "records",
    [
        [{"theta": [0.1]}, {"fidelity": 0.5}],
        [{"theta": [0.1]}, 2],
        [{"theta": [0.1]}, {"theta": [0.1, 0.2]}],
        [{"theta": [[0.1]]}],
        [{"theta": None}],
        [{"theta": 0.1}],
        [{"theta": [True]}],
        [{"theta": ["0.1"]}],
        [{"theta": [0.1j]}],
        [{"theta": [np.nan]}],
        [{"theta": [np.inf]}],
    ],
)
def test_pack_rejects_invalid_parameter_records(records: list[Any]) -> None:
    """Malformed parameter arrays fail instead of losing or coercing information."""
    with pytest.raises(ValueError, match=r"Parameter history|parameter history"):
        pack_histories({"records": records})


def test_pack_preserves_compact_references_but_rejects_conflicting_raw_history() -> None:
    """A compact receipt is stable on resave, while ambiguous histories fail."""
    compact, _ = pack_histories({"records": [{"theta": [0.2], "fidelity": 0.9}]})
    repeated, arrays = pack_histories(compact)
    assert repeated == compact
    assert not arrays
    compact["records"][0]["theta"] = [0.2]
    with pytest.raises(ValueError, match="conflict"):
        pack_histories(compact)


def test_pack_rejects_mixed_existing_binary_and_raw_histories() -> None:
    """Packing a partial migration cannot overwrite an already referenced array key."""
    compact, _ = pack_histories({"records": [{"theta": [0.2], "fidelity": 0.9}]})
    with pytest.raises(ValueError, match="Cannot mix"):
        pack_histories({"construction": compact, "noisy": {"records": [{"theta": [0.3]}]}})


@pytest.mark.parametrize("records", [None, [], [{"theta": [0.2]}], [2], [{}, {}]])
def test_descriptor_must_align_with_scalar_history_records(records: object) -> None:
    """A valid array descriptor alone cannot conceal missing or misaligned curves."""
    compact, _ = pack_histories({"records": [{"theta": [0.2], "fidelity": 0.9}]})
    compact["records"] = records
    with pytest.raises(ValueError, match="align"):
        history_descriptors(compact)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("file", "../histories.npz"),
        ("file", "other.npz"),
        ("key", "../history_0000"),
        ("key", "history_0"),
        ("key", 0),
        ("dtype", "float64"),
        ("dtype", "<f4"),
        ("shape", [1]),
        ("shape", [-1, 2]),
        ("shape", [True, 2]),
        ("shape", [1.0, 2]),
        ("shape", "12"),
    ],
)
def test_descriptor_validation_rejects_invalid_fields(field: str, value: object) -> None:
    """Archive references cannot change storage location or numerical interpretation."""
    descriptor: dict[str, object] = {"file": "histories.npz", "key": "history_0000", "dtype": "<f8", "shape": [1, 2]}
    descriptor[field] = value
    with pytest.raises(ValueError, match="Malformed"):
        history_descriptors({"optimizer": {"parameter_history": descriptor}})


@pytest.mark.parametrize("descriptor", [None, [], {}, {"file": "histories.npz"}])
def test_descriptor_validation_rejects_incomplete_shapes(descriptor: object) -> None:
    """A reserved descriptor must be complete, including when repacking a receipt."""
    with pytest.raises(ValueError, match="Malformed"):
        pack_histories({"parameter_history": descriptor})


def test_descriptor_collection_returns_private_copies_and_repeated_references() -> None:
    """Repeated references can share one array without mutable receipt aliases."""
    compact, _ = pack_histories({"records": [{"theta": [0.1]}]})
    descriptors = history_descriptors({"first": compact, "second": compact})
    assert len(descriptors) == 2
    assert descriptors[0] == descriptors[1]
    descriptors[0]["shape"][0] = 4
    assert compact["parameter_history"]["shape"] == [1, 1]
    assert descriptors[1]["shape"] == [1, 1]


def test_archive_is_lossless_compressed_and_byte_deterministic(tmp_path: Path) -> None:
    """Canonical float64 values survive NPZ and produce stable hashes on resave."""
    values = np.arange(2000, dtype=np.float64).reshape(40, 50)
    values[0, 0] = -0.0
    arrays = {"history_0001": values[:, ::-1], "history_0000": np.zeros((100, 30), dtype=np.float64)}
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    write_history_archive(first, arrays)
    write_history_archive(second, {key: np.asfortranarray(arrays[key]) for key in reversed(arrays)})
    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size < sum(value.nbytes for value in arrays.values())
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["history_0000.npy", "history_0001.npy"]
        assert all(member.date_time == (1980, 1, 1, 0, 0, 0) for member in archive.infolist())
        assert all(member.compress_type == zipfile.ZIP_DEFLATED for member in archive.infolist())
    for key, expected in arrays.items():
        actual = load_history_array(first, key, expected.shape)
        assert actual.dtype.str == "<f8"
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    with np.load(first, allow_pickle=False) as archive:
        assert set(archive.files) == set(arrays)


def test_empty_archive_is_valid(tmp_path: Path) -> None:
    """An empty archive remains readable without invented parameter histories."""
    path = tmp_path / "histories.npz"
    write_history_archive(path, {})
    with np.load(path, allow_pickle=False) as archive:
        assert archive.files == []


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.int64),
        np.zeros((1, 2), dtype=object),
        np.zeros((1, 2), dtype=complex),
        np.zeros((1, 2), dtype=">f8"),
        np.zeros(2, dtype=np.float64),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
    ],
)
def test_archive_write_rejects_invalid_arrays_without_overwriting(tmp_path: Path, array: np.ndarray) -> None:
    """Validation precedes writes so an existing receipt cannot be damaged."""
    path = tmp_path / "histories.npz"
    path.write_bytes(b"previous archive")
    with pytest.raises(ValueError, match="finite two-dimensional"):
        write_history_archive(path, {"history_0000": array})
    assert path.read_bytes() == b"previous archive"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("key", ["../history_0000", "history_0", "other", "history_0000.npy", "history_0000\n"])
def test_archive_rejects_unsafe_or_noncanonical_keys(tmp_path: Path, key: str) -> None:
    """Neither writes nor reads interpret archive members as arbitrary paths."""
    path = tmp_path / "histories.npz"
    with pytest.raises(ValueError, match="Invalid parameter history key"):
        write_history_archive(path, {key: np.zeros((1, 1))})
    with pytest.raises(ValueError, match="Invalid parameter history key"):
        load_history_array(path, key, [1, 1])


def test_archive_failure_cleans_temporary_file_and_preserves_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disk or serialization failure never replaces the last complete archive."""
    path = tmp_path / "histories.npz"
    path.write_bytes(b"previous archive")

    def fail_write(*_args: object, **_kwargs: object) -> None:
        msg = "simulated disk failure"
        raise OSError(msg)

    monkeypatch.setattr(zipfile.ZipFile, "writestr", fail_write)
    with pytest.raises(OSError, match="simulated disk failure"):
        write_history_archive(path, {"history_0000": np.zeros((1, 1))})
    assert path.read_bytes() == b"previous archive"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("shape", [[2, 2], [1], [-1, 2], [True, 2], [1.0, 2]])
def test_archive_load_rejects_mismatched_or_invalid_shapes(tmp_path: Path, shape: list) -> None:
    """Shape metadata is authenticated numerically instead of trusting JSON alone."""
    path = tmp_path / "histories.npz"
    write_history_archive(path, {"history_0000": np.zeros((1, 2))})
    with pytest.raises(ValueError, match="shape"):
        load_history_array(path, "history_0000", shape)


@pytest.mark.parametrize(
    "array", [np.array([[object()]], dtype=object), np.array([[np.inf]]), np.zeros((1, 1), dtype=np.float32)]
)
def test_archive_load_rejects_pickle_nonfinite_and_wrong_dtype(tmp_path: Path, array: np.ndarray) -> None:
    """Externally altered archives cannot introduce object loading or silent casts."""
    path = tmp_path / "histories.npz"
    np.savez(path, history_0000=array)
    with pytest.raises(ValueError, match=r"Object arrays|Parameter history arrays"):
        load_history_array(path, "history_0000", [1, 1])


def test_archive_load_rejects_missing_key(tmp_path: Path) -> None:
    """A descriptor cannot silently bind to the wrong optimizer's parameter matrix."""
    path = tmp_path / "histories.npz"
    write_history_archive(path, {"history_0000": np.zeros((1, 2))})
    with pytest.raises(ValueError, match="exactly one"):
        load_history_array(path, "history_0001", [1, 2])


def test_archive_load_rejects_duplicate_members(tmp_path: Path) -> None:
    """Ambiguous ZIP members fail even if both payloads have an acceptable shape."""
    path = tmp_path / "histories.npz"
    payload = io.BytesIO()
    np.lib.format.write_array(payload, np.zeros((1, 2)), allow_pickle=False)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("history_0000.npy", payload.getvalue())
        with pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("history_0000.npy", payload.getvalue())
    with pytest.raises(ValueError, match="exactly one"):
        load_history_array(path, "history_0000", [1, 2])
