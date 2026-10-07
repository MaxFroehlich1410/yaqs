# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Lossless binary storage for optimizer parameter histories and small JSON receipts."""

from __future__ import annotations

import copy
import io
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import BinaryIO

    from numpy.typing import NDArray

_HISTORY_KEY = re.compile(r"history_[0-9]{4,}\Z")
_HISTORY_DTYPE = np.dtype("<f8")


def _history_matrix(records: list[Any]) -> NDArray[np.float64] | None:
    """Validate a parameter-bearing record list and return its private matrix."""
    if not any(isinstance(record, dict) and "theta" in record for record in records):
        return None
    if not all(isinstance(record, dict) and "theta" in record for record in records):
        msg = "Parameter history records must all contain theta vectors."
        raise ValueError(msg)
    vectors = []
    for record in records:
        vector = np.asarray(record["theta"])
        if vector.ndim != 1 or vector.dtype.kind not in "fiu" or not np.all(np.isfinite(vector)):
            msg = "Each parameter history theta must be a finite, real numeric vector."
            raise ValueError(msg)
        vectors.append(vector)
    if len({vector.size for vector in vectors}) != 1:
        msg = "Parameter history theta vectors must have a constant length."
        raise ValueError(msg)
    matrix = np.asarray(vectors, dtype=_HISTORY_DTYPE)
    if not np.all(np.isfinite(matrix)):
        msg = "Parameter history theta vectors must be representable as finite float64 values."
        raise ValueError(msg)
    return matrix


def pack_histories(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, NDArray[np.float64]]]:
    """Move per-iteration theta vectors to binary arrays without changing diagnostics.

    Every parameter-bearing ``records`` list retains its entries and their order,
    fidelity curves, timing, and scalar diagnostics. Its parent gains a
    ``parameter_history`` descriptor identifying the row-aligned theta matrix.
    Initial, best, and final parameters outside those lists remain in JSON.
    Neither the input nor any of its nested containers is modified.

    Args:
        record: A JSON-compatible construction receipt, possibly containing
            multiple optimizers or candidates.

    Returns:
        A private compact receipt and its deterministically named float64 arrays.

    Raises:
        ValueError: If theta records are partial, ragged, nonnumeric, nonfinite,
            non-vector values, or conflict with an existing history descriptor.
    """
    arrays: dict[str, NDArray[np.float64]] = {}

    def visit(value: object) -> object:
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return copy.deepcopy(value)
        mapping = cast("dict[str, Any]", value)
        entries = mapping.get("records")
        matrix = _history_matrix(entries) if isinstance(entries, list) and entries else None
        if matrix is not None and "parameter_history" in mapping:
            msg = "Parameter history records conflict with an existing parameter_history descriptor."
            raise ValueError(msg)
        visited = {}
        descriptor = None
        for key in sorted(mapping):
            if key == "records" and matrix is not None:
                assert isinstance(entries, list)
                array_key = f"history_{len(arrays):04d}"
                arrays[array_key] = matrix
                descriptor = {
                    "file": "histories.npz",
                    "key": array_key,
                    "dtype": _HISTORY_DTYPE.str,
                    "shape": list(matrix.shape),
                }
                visited[key] = [visit({name: item for name, item in row.items() if name != "theta"}) for row in entries]
            else:
                visited[key] = visit(mapping[key])
        result = {key: visited[key] for key in mapping}
        if descriptor is not None:
            result["parameter_history"] = descriptor
        return result

    compact = cast("dict[str, Any]", visit(record))
    assert isinstance(compact, dict)
    descriptors = history_descriptors(compact)
    if arrays and len(descriptors) != len(arrays):
        msg = "Cannot mix existing binary parameter histories and new raw parameter histories."
        raise ValueError(msg)
    return compact, arrays


def history_descriptors(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect validated binary history references in deterministic traversal order.

    Args:
        record: A compact JSON-compatible construction receipt.

    Returns:
        Private copies of each descriptor, including repeated references.

    Raises:
        ValueError: If a reserved ``parameter_history`` field is malformed.
    """
    result = []

    def visit(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            mapping = cast("dict[str, Any]", value)
            for key in sorted(mapping):
                if key != "parameter_history":
                    visit(mapping[key])
                    continue
                candidate = mapping[key]
                descriptor = cast("dict[str, Any]", candidate)
                if (
                    not isinstance(candidate, dict)
                    or set(descriptor) != {"file", "key", "dtype", "shape"}
                    or descriptor["file"] != "histories.npz"
                    or not isinstance(descriptor["key"], str)
                    or not _HISTORY_KEY.fullmatch(descriptor["key"])
                    or descriptor["dtype"] != _HISTORY_DTYPE.str
                    or not isinstance(descriptor["shape"], list)
                    or len(descriptor["shape"]) != 2
                    or any(
                        isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in descriptor["shape"]
                    )
                ):
                    msg = "Malformed parameter_history descriptor."
                    raise ValueError(msg)
                records = mapping.get("records")
                if (
                    not isinstance(records, list)
                    or len(records) != descriptor["shape"][0]
                    or not all(isinstance(row, dict) and "theta" not in row for row in records)
                ):
                    msg = "Parameter history descriptor must align with scalar records without theta fields."
                    raise ValueError(msg)
                result.append(copy.deepcopy(descriptor))

    visit(record)
    return result


def _validate_array(key: str, array: object) -> NDArray[np.float64]:
    """Require a named finite two-dimensional little-endian float64 history."""
    if not _HISTORY_KEY.fullmatch(key):
        msg = f"Invalid parameter history key: {key!r}."
        raise ValueError(msg)
    value = np.asarray(array)
    if value.dtype != _HISTORY_DTYPE or value.ndim != 2 or not np.all(np.isfinite(value)):
        msg = "Parameter history arrays must be finite two-dimensional little-endian float64 matrices."
        raise ValueError(msg)
    return value


def _write_archive_payload(stream: BinaryIO, arrays: Mapping[str, NDArray[np.float64]]) -> None:
    """Write canonical NPY members into a fixed-metadata compressed ZIP stream."""
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for key, value in arrays.items():
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.ascontiguousarray(value), allow_pickle=False)
            member = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.create_system = 3
            member.external_attr = 0o600 << 16
            archive.writestr(member, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    stream.flush()
    os.fsync(stream.fileno())


def write_history_archive(path: Path, arrays: Mapping[str, NDArray[np.float64]]) -> None:
    """Atomically write a deterministic compressed NPZ without object pickles.

    ZIP member ordering, timestamps, attributes, compression, and NPY array order
    are fixed so that the same values produce the same archive hash.

    Args:
        path: Destination archive in an existing directory.
        arrays: Named finite float64 matrices from ``pack_histories``.

    Raises:
        ValueError: If any name, dtype, shape, or value is invalid.
        OSError: If the archive cannot be written or atomically replaced.
    """
    validated = {key: _validate_array(key, arrays[key]) for key in sorted(arrays)}
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            _write_archive_payload(stream, validated)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_history_array(path: Path, key: str, shape: Sequence[int]) -> NDArray[np.float64]:
    """Load and validate one parameter matrix without enabling object pickles.

    Args:
        path: Archive referenced by a construction receipt.
        key: Array key from its ``parameter_history`` descriptor.
        shape: Expected row and parameter counts from that descriptor.

    Returns:
        A private finite little-endian float64 matrix in iteration order.

    Raises:
        ValueError: If the requested key, array dtype, shape, or values are invalid.
        OSError: If the archive cannot be read.
    """
    if not _HISTORY_KEY.fullmatch(key):
        msg = f"Invalid parameter history key: {key!r}."
        raise ValueError(msg)
    if len(shape) != 2 or any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape):
        msg = "Expected parameter history shape must contain two nonnegative integers."
        raise ValueError(msg)
    with np.load(path, allow_pickle=False) as archive:
        if archive.files.count(key) != 1:
            msg = f"Parameter history archive must contain exactly one array named {key!r}."
            raise ValueError(msg)
        value = _validate_array(key, archive[key])
        if value.shape != tuple(shape):
            msg = f"Parameter history array shape {value.shape} does not match expected {tuple(shape)}."
            raise ValueError(msg)
        return value.copy()
