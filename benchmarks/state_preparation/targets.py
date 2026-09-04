# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Validated access to the fixed state-preparation target fixture."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, NoReturn, cast

import numpy as np

from mqt.yaqs.core.data_structures.mps import MPS

from .constants import (
    SUPPORTED_QUBIT_COUNTS,
    TARGET_FIXTURE_FORMAT,
    TARGET_GENERATION_SEEDS,
    TARGET_IDS,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from numpy.typing import NDArray

DEFAULT_TARGET_PATH = Path(__file__).parents[1] / "state_preparation_target_states.json"
NORM_ATOL = 1e-10
_ROOT_KEYS = frozenset({
    "format",
    "generated_by",
    "complex_encoding",
    "basis_order",
    "global_phase",
    "numpy_version",
    "scipy_version",
    "qubit_counts",
    "target_ids",
    "targets",
})
_RECORD_KEYS = frozenset({"target_id", "num_qubits", "seed", "parameters", "norm", "state_vector"})
_ROOT_METADATA = {
    "basis_order": "little_endian: amplitude index k has qubit i equal to bit i of k",
    "complex_encoding": "[real, imaginary]",
    "generated_by": "benchmarks/generate_state_preparation_targets.py",
    "global_phase": "fixed so the largest-magnitude amplitude is positive real",
}


def _freeze_json(value: object) -> object:
    """Return a recursively immutable copy of JSON-compatible data."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _reject_json_constant(value: str) -> NoReturn:
    """Reject non-standard JSON constants.

    Raises:
        ValueError: Always, because the constant is non-finite.
    """
    msg = f"Non-finite JSON constant {value!r} is not supported."
    raise ValueError(msg)


def _without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a mapping while rejecting duplicate JSON members.

    Returns:
        Mapping containing the parsed members.

    Raises:
        ValueError: If a member occurs more than once.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = f"Duplicate JSON key {key!r}."
            raise ValueError(msg)
        result[key] = value
    return result


def _require_mapping(value: object, name: str) -> dict[str, object]:
    """Return a string-keyed JSON mapping.

    Raises:
        ValueError: If the value is not a string-keyed mapping.
    """
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        msg = f"{name} must be a JSON object with string keys."
        raise ValueError(msg)
    return cast("dict[str, object]", value)


def _require_exact_keys(value: dict[str, object], keys: frozenset[str], name: str) -> None:
    """Require exactly the fields in a versioned record.

    Raises:
        ValueError: If fields are missing or unsupported fields are present.
    """
    if value.keys() != keys:
        missing = sorted(keys - value.keys())
        extra = sorted(value.keys() - keys)
        msg = f"{name} has invalid fields: missing={missing!r}, extra={extra!r}."
        raise ValueError(msg)


def _require_int(value: object, name: str) -> int:
    """Return an exact integer without accepting booleans.

    Raises:
        ValueError: If the value is not an exact integer.
    """
    if type(value) is not int:
        msg = f"{name} must be an integer."
        raise ValueError(msg)
    return value


def _require_float(value: object, name: str) -> float:
    """Return a finite JSON number.

    Raises:
        ValueError: If the value is not a finite JSON number.
    """
    if type(value) not in {int, float}:
        msg = f"{name} must be a JSON number."
        raise ValueError(msg)
    result = float(cast("int | float", value))
    if not math.isfinite(result):
        msg = f"{name} must be finite."
        raise ValueError(msg)
    return result


def _decode_vector(value: object, num_qubits: int, target_id: str) -> NDArray[np.complex128]:
    """Return one decoded and validated complex-pair state vector.

    Raises:
        ValueError: If the encoding, dimension, norm, or phase is invalid.
    """
    if not isinstance(value, list) or len(value) != 2**num_qubits:
        msg = f"Target {(num_qubits, target_id)!r} must have {2**num_qubits} amplitudes."
        raise ValueError(msg)
    vector = np.empty(len(value), dtype=np.complex128)
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            msg = f"Amplitude {index} must be a [real, imaginary] pair."
            raise ValueError(msg)
        vector[index] = complex(
            _require_float(item[0], f"amplitude {index} real component"),
            _require_float(item[1], f"amplitude {index} imaginary component"),
        )
    norm = float(np.linalg.norm(vector))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=NORM_ATOL):
        msg = f"Target {(num_qubits, target_id)!r} has norm {norm}, expected one."
        raise ValueError(msg)
    pivot = vector[int(np.argmax(np.abs(vector)))]
    if pivot.real <= 0.0 or abs(pivot.imag) > NORM_ATOL:
        msg = f"Target {(num_qubits, target_id)!r} violates the fixed global-phase convention."
        raise ValueError(msg)
    return vector


@dataclass(frozen=True, slots=True)
class TargetRecord:
    """One immutable target state and its generation metadata."""

    num_qubits: int
    target_id: str
    seed: int | None
    norm: float
    _parameters: Mapping[str, object] = field(repr=False)
    _vector_bytes: bytes = field(repr=False)

    @property
    def key(self) -> tuple[int, str]:
        """Return the unique fixture key."""
        return self.num_qubits, self.target_id

    @property
    def parameters(self) -> Mapping[str, object]:
        """Return immutable target-generation metadata."""
        return self._parameters

    @property
    def state_vector(self) -> NDArray[np.complex128]:
        """Return a read-only view of the state vector."""
        return np.frombuffer(self._vector_bytes, dtype=np.complex128)

    def state_vector_copy(self) -> NDArray[np.complex128]:
        """Return a mutable copy of the state vector."""
        return self.state_vector.copy()

    def to_mps(self) -> MPS:
        """Return an exact MPS representation of the target."""
        return MPS.from_statevector(self.state_vector)


@dataclass(frozen=True, slots=True)
class TargetCollection:
    """A validated target fixture with deterministic lookup order."""

    checksum: str
    records: tuple[TargetRecord, ...]
    metadata: Mapping[str, object]
    _index: Mapping[tuple[int, str], TargetRecord] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build an immutable composite-key index.

        Raises:
            ValueError: If records are duplicated or incomplete.
        """
        index = {record.key: record for record in self.records}
        if len(index) != len(self.records):
            msg = "Target fixture contains duplicate (num_qubits, target_id) records."
            raise ValueError(msg)
        expected = {(num_qubits, target_id) for num_qubits in SUPPORTED_QUBIT_COUNTS for target_id in TARGET_IDS}
        if index.keys() != expected:
            missing = sorted(expected - index.keys())
            extra = sorted(index.keys() - expected)
            msg = f"Target fixture is incomplete: missing={missing!r}, extra={extra!r}."
            raise ValueError(msg)
        object.__setattr__(self, "_index", MappingProxyType(index))

    def load(self, num_qubits: int, target_id: str) -> TargetRecord:
        """Return one target selected by qubit count and identifier.

        Raises:
            ValueError: If the target key is unsupported.
        """
        try:
            return self._index[num_qubits, target_id]
        except KeyError as error:
            msg = f"Unsupported target {(num_qubits, target_id)!r}."
            raise ValueError(msg) from error

    def select(
        self,
        *,
        num_qubits: tuple[int, ...] | None = None,
        target_ids: tuple[str, ...] | None = None,
    ) -> tuple[TargetRecord, ...]:
        """Return targets matching optional ordered filters."""
        qubits = SUPPORTED_QUBIT_COUNTS if num_qubits is None else num_qubits
        identifiers = TARGET_IDS if target_ids is None else target_ids
        return tuple(self.load(count, identifier) for count in qubits for identifier in identifiers)


def _parse_record(value: object) -> TargetRecord:
    """Return one parsed target record.

    Raises:
        ValueError: If the record violates the benchmark schema.
    """
    record = _require_mapping(value, "target record")
    _require_exact_keys(record, _RECORD_KEYS, "Target record")
    num_qubits = _require_int(record["num_qubits"], "num_qubits")
    target_id = record["target_id"]
    if num_qubits not in SUPPORTED_QUBIT_COUNTS or not isinstance(target_id, str) or target_id not in TARGET_IDS:
        msg = f"Unsupported target key {(num_qubits, target_id)!r}."
        raise ValueError(msg)
    seed = record["seed"]
    if seed is not None:
        seed = _require_int(seed, "seed")
    if seed != TARGET_GENERATION_SEEDS[target_id]:
        msg = f"Target {(num_qubits, target_id)!r} has an unexpected generation seed."
        raise ValueError(msg)
    parameters = _require_mapping(record["parameters"], "parameters")
    vector = _decode_vector(record["state_vector"], num_qubits, target_id)
    stored_norm = _require_float(record["norm"], "norm")
    actual_norm = float(np.linalg.norm(vector))
    if not math.isclose(stored_norm, actual_norm, rel_tol=0.0, abs_tol=NORM_ATOL):
        msg = f"Target {(num_qubits, target_id)!r} stores an inconsistent norm."
        raise ValueError(msg)
    frozen_parameters = cast("Mapping[str, object]", _freeze_json(json.loads(json.dumps(parameters))))
    return TargetRecord(
        num_qubits=num_qubits,
        target_id=target_id,
        seed=seed,
        norm=stored_norm,
        _parameters=frozen_parameters,
        _vector_bytes=vector.tobytes(),
    )


def load_target_collection(path: Path | None = None) -> TargetCollection:
    """Load and validate the complete target fixture.

    Args:
        path: Optional fixture path. The checked-in fixture is the default.

    Returns:
        Validated target collection.

    Raises:
        ValueError: If the fixture is unreadable or violates its schema.
        TypeError: If the target array has an invalid JSON type.
    """
    fixture_path = DEFAULT_TARGET_PATH if path is None else Path(path)
    try:
        encoded = fixture_path.read_bytes()
        document = json.loads(
            encoded.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_without_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        msg = f"Could not load target fixture {fixture_path}: {error}"
        raise ValueError(msg) from error
    root = _require_mapping(document, "target fixture")
    _require_exact_keys(root, _ROOT_KEYS, "Target fixture")
    if root["format"] != TARGET_FIXTURE_FORMAT:
        msg = f"Unsupported target fixture format {root['format']!r}."
        raise ValueError(msg)
    for key, expected in _ROOT_METADATA.items():
        if root[key] != expected:
            msg = f"Target fixture field {key!r} does not match the benchmark convention."
            raise ValueError(msg)
    for key in ("numpy_version", "scipy_version"):
        if not isinstance(root[key], str) or not cast("str", root[key]).strip():
            msg = f"Target fixture field {key!r} must be a nonempty version string."
            raise ValueError(msg)
    if root["qubit_counts"] != list(SUPPORTED_QUBIT_COUNTS) or root["target_ids"] != list(TARGET_IDS):
        msg = "Target fixture declarations do not match the frozen benchmark identifiers."
        raise ValueError(msg)
    raw_records = root["targets"]
    if not isinstance(raw_records, list):
        msg = "Target fixture field 'targets' must be an array."
        raise TypeError(msg)
    records = tuple(_parse_record(record) for record in raw_records)
    metadata = cast(
        "Mapping[str, object]", _freeze_json({key: value for key, value in root.items() if key != "targets"})
    )
    return TargetCollection(
        checksum=f"sha256:{hashlib.sha256(encoded).hexdigest()}",
        records=records,
        metadata=metadata,
    )


def load_target(num_qubits: int, target_id: str, path: Path | None = None) -> TargetRecord:
    """Return one target from a validated fixture."""
    return load_target_collection(path).load(num_qubits, target_id)


def iter_targets(
    *,
    num_qubits: tuple[int, ...] | None = None,
    target_ids: tuple[str, ...] | None = None,
    path: Path | None = None,
) -> Iterator[TargetRecord]:
    """Return an iterator over selected targets in benchmark order."""
    return iter(load_target_collection(path).select(num_qubits=num_qubits, target_ids=target_ids))


__all__ = [
    "DEFAULT_TARGET_PATH",
    "TargetCollection",
    "TargetRecord",
    "iter_targets",
    "load_target",
    "load_target_collection",
]
