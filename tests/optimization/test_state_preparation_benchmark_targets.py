# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for validated state-preparation target loading."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from benchmarks.state_preparation import TARGET_IDS, load_target, load_target_collection

TARGET_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "state_preparation_target_states.json"


def test_collection_loads_all_targets_in_benchmark_order() -> None:
    """The loader validates all 18 composite keys and preserves declared order."""
    collection = load_target_collection()

    assert len(collection.records) == 18
    assert len(collection.checksum) == 71
    assert collection.checksum.startswith("sha256:")
    assert [record.target_id for record in collection.records[:9]] == list(TARGET_IDS)
    assert [record.num_qubits for record in collection.records] == [6] * 9 + [12] * 9


def test_target_vectors_are_read_only_and_convert_exactly_to_mps() -> None:
    """A selected target exposes protected dense data and an equivalent MPS."""
    target = load_target(6, "tfim_critical")

    assert target.state_vector.shape == (64,)
    with pytest.raises(ValueError, match="read-only"):
        target.state_vector[0] = 0.0
    np.testing.assert_allclose(target.to_mps().to_vec(), target.state_vector, atol=1e-12)
    copy = target.state_vector_copy()
    copy[0] = 0.0
    assert not np.array_equal(copy, target.state_vector)


def test_target_metadata_is_recursively_immutable() -> None:
    """Nested fixture parameters cannot mutate a loaded record."""
    target = load_target(6, "tfim_critical")
    fields = cast("tuple[float, ...]", target.parameters["transverse_fields"])

    assert isinstance(target.parameters, Mapping)
    assert isinstance(fields, tuple)
    mutable_view = cast("dict[str, object]", target.parameters)
    with pytest.raises(TypeError):
        mutable_view["transverse_field"] = 0.0


def test_selection_rejects_unknown_target_key() -> None:
    """Unsupported target identifiers fail before experiment construction."""
    with pytest.raises(ValueError, match="Unsupported target"):
        load_target_collection().load(6, "missing")


def test_loader_rejects_corrupted_vector_dimension(tmp_path: Path) -> None:
    """A malformed fixture cannot silently enter the benchmark."""
    document = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    document["targets"][0]["state_vector"].pop()
    corrupted = tmp_path / "targets.json"
    corrupted.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="must have 64 amplitudes"):
        load_target_collection(corrupted)


def test_loader_rejects_changed_basis_convention(tmp_path: Path) -> None:
    """A fixture cannot redefine the meaning of its amplitude indices."""
    document = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    document["basis_order"] = "big_endian"
    corrupted = tmp_path / "targets.json"
    corrupted.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="basis_order"):
        load_target_collection(corrupted)
