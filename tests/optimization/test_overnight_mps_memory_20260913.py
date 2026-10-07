# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Retained forward-tape storage measurements on public q1-q3 and shared-buffer fixtures."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from experiments.overnight_state_preparation_20260913_0108 import mps_memory
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KrotovNoiseMap,
    KrotovTJMOptions,
    KrotovTrajectory,
    KrotovTruncation,
    ParameterizedCircuit,
    ParameterizedGate,
    noisy_state_preparation_contribution,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def state(tensor: NDArray) -> MPS:
    """Keep one supplied tensor identity unchanged in a q1 MPS fixture.

    Returns:
        A one-site state retaining the original buffer or view.
    """
    return MPS(1, tensors=[tensor])


def test_repeated_state_references_and_disjoint_views_share_one_allocation() -> None:
    """Logical occurrences grow while one shared owner, including unseen elements, is retained once."""
    owner = np.arange(24, dtype=np.complex128)
    first = owner[:2].reshape(2, 1, 1)
    second = owner[-2:].reshape(2, 1, 1)
    initial = state(first)
    trajectory = KrotovTrajectory(states=[initial, initial], gate_outputs=[state(second)], noise_maps=[])
    target = state(first)
    report = mps_memory.forward_tape_storage([trajectory, trajectory], target)
    tape = report["forward_tape"]
    assert tape["tensor_references"] == 6
    assert tape["unique_array_objects"] == 2
    assert tape["logical_tensor_bytes"] == 6 * first.nbytes
    assert tape["unique_array_logical_bytes"] == first.nbytes + second.nbytes
    assert tape["complete_backing_payload_bytes"] == owner.nbytes
    assert report["shared_tape_target_buffers"] == 1
    assert report["shared_known_numpy_payload_bytes"] == owner.nbytes
    assert report["target_incremental_known_numpy_payload_bytes"] == 0
    assert report["retained_union"]["complete_backing_payload_bytes"] == owner.nbytes
    assert len(report["buffers"]) == 1
    assert report["unique_state_objects"] == 2
    assert json.loads(json.dumps(report)) == report


def test_noncontiguous_negative_stride_views_keep_full_owner() -> None:
    """Logical view bytes are not substituted for retained backing size or stride span."""
    owner = np.arange(20, dtype=np.complex128)
    tensor = owner[10:2:-4].reshape(2, 1, 1)
    trajectory = KrotovTrajectory(states=[state(tensor)], gate_outputs=[], noise_maps=[])
    report = mps_memory.forward_tape_storage([trajectory])
    assert report["forward_tape"]["noncontiguous_array_objects"] == 1
    assert report["forward_tape"]["logical_tensor_bytes"] == 32
    assert report["forward_tape"]["complete_backing_payload_bytes"] == 320
    assert tensor.strides[0] < 0


def test_stride_wrapper_and_memoryview_follow_numpy_owner() -> None:
    """Public base chains through stride wrappers and memoryviews resolve to the same owner."""
    owner = np.arange(4, dtype=np.complex128)
    first = np.lib.stride_tricks.as_strided(owner, shape=(2, 1, 1), strides=(16, 16, 16))
    second = np.frombuffer(owner.data, dtype=np.complex128)[2:].reshape(2, 1, 1)
    trajectory = KrotovTrajectory(states=[state(first)], gate_outputs=[state(second)], noise_maps=[])
    report = mps_memory.forward_tape_storage([trajectory])
    assert report["forward_tape"]["unique_backing_buffers"] == 1
    assert report["forward_tape"]["complete_backing_payload_bytes"] == owner.nbytes


def test_distinct_identical_arrays_and_independent_target_count_separately() -> None:
    """Equal numerical values never merge physically independent allocations."""
    first = np.ones((2, 1, 1), dtype=np.complex128)
    second = first.copy()
    target = first.copy()
    trajectory = KrotovTrajectory(states=[state(first)], gate_outputs=[state(second)], noise_maps=[])
    report = mps_memory.forward_tape_storage([trajectory], target)
    assert report["forward_tape"]["complete_backing_payload_bytes"] == 64
    assert report["exact_target"]["complete_backing_payload_bytes"] == 32
    assert report["retained_union"]["complete_backing_payload_bytes"] == 96
    assert report["target_incremental_known_numpy_payload_bytes"] == 32
    assert report["shared_tape_target_buffers"] == 0


def test_empty_tape_and_zero_length_tensor_have_explicit_zero_payloads() -> None:
    """Empty observed storage is zero; an absent target is separately labeled as absent."""
    report = mps_memory.forward_tape_storage([])
    assert report["trajectory_count"] == 0
    assert not report["target_supplied"]
    assert report["retained_union"]["complete_backing_payload_bytes"] == 0
    target = np.empty((0,), dtype=np.complex128)
    observed = mps_memory.forward_tape_storage([], target)
    assert observed["target_supplied"]
    assert observed["exact_target"]["unique_array_objects"] == 1
    assert observed["exact_target"]["complete_backing_payload_bytes"] == 0


def test_external_buffer_does_not_become_known_owned_zero_memory() -> None:
    """Unknown foreign allocation bytes remain null although logical elements and aliases are known."""
    external = bytearray(64)
    first = np.frombuffer(external, dtype=np.complex128)[:2].reshape(2, 1, 1)
    second = np.frombuffer(external, dtype=np.complex128)[2:].reshape(2, 1, 1)
    trajectory = KrotovTrajectory(states=[state(first)], gate_outputs=[], noise_maps=[])
    report = mps_memory.forward_tape_storage([trajectory], second)
    assert report["retained_union"]["logical_tensor_bytes"] == 64
    assert report["retained_union"]["unique_backing_buffers"] == 1
    assert report["retained_union"]["unknown_backing_buffers"] == 1
    assert report["retained_union"]["complete_backing_payload_bytes"] is None
    assert report["retained_union"]["known_numpy_owned_payload_bytes"] == 0
    assert report["shared_tape_target_buffers"] == 1


@pytest.mark.parametrize("qubits", [1, 2, 3])
def test_public_returned_tape_includes_states_and_pre_noise_gate_outputs(qubits: int) -> None:
    """Small actual public adjoints expose every retained forward state without instrumentation or replay."""
    circuit = ParameterizedCircuit(
        qubits, [ParameterizedGate("ry", (0,), param_index=0), ParameterizedGate("rz", (qubits - 1,), param_index=1)]
    )
    target = MPS(qubits, state="zeros")
    _, _, _, trajectories = noisy_state_preparation_contribution(
        circuit,
        np.array([0.2, -0.4]),
        target,
        None,
        KrotovTJMOptions(num_trajectories=2),
        MPS(qubits, state="zeros"),
        KrotovTruncation(),
        fixed_noise_maps=[[KrotovNoiseMap() for _ in circuit.gates] for _ in range(2)],
    )
    before = copy.deepcopy(trajectories)
    report = mps_memory.forward_tape_storage(trajectories, target)
    assert report["states"]["tensor_references"] == 2 * 3 * qubits
    assert report["gate_outputs"]["tensor_references"] == 2 * 2 * qubits
    assert report["forward_tape"]["tensor_references"] == 2 * 5 * qubits
    assert report["forward_tape"]["complete_backing_payload_bytes"] > report["states"]["complete_backing_payload_bytes"]
    assert (
        report["retained_union"]["complete_backing_payload_bytes"]
        >= report["forward_tape"]["complete_backing_payload_bytes"]
    )
    for original, after in zip(before, trajectories, strict=True):
        for left, right in zip(
            [*original.states, *original.gate_outputs], [*after.states, *after.gate_outputs], strict=True
        ):
            for a, b in zip(left.tensors, right.tensors, strict=True):
                np.testing.assert_array_equal(a, b)
    assert not report["peak_claim"]
    assert "unreturned adjoint and derivative work" in report["exclusions"]


@pytest.mark.parametrize("target", [np.array([object()]), "not an array"])
def test_object_payloads_and_nonarray_targets_fail_closed(target: object) -> None:
    """Python object referents cannot be reported as complete NumPy numeric storage."""
    with pytest.raises(ValueError, match="without Python object"):
        mps_memory.forward_tape_storage([], cast("NDArray", target))
