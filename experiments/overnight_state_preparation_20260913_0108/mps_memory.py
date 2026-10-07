# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Observe retained public forward MPS tapes without altering numerical execution.

This measures NumPy tensor payloads reachable through both ``states`` and
``gate_outputs``. It is not a process RSS or a peak-working-memory estimator:
noise-map matrices, Python objects, allocator overhead, and unreturned adjoint,
derivative, contraction and factorization temporaries are outside the scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from mqt.yaqs.core.data_structures.mps import MPS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import KrotovTrajectory

VERSION = "overnight.retained_forward_mps_tensor_storage.v1"


def _backing(array: NDArray) -> tuple[object, int | None]:
    """Find a NumPy-owned base payload, retaining opaque foreign buffers as unknown."""
    current: object = array
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, np.ndarray) and current.flags["OWNDATA"]:
            return current, int(current.nbytes)
        parent = current.obj if isinstance(current, memoryview) else getattr(current, "base", None)
        if parent is None:
            return current, None
        current = parent
    return current, None


@dataclass
class _Inventory:
    """Deduplicate live array identities and base allocations without copying tensor data."""

    arrays: dict[int, NDArray] = field(default_factory=dict)
    backing_for_array: dict[int, int] = field(default_factory=dict)
    buffers: dict[int, dict[str, Any]] = field(default_factory=dict)

    def register(self, tensor: NDArray) -> int:
        """Record one numeric tensor reference and assign its backing a local ordinal."""
        if not isinstance(tensor, np.ndarray) or tensor.dtype.hasobject:
            msg = "MPS memory observation requires NumPy arrays without Python object payloads."
            raise ValueError(msg)
        identity = id(tensor)
        if identity not in self.arrays:
            owner, payload_bytes = _backing(tensor)
            self.arrays[identity] = tensor
            self.backing_for_array[identity] = id(owner)
            if id(owner) not in self.buffers:
                self.buffers[id(owner)] = {
                    "buffer_index": len(self.buffers),
                    "owner_type": type(owner).__name__,
                    "numpy_owned_payload_bytes": payload_bytes,
                    "kind": "numpy_owned" if payload_bytes is not None else "opaque_external_backing",
                }
        return identity

    def backing_ids(self, references: Sequence[int]) -> set[int]:
        """Return distinct retained backing identities for a tensor-reference collection."""
        return {self.backing_for_array[key] for key in references}

    def owned_bytes(self, backing_ids: set[int]) -> int:
        """Sum only measured NumPy-owned payloads, leaving unknown owners outside this subtotal."""
        return sum(self.buffers[key]["numpy_owned_payload_bytes"] or 0 for key in backing_ids)

    def summarize(self, references: Sequence[int]) -> dict[str, Any]:
        """Separate logical tensor occurrences, unique ndarray objects, and retained backing storage."""
        unique = set(references)
        backing = self.backing_ids(references)
        unknown = sum(self.buffers[key]["numpy_owned_payload_bytes"] is None for key in backing)
        owned = self.owned_bytes(backing)
        return {
            "tensor_references": len(references),
            "unique_array_objects": len(unique),
            "repeated_array_references": len(references) - len(unique),
            "logical_tensor_bytes": sum(int(self.arrays[key].nbytes) for key in references),
            "unique_array_logical_bytes": sum(int(self.arrays[key].nbytes) for key in unique),
            "view_array_objects": sum(not self.arrays[key].flags["OWNDATA"] for key in unique),
            "noncontiguous_array_objects": sum(
                not (self.arrays[key].flags["C_CONTIGUOUS"] or self.arrays[key].flags["F_CONTIGUOUS"]) for key in unique
            ),
            "unique_backing_buffers": len(backing),
            "known_numpy_owned_payload_bytes": owned,
            "unknown_backing_buffers": unknown,
            "complete_backing_payload_bytes": owned if unknown == 0 else None,
            "status": "complete_numpy_payloads" if unknown == 0 else "unknown_external_backing_size",
            "buffer_indices": sorted(self.buffers[key]["buffer_index"] for key in backing),
        }


def forward_tape_storage(
    trajectories: Sequence[KrotovTrajectory], target: MPS | NDArray | None = None
) -> dict[str, Any]:
    """Measure all retained forward-state tensors and a separately supplied target.

    Both public ``trajectory.states`` and ``trajectory.gate_outputs`` are
    included. Repeated state/tensor references and views may share whole NumPy
    base allocations; each allocation contributes once to the retained union.
    Noncontiguous view ``nbytes`` measures logical elements, while base ``nbytes``
    measures its full NumPy-owned payload, including elements outside the view.

    Args:
        trajectories: Live trajectories returned by
            ``noisy_state_preparation_contribution``. An empty sequence is valid.
        target: Optional exact target retained by the caller, as an MPS or dense
            array. This utility never factors it or verifies its approximation
            error; exactness remains the caller's responsibility.

    Returns:
        A JSON-compatible snapshot with separate tape, target and union totals,
        target/tape shared allocations, and a buffer registry without addresses.
        Foreign-backed arrays retain their logical bytes and an explicit unknown
        backing total. No tensor contents, arrays or live references are returned.

    Raises:
        ValueError: If a state lacks numeric NumPy tensors or a target has an
            unsupported representation.
        TypeError: If a retained trajectory contains a non-MPS state object.
    """
    inventory = _Inventory()
    states: list[int] = []
    outputs: list[int] = []
    state_references: list[int] = []
    per_trajectory = []
    for trajectory in trajectories:
        local_states, local_outputs = [], []
        for collection, destination in ((trajectory.states, local_states), (trajectory.gate_outputs, local_outputs)):
            for state in collection:
                if not isinstance(state, MPS):
                    msg = "A retained forward trajectory must contain MPS state objects."
                    raise TypeError(msg)
                state_references.append(id(state))
                destination.extend(inventory.register(tensor) for tensor in state.tensors)
        states.extend(local_states)
        outputs.extend(local_outputs)
        per_trajectory.append({
            "states": len(trajectory.states),
            "gate_outputs": len(trajectory.gate_outputs),
            "storage": inventory.summarize([*local_states, *local_outputs]),
        })
    target_tensors = target.tensors if isinstance(target, MPS) else [] if target is None else [target]
    target_refs = [inventory.register(tensor) for tensor in target_tensors]
    tape_refs = [*states, *outputs]
    tape_backing, target_backing = inventory.backing_ids(tape_refs), inventory.backing_ids(target_refs)
    shared = tape_backing & target_backing
    incremental = target_backing - tape_backing
    return {
        "version": VERSION,
        "trajectory_count": len(trajectories),
        "state_references": len(state_references),
        "unique_state_objects": len(set(state_references)),
        "states": inventory.summarize(states),
        "gate_outputs": inventory.summarize(outputs),
        "forward_tape": inventory.summarize(tape_refs),
        "target_supplied": target is not None,
        "exact_target": inventory.summarize(target_refs),
        "retained_union": inventory.summarize([*tape_refs, *target_refs]),
        "shared_tape_target_buffers": len(shared),
        "shared_known_numpy_payload_bytes": inventory.owned_bytes(shared),
        "target_incremental_known_numpy_payload_bytes": inventory.owned_bytes(incremental),
        "per_trajectory": per_trajectory,
        "buffers": list(inventory.buffers.values()),
        "scope": "retained public states and gate_outputs MPS tensor payloads plus the supplied exact target",
        "sharing_scope": "whole retained base allocations, not intersection of logical view elements",
        "target_exactness": "caller supplied; no target construction, truncation or exactness verification performed",
        "exclusions": [
            "unreturned adjoint and derivative work",
            "transient contractions and target factorization",
            "noise-map and gate matrices",
            "Python object/container overhead",
            "allocator padding and process RSS",
            "states not simultaneously retained in the supplied collection",
        ],
        "peak_claim": False,
    }
