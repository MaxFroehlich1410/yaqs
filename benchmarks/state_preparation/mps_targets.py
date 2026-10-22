# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Generate and authenticate random targets directly as small MPS tensor files."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization.mps_backend import mps_overlap

from .binary_targets import file_sha256, payload_path

if TYPE_CHECKING:
    from pathlib import Path

FORMAT = "yaqs.state_preparation_targets.native_mps.v1"
BASIS_ORDER = "site i is bit i of the little-endian amplitude index"
PHASE = "all-zero amplitude positive real"


def _definition(num_qubits: int, bond: int, seed: int) -> dict:
    """Validate the explicitly separate native-target definition.

    Returns:
        Portable generation metadata independent of any dense fixture.

    Raises:
        ValueError: If the register, bond dimension or seed is invalid.
    """
    if (
        type(num_qubits) is not int
        or not 2 <= num_qubits <= 64
        or type(bond) is not int
        or not 1 <= bond <= 64
        or type(seed) is not int
        or not 0 <= seed < 2**32
    ):
        msg = "Require 2..64 qubits, bond 1..64, and an integer seed in [0, 2**32)."
        raise ValueError(msg)
    return {
        "format": FORMAT,
        "num_qubits": num_qubits,
        "bond": bond,
        "seed": seed,
        "target_id": f"random_mps_bond{bond}",
        "basis_order": BASIS_ORDER,
        "global_phase": PHASE,
        "generator": "numpy.default_rng(seed).standard_normal; open-boundary physical/left/right tensors",
    }


def zero_amplitude(target: MPS) -> complex:
    """Contract the all-zero amplitude without constructing a state vector.

    Returns:
        The complex all-zero computational-basis amplitude.
    """
    value = np.ones(1, dtype=np.complex128)
    for tensor in target.tensors:
        value = np.matmul(value, tensor[0])
    return complex(value[0])


def random_mps_target(num_qubits: int = 30, bond: int = 2, seed: int = 5002) -> MPS:
    """Generate the existing real Gaussian-tensor MPS family without dense expansion.

    Global phase uses the zero amplitude, since searching the largest amplitude
    would require exponential work. Physical states agree with the dense family
    up to global phase. Target normalization uses a two-index transfer matrix.

    Returns:
        A private, normalized open-boundary MPS.

    Raises:
        ValueError: If arguments are invalid or normalization is singular.
    """
    _definition(num_qubits, bond, seed)
    rng = np.random.default_rng(seed)
    bonds = (1, *([bond] * (num_qubits - 1)), 1)
    target = MPS(
        num_qubits,
        tensors=[
            np.asarray(rng.standard_normal((2, bonds[i], bonds[i + 1])), dtype=np.complex128) for i in range(num_qubits)
        ],
    )
    norm = float(mps_overlap(target, target).real)
    if not np.isfinite(norm) or norm <= 0:
        msg = "Native target has singular normalization."
        raise ValueError(msg)
    target.tensors[0] /= np.sqrt(norm)
    pivot = zero_amplitude(target)
    if not np.isfinite(pivot) or abs(pivot) == 0:
        msg = "Native target has no usable all-zero phase pivot."
        raise ValueError(msg)
    target.tensors[0] *= pivot.conjugate() / abs(pivot)
    return target


def write_mps_target(path: Path, num_qubits: int = 30, bond: int = 2, seed: int = 5002) -> dict:
    """Write tensor-only NPZ data and its authenticated definition, refusing overwrite.

    Args:
        path: New JSON manifest path, with a sibling NPZ payload.
        num_qubits: Register size, independent of the archived fixture widths.
        bond: Maximum internal tensor bond dimension.
        seed: Target-generation seed, separate from the optimizer seed.

    Returns:
        The committed manifest.

    Raises:
        ValueError: If the definition or manifest suffix is invalid.
        FileExistsError: If a manifest or tensor payload already exists.
    """
    definition = _definition(num_qubits, bond, seed)
    if path.suffix != ".json":
        msg = "Native target manifest must use a .json suffix."
        raise ValueError(msg)
    if path.exists():
        raise FileExistsError(path)
    target = random_mps_target(num_qubits, bond, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = path.with_suffix(".npz")
    # Exclusive creation protects an existing partial payload as well as a manifest.
    with ZipFile(array, "x", compression=ZIP_DEFLATED) as archive:
        for i, tensor in enumerate(target.tensors):
            member = ZipInfo(f"site_{i}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = ZIP_DEFLATED
            with archive.open(member, "w") as stream:
                np.lib.format.write_array(stream, tensor, allow_pickle=False)
    record = {
        **definition,
        "numpy_version": np.__version__,
        "array": {"file": array.name, "sha256": file_sha256(array)},
        "tensor_bytes": sum(t.nbytes for t in target.tensors),
    }
    with path.open("x", encoding="utf-8") as document:
        document.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def load_mps_target(path: Path) -> tuple[MPS, dict]:
    """Authenticate tensor bytes, scientific definition, shapes, norm and phase.

    Returns:
        A fresh mutable MPS and the manifest, without opening any dense fixture.

    Raises:
        ValueError: If metadata or the numerical representation is invalid.
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = _definition(record["num_qubits"], record["bond"], record["seed"])
    if any(record.get(k) != v for k, v in expected.items()):
        msg = "Native target definition differs from its format."
        raise ValueError(msg)
    array = payload_path(path.parent, record["array"]["file"])
    if file_sha256(array) != record["array"]["sha256"]:
        msg = "Native target checksum mismatch."
        raise ValueError(msg)
    count, bond = record["num_qubits"], record["bond"]
    bonds = (1, *([bond] * (count - 1)), 1)
    with np.load(array, allow_pickle=False) as archive:
        if set(archive.files) != {f"site_{i}" for i in range(count)}:
            msg = "Native target tensor keys differ."
            raise ValueError(msg)
        tensors = [archive[f"site_{i}"] for i in range(count)]
    if (
        any(
            t.dtype != np.complex128 or t.shape != (2, bonds[i], bonds[i + 1]) or not np.all(np.isfinite(t))
            for i, t in enumerate(tensors)
        )
        or sum(t.nbytes for t in tensors) != record["tensor_bytes"]
    ):
        msg = "Native target tensor shape, dtype, values or byte count is invalid."
        raise ValueError(msg)
    target = MPS(count, tensors=tensors)
    pivot = zero_amplitude(target)
    if abs(mps_overlap(target, target) - 1) > 1e-10 or pivot.real <= 0 or abs(pivot.imag) > 1e-12:
        msg = "Native target normalization or phase is invalid."
        raise ValueError(msg)
    regenerated = random_mps_target(count, bond, record["seed"])
    if any(
        not np.allclose(a, b, atol=1e-14, rtol=1e-13) for a, b in zip(target.tensors, regenerated.tensors, strict=True)
    ):
        msg = "Native target tensors do not match their generation seed."
        raise ValueError(msg)
    return target, record
