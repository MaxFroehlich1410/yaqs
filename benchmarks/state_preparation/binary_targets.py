# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Streamed generation and validation of fixed binary state-preparation targets."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import scipy

from benchmarks import generate_state_preparation_targets as generation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

BINARY_QUBIT_COUNTS = (20, 24)
BINARY_TARGET_FORMAT = "yaqs.state_preparation_targets.binary.v1"
DEFAULT_BINARY_TARGET_PATH = Path(__file__).parents[1] / "state_preparation_targets" / "manifest.json"
LEGACY_SHA256 = "d53cd272b6c665b3a92de2d6b1e335b85ef77f74016b0f20cf7ea6a0f486b839"
CHUNK_SIZE = 2**18


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without reading an entire array into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_path(root: Path, filename: object) -> Path:
    """Resolve a manifest payload within its own directory.

    Returns:
        Absolute payload path.

    Raises:
        ValueError: If the payload path escapes the manifest directory.
    """
    if not isinstance(filename, str) or not filename or Path(filename).is_absolute():
        msg = "Binary payload file must be a relative path."
        raise ValueError(msg)
    path = (root / filename).resolve()
    if not path.is_relative_to(root.resolve()):
        msg = "Binary payload file escapes the manifest directory."
        raise ValueError(msg)
    return path


def validated_vector(path: Path, checksum: str, num_qubits: int, norm: float) -> NDArray[np.complex128]:
    """Memory-map and authenticate one normalized little-endian complex128 target.

    Returns:
        Read-only memory-mapped state vector.

    Raises:
        ValueError: If the checksum, array representation, norm, or phase is invalid.
    """
    if file_sha256(path) != checksum:
        msg = f"Binary target checksum mismatch: {path}."
        raise ValueError(msg)
    vector = np.load(path, mmap_mode="r", allow_pickle=False)
    if vector.dtype.str != "<c16" or vector.shape != (2**num_qubits,) or not vector.flags.c_contiguous:
        msg = f"Invalid binary target dtype or shape: {path}."
        raise ValueError(msg)
    squared_norm = 0.0
    largest = 0.0j
    for offset in range(0, vector.size, CHUNK_SIZE):
        chunk = vector[offset : offset + CHUNK_SIZE]
        if not np.all(np.isfinite(chunk)):
            msg = f"Non-finite binary target amplitudes: {path}."
            raise ValueError(msg)
        squared_norm += float(np.vdot(chunk, chunk).real)
        pivot = complex(chunk[int(np.argmax(np.abs(chunk)))])
        if abs(pivot) > abs(largest):
            largest = pivot
    if not np.isclose(squared_norm, 1.0, rtol=0, atol=1e-10) or not np.isclose(
        norm, np.sqrt(squared_norm), rtol=0, atol=1e-10
    ):
        msg = f"Binary target norm is inconsistent: {path}."
        raise ValueError(msg)
    if largest.real <= 0 or abs(largest.imag) > 1e-10:
        msg = f"Binary target violates the global-phase convention: {path}."
        raise ValueError(msg)
    return cast("NDArray[np.complex128]", vector)


def tfim_diagnostics(state: NDArray[np.complex128], num_qubits: int, field: float, energy: float) -> dict[str, float]:
    """Independently check TFIM ground energy, even parity, and eigenpair residual.

    The open-chain free-fermion energy is minus the sum of singular values of
    the n-by-n bidiagonal matrix with h on its diagonal and J=1 below it.

    Returns:
        Ground-state validation diagnostics.

    Raises:
        ValueError: If the state is not the symmetric finite-chain ground state.
    """
    half = state.size // 2
    reduced = np.array(state[:half].real * np.sqrt(2.0))
    operator = generation.EvenTfimLinearOperator(num_qubits, field)
    residual = float(np.linalg.norm(operator @ reduced - energy * reduced))
    model = np.diag(np.full(num_qubits, field)) + np.diag(np.ones(num_qubits - 1), -1)
    exact_energy = -float(np.sum(np.linalg.svd(model, compute_uv=False)))
    parity_error = 0.0
    imaginary_error = 0.0
    for offset in range(0, half, CHUNK_SIZE):
        end = min(offset + CHUNK_SIZE, half)
        parity_error = max(parity_error, float(np.max(np.abs(state[offset:end] - state[::-1][offset:end]))))
        imaginary_error = max(imaginary_error, float(np.max(np.abs(state[offset:end].imag))))
    if residual > 1e-9 or abs(exact_energy - energy) > 1e-9 or parity_error > 1e-12 or imaginary_error > 1e-12:
        msg = "TFIM target failed the residual, exact-energy, reality, or even-parity check."
        raise ValueError(msg)
    return {
        "eigenpair_residual_norm": residual,
        "free_fermion_ground_energy": exact_energy,
        "ground_energy_absolute_error": abs(exact_energy - energy),
        "global_x_even_max_error": parity_error,
    }


def _native_mps(num_qubits: int, seed: int, bond: int, state: NDArray[np.complex128]) -> dict[str, NDArray[np.float64]]:
    """Recreate the original random tensors with the dense state's normalization and phase.

    Returns:
        Site-indexed real tensors with the dense target's exact normalization.
    """
    rng = np.random.default_rng(seed)
    bonds = (1, *([bond] * (num_qubits - 1)), 1)
    tensors = [rng.standard_normal((2, bonds[i], bonds[i + 1])) for i in range(num_qubits)]
    pivot = int(np.argmax(np.abs(state)))
    amplitude = np.ones(1)
    for site, tensor in enumerate(tensors):
        amplitude = np.matmul(amplitude, tensor[(pivot >> site) & 1])
    tensors[0] *= float(state[pivot].real / amplitude[0])
    return {f"site_{site}": tensor for site, tensor in enumerate(tensors)}


def _save_native_mps(path: Path, tensors: dict[str, NDArray[np.float64]]) -> None:
    """Write portable pickle-free NPZ tensors with deterministic ZIP metadata."""
    temporary = path.with_suffix(".tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        for name, tensor in tensors.items():
            member = ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = ZIP_DEFLATED
            with archive.open(member, "w") as stream:
                np.lib.format.write_array(stream, tensor, allow_pickle=False)
    temporary.replace(path)


def validate_definition(identifier: str, count: int, parameters: dict[str, object]) -> None:
    """Reject metadata that changes a fixed target's mathematical definition.

    Raises:
        ValueError: If scientific parameters differ from the fixed target family.
    """
    if identifier.startswith("tfim_"):
        spec = next(spec for spec in generation.TFIM_SPECS if spec.target_id == identifier)
        expected = {
            "boundary_conditions": "open",
            "model": "uniform_1d_transverse_field_ising_model",
            "hamiltonian": "-J*sum_i Z_i Z_{i+1} - h*sum_i X_i",
            "regime": spec.regime,
            "h_over_j": spec.h_over_j,
            "j_coupling": 1.0,
            "j_couplings": [1.0] * (count - 1),
            "transverse_field": spec.h_over_j,
            "transverse_fields": [spec.h_over_j] * count,
            "eigensolver_initial_vector_seed": spec.seed + 10_000 * count,
            "eigensolver": "scipy.sparse.linalg.eigsh in global-X-even sector",
            "eigensolver_ncv": 12,
            "eigensolver_tolerance": 1e-13,
        }
        valid = all(parameters.get(key) == value for key, value in expected.items())
        energy = parameters.get("ground_energy")
        valid = valid and isinstance(energy, (int, float)) and np.isfinite(energy)
    else:
        small_metadata, _ = next(generation.iter_target_arrays((2,), target_ids=(identifier,)))
        expected = cast("dict", small_metadata["parameters"])
        if identifier.startswith("random_mps_"):
            bond = expected["max_bond_dimension"]
            expected["bond_dimensions"] = [1, *([bond] * (count - 1)), 1]
        valid = parameters == expected
    if not valid:
        msg = f"Binary target parameters change the fixed scientific definition: {identifier}."
        raise ValueError(msg)


def _atomic_json(path: Path, document: dict) -> None:
    """Publish a complete manifest after its payload files are durable."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def generate_binary_targets(
    manifest_path: Path,
    num_qubits: Sequence[int],
    target_ids: Sequence[str],
) -> None:
    """Generate selected binary targets, retaining unrelated completed records.

    Args:
        manifest_path: Destination manifest, separate from the frozen legacy JSON.
        num_qubits: Chain lengths to generate.
        target_ids: Selected targets, emitted in benchmark order.

    Raises:
        ValueError: If an existing manifest uses another schema or frozen fixture.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "format": BINARY_TARGET_FORMAT,
        "generated_by": "benchmarks/generate_state_preparation_targets.py --binary",
        "basis_order": "little_endian: amplitude index k has qubit i equal to bit i of k",
        "global_phase": "fixed so the largest-magnitude amplitude is positive real",
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "legacy_fixture_sha256": LEGACY_SHA256,
        "qubit_counts": [],
        "target_ids": list(generation.TARGET_IDS),
        "targets": [],
    }
    if manifest_path.exists():
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if document["format"] != BINARY_TARGET_FORMAT or document["legacy_fixture_sha256"] != LEGACY_SHA256:
            msg = "Existing binary manifest has an incompatible format or legacy fixture."
            raise ValueError(msg)
    records = {(r["num_qubits"], r["target_id"]): r for r in cast("list[dict]", document["targets"])}
    # Generate one target per iterator to release previous solver/state allocations.
    for count in num_qubits:
        for identifier in generation.TARGET_IDS:
            if identifier not in target_ids:
                continue
            sys.stdout.write(f"Generating q{count} {identifier}\n")
            sys.stdout.flush()
            metadata, state = next(
                generation.iter_target_arrays((count,), target_ids=(identifier,), scalable_tfim=True)
            )
            folder = manifest_path.parent / f"q{count}"
            folder.mkdir(exist_ok=True)
            destination = folder / f"{identifier}.npy"
            temporary = destination.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, np.asarray(state, dtype="<c16"), allow_pickle=False)
            temporary.replace(destination)
            metadata["array"] = {
                "file": destination.relative_to(manifest_path.parent).as_posix(),
                "sha256": file_sha256(destination),
                "dtype": "<c16",
                "shape": [2**count],
                "size_bytes": destination.stat().st_size,
            }
            parameters = cast("dict", metadata["parameters"])
            if identifier.startswith("tfim_"):
                parameters["eigensolver"] = "scipy.sparse.linalg.eigsh in global-X-even sector"
                parameters["eigensolver_ncv"] = 12
                parameters["eigensolver_tolerance"] = 1e-13
                metadata["validation"] = tfim_diagnostics(
                    state, count, parameters["h_over_j"], parameters["ground_energy"]
                )
            if identifier.startswith("random_mps_"):
                tensors_path = destination.with_suffix(".mps.npz")
                tensors = _native_mps(count, cast("int", metadata["seed"]), parameters["max_bond_dimension"], state)
                _save_native_mps(tensors_path, tensors)
                metadata["mps"] = {
                    "file": tensors_path.relative_to(manifest_path.parent).as_posix(),
                    "sha256": file_sha256(tensors_path),
                }
            records[count, identifier] = metadata
            document["qubit_counts"] = sorted({key[0] for key in records})
            document["targets"] = [
                records[key] for key in sorted(records, key=lambda key: (key[0], generation.TARGET_IDS.index(key[1])))
            ]
            _atomic_json(manifest_path, document)
            del state


def check_binary_targets(manifest_path: Path, num_qubits: Sequence[int], target_ids: Sequence[str]) -> None:
    """Check stored arrays against hashes, definitions, and independent TFIM diagnostics.

    Raises:
        ValueError: If selected targets are missing or differ from their definitions.
    """
    from .targets import load_target_collection  # noqa: PLC0415 - loader/generator import cycle.

    collection = load_target_collection(manifest_path)
    for count in num_qubits:
        for identifier in target_ids:
            record = collection.load(count, identifier)
            state = record.state_vector
            if identifier.startswith("tfim_"):
                parameters = record.parameters
                tfim_diagnostics(
                    state, count, cast("float", parameters["h_over_j"]), cast("float", parameters["ground_energy"])
                )
            else:
                _, expected = next(generation.iter_target_arrays((count,), target_ids=(identifier,)))
                if not np.allclose(state, expected, rtol=1e-12, atol=1e-14):
                    msg = f"Binary target does not match its generation definition: {record.key}."
                    raise ValueError(msg)
                del expected
            if identifier.startswith("random_mps_"):
                mps = record.to_mps()
                # Sequentially contract the tiny native MPS for independent full-vector validation.
                dense = mps.to_vec()
                if not np.allclose(dense, state, rtol=1e-12, atol=1e-14):
                    msg = f"Native MPS does not match its dense target: {record.key}."
                    raise ValueError(msg)
                del dense, mps
            sys.stdout.write(f"Validated q{count} {identifier}\n")
            sys.stdout.flush()
            del state


def binary_main(
    output: Path | None, num_qubits: Sequence[int] | None, target_ids: Sequence[str] | None, *, check: bool
) -> int:
    """Run the explicitly selected binary target generation or integrity check.

    Returns:
        Zero after successful generation/checking.

    Raises:
        ValueError: If a requested chain length is unsupported.
    """
    counts = BINARY_QUBIT_COUNTS if num_qubits is None else tuple(num_qubits)
    identifiers = generation.TARGET_IDS if target_ids is None else tuple(target_ids)
    if not counts or len(set(counts)) != len(counts) or any(count not in BINARY_QUBIT_COUNTS for count in counts):
        msg = f"Binary target qubit counts must be unique members of {BINARY_QUBIT_COUNTS}."
        raise ValueError(msg)
    manifest_path = DEFAULT_BINARY_TARGET_PATH if output is None else output
    legacy_path = generation.DEFAULT_OUTPUT_PATH
    if manifest_path.resolve() == legacy_path.resolve() or file_sha256(legacy_path) != LEGACY_SHA256:
        msg = "The frozen six/twelve-qubit fixture must remain unchanged."
        raise ValueError(msg)
    if check:
        check_binary_targets(manifest_path, counts, identifiers)
    else:
        generate_binary_targets(manifest_path, counts, identifiers)
    return 0
