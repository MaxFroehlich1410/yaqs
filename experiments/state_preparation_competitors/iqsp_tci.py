# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Published two-site TCI target construction for single-tooth IQSP benchmarks.

TensorCrossInterpolation.jl supplies the algorithm. This bridge specifies its
inputs, exports the tensor train without SVD, and independently checks the
small benchmark fixtures. It is not a reconstruction of unpublished author
settings, nor a general tree-TCI implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from experiments.state_preparation_competitors.common import validate_target
from experiments.state_preparation_competitors.iqsp import gaussian_state
from experiments.state_preparation_competitors.iqsp_comb_backend import CombLayout, CombTarget

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

    from numpy.typing import ArrayLike, NDArray

resource: ModuleType | None
try:
    import resource as _resource
except ImportError:  # Windows can still import and use the legacy exact backend.
    resource = None
else:
    resource = _resource


TCI_PROJECT = Path(__file__).with_name("tci_julia")
TCI_SOURCE = "TensorCrossInterpolation.jl/0.9.19/crossinterpolate2"


@dataclass(frozen=True)
class TCIOptions:
    """Recorded numerical settings; these are not claimed as Ballarin defaults.

    The target rank cap follows the supplement's largest tested rank. The
    tolerance is a library interpolation criterion, not a state fidelity bound.
    The independent normalized-vector check must pass before using a target.
    """

    max_bond: int = 16
    tolerance: float = 1e-12
    max_iterations: int = 20
    seed: int = 0
    max_vector_error: float = 1e-10
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        """Reject invalid ranks, budgets, tolerances, and seeds."""
        for name in ("max_bond", "max_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                msg = f"{name} must be a positive integer."
                raise ValueError(msg)
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            msg = "seed must be an integer in [0, 2**31)."
            raise ValueError(msg)
        for name in ("tolerance", "max_vector_error", "timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                msg = f"{name} must be finite and positive."
                raise ValueError(msg)
        if self.tolerance >= 1 or self.max_vector_error >= 1:
            msg = "Interpolation and validation tolerances must be less than one."
            raise ValueError(msg)


DEFAULT_TCI_OPTIONS = TCIOptions()


def _decode_target(
    raw: dict[str, Any], reference: NDArray[np.complex128], options: TCIOptions
) -> tuple[CombTarget, dict]:
    """Normalize TCI cores by contraction and audit every small-fixture amplitude."""
    n = reference.size.bit_length() - 1
    cores = []
    for encoded in raw["cores"]:
        values = np.asarray(encoded["real"]) + 1j * np.asarray(encoded["imag"])
        core = values.reshape(tuple(encoded["shape"]), order="F")
        if core.ndim != 3 or core.shape[1] != 2 or not np.isfinite(core).all():
            msg = "Invalid TCI tensor dimensions or entries."
            raise ValueError(msg)
        cores.append(core)
    if len(cores) != n or cores[0].shape[0] != 1 or cores[-1].shape[2] != 1:
        msg = "TCI tensor train does not match the register."
        raise ValueError(msg)
    environment = np.ones((1, 1), dtype=np.complex128)
    for core in cores:
        environment = np.einsum("ab,api,bpj->ij", environment, core, core.conj(), optimize=True)
    norm_squared = complex(environment.item())
    if not np.isfinite(norm_squared) or norm_squared.real <= 0 or abs(norm_squared.imag) > 1e-10 * norm_squared.real:
        msg = "TCI target has an invalid norm."
        raise ValueError(msg)
    cores[0] /= math.sqrt(norm_squared.real)
    # Dense contraction is an independent bounded audit, never a TCI input or factorization.
    dense = cores[0][0]
    for core in cores[1:]:
        dense = np.tensordot(dense, core, axes=(-1, 0))
    vector = dense[..., 0].transpose(tuple(reversed(range(n)))).reshape(-1)
    overlap = complex(np.vdot(reference, vector))
    error = float(np.linalg.norm(vector * np.exp(-1j * np.angle(overlap)) - reference))
    if error > options.max_vector_error:
        msg = f"TCI target vector error {error:.3g} exceeds {options.max_vector_error:.3g}; no fallback or retuning."
        raise ValueError(msg)
    tensors = tuple(torch.tensor(core[..., None] if j == 0 else core) for j, core in enumerate(cores))
    target = CombTarget(CombLayout(n), tensors, TCI_SOURCE)
    receipt = {key: value for key, value in raw.items() if key != "cores"}
    receipt.update({
        "bond_dimensions": [core.shape[2] for core in cores[:-1]],
        "stored_elements": sum(core.size for core in cores),
        "unnormalized_norm_squared": norm_squared.real,
        "vector_error": error,
        "target_infidelity": max(0.0, 1 - abs(overlap) ** 2),
        "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        "tensor_sha256": hashlib.sha256(b"".join(core.tobytes() for core in cores)).hexdigest(),
    })
    return target, receipt


def tci_target_bank(
    targets: Sequence[ArrayLike],
    *,
    gaussian_coordinates: Sequence[float] | None = None,
    options: TCIOptions = DEFAULT_TCI_OPTIONS,
) -> tuple[tuple[CombTarget, ...], dict[str, Any]]:
    """Build a bank of normalized single-tooth targets with published two-site TCI.

    Args:
        targets: Exact q2-q12 fixtures for independent validation. For generic
            targets the library queries their entries through an index oracle.
        gaussian_coordinates: Analytic Gaussian coordinates, when applicable;
            then no dense amplitudes are sent to the construction subprocess.
        options: Explicit rank, accuracy, seed and resource settings.

    Returns:
        Comb targets and provenance, sampling, accuracy, and timing receipts.
        CPU time includes child process startup and compilation, separately
        from the parent CPU. Wall time includes all validation work.

    Raises:
        ValueError: If inputs or target approximation accuracy are invalid.
        RuntimeError: If Julia/dependencies are missing or construction fails.
        TimeoutError: If the construction subprocess exceeds its time budget.
    """
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    if resource is None:
        msg = "TCI child-process CPU accounting requires Unix; legacy exact execution remains available."
        raise RuntimeError(msg)
    if not isinstance(options, TCIOptions) or not len(targets):
        msg = "Require TCIOptions and a nonempty target bank."
        raise ValueError(msg)
    references = [validate_target(value, max_qubits=12) for value in targets]
    n = references[0].size.bit_length() - 1
    if n < 2 or any(value.shape != references[0].shape for value in references):
        msg = "All TCI targets must share a q2-q12 register."
        raise ValueError(msg)
    if gaussian_coordinates is not None and len(gaussian_coordinates) != len(references):
        msg = "Require one Gaussian coordinate per target."
        raise ValueError(msg)
    specs = []
    for j, value in enumerate(references):
        if gaussian_coordinates is None:
            pivot = int(np.argmax(abs(value)))
            specs.append({
                "kind": "table",
                "real": value.real.tolist(),
                "imag": value.imag.tolist(),
                "initial_pivot": [1 + ((pivot >> site) & 1) for site in range(n)],
            })
        else:
            coordinate = gaussian_coordinates[j]
            if isinstance(coordinate, bool) or not isinstance(coordinate, (float, int)) or not 0 <= coordinate <= 1:
                msg = "Gaussian coordinates must be finite and in [0,1]."
                raise ValueError(msg)
            if not np.allclose(value, gaussian_state(n, coordinate), atol=1e-14, rtol=1e-13):
                msg = "Gaussian oracle does not match the supplied fixture."
                raise ValueError(msg)
            specs.append({"kind": "gaussian", "coordinate": coordinate})
    executable = shutil.which("julia")
    if executable is None:
        msg = "TCI requires Julia and the pinned tci_julia environment; see IQSP_TCI.md."
        raise RuntimeError(msg)
    request = json.dumps({"qubits": n, "targets": specs, "options": asdict(options)}, allow_nan=False)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        # Fixed local script and argument list; target data travels only on stdin.
        result = subprocess.run(  # noqa: S603
            [
                executable,
                "--startup-file=no",
                "--threads=1",
                f"--project={TCI_PROJECT}",
                str(TCI_PROJECT / "targets.jl"),
            ],
            input=request,
            capture_output=True,
            text=True,
            check=False,
            timeout=options.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        msg = "TCI construction exceeded its declared time budget."
        raise TimeoutError(msg) from exc
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
    if result.returncode:
        msg = f"TCI construction failed; instantiate tci_julia first. {result.stderr[-2000:]}"
        raise RuntimeError(msg)
    data = json.loads(result.stdout)
    if data["schema"] != 1 or data["library_version"] != "0.9.19" or len(data["targets"]) != len(references):
        msg = "TCI bridge schema, version, or target count mismatch."
        raise RuntimeError(msg)
    decoded = [_decode_target(raw, ref, options) for raw, ref in zip(data["targets"], references, strict=True)]
    receipt = {
        "algorithm": TCI_SOURCE,
        "settings": asdict(options),
        "settings_provenance": "explicit benchmark choices; not unpublished Ballarin settings",
        "julia_version": data["julia_version"],
        "oracle": "analytic_gaussian" if gaussian_coordinates is not None else "dense_fixture_index_oracle",
        "pivotsearch": "full",
        "sweepstrategy": "backandforth",
        "normalizeerror": True,
        "library_other_options": "pinned library defaults",
        "sources": {
            name: hashlib.sha256((TCI_PROJECT / name).read_bytes()).hexdigest()
            for name in ("targets.jl", "Project.toml", "Manifest.toml")
        },
        "targets": [item[1] for item in decoded],
        "child_cpu_seconds": child_cpu,
        "cpu_seconds": time.process_time() - started_cpu + child_cpu,
        "wall_seconds": time.perf_counter() - started_wall,
    }
    return tuple(item[0] for item in decoded), receipt
