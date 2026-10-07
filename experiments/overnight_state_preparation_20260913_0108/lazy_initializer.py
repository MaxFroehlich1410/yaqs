# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Probe-first conditional preparation with a genuinely lazy legacy constructor."""

from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import asdict
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import (
    compilation,
    confirmation_construction,
    construction,
    initializer_policy,
    runner,
    synthesis,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

VERSION = "overnight.lazy_conditional_initializer.v1"


def constructor_identity(phase: str) -> dict[str, Any] | None:
    """Return the exact scientific fallback recipe, leaving synthetic test receipts explicit."""
    if phase == "development":
        return {"version": construction.VERSION, "config": asdict(construction.Config())}
    if phase == "confirmation":
        return confirmation_construction.constructor_spec()
    return None


def read(path: Path) -> dict[str, Any]:
    """Read a checked heterogeneous research record."""
    return cast("dict[str, Any]", pilot.read_record(path))


def sources() -> dict[str, str]:
    """Freeze the unchanged numerical kernels and this new orchestration."""
    result = confirmation_construction.sources()
    for path in (
        Path(__file__),
        Path(__file__).with_name("LAZY_INITIALIZER_PROTOCOL.md"),
        Path(initializer_policy.__file__),
        Path(synthesis.__file__),
        Path(compilation.__file__),
    ):
        result[str(path.relative_to(runner.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def specification() -> dict[str, Any]:
    """Return the exact initializer object required by a future confirmation parent."""
    return {
        "version": VERSION,
        "threshold": initializer_policy.OVERLAP_THRESHOLD,
        "compiler": compilation.VERSION,
        "probe": "same deterministic physical synthesis, overlap and unitary compiler as initializer_policy",
        "legacy": "called only when ideal TT overlap is below 0.99",
        "sources": sources(),
    }


def validate(
    target: NDArray[np.complex128], target_id: str, replica: int, phase: str, parent: dict[str, Any] | None
) -> NDArray[np.complex128]:
    """Require a supplied normalized case; never load or infer a target vector."""
    vector = np.asarray(target, dtype=np.complex128)
    size = vector.size
    if (
        vector.ndim != 1
        or size < 2
        or size > 64
        or size & (size - 1)
        or not np.all(np.isfinite(vector))
        or abs(np.linalg.norm(vector) - 1) > 1e-12
        or type(replica) is not int
        or replica < 0
        or phase not in {"engineering", "development", "confirmation"}
        or (phase == "engineering" and (size > 16 or not target_id.startswith("synthetic_") or parent is not None))
        or (
            phase == "development"
            and (size != 64 or target_id not in construction.TARGETS or replica != 11 or parent is not None)
        )
    ):
        msg = "Lazy initialization requires normalized fixed development or small synthetic inputs."
        raise ValueError(msg)
    if phase == "confirmation":
        if parent is None:
            msg = "Confirmation lazy initialization requires a complete frozen parent."
            raise ValueError(msg)
        confirmation_construction.validate(vector, target_id, replica, construction.Config(), parent)
    return vector.copy()


def parent_binding(binding: dict[str, Any] | None) -> dict[str, Any] | None:
    """Require the entire complete parent to bind the exact tested lazy specification."""
    if binding is None:
        return None
    parent = confirmation_construction.parent_protocol(Path(binding["path"]), binding["checksum"])
    if parent["algorithm"]["initializer"] != specification() or any(
        parent["source_hashes"].get(k) != v for k, v in sources().items()
    ):
        msg = "Confirmation parent does not bind this exact lazy initializer."
        raise ValueError(msg)
    return parent


def prepare(
    root: Path,
    target: NDArray[np.complex128],
    target_id: str,
    replica: int,
    *,
    phase: str,
    constructor: dict[str, Any],
    parent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze one isolated case before synthesis or constructor calls."""
    if phase == "confirmation" and parent is None:
        msg = "Freeze the complete parent before confirmation initialization."
        raise ValueError(msg)
    vector = validate(target, target_id, replica, phase, parent_binding(parent))
    expected = constructor_identity(phase)
    if not constructor or (expected is not None and constructor != expected):
        msg = "The callback constructor must have the exact frozen configuration identity."
        raise ValueError(msg)
    manifest = {
        "version": VERSION,
        "phase": phase,
        "target_id": target_id,
        "replica": replica,
        "target_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
        "constructor": constructor,
        "parent": parent,
        "specification": specification(),
        "environment": construction.environment(),
        "benchmark_hashes": runner.custody()["benchmark_hashes"],
    }
    path = root / "PROTOCOL.json"
    if path.exists() and pilot.digest(read(path)) != pilot.digest(manifest):
        msg = "Lazy initializer case changed; use a new root."
        raise ValueError(msg)
    if not path.exists():
        if root.exists() and any(root.iterdir()):
            msg = "A lazy initializer cannot adopt preexisting constructor work."
            raise ValueError(msg)
        pilot.write_record(path, manifest)
    return read(path)


def verify(manifest: dict[str, Any], target: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Authenticate scientific scope, parent, numerical sources and supplied bytes."""
    if (
        manifest["version"] != VERSION
        or manifest["specification"] != specification()
        or manifest["environment"] != construction.environment()
        or manifest["benchmark_hashes"] != runner.custody()["benchmark_hashes"]
        or (
            constructor_identity(manifest["phase"]) is not None
            and manifest["constructor"] != constructor_identity(manifest["phase"])
        )
    ):
        msg = "Lazy initializer source or benchmark custody changed."
        raise ValueError(msg)
    vector = validate(
        target, manifest["target_id"], manifest["replica"], manifest["phase"], parent_binding(manifest["parent"])
    )
    if hashlib.sha256(vector.tobytes()).hexdigest() != manifest["target_sha256"]:
        msg = "Lazy initializer target bytes differ from its frozen case."
        raise ValueError(msg)
    return vector


def probe(target: NDArray[np.complex128]) -> dict[str, Any]:
    """Perform exactly the existing target-only physical rank-two probe and compiler."""
    cpu, wall = time.process_time(), time.perf_counter()
    rank_two = synthesis.synthesize(target)
    approximation = np.asarray(list(starmap(complex, rank_two["approximation"])))
    overlap = float(abs(np.vdot(target, approximation)) ** 2)
    physical = initializer_policy.state_vector(rank_two["circuit"], np.asarray(rank_two["theta"]))
    lowering_overlap = float(abs(np.vdot(approximation, physical)) ** 2)
    if (
        abs(overlap - rank_two["compression"]["approximation_fidelity"]) > 1e-10
        or not np.isfinite(lowering_overlap)
        or abs(lowering_overlap - 1) > 1e-10
    ):
        msg = "Lazy rank-two overlap or physical lowering audit disagrees."
        raise FloatingPointError(msg)
    chosen = initializer_policy.use_rank_two(overlap)
    compiled = initializer_policy.compile_source(rank_two)
    return {
        "rank_two_selected": chosen,
        "ideal_tt_overlap": overlap,
        "rank_two_physical_overlap": lowering_overlap,
        "synthesis": rank_two,
        "compilation": compiled,
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall,
    }


def dependencies(root: Path) -> dict[str, str]:
    """Snapshot every durable numerical receipt without copying nested histories."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.json"))
        if p not in {root / "RESULT.json", root / "ATTEMPTS.json"}
    }


def receipt(root: Path, supplied: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Require a complete fresh/resumed constructor result inside the owned child root."""
    result = read(root / "RESULT.json")
    protocol, ledger = read(root / "PROTOCOL.json"), read(root / "ATTEMPTS.json")
    if (
        pilot.digest(result) != pilot.digest(supplied)
        or result["status"] != "complete"
        or result["signature"] != pilot.digest(protocol)
        or ledger["signature"] != result["signature"]
        or result["attempts_checksum"] != pilot.digest(ledger)
        or not result["timing_complete"]
        or result["target_id"] != manifest["target_id"]
        or result["replica"] != manifest["replica"]
        or protocol["target_vector_sha256"] != manifest["target_sha256"]
        or ("config" in manifest["constructor"] and protocol["config"] != manifest["constructor"]["config"])
        or any(
            result.get(k) is None or not np.isfinite(result[k]) or result[k] < 0
            for k in ("construction_cpu_seconds", "construction_wall_seconds")
        )
        or any(
            a["status"] not in {"complete", "interrupted", "failed"}
            or any(a.get(k) is None or not np.isfinite(a[k]) or a[k] < 0 for k in ("cpu_seconds", "wall_seconds"))
            for a in ledger["attempts"]
        )
        or not ledger["attempts"]
        or ledger["attempts"][-1]["status"] != "complete"
        or any(
            result[f"construction_{kind}_seconds"] != sum(a[f"{kind}_seconds"] for a in ledger["attempts"])
            for kind in ("cpu", "wall")
        )
        or any(
            pilot.digest(read(root / "blocks" / f"{name}.json")) != block["checksum"]
            for name, block in result["blocks"].items()
        )
    ):
        msg = "Lazy fallback requires an authenticated complete constructor receipt in its owned root."
        raise ValueError(msg)
    return result


def execute(
    root: Path,
    manifest: dict[str, Any],
    vector: NDArray[np.complex128],
    construct: Callable[[Path], dict[str, Any]],
    ledger: dict[str, Any],
) -> dict[str, Any]:
    """Execute the two conditional branches with the same fixed numerical primitives."""
    signature = pilot.digest(manifest)
    ledger_path, attempt = root / "ATTEMPTS.json", ledger["attempts"][-1]
    probe_path = root / "PROBE.json"
    if not probe_path.exists():
        pilot.write_record(probe_path, probe(vector))
    tested = read(probe_path)
    legacy = None
    if tested["rank_two_selected"]:
        compiled = tested["compilation"]
    else:
        attempt["constructor_called"] = True
        pilot.write_record(ledger_path, ledger)
        legacy = receipt(root / "legacy", construct(root / "legacy"), manifest)
        if legacy["circuit"]["num_qubits"] != vector.size.bit_length() - 1:
            msg = "Lazy fallback constructor returned another register size."
            raise ValueError(msg)
        compiled_path = root / "LEGACY_COMPILED.json"
        if not compiled_path.exists():
            pilot.write_record(compiled_path, initializer_policy.compile_source(legacy))
        compiled = read(compiled_path)
    return assemble(signature, tested, compiled, legacy)


def assemble(
    signature: str, tested: dict[str, Any], compiled: dict[str, Any], legacy: dict[str, Any] | None
) -> dict[str, Any]:
    """Build the exact endpoint schema only from authenticated immutable stage records."""
    return {
        "version": VERSION,
        "signature": signature,
        "status": "complete",
        "rank_two_selected": tested["rank_two_selected"],
        "ideal_tt_overlap": tested["ideal_tt_overlap"],
        "rank_two_physical_overlap": tested["rank_two_physical_overlap"],
        "endpoint": {
            "circuit": copy.deepcopy(compiled["circuit"]),
            "theta": list(compiled["theta"]),
            "selected_source": "rank_two" if tested["rank_two_selected"] else "legacy",
            "compilation_checksum": pilot.digest(compiled),
            "physical_counts": runner.counts(pilot.restore_circuit(compiled["circuit"])),
        },
        "probe_checksum": pilot.digest(tested),
        "probe_cpu_seconds": tested["cpu_seconds"],
        "discarded_probe_cpu_seconds": 0.0 if tested["rank_two_selected"] else tested["cpu_seconds"],
        "legacy_receipt_checksum": None if legacy is None else pilot.digest(legacy),
        "nested_legacy_cpu_seconds": None if legacy is None else legacy["construction_cpu_seconds"],
        "cost_scope": (
            "inclusive actual lazy initialization, fallback, copies, compilation, audits and all caught replay; "
            "nested constructor CPU is diagnostic and is not added again"
        ),
    }


def complete(result: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    """Attach only the complete inclusive ledger totals to a reconstructed endpoint."""
    result["initializer_cpu_seconds"] = sum(a["cpu_seconds"] for a in ledger["attempts"])
    result["initializer_wall_seconds"] = sum(a["wall_seconds"] for a in ledger["attempts"])
    result["endpoint"]["initializer_cpu_seconds"] = result["initializer_cpu_seconds"]
    result["endpoint"]["timing_complete"] = result["timing_complete"] = True
    result["attempts_checksum"] = pilot.digest(ledger)
    return result


def cached_result(root: Path, manifest: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    """Reject edits to any final field by rebuilding it from the authenticated stage inputs."""
    tested = read(root / "PROBE.json")
    legacy = None
    if tested["rank_two_selected"]:
        compiled = tested["compilation"]
    else:
        legacy = receipt(root / "legacy", read(root / "legacy/RESULT.json"), manifest)
        compiled = read(root / "LEGACY_COMPILED.json")
    expected = complete(assemble(pilot.digest(manifest), tested, compiled, legacy), ledger)
    result = read(root / "RESULT.json")
    if pilot.digest(result) != pilot.digest(expected):
        msg = "Completed lazy initializer endpoint, branch or complete cost differs from its authenticated stages."
        raise ValueError(msg)
    return result


def run(
    root: Path,
    manifest: dict[str, Any],
    target: NDArray[np.complex128],
    construct: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    """Resume the actual lazy algorithm and charge its inclusive attempts exactly once."""
    cpu, wall = time.process_time(), time.perf_counter()
    vector = verify(manifest, target)
    signature = pilot.digest(manifest)
    if pilot.digest(read(root / "PROTOCOL.json")) != signature:
        msg = "Lazy initializer invocation differs from its stored case."
        raise ValueError(msg)
    ledger_path, result_path = root / "ATTEMPTS.json", root / "RESULT.json"
    ledger: dict[str, Any] = read(ledger_path) if ledger_path.exists() else {"signature": signature, "attempts": []}
    if ledger["signature"] != signature or any(
        a["status"] not in {"complete", "interrupted", "failed"}
        or any(a.get(k) is None or not np.isfinite(a[k]) or a[k] < 0 for k in ("cpu_seconds", "wall_seconds"))
        for a in ledger["attempts"]
    ):
        msg = "Unknown lazy initializer cost or changed attempt identity blocks resume."
        raise ValueError(msg)
    snapshot = dependencies(root)
    if ledger["attempts"]:
        if ledger["attempts"][-1]["dependencies"] != snapshot:
            msg = "Lazy initializer stage or constructor checkpoint changed."
            raise ValueError(msg)
    elif set(snapshot) != {"PROTOCOL.json"} or result_path.exists():
        msg = "Lazy initializer cannot adopt work without a measured attempt ledger."
        raise ValueError(msg)
    if result_path.exists():
        return cached_result(root, manifest, ledger)
    attempt: dict[str, Any] = {
        "status": "running",
        "cpu_seconds": None,
        "wall_seconds": None,
        "constructor_called": False,
    }
    ledger["attempts"].append(attempt)
    pilot.write_record(ledger_path, ledger)
    error: BaseException | None = None
    result: dict[str, Any] = {}
    try:
        result = execute(root, manifest, vector, construct, ledger)
        verify(manifest, target)
    except BaseException as caught:
        error = caught
    finally:
        try:
            attempt["dependencies"] = dependencies(root)
        except OSError as caught:
            attempt["dependencies"] = None
            error = caught if error is None else error
        attempt.update(
            status="complete" if error is None else "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            cpu_seconds=time.process_time() - cpu,
            wall_seconds=time.perf_counter() - wall,
        )
        if error is not None:
            attempt["failure"] = f"{type(error).__name__}: {error}"
        pilot.write_record(ledger_path, ledger)
    if error is not None:
        raise error
    result = complete(result, ledger)
    pilot.write_record(result_path, result)
    return read(result_path)
