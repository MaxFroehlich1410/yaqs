# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Parent-bound support for future fresh confirmation construction from zero.

This module neither loads targets nor offers a standalone confirmation CLI.
Its q6 entry point requires a complete immutable parent protocol. It directly
reuses the frozen legacy pipeline with the unchanged development recipe; only
the outer scientific identity and custody boundary are new.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import construction, runner

if TYPE_CHECKING:
    from numpy.typing import NDArray

VERSION = "overnight.parent_bound_confirmation_construction.v1"
PARENT_SCHEMA = "overnight.complete_confirmation_parent.v1"
TARGETS = ("tfim_ferro", "tfim_para", "haar_random_2", "haar_random_3", "random_mps_bond3")
ALGORITHM_FIELDS = ("initializer", "compiler", "training", "structure", "endpoint")
REPO = runner.REPO


def read(path: Path) -> dict[str, Any]:
    """Authenticate a heterogeneous checksummed research artifact."""
    return cast("dict[str, Any]", pilot.read_record(path))


def sources() -> dict[str, str]:
    """Freeze this boundary, its protocol and all unchanged numerical dependencies."""
    result = construction.sources()
    for path in (Path(__file__), Path(__file__).with_name("CONFIRMATION_CONSTRUCTION_PROTOCOL.md")):
        result[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def constructor_spec() -> dict[str, Any]:
    """Return the exact constructor section required in the eventual parent freeze."""
    return {
        "version": VERSION,
        "pipeline_version": construction.VERSION,
        "config": asdict(construction.Config()),
        "root_seed": 2026090520,
        "truncation": construction.TRUNCATION.copy(),
        "sources": sources(),
        "endpoint": {"updates": 570, "one_qubit": 78, "two_qubit": 30, "parameters": 108},
        "initialization": "legacy pilot initialization seed; physical zero input; H25_lr0p7:cold_final",
    }


def valid_hash(value: object) -> bool:
    """Require a canonical lowercase SHA-256, not a placeholder identity."""
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def nonempty_record(value: object) -> bool:
    """Check required structured sections without interpreting scientific results."""
    return isinstance(value, dict) and bool(value)


def parent_protocol(path: Path, expected: str) -> dict[str, Any]:
    """Authenticate an entire frozen confirmation design before accepting a q6 case."""
    parent = read(path)
    required = {
        "schema",
        "version",
        "status",
        "phase",
        "targets",
        "replicas",
        "cases",
        "algorithm",
        "arms",
        "budgets",
        "acceptance",
        "seed_policy",
        "constructor",
        "source_hashes",
        "benchmark_hashes",
        "environment",
    }
    if not valid_hash(expected) or pilot.digest(parent) != expected or not required <= parent.keys():
        msg = "A complete checksummed parent confirmation protocol is required."
        raise ValueError(msg)
    if (
        parent["schema"] != PARENT_SCHEMA
        or not isinstance(parent["version"], str)
        or not parent["version"].strip()
        or parent["status"] != "frozen"
        or parent["phase"] != "confirmation"
        or parent["targets"] != list(TARGETS)
        or not isinstance(parent["replicas"], list)
        or not parent["replicas"]
        or any(type(r) is not int or r not in {11, 29} for r in parent["replicas"])
        or sorted(set(parent["replicas"])) != parent["replicas"]
        or parent["constructor"] != constructor_spec()
        or parent["benchmark_hashes"] != runner.custody()["benchmark_hashes"]
        or parent["environment"] != construction.environment()
        or not nonempty_record(parent["algorithm"])
        or any(not nonempty_record(parent["algorithm"].get(field)) for field in ALGORITHM_FIELDS)
        or not nonempty_record(parent["seed_policy"])
        or not nonempty_record(parent["arms"])
        or len(parent["arms"]) < 2
        or not nonempty_record(parent["budgets"])
        or set(parent["arms"]) != set(parent["budgets"])
        or not nonempty_record(parent["acceptance"])
        or not nonempty_record(parent["source_hashes"])
    ):
        msg = "Parent confirmation phase, full algorithm, constructor or custody is incomplete."
        raise ValueError(msg)
    for name, arm in parent["arms"].items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not nonempty_record(arm)
            or not isinstance(arm.get("roles"), list)
            or not arm["roles"]
            or any(not isinstance(role, str) or not role.strip() for role in arm["roles"])
            or not nonempty_record(arm.get("configuration"))
            or not nonempty_record(parent["budgets"][name])
        ):
            msg = "Every confirmation arm needs explicit roles, configuration and budget."
            raise ValueError(msg)
    acceptance = parent["acceptance"]
    rule = acceptance.get("rule")
    if (
        not isinstance(acceptance.get("candidate"), str)
        or not isinstance(acceptance.get("primary_baseline"), str)
        or acceptance.get("candidate") not in parent["arms"]
        or acceptance.get("primary_baseline") not in parent["arms"]
        or acceptance["candidate"] == acceptance["primary_baseline"]
        or not isinstance(acceptance.get("primary_metric"), str)
        or not acceptance["primary_metric"].strip()
        or not nonempty_record(rule)
        or not isinstance(rule.get("route"), str)
        or rule.get("route") not in {"fidelity", "resource"}
        or not isinstance(rule.get("criteria"), list)
        or not rule["criteria"]
        or any(not isinstance(c, str) or not c.strip() for c in rule["criteria"])
        or rule.get("complete_panel_required") is not True
        or rule.get("failed_case_policy") != "retain_and_fail"
    ):
        msg = "Confirmation acceptance requires a distinct primary baseline and an explicit complete-panel rule."
        raise ValueError(msg)
    for source, expected_hash in parent["source_hashes"].items():
        source_path = REPO / source
        if (
            not isinstance(source, str)
            or Path(source).is_absolute()
            or not valid_hash(expected_hash)
            or not source_path.resolve().is_relative_to(REPO.resolve())
            or not source_path.is_file()
            or hashlib.sha256(source_path.read_bytes()).hexdigest() != expected_hash
        ):
            msg = "Frozen parent numerical source changed or is invalid."
            raise ValueError(msg)
    if any(parent["source_hashes"].get(name) != value for name, value in sources().items()):
        msg = "The parent source manifest does not bind the tested constructor."
        raise ValueError(msg)
    cases = parent["cases"]
    if (
        not isinstance(cases, list)
        or len(cases) != len(TARGETS) * len(parent["replicas"])
        or any(
            not isinstance(c, dict) or set(c) != {"target_id", "replica", "num_qubits", "target_vector_sha256"}
            for c in cases
        )
        or any(
            not isinstance(c["target_id"], str) or type(c["replica"]) is not int or type(c["num_qubits"]) is not int
            for c in cases
        )
        or {(c["target_id"], c["replica"]) for c in cases} != {(t, r) for t in TARGETS for r in parent["replicas"]}
        or any(c["num_qubits"] != 6 or not valid_hash(c["target_vector_sha256"]) for c in cases)
        or any(len({c["target_vector_sha256"] for c in cases if c["target_id"] == target}) != 1 for target in TARGETS)
    ):
        msg = "Parent confirmation case hashes must cover the entire fixed target/replica product."
        raise ValueError(msg)
    return parent


def validate(
    vector: NDArray[np.complex128],
    target_id: str,
    replica: int,
    config: construction.Config,
    parent: dict[str, Any] | None,
) -> NDArray[np.complex128]:
    """Copy a normalized supplied vector without reading any benchmark target."""
    if config.mode == "engineering":
        if parent is not None:
            msg = "Engineering construction cannot inherit a confirmation parent."
            raise ValueError(msg)
        return construction.validate_target(vector, target_id, replica, config)
    target = np.asarray(vector, dtype=np.complex128)
    if (
        parent is None
        or target.shape != (64,)
        or not np.all(np.isfinite(target))
        or abs(np.linalg.norm(target) - 1) > 1e-12
        or type(replica) is not int
        or target_id not in TARGETS
        or replica not in parent["replicas"]
        or not any(
            c["target_id"] == target_id
            and c["replica"] == replica
            and c["target_vector_sha256"] == hashlib.sha256(target.tobytes()).hexdigest()
            for c in parent["cases"]
        )
    ):
        msg = "Supplied q6 vector/identity differs from the frozen parent case."
        raise ValueError(msg)
    return target.copy()


def prepare(
    root: Path,
    vector: NDArray[np.complex128],
    target_id: str,
    replica: int,
    *,
    parent_path: Path | None = None,
    parent_checksum: str | None = None,
    config: construction.Config | None = None,
) -> dict[str, Any]:
    """Freeze one supplied case only after complete parent authorization, without simulation."""
    config = construction.Config() if config is None else config
    if config.mode == "development" and (parent_path is None or parent_checksum is None):
        msg = "A frozen complete parent protocol is required before any q6 preparation."
        raise ValueError(msg)
    if config.mode == "engineering" and (parent_path is not None or parent_checksum is not None):
        msg = "Engineering inputs cannot use a confirmation parent."
        raise ValueError(msg)
    parent = (
        parent_protocol(parent_path, parent_checksum)
        if parent_path is not None and parent_checksum is not None
        else None
    )
    target = validate(vector, target_id, replica, config, parent)
    manifest: dict[str, Any] = {
        "version": VERSION,
        "phase": "engineering" if parent is None else "confirmation",
        "config": asdict(config),
        "target_id": target_id,
        "replica": replica,
        "num_qubits": target.size.bit_length() - 1,
        "target_vector_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        "initialization_seed": construction.seed_for("initialization", target_id, replica),
        "root_seed": construction.ROOT_SEED,
        "sources": sources(),
        "environment": construction.environment(),
        "benchmark_hashes": runner.custody()["benchmark_hashes"],
        "truncation": construction.TRUNCATION.copy(),
        "archive_reference": None,
        "parent": None if parent_path is None else {"path": str(parent_path.resolve()), "checksum": parent_checksum},
    }
    path = root / "PROTOCOL.json"
    if path.exists() and pilot.digest(read(path)) != pilot.digest(manifest):
        msg = "Construction case changed; use a new output root."
        raise ValueError(msg)
    if not path.exists():
        if root.exists() and any(root.iterdir()):
            msg = "Fresh constructor cannot adopt an existing nonempty output root."
            raise ValueError(msg)
        pilot.write_record(path, manifest)
    return read(path)


def verify(
    manifest: dict[str, Any], vector: NDArray[np.complex128]
) -> tuple[NDArray[np.complex128], construction.Config]:
    """Revalidate parent, exact sources, recipe, seeds and supplied target on every resume."""
    config = construction.Config(**manifest["config"])
    if (
        manifest["version"] != VERSION
        or manifest["sources"] != sources()
        or manifest["environment"] != construction.environment()
        or manifest["benchmark_hashes"] != runner.custody()["benchmark_hashes"]
        or manifest["truncation"] != construction.TRUNCATION
        or manifest["root_seed"] != 2026090520
        or manifest["root_seed"] != construction.ROOT_SEED
        or manifest["archive_reference"] is not None
        or manifest["initialization_seed"]
        != construction.seed_for("initialization", manifest["target_id"], manifest["replica"])
        or manifest["phase"] != ("engineering" if config.mode == "engineering" else "confirmation")
        or (manifest["parent"] is None) != (config.mode == "engineering")
    ):
        msg = "Construction case, recipe or numerical custody changed."
        raise ValueError(msg)
    binding = manifest["parent"]
    parent = None if binding is None else parent_protocol(Path(binding["path"]), binding["checksum"])
    target = validate(vector, manifest["target_id"], manifest["replica"], config, parent)
    if (
        target.size != 2 ** manifest["num_qubits"]
        or hashlib.sha256(target.tobytes()).hexdigest() != manifest["target_vector_sha256"]
    ):
        msg = "Construction target bytes differ from its frozen case."
        raise ValueError(msg)
    return target, config


def block_hashes(root: Path) -> dict[str, str]:
    """Authenticate complete current chunk records without modifying legacy block files."""
    return {p.stem: pilot.digest(read(p)) for p in sorted((root / "blocks").glob("*.json"))}


def attempts(root: Path, signature: str) -> dict[str, Any]:
    """Fail closed on missing cost provenance, hard kills or changed chunk checkpoints."""
    path = root / "ATTEMPTS.json"
    blocks = block_hashes(root)
    if not path.exists():
        if blocks or (root / "RESULT.json").exists():
            msg = "Construction blocks or results exist without their physical cost ledger."
            raise ValueError(msg)
        return {"signature": signature, "attempts": []}
    record = read(path)
    if record["signature"] != signature or not record["attempts"]:
        msg = "Construction attempt ledger identity or history is incomplete."
        raise ValueError(msg)
    for attempt in record["attempts"]:
        if attempt["status"] not in {"complete", "interrupted", "failed"} or any(
            isinstance(attempt.get(k), bool)
            or not isinstance(attempt.get(k), (int, float))
            or not np.isfinite(attempt[k])
            or attempt[k] < 0
            for k in ("cpu_seconds", "wall_seconds")
        ):
            msg = "Unknown construction cost blocks resume; external accounting is required."
            raise ValueError(msg)
    if record["attempts"][-1].get("block_checksums") != blocks:
        msg = "Construction chunk files differ from the last measured attempt."
        raise ValueError(msg)
    return record


def run(root: Path, manifest: dict[str, Any], vector: NDArray[np.complex128]) -> dict[str, Any]:
    """Run or resume the unchanged constructor, with every physical attempt charged."""
    cpu, wall = time.process_time(), time.perf_counter()
    target, config = verify(manifest, vector)
    signature = pilot.digest(manifest)
    if pilot.digest(read(root / "PROTOCOL.json")) != signature:
        msg = "Construction invocation differs from its saved case protocol."
        raise ValueError(msg)
    ledger = attempts(root, signature)
    path = root / "RESULT.json"
    if path.exists():
        result = read(path)
        if (
            result["signature"] != signature
            or result["attempts_checksum"] != pilot.digest(ledger)
            or {name: block["checksum"] for name, block in result["blocks"].items()} != block_hashes(root)
        ):
            msg = "Completed constructor endpoint, blocks or cost ledger changed."
            raise ValueError(msg)
        return result
    attempt: dict[str, Any] = {
        "status": "running",
        "cpu_seconds": None,
        "wall_seconds": None,
        "block_checksums_before": block_hashes(root),
    }
    ledger["attempts"].append(attempt)
    pilot.write_record(root / "ATTEMPTS.json", ledger)
    error: BaseException | None = None
    result: dict[str, Any] = {}
    try:
        result = construction.run_pipeline(root, manifest, target, config)
        verify(manifest, vector)
    except BaseException as caught:
        error = caught
    finally:
        try:
            attempt["block_checksums"] = block_hashes(root)
        except (ValueError, OSError) as audit_error:
            attempt["block_audit_failure"] = f"{type(audit_error).__name__}: {audit_error}"
            if error is None:
                error = audit_error
        attempt.update(
            status="complete" if error is None else "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            cpu_seconds=time.process_time() - cpu,
            wall_seconds=time.perf_counter() - wall,
        )
        if error is not None:
            attempt["failure"] = f"{type(error).__name__}: {error}"
        pilot.write_record(root / "ATTEMPTS.json", ledger)
    if error is not None:
        raise error
    result.update({
        "wrapper_version": VERSION,
        "phase": manifest["phase"],
        "parent_checksum": None if manifest["parent"] is None else manifest["parent"]["checksum"],
        "construction_cpu_seconds": sum(a["cpu_seconds"] for a in ledger["attempts"]),
        "construction_wall_seconds": sum(a["wall_seconds"] for a in ledger["attempts"]),
        "timing_complete": True,
        "attempts_checksum": pilot.digest(ledger),
        "cost_scope": (
            "all successful and interrupted invocation work, target copies, custody, unchanged pipeline, "
            "pruning, chunk I/O and diagnostics; final result and terminal ledger serialization excluded"
        ),
    })
    pilot.write_record(path, result)
    return read(path)
