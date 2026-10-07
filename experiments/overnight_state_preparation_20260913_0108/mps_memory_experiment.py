# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Measure one retained forward tape at each of eight authenticated q6 development endpoints."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import resource
import sys
from dataclasses import asdict
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import (
    compilation_experiment,
    mps_memory,
    trajectory_mixture,
)
from experiments.overnight_state_preparation_20260913_0108 import mps_overlap_experiment as old
from experiments.overnight_state_preparation_20260913_0108 import mps_physical_experiment as physical
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KrotovTJMOptions,
    KrotovTruncation,
    StatevectorBackend,
    noisy_state_preparation_contribution,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import KrotovNoiseMap, ParameterizedCircuit

VERSION = "overnight.one_path_retained_mps_memory.v1"
TOLERANCE = 1e-10


def settings() -> dict[str, Any]:
    """Freeze one first path at unchanged source theta, with no optimizer update or path grid."""
    return {
        "update_index": 0,
        "path_index": 0,
        "paths_per_channel_for_identity": old.PATHS_PER_CHANNEL,
        "gradient_paths": 1,
        "diagnostic_path_weight": 1.0,
        "truncation": asdict(KrotovTruncation()),
        "tolerance": TOLERANCE,
        "seed_namespace": "mps_overlap",
        "sampler_namespace": "development_diagnostics",
        "target_source": "saved exact constructor vector and saved exact overlap TARGET_MPS",
        "scope": "one fixed path retained forward tape; not representative full weighted-batch or peak gradient memory",
        "optimizer_updates": 0,
    }


def sources() -> dict[str, str]:
    """Freeze only new standalone support plus the existing immutable numerical dependencies."""
    paths = [
        Path(__file__),
        Path(mps_memory.__file__),
        Path(__file__).with_name("MPS_MEMORY.md"),
        Path(__file__).with_name("MPS_MEMORY_PROTOCOL.md"),
    ]
    return {**physical.sources(), **{str(p.relative_to(old.runner.REPO)): old.file_hash(p) for p in paths}}


def authenticate_files(files: dict[str, str]) -> None:
    """Require exact source bytes without calling a target loader or numerical backend."""
    if not files or any(
        not Path(name).is_file() or old.file_hash(Path(name)) != value for name, value in files.items()
    ):
        msg = "Memory diagnostic input artifact custody changed."
        raise ValueError(msg)


def inputs(overlap_root: Path, physical_root: Path, target_root: Path) -> dict[str, Any]:
    """Read the complete old panel, saved pre-update theta and exact vector receipts without replay."""
    overlap, checked = old.read(overlap_root / "PROTOCOL.json"), old.read(physical_root / "PROTOCOL.json")
    old.verify(overlap)
    summaries = [old.read(root / "SUMMARY.json") for root in (overlap_root, physical_root, target_root)]
    batch = old.read(target_root / "BATCH_PROTOCOL.json")
    if (
        checked["version"] != physical.VERSION
        or checked["sources"] != physical.sources()
        or checked["settings"] != physical.settings()
        or checked["input_panel"]["protocol_checksum"] != pilot.digest(overlap)
        or checked["input_panel"]["cases"] != overlap["cases"]
        or summaries[0]["protocol_checksum"] != pilot.digest(overlap)
        or summaries[0]["accepted"] is not False
        or summaries[1]["protocol_checksum"] != pilot.digest(checked)
        or summaries[1]["accepted"] is not True
        or set(summaries[0]["cases"]) != set(old.TARGETS)
        or set(summaries[1]["cases"]) != set(old.TARGETS)
        or batch["version"] != "overnight.legacy_construction_development_batch.v1"
        or summaries[2]["status"] != "complete"
        or summaries[2]["protocol_checksum"] != pilot.digest(batch)
        or summaries[2]["all_archive_reproductions_match"] is not True
    ):
        msg = "Memory audit requires complete failed overlap, passed physical follow-up and exact constructor receipts."
        raise ValueError(msg)
    files = {
        str(root / name): old.file_hash(root / name)
        for root, names in (
            (overlap_root, ("PROTOCOL.json", "SUMMARY.json")),
            (physical_root, ("PROTOCOL.json", "SUMMARY.json")),
            (target_root, ("BATCH_PROTOCOL.json", "SUMMARY.json")),
        )
        for name in names
    }
    files.update(checked["input_panel"]["files"])
    entries = []
    for case in overlap["cases"]:
        target_id = case["target_id"]
        prior = overlap_root / f"{target_id}__11"
        followup = physical_root / f"{target_id}__11"
        old_case, new_case = old.read(prior / "RESULT.json"), old.read(followup / "RESULT.json")
        for root, result, summary in ((prior, old_case, summaries[0]), (followup, new_case, summaries[1])):
            if result["status"] != "complete" or summary["cases"][target_id]["checksum"] != pilot.digest(result):
                msg = "A required memory source case is incomplete or changed."
                raise ValueError(msg)
            files[str(root / "RESULT.json")] = old.file_hash(root / "RESULT.json")
            files.update({str(root / name): value for name, value in result["dependencies"].items()})
        vector_path = target_root / f"{target_id}__11/PROTOCOL.json"
        vector = old.read(vector_path)
        source = next(row for row in batch["cases"] if row["target_id"] == target_id and row["replica"] == 11)
        saved_target = old.read(prior / "TARGET_MPS.json")
        if (
            pilot.digest(vector) != source["manifest_checksum"]
            or vector["target_id"] != target_id
            or vector["replica"] != 11
            or vector["target_vector_sha256"] != saved_target["target_sha256"]
        ):
            msg = "The saved exact dense target and overlap MPS do not have the same authenticated identity."
            raise ValueError(msg)
        files[str(vector_path)] = old.file_hash(vector_path)
        for label in old.ENDPOINTS:
            endpoint = case["endpoints"][label]
            step = old.read(prior / label / "STATE.json")["steps"][0]
            first = new_case["endpoints"][label]["steps"][0]
            raw_path = followup / label / first["raw_file"]
            if old.file_hash(raw_path) != first["raw_sha256"]:
                msg = "The saved physical first-step raw record changed."
                raise ValueError(msg)
            raw = json.loads(gzip.decompress(raw_path.read_bytes()))
            if step["update_index"] != 0 or any(
                raw["replay"]["before"][name]["theta"] != endpoint["theta"] for name in ("dense", "mps")
            ):
                msg = "The selected theta must be the unchanged first recorded pre-update theta."
                raise ValueError(msg)
            entries.append({
                "target_id": target_id,
                "replica": 11,
                "label": label,
                "endpoint": endpoint,
                "first_step": {key: step[key] for key in ("map_sha256", "weights", "update_index")},
                "target_vector_file": str(vector_path),
                "target_mps_file": str(prior / "TARGET_MPS.json"),
                "preupdate_raw_sha256": first["raw_sha256"],
            })
    if len(entries) != 8 or {(e["target_id"], e["label"]) for e in entries} != {
        (t, label) for t in old.TARGETS for label in old.ENDPOINTS
    }:
        msg = "The memory diagnostic requires all eight fixed endpoints."
        raise ValueError(msg)
    authenticate_files(files)
    return {
        "overlap_root": str(overlap_root),
        "physical_root": str(physical_root),
        "target_root": str(target_root),
        "entries": entries,
        "files": files,
    }


def prepare(root: Path, overlap_root: Path, physical_root: Path, target_root: Path) -> dict[str, Any]:
    """Freeze all eight read-only source bindings before any diagnostic gradient call."""
    if root.resolve() in {path.resolve() for path in (overlap_root, physical_root, target_root)}:
        msg = "Memory output must be separate from every immutable source root."
        raise ValueError(msg)
    protocol = {
        "version": VERSION,
        "phase": "development_diagnostic",
        "settings": settings(),
        "sources": sources(),
        "environment": old.construction.environment(),
        "benchmark_hashes": old.runner.custody()["benchmark_hashes"],
        "input_panel": inputs(overlap_root.resolve(), physical_root.resolve(), target_root.resolve()),
    }
    path = root / "PROTOCOL.json"
    if path.exists() and old.read(path) != protocol:
        msg = "Frozen memory diagnostic changed; use a new output root."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return old.read(path)


def verify(protocol: dict[str, Any]) -> None:
    """Authenticate protocol fields and all frozen raw bytes without replaying numerical work."""
    panel = protocol["input_panel"]
    if (
        protocol["version"] != VERSION
        or protocol["settings"] != settings()
        or protocol["sources"] != sources()
        or protocol["phase"] != "development_diagnostic"
        or protocol["environment"] != old.construction.environment()
        or protocol["benchmark_hashes"] != old.runner.custody()["benchmark_hashes"]
        or panel != inputs(Path(panel["overlap_root"]), Path(panel["physical_root"]), Path(panel["target_root"]))
    ):
        msg = "Memory protocol, complete panel or numerical source identity changed."
        raise ValueError(msg)


def peak_rss() -> dict[str, Any]:
    """Record worker-lifetime RSS with explicit platform units, never incremental tape memory."""
    if sys.platform not in {"darwin", "linux"}:
        msg = "Memory RSS normalization is defined only on macOS and Linux."
        raise RuntimeError(msg)
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "raw_value": raw,
        "raw_unit": "bytes" if sys.platform == "darwin" else "KiB",
        "bytes": int(raw * (1 if sys.platform == "darwin" else 1024)),
        "platform": sys.platform,
        "scope": (
            "worker lifetime high-water including inputs, dense oracle and prior endpoints; "
            "not incremental or peak algorithm memory"
        ),
    }


def measure(
    circuit: ParameterizedCircuit,
    theta: NDArray,
    target: NDArray,
    tensor_target: MPS,
    path: Sequence[KrotovNoiseMap],
    attempt: old.Attempt,
) -> dict[str, Any]:
    """Execute one unchanged MPS path and independent dense gradient, auditing storage afterward."""
    if (
        theta.shape != (circuit.num_params,)
        or target.shape != (2**circuit.num_qubits,)
        or not np.all(np.isfinite(theta))
        or not np.all(np.isfinite(target))
        or abs(np.linalg.norm(target) - 1) > 1e-12
        or len(path) != len(circuit.gates)
    ):
        msg = "Memory measurement requires finite compatible input arrays and a complete fixed path."
        raise ValueError(msg)
    target_error = attempt.call("saved_target_equivalence", lambda: old.difference(tensor_target.to_vec(), target))
    gradient, loss, fidelity, trajectories = attempt.call(
        "mps_gradient",
        lambda: noisy_state_preparation_contribution(
            circuit,
            theta,
            tensor_target,
            None,
            KrotovTJMOptions(num_trajectories=1),
            MPS(circuit.num_qubits, state="zeros"),
            KrotovTruncation(),
            fixed_noise_maps=[list(path)],
        ),
    )
    memory = attempt.call("retained_memory_audit", lambda: mps_memory.forward_tape_storage(trajectories, tensor_target))
    engine = attempt.call("dense_setup", lambda: StatevectorBackend(circuit, target))
    dense, dense_loss, dense_fidelity = attempt.call("dense_gradient", lambda: engine.gradient(theta, [list(path)]))
    errors = {
        "gradient": old.difference(gradient, dense),
        "loss": abs(loss - dense_loss),
        "fidelity": abs(fidelity - dense_fidelity),
        "saved_target": target_error,
    }
    states = [state for trajectory in trajectories for state in (*trajectory.states, *trajectory.gate_outputs)]
    norm_error = attempt.call(
        "forward_norm_audit", lambda: max(abs(float(s.scalar_product(s).real) - 1) for s in states)
    )
    errors["forward_norm"] = norm_error
    return {
        "accepted": all(math.isfinite(value) and value <= TOLERANCE for value in errors.values())
        and memory["retained_union"]["complete_backing_payload_bytes"] is not None,
        "theta": theta.tolist(),
        "mps_gradient": gradient.tolist(),
        "dense_gradient": dense.tolist(),
        "mps_loss": loss,
        "dense_loss": dense_loss,
        "mps_fidelity": fidelity,
        "dense_fidelity": dense_fidelity,
        "errors": errors,
        "memory": memory,
        "peak_rss": peak_rss(),
        "forward_max_bond": max(max(old.mps_bridge.bonds(s), default=1) for s in states),
        "target_bonds": old.mps_bridge.bonds(tensor_target),
        "dense_oracle_target_bytes": int(target.nbytes),
        "dense_oracle_target_scope": "separate validation vector; excluded from the MPS tape-plus-target union",
        "truncation": asdict(KrotovTruncation()),
        "operator_count": sum(len(noise.operators) for noise in path),
        "analytic_no_factor_path": all(not noise.operators for noise in path),
        "work_counts": {
            "mps_gradient_paths": 1,
            "dense_gradient_paths": 1,
            "direction_passes": 4,
            "optimizer_updates": 0,
            "norm_audit_forward_states": len(states),
        },
        "scope": settings()["scope"],
        "prior_overlap_gate": "failed; unchanged",
    }


def compute(entry: dict[str, Any], attempt: old.Attempt) -> dict[str, Any]:
    """Restore existing inputs and authenticate full old batch identity before selecting path zero."""
    vector = attempt.call("saved_target_input", lambda: old.read(Path(entry["target_vector_file"])))
    target = np.asarray(list(starmap(complex, vector["target"])), dtype=np.complex128)
    saved = attempt.call("saved_target_mps_input", lambda: old.read(Path(entry["target_mps_file"])))
    if (
        hashlib.sha256(target.tobytes()).hexdigest() != vector["target_vector_sha256"]
        or vector["target_vector_sha256"] != saved["target_sha256"]
    ):
        msg = "Saved dense/MPS target input bytes changed."
        raise ValueError(msg)
    tensors = [np.asarray(t["real"]) + 1j * np.asarray(t["imag"]) for t in saved["tensors"]]
    tensor_target = MPS(len(tensors), tensors=tensors)
    circuit = attempt.call(
        "circuit_restore", lambda: compilation_experiment.restore_source(entry["endpoint"]["circuit"])
    )
    sampler = attempt.call("sampler_setup", lambda: trajectory_mixture.MixtureSampler(circuit))
    batch = attempt.call(
        "map_identity_regeneration",
        lambda: sampler.ensemble(
            old.PATHS_PER_CHANNEL,
            seed=old.runner.seed_for("mps_overlap", entry["target_id"], 11),
            update_index=0,
            namespace="development_diagnostics",
        ),
    )
    physical.authenticate_batch(batch, entry["first_step"])
    result = measure(circuit, np.asarray(entry["endpoint"]["theta"]), target, tensor_target, batch.maps[0], attempt)
    return {
        **result,
        "full_source_batch_map_sha256": old.maps_hash(batch),
        "selected_path_index": 0,
        "source_path_weight": float(batch.weights[0]),
        "diagnostic_path_weight": 1.0,
    }


def registry(root: Path) -> dict[str, str]:
    """Bind every raw diagnostic artifact apart from its public result and completion seal."""
    excluded = {root / "RESULT.json", root / "COMPLETED.json"}
    return {
        str(path.relative_to(root)): old.file_hash(path)
        for path in root.rglob("*")
        if path.is_file() and path not in excluded
    }


def assembled(root: Path, signature: str, measurement: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct every public acceptance, cost and measurement field from durable child records."""
    return {
        "version": VERSION,
        "signature": signature,
        "status": "complete",
        "accepted": measurement["accepted"],
        "measurement": measurement,
        "measurement_checksum": pilot.digest(measurement),
        "costs": old.costs(ledger),
        "dependencies": registry(root),
    }


def run_endpoint(root: Path, identity: object, function: Callable[[old.Attempt], dict[str, Any]]) -> dict[str, Any]:
    """Retain every attempt, reuse complete results read-only and block unknown hard-kill costs."""
    signature = pilot.digest([VERSION, identity])
    path = root / "RESULT.json"
    ledger_path = root / "ATTEMPTS.json"
    if ledger_path.exists():
        ledger = old.read(ledger_path)
        if ledger["signature"] != signature or any(
            a["status"] == "running" or any(a.get(k) is None for k in ("cpu_seconds", "wall_seconds"))
            for a in ledger["attempts"]
        ):
            msg = "Unknown memory diagnostic attempt cost or identity blocks resume."
            raise ValueError(msg)
    if path.exists():
        result, sealed = old.read(path), old.read(root / "COMPLETED.json")
        raw = old.read(root / "MEASUREMENT.json")
        if result != assembled(root, signature, raw, old.read(ledger_path)) or sealed != {
            "signature": signature,
            "result_checksum": pilot.digest(result),
        }:
            msg = "Memory result, raw measurements, costs or complete dependency registry changed."
            raise ValueError(msg)
        return result
    attempt = physical.open_attempt(root, signature)
    error: BaseException | None = None
    measurement: dict[str, Any] = {}
    try:
        measurement = function(attempt)
        pilot.write_record(root / "MEASUREMENT.json", measurement)
    except BaseException as caught:
        error = caught
    finally:
        attempt.finish(error)
    if error is not None:
        raise error
    result = assembled(root, signature, measurement, attempt.record)
    pilot.write_record(root / "COMPLETED.json", {"signature": signature, "result_checksum": pilot.digest(result)})
    pilot.write_record(path, result)
    return old.read(path)


def summarize(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Keep every expected endpoint visible and refuse an incomplete all-eight agreement gate."""
    rows = []
    for entry in protocol["input_panel"]["entries"]:
        child = root / f"{entry['target_id']}__11" / entry["label"]

        def forbidden(attempt: old.Attempt) -> dict[str, Any]:
            del attempt
            msg = "Read-only memory summary cannot execute a missing diagnostic."
            raise ValueError(msg)

        result = run_endpoint(child, [protocol, entry], forbidden) if (child / "RESULT.json").exists() else None
        ledger = old.read(child / "ATTEMPTS.json") if (child / "ATTEMPTS.json").exists() else None
        rows.append({
            "target_id": entry["target_id"],
            "replica": 11,
            "label": entry["label"],
            "status": "pending_or_failed" if result is None else result["status"],
            "accepted": result is not None and result["accepted"],
            "checksum": None if result is None else pilot.digest(result),
            "costs": None if ledger is None else old.costs(ledger),
        })
    summary = {
        "version": VERSION,
        "protocol_checksum": pilot.digest(protocol),
        "cases": rows,
        "status": "complete" if all(r["status"] == "complete" for r in rows) else "incomplete_or_failed",
        "accepted": len(rows) == 8 and all(r["accepted"] for r in rows),
        "scope": settings()["scope"],
        "prior_overlap_gate": "failed; unchanged",
    }
    pilot.write_record(root / "SUMMARY.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare or execute the complete fixed diagnostic with one explicitly single-thread worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=old.runner.OUTPUT / "mps-retained-memory")
    parser.add_argument("--overlap-root", type=Path, default=old.runner.OUTPUT / "mps-overlap")
    parser.add_argument("--physical-root", type=Path, default=old.runner.OUTPUT / "mps-physical-followup")
    parser.add_argument("--target-root", type=Path, default=old.runner.OUTPUT / "construction-reproduction")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-only", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if any(
        os.environ.get(key) != "1"
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    ):
        parser.error("Set all four native thread limits to1; this diagnostic uses one worker.")
    if args.prepare_only:
        prepare(args.output, args.overlap_root, args.physical_root, args.target_root)
        return 0
    protocol = old.read(args.output / "PROTOCOL.json")
    verify(protocol)
    for entry in protocol["input_panel"]["entries"]:
        child = args.output / f"{entry['target_id']}__11" / entry["label"]
        try:
            run_endpoint(child, [protocol, entry], lambda attempt, entry=entry: compute(entry, attempt))
        except (ArithmeticError, RuntimeError, MemoryError) as error:
            pilot.write_record(child / "FAILURE.json", {"failure": f"{type(error).__name__}: {error}"})
        summarize(args.output, protocol)
    verify(protocol)
    result = summarize(args.output, protocol)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
