# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Frozen eleven-group q12 comparison, with independent competitor depth jobs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np

from benchmarks.state_preparation.constants import TARGET_IDS
from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH, load_target
from experiments import topdown_backend_support as checkpoints
from experiments import topdown_mps
from experiments.state_preparation_competitors import fontana, logical_evaluation, robuststate
from experiments.state_preparation_competitors.run import ROOT, digest, load_bundle, prepare_candidates, write_json

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from qiskit import QuantumCircuit

TARGETS = ("gaussian_mu0p5_sigma0p1", "tfim_critical", "random_mps_bond3", "haar_random_1")
PACKAGES = ("numpy", "scipy", "qiskit", "threadpoolctl")


def protocol() -> dict:
    """Return the agreed scientific settings, separate from execution limits."""
    return {
        "qubits": 12,
        "seed": 300,
        "hardware": "IBM",
        "noise_basis": "logical",
        "targets": list(TARGETS),
        "depths": [1, 2, 3, 4],
        "ours": asdict(topdown_mps.Config(qubits=12, max_depth=4)),
        "robuststate": {
            "noiseless_steps": 500,
            "noisy_steps": 50,
            "learning_rate": 0.005,
            "feedback": "pauli_shadows",
            "num_bases": 400,
            "shots_per_basis": 1024,
        },
        "fontana": {"construction_steps": 200, "noisy_steps": 200},
        "validation_paths": 32768,
        "final_batches": 4,
        "final_paths_per_batch": 8192,
        "excluded": ["ballarin", "nacl", "ideal_only_methods"],
        "measurement_scope": "400 bases is a resource-limited extrapolation, not sampling-converged",
    }


def jobs(settings: dict) -> list[dict]:
    """Expand independent jobs without breaking our sequential warm-start chain.

    Priority is a scheduling heuristic, not a measured runtime prediction.
    """
    result = []
    for target in settings["targets"]:
        result.append({
            "id": f"ours__{target}",
            "method": "ours",
            "target": target,
            "kind": "chain",
            "dependencies": [],
            "priority": 50,
        })
        for method in ("robuststate", "fontana"):
            if method == "fontana" and target == "haar_random_1":
                continue
            dependencies = []
            for depth in settings["depths"]:
                identity = f"{method}__{target}__d{depth}"
                dependencies.append(identity)
                result.append({
                    "id": identity,
                    "method": method,
                    "target": target,
                    "kind": "candidate",
                    "depth": depth,
                    "dependencies": [],
                    "priority": (20 if method == "fontana" else 10) * depth,
                })
            result.append({
                "id": f"{method}__{target}__final",
                "method": method,
                "target": target,
                "kind": "final",
                "dependencies": dependencies,
                "priority": 1000,
            })
    return result


def source_hashes() -> dict[str, str]:
    """Fingerprint the complete Python source snapshot and immutable fixture."""
    paths = {path for name in ("src", "experiments", "benchmarks") for path in (ROOT / name).rglob("*.py")}
    paths.update({DEFAULT_TARGET_PATH, ROOT / "pyproject.toml"})
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def authenticate(root: Path) -> dict:
    """Reject changed manifests, code, fixtures or numerical package versions."""
    path = root / "manifest.json"
    if digest(path) != (root / "manifest.sha256").read_text().strip():
        msg = "Campaign manifest checksum mismatch."
        raise ValueError(msg)
    manifest = json.loads(path.read_text())
    if manifest["sources"] != source_hashes():
        msg = "Campaign source/fixture snapshot mismatch."
        raise ValueError(msg)
    actual = {name: importlib.metadata.version(name) for name in PACKAGES}
    if manifest["packages"] != actual:
        msg = f"Campaign numerical package mismatch: {actual}"
        raise ValueError(msg)
    return manifest


def assessment_entropy(job: dict, seed: int, role: str, batch: int = 0) -> list[int]:
    """Derive host-independent streams with disjoint validation and test roles."""
    if role not in {"validation", "test"}:
        msg = "Unknown assessment seed role."
        raise ValueError(msg)
    payload = json.dumps([job["method"], job["target"], job.get("depth"), seed, role, batch]).encode()
    fingerprint = hashlib.sha256(payload).digest()
    return [20260928, *(int.from_bytes(fingerprint[i : i + 4], "little") for i in range(0, 32, 4))]


def assess(folder: Path, circuit: QuantumCircuit, target: NDArray, count: int, entropy: list[int]) -> dict:
    """Seal unconditional held-out histories and score without resampling jumps."""
    saved = checkpoints.load_checkpoint(folder)
    if saved is not None:
        if saved[0]["entropy"] != entropy or len(saved[1]["codes"]) != count:
            msg = "Assessment resume has different seeds or sample count."
            raise ValueError(msg)
        return saved[0]
    start = time.process_time()
    codes = logical_evaluation.sample_paths(circuit, "IBM", count, entropy)
    score = logical_evaluation.validation_from_paths(circuit, target, "IBM", codes)
    score.update({
        "entropy": entropy,
        "cpu_seconds": time.process_time() - start,
        "status": "complete" if not circuit.data or score["jump_paths"] else "no_jump_bank",
    })
    return checkpoints.save_checkpoint(folder, score, {"codes": codes})


def candidate(root: Path, job: dict, settings: dict, target: NDArray) -> dict:
    """Call the existing published-method adapter at one independent depth."""
    folder = root / "jobs" / job["id"]
    factory = robuststate.robuststate_candidates if job["method"] == "robuststate" else fontana.fontana_candidates
    candidates, receipt = prepare_candidates(
        folder / "training",
        lambda: factory(
            target,
            hardware="IBM",
            seed=settings["seed"],
            depths=(job["depth"],),
            noise_basis="logical",
            **settings[job["method"]],
        ),
    )
    circuit = candidates[0].circuit
    score = assess(
        folder / "validation",
        circuit,
        target,
        settings["validation_paths"],
        assessment_entropy(job, settings["seed"], "validation"),
    )
    return {
        "status": score["status"],
        "validation": score,
        "training_reference": f"jobs/{job['id']}/training",
        "training_seal_sha256": digest(folder / "training" / "seal.json"),
        "resources": logical_evaluation.logical_resources(circuit, target),
        "training_cpu_seconds": receipt["construction_cpu_seconds"],
    }


def finalize(root: Path, job: dict, settings: dict, target: NDArray) -> dict:
    """Lock validation-only selection before scoring fresh final trajectory batches."""
    folder = root / "jobs" / job["id"]
    records = []
    for dependency in job["dependencies"]:
        saved = checkpoints.load_checkpoint(root / "jobs" / dependency / "result")
        if saved is None:
            msg = "A candidate result is missing."
            raise ValueError(msg)
        record = saved[0]
        load_bundle(root / record["training_reference"])
        if digest(root / record["training_reference"] / "seal.json") != record["training_seal_sha256"]:
            msg = "Training seal differs from the candidate result."
            raise ValueError(msg)
        records.append((dependency, record))
    eligible = [(name, record) for name, record in records if record["status"] == "complete"]
    if not eligible:
        return {"status": "no_valid_candidate"}
    selected, best = min(
        eligible,
        key=lambda item: (-item[1]["validation"]["fidelity"], item[1]["resources"]["logical_two_qubit"], item[0]),
    )
    lock = {"selected_job": selected, "validation_records": [f"jobs/{name}/result/record.json" for name, _ in records]}
    path = folder / "selection_locked.json"
    if path.exists() and json.loads(path.read_text()) != lock:
        msg = "Validation selection differs from its lock."
        raise ValueError(msg)
    write_json(path, lock)
    _, circuits = load_bundle(root / best["training_reference"])
    batches = [
        assess(
            folder / f"test_{batch:02d}",
            circuits[0],
            target,
            settings["final_paths_per_batch"],
            assessment_entropy(job, settings["seed"], "test", batch),
        )
        for batch in range(settings["final_batches"])
    ]
    fidelity = float(np.mean([batch["fidelity"] for batch in batches]))
    se = float(np.sqrt(sum(batch["standard_error"] ** 2 for batch in batches)) / len(batches))
    return {
        "status": "complete" if all(b["status"] == "complete" for b in batches) else "test_jump_coverage_failure",
        **lock,
        "selection_sha256": digest(path),
        "fidelity": fidelity,
        "trajectory_standard_error": se,
        "ci95_normal": [max(0.0, fidelity - 1.96 * se), min(1.0, fidelity + 1.96 * se)],
        "batch_mean_standard_error": float(np.std([b["fidelity"] for b in batches], ddof=1) / np.sqrt(len(batches))),
        "test_batches": [f"jobs/{job['id']}/test_{i:02d}/record.json" for i in range(len(batches))],
        "resources": best["resources"],
        "total_search_cpu_seconds": sum(r["training_cpu_seconds"] + r["validation"]["cpu_seconds"] for _, r in records),
        "test_cpu_seconds": sum(b["cpu_seconds"] for b in batches),
    }


def execute(root: Path, job_id: str) -> dict:
    """Execute or resume one authenticated job, retaining compact stage histories."""
    manifest = authenticate(root)
    job = next(job for job in manifest["jobs"] if job["id"] == job_id)
    settings = manifest["protocol"]
    target = load_target(settings["qubits"], job["target"]).state_vector_copy()
    folder = root / "jobs" / job_id
    # Always replay adapters on resume so underlying stage seals are authenticated.
    if job["kind"] == "chain":
        result = topdown_mps.run_case(
            folder / "chain",
            target,
            topdown_mps.Config(**settings["ours"]),
            target_id=job["target"],
            target_index=list(TARGET_IDS).index(job["target"]),
        )
        result = {
            "status": result["status"],
            "final_reference": f"jobs/{job_id}/chain/final/record.json",
            "final_seal_sha256": digest(folder / "chain" / "final" / "complete.json"),
            "fidelity": result.get("fidelity"),
        }
    elif job["kind"] == "candidate":
        result = candidate(root, job, settings, target)
    else:
        result = finalize(root, job, settings, target)
    authenticate(root)
    return checkpoints.save_checkpoint(folder / "result", result, {})
