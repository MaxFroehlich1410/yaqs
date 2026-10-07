# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Move only the frozen top-down pilot's noisy stages to native gate windows."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks.state_preparation.constants import TARGET_IDS
from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH, load_target_collection
from experiments import topdown_batch_refinement as batch
from experiments import topdown_depth_search as depth_search
from experiments.legacy_topdown_weighted import load_circuit
from experiments.state_preparation_competitors import evaluation, native_topdown, reference
from experiments.state_preparation_competitors import run as storage
from experiments.state_preparation_competitors.common import Candidate
from experiments.state_preparation_competitors.run_logical import TARGETS
from experiments.state_preparation_competitors.run_noisy import PINNED

if TYPE_CHECKING:
    from collections.abc import Sequence

METHOD = "current_b_native_refinement"


def make_parser() -> argparse.ArgumentParser:
    """Expose only the authorized three-target IBM seed-300 experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--reference-root", type=Path, default=reference.DEFAULT_REFERENCE_ROOT)
    return parser


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Freeze sources, numerical versions, noise and the unchanged search policy.

    Raises:
        ValueError: If targets or installed numerical versions are unsupported.
    """
    if not args.targets or len(set(args.targets)) != len(args.targets) or any(t not in TARGETS for t in args.targets):
        msg = "Expected unique pilot targets."
        raise ValueError(msg)
    versions = {name: importlib.metadata.version(name) for name in PINNED}
    if versions != PINNED:
        msg = "Use the pinned noisy-competitor environment."
        raise ValueError(msg)
    sources = set((storage.ROOT / "experiments").rglob("*.py"))
    sources |= set((storage.ROOT / "src/mqt/yaqs").rglob("*.py"))
    sources |= set((storage.ROOT / "benchmarks/state_preparation").glob("*.py"))
    sources.add(DEFAULT_TARGET_PATH)
    return {
        "schema": "yaqs.native_topdown_pilot.v1",
        "targets": args.targets,
        "reference_root": str(args.reference_root.resolve()),
        "versions": versions,
        "sources": {str(p.relative_to(storage.ROOT)): storage.digest(p) for p in sorted(sources)},
        "optimizer_seed": 300,
        "hardware": "IBM",
        "protocol": batch.protocol(),
        "intervention": "symbolic native compilation before noisy refinement; frozen schedule for validation and test",
        "online_order": "compiler may reorder trainable occurrences; native chronological sweeps follow that order",
        "noisy_optimizer": "unchanged final iterate, 100 online then 100 cross batch updates, rate .2, M100",
        "validation": "32768 independent unconditional paths; native role entropy; original patience 3 and cap 16",
        "reuse": "authenticated noiseless construction and B-pruned inputs only; charge their historical CPU",
        "final": "exact Qiskit density matrix after depth selection lock; no sampled test error",
    }


def checkpoint_inputs(
    source_case: Path, artifacts: dict[str, str], output: Path, target: np.ndarray, depth: int, previous: np.ndarray
) -> tuple[dict, dict, dict]:
    """Reuse authenticated noise-independent inputs or extend the same lineage."""
    records, receipts = [], []
    for stem in (f"construction_{depth:02d}", f"depth_{depth:02d}_B_pruned"):
        if f"search/{stem}.json" in artifacts:
            record = reference.authenticated_checkpoint(source_case, stem, artifacts)
            receipts.append({
                "path": str(source_case / "search" / (stem + ".json")),
                "sha256": artifacts[f"search/{stem}.json"],
            })
        else:
            record = (
                batch.construct(output / "lineage", target, 300, depth, previous, "lbfgs")
                if not records
                else depth_search.prune_checkpoint(output / "lineage", target, records[0], "B")
            )
            path = output / "lineage" / (stem + ".json")
            receipts.append({"path": str(path), "sha256": storage.digest(path)})
        records.append(record)
    built, pruned = records
    if (
        built["depth"] != depth
        or built["policy"] != "lbfgs"
        or pruned["depth"] != depth
        or pruned["pruning"]["variant"] != "B"
    ):
        msg = "Checkpoint identity differs from the frozen construction/pruning policy."
        raise ValueError(msg)
    return built, pruned, {"construction": receipts[0], "pruning": receipts[1]}


def validate_candidate(folder: Path, candidate: Candidate, target: np.ndarray, target_id: str, depth: int) -> dict:
    """Validate once on a separate native bank and retain its binary jump codes."""
    cpu, wall = time.process_time(), time.perf_counter()
    compiled = evaluation.preserve_native_candidate(candidate, target)
    entropy = [20260920, 6, TARGET_IDS.index(target_id), 300, 0, 0, depth, 91000]
    codes = evaluation.sample_validation(compiled.circuit, "IBM", depth_search.PROTOCOL["validation_paths"], entropy)
    score = evaluation.validation_score(compiled, "IBM", codes)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "validation.npz", codes=codes)
    return {
        **score,
        "raw_bank_sha256": score["bank_sha256"],
        "entropy": entropy,
        "bank_file": "validation.npz",
        "bank_sha256": storage.digest(folder / "validation.npz"),
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall,
        "resources": compiled.metadata,
    }


def run_depth(folder: Path, pruned: dict, target: np.ndarray, target_id: str, sources: dict) -> dict:
    """Train and validate one depth with authenticated resume and source references.

    Raises:
        ValueError: If saved inputs or the validation bank have changed.
    """
    if (folder / "seal.json").exists():
        record, _ = storage.load_bundle(folder)
        if (
            record["sources"] != sources
            or storage.digest(folder / "validation.npz") != record["validation"]["bank_sha256"]
        ):
            msg = "Saved depth inputs or validation bank changed."
            raise ValueError(msg)
        return record
    initial, circuit = np.asarray(pruned["theta"]), load_circuit(pruned)
    cpu = time.process_time()
    try:
        candidate = native_topdown.train_candidate(circuit, initial, target)
    except RuntimeError as error:
        if "sampled bank has no jumps" not in str(error):
            raise
        native = native_topdown.compile_training(circuit, target)
        candidate = Candidate(
            native.bind(initial),
            {
                "method": METHOD,
                "status": "no_jump_training_bank",
                "preserve_native_schedule": True,
                "reason": str(error),
                "initial_theta": initial.tolist(),
                "final_theta": initial.tolist(),
                "training": None,
                "cpu_seconds": time.process_time() - cpu,
            },
        )
    depth = pruned["depth"]
    candidate = Candidate(
        candidate.circuit, {**candidate.metadata, "depth": depth, "candidate_id": f"depth_{depth:02d}"}
    )
    validation = validate_candidate(folder, candidate, target, target_id, depth)
    compiled_input = native_topdown.compile_training(circuit, target).bind(initial)
    return storage.save_bundle(
        folder,
        {
            "depth": depth,
            "sources": sources,
            "candidate": candidate.metadata,
            "validation": validation,
        },
        [candidate.circuit, compiled_input],
    )


def run_target(output: Path, reference_root: Path, target_id: str) -> dict:
    """Restart native patience, lock selection, then score every visited endpoint."""
    target = load_target_collection().load(6, target_id).state_vector_copy()
    reference.frozen_reference(reference_root, target_id, 300, "IBM", target)
    source_case = reference_root / "pipeline/lbfgs" / f"{target_id}__seed300"
    artifacts = json.loads((source_case / "complete.json").read_text())["artifacts"]
    case = output / target_id
    case.mkdir(parents=True, exist_ok=True)
    tracker, previous = depth_search.Patience(), np.empty(0)
    rows, decisions = [], []
    costs = {
        "construction_cpu_seconds": 0.0,
        "pruning_cpu_seconds": 0.0,
        "native_training_cpu_seconds": 0.0,
        "validation_cpu_seconds": 0.0,
    }
    for depth in range(1, depth_search.PROTOCOL["depth_cap"] + 1):
        built, pruned, sources = checkpoint_inputs(source_case, artifacts, case, target, depth, previous)
        previous = np.asarray(built["theta"])
        folder = case / f"depth_{depth:02d}"
        record = run_depth(folder, pruned, target, target_id, sources)
        val = record["validation"]
        eligible = record["candidate"]["status"] == "complete" and (
            val["jump_paths"] > 0 or not pruned["circuit"]["gates"]
        )
        decision = tracker.observe(
            depth, val["fidelity"] if eligible else None, val["standard_error"] if eligible else 0.0
        )
        decisions.append({
            "depth": depth,
            **decision,
            "validation_fidelity": val["fidelity"],
            "validation_se": val["standard_error"],
        })
        costs["construction_cpu_seconds"] += built["cpu_seconds"]
        costs["pruning_cpu_seconds"] += pruned["pruning"]["cpu_seconds"]
        costs["native_training_cpu_seconds"] += record["candidate"]["cpu_seconds"]
        costs["validation_cpu_seconds"] += val["cpu_seconds"]
        rows.append({
            "depth": depth,
            "bundle": str(folder),
            "seal_sha256": storage.digest(folder / "seal.json"),
            "validation": val,
        })
        print(
            target_id,
            "depth",
            depth,
            "native validation",
            round(val["fidelity"], 7),
            "stale",
            tracker.stale,
            flush=True,
        )
        if tracker.stale >= depth_search.PROTOCOL["patience"]:
            break
    lock = {
        "target_id": target_id,
        "selected_depth": tracker.best_depth,
        "decisions": decisions,
        "candidates": rows,
        "cost": costs,
        "stop_reason": "patience" if tracker.stale >= depth_search.PROTOCOL["patience"] else "depth_cap",
    }
    lock_path = case / "selection_locked.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        msg = "Resumed native search differs from its locked selection."
        raise ValueError(msg)
    storage.write_json(lock_path, lock)
    lock_hash = storage.digest(lock_path)
    finals = []
    for row in rows:
        folder = Path(row["bundle"])
        record, circuits = storage.load_bundle(folder)
        meta = record["candidate"]
        scores = []
        for circuit in circuits:
            compiled = evaluation.preserve_native_candidate(Candidate(circuit, meta), target)
            scores.append(evaluation.exact_score(compiled, "IBM"))
        exact, initial_exact = scores
        native = native_topdown.compile_training(load_circuit(reference_record(record["sources"]["pruning"])), target)
        replay = native.bind(np.asarray(meta["final_theta"]))
        if replay != circuits[0]:
            msg = "Final parameters do not reproduce the saved native circuit."
            raise ValueError(msg)
        finals.append({
            "depth": row["depth"],
            "exact": exact,
            "initial_exact": initial_exact,
            "fine_tuning_gain_pp": 100 * (exact["fidelity"] - initial_exact["fidelity"]),
            "parameter_replay_equal": True,
            "candidate_reference": row["bundle"],
            "training_bank": {k: v for k, v in meta["training"]["noise_banks"][0].items() if k != "events"}
            if meta["training"]
            else None,
        })
    if storage.digest(lock_path) != lock_hash:
        msg = "Final evaluation changed the selection lock."
        raise ValueError(msg)
    result = {
        **lock,
        "selection_sha256": lock_hash,
        "final_scores": finals,
        "selected": next((r for r in finals if r["depth"] == tracker.best_depth), None),
        "build_train_cpu_seconds": sum(v for k, v in costs.items() if k != "validation_cpu_seconds"),
        "total_search_cpu_seconds": sum(costs.values()),
    }
    storage.write_json(case / "result.json", result)
    return result


def reference_record(receipt: dict) -> dict:
    """Read a checkpoint only after confirming its recorded digest.

    Raises:
        ValueError: If the reference changed.
    """
    path = Path(receipt["path"])
    if storage.digest(path) != receipt["sha256"]:
        msg = "Referenced construction/pruning checkpoint changed."
        raise ValueError(msg)
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict:
    """Run or resume fixed native training and retain complete source receipts."""
    manifest = build_manifest(args)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        msg = "Output manifest differs; use a new directory."
        raise ValueError(msg)
    storage.write_json(path, manifest)
    started, cpu = time.perf_counter(), time.process_time()
    results = [run_target(args.output.resolve(), args.reference_root.resolve(), t) for t in args.targets]
    if any(storage.digest(storage.ROOT / p) != d for p, d in manifest["sources"].items()):
        msg = "Experiment sources changed during execution."
        raise ValueError(msg)
    summary = {
        "manifest_sha256": storage.digest(path),
        "results": results,
        "execution_wall_seconds": time.perf_counter() - started,
        "execution_cpu_seconds": time.process_time() - cpu,
    }
    storage.write_json(args.output / "summary.json", summary)
    storage.write_json(
        args.output / "seal.json",
        {
            str(p.relative_to(args.output)): storage.digest(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file() and p != args.output / "seal.json"
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """Run the fixed native-noise pilot from the command line."""
    run(make_parser().parse_args(argv))


if __name__ == "__main__":
    main()
