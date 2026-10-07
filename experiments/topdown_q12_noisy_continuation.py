# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Prune authenticated twelve-qubit ideal endpoints and refine their noisy circuits."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.state_preparation.constants import TARGET_IDS
from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH, load_target
from experiments import legacy_topdown_dense as legacy
from experiments import topdown_backend_support as support
from experiments import topdown_depth_search as depth_search
from experiments import topdown_mps
from mqt.yaqs.optimization import StatevectorBackend

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

VERSION = "topdown.q12_noisy_continuation.v1"
PROTOCOL: dict = {
    "version": VERSION,
    "qubits": 12,
    "pruning": "simultaneous B, signed ideal deletion scores, cutoff 0.01",
    "online_updates": 100,
    "batch_updates": 100,
    "rate": 0.2,
    "hardware": "IBM",
    "training_paths": 100,
    "training_seed_offset": 5000,
    "training": "unchanged cross-trajectory signal; fixed unconditional bank with no-jump multiplicity weights",
    "no_jump_bank": "stop without resampling",
    "returned_iterate": "final iterate; no selection using assessment scores",
    "final_batches": 4,
    "final_paths_per_batch": 8192,
    "assessment": "paired before/after scores on fresh independent histories; exact statevectors, no truncation",
    "construction_updates": 0,
    "depth_reselection": False,
}


def sources() -> dict[str, str]:
    """Fingerprint this driver, the unchanged numerical sources and target fixture.

    Returns:
        Relative source paths and SHA-256 digests.
    """
    return {
        **topdown_mps.source_hashes(),
        str(DEFAULT_TARGET_PATH.relative_to(topdown_mps.ROOT)): support.digest(DEFAULT_TARGET_PATH),
        str(Path(__file__).relative_to(topdown_mps.ROOT)): support.digest(Path(__file__)),
    }


def required(folder: Path) -> tuple:
    """Load a sealed checkpoint, refusing incomplete stages.

    Returns:
        Authenticated metadata and private arrays.

    Raises:
        ValueError: If the stage is not complete.
    """
    saved = support.load_checkpoint(folder)
    if saved is None:
        msg = f"Incomplete checkpoint: {folder}"
        raise ValueError(msg)
    return saved


def authenticate_parent(parent: Path, target_id: str, seed: int) -> tuple:
    """Check original completion, frozen source, target and selected ideal endpoint.

    Returns:
        Parent provenance, selected construction metadata and private parameters.

    Raises:
        ValueError: If a parent identity, receipt, source or replay differs.
    """
    root = parent.parent.parent
    manifest = json.loads((parent / "manifest.json").read_text())
    final, _ = required(parent / "final")
    receipt_path = root / "logs" / f"{parent.name}.process.json"
    receipt = json.loads(receipt_path.read_text())
    claim = json.loads((root / "claims" / parent.name / "launch.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    jobs = [job for job in plan["jobs"] if job["job_id"] == parent.name]
    target = load_target(12, target_id).state_vector_copy()
    fixture_sha256 = manifest.get("fixture_sha256")
    if fixture_sha256 is None:
        inherited = manifest["parent"]
        original_manifest = Path(inherited["folder"]) / "manifest.json"
        if support.digest(original_manifest) != inherited["manifest_sha256"]:
            msg = "Inherited parent manifest differs."
            raise ValueError(msg)
        fixture_sha256 = json.loads(original_manifest.read_text())["fixture_sha256"]
    checkpoint = (parent / final["selected_checkpoint"]).resolve()
    if (
        not checkpoint.is_relative_to(parent.resolve())
        or manifest["target_id"] != target_id
        or manifest["seed"] != seed
        or manifest["target_sha256"] != topdown_mps.target_digest(target)
        or fixture_sha256 != support.digest(DEFAULT_TARGET_PATH)
        or final["target_id"] != target_id
        or final["seed"] != seed
        or final["noise_or_pruning_performed"]
        or receipt["status"] != "completed"
        or receipt["exit_code"] != 0
        or len(jobs) != 1
        or receipt["command"] != jobs[0]["command"]
        or receipt["command"] != claim["command"][claim["command"].index("--") + 1 :]
    ):
        msg = "Parent identity, target or successful supervisor receipt differs."
        raise ValueError(msg)
    for name, digest in manifest["sources"].items():
        if support.digest(root / "source" / name) != digest or support.digest(topdown_mps.ROOT / name) != digest:
            msg = "Parent numerical source differs from the frozen continuation."
            raise ValueError(msg)
    stage, arrays = required(checkpoint)
    rows = [row for row in final["completed_depths"] if row["depth"] == final["best_depth"]]
    circuit = support.restore_circuit(stage["training"]["circuit"])
    fidelity = StatevectorBackend(circuit, target).metrics(arrays["theta"])[1]
    if (
        circuit.num_qubits != 12
        or stage["training"]["depth"] != final["best_depth"]
        or len(rows) != 1
        or rows[0]["checkpoint"] != final["selected_checkpoint"]
        or rows[0]["checkpoint_seal_sha256"] != support.digest(checkpoint / "complete.json")
        or not np.isfinite(fidelity)
        or max(abs(fidelity - final["best_fidelity"]), abs(fidelity - stage["fidelity"])) > 1e-9
    ):
        msg = "Parent selected checkpoint fails sealed exact replay."
        raise ValueError(msg)
    provenance = {
        "target_id": target_id,
        "seed": seed,
        "arm": manifest["arm"],
        "depth": final["best_depth"],
        "ideal_fidelity": fidelity,
        "parent": str(parent.resolve()),
        "selected_checkpoint": str(checkpoint),
        "parent_manifest_sha256": support.digest(parent / "manifest.json"),
        "parent_final_seal_sha256": support.digest(parent / "final/complete.json"),
        "parent_checkpoint_seal_sha256": support.digest(checkpoint / "complete.json"),
        "supervisor_receipt_sha256": support.digest(receipt_path),
        "historical_preparation_cpu_seconds": final.get(
            "total_preparation_cpu_seconds", final.get("total_cpu_seconds")
        ),
    }
    return provenance, stage, arrays


def prepare(references: Path, output: Path) -> dict:
    """Copy all 27 verified endpoints into a new immutable input panel.

    Returns:
        Frozen protocol, source fingerprints, jobs and input receipt hashes.

    Raises:
        ValueError: If the panel is incomplete or output already exists.
    """
    rows = json.loads(references.read_text(encoding="utf-8"))
    expected = {(target, seed) for target in TARGET_IDS for seed in (300, 400, 500)}
    if len(rows) != 27 or {(r["target_id"], r["seed"]) for r in rows} != expected or output.exists():
        msg = "Require a fresh output and all nine targets with three seeds."
        raise ValueError(msg)
    jobs = []
    for row in rows:
        provenance, stage, arrays = authenticate_parent(Path(row["parent"]), row["target_id"], row["seed"])
        job_id = f"{row['target_id']}__seed{row['seed']}"
        folder = output / "inputs" / job_id
        support.save_checkpoint(folder / "construction", stage, arrays)
        receipt = {
            "version": VERSION,
            **provenance,
            "construction_seal_sha256": support.digest(folder / "construction/complete.json"),
        }
        support.write_json(folder / "input.json", receipt)
        jobs.append({"job_id": job_id, "target_id": row["target_id"], "seed": row["seed"], "host": row["host"]})
    plan = {
        "protocol": PROTOCOL,
        "sources": sources(),
        "jobs": jobs,
        "inputs": {j["job_id"]: support.digest(output / "inputs" / j["job_id"] / "input.json") for j in jobs},
        "references_sha256": support.digest(references),
    }
    shutil.copyfile(references, output / "parent-references.json")
    support.write_json(output / "plan.json", plan)
    return plan


def authenticate_input(folder: Path) -> tuple:
    """Authenticate the input panel, construction seal and exact ideal fidelity.

    Returns:
        Receipt, circuit, private angles, target and the frozen job identity.

    Raises:
        ValueError: If the protocol, source, bytes or replay differ.
    """
    plan = json.loads((folder.parent.parent / "plan.json").read_text())
    receipt = json.loads((folder / "input.json").read_text())
    if (
        plan["protocol"] != PROTOCOL
        or plan["sources"] != sources()
        or receipt["version"] != VERSION
        or support.digest(folder / "input.json") != plan["inputs"][folder.name]
        or support.digest(folder / "construction/complete.json") != receipt["construction_seal_sha256"]
    ):
        msg = "Input panel protocol, numerical source or construction seal differs."
        raise ValueError(msg)
    record, arrays = required(folder / "construction")
    circuit = support.restore_circuit(record["training"]["circuit"])
    target = load_target(12, receipt["target_id"]).state_vector_copy()
    replay = StatevectorBackend(circuit, target).metrics(arrays["theta"])[1]
    if circuit.num_qubits != 12 or not np.isfinite(replay) or abs(replay - receipt["ideal_fidelity"]) > 1e-9:
        msg = "Input endpoint fails exact ideal replay."
        raise ValueError(msg)
    identity = {"protocol": PROTOCOL, "input_sha256": plan["inputs"][folder.name], "sources": plan["sources"]}
    return receipt, circuit, arrays["theta"], target, identity


def train_phase(
    folder: Path,
    circuit: ParameterizedCircuit,
    initial: NDArray,
    target: NDArray,
    identity: dict,
    seed: int,
    *,
    online: bool,
) -> tuple:
    """Seal one unchanged noisy phase; authenticate its initial angles on reuse.

    Returns:
        Complete phase metadata and private endpoint arrays.

    Raises:
        ValueError: If a cached phase has different inputs.
    """
    phase_identity = {**identity, "phase": "online" if online else "batch"}
    saved = support.load_checkpoint(folder)
    if saved is None:
        start = time.process_time()
        theta, trace = legacy.train(
            circuit,
            initial,
            target,
            steps=PROTOCOL["online_updates"] if online else PROTOCOL["batch_updates"],
            online=PROTOCOL["online_updates"] if online else 0,
            rate=PROTOCOL["rate"],
            seed=seed + PROTOCOL["training_seed_offset"],
            noisy=True,
            crn=True,
            hardware=PROTOCOL["hardware"],
            num_trajectories=PROTOCOL["training_paths"],
            compress_no_jump=True,
            record_noise_banks=True,
            require_jump_coverage=True,
            engine=StatevectorBackend(circuit, target),
        )
        support.save_checkpoint(
            folder,
            {"identity": phase_identity, "training": trace, "cpu_seconds": time.process_time() - start},
            {"initial": initial, "theta": theta},
        )
        saved = required(folder)
    if saved[0]["identity"] != phase_identity or not np.array_equal(saved[1]["initial"], initial):
        msg = "Cached phase identity or initial parameters differ."
        raise ValueError(msg)
    return saved


def run_case(folder: Path, input_folder: Path) -> dict:
    """Prune once, refine two phases, then score independent paired final batches.

    Returns:
        Sealed final scores, resources, CPU costs and checkpoint references.

    Raises:
        ValueError: If stage provenance or the shared noise bank differs.
    """
    receipt, original, theta, target, identity = authenticate_input(input_folder)
    manifest = folder / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        msg = "Run identity differs; use a fresh directory."
        raise ValueError(msg)
    support.write_json(manifest, identity)

    def progress(stage: str) -> None:
        record = {"target_id": receipt["target_id"], "seed": receipt["seed"], "stage": stage, "at_unix": time.time()}
        support.write_json(folder / "progress.json", record)
        print(json.dumps(record), flush=True)

    progress("pruning")
    pruned = support.load_checkpoint(folder / "pruning")
    if pruned is None:
        start = time.process_time()
        circuit, values, trace = support.prune_b(StatevectorBackend(original, target), theta)
        support.save_checkpoint(
            folder / "pruning",
            {
                "identity": identity,
                "circuit": support.circuit_record(circuit),
                "pruning": trace,
                "ideal_fidelity": StatevectorBackend(circuit, target).metrics(values)[1],
                "cpu_seconds": time.process_time() - start,
            },
            {"theta": values},
        )
        pruned = required(folder / "pruning")
    if pruned[0]["identity"] != identity:
        msg = "Cached pruning identity differs."
        raise ValueError(msg)
    circuit, initial = support.restore_circuit(pruned[0]["circuit"]), pruned[1]["theta"]
    progress("online")
    online = train_phase(folder / "online", circuit, initial, target, identity, receipt["seed"], online=True)
    progress("batch")
    batch = train_phase(folder / "batch", circuit, online[1]["theta"], target, identity, receipt["seed"], online=False)
    if online[0]["training"]["noise_banks"][0]["raw_sha256"] != batch[0]["training"]["noise_banks"][0]["raw_sha256"]:
        msg = "Online and batch training noise banks differ."
        raise ValueError(msg)
    engine = StatevectorBackend(circuit, target)
    scores = []
    for index in range(PROTOCOL["final_batches"]):
        progress(f"test_{index:02d}")
        directory = folder / f"test_{index:02d}"
        entropy = [20261007, 12, TARGET_IDS.index(receipt["target_id"]), receipt["seed"], 91000, index]
        saved = support.load_checkpoint(directory)
        if saved is None:
            start = time.process_time()
            codes = depth_search.sample_validation(circuit, "IBM", PROTOCOL["final_paths_per_batch"], entropy)
            before = support.score_codes(engine, initial, "IBM", codes)
            after = support.score_codes(engine, batch[1]["theta"], "IBM", codes)
            support.save_checkpoint(
                directory,
                {
                    "identity": identity,
                    "entropy": entropy,
                    "before": before,
                    "after": after,
                    "batch_seal_sha256": support.digest(folder / "batch/complete.json"),
                    "cpu_seconds": time.process_time() - start,
                },
                {"codes": codes},
            )
            saved = required(directory)
        if (
            saved[0]["identity"] != identity
            or saved[0]["entropy"] != entropy
            or saved[0]["batch_seal_sha256"] != support.digest(folder / "batch/complete.json")
        ):
            msg = "Cached assessment identity or training endpoint differs."
            raise ValueError(msg)
        scores.append(saved[0])
    aggregates = {}
    for role in ("before", "after"):
        mean = float(np.mean([row[role]["fidelity"] for row in scores]))
        se = float(np.sqrt(sum(row[role]["standard_error"] ** 2 for row in scores)) / len(scores))
        aggregates[role] = {
            "fidelity": mean,
            "standard_error": se,
            "ci95_normal": [max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se)],
        }
    result = {
        "identity": identity,
        "parent": receipt,
        "status": "complete"
        if all(s["after"]["status"] == "complete" for s in scores)
        else "test_jump_coverage_failure",
        "noisy_pruned": aggregates["before"],
        "noisy_refined": aggregates["after"],
        "ideal_pruned": pruned[0]["ideal_fidelity"],
        "ideal_refined": engine.metrics(batch[1]["theta"])[1],
        "fidelity": aggregates["after"]["fidelity"],
        "resources": support.compile_resources(circuit, batch[1]["theta"]),
        "original_logical_gate_count": len(original.gates),
        "pruned_logical_gate_count": len(circuit.gates),
        "training_cpu_seconds": sum(stage[0]["cpu_seconds"] for stage in (pruned, online, batch)),
        "assessment_cpu_seconds": sum(s["cpu_seconds"] for s in scores),
        "selected_parameters": "batch/arrays.npz#theta",
        "stage_seals": {
            name: support.digest(folder / name / "complete.json") for name in ("pruning", "online", "batch")
        },
        "test_seals": {
            f"test_{i:02d}": support.digest(folder / f"test_{i:02d}/complete.json") for i in range(len(scores))
        },
    }
    saved = support.load_checkpoint(folder / "final")
    if saved is not None and {k: v for k, v in saved[0].items() if k != "arrays"} != result:
        msg = "Cached final result differs from authenticated stages."
        raise ValueError(msg)
    if saved is None:
        support.save_checkpoint(folder / "final", result, {})
    progress(result["status"])
    return result


def main(argv: list[str] | None = None) -> None:
    """Require explicit preparation or one independent numerical run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args(argv)
    with threadpool_limits(limits=1):
        if args.action == "prepare":
            if args.references is None:
                parser.error("Preparation requires references.")
            result = prepare(args.references, args.output)
        else:
            if args.input is None:
                parser.error("Run requires a sealed input.")
            result = run_case(args.output, args.input)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
