# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Continue sealed unpruned construction with independent ideal-fidelity patience."""

from __future__ import annotations

import argparse
import json
import operator
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH, load_target
from experiments import legacy_topdown_dense as legacy
from experiments import topdown_backend_support as support
from experiments import topdown_batch_refinement_policies as policies
from experiments import topdown_mps
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import StatevectorBackend, create_backend
from mqt.yaqs.optimization import create_brickwall_matrix_product_disentangler_parameterized_circuit as brickwall

if TYPE_CHECKING:
    from numpy.typing import NDArray

TARGETS = ("gaussian_mu0p5_sigma0p1", "random_mps_bond3", "haar_random_1")
VERSION = "topdown.noiseless_continuation.v2"


@dataclass(frozen=True)
class GrowthRule:
    """Control depth growth separately from the unchanged optimizer budget."""

    max_depth: int = 16
    patience: int = 3
    relative_gain: float = 0.01
    absolute_gain: float = 1e-12
    max_cpu_seconds: float = 21600.0

    def validate(self, start_depth: int) -> None:
        """Reject invalid convergence thresholds and continuation budgets.

        Raises:
            ValueError: If any budget or tolerance is invalid.
        """
        if (
            type(start_depth) is not int
            or start_depth < 1
            or type(self.max_depth) is not int
            or self.max_depth <= start_depth
            or type(self.patience) is not int
            or self.patience < 1
            or not 0 < self.relative_gain < 1
            or not 0 < self.absolute_gain < 1
            or not np.isfinite(self.max_cpu_seconds)
            or self.max_cpu_seconds <= 0
        ):
            msg = "Invalid noiseless continuation budget or tolerance."
            raise ValueError(msg)


@dataclass
class IdealPatience:
    """Track the material ideal record separately from the best actual checkpoint."""

    rule: GrowthRule
    record: float
    best: float
    best_depth: int
    stale: int = 0

    def observe(self, depth: int, fidelity: float) -> dict:
        """Advance ideal-only patience, accumulating subthreshold gains.

        Returns:
            The decision and numerical threshold relative to the previous record.

        Raises:
            ValueError: If the ideal fidelity is nonphysical or nonfinite.
        """
        if not np.isfinite(fidelity) or not -1e-10 <= fidelity <= 1 + 1e-10:
            msg = "Nonphysical ideal fidelity."
            raise ValueError(msg)
        fidelity = float(np.clip(fidelity, 0, 1))
        threshold = max(self.rule.absolute_gain, self.rule.relative_gain * min(self.record, 1 - self.record))
        gain = fidelity - self.record
        improved = gain > threshold
        if improved:
            self.record, self.stale = fidelity, 0
        else:
            self.stale += 1
        if fidelity > self.best:
            self.best, self.best_depth = fidelity, depth
        return {
            "depth": depth,
            "fidelity": fidelity,
            "gain_over_material_record": gain,
            "required_gain": threshold,
            "material_improvement": improved,
            "record": self.record,
            "stale": self.stale,
            "best_fidelity": self.best,
            "best_depth": self.best_depth,
        }


def sources() -> dict[str, str]:
    """Fingerprint this driver and the existing numerical implementation.

    Returns:
        Repository-relative paths and their SHA-256 digests.
    """
    return {
        **topdown_mps.source_hashes(),
        str(Path(__file__).relative_to(topdown_mps.ROOT)): support.digest(Path(__file__)),
    }


def parent_checkpoints(parent: Path, target: NDArray, target_id: str, start_depth: int) -> tuple:
    """Authenticate the original unpruned checkpoints without writing to them.

    Returns:
        Original configuration, parent manifest and compact checkpoint references.

    Raises:
        ValueError: If the parent settings, sources, target or checkpoint differ.
    """
    manifest = json.loads((parent / "manifest.json").read_text())
    config = topdown_mps.Config(**manifest["config"])
    config.validate()
    if (
        manifest["target_id"] != target_id
        or manifest["target_sha256"] != topdown_mps.target_digest(target)
        or manifest["fixture_sha256"] != support.digest(DEFAULT_TARGET_PATH)
        or manifest["sources"] != topdown_mps.source_hashes()
        or config.max_bond_dim is not None
        or config.svd_threshold != 0
        or config.compress_target
        or config.backend != "mps"
        or config.qubits > 12
    ):
        msg = "Parent target, numerical sources or untruncated MPS configuration differ."
        raise ValueError(msg)
    references = []
    for depth in range(1, start_depth + 1):
        folder = parent / f"depth_{depth:02d}" / "construction"
        loaded = support.load_checkpoint(folder)
        if loaded is None:
            msg = "Missing sealed parent construction checkpoint."
            raise ValueError(msg)
        record, arrays = loaded
        circuit = brickwall(config.qubits, depth, initial_single_qubit_layer=True)
        expected_circuit = json.loads(json.dumps(support.circuit_record(circuit)))
        if record["circuit"] != expected_circuit or arrays["theta"].shape != (circuit.num_params,):
            msg = "Parent checkpoint topology or parameters differ."
            raise ValueError(msg)
        fidelity = StatevectorBackend(circuit, target).metrics(arrays["theta"])[1]
        if abs(fidelity - record["training"]["best_fidelity"]) > 1e-9:
            msg = "Parent fidelity fails exact statevector replay."
            raise ValueError(msg)
        references.append({
            "depth": depth,
            "folder": str(folder.resolve()),
            "seal_sha256": support.digest(folder / "complete.json"),
            "fidelity": float(fidelity),
            "historical_cpu_seconds": record["cpu_seconds"],
        })
    return config, manifest, references


def construct(folder: Path, target: NDArray, previous: NDArray, config: topdown_mps.Config, depth: int) -> dict:
    """Train one unchanged ideal extension and save its best iterate once.

    Returns:
        Sealed checkpoint record with a fresh statevector endpoint replay.

    Raises:
        ValueError: If independent ideal fidelity disagrees with the optimizer.
    """
    config.validate()
    circuit = brickwall(config.qubits, depth, initial_single_qubit_layer=True)
    initial = legacy.extend(previous, circuit.num_params, config.seed + 1000 + depth, 0.001)
    start, wall = time.process_time(), time.perf_counter()
    backend_target = MPS.from_statevector(target) if config.backend == "mps" else target
    engine = create_backend(circuit, backend_target, backend=config.backend)
    theta, trace = policies.train_noiseless(
        circuit,
        initial,
        target,
        policy="lbfgs",
        steps=config.construction_steps,
        online=config.construction_online,
        engine=engine,
    )
    training_cpu = time.process_time() - start
    fidelity = StatevectorBackend(circuit, target).metrics(theta)[1]
    error = abs(fidelity - trace["best_fidelity"])
    if error > 1e-9:
        msg = "Continued checkpoint fails exact statevector replay."
        raise ValueError(msg)
    # Store parameter snapshots and the fidelity curve only in the sealed NPZ.
    arrays = {"theta": theta, "initial": initial, "fidelity": np.asarray(trace.pop("fidelity_trace"))}
    for key in ("initial_theta", "online_final_theta", "best_theta", "final_theta"):
        arrays[key] = np.asarray(trace.pop(key))
        trace[key] = {"file": "arrays.npz", "key": key}
    trace["fidelity_curve"] = {"file": "arrays.npz", "key": "fidelity"}
    return support.save_checkpoint(
        folder,
        {
            "depth": depth,
            "backend": config.backend,
            "circuit": support.circuit_record(circuit),
            "training": trace,
            "fidelity": float(fidelity),
            "statevector_replay_error": float(error),
            **({"statevector_mps_error": float(error)} if config.backend == "mps" else {}),
            "training_cpu_seconds": training_cpu,
            "cpu_seconds": time.process_time() - start,
            "wall_seconds": time.perf_counter() - wall,
            "diagnostics": topdown_mps.diagnostics(engine),
        },
        arrays,
    )


def run_case(
    output: Path,
    parent: Path,
    target: NDArray,
    target_id: str,
    rule: GrowthRule,
    start_depth: int = 4,
    *,
    backend: str = "mps",
) -> dict:
    """Continue a frozen ideal chain, retaining resumable checkpoints and stop reasons.

    Returns:
        Final summary; budget exhaustion is explicitly separate from stagnation.

    Raises:
        ValueError: If resume provenance or saved decisions no longer match.
    """
    rule.validate(start_depth)
    if backend not in {"mps", "statevector"}:
        msg = "Continuation backend must be mps or statevector."
        raise ValueError(msg)
    output, parent = output.resolve(), parent.resolve()
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        msg = "Continuation output must be separate from the original experiment."
        raise ValueError(msg)
    with threadpool_limits(limits=1):
        config, parent_manifest, references = parent_checkpoints(parent, target, target_id, start_depth)
        parent_backend = config.backend
        config = replace(config, backend=backend)
        manifest = {
            "version": VERSION,
            "target_id": target_id,
            "parent": str(parent),
            "parent_manifest_sha256": support.digest(parent / "manifest.json"),
            "target_sha256": parent_manifest["target_sha256"],
            "rule": asdict(rule),
            "start_depth": start_depth,
            "parent_backend": parent_backend,
            "execution_backend": backend,
            "continuation_scope": "shared parent checkpoint; later depths use this run's own best parameters",
            "construction_config": asdict(config),
            "parent_checkpoints": references,
            "sources": sources(),
            "patience_scope": "new ideal extensions only; historical noisy patience is discarded",
        }
        path = output / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            msg = "Continuation resume manifest mismatch."
            raise ValueError(msg)
        if not path.exists():
            support.write_json(path, manifest)
        best = max(references, key=operator.itemgetter("fidelity"))
        tracker = IdealPatience(rule, best["fidelity"], best["fidelity"], best["depth"])
        loaded = support.load_checkpoint(Path(references[-1]["folder"]))
        assert loaded is not None
        previous = loaded[1]["theta"]
        decisions, new_cpu = [], 0.0
        for depth in range(start_depth + 1, rule.max_depth + 1):
            folder = output / f"depth_{depth:02d}" / "construction"
            saved = support.load_checkpoint(folder)
            if saved is None:
                support.write_json(
                    output / "progress.json", {"status": "running", "active_depth": depth, "decisions": decisions}
                )
                construct(folder, target, previous, config, depth)
                saved = support.load_checkpoint(folder)
            assert saved is not None
            record, arrays = saved
            previous = arrays["theta"]
            new_cpu += record["cpu_seconds"]
            decision = tracker.observe(depth, record["fidelity"])
            decision.update({"checkpoint": str(folder.relative_to(output)), "cpu_seconds": record["cpu_seconds"]})
            decisions.append(decision)
            support.write_json(
                output / "progress.json", {"status": "running", "completed_depth": depth, "decisions": decisions}
            )
            print(json.dumps({"target": target_id, **decision}), flush=True)
            if tracker.stale >= rule.patience or new_cpu >= rule.max_cpu_seconds:
                break
        exhausted = new_cpu >= rule.max_cpu_seconds
        stop = "cpu_budget" if exhausted else "ideal_patience" if tracker.stale >= rule.patience else "depth_cap"
        if tracker.best_depth <= start_depth:
            selected = references[tracker.best_depth - 1]["folder"]
        else:
            selected = str(output / f"depth_{tracker.best_depth:02d}" / "construction")
        result = {
            "target": target_id,
            "execution_backend": backend,
            "status": "stagnated" if stop == "ideal_patience" else "budget_exhausted",
            "stop_reason": stop,
            "best_depth": tracker.best_depth,
            "best_ideal_fidelity": tracker.best,
            "selected_checkpoint": selected,
            "decisions": decisions,
            "new_cpu_seconds": new_cpu,
            "historical_construction_cpu_seconds": sum(r["historical_cpu_seconds"] for r in references),
            "noise_or_pruning_performed": False,
        }
        if sources() != manifest["sources"]:
            msg = "Continuation sources changed during execution."
            raise ValueError(msg)
        old = support.load_checkpoint(output / "final")
        if old is not None:
            expected = {key: value for key, value in old[0].items() if key != "arrays"}
            if result != expected:
                msg = "Continuation final result differs on replay."
                raise ValueError(msg)
            return old[0]
        support.write_json(output / "progress.json", result)
        return support.save_checkpoint(output / "final", result, {})


def main(argv: list[str] | None = None) -> None:
    """Expose only the three authorized twelve-qubit noiseless continuations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--backend", choices=("mps", "statevector"), default="mps")
    args = parser.parse_args(argv)
    target = load_target(12, args.target).state_vector_copy()
    parent = args.parent_campaign / "jobs" / f"ours__{args.target}" / "chain"
    result = run_case(
        args.output / args.target,
        parent,
        target,
        args.target,
        GrowthRule(max_depth=args.max_depth),
        backend=args.backend,
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
