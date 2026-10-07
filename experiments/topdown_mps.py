# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Current pruning-B method with interchangeable dense/MPS execution.

Run in a fresh process for meaningful worker-lifetime peak RSS. Default budgets
match the current optional every-depth search. Final independent trajectories
replace mandatory dense-matrix assessment; an exact q6 oracle is opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.state_preparation.constants import TARGET_IDS
from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH, load_target_collection
from experiments import legacy_topdown_dense as legacy
from experiments import topdown_backend_support as support
from experiments import topdown_batch_refinement_policies as policies
from experiments import topdown_depth_search as depth_search
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KrotovTruncation,
    MPSBackend,
    create_backend,
    create_brickwall_matrix_product_disentangler_parameterized_circuit,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit
    from mqt.yaqs.optimization.backends import PreparationBackend

ROOT = Path(__file__).resolve().parents[1]
VERSION = "topdown.weighted_mps.v1"


@dataclass(frozen=True)
class Config:
    """Explicit numerical and experimental settings authenticated on resume."""

    backend: str = "mps"
    qubits: int = 6
    seed: int = 300
    hardware: str = "IBM"
    max_depth: int = 16
    construction_steps: int = 200
    construction_online: int = 100
    noisy_steps: int = 200
    noisy_online: int = 100
    training_paths: int = 100
    validation_paths: int = 32768
    final_paths_per_batch: int = 8192
    final_batches: int = 4
    max_bond_dim: int | None = None
    svd_threshold: float = 0.0
    compress_target: bool = False
    exact_density_check: bool = False
    profile_memory: bool = False

    def validate(self) -> None:
        """Reject ambiguous budgets and expensive optional density checks above q6.

        Raises:
            ValueError: If any configuration field is unsupported.
        """
        positive = ("qubits", "max_depth", "training_paths", "final_batches")
        nonnegative = ("seed", "construction_steps", "construction_online", "noisy_steps", "noisy_online")
        if any(type(getattr(self, key)) is not int or getattr(self, key) < 1 for key in positive) or any(
            type(getattr(self, key)) is not int or getattr(self, key) < 0 for key in nonnegative
        ):
            msg = "Invalid integer budget."
            raise ValueError(msg)
        if any(
            type(getattr(self, key)) is not int or getattr(self, key) < 2
            for key in ("validation_paths", "final_paths_per_batch")
        ):
            msg = "Assessment requires at least two histories per bank."
            raise ValueError(msg)
        if (
            self.backend not in {"mps", "statevector"}
            or self.hardware not in {"IBM", "Google"}
            or self.construction_online > self.construction_steps
            or self.noisy_online > self.noisy_steps
        ):
            msg = "Invalid backend, hardware or phase budget."
            raise ValueError(msg)
        if self.max_bond_dim is not None and (type(self.max_bond_dim) is not int or self.max_bond_dim < 1):
            msg = "Invalid bond cap."
            raise ValueError(msg)
        if (
            not np.isfinite(self.svd_threshold)
            or self.svd_threshold < 0
            or (
                self.backend == "statevector"
                and (self.max_bond_dim is not None or self.svd_threshold or self.compress_target)
            )
        ):
            msg = "Truncation options require an MPS backend."
            raise ValueError(msg)
        if self.exact_density_check and self.qubits > 6:
            msg = "The optional dense oracle is limited to six qubits."
            raise ValueError(msg)


def diagnostics(engine: PreparationBackend) -> dict:
    """Describe backend numerical semantics and cumulative measured tensor work."""
    if isinstance(engine, MPSBackend):
        return {
            **engine.diagnostics.record(),
            "truncation": asdict(engine.truncation),
            "target_compression_infidelity": engine.target_compression_infidelity,
            "gradient_semantics": (
                "exact adjoint without truncation; otherwise ignores SVD rank derivatives; cross is historical signal"
            ),
        }
    return {"backend": "statevector", "truncation": None}


def source_hashes() -> dict[str, str]:
    """Authenticate numerical sources and experiment helpers, excluding result files."""
    paths = {
        p for directory in ("src/mqt/yaqs", "benchmarks/state_preparation") for p in (ROOT / directory).rglob("*.py")
    }
    paths.update(
        ROOT / f"experiments/{name}.py"
        for name in (
            "topdown_mps",
            "topdown_backend_support",
            "topdown_batch_refinement_policies",
            "topdown_depth_search",
            "legacy_topdown_dense",
            "legacy_pruning_ablation",
            "legacy_topdown_q6",
            "legacy_topdown_weighted",
            "topdown_optimizer_stability",
        )
    )
    paths.add(ROOT / "experiments/overnight_state_preparation_20260913_0108/optimizers.py")
    return {str(p.relative_to(ROOT)): support.digest(p) for p in sorted(paths)}


def target_digest(target: MPS | NDArray) -> str:
    """Authenticate tensor-only targets without constructing dense amplitudes."""
    digest = hashlib.sha256()
    for array in target.tensors if isinstance(target, MPS) else [np.asarray(target)]:
        digest.update(str(array.shape).encode())
        digest.update(np.asarray(array, dtype=np.complex128).tobytes())
    return digest.hexdigest()


def _engine(
    circuit: ParameterizedCircuit, target: MPS | NDArray, config: Config, *, assessment: bool = False
) -> PreparationBackend:
    """Create a backend; assessment always uses the original untruncated target."""
    truncation = (
        None
        if config.backend == "statevector" or assessment
        else KrotovTruncation(max_bond_dim=config.max_bond_dim, svd_threshold=config.svd_threshold)
    )
    return create_backend(
        circuit,
        target,
        backend=config.backend,
        truncation=truncation,
        compress_target=config.compress_target and not assessment,
    )


def _save_trace(folder: Path, record: dict, theta: NDArray, trace: dict, **arrays: NDArray) -> None:
    """Store each complete stage trace once and reference its binary fidelity curve."""
    trace = dict(trace)
    curve = np.array(trace.pop("fidelity_trace", []), dtype=float)
    trace["fidelity_curve"] = {"file": "arrays.npz", "key": "fidelity"}
    support.save_checkpoint(folder, {**record, "training": trace}, {"theta": theta, "fidelity": curve, **arrays})


def run_case(
    folder: Path,
    target: MPS | NDArray,
    config: Config,
    *,
    target_id: str,
    target_index: int = 0,
    initial_theta: NDArray | None = None,
) -> dict:
    """Run/resume one target, locking validation selection before final assessment.

    Every completed stage is sealed. Interrupted stages are recomputed from the
    previous completed stage with identical seeds. Complete resumes authenticate
    all receipts. Old q6 result directories are never modified or reused as cache.
    An optional explicit parameter vector replaces only depth-one initialization;
    all subsequent extension, pruning, optimizer and noise policies are unchanged.

    Returns:
        A compact final summary referencing stage histories and independent banks.

    Raises:
        ValueError: If configuration, sources, target or cached payloads differ.
    """
    config.validate()
    length = target.length if isinstance(target, MPS) else len(target).bit_length() - 1
    if length != config.qubits or type(target_index) is not int or target_index < 0:
        msg = "Target or target index does not match the declared configuration."
        raise ValueError(msg)
    if initial_theta is not None:
        circuit = create_brickwall_matrix_product_disentangler_parameterized_circuit(
            config.qubits, 1, initial_single_qubit_layer=True
        )
        initial_theta = np.asarray(initial_theta)
        if (
            initial_theta.shape != (circuit.num_params,)
            or initial_theta.dtype.kind not in "iuf"
            or not np.all(np.isfinite(initial_theta))
        ):
            msg = "Explicit initial parameters must be finite real depth-one angles."
            raise ValueError(msg)
        initial_theta = initial_theta.astype(np.float64, copy=True)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": VERSION,
        "config": asdict(config),
        "target_id": target_id,
        "target_index": target_index,
        "target_sha256": target_digest(target),
        "fixture_sha256": support.digest(DEFAULT_TARGET_PATH),
        "sources": source_hashes(),
        "noise_basis": "logical",
        "noise_law": "legacy at-most-one Pauli jump per logical gate",
        "selection": "optional every-depth validation/patience search",
        "assessment": "untruncated independent trajectory batches",
        "construction_endpoint": "best iterate",
        "noisy_endpoint": "final iterate",
        "pruning": "B, cutoff 0.01",
    }
    if initial_theta is not None:
        manifest["initial_theta_sha256"] = hashlib.sha256(initial_theta.tobytes()).hexdigest()
    path = folder / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        msg = "Resume manifest differs in configuration, target or source hashes."
        raise ValueError(msg)
    if not path.exists():
        support.write_json(path, manifest)
    for seal in folder.glob("**/complete.json"):
        support.load_checkpoint(seal.parent)
    completed = support.load_checkpoint(folder / "final")
    if completed is not None:
        if (
            completed[0].get("selection_sha256") is not None
            and support.digest(folder / "selection_locked.json") != completed[0]["selection_sha256"]
        ):
            msg = "Locked selection checksum mismatch."
            raise ValueError(msg)
        return completed[0]
    with measure_resources(trace_allocations=config.profile_memory) as resources, threadpool_limits(limits=1):
        setup_start = time.process_time()
        cached_target = (
            MPS.from_statevector(target) if config.backend == "mps" and not isinstance(target, MPS) else target
        )
        setup_cpu = time.process_time() - setup_start
        result = _search(folder, cached_target, config, target_index, initial_theta=initial_theta)
        result["target_factorization_cpu_seconds"] = setup_cpu
    result["invocation_resources"] = resources
    if source_hashes() != manifest["sources"]:
        msg = "Numerical sources changed during the run."
        raise ValueError(msg)
    return support.save_checkpoint(folder / "final", result, {})


@contextmanager
def measure_resources(*, trace_allocations: bool = True) -> Generator[dict, None, None]:
    """Measure a complete invocation including temporary arrays and adjoint tapes.

    Yields:
        A receipt populated on exit. RSS is a worker-lifetime high-water mark;
        NumPy/Python traced allocations exclude some native-library workspaces.
    """
    cpu, wall = time.process_time(), time.perf_counter()
    owned_trace = trace_allocations and not tracemalloc.is_tracing()
    if owned_trace:
        tracemalloc.start()
    result = {}
    try:
        yield result
    finally:
        peak = tracemalloc.get_traced_memory()[1] if tracemalloc.is_tracing() else None
        result.update({
            "cpu_seconds": time.process_time() - cpu,
            "wall_seconds": time.perf_counter() - wall,
            "traced_peak_bytes": peak,
            "traced_scope": "disabled"
            if peak is None
            else "this invocation"
            if owned_trace
            else "preexisting process tracing session",
            "worker_lifetime_peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
            ),
            "rss_scope": "entire worker lifetime, including imports; use a fresh process for comparisons",
        })
        if owned_trace:
            tracemalloc.stop()


def _search(
    folder: Path,
    target: MPS | NDArray,
    config: Config,
    target_index: int,
    *,
    initial_theta: NDArray | None = None,
) -> dict:
    """Execute frozen construction/pruning/refinement stages and independent selection."""
    tracker, previous = depth_search.Patience(), np.empty(0)
    candidates, stage_costs = [], []
    for depth in range(1, config.max_depth + 1):
        base = folder / f"depth_{depth:02d}"
        circuit = create_brickwall_matrix_product_disentangler_parameterized_circuit(
            config.qubits, depth, initial_single_qubit_layer=True
        )
        built = support.load_checkpoint(base / "construction")
        if built is None:
            start = time.process_time()
            initial = legacy.extend(
                previous, circuit.num_params, config.seed + 1000 + depth, 0.05 if depth == 1 else 0.001
            )
            if depth == 1 and initial_theta is not None:
                initial = initial_theta.copy()
            engine = _engine(circuit, target, config)
            theta, trace = policies.train_noiseless(
                circuit,
                initial,
                target,
                policy="lbfgs",
                steps=config.construction_steps,
                online=config.construction_online,
                engine=engine,
            )
            _save_trace(
                base / "construction",
                {
                    "circuit": support.circuit_record(circuit),
                    "diagnostics": diagnostics(engine),
                    "cpu_seconds": time.process_time() - start,
                },
                theta,
                trace,
                initial=initial,
            )
            built = support.load_checkpoint(base / "construction")
        assert built is not None
        previous = built[1]["theta"]
        stage_costs.append(built[0]["cpu_seconds"])
        pruned = support.load_checkpoint(base / "pruning")
        if pruned is None:
            start = time.process_time()
            engine = _engine(circuit, target, config)
            reduced, theta, trace = support.prune_b(engine, previous)
            support.save_checkpoint(
                base / "pruning",
                {
                    "circuit": support.circuit_record(reduced),
                    "pruning": trace,
                    "diagnostics": diagnostics(engine),
                    "cpu_seconds": time.process_time() - start,
                },
                {"theta": theta},
            )
            pruned = support.load_checkpoint(base / "pruning")
        assert pruned is not None
        circuit, initial = support.restore_circuit(pruned[0]["circuit"]), pruned[1]["theta"]
        stage_costs.append(pruned[0]["cpu_seconds"])
        refined = support.load_checkpoint(base / "refinement")
        if refined is None:
            start = time.process_time()
            engine = _engine(circuit, target, config)
            status, reason = "complete", None
            try:
                theta, trace = legacy.train(
                    circuit,
                    initial,
                    target,
                    steps=config.noisy_steps,
                    online=config.noisy_online,
                    rate=0.2,
                    seed=config.seed + 5000,
                    noisy=True,
                    hardware=config.hardware,
                    num_trajectories=config.training_paths,
                    compress_no_jump=True,
                    record_noise_banks=True,
                    require_jump_coverage=True,
                    engine=engine,
                )
            except RuntimeError as error:
                if "sampled bank has no jumps" not in str(error):
                    raise
                theta, trace, status, reason = initial, {}, "no_jump_training_bank", str(error)
            _save_trace(
                base / "refinement",
                {
                    "status": status,
                    "reason": reason,
                    "diagnostics": diagnostics(engine),
                    "cpu_seconds": time.process_time() - start,
                },
                theta,
                trace,
            )
            refined = support.load_checkpoint(base / "refinement")
        assert refined is not None
        stage_costs.append(refined[0]["cpu_seconds"])
        validation = support.load_checkpoint(base / "validation")
        if validation is None:
            start = time.process_time()
            entropy = [
                20260914,
                config.qubits,
                target_index,
                config.seed,
                0,
                ("IBM", "Google").index(config.hardware),
                depth,
                90000,
            ]
            codes = depth_search.sample_validation(circuit, config.hardware, config.validation_paths, entropy)
            if refined[0]["status"] == "complete":
                engine = _engine(circuit, target, config, assessment=True)
                score = support.score_codes(engine, refined[1]["theta"], config.hardware, codes)
            else:
                score = {"status": refined[0]["status"], "fidelity": None, "standard_error": 0.0}
            support.save_checkpoint(
                base / "validation",
                {**score, "entropy": entropy, "cpu_seconds": time.process_time() - start},
                {"codes": codes},
            )
            validation = support.load_checkpoint(base / "validation")
        assert validation is not None
        stage_costs.append(validation[0]["cpu_seconds"])
        score = validation[0]
        decision = tracker.observe(
            depth, score["fidelity"] if score["status"] == "complete" else None, score["standard_error"]
        )
        candidates.append({
            "depth": depth,
            "validation": f"depth_{depth:02d}/validation/record.json",
            "decision": decision,
        })
        if tracker.stale >= depth_search.PROTOCOL["patience"]:
            break
    selection = {
        "selected_depth": tracker.best_depth,
        "candidates": candidates,
        "stop_reason": "patience" if tracker.stale >= depth_search.PROTOCOL["patience"] else "depth_cap",
    }
    locked = folder / "selection_locked.json"
    if locked.exists() and json.loads(locked.read_text()) != selection:
        msg = "Replayed selection differs from the locked decision."
        raise ValueError(msg)
    if not locked.exists():
        support.write_json(locked, selection)
    selected = tracker.best_depth
    if selected is None:
        return {
            **selection,
            "status": "no_valid_candidate",
            "total_search_cpu_seconds": sum(stage_costs),
            "selection_sha256": support.digest(locked),
        }
    base = folder / f"depth_{selected:02d}"
    pruned, refined = support.load_checkpoint(base / "pruning"), support.load_checkpoint(base / "refinement")
    assert pruned is not None
    assert refined is not None
    circuit, theta = support.restore_circuit(pruned[0]["circuit"]), refined[1]["theta"]
    engine = _engine(circuit, target, config, assessment=True)
    batches = []
    for batch in range(config.final_batches):
        directory = folder / f"test_{batch:02d}"
        saved = support.load_checkpoint(directory)
        if saved is None:
            start = time.process_time()
            entropy = [
                20260914,
                config.qubits,
                target_index,
                config.seed,
                0,
                ("IBM", "Google").index(config.hardware),
                selected,
                91000,
                batch,
            ]
            codes = depth_search.sample_validation(circuit, config.hardware, config.final_paths_per_batch, entropy)
            score = support.score_codes(engine, theta, config.hardware, codes)
            support.save_checkpoint(
                directory,
                {
                    **score,
                    "entropy": entropy,
                    "selection_sha256": support.digest(locked),
                    "cpu_seconds": time.process_time() - start,
                },
                {"codes": codes},
            )
            saved = support.load_checkpoint(directory)
        assert saved is not None
        if saved[0]["selection_sha256"] != support.digest(locked):
            msg = "Test bank belongs to a different selection lock."
            raise ValueError(msg)
        batches.append(saved[0])
    means = np.array([b["fidelity"] for b in batches])
    mean = float(means.mean())
    trajectory_se = float(np.sqrt(sum(b["standard_error"] ** 2 for b in batches)) / len(batches))
    result = {
        **selection,
        "status": "complete" if all(b["status"] == "complete" for b in batches) else "test_jump_coverage_failure",
        "selected_parameters": f"depth_{selected:02d}/refinement/arrays.npz#theta",
        "fidelity": mean,
        "trajectory_standard_error": trajectory_se,
        "ci95_normal": [max(0.0, mean - 1.96 * trajectory_se), min(1.0, mean + 1.96 * trajectory_se)],
        "batch_mean_standard_error": float(means.std(ddof=1) / np.sqrt(len(means))) if len(means) > 1 else None,
        "test_batches": [f"test_{i:02d}/record.json" for i in range(len(batches))],
        "resources": support.compile_resources(circuit, theta),
        "assessment_diagnostics": diagnostics(engine),
        "total_search_cpu_seconds": sum(stage_costs),
        "test_cpu_seconds": sum(b["cpu_seconds"] for b in batches),
        "selection_sha256": support.digest(locked),
    }
    if config.exact_density_check:
        dense_target = target.to_vec() if isinstance(target, MPS) else target
        result["exact_density_oracle"] = legacy.matrix_score(circuit, theta, dense_target, config.hardware, "tjm")
    return result


def main(argv: list[str] | None = None) -> None:
    """Run selected six-/twelve-qubit fixture targets in the requested backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=TARGET_IDS, default=list(TARGET_IDS))
    for name, default in asdict(Config()).items():
        argument = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(argument, action="store_true")
        else:
            parser.add_argument(argument, type=int if default is None else type(default), default=default)
    args = parser.parse_args(argv)
    config = Config(**{name: getattr(args, name) for name in asdict(Config())})
    config.validate()
    collection = load_target_collection()
    for target_id in args.targets:
        record = collection.load(config.qubits, target_id)
        target = record.state_vector_copy()
        result = run_case(
            args.output / target_id, target, config, target_id=target_id, target_index=list(TARGET_IDS).index(target_id)
        )
        print(
            json.dumps({"target": target_id, "status": result["status"], "fidelity": result.get("fidelity")}),
            flush=True,
        )


if __name__ == "__main__":
    main()
