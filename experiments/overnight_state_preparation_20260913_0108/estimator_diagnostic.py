# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Preregistered exact-anchor control-variate diagnostic on development starts.

This is an estimator experiment, not a state-preparation fidelity comparison.
The known SVRG control variate is implemented in variance.py. A zero-distance
correction is exact by construction and cannot establish useful anchor lifetime.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import variance
from mqt.yaqs.optimization import StatevectorBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

ROOT = Path(__file__).parent
REPO = ROOT.parents[1]
OUTPUT = REPO / "output/overnight-state-preparation-20260913-0108"
TARGETS = ("gaussian_mu0p5_sigma0p1", "tfim_critical", "haar_random_1", "random_mps_bond2")
DISTANCES = (0.0, 0.05, 0.2, 1.0)
REPLICATES = 32
CV_PATHS = 32


def diagnostic_seed(target_id: str, distance_index: int, replicate: int) -> int:
    """Return a reproducible seed in the separate development-diagnostics domain."""
    payload = [202609130108, "development_diagnostics", "anchor_cv", target_id, distance_index, replicate]
    return int.from_bytes(hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).digest()[:8], "little")


def direction(theta: NDArray[np.float64], seed: int) -> NDArray[np.float64]:
    """Return a deterministic unit direction in the unwrapped parameter space.

    Raises:
        ValueError: If no finite trainable parameter vector is supplied.
    """
    if theta.ndim != 1 or not theta.size or not np.isrealobj(theta) or not np.all(np.isfinite(theta)):
        msg = "A nonempty finite real parameter vector is required."
        raise ValueError(msg)
    vector = np.random.default_rng(seed).normal(size=theta.size)
    return vector / np.linalg.norm(vector)


def _custody() -> dict[str, Any]:
    """Authenticate both immutable benchmark files against the startup manifest.

    Raises:
        ValueError: If either benchmark file changed.
    """
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    for name, expected in manifest["benchmark_hashes"].items():
        if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != expected:
            msg = f"Estimator diagnostic benchmark custody mismatch: {name}."
            raise ValueError(msg)
    return manifest


def _source_hashes() -> dict[str, str]:
    """Fingerprint actual numerical dependencies independently of other runners."""
    paths = {Path(__file__), ROOT / "variance.py", ROOT / "__init__.py"}
    paths.update((REPO / "src/mqt/yaqs").rglob("*.py"))
    paths.update((REPO / "benchmarks/state_preparation").glob("*.py"))
    paths.add(REPO / "experiments/state_preparation_pilot.py")
    return {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def _validate_seed_ranges(seeds: Sequence[int], paths: int) -> None:
    """Require disjoint consecutive per-path seed ranges across replications.

    Raises:
        ValueError: If the count or seed intervals are invalid.
    """
    if type(paths) is not int or paths < 2 or any(type(seed) is not int or seed < 0 for seed in seeds):
        msg = "Diagnostic path counts and seeds must be valid integers."
        raise ValueError(msg)
    ordered = sorted(seeds)
    if any(right - left < paths for left, right in itertools.pairwise(ordered)):
        msg = "Diagnostic replication seed ranges overlap."
        raise ValueError(msg)


def prepare(output: Path, starts_path: Path) -> dict[str, Any]:
    """Write or authenticate the complete protocol before numerical execution.

    Args:
        output: Unique diagnostic artifact directory.
        starts_path: Authenticated frozen development source starts.

    Returns:
        Checksummed protocol payload with all settings, seeds, and custody.

    Raises:
        ValueError: If sources, retained protocol, or target identities differ.
    """
    manifest = _custody()
    sources = cast("list[dict[str, Any]]", pilot.read_record(starts_path)["starts"])
    selected = [row for row in sources if row["replica"] == 11 and row["target_id"] in TARGETS]
    if len(selected) != len(TARGETS) or {row["target_id"] for row in selected} != set(TARGETS):
        msg = "Exactly four frozen development targets at construction replica 11 are required."
        raise ValueError(msg)
    seeds = {
        target_id: [
            [diagnostic_seed(target_id, index, replicate) for replicate in range(REPLICATES)]
            for index in range(len(DISTANCES))
        ]
        for target_id in TARGETS
    }
    _validate_seed_ranges([seed for sets in seeds.values() for group in sets for seed in group], 2 * CV_PATHS)
    protocol = {
        "schema": "yaqs.overnight.exact_anchor_diagnostic.v1",
        "phase": "development_diagnostics",
        "targets": list(TARGETS),
        "construction_replica": 11,
        "distances": list(DISTANCES),
        "distance_norm": "Euclidean norm in unwrapped saved parameter coordinates",
        "direction_seeds": {target_id: diagnostic_seed(target_id, -1, 0) for target_id in TARGETS},
        "replicates": REPLICATES,
        "cv_paths": CV_PATHS,
        "weighted_paths": 2 * CV_PATHS,
        "directional_passes_per_estimate": 4 * CV_PATHS,
        "snapshot": "exact_density_gradient_at_frozen_source_theta_once_per_target",
        "correction_seeds": seeds,
        "pairing": "same starting seed within each CV32/weighted64 pair; independent between replications",
        "order": "alternate CV-first and weighted-first by replicate index",
        "checkpoint_unit": "completed paired replication; interrupted uncommitted work must be logged separately",
        "source_starts_path": str(starts_path.resolve()),
        "source_starts_sha256": hashlib.sha256(starts_path.read_bytes()).hexdigest(),
        "selected_start_checksums": {row["target_id"]: pilot.digest(row) for row in selected},
        "benchmark_hashes": manifest["benchmark_hashes"],
        "source_hashes": _source_hashes(),
        "calibration": {
            "one_qubit_error": 0.00064,
            "two_qubit_error": 0.0051,
            "execution": "exact_independent_commuting_Pauli_factors",
        },
        "promotion": "At least 2x MSE-times-CPU benefit beyond zero distance with finite snapshot amortization",
        "falsification": "No nonzero-distance efficiency benefit, or no finite snapshot break-even reuse count",
        "confounders": ["small gradient norms", "laptop timing noise", "anchor age", "coordinate gauge"],
        "scope": "No parameter training, no confirmation targets, no final evaluation randomness",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "PROTOCOL.json"
    if path.exists():
        if pilot.read_record(path) != protocol:
            msg = "Retained diagnostic protocol or source hashes changed."
            raise ValueError(msg)
    else:
        pilot.write_record(path, protocol)
    return protocol


def _estimate_record(estimate: variance.Estimate, oracle: NDArray[np.float64]) -> dict[str, Any]:
    """Serialize one estimate with error and cost metrics against the exact oracle."""
    error = estimate.gradient - oracle
    norm = float(np.linalg.norm(estimate.gradient))
    oracle_norm = float(np.linalg.norm(oracle))
    squared = float(np.vdot(error, error).real)
    cosine = float(np.dot(estimate.gradient, oracle) / (norm * oracle_norm)) if norm * oracle_norm > 0 else None
    return {
        "gradient": estimate.gradient.tolist(),
        "loss": estimate.loss,
        "gradient_norm": norm,
        "squared_error": squared,
        "relative_error": math.sqrt(squared) / oracle_norm if oracle_norm > 0 else None,
        "cosine": None if cosine is None else float(np.clip(cosine, -1, 1)),
        "paths": estimate.paths,
        "directional_passes": estimate.directional_passes,
        "exact_channel_evaluations": estimate.exact_channel_evaluations,
        "cpu_seconds": estimate.cpu_seconds,
        "wall_seconds": estimate.wall_seconds,
        "seed": estimate.seed,
        "bias_status": estimate.bias_status,
    }


def summarize(rows: list[dict[str, Any]], oracle: NDArray[np.float64], snapshot_cpu: float) -> dict[str, Any]:
    """Aggregate gradient accuracy and derive strict efficiency break-even reuse.

    MSE is E[||g_hat-g||**2]. With K snapshot reuses, the compared cost products
    are MSE_cv*(t_cv+snapshot_cpu/K) and MSE_weighted*t_weighted. Report the
    smallest integer K giving a strict estimated benefit, or None when none
    exists. Timing uncertainty is not an estimator confidence interval.

    Returns:
        Method metrics, paired error ratio and amortization diagnostics.

    Raises:
        ValueError: If there are no replication rows or snapshot cost is invalid.
    """
    if not rows or not math.isfinite(snapshot_cpu) or snapshot_cpu < 0:
        msg = "Nonempty diagnostic rows and finite nonnegative snapshot CPU are required."
        raise ValueError(msg)
    result: dict[str, Any] = {}
    oracle_norm = float(np.linalg.norm(oracle))
    for method in ("cv", "weighted"):
        values = [row[method] for row in rows]
        errors = np.asarray([value["squared_error"] for value in values])
        mse = float(errors.mean())
        cpu = float(np.mean([value["cpu_seconds"] for value in values]))
        cosines = [value["cosine"] for value in values if value["cosine"] is not None]
        estimate_mean = np.mean([value["gradient"] for value in values], axis=0)
        result[method] = {
            "mse": mse,
            "rmse": math.sqrt(mse),
            "relative_rmse": math.sqrt(mse) / oracle_norm if oracle_norm > 0 else None,
            "mean_gradient_norm": float(np.mean([value["gradient_norm"] for value in values])),
            "mean_cosine": float(np.mean(cosines)) if cosines else None,
            "defined_cosine_count": len(cosines),
            "empirical_mean_error_norm": float(np.linalg.norm(estimate_mean - oracle)),
            "mse_standard_error": float(np.std(errors, ddof=1) / math.sqrt(len(rows))) if len(rows) > 1 else None,
            "mean_cpu_seconds": cpu,
            "total_cpu_seconds": float(sum(value["cpu_seconds"] for value in values)),
            "mean_wall_seconds": float(np.mean([value["wall_seconds"] for value in values])),
            "mse_times_cpu": mse * cpu,
            "directional_passes": sum(value["directional_passes"] for value in values),
            "paths": sum(value["paths"] for value in values),
        }
    cv = result["cv"]
    weighted = result["weighted"]
    benefit = weighted["mse_times_cpu"] - cv["mse_times_cpu"]
    reuse = math.floor(cv["mse"] * snapshot_cpu / benefit) + 1 if benefit > 0 else None
    cpu_saving = weighted["mean_cpu_seconds"] - cv["mean_cpu_seconds"]
    result.update({
        "replicates": len(rows),
        "oracle_gradient_norm": oracle_norm,
        "snapshot_cpu_seconds": snapshot_cpu,
        "snapshot_mse_cpu_break_even_reuses": reuse,
        "snapshot_cpu_only_break_even_reuses": math.floor(snapshot_cpu / cpu_saving) + 1 if cpu_saving > 0 else None,
        "cv_mse_times_cpu_with_snapshot_at_setting_reuses": cv["mse"]
        * (cv["mean_cpu_seconds"] + snapshot_cpu / len(rows)),
        "cv_over_weighted_mse_ratio": cv["mse"] / weighted["mse"] if weighted["mse"] > 0 else None,
        "paired_lower_squared_error_count": sum(
            row["cv"]["squared_error"] < row["weighted"]["squared_error"] for row in rows
        ),
        "equal_directional_work": cv["directional_passes"] == weighted["directional_passes"],
    })
    return result


def diagnose_setting(
    engine: StatevectorBackend,
    theta: NDArray[np.float64],
    calibration: CalibratedLocalDepolarizingProvider,
    snapshot: variance.Snapshot,
    *,
    seeds: Sequence[int],
    paths: int,
    prior: dict[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run or exactly resume paired estimator replications at fixed parameters.

    Args:
        engine: Fixed zero-input problem backend.
        theta: Current fixed parameter vector, possibly displaced from anchor.
        calibration: Exact parameter-independent local Pauli calibration.
        snapshot: Exact anchor gradient; sampled snapshots are not supported.
        seeds: Separate per-replication seeds, shared within each paired method.
        paths: Control-variate path count; weighted estimator receives twice it.
        prior: Authenticated prior checkpoint payload for exact resume.
        checkpoint: Optional writer called after each completed replication.

    Returns:
        Raw gradients, oracle, costs, completed replications and summary.

    Raises:
        ValueError: If the settings, snapshot, or retained state are incompatible.
    """
    if type(paths) is not int or paths < 2:
        msg = "Control-variate paths must be an integer >= 2."
        raise ValueError(msg)
    _validate_seed_ranges(seeds, 2 * paths)
    if not seeds or snapshot.estimate.exact_channel_evaluations != 1:
        msg = "Diagnostic requires independent seeds and an exact snapshot."
        raise ValueError(msg)
    signature = pilot.digest({
        "problem": snapshot.problem_signature,
        "theta": theta.tolist(),
        "anchor": snapshot.theta.tolist(),
        "anchor_gradient": snapshot.estimate.gradient.tolist(),
        "paths": paths,
        "seeds": list(seeds),
    })
    state: dict[str, Any] | None = copy.deepcopy(prior)
    if state is not None:
        if state["signature"] != signature or not 0 <= len(state["replications"]) <= len(seeds):
            msg = "Diagnostic checkpoint configuration mismatch."
            raise ValueError(msg)
        for index, row in enumerate(state["replications"]):
            if row["index"] != index or row["seed"] != seeds[index]:
                msg = "Diagnostic checkpoint replication order or seeds changed."
                raise ValueError(msg)
    else:
        started_cpu = time.process_time()
        at_anchor = np.array_equal(theta, snapshot.theta)
        if at_anchor:
            oracle = snapshot.estimate.gradient.copy()
            oracle_loss = snapshot.estimate.loss
        else:
            oracle, oracle_loss, _ = density_loss_gradient(engine.circuit, theta, engine.target, calibration)
        state = {
            "signature": signature,
            "theta": theta.tolist(),
            "exact_by_construction": at_anchor,
            "oracle_gradient": oracle.tolist(),
            "oracle_loss": oracle_loss,
            "oracle_cpu_seconds": 0.0 if at_anchor else time.process_time() - started_cpu,
            "oracle_exact_evaluations": 0 if at_anchor else 1,
            "replications": [],
            "status": "running",
        }
    oracle = np.asarray(state["oracle_gradient"], dtype=float)
    for index in range(len(state["replications"]), len(seeds)):
        seed = seeds[index]
        row: dict[str, Any] = {"index": index, "seed": seed}
        order = ("cv", "weighted") if index % 2 == 0 else ("weighted", "cv")
        for method in order:
            if method == "cv":
                estimate = variance.control_variate_estimate(
                    engine, theta, calibration, snapshot, paths=paths, seed=seed
                )
            else:
                estimate = variance.weighted_estimate(engine, theta, calibration, paths=2 * paths, seed=seed)
            row[method] = _estimate_record(estimate, oracle)
        state["replications"].append(row)
        if checkpoint:
            checkpoint(state)
    state["summary"] = summarize(state["replications"], oracle, snapshot.estimate.cpu_seconds)
    state["status"] = "complete"
    if checkpoint:
        checkpoint(state)
    return state


def _snapshot_record(snapshot: variance.Snapshot) -> dict[str, Any]:
    """Serialize exact snapshot arrays, problem signature and creation cost."""
    return {
        "theta": snapshot.theta.tolist(),
        "problem_signature": snapshot.problem_signature,
        "estimate": _estimate_record(snapshot.estimate, snapshot.estimate.gradient),
    }


def _restore_snapshot(record: dict[str, Any]) -> variance.Snapshot:
    """Restore a checksummed exact snapshot with read-only numerical arrays."""
    value = record["estimate"]
    gradient = np.asarray(value["gradient"], dtype=float)
    theta = np.asarray(record["theta"], dtype=float)
    gradient.setflags(write=False)
    theta.setflags(write=False)
    estimate = variance.Estimate(
        gradient=gradient,
        **{
            key: value[key]
            for key in (
                "loss",
                "paths",
                "directional_passes",
                "exact_channel_evaluations",
                "cpu_seconds",
                "wall_seconds",
                "seed",
                "bias_status",
            )
        },
    )
    return variance.Snapshot(theta, estimate, record["problem_signature"])


def run(output: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Execute the frozen panel serially, resuming checksummed replications.

    Snapshot cost is charged once per target. Exact displaced-oracle cost is
    diagnostic work, kept separately from estimator cost. Backend setup is
    shared and separately reported. No optimizer or confirmation set is used.

    Returns:
        Completed per-setting identities, summaries and aggregate cost ledger.

    Raises:
        ValueError: If custody, sources, or checkpoint signatures change.
    """
    starts_path = Path(protocol["source_starts_path"])
    if prepare(output, starts_path) != protocol:
        msg = "Frozen estimator protocol mismatch."
        raise ValueError(msg)
    sources = cast("list[dict[str, Any]]", pilot.read_record(starts_path)["starts"])
    selected = {row["target_id"]: row for row in sources if row["replica"] == 11 and row["target_id"] in TARGETS}
    calibration = CalibratedLocalDepolarizingProvider()
    summaries: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    runtime_hash = pilot.digest(protocol)

    def authenticate() -> None:
        _custody()
        if _source_hashes() != protocol["source_hashes"]:
            msg = "Estimator numerical source fingerprint changed during execution."
            raise ValueError(msg)
        if hashlib.sha256(starts_path.read_bytes()).hexdigest() != protocol["source_starts_sha256"]:
            msg = "Estimator frozen starting artifacts changed during execution."
            raise ValueError(msg)

    for target_id in TARGETS:
        authenticate()
        source = selected[target_id]
        circuit = pilot.restore_circuit(source["candidate"]["circuit"])
        theta = np.asarray(source["candidate"]["theta"], dtype=float)
        target = load_target(6, target_id).state_vector_copy()
        setup_start = time.process_time()
        engine = StatevectorBackend(circuit, target)
        setup_cpu = time.process_time() - setup_start
        anchor_path = output / f"{target_id}__anchor.json"
        if anchor_path.exists():
            anchor: dict[str, Any] = pilot.read_record(anchor_path)
            if anchor["runtime_hash"] != runtime_hash or anchor["source_checksum"] != pilot.digest(source):
                msg = "Retained estimator anchor signature mismatch."
                raise ValueError(msg)
            snapshot = _restore_snapshot(anchor["snapshot"])
        else:
            snapshot = variance.build_snapshot(engine, theta, calibration)
            anchor = {
                "runtime_hash": runtime_hash,
                "source_checksum": pilot.digest(source),
                "snapshot": _snapshot_record(snapshot),
                "direction": direction(theta, protocol["direction_seeds"][target_id]).tolist(),
                "shared_backend_setup_cpu_seconds": setup_cpu,
            }
            authenticate()
            pilot.write_record(anchor_path, anchor)
        anchors.append(anchor)
        for distance_index, distance in enumerate(DISTANCES):
            authenticate()
            path = output / f"{target_id}__distance_{distance_index}.json"
            shifted = theta + distance * np.asarray(anchor["direction"])
            saved = cast("dict[str, Any]", pilot.read_record(path)) if path.exists() else None
            if saved is not None and (
                saved["runtime_hash"] != runtime_hash or saved["anchor_checksum"] != pilot.digest(anchor)
            ):
                msg = "Retained estimator run signature mismatch."
                raise ValueError(msg)

            def write_state(
                state: dict[str, Any],
                *,
                destination: Path = path,
                anchor_checksum: str = pilot.digest(anchor),
                identifier: str = target_id,
                requested_distance: float = distance,
            ) -> None:
                authenticate()
                pilot.write_record(
                    destination,
                    {
                        "runtime_hash": runtime_hash,
                        "anchor_checksum": anchor_checksum,
                        "target_id": identifier,
                        "distance": requested_distance,
                        "state": state,
                    },
                )

            state = diagnose_setting(
                engine,
                shifted,
                calibration,
                snapshot,
                seeds=protocol["correction_seeds"][target_id][distance_index],
                paths=CV_PATHS,
                prior=None if saved is None else saved["state"],
                checkpoint=write_state,
            )
            record = pilot.read_record(path)
            summaries.append({
                "target_id": target_id,
                "distance": distance,
                "run_checksum": pilot.digest(record),
                "exact_by_construction": state["exact_by_construction"],
                "oracle_cpu_seconds": state["oracle_cpu_seconds"],
                "oracle_exact_evaluations": state["oracle_exact_evaluations"],
                "summary": state["summary"],
            })
            print(target_id, distance, state["summary"]["cv_over_weighted_mse_ratio"], flush=True)
    result: dict[str, Any] = {
        "runtime_hash": runtime_hash,
        "status": "complete",
        "settings": summaries,
        "anchor_checksums": [pilot.digest(anchor) for anchor in anchors],
        "ledger": {
            "exact_snapshot_evaluations": len(anchors),
            "snapshot_cpu_seconds": sum(anchor["snapshot"]["estimate"]["cpu_seconds"] for anchor in anchors),
            "shared_backend_setup_cpu_seconds": sum(anchor["shared_backend_setup_cpu_seconds"] for anchor in anchors),
            "oracle_exact_evaluations": sum(row["oracle_exact_evaluations"] for row in summaries),
            "oracle_cpu_seconds": sum(row["oracle_cpu_seconds"] for row in summaries),
            "cv_cpu_seconds": sum(row["summary"]["cv"]["total_cpu_seconds"] for row in summaries),
            "weighted_cpu_seconds": sum(row["summary"]["weighted"]["total_cpu_seconds"] for row in summaries),
            "snapshot_actual_reuses_per_target": len(DISTANCES) * REPLICATES,
            "directional_passes_per_method": sum(row["summary"]["cv"]["directional_passes"] for row in summaries),
            "timing_scope": (
                "Numerical estimator calls; custody hashing, imports, output I/O and orchestration excluded"
            ),
        },
    }
    authenticate()
    final_path = output / "RESULT.json"
    if final_path.exists():
        if pilot.read_record(final_path) != result:
            msg = "Completed diagnostic result differs on resume."
            raise ValueError(msg)
    else:
        pilot.write_record(final_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare or run the fixed diagnostic; each completed replication can resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT / "estimator-diagnostic")
    parser.add_argument("--starts", type=Path, default=OUTPUT / "starts.json")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    protocol = prepare(args.output_dir, args.starts)
    if not args.prepare_only:
        run(args.output_dir, protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
