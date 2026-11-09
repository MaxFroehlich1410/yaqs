#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
# SPDX-License-Identifier: MIT
# Licensed under the MIT License

"""Reproducible small-system experiment for scalar conservation recovery.

Run ``python fallback_pilot.py --help`` for the pilot grid and extension controls.
Each case compares three trajectories from identical prepared tensors. Diagnostic
hooks observe the production correction without replacing its algorithm.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import subprocess
import sys
import tarfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fallback_models as models
import fixtures_n as fx
import numpy as np
from scipy.sparse.linalg import expm_multiply

from mqt.yaqs import AnalogSimParams, EvolutionMode, track_conservation
from mqt.yaqs.analog import evolution
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.methods import bug, conservation
from mqt.yaqs.core.parallel_utils import limit_worker_threads, safe_set_numba_threads

ROOT = Path(__file__).resolve().parents[3]
VARIANTS = ("uncorrected", "scalar", "fallback")
SCHEMA = 1


@dataclass(frozen=True)
class Case:
    """Complete physical and numerical parameters for a matched triple."""

    model: str
    length: int
    cap: int
    initial: str
    dt: float
    seed: int = fx.SEED
    total_time: float = 2.0
    integrator: str = "bug"
    conserve_tol: float = 1e-13
    krylov_tol: float = 1e-14
    width_per_site: float | None = None
    initial_chi: int | None = None

    @property
    def key(self) -> str:
        """Stable identifier independent of process and execution order."""
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]
        return f"{self.model}-L{self.length}-chi{self.cap}-{self.initial}-{digest}"


def write_json(path: Path, data: Any) -> None:
    """Atomically checkpoint JSON, rejecting non-finite values."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def infidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Return normalized, phase-independent infidelity."""
    return float(np.clip(1 - abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real), 0, 1))


def energy(vector: np.ndarray, operator: Any) -> float:
    """Measure the Rayleigh quotient through the independently assembled matrix."""
    return float(np.vdot(vector, operator @ vector).real / np.vdot(vector, vector).real)


def canonical_vector(state: MPS) -> np.ndarray:
    """Return the physical vector in the fixture's site-0-LSB orientation."""
    if not state.flipped:
        return state.to_vec()
    copy = deepcopy(state)
    copy.flip_network()
    return copy.to_vec()


def save_snapshot(path: Path, state: MPS) -> None:
    """Store tensors and gauge without pickle, preserving reflected BUG frames."""
    np.savez_compressed(
        path,
        **{f"tensor_{i}": t for i, t in enumerate(state.tensors)},
        length=state.length,
        center=state.orthogonality_center,
        flipped=state.flipped,
        physical_dimensions=state.physical_dimensions,
    )


def load_snapshot(path: Path) -> MPS:
    """Reconstruct the exact pre-correction frame from a safe array archive."""
    with np.load(path, allow_pickle=False) as arrays:
        length = int(arrays["length"])
        state = MPS(
            length,
            tensors=[arrays[f"tensor_{i}"].copy() for i in range(length)],
            physical_dimensions=arrays["physical_dimensions"].tolist(),
        )
        state.flipped = bool(arrays["flipped"])
        state.set_center(int(arrays["center"]))
    return state


def scalar_diagnostics(state: MPS, mpo: Any, target: float) -> dict[str, Any]:
    """Explain a rejected direction using the original quadratic coefficients."""
    center = state.tensors[state.orthogonality_center]
    apply = conservation._center_operator(state, mpo, state.orthogonality_center)
    norm = float(np.vdot(center, center).real)
    projected = apply(center)
    value = float(np.vdot(center, projected).real / norm)
    gradient = projected - value * center
    squared = float(np.vdot(gradient, gradient).real)
    curvature = float(np.vdot(gradient, apply(gradient)).real)
    a, b, c = curvature - target * squared, 2 * squared, norm * (value - target)
    discriminant = b * b - 4 * a * c
    reason = (
        "zero_gradient"
        if squared < conservation._ZERO_GRADIENT_SQUARED
        else ("negative_discriminant" if discriminant < 0 else "unusable_root")
    )
    return {"reason": reason, "gradient_squared": squared, "discriminant": discriminant}


@contextmanager
def observe_corrections(stats: Any, directory: Path, dense: Any, case: Case, variant: str):
    """Observe production calls and save every actual fallback entry state.

    Hook overhead is measured separately from the production correction. Hooks
    are local to a worker process and are restored even if evolution raises.
    """
    original_bug, original_evolution = bug.apply_conservation, evolution.apply_conservation
    observation = {
        "step": 0,
        "call_in_step": 0,
        "overhead_seconds": 0.0,
        "correction_seconds": 0.0,
        "fallback_seconds": 0.0,
        "events": [],
        "max_pre_drift": 0.0,
        "max_post_drift": 0.0,
    }

    def observed(state: MPS, mpo: Any, target: Any) -> bool:
        begin = time.perf_counter()
        before = deepcopy(state) if variant == "fallback" else None
        previous = stats.to_dict()
        observation["call_in_step"] += 1
        pre_energy = energy(canonical_vector(state), dense)
        core_begin = time.perf_counter()
        changed = conservation.apply_conservation(state, mpo, target)
        elapsed = time.perf_counter() - core_begin
        observation["correction_seconds"] += elapsed
        post_energy = energy(canonical_vector(state), dense)
        observation["max_pre_drift"] = max(observation["max_pre_drift"], abs(pre_energy - target.value))
        observation["max_post_drift"] = max(observation["max_post_drift"], abs(post_energy - target.value))
        if stats.fallback_attempts > previous["fallback_attempts"]:
            assert before is not None
            delta = {key: value - previous[key] for key, value in stats.to_dict().items()}
            index = len(observation["events"])
            filename = f"event-{index:05d}.npz"
            save_snapshot(directory / filename, before)
            before_vector, after_vector = canonical_vector(before), canonical_vector(state)
            stage = next(
                (
                    label
                    for counter, label in [
                        ("scan_successes", "scan"),
                        ("frame_successes", "frame"),
                        ("infeasible", "infeasible"),
                        ("no_minimizer", "no_minimizer"),
                        ("solver_failures", "solver_failure"),
                        ("gauge_failures", "gauge_failure"),
                    ]
                    if delta[counter]
                ),
                "unresolved",
            )
            observation["events"].append({
                "index": index,
                "step": observation["step"],
                "call_in_step": observation["call_in_step"],
                "step_end_time": observation["step"] * case.dt,
                "center": before.orthogonality_center,
                "flipped": before.flipped,
                "snapshot": filename,
                "target": target.value,
                "pre_energy": pre_energy,
                "post_energy": post_energy,
                "displacement_norm": float(np.linalg.norm(after_vector - before_vector)),
                "normalized_infidelity_to_input": infidelity(before_vector, after_vector),
                "seconds": elapsed,
                "stage": stage,
                "counts": delta,
                **scalar_diagnostics(before, mpo, target.value),
            })
            observation["fallback_seconds"] += elapsed
        observation["overhead_seconds"] += time.perf_counter() - begin - elapsed
        return changed

    bug.apply_conservation = evolution.apply_conservation = observed
    try:
        yield observation
    finally:
        bug.apply_conservation, evolution.apply_conservation = original_bug, original_evolution


def run_case(case: Case, output: str) -> dict[str, Any]:
    """Run three matched arms and atomically save their trajectories and events."""
    directory = Path(output) / "cases" / case.key
    directory.mkdir(parents=True, exist_ok=True)
    try:
        limit_worker_threads(1)
        safe_set_numba_threads(1)
        mpo, dense, model_metadata = models.hamiltonians(case.length, case.model, case.width_per_site)
        # Check all matrix entries, not just one expectation, to catch site-order errors.
        matrix_error = float(np.max(np.abs(mpo.to_matrix_mps_order() - dense.toarray())))
        if matrix_error > 1e-12:
            raise ValueError(f"Independent Hamiltonians disagree: {matrix_error}")
        initial_chi = min(case.cap, fx.INITIAL_CHI) if case.initial_chi is None else case.initial_chi
        if initial_chi < 1 or initial_chi > case.cap:
            raise ValueError("Initial bond dimension must be between 1 and the evolution cap")
        initial = fx.initial_state(case.length, case.initial, chi=initial_chi, seed=case.seed)
        initial_vector = initial.to_vec()
        save_snapshot(directory / "initial.npz", initial)
        target = conservation.expectation_value(initial, mpo)
        reference_energy = energy(initial_vector, dense)
        if abs(target - reference_energy) > 1e-12:
            raise ValueError("MPO and independent initial energies disagree")
        steps = round(case.total_time / case.dt)
        reference = expm_multiply(-1j * dense, initial_vector, start=0, stop=case.total_time, num=steps + 1)
        exact_energy_error = max(abs(energy(vector, dense) - reference_energy) for vector in reference)
        unpadded = (
            None
            if case.initial == "random_entangled"
            else fx.initial_state(case.length, case.initial, chi=1, seed=case.seed).to_vec()
        )
        result = {
            "schema": SCHEMA,
            "key": case.key,
            "case": asdict(case),
            "status": "complete",
            "initial_chi": initial_chi,
            "padding_noise_scale": None if unpadded is None else fx.NOISE_SCALE,
            "initial_infidelity_to_unpadded": None if unpadded is None else infidelity(unpadded, initial_vector),
            "initial_vector_sha256": hashlib.sha256(initial_vector.tobytes()).hexdigest(),
            "hamiltonian": model_metadata,
            "matrix_max_error": matrix_error,
            "target_energy": target,
            "reference_max_energy_drift": exact_energy_error,
            "arms": [],
        }
        for variant in VARIANTS:
            arm_directory = directory / variant
            arm_directory.mkdir(exist_ok=True)
            state = deepcopy(initial)
            params = AnalogSimParams(
                elapsed_time=case.dt,
                dt=case.dt,
                evolution_mode=EvolutionMode(case.integrator),
                max_bond_dim=case.cap,
                trunc_mode="relative_discarded_weight",
                svd_threshold=0.0,
                krylov_tol=case.krylov_tol,
                conserve_energy=variant != "uncorrected",
                conserve_fallback=variant == "fallback",
                conserve_tol=case.conserve_tol,
            )
            configured_target = (
                None
                if variant == "uncorrected"
                else conservation.EnergyTarget(target, case.conserve_tol, fallback=variant == "fallback")
            )
            energies, fidelities, bonds = [reference_energy], [0.0], [max(state.bond_dimensions())]
            evolution_seconds = 0.0
            started = time.perf_counter()
            with (
                track_conservation() as stats,
                observe_corrections(stats, arm_directory, dense, case, variant) as observation,
            ):
                for step in range(1, steps + 1):
                    observation["step"], observation["call_in_step"] = step, 0
                    tick = time.perf_counter()
                    evolution.apply_unitary_evolution(state, mpo, params, conservation_target=configured_target)
                    evolution_seconds += time.perf_counter() - tick
                    vector = state.to_vec()
                    if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0:
                        raise ValueError(f"Invalid state in {variant} at step {step}")
                    energies.append(energy(vector, dense))
                    fidelities.append(infidelity(reference[step], vector))
                    bonds.append(max(state.bond_dimensions()))
            counts = stats.to_dict()
            needed = counts["scalar_calls"] - counts["already_satisfied"]
            attempts = counts["fallback_attempts"]
            rejected = needed - counts["scalar_successes"]
            if variant != "uncorrected":
                expected = steps * (2 if case.integrator == "bug" else 1)
                assert counts["scalar_calls"] == expected, (counts, expected)
            assert max(bonds) <= case.cap
            assert len(observation["events"]) == attempts
            arm = {
                "variant": variant,
                "counts": counts,
                "scalar_rejections": rejected,
                "fallback_frequency": attempts / counts["scalar_calls"] if counts["scalar_calls"] else None,
                "fallback_frequency_when_needed": attempts / needed if needed else None,
                "wall_seconds": time.perf_counter() - started,
                "evolution_seconds": evolution_seconds - observation["overhead_seconds"],
                "observer_seconds": observation["overhead_seconds"],
                "correction_seconds": observation["correction_seconds"],
                "fallback_seconds": observation["fallback_seconds"],
                "max_energy_drift": max(abs(value - target) for value in energies),
                "max_pre_correction_drift": observation["max_pre_drift"] if variant != "uncorrected" else None,
                "max_post_correction_drift": observation["max_post_drift"] if variant != "uncorrected" else None,
                "final_infidelity": fidelities[-1],
                "peak_bond": max(bonds),
                "times": (np.arange(steps + 1) * case.dt).tolist(),
                "energies": energies,
                "infidelities": fidelities,
                "bonds": [int(value) for value in bonds],
                "events": observation["events"],
            }
            result["arms"].append(arm)
        write_json(directory / "result.json", result)
        return result
    except Exception:
        result = {"key": case.key, "case": asdict(case), "status": "error", "traceback": traceback.format_exc()}
        write_json(directory / "error.json", result)
        return result


def source_files() -> list[Path]:
    """List code whose hashes identify the implementation used in a run."""
    return sorted([
        *ROOT.joinpath("src").rglob("*.py"),
        *Path(__file__).parent.glob("fallback_*.py"),
        Path(__file__).with_name("fixtures_n.py"),
    ])


def initialize_output(output: Path, cases: list[Case]) -> dict[str, Any]:
    """Create or validate a resumable manifest and archive the executed source."""
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files()}
    signature = {"schema": SCHEMA, "cases": [asdict(case) for case in cases], "source_sha256": hashes}
    path = output / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["signature"] != signature:
            raise ValueError("Output belongs to a different grid or implementation; use a fresh directory")
        return previous
    output.mkdir(parents=True, exist_ok=True)
    versions = {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "numba", "qiskit"]}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    manifest = {
        "signature": signature,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_base_commit": revision.stdout.strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
        "variants": list(VARIANTS),
        "notes": "Hard bond cap; SVD threshold 0; fixture padding retained; references start from actual tensors. "
        "Rates count correction calls, not time steps. Timings exclude measured observer overhead, "
        "but are descriptive single-run measurements, not calibrated performance benchmarks.",
    }
    write_json(path, manifest)
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for source in source_files():
            archive.add(source, arcname=source.relative_to(ROOT))
    return manifest


def execute(cases: list[Case], output: Path, *, workers: int = 2) -> list[dict[str, Any]]:
    """Run missing matched triples; resume only exact matching manifests."""
    limit_worker_threads(1)
    safe_set_numba_threads(1)
    initialize_output(output, cases)
    completed, pending = [], []
    for case in cases:
        path = output / "cases" / case.key / "result.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["case"] != asdict(case) or result["status"] != "complete":
                raise ValueError(f"Invalid checkpoint: {path}")
            completed.append(result)
        else:
            pending.append(case)

    def consume(result: dict[str, Any]) -> None:
        completed.append(result)
        counts = result["arms"][-1]["counts"] if result["status"] == "complete" else {}
        print(
            f"[{len(completed)}/{len(cases)}] {result['key']} {result['status']} "
            f"fallback={counts.get('fallback_attempts', '?')} recovered={counts.get('fallback_successes', '?')}",
            flush=True,
        )

    if workers == 1:
        for case in pending:
            consume(run_case(case, str(output)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_case, case, str(output)) for case in pending]
            for future in as_completed(futures):
                consume(future.result())
    return sorted(completed, key=lambda result: result["key"])


def parse_args() -> argparse.Namespace:
    """Parse the pilot grid or a tolerance rerun of previously eventful cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[6])
    parser.add_argument("--models", nargs="+", choices=(*models.MODELS, "stress"), default=list(fx.MODELS))
    parser.add_argument("--caps", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument(
        "--initials", nargs="+", choices=fx.INITIAL_STATES, default=["tilted_neel", "domain_wall", "random_product"]
    )
    parser.add_argument("--dts", nargs="+", type=float, default=[0.02, 0.1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[fx.SEED])
    parser.add_argument("--total-time", type=float, default=2.0)
    parser.add_argument("--integrator", choices=["bug", "tdvp"], default="bug")
    parser.add_argument("--conserve-tol", type=float, default=1e-13)
    parser.add_argument("--krylov-tol", type=float, default=1e-14)
    parser.add_argument("--width-per-site", type=float, help="Normalize spectral width per site to this value")
    parser.add_argument("--initial-chi", type=int, help="Fixed initial bond dimension shared across evolution caps")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eventful-from", type=Path, help="Rerun eventful cases, changing only conserve-tol")
    return parser.parse_args()


def main() -> int:
    """Execute the grid; return failure if any trajectory failed."""
    args = parse_args()
    selected_models = list(
        dict.fromkeys(
            name for choice in args.models for name in (models.STRESS_MODELS if choice == "stress" else [choice])
        )
    )
    if args.eventful_from:
        cases = []
        for path in sorted((args.eventful_from / "cases").glob("*/result.json")):
            result = json.loads(path.read_text())
            if any(arm["counts"]["fallback_attempts"] for arm in result["arms"]):
                cases.append(replace(Case(**result["case"]), conserve_tol=args.conserve_tol))
    else:
        cases = [
            Case(
                model,
                length,
                cap,
                initial,
                dt,
                seed,
                args.total_time,
                args.integrator,
                args.conserve_tol,
                args.krylov_tol,
                args.width_per_site,
                args.initial_chi,
            )
            for model, length, cap, initial, dt, seed in itertools.product(
                selected_models, args.lengths, args.caps, args.initials, args.dts, args.seeds
            )
        ]
    for case in cases:
        if case.length < 2 or case.cap < 1 or case.dt <= 0 or case.total_time <= 0:
            raise ValueError("Require length >= 2 and positive cap, dt, and total time")
        if case.initial_chi is not None and not 1 <= case.initial_chi <= case.cap:
            raise ValueError("Initial chi must be between 1 and every bond cap")
        if case.width_per_site is not None and (not np.isfinite(case.width_per_site) or case.width_per_site <= 0):
            raise ValueError("Spectral width per site must be finite and positive")
        if not np.isclose(case.total_time / case.dt, round(case.total_time / case.dt), rtol=0, atol=1e-9):
            raise ValueError("Total time must be an integer multiple of dt")
    if args.workers < 1 or not cases or len({case.key for case in cases}) != len(cases):
        raise ValueError("Require workers >= 1 and a nonempty grid without duplicate cases")
    results = execute(cases, args.output.resolve(), workers=args.workers)
    return int(any(result["status"] != "complete" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
