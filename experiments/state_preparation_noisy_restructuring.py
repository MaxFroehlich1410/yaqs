# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Small-system noisy deletion-and-repair headroom test with compute-matched controls."""

from __future__ import annotations

import argparse
import copy
import json
import operator
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

from benchmarks.state_preparation.execution import WorkerSetting, completed_jobs, parse_workers, resolve_workers
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider, delete_gates, exact_density_fidelity
from benchmarks.state_preparation.targets import load_target
from benchmarks.state_preparation.weighted_sampling import FactorSampler
from experiments import state_preparation_exact_adam as exact
from experiments import state_preparation_pilot as pilot
from experiments import state_preparation_sampling_budget as budget
from experiments import state_preparation_trajectory_comparison as comparison
from experiments import state_preparation_trajectory_diagnostics as development
from mqt.yaqs.optimization import StatevectorBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

VERSION = "yaqs.noisy_restructuring_headroom.v1"
METHODS = ("adam", "adaptive_online")
T = TypeVar("T")


@dataclass(frozen=True)
class Config:
    """Predeclared scientific defaults; smaller values serve engineering tests."""

    warmup: int = 50
    online_updates: int = 25
    online_rate: float = 0.7
    rate: float = 0.02
    trajectories: int = 32
    rounds: int = 5
    shortlist: int = 2
    repair: int = 10
    control_max_updates: int = 4000
    root_seed: int = 2026091207
    acceptance_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        """Reject invalid schedules and budgets before creating artifacts."""
        positive = (self.warmup, self.rounds, self.shortlist, self.repair, self.control_max_updates)
        if (
            any(type(x) is not int or x < 1 for x in positive)
            or type(self.trajectories) is not int
            or self.trajectories < 2
            or type(self.online_updates) is not int
            or not 0 <= self.online_updates < self.warmup
            or self.control_max_updates < self.warmup
            or type(self.root_seed) is not int
            or self.root_seed < 0
            or any(
                isinstance(x, bool) or not np.isfinite(x) or x <= 0
                for x in (self.online_rate, self.rate, self.acceptance_tolerance)
            )
        ):
            msg = "Invalid restructuring configuration."
            raise ValueError(msg)


def load_cases(source_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Authenticate both old construction starts per development target."""
    provenance, starts = exact.load_starts(source_root)
    families = dict(zip(pilot.TARGETS, comparison.FAMILIES, strict=True))
    cases = []
    for start in starts:
        name = cast("str", start["target_id"])
        candidate = cast("dict[str, Any]", start["candidate"])
        cases.append({
            "target": comparison.target_record(name, families[name], load_target(6, name).state_vector_copy()),
            "replica": start["replica"],
            "circuit": candidate["circuit"],
            "initial": candidate["theta"],
            "expected_fidelity": candidate["exact_noisy_fidelity"],
            "source_candidate_checksum": pilot.digest(candidate),
        })
    return provenance, cases


def initial_state(case: dict[str, Any]) -> dict[str, Any]:
    """Create independent optimizer state for uniquely parameterized scalar gates."""
    circuit = pilot.restore_circuit(case["circuit"])
    indices = [g.param_index for g in circuit.gates]
    origins = [g.logical_gate_id for g in circuit.gates]
    theta = np.asarray(case["initial"], dtype=float)
    if (
        any(type(x) is not int for x in (*indices, *origins))
        or sorted(indices) != list(range(circuit.num_params))
        or len(set(origins)) != len(origins)
        or theta.shape != (circuit.num_params,)
        or not np.all(np.isfinite(theta))
    ):
        msg = "Restructuring requires unique parameters, stable integer origins and finite theta."
        raise ValueError(msg)
    return {
        "circuit": copy.deepcopy(case["circuit"]),
        "theta": theta.tolist(),
        "first": np.zeros_like(theta).tolist(),
        "second": np.zeros_like(theta).tolist(),
        "counts": np.zeros(len(theta), dtype=int).tolist(),
        "updates": 0,
    }


def deleted_state(state: dict[str, Any], index: int) -> dict[str, Any]:
    """Delete a gate and its noise window, preserving retained moments and counters."""
    circuit = pilot.restore_circuit(state["circuit"])
    compact, theta = delete_gates(circuit, np.asarray(state["theta"]), [index])
    retained = sorted(g.param_index for i, g in enumerate(circuit.gates) if i != index)
    result = copy.deepcopy(state)
    result.update({"circuit": pilot.circuit_record(compact), "theta": theta.tolist()})
    for key in ("first", "second", "counts"):
        result[key] = [state[key][i] for i in retained]
    return result


def fidelity(state: dict[str, Any], target: NDArray[np.complex128]) -> float:
    """Score the actual retained circuit using the independent exact channel oracle."""
    return exact_density_fidelity(
        pilot.restore_circuit(state["circuit"]), np.asarray(state["theta"]), target, ExactLocalPauliProvider()
    )


def train_block(
    state: dict[str, Any],
    target: NDArray[np.complex128],
    method: str,
    config: Config,
    seed: int,
    epoch_start: int,
    updates: int,
) -> dict[str, Any]:
    """Continue moments through a fresh weighted ensemble per update, then score once."""
    if method not in METHODS or updates < 1:
        msg = "Unknown optimizer or empty training block."
        raise ValueError(msg)
    result = copy.deepcopy(state)
    engine = StatevectorBackend(pilot.restore_circuit(state["circuit"]), target)
    sampler = FactorSampler(engine.circuit, ExactLocalPauliProvider().calibration)
    theta, first, second = (np.asarray(state[k], dtype=float) for k in ("theta", "first", "second"))
    counts = np.asarray(state["counts"], dtype=int)
    trace = []
    for offset in range(1, updates + 1):
        step = state["updates"] + offset
        epoch = epoch_start + offset
        maps, weights = sampler.ensemble(config.trajectories, seed + 1_000_003 * epoch)
        if method == "adam":
            signal = engine.gradient(theta, maps, weights=weights)[0]
            theta, first, second = exact.adam_step(
                theta, signal, first, second, step, exact.AdamConfig(learning_rate=config.rate)
            )
            phase = "adam"
        elif step <= config.online_updates:
            theta, signal = engine.online_update(theta, config.online_rate, maps, weights=weights)
            phase = "online"
        else:
            theta, signal, second, counts = engine.adaptive_online_update(
                theta, config.rate, second, counts, maps, weights=weights
            )
            phase = "adaptive_online"
        if not all(np.all(np.isfinite(a)) for a in (theta, first, second, signal)):
            msg = "Nonfinite restructuring update."
            raise FloatingPointError(msg)
        trace.append({
            "step": step,
            "epoch": epoch,
            "phase": phase,
            "ensemble_digest": development.ensemble_digest(maps, weights),
            "signal_norm": float(np.linalg.norm(signal)),
            "trajectory_passes": config.trajectories * (2 if method == "adam" else 3),
        })
    result.update({
        "theta": theta.tolist(),
        "first": first.tolist(),
        "second": second.tolist(),
        "counts": counts.tolist(),
        "updates": state["updates"] + updates,
    })
    return {
        "state": result,
        "fidelity": fidelity(result, target),
        "trace": trace,
        "no_factor_probability": sampler.no_factor_probability,
        "trajectory_passes": sum(t["trajectory_passes"] for t in trace),
    }


def rank_deletions(state: dict[str, Any], target: NDArray[np.complex128]) -> list[dict[str, Any]]:
    """Exhaustively score finite noisy deletions, protecting the original product layer."""
    circuit = pilot.restore_circuit(state["circuit"])
    rows = []
    for index, gate in enumerate(circuit.gates):
        if type(gate.logical_gate_id) is not int:
            msg = "Expected stable integer gate origins."
            raise ValueError(msg)
        if gate.logical_gate_id < 3 * circuit.num_qubits:
            continue
        rows.append({
            "index": index,
            "origin": gate.logical_gate_id,
            "arity": len(gate.sites),
            "fidelity": fidelity(deleted_state(state, index), target),
        })
    return sorted(rows, key=lambda r: (-round(r["fidelity"] / 1e-12), r["origin"]))


def choose_candidate(candidates: list[dict[str, Any]], tolerance: float) -> int:
    """Prefer the unchanged input, then repaired unchanged circuit, on numerical ties."""
    chosen = 0
    for index in range(1, len(candidates)):
        if candidates[index]["fidelity"] > candidates[chosen]["fidelity"] + tolerance:
            chosen = index
    return chosen


def charged(function: Callable[[], T], ledger: list[dict[str, Any]], role: str) -> T:
    """Charge complete decision operations, including discarded candidate work."""
    value, cost = development.measured(function)
    ledger.append({"role": role, **cost})
    return value


def cpu_cost(ledger: list[dict[str, Any]]) -> float:
    """Return cumulative CPU including exact scoring, sampling, setup and repair."""
    return sum(row["cpu_seconds"] for row in ledger)


def point(state: dict[str, Any], score: float, ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Store a deployable incumbent and its availability time, not an interpolated iterate."""
    return {
        "state": copy.deepcopy(state),
        "fidelity": score,
        "cpu_seconds": cpu_cost(ledger),
        "completed_updates": state["updates"],
    }


def restructure(
    prefix: dict[str, Any],
    target: NDArray[np.complex128],
    method: str,
    config: Config,
    seed: int,
    *,
    ranker: Callable[[dict[str, Any], NDArray[np.complex128]], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Try two screened deletions and an unchanged repair per round, with exact acceptance."""
    ledger = copy.deepcopy(prefix["ledger"])
    state, score = copy.deepcopy(prefix["state"]), prefix["fidelity"]
    curve = copy.deepcopy(prefix["curve"])
    rounds = []
    for iteration in range(config.rounds):
        ranking = charged(
            partial(rank_deletions if ranker is None else ranker, state, target), ledger, "deletion_scoring"
        )
        candidates: list[dict[str, Any]] = [{"origin": None, "kind": "retain", "state": state, "fidelity": score}]
        choices = [(None, None), *((r["index"], r["origin"]) for r in ranking[: config.shortlist])]
        for index, origin in choices:

            def repair(
                index: int | None = index, state: dict[str, Any] = state, iteration: int = iteration
            ) -> dict[str, Any]:
                start = state if index is None else deleted_state(state, index)
                return train_block(
                    start, target, method, config, seed, config.warmup + iteration * config.repair, config.repair
                )

            trial = charged(repair, ledger, "unchanged_repair" if index is None else "deletion_repair")
            candidates.append({"origin": origin, "kind": "repair" if index is None else "delete", **trial})
        selected = choose_candidate(candidates, config.acceptance_tolerance)
        state, score = candidates[selected]["state"], candidates[selected]["fidelity"]
        rounds.append({"round": iteration + 1, "ranking": ranking, "candidates": candidates, "selected": selected})
        curve.append(point(state, score, ledger))
    return {"state": state, "fidelity": score, "ledger": ledger, "curve": curve, "rounds": rounds}


def fixed_control(
    prefix: dict[str, Any],
    target: NDArray[np.complex128],
    method: str,
    config: Config,
    seed: int,
    budget_seconds: float,
    *,
    epoch_start: int | None = None,
    role: str = "fixed_repair",
) -> dict[str, Any]:
    """Continue a fixed circuit until the shared CPU cutoff, recording best scored endpoints.

    Continue the optimization trajectory even after an unfavorable score. Only
    the exported incumbent is selected by exact fidelity, matching the structural
    arm's access to exact acceptance. The crossing block is not available at the
    smaller cutoff, and a hard update cap is explicitly reported.
    """
    ledger = copy.deepcopy(prefix["ledger"])
    state = copy.deepcopy(prefix["state"])
    epoch = state["updates"] if epoch_start is None else epoch_start
    if type(epoch) is not int or epoch < state["updates"]:
        msg = "Continuation epochs must not replay previously consumed updates."
        raise ValueError(msg)
    curve = copy.deepcopy(prefix["curve"])
    incumbent = max(curve, key=operator.itemgetter("fidelity"))
    blocks = []
    while cpu_cost(ledger) <= budget_seconds and state["updates"] < config.control_max_updates:
        count = min(config.repair, config.control_max_updates - state["updates"])
        block = charged(partial(train_block, state, target, method, config, seed, epoch, count), ledger, role)
        state = block["state"]
        epoch += count
        blocks.append(block)
        if block["fidelity"] > incumbent["fidelity"] + config.acceptance_tolerance:
            incumbent = block
        curve.append({
            **point(incumbent["state"], incumbent["fidelity"], ledger),
            "completed_updates": state["updates"],
        })
    return {
        "state": incumbent["state"],
        "last_state": state,
        "fidelity": incumbent["fidelity"],
        "curve": curve,
        "ledger": ledger,
        "blocks": blocks,
        "covered_budget": cpu_cost(ledger) > budget_seconds,
        "budget_seconds": budget_seconds,
    }


def at_budget(arm: dict[str, Any], seconds: float) -> dict[str, Any]:
    """Return only a completed incumbent that was available within the CPU cutoff."""
    eligible = [p for p in arm["curve"] if p["cpu_seconds"] <= seconds]
    if not eligible:
        msg = "CPU cutoff precedes the first scored circuit."
        raise ValueError(msg)
    return eligible[-1]


@dataclass(frozen=True)
class Job:
    """One job owns all four arms for one target/construction start."""

    root: Path
    case: dict[str, Any]
    config: Config
    manifest_checksum: str


def run_job(job: Job) -> dict[str, Any]:
    """Complete or authenticate an atomic case; interrupted cases are rerun from their starts."""
    case, config = job.case, job.config
    name, replica = case["target"]["target_id"], case["replica"]
    relative = f"{name}/seed_{replica}.json"
    path = job.root / relative
    signature = pilot.digest([job.manifest_checksum, case, asdict(config)])
    if path.exists():
        record = cast("dict[str, Any]", pilot.read_record(path))
        if record["signature"] != signature or record["status"] != "complete":
            msg = "Completed case signature/status mismatch."
            raise ValueError(msg)
    else:
        target = comparison.target_vector(case["target"])
        start = initial_state(case)
        seed = int(pilot.digest([VERSION, config.root_seed, name, replica])[:16], 16) << 32
        prefixes, structural = {}, {}
        # Alternate method order independently of target amplitudes or outcomes.
        method_order = METHODS if replica % 4 == 3 else METHODS[::-1]
        for method in method_order:
            ledger: list[dict[str, Any]] = []
            initial_score = charged(lambda: fidelity(start, target), ledger, "initial_score")
            if abs(initial_score - case["expected_fidelity"]) > 1e-10:
                msg = "Construction source fidelity reproduction failed."
                raise ValueError(msg)
            curve = [point(start, initial_score, ledger)]
            warm = charged(
                partial(train_block, start, target, method, config, seed, 0, config.warmup), ledger, "warmup"
            )
            curve.append(point(warm["state"], warm["fidelity"], ledger))
            prefixes[method] = {**warm, "ledger": ledger, "curve": curve}
            structural[method] = restructure(prefixes[method], target, method, config, seed)
        cutoff = max(cpu_cost(arm["ledger"]) for arm in structural.values())
        fixed = {
            method: fixed_control(prefixes[method], target, method, config, seed, cutoff) for method in method_order
        }
        record = {
            "status": "complete",
            "signature": signature,
            "target_id": name,
            "family": case["target"]["family"],
            "replica": replica,
            "seed": seed,
            "prefixes": prefixes,
            "structural": structural,
            "fixed": fixed,
        }
        pilot.write_record(path, record)
    return {"target_id": name, "replica": replica, "path": relative, "checksum": pilot.digest(record)}


def summarize(root: Path, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute matched-cutoff comparisons, retaining all endpoints and safety-cap flags."""
    rows = []
    for job in jobs:
        record = cast("dict[str, Any]", pilot.read_record(root / job["path"]))
        if pilot.digest(record) != job["checksum"]:
            msg = "Case changed during aggregation."
            raise ValueError(msg)
        for method in METHODS:
            arm = record["structural"][method]
            seconds = cpu_cost(arm["ledger"])
            row = {
                "target_id": record["target_id"],
                "family": record["family"],
                "replica": record["replica"],
                "method": method,
                "fidelity": arm["fidelity"],
                "cpu_seconds": seconds,
                "gates": len(arm["state"]["circuit"]["gates"]),
                "two_qubit_gates": sum(len(g["sites"]) == 2 for g in arm["state"]["circuit"]["gates"]),
                "deleted_origins": [
                    r["candidates"][r["selected"]]["origin"]
                    for r in arm["rounds"]
                    if r["candidates"][r["selected"]]["kind"] == "delete"
                ],
            }
            for control in METHODS:
                fixed = record["fixed"][control]
                point_at_cutoff = at_budget(fixed, seconds)
                row[f"fixed_{control}_fidelity"] = point_at_cutoff["fidelity"]
                row[f"fixed_{control}_updates"] = point_at_cutoff["completed_updates"]
                row[f"fixed_{control}_selected_update"] = point_at_cutoff["state"]["updates"]
                row[f"fixed_{control}_covered"] = fixed["covered_budget"]
            rows.append(row)
    return rows


def run(
    root: Path,
    source_root: Path,
    *,
    config: Config | None = None,
    workers: WorkerSetting = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    """Freeze provenance and run eight paired cases without new targets or rate tuning."""
    config = Config() if config is None else config
    resolve_workers(workers, 1)
    source, cases = load_cases(source_root)
    manifest = json.loads(
        json.dumps({
            "version": VERSION,
            "config": asdict(config),
            "source_root": str(source_root.resolve()),
            "source": source,
            "cases": cases,
            "runtime": budget.runtime(),
            "driver": pilot.digest(Path(__file__).read_text(encoding="utf-8")),
            "noise": ExactLocalPauliProvider().scientific_dict(),
            "protocol": pilot.digest(Path(__file__).with_suffix(".md").read_text(encoding="utf-8")),
        })
    )
    if root.exists():
        if not resume or not (root / "manifest.json").exists() or pilot.read_record(root / "manifest.json") != manifest:
            msg = "Existing output requires explicit resume with an identical manifest."
            raise ValueError(msg)
    else:
        if resume:
            msg = "Cannot resume missing output."
            raise ValueError(msg)
        pilot.write_record(root / "manifest.json", manifest)
    identity = pilot.digest(manifest)
    jobs = [Job(root, case, config, identity) for case in cases]
    order = np.random.default_rng(config.root_seed).permutation(len(jobs))
    completed = {}
    for index, row in completed_jobs(run_job, [jobs[i] for i in order], workers=workers):
        completed[int(order[index])] = row
        print(f"noisy restructuring {len(completed)}/{len(jobs)}: {row['target_id']} seed {row['replica']}", flush=True)
    result: dict[str, Any] = {
        "status": "complete",
        "manifest_checksum": identity,
        "jobs": [completed[i] for i in range(len(jobs))],
    }
    result["rows"] = summarize(root, result["jobs"])
    path = root / "result.json"
    if path.exists():
        if pilot.read_record(path) != result:
            msg = "Completed result differs from authenticated cases."
            raise ValueError(msg)
    else:
        pilot.write_record(path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the predeclared screen, with case-level atomic resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=parse_workers, default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run(args.output_root, args.source_root, workers=args.workers, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
