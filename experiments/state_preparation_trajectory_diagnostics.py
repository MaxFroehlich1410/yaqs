# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Development-only statevector gradient, scheduling and adaptive-tail studies."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from benchmarks.state_preparation.execution import WorkerSetting, completed_jobs, parse_workers, resolve_workers
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider, exact_density_fidelity
from benchmarks.state_preparation.targets import load_target
from benchmarks.state_preparation.weighted_sampling import FactorSampler
from experiments import state_preparation_exact_adam as exact
from experiments import state_preparation_pilot as pilot
from experiments import state_preparation_trajectory_comparison as comparison
from mqt.yaqs.optimization import KrotovTJMOptions, StatevectorBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization.statevector import Estimator, NoiseMaps

VERSION = "yaqs.statevector_development_studies.v1"
STUDIES = ("gradient", "tail", "crn", "backtracking", "adaptive")
T = TypeVar("T")


def measured(function: Callable[[], T]) -> tuple[T, dict[str, float]]:
    """Measure CPU and wall time of one explicitly scoped operation."""
    cpu, wall = time.process_time(), time.perf_counter()
    value = function()
    return value, {"cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - wall}


@dataclass(frozen=True)
class Config:
    """Frozen defaults; smaller budgets are exposed for engineering tests only."""

    updates: int = 50
    online_updates: int = 25
    online_rate: float = 0.7
    trajectories: int = 32
    diagnostic_counts: tuple[int, ...] = (8, 32, 128)
    diagnostic_repeats: int = 8
    checkpoints: tuple[int, ...] = (0, 25, 50)
    replicas: tuple[int, ...] = (11, 29)
    root_seed: int = 2026091101
    max_backtracks: int = 6
    contraction: float = 0.5
    armijo: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        """Reject malformed budgets before creating output."""
        positive = (self.updates, self.trajectories, self.diagnostic_repeats, *self.diagnostic_counts)
        counters = (self.online_updates, self.root_seed, self.max_backtracks, *self.checkpoints, *self.replicas)
        if (
            any(type(x) is not int or x < 1 for x in positive)
            or any(type(x) is not int or x < 0 for x in counters)
            or not 0 <= self.online_updates <= self.updates
            or self.diagnostic_repeats < 2
            or any(not x or len(set(x)) != len(x) for x in (self.diagnostic_counts, self.checkpoints, self.replicas))
            or any(
                isinstance(x, bool) or not np.isfinite(x)
                for x in (self.online_rate, self.contraction, self.armijo, self.beta1, self.beta2, self.epsilon)
            )
            or self.online_rate <= 0
            or not 0 < self.contraction < 1
            or not 0 < self.armijo < 1
            or not 0 <= self.beta1 < 1
            or not 0 <= self.beta2 < 1
            or self.epsilon <= 0
        ):
            msg = "Invalid study configuration."
            raise ValueError(msg)


@dataclass(frozen=True)
class Variant:
    """One predeclared schedule; refresh=0 freezes maps, otherwise counts updates."""

    name: str
    online_updates: int
    tail: str = "batch"
    rate: float = 0.1
    refresh: int = 1

    def __post_init__(self) -> None:
        """Validate labels, sampling policies and update rules."""
        if (
            not self.name
            or not self.name.replace("_", "").isalnum()
            or type(self.online_updates) is not int
            or self.online_updates < 0
            or type(self.refresh) is not int
            or self.refresh < 0
            or self.tail not in {"batch", "adam", "momentum", "rmsprop", "backtracking", "online_rmsprop"}
            or isinstance(self.rate, bool)
            or not np.isfinite(self.rate)
            or self.rate <= 0
        ):
            msg = "Invalid study variant."
            raise ValueError(msg)

    def epoch(self, step: int) -> int:
        """Return the first one-based update sharing this ensemble."""
        if type(step) is not int or step < 1:
            msg = "Noise step must be a positive integer."
            raise ValueError(msg)
        return 1 if self.refresh == 0 else 1 + ((step - 1) // self.refresh) * self.refresh


def variants(study: str, config: Config) -> tuple[Variant, ...]:
    """Return controlled screens, with no automatic promotion or winner selection."""
    baseline = Variant("hybrid_batch_0p1", config.online_updates)
    if study == "tail":
        return (
            baseline,
            Variant("hybrid_batch_0p2", config.online_updates, rate=0.2),
            Variant("hybrid_batch_0p3", config.online_updates, rate=0.3),
            Variant("all_online", config.updates),
        )
    if study == "crn":
        return (
            baseline,
            Variant("refresh_five", config.online_updates, refresh=5),
            Variant("fixed_crn", config.online_updates, refresh=0),
        )
    if study == "backtracking":
        return (baseline, Variant("crn_backtracking", config.online_updates, tail="backtracking", rate=0.3))
    if study == "adaptive":
        return (
            baseline,
            Variant("krotov_momentum", config.online_updates, tail="momentum"),
            Variant("krotov_rmsprop", config.online_updates, tail="rmsprop", rate=0.005),
            Variant("krotov_adam", config.online_updates, tail="adam", rate=0.005),
            Variant("trajectory_adam", 0, tail="adam", rate=0.005),
        )
    msg = "Unknown refinement study."
    raise ValueError(msg)


def seed_for(config: Config, role: str, case: str, replica: int, checkpoint: int = 0) -> int:
    """Pair variants/estimators/counts, separating diagnostic and training streams."""
    return int(pilot.digest([VERSION, config.root_seed, role, case, replica, checkpoint])[:16], 16) << 32


def load_cases(source_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Authenticate old Experiment-2 development checkpoints without changing them.

    Only the predeclared trajectory-hybrid rate 0.7 is imported, not a new
    nomination on confirmation targets. Earlier MPS source fingerprints remain
    provenance, not a requirement to reproduce that software environment.
    """
    manifest = pilot.read_record(source_root / "manifest.json")
    result = pilot.read_record(source_root / "result.json")
    selection = pilot.read_record(source_root / "selection.json")
    source_config = cast("dict[str, Any]", manifest["config"])
    if (
        manifest["version"] != comparison.VERSION
        or result["status"] != "complete"
        or result["manifest_checksum"] != pilot.digest(manifest)
        or result["selection_checksum"] != pilot.digest(selection)
        or manifest["noise"] != ExactLocalPauliProvider().scientific_dict()
        or {k: source_config[k] for k in ("updates", "switch", "trajectories")}
        != {"updates": 50, "switch": 25, "trajectories": 32}
    ):
        msg = "Expected a complete compatible Experiment-2 source."
        raise ValueError(msg)
    provenance, starts = exact.load_starts(Path(cast("str", manifest["source_root"])))
    if manifest["source_provenance"] != provenance:
        msg = "Original construction source provenance changed."
        raise ValueError(msg)
    wanted = asdict(comparison.Method("trajectory_hybrid", 0.7))
    rows = [r for r in cast("list[dict[str, Any]]", result["development"]) if r["method"] == wanted]
    if len(rows) != 4 or {(r["target_id"], r["replica"]) for r in rows} != {(t, 11) for t in pilot.TARGETS}:
        msg = "Expected four distinct development targets at the frozen hybrid setting."
        raise ValueError(msg)
    cases, checksums = [], {}
    for target_id, family in zip(pilot.TARGETS, comparison.FAMILIES, strict=True):
        row = next(r for r in rows if r["target_id"] == target_id)
        relative = f"development/{target_id}/seed_11/trajectory_hybrid_r0p7_t0p1.json"
        state = cast("dict[str, Any]", pilot.read_record(source_root / relative))
        start = cast(
            "dict[str, Any]", next(s["candidate"] for s in starts if s["target_id"] == target_id and s["replica"] == 11)
        )
        target = comparison.target_record(
            target_id, family, load_target(6, target_id).state_vector_copy(), role="development"
        )
        signature = pilot.digest({
            "start": start,
            "target": target,
            "method": wanted,
            "config": source_config,
            "provenance": pilot.digest(manifest),
            "seed": comparison.seed_for("development", target_id, 11),
            "version": comparison.VERSION,
        })
        evaluations = state["evaluations"]
        if (
            row["path"] != relative
            or row["family"] != family
            or row["checksum"] != pilot.digest(state)
            or state["signature"] != signature
            or state["config"] != source_config
            or state["method"] != wanted
            or state["status"] != "complete"
            or state["completed_updates"] != 50
            or [r["step"] for r in state["trace"]] != list(range(1, 51))
            or [e["update"] for e in evaluations] != [0, 10, 25, 50]
            or evaluations[0]["theta"] != start["theta"]
            or evaluations[-1]["theta"] != state["theta"]
        ):
            msg = "Source development checkpoint audit failed."
            raise ValueError(msg)
        checksums[relative] = pilot.digest(state)
        cases.append({
            "target": target,
            "circuit": start["circuit"],
            "initial": start["theta"],
            "checkpoints": evaluations,
        })
    return {"manifest": pilot.digest(manifest), "result": pilot.digest(result), "checkpoints": checksums}, cases


def noise_events(maps: NoiseMaps) -> dict[str, int]:
    """Count realized Pauli factors and trajectories with no sampled factors."""
    counts = [sum(len(m.operators) for m in row) for row in maps]
    return {"sampled_factors": sum(counts), "event_free_trajectories": counts.count(0)}


def ensemble_digest(maps: NoiseMaps, weights: NDArray[np.float64] | None) -> str:
    """Hash physical replay operators and weights, excluding parameter metadata."""
    return pilot.digest({
        "count": len(maps),
        "weights": None if weights is None else weights.tolist(),
        "operators": [
            [r, k, sites, matrix.real.tolist(), matrix.imag.tolist()]
            for r, row in enumerate(maps)
            for k, noise_map in enumerate(row)
            for matrix, sites in noise_map.operators
        ],
    })


def gradient_errors(estimate: NDArray[np.float64], reference: NDArray[np.float64]) -> dict[str, Any]:
    """Return scale and direction diagnostics, using null for undefined ratios."""
    error = float(np.linalg.norm(estimate - reference))
    norm, estimated_norm = float(np.linalg.norm(reference)), float(np.linalg.norm(estimate))
    dot = float(estimate @ reference)
    return {
        "absolute_error": error,
        "squared_error": error**2,
        "reference_norm": norm,
        "estimated_norm": estimated_norm,
        "relative_error": error / norm if norm > 1e-14 else None,
        "cosine": float(np.clip(dot / (norm * estimated_norm), -1, 1)) if min(norm, estimated_norm) > 1e-14 else None,
        "descent_dot": dot,
    }


def diagnostic(
    engine: StatevectorBackend,
    theta: NDArray[np.float64],
    config: Config,
    case_id: str,
    checkpoint: int,
    *,
    include_cross: bool = False,
) -> dict[str, Any]:
    """Compare paired repeated ensembles with a separately timed exact gradient."""
    (reference, loss, fidelity), oracle_cost = measured(
        lambda: density_loss_gradient(engine.circuit, theta, engine.target, CalibratedLocalDepolarizingProvider())
    )
    rows: list[dict[str, Any]] = []
    estimators: tuple[Estimator, ...] = ("independent", "cross") if include_cross else ("independent",)
    provider = ExactLocalPauliProvider()
    for count in config.diagnostic_counts:
        for repeat in range(config.diagnostic_repeats):
            options = KrotovTJMOptions(
                num_trajectories=count, random_seed=seed_for(config, "diagnostic", case_id, repeat, checkpoint)
            )
            maps, sampling_cost = measured(partial(engine.sample_noise_maps, theta, options, provider))
            # Alternate estimator order to avoid always assigning first-call cost to one method.
            for estimator in estimators[:: -1 if repeat % 2 else 1]:
                (gradient, sample_loss, _), cost = measured(partial(engine.gradient, theta, maps, estimator=estimator))
                rows.append({
                    "trajectories": count,
                    "repeat": repeat,
                    "estimator": estimator,
                    "gradient": gradient.tolist(),
                    "sample_loss": sample_loss,
                    **gradient_errors(gradient, reference),
                    **noise_events(maps),
                    "sampling": sampling_cost,
                    "gradient_cost": cost,
                    "cpu_seconds": sampling_cost["cpu_seconds"] + cost["cpu_seconds"],
                })
    aggregates = []
    for count in config.diagnostic_counts:
        for estimator in estimators:
            subset = [r for r in rows if r["trajectories"] == count and r["estimator"] == estimator]
            gradients = np.array([r["gradient"] for r in subset])
            mse = float(np.mean([r["squared_error"] for r in subset]))
            cpu = float(np.mean([r["cpu_seconds"] for r in subset]))
            aggregates.append({
                "trajectories": count,
                "estimator": estimator,
                "mse": mse,
                "bias_norm": float(np.linalg.norm(gradients.mean(axis=0) - reference)),
                "variance_trace": float(np.var(gradients, axis=0, ddof=1).sum()),
                "mean_cpu_seconds": cpu,
                "mse_times_cpu": mse * cpu,
                "mean_cosine": nullable_mean([r["cosine"] for r in subset]),
                "mean_relative_error": nullable_mean([r["relative_error"] for r in subset]),
                "descent_fraction": float(np.mean([r["descent_dot"] > 0 for r in subset])),
            })
    return {
        "reference_gradient": reference.tolist(),
        "exact_loss": loss,
        "exact_fidelity": fidelity,
        "oracle_cost": oracle_cost,
        "rows": rows,
        "aggregate": aggregates,
    }


def nullable_mean(values: list[float | None]) -> float | None:
    """Average only defined diagnostics, preserving undefined zero-gradient cases."""
    defined = [x for x in values if x is not None]
    return float(np.mean(defined)) if defined else None


def tail_update(
    engine: StatevectorBackend,
    theta: NDArray[np.float64],
    maps: NoiseMaps,
    variant: Variant,
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    tail_step: int,
    config: Config,
    *,
    weights: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    """Apply one tail step; Adam/EMA moments start at zero at the phase switch.

    Backtracking uses the same maps for all trial losses and never consults
    exact evaluation. Its budget is max_backtracks+1 trials; the gradient
    forward pass already supplies the baseline loss without another evaluation.
    """
    if variant.tail == "online_rmsprop":
        msg = "Adaptive online tails require the gate-sequential updater."
        raise ValueError(msg)
    gradient, before, _ = (
        engine.gradient(theta, maps) if weights is None else engine.gradient(theta, maps, weights=weights)
    )
    info: dict[str, Any] = {
        "signal_norm": float(np.linalg.norm(gradient)),
        "accepted": True,
        "rate": variant.rate,
        "objective_evaluations": 0,
        "trial_rates": [],
    }
    if variant.tail == "adam":
        updated, first, second = exact.adam_step(
            theta,
            gradient,
            first,
            second,
            tail_step,
            exact.AdamConfig(
                updates=config.updates,
                learning_rate=variant.rate,
                beta1=config.beta1,
                beta2=config.beta2,
                epsilon=config.epsilon,
            ),
        )
    elif variant.tail == "momentum":
        first = config.beta1 * first + (1 - config.beta1) * gradient
        updated = theta - variant.rate * first
    elif variant.tail == "rmsprop":
        second = config.beta2 * second + (1 - config.beta2) * gradient**2
        updated = theta - variant.rate * gradient / (np.sqrt(second / (1 - config.beta2**tail_step)) + config.epsilon)
    elif variant.tail == "backtracking":
        info.update({"before_sample_loss": before, "accepted": False, "rate": 0.0})
        updated = theta.copy()
        for attempt in range(config.max_backtracks + 1):
            rate = variant.rate * config.contraction**attempt
            candidate = theta - rate * gradient
            loss, _ = (
                engine.metrics(candidate, maps) if weights is None else engine.metrics(candidate, maps, weights=weights)
            )
            info["objective_evaluations"] += 1
            info["trial_rates"].append(rate)
            if loss <= before - config.armijo * rate * float(gradient @ gradient):
                updated = candidate
                info.update({"accepted": True, "rate": rate, "after_sample_loss": loss})
                break
        if not info["accepted"]:
            info["after_sample_loss"] = before
    else:
        updated = theta - variant.rate * gradient
    return updated, first, second, info


def optimize(
    path: Path,
    case: dict[str, Any],
    variant: Variant,
    config: Config,
    replica: int,
    provenance: str,
    *,
    stop_after: int | None = None,
    sampling: str = "ordinary",
) -> dict[str, Any]:
    """Run or resume a dense refinement, storing every completed update atomically."""
    limit = config.updates if stop_after is None else stop_after
    if sampling not in {"ordinary", "weighted"} or (sampling == "weighted" and config.trajectories < 2):
        msg = "Unknown sampling policy or fewer than two weighted paths."
        raise ValueError(msg)
    if type(limit) is not int or not 0 <= limit <= config.updates or variant.online_updates > config.updates:
        msg = "Invalid refinement budget."
        raise ValueError(msg)
    signature = pilot.digest({
        "case": case,
        "variant": asdict(variant),
        "config": asdict(config),
        "replica": replica,
        "provenance": provenance,
        "version": VERSION,
        "sampling": sampling,
        "update_kernel": "weighted_online_rms.v1",
    })
    initial = np.asarray(case["initial"], dtype=float)
    state: dict[str, Any] = {
        "signature": signature,
        "theta": initial.tolist(),
        "first": np.zeros_like(initial).tolist(),
        "second": np.zeros_like(initial).tolist(),
        "online_counts": np.zeros(len(initial), dtype=int).tolist(),
        "completed": 0,
        "tail_step": 0,
        "trace": [],
        "evaluations": [],
        "setup": [],
        "status": "running",
    }
    if path.exists():
        state = cast("dict[str, Any]", pilot.read_record(path))
        if state["signature"] != signature:
            msg = "Refinement checkpoint configuration mismatch."
            raise ValueError(msg)
    completed = state["completed"]
    theta, first, second = (np.asarray(state[k], dtype=float) for k in ("theta", "first", "second"))
    online_counts = np.asarray(state["online_counts"])
    occurrences = np.bincount(
        [g["param_index"] for g in case["circuit"]["gates"] if g["param_index"] is not None], minlength=len(initial)
    )
    if (
        type(completed) is not int
        or not 0 <= completed <= config.updates
        or state["tail_step"] != max(0, completed - variant.online_updates)
        or [r["step"] for r in state["trace"]] != list(range(1, completed + 1))
        or any(r["epoch"] != variant.epoch(r["step"]) for r in state["trace"])
        or any(a.shape != initial.shape or not np.all(np.isfinite(a)) for a in (theta, first, second))
        or np.any(second < 0)
        or online_counts.dtype.kind not in "iu"
        or not np.array_equal(
            online_counts,
            occurrences * (max(0, completed - variant.online_updates) if variant.tail == "online_rmsprop" else 0),
        )
        or state["status"] not in {"running", "complete"}
        or (state["status"] == "complete") != (completed == config.updates)
        or (state["evaluations"] and [e["step"] for e in state["evaluations"]] != list(range(completed + 1)))
        or (completed > 0 and not state["evaluations"])
    ):
        msg = "Invalid refinement checkpoint progress."
        raise ValueError(msg)
    if state["status"] == "complete":
        return state
    engine, setup = measured(
        lambda: StatevectorBackend(pilot.restore_circuit(case["circuit"]), comparison.target_vector(case["target"]))
    )
    state["setup"].append(setup)
    seed = seed_for(config, "training", case["target"]["target_id"], replica)
    options = KrotovTJMOptions(num_trajectories=config.trajectories, random_seed=seed)
    provider = ExactLocalPauliProvider()
    sampler = None
    weights = None
    if sampling == "weighted":
        sampler, sampler_setup = measured(lambda: FactorSampler(engine.circuit, provider.calibration))
        state["setup"].append({"factor_sampler": True, **sampler_setup})

    def sample(epoch: int) -> tuple[NoiseMaps, NDArray[np.float64] | None]:
        """Use method-independent streams and the same ensemble throughout each update."""
        if sampler is not None:
            return sampler.ensemble(config.trajectories, seed + 1_000_003 * epoch)
        return engine.sample_noise_maps(initial, options, provider, iteration=epoch), None

    maps, cached_epoch = None, None
    if completed and completed < limit and variant.epoch(completed) == variant.epoch(completed + 1):
        cached_epoch = variant.epoch(completed)
        (maps, weights), replay = measured(partial(sample, cached_epoch))
        state["setup"].append({"replay": True, **replay})

    def evaluate(step: int) -> None:
        fidelity, cost = measured(lambda: exact_density_fidelity(engine.circuit, theta, engine.target, provider))
        state["evaluations"].append({"step": step, "fidelity": fidelity, **cost})

    if not state["evaluations"]:
        evaluate(0)
        if abs(state["evaluations"][0]["fidelity"] - case["checkpoints"][0]["exact_noisy_fidelity"]) > 1e-10:
            msg = "Source initial fidelity reproduction failed."
            raise ValueError(msg)
    for step in range(completed + 1, limit + 1):
        epoch = variant.epoch(step)
        sampling_cost = {"cpu_seconds": 0.0, "wall_seconds": 0.0}
        sampled = epoch != cached_epoch
        if sampled:
            (maps, weights), sampling_cost = measured(partial(sample, epoch))
            cached_epoch = epoch
        assert maps is not None
        before = theta.copy()
        info: dict[str, Any]
        if step <= variant.online_updates:
            (theta, signal), cost = measured(
                partial(engine.online_update, theta, config.online_rate, maps, weights=weights)
            )
            info = {
                "signal_norm": float(np.linalg.norm(signal)),
                "accepted": True,
                "rate": config.online_rate,
                "objective_evaluations": 0,
                "trial_rates": [],
            }
            phase = "online"
        else:
            tail_step = step - variant.online_updates
            if variant.tail == "online_rmsprop":
                (theta, signal, second, online_counts), cost = measured(
                    partial(
                        engine.adaptive_online_update,
                        theta,
                        variant.rate,
                        second,
                        online_counts,
                        maps,
                        weights=weights,
                        beta2=config.beta2,
                        epsilon=config.epsilon,
                    )
                )
                info = cast(
                    "dict[str, Any]",
                    {
                        "signal_norm": float(np.linalg.norm(signal)),
                        "accepted": True,
                        "rate": variant.rate,
                        "objective_evaluations": 0,
                        "trial_rates": [],
                    },
                )
            else:
                (theta, first, second, info), cost = measured(
                    partial(
                        tail_update, engine, theta, maps, variant, first, second, tail_step, config, weights=weights
                    )
                )
            state["tail_step"] = tail_step
            phase = variant.tail
        if not all(np.all(np.isfinite(a)) for a in (theta, first, second)) or not np.isfinite(info["signal_norm"]):
            msg = "Nonfinite refinement update."
            raise FloatingPointError(msg)
        state["trace"].append({
            "step": step,
            "phase": phase,
            "epoch": epoch,
            "sampled_ensemble": sampled,
            "sampling": sampling_cost,
            "ensemble_digest": ensemble_digest(maps, weights),
            "path_weights": None if weights is None else weights.tolist(),
            "update_cost": cost,
            **info,
            **noise_events(maps),
            "update_norm": float(np.linalg.norm(theta - before)),
            "trajectory_passes": config.trajectories
            * ((3 if phase in {"online", "online_rmsprop"} else 2) + info["objective_evaluations"]),
        })
        # Validation is descriptive only, never used for training or acceptance.
        evaluate(step)
        state.update({
            "theta": theta.tolist(),
            "first": first.tolist(),
            "second": second.tolist(),
            "online_counts": online_counts.tolist(),
            "completed": step,
            "status": "complete" if step == config.updates else "running",
        })
        pilot.write_record(path, state)
    if limit == 0:
        pilot.write_record(path, state)
    return state


@dataclass(frozen=True)
class Job:
    """One private artifact owner; independent jobs can use separate CPU cores."""

    root: Path
    case: dict[str, Any]
    config: Config
    provenance: str
    variant: Variant | None = None
    replica: int = 0
    checkpoint: int = 0
    include_cross: bool = False
    sampling: str = "ordinary"

    @property
    def path(self) -> Path:
        """Return a stable private path, independent of worker allocation."""
        name = f"gradient_{self.checkpoint}" if self.variant is None else f"{self.variant.name}_{self.replica}"
        if self.sampling != "ordinary":
            name = f"{self.sampling}_{name}"
        return self.root / str(self.case["target"]["target_id"]) / f"{name}.json"


def run_job(job: Job) -> dict[str, Any]:
    """Run one diagnostic or refinement and authenticate retained job artifacts."""
    if job.variant is not None:
        record = optimize(
            job.path, job.case, job.variant, job.config, job.replica, job.provenance, sampling=job.sampling
        )
        result = {
            "variant": job.variant.name,
            "replica": job.replica,
            "initial_fidelity": record["evaluations"][0]["fidelity"],
            "final_fidelity": record["evaluations"][-1]["fidelity"],
            "split_update": job.config.online_updates,
            "late_gain_pp": 100
            * (record["evaluations"][-1]["fidelity"] - record["evaluations"][job.config.online_updates]["fidelity"]),
            "training_cpu_seconds": sum(
                r["sampling"]["cpu_seconds"] + r["update_cost"]["cpu_seconds"] for r in record["trace"]
            ),
            "training_wall_seconds": sum(
                r["sampling"]["wall_seconds"] + r["update_cost"]["wall_seconds"] for r in record["trace"]
            ),
            "evaluation_cpu_seconds": sum(r["cpu_seconds"] for r in record["evaluations"]),
            "objective_evaluations": sum(r["objective_evaluations"] for r in record["trace"]),
            "trajectory_passes": sum(r["trajectory_passes"] for r in record["trace"]),
        }
    else:
        signature = pilot.digest({
            "case": job.case,
            "config": asdict(job.config),
            "provenance": job.provenance,
            "checkpoint": job.checkpoint,
            "include_cross": job.include_cross,
        })
        if job.path.exists():
            record = pilot.read_record(job.path)
            if record["signature"] != signature or record["status"] != "complete":
                msg = "Diagnostic checkpoint configuration mismatch."
                raise ValueError(msg)
        else:
            checkpoint = next(c for c in job.case["checkpoints"] if c["update"] == job.checkpoint)
            engine = StatevectorBackend(
                pilot.restore_circuit(job.case["circuit"]), comparison.target_vector(job.case["target"])
            )
            record = diagnostic(
                engine,
                np.asarray(checkpoint["theta"], dtype=float),
                job.config,
                job.case["target"]["target_id"],
                job.checkpoint,
                include_cross=job.include_cross,
            )
            if abs(record["exact_fidelity"] - checkpoint["exact_noisy_fidelity"]) > 1e-10:
                msg = "Source checkpoint fidelity reproduction failed."
                raise ValueError(msg)
            record.update({"signature": signature, "status": "complete"})
            pilot.write_record(job.path, record)
        result = {"checkpoint": job.checkpoint, "aggregate": record["aggregate"]}
    return {
        "target_id": job.case["target"]["target_id"],
        "family": job.case["target"]["family"],
        "path": str(job.path.relative_to(job.root)),
        "checksum": pilot.digest(record),
        **result,
    }


def report(study: str, rows: list[dict[str, Any]]) -> str:
    """Summarize matched development cases without claiming a selected winner."""
    lines = [
        f"# Statevector development study: {study}",
        "",
        "Exploratory development results, not held-out confirmation or a state-of-the-art claim.",
        "",
    ]
    if study == "gradient":
        lines += [
            "| Estimator | R | Mean relative error | Mean cosine | Mean MSE * CPU seconds |",
            "| --- | --- | --- | --- | --- |",
        ]
        groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in rows:
            for item in row["aggregate"]:
                groups.setdefault((item["estimator"], item["trajectories"]), []).append(item)
        for (estimator, count), items in sorted(groups.items()):
            relative = nullable_mean([i["mean_relative_error"] for i in items])
            cosine = nullable_mean([i["mean_cosine"] for i in items])
            lines.append(
                f"| {estimator} | {count} | {relative} | {cosine} | "
                f"{np.mean([i['mse_times_cpu'] for i in items]):.6g} |"
            )
        lines += [
            "",
            (
                "Lower MSE * CPU is a descriptive cost/error measure, "
                "not proof that a biased estimator improves under averaging."
            ),
            (
                "Inspect per-checkpoint bias, variance and near-zero reference norms "
                "in the JSON artifacts before promoting cross."
            ),
        ]
    else:
        lines += [
            "| Variant | Mean final fidelity | Mean post-split gain (pp) | Mean train CPU (s) | Extra loss calls |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name in sorted({r["variant"] for r in rows}):
            items = [r for r in rows if r["variant"] == name]
            lines.append(
                f"| {name} | {np.mean([r['final_fidelity'] for r in items]):.8f} | "
                f"{np.mean([r['late_gain_pp'] for r in items]):.6f} | "
                f"{np.mean([r['training_cpu_seconds'] for r in items]):.4f} | "
                f"{sum(r['objective_evaluations'] for r in items)} |"
            )
        lines += [
            "",
            "CPU includes sampling and all trial losses; exact scoring and setup are recorded separately.",
            "The split is the configured hybrid boundary for every control, including all-online and full Adam.",
            "Krotov-Adam and Krotov-RMSProp are combinations, not pure Krotov results.",
        ]
    return "\n".join(lines) + "\n"


def run(
    root: Path,
    source_root: Path,
    *,
    study: str = "gradient",
    config: Config | None = None,
    include_cross: bool = False,
    workers: WorkerSetting = 1,
    resume: bool = False,
) -> dict[str, Any]:
    """Freeze sources/configuration and execute only the explicitly requested study."""
    config = Config() if config is None else config
    if study not in STUDIES or (include_cross and study != "gradient"):
        msg = "Unknown study or cross flag outside gradient diagnostics."
        raise ValueError(msg)
    resolve_workers(workers, 1)
    provenance, cases = load_cases(source_root)
    if any(not set(config.checkpoints) <= {c["update"] for c in case["checkpoints"]} for case in cases):
        msg = "Missing requested source checkpoints."
        raise ValueError(msg)
    manifest = json.loads(
        json.dumps({
            "version": VERSION,
            "study": study,
            "variants": [] if study == "gradient" else [asdict(v) for v in variants(study, config)],
            "config": asdict(config),
            "include_cross": include_cross,
            "backend": "statevector",
            "cases": cases,
            "source_root": str(source_root.resolve()),
            "source": provenance,
            "noise": ExactLocalPauliProvider().scientific_dict(),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": version("scipy"),
                "qiskit": version("qiskit"),
                "code": pilot.digest([comparison.runtime_fingerprint(), Path(__file__).read_text(encoding="utf-8")]),
            },
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
    jobs = (
        [
            Job(root, case, config, identity, checkpoint=k, include_cross=include_cross)
            for case in cases
            for k in config.checkpoints
        ]
        if study == "gradient"
        else [
            Job(root, case, config, identity, variant=v, replica=r)
            for case in cases
            for v in variants(study, config)
            for r in config.replicas
        ]
    )
    completed = {}
    for index, row in completed_jobs(run_job, jobs, workers=workers):
        completed[index] = row
        print(f"{study} {len(completed)}/{len(jobs)}: {row['target_id']} {jobs[index].path.name}", flush=True)
    result: dict[str, Any] = {
        "status": "complete",
        "manifest_checksum": identity,
        "rows": [completed[k] for k in range(len(jobs))],
    }
    path = root / "result.json"
    if path.exists() and pilot.read_record(path) != result:
        msg = "Completed result does not match authenticated job artifacts."
        raise ValueError(msg)
    pilot.write_record(path, result)
    (root / "summary.md").write_text(report(study, result["rows"]), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run one opt-in study; gradient diagnostics are the safe default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--study", choices=STUDIES, default="gradient")
    parser.add_argument("--include-cross", action="store_true")
    parser.add_argument("--workers", type=parse_workers, default="auto")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    run(
        arguments.output_root,
        arguments.source_root,
        study=arguments.study,
        include_cross=arguments.include_cross,
        workers=arguments.workers,
        resume=arguments.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
