# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Experimental exact no-factor stratification for the fixed local Pauli channel."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from benchmarks.state_preparation.execution import WorkerSetting, completed_jobs, parse_workers, resolve_workers
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider
from benchmarks.state_preparation.weighted_sampling import FactorSampler
from experiments import state_preparation_pilot as pilot
from experiments import state_preparation_sampling_budget as budget
from experiments import state_preparation_trajectory_comparison as comparison
from experiments import state_preparation_trajectory_diagnostics as development
from mqt.yaqs.optimization import KrotovTJMOptions, StatevectorBackend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization.statevector import NoiseMaps

VERSION = "yaqs.no_factor_stratified_gradient.v1"


def weighted_gradient(
    engine: StatevectorBackend, theta: NDArray[np.float64], sampler: FactorSampler, maps: NoiseMaps | None
) -> tuple[NDArray[np.float64], float]:
    """Combine the exact no-factor path and a conditional sample-average gradient.

    Probabilities are independent of parameters, so no likelihood derivative is
    missing. The stratum is no sampled factors, not no net errors; cancellations
    remain in the conditional stratum. This is not a nonunitary no-jump rule.
    """
    zero_gradient, zero_loss, _ = engine.gradient(theta)
    if sampler.event_probability == 0:
        return zero_gradient, zero_loss
    if maps is None or len(sampler.factors) == 0:
        msg = "Positive-noise weighting requires conditional maps."
        raise ValueError(msg)
    if any(not any(m.operators for m in row) for row in maps):
        msg = "Conditional maps must contain a sampled factor in every trajectory."
        raise ValueError(msg)
    gradient, loss, _ = engine.gradient(theta, maps)
    return (
        sampler.no_factor_probability * zero_gradient + sampler.event_probability * gradient,
        sampler.no_factor_probability * zero_loss + sampler.event_probability * loss,
    )


@dataclass(frozen=True)
class Job:
    """One development checkpoint with independent repeated estimator comparisons."""

    root: Path
    case: dict[str, Any]
    checkpoint: int
    config: development.Config
    provenance: str


def run_job(job: Job) -> dict[str, Any]:
    """Measure both estimators at identical parameters and total path budgets."""
    target = job.case["target"]
    path = job.root / target["target_id"] / f"gradient_{job.checkpoint}.json"
    signature = pilot.digest({
        "case": job.case,
        "checkpoint": job.checkpoint,
        "config": asdict(job.config),
        "provenance": job.provenance,
    })
    if path.exists():
        record = pilot.read_record(path)
        if record["signature"] != signature or record["status"] != "complete":
            msg = "Weighted diagnostic checkpoint configuration mismatch."
            raise ValueError(msg)
    else:
        engine = StatevectorBackend(pilot.restore_circuit(job.case["circuit"]), comparison.target_vector(target))
        checkpoint = next(c for c in job.case["checkpoints"] if c["update"] == job.checkpoint)
        theta = np.array(checkpoint["theta"])
        calibration = CalibratedLocalDepolarizingProvider()
        (reference, _, fidelity), reference_cost = development.measured(
            lambda: density_loss_gradient(engine.circuit, theta, engine.target, calibration)
        )
        if abs(fidelity - checkpoint["exact_noisy_fidelity"]) > 1e-10:
            msg = "Saved diagnostic fidelity did not reproduce."
            raise ValueError(msg)
        sampler, setup = development.measured(lambda: FactorSampler(engine.circuit, calibration))
        rows: list[dict[str, Any]] = []
        for count in job.config.diagnostic_counts:
            if count < 2:
                msg = "Weighted diagnostics require at least two total trajectories."
                raise ValueError(msg)
            for repeat in range(job.config.diagnostic_repeats):
                seed = development.seed_for(job.config, VERSION, target["target_id"], repeat, job.checkpoint)
                for estimator in ("independent", "weighted")[:: -1 if repeat % 2 else 1]:
                    if estimator == "independent":
                        maps, sampling = development.measured(
                            lambda count=count, seed=seed: engine.sample_noise_maps(
                                theta,
                                KrotovTJMOptions(num_trajectories=count, random_seed=seed),
                                ExactLocalPauliProvider(),
                            )
                        )
                        (gradient, loss, _), cost = development.measured(partial(engine.gradient, theta, maps))
                    else:
                        maps, sampling = development.measured(
                            lambda count=count, seed=seed: sampler.maps(sampler.draw(count - 1, seed))
                        )
                        (gradient, loss), cost = development.measured(
                            partial(weighted_gradient, engine, theta, sampler, maps)
                        )
                    rows.append({
                        "estimator": estimator,
                        "trajectories": count,
                        "conditional_trajectories": count - 1 if estimator == "weighted" else 0,
                        "repeat": repeat,
                        "gradient": gradient.tolist(),
                        "sample_loss": loss,
                        **development.gradient_errors(gradient, reference),
                        **development.noise_events(maps),
                        "sampling": sampling,
                        "gradient_cost": cost,
                        # Conservatively charge weighted setup once per estimate, not amortized.
                        "setup_cpu_seconds": setup["cpu_seconds"] if estimator == "weighted" else 0,
                        "cpu_seconds": sampling["cpu_seconds"]
                        + cost["cpu_seconds"]
                        + (setup["cpu_seconds"] if estimator == "weighted" else 0),
                    })
        aggregates = []
        for count in job.config.diagnostic_counts:
            for estimator in ("independent", "weighted"):
                subset = [r for r in rows if r["trajectories"] == count and r["estimator"] == estimator]
                gradients = np.array([r["gradient"] for r in subset])
                mse = float(np.mean([r["squared_error"] for r in subset]))
                cpu = float(np.mean([r["cpu_seconds"] for r in subset]))
                aggregates.append({
                    "estimator": estimator,
                    "trajectories": count,
                    "mse": mse,
                    "bias_norm": float(np.linalg.norm(gradients.mean(axis=0) - reference)),
                    "variance_trace": float(np.var(gradients, axis=0, ddof=1).sum()),
                    "mean_cpu_seconds": cpu,
                    "mse_times_cpu": mse * cpu,
                    "mean_cosine": development.nullable_mean([r["cosine"] for r in subset]),
                    "mean_relative_error": development.nullable_mean([r["relative_error"] for r in subset]),
                    "descent_fraction": float(np.mean([r["descent_dot"] > 0 for r in subset])),
                })
        record = {
            "status": "complete",
            "signature": signature,
            "no_factor_probability": sampler.no_factor_probability,
            "event_probability": sampler.event_probability,
            "reference_gradient": reference.tolist(),
            "exact_fidelity": fidelity,
            "reference_cost": reference_cost,
            "setup": setup,
            "rows": rows,
            "aggregate": aggregates,
        }
        pilot.write_record(path, record)
    return {
        "target_id": target["target_id"],
        "family": target["family"],
        "checkpoint": job.checkpoint,
        "path": str(path.relative_to(job.root)),
        "checksum": pilot.digest(record),
        "aggregate": record["aggregate"],
    }


def run(
    root: Path,
    source_root: Path,
    *,
    config: development.Config | None = None,
    workers: WorkerSetting = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    """Run only the weighted estimator diagnostic, without optimizer/target selection."""
    config = development.Config() if config is None else config
    resolve_workers(workers, 1)
    if min(config.diagnostic_counts) < 2:
        msg = "Weighted diagnostics need at least two total paths."
        raise ValueError(msg)
    source, cases = development.load_cases(source_root)
    if any(not set(config.checkpoints) <= {c["update"] for c in case["checkpoints"]} for case in cases):
        msg = "Missing requested checkpoint."
        raise ValueError(msg)
    manifest = json.loads(
        json.dumps({
            "version": VERSION,
            "config": asdict(config),
            "cases": cases,
            "source_root": str(source_root.resolve()),
            "source": source,
            "runtime": budget.runtime(),
            "driver": pilot.digest(Path(__file__).read_text(encoding="utf-8")),
            "noise": ExactLocalPauliProvider().scientific_dict(),
            "path_budget": "R ordinary versus one deterministic plus R-1 conditional paths",
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
    jobs = [Job(root, case, checkpoint, config, identity) for case in cases for checkpoint in config.checkpoints]
    completed = {}
    for index, row in completed_jobs(run_job, jobs, workers=workers):
        completed[index] = row
        print(f"weighted {len(completed)}/{len(jobs)}: {row['target_id']} update {row['checkpoint']}", flush=True)
    result: dict[str, Any] = {
        "status": "complete",
        "manifest_checksum": identity,
        "rows": [completed[i] for i in range(len(jobs))],
    }
    path = root / "result.json"
    if path.exists() and pilot.read_record(path) != result:
        msg = "Weighted result differs from authenticated records."
        raise ValueError(msg)
    pilot.write_record(path, result)
    (root / "summary.md").write_text(development.report("gradient", result["rows"]), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Expose only the fixed development diagnostic on the command line."""
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
