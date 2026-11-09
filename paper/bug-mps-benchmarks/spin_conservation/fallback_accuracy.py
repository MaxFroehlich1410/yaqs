#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
# SPDX-License-Identifier: MIT
# Licensed under the MIT License

"""Plan, audit, and explicitly run accuracy-controlled fallback experiments.

No computation runs on import, with --help, or through the plan subcommand.
Use FALLBACK_ACCURACY.md for the protocol and the separate execution commands.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import sys
import tarfile
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import fallback_pilot as pilot
import fallback_reference as ref
import numpy as np

from mqt.yaqs import AnalogSimParams, EvolutionMode, track_conservation
from mqt.yaqs.analog import evolution
from mqt.yaqs.core.methods import bug, conservation
from mqt.yaqs.core.parallel_utils import limit_worker_threads, safe_set_numba_threads

SCHEMA = 1
STAGES = ("scan", "frame", "infeasible", "no_minimizer", "solver_failure", "gauge_failure", "unresolved")


def digest(path):
    """Hash an input file whose bytes must stay fixed across the study."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path, root):
    """Describe a read-only source artifact using a relative path and content hash."""
    return {"path": str(path.relative_to(root)), "sha256": digest(path)}


def checked_input(source, artifact):
    """Resolve a pinned artifact and reject changed input before computation."""
    path = (Path(source) / artifact["path"]).resolve()
    if not path.is_relative_to(Path(source).resolve()) or digest(path) != artifact["sha256"]:
        raise ValueError(f"Source artifact changed or escaped its root: {artifact['path']}")
    return path


def validate_case(case, probes=()):
    """Require usable caps, finite clocks, and exactly accessible probe times."""
    full_rank = 2 ** (case.length // 2)
    if case.length < 2 or not 2 <= case.cap <= full_rank or case.integrator != "bug":
        raise ValueError("Accuracy study requires BUG, L >= 2, and 2 <= cap <= full Schmidt rank")
    if case.initial_chi != 2 or case.cap < case.initial_chi:
        raise ValueError("Use a fixed initial bond of two, independent of the cap")
    if any(not np.isfinite(v) or v <= 0 for v in (case.dt, case.total_time, case.conserve_tol, case.krylov_tol)):
        raise ValueError("Times and solver tolerances must be finite and positive")
    if case.width_per_site != 1.0:
        raise ValueError("The accuracy study uses spectral width per site equal to one")
    if not np.isclose(case.total_time / case.dt, round(case.total_time / case.dt), rtol=0, atol=1e-9):
        raise ValueError("Total time must be an integer number of steps")
    for value in probes:
        half_steps = 2 * value / case.dt
        if not 0 < value <= case.total_time or not np.isclose(half_steps, round(half_steps), rtol=0, atol=1e-9):
            raise ValueError("Probe time must lie on the BUG half-step grid")


def make_plan(source, *, dts=(0.1, 0.05, 0.02, 0.01), thresholds=None):
    """Read existing metadata and pin inputs; perform no physics calculations.

    All completed source results are checked so selection does not depend on a
    possibly stale replay index. Existing eventful cases are deduplicated by
    physical parameters and exact prepared vector identity before expanding caps.
    """
    source = Path(source).resolve()
    thresholds = ref.Accuracy() if thresholds is None else thresholds
    groups, audits = {}, []
    results = sorted(source.glob("cases/*/result.json"))
    if not results:
        raise ValueError("No source results found")
    for path in results:
        result = json.loads(path.read_text())
        if result.get("status") != "complete":
            raise ValueError(f"Incomplete source result: {path}")
        arm = next(a for a in result["arms"] if a["variant"] == "fallback")
        if len(arm["events"]) != arm["counts"]["fallback_attempts"]:
            raise ValueError("Source event count does not match fallback counters")
        if not arm["events"]:
            continue
        case = pilot.Case(**result["case"])
        validate_case(case)
        initial = pin(path.parent / "initial.npz", source)
        signature = asdict(case)
        for name in ("cap", "dt"):
            signature.pop(name)
        signature.update(initial_vector_sha256=result["initial_vector_sha256"], scale=result["hamiltonian"]["scale"])
        group_key = json.dumps(signature, sort_keys=True)
        group = groups.setdefault(
            group_key,
            {
                "case": case,
                "caps": set(),
                "probe_times": set(),
                "initial": initial,
                "scale": result["hamiltonian"]["scale"],
                "source_cases": [],
            },
        )
        group["caps"].add(case.cap)
        group["source_cases"].append(case.key)
        for event in arm["events"]:
            physical_time = ref.event_time(result["case"], event)
            group["probe_times"].add(round(physical_time, 12))
            audits.append({
                "source_key": case.key,
                "result": pin(path, source),
                "initial": initial,
                "snapshot": pin(path.parent / "fallback" / event["snapshot"], source),
                "event_index": event["index"],
                "physical_time": physical_time,
            })
    studies = []
    for group in groups.values():
        base = group["case"]
        maximum = 2 ** (base.length // 2)
        caps = sorted({*group["caps"], *[2**i for i in range(1, base.length // 2 + 1) if 2**i >= min(group["caps"])]})
        for cap, dt in itertools.product(caps, sorted(set(dts), reverse=True)):
            case = replace(base, cap=cap, dt=dt)
            probes = sorted(group["probe_times"])
            validate_case(case, probes)
            studies.append({
                "study": "convergence",
                "case": asdict(case),
                "key": case.key,
                "initial": group["initial"],
                "scale": group["scale"],
                "probe_times": probes,
                "source_cases": group["source_cases"],
                "full_rank_control": cap == maximum,
            })
    for length, cap, dt, model in itertools.product(
        (12, 16), (16, 32, 64), (0.02, 0.01), ("power_a2_d1", "dimer_0.05", "dimer_0.2", "dimer_0.5")
    ):
        case = pilot.Case(
            model,
            length,
            cap,
            "domain_wall" if model == "power_a2_d1" else "singlet_product",
            dt,
            total_time=8.0,
            width_per_site=1.0,
            initial_chi=2,
        )
        validate_case(case)
        studies.append({
            "study": "larger",
            "case": asdict(case),
            "key": case.key,
            "initial": None,
            "scale": None,
            "probe_times": [],
            "source_cases": [],
            "full_rank_control": cap == 2 ** (length // 2),
        })
    if not audits:
        raise ValueError("No existing fallback events found to audit")
    return {
        "schema": SCHEMA,
        "source_root": str(source),
        "thresholds": asdict(thresholds),
        "variants": list(pilot.VARIANTS),
        "audits": audits,
        "studies": studies,
        "summary": {
            "source_results": len(results),
            "old_events": len(audits),
            "convergence_cases": sum(s["study"] == "convergence" for s in studies),
            "larger_cases": sum(s["study"] == "larger" for s in studies),
            "trajectories": 3 * len(studies),
        },
        "status": "planned_only",
    }


def prepare_initial(specification, source):
    """Load pinned original tensors or prepare one seed-stable rank-two state."""
    case = pilot.Case(**specification["case"])
    if specification["initial"] is not None:
        state = pilot.load_snapshot(checked_input(source, specification["initial"]))
    elif case.initial == "singlet_product":
        state = ref.singlet_product(case.length, seed=case.seed, pad=True)
    else:
        state = pilot.fx.initial_state(case.length, case.initial, chi=2, seed=case.seed)
    if state.length != case.length or max(state.bond_dimensions()) > case.cap:
        raise ValueError("Prepared state is incompatible with the case")
    return state


def reference_health(vector, operator, initial_norm, initial_energy, tolerance):
    """Reject a numerically unreliable reference before labeling any call accurate."""
    norm_drift = abs(float(np.linalg.norm(vector)) - initial_norm)
    energy_drift = abs(pilot.energy(vector, operator) - initial_energy)
    if (
        not np.all(np.isfinite(vector))
        or norm_drift > tolerance
        or energy_drift > tolerance * max(1, abs(initial_energy))
    ):
        raise ValueError("Full-state reference failed norm/energy validation")
    return {"norm_drift": norm_drift, "energy_drift": energy_drift}


def stage_from_counts(counts):
    """Keep recovery stages and distinct numerical failure modes separate."""
    return next(
        (
            label
            for counter, label in (
                ("scan_successes", "scan"),
                ("frame_successes", "frame"),
                ("infeasible", "infeasible"),
                ("no_minimizer", "no_minimizer"),
                ("solver_failures", "solver_failure"),
                ("gauge_failures", "gauge_failure"),
            )
            if counts[counter]
        ),
        "unresolved",
    )


def audit_events(plan, output):
    """Compute event-time errors and replay saved fallback corrections explicitly."""
    limit_worker_threads(1)
    safe_set_numba_threads(1)
    output = initialize_output(plan, output, mode="audit")
    thresholds = ref.Accuracy(**plan["thresholds"])
    source = Path(plan["source_root"])
    rows = []
    for item in plan["audits"]:
        result = json.loads(checked_input(source, item["result"]).read_text())
        case = result["case"]
        arm = next(a for a in result["arms"] if a["variant"] == "fallback")
        event = next(e for e in arm["events"] if e["index"] == item["event_index"])
        physical_time = ref.event_time(case, event)
        if abs(physical_time - item["physical_time"]) > 1e-12:
            raise ValueError("Audit event time changed")
        initial = pilot.canonical_vector(pilot.load_snapshot(checked_input(source, item["initial"])))
        before = pilot.load_snapshot(checked_input(source, item["snapshot"]))
        mpo, operator, metadata = ref.hamiltonians(case["length"], case["model"], scale=result["hamiltonian"]["scale"])
        action_error = ref.check_operators(mpo, operator)
        exact = ref.propagate(operator, initial, physical_time)
        health = reference_health(
            exact, operator, np.linalg.norm(initial), pilot.energy(initial, operator), thresholds.reference
        )
        after = deepcopy(before)
        frame = mpo.reflected() if before.flipped else mpo
        with track_conservation() as stats:
            conservation.apply_conservation(
                after, frame, conservation.EnergyTarget(event["target"], case["conserve_tol"])
            )
        counts = stats.to_dict()
        gauge_preserved = after.flipped == before.flipped and after.orthogonality_center == before.orthogonality_center
        bonds_preserved = after.bond_dimensions() == before.bond_dimensions()
        replay_valid = (
            counts == event["counts"]
            and gauge_preserved
            and bonds_preserved
            and abs(pilot.energy(pilot.canonical_vector(after), operator) - event["post_energy"]) < 1e-11
        )
        if not replay_valid:
            raise ValueError(f"Saved fallback did not replay: {item['source_key']}")
        observables = ref.Observables(case["length"])
        rows.append({
            "source_key": item["source_key"],
            "event_index": event["index"],
            "case": case,
            "physical_time": physical_time,
            "old_step_end_time": event["step_end_time"],
            "stage": stage_from_counts(counts),
            "replay_valid": replay_valid,
            "pre": observables.compare(pilot.canonical_vector(before), exact, thresholds),
            "post": observables.compare(pilot.canonical_vector(after), exact, thresholds),
            "original_final_infidelity": arm["final_infidelity"],
            "reference_health": health,
            "operator_action_error": action_error,
            "hamiltonian": metadata,
        })
        pilot.write_json(
            output / "audit.json",
            {"thresholds": plan["thresholds"], "events": rows, "complete": len(rows) == len(plan["audits"])},
        )
    return rows


@contextmanager
def accuracy_hooks(stats, directory, operator, case, variant, thresholds, clock, probes):
    """Measure every correction before treatment and each requested half-step probe.

    The existing observer still owns the correction and event snapshots. The
    second hook observes BUG postprocessing to obtain the uncorrected arm at the
    same half-step times; all hooks are restored even on numerical failure.
    """
    observables = ref.Observables(case.length)
    record = {"qualified_calls": 0, "qualified_events": [], "probes": [], "correction_calls": 0}
    old_postprocess = bug._postprocess_bug_state
    with pilot.observe_corrections(stats, directory, operator, case, variant) as observation:
        observer = bug.apply_conservation

        def corrected(state, mpo, target):
            call = observation["call_in_step"] + 1
            exact = clock["vectors"][call - 1]
            pre = observables.compare(pilot.canonical_vector(state), exact, thresholds)
            record["correction_calls"] += 1
            record["qualified_calls"] += int(pre["within_thresholds"])
            old_count = len(observation["events"])
            changed = observer(state, mpo, target)
            if len(observation["events"]) > old_count:
                event = observation["events"][-1]
                event.update(
                    physical_time=ref.event_time(asdict(case), event),
                    accuracy_pre=pre,
                    accuracy_post=observables.compare(pilot.canonical_vector(state), exact, thresholds),
                )
                if pre["within_thresholds"]:
                    record["qualified_events"].append(event["stage"])
                event["post_snapshot"] = f"event-{event['index']:05d}-post.npz"
                pilot.save_snapshot(directory / event["post_snapshot"], state)
            return changed

        def postprocessed(state, mpo, params, **kwargs):
            changed = old_postprocess(state, mpo, params, **kwargs)
            clock["postprocess_call"] += 1
            call = clock["postprocess_call"]
            physical_time = (observation["step"] - 1 + call / 2) * case.dt
            if any(abs(physical_time - t) < 1e-10 for t in probes):
                vector = pilot.canonical_vector(state)
                record["probes"].append({
                    "time": physical_time,
                    **observables.compare(vector, clock["vectors"][call - 1], thresholds),
                    "observables": observables.values(vector).tolist(),
                    "bond": int(max(state.bond_dimensions())),
                })
            return changed

        bug.apply_conservation = evolution.apply_conservation = corrected
        bug._postprocess_bug_state = postprocessed
        try:
            yield observation, record
        finally:
            bug._postprocess_bug_state = old_postprocess


def run_case(specification, source, output, thresholds, *, variants=None):
    """Run a matched triple with streamed half-step full-state references.

    Each arm regenerates the deterministic reference stream, bounding storage
    independently of the number of time steps. Timing includes these diagnostics
    and is deliberately not advertised as an algorithm performance comparison.
    """
    variants = pilot.VARIANTS if variants is None else tuple(variants)
    if not variants or len(set(variants)) != len(variants) or any(v not in pilot.VARIANTS for v in variants):
        raise ValueError("Invalid or duplicate trajectory variants")
    case = pilot.Case(**specification["case"])
    validate_case(case, specification["probe_times"])
    limit_worker_threads(1)
    safe_set_numba_threads(1)
    directory = Path(output) / "cases" / case.key
    directory.mkdir(parents=True, exist_ok=True)
    mpo, operator, metadata = ref.hamiltonians(
        case.length, case.model, case.width_per_site, scale=specification["scale"]
    )
    action_error = ref.check_operators(mpo, operator)
    initial = prepare_initial(specification, source)
    initial_vector = pilot.canonical_vector(initial)
    pilot.save_snapshot(directory / "initial.npz", initial)
    target = conservation.expectation_value(initial, mpo)
    initial_energy = pilot.energy(initial_vector, operator)
    if abs(target - initial_energy) > 1e-11:
        raise ValueError("Independent initial energies disagree")
    result = {
        "schema": SCHEMA,
        "key": case.key,
        "case": asdict(case),
        "specification": specification,
        "thresholds": asdict(thresholds),
        "initial_vector_sha256": hashlib.sha256(initial_vector.tobytes()).hexdigest(),
        "hamiltonian": metadata,
        "target_energy": target,
        "operator_action_error": action_error,
        "initial_infidelity_to_exact_singlets": None
        if case.initial != "singlet_product"
        else pilot.infidelity(initial_vector, ref.singlet_product(case.length).to_vec()),
        "arms": [],
    }
    measurements = ref.Observables(case.length)
    steps = round(case.total_time / case.dt)
    for variant in variants:
        started = time.perf_counter()
        folder = directory / variant
        folder.mkdir(exist_ok=True)
        state, exact = deepcopy(initial), initial_vector.copy()
        params = AnalogSimParams(
            elapsed_time=case.dt,
            dt=case.dt,
            evolution_mode=EvolutionMode.BUG,
            max_bond_dim=case.cap,
            trunc_mode="relative_discarded_weight",
            svd_threshold=0.0,
            krylov_tol=case.krylov_tol,
            conserve_energy=variant != "uncorrected",
            conserve_fallback=variant == "fallback",
            conserve_tol=case.conserve_tol,
        )
        configured = (
            None
            if variant == "uncorrected"
            else conservation.EnergyTarget(target, case.conserve_tol, fallback=variant == "fallback")
        )
        trace, clock = [], {}
        health_max = {"norm_drift": 0.0, "energy_drift": 0.0}
        with (
            track_conservation() as stats,
            accuracy_hooks(stats, folder, operator, case, variant, thresholds, clock, specification["probe_times"]) as (
                obs,
                record,
            ),
        ):
            for step in range(steps + 1):
                if step:
                    half = ref.propagate(operator, exact, case.dt / 2)
                    exact = ref.propagate(operator, half, case.dt / 2)
                    for vector in (half, exact):
                        health = reference_health(
                            vector, operator, np.linalg.norm(initial_vector), initial_energy, thresholds.reference
                        )
                        health_max = {k: max(health_max[k], health[k]) for k in health}
                    clock.update(vectors=(half, exact), postprocess_call=0)
                    obs["step"], obs["call_in_step"] = step, 0
                    evolution.apply_unitary_evolution(state, mpo, params, conservation_target=configured)
                    if clock["postprocess_call"] != 2:
                        raise ValueError("BUG half-step protocol changed")
                vector = pilot.canonical_vector(state)
                if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0:
                    raise ValueError(f"Invalid state in {variant} step {step}")
                trace.append({
                    "time": step * case.dt,
                    **measurements.compare(vector, exact, thresholds),
                    "energy_drift": abs(pilot.energy(vector, operator) - target),
                    "bond": int(max(state.bond_dimensions())),
                    "observables": measurements.values(vector).tolist(),
                })
                if step % 10 == 0 or step == steps:
                    pilot.write_json(
                        folder / "progress.json",
                        {
                            "step": step,
                            "steps": steps,
                            "physical_time": step * case.dt,
                            "wall_seconds": time.perf_counter() - started,
                            "counts": stats.to_dict(),
                        },
                    )
        counts = stats.to_dict()
        expected_calls = 0 if variant == "uncorrected" else 2 * steps
        if (
            counts["scalar_calls"] != expected_calls
            or record["correction_calls"] != expected_calls
            or len(obs["events"]) != counts["fallback_attempts"]
            or max(t["bond"] for t in trace) > case.cap
            or len(record["probes"]) != len(specification["probe_times"])
        ):
            raise ValueError("Correction count, bond cap, or probe coverage changed")
        result["arms"].append({
            "variant": variant,
            "counts": counts,
            "trace": trace,
            "events": obs["events"],
            **record,
            "reference_health": health_max,
            "wall_seconds": time.perf_counter() - started,
        })
    result["status"] = "complete"
    pilot.write_json(directory / "result.json", result)
    return result


def initialize_output(plan, output, *, mode):
    """Pin plan, source code, environment and inputs; never write into old results."""
    output = Path(output).resolve()
    source = Path(plan["source_root"]).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Use a separate output directory outside the source results")
    if plan["schema"] != SCHEMA:
        raise ValueError("Unsupported plan schema")
    ref.Accuracy(**plan["thresholds"])
    for item in plan["audits"]:
        for name in ("result", "initial", "snapshot"):
            checked_input(source, item[name])
    for spec in plan["studies"]:
        validate_case(pilot.Case(**spec["case"]), spec["probe_times"])
        if spec["key"] != pilot.Case(**spec["case"]).key:
            raise ValueError("Case identity changed")
        if spec["initial"]:
            checked_input(source, spec["initial"])
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "numba", "qiskit")}
    signature = {
        "schema": SCHEMA,
        "runtime": {"versions": versions, "python": sys.version, "platform": platform.platform()},
        "mode": mode,
        "plan": plan,
        "source_sha256": {str(p.relative_to(pilot.ROOT)): digest(p) for p in pilot.source_files()},
    }
    manifest = output / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous["signature"] != signature:
            raise ValueError("Different plan, mode, runtime, or implementation; use a fresh output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("New output directory must be empty")
        output.mkdir(parents=True, exist_ok=True)
        versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "numba", "qiskit")}
        pilot.write_json(
            manifest,
            {
                "signature": signature,
                "versions": versions,
                "python": sys.version,
                "platform": platform.platform(),
            },
        )
        with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
            for path in pilot.source_files():
                archive.add(path, arcname=str(path.relative_to(pilot.ROOT)))
    return output


def execute(plan, output, *, study="all", limit=None):
    """Explicit serial execution with exact-manifest resume; never called by plan."""
    output = initialize_output(plan, output, mode="run")
    thresholds = ref.Accuracy(**plan["thresholds"])
    selected = [s for s in plan["studies"] if study == "all" or s["study"] == study]
    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be positive")
        selected = selected[:limit]
    results = []
    for index, spec in enumerate(selected, 1):
        path = output / "cases" / spec["key"] / "result.json"
        if path.exists():
            result = json.loads(path.read_text())
            if (
                result.get("status") != "complete"
                or result["specification"] != spec
                or result["thresholds"] != plan["thresholds"]
            ):
                raise ValueError("Invalid or incompatible completed result")
        else:
            print(f"[{index}/{len(selected)}] {spec['key']}", flush=True)
            result = run_case(spec, plan["source_root"], output, thresholds)
        results.append(result)
    return results


def main(argv=None):
    """Require an explicit subcommand; execution is never the default."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan", help="Read metadata and write a plan; no simulations or audits")
    planning.add_argument("--source", type=Path, required=True)
    planning.add_argument("--output", type=Path, required=True)
    planning.add_argument("--infidelity-tol", type=float, default=1e-3)
    planning.add_argument("--observable-tol", type=float, default=1e-3)
    for command in ("audit", "run"):
        sub = commands.add_parser(
            command, help="Compute old event errors" if command == "audit" else "Execute new trajectories"
        )
        sub.add_argument("--plan", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        if command == "run":
            sub.add_argument("--study", choices=("all", "convergence", "larger"), default="all")
            sub.add_argument("--limit", type=int, help="Run the first N selected configurations")
    reporting = commands.add_parser("report", help="Summarize existing accuracy results without evolution")
    reporting.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        if args.output.exists():
            raise ValueError("Plan already exists; choose a new filename")
        plan = make_plan(args.source, thresholds=ref.Accuracy(args.infidelity_tol, args.observable_tol))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pilot.write_json(args.output, plan)
        print(json.dumps(plan["summary"], indent=2))
    elif args.command == "report":
        from fallback_accuracy_report import report

        report(args.output)
    else:
        plan = json.loads(args.plan.read_text())
        if args.command == "audit":
            audit_events(plan, args.output)
        else:
            execute(plan, args.output, study=args.study, limit=args.limit)


if __name__ == "__main__":
    main()
