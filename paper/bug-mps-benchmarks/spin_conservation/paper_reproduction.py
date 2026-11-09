#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
# SPDX-License-Identifier: MIT
# Licensed under the MIT License

"""Deduplicated reproduction of the September 27 conservation manuscript.

Planning, reference preparation, trajectories and reporting are separate explicit
commands. Production runs retain the manuscript's unscaled Hamiltonian and exact
legacy correction schedules. No simulation starts when this module is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
import tarfile
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import fallback_pilot as pilot
import fallback_reference as ref
import fixtures_n as fx
import l16_joint_table as jt
import l16_observables as obs
import numpy as np
from scipy.sparse.linalg import expm_multiply

from mqt.yaqs import track_conservation
from mqt.yaqs.analog import evolution
from mqt.yaqs.analog.analog_tjm import capture_conservation_target
from mqt.yaqs.core.methods import bug, conservation
from mqt.yaqs.core.parallel_utils import limit_worker_threads, safe_set_numba_threads

ROOT = pilot.ROOT
HERE = Path(__file__).resolve().parent
SCHEMA = 1
CORE = [
    ("bug", "none", "none"),
    ("bug", "joint4", "none"),
    ("bug", "joint4", "full"),
    ("bug", "jointS2", "none"),
    ("bug", "jointS2", "full"),
    ("bug", "joint5", "none"),
    ("bug", "joint5", "full"),
    ("tdvp", "none", "none"),
    ("tdvp", "joint4", "none"),
]


def read(path):
    """Read JSON metadata."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    """Hash a file without loading large reference arrays into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def sources():
    """Include the production package, fixtures, schedules, and report code."""
    return sorted([*ROOT.joinpath("src").rglob("*.py"), *HERE.glob("*.py")])


def source_hashes():
    """Identify all executed scientific and postprocessing source files."""
    return {str(path.relative_to(ROOT)): digest(path) for path in sources()}


def runtime():
    """Record the numerical environment and actual worker identity."""
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "numba", "qiskit")},
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }


def arm_id(integrator, variant, sweep, fallback=None) -> str:
    """Use an explicit fallback flag only where the scalar dispatcher uses it."""
    return f"{integrator}-{variant}-{sweep}" + (f"-fallback-{int(fallback)}" if fallback is not None else "")


def make_plan(*, length=16, cap=32, total_time=6.0, dt=0.01, manuscript=None):
    """Describe nine original trajectories and four scalar ablation trajectories."""
    if length < 4 or cap < 4 or dt <= 0 or total_time <= 0 or not np.isclose(total_time / dt, round(total_time / dt)):
        msg = "Require L >= 4, cap >= 4, and a positive integral number of steps"
        raise ValueError(msg)
    arms = [
        {"id": arm_id(i, v, s), "integrator": i, "variant": v, "sweep": s, "fallback": None, "role": "manuscript"}
        for i, v, s in CORE
    ]
    arms += [
        {
            "id": arm_id(i, "pinH", "none", f),
            "integrator": i,
            "variant": "pinH",
            "sweep": "none",
            "fallback": f,
            "role": "scalar_ablation",
        }
        for i in ("bug", "tdvp")
        for f in (False, True)
    ]
    return {
        "schema": SCHEMA,
        "protocol": {
            "length": length,
            "cap": cap,
            "dt": dt,
            "total_time": total_time,
            "steps": round(total_time / dt),
            "model": "xxx",
            "hamiltonian_scale": 1.0,
            "initial_state": "tilted_neel",
            "seed": fx.SEED,
            "initial_chi": fx.INITIAL_CHI,
            "tilt_angle": float(fx.TILT_ANGLE),
            "noise_scale": fx.NOISE_SCALE,
            "svd_threshold": jt.SVD_THRESHOLD,
            "krylov_tol": jt.KRYLOV_TOL,
            "conserve_tol": jt.CONSERVE_TOL,
            "trunc_mode": "relative_discarded_weight",
            "accuracy_infidelity": 1e-3,
            "accuracy_observable": 1e-3,
        },
        "arms": arms,
        "manuscript": None if manuscript is None else {"name": Path(manuscript).name, "sha256": digest(manuscript)},
        "table_I": [arm_id(i, v, "none") for i in ("bug", "tdvp") for v in ("none", "joint4")],
        "table_II": [a["id"] for a in arms if a["role"] == "manuscript" and a["integrator"] == "bug"],
        "table_III": ["bug-none-none", "bug-joint4-full", "bug-joint5-full"],
        "figure_1": [arm_id(i, v, s) for i, v, s in CORE if i == "bug"],
        "reference": "one shared sparse expm_multiply stream at half-step times",
        "new_dimer_section": "reuse completed accuracy study; no duplicate simulations",
    }


def prepare(plan, output):
    """Freeze a new output directory and source archive without starting a run."""
    output = Path(output)
    if output.exists():
        msg = "Use a new output directory; never overwrite a prepared study"
        raise ValueError(msg)
    output.mkdir(parents=True)
    for name in ("reference", "arms", "claims", "hosts"):
        (output / name).mkdir()
    manifest = {
        "schema": SCHEMA,
        "plan": plan,
        "source_sha256": source_hashes(),
        "runtime": runtime(),
        "created": time.time(),
    }
    pilot.write_json(output / "manifest.json", manifest)
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for path in sources():
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    return manifest


def verify(output):
    """Reject source or numerical dependency drift before any work starts."""
    manifest = read(Path(output) / "manifest.json")
    if manifest["source_sha256"] != source_hashes():
        msg = "Scientific source changed since preparation"
        raise ValueError(msg)
    if manifest["runtime"]["versions"] != runtime()["versions"]:
        msg = "Numerical dependency versions changed since preparation"
        raise ValueError(msg)
    return manifest["plan"]


def operators(length):
    """Use the original manuscript operator and observable conventions."""
    pinned = {
        "H": fx.sparse_hamiltonian(length, "xxx"),
        "Sx": fx.sparse_sx(length),
        "Sy": fx.sparse_sy(length),
        "Sz": fx.sparse_sz(length),
        "S2": fx.sparse_s2(length),
    }
    families = obs.build_observables(length)
    obs.check_conventions(length, families)
    return pinned, families


def measurements(vector, pinned, families):
    """Measure all legacy observable families from one normalized vector."""
    return (
        {name: pilot.energy(vector, op) for name, op in pinned.items()},
        {name: obs.expectations(vector, ops).tolist() for name, ops in families.items()},
    )


def reference(output):
    """Create a checked shared reference, initial tensors, and endpoint observables."""
    plan = verify(output)
    p, folder = plan["protocol"], Path(output) / "reference"
    if (folder / "done.json").exists():
        msg = "Reference already completed; do not submit duplicate work"
        raise ValueError(msg)
    limit_worker_threads(1)
    safe_set_numba_threads(1)
    start = time.perf_counter()
    initial = fx.initial_state(p["length"], p["initial_state"], chi=p["initial_chi"], seed=p["seed"])
    pilot.save_snapshot(folder / "initial.npz", initial)
    vector = pilot.canonical_vector(initial)
    pinned, families = operators(p["length"])
    action_error = ref.check_operators(fx.hamiltonian_mpo(p["length"], "xxx"), pinned["H"])
    initial_values, _ = measurements(vector, pinned, families)
    count = 2 * p["steps"] + 1
    temporary = folder / "vectors.partial.npy"
    vectors = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.complex128, shape=(count, len(vector)))
    vectors[0] = vector
    maximum = {"norm_drift": 0.0, "energy_drift": 0.0}
    endpoint_values, endpoint_families, spin_values = [], [], []
    spin = ref.Observables(p["length"])
    for first in range(0, count - 1, 32):
        n = min(32, count - 1 - first)
        block = expm_multiply(-1j * pinned["H"], vectors[first], start=0, stop=n * p["dt"] / 2, num=n + 1, traceA=0)
        vectors[first + 1 : first + n + 1] = block[1:]
        pilot.write_json(
            folder / "progress.json",
            {"half_step": first + n, "half_steps": count - 1, "wall_seconds": time.perf_counter() - start},
        )
    for index, exact in enumerate(vectors):
        maximum["norm_drift"] = max(maximum["norm_drift"], abs(float(np.linalg.norm(exact)) - np.linalg.norm(vector)))
        maximum["energy_drift"] = max(
            maximum["energy_drift"], abs(pilot.energy(exact, pinned["H"]) - initial_values["H"])
        )
        spin_values.append(spin.values(exact).tolist())
        if index % 2 == 0:
            values, family_values = measurements(exact, pinned, families)
            endpoint_values.append(values)
            endpoint_families.append(family_values)
    if max(maximum.values()) > 1e-9:
        msg = f"Unhealthy reference: {maximum}"
        raise ValueError(msg)
    # Independent one-shot endpoint checks the time grid/chunk continuation.
    direct = expm_multiply(-1j * p["total_time"] * pinned["H"], vector, traceA=0)
    endpoint_error = float(np.linalg.norm(direct - vectors[-1]))
    if endpoint_error > 1e-9:
        msg = "Reference endpoint disagrees with direct propagation"
        raise ValueError(msg)
    vectors.flush()
    del vectors
    temporary.replace(folder / "vectors.npy")
    payload = {
        "initial_values": initial_values,
        "values": endpoint_values,
        "families": endpoint_families,
        "spin_values": spin_values,
        "health": maximum,
        "operator_action_error": action_error,
        "endpoint_check": endpoint_error,
        "initial_sha256": digest(folder / "initial.npz"),
        "vectors_sha256": digest(folder / "vectors.npy"),
        "shape": [count, len(vector)],
        "wall_seconds": time.perf_counter() - start,
    }
    pilot.write_json(folder / "data.json", payload)
    pilot.write_json(
        folder / "done.json",
        {"runtime": runtime(), "finished": time.time(), "data_sha256": digest(folder / "data.json")},
    )
    return payload


def configure(p, arm):
    """Call the legacy parameter constructor with explicit scalar fallback flags."""
    old_length = jt.LENGTH
    jt.LENGTH = p["length"]
    try:
        params = jt.make_params(arm["variant"], p["cap"], arm["integrator"])
    finally:
        jt.LENGTH = old_length
    params.dt = params.elapsed_time = p["dt"]
    params.svd_threshold, params.krylov_tol = p["svd_threshold"], p["krylov_tol"]
    params.conserve_tol = p["conserve_tol"]
    if arm["fallback"] is not None:
        params.conserve_fallback = arm["fallback"]
    return params


def residuals(state, mpo, target):
    """Measure all requested constraints through their current canonical frame."""
    if isinstance(target, conservation.EnergyTarget):
        mpos, goals = [mpo], [target.value]
    else:
        mpos, goals = target.mpos, target.values
    scale = max(1.0, *map(abs, goals))
    errors = [abs(conservation.expectation_value(state, op) - value) for op, value in zip(mpos, goals, strict=True)]
    return max(errors, default=0.0) / scale


@contextmanager
def diagnostics(p, arm, stats, folder, vectors, reference_data, pinned):
    """Observe whole correction schedules, preserving the original joint schedule.

    Joint calls/unresolved counts refer to individual center solves. The separate
    schedule residual reports success after the entire one/every-center schedule.
    Scalar qualification uses references at the actual BUG half-step or TDVP time.
    """
    original_bug, original_evolution = bug.apply_conservation, evolution.apply_conservation
    jt.install_correction(arm["sweep"])
    scheduled = bug.apply_conservation
    record = {
        "step": 0,
        "call_in_step": 0,
        "schedule_calls": 0,
        "schedule_unresolved": 0,
        "max_pre_residual": 0.0,
        "max_post_residual": 0.0,
        "correction_seconds": 0.0,
        "diagnostic_seconds": 0.0,
        "qualified_calls": 0,
        "qualified_fallbacks": 0,
        "events": [],
    }
    spin = ref.Observables(p["length"])

    def observed(state, mpo, target):
        begin = time.perf_counter()
        record["call_in_step"] += 1
        record["schedule_calls"] += 1
        half_index = 2 * (record["step"] - 1) + (record["call_in_step"] if arm["integrator"] == "bug" else 2)
        pre_residual = residuals(state, mpo, target)
        record["max_pre_residual"] = max(record["max_pre_residual"], pre_residual)
        before, pre_accuracy = None, None
        previous = stats.to_dict()
        if arm["variant"] == "pinH":
            v = pilot.canonical_vector(state)
            pre_accuracy = {
                "infidelity": pilot.infidelity(vectors[half_index], v),
                "observable_error": float(np.max(np.abs(spin.values(v) - reference_data["spin_values"][half_index]))),
            }
            pre_accuracy["qualified"] = (
                pre_accuracy["infidelity"] <= p["accuracy_infidelity"]
                and pre_accuracy["observable_error"] <= p["accuracy_observable"]
            )
            record["qualified_calls"] += int(pre_accuracy["qualified"])
            if arm["fallback"]:
                before = deepcopy(state)
        core_begin = time.perf_counter()
        changed = scheduled(state, mpo, target)
        elapsed = time.perf_counter() - core_begin
        record["correction_seconds"] += elapsed
        post = residuals(state, mpo, target)
        record["max_post_residual"] = max(record["max_post_residual"], post)
        record["schedule_unresolved"] += int(post > p["conserve_tol"])
        delta = {k: v - previous[k] for k, v in stats.to_dict().items()}
        if delta["fallback_attempts"]:
            index = len(record["events"])
            pilot.save_snapshot(folder / f"event-{index:05d}-pre.npz", before)
            pilot.save_snapshot(folder / f"event-{index:05d}-post.npz", state)
            stage = next(
                (
                    name
                    for counter, name in (
                        ("scan_successes", "scan"),
                        ("frame_successes", "full_frame"),
                        ("infeasible", "infeasible"),
                        ("solver_failures", "solver_failure"),
                        ("no_minimizer", "no_minimizer"),
                        ("gauge_failures", "gauge_failure"),
                    )
                    if delta[counter]
                ),
                "unresolved",
            )
            after_v = pilot.canonical_vector(state)
            record["events"].append({
                "time": half_index * p["dt"] / 2,
                "step": record["step"],
                "stage": stage,
                "counts": delta,
                "accuracy_pre": pre_accuracy,
                "post_infidelity": pilot.infidelity(vectors[half_index], after_v),
                "post_observable_error": float(
                    np.max(np.abs(spin.values(after_v) - reference_data["spin_values"][half_index]))
                ),
                "pre_residual": pre_residual,
                "post_residual": post,
                "bonds_preserved": before.bond_dimensions() == state.bond_dimensions(),
                "gauge_preserved": (before.orthogonality_center, before.flipped)
                == (state.orthogonality_center, state.flipped),
                **pilot.scalar_diagnostics(before, mpo, target.value),
            })
            record["qualified_fallbacks"] += int(pre_accuracy["qualified"])
        record["diagnostic_seconds"] += time.perf_counter() - begin - elapsed
        return changed

    bug.apply_conservation = evolution.apply_conservation = observed
    try:
        yield record
    finally:
        bug.apply_conservation, evolution.apply_conservation = original_bug, original_evolution


def family_errors(values, expected):
    """Retain the four families reported in manuscript Table III."""
    errors = {name: float(np.max(np.abs(np.asarray(value) - expected[name]))) for name, value in values.items()}
    return {
        "local_spin": max(errors[f"local_S{axis}"] for axis in "xyz"),
        "bond_energy": errors["bond_energy"],
        "corr_zz": errors["corr_zz"],
        "staggered": errors["staggered"],
    }


def run_arm(output, identifier):
    """Evolve one unique trajectory and produce all its table and figure data."""
    plan = verify(output)
    p = plan["protocol"]
    arm = next(a for a in plan["arms"] if a["id"] == identifier)
    output = Path(output)
    folder = output / "arms" / identifier
    folder.mkdir(exist_ok=False)
    ref_folder = output / "reference"
    done = read(ref_folder / "done.json")
    if digest(ref_folder / "data.json") != done["data_sha256"]:
        msg = "Reference metadata changed"
        raise ValueError(msg)
    reference_data = read(ref_folder / "data.json")
    if digest(ref_folder / "initial.npz") != reference_data["initial_sha256"]:
        msg = "Initial tensors changed"
        raise ValueError(msg)
    if digest(ref_folder / "vectors.npy") != reference_data["vectors_sha256"]:
        msg = "Reference vectors changed"
        raise ValueError(msg)
    vectors = np.load(ref_folder / "vectors.npy", mmap_mode="r", allow_pickle=False)
    if list(vectors.shape) != reference_data["shape"]:
        msg = "Reference shape mismatch"
        raise ValueError(msg)
    limit_worker_threads(1)
    safe_set_numba_threads(1)
    state = pilot.load_snapshot(ref_folder / "initial.npz")
    mpo = fx.hamiltonian_mpo(p["length"], "xxx")
    pinned, families = operators(p["length"])
    spin = ref.Observables(p["length"])
    params = configure(p, arm)
    target = capture_conservation_target(state, mpo, None, params)
    rows, start, step_seconds, measurement_seconds = [], time.perf_counter(), 0.0, 0.0
    jt.SOLVES["n"] = 0
    with track_conservation() as stats, diagnostics(p, arm, stats, folder, vectors, reference_data, pinned) as record:
        for step in range(p["steps"] + 1):
            record["step"], record["call_in_step"] = step, 0
            if step:
                tick = time.perf_counter()
                evolution.apply_unitary_evolution(state, mpo, params, conservation_target=target)
                step_seconds += time.perf_counter() - tick
                expected = (2 if arm["integrator"] == "bug" else 1) if target is not None else 0
                if record["call_in_step"] != expected:
                    msg = "Correction schedule count changed"
                    raise ValueError(msg)
            tick = time.perf_counter()
            vector = pilot.canonical_vector(state)
            if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0:
                msg = "Non-finite or zero state"
                raise ValueError(msg)
            if max(state.bond_dimensions()) > p["cap"]:
                msg = "Bond cap exceeded"
                raise ValueError(msg)
            values, family_values = measurements(vector, pinned, families)
            rows.append({
                "time": step * p["dt"],
                "values": values,
                "families": family_values,
                "family_errors": family_errors(family_values, reference_data["families"][step]),
                "infidelity": pilot.infidelity(vectors[2 * step], vector),
                "spin_observable_error": float(
                    np.max(np.abs(spin.values(vector) - reference_data["spin_values"][2 * step]))
                ),
                "bond": int(max(state.bond_dimensions())),
                "norm": float(np.linalg.norm(vector)),
            })
            measurement_seconds += time.perf_counter() - tick
            if step % 10 == 0 or step == p["steps"]:
                pilot.write_json(
                    folder / "progress.json",
                    {
                        "step": step,
                        "steps": p["steps"],
                        "time": step * p["dt"],
                        "counts": stats.to_dict(),
                        "schedule_unresolved": record["schedule_unresolved"],
                        "wall_seconds": time.perf_counter() - start,
                    },
                )
    counts = stats.to_dict()
    if len(record["events"]) != counts["fallback_attempts"] or any(
        not e["bonds_preserved"] or not e["gauge_preserved"] for e in record["events"]
    ):
        msg = "Fallback event integrity failure"
        raise ValueError(msg)
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "arm": arm,
        "protocol": p,
        "runtime": runtime(),
        "initial_sha256": reference_data["initial_sha256"],
        "reference_sha256": reference_data["vectors_sha256"],
        "initial_values": reference_data["initial_values"],
        "trace": rows,
        "counts": counts,
        "diagnostics": record,
        "solves": jt.SOLVES["n"],
        "timing": {
            "wall_seconds": time.perf_counter() - start,
            "instrumented_step_seconds": step_seconds,
            "measurement_seconds": measurement_seconds,
            "estimated_evolution_seconds": step_seconds - record["diagnostic_seconds"],
            "note": "Includes tracking overhead; not an uninstrumented algorithm benchmark.",
        },
    }
    pilot.save_snapshot(folder / "final.npz", state)
    pilot.write_json(folder / "result.json", result)
    pilot.write_json(folder / "done.json", {"finished": time.time(), "result_sha256": digest(folder / "result.json")})
    return result


def main(argv=None) -> int:
    """Run only the explicitly selected pipeline stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "prepare", "reference", "run", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--manuscript", type=Path)
    parser.add_argument("--arm")
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--cap", type=int, default=32)
    parser.add_argument("--total-time", type=float, default=6.0)
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args(argv)
    if args.command == "plan":
        if args.output.exists():
            msg = "Plan already exists"
            raise ValueError(msg)
        pilot.write_json(
            args.output,
            make_plan(
                length=args.length, cap=args.cap, total_time=args.total_time, dt=args.dt, manuscript=args.manuscript
            ),
        )
    elif args.command == "prepare":
        prepare(read(args.plan), args.output)
    elif args.command == "reference":
        reference(args.output)
    elif args.command == "run":
        run_arm(args.output, args.arm)
    else:
        from paper_reproduction_report import report

        report(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
