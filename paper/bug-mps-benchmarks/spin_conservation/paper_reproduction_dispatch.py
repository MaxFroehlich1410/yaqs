#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
# SPDX-License-Identifier: MIT
# Licensed under the MIT License

"""An explicit shared-filesystem queue for the manuscript reproduction.

Workers change only their own priority and affinity. Claims are never stolen;
failed or interrupted jobs require inspection. No server jobs are launched by
planning, preparation, inventory or reporting.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import socket
import time
import traceback
from pathlib import Path

import fallback_dispatch as resources
import paper_reproduction as run


def queue(plan, estimates=None):
    """Share one prerequisite reference and schedule expensive trajectories first."""
    estimates = estimates or {}
    jobs = [{"id": "reference", "kind": "reference", "weight": float("inf")}]
    arms = [
        {
            "id": a["id"],
            "kind": "arm",
            "weight": estimates.get(
                a["id"],
                (16 if a["sweep"] == "full" else 1)
                * (3 if a["variant"] == "joint5" else 2 if a["variant"].startswith("joint") else 1),
            ),
        }
        for a in plan["arms"]
    ]
    jobs.extend(sorted(arms, key=lambda j: (-j["weight"], j["id"])))
    return jobs


def claim(output, jobs):
    """Atomically assign one eligible job, never resubmitting existing claims."""
    output = Path(output)
    ready = (output / "reference" / "done.json").exists()
    for job in jobs:
        if job["kind"] == "arm" and not ready:
            continue
        path = output / "claims" / job["id"]
        try:
            path.mkdir()
        except FileExistsError:
            continue
        run.pilot.write_json(path / "owner.json", {**run.runtime(), "started": time.time()})
        return job
    return None


def state(output):
    """Distinguish queued, claimed, failed, and successfully finished trajectories."""
    output = Path(output)
    plan = run.read(output / "manifest.json")["plan"]
    records = []
    for job in queue(plan):
        claim_path = output / "claims" / job["id"]
        destination = output / "reference" if job["kind"] == "reference" else output / "arms" / job["id"]
        status = (
            "failed"
            if (claim_path / "failed.json").exists()
            else "complete"
            if (destination / "done.json").exists()
            else "claimed_or_interrupted"
            if claim_path.exists()
            else "queued"
        )
        progress_path = destination / "progress.json"
        records.append({
            "id": job["id"],
            "kind": job["kind"],
            "status": status,
            "progress": run.read(progress_path) if progress_path.exists() else None,
        })
    counts = {
        s: sum(r["status"] == s for r in records) for s in ("complete", "failed", "claimed_or_interrupted", "queued")
    }
    return {"updated": time.time(), "jobs": records, "counts": counts, "total": len(records)}


def worker(output, cpu, reserve_gib, allowed=None) -> None:
    """Process a single-thread queue at low priority without touching other jobs."""
    os.nice(19)
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    plan = run.verify(output)
    jobs = queue(plan)
    if allowed is not None:
        jobs = [j for j in jobs if j["kind"] == "reference" or j["id"] in allowed]
    while True:
        snapshot = state(output)
        if snapshot["counts"]["failed"]:
            return
        if snapshot["counts"]["complete"] == snapshot["total"]:
            return
        if resources.available_gib() < reserve_gib:
            time.sleep(10)
            continue
        job = claim(output, jobs)
        if job is None:
            # Other workers own all remaining work, or the shared reference is pending.
            if not any(not (Path(output) / "claims" / j["id"]).exists() for j in jobs):
                return
            time.sleep(2)
            continue
        try:
            if job["kind"] == "reference":
                run.reference(output)
            else:
                run.run_arm(output, job["id"])
        except Exception:
            run.pilot.write_json(
                Path(output) / "claims" / job["id"] / "failed.json",
                {"finished": time.time(), "runtime": run.runtime(), "traceback": traceback.format_exc()},
            )
            return


def host(output, workers, reserve_gib=96.0, allowed=None) -> None:
    """Start an explicitly sized host pool, leaving physical-core/memory headroom."""
    plan = run.verify(output)
    if allowed is not None and (not allowed or not set(allowed) <= {a["id"] for a in plan["arms"]}):
        raise ValueError("Unknown or empty host arm assignment")
    cpus = resources.physical_cpus()
    if not 1 <= workers <= min(13, len(cpus) - 8) or reserve_gib < 64:
        msg = "At most 13 workers; leave eight physical cores and at least 64 GiB memory"
        raise ValueError(msg)
    folder = Path(output) / "hosts" / socket.gethostname()
    folder.mkdir(exist_ok=False)
    run.pilot.write_json(
        folder / "started.json",
        {
            **run.runtime(),
            "workers": workers,
            "cpus": cpus[:workers],
            "reserve_gib": reserve_gib,
            "allowed_arms": allowed,
            "started": time.time(),
        },
    )
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=worker, args=(Path(output), cpu, reserve_gib, allowed)) for cpu in cpus[:workers]
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    run.pilot.write_json(
        folder / "finished.json", {"finished": time.time(), "exitcodes": [p.exitcode for p in processes]}
    )


def watch(output) -> None:
    """Report completion only after all 13 arms pass collection validation."""
    while True:
        snapshot = state(output)
        run.pilot.write_json(Path(output) / "dispatch-status.json", snapshot)
        if snapshot["counts"]["failed"]:
            msg = "A job failed; inspect its saved traceback. No jobs were restarted."
            raise RuntimeError(msg)
        if snapshot["counts"]["complete"] == snapshot["total"]:
            from paper_reproduction_report import report

            report(output)
            return
        time.sleep(30)


def main(argv=None) -> int:
    """Require an explicit host or watch command before launching processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "host", "watch"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--reserve-gib", type=float, default=96)
    parser.add_argument("--arms", help="Optional comma-separated exclusive trajectory assignment for this host")
    args = parser.parse_args(argv)
    if args.command == "host":
        host(args.output, args.workers, args.reserve_gib, args.arms.split(",") if args.arms else None)
    elif args.command == "watch":
        watch(args.output)
    else:
        print(run.json.dumps(state(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
