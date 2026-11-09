#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
# SPDX-License-Identifier: MIT
# Licensed under the MIT License

"""Explicit distributed accuracy runs using an atomic shared-filesystem queue.

One coordinator prepares the queue. Independent low-priority process workers
claim whole trajectories on any host sharing the output directory. Claims are
never stolen automatically: a failed or interrupted job requires inspection.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import multiprocessing
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import fallback_accuracy as accuracy
import fallback_pilot as pilot


def read(path):
    """Read one atomically published JSON artifact."""
    return json.loads(Path(path).read_text())


def identity():
    """Record the actual worker environment, including host and affinity."""
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "numba", "qiskit")},
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }


def verify_worker(output):
    """Reject workers with different scientific sources or dependency versions."""
    signature = read(output / "manifest.json")["signature"]
    current = {str(p.relative_to(pilot.ROOT)): accuracy.digest(p) for p in pilot.source_files()}
    if current != signature["source_sha256"] or identity()["versions"] != signature["runtime"]["versions"]:
        raise ValueError("Worker code or numerical dependencies differ from the prepared study")
    return signature["plan"]


def prepare(plan, output, estimates=None):
    """Pin provenance and publish longest-first jobs without running trajectories."""
    output = accuracy.initialize_output(plan, output, mode="run")
    weights = {}
    if estimates:
        for case in estimates["cases"]:
            for arm in case["arms"]:
                weights[case["key"], arm["variant"]] = arm["trajectory_seconds"]
    jobs = [
        {
            "id": f"{spec['key']}--{variant}",
            "specification": spec,
            "variant": variant,
            "weight": weights.get((spec["key"], variant), spec["case"]["length"] ** 3 / spec["case"]["dt"]),
        }
        for spec in plan["studies"]
        for variant in pilot.VARIANTS
    ]
    jobs.sort(key=lambda job: (-job["weight"], job["id"]))
    path = output / "queue.json"
    if path.exists() and read(path) != jobs:
        raise ValueError("Queue has changed; use a fresh output directory")
    for name in ("claims", "jobs", "hosts"):
        (output / name).mkdir(exist_ok=True)
    pilot.write_json(path, jobs)
    return jobs


def claim(output, jobs):
    """Use atomic mkdir to give each trajectory exactly one process owner."""
    for job in jobs:
        folder = output / "claims" / job["id"]
        try:
            folder.mkdir()
        except FileExistsError:
            continue
        pilot.write_json(folder / "owner.json", {**identity(), "started": time.time()})
        return job
    return None


def available_gib():
    """Read Linux memory availability, not summed process RSS or free pages."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("Linux MemAvailable is required for memory admission")


def physical_cpus():
    """Choose one available logical CPU per physical core, interleaving NUMA nodes."""
    allowed = os.sched_getaffinity(0)
    rows = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET,NODE"], text=True)
    cores = {}
    for line in rows.splitlines():
        if line.startswith("#"):
            continue
        cpu, core, socket_id, node = map(int, line.split(","))
        if cpu in allowed:
            cores.setdefault((socket_id, core), (node, cpu))
    nodes = {}
    for node, cpu in cores.values():
        nodes.setdefault(node, []).append(cpu)
    return [
        values[index]
        for index in range(max(map(len, nodes.values())))
        for _, values in sorted(nodes.items())
        if index < len(values)
    ]


def worker(output, cpu, reserve_gib):
    """Process isolated arms at low priority, with no action on other processes."""
    os.nice(19)
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    plan = verify_worker(output)
    jobs = read(output / "queue.json")
    while True:
        # Reserve is deliberately conservative; already running arms finish.
        while available_gib() < reserve_gib:
            time.sleep(15)
        job = claim(output, jobs)
        if job is None:
            return
        folder = output / "jobs" / job["id"]
        folder.mkdir()
        owner = identity()
        try:
            accuracy.run_case(
                job["specification"],
                plan["source_root"],
                folder,
                accuracy.ref.Accuracy(**plan["thresholds"]),
                variants=[job["variant"]],
            )
            pilot.write_json(folder / "done.json", {**owner, "finished": time.time()})
        except Exception:
            pilot.write_json(
                folder / "failed.json", {**owner, "finished": time.time(), "traceback": traceback.format_exc()}
            )


def host(output, workers, reserve_gib):
    """Start a bounded process pool and preserve an exclusive host launch record."""
    verify_worker(output)
    cpus = physical_cpus()
    if workers < 1 or workers > len(cpus) - 8 or reserve_gib < 64:
        raise ValueError("Leave at least eight physical cores and 64 GiB memory headroom")
    folder = output / "hosts" / socket.gethostname()
    folder.mkdir()
    pilot.write_json(
        folder / "started.json",
        {**identity(), "workers": workers, "cpus": cpus[:workers], "reserve_gib": reserve_gib, "started": time.time()},
    )
    context = multiprocessing.get_context("spawn")
    processes = []
    for cpu in cpus[:workers]:
        process = context.Process(target=worker, args=(output, cpu, reserve_gib))
        process.start()
        processes.append(process)
        time.sleep(0.2)
    for process in processes:
        process.join()
    pilot.write_json(folder / "finished.json", {"finished": time.time(), "exitcodes": [p.exitcode for p in processes]})


def collect(output):
    """Assemble compatible completed triples in the existing reporting format."""
    plan = read(output / "manifest.json")["signature"]["plan"]
    jobs = read(output / "queue.json")
    for spec in plan["studies"]:
        destination = output / "cases" / spec["key"]
        if (destination / "result.json").exists():
            continue
        folders = [output / "jobs" / f"{spec['key']}--{variant}" for variant in pilot.VARIANTS]
        if not all((folder / "done.json").exists() for folder in folders):
            continue
        results = [read(folder / "cases" / spec["key"] / "result.json") for folder in folders]
        first = results[0]
        for variant, result in zip(pilot.VARIANTS, results, strict=True):
            for field in ("schema", "key", "case", "specification", "thresholds", "initial_vector_sha256"):
                if result[field] != first[field]:
                    raise ValueError(f"Mismatched trajectory {field}")
            if result["status"] != "complete" or [a["variant"] for a in result["arms"]] != [variant]:
                raise ValueError("Incomplete or mislabeled trajectory")
            if abs(result["hamiltonian"]["scale"] - first["hamiltonian"]["scale"]) > 1e-13:
                raise ValueError("Trajectory Hamiltonian scales differ")
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folders[0] / "cases" / spec["key"] / "initial.npz", destination / "initial.npz")
        for variant, folder in zip(pilot.VARIANTS, folders, strict=True):
            shutil.copytree(folder / "cases" / spec["key"] / variant, destination / variant, dirs_exist_ok=True)
        first["arms"] = [result["arms"][0] for result in results]
        first["workers"] = [read(folder / "done.json") for folder in folders]
        pilot.write_json(destination / "result.json", first)
    done = sum((output / "jobs" / job["id"] / "done.json").exists() for job in jobs)
    failed = sum((output / "jobs" / job["id"] / "failed.json").exists() for job in jobs)
    claimed = sum((output / "claims" / job["id"]).exists() for job in jobs)
    status = {
        "total": len(jobs),
        "completed": done,
        "failed": failed,
        "running_or_interrupted": claimed - done - failed,
        "unclaimed": len(jobs) - claimed,
        "updated": time.time(),
    }
    pilot.write_json(output / "dispatch-status.json", status)
    return status


def watch(output):
    """Publish progress and produce the full report when every arm completes."""
    while True:
        status = collect(output)
        print(json.dumps(status), flush=True)
        if status["completed"] + status["failed"] == status["total"]:
            from fallback_accuracy_report import report

            report(output)
            return
        time.sleep(30)


def main():
    """Require separate explicit preparation, host launch, and monitoring actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "host", "collect", "watch"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--estimates", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--reserve-gib", type=float, default=96)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(read(args.plan), args.output, read(args.estimates) if args.estimates else None)
    elif args.command == "host":
        host(args.output, args.workers, args.reserve_gib)
    elif args.command == "collect":
        print(json.dumps(collect(args.output)))
    else:
        watch(args.output)


if __name__ == "__main__":
    main()
