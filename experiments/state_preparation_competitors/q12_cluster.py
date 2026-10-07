# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Prepare, stage and explicitly launch a capacity-aware shared Leonhard queue.

The default action is a read-only capacity plan. Only ``launch`` starts remote
supervisors; ``work`` and ``supervise`` are their internal execution commands.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from experiments import topdown_backend_support as checkpoints
from experiments.state_preparation_competitors import q12_campaign as campaign
from experiments.state_preparation_competitors.bounded_probe import run_bounded
from experiments.state_preparation_competitors.run import ROOT, digest, write_json

MODULE = "experiments.state_preparation_competitors.q12_cluster"
THREAD_ENV = dict.fromkeys(
    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"
)
REMOTE_BASE = "/Home/varioper/froehlich/yaqs-checks/q12-logical-matrix-20260928"
REMOTE_PYTHON = "/Home/varioper/froehlich/yaqs-checks/q12-20260927-batches/.venv/bin/python"
PROBE = """import os,json,socket,time,importlib.metadata
m=dict((k,int(v.split()[0])*1024) for k,v in (x.split(':',1) for x in open('/proc/meminfo')))
print(json.dumps(dict(host=socket.gethostname(),cpus=os.cpu_count(),load=os.getloadavg(),
available_bytes=m['MemAvailable'],total_bytes=m['MemTotal'],at_unix=time.time(),
packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','qiskit','threadpoolctl']})))
"""


def prepare(root: Path) -> dict:
    """Freeze the agreed matrix and deployment settings without contacting nodes."""
    if root.exists():
        msg = "Prepare requires a new campaign output directory."
        raise ValueError(msg)
    settings = campaign.protocol()
    manifest = {
        "schema": "yaqs.q12.logical_cluster.v1",
        "created_unix": time.time(),
        "protocol": settings,
        "jobs": campaign.jobs(settings),
        "sources": campaign.source_hashes(),
        "packages": {name: importlib.metadata.version(name) for name in campaign.PACKAGES},
        "execution": {
            "max_workers_per_host": 16,
            "memory_bytes_per_job": 16 * 1024**3,
            "seconds_per_job": 7 * 24 * 3600,
            "memory_headroom_fraction": 0.1,
            "thread_environment": THREAD_ENV,
            "priority": "longest-estimated-first, then ready finalizers",
            "retry": "no automatic retries; completed stages are reusable",
        },
        "deployment": {
            "base": REMOTE_BASE,
            "python": REMOTE_PYTHON,
            "hosts": {
                "leonhard-13": [],
                "leonhard-16": [
                    "-o",
                    f"UserKnownHostsFile={ROOT / 'output/q12-remote-checks-20260927/cluster_known_hosts'}",
                ],
            },
        },
    }
    write_json(root / "manifest.json", manifest)
    (root / "manifest.sha256").write_text(digest(root / "manifest.json") + "\n")
    return manifest


def ssh_arguments(host: str, manifest: dict) -> list[str]:
    """Return a strict, noninteractive argv for one explicitly configured host."""
    if host not in manifest["deployment"]["hosts"]:
        msg = "Host is not part of the frozen deployment."
        raise ValueError(msg)
    return [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=12",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        *manifest["deployment"]["hosts"][host],
    ]


def invoke(argv: list[str]) -> str:
    """Run explicit administrative argv and retain diagnostic stderr on failure."""
    result = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=120)  # noqa: S603 -- Internal administrative argv, never a shell command.
    return result.stdout


def probe(host: str, manifest: dict) -> dict:
    """Read current load, RAM and installed numerical versions without training."""
    command = shlex.join([manifest["deployment"]["python"], "-c", PROBE])
    result = json.loads(invoke(["ssh", *ssh_arguments(host, manifest), host, command]))
    if result["host"].split(".")[0] != host or result["packages"] != manifest["packages"]:
        msg = f"Host identity or numerical environment mismatch on {host}."
        raise ValueError(msg)
    return result


def slots(snapshot: dict, execution: dict) -> int:
    """Bound concurrency by free RAM, load and reserved physical-core headroom.

    Counting half the logical CPUs is conservative on the verified HT Xeons.
    Availability is advisory; it is rechecked before each additional dispatch.
    """
    memory = snapshot["available_bytes"] - execution["memory_headroom_fraction"] * snapshot["total_bytes"]
    cpu = math.floor(0.75 * snapshot["cpus"] / 2 - max(snapshot["load"]))
    return max(0, min(execution["max_workers_per_host"], cpu, int(memory // execution["memory_bytes_per_job"])))


def probe_capacity(host: str, manifest: dict) -> dict:
    """Exclude unavailable or incompatible nodes while retaining their diagnostics."""
    try:
        snapshot = probe(host, manifest)
        return {"snapshot": snapshot, "slots": slots(snapshot, manifest["execution"])}
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        return {"slots": 0, "error": str(error)}


def plan(root: Path, manifest: dict) -> dict:
    """Record fresh capacity and an initial balanced assignment; never dispatch."""
    hosts = {host: probe_capacity(host, manifest) for host in manifest["deployment"]["hosts"]}
    assignments = dict.fromkeys(hosts, 0)
    initial = []
    for job in sorted((j for j in manifest["jobs"] if not j["dependencies"]), key=lambda j: (-j["priority"], j["id"])):
        available = [host for host, info in hosts.items() if assignments[host] < info["slots"]]
        if not available:
            break
        host = min(available, key=lambda h: (assignments[h] / hosts[h]["slots"], h))
        assignments[host] += 1
        initial.append({"job": job["id"], "host": host})
    result = {
        "at_unix": time.time(),
        "launched": False,
        "hosts": hosts,
        "initial_assignment_preview": initial,
        "training_jobs": sum(j["kind"] != "final" for j in manifest["jobs"]),
        "finalizers": sum(j["kind"] == "final" for j in manifest["jobs"]),
        "dispatch": "dynamic shared queue; preview is not a fixed allocation",
    }
    write_json(root / "capacity-plan.json", result)
    return result


def stage(root: Path, manifest: dict) -> None:
    """Stage an immutable source bundle on shared NFS, without starting workers."""
    archive = root / "source-bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for path in manifest["sources"]:
            stream.add(ROOT / path, arcname=f"source/{path}")
        for name in ("manifest.json", "manifest.sha256"):
            stream.add(root / name, arcname=f"campaign/{name}")
    host = next(iter(manifest["deployment"]["hosts"]))
    remote = manifest["deployment"]["base"]
    options = ssh_arguments(host, manifest)
    # Refuse overwrite, including partial stages, rather than mutating an active snapshot.
    invoke(["ssh", *options, host, shlex.join(["mkdir", remote])])
    invoke(["scp", *options, str(archive), f"{host}:{remote}/source-bundle.tar.gz"])
    invoke(["ssh", *options, host, shlex.join(["tar", "-xzf", f"{remote}/source-bundle.tar.gz", "-C", remote])])
    for node in manifest["deployment"]["hosts"]:
        command = remote_command(manifest, "check")
        invoke(["ssh", *ssh_arguments(node, manifest), node, command])
    write_json(root / "staged.json", {"remote": remote, "archive_sha256": digest(archive), "launched": False})


def remote_command(manifest: dict, action: str) -> str:
    """Quote a remote command with fixed one-thread environments and source path."""
    base = manifest["deployment"]["base"]
    env = {**THREAD_ENV, "PYTHONPATH": f"{base}/source:{base}/source/src"}
    argv = [
        "env",
        *(f"{key}={value}" for key, value in env.items()),
        "nice",
        "-n",
        "10",
        manifest["deployment"]["python"],
        "-u",
        "-m",
        MODULE,
        action,
        "--root",
        f"{base}/campaign",
    ]
    return shlex.join(argv)


def launch(root: Path, manifest: dict) -> dict:
    """Explicitly start detached supervisors only after staging and fresh checks."""
    if (root / "launch.json").exists():
        msg = "An existing launch receipt requires inspection before relaunch."
        raise ValueError(msg)
    if not (root / "staged.json").exists():
        msg = "Stage and verify the frozen snapshot before launch."
        raise ValueError(msg)
    capacity = plan(root, manifest)
    if not any(info["slots"] for info in capacity["hosts"].values()):
        msg = "No configured host currently has validated capacity."
        raise ValueError(msg)
    receipt = {}
    for host, info in capacity["hosts"].items():
        if not info["slots"]:
            continue
        options = ssh_arguments(host, manifest)
        invoke(["ssh", *options, host, remote_command(manifest, "check")])
        log = f"{manifest['deployment']['base']}/campaign/supervisor-{host}.log"
        command = f"nohup {remote_command(manifest, 'supervise')} > {shlex.quote(log)} 2>&1 < /dev/null & echo $!"
        pid = invoke(["ssh", *options, host, command]).strip()
        receipt[host] = {"supervisor_pid": pid, "log": log, "capacity_at_dispatch": info["slots"]}
        write_json(root / "launch.json", receipt)
    return receipt


def claim(root: Path, manifest: dict) -> dict | None:
    """Atomically claim a ready job on NFS, without expiring running-worker locks."""
    claims = root / "claims"
    claims.mkdir(exist_ok=True)
    ordered = sorted(manifest["jobs"], key=lambda j: (-j["priority"], j["id"]))
    hosts = sorted(manifest.get("deployment", {}).get("hosts", {}))
    independent = [job for job in ordered if not job["dependencies"]]
    preferred = {job["id"]: hosts[i % len(hosts)] for i, job in enumerate(independent)} if hosts else {}
    host = socket.gethostname().split(".")[0]
    # Balanced first-wave affinity avoids filling the first node with all long jobs.
    # After its preferred jobs are claimed, a node can steal any remaining work.
    ordered.sort(key=lambda j: (j["kind"] != "final", preferred.get(j["id"], host) != host, -j["priority"], j["id"]))
    for job in ordered:
        identity = job["id"]
        if any((root / marker / f"{identity}.json").exists() for marker in ("done", "failed")):
            continue
        if not all((root / "done" / f"{dependency}.json").exists() for dependency in job["dependencies"]):
            continue
        lock = claims / identity
        try:
            lock.mkdir()
        except FileExistsError:
            continue
        write_json(
            lock / "owner.json",
            {"host": socket.gethostname(), "supervisor_pid": os.getpid(), "claimed_unix": time.time()},
        )
        return job
    return None


def guarded_job(root: Path, manifest: dict, job: dict) -> dict:
    """Run a claimed job in its own guarded process and publish completion last."""
    identity = job["id"]
    command = [sys.executable, "-u", "-m", MODULE, "work", "--root", str(root), "--job", identity]
    last_heartbeat = -float("inf")

    def heartbeat(pid: int, rss: int, physical: int) -> None:
        """Record the actual child PID and periodic memory observations for recovery."""
        nonlocal last_heartbeat
        if time.monotonic() - last_heartbeat >= 60:
            write_json(
                root / "claims" / identity / "worker.json",
                {
                    "host": socket.gethostname(),
                    "pid": pid,
                    "supervisor_pid": os.getpid(),
                    "observed_unix": time.time(),
                    "rss_bytes": rss,
                    "physical_bytes": physical,
                    "command": command,
                },
            )
            last_heartbeat = time.monotonic()

    receipt = run_bounded(
        command,
        root / "logs" / f"{identity}.log",
        seconds=manifest["execution"]["seconds_per_job"],
        memory_bytes=manifest["execution"]["memory_bytes_per_job"],
        observer=heartbeat,
    )
    receipt.update({"host": socket.gethostname(), "job": identity, "completed_unix": time.time()})
    marker = "failed"
    if receipt["status"] == "completed":
        result = checkpoints.load_checkpoint(root / "jobs" / identity / "result")
        if result is None:
            receipt["status"] = "missing_result"
        else:
            marker = "done"
            receipt["scientific_status"] = result[0]["status"]
            receipt["result_seal_sha256"] = digest(root / "jobs" / identity / "result" / "complete.json")
    write_json(root / marker / f"{identity}.json", receipt)
    # Leave claims as durable ownership records, including on failure.
    return receipt


def local_snapshot() -> dict:
    """Read Linux host capacity without an SSH connection or numerical work."""
    memory = {
        key: int(value.split()[0]) * 1024
        for key, value in (
            line.split(":", 1) for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        )
    }
    return {
        "cpus": os.cpu_count(),
        "load": os.getloadavg(),
        "available_bytes": memory["MemAvailable"],
        "total_bytes": memory["MemTotal"],
    }


def status(root: Path, manifest: dict) -> dict:
    """Report completion, failures and blocked dependencies without launching."""
    result = {}
    for job in manifest["jobs"]:
        identity = job["id"]
        if (root / "done" / f"{identity}.json").exists():
            receipt = json.loads((root / "done" / f"{identity}.json").read_text())
            folder = root / "jobs" / identity / "result"
            checkpoints.load_checkpoint(folder)
            if digest(folder / "complete.json") != receipt["result_seal_sha256"]:
                msg = "Completed result differs from dispatch receipt."
                raise ValueError(msg)
            result[identity] = "done"
        elif (root / "failed" / f"{identity}.json").exists():
            result[identity] = "failed"
        elif (root / "claims" / identity).exists():
            result[identity] = "claimed; consult owner and process logs"
        else:
            result[identity] = "pending"
    return result


def _supervise_jobs(root: Path, manifest: dict, pool: ThreadPoolExecutor, active: list[Future]) -> None:
    """Schedule pending jobs until completion or failed prerequisites stop progress."""
    while True:
        for future in active.copy():
            if future.done():
                future.result()
                active.remove(future)
        available = slots(local_snapshot(), manifest["execution"])
        while len(active) < available:
            job = claim(root, manifest)
            if job is None:
                break
            active.append(pool.submit(guarded_job, root, manifest, job))
        remaining = status(root, manifest)
        if not active and all(value != "pending" for value in remaining.values()):
            break
        # Failed prerequisites cannot produce their finalizer; other nodes may still be busy.
        unfinished = [j for j in manifest["jobs"] if remaining[j["id"]] == "pending"]
        if (
            not active
            and unfinished
            and all(any(remaining[d] == "failed" for d in j["dependencies"]) for j in unfinished)
        ):
            break
        time.sleep(15)


def supervise(root: Path, manifest: dict) -> None:
    """Fill available slots dynamically; keep interrupted ownership for inspection."""
    host = socket.gethostname().split(".")[0]
    if host not in manifest["deployment"]["hosts"]:
        msg = "Supervisor is running on an unconfigured host."
        raise ValueError(msg)
    lock = root / f"supervisor-{host}.lock"
    lock.mkdir()  # A second supervisor cannot oversubscribe this campaign on the same node.
    write_json(lock / "owner.json", {"pid": os.getpid(), "host": host, "started_unix": time.time()})
    os.environ.update(THREAD_ENV)
    active: list[Future] = []
    try:
        with ThreadPoolExecutor(max_workers=manifest["execution"]["max_workers_per_host"]) as pool:
            _supervise_jobs(root, manifest, pool, active)
    finally:
        shutil.rmtree(lock)


def main(argv: list[str] | None = None) -> None:
    """Expose preparation and read-only planning separately from explicit launch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        default="plan",
        choices=("prepare", "plan", "stage", "launch", "check", "supervise", "work", "status"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job")
    args = parser.parse_args(argv)
    if args.action == "prepare":
        manifest = prepare(args.root)
        print(json.dumps({"prepared": str(args.root), "jobs": len(manifest["jobs"]), "launched": False}))
        return
    manifest = campaign.authenticate(args.root)
    if args.action == "work" and args.job not in {job["id"] for job in manifest["jobs"]}:
        parser.error("work requires --job with an identifier from the frozen manifest")
    if args.action == "plan":
        result = plan(args.root, manifest)
    elif args.action == "stage":
        stage(args.root, manifest)
        result = {"staged": True, "launched": False}
    elif args.action == "launch":
        result = launch(args.root, manifest)
    elif args.action == "work":
        result = campaign.execute(args.root, args.job)
    elif args.action == "supervise":
        supervise(args.root, manifest)
        result = status(args.root, manifest)
    elif args.action == "status":
        result = status(args.root, manifest)
    else:
        result = {"authenticated": True, "launched": False}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
