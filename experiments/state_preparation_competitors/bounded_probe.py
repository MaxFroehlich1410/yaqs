# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Bound one numerical worker while retaining its output and process receipt."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class _Usage(ctypes.Structure):
    """macOS RUSAGE_INFO_V0 layout, including physical footprint."""

    _fields_ = [("uuid", ctypes.c_ubyte * 16)] + [
        (name, ctypes.c_uint64)
        for name in ("user", "system", "idle", "interrupt", "pageins", "wired", "resident", "physical", "start", "end")
    ]


def process_memory(pid: int) -> tuple[int, int]:
    """Read worker RSS and, on macOS, physical footprint including compressed RAM."""
    executable = shutil.which("ps")
    if executable is None:
        msg = "The resource guard requires ps."
        raise RuntimeError(msg)
    result = subprocess.run(  # noqa: S603 -- fixed executable/flags and integer PID; no shell
        [executable, "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    rss = int(result.stdout.strip() or 0) * 1024
    physical = 0
    if sys.platform == "darwin":
        library = ctypes.CDLL("/usr/lib/libproc.dylib")
        usage = _Usage()
        if library.proc_pid_rusage(pid, 0, ctypes.byref(usage)) == 0:
            physical = int(usage.physical)
    return rss, physical


def _monitor(
    process: subprocess.Popen,
    start: float,
    seconds: float,
    memory_bytes: int,
    observer: Callable[[int, int, int], str | None] | None,
) -> tuple[int, int, str]:
    """Poll one worker; return a resource-limit reason without killing unrelated work."""
    peak_rss, peak_physical, status = 0, 0, "running"
    while process.poll() is None:
        rss, physical = process_memory(process.pid)
        peak_rss, peak_physical = max(rss, peak_rss), max(physical, peak_physical)
        if observer is not None and (reason := observer(process.pid, rss, physical)) is not None:
            status = reason
            break
        if max(peak_rss, peak_physical) > memory_bytes or time.monotonic() - start > seconds:
            status = "memory_limit" if max(peak_rss, peak_physical) > memory_bytes else "time_limit"
            break
        time.sleep(0.25)
    return peak_rss, peak_physical, status


def run_bounded(
    command: list[str],
    log: Path,
    *,
    seconds: float = 5400,
    memory_bytes: int = 8 * 1024**3,
    observer: Callable[[int, int, int], str | None] | None = None,
) -> dict:
    """Run an explicit argv without a shell and stop only that worker on limits."""
    if not command or not math.isfinite(seconds) or seconds <= 0 or type(memory_bytes) is not int or memory_bytes < 1:
        msg = "Require a command, positive deadline, and positive byte limit."
        raise ValueError(msg)
    if observer is not None and not callable(observer):
        msg = "observer must be callable."
        raise TypeError(msg)
    log.parent.mkdir(parents=True, exist_ok=True)
    start, peak_rss, peak_physical, status = time.monotonic(), 0, 0, "running"
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)  # noqa: S603 -- explicit local argv
        try:
            peak_rss, peak_physical, status = _monitor(process, start, seconds, memory_bytes, observer)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    if status == "running":
        status = "completed" if process.returncode == 0 else "error"
    result = {
        "command": command,
        "status": status,
        "exit_code": process.returncode,
        "wall_seconds": time.monotonic() - start,
        "timeout_seconds": seconds,
        "sampled_peak_rss_bytes": peak_rss,
        "sampled_peak_physical_bytes": peak_physical,
        "memory_limit_bytes": memory_bytes,
    }
    log.with_suffix(".process.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> None:
    """Guard the supplied command, with a separate budget for each invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=5400)
    parser.add_argument("--memory-gib", type=float, default=8)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    result = run_bounded(command, args.log, seconds=args.seconds, memory_bytes=int(args.memory_gib * 1024**3))
    print(json.dumps(result), flush=True)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
