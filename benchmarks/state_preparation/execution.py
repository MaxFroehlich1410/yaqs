# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Bounded process execution for independent, deterministically seeded jobs."""

from __future__ import annotations

import argparse
import itertools
import multiprocessing
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Literal, TypeVar

from threadpoolctl import threadpool_limits

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

WorkerSetting = int | Literal["auto"]
Job = TypeVar("Job")
Result = TypeVar("Result")
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def resolve_workers(workers: WorkerSetting, job_count: int) -> int:
    """Resolve a worker request, capped by CPUs and independent pending jobs.

    Args:
        workers: Positive process count or ``"auto"`` for all available CPUs.
        job_count: Number of independent pending jobs.

    Returns:
        Effective worker count, or zero for an empty queue.

    Raises:
        ValueError: If the worker request or job count is invalid.
    """
    if workers != "auto" and (type(workers) is not int or workers < 1):
        msg = "workers must be a positive integer or 'auto'."
        raise ValueError(msg)
    if type(job_count) is not int or job_count < 0:
        msg = "job_count must be a nonnegative integer."
        raise ValueError(msg)
    available = os.cpu_count() or 1
    requested = available if workers == "auto" else workers
    return min(requested, available, job_count)


def parse_workers(value: str) -> WorkerSetting:
    """Parse the CLI worker count without accepting zero or negative values.

    Returns:
        The validated process count or automatic-allocation marker.

    Raises:
        argparse.ArgumentTypeError: If the value is not a valid worker request.
    """
    if value == "auto":
        return "auto"
    try:
        workers = int(value)
        resolve_workers(workers, 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return workers


@contextmanager
def single_threaded_numerics() -> Iterator[None]:
    """Limit loaded pools and child imports, restoring parent settings on exit.

    The environment scope is process-global; call from a single coordinator,
    never from concurrent threads that independently change these settings.
    """
    previous = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    try:
        for name in THREAD_ENVIRONMENT:
            os.environ[name] = "1"
        with threadpool_limits(limits=1):
            yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_single_threaded(function: Callable[[Job], Result], job: Job) -> Result:
    """Execute one job with loaded BLAS/OpenMP pools restricted to one thread.

    Returns:
        The independent job's result.
    """
    with single_threaded_numerics():
        return function(job)


def completed_jobs(
    function: Callable[[Job], Result], jobs: Sequence[Job], *, workers: WorkerSetting = 1
) -> Iterator[tuple[int, Result]]:
    """Yield indexed results as jobs finish, with at most one job per worker.

    Each job must own its checkpoint paths and derive randomness from its
    scientific identity, not its PID or completion order. Only the caller
    writes shared aggregate files. There is no nested process pool.
    Invalid worker settings and job/process errors propagate to the caller;
    completed artifacts remain available for resume.

    Args:
        function: Importable, picklable single-job callable for parallel runs.
        jobs: Independent jobs in canonical input order.
        workers: Positive process count or ``"auto"``.

    Yields:
        Original job index and result, possibly out of order.

    """
    count = resolve_workers(workers, len(jobs))
    if count == 0:
        return
    with single_threaded_numerics():
        if count == 1:
            for index, job in enumerate(jobs):
                yield index, function(job)
            return
        with ExitStack() as cleanup:
            executor = ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context("spawn"))
            cleanup.callback(executor.shutdown, wait=True, cancel_futures=True)
            pending = {}
            remaining = iter(enumerate(jobs))
            for index, job in itertools.islice(remaining, count):
                pending[executor.submit(_run_single_threaded, function, job)] = index
            while pending:
                done, _not_done = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    yield index, future.result()
                    item = next(remaining, None)
                    if item is not None:
                        next_index, job = item
                        pending[executor.submit(_run_single_threaded, function, job)] = next_index
