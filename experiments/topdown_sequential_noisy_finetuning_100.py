# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run the frozen sequential-pruning refinement with 100 online and 100 batch updates."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from experiments import topdown_sequential_noisy_finetuning as base

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

VERSION = base.VERSION
PROTOCOL = {
    **base.PROTOCOL,
    "batch_updates": 100,
    "total_updates": 200,
    "returned_iterate": "final iterate after all 200 updates; no exact-test-based selection",
}


@contextmanager
def fixed_budget() -> Generator[None, None, None]:
    """Apply the shorter budget in one worker process, restoring the frozen default.

    Yields:
        Control with the unchanged driver using the 100-plus-100 protocol.
    """
    original = base.PROTOCOL
    base.PROTOCOL = PROTOCOL
    try:
        yield
    finally:
        base.PROTOCOL = original


def prepare(parent: Path, output: Path) -> dict:
    """Prepare the same authenticated 27 inputs with a distinct budget identity.

    Returns:
        The sealed plan for the complete nine-target, three-seed panel.
    """
    with fixed_budget():
        return base.prepare(parent, output)


def run_case(folder: Path, input_folder: Path) -> dict:
    """Run fresh online and batch phases using the unchanged numerical updates.

    Returns:
        The sealed independently evaluated final endpoint and measured costs.
    """
    with fixed_budget():
        return base.run_case(folder, input_folder)


def main() -> None:
    """Use the original prepare/run CLI with the separate 100-plus-100 budget."""
    with fixed_budget():
        base.main()


if __name__ == "__main__":
    main()
