# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Matched weighted adaptive-online Krotov and Adam development rate screen."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.state_preparation.execution import WorkerSetting, parse_workers
from experiments import state_preparation_pilot as pilot
from experiments import state_preparation_trajectory_diagnostics as development
from experiments import state_preparation_weighted_optimization as weighted

if TYPE_CHECKING:
    from collections.abc import Sequence

VERSION = "yaqs.adaptive_online_screen.v1"
RATES = (0.005, 0.01, 0.02)


def variants(config: development.Config) -> tuple[development.Variant, ...]:
    """Return three Adam rates, three adaptive-online rates and two fixed controls."""
    return (
        development.Variant("all_online", config.updates),
        development.Variant("krotov_adam_0p02", config.online_updates, tail="adam", rate=0.02),
        *(
            development.Variant(f"trajectory_adam_{str(rate).replace('.', 'p')}", 0, tail="adam", rate=rate)
            for rate in RATES
        ),
        *(
            development.Variant(
                f"adaptive_online_{str(rate).replace('.', 'p')}",
                config.online_updates,
                tail="online_rmsprop",
                rate=rate,
            )
            for rate in RATES
        ),
    )


def run(
    root: Path,
    source_root: Path,
    *,
    config: development.Config | None = None,
    workers: WorkerSetting = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    """Run 64 development refinements with paired streams and unchanged source targets."""
    config = development.Config() if config is None else config
    return weighted.run(
        root,
        source_root,
        config=config,
        counts=(config.trajectories,),
        workers=workers,
        resume=resume,
        variants=variants(config),
        samplers=("weighted",),
        provenance={"version": VERSION, "driver": pilot.digest(Path(__file__).read_text(encoding="utf-8"))},
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Expose only the fixed adaptive-online development screen on the command line."""
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
