# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Smoke tests for portable historical figure-generation scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("plot_name", ["plot_sota_comparison", "plot_ablation_studies"])
def test_paper_figures_write_png_to_selected_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plot_name: str
) -> None:
    """Generate valid images without requiring the author's workstation paths."""
    plt = pytest.importorskip("matplotlib.pyplot")
    from scripts import plot_paper_figures

    output = tmp_path / "figures"
    monkeypatch.setattr(plot_paper_figures, "out_dir", output)
    try:
        getattr(plot_paper_figures, plot_name)()
        images = list(output.glob("*.png"))
        assert len(images) == 1
        assert images[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close("all")


def test_trajectory_figure_writes_to_selected_path(tmp_path: Path) -> None:
    """Importing the script has no rendering side effect; an explicit call writes."""
    plt = pytest.importorskip("matplotlib.pyplot")
    from scripts import plot_trajectory_convergence

    output = tmp_path / "figures" / "trajectory.png"
    try:
        plot_trajectory_convergence.plot_trajectory_convergence(output)
        assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close("all")
