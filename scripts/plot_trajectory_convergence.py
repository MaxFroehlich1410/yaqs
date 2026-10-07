# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Generate the historical trajectory convergence figure."""

from pathlib import Path

import matplotlib.pyplot as plt


def plot_trajectory_convergence(output: Path | None = None) -> None:
    """Write the historical convergence figure to a portable local output path."""
    if output is None:
        output = Path(__file__).resolve().parents[1] / "output" / "paper-figures" / "trajectory_convergence.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    trajectories = [1, 10, 25, 50, 100]
    fidelities_state1 = [0.8959, 0.8959, 0.7719, 0.8789, 0.8305]

    plt.figure(figsize=(8, 5))
    plt.plot(
        trajectories,
        fidelities_state1,
        marker="o",
        linestyle="-",
        linewidth=2,
        color="coral",
        label="State 1 Convergence",
        zorder=3,
    )
    plt.plot(
        [1, 10],
        [0.9741, 0.8944],
        marker="s",
        linestyle="--",
        linewidth=2,
        color="teal",
        label="State 2 (Partial)",
        zorder=3,
    )

    plt.axhline(y=0.8305, color="coral", linestyle=":", alpha=0.7, label="Asymptotic True Fidelity (State 1)")

    plt.grid(visible=True, linestyle="--", alpha=0.6, zorder=0)
    plt.title("Optimization Fidelity vs. Number of TJM Trajectories", fontsize=14, pad=15)
    plt.xlabel("Number of Stochastic Trajectories ($N_{traj}$)", fontsize=12)
    plt.ylabel("Calculated Fine-Tuning Fidelity", fontsize=12)
    plt.xscale("log")
    plt.xticks(trajectories, labels=[str(t) for t in trajectories])
    plt.legend(fontsize=11)
    plt.tight_layout()

    plt.savefig(output, dpi=300)


if __name__ == "__main__":
    plot_trajectory_convergence()
