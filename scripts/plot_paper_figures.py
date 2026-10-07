# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Generate the historical paper comparison and ablation figures."""

from pathlib import Path

import matplotlib.pyplot as plt

# Set style
plt.style.use("ggplot")
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14

out_dir = Path(__file__).resolve().parents[1] / "output" / "paper-figures"


def plot_sota_comparison() -> None:
    """Plot the historical parameter-count and fidelity comparison."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _fig, ax = plt.subplots(figsize=(12, 7))

    # Data for State 4 (The "Hero State" where ADAPT gets stuck in local minima)
    methods = ["Bottom-Up ADAPT", "Standard VQA", "Top-Down Pruning (Ours)"]
    params = [150, 276, 150]
    fidelities = [0.6468, 0.9128, 0.9600]
    colors = ["#FFA07A", "#87CEFA", "#32CD32"]
    markers = ["X", "s", "*"]
    sizes = [300, 300, 600]

    # Plot points
    for i in range(len(methods)):
        ax.scatter(
            params[i],
            fidelities[i],
            color=colors[i],
            marker=markers[i],
            s=sizes[i],
            edgecolor="black",
            linewidth=1.5,
            label=methods[i],
            zorder=4,
        )

        # Add labels near points, carefully avoiding overlap!
        xytext = (0, -30) if i in {0, 1} else (0, 25)

        ax.annotate(
            f"{fidelities[i] * 100:.1f}%",
            xy=(params[i], fidelities[i]),
            xytext=xytext,
            textcoords="offset points",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=14,
        )

    # Plot Pareto Front (Ideal boundary)
    ax.plot([150, 276], [0.9600, 0.9128], color="gray", linestyle="--", zorder=1, alpha=0.5)

    # Add clean text boxes highlighting the victories instead of overlapping arrows
    props = {"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.9, "edgecolor": "gray"}

    ax.text(
        165,
        0.82,
        "★ SOTA Victory!\\n"
        "Our method beats ADAPT\\n"
        "by +31.3% fidelity (same gates)\\n\\n"
        "And beats VQA by +4.7% fidelity\\n"
        "using 45% FEWER gates!",
        fontsize=13,
        va="center",
        ha="left",
        bbox=props,
        zorder=3,
    )

    # Formatting
    ax.set_xlabel("Circuit Complexity (Number of Gates)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Ground State Fidelity (IBM Heron Noise)", fontsize=15, fontweight="bold")
    ax.set_title("Efficiency Trade-off on Highly Entangled Topology (State 4)", fontsize=18, pad=20, fontweight="bold")

    ax.grid(visible=True, linestyle="--", alpha=0.4, zorder=0)
    ax.legend(loc="lower center", fontsize=13, framealpha=1, ncol=3, bbox_to_anchor=(0.5, -0.2))
    ax.set_ylim(0.55, 1.05)
    ax.set_xlim(120, 310)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/sota_comparison.png", dpi=300, bbox_inches="tight")


def plot_ablation_studies() -> None:
    """Plot the historical initialization and pruning ablations."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Ablation A: CRN vs No-CRN
    crn_methods = ["Standard TJM\\n(No CRN)", "CRN Stabilized"]
    crn_fidelities = [0.6083, 0.9163]

    bars1 = ax1.bar(crn_methods, crn_fidelities, color=["#FFB6C1", "#20B2AA"], edgecolor="black", width=0.6)
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(
            f"{height * 100:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax1.set_ylabel("Fidelity under IBM Heron Noise")
    ax1.set_title("Ablation A: Noise Mitigation")
    ax1.set_ylim(0, 1.1)

    # Ablation B: Layerwise vs All-at-Once
    pre_methods = ["All-at-Once\\n(1000 iters)", "Layerwise Expansion\\n(1000 iters total)"]
    pre_fidelities = [0.1814, 0.9921]

    bars2 = ax2.bar(pre_methods, pre_fidelities, color=["#DDA0DD", "#BA55D3"], edgecolor="black", width=0.6)
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(
            f"{height * 100:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax2.set_ylabel("Pre-Training Fidelity (Noiseless)")
    ax2.set_title("Ablation B: Overparameterized Pre-Training")
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/ablation_studies.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    plot_sota_comparison()
    plot_ablation_studies()
