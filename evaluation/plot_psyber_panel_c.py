from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from psyber_benchmark_plot_utils import add_panel_label, plot_backbone_coverage, plot_scene_coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot panel (C) of the PsyBer benchmark figure.")
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/reports/psyber_panel_c.pdf",
        help="Path to save the generated panel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    fig = plt.figure(figsize=(9.2, 4.2), facecolor="white")
    gs = GridSpec(1, 2, figure=fig, wspace=0.36)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    plot_scene_coverage(ax1, bar_width=0.62)
    plot_backbone_coverage(
        ax2,
        colors=["#009E74", "#D26101", "#0073B1", "#E59F05", "#D2ECFA"],
    )

    add_panel_label(ax1, "(C)", x=-0.24, y=1.14)
    ax1.text(-0.14, 1.03, "(a)", transform=ax1.transAxes, fontsize=11, fontweight="bold")
    ax2.text(-0.14, 1.03, "(b)", transform=ax2.transAxes, fontsize=11, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path.resolve()}")


if __name__ == "__main__":
    main()
