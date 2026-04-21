from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from psyber_benchmark_plot_utils import (
    add_panel_label,
    plot_backbone_coverage,
    plot_density_bubble,
    plot_scene_coverage,
    plot_users_items,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the merged panels (B) and (C) of the PsyBer benchmark figure."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/reports/psyber_panels_bc.pdf",
        help="Path to save the generated figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )

    fig = plt.figure(figsize=(10.4, 7.2), facecolor="white")
    gs = GridSpec(
        2,
        2,
        figure=fig,
        wspace=0.20,
        hspace=0.34,
        left=0.07,
        right=0.98,
        top=0.94,
        bottom=0.08,
    )

    ax_b1 = fig.add_subplot(gs[0, 0])
    ax_b2 = fig.add_subplot(gs[0, 1])
    ax_c1 = fig.add_subplot(gs[1, 0])
    ax_c2 = fig.add_subplot(gs[1, 1])

    plot_users_items(
        ax_b1,
        user_color="#C6E5FB",
        item_color="#F79B9D",
        legend_loc="upper right",
    )
    plot_density_bubble(
        ax_b2,
        scene_colors={
            "Movie": "#22BDD2",
            "Shopping": "#1B78B2",
            "Short-video": "#9368AB",
        },
        xlim=(4.5, 260.0),
        ylim=(94.6, 100.45),
    )
    plot_scene_coverage(
        ax_c1,
        bar_width=0.62,
        colors=["#C6E5FB", "#F79B9D", "#F7D097"],
    )
    plot_backbone_coverage(
        ax_c2,
        colors=["#DCDCDC", "#CFE9FC", "#F9B8BB", "#F9DCA9", "#D7EFFD"],
    )

    add_panel_label(ax_b1, "(B)", x=-0.18, y=1.10)
    add_panel_label(ax_c1, "(C)", x=-0.18, y=1.10)
    ax_b1.text(-0.11, 1.01, "(a)", transform=ax_b1.transAxes, fontsize=11, fontweight="bold")
    ax_b2.text(-0.11, 1.01, "(b)", transform=ax_b2.transAxes, fontsize=11, fontweight="bold")
    ax_c1.text(-0.11, 1.01, "(a)", transform=ax_c1.transAxes, fontsize=11, fontweight="bold")
    ax_c2.text(-0.11, 1.01, "(b)", transform=ax_c2.transAxes, fontsize=11, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path.resolve()}")


if __name__ == "__main__":
    main()
