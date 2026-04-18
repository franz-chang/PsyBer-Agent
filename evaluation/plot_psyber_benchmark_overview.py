from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from psyber_benchmark_plot_utils import (
    add_panel_label,
    plot_backbone_coverage,
    plot_density_bubble,
    plot_radial_benchmark_wheel,
    plot_scene_coverage,
    plot_users_items,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a multi-panel overview figure for the PsyBer benchmark."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/reports/psyber_benchmark_overview.png",
        help="Path to save the generated figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    fig = plt.figure(figsize=(14.3, 8.6), facecolor="white")
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.75, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.35,
        hspace=0.62,
    )

    ax_a = fig.add_subplot(gs[:, 0], polar=True)
    ax_b1 = fig.add_subplot(gs[0, 1])
    ax_b2 = fig.add_subplot(gs[0, 2])
    ax_c1 = fig.add_subplot(gs[1, 1])
    ax_c2 = fig.add_subplot(gs[1, 2])

    plot_radial_benchmark_wheel(ax_a)
    plot_users_items(ax_b1)
    plot_density_bubble(ax_b2)
    plot_scene_coverage(ax_c1)
    plot_backbone_coverage(ax_c2)

    add_panel_label(ax_a, "(A)", x=-0.06, y=1.08)
    add_panel_label(ax_b1, "(B)", x=-0.28, y=1.16)
    add_panel_label(ax_c1, "(C)", x=-0.28, y=1.16)
    ax_b1.text(-0.14, 1.03, "(a)", transform=ax_b1.transAxes, fontsize=11, fontweight="bold")
    ax_b2.text(-0.14, 1.03, "(b)", transform=ax_b2.transAxes, fontsize=11, fontweight="bold")
    ax_c1.text(-0.14, 1.03, "(a)", transform=ax_c1.transAxes, fontsize=11, fontweight="bold")
    ax_c2.text(-0.14, 1.03, "(b)", transform=ax_c2.transAxes, fontsize=11, fontweight="bold")

    fig.text(
        0.5,
        0.02,
        "Panel (A) normalizes bar lengths within each metric; all labels report original values.",
        ha="center",
        fontsize=9,
        color="#666666",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path.resolve()}")


if __name__ == "__main__":
    main()
