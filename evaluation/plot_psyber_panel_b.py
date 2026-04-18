from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from psyber_benchmark_plot_utils import add_panel_label, plot_density_bubble, plot_users_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot panel (B) of the PsyBer benchmark figure.")
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/reports/psyber_panel_b.pdf",
        help="Path to save the generated panel.",
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

    fig = plt.figure(figsize=(10.4, 4.2), facecolor="white")
    gs = GridSpec(1, 2, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    plot_users_items(
        ax1,
        user_color="#C6E5FB",
        item_color="#F79B9D",
        legend_loc="upper right",
    )
    plot_density_bubble(
        ax2,
        scene_colors={
            "Movie": "#22BDD2",
            "Shopping": "#1B78B2",
            "Short-video": "#9368AB",
        },
        xlim=(4.5, 260.0),
        ylim=(94.6, 100.45),
    )

    add_panel_label(ax1, "(B)", x=-0.26, y=1.14)
    ax1.text(-0.14, 1.03, "(a)", transform=ax1.transAxes, fontsize=11, fontweight="bold")
    ax2.text(-0.14, 1.03, "(b)", transform=ax2.transAxes, fontsize=11, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path.resolve()}")


if __name__ == "__main__":
    main()
