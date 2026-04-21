from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from psyber_benchmark_plot_utils import add_panel_label, plot_radial_benchmark_wheel

PANEL_A_METRICS = [
    ("users", "#Users", "#D6D6D6"),
    ("items", "#Items", "#C6E5FB"),
    ("interactions", "#Inter.", "#F79B9D"),
    ("sparsity", "Sparsity", "#F7D097"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot panel (A) of the PsyBer benchmark figure.")
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/reports/psyber_panel_a.pdf",
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

    fig = plt.figure(figsize=(6.35, 5.95), facecolor="white")
    ax = fig.add_subplot(111, polar=True)
    ax.set_position([0.035, 0.135, 0.93, 0.80])
    plot_radial_benchmark_wheel(
        ax,
        metrics=PANEL_A_METRICS,
        bar_fill_ratio=0.70,
        dataset_label_mode="none",
        center_lines=[],
        use_inner_dataset_ring=True,
        inner_ring_radius_offset=0.64,
        inner_ring_fontsize=10.2,
        angular_offset=-0.48,
        draw_inner_arcs=True,
        inner_arc_radius_offset=0.14,
        legend_ncol=2,
        legend_bbox=(0.5, -0.095),
    )
    add_panel_label(ax, "(A)", x=-0.038, y=1.045)

    fig.text(
        0.5,
        0.02,
        "Bar lengths are normalized within each metric; labels report original values.",
        ha="center",
        fontsize=10,
        color="#666666",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path.resolve()}")


if __name__ == "__main__":
    main()
