from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from psyber_benchmark_plot_utils import add_panel_label, plot_radial_benchmark_wheel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot panel (A) of the PsyBer benchmark figure.")
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/reports/psyber_panel_a.png",
        help="Path to save the generated panel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    fig = plt.figure(figsize=(7.8, 7.2), facecolor="white")
    ax = fig.add_subplot(111, polar=True)
    plot_radial_benchmark_wheel(ax)
    add_panel_label(ax, "(A)", x=-0.06, y=1.08)

    fig.text(
        0.5,
        0.025,
        "Bar lengths are normalized within each metric; labels report original values.",
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
