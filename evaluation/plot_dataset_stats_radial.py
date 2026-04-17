from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    }
)


DATASETS = [
    {
        "name": "MM-ML-1M",
        "users": 6040,
        "items": 3952,
        "interactions": 1_000_209,
        "sparsity": 95.81,
        "avg_len": 165.60,
    },
    {
        "name": "Amazon Musical",
        "users": 25_402,
        "items": 10_611,
        "interactions": 210_646,
        "sparsity": 99.92,
        "avg_len": 8.29,
    },
    {
        "name": "Amazon Beauty",
        "users": 22_363,
        "items": 12_101,
        "interactions": 198_502,
        "sparsity": 99.93,
        "avg_len": 8.88,
    },
]

METRICS = [
    ("users", "#Users", "#D1495B"),
    ("items", "#Items", "#F08A5D"),
    ("interactions", "#Inter.", "#7F1D1D"),
    ("sparsity", "Sparsity", "#F6BD60"),
    ("avg_len", "Avg. Len.", "#B23A48"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a NeurIPS-style radial dataset statistics figure."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/reports/dataset_stats_radial.png",
        help="Path to save the generated figure.",
    )
    return parser.parse_args()


def normalize_columns(values: np.ndarray, floor: float = 0.25) -> np.ndarray:
    """Normalize each metric column independently for radial bar lengths."""
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    spans = maxs - mins

    normalized = np.ones_like(values, dtype=float)
    for col in range(values.shape[1]):
        if spans[col] <= 1e-12:
            normalized[:, col] = 1.0
        else:
            normalized[:, col] = floor + (values[:, col] - mins[col]) / spans[col] * (1.0 - floor)
    return normalized


def format_value(metric_key: str, value: float) -> str:
    if metric_key == "sparsity":
        return f"{value:.2f}%"
    if metric_key == "avg_len":
        return f"{value:.2f}"
    if metric_key == "interactions":
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        if value >= 1_000:
            return f"{value / 1_000:.1f}K"
    return f"{int(value):,}"


def upright_rotation(theta: float) -> tuple[float, str]:
    rotation = np.degrees(theta) - 90
    if rotation < -90:
        rotation += 180
        return rotation, "right"
    if rotation > 90:
        rotation -= 180
        return rotation, "right"
    return rotation, "left"


def draw_radial_dataset_stats(output_path: Path) -> None:
    dataset_names = [row["name"] for row in DATASETS]
    display_names = {
        "MM-ML-1M": "MM-ML-1M",
        "Amazon Musical": "Amazon\nMusical",
        "Amazon Beauty": "Amazon\nBeauty",
    }
    raw_values = np.array(
        [[row[metric_key] for metric_key, _, _ in METRICS] for row in DATASETS], dtype=float
    )
    normalized_values = normalize_columns(raw_values)

    num_groups = len(DATASETS)
    bars_per_group = len(METRICS)

    group_gap = np.deg2rad(10)
    total_bar_slots = num_groups * bars_per_group
    bar_width = (2 * np.pi - group_gap * num_groups) / total_bar_slots
    bar_pad = bar_width * 0.06
    full_bar_width = bar_width - bar_pad

    inner_radius = 2.45
    radial_span = 2.85
    max_radius = inner_radius + radial_span + 1.05

    fig = plt.figure(figsize=(8.4, 8.8), facecolor="white")
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, max_radius)
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)

    # Light reference rings and labels, matching the style of the example figure.
    tick_levels = np.linspace(0, 1, 5)
    tick_labels = ["0%", "25%", "50%", "75%", "100%"]
    for frac, label in zip(tick_levels, tick_labels):
        radius = inner_radius + frac * radial_span
        ax.plot(np.linspace(0, 2 * np.pi, 400), np.full(400, radius), color="#ECECEC", lw=0.75, zorder=0)
        ax.text(
            0.0,
            radius,
            label,
            fontsize=8,
            color="#777777",
            ha="center",
            va="bottom",
        )

    # Thin separators between dataset groups.
    for group_idx in range(num_groups):
        start_angle = group_idx * (bars_per_group * bar_width + group_gap)
        end_angle = start_angle + bars_per_group * bar_width

        arc = np.linspace(start_angle - 0.03, end_angle + 0.03, 120)
        ax.plot(
            arc,
            np.full_like(arc, inner_radius - 0.04),
            color="#D7D7D7",
            lw=1.0,
            zorder=2,
        )

        for theta in (start_angle - group_gap / 2, end_angle + group_gap / 2):
            ax.plot(
                [theta, theta],
                [inner_radius - 0.22, inner_radius + radial_span + 0.32],
                color="#CFCFCF",
                lw=1.1,
                zorder=1,
            )

    # Bars + outer numeric annotations.
    for group_idx, dataset_name in enumerate(dataset_names):
        start_angle = group_idx * (bars_per_group * bar_width + group_gap)
        group_mid = start_angle + (bars_per_group - 1) * bar_width / 2

        for bar_idx, (metric_key, metric_label, color) in enumerate(METRICS):
            theta = start_angle + bar_idx * bar_width
            bar_length = normalized_values[group_idx, bar_idx] * radial_span
            value = raw_values[group_idx, bar_idx]

            ax.bar(
                theta,
                bar_length,
                width=full_bar_width,
                bottom=inner_radius,
                color=color,
                edgecolor="white",
                linewidth=1.1,
                zorder=3,
                alpha=0.97,
            )

            label_radius = inner_radius + bar_length + 0.18
            rotation, align = upright_rotation(theta)
            ax.text(
                theta,
                label_radius,
                format_value(metric_key, value),
                fontsize=9.5,
                rotation=rotation,
                rotation_mode="anchor",
                ha=align,
                va="center",
                color="#222222",
                zorder=4,
            )

    # Central donut to mimic the layout of the reference figure.
    donut = Circle((0, 0), inner_radius - 0.12, transform=ax.transData._b, color="white", zorder=10)
    ax.add_artist(donut)
    ax.add_artist(
        Circle(
            (0, 0),
            inner_radius - 0.12,
            transform=ax.transData._b,
            fill=False,
            edgecolor="#B8B8B8",
            linewidth=1.2,
            zorder=11,
        )
    )
    center_ax = fig.add_axes(ax.get_position(), frameon=False)
    center_ax.set_axis_off()
    center_ax.text(0.50, 0.69, display_names["MM-ML-1M"], ha="center", va="center", fontsize=9.3, color="#4B4B4B")
    center_ax.text(0.29, 0.47, display_names["Amazon Musical"], ha="center", va="center", fontsize=8.8, color="#4B4B4B", linespacing=0.88)
    center_ax.text(0.71, 0.47, display_names["Amazon Beauty"], ha="center", va="center", fontsize=8.8, color="#4B4B4B", linespacing=0.88)
    center_ax.text(0.5, 0.54, "Dataset\nStatistics", ha="center", va="center", fontsize=18, fontweight="bold")
    center_ax.text(0.5, 0.40, "3 datasets · 5 metrics", ha="center", va="center", fontsize=10, color="#6E6E6E")

    legend_handles = [
        Line2D([0], [0], color=color, lw=10, solid_capstyle="butt", label=metric_label)
        for _, metric_label, color in METRICS
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.095),
        ncol=5,
        frameon=False,
        fontsize=10,
        columnspacing=1.35,
        handlelength=1.5,
    )

    fig.text(0.08, 0.93, "(A)", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.022,
        "Bar lengths are normalized within each metric; labels show original values.",
        ha="center",
        fontsize=9,
        color="#666666",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    draw_radial_dataset_stats(output_path)
    print(f"Saved figure to {output_path.resolve()}")


if __name__ == "__main__":
    main()
