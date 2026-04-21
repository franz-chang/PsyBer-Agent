from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    }
)


DATASETS = [
    {
        "name": "MM-ML-1M",
        "short": "MM-ML-1M",
        "scene": "Movie",
        "users": 6_040,
        "items": 3_952,
        "interactions": 1_000_209,
        "sparsity": 95.81,
        "avg_len": 165.60,
    },
    {
        "name": "Amazon Magazine",
        "short": "A-Mag",
        "scene": "Shopping",
        "users": 380,
        "items": 865,
        "interactions": 2_679,
        "sparsity": 99.18,
        "avg_len": 7.05,
    },
    {
        "name": "Amazon Beauty",
        "short": "A-Beauty",
        "scene": "Shopping",
        "users": 22_363,
        "items": 12_101,
        "interactions": 198_502,
        "sparsity": 99.93,
        "avg_len": 8.88,
    },
    {
        "name": "Amazon Luxury",
        "short": "A-Lux",
        "scene": "Shopping",
        "users": 1_054,
        "items": 791,
        "interactions": 12_397,
        "sparsity": 98.51,
        "avg_len": 11.76,
    },
    {
        "name": "Amazon Musical",
        "short": "A-Music",
        "scene": "Shopping",
        "users": 25_402,
        "items": 10_611,
        "interactions": 210_646,
        "sparsity": 99.92,
        "avg_len": 8.29,
    },
    {
        "name": "MicroLens-100K",
        "short": "MicroLens",
        "scene": "Short-video",
        "users": 100_000,
        "items": 19_738,
        "interactions": 719_405,
        "sparsity": 99.96,
        "avg_len": 7.19,
    },
    {
        "name": "NineRec-Bili500K",
        "short": "NineRec",
        "scene": "Short-video",
        "users": 500_000,
        "items": 138_033,
        "interactions": 7_845_805,
        "sparsity": 99.99,
        "avg_len": 15.69,
    },
]

RADIAL_METRICS = [
    ("users", "#Users", "#8EA8C3"),
    ("items", "#Items", "#B7CADE"),
    ("interactions", "#Inter.", "#E79A7C"),
    ("avg_len", "Avg. Len.", "#7E2F2F"),
]

SCENE_COLORS = {
    "Movie": "#D65A4A",
    "Shopping": "#F2A65A",
    "Short-video": "#5B7EA4",
}

BACKBONE_FAMILIES = [
    ("Heuristic", 1, "#D6E4F0"),
    ("CF", 1, "#B9D0E5"),
    ("GNN", 1, "#97B9D6"),
    ("Generative", 1, "#729FC6"),
    ("Sequential", 3, "#456C8E"),
]


def normalize_columns(values: np.ndarray, floor: float = 0.22) -> np.ndarray:
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


def format_metric_value(metric_key: str, value: float) -> str:
    if metric_key == "sparsity":
        return f"{value:.2f}%"
    if metric_key == "avg_len":
        return f"{value:.1f}"
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


def tangent_rotation(theta: float) -> tuple[float, str]:
    rotation = 180 - np.degrees(theta)
    while rotation > 180:
        rotation -= 360
    while rotation <= -180:
        rotation += 360
    if rotation > 90:
        rotation -= 180
        return rotation, "right"
    if rotation < -90:
        rotation += 180
        return rotation, "right"
    return rotation, "left"


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.11) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
    )


def style_cartesian_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.9)
    ax.set_axisbelow(True)


def plot_radial_benchmark_wheel(
    ax: plt.Axes,
    *,
    metrics: list[tuple[str, str, str]] | None = None,
    bar_fill_ratio: float = 0.92,
    dataset_label_mode: str = "scene",
    center_lines: list[str] | None = None,
    use_inner_dataset_ring: bool = False,
    inner_ring_radius_offset: float = 0.46,
    inner_ring_fontsize: float = 8.2,
    angular_offset: float = 0.0,
    draw_inner_arcs: bool = False,
    inner_arc_radius_offset: float = 0.10,
    legend_ncol: int = 4,
    legend_bbox: tuple[float, float] = (0.5, -0.10),
) -> None:
    metrics = metrics or RADIAL_METRICS
    raw_values = np.array(
        [[row[metric_key] for metric_key, _, _ in metrics] for row in DATASETS],
        dtype=float,
    )
    normalized_values = normalize_columns(raw_values)

    num_groups = len(DATASETS)
    bars_per_group = len(metrics)
    group_gap = np.deg2rad(7.5)
    total_bar_slots = num_groups * bars_per_group
    bar_width = (2 * np.pi - group_gap * num_groups) / total_bar_slots
    full_bar_width = bar_width * bar_fill_ratio

    inner_radius = 2.15
    radial_span = 2.65
    max_radius = inner_radius + radial_span + 0.95

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, max_radius)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)

    tick_levels = np.linspace(0, 1, 5)
    for frac, label in zip(tick_levels, ["0%", "25%", "50%", "75%", "100%"]):
        radius = inner_radius + frac * radial_span
        ax.plot(
            np.linspace(0, 2 * np.pi, 400),
            np.full(400, radius),
            color="#ECECEC",
            lw=0.75,
            zorder=0,
        )
        ax.text(0.0, radius, label, fontsize=10, color="#6E6E6E", ha="center", va="bottom")

    for group_idx, row in enumerate(DATASETS):
        start_angle = angular_offset + group_idx * (bars_per_group * bar_width + group_gap)
        end_angle = start_angle + bars_per_group * bar_width
        group_mid = start_angle + (bars_per_group - 1) * bar_width / 2

        ax.plot(
            np.linspace(start_angle - 0.02, end_angle + 0.02, 100),
            np.full(100, inner_radius - 0.05),
            color="#D8D8D8",
            lw=1.0,
            zorder=2,
        )
        for theta in (start_angle - group_gap / 2, end_angle + group_gap / 2):
            ax.plot(
                [theta, theta],
                [inner_radius - 0.22, inner_radius + radial_span + 0.28],
                color="#D1D1D1",
                lw=1.0,
                zorder=1,
            )

        if dataset_label_mode != "none":
            rotation, align = upright_rotation(group_mid)
            label_color = SCENE_COLORS[row["scene"]] if dataset_label_mode == "scene" else "#616161"
            ax.text(
                group_mid,
                inner_radius - 0.28,
                row["short"],
                fontsize=8.4,
                color=label_color,
                rotation=rotation,
                rotation_mode="anchor",
                ha=align,
                va="center",
                fontweight="bold",
            )

        for bar_idx, (metric_key, _, color) in enumerate(metrics):
            theta = start_angle + bar_idx * bar_width
            bar_length = normalized_values[group_idx, bar_idx] * radial_span
            value = row[metric_key]
            ax.bar(
                theta,
                bar_length,
                width=full_bar_width,
                bottom=inner_radius,
                color=color,
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
                alpha=0.97,
            )

            label_radius = inner_radius + bar_length + 0.14
            rotation, align = tangent_rotation(theta)
            ax.text(
                theta,
                label_radius,
                format_metric_value(metric_key, value),
                fontsize=10.2,
                rotation=rotation,
                rotation_mode="anchor",
                ha=align,
                va="center",
                color="#252525",
                zorder=4,
            )

    donut = Circle((0, 0), inner_radius - 0.12, transform=ax.transData._b, color="white", zorder=10)
    ax.add_artist(donut)
    ax.add_artist(
        Circle(
            (0, 0),
            inner_radius - 0.12,
            transform=ax.transData._b,
            fill=False,
            edgecolor="#B9B9B9",
            linewidth=1.15,
            zorder=11,
        )
    )

    if use_inner_dataset_ring:
        ring_radius = inner_radius - inner_ring_radius_offset
        for group_idx, row in enumerate(DATASETS):
            start_angle = angular_offset + group_idx * (bars_per_group * bar_width + group_gap)
            end_angle = start_angle + bars_per_group * bar_width
            theta = start_angle + (bars_per_group - 1) * bar_width / 2
            rotation = 90 - np.degrees(theta)
            if rotation > 90:
                rotation -= 180
            elif rotation < -90:
                rotation += 180

            ax.text(
                theta,
                ring_radius,
                row["short"],
                fontsize=inner_ring_fontsize,
                color="#3E3E3E",
                rotation=rotation,
                rotation_mode="anchor",
                ha="center",
                va="center",
                fontweight="bold",
                zorder=20,
            )

            if draw_inner_arcs:
                arc = np.linspace(start_angle + 0.05, end_angle - 0.05, 80)
                ax.plot(
                    arc,
                    np.full_like(arc, inner_radius - inner_arc_radius_offset),
                    color="#222222",
                    lw=0.8,
                    zorder=19,
                )

    if center_lines is None:
        center_lines = [
            "PsyBer-Test",
            "7 datasets",
            "3 scenes · 7 backbones",
            "4 normalized metrics",
        ]
    center_y = [0.56, 0.48, 0.42, 0.36]
    center_sizes = [17, 10, 10, 10]
    center_weights = ["bold", "normal", "normal", "normal"]
    center_colors = ["#1E1E1E", "#5C5C5C", "#5C5C5C", "#5C5C5C"]
    for text, y, size, weight, color in zip(center_lines, center_y, center_sizes, center_weights, center_colors):
        ax.text(
            0.5,
            y,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=size,
            fontweight=weight,
            color=color,
            zorder=20,
        )

    legend_handles = [
        Line2D([0], [0], color=color, lw=8, solid_capstyle="butt", label=label)
        for _, label, color in metrics
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=legend_bbox,
        ncol=legend_ncol,
        frameon=False,
        fontsize=9,
        columnspacing=1.2,
        handlelength=1.5,
    )


def plot_users_items(
    ax: plt.Axes,
    *,
    user_color: str | None = None,
    item_color: str | None = None,
    legend_loc: str = "lower right",
) -> None:
    style_cartesian_axis(ax)

    dataset_names = [row["short"] for row in DATASETS]
    y = np.arange(len(DATASETS))
    users = np.array([row["users"] for row in DATASETS])
    items = np.array([row["items"] for row in DATASETS])
    user_color = user_color or RADIAL_METRICS[0][2]
    item_color = item_color or RADIAL_METRICS[1][2]

    offset = 0.18
    ax.barh(y + offset, users, height=0.32, color=user_color, label="#Users")
    ax.barh(y - offset, items, height=0.32, color=item_color, label="#Items")
    ax.set_yticks(y)
    ax.set_yticklabels(dataset_names, fontsize=11.5)
    ax.set_xscale("log")
    ax.set_xlabel("Count (log scale)", fontsize=14)
    ax.set_title("Users and Items", fontsize=15)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.tick_params(axis="y", labelsize=11.5)
    ax.legend(frameon=False, fontsize=10, loc=legend_loc)
    ax.invert_yaxis()


def plot_density_bubble(
    ax: plt.Axes,
    *,
    scene_colors: dict[str, str] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    ax.grid(color="#E7E7E7", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    scene_colors = scene_colors or SCENE_COLORS

    interactions = np.array([row["interactions"] for row in DATASETS], dtype=float)
    avg_lens = np.array([row["avg_len"] for row in DATASETS], dtype=float)
    sparsities = np.array([row["sparsity"] for row in DATASETS], dtype=float)
    sizes = 45 + 900 * np.sqrt(interactions / interactions.max())

    label_offsets = {
        "MM-ML-1M": (6, 0),
        "A-Mag": (6, 6),
        "A-Beauty": (-22, 8),
        "A-Lux": (6, 6),
        "A-Music": (-18, 8),
        "MicroLens": (-12, 8),
        "NineRec": (4, 8),
    }

    for row, size, x, y in zip(DATASETS, sizes, avg_lens, sparsities):
        ax.scatter(
            x,
            y,
            s=size,
            color=scene_colors[row["scene"]],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.8,
        )
        ax.annotate(
            row["short"],
            (x, y),
            textcoords="offset points",
            xytext=label_offsets.get(row["short"], (6, 5)),
            fontsize=10.0,
            color="#333333",
        )

    scene_handles = [
        Line2D([0], [0], marker="o", color="w", label=scene, markerfacecolor=color, markersize=8)
        for scene, color in scene_colors.items()
    ]
    ax.legend(handles=scene_handles, frameon=False, fontsize=10, loc="lower left")

    ax.set_xscale("log")
    ax.set_xlabel("Avg. sequence length (log scale)", fontsize=14)
    ax.set_ylabel("Sparsity (%)", fontsize=14)
    ax.set_title("Density and Sequence Length", fontsize=15)
    ax.set_xlim(*(xlim or (5.5, 220.0)))
    ax.set_ylim(*(ylim or (94.8, 100.35)))
    ax.tick_params(axis="x", labelsize=11.5)
    ax.tick_params(axis="y", labelsize=11.5)

    size_levels = [1e4, 5e5, 5e6]
    size_handles = [
        plt.scatter([], [], s=45 + 900 * math.sqrt(v / interactions.max()), color="#BBBBBB", alpha=0.55)
        for v in size_levels
    ]
    size_labels = ["10K inter.", "500K inter.", "5M inter."]
    size_legend = ax.legend(
        size_handles,
        size_labels,
        frameon=False,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.90, 1.02),
        title="Bubble size",
        title_fontsize=10,
    )
    ax.add_artist(size_legend)
    ax.legend(handles=scene_handles, frameon=False, fontsize=10, loc="lower left")


def plot_scene_coverage(
    ax: plt.Axes,
    *,
    bar_width: float = 0.8,
    colors: list[str] | None = None,
) -> None:
    style_cartesian_axis(ax)

    ordered_scenes = ["Movie", "Shopping", "Short-video"]
    counts = [sum(1 for row in DATASETS if row["scene"] == scene) for scene in ordered_scenes]
    total = len(DATASETS)
    bar_colors = colors or [SCENE_COLORS[scene] for scene in ordered_scenes]

    bars = ax.bar(
        ordered_scenes,
        counts,
        color=bar_colors,
        width=bar_width,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_title("Scenario Coverage", fontsize=15)
    ax.set_ylabel("#Datasets", fontsize=14)
    ax.set_ylim(0, max(counts) + 1.2)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.tick_params(axis="y", labelsize=11.5)

    for bar, count in zip(bars, counts):
        pct = 100.0 * count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.10,
            f"{count} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )


def plot_backbone_coverage(
    ax: plt.Axes,
    *,
    colors: list[str] | None = None,
) -> None:
    style_cartesian_axis(ax)

    labels = [row[0] for row in BACKBONE_FAMILIES]
    values = [row[1] for row in BACKBONE_FAMILIES]
    colors = colors or [row[2] for row in BACKBONE_FAMILIES]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Backbone Families", fontsize=15)
    ax.set_ylabel("#Models", fontsize=14)
    ax.set_ylim(0, max(values) + 1.0)
    ax.tick_params(axis="x", labelrotation=20, labelsize=11.5)
    ax.tick_params(axis="y", labelsize=11.5)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.08,
            str(value),
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )
