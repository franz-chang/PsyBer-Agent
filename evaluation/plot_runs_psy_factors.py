from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot average C/E/S trajectories across agent runs."
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Directory containing run subfolders with memory.json files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/reports/runs_avg_psy_factors.pdf",
        help="Path to save the generated PDF figure.",
    )
    parser.add_argument(
        "--min-support-ratio",
        type=float,
        default=0.5,
        help=(
            "Keep only steps with at least this fraction of the maximum per-step agent count. "
            "This removes unsupported long tails when only a few runs continue."
        ),
    )
    return parser.parse_args()


def load_stepwise_state_averages(
    runs_dir: Path, min_support_ratio: float
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[int, int], int]:
    accum: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"C": [], "E": [], "S": []})
    valid_runs = 0

    for memory_path in sorted(runs_dir.glob("*/memory.json")):
        try:
            data = json.loads(memory_path.read_text())
        except Exception:
            continue

        history = data.get("psy_state_history", [])
        if not history:
            continue

        valid_runs += 1
        for record in history:
            try:
                step = int(record["step"])
                state = record.get("state", {})
                for key in ("C", "E", "S"):
                    value = state.get(key)
                    if value is not None:
                        accum[step][key].append(float(value))
            except Exception:
                continue

    if not accum:
        raise ValueError(f"No valid psy_state_history found under {runs_dir}")

    step_counts = {step: len(values["C"]) for step, values in accum.items()}
    max_count = max(step_counts.values())
    min_count = max(1, int(np.ceil(max_count * float(min_support_ratio))))
    kept_steps = np.array(
        [step for step in sorted(accum) if step_counts.get(step, 0) >= min_count],
        dtype=np.int32,
    )

    means = {
        key: np.array(
            [float(np.mean(accum[step][key])) for step in kept_steps],
            dtype=np.float64,
        )
        for key in ("C", "E", "S")
    }
    return kept_steps, means, step_counts, valid_runs


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    output_path = Path(args.output)

    steps, means, step_counts, valid_runs = load_stepwise_state_averages(
        runs_dir, min_support_ratio=float(args.min_support_ratio)
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.2), facecolor="white")
    ax.set_facecolor("#FCFCFC")
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.24)

    palette = {
        "C": "#1F4E79",
        "E": "#D62728",
        "S": "#2A9D8F",
    }
    labels = {
        "C": "Cognition (C)",
        "E": "Emotion (E)",
        "S": "Self-congruence (S)",
    }

    for key in ("C", "E", "S"):
        ax.plot(
            steps,
            means[key],
            color=palette[key],
            linewidth=2.6,
            label=labels[key],
            zorder=3,
        )

    ax.set_title("Average psychological factor trajectories across agents")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Value")
    ax.set_xlim(int(steps.min()), int(steps.max()))
    ax.set_ylim(0.0, 1.0)
    ax.margins(x=0.01)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    ax.legend(frameon=False, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    kept_count_min = min(step_counts[int(step)] for step in steps)
    kept_count_max = max(step_counts[int(step)] for step in steps)
    print(f"Loaded {valid_runs} runs from {runs_dir.resolve()}")
    print(
        f"Kept {len(steps)} rounds with per-round support in "
        f"[{kept_count_min}, {kept_count_max}] agents"
    )
    print(f"Saved figure to {output_path.resolve()}")


if __name__ == "__main__":
    main()
