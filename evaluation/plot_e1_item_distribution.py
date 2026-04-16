#!/usr/bin/env python3
"""Plot E1 item-distribution comparison between agent clicks and offline logs.

Inputs:
- Agent clicks: runs/*/memory.json -> click_sequence.item_id
- Offline logs (ML-1M): ratings.dat -> item_id counts

Outputs:
- Top-K item probability comparison chart (PNG)
- Rank-frequency comparison chart (PNG)
- Summary metrics and top lists (JSON)
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ML1M_RATINGS = Path("/Users/chongzhang/WebSim_Dataset/MM-ML-1M-main/ratings.dat")
DEFAULT_ML1M_MOVIES = Path("/Users/chongzhang/WebSim_Dataset/MM-ML-1M-main/movies.dat")


def _safe_div(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return a / b


def _normalize_counter(counter: Counter[str]) -> dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0:
        return {}
    return {k: float(v) / total for k, v in counter.items()}


def _js_divergence(a: dict[str, float], b: dict[str, float], eps: float = 1e-12) -> float | None:
    if not a and not b:
        return None
    keys = sorted(set(a.keys()) | set(b.keys()))
    pa = [float(a.get(k, 0.0)) for k in keys]
    pb = [float(b.get(k, 0.0)) for k in keys]

    def _kl(p: list[float], q: list[float]) -> float:
        val = 0.0
        for pi, qi in zip(p, q):
            if pi <= 0:
                continue
            val += pi * math.log((pi + eps) / (qi + eps))
        return val

    m = [(x + y) / 2.0 for x, y in zip(pa, pb)]
    return 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)


def _topk_jaccard(a: Counter[str], b: Counter[str], k: int) -> dict[str, float]:
    a_top = {x for x, _ in a.most_common(k)}
    b_top = {x for x, _ in b.most_common(k)}
    if not a_top and not b_top:
        return {"overlap": 0.0, "jaccard": 1.0}
    overlap = float(len(a_top & b_top))
    union = float(len(a_top | b_top))
    return {
        "overlap": overlap,
        "jaccard": overlap / union if union > 0 else 0.0,
    }


def _load_agent_click_distribution(
    runs_root: Path,
    dataset_key: str,
) -> tuple[Counter[str], dict[str, str], int]:
    counter: Counter[str] = Counter()
    title_map: dict[str, str] = {}
    used_runs = 0

    for mem_file in sorted(runs_root.glob("*/memory.json")):
        try:
            data = json.loads(mem_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(data.get("selected_dataset_key") or "").strip().lower() != dataset_key:
            continue
        used_runs += 1
        for rec in data.get("click_sequence") or []:
            item_id = str(rec.get("item_id") or "").strip()
            if not item_id:
                continue
            counter[item_id] += 1
            title = str(rec.get("title") or "").strip()
            if title and item_id not in title_map:
                title_map[item_id] = title
    return counter, title_map, used_runs


def _load_offline_item_distribution(
    ratings_file: Path,
    *,
    rating_threshold: float,
) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not ratings_file.exists():
        raise FileNotFoundError(f"Offline ratings file not found: {ratings_file}")

    with ratings_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("::")
            if len(parts) < 4:
                continue
            item_id = parts[1].strip()
            if not item_id:
                continue
            try:
                rating = float(parts[2])
            except Exception:
                rating = 0.0
            if rating < rating_threshold:
                continue
            counter[item_id] += 1
    return counter


def _load_ml1m_title_map(movies_file: Path | None) -> dict[str, str]:
    if movies_file is None or (not movies_file.exists()):
        return {}
    out: dict[str, str] = {}
    with movies_file.open("r", encoding="latin-1") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("::")
            if len(parts) < 3:
                continue
            item_id = parts[0].strip()
            title = parts[1].strip()
            if item_id and title and item_id not in out:
                out[item_id] = title
    return out


def _label_for_item(item_id: str, title_map: dict[str, str], max_title_len: int = 42) -> str:
    title = title_map.get(item_id) or ""
    if len(title) > max_title_len:
        title = title[: max_title_len - 3] + "..."
    if title:
        return f"{item_id} | {title}"
    return item_id


def _plot_topk_bar(
    ids: list[str],
    agent_prob: dict[str, float],
    offline_prob: dict[str, float],
    label_map: dict[str, str],
    out_file: Path,
    title: str,
) -> None:
    labels = [_label_for_item(x, label_map) for x in ids]
    agent_vals = [float(agent_prob.get(x, 0.0)) for x in ids]
    offline_vals = [float(offline_prob.get(x, 0.0)) for x in ids]
    y = np.arange(len(ids), dtype=np.float32)
    height = 0.4

    fig_h = max(6.0, min(20.0, 0.36 * len(ids) + 2.0))
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.barh(y + height / 2.0, offline_vals, height=height, label="Offline", color="#2563eb", alpha=0.88)
    ax.barh(y - height / 2.0, agent_vals, height=height, label="Agent", color="#ea580c", alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def _plot_rank_curve(
    agent_prob: dict[str, float],
    offline_prob: dict[str, float],
    out_file: Path,
    title: str,
) -> None:
    a_vals = sorted((v for v in agent_prob.values() if v > 0), reverse=True)
    o_vals = sorted((v for v in offline_prob.values() if v > 0), reverse=True)
    if not a_vals and not o_vals:
        raise RuntimeError("Both agent and offline distributions are empty.")

    fig, ax = plt.subplots(figsize=(10, 6))
    if o_vals:
        ax.loglog(np.arange(1, len(o_vals) + 1), o_vals, label="Offline", color="#2563eb", linewidth=2)
    if a_vals:
        ax.loglog(np.arange(1, len(a_vals) + 1), a_vals, label="Agent", color="#ea580c", linewidth=2)
    ax.set_xlabel("Item rank")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.grid(which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def _default_offline_ratings(dataset_key: str) -> Path | None:
    if dataset_key == "ml1m":
        return DEFAULT_ML1M_RATINGS
    return None


def _default_offline_titles(dataset_key: str) -> Path | None:
    if dataset_key == "ml1m":
        return DEFAULT_ML1M_MOVIES
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot E1 item distribution comparison.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--dataset-key", type=str, default="ml1m")
    parser.add_argument("--offline-ratings-file", type=Path, default=None)
    parser.add_argument("--offline-titles-file", type=Path, default=None)
    parser.add_argument(
        "--rating-threshold",
        type=float,
        default=0.0,
        help="Only keep offline interactions with rating >= threshold. Default 0 means all.",
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/reports"))
    parser.add_argument("--output-prefix", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_key = str(args.dataset_key).strip().lower()
    if not dataset_key:
        raise SystemExit("dataset_key cannot be empty")
    top_k = max(1, int(args.top_k))

    runs_root = args.runs_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    offline_ratings_file = (
        args.offline_ratings_file.expanduser().resolve()
        if args.offline_ratings_file is not None
        else _default_offline_ratings(dataset_key)
    )
    if offline_ratings_file is None:
        raise SystemExit(
            f"No default offline file for dataset_key={dataset_key}. "
            "Please provide --offline-ratings-file."
        )
    offline_ratings_file = offline_ratings_file.expanduser().resolve()

    offline_titles_file = (
        args.offline_titles_file.expanduser().resolve()
        if args.offline_titles_file is not None
        else _default_offline_titles(dataset_key)
    )

    agent_counter, memory_title_map, used_runs = _load_agent_click_distribution(
        runs_root=runs_root,
        dataset_key=dataset_key,
    )
    offline_counter = _load_offline_item_distribution(
        ratings_file=offline_ratings_file,
        rating_threshold=float(args.rating_threshold),
    )

    if sum(agent_counter.values()) <= 0:
        raise SystemExit(
            f"No agent clicks found under {runs_root} for dataset_key={dataset_key}."
        )
    if sum(offline_counter.values()) <= 0:
        raise SystemExit(
            f"No offline interactions found in {offline_ratings_file} "
            f"with rating_threshold={args.rating_threshold}."
        )

    offline_title_map = _load_ml1m_title_map(offline_titles_file)
    label_map = dict(offline_title_map)
    label_map.update(memory_title_map)

    agent_prob = _normalize_counter(agent_counter)
    offline_prob = _normalize_counter(offline_counter)

    real_top = [x for x, _ in offline_counter.most_common(top_k)]
    agent_top = [x for x, _ in agent_counter.most_common(top_k)]
    plot_ids: list[str] = []
    for item_id in real_top + agent_top:
        if item_id in plot_ids:
            continue
        plot_ids.append(item_id)
        if len(plot_ids) >= top_k:
            break

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.output_prefix.strip() or f"e1_item_dist_{dataset_key}"
    topk_png = output_dir / f"{prefix}_top{top_k}_{stamp}.png"
    rank_png = output_dir / f"{prefix}_rank_{stamp}.png"
    metrics_json = output_dir / f"{prefix}_metrics_{stamp}.json"

    _plot_topk_bar(
        ids=plot_ids,
        agent_prob=agent_prob,
        offline_prob=offline_prob,
        label_map=label_map,
        out_file=topk_png,
        title=f"Item Probability Comparison (dataset={dataset_key}, top={top_k})",
    )
    _plot_rank_curve(
        agent_prob=agent_prob,
        offline_prob=offline_prob,
        out_file=rank_png,
        title=f"Rank-Frequency Comparison (dataset={dataset_key})",
    )

    js = _js_divergence(agent_prob, offline_prob)
    topk_overlap = _topk_jaccard(agent_counter, offline_counter, top_k)

    payload: dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "runs_root": str(runs_root),
            "dataset_key": dataset_key,
            "used_run_count": used_runs,
            "offline_ratings_file": str(offline_ratings_file),
            "offline_titles_file": str(offline_titles_file) if offline_titles_file else None,
            "rating_threshold": float(args.rating_threshold),
            "top_k": top_k,
        },
        "metrics": {
            "js_divergence_item_distribution": js,
            "agent_click_total": int(sum(agent_counter.values())),
            "agent_unique_items": int(len(agent_counter)),
            "offline_click_total": int(sum(offline_counter.values())),
            "offline_unique_items": int(len(offline_counter)),
            "topk_overlap_count": int(topk_overlap["overlap"]),
            "topk_jaccard": float(topk_overlap["jaccard"]),
        },
        "top_items": {
            "agent_top_k": [
                {
                    "item_id": item_id,
                    "title": label_map.get(item_id, ""),
                    "count": int(cnt),
                    "prob": float(agent_prob.get(item_id, 0.0)),
                }
                for item_id, cnt in agent_counter.most_common(top_k)
            ],
            "offline_top_k": [
                {
                    "item_id": item_id,
                    "title": label_map.get(item_id, ""),
                    "count": int(cnt),
                    "prob": float(offline_prob.get(item_id, 0.0)),
                }
                for item_id, cnt in offline_counter.most_common(top_k)
            ],
        },
        "artifacts": {
            "topk_plot": str(topk_png),
            "rank_plot": str(rank_png),
        },
    }
    metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Done] dataset={dataset_key}")
    print(f"[Done] runs_used={used_runs}")
    print(f"[Done] metrics={metrics_json}")
    print(f"[Done] topk_plot={topk_png}")
    print(f"[Done] rank_plot={rank_png}")
    print(
        "[Metrics] "
        f"JS={payload['metrics']['js_divergence_item_distribution']}, "
        f"Top{top_k} Jaccard={payload['metrics']['topk_jaccard']}"
    )


if __name__ == "__main__":
    main()
