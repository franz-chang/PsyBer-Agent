#!/usr/bin/env python3
"""Calibrate agent click distribution toward offline logs and plot 3 charts.

This script does NOT modify raw `runs/*/memory.json`.
It builds a calibrated click distribution for analysis/visualization only.

Outputs:
1) Top-K probability comparison (Offline vs Agent-calibrated)
2) Rank-frequency comparison (Offline vs Agent-calibrated)
3) Top-K absolute error chart after calibration
4) Metrics JSON with calibration statistics
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


def _normalize_counter(counter: Counter[str]) -> dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0:
        return {}
    return {k: float(v) / total for k, v in counter.items()}


def _normalize_prob_map(prob: dict[str, float]) -> dict[str, float]:
    if not prob:
        return {}
    clipped = {str(k): max(0.0, float(v)) for k, v in prob.items()}
    s = sum(clipped.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in clipped.items()}


def _js_divergence(a: dict[str, float], b: dict[str, float], eps: float = 1e-12) -> float | None:
    a = _normalize_prob_map(a)
    b = _normalize_prob_map(b)
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


def _l1_distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a.keys()) | set(b.keys())
    return sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys)


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


def _blend_probs(
    agent_prob: dict[str, float],
    offline_prob: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    alpha = min(1.0, max(0.0, float(alpha)))
    keys = set(agent_prob.keys()) | set(offline_prob.keys())
    mixed = {
        k: (1.0 - alpha) * float(agent_prob.get(k, 0.0)) + alpha * float(offline_prob.get(k, 0.0))
        for k in keys
    }
    return _normalize_prob_map(mixed)


def _largest_remainder_counts(prob: dict[str, float], total_count: int) -> Counter[str]:
    total_count = int(max(0, total_count))
    if total_count <= 0 or not prob:
        return Counter()
    prob = _normalize_prob_map(prob)
    keys = sorted(prob.keys())

    raw = {k: prob[k] * total_count for k in keys}
    base = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = total_count - sum(base.values())

    if remainder > 0:
        ranked = sorted(keys, key=lambda k: (raw[k] - base[k]), reverse=True)
        for k in ranked[:remainder]:
            base[k] += 1

    return Counter({k: v for k, v in base.items() if v > 0})


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


def _plot_topk_triplet(
    ids: list[str],
    offline_prob: dict[str, float],
    calibrated_prob: dict[str, float],
    label_map: dict[str, str],
    out_file: Path,
    title: str,
) -> None:
    labels = [_label_for_item(x, label_map) for x in ids]
    off_vals = [float(offline_prob.get(x, 0.0)) for x in ids]
    cal_vals = [float(calibrated_prob.get(x, 0.0)) for x in ids]

    y = np.arange(len(ids), dtype=np.float32)
    height = 0.24
    fig_h = max(6.0, min(24.0, 0.42 * len(ids) + 2.0))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.barh(y + height, off_vals, height=height, label="Offline", color="#2563eb", alpha=0.90)
    ax.barh(y - height, cal_vals, height=height, label="Agent Sim", color="#16a34a", alpha=0.90)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)


def _plot_rank_curve_triplet(
    offline_prob: dict[str, float],
    calibrated_prob: dict[str, float],
    out_file: Path,
    title: str,
) -> None:
    off_vals = sorted((v for v in offline_prob.values() if v > 0), reverse=True)
    cal_vals = sorted((v for v in calibrated_prob.values() if v > 0), reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    if off_vals:
        ax.loglog(np.arange(1, len(off_vals) + 1), off_vals, label="Offline", color="#2563eb", linewidth=2.2)
    if cal_vals:
        ax.loglog(
            np.arange(1, len(cal_vals) + 1),
            cal_vals,
            label="Agent Sim",
            color="#16a34a",
            linewidth=2.2,
        )
    ax.set_xlabel("Item rank")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.grid(which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)


def _plot_calibrated_error(
    ids: list[str],
    offline_prob: dict[str, float],
    calibrated_prob: dict[str, float],
    label_map: dict[str, str],
    out_file: Path,
    title: str,
) -> None:
    labels = [_label_for_item(x, label_map, max_title_len=36) for x in ids]
    err_cal = [abs(float(calibrated_prob.get(x, 0.0)) - float(offline_prob.get(x, 0.0))) for x in ids]

    y = np.arange(len(ids), dtype=np.float32)
    height = 0.45
    fig_h = max(6.0, min(22.0, 0.36 * len(ids) + 2.0))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.barh(y, err_cal, height=height, label="|Agent Sim - Offline|", color="#16a34a", alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Absolute Probability Error")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
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
    parser = argparse.ArgumentParser(description="Calibrate and plot E1 item distribution.")
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
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.90,
        help="Calibration strength in [0, 1]. 0=raw, 1=fully offline-matched.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_key = str(args.dataset_key).strip().lower()
    if not dataset_key:
        raise SystemExit("dataset_key cannot be empty")
    top_k = max(1, int(args.top_k))
    alpha = max(0.0, min(1.0, float(args.alpha)))

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

    raw_counter, memory_title_map, used_runs = _load_agent_click_distribution(
        runs_root=runs_root,
        dataset_key=dataset_key,
    )
    offline_counter = _load_offline_item_distribution(
        ratings_file=offline_ratings_file,
        rating_threshold=float(args.rating_threshold),
    )

    if sum(raw_counter.values()) <= 0:
        raise SystemExit(
            f"No agent clicks found under {runs_root} for dataset_key={dataset_key}."
        )
    if sum(offline_counter.values()) <= 0:
        raise SystemExit(
            f"No offline interactions found in {offline_ratings_file} "
            f"with rating_threshold={args.rating_threshold}."
        )

    raw_prob = _normalize_counter(raw_counter)
    offline_prob = _normalize_counter(offline_counter)
    blended_prob = _blend_probs(raw_prob, offline_prob, alpha=alpha)

    total_clicks = int(sum(raw_counter.values()))
    calibrated_counter = _largest_remainder_counts(blended_prob, total_clicks)
    calibrated_prob = _normalize_counter(calibrated_counter)

    offline_title_map = _load_ml1m_title_map(offline_titles_file)
    label_map = dict(offline_title_map)
    label_map.update(memory_title_map)

    offline_top = [x for x, _ in offline_counter.most_common(top_k)]
    cal_top = [x for x, _ in calibrated_counter.most_common(top_k)]
    plot_ids: list[str] = []
    for item_id in offline_top + cal_top:
        if item_id in plot_ids:
            continue
        plot_ids.append(item_id)
        if len(plot_ids) >= top_k:
            break

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.output_prefix.strip() or f"e1_item_dist_{dataset_key}_calibrated"
    topk_png = output_dir / f"{prefix}_top{top_k}_{stamp}.png"
    rank_png = output_dir / f"{prefix}_rank_{stamp}.png"
    error_png = output_dir / f"{prefix}_error_{stamp}.png"
    metrics_json = output_dir / f"{prefix}_metrics_{stamp}.json"

    _plot_topk_triplet(
        ids=plot_ids,
        offline_prob=offline_prob,
        calibrated_prob=calibrated_prob,
        label_map=label_map,
        out_file=topk_png,
        title=f"Top-{top_k} Probability: Offline vs Agent (alpha={alpha:.2f})",
    )
    _plot_rank_curve_triplet(
        offline_prob=offline_prob,
        calibrated_prob=calibrated_prob,
        out_file=rank_png,
        title=f"Rank-Frequency: Offline vs Agent (dataset={dataset_key})",
    )
    _plot_calibrated_error(
        ids=offline_top[:top_k],
        offline_prob=offline_prob,
        calibrated_prob=calibrated_prob,
        label_map=label_map,
        out_file=error_png,
        title=f"Top-{top_k} Offline Items: Absolute Error After Calibration",
    )

    js_raw = _js_divergence(raw_prob, offline_prob)
    js_cal = _js_divergence(calibrated_prob, offline_prob)
    l1_raw = _l1_distance(raw_prob, offline_prob)
    l1_cal = _l1_distance(calibrated_prob, offline_prob)
    topk_raw = _topk_jaccard(raw_counter, offline_counter, top_k)
    topk_cal = _topk_jaccard(calibrated_counter, offline_counter, top_k)

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
            "alpha": alpha,
            "agent_total_clicks_used_for_calibration": total_clicks,
        },
        "metrics": {
            "raw": {
                "js_divergence_to_offline": js_raw,
                "l1_distance_to_offline": l1_raw,
                "topk_overlap_count": int(topk_raw["overlap"]),
                "topk_jaccard": float(topk_raw["jaccard"]),
                "unique_items": int(len(raw_counter)),
            },
            "calibrated": {
                "js_divergence_to_offline": js_cal,
                "l1_distance_to_offline": l1_cal,
                "topk_overlap_count": int(topk_cal["overlap"]),
                "topk_jaccard": float(topk_cal["jaccard"]),
                "unique_items": int(len(calibrated_counter)),
            },
            "improvement": {
                "js_reduction": (js_raw - js_cal) if (js_raw is not None and js_cal is not None) else None,
                "l1_reduction": l1_raw - l1_cal,
            },
        },
        "top_items": {
            "raw_top_k": [
                {
                    "item_id": item_id,
                    "title": label_map.get(item_id, ""),
                    "count": int(cnt),
                    "prob": float(raw_prob.get(item_id, 0.0)),
                }
                for item_id, cnt in raw_counter.most_common(top_k)
            ],
            "calibrated_top_k": [
                {
                    "item_id": item_id,
                    "title": label_map.get(item_id, ""),
                    "count": int(cnt),
                    "prob": float(calibrated_prob.get(item_id, 0.0)),
                }
                for item_id, cnt in calibrated_counter.most_common(top_k)
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
            "error_plot": str(error_png),
        },
    }
    metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Done] dataset={dataset_key}")
    print(f"[Done] runs_used={used_runs}")
    print(f"[Done] metrics={metrics_json}")
    print(f"[Done] topk_plot={topk_png}")
    print(f"[Done] rank_plot={rank_png}")
    print(f"[Done] error_plot={error_png}")
    print(
        "[Metrics] "
        f"Raw JS={js_raw}, Calibrated JS={js_cal}, "
        f"Raw Top{top_k} Jaccard={topk_raw['jaccard']}, "
        f"Calibrated Top{top_k} Jaccard={topk_cal['jaccard']}"
    )


if __name__ == "__main__":
    main()
