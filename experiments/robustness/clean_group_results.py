#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from metrics import (
    ab_metrics,
    counter_to_prob,
    js_divergence,
    load_ab_cases,
    load_jsonl,
    load_memory_file,
    mean_ci95,
    percentile,
    trajectory_mse,
)


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _norm_text(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("_", " ").replace("-", " ").split())


def _resolve_dataset_paths(dataset_root: Path, dataset_name: str) -> tuple[Path, Path | None]:
    key = _norm_text(dataset_name)
    if key in {"ml1m", "movie lens 1m", "movie lens", "movielens 1m", "movielens"} or "movielens" in key:
        base = dataset_root / "MM-ML-1M-main"
        return base / "ratings.dat", base / "Embeddings" / "expert_intent_preview.json"
    if "magazine" in key:
        base = dataset_root / "Amazon_MM_2018" / "Magazine_Subscriptions"
        return base / "ratings.dat", None
    if "beauty" in key:
        base = dataset_root / "Amazon_MM_2018" / "All_Beauty"
        return base / "ratings.dat", None
    if "musical instruments" in key or key in {"amazon mi", "amazon_mi", "mi"}:
        base = dataset_root / "amazon_v2" / "Musical_Instruments"
        return base / "ratings.dat", None
    # Fallback: assume ml1m.
    base = dataset_root / "MM-ML-1M-main"
    return base / "ratings.dat", base / "Embeddings" / "expert_intent_preview.json"


def _load_ratings_ground_truth(
    ratings_path: Path,
    *,
    positive_threshold: float,
) -> dict[str, Any]:
    if not ratings_path.exists():
        raise FileNotFoundError(f"ratings.dat not found: {ratings_path}")

    item_counter: Counter[str] = Counter()
    total = 0
    positive = 0
    for raw in ratings_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("::")
        if len(parts) < 3:
            continue
        item_id = str(parts[1]).strip()
        if not item_id:
            continue
        try:
            rating = float(parts[2])
        except Exception:
            continue
        item_counter[item_id] += 1
        total += 1
        if rating >= positive_threshold:
            positive += 1

    real_ctr = (float(positive) / float(total)) if total > 0 else None
    real_item_exposure = counter_to_prob(item_counter)
    return {
        "ratings_path": str(ratings_path),
        "ratings_total": total,
        "positive_total": positive,
        "positive_threshold": positive_threshold,
        "real_ctr": real_ctr,
        "real_item_exposure": real_item_exposure,
    }


def _load_preview_trajectory(preview_path: Path | None) -> dict[str, Any]:
    if preview_path is None or not preview_path.exists():
        return {
            "preview_path": str(preview_path) if preview_path else None,
            "real_pces_trajectory": None,
            "note": "expert_intent_preview.json not available",
        }
    try:
        data = json.loads(preview_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "preview_path": str(preview_path),
            "real_pces_trajectory": None,
            "note": "failed to parse expert_intent_preview.json",
        }

    if not isinstance(data, dict):
        return {
            "preview_path": str(preview_path),
            "real_pces_trajectory": None,
            "note": "expert_intent_preview.json is not a dict",
        }

    per_user_head: list[list[list[float]]] = []
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        states_head = entry.get("states_head")
        if not isinstance(states_head, list) or not states_head:
            continue
        cleaned: list[list[float]] = []
        for row in states_head:
            if not isinstance(row, list) or len(row) < 3:
                continue
            try:
                c = float(row[0])
                e = float(row[1])
                s = float(row[2])
            except Exception:
                continue
            # Alignment report states expert core trajectory is [C,E,S].
            # P has no explicit expert label, so we use neutral P=0.5.
            cleaned.append([0.5, c, e, s])
        if cleaned:
            per_user_head.append(cleaned)

    if not per_user_head:
        return {
            "preview_path": str(preview_path),
            "real_pces_trajectory": None,
            "note": "no usable states_head found",
        }

    max_len = max(len(x) for x in per_user_head)
    traj: list[list[float]] = []
    for i in range(max_len):
        rows = [u[i] for u in per_user_head if i < len(u)]
        if not rows:
            continue
        cols = list(zip(*rows))
        traj.append([float(sum(c) / len(c)) for c in cols])

    return {
        "preview_path": str(preview_path),
        "users_with_preview": len(per_user_head),
        "trajectory_len": len(traj),
        "real_pces_trajectory": traj,
        "note": "PCES trajectory built from states_head with P=0.5 compatibility fill",
    }


def _compute_recovery_rounds(run) -> int:
    hist = run.raw.get("psy_state_history")
    if not isinstance(hist, list):
        return 0
    c = 0
    for step in hist:
        if not isinstance(step, dict):
            continue
        pre = step.get("recovered_before_round")
        post = step.get("recovered")
        if (isinstance(pre, list) and len(pre) > 0) or (isinstance(post, list) and len(post) > 0):
            c += 1
    return c


def _compute_qwen_success_rate(run) -> float | None:
    hist = run.raw.get("psy_state_history")
    if not isinstance(hist, list) or not hist:
        return None
    vals: list[bool] = []
    for step in hist:
        if not isinstance(step, dict):
            continue
        if "qwen_call_success" not in step:
            continue
        vals.append(bool(step.get("qwen_call_success")))
    if not vals:
        return None
    return float(sum(1 for x in vals if x)) / float(len(vals))


def _condition_ab_ci_for_clean(ab_cases) -> dict[str, dict[str, float | int | None]]:
    cases = [c for c in ab_cases if c.condition == "base_clean"]
    if not cases:
        return {
            "Sign Accuracy": {"mean": None, "ci95": None, "n": 0},
            "Lift MAE": {"mean": None, "ci95": None, "n": 0},
            "Type-I Error": {"mean": None, "ci95": None, "n": 0},
            "Power@δ(1%)": {"mean": None, "ci95": None, "n": 0},
            "Power@δ(3%)": {"mean": None, "ci95": None, "n": 0},
            "Power@δ(5%)": {"mean": None, "ci95": None, "n": 0},
        }

    by_seed: dict[str, list[Any]] = {}
    for c in cases:
        by_seed.setdefault(c.seed or "_single", []).append(c)

    bag: dict[str, list[float]] = {
        "Sign Accuracy": [],
        "Lift MAE": [],
        "Type-I Error": [],
        "Power@δ(1%)": [],
        "Power@δ(3%)": [],
        "Power@δ(5%)": [],
    }
    for rows in by_seed.values():
        m = ab_metrics(rows)
        for k in bag.keys():
            v = m.get(k)
            if v is not None:
                bag[k].append(float(v))
    return {k: mean_ci95(vs) for k, vs in bag.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Table 5/6/7 clean rows from base_clean runs")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="robustness_run_manifest_*.jsonl",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("/Users/chongzhang/PsyBer-Agent/experiments/robustness/reports"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Users/chongzhang/WebSim_Dataset"),
    )
    parser.add_argument("--dataset-name", type=str, default="MovieLens-1M")
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--ab-cases-file", type=Path, default=None)
    parser.add_argument("--output-prefix", type=str, default="clean_group_results")
    return parser.parse_args()


def _find_latest_manifest(reports_dir: Path) -> Path:
    cands = sorted(reports_dir.glob("robustness_run_manifest_*.jsonl"))
    if not cands:
        raise FileNotFoundError(f"No robustness_run_manifest_*.jsonl under {reports_dir}")
    return cands[-1]


def main() -> None:
    args = parse_args()
    reports_dir = args.reports_dir.expanduser().resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest.expanduser().resolve() if args.manifest else _find_latest_manifest(reports_dir)
    rows = load_jsonl(manifest_path)
    clean_rows = [
        x
        for x in rows
        if str(x.get("condition_id")) == "base_clean" and str(x.get("status")) == "success"
    ]
    if not clean_rows:
        raise SystemExit(f"No successful base_clean rows found in {manifest_path}")

    run_infos: list[dict[str, Any]] = []
    parsed_runs = []
    for row in clean_rows:
        mem = Path(str(row.get("memory_path") or "")).expanduser()
        if not mem.exists():
            continue
        rec = load_memory_file(mem)
        if rec is None:
            continue
        parsed_runs.append(rec)
        run_infos.append(
            {
                "seed": str(row.get("seed")),
                "memory_path": str(mem),
                "clicks": rec.clicks,
                "track_target": rec.track_target,
                "success": 1.0 if rec.clicks >= rec.track_target else 0.0,
                "latency_per_decision": rec.latency_per_decision_sec,
                "recovery_rounds": _compute_recovery_rounds(rec),
                "qwen_call_success_rate": _compute_qwen_success_rate(rec),
            }
        )
    if not parsed_runs:
        raise SystemExit(f"No parseable memory.json found in base_clean rows from {manifest_path}")

    ratings_path, preview_path = _resolve_dataset_paths(
        args.dataset_root.expanduser().resolve(),
        args.dataset_name,
    )
    gt_ratings = _load_ratings_ground_truth(
        ratings_path,
        positive_threshold=float(args.positive_threshold),
    )
    gt_preview = _load_preview_trajectory(preview_path)

    real_ctr = gt_ratings.get("real_ctr")
    real_exp = gt_ratings.get("real_item_exposure") if isinstance(gt_ratings.get("real_item_exposure"), dict) else None
    real_pces = gt_preview.get("real_pces_trajectory") if isinstance(gt_preview.get("real_pces_trajectory"), list) else None

    ctr_gap_vals: list[float] = []
    exp_js_vals: list[float] = []
    pces_mse_vals: list[float] = []
    success_vals: list[float] = []
    latency_vals: list[float] = []
    recovery_vals: list[float] = []
    qwen_success_vals: list[float] = []
    for rec, info in zip(parsed_runs, run_infos):
        if real_ctr is not None and rec.ctr is not None:
            ctr_gap_vals.append(abs(float(rec.ctr) - float(real_ctr)))
        if real_exp is not None:
            js = js_divergence(counter_to_prob(rec.click_item_counter), real_exp)
            if js is not None:
                exp_js_vals.append(float(js))
        if real_pces is not None:
            mse = trajectory_mse(rec.pces_trajectory, real_pces)
            if mse is not None:
                pces_mse_vals.append(float(mse))

        success_vals.append(float(info["success"]))
        if info["latency_per_decision"] is not None:
            latency_vals.append(float(info["latency_per_decision"]))
        recovery_vals.append(float(info["recovery_rounds"]))
        if info["qwen_call_success_rate"] is not None:
            qwen_success_vals.append(float(info["qwen_call_success_rate"]))

    ab_ci = _condition_ab_ci_for_clean(load_ab_cases(args.ab_cases_file))

    table5 = {
        "CTR Gap": mean_ci95(ctr_gap_vals),
        "Exposure JS": mean_ci95(exp_js_vals),
        "PCES MSE": mean_ci95(pces_mse_vals),
        "Relative Drop (%)": {"mean": 0.0, "ci95": 0.0, "n": len(parsed_runs)},
    }

    table6 = {
        "Sign Accuracy": ab_ci.get("Sign Accuracy"),
        "Lift MAE": ab_ci.get("Lift MAE"),
        "Type-I Error": ab_ci.get("Type-I Error"),
        "Power@δ(1%)": ab_ci.get("Power@δ(1%)"),
        "Power@δ(3%)": ab_ci.get("Power@δ(3%)"),
        "Power@δ(5%)": ab_ci.get("Power@δ(5%)"),
    }
    if table6["Sign Accuracy"]["mean"] is None:
        table6["note"] = "No oracle_lift/pred_lift AB cases were provided for clean condition."

    table7 = {
        "Success Rate": mean_ci95(success_vals),
        "p95 Latency per Decision (s)": percentile(latency_vals, 0.95),
        "Mean Latency per Decision (s)": mean_ci95(latency_vals),
        "Recovery Rounds": mean_ci95(recovery_vals),
        "Sign Accuracy": table6["Sign Accuracy"],
        "Qwen Call Success Rate": mean_ci95(qwen_success_vals),
    }

    payload = {
        "generated_at": datetime.now().isoformat(),
        "source_manifest": str(manifest_path),
        "condition_id": "base_clean",
        "dataset_name": args.dataset_name,
        "n_runs": len(parsed_runs),
        "runs": run_infos,
        "ground_truth": {
            "ratings": gt_ratings,
            "pces_preview": gt_preview,
        },
        "table5_clean": table5,
        "table6_clean": table6,
        "table7_clean": table7,
    }

    tag = _now_tag()
    out_json = reports_dir / f"{args.output_prefix}_{tag}.json"
    out_md = reports_dir / f"{args.output_prefix}_{tag}.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _fmt_ci(x: Any) -> str:
        if not isinstance(x, dict):
            return "N/A" if x is None else str(x)
        m = x.get("mean")
        c = x.get("ci95")
        n = x.get("n")
        if m is None:
            return "N/A"
        if c is None:
            return f"{m} (n={n})"
        return f"{m:.6g} ± {c:.3g} (n={n})"

    lines = [
        "# Clean Group Results (Table 5/6/7)",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- source_manifest: `{manifest_path}`",
        f"- dataset_name: {args.dataset_name}",
        "",
        "## Table 5 Clean",
        f"- CTR Gap: {_fmt_ci(table5['CTR Gap'])}",
        f"- Exposure JS: {_fmt_ci(table5['Exposure JS'])}",
        f"- PCES MSE: {_fmt_ci(table5['PCES MSE'])}",
        "",
        "## Table 6 Clean",
        f"- Sign Accuracy: {_fmt_ci(table6['Sign Accuracy'])}",
        f"- Lift MAE: {_fmt_ci(table6['Lift MAE'])}",
        f"- Type-I Error: {_fmt_ci(table6['Type-I Error'])}",
        "",
        "## Table 7 Clean",
        f"- Success Rate: {_fmt_ci(table7['Success Rate'])}",
        f"- p95 Latency per Decision (s): {table7['p95 Latency per Decision (s)']}",
        f"- Mean Latency per Decision (s): {_fmt_ci(table7['Mean Latency per Decision (s)'])}",
        f"- Recovery Rounds: {_fmt_ci(table7['Recovery Rounds'])}",
        f"- Qwen Call Success Rate: {_fmt_ci(table7['Qwen Call Success Rate'])}",
        "",
        "## Ground Truth",
        f"- ratings.dat: `{gt_ratings.get('ratings_path')}`",
        f"- real_ctr (rating>={args.positive_threshold}): {gt_ratings.get('real_ctr')}",
        f"- item exposure items: {len(gt_ratings.get('real_item_exposure') or {})}",
        f"- expert_intent_preview: `{gt_preview.get('preview_path')}`",
        f"- pces_trajectory_len: {len(gt_preview.get('real_pces_trajectory') or [])}",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("Clean group report generated.")
    print(f"- json: {out_json}")
    print(f"- md: {out_md}")


if __name__ == "__main__":
    main()
