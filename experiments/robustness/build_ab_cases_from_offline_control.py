#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from metrics import load_jsonl, load_memory_file


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _norm_text(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("_", " ").replace("-", " ").split())


def _resolve_ratings_path(dataset_root: Path, dataset_name: str) -> Path:
    key = _norm_text(dataset_name)
    if key in {"ml1m", "movie lens 1m", "movie lens", "movielens 1m", "movielens"} or "movielens" in key:
        return dataset_root / "MM-ML-1M-main" / "ratings.dat"
    if "magazine" in key:
        return dataset_root / "Amazon_MM_2018" / "Magazine_Subscriptions" / "ratings.dat"
    if "beauty" in key:
        return dataset_root / "Amazon_MM_2018" / "All_Beauty" / "ratings.dat"
    if "musical instruments" in key or key in {"amazon mi", "amazon_mi", "mi"}:
        return dataset_root / "amazon_v2" / "Musical_Instruments" / "ratings.dat"
    return dataset_root / "MM-ML-1M-main" / "ratings.dat"


def _compute_real_ctr_from_ratings(ratings_path: Path, positive_threshold: float) -> float:
    if not ratings_path.exists():
        raise FileNotFoundError(f"ratings.dat not found: {ratings_path}")
    total = 0
    positive = 0
    for raw in ratings_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("::")
        if len(parts) < 3:
            continue
        try:
            rating = float(parts[2])
        except Exception:
            continue
        total += 1
        if rating >= positive_threshold:
            positive += 1
    if total <= 0:
        raise RuntimeError(f"No valid ratings rows parsed from {ratings_path}")
    return float(positive) / float(total)


def _find_latest_manifest(reports_dir: Path) -> Path:
    cands = sorted(reports_dir.glob("robustness_run_manifest_*.jsonl"))
    if not cands:
        raise FileNotFoundError(f"No robustness_run_manifest_*.jsonl found under {reports_dir}")
    return cands[-1]


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return float(sum(xs)) / float(len(xs))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build ab_cases.json from robustness manifest using offline ratings "
            "CTR as control."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="robustness_run_manifest_*.jsonl (default: latest in reports-dir)",
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
    parser.add_argument(
        "--base-condition-id",
        type=str,
        default="base_clean",
        help="Base condition used as oracle reference per seed.",
    )
    parser.add_argument(
        "--condition-ids",
        type=str,
        default="",
        help="Optional CSV filter of condition_id to include.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output ab_cases json path (default: reports/ab_cases_offline_control_*.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    reports_dir = args.reports_dir.expanduser().resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else _find_latest_manifest(reports_dir)
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    dataset_root = args.dataset_root.expanduser().resolve()
    ratings_path = _resolve_ratings_path(dataset_root, args.dataset_name)
    real_ctr = _compute_real_ctr_from_ratings(ratings_path, float(args.positive_threshold))

    keep_conditions: set[str] | None = None
    if str(args.condition_ids).strip():
        keep_conditions = {x.strip() for x in str(args.condition_ids).split(",") if x.strip()}

    rows = load_jsonl(manifest_path)
    run_rows: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("status")) != "success":
            continue
        cond = str(row.get("condition_id") or "").strip()
        seed = str(row.get("seed") or "").strip()
        mem = Path(str(row.get("memory_path") or "")).expanduser().resolve()
        if not cond or not seed or not mem.exists():
            continue
        if keep_conditions is not None and cond not in keep_conditions:
            continue
        rec = load_memory_file(mem)
        if rec is None or rec.ctr is None:
            continue
        run_rows.append(
            {
                "condition": cond,
                "seed": seed,
                "memory_path": str(mem),
                "run_ctr": float(rec.ctr),
                "pred_lift": float(rec.ctr) - float(real_ctr),
            }
        )

    if not run_rows:
        raise RuntimeError("No successful run rows with parseable CTR found in manifest.")

    base_by_seed: dict[str, float] = {}
    for r in run_rows:
        if r["condition"] == args.base_condition_id:
            base_by_seed[str(r["seed"])] = float(r["pred_lift"])
    if not base_by_seed:
        raise RuntimeError(
            f"No base rows found for base condition `{args.base_condition_id}` in {manifest_path}"
        )

    base_mean = _mean(list(base_by_seed.values()))
    if base_mean is None:
        raise RuntimeError("Failed to compute base oracle mean.")

    ab_cases: list[dict[str, Any]] = []
    for r in sorted(run_rows, key=lambda x: (str(x["condition"]), str(x["seed"]))):
        seed = str(r["seed"])
        oracle = float(base_by_seed.get(seed, base_mean))
        ab_cases.append(
            {
                "condition": str(r["condition"]),
                "seed": seed,
                "oracle_lift": oracle,
                "pred_lift": float(r["pred_lift"]),
                "meta": {
                    "oracle_source": "offline_control_base_seed_lift",
                    "real_ctr_offline": float(real_ctr),
                    "memory_path": str(r["memory_path"]),
                    "run_ctr": float(r["run_ctr"]),
                },
            }
        )

    output_file = (
        args.output_file.expanduser().resolve()
        if args.output_file is not None
        else reports_dir / f"ab_cases_offline_control_{_now_tag()}.json"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "source_manifest": str(manifest_path),
            "dataset_name": args.dataset_name,
            "dataset_root": str(dataset_root),
            "ratings_path": str(ratings_path),
            "positive_threshold": float(args.positive_threshold),
            "real_ctr_offline": float(real_ctr),
            "base_condition_id": str(args.base_condition_id),
            "base_seed_count": len(base_by_seed),
            "case_count": len(ab_cases),
            "note": (
                "oracle_lift is taken from base condition per-seed lift "
                "(run_ctr - offline_real_ctr); pred_lift is each condition per-seed lift."
            ),
        },
        "ab_cases": ab_cases,
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("AB cases generated from offline control.")
    print(f"- output: {output_file}")
    print(f"- real_ctr_offline: {real_ctr:.8f}")
    print(f"- cases: {len(ab_cases)}")


if __name__ == "__main__":
    main()
