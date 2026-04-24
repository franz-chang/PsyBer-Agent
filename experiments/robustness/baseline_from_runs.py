#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from metrics import aggregate_behavior, load_runs_recursive


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _group_by(items, key_fn):
    out = {}
    for x in items:
        k = key_fn(x)
        out.setdefault(k, []).append(x)
    return out


def build_report(runs_root: Path) -> dict[str, Any]:
    runs = load_runs_recursive(runs_root)
    if not runs:
        return {
            "status": "insufficient_data",
            "reason": f"No memory.json found under {runs_root}",
        }

    overall = aggregate_behavior(runs)

    by_dataset = {}
    for ds, ds_runs in sorted(_group_by(runs, lambda r: r.dataset or "unknown").items()):
        by_dataset[ds] = aggregate_behavior(ds_runs)

    by_model = {}
    for mk, mk_runs in sorted(_group_by(runs, lambda r: r.model or "unknown").items()):
        by_model[mk] = aggregate_behavior(mk_runs)

    by_track = {}
    for tk, tk_runs in sorted(_group_by(runs, lambda r: str(r.track_target)).items()):
        by_track[tk] = aggregate_behavior(tk_runs)

    # Baseline from existing runs has no explicit A/B oracle-pred pairs unless user provides them.
    ab_metrics = {
        "Sign Accuracy": None,
        "Lift MAE": None,
        "Type-I Error": None,
        "Power@δ(1%)": None,
        "Power@δ(3%)": None,
        "Power@δ(5%)": None,
    }

    return {
        "status": "ok",
        "scope": "baseline_from_existing_runs",
        "runs_root": str(runs_root),
        "run_count": len(runs),
        "main_metrics": {
            "CTR": overall.get("ctr"),
            "Item Exposure Dist": overall.get("item_exposure_dist"),
            "PCES Avg Trajectory": overall.get("avg_pces_trajectory"),
            "Latency per Decision": overall.get("latency_per_decision_sec"),
        },
        "ab_metrics": ab_metrics,
        "overall": overall,
        "by_dataset": by_dataset,
        "by_model": by_model,
        "by_track": by_track,
        "notes": [
            "This baseline reuses existing runs/**/memory.json only.",
            "A/B metrics are null because oracle_lift/pred_lift pairs are not available in runs memories.",
            "Use run_robustness_suite.py with --ab-cases-file to fill decision metrics.",
        ],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ["# Baseline from Existing Runs", ""]
    if report.get("status") != "ok":
        lines.append(f"- status: {report.get('status')}")
        lines.append(f"- reason: {report.get('reason')}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append(f"- runs_root: `{report['runs_root']}`")
    lines.append(f"- run_count: {report['run_count']}")
    lines.append("")
    lines.append("## Overall")
    overall = report.get("overall", {})
    lines.append(f"- ctr: {overall.get('ctr')}")
    lines.append(f"- latency_per_decision_sec: {overall.get('latency_per_decision_sec')}")
    lines.append(f"- clicks: {overall.get('clicks')}")
    lines.append(f"- decisions: {overall.get('decisions')}")
    lines.append("")
    lines.append("## A/B Metrics")
    for k, v in (report.get("ab_metrics") or {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Notes")
    for note in report.get("notes", []):
        lines.append(f"- {note}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build baseline robustness report from existing runs")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/Users/chongzhang/PsyBer-Agent/runs"),
        help="Root dir containing run folders and memory.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/chongzhang/PsyBer-Agent/experiments/robustness/reports"),
    )
    parser.add_argument("--output-prefix", type=str, default="baseline_runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = _now_tag()

    report = build_report(args.runs_root)

    out_json = args.output_dir / f"{args.output_prefix}_{tag}.json"
    out_md = args.output_dir / f"{args.output_prefix}_{tag}.md"

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, out_md)

    print("Baseline report generated.")
    print(f"- json: {out_json}")
    print(f"- md: {out_md}")


if __name__ == "__main__":
    main()
