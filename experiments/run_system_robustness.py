#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SYSTEM_CONDITION_IDS = [
    "base_clean",
    "system_mem_trunc_25",
    "system_mem_trunc_50",
    "system_mem_shuffle",
    "system_stale_injection",
    "system_mem_dropout_30",
    "system_concurrency_100",
    "system_concurrency_500",
    "system_concurrency_1k",
    "system_concurrency_5k",
    "system_concurrency_10k",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run System Robustness experiments for PsyBer-Agent."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/Users/chongzhang/PsyBer-Agent"),
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--seeds", type=str, default="101,202,303")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-launch", action="store_true")

    parser.add_argument("--base-track", type=int, default=20)
    parser.add_argument("--base-dataset", type=str, default="MovieLens-1M")
    parser.add_argument("--base-model", type=str, default="LightGCN")

    parser.add_argument("--decision-model", type=str, default="qwen-plus-0112")
    parser.add_argument("--qwen-base-url", type=str, default=None)
    parser.add_argument("--ab-cases-file", type=Path, default=None)

    parser.add_argument("--max-concurrency", type=int, default=120)
    parser.add_argument("--launch-rate", type=float, default=15.0)
    parser.add_argument("--task-timeout-sec", type=float, default=1800.0)

    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("/Users/chongzhang/PsyBer-Agent/experiments/robustness/reports"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    reports_dir = args.reports_dir.expanduser().resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    suite_script = project_root / "experiments" / "robustness" / "run_robustness_suite.py"
    if not suite_script.exists():
        raise FileNotFoundError(f"Missing script: {suite_script}")

    cmd = [
        args.python_bin,
        str(suite_script),
        "--project-root",
        str(project_root),
        "--python-bin",
        args.python_bin,
        "--output-dir",
        str(reports_dir),
        "--condition-ids",
        ",".join(SYSTEM_CONDITION_IDS),
        "--seeds",
        args.seeds,
        "--base-track",
        str(args.base_track),
        "--base-dataset",
        args.base_dataset,
        "--base-model",
        args.base_model,
        "--decision-model",
        args.decision_model,
        "--max-concurrency",
        str(args.max_concurrency),
        "--launch-rate",
        str(args.launch_rate),
        "--task-timeout-sec",
        str(args.task_timeout_sec),
    ]
    if args.qwen_base_url:
        cmd.extend(["--qwen-base-url", args.qwen_base_url])
    if args.ab_cases_file:
        cmd.extend(["--ab-cases-file", str(args.ab_cases_file.expanduser().resolve())])
    if args.headless:
        cmd.append("--headless")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.skip_launch:
        cmd.append("--skip-launch")

    env = os.environ.copy()
    if args.decision_model:
        env["QWEN_MODEL"] = args.decision_model
    if args.qwen_base_url:
        env["QWEN_BASE_URL"] = args.qwen_base_url

    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
