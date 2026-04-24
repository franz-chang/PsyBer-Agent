#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


BEHAVIOR_CONDITION_IDS = [
    "base_clean",
    "behavior_prompt_paraphrase_mild",
    "behavior_affective_shift_mild",
    "behavior_history_noise_10",
    "behavior_history_noise_20",
    "behavior_history_noise_30",
    "behavior_long_horizon_50",
    "behavior_long_horizon_100",
    "behavior_cross_domain",
]


def _latest_manifest(reports_dir: Path) -> Path | None:
    files = sorted(reports_dir.glob("robustness_run_manifest_*.jsonl"))
    return files[-1] if files else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Behavior Robustness experiments for PsyBer-Agent."
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
    parser.add_argument(
        "--cross-domain-dataset",
        type=str,
        default="Amazon Magazine Subscriptions",
    )

    parser.add_argument("--decision-model", type=str, default="qwen-plus-0112")
    parser.add_argument("--qwen-base-url", type=str, default=None)
    parser.add_argument("--ab-cases-file", type=Path, default=None)

    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("/Users/chongzhang/PsyBer-Agent/experiments/robustness/reports"),
    )

    parser.add_argument(
        "--build-clean-ground-truth",
        action="store_true",
        help="After run, build clean Table 5/6/7 row vs WebSim_Dataset ground truth.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Users/chongzhang/WebSim_Dataset"),
    )
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    reports_dir = args.reports_dir.expanduser().resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    suite_script = project_root / "experiments" / "robustness" / "run_robustness_suite.py"
    clean_script = project_root / "experiments" / "robustness" / "clean_group_results.py"
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
        ",".join(BEHAVIOR_CONDITION_IDS),
        "--seeds",
        args.seeds,
        "--base-track",
        str(args.base_track),
        "--base-dataset",
        args.base_dataset,
        "--base-model",
        args.base_model,
        "--cross-domain-dataset",
        args.cross_domain_dataset,
        "--decision-model",
        args.decision_model,
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

    if args.build_clean_ground_truth and not args.dry_run and not args.skip_launch:
        manifest = _latest_manifest(reports_dir)
        if manifest is not None and clean_script.exists():
            clean_cmd = [
                args.python_bin,
                str(clean_script),
                "--manifest",
                str(manifest),
                "--reports-dir",
                str(reports_dir),
                "--dataset-root",
                str(args.dataset_root.expanduser().resolve()),
                "--dataset-name",
                args.base_dataset,
                "--positive-threshold",
                str(args.positive_threshold),
            ]
            if args.ab_cases_file:
                clean_cmd.extend(
                    ["--ab-cases-file", str(args.ab_cases_file.expanduser().resolve())]
                )
            clean_proc = subprocess.run(
                clean_cmd,
                cwd=str(project_root),
                env=env,
                check=False,
            )
            if clean_proc.returncode != 0:
                raise SystemExit(clean_proc.returncode)


if __name__ == "__main__":
    main()
