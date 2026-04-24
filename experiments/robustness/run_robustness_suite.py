#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from metrics import (
    RunRecord,
    ab_metrics,
    action_flip_rate,
    aggregate_behavior,
    counter_to_prob,
    js_divergence,
    load_ab_cases,
    load_jsonl,
    load_memory_file,
    mean,
    mean_ci95,
    pairwise_js_exposure,
    percentile,
    policy_shift_tv,
    trajectory_mse,
)


@dataclass
class ConditionSpec:
    condition_id: str
    bucket: str
    setting: str
    severity: str
    env: dict[str, str]
    track_override: int | None = None
    dataset_override: str | None = None
    model_override: str | None = None
    concurrency_scale: int | None = None


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_conditions(
    *,
    base_track: int,
    base_dataset: str,
    base_model: str,
    cross_domain_dataset: str,
) -> list[ConditionSpec]:
    return [
        # Base
        ConditionSpec(
            condition_id="base_clean",
            bucket="behavior",
            setting="Base",
            severity="clean",
            env={},
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        # Behavior robustness
        ConditionSpec(
            condition_id="behavior_prompt_paraphrase_mild",
            bucket="behavior",
            setting="Perturbation",
            severity="mild",
            env={"ROBUST_PROMPT_PARAPHRASE": "1"},
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="behavior_affective_shift_mild",
            bucket="behavior",
            setting="Affective shift",
            severity="mild",
            env={"ROBUST_AFFECTIVE_SHIFT": "0.08"},
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="behavior_history_noise_10",
            bucket="behavior",
            setting="History noise",
            severity="10%",
            env={"ROBUST_HISTORY_NOISE": "0.10"},
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="behavior_history_noise_20",
            bucket="behavior",
            setting="History noise",
            severity="20%",
            env={"ROBUST_HISTORY_NOISE": "0.20"},
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="behavior_history_noise_30",
            bucket="behavior",
            setting="History noise",
            severity="30%",
            env={"ROBUST_HISTORY_NOISE": "0.30"},
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="behavior_long_horizon_50",
            bucket="behavior",
            setting="Long horizon",
            severity="track=50",
            env={},
            track_override=50,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="behavior_long_horizon_100",
            bucket="behavior",
            setting="Long horizon",
            severity="track=100",
            env={},
            track_override=100,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="behavior_cross_domain",
            bucket="behavior",
            setting="Cross-domain",
            severity="target domain",
            env={},
            track_override=base_track,
            dataset_override=cross_domain_dataset,
            model_override=base_model,
        ),
        # Decision robustness
        ConditionSpec(
            condition_id="decision_mask_10",
            bucket="decision",
            setting="Click/history mask",
            severity="10%",
            env={"ROBUST_CLICK_HISTORY_MASK": "0.10"},
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="decision_mask_20",
            bucket="decision",
            setting="Click/history mask",
            severity="20%",
            env={"ROBUST_CLICK_HISTORY_MASK": "0.20"},
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="decision_mask_30",
            bucket="decision",
            setting="Click/history mask",
            severity="30%",
            env={"ROBUST_CLICK_HISTORY_MASK": "0.30"},
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="decision_adjacent_swap",
            bucket="decision",
            setting="Adjacent swap",
            severity="1 pair",
            env={"ROBUST_RANKING_PERTURB": "adjacent_swap"},
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="decision_top1_replacement",
            bucket="decision",
            setting="Top-rank replacement",
            severity="top-1",
            env={"ROBUST_RANKING_PERTURB": "top_rank_replacement"},
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="decision_distractor_insertion",
            bucket="decision",
            setting="Distractor insertion",
            severity="1 item",
            env={"ROBUST_RANKING_PERTURB": "distractor_insertion"},
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="decision_conflicting_signals",
            bucket="decision",
            setting="Conflicting signals",
            severity="mixed",
            env={"ROBUST_RANKING_PERTURB": "conflicting_signals"},
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        # System robustness (memory perturbation)
        ConditionSpec(
            condition_id="system_mem_trunc_25",
            bucket="system",
            setting="Memory truncation",
            severity="25%",
            env={"ROBUST_MEMORY_TRUNCATION": "0.25"},
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="system_mem_trunc_50",
            bucket="system",
            setting="Memory truncation",
            severity="50%",
            env={"ROBUST_MEMORY_TRUNCATION": "0.50"},
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="system_mem_shuffle",
            bucket="system",
            setting="Memory shuffle",
            severity="full",
            env={"ROBUST_MEMORY_SHUFFLE": "1"},
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="system_stale_injection",
            bucket="system",
            setting="Stale memory injection",
            severity="full",
            env={"ROBUST_STALE_MEMORY_INJECTION": "1"},
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="system_mem_dropout_30",
            bucket="system",
            setting="Memory dropout",
            severity="30%",
            env={"ROBUST_MEMORY_DROPOUT": "0.30"},
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        # System robustness (concurrency)
        ConditionSpec(
            condition_id="system_concurrency_100",
            bucket="system_concurrency",
            setting="Concurrency",
            severity="100",
            env={},
            concurrency_scale=100,
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="system_concurrency_500",
            bucket="system_concurrency",
            setting="Concurrency",
            severity="500",
            env={},
            concurrency_scale=500,
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="system_concurrency_1k",
            bucket="system_concurrency",
            setting="Concurrency",
            severity="1k",
            env={},
            concurrency_scale=1000,
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="system_concurrency_5k",
            bucket="system_concurrency",
            setting="Concurrency",
            severity="5k",
            env={},
            concurrency_scale=5000,
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
        ConditionSpec(
            condition_id="system_concurrency_10k",
            bucket="system_concurrency",
            setting="Concurrency",
            severity="10k",
            env={},
            concurrency_scale=10000,
            track_override=base_track,
            dataset_override=base_dataset,
            model_override=base_model,
        ),
    ]


def _parse_memory_path(stdout: str) -> Path | None:
    match = re.search(r"(?m)^- memory:\s*(.+?)\s*$", stdout)
    if not match:
        return None
    return Path(match.group(1).strip())


def _parse_batch_dir(stdout: str) -> Path | None:
    match = re.search(r"(?m)^- batch_dir:\s*(.+?)\s*$", stdout)
    if not match:
        return None
    return Path(match.group(1).strip())


def run_single_condition(
    *,
    project_root: Path,
    python_bin: str,
    agent_script: Path,
    cond: ConditionSpec,
    seed: int,
    decision_model: str | None,
    qwen_base_url: str | None,
    headless: bool,
    dry_run: bool,
) -> dict[str, Any]:
    cmd = [
        python_bin,
        str(agent_script),
        "--output-root",
        str(project_root),
    ]
    if headless:
        cmd.append("--headless")
    if cond.track_override is not None:
        cmd.extend(["--track-override", str(cond.track_override)])
    if cond.dataset_override:
        cmd.extend(["--dataset-override", cond.dataset_override])
    if cond.model_override:
        cmd.extend(["--model-override", cond.model_override])
    if decision_model and decision_model.strip():
        cmd.extend(["--decision-model", decision_model.strip()])
    if qwen_base_url and qwen_base_url.strip():
        cmd.extend(["--qwen-base-url", qwen_base_url.strip()])

    env = os.environ.copy()
    env["ROBUST_RUN_LABEL"] = cond.condition_id
    env["ROBUST_SEED"] = str(seed)
    env["PROFILE_SELECTION_MODE"] = "agent_num"
    env["PROFILE_SELECTION_SEED"] = str(seed)
    if decision_model and decision_model.strip():
        env["QWEN_MODEL"] = decision_model.strip()
    if qwen_base_url and qwen_base_url.strip():
        env["QWEN_BASE_URL"] = qwen_base_url.strip()
    for k, v in cond.env.items():
        env[k] = str(v)

    if dry_run:
        return {
            "condition_id": cond.condition_id,
            "bucket": cond.bucket,
            "setting": cond.setting,
            "severity": cond.severity,
            "seed": str(seed),
            "status": "dry_run",
            "command": cmd,
            "env_overrides": {k: env[k] for k in sorted(set(["ROBUST_RUN_LABEL", "ROBUST_SEED", "PROFILE_SELECTION_MODE", "PROFILE_SELECTION_SEED", *cond.env.keys()]))},
            "memory_path": None,
        }

    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    memory_path = _parse_memory_path(stdout)

    return {
        "condition_id": cond.condition_id,
        "bucket": cond.bucket,
        "setting": cond.setting,
        "severity": cond.severity,
        "seed": str(seed),
        "status": "success" if (proc.returncode == 0 and memory_path is not None) else "failed",
        "return_code": proc.returncode,
        "command": cmd,
        "memory_path": str(memory_path) if memory_path is not None else None,
        "stdout_tail": "\n".join(stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-20:]),
    }


def run_concurrency_condition(
    *,
    project_root: Path,
    python_bin: str,
    batch_script: Path,
    agent_script: Path,
    cond: ConditionSpec,
    max_concurrency: int,
    launch_rate: float,
    task_timeout_sec: float,
    decision_model: str | None,
    qwen_base_url: str | None,
    headless: bool,
    dry_run: bool,
) -> dict[str, Any]:
    assert cond.concurrency_scale is not None
    concurrency = min(int(cond.concurrency_scale), int(max_concurrency))

    cmd = [
        python_bin,
        str(batch_script),
        "--total-tasks",
        str(cond.concurrency_scale),
        "--max-concurrency",
        str(max(1, concurrency)),
        "--launch-rate",
        str(launch_rate),
        "--task-timeout-sec",
        str(task_timeout_sec),
        "--agent-script",
        str(agent_script),
        "--output-root",
        str(project_root),
        "--profile-selection-mode",
        "agent_num",
        "--disable-api-pool",
        "--track-override",
        str(cond.track_override or 20),
    ]
    if headless:
        cmd.append("--headless")

    env = os.environ.copy()
    env["ROBUST_RUN_LABEL"] = cond.condition_id
    env["ROBUST_SEED"] = "0"
    if cond.dataset_override:
        env["ROBUST_DATASET_OVERRIDE"] = cond.dataset_override
    if cond.model_override:
        env["ROBUST_MODEL_OVERRIDE"] = cond.model_override
    if decision_model and decision_model.strip():
        env["QWEN_MODEL"] = decision_model.strip()
    if qwen_base_url and qwen_base_url.strip():
        env["QWEN_BASE_URL"] = qwen_base_url.strip()
    for k, v in cond.env.items():
        env[k] = str(v)

    if dry_run:
        return {
            "condition_id": cond.condition_id,
            "bucket": cond.bucket,
            "setting": cond.setting,
            "severity": cond.severity,
            "status": "dry_run",
            "command": cmd,
            "batch_summary_path": None,
            "batch_results_path": None,
        }

    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    batch_dir = _parse_batch_dir(stdout)

    summary_path = None
    results_path = None
    if batch_dir is not None:
        cand_summary = batch_dir / "summary.json"
        cand_results = batch_dir / "results.jsonl"
        summary_path = str(cand_summary) if cand_summary.exists() else None
        results_path = str(cand_results) if cand_results.exists() else None

    status = "success" if (proc.returncode == 0 and summary_path is not None) else "failed"
    return {
        "condition_id": cond.condition_id,
        "bucket": cond.bucket,
        "setting": cond.setting,
        "severity": cond.severity,
        "status": status,
        "return_code": proc.returncode,
        "command": cmd,
        "batch_summary_path": summary_path,
        "batch_results_path": results_path,
        "stdout_tail": "\n".join(stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-20:]),
    }


def _condition_ab_ci(condition_id: str, ab_cases) -> dict[str, Any]:
    cases = [c for c in ab_cases if c.condition == condition_id]
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
        k = c.seed or "_single"
        by_seed.setdefault(k, []).append(c)

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


def evaluate_suite(
    *,
    conditions: list[ConditionSpec],
    run_manifest: list[dict[str, Any]],
    batch_manifest: list[dict[str, Any]],
    ab_cases,
) -> dict[str, Any]:
    run_by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in run_manifest:
        run_by_condition.setdefault(str(row.get("condition_id")), []).append(row)

    loaded_runs: dict[tuple[str, str], RunRecord] = {}
    for cond_id, rows in run_by_condition.items():
        for r in rows:
            if r.get("status") not in {"success", "dry_run"}:
                continue
            if not r.get("memory_path"):
                continue
            mem = Path(str(r.get("memory_path")))
            if not mem.exists():
                continue
            parsed = load_memory_file(mem)
            if parsed is None:
                continue
            loaded_runs[(cond_id, str(r.get("seed") or ""))] = parsed

    base_seed_runs = {
        seed: rec
        for (cond_id, seed), rec in loaded_runs.items()
        if cond_id == "base_clean"
    }

    behavior_rows = []
    decision_rows = []
    system_rows = []

    for cond in conditions:
        if cond.bucket not in {"behavior", "decision", "system"}:
            continue
        rows = run_by_condition.get(cond.condition_id, [])
        cond_seed_runs = {
            str(r.get("seed") or ""): loaded_runs[(cond.condition_id, str(r.get("seed") or ""))]
            for r in rows
            if (cond.condition_id, str(r.get("seed") or "")) in loaded_runs
        }

        ctr_gaps: list[float] = []
        exp_js: list[float] = []
        pces_mse: list[float] = []
        latency_vals: list[float] = []
        state_drift_vals: list[float] = []
        action_flip_vals: list[float] = []
        policy_shift_vals: list[float] = []
        rel_drop_vals: list[float] = []

        for seed, run in cond_seed_runs.items():
            if run.latency_per_decision_sec is not None:
                latency_vals.append(float(run.latency_per_decision_sec))

            base = base_seed_runs.get(seed)
            if cond.condition_id == "base_clean" or base is None:
                continue

            ctr_gap = abs(float(run.ctr or 0.0) - float(base.ctr or 0.0))
            js = js_divergence(
                counter_to_prob(run.click_item_counter),
                counter_to_prob(base.click_item_counter),
            )
            mse = trajectory_mse(run.pces_trajectory, base.pces_trajectory)

            if ctr_gap is not None:
                ctr_gaps.append(float(ctr_gap))
            if js is not None:
                exp_js.append(float(js))
            if mse is not None:
                pces_mse.append(float(mse))
                state_drift_vals.append(float(mse))

            afr = action_flip_rate(base.action_sequence, run.action_sequence)
            pshift = policy_shift_tv(base.action_sequence, run.action_sequence)
            if afr is not None:
                action_flip_vals.append(float(afr))
            if pshift is not None:
                policy_shift_vals.append(float(pshift))

            rel_parts: list[float] = []
            base_ctr = float(base.ctr or 0.0)
            rel_parts.append(abs(float(run.ctr or 0.0) - base_ctr) / max(abs(base_ctr), 1e-6))
            if js is not None:
                rel_parts.append(float(js))
            if mse is not None:
                rel_parts.append(float(mse))
            if run.latency_per_decision_sec is not None and base.latency_per_decision_sec is not None:
                rel_parts.append(
                    max(
                        0.0,
                        (float(run.latency_per_decision_sec) - float(base.latency_per_decision_sec))
                        / max(float(base.latency_per_decision_sec), 1e-6),
                    )
                )
            rel_drop_vals.append(100.0 * float(mean(rel_parts) or 0.0))

        intra_profile_js = pairwise_js_exposure(list(cond_seed_runs.values()))

        row = {
            "condition_id": cond.condition_id,
            "setting": cond.setting,
            "severity": cond.severity,
            "run_count": len(cond_seed_runs),
            "Sign Accuracy": _condition_ab_ci(cond.condition_id, ab_cases).get("Sign Accuracy"),
            "Lift MAE": _condition_ab_ci(cond.condition_id, ab_cases).get("Lift MAE"),
            "Type-I Error": _condition_ab_ci(cond.condition_id, ab_cases).get("Type-I Error"),
            "Power@δ(1%)": _condition_ab_ci(cond.condition_id, ab_cases).get("Power@δ(1%)"),
            "Power@δ(3%)": _condition_ab_ci(cond.condition_id, ab_cases).get("Power@δ(3%)"),
            "Power@δ(5%)": _condition_ab_ci(cond.condition_id, ab_cases).get("Power@δ(5%)"),
            "CTR Gap": mean_ci95(ctr_gaps) if cond.condition_id != "base_clean" else {"mean": 0.0, "ci95": 0.0, "n": len(cond_seed_runs)},
            "Item Exposure JS": mean_ci95(exp_js) if cond.condition_id != "base_clean" else {"mean": 0.0, "ci95": 0.0, "n": len(cond_seed_runs)},
            "PCES Trajectory MSE": mean_ci95(pces_mse) if cond.condition_id != "base_clean" else {"mean": 0.0, "ci95": 0.0, "n": len(cond_seed_runs)},
            "Latency per Decision": mean_ci95(latency_vals),
            "Relative Drop (%)": mean_ci95(rel_drop_vals) if cond.condition_id != "base_clean" else {"mean": 0.0, "ci95": 0.0, "n": len(cond_seed_runs)},
            "Intra-profile Exposure JS": intra_profile_js,
            "State Drift": mean(state_drift_vals),
            "Action Flip Rate": mean(action_flip_vals),
            "Policy Shift": mean(policy_shift_vals),
        }

        if cond.bucket == "behavior":
            behavior_rows.append(row)
        elif cond.bucket == "decision":
            decision_rows.append(row)
        elif cond.bucket == "system":
            # Success rate for non-concurrency system perturbations (track completion)
            succ = []
            for run in cond_seed_runs.values():
                succ.append(1.0 if run.clicks >= run.track_target else 0.0)
            row["Success Rate"] = mean_ci95(succ)
            # Recovery rounds from memory payload
            rec_vals = []
            for run in cond_seed_runs.values():
                hist = run.raw.get("psy_state_history") if isinstance(run.raw.get("psy_state_history"), list) else []
                c = 0
                for s in hist:
                    if not isinstance(s, dict):
                        continue
                    pre = s.get("recovered_before_round")
                    post = s.get("recovered")
                    if (isinstance(pre, list) and pre) or (isinstance(post, list) and post):
                        c += 1
                rec_vals.append(float(c))
            row["Recovery Rounds"] = mean_ci95(rec_vals)
            system_rows.append(row)

    # Add concurrency rows
    for row in batch_manifest:
        cond_id = str(row.get("condition_id"))
        cond = next((c for c in conditions if c.condition_id == cond_id), None)
        if cond is None:
            continue
        summary_path = Path(str(row.get("batch_summary_path"))) if row.get("batch_summary_path") else None
        results_path = Path(str(row.get("batch_results_path"))) if row.get("batch_results_path") else None

        success_rate = None
        throughput = None
        timeout_rate = None
        p95_latency = None
        if summary_path is not None and summary_path.exists():
            try:
                s = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                s = {}
            stats = s.get("stats") if isinstance(s.get("stats"), dict) else {}
            submitted = float(stats.get("submitted", 0.0) or 0.0)
            succ = float(stats.get("success", 0.0) or 0.0)
            timeout = float(stats.get("timeout", 0.0) or 0.0)
            success_rate = succ / submitted if submitted > 0 else None
            timeout_rate = timeout / submitted if submitted > 0 else None
            throughput = stats.get("throughput_tasks_per_sec")

        if results_path is not None and results_path.exists():
            rows = load_jsonl(results_path)
            elapsed = [float(x.get("elapsed_sec")) for x in rows if x.get("elapsed_sec") is not None]
            p95_latency = percentile(elapsed, 0.95)

        ab_ci = _condition_ab_ci(cond_id, ab_cases)
        system_rows.append(
            {
                "condition_id": cond_id,
                "setting": cond.setting,
                "severity": cond.severity,
                "run_count": None,
                "Success Rate": {"mean": success_rate, "ci95": None, "n": 1 if success_rate is not None else 0},
                "p95 Latency": p95_latency,
                "Throughput": throughput,
                "Timeout Rate": timeout_rate,
                "Sign Accuracy": ab_ci.get("Sign Accuracy"),
                "Lift MAE": ab_ci.get("Lift MAE"),
                "Type-I Error": ab_ci.get("Type-I Error"),
                "Power@δ(1%)": ab_ci.get("Power@δ(1%)"),
                "Power@δ(3%)": ab_ci.get("Power@δ(3%)"),
                "Power@δ(5%)": ab_ci.get("Power@δ(5%)"),
            }
        )

    return {
        "generated_at": datetime.now().isoformat(),
        "behavior_robustness": behavior_rows,
        "decision_robustness": decision_rows,
        "system_robustness": system_rows,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# PsyBer-Agent Robustness Suite")
    lines.append("")
    lines.append(f"Generated at: {report.get('generated_at')}")
    lines.append("")

    def _fmt_ci(x: Any) -> str:
        if not isinstance(x, dict):
            return str(x)
        m = x.get("mean")
        c = x.get("ci95")
        n = x.get("n")
        if m is None:
            return "N/A"
        if c is None:
            return f"{m} (n={n})"
        return f"{m:.6g} ± {c:.3g} (n={n})"

    lines.append("## Behavior Robustness")
    for row in report.get("behavior_robustness", []):
        lines.append(
            f"- {row.get('setting')} | {row.get('severity')} | "
            f"CTR Gap={_fmt_ci(row.get('CTR Gap'))}, "
            f"Exposure JS={_fmt_ci(row.get('Item Exposure JS'))}, "
            f"PCES MSE={_fmt_ci(row.get('PCES Trajectory MSE'))}, "
            f"Relative Drop={_fmt_ci(row.get('Relative Drop (%)'))}"
        )
    lines.append("")

    lines.append("## Decision Robustness")
    for row in report.get("decision_robustness", []):
        lines.append(
            f"- {row.get('setting')} | {row.get('severity')} | "
            f"Sign Accuracy={_fmt_ci(row.get('Sign Accuracy'))}, "
            f"Lift MAE={_fmt_ci(row.get('Lift MAE'))}, "
            f"Type-I={_fmt_ci(row.get('Type-I Error'))}, "
            f"Action Flip Rate={row.get('Action Flip Rate')}, "
            f"Policy Shift={row.get('Policy Shift')}"
        )
    lines.append("")

    lines.append("## System Robustness")
    for row in report.get("system_robustness", []):
        lines.append(
            f"- {row.get('setting')} | {row.get('severity')} | "
            f"Success={_fmt_ci(row.get('Success Rate'))}, "
            f"p95 Latency={row.get('p95 Latency')}, "
            f"Sign Accuracy={_fmt_ci(row.get('Sign Accuracy'))}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run robustness suite for PsyBer-Agent")
    parser.add_argument("--project-root", type=Path, default=Path("/Users/chongzhang/PsyBer-Agent"))
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--agent-script", type=Path, default=Path("/Users/chongzhang/PsyBer-Agent/experiments/robustness/robust_websim_agent.py"))
    parser.add_argument("--batch-script", type=Path, default=Path("/Users/chongzhang/PsyBer-Agent/src/launch_websim_batch.py"))
    parser.add_argument("--output-dir", type=Path, default=Path("/Users/chongzhang/PsyBer-Agent/experiments/robustness/reports"))
    parser.add_argument("--seeds", type=str, default="101,202,303")

    parser.add_argument("--base-track", type=int, default=20)
    parser.add_argument("--base-dataset", type=str, default="MovieLens-1M")
    parser.add_argument("--base-model", type=str, default="LightGCN")
    parser.add_argument("--cross-domain-dataset", type=str, default="Amazon Magazine Subscriptions")
    parser.add_argument("--decision-model", type=str, default=None)
    parser.add_argument("--qwen-base-url", type=str, default=None)

    parser.add_argument("--max-concurrency", type=int, default=120)
    parser.add_argument("--launch-rate", type=float, default=15.0)
    parser.add_argument("--task-timeout-sec", type=float, default=1800.0)

    parser.add_argument("--ab-cases-file", type=Path, default=None)
    parser.add_argument(
        "--condition-ids",
        type=str,
        default="",
        help="Comma-separated condition ids to run, e.g. base_clean,system_concurrency_100",
    )
    parser.add_argument("--skip-launch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tag = now_tag()
    run_manifest_path = args.output_dir / f"robustness_run_manifest_{tag}.jsonl"
    batch_manifest_path = args.output_dir / f"robustness_batch_manifest_{tag}.jsonl"
    report_json_path = args.output_dir / f"robustness_report_{tag}.json"
    report_md_path = args.output_dir / f"robustness_report_{tag}.md"

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    conditions = default_conditions(
        base_track=args.base_track,
        base_dataset=args.base_dataset,
        base_model=args.base_model,
        cross_domain_dataset=args.cross_domain_dataset,
    )
    if args.condition_ids.strip():
        allowed = {x.strip() for x in args.condition_ids.split(",") if x.strip()}
        conditions = [c for c in conditions if c.condition_id in allowed]
        if not conditions:
            raise ValueError(
                f"No valid conditions matched --condition-ids={args.condition_ids!r}"
            )

    run_manifest: list[dict[str, Any]] = []
    batch_manifest: list[dict[str, Any]] = []

    if not args.skip_launch:
        for cond in conditions:
            if cond.bucket == "system_concurrency":
                row = run_concurrency_condition(
                    project_root=args.project_root,
                    python_bin=args.python_bin,
                    batch_script=args.batch_script,
                    agent_script=args.agent_script,
                    cond=cond,
                    max_concurrency=args.max_concurrency,
                    launch_rate=args.launch_rate,
                    task_timeout_sec=args.task_timeout_sec,
                    decision_model=args.decision_model,
                    qwen_base_url=args.qwen_base_url,
                    headless=args.headless,
                    dry_run=args.dry_run,
                )
                batch_manifest.append(row)
                print(f"[Concurrency] {cond.condition_id}: {row.get('status')}")
                continue

            for seed in seeds:
                row = run_single_condition(
                    project_root=args.project_root,
                    python_bin=args.python_bin,
                    agent_script=args.agent_script,
                    cond=cond,
                    seed=seed,
                    decision_model=args.decision_model,
                    qwen_base_url=args.qwen_base_url,
                    headless=args.headless,
                    dry_run=args.dry_run,
                )
                run_manifest.append(row)
                print(f"[Run] {cond.condition_id} seed={seed}: {row.get('status')}")

        run_manifest_path.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in run_manifest) + ("\n" if run_manifest else ""),
            encoding="utf-8",
        )
        batch_manifest_path.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in batch_manifest) + ("\n" if batch_manifest else ""),
            encoding="utf-8",
        )
    else:
        # Load latest manifests if skip-launch is enabled.
        run_manifest = load_jsonl(run_manifest_path) if run_manifest_path.exists() else []
        batch_manifest = load_jsonl(batch_manifest_path) if batch_manifest_path.exists() else []

    ab_cases = load_ab_cases(args.ab_cases_file)

    report = evaluate_suite(
        conditions=conditions,
        run_manifest=run_manifest,
        batch_manifest=batch_manifest,
        ab_cases=ab_cases,
    )

    report["inputs"] = {
        "project_root": str(args.project_root),
        "python_bin": args.python_bin,
        "agent_script": str(args.agent_script),
        "batch_script": str(args.batch_script),
        "seeds": seeds,
        "base_track": args.base_track,
        "base_dataset": args.base_dataset,
        "base_model": args.base_model,
        "cross_domain_dataset": args.cross_domain_dataset,
        "decision_model": args.decision_model,
        "qwen_base_url": args.qwen_base_url,
        "ab_cases_file": str(args.ab_cases_file) if args.ab_cases_file else None,
        "skip_launch": bool(args.skip_launch),
        "dry_run": bool(args.dry_run),
    }
    report["artifacts"] = {
        "run_manifest": str(run_manifest_path),
        "batch_manifest": str(batch_manifest_path),
        "report_json": str(report_json_path),
        "report_md": str(report_md_path),
    }

    report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, report_md_path)

    print("Robustness suite report generated.")
    print(f"- run manifest: {run_manifest_path}")
    print(f"- batch manifest: {batch_manifest_path}")
    print(f"- report json: {report_json_path}")
    print(f"- report md: {report_md_path}")


if __name__ == "__main__":
    main()
