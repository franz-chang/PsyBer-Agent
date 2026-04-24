#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


BATCH_DIR_RE = re.compile(r"(?m)^- batch_dir:\s*(.+?)\s*$")
DT_RE = re.compile(r"\bDT=([0-9]+(?:\.[0-9]+)?)\b")
PROGRESS_RE = re.compile(r"^\[Progress\]\s+finished=(\d+)/(\d+)\b")


@dataclass
class RunMetrics:
    n: int
    repeat_id: int
    status: str
    return_code: int
    batch_dir: str
    throughput_tasks_per_sec: float | None
    p95_latency_per_decision_sec: float | None
    success_rate: float | None
    cost_token: int
    runtime_sec: float | None
    submitted: int | None
    success: int | None
    failed: int | None
    timeout: int | None
    spawn_error: int | None
    note: str = ""


def parse_n_list(raw: str) -> list[int]:
    out: list[int] = []
    for chunk in str(raw).split(","):
        text = chunk.strip().lower()
        if not text:
            continue
        text = text.replace("_", "")
        if text.endswith("k"):
            value = int(float(text[:-1]) * 1000)
        elif text.endswith("m"):
            value = int(float(text[:-1]) * 1_000_000)
        else:
            value = int(text)
        if value <= 0:
            raise ValueError(f"Concurrency must be > 0, got: {chunk}")
        out.append(value)
    if not out:
        raise ValueError("No valid concurrency values were provided.")
    return out


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    arr = sorted(float(x) for x in values)
    pos = (len(arr) - 1) * max(0.0, min(1.0, q))
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(arr[lo])
    frac = pos - lo
    return float(arr[lo] * (1.0 - frac) + arr[hi] * frac)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def stdev_or_none(values: list[float]) -> float | None:
    if len(values) <= 1:
        return None
    return float(statistics.pstdev(values))


def parse_batch_dir(stdout_text: str) -> Path | None:
    m = BATCH_DIR_RE.search(stdout_text)
    if not m:
        return None
    return Path(m.group(1).strip()).expanduser().resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        out.append(json.loads(text))
    return out


def collect_p95_decision_latency_from_logs(results_rows: list[dict[str, Any]]) -> float | None:
    dts: list[float] = []
    for row in results_rows:
        log_file = str(row.get("log_file") or "").strip()
        if not log_file:
            continue
        log_path = Path(log_file).expanduser()
        if not log_path.exists():
            continue
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in DT_RE.finditer(text):
            try:
                dts.append(float(m.group(1)))
            except Exception:
                continue
    return percentile(dts, 0.95)


def _usage_total_from_record(record: dict[str, Any]) -> int:
    usage = record.get("qwen_usage")
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens")
    if total is None:
        in_tok = usage.get("input_tokens")
        out_tok = usage.get("output_tokens")
        try:
            return int((in_tok or 0) + (out_tok or 0))
        except Exception:
            return 0
    try:
        return max(0, int(total))
    except Exception:
        return 0


def collect_cost_token_from_memories(results_rows: list[dict[str, Any]]) -> int:
    total_tokens = 0
    for row in results_rows:
        memory_file = str(row.get("memory_file") or "").strip()
        if not memory_file:
            continue
        memory_path = Path(memory_file).expanduser()
        if not memory_path.exists():
            continue
        try:
            data = read_json(memory_path)
        except Exception:
            continue
        history = data.get("psy_state_history", [])
        if not isinstance(history, list):
            continue
        for rec in history:
            if not isinstance(rec, dict):
                continue
            total_tokens += _usage_total_from_record(rec)
    return int(total_tokens)


def _format_progress_bar(done: int, total: int, *, width: int = 28) -> str:
    safe_total = max(1, int(total))
    safe_done = max(0, min(int(done), safe_total))
    filled = int((safe_done / safe_total) * width)
    if filled > width:
        filled = width
    bar = "#" * filled + "-" * (width - filled)
    pct = (safe_done / safe_total) * 100.0
    return f"[{bar}] {safe_done}/{safe_total} ({pct:5.1f}%)"


def build_cmd(
    *,
    python_bin: str,
    batch_script: Path,
    n: int,
    total_tasks: int,
    max_concurrency: int,
    launch_rate: float,
    task_timeout_sec: float,
    track_override: int | None,
    decision_model: str | None,
    qwen_base_url: str | None,
    dry_run: bool,
    headless: bool,
) -> list[str]:
    cmd = [
        python_bin,
        str(batch_script),
        "--total-tasks",
        str(total_tasks),
        "--max-concurrency",
        str(max_concurrency),
        "--launch-rate",
        str(float(launch_rate)),
        "--task-timeout-sec",
        str(float(task_timeout_sec)),
    ]
    if headless:
        cmd.append("--headless")
    if track_override is not None and int(track_override) > 0:
        cmd.extend(["--track-override", str(int(track_override))])
    if decision_model:
        cmd.extend(["--decision-model", decision_model])
    if qwen_base_url:
        cmd.extend(["--qwen-base-url", qwen_base_url])
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def run_once(
    *,
    project_root: Path,
    cmd: list[str],
    n: int,
    repeat_id: int,
) -> RunMetrics:
    progress_total = max(1, int(n))
    progress_done = 0
    progress_prefix = (
        f"[ProgressBar] concurrent_users_total={progress_total} repeat={repeat_id}"
    )
    last_rendered_done = -1

    def print_progress(done: int) -> None:
        nonlocal last_rendered_done
        if done == last_rendered_done:
            return
        last_rendered_done = done
        line = _format_progress_bar(done, progress_total)
        sys.stdout.write(f"\r{progress_prefix} {line}")
        sys.stdout.flush()

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    combined_output_lines: list[str] = []
    if proc.stdout is not None:
        for line in proc.stdout:
            combined_output_lines.append(line)
            stripped = line.strip()
            m = PROGRESS_RE.match(stripped)
            if not m:
                continue
            try:
                finished = int(m.group(1))
            except Exception:
                continue
            next_done = max(0, min(finished, progress_total))
            if next_done > progress_done:
                progress_done = next_done
                print_progress(progress_done)

    return_code = proc.wait()
    elapsed = time.perf_counter() - t0
    if return_code == 0 and progress_done < progress_total:
        progress_done = progress_total
    print_progress(progress_done)
    sys.stdout.write("\n")
    sys.stdout.flush()

    stdout_text = "".join(combined_output_lines)
    stderr_text = ""
    batch_dir = parse_batch_dir(stdout_text)

    if batch_dir is None:
        note = "batch_dir_not_found_in_stdout"
        if stdout_text.strip():
            note = f"{note}; output={stdout_text.strip()[-400:]}"
        return RunMetrics(
            n=n,
            repeat_id=repeat_id,
            status="failed",
            return_code=int(return_code),
            batch_dir="",
            throughput_tasks_per_sec=None,
            p95_latency_per_decision_sec=None,
            success_rate=None,
            cost_token=0,
            runtime_sec=elapsed,
            submitted=None,
            success=None,
            failed=None,
            timeout=None,
            spawn_error=None,
            note=note,
        )

    summary_path = batch_dir / "summary.json"
    results_path = batch_dir / "results.jsonl"
    if not summary_path.exists():
        return RunMetrics(
            n=n,
            repeat_id=repeat_id,
            status="failed",
            return_code=int(return_code),
            batch_dir=str(batch_dir),
            throughput_tasks_per_sec=None,
            p95_latency_per_decision_sec=None,
            success_rate=None,
            cost_token=0,
            runtime_sec=elapsed,
            submitted=None,
            success=None,
            failed=None,
            timeout=None,
            spawn_error=None,
            note="summary_json_missing",
        )

    summary = read_json(summary_path)
    results_rows = read_jsonl(results_path)
    stats = summary.get("stats", {}) if isinstance(summary, dict) else {}
    submitted = int(stats.get("submitted", 0) or 0)
    success = int(stats.get("success", 0) or 0)
    failed = int(stats.get("failed", 0) or 0)
    timeout = int(stats.get("timeout", 0) or 0)
    spawn_error = int(stats.get("spawn_error", 0) or 0)

    success_rate = (success / submitted) if submitted > 0 else None
    throughput = stats.get("throughput_tasks_per_sec")
    try:
        throughput_val = float(throughput) if throughput is not None else None
    except Exception:
        throughput_val = None

    p95_latency = collect_p95_decision_latency_from_logs(results_rows)
    cost_token = collect_cost_token_from_memories(results_rows)
    runtime_sec = summary.get("elapsed_sec")
    try:
        runtime_val = float(runtime_sec) if runtime_sec is not None else elapsed
    except Exception:
        runtime_val = elapsed

    status = "success" if return_code == 0 else "failed"
    note = ""
    if return_code != 0 and stdout_text.strip():
        note = stdout_text.strip()[-400:]
    return RunMetrics(
        n=n,
        repeat_id=repeat_id,
        status=status,
        return_code=int(return_code),
        batch_dir=str(batch_dir),
        throughput_tasks_per_sec=throughput_val,
        p95_latency_per_decision_sec=p95_latency,
        success_rate=success_rate,
        cost_token=cost_token,
        runtime_sec=runtime_val,
        submitted=submitted,
        success=success,
        failed=failed,
        timeout=timeout,
        spawn_error=spawn_error,
        note=note,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scalability & cost evaluation for PsyBer-Agent."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument(
        "--batch-script",
        type=Path,
        default=Path("src/launch_websim_batch.py"),
    )
    parser.add_argument(
        "--n-list",
        type=str,
        default="500,1000,100000,1000000,5000000",
        help="Comma-separated concurrency list, supports k/m suffix.",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=None,
        help="Run a single concurrency value. Overrides --n-list.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--total-tasks-override",
        type=int,
        default=None,
        help="If set, use fixed total_tasks for each N (otherwise total_tasks=N).",
    )
    parser.add_argument(
        "--max-concurrency-cap",
        type=int,
        default=None,
        help="Optional cap for max-concurrency to avoid exhausting resources.",
    )
    parser.add_argument("--launch-rate", type=float, default=20.0)
    parser.add_argument("--task-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--track-override", type=int, default=20)
    parser.add_argument("--decision-model", type=str, default=None)
    parser.add_argument("--qwen-base-url", type=str, default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/scalability_cost/reports"),
    )
    return parser.parse_args()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(x) for x in value]
    if isinstance(value, tuple):
        return [_to_jsonable(x) for x in value]
    return value


def _fmt_metric(value: float | None, *, percent: bool = False, precision: int = 6) -> str:
    if value is None:
        return "N/A"
    if percent:
        return f"{value * 100:.2f}%"
    return f"{value:.{precision}f}"


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    batch_script = (project_root / args.batch_script).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not batch_script.exists():
        raise FileNotFoundError(f"Batch script not found: {batch_script}")

    if args.n is not None:
        n_values = [int(args.n)]
    else:
        n_values = parse_n_list(args.n_list)
    repeats = max(1, int(args.repeats))

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_records: list[RunMetrics] = []

    print("[Scalability] start")
    print(f"- project_root: {project_root}")
    print(f"- batch_script: {batch_script}")
    print(f"- n_values: {n_values}")
    print(f"- repeats: {repeats}")
    print(f"- dry_run: {args.dry_run}")

    for n in n_values:
        for rid in range(1, repeats + 1):
            total_tasks = (
                int(args.total_tasks_override)
                if args.total_tasks_override is not None
                else int(n)
            )
            max_concurrency = int(n)
            if args.max_concurrency_cap is not None:
                max_concurrency = min(max_concurrency, max(1, int(args.max_concurrency_cap)))
            cmd = build_cmd(
                python_bin=args.python_bin,
                batch_script=batch_script,
                n=n,
                total_tasks=total_tasks,
                max_concurrency=max_concurrency,
                launch_rate=float(args.launch_rate),
                task_timeout_sec=float(args.task_timeout_sec),
                track_override=args.track_override,
                decision_model=args.decision_model,
                qwen_base_url=args.qwen_base_url,
                dry_run=bool(args.dry_run),
                headless=bool(args.headless),
            )
            print(
                f"[Scalability] N={n} repeat={rid}/{repeats} "
                f"total_tasks={total_tasks} max_concurrency={max_concurrency}"
            )
            row = run_once(
                project_root=project_root,
                cmd=cmd,
                n=n,
                repeat_id=rid,
            )
            raw_records.append(row)
            print(
                f"[Scalability] N={n} repeat={rid} status={row.status} "
                f"rc={row.return_code} throughput={row.throughput_tasks_per_sec} "
                f"p95={row.p95_latency_per_decision_sec} success_rate={row.success_rate} "
                f"cost_token={row.cost_token} runtime={row.runtime_sec}"
            )

    # Per-N aggregate (mean over successful runs).
    table_rows: list[dict[str, Any]] = []
    for n in n_values:
        group = [x for x in raw_records if x.n == n and x.status == "success"]
        throughput_vals = [x.throughput_tasks_per_sec for x in group if x.throughput_tasks_per_sec is not None]
        p95_vals = [x.p95_latency_per_decision_sec for x in group if x.p95_latency_per_decision_sec is not None]
        succ_vals = [x.success_rate for x in group if x.success_rate is not None]
        cost_vals = [float(x.cost_token) for x in group]
        run_vals = [x.runtime_sec for x in group if x.runtime_sec is not None]
        table_rows.append(
            {
                "N": n,
                "runs_ok": len(group),
                "runs_total": repeats,
                "throughput_tasks_per_sec_mean": mean_or_none([float(v) for v in throughput_vals]),
                "throughput_tasks_per_sec_std": stdev_or_none([float(v) for v in throughput_vals]),
                "p95_latency_per_decision_sec_mean": mean_or_none([float(v) for v in p95_vals]),
                "p95_latency_per_decision_sec_std": stdev_or_none([float(v) for v in p95_vals]),
                "success_rate_mean": mean_or_none([float(v) for v in succ_vals]),
                "success_rate_std": stdev_or_none([float(v) for v in succ_vals]),
                "cost_token_mean": mean_or_none([float(v) for v in cost_vals]),
                "cost_token_std": stdev_or_none([float(v) for v in cost_vals]),
                "runtime_sec_mean": mean_or_none([float(v) for v in run_vals]),
                "runtime_sec_std": stdev_or_none([float(v) for v in run_vals]),
            }
        )

    summary = {
        "generated_at": datetime.now().isoformat(),
        "project_root": str(project_root),
        "batch_script": str(batch_script),
        "args": _to_jsonable(vars(args)),
        "raw_runs": [x.__dict__ for x in raw_records],
        "table_rows": table_rows,
    }

    json_path = output_dir / f"scalability_cost_{tag}.json"
    csv_path = output_dir / f"scalability_cost_{tag}.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "N",
                "runs_ok",
                "runs_total",
                "throughput_tasks_per_sec_mean",
                "p95_latency_per_decision_sec_mean",
                "success_rate_mean",
                "cost_token_mean",
                "runtime_sec_mean",
                "throughput_tasks_per_sec_std",
                "p95_latency_per_decision_sec_std",
                "success_rate_std",
                "cost_token_std",
                "runtime_sec_std",
            ],
        )
        writer.writeheader()
        for row in table_rows:
            writer.writerow(row)

    print("[Scalability] done")
    print(f"- json: {json_path}")
    print(f"- csv:  {csv_path}")
    print("[Scalability] 4 core metrics (mean over successful runs)")
    for row in table_rows:
        print(f"- N={row['N']}")
        print(
            f"  Throughput(tasks/s): {_fmt_metric(row['throughput_tasks_per_sec_mean'])}"
        )
        print(
            f"  p95 Latency/decision(s): {_fmt_metric(row['p95_latency_per_decision_sec_mean'])}"
        )
        print(f"  Success Rate: {_fmt_metric(row['success_rate_mean'], percent=True)}")
        print(f"  Cost Token: {_fmt_metric(row['cost_token_mean'], precision=2)}")
        print(f"  Runtime(s): {_fmt_metric(row['runtime_sec_mean'])}")


if __name__ == "__main__":
    main()
