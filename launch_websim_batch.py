#!/usr/bin/env python3
"""Launch a large number of WebSim agent runs with concurrency and rate limiting.

Example:
  python launch_websim_batch.py \
    --total-tasks 10000 \
    --max-concurrency 120 \
    --launch-rate 15 \
    --track-override 20 \
    --headless
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_AGENT_SCRIPT = PROJECT_DIR / "web_sim_camel_agent.py"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR
DEFAULT_WEB_SIM_DIR = Path("/Users/chongzhang/Web_sim")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_python_bin() -> str:
    dedicated = Path("/opt/anaconda3/envs/psyber/bin/python")
    if dedicated.exists() and os.access(dedicated, os.X_OK):
        return str(dedicated)

    conda_prefix = os.getenv("CONDA_PREFIX", "").strip()
    if conda_prefix:
        cand = Path(conda_prefix) / "bin" / "python"
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)

    return sys.executable or "python3"


def detect_service_python_bin(fallback_python: str) -> str:
    default = Path("/opt/anaconda3/bin/python3")
    if default.exists() and os.access(default, os.X_OK):
        return str(default)
    return fallback_python


def service_is_alive(
    port: int,
    timeout_sec: float = 3.0,
    *,
    host: str = "127.0.0.1",
    path: str = "/",
) -> bool:
    # 1) Fast TCP probe: avoid triggering heavy endpoints (e.g. /health loads models).
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_sec):
            pass
    except Exception:
        return False

    # 2) Optional HTTP probe on a light page.
    try:
        with urlopen(f"http://{host}:{port}{path}", timeout=timeout_sec) as resp:
            return int(resp.status) == 200
    except Exception:
        # TCP is already alive; treat as up.
        return True


@dataclass
class LaunchResult:
    task_id: int
    status: str
    return_code: int | None
    elapsed_sec: float
    start_utc: str
    end_utc: str
    log_file: str
    command: list[str]
    error: str = ""


@dataclass
class SharedStats:
    submitted: int
    started: int = 0
    finished: int = 0
    success: int = 0
    failed: int = 0
    timeout: int = 0
    spawn_error: int = 0


class StartRateLimiter:
    def __init__(self, starts_per_sec: float):
        self.starts_per_sec = float(max(0.0, starts_per_sec))
        self.interval = 0.0 if self.starts_per_sec <= 0 else 1.0 / self.starts_per_sec
        self.next_ready = 0.0
        self.lock = asyncio.Lock()

    async def wait_slot(self) -> None:
        if self.interval <= 0:
            return
        async with self.lock:
            now = time.monotonic()
            if self.next_ready <= 0:
                self.next_ready = now
            wait_for = self.next_ready - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
                now = time.monotonic()
            self.next_ready = max(self.next_ready, now) + self.interval


def build_cmd(
    *,
    python_bin: str,
    agent_script: Path,
    output_root: Path,
    headless: bool,
    track_override: int | None,
    decision_model: str | None,
    qwen_base_url: str | None,
) -> list[str]:
    cmd = [
        python_bin,
        str(agent_script),
        "--output-root",
        str(output_root),
    ]
    if headless:
        cmd.append("--headless")
    if track_override and track_override > 0:
        cmd.extend(["--track-override", str(track_override)])
    if decision_model:
        cmd.extend(["--decision-model", decision_model])
    if qwen_base_url:
        cmd.extend(["--qwen-base-url", qwen_base_url])
    return cmd


def start_service_if_needed(
    *,
    enabled: bool,
    web_sim_dir: Path,
    port: int,
    timeout_sec: float,
    log_file: Path,
    python_bin: str,
) -> tuple[subprocess.Popen[str] | None, bool]:
    if not enabled:
        return None, False
    if service_is_alive(port, timeout_sec=1.5, path="/"):
        print(f"[Service] Already alive on 127.0.0.1:{port}")
        return None, False

    env = os.environ.copy()
    env["PORT"] = str(port)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_file.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [python_bin, "app.py"],
            cwd=str(web_sim_dir),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=env,
        )
    finally:
        log_fp.close()

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if service_is_alive(port, timeout_sec=min(2.0, timeout_sec), path="/"):
            print(f"[Service] Started on 127.0.0.1:{port}, log={log_file}")
            return proc, True
        time.sleep(0.5)

    # Avoid leaking the spawned Flask process on startup timeout.
    try:
        if proc.pid:
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    raise RuntimeError(
        f"Service failed to start on 127.0.0.1:{port} within {timeout_sec}s. "
        f"Check log: {log_file}"
    )


def stop_service(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=8)
        return
    except Exception:
        pass
    try:
        if proc.pid:
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass


async def run_one_task(
    *,
    task_id: int,
    worker_idx: int,
    cmd: list[str],
    env: dict[str, str],
    log_file: Path,
    timeout_sec: float,
    limiter: StartRateLimiter,
    print_task_pid: bool,
) -> LaunchResult:
    await limiter.wait_slot()
    start_ts = time.perf_counter()
    start_utc = utc_now()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    proc: asyncio.subprocess.Process | None = None
    try:
        with log_file.open("wb") as fp:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=fp,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            if print_task_pid:
                print(
                    f"[TaskPID] worker={worker_idx} task_id={task_id} pid={proc.pid} log={log_file}",
                    flush=True,
                )
            try:
                rc = await asyncio.wait_for(proc.wait(), timeout=timeout_sec)
            except asyncio.TimeoutError:
                if proc.returncode is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except Exception:
                        proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=6)
                    except Exception:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except Exception:
                            proc.kill()
                        await proc.wait()
                return LaunchResult(
                    task_id=task_id,
                    status="timeout",
                    return_code=proc.returncode,
                    elapsed_sec=time.perf_counter() - start_ts,
                    start_utc=start_utc,
                    end_utc=utc_now(),
                    log_file=str(log_file),
                    command=cmd,
                    error=f"timeout>{timeout_sec}s",
                )

        status = "success" if rc == 0 else "failed"
        return LaunchResult(
            task_id=task_id,
            status=status,
            return_code=rc,
            elapsed_sec=time.perf_counter() - start_ts,
            start_utc=start_utc,
            end_utc=utc_now(),
            log_file=str(log_file),
            command=cmd,
        )
    except Exception as err:
        if proc and proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        return LaunchResult(
            task_id=task_id,
            status="spawn_error",
            return_code=None,
            elapsed_sec=time.perf_counter() - start_ts,
            start_utc=start_utc,
            end_utc=utc_now(),
            log_file=str(log_file),
            command=cmd,
            error=str(err),
        )


async def progress_reporter(
    stats: SharedStats,
    lock: asyncio.Lock,
    *,
    start_monotonic: float,
    every_sec: float,
) -> None:
    while True:
        await asyncio.sleep(every_sec)
        async with lock:
            finished = stats.finished
            submitted = stats.submitted
            running = max(0, stats.started - stats.finished)
            success = stats.success
            failed = stats.failed
            timeout = stats.timeout
            spawn_error = stats.spawn_error
        elapsed = max(0.0, time.perf_counter() - start_monotonic)
        rps = finished / elapsed if elapsed > 0 else 0.0
        print(
            f"[Progress] finished={finished}/{submitted} running={running} "
            f"ok={success} fail={failed} timeout={timeout} spawn_err={spawn_error} "
            f"elapsed={elapsed:.1f}s done_rate={rps:.2f}/s"
        )
        if finished >= submitted:
            return


async def launch_batch(args: argparse.Namespace) -> dict[str, Any]:
    batch_start_utc = utc_now()
    python_bin = args.python_bin or detect_python_bin()
    agent_script = Path(args.agent_script).resolve()
    output_root = Path(args.output_root).resolve()

    if not agent_script.exists():
        raise FileNotFoundError(f"Agent script not found: {agent_script}")

    launch_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = output_root / "batch_launches" / launch_tag
    logs_dir = batch_dir / "task_logs"
    batch_dir.mkdir(parents=True, exist_ok=True)

    service_proc: subprocess.Popen[str] | None = None
    service_started_by_launcher = False
    ensure_service_enabled = bool(args.ensure_service and (not args.dry_run))
    if ensure_service_enabled:
        service_python_bin = args.service_python_bin or detect_service_python_bin(python_bin)
        service_proc, service_started_by_launcher = start_service_if_needed(
            enabled=True,
            web_sim_dir=Path(args.web_sim_dir).resolve(),
            port=args.service_port,
            timeout_sec=args.service_timeout_sec,
            log_file=batch_dir / "web_service.log",
            python_bin=service_python_bin,
        )

    task_ids = list(range(args.start_index, args.start_index + args.total_tasks))
    queue: asyncio.Queue[int | None] = asyncio.Queue()
    for tid in task_ids:
        queue.put_nowait(tid)

    results: list[LaunchResult] = []
    stats = SharedStats(submitted=len(task_ids))
    lock = asyncio.Lock()
    limiter = StartRateLimiter(args.launch_rate)

    async def worker(worker_idx: int) -> None:
        while True:
            tid = await queue.get()
            if tid is None:
                queue.task_done()
                return

            cmd = build_cmd(
                python_bin=python_bin,
                agent_script=agent_script,
                output_root=output_root,
                headless=args.headless,
                track_override=args.track_override,
                decision_model=args.decision_model,
                qwen_base_url=args.qwen_base_url,
            )

            env = os.environ.copy()
            env["PROFILE_SELECTION_MODE"] = args.profile_selection_mode
            if args.profile_seed_base is not None:
                env["PROFILE_SELECTION_SEED"] = str(args.profile_seed_base + int(tid))
            env["PYTHONUNBUFFERED"] = "1"

            async with lock:
                stats.started += 1

            if args.dry_run:
                result = LaunchResult(
                    task_id=int(tid),
                    status="success",
                    return_code=0,
                    elapsed_sec=0.0,
                    start_utc=utc_now(),
                    end_utc=utc_now(),
                    log_file=str(logs_dir / f"task_{tid:06d}.log"),
                    command=cmd,
                )
            else:
                result = await run_one_task(
                    task_id=int(tid),
                    worker_idx=worker_idx,
                    cmd=cmd,
                    env=env,
                    log_file=logs_dir / f"task_{tid:06d}.log",
                    timeout_sec=args.task_timeout_sec,
                    limiter=limiter,
                    print_task_pid=args.print_task_pid,
                )

            async with lock:
                results.append(result)
                stats.finished += 1
                if result.status == "success":
                    stats.success += 1
                elif result.status == "timeout":
                    stats.timeout += 1
                elif result.status == "spawn_error":
                    stats.spawn_error += 1
                else:
                    stats.failed += 1

            queue.task_done()

    start_t = time.perf_counter()
    workers = [
        asyncio.create_task(worker(i + 1)) for i in range(max(1, args.max_concurrency))
    ]
    reporter = asyncio.create_task(
        progress_reporter(
            stats,
            lock,
            start_monotonic=start_t,
            every_sec=max(1.0, args.progress_every_sec),
        )
    )

    try:
        await queue.join()
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers)
    finally:
        await reporter
        if args.stop_service_when_done and service_started_by_launcher:
            stop_service(service_proc)

    elapsed = time.perf_counter() - start_t
    results_sorted = sorted(results, key=lambda x: x.task_id)
    failed_tasks = [x.task_id for x in results_sorted if x.status != "success"]
    summary = {
        "launch_tag": launch_tag,
        "start_utc": batch_start_utc,
        "end_utc": utc_now(),
        "elapsed_sec": elapsed,
        "config": {
            "total_tasks": args.total_tasks,
            "start_index": args.start_index,
            "max_concurrency": args.max_concurrency,
            "launch_rate": args.launch_rate,
            "task_timeout_sec": args.task_timeout_sec,
            "headless": args.headless,
            "track_override": args.track_override,
            "decision_model": args.decision_model,
            "profile_selection_mode": args.profile_selection_mode,
            "profile_seed_base": args.profile_seed_base,
            "print_task_pid": args.print_task_pid,
            "ensure_service": args.ensure_service,
            "service_port": args.service_port,
            "stop_service_when_done": args.stop_service_when_done,
            "dry_run": args.dry_run,
            "agent_script": str(agent_script),
            "python_bin": python_bin,
            "output_root": str(output_root),
        },
        "stats": {
            "submitted": stats.submitted,
            "started": stats.started,
            "finished": stats.finished,
            "success": stats.success,
            "failed": stats.failed,
            "timeout": stats.timeout,
            "spawn_error": stats.spawn_error,
            "throughput_tasks_per_sec": (stats.finished / elapsed) if elapsed > 0 else 0.0,
        },
        "batch_dir": str(batch_dir),
        "failed_task_count": len(failed_tasks),
        "failed_tasks": failed_tasks[:5000],
    }

    (batch_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (batch_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(asdict(x), ensure_ascii=False) for x in results_sorted) + "\n",
        encoding="utf-8",
    )
    if failed_tasks:
        (batch_dir / "failed_tasks.txt").write_text(
            "\n".join(str(x) for x in failed_tasks) + "\n",
            encoding="utf-8",
        )

    print("[Done] Batch launch completed.")
    print(f"- batch_dir: {batch_dir}")
    print(
        "- stats: "
        f"submitted={stats.submitted}, success={stats.success}, failed={stats.failed}, "
        f"timeout={stats.timeout}, spawn_error={stats.spawn_error}, elapsed={elapsed:.1f}s"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch many web_sim_camel_agent tasks")
    parser.add_argument("--total-tasks", type=int, default=10000)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=120)
    parser.add_argument(
        "--launch-rate",
        type=float,
        default=15.0,
        help="Max process starts per second. 0 means no start-rate limit.",
    )
    parser.add_argument("--task-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--progress-every-sec", type=float, default=10.0)

    parser.add_argument("--python-bin", type=str, default="")
    parser.add_argument("--agent-script", type=Path, default=DEFAULT_AGENT_SCRIPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--track-override", type=int, default=None)
    parser.add_argument("--decision-model", type=str, default=None)
    parser.add_argument("--qwen-base-url", type=str, default=None)

    parser.add_argument(
        "--profile-selection-mode",
        type=str,
        default="random",
        choices=["random", "agent_num", "deterministic", "round_robin"],
    )
    parser.add_argument(
        "--profile-seed-base",
        type=int,
        default=None,
        help="If set, each task uses PROFILE_SELECTION_SEED=base+task_id.",
    )

    parser.add_argument("--ensure-service", dest="ensure_service", action="store_true")
    parser.add_argument("--no-ensure-service", dest="ensure_service", action="store_false")
    parser.add_argument("--web-sim-dir", type=Path, default=DEFAULT_WEB_SIM_DIR)
    parser.add_argument("--service-port", type=int, default=19002)
    parser.add_argument("--service-timeout-sec", type=float, default=40.0)
    parser.add_argument("--service-python-bin", type=str, default="")
    parser.add_argument("--stop-service-when-done", action="store_true")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-task-pid", dest="print_task_pid", action="store_true")
    parser.add_argument("--no-print-task-pid", dest="print_task_pid", action="store_false")
    parser.set_defaults(headless=True, ensure_service=True)
    parser.set_defaults(print_task_pid=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.total_tasks <= 0:
        raise ValueError("--total-tasks must be > 0")
    if args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be > 0")
    if args.launch_rate < 0:
        raise ValueError("--launch-rate must be >= 0")
    if args.task_timeout_sec <= 0:
        raise ValueError("--task-timeout-sec must be > 0")


def main() -> None:
    args = parse_args()
    validate_args(args)
    try:
        asyncio.run(launch_batch(args))
    except KeyboardInterrupt:
        print("\n[Abort] Interrupted by user.")


if __name__ == "__main__":
    main()
