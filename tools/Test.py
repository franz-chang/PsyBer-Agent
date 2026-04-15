#!/usr/bin/env python3
"""Benchmark local Ollama response latency by asking the same question repeatedly."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask one question to local Ollama repeatedly and measure latency."
    )
    parser.add_argument(
        "--model",
        default="qwen3.5:35b",
        help="Ollama model name, default: qwen3.5:35b",
    )
    parser.add_argument(
        "--question",
        default="请用一句话介绍你自己。",
        help="Question/prompt sent every round.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="How many times to ask the same question, default: 100",
    )
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
        help="Ollama host, default: http://127.0.0.1:11434",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds, default: 300",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ollama_latency_results.json"),
        help="Output JSON file path.",
    )
    return parser.parse_args()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    if low == high:
        return values[low]
    weight = rank - low
    return values[low] * (1.0 - weight) + values[high] * weight


def call_ollama(
    host: str,
    model: str,
    question: str,
    timeout: float,
) -> tuple[bool, float, str, str]:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": question,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
        elapsed = time.perf_counter() - start
        data: dict[str, Any] = json.loads(raw)
        answer = str(data.get("response", "")).strip()
        return True, elapsed, answer, ""
    except HTTPError as err:
        elapsed = time.perf_counter() - start
        return False, elapsed, "", f"HTTPError {err.code}: {err.reason}"
    except URLError as err:
        elapsed = time.perf_counter() - start
        return False, elapsed, "", f"URLError: {err.reason}"
    except Exception as err:
        elapsed = time.perf_counter() - start
        return False, elapsed, "", f"Exception: {err}"


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be > 0")

    records: list[dict[str, Any]] = []
    print(
        f"Start benchmark: model={args.model}, count={args.count}, host={args.host}"
    )
    print(f"Question: {args.question}")

    for i in range(1, args.count + 1):
        ok, elapsed, answer, error = call_ollama(
            host=args.host,
            model=args.model,
            question=args.question,
            timeout=args.timeout,
        )
        record = {
            "round": i,
            "ok": ok,
            "elapsed_sec": round(elapsed, 6),
            "answer": answer,
            "error": error,
        }
        records.append(record)
        status = "OK" if ok else "FAIL"
        print(f"[{i:03d}/{args.count}] {status} {elapsed:.3f}s")

    success_times = sorted([r["elapsed_sec"] for r in records if r["ok"]])
    fail_count = len(records) - len(success_times)

    summary: dict[str, Any] = {
        "model": args.model,
        "host": args.host,
        "question": args.question,
        "count": args.count,
        "success_count": len(success_times),
        "fail_count": fail_count,
    }
    if success_times:
        summary.update(
            {
                "avg_sec": round(statistics.mean(success_times), 6),
                "min_sec": round(min(success_times), 6),
                "max_sec": round(max(success_times), 6),
                "p50_sec": round(percentile(success_times, 0.50), 6),
                "p90_sec": round(percentile(success_times, 0.90), 6),
                "p95_sec": round(percentile(success_times, 0.95), 6),
            }
        )

    output_payload = {
        "summary": summary,
        "records": records,
    }
    args.output.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSummary:")
    for k, v in summary.items():
        print(f"- {k}: {v}")
    print(f"\nSaved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
