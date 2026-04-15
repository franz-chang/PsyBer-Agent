#!/usr/bin/env python3
"""Minimal Qwen API smoke test (OpenAI-compatible endpoint).

Usage:
  python test_qwen.py
  python test_qwen.py --prompt "Say hello in Chinese."
  QWEN_API_KEY=your_key python test_qwen.py --model qwen-plus
"""

from __future__ import annotations

import argparse
import os
import sys

from openai import OpenAI


# TODO: Put your Qwen API key here if you do not want to use environment vars.
# Priority:
# 1) --api-key argument
# 2) QWEN_API_KEY env
# 3) DASHSCOPE_API_KEY env
# 4) INLINE_QWEN_API_KEY below
INLINE_QWEN_API_KEY = "sk-169820d28e16466a8e58c52b8822f130"

# You can also change these defaults directly here.
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


def resolve_api_key(cli_api_key: str) -> str:
    if cli_api_key.strip():
        return cli_api_key.strip()
    if os.getenv("QWEN_API_KEY", "").strip():
        return os.environ["QWEN_API_KEY"].strip()
    if os.getenv("DASHSCOPE_API_KEY", "").strip():
        return os.environ["DASHSCOPE_API_KEY"].strip()
    return INLINE_QWEN_API_KEY.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple Qwen API test script")
    parser.add_argument("--api-key", type=str, default="", help="Qwen API key")
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("QWEN_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("QWEN_MODEL", DEFAULT_MODEL),
        help="Model name",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Please reply with exactly: hello qwen",
        help="User prompt",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print full raw response object",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = resolve_api_key(args.api_key)
    if not api_key:
        print(
            "[ERROR] Missing API key.\n"
            "Please set one of:\n"
            "1) --api-key\n"
            "2) QWEN_API_KEY env\n"
            "3) DASHSCOPE_API_KEY env\n"
            "4) INLINE_QWEN_API_KEY in this file",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": args.prompt},
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if args.raw:
        print(resp)
        return

    content = ""
    if resp.choices and resp.choices[0].message:
        content = resp.choices[0].message.content or ""

    print("=== Qwen API Test ===")
    print(f"base_url: {args.base_url}")
    print(f"model: {args.model}")
    print("--- response ---")
    print(content.strip())


if __name__ == "__main__":
    main()
