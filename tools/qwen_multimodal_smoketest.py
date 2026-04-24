#!/usr/bin/env python3
"""Qwen 多模态连通性测试脚本（图片 + 问题）。

示例:
  python tools/qwen_multimodal_smoketest.py \
    --image /abs/path/demo.png \
    --question "这张图里有什么？"
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_API_POOL_FILE = PROJECT_DIR / "settings" / "api_pool.yaml"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-vl-plus"


@dataclass(frozen=True)
class ApiSlot:
    slot_id: str
    api_key: str
    base_url: str | None = None
    model: str | None = None


def _str_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _mask_key(api_key: str) -> str:
    key = api_key.strip()
    if len(key) <= 12:
        return key[:2] + "***" + key[-2:] if len(key) >= 4 else "***"
    return f"{key[:6]}...{key[-4:]}"


def load_api_pool(api_pool_file: Path) -> list[ApiSlot]:
    if not api_pool_file.exists():
        raise FileNotFoundError(f"API pool 文件不存在: {api_pool_file}")

    raw = yaml.safe_load(api_pool_file.read_text(encoding="utf-8"))
    defaults: dict[str, Any] = {}
    entries: list[Any] = []

    if raw is None:
        return []
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        defaults_raw = raw.get("defaults", {})
        if isinstance(defaults_raw, dict):
            defaults = defaults_raw
        candidates = raw.get("apis")
        if candidates is None:
            candidates = raw.get("api_pool")
        if candidates is None:
            candidates = []
        if not isinstance(candidates, list):
            raise ValueError(
                f"API pool 格式错误: `apis` 必须是 list, file={api_pool_file}"
            )
        entries = candidates
    else:
        raise ValueError(
            f"API pool 格式错误: 顶层需为 dict/list, 实际是 {type(raw).__name__}"
        )

    default_base_url = _str_or_empty(
        defaults.get("base_url") or defaults.get("qwen_base_url")
    ) or DEFAULT_BASE_URL
    default_model = _str_or_empty(
        defaults.get("qwen_model") or defaults.get("decision_model")
    )

    slots: list[ApiSlot] = []
    for idx, item in enumerate(entries, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"API pool entry #{idx} 格式错误: 需为 dict, 实际是 {type(item).__name__}"
            )
        if not _is_enabled(item.get("enabled", True)):
            continue

        api_key = _str_or_empty(
            item.get("api_key") or item.get("qwen_api_key") or item.get("key")
        )
        if not api_key:
            raise ValueError(f"API pool entry #{idx} 缺少 `api_key`")

        slot_id = _str_or_empty(item.get("id") or item.get("api_id") or f"api-{idx:03d}")
        base_url = _str_or_empty(
            item.get("base_url") or item.get("qwen_base_url") or default_base_url
        )
        model = _str_or_empty(
            item.get("qwen_model") or item.get("decision_model") or default_model
        )
        slots.append(
            ApiSlot(
                slot_id=slot_id,
                api_key=api_key,
                base_url=base_url or None,
                model=model or None,
            )
        )
    return slots


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    payload = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{payload}"


def extract_output_text(response_obj: Any) -> str:
    text = getattr(response_obj, "output_text", "") or ""
    if text:
        return text
    if hasattr(response_obj, "model_dump"):
        data = response_obj.model_dump()
        output = data.get("output", [])
        if isinstance(output, list):
            chunks: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content", [])
                if not isinstance(content, list):
                    continue
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    if c.get("type") in {"output_text", "text"}:
                        txt = c.get("text")
                        if isinstance(txt, str) and txt.strip():
                            chunks.append(txt)
            if chunks:
                return "\n".join(chunks)
        return json.dumps(data, ensure_ascii=False)
    return str(response_obj)


def extract_response_error(response_obj: Any) -> tuple[str, str]:
    status = _str_or_empty(getattr(response_obj, "status", ""))

    error_obj = getattr(response_obj, "error", None)
    if isinstance(error_obj, dict):
        msg = _str_or_empty(error_obj.get("message") or error_obj.get("code"))
        if msg:
            return status, msg

    if hasattr(response_obj, "model_dump"):
        data = response_obj.model_dump()
        err = data.get("error")
        if isinstance(err, dict):
            msg = _str_or_empty(err.get("message") or err.get("code"))
            if msg:
                return status, msg

    return status, ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen 多模态连通性测试（图像+问题）")
    parser.add_argument("--image", required=True, help="本地图像路径")
    parser.add_argument("--question", required=True, help="要问模型的问题")
    parser.add_argument(
        "--api-pool-file",
        default=str(DEFAULT_API_POOL_FILE),
        help=f"API 池配置文件路径 (默认: {DEFAULT_API_POOL_FILE})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="模型名，默认优先取 API 池里的 qwen_model，若无则用 qwen-vl-plus",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="覆盖 base_url；默认优先取 API 池配置",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子（用于复现实验）")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度")
    parser.add_argument("--max-output-tokens", type=int, default=256, help="最大输出 token")
    parser.add_argument("--timeout", type=float, default=60.0, help="请求超时秒数")
    parser.add_argument(
        "--show-full-response",
        action="store_true",
        help="打印完整结构化响应（较长）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        print(f"[ERROR] 图像不存在: {image_path}")
        return 2

    api_pool_file = Path(args.api_pool_file).expanduser().resolve()
    slots = load_api_pool(api_pool_file)
    if not slots:
        print(f"[ERROR] API 池没有可用的 enabled 条目: {api_pool_file}")
        return 2

    rng = random.Random(args.seed)
    selected = rng.choice(slots)
    model = _str_or_empty(args.model) or selected.model or DEFAULT_MODEL
    base_url = _str_or_empty(args.base_url) or selected.base_url or DEFAULT_BASE_URL
    question = args.question.strip()
    if not question:
        print("[ERROR] --question 不能为空")
        return 2

    image_data_url = image_to_data_url(image_path)
    client = OpenAI(api_key=selected.api_key, base_url=base_url, timeout=args.timeout)

    print("[Qwen MM SmokeTest] 配置如下:")
    print(f"- api_pool: {api_pool_file}")
    print(f"- api_slot: {selected.slot_id}")
    print(f"- api_key: {_mask_key(selected.api_key)}")
    print(f"- base_url: {base_url}")
    print(f"- model: {model}")
    print(f"- image: {image_path}")
    print(f"- question: {question}")

    started = time.perf_counter()
    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": question},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }
            ],
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
    except Exception as err:
        elapsed = time.perf_counter() - started
        print(f"[FAIL] 调用失败，耗时 {elapsed:.2f}s")
        print(f"[FAIL] 错误: {err}")
        return 1

    elapsed = time.perf_counter() - started
    status, error_message = extract_response_error(response)
    if error_message or (status and status.lower() in {"failed", "incomplete", "cancelled"}):
        print(f"[FAIL] 调用返回失败状态，耗时 {elapsed:.2f}s")
        if status:
            print(f"- status: {status}")
        if error_message:
            print(f"- error: {error_message}")
        if args.show_full_response:
            print("- full_response:")
            if hasattr(response, "model_dump_json"):
                print(response.model_dump_json(indent=2, ensure_ascii=False))
            elif hasattr(response, "model_dump"):
                print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
            else:
                print(str(response))
        return 1

    answer_text = extract_output_text(response).strip()
    response_id = getattr(response, "id", "")
    print(f"[OK] 调用成功，耗时 {elapsed:.2f}s")
    if status:
        print(f"- status: {status}")
    if response_id:
        print(f"- response_id: {response_id}")
    print("- answer:")
    print(answer_text if answer_text else "(empty)")

    if args.show_full_response:
        print("- full_response:")
        if hasattr(response, "model_dump_json"):
            print(response.model_dump_json(indent=2, ensure_ascii=False))
        elif hasattr(response, "model_dump"):
            print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
        else:
            print(str(response))

    return 0


if __name__ == "__main__":
    sys.exit(main())
