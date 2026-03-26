#!/usr/bin/env python3
"""Web_sim click simulator based on CAMEL HybridBrowserToolkit.
Workflow:
1) Read task config from /Users/chongzhang/PsyBer-Agent/task.yaml.
2) Ensure Web_sim service is running at http://127.0.0.1:19002/.
3) Use CAMEL Browser Toolkit to select dataset/model, click reset, then
   repeatedly click one of four items or go to next page.
4) Record click sequence into agent memory until track target is reached.
"""

from __future__ import annotations
import argparse
import asyncio
import base64
import json
import os
import re
import shutil
import hashlib
import importlib.util
import signal
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlopen

from openai import OpenAI
import yaml
import numpy as np

CAMEL_ROOT = Path("/Users/chongzhang/camel")
if str(CAMEL_ROOT) not in sys.path:
    sys.path.insert(0, str(CAMEL_ROOT))

warnings.filterwarnings(
    "ignore",
    message="urllib3 .* doesn't match a supported version!",
    category=Warning,
)

from camel.toolkits import HybridBrowserToolkit  # noqa: E402


WEB_SIM_DIR = Path("/Users/chongzhang/Web_sim")
PROJECT_DIR = Path(__file__).resolve().parent
WEB_SIM_TASK_FILE = PROJECT_DIR / "task.yaml"
LEGACY_WEB_SIM_TASK_FILE = WEB_SIM_DIR / "task.yaml"
SERVICE_URL = "http://127.0.0.1:19002"
RUN_SERVICE_SCRIPT = WEB_SIM_DIR / "run_service.sh"
ML1M_EMBEDDINGS_FILE = Path(
    "/Users/chongzhang/WebSim_Dataset/MM-ML-1M-main/Embeddings/item_embeddings.json"
)
ML1M_PROFILES_FILE = Path(
    "/Users/chongzhang/WebSim_Dataset/MM-ML-1M-main/Embeddings/agent_profiles.json"
)
AMAZON_MI_EMBEDDINGS_FILE = Path(
    "/Users/chongzhang/WebSim_Dataset/amazon_v2/Musical_Instruments/Embeddings/item_embeddings.json"
)
AMAZON_MI_PROFILES_FILE = Path(
    "/Users/chongzhang/WebSim_Dataset/amazon_v2/Musical_Instruments/Embeddings/agent_profiles.json"
)
PSY_ENGINE_FILE = PROJECT_DIR / "Psy-engine.py"
# Default to Qwen official OpenAI-compatible endpoint.
# You can override by env:
# - QWEN_API_KEY / DASHSCOPE_API_KEY
# - QWEN_BASE_URL
# - QWEN_MODEL
QWEN_MODEL_DEFAULT = "qwen3-vl-flash"
QWEN_BASE_URL_DEFAULT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# Fill your key here if you don't want to use environment variables.
# Example: QWEN_API_KEY = "sk-xxxx"
QWEN_API_KEY = "sk-fc7990c58df346f6909e29ba6a3daf1a"

DATASET_KEY_MAP = {
    "ml1m": "ml1m",
    "movielens-1m": "ml1m",
    "movielens": "ml1m",
    "movie lens-1m": "ml1m",
    "movie lens": "ml1m",
    "amazon_mi": "amazon_mi",
    "amazon musical instruments": "amazon_mi",
    "amazon musical_instruments": "amazon_mi",
    "musical instruments": "amazon_mi",
    "musical_instruments": "amazon_mi",
}

MODEL_KEY_MAP = {
    "sasrec": "sasrec",
    "lightgcn": "lightgcn",
    "light gcn": "lightgcn",
    "light-gcn": "lightgcn",
    "multvae": "multvae",
    "multi-vae": "multvae",
    "mult-vae": "multvae",
}


@dataclass
class TaskConfig:
    dataset_candidates: list[str]
    model_candidates: list[str]
    track: int
    agent_num: int
    raw: dict[str, Any]


@dataclass
class ClickRecord:
    order: int
    item_id: str
    title: str
    dataset_key: str
    model_key: str
    status_text: str
    timestamp_utc: str


@dataclass
class SkipRecord:
    reason: str
    status_text: str
    timestamp_utc: str


@dataclass
class AgentMemory:
    task_config: dict[str, Any]
    selected_dataset_key: str
    selected_model_key: str
    track_target: int
    agent_num: int
    agent_profile: dict[str, Any] = field(default_factory=dict)
    psy_state_history: list[dict[str, Any]] = field(default_factory=list)
    click_sequence: list[ClickRecord] = field(default_factory=list)
    skip_sequence: list[SkipRecord] = field(default_factory=list)

    def append_click(self, record: ClickRecord) -> None:
        self.click_sequence.append(record)

    def append_skip(self, record: SkipRecord) -> None:
        self.skip_sequence.append(record)

    def append_psy_state(self, record: dict[str, Any]) -> None:
        self.psy_state_history.append(record)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_config": self.task_config,
            "selected_dataset_key": self.selected_dataset_key,
            "selected_model_key": self.selected_model_key,
            "track_target": self.track_target,
            "agent_num": self.agent_num,
            "agent_profile": self.agent_profile,
            "click_count": len(self.click_sequence),
            "skip_count": len(self.skip_sequence),
            "psy_state_count": len(self.psy_state_history),
            "click_sequence": [asdict(x) for x in self.click_sequence],
            "skip_sequence": [asdict(x) for x in self.skip_sequence],
            "psy_state_history": self.psy_state_history,
        }


class WebSimCamelAgent:
    def __init__(
        self,
        output_root: Path,
        *,
        headless: bool,
        skip_probability: float,
        seed: int,
        close_service_when_done: bool,
        track_override: int | None,
        decision_model: str | None,
        qwen_base_url: str | None,
    ):
        self.output_root = output_root
        self.headless = headless
        # Deprecated legacy args are accepted for compatibility only.
        _ = skip_probability
        _ = seed
        self.close_service_when_done = close_service_when_done
        self.track_override = track_override if (track_override or 0) > 0 else None
        self.decision_model = (
            (decision_model or "").strip()
            or os.getenv("QWEN_MODEL", "").strip()
            or QWEN_MODEL_DEFAULT
        )
        self.qwen_base_url = (
            (qwen_base_url or "").strip()
            or os.getenv("QWEN_BASE_URL", "").strip()
            or QWEN_BASE_URL_DEFAULT
        )

        self.service_proc: subprocess.Popen[str] | None = None
        self.service_started_by_agent = False

        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_root / "runs" / run_tag
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.memory_file = self.run_dir / "memory.json"
        self.summary_file = self.run_dir / "summary.txt"
        self.service_log_file = self.run_dir / "web_service.log"
        self.openai_client: OpenAI | None = None
        self.item_feature_map: dict[str, dict[str, Any]] = {}
        self.agent_profiles: list[dict[str, Any]] = []
        self.psy_engine: Any = None

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalize_to_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    out.append(text)
            return out
        text = str(value).strip()
        return [text] if text else []

    @staticmethod
    def _parse_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
            if parsed <= 0:
                return default
            return parsed
        except Exception:
            return default

    @staticmethod
    def _norm_key(text: str) -> str:
        return " ".join(str(text).strip().lower().replace("_", " ").split())

    def _dataset_to_key(self, value: str) -> str | None:
        key = self._norm_key(value)
        return DATASET_KEY_MAP.get(key)

    def _model_to_key(self, value: str) -> str | None:
        key = self._norm_key(value)
        return MODEL_KEY_MAP.get(key)

    def ensure_task_file(self) -> None:
        if WEB_SIM_TASK_FILE.exists():
            return
        if LEGACY_WEB_SIM_TASK_FILE.exists():
            shutil.copyfile(LEGACY_WEB_SIM_TASK_FILE, WEB_SIM_TASK_FILE)
            return
        default_config = {
            "dataset": ["MovieLens-1M", "Amazon Musical Instruments"],
            "model": ["SASRec", "LIghtGCN", "Multi-VAE"],
            "track": 100,
            "agent_num": 100,
        }
        WEB_SIM_TASK_FILE.write_text(
            yaml.safe_dump(default_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def load_task_config(self) -> TaskConfig:
        self.ensure_task_file()
        raw = yaml.safe_load(WEB_SIM_TASK_FILE.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid task config format: {WEB_SIM_TASK_FILE}")

        dataset_value = (
            raw.get("dataset")
            or raw.get("datasets")
            or raw.get("Dataset")
            or raw.get("DATASET")
        )
        model_value = (
            raw.get("model")
            or raw.get("models")
            or raw.get("Model")
            or raw.get("MODEL")
        )
        track_value = raw.get("track") or raw.get("Track") or raw.get("TRACK")
        agent_num_value = (
            raw.get("agent_num")
            or raw.get("agentNum")
            or raw.get("Agent_num")
            or raw.get("AGENT_NUM")
        )

        datasets = self._normalize_to_list(dataset_value)
        models = self._normalize_to_list(model_value)

        if not datasets:
            datasets = ["MovieLens-1M", "Amazon Musical Instruments"]
        if not models:
            models = ["SASRec", "LIghtGCN", "Multi-VAE"]

        track = self._parse_int(track_value, 100)
        agent_num = self._parse_int(agent_num_value, 100)

        return TaskConfig(
            dataset_candidates=datasets,
            model_candidates=models,
            track=track,
            agent_num=agent_num,
            raw=raw,
        )

    def _service_is_alive(self) -> bool:
        try:
            with urlopen(f"{SERVICE_URL}/health", timeout=2) as resp:
                return int(resp.status) == 200
        except Exception:
            return False

    def ensure_service(self, timeout_sec: int = 40) -> None:
        if self._service_is_alive():
            return
        if not RUN_SERVICE_SCRIPT.exists():
            raise FileNotFoundError(f"Missing script: {RUN_SERVICE_SCRIPT}")

        log_fp = self.service_log_file.open("w", encoding="utf-8")
        self.service_proc = subprocess.Popen(
            ["bash", str(RUN_SERVICE_SCRIPT)],
            cwd=str(WEB_SIM_DIR),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self.service_started_by_agent = True

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self._service_is_alive():
                return
            time.sleep(0.5)

        raise RuntimeError(
            "Web_sim service did not become ready in time. "
            f"Check log: {self.service_log_file}"
        )

    def shutdown_service_if_needed(self) -> None:
        if not self.close_service_when_done:
            return
        if not self.service_started_by_agent or self.service_proc is None:
            return

        try:
            self.service_proc.terminate()
            self.service_proc.wait(timeout=8)
        except Exception:
            try:
                if self.service_proc.pid:
                    os.killpg(self.service_proc.pid, signal.SIGKILL)
            except Exception:
                pass

    @staticmethod
    def _parse_console_result(result_text: str) -> Any:
        prefix = "Code execution result:"
        payload = result_text.strip()
        if payload.startswith(prefix):
            payload = payload[len(prefix) :].strip()
        if not payload:
            return None

        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload

    async def _get_state(self, toolkit: HybridBrowserToolkit) -> dict[str, Any]:
        js = r"""
(() => {
  const dataset = document.getElementById('datasetSelect');
  const model = document.getElementById('algoSelect');
  const resetBtn = document.getElementById('resetBtn');
  const nextBtn = document.getElementById('nextBtn');
  const status = document.getElementById('statusText');
  const cards = Array.from(document.querySelectorAll('.card')).map((el, idx) => ({
    idx,
    ref: el.getAttribute('aria-ref') || '',
    item_id: el.getAttribute('data-movie-id') || '',
    title: (el.querySelector('.title')?.textContent || '').trim(),
  }));

  return {
    dataset_ref: dataset?.getAttribute('aria-ref') || '',
    model_ref: model?.getAttribute('aria-ref') || '',
    reset_ref: resetBtn?.getAttribute('aria-ref') || '',
    next_ref: nextBtn?.getAttribute('aria-ref') || '',
    next_disabled: Boolean(nextBtn?.disabled),
    dataset_value: dataset?.value || '',
    model_value: model?.value || '',
    available_dataset_values: Array.from(dataset?.options || []).filter(x => !x.disabled).map(x => x.value),
    available_model_values: Array.from(model?.options || []).filter(x => !x.disabled).map(x => x.value),
    status_text: (status?.textContent || '').trim(),
    cards,
  };
})()
"""
        result = await toolkit.browser_console_exec(js)
        parsed = self._parse_console_result(result.get("result", ""))
        if not isinstance(parsed, dict):
            raise RuntimeError(
                f"Failed to parse browser state from console result: {result}"
            )
        return parsed

    async def _capture_som_screenshot_path(
        self, toolkit: HybridBrowserToolkit
    ) -> str | None:
        await toolkit.browser_get_som_screenshot(read_image=False)
        cache_dir = self.run_dir / "browser_cache"
        files = sorted(
            cache_dir.glob("*_som.png"),
            key=lambda p: p.stat().st_mtime,
        )
        if not files:
            return None
        return str(files[-1])

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        raw = (text or "").strip()
        if not raw:
            return None
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    @staticmethod
    def _stable_float_01(text: str) -> float:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        raw = int(digest[:8], 16)
        return raw / float(0xFFFFFFFF)

    @staticmethod
    def _deterministic_vector(item_id: str, dim: int) -> list[float]:
        seed_int = int(hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed_int)
        vec = rng.normal(0, 1, dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec = vec / norm
        return [float(x) for x in vec.tolist()]

    def _get_embedding_paths(self, dataset_key: str) -> tuple[Path, Path]:
        if dataset_key == "amazon_mi":
            return AMAZON_MI_EMBEDDINGS_FILE, AMAZON_MI_PROFILES_FILE
        return ML1M_EMBEDDINGS_FILE, ML1M_PROFILES_FILE

    def _load_psy_resources(self, dataset_key: str) -> None:
        emb_file, profile_file = self._get_embedding_paths(dataset_key)
        self.item_feature_map = {}
        self.agent_profiles = []

        if emb_file.exists():
            raw = json.loads(emb_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for item_id, payload in raw.items():
                    if not isinstance(payload, dict):
                        continue
                    psy_input = payload.get("psy_input")
                    if isinstance(psy_input, dict):
                        self.item_feature_map[str(item_id)] = psy_input
                        continue

                    # Backward compatibility: construct from legacy embeddings
                    text_vec = payload.get("text_embedding")
                    visual_vec = payload.get("visual_embedding")
                    if isinstance(text_vec, list) and isinstance(visual_vec, list) and text_vec and visual_vec:
                        text_arr = np.array(text_vec, dtype=np.float32)
                        visual_arr = np.array(visual_vec, dtype=np.float32)
                        fused = ((text_arr + visual_arr) / 2.0).astype(np.float32)
                        fused_norm = float(np.linalg.norm(fused))
                        if fused_norm > 1e-12:
                            fused = fused / fused_norm
                        cos = float(np.dot(text_arr, visual_arr) / max(float(np.linalg.norm(text_arr) * np.linalg.norm(visual_arr)), 1e-12))
                        self.item_feature_map[str(item_id)] = {
                            "item_v": [float(x) for x in fused.tolist()],
                            "item_n": max(0.0, min(1.0, (1.0 - cos) / 2.0)),
                            "item_c": 0.5,
                            "item_pop": self._stable_float_01(str(item_id)),
                        }

        if profile_file.exists():
            raw_profiles = json.loads(profile_file.read_text(encoding="utf-8"))
            if isinstance(raw_profiles, list):
                self.agent_profiles = [x for x in raw_profiles if isinstance(x, dict)]

    def _load_psyber_engine_class(self) -> type:
        if not PSY_ENGINE_FILE.exists():
            raise FileNotFoundError(f"Missing psy engine file: {PSY_ENGINE_FILE}")
        spec = importlib.util.spec_from_file_location("psy_engine_runtime", str(PSY_ENGINE_FILE))
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load PsyBerEngine module spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        engine_cls = getattr(module, "PsyBerEngine", None)
        if engine_cls is None:
            raise RuntimeError("PsyBerEngine class not found in Psy-engine.py")
        return engine_cls

    def _build_initial_engine_memory(self, memory: AgentMemory) -> list[list[float]]:
        history_vectors: list[list[float]] = []
        for rec in memory.click_sequence:
            feat = self.item_feature_map.get(str(rec.item_id))
            if isinstance(feat, dict) and isinstance(feat.get("item_v"), list):
                history_vectors.append([float(x) for x in feat["item_v"]])
        return history_vectors

    def _select_agent_profile(self, agent_num: int) -> dict[str, Any]:
        if not self.agent_profiles:
            return {}
        idx = max(0, agent_num - 1) % len(self.agent_profiles)
        return dict(self.agent_profiles[idx])

    def _build_psy_candidates(
        self,
        cards: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for card in cards:
            item_id = str(card.get("item_id") or "")
            title = str(card.get("title") or "")
            feat = self.item_feature_map.get(item_id, {})
            item_v = feat.get("item_v")
            item_n = feat.get("item_n")
            item_c = feat.get("item_c")
            item_pop = feat.get("item_pop")

            if not isinstance(item_v, list) or not item_v:
                item_v = self._deterministic_vector(item_id or title or "unknown", 512)
            if not isinstance(item_n, (int, float)):
                item_n = self._stable_float_01(f"n::{item_id}")
            if not isinstance(item_c, (int, float)):
                item_c = self._stable_float_01(f"c::{item_id}")
            if not isinstance(item_pop, (int, float)):
                item_pop = self._stable_float_01(f"pop::{item_id}")

            out.append(
                {
                    "item_id": item_id,
                    "title": title,
                    "v": np.array(item_v, dtype=np.float32),
                    "n": float(max(0.0, min(1.0, float(item_n)))),
                    "c": float(max(0.0, min(1.0, float(item_c)))),
                    "pop": float(max(0.0, min(1.0, float(item_pop)))),
                }
            )
        return out

    def _ensure_openai_client(self) -> OpenAI:
        if self.openai_client is not None:
            return self.openai_client
        api_key = (
            os.getenv("QWEN_API_KEY", "").strip()
            or os.getenv("DASHSCOPE_API_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
            or QWEN_API_KEY.strip()
        )
        if not api_key:
            raise RuntimeError(
                "Qwen API key missing. Please set QWEN_API_KEY or DASHSCOPE_API_KEY."
            )
        self.openai_client = OpenAI(api_key=api_key, base_url=self.qwen_base_url)
        return self.openai_client

    @staticmethod
    def _image_to_data_url(image_path: str) -> str:
        payload = Path(image_path).read_bytes()
        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _decide_action_from_agent_reply(
        self,
        reply_text: str,
        cards: list[dict[str, Any]],
        next_enabled: bool,
    ) -> tuple[str, str]:
        parsed = self._extract_json_object(reply_text)
        valid_ids = {str(x.get("item_id") or "") for x in cards}
        valid_ids.discard("")

        if isinstance(parsed, dict):
            action = str(parsed.get("action") or "click").strip().lower()
            item_id = str(parsed.get("item_id") or "").strip()
            if action == "next" and next_enabled:
                return "next", ""
            if action == "click" and item_id in valid_ids:
                return "click", item_id

        first_id = str(cards[0].get("item_id") or "")
        return "click", first_id

    def _build_decision_prompt(
        self,
        *,
        cards: list[dict[str, Any]],
        next_enabled: bool,
        status_text: str,
        memory: AgentMemory,
        psy_state: dict[str, Any],
        psy_engine_choice: dict[str, Any],
    ) -> str:
        card_lines = []
        for idx, card in enumerate(cards, start=1):
            card_lines.append(
                f"{idx}. item_id={card.get('item_id','')} | title={card.get('title','')}"
            )
        recent = memory.click_sequence[-5:]
        recent_lines = [
            f"{x.order}. {x.item_id} | {x.title}" for x in recent
        ] or ["none"]
        return (
            "请根据截图模拟用户点击行为。你必须只输出 JSON，不要输出其它文本。\n"
            "可选动作:\n"
            '1) {"action":"click","item_id":"<候选中的item_id>","reason":"..."}\n'
            '2) {"action":"next","reason":"..."}\n'
            "规则:\n"
            "- 只能在候选 item_id 中选择 click。\n"
            "- 只有当 next_enabled=true 且你对当前 4 项都不感兴趣时才选 next。\n"
            "- 如果 next_enabled=false，必须选 click。\n\n"
            f"dataset={memory.selected_dataset_key}, model={memory.selected_model_key}\n"
            f"next_enabled={str(next_enabled).lower()}\n"
            f"status_text={status_text}\n"
            f"psy_state={psy_state}\n"
            f"psy_engine_choice={psy_engine_choice}\n"
            f"recent_clicks={recent_lines}\n"
            f"candidates={card_lines}\n"
        )

    def _call_openai_decision(
        self,
        *,
        prompt: str,
        screenshot_path: str | None,
    ) -> str:
        client = self._ensure_openai_client()
        text_only_content: list[dict[str, Any]] = [
            {"type": "input_text", "text": prompt}
        ]
        with_image_content = list(text_only_content)
        has_image = bool(screenshot_path and Path(screenshot_path).exists())
        if has_image:
            with_image_content.append(
                {
                    "type": "input_image",
                    "image_url": self._image_to_data_url(screenshot_path),
                }
            )
        try:
            resp = client.responses.create(
                model=self.decision_model,
                input=[{"role": "user", "content": with_image_content}],
                temperature=0,
                max_output_tokens=200,
            )
        except Exception:
            if not has_image:
                raise
            # Some Qwen models are text-only; retry without image.
            resp = client.responses.create(
                model=self.decision_model,
                input=[{"role": "user", "content": text_only_content}],
                temperature=0,
                max_output_tokens=200,
            )
        if isinstance(resp, str):
            return resp
        text = getattr(resp, "output_text", "") or ""
        if text:
            return text
        if hasattr(resp, "model_dump"):
            return json.dumps(resp.model_dump(), ensure_ascii=False)
        return str(resp)

    def _decide_action_with_agent(
        self,
        *,
        screenshot_path: str | None,
        cards: list[dict[str, Any]],
        next_enabled: bool,
        status_text: str,
        memory: AgentMemory,
        psy_state: dict[str, Any],
        psy_engine_choice: dict[str, Any],
    ) -> tuple[str, str, str]:
        prompt = self._build_decision_prompt(
            cards=cards,
            next_enabled=next_enabled,
            status_text=status_text,
            memory=memory,
            psy_state=psy_state,
            psy_engine_choice=psy_engine_choice,
        )
        reply_text = self._call_openai_decision(
            prompt=prompt,
            screenshot_path=screenshot_path,
        )
        action, item_id = self._decide_action_from_agent_reply(
            reply_text, cards, next_enabled
        )
        return action, item_id, reply_text

    @staticmethod
    async def _ensure_snapshot(toolkit: HybridBrowserToolkit) -> None:
        await toolkit.browser_get_page_snapshot()

    async def _click_by_ref(
        self, toolkit: HybridBrowserToolkit, ref: str, action_name: str
    ) -> dict[str, Any]:
        result = await toolkit.browser_click(ref=ref)
        message = str(result.get("result", ""))
        if "Error:" in message:
            raise RuntimeError(f"{action_name} failed: {message}")
        return result

    async def _select_by_ref(
        self, toolkit: HybridBrowserToolkit, ref: str, value: str, action_name: str
    ) -> dict[str, Any]:
        result = await toolkit.browser_select(ref=ref, value=value)
        message = str(result.get("result", ""))
        if "Error:" in message:
            raise RuntimeError(f"{action_name} failed: {message}")
        return result

    async def _prepare_page(
        self, toolkit: HybridBrowserToolkit, dataset_key: str, model_key: str
    ) -> tuple[str, str]:
        await self._ensure_snapshot(toolkit)
        state = await self._get_state(toolkit)

        if not state.get("dataset_ref") or not state.get("model_ref"):
            raise RuntimeError("Cannot find dataset/model controls in page")

        await self._select_by_ref(
            toolkit, state["dataset_ref"], dataset_key, "select dataset"
        )
        await asyncio.sleep(0.3)

        await self._ensure_snapshot(toolkit)
        state = await self._get_state(toolkit)

        available_models = state.get("available_model_values") or []
        final_model_key = model_key
        if final_model_key not in available_models:
            if not available_models:
                raise RuntimeError("No available models in current dataset")
            final_model_key = str(available_models[0])

        await self._select_by_ref(
            toolkit, state["model_ref"], final_model_key, "select model"
        )
        await asyncio.sleep(0.3)

        await self._ensure_snapshot(toolkit)
        state = await self._get_state(toolkit)

        if not state.get("reset_ref"):
            raise RuntimeError("Cannot find reset button")
        await self._click_by_ref(toolkit, state["reset_ref"], "click reset")
        await asyncio.sleep(0.5)

        await self._ensure_snapshot(toolkit)
        state = await self._get_state(toolkit)
        final_dataset_key = str(state.get("dataset_value") or dataset_key)
        final_model_key = str(state.get("model_value") or final_model_key)
        return final_dataset_key, final_model_key

    def _choose_from_candidates(
        self,
        candidates: list[str],
        mapper: Callable[[str], str | None],
        default: str,
    ) -> str:
        keys = [mapper(x) for x in candidates]
        keys = [x for x in keys if x]
        if not keys:
            return default
        return str(keys[0])

    def _persist_memory(self, memory: AgentMemory) -> None:
        self.memory_file.write_text(
            json.dumps(memory.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    async def run_click_simulation(self, task: TaskConfig) -> AgentMemory:
        dataset_key = self._choose_from_candidates(
            task.dataset_candidates, self._dataset_to_key, "ml1m"
        )
        model_key = self._choose_from_candidates(
            task.model_candidates, self._model_to_key, "sasrec"
        )

        toolkit = HybridBrowserToolkit(
            mode="python",
            headless=self.headless,
            session_id=f"websim-{int(time.time())}",
            cache_dir=str(self.run_dir / "browser_cache"),
            default_start_url=f"{SERVICE_URL}/",
            enabled_tools=[
                "browser_open",
                "browser_close",
                "browser_visit_page",
                "browser_get_page_snapshot",
                "browser_get_som_screenshot",
                "browser_click",
                "browser_select",
                "browser_console_exec",
            ],
        )

        memory = AgentMemory(
            task_config=task.raw,
            selected_dataset_key=dataset_key,
            selected_model_key=model_key,
            track_target=task.track,
            agent_num=task.agent_num,
        )

        try:
            await toolkit.browser_open()
            await toolkit.browser_visit_page(f"{SERVICE_URL}/")

            final_dataset_key, final_model_key = await self._prepare_page(
                toolkit, dataset_key, model_key
            )
            memory.selected_dataset_key = final_dataset_key
            memory.selected_model_key = final_model_key

            self._load_psy_resources(final_dataset_key)
            profile = self._select_agent_profile(memory.agent_num)
            memory.agent_profile = profile
            PsyBerEngineCls = self._load_psyber_engine_class()
            init_memory = self._build_initial_engine_memory(memory)
            emb_dim = 512
            if self.item_feature_map:
                first = next(iter(self.item_feature_map.values()))
                if isinstance(first, dict) and isinstance(first.get("item_v"), list) and first["item_v"]:
                    emb_dim = len(first["item_v"])
            self.psy_engine = PsyBerEngineCls(
                emb_dim=emb_dim,
                persona=profile,
                initial_memory=init_memory,
                stochastic=False,
            )
            self._persist_memory(memory)

            decision_step = 0
            while len(memory.click_sequence) < task.track:
                await self._ensure_snapshot(toolkit)
                state = await self._get_state(toolkit)

                cards = [x for x in state.get("cards", []) if x.get("ref")]
                next_ref = str(state.get("next_ref") or "")
                next_disabled = bool(state.get("next_disabled", False))
                next_enabled = bool(next_ref) and (not next_disabled)
                status_text = str(state.get("status_text") or "")

                if not cards:
                    if next_enabled:
                        await self._click_by_ref(toolkit, next_ref, "click next")
                        memory.append_skip(
                            SkipRecord(
                                reason="no_card_on_page_then_next",
                                status_text=status_text,
                                timestamp_utc=self._utc_now(),
                            )
                        )
                        self._persist_memory(memory)
                        await asyncio.sleep(0.5)
                        continue

                    reset_ref = str(state.get("reset_ref") or "")
                    if reset_ref:
                        await self._click_by_ref(toolkit, reset_ref, "click reset")
                        memory.append_skip(
                            SkipRecord(
                                reason="no_card_then_reset",
                                status_text=status_text,
                                timestamp_utc=self._utc_now(),
                            )
                        )
                        self._persist_memory(memory)
                        await asyncio.sleep(0.5)
                        continue

                    raise RuntimeError("No cards and no next/reset action available")

                screenshot_path = await self._capture_som_screenshot_path(toolkit)
                decision_step += 1
                decision_start = time.perf_counter()
                psy_recovered_pre = list(self.psy_engine.apply_recovery_if_needed())
                psy_state_before = self.psy_engine.export_state()
                psy_candidates = self._build_psy_candidates(cards)
                psy_choice_idx, pareto_front = self.psy_engine.select_best_item(psy_candidates)
                psy_choice = psy_candidates[psy_choice_idx]
                psy_deltas, psy_recovered, psy_metrics = self.psy_engine.consume_item(
                    psy_choice["v"],
                    psy_choice["n"],
                    psy_choice["c"],
                    psy_choice["pop"],
                )
                psy_state = self.psy_engine.export_state()
                psy_record = {
                    "step": decision_step,
                    "timestamp_utc": self._utc_now(),
                    "selected_item_id": psy_choice.get("item_id", ""),
                    "selected_title": psy_choice.get("title", ""),
                    "pareto_front": [int(x) for x in pareto_front],
                    "deltas": [float(x) for x in np.array(psy_deltas, dtype=np.float32).tolist()],
                    "recovered_before_round": list(psy_recovered_pre),
                    "recovered": list(psy_recovered),
                    "metrics": {k: float(v) for k, v in psy_metrics.items()},
                    "state_before": psy_state_before,
                    "state": psy_state,
                }
                memory.append_psy_state(psy_record)

                force_next_by_psy = (
                    int(psy_state.get("Surf", 1)) == 0
                    or any(float(psy_state.get(k, 1.0)) <= 0.0 for k in ("P", "C", "E", "S"))
                )
                if force_next_by_psy:
                    action, selected_item_id, _reply_text = (
                        "next",
                        "",
                        "[FORCED_NEXT_BY_PSYBER_SURF_0]",
                    )
                else:
                    action, selected_item_id, _reply_text = self._decide_action_with_agent(
                        screenshot_path=screenshot_path,
                        cards=cards,
                        next_enabled=next_enabled,
                        status_text=status_text,
                        memory=memory,
                        psy_state=psy_state,
                        psy_engine_choice={
                            "item_id": psy_choice.get("item_id", ""),
                            "title": psy_choice.get("title", ""),
                            "item_n": float(psy_choice.get("n", 0.0)),
                            "item_c": float(psy_choice.get("c", 0.0)),
                            "item_pop": float(psy_choice.get("pop", 0.0)),
                        },
                    )
                decision_seconds = time.perf_counter() - decision_start
                psy_delta_vals = np.array(psy_deltas, dtype=np.float32).tolist()
                psy_state_log = (
                    f"P={float(psy_state.get('P', 0.0)):.3f},"
                    f"C={float(psy_state.get('C', 0.0)):.3f},"
                    f"E={float(psy_state.get('E', 0.0)):.3f},"
                    f"S={float(psy_state.get('S', 0.0)):.3f},"
                    f"Surf={int(psy_state.get('Surf', 1))}"
                )
                recovered_log = ",".join(psy_recovered_pre) if psy_recovered_pre else "-"
                psy_delta_log = (
                    f"dP={float(psy_delta_vals[0]):+.3f},"
                    f"dC={float(psy_delta_vals[1]):+.3f},"
                    f"dE={float(psy_delta_vals[2]):+.3f},"
                    f"dS={float(psy_delta_vals[3]):+.3f}"
                )
                page_pos: str | int = "-"
                if action == "click":
                    for idx, card in enumerate(cards, start=1):
                        if str(card.get("item_id") or "") == selected_item_id:
                            page_pos = idx
                            break
                decision_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[QWEN] {decision_time} step={decision_step} "
                    f"action={action} item_id={selected_item_id or '-'} "
                    f"page_pos={page_pos} DT={decision_seconds:.3f} "
                    f"forced_next={str(force_next_by_psy).lower()} "
                    f"recovered_before_round={recovered_log} "
                    f"psy_state=({psy_state_log}) "
                    f"psy_delta=({psy_delta_log})"
                )

                if action == "next":
                    if next_enabled:
                        await self._click_by_ref(toolkit, next_ref, "click next")
                        memory.append_skip(
                            SkipRecord(
                                reason=(
                                    "psyber_forced_next_due_surf_0"
                                    if force_next_by_psy
                                    else "user_not_interested_next_page"
                                ),
                                status_text=status_text,
                                timestamp_utc=self._utc_now(),
                            )
                        )
                        self._persist_memory(memory)
                        await asyncio.sleep(0.5)
                        continue

                    reset_ref = str(state.get("reset_ref") or "")
                    if reset_ref:
                        await self._click_by_ref(toolkit, reset_ref, "click reset")
                        memory.append_skip(
                            SkipRecord(
                                reason="next_not_available_then_reset",
                                status_text=status_text,
                                timestamp_utc=self._utc_now(),
                            )
                        )
                        self._persist_memory(memory)
                        await asyncio.sleep(0.3)
                        continue

                    raise RuntimeError("Action next selected but no next/reset available")

                match = [
                    x for x in cards if str(x.get("item_id") or "") == selected_item_id
                ]
                choice = match[0] if match else cards[0]
                item_id = str(choice.get("item_id") or "")
                title = str(choice.get("title") or "")
                click_ok = False
                click_error = ""
                for _ in range(3):
                    js_click = (
                        "(() => {"
                        "const cards = Array.from(document.querySelectorAll('.card'));"
                        "if (cards.length === 0) return {ok:false, reason:'no_card'};"
                        f"const target = cards.find(el => (el.dataset.movieId || '') === '{item_id}') || cards[0];"
                        "target.click();"
                        "return {ok:true, item_id: target.dataset.movieId || ''};"
                        "})()"
                    )
                    fallback = await toolkit.browser_console_exec(js_click)
                    fallback_parsed = self._parse_console_result(
                        str(fallback.get("result", ""))
                    )
                    if (
                        isinstance(fallback_parsed, dict)
                        and fallback_parsed.get("ok") is True
                    ):
                        click_ok = True
                        if str(fallback_parsed.get("item_id") or ""):
                            item_id = str(fallback_parsed.get("item_id"))
                        break

                    click_error = f"js_click_failed: {fallback.get('result', '')}"
                    await asyncio.sleep(0.3)
                    await self._ensure_snapshot(toolkit)
                    retry_state = await self._get_state(toolkit)
                    retry_cards = [
                        x for x in retry_state.get("cards", []) if x.get("ref")
                    ]
                    if not retry_cards:
                        continue
                    new_choice = retry_cards[0]
                    item_id = str(new_choice.get("item_id") or item_id)
                    title = str(new_choice.get("title") or title)

                if not click_ok:
                    raise RuntimeError(click_error or "click card failed")

                memory.append_click(
                    ClickRecord(
                        order=len(memory.click_sequence) + 1,
                        item_id=item_id,
                        title=title,
                        dataset_key=memory.selected_dataset_key,
                        model_key=memory.selected_model_key,
                        status_text=status_text,
                        timestamp_utc=self._utc_now(),
                    )
                )
                self._persist_memory(memory)
                await asyncio.sleep(0.3)

            return memory
        finally:
            try:
                await toolkit.browser_close()
            except Exception:
                pass

    def write_summary(self, memory: AgentMemory) -> None:
        lines = [
            f"run_dir={self.run_dir}",
            f"task_file={WEB_SIM_TASK_FILE}",
            f"dataset={memory.selected_dataset_key}",
            f"model={memory.selected_model_key}",
            f"track_target={memory.track_target}",
            f"click_count={len(memory.click_sequence)}",
            f"skip_count={len(memory.skip_sequence)}",
            f"memory_file={self.memory_file}",
        ]
        self.summary_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(self) -> None:
        task = self.load_task_config()
        if self.track_override is not None:
            task.track = self.track_override
            task.raw = dict(task.raw)
            task.raw["track"] = self.track_override
        self.ensure_service()

        memory = asyncio.run(self.run_click_simulation(task))
        self.write_summary(memory)

        self.shutdown_service_if_needed()

        print("Run completed.")
        print(f"- output: {self.run_dir}")
        print(f"- memory: {self.memory_file}")
        print(f"- clicks: {len(memory.click_sequence)} / {memory.track_target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Web_sim CAMEL browser agent")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Users/chongzhang/PsyBer-Agent"),
        help="Output root directory for run artifacts",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (default: visible browser)",
    )
    parser.add_argument(
        "--skip-probability",
        type=float,
        default=0.3,
        help="[Deprecated] kept for compatibility, ignored",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="[Deprecated] kept for compatibility, ignored",
    )
    parser.add_argument(
        "--decision-model",
        type=str,
        default=None,
        help="Qwen model name, e.g. qwen-plus / qwen-max",
    )
    parser.add_argument(
        "--qwen-base-url",
        type=str,
        default=None,
        help="Qwen OpenAI-compatible endpoint. Default: dashscope compatible-mode",
    )
    parser.add_argument(
        "--close-service-when-done",
        action="store_true",
        help="Stop Web_sim service at the end if this run started it",
    )
    parser.add_argument(
        "--track-override",
        type=int,
        default=None,
        help="Override track count for this run only",
    )
    args = parser.parse_args()

    agent = WebSimCamelAgent(
        output_root=args.output_root,
        headless=args.headless,
        skip_probability=args.skip_probability,
        seed=args.seed,
        close_service_when_done=args.close_service_when_done,
        track_override=args.track_override,
        decision_model=args.decision_model,
        qwen_base_url=args.qwen_base_url,
    )
    agent.run()


if __name__ == "__main__":
    main()
