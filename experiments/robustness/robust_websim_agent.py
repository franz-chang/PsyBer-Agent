#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import WebSim_agent as base  # noqa: E402


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, "").strip().lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "y", "on"}


def _env_float(key: str, default: float = 0.0) -> float:
    txt = os.getenv(key, "").strip()
    if not txt:
        return default
    try:
        return float(txt)
    except Exception:
        return default


def _env_int(key: str, default: int = 0) -> int:
    txt = os.getenv(key, "").strip()
    if not txt:
        return default
    try:
        return int(txt)
    except Exception:
        return default


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


@dataclass
class RobustnessConfig:
    run_label: str
    seed: int
    prompt_paraphrase: bool
    affective_shift: float
    history_noise: float
    click_history_mask: float
    ranking_perturbation: str
    memory_truncation: float
    memory_shuffle: bool
    memory_dropout: float
    stale_memory_injection: bool

    dataset_override: str | None
    model_override: str | None
    agent_num_override: int | None

    @classmethod
    def from_env(
        cls,
        *,
        dataset_override: str | None,
        model_override: str | None,
        agent_num_override: int | None,
    ) -> "RobustnessConfig":
        return cls(
            run_label=(os.getenv("ROBUST_RUN_LABEL", "").strip() or "clean"),
            seed=_env_int("ROBUST_SEED", 42),
            prompt_paraphrase=_env_bool("ROBUST_PROMPT_PARAPHRASE", False),
            affective_shift=_env_float("ROBUST_AFFECTIVE_SHIFT", 0.0),
            history_noise=max(0.0, min(1.0, _env_float("ROBUST_HISTORY_NOISE", 0.0))),
            click_history_mask=max(0.0, min(1.0, _env_float("ROBUST_CLICK_HISTORY_MASK", 0.0))),
            ranking_perturbation=(os.getenv("ROBUST_RANKING_PERTURB", "none").strip().lower() or "none"),
            memory_truncation=max(0.0, min(1.0, _env_float("ROBUST_MEMORY_TRUNCATION", 0.0))),
            memory_shuffle=_env_bool("ROBUST_MEMORY_SHUFFLE", False),
            memory_dropout=max(0.0, min(1.0, _env_float("ROBUST_MEMORY_DROPOUT", 0.0))),
            stale_memory_injection=_env_bool("ROBUST_STALE_MEMORY_INJECTION", False),
            dataset_override=dataset_override or (os.getenv("ROBUST_DATASET_OVERRIDE", "").strip() or None),
            model_override=model_override or (os.getenv("ROBUST_MODEL_OVERRIDE", "").strip() or None),
            agent_num_override=(
                agent_num_override
                if agent_num_override is not None
                else (_env_int("ROBUST_AGENT_NUM_OVERRIDE", 0) or None)
            ),
        )


class RobustWebSimAgent(base.WebSimCamelAgent):
    def __init__(
        self,
        *args: Any,
        dataset_override: str | None = None,
        model_override: str | None = None,
        agent_num_override: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.robust_cfg = RobustnessConfig.from_env(
            dataset_override=dataset_override,
            model_override=model_override,
            agent_num_override=agent_num_override,
        )
        self.robust_rng = np.random.default_rng(self.robust_cfg.seed)

    # -------------------------------
    # Injectors
    # -------------------------------

    def _select_agent_profile(self, agent_num: int) -> dict[str, Any]:
        profile = dict(super()._select_agent_profile(agent_num))
        if abs(self.robust_cfg.affective_shift) <= 1e-12:
            return profile

        init_state = profile.get("initial_state")
        if not isinstance(init_state, dict):
            init_state = {}

        shift = float(self.robust_cfg.affective_shift)
        c = _clip01(float(init_state.get("C", 0.55)) - 0.5 * shift)
        e = _clip01(float(init_state.get("E", 0.55)) + 1.0 * shift)
        s = _clip01(float(init_state.get("S", 0.55)) - 0.2 * shift)
        init_state.update({"C": c, "E": e, "S": s})
        profile["initial_state"] = init_state
        return profile

    def _clone_prompt_memory(self, memory: base.AgentMemory) -> base.AgentMemory:
        # Prompt-only corruption: keep internal engine untouched.
        masked: list[base.ClickRecord] = []
        for rec in memory.click_sequence:
            if self.robust_cfg.click_history_mask > 0 and self.robust_rng.random() < self.robust_cfg.click_history_mask:
                continue
            title = str(rec.title)
            if self.robust_cfg.history_noise > 0 and self.robust_rng.random() < self.robust_cfg.history_noise:
                title = f"{title} [noisy]"
            masked.append(
                base.ClickRecord(
                    order=int(rec.order),
                    item_id=str(rec.item_id),
                    title=title,
                    dataset_key=str(rec.dataset_key),
                    model_key=str(rec.model_key),
                    status_text=str(rec.status_text),
                    timestamp_utc=str(rec.timestamp_utc),
                )
            )

        skip_copy = [
            base.SkipRecord(
                reason=str(x.reason),
                status_text=str(x.status_text),
                timestamp_utc=str(x.timestamp_utc),
            )
            for x in memory.skip_sequence
        ]

        return base.AgentMemory(
            task_config=dict(memory.task_config),
            selected_dataset_key=memory.selected_dataset_key,
            selected_model_key=memory.selected_model_key,
            track_target=memory.track_target,
            agent_num=memory.agent_num,
            decision_mode=memory.decision_mode,
            fast_policy_path=memory.fast_policy_path,
            agent_profile=dict(memory.agent_profile),
            psy_state_history=list(memory.psy_state_history),
            click_sequence=masked,
            skip_sequence=skip_copy,
        )

    def _perturb_cards(self, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cards2 = [dict(x) for x in cards]
        mode = self.robust_cfg.ranking_perturbation
        if mode == "adjacent_swap" and len(cards2) >= 2:
            cards2[0], cards2[1] = cards2[1], cards2[0]
        elif mode == "top_rank_replacement" and len(cards2) >= 2:
            cards2 = [cards2[-1]] + cards2[:-1]
        elif mode == "distractor_insertion" and cards2:
            fake_id = f"distractor_{self.robust_cfg.seed}_{len(cards2)}"
            distractor = {
                "item_id": fake_id,
                "title": "[Distractor] sponsored content",
                "ref": cards2[0].get("ref", ""),
            }
            cards2 = [cards2[0], distractor] + cards2[1:]
        return cards2

    def _perturb_fast_hints(
        self,
        hints: dict[str, Any] | None,
        *,
        next_enabled: bool,
    ) -> dict[str, Any] | None:
        if not isinstance(hints, dict):
            return hints
        if self.robust_cfg.ranking_perturbation != "conflicting_signals":
            return hints

        h = copy.deepcopy(hints)
        choices = h.get("choices")
        if isinstance(choices, list) and choices:
            choices = list(reversed(choices))
            if next_enabled:
                choices[0]["action"] = "next"
                choices[0]["item_id"] = ""
                choices[0]["title"] = "[SKIP]"
            h["choices"] = choices
        return h

    def _build_decision_prompt(self, **kwargs: Any) -> str:
        prompt = super()._build_decision_prompt(**kwargs)
        if not self.robust_cfg.prompt_paraphrase:
            return prompt

        # Paraphrase while preserving constraints and JSON-only policy.
        prompt = prompt.replace(
            "你是一名线上用户，请根据截图选择你喜欢的，然后点击你喜欢的图片。你必须只输出 JSON,不要输出其它文本。",
            "你在模拟真实用户浏览推荐流。请结合截图和上下文做最终点击决策。仅输出 JSON，不要输出其它文本。",
        )
        prompt = prompt.replace(
            "你会同时收到以下决策信息: 你此时的心理状态: psy_state、根据你的心理状态下可能的选择:fast_policy_candidates(top-k)、最近点击: recent_clicks 与页面上可选项: candidates。",
            "你会获得心理状态、候选偏好提示、最近行为和当前候选项。请综合这些信号形成连贯选择。",
        )
        return prompt

    def _apply_memory_perturbation(self) -> None:
        if self.psy_engine is None or not hasattr(self.psy_engine, "memory"):
            return

        memory = list(getattr(self.psy_engine, "memory", []))
        if not memory:
            return

        # recent-window truncation
        if self.robust_cfg.memory_truncation > 0:
            keep_ratio = max(0.0, 1.0 - self.robust_cfg.memory_truncation)
            keep_n = max(1, int(round(len(memory) * keep_ratio)))
            memory = memory[-keep_n:]

        # temporal shuffling
        if self.robust_cfg.memory_shuffle and len(memory) > 1:
            self.robust_rng.shuffle(memory)

        # random dropout
        if self.robust_cfg.memory_dropout > 0:
            kept = [m for m in memory if self.robust_rng.random() >= self.robust_cfg.memory_dropout]
            memory = kept if kept else [memory[-1]]

        # stale injection
        if self.robust_cfg.stale_memory_injection and memory:
            stale = np.array(memory[0], dtype=np.float32)
            inject_n = max(1, int(round(0.3 * len(memory))))
            for i in range(min(inject_n, len(memory))):
                memory[i] = stale

        max_memory = int(getattr(self.psy_engine, "max_memory_size", len(memory)))
        memory = memory[-max_memory:] if max_memory > 0 else memory
        setattr(self.psy_engine, "memory", [np.array(v, dtype=np.float32) for v in memory])

    def _build_psy_candidates(self, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._apply_memory_perturbation()
        return super()._build_psy_candidates(cards)

    def _decide_action_with_agent(self, *, cards: list[dict[str, Any]], memory: base.AgentMemory, fast_policy_hints: dict[str, Any] | None, next_enabled: bool, **kwargs: Any):  # type: ignore[override]
        cards2 = self._perturb_cards(cards)
        memory2 = self._clone_prompt_memory(memory)
        hints2 = self._perturb_fast_hints(fast_policy_hints, next_enabled=next_enabled)
        return super()._decide_action_with_agent(
            cards=cards2,
            memory=memory2,
            fast_policy_hints=hints2,
            next_enabled=next_enabled,
            **kwargs,
        )

    # -------------------------------
    # Persist robustness metadata
    # -------------------------------

    def _persist_memory(self, memory: base.AgentMemory) -> None:
        payload = memory.to_dict()
        payload["robustness"] = asdict(self.robust_cfg)
        self.memory_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(self) -> None:
        task = self.load_task_config()

        if self.robust_cfg.dataset_override:
            task.dataset_candidates = [self.robust_cfg.dataset_override]
            task.raw = dict(task.raw)
            task.raw["dataset"] = [self.robust_cfg.dataset_override]
        if self.robust_cfg.model_override:
            task.model_candidates = [self.robust_cfg.model_override]
            task.raw = dict(task.raw)
            task.raw["model"] = [self.robust_cfg.model_override]
        if self.robust_cfg.agent_num_override is not None and self.robust_cfg.agent_num_override > 0:
            task.agent_num = int(self.robust_cfg.agent_num_override)
            task.raw = dict(task.raw)
            task.raw["agent_num"] = int(self.robust_cfg.agent_num_override)
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
        print(f"- robustness_label: {self.robust_cfg.run_label}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness-enabled WebSim agent")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--skip-probability", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decision-model", type=str, default=None)
    parser.add_argument("--qwen-base-url", type=str, default=None)
    parser.add_argument("--close-service-when-done", action="store_true")
    parser.add_argument("--track-override", type=int, default=None)
    parser.add_argument("--require-fast-inference", action="store_true")
    parser.add_argument("--disable-fast-inference", action="store_true")
    parser.add_argument("--fast-policy-path", type=Path, default=None)
    parser.add_argument("--fast-policy-stochastic", action="store_true")

    parser.add_argument("--dataset-override", type=str, default=None)
    parser.add_argument("--model-override", type=str, default=None)
    parser.add_argument("--agent-num-override", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = RobustWebSimAgent(
        output_root=args.output_root,
        headless=args.headless,
        skip_probability=args.skip_probability,
        seed=args.seed,
        close_service_when_done=args.close_service_when_done,
        track_override=args.track_override,
        decision_model=args.decision_model,
        qwen_base_url=args.qwen_base_url,
        disable_fast_inference=args.disable_fast_inference,
        fast_policy_path=args.fast_policy_path,
        fast_policy_stochastic=args.fast_policy_stochastic,
        require_fast_inference=args.require_fast_inference,
        dataset_override=args.dataset_override,
        model_override=args.model_override,
        agent_num_override=args.agent_num_override,
    )
    agent.run()


if __name__ == "__main__":
    main()
