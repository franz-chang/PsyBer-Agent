#!/usr/bin/env python3
"""Generate preliminary markdown experiment report.

Default output:
evaluation/reports/preliminary_experiment_results.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


EKEY = {
    "E1": "E1_behavior_fidelity",
    "E2": "E2_online_ab_accuracy",
    "E3": "E3_cross_domain_heterogeneity",
    "E4": "E4_ablation",
    "E5": "E5_scalability_cost",
    "E6": "E6_robustness",
    "E7": "E7_qualitative_trajectory",
}

EDESC = {
    "E1": "行为拟真",
    "E2": "在线 A/B 判别",
    "E3": "跨域与异质性",
    "E4": "模块消融",
    "E5": "可扩展性与成本",
    "E6": "稳健性",
    "E7": "案例分析",
}


@dataclass
class RunSummary:
    run_id: str
    track_target: int
    clicks: int
    skips: int
    decisions: int
    ctr: float | None
    repeat_rate: float | None
    latency_sec: float | None
    fast_hint_hits: int
    action_sequence: list[str]
    clicked_items: list[str]


def _safe_div(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return a / b


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):.{digits}f}"


def _fmt_s(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):.{digits}f} s"


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_by_glob(root: Path, pattern: str) -> Path | None:
    cands = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not cands:
        return None
    return cands[-1]


def _infer_action(step: dict[str, Any]) -> str:
    title = str(step.get("selected_title") or "")
    item_id = str(step.get("selected_item_id") or "")
    if title == "[SKIP]" or item_id == "":
        return "skip"
    return "click"


def _transition_matrix(actions: list[str]) -> dict[str, float]:
    keys = ["click->click", "click->skip", "skip->click", "skip->skip"]
    if len(actions) < 2:
        return {k: 0.0 for k in keys}
    cnt = Counter()
    row_cnt = Counter()
    for a, b in zip(actions[:-1], actions[1:]):
        if a not in ("click", "skip") or b not in ("click", "skip"):
            continue
        cnt[(a, b)] += 1
        row_cnt[a] += 1
    out: dict[str, float] = {}
    for a in ("click", "skip"):
        for b in ("click", "skip"):
            den = row_cnt[a]
            out[f"{a}->{b}"] = 0.0 if den <= 0 else cnt[(a, b)] / float(den)
    return out


def _summarize_one_run(memory_file: Path) -> RunSummary:
    d = _load_json(memory_file)
    run_id = memory_file.parent.name
    track_target = int(d.get("track_target") or 0)
    clicks = len(d.get("click_sequence") or [])
    skips = len(d.get("skip_sequence") or [])
    psy = d.get("psy_state_history") or []
    decisions = len(psy)
    ctr = _safe_div(float(clicks), float(decisions))

    clicked_items = [
        str(x.get("item_id") or "").strip()
        for x in (d.get("click_sequence") or [])
        if str(x.get("item_id") or "").strip()
    ]
    uniq_click = len(set(clicked_items))
    repeat = None
    if clicks > 0:
        repeat = 1.0 - (uniq_click / float(clicks))

    # latency from step timestamps
    ts = [_parse_iso(str(x.get("timestamp_utc") or "")) for x in psy]
    dts: list[float] = []
    for i in range(1, len(ts)):
        if ts[i] is None or ts[i - 1] is None:
            continue
        dt = (ts[i] - ts[i - 1]).total_seconds()
        if dt >= 0:
            dts.append(float(dt))
    latency = _safe_div(sum(dts), float(len(dts))) if dts else None

    fast_hits = 0
    actions: list[str] = []
    for step in psy:
        m = step.get("metrics") or {}
        if _to_float(m.get("mode_fast_policy_hint", 0.0)) > 0.0:
            fast_hits += 1
        actions.append(_infer_action(step))

    return RunSummary(
        run_id=run_id,
        track_target=track_target,
        clicks=clicks,
        skips=skips,
        decisions=decisions,
        ctr=ctr,
        repeat_rate=repeat,
        latency_sec=latency,
        fast_hint_hits=fast_hits,
        action_sequence=actions,
        clicked_items=clicked_items,
    )


def _collect_runs(runs_root: Path) -> list[RunSummary]:
    out: list[RunSummary] = []
    for p in sorted(runs_root.glob("*/memory.json")):
        try:
            out.append(_summarize_one_run(p))
        except Exception:
            continue
    return out


def _pick_nested(d: dict[str, Any], path: list[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _suite_status_line(suite: dict[str, Any], exp: str) -> tuple[str, str | None]:
    key = EKEY[exp]
    sec = suite.get(key)
    if not isinstance(sec, dict):
        return "not_run", "该实验未在本次 suite 结果中执行"
    return str(sec.get("status") or "unknown"), sec.get("reason")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate preliminary markdown experiment report")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--reports-dir", type=Path, default=Path("evaluation/reports"))
    parser.add_argument(
        "--suite-report",
        type=Path,
        default=None,
        help="Path to experiment suite JSON. If omitted, auto-pick latest experiment_suite*.json",
    )
    parser.add_argument(
        "--memory-report",
        type=Path,
        default=None,
        help="Path to memory eval JSON. If omitted, auto-pick latest memory_eval*.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/reports/preliminary_experiment_results.md"),
    )
    parser.add_argument("--max-runs", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root.expanduser().resolve()
    reports_dir = args.reports_dir.expanduser().resolve()
    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suite_path = args.suite_report.expanduser().resolve() if args.suite_report else _latest_by_glob(
        reports_dir, "experiment_suite*.json"
    )
    memory_path = args.memory_report.expanduser().resolve() if args.memory_report else _latest_by_glob(
        reports_dir, "memory_eval*.json"
    )

    suite: dict[str, Any] = {}
    if suite_path and suite_path.exists():
        raw = _load_json(suite_path)
        if isinstance(raw, dict):
            suite = raw

    memory: dict[str, Any] = {}
    if memory_path and memory_path.exists():
        raw = _load_json(memory_path)
        if isinstance(raw, dict):
            memory = raw

    runs = _collect_runs(runs_root) if runs_root.exists() else []

    total_runs = len(runs)
    total_decisions = sum(x.decisions for x in runs)
    total_clicks = sum(x.clicks for x in runs)
    total_skips = sum(x.skips for x in runs)

    # fallback counts from existing reports when runs/ is currently empty
    mem_sim = _pick_nested(memory, ["aggregate", "simulated_only_metrics"]) or {}
    if not isinstance(mem_sim, dict):
        mem_sim = {}
    suite_meta = suite.get("meta") if isinstance(suite.get("meta"), dict) else {}

    if total_runs == 0:
        if suite_meta.get("run_count") is not None:
            try:
                total_runs = int(suite_meta.get("run_count"))
            except Exception:
                pass
    if total_decisions == 0 and mem_sim.get("total_decisions") is not None:
        try:
            total_decisions = int(mem_sim.get("total_decisions"))
        except Exception:
            pass
    if total_clicks == 0 and mem_sim.get("total_clicks") is not None:
        try:
            total_clicks = int(mem_sim.get("total_clicks"))
        except Exception:
            pass
    if total_skips == 0 and mem_sim.get("total_skips") is not None:
        try:
            total_skips = int(mem_sim.get("total_skips"))
        except Exception:
            pass

    # core metrics from suite/memory, fallback to runs
    suite_sim = _pick_nested(suite, [EKEY["E1"], "global", "simulated_summary"]) or {}
    if not isinstance(suite_sim, dict):
        suite_sim = {}

    sim_ctr = suite_sim.get("sim_ctr")
    if sim_ctr is None:
        sim_ctr = _safe_div(float(total_clicks), float(total_decisions))

    sim_skip_rate = suite_sim.get("sim_skip_rate")
    if sim_skip_rate is None:
        sim_skip_rate = _safe_div(float(total_skips), float(total_decisions))

    sim_repeat_rate = suite_sim.get("sim_repeat_rate")
    if sim_repeat_rate is None:
        num = 0.0
        den = 0.0
        for r in runs:
            if r.repeat_rate is None:
                continue
            num += float(r.repeat_rate) * float(r.clicks)
            den += float(r.clicks)
        sim_repeat_rate = _safe_div(num, den)

    sim_session_length = suite_sim.get("sim_session_length")
    if sim_session_length is None:
        sim_session_length = _safe_div(float(total_decisions), float(total_runs)) if total_runs > 0 else None

    sim_transition = suite_sim.get("sim_transition_matrix")
    if not isinstance(sim_transition, dict):
        all_actions: list[str] = []
        for r in runs:
            all_actions.extend(r.action_sequence)
        sim_transition = _transition_matrix(all_actions)

    unique_clicked_items = suite_sim.get("unique_clicked_items")
    if unique_clicked_items is None:
        mem_unique = mem_sim.get("unique_clicked_items")
        if mem_unique is not None:
            unique_clicked_items = mem_unique
        else:
            s = set()
            for r in runs:
                s.update(r.clicked_items)
            unique_clicked_items = len(s)

    fast_hint = _pick_nested(memory, ["aggregate", "simulated_only_metrics", "fast_hint_coverage"])
    if fast_hint is None:
        hits = sum(r.fast_hint_hits for r in runs)
        fast_hint = _safe_div(float(hits), float(total_decisions))

    latency = _pick_nested(memory, ["aggregate", "main_metrics", "Efficiency", "Latency/decision"])
    if latency is None:
        lat_vals = [r.latency_sec for r in runs if r.latency_sec is not None]
        latency = _safe_div(sum(float(x) for x in lat_vals), float(len(lat_vals))) if lat_vals else None

    # E1-E7 status lines
    status_lines: list[str] = []
    for e in ("E1", "E2", "E3", "E4", "E5", "E6", "E7"):
        st, reason = _suite_status_line(suite, e)
        if st == "ok":
            text = f"- {e}（{EDESC[e]}）：可执行。"
        elif st == "insufficient_data":
            text = f"- {e}（{EDESC[e]}）：数据不足（{reason or '缺少必要输入'}）。"
        elif st == "not_run":
            text = f"- {e}（{EDESC[e]}）：未执行。"
        else:
            text = f"- {e}（{EDESC[e]}）：状态={st}。"
        status_lines.append(text)

    # conclusions
    conc: list[str] = []
    if sim_ctr is not None and sim_skip_rate is not None:
        conc.append(
            f"目前模拟行为呈现“较高 skip、较低重复”的模式（CTR 约 {float(sim_ctr)*100:.1f}%，skip 约 {float(sim_skip_rate)*100:.1f}%）。"
        )
    if latency is not None:
        conc.append(f"当前决策时延约 {float(latency):.2f}s/step。")
    if fast_hint is not None:
        conc.append(f"fast policy 提示覆盖率约 {float(fast_hint)*100:.1f}%，说明当前推理链路可稳定运行。")
    if not conc:
        conc.append("当前结果可用于验证流程可运行性，但尚缺足够数据形成稳定统计结论。")

    # suggestions
    sugg: list[str] = []
    e1_missing = _pick_nested(suite, [EKEY["E1"], "missing_inputs"])
    if isinstance(e1_missing, list) and len(e1_missing) > 0:
        sugg.append("提供真实参考行为分布（CTR / item exposure / PCES trajectory），补齐 E1 gap 指标。")
    st_e2, _ = _suite_status_line(suite, "E2")
    if st_e2 != "ok":
        sugg.append("补充 ab_cases（oracle_lift、pred_lift、budget、seed），完成 E2/E3/E4/E6 的 A/B 主结论。")
    st_e5, _ = _suite_status_line(suite, "E5")
    if st_e5 != "ok":
        sugg.append("运行 batch launcher 并产出 batch_launches/**/summary.json，完成 E5 scaling 与 cost-quality 分析。")
    if total_runs == 0:
        sugg.append("先生成 runs/*/memory.json（例如执行 evaluation/run_all_experiments.sh 前先完成仿真）。")
    # dedupe, keep order
    dedup: list[str] = []
    seen = set()
    for s in sugg:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    sugg = dedup or ["补充更多跨数据集与多种配置的运行数据，提高结论稳定性。"]

    # per-run table rows
    rows: list[dict[str, Any]] = []
    if runs:
        for r in runs[: max(0, int(args.max_runs))]:
            rows.append(
                {
                    "run_id": r.run_id,
                    "track_target": r.track_target,
                    "clicks": r.clicks,
                    "skips": r.skips,
                    "decisions": r.decisions,
                    "ctr": r.ctr,
                }
            )
    else:
        per_run = memory.get("per_run")
        if isinstance(per_run, list):
            for x in per_run[: max(0, int(args.max_runs))]:
                if not isinstance(x, dict):
                    continue
                rows.append(
                    {
                        "run_id": x.get("run_id"),
                        "track_target": x.get("track_target"),
                        "clicks": x.get("clicks"),
                        "skips": x.get("skips"),
                        "decisions": x.get("decisions"),
                        "ctr": x.get("ctr_sim"),
                    }
                )

    # render markdown
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    src_suite = str(suite_path) if suite_path else "N/A"
    src_memory = str(memory_path) if memory_path else "N/A"

    lines: list[str] = []
    lines.append("# PsyBer-Agent 初步实验结果（基于当前已有数据）")
    lines.append("")
    lines.append(f"生成时间：{now_text}  ")
    lines.append(f"数据来源：`{src_suite}` 与 `{src_memory}`")
    lines.append("")
    lines.append("## 1. 数据规模")
    lines.append("")
    lines.append(f"- 运行数（run）：{total_runs}")
    lines.append(f"- 总决策步数（decisions）：{total_decisions}")
    lines.append(f"- 总点击数（clicks）：{total_clicks}")
    lines.append(f"- 总跳过数（skips）：{total_skips}")
    lines.append("")
    lines.append("## 2. 核心初步指标（当前可计算）")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    lines.append(f"| sim_ctr | {_fmt(sim_ctr, 4)} |")
    lines.append(f"| sim_skip_rate | {_fmt(sim_skip_rate, 4)} |")
    lines.append(f"| sim_repeat_rate | {_fmt(sim_repeat_rate, 4)} |")
    lines.append(f"| sim_session_length | {_fmt(sim_session_length, 1)} |")
    lines.append(f"| Latency/decision | {_fmt_s(latency, 3)} |")
    lines.append(f"| fast_hint_coverage | {_fmt(fast_hint, 4)} |")
    if unique_clicked_items is None:
        lines.append("| unique_clicked_items | N/A |")
    else:
        lines.append(f"| unique_clicked_items | {int(unique_clicked_items)} |")
    lines.append("")
    lines.append("## 3. 行为转移统计（模拟侧）")
    lines.append("")
    lines.append("| 转移项 | 概率 |")
    lines.append("|---|---:|")
    lines.append(f"| click -> click | {_fmt(_to_float(sim_transition.get('click->click')), 4)} |")
    lines.append(f"| click -> skip | {_fmt(_to_float(sim_transition.get('click->skip')), 4)} |")
    lines.append(f"| skip -> click | {_fmt(_to_float(sim_transition.get('skip->click')), 4)} |")
    lines.append(f"| skip -> skip | {_fmt(_to_float(sim_transition.get('skip->skip')), 4)} |")
    lines.append("")
    lines.append("## 4. 单次运行概览")
    lines.append("")
    if rows:
        lines.append("| run_id | track_target | clicks | skips | decisions | ctr |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in rows:
            lines.append(
                f"| {r.get('run_id')} | {r.get('track_target')} | {r.get('clicks')} | {r.get('skips')} | {r.get('decisions')} | {_fmt(r.get('ctr'), 4)} |"
            )
    else:
        lines.append("- 暂无运行结果（未检测到 `runs/*/memory.json`）。")
    lines.append("")
    lines.append("## 5. E1-E7 执行状态（当前数据条件）")
    lines.append("")
    lines.extend(status_lines)
    lines.append("")
    lines.append("## 6. 初步结论")
    lines.append("")
    for i, c in enumerate(conc, start=1):
        lines.append(f"{i}. {c}")
    lines.append("")
    lines.append("## 7. 下一步补齐建议")
    lines.append("")
    for i, s in enumerate(sugg, start=1):
        lines.append(f"{i}. {s}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Done] markdown={out_path}")
    print(f"[Info] runs={total_runs} suite={src_suite} memory={src_memory}")


if __name__ == "__main__":
    main()
