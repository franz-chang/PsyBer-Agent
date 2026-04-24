#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    run_id: str
    memory_path: Path
    dataset: str
    model: str
    track_target: int
    agent_num: int
    profile_id: str
    clicks: int
    skips: int
    decisions: int
    ctr: float | None
    latency_per_decision_sec: float | None
    click_item_counter: Counter[str]
    pces_trajectory: list[list[float]]
    action_sequence: list[str]
    raw: dict[str, Any]


@dataclass
class AbCase:
    condition: str
    seed: str | None
    oracle_lift: float
    pred_lift: float


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_div(a: float, b: float) -> float | None:
    if abs(b) <= 1e-12:
        return None
    return a / b


def mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / float(len(xs))


def mean_ci95(xs: list[float]) -> dict[str, float | int | None]:
    if not xs:
        return {"mean": None, "ci95": None, "n": 0}
    m = mean(xs)
    if len(xs) < 2:
        return {"mean": m, "ci95": 0.0, "n": len(xs)}
    sd = statistics.stdev(xs)
    ci95 = 1.96 * sd / math.sqrt(len(xs))
    return {"mean": m, "ci95": ci95, "n": len(xs)}


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def counter_to_prob(counter: Counter[str]) -> dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0:
        return {}
    return {k: float(v) / total for k, v in counter.items()}


def normalize_prob_like(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    vals = {str(k): max(0.0, _to_float(v)) for k, v in d.items()}
    s = sum(vals.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in vals.items()}


def js_divergence(a: dict[str, float], b: dict[str, float], eps: float = 1e-12) -> float | None:
    pa = normalize_prob_like(a)
    pb = normalize_prob_like(b)
    if not pa and not pb:
        return None
    keys = sorted(set(pa.keys()) | set(pb.keys()))
    va = [float(pa.get(k, 0.0)) for k in keys]
    vb = [float(pb.get(k, 0.0)) for k in keys]

    def _kl(p: list[float], q: list[float]) -> float:
        s = 0.0
        for pi, qi in zip(p, q):
            if pi <= 0.0:
                continue
            s += pi * math.log((pi + eps) / (qi + eps))
        return s

    vm = [(x + y) / 2.0 for x, y in zip(va, vb)]
    return 0.5 * _kl(va, vm) + 0.5 * _kl(vb, vm)


def trajectory_mse(a: list[list[float]], b: list[list[float]]) -> float | None:
    if not a or not b:
        return None
    n = min(len(a), len(b))
    if n <= 0:
        return None
    err = 0.0
    cnt = 0
    for i in range(n):
        m = min(len(a[i]), len(b[i]))
        for j in range(m):
            d = float(a[i][j]) - float(b[i][j])
            err += d * d
            cnt += 1
    if cnt <= 0:
        return None
    return err / float(cnt)


def mean_trajectory(trajs: list[list[list[float]]]) -> list[list[float]]:
    if not trajs:
        return []
    max_len = max((len(t) for t in trajs), default=0)
    out: list[list[float]] = []
    for i in range(max_len):
        buf: list[list[float]] = []
        for t in trajs:
            if i < len(t):
                row = t[i]
                if len(row) >= 4:
                    buf.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        if not buf:
            continue
        cols = list(zip(*buf))
        out.append([float(sum(c) / len(c)) for c in cols])
    return out


def _infer_action(step: dict[str, Any]) -> str:
    title = str(step.get("selected_title") or "")
    item_id = str(step.get("selected_item_id") or "")
    if title == "[SKIP]" or not item_id:
        return "skip"
    return "click"


def load_memory_file(memory_path: Path) -> RunRecord | None:
    try:
        data = json.loads(memory_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    run_id = memory_path.parent.name
    dataset = str(data.get("selected_dataset_key") or "").strip().lower()
    model = str(data.get("selected_model_key") or "").strip().lower()
    track_target = int(_to_float(data.get("track_target"), 0.0))
    agent_num = int(_to_float(data.get("agent_num"), 0.0))

    agent_profile = data.get("agent_profile") if isinstance(data.get("agent_profile"), dict) else {}
    profile_id = str(agent_profile.get("agent_id") if agent_profile else "")

    click_seq = data.get("click_sequence") if isinstance(data.get("click_sequence"), list) else []
    skip_seq = data.get("skip_sequence") if isinstance(data.get("skip_sequence"), list) else []
    psy_hist = data.get("psy_state_history") if isinstance(data.get("psy_state_history"), list) else []

    clicks = len(click_seq)
    skips = len(skip_seq)
    decisions = len(psy_hist)
    ctr = _safe_div(float(clicks), float(decisions))

    click_counter = Counter(
        str(x.get("item_id") or "").strip()
        for x in click_seq
        if isinstance(x, dict) and str(x.get("item_id") or "").strip()
    )

    timestamps: list[datetime | None] = []
    for x in psy_hist:
        if isinstance(x, dict):
            timestamps.append(_parse_iso(str(x.get("timestamp_utc") or "")))
    deltas: list[float] = []
    for i in range(1, len(timestamps)):
        t1 = timestamps[i - 1]
        t2 = timestamps[i]
        if t1 is None or t2 is None:
            continue
        dt = (t2 - t1).total_seconds()
        if dt >= 0:
            deltas.append(float(dt))
    latency = mean(deltas)

    pces: list[list[float]] = []
    actions: list[str] = []
    for step in psy_hist:
        if not isinstance(step, dict):
            continue
        st = step.get("state") if isinstance(step.get("state"), dict) else {}
        pces.append([
            _to_float(st.get("P", 0.0)),
            _to_float(st.get("C", 0.0)),
            _to_float(st.get("E", 0.0)),
            _to_float(st.get("S", 0.0)),
        ])
        actions.append(_infer_action(step))

    return RunRecord(
        run_id=run_id,
        memory_path=memory_path,
        dataset=dataset,
        model=model,
        track_target=track_target,
        agent_num=agent_num,
        profile_id=profile_id,
        clicks=clicks,
        skips=skips,
        decisions=decisions,
        ctr=ctr,
        latency_per_decision_sec=latency,
        click_item_counter=click_counter,
        pces_trajectory=pces,
        action_sequence=actions,
        raw=data,
    )


def load_runs_recursive(runs_root: Path) -> list[RunRecord]:
    out: list[RunRecord] = []
    if not runs_root.exists():
        return out
    for memory_path in sorted(runs_root.rglob("memory.json")):
        rec = load_memory_file(memory_path)
        if rec is not None:
            out.append(rec)
    return out


def aggregate_behavior(runs: list[RunRecord]) -> dict[str, Any]:
    total_clicks = sum(r.clicks for r in runs)
    total_decisions = sum(r.decisions for r in runs)
    ctr = _safe_div(float(total_clicks), float(total_decisions))

    exposure = Counter()
    for r in runs:
        exposure.update(r.click_item_counter)
    exposure_dist = counter_to_prob(exposure)

    avg_traj = mean_trajectory([r.pces_trajectory for r in runs if r.pces_trajectory])
    lat_vals = [float(r.latency_per_decision_sec) for r in runs if r.latency_per_decision_sec is not None]

    return {
        "run_count": len(runs),
        "clicks": total_clicks,
        "decisions": total_decisions,
        "ctr": ctr,
        "item_exposure_dist": exposure_dist,
        "avg_pces_trajectory": avg_traj,
        "latency_per_decision_sec": mean(lat_vals),
        "latency_per_decision_ci95": mean_ci95(lat_vals),
    }


def action_flip_rate(base_actions: list[str], pert_actions: list[str]) -> float | None:
    n = min(len(base_actions), len(pert_actions))
    if n <= 0:
        return None
    flips = 0
    for i in range(n):
        flips += 1 if base_actions[i] != pert_actions[i] else 0
    return flips / float(n)


def policy_shift_tv(base_actions: list[str], pert_actions: list[str]) -> float | None:
    if not base_actions or not pert_actions:
        return None

    def _dist(actions: list[str]) -> dict[str, float]:
        c = Counter(actions)
        total = float(sum(c.values()))
        if total <= 0:
            return {}
        return {k: float(v) / total for k, v in c.items()}

    pa = _dist(base_actions)
    pb = _dist(pert_actions)
    keys = sorted(set(pa.keys()) | set(pb.keys()))
    tv = 0.5 * sum(abs(float(pa.get(k, 0.0)) - float(pb.get(k, 0.0))) for k in keys)
    return tv


def pairwise_js_exposure(runs: list[RunRecord]) -> float | None:
    if len(runs) < 2:
        return None
    vals: list[float] = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            js = js_divergence(
                counter_to_prob(runs[i].click_item_counter),
                counter_to_prob(runs[j].click_item_counter),
            )
            if js is not None:
                vals.append(js)
    return mean(vals)


def load_ab_cases(path: Path | None) -> list[AbCase]:
    if path is None or not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    rows: list[Any]
    if isinstance(raw, dict) and isinstance(raw.get("ab_cases"), list):
        rows = raw.get("ab_cases")
    elif isinstance(raw, dict) and isinstance(raw.get("cases"), list):
        rows = raw.get("cases")
    elif isinstance(raw, list):
        rows = raw
    else:
        return []

    out: list[AbCase] = []
    for x in rows:
        if not isinstance(x, dict):
            continue
        cond = str(
            x.get("condition")
            or x.get("perturbation")
            or x.get("variant")
            or ""
        ).strip()
        if not cond:
            continue
        out.append(
            AbCase(
                condition=cond,
                seed=(str(x.get("seed")).strip() if x.get("seed") is not None else None),
                oracle_lift=_to_float(x.get("oracle_lift", 0.0)),
                pred_lift=_to_float(x.get("pred_lift", 0.0)),
            )
        )
    return out


def ab_metrics(cases: list[AbCase], no_effect_delta: float = 1e-3) -> dict[str, float | None]:
    if not cases:
        return {
            "Sign Accuracy": None,
            "Lift MAE": None,
            "Type-I Error": None,
            "Power@δ(1%)": None,
            "Power@δ(3%)": None,
            "Power@δ(5%)": None,
        }

    def _sign(x: float) -> int:
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    n = len(cases)
    sign_ok = 0
    mae = 0.0
    oracle = [c.oracle_lift for c in cases]
    pred = [c.pred_lift for c in cases]

    for o, p in zip(oracle, pred):
        sign_ok += 1 if _sign(o) == _sign(p) else 0
        mae += abs(p - o)

    no_eff_idx = [i for i, o in enumerate(oracle) if abs(o) <= no_effect_delta]
    reject_idx = [i for i, p in enumerate(pred) if abs(p) > no_effect_delta]
    if no_eff_idx:
        fp = sum(1 for i in no_eff_idx if i in reject_idx)
        type1 = fp / float(len(no_eff_idx))
    else:
        type1 = None

    def _power(th: float) -> float | None:
        idx = [i for i, o in enumerate(oracle) if abs(o) >= th]
        if not idx:
            return None
        ok = 0
        for i in idx:
            ok += 1 if _sign(oracle[i]) == _sign(pred[i]) else 0
        return ok / float(len(idx))

    return {
        "Sign Accuracy": sign_ok / float(n),
        "Lift MAE": mae / float(n),
        "Type-I Error": type1,
        "Power@δ(1%)": _power(0.01),
        "Power@δ(3%)": _power(0.03),
        "Power@δ(5%)": _power(0.05),
    }


def percentile(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    xs = sorted(float(x) for x in vals)
    if q <= 0:
        return xs[0]
    if q >= 1:
        return xs[-1]
    idx = q * (len(xs) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return xs[lo]
    frac = idx - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows
