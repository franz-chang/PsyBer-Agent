#!/usr/bin/env python3
"""Run PsyBer experiment suite E1-E7.

This script implements the experiment structure in:
- ref/PsyBer-Agent实验设计.pdf
- EXPERIMENT_DESIGN.md

It is designed to work with partial inputs. When an experiment cannot be fully
computed due to missing files/fields, the report keeps the section and explains
what is missing.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ------------------------------
# Generic utils
# ------------------------------


def _safe_div(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return a / b


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / float(len(xs))


def _mean_ci95(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"mean": None, "ci95": None, "n": 0}
    m = _mean(xs)
    if len(xs) < 2:
        return {"mean": m, "ci95": 0.0, "n": len(xs)}
    sd = statistics.stdev(xs)
    ci = 1.96 * sd / math.sqrt(len(xs))
    return {"mean": m, "ci95": ci, "n": len(xs)}


def _counter_to_prob(counter: Counter[str]) -> dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0:
        return {}
    return {k: float(v) / total for k, v in counter.items()}


def _normalize_prob_like(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    vals = {str(k): max(0.0, _to_float(v)) for k, v in d.items()}
    s = sum(vals.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in vals.items()}


def _align_keys(a: dict[str, float], b: dict[str, float]) -> tuple[list[float], list[float], list[str]]:
    keys = sorted(set(a.keys()) | set(b.keys()))
    pa = [float(a.get(k, 0.0)) for k in keys]
    pb = [float(b.get(k, 0.0)) for k in keys]
    return pa, pb, keys


def _js_divergence(a: dict[str, float], b: dict[str, float], eps: float = 1e-12) -> float | None:
    a = _normalize_prob_like(a)
    b = _normalize_prob_like(b)
    if not a and not b:
        return None
    pa, pb, _ = _align_keys(a, b)

    def _kl(p: list[float], q: list[float]) -> float:
        s = 0.0
        for pi, qi in zip(p, q):
            if pi <= 0:
                continue
            s += pi * math.log((pi + eps) / (qi + eps))
        return s

    m = [(x + y) / 2.0 for x, y in zip(pa, pb)]
    return 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)


def _trajectory_mse(a: list[list[float]], b: list[list[float]]) -> float | None:
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
    return _safe_div(err, float(cnt))


def _flatten_trajectory(trajs: list[list[list[float]]]) -> list[list[float]]:
    out: list[list[float]] = []
    for t in trajs:
        out.extend(t)
    return out


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = idx - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _transition_matrix(actions: list[str]) -> dict[str, float]:
    keys = ["click->click", "click->skip", "skip->click", "skip->skip"]
    if len(actions) < 2:
        return {k: 0.0 for k in keys}
    counts: dict[tuple[str, str], int] = defaultdict(int)
    row_totals: dict[str, int] = defaultdict(int)
    for a, b in zip(actions[:-1], actions[1:]):
        if a not in ("click", "skip") or b not in ("click", "skip"):
            continue
        counts[(a, b)] += 1
        row_totals[a] += 1

    def _v(a: str, b: str) -> float:
        den = row_totals[a]
        if den <= 0:
            return 0.0
        return counts[(a, b)] / float(den)

    return {
        "click->click": _v("click", "click"),
        "click->skip": _v("click", "skip"),
        "skip->click": _v("skip", "click"),
        "skip->skip": _v("skip", "skip"),
    }


def _normalize_transition_input(x: Any) -> dict[str, float] | None:
    if isinstance(x, dict):
        vals = {str(k): _to_float(v) for k, v in x.items()}
        needed = ["click->click", "click->skip", "skip->click", "skip->skip"]
        if all(k in vals for k in needed):
            return {k: vals[k] for k in needed}
    if isinstance(x, list) and len(x) == 2:
        try:
            cc = _to_float(x[0][0])
            cs = _to_float(x[0][1])
            sc = _to_float(x[1][0])
            ss = _to_float(x[1][1])
            return {
                "click->click": cc,
                "click->skip": cs,
                "skip->click": sc,
                "skip->skip": ss,
            }
        except Exception:
            return None
    return None


def _transition_distance(a: dict[str, float], b: dict[str, float]) -> float | None:
    if not a or not b:
        return None
    keys = ["click->click", "click->skip", "skip->click", "skip->skip"]
    diffs = [abs(_to_float(a.get(k)) - _to_float(b.get(k))) for k in keys]
    return _mean(diffs)


def _infer_action(step: dict[str, Any]) -> str:
    title = str(step.get("selected_title") or "")
    item_id = str(step.get("selected_item_id") or "")
    if title == "[SKIP]" or item_id == "":
        return "skip"
    return "click"


def _drop_pct(base: float | None, pert: float | None, high_is_better: bool) -> float | None:
    if base is None or pert is None:
        return None
    if abs(base) <= 1e-12:
        return None
    if high_is_better:
        return (base - pert) / abs(base)
    return (pert - base) / abs(base)


# ------------------------------
# Data models
# ------------------------------


@dataclass
class RunRecord:
    run_id: str
    path: Path
    dataset: str
    model: str
    decision_mode: str
    track_target: int
    clicks: int
    skips: int
    decisions: int
    ctr: float | None
    skip_rate: float | None
    repeat_rate: float | None
    session_length: int
    latency_per_decision_sec: float | None
    fast_hint_coverage: float | None
    avg_fast_hint_count: float | None
    click_item_counter: Counter[str]
    pces_trajectory: list[list[float]]
    action_sequence: list[str]
    transition_matrix: dict[str, float]
    recovered_rounds: int
    avg_psy_metrics: dict[str, float]
    psy_history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AbCase:
    oracle_lift: float
    pred_lift: float
    budget: int | None
    seed: str | None
    dataset: str | None
    group: str | None
    variant: str | None
    perturbation: str | None
    launch_tag: str | None
    pair: str | None
    run_id: str | None
    cost: float | None
    is_base: bool


# ------------------------------
# Loaders
# ------------------------------


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    d = _load_json(path)
    if isinstance(d, dict) and isinstance(d.get("run_meta"), dict):
        return {str(k): dict(v) for k, v in d["run_meta"].items() if isinstance(v, dict)}
    if isinstance(d, dict) and isinstance(d.get("runs"), list):
        out: dict[str, dict[str, Any]] = {}
        for x in d["runs"]:
            if not isinstance(x, dict):
                continue
            rid = str(x.get("run_id") or "").strip()
            if not rid:
                continue
            out[rid] = dict(x)
        return out
    if isinstance(d, list):
        out: dict[str, dict[str, Any]] = {}
        for x in d:
            if not isinstance(x, dict):
                continue
            rid = str(x.get("run_id") or "").strip()
            if rid:
                out[rid] = dict(x)
        return out
    return {}


def _extract_ref_global(ref: dict[str, Any]) -> dict[str, Any]:
    base = {
        "real_ctr": ref.get("real_ctr"),
        "real_item_exposure": ref.get("real_item_exposure"),
        "real_pces_trajectory": ref.get("real_pces_trajectory"),
        "real_skip_rate": ref.get("real_skip_rate"),
        "real_session_length": ref.get("real_session_length"),
        "real_repeat_rate": ref.get("real_repeat_rate"),
        "real_transition_matrix": ref.get("real_transition_matrix"),
        "real_popularity_bucket_dist": ref.get("real_popularity_bucket_dist"),
        "real_item_popularity": ref.get("real_item_popularity"),
        "avg_run_cost": ref.get("avg_run_cost"),
    }

    behavior_ref = ref.get("behavior_reference")
    if isinstance(behavior_ref, dict):
        g = behavior_ref.get("global")
        if isinstance(g, dict):
            base.update(g)

    return base


def _extract_ref_by_dataset(ref: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_dataset: dict[str, dict[str, Any]] = {}
    candidates: list[Any] = []
    if isinstance(ref.get("by_dataset"), dict):
        candidates.append(ref.get("by_dataset"))
    if isinstance(ref.get("reference_by_dataset"), dict):
        candidates.append(ref.get("reference_by_dataset"))
    behavior_ref = ref.get("behavior_reference")
    if isinstance(behavior_ref, dict) and isinstance(behavior_ref.get("by_dataset"), dict):
        candidates.append(behavior_ref.get("by_dataset"))

    for cand in candidates:
        assert isinstance(cand, dict)
        for k, v in cand.items():
            if isinstance(v, dict):
                by_dataset[str(k).lower()] = dict(v)
    return by_dataset


def _get_ref_for_dataset(reference: dict[str, Any], dataset: str) -> dict[str, Any]:
    global_ref = _extract_ref_global(reference)
    by_dataset = _extract_ref_by_dataset(reference)
    d = dict(global_ref)
    key = dataset.lower().strip()
    if key in by_dataset:
        d.update(by_dataset[key])
    return d


def _load_reference(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    d = _load_json(path)
    if not isinstance(d, dict):
        return {}
    return d


def _load_ab_cases(path: Path | None, fallback_reference: dict[str, Any]) -> list[AbCase]:
    raw: Any = None
    if path is not None:
        raw = _load_json(path)
    elif isinstance(fallback_reference.get("ab_cases"), list):
        raw = fallback_reference.get("ab_cases")
    elif isinstance(fallback_reference.get("ab"), dict) and isinstance(
        fallback_reference.get("ab", {}).get("cases"), list
    ):
        raw = fallback_reference.get("ab", {}).get("cases")

    if isinstance(raw, dict):
        if isinstance(raw.get("ab_cases"), list):
            raw = raw.get("ab_cases")
        elif isinstance(raw.get("cases"), list):
            raw = raw.get("cases")

    if not isinstance(raw, list):
        return []

    out: list[AbCase] = []
    for x in raw:
        if not isinstance(x, dict):
            continue
        budget_raw = x.get("budget", x.get("N", x.get("n")))
        budget: int | None = None
        if budget_raw is not None:
            try:
                budget = int(budget_raw)
            except Exception:
                budget = None
        seed = x.get("seed")
        seed_s = str(seed) if seed is not None and str(seed).strip() else None
        out.append(
            AbCase(
                oracle_lift=_to_float(x.get("oracle_lift", 0.0)),
                pred_lift=_to_float(x.get("pred_lift", 0.0)),
                budget=budget,
                seed=seed_s,
                dataset=(str(x.get("dataset")).strip().lower() if x.get("dataset") else None),
                group=(str(x.get("group")).strip() if x.get("group") else None),
                variant=(str(x.get("variant")).strip() if x.get("variant") else None),
                perturbation=(
                    str(x.get("perturbation")).strip() if x.get("perturbation") else None
                ),
                launch_tag=(str(x.get("launch_tag")).strip() if x.get("launch_tag") else None),
                pair=(str(x.get("pair")).strip() if x.get("pair") else None),
                run_id=(str(x.get("run_id")).strip() if x.get("run_id") else None),
                cost=(
                    _to_float(x.get("cost"))
                    if x.get("cost") is not None
                    else (_to_float(x.get("case_cost")) if x.get("case_cost") is not None else None)
                ),
                is_base=bool(x.get("is_base", False)),
            )
        )
    return out


def _load_batch_summaries(batch_root: Path) -> list[dict[str, Any]]:
    if not batch_root.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(batch_root.glob("**/summary.json")):
        try:
            d = _load_json(p)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        cfg = d.get("config") or {}
        stats = d.get("stats") or {}
        total_tasks = cfg.get("total_tasks")
        submitted = stats.get("submitted", total_tasks)
        success = stats.get("success")
        timeout = stats.get("timeout")
        finished = stats.get("finished")
        elapsed = _to_float(d.get("elapsed_sec", 0.0))
        submitted_i = int(submitted) if submitted is not None else None
        success_i = int(success) if success is not None else 0
        timeout_i = int(timeout) if timeout is not None else 0
        finished_i = int(finished) if finished is not None else 0

        success_rate = (
            _safe_div(float(success_i), float(submitted_i))
            if submitted_i not in (None, 0)
            else None
        )
        timeout_rate = (
            _safe_div(float(timeout_i), float(submitted_i))
            if submitted_i not in (None, 0)
            else None
        )
        task_latency = (
            _safe_div(elapsed, float(success_i))
            if success_i > 0
            else (_safe_div(elapsed, float(finished_i)) if finished_i > 0 else None)
        )

        out.append(
            {
                "path": str(p),
                "launch_tag": str(d.get("launch_tag") or p.parent.name),
                "elapsed_sec": elapsed,
                "total_tasks": int(total_tasks) if total_tasks is not None else None,
                "throughput_tasks_per_sec": _to_float(stats.get("throughput_tasks_per_sec", 0.0)),
                "success": success_i,
                "failed": int(stats.get("failed", 0)),
                "timeout": timeout_i,
                "spawn_error": int(stats.get("spawn_error", 0)),
                "success_rate": success_rate,
                "timeout_rate": timeout_rate,
                "latency_per_task_sec": task_latency,
            }
        )
    return out


def _load_runs(runs_root: Path, run_meta: dict[str, dict[str, Any]]) -> list[RunRecord]:
    out: list[RunRecord] = []
    for mem_path in sorted(runs_root.glob("*/memory.json")):
        d = _load_json(mem_path)
        if not isinstance(d, dict):
            continue

        run_id = mem_path.parent.name
        dataset = str(d.get("selected_dataset_key") or "").strip().lower()
        model = str(d.get("selected_model_key") or "").strip().lower()
        decision_mode = str(d.get("decision_mode") or "").strip().lower()
        track_target = int(d.get("track_target") or 0)

        click_seq = d.get("click_sequence") or []
        skip_seq = d.get("skip_sequence") or []
        psy_hist = d.get("psy_state_history") or []

        clicks = int(len(click_seq))
        skips = int(len(skip_seq))
        decisions = int(len(psy_hist))

        ctr = _safe_div(float(clicks), float(decisions))
        skip_rate = _safe_div(float(skips), float(decisions))

        click_items = [
            str(x.get("item_id") or "").strip()
            for x in click_seq
            if str(x.get("item_id") or "").strip()
        ]
        click_counter = Counter(click_items)
        repeat_rate = None
        if clicks > 0:
            repeat_rate = 1.0 - (len(click_counter) / float(clicks))

        timestamps = [_parse_iso(str(x.get("timestamp_utc") or "")) for x in psy_hist]
        deltas: list[float] = []
        for i in range(1, len(timestamps)):
            t1 = timestamps[i - 1]
            t2 = timestamps[i]
            if t1 is None or t2 is None:
                continue
            dt = (t2 - t1).total_seconds()
            if dt >= 0:
                deltas.append(float(dt))
        latency = _mean(deltas)

        action_sequence = [_infer_action(x) for x in psy_hist]
        transition_matrix = _transition_matrix(action_sequence)

        hint_hits = 0
        hint_cnt_sum = 0.0
        hint_cnt_n = 0
        recovered_rounds = 0
        metric_bag: dict[str, list[float]] = defaultdict(list)
        pces_traj: list[list[float]] = []
        for x in psy_hist:
            metrics = x.get("metrics") or {}
            if _to_float(metrics.get("mode_fast_policy_hint", 0.0)) > 0.0:
                hint_hits += 1
            if "fast_policy_hint_count" in metrics:
                hint_cnt_sum += _to_float(metrics.get("fast_policy_hint_count", 0.0))
                hint_cnt_n += 1
            rec_pre = x.get("recovered_before_round") or []
            rec = x.get("recovered") or []
            if (isinstance(rec_pre, list) and len(rec_pre) > 0) or (isinstance(rec, list) and len(rec) > 0):
                recovered_rounds += 1
            for mk, mv in metrics.items():
                if isinstance(mv, (int, float)):
                    metric_bag[str(mk)].append(float(mv))

            st = x.get("state") or {}
            pces_traj.append(
                [
                    _to_float(st.get("P", 0.0)),
                    _to_float(st.get("C", 0.0)),
                    _to_float(st.get("E", 0.0)),
                    _to_float(st.get("S", 0.0)),
                ]
            )

        hint_cov = _safe_div(float(hint_hits), float(decisions))
        avg_hint_cnt = _safe_div(hint_cnt_sum, float(hint_cnt_n))
        avg_psy_metrics = {k: _mean(v) for k, v in metric_bag.items() if v}

        out.append(
            RunRecord(
                run_id=run_id,
                path=mem_path,
                dataset=dataset,
                model=model,
                decision_mode=decision_mode,
                track_target=track_target,
                clicks=clicks,
                skips=skips,
                decisions=decisions,
                ctr=ctr,
                skip_rate=skip_rate,
                repeat_rate=repeat_rate,
                session_length=decisions,
                latency_per_decision_sec=latency,
                fast_hint_coverage=hint_cov,
                avg_fast_hint_count=avg_hint_cnt,
                click_item_counter=click_counter,
                pces_trajectory=pces_traj,
                action_sequence=action_sequence,
                transition_matrix=transition_matrix,
                recovered_rounds=recovered_rounds,
                avg_psy_metrics={k: v for k, v in avg_psy_metrics.items() if v is not None},
                psy_history=psy_hist,
                metadata=run_meta.get(run_id, {}),
            )
        )

    return out


# ------------------------------
# Core metric helpers
# ------------------------------


def _aggregate_ctr(runs: list[RunRecord]) -> float | None:
    dec = sum(x.decisions for x in runs)
    clk = sum(x.clicks for x in runs)
    return _safe_div(float(clk), float(dec))


def _aggregate_skip_rate(runs: list[RunRecord]) -> float | None:
    dec = sum(x.decisions for x in runs)
    sk = sum(x.skips for x in runs)
    return _safe_div(float(sk), float(dec))


def _aggregate_repeat_rate(runs: list[RunRecord]) -> float | None:
    num = 0.0
    den = 0.0
    for r in runs:
        if r.repeat_rate is None:
            continue
        num += r.repeat_rate * float(r.clicks)
        den += float(r.clicks)
    return _safe_div(num, den)


def _aggregate_latency_per_decision(runs: list[RunRecord]) -> float | None:
    vals = [r.latency_per_decision_sec for r in runs if r.latency_per_decision_sec is not None]
    return _mean([float(x) for x in vals]) if vals else None


def _aggregate_action_sequence(runs: list[RunRecord]) -> list[str]:
    out: list[str] = []
    for r in runs:
        out.extend(r.action_sequence)
    return out


def _aggregate_click_counter(runs: list[RunRecord]) -> Counter[str]:
    c: Counter[str] = Counter()
    for r in runs:
        c.update(r.click_item_counter)
    return c


def _build_popularity_bucket_dist(
    item_counter: Counter[str],
    item_popularity: dict[str, float] | None,
) -> dict[str, float]:
    if not item_counter:
        return {}

    bucket_counter: Counter[str] = Counter()

    if item_popularity:
        pop = {str(k): _to_float(v) for k, v in item_popularity.items()}
        vals = sorted(pop.values())
        if not vals:
            item_popularity = None
        else:
            q1 = _quantile(vals, 1 / 3)
            q2 = _quantile(vals, 2 / 3)
            for item_id, cnt in item_counter.items():
                score = pop.get(str(item_id), q1)
                if score <= q1:
                    b = "low"
                elif score <= q2:
                    b = "mid"
                else:
                    b = "high"
                bucket_counter[b] += int(cnt)
            return _counter_to_prob(bucket_counter)

    # fallback: use rank-by-frequency as popularity proxy
    items_sorted = [k for k, _ in item_counter.most_common()]
    n = len(items_sorted)
    if n <= 0:
        return {}
    top_cut = max(1, int(math.ceil(n * 0.2)))
    mid_cut = max(top_cut + 1, int(math.ceil(n * 0.6)))
    top_set = set(items_sorted[:top_cut])
    mid_set = set(items_sorted[top_cut:mid_cut])

    for item_id, cnt in item_counter.items():
        if item_id in top_set:
            b = "high"
        elif item_id in mid_set:
            b = "mid"
        else:
            b = "low"
        bucket_counter[b] += int(cnt)

    return _counter_to_prob(bucket_counter)


def _derive_real_pop_bucket(
    ref: dict[str, Any],
) -> dict[str, float] | None:
    direct = ref.get("real_popularity_bucket_dist")
    if isinstance(direct, dict):
        return _normalize_prob_like({str(k): _to_float(v) for k, v in direct.items()})

    real_exposure = ref.get("real_item_exposure")
    real_pop = ref.get("real_item_popularity")
    if isinstance(real_exposure, dict) and isinstance(real_pop, dict):
        pseudo_counter = Counter({str(k): int(round(_to_float(v) * 100000)) for k, v in real_exposure.items()})
        return _build_popularity_bucket_dist(pseudo_counter, {str(k): _to_float(v) for k, v in real_pop.items()})

    return None


def _ab_metrics(cases: list[AbCase], no_effect_delta: float = 1e-3) -> dict[str, float | None]:
    if not cases:
        return {
            "Sign Accuracy": None,
            "Lift MAE": None,
            "Type-I Error": None,
            "Power@δ(1%)": None,
            "Power@δ(3%)": None,
            "Power@δ(5%)": None,
            "Lift RMSE": None,
            "Regret": None,
        }

    oracle = [c.oracle_lift for c in cases]
    pred = [c.pred_lift for c in cases]
    n = len(cases)

    sign_hit = 0
    mae = 0.0
    mse = 0.0
    regret = 0.0

    def _sign(x: float) -> int:
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    for o, p in zip(oracle, pred):
        so = _sign(o)
        sp = _sign(p)
        sign_hit += 1 if so == sp else 0
        e = p - o
        mae += abs(e)
        mse += e * e
        if so != 0 and sp != so:
            regret += abs(o)

    no_eff_idx = [i for i, o in enumerate(oracle) if abs(o) <= no_effect_delta]
    reject_idx = [i for i, p in enumerate(pred) if abs(p) > no_effect_delta]
    if no_eff_idx:
        false_pos = sum(1 for i in no_eff_idx if i in reject_idx)
        type1 = _safe_div(float(false_pos), float(len(no_eff_idx)))
    else:
        type1 = None

    def _power(th: float) -> float | None:
        idx = [i for i, o in enumerate(oracle) if abs(o) >= th]
        if not idx:
            return None
        ok = 0
        for i in idx:
            so = _sign(oracle[i])
            sp = _sign(pred[i])
            ok += 1 if sp == so else 0
        return _safe_div(float(ok), float(len(idx)))

    return {
        "Sign Accuracy": _safe_div(float(sign_hit), float(n)),
        "Lift MAE": _safe_div(mae, float(n)),
        "Type-I Error": type1,
        "Power@δ(1%)": _power(0.01),
        "Power@δ(3%)": _power(0.03),
        "Power@δ(5%)": _power(0.05),
        "Lift RMSE": math.sqrt(_safe_div(mse, float(n)) or 0.0),
        "Regret": _safe_div(regret, float(n)),
    }


def _cost_per_correct_from_cases(cases: list[AbCase]) -> float | None:
    if not cases:
        return None
    with_cost = [c for c in cases if c.cost is not None]
    if not with_cost:
        return None

    def _sign(x: float) -> int:
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    correct = 0
    total_cost = 0.0
    for c in with_cost:
        total_cost += float(c.cost or 0.0)
        if _sign(c.oracle_lift) == _sign(c.pred_lift):
            correct += 1
    if correct <= 0:
        return None
    return total_cost / float(correct)


# ------------------------------
# E1: Behavior Fidelity
# ------------------------------


def _compute_behavior_block(runs: list[RunRecord], ref: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    missing: list[str] = []
    if not runs:
        return {
            "CTR gap": None,
            "Item exposure JS": None,
            "PCES trajectory MSE": None,
            "Skip-rate gap": None,
            "Session length error": None,
            "Repeat rate gap": None,
            "Popularity-bucket JS": None,
            "Transition-matrix distance": None,
        }, ["no runs"]

    sim_ctr = _aggregate_ctr(runs)
    sim_skip = _aggregate_skip_rate(runs)
    sim_repeat = _aggregate_repeat_rate(runs)
    sim_sess = _mean([float(r.session_length) for r in runs])

    sim_exposure = _counter_to_prob(_aggregate_click_counter(runs))
    real_exposure = ref.get("real_item_exposure")
    if isinstance(real_exposure, dict):
        real_exposure = _normalize_prob_like({str(k): _to_float(v) for k, v in real_exposure.items()})
    else:
        real_exposure = None

    sim_pces = _flatten_trajectory([r.pces_trajectory for r in runs])
    real_pces = ref.get("real_pces_trajectory")
    if isinstance(real_pces, list):
        real_pces = [
            [float(y) for y in x]
            for x in real_pces
            if isinstance(x, list) and len(x) >= 4
        ]
    else:
        real_pces = None

    real_transition = _normalize_transition_input(ref.get("real_transition_matrix"))
    sim_transition = _transition_matrix(_aggregate_action_sequence(runs))

    sim_pop_bucket = _build_popularity_bucket_dist(
        _aggregate_click_counter(runs),
        (
            {str(k): _to_float(v) for k, v in (ref.get("real_item_popularity") or {}).items()}
            if isinstance(ref.get("real_item_popularity"), dict)
            else None
        ),
    )
    real_pop_bucket = _derive_real_pop_bucket(ref)

    ctr_gap = None
    if sim_ctr is not None and ref.get("real_ctr") is not None:
        ctr_gap = abs(sim_ctr - _to_float(ref.get("real_ctr")))
    else:
        missing.append("real_ctr")

    exposure_js = None
    if real_exposure is not None:
        exposure_js = _js_divergence(sim_exposure, real_exposure)
    else:
        missing.append("real_item_exposure")

    pces_mse = None
    if real_pces is not None:
        pces_mse = _trajectory_mse(sim_pces, real_pces)
    else:
        missing.append("real_pces_trajectory")

    skip_gap = None
    if sim_skip is not None and ref.get("real_skip_rate") is not None:
        skip_gap = abs(sim_skip - _to_float(ref.get("real_skip_rate")))
    else:
        missing.append("real_skip_rate")

    session_err = None
    if sim_sess is not None and ref.get("real_session_length") is not None:
        session_err = abs(sim_sess - _to_float(ref.get("real_session_length")))
    else:
        missing.append("real_session_length")

    repeat_gap = None
    if sim_repeat is not None and ref.get("real_repeat_rate") is not None:
        repeat_gap = abs(sim_repeat - _to_float(ref.get("real_repeat_rate")))
    else:
        missing.append("real_repeat_rate")

    pop_js = None
    if real_pop_bucket is not None:
        pop_js = _js_divergence(sim_pop_bucket, real_pop_bucket)
    else:
        missing.append("real_popularity_bucket_dist or (real_item_exposure + real_item_popularity)")

    trans_dist = None
    if real_transition is not None:
        trans_dist = _transition_distance(sim_transition, real_transition)
    else:
        missing.append("real_transition_matrix")

    block = {
        "CTR gap": ctr_gap,
        "Item exposure JS": exposure_js,
        "PCES trajectory MSE": pces_mse,
        "Skip-rate gap": skip_gap,
        "Session length error": session_err,
        "Repeat rate gap": repeat_gap,
        "Popularity-bucket JS": pop_js,
        "Transition-matrix distance": trans_dist,
        "simulated_summary": {
            "sim_ctr": sim_ctr,
            "sim_skip_rate": sim_skip,
            "sim_repeat_rate": sim_repeat,
            "sim_session_length": sim_sess,
            "sim_transition_matrix": sim_transition,
            "sim_popularity_bucket_dist": sim_pop_bucket,
            "total_runs": len(runs),
        },
    }
    return block, sorted(set(missing))


def experiment_e1(runs: list[RunRecord], reference: dict[str, Any]) -> dict[str, Any]:
    if not runs:
        return {"status": "insufficient_data", "reason": "No runs found under runs_root."}

    global_ref = _extract_ref_global(reference)
    global_block, global_missing = _compute_behavior_block(runs, global_ref)

    by_dataset: dict[str, Any] = {}
    for ds in sorted(set(r.dataset for r in runs if r.dataset)):
        sub = [r for r in runs if r.dataset == ds]
        ds_ref = _get_ref_for_dataset(reference, ds)
        block, missing = _compute_behavior_block(sub, ds_ref)
        by_dataset[ds] = {
            "run_count": len(sub),
            "metrics": block,
            "missing_inputs": missing,
        }

    return {
        "status": "ok",
        "goal": "Behavior fidelity with psychology trajectory alignment",
        "main_metrics": {
            "CTR gap": global_block.get("CTR gap"),
            "Item exposure JS": global_block.get("Item exposure JS"),
            "PCES trajectory MSE": global_block.get("PCES trajectory MSE"),
        },
        "diagnostics": {
            "Skip-rate gap": global_block.get("Skip-rate gap"),
            "Session length error": global_block.get("Session length error"),
            "Repeat rate gap": global_block.get("Repeat rate gap"),
            "Popularity-bucket JS": global_block.get("Popularity-bucket JS"),
            "Transition-matrix distance": global_block.get("Transition-matrix distance"),
        },
        "global": global_block,
        "by_dataset": by_dataset,
        "missing_inputs": global_missing,
    }


# ------------------------------
# E2: Online A/B accuracy
# ------------------------------


def experiment_e2(ab_cases: list[AbCase], reference: dict[str, Any]) -> dict[str, Any]:
    if not ab_cases:
        return {
            "status": "insufficient_data",
            "reason": "No A/B cases found. Provide --ab-cases-file or reference.ab_cases.",
            "required_fields": ["oracle_lift", "pred_lift", "budget(optional)", "seed(optional)", "cost(optional)"],
        }

    budgets = [200, 500, 1000, 2000, 5000]
    by_budget: dict[str, Any] = {}
    for b in budgets:
        sub = [x for x in ab_cases if x.budget == b]
        if not sub:
            by_budget[str(b)] = {"case_count": 0, "metrics": _ab_metrics([])}
            continue
        by_budget[str(b)] = {
            "case_count": len(sub),
            "metrics": _ab_metrics(sub),
        }

    no_budget_cases = [x for x in ab_cases if x.budget is None]
    if no_budget_cases:
        by_budget["unlabeled_budget"] = {
            "case_count": len(no_budget_cases),
            "metrics": _ab_metrics(no_budget_cases),
        }

    overall = _ab_metrics(ab_cases)

    cost_per_correct = _cost_per_correct_from_cases(ab_cases)
    if cost_per_correct is None and reference.get("avg_run_cost") is not None and overall.get("Sign Accuracy") not in (None, 0):
        cost_per_correct = _to_float(reference.get("avg_run_cost")) / float(overall.get("Sign Accuracy") or 1.0)

    return {
        "status": "ok",
        "goal": "Online A/B winner identification under finite traffic budgets",
        "overall": overall,
        "by_budget": by_budget,
        "efficiency": {
            "Cost per correct A/B decision": cost_per_correct,
        },
        "diagnostics": {
            "Lift RMSE": overall.get("Lift RMSE"),
            "Regret": overall.get("Regret"),
        },
        "notes": [
            "Sign Accuracy and Lift MAE are the primary E2 metrics.",
            "Budget curves can be plotted from by_budget[*].metrics.",
        ],
    }


# ------------------------------
# E3: Cross-domain & heterogeneity
# ------------------------------


def _run_level_behavior_metrics(run: RunRecord, reference: dict[str, Any]) -> dict[str, float | None]:
    ref = _get_ref_for_dataset(reference, run.dataset)
    ctr_gap = None
    if run.ctr is not None and ref.get("real_ctr") is not None:
        ctr_gap = abs(float(run.ctr) - _to_float(ref.get("real_ctr")))

    item_js = None
    real_exp = ref.get("real_item_exposure")
    if isinstance(real_exp, dict):
        real_exp_norm = _normalize_prob_like({str(k): _to_float(v) for k, v in real_exp.items()})
        item_js = _js_divergence(_counter_to_prob(run.click_item_counter), real_exp_norm)

    return {
        "CTR gap": ctr_gap,
        "Item exposure JS": item_js,
    }


def experiment_e3(runs: list[RunRecord], ab_cases: list[AbCase], reference: dict[str, Any]) -> dict[str, Any]:
    if not runs:
        return {"status": "insufficient_data", "reason": "No run memories found."}

    # Cross-domain table by dataset
    datasets = sorted(set(r.dataset for r in runs if r.dataset))
    table3: dict[str, Any] = {}
    for ds in datasets:
        ds_runs = [r for r in runs if r.dataset == ds]
        e1_ds = _compute_behavior_block(ds_runs, _get_ref_for_dataset(reference, ds))[0]
        ds_ab = [c for c in ab_cases if c.dataset == ds] if ab_cases else []
        table3[ds] = {
            "run_count": len(ds_runs),
            "ab_case_count": len(ds_ab),
            "A/B": _ab_metrics(ds_ab),
            "Behavior": {
                "CTR gap": e1_ds.get("CTR gap"),
                "Item exposure JS": e1_ds.get("Item exposure JS"),
            },
        }

    # Heterogeneity by user group (from run metadata group OR ab case group)
    group_runs: dict[str, list[RunRecord]] = defaultdict(list)
    for r in runs:
        g = r.metadata.get("group")
        if g is not None and str(g).strip():
            group_runs[str(g).strip()].append(r)

    group_cases: dict[str, list[AbCase]] = defaultdict(list)
    for c in ab_cases:
        if c.group:
            group_cases[c.group].append(c)

    hetero: dict[str, Any] = {}
    all_groups = sorted(set(group_runs.keys()) | set(group_cases.keys()))
    for g in all_groups:
        br = group_runs.get(g, [])
        bc = group_cases.get(g, [])
        beh = None
        if br:
            ctr_gaps = []
            js_vals = []
            for r in br:
                m = _run_level_behavior_metrics(r, reference)
                if m["CTR gap"] is not None:
                    ctr_gaps.append(float(m["CTR gap"]))
                if m["Item exposure JS"] is not None:
                    js_vals.append(float(m["Item exposure JS"]))
            beh = {
                "CTR gap": _mean(ctr_gaps),
                "Item exposure JS": _mean(js_vals),
            }
        hetero[g] = {
            "run_count": len(br),
            "ab_case_count": len(bc),
            "A/B": _ab_metrics(bc),
            "Behavior": beh,
        }

    # worst-group performance
    wg_sign: float | None = None
    wg_ctr_gap: float | None = None
    sign_vals = [
        _to_float(v.get("A/B", {}).get("Sign Accuracy"))
        for v in hetero.values()
        if v.get("A/B", {}).get("Sign Accuracy") is not None
    ]
    if sign_vals:
        wg_sign = min(sign_vals)
    ctr_vals = [
        _to_float(v.get("Behavior", {}).get("CTR gap"))
        for v in hetero.values()
        if isinstance(v.get("Behavior"), dict) and v["Behavior"].get("CTR gap") is not None
    ]
    if ctr_vals:
        wg_ctr_gap = max(ctr_vals)

    # cross-domain degradation if domain_split labels exist in AB cases
    splits: dict[str, list[AbCase]] = defaultdict(list)
    for c in ab_cases:
        key = None
        if c.variant and c.variant.lower() in {"in_domain", "cross_domain", "few_shot"}:
            key = c.variant.lower()
        if c.perturbation and c.perturbation.lower() in {"in_domain", "cross_domain", "few_shot"}:
            key = c.perturbation.lower()
        if key:
            splits[key].append(c)

    cross_deg: dict[str, float | None] = {
        "Sign Accuracy degradation (cross vs in-domain)": None,
        "Sign Accuracy degradation (few-shot vs in-domain)": None,
    }
    in_m = _ab_metrics(splits.get("in_domain", [])) if splits.get("in_domain") else None
    cross_m = _ab_metrics(splits.get("cross_domain", [])) if splits.get("cross_domain") else None
    few_m = _ab_metrics(splits.get("few_shot", [])) if splits.get("few_shot") else None
    if in_m and cross_m:
        b = in_m.get("Sign Accuracy")
        p = cross_m.get("Sign Accuracy")
        if b is not None and p is not None and abs(b) > 1e-12:
            cross_deg["Sign Accuracy degradation (cross vs in-domain)"] = (b - p) / abs(b)
    if in_m and few_m:
        b = in_m.get("Sign Accuracy")
        p = few_m.get("Sign Accuracy")
        if b is not None and p is not None and abs(b) > 1e-12:
            cross_deg["Sign Accuracy degradation (few-shot vs in-domain)"] = (b - p) / abs(b)

    notes = []
    if not all_groups:
        notes.append("No group labels found (run metadata[group] or ab_cases[group]).")
    if not splits:
        notes.append("No in_domain/cross_domain/few_shot labels found; degradation is unavailable.")

    return {
        "status": "ok",
        "goal": "Cross-domain generalization and heterogeneity robustness",
        "table3_cross_domain": table3,
        "group_breakdown": hetero,
        "worst_group_performance": {
            "A/B Sign Accuracy (higher is better)": wg_sign,
            "Behavior CTR gap (lower is better)": wg_ctr_gap,
        },
        "cross_domain_degradation": cross_deg,
        "notes": notes,
    }


# ------------------------------
# E4: Mechanism ablation
# ------------------------------


def _pick_baseline_label(labels: list[str], marked_base: list[str]) -> str | None:
    if marked_base:
        return sorted(set(marked_base))[0]
    for p in ["full", "base", "baseline", "default", "psyber_full"]:
        for l in labels:
            if l.lower() == p:
                return l
    return sorted(labels)[0] if labels else None


def experiment_e4(runs: list[RunRecord], ab_cases: list[AbCase], reference: dict[str, Any]) -> dict[str, Any]:
    if not runs and not ab_cases:
        return {"status": "insufficient_data", "reason": "No runs and no A/B cases."}

    # Build variant buckets. Priority: metadata.variant -> decision_mode
    run_by_variant: dict[str, list[RunRecord]] = defaultdict(list)
    base_marks: list[str] = []
    for r in runs:
        v = str(r.metadata.get("variant") or r.decision_mode or "unknown")
        run_by_variant[v].append(r)
        if bool(r.metadata.get("is_base", False)):
            base_marks.append(v)

    case_by_variant: dict[str, list[AbCase]] = defaultdict(list)
    for c in ab_cases:
        v = c.variant or "unknown"
        case_by_variant[v].append(c)
        if c.is_base:
            base_marks.append(v)

    all_variants = sorted(set(run_by_variant.keys()) | set(case_by_variant.keys()))
    baseline = _pick_baseline_label(all_variants, base_marks)

    table: dict[str, Any] = {}
    for v in all_variants:
        vruns = run_by_variant.get(v, [])
        vcases = case_by_variant.get(v, [])

        # behavior + psychology + efficiency from runs
        ctr_gaps = []
        js_vals = []
        pces_vals = []
        lat_vals = []
        for r in vruns:
            ref = _get_ref_for_dataset(reference, r.dataset)
            if r.ctr is not None and ref.get("real_ctr") is not None:
                ctr_gaps.append(abs(float(r.ctr) - _to_float(ref.get("real_ctr"))))
            if isinstance(ref.get("real_item_exposure"), dict):
                js = _js_divergence(
                    _counter_to_prob(r.click_item_counter),
                    _normalize_prob_like(
                        {str(k): _to_float(val) for k, val in (ref.get("real_item_exposure") or {}).items()}
                    ),
                )
                if js is not None:
                    js_vals.append(js)
            if isinstance(ref.get("real_pces_trajectory"), list):
                real_pces = [
                    [float(y) for y in x]
                    for x in ref.get("real_pces_trajectory")
                    if isinstance(x, list) and len(x) >= 4
                ]
                mse = _trajectory_mse(r.pces_trajectory, real_pces)
                if mse is not None:
                    pces_vals.append(mse)
            if r.latency_per_decision_sec is not None:
                lat_vals.append(float(r.latency_per_decision_sec))

        abm = _ab_metrics(vcases)
        cost_per_correct = _cost_per_correct_from_cases(vcases)

        table[v] = {
            "run_count": len(vruns),
            "ab_case_count": len(vcases),
            "A/B": {
                "Sign Accuracy": abm.get("Sign Accuracy"),
                "Lift MAE": abm.get("Lift MAE"),
                "Type-I Error": abm.get("Type-I Error"),
            },
            "Behavior": {
                "CTR gap": _mean(ctr_gaps),
                "Item exposure JS": _mean(js_vals),
            },
            "Psychology": {
                "PCES trajectory MSE": _mean(pces_vals),
            },
            "Efficiency": {
                "Latency/decision": _mean(lat_vals),
                "Cost per correct A/B decision": cost_per_correct,
            },
        }

    delta_vs_base: dict[str, Any] = {}
    if baseline and baseline in table:
        b = table[baseline]
        for v in all_variants:
            if v == baseline:
                continue
            t = table[v]
            delta_vs_base[v] = {
                "Sign Accuracy Drop%": _drop_pct(
                    b["A/B"].get("Sign Accuracy"), t["A/B"].get("Sign Accuracy"), high_is_better=True
                ),
                "Lift MAE Drop%": _drop_pct(
                    b["A/B"].get("Lift MAE"), t["A/B"].get("Lift MAE"), high_is_better=False
                ),
                "CTR gap Drop%": _drop_pct(
                    b["Behavior"].get("CTR gap"), t["Behavior"].get("CTR gap"), high_is_better=False
                ),
                "Item exposure JS Drop%": _drop_pct(
                    b["Behavior"].get("Item exposure JS"), t["Behavior"].get("Item exposure JS"), high_is_better=False
                ),
                "PCES trajectory MSE Drop%": _drop_pct(
                    b["Psychology"].get("PCES trajectory MSE"),
                    t["Psychology"].get("PCES trajectory MSE"),
                    high_is_better=False,
                ),
                "Latency/decision Drop%": _drop_pct(
                    b["Efficiency"].get("Latency/decision"),
                    t["Efficiency"].get("Latency/decision"),
                    high_is_better=False,
                ),
            }

    notes = []
    if len(all_variants) <= 1:
        notes.append("Only one variant found. Add metadata.variant or ab_cases.variant for full ablation.")

    return {
        "status": "ok",
        "goal": "Mechanism necessity via module ablation",
        "baseline_variant": baseline,
        "ablation_table": table,
        "delta_vs_baseline": delta_vs_base,
        "notes": notes,
    }


# ------------------------------
# E5: Scalability & cost
# ------------------------------


def experiment_e5(
    batch_summaries: list[dict[str, Any]],
    runs: list[RunRecord],
    ab_cases: list[AbCase],
) -> dict[str, Any]:
    if not batch_summaries:
        return {
            "status": "insufficient_data",
            "reason": "No batch launch summaries found. Expected batch_launches/**/summary.json",
            "required": [
                "src/launch_websim_batch.py outputs",
                "summary.json with stats.throughput/success/timeout",
            ],
        }

    # Build launch_tag -> sign_accuracy from AB cases when available
    launch_sign: dict[str, float | None] = {}
    cases_by_launch: dict[str, list[AbCase]] = defaultdict(list)
    for c in ab_cases:
        if c.launch_tag:
            cases_by_launch[c.launch_tag].append(c)
    for tag, cs in cases_by_launch.items():
        launch_sign[tag] = _ab_metrics(cs).get("Sign Accuracy")

    rows = []
    for s in batch_summaries:
        tag = str(s.get("launch_tag") or "")
        rows.append(
            {
                "launch_tag": tag,
                "N_agents": s.get("total_tasks"),
                "throughput_tasks_per_sec": s.get("throughput_tasks_per_sec"),
                "success_rate": s.get("success_rate"),
                "timeout_rate": s.get("timeout_rate"),
                "latency_per_task_sec": s.get("latency_per_task_sec"),
                "Sign Accuracy (if linked AB)": launch_sign.get(tag),
                "summary_path": s.get("path"),
            }
        )

    # Latency/decision from runs as global efficiency anchor
    lat_dec = _aggregate_latency_per_decision(runs) if runs else None

    # iso-quality / iso-cost placeholders when cost exists
    # We try to infer a "cost" from cases linked by launch_tag
    launch_cost: dict[str, float] = defaultdict(float)
    for c in ab_cases:
        if c.launch_tag and c.cost is not None:
            launch_cost[c.launch_tag] += float(c.cost)

    iso_quality = None
    if launch_cost and launch_sign:
        target = 0.8
        feasible = [(tag, cost) for tag, cost in launch_cost.items() if launch_sign.get(tag) is not None and (launch_sign.get(tag) or 0.0) >= target]
        if feasible:
            best = min(feasible, key=lambda x: x[1])
            iso_quality = {
                "target_sign_accuracy": target,
                "min_cost_launch_tag": best[0],
                "min_cost": best[1],
            }

    iso_cost = None
    if launch_cost and launch_sign:
        # choose median cost as budget and report best quality under budget
        vals = sorted(launch_cost.values())
        budget = vals[len(vals) // 2]
        feasible = [(tag, launch_sign.get(tag)) for tag, c in launch_cost.items() if c <= budget and launch_sign.get(tag) is not None]
        if feasible:
            best = max(feasible, key=lambda x: x[1] or -1)
            iso_cost = {
                "budget_cost": budget,
                "best_launch_tag": best[0],
                "best_sign_accuracy": best[1],
            }

    notes = []
    if not launch_sign:
        notes.append("No launch_tag-linked A/B cases; iso-cost/iso-quality are limited.")
    if not launch_cost:
        notes.append("No per-case cost in AB cases; cost-quality analysis is unavailable.")

    return {
        "status": "ok",
        "goal": "Scalability, throughput, reliability, and cost-quality tradeoffs",
        "efficiency_table": rows,
        "headline_metrics": {
            "Latency/decision (from runs)": lat_dec,
            "Throughput": _mean([_to_float(x.get("throughput_tasks_per_sec")) for x in batch_summaries]),
            "Success rate": _mean([_to_float(x.get("success_rate")) for x in batch_summaries if x.get("success_rate") is not None]),
            "Timeout rate": _mean([_to_float(x.get("timeout_rate")) for x in batch_summaries if x.get("timeout_rate") is not None]),
        },
        "iso_cost": iso_cost,
        "iso_quality": iso_quality,
        "notes": notes,
    }


# ------------------------------
# E6: Robustness
# ------------------------------


def _group_label_for_run(r: RunRecord) -> str:
    if r.metadata.get("perturbation"):
        return str(r.metadata.get("perturbation"))
    # Fallback label encodes mode + track to expose simple perturbation dimensions.
    return f"mode={r.decision_mode}|track={r.track_target}"


def _group_label_for_case(c: AbCase) -> str:
    if c.perturbation:
        return c.perturbation
    if c.variant:
        return c.variant
    return "default"


def _base_label_from_runs_cases(run_groups: dict[str, list[RunRecord]], case_groups: dict[str, list[AbCase]]) -> str:
    marked = []
    for g, rr in run_groups.items():
        if any(bool(x.metadata.get("is_base", False)) for x in rr):
            marked.append(g)
    for g, cc in case_groups.items():
        if any(x.is_base for x in cc):
            marked.append(g)
    if marked:
        return sorted(set(marked))[0]
    for p in ["base", "baseline", "default"]:
        for g in sorted(set(run_groups.keys()) | set(case_groups.keys())):
            if g.lower() == p:
                return g
    return sorted(set(run_groups.keys()) | set(case_groups.keys()))[0]


def experiment_e6(runs: list[RunRecord], ab_cases: list[AbCase], reference: dict[str, Any]) -> dict[str, Any]:
    if not runs and not ab_cases:
        return {"status": "insufficient_data", "reason": "No runs and no A/B cases for robustness."}

    run_groups: dict[str, list[RunRecord]] = defaultdict(list)
    for r in runs:
        run_groups[_group_label_for_run(r)].append(r)

    case_groups: dict[str, list[AbCase]] = defaultdict(list)
    for c in ab_cases:
        case_groups[_group_label_for_case(c)].append(c)

    labels = sorted(set(run_groups.keys()) | set(case_groups.keys()))
    if not labels:
        return {"status": "insufficient_data", "reason": "No perturbation labels inferred."}

    base = _base_label_from_runs_cases(run_groups, case_groups)

    table: dict[str, Any] = {}
    for g in labels:
        rr = run_groups.get(g, [])
        cc = case_groups.get(g, [])

        ctr_gaps = []
        js_vals = []
        pces_vals = []
        lat_vals = []
        for r in rr:
            ref = _get_ref_for_dataset(reference, r.dataset)
            if r.ctr is not None and ref.get("real_ctr") is not None:
                ctr_gaps.append(abs(float(r.ctr) - _to_float(ref.get("real_ctr"))))
            if isinstance(ref.get("real_item_exposure"), dict):
                js = _js_divergence(
                    _counter_to_prob(r.click_item_counter),
                    _normalize_prob_like(
                        {str(k): _to_float(v) for k, v in (ref.get("real_item_exposure") or {}).items()}
                    ),
                )
                if js is not None:
                    js_vals.append(js)
            if isinstance(ref.get("real_pces_trajectory"), list):
                real = [
                    [float(y) for y in x]
                    for x in ref.get("real_pces_trajectory")
                    if isinstance(x, list) and len(x) >= 4
                ]
                mse = _trajectory_mse(r.pces_trajectory, real)
                if mse is not None:
                    pces_vals.append(mse)
            if r.latency_per_decision_sec is not None:
                lat_vals.append(float(r.latency_per_decision_sec))

        # AB metrics by seed, then mean±ci across seeds (if available)
        by_seed: dict[str, list[AbCase]] = defaultdict(list)
        for c in cc:
            by_seed[c.seed or "_single"] .append(c)

        sign_list: list[float] = []
        mae_list: list[float] = []
        type1_list: list[float] = []
        p1_list: list[float] = []
        p3_list: list[float] = []
        p5_list: list[float] = []
        if by_seed:
            for seed_cases in by_seed.values():
                m = _ab_metrics(seed_cases)
                if m.get("Sign Accuracy") is not None:
                    sign_list.append(float(m.get("Sign Accuracy") or 0.0))
                if m.get("Lift MAE") is not None:
                    mae_list.append(float(m.get("Lift MAE") or 0.0))
                if m.get("Type-I Error") is not None:
                    type1_list.append(float(m.get("Type-I Error") or 0.0))
                if m.get("Power@δ(1%)") is not None:
                    p1_list.append(float(m.get("Power@δ(1%)") or 0.0))
                if m.get("Power@δ(3%)") is not None:
                    p3_list.append(float(m.get("Power@δ(3%)") or 0.0))
                if m.get("Power@δ(5%)") is not None:
                    p5_list.append(float(m.get("Power@δ(5%)") or 0.0))

        table[g] = {
            "run_count": len(rr),
            "ab_case_count": len(cc),
            "A/B (mean±95%CI)": {
                "Sign Accuracy": _mean_ci95(sign_list),
                "Lift MAE": _mean_ci95(mae_list),
                "Type-I Error": _mean_ci95(type1_list),
                "Power@δ(1%)": _mean_ci95(p1_list),
                "Power@δ(3%)": _mean_ci95(p3_list),
                "Power@δ(5%)": _mean_ci95(p5_list),
            },
            "Behavior (mean±95%CI)": {
                "CTR gap": _mean_ci95(ctr_gaps),
                "Item exposure JS": _mean_ci95(js_vals),
            },
            "Psychology (mean±95%CI)": {
                "PCES trajectory MSE": _mean_ci95(pces_vals),
            },
            "Efficiency (mean±95%CI)": {
                "Latency/decision": _mean_ci95(lat_vals),
                "Cost per correct A/B decision": _cost_per_correct_from_cases(cc),
            },
        }

    b = table.get(base, {})

    def _mean_of(metric_block: dict[str, Any], metric_name: str) -> float | None:
        x = metric_block.get(metric_name)
        if isinstance(x, dict):
            return x.get("mean")
        if isinstance(x, (int, float)):
            return float(x)
        return None

    drop: dict[str, Any] = {}
    for g in labels:
        if g == base:
            continue
        tg = table[g]
        drop[g] = {
            "Sign Accuracy Drop%": _drop_pct(
                _mean_of(b.get("A/B (mean±95%CI)", {}), "Sign Accuracy"),
                _mean_of(tg.get("A/B (mean±95%CI)", {}), "Sign Accuracy"),
                high_is_better=True,
            ),
            "Lift MAE Drop%": _drop_pct(
                _mean_of(b.get("A/B (mean±95%CI)", {}), "Lift MAE"),
                _mean_of(tg.get("A/B (mean±95%CI)", {}), "Lift MAE"),
                high_is_better=False,
            ),
            "CTR gap Drop%": _drop_pct(
                _mean_of(b.get("Behavior (mean±95%CI)", {}), "CTR gap"),
                _mean_of(tg.get("Behavior (mean±95%CI)", {}), "CTR gap"),
                high_is_better=False,
            ),
            "Item exposure JS Drop%": _drop_pct(
                _mean_of(b.get("Behavior (mean±95%CI)", {}), "Item exposure JS"),
                _mean_of(tg.get("Behavior (mean±95%CI)", {}), "Item exposure JS"),
                high_is_better=False,
            ),
            "PCES trajectory MSE Drop%": _drop_pct(
                _mean_of(b.get("Psychology (mean±95%CI)", {}), "PCES trajectory MSE"),
                _mean_of(tg.get("Psychology (mean±95%CI)", {}), "PCES trajectory MSE"),
                high_is_better=False,
            ),
            "Latency/decision Drop%": _drop_pct(
                _mean_of(b.get("Efficiency (mean±95%CI)", {}), "Latency/decision"),
                _mean_of(tg.get("Efficiency (mean±95%CI)", {}), "Latency/decision"),
                high_is_better=False,
            ),
        }

    # worst case: min for high-is-better, max for low-is-better
    sign_means = [
        _mean_of(v.get("A/B (mean±95%CI)", {}), "Sign Accuracy")
        for v in table.values()
    ]
    sign_means = [x for x in sign_means if x is not None]
    mae_means = [
        _mean_of(v.get("A/B (mean±95%CI)", {}), "Lift MAE")
        for v in table.values()
    ]
    mae_means = [x for x in mae_means if x is not None]

    worst_case = {
        "Sign Accuracy (min over perturbations)": min(sign_means) if sign_means else None,
        "Lift MAE (max over perturbations)": max(mae_means) if mae_means else None,
    }

    return {
        "status": "ok",
        "goal": "Robustness under model/prompt/seed/track perturbations",
        "base_condition": base,
        "mean_ci_table": table,
        "drop_percent_vs_base": drop,
        "worst_case_performance": worst_case,
        "notes": [
            "Perturbation labels come from run_metadata[perturbation] or ab_cases[perturbation/variant].",
            "A/B CI is computed across seeds when seed labels are available.",
        ],
    }


# ------------------------------
# E7: Qualitative trajectory analysis
# ------------------------------


def _case_ab_info(r: RunRecord, ab_cases: list[AbCase]) -> dict[str, Any] | None:
    # Priority: exact run_id match in ab_cases
    cs = [c for c in ab_cases if c.run_id and c.run_id == r.run_id]
    if not cs:
        # fallback by dataset if only one record
        ds = [c for c in ab_cases if c.dataset and c.dataset == r.dataset]
        if len(ds) == 1:
            cs = ds
    if not cs:
        return None

    c = cs[0]
    so = 1 if c.oracle_lift > 0 else (-1 if c.oracle_lift < 0 else 0)
    sp = 1 if c.pred_lift > 0 else (-1 if c.pred_lift < 0 else 0)
    if abs(c.oracle_lift) <= 1e-3:
        decision = "inconclusive"
    elif so == sp:
        decision = "correct"
    else:
        decision = "wrong"
    return {
        "oracle_lift": c.oracle_lift,
        "pred_lift": c.pred_lift,
        "decision": decision,
        "pair": c.pair,
    }


def _build_case_payload(r: RunRecord, label: str, ab_info: dict[str, Any] | None) -> dict[str, Any]:
    traj = []
    for s in r.psy_history:
        metrics = s.get("metrics") or {}
        state = s.get("state") or {}
        action = _infer_action(s)
        traj.append(
            {
                "step": int(s.get("step") or 0),
                "timestamp_utc": s.get("timestamp_utc"),
                "action": action,
                "selected_item_id": str(s.get("selected_item_id") or ""),
                "selected_title": str(s.get("selected_title") or ""),
                "pces": {
                    "P": _to_float(state.get("P", 0.0)),
                    "C": _to_float(state.get("C", 0.0)),
                    "E": _to_float(state.get("E", 0.0)),
                    "S": _to_float(state.get("S", 0.0)),
                },
                "psych_intermediate": {
                    "Nov_t": _to_float(metrics.get("Nov_t", 0.0)),
                    "Bel_t": _to_float(metrics.get("Bel_t", 0.0)),
                    "Offense_t": _to_float(metrics.get("Offense_t", 0.0)),
                },
                "strategy_source": (
                    "fast_policy"
                    if _to_float(metrics.get("mode_fast_policy_hint", 0.0)) > 0.0
                    else "llm_fallback"
                ),
                "fast_policy_hint_count": _to_float(metrics.get("fast_policy_hint_count", 0.0)),
            }
        )

    return {
        "label": label,
        "run_id": r.run_id,
        "dataset": r.dataset,
        "model": r.model,
        "decision_mode": r.decision_mode,
        "track_target": r.track_target,
        "summary": {
            "decisions": r.decisions,
            "clicks": r.clicks,
            "skips": r.skips,
            "ctr": r.ctr,
            "skip_rate": r.skip_rate,
            "repeat_rate": r.repeat_rate,
            "recovered_rounds": r.recovered_rounds,
            "latency_per_decision_sec": r.latency_per_decision_sec,
            "fast_hint_coverage": r.fast_hint_coverage,
            "avg_fast_hint_count": r.avg_fast_hint_count,
        },
        "ab_judgement": ab_info,
        "trajectory": traj,
        "interpretation_hint": (
            "Check action switches and PCES trend around high Offense_t/low Bel_t periods to diagnose errors."
        ),
    }


def experiment_e7(runs: list[RunRecord], ab_cases: list[AbCase], max_cases: int = 5) -> dict[str, Any]:
    if not runs:
        return {"status": "insufficient_data", "reason": "No runs for qualitative analysis."}

    # Candidate selectors
    selected: list[tuple[str, RunRecord]] = []
    used: set[str] = set()

    def _pick(label: str, cand: RunRecord | None) -> None:
        if cand is None:
            return
        if cand.run_id in used:
            return
        selected.append((label, cand))
        used.add(cand.run_id)

    # recovery case
    rec = max(runs, key=lambda r: (r.recovered_rounds, r.decisions), default=None)
    _pick("state_recovery_case", rec)

    # high-confidence success-like case
    high_ctr = max(runs, key=lambda r: (_to_float(r.ctr, -1.0), r.clicks), default=None)
    _pick("success_like_case", high_ctr)

    # failure/mispred-like: highest skip rate
    high_skip = max(runs, key=lambda r: (_to_float(r.skip_rate, -1.0), r.skips), default=None)
    _pick("failure_like_case", high_skip)

    # boundary-like: ctr near median
    ctr_vals = sorted([_to_float(r.ctr, 0.0) for r in runs])
    if ctr_vals:
        med = ctr_vals[len(ctr_vals) // 2]
        boundary = min(runs, key=lambda r: abs(_to_float(r.ctr, 0.0) - med))
        _pick("boundary_like_case", boundary)

    # fill to max_cases by longest trajectory
    for r in sorted(runs, key=lambda x: x.decisions, reverse=True):
        if len(selected) >= max_cases:
            break
        _pick("additional_case", r)

    cases = [_build_case_payload(r, label, _case_ab_info(r, ab_cases)) for label, r in selected[:max_cases]]

    return {
        "status": "ok",
        "goal": "Qualitative trajectory analysis for success/failure/boundary/recovery patterns",
        "case_count": len(cases),
        "cases": cases,
        "notes": [
            "If ab_cases includes run_id mapping, each case includes explicit A/B judgement.",
            "Use exported trajectory to draw timeline and PCES line charts.",
        ],
    }


# ------------------------------
# CLI
# ------------------------------


EXPERIMENT_ID_TO_KEY = {
    "E1": "E1_behavior_fidelity",
    "E2": "E2_online_ab_accuracy",
    "E3": "E3_cross_domain_heterogeneity",
    "E4": "E4_ablation",
    "E5": "E5_scalability_cost",
    "E6": "E6_robustness",
    "E7": "E7_qualitative_trajectory",
}


def _parse_experiment_selection(raw: str) -> list[str]:
    txt = (raw or "").strip()
    if not txt or txt.lower() == "all":
        return list(EXPERIMENT_ID_TO_KEY.keys())

    picked: list[str] = []
    for part in txt.split(","):
        p = part.strip().upper()
        if not p:
            continue
        if p not in EXPERIMENT_ID_TO_KEY:
            raise SystemExit(
                f"Invalid experiment id: {p}. "
                f"Supported: {', '.join(EXPERIMENT_ID_TO_KEY.keys())}, or all."
            )
        if p not in picked:
            picked.append(p)
    if not picked:
        return list(EXPERIMENT_ID_TO_KEY.keys())
    return picked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PsyBer experiment suite E1-E7")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--batch-root", type=Path, default=Path("batch_launches"))
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON for real behavior references. Supports legacy keys "
            "(real_ctr, real_item_exposure, real_pces_trajectory, ab_cases, avg_run_cost) "
            "and extended by_dataset/behavior_reference format."
        ),
    )
    parser.add_argument(
        "--ab-cases-file",
        type=Path,
        default=None,
        help="Optional JSON for A/B cases. If omitted, fallback to reference.ab_cases.",
    )
    parser.add_argument(
        "--run-metadata-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping run_id->metadata. Useful fields: "
            "variant, group, perturbation, is_base."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/reports"))
    parser.add_argument("--output-name", type=str, default="")
    parser.add_argument("--max-e7-cases", type=int, default=5)
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        help=(
            "Comma-separated experiment ids to run, e.g. E1,E3,E7. "
            "Use 'all' to run E1-E7."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root.expanduser().resolve()
    batch_root = args.batch_root.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not runs_root.exists():
        raise SystemExit(f"runs_root not found: {runs_root}")

    run_meta = _load_run_metadata(args.run_metadata_file.expanduser().resolve() if args.run_metadata_file else None)
    reference = _load_reference(args.reference_file.expanduser().resolve() if args.reference_file else None)
    ab_cases = _load_ab_cases(args.ab_cases_file.expanduser().resolve() if args.ab_cases_file else None, reference)
    batch_summaries = _load_batch_summaries(batch_root)
    runs = _load_runs(runs_root, run_meta)

    selected_experiments = _parse_experiment_selection(args.experiments)
    section_results: dict[str, Any] = {}

    if "E1" in selected_experiments:
        section_results[EXPERIMENT_ID_TO_KEY["E1"]] = experiment_e1(runs, reference)
    if "E2" in selected_experiments:
        section_results[EXPERIMENT_ID_TO_KEY["E2"]] = experiment_e2(ab_cases, reference)
    if "E3" in selected_experiments:
        section_results[EXPERIMENT_ID_TO_KEY["E3"]] = experiment_e3(runs, ab_cases, reference)
    if "E4" in selected_experiments:
        section_results[EXPERIMENT_ID_TO_KEY["E4"]] = experiment_e4(runs, ab_cases, reference)
    if "E5" in selected_experiments:
        section_results[EXPERIMENT_ID_TO_KEY["E5"]] = experiment_e5(batch_summaries, runs, ab_cases)
    if "E6" in selected_experiments:
        section_results[EXPERIMENT_ID_TO_KEY["E6"]] = experiment_e6(runs, ab_cases, reference)
    if "E7" in selected_experiments:
        section_results[EXPERIMENT_ID_TO_KEY["E7"]] = experiment_e7(
            runs,
            ab_cases,
            max_cases=max(1, args.max_e7_cases),
        )

    now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_name.strip():
        out_name = args.output_name.strip()
    elif len(selected_experiments) == 1:
        out_name = f"experiment_{selected_experiments[0]}_{now_tag}.json"
    else:
        out_name = f"experiment_suite_{now_tag}.json"
    out_path = out_dir / out_name

    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "runs_root": str(runs_root),
            "batch_root": str(batch_root),
            "run_count": len(runs),
            "ab_case_count": len(ab_cases),
            "batch_summary_count": len(batch_summaries),
            "reference_file": str(args.reference_file) if args.reference_file else None,
            "ab_cases_file": str(args.ab_cases_file) if args.ab_cases_file else None,
            "run_metadata_file": str(args.run_metadata_file) if args.run_metadata_file else None,
            "selected_experiments": selected_experiments,
        },
    }
    payload.update(section_results)

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Done] report={out_path}")
    print(
        "[Info] "
        f"runs={len(runs)} ab_cases={len(ab_cases)} batch_summaries={len(batch_summaries)} "
        f"experiments={','.join(selected_experiments)}"
    )


if __name__ == "__main__":
    main()
