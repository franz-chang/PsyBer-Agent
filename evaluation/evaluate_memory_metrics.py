#!/usr/bin/env python3
"""Evaluate stored run memories with PsyBer experiment metrics.

This script follows the metric naming in:
ref/PsyBer-Agent实验设计.pdf
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class RunMetrics:
    run_id: str
    dataset: str
    model: str
    decision_mode: str
    track_target: int
    clicks: int
    skips: int
    decisions: int
    ctr_sim: float | None
    skip_rate: float | None
    repeat_rate: float | None
    latency_per_decision_sec: float | None
    fast_hint_coverage: float | None
    avg_fast_hint_count: float | None
    final_pces: dict[str, float]
    click_item_counter: Counter[str]
    pces_trajectory: list[list[float]]


def _safe_div(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return a / b


def _parse_utc_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _prob_dist_from_counter(counter: Counter[str]) -> dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0:
        return {}
    return {k: float(v) / total for k, v in counter.items()}


def _align_prob_keys(a: dict[str, float], b: dict[str, float]) -> tuple[list[float], list[float]]:
    keys = sorted(set(a.keys()) | set(b.keys()))
    pa = [float(a.get(k, 0.0)) for k in keys]
    pb = [float(b.get(k, 0.0)) for k in keys]
    return pa, pb


def _js_divergence(a: dict[str, float], b: dict[str, float], eps: float = 1e-12) -> float | None:
    if not a and not b:
        return None
    pa, pb = _align_prob_keys(a, b)

    def _kl(p: list[float], q: list[float]) -> float:
        s = 0.0
        for pi, qi in zip(p, q):
            if pi <= 0:
                continue
            s += pi * math.log((pi + eps) / (qi + eps))
        return s

    m = [(x + y) / 2.0 for x, y in zip(pa, pb)]
    return 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)


def _pces_from_state(state: dict[str, Any]) -> list[float]:
    return [
        float(state.get("P", 0.0)),
        float(state.get("C", 0.0)),
        float(state.get("E", 0.0)),
        float(state.get("S", 0.0)),
    ]


def _trajectory_mse(a: list[list[float]], b: list[list[float]]) -> float | None:
    if not a or not b:
        return None
    n = min(len(a), len(b))
    if n <= 0:
        return None
    err = 0.0
    cnt = 0
    for i in range(n):
        va = a[i]
        vb = b[i]
        m = min(len(va), len(vb))
        for j in range(m):
            d = float(va[j]) - float(vb[j])
            err += d * d
            cnt += 1
    return _safe_div(err, float(cnt))


def _compute_ab_metrics(
    ab_cases: list[dict[str, Any]],
    *,
    no_effect_delta: float = 1e-3,
) -> dict[str, float | None]:
    if not ab_cases:
        return {
            "Sign Accuracy": None,
            "Lift MAE": None,
            "Type-I Error": None,
            "Power@δ(1%)": None,
            "Power@δ(3%)": None,
            "Power@δ(5%)": None,
        }

    oracle = [float(x.get("oracle_lift", 0.0)) for x in ab_cases]
    pred = [float(x.get("pred_lift", 0.0)) for x in ab_cases]
    n = len(oracle)

    sign_hit = 0
    lift_abs = 0.0
    for o, p in zip(oracle, pred):
        lift_abs += abs(p - o)
        so = 1 if o > 0 else (-1 if o < 0 else 0)
        sp = 1 if p > 0 else (-1 if p < 0 else 0)
        sign_hit += 1 if so == sp else 0

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
            so = 1 if oracle[i] > 0 else -1
            sp = 1 if pred[i] > 0 else (-1 if pred[i] < 0 else 0)
            ok += 1 if sp == so else 0
        return _safe_div(float(ok), float(len(idx)))

    return {
        "Sign Accuracy": _safe_div(float(sign_hit), float(n)),
        "Lift MAE": _safe_div(lift_abs, float(n)),
        "Type-I Error": type1,
        "Power@δ(1%)": _power(0.01),
        "Power@δ(3%)": _power(0.03),
        "Power@δ(5%)": _power(0.05),
    }


def load_run_metrics(memory_file: Path) -> RunMetrics:
    d = json.loads(memory_file.read_text(encoding="utf-8"))
    run_id = memory_file.parent.name

    click_seq = d.get("click_sequence") or []
    skip_seq = d.get("skip_sequence") or []
    psy = d.get("psy_state_history") or []

    decisions = int(len(psy))
    clicks = int(len(click_seq))
    skips = int(len(skip_seq))

    ctr_sim = _safe_div(float(clicks), float(decisions))
    skip_rate = _safe_div(float(skips), float(decisions))

    click_items = [str(x.get("item_id") or "") for x in click_seq if str(x.get("item_id") or "")]
    click_counter = Counter(click_items)
    unique_clicks = len(click_counter)
    repeat_rate = None
    if clicks > 0:
        repeat_rate = 1.0 - (float(unique_clicks) / float(clicks))

    ts = [_parse_utc_iso(str(x.get("timestamp_utc") or "")) for x in psy]
    lat = []
    for i in range(1, len(ts)):
        if ts[i] is None or ts[i - 1] is None:
            continue
        dt = (ts[i] - ts[i - 1]).total_seconds()
        if dt >= 0:
            lat.append(float(dt))
    latency = _safe_div(sum(lat), float(len(lat))) if lat else None

    hint_hits = 0
    hint_count_sum = 0.0
    hint_count_n = 0
    for x in psy:
        metrics = x.get("metrics") or {}
        if float(metrics.get("mode_fast_policy_hint", 0.0)) > 0.0:
            hint_hits += 1
        if "fast_policy_hint_count" in metrics:
            hint_count_sum += float(metrics.get("fast_policy_hint_count", 0.0))
            hint_count_n += 1
    hint_cov = _safe_div(float(hint_hits), float(decisions))
    avg_hint_cnt = _safe_div(hint_count_sum, float(hint_count_n))

    final_state = (psy[-1].get("state") if psy else {}) or {}
    final_pces = {
        "P": float(final_state.get("P", 0.0)),
        "C": float(final_state.get("C", 0.0)),
        "E": float(final_state.get("E", 0.0)),
        "S": float(final_state.get("S", 0.0)),
    }
    pces_traj = [_pces_from_state((x.get("state") or {})) for x in psy]

    return RunMetrics(
        run_id=run_id,
        dataset=str(d.get("selected_dataset_key") or ""),
        model=str(d.get("selected_model_key") or ""),
        decision_mode=str(d.get("decision_mode") or ""),
        track_target=int(d.get("track_target") or 0),
        clicks=clicks,
        skips=skips,
        decisions=decisions,
        ctr_sim=ctr_sim,
        skip_rate=skip_rate,
        repeat_rate=repeat_rate,
        latency_per_decision_sec=latency,
        fast_hint_coverage=hint_cov,
        avg_fast_hint_count=avg_hint_cnt,
        final_pces=final_pces,
        click_item_counter=click_counter,
        pces_trajectory=pces_traj,
    )


def aggregate_metrics(
    runs: list[RunMetrics],
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    total_decisions = sum(x.decisions for x in runs)
    total_clicks = sum(x.clicks for x in runs)
    total_skips = sum(x.skips for x in runs)
    sim_ctr = _safe_div(float(total_clicks), float(total_decisions))
    sim_skip_rate = _safe_div(float(total_skips), float(total_decisions))

    rep_num = 0.0
    rep_den = 0.0
    for x in runs:
        if x.repeat_rate is None:
            continue
        rep_num += x.repeat_rate * float(x.clicks)
        rep_den += float(x.clicks)
    repeat_rate = _safe_div(rep_num, rep_den)

    lat_num = 0.0
    lat_den = 0.0
    for x in runs:
        if x.latency_per_decision_sec is None:
            continue
        lat_num += x.latency_per_decision_sec * float(max(0, x.decisions - 1))
        lat_den += float(max(0, x.decisions - 1))
    latency = _safe_div(lat_num, lat_den)

    hint_num = 0.0
    hint_den = 0.0
    for x in runs:
        if x.fast_hint_coverage is None:
            continue
        hint_num += x.fast_hint_coverage * float(x.decisions)
        hint_den += float(x.decisions)
    hint_coverage = _safe_div(hint_num, hint_den)

    all_clicks = Counter()
    all_pces = []
    for x in runs:
        all_clicks.update(x.click_item_counter)
        all_pces.extend(x.pces_trajectory)
    sim_exposure = _prob_dist_from_counter(all_clicks)

    ref = reference or {}
    real_ctr = ref.get("real_ctr")
    real_exposure = ref.get("real_item_exposure")
    real_pces = ref.get("real_pces_trajectory")
    ab_cases = ref.get("ab_cases") or []
    avg_cost_per_run = ref.get("avg_run_cost")

    ctr_gap = None
    if real_ctr is not None and sim_ctr is not None:
        ctr_gap = abs(float(sim_ctr) - float(real_ctr))

    js = None
    if isinstance(real_exposure, dict):
        real_exp = {str(k): float(v) for k, v in real_exposure.items()}
        js = _js_divergence(sim_exposure, real_exp)

    pces_mse = None
    if isinstance(real_pces, list):
        pces_mse = _trajectory_mse(all_pces, [[float(y) for y in x] for x in real_pces])

    ab_main = _compute_ab_metrics(ab_cases)
    sign_acc = ab_main.get("Sign Accuracy")
    cost_per_correct = None
    if avg_cost_per_run is not None and sign_acc not in (None, 0):
        cost_per_correct = float(avg_cost_per_run) / float(sign_acc)

    missing_inputs = []
    if real_ctr is None:
        missing_inputs.append("real_ctr (for CTR gap)")
    if not isinstance(real_exposure, dict):
        missing_inputs.append("real_item_exposure (for Item exposure JS)")
    if not isinstance(real_pces, list):
        missing_inputs.append("real_pces_trajectory (for PCES trajectory MSE)")
    if not ab_cases:
        missing_inputs.append("ab_cases[{oracle_lift,pred_lift}] (for A/B metrics)")
    if avg_cost_per_run is None:
        missing_inputs.append("avg_run_cost (for Cost per correct A/B decision)")

    return {
        "main_metrics": {
            "A/B": ab_main,
            "Behavior": {
                "CTR gap": ctr_gap,
                "Item exposure JS": js,
            },
            "Psychology": {
                "PCES trajectory MSE": pces_mse,
            },
            "Efficiency": {
                "Latency/decision": latency,
                "Cost per correct A/B decision": cost_per_correct,
            },
        },
        "simulated_only_metrics": {
            "sim_ctr": sim_ctr,
            "sim_skip_rate": sim_skip_rate,
            "sim_repeat_rate": repeat_rate,
            "fast_hint_coverage": hint_coverage,
            "total_decisions": total_decisions,
            "total_clicks": total_clicks,
            "total_skips": total_skips,
            "unique_clicked_items": len(all_clicks),
        },
        "missing_inputs": missing_inputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PsyBer run memories")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON to enable full metrics. Supported keys: "
            "real_ctr, real_item_exposure, real_pces_trajectory, "
            "ab_cases[{oracle_lift,pred_lift}], avg_run_cost."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/reports"))
    parser.add_argument("--output-name", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    memory_files = sorted(runs_root.glob("*/memory.json"))
    if not memory_files:
        raise SystemExit(f"No memory.json found under: {runs_root}")

    runs = [load_run_metrics(p) for p in memory_files]

    reference = None
    if args.reference_file is not None:
        ref_path = args.reference_file.expanduser().resolve()
        reference = json.loads(ref_path.read_text(encoding="utf-8"))

    aggregate = aggregate_metrics(runs, reference)

    now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = args.output_name.strip() or f"memory_eval_{now_tag}.json"
    out_file = output_dir / out_name

    payload = {
        "meta": {
            "runs_root": str(runs_root),
            "run_count": len(runs),
            "reference_file": str(args.reference_file) if args.reference_file else None,
            "generated_at": datetime.now().isoformat(),
        },
        "aggregate": aggregate,
        "per_run": [
            {
                "run_id": x.run_id,
                "dataset": x.dataset,
                "model": x.model,
                "decision_mode": x.decision_mode,
                "track_target": x.track_target,
                "clicks": x.clicks,
                "skips": x.skips,
                "decisions": x.decisions,
                "ctr_sim": x.ctr_sim,
                "skip_rate": x.skip_rate,
                "repeat_rate": x.repeat_rate,
                "latency_per_decision_sec": x.latency_per_decision_sec,
                "fast_hint_coverage": x.fast_hint_coverage,
                "avg_fast_hint_count": x.avg_fast_hint_count,
                "final_pces": x.final_pces,
                "unique_clicked_items": len(x.click_item_counter),
            }
            for x in runs
        ],
    }
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Done] report={out_file}")
    print(f"[Info] run_count={len(runs)} total_decisions={payload['aggregate']['simulated_only_metrics']['total_decisions']}")
    print("[Info] main metric coverage:")
    for item in payload["aggregate"]["missing_inputs"]:
        print(f"- missing: {item}")


if __name__ == "__main__":
    main()
