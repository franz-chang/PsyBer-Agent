# Decision Robustness Redesign (Measurable-First)

## 1) Why current Table 10 gets `N/A`

Current pipeline computes:

- `Sign Accuracy`, `Lift MAE`, `Type-I Error`, `Power@δ`

from `ab_cases[{condition, seed, oracle_lift, pred_lift}]`.

`N/A` appears when:

1. `ab_cases` is missing; or
2. no **null-effect** cases exist (`|oracle_lift| <= 1e-3`), so `Type-I Error` is undefined.

This is expected behavior in `experiments/robustness/metrics.py`.

---

## 2) Replace with metrics that are always measurable from runs

For decision robustness (noise/mask/ranking perturbation), use the following as primary metrics:

1. `Decision Agreement ↑` (`1 - Action Flip Rate`)
- What it measures: consistency of decisions under perturbation vs clean seed-matched run.
- Why suitable: directly tests smoothness/stability of decision policy.

2. `Policy Shift (TV) ↓`
- What it measures: total variation distance between action distributions.
- Why suitable: captures global policy drift, not only per-step mismatch.

3. `Latency per Decision ↓`
- Why suitable: robustness should not come with severe inference slowdown.

Optional secondary metrics:

- `Action Flip Rate ↓` (diagnostic counterpart of Decision Agreement)
- `CTR Gap ↓` (vs clean)
- `PCES Trajectory MSE ↓` (vs clean)
- `Completion Rate ↑`
- `Qwen Call Success Rate ↑`

These are now output by `run_robustness_suite.py` (measurable without oracle).

---

## 3) Keep A/B correctness metrics as a separate oracle-backed track

Do **not** mix oracle-dependent metrics with oracle-free robustness stability in one table.

### Track A (always runnable): Decision Stability Robustness

- Conditions: `base_clean`, mask 10/20/30, adjacent swap, top-1 replacement, distractor insertion, conflicting signals.
- Seeds: 3-5.
- Report: `Decision Agreement`, `Policy Shift`, `Latency/decision`.

### Track B (oracle-backed): A/B Conclusion Quality

Use only when you can build valid `ab_cases` with:

- positive/negative effect cases (for `Sign Accuracy`, `Power@δ`);
- **null-effect** cases (for `Type-I Error`).

Recommended null-effect construction:

1. `A vs A` same recommender, independent traffic slices/seeds;
2. matched low-effect pairs with `|oracle_lift| <= δ`.

Then report:

- `Sign Accuracy ↑`
- `Lift MAE ↓`
- `Type-I Error ↓`
- `Power@δ (1/3/5%) ↑`

---

## 4) Updated table design (recommended)

### Table 10A (Decision Stability, no oracle required)

Columns:

- `Decision Agreement ↑`
- `Policy Shift (TV) ↓`
- `Latency/Decision ↓`

### Table 10B (A/B Quality, oracle required)

Columns:

- `Sign Accuracy ↑`
- `Lift MAE ↓`
- `Type-I Error ↓`
- `Power@δ(1/3/5%) ↑`

---

## 5) Reproducible execution protocol

1. Run decision robustness:

```bash
script/14_run_decision_robustness.sh -m qwen-plus
```

2. If you have oracle AB cases:

```bash
script/14_run_decision_robustness.sh -m qwen-plus -a /abs/path/ab_cases.json
```

3. Use latest report:

- `experiments/robustness/reports/robustness_report_*.json`

and extract:

- Track A metrics from `decision_robustness[*]` directly;
- Track B metrics only when AB fields are non-`N/A`.

---

## 6) Practical recommendation for this project now

Given current data availability, publish **Track A as main decision robustness table**.
Treat oracle-backed A/B metrics as supplementary until a balanced AB case pool with null-effect cases is prepared.
