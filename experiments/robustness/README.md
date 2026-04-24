# PsyBer-Agent Robustness Experiments

本目录实现了你定义的三类鲁棒性验证：

- Behavior robustness
- Decision robustness
- System robustness

并支持统一输出 `mean ± 95% CI` 与相对退化。

## 目录结构

- `baseline_from_runs.py`
  - 使用现有 `runs/**/memory.json` 生成 baseline 报告（不重跑）。
- `robust_websim_agent.py`
  - 在 `src/WebSim_agent.py` 基础上注入扰动（prompt/history/ranking/memory）。
- `run_robustness_suite.py`
  - 批量执行 robustness 条件（可重跑模拟），并汇总 JSON + Markdown 报告。
- `metrics.py`
  - 通用指标计算。

## 1) Baseline（复用已有 runs）

```bash
cd /Users/chongzhang/PsyBer-Agent
python experiments/robustness/baseline_from_runs.py \
  --runs-root /Users/chongzhang/PsyBer-Agent/runs \
  --output-dir /Users/chongzhang/PsyBer-Agent/experiments/robustness/reports
```

## 2) Robustness 全套实验（会重跑模拟）

```bash
cd /Users/chongzhang/PsyBer-Agent
python experiments/robustness/run_robustness_suite.py \
  --headless \
  --base-track 20 \
  --base-dataset "MovieLens-1M" \
  --base-model "LightGCN" \
  --decision-model "qwen-plus-0112" \
  --cross-domain-dataset "Amazon Magazine Subscriptions" \
  --seeds "101,202,303"
```

输出：

- `experiments/robustness/reports/robustness_run_manifest_*.jsonl`
- `experiments/robustness/reports/robustness_batch_manifest_*.jsonl`
- `experiments/robustness/reports/robustness_report_*.json`
- `experiments/robustness/reports/robustness_report_*.md`

## 3) 仅预览命令（不实际执行）

```bash
python experiments/robustness/run_robustness_suite.py --dry-run --headless
```

## 4) AB 指标注入（可选）

若有 `oracle_lift/pred_lift` 数据，可提供：

```bash
python experiments/robustness/run_robustness_suite.py \
  --ab-cases-file /path/to/ab_cases.json
```

`ab_cases` 支持字段：

- `condition`（建议直接使用 condition_id）
- `seed`（可选）
- `oracle_lift`
- `pred_lift`

没有 AB 文件时，`Sign Accuracy / Lift MAE / Type-I / Power@δ` 会显示为 `N/A`。

## 5) 只看 Table 5/6/7 的 Clean 行（对齐 WebSim_Dataset ground truth）

先跑 clean：

```bash
QWEN_API_KEY=... python experiments/robustness/run_robustness_suite.py \
  --condition-ids base_clean \
  --headless \
  --base-track 20 \
  --base-dataset "MovieLens-1M" \
  --base-model "LightGCN" \
  --decision-model "qwen-plus-0112" \
  --seeds "101,202,303"
```

再生成 clean 表格行：

```bash
python experiments/robustness/clean_group_results.py \
  --dataset-root /Users/chongzhang/WebSim_Dataset \
  --dataset-name "MovieLens-1M"
```
