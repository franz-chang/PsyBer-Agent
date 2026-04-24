# Experiments Entrypoints

This directory provides top-level scripts for robustness experiments:

- `run_behavior_robustness.py`
- `run_system_robustness.py`
- `run_param_sensitivity.py`

Both scripts call `experiments/robustness/run_robustness_suite.py` internally and filter condition sets automatically.

## Behavior Robustness

```bash
cd /Users/chongzhang/PsyBer-Agent
QWEN_API_KEY=... python experiments/run_behavior_robustness.py \
  --headless \
  --seeds "101,202,303" \
  --base-track 20 \
  --base-dataset "MovieLens-1M" \
  --base-model "LightGCN" \
  --cross-domain-dataset "Amazon Magazine Subscriptions" \
  --decision-model "qwen-plus-0112" \
  --build-clean-ground-truth
```

## System Robustness

```bash
cd /Users/chongzhang/PsyBer-Agent
QWEN_API_KEY=... python experiments/run_system_robustness.py \
  --headless \
  --seeds "101,202,303" \
  --base-track 20 \
  --base-dataset "MovieLens-1M" \
  --base-model "LightGCN" \
  --decision-model "qwen-plus-0112" \
  --max-concurrency 120 \
  --launch-rate 15.0 \
  --task-timeout-sec 1800
```

## Quick Validation

```bash
python experiments/run_behavior_robustness.py --dry-run --headless --seeds "101"
python experiments/run_system_robustness.py --dry-run --headless --seeds "101"
python experiments/run_param_sensitivity.py --dry-run --headless
```

## Parameter Sensitivity (tau / rho / lr)

```bash
cd /Users/chongzhang/PsyBer-Agent
python experiments/run_param_sensitivity.py \
  --headless \
  --seeds "101,202,303,404,505" \
  --tau-values "0.10,0.20,0.35,0.50,0.70" \
  --rho-values "0.05,0.15,0.25,0.35" \
  --lr-values "1e-3,3e-3,5e-3,1e-2"
```

Details:

- `experiments/param_sensitivity/README.md`
