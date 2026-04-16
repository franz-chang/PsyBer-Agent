# Evaluation

`evaluation/` now provides two entry scripts:

- `evaluate_memory_metrics.py`: lightweight aggregate metrics (legacy single-report mode).
- `run_experiment_suite.py`: full `E1-E7` experiment suite aligned with `ref/PsyBer-Agent实验设计.pdf`.

## 1) Quick run (full E1-E7)

```bash
cd /Users/chongzhang/PsyBer-Agent
python evaluation/run_experiment_suite.py
```

Output:
- `evaluation/reports/experiment_suite_*.json`

This command works with only `runs/*/memory.json`. Missing external inputs will be
reported per experiment section as `insufficient_data` or `missing_inputs`.

## 2) Full-input run (recommended)

```bash
python evaluation/run_experiment_suite.py \
  --reference-file evaluation/experiment_reference_example.json \
  --ab-cases-file evaluation/ab_cases_example.json \
  --run-metadata-file evaluation/run_metadata_example.json
```

## 3) Inputs

### `--reference-file` (behavior/psychology ground truth)

Supports:

- Legacy keys (same as old script):
  - `real_ctr`
  - `real_item_exposure`
  - `real_pces_trajectory`
  - `ab_cases`
  - `avg_run_cost`
- Extended structure:
  - `behavior_reference.global`
  - `behavior_reference.by_dataset`
  - optional diagnostics references:
    - `real_skip_rate`
    - `real_session_length`
    - `real_repeat_rate`
    - `real_transition_matrix`
    - `real_popularity_bucket_dist` or (`real_item_exposure` + `real_item_popularity`)

See `evaluation/experiment_reference_example.json`.

### `--ab-cases-file` (A/B judgments)

Expected fields per case:

- Required:
  - `oracle_lift`
  - `pred_lift`
- Optional (for E2/E3/E4/E5/E6/E7):
  - `budget`, `seed`, `dataset`, `group`, `variant`, `perturbation`, `launch_tag`, `pair`, `run_id`, `cost`, `is_base`

See `evaluation/ab_cases_example.json`.

### `--run-metadata-file` (run-level labels)

Supported format:

- `{ "run_meta": { "<run_id>": { ... } } }`

Useful fields:

- `variant` (E4)
- `group` (E3)
- `perturbation`, `is_base` (E6)
- `domain_split` (optional, for your own downstream slicing)

See `evaluation/run_metadata_example.json`.

## 4) Legacy aggregate script

```bash
python evaluation/evaluate_memory_metrics.py --reference-file evaluation/reference_example.json
```

Output:
- `evaluation/reports/memory_eval_*.json`

## 5) Shell scripts (integrated pipeline + single experiment)

### 5.1 One-command full pipeline (recommended)

Run E1-E7, then memory metrics, then generate markdown report:

```bash
cd /Users/chongzhang/PsyBer-Agent
bash evaluation/run_all_experiments.sh
```

Default outputs:

- `evaluation/reports/experiment_suite_<timestamp>.json`
- `evaluation/reports/memory_eval_<timestamp>.json`
- `evaluation/reports/preliminary_experiment_results.md`

Optional env vars:

- `PYTHON_BIN`
- `RUNS_ROOT`
- `BATCH_ROOT`
- `OUTPUT_DIR` / `REPORTS_DIR`
- `REFERENCE_FILE` / `AB_CASES_FILE` / `RUN_METADATA_FILE`
- `MEMORY_REFERENCE_FILE`
- `OUTPUT_MD`
- `MAX_E7_CASES`
- `MAX_RUNS`

### 5.2 Run only E1-E7 suite

You can run experiments directly with shell wrappers in `evaluation/steps/`:

```bash
cd /Users/chongzhang/PsyBer-Agent

# run all E1-E7
bash evaluation/run_experiment.sh all

# run single experiment
bash evaluation/steps/run_e1.sh
bash evaluation/steps/run_e2.sh
bash evaluation/steps/run_e3.sh
bash evaluation/steps/run_e4.sh
bash evaluation/steps/run_e5.sh
bash evaluation/steps/run_e6.sh
bash evaluation/steps/run_e7.sh
```

Optional env vars (all wrappers support):

- `REFERENCE_FILE`
- `AB_CASES_FILE`
- `RUN_METADATA_FILE`
- `RUNS_ROOT`
- `BATCH_ROOT`
- `OUTPUT_DIR`
- `MAX_E7_CASES`
- `PYTHON_BIN`

Example:

```bash
REFERENCE_FILE=evaluation/experiment_reference_example.json \
AB_CASES_FILE=evaluation/ab_cases_example.json \
RUN_METADATA_FILE=evaluation/run_metadata_example.json \
bash evaluation/run_all_experiments.sh
```

## 6) Generate preliminary markdown report

Generate markdown like:

- `evaluation/reports/preliminary_experiment_results.md`

Command:

```bash
cd /Users/chongzhang/PsyBer-Agent
bash evaluation/run_preliminary_report.sh
```

Optional env vars:

- `SUITE_REPORT` (explicit suite json)
- `MEMORY_REPORT` (explicit memory eval json)
- `RUNS_ROOT`
- `REPORTS_DIR`
- `OUTPUT_MD`
- `MAX_RUNS`

Example:

```bash
SUITE_REPORT=evaluation/reports/experiment_suite_20260415_174409.json \
MEMORY_REPORT=evaluation/reports/memory_eval_post_suite_check.json \
OUTPUT_MD=evaluation/reports/preliminary_experiment_results.md \
bash evaluation/run_preliminary_report.sh
```

## 7) Step scripts folder

Each pipeline step now has its own startup script under:

- `evaluation/steps/01_run_experiment_suite.sh`
- `evaluation/steps/02_run_memory_eval.sh`
- `evaluation/steps/03_generate_markdown_report.sh`
- `evaluation/steps/04_plot_e1_item_distribution.sh`
- `evaluation/steps/run_e1.sh` ... `evaluation/steps/run_e7.sh` (single experiment wrappers)

You can run these step-by-step manually, or call `evaluation/run_all_experiments.sh` for end-to-end execution.

## 8) E1 item distribution plot (agent vs offline logs)

For preliminary E1 behavior-fidelity visualization, compare:

- Agent click-item distribution from `runs/*/memory.json`
- Offline item distribution from `ratings.dat` (ML-1M)

Run:

```bash
cd /Users/chongzhang/PsyBer-Agent
bash evaluation/steps/04_plot_e1_item_distribution.sh
```

Default output files in `evaluation/reports/`:

- `e1_item_dist_ml1m_top30_<timestamp>.png`
- `e1_item_dist_ml1m_rank_<timestamp>.png`
- `e1_item_dist_ml1m_metrics_<timestamp>.json`

Optional env vars:

- `RUNS_ROOT`
- `OUTPUT_DIR`
- `DATASET_KEY` (default: `ml1m`)
- `RATING_THRESHOLD` (default: `0`, means all ratings)
- `TOP_K` (default: `30`)
- `OFFLINE_RATINGS_FILE` (override default ML-1M path)
- `OFFLINE_TITLES_FILE` (override default `movies.dat`)
- `OUTPUT_PREFIX`
