# PsyBer-Agent 初步实验结果（基于当前已有数据）

生成时间：2026-04-15 23:46:21  
数据来源：`N/A` 与 `N/A`

## 1. 数据规模

- 运行数（run）：0
- 总决策步数（decisions）：0
- 总点击数（clicks）：0
- 总跳过数（skips）：0

## 2. 核心初步指标（当前可计算）

| 指标 | 数值 |
|---|---:|
| sim_ctr | N/A |
| sim_skip_rate | N/A |
| sim_repeat_rate | N/A |
| sim_session_length | N/A |
| Latency/decision | N/A |
| fast_hint_coverage | N/A |
| unique_clicked_items | 0 |

## 3. 行为转移统计（模拟侧）

| 转移项 | 概率 |
|---|---:|
| click -> click | 0.0000 |
| click -> skip | 0.0000 |
| skip -> click | 0.0000 |
| skip -> skip | 0.0000 |

## 4. 单次运行概览

- 暂无运行结果（未检测到 `runs/*/memory.json`）。

## 5. E1-E7 执行状态（当前数据条件）

- E1（行为拟真）：未执行。
- E2（在线 A/B 判别）：未执行。
- E3（跨域与异质性）：未执行。
- E4（模块消融）：未执行。
- E5（可扩展性与成本）：未执行。
- E6（稳健性）：未执行。
- E7（案例分析）：未执行。

## 6. 初步结论

1. 当前结果可用于验证流程可运行性，但尚缺足够数据形成稳定统计结论。

## 7. 下一步补齐建议

1. 补充 ab_cases（oracle_lift、pred_lift、budget、seed），完成 E2/E3/E4/E6 的 A/B 主结论。
2. 运行 batch launcher 并产出 batch_launches/**/summary.json，完成 E5 scaling 与 cost-quality 分析。
3. 先生成 runs/*/memory.json（例如执行 evaluation/run_all_experiments.sh 前先完成仿真）。
