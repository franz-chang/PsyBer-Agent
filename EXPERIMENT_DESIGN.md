# 实验设计

## 4. Metrics（按任务分组）

为避免不同实验口径不一致，所有实验统一先按任务分组定义主指标，再在各实验中补充必要的辅助诊断指标。

| 任务组 | 主指标 | 典型使用实验 |
| --- | --- | --- |
| A/B 判别 | `Sign Accuracy`，`Lift MAE`，`Type-I Error`，`Power@δ(1/3/5%)` | `E2`，`E3`，`E4`，`E6` |
| 行为拟真 | `CTR gap`，`Item exposure JS` | `E1`，`E3`，`E4`，`E6` |
| 心理状态拟真 | `PCES trajectory MSE` | `E1`，`E4`，`E6`，`E7` |
| 效率 | `Latency/decision`，`Cost per correct A/B decision` | `E2`，`E4`，`E5`，`E6` |

统一说明：

- `Sign Accuracy` 作为统一的 A/B 主指标，替代不统一的 `Winner Accuracy` 等命名。
- `Power@δ(1/3/5%)` 在 oracle lift 分别超过 `1%`、`3%`、`5%` 的样本子集上计算。
- `CTR gap` 与 `Item exposure JS` 是行为拟真的 headline metrics，其余行为统计一律作为辅助诊断指标。
- `PCES trajectory MSE` 是方法特有的默认心理状态指标；若需要展示时间错位鲁棒性，可将 `DTW` 放入附录诊断。

## 5. 实验细则

### E1. 行为拟真（Behavior Fidelity）（核心）

1. 目标：验证模拟行为分布是否接近真实日志。
2. 输入：test split 中真实用户行为序列。
3. 流程：
   1. 在 train split 上训练各模拟器。
   2. 对 test 用户逐个 rollout，建议使用 `track=50`、`slate=4`。
   3. 生成模拟日志，并与真实 test 日志按用户和时间窗口对齐比较。
4. 指标：
   1. 主指标（行为拟真任务组）：`CTR gap`，`Item exposure JS`。
   2. 方法特有指标（心理状态任务组）：`PCES trajectory MSE`。
   3. 辅助诊断指标：`Skip-rate gap`，`Session length error`，`Repeat rate gap`，`Popularity-bucket JS`，`Transition-matrix distance`。
5. 输出：
   1. 整体 fidelity 指标表。
   2. CTR / exposure-popularity / transition 的分布对比图。
   3. PCES 真实轨迹与模拟轨迹对比图。

### E2. 在线 A/B 判别准确性（核心）

1. 目标：在有限流量下，是否能选对推荐系统 winner。
2. 场景：真实模型对，例如 `SASRec vs LightGCN`、`SASRec vs Multi-VAE`、`LightGCN vs Multi-VAE`。
3. 流程：
   1. 先运行 `N_oracle`，得到每个 pair 的 oracle winner `winner*` 和 oracle lift `Δ*`。
   2. 在预算 `N ∈ {200, 500, 1000, 2000, 5000}` 下重复实验，并使用多个 seeds。
   3. 每次输出 A/B 结论：`B better`、`A better` 或 `inconclusive`。
4. 指标：
   1. 主指标（A/B 判别任务组）：`Sign Accuracy`，`Lift MAE`，`Type-I Error`，`Power@δ(1/3/5%)`。
   2. 效率指标（效率任务组）：`Cost per correct A/B decision`。
   3. 辅助诊断指标：`Lift RMSE`，`Regret`。
5. 输出：
   1. A/B 判别主结果表。
   2. `Sign Accuracy` vs 流量预算曲线。
   3. `Lift MAE` vs 流量预算曲线。

### E3. 跨域与异质性泛化

1. 目标：验证方法在新领域和不同用户群体上的稳定性。
2. 设置：
   1. In-domain：同域训练，同域测试。
   2. Cross-domain zero-shot：在 `ML-1M` 训练，在 `Amazon` 或 `KuaiRand` 测试。
   3. Few-shot adapt：使用目标域 `5%` 数据微调后测试。
   4. 用户分组：按画像或行为统计聚类，例如高新奇偏好、高认知负荷敏感、高活跃、长尾偏好。
3. 指标：
   1. 主指标（A/B 判别任务组）：`Sign Accuracy`，`Lift MAE`，`Type-I Error`，`Power@δ(1/3/5%)`。
   2. 主指标（行为拟真任务组）：`CTR gap`，`Item exposure JS`。
   3. 泛化与公平性辅助指标：`Worst-group performance`，跨域退化率。
4. 输出：
   1. Table 3：跨域结果表。
   2. Figure 4：分组条形图与 worst-group performance。

### E4. 模块消融（Mechanism Study）

1. 目标：证明各模块的必要性。
2. 流程：
   1. 在主数据集和 1 个外部数据集上复现 `E1 + E2` 的关键结果。
   2. 每次移除一个模块，其余设置保持不变。
3. 指标：
   1. 主指标（A/B 判别任务组）：`Sign Accuracy`，`Lift MAE`，`Type-I Error`。
   2. 主指标（行为拟真任务组）：`CTR gap`，`Item exposure JS`。
   3. 主指标（心理状态任务组）：`PCES trajectory MSE`。
   4. 主指标（效率任务组）：`Latency/decision`，必要时补充 `Cost per correct A/B decision`。
4. 输出：
   1. 主消融表。
   2. 模块贡献雷达图或性能降幅图。

### E5. 可扩展性与成本效率（Scalability & Cost）

1. 目标：验证大规模可用性和性价比。
2. 规模：`N agents = {100, 500, 1k, 5k, 10k}`，每个点重复 3 次。
3. 工程设置：
   1. 使用 `launch_websim_batch.py` 控制并发。
   2. 记录 wall-clock、吞吐、失败率和超时率。
4. 指标：
   1. 主指标（效率任务组）：`Latency/decision`，`Cost per correct A/B decision`。
   2. 辅助诊断指标：`Throughput`，`Success rate`，`Timeout rate`，`Token/API cost per 10k interactions`。
   3. 固定预算质量分析：在 iso-cost 条件下补充报告 `Sign Accuracy`。
5. 额外分析：
   1. Iso-cost：固定成本下的最佳质量。
   2. Iso-quality：达到目标 `Sign Accuracy` 所需的最小成本。
6. 输出：
   1. 效率表。
   2. Scaling curve。
   3. Cost-quality 曲线。

## Appendix：Robustness & Interpretability

### E6. 稳健性实验（Robustness Stress Test）

1. 目标：验证方法对实现细节和运行条件扰动不敏感。
2. 扰动维度：
   1. Base LLM：`qwen3-vl-flash` vs `qwen3-vl-plus`。
   2. Prompt 扰动：语义改写、指令顺序打乱、加入轻微冗余上下文。
   3. 随机种子扰动：至少 `5-10` 个 seeds。
   4. 轨迹长度扰动：`track=20/50/100`。
3. 设计：
   1. 先做单因素扰动，其余条件固定。
   2. 再做组合压力测试，即模型、prompt、seed、track 同时变化。
4. 指标：
   1. 主指标（A/B 判别任务组）：`Sign Accuracy`，`Lift MAE`，`Type-I Error`，`Power@δ(1/3/5%)`。
   2. 主指标（行为拟真任务组）：`CTR gap`，`Item exposure JS`。
   3. 主指标（心理状态任务组）：`PCES trajectory MSE`。
   4. 主指标（效率任务组）：`Latency/decision`，`Cost per correct A/B decision`。
5. 报告方式：
   1. 每项指标报告 `mean ± 95% CI`。
   2. 额外报告相对退化率 `Drop% = (perturbed - base) / base`。
   3. 单独展示 worst-case performance。

### E7. 案例分析（Qualitative Trajectory Analysis）

1. 目标：解释模型为何成功或失败，补足定量结果。
2. 样本选择：
   1. 展示 `3-5` 条典型轨迹。
   2. 覆盖成功判别、误判、真实 lift 接近 `0` 的边界样本，以及至少 1 条 “状态崩溃-恢复” 轨迹。
3. 每条案例统一展示：
   1. `step`。
   2. 候选动作。
   3. 最终动作（`click/skip`）。
   4. A/B 判别结论。
   5. PCES 状态序列。
   6. 关键心理中间量，例如 `Nov_t`、`Bel_t`、`Offense_t`。
   7. 策略来源（`fast policy` 或 `LLM fallback`）。
4. 可视化：
   1. 一张时间轴图展示动作切换。
   2. 一张 PCES 折线图展示状态演化。
   3. 一段简短的错误归因说明。
5. 结论模板：
   1. 总结何种状态模式下方法更稳定。
   2. 总结何种边界条件下更容易误判。
   3. 将这些观察映射到具体改进方向，例如阈值校准或 prompt 约束。

总实验矩阵中保持 `E1-E5` 不变，并将 `E6` 与 `E7` 追加为 `Robustness & Interpretability` 两个新增章节。
