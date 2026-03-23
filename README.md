# PsyBer-Agent Web_sim CAMEL Agent

基于 CAMEL `HybridBrowserToolkit` 的网页点击模拟 Agent。

## 功能

- 从 `/Users/chongzhang/PsyBer-Agent/task.yaml` 读取任务设置：`dataset`、`model`、`track`、`agent_num`
- 通过 `/Users/chongzhang/Web_sim/run_service.sh` 确保服务运行在 `http://127.0.0.1:19002/`
- 打开网页后自动：
  - 选择数据集与模型
  - 点击“随机重置”
  - 截取当前页面截图，由 Agent 决定点击哪个 item（或不感兴趣时点击“下一页”）
- 连续记录点击序列到 Agent memory，达到 `track`（默认 100）后关闭浏览器

> 说明：截图决策使用 OpenAI Responses API，模型固定为 `gpt-5.1-codex-mini`。  
> 你可以二选一配置 API：  
> 1) 环境变量 `OPENAI_API_KEY`（可选 `OPENAI_BASE_URL`）  
> 2) 直接在 `web_sim_camel_agent.py` 顶部填写 `OPENAI_API_KEY_OVERRIDE` / `OPENAI_BASE_URL_OVERRIDE`。  
> 若 API 不可用，会自动回退到随机策略，避免任务中断。

## 运行

```bash
cd /Users/chongzhang/PsyBer-Agent
/Users/chongzhang/camel/.venv/bin/python /Users/chongzhang/PsyBer-Agent/web_sim_camel_agent.py
```

可选参数：

- `--headless`: 无界面运行
- `--skip-probability 0.2`: “不感兴趣 -> 下一页”的概率
- `--seed 42`: 随机种子
- `--track-override 10`: 本次运行临时覆盖 track，便于快速测试
- `--close-service-when-done`: 若本次由 Agent 启动服务，结束时自动关闭服务

## 结果输出

每次运行会生成：

- `/Users/chongzhang/PsyBer-Agent/runs/<timestamp>/memory.json`
- `/Users/chongzhang/PsyBer-Agent/runs/<timestamp>/summary.txt`
- `/Users/chongzhang/PsyBer-Agent/runs/<timestamp>/web_service.log`（仅当由 Agent 启动服务）

## task.yaml 说明

如果 `/Users/chongzhang/PsyBer-Agent/task.yaml` 不存在，Agent 会自动创建默认配置：

```yaml
dataset:
  - MovieLens-1M
  - Amazon Musical Instruments
model:
  - SASRec
  - LIghtGCN
  - Multi-VAE
track: 100
agent_num: 100
```

> 其中 dataset/model 若给多个候选，会按随机种子选择其中一个用于本次运行。
# PsyBer-Agent
