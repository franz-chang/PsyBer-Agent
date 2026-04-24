# PsyBer-Agent：A psychological behavior-driven user Agent

![What is PayBer-Agent](Method.jpg "PsyBerAgent")

PsyBer-Agent是一个基于心理驱动的推荐系统用户agent框架。通过对角色、认知和行为动力学进行建模，模拟真实的、长时间的用户交互，并通过轻量级的交互推荐基准仿真平台WebSim支持部署前的评估。该项目提供了一个统一的环境，用于研究用户行为的保真度、在线a /B测试的可靠性、模型评估和跨多个推荐场景的鲁棒性，包括电影、购物、图文和短视频推荐。

## 环境配置

1. 创建 `psyber` 环境步骤如下（可直接复制）：

```bash
# 1) 创建并激活环境
conda create -y -n psyber python=3.11 pip
conda activate psyber

# 2) 安装依赖
pip install -U pip
pip install -r requirements.txt

# 3) 安装 Playwright 浏览器（给 WebSim_agent 用）
playwright install chromium
```

验证命令：

```bash
python src/prelearning_agent.py --help
python src/alignment_pretrain_agent.py --help
python src/WebSim_agent.py --help
python app.py
```

后续如果你改了环境，重新导出可用：

```bash
conda run -n psyber pip freeze | LC_ALL=C sort > /Users/chongzhang/PsyBer-Agent/requirements.txt
```

## 实验平台配置

这里需要使用到WebSim平台：[https://github.com/franz-chang/Web_simulation.git](https://github.com/franz-chang/Web_simulation.git "配置平台")

```bash
git clone https://github.com/franz-chang/Web_simulation.git
cd /Users/chongzhang/WebSim
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` 当前依赖：

- `flask==3.0.2`
- `torch>=2.6.0`
- `pandas>=2.2.0`
- `numpy>=1.26.0`

## 功能结构

- 从 `~/PsyBer-Agent/settings/task.yaml` 读取任务设置：`dataset`、`model`、`track`、`agent_num`（兼容 `Settings/task.yaml`）
- 通过 `~/WebSim/run_service.sh`确保服务运行在 `http://127.0.0.1:19002/`
- 打开网页后自动：
  - 选择数据集与模型
  - 点击“随机重置”
  - 截取当前页面截图，由 Agent 决定点击哪个 item（或不感兴趣时点击“下一页”）
- 连续记录点击序列到 Agent memory，达到 `track`（track值可设置）后关闭浏览器
- 新增离线对齐预训练阶段：`src/alignment_pretrain_agent.py` 会从 `ratings.dat` 重建隐式轨迹，进行 OT 对齐训练，并输出在线快速策略模型 `fast_inference_policy.pt`（默认保存到 `PsyBer-Agent/outputs/fast_policies/{ml1m|amazon_mi}/fast_inference_policy.pt`）
- 在线阶段优先使用快速策略推理（5 动作并行评估 + 帕累托过滤 + 轻量策略采样）；最终通过Prompt汇总以上结果让LLM定夺决策

## 预训练阶段

离线预处理流水线（建议顺序）：

```bash
cd /Users/chongzhang/PsyBer-Agent
./Script/01_run_prelearning_agent.sh         # 数据预处理
./Script/02_run_alignment_pretrain_agent.sh  # 心理-行为对齐
```

说明：`02_run_alignment_pretrain_agent.sh` 默认启用 `softlabel_ugw`，即先训练回归先验生成 UGW 初始对齐，再由结构对齐生成动作软标签做监督训练。

## 在线模拟用户阶段：

```bash
/PsyBer-Agent/src/WebSim_agent.py       # 用户模拟Agent
```

可选参数：

- `--headless`: 无界面运行
- `--skip-probability 0.2`: “不感兴趣 -> 下一页”的概率
- `--seed 42`: 随机种子
- `--track-override 10`: 本次运行临时覆盖 track，便于快速测试
- `--close-service-when-done`: 若本次由 Agent 启动服务，结束时自动关闭服务
- `--disable-fast-inference`: 关闭快速策略推理，强制走纯LLM-driven决策路径
- `--fast-policy-path <path>`: 指定快速策略模型文件（默认自动按数据集路径寻找）
- `--fast-policy-stochastic`: 快速策略按概率采样动作（默认贪心）
- `--ablation-wo-template-random-selection`: 消融实验预留，关闭随机模板/候选选择，改为确定性选择

## API 池（高并发推荐）

为避免高并发时所有 Agent 挤占同一个 API，项目已支持 **API 池**：

- 配置文件路径：`./PsyBer-Agent/settings/api_pool.yaml`（兼容 `Settings/api_pool.yaml`）
- 每个任务启动前从池中获取一个 API 槽位（`api_id`）
- 任务结束后自动释放该 API 槽位
- 当池中无可用 API 时，任务会等待，不会抢同一个 API

`api_pool.yaml` 格式示例：

```yaml
defaults:
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  qwen_model: "qwen3-vl-flash"

apis:
  - id: "api-001"
    api_key: "sk-xxxx"
    enabled: true
  - id: "api-002"
    api_key: "sk-yyyy"
    enabled: true
    capacity: 1
```

说明：

- 在执行高并发（用户量级大于1k）时，会读取：`api_pool.yaml`
- 若池文件缺失或没有可用 API，脚本会直接失败（避免退回单 API 造成拒绝服务）
- `capacity` 默认为 `1`，表示同一个 API 同时最多被 1 个任务占用
- 你可以准备约 200 个 `apis` 条目来支撑大规模并发

## 结果输出

每次运行会生成：

- `./PsyBer-Agent/runs/<timestamp>/memory.json`
- `./PsyBer-Agent/runs/<timestamp>/summary.txt`
- `./PsyBer-Agent/runs/<timestamp>/web_service.log`（仅当由 Agent 启动服务）
- `summary.txt` 中会额外记录 `api_id`（当任务来自 API 池分配时）。

## task.yaml 说明

如果 `./PsyBer-Agent/settings/task.yaml` 不存在，Agent 会自动创建默认配置（兼容 `Settings/task.yaml`）：

```yaml
dataset:
  - MovieLens-1M
  # - Amazon All Beauty
  # - Amazon Magazine Subscriptions
model:
  # - LightGCN
  # - PopRec
  # - BPR-MF
  # - GRU4Rec
  - BERT4Rec
  # - SASRec
  # - Mult-VAE
track: 30
agent_num: 200
```

可以通过选择取消注释来确定具体实验任务，具体可支持的数据集和model参考[WebSim实验平台](https://github.com/franz-chang/Web_simulation.git "配置平台")
