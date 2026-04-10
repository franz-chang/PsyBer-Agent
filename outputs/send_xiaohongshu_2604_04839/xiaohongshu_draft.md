# 小红书草稿｜MERIT 论文

📚文章题目｜MERIT：面向中文中心低资源机器翻译的多语专家奖励调优
原题：MERIT: Multilingual Expert-Reward Informed Tuning for Chinese-Centric Low-Resource Machine Translation

📚作者信息｜Zhixiang Lu、Chong Zhang、Chenyu Xue、Angelos Stefanidis、Chong Li、Jionglong Su、Zhengyong Jiang
单位：Xi’an Jiaotong-Liverpool University（西交利物浦大学）

📚研究背景｜中文到东南亚低资源语言的机器翻译一直卡在“高质量平行语料太少、噪声数据太多”。这篇工作瞄准中文中心场景，想证明在低资源翻译里，数据质量和奖励优化比一味堆大模型更关键。

📚本文主要贡献｜
1. 提出了 CALT 基准：把原本偏英语中心的 ALT 数据重构成中文中心评测集，覆盖 5 种东南亚低资源语言，更适合直接评估“低资源语言→中文”翻译。
2. 提出了 MERIT 框架：把语言特定前缀（LTP）、监督微调（SFT）和基于语义对齐奖励（SAR）的 GRPO 结合起来，形成一套从数据筛选到模型优化的完整流程。
3. 做了奖励驱动的数据蒸馏：先训练 QE 代理给翻译打分，再把高质量样本筛出来训练，让“小而精”的数据真正服务低资源翻译。

📚实验总结｜MERIT-3B 只用了原始训练数据的 22.8%，整体 BLEU-chrF 依然从 15.53 提升到 18.23，最终达到 19.94。它不仅明显超过 NLLB-200 3.3B、M2M-100 等开源基线，也说明低资源翻译里“更干净的数据 + 更对齐的奖励”确实比单纯扩大参数量更有效。

📚标签｜#机器翻译 #低资源语言 #多语言大模型 #奖励建模 #数据蒸馏 #GRPO #NLP #大模型论文 #论文阅读 #AI科研 🐶
