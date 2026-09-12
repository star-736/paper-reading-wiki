---
type: Model
title: "DeepSeekMath"
description: "DeepSeek-AI 的 7B 数学推理模型族（Base / Instruct / RL），从 DeepSeek-Coder-Base-v1.5 继续预训练，RL 变体是 GRPO 的发布检查点；纯文本。"
tags: ["model", "deepseekmath", "grpo", "math-reasoning"]
timestamp: 2026-09-12
---

# DeepSeekMath

## 身份

DeepSeek-AI 在 2024-04 发布的 7B 数学推理模型族。从 DeepSeek-Coder-Base-v1.5 7B 初始化，用 120B web math + code / 自然语言继续预训练，再经数学指令微调与 GRPO。权重与代码见 [DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math)。

它同时是 **GRPO 的发布检查点**：算法定义在来源页，不要另建 GRPO 模型或概念页。

## 关键事实

| 属性 | DeepSeekMath-Base | DeepSeekMath-Instruct | DeepSeekMath-RL |
| --- | --- | --- | --- |
| 总参数 | 7B dense | 同 | 同 |
| **模态** | 纯文本 | 纯文本 | 纯文本 |
| 初始化 | DeepSeek-Coder-Base-v1.5 7B | Base | Instruct |
| 训练 | 500B tokens 继续预训练 | 776K 中英数学 SFT | GRPO，约 144K GSM8K+MATH CoT 题 |
| 无工具 MATH | 36.2%（few-shot CoT） | 46.8% | **51.7%**（64 样本 SC 60.9%） |
| 无工具 GSM8K | 64.2% | 82.9% | **88.2%** |
| 来源 | [DeepSeekMath](../sources/deepseekmath.md)（arXiv:2402.03300v3） | 同 | 同 |

**模态**：已据原文核实为纯文本。评测含「用 Python 写程序解题」和 Isabelle 形式化，那是**生成代码 / 形式证明文本**，不是图像或工具 API 输入。

## 技术身份

DeepSeekMath 不是新注意力架构。它在 2024 年的位置是：

1. **数据**：迭代 fastText 从 Common Crawl 挖 120B 数学网页，证明公开网页足够训出接近 Minerva 540B 的 7B 基座。
2. **初始化**：code 基座优于 general LLM；纯 arXiv 在本报告 benchmark 上几乎无用。
3. **RL**：GRPO 丢掉 PPO critic，用同题组内相对奖励。作者用 Maj@K↑ / Pass@K≈ 解释涨分来自分布校准，不是基本能力扩张。

2026 的 agentic / RLVR 栈几乎都还在用这条 group-relative 目标，但奖励已换成可验证规则，规模也不再是 7B 数学 SFT 子集。读后续报告时把「GRPO 算法」和「DeepSeekMath-RL 这个检查点」分开。

## 相关页面

- 来源：[DeepSeekMath](../sources/deepseekmath.md)
- 比较：[LLM RL policy optimization 对比](../comparisons/llm-rl-policy-optimization.md)
- 概念：[Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)
