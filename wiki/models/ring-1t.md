---
type: Model
title: "Ring-1T"
description: "Inclusion AI 的开源万亿 thinking MoE（1T / 约 50B 激活），从 Ling-1T-base（GQA）经 Long-CoT SFT + IcePop Reasoning RL + GRPO General RL 训出，纯文本。"
tags: ["model", "ring-1t", "moe", "thinking", "icepop"]
timestamp: 2026-09-12
---

# Ring-1T

## 身份

Inclusion AI（Ling Team）在 2025-10 发布的开源万亿 thinking 模型。从 Ling-1T-base（Ling 2.0 架构）出发，不改预训练基座，用 Long-CoT SFT → Reasoning RL（IcePop + C3PO++）→ General RL（GRPO）三段后训练做成 thinking 模型。权重公开：[Hugging Face](https://huggingface.co/inclusionAI/Ring-1T)。

不要和 [Ling-2.6 / Ring-2.6](ling-2.6.md) 混成同一个实体：Ring-2.6 是 2026-06 的后续 thinking 线，激活约 8B，注意力已 retrofit 成 7:1 Lightning Attention + MLA，RL 算法换成 KPop。

## 关键事实

| 属性 | Ring-1T |
| --- | --- |
| 总参数 | 1T |
| 激活参数 | 约 50B / token |
| **模态** | 纯文本 |
| 基座 | Ling-1T-base（Ling 2.0） |
| 注意力 | GQA（报告 Limitations 写明；本 PDF 不给头数 / 层数） |
| MoE | 是；专家数 / routed-k 本报告未给 |
| 上下文（评测） | 最长 128K（不够则 YaRN）；SFT pack 64K；Reasoning RL 最大生成 65,536 |
| 后训练 | Long-CoT SFT → IcePop Reasoning RL + C3PO++ → GRPO General RL |
| 开源 | 权重 + [Ring-V2 代码](https://github.com/inclusionAI/Ring-V2) |
| 来源 | [Ring-1T 技术报告](../sources/ring-1t.md)（arXiv:2510.18855v2） |

**模态**：已据原文核实为纯文本。Science RL 数据把分子结构等图转成结构化文本再训练（image-semantization，§2.3.1）；评测与 IMO 实验都强调「只靠自然语言推理」。报告未声称图像 / 音频输入。

## 技术身份

Ring-1T 的身份是**万亿规模 thinking RL 的可行性报告**，不是新注意力架构：

1. **稳定层**：IcePop 处理 MoE RL 里训练引擎与推理引擎的概率失配——$k=\pi_{\mathrm{train}}(\theta_{\mathrm{old}})/\pi_{\mathrm{infer}}$ 落在 $[\alpha,\beta]$ 内则校准，越界丢弃。一手出处见来源页，不要只从 Ring-2.6 对「前代 IcePop」的转述来理解。
2. **吞吐层**：C3PO++ 用 token budget 切开长 rollout，未完成轨迹跨 policy version 续跑。
3. **系统层**：ASystem（Hybrid Runtime / AMem / AState / ASandbox）把 1T RL 写成可跑的分布式系统。

作者在 Limitations 里承认：GQA 对超长 thought 的推理成本仍高；IcePop 没有做到完全 train–inference 一致；工具使用等 agentic 技能 under-optimized。后续 Ring-2.6 走的正是这三条缺口（线性注意力 retrofit、KPop、agentic RL）。

## 相关页面

- 来源：[Ring-1T 技术报告](../sources/ring-1t.md)
- 后续同族：[Ling-2.6 / Ring-2.6](ling-2.6.md)
- 概念：[训练—rollout 一致性](../concepts/train-rollout-consistency.md)、[异步 Agent RL](../concepts/asynchronous-agent-rl.md)、[Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)
- 比较：[LLM RL policy optimization 对比](../comparisons/llm-rl-policy-optimization.md)
