---
type: Model
title: "Intern-S2-Mobius"
description: "上海 AI Lab 的 35B 架构转换模型：全局共享 FFN Memory 与多层 Self-Attn Reasoner 分离知识存储和推理。"
tags: ["model", "intern-s2-mobius", "mobius", "latent-reasoning", "moe"]
timestamp: 2026-08-23
---

# Intern-S2-Mobius

## 身份

Intern-S2-Mobius 是 Shanghai AI Laboratory 提出的 Foundation Model 架构 Mobius 的 35B 级报告模型：从 Qwen3.5-35B-A3B checkpoint 架构转换、继续预训练 1T tokens，再经过 SFT 和 RL。报告另有从头训练的 **Mobius-7B-A1B MoE** 配对实验；二者不应混为同一规模或训练路线。

来源：[Intern-S2-Mobius 技术报告](../sources/intern-s2-mobius.md)。

## 关键事实

| 项 | 信息 |
| --- | --- |
| 公开名称 | Intern-S2-Mobius / Mobius-v0 |
| 团队 | Intern-S2-Mobius Team，Shanghai AI Laboratory |
| 规模 | 35B 级；从 Qwen3.5-35B-A3B 转换（总参/激活参数细节未披露） |
| 对照性小模型 | Mobius-7B-A1B MoE，从头训练、与 Transformer 配对 |
| 模态 | **纯文本（报告仅披露 token / 文本 reasoning 训练与评测，未声明视觉、音频或其他输入通路）** |
| 核心结构 | 全局共享 FFN knowledge-vector Memory + 多个 Self-Attn Reasoner + hidden-state cache |
| 训练 | 35B：architecture conversion → 1T-token continual pre-training → SFT → RL；细节未公开 |
| 权重 | [Hugging Face](https://huggingface.co/internlm/Intern-S2-Mobius)（报告首页给出链接） |

## 技术身份

其关键改变不是用 MoE 取代 Transformer，而是**改变 FFN 与层的归属关系**：传统层里 Self-Attn 和 FFN 成对堆叠；Mobius 把 FFN 横向拼为所有 Reasoner 可访问的 Memory。大规模版本再将 Memory 分块并稀疏激活，借用 MoE 式分区控制 activation 成本。

作者把由此而来的跨层全知识访问称作间接 **Backward Residual Connection**，并声称 hidden-state 的反复更新可将推理压缩到 latent space、最终同步预测多个 token。具体检索算子、循环调度和架构配置未披露，所以“近 4× end-to-end inference speedup”应作为报告实验结果读取，不能从公开信息推导其可复现实作。

### 与相邻架构的边界

- 不是 [Attention Residuals](../concepts/attention-residuals.md)：后者 attention 聚合先前层的**激活**，Mobius 共享的是 FFN **知识向量参数库**。
- 不是 [Looped Transformers](../concepts/looped-transformers.md)：两者都有 recurrent latent computation，但报告未称 Mobius 的 Reasoner 跨层共享同一套参数。
- 不是 [线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md)：它保留 Self-Attn，优化目标是让每次 attention 从更大的共享知识库获得更高信息密度，而不是把 token mixer 线性化。

## 相关页面

- 来源：[Intern-S2-Mobius 技术报告](../sources/intern-s2-mobius.md)
- [Attention Residuals](../concepts/attention-residuals.md)
- [2026 前沿模型技术报告对比](../comparisons/2026-open-model-technical-reports.md)
