---
type: Model
title: "VibeThinker-3B"
description: "新浪微博 3B dense reasoning 模型，基于 Qwen2.5-Coder-3B，Spectrum-to-Signal 后训练范式，verifiable reasoning 追平旗舰。"
tags: ["model", "reasoning", "small-language-model", "dense", "rl"]
timestamp: 2026-07-11
---

# VibeThinker-3B

## 身份

新浪微博 AI 团队的 compact reasoning 模型，3B dense 参数，基于 Qwen2.5-Coder-3B base。前作 VibeThinker-1.5B（arXiv:2511.06221）证明了小模型可以产生稳定推理，3B 版进一步探索 verifiable reasoning 的能力上限。开源（GitHub: WeiboAI/VibeThinker，HuggingFace: WeiboAI/VibeThinker-3B）。

## 关键事实

| 属性 | 值 | 来源 |
| --- | --- | --- |
| 总参数 | 3B | § Abstract |
| 架构 | dense（非 MoE） | § Abstract |
| 基座 | Qwen2.5-Coder-3B | §2 Methods |
| **模态** | 纯文本 | 全文无视觉/音频模态描述 |
| 后训练范式 | Spectrum-to-Signal Principle（SSP） | §2 |
| RL 算法 | MGPO（MaxEnt-Guided Policy Optimization） | §2.2.1 |
| SFT 策略 | 两阶段 curriculum + Diversity-Exploring Distillation | §2.1 |
| Distillation | Offline Self-Distillation（learning-potential filtering） | §2.3 |
| Test-time scaling | CLR（Claim-Level Reliability Assessment） | §3.1 |
| 开源 | 是（GitHub + HuggingFace） | § Abstract |
| 团队 | Sina Weibo Inc. | § Abstract |

**headline 性能**：AIME26 94.3（+CLR 97.1）/ LiveCodeBench v6 80.2 / IFEval 93.4 / GPQA-D 70.2。在 verifiable reasoning 上追平 DeepSeek V3.2（671B）、GLM-5（744B）、Kimi K2.5（1T）等旗舰模型，但知识密集型任务（GPQA-D）仍有 14–15 点差距。

## 技术身份

VibeThinker-3B 的核心定位不是架构创新，而是**后训练系统工程**：如何在严格 3B 参数预算下，通过 data synthesis + curriculum SFT + multi-domain RL + offline self-distillation + Instruct RL 的完整流水线，把 verifiable reasoning 推到旗舰级。论文提出的 **Parametric Compression-Coverage Hypothesis** 给出了理论框架——verifiable reasoning 属于 parameter-dense capability（可压缩进紧凑推理核心），而知识/通用能力属于 parameter-expansive capability（需广泛参数覆盖）。

**MGPO** 是核心 RL 算法，在 GRPO clipped objective 上加 prompt-level exponential weight，降权全对/全错 prompt，聚焦能力边界 prompt。与 DAPO 的 Dynamic Sampling（hard filter）动机一致但实现不同——MGPO 是 soft weighting，DAPO 是 hard filter。详见 [来源页](../sources/vibethinker-3b.md#mgpo-算法骨架) 和 [LLM RL policy optimization 对比](../comparisons/llm-rl-policy-optimization.md)。

**Long2Short RL** 是 VibeThinker-3B 的独有设计——在 Math RL 的第二阶段，只在不改变 group-level reward baseline 的前提下，通过零和的 centered length-aware reward shift，reshape 正确轨迹间的相对偏好，偏好更短推理路径。这是 accuracy→efficiency 的训练时优化，与 CLR（test-time scaling）互补。

## 相关页面

- 来源：[VibeThinker-3B 技术报告](../sources/vibethinker-3b.md)
- 比较：[LLM RL policy optimization 对比](../comparisons/llm-rl-policy-optimization.md)
- 概念：[Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)
