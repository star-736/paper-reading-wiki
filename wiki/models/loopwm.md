---
title: "LoopWM"
type: Model
description: "FaceMind 约 1B 的 looped world model：共享 Transformer 动力学核做 action-conditioned 隐状态精炼；公开数字是 ScienceWorld / AlfWorld 五步文本 next-state，配方未披露。"
tags: [model, loopwm, looped-transformer, world-model, latent-dynamics]
timestamp: 2026-09-12
---

# LoopWM

## 身份

LoopWM（Looped World Models）是 FaceMind Research Asia 在 arXiv:2606.18208v1 提出的 **looped latent world model**：观测与动作编码后，用 Prelude–Recurrent–Coda 的参数共享 Transformer 在单步转移内迭代精炼隐状态，再解码下一观测 / 奖励 / 终止。论文把它写成「第一个用于环境模拟的 looped transformer architecture」，而不是通用 LLM。详见 [来源页](../sources/looped-world-models.md)。

## 关键事实

| 属性 | 值 | 来源 |
| --- | --- | --- |
| **架构** | 四模块 world model：Encoder + Action Embedder + Looped Dynamics Core + Prediction Heads | 已据原文核实，§3.1 |
| **动力学核** | Prelude（独立）+ Recurrent（参数共享，循环 $T$ 次）+ Coda（独立）；线性保留 $\bar A$ 谱半径 $<1$ | 已据原文核实，§3.2 |
| **总参数** | 约 1B | 已据原文核实，§4.1 / Table 2 caption；精确计数未给 |
| **层数 / 隐维 / $T$ / $\mu_{\mathrm{rec}}$** | 未披露 | — |
| **训练数据 / 优化器** | 未披露 | §6 写披露范围有意收窄 |
| **模态** | **公开实验为文本观测**（ScienceWorld / AlfWorld 的环境文本）；架构描述允许图像或 state tokens，视觉连续环境无公开指标 | 已据原文核实，§4、§6、Figure 1 |
| **评测协议** | 连续 5 个 action 后对终态观测打 EM / F1 / BLEU-4 / Entity；不是 agent 任务成功率 | 已据原文核实，Table 2 caption |

**技术身份**：LoopWM 不是 [Qwen-AgentWorld](qwen-agent-world.md) 那种 next-token language world model，也不是 [LoopCoder-v2](loopcoder-v2.md) 的 PLT coder。它把 Huginn / Parcae 式顺序循环接到 Dreamer 风格的 action-conditioned latent rollout；可选延迟解码让中间步不重建观测。公开对照是 Claude / Gemini / Qwen 的通用 API，不是 RSSM 或固定深度 world model。

**公开读数**（来源页表）：ScienceWorld 总体 EM 68.4 vs Claude 47.2；AlfWorld 总体 EM 51.6，低于 Claude 的 53.0，BLEU 在四模型中最高。摘要「up to 100× parameter efficiency」按正文是 1B vs 闭源大模型，不是 iso-data 的架构效率。

## 相关页面

- [Looped World Models 来源页](../sources/looped-world-models.md)
- [Looped Transformers](../concepts/looped-transformers.md)
- [Qwen-AgentWorld](qwen-agent-world.md) — 另一条 world-model 线：7 域 native LWM
- [LoopCoder-v2](loopcoder-v2.md) — 循环深度的 coder 实证，有匹配非循环基线
