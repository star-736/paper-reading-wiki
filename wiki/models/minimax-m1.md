---
type: Model
title: "MiniMax-M1"
description: "MiniMax 的 456B / 45.9B hybrid-attention reasoning 模型族（M1-40k / M1-80k），7:1 Lightning Attention + softmax，CISPO RL，原生 1M 上下文；纯文本。"
tags: ["model", "minimax-m1", "cispo", "moe", "reasoning"]
timestamp: 2026-09-12
---

# MiniMax-M1

## 身份

MiniMax 在 2025-06 开源的大规模 hybrid-attention reasoning 模型。从 MiniMax-Text-01 继续预训练，再经长 CoT SFT 与 CISPO RL。权重与部署见 [MiniMax-M1](https://github.com/MiniMax-AI/MiniMax-M1)。发布两档 thinking budget：MiniMax-M1-40k 与 MiniMax-M1-80k；40k 是 80k 训练的中间相。

它同时是 **CISPO 的发布检查点**：算法定义在来源页。不要把后作 [MiniMax-M2 Series](minimax-m2-series.md) 写成 CISPO 源头，也不要把 Laguna 的 clip 数字当成本模型的训练配方。

## 关键事实

| 属性 | MiniMax-M1-40k | MiniMax-M1-80k |
| --- | --- | --- |
| 总参数 / 激活 | 456B / 45.9B | 同 |
| Experts | 32 | 同 |
| **模态** | 纯文本 | 纯文本 |
| 注意力 | 7 Lightning Attention : 1 softmax | 同 |
| 原生上下文 | 1M | 1M |
| 最大输出 | 40K | 80K |
| 后训练 | 继续预训练 7.5T → SFT → CISPO RL | 从 40k 检查点按窗扩到 80K |
| AIME 2024 | 83.3 | **86.0** |
| SWE-bench Verified | 55.6 | **56.0** |
| TAU-bench airline | 60.0 | **62.0** |
| 来源 | [MiniMax-M1](../sources/minimax-m1.md)（arXiv:2506.13585v1） | 同 |

**模态**：已据原文核实为纯文本。SWE 沙箱与 TAU-bench 工具是**外部环境**，不是图像或多模态输入。HLE 报的是 text-only 子集。

## 技术身份

M1 在 2025 年的位置是：

1. **混合注意力 reasoning**：把 MiniMax-Text-01 的 Lightning Attention 混合栈开到开源 LRM；卖点是长生成 FLOPs，不是新的状态方程。机制页由 B 路维护，见 [Lightning Attention-2](../sources/lightning-attention-2.md)。
2. **CISPO**：夹 IS 权重、不丢 token，解决 hybrid 上反思 token 被 PPO clip 掉的问题。对照实验在 Qwen2.5-32B，不是 456B 上的算法赛。
3. **长上下文 + 软件工程 RL**：1M 输入、沙箱执行奖励，长上下文与工具使用相对当时开源 LRM 更强；竞赛数学落后最新 DeepSeek-R1-0528。

后续 M2 系列改回 full attention、激活压到约 10B，算法与系统重心转到 Forge。读 M2 时把「CISPO 定义」和「M2 训练栈」分开。

## 相关页面

- 来源：[MiniMax-M1](../sources/minimax-m1.md)
- 后作：[MiniMax-M2 Series](minimax-m2-series.md)
- 比较：[LLM RL policy optimization 对比](../comparisons/llm-rl-policy-optimization.md)
- 概念：[Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)、[训练—rollout 一致性](../concepts/train-rollout-consistency.md)
- 采用方：[Laguna](laguna.md)
