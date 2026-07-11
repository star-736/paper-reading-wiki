---
title: "LoopCoder-v2"
type: Model
description: "7B PLT coder，looped Transformer 的 loop-count 选择研究模型，R=2 最优"
tags: [looped-transformer, PLT, code-generation, agentic-coding, test-time-compute]
timestamp: 2026-07-11
---

## 身份

LoopCoder-v2 是北航 + IQuest Research + Langboat 联合发布的 7B 参数 PLT（Parallel Loop Transformer）coder 模型族，用于研究 PLT loop-count 选择问题。权重发布在 HuggingFace（[Multilingual-Multimodal-NLP/LoopCoder-V2](https://huggingface.co/Multilingual-Multimodal-NLP/LoopCoder-V2)）。

## 关键事实

| 属性 | 值 | 来源 |
|---|---|---|
| **架构** | PLT（Parallel Loop Transformer） | 已据原文核实 |
| **总参数** | ≈7B | 已据原文核实 |
| **共享 block 层数 L** | 14 | 已据原文核实 |
| **Hidden size** | 5120 | 已据原文核实 |
| **Attention heads** | 40 (GQA, 8 KV groups) | 已据原文核实 |
| **Head dimension** | 128 | 已据原文核实 |
| **FFN intermediate** | 27,648 (SwiGLU) | 已据原文核实 |
| **Normalization** | RMSNorm (ε=10⁻⁵) | 已据原文核实 |
| **Position embedding** | RoPE (base 5×10⁵) | 已据原文核实 |
| **Vocabulary** | 76,800 | 已据原文核实 |
| **模态** | 纯文本 | 已据原文核实 |
| **Training tokens** | 18T（1:1 text:code，100+ 编程语言） | 已据原文核实 |
| **Loop counts** | R ∈ {1, 2, 3, 4} | 已据原文核实 |
| **G-SWA window size** | w=64 | 已据原文核实 |
| **训练精度** | bf16 + gradient checkpointing | 已据原文核实 |
| **总 GPU hours** | 1M | 已据原文核实 |
| **指令微调** | 6M SFT 样本（所有 R 共用同一配方） | 已据原文核实 |

**技术身份**：LoopCoder-v2 不是常规 dense LLM，而是一组共享同一个 14 层 block 的 weight-tied looped Transformer。R=1 是非 loop baseline（标准 14 层 dense Transformer），R=2/3/4 将同一 block 重复执行 2/3/4 次。PLT 的两个核心机制——shared-KV Gated Sliding-Window Attention（G-SWA）和 Cross-Loop Position Offset（CLP）——使延迟和 KV-cache 不随 R 增长。详见 [来源页](../sources/loopcoder-v2.md)。

**核心发现**：R=2 是最优操作点（SWE-bench Verified 64.4%，超过 Kimi-Dev-72B），R≥3 退化（强非单调）。机制原因是 gain–cost 剪刀：第二个 loop 的 refinement gain 最大且 effective rank 达峰，之后的 loop 收益急缩但 CLP offset cost Ω(r) 恒定，成本主导。

## 变体

| 变体 | R | SWE-bench Verified | Avg. (10 benchmarks) |
|---|---|---|---|
| Baseline | 1 | 43.0 | 38.0 |
| LoopCoder-v2 | 2 | **64.4** | **46.5** |
| LoopCoder-v2 | 3 | 27.6 | 36.9 |
| LoopCoder-v2 | 4 | 22.4 | 34.3 |

此外有 thinking 变体（explicit CoT + latent loop），在 R=2 上训练，LiveCodeBench +26.9 over instruct 变体，与 latent loop 呈超加性互补。

## 相关页面

- [LoopCoder-v2 来源页](../sources/loopcoder-v2.md) — 完整机制分析、训练配置和评测结果
- [Looped Transformers 概念页](../concepts/looped-transformers.md) — PLT 在 looped Transformer 谱系中的定位
- [Agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md) — SWE-bench Verified / Terminal-Bench 数据点
