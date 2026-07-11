---
type: Model
title: "Gemma 4（家族）"
description: "Gemma 4 多模态 dense + MoE 家族（E2B / E4B / 12B / 26B-A4B / 31B），原生文本+图像+音频，混合 SWA/GA 长上下文，12B 为 encoder-free 架构。"
tags: ["model", "gemma-4", "multimodal", "dense", "moe"]
timestamp: 2026-07-11
---

# Gemma 4（家族）

## 身份

Gemma 4 是 Google DeepMind 2026-06 释出的第四代开放权重模型家族，**原生多模态**（文本 + 图像 + 音频），包含 4 个 dense 变体和 1 个 MoE 变体，规模从 2.3B 有效参数到 31B 总参数。12B 变体首次采用 **encoder-free 架构**，直接在 LLM 内处理原始图像 patch 和音频 chunk。所有变体配 thinking mode 和 MTP drafter。

## 关键事实

| 项 | 取值 |
| --- | --- |
| **模态** | **多模态**（文本 + 图像 + 音频） |
| 变体 | 4 dense（E2B / E4B / 12B / 31B）+ 1 MoE（26B-A4B） |
| E2B | 2.3B 有效 / 5B 总（per-layer embeddings），4:1 SWA:GA |
| E4B | 4.5B 有效 / 8B 总，5:1 SWA:GA |
| 12B | 12B dense，**encoder-free**，5:1 SWA:GA |
| 26B-A4B | 26B 总 / 3.8B 激活（MoE），5:1 SWA:GA |
| 31B | 31B dense，5:1 SWA:GA |
| 架构 | decoder-only Transformer，RMSNorm + QKNorm |
| 上下文 | 128K+（RULER 128k 评测） |
| 位置编码 | 全局层 p-RoPE (p=0.25)，局部层 RoPE (base 10k) |
| KV cache 优化 | values=keys（全局层，E2B/E4B 除外）+ KV sharing（E2B 20/35, E4B 18/42），全局 KV -37.5% |
| 视觉编码器 | 150M ViT（E2B/E4B）/ 550M ViT（26B-A4B, 31B）/ encoder-free（12B） |
| 音频编码器 | 305M USM Conformer（E2B/E4B）/ encoder-free（12B） |
| MTP drafter | 4 层 Transformer，cross-attention 复用主模型 KV，dim 256/1024 |
| 量化 | QAT（mobile quant int2/int4 + Q4_0） |
| 词表 | 262k（SentencePiece，与 Gemini 共享 tokenizer） |
| Thinking mode | 是（`<|think|>` 激活，`<|channel>thought ...<channel|>` 标记） |
| 许可 | Apache 2.0 |

## 技术身份

Gemma 4 的架构定位是 **Gemma 3 的效率导向演进**，而非追求参数规模前沿。核心差异化：

1. **混合 SWA/GA 的工程保守路线**：沿用 Gemma 3 的 local sliding window + global attention 混合（5:1 或 4:1），没有走 DSA / CSA / 线性注意力等更激进的路线。创新在于 KV cache 侧的 key-as-value + p-RoPE + KV sharing 三件套，把全局 KV cache 压低 37.5%。与 [MiMo-V2-Flash](mimo-v2-flash.md) 的 5:1 SWA/GA 同属模式稀疏路线，但 Gemma 4 额外加了 key-as-value 这个 MLA 式的 KV 压缩思路。

2. **Encoder-free 作为多模态新路径**：12B 模型完全去掉 ViT 和 USM 编码器，用 35M 参数的单个 matmul 投影原始 48×48 image patch，音频则直接投影 640 维 chunk。这与 Qwen3-VL / Qwen3.5-Omni / Kimi K2.5 的"冻结/联合训练独立编码器"路线形成对照——Gemma 4 12B 证明了 encoder-free 在 12B 规模上可行，但音频性能仍有差距。

3. **MTP drafter 的 cross-attention 设计**：drafter 不复制主模型 KV（如 EAGLE 式），而是通过 cross-attention 访问主模型 KV cache，**无需 MTP prefill**，支持任意 draft 长度。E2B/E4B 的 top-k on token clusters 把最终投影从 d×262k 降到 d×4k，是词表规模下的效率适配。

4. **QAT 全栈量化**：不仅量化 LLM 权重，还量化 ViT 和音频编码器。ViT W8A8 前向内存减半，音频编码器 on-disk 从 390MB 降到 87MB。每个 block 加 scalar scale 以支持 fp16 稳定推理。

## Arena 定位

Gemma 4 31B 以 Elo 1451 居 **open dense 模型首位**，超过 Gemma 3 27B（1366）并逼近 GLM-5（1457）、DeepSeek-V4 Pro（1456）等更大 MoE。26B-A4B（1438）在 MoE 中也具竞争力。E2B（2.3B 有效参数）在多个 benchmark 上匹配 Gemma 3 27B，实现约 10× 参数效率提升。

## 相关页面

- 来源：[Gemma 4 技术报告](../sources/gemma-4.md)
- 概念：[高效长上下文注意力](../concepts/efficient-long-context-attention.md)（SWA/GA 混合路线）
- 概念：[多 Token 预测](../concepts/multi-token-prediction.md)（MTP drafter）
- 概念：[MoE 前沿模型扩展](../concepts/moe-frontier-model-scaling.md)
- 比较：[2026 前沿模型技术报告对比](../comparisons/2026-open-model-technical-reports.md)
