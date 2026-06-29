---
type: Source
title: "DeepSeek-V4 技术报告"
description: "DeepSeek-V4 的 Hugging Face 官方 PDF，重点是百万 token 上下文效率。"
tags: ["source", "deepseek-v4"]
timestamp: 2026-06-06
resource: "../../raw/deepseek-v4-hf-technical-report.pdf"
---

# DeepSeek-V4 技术报告

## 来源

- 原始 PDF：[raw/deepseek-v4-hf-technical-report.pdf](../../raw/deepseek-v4-hf-technical-report.pdf)
- 标题：DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
- 来源说明：Hugging Face 官方技术报告 PDF；初次下载时未找到 arXiv 条目。
- 团队：DeepSeek-AI
- 模型页：[DeepSeek-V4](../models/deepseek-v4.md)

## 核心结论

DeepSeek-V4 是一个预览版模型族，主线是原生 1M token 上下文效率。报告覆盖 DeepSeek-V4-Flash（284B 总参数 / 13B 激活参数）和 DeepSeek-V4-Pro（1.6T 总参数 / 49B 激活参数）。

报告的核心判断是：长上下文扩展不仅是能力问题，更是效率和系统问题。DeepSeek-V4 直接瞄准 attention FLOPs 和 KV-cache 瓶颈，使长周期任务、大规模跨文档分析和 test-time scaling 更可行。

## 架构与训练

DeepSeek-V4 继承 DeepSeekMoE 和 MTP，并加入混合 [高效长上下文注意力](../concepts/efficient-long-context-attention.md)：Compressed Sparse Attention（CSA）和 Heavily Compressed Attention（HCA）。CSA 在 [DeepSeek Sparse Attention](../concepts/deepseek-sparse-attention.md)（V3.2 引入）基础上，先将多个 token 的 KV 压缩成一个 entry，再做稀疏选择；HCA 使用更重的 KV 压缩，并在压缩状态上做密集注意力。

**CSA 的 core attention 是 Shared-Key-Value MQA**（§2.3.1，引 Shazeer 2019）：稀疏选出 top-k 压缩 KV entry 后，所有 query head 共享同一份 entry，且该 entry **同时充当 key 和 value**（K=V，比普通 MQA 共享度更高）。query 侧仍保留 [MLA](../concepts/multi-head-latent-attention.md) 式低秩结构——h 个 head 的 query 由压缩 latent 向量上投影得到，且该 latent 与 indexer query 共享；因 head 数很大，输出端用 grouped output projection 降投影成本。所以 CSA 的完整定性是「MLA 式 latent query +（token 压缩 → DSA 稀疏选择）→ Shared-KV MQA core」，而非单纯的 MLA。

其他关键组件包括 Manifold-Constrained Hyper-Connections（mHC）、Muon optimizer、针对 routed experts 和 CSA indexer QK path 的 FP4 quantization-aware training，以及支持 on-disk shared-prefix reuse 的异构 KV-cache 管理。

DeepSeek-V4-Flash 训练 32T tokens，DeepSeek-V4-Pro 训练 33T tokens。训练从 4K 序列长度开始，逐步扩展到 16K、64K 和 1M；稀疏注意力在 dense warmup 后引入。

## 后训练：OPD 替代 mixed RL

V4 的后训练流水线"largely mirrored that of DeepSeek-V3.2, [but] a critical methodological substitution was made: the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD)"（§5.1 原文）。先按 V3.2 的方式训出 >10 个 domain specialist（SFT + GRPO），然后用加权多专家 reverse KL 做 OPD 把它们融合进统一 student。

V4 OPD 选择**full-vocabulary logit distillation**而非 token-level KL estimate，因为后者"leads to high variance in gradient estimation and often causes training instability"（§5.1.2 原文）。代价是工程极重：要在 >10 个 teacher × 万亿参数 × |V|>100k 词表的规模下做完整 KL，V4 用 **teacher hidden-state caching**（只缓存最后层 hidden state，训练时过 prediction head 重建 logits）+ **按 teacher 排序调度**（每个 teacher head 一个 mini-batch 内只 load 一次）+ **TileLang KL kernel** + **FP4 量化**所有 inference-only forward + **ZeRO 式 teacher 权重分片**从中心化存储按需加载，才把 full-vocab OPD 做到可执行。

> 与 MiMo MOPD（token-level KL）和 Qwen3 Strong-to-Weak（单 teacher）的对比详见 [On-Policy Distillation 跨报告对比](../comparisons/on-policy-distillation.md)。OPD 算法的奠基性介绍见 [Thinking Machines Lab On-Policy Distillation 博客源页](thinking-machines-on-policy-distillation.md)。

## 效率主张

在 1M context 下，报告称 DeepSeek-V4-Pro 只需要 DeepSeek-V3.2 的 27% single-token inference FLOPs 和 10% KV cache；DeepSeek-V4-Flash 进一步降低到 10% FLOPs 和 7% KV cache。相对 BF16 GQA8 注意力基线，报告称 1M context 下 KV cache 可降到约 2%。

## 评测要点

报告称 DeepSeek-V4-Flash-Base 在更少总参数和激活参数下，超过 DeepSeek-V3.2-Base 的多数 benchmark。DeepSeek-V4-Pro-Base 被定位为 DeepSeek 系列最强 foundation model，在知识、推理、代码和长上下文任务上全面提升。DeepSeek-V4-Pro-Max 被定位为开放模型中的 SOTA reasoning 模式，并具备较强 agentic 能力。

## 待追问

- 架构能力强但复杂；报告本身也把架构简化列为未来方向。
- 当前来源不是 arXiv，后续如果出现 arXiv 版本或新 revision，需要更新来源页和日志。
- **生产 serving 的 speculative decoding 已被 DSpark 替换**：报告原文写 MTP depth=1，但 V4 preview 上线两周后 [DSpark](dspark.md)（semi-AR drafter + confidence-scheduled verification，PKU + DeepSeek-AI 后续论文）整体取代了 MTP-1，V4-Flash per-user 速度 +60–85%、V4-Pro +57–78%。当前 deepseek.com 服务背后跑的就是 DSpark，不是 V4 本报告里的 MTP-1。

