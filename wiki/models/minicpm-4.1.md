---
type: Model
title: "MiniCPM4.1"
description: "OpenBMB 基于 InfLLM-V2 开源的 8B hybrid reasoning 模型；短序列可切回 dense，长序列块稀疏。不要和 MiniCPM-o 4.5 混。纯文本（据 InfLLM-V2 报告）。"
tags: ["model", "minicpm", "infllm-v2", "sparse-attention"]
timestamp: 2026-09-13
---

# MiniCPM4.1

## 身份

OpenBMB 在 InfLLM-V2 报告里开源的 hybrid reasoning 检查点，Hugging Face：[openbmb/MiniCPM4.1-8B](https://huggingface.co/openbmb/MiniCPM4.1-8B)。注意力是 [InfLLM-V2](../sources/infllm-v2.md) 的 dense–sparse 可切换块稀疏，不是 NSA 三分支。

不要写成 [MiniCPM-o 4.5](minicpm-o-4-5.md)。4.5 是全双工全模态；4.1 是本报告的长上下文稀疏。

## 关键事实

| 属性 | 值 |
| --- | --- |
| 总参数 | 8B（报告实验骨干） |
| **模态** | 纯文本（据 InfLLM-V2 原文核实；未写图像输入） |
| 注意力 | InfLLM-V2：GQA + 共享 KV 的块稀疏，短序列可切 dense |
| 可见 token（稀疏） | 6k（$|I|=96$，$B=64$） |
| 短预训练 | 8T token @ 4k（FineWeb-Edu + Stack-v2） |
| 长适应 | 5B token，最长 32k |
| 来源 | [InfLLM-V2](../sources/infllm-v2.md)（arXiv:2509.24663v1） |

**模态**：评测是 RULER / LongBench / 数学与代码推理，输入是文本。不要因为 OpenBMB 还有 MiniCPM-V / MiniCPM-o 就把 4.1 标成多模态。

## 技术身份

报告 §4 的 8B GQA（$d=4096$，$h_q=32$，$h_{kv}=2$）是机制与数字的来源。摘要把 MiniCPM4.1 写成「基于 InfLLM-V2 框架训练并开源」。在读到独立的 MiniCPM4.1 模型卡之前，把 §4 数字当作这套框架的 8B 实验，不要把 HF 页面上的未核项写进本表。

相对 NSA：同一短检查点上做长微调时，本架构不另加 KV 投影。相对 MoBA：有 pooling 选择器，不是块均值点积。

## 相关页面

- 来源：[InfLLM-V2](../sources/infllm-v2.md)
- 对比：[稀疏注意力机制对比](../comparisons/sparse-attention-mechanisms.md)
- 不要混淆：[MiniCPM-o 4.5](minicpm-o-4-5.md)
- 对照稀疏：[NSA](../sources/nsa.md)、[MoBA](../sources/moba.md)
