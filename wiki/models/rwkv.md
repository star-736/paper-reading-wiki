---
type: Model
title: "RWKV"
description: "Peng 等在 EMNLP 2023 发布的 dense RNN 语言模型族（169M–14B），Pile 330B；Time Mixing 的 WKV 是 channel-wise 衰减，不是矩阵线性注意力或 delta rule。纯文本。"
tags: ["model", "rwkv", "rnn", "linear-attention"]
timestamp: 2026-09-13
---

# RWKV

## 身份

EleutherAI 等在 2023 发布的 Receptance Weighted Key Value 语言模型族。论文把同一套公式写成可并行训练的「Transformer-like」和常数复杂度的 RNN。权重见 [HuggingFace/RWKV](https://huggingface.co/RWKV)。后作 RWKV-5/6/7 不在本页；本页只覆盖 arXiv:2305.13048v2 的检查点，社区后称 **RWKV-4**。

## 关键事实

| 属性 | 169M | 430M | 1.5B | 3B | 7B | 14B |
| --- | --- | --- | --- | --- | --- | --- |
| 层数 / 宽度 | 12 / 768 | 24 / 1024 | 24 / 2048 | 32 / 2560 | 32 / 4096 | 40 / 5120 |
| **模态** | 纯文本 | 同 | 同 | 同 | 同 | 同 |
| 数据 | The Pile，330B token（一 epoch） | 同 | 同 | 同 | 同 | 同 |
| 训练上下文 | 1024，再可扩到 8192 | 同 | 同 | 同 | 同 | 同 |
| 来源 | [RWKV](../sources/rwkv.md)（arXiv:2305.13048v2） | 同 | 同 | 同 | 同 | 同 |

**模态**：已据原文核实为纯文本。LRA 含图像/数学表达式任务，那是**评测格式**，不是本模型的输入模态。

## 技术身份

RWKV-4 在线性时间模型谱系里是 **AFT → 可递归的 channel-wise WKV**，不是后来矩阵状态那一支：

1. 推理状态 $O(d)$，不是 GLA/GDN 的 $d_k\times d_v$ 矩阵。
2. 衰减 $w$ 按通道学、与当前 token 无关。
3. 主文卖点是「14B dense RNN 能跟 FLOP 对齐的 Transformer 打平」，不是长上下文稀疏或 delta rule。

读 2025–2026 的混合线性栈（KDA、Lightning、Mamba-2）时，不要把 RWKV 当成同一条递推。

## 相关页面

- 来源：[RWKV](../sources/rwkv.md)
- 概念：[线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md)
- 矩阵状态对照：[Gated Linear Attention](../sources/gated-linear-attention.md)、[Lightning Attention-2](../sources/lightning-attention-2.md)
