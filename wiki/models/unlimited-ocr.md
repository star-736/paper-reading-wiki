---
type: Model
title: "Unlimited OCR"
description: "Baidu 的 OCR-specialized VLM，基于 DeepSeek OCR，用 R-SWA 保持恒定 KV cache 实现长文档一次性转录。"
tags: ["model", "ocr", "vlm", "attention", "sliding-window"]
timestamp: 2026-07-11
---

# Unlimited OCR

## 身份

Unlimited OCR 是 Baidu 发布的 OCR-specialized VLM，以 [DeepSeek OCR](https://arxiv.org/abs/2510.18234) 为基线，核心改动是用 Reference Sliding Window Attention (R-SWA) 替换 decoder LLM 全部注意力层，使 KV cache 在解码过程中恒定（$L_m + n$），不随输出长度增长。这使模型能在 32K 标准上下文长度下单次前向转录数十页文档，同时单页 OCR 精度不降反升（OmniDocBench v1.5 +6.22 分 vs 基线）。

## 关键事实

| 属性 | 值 |
| --- | --- |
| **机构** | Baidu Inc. |
| **参数量** | 3B 总参 / 500M 激活（MoE） |
| **模态** | 多模态（文本 + 图像） |
| **视觉编码器** | DeepEncoder（SAM-ViT + CLIP-ViT 级联，16× token 压缩，1024×1024 → 256 token） |
| **LLM** | DeepSeek OCR 的 MoE decoder，全部 MHA 替换为 R-SWA |
| **注意力机制** | R-SWA（reference 全局可见 + 128-token causal sliding window） |
| **上下文窗口** | 32K（训练与推理） |
| **KV cache** | 恒定 $L_m + n$（prefix + 128），不随输出长度增长 |
| **训练方式** | 从 DeepSeek OCR checkpoint continue-training，4000 步，DeepEncoder 冻结 |
| **开源** | 模型权重 + 代码（GitHub） |
| **报告日期** | 2026-06-22（arXiv:2606.23050） |

## 技术身份

Unlimited OCR 不是通用 VLM，而是面向长文档解析的 OCR-specialized 模型。它的定位介于两条路线之间：

与 [HunyuanOCR-1.5](hunyuan-ocr-1.5.md) 对比：HunyuanOCR-1.5（1B）走推测解码加速（DFlash）+ agentic 数据构造补长尾能力；Unlimited OCR（3B-A0.5B）走注意力机制设计（R-SWA）保持恒定 KV cache。前者解决「更快」，后者解决「更长」--一个是 decode 加速，一个是 decode 不随长度退化。

在长上下文注意力谱系中，R-SWA 属于「模式稀疏」路线（与 [MiMo-V2-Flash](glm-5.md) 的 SWA/GA 混合、[Gemma 4](gemma-4.md) 的 5:1 SWA/GA 同族），但有一个关键区别：R-SWA 把视觉/reference token 排除在滑动窗口之外，使其全局固定可见且不做状态转移。这避免了 vanilla SWA 对视觉特征的渐进模糊。详见 [高效长上下文注意力](../concepts/efficient-long-context-attention.md)。

## 相关页面

- [Unlimited OCR Works 技术报告](../sources/unlimited-ocr.md)
- [HunyuanOCR-1.5](hunyuan-ocr-1.5.md) - 同属 OCR VLM 家族，不同技术路线
- [高效长上下文注意力](../concepts/efficient-long-context-attention.md) - R-SWA 在注意力路线谱系中的定位
