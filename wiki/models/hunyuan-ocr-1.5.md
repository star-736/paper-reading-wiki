---
type: Model
title: "HunyuanOCR-1.5"
description: "腾讯 + 中科院信工所 + 南开的轻量端到端 OCR VLM，DFlash 推测解码 + Agentic Data Flow。"
tags: ["model", "ocr", "vlm", "lightweight"]
timestamp: 2026-07-11
---

# HunyuanOCR-1.5

## 身份

HunyuanOCR-1.5 是腾讯 + 中科院信工所 + 南开大学发布的轻量端到端 OCR-specialized VLM，在 HunyuanOCR-1.0 基础上围绕「更快」和「更好」做系统升级，不重新设计架构。核心升级是 DFlash 推测解码（推理加速）和 Agentic Data Flow（agent 驱动数据构造补长尾能力）。

## 关键事实

| 属性 | 值 |
| --- | --- |
| **机构** | 腾讯（大语言模型部）+ 中科院信工所 + 南开大学 |
| **参数量** | 1B（LLM 部分 Hunyuan-0.5B） |
| **模态** | 多模态（文本 + 图像 + 视频） |
| **视觉编码器** | Hunyuan-ViT，原生分辨率，最大 4K |
| **LLM** | Hunyuan-0.5B + XD-RoPE |
| **上下文窗口** | 128K |
| **推测解码** | DFlash（block-diffusion draft model，~90.7M / 5 层，B=16） |
| **训练阶段** | 预训练（Stage3 重规划）+ SFT + RL（IcePop + 三组件 reward） |
| **开源** | 模型权重 + 训练代码（HuggingFace + GitHub） |
| **报告日期** | 2026-07-07（arXiv:2607.04884） |

## 技术身份

HunyuanOCR-1.5 的定位不是通用 VLM，而是 OCR-specialized VLM——把文档解析、文字定位、信息抽取、OCR QA、图像翻译、古文字识别、图表解析、视频字幕提取统一在一个端到端模型里。与通用 VLM（GPT-5、Gemini 3、Qwen3-VL 等）的 OCR 能力相比，它的优势在轻量部署（1B）和 OCR 专项优化（结构化输出保真、退化抑制、长尾覆盖）。

两个核心创新：

- **DFlash** 是 block-diffusion 式推测解码，与 [DSpark](../sources/dspark.md)、[Gemma 4](../models/gemma-4.md) MTP drafter 同属推测解码加速家族，但用 block-diagonal FlexAttention mask 训练并行 draft block，而非 sequential 多 token 预测头。详见 [多 Token 预测](../concepts/multi-token-prediction.md)。
- **Agentic Data Flow** 是 agent 驱动的自动数据构造系统，与 [agentic engineering](../concepts/agentic-engineering.md) 趋势一致--用 agent 自动化数据 pipeline 开发，而非人工标注。

与 [Unlimited OCR](unlimited-ocr.md) 的对比：两者都是 OCR-specialized VLM，但技术路线不同。HunyuanOCR-1.5 走推测解码加速（DFlash，decode 步数减少）+ agentic 数据构造补长尾；Unlimited OCR 走注意力机制设计（R-SWA，KV cache 恒定不随输出增长）。前者解决「更快」，后者解决「更长」。OmniDocBench v1.6 上 HunyuanOCR-1.5（1B）94.74 vs Unlimited-OCR（3B-A0.5B）93.92，但评测条件不完全可比。

## 相关页面

- [HunyuanOCR-1.5 技术报告](../sources/hunyuan-ocr-1.5.md)
- [Unlimited OCR](unlimited-ocr.md) - 同属 OCR VLM 家族，走 R-SWA 恒定 KV cache 路线
- [多 Token 预测](../concepts/multi-token-prediction.md) - DFlash 在推测解码谱系中的位置
- [Agentic Engineering](../concepts/agentic-engineering.md) - Agentic Data Flow 的上下文
