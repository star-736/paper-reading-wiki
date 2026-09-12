---
type: Model
title: "HunyuanOCR-1.0"
description: "腾讯 Hunyuan Vision 的 1B 端到端 OCR VLM 首发：Hunyuan-ViT + Hunyuan-0.5B，四阶段预训练 + GRPO；OmniDocBench v1.5 自报 94.10。"
tags: ["model", "ocr", "vlm", "lightweight", "hunyuan"]
timestamp: 2026-09-12
---

# HunyuanOCR-1.0

## 身份

HunyuanOCR-1.0 是腾讯 Hunyuan Vision Team 发布的轻量端到端 OCR-specialized VLM。原报告标题只写 HunyuanOCR；本库用 1.0 与 [HunyuanOCR-1.5](hunyuan-ocr-1.5.md) 区分，**不覆盖** 1.5 页。架构是 Native ViT + Adaptive MLP + Hunyuan-0.5B，一次推理覆盖 spotting、parsing、IE、VQA、图像翻译。后作不重设计这条栈，只加 DFlash、Agentic Data Flow 和 IcePop。

## 关键事实

| 属性 | 值 |
| --- | --- |
| **机构** | 腾讯 Hunyuan Vision Team |
| **参数量** | 约 1B（ViT ~0.4B + Hunyuan-0.5B） |
| **模态** | 多模态（文本 + 图像输入；结构化文本输出）。字幕任务吃的是视频截图，不是原生视频流（已据原文核实，§4.1.3） |
| **视觉编码器** | Hunyuan-ViT，SigLIP-v2-400M 基座，原生分辨率 |
| **LLM** | Hunyuan-0.5B 稠密 + XD-RoPE（text / height / width / time） |
| **上下文窗口** | 预训练 Stage-3/4 为 32K |
| **训练阶段** | 四阶段预训练（50B+300B+80B+24B tokens）+ GRPO RL |
| **开源** | HuggingFace + GitHub；vLLM 部署 |
| **报告日期** | 2025-12-11（arXiv:2511.19575v2） |

## 技术身份

定位不是通用 VLM，而是 **OCR 专家 VLM**：用 1B 端到端替代「多专家级联」和「过大通用 VLM」。与后作的分界：

- **1.0**：证明纯端到端 + 高质量应用向数据 + GRPO 就够在 v1.5 协议上超过模块化 0.9–1.2B（PaddleOCR-VL 92.86、MinerU2.5 90.67）。
- **1.5**：同一骨架上做「更快」（DFlash）和「更好」（Agentic Data Flow + IcePop），分辨率/上下文再扩。

与同代对照：

- vs [GLM-OCR](glm-ocr.md)：都是 ~1B 端到端。GLM-OCR 走 MTP 共享头 + 布局两阶段；本模型走纯端到端、无 MTP。v1.5 自报 GLM-OCR 94.62 vs 本模型 94.10，协议相近但不是同一张统一重测表。
- vs [MinerU2.5](mineru-2-5.md) / [MinerU2.5-Pro](mineru-2-5-pro.md)：MinerU 解耦 coarse-to-fine + 数据引擎；本模型不拆 layout。v1.6 上读本模型须用 MinerU 统一重测 89.87，不要用 94.10。
- vs [Unlimited OCR](unlimited-ocr.md)：Unlimited 用 R-SWA 保恒定 KV；本模型不解「更长」，只解「端到端 + 轻量」。

## 相关页面

- [HunyuanOCR 1.0 技术报告](../sources/hunyuan-ocr-1.0.md)
- [HunyuanOCR-1.5](hunyuan-ocr-1.5.md) - 同架构后作
- [GLM-OCR](glm-ocr.md) · [MinerU2.5-Pro](mineru-2-5-pro.md) · [Unlimited OCR](unlimited-ocr.md)
