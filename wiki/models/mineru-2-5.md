---
type: Model
title: "MinerU2.5"
description: "上海 AI Lab + PKU + SJTU 的 1.2B 解耦文档解析 VLM，coarse-to-fine 两阶段 + Data Engine（IMIC 单模型一致性挖 hard case）。MinerU2.5-Pro 的基座。"
tags: ["model", "document-parsing", "ocr", "vlm", "data-centric", "lightweight"]
timestamp: 2026-07-12
---

# MinerU2.5

## 身份

MinerU2.5 是上海人工智能实验室 + 北京大学 + 上海交通大学发布的 1.2B 文档解析 VLM，核心是 **coarse-to-fine 解耦两阶段**（下采样图布局分析 + 原生分辨率裁剪内容识别）。它是 [MinerU2.5-Pro](../sources/mineru-2-5-pro.md) 的基座模型——MinerU2.5-Pro 完全继承其架构（NativeRes-ViT 675M + Qwen2-0.5B）不变，所有改进集中在 Data Engine 和训练策略。本报告的 IMIC（单模型推理一致性挖 hard case）是 MinerU2.5-Pro CMCV（多模型交叉验证）的直接改进对象。

## 关键事实

| 属性 | 值 |
| --- | --- |
| **机构** | 上海人工智能实验室 + 北京大学 + 上海交通大学 |
| **参数量** | 1.2B（NativeRes-ViT 675M 视觉编码器 + Qwen2-0.5B 语言解码器） |
| **模态** | 多模态（文本 + 图像输入；结构化 Markdown 输出） |
| **架构** | 解耦 coarse-to-fine 两阶段（Stage I 布局分析 + Stage II 内容识别），被 MinerU2.5-Pro 完全继承 |
| **视觉编码器** | NativeRes-ViT 675M |
| **LLM** | Qwen2-0.5B |
| **Data Engine** | 三阶段独立：Data Curation（多维过滤）+ Pre-training Preparation（强模型精修标注）+ Fine-tuning Construction（IMIC 挖 hard case + 专家标注） |
| **hard case 挖掘** | IMIC：单模型多次随机推理一致性（PageIoU/CDM/TEDS），低一致性 = hard |
| **训练阶段** | Stage 0 模态对齐 + Stage 1 预训练（6.9M × 2 epochs）+ Stage 2 微调（630K × 3 epochs） |
| **表格语言** | OTSL（5 结构 token vs HTML 28+，序列缩短 ~50%） |
| **部署** | vLLM，A100 2.12 pages/s，按布局类型动态调重复惩罚 |
| **开源** | 模型权重 + 代码（HuggingFace + GitHub） |
| **报告日期** | 2025-09-29（arXiv:2509.22186v2） |

> 模态已据报告原文核实：输入文档图像，输出结构化 Markdown，不涉及音频/视频。

## 技术身份

MinerU2.5 的核心定位是**高效高分辨率文档解析的解耦 VLM**，四个技术身份点：

1. **coarse-to-fine 解耦降计算**：Stage I 下采样图布局分析避开高分辨率开销，Stage II 原生分辨率裁剪保细节。与 [GLM-OCR](../sources/glm-ocr.md)（PP-DocLayout-V3 + 并行识别）、Dolphin/MonkeyOCR 同属解耦范式，区别于 dots.ocr/Nougat 的 end-to-end 路线。
2. **Data Engine 三阶段独立**：Data Curation（多维过滤）+ Pre-training Preparation（强模型精修）+ Fine-tuning Construction（IMIC + 专家）。**三阶段独立运作未联合优化**——这是 MinerU2.5-Pro 改进的核心痛点：采样不被难度告知、标注精修不分难度、IMIC 挖的 hard case 标注仍不可靠。
3. **IMIC 单模型内省**：用 MinerU2.5 Stage-1 checkpoint 多次随机推理一致性挖 hard case。局限是只能捕获单模型认知不确定性，无法区分模型特定盲点 vs 普遍难题。MinerU2.5-Pro 的 CMCV 用三异构模型交叉验证替代。
4. **OTSL 表格语言**：5 结构 token vs HTML 28+，序列缩短 50%，是 VLM 友好的表格中间表示。[GLM-OCR](../sources/glm-ocr.md) 也用 OTSL（OTSL-based token sequence → HTML），两篇同采用 OTSL 路线。

与 [MinerU2.5-Pro](../sources/mineru-2-5-pro.md) 的关系（基座 → 数据中心改进）：

| 维度 | MinerU2.5 | MinerU2.5-Pro |
| --- | --- | --- |
| 架构 | NativeRes-ViT 675M + Qwen2-0.5B | **完全继承不变** |
| 数据规模 | Stage 1: 6.9M × 2 epochs | Stage 1: 65.5M × 1 epoch（~10×） |
| hard case 挖掘 | IMIC（单模型多次推理一致性） | CMCV（三异构模型交叉验证，分 Easy/Medium/Hard） |
| 标注精修 | 强模型统一精修（不分难度） | Judge-and-Refine render-then-verify（仅 Hard）+ 专家定向 |
| 三阶段关系 | 独立运作 | DDAS+CMCV+Judge-and-Refine 协同（覆盖/信息量/准确性） |
| RL | 无 | Stage 3 GRPO 对齐 |
| OmniDocBench v1.6 | 92.98 | 95.69（+2.71） |

与 [GLM-OCR](../sources/glm-ocr.md) 的对比：两者都是 0.9–1.2B 轻量解耦文档解析 VLM + 两阶段 pipeline。MinerU2.5 用 NativeRes-ViT + IMIC + OTSL；GLM-OCR 用 CogViT + MTP + OTSL。OmniDocBench v1.5 上 GLM-OCR 94.6 > MinerU2.5 90.7，但 MinerU2.5 是 MinerU2.5-Pro 的基座（v1.6 上 MinerU2.5-Pro 95.69 反超 GLM-OCR 95.15）。

## 相关页面

- [MinerU2.5 技术报告](../sources/mineru-2-5.md)
- [MinerU2.5-Pro](../sources/mineru-2-5-pro.md) - 直接继承本架构，Data Engine 协同改进 + CMCV 改进 IMIC
- [GLM-OCR](../sources/glm-ocr.md) - 同属轻量解耦文档解析 VLM，OTSL + MTP vs OTSL + IMIC 路线对照
- [HunyuanOCR-1.5](hunyuan-ocr-1.5.md) - 同属轻量文档解析 VLM
- [数据混合优化](../concepts/data-mixture-optimization.md) - IMIC vs CMCV 对照
