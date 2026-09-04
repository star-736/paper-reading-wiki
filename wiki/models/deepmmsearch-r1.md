---
type: Model
title: "DeepMMSearch-R1"
description: "Apple 的多模态 web search 模型，基于 Qwen2.5-VL-7B-Instruct，SFT+GRPO 训练做多轮文本搜索与裁剪图像搜索。"
tags: ["model", "multimodal-search", "search-augmented", "7b"]
timestamp: 2026-09-05
---

# DeepMMSearch-R1

## 身份

| 字段 | 值 |
| --- | --- |
| 团队 | Apple + Johns Hopkins University |
| 发布日期 | 2025-10-15 |
| 基座 | Qwen2.5-VL-7B-Instruct |
| 总参数 | ~7B（视觉编码器冻结，LLM 加 LoRA r=8） |
| 模态 | 多模态（文本 + 图像输入；文本输出） |
| 许可 | 未明确（论文未提及开源权重） |
| 关联报告 | [DeepMMSearch-R1 技术报告](../sources/deepmmsearch-r1.md) |

## 关键事实

- **定位**：首个能做多轮按需 web search 的多模态 LLM。
- **工具链**：三个搜索工具——text search（文本 query → 网页摘要）、image search（整图/裁剪图 → 视觉相似结果）、Grounding DINO（referring expression → 裁剪区域）。
- **训练**：SFT（DeepMMSearchVQA 10K 样本）→ Online GRPO（FVQA 训练集）。
- **核心能力**：self-reflection（多轮文本搜索迭代修正 query）、self-correction（根据检索结果调整策略）、裁剪图像搜索（消除背景噪声）。
- **评测**：6 个 knowledge-intensive VQA benchmark 平均 57.13，超 RAG workflow +21pp，超 prompt-based agent +9pp，与 o3 竞争。
- **通用 VQA 无退化**：LoRA + KL penalty 保持 OCRBench / MMVet / MMBench 等基准。

## 技术身份说明

DeepMMSearch-R1 不是一个新的基础模型架构，而是在 Qwen2.5-VL-7B-Instruct 上通过 SFT + GRPO 训练出的 search-augmented agent。其技术贡献在于：

1. **裁剪图像搜索机制**：模型先生成 referring expression，Grounding DINO 裁剪相关区域，再用裁剪图做 image search。这解决了整图搜索中背景噪声干扰的问题。
2. **结构化工具调用协议**：用 XML-like tags（`<reason>`, `<text_search>`, `<img_search>`, `<answer>`, `<information>`）编码多轮搜索对话，`<information>` 内容在训练时被 mask。
3. **DeepMMSearchVQA 数据管线**：从 InfoSeek 采样 → Gemini-2.5-Pro 生成多轮对话 → 答案正确性过滤 → 知识类别均匀采样，产出 10K 平衡数据集。
4. **RL 精炼工具行为**：SFT 学会工具使用，RL 减少不必要的裁剪搜索（-37%）、增加多轮文本搜索（+2.6%），使工具调用更高效。

与同方向的 [MMSearch-R1](https://arxiv.org/abs/2506.20670) 相比，DeepMMSearch-R1 的核心差异是：多轮调用（vs 单次）+ 裁剪图像搜索（vs 整图）。

## 相关页面

- [DeepMMSearch-R1 技术报告](../sources/deepmmsearch-r1.md)
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)
- [Agentic engineering](../concepts/agentic-engineering.md)
