---
type: Model
title: "MMSearch-R1"
description: "ByteDance + NTU 的 7B 按需多模态搜索模型，基于 Qwen2.5-VL-7B-Instruct + GRPO RL，支持图像搜索和文本搜索。"
tags: ["model", "multimodal-search", "search-augmented", "7b"]
timestamp: 2026-09-05
---

# MMSearch-R1

## 身份

- 来源：[MMSearch-R1 技术报告](../sources/mmsearch-r1.md)
- 基座：Qwen2.5-VL-7B-Instruct
- 训练方法：GRPO RL（无 SFT cold-start）
- 后续模型：[DeepMMSearch-R1](deepmmsearch-r1.md)（同一方向的改进版）

## 关键事实

| 属性 | 值 |
| --- | --- |
| 总参数 | 7B（继承 Qwen2.5-VL-7B） |
| 模态 | 多模态（文本 + 图像输入；文本输出） |
| 搜索工具 | 图像搜索（SerpApi）+ 文本搜索（SerpApi + Jina Reader + Qwen3-32B 摘要） |
| 训练数据 | FVQA-train 5K 样本 |
| RL 算法 | GRPO + outcome-based reward + search penalty |
| 评测平均分 | 54.6%（5 benchmark） |
| 搜索率 | 67.1%（vs RAG 100%） |

## 技术说明

MMSearch-R1 是首个通过端到端 RL 训练实现按需多模态搜索的 LMM。核心创新：

1. **按需搜索行为**：search penalty 惩罚不必要的搜索，模型先尝试用内部知识回答，不足时才搜索。
2. **结构化 action space**：`<reason>` → `<search>` / `<text_search>` / `<answer>` 标签协议。
3. **搜索内容 mask**：检索返回的 `<information>` 内容在 loss 计算时被 mask，避免环境反馈引入训练偏差。
4. **Search balancing**：训练集均衡包含 search-required 和 search-free 样本，防止模型总是搜索或从不搜索。

模型在 FVQA-test 上 58.4% 超过 RAG: Qwen2.5-VL-32B 的 57.0%（用 7B 匹配 32B），搜索率仅 66.8% vs 100%。

## 相关页面

- [MMSearch-R1 技术报告](../sources/mmsearch-r1.md)
- [DeepMMSearch-R1](deepmmsearch-r1.md)
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)
