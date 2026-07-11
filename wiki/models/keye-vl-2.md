---
type: Model
title: "Keye-VL-2.0"
description: "快手 Keye Team 的开源 30B-A3B 多模态 MoE 模型，GQA+DSA 256K 长视频理解 + Cross-Modal MOPD。"
tags: ["model", "keye-vl-2", "moe", "multimodal", "dsa", "video"]
timestamp: 2026-07-11
---

# Keye-VL-2.0

## 身份

快手 Keye Team 的第二代多模态基础模型，前身为 Keye-VL（2507.01949）和 Keye-VL-1.5（2509.01563）。开源模型权重（HuggingFace: `Kwai-Keye/Keye-VL-2.0-30B-A3B`），主打长视频理解和多模态 agentic intelligence。

## 关键事实

| 属性 | 值 | 来源 |
| --- | --- | --- |
| 总参数 | ~31.1B (BF16) | 已据 HF config 核实（safetensors.total = 31,122,082,160） |
| 激活参数 | ~3B | 报告标题 30B-A3B；HF config num_experts_per_tok=8 |
| **模态** | 多模态（文本 + 图像 + 视频） | 报告 + HF config 含 vision_config |
| 架构 | KeyeVL2MoeForConditionalGeneration | 已据 HF config 核实 |
| LLM backbone | Qwen3-30B-A3B-Thinking-2507 | 报告 §2（§5 第一段） |
| num_hidden_layers | 48 | 已据 HF config 核实 |
| num_attention_heads | 32 | 已据 HF config 核实 |
| num_key_value_heads | 4 (GQA) | 已据 HF config 核实 |
| hidden_size | 2048 | 已据 HF config 核实 |
| num_experts | 128 | 已据 HF config 核实 |
| num_experts_per_tok | 8 | 已据 HF config 核实 |
| max_position_embeddings | 262,144 (256K) | 已据 HF config 核实 |
| Vision Encoder | SigLIP-400M-384-14 变体，27 层，hidden 1152 | 报告 §2.1 + HF config |
| 上下文窗口 | 256K（预训练 Stage 3 达到） | 报告 §3.4 |
| 注意力策略 | GQA-based DSA（Lightning Indexer MQA + GQA Sparse Aggregation） | 报告 §2.3 |
| 后训练核心 | Cross-Modal MOPD（13 个 RL teacher） | 报告 §4.2.6 |
| 开源协议 | 开源（HuggingFace + GitHub） | 报告标题页 |

## 技术身份

Keye-VL-2.0 在架构上的核心定位是**首个把 DSA 适配到 GQA 多模态 backbone 的模型**。此前 DSA 只在 MLA-based 模型上实现（[DeepSeek-V3.2](../sources/deepseek-v32.md)、[GLM-5](../sources/glm-5.md)）。Keye-VL-2.0 把 DSA 的 Lightning Indexer（MQA-style 共享 key 打分）与 GQA backbone 的 per-group 稀疏聚合结合，证明 DSA 不依赖 MLA，可以在 GQA 模型上作为长上下文扩展方案。

后训练方面，Keye-VL-2.0 用 Cross-Modal MOPD 把 13 个 RL-trained domain teacher 的能力融合进单一 MoE student，是 [MOPD 范式](../concepts/multi-teacher-on-policy-distillation.md)在多模态场景的首次大规模应用。其 top-k overlap estimator + SPRR re-tokenization + token-category-aware scaling + localized repetition penalty 是对 MiMo MOPD 的工程增强。

## 相关页面

- [Keye-VL-2.0 技术报告](../sources/keye-vl-2.md)
- [DeepSeek Sparse Attention](../concepts/deepseek-sparse-attention.md)
- [Multi-Teacher On-Policy Distillation](../concepts/multi-teacher-on-policy-distillation.md)
- [On-Policy Distillation 跨报告对比](../comparisons/on-policy-distillation.md)
- [高效长上下文注意力](../concepts/efficient-long-context-attention.md)
- [Qwen3](../models/qwen3.md)：LLM backbone 来源
