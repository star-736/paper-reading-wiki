---
type: Model
title: "Macaron-V1"
description: "Mind Lab 的开放 agent-model 家族；以 frozen shared base、四个按 turn 路由的 LoRA specialist、HCP 版本化 harness 和 MindForge RSI 生命周期构成。"
tags: ["model", "agentic", "lora", "continual-learning"]
timestamp: 2026-08-23
---

# Macaron-V1

## 身份

Macaron-V1 是 Mind Lab 的开放 agent-model 家族，重点不在训练一个更大的统一 checkpoint，而在将模型、specialist adapter 和 agent harness 拆成可独立更新又共同评测的层。主要来源：[Macaron-V1 技术报告](../sources/macaron-v1.md)。

## 关键事实

| 项目 | Macaron-V1-Venti | Macaron-V1-Tall |
| --- | --- | --- |
| base | GLM-5.2（报告称 744B） | Qwen3.6-35B-A3B |
| release label | 748B | 50B |
| LoRA specialist | L0 Chat、L1 Agent、L2 Coding、L3 GenUI | 同一四类 |
| 每个公开 adapter 的 stored values | 7.688B | 3.776B（L2 以 F32 存储） |
| nominal base + 四 adapter | 约 774.8B | 约 50.1B |
| serving | Proxy 让 L0 每 user turn 路由到一个 adapter | 同 |

**模态**：Venti 是纯文本 agentic model；Tall 的 adapter 训练为纯文本，但报告对其服务跑了图像输入的视觉 benchmark，结果不足以确证多模态能力保留或迁移。这里不把这两件事混同为“已核实的原生多模态模型”。

## 技术身份

MoL 的特点是 request-level composition，而非 token-level adapter mixing 或将所有 LoRA 合并。L0 在受限标签空间内决定一个 adapter；被选中 adapter 回答并写 192-token server-side summary。各 adapter 只保留自己的完整历史、从其他 adapter 继承摘要，因此 re-entry 时可形成稳定 own-view，复用本 adapter 的 prefix KV。

Macaron-V1 的「continual」是由 versioned model–harness pair 的循环承载：HCP 定义可复现的 runtime contract，MindForge 保存 problem bank、evaluation、trajectory、training job、HCP 和 adapter revision 的 lineage，MinT 再将可服务的 immutable adapter revision 与 trainer checkpoint / optimizer state 分开。报告验证了这些部件及一次 frozen-model harness search；还没有实证跨代持续学习、第三方 adapter 的 collective intelligence 或 personalization 的质量/隐私边界。

## 相关页面

- 来源：[Macaron-V1 技术报告](../sources/macaron-v1.md)
- [Agentic engineering](../concepts/agentic-engineering.md)
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)
- [Agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md)
- [2026 前沿模型技术报告对比](../comparisons/2026-open-model-technical-reports.md)
