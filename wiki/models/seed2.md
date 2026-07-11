---
type: Model
title: "Seed2.0"
description: "字节跳动 Seed 团队的 Seed2.0 Series（Pro / Lite / Mini）多模态模型族，面向大规模生产部署。"
tags: ["model", "seed2", "bytedance", "multimodal"]
timestamp: 2026-07-11
---

# Seed2.0

## 身份

Seed2.0 Series 是字节跳动 Seed 团队发布的模型族（Pro / Lite / Mini），面向大规模在线部署，强调多模态理解、快速推理、复杂指令执行和 coding 辅助。前代包括 Seed1.6/1.8、Seed1.5-VL、Seed-OSS、Seed-Coder、Seed Diffusion、Seed-Prover 和 Seedream/Seedance。

主要来源：[Seed2.0 Model Card](../sources/seed2.md)。

## 关键事实

| 项目 | 数值 |
| --- | --- |
| 模型族 | Seed2.0 Pro / Lite / Mini（三档） |
| **模态** | 多模态（文本 + 图像 + 视频；50 图像 benchmark + 24 视频 benchmark 评测，原文确证） |
| 总参数 | **未披露**（Model Card 不含架构/训练细节） |
| 激活参数 | **未披露** |
| 架构 | **未披露** |
| 上下文 | **未披露**（评测中涉及长上下文 benchmark 如 Graphwalks <128K、LongBench v2 128K） |
| 团队 | ByteDance Seed |
| 定价（Pro） | Prefill $0.47 / Decode $2.37 per 1M tokens |
| 定价（Lite） | Prefill $0.09 / Decode $0.53 per 1M tokens |
| 定价（Mini） | Prefill $0.03 / Decode $0.31 per 1M tokens |

**模态**：多模态（文本 + 图像 + 视频）。已据报告原文核实：§4.2 评测 50 个图像 benchmark + 24 个视频 benchmark，§3.2 明确列出九类图像能力和六类视频能力维度。

## 技术身份

Seed2.0 的 Model Card 定位与已收录的其他模型报告有本质区别：**它不是技术报告，不描述架构设计、训练数据或训练方法**。报告的核心是评测框架、部署洞察和真实世界 case studies，展示模型"能做什么"而非"怎么做的"。

这使得 Seed2.0 在 [2026 前沿模型技术报告对比](../concepts/moe-frontier-model-scaling.md) 表中无法填入参数/注意力策略/后训练重点等列。其价值在于：

1. **部署导向**：MaaS 使用分布和 agentic coding 查询模式提供了其他报告缺乏的真实生产环境洞察。
2. **四维评测框架**：Science Discovery / Vibe Coding / Context Learning / Real-World Tasks，把"real-world complexity"操作化为可衡量的维度。
3. **诚实的差距承认**：明确指出与 Claude 在 coding、与 Gemini 在长尾知识方面的差距。
4. **丰富的 case studies**：FreeCAD/CapCut GUI 操作、跨学科科研编程、Erdős 数学证明等，展示模型推理深度（虽缺乏对照实验）。

与 [DAPO](../sources/dapo.md) 的关系：DAPO 是 ByteDance Seed + 清华 AIR 的 RL 系统论文，可能在 Seed2.0 的后训练中使用，但 Model Card 未提及两者关系。

## 相关页面

- [Seed2.0 Model Card](../sources/seed2.md)
- [Agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md)
- [Agentic Engineering](../concepts/agentic-engineering.md)
- [2026 前沿模型技术报告对比](../comparisons/2026-open-model-technical-reports.md)
- [MoE 前沿模型扩展](../concepts/moe-frontier-model-scaling.md)
- [DAPO 技术报告](../sources/dapo.md) - 同为 ByteDance Seed 团队
