---
type: Model
title: "GLM-5V-Turbo"
description: "Z.ai & 清华的多模态 agent 基座模型，基于 GLM-5-Turbo 扩展视觉能力（CogViT + MMTP + 多模态 RL）。"
tags: ["model", "glm-5v-turbo"]
timestamp: 2026-07-11
---

# GLM-5V-Turbo

## 身份

GLM-5V-Turbo 是 Zhipu AI 与 Tsinghua University 发布的多模态 agent 基座模型，是 [GLM-5](glm-5.md) 家族的多模态扩展。其纯文本基座为 GLM-5-Turbo。报告定位为 \"native foundation model for multimodal agents\"——多模态感知被整合为推理、规划、工具使用和执行的核心组件，而非语言模型的辅助接口。

主要来源：[GLM-5V-Turbo 技术报告](../sources/glm-5v-turbo.md)。

## 关键事实

| 项目 | 数值 |
| --- | --- |
| 总参数 | 报告未披露 |
| 激活参数 | 报告未披露 |
| **模态** | 多模态（文本 + 图像 + 视频 + GUI + 文档 + 网页） |
| 视觉编码器 | CogViT（CogViT-L = 403M，两阶段预训练：distillation MIM + contrastive IT） |
| LLM backbone | GLM-5-Turbo（报告未说明是否沿用 GLM-5 的 DSA + MoE 架构） |
| MTP | Multimodal Multi-Token Prediction（共享 learnable `<\|image\|>` token 方案） |
| RL 任务数 | 30+ 类别（感知 + 推理 + agentic） |
| Agent 框架 | Claude Code、AutoClaw、OpenClaw |
| 官方 skill 数 | 15（5 native + 5 external tool + 5 specialized） |

模态已据报告原文核实：Abstract 和 § 1 Overview 明确列出图像、视频、网页、文档、GUI 作为原生输入。

## 解释

GLM-5V-Turbo 的差异点不在模型规模（报告刻意不披露参数量），而在多模态感知与 agent 能力的深度整合。三个技术支柱：

CogViT 视觉编码器以 403M 参数在零样本 benchmark 上超过 427M–632M 的竞品（SigLIP2-SO / DFN-H / MetaCLIP2-H），两阶段预训练（distillation MIM + contrastive IT）兼顾表示学习与跨模态对齐。

MMTP 解决多模态 MTP 的核心问题：图像 token 如何传给 MTP head。采用共享 learnable `<\|image\|>` token 替代直接传视觉 embedding，避免跨 pipeline-parallel 阶段传播，且 0.5B 消融显示训练 loss 更低、收敛更稳。

多模态 RL 覆盖 30+ 任务类别，观察到 RL 跨域干扰弱于 SFT——多域可同时稳定提升。RL 基础设施层面做了全流程解耦异步、多模态细粒度内存管理和拓扑感知分区。

三个 design lens 具有跨报告参考价值：感知是高层能力天花板；agent 能力适合分层优化；端到端任务的关键是清晰规格 + 可靠验证 + 受控评测。

## 相关页面

- [GLM-5](glm-5.md) - 纯文本基座模型
- [多模态 Agentic 训练](../concepts/multimodal-agentic-training.md)
- [多 Token 预测](../concepts/multi-token-prediction.md)
- [Agentic Engineering](../concepts/agentic-engineering.md)
- [Agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md)
- [2026 前沿模型技术报告对比](../comparisons/2026-open-model-technical-reports.md)
