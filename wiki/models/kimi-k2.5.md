# Kimi K2.5

## 身份

Kimi K2.5 是 Kimi Team 发布的 open-source multimodal agentic model，目标是把文本、视觉、代码、搜索、GUI 和工具使用统一到一个 agentic 模型中。

主要来源：[Kimi K2.5 技术报告](../sources/kimi-k2.5.md)。

## 关键事实

| 项目 | 数值或描述 |
| --- | --- |
| Base | Kimi K2 MoE Transformer |
| 模态 | 多模态（文本 + 图像 + 视频，原生 agentic 多模态） |
| 总参数 | 1.04T |
| 激活参数 | 32B / token |
| Experts | 384 experts，8 active / token |
| 语言预训练基础 | Kimi K2 15T high-quality text tokens |
| K2.5 联合训练 | 约 15T vision-text tokens，后续 long-context mid-training |
| Vision encoder | MoonViT-3D，native resolution，NaViT packing |
| Context | 评测通常使用 256K tokens |

## 技术身份

K2.5 的技术身份有两条线。第一条是 [多模态 Agentic 训练](../concepts/multimodal-agentic-training.md)：early vision fusion、MoonViT-3D、zero-vision SFT 和 joint multimodal RL。第二条是 [Agent Swarm](../concepts/agent-swarm.md)：模型学习创建和调度并行 sub-agents，以降低长任务延迟并改善搜索型任务的覆盖率。

## 为什么重要

K2.5 把“多模态”从输入能力扩展到 agent 工作流。图像和视频不仅用于回答视觉问题，也进入代码生成、GUI 操作、网页任务和长文档分析。Agent Swarm 则说明，未来 agent 能力可能不只来自更长 context，还来自把 context 分片给多个局部工作记忆。

## 相关页面

- [多模态 Agentic 训练](../concepts/multimodal-agentic-training.md)
- [Agent Swarm](../concepts/agent-swarm.md)
- [Agentic engineering](../concepts/agentic-engineering.md)
- [Agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md)
- [2026 前沿模型技术报告对比](../comparisons/2026-open-model-technical-reports.md)
