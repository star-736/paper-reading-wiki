---
type: Model
title: "Qwen3-VL（家族）"
description: "Qwen3-VL 多模态家族（2B/4B/8B/32B dense + 30B-A3B / 235B-A22B MoE），256K context，LLM backbone 用标准 GQA 的 Qwen3，叠 SigLIP-2 + DeepStack + Interleaved MRoPE + 文本时间戳。"
tags: ["model", "qwen3-vl"]
timestamp: 2026-06-22
---

# Qwen3-VL（家族）

## 身份

Qwen3-VL 是 Qwen 团队 2025-11 释出的多模态大模型家族，**= [Qwen3](qwen3.md) 标准 GQA LLM backbone + SigLIP-2 视觉编码器 + DeepStack 跨层注入 + Interleaved MRoPE**。原生支持文本 + 图像 + 视频交错输入，**256K context**，每尺寸释 Instruct（non-thinking）和 Thinking 双权重。

## 关键事实

| 项 | 取值 |
| --- | --- |
| **模态** | **多模态**（文本 + 图像 + 视频；视频帧前缀文本时间戳 token） |
| 变体 | 4 dense（2B / 4B / 8B / 32B）+ 2 MoE（30B-A3B / 235B-A22B） |
| 旗舰 | Qwen3-VL-235B-A22B（235B 总 / 22B 激活，LLM 用 Qwen3-235B-A22B） |
| LLM backbone | **标准 GQA 的 [Qwen3](qwen3.md)**（不是 Qwen3-Next/3.5 hybrid） |
| Vision encoder | **SigLIP-2 SO-400M**（小模型 2B/4B 用 SigLIP-2 Large 300M），2D-RoPE + CoMP 插值，原生分辨率动态训练 |
| Vision-LM merger | 两层 MLP，2×2 patch → 1 视觉 token；DeepStack 额外 3 个专用 merger |
| 位置编码 | **Interleaved MRoPE**（t/h/w 跨频段交错，修 Qwen2.5-VL 的频谱失衡） |
| 视频时序 | **文本时间戳 token**（`<3.0 seconds>` / HMS 格式），替换 Qwen2.5-VL 的 T-RoPE |
| 上下文 | **256K**（原生，预训练 S3 已训到 262,144） |
| 输出双变体 | Instruct + Thinking 各一份权重（与 Qwen3 base 的单权重双模式不同） |
| 许可 | Apache 2.0 |
| 许可链接 | [HuggingFace Qwen3-VL collection](https://huggingface.co/collections/Qwen/qwen3-vl) |

## 技术身份

Qwen3-VL 是 Qwen3 系视觉扩展的**标准 GQA 分支**，三个核心架构差异化点：

1. **DeepStack 注入到 LLM 前 3 层**：ViT 3 个不同层级的视觉特征经独立 merger 投影后，**直接 residual add 进 LLM 的 Block 1 / Block 2 / Block 3** 的 hidden state——保留 low-/mid-/high-level 视觉信息，且**不增加 context 长度**。这是和 Qwen2.5-VL 拉开多模态推理差距的关键。
2. **Interleaved MRoPE**：原 MRoPE 把 t/h/w 切成不交叠子空间各自分配 RoPE 频率，造成每条轴**频谱不均**，长视频退化。Qwen3-VL 让 t/h/w 在 embedding 维度上交错分布，每条轴均匀覆盖高低频。
3. **视频帧文本时间戳**：每个时间 patch 前缀 `<3.0 seconds>` 或 HMS 格式的文本 token——替换 T-RoPE 的绝对时间 position id 方案，时序信息直接以可读 token 形式参与 attention，视频 grounding 准确度大涨。

LLM backbone **直接复用 [Qwen3 base 报告](../sources/qwen3.md) 释出的 8 个标准 GQA 变体**（Qwen3-1.7B / -4B / -8B / -32B / -30B-A3B / -235B-A22B），**不是** Qwen3-Next 那条 GDN+gated-attention hybrid 路线。这点与 [Qwen3.5-Omni](../sources/qwen3.5-omni.md)（走 Qwen3-Next/3.5 hybrid 基座 + 文本/图像/视频/音频）形成对照——两条多模态线**LLM backbone 不同**：

| | LLM backbone | 模态 | 注意力栈 |
| --- | --- | --- | --- |
| **Qwen3-VL** | Qwen3（标准 GQA） | 文本 + 图像 + 视频 | 全 GQA |
| **Qwen3.5-Omni** | Qwen3.5（3 GDN : 1 gated-attention hybrid） | 文本 + 图像 + 视频 + **音频** | GDN + gated full attention |

后训练上 Qwen3-VL **不沿用 Qwen3 base 的「单权重 thinking mode fusion」**，而是 thinking / non-thinking 各训一份权重；前者在视觉推理任务上显著领先。视觉 agent 后训练走「**10k cold-start → SFT-on-Qwen2.5-VL-32B → 蒸到 120k 多轮 → tool-integrated RL**」两阶段，引入 Tool-Calling Reward 防止模型偷懒只调一次工具就交答。

## 相关页面

- 来源：[Qwen3-VL 技术报告](../sources/qwen3-vl.md)
- LLM backbone：[Qwen3](qwen3.md)、[Qwen3 技术报告](../sources/qwen3.md)
- 同家族多模态：[Qwen3.5-Omni 来源页](../sources/qwen3.5-omni.md)
- 概念：[多模态 Agentic 训练](../concepts/multimodal-agentic-training.md)、[Agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md)
- 后续 RL optimizer：[Soft Adaptive Policy Optimization](../sources/soft-adaptive-policy-optimization.md)（外部方法论文，用 Qwen3-VL-30B-A3B preliminary cold-start 展示 SAPO > GSPO / GRPO-R2；不替换本模型页的源报告事实）
