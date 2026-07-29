---
type: Concept
title: "MoE 前沿模型扩展"
description: "多篇报告中的总参数、激活参数和系统成本对比。"
tags: ["concept", "moe-frontier-model-scaling"]
timestamp: 2026-06-06
---

# MoE 前沿模型扩展

## 概览

当前沉淀的前沿模型报告大多使用 Mixture-of-Experts（MoE）架构。关键比较不只是总参数，而是每 token 激活参数，以及服务长上下文、agent rollout 和多模态输入所需的系统成本。

## 当前对比

| 模型 | 总参数 | 激活参数 | Expert 设计备注 |
| --- | ---: | ---: | --- |
| [GLM-5](../models/glm-5.md) | 744B | 40B | 256 experts；相比 GLM-4.5 显著放大。 |
| [MiMo-V2-Flash](../models/mimo-v2-flash.md) | 309B | 15B | 256 experts，8 active，无 shared experts。 |
| [DeepSeek-V4-Flash](../models/deepseek-v4.md) | 284B | 13B | 256 routed experts + 1 shared expert；6 routed active。 |
| [DeepSeek-V4-Pro](../models/deepseek-v4.md) | 1.6T | 49B | 384 routed experts + 1 shared expert；6 routed active。 |
| [MiniMax-M2](../models/minimax-m2-series.md) | 229.9B | 9.8B | 256 fine-grained experts，8 active，sigmoid gating。 |
| [MiniMax-M3](../models/minimax-m3.md) | 428B | 22B (+ 600M visual encoder) | 128 routed experts，每 token 激活 4 个（NVIDIA blog）。 |
| [Kimi K2.5](../models/kimi-k2.5.md) | 1.04T | 32B | 384 experts，8 active，继承 Kimi K2 MoE backbone。 |
| [Kimi K3](../models/kimi-k3.md) | **2.78T** | **104.2B** | **首个开源 3T 级**。896 routed + 2 shared，16 active（sparsity 56）；[Stable LatentMoE](stable-latentmoe.md)（routed 在 latent 空间 ℓ=d/2=3584，SiTU-GLU bounded activation + Quantile Balancing）；69 KDA + 24 Gated MLA 混合注意力。 |
| [Gemma 4 26B-A4B](../models/gemma-4.md) | 26B | 3.8B | 报告未公开 expert 数和路由配置；与 dense 变体（E2B/E4B/12B/31B）共享架构。 |
| [Seed2.0](../models/seed2.md) | **未披露** | **未披露** | Model Card 不含架构/训练细节；三档定价 Pro/Lite/Mini 对标 GPT-5.2/GPT-5-mini 级别。 |
| [Ling-2.6-1T](../models/ling-2.6.md) | ~1T | ~8B | 256 routed experts + 1 shared，8 active；fine-grained MoE（expert intermediate=2,048）；前 4 层 dense FFN。 |
| [Ling-2.6-flash](../models/ling-2.6.md) | ~104B | ~5B | 同上 256+1 配置，32 层，前 1 层 dense。 |
| [Mach-Mind-4-Flash](../models/mach-mind-4-flash.md) | 35B | 3B | 继承 [Qwen3.5-35B-A3B](../models/qwen3.5.md) 架构（后训练不改预训练权重）。 |

## 解释

这些报告的激活参数大致落在 9.8B 到 49B。设计前沿不再只是“模型更大”，而是如何组合稀疏激活、长上下文注意力、后训练、agent scaffold 和 serving 基础设施。

MiniMax-M2 是当前知识库里激活参数最低的前沿 agentic 案例，它用 Forge、数据和系统优化补足约 10B active 的模型预算。Kimi K2.5 则处在 1T total / 32B active 层级，并把视觉编码器和 Agent Swarm 纳入整体能力。

Kimi K3 把开源前沿推到 **2.78T / 104.2B active**——首个 3T 级开源模型，激活参数也最大（104B vs DeepSeek-V4-Pro 49B、K2.5 32B）。支撑这个规模的关键不是单纯加 expert，而是 [Stable LatentMoE](stable-latentmoe.md)：LatentMoE 把 routed expert 解耦到 latent 空间（ℓ=d/2）压通信，Normalized + SiTU-GLU 压激活爆炸，Quantile Balancing 解 896-expert 负载失衡。K3 的 896 routed / 16 active（sparsity 56）是当前 wiki 收录最稀疏的 MoE，比 DeepSeek-V4-Pro 的 384/6（sparsity 64）更激进且 expert 池更大。配 MoonEP 完美平衡 EP（E/R 冗余 bound 证明 tight）做 3T 级预训练。

MiMo-V2-Flash 与 DeepSeek-V4-Flash 的激活参数预算接近，但注意力策略完全不同。GLM-5、Kimi K2.5 和 DeepSeek-V4-Pro 则处在更大的 frontier model 层级。

Mach-Mind-4-Flash 以 35B 总参 / 3B 激活成为已收录 agentic MoE 中激活参数最低的之一（与 Gemma 4 26B-A4B 的 3.8B 激活接近，低于 MiniMax-M2 的 9.8B），证明纯后训练 scaling 可让紧凑模型在 AIME'26 / IFBench / Behavioral-SafetyBench 等维度追平甚至超越 10–30× 激活规模的模型。

## 观察点

- 激活参数数值不能单独决定长上下文推理成本。
- 总专家数增加后，expert routing 稳定性与通信开销更重要。
- Quantization 和 KV-cache 设计可能主导实际 serving cost。
- 对 agentic model，rollout 调度、MTP 接受率和工具等待时间也会改变“有效成本”。
