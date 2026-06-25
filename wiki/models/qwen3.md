# Qwen3（家族基座）

## 身份

Qwen3 是 Qwen 团队（Alibaba）2025-05 释出的开源大模型家族，**标准 Transformer 架构**（GQA + RoPE + RMSNorm + SwiGLU），是后续 [Qwen3-Next](../sources/qwen3-next-blog.md)、[Qwen3.5](qwen3.5.md)、[Qwen3-Coder-Next](qwen3-coder-next.md)、[Qwen3.5-Omni](../sources/qwen3.5-omni.md)、[Qwen3-VL](qwen3-vl.md) 的**直接前作与 LLM backbone**。本页只覆盖 [Qwen3 技术报告](../sources/qwen3.md) 释出的 8 个原始变体；hybrid GDN/gated-attention 一脉是 Qwen3-Next 之后才引入的，那部分请见 Qwen3.5 / Qwen3-Coder-Next 页。

## 关键事实

| 项 | 取值 |
| --- | --- |
| **模态** | 纯文本（Qwen3 base 无 vision/audio；多模态走 Qwen3-VL / Qwen3.5-Omni） |
| 类型 | 6 dense + 2 MoE，单家族 |
| 旗舰 | Qwen3-235B-A22B（94 层，64 Q / 4 KV head，128 expert / 8 active，无 shared expert） |
| 总参 / 激活 | 0.6B / 1.7B / 4B / 8B / 14B / 32B（dense）；30B-A3B（48 层）/ 235B-A22B（94 层）（MoE） |
| 注意力 | **标准 GQA**（Q/KV head 见下表）+ RoPE + RMSNorm pre-norm + SwiGLU，去 QKV-bias，加 **QK-Norm** |
| MoE | 128 expert / 8 active，**无 shared expert**（与 Qwen2.5-MoE 不同），fine-grained segmentation，**global-batch load balancing loss** |
| 上下文 | 32K（0.6B / 1.7B）或 128K（其余）；推理叠 YARN + DCA 可再 4× 扩到 ~512K（论文原话 four-fold inference） |
| 训练 token | **36T**（Qwen2.5 是 18T） |
| 语言 | **119 种**（Qwen2.5 是 29 种） |
| Tokenizer | Qwen BBPE，词表 151,669 |
| 许可 | Apache 2.0 |

变体表（已据报告 Table 1 / Table 2 核实）：

| 变体 | 层数 | Q / KV head | Tie embed | Context |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B / 1.7B | 28 | 16 / 8 | Yes | 32K |
| Qwen3-4B | 36 | 32 / 8 | Yes | 128K |
| Qwen3-8B | 36 | 32 / 8 | No | 128K |
| Qwen3-14B | 40 | 40 / 8 | No | 128K |
| Qwen3-32B | 64 | 64 / 8 | No | 128K |
| Qwen3-30B-A3B | 48 | 32 / 4 | — | 128K |
| Qwen3-235B-A22B | 94 | 64 / 4 | — | 128K |

## 技术身份

Qwen3 的两个**真正卖点**不在架构而在后训练：

1. **统一 thinking / non-thinking 单一权重**：`/think` `/no_think` flag 切换，配 thinking budget；详见 [Qwen3 技术报告](../sources/qwen3.md) § Stage 3 Thinking Mode Fusion。
2. **Strong-to-Weak Distillation 完胜 RL**：小模型（30B-A3B 及以下）走 off-policy → on-policy distillation 两阶段，**性能更好且 GPU 时长仅 1/10**（Qwen3-8B 上对比 1,800 vs 17,920 GPU·h），还能撑高 pass@64（RL 不能）。这是 Qwen3 给小模型的核心交付方式。

架构层面 Qwen3 是 Qwen2.5 风格的「保守、稳健」延续——仅去 QKV-bias + 加 QK-Norm + MoE 去 shared expert——**没有** linear attention / hybrid stack / gated attention。那些是 Qwen3-Next 才引入的下一代设计（详见 [Qwen3-Next 官方博客](../sources/qwen3-next-blog.md)）。

与后续 Qwen 家族成员的接续关系：

- **Qwen3-Next**（无独立技术报告，仅官方博客）：基于 Qwen3 改成 75% GDN + 25% gated full attention 的 hybrid 栈，引入 Zero-Centered RMSNorm + 512 expert MoE + native MTP。
- **Qwen3.5**（仅 Omni 报告 + HF 权重，无独立 base 报告）：把 Qwen3-Next 的 3:1 hybrid 推到 397B-A17B 旗舰 + 多模态。
- **Qwen3-Coder-Next**：Qwen3-Next 的编码 agent 变体，继承 3:1 hybrid。
- **Qwen3.5-Omni**：在 Qwen3.5 hybrid 基座上做全模态（文本 + 图像 + 视频 + 音频）。
- **[Qwen3-VL](qwen3-vl.md)**：**直接用 Qwen3 backbone**（标准 GQA Qwen3，非 Qwen3-Next hybrid）+ SigLIP-2 视觉编码器 + DeepStack 注入。这是 Qwen3 backbone 上的多模态扩展，与 hybrid 家族**走的另一条路**。

## 相关页面

- 来源：[Qwen3 技术报告](../sources/qwen3.md)
- 后续家族：[Qwen3-VL](qwen3-vl.md)、[Qwen3.5](qwen3.5.md)、[Qwen3-Coder-Next](qwen3-coder-next.md)
- 概念：[Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)、[MoE 前沿模型扩展](../concepts/moe-frontier-model-scaling.md)
- 外部后训练算法：[Agentic Reinforced Policy Optimization](../sources/agentic-reinforced-policy-optimization.md)（ARPO 用 Qwen3-8B/14B 做 deep search RL backbone；不是 Qwen3 官方报告的一部分）
