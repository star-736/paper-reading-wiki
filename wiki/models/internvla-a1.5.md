---
type: Model
title: "InternVLA-A1.5"
description: "上海 AI Lab 的统一 VLA 模型，Qwen-3.5 2B backbone + 460M unified expert + latent foresight，6 项仿真 benchmark 全部最优。"
tags: ["model", "vla", "robotics"]
timestamp: 2026-07-11
---

# InternVLA-A1.5

## 身份

InternVLA-A1.5 是上海 AI Laboratory / Physical Intelligence Team 的统一 VLA（Vision-Language-Action）机器人操作模型，把视觉语言理解、latent foresight（视觉前瞻）和动作生成统一在一个 Mixture-of-Transformers 框架里。VLM backbone 采用 [Qwen3.5 2B](../models/qwen3.5.md)（3:1 GDN:full attention 混合），配一个 460M 的轻量 unified expert 做动作预测和 foresight reasoning。预训练于 1.2M 机器人 episodes + 3M 多模态样本。

这是 GDN 混合注意力架构在 **VLA / 机器人操作** 领域的首次落地——Qwen3.5 的 hybrid attention 不只用于语言/多模态对话，还被选为机器人控制的实时感知 backbone。

## 关键事实

| 字段 | 值 |
| --- | --- |
| **总参数** | ~2.5B（VLM 2B + Unified Expert 460M） |
| **VLM backbone** | Qwen-3.5 2B（3 GDN : 1 full attention 混合，6 层堆叠） |
| **Unified Expert** | 460M，同 hybrid 架构缩小 hidden dim |
| **视频生成模型** | WAN2.2-5B（frozen，仅训练时使用，推理丢弃） |
| **模态** | 多模态（文本 + 图像 + 机器人状态 → 连续动作） |
| **动作表示** | Stage 1: FAST 离散 token；Stage 2+: flow-matching 连续 chunk（chunk size 50） |
| **Foresight tokens** | 50 个 learnable tokens |
| **训练数据** | 1.2M 机器人 episodes / 861M frames + 3M 多模态样本 |
| **训练规模** | Stage 1: 300K steps / Stage 2: 600K steps / Posttrain: 60K steps |
| **来源** | [技术报告](../sources/internvla-a1.5.md)（arXiv:2607.04988v1，2026-07-06） |

## 技术身份

InternVLA-A1.5 的核心创新不在单一组件，而在**组合方式**：

1. **VLM 语义不侵蚀**：与 π0.5 类似，VLM backbone 在加动作目标后仍持续训练 VQA 和 subtask 预测，所有目标统一到 chat template 的 next-token loss。论文 lesson 1："prompt design matters"——把 state、control mode、action cast 进 VLM 原生 chat template，保留预训练表示。
2. **MoT 解耦异质目标**：VLM 和 unified expert 共享 full attention 层但各自独立 GDN 层。语义处理和动作/foresight 处理在大部分层互不干扰，只在全局 full attention 层汇总。
3. **Latent foresight 而非像素生成**：不从零学像素级未来预测，而是用 50 个 learnable tokens 从 frozen WAN2.2-5B 蒸馏时空先验。论文 lesson 2："a handful of latent tokens is enough to encode the future information that action learning needs"——policy 只学"想象什么"，预训练视频模型已知道"世界怎么演化"。

**与 wiki 已收录架构的关系**：VLM backbone 是 [Qwen3.5](../models/qwen3.5.md) 2B，其 3:1 GDN:gated-attention 混合栈来自 [Qwen3-Next](../sources/qwen3-next-blog.md) 的设计。这意味着 GDN 线性注意力降 KV-cache I/O 的价值不仅在语言/音视频长上下文有用，在机器人实时控制（多视角图像 token 序列）场景也被采用——是 GDN 跨领域采用的证据。

**与前作 InternVLA-A1 的关系**：A1 把未来视觉状态和动作 jointly 作为训练目标，A1.5 的改进是 (1) VLM 持续语义训练；(2) MoT 架构解耦；(3) latent foresight 替代像素级生成，利用预训练视频模型先验。

## 相关页面

- 来源：[InternVLA-A1.5 技术报告](../sources/internvla-a1.5.md)
- 架构基座：[Qwen3.5](../models/qwen3.5.md)（VLM backbone = Qwen-3.5 2B）
- 同基座、不同任务：[WeMM-Embedding](wemm-embedding.md)（Qwen3.5 2B/4B/9B 做通用多模态 embedding）
- [线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md)（GDN 在 VLA 领域的采用证据）
- [注意力门控](../concepts/attention-gating.md)（Qwen3.5 hybrid 架构的 gated attention 在 VLA 中被继承）
