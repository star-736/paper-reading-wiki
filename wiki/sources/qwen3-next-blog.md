---
type: Source
title: "Qwen3-Next 官方博客"
description: "Qwen3-Next 无技术报告，本官方博客是其架构设计动机的一手出处：3:1 混合（75% GDN / 25% standard）、选 GDN 因 in-context learning 强于 SWA/Mamba2、全局层加 output gating 去 sink、Zero-Centered RMSNorm + 512-expert MoE + MTP。"
tags: ["source", "qwen3-next-blog"]
timestamp: 2026-06-21
resource: "../../raw/Qwen%20Team%20-%202025%20-%20Qwen3-Next%20blog%20%28Towards%20Ultimate%20Training%20and%20Inference%20Efficiency%29.md"
---

# Qwen3-Next 官方博客

## 来源

- 原始材料：[raw/Qwen Team - 2025 - Qwen3-Next blog (Towards Ultimate Training and Inference Efficiency).md](../../raw/Qwen%20Team%20-%202025%20-%20Qwen3-Next%20blog%20%28Towards%20Ultimate%20Training%20and%20Inference%20Efficiency%29.md)
- 标题：Qwen3-Next: Towards Ultimate Training & Inference Efficiency
- 团队：Qwen Team（Alibaba），2025-09-11
- 提取来源：Alibaba Cloud 官方博客镜像（来源：[Qwen3-Next: Towards Ultimate Training & Inference Efficiency](https://www.alibabacloud.com/blog/602580)）。原 `qwenlm.github.io/blog/qwen3_next/` 已 404 重定向至 qwen.ai 的 JS 站；qwen.ai 原链见博客末 References。
- 前作：[Qwen3 技术报告](qwen3.md) / [Qwen3 模型页](../models/qwen3.md) ——本博客与之的关键差异是把标准 GQA 栈换成 3:1 GDN+gated-attention hybrid，并把 MoE 从 128/8 routed 扩到 512/10+1 shared、加 Zero-Centered RMSNorm 替代 QK-Norm。
- 模型：[Qwen3-Coder-Next](../models/qwen3-coder-next.md)、[Qwen3.5](../models/qwen3.5.md)

## 这页在本 wiki 的定位（重要）

**Qwen3-Next 没有技术报告，这篇官方博客是它架构的一手出处（tier-2 外部佐证）。** 它讲的是 **Qwen 怎么组装架构**（配比、动机、全局层增强、MoE/MTP），**不是 GDN 本身的机制**——gated delta rule 的状态方程、α/β 门、q/k/v/conv 投影那些，机制 tier-1 仍在 [GDN 原论文](gated-delta-net.md) 与 [Gated Attention 论文](gated-attention.md)，GDN 模块的**具体实现细节**（value head 数、Mamba2 式 α 离散化、双 kernel）tier-1 在官方 `transformers` modeling + HF `config.json`。本博客填的是「为什么这么配」这一层。

## 核心结论（官方原话）

- **混合动机**：「linear attention is fast but **weak at recall**; standard attention is **expensive and slow** during inference.」——单用任一种都有短板，故混合。
- **为什么选 GDN**：「systematic experiments」下，**Gated DeltaNet 的 in-context learning 能力强于 Sliding Window Attention 或 Mamba2**。
- **3:1 比例（官方明说，非 config 推断）**：「mix Gated DeltaNet with standard attention at a **3:1 ratio（75% layers use Gated DeltaNet, 25% keep standard attention）**」，混合栈「consistently outperforms any monolithic architecture」。
- **全局 attention 层的增强**：加 **attention output gating** 以「eliminate **Attention Sink** and **Massive Activation**」（即 [Gated Attention](gated-attention.md) 那篇的结论被 Qwen 官方采纳）。

## 架构与训练

- **稳定性设计**：用 **Zero-Centered RMSNorm** 替代 Qwen3 的 QK-Norm（因发现部分 layer norm 权重异常增大），并对 norm 权重加 **weight decay** 防无界增长；MoE router 参数**初始化时归一化**保证早期无偏专家选择。
- **MoE**：512 total experts（10 routed + 1 shared），对比 Qwen3 的 128 experts / 8 routed；80B 总参 / ~3B 激活。
- **MTP**：原生 **Multi-Token Prediction**，高接受率 speculative decoding，并做多步训练保持训练/推理一致。
- **训练效率**：15T token（Qwen3 36T 语料的均匀子集），GPU 时低于 Qwen3-30A-3B 的 80%、仅 Qwen3-32B 的 9.3% 算力；32K+ 上下文推理吞吐 >10×。
- **长上下文**：原生 262,144 token，YaRN 外推验证到 1M。

## 评测要点

- RULER 上 Qwen3-Next-80B-A3B-Instruct 在各长度超过层数更多的 Qwen3-30B-A3B-Instruct-2507，256K 内甚至超过 Qwen3-235B-A22B-Instruct-2507——官方称这「proves the strength of the Gated DeltaNet + Gated Attention hybrid design」。
- Instruct 版整体接近旗舰 Qwen3-235B-A22B-Instruct-2507；Thinking 版多项基准超过 Gemini-2.5-Flash-Thinking。

## 待追问

- 博客只给「3:1、选 GDN」的结论，未公开支撑「GDN > SWA/Mamba2」的消融数字——这些在不存在的 base report 里才可能有，目前无从核实。
- 博客末明说「Looking ahead, we will ... develop **Qwen3.5**」——坐实 Qwen3.5 是 Qwen3-Next 架构的直接延续（与 [Qwen3.5 家族页](../models/qwen3.5.md) 的 config 证据一致）。

## 相关页面

- 机制一手出处：[Gated DeltaNet 报告](gated-delta-net.md)、[Gated Attention 报告](gated-attention.md)
- 概念：[线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md)、[注意力门控](../concepts/attention-gating.md)
- 模型：[Qwen3-Coder-Next](../models/qwen3-coder-next.md)、[Qwen3.5](../models/qwen3.5.md)
- 同家族来源：[Qwen3-Coder-Next 技术报告](qwen3-coder-next.md)、[Qwen3.5-Omni 技术报告](qwen3.5-omni.md)
