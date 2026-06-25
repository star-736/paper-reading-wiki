---
type: Source
title: "DeepSeek-V2 技术报告"
description: "DeepSeek-V2 的 arXiv 论文，Multi-Head Latent Attention（MLA）的首次提出处。"
tags: ["source", "deepseek-v2"]
timestamp: 2026-06-20
resource: "../../raw/DeepSeek-AI%20%E7%AD%89%20-%202024%20-%20DeepSeek-V2%20A%20strong%2C%20economical%2C%20and%20efficient%20mixture-of-experts%20language%20model.pdf"
---

# DeepSeek-V2 技术报告

## 来源

- 原始 PDF：[raw/DeepSeek-AI 等 - 2024 - DeepSeek-V2 A strong, economical, and efficient mixture-of-experts language model.pdf](../../raw/DeepSeek-AI%20%E7%AD%89%20-%202024%20-%20DeepSeek-V2%20A%20strong%2C%20economical%2C%20and%20efficient%20mixture-of-experts%20language%20model.pdf)
- 标题：DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
- 版本/日期：arXiv:2405.04434，2024-05
- 团队：DeepSeek-AI
- 关键身份：**MLA（Multi-Head Latent Attention）的首次提出处**，也是 DeepSeekMoE 的奠基论文之一。后续 [DeepSeek-V3.2](deepseek-v32.md) 的 DSA、[DeepSeek-V4](deepseek-v4.md) 的 CSA 都架在 MLA 之上。
- 概念页：[Multi-Head Latent Attention](../concepts/multi-head-latent-attention.md)

## 核心结论

DeepSeek-V2 是 236B 总参 / 21B 激活的 MoE 模型，128K 上下文，预训练 8.1T tokens。它的两个架构主张：

- **MLA**：通过对 KV 做低秩联合压缩，在推理时把 KV cache **降低 93.3%**、最大生成吞吐**提升到 5.76×**，同时性能**强于 MHA**。
- **DeepSeekMoE**：细粒度专家 + 共享专家，使训练成本相对 dense 同级模型节省 42.5%。

报告明确指出：GQA / MQA 虽然减小 KV cache，但**性能达不到 MHA**（附录 D.1 用 7B dense 模型在四个 hard benchmark 上的消融证明 MHA 显著优于 GQA / MQA）。MLA 的设计目标就是「鱼与熊掌兼得」——KV cache 比 GQA/MQA 还小，质量却超过 MHA。

## 架构与训练

### MLA：低秩 KV 联合压缩（§2.1.2）

- 把输入 hidden $h_t$ 下投影成一个**压缩 latent 向量** $c^{KV}_t$（维度 $d_c$，远小于 $n_h d_h$），**cache 里只存这个 latent**。用时再通过上投影 $W^{UK}$、$W^{UV}$ 恢复出 per-head 的 K、V。
- **矩阵吸收**：推理时 $W^{UK}$ 可吸进 query 投影 $W^Q$、$W^{UV}$ 可吸进输出投影 $W^O$，于是不必显式还原 per-head K/V，直接对 latent 算注意力——这正是 [V3.2 所说的「MQA mode」](deepseek-v32.md)的来源。
- query 侧也做**低秩压缩**（latent $c^Q_t$，维度 $d_c'=1536$），但目的与 KV 那条**完全不同**：原文明说是「为减少训练时的 activation memory，即便压不了 KV cache」（§2.1.2："in order to reduce the activation memory during training … **even if it cannot reduce the KV cache**"）。V2 的展开 query 是 128 head × 128 = **16384 维**（比 hidden 5120 还大 3.2×），故吃显存。**旁证**：V2-Lite「it does not compress the queries」——小模型激活压力小就不压，反证此步纯为训练显存、与 cache / 能力无关。（注：论文只给「压缩 → 省显存」结论，**具体怎么省的机制未展开**；「存细腰 latent + 重算大 query」是合理推测，见 [MLA 概念页待追问](../concepts/multi-head-latent-attention.md#待追问)。）

### Decoupled RoPE（§2.1.3）

RoPE 与低秩 KV 压缩**不兼容**：若直接对 K 施加 RoPE，位置敏感的 RoPE 矩阵会卡在 $W^{UK}$ 中间，导致它**无法再被吸收**进 $W^Q$。解法是 **decoupled RoPE**——额外引入一组 multi-head **decoupled query** $q^R$ 和一个**所有 head 共享的 decoupled key** $k^R$ 专门承载 RoPE；最终的 K/Q 是「压缩部分（可吸收，不带 RoPE）+ decoupled 部分（带 RoPE）」拼接。decoupled key 也要进 cache，故每 token 总 KV cache = $(d_c + d_R)$ 个元素。

### KV cache 对比（Table 1）

| 注意力 | 每 token KV cache（元素数）| 能力 |
| --- | --- | --- |
| MHA | $2 n_h d_h$ | 强 |
| GQA（$n_g$ 组）| $2 n_g d_h$ | 中 |
| MQA | $2 d_h$ | 弱 |
| **MLA** | $d_c + d_R \approx \tfrac{9}{2} d_h$ | **强（> MHA）** |

报告原文：MLA 的 KV cache **等效于只有 2.25 组的 GQA**，但性能强于 MHA。

### 训练

8.1T tokens 预训练；pipeline parallelism 部署不同层；DeepSeekMoE 用 2 shared + 160 routed expert 配置。128K 上下文经 Needle-in-a-Haystack 压力测试。

## 后训练

SFT + GRPO 强化学习（DeepSeek 自家 RL 算法），对齐到对话与人类偏好。本页聚焦 MLA 架构，后训练细节从略——如需可单独 deepen。

## 评测要点

- 仅 21B 激活参数即在开源模型中达到 top-tier；KV cache −93.3%、吞吐 5.76× 是其「economical / efficient」定位的核心数字。
- 附录 D.1（MHA vs GQA vs MQA 消融）是理解 MLA 动机的关键：它用实验说明「减头数」路线（GQA/MQA）有质量代价，从而论证 MLA「压低秩」这条正交轴的必要性。

## 待追问

- MLA 的 $d_c$、$d_c'$、$d_R$ 具体取值与各上/下投影维度，本页未抄全；如需精确复现可回 §2.1.2–2.1.3 与配置表补。
- 附录 D.2「MLA vs MHA」的具体数字未沉淀，可作为「MLA 真的 > MHA」这一主张的证据补充。
- DeepSeek-V2 → V3 → V3.2（DSA）→ V4（CSA）这条 MLA 演进链的每一步「在 MLA 上加了什么」，可在 [MLA 概念页](../concepts/multi-head-latent-attention.md) 做一张演进表。

## 相关页面

- 概念：[Multi-Head Latent Attention](../concepts/multi-head-latent-attention.md)、[DeepSeek Sparse Attention](../concepts/deepseek-sparse-attention.md)、[高效长上下文注意力](../concepts/efficient-long-context-attention.md)
- 后续来源：[DeepSeek-V3.2](deepseek-v32.md)、[DeepSeek-V4](deepseek-v4.md)
- 比较：[稀疏注意力机制对比](../comparisons/sparse-attention-mechanisms.md)
