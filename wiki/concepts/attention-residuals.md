---
type: Concept
title: "Attention Residuals"
description: "把标准残差（沿深度压成单一状态）换成每层选择性从所有前层检索表示——沿深度做 attention，类比 Transformer 把序列位置的 RNN 换成 attention。"
tags: ["concept", "attention-residuals", "depth-mixing"]
timestamp: 2026-07-30
---

# Attention Residuals

## 定义

**Attention Residuals（AttnRes）** 是一种跨层信息流机制：标准残差连接把所有先前层的信息压缩进单一状态 `h_l`（一个沿网络深度的 RNN 瓶颈），AttnRes 让**每一层选择性从所有前层检索表示**——把 Transformer 当年「用 attention 替代序列方向 RNN」的方法论搬到网络深度方向。

> 类比：序列方向上，RNN 把历史压进一个 hidden state（瓶颈），Transformer 用 attention 让每位置 attend 所有前位置（解除瓶颈）。深度方向上，标准残差把前层压进一个 residual stream（瓶颈），AttnRes 让每层 attend 所有前层（解除瓶颈）。

来源：[Kimi K3 技术报告](../sources/kimi-k3.md) § 2.2，引原始工作 [57]。K3 是首个把 AttnRes 用进 3T 级生产模型的开源报告。

## 机制（已据 Kimi K3 原文 § 2.2 核实）

### Full Attention Residuals

对层 `l`，定义层专属可学习 pseudo-query `q_l = w_l ∈ R^d`，keys 与 values 为：

```
k_i = v_i = { h_1            if i = 0       (token embedding)
            { f_i(h_i)       if 1 ≤ i ≤ l-1  (层 i 的输出)
```

attention 权重用 softmax kernel `ϕ(q, k) = exp(q^⊤ RMSNorm(k))`（RMSNorm 防止大输出层主导权重）：

```
α_{i→l} = ϕ(q_l, k_i) / Σ_{j=0}^{l-1} ϕ(q_l, k_j)
h_l = Σ_{i=0}^{l-1} α_{i→l} · v_i
```

网络深度适中（`L < 100`），`O(L²d)` 算力可承受；实际开销是 `O(Ld)` 内存（及 pipeline parallelism 下跨阶段通信）保留所有层输出。

### Block Attention Residuals（K3 实际用）

Full AttnRes 的 `O(Ld)` 内存 + pipeline 跨阶段通信在 93 层、2.8T 规模下不可忽略。Block AttnRes 把 L 层分成 N 个 block（每 block S=L/N 层）：

- **block 内**：层输出累加成单一表示 `b_n = Σ_{j∈B_n} f_j(h_j)`，`b_n^i` 表示 block n 前 i 层的 partial sum；设 `b_0 = h_1`（token embedding 恒为源）。
- **block 间**：仅对 N 个 block-level 表示做完整 attention。block n 的第 i 层，value 矩阵为：
  - `i = 1`（block 首层）：`V = [b_0, b_1, ..., b_{n-1}]^⊤`
  - `i ≥ 2`（block 后续层）：`V = [b_0, b_1, ..., b_{n-1}, b_n^{i-1}]^⊤`
- 最终输出层聚合所有 N 个 block 表示。

内存/通信从 `O(Ld)` 降到 `O(Nd)`；block 结构还 bound 推理时状态，使并行 inter-block 结果与顺序 intra-block partial sum 可经 **online softmax** 合并，显著降推理时延。

**K3 配置**：N=8 blocks × S=12 layers。93 层 → 8 个完整 block（各 12 层）+ 1 个 partial final block（9 层），含 embedding 共 9 个 block。报告引 [57] 称"N ≈ 8 recovers most of the benefit across model scales"。

## 为什么重要

- **解除深度方向的 RNN 瓶颈**。标准残差是沿深度的 RNN——信息逐层累加进单一 stream，深层要访问早期层信息只能从这份压缩态读。AttnRes 让深层直接 attend 任意前层，与 Transformer 当年解除序列方向 RNN 瓶颈同构。这在 93 层的 K3 里尤其有价值——深层不再受限于 residual stream 的信息瓶颈。
- **与 AttnRes 配套的 serving 优化**。Block AttnRes 的 `O(Nd)` 有界状态让推理时 online softmax 合并可行，K3 据此设计专用 kernel（prefill 用 sequence parallelism 避免每 TP rank materialize block 表示；decode 把 inter-block kernel 放 side stream 与主 stream 重叠，intra-block kernel 融合进 TP all-reduce + RMSNorm）。
- **与 KDA/MLA 的正交性**。AttnRes 作用在**深度维**，KDA/MLA 作用在**序列维**，Stable LatentMoE 作用在**宽度维**——K3 明确把架构分成三条信息流轴，AttnRes 负责其中一条。这种正交分解让三个机制可独立设计、独立 ablate。

## 跨报告信号

- **[Kimi K3](../sources/kimi-k3.md)（Moonshot AI，2026-07）**：本机制的生产级首个开源采用者。K3 把 AttnRes 与 KDA-MLA 混合注意力 + Stable LatentMoE 并列，明确作为"layer mixing"维度（Figure 2 架构图把 AttnRes 标为跨 block 的 layer mixing 路径）。EAGLE-3 draft model 的 feature fusion 也利用 AttnRes block 输出（取 1st/4th/final AttnRes block 作低/中/高层特征）。
- **原始工作 [57]**：K3 引用但未在报告内展开原始论文细节。N≈8 的经验结论、block size 选择依据均来自 [57]。**待追问**：[57] 的原始实验尺度是否覆盖 3T 级，N=8 在 93 层下是否仍最优。
- **[Linear Attention Architectures](../sources/linear-attention-architectures.md)（ETH Zurich，2026-07）**：给出适用于 DeltaNet / GDN 的另一条跨层路径 CLVR。它不对 layer outputs 做 depth attention，也不替换 residual sum；它取层内线性记忆正要写入的 $v_{l,t}$，经零初始化投影后**加到**共享 residual stream。350M–1.3B 的 single-run 结果中 CLVR 优于传 write error 的 CLER-H，但增益小且随规模 / token budget 缩小。这是“跨层可检索输出”与“跨层暴露内部写入量”两种不同接口的实证边界。
- **[Intern-S2-Mobius](../sources/intern-s2-mobius.md)（Shanghai AI Lab，2026-08）**：提出的 BRC 也针对层级信息瓶颈，但并不让深层 attention 聚合早层 activation。它把所有 FFN 横向拼成共享 knowledge Memory，让各层 Reasoner 检索同一参数化知识库；作者将此称为“间接”双向知识访问。因接口是共享 FFN 参数而非 depth-attention state，不能把 Mobius 的 35B conversion 结果当作 AttnRes 的独立复现或 ablation。

## 与 CLVR 的边界

AttnRes 和 CLVR 都试图缓解深度信息被单一 residual stream 稀释，但代价模型不同：

| 机制 | 跨层载荷 | 融合方式 | 对线性记忆的依赖 |
| --- | --- | --- | --- |
| AttnRes | 早期层 / block 的输出表示 | learned softmax depth attention，替换标准 residual sum | 无；可用于一般 Transformer 深度混合 |
| CLVR | DeltaNet-style memory 的内部 write value | 零初始化投影后加到共享 residual stream | 有；需能产生 $v_{l,t}$ 的 recurrent memory layer |

因此 CLVR 不是 AttnRes 的轻量实现：前者保留 host residual path 和线性时间 token mixer，代价是只暴露局部内部量；后者让每层按深度选择任意前层表示，表达力更直接但需要保留 / 聚合 depth representations（[Linear Attention Architectures](../sources/linear-attention-architectures.md) § 4、§ 6.6）。

## 待追问

- **N=8 / S=12 在 93 层 2.8T 规模下是否最优？** 报告引 [57] 说"N≈8 recovers most of the benefit across model scales"，但 [57] 原始实验尺度未在 K3 报告展开。层数从 [57] 实验尺度到 93 层，最优 N 是否该 scale？需查 [57] 原文。
- **AttnRes 的 ablation 增益**。K3 报告把 AttnRes 作为架构组成部分描述，但未给"有 AttnRes vs 无 AttnRes"的 head-to-head ablation 数字（validation loss / downstream benchmark）。2.5× scaling efficiency 是综合改进的合计，AttnRes 贡献占比未拆解。
- **Full vs Block AttnRes 的质量差距**。报告说 Block 是为降开销，N=8 recover most benefit，但"most"是定性。Full AttnRes 在 K3 规模下能多带来多少增益？
- **pseudo-query `q_l = w_l` 的作用**。每层一个可学习向量作 query，它学到的模式是什么？是否某些层倾向 attend 近层、某些 attend 远层（含 embedding）？可解释性分析缺。
- **与 DenseFormer / Feathers 等跨层连接工作的关系**。AttnRes 不是唯一做跨层信息流的工作，DenseFormer（dense connection）等走的是加法而非 attention。这些机制在 K3 规模下的对比缺。

## 相关页面

- 来源：[Linear Attention Architectures 技术报告](../sources/linear-attention-architectures.md)（CLVR 与 AttnRes 的受控对照）

- 来源：[Kimi K3 技术报告](../sources/kimi-k3.md)（§ 2.2 Attention Residuals）
- 模型：[Kimi K3](../models/kimi-k3.md)
- [Stable LatentMoE](stable-latentmoe.md)（K3 宽度维新机制，与 AttnRes 正交）
- [线性注意力与 delta rule](linear-attention-and-delta-rule.md)（K3 序列维机制，与 AttnRes 正交）
- [Multi-Head Latent Attention](multi-head-latent-attention.md)（K3 全局层底座）
- [Intern-S2-Mobius 技术报告](../sources/intern-s2-mobius.md)（共享 FFN Memory 的 BRC，与 AttnRes 的接口边界）
