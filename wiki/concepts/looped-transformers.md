---
title: "Looped Transformers"
type: Concept
description: "权重共享的循环 Transformer：用同一 block 反复执行增加有效深度，PLT 使延迟和 KV-cache 不随 loop count 增长"
tags: [looped-transformer, PLT, weight-sharing, test-time-compute, depth-recurrence]
timestamp: 2026-07-11
---

## 定义

**Looped Transformer** 用一个共享的 Transformer block $f_\theta$（L 层）反复执行 R 次来替代堆叠 R×L 个独立层，从而在不增加参数量的前提下增大有效计算深度：

$$h^{(0)} = \text{Embed}(x), \quad h^{(r)} = f_\theta(h^{(r-1)}), \quad r=1,\dots,R$$

参数量 N 固定，但有效深度 R·L 随 R 增长。这使得模型能在小参数预算上获得深度敏感任务的能力，是 **test-time compute scaling** 的一条路径——不生成额外推理 token，而在表征空间做迭代精炼（latent reasoning）。

## 谱系

### 标准 looped Transformer（序列式）

原始设计（Universal Transformer [Dehghani et al. 2018]）的 recurrence 是严格序列的：h(r) 必须在 h(r-1) 之后计算。代价随 R 线性增长：

- 延迟：O(R·C_block)——R 次顺序前向传播
- KV-cache：O(R·L·S·d)——每层每 loop 都缓存 KV

这使得深度 loop 在延迟/内存受限的部署中不实用。Giannou et al. [2023] 证明少量 encoder 层在 loop 中足以模拟通用指令集计算机；Yang et al. [2023] 证明 looped Transformer 用 <10% 参数在 in-context learning 上匹配标准 Transformer。

### 效率改进路线

| 方法 | 机制 | 解决的问题 |
|---|---|---|
| **MELT** [15] | 单个共享 KV cache per layer，learnable gating 更新 | KV-cache 随 R 线性增长 |
| **PLT** [16] | CLP（cross-loop position offset）打破序列依赖 + G-SWA（shared-KV gated SWA） | 延迟 + KV-cache 同时随 R 增长 |
| **LT2** [6] | 用线性/稀疏 attention 替代 softmax | loop 内 attention 的二次复杂度 |

**PLT 是唯一同时解决延迟和内存的方案**：CLP 使 loop 间可并行（延迟 ≈ C_block），G-SWA 冻结 loop-1 KV cache 共享给所有后续 loop（内存 O(L·S·d) 不随 R 增长）。

### PLT 的结构性代价：CLP offset

PLT 的并行化不是免费的。CLP 使 loop r≥2 的输入为：

$$B^{(r)} = \text{Embed}(x) + \text{shift}(h^{(r-1)})$$

token $x_i$ 在 loop r 收到的是 $x_{i-1}$ 的 loop-(r-1) hidden state 而非自身的。这引入了 **per-token positional mismatch**——标准 looped Transformer 不存在此成本，但也不并行。

LoopCoder-v2 的核心贡献是量化这个代价：定义 intrinsic offset cost Ω(r) = (1/S) Σ_i ‖h(r-1)_i - h(r-1)_{i-1}‖₂，发现 Ω(r) 在各 loop 间大致**恒定**。而每个 loop 的 refinement gain 急速递减。固定成本 + 递减收益 = gain–cost 剪刀，使 PLT 在 R=2 饱和。

## 跨报告信号

- **[LoopCoder-v2](../sources/loopcoder-v2.md)**（arXiv:2606.18023v1）：首个在 18T tokens 上从头训练 PLT coder 的大规模实验。7B 模型 R=2 最优（SWE-bench Verified 64.4%），R≥3 退化。gain–cost 框架 + per-loop 可解释性诊断（hidden-state dynamics / attention evolution / output-distribution shift 三镜头三角验证）。
- **Huginn-3.5B** [Geiping et al. 2025]：3.5B depth-recurrent Transformer，800B tokens 预训练，推理时最多 50 loops，等效 50B 参数计算预算。使用标准序列 loop（非 PLT），延迟随 loop count 线性增长。
- **Scaling law** [Schwethelm et al.]：loop 一个 block r 次等效 r^0.46 个独立参数层——远低于真正加层的线性等效。这从 scaling law 角度独立支持了 loop 收益递减的结论。
- **稳定性** [Yang et al.]：性能可能在中间 loop depth 达峰后崩溃，提出 fixed-point regularization 稳定循环动态。

## 为什么重要

Looped Transformer 代表了一种与本 wiki 已收录的效率路线**正交**的 test-time compute scaling 思路：

- **高效长上下文注意力**（DSA / MSA / 线性注意力）解决的是**单次前向传播内**如何高效处理长序列；
- **MTP**（multi-token prediction）解决的是**单次前向传播内**如何预测多个 token 摊销计算；
- **Looped Transformer** 解决的是**跨多次前向传播**如何迭代精炼表征——增加的是**深度**而非宽度或 token 数。

三者在理论上可以叠加：一个 looped Transformer 的每次 loop 内部可以用稀疏注意力，也可以结合 MTP。但 PLT 的 CLP offset 成本 Ω(r) 是其独有的结构性约束，决定了 loop count 的上限。

**Latent loop 与 explicit CoT 互补**：LoopCoder-v2 发现 explicit CoT + latent loop 在 R=2 时呈超加性（LiveCodeBench +26.9，远超各自单独增益之和）。两机制在不同粒度操作：CoT 分解问题为文本步骤，latent loop 精炼每步底层表征。这与 thinking models（如 [GLM-5](../models/glm-5.md) 的 thinking mode）的 explicit CoT 是互补而非竞争关系。

## 待追问

- **PLT + 稀疏注意力 / 线性注意力**：如果 PLT 的每次 loop 内部用 DSA 或 GDN 替代 full attention，Ω(r) 会改变吗？loop-count 饱和点会移动吗？
- **PLT + MTP 叠加**：latent loop 和 MTP 都是 test-time compute scaling 路径，两者是否如 latent loop + explicit CoT 一样互补？
- **CLP offset 自适应**：当前 Ω(r) 恒定是 CLP 固定右移 1 位的结果。如果 offset 量随 loop 自适应减小（如 learnable shift amount），能否推迟饱和点？
- **跨参数量 scaling**：7B 上 R=2 最优，更大模型上 gain–cost 交叉点是否会不同？Schwethelm 的 r^0.46 scaling law 是否暗示更大模型也需要更多 loop 才能饱和？

## 相关页面

- [LoopCoder-v2 来源页](../sources/loopcoder-v2.md) — PLT gain–cost 分析的一手出处
- [LoopCoder-v2 模型页](../models/loopcoder-v2.md) — 7B PLT coder 模型族
- [高效长上下文注意力](efficient-long-context-attention.md) — 正交路线：单次前向传播内的长序列效率
- [多 token 预测](multi-token-prediction.md) — 正交路线：单次前向传播内的多 token 摊销
- [Agentic 评测体系](agentic-evaluation-benchmarks.md) — LoopCoder-v2 在 SWE-bench / Terminal-Bench 上的数据点
