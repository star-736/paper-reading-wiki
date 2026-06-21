# Gated DeltaNet 技术报告

## 来源

- 原始 PDF：[raw/Yang 等 - 2025 - GATED DELTA NETWORKS IMPROVING MAMBA2 WITH DELTA RULE.pdf](../../raw/Yang%20%E7%AD%89%20-%202025%20-%20GATED%20DELTA%20NETWORKS%20IMPROVING%20MAMBA2%20WITH%20DELTA%20RULE.pdf)
- 标题：Gated Delta Networks: Improving Mamba2 with Delta Rule
- 版本/会议：ICLR 2025
- 团队：Songlin Yang（MIT CSAIL）、Jan Kautz、Ali Hatamizadeh（NVIDIA）
- 配套实现：[NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)
- 概念页：[线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md)

## 核心结论

GDN 是 [线性注意力](../concepts/linear-attention-and-delta-rule.md) 演进链上的关键一环，也是 [Kimi Linear](kimi-linear.md) 的 KDA 和 Qwen3-Next 系列线性层的**直接前身**。它的中心观察是：线性注意力此前补救质量的两条路——**门控（gating，自适应记忆控制）** 和 **delta rule（精确记忆修改）**——是**互补**的：

- **门控**（如 Mamba2 的标量衰减 $\alpha_t$）能**快速清空整块记忆**，但「一视同仁」地衰减所有 key-value 关联，无法定向遗忘某一条。
- **delta rule**（DeltaNet）能**定向替换**单条 key-value 关联（精确更新），但一次只动一条，缺乏在 context 切换时快速清空旧信息的机制。

GDN 把两者合成 **gated delta rule**，一个规则同时拥有「快速清空」和「定向更新」两种能力，并配套 chunkwise 并行训练算法保证硬件效率。结果：在语言建模、常识推理、in-context retrieval、长度外推、长上下文理解多个 benchmark 上一致超过 Mamba2 和 DeltaNet；进一步用 GDN 层 + 滑窗注意力或 Mamba2 层的混合架构再提升效率与质量。

## 机制（已据原文核实）

逐级看三条递推（$S_t\in\mathbb{R}^{d_v\times d_k}$ 是矩阵状态，$o_t=S_t q_t$）：

| 机制 | 状态更新 | 遗忘能力 | 局限 |
| --- | --- | --- | --- |
| Mamba2（门控/衰减） | $S_t = \alpha_t S_{t-1} + v_t k_t^\top$，$\alpha_t\in(0,1)$ 数据相关标量 | 整块按 $\alpha_t$ 统一衰减 | 无法定向遗忘单条关联 |
| DeltaNet（delta rule） | 软替换：把旧的 $k\to v$ 关联朝新的 $k_t\to v_t$ 纠正 | 定向、精确 | 一次只动一条，context 切换时清不掉旧信息 |
| **Gated DeltaNet（本文）** | gated delta rule：在 delta 更新前先乘一个**数据相关标量门** $\alpha_t$ | 两者兼得 | —— |

直觉（原文 § 1）：gated delta rule 让记忆控制变得灵活——令 $\alpha_t\to 0$ 可瞬间清空记忆；令 $\alpha_t\to 1$ 则退化成纯 delta rule，只定向更新某条内容而不动其他。

**为什么线性注意力需要遗忘**：线性 Transformer 本质是外积形式的 key-value 关联记忆，能存的正交 key-value 对数受限于模型维度；序列长度超过维度时「memory collision」不可避免，精确检索失效（原文引 Schlag 等）。门控 + delta rule 正是为缓解这个有限状态瓶颈。

**硬件效率**：在 DeltaNet 用 WY representation 并行化 delta 计算的基础上，GDN 把门控项也纳入 chunkwise 并行形式，保持线性时间训练。

## 与已有沉淀的关系

- **GDN → KDA 这条线是本 wiki 线性注意力页的主轴**。[Kimi Linear](kimi-linear.md) 的 KDA 明确是「extends Gated DeltaNet with a finer-grained gating mechanism」——把 GDN 的 **head-wise 标量门** $\alpha_t$ 升级成 **channel-wise 细粒度门** $\mathrm{Diag}(\alpha_t)$。本报告把那条演进链的中间一环从二手转述升级为 tier-1 原文确证，详见 [线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md)。
- **GDN 是 Qwen3-Next 系列线性层的底座**。[Qwen3-Coder-Next](qwen3-coder-next.md)、[Qwen3.5-Omni](qwen3.5-omni.md) 的 hybrid 栈里那一支「线性层」就是 GDN 模块。
- **混合思路同源**：GDN 论文自己就提出「GDN 层 + 滑窗注意力 / Mamba2 层」的混合架构；Kimi Linear（GDN-style 线性 + MLA）、Qwen3-Next（GDN + gated attention）都是这一思路在生产模型上的放大。

## 待追问

- GDN 的标量门是 head-wise；KDA 改成 channel-wise。报告里 GDN 自己有没有讨论过 channel-wise 门、为何最终选标量？需读 § 方法与消融。
- 论文做的混合是 GDN + SWA / Mamba2；与后来 Kimi Linear 选 GDN-style + MLA、Qwen3-Next 选 GDN + gated full attention 相比，混合「另一支」用什么差异有多大？
- GDN 的具体实验规模（参数量、训练 token）与 Kimi Linear/Qwen3-Next 生产尺度差距，决定了「小尺度结论能否外推」，需补主表数字。
