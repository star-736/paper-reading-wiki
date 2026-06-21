# 注意力门控

## 定义

「门控（gating）」是给某个张量乘上一个由输入算出的、值域受限的分数 $\sigma(XW)$，从而选择性地保留或抹掉它的特征：

$$Y' = Y \odot \sigma(XW)$$

门控在神经网络里历史悠久（LSTM、Highway、GRU），现代序列模型里到处都是。但「门到底起什么作用」长期缺乏系统研究。这一页收的是 **softmax 注意力里的门控**——即在标准 SDPA 流程的某个位置插入门，主要证据来自 [Gated Attention 报告](../sources/gated-attention.md)（Qwen 团队，NeurIPS 2025）。

> **与线性注意力的门别混**。[线性注意力与 delta rule](linear-attention-and-delta-rule.md) 里 GDN/KDA 的「遗忘门」$\alpha_t$ 控制的是 RNN 状态记忆的寿命；本页的门作用在 softmax 注意力的 SDPA 输出上，目标是**注入非线性 + 消除 attention sink**。同名「gating」，两条路线里干的是不同的事。

## 门可以加在哪、怎么加（设计空间）

Gated Attention 报告沿五个维度遍历了 30 个变体：

| 维度 | 取值（**粗体 = 最优默认**） |
| --- | --- |
| 位置 | **G1（SDPA 输出后）**、G2/G3/G4（V/K/Q 投影后）、G5（最终输出后） |
| 粒度 | headwise（整头一个标量）↔ **elementwise**（逐维，与 Y 同维） |
| head 特异性 | **head-specific**（每头独立门）↔ head-shared |
| 形式 | **multiplicative** $Y\odot\sigma(XW)$ ↔ additive $Y+\sigma(XW)$ |
| 激活 | **sigmoid**（[0,1]，配乘性）↔ SiLU（无界，配加性） |

**结论：head-specific 的 elementwise sigmoid 门、加在 SDPA 输出后（G1），效果最好**——PPL 降最多 0.2、MMLU 涨 2 分，且配略大学习率更稳。这个 G1 输出门已被用进 **Qwen3-Next**。

## 为什么有效：非线性 + 稀疏（去 sink）

报告把效果归到两个独立因素：

1. **非线性**：softmax 注意力里 value 投影 $W_V$ 与输出投影 $W_O$ 两个连续线性层可合并成一个低秩线性映射。在 G1/G2 插门，就是给这个低秩变换注入非线性，提升表达力。
2. **稀疏 + 消除 attention sink**：门分数本身高度稀疏，给 SDPA 输出加上 input-dependent 稀疏。此前研究把 **attention sink**（首 token 吸走大量注意力）和 **massive activation**（首 token 隐状态值异常大）归因于 softmax 非负归一化的冗余堆积。实测：baseline 平均 **46.7%** 注意力被首 token 吸走（某层 83%），加 query-dependent 稀疏门后降到 **4.8%**（该层 4%），dense 和 MoE 模型（3.5T token）都不再出现 attention sink。

副产品：attention-sink-free 的模型**长度外推**显著更好，RULER 涨 10+ 分。

## 跨报告信号

- **[Gated Attention](../sources/gated-attention.md)（Qwen 团队，2025，NeurIPS 2025 Best Paper）**：本页主证据。系统消融 30 个门控变体，确立「SDPA 输出 head-specific sigmoid 门」为最优，归因到非线性 + 稀疏去 sink。该工作拿到 NeurIPS 2025 Best Paper Award（先获 Oral，top 1.5%），其 G1 输出门是 Qwen3-Next 系架构的组成部分。
- **采用它的生产模型（G1 输出门已落地）**：
  - **Qwen3-Next**（80B-A3B）：3:1 混合栈——3 层 [GDN 线性注意力](linear-attention-and-delta-rule.md) + 1 层带输出门的 gated full attention。
  - **[Qwen3-Coder-Next](../sources/qwen3-coder-next.md)**（80B-A3B，编码 agent）：明确「based on Qwen3-Next」，继承同一 hybrid attention MoE。
  - **[Qwen3.5-Omni](../sources/qwen3.5-omni.md)**（全模态）：Thinker/Talker 都用 Qwen3.5 的 Hybrid Attention MoE（含 GDN + gated attention），把它延伸到长音视频序列。
  - **Trinity Large**（Arcee AI，400B MoE / 13B active，**非 Qwen**）：独立采用者，且用法不同——不在混合栈里，而是在更常规的 full-attention 栈里用「SDPA 输出后、输出投影前」的门。说明 gated attention 不是 Qwen 专属技巧。（来源：[Sebastian Raschka, Gated Attention](https://sebastianraschka.com/llm-architecture-gallery/gated-attention)）
- **[Kimi Linear](../sources/kimi-linear.md)（KDA）**：两层关系。其一，KDA 在输出投影前也用了 **data-dependent sigmoid 输出门**（低秩参数化），报告明说目的之一是**缓解 attention sink**——与 Gated Attention 完全独立的团队/架构上得到同一类结论，互为佐证。其二，据第三方分析，Kimi Linear 本质是把 **Qwen3-Next 那个 gated-attention 全局层换成了 MLA**——两者是「同一混合骨架、不同全局层」的对照（来源：[Sebastian Raschka, Beyond Standard LLMs](https://magazine.sebastianraschka.com/p/beyond-standard-llms)）。
- **延伸**：NSA、Switch Heads 等也带门控，但常和稀疏/路由耦合在一起。Gated Attention 的贡献正是把「门本身的价值」从路由/稀疏里**解耦**出来（它发现 Switch Heads 退化到单 expert、门只调制 value 输出时增益仍在）。

## 为什么重要

- **一个近乎免费的质量 + 稳定性升级**。G1 门改动极小（一个 sigmoid 投影），却同时拿到 PPL/MMLU 提升、loss spike 消除、可用更大学习率、更好 scaling——这是为什么它能直接进 Qwen3-Next 这种生产模型。
- **它给「attention sink 是不是必要的」一个反例**。registers/StreamingLLM 一派把 sink 当成「有用的注意力垃圾桶」；这里证明 query-dependent 稀疏门可以直接消掉 sink 而质量更好，长度外推还更强。
- **它和线性注意力的门是同一思想的两次独立印证**。两条路线（softmax / 线性 RNN）都发现「在输出端加数据相关的门」有用，提示门控可能是比具体注意力形式更底层的有效组件。

## 待追问

- G1 elementwise head-specific 门在大模型上的真实参数/推理增量有多少？论文主打「简单」，但需核对配置表。
- 消除 attention sink 后，原本依赖 sink 做「注意力缓冲」的机制如何补偿？报告是经验观测，机制论证可深挖。
- 论文消融止于 15B；更大尺度上「去 sink → 长度外推增益」是否成立，Qwen3-Next/Qwen3.5（397B 级）已采用该门作为间接证据，但这些报告**继承而非重新验证**该收益，仍缺一个大尺度上专门复测「去 sink」效应的公开数据。

## 相关页面

- 来源：[Gated Attention 技术报告](../sources/gated-attention.md)、[Kimi Linear 技术报告](../sources/kimi-linear.md)、[Qwen3-Coder-Next](../sources/qwen3-coder-next.md)、[Qwen3.5-Omni](../sources/qwen3.5-omni.md)
- [线性注意力与 delta rule](linear-attention-and-delta-rule.md)（线性注意力里的「门」是另一回事；Qwen3-Next 系把 gated attention 和 GDN 配在一起）
- [高效长上下文注意力](efficient-long-context-attention.md)（去 sink 改善长度外推）
- 模型：[Kimi Linear](../models/kimi-linear.md)
