# Gated Attention 技术报告

## 来源

- 原始 PDF：[raw/Qiu 等 - Gated attention for large language models Non-linearity, sparsity, and attention-sink-free.pdf](../../raw/Qiu%20%E7%AD%89%20-%20Gated%20attention%20for%20large%20language%20models%20Non-linearity%2C%20sparsity%2C%20and%20attention-sink-free.pdf)
- 标题：Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free
- 版本/会议：NeurIPS 2025
- 团队：Qwen Team（Alibaba Group），合作 University of Edinburgh、Stanford、MIT、Tsinghua
- 荣誉：**NeurIPS 2025 Best Paper Award**（先入选 Oral，top 1.5%，77/5290）
- 配套实现：开源代码 [qiuzh20/gated_attention](https://github.com/qiuzh20/gated_attention)；该 SDPA 输出门已用于 **Qwen3-Next** 及其衍生模型

## 核心结论

这是一篇**系统性消融研究**，而非新模型报告：在 3.5T token 数据上，对 **30 个 gating 变体**做对照实验（15B MoE / 15A2B 与 1.7B dense 两套模型），系统拆解「在 softmax 注意力里加门」到底带来什么。

中心发现：一个极简改动——**在 SDPA 输出后加一个 head-specific 的 sigmoid 门**（论文记作 G1 位置）——稳定地提升性能（PPL 降最多 0.2、MMLU 涨 2 分），同时增强训练稳定性、容忍更大学习率、改善 scaling。论文把效果归因到两个因素：**(1) 非线性**——给 softmax 注意力里 value + dense 两个线性层构成的低秩映射注入非线性；**(2) 稀疏性**——query-dependent 的稀疏门分数调制 SDPA 输出，引入输入相关的稀疏。

更引人注目的副产品：这个稀疏门**消除了 massive activation 和 attention sink**。baseline 模型平均 46.7% 的注意力被首 token 吸走（某层高达 83%），加门后降到 4.8%（该层降到 4%）；且 attention-sink-free 的模型在长度外推上 RULER 涨 **10+ 分**。

## 门的设计空间（五个维度）

论文沿五个维度遍历 gating 变体：

| 维度 | 取值 |
| --- | --- |
| **位置** | G1（SDPA 输出后，最优）、G2/G3/G4（V/K/Q 投影后）、G5（最终 concat 输出后） |
| **粒度** | headwise（一个标量门整头）↔ elementwise（与 Y 同维、逐维调制） |
| **head 特异性** | head-specific（每头独立门）↔ head-shared（跨头共享 W 与门分数） |
| **乘性 / 加性** | multiplicative $Y'=Y\odot\sigma(XW)$ ↔ additive $Y'=Y+\sigma(XW)$ |
| **激活** | sigmoid（[0,1]，配乘性）↔ SiLU（无界，配加性） |

门的通式：$g(Y,X,W)=Y\odot\sigma(XW)$，其中 $X$ 取 pre-norm 后的 hidden state。默认配置：**head-specific、multiplicative、sigmoid**。**实践建议：用 elementwise 的 SDPA 输出门（G1），并配略大的学习率。**

## 两个机制因素

1. **非线性补偿低秩**：value 投影 $W_V$ 与 dense 输出投影 $W_O$ 两个连续线性层可合并成一个低秩线性映射；在 G1/G2 位置插入门的非线性，直接提升这个低秩变换的表达力。
2. **稀疏调制 + 消除 sink**：门分数本身高度稀疏，给 SDPA 输出加上 input-dependent 稀疏。此前工作把 attention sink 解释为「softmax 非负归一化导致的冗余注意力堆积」——当 query-dependent 稀疏门作用在 SDPA 输出上时，dense 和 MoE 模型（3.5T token）实测**不再出现 attention sink**，长度泛化随之大幅改善。

## 实验配置

- **MoE**：15B 总参 / 2.54B 激活（15A2B），128 expert top-8，fine-grained expert、global-batch LBL、z-loss，注意力用 GQA。
- **Dense**：1.7B。
- 数据：3.5T token（训练稳定性图覆盖 3T）。

## 与已有沉淀的关系

- 与 [Kimi Linear](kimi-linear.md) 是「门控」这条主线的两个独立证据：Kimi Linear 在 KDA 输出端也用了 data-dependent sigmoid 输出门来缓解 attention sink，与本文 G1 输出门同源。两者一起支撑 [注意力门控](../concepts/attention-gating.md) 这个概念页。
- 本文是 softmax 注意力上的「门控」，区别于线性注意力里的「遗忘门 / decay 门」（GDN、KDA）——同名「gating」在两条路线里目标不同（这里是非线性 + 去 sink，那里是控制 RNN 状态记忆寿命）。这层区分写在 [注意力门控](../concepts/attention-gating.md) 里。
- attention sink 的消除对长上下文外推有用，与 [高效长上下文注意力](../concepts/efficient-long-context-attention.md) 关心的长上下文质量有交集，但本文不改注意力复杂度（仍是 full softmax），所以它是「质量/稳定性」改进而非「效率」路线。
- **采用谱系**：G1 输出门已落到 Qwen3-Next、[Qwen3-Coder-Next](qwen3-coder-next.md)、[Qwen3.5-Omni](qwen3.5-omni.md)（均与 [GDN 线性层](gated-delta-net.md) 配成 3:1 混合栈），以及非 Qwen 的 Trinity Large（常规全注意力栈）。完整采用表见 [注意力门控](../concepts/attention-gating.md) 的跨报告信号。

## 待追问

- G1 elementwise 门额外参数量与推理开销具体多少？论文主打「简单」，但 head-specific elementwise 门在大模型上的真实增量值得核对配置表。
- attention sink 消除后，原本被认为「sink 是有用的注意力垃圾桶」的那派观点（registers/StreamingLLM）在这个框架下如何解释？论文给的是经验观测，机制论证可再深挖。
- 该门已进 Qwen3-Next 系（含 Qwen3-Coder-Next、Qwen3.5-Omni）和 Trinity Large，但论文消融只到 15B 尺度；更大尺度上「去 sink → 长度外推增益」是否同样成立，这些采用方的报告**继承而非重新验证**该收益，仍缺大尺度的专门复测数据。
