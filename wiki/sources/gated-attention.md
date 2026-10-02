---
type: Source
title: "Gated Attention 技术报告"
description: "Qwen 团队系统消融 30 个门控变体，SDPA 输出 head-specific sigmoid 门最优（注入非线性 + 消除 attention sink），NeurIPS 2025 Best Paper，已用于 Qwen3-Next 系与 Trinity Large。"
tags: ["source", "gated-attention"]
timestamp: 2026-06-21
resource: "../../raw/Qiu%20%E7%AD%89%20-%20Gated%20attention%20for%20large%20language%20models%20Non-linearity%2C%20sparsity%2C%20and%20attention-sink-free.pdf"
---

# Gated Attention 技术报告

## 来源

- 原始 PDF：[raw/Qiu 等 - Gated attention for large language models Non-linearity, sparsity, and attention-sink-free.pdf](../../raw/Qiu%20%E7%AD%89%20-%20Gated%20attention%20for%20large%20language%20models%20Non-linearity%2C%20sparsity%2C%20and%20attention-sink-free.pdf)
- 标题：Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free
- 版本/会议：NeurIPS 2025
- 团队：Qwen Team（Alibaba Group），合作 University of Edinburgh、Stanford、MIT、Tsinghua
- 荣誉：**NeurIPS 2025 Best Paper Award**（先入选 Oral，top 1.5%，77/5290）
- 配套实现：开源代码 [qiuzh20/gated_attention](https://github.com/qiuzh20/gated_attention)；该 SDPA 输出门已用于 **Qwen3-Next** 及其衍生模型

## 核心结论

这是一篇**系统性消融研究**，而非新模型报告：对 **30 个 gating 变体**做对照实验，使用不同训练预算（MoE 主表为 400B tokens，dense 实验最高 3.5T tokens）（15B MoE / 15A2B 与 1.7B dense 两套模型），系统拆解「在 softmax 注意力里加门」到底带来什么。

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
2. **稀疏调制 + 消除 sink**：门分数本身高度稀疏，给 SDPA 输出加上 input-dependent 稀疏。此前工作把 attention sink 解释为「softmax 非负归一化导致的冗余注意力堆积」——当 query-dependent 稀疏门作用在 SDPA 输出上时，所测 dense / MoE 配置实测 attention sink 显著减弱，长度泛化随之大幅改善。

## 实验配置

- **MoE**：15B 总参 / 2.54B 激活（15A2B），128 expert top-8，fine-grained expert、global-batch LBL、z-loss，注意力用 GQA。
- **Dense**：1.7B。
- 数据：MoE Table 1 为 400B tokens；dense Table 2 分别有 400B、1T 与 3.5T 设置，不能把全部变体视为均训练了 3.5T。

## 门控参数与成本边界

**已据原文核实（`supported`）**：Table 1 的 MoE 使用 24 层、hidden size 2048、32 个 query heads、4 个 KV heads、head dim 128（Table 7）。G1 的门作用于 SDPA 输出，因此逐元素门覆盖的是 query heads，而不是只有 4 个 KV heads。

| 方法（Table 1） | 门分数形状 | 新增参数（约，M） | Avg PPL | MMLU |
| --- | --- | ---: | ---: | ---: |
| Baseline | 无 | 0 | 6.026 | 58.79 |
| G1 elementwise，逐头逐维 | $n\times32\times128$ | 201 | 5.761 | 60.82 |
| G1 headwise，每头一个标量 | $n\times32$ | 1.6 | 5.792 | 60.05 |
| G2 elementwise，value 输出门 | $n\times4\times128$ | 25 | 5.820 | 59.17 |

**本页计算**：按 Eq. 5 的 $XW_\theta$，只计无 bias 的门投影权重，G1 elementwise 为 $24\times2048\times(32\times128)=201{,}326{,}592$ 个参数；headwise 为 $24\times2048\times32=1{,}572{,}864$，两者相差 128 倍。G2 elementwise 则为 $24\times2048\times(4\times128)=25{,}165{,}824$。这些结果对应原表的约数；201M 相对约 15B 总参数为 1.34%，不是零成本。

Dense 实验（§3.2.2、Table 2）**通过缩小 FFN 宽度保持总参数不变**，因此“1.7B 加门后仍是 1.7B”不表示门没有参数，而是把预算从 FFN 分给门。MoE 表的新增参数口径与 dense 表的固定总参数口径必须分开。

**延迟证据的范围**：§3.1 在实验设置中自报 gating 带来的 wall-time latency 小于 2%。这支持论文设置下低开销的描述，但该句没有给独立的 prefill / decode、batch、硬件与上下文长度分项测量。不能把它当成任意服务场景下“推理开销恒小于 2%”的保证；部署成本仍见待追问。

## 与已有沉淀的关系

- 与 [Kimi Linear](kimi-linear.md) 是「门控」这条主线的两个独立证据：Kimi Linear 在 KDA 输出端也用了 data-dependent sigmoid 输出门来缓解 attention sink，与本文 G1 输出门同源。两者一起支撑 [注意力门控](../concepts/attention-gating.md) 这个概念页。
- 本文是 softmax 注意力上的「门控」，区别于线性注意力里的「遗忘门 / decay 门」（GDN、KDA）——同名「gating」在两条路线里目标不同（这里是非线性 + 去 sink，那里是控制 RNN 状态记忆寿命）。这层区分写在 [注意力门控](../concepts/attention-gating.md) 里。
- attention sink 的消除对长上下文外推有用，与 [高效长上下文注意力](../concepts/efficient-long-context-attention.md) 关心的长上下文质量有交集，但本文不改注意力复杂度（仍是 full softmax），所以它是「质量/稳定性」改进而非「效率」路线。
- **采用谱系**：G1 输出门已落到 Qwen3-Next、[Qwen3-Coder-Next](qwen3-coder-next.md)、[Qwen3.5-Omni](qwen3.5-omni.md)、[Qwen3.8-Flash-Next](qwen3.8-next.md)（均与 [GDN 线性层](gated-delta-net.md) 配成 3:1 混合栈），以及非 Qwen 的 Trinity Large（常规全注意力栈）。3.8 还把同类 sigmoid 门用在残差读（Gated Residual）。完整采用表见 [注意力门控](../concepts/attention-gating.md) 的跨报告信号。

## 待追问

- **需实验或作者披露**：G1 elementwise / headwise 的参数已由 Table 1/7 核实；还需同一硬件、模型、batch 与上下文下的 prefill / decode 延迟、吞吐及峰值显存曲线。§3.1 的 wall-time <2% 自报没有这些分项，不能直接外推到更大生产模型。
- **需实验或作者披露**：attention sink 消除后，原本被认为「sink 是有用的注意力垃圾桶」的那派观点（registers/StreamingLLM）在这个框架下如何解释？论文给的是经验观测，机制论证可再深挖。
- **需实验或作者披露**：该门已进 Qwen3-Next 系（含 Qwen3-Coder-Next、Qwen3.5-Omni）和 Trinity Large，但论文消融只到 15B 尺度；更大尺度上「去 sink → 长度外推增益」是否同样成立，这些采用方的报告**继承而非重新验证**该收益，仍缺大尺度的专门复测数据。Qwen3.5 已到 397B 级；Qwen3.8-Flash-Next 的新增门控证据集中在稳定性（GatedNorm / GR），不是重新统计 attention sink。

关联提问页：[注意力门控](../concepts/attention-gating.md#相关追问)。
