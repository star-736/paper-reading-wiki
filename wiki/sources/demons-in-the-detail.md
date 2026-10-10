---
type: Source
title: "Demons in the Detail"
description: "Qwen 团队 2025 年的 MoE load-balancing loss 实现论文：micro-batch LBL 几乎是序列级约束，会抑制领域专家专业化；改为同步 fi 的 global-batch LBL。"
tags: ["source", "moe", "load-balancing", "qwen"]
timestamp: 2026-10-11
resource: "../../raw/2501.11873v2.pdf"
---

# Demons in the Detail

## 来源

- 文件：`raw/2501.11873v2.pdf`
- 标题：Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models
- 版本：arXiv:2501.11873v2，2025-02-04
- 团队：Zihan Qiu、Zeyu Huang、Bo Zheng 等；Qwen Team（Alibaba）+ University of Edinburgh + Stanford
- 定位：**方法论文**。Qwen3 报告只引用这篇来说明自己用 global-batch load balancing loss，公式和表不在 [Qwen3](qwen3.md) 正文里。

## 核心结论

1. **常见实现把 LBL 算在 micro-batch 上。** 十亿级预训练的一个 micro-batch 往往只有很少几条序列，而且常按领域打包。于是 router 被要求在每条领域序列内部把 token 均匀分给所有 expert，代码或数学序列也做不到领域专业化（§1–§2.2）。
2. **global-batch LBL 只同步选择频率 $f_i$。** 用跨并行组的 $\bar f_i$ 代替各组自己的 $f_i^j$，负载平衡改到语料尺度。通信的是 $f_i\in\mathbb{R}^{N_E}$，不必传 token–expert 矩阵（§3，Eq. 4–6）。节点不够、Balance BSZ 凑不满 global batch 时，用 buffer 在 gradient accumulation 期间累加选择计数，做完一次更新后清零（Algorithm 1）。
3. **更大的 Balance BSZ 降低 held-out 平均 PPL，并让领域选择频率拉开。** 这是 3.4B 到约 43B 总参、最多 400B token 的研究模型，不是 Qwen3-235B 的复现。

## 架构与训练

标准 top-K MoE 的负载项沿用 Switch/GShard 形式（Eq. 2）：

$$
\mathrm{LBL}=N_E\sum_{i=1}^{N_E} f_i P_i
$$

$f_i$ 是 expert $i$ 被选中的 token 比例，$P_i$ 是分给它的平均 gating 概率。micro-batch 实现先在每个并行组内算，再对组取平均（Eq. 3）：

$$
\mathrm{LBL}_{\mathrm{micro}}=\frac{1}{N_P}\sum_{j=1}^{N_P}\left(N_E\sum_{i=1}^{N_E} f_i^j P_i^j\right)
$$

global-batch 把频率换成同步后的 $\bar f_i$（Eq. 4–6）。论文证明在这个写法下，它等于用全局频率去配各组自己的 $P_i^j$ 再平均，因此各组仍用本地 gating 分数，只多一次 $f_i$ 的通信。作者注明 GRIN（Liu et al., 2024b）独立提出过同样的同步。

**Balance BSZ** 指计算 $f_i$ 时纳入的 token 数，不是优化器的 global batch size 本身。序列长度 4096。120B token 设定的 global batch 为 512（约 60k step）；400B token 设定为 1024（约 100k step）。优化器是默认 AdamW。

## 后训练

不涉及。实验是预训练 MoE。

## 评测要点

零样本看 Hellaswag、MMLU、GSM8k、C-eval；分析主要看分领域 held-out PPL（SFT-EN、EN-Literature、SFT-Code、SFT-Math、SFT-ZH、ZH-Law、ZH-Literature、SFT-Other）的平均。Table 1：

| 模型 / 训练量 | 方法 | Balance BSZ | Avg PPL |
| --- | --- | ---: | ---: |
| 3.4A0.6B / 120B | LBL | 4 | 8.167 |
| 3.4A0.6B / 120B | LBL+sync | 512 | 8.038 |
| 3.4A0.6B / 400B | LBL | 4 | 7.347 |
| 3.4A0.6B / 400B | LBL+sync | 512 | 7.198 |
| 3.4A0.6B / 400B | LBL+buffer | 128 | 7.199 |
| 15A2.54B / 400B | LBL | 16 | 5.778 |
| 15A2.54B / 400B | LBL+sync | 512 | 5.603 |
| 43A6.6B / 120B | LBL | 8 | 5.862 |
| 43A6.6B / 120B | LBL+buffer | 128 | 5.779 |

摘要写的约 0.1 PPL 对应这种量级（例如 120B 的 3.4A0.6B 从 8.167 到 8.038）。**不能把每一列都读成随 BSZ 单调变好**：同一 400B 设定里，GSM8k 在 BSZ 128 为 28.81，到 BSZ 512 为 25.40，低于 128 但仍高于 micro-batch 的 21.30。论文说「各项一致变好」指的是 Table 1 第一块（120B、3.4A0.6B）。

Aux-free（Wang et al., 2024）在同一张表里也是 global 优于 micro：3.4A0.6B / 120B 的 Avg PPL 从 8.521（BSZ 4）到 8.080（BSZ 512）。这是这篇论文自己的对照，不是 DeepSeek 生产配方的复现。

领域专业化（Figure 1、§4.2）：micro-batch 下 EN-Literature 的选择频率几乎拉平；SFT-Code / SFT-Math 只有少数 expert 略高，且没有超过 0.15。global-batch 才看出领域差。摘要写最大模型到 42.8B 总参；表里对应行是 MoE-43A6.6B，且这一档只训了 120B token，400B 跑在 15A2.54B 上。

## 与已有沉淀的关系

- 这是 [Qwen3](qwen3.md)「global-batch load balancing loss」和 [Laguna](laguna-m1-xs2.md) aux loss 的一手公式，属于 **auxiliary loss** 阵营。它不更新 expert bias，不能并进 [Loss-Free Balancing](loss-free-balancing.md) 的 sign update。
- Loss-Free Balancing 批评的是 aux loss 干扰语言建模梯度。本页批评的是 **micro-batch 口径把平衡做成了序列级约束**。两件事都成立，解决的不是同一个问题。

## 相关页面

- 采用引用：[Qwen3 技术报告](qwen3.md)
- 概念：[MoE 负载均衡谱系](../concepts/moe-load-balancing.md)
- 对照路线：[Loss-Free Balancing](loss-free-balancing.md)
