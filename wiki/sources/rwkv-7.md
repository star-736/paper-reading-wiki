---
type: Source
title: "RWKV-7 Goose"
description: "Peng 等 2025：把 delta rule 推广成向量衰减加秩一修正的状态转移，转移矩阵不是 Householder。0.19B–2.9B，常数内存推理。"
tags: ["source", "rwkv", "delta-rule", "linear-attention"]
timestamp: 2026-10-11
resource: "../../raw/2503.14456v2.pdf"
---

# RWKV-7 Goose

## 来源

- 文件：`raw/2503.14456v2.pdf`
- 标题：RWKV-7 "Goose" with Expressive Dynamic State Evolution
- 版本：arXiv:2503.14456v2，2025-03-30
- 团队：Bo Peng、Ruichong Zhang、Daniel Goldstein 等（RWKV Project / EleutherAI / 清华等）
- 代码与权重：论文写 Apache 2.0，[RWKV/RWKV-LM](https://github.com/RWKV/RWKV-LM)，权重在 Hugging Face 的 RWKV 组织
- 定位：**架构论文**。[RWKV](rwkv.md) 那一页是 EMNLP 2023 的 RWKV-4，状态是 channel-wise 向量。本页的 WKV 才是矩阵状态上的 generalized delta rule。

## 核心结论

1. **generalized delta rule 是对角加秩一，不是 GDN 的标量 Householder。** 通式 $S_t=S_{t-1}(\mathrm{diag}(w_t)+z_t^\top b_t)+v_t^\top k_t$（§3）。RWKV-7 选用的实例把移除键和写入键解开，转移矩阵的特征值可以落在 $[-1,1]$，而不只剩「一个方向被擦、其余特征值全是 1」。
2. **因此它能做状态跟踪并识别全部正则语言**（在 $\mathrm{TC}^0\neq\mathrm{NC}^1$ 的复杂度猜想下），单层可做 $S_5$ 状态跟踪，常数层识别任意正则语言（摘要、Appendix D）。这是表达力主张，不是语言建模分数。
3. **发布的是小模型，不是 14B 的续训。** 四个 World v3 模型从 0.19B 到 2.9B，见到的 token 数为 1.6T 到 5.6T；公开多语言语料组件为 3.1T tokens（摘要与 contributions）。另有三档 Pile 模型（0.17B–1.47B，GPT-NeoX 词表）供架构对照。

## 架构与训练

论文约定向量是行向量，矩阵从右边乘。每个 head 的状态是 $(D/h)\times(D/h)$；Figure 2 的图注写实际大小是 **每 head $64\times 64$**。相对 RWKV-5/6，这里的 $wkv$ 做了转置，为了和代码一致（§4.1）。

递推（Eq. 16–17）：

$$
\mathrm{wkv}_0=0,\qquad
\mathrm{wkv}_t=\mathrm{wkv}_{t-1}\bigl(\mathrm{diag}(w_t)-\hat\kappa_t^\top(a_t\odot\hat\kappa_t)\bigr)+v_t^\top\tilde k_t
$$

$\hat\kappa_t=\kappa_t/\lVert\kappa_t\rVert_2$ 是按 head 的 L2 归一化。作者把 $w_t$ 叫做 in-context weight decay（向量，数据相关），把 $a_t$ 叫做 in-context learning rate（也是向量）。转移矩阵（Eq. 19）

$$
G_t=\mathrm{diag}(w_t)-\hat\kappa_t^\top(a_t\odot\hat\kappa_t)
$$

只是 Householder 的缩放近似：因为 $\hat\kappa_t\neq (a_t/w_t)\odot\hat\kappa_t$，它不是 Schlag / DeltaNet 那种 $I-a\,k^\top k$。附录 C 讨论特征值何时仍在稳定区间。

和相邻规则的差（Table 1，原文记号）：

| 规则 | 状态更新里和 delta 有关的部分 |
| --- | --- |
| [Gated DeltaNet](gated-delta-net.md) | 标量 $w_t$ 乘在 $(I-a_t k_t^\top k_t)$ 上 |
| Generalized delta | $\mathrm{diag}(w_t)+z_t^\top b_t$，移除方向和写入方向可以不同 |
| RWKV-7 | 上式的一个实例：移除键是 $\hat\kappa_t$，写入键是 $\tilde k_t$ |

[Gated DeltaNet](gated-delta-net.md) §5 把并发的 RWKV-7 写成 diagonal-plus-low-rank，脚注当时指向 GitHub 树。本 PDF 就是那条转移的论文形式。它不是 [KDA](kimi-linear.md)：KDA 把 DPLR 的两支低秩向量都绑回 $k$，RWKV-7 允许移除键和替换键不同。

Table 1 同时把 RWKV-4 收成两条 channel-wise 向量扫描 $s_t=e^{-w}\odot s_{t-1}+e^{k_t}\odot v_t$，没有矩阵 $S$。不要用本页的矩阵递回去改写 [RWKV-4 的 Eq. 16](rwkv.md)。

## 后训练

本页主结果是预训练。多模态实验在 §9，不构成语言后训练配方。

## 评测要点

摘要的质量句是：2.9B 在多语言任务上是当时 3B 档的新 SoTA，英文下游与当时 3B SoTA 相当，而且训练 token 少于对照里的顶尖模型。本页不把这句话展开成逐项榜，避免在没逐表核对时改写分数。表达力实验（正则语言、$S_5$）和语言建模分数是两套证据。

## 相关页面

- 前作向量状态：[RWKV](rwkv.md)（RWKV-4）
- 概念：[线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md)
- 标量门 + Householder delta：[Gated DeltaNet](gated-delta-net.md)
- 细门但无 delta：[Gated Linear Attention](gated-linear-attention.md)
- 细门 + 绑回 $k$ 的 DPLR：[Kimi Linear](kimi-linear.md)
