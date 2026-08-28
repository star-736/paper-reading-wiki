---
type: Concept
title: "零样本 RoPE 上下文扩展"
description: "不微调、只在推理时改位置映射，让 RoPE 模型用过训练窗：频率缩放（PI / NTK / YaRN）对分组别名（Self-Extend / DCA / Jet-Long），以及 NoPE 这条避开问题的旁路。"
tags: ["concept", "zero-shot-rope-context-extension", "rope", "yarn", "context-extension"]
timestamp: 2026-08-29
---

# 零样本 RoPE 上下文扩展

## 定义

多数开源 LLM 用 Rotary Position Embedding（RoPE）：把 query / key 按几何间隔频率旋转，注意力只依赖相对位置。预训练窗通常是 4K–32K；仓库级代码、RAG 和 agentic 轨迹经常到 100K+。**零样本上下文扩展**指：不继续预训练、不改权重，只在推理时改位置映射或 RoPE 频率，让同一份 checkpoint 吃更长输入。

RoPE 越出训练窗有两个独立失败模式（[Jet-Long](../sources/jet-long.md) §2.1）：

1. **位置 OOD**：低频分量的旋转角超出训练分布，注意力分数失稳。
2. **Softmax 扩散 + 中间偏差**：key 集合变大后概率摊平，再叠加 U 形位置偏差，中间证据召不回来。

零样本方法几乎都在打第一条。第二条更常被稀疏注意力、线性 / SSM、或温度修正（YaRN 的 attention temperature）顺带处理，不是位置映射的主战场。这与 [高效长上下文注意力](efficient-long-context-attention.md) 是**正交轴**：那边管「算哪些 token、KV 多大」；这边管「位置角还在不在训练网格上」。一份 32K 训好的权重，即使换成 DSA / SWA，只要 RoPE 角 OOD，长窗照样崩。

三条技术族：

| 族 | 代表 | 做法 | 短窗 / 长窗权衡 |
| --- | --- | --- | --- |
| 频率缩放 | PI、NTK、DNTK、YaRN | 把位置或 RoPE base 压进训练范围；YaRN 再按维切分频率 + 温度修正 | 预先固定 factor：大则伤短窗，小则长窗不够 |
| 分组 / 分块位置 | Self-Extend、DCA、[Jet-Long](../sources/jet-long.md) | 局部窗保留原版 RoPE；远处让一块 token 共享一个训练期内的位置索引 | 固定 group / chunk 同样有短/长冲突；Jet-Long 用当前 $L$ 解析出最小整数 $G$ |
| 绕开 RoPE | NoPE（[Kimi Linear](../sources/kimi-linear.md) / [Kimi K3](../sources/kimi-k3.md) 的全局 MLA 层） | 全局层不加显式位置，位置 / recency 交给线性层的 decay | 通常要在混合架构里从头或 retrofit 训练，不是纯推理补丁 |

长度自适应变体（AdaGroPE、LaMPE、SELF）也按输入长度改映射，但压缩比来自拟合 sigmoid 或 logistic，需要 per-model 日程。Jet-Long 的 $G=\max(1,\lceil L/w_{\text{pretrained}}\rceil)$ 没有拟合参数，并且 $L\le w_{\text{pretrained}}$ 时严格退回基座。

## 跨报告信号

- **[Qwen3](../sources/qwen3.md)**：S3 把序列从 4K 拉到 32K，RoPE 基频 10k→1e6（ABF）。公开推理配方是 **YaRN + Dual Chunk Attention，再 4×**，把有效长度扩到约 128K（1.7B 模型卡仍标 32K；4B/8B 标 128K，来源就是这套推理扩展，不是再训到 128K）。[Jet-Long](../sources/jet-long.md) 把同一 YaRN factor=4 当作基线：在 Qwen3-1.7B/4B/8B-Base 上 RULER 全面落后动态分组，1.7B 上 YaRN 甚至低于不扩展的 Base（52.99 vs 60.93）。读 Qwen3 的「128K 上下文」时，应把它当成 YaRN+DCA 的部署结果，而不是原生训练窗。
- **[Qwen3-Next](../sources/qwen3-next-blog.md)**：原生 262,144，再用 YaRN 外推到 1M。这里 YaRN 作用在已经 256K 训过的 hybrid 栈上，和 Qwen3 从 32K 零样本跳 4× 不是同一难度。
- **[Laguna XS.2](../sources/laguna-m1-xs2.md)**：4K 预训练 → 32K → 128K（**YaRN 只加在 GA 层**）→ 256K（GA 层 RoPE scale 翻倍、无训练）。这是「中段继续训练 + 末端零样本缩放」，不是纯 tuning-free。256K 的真实长程任务分没有单独报。
- **Kimi 系 NoPE**：[Kimi Linear](../sources/kimi-linear.md) 和 [Kimi K3](../sources/kimi-k3.md) 的全局 MLA 层不加 RoPE，位置编码全部交给 KDA 的 decay。好处是扩到 1M **不用 retune RoPE base、不用 YaRN**。这是架构级规避，不是给现成 RoPE checkpoint 的推理补丁。
- **Gemma 4 / Laguna 的 partial RoPE**：全局层 p-RoPE 或只对 50% head dim 加 RoPE，属于训练期位置编码设计，用来降 KV 或稳住 SWA/GA 混合，不是零样本扩展方法。
- **Jet-Long 迁到 hybrid**：同一套双焦点在 Jet-Nemotron（softmax 与线性层交错）上不重训就能用——它只改仍带 RoPE 的 softmax 层。说明这条轴可以叠在 [线性注意力](linear-attention-and-delta-rule.md) 混合栈上，但 128K 绝对值仍低（~33 RULER），hybrid 基座自己的长窗能力是另一回事。

## 为什么重要

开源权重很少会为每个下游窗宽再发一个长上下文 checkpoint。YaRN 已经是 Qwen3 / Qwen3-Next / Laguna 的默认外推旋钮；Jet-Long 给出一个**同协议下打赢 YaRN 和固定分组**的替代，并且用 cache 不变量解决了「生成过程中 $G$ 会变」这个动态方法的老问题。若只看稀疏 / 线性注意力，会漏掉「位置角是否还在分布内」这一层——Laguna 用纯 RoPE scale 从 128K 推到 256K、Kimi 用 NoPE 躲开 YaRN，都是在这一层做选择。

部署含义：未融合的三次注意力 + 逐 token 旋转会把 decode 打到 FA2 的 0.14×；融进单一 kernel 之后才 ≤4% 开销。方法论文的「tuning-free」要连 kernel 一起看。

## 待追问

- Jet-Long 与 DSA / MLA decoupled RoPE / SWA 全局层能否直接叠加，目前没有实验。MLA 的 RoPE 只作用在 decoupled 分支上，correction rotation 是否仍可加，需要核 [DeepSeek-V2](../sources/deepseek-v2.md) 的 Decoupled RoPE 再设计。
- 官方「训到 128K/256K 再 YaRN」与「32K 权重 + Jet-Long」谁在真实仓库级 / agent 轨迹上更好，知识库里还没有同模型对照。
- Softmax 扩散这一条失败模式，零样本位置映射覆盖不了。长窗质量的上限可能仍在注意力架构，而不在 $G$。
- 128K 以上频率插值开始在个别 RULER 任务反超位置别名（Jet-Long Table 6）。1M 级该用纯别名、纯 YaRN，还是 hybrid 映射，开放。

## 相关页面

- 来源：[Jet-Long](../sources/jet-long.md)（本轴的方法论文）、[Qwen3 技术报告](../sources/qwen3.md)、[Qwen3-Next 官方博客](../sources/qwen3-next-blog.md)、[Laguna M.1/XS.2](../sources/laguna-m1-xs2.md)、[Kimi Linear](../sources/kimi-linear.md)、[Kimi K3](../sources/kimi-k3.md)
- 正交的计算轴：[高效长上下文注意力](efficient-long-context-attention.md)、[线性注意力与 delta rule](linear-attention-and-delta-rule.md)、[Multi-Head Latent Attention](multi-head-latent-attention.md)
- 模型：[Qwen3](../models/qwen3.md)、[Laguna](../models/laguna.md)、[Kimi Linear](../models/kimi-linear.md)、[Kimi K3](../models/kimi-k3.md)
