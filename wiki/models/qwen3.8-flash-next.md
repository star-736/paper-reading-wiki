---
type: Model
title: "Qwen3.8-Flash-Next"
description: "Qwen 的 125B/6B 激活 MoE，另加 51B 主机 n-gram；3:1 GDN + QSA，Gated Residual，Muon 预训练。架构报告未给 instruct / RL 配方。"
tags: ["model", "qwen3.8-flash-next", "moe", "gdn", "qsa"]
timestamp: 2026-09-05
---

# Qwen3.8-Flash-Next

## 身份

Qwen3.8-Flash-Next 是 Qwen Team 的预训练架构模型，用来证明：在 3:1 [GDN](../concepts/linear-attention-and-delta-rule.md) 混合上换 [QSA](../sources/qwen3.8-next.md) 全局层、加上 Gated Residual 和主机 n-gram，可以用约 1/9 的训练 FLOPs 接近上一代 397B-A17B 旗舰的 base 分数。一手出处是 [Qwen3.8-Next 架构报告](../sources/qwen3.8-next.md)（arXiv:2608.30320）。同报告 Table 11 还列了同代 dense 对照 Qwen3.8-27B-Base，但没有单独展开。

## 关键事实

| 项 | 取值 |
| --- | --- |
| **模态** | 纯文本（已据本报告核实：无 vision/audio 编码器或视觉 token；n-gram 作用在文本 tokenizer 上） |
| 类型 | 稀疏 MoE + 加速器外 n-gram 表 |
| 总参 / 激活 | 125B / 6B；另 51B n-gram embedding（主机内存，不计入加速器常驻） |
| 注意力 | 3 GDN : 1 全局层；CPT 后全局层为 QSA（$K=2048$，$r=4$，indexer 4 query / 1 key） |
| 残差 | Gated Residual，$n_r=4$ 支，elementwise 门读、标量写，无 $H_{res}$；推理可 FP8 存残差 |
| 优化器 | Muon（二维线性映射）+ AdamW（embedding / router / GR 低秩）；无 batch-size warmup |
| 上下文 | CPT 256K；QSA 评到 RULER / MRCR 1M |
| Tokenizer | 报告写 Qwen3.5 词表 $V=250$K（§2.3.2） |
| 后训练 | 本报告未给 SFT/RL 配方或 instruct 分 |

同表对照（原文确证，Table 11）：Qwen3.8-27B-Base 为 27B dense；Qwen3.7-Plus-Base 为 397B / 17B 激活。层数、hidden、expert 数、路由未披露。

## 技术身份

这条线接 [Qwen3-Next](../sources/qwen3-next-blog.md) / [Qwen3.5](qwen3.5.md) 的 3:1 GDN + gated full attention，但做了三件前作博客没写的事：全局层在长上下文 CPT 换成内容稀疏的 QSA；残差从单流换成四支 GR；用 Muon 把最优 LR/batch 上推，并靠门做稳定性，不再依赖 qk-clip。线性层仍停在 GDN（sigmoid 输出门 + Zero-Centered RMSNorm），**没有**升到 Kimi 的 KDA。

和 [Kimi K3](kimi-k3.md) 对照：两边都是 3:1 线性/全局混合，都动深度方向的残差（K3 用 AttnRes，这里用 GR），但全局层 K3 是 Gated MLA，这里是 QSA；K3 是 2.78T/104B 的开源旗舰，这里是 6B 激活的效率点。QSA 相对 [DSA](../concepts/deepseek-sparse-attention.md) 的差异是 indexer 先压 micro-block，显式针对 hybrid 里跨层共享变弱。

51B 主机 n-gram 属于 [条件记忆](../concepts/conditional-memory.md) 这条稀疏轴，与 [Engram](../sources/engram.md) 的 hashed $N$-gram 是平行实例：都靠确定性查找把容量放到加速器外。Qwen 选择把表叠在固定 MoE 预算之上，而不是按 Engram 的 U 形律从 experts 里重分配。

## 相关页面

- 来源：[Qwen3.8-Next 架构报告](../sources/qwen3.8-next.md)
- 家族：[Qwen3](qwen3.md)、[Qwen3.5](qwen3.5.md)、[Qwen3-Coder-Next](qwen3-coder-next.md)
- 概念：[线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md)、[DeepSeek Sparse Attention](../concepts/deepseek-sparse-attention.md)、[Attention Residuals](../concepts/attention-residuals.md)、[MoE 前沿模型扩展](../concepts/moe-frontier-model-scaling.md)、[条件记忆](../concepts/conditional-memory.md)
- 比较：[稀疏注意力机制对比](../comparisons/sparse-attention-mechanisms.md)、[2026 前沿模型技术报告对比](../comparisons/2026-open-model-technical-reports.md)
