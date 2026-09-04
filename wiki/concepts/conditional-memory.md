---
type: Concept
title: "条件记忆"
description: "与 MoE 条件计算互补的第二条稀疏轴：用 O(1) 查找取静态局部模式，而不是用深度去重建。DeepSeek Engram 与 Qwen3.8-Next 主机 n-gram 是两个实例。"
tags: ["concept", "conditional-memory", "engram", "n-gram", "moe", "sparsity"]
timestamp: 2026-09-05
---

# 条件记忆

## 定义

**条件记忆（conditional memory）** 是 [Engram](../sources/engram.md) 给的名字：在 MoE 那种「按 token 稀疏激活计算」之外，再加一条「按局部上下文稀疏查找静态向量」的轴。语言建模被拆成两件不同的事——组合推理要动态计算，命名实体和套话则是局部、静态、高度套路的，经典 $N$-gram 本来就能 $O(1)$ 查到。标准 Transformer 没有 lookup 原语，只能用前几层 Attention + FFN 把静态表当场重建。

和相邻概念的边界：

| | 条件记忆 | [MoE](moe-frontier-model-scaling.md) | [Intern-S2-Mobius](../sources/intern-s2-mobius.md) | RAG / RETRO 类 |
| --- | --- | --- | --- | --- |
| 稀疏对象 | embedding 槽 | expert 参数 | 共享 FFN 知识块 | 外部文档 |
| 寻址 | 确定性 hash（input ID） | 动态 hidden 路由 | Reasoner 检索（计算） | 检索器 |
| 每 token 成本 | 常数槽数，与表大小无关 | top-$k$ 专家 | 仍是神经网络前向 | 检索 + 注意力 |
| 主机 offload | index 可预取 | 路由依赖 hidden，难预取 | 未作为一等设计 | 本来就在模型外 |

**推断 / 本页综合**：条件记忆是 *parametric、in-graph、deterministic lookup*；Mobius 仍用计算读共享 FFN；RAG 把知识放到可编辑的模型外存储。三条都能说「别把事实塞进深度」，接口完全不同。

## 跨报告信号

当前 wiki 里有两个可核实例，外加一条相邻但不同的拆分。

### Engram（DeepSeek，方法论文）

[Engram](../sources/engram.md) 把 hashed $N$-gram 做成一等模块：tokenizer compression、多头哈希、hidden-state 门控、短卷积、mHC 分支门，插在选定层（27B 是 2 和 15）而不是 Layer 0。核心定量结果：

- **U 形分配律**：固定总参和激活，把 20–25% 的 inactive 预算从 routed experts 改给 Engram，验证损失低于纯 MoE；$\rho\approx 40\%$ 仍追平纯 MoE（§3.1 / Figure 3）。
- **iso-param 27B**：72→55 experts 换 5.7B 表，同 3.8B 激活 / 262B tokens，BBH +5.0、CMMLU +4.0、HumanEval +3.0（Table 1）。
- **确定性预取**：100B 表全在 host 上，4B/8B dense 吞吐掉 1.9% / 2.8%（Table 4）。

作者的机制叙事：Engram 把早期静态重建卸掉，CKA 上浅层对齐 MoE 更深的层；关掉表之后事实知识崩、阅读理解几乎不动。这是相关证据，不是加层对照。

### Qwen3.8-Next 的主机 n-gram（生产架构）

[Qwen3.8-Flash-Next](../models/qwen3.8-flash-next.md) 在 Layer 2 插多头哈希 n-gram，contextual gating 注入残差，51B 表放主机、几乎不加每 token FLOPs（[架构报告](../sources/qwen3.8-next.md) §2.3）。和 Engram 的系统直觉同构：Layer 2 是为了让第一层计算掩盖预取。

关键实验分叉（原文确证，Qwen Table 8 vs Engram Figure 3）：

- Qwen **固定总参、用 expert 数换 n-gram 词表**：loss 在约 25% 参数（10× 词表）最低，**下游没有清楚超过纯 MoE**。作者据此判定 n-gram 与 MoE 不是同一类容量，生产配置改成 **MoE 预算不动、只加表**。
- Engram **同样 iso-param / iso-FLOPs 重分配**：报下游全面领先纯 MoE。

两边都承认「n-gram 不是更小的 MoE」。分歧是：**重分配该不该做**。Qwen 的回答是「别从 expert 里抠，往上叠」；Engram 的回答是「抠 20–25% 值得」。模块件（压缩 / 门控 / 双层 vs 单层）、backbone（MLA+mHC vs GDN+QSA）和评测都不同，不能把一边的表当成对另一边的证伪。

Qwen 来源页未引用 Engram（本页检索 `wiki/sources/qwen3.8-next.md` 无此名）；时间线上 Engram v1 在 2026-01、Qwen3.8-Next 在 2026-08。把两者写成「Qwen 实现了 Engram」是推断，只宜当平行实例。

### 不是 Mobius 的 FFN Memory

[Intern-S2-Mobius](../sources/intern-s2-mobius.md) 把各层 FFN 横拼成全局共享 knowledge-vector Memory，Reasoner 反复读。那是 **把计算模块改成全层可寻址的参数库**，不是 $O(1)$ hash 表，也不能确定性预取。和 [Attention Residuals](attention-residuals.md) 一样，Mobius 动的是深度方向的信息接口；条件记忆动的是「静态模式还要不要用深度去算」。

## 为什么重要

- **稀疏容量的会计要从「总参 / 激活」扩到第三个数。** [MoE 前沿模型扩展](moe-frontier-model-scaling.md) 已经为 Qwen 的 51B 主机表留了位置；Engram 给这个第三个数一条可优化的分配律，而不只是「再加一张表」。
- **系统约束反过来选层位。** 建模想要早插入（卸掉底层重建），预取想要够深的计算窗口。Layer 2 是两边都收敛到的工程点；Engram 再用第二处（15 或消融里的 6）补后期 gating。
- **长上下文收益可以来自「别占用注意力」而不是改注意力。** Engram 不稀疏 attention，但 Multi-Query NIAH 从 84.2 到 97.0（32K RULER）。这和 DSA / QSA / 线性混合是正交轴，见 [高效长上下文注意力](efficient-long-context-attention.md)。
- **主机内存上的参数有两类。** 条件记忆是 *可预取的 embedding 行*；[端侧 MoE serving](edge-native-moe-serving.md) 是 *路由之后才知道的 expert 块*。前者通信体积随 active 槽而非常驻表大小，后者必须跟动态路由赛跑。

## 待追问

- **重分配 vs 外加表**：同一 iso-param 问题，Engram 与 Qwen Table 8 结论相反。需要在同一 backbone、同一模块件上复现，才能判断是 Engram 的 tokenizer compression / 门控 / 双层插入把 U 形托起来，还是评测噪声。
- **生产 DeepSeek 模型有没有 Engram。** 本 wiki 的 [V4](../sources/deepseek-v4.md) 报告未写 hashed $N$-gram；Engram-27B 只是研究配置。
- **RL / tool 交错下预取还成不成。** index 依赖完整 input ID；生成中途插入工具观察后，后续 $N$-gram 的预取窗口如何切，两边报告都没测。
- **事实知识是否真的「存在表里」。** Figure 6 的 post-hoc 消融有 train–test mismatch；没有定位到具体槽→具体事实的编辑实验（对比 ROME/MEMIT 那条 FFN 知识文献）。

## 相关页面

- 方法出处：[Engram](../sources/engram.md)
- 生产实例：[Qwen3.8-Next 架构报告](../sources/qwen3.8-next.md)、[Qwen3.8-Flash-Next](../models/qwen3.8-flash-next.md)
- 相邻拆分：[Intern-S2-Mobius](../sources/intern-s2-mobius.md)
- 稀疏会计：[MoE 前沿模型扩展](moe-frontier-model-scaling.md)、[端侧 MoE serving](edge-native-moe-serving.md)
- 正交的长上下文：[高效长上下文注意力](efficient-long-context-attention.md)
- 门控家族（不同对象）：[注意力门控](attention-gating.md)（SDPA 输出门，不是 memory fusion 门）
