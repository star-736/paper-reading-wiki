---
type: Concept
title: "跨层索引复用"
description: "IndexCache、YOIO/CLSA、Kascade、HySparse 等如何让多数层共用一次 top-k；YOIO 把路由绑到 YOCO 的共享 KV 上，只算一次。"
tags: ["concept", "cross-layer-index-reuse"]
timestamp: 2026-09-12
---

# 跨层索引复用

## 一句话定义

跨层索引复用（cross-layer index reuse）指的是：在多层稀疏注意力中，让多数层不再独立计算"该看哪些 token"，而是共用一次 top-k。常见形态是少数 anchor 层算出索引、其余层继承；[YOIO](../sources/yoio.md) 则把 indexer 直接绑到已共享的那一份 KV 上，结构上只有一次路由。这一类方法把 token 选择视为可跨层共享的中间结果，而不是每层都要重做的工作。

## 为什么可行

观察根源是 transformer 相邻层"重要 token 集合"高度稳定：相邻层 top-k 重叠率往往 70-100%，且在层方向上呈现成簇结构（多个连续层共享几乎相同的注意力支撑集）。Deshmukh 等（Kascade，2025）和 Gao 等（HySparse，2026）系统化地报告了这一现象，并提出"少数 anchor 层 + 多数共享层"的范式。

## 三种 oracle 形态

| oracle 类型 | 谁充当 anchor | 代表工作 |
| --- | --- | --- |
| **Full-attention oracle** | 少数层保留 full O(L²) attention，其余层稀疏地复用 anchor 的 top-k | TidalDecode、LessIsMore、OmniKV、DELTA、Kascade、HySparse |
| **Sparse-indexer oracle** | 少数层保留稀疏 indexer 自身（如 DSA lightning indexer），其余层复用 indexer 输出的 top-k | [IndexCache](../sources/indexcache.md)（在逐层 KV 的全稀疏栈上首次系统化） |
| **Shared-memory indexer** | 记忆本身已经跨层共享，routing 绑到这一份记忆上，结构上只有一个 indexer | [YOIO / CLSA](../sources/yoio.md) |

前两条的差别：Full-attention oracle 仍依赖稠密层提供"真值" top-k，DSA / MSA 这种全稀疏栈套不上；Sparse-indexer oracle 用更便宜的 indexer 当 oracle，IndexCache 把范式扩到「完全没有 full attention 层」。第三条不是 retention 比例更激进，而是前提变了：YOCO 已经只有一份全局 KV，CLSA 认为「读同一份记忆的层也应共用同一份路由」，因此 16 层 cross-decoder 摊销一次 token-level top-k，而不是在逐层 KV 上保留 1/4 indexer。YOIO 引用了 IndexCache / HySparse，时间上也更晚（2026-06 vs IndexCache 2026-03）。

## 怎么决定哪些层是 anchor

- **均匀间隔**（FSSSF…）：实现最简单，但 IndexCache 的实验显示它在激进 retention（≤1/4）下会掉很多长上下文点，因为模型里某些早期/过渡层对 indexer 很敏感。
- **贪心校准搜索**：从全 anchor 起步，每步把"翻成共享层后 LM loss 上升最小"的那一层翻掉，直到达到目标 retention。可借助 pipeline parallelism 做近 P 倍加速。
- **动态规划**（Kascade）：在跨层相似度矩阵上搜全局最优 anchor 集；与贪心相比保证全局最优但实现更重。
- **训练后让模型适应**：IndexCache 的 training-aware 模式表明，只要给保留下来的 indexer 加一个"对所有被服务层注意力分布质心"做 KL 蒸馏的目标（多层 KL 与对均值分布的单 KL 在梯度上等价），均匀间隔就能追平贪心搜索——pattern 敏感性是训练时机问题，不是结构性问题。

## 与其他跨层共享的关系

跨层复用其实有三层独立的轴：

- **索引复用**（本页）：top-k 选择跨层共享，KV 仍然各层独立。
- **KV cache 共享**（cross-layer KV sharing，如 CLA、MiniCache）：键值张量本身跨层共享，索引各层独立。
- **二者合一**：HySparse 同时复用 top-k block 索引和 KV cache。

IndexCache 的作者明确说明本工作只动第一条；如果叠加 KV 共享，需要重新评估 S 层经过 sharing-aware 训练后是否还和 KV 共享假设兼容。[YOIO](../sources/yoio.md) 从另一头做：先有 KV 共享，再把 index 绑上去，两条轴变成同一份对象。它没有回答「在仍是逐层 KV 的 DSA 上能否把 retention 压到 1/16」。

## 跨层共享不是压 indexer 的唯一办法

[Qwen3.8-Flash-Next](../sources/qwen3.8-next.md) 的 QSA 把 IndexShare（training-aware 跨层复用）当成对照，而不是采用它。设定是 3:1 GDN hybrid：两个全局层之间隔着三层 GDN。Fig. 5a 上 QSA 用 $r=4$ 层内压缩在相对 indexer 延迟 0.25 追平 full attention；IndexShare 即使只在相邻全局层间共享、相对延迟 0.5，仍低于基线。作者的解释是 hybrid 对跨层相似度的依赖更弱，层内压缩更合适（原文确证，§2.1.2）。这没有否定 IndexCache 在纯 DSA / 全稀疏栈上的结果，只说明 **oracle 层被线性层隔开时，跨层复用的前提会变差**。

## 相关页面

- [DeepSeek Sparse Attention](deepseek-sparse-attention.md)
- [高效长上下文注意力](efficient-long-context-attention.md)
- [百万 token 上下文服务](million-token-context-serving.md)
- 来源：[IndexCache](../sources/indexcache.md)、[YOIO](../sources/yoio.md)、[YOCO](../sources/yoco.md)、[Qwen3.8-Next](../sources/qwen3.8-next.md)
