---
type: Concept
title: "Hierarchy-of-Groups Policy Optimization"
description: "HGPO 将同 state 的 step-level relative advantage 按近期历史的一致性拆成嵌套组，再以深度权重聚合，用既有 rollout 交换 context bias 与小组方差。"
tags: ["concept", "hgpo", "agentic-rl", "advantage-estimation"]
timestamp: 2026-09-07
---

# Hierarchy-of-Groups Policy Optimization

## 定义

Hierarchy-of-Groups Policy Optimization（HGPO）是 [HGPO 论文](../sources/hierarchy-of-groups-policy-optimization.md)提出的 critic-free 长周期 agent RL 算法。它接受已经从同一初始状态滚出的 trajectory group，不追加 rollout；对每个 step 先按当前环境 state 匹配，再按最近 $0\ldots K$ 段历史是否一致建立嵌套 group，分别估计 group-relative advantage 并加权。

它要解决的不是“有没有重复 state”，而是更细的问题：**当 step-wise policy 的 prompt 含有限 memory 时，当前 state 一样并不表示条件分布一样。**最深组最接近相同 effective prompt，但样本少；0-context 组样本多，却把不同历史的决策混在一起。HGPO 的折中是保留各层信号，而非只用最深的 Oracle-like group。

## 跨报告信号

- **[GiGPO](group-in-group-policy-optimization.md)**：它把同一状态的碰撞当免费 step-level 对照，用 $A^E+\omega A^S$ 同时保留 trajectory 与 step 信号。HGPO 继承同 state hashing 和零额外 rollout，却移除 trajectory-level advantage，改为多个 history-aware $A_k^H$；HGPO 的 Table 5 显示重新加入 trajectory-level advantage 多数会降分。两者并不是“更细的同一个 $A^S$”：HGPO 的核心批评是 GiGPO 的 anchor 还可能有 memory context 混杂。
- **[ARPO](agentic-reinforced-policy-optimization.md)**：ARPO 在 high-entropy tool-feedback step 额外分叉 partial rollout，以花采样预算换取局部 action 对照；HGPO 固定 rollout budget，在已有样本上换取更匹配的 group baseline。两篇都针对 step credit，却分别改 sampling 和 conditioning，尚无直接比较。
- **[SAO](single-rollout-asynchronous-optimization.md)**：SAO 认为异步长周期任务不该等待组，令 group size=1 后用 critic / GAE 回收 advantage；HGPO 的方差控制仍依赖大 group，因此不能直接作为 SAO 的替代。
- **[DAPO](../sources/dapo.md)、[GSPO](../sources/group-sequence-policy-optimization.md)、[SAPO](../sources/soft-adaptive-policy-optimization.md)**：这些工作主要修改 GRPO recipe、ratio 单元或 trust-region gate。HGPO 保留逐步 clipped ratio，改的是其前面的 advantage conditioning；概念上可组合，论文未验证。

## 为什么重要

对长周期 agent 而言，整条轨迹一个 advantage 太粗，而“same state 就可比”又过强。HGPO 把这个常被隐藏的统计前提变成可操作的轴：**每多匹配一段历史，偏差可能下降，但 group size 与 step utilization 也会下降。**它使 agent RL 的 credit assignment 不再只按 trajectory / token / state 三层划分，也要问“比较对象究竟共享多少决策上下文”。

这也给系统设计一条直接约束：memory 的表现形式决定能否建组。截断但可序列化的 observation-action history 可精确 hash；摘要 memory、检索 memory 或 latent state 则不能直接按文字前缀匹配。不能把 HGPO 在 ALFWorld / WebShop 的结果直接外推成“摘要 memory 一定适合层次分组”。

## 待追问

- **需实验或作者披露**：实际 prompt 若含 task instruction、tool schema、action history 和摘要，$C_k$ 只匹配 state 是否会漏掉关键条件？ 最深的 $G_K^H$ 是否接近 Oracle，还取决于 system prompt、memory 实现和最近 $K$ 个 state 以外的条件。
- **需实验或作者披露**：最优深度与权重是否可从 group size、return variance 或 estimator uncertainty 自适应学习，而非手调 $K,\alpha$？
- **需实验或作者披露**：当 state 只可语义近似匹配时，错误合并引入的 bias 会否大于 history-aware grouping 消除的 bias？
- **需实验或作者披露**：ARPO 的局部 re-rollout、GiGPO 的 state collision 和 HGPO 的 history hierarchy 在同一 token / 环境预算下如何取舍？

## 相关页面

- 来源：[HGPO](../sources/hierarchy-of-groups-policy-optimization.md)、[GiGPO](../sources/gigpo.md)、[Agentic Reinforced Policy Optimization](../sources/agentic-reinforced-policy-optimization.md)、[Single-Rollout Asynchronous Optimization](../sources/single-rollout-asynchronous-optimization.md)
- 概念：[Group-in-Group Policy Optimization](group-in-group-policy-optimization.md)、[Agentic 模型的后训练](post-training-for-agentic-models.md)
- 比较：[LLM RL policy optimization 对比](../comparisons/llm-rl-policy-optimization.md)

关联提问页：[Hierarchy-of-Groups Policy Optimization（HGPO）](../sources/hierarchy-of-groups-policy-optimization.md#相关追问)。
