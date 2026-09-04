---
type: Concept
title: "Single-Rollout Asynchronous Optimization"
description: "SAO 如何用单条 rollout 替代 GRPO 组采样，并用 DIS mask 与加速 critic 在异步 agentic RL 里同时处理 off-policy 与方差。"
tags: ["concept", "sao", "llm-rl-policy-optimization", "agentic-rl", "asynchronous-rl"]
timestamp: 2026-09-05
---

# Single-Rollout Asynchronous Optimization

## 定义

Single-Rollout Asynchronous Optimization（SAO）是 [SAO 论文](../sources/single-rollout-asynchronous-optimization.md)提出的 agentic RL 算法。它针对异步训练：轨迹一完成就进梯度，不再等同一个 prompt 的一组 rollout。没有组内相对奖励之后，advantage 改由 value model + GAE 提供；off-policy 则用 Direct Double-Sided Importance Sampling，把 $\pi_\theta/\pi_{\mathrm{rollout}}$ 出界的 token 从梯度里 mask 掉。

一句话：**SAO 不是换 reward，也不是多采 step；它是把 GRPO 的组屏障拆掉，用 critic 换回单条轨迹可用的 baseline，并用比 PPO 更狠的 token mask 扛异步 lag。**

## 跨报告信号

- **[异步 Agent RL](asynchronous-agent-rl.md)**：GLM-5 报告已经把 DIS、TITO、stale dropping 写成系统机制。SAO 是同一方向上的算法论文，补上「为什么组采样在异步下是错配」和「单 rollout 时 critic 怎么训」。K3 / Ring-2.6 / Laguna 仍多在系统调度层，不一定放弃 group。
- **[VAPO](../sources/vapo.md)**：Length-Adaptive GAE 被 SAO 直接采用。无 DIS 的 vanilla VAPO 在 SAO 的异步设置下约 90 step 崩，说明 critic 配方本身不够。
- **[ARPO](agentic-reinforced-policy-optimization.md) / [GiGPO](group-in-group-policy-optimization.md)**：两者默认还在 group 里工作——ARPO 在高熵工具步额外分叉，GiGPO 用重复状态做 step 对照。SAO 问的是要不要组。目前没有同表。
- **IcePop / [KPop](../sources/ling-2.6.md)**：DIS 的固定 ratio 区间 mask 接近 IcePop，并丢掉 $\pi_{\theta_{\mathrm{old}}}$。KPop 后改 binary KL。同一层问题的不同边界形状。
- **[GLM-5.3](../models/glm-5-3.md)**：发布博客称延续 GLM-5.2 的 SAO with compaction。compaction 不在算法论文里；生产声明与 30B 实验不是同一份证据。

## 为什么重要

2024–2025 的 LLM RL 主潮是砍 critic：GRPO / DAPO / GSPO / CISPO 用组内相对奖励换掉 value model。异步长周期 agent 把这个交换的前提拆掉——组必须等最慢样本，而真实环境常常只给一条轨迹。

SAO 的可执行假设是：**异步的效率来自「完成即训」，所以采样单元必须是单条轨迹；单条轨迹的方差只能靠 critic，异步的 off-policy 只能靠比 PPO 更严的 token 过滤。** Figure 3 把这两件事拆开：DIS 让 GRPO 也能跑满 1000 step，单 rollout + critic 才在约 400 step 后继续拉开。

## 算法骨架（据原文核实）

1. 每个 prompt 采 1 条 rollout，完成后立刻进入训练 batch（group size = 1）。
2. Importance ratio 用 rollout 当时记下的 log-prob：$r_t=\pi_\theta/\pi_{\mathrm{rollout}}$，不再维护 $\pi_{\theta_{\mathrm{old}}}$。
3. $r_t\notin(1-\epsilon_\ell,1+\epsilon_h)$ 的 token 对梯度贡献为 0（DIS）。
4. Advantage 来自 critic + length-adaptive GAE；多轮轨迹用 Skip-Observation，跳过环境 observation token。
5. 每个 policy step 更新 critic $K=2$ 次；冻结 critic 的 attention，只训 MoE 投影。

这个骨架来自原文 §3.1–§3.2 与 Eq. 1–5。它改的是 **采样单元和 baseline 来源**，不是 sequence-level ratio，也不是 step-level grouping。

## 实验信号

- Qwen3-30B-A3B，TIR 数学：SAO 相对 GRPO（崩前）AIME 2025 84.2→97.3、BeyondAIME 54.8→74.8；相对已加 DIS 的 GRPO 仍有约 3–4 个百分点（Table 1）。
- SWE-Bench Verified / OpenHands：23.0 → GRPO+DIS 27.0 → SAO 29.8（Table 2）。
- 稳定性：vanilla GRPO ~160 step 崩，vanilla VAPO 无 DIS ~90 step 崩，SAO 报告约 1000 step（Figure 3–4）。
- 作者写已部署到 GLM-5.2（750B-A40B）agentic RL pipeline；这是原文声明，不是本页 30B 表能外推的结果。

## 待追问

- Frozen-attention critic 在 dense 模型上是否还成立？
- 与 ARPO / GiGPO 同一异步预算下，放弃 group 是否总是值得？
- GLM-5.3 的 compaction 具体做什么，算法论文没有说。

## 相关页面

- 来源：[Single-Rollout Asynchronous Optimization](../sources/single-rollout-asynchronous-optimization.md)
- [异步 Agent RL](asynchronous-agent-rl.md)
- [Agentic Reinforced Policy Optimization](agentic-reinforced-policy-optimization.md)
- [Group-in-Group Policy Optimization](group-in-group-policy-optimization.md)
- [Agentic 模型的后训练](post-training-for-agentic-models.md)
- [LLM RL policy optimization 对比](../comparisons/llm-rl-policy-optimization.md)
- [GLM-5.3](../models/glm-5-3.md)
