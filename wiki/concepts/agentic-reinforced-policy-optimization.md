---
type: Concept
title: "Agentic Reinforced Policy Optimization"
description: "ARPO 如何用工具反馈后的 entropy spike 指导 partial rollout 分叉，并做共享/分叉段 advantage attribution。"
tags: ["concept", "agentic-reinforced-policy-optimization"]
timestamp: 2026-06-25
---

# Agentic Reinforced Policy Optimization

## 定义

Agentic Reinforced Policy Optimization（ARPO）是 [Agentic Reinforced Policy Optimization 论文](../sources/agentic-reinforced-policy-optimization.md)提出的 agentic RL 算法。它针对多轮 tool-use agent：模型每次收到工具反馈后，不确定性会在接下来前若干 token 上升；ARPO 因此把 rollout 预算集中到高熵工具调用步，从这些节点分叉 partial rollouts，再用 advantage attribution 处理共享前缀与分叉段的 credit assignment。

一句话：**ARPO 不是换 reward，也不是换 agent harness；它是把 GRPO 类 trajectory-level RL 的「整条轨迹多采样」改成「工具反馈后的高熵节点重点分叉」。**

## 跨报告信号

- **GLM-5 的 [异步 Agent RL](asynchronous-agent-rl.md)**：重点是长尾 agent rollout 的训练系统问题——rollout engine / training engine 解耦、stale sample dropping、token-level clipping、TITO、DP-aware routing。它默认还是在真实环境里大规模收轨迹，核心瓶颈是吞吐和 off-policy 控制。
- **MiniMax-M2 的 [Forge Agent-Native RL](forge-agent-native-rl.md)**：重点是把任意 agent scaffold、工具、上下文管理和 reward 接成可训练环境，LLM completion 是 action，工具与 context state transition 是 environment。它解决「怎么接入和调度 agent 环境」。
- **ARPO**：重点是轨迹内部的采样结构。论文观察到工具反馈后 entropy spike，于是从高熵 tool-call step 分叉 partial rollouts，试图让模型学到 step-level tool-use behavior。
- **[Agentic 评测体系](agentic-evaluation-benchmarks.md)**：ARPO 的 deep search 评测覆盖 GAIA / WebWalkerQA / HLE / XBench；数学与知识推理覆盖 AIME、MATH、HotpotQA、2Wiki、MuSiQue、Bamboogle 等。它强化了一个已有结论：同名 benchmark 分数要连同 agent harness、search/browser 设置和 judge 一起看。

## 为什么重要

ARPO 把 agentic RL 的问题拆得更细：trajectory-level RL 假设「完整轨迹好坏」足够指导模型，但多轮工具任务里，真正关键的决策往往发生在工具返回之后：模型要判断结果是否可信、下一步搜什么、是否改用 Python、是否回溯计划。工具反馈带来的 distribution shift 会让这几个 token 的 entropy 明显上升。

因此 ARPO 的价值不是「证明 entropy 是万能指标」，而是提供了一个可执行的训练假设：**在 agent 环境里，探索预算应该跟随环境反馈后的不确定性移动，而不是平均铺在整条轨迹上。** 这和 OPD 的 dense token-level supervision、Forge 的 episode / reward-to-go、GLM-5 的异步 rollout 系统一起，说明 agentic 后训练正在从单一 RL loss 扩展成「采样结构 + credit assignment + 系统调度 + 环境设计」的组合问题。

## 算法骨架（据原文核实）

1. 初始生成 $N$ 条 global trajectories；总 rollout 预算为 $M$，剩余 $M-N$ 留给 partial sampling。
2. 计算每条轨迹起始前 $k$ 个 token 的 entropy，作为 $H_{initial}$。
3. 每次工具调用返回后，让模型再生成 $k$ 个 token，计算 step-level entropy $H_t$，并得到 $\Delta H_t = Normalize(H_t-H_{initial})$。
4. 用 $P_t = \alpha + \beta\Delta H_t$ 决定是否分叉：若 $P_t > \tau$，从当前节点 branch 出 $Z$ 条 partial rollouts；否则继续。
5. 对分叉后的样本计算 reward 与 group-relative advantage。共享前缀可以显式用 hard shared advantage；论文默认采用 soft advantage estimation，让 GRPO importance ratio 隐式处理共享前缀与分叉段。

这个骨架来自原文 §3.1–§3.2 与 Appendix E Algorithm 1；论文的理论部分把 token 序列切成 macro actions，用 Generalized Policy Gradient Theorem 支撑 partial rollout segment 也能作为 policy-gradient 更新单元。

## 实验信号

- 10 个数学 + 知识密集推理 benchmark：ARPO 在 Qwen2.5-3B、Llama3.1-8B、Qwen2.5-7B 上平均分 52.8 / 55.3 / 58.3，均超过同表 trajectory-level RL baseline。
- Deep search：Qwen3-8B/14B 只用 1K RL samples；ARPO overall avg. 25.0 / 32.0，高于 GRPO 20.0 / 27.0。Qwen3-14B + ARPO 在 Table 2 中 GAIA avg. 43.7、WebWalkerQA avg. 36.0、HLE avg. 10.0、XBench avg. 32.0。
- Tool-call efficiency：Figure 7 显示 Qwen2.5-7B 上 ARPO 用约一半工具调用数取得更高 overall accuracy。
- 超参边界：Figure 8 显示 entropy weight 0.4 左右最佳，升到 1.0 反而掉分；initial sampling size 在 8 达峰；global rollout size 增大仍有收益。

## 待追问

- ARPO 的 entropy spike 目前主要来自 search / browser / Python interpreter；coding agent 的编辑-测试-修复 loop 是否同样适用？
- 论文的 deep search 结果依赖 Bing、browser agent 和 LLM-as-judge；换工具栈后相对 GRPO 的收益是否稳定？
- entropy-based branching 与推理期 tree search / self-consistency / Agent Swarm 的运行时并行探索如何结合？
- soft advantage estimation 是实测更稳，但共享前缀 credit assignment 在更长轨迹中是否会被 group normalization 稀释？

## 相关页面

- 来源：[Agentic Reinforced Policy Optimization](../sources/agentic-reinforced-policy-optimization.md)
- [Agentic 模型的后训练](post-training-for-agentic-models.md)
- [异步 Agent RL](asynchronous-agent-rl.md)
- [Forge Agent-Native RL](forge-agent-native-rl.md)
- [Agentic 评测体系](agentic-evaluation-benchmarks.md)
