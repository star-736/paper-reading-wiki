---
type: Concept
title: "异步 Agent RL"
description: "GLM-5 如何用异步 rollout、TITO 和 token-level clipping 训练 agent。"
tags: ["concept", "asynchronous-agent-rl"]
timestamp: 2026-06-06
---

# 异步 Agent RL

## 一句话定义

异步 Agent RL 是 GLM-5 用来训练长周期 agent 任务的强化学习框架：rollout engine 持续生成轨迹，training engine 异步消费轨迹并更新模型，从而减少同步 RL 中等待长尾样本造成的 GPU 空转。

## 为什么同步 RL 不够

在普通数学或单轮问答 RL 中，样本长度差异相对可控。但 agent 任务有长尾：

- coding agent 可能需要多次编辑、运行测试、修 bug；
- search agent 可能需要多跳搜索和证据验证；
- terminal agent 可能遇到环境安装、命令失败、长输出；
- 工具调用历史会带来大量 prefill。

如果所有 rollout 必须同步结束再训练，最慢样本会决定整个 step 的 wall-clock 时间。

## GLM-5 的设计

[GLM-5](../models/glm-5.md) 将 inference engine 和 training engine 放在不同 GPU 设备上。Inference engine 不断生成 agent trajectories；当轨迹数量达到阈值，就送到 training engine 更新模型。模型权重周期性从 training engine 同步回 rollout engine，以降低 policy lag。

为了支持多任务，GLM-5 使用 server-based Multi-Task Rollout Orchestrator。每类任务把 rollout 和 reward 逻辑实现成独立 microservice，统一注册到 orchestrator。Orchestrator 控制各任务 rollout 比例、生成速度和日志，并把不同任务轨迹标准化成 message-list representation。

## 稳定性机制

异步带来 off-policy 风险：不同轨迹可能来自不同模型版本。GLM-5 使用几类机制控制它：

- TITO：Token-in-Token-out 直接记录 rollout engine 产生的 token IDs 和 metadata，避免 text-in-text-out 重新分词导致动作、奖励、loss 错位。
- Direct double-sided importance sampling：用 rollout log-prob 作为 behavior proxy，计算 token-level ratio，并把超出 `[1-epsilon_l, 1+epsilon_h]` 的 token 从梯度中 mask 掉。
- Stale sample dropping：记录每条轨迹涉及的模型版本，丢弃太旧的样本。
- Environment failure filtering：如果 sandbox 崩溃等环境原因导致失败，不把它当作模型能力失败。
- DP-aware routing：同一个 rollout 的多轮请求固定路由到同一个 DP rank，复用 KV cache，避免重复 prefill。

## 训练环境扩展

GLM-5 报告中构造了超过 10K 个 verifiable SWE environments，覆盖 Python、Java、Go、C/C++、JavaScript、TypeScript、PHP、Ruby 等语言。它还构造 terminal-agent environments 和 multi-hop search tasks，使 agentic RL 不只依赖静态 benchmark。

## 关键判断

异步 Agent RL 的本质不是“换一个 RL 算法”，而是把 RL 训练变成一个分布式系统问题：调度、容错、token 对齐、KV-cache locality、环境质量都会影响最终模型能力。

[Agentic Reinforced Policy Optimization](agentic-reinforced-policy-optimization.md) 则是互补的算法层问题：在一个 agent 轨迹内部，工具反馈后的高熵 step 是否应该追加 partial rollout 探索。前者解决长尾 rollout 怎么跑得动，后者解决有限 rollout 预算投到轨迹哪里。

