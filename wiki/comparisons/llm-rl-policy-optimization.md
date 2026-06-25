---
type: Comparison
title: "LLM RL policy optimization 对比"
description: "DAPO / GSPO / SAPO / ARPO 四类 LLM RL policy optimization 方法的抽象层级对比：GRPO recipe、sequence-level ratio、soft trust region、agentic partial rollout。"
tags: ["comparison", "llm-rl-policy-optimization", "rl"]
timestamp: 2026-06-25
---

# LLM RL policy optimization 对比

## 为什么开这一页

DAPO、GSPO、SAPO、ARPO 都在名字上是 policy optimization，但它们解决的不是同一个问题：

- DAPO 补的是 **GRPO 在 long-CoT 数学 RL 大规模复现时的 recipe 缺口**；
- GSPO 改的是 **importance ratio / clipping 的优化单元**，从 token-level 改到 sequence-level；
- SAPO 改的是 **trust region 的形状**，从 hard clipping 改成 temperature-controlled soft gate；
- ARPO 改的是 **agentic rollout 的采样位置**，从完整轨迹采样改到工具反馈后的高熵 step 分叉。

如果把它们都简写成「比 GRPO 更好」，检索时会混掉层级。本页按抽象层级拆开。

## 一张表

| 方法 | 论文 / 团队 | 改动层级 | 主要机制 | 解决的痛点 | 主要实验对象 | 和 GRPO 的关系 |
| --- | --- | --- | --- | --- | --- | --- |
| [DAPO](../sources/dapo.md) | ByteDance Seed + 清华 AIR 等，2025-03 | **GRPO recipe / 系统工程** | Clip-Higher、Dynamic Sampling、token-level loss、Overlong Reward Shaping、DAPO-Math-17K | entropy collapse、全对/全错 prompt 零梯度、长 CoT token 权重稀释、截断 reward noise | Qwen2.5-32B，AIME 2024 | 保留 GRPO group-relative clipped objective，补齐 recipe |
| [GSPO](../sources/group-sequence-policy-optimization.md) | Qwen Team，2025-07 | **importance ratio / clipping 单元** | sequence likelihood ratio $s_i=(\pi_\theta(y_i)/\pi_{old}(y_i))^{1/|y_i|}$，response-level clipping | token-level ratio 与 sequence-level reward 不匹配；MoE expert routing 波动使 token ratio 失效 | Qwen3-30B-A3B，AIME / LiveCodeBench / CodeForces | 用 sequence-level ratio 替代 GRPO token ratio |
| [SAPO](../sources/soft-adaptive-policy-optimization.md) | Qwen Team，2025-12 | **soft trust region / gate 形状** | sigmoid soft gate + $sech^2$ gradient weight；$\tau_{neg}>\tau_{pos}$ | hard clipping 过脆；GSPO 整条 sequence 被裁、GRPO token 越界即零梯度 | Qwen3-30B-A3B、Qwen3-VL-30B-A3B | 保留 group-based RL，替换硬裁剪为软门控 |
| [ARPO](../sources/agentic-reinforced-policy-optimization.md) | 人大 + 快手，2025-07 | **agentic rollout 采样结构** | 工具反馈后监控 token entropy，在高熵 tool-call step 分叉 partial rollouts；advantage attribution | trajectory-level RL 忽略工具反馈后的 step-level 决策 | Qwen2.5 / Llama3.1 / Qwen3，math/QA/deep search | 把 DAPO/GRPO/REINFORCE++ 当 trajectory-level baseline |

## 关键分叉：token、sequence、step 三个「单位」

### DAPO：还是 token-level GRPO，但把 recipe 做对

DAPO 的 objective 仍然是 token-level ratio + clipped objective + group reward normalization。它的贡献是把大规模 long-CoT RL 里几个会让 GRPO 跑不起来的工程细节补上：

- Clip-Higher 放宽上界，防止低概率探索 token 被上界压死；
- Dynamic Sampling 保证 batch 中每个 prompt 既有对也有错，从而有非零 advantage；
- token-level loss 让长 CoT 的 token 不被 sample-level 平均稀释；
- overlong shaping 避免把截断样本粗暴当错，降低 reward noise。

所以 DAPO 是「可复现的 GRPO 系统 recipe」，而不是从理论上否定 token-level ratio。

### GSPO：否定 token-level ratio，转到 sequence-level

GSPO 的判断更根本：既然 reward 是整条 response 的 reward，那么 importance sampling / clipping 也应该按整条 response 来。它认为 GRPO 的 token-level ratio 在长序列里会累积高方差，尤其 MoE 中 expert routing 波动会让 token ratio 失真。

GSPO 的 sequence-level ratio 让同一 response 内所有 token 共享一个权重，训练目标和 sequence reward 对齐。代价是它牺牲了局部 token adaptivity：一条 sequence 里少数 token off-policy，hard clipping 可能让整条 sequence 的有效信号被压掉。

### SAPO：保留 sequence coherence，但恢复 token adaptivity

SAPO 继承 GSPO 对 sequence coherence 的追求，但不接受 GSPO 的 hard sequence clipping。它用 smooth gate 做 token-level attenuation：正常情况下平均 token gate 近似 sequence-level gate，满足 GSPO 的对齐诉求；异常情况下只下调 outlier token，不让 near-on-policy token 一起陪葬。

这就是 SAPO 的定位：**GSPO-like when ratios are coherent, token-adaptive when they are not**。

### ARPO：不争 ratio，而争 rollout 预算投到哪里

ARPO 和前三者不在同一轴上。它不主要讨论 token ratio 或 clipping 形式，而是问多轮 agent 轨迹里哪里值得采更多样本。它的回答是：工具返回后，模型前 10–50 个 token entropy 升高，说明这里是 step-level tool-use 行为的关键不确定点；因此从这里 branch partial rollouts 比单纯采完整 trajectory 更有效。

这意味着 ARPO 可以理论上叠加 DAPO / GSPO / SAPO 的 policy loss；它改的是 rollout collection / sampling structure。

## 与模型报告的关系

- [Qwen3 技术报告](../sources/qwen3.md)：官方 2025-05 报告的 reasoning RL 阶段写的是 GRPO；DAPO/GSPO/SAPO 都是后续或外部算法论文，不能回写成原报告事实。
- [Qwen3-VL 技术报告](../sources/qwen3-vl.md)：原报告有 own post-training pipeline；SAPO 论文提供后续/配套 Qwen3-VL RL 训练证据，说明 SAPO 用在 Qwen3-VL-30B-A3B preliminary cold-start 上，但不替换源报告 pipeline。
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)：本页可作为该概念页里「RL policy optimization 子谱系」的展开。

## 待追问

- DAPO 的 token-level loss 与 GSPO 的 sequence-level ratio 是否冲突？一个在 loss reduction 上按 token 平均，一个在 importance ratio 上按 sequence 加权；二者能否组合，需要实验。
- SAPO 是否会成为 GSPO 的严格替代，还是只在 outlier token 多 / hard clipping 脆的阶段更好？论文承认所有方法最终仍可能 instability。
- ARPO 的 partial rollout 如果配 GSPO / SAPO，shared prefix 与 branch token 的 advantage attribution 应该用 sequence-level 还是 token-adaptive gate？
- MoE 的 routing volatility 是 GSPO/SAPO 的核心动机之一；dense 模型上 sequence-level 方法相对 DAPO recipe 的收益是否同样大？

## 相关页面

- 来源：[DAPO](../sources/dapo.md)、[Group Sequence Policy Optimization](../sources/group-sequence-policy-optimization.md)、[Soft Adaptive Policy Optimization](../sources/soft-adaptive-policy-optimization.md)、[Agentic Reinforced Policy Optimization](../sources/agentic-reinforced-policy-optimization.md)
- 概念：[Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)、[异步 Agent RL](../concepts/asynchronous-agent-rl.md)
- 模型：[Qwen3](../models/qwen3.md)、[Qwen3-VL](../models/qwen3-vl.md)
