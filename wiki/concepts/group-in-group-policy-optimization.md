---
type: Concept
title: "Group-in-Group Policy Optimization"
description: "GiGPO 如何用 episode-level 宏优势加 anchor state grouping 的 step-level 微优势，在不增加 rollout 的前提下给长周期 LLM agent 做细粒度 credit assignment。"
tags: ["concept", "gigpo", "llm-rl-policy-optimization", "agentic-rl"]
timestamp: 2026-09-05
---

# Group-in-Group Policy Optimization

## 定义

Group-in-Group Policy Optimization（GiGPO）是 [GiGPO 论文](../sources/gigpo.md)提出的 critic-free agentic RL 算法。它在 GRPO 已经采好的轨迹组上再嵌一层：把跨轨迹重复出现的环境状态当成 anchor，比较同一状态下不同动作的折扣回报，得到 step-level 相对优势，再与轨迹级相对优势相加。

一句话：**GiGPO 不是换 reward，也不是多采 partial rollout；它是把「同一 prompt 的多条完整轨迹」里已经存在的状态碰撞，回收成 step-level 对照组。**

## 跨报告信号

- **[ARPO](agentic-reinforced-policy-optimization.md)**：同样针对多轮 agent 的 step 信号，但花钱的位置相反。ARPO 在工具反馈后的高熵节点**额外分叉** partial rollouts；GiGPO **不追加采样**，只对已有 group 做 hashmap 分组。两者都承认 trajectory-level GRPO 太粗，解法正交，论文之间没有对照。
- **[DAPO](../sources/dapo.md)**：补的是单轮 long-CoT GRPO 的 recipe（Clip-Higher / Dynamic Sampling / token-level loss / overlong shaping）。GiGPO 附录把 DAPO 技巧接到自己身上（`GiGPO_dynamic`），WebShop / 1.5B 成功率从 DAPO 的 66.1 再到 75.0，用来支持「层次优势与单轮 group 技巧可叠加」。
- **[VAPO](../sources/vapo.md)**：用独立 critic + Length-Adaptive GAE 给 long-CoT 做 token-level credit。GiGPO 拒绝 critic，用「重复状态 ≈ 免费对照实验」代替 value estimate。没有共同 backbone / 共同任务。
- **GLM-5 / Forge / Laguna 的 agent RL 系统**：关心 rollout 吞吐、harness 接入、异步 off-policy。GiGPO 假设已经能同步采到同一初始状态的一组轨迹；它不解决调度，只改这组轨迹内部的 advantage。
- **[SAO](single-rollout-asynchronous-optimization.md)**：认为异步下不该再等一组。GiGPO 的 $A^S$ 依赖组内状态碰撞；SAO 用 critic 换掉组。

## 为什么重要

长周期 agent 的稀疏奖励让 GRPO 陷入一个具体失败模式：一条成功轨迹里既有关键正确动作，也有绕路、点错商品、重复搜索；一条失败轨迹里也可能有局部正确的一步。轨迹级相对优势会把这些 step 涂成同一个标量。

GiGPO 的可执行假设是：**只要 group 内初始状态相同，环境转移又是（近似）确定性的，重复状态就会自然出现；这些碰撞足以构成 step-level 的组内基线，不必再训 critic，也不必从每个 $s_t$ 重新 rollout。** Figure 5 显示 ALFWorld 训练全程 size-1 组 <35%，即一半以上状态真的在组内复现。这解释了它为什么能在不增加 LLM 调用的前提下拉开与 GRPO 的差距，也划出了它的失效条件：状态几乎不重复时，$A^S$ 消失，算法退回 GRPO。

## 算法骨架（据原文核实）

1. 对任务 $x$ 初始化 $N$ 份相同环境，用 $\pi_{\theta_{\mathrm{old}}}$ 同步滚出 $\{\tau_i\}_{i=1}^{N}$。
2. 用总回报算 episode 相对优势 $A^E(\tau_i)$；$F_{\mathrm{norm}}$ 取 std（GRPO）或 1（RLOO 的常数倍）。
3. 把 group 内所有 $s_t^{(i)}$ 按精确相等（或 QA 上的 subsequence 相似度 $>0.9$）收成 $G^S(\tilde s)$。
4. 对组内每个动作用折扣回报 $R_t=\sum_{k\ge t}\gamma^{k-t}r_k$ 算 $A^S$；$\gamma=0.95$。
5. $A=A^E+\omega A^S$，默认 $\omega=1$，再做逐步 clipped surrogate + KL。

这个骨架来自原文 §4 与 Appendix D Algorithm 1。它改的是 **advantage 的构造**，不改 importance ratio 的粒度（仍是逐步 $\rho_\theta(a_t)$），也不改采样图。

## 实验信号

- ALFWorld / WebShop，Qwen2.5-1.5B 与 7B：相对 GRPO 的总体成功率分别约 +13 / +11 和 +13 / +9 个百分点（Table 1 的 `w/o std`）。闭源 prompting 远低于 RL。
- Search QA：3B 平均 42.1、7B 47.2，高于 Search-R1 / ZeroSearch；作者把工具调用次数下降解释为重复 query 被同组压制。
- 结构消融：去掉 $A^E$ 或 $A^S$ 都明显掉点，且大于 `w/ std` vs `w/o std` 的差距（Figure 4）。
- 成本：同一 GPU 显存与同一组 rollout；新增 grouping / $A^S$ 在 Figure 6 上是 0.01s / 0.53s 量级。
- 正交性：接上 DAPO recipe 后 WebShop 成功率继续涨；Qwen2.5-VL-3B 的 Sokoban / EZPoints 作为 VLM 补充，不是主结论。

## 待追问

- 开放工具、GUI、仓库编辑里状态几乎不精确重复时，相似度 grouping 会不会把不同决策情境错误对齐？
- 与 ARPO 同一 rollout 预算下的对照仍然缺失。
- $\omega$ 与 $\gamma$ 在主实验未扫；折扣回报在终局二元奖励下，等价于「越晚成功的正确动作 advantage 越大」，这是否会惩罚必要的早期探索？

## 相关页面

- 来源：[GiGPO](../sources/gigpo.md)
- [Agentic Reinforced Policy Optimization](agentic-reinforced-policy-optimization.md)
- [Single-Rollout Asynchronous Optimization](single-rollout-asynchronous-optimization.md)
- [Agentic 模型的后训练](post-training-for-agentic-models.md)
- [LLM RL policy optimization 对比](../comparisons/llm-rl-policy-optimization.md)
- [Agentic 评测体系](agentic-evaluation-benchmarks.md)
