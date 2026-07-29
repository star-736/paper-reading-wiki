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

## 跨报告信号：Kimi K3 的 partial rollout + AgentENV

[Kimi K3](../sources/kimi-k3.md) 在 1M 上下文 agentic RL 上走了与 GLM-5 / Ring-2.6 同源但工程更重的 partial rollout 路线：

- **Partial rollout（沿用 Kimi K1.5 / K2.5）**：每 iteration 采样 K completions × N prompts，维持 `N×K` 活跃轨迹。**当 λ∈(0,1) 比例轨迹完成即暂停生成**进入 policy optimization，不等所有 rollout 结束。未完成轨迹入队下一 iteration 优先恢复。单条长 horizon 轨迹自然跨多 iteration，引入 data staleness——K3 靠 per-token regularization 约束 policy update 在局部邻域内，容忍极端 off-policy 数据（沿用 K2.5 算法）。
- **AgentENV microVM 沙箱**：partial rollout 的「暂停-恢复」要 sandbox 状态可持久化。K3 用 AgentENV（Firecracker microVM，开源 github.com/kvcache-ai/AgentENV）——incremental checkpointing 只存脏内存页（checkpoint 49ms / resume 133ms），三个高层操作：(a) Pause/Resume（paused sandbox 不耗 CPU/内存，agent 等推理结果时可 pause，占 lifetime 98%）；(b) Fork（从原状态 fork 新沙箱做 reward judging 无副作用）；(c) Snapshot（定期快照错误恢复）。OverlayBD 镜像 + ublk driver + P2P transport 实现 sub-second launch，copy-on-write memory 实现 6.5× 内存 overcommit。K3 训练+评测共创建 **51,219,741 个 sandbox** 跨 1,505,678 个镜像。
- **External KV cache pool**：1M multi-step rollout 下 prefix KV-cache miss 极贵，partial rollout + speculative decoding 加剧 prefix-block churn。K3 用 write-back 设计——active decoding block 留 GPU，可复用 idle prefix 仅在被驱逐时 write-back 到 CPU DRAM 外部池，下次复用 prefetch 回。**KDA states 与 MLA KV cache blocks 一起 offload/prefetch，生命周期对齐**（这是 K3 混合 KDA-MLA 架构特有的——两套 cache 必须同步管理）。训练 iteration 结束后训练状态 offload 到 NVMe 腾 DRAM。
- **Rollout auto-throttling scheduler**：多步 rollout context 渐进增长，固定 concurrency 难估且早期过保守。用 runtime signals（active/queued request count、KV cache utilization）动态控制发往 inference engine 的请求数。
- **Gradient-buffer reuse for non-policy model forwarding**：RL loss 需 forward-only non-policy 模型（如 reference model），权重太大无法常驻 GPU。把权重放 CPU，用 policy model 的 FP32 gradient-buffer storage 作后盾 materialize，ZeRO-2 下每 GPU 只持两个 VPP chunk 的 gradient buffer，streaming reference weights chunk by chunk（一个 slot 算当前 chunk，另一个 prefetch 下一个）。

**与 GLM-5 / Ring-2.6 的对比**：三者都解决长尾 rollout 的同步瓶颈，但 K3 的特点是 (1) 沙箱工程最重（microVM + Firecracker，vs GLM-5 的 container、Ring-2.6 的 rollout buffer）；(2) 混合 KDA-MLA 架构带来两套 cache 同步管理的独特挑战（KDA fixed-size recurrent state + MLA per-token KV cache 大小/生命周期根本不同，K3 设计 unified paged layout 让两者共享同一 pool）；(3) 1M 上下文规模最大（K3 训练上下文 1M vs GLM-5 SFT 到 202,752、Ring-2.6 256K）。K3 的 partial rollout 阈值用 λ 比例（vs GLM-5 轨迹数量阈值、Ring-2.6 global token budget Φ）。

## 跨报告信号：Ling-2.6 / Ring-2.6 的异步 RL

[Ling-2.6 / Ring-2.6](../sources/ling-2.6.md) 的异步 RL（ASystem + ARouter）与 GLM-5 解决同一问题但路径不同：

- **步边界约束**：GLM-5 用轨迹数量阈值；Ring-2.6 用 **global token budget $\Phi$** 约束每步——未完成轨迹 pause + persist（含 KV-cache fingerprint）到 cross-version rollout buffer，下一 policy version 恢复，而非丢弃或等待。
- **尾请求处理**：GLM-5 丢弃 stale sample；Ring-2.6 的 ARouter 把尾请求 **offload 到专用推理组**（spillover-based training-inference overlap），主推理组释放计算开始训练侧梯度累积，端到端性能提升 >80%。
- **版本偏移控制**：GLM-5 用 stale sample dropping；Ring-2.6 用 staleness manager（max_staleness × consumer_batch_size 约束），超限 segment 退役。
- **训练-推理精度对齐**：GLM-5 用 TITO 避免 text 重新分词；Ring-2.6 用 module-aware FP8 quantization（LM Head 走 FP32 ~2 点 reward 改善；Attention/Shared Experts 保持 BF16，Routed Experts 用 Blockwise FP8），控制 log-probability drift。
- **KPop vs IcePop**：Ring-2.6 的 KPop 用 binary KL divergence 替代前代 IcePop 的 uniform fixed-ratio constraint，解决 MoE RL 中训练-推理 mismatch 的异质性问题——低概率 token 的 ratio noise 更大，固定比率会过度 mask 它们。这与 GLM-5 的 double-sided importance sampling 是同一层问题的不同解法。

