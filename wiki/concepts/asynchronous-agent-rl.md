---
type: Concept
title: "异步 Agent RL"
description: "GLM-5 如何用异步 rollout、TITO 和 token-level clipping 训练 agent；SAO 把 DIS 做成单 rollout 算法。"
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

GLM-5.3 的官方发布博客补上了环境如何继续扩的公开描述：由 research agent 从真实工作收集模式并生成含隐藏状态、多步依赖的可执行环境；judge agent 先尝试完成任务；verifier 在看不到参考解的条件下合成，并用 solver trajectory 找 reward shortcut。只有通过 oracle、no-op 与 unsolved-state 检查的 verifier 才提供二元 reward。该描述将“环境失败过滤”向前推进为“环境在进入训练前先经可解性和奖励可靠性审计”，但其覆盖率与误判率尚未公开。详见 [GLM-5.3 官方发布博客](../sources/glm-5-3-blog.md)。

## 关键判断

异步 Agent RL 的本质不是“换一个 RL 算法”，而是把 RL 训练变成一个分布式系统问题：调度、容错、token 对齐、KV-cache locality、环境质量都会影响最终模型能力。

GLM-5.3 进一步把**训练—rollout 数值一致性**显式当作这一系统问题的一部分：作者称 `slime` 支持全数值对齐，平均 log-prob 差异控制至 $10^{-7}$ 量级，并为长周期 rollout 自动调节 prefill/decode 比例和并发。这种数字是厂商自报，但强调了一个可复用判断：只要 policy update 依赖 rollout log-prob，训练和采样路径的微小数值漂移也可能是 agentic RL 缩放的隐性变量。

[Agentic Reinforced Policy Optimization](agentic-reinforced-policy-optimization.md) 则是互补的算法层问题：在一个 agent 轨迹内部，工具反馈后的高熵 step 是否应该追加 partial rollout 探索。前者解决长尾 rollout 怎么跑得动，后者解决有限 rollout 预算投到轨迹哪里。

[SAO](single-rollout-asynchronous-optimization.md) 把上面「系统机制」里的 DIS 收成一篇独立算法论文，并论证 **GRPO 组采样在异步下是结构错配**：组必须等最慢样本，等于把 straggler 屏障请回来。它的回答是 group size = 1，轨迹完成即训；没有组内 baseline 之后把 critic 请回来（更快 value 更新、冻结 attention、Skip-Observation GAE）。消融把贡献拆开：DIS 让 GRPO 也能跑满约 1000 step，单 rollout + critic 才在约 400 step 后继续拉开。详见来源页 [Single-Rollout Asynchronous Optimization](../sources/single-rollout-asynchronous-optimization.md)。GLM-5.3 博客写的「SAO with compaction」是发布声明，compaction 不在该 PDF。

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

## 跨报告信号：Laguna 的在线 agentic RL 基础设施

[Laguna](../sources/laguna-m1-xs2.md)（Poolside，2026-05）的 agentic RL 走**在线**路线（非 GLM-5 的异步解耦，而是 trainer 与 inference fleet 物理隔离但高带宽互联的在线同步），三处工程值得与 GLM-5 / K3 / Ring-2.6 对照：

- **TITO + chat-template 对齐**：用 token-in-token-out API for RL actors（与 GLM-5 同一动机——保 token ID 跨多轮稳定，避免 text 重新分词错位）。独有补充是 `render_assistant_messages_raw` flag：RL 渲染器与生产 chat template 独立迭代，但每次生成后用该 flag 把 rollout token 原样插回 assistant message 块并断言解码字符串与 rollout 存的 prefix 严格相等——从机制上消除「RL 训的格式 ≠ 部署格式」的退化（缺一个换行/尾空格就能显著掉点）。
- **trainer↔inference 权重同步**：NCCL point-to-point GPUDirect RDMA，n→m fan-out（m 在 2n–3n 间），每 2 optimizer step 广播一次，异步（训练不阻塞）。两个安全原语保证在线 RL 良定义：权重广播触发 inference 侧 KV-cache reset（防不同权重版本 token 混入）；权重更新 block 在途 rollout step（保证单 rollout 视 policy 为 piecewise-constant，与 loss 里 importance ratio $\rho_t$ 假设一致）。staleness 上限 10 optimizer step，但因训练/推理 GPU 配比实际从未触及——所有轨迹都非浪费。这与 GLM-5 的 stale sample dropping、Ring-2.6 的 staleness manager 是同一问题的不同治理粒度（Laguna 靠配比自然消除，另两者靠显式丢弃/退役）。
- **FP8 KV cache for rollout**：131072 全 context 下 KV cache 主导 inference-replica 显存，存 FP8 约翻倍单 replica 并发轨迹数。release 跑 BF16 权重 + FP8 KV cache；预发布消融也试过 FP8 权重（in-flight block-wise 再量化无校准），稳定性无碍但 train-inference KL mismatch 变大，release 仍保 BF16 权重——这是「为安全牺牲一半并发」的诚实权衡。

**与三家的定位**：Laguna 不做 partial rollout（K3/ Ring-2.6）也不做轨迹数量阈值的异步解耦（GLM-5），而是靠「trainer↔inference 高带宽直连 + 每 2 step 同步 + 配比自然消 staleness」把在线 RL 跑到吞吐可行。它的独有价值在 chat-template 对齐断言——这是已收录报告里最严格的「RL 训练格式 = 部署格式」工程保证。

## 相关页面

- 算法：[Single-Rollout Asynchronous Optimization](single-rollout-asynchronous-optimization.md)
- 来源：[SAO 论文](../sources/single-rollout-asynchronous-optimization.md)、[GLM-5 技术报告](../sources/glm-5.md)、[GLM-5.3 官方发布博客](../sources/glm-5-3-blog.md)
- [Agentic Reinforced Policy Optimization](agentic-reinforced-policy-optimization.md)、[Group-in-Group Policy Optimization](group-in-group-policy-optimization.md)
- [LLM RL policy optimization 对比](../comparisons/llm-rl-policy-optimization.md)
