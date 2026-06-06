# Agent Swarm

## 定义

Agent Swarm 是 [Kimi K2.5](../models/kimi-k2.5.md) 的并行 agent 编排框架。核心思想是：复杂任务不一定要由一个 agent 顺序执行所有工具调用，而可以由一个 trainable orchestrator 动态创建多个 frozen sub-agents，把任务拆成并行子任务。

## PARL 训练方式

Kimi K2.5 使用 Parallel-Agent Reinforcement Learning（PARL）训练 orchestrator。训练时 sub-agents 来自固定策略 checkpoint，并保持冻结；它们的轨迹不进入优化目标，只作为环境观察返回给 orchestrator。这样做的原因是减少多 agent 端到端训练中的 credit assignment 歧义：最终答案成功或失败，很难判断是 orchestrator 拆解错了，还是某个 sub-agent 执行差。

PARL 的 reward 包含三部分：

- `rparallel`：鼓励创建 sub-agent，避免 orchestrator 退化成单 agent 串行执行。
- `rfinish`：奖励 sub-agent 完成分配任务，抑制无意义地大量创建 sub-agent 的 reward hacking。
- `rperf`：最终任务质量或成功率。

训练后期 `rparallel` 和 `rfinish` 的权重逐步退火到 0，让最终策略回到任务质量目标。

## Critical steps

普通 step count 会惩罚并行，因为所有 sub-agent 步数都会被累加。Kimi 引入 critical steps，类似计算图 critical path：一个并行阶段的耗时由最慢 sub-agent 决定，而不是所有 sub-agent 步数相加。

这个指标让 orchestrator 学到“有效并行”：只有当拆分任务能缩短最长分支时才有收益；单纯制造更多 sub-agent 不会降低 critical path。

## 主动 Context Management

Agent Swarm 也可以看成 context management。传统策略如 summary、hide tool result、discard-all 通常是在上下文溢出后被动压缩。Agent Swarm 在任务开始时就把信息需求分片：每个 sub-agent 持有局部上下文和工作记忆，只把关键结果回传给 orchestrator。

这种 context sharding 的优点是减少全局上下文污染，让不同搜索分支、文件分析或视频片段理解可以并行推进。风险是 orchestrator 必须正确聚合结果；如果子任务拆错，局部结论可能难以纠正。

## 报告中的结果

Kimi K2.5 单 agent 在 BrowseComp 为 60.6；使用 Agent Swarm 后为 78.4。WideSearch 从 72.7 提升到 79.0。报告还称，在 WideSearch 中达到目标 Item-F1 所需执行时间比单 agent 降低约 3-4.5x。

这些结果说明 Agent Swarm 的收益不只是更快；并行探索还提升了复杂搜索任务的覆盖率和多源验证质量。

## 与 Forge 的区别

[Forge](forge-agent-native-rl.md) 是训练基础设施，可以接入各种 white-box 或 black-box agent。Agent Swarm 是 Kimi K2.5 里被 RL 训练出来的编排策略。前者解决“怎么大规模训练 agent”；后者解决“agent 运行时如何把任务并行化”。
