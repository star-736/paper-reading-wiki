# Agentic Engineering

## 定义

Agentic engineering 是这些报告中的共同趋势：软件工作正在从一次性代码生成，转向长周期工程任务。模型需要阅读仓库、规划修改、调用工具、运行测试、修复错误，并在多轮操作中维护状态。

这个词在 [GLM-5](../models/glm-5.md) 中最核心，但 [MiMo-V2-Flash](../models/mimo-v2-flash.md)、[DeepSeek-V4](../models/deepseek-v4.md)、[MiniMax-M2 Series](../models/minimax-m2-series.md)、[Kimi K2.5](../models/kimi-k2.5.md) 和 [ARPO](agentic-reinforced-policy-optimization.md) 都指向类似方向。

## 跨报告信号

- GLM-5 强调 ARC 能力：agentic、reasoning、coding，并加入异步 agent RL、preserved thinking 和 CC-Bench-V2 真实工程评测。
- MiMo-V2-Flash 强调软件工程 benchmark、search agent、tau2-Bench 工具使用，以及通过 MOPD 融合专门 teacher。
- DeepSeek-V4 强调长周期工作流和百万 token 上下文，把它视为 test-time scaling 与未来 online learning 的基础设施。
- MiniMax-M2 把 agentic engineering 进一步系统化：任务要有可执行环境、verifiable reward 或可信 judge evidence；训练端用 Forge 解耦 agent、rollout、training 和 reward。
- Kimi K2.5 把 agentic engineering 扩展到视觉和并行 agent：模型不仅要调用工具，还要创建 sub-agent、拆解任务、分片上下文，并聚合多源结果。
- ARPO 从算法采样结构切入：工具返回后模型前 10–50 个 token entropy 升高，因此 RL 不应只比较完整轨迹，而应在高熵工具调用步分叉 partial rollouts 来学习 step-level tool-use 行为。

## 为什么重要

Agentic engineering 改变了瓶颈。模型不只是生成正确片段，还要管理上下文、使用工具、吸收环境反馈、记住先前动作，并在长历史中保持稳定推理。

新增两篇报告后，一个更清晰的趋势是：agentic 能力越来越依赖运行系统。MiniMax 的 Forge 说明训练系统要能承受长尾 rollout、black-box scaffold 和大规模可验证反馈；Kimi 的 Agent Swarm 说明推理时也可以通过并行 sub-agent 改变任务复杂度和上下文形态。ARPO 进一步说明，哪怕不改 harness，**rollout 预算在轨迹内部的分配方式**也会影响 tool-use 行为学习：工具反馈后的高熵节点比完整轨迹开头更值得追加探索。

## 待追问

- 哪些 benchmark 最能预测真实 coding agent 生产力？
- Thinking preservation 在部署中应该暴露到什么程度？
- 更长上下文会减少 context management 的必要性，还是让 context management 更重要？

## 相关页面

- [异步 Agent RL](asynchronous-agent-rl.md)
- [Forge Agent-Native RL](forge-agent-native-rl.md)
- [Agent Swarm](agent-swarm.md)
- [Agentic 评测体系](agentic-evaluation-benchmarks.md)
- [Agentic Reinforced Policy Optimization](agentic-reinforced-policy-optimization.md)
- [百万 token 上下文服务](million-token-context-serving.md)
