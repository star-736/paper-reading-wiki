# Forge Agent-Native RL

## 定义

Forge 是 [MiniMax-M2 Series](../models/minimax-m2-series.md) 报告中的 agent-native RL 系统。它的目标不是训练普通问答模型，而是训练能在真实工具环境中长期行动的 agent：读仓库、跑终端、浏览网页、调用 API、管理上下文、拆分任务，并根据环境反馈更新策略。

## 为什么需要专门系统

Agent RL 的 rollout 有三个难点。第一，轨迹极长，可能接近 192K tokens，并包含上千个中间动作。第二，任务耗时长尾明显，简单 API 调用可能几秒完成，复杂工程任务可能数小时。第三，agent harness 会重写上下文、调用 sub-agent 或压缩记忆，训练系统如果只看到静态 prompt-response，很难还原真实推理分布。

Forge 的建模边界很清楚：LLM completion 是 action；工具执行、context management、memory access 和 agent state transition 都属于环境。这样训练可以把每个 `(state, action)` 对作为 policy-gradient 样本，同时在 episode 层面计算 reward-to-go。

## 系统结构

Forge 被拆成三层：

- Agent Side：运行任意 agent scaffold，负责工具调用、环境交互、上下文管理和轨迹记录。
- Middleware：Gateway Server 路由 completion 请求，Data Pool 异步收集 rollout 数据。
- Training/Inference Side：Rollout Engine 生成 token，Train Engine 消费轨迹并同步权重。

这个结构支持两类 agent。White-box agent 会暴露 context management 逻辑，训练系统可以重建真实状态。Black-box agent 只暴露每次请求时呈现给 LLM 的上下文，因此也能纳入 API-only、层级 multi-agent 或内部复杂 scaffold。

## 奖励与优化

MiniMax 使用 CISPO 做 policy optimization，并用不对称 token-level importance clipping 控制更新幅度。长轨迹的 advantage 通过 reward-to-go 和 trajectory-level baseline 估计。

奖励是组合式的：

- process reward：惩罚格式错误、语言混杂等中间行为，奖励结构化推理。
- completion-time reward：鼓励正确且更快的轨迹，显式奖励并行工具调用或 sub-agent 调度带来的耗时下降。
- performance reward：任务最终成功率或质量。

这使 Forge 不只优化“答对”，还优化 agent 的行动过程和执行效率。

## 吞吐与长上下文效率

Forge 的工程优化集中解决 straggler 和重复前缀计算：

- Windowed FIFO：在滑动窗口内贪心消费已完成轨迹，窗口外保持 FIFO，平衡吞吐与数据分布一致性。
- Prefix tree merging：同一 rollout group 中共享长前缀的样本只计算一次前缀，报告称可带来最高 40x 训练加速。
- MTP co-training：RL 过程中持续训练 MTP draft path，避免 policy 变化导致 speculative decoding 接受率下降。
- Prefill-decode disaggregation 与 global KV cache pool：降低长上下文多轮 agent 的重复 prefill 成本。

## 与其他路线的关系

Forge 更像训练和系统基础设施；[Agent Swarm](agent-swarm.md) 更像模型学到的并行编排策略。两者都承认 agent 的瓶颈不只在模型参数，也在上下文、工具、轨迹调度和反馈质量。

## 待追问

- Black-box agent 只观察外部请求流时，能否充分学习隐藏 context management 策略？
- completion-time reward 是否会鼓励过度并行，导致工具成本或错误率上升？
- Windowed FIFO 的窗口大小是否需要随任务难度和 reward 稀疏度自适应？
