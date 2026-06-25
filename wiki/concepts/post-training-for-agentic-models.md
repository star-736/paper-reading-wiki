---
type: Concept
title: "Agentic 模型的后训练"
description: "面向 agent 的 RL、MOPD、蒸馏与 ARPO 这类 step-level rollout 采样模式。"
tags: ["concept", "post-training-for-agentic-models"]
timestamp: 2026-06-06
---

# Agentic 模型的后训练

## 为什么后训练成为主战场

Base model 给出知识、代码和推理基础，但 agentic model 还需要在环境中行动：调用工具、观察反馈、调整计划、处理失败、跨多轮保留状态。当前沉淀的报告都把后训练作为把 base model 变成 agentic model 的关键路径。

## GLM-5：异步 Agent RL

[GLM-5](../models/glm-5.md) 的后训练流水线包括 SFT、reasoning RL、agentic RL、general RL 和 on-policy cross-stage distillation。它最值得沉淀的是 [异步 Agent RL](asynchronous-agent-rl.md)。

长周期 agent rollout 有明显长尾：某些任务会跑很久，导致同步 RL 中大量 GPU 等待最慢样本。GLM-5 把 rollout inference 和 training engine 放到不同 GPU 设备上，inference engine 持续生成轨迹，达到阈值后送给 training engine 更新。为了控制异步带来的 off-policy 问题，它使用 rollout log-prob、token-level clipping、stale sample dropping 和 optimizer reset 等机制。

## MiMo-V2-Flash：MOPD

[MiMo-V2-Flash](../models/mimo-v2-flash.md) 的核心是 [Multi-Teacher On-Policy Distillation](multi-teacher-on-policy-distillation.md)。它先训练多个 domain-specialized teachers，例如数学、代码、搜索、工具使用和安全；然后让 student 从自己的分布采样，并从对应 teacher 得到 token-level KL 奖励。

这和普通蒸馏的差别在于：student 学的是自己真实会生成的轨迹，而不是离线 teacher 数据。因此 MOPD 试图减少 exposure bias，并缓解 sequential training 中常见的 capability see-saw。

## DeepSeek-V4：多 reasoning mode 与 OPD

[DeepSeek-V4](../models/deepseek-v4.md) 的后训练强调三种 reasoning effort：Non-think、Think High、Think Max。不同模式使用不同 RL 配置、length penalty 和 context window。报告还使用多 teacher On-Policy Distillation（OPD）把十多个 domain expert 合并到统一模型中，并为了稳定性采用 full-vocabulary logit distillation。

DeepSeek-V4 的特别之处是把后训练和百万 token 基础设施绑在一起：OPD、RL rollout、teacher scheduling、FP4 QAT、token-granular WAL 都是为了让超长上下文下的训练和服务可执行。

## MiniMax-M2：Forge 与 Interleaved Thinking

[MiniMax-M2 Series](../models/minimax-m2-series.md) 的后训练先用 SFT 学 interleaved thinking：模型在 reasoning、tool call、tool observation 之间交替，并保留跨轮 thinking state。这样 agent 在长任务中可以复用先前假设和调试线索，而不是每一轮重新推理。

RL 阶段由 [Forge](forge-agent-native-rl.md) 承载。Forge 把工具、context management、memory 和 agent harness 都视为环境，只把 LLM completion 视为 action。奖励不仅包含最终任务质量，还包含 process reward 和 relative completion-time reward，因此训练目标同时覆盖正确性、过程质量和执行效率。

## Kimi K2.5：Joint Multimodal RL 与 PARL

[Kimi K2.5](../models/kimi-k2.5.md) 的后训练有两个特殊点。第一是 [多模态 agentic training](multimodal-agentic-training.md)：zero-vision SFT 之后做 joint text-vision RL，让视觉任务和文本任务共同优化。第二是 [Agent Swarm](agent-swarm.md) 的 PARL：冻结 sub-agents，只训练 orchestrator 学会创建、调度和聚合并行任务。

PARL 的辅助奖励先鼓励 parallel exploration 和 sub-agent 完成率，随后退火到 0，让最终策略回到任务质量。这相当于把“是否并行、怎样并行”也变成 RL 可学习的 agent 行为。

## ARPO：高熵工具调用步的 partial rollout

[Agentic Reinforced Policy Optimization](agentic-reinforced-policy-optimization.md) 不是新模型报告，而是把 Qwen2.5 / Llama3.1 / Qwen3 等 backbone 放进多轮工具环境里比较 RL 算法。它的关键观察是：模型收到外部工具反馈后，后续前 10–50 个 token entropy 会显著上升，搜索反馈的不确定性又高于 Python 反馈。因此 ARPO 不再只做完整 trajectory-level sampling，而是在高熵工具调用步从当前节点分叉 partial rollouts，并用 advantage attribution 区分共享前缀与分叉段。

这补上了现有几条路线之间的空档：GLM-5 / Forge 关心 rollout 系统如何高吞吐、低偏差地跑起来；ARPO 关心 rollout 预算在一条 agent 轨迹内部应该投到哪里。

## 综合框架

可以把这些报告的后训练看成六种互补能力：

- GLM-5：如何让 agent 在真实环境中高吞吐学习。
- MiMo-V2-Flash：如何把多个专门 teacher 的能力合成到一个 student。
- DeepSeek-V4：如何在超长上下文和多 reasoning mode 下做稳定蒸馏与 RL。
- MiniMax-M2：如何把 agent harness、reward、rollout、training 和 serving 组织成可扩展系统。
- Kimi K2.5：如何把视觉、文本和并行 agent orchestration 纳入同一后训练框架。
- ARPO：如何把探索预算从完整轨迹平均采样，转移到工具反馈后的高熵 step-level 行为。

## 待追问

- 异步 RL 的 off-policy 偏差和 MOPD 的 teacher-student gap 是否能统一建模？
- ARPO 的 entropy-based branching 能否嵌进 Forge / GLM-5 这类大规模 agent RL 系统，还是只适合较小规模 search/Python 工具实验？
- Agentic benchmark 的 reward 是否足够可靠，还是会过拟合 harness？
- “保留 thinking”提升长周期任务的同时，会不会带来隐私、延迟或上下文污染问题？
- Agent Swarm 这类运行时并行策略，应该训练进模型，还是保留在外部 agent framework 中？
