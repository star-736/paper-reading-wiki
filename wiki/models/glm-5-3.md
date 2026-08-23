---
type: Model
title: "GLM-5.3"
description: "Z.ai 的 agentic coding 模型发布版；官方博客称沿用 GLM-5.2 base model，增益来自长周期可验证环境和后训练规模化。"
tags: ["model", "glm-5-3", "agentic-coding"]
timestamp: 2026-08-14
---

# GLM-5.3

## 身份

GLM-5.3 是 Z.ai 发布的 agentic coding / cyber 能力模型。官方发布博客称它与 GLM-5.2 使用同一 base model，所有相对提升来自后训练；这不是一份架构或参数披露报告。主要来源：[GLM-5.3 官方发布博客](../sources/glm-5-3-blog.md)。

## 关键事实

| 项目 | 数值 |
| --- | --- |
| 总参数 | 未披露 |
| 激活参数 | 未披露 |
| 模态 | 未披露；发布页只覆盖 coding / cyber agent，不能据此推断模型输入模态 |
| Base model | 与 GLM-5.2 相同（官方发布声明） |
| 主要增益来源 | 后训练环境、任务多样性和训练计算量扩展（官方发布声明） |
| RL stack | SAO with compaction；`slime`（Megatron training + SGLang rollout） |
| Reasoning effort | `low` / `high` / `max`；发布页称不再支持关闭 thinking |

## 技术身份

GLM-5.3 的可沉淀定位是**后训练缩放的结果样本**：发布页把能力提升归因于更贴近真实专业工作的可验证长周期环境，而不是替换 base model。环境通过任务模式收集、可执行化、judge 解题检查、无参考解 verifier 合成和 reward-shortcut 回填形成闭环。

它也把 [异步 Agent RL](../concepts/asynchronous-agent-rl.md) 的工程瓶颈推进到训练—rollout 一致性和资源调度：多 teacher OPD 的预取、local-storage 分层缓存、以及按 rollout 工作负载自动调 prefill/decode 比例与并发。博客只给出结果级描述，不能据此复原算法或系统实现。

## 相关页面

- [GLM-5.3 官方发布博客](../sources/glm-5-3-blog.md)
- [GLM-5](glm-5.md) - 已公开技术报告对应的较早家族模型，不能与未披露的 GLM-5.2 base 直接画等号
- [Agentic engineering](../concepts/agentic-engineering.md)
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)
- [异步 Agent RL](../concepts/asynchronous-agent-rl.md)
- [Agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md)
