---
type: Concept
title: "Agentic 评测体系"
description: "SWE-bench、Terminal-Bench、BrowseComp、MCP-Atlas 等 benchmark 的作用和可比性风险。"
tags: ["concept", "agentic-evaluation-benchmarks"]
timestamp: 2026-06-06
---

# Agentic 评测体系

## 为什么单一 benchmark 不够

Agentic model 的评测不只是回答正确率。它需要覆盖代码修改、终端操作、网页搜索、工具调用、长周期规划、经济任务和多轮环境反馈。已沉淀的模型报告与 agentic RL 算法论文都在强调 agentic benchmark，但每个 benchmark 衡量的能力不同。

## 常见 benchmark

| Benchmark | 主要能力 | 在报告中的用途 |
| --- | --- | --- |
| SWE-bench Verified | 真实 GitHub issue 修复，经过人工验证的子集 | GLM-5、MiMo-V2-Flash、DeepSeek-V4 都用来衡量 coding agent。 |
| SWE-bench Pro | 更偏行业级、长周期 repository repair | MiniMax-M2、Kimi K2.5 用来衡量复杂工程修复能力。 |
| SWE-bench Multilingual | 多语言软件工程问题修复 | 用于测试模型跨语言代码维护能力。 |
| Multi-SWE-bench | 多 repo / 多语言 issue resolving | MiniMax-M2 用来测试跨仓库迁移和更宽软件工程覆盖。 |
| Terminal-Bench 2.0 | 终端环境中的任务执行 | 测试 shell、文件、环境和多步操作能力。 |
| BrowseComp | 高难度网页搜索与多跳信息综合 | 测试 search agent 和 context management。 |
| WideSearch | 广域、多源信息搜索 | Kimi K2.5 和 MiniMax-M2 用来测试并行搜索、覆盖率和综合能力。 |
| MCP-Atlas | 使用 MCP servers 的多步工具工作流 | 测试真实工具协议下的 tool-use 能力。 |
| tau2-Bench | 双控制环境中的对话 agent | 测试代理在任务约束和用户互动中的执行能力。 |
| Toolathlon | 长周期异构工具任务 | 测试多工具、多步骤任务完成。 |
| Vending-Bench 2 | 模拟经营任务 | 测试长期规划和资源管理。 |
| GDPval-AA | 办公与经济价值任务的 Artificial Analysis 子集 | MiniMax-M2、Kimi K2.5 用来测试 office/cowork agent 能力。 |
| XBench / xbench-DeepSearch | 中文 deep search / deep research split | ARPO、WebSailor 等用来测试深搜索；读分数时要看中文 prompt、judge 和浏览器设置。 |
| MLE Bench Lite | 自动机器学习工程任务 | MiniMax-M2 用来展示 M2.7 的 self-evolution 和 scaffold 修改能力。 |
| OSWorld / WebArena | GUI 与网页环境中的 computer-use | Kimi K2.5 用来测试视觉-操作结合的 agent 能力。 |

## 外部来源

- SWE-bench Verified：OpenAI 对 SWE-bench 子集进行人工筛选，目标是减少原始 benchmark 中不明确或不可解的问题。参考：[Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/).
- BrowseComp：OpenAI 提出的网页浏览困难问题集，用于测试 agent 搜索和综合能力。参考：[arXiv:2504.12516](https://arxiv.org/abs/2504.12516).
- Terminal-Bench 2.0：终端任务 benchmark，GLM-5 和 DeepSeek-V4 都提到环境歧义或 verified subset 问题。参考：[Terminal-Bench 2.0 PDF](https://openreview.net/pdf/574281303882f822808ab57ac3a57a2bddfbc7a3.pdf).
- MCP-Atlas：面向 MCP tool-use 工作流的 benchmark。参考：[MCP Atlas report](https://static.scale.com/uploads/654197dc94d34f66c0f5184e/MCP_Atlas_v4.pdf).

## 解读注意事项

不同报告中的同名 benchmark 未必完全可比。差异可能来自：

- agent framework：OpenHands、Claude Code、内部 harness 等会改变工具集合和提示。
- context management：discard-all、keep-recent-k、hierarchical context management 会显著影响 BrowseComp。
- parallelism：Agent Swarm 等并行 agent 编排会改变 WideSearch、BrowseComp 的延迟和覆盖率。
- judge model：BrowseComp 等需要 judge 的任务会受 judge prompt 和 judge model 影响。
- timeout 和 step budget：Terminal-Bench、MCP-Atlas、BrowseComp 的最大步数和超时会改变结果。
- verified subset：Terminal-Bench 2.0 Verified 与原始 Terminal-Bench 2.0 结果不可直接混用。

## 对已沉淀报告的影响

GLM-5 的强项是非常系统地讨论了 agentic engineering 环境构建和 context management。MiMo-V2-Flash 更强调在较小模型规模下 SWE-bench 与 BrowseComp 的提升。DeepSeek-V4 则把 agent benchmark 放在 512K context、内部 harness 和 1M context 能力背景下理解。MiniMax-M2 更强调统一 scaffold、verifiable reward 和 self-evolution 工作流。Kimi K2.5 则把 Agent Swarm 作为 benchmark 变量，直接比较单 agent、context management 和并行 agent 编排。ARPO 不是发布新模型，而是把 GAIA / WebWalkerQA / HLE / XBench 作为算法测试床，强调同一个 backbone 下 rollout 采样结构也会显著改变 tool-use 分数。

因此，读 benchmark 表时要先问：模型本体、agent harness、工具集合、context strategy、rollout / sampling 策略和 judge 设置分别是什么。
