---
type: Model
title: "Agent-World"
description: "人大 + ByteDance Seed 的自演化 agent 训练场产出的 8B/14B 模型族，Qwen3 backbone，跨 23 benchmark。"
tags: ["model", "agent-world", "agentic", "environment-synthesis"]
timestamp: 2026-07-12
---

# Agent-World

## 身份

Agent-World 是中国人民大学 + 字节跳动 Seed 在 [Agent-World 论文](../sources/agent-world.md) 中发布的 agent 模型族（**Agent-World-8B / Agent-World-14B**），是"自演化训练场"方法论的产出物而非独立基座——基座是 [Qwen3-8B / 14B](qwen3.md)，经过冷启动 SFT + 多环境 GRPO RL + 两轮自演化 arena 训练得到。论文焦点不在模型本身，而在**如何可扩展地合成真实环境并用其持续诊断、驱动 agent 进化**。

## 关键事实

| 项 | 值 |
| --- | --- |
| **总参数** | 8B / 14B（继承 Qwen3-8B / Qwen3-14B，未提及新增模块） |
| **激活参数** | dense 模型（Qwen3-8B / 14B 为 dense），未单独披露 |
| **模态** | 纯文本（已据报告原文核实：全文围绕 MCP 工具/数据库/代码交互，23 benchmark 均为文本/工具/代码，无视觉/音频输入） |
| **基座** | Qwen3-8B / Qwen3-14B |
| **后训练** | 冷启动 SFT（40K 轨迹，Doubao-Seed-1.8 生成）→ GRPO RLVR（5K 样本）→ 2 轮自演化 arena（诊断 → 定向任务扩展 → continue RL） |
| **RL 算法** | GRPO，clip `ε_low=0.2` / `ε_high=0.28`（Clip-Higher），max 轨迹 80K tokens |
| **数据/诊断侧策略模型** | GPT-OSS-120B（环境挖掘 / 任务合成 / 代码生成 / agentic diagnosis） |
| **训练环境生态** | 1978 环境 / 19822 工具（来自 ~2.8K MCP servers + ~0.5K 工具文档 + ~0.2K 工业 PRD） |
| **评测** | 23 个 agent benchmark；主结果见表 1（MCP-Mark / BFCL V4 / τ2-Bench） |
| **机构** | 人大高瓴人工智能学院 + ByteDance Seed |
| **发布** | arXiv:2604.18292v1（2026-04-20） |

## 技术身份

1. **不是新基座，是训练方法论的产出**。Agent-World-8B/14B 的价值在于验证"可扩展真实环境合成 + 连续自演化训练"在小模型（8B/14B）上即可一致超过参数量大得多的基础模型（如 Qwen3-235B-A22B）和既有环境扩展基线。模型本体继承 Qwen3 dense 架构，未披露架构改动。
2. **环境即基础设施**。1978 环境 / 19822 工具的生态既是训练数据源、又是诊断 arena——同一个 `E` 在多环境 RL 阶段供训练，在自演化阶段供评测/诊断/定向扩展。这与 [Forge](../concepts/forge-agent-native-rl.md) 把 harness/context/memory 都当环境的思路同源，但 Agent-World 更进一步把"环境本身的可扩展合成"作为第一性问题。
3. **自演化 = agent-environment co-evolution**。每轮：分层采样 arena 环境 → 动态合成新评测任务 → agentic diagnosis 定位弱环境与错误模式 → 定向任务扩展（必要时数据库复杂化）→ continue RL。区别于一次性静态训练，也区别于 MiniMax-M2.7 的 self-evolution（后者强调 scaffold 修改与 MLE Bench，前者强调环境-任务层面的诊断驱动课程）。
4. **可执行 reward 双路**。graph-based 任务用 rubric-conditioned LLM-as-judge 逐条评 + 均值；programmatic 任务用可执行验证脚本 `V_code` 在 sandbox 验证答案与库状态。两者都靠 sandbox 执行收集 trace 推导 ground truth，保留可验证性。

## 与其它 agent 模型的关系

- 与 [Qwen3](qwen3.md)：直接以 Qwen3-8B/14B 为基座，是其下游 agent 训练产物。
- 与 [KAT-Coder](kat-coder.md)：两者都把 agentic 能力瓶颈定位在训练基础设施而非模型规模。KAT-Coder 聚焦 coding agent 的沙箱可靠性 / harness 泛化 / MoE RL 稳定性；Agent-World 聚焦**通用 agent 的环境多样性与自演化课程**，覆盖更广的 MCP 工具生态而非单一 coding 域。
- 与 [daVinci-Agency](../sources/davinci-agency.md)：daVinci-Agency 在 SFT 数据结构层面注入长周期监督（chain-of-PRs），Agent-World 在 RL 阶段用环境诊断驱动定向数据扩展——一个解决"训练数据该长什么样"，一个解决"训练中如何持续发现并补能力缺口"。
- 与 [Xiaomi-GUI-0](xiaomi-gui-0.md)：Xiaomi-GUI-0 的 error-driven flywheel 把模型自身错误转为反思/纠正监督（GUI 域）；Agent-World 的 agentic diagnosis 把失败 trace 转为弱环境定位与定向任务生成（通用 MCP 工具域），是同一"错误驱动"思想在不同粒度的落地。
- 与 [ARPO](../concepts/agentic-reinforced-policy-optimization.md) / [DAPO](../sources/dapo.md)：Agent-World 直接采用 GRPO + Clip-Higher（`ε_high=0.28`）这一已沉淀的 RL recipe，创新点不在 policy optimization 算法本身，而在多环境 rollout 的数据组织与自演化课程。

## 相关页面

- [Agent-World 来源页](../sources/agent-world.md)
- [Qwen3](qwen3.md) - 基座
- [Agentic Engineering](../concepts/agentic-engineering.md)
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)
- [Agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md)
