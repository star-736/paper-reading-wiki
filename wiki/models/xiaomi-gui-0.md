---
type: Model
title: "Xiaomi-GUI-0"
description: "小米 SeerRay 团队的 native end-to-end multimodal GUI agent，基于 Qwen3-VL-30B-A3B-Instruct，真机闭环训练。"
tags: ["model", "gui-agent", "xiaomi", "multimodal"]
timestamp: 2026-07-11
---

# Xiaomi-GUI-0

## 身份

Xiaomi-GUI-0 是小米 SeerRay Team 开发的 native end-to-end multimodal GUI agent，面向真实移动环境（手机 / 平板 / 车机）。它不是通用 LLM，而是专门训练的 GUI 交互 agent——感知屏幕截图、理解自然语言指令、通过界面级动作（tap / swipe / text entry / navigation）完成任务。核心定位是**真机闭环**：训练和评测都以物理设备上的真实应用执行为中心，而非离线轨迹、模拟环境或标准化 benchmark。

## 关键事实

| 属性 | 值 |
| --- | --- |
| **团队** | 小米 SeerRay Team（通讯作者 Heng Qu / Jian Luan） |
| **基座** | Qwen3-VL-30B-A3B-Instruct |
| **总参数** | 报告未独立披露（继承 Qwen3-VL-30B-A3B 基座的 30B 总参 / 3B 激活，未提及新增模块） |
| **激活参数** | 同上 |
| **模态** | 多模态（文本 + 图像输入；文本输出）（已据报告原文核实：native end-to-end multimodal GUI agent，输入截图 + 文本指令，输出结构化 CoT + action JSON） |
| **训练硬件** | 64× NVIDIA H100（8 节点 × 8 GPU） |
| **训练框架** | verl（RL）+ Megatron-Core（训练后端）+ SGLang（rollout engine） |
| **训练数据** | SFT: ~1.2M GUI step samples + 4.4M grounding samples；Step RL: ~0.4M GUI step samples；Agentic RL: 数千任务在线生成 |
| **Benchmark** | RealMobile 72.0% success / 85.8% progress；AndroidWorld 78.9% |
| **arXiv** | 2606.31410v2 |

## 技术身份

四个关键设计定义了 Xiaomi-GUI-0 的技术身份（已据原文核实）：

1. **真机为主的混合基础设施**：物理设备为主执行环境，沙箱为辅。Device-Pull 调度让空闲设备按就绪状态主动拉取任务，适配真机的不可预测性（离线 / 登录失效 / 风控 / 冷却期）。这区别于 CUA-Gym / MobileGym 等合成环境方案。

2. **Error-driven data flywheel**：不简单扩数据量，而是围绕模型自身错误分布做定向修复。两阶段互补——交互式标注定位首个关键错误并纠正（GUI 失败级联，早期错误是根因），teacher 打分与接管在 scale 上产生 deviation–diagnosis–recovery 段。传统 flywheel 保留成功丢弃失败，Xiaomi-GUI-0 刻意产生从错误状态桥接回正确路径的数据。

3. **三阶段渐进训练**（SFT → Step RL → Agentic RL）：从 dense 到 sparse feedback 的课程。Step RL 用 [GSPO](../sources/group-sequence-policy-optimization.md) + cascade reward（L1-A / L1-B / L2 / L3 / L4 top-down early-exit，三值 reward −1.0 / −0.5 / 1.0）。Agentic RL 用 turn-level batching + curriculum sampling，把 trajectory-level advantage 分配到短序列 turn 上做高效在线更新。

4. **RealMobile 真机评测**：100 任务 / 14 应用 / 4 能力域 / 57% 跨应用。细粒度 sub-goal 打分（partial credit）+ veto 机制 + conditional branching + 双验证（XPath + logical semantic rules）。Safety & Reflection 是所有模型最弱域，揭示安全自纠正仍是 GUI agent 共同瓶颈。

## 与相关模型的关系

- 基于 [Qwen3-VL](qwen3-vl.md) 家族的 30B-A3B-Instruct 变体，叠加 GUI agent 专用后训练。
- 与通用 agentic 模型（[GLM-5](glm-5.md) / [MiMo-V2-Flash](mimo-v2-flash.md) / [DeepSeek-V4](deepseek-v4.md) / [MiniMax-M2](minimax-m2-series.md) / [Kimi K2.5](kimi-k2.5.md)）不同，Xiaomi-GUI-0 不追求通用 reasoning / coding 能力，而是专攻移动 GUI 交互。但其三阶段训练和 error-driven flywheel 的方法论与通用 agentic model 的后训练趋势一致。
- 与桌面 foundation GUI agent [UI-Mate](ui-mate.md) 对照：两者都做截图→动作的 end-to-end GUI，但 Xiaomi 优化真机异常态与错误恢复，UI-Mate 优化可验证办公环境和 in-context 示范。评测不可直接比——RealMobile 是移动真机，OSWorkerBench 是桌面 mock-app 办公流。
- 训练中复用 [GSPO](../sources/group-sequence-policy-optimization.md) 和 [DAPO](../sources/dapo.md) 的 dynamic sampling，是 LLM RL policy optimization 方法在 GUI agent 领域的直接应用。

## 相关页面

- [Xiaomi-GUI-0 技术报告](../sources/xiaomi-gui-0.md) - 来源页（图文交错）
- [Qwen3-VL](qwen3-vl.md) - 基座模型
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md) - 三阶段训练 + GSPO + cascade reward
- [Agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md) - RealMobile benchmark
- [Agentic Engineering](../concepts/agentic-engineering.md) - GUI agent 作为 agentic engineering 的移动端分支
- [Group Sequence Policy Optimization](../sources/group-sequence-policy-optimization.md) - Step RL / Agentic RL 的优化框架
- [DAPO](../sources/dapo.md) - dynamic sampling 来源
- [UI-Mate](ui-mate.md) - 桌面 CUA + DemoCUA 对照
