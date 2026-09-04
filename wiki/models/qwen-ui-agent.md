---
type: Model
title: "Qwen-UI-Agent"
description: "阿里 MAI-UI Team 的 foundation GUI agent 家族（27B / 35B-A3B / 4B），真机+沙箱、hybrid GUI+CLI 批动作、Action RL→Online RL，以及 proactive / 跨平台 harness；主评测变体是 27B。"
tags: ["model", "gui-agent", "qwen-ui-agent", "real-device", "computer-use", "alibaba"]
timestamp: 2026-09-05
---

# Qwen-UI-Agent

## 身份

Qwen-UI-Agent 是阿里 MAI-UI Team / Token Hub 发布的 **real-world-centric foundation GUI agent** 家族，定位为 MAI-UI 的续作。它不是通用 LLM 发布，而是专门做跨设备数字执行：看截图、读 CLI/API 反馈、在一次模型 turn 里发出单个或一批 GUI / bash / API 动作。覆盖 mobile、computer-use、browser 和 DeepSearch，并用一层 harness 把通知驱动的主动服务和跨手机–电脑工作流接上。

## 关键事实

| 属性 | 27B（主评测） | 35B-A3B | 4B |
| --- | --- | --- | --- |
| **团队** | MAI-UI Team, Alibaba Token Hub | 同左 | 同左 |
| **基座** | Qwen3.5-27B（Table 9 表注写成 base model） | 报告只写 corresponding base checkpoint；尺寸对应 Qwen3.5-35B-A3B（推断） | 同上，尺寸对应 Qwen3.5-4B（推断） |
| **总参数** | 27B dense（按基座规模） | 35B / 3B 激活（按家族命名） | 4B |
| **激活参数** | 27B | ~3B | 4B |
| **模态** | 多模态（文本 + 图像输入；文本输出）（已据原文核实：观察是截图 + CLI/API 文本，输出是 reasoning + 动作） | 同左 | 同左 |
| **后训练** | SFT（域 expert merge）→ Action RL → verifier-guided Online GRPO | 同配方；CUA / DeepSearch 训练发稿时未完成 | 报告主要给 grounding 数字 |
| **Benchmark** | MobileWorld 82.1%；MobileWorld-Real 92.2%；AndroidDaily 97.5%；OSWorld-Verified 79.5%；WebArena 73.6% | MobileWorld 65.0%；MobileWorld-Real 87.4%；AndroidDaily 93.9%；WebArena 69.2% | ScreenSpot-Pro 67.8%（zoom-in 74.0%） |
| **arXiv** | 2607.28227v1 | 同左 | 同左 |

## 技术身份

四个设计定义 Qwen-UI-Agent，而不是「再训一遍 GUI 的 Qwen」（已据原文核实）：

1. **沙箱规模化 + 真机闭环**：约 10,000 并发沙箱（redroid 移动容器、带 in-VM bash 的 OSWorld、Playwright 浏览器、Serper/Jina DeepSearch）配 100+ 真机 / 150+ 应用。health-aware 调度、一机多虚拟屏（真机吞吐约 20×）、以及把 model failure 与 environment failure 拆开的 AutoJudge。
2. **Hybrid GUI+CLI 与 batched action**：一次决策可发有序批动作。OSWorld-Verified 上 CLI 占 40.7% 动作、批动作 39.6%；OSWorld-v2 上分别 55.1% / 41.6%。Online RL 后出现 Bash 改状态、GUI 做眼睛的交叉验证，作者称没有显式 modality 目标。
3. **分层监督的飞轮 + 两段 RL**：VLM step-judge 扩 SFT（含失败轨迹的恢复段），可执行 verifier 留给 Online RL。Action RL 打六类局部复发错误；Online RL 用约 10,000 对 task–verifier 和成功率课程优化 >100 步轨迹。
4. **Harness 改「何时开始、在哪继续」**：通知 → event / affair / task；低风险可先准备，改票/支付仍要确认。跨平台用 OpenClaw-like planner 调 GUI subagent，虚拟屏并行且不挡用户。这层膜没有成对消融。

## 与相关模型的关系

- 与 [Xiaomi-GUI-0](xiaomi-gui-0.md) 同做移动真机，但 Qwen-UI-Agent 把真机嵌进跨域 foundation agent：同一策略还跑桌面 hybrid CLI、浏览器和 DeepSearch，并加 proactive harness。Xiaomi 的主数字在 RealMobile 100 任务；本报告主数字在自建 MobileWorld-Real 409 任务，**不能直接比**。
- 与 [UI-Mate](ui-mate.md) 同做桌面 computer-use。OSWorld-Verified 上 27B 报 79.5%，UI-Mate-27B 报 77.0%，评测是否同协议未交叉核实。UI-Mate 的差异化在 DemoCUA 示范工作流；Qwen-UI-Agent 的差异化在 CLI 批动作和主动服务膜。
- 27B 叠在 [Qwen3.5](qwen3.5.md) 上。通用 multimodal agent（[GLM-5V-Turbo](glm-5v-turbo.md) / [Kimi K2.5](kimi-k2.5.md)）把 GUI 当多能力之一；本模型把 GUI 执行当主任务，再用 in-distribution 复述保住基座 agentic 分。
- Online RL 是 MAI-UI 的 GUI GRPO 变体，不是 [GSPO](../sources/group-sequence-policy-optimization.md)。Action RL 的逐步结构化奖励与 Xiaomi cascade reward 同属「局部 dense 信号」，但公式和错误模式清单不同。
- Harness 自称 OpenClaw-like，和 [Prime Agent](../sources/prime-agent.md) / [Macaron-V1](../sources/macaron-v1.md) 同属执行膜，但本报告的膜绑在 GUI subagent 和通知 affair 上，没有 REPL 或 HCP 搜索实验。

## 相关页面

- [Qwen-UI-Agent 技术报告](../sources/qwen-ui-agent.md) - 来源页（图文交错）
- [Xiaomi-GUI-0](xiaomi-gui-0.md) - 移动真机 GUI agent 对照
- [UI-Mate](ui-mate.md) - 桌面 CUA + DemoCUA 对照
- [Qwen3.5](qwen3.5.md) - 27B 基座家族
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)
- [Agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md)
- [Agent harness](../concepts/agent-harness.md)
- [Agentic Engineering](../concepts/agentic-engineering.md)
