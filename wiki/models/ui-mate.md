---
type: Model
title: "UI-Mate"
description: "腾讯 HY Frontier 的开源权重 foundation GUI agent 家族（9B / 27B），环境接地 SFT+RL 加上 DemoCUA 上下文示范；桌面 computer-use，不是移动真机 agent。"
tags: ["model", "gui-agent", "ui-mate", "computer-use", "tencent"]
timestamp: 2026-08-29
---

# UI-Mate

## 身份

UI-Mate 是腾讯 HY Frontier Team 发布的 **foundation GUI agent** 家族。它不是通用 LLM，而是专门做桌面 computer-use：看屏幕截图、读自然语言指令（可选一条多模态示范），输出 reasoning + GUI 动作，在 Ubuntu / Windows / macOS 上直接操作应用。核心定位是两件事同时成立——指令-only 的开源 computer-use 要够强，并且在指令说不清程序时能把一条示范当成可改编的 subtask workflow，而不是 RPA 式回放。

## 关键事实

| 属性 | UI-Mate-9B | UI-Mate-27B |
| --- | --- | --- |
| **团队** | Tencent HY Frontier Team（Project Lead Zilong Huang） | 同左 |
| **基座** | Qwen3.5-9B | Qwen3.6-27B |
| **总参数** | 9B（报告按基座规模标注，未提及新增模块） | 27B（同上） |
| **激活参数** | 报告未把 9B/27B 写成 MoE；按基座规模理解 | 同左 |
| **模态** | 多模态（文本 + 图像输入；文本输出）（已据原文核实：观察是截图，输出是 reasoning + 动作；示范含前后帧但不注入像素坐标） | 同左 |
| **后训练** | SFT → 在线 GRPO agentic RL；9B 更依赖显式 reasoning 与更严 evaluator | 同配方；27B 可混用有/无 reasoning 轨迹 |
| **Benchmark** | OSWorld-Verified 66.2%；WAA 61.7%；OSWorkerBench 34.0% / 66.55% | OSWorld-Verified 77.0%；WAA 66.2%；OSWorkerBench 41.0% / 76.86% |
| **arXiv** | 2608.15930v1 | 同左 |

## 技术身份

四个设计定义 UI-Mate，而不是「更大的 Qwen 再训一遍 GUI」（已据原文核实）：

1. **环境接地数据飞轮**：任务、可运行初始态、过滤轨迹和 capability-tree 再平衡闭环。难度被放在真实文件与跨应用状态上，而不是把指令写得更绕。
2. **Verifier-first RL**：任务 bundle 必须满足 `R(E0)=0, R(E⋆)=1`，再用 hard-negative / alternative-positive 探针修 evaluator。决策单元是 decision turn；credit 做 turn centering、可选 PCM、token 归一化，并用 IcePop + SeqClip 挡异步 staleness。
3. **DemoCUA**：示范是坐标无关的子任务工作流。live 截图权威；训练故意少给中间动作、混入错位/无关示范，推理再给完整序列。
4. **OSWorkerBench 成对协议**：100 个办公长任务上，self-demo / variant-demo 与指令-only 共用同一 verifier。本报告只把 33-task self-demo 写成主结果；45-task 迁移设置未系统汇总。

## 与相关模型的关系

- 与 [Xiaomi-GUI-0](xiaomi-gui-0.md) 同属 GUI agent，但平台和瓶颈不同：Xiaomi 做移动真机分布对齐和 error-driven 恢复监督；UI-Mate 做桌面 CUA 的可验证环境规模化，以及「一条示范如何变成可靠程序」的交互协议。
- 与 [Qwen-UI-Agent](qwen-ui-agent.md) 同报 OSWorld-Verified（本页 77.0%，对方 79.5%），但对方主打 hybrid GUI+CLI 批动作和 proactive harness，本页主打 DemoCUA；是否同评测协议未交叉核实。
- 9B 明确叠在 [Qwen3.5](qwen3.5.md) 上；27B 叠在 Qwen3.6-27B 上，wiki 尚无该基座的一手架构页。
- 通用 multimodal agent（[GLM-5V-Turbo](glm-5v-turbo.md) / [Kimi K2.5](kimi-k2.5.md)）把 GUI 当多能力之一；UI-Mate 把 computer-use 当主任务，并单独训练示范条件化策略。
- RL 是 GRPO 变体而不是 [GSPO](../sources/group-sequence-policy-optimization.md)；异步 mismatch 过滤复用 IcePop 思路，见 [Ling-2.6](ling-2.6.md)。

## 相关页面

- [UI-Mate 技术报告](../sources/ui-mate.md) - 来源页（图文交错）
- [Xiaomi-GUI-0](xiaomi-gui-0.md) - 移动真机 GUI agent 对照
- [Qwen-UI-Agent](qwen-ui-agent.md) - 跨域 GUI agent + hybrid CLI 对照
- [Qwen3.5](qwen3.5.md) - 9B 基座家族
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)
- [Agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md) - OSWorkerBench
- [Agentic Engineering](../concepts/agentic-engineering.md)
