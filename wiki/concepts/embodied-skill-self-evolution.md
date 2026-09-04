---
type: Concept
title: "具身 skill 自进化"
description: "把可检索、可版本化的执行知识（程序 + 修复）通过闭环痕迹诊断/验证后写入技能库，使第 N 个机器人任务不再从零开始；ASPIRE 是已 ingest 的 code-as-policy 实例，不是 VLA 动作头。"
tags: ["concept", "embodied-skill", "code-as-policy", "skill-library", "robotics"]
timestamp: 2026-09-05
---

# 具身 skill 自进化

## 定义

**具身 skill 自进化**指：技能是一份**可检索、可版本化的执行知识**，而不是一次 rollout 的权重更新。系统在真实或仿真机器人上跑闭环，用执行痕迹诊断失败、合成修复、验证后再决定是否写入库；之后的任务检索这些条目，而不是从空白 prompt 再搜一遍。

在已 ingest 的 [ASPIRE](../sources/aspire.md) 里，这份知识具体是 **code-as-policy 程序补丁**：失败签名、when-to-apply 条件、修复策略，必要时附代码草图（原文 §2.2）。它不是整段任务脚本，也不是 VLA 词表里的技能 token。

三个对比要先分清：

- **与 [VLA](vision-language-action.md)**：VLA 更新的是动作网络权重（离散 bin / flow expert / co-training）。ASPIRE 对照表里出现 OpenVLA / π0 / π0.5，但系统本身写/改程序、把经验写入 skill library，coder 权重冻结（Claude Opus 4.6；§3.1、§5）。不要把它读成第四种动作头。
- **与软件 agent 的 skill 文件**（[Macaron HCP](../sources/macaron-v1.md) / [Prime Agent](../sources/prime-agent.md) L3 skills）：对象从终端、工具、REPL 变成机器人感知–运动与接触动力学。同构的是「L0 不动、改可复用程序」；差异是痕迹是多模态 robot trace，sim-to-real 运的是 know-how 而不是像素或权重。
- **未 ingest 的近邻只作外部线索**：EmbodiSkill（清华 AIR + MSR，arXiv:2605.10332，frozen LLM 的 skill-aware reflection）、EmbodiedSkills（浙大，arXiv:2609.01281，VLA 上层 AgentLoop）、AtomicVLA（VLA + atomic skill-MoE）。本页没有读过它们的 PDF，不能编机制，也不能和 ASPIRE 混名。

## 跨报告信号

- **[ASPIRE](../sources/aspire.md)（2026，code-as-policy 实例）**：三件套是闭环执行引擎、skill library、进化搜索。引擎按 primitive 暴露 overlay / 规划返回 / 碰撞，而不是任务级 0/1。Coordinator 只晋升通过 debug 验证的可复用修复。Headline 数字是成功率百分点：LIBERO-Pro Object 相对 CaP-Agent0 约 +77；Robosuite handover 20→92；BEHAVIOR-1K Radio Task 56→88；LIBERO-Pro Long 零样本 Overall 30.5% vs CaP 3.8%（Table 2–5）。真机三条技能降低 token，不是直接部署仿真策略（Table 1）。
- **[Agent harness](agent-harness.md)**：Prime Agent / Macaron 已经证明冻结 L0 时改 runtime 程序可以扩展可达策略。ASPIRE 把同一层膜接到机器人执行引擎：膜标准化的是 traces、API 与 admission，策略构造仍留给 coding agent。
- **[Agent 记忆生命周期](agent-memory-lifecycle.md)**：技能库是持久记忆；coordinator 的审计与 debug 验证是一层 gate。ASPIRE §5 自己写还没有 pruning / ranking / 再验证，Table 6 出现随库增大的非单调，对应「gate 有、rollback/淘汰不足」。
- **VLA 页里的 OpenVLA / π0 / π0.5 / InternVLA-A1.5**：提供低层动作头对照。后续若有论文把「技能」写成 token 或 MoE expert，应回到 [VLA](vision-language-action.md) 的三条动作头，而不是本页的程序库。

## 为什么重要

具身系统要跨任务变强，有两条经常被写成同一句话的路：改策略网络，或改可复用执行知识。ASPIRE 把后一条做成可检查的对象——程序可 diff、技能有 when-to-apply、入库要验证。这对检索的含义是：问「VLA 怎么出动作」走 VLA 页；问「第 100 个操作任务为什么不该从零写」走本页。

它也把软件 agent 已经碰到的风险搬进机器人：持久化会保存过特或误导的修复；API 边界之外的行为写不出来；冻结的大 coder 是否可替换还没有验证。这些是机制缺口，不是宣传点。

## 待追问

- 程序库技能与 VLA 技能 token / atomic skill-MoE 能否叠在同一运行时，还是必须二选一？AtomicVLA / EmbodiedSkills 尚未 ingest。
- EmbodiSkill 的 training-free reflection 与 ASPIRE 的进化搜索+入库，失败归因和验证标准差在哪一层？PDF 未读。
- 技能库规模化后，检索会不会把过时条目送进新任务？ASPIRE 只把问题写进 Limitations，没有机制消融。
- 真机要成为终身学习，缺的是成功检测 / 复位 / 安全，还是跨本体 API 对齐？Table 1 分不开。

## 相关页面

- 已 ingest 实例：[ASPIRE](../sources/aspire.md)
- 相邻概念：[Vision-Language-Action](vision-language-action.md)、[Agent harness](agent-harness.md)、[Agent 记忆生命周期](agent-memory-lifecycle.md)
- VLA 对照来源：[OpenVLA](../sources/openvla.md)、[π0](../sources/pi0.md)、[π0.5](../sources/pi0.5.md)
- 软件侧 skill / harness：[Prime Agent](../sources/prime-agent.md)、[Macaron-V1](../sources/macaron-v1.md)
