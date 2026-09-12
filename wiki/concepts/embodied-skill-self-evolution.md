---
type: Concept
title: "具身 skill 自进化"
description: "具身技能如何跨任务变强的三条路：ASPIRE 写/改程序并扩张库；EmbodiSkill 冻结 LLM、用 skill-aware reflection 只改技能正文；EmbodiedSkills 固定 typed 合同 + AgentLoop，库不扩张。"
tags: ["concept", "embodied-skill", "code-as-policy", "skill-library", "robotics"]
timestamp: 2026-09-12
---

# 具身 skill 自进化

## 定义

**具身 skill 自进化**指：技能是一份**可检索、可版本化的执行知识**，而不是一次 rollout 的权重更新。跨任务变强可以改这份知识，也可以不改、只把低层策略和 runtime 合同拆开。已 ingest 的三篇不要混名，数字也不许互填：

| 路 | 技能是什么 | 谁在变 | 典型产物 |
| --- | --- | --- | --- |
| [ASPIRE](../sources/aspire.md) | **可执行程序补丁**（失败签名 / when-to-apply / 修复草图） | 冻结 Claude 写/改程序，coordinator 验证后写入**会扩张的库** | skill library |
| [EmbodiSkill](../sources/embodiskill.md) | **自然语言程序规格** \(S=(S_{\mathrm{body}},S_{\mathrm{app}})\) | 冻结执行器；进化模型做 skill-aware reflection，只改正文，执行偏差只进附录 | 一份可修订技能文档 |
| [EmbodiedSkills](../sources/embodied-skills.md) | **固定 typed 执行合同** | 合同不扩张；低层任务特化 [π0.5](../models/pi0.5.md)，高层 Qwen3-VL scheduler SFT | AgentLoop / harness |

三个对比还要先分清：

- **与 [VLA](vision-language-action.md)**：VLA 更新的是动作网络权重（离散 bin / flow expert / co-training）。ASPIRE 对照表里出现 OpenVLA / π0 / π0.5，但系统本身写/改程序；EmbodiSkill 的执行器是冻结 Qwen，不是动作头。不要把这三条路读成第四种动作头。
- **与 [SayCan](../sources/saycan.md)**：那边的 skill 是评测时**固定**的语言条件策略 + value function，LLM 只在这张表上打分；库不会因失败写入新条目。ASPIRE 的库会扩张，EmbodiSkill 的正文会改写，EmbodiedSkills 的合同保持固定。不要用 SayCan 的 84% 规划去填这三张表。
- **与软件 agent 的 skill 文件**（[Macaron HCP](../sources/macaron-v1.md) / [Prime Agent](../sources/prime-agent.md) L3 skills）：对象从终端、工具、REPL 变成机器人感知–运动与接触动力学。同构的是「L0 不动、改可复用程序」；差异是痕迹是多模态 robot trace 或具身轨迹，sim-to-real 运的是 know-how 而不是像素或权重。
- **与 [AtomicVLA](../sources/atomicvla.md)**：那边的 skill 是 **SG-MoE 专家模块**，低层仍是 [π0](../models/pi0.md) / [π0.5](../models/pi0.5.md) 连续 action expert，不是本页三条路里的程序、正文或 typed 合同。入口在 [VLA](vision-language-action.md)，不要读成第四种动作头，也不要用它的 LIBERO / 真机数字填本页。

## 跨报告信号

- **[ASPIRE](../sources/aspire.md)（2026，程序库 + 进化搜索）**：三件套是闭环执行引擎、skill library、进化搜索。引擎按 primitive 暴露 overlay / 规划返回 / 碰撞，而不是任务级 0/1。Coordinator 只晋升通过 debug 验证的可复用修复。Headline 数字是成功率**百分点**：LIBERO-Pro Object 相对 CaP-Agent0 约 +77；Robosuite handover 20→92；BEHAVIOR-1K Radio Task 56→88；LIBERO-Pro Long 零样本 Overall 30.5% vs CaP 3.8%（Table 2–5）。真机三条技能降低 token，不是直接部署仿真策略（Table 1）。不要用这些数填 EmbodiSkill 的 ALFWorld 表。
- **[EmbodiSkill](../sources/embodiskill.md)（2026，training-free，改技能正文）**：冻结执行器（ALFWorld 上是 Qwen3.5-27B / Qwen2.5-14B），进化模型（GPT-5.2 / Gemini-3-flash）对轨迹做 skill-aware reflection。成功只允许 Discovery / Optimization，失败才判 SkillDefect 或 ExecutionLapse；后者只更新附录，不改规则（§3.2.1–3.2.2）。Headline 93.28% 是 ALFWorld、Qwen3.5-27B + GPT-5.2；摘要 31.58% / 25.01% / 19.04% 是**相对涨幅**，Table 3 的 \(\Delta_{\mathrm{aware}}\) +14.92 才是百分点。没有真机，也没有 VLA 后端。不要用这些数填 ASPIRE / EmbodiedSkills。
- **[EmbodiedSkills](../sources/embodied-skills.md)（2026，VLA 上层合同，库不扩张）**：六相 AgentLoop + policy–runtime 分离。RoboTwin 2.0 86.20% / LIBERO 97.40% 是任务特化低层 π0.5 的执行成绩；环的证据是消融（去验证 48.2、去 subtask 34.4、单 chunk 19.5）。技能集合按相配置，没有 coordinator 晋升新修复，也不改技能正文。入口仍是 [VLA](vision-language-action.md) 和 [Agent harness](agent-harness.md)。
- **[Agent harness](agent-harness.md)**：Prime Agent / Macaron 已经证明冻结 L0 时改 runtime 程序可以扩展可达策略。ASPIRE 把同一层膜接到机器人执行引擎；EmbodiSkill 的膜是技能文档本身；EmbodiedSkills 的膜是 guarded runtime。
- **[Agent 记忆生命周期](agent-memory-lifecycle.md)**：ASPIRE 的技能库是持久记忆，coordinator 的审计与 debug 验证是一层 gate；§5 自己写还没有 pruning / ranking / 再验证，Table 6 出现随库增大的非单调。EmbodiSkill 攒满 \(B\) 条反思才改正文，但没有独立 Limitations，文档膨胀和附录噪音没有消融。
- **[SayCan](../sources/saycan.md)（2022，固定技能表 + affordance，不是本页实例）**：LLM × value function 选下一步，技能集合在评测中固定（加抽屉技能是改选项集和 prompt，不是进化搜索）。规划 84% / 执行 74% 是厨房 101 条指令，协议不能和 ASPIRE 的 LIBERO-Pro 或 EmbodiSkill 的 ALFWorld 横比。EmbodiSkill §2.2 把它写成「选已有技能、不从轨迹改技能」。
- **VLA 页里的 OpenVLA / π0 / π0.5 / π0.7 / InternVLA-A1.5 / AtomicVLA**：提供低层动作头对照。[π0.7](../sources/pi0.7.md) 已 ingest：仍是连续 flow，但换成 Gemma 3 + MEM + 860M expert，不再是 PaliGemma + FAST→flow。[AtomicVLA](../sources/atomicvla.md) 已 ingest：技能是 SG-MoE 路由，低层仍停在 π0 / π0.5 连续专家，应回到 [VLA](vision-language-action.md)，而不是本页这三条路。

## 为什么重要

具身系统要跨任务变强，经常被写成同一句话的其实是三条路：改策略网络；改可复用执行知识（程序库或技能正文）；或者不改技能合同、只把 proposal 和 runtime 拆开。ASPIRE 把程序做成可检查对象——可 diff、有 when-to-apply、入库要验证。EmbodiSkill 把同一「冻结 L0、改外部知识」收成 skill-aware 文档编辑：先判断是条款错了还是没遵守，再决定改正文还是只加附录。EmbodiedSkills 走的是合同固定：低层策略可以特化或替换，跨任务变强不靠库扩张。

对检索：问「VLA 怎么出动作」走 [VLA](vision-language-action.md)；问「第 100 个操作任务为什么不该从零写程序」走 ASPIRE；问「失败是技能写错还是没按技能做」走 EmbodiSkill；问「proposal 如何先验后验」走 EmbodiedSkills / harness。

它也把软件 agent 已经碰到的风险搬进机器人：持久化会保存过特或误导的修复；API 边界之外的行为写不出来；冻结的大 coder / 进化模型是否可替换还没有验证。这些是机制缺口，不是宣传点。

## 待追问

- 程序库技能、技能正文、typed 合同和 AtomicVLA 的 skill-MoE 能否叠在同一运行时，还是必须四选一？[EmbodiedSkills](../sources/embodied-skills.md) 只把低层 VLA 做成可替换后端，没有程序库或 skill-MoE。[EmbodiSkill](../sources/embodiskill.md) 没有 VLA 后端。[AtomicVLA](../sources/atomicvla.md) 的库是专家权重，不是程序。
- 失败归因差在哪一层？ASPIRE 靠 per-primitive 多模态痕迹 + debug 验证再入库；EmbodiSkill 靠反思类型（条款错 vs 没遵守），执行偏差不改正文，但没有独立验证器；EmbodiedSkills 的验证是 post-action 路由（Advance/Continue/Reobserve/Replan/Recover/Finish）。三篇原文都没互相对照。
- 技能库 / 技能文档规模化后，检索或上下文会不会把过时条目送进新任务？ASPIRE 只把问题写进 Limitations；EmbodiSkill 没有 Limitations 节。
- 真机要成为终身学习，缺的是成功检测 / 复位 / 安全，还是跨本体 API 对齐？ASPIRE Table 1 分不开；EmbodiSkill 没有真机。

## 相关页面

- 已 ingest 的三条路：[ASPIRE](../sources/aspire.md) · [EmbodiSkill](../sources/embodiskill.md) · [EmbodiedSkills](../sources/embodied-skills.md)
- 相邻概念：[Vision-Language-Action](vision-language-action.md)、[Agent harness](agent-harness.md)、[Agent 记忆生命周期](agent-memory-lifecycle.md)
- 固定技能表 + value function，不是本页：[SayCan](../sources/saycan.md)
- VLA 对照来源：[OpenVLA](../sources/openvla.md)、[π0](../sources/pi0.md)、[π0.5](../sources/pi0.5.md)、[π0.7](../sources/pi0.7.md)（Gemma 3 + MEM，不是第四种动作头）、[AtomicVLA](../sources/atomicvla.md)（SG-MoE 路由，不是本页三条路）
- 软件侧 skill / harness：[Prime Agent](../sources/prime-agent.md)、[Macaron-V1](../sources/macaron-v1.md)
