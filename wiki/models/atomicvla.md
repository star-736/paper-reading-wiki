---
type: Model
title: "AtomicVLA"
description: "中山大学等的 VLA：AtomicVLA 建在 π0、AtomicVLA* 建在 π0.5；技能是 SG-MoE 按原子技能抽象做 top-1 路由，低层仍是连续 action expert。不是第四种动作头。"
tags: ["model", "vla", "robotics", "atomicvla"]
timestamp: 2026-09-12
---

# AtomicVLA

## 身份

AtomicVLA 是中山大学 + 鹏城实验室 + 引望智能的机器人 VLA：在预训练 [π0](pi0.md) / [π0.5](pi0.5.md) 上加 **Skill-Guided MoE**，用原子技能抽象路由专家，同一 VLM 切换 `[think]` / `[act]`。它不是新的离散 bin，也不是另一套 flow；低层动作生成仍走 π0 家族的连续 action expert。来源是 [AtomicVLA 论文](../sources/atomicvla.md)（arXiv:2603.07648v2）。

两个检查点：

- **AtomicVLA**：基座 π0
- **AtomicVLA\***：基座 π0.5

## 关键事实

| 字段 | 值 |
| --- | --- |
| **基座** | AtomicVLA ← [π0](pi0.md)；AtomicVLA* ← [π0.5](pi0.5.md)（§4.1） |
| **动作头** | 仍是 π0 / π0.5 的连续 action expert；共享专家「maintains the pre-trained action generation capabilities of π0」（§3.3） |
| **技能机制** | SG-MoE：1 个共享专家 + K 个技能专家，路由 top-1 再与共享专家加权（Eq. 3） |
| **专家数** | LIBERO / 真机 K=5；CALVIN K=8（§4.1） |
| **参数（Table 10）** | π0 3.24B；K=5 4.17B；K=8 4.81B；K=12 5.65B |
| **模态** | 多模态（多路 RGB + 语言指令 + 本体感觉 → 连续动作块）。**无**音频。已据来源核实。 |
| **Think / Act** | 特殊 token `[think]` / `[act]`；think 吐任务链 + 原子技能抽象，act 按最近一次抽象路由（Algorithm 1） |
| **训练** | 仿真 100k iter / 真机 30k，batch 64，8×H200；技能专家随机初始化（A.3.1） |
| **来源** | [AtomicVLA 论文](../sources/atomicvla.md) |

## 技术身份

把它读回三条动作头时，只加一层路由，不要开第四列：

1. **不是 OpenVLA 的 256-bin。** 表里 OpenVLA 是对照，不是本模型的动作表示。
2. **低层仍是 π0 连续专家。** 共享专家保留预训练动作生成；技能专家是 Gemma 风格、独立 SwiGLU FFN 的附加模块（A.3.1）。输出仍是动作块 \(A_t\)。
3. **AtomicVLA\* 叠在 π0.5 上**，因此也带上开世界那一代基座；本页自己的规划接口是 `[think]` 的任务链 + 原子技能词（Pick / Place / Open / Close / Turn 等），不是另写一套 FAST→flow。
4. **技能 = 专家路由，不是程序、正文或 typed 合同。** 和 [ASPIRE](../sources/aspire.md) / [EmbodiSkill](../sources/embodiskill.md) / [EmbodiedSkills](../sources/embodied-skills.md) 不是同一对象。

**不要误读的边界**：Figure 1 写「Unlike previous VLA models with a single action head」——这里的 head 是「一个解码器吃所有技能」，不是「动作表示从 bin/flow 换成了第四种」。无音频。π\*0.6 只在附录未来工作里出现。

## 相关页面

- 来源：[AtomicVLA 论文](../sources/atomicvla.md)
- 基座：[π0](pi0.md) · [π0.5](pi0.5.md)
- 同系列下一代，不是本页基座：[π0.7](pi0.7.md)
- 概念：[Vision-Language-Action](../concepts/vision-language-action.md)
- 离散 token 对照：[OpenVLA](openvla.md)
- 不要混名的 skill 三路：[ASPIRE](../sources/aspire.md) · [EmbodiSkill](../sources/embodiskill.md) · [EmbodiedSkills](../sources/embodied-skills.md)
