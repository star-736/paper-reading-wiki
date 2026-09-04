---
type: Model
title: "π0.5"
description: "Physical Intelligence 的开世界 VLA：π0 的 PaliGemma + flow action expert，加上异构 co-training 与统一 subtask 头，在未见过的家里做长周期家务。"
tags: ["model", "vla", "robotics", "pi0.5"]
timestamp: 2026-09-05
---

# π0.5

## 身份

π0.5（读作 “pi oh five”）是 Physical Intelligence 在 [π0](pi0.md) 上的开世界泛化模型。骨干仍是 PaliGemma + 300M flow matching **action expert**；增量是 **heterogeneous co-training**（多机器人、web 语义、subtask 文本、口头逐步指令）和 **同一模型**先预测 semantic subtask、再出底层动作。它是 2025–2026 具身论文里常被当低层执行器/对照的那一代；[InternVLA-A1.5](internvla-a1.5.md) 的真机表和 LIBERO-Plus 都拿它比。[OpenVLA](openvla.md) 仍是更早的离散 token 基线，不是这一代。

## 关键事实

| 字段 | 值 |
| --- | --- |
| **总参数** | 与 π0 同级：PaliGemma VLM + ~300M action expert（Appendix A-E） |
| **VLM backbone** | PaliGemma（Figure 3：SigLIP 400M + Gemma 2.6B；附录写 2B VLM，配置同 π0） |
| **Action expert** | ~300M；后训练阶段才加入，随机初始化 |
| **模态** | 多模态（**最多四路 RGB** + **语言指令** + **本体感觉（文本 token）** → **subtask 文本** + **连续动作块**）。预训练还吃 caption / VQA / 检测。**无**音频 |
| **动作表示** | 预训练：FAST 离散 token；后训练/推理：flow matching 连续 chunk（长度 50，10 步去噪），50 Hz |
| **层次** | 同一权重：\(\hat\ell=\pi(\cdot\mid o_t,\ell)\)，再 \(a_{t:t+H}=\pi(\cdot\mid o_t,\hat\ell)\) |
| **目标平台** | 两款移动双臂，18–19 DoF，四相机 |
| **关键数据** | MM 约 400 小时 / ~100 个家；加 ME、CE（含 OXE）、HL、WD；后训练加 VI、去掉实验室 CE |
| **训练步数** | 预训练 280k（离散）+ 后训练 80k（\(\alpha=10\)） |
| **来源** | [π0.5 论文](../sources/pi0.5.md)（arXiv:2504.16054v1，2025-04-22） |

## 技术身份

π0.5 的主张是「开世界家务靠配方，不靠换架构」：

1. **专家还是 π0 那个 flow expert**。连续高频控制这条没换；换的是先用 FAST 把 VLA 当普通 VLM 训，再长 expert，以及用 attention mask 隔开离散/连续两种动作表示。
2. **高低层合一**。π0 的高层是另一个 VLM；π0.5 把 subtask 当成自己的文本输出，底层动作只条件在 subtask 上。Web 与 HL 数据主要喂这一层；ME/CE 的动作数据主要喂底层。
3. **97.6% 预训练不是目标平台家务**（§I）。作者用消融说明：跨本体（ME/CE）撑 mock 家务；web 数据撑 OOD 物体和高层；只堆移动操作小时数不够。
4. **评测协议是新房子**，不是 π0 那种预训练见过的叠衣服/收拾桌子。因此 π0.5 vs π0 的 Figure 12 是「同一机器人数据、π0.5 多了 HL/WD 与离散预训练」的家务泛化，不是 OpenVLA Bridge 协议。

**不要误读的边界**：模态没有音频。InternVLA-A1.5 表里的 π0.5 分数是那边的试管/化学任务重测，不是本模型论文的家庭进度条。EmbodiedSkills 等后续工作若写「用 π0.5 当低层」，指的是这一代 flow expert + subtask，不是 OpenVLA 的 256-bin。

## 相关页面

- 来源：[π0.5 论文](../sources/pi0.5.md)
- 前作：[π0](pi0.md)
- 离散 token 基线：[OpenVLA](openvla.md)
- 概念：[Vision-Language-Action](../concepts/vision-language-action.md)
- 真机对照：[InternVLA-A1.5](internvla-a1.5.md)
