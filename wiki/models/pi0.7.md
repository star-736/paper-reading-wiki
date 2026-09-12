---
type: Model
title: "π0.7"
description: "Physical Intelligence 约 5B 的可steer VLA：Gemma 3 4B + MEM 历史视觉 + 860M flow matching action expert。不再是 PaliGemma + FAST→flow 两阶段。"
tags: ["model", "vla", "robotics", "pi0.7"]
timestamp: 2026-09-12
---

# π0.7

## 身份

π0.7（读作 “pi oh seven”）是 Physical Intelligence 的下一代通用机器人 VLA，与 [π0](pi0.md) / [π0.5](pi0.5.md) 并列。动作头仍是 **flow matching action expert**，不是离散 bin，也不是第四种动作表示。换的是骨干和上下文：VLM 从 PaliGemma 换成 **Gemma 3 4B**，加上 **MEM** 风格历史视觉，专家从约 300M 放到 **860M**，prompt 里再写 episode metadata 和多视角 subgoal 图。来源是 [π0.7 论文](../sources/pi0.7.md)（arXiv:2604.15483v2）。

**已闭合：不建 π0.6 模型页。** 原文把 π0.6 / π0.6-MEM / π\*0.6 写成前作与 specialist 对照；那些 PDF 不在 `raw/`，不能用本页柱图反推。

## 关键事实

| 字段 | 值 |
| --- | --- |
| **总参数** | 约 5B（§IV、Fig. 2） |
| **VLM backbone** | Gemma 3 4B（含约 400M 视觉编码器；Figure 2 标 SigLIP 400M + Gemma 4B） |
| **历史** | MEM-style 视频历史编码器；最多 4 路相机 × 6 帧 |
| **Action expert** | **860M** flow matching；50 token chunk；adaptive RMSNorm 注入时间 |
| **模态** | 多模态（**多路 RGB + 历史** + **任务/子任务语言** + **本体感觉** + **可选 subgoal 图** + **episode metadata** → **连续动作块**）。**无**音频 |
| **动作表示** | flow matching 连续 chunk，\(H=50\)；推理 5 步 denoising；执行 \(\hat H\in\{15,25\}\) |
| **FAST** | 只作 VLM 的 Knowledge Insulation 训练信号；推理不吐 FAST。**不是** π0.5 的 FAST→flow 两阶段 |
| **高层 / 世界模型** | 高层策略同架构 Gemma 3 4B；subgoal 由独立 **BAGEL 14B** 生成，不是 VLA 本体 |
| **来源** | [π0.7 论文](../sources/pi0.7.md)（arXiv:2604.15483v2，2026-04-24） |

## 技术身份

π0.7 的主张是「可steer 的上下文」，不是再换一套动作头：

1. **专家还是 flow，但更大、骨干换了。** 连续高频控制这条没断；断的是 PaliGemma 和「先 FAST 预训练再拉长 expert」。FAST 还在，只负责训 VLM，梯度不从 expert 回灌（Knowledge Insulation，§III）。
2. **把「怎么做」写进 prompt。** 语言、质量/速度/是否犯错、多视角 subgoal、joint/ee 控制模式都可以 dropout，测试时任选。这样才能吃失败演示和自主 rollout，而不把策略平均掉（§I、§V）。
3. **开箱对齐 specialist，不是另报仿真榜。** Fig. 6 对标的是 π\*0.6 RL / π0.6 SFT 的任务特化策略；洗衣和装箱吞吐甚至超过 RL specialist。这不是 [AtomicVLA](atomicvla.md) 的 LIBERO 表，也不是 [π0.5](pi0.5.md) 的新房子家务。
4. **跨本体衬衫折叠**给了少有的精确数：从未见过该任务的双臂 UR5e 上，π0.7 (GC) 进度 85.6% / 成功 80%，对照人类遥操 90.9% / 80.6%（§IX-C）。

**不要误读的边界**：BAGEL 不是动作头。π0.6 不是本库条目。[AtomicVLA](atomicvla.md) 仍建在 π0 / π0.5 上，不是本页。评测柱图不要目测填百分比。

## 相关页面

- 来源：[π0.7 论文](../sources/pi0.7.md)
- 前作：[π0](pi0.md) · [π0.5](pi0.5.md)
- 离散 token 基线：[OpenVLA](openvla.md) · [RT-2](rt-2.md)
- 概念：[Vision-Language-Action](../concepts/vision-language-action.md)
- 仍建在 π0 / π0.5 上的技能路由：[AtomicVLA](atomicvla.md)
- 真机表仍对照 π0.5 的后续模型：[InternVLA-A1.5](internvla-a1.5.md)
