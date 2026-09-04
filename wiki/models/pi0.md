---
type: Model
title: "π0"
description: "Physical Intelligence 的 3.3B VLA：PaliGemma + 300M flow matching action expert，跨单臂/双臂/移动操作，连续动作块最高 50 Hz。"
tags: ["model", "vla", "robotics", "pi0"]
timestamp: 2026-09-05
---

# π0

## 身份

π0（读作 “pi zero”）是 Physical Intelligence 的通用机器人 VLA：在 [PaliGemma](https://arxiv.org/abs/2407.07726) 预训练 VLM 上加一个约 300M 的 **action expert**，用 **flow matching** 输出连续动作块。它是本库里「连续 flow + 动作专家」这条线的源头；后来的 [π0.5](pi0.5.md) 沿用同一套专家，改的是异构 co-training 和统一的 semantic subtask。[OpenVLA](openvla.md) 是对照的开源离散 token 基线，不是近亲换皮。

## 关键事实

| 字段 | 值 |
| --- | --- |
| **总参数** | 约 3.3B（PaliGemma ~3B + action expert ~300M；§IV） |
| **VLM backbone** | PaliGemma（Figure 3：SigLIP 400M + Gemma 2.6B；Appendix B 给 Gemma 2B 配置 width=2048 / depth=18 / MQA） |
| **Action expert** | ~300M；width=1024，mlp dim=4096；只处理本体状态与噪声动作 |
| **模态** | 多模态（**多路 RGB 图像** + **语言指令** + **本体感觉** \(q_t\) → **连续动作块**）。**无**音频 |
| **动作表示** | flow matching 连续 chunk，\(H=50\)；推理 10 步 Euler；开环执行 |
| **控制频率** | 最高 50 Hz（UR5e / Franka 为 20 Hz） |
| **训练数据** | 自有约 10,000 小时 / 903M timesteps，7 配置 × 68 任务；混合物另含 9.1% OXE Magic Soup / Bridge / DROID |
| **预训练** | 700k 步；动作维零填充到 18 |
| **来源** | [π0 论文](../sources/pi0.md)（arXiv:2410.24164v4，RSS 2025） |

## 技术身份

把「预训练 VLM + 动作头」从离散 next-token 换成连续生成：

1. **动作不是词表里的 bin**。OpenVLA 把 7 维控制量化成 256-bin、自回归吐 token；π0 用条件 flow matching 一次吐出 50 步连续动作。原文在同一混合物上重训 OpenVLA，将其失败归因于「不支持 action chunking 或高频控制」（§VI-A）——这是原文对照，不是后文推断。
2. **Action expert 与 VLM 分权**。图像和语言走 PaliGemma；\(q_t\) 与噪声动作走较小的第二套权重，只在 attention 里汇合（Transfusion + MoE 式设计，§IV）。
3. **跨本体 pad / mask**。不同机器人相机数和动作维不同，用固定 18 维动作和最多三路图像，缺的维零填、缺的图 mask。
4. **高层语言是外挂**。复杂任务可以另接一个 VLM 产出中间指令（§V-B）；π0 自己不做 subtask 预测。统一高低层是 [π0.5](pi0.5.md) 才加的。

**不要误读的边界**：out-of-box 是对**预训练见过的任务**直接 prompt，不是未见厨房的开世界。开世界家务是 π0.5 的设定。评测里的「multimodal」是相机+语言+关节角，不是语音。

## 相关页面

- 来源：[π0 论文](../sources/pi0.md)
- 开世界后作：[π0.5](pi0.5.md)
- 离散 token 基线：[OpenVLA](openvla.md)
- 概念：[Vision-Language-Action](../concepts/vision-language-action.md)
- 真机表里对照 π0.5 的后续模型：[InternVLA-A1.5](internvla-a1.5.md)
