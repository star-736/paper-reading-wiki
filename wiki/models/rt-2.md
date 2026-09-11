---
type: Model
title: "RT-2"
description: "Google DeepMind 的封闭 VLA 家族：PaLI-X 5B/55B 与 PaLM-E 12B，把 8 维控制写成 256-bin text token，与互联网 VQA 共微调；VLA 一词的出处。"
tags: ["model", "vla", "robotics", "rt-2"]
timestamp: 2026-09-12
---

# RT-2

## 身份

RT-2（Robotics Transformer 2）是 Google DeepMind 的封闭 **Vision-Language-Action（VLA）** 家族：在已经预训练的 PaLI-X / PaLM-E 上，把机器人动作写成和语言一样的 token，与互联网视觉语言任务 **co-fine-tune**，推理时 de-tokenize 成末端控制。论文用它给 VLA 这个类别命名（来源页摘要、§1）。本库里的 [OpenVLA](openvla.md) 是同一套离散 token 配方的开源 7B 复现；[π0](pi0.md) 把动作头换成连续 flow，不再走这条词表。

不要和 **RT-2-X** 混：那是后续 Open X-Embodiment 在 OXE 上训的 55B，[OpenVLA](openvla.md) 的对照对象；本页评测数据是 RT-1 厨房演示。

## 关键事实

| 字段 | 值 |
| --- | --- |
| **家族** | RT-2-PaLI-X-5B / 55B、RT-2-PaLM-E-12B；Language-Table 另有 RT-2-PaLI-3B |
| **总参数** | 主实验最大 55B；另有 5B 与 12B（§3.1、§4） |
| **VLM backbone** | PaLI-X：ViT-22B + 32B encoder-decoder；PaLM-E-12B：decoder-only LLM + ViT-4B（Appendix D） |
| **模态** | 多模态（机器人 **相机图像** + **语言指令** → 离散动作 token → 连续末端控制）。**无**音频；正文默认策略也 **不**吃本体感觉 |
| **动作表示** | 6-DoF 位移 + 夹爪 + terminate，共 8 个数，连续维均匀 256-bin；PaLI-X 映射整数 token，PaLM-E 覆盖低频 256 token |
| **训练数据** | 网页：WebLI 滤后约 1B + caption/VQA；机器人：RT-1 数据集（13 台厨房移动操作臂、17 个月）。混合物里机器人约占 50%（PaLI-X）或 66%（PaLM-E） |
| **训练** | next-token / behavior cloning；55B：LR 1e-3、batch 2048、80K step（Appendix E） |
| **推理** | 多 TPU 云服务；55B 1–3 Hz，5B / PaLI-3B 约 5 Hz（§3.3） |
| **来源** | [RT-2](../sources/rt-2.md)（arXiv:2307.15818v1，2023-07-28） |

## 技术身份

把「预训练 VLM 直接当低层策略」落到封闭大模型上：

1. **VLA 的定义实例**。不加新参数，动作与语言共享同一套输出空间；和 CLIPort/MOO 的结构先验、Gato 从零搭通用 agent 都不是同一条路（来源 §2）。
2. **必须 co-fine-tune**。只在机器人数据上 fine-tune 或从零训，未见平均都低于网页+机器人共微调（Table 6：55B co-fine-tune 63 vs fine-tune 52 vs 5B scratch 9）。
3. **网页知识迁的是语义，不是新动作**。未见物体/符号/人物识别上相对 RT-1 约 2–3×；擦拭、按部位抓、新动力学仍失败（§5、Appendix G）。

**不要误读的边界**：评测里的「multimodal / reasoning」是第三人称图 + 自然语言，不是语音。主表是厨房移动臂上的原版 RT-2，不是 OpenVLA 表里的 RT-2-X。Language-Table 的 3B 是另一套二维仿真/真机推物，不能和 55B 厨房表横比。

## 相关页面

- 来源：[RT-2](../sources/rt-2.md)
- 概念：[Vision-Language-Action](../concepts/vision-language-action.md)
- 分层 LLM planner 对照：[SayCan](../sources/saycan.md)
- 开源离散 token：[OpenVLA](openvla.md)
- 连续 flow：[π0](pi0.md)
- 开世界 co-training：[π0.5](pi0.5.md)
- 后续 MoT + flow：[InternVLA-A1.5](internvla-a1.5.md)
