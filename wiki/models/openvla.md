---
type: Model
title: "OpenVLA"
description: "开源 7B VLA：Prismatic-7B（Llama-2-7B + DINOv2/SigLIP）把 7 维动作写成 256-bin token，在 Open X-Embodiment 约 970k 真实轨迹上全量 fine-tune。"
tags: ["model", "vla", "robotics", "openvla"]
timestamp: 2026-09-05
---

# OpenVLA

## 身份

OpenVLA 是 Stanford 等团队发布的开源 7B **Vision-Language-Action（VLA）** 模型：在 Prismatic-7B VLM（Llama 2 7B + 融合 DINOv2 / SigLIP 视觉编码器）上，用 Open X-Embodiment 约 970k 条真实机器人演示做全量 fine-tune，让语言模型直接预测离散动作 token，再反量化成连续控制。它是本库里的**开源离散动作 token 基线**；后来的 [π0](pi0.md) 改成 flow matching action expert，[π0.5](pi0.5.md) 再加开世界 co-training，[InternVLA-A1.5](internvla-a1.5.md) 走 Mixture-of-Transformers + latent foresight，都不是这一套 256-bin。

权重、PyTorch 训练代码与 HuggingFace 加载脚本随论文开源（来源页给项目页与 GitHub；博客/README 不作原文确证）。

## 关键事实

| 字段 | 值 |
| --- | --- |
| **总参数** | 论文称 7B-parameter VLA；Table 1 全量微调 7,188.1×10⁶ ≈ 7.19B（含视觉塔） |
| **VLM backbone** | Prismatic-7B：Llama 2 7B + 约 600M 融合视觉编码器 + 2 层 MLP projector（§3.1） |
| **视觉编码器** | DINOv2 与 SigLIP 分路编码、通道维拼接（最终模型；LoRA/量化消融用过 SigLIP-only 变体，脚注 4） |
| **模态** | 多模态（第三人称 **图像** + **语言指令** → 离散动作 token → 连续 7D 控制）。**无**音频；默认策略也 **不**吃本体感觉 / 腕部相机 / 观察历史（§6） |
| **动作表示** | 7 维末端控制（\(\Delta x, \Delta\theta, \Delta\mathrm{Grip}\)），每维 256-bin；覆盖 Llama 词表最后 256 个 token；自回归 next-token，推理时 detokenize |
| **训练数据** | Open X-Embodiment 策展子集，约 970k 真实轨迹；单臂、至少一路第三人称相机 |
| **训练规模** | 27 epoch；batch 2048；64×A100 × 14 天 ≈ 21,500 A100-hours；LR 2e-5 |
| **推理** | bfloat16 约 15GB、RTX 4090 约 6 Hz；int4 可压到约 7GB 且 Bridge 子集成功率持平（Table 2） |
| **来源** | [OpenVLA](../sources/openvla.md)（arXiv:2406.09246v3，2024-09-05） |

## 技术身份

把「预训练 VLM + 动作头」落到可复现的开源实现：

1. **动作就是语言 token**。连续 7D 控制按分位数切成 256 个 bin，当普通词来自回归；没有单独的扩散/flow 动作专家。这与 [π0](pi0.md) / [InternVLA-A1.5](internvla-a1.5.md) Stage 2 的 flow-matching 连续 chunk 不是同一条路；InternVLA / π0.5 预训练用的 FAST 离散 token 也只是过渡配方，不是本页的 256-bin。
2. **视觉塔要为控制解冻**。Prismatic 原论文倾向冻结视觉编码器；OpenVLA 发现 VLA 训练必须 fine-tune 视觉塔，否则空间细节不够（§3.4）。融合 DINOv2（空间）+ SigLIP（语义）是选 Prismatic 而不是 LLaVA / IDEFICS-1 的主因。
3. **开源 + 可微调**。相对 RT-2-X：更小（7B vs 55B）、数据更多（970k vs 350k）、提供 LoRA / 量化，消费级 GPU 能做下游适配。语义泛化仍弱于做了互联网 co-training 的封闭模型（§5.1）。

**不要误读的边界**：评测里的「multimodal / language grounding」指第三人称图像 + 语言指令，不是音频。Google robot / WidowX 是操作机械臂。LIBERO 数字（Table 12）是附录里对仿真套件做 LoRA 微调，不是 zero-shot，也不能直接和 [InternVLA-A1.5](internvla-a1.5.md) 后来报的 LIBERO 98.9 比——数据清洗、是否多套件联合训练、动作头都不同。

## 相关页面

- 来源：[OpenVLA](../sources/openvla.md)
- 概念：[Vision-Language-Action](../concepts/vision-language-action.md)
- 连续 flow + action expert：[π0](pi0.md)
- 开世界 co-training：[π0.5](pi0.5.md)
- 后续实例（MoT + 连续动作，不是近亲重复）：[InternVLA-A1.5](internvla-a1.5.md)
