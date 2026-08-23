---
type: Model
title: "MOSS-VL"
description: "OpenMOSS 11.3B 实时视觉语言模型：视觉 token 经 gated cross-attention 独立于解码序列，XRoPE + Realtime-SFT 支持持续看、适时说与自我修订。"
tags: ["model", "moss-vl", "real-time", "streaming", "vision-language"]
timestamp: 2026-08-23
---

# MOSS-VL

## 身份

MOSS-VL 是 OpenMOSS 发布的开源实时视觉语言模型族。它把模型的目标从“看完视频后回答”改为：视频仍在到达时，模型自行决定保持静默、开始说话，或在新证据出现后修订正在进行的回复。论文报告的 0708 训练 run 依次产出 Base、Instruct 与 Realtime 三个 checkpoint；另释出一个独立训练的 0408 Base/Instruct 对。

## 关键事实

| 项 | 取值 |
| --- | --- |
| **模态** | **多模态**（文本 + 图像 + 视频；报告未把音频列为模型输入） |
| 总参数 | 11.3B（BF16） |
| 参数构成 | 约 8.2B language backbone + 2.3B cross-attention stack + 0.8B vision encoder / projection |
| LLM 初始化 | [Qwen3-8B](qwen3.md) |
| 视觉初始化 | [Qwen3-VL](qwen3-vl.md) 的 vision encoder |
| 解码器 | 48 层：36 self-attention + 12 gated cross-attention；每第 4 层插一个后者 |
| Cross-attention | 32 query / 8 KV head GQA + QK-RMSNorm；tanh scalar gates 零初始化 |
| 视觉编码 | 27 层，native dynamic resolution（4,096--16.8M pixels），2×2 patch merge |
| 位置 / 时间 | XRoPE 三轴 `(t,h,w)` + 每帧文本绝对时间戳 |
| 上下文 | 262,144 tokens；训练视频最高 2,048 帧，评测最高 768 帧 |
| 实时控制 | `<|silence|>` / `<|response|>` 两个新增 token；无需独立 decision head |
| 后训练 | 先标准 SFT，后 Realtime-SFT（0.56M samples / 34.8B tokens） |
| 发布 | Base-0708 / Instruct-0708 / Realtime / Base-0408 / Instruct-0408；代码与权重开放 |

## 技术身份

### 实时性的关键不是“更快地重跑 prompt”

传统 interleaved VLM 把大量视觉 token 插进 language decoder 的上下文。MOSS-VL 的选择是让视觉 token 永远留在 decoder 外侧：它们只作为 gated cross-attention 的 key/value，语言序列只记录时间戳与 placeholder。于是新帧的处理是“编码该帧并追加视觉 KV”，而非重编码整段视频或把所有 patch 重塞进自回归序列。

这是一项**原文确证的架构性质**，也是其测得长视觉上下文 TTFT 增长较慢的解释（来源页的 Figure 4）。它不自动证明任何实际产品都有更低交互延迟：端到端体验仍受采样、视觉编码、解码长度、TTS/UI 和调度影响。

### XRoPE：视觉 cross-attention 也需要时空位置

MOSS-VL 不把 RoPE 只留给 text self-attention。XRoPE 让文本 query 与视觉 key 在交叉注意力前共享 `(t,h,w)` 坐标：文本沿三轴同进，帧 patch 共用时间锚点、在空间轴铺开。每帧的绝对时间戳 token 则把“第几个采样位置”与“现实中过了几秒”分开。

### Realtime-SFT：把 timing 做成语言建模

模型每收到一帧，就预测一个普通的下一个 token：若最可能的是 `<|silence|>`，继续看；否则开始回复。训练用逐帧 decision slot 模拟速率受限的输出，使同一个用户指令可出现多次 emission。对两类状态 token 的 focal + inverse-frequency 重加权，是对“永远沉默”这一类别不平衡捷径的直接处理。

| | MOSS-VL | [JoyAI-VL-Interaction](joyai-vl-interaction.md) | [MiniCPM-o 4.5](minicpm-o-4-5.md) |
| --- | --- | --- | --- |
| 主目标 | 生成时仍接收新视觉帧 | 每秒决定说话 / 静默 / 委托 | 全双工全模态交互 |
| 视觉接入 | 追加式 gated cross-attention KV，patch 不进 decoder 序列 | AdaCodec：参考帧 ViT token + 预测帧 P-token | Omni-Flow 的时间对齐交错流 |
| 交互动作 | silence / response；自然语言修订 | silence / response / delegation | 文本与语音的连续 I/O |
| 关键训练 | Realtime-SFT，无 RL | 角色加权 SFT + GRPO | 渐进式联合训练、SFT、RL |
| 评测边界 | L2--L4 有公开量化；L5 仅定性展示 | 真实产品 head-to-head 人工评测 | 全双工 benchmark + 报告的系统评测 |

该表是基于三份来源页的**本页原创综合**；“L5”是 MOSS 论文自己的能力分级，不应不加验证地套到其他模型上。

## 相关页面

- 来源：[MOSS-VL 技术报告](../sources/moss-vl.md)
- 视觉交互对照：[JoyAI-VL-Interaction](joyai-vl-interaction.md)、[MiniCPM-o 4.5](minicpm-o-4-5.md)
- 基座：[Qwen3](qwen3.md)、[Qwen3-VL](qwen3-vl.md)
- 概念：[多模态 Agentic 训练](../concepts/multimodal-agentic-training.md)、[Agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md)
