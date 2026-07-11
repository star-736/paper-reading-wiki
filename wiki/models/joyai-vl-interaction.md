---
type: Model
title: "JoyAI-VL-Interaction"
description: "JD.com 的 8B 视觉驱动交互模型，基于 Qwen3-8B + Qwen3-VL ViT + AdaCodec，每秒自主决定说话/静默/委托，配合完整可部署系统。"
tags: ["model", "joyai-vl-interaction"]
timestamp: 2026-07-11
---

# JoyAI-VL-Interaction

## 身份

JoyAI-VL-Interaction 是 JD.com 发布的 **8B 视觉驱动交互模型（interaction model）**，核心特征是：持续观看视频流，每秒自主决定说话、保持静默或委托后台模型，而非等用户提问才响应。论文声称是首个开源的、视觉驱动的交互模型，同时发布训练 recipe、数据和完整可部署系统。

## 关键事实

| 项 | 取值 |
| --- | --- |
| **模态** | **多模态**（文本 + 图像 + 视频；语音经外置 ASR/TTS 转导，模型本身不处理音频） |
| 规模 | 8B（LLM 初始化自 [Qwen3-8B](qwen3.md)） |
| LLM backbone | [Qwen3-8B](qwen3.md)（标准 GQA，非 hybrid） |
| 视觉编码器 | [Qwen3-VL](qwen3-vl.md) ViT |
| 视频编码 | **AdaCodec**（预测式编码：参考帧 256 token / 可预测帧 ~16 P-token，~16× 压缩） |
| 基座 | JoyAI-VL 1.0（三阶段训练：representation alignment -> VL pre-training -> OPD + RL post-train） |
| 交互行为 | 每秒三选一：`</silence>` / `</response>` / `</delegation>` |
| 训练数据 | 4M+ 时间对齐流式片段，六族（告警/QA/计数/评论/闲聊/委托） |
| SFT 目标 | 角色加权 cross-entropy（$w^\text{repeated}_\text{silence}=0.4$，$w^\text{response}=1.5$） |
| RL | GRPO + answer-centered window sampling + stream-level reward（含 LLM judge） |
| RL 框架 | EasyVideoR1 |
| 上下文 | 约 2 小时连续视频（三层记忆：短期 raw token -> 中期文本摘要 -> 长期压缩块） |
| Serving | vLLM-native，prefix-reuse 设计 |
| 系统 | 双并发循环（实时 + 异步），后台 brain 可插拔（Hermes Agent / OpenClaw / 任意 API） |
| 评测 | vs Doubao 胜率 77.6% / vs Gemini 胜率 87.9%（6 场景 58 case 人工盲评） |
| 许可 | 开源（模型 + recipe + 数据 + 系统） |

## 技术身份

JoyAI-VL-Interaction 的定位与知识库中其他多模态模型有本质区别——它不是更强的 turn-based VLM，而是 **interaction model** 范式的实例：

1. **"何时行动"是模型内部能力，不是外部 harness**：turn-based 模型（包括 GPT-Realtime-2、[Qwen3.5-Omni](../sources/qwen3.5-omni.md)）即使做到端到端低延迟，优化目标仍是"用户说完话后多快回答"，交互围绕 dialogue turn 组织。JoyAI-VL-Interaction 的决策是"此刻是否值得说话"，由模型在每秒的 token 生成中内部完成。

2. **vision-first，speech-as-I/O**：与 TML interaction model（把语音 fuse 进模型）不同，JoyAI-VL-Interaction 把视觉触发作为模型原生能力，语音经外部 ASR/TTS 转导。这是一个刻意的解耦：自主核心只管"看+决定"，I/O 由部署方按需替换。

3. **delegate 闭合"看到-做到"循环**：模型遇到超出自身能力的难题时不是硬答，而是通过 background-agnostic text contract 委托给后台模型/agent，结果异步回填。8B 模型因此可按需访问更重的推理能力。

4. **AdaCodec 使无界流式消费可行**：逐帧全量 ViT 编码会让成本和延迟随帧数线性增长；AdaCodec 只在场景变化时花完整 token，可预测帧用 ~16 token 的 P-token，使预算增长与场景变化量而非帧数成正比。

## 与已有模型的关系

| | JoyAI-VL-Interaction | [Qwen3.5-Omni](../sources/qwen3.5-omni.md) | [Kimi K2.5](kimi-k2.5.md) | [Qwen3-VL](qwen3-vl.md) |
| --- | --- | --- | --- | --- |
| 范式 | interaction model | omni-modal turn-based | agentic | turn-based VL |
| 触发 | 视觉事件驱动 | 对话 turn 驱动 | 用户/工具驱动 | 用户提问驱动 |
| 语音 | 外置 ASR/TTS（可插拔） | 模型内（Thinker/Talker/Vocoder） | 不涉及 | 不涉及 |
| 视频流 | 持续流式（AdaCodec） | 有限长（400s 720P） | 不涉及 | 有限长 |
| 何时说话 | 模型每秒内部决策 | turn-taking | 用户/工具触发 | 用户提问 |
| 后台委托 | delegate -> 异步 brain | 无 | Agent Swarm | 无 |

## 相关页面

- [JoyAI-VL-Interaction 技术报告](../sources/joyai-vl-interaction.md) - 来源页（含完整架构/训练/评测细节）
- [多模态 Agentic 训练](../concepts/multimodal-agentic-training.md) - Kimi K2.5 的视觉 agentic 路线
- [Any-to-any 多模态 serving](../concepts/any-to-any-multimodal-serving.md) - vLLM-Omni 的 stage graph serving 范式
- [Qwen3](qwen3.md) - LLM backbone 来源
- [Qwen3-VL](qwen3-vl.md) - 视觉编码器来源
