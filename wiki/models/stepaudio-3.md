---
type: Model
title: "StepAudio 3"
description: "StepFun 音频语言家族。Realtime 与 ASR Max 共享预训练和 mid-training，SFT 分叉。音频与文本进；Realtime 出流式语音并可调工具，ASR Max 出转写。MoE，规模未披露。"
tags: ["model", "stepaudio", "full-duplex", "audio", "asr"]
timestamp: 2026-10-03
---

# StepAudio 3

## 身份

StepFun-Audio Team 在 StepAudio 3 Realtime 技术报告（arXiv:2609.14005v2，2026-09-19）里发布的音频语言家族。对外是两支检查点：StepAudio 3 Realtime 负责听、说、想和工具；StepAudio 3 ASR Max 负责转写。两者共享预训练与 mid-training，只在监督微调处分叉。

## 关键事实

| 属性 | 值 |
| --- | --- |
| **总参数 / 激活参数** | 未披露。正文只写语言解码器是 MoE |
| **音频编码器** | Qwen3-Omni 的 Audio Transformer（AuT，arXiv:2509.17765）+ adapter。层数与帧率未在本报告给出 |
| **语言解码器** | MoE。注意力类型、专家数、基座名称未披露 |
| **语音输出** | Realtime：Generator 接在 LLM decoder 之后，流式回到模型音频流。codec 与采样率未披露 |
| **模态** | 多模态里的音频–文本：用户音频、模型自身音频与文本进。Realtime 出流式语音，并可发起工具调用。ASR Max 出规范化转写。无图像或视频输入 |
| **上下文** | 预训练固定 32K，三阶段合计 1.2T token；mid-training 扩到 128K |
| **全双工时间单位** | 每 320 ms 音频块后跟一个状态或文本 token |
| **两支关系** | 预训练与 mid-training 相同；SFT 分叉。ASR 的 SFT 冻结音频编码器 |
| **教师合成** | 四名同基座教师，参数权重 3:1:1:1。推理时无额外路由 |
| **来源** | [技术报告](../sources/stepaudio-3-realtime.md)（arXiv:2609.14005v2） |

## 技术身份

Realtime 把听、地板、推理和工具放进同一个对话状态。用户流和模型流一起进编码器，所以重叠说话时模型看得到自己正在播什么。需要深想时，同一次权重被并发叫两次：一次写私下推理，一次按已经说出的内容往下接。默认先开口，不等推理写完。MTP 只加速私下思考；说出口的 token 仍按目标模型严格验证。

ASR Max 不是这套交互系统的转写模式。它在 SFT 里改去产规范化文本，评测数字（LibriSpeech clean 1.18、AISHELL-1 0.49、ContextASR 四子集全最低）不能当成 Realtime 的识别错误率。

和库内其他实时模型的分界：它没有视觉通道，因此不是 [Qwen3.8-Omni-Flash](qwen3.8-omni-flash.md) 那种 Thinker–Talker 全模态，也不是 [MiniCPM-o 4.5](minicpm-o-4-5.md) 的看听同时说。工具执行写在模型的对话环里，没有另附一张可替换的 harness。

## 相关页面

- 来源：[StepAudio 3 Realtime 技术报告](../sources/stepaudio-3-realtime.md)
- 全双工对照：[MiniCPM-o 4.5](minicpm-o-4-5.md)、[Qwen3.8-Omni-Flash](qwen3.8-omni-flash.md)、[MOSS-VL](moss-vl.md)、[JoyAI-VL-Interaction](joyai-vl-interaction.md)
- 概念：[Any-to-any 多模态 serving](../concepts/any-to-any-multimodal-serving.md)、[多模态 Agentic 训练](../concepts/multimodal-agentic-training.md)、[多 token 预测](../concepts/multi-token-prediction.md)、[Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)
