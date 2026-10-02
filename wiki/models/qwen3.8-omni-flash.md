---
type: Model
title: "Qwen3.8-Omni-Flash"
description: "Qwen 的原生全模态 agent：Thinker–Talker，Thinker 继承 Qwen3.8-Next 稀疏 MoE 并在多模态预训练后启用 QSA；参数量未披露。Realtime 变体是非思考模式的流式语音。"
tags: ["model", "qwen3.8-omni", "omni", "moe", "qsa"]
timestamp: 2026-10-03
---

# Qwen3.8-Omni-Flash

## 身份

Qwen3.8-Omni-Flash 是 Qwen Team 的原生全模态 agent 模型，用来把长周期音视频生产（剪辑、翻译、配乐影像、从视频抽笔记和技能）放进同一个 Thinker–Talker，而不是只做感知或实时对话。一手出处是 [Qwen3.8-Omni 技术报告](../sources/qwen3.8-omni.md)（arXiv:2609.25611v1）。同报告的 Qwen3.8-Omni-Flash-Realtime 是面向低延迟的非思考模式，不是另一个基座。

## 关键事实

| 项 | 取值 |
| --- | --- |
| **模态** | 多模态（已据本报告核实：文本、图像、音频、空间音频、视频输入；文本与语音输出。Realtime 另做流式语音） |
| 类型 | Thinker–Talker。Thinker 是稀疏 MoE 语言模型；Talker 消费 Thinker 的高层表示，用 RVQ + MTP 出语音 token，Code2Wav 合成到 48 kHz |
| 总参 / 激活 | 本报告未披露。不能把 [Qwen3.8-Flash-Next](qwen3.8-flash-next.md) 的 125B/6B 直接写成 Omni 的参数量 |
| 注意力 | 继承 Qwen3.8-Next 的 GDN + 交错注意力，预训练末段切到 QSA。3:1、$K$、$r$、Gated Residual、主机 n-gram 本报告都没有复述 |
| 编码器 | Vision、AuT（6.25 Hz）、Spatial AuT（FOA 复数 STFT）。视觉编码器出处在「Qwen3.8-Next」与「Qwen3.5」两句之间冲突 |
| 上下文 | 预训练全程 256K（四阶段都写 262,144）；后训练之后扩展到 1M，方法未写 |
| Tokenizer | Qwen3.8-Next byte-level BPE，约 250K |
| 后训练 | Thinker：多教师轨迹蒸馏（未写 on-policy KL）→ 跨模态统一 RL。Talker：五阶段，多语言语音点名 MOPD（Ma et al. 2026），最后用 GSPO，奖励来自对齐后的 Thinker |
| 系统 | [Qwen-MM-Plugins](https://github.com/QwenLM/Qwen-MM-Plugins)、[Qwen-Live-Harness](https://github.com/QwenLM/Qwen-Live-Harness) |

## 技术身份

这条线接的是 [Qwen3.8-Flash-Next](qwen3.8-flash-next.md) 的语言骨干，不是 [Qwen3.5](qwen3.5.md) 的 hybrid 栈直接放大。和 [Qwen3.5-Omni](../sources/qwen3.5-omni.md) 共享 Thinker–Talker、AuT、ARIA 和文本时间戳；新增的是空间音频支路、多模态预训练之后重做的 QSA，以及把 agent 执行（按需取证、插件、实时编排）写成这一代的主目标。

QSA 的机制定义仍在架构报告。Omni 改变的是日程：S2 用约 2.5T 多模态 token 把编码器接上，S3 在 dense attention 仍打开时只训 indexer，S4 才启用稀疏。这不等于架构报告里那段 256K 文本 CPT 被原样搬了过来。

Thinker 的第一阶段蒸馏不要和 Talker 的 MOPD 合成一件事。前者是专家轨迹混进一个 student，没有写 KL；后者明确引用另一篇 MOPD 论文，只覆盖多语言语音。Realtime 的 Table 10 是非思考模式，不能拿去和思考模式的音视频主表比「同一模型退步了」。

## 相关页面

- 来源：[Qwen3.8-Omni 技术报告](../sources/qwen3.8-omni.md)
- 骨干与前代：[Qwen3.8-Flash-Next](qwen3.8-flash-next.md)、[Qwen3.8-Next 架构报告](../sources/qwen3.8-next.md)、[Qwen3.5-Omni](../sources/qwen3.5-omni.md)、[Qwen3.5](qwen3.5.md)
- 概念：[线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md)、[多模态 Agentic 训练](../concepts/multimodal-agentic-training.md)、[Any-to-any 多模态 serving](../concepts/any-to-any-multimodal-serving.md)、[Agent harness](../concepts/agent-harness.md)、[Multi-Teacher On-Policy Distillation](../concepts/multi-teacher-on-policy-distillation.md)
- 比较：[稀疏注意力机制对比](../comparisons/sparse-attention-mechanisms.md)、[2026 前沿模型技术报告对比](../comparisons/2026-open-model-technical-reports.md)
- 音频全双工对照：[StepAudio 3](stepaudio-3.md)。AuT 引用的是 Qwen3-Omni 报告，不是本模型写明的编码器配置
