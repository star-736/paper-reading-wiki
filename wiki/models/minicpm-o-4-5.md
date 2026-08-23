---
type: Model
title: "MiniCPM-o 4.5"
description: "OpenBMB 9B 全双工全模态交互模型，Omni-Flow 框架沿共享时间轴对齐多模态输入输出流，端侧 < 12GB RAM。"
tags: ["model", "minicpm", "omni-modal", "full-duplex", "streaming", "edge-deployment"]
timestamp: 2026-07-11
---

# MiniCPM-o 4.5

## 身份

OpenBMB（清华 NLP 实验室）于 2026-04-30 发布的 9B 全模态交互模型，定位为**首个全双工全模态 LLM**。能在实时场景中同时看、听、说，并基于持续理解发出主动行为（提醒/评论）。同时兼容传统回合制模式。LLM backbone 为 Qwen3-8B。

## 关键事实

| 属性 | 值 |
|------|-----|
| **总参数** | ~9B（含编码器 + LLM + 语音解码器，全部可学习参数端到端可微） |
| **LLM backbone** | Qwen3-8B |
| **视觉编码器** | 继承 MiniCPM-V 4.5（LLaVA-UHD + SigLIP） |
| **音频编码器** | Whisper Medium（0.3B），chunk-based 流式，50 token/s → 5× 压缩 → 10 audio token/s |
| **语音 token 解码器** | ~0.3B Llama 架构，生成 S3 语音 token |
| **波形合成** | streaming flow-matching decoder |
| **模态** | 多模态（文本 + 图像 + 视频 + 音频输入；文本 + 音频输出） |
| **全双工** | 是（Omni-Flow 框架，1.0s chunk，LS 控制 + explicit boundary） |
| **端侧部署** | INT4 < 12GB RAM；llama.cpp-omni RTX 4090 RTF 0.21 |
| **语音克隆** | 支持（multimodal system prompt 含参考音频） |
| **来源** | [技术报告](../sources/minicpm-o-4-5.md)（arXiv:2604.27393v1） |

## 技术身份

MiniCPM-o 4.5 的核心创新不在单一模态能力，而在**交互范式**：从回合制（perception → response 交替）转向全双工（感知与生成在 token-level 时间上持续耦合）。这通过 Omni-Flow 框架实现——时分复用式的时间窗口切分，每窗口内模型同时处理新感知 token 和生成输出 token。

架构设计的关键决策是 **LLM 只生成文本域 token**（3-4 step/s = 人类语速），语音 token 生成委托给轻量 speech decoder。这避免了 LLM 直接生成语音 token（~25 token/s）带来的效率瓶颈和语言能力退化。文本与语音 token 以 TAIL（Time-Aligned Interleaving）策略时间对齐交错，考虑累积播放进度做自适应调整。

训练采用四阶段渐进流水线：Speech Pretraining（冻结基座，仅训新模块）→ Joint Pretraining（全解冻联合训练）→ Joint SFT（指令微调）→ RL（GRPO + smooth length reward + RLAIF-V）。基于 MiniCPM-V 4.5 预训练 checkpoint。

## 相关页面

- 来源：[MiniCPM-o 4.5 技术报告](../sources/minicpm-o-4-5.md)
- [Any-to-any 多模态 serving](../concepts/any-to-any-multimodal-serving.md) - 端到端架构 + llama.cpp-omni 是 any-to-any serving 的端侧实例
- [多模态 Agentic 训练](../concepts/multimodal-agentic-training.md) - 渐进式多模态融合策略
- [MOSS-VL](moss-vl.md) - 视觉实时交互的另一条路线：以独立 gated cross-attention KV 通道让新帧可在文本生成期间追加；MOSS 的 L5 量化 benchmark 仍缺失
- [Qwen3](../models/qwen3.md) - LLM backbone
- [Qwen3.5-Omni](../models/qwen3.5.md) - 同类全模态模型，走 Hybrid MoE + GDN 路线；MiniCPM-o 4.5 以 9B dense 走端侧效率路线
