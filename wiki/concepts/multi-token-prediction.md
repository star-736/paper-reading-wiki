---
type: Concept
title: "多 Token 预测"
description: "MTP 作为训练目标和 speculative decoding 机制。"
tags: ["concept", "multi-token-prediction"]
timestamp: 2026-06-06
---

# 多 Token 预测

## 定义

Multi-token prediction（MTP）让模型训练或配备用于预测多个未来 token 的模块。在这些报告中，MTP 同时是训练信号、draft model、推理加速路径和 RL rollout 加速器。

## 为什么对 agent 有价值

普通推理服务通常追求整体吞吐；agentic workload 更受长尾延迟影响。一次 agent 任务可能包含多轮工具调用、长前缀 prefill、少 batch decode 和长输出。如果某个 rollout 特别慢，整个 RL step 或用户任务都会被拖慢。MTP 的价值在于提高每次 decode step 产出的 token 数，特别适合 small-batch、long-tail 的解码场景。

## 三篇报告中的用法

| 模型 | MTP 角色 | 细节 |
| --- | --- | --- |
| [GLM-5](../models/glm-5.md) | 训练 + speculative decoding + RL rollout 加速 | 使用参数共享的 MTP 层，报告称在私有 prompt 集上 acceptance length 为 2.76。 |
| [GLM-5V-Turbo](../models/glm-5v-turbo.md) | 多模态 MTP（MMTP） | 扩展到图像 token：用共享 learnable `<\|image\|>` token 替代视觉 embedding 传入 MTP head，避免跨 pipeline-parallel 阶段传播视觉 embedding。0.5B 消融显示比直接传视觉 embedding 训练 loss 更低、收敛更稳（MTP head 轻量，难以吸收与文本 embedding 分布差异大的视觉表示）。 |
| [MiMo-V2-Flash](../models/mimo-v2-flash.md) | 明确作为部署加速模块 | 每个 MTP block 0.33B；3-layer MTP 在 16K 输入 / 1K 输出测试中约 1.86-2.70x 加速。 |
| [DeepSeek-V4](../models/deepseek-v4.md) | 继承 DeepSeek 系列设计 + 生产端已被 DSpark 替换 | Flash 和 Pro 的报告原文 MTP depth=1（即沿用 V3/V3.2 single-token MTP）；但 V4 preview 上线**两周后**，生产 serving 引擎里 MTP-1 已被 **[DSpark](../sources/dspark.md)**（semi-AR drafter + confidence-scheduled verification）替换，per-user 生成速度 +60–85%。MTP-1 之所以一直没扩到 MTP-3/5，是因为静态多 token drafter 在高并发下吞吐严格下降--这正是 DSpark hardware-aware scheduler 解决的问题。 |
| [MiniMax-M2 Series](../models/minimax-m2-series.md) | 预训练信号 + speculative decoding + Forge rollout 加速 | 预训练先用单 MTP module，继续预训练阶段通过权重复制扩展到 3 个 MTP modules。 |
| [Gemma 4](../models/gemma-4.md) | Speculative decoding drafter | 4 层小 Transformer drafter，通过 **cross-attention 复用主模型 KV cache**（而非复制 KV），无需 MTP prefill，支持任意 draft 长度。E2B/E4B 用 top-k on token clusters 把最终投影从 d×262k 降到 d×4k。dim 256（小模型）/ 1024（大模型）。 |
| [HunyuanOCR-1.5](../models/hunyuan-ocr-1.5.md) | 推测解码（block-diffusion drafter） | DFlash：~90.7M / 5 层 draft model（从 target 最后 5 层初始化），block size B=16，用 joint FlexAttention block-diagonal mask 一次 forward 训练 K=16 个 draft block。Transformers 6.37× / vLLM 2.14× 加速。输出越长加速越明显（表格 > 公式 > 文本），因结构化 OCR 输出局部规律性强、draft 接受率高。 |
| [GLM-OCR](../models/glm-ocr.md) | 训练 + 推理共用（共享参数多头） | k 个共享参数辅助头预测未来 k token，训练 10 tokens/step，推理平均 5.2 tokens/step，~50% 吞吐提升。与 GLM-5 共享参数思路一致（引用 [GLM-5](../sources/glm-5.md)）。PDF 吞吐 1.86 pages/s 约为 MinerU2.5 的 3.9×。MTP 还带来结构化输出质量收益——鼓励模型向前规划，产出更少「破损」表格标签。 |
| [Ling-2.6](../models/ling-2.6.md) | 继续训练 MTP + 参数共享 | post-training 阶段引入两个额外 MTP 层继续训练。MTP-3-share（参数共享 + 仅第一层梯度回传 base model）accept length 从 MTP-1 的 2.71 提升到 3.31。发现仅第一层 MTP 预测所有后续 token 也有改善，说明新引入的 MTP 层训练不足，参数共享 + 梯度隔离是有效的补偿。配合 linghe fused-kernel，FP8 BS=1 下 MTP+linghe 比 baseline +119%。 |

## MiMo 的经验

MiMo-V2-Flash 对 MTP 讲得最系统。它把 MTP block 做得很轻：SWA + dense FFN，而不是 MoE + global attention。这样 MTP 作为 draft model 时不会本身成为瓶颈。报告还指出 acceptance length 与 next-token cross-entropy 强相关：低熵任务如 WebDev 更容易连续接受多个 draft token，高熵任务如 MMLU-Pro 更容易发生预测分歧。

## GLM 的经验

GLM-5 关注 MTP 的内存成本。朴素训练多个 MTP 层会让参数和 KV cache 随 speculative steps 线性增长。GLM-5 的做法是共享 3 个 MTP 层参数，使 draft-model 内存成本接近单层方案，同时减少训练-推理不一致带来的 acceptance rate 损失。

## MiniMax 的经验

MiniMax-M2 把 MTP 放进 agent RL 系统。三层 MTP modules 不只是离线推理加速器，还在 Forge RL 中与 policy 持续 co-training，使 draft path 跟上非平稳 policy 更新。否则 RL 过程中模型分布不断变化，speculative decoding 的接受率会下降，rollout 加速收益也会衰减。

MiniMax 的另一个要点是初始化方式：从主模型复制权重扩展 MTP modules，而不是随机初始化。这样可以减少扩展阶段对主模型表示的扰动，并让 MTP modules 更快收敛。

## 当 MTP-1 不够：DSpark 的接管

DeepSeek 系列在 V3 / V3.2 / V4 的报告原文里一直停留在 MTP-1（single-token MTP），而不是扩到 MTP-3/5 这种更激进的多 token draft。[DSpark 论文](../sources/dspark.md) § 5.4 给出了**生产侧的诚实原因**：静态多 token drafter 在高并发下会**严格降低 aggregate throughput**，因为长 draft block 里靠后的低置信 token 会无差别占用 target 验证 batch 的容量。DSpark 把"该验多长"做成动态的全局吞吐最大化问题——confidence head 估每个 prefix 存活概率，hardware-aware scheduler 按 profile 出的 SPS(B) 曲线在线截断——于是大 draft block 在轻负载下能展开吃干算力、在高负载下又会自动压缩，避免 MTP-3/5 的吞吐塌方。V4 preview 上线两周后，生产 serving 引擎里 MTP-1 已被 DSpark 整体替换。

这把 MTP 在 DeepSeek 系里的角色推到下一阶段：从"训练目标 + 静态 draft" 进入"训练目标 + dynamic-block draft + per-request scheduling"，draft model 不再是固定深度的并行预测头，而是 parallel backbone + 轻量 sequential head + 校准过的 confidence head 三件套。

## Gemma 4 的经验

Gemma 4 的 MTP drafter 设计与 GLM-5 / MiMo 的关键差异在于 **cross-attention 复用主模型 KV**：drafter 不复制或重新计算主模型的 context representation，而是通过 4 层小 Transformer 的 cross-attention 直接访问主模型已计算的 KV cache。这消除了 MTP prefill 阶段（传统 MTP 需要先对 prompt 做一次 prefill 才能开始 draft），并支持任意 draft 长度。E2B/E4B 的 top-k on token clusters 优化解决了 262k 大词表下最终投影的瓶颈——把 d×262k 降到 d×4k 而不损失 acceptance rate，这对共享 Gemini tokenizer 的大词表模型尤其重要。

## GLM-OCR 的经验：MTP 在确定性 OCR 任务下的双重收益

[GLM-OCR](../sources/glm-ocr.md) 把 MTP 用到 OCR 域，揭示了一个其他报告没强调的点：**MTP 不只是加速器，对确定性结构化输出任务还是质量提升手段**。OCR 是强局部依赖 + 显式结构监督的确定性任务（vs 数学推理 / agentic 的高熵），标准自回归逐 token 解码在此本就低效。GLM-OCR 训练预测 10 tokens/step、推理平均 5.2 tokens/step，~50% 吞吐提升。但更关键的是 MTP 鼓励模型「向前规划」——结构化 token（表格标签、Markdown 语法）有强局部依赖，多 token 预测让模型在生成开标签时就考虑闭标签，产出更少「破损」标签、更鲁棒的结构化输出。

这与 [HunyuanOCR-1.5](../sources/hunyuan-ocr-1.5.md) DFlash 的观察呼应（结构化 OCR 输出局部规律性强、draft 接受率高，表格加速 > 公式 > 文本），但 GLM-OCR 进一步把 MTP 做进训练目标（DFlash 是训练独立 draft model + 推理验证）。两者共同说明：**OCR/结构化输出是 MTP 的甜区**——低熵 + 强局部依赖让多 token 预测既准又快。这也是 GLM-OCR 0.9B 在吞吐 1.86 pages/s（约为 MinerU2.5 的 3.9×）的同时仍能拿 OmniDocBench v1.5 SOTA 的原因。

## 观察点

- MTP 的加速不是固定常数；它依赖任务熵、batch size、kernel 效率和 acceptance length。
- MTP 对 RL rollout 的收益可能大于普通聊天，因为 RL 中 small-batch decode 和长尾样本更常见。
- MTP 与 PD disaggregation、KV-cache reuse、FP8/FP4 推理共同组成 agent serving 的效率栈。
- **静态多 token MTP 在高并发下会反噬吞吐**——这是 V3/V3.2/V4 一直只敢部署 MTP-1 的物理原因，也是 DSpark 把验证长度做成 hardware-aware 动态截断的动机。
