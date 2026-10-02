---
type: Model
title: "MiMo-V2.6"
description: "Xiaomi 的全模态 MoE 系列：Pro 1.02T/42B、Flash 310B/15B，以及蒸馏到 Qwen3.5-9B 的 9B。"
tags: ["model", "mimo-v2.6"]
timestamp: 2026-10-03
---

# MiMo-V2.6

## 身份

MiMo-V2.6 是 Xiaomi LLM-Core 的全模态 MoE 系列，用来把大规模 agentic RL 接到可交互环境上。公开档位是 MiMo-V2.6-Pro、MiMo-V2.6-Flash，以及从该系列蒸馏到 Qwen3.5-9B 的 MiMo-V2.6-Distill-Qwen-9B。

主要来源：[MiMo-V2.6 技术报告](../sources/mimo-v2.6.md)。文本骨干的更早描述在 [MiMo-V2-Flash](mimo-v2-flash.md)。

## 关键事实

| 项目 | Pro | Flash | Distill-Qwen-9B |
| --- | --- | --- | --- |
| 总参数 | 1.02T | 310B | Qwen3.5-9B 的 SFT |
| 激活参数 | 42B | 15B | 与底座同量级；报告未单列 |
| 模态 | 文本 + 图像 + 视频 + 音频输入，文本输出（已据 §2 与 Figure 2 核实；音频与视觉是进入骨干的编码器） | 同左 | 训练数据含代码、通用、视觉、网络安全；报告未另写一套模态 |
| 层数 | 70 层：60 SWA、10 GA | 48 层：39 SWA、9 GA | 继承 Qwen3.5-9B |
| 专家 | 384 中激活 8，无 shared expert | 256 中激活 8，无 shared expert | 继承底座 |
| 上下文 | 预训练中途到 256K；mid-training 最后扩到 1M；RL 上下文直到 1M | 同左 | 未单列 |
| 预训练 tokens | 30T（文本 27T + 全模态 3T） | 48T（文本 26T + 全模态 22T） | SFT 77.4B，其中 27.2B 计入损失 |
| 注意力 | 窗口 128 的 hybrid SWA/GA；第一层是 dense FFN 的 GA | 同模式；层数切分与 V2-Flash 相同 | 继承底座 |
| 后训练 | 短 SFT → 一次混合 GRPO → MOPD2（Multi-Prefix） | 同左；RL 花费 $0.9M | 分域 GRPO；代码表是单 harness，多 harness 另跑 |
| 推理加速 | 5 层 SWA 的 DFlash drafter，一次预测 7 token；RL 用 block-6 | 同左，hidden 4096 | 未单列 |

## 解释

Pro 与 Flash 共用 ViT（681M）、音频 tokenizer（308M）和 audio patch encoder（127M），差别在文本骨干宽度、专家数和预训练里全模态 token 的比例。报告用 310B / 1.02T 称呼两档模型；Table 1 把这对数字放在主骨干块，编码器另行列出，没有写是否已经计入。Flash 的 GA 是 64 个 query head、4 个 KV head；Pro 的 SWA 和 GA 都是 128 / 8。

后训练的重心是一次混任务 RL，而不是按域顺序各训一轮。1,568×16 的组、1M 上下文、mini-harness 和 groupwise grading 绑在同一次运行里。router 在 RL 中冻结。MOPD2 放在这次 RL 之后，用整段 rollout 蒸馏可验证域，用历史前缀上的单轮蒸馏难验证域。这个 MOPD2 是 Multi-Prefix 的名字，和 Nemotron 第二轮蒸馏的同名缩写无关。

Distill-Qwen-9B 是开放实验底座：先吃 MiMo 生成的轨迹，再在放出的大约 7k 条环境上分域做 GRPO。它的分数不能写成 Pro 或 Flash 的分数。

## 相关页面

- [MiMo-V2-Flash](mimo-v2-flash.md)
- [Multi-Teacher On-Policy Distillation](../concepts/multi-teacher-on-policy-distillation.md)
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)
- [训练—rollout 一致性](../concepts/train-rollout-consistency.md)
- [MoE 负载均衡谱系](../concepts/moe-load-balancing.md)
- [高效长上下文注意力](../concepts/efficient-long-context-attention.md)
- [多 token 预测](../concepts/multi-token-prediction.md)
- [MoE 前沿模型扩展](../concepts/moe-frontier-model-scaling.md)
- [2026 前沿模型技术报告对比](../comparisons/2026-open-model-technical-reports.md)
