---
type: Source
title: "Gemma 2 技术报告里的蒸馏"
description: "Google DeepMind（arXiv:2408.00118v3，2024-10-02，21 页）：2B/9B 预训练是语料上下文上的 teacher 软标签，不是学生 rollout。后训练 SFT 段另写了一句，在学生分布上向 teacher 蒸馏，并同时引用 GKD 与 MiniLLM。散度方向和 on-policy 比例都没有写。"
tags: ["source", "gemma-2", "distillation"]
timestamp: 2026-10-11
resource: "../../raw/2408.00118v3.pdf"
---

# Gemma 2 技术报告里的蒸馏

## 来源

- 原始 PDF：[`raw/2408.00118v3.pdf`](../../raw/2408.00118v3.pdf)
- 标题：Gemma 2: Improving Open Language Models at a Practical Size
- 版本 / 日期：arXiv:2408.00118v3，2024-10-02。首页另印 2024-06-27
- 作者：Gemma Team, Google DeepMind
- 模型链接：**未建模型页**。本页只核对蒸馏是不是学生 rollout 上的 OPD

## 核心结论

1. **预训练蒸馏不是学生 rollout。** `§3.2`：用大模型当 teacher，在每个 token \(x\) 及其上下文 \(x_c\) 上最小化 \(-\sum_x P_T(x\mid x_c)\log P_S(x\mid x_c)\)。上下文来自预训练语料。2B 与 9B 用这套蒸馏代替 next-token；27B 从零训练，不走这套蒸馏。token 量是 27B 为 13T、9B 为 8T、2B 为 2T（`§3.1`）。模型不是多模态。
2. **后训练有一句学生分布上的蒸馏，协议没有展开。** `§4` Supervised fine-tuning：行为克隆用的回复主要是更大的 teacher 生成的。同一段接着写 “We also run distillation from the teacher on the student’s distribution”，引用 Agarwal et al. 2024 与 Gu et al. 2024，也就是 [GKD](generalized-knowledge-distillation.md) 和 [MiniLLM](minillm.md)。报告没有写散度方向、学生自生成的比例、或单独的 OPD 消融表。
3. **综述 Table 2 的 “KD in pre-training” 对得上 `§3.2`，对不上后训练那一句。** 引言把 Gemma 2 和 Qwen3、DeepSeek-V4、MiMo 并列成「采用 OPD」，超出了 Table 2 自己的格子。能从技术报告确证的是：预训练是语料上的软标签；后训练声明做了学生分布上的蒸馏，但没有学生 rollout 的曲线或配方。

Table 6：2B、500B token，从 7B teacher 蒸馏，三个基准平均 67.7，对照从零训练的 60.3。这是预训练蒸馏消融，teacher 选 7B 是为了贴近 27B→9B 的尺寸比，不是后训练 OPD。

## 架构与训练

和蒸馏主张直接相关的结构事实：局部–全局注意力交错，GQA，logit soft-cap。词表是与 Gemma 1 / Gemini 相同的 SentencePiece，256k。2B 用 TPUv5e 512 芯片，9B 用 TPUv4 4096 芯片，27B 用 TPUv5p 6144 芯片（Table 3）。更细的层号不在本页范围。

## 后训练

顺序是 SFT、再 RLHF、再把各阶段模型平均。SFT 数据相对 Gemma 1.1 加了内部和公开数据，其中 LMSYS-chat-1M 只用 prompt、不用原回答。RLHF 的奖励模型比策略大约大一个数量级，更偏向多轮对话。学生分布上的蒸馏写在 SFT 小节里，没有单独的阶段名。

## 评测要点

本页不复述 MMLU 总表。和蒸馏直接相关的是 Table 6 的从零对比，以及 Table 7：teacher 固定 7B 时，200M / 400M / 1B 的验证困惑度在蒸馏后低于从零训练。这些数字都是预训练设定。

## 证据边界与阅读提示

- 同时引用 GKD 和 MiniLLM，不能推出后训练用的是 forward KL 还是 reverse KL。两篇的散度选择并不相同。
- 27B 的预训练不是蒸馏。不要把「Gemma 2 用了 KD」写成三档全用。
- [OPD 综述](opd-survey.md) 的引言名单不能代替本节两句的区分。

## 待追问

- **原文没标方向**：后训练那一句没有散度方向，也没有 on-policy 比例。只凭引用 GKD 与 MiniLLM，不能判断它更接近哪一篇的目标。

## 相关页面

- [OPD 综述](opd-survey.md)
- [GKD](generalized-knowledge-distillation.md)
- [MiniLLM](minillm.md)
- [On-Policy Distillation 跨报告对比](../comparisons/on-policy-distillation.md)
