---
type: Model
title: "StartLux-Decision"
description: "StartLux 的决策模型族（v1.1）：Instant 0.8B–27B 一次前向读出选项概率；AutoThink 4B / 9B / 27B 在最高概率低于论文门槛 0.9 时才生成推理。输入是文本、JSON，以及可选图像；视觉塔冻结。权重 CC BY-NC 4.0。"
tags: ["model", "startlux-decision", "decision-model"]
timestamp: 2026-10-24
---

# StartLux-Decision

## 身份

[StartLux-Decision 技术报告](../sources/startlux-decision.md)（arXiv:2610.17309v2）发布的决策模型族。给定状态和一组带类型的问题，模型为每个选项给出概率：`choice`、二元判断，或 2–10 级的 `score`。Instant 不生成文本。AutoThink 与 Instant 从同一个 Stage 1 出发，Stage 2 起注意力不同，Stage 3 才加上推理监督和 RLCD2。

权重在 [startlux-models](https://huggingface.co/startlux-models)，推理代码在 [StartLuxLabs/StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision)。报告对应的发布名是 v1.1。仓库还服务更早的 v1（含一个 35B-A3B，只做单次读出）；v1 不在这篇 PDF 的结果表里。

## 关键事实

| 项目 | 值 |
| --- | --- |
| 模态 | 文本或 JSON 状态，可选图像（视觉塔冻结，`§2.2`）。输出是选项上的概率；AutoThink 在不确定时另生成推理文本。报告中的分数是文本决策，没有视觉 benchmark |
| 规模 | Instant：0.8B、2B、4B、9B、27B。AutoThink：4B、9B、27B。论文用这些名义参数量，没有另给精确参数计数 |
| 基座 | Figure 2 标成 Qwen3.5、Qwen3.8，未按尺寸点名。v1.1 的 4B 与 27B Instant 配置（外部）为 `model_type: qwen3_5`。4B：32 层、hidden 2560；27B：64 层、hidden 5120。上下文 262,144 |
| 注意力 | Instant 去掉 full-attention 层的因果 mask，线性层保持因果。正文写 4B 与 9B 改的是 32 层中的 8 层。AutoThink 保持因果 |
| 读出 | 无额外预测头。最后一个 prompt 位置上，选项字母 logits 按题型温度做 softmax。一次最多 26 个选项 |
| 推理门 | 论文评测 \(g=0.9\)，事后读出权重 \(\alpha=0.75\)，生成上限 4,096，RL 软预算 3,072。仓库服务默认 gate 为 0.6，与表内协议不同 |
| 许可 | 权重 CC BY-NC 4.0；代码 Apache-2.0 |

## 技术身份说明

这是一组继续训练出来的决策检查点，不是新的预训练基座。Stage 1 全参，Stage 2 用 adapter 再合并；Instant 还把 0.8B 的三次运行或 2B–27B 的四次运行做均匀平均。AutoThink 的概率读出和推理生成共用同一套权重。

已核对的 v1.1 配置与 [Qwen3.5](qwen3.5.md) 的 hybrid 栈同族：`layer_types` 为三层 `linear_attention` 接一层 `full_attention`。论文没有使用 GDN 这个名字。Figure 2 里的 “Qwen3.8” 不能改写成 [Qwen3.8-Flash-Next](qwen3.8-flash-next.md)。

模型卡写每个请求最多 8 张图（外部，例如 [0.8B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant)）。论文只写状态可以带图像、视觉塔冻结。

## 相关页面

- [StartLux-Decision 技术报告](../sources/startlux-decision.md)：三阶段训练、RLCD2、Decision Index 与 DecisionPivot。
- [Qwen3.5](qwen3.5.md)：发布配置对得上的基座家族。
- [LLM RL policy optimization 对比](../comparisons/llm-rl-policy-optimization.md)：RLCD2 在 GRPO 谱系里改的是奖励，不是 ratio 单元。
