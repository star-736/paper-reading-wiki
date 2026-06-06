# 多 Token 预测

## 定义

Multi-token prediction（MTP）让模型训练或配备用于预测多个未来 token 的模块。在这些报告中，MTP 同时是训练信号、draft model、推理加速路径和 RL rollout 加速器。

## 为什么对 agent 有价值

普通推理服务通常追求整体吞吐；agentic workload 更受长尾延迟影响。一次 agent 任务可能包含多轮工具调用、长前缀 prefill、少 batch decode 和长输出。如果某个 rollout 特别慢，整个 RL step 或用户任务都会被拖慢。MTP 的价值在于提高每次 decode step 产出的 token 数，特别适合 small-batch、long-tail 的解码场景。

## 三篇报告中的用法

| 模型 | MTP 角色 | 细节 |
| --- | --- | --- |
| [GLM-5](../models/glm-5.md) | 训练 + speculative decoding + RL rollout 加速 | 使用参数共享的 MTP 层，报告称在私有 prompt 集上 acceptance length 为 2.76。 |
| [MiMo-V2-Flash](../models/mimo-v2-flash.md) | 明确作为部署加速模块 | 每个 MTP block 0.33B；3-layer MTP 在 16K 输入 / 1K 输出测试中约 1.86-2.70x 加速。 |
| [DeepSeek-V4](../models/deepseek-v4.md) | 继承 DeepSeek 系列设计 | Flash 和 Pro 的 MTP depth 都为 1，更像标准效率组件。 |
| [MiniMax-M2 Series](../models/minimax-m2-series.md) | 预训练信号 + speculative decoding + Forge rollout 加速 | 预训练先用单 MTP module，继续预训练阶段通过权重复制扩展到 3 个 MTP modules。 |

## MiMo 的经验

MiMo-V2-Flash 对 MTP 讲得最系统。它把 MTP block 做得很轻：SWA + dense FFN，而不是 MoE + global attention。这样 MTP 作为 draft model 时不会本身成为瓶颈。报告还指出 acceptance length 与 next-token cross-entropy 强相关：低熵任务如 WebDev 更容易连续接受多个 draft token，高熵任务如 MMLU-Pro 更容易发生预测分歧。

## GLM 的经验

GLM-5 关注 MTP 的内存成本。朴素训练多个 MTP 层会让参数和 KV cache 随 speculative steps 线性增长。GLM-5 的做法是共享 3 个 MTP 层参数，使 draft-model 内存成本接近单层方案，同时减少训练-推理不一致带来的 acceptance rate 损失。

## MiniMax 的经验

MiniMax-M2 把 MTP 放进 agent RL 系统。三层 MTP modules 不只是离线推理加速器，还在 Forge RL 中与 policy 持续 co-training，使 draft path 跟上非平稳 policy 更新。否则 RL 过程中模型分布不断变化，speculative decoding 的接受率会下降，rollout 加速收益也会衰减。

MiniMax 的另一个要点是初始化方式：从主模型复制权重扩展 MTP modules，而不是随机初始化。这样可以减少扩展阶段对主模型表示的扰动，并让 MTP modules 更快收敛。

## 观察点

- MTP 的加速不是固定常数；它依赖任务熵、batch size、kernel 效率和 acceptance length。
- MTP 对 RL rollout 的收益可能大于普通聊天，因为 RL 中 small-batch decode 和长尾样本更常见。
- MTP 与 PD disaggregation、KV-cache reuse、FP8/FP4 推理共同组成 agent serving 的效率栈。
