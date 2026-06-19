# MoE 前沿模型扩展

## 概览

当前沉淀的前沿模型报告大多使用 Mixture-of-Experts（MoE）架构。关键比较不只是总参数，而是每 token 激活参数，以及服务长上下文、agent rollout 和多模态输入所需的系统成本。

## 当前对比

| 模型 | 总参数 | 激活参数 | Expert 设计备注 |
| --- | ---: | ---: | --- |
| [GLM-5](../models/glm-5.md) | 744B | 40B | 256 experts；相比 GLM-4.5 显著放大。 |
| [MiMo-V2-Flash](../models/mimo-v2-flash.md) | 309B | 15B | 256 experts，8 active，无 shared experts。 |
| [DeepSeek-V4-Flash](../models/deepseek-v4.md) | 284B | 13B | 256 routed experts + 1 shared expert；6 routed active。 |
| [DeepSeek-V4-Pro](../models/deepseek-v4.md) | 1.6T | 49B | 384 routed experts + 1 shared expert；6 routed active。 |
| [MiniMax-M2](../models/minimax-m2-series.md) | 229.9B | 9.8B | 256 fine-grained experts，8 active，sigmoid gating。 |
| [MiniMax-M3](../models/minimax-m3.md) | ~428B | ~23B | 128 routed + 1 shared experts，top-4 |
| [Kimi K2.5](../models/kimi-k2.5.md) | 1.04T | 32B | 384 experts，8 active，继承 Kimi K2 MoE backbone。 |

## 解释

这些报告的激活参数大致落在 9.8B 到 49B。设计前沿不再只是“模型更大”，而是如何组合稀疏激活、长上下文注意力、后训练、agent scaffold 和 serving 基础设施。

MiniMax-M2 是当前知识库里激活参数最低的前沿 agentic 案例，它用 Forge、数据和系统优化补足约 10B active 的模型预算。Kimi K2.5 则处在 1T total / 32B active 层级，并把视觉编码器和 Agent Swarm 纳入整体能力。

MiMo-V2-Flash 与 DeepSeek-V4-Flash 的激活参数预算接近，但注意力策略完全不同。GLM-5、Kimi K2.5 和 DeepSeek-V4-Pro 则处在更大的 frontier model 层级。

## 观察点

- 激活参数数值不能单独决定长上下文推理成本。
- 总专家数增加后，expert routing 稳定性与通信开销更重要。
- Quantization 和 KV-cache 设计可能主导实际 serving cost。
- 对 agentic model，rollout 调度、MTP 接受率和工具等待时间也会改变“有效成本”。
