# MiniMax-M2 Series

## 身份

MiniMax-M2 Series 是 MiniMax 发布的低激活 MoE agentic model 系列。当前报告重点覆盖 M2 backbone，以及经过数据、RL 和 agent scaffold 演进后的 M2.5、M2.7。

主要来源：[MiniMax-M2 Series 技术报告](../sources/minimax-m2-series.md)。

## 关键事实

| 项目 | 数值或描述 |
| --- | --- |
| Backbone | 62-layer decoder-only Transformer |
| 总参数 | 229.9B |
| 激活参数 | 9.8B / token |
| Experts | 256 fine-grained experts，8 active / token |
| Attention | Full multi-head attention with GQA，48 query heads / 8 KV heads |
| Context | 192K native context |
| 预训练 | 29.2T tokens |
| 加速组件 | 3-layer MTP speculative decoding path |

## 技术身份

M2 的核心不是“参数最大”，而是把每 token 激活压到约 10B，同时通过高质量 agent 数据、[Forge agent-native RL](../concepts/forge-agent-native-rl.md) 和系统级推理加速维持长任务能力。

M2.7 是报告中的最新能力 checkpoint。它强调三类任务：agentic coding、agentic cowork 和 reasoning/knowledge。尤其值得关注的是 self-evolution：模型可以在内部 Model Iteration System 中读日志、诊断训练异常、修改 agent scaffold，并承担 30%-50% 的日常迭代工作量。

## 如何解读结果

M2.7 的 benchmark 结果往往依赖统一 scaffold、工具环境和 interleaved thinking。它对研究的启发是：前沿 agent 能力越来越像“模型权重 + 数据环境 + RL 系统 + scaffold + serving stack”的组合产物。

## 相关页面

- [Forge Agent-Native RL](../concepts/forge-agent-native-rl.md)
- [Agentic engineering](../concepts/agentic-engineering.md)
- [多 token 预测](../concepts/multi-token-prediction.md)
- [MoE 前沿模型扩展](../concepts/moe-frontier-model-scaling.md)
- [2026 前沿模型技术报告对比](../comparisons/2026-open-model-technical-reports.md)
