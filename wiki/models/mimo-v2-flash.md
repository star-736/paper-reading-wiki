# MiMo-V2-Flash

## 身份

MiMo-V2-Flash 是 Xiaomi LLM-Core 发布的开放权重 MoE 模型，目标是在较小激活参数预算下服务快速 reasoning 和 agentic 工作负载。

主要来源：[MiMo-V2-Flash 技术报告](../sources/mimo-v2-flash.md)。

## 关键事实

| 项目 | 数值 |
| --- | --- |
| 总参数 | 309B |
| 激活参数 | 15B |
| 模态 | 纯文本（已据报告原文核实；报告中的 "multimodal verifier / vision-based verifier" 是 RL 数据管线里给渲染视频打分的判别器，非模型本体输入） |
| 层数 | 48 total；39 SWA、9 GA |
| 专家数 | 256 total；每 token 激活 8 个 |
| 上下文 | 原生 32K，扩展到 256K |
| 训练 tokens | 27T |
| 注意力 | 5:1 hybrid Sliding Window / Global Attention |
| 后训练 | Multi-Teacher On-Policy Distillation |
| 推理加速 | 3-layer MTP speculative decoding |

## 解释

在首批三篇报告中，MiMo-V2-Flash 是最明显的“效率优先且结构相对简单”的方案。它不追求百万 token 上下文，而是使用固定的 local/global 混合注意力模式，并把重点放在后训练和推理速度上。

它是 [DeepSeek-V4](deepseek-v4.md) 的重要对照：MiMo 使用更容易解释的 SWA/GA 设计；DeepSeek-V4 使用更激进的 compressed sparse attention 和更复杂的 cache 系统。

## 相关页面

- [多 token 预测](../concepts/multi-token-prediction.md)
- [Multi-Teacher On-Policy Distillation](../concepts/multi-teacher-on-policy-distillation.md)
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)
- [MoE 前沿模型扩展](../concepts/moe-frontier-model-scaling.md)
