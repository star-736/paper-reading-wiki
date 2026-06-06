# DeepSeek-V4

## 身份

DeepSeek-V4 是 DeepSeek-AI 发布的模型族，核心目标是高效原生 1M token 上下文。当前沉淀的报告覆盖 DeepSeek-V4-Flash 和 DeepSeek-V4-Pro。

主要来源：[DeepSeek-V4 技术报告](../sources/deepseek-v4.md)。

## 关键事实

| 变体 | 总参数 | 激活参数 | 训练 tokens | 上下文目标 |
| --- | ---: | ---: | ---: | --- |
| DeepSeek-V4-Flash | 284B | 13B | 32T | 1M tokens |
| DeepSeek-V4-Pro | 1.6T | 49B | 33T | 1M tokens |

## 技术身份

DeepSeek-V4 的定义性特征是混合压缩注意力：

- CSA 压缩 KV entries 后进行稀疏选择。
- HCA 以更重压缩率在压缩 entries 上做注意力。
- Sliding-window branch 保留局部细节。
- 异构 KV cache 与 on-disk prefix reuse 让服务端推理更可行。

模型还使用 mHC 残差增强、Muon optimizer、DeepSeekMoE、MTP depth 1，以及针对部分路径的 FP4 quantization-aware training。

## 解释

DeepSeek-V4 是当前知识库中最典型的“把上下文长度当成系统问题处理”的案例。它的百万 token 主张依赖架构、优化器、kernel、KV-cache layout 和 serving 策略共同成立。

## 相关页面

- [高效长上下文注意力](../concepts/efficient-long-context-attention.md)
- [百万 token 上下文服务](../concepts/million-token-context-serving.md)
- [MoE 前沿模型扩展](../concepts/moe-frontier-model-scaling.md)
- [2026 开放模型技术报告对比](../comparisons/2026-open-model-technical-reports.md)
