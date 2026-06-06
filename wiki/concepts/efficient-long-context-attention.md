# 高效长上下文注意力

## 问题

长上下文模型的瓶颈不是单一的“能不能放下更多 token”，而是 attention FLOPs、KV-cache、prefill latency、decode latency、cache reuse 和训练稳定性共同构成的系统问题。首批三篇报告都在处理这个瓶颈，但路线不同。

## 三条路线

| 路线 | 使用模型 | 核心思想 | 主要收益 | 主要风险 |
| --- | --- | --- | --- | --- |
| 内容稀疏 | [GLM-5](../models/glm-5.md) | DSA 根据内容选择重要 token。 | 长程信息访问更自适应。 | indexer 的 top-k 稳定性会影响 RL。 |
| 模式稀疏 | [MiMo-V2-Flash](../models/mimo-v2-flash.md) | 5 个 SWA 层配 1 个 GA 层。 | 架构简单，KV 和 attention 成本下降。 | 对需要任意长程交互的任务可能不如内容自适应。 |
| 压缩注意力 | [DeepSeek-V4](../models/deepseek-v4.md) | CSA/HCA 先压缩 KV，再做稀疏或密集注意力。 | 支持 1M context，KV-cache 极大降低。 | 架构、kernel、cache 管理复杂。 |

## 机制层理解

DSA 不是固定窗口，而是先用 indexer 为 query 找出重要的历史 KV entries，再只对这些 entries 做注意力。它的直觉是：长上下文里绝大多数历史 token 对当前 token 并不重要，因此密集注意力浪费计算。GLM-5 报告中，DSA 从 dense base model 继续训练而来，目标是在不从头训练的情况下把长上下文成本降下来。

MiMo-V2-Flash 的 hybrid SWA/GA 更像工程上保守的折中。SWA 限制局部窗口，GA 周期性提供全局通路。5:1 比例和 128-token window 让大多数层成本较低，同时避免纯局部模型完全失去远程通信能力。

DeepSeek-V4 的 CSA/HCA 更激进。CSA 每 `m` 个 token 压缩成一个 KV entry，然后再做 DSA 式 top-k selection；HCA 用更大的 `m'` 做重压缩，并在压缩状态上做 attention。它还额外保留 sliding-window branch，弥补压缩块内部局部细节不足。

## 对比判断

如果目标是相对简单、可部署、可解释的 256K 级长上下文，MiMo 的 SWA/GA 路线很有吸引力。如果目标是百万 token 和共享前缀复用，DeepSeek-V4 的压缩 + 异构 KV-cache 更接近完整系统方案。GLM-5 的 DSA 更适合理解“内容自适应稀疏”在 agentic RL 中的价值和稳定性风险。

## 进一步阅读

- [DeepSeek Sparse Attention](deepseek-sparse-attention.md)
- [百万 token 上下文服务](million-token-context-serving.md)
- [2026 开放模型技术报告对比](../comparisons/2026-open-model-technical-reports.md)

