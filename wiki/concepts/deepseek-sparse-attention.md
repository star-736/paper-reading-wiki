# DeepSeek Sparse Attention

## 一句话定义

DeepSeek Sparse Attention（DSA）是一种内容自适应的长上下文稀疏注意力机制：它不固定看最近窗口，也不对所有历史 token 做 dense attention，而是为当前 query 动态选出最相关的一批历史 KV entries。

## 解决的问题

长上下文 dense attention 的成本随序列长度快速增长。对 agent 来说，这个问题更严重：工具调用历史、代码上下文、搜索结果和多轮推理会把上下文推到几十万 token。很多历史 token 对当前动作并不重要，但 dense attention 仍然会为它们付费。

DSA 的核心假设是：长上下文中存在大量注意力冗余，可以通过内容相关性选择保留关键 token。

## GLM-5 中的 DSA

[GLM-5](../models/glm-5.md) 把 DSA 用作长上下文效率组件。报告中 DSA 从 mid-training 结束后的 dense base model 继续训练而来，而不是从头训练。流程包括：

1. DSA warmup：用 202,752-token 序列训练 indexer。
2. Sparse adaptation：继续使用 mid-training 数据和超参数训练 20B tokens。
3. 对比 MLA dense baseline，长上下文 benchmark 表现接近，SFT loss 也基本打平。

报告给出的直观结论是：在 128K 级长上下文里，约 90% attention entries 是冗余的；DSA 能把长序列 attention computation 降低约 1.5-2 倍。

## RL 稳定性问题

GLM-5 的重要细节是：DSA 在 RL 中不只是推理优化，还会影响训练稳定性。DSA indexer 每个 token 都要选 top-k KV entries。如果 top-k 算子非确定性，训练引擎和推理引擎看到的 token selection 可能不一致，导致 RL 几步后 entropy 急剧下降。

GLM-5 的处理方式是：

- 训练中使用 deterministic `torch.topk`，即使它稍慢。
- 默认冻结 indexer 参数，减少不稳定学习。
- 避免像 MoE routing replay 那样保存每个 token 的 top-k index，因为 DSA 的 k 可达 2048，存储和通信成本过高。

## 与 DeepSeek-V4 的关系

[DeepSeek-V4](../models/deepseek-v4.md) 的 CSA 可以看作“压缩后的 DSA”：先把多个 token 的 KV 压缩成一个 entry，再用 lightning indexer 做 top-k selection。这样 DSA 的选择对象从原始 token 变成压缩块，进一步降低百万 token 场景下的 KV 和 attention 成本。

## 复现风险

- DSA 的收益依赖 indexer 训练是否稳定。
- top-k selection 的确定性是 RL 复现的关键细节。
- 只看检索 benchmark 不够，还要看 noisy long-context reasoning 和 agentic rollout。

