# DeepSeek Sparse Attention

## 一句话定义

DeepSeek Sparse Attention(DSA)是一种内容自适应的长上下文稀疏注意力机制:它不固定看最近窗口,也不对所有历史 token 做 dense attention,而是为当前 query 动态选出最相关的一批历史 KV entries。

## 解决的问题

长上下文 dense attention 的成本随序列长度快速增长。对 agent 来说,这个问题更严重:工具调用历史、代码上下文、搜索结果和多轮推理会把上下文推到几十万 token。很多历史 token 对当前动作并不重要,但 dense attention 仍然会为它们付费。

DSA 的核心假设是：长上下文中存在大量注意力冗余，可以通过内容相关性选择保留关键 token。

## 架构概览

DSA 由 DeepSeek-V3.2 首次引入（[来源](../sources/deepseek-v32.md)），包含两个核心组件：

- **Lightning Indexer**：基于内容计算 query token 与历史 token 的 index score，使用少量 attention head（V3.2 未公开 head 数，GLM-5 附录报告为 32 头 / head dim 128），FP8 友好。
- **Fine-grained Token Selection**：对每个 query 位置取 index score 的 top-k 个历史 KV entry（V3.2 中 k = 2048），做精确 softmax 注意力。

DSA 实例化在 [MLA](multi-head-latent-attention.md) 的 **MQA 模式**下——所有 query head 共享同一组被选中的 KV entry。这一点不是 DSA 的设计取舍，而是 MLA-MQA mode 本来就只有一份共享 latent KV 可选；为什么 MLA 会有这种「所有 head 共享 latent」的结构，见 [Multi-Head Latent Attention](multi-head-latent-attention.md)。详细训练方案见 [DeepSeek-V3.2 来源页](../sources/deepseek-v32.md)。

## GLM-5 中的 DSA

[GLM-5](../models/glm-5.md) 把 DSA 用作长上下文效率组件。报告中 DSA 从 mid-training 结束后的 dense base model 继续训练而来,而不是从头训练。这条「dense warm-up + sparse training adaptation」两阶段范式由 DeepSeek-V3.2-Exp 引入并命名,GLM-5 沿用。

具体训练配方(论文 §2.1.1 + 附录 A):

1. **DSA warmup**:从 mid-training 末的 dense MLA checkpoint 起,跑 **1000 步**,每步 14 条 **202,752 token** 序列;学习率从 **5e-3 衰减到 2e-4**。主文本未直说 warmup 阶段冻结 base,但附录里 GLM-4.7-Flash 上的小规模消融把这条范式写得更明确:「trains only the indexer for 1,000 steps (batch size 16) while keeping all base-model weights frozen」,可作 GLM-5 主流程的佐证。
2. **Sparse adaptation**:沿用 mid-training 的数据和超参,**恒定学习率 1e-5**,再训 **20B tokens**。这一步让主干适应稀疏 KV。
3. 对比 MLA dense baseline,长上下文 benchmark 表现接近,相同 SFT 数据下训练 loss 与评测打平。

Indexer 的具体配置(附录 Table 10):**32 个 indexer attention head,head dim 128**;与 MLA 主分支并列存在,不参与 KV cache。

报告给出的直观结论是:在 128K 级长上下文里,约 **90% attention entries 是冗余的**;DSA 能把长序列 attention computation 降低约 **1.5-2 倍**。

## RL 稳定性问题

GLM-5 的重要细节是:DSA 在 RL 中不只是推理优化,还会影响训练稳定性。DSA indexer 每个 token 都要选 top-k KV entries(GLM-5 中 **k = 2048**)。如果 top-k 算子非确定性,训练引擎和推理引擎看到的 token selection 可能不一致,导致 RL 几步后 entropy 急剧下降。

GLM-5 的处理方式(论文「DSA RL insights」段,原文措辞):

- 训练中使用 deterministic `torch.topk`:相比 SGLang DSA Indexer 用的 non-deterministic CUDA 实现稍慢,但输出一致。其他 non-deterministic 实现(CUDA、TileLang)会让 RL 几步后性能塌陷并伴随 entropy 急降。
- **整个 RL 阶段都用 `torch.topk` 作为训练引擎默认 top-k 算子**。
- **默认冻结 indexer 参数**,既加速训练,也防止 indexer 自身不稳定学习。
- 不采用 MoE routing replay 那种保存每 token top-k index 的策略--DSA 的 k = 2048 远大于 MoE 的 k,存储和训练/推理引擎间的通信成本都会爆炸。

## 与 DeepSeek-V3.2 / DeepSeek-V4 的关系

DSA 由 [DeepSeek-V3.2](../sources/deepseek-v32.md) 首次引入并命名，其训练方案（dense warmup → sparse training with KL alignment）已成为后续模型的参考范式。

[DeepSeek-V4](../models/deepseek-v4.md) 的 CSA（Compressed Sparse Attention）可以看作 DSA 的演进：先将多个 token 的 KV 压缩成一个 entry，再用 lightning indexer 做 top-k selection。这样 DSA 的选择对象从原始 token 变成压缩块，进一步降低百万 token 场景下的 KV 和 attention 成本。

## Post-training 中 indexer 的处理对比

DeepSeek-V3.2 的 post-training 阶段**继续使用 sparse attention**（配置与 sparse continued pretraining 相同），即 indexer 在 RL 阶段保持更新。这与 GLM-5 的策略形成对比——GLM-5 在 RL 阶段**冻结 indexer 参数**并强制使用 deterministic `torch.topk` 来保证训练-推理一致性。两种路线反映了不同的工程取舍：V3.2 选择让 indexer 继续适应 RL 分布，代价是训练复杂度更高；GLM-5 选择冻结 indexer 以消除非确定性风险，代价是推理时的 indexer quality 固定。

## 与 MSA 的对比

[MSA](../sources/msa.md) 是同代另一种 natively trained 稀疏注意力。两者关键区别:

- **选择粒度**:DSA 是 token-level,MSA 是 block-level(B=128,n=16)。
- **跨头共享**:DSA 让所有 query head 共享一个 top-k;MSA 让每个 GQA group 独立选 top-k,组间不共享。
- **底层 attention**:DSA 嵌入 MLA(其 MQA 模式);MSA 嵌入 GQA。
- **训练目标**:DSA 用 lightning indexer 对 full attention 蒸馏;MSA 同样用 KL 对齐,但额外证明了多层 KL 与对均值分布单 KL 等价(在 [IndexCache](../sources/indexcache.md) 的多层蒸馏里被独立用到)。

## indexer 自身仍是 O(L2) 的瓶颈

DSA 的 lightning indexer 虽然每层比主注意力便宜一个数量级,但每层都得对所有历史 token 打分,N 层叠加成 O(NL2)。在 30B-A3B DSA 模型上,它在 prefill 阶段占总延迟 50-81%、decode 阶段占 27-41%。

[IndexCache](../sources/indexcache.md) 发现相邻层 indexer 的 top-k 重合率高达 70-100%,因此可以让多数层跳过 indexer 直接复用前一个 Full 层的索引,并通过贪心搜索或多层蒸馏决定哪些层保留 indexer。在 GLM-5 上 1/4 retention 仍能接近原始长上下文表现,端到端 1.3× 起步。这条线的上下文见 [跨层索引复用](cross-layer-index-reuse.md)。

## 复现风险

- DSA 的收益依赖 indexer 训练是否稳定。
- top-k selection 的确定性是 RL 复现的关键细节。
- 只看检索 benchmark 不够,还要看 noisy long-context reasoning 和 agentic rollout。
- 部署时 indexer 自身的 O(NL2) 不可忽略;要把"主注意力变稀疏"和"indexer 跨层复用"看成两件互补、必须一起算账的事情。

