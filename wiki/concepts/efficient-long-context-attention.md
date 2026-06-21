# 高效长上下文注意力

## 问题

长上下文模型的瓶颈不是单一的“能不能放下更多 token”，而是 attention FLOPs、KV-cache、prefill latency、decode latency、cache reuse 和训练稳定性共同构成的系统问题。多篇报告都在处理这个瓶颈，但路线不同。

## 五条路线

前四条都仍在 softmax 注意力框架内（只是少看 token 或压缩 KV）；第五条「线性/混合」直接换掉 token mixer，是与前四条**正交**的另一条路线。

| 路线 | 代表方案 / 模型 | 核心思想 | 主要收益 | 主要风险 |
| --- | --- | --- | --- | --- |
| 内容稀疏（token 级） | DSA / [GLM-5](../models/glm-5.md) | lightning indexer 给 query 选 top-k token，所有 query head 共享一个 top-k。 | 长程信息访问自适应；可从 dense checkpoint 继续训练得到。 | indexer 的 top-k 稳定性会影响 RL；indexer 自身仍是 O(NL²)。 |
| 内容稀疏（block 级） | [MSA](../sources/msa.md) / MiniMax-M3 | 在 GQA 之上加 Index Branch，每个 GQA group 独立选 n 个 KV 块。 | 块级 IO 更规整，KV-outer kernel 易拿到 1M context 14× prefill / 7× decode；对 GQA 改动小。 | 评测仅在 109B-MoE 预训练上做过，RL 后训练阶段稳定性还没有公开数据。 |
| 模式稀疏 | [MiMo-V2-Flash](../models/mimo-v2-flash.md) | 5 个 SWA 层配 1 个 GA 层。 | 架构简单，KV 和 attention 成本下降。 | 对需要任意长程交互的任务可能不如内容自适应。 |
| 压缩注意力 | [DeepSeek-V4](../models/deepseek-v4.md) | CSA/HCA 先压缩 KV，再做稀疏或密集注意力。 | 支持 1M context，KV-cache 极大降低。 | 架构、kernel、cache 管理复杂。 |
| 线性 / 混合 | KDA / [Kimi Linear](../models/kimi-linear.md) | 大多数层用线性注意力（RNN 固定状态，无随长度增长的 KV），少数层（3:1）保留全局 [MLA](multi-head-latent-attention.md)。 | decode 时线性层无 KV cache，1M context KV 降 75%、吞吐 6.3×；首个公平对比下全面追平 full attention 的混合线性方案。 | 固定状态容量有限，长程精确检索靠那 1/4 全局层兜底；线性层在长 trajectory RL 上的稳健性证据仍有限。 |

## 机制层理解

DSA 不是固定窗口，而是先用 indexer 为 query 找出重要的历史 KV entries，再只对这些 entries 做注意力。它的直觉是：长上下文里绝大多数历史 token 对当前 token 并不重要，因此密集注意力浪费计算。GLM-5 报告中，DSA 从 dense base model 继续训练而来，目标是在不从头训练的情况下把长上下文成本降下来。

MSA 在内容稀疏这一支里走相反方向：把粒度从 token 抬到 block（B=128，n=16），并把 top-k 从"所有 query head 共享"改成"每个 GQA group 独立选块"。block-level 让访存更规整，KV-outer iteration 配合 query gather 能把 FLOPs/IO 比从 d 提到约 ⅔·G·d；GQA-group 独立选块则保留了多组检索的多样性。代价是新增了两个 idx 投影矩阵，部署门槛比 DSA 那种"零参数改动"的 indexer 高一点。

MiMo-V2-Flash 的 hybrid SWA/GA 更像工程上保守的折中。SWA 限制局部窗口，GA 周期性提供全局通路。5:1 比例和 128-token window 让大多数层成本较低，同时避免纯局部模型完全失去远程通信能力。

DeepSeek-V4 的 CSA/HCA 更激进。CSA 每 `m` 个 token 压缩成一个 KV entry，然后再做 DSA 式 top-k selection；HCA 用更大的 `m'` 做重压缩，并在压缩状态上做 attention。它还额外保留 sliding-window branch，弥补压缩块内部局部细节不足。

## 一个常被忽略的瓶颈：indexer 自身

DSA、MSA 这类内容稀疏方案都把"主注意力"成本从 O(L²) 降到 O(L·k)，但 indexer 自身仍然是 O(L²) per layer，N 层叠加是 O(NL²)。在 30B-A3B DSA 模型上，长上下文 prefill 阶段 indexer 能占到总延迟的 50–81%。把这一项进一步降下来不是靠改 indexer，而是靠[跨层索引复用](cross-layer-index-reuse.md)：让多数层跳过 indexer，直接继承前一个 anchor 层选好的 top-k。[IndexCache](../sources/indexcache.md) 是这条思路在 DSA 上的首个系统化实现，30B 模型可去掉 75% indexer 计算，GLM-5 上能拿到 ≥1.3× 端到端加速。

## 对比判断

如果目标是相对简单、可部署、可解释的 256K 级长上下文，MiMo 的 SWA/GA 路线很有吸引力。如果目标是百万 token 和共享前缀复用，DeepSeek-V4 的压缩 + 异构 KV-cache 更接近完整系统方案。GLM-5 的 DSA 更适合理解"内容自适应稀疏"在 agentic RL 中的价值和稳定性风险。MSA 则给出了 block-level + GQA-group 路线在 1M context 下的较干净 14×/7× 数字，是 GQA 模型族升级长上下文能力的一个低成本路径。

无论选哪条路线，都得把"主注意力 + indexer/选择器 + KV-cache 访存"当成三件互相影响的事情一起算账，光看主注意力的 FLOPs reduction 是不够的。

## 进一步阅读

- [线性注意力与 delta rule](linear-attention-and-delta-rule.md)（正交的第五条路线）
- [DeepSeek Sparse Attention](deepseek-sparse-attention.md)
- [跨层索引复用](cross-layer-index-reuse.md)
- [百万 token 上下文服务](million-token-context-serving.md)
- [2026 开放模型技术报告对比](../comparisons/2026-open-model-technical-reports.md)

