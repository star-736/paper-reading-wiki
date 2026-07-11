---
type: Concept
title: "高效长上下文注意力"
description: "DSA、混合 SWA/GA、CSA 和 HCA 的对比。"
tags: ["concept", "efficient-long-context-attention"]
timestamp: 2026-06-06
---

# 高效长上下文注意力

## 问题

长上下文模型的瓶颈不是单一的“能不能放下更多 token”，而是 attention FLOPs、KV-cache、prefill latency、decode latency、cache reuse 和训练稳定性共同构成的系统问题。多篇报告都在处理这个瓶颈，但路线不同。

## 五条路线 + 学习式驱逐

前四条都仍在 softmax 注意力框架内（只是少看 token 或压缩 KV）；第五条「线性/混合」直接换掉 token mixer，是与前四条**正交**的另一条路线。**学习式驱逐**（learned eviction）是另一个正交维度：它不在 per-query 层面选 token（sparse retrieval），而是永久丢弃 token 以强制固定 cache 大小，解决的是 **memory bound** 而非 attention compute。

| 路线 | 代表方案 / 模型 | 核心思想 | 主要收益 | 主要风险 |
| --- | --- | --- | --- | --- |
| 内容稀疏（token 级） | DSA / [GLM-5](../models/glm-5.md) | lightning indexer 给 query 选 top-k token，所有 query head 共享一个 top-k。 | 长程信息访问自适应；可从 dense checkpoint 继续训练得到。 | indexer 的 top-k 稳定性会影响 RL；indexer 自身仍是 O(NL²)。 |
| 内容稀疏（block 级） | [MSA](../sources/msa.md) / MiniMax-M3 | 在 GQA 之上加 Index Branch，每个 GQA group 独立选 n 个 KV 块。 | 块级 IO 更规整，KV-outer kernel 易拿到 1M context 14× prefill / 7× decode；对 GQA 改动小。 | 评测仅在 109B-MoE 预训练上做过，RL 后训练阶段稳定性还没有公开数据。 |
| 模式稀疏 | [MiMo-V2-Flash](../models/mimo-v2-flash.md) / [Gemma 4](../models/gemma-4.md) / [Unlimited OCR](../models/unlimited-ocr.md) | 5 个 SWA 层配 1 个 GA 层（Gemma 4 E2B 用 4:1）。Gemma 4 额外在全局层做 key-as-value + p-RoPE + KV sharing，全局 KV cache -37.5%。Unlimited OCR 的 R-SWA 更激进：全部层用 SWA，但把 reference token（视觉+prompt）排除在滑动窗口之外全局固定可见，KV cache 恒定 $L_m + n$ 不随输出增长。 | 架构简单，KV 和 attention 成本下降。Unlimited OCR 在 6144 token 输出时 TPS 比 DeepSeek OCR 高 35%，KV cache 完全恒定。 | 对需要任意长程交互的任务可能不如内容自适应。Unlimited OCR 的 128-token 窗口对跨页远距离引用（如第 20 页引用第 1 页的图表编号）覆盖力不足，除非信息经 reference token 间接传递。 |
| 压缩注意力 | [DeepSeek-V4](../models/deepseek-v4.md) | CSA/HCA 先压缩 KV，再做稀疏或密集注意力。 | 支持 1M context，KV-cache 极大降低。 | 架构、kernel、cache 管理复杂。 |
| 线性 / 混合 | KDA / [Kimi Linear](../models/kimi-linear.md) | 大多数层用线性注意力（RNN 固定状态，无随长度增长的 KV），少数层（3:1）保留全局 [MLA](multi-head-latent-attention.md)。 | decode 时线性层无 KV cache，1M context KV 降 75%、吞吐 6.3×；首个公平对比下全面追平 full attention 的混合线性方案。 | 固定状态容量有限，长程精确检索靠那 1/4 全局层兜底；线性层在长 trajectory RL 上的稳健性证据仍有限。 |
| 学习式驱逐 | [KVpop](../sources/kvpop.md) | 用 future-attention target 在 eviction boundary 监督 keep-or-drop 决策，永久丢弃 token 强制固定 per-head KV budget $B=s+w+k$。可选 mLSTM 延迟打分利用近未来上下文。 | Qwen3-8B 在 88% 压缩下保留 100% teacher 性能；固定 per-head budget 使 GPU 执行更规整，比 DMS 更快；memory bound 解决后 131k token 生成 VRAM 仅 19GB。 | 与 sparse retrieval 正交（后者不 bound memory）；目前是 post-training retrofit，未验证 from-scratch 或 MLA 架构。 |

## 机制层理解

DSA 不是固定窗口，而是先用 indexer 为 query 找出重要的历史 KV entries，再只对这些 entries 做注意力。它的直觉是：长上下文里绝大多数历史 token 对当前 token 并不重要，因此密集注意力浪费计算。GLM-5 报告中，DSA 从 dense base model 继续训练而来，目标是在不从头训练的情况下把长上下文成本降下来。

MSA 在内容稀疏这一支里走相反方向：把粒度从 token 抬到 block（B=128，n=16），并把 top-k 从"所有 query head 共享"改成"每个 GQA group 独立选块"。block-level 让访存更规整，KV-outer iteration 配合 query gather 能把 FLOPs/IO 比从 d 提到约 ⅔·G·d；GQA-group 独立选块则保留了多组检索的多样性。代价是新增了两个 idx 投影矩阵，部署门槛比 DSA 那种"零参数改动"的 indexer 高一点。

MiMo-V2-Flash 的 hybrid SWA/GA 更像工程上保守的折中。SWA 限制局部窗口，GA 周期性提供全局通路。5:1 比例和 128-token window 让大多数层成本较低，同时避免纯局部模型完全失去远程通信能力。

[Gemma 4](../models/gemma-4.md) 走同一路线（E2B 4:1，其余 5:1），但在 KV 侧做了更激进的压缩：全局层直接复用 key 作为 value（values = keys，与 [MLA](multi-head-latent-attention.md) 的 KV 压缩思路类似但不完全相同）、全局层用 p-RoPE (p=0.25) 替代标准 RoPE、并在 E2B/E4B 上做 KV cache sharing（20/35 和 18/42）。三者组合把全局 KV cache 压低 37.5%。Gemma 4 的 RULER 128k 评测中 31B 达 96.4、E4B 达 86.6，远超 Gemma 3 27B 的 66.0，说明 SWA/GA 混合 + KV 侧优化在 128K 级上下文足够有效，不需要切换到 DSA/CSA 等内容稀疏方案。

[Unlimited OCR](../models/unlimited-ocr.md) 的 R-SWA 是模式稀疏路线中一个极端变体。MiMo-V2-Flash 和 Gemma 4 都保留了周期性全局注意力层（5:1 或 4:1），而 R-SWA 把**全部**注意力层替换为 sliding window--但加了一个关键约束：reference token（视觉 token + prompt）全局固定可见且不做状态转移。这让 R-SWA 跨越了模式稀疏与线性注意力的边界：output 侧的行为像 SWA（固定窗口滑动、soft forgetting），reference 侧的行为像「冻结的全局 prefix」。论文明确指出这**不是**线性注意力--视觉/reference token 不参与 recurrent state update，否则会渐进模糊视觉特征。R-SWA 把恒定 KV cache 推到了 $L_m + n$ 的极端：不仅 decode 延迟恒定，GPU 显存也恒定，在 6144 token 输出时 TPS 比 full-attention 基线高 35%。代价是 128-token output 窗口无法覆盖跨页远距离引用。详见 [Unlimited OCR Works](../sources/unlimited-ocr.md)。

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

