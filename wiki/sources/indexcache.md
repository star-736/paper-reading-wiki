# IndexCache 技术报告

## 来源

- 原始 PDF：[raw/Bai 等 - 2026 - IndexCache Accelerating sparse attention via cross-layer index reuse.pdf](../../raw/Bai%20%E7%AD%89%20-%202026%20-%20IndexCache%20Accelerating%20sparse%20attention%20via%20cross-layer%20index%20reuse.pdf)
- 标题：IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse
- 版本/日期：arXiv:2603.12201v1，2026-03-12
- 团队：清华大学 + Z.ai（智谱），第一作者 Yushi Bai 在 Z.ai 实习期间完成
- 与 GLM-5 的关系：同时在 30B-A3B 验证模型和 744B 的 [GLM-5](../models/glm-5.md) 上做了实验，可视为 GLM-5 之上的推理优化扩展

## 核心结论

[DeepSeek Sparse Attention](../concepts/deepseek-sparse-attention.md)（DSA）已经把 core attention 从 O(L²) 降到 O(Lk)，但每层那个 lightning indexer 自己仍是 O(L²)，N 层叠加变成 O(NL²)，长上下文 prefill 时占总延迟 50–81%、decode 时占 27–41%。IndexCache 的观察是：**相邻层 indexer 选出的 top-k 高度相似（70–100% 重合）**，因此把多数层的 indexer 计算省掉、直接复用前一个 Full 层的索引，就能消掉这一项开销。

把 47 层中只保留 1/4 indexer，30B 模型质量基本无损：200K context 下 prefill 1.82×、单请求 decode 1.48×、满 KV cache decode 1.51×；GLM-5 上 1/4 retention 仍能在 5 个长上下文 benchmark 上维持 78.0 vs. 78.4 的平均分。

## 方法

把 N 层划成两类：

- **F (Full)**：保留 indexer，正常算 top-k 并写入 `Tcache`（常驻 buffer，每层覆盖）。
- **S (Shared)**：跳过 indexer，直接从 `Tcache` 读最近的 F 层 top-k 集，做稀疏注意力。

第一层永远是 F，给后续 S 层提供初始索引。推理代码改动只是一条 `if c=F` 分支，不需要新增 GPU 内存。关键问题是怎么决定 pattern `c = c₁c₂…c_N`。

### Training-Free：贪心层选择

均匀间隔（FSSSF…）会失败——某些层（尤其早期/过渡层）对 indexer 移除高度敏感。论文用贪心搜索代替：

1. 从全 F 起步，每步在所有当前 F 层（除第 1 层外）里逐个试翻成 S，挑选 LM loss 最小的一翻定下，重复 K = 3N/4 步。
2. 用 200K context、768 batch 的 SFT 数据做 calibration，所有候选 pattern 在同一批数据上比较，loss 差异只反映 pattern。
3. 流水并行下还可以把模型切成 P 段，每段独立选最优翻动，把 forward 数从 N(N-1)/2 减少约 P 倍。

LM loss 曲线呈现"前 20 步几乎无损 → 35 步后陡升"的明显分界，说明确实存在天然的 indexer 重要性排序。

### Training-Aware：多层蒸馏损失

如果可以训练，让保留的 indexer 同时服务多层即可。设 F 层 ℓ 后接 m 个 S 层，多层 KL 损失为

$$L^\text{multi}_I = \sum_t \frac{1}{m+1}\sum_{j=0}^m D_\text{KL}(p_t^{(\ell+j)} \,\|\, q_t^{(\ell)})$$

**Proposition 1（命题）**：因为只有 q_t^{(ℓ)} 含可训练参数、p 的熵在求导后消失，多层 KL 在梯度上**严格等价于**对均值分布 p̄_t = (1/(m+1)) Σ p_t^{(ℓ+j)} 做单 KL。换句话说，indexer 学到的是被服务层注意力分布的质心，而不是只对自己那层过拟合。实现上仍用多层 KL（避免把 p^{(ℓ)} 显式传到 S 层带来的内存开销）。

训练沿用 DSA 标准两阶段：dense warmup（只动 F 层 indexer）→ sparse training（多层 KL + LM loss）。

## 实验

**模型**：30B-A3B MoE / 47 层 / MLA，从 GLM-4.7-Flash 出发训成 DSA 模型；training-aware 走的是缩短版 DSA 训练（1k 步 dense warmup + 4k 步 sparse 训练）。

**主要观察**：

| 设置 | 1/4 retention（保留 12 层） | 备注 |
| --- | --- | --- |
| Training-free + 贪心 pattern | Long Avg 49.9 vs. 50.2，G&R 74.9 vs. 74.4 | 比 1/4 均匀（43.0 / 73.8）高 6.9 个长上下文点 |
| Training-aware + 均匀 pattern | Long Avg 50.6, G&R 74.1 | 与 baseline 相当；甚至略好 |
| Training-aware + 贪心 pattern | 与均匀基本持平 | **训练后 pattern 敏感性消失** |
| Training-aware 去掉跨层 KL | Long Avg 49.8（AA-LCR 49.8 → 44.0） | 说明 multi-layer 蒸馏目标本身有用，不只是更长训练时长 |

**端到端推理**（30B，SGLang dp=8，H100）：1/4 retention 在 200K 下 prefill 19.5s → 10.7s（1.82×），单请求 decode 58 → 86 tok/s（1.48×），满 KV cache 总 decode 197 → 297 tok/s（1.51×）。

**GLM-5 (744B)**：training-free + 贪心 1/4 retention，Long Avg 78.0 vs. 78.4，端到端至少 1.3× 提升；training-aware 版本"近期计划应用"。

## 待追问

- 贪心找出的"关键层"在不同 DSA 模型间是否稳定？论文说同一模型下与 calibration 数据无关，但跨模型迁移性还没有数据。
- 1/8 retention 时即便用搜索 pattern，Long Avg 也从 50.2 掉到 46.1。压缩极限在哪里、是否能用更激进的 training-aware 进一步推进？
- IndexCache 的核心依赖"indexer 输出跨层稳定"。如果将来稀疏注意力换成 MSA 这类 group-shared block-level 选择，跨层稳定性的形式和量级是否相同？论文 §5.2 提到 MoBA 和 NSA 也可能受益但未实验。
- F 层之外的 S 层在 training-aware 下"主动适应"了继承的索引——这种适应是否会让 KV cache cross-layer sharing（如 HySparse、MiniCache）变得更难，因为 S 层对 KV 的依赖结构变了。

## 与已有沉淀的关系

- 直接扩展 [DeepSeek Sparse Attention](../concepts/deepseek-sparse-attention.md)：把 DSA indexer 自身的 O(NL²) 成本砍掉。
- 与 [百万 token 上下文服务](../concepts/million-token-context-serving.md) 互补：DeepSeek-V4 用压缩注意力 + 异构 KV-cache 降访存，IndexCache 降 indexer 计算，两条路可叠加。
- 与 Kascade、TidalDecode、HySparse 等"anchor 层做 full attention、其他层复用 top-k"的工作思想一致，但前者们都需要 full attention 作 oracle；IndexCache 第一次把跨层索引复用范式迁移到"oracle 也只是稀疏 indexer"的设定。详见 [跨层索引复用](../concepts/cross-layer-index-reuse.md)。
