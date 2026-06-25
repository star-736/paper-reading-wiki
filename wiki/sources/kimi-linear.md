---
type: Source
title: "Kimi Linear 技术报告"
description: "Moonshot AI 的混合线性注意力，KDA（细粒度门 delta rule）3:1 配 Full MLA，首次在公平对比下全面追平 full attention，1M context KV 降 75%、decode 6.3×。"
tags: ["source", "kimi-linear"]
timestamp: 2026-06-21
resource: "../../raw/Team%20%E7%AD%89%20-%202025%20-%20Kimi%20linear%20An%20expressive%2C%20efficient%20attention%20architecture.pdf"
---

# Kimi Linear 技术报告

## 来源

- 原始 PDF：[raw/Team 等 - 2025 - Kimi linear An expressive, efficient attention architecture.pdf](../../raw/Team%20%E7%AD%89%20-%202025%20-%20Kimi%20linear%20An%20expressive%2C%20efficient%20attention%20architecture.pdf)
- 标题：Kimi Linear: An Expressive, Efficient Attention Architecture
- 版本/日期：arXiv:2510.26692v2，2025-11-01
- 团队：Kimi Team（Moonshot AI）
- 配套实现：[KDA kernel（flash-linear-attention）](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda)、vLLM 集成；checkpoint [Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)
- 模型页：[Kimi Linear](../models/kimi-linear.md)

## 核心结论

Kimi Linear 是一个**混合线性注意力**架构：用大多数层跑线性注意力（[Kimi Delta Attention，KDA](../concepts/linear-attention-and-delta-rule.md)），周期性插入少数全局 softmax 注意力层（Full [MLA](../concepts/multi-head-latent-attention.md)），比例固定 **3:1**（3 个 KDA 层配 1 个 MLA 层）。报告主张：在严格公平对比下（同训练配方、1.4T token），Kimi Linear 在短上下文、长上下文和 RL 三类场景下**全面追平或超过 full attention**——这是混合线性注意力首次做到这一点。

定量上：48B 总参 / 3B 激活的模型，相对 full MLA 在 MMLU-Pro（4k）51.0 vs 47.2、RULER（128k）84.3 vs 81.3 领先，同时把长序列生成的 KV cache 降低**最多 75%**，1M context 下 decode 吞吐**最多 6.3×**（TPOT 1.84ms vs MLA 11.48ms）。它定位成「full attention 的 drop-in 替换」，不需要改 cache / 调度接口。

![Kimi Linear Figure 1：性能 vs 加速。(a) 1.4T token 严格公平对比下，红星为 MMLU-Pro（4k），Kimi Linear 以 51.0 领先且速度相当；蓝圈为 RULER（128k），Kimi Linear 取得 84.3 的最高分并拿到 3.98× 加速，处于 Pareto 最优。(b) 每输出 token 时间（TPOT）vs 解码长度，Kimi Linear（蓝线）在长序列上维持低 TPOT，与 GDN-H 接近、显著优于 MLA，1M token 处 TPOT 6.3×（1.84ms vs MLA 11.48ms）。](../assets/kimi-linear/fig1-performance-vs-acceleration.png)

> Figure 1（原文截图，§ 摘要旁）：\"(a) Performance vs. acceleration. … On RULER (128k context length, blue circles), it is Pareto-optimal, achieving top performance (84.3) and 3.98× acceleration. (b) Time per output token (TPOT) vs. decoding length. … a 6.3× faster TPOT (1.84ms vs. 11.48ms) than MLA at 1M tokens.\"

> 注意区分：本页的 **Kimi Linear** 是一篇**注意力架构论文 + 48B 研究模型**，与生产旗舰 [Kimi K2.5](kimi-k2.5.md)（1.04T 多模态 agentic 模型）不是同一个东西，只是同属 Moonshot AI。

## 架构与训练

**KDA（核心创新）**：在 [Gated DeltaNet（GDN）](../concepts/linear-attention-and-delta-rule.md) 的基础上，把 GDN 那个 head-wise 标量遗忘门 $\alpha_t$ 换成 **channel-wise 细粒度门** $\mathrm{Diag}(\alpha_t)$——每个特征维度有独立的遗忘速率（思路来自 GLA），从而更精细地调度有限的 RNN 状态记忆。状态更新：

$$S_t = \left(I - \beta_t k_t k_t^\top\right)\mathrm{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^\top,\qquad o_t = S_t q_t$$

**硬件效率**：细粒度门一般要求用 Diagonal-Plus-Low-Rank（DPLR）转移矩阵，但通用 DPLR 算力贵。KDA 用一个**专门化的 DPLR 变体**——把低秩两支 $a,b$ 都绑定到 $k$——配合 chunkwise-parallel 算法（WY representation + UT transform），把二级 chunk 矩阵计算从 4 次降到 2 次、再省掉 3 次矩阵乘，算子效率比通用 DPLR 提升约 **100%**，同时保持与经典 delta rule 一致。

**神经参数化**（每个 KDA head）：q/k/v 都先过 ShortConv + Swish；q、k 再做 L2Norm 稳定特征值；channel 衰减 $\alpha_t$ 由低秩投影 + 衰减函数 $f(\cdot)$ 给出；输出端用 head-wise RMSNorm + **data-dependent 输出门**（低秩参数化的 sigmoid 门，缓解 attention sink——与 [注意力门控](../concepts/attention-gating.md) 同源思路）。

**混合策略**：
- **layerwise** 混合（整层交替）而非 headwise（层内混头），理由是基础设施更简单、训练更稳。
- **3:1** 是消融选出的最优质量/吞吐折中。Table 1（原文 § 4，数据重排为 Markdown；训练/验证 PPL 越低越好，灰底为最终选用配置）：

  | 配置 | Training PPL ↓ | Validation PPL ↓ |
  | --- | --- | --- |
  | **Hybrid ratio 3:1**（选用） | **9.23** | **5.65** |
  | 0:1（纯 MLA） | 9.45 | 5.77 |
  | 1:1 | 9.29 | 5.66 |
  | 7:1 | 9.23 | 5.70 |
  | 15:1 | 9.34 | 5.82 |
  | w/o output gate | 9.25 | 5.67 |
  | w/ swish output gate | 9.43 | 5.81 |
  | w/o convolution layer | 9.29 | 5.70 |

  读法：比例越高（线性层越多）验证 PPL 越差（15:1 最差 5.82），但 3:1 已优于纯 MLA（0:1 的 5.77）且接近 1:1，是质量/吞吐甜点；输出门和卷积层的消融（去掉或换 swish）都使 PPL 变差，印证 KDA 的 sigmoid 输出门 + ShortConv 设计有效。
- **NoPE for MLA layers**：所有全局 MLA 层不加位置编码，把位置/recency 编码的责任全交给 KDA 层。好处：MLA 层推理时可退化成高效的纯 MQA；长上下文训练不必调 RoPE 频率或用 YaRN。

**训练配置**：backbone 沿用 Moonlight，MoE sparsity 32（256 expert 选 8，含 1 shared expert），48B 总参 / 3B 激活，首层为 dense；优化器 Muon；公平对比的主实验用 1.4T token。

## 评测要点

- **合成任务**（Palindrome / MQAR / Stack）：KDA 在 copying、多查询关联召回、状态跟踪上收敛和峰值精度均优于 GDN、Mamba2，验证细粒度门对有限状态记忆的调度优势。
- **短上下文**：MMLU-Pro（4k）51.0，领先 full MLA（47.2）和 GDN-H（47.9），且速度相当。
- **长上下文**：RULER（128k）84.3，Pareto 最优，兼得 3.98× 加速；Table 5 显示在多个长上下文 benchmark 上平均分最高。
- **效率**：1M context 下 KV cache 降最多 75%，decode 吞吐最多 6.3×（更低 TPOT 还允许更大 batch）。
- **RL**：报告声称在 RL-style post-training 上同样追平/超过 full attention，但这是混合线性注意力此前的薄弱环节，是其主要卖点之一。

## 与已有沉淀的关系

- 这是 wiki 里**第一篇线性注意力（RNN-state）路线**的报告，与现有的稀疏-softmax 路线（DSA / MSA / CSA）正交——后者仍是 softmax 注意力只是少看 token，前者把 token mixer 整个换成线性 RNN。详见 [线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md) 和 [高效长上下文注意力](../concepts/efficient-long-context-attention.md) 新增的「线性/混合」一行。
- 它**仍然依赖 [MLA](../concepts/multi-head-latent-attention.md)** 作为那 1/4 的全局层，所以不是「抛弃 MLA」而是「把 MLA 稀释到 1/4 层、其余换 KDA」。这跟 DSA/CSA「在 MLA 上加稀疏选择」是不同的省法。
- 输出门缓解 attention sink 的做法与 [注意力门控](../concepts/attention-gating.md)（Qwen 团队的 Gated Attention）是同一族思想的两个独立实例。

## 待追问

- 报告称 RL 阶段也追平 full attention，但具体 RL 配方、任务和稳定性细节（线性注意力的有限状态会不会在长 trajectory RL 上掉链子）正文给的篇幅有限，值得回原文 § 5 后半段细读。
- 3:1 比例是在 48B/1.4T 这个尺度上选出的；更大尺度（如 1T 级）这个比例是否仍最优，论文未给。
- KDA 的 channel-wise 门相比 GDN 的 head-wise 门，参数量和显存增量具体多少？报告强调用低秩投影控制，但精确数字需查配置表。
- NoPE 把位置编码全压给 KDA 层，这在超过训练长度的极端外推（如 4M+）下是否仍稳，报告未覆盖。
