---
type: Source
title: "MiniMax Sparse Attention 技术报告"
description: "MiniMax Sparse Attention 的 arXiv 报告，GQA-block 级稀疏 + 每个 group 独立 top-k，1M context 下 14× prefill / 7× decode。"
tags: ["source", "msa"]
timestamp: 2026-06-19
resource: "../../raw/Lai%20%E7%AD%89%20-%202026%20-%20MiniMax%20sparse%20attention.pdf"
---

# MiniMax Sparse Attention 技术报告

## 来源

- 原始 PDF：[raw/Lai 等 - 2026 - MiniMax sparse attention.pdf](../../raw/Lai%20%E7%AD%89%20-%202026%20-%20MiniMax%20sparse%20attention.pdf)
- 标题：MiniMax Sparse Attention
- 版本/日期：arXiv:2606.13392v2，2026-06-12
- 团队：MiniMax 与 Peking University、NVIDIA、Zhejiang University 等合作机构
- 配套实现：[MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA) inference kernel；生产级模型 [MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3)
- 模型页：[MiniMax-M3](../models/minimax-m3.md)

## 核心结论

MSA 是构建在 Grouped Query Attention（GQA）之上的 blockwise 稀疏注意力。它给每个 GQA group 加一条轻量 Index Branch，由它独立选 top-k key-value 块；Main Branch 只对选中的块做精确 softmax 注意力。设计准则是 Occam's razor——只保留必需组件，使其能在多种 GPU 上稳定部署。

在 109B 参数 MoE（6B 激活）原生多模态模型上，3T token 训练预算下，MSA 与 GQA full attention 在 LM loss、梯度范数和大多数下游 benchmark 上基本打平；同时在 1M context 下 per-token attention FLOPs 降低 28.4×，配套 kernel 在 H800 上达到 14.2× prefill 加速、7.6× decode 加速。报告把 MSA 称为可以从 GQA checkpoint **near-lossless 转换**，也支持从头原生稀疏预训练。

## 架构与训练

**Index Branch**：在每个 GQA group 内引入一个 index query head 和组间共享的 index key head（两个新增 projection 矩阵）。先在 token 级别打分，再用 max-pooling 聚合到块级，最后 top-k 选 n=16 个块（block size B=128）。每个 group 独立选块，但 group 内所有 query head 共享一个 index 集合。包含当前位置的 local block 始终被强制选入。

**Main Branch**：只对被选中的块做标准 scaled dot-product attention，复杂度从 O(L²) 降到 O(L·n·B)，与序列长度近乎独立。

**训练机制**——top-k 不可微，所以用三件事稳住稀疏训练：

1. **KL 对齐损失**：Index Branch 的分布对齐到 Main Branch 在所选 token 上的 group-averaged 概率分布（teacher 通过 stop-gradient 切断到 backbone 的反传）。论文证明这等价于对"被服务层的注意力分布质心"做单 KL 蒸馏。
2. **Gradient Detach**：Index Branch 的输入也 stop-gradient，使 KL 损失只更新 idx 投影、不污染 backbone。
3. **Indexer Warmup**：前一段先两路都跑 full attention 训练新加的 idx 投影；warmup 后才切到稀疏。从 dense GQA checkpoint 做 sparse continued pretraining（MSA-CPT）也走同样 warmup。

## Kernel 设计

- **Exp-free TopK**：softmax 顺序保持，直接对原始分数排序，跳过 max/exp/sum；针对 small-k（n=16）配 per-thread register min-heap，比 `torch.topk` 和 TileLang radix-select 都快（H800 上 1.2–5.1× 不等）。
- **KV-outer 稀疏注意力**：相比 Q-outer，迭代 KV 块、按反向索引 gather query 能把 FLOPs/IO 比从 d 提到 ~⅔·G·d（G 是 GQA 组比），更适合稀疏 prefill。
- **Pre-scheduled tile chunking + two-phase combine**：把热块沿 query 维度切片到多个 CTA，避免 atomic 写；split-K partial 用 logsumexp 在 combine kernel 里合并。
- **Query concatenation**：KV-outer 下同一 KV tile 的 query 共享 KV 操作数，可以把 128/G 个 query 位置打包成 128×128 的 score MMA，避免 MMA M 维度欠填充。
- **Sparse KL backward**：把 LSE 融合进 forward 直接写出，跳过 KL forward；persistent grid + atomic counter 做动态负载均衡。

## 实验配置

- 同一族 41 层 MoE backbone（3 dense + 38 MoE，128 routed expert + 1 shared expert，top-4），109B 总参/6B 激活，d_model=3072，64 query head / 4 KV head（GQA 组比 G=16），head dim 128，RoPE dim 64。
- MSA-PT（from scratch）：40B token indexer warmup → 余下 sparse 训练，总 3T token。
- MSA-CPT（continued pretraining）：从 GQA full-attention 在 2.6T 时的 checkpoint 起，替换为 MSA，再训 400B token（前 40B warmup）。

## 评测要点

- LM loss/梯度范数与 full attention 全程几乎重合，sparse CPT 后 KL loss 也保持低位，block recall 和 score recall 表明被选块覆盖了大部分主分支注意力质量。
- 主表上 MSA-PT 在数学、图像、视频、长上下文 retrieval 多数项上略胜 full baseline；MSA-CPT 在文本/代码/PPL 项上更接近 full baseline。
- HELMET-128K 上 MSA-CPT Overall 仅低 0.6（46.53 → 45.93）；RULER-128K 平均反高 0.12 分；显式表明每 query 仅看 n·B = 2,048 个 KV token 时长上下文能力可保持。
- 效率：1M context 下 attention FLOPs 减 28.4×；H800 实测 prefill 加速 14.2×、decode 加速 7.6×（运行时收益小于理论 FLOPs 收益，因为索引、TopK、reverse-index、query gather 都有 overhead）。

## 与已有沉淀的关系

- 同样属于 [高效长上下文注意力](../concepts/efficient-long-context-attention.md) 中的"内容自适应稀疏"路线，与 [DeepSeek Sparse Attention](../concepts/deepseek-sparse-attention.md) 是同代竞争方案。两者关键差异：DSA 是 **token-level、所有 query head 共享一个 top-k**；MSA 是 **block-level、每个 GQA group 独立 top-k**。
- 与 NSA、MoBA、InfLLM-V2 一起属于"natively trained sparse attention"族，区别于 H2O / SnapKV / Quest / MInference / FlexPrefill 这类训练后稀疏化方法。
- MSA 报告里第一次把 sparse-attention 训练稳定性和 KL 对齐目标的等价性写得比较干净（Proposition：multi-token 多 KL = 对均值分布的单 KL）。这条数学事实在 [IndexCache](indexcache.md) 的多层蒸馏里也被独立用到。

## 公开 kernel 实现

[MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA) 的 README 里给出了一些论文外的工程信息：

- **目标 GPU**：NVIDIA **SM100（Blackwell）**。这与论文实验用的 H800（SM90 / Hopper）不一致——公开 kernel 的优化路径是 Blackwell，而论文里 14.2× / 7.6× 的测速数字是 Hopper 上的。
- **两套 JIT 栈**：`csrc` 一栈实现 dense FMHA（`fmha_sm100`、`fmha_sm100_plan`）和 `sparse_topk_select` indexer；`CuTe-DSL` 一栈实现完整稀疏注意力 forward 和 paged FP8 decode，加一个 dense→sparse prefill 桥接。
- **精度支持**：BF16、FP8、NVFP4、FP4（NVFP4 仅用于 sparse prefill）。论文里只讨论 BF16 和 FP8，FP4 / NVFP4 是公开 kernel 的扩展。
- **示例参数**：`page_size=128`、`topk=16`，与论文的 B=128 / n=16 完全一致。
- **License**：MIT，第三方组件含 BSD-3-Clause 和 Apache-2.0。
- **HF kernel 入口**：通过 `kernels` 库 `get_kernel("MiniMaxAI/msa", version=0)` 调用；README 还引用了 `huggingface.co/kernels/kernels-staging/msa`。
- **BibTeX**：README 的 BibTeX 写的是占位符（"coming once the companion paper / technical report has a stable identifier"），暗示 README 早于 arXiv 公开版本。

## 待追问

- MSA 的 group-shared block-level top-k 与 DSA 的 head-shared token-level top-k，哪一边在 RL 阶段更稳定？GLM-5 报告提到 DSA 因 top-k 非确定性会让 RL entropy 崩塌；MSA 论文 Outlook 节亲口说「extending the same selector-only design to settings beyond pretraining, including reinforcement-learning post-training and agentic deployment」是待做工作，主体报告只覆盖 pretraining。这是明确的 open problem，不是「论文未表态」。
- MSA-PT 在多数 benchmark 上小胜 full baseline，是稀疏作为正则的副作用，还是 3T 预算下 full attention 还没充分训练？需要更长 horizon 或匹配的更大 baseline 才能判断。
- 报告强调 MSA 可以从 GQA 直接转换，但其核心仍是新加 idx 投影；与 DSA 那种 indexer 不变、只换 top-k 范式的"零参数改动"路线相比，部署门槛差多少。
- 论文 14.2× prefill / 7.6× decode 是在 H800 上测的，但公开 kernel 已经针对 SM100 / Blackwell 重写。在 Blackwell 上这两个数字会变成什么？没有官方数据。
