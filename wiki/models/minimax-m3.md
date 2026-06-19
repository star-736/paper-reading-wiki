# MiniMax-M3

## 身份

MiniMax-M3 是 MiniMax 在 [MSA 报告](../sources/msa.md)中作为生产化释出的"原生稀疏注意力 + 原生多模态"模型，与同期 [MiniMax-M2 Series](minimax-m2-series.md) 共同构成 MiniMax 的开放模型谱系，但定位明显不同：M2 走的是"低激活 MoE + Forge agentic RL + 长 horizon agent"，M3 走的是"长上下文效率 + 多模态预训练"。

主要来源：[MSA 技术报告](../sources/msa.md)。配套 inference kernel 公开在 [MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA)，权重在 [huggingface.co/MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3)（HF 页面信息当前未沉淀，待补）。

## 关键事实

| 项目 | 数值或描述 |
| --- | --- |
| Backbone | 41-layer MoE Transformer（前 3 层 dense，后 38 层 MoE） |
| 总参数 | 约 109B |
| 激活参数 | 约 6B / token |
| Experts | 128 routed + 1 shared，top-4 路由 |
| Hidden dim | d_model = 3072 |
| Attention | [MSA](../sources/msa.md)（block-level 稀疏 + GQA），64 query heads / 4 KV heads（GQA 组比 G=16） |
| Head dim | 128（RoPE 维度 64） |
| 选择 budget | 每 query 看 n·B = 16 × 128 = 2,048 个 KV token |
| 词表 | 200K tokens |
| 模态 | 原生多模态（文本 + 图像 + 视频联合预训练） |
| 预训练预算 | 3T tokens（MSA-PT 路线为代表配置） |
| Long-context | 后续 ~140B tokens 长上下文扩展，HELMET-128K / RULER-128K 上接近 full attention baseline |

## 技术身份

M3 的设计主张是"用稀疏注意力作为长上下文部署的一等假设"：和 [GLM-5](glm-5.md) 走 token-level 的 [DSA](../concepts/deepseek-sparse-attention.md)、[DeepSeek-V4](deepseek-v4.md) 走 KV 压缩的 CSA/HCA 相比，MSA 选了 block-level + 每个 GQA group 独立 top-k 的中间路径。块级让 IO 更规整，KV-outer kernel 配合 query gather 把 FLOPs/IO 比从 d 提到 ~⅔·G·d；GQA-group 独立又比"所有 head 共享一个 top-k"保留更多检索多样性。

代价是 attention 模块多两个 idx 投影（idx query / idx key），训练要带 KL 对齐损失 + indexer warmup。论文显式声明 MSA 既能从头原生稀疏预训练（MSA-PT），也能从 GQA full-attention checkpoint 做 near-lossless continued pretraining（MSA-CPT），所以 M3 这条路在工程上对其他 GQA 模型族迁移成本不高。

在 1M context 下 per-token attention FLOPs 较 GQA 降低 28.4×；H800 实测 prefill 14.2×、decode 7.6× 加速。这个数字是 M3 区别于 M2 的核心运营优势：M2 用的是 full GQA + 192K 上下文，靠 [Forge agent-native RL](../concepts/forge-agent-native-rl.md) 和 3-layer MTP 撑长 horizon；M3 把效率收益放在 attention 这一层。

## 与 M2 的对比

| 维度 | MiniMax-M2 / M2.7 | MiniMax-M3 |
| --- | --- | --- |
| 总参 / 激活 | 229.9B / 9.8B | 约 109B / 约 6B |
| 层数 | 62 | 41 |
| Attention | full multi-head + GQA | MSA（block 稀疏 + GQA） |
| 模态 | 文本主线 | 原生多模态（文本 + 图像 + 视频） |
| 长上下文 | 192K native | 128K 扩展后接近 full baseline，主体训练 1M-friendly |
| 主要差异化 | Forge agent-native RL、MTP、self-evolution | 稀疏注意力 kernel、原生多模态预训练 |

二者代表 MiniMax 同时下的两个赌注：M2 押"更小激活 + 更强 agentic RL 系统"，M3 押"稀疏注意力 + 多模态规模化"。

## 如何解读结果

MSA 论文对 M3 的 benchmark 是预训练评测（MMLU、RULER、HELMET、AI2D、VideoMME、TemporalBench 等），而不是 post-training agent benchmark。这意味着 M3 当前公开数据更多反映稀疏注意力对预训练能力的影响，而不是 agentic 表现。把 M3 与 M2.7 直接放进 [agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md)对比为时尚早；后续如果发布 M3-instruct 或 agentic 后训练版本，再补充。

另一点要注意的是 MSA-PT 在多个数学、视觉、视频和长上下文项上略高于 full-attention baseline。论文倾向于解释为"原生稀疏让模型表征更适配稀疏注意力模式"，但也可能是 3T 预算下 full attention baseline 还未充分训练。这是判断 M3 成色时要单独跟踪的不确定性。

## 待追问

- HF 释出版本是否就是 MSA-PT 的 109B 模型，还是经过了进一步 post-training？需要去 HF 页面或 model card 确认（当前 WebFetch 通道无法访问 huggingface.co，只能等本地手工查证）。
- M3 是否有对应的 instruct / RL 版本？如果有，agentic 评测要单独沉淀。
- M3 与 M2 是否会合流？两条路线在 MiniMax 内部资源上的优先级目前不清楚。
- 公开 [MSA kernel](https://github.com/MiniMax-AI/MSA) 已经针对 NVIDIA Blackwell（SM100）重写并支持 FP8 / NVFP4 / FP4，但论文里的速度数字是 H800 上测的——M3 的实际 serving 平台用哪一档？没有官方数据。

## 相关页面

- 来源：[MSA 技术报告](../sources/msa.md)
- 同族模型：[MiniMax-M2 Series](minimax-m2-series.md)
- 机制：[高效长上下文注意力](../concepts/efficient-long-context-attention.md)、[DeepSeek Sparse Attention](../concepts/deepseek-sparse-attention.md)
- 比较：[稀疏注意力机制对比](../comparisons/sparse-attention-mechanisms.md)
