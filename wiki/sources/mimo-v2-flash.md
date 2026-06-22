# MiMo-V2-Flash 技术报告

## 来源

- 原始 PDF：[raw/mimo-v2-flash-2601.02780.pdf](../../raw/mimo-v2-flash-2601.02780.pdf)
- 标题：MiMo-V2-Flash Technical Report
- 版本/日期：arXiv:2601.02780v2，2026-01-08
- 团队：LLM-Core Xiaomi
- 模型页：[MiMo-V2-Flash](../models/mimo-v2-flash.md)

## 核心结论

MiMo-V2-Flash 是一个偏效率的开放权重 MoE 模型，目标是在较小激活参数预算下获得强 reasoning 和 agentic 能力。模型规模为 309B 总参数 / 15B 激活参数，明显小于一些同类开放模型。

核心架构是混合注意力：每个 hybrid block 由 5 个 Sliding Window Attention（SWA）层和 1 个 Global Attention（GA）层交替构成。SWA 窗口为 128 tokens。报告认为该设计在长上下文中可带来接近 6 倍的 KV-cache 与注意力计算降低。

## 架构与训练

MiMo-V2-Flash 共有 48 层，其中 39 层 SWA、9 层 GA。每个 MoE 层有 256 个专家，每 token 激活 8 个，没有 shared experts。预训练总量为 27T tokens，分三阶段：22T general pre-training，4T 加强代码和合成推理数据的 mid-training，以及 1T context extension 到 256K。

模型也使用 [多 token 预测](../concepts/multi-token-prediction.md)。预训练阶段使用一个 MTP 层；后训练阶段可复制成多步 MTP，用于 speculative decoding。每个 MTP block 只有 0.33B 参数。

## 后训练

报告最重要的后训练贡献是 Multi-Teacher On-Policy Distillation（MOPD），见 [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)。MOPD 不采用模型权重合并，也不只依赖离线 teacher 数据；student 以 on-policy 方式采样，再从多个专门 teacher 获得 dense token-level supervision。

> MOPD 走 token-level KL estimate + ORM advantage 加权混合（与 DeepSeek-V4 的 full-vocabulary logit distillation 是同一类目的下的两种 KL 形式取舍）；teacher 类型可以是 RL/SFT/Self 三种。详见 [On-Policy Distillation 跨报告对比](../comparisons/on-policy-distillation.md) 轴二。

MOPD 的目标是缓解能力之间的 "see-saw" 问题：一个能力提升时，另一个能力不应显著退化。报告把它作为融合 reasoning、coding、writing、long-context 和 tool-use 能力的统一框架。

## 评测要点

报告给出的后训练结果包括 SWE-bench Verified 73.4、SWE-bench Multilingual 71.7、BrowseComp 45.4、带 context management 的 BrowseComp 58.3，以及 tau2-Bench 中 Telecom 95.3、Retail 79.5、Airline 66.0。

推理效率方面，3-layer MTP 在 16K 输入 / 1K 输出设置下，根据 batch size 和 acceptance length 的不同，报告测得约 1.86-2.70 倍解码加速。

## 待追问

- 混合 SWA/GA 设计简单高效，但在更强对抗性的长上下文推理中是否稳定？
- MOPD 的收益高度依赖 teacher 质量和任务覆盖；后续应与 GLM-5 的 cross-stage distillation 对照。

