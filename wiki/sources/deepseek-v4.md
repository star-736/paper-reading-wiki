# DeepSeek-V4 技术报告

## 来源

- 原始 PDF：[raw/deepseek-v4-hf-technical-report.pdf](../../raw/deepseek-v4-hf-technical-report.pdf)
- 标题：DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
- 来源说明：Hugging Face 官方技术报告 PDF；初次下载时未找到 arXiv 条目。
- 团队：DeepSeek-AI
- 模型页：[DeepSeek-V4](../models/deepseek-v4.md)

## 核心结论

DeepSeek-V4 是一个预览版模型族，主线是原生 1M token 上下文效率。报告覆盖 DeepSeek-V4-Flash（284B 总参数 / 13B 激活参数）和 DeepSeek-V4-Pro（1.6T 总参数 / 49B 激活参数）。

报告的核心判断是：长上下文扩展不仅是能力问题，更是效率和系统问题。DeepSeek-V4 直接瞄准 attention FLOPs 和 KV-cache 瓶颈，使长周期任务、大规模跨文档分析和 test-time scaling 更可行。

## 架构与训练

DeepSeek-V4 继承 DeepSeekMoE 和 MTP，并加入混合 [高效长上下文注意力](../concepts/efficient-long-context-attention.md)：Compressed Sparse Attention（CSA）和 Heavily Compressed Attention（HCA）。CSA 压缩 KV entries 后进行 DSA 风格稀疏选择；HCA 使用更重的 KV 压缩，并在压缩状态上做密集注意力。

其他关键组件包括 Manifold-Constrained Hyper-Connections（mHC）、Muon optimizer、针对 routed experts 和 CSA indexer QK path 的 FP4 quantization-aware training，以及支持 on-disk shared-prefix reuse 的异构 KV-cache 管理。

DeepSeek-V4-Flash 训练 32T tokens，DeepSeek-V4-Pro 训练 33T tokens。训练从 4K 序列长度开始，逐步扩展到 16K、64K 和 1M；稀疏注意力在 dense warmup 后引入。

## 效率主张

在 1M context 下，报告称 DeepSeek-V4-Pro 只需要 DeepSeek-V3.2 的 27% single-token inference FLOPs 和 10% KV cache；DeepSeek-V4-Flash 进一步降低到 10% FLOPs 和 7% KV cache。相对 BF16 GQA8 注意力基线，报告称 1M context 下 KV cache 可降到约 2%。

## 评测要点

报告称 DeepSeek-V4-Flash-Base 在更少总参数和激活参数下，超过 DeepSeek-V3.2-Base 的多数 benchmark。DeepSeek-V4-Pro-Base 被定位为 DeepSeek 系列最强 foundation model，在知识、推理、代码和长上下文任务上全面提升。DeepSeek-V4-Pro-Max 被定位为开放模型中的 SOTA reasoning 模式，并具备较强 agentic 能力。

## 待追问

- 架构能力强但复杂；报告本身也把架构简化列为未来方向。
- 当前来源不是 arXiv，后续如果出现 arXiv 版本或新 revision，需要更新来源页和日志。

