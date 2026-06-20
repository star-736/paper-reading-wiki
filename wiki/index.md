# 索引

这是论文阅读知识库的入口页。后续查询先读这里，再进入相关的来源页、模型页、概念页或比较页。

## 来源

- [GLM-5 技术报告](sources/glm-5.md) - GLM-5 的 arXiv 技术报告，重点是 agentic engineering、DSA 和异步 RL。
- [MiMo-V2-Flash 技术报告](sources/mimo-v2-flash.md) - MiMo-V2-Flash 的 arXiv 技术报告，重点是混合 SWA/GA 注意力、MOPD 和 MTP 加速。
- [DeepSeek-V2 技术报告](sources/deepseek-v2.md) - DeepSeek-V2 的 arXiv 论文，Multi-Head Latent Attention（MLA）的首次提出处。
- [DeepSeek-V3.2 技术报告](sources/deepseek-v32.md) - DeepSeek-V3.2 的 arXiv 论文，引入 DeepSeek Sparse Attention（DSA）。
- [DeepSeek-V4 技术报告](sources/deepseek-v4.md) - DeepSeek-V4 的 Hugging Face 官方 PDF，重点是百万 token 上下文效率。
- [MiniMax-M2 Series 技术报告](sources/minimax-m2-series.md) - MiniMax-M2/M2.7 的 arXiv 技术报告，重点是低激活 MoE、Forge RL 和 self-evolution。
- [Kimi K2.5 技术报告](sources/kimi-k2.5.md) - Kimi K2.5 的 arXiv 技术报告，重点是视觉 agentic intelligence、zero-vision SFT 和 Agent Swarm。
- [MSA 技术报告](sources/msa.md) - MiniMax Sparse Attention 的 arXiv 报告，GQA-block 级稀疏 + 每个 group 独立 top-k，1M context 下 14× prefill / 7× decode。
- [IndexCache 技术报告](sources/indexcache.md) - 清华 + Z.ai 在 DSA 上做跨层索引复用，1/4 retention 即可在 30B 和 GLM-5 上保留质量并拿到端到端加速。

## 模型

- [GLM-5](models/glm-5.md) - 744B 总参数 / 40B 激活参数的 MoE 模型，定位在 agentic、reasoning、coding 能力。
- [MiMo-V2-Flash](models/mimo-v2-flash.md) - 309B 总参数 / 15B 激活参数的 MoE 模型，优化快速推理和 agentic 工作负载。
- [DeepSeek-V4](models/deepseek-v4.md) - 包含 DeepSeek-V4-Flash 和 DeepSeek-V4-Pro 的模型族，目标是原生 1M token 上下文。
- [MiniMax-M2 Series](models/minimax-m2-series.md) - 229.9B 总参数 / 9.8B 激活参数的低激活 MoE agentic 模型系列。
- [MiniMax-M3](models/minimax-m3.md) - 428B 总参数 / 22B 激活参数（+ 600M visual encoder）的原生 MSA 多模态 MoE 模型，配套 MSA 报告释出。
- [Kimi K2.5](models/kimi-k2.5.md) - 1.04T 总参数 / 32B 激活参数的 multimodal agentic MoE 模型，强调 Agent Swarm。

## 概念

- [Agentic engineering](concepts/agentic-engineering.md) - 这些报告如何定义长周期软件工程和工具使用任务。
- [高效长上下文注意力](concepts/efficient-long-context-attention.md) - DSA、混合 SWA/GA、CSA 和 HCA 的对比。
- [Agentic 模型的后训练](concepts/post-training-for-agentic-models.md) - 面向 agent 的 RL、MOPD 和蒸馏模式。
- [多 token 预测](concepts/multi-token-prediction.md) - MTP 作为训练目标和 speculative decoding 机制。
- [MoE 前沿模型扩展](concepts/moe-frontier-model-scaling.md) - 多篇报告中的总参数、激活参数和系统成本对比。

## 细讲模块

- [DeepSeek Sparse Attention](concepts/deepseek-sparse-attention.md) - DSA 的长上下文稀疏选择、GLM-5 中的训练方式和 RL 稳定性问题。
- [Multi-Head Latent Attention](concepts/multi-head-latent-attention.md) - MLA 的「减头 vs 压秩」定位、MHA/MQA 两种 mode，以及 DSA / CSA 为何架在它的 MQA mode 上。
- [异步 Agent RL](concepts/asynchronous-agent-rl.md) - GLM-5 如何用异步 rollout、TITO 和 token-level clipping 训练 agent。
- [Multi-Teacher On-Policy Distillation](concepts/multi-teacher-on-policy-distillation.md) - MiMo-V2-Flash 的 MOPD 范式及其与 DeepSeek-V4 OPD 的关系。
- [百万 token 上下文服务](concepts/million-token-context-serving.md) - DeepSeek-V4 的异构 KV-cache、on-disk cache 和 shared-prefix reuse。
- [Agentic 评测体系](concepts/agentic-evaluation-benchmarks.md) - SWE-bench、Terminal-Bench、BrowseComp、MCP-Atlas 等 benchmark 的作用和可比性风险。
- [Forge Agent-Native RL](concepts/forge-agent-native-rl.md) - MiniMax-M2 如何把 agent harness、RL 训练、长上下文 rollout 和 serving 加速解耦。
- [Agent Swarm](concepts/agent-swarm.md) - Kimi K2.5 的 PARL 并行 agent 编排，以及 context sharding 解释。
- [多模态 Agentic 训练](concepts/multimodal-agentic-training.md) - Kimi K2.5 的 early vision fusion、MoonViT-3D、zero-vision SFT 和 joint multimodal RL。
- [跨层索引复用](concepts/cross-layer-index-reuse.md) - IndexCache、Kascade、HySparse 等如何让多数层共用 anchor 层选好的 top-k 索引。

## 比较

- [2026 前沿模型技术报告对比](comparisons/2026-open-model-technical-reports.md) - GLM-5、MiMo-V2-Flash、DeepSeek-V4、MiniMax-M2 和 Kimi K2.5 的横向比较。
- [稀疏注意力机制对比](comparisons/sparse-attention-mechanisms.md) - DSA、MSA、NSA、MoBA、CSA/HCA、IndexCache 等沿"粒度 / 跨头共享 / 跨层共享"三轴的对比。
