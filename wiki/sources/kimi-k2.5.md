# Kimi K2.5 技术报告

## 来源

- 原始 PDF：[raw/kimi-k2.5-2602.02276.pdf](../../raw/kimi-k2.5-2602.02276.pdf)
- 标题：Kimi K2.5: Visual Agentic Intelligence
- 版本/日期：arXiv:2602.02276v1，2026-02-02
- 团队：Kimi Team
- 模型页：[Kimi K2.5](../models/kimi-k2.5.md)

## 核心结论

Kimi K2.5 把开放 multimodal model 的重点从“能看图”推进到“视觉 agentic intelligence”：模型要在文本、图像、视频、代码、搜索、GUI 和工具环境中统一推理。报告称已发布 post-trained Kimi K2.5 checkpoint。

它建立在 Kimi K2 的 trillion-parameter MoE backbone 上：1.04T 总参数 / 32B 激活参数，384 experts，每 token 激活 8 个专家。K2.5 继续做约 15T 混合视觉-文本 token 的联合训练，并在评测中通常使用 256K context。

## 多模态训练路线

K2.5 的关键经验是 early vision fusion：不要在文本 backbone 末期大量注入视觉 token，而是在整个训练过程中以较低、稳定比例混合文本和视觉 token。视觉编码器为 MoonViT-3D，支持 native-resolution 输入、NaViT-style packing，并把连续 4 帧视频合成一个时空块处理，实现约 4x 时间压缩。

后训练阶段提出 zero-vision SFT：只用文本 SFT 也能激活视觉推理和工具使用。报告认为原因是联合预训练已经建立了强视觉-文本对齐；人工设计的视觉轨迹反而可能损害泛化。随后进行 joint multimodal RL，视觉 RL 不只提升视觉任务，还改善 MMLU-Pro、GPQA-Diamond 和 LongBench v2 等文本 benchmark。

## Agent Swarm

K2.5 的另一个核心贡献是 [Agent Swarm](../concepts/agent-swarm.md)。它用 Parallel-Agent Reinforcement Learning（PARL）训练一个 orchestrator，让它学习何时创建 sub-agent、如何拆分任务、如何并行调度。训练时 sub-agents 冻结，只有 orchestrator 更新，避免端到端多 agent 共同训练中的 credit assignment 混乱。

报告把 Agent Swarm 也解释为一种主动 context management：不同 sub-agent 保持独立工作记忆，只把关键中间结果回传给 orchestrator，从而把长任务拆成并行、局部、有边界的上下文。

## 评测要点

Kimi K2.5 在 SWE-Bench Verified 为 76.8，SWE-Bench Pro public 为 50.7，SWE-Bench Multilingual 为 73.0。BrowseComp 单 agent 为 60.6，使用 discard-all context management 为 74.9，Agent Swarm 为 78.4；WideSearch 从 72.7 提升到 Agent Swarm 的 79.0，并在 WideSearch 上带来约 3-4.5x 执行时间降低。

多模态方面，报告给出 VideoMMMU 86.6、MMVU 80.4、OCRBench 92.3、OSWorld-Verified 63.3 和 WebArena 58.9。

## 待追问

- zero-vision SFT 的成立条件是什么？是否必须先有足够强的 joint pretraining？
- Agent Swarm 的提升有多少来自并行搜索，有多少来自 context sharding？
- 冻结 sub-agent 能稳定训练，但是否限制了长期协作策略的上限？
