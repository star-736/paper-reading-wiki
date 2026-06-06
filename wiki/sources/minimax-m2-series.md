# MiniMax-M2 Series 技术报告

## 来源

- 原始 PDF：[raw/minimax-m2-series-2605.26494.pdf](../../raw/minimax-m2-series-2605.26494.pdf)
- 标题：The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence
- 版本/日期：arXiv:2605.26494v1，2026-05-26
- 团队：MiniMax
- 模型页：[MiniMax-M2 Series](../models/minimax-m2-series.md)

## 核心结论

MiniMax-M2 Series 的主张是“低激活参数也能释放真实世界智能”。旗舰 M2 backbone 为 229.9B 总参数 / 9.8B 激活参数，使用 62 层 decoder-only Transformer、256 个细粒度专家、每 token 激活 8 个专家、sigmoid gating、GQA full attention，并原生支持 192K context。预训练规模为 29.2T tokens。

报告真正强调的不是单次聊天能力，而是从 M2 到 M2.5、M2.7 的 agentic 能力演进：高可信 agent 数据、[Forge agent-native RL](../concepts/forge-agent-native-rl.md)、interleaved thinking，以及 M2.7 的初步 self-evolution。

## 架构与训练

M2 采用 full attention，而不是沿用 MiniMax-Text-01 中的 hybrid attention。报告中的消融显示，SWA 在部分短任务上可行，但在长上下文 agent、复杂检索和多跳任务上会损失明显，因此 M2 在前沿规模上保留 full attention。

M2 还使用 [多 token 预测](../concepts/multi-token-prediction.md)。预训练阶段先训练单个 MTP module，继续预训练衰减阶段通过权重复制扩展到 3 个 MTP modules，并在推理中作为 speculative decoding draft path。

## 后训练与系统

后训练数据覆盖 chat、reasoning、code、cowork，并强调每条轨迹要有可执行环境、verifiable reward 或 judge evidence。SFT 阶段训练 interleaved thinking，使模型在推理、工具调用、观察反馈之间交替，并把 thinking state 保留到后续轮次。

Forge 把 agent RL 建模为：LLM completion 是 action，工具、context management、memory、agent harness 都属于环境。系统上，Forge 将 Agent Side、Training/Inference Side 和 Gateway/Data Pool 中间层解耦，支持 white-box agents 和 black-box/API-only agents。为了适应长尾 agent rollout，它使用 windowed-FIFO 调度、prefix tree merging、MTP co-training、prefill-decode disaggregation 和 global KV cache pool。

## 评测要点

报告重点展示 M2.7。代表性结果包括 SWE-bench Pro 56.2、SWE-bench Multilingual 76.5、Multi-SWE-bench 52.7、Terminal-Bench 2.0 57.0、BrowseComp 77.8、GDPval-AA 50.0、Toolathlon 46.3、AIME 2026 94.2 和 GPQA-Diamond 89.8。

这些数字应理解为“模型 + interleaved thinking + agent scaffold + 工具环境”的整体结果，而不只是裸模型能力。

## 待追问

- M2.7 的提升中，数据管线、Forge RL、interleaved thinking 和 self-evolution 各自贡献多少？
- full attention 在 192K context 下成本很高；低激活 MoE 是否足以抵消长上下文 attention 的部署压力？
- black-box agent 支持是否会成为不同 agent 框架迁移 RL 的通用接口？
