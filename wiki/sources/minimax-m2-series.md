---
type: Source
title: "MiniMax-M2 Series 技术报告"
description: "MiniMax-M2/M2.7 的 arXiv 技术报告，重点是低激活 MoE、Forge RL 和 self-evolution。"
tags: ["source", "minimax-m2-series"]
timestamp: 2026-06-06
resource: "../../raw/minimax-m2-series-2605.26494.pdf"
---

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

### 专家偏置的联合优化与证据边界

**已据原文核实（`supported`）**：§2.1 与 §2.2.1 “Expert Bias” 使用 sigmoid gating，并把 expert-specific bias 作为各专家 routing score 的偏移。关键句是 “These biases are optimized jointly with model parameters”；下一页接着说明，这使辅助负载均衡损失可以大幅降低。因此报告描述的是**与模型参数联合优化的可学习偏置**，不只是一个未定义的 learnable 标签。

**不能等同于 Loss-Free Balancing 的 Algorithm 1**（`refuted`，针对“原样采用该更新规则”的归属）：[Loss-Free Balancing](loss-free-balancing.md) 的 bias 按历史 token 负载更新，只影响 top-k 选择，不进入专家输出权重；M2 引用这篇论文，却没有声明复用该 sign 更新。引用关系不等于实现完全相同，见 [负载均衡谱系](../concepts/moe-load-balancing.md)。

**仍有实现边界**：联合优化是报告明文，但本报告未给 bias 的梯度路径、是否进入选中专家的输出权重、优化器参数组或独立学习率，也未给剩余 auxiliary loss 的系数。“大幅降低”不能写成“完全移除”；本页不把论文文字冒充训练代码核验。Table 1 消融的是 MTP / fine-grained experts，并非 bias 更新规则的独立对照。


M2 还使用 [多 token 预测](../concepts/multi-token-prediction.md)。预训练阶段先训练单个 MTP module，继续预训练衰减阶段通过权重复制扩展到 3 个 MTP modules，并在推理中作为 speculative decoding draft path。

## 后训练与系统

后训练数据覆盖 chat、reasoning、code、cowork，并强调每条轨迹要有可执行环境、verifiable reward 或 judge evidence。SFT 阶段训练 interleaved thinking，使模型在推理、工具调用、观察反馈之间交替，并把 thinking state 保留到后续轮次。

Forge 把 agent RL 建模为：LLM completion 是 action，工具、context management、memory、agent harness 都属于环境。系统上，Forge 将 Agent Side、Training/Inference Side 和 Gateway/Data Pool 中间层解耦，支持 white-box agents 和 black-box/API-only agents。为了适应长尾 agent rollout，它使用 windowed-FIFO 调度、prefix tree merging、MTP co-training、prefill-decode disaggregation 和 global KV cache pool。

## 评测要点

报告重点展示 M2.7。代表性结果包括 SWE-bench Pro 56.2、SWE-bench Multilingual 76.5、Multi-SWE-bench 52.7、Terminal-Bench 2.0 57.0、BrowseComp 77.8、GDPval-AA 50.0、Toolathlon 46.3、AIME 2026 94.2 和 GPQA-Diamond 89.8。

这些数字应理解为“模型 + interleaved thinking + agent scaffold + 工具环境”的整体结果，而不只是裸模型能力。

## 待追问

- **需实验或作者披露**：M2.7 的提升中，数据管线、Forge RL、interleaved thinking 和 self-evolution 各自贡献多少？
- **需实验或作者披露**：full attention 在 192K context 下成本很高；低激活 MoE 是否足以抵消长上下文 attention 的部署压力？
- **需实验或作者披露**：black-box agent 支持是否会成为不同 agent 框架迁移 RL 的通用接口？
