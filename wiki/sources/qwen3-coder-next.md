# Qwen3-Coder-Next 技术报告

## 来源

- 原始 PDF：[raw/Cao 等 - 2026 - Qwen3-coder-next technical report.pdf](../../raw/Cao%20%E7%AD%89%20-%202026%20-%20Qwen3-coder-next%20technical%20report.pdf)
- 标题：Qwen3-Coder-Next Technical Report
- 团队：Qwen Team（Alibaba Group），Cao 等，2026
- 配套：base + instruction-tuned 开源权重

## 核心结论

Qwen3-Coder-Next 是一个 **80B 总参 / 3B 激活** 的开源编码模型，明确「based on Qwen3-Next」——即沿用 Qwen3-Next 的 **hybrid attention + MoE** 架构（3 层 Gated DeltaNet 线性层配 1 层带输出门的 gated attention），面向 coding agent 和本地开发。报告主旨不是架构创新，而是**把 agentic training 推到小激活模型上能走多远**：通过大规模合成可验证编码任务 + 可执行环境，用 mid-training 和 RL 直接从环境反馈学习。在 SWE-Bench、Terminal-Bench 2.0、Aider 等 agent-centric benchmark 上，相对其激活参数量取得有竞争力的表现。

> **这页对本 wiki 的价值是「采用证据」**：它确认 [Gated Attention](../concepts/attention-gating.md) 的 SDPA 输出门和 [GDN 线性注意力](../concepts/linear-attention-and-delta-rule.md) 已落到一个真实发布的生产编码模型上。架构细节它直接继承 Qwen3-Next、未重新推导，所以注意力机制的一手出处仍是 [Gated Attention 报告](gated-attention.md) 和 [GDN 报告](gated-delta-net.md)，本报告不构成机制层 tier-1 来源。

## 架构与训练

- **架构（已据 HF config 补全）**：48 层，`full_attention_interval=4` 即 **3 GDN : 1 full-attention**；512 expert 选 10（+1 shared），hidden 2048，head_dim 256，上下文 262,144，纯文本。详见模型页 [Qwen3-Coder-Next](../models/qwen3-coder-next.md)。
- **长上下文**：预训练上下文扩到 **262,144 token**；除标准 next-token，额外用 **fill-in-the-middle（FIM）** 目标（代码编辑关键），并用 best-fit packing（BFP）避免拼样本时的 context hallucination 和 head-side 截断。
- **训练重点**：大规模合成可验证编码任务 + 可执行环境，mid-training + RL 从环境反馈学习，覆盖长 horizon 多轮 agentic trajectory。
- **部署**：从领域专家模型蒸馏，合成单一统一部署模型。

## 与已有沉淀的关系

- 与 [Qwen3.5-Omni](qwen3.5-omni.md) 同属 Qwen3-Next/Qwen3.5 这条 hybrid 谱系，共享 GDN + gated attention 设计，区别只在领域（编码 agent vs 全模态）和训练配方。
- 它和 [Kimi Linear](kimi-linear.md) 是同尺度（80B-A3B vs 48B-A3B）、同思路（线性层 + 少量全局层）但不同全局层选择的对照：Qwen3-Next 系的全局层是**带输出门的 full attention**，Kimi Linear 把它换成了 **MLA**（见 [线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md) 的跨报告信号）。
- agentic coding 训练这一支与已有的 [Agentic engineering](../concepts/agentic-engineering.md)、[Agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md) 相关。

## 待追问

- ~~具体层数、GDN:attention 比例~~ **已据 HF config 坐实**：48 层、`full_attention_interval=4`（3 GDN : 1 full-attention）、512 expert 选 10，见模型页 [Qwen3-Coder-Next](../models/qwen3-coder-next.md)。
- 80B-A3B 在 SWE-Bench 上的确切分数与对照模型，可作为「小激活 + 强 agentic 训练」主张的证据补充。
- 它和主线 [Kimi K2.5](kimi-k2.5.md)、GLM-5 等更大 agentic 模型在编码 benchmark 上的可比性，需要统一口径才能对比。

## 相关页面

- 模型：[Qwen3-Coder-Next](../models/qwen3-coder-next.md)
- 概念：[注意力门控](../concepts/attention-gating.md)、[线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md)
- 同家族来源：[Qwen3.5-Omni](qwen3.5-omni.md)、[Gated Attention](gated-attention.md)、[Gated DeltaNet](gated-delta-net.md)
