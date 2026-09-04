---
type: Comparison
title: "2026 前沿模型技术报告对比"
description: "GLM-5、Macaron-V1、MiMo-V2-Flash、DeepSeek-V4、MiniMax-M2、Kimi 等 2026 前沿模型技术报告的横向比较。"
tags: ["comparison", "2026-open-model-technical-reports"]
timestamp: 2026-06-06
---

# 2026 前沿模型技术报告对比

## 范围

本页比较当前沉淀的报告：[GLM-5](../sources/glm-5.md)、[Macaron-V1](../sources/macaron-v1.md)、[GLM-5V-Turbo](../sources/glm-5v-turbo.md)、[MiMo-V2-Flash](../sources/mimo-v2-flash.md)、[DeepSeek-V4](../sources/deepseek-v4.md)、[MiniMax-M2 Series](../sources/minimax-m2-series.md)、[Kimi K2.5](../sources/kimi-k2.5.md)、[Kimi K3](../sources/kimi-k3.md)、[Intern-S2-Mobius](../sources/intern-s2-mobius.md)、[Gemma 4](../sources/gemma-4.md)、[Laguna](../sources/laguna-m1-xs2.md)、[Seed2.0](../sources/seed2.md) 和 [Qwen3.8-Flash-Next](../sources/qwen3.8-next.md)。

注：Seed2.0 是 Model Card 而非技术报告，不含架构/训练/参数量信息，因此下表对应列为空白。其价值在部署洞察和评测框架，见 [Seed2.0 Model Card](../sources/seed2.md)。

## 对比表

| 模型 | 主要目标 | 规模 | 上下文 | 注意力策略 | 后训练重点 |
| --- | --- | --- | --- | --- | --- |
| GLM-5 | Agentic engineering 与 ARC 能力 | 744B / 40B active | SFT 到 202,752 tokens | MLA backbone 上的 DSA | 异步 agent RL、reasoning RL、general RL、cross-stage distillation |
| Macaron-V1 | 将 agent 变成可版本化 model–harness system | Venti：744B GLM-5.2 base + 4 LoRA；Tall：35B-A3B base + 4 LoRA | Venti 长上下文 serving；Tall 未披露 native 值 | frozen base 上按 user turn 路由一个 LoRA（MoL），per-adapter own-view KV reuse | HCP 配置搜索 + MindForge RSI lineage + frozen-base LoRA GRPO；跨代增益尚未实证 |
| GLM-5V-Turbo | Native multimodal agent（感知即推理） | 未披露 | 未披露 | CogViT + MMTP + GLM-5-Turbo backbone | 30+ 类别多模态联合 RL、agent 框架集成（Claude Code / AutoClaw） |
| MiMo-V2-Flash | 在紧凑激活规模下获得快速 reasoning 与 agentic 能力 | 309B / 15B active | 32K native，256K extended | 128-token SWA 的 5:1 hybrid SWA/GA | MOPD multi-teacher on-policy distillation |
| DeepSeek-V4 | 高效百万 token 上下文智能 | Flash 284B / 13B active；Pro 1.6T / 49B active | 1M native target | hybrid CSA/HCA compressed attention | reasoning modes、tool-use formats、超长上下文 RL/OPD 基础设施 |
| MiniMax-M2 / M2.7 | 低激活 MoE 的真实 agent 任务能力 | 229.9B / 9.8B active | 192K native | full attention with GQA | Forge agent-native RL、interleaved thinking、self-evolution |
| Kimi K2.5 | 视觉 agentic intelligence 与并行 agent 编排 | 1.04T / 32B active | 评测常用 256K | Kimi K2 MoE + MoonViT-3D | zero-vision SFT、joint multimodal RL、PARL Agent Swarm |
| Kimi K3 | 首个开源 3T 级 frontier 模型 | 2.78T / 104B active | 1M native | Hybrid KDA–MLA（3:1）+ Attention Residuals + Stable LatentMoE | 3-stage SFT→9-专家 RL→MOPD、Unified White-Box RL Env、AgentENV microVM 沙箱、QAT + MTP→EAGLE-3 |
| Intern-S2-Mobius | 知识存储与组合推理解耦，压缩外显 CoT | 35B 级（由 Qwen3.5-35B-A3B 转换；细节未披露） | 未披露 | Shared FFN Memory + Self-Attn Reasoner；Memory 分块 sparse activation | SFT + RL 仅被列出、无算法或数据细节；主实证是 1T-token continued pre-training 后的效率 / 长度对比 |
| Gemma 4 | 效率导向的多模态 dense + MoE | E2B 2.3B / E4B 4.5B / 12B / 26B-A4B / 31B | 128K+ | 5:1 hybrid SWA/GA + key-as-value + p-RoPE + KV Sharing | thinking mode、QAT 量化、MTP drafter |
| Seed2.0 | 面向大规模生产部署的多模态模型族 | 未披露（Pro/Lite/Mini 三档） | 未披露 | 未披露 | 未披露（Model Card 不含训练细节） |
| Ling-2.6 / Ring-2.6 | 万亿参数 agentic 双线（instant + thinking） | flash ~104B/~5B；1T ~1T/~8B | 256K | 7:1 Lightning Attention + MLA（retrofit from GQA） | Ling: specialization-then-distillation + token efficiency；Ring: KPop RL + 异步 RL |
| Laguna M.1 / XS.2 | 长周期 agentic coding + Model Factory 工业化流程 | M.1 225.8B/23.4B；XS.2 33.4B/3B | 256K（RoPE scale 翻倍无训练） | M.1 逐层 GA；XS.2 3:1 SWA/GA + softplus per-head gating | mid-train→SFT→agentic RL(CISPO)；AutoMixer 数据混合；合成代码环境贯穿 SFT/RL |
| Qwen3.8-Flash-Next | 用更小激活预算保住 397B-A17B 级预训练质量 | 125B / 6B（+51B 主机 n-gram） | CPT 256K；QSA 评到 1M | 3:1 GDN + QSA（全局层 CPT 稀疏化）+ Gated Residual | 本报告只有 base；Muon 预训练，无 SFT/RL 配方 |

## 主综合

GLM-5 最明确地提出 agentic engineering。MiMo-V2-Flash 最强调紧凑规模和部署速度。DeepSeek-V4 最激进地推进长上下文系统设计。MiniMax-M2 把重点放在低激活 MoE 如何通过 Forge、数据和系统栈追赶 frontier agent 能力。Kimi K2.5 则把多模态和多 agent 并行纳入同一 agentic 叙事。

这些报告共同说明：前沿模型竞争不再只看 benchmark 分数。关键差异正在转向注意力效率、激活参数预算、后训练基础设施、多模态对齐、agent harness，以及运行长 agent trajectory 的能力。

## 实用阅读顺序

1. 先读 [MoE 前沿模型扩展](../concepts/moe-frontier-model-scaling.md)，理解总参数与激活参数的差异。
2. 再读 [高效长上下文注意力](../concepts/efficient-long-context-attention.md)，比较上下文架构路线。
3. 然后读 [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)，比较 RL、蒸馏和 agent orchestration。
4. 最后读 [Agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md)，避免误读 benchmark。

## 深入阅读路径

如果要看机制细节，可以按下面顺序继续：

1. 长上下文机制：[DeepSeek Sparse Attention](../concepts/deepseek-sparse-attention.md) -> [百万 token 上下文服务](../concepts/million-token-context-serving.md)。
2. 后训练机制：[异步 Agent RL](../concepts/asynchronous-agent-rl.md) -> [Multi-Teacher On-Policy Distillation](../concepts/multi-teacher-on-policy-distillation.md) -> [Forge Agent-Native RL](../concepts/forge-agent-native-rl.md)。
3. 并行与多模态 agent：[Agent Swarm](../concepts/agent-swarm.md) -> [多模态 Agentic 训练](../concepts/multimodal-agentic-training.md)。
4. 评测解释：[Agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md) -> [Agent harness](../concepts/agent-harness.md)。

## 二版综合

五篇报告的差异可以浓缩成五种工程哲学：

- GLM-5 认为 agentic 能力来自训练环境和异步 RL 基础设施，模型架构服务于长周期 agent rollout。
- Macaron-V1 认为持续学习的最小可审计单元是 model–harness pair：MoL 使 specialist 权重可独立升级，HCP 把 runtime 写成可复放配置，MindForge 将它们同 trajectory/evaluation 连成 lineage。其证据目前只到 single snapshot 和 frozen-model harness search，不能把「系统为持续学习而设计」读成「已证明跨代进步」。
- MiMo-V2-Flash 认为紧凑 MoE、简单长上下文架构和 MOPD 可以在较小激活预算下接近大模型能力。
- DeepSeek-V4 认为百万 token 上下文需要从 attention、KV-cache、QAT、teacher scheduling 到 fault tolerance 全栈重构。
- MiniMax-M2 认为低激活 MoE 可以通过高可信 agent 数据、Forge RL 和 self-evolution scaffold 获得真实任务能力。
- Kimi K2.5 认为视觉-文本联合训练与并行 sub-agent 编排可以共同提升 agentic 工作流。
- Intern-S2-Mobius 认为长 CoT 的成本不只能靠 decoder 或 attention kernel 优化：若把 FFN 知识从层绑定中释放成共享 Memory，Reasoner 可以在 latent state 中多轮取用知识、再用更短的可见 trace 输出。现有证据限于 7B 配对预训练与一个从 Qwen3.5 转换的 35B 路线；共享 Memory 的实际 retrieval 成本、latent iteration 的因果作用及公平的 continued-pretraining 对照尚未披露。
- Gemma 4 认为效率优先于规模--用 SWA/GA 混合 + KV 侧压缩（而非内容稀疏）即可在 128K 级上下文达到竞争力，用 encoder-free 架构消除多模态的编码器开销，用 QAT + MTP drafter 把部署成本压到端侧可承受。
- Seed2.0 认为 model card 的价值在于真实部署洞察和评测框架而非架构披露--MaaS 使用分布揭示前端开发和 bug fixing 占 agentic coding 主导，四维评测框架（Science Discovery / Vibe Coding / Context Learning / Real-World Tasks）把"real-world complexity"操作化，定价比 frontier 模型低一个数量级以推动大规模企业部署。
- GLM-5V-Turbo 认为多模态感知是 agent 能力的核心组件而非辅助接口--CogViT + MMTP + 30+ 类别多模态联合 RL，配合三个 design lens（感知是天花板 / 分层优化 / 清晰规格+可靠验证+受控评测），把 agentic engineering 从文本扩展到图像、视频、GUI、文档、网页。加视觉未侵蚀文本 coding（CC-Backend/CC-RepoExploration 反超纯文本基座 GLM-5-Turbo）。
- Ling-2.6 / Ring-2.6 认为**不必从头训练也能换架构**——从已投入 20T tokens 的 GQA checkpoint 经四阶段 smooth retrofit 迁移到 7:1 hybrid linear attention + MLA，配合 token efficiency 后训练（~4× token efficiency）和 KPop 异步 agentic RL，在万亿参数规模同时拿下效率、token 质量和 agentic 能力。其双线设计（Ling instant / Ring thinking）把"快速响应"和"深度推理"解耦为两个优化目标。
- Kimi K3 认为**开源前沿要同时 push 两条 scaling 轴**——预训练 foundation 扩到 3T 级（2.8T/104B active，首个开源 3T）+ RL/reasoning effort/long-horizon interaction 推到 1M context。架构沿序列（KDA scaled sigmoid + Gated MLA NoPE）、深度（Attention Residuals）、宽度（Stable LatentMoE：LatentMoE + SiTU-GLU + Quantile Balancing）三轴重新设计，综合拿到 2.5× scaling efficiency。后训练 9-专家 RL（3 域 × 3 effort）+ MOPD 融合 + Unified White-Box RL Env（harness-agnostic）+ AgentENV microVM 沙箱支撑 1M partial rollout。仍承认落后 Claude Fable 5 / GPT-5.6 Sol，但 WebDev Arena 首个开源登顶、多项 benchmark #1。
- Laguna 认为**模型开发流程本身是 frontier 竞争的主杠杆**——不追绝对规模（M.1 225.8B/23.4B、XS.2 33.4B/3B 都非最大），而把 Model Factory 工业化流程当核心：M.1 预训练结束后五周从零交付 XS.2，四项架构改动（3:1 SWA/GA、WSD、expert 调制、dense 层 3→1）全靠 16B proxy 消融后翻配置 flag。把 [AutoMixer 数据混合优化](../concepts/data-mixture-optimization.md) + [Gated Attention](../concepts/attention-gating.md) + CISPO RL + 合成代码环境等已有成果组合进可配置流水线，XS.2 以 3B 激活在 SWE-bench Verified 73.4 略胜 Qwen3.6-35B-A3B。是「工艺→工业」转型在 agentic coding 域的干净样本。
- Qwen3.8-Flash-Next 认为**架构、效率和优化是同一个设计问题**：3:1 GDN 混合在 CPT 把全局层换成 QSA，残差加宽成 GR，容量加在主机 n-gram 上，Muon 把最优 LR/batch 上推。Headline 是 6B 激活、约 1/9 FLOPs 在 14 项 base 上 8 胜 6 负追平 397B/17B。它几乎不谈 agent 后训练；价值在预训练配方和「loss 与下游会分叉」的消融纪律。

这意味着后续比较不应只看 SWE-bench 或 HLE 分数，而要比较“模型 + agent harness + context strategy + serving system”的整体能力。

[Prime Agent](../sources/prime-agent.md) 不是模型报告，但把最后这一句做成了可检查的评测膜：固定 Opus 5，官方 ARC harness 30.2% vs Prime Agent 95.5% RHAE。作者同时写明 native-harness 复跑低于官方分，所以这是 situating 而不是已隔离因果。它与 Macaron 的 HCP 搜索一起说明：2026 的 agent 分数里，harness 已经不能再当脚注。细讲见 [Agent harness](../concepts/agent-harness.md)。

## 后续问题

- 对 coding agent 来说，DSA、hybrid SWA/GA、CSA/HCA 哪条路线的成本/能力比最好？
- 不同 agent framework 下报告的 SWE-bench 和 agent benchmark 可比性有多强？
- MOPD 式 teacher integration 能否与 GLM 式异步 agent RL 结合？
- Forge 这类训练系统能否与 Agent Swarm 这类运行时并行策略结合？
- 多模态 RL 带来的文本能力提升，是真实迁移，还是评测分布和 reward 设计的副作用？
