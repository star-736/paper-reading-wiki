---
type: Concept
title: "Agentic 模型的后训练"
description: "面向 agent 的 RL、MOPD、蒸馏、LLM RL policy optimization 与 ARPO 这类 step-level rollout 采样模式。"
tags: ["concept", "post-training-for-agentic-models"]
timestamp: 2026-06-06
---

# Agentic 模型的后训练

## 为什么后训练成为主战场

Base model 给出知识、代码和推理基础，但 agentic model 还需要在环境中行动：调用工具、观察反馈、调整计划、处理失败、跨多轮保留状态。当前沉淀的报告都把后训练作为把 base model 变成 agentic model 的关键路径。

## GLM-5：异步 Agent RL

[GLM-5](../models/glm-5.md) 的后训练流水线包括 SFT、reasoning RL、agentic RL、general RL 和 on-policy cross-stage distillation。它最值得沉淀的是 [异步 Agent RL](asynchronous-agent-rl.md)。

长周期 agent rollout 有明显长尾：某些任务会跑很久，导致同步 RL 中大量 GPU 等待最慢样本。GLM-5 把 rollout inference 和 training engine 放到不同 GPU 设备上，inference engine 持续生成轨迹，达到阈值后送给 training engine 更新。为了控制异步带来的 off-policy 问题，它使用 rollout log-prob、token-level clipping、stale sample dropping 和 optimizer reset 等机制。

## MiMo-V2-Flash：MOPD

[MiMo-V2-Flash](../models/mimo-v2-flash.md) 的核心是 [Multi-Teacher On-Policy Distillation](multi-teacher-on-policy-distillation.md)。它先训练多个 domain-specialized teachers，例如数学、代码、搜索、工具使用和安全；然后让 student 从自己的分布采样，并从对应 teacher 得到 token-level KL 奖励。

这和普通蒸馏的差别在于：student 学的是自己真实会生成的轨迹，而不是离线 teacher 数据。因此 MOPD 试图减少 exposure bias，并缓解 sequential training 中常见的 capability see-saw。

## DeepSeek-V4：多 reasoning mode 与 OPD

[DeepSeek-V4](../models/deepseek-v4.md) 的后训练强调三种 reasoning effort：Non-think、Think High、Think Max。不同模式使用不同 RL 配置、length penalty 和 context window。报告还使用多 teacher On-Policy Distillation（OPD）把十多个 domain expert 合并到统一模型中，并为了稳定性采用 full-vocabulary logit distillation。

DeepSeek-V4 的特别之处是把后训练和百万 token 基础设施绑在一起：OPD、RL rollout、teacher scheduling、FP4 QAT、token-granular WAL 都是为了让超长上下文下的训练和服务可执行。

## MiniMax-M2：Forge 与 Interleaved Thinking

[MiniMax-M2 Series](../models/minimax-m2-series.md) 的后训练先用 SFT 学 interleaved thinking：模型在 reasoning、tool call、tool observation 之间交替，并保留跨轮 thinking state。这样 agent 在长任务中可以复用先前假设和调试线索，而不是每一轮重新推理。

RL 阶段由 [Forge](forge-agent-native-rl.md) 承载。Forge 把工具、context management、memory 和 agent harness 都视为环境，只把 LLM completion 视为 action。奖励不仅包含最终任务质量，还包含 process reward 和 relative completion-time reward，因此训练目标同时覆盖正确性、过程质量和执行效率。

## Kimi K2.5：Joint Multimodal RL 与 PARL

[Kimi K2.5](../models/kimi-k2.5.md) 的后训练有两个特殊点。第一是 [多模态 agentic training](multimodal-agentic-training.md)：zero-vision SFT 之后做 joint text-vision RL，让视觉任务和文本任务共同优化。第二是 [Agent Swarm](agent-swarm.md) 的 PARL：冻结 sub-agents，只训练 orchestrator 学会创建、调度和聚合并行任务。

PARL 的辅助奖励先鼓励 parallel exploration 和 sub-agent 完成率，随后退火到 0，让最终策略回到任务质量。这相当于把“是否并行、怎样并行”也变成 RL 可学习的 agent 行为。

## Kimi K3：3-stage + White-Box RL Environment + 1M Agentic RL

[Kimi K3](../models/kimi-k3.md) 的后训练是三阶段范式：**SFT → RL（9 专家）→ MOPD 融合**，三处独特：

**1. RL 跨 3 域 × 3 reasoning effort = 9 专家**。不按单任务训专用模型，而是跨三大域（general tasks / general agents / coding agents）× 三 reasoning effort（`{low, high, max}`）训 9 个专家。Reasoning Effort RL 用 per-problem budget control——每问题给初始 token budget `b0(x)`（cold-start 估），轨迹总 token `T(y) > τ·b0(x)` 则 reward override 为 -1；先训 max-budget 变体（大 τ 但封顶防 overthinking），再退火 τ 得 high/low 专家。Agentic Generative Reward Model（GRM）用 tournament-style binary comparison + 强制四步协议（读输出→生成 rubric→按 rubric 打分→记 scorepad）+ verbosity control 防 reward hacking 向冗长漂移。

**2. Unified White-Box RL Environment**。把 agent harness 表示成可配置可组合模块集合（tool interfaces / system prompts / context management / skills / memories / subagents）。配置化实例化主流 harness（Kimi Code / Claude Code / Codex / OpenClaw / Hermes）及全新 harness。RL 训练时为不同任务组动态构造不同 harness 配置，**避免过拟合单一 harness 的 tool schema / system prompt / context 管理机制**——这是对「framework > model」（[UniClawBench](../sources/uniclawbench.md) 结论）的直接训练侧回应：不让模型绑死在某个 harness 上。配套 Knowledge-Graph-Guided Task Synthesis（自演化 DAG 知识图谱，agent 持续 web-scale 探索扩展）+ 多类 verifiable 任务环境（kernel optimization / personal assistant / Autonomous Execution Tasks / web development）。

**3. MOPD 9 teacher + Deployment-Aware QAT**。九个 teacher 专家把跨 reasoning effort 的领域能力融合进单一 student（per-token OPD reward with R_max clip）。从 SFT 起即 QAT（MXFP4 权重 + MXFP8 激活），RL 时 rollout 与 training 共用同一量化方案消除 train-inference mismatch。预训练 MTP 层 fine-tune 成 EAGLE-3 draft（融合 1st/4th/final AttnRes block 特征，直接优化 LK loss = acceptance rate 负对数）。

**1M Agentic RL 基础设施**是 K3 后训练的工程重心：partial rollout（λ 比例完成即暂停）+ AgentENV microVM 沙箱（51M+ sandbox，incremental checkpointing 49ms/133ms，pause/resume/fork/snapshot）+ External KV cache pool（write-back，KDA states 与 MLA KV 同步 offload/prefetch）+ rollout auto-throttling。详见 [异步 Agent RL](asynchronous-agent-rl.md)。

## Laguna：Model Factory + 三阶段 + CISPO + 合成代码环境

[Laguna](../models/laguna.md)（Poolside，2026-05）的后训练是三阶段 mid-train→SFT→agentic RL，recipe 在 M.1/XS.2 间一致。三处值得沉淀：

**1. 合成代码环境作统一可验证任务源**。把真实 git commit 转成可验证任务——problem statement + repo checkout + 隐藏 test patch 取自 commit，commit diff 作 gold solution。**双端正确性检查**（gold 过测 + 空解失败测）滤掉 trivial test 与不测变更的 test；再按 repo 热度 + 代码质量百分位滤，从 ~236k commits 留 30–60k 任务。这批任务**同时喂 SFT（teacher 轨迹）和 RL（per-repo test suite 作 binary verifier）**——同一可验证环境贯穿两阶段，与 [KAT-Coder](../models/kat-coder.md) 的 KwaiEnv 模块化沙箱同思路，但 Laguna 强调从真实 commit 挖而非全合成。

**2. CISPO + length-weighted LOO advantage**。RL 用 token-level REINFORCE surrogate + **CISPO** clipping [14]（[14]=[MiniMax-M1](../sources/minimax-m2-series.md)，CISPO 源头）+ length-weighted leave-one-out group-relative advantage。asymmetric clipping $(c_{low},c_{high})=(1,4)$，有效 ratio clip $[0,5]$。报告明说消融 vs [GRPO](agentic-reinforced-policy-optimization.md) / [GSPO](group-sequence-policy-optimization.md) 后选 CISPO 是因「最终评测质量 + 训练稳定性组合最好」——这是 [LLM RL policy optimization 对比](../comparisons/llm-rl-policy-optimization.md) 里 CISPO 首次有公开的 vs GRPO/GSPO 消融选择理由。reward 设计只让 binary task verifier 给正 reward（1.0），其余全是小负惩罚（parsing/min-steps/tool-error per-token），长周期信用分配全靠末尾 verifier 经 advantage 传到每个 token。

**3. IF judge + multi-harness 防过拟合**。agentic SFT 的关键挑战是 instruction following——没有显式 IF 监督，RL 会灾难性遗忘，超半数 agentic 任务因违反 system-message 约束在 coding 被评前就零分。用 EvolInstruct-style generator 给每任务造多个合成 system message + 专用 IF judge 逐项打分。SFT 还混 1.3B tokens 多 harness 轨迹（OpenHands / OpenCode2 / Mini-SWE-Agent），刻意保留各 harness 原生行为（subagent spawning / context compaction / planning scaffold）以利泛化——与 [Kimi K3 的 Unified White-Box RL Env](../sources/kimi-k3.md)（harness-agnostic 训练）同动机但更轻量（数据层而非 RL 配置层）。

**工程**：TITO API for RL actors（保 token ID 跨多轮稳定，与 [GLM-5 异步 Agent RL](asynchronous-agent-rl.md) 同动机）+ trainer↔inference 权重同步（NCCL GPUDirect RDMA，每 2 optimizer step 广播，KV-cache reset 防版本混入，staleness 上限 10 step）+ FP8 KV cache 跑 131072 context rollout（翻倍并发轨迹）。

## ARPO 与 LLM RL policy optimization：采样结构和优化目标

[Agentic Reinforced Policy Optimization](agentic-reinforced-policy-optimization.md) 不是新模型报告，而是把 Qwen2.5 / Llama3.1 / Qwen3 等 backbone 放进多轮工具环境里比较 RL 算法。它的关键观察是：模型收到外部工具反馈后，后续前 10–50 个 token entropy 会显著上升，搜索反馈的不确定性又高于 Python 反馈。因此 ARPO 不再只做完整 trajectory-level sampling，而是在高熵工具调用步从当前节点分叉 partial rollouts，并用 advantage attribution 区分共享前缀与分叉段。

[DAPO](../sources/dapo.md)、[GSPO](../sources/group-sequence-policy-optimization.md)、[SAPO](../sources/soft-adaptive-policy-optimization.md) 则在另一条轴上回答「group-based RL 自己怎么稳定」：DAPO 补 long-CoT GRPO recipe（Clip-Higher / Dynamic Sampling / token-level loss / overlong shaping），GSPO 把 GRPO 的 token-level ratio 改成 sequence-level ratio 以稳定 MoE，SAPO 再用 soft gate 替代 hard clipping，兼顾 sequence coherence 与 token adaptivity。

这补上了现有几条路线之间的空档：GLM-5 / Forge 关心 rollout 系统如何高吞吐、低偏差地跑起来；ARPO 关心 rollout 预算在一条 agent 轨迹内部应该投到哪里；DAPO / GSPO / SAPO 关心采到的数据如何被稳定、高效地转成 policy update。详见 [LLM RL policy optimization 对比](../comparisons/llm-rl-policy-optimization.md)。

## 综合框架

可以把这些报告的后训练看成七种互补能力：

- GLM-5：如何让 agent 在真实环境中高吞吐学习。
- MiMo-V2-Flash：如何把多个专门 teacher 的能力合成到一个 student。
- DeepSeek-V4：如何在超长上下文和多 reasoning mode 下做稳定蒸馏与 RL。
- MiniMax-M2：如何把 agent harness、reward、rollout、training 和 serving 组织成可扩展系统。
- Kimi K2.5：如何让多模态和并行 sub-agent 编排共同提升 agentic 工作流。
- Kimi K3：如何在 3T 规模 + 1M 上下文下做 9-专家 RL + MOPD 融合 + harness-agnostic 训练 + microVM 沙箱支撑 partial rollout。
- DAPO / GSPO / SAPO：如何把 group-based RL 的 policy update 做稳、做可扩展。
- ARPO：如何把探索预算从完整轨迹平均采样，转移到工具反馈后的高熵 step-level 行为。
- Laguna：如何把模型开发本身做成工业流程——合成代码环境贯穿 SFT/RL、CISPO + length-weighted LOO、IF judge + multi-harness 防过拟合，全流程靠 Model Factory 翻配置 flag 复用。
- HunyuanOCR-1.5：如何用 agent 自动化数据构造（Agentic Data Flow）补长尾能力 + 三组件 reward（事实性 / 一致性判官 / 退化抑制）做 OCR 专项 RL。
- KAT-Coder：如何把训练基础设施（可验证环境 + 沙箱可靠性 + harness 泛化）当作 agentic 能力的第一性问题。V2 用 MCLA 降 MoE RL log-prob 方差 + Tree Training 消树状轨迹冗余 + turn-level GSPO 折中；V2.5 发现 ~16% 训练失败源于沙箱而非算法，切换到 asymmetric PPO + hindsight-augmented critic，并用 harness randomization + process-aware 轨迹过滤 + 长上下文 MOPD 稳定化（cold start + drift-aware truncation）系统性重构。
- [daVinci-Agency](../sources/davinci-agency.md)：如何在 SFT 阶段就从数据结构层面注入长周期监督。不靠 RL 或 distillation，而是用 GitHub chain-of-PRs 的跨 stage 依赖把孤立 coding 任务串成项目演化级工作流，239 样本即超过 66k 样本的 SWE-Smith。与上述 RL 路线互补--它是 RL 之前的数据层问题：训练数据本身该长什么样才能让 agent 内化 task decomposition / long-term consistency / refinement。
- [VibeThinker-3B](../sources/vibethinker-3b.md)：如何在严格小模型（3B dense）上把 verifiable reasoning 推到旗舰级。后训练范式是 Spectrum-to-Signal Principle--SFT 构造多样化解空间（Diversity-Exploring Distillation 按 Pass@K 选域 specialist 再参数级 merge），RL 用 MGPO（最大熵 prompt 权重降权全对/全错 prompt）放大推理信号。独有设计：(1) Long2Short RL 用零和 centered length-aware reward shift 在不改变 group baseline 的前提下偏好更短正确轨迹；(2) CLR test-time scaling 聚焦 claim 级自验证而非整条 trace 聚合；(3) 渐进式 context window 扩展在 3B 上失效（与 1.5B 相反），高 SFT 质量下 warm-up 截断破坏已有长推理行为。提出 Parametric Compression-Coverage Hypothesis 解释为何 3B 能追平旗舰数学/编程但 GPQA-D 仍差 14 点。
- [Ling-2.6 / Ring-2.6](../sources/ling-2.6.md)：万亿参数 agentic 双线模型。Ling-2.6（instant）用 specialization-then-distillation 把 reasoning specialist（Evo-CoT + LPO + redundancy penalty）和 agentic specialist（[GSPO](../sources/group-sequence-policy-optimization.md) + zlib 压缩比惩罚）并行训练后蒸馏，核心目标是 **capability per output token**（~4× token efficiency）。Ring-2.6（thinking）在后训练基础上增强 long-horizon agentic，核心创新是 **KPop** RL 算法：用 binary KL divergence 替代前代 IcePop 的 uniform fixed-ratio constraint，更好捕获不同概率 token 的异质训练-推理 mismatch，由单一超参 $\phi$ 控制。配合异步 RL（partial-rollout pipeline + ARouter + staleness manager）使长尾环境交互轨迹在万亿规模可训练。与 [GLM-5 异步 Agent RL](asynchronous-agent-rl.md) 的区别：Ling-2.6 用 token budget $\Phi$ 约束每步而非轨迹数量阈值，尾请求 offload 到专用推理组而非简单丢弃。
- [Keye-VL-2.0](../sources/keye-vl-2.md)：如何把 DSA 长上下文效率与多模态 agentic 能力统一在同一个 MoE 模型里。后训练用五阶段 RL（General -> Specialized -> Video -> Agentic -> Cross-Modal MOPD）逐步注入异构能力，最后用 13-teacher Cross-Modal MOPD 融合。独有设计：(1) 首个把 DSA 从 MLA 适配到 GQA backbone（indexer MQA + aggregation GQA），证明 DSA 不依赖 MLA；(2) top-k overlap estimator 只在 teacher/student 双方高概率 token 上算 advantage，解决多模态多 teacher 场景下 token 级 KL 估计的不稳定性；(3) shared colocated partial-rollout 让未完成 agent 轨迹跨 rollout step 续跑，提高长 horizon rollout 利用率。
- [JoyAI-VL-Interaction](../sources/joyai-vl-interaction.md)：如何把后训练从"回答问题"转向"每秒决定是否行动"。交互行为通过两阶段获得：SFT 用角色加权 cross-entropy 对抗 silence/response 不平衡（$w^\text{repeated}_\text{silence}=0.4$，$w^\text{response}=1.5$），RL 用 GRPO + answer-centered window sampling 直接优化每秒策略。answer-centered window sampling 是流式交互场景对 ARPO step-level rollout 思想的平行演进：ARPO 在工具调用后的高熵步分叉 partial rollout，JoyAI 在 gold response 周围保留流式因果性但压缩 horizon。
- [Xiaomi-GUI-0](../sources/xiaomi-gui-0.md)：如何把后训练从"离线成功轨迹"转向"真机错误闭环"。三阶段 SFT -> Step RL -> Agentic RL 按 dense->sparse feedback 课程排列，每阶段初始化自前一阶段 checkpoint。三处独有设计：(1) **Cascade reward**--不独立打分各维度再加权，而是 L1-A/L1-B/L2（rule-based）-> L3/L4（LLM-as-judge）top-down early-exit，三值 reward（−1.0 / −0.5 / 1.0），把评估成本约束在通过低成本的响应上才调用 LLM judge；(2) **Turn-level batching**--Agentic RL 把长轨迹切成 turn（每 turn = 一个 `(x, h_t, o_t, y_t)`），advantage 仍从完整轨迹推导但训练序列短，是 GSPO sequence-level ratio 在多步交互场景的适配（与 KAT-Coder V2 的 turn-level GSPO 折中同构）；(3) **Error-driven data flywheel**--teacher 打分与接管刻意产生 deviation–diagnosis–recovery 段，传统 flywheel 丢弃失败轨迹，Xiaomi-GUI-0 把模型自身错误分布转为反思/纠正/恢复监督。复用 [GSPO](../sources/group-sequence-policy-optimization.md) + [DAPO](../sources/dapo.md) dynamic sampling，是 LLM RL policy optimization 方法在 GUI agent 领域的直接落地。
- [Mach-Mind-4-Flash](../sources/mach-mind-4-flash.md)：如何把"后训练 scaling"作为紧凑模型追赶前沿的主路径（不扩预训练算力）。基于 [Qwen3.5-35B-A3B](../models/qwen3.5.md)，specialization-then-integration 流水线：SFT → 三轨并行 RL（Reasoning / General / Agent 共 10+ 专家）→ [MOPD](../concepts/multi-teacher-on-policy-distillation.md) 融合 → HMPO token 效率。独有设计：(1) **统一 RL/OPD loss**（`L = α·L_OPD + β·L_RL`）--已收录报告中唯一把 RL 和 OPD 混进单一加权 loss 的实现，支持纯 RL / 纯 OPD / 联合三模式切换，OPD 直接复用 RL 框架的分布式调度和异步 reward routing；(2) **HMPO**（Hybrid Median-length Policy Optimization）--单阶段 token 效率 RL，从正确 rollout 中位长度推导 group-adaptive budget `b`，乘法 reward `R_final = R_acc · R_token` 保证 correctness-first 防止 reward hacking，仅数学训练压缩 19–46% 且跨域泛化；(3) **EnvScaling**--Tool-Use RL 不扩展静态 tool-call trace 而是扩展环境（190+ stateful domain / 3.5K tool interface），三类环境（file-system execution / programmatically verifiable / model-simulated）统一为 agent-environment 接口；(4) MOPD Appendix C 消融发现 code teacher 的跨域迁移效应（加 code teacher 后 AIME 涨分，未加数学 teacher），是已收录报告中唯一明确报告 MOPD teacher 间跨域迁移量化数据的。
- [Agent-World](../sources/agent-world.md)：如何把可扩展环境同时用作训练源和诊断 arena，形成 **agent-environment co-evolution**。后训练用 Qwen3-8B/14B backbone + 冷启动 SFT（40K 轨迹）+ GRPO RLVR（clip `ε_low=0.2`/`ε_high=0.28`，即 [DAPO](../sources/dapo.md) Clip-Higher），创新点不在 policy optimization 算法而在数据组织与课程。独有设计：(1) **多环境 rollout**——每个 global batch 内任务配独立动态环境（agent–tool–database 三元闭环），reward 双路（graph-based 用 rubric LLM judge 取均值、programmatic 用可执行 `V_code` 验证答案+库状态），可执行 reward 提供 state-aware 监督；(2) **自演化 arena**——分层采样 K=5 环境/一级类别构造评测集，每轮动态合成新任务防过拟合，agentic diagnosis agent 从失败 trace + 错误分布 + 环境元数据定位弱环境 `W^(r)` 与任务生成指南 `G_guide^(r)`，定向任务扩展（必要时数据库复杂化）后 continue RL，形成 `evaluate → diagnose+target → continue RL` 循环；(3) 两轮自演化使 MCP-Mark +8.6（环境特定弱点的最大增益出现在需强状态追踪的真实 MCP server 交互）。与 [Forge](forge-agent-native-rl.md) self-evolution 互补：Forge 把 harness/context/memory 当环境解耦训练系统吞吐，Agent-World 把"环境本身的可扩展合成 + 诊断驱动课程"当第一性问题；与 [daVinci-Agency](../sources/davinci-agency.md) 互补：daVinci 在 SFT 数据结构注入长周期监督，Agent-World 在 RL 阶段用诊断持续补能力缺口。
- [Qwen-AgentWorld](../sources/qwen-agent-world.md)：如何把 **world modeling 作为 agent 基座的训练目标**，而非只是 policy optimization。后训练范式是 native language world model——从 CPT 阶段就以环境建模为训练目标（"CPT injects, SFT activates, RL sharpens" 三阶段），产出预测环境观测 `ô_{t+1} = f_θ(c, o_≤t, a_≤t)` 的 world model。独有设计：(1) **Turn-level 信息论 loss masking**——按 Overlap/Novelty/Jaccard/length ratio 四统计量把 turn 分 7 类（retrieval/expansion/action 100% keep，boilerplate 10%、echo 5%），被 mask 的 turn 保留作上下文但排除出 loss，把"学下一状态"与"学下一 token"解耦；(2) **hybrid rubric-and-rule reward（9:1）**——五维 rubric（Format/Factuality/Consistency/Realism/Quality，LLM judge）配 rule-based verifier（binary 锚点），三种稳定性解法（每轨迹单 turn 展开防 Echo Trap、rubric+rule 而非 reference/turing-test reward、tag extraction 防 self-praise）；(3) **LWM warm-up 跨任务迁移**——单轮无工具调用的 LWM RL warm-up 在 7 个多轮工具调用 agentic benchmark（含 3 个完全 OOD 域）一致提升，机制是 next-state prediction 内化为 meta-reasoning 模式（执行前心智模拟环境响应）。用 [GSPO](../sources/group-sequence-policy-optimization.md) 作 RL 算法，面临独特的 **prompt–output 极端不对称**（prompt 是完整轨迹历史常达数万 token，output 是单预测观测几百到几千 token，单样本计算被 prompt 处理主导）。与 Agent-World 互补：Agent-World 把环境当可扩展合成 + 诊断对象，Qwen-AgentWorld 把环境当可学习的 world model——前者是 code-driven 确定性路线，后者是 learned neural simulator 通用性路线（论文自述 "trades determinism for generality"）。

## 待追问

- 异步 RL 的 off-policy 偏差和 MOPD 的 teacher-student gap 是否能统一建模？
- ARPO 的 entropy-based branching 能否嵌进 Forge / GLM-5 这类大规模 agent RL 系统，还是只适合较小规模 search/Python 工具实验？
- DAPO 的 recipe、GSPO 的 sequence-level ratio、SAPO 的 soft gate 能否组合成同一个训练栈，还是彼此在 loss reduction / ratio 单元 / clipping 形状上有冲突？
- Agentic benchmark 的 reward 是否足够可靠，还是会过拟合 harness？
- “保留 thinking”提升长周期任务的同时，会不会带来隐私、延迟或上下文污染问题？
- Agent Swarm 这类运行时并行策略，应该训练进模型，还是保留在外部 agent framework 中？
