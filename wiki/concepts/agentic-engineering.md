---
type: Concept
title: "Agentic Engineering"
description: "这些报告如何定义长周期软件工程和工具使用任务。"
tags: ["concept", "agentic-engineering"]
timestamp: 2026-06-06
---

# Agentic Engineering

## 定义

Agentic engineering 是这些报告中的共同趋势：软件工作正在从一次性代码生成，转向长周期工程任务。模型需要阅读仓库、规划修改、调用工具、运行测试、修复错误，并在多轮操作中维护状态。

这个词在 [GLM-5](../models/glm-5.md) 中最核心，但 [MiMo-V2-Flash](../models/mimo-v2-flash.md)、[DeepSeek-V4](../models/deepseek-v4.md)、[MiniMax-M2 Series](../models/minimax-m2-series.md)、[Kimi K2.5](../models/kimi-k2.5.md) 和 [ARPO](agentic-reinforced-policy-optimization.md) 都指向类似方向。

## 跨报告信号

- GLM-5 强调 ARC 能力：agentic、reasoning、coding，并加入异步 agent RL、preserved thinking 和 CC-Bench-V2 真实工程评测。
- GLM-5.3 官方发布博客把同一路线的瓶颈进一步落在**环境生产的审计闭环**：research agent 从真实工作抽取模式并合成可执行环境，judge agent 检查可解性，verifier 不看参考解生成，再由 solver trajectory 发现 reward shortcut；只有通过 oracle / no-op / unsolved-state 的 verifier 才输出训练用二元 reward。它把“足够多的真实任务”细化成“可生成、可解、且难被 reward hacking 的任务环境”，但未公开人工参与比例和误判率。详见 [GLM-5.3 官方发布博客](../sources/glm-5-3-blog.md)。
- MiMo-V2-Flash 强调软件工程 benchmark、search agent、tau2-Bench 工具使用，以及通过 MOPD 融合专门 teacher。
- DeepSeek-V4 强调长周期工作流和百万 token 上下文，把它视为 test-time scaling 与未来 online learning 的基础设施。
- MiniMax-M2 把 agentic engineering 进一步系统化：任务要有可执行环境、verifiable reward 或可信 judge evidence；训练端用 Forge 解耦 agent、rollout、training 和 reward。
- Kimi K2.5 把 agentic engineering 扩展到视觉和并行 agent：模型不仅要调用工具，还要创建 sub-agent、拆解任务、分片上下文，并聚合多源结果。
- ARPO 从算法采样结构切入：工具返回后模型前 10–50 个 token entropy 升高，因此 RL 不应只比较完整轨迹，而应在高熵工具调用步分叉 partial rollouts 来学习 step-level tool-use 行为。
- HunyuanOCR-1.5 的 Agentic Data Flow 把 agent 自动化用到数据构造而非任务执行：算法工程师用自然语言描述能力需求（如「为低资源语言造合成 OCR 数据」），agent 自主分解任务、搜索物料、开发渲染/QA 生成 pipeline、跑 hard-case 挖掘，并与人类多轮迭代。这与 AgentInstruct / TaskCraft / MetaSynth 等 agentic 合成数据系统方向一致，但落地在 OCR 领域的长尾能力扩展。
- KAT-Coder（V2 / V2.5）把 agentic engineering 的瓶颈定位在**训练基础设施**而非模型规模。V2 用 Specialize-then-Unify 范式分治五域专家 + KwaiEnv 模块化沙箱（万级并发、网络层代理零代码集成 scaffold）+ Tree Training 消除树状轨迹冗余（6.2× 加速）。V2.5 系统性重构：发现 ~16% 轨迹失败源于沙箱而非模型（降到 <2%）、harness randomization 治 scaffold overfitting（format/context-structure/control-flow 三种）、asymmetric PPO with hindsight critic 做 turn-level credit assignment。两代都用 AutoBuilder 从真实仓库自动造可验证 SWE 任务（F2P+P2P）。
- daVinci-Agency 把长周期 agent 数据合成的瓶颈定位在**监督信号的结构**而非数据量。核心洞察是真实 GitHub PR 链天然编码了 task decomposition / long-term consistency / verifiable refinement 三种长周期技能的监督--chain-of-PRs 把孤立 coding 任务变成有跨 stage 依赖的多步工作流。仅 239 样本 SFT GLM-4.6 即在 Toolathlon +47%、AVG 超过 66k 样本的 SWE-Smith。这与 KAT-Coder 走的 RL + 基础设施路线互补：KAT-Coder 解决「怎么稳定训练」，daVinci-Agency 解决「训练数据本身该长什么样」。
- Seed2.0 Model Card 从**生产部署视角**补充 agentic engineering：MaaS 使用数据显示真实 agentic coding 查询中前端开发占 >50%、bug fixing 占任务主导，模型优化应优先前端生成质量和调试能力。四维评测框架把"real-world complexity"操作化为 Science Discovery / Vibe Coding / Context Learning / Real-World Tasks，并诚实标注 repository-level coding（NL2Repo-Bench 27.9、SWE-Evo 8.5）为短板。Case studies（FreeCAD 96 步建模 / CapCut 视频编辑 / Qiskit 量子计算 bug 修复）展示 GUI 操作和跨学科科研编程的推理深度，但缺乏对照实验。
- GLM-5V-Turbo 把 agentic engineering 扩展到**多模态**：模型不只调用工具，还在 GUI 中截图、交互、导航，再结合 native UI-to-code 能力复现网站。工具链覆盖多模态搜索、图像处理（裁剪/标注/3D 框/视频跟踪）和创作（网页/PPT）。三个 design lens 影响对 agentic engineering 的理解：（1）感知是 agent 能力天花板——许多看似高层失败始于模型看不准 GUI 细节；（2）分层优化——低层任务（元素感知/grounding/单步动作）更易构造和验证，高层长周期任务在低层不牢时直接推会失稳；（3）端到端任务的关键是清晰规格 + 可靠验证 + 受控评测（Vision2Web benchmark 把任务规格从文本指令扩展到 PRD/mockup/reference page/resource asset，用 workflow-based verification 替代单一终态判断）。详见 [GLM-5V-Turbo 来源页](../sources/glm-5v-turbo.md)。
- Agent-World 把 agentic engineering 的瓶颈定位在**可扩展真实环境合成 + 连续自演化训练机制**，而非模型参数。核心论点是通用 agent 需要多样、真实、有状态的环境来训练 long-horizon 工具编排与状态追踪，但人工造环境昂贵且难扩展，而既有合成环境要么纯 LLM 模拟（易幻觉、偏离真实动态）要么来自有限开源工具链（复杂度不足）。Agent-World 从真实 MCP servers（~2.8K）/ 工具文档（~0.5K）/ 工业 PRD（~0.2K）挖主题，用 deep-research agent 从 web 自动建主题对齐数据库（含数据库复杂化迭代）+ coding agent 生成并交叉验证可执行工具（编译通过 + 测试准确率 >0.5），合成 **1978 环境 / 19822 工具**的生态；任务合成走 graph-based（DAG 工具图 + 随机游走，反向工程合法调用序列再生成任务）与 programmatic（解代码 + 可执行 verifier 脚本）两条路，均靠 sandbox 执行推导 ground truth 并保留可验证性。与 KAT-Coder（训练基础设施稳定性）/ daVinci-Agency（SFT 数据结构）/ Xiaomi-GUI-0（训练-评测分布对齐）互补：Agent-World 解决"训练环境本身如何可扩展地合成并持续演化"这一更底层的数据基础设施问题。详见 [Agent-World 来源页](../sources/agent-world.md)。
- Xiaomi-GUI-0 把 agentic engineering 的瓶颈定位在**训练与评测的分布对齐**而非模型规模。核心论点是 benchmark 高分不等于真实可用性——真实设备上的账号状态 / 权限弹窗 / 支付验证 / 风控拦截持续改变执行状态分布，而离线成功轨迹 / 模拟环境 / 标准化 benchmark 无法覆盖。三处设计回应这一论断：（1）真机为主的混合基础设施 + Device-Pull 调度，让数据采集 / 训练 rollout / 评测共享接近部署的执行分布；（2）error-driven data flywheel 围绕模型自身错误分布做定向修复（首个关键错误标注 + teacher 打分接管产生 deviation–diagnosis–recovery 段），而非简单扩数据量——传统 flywheel 丢弃失败轨迹，Xiaomi-GUI-0 把失败转为反思/纠正/恢复监督；（3）RealMobile 真机评测用细粒度 sub-goal + veto + 双验证，按 4 能力域分解揭示 Safety & Reflection 是所有模型共同瓶颈。与 KAT-Coder / daVinci-Agency 互补：KAT-Coder 解决训练基础设施稳定性，daVinci-Agency 解决 SFT 数据结构，Xiaomi-GUI-0 解决训练/评测与真实部署分布的对齐。详见 [Xiaomi-GUI-0 来源页](../sources/xiaomi-gui-0.md)。
- Qwen-AgentWorld 把 agentic engineering 的瓶颈定位在 **world modeling 这一缺失的拼图**——当前 LLM agent 研究几乎只关注 policy 侧（states → actions），忽略了 world model（(states, actions) → subsequent states）。论文论证 world modeling 从两个互补轴增强 agent：(1) **Decouple**——LWM 作独立环境模拟器做 Sim RL，靠可控模拟（注入定向扰动 / 构造完全虚构但自洽的世界）暴露真实环境罕见的弱点，甚至超过真实环境训练（WideSearch Sim RL 50.3% vs Real RL 45.6%；MCPMark controlled +12.3 vs 标准 Sim RL 反降）；(2) **Unify**——LWM RL warm-up（单轮、无工具调用）把 next-state prediction 内化为 meta-reasoning 模式，跨 7 个 agentic benchmark（含 3 个完全 OOD 域）一致提升（Claw-Eval +11.3、BFCL v4 +9.0）。机制证据是 prediction-driven action refinement：RL 后模型在执行前系统性心智模拟环境响应（mailman 任务正确预测 Postfix 处理流程），预测准确率 69.9%→78.3%。与 [Agent-World](../sources/agent-world.md)（code-driven 环境合成）互补：Agent-World 用程序化合成保证确定性执行与可验证 reward，Qwen-AgentWorld 用 learned neural simulator 覆盖 code 难以指定的域（搜索引擎、真实 MCP servers）——论文明确自定位为 "trades determinism for generality"。详见 [Qwen-AgentWorld 来源页](../sources/qwen-agent-world.md)。
- [Laguna](../sources/laguna-m1-xs2.md)（Poolside，2026-05）把 agentic engineering 的瓶颈定位在**模型开发流程本身**，而非模型规模或单点算法。核心论点是「把模型开发当工业流程而非手工艺」是 frontier 模型开发最 consequential 的杠杆——M.1 预训练结束后**五周**从零交付 XS.2 是这套 Model Factory 的实证。三条原则（experiments as code / composable decoupled components / reserve human attention for novel decisions）落到 agentic coding 上具体体现为：(1) 合成代码环境把真实 git commit 转成可验证任务（双端正确性检查 + repo 热度过滤），**同一可验证环境贯穿 SFT（teacher 轨迹）与 RL（per-repo test suite 作 binary verifier）**，复用率高于分别造数据；(2) RL 用 CISPO（[MiniMax-M1](../sources/minimax-m2-series.md) 源头）+ length-weighted LOO advantage，asymmetric clip (1,4)，明说消融 vs GRPO/GSPO 后选它；(3) IF judge + multi-harness 训练（OpenHands/OpenCode2/Mini-SWE-Agent）防 scaffold 过拟合。与 [KAT-Coder](../models/kat-coder.md)（训练基础设施稳定性）/ [daVinci-Agency](../sources/davinci-agency.md)（SFT 数据结构）/ [Kimi K3 Unified White-Box RL Env](../sources/kimi-k3.md)（harness-agnostic RL 配置）互补：Laguna 解决「整个开发流程如何可复用、可由非核心团队 self-service 跑起来」这一更外层的工程问题——XS.2 的初始 imitation learning 阶段由核心后训练团队之外的人独立跑完。

- Macaron-V1 把瓶颈进一步落在**model–harness 的共同版本化**：MoL 让 Chat / Agent / Coding / GenUI specialist 在 frozen base 上按 turn 路由，而 HCP 将 router、工具、skills、prompts、memory 与 workspace 变成可审计 runtime contract。其 122/122 TerminalBench base-failure task 覆盖实验只改变 HCP/skill/hook、base 全程冻结；它支持「很多失败是未被正确 elicitation 的能力」这一工程判断，却不等价于 adapter 训练、跨代持续学习或 collective intelligence 已被实证。详见 [Macaron-V1 技术报告](../sources/macaron-v1.md)。

## 为什么重要

Agentic engineering 改变了瓶颈。模型不只是生成正确片段，还要管理上下文、使用工具、吸收环境反馈、记住先前动作，并在长历史中保持稳定推理。

新增两篇报告后，一个更清晰的趋势是：agentic 能力越来越依赖运行系统。MiniMax 的 Forge 说明训练系统要能承受长尾 rollout、black-box scaffold 和大规模可验证反馈；Kimi 的 Agent Swarm 说明推理时也可以通过并行 sub-agent 改变任务复杂度和上下文形态。ARPO 进一步说明，哪怕不改 harness，**rollout 预算在轨迹内部的分配方式**也会影响 tool-use 行为学习：工具反馈后的高熵节点比完整轨迹开头更值得追加探索。

## 待追问

- 哪些 benchmark 最能预测真实 coding agent 生产力？
- Thinking preservation 在部署中应该暴露到什么程度？
- 更长上下文会减少 context management 的必要性，还是让 context management 更重要？

## 相关页面

- [异步 Agent RL](asynchronous-agent-rl.md)
- [Forge Agent-Native RL](forge-agent-native-rl.md)
- [Agent Swarm](agent-swarm.md)
- [Agentic 评测体系](agentic-evaluation-benchmarks.md)
- [Agentic Reinforced Policy Optimization](agentic-reinforced-policy-optimization.md)
- [百万 token 上下文服务](million-token-context-serving.md)
- [HunyuanOCR-1.5](../models/hunyuan-ocr-1.5.md) - Agentic Data Flow
- [KAT-Coder](../models/kat-coder.md) - Specialize-then-Unify + KwaiEnv + Tree Training
- [daVinci-Agency](../sources/davinci-agency.md) - chain-of-PRs 数据合成范式
- [Seed2.0](../models/seed2.md) - 四维评测框架 + 部署洞察
- [Xiaomi-GUI-0](../models/xiaomi-gui-0.md) - 真机闭环 + error-driven flywheel
- [Agent-World](../models/agent-world.md) - 可扩展真实环境合成 + 自演化训练场
- [Qwen-AgentWorld](../models/qwen-agent-world.md) - language world model（Sim RL 模拟器 + agent 基座 warm-up）
- [Laguna](../models/laguna.md) - Model Factory 工业化流程 + 合成代码环境贯穿 SFT/RL + CISPO
- [Macaron-V1](../models/macaron-v1.md) - MoL specialist composition + HCP 版本化 harness + 未闭合的 RSI 证据边界
- [GLM-5.3](../models/glm-5-3.md) - 可验证环境合成与 verifier 审计闭环
