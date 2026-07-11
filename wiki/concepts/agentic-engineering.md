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
