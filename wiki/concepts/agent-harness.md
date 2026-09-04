---
type: Concept
title: "Agent harness"
description: "模型与世界之间的执行膜：标准化执行、恢复、验证和资源记账，把策略构造留给模型；跨报告里它已经是与权重同量级的性能变量。"
tags: ["concept", "agent-harness", "rlm", "continual-harness"]
timestamp: 2026-09-05
---

# Agent harness

## 定义

**Agent harness** 是模型观察和动作世界时必须穿过的那层 runtime：工具与沙箱、上下文如何进入下一次调用、失败如何恢复、何时停止、以及花费如何记账。它不是模型权重，也不是某一个 planner prompt。

[Prime Agent](../sources/prime-agent.md) 把这层说成膜（membrane）：内侧标准化执行、恢复、验证和资源会计，外侧把策略构造留给模型，避免「harness 失败被读成模型失败」（原文确证，Prime Agent §1）。它同时把 **expressivity** 定为关键性质：好的 harness 暴露 primitive（持久 REPL、递归 `rlm()`、typed 可修订状态、agent-to-agent 队列），而不是编码一条固定工作流。

这和「把 agent 做成产品」不是同一句话。产品 harness 常常为了一种 UX 收紧动作空间；评测 / 训练 harness 的目标是让测量靠近模型真实可达能力，或让同一套 runtime 同时服务 rollout 与 serving。

## 跨报告信号

### Prime Agent：表达性膜 + 权重冻结的 self-improvement

[Prime Agent](../sources/prime-agent.md) 把状态分成 L0 权重 / L1 可见上下文 / L2 REPL 与 subagent / L3 磁盘上的 history、memories、skills、prompts 和 subagent specs。L1–L2 边界是 model–context boundary：L2 的 Python 值必须被序列化才会进入生成。Continual Harness 在轨迹时间内把执行证据写成版本化 L3，不改 L0。ARC-AGI-3 上 Prime Agent + Opus 5 报 95.5% RHAE，对照官方 ARC harness 的 30.2%；作者明确这些外部点只是 situating，因为他们自己的 native 复跑低于官方分。

### Macaron-V1：把 harness 做成可版本化、可搜索的契约

[Macaron-V1](../sources/macaron-v1.md) 的 HCP 把 router、工具 allowlist、MCP、hooks、prompt/skills、session 与 workspace 写成 TOML 契约。最醒目的实验故意不更新权重：122 个 TerminalBench 2.1 基线必失败任务，69 个 adaptive HCP/skill/hook job 后覆盖 122/122。它支持「失败常常是未被正确 elicit 的能力」，但不能当成 adapter 学习曲线或跨代持续学习。与 Prime Agent 的差别：Macaron 搜索的是可评测 configuration portfolio；Prime Agent 在单条长轨迹里持续 refine 运行时状态。

### UniClawBench：读分时 framework 与 model 同量级

[UniClawBench](../sources/uniclawbench.md) 在同一模型换框架：GPT-5.4 在 OpenClaw / EDICT / Nanobot 下 Overall PR 为 0.407 / 0.338 / 0.290。这是「harness 是一等变量」的评测侧证据，不是 expressivity 设计。OpenClaw 的优势被解释为集中式单 agent 信息损失最小；EDICT 付出多 agent token 却常卡在 coordination friction。

### Qwen-UI-Agent：通知 affair 与跨设备 planner

[Qwen-UI-Agent](../sources/qwen-ui-agent.md) 的 harness 不搜索 HCP，也不提供 REPL。它回答两件模型权重单独做不了的事：何时从手机通知主动开工，以及工作流如何在手机和电脑之间不断上下文。核心抽象把 event / affair / task 分开；低风险预备可以先做，改票和支付仍要用户确认。跨平台侧是 OpenClaw-like planner，把 Qwen-UI-Agent 当 GUI subagent 按设备调用，虚拟屏并行且不挡用户。这与 UI-Mate DemoCUA 的差别：后者改的是「同一任务可见哪段程序」；前者改的是任务何时开始、在哪台设备继续。当前证据是定性轨迹（Figure 11–12 / 28–29），没有开关膜的成对数字。

### UI-Mate DemoCUA：示范工作流是一层会改策略的膜

[UI-Mate](../sources/ui-mate.md) 的 Demo Workflow Harness 不搜索 HCP，也不提供 REPL。它把一条录屏变成指针驱动的 subtask checklist：当前目标、完成判据、不含坐标的里程碑，外加 `subtask_complete`。live 截图对示范有否决权，所以这层膜改变的是条件化，不是动作空间本身。33-task self-demo 上严格成功 17.2%→35.4%，是「同一模型换可见程序」的成对证据；作者同时承认 workflow 钉在上下文开头会破坏 KV 前缀，属于膜的实现税。它与 Prime Agent / Macaron 的差别：后两者在冻结 L0 时改 runtime 原语或配置；UI-Mate 还专门为这层膜做了 SFT（对齐 / 错位 / 无关示范），权重和膜是一起训的。

### 训练侧：不要让模型过拟合某一个 scaffold

多份后训练报告把 harness 当训练分布的一部分，而不是评测后才换上的皮肤：

- [KAT-Coder-V2.5](../sources/kat-coder-v2.5.md) 用 harness randomization（format / context-structure / control-flow）和 harness rewriting 对抗 scaffold overfitting。
- [Laguna](../sources/laguna-m1-xs2.md) 用 OpenHands / OpenCode2 / Mini-SWE-Agent 多 harness 训练，并加 IF judge。
- [Kimi K3](../sources/kimi-k3.md) 的 Unified White-Box RL Env 按配置实例化 Kimi Code / Claude Code / Codex / OpenClaw / Hermes，目标是 harness-agnostic RL。

这些和 Prime Agent 的「先做一张表达性膜再 co-train」方向相反但互补：前者防止模型绑死在一种 CLI，后者假设当前模型还不会用满一张更强的膜。

### ASPIRE：具身侧的 coding agent 膜

[ASPIRE](../sources/aspire.md) 把 coding agent、机器人执行引擎和 skill library 当成具身侧的膜。内侧标准化的是 per-primitive 多模态 traces、允许的感知/规划/控制 API，以及 coordinator 对可复用修复的入库审计；外侧仍把「写哪份程序、补哪处失败」留给冻结的 coder（仿真 Claude Opus 4.6，真机 GPT-5.5）。底层策略网络若存在也不被更新——对照表里的 OpenVLA / π0 / π0.5 只是评测基线。

与 Prime Agent / Macaron 的同构：L0 不动，改的是可复用程序（这里是 skill library 里的 code-as-policy 修复，不是 HCP TOML 或 REPL skill）。差异：痕迹是机器人感知–运动与接触动力学，不是终端日志；sim-to-real 运的是 know-how（in-context 指导），不是像素或权重。详见 [具身 skill 自进化](embodied-skill-self-evolution.md)。

### Agent Swarm：编排策略 ≠ 执行膜

[Agent Swarm](agent-swarm.md) 是 Kimi K2.5 里被 PARL 训出来的 orchestrator 策略：动态创建 frozen subagent、按 critical path 并行。Prime Agent 的递归 subagent 是同一 daemon 上的持久 session 加消息队列，**没有**报告 RL 训练编排器。Kimi 改的是「谁被梯度更新」；Prime Agent 改的是「子 agent 是否作为可恢复计算节点存在」。两者都做 context sharding，但一层是学到的拆任务，一层是 runtime 语义。

## 为什么重要

1. **Agent 分数默认是 (model, harness, budget, context policy) 的联合。** UniClawBench 和 Prime Agent 从两个方向重复这件事：换框架可以超过换模型；换一张更表达的膜可以大幅改变 ARC-AGI-3 曲线形状。[UI-Mate](../sources/ui-mate.md) 再加一维：同一 verifier 下开关一条示范，严格成功可以从 17.2% 到 35.4%。[Qwen-UI-Agent](../sources/qwen-ui-agent.md) 则把「何时开始、在哪台设备继续」也放进膜，但还没有成对消融。读 [Agentic 评测体系](agentic-evaluation-benchmarks.md) 时，harness 和示范都不能再当脚注。
2. **冻结权重仍能改可达策略集。** Macaron 的 HCP 搜索和 Prime Agent 的 Continual Harness 都在 L0 不动时扩展策略。[ASPIRE](../sources/aspire.md) 把同一判断搬到机器人：改 skill library 而不是 VLA 权重。这不是持续学习已经发生，而是「elicit vs 学会」必须分开记账。
3. **持久化会保存作弊。** Prime Agent 的 Factorio RCON skill 说明：refinement / memory 若没有独立校验和 rollback，self-improvement 会把 specification exploit 写成可复用程序。这是 [Agent 记忆生命周期](agent-memory-lifecycle.md) 的 gate/rollback 在 agent runtime 上的对应物。
4. **训练与评测正在抢同一层。** 一边随机化、多 harness、harness-agnostic RL，一边把评测膜做得更表达、更可记账。后续 model–harness co-learning 若真发生，这两条线会撞到同一组 primitive（工具 schema、上下文管理、subagent API）。

## 待追问

- 表达性膜的收益有多少来自「模型本来就会用代码」，有多少必须靠围着 `rlm` / typed state 做后训练才会出现？Prime Agent §5 把后者标为预期，没有实验。
- 多 harness 训练会不会训出「在每张膜上都平庸」的策略，从而抹掉 Prime Agent 这类高表达接口的优势？
- 如何把 harness 失败（丢状态、记错成本、过早停）从 leaderboard 里单独审计出来，而不是混进模型能力？
- Continual Harness 与 HCP 能否共用一套可审计契约（typed state + provenance + rollback），还是评测膜和产品契约会继续分叉？
- DemoCUA 的 variant-demo 迁移是否必须围着「部分匹配的示范」做后训练，还是 self-demo SFT 已经够用？UI-Mate §10 只有 10 题试点。
- Qwen-UI-Agent 的 proactive harness 有多少增益来自 affair 记忆和跨设备 planner，有多少其实是 27B 策略自己已经会的 GUI+CLI？报告只有定性轨迹。

## 相关页面

- 来源：[Prime Agent 技术报告](../sources/prime-agent.md)、[Macaron-V1 技术报告](../sources/macaron-v1.md)、[UniClawBench](../sources/uniclawbench.md)、[KAT-Coder-V2.5 技术报告](../sources/kat-coder-v2.5.md)、[Laguna M.1/XS.2 技术报告](../sources/laguna-m1-xs2.md)、[Kimi K3 技术报告](../sources/kimi-k3.md)、[UI-Mate 技术报告](../sources/ui-mate.md)、[Qwen-UI-Agent 技术报告](../sources/qwen-ui-agent.md)、[ASPIRE](../sources/aspire.md)
- 相邻概念：[Agentic engineering](agentic-engineering.md)、[Agent Swarm](agent-swarm.md)、[Agent 记忆生命周期](agent-memory-lifecycle.md)、[Agentic 评测体系](agentic-evaluation-benchmarks.md)、[Forge Agent-Native RL](forge-agent-native-rl.md)、[具身 skill 自进化](embodied-skill-self-evolution.md)
- 比较：[2026 前沿模型技术报告对比](../comparisons/2026-open-model-technical-reports.md)
