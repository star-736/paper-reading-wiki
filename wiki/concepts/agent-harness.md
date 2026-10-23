---
type: Concept
title: "Agent harness"
description: "模型与世界之间的执行膜：标准化执行、恢复、验证和资源记账，把策略构造留给模型；跨报告里它已经是与权重同量级的性能变量。Pi 是极小核心产品膜，Prime Agent 是表达性评测膜，Harness² 在无标签任务流上同时改执行 harness 与改进库。"
tags: ["concept", "agent-harness", "rlm", "continual-harness", "pi-coding-agent", "harness2"]
timestamp: 2026-10-24
---

# Agent harness

## 定义

**Agent harness** 是模型观察和动作世界时必须穿过的那层 runtime：工具与沙箱、上下文如何进入下一次调用、失败如何恢复、何时停止、以及花费如何记账。它不是模型权重，也不是某一个 planner prompt。

[Prime Agent](../sources/prime-agent.md) 把这层说成膜（membrane）：内侧标准化执行、恢复、验证和资源会计，外侧把策略构造留给模型，避免「harness 失败被读成模型失败」（原文确证，Prime Agent §1）。它同时把 **expressivity** 定为关键性质：好的 harness 暴露 primitive（持久 REPL、递归 `rlm()`、typed 可修订状态、agent-to-agent 队列），而不是编码一条固定工作流。

这和「把 agent 做成产品」不是同一句话。产品 harness 常常为了一种 UX 收紧动作空间；评测 / 训练 harness 的目标是让测量靠近模型真实可达能力，或让同一套 runtime 同时服务 rollout 与 serving。

## 跨报告信号

### Pi：极小核心 + 扩展膜

[Pi coding agent 设计博客](../sources/pi-coding-agent.md) 把产品 harness 收成四工具（`read` / `write` / `edit` / `bash`）和合计 <1000 token 的系统提示+工具定义。核心论题是**上下文必须可见**：作者从 Claude Code 拆出来，是因为系统提示、工具和 mid-session reminders 会在背后改模型看到的东西。MCP、plan mode、todo、background bash、dedicated sub-agent 工具被明确排除出核心，需要时写文件、用 tmux，或让 agent 自己写 TypeScript extension。

这和 Prime Agent / Macaron 同属「改膜不改权重」，但方向相反：Prime Agent 把 REPL 和递归 `rlm()` 做成核心原语以换 expressivity；Macaron 搜索可版本化 HCP；Pi 把能力推到用户空间，核心保持可观察的四工具循环。Terminal-Bench 2.0 上 Pi + Claude Opus 4.5 自报约 50%，但是 mixed-model 榜，不是同模型换膜。OpenClaw 以 Pi 为底盘，所以 [UniClawBench](../sources/uniclawbench.md) 的 OpenClaw 行测的是这张极小膜上的产品框架。原文没有同模型成对实验来区分「四工具已经够用」和「必须靠 REPL/`rlm()` 才能 elicit」。Databricks 的同模型换膜没有测 REPL/`rlm()`。V4.1-Flash 上 Minimal 高于 Standard，但是自家模型加自家 Minimal，不是跨训练分布的交叉。

2026-08 的 `pi-agent-core` 0.84.0 把 durable `AgentHarness`（内部曾叫 runtime2）升成默认导出：改的是可崩溃恢复的 session/lane 契约，**不是**内建 fan-out 子 agent。CLI 到 0.85.1 仍走旧 `AgentSessionRuntime`。

### SoL-Pi：同一张膜上减重复开销

[SoL-Pi](../sources/sol-pi.md) 不换底盘，给 Pi 加一层 opt-in 效率扩展。Auto-research 从 152 个方向里留下四个机制：Action Fusion、Online Context Compact、ObservationPack、Evidence-Preserving Reducer。相对 Pi，token 少 45–49%、成本约低 1/3、平均分保留约 94%；相对 Codex / Claude Code 原生 harness，token 少 35–64%。它和 Macaron MindForge 都做 harness 侧 RSI，但 Macaron 搜的是可评测 HCP 配置，SoL-Pi 搜的是交互环里的重复开销。钉死 Pi 0.84.2 的公开 ExtensionAPI，不用 durable `AgentHarness`。Held-out 主台是 [EdgeBench](../sources/edgebench.md) 的 51 公开题。博客没有四个机制的单机制消融，也没有把 45–49% 的 token 下降拆到各项；94% 是拼装后的平均分，原文没有把它拆成地板泄漏和 EdgeBench 噪声的对照。

独立外部成对数字见 [Databricks](../sources/databricks-coding-agents.md)：同一模型同一 thinking 档，Pi 相对 Claude Code / Codex 任务成本可低到 2× 以上、每轮约少送 3× context；质量多数档接近，但 Opus **max** 通过率 82% vs 89%。这补上了 Pi 设计博客只有 mixed-model Terminal-Bench 的缺口。

### DeepSeek Harness：连 loop 都是插件

[DeepSeek Harness](../sources/deepseek-harness.md)（`dsh`）把「Everything is a Plugin」做到 agent loop / session / sandbox / UI。官方公式是 Agent = Model + Harness；扩展是在 Cordis 树上再挂插件，而不是打补丁。这和 Pi 相反：Pi 的四工具循环是故意不可替换的核心。DSH 的 **Minimal** preset 只留 `bash` + `str_replace_editor`，和 Pi 极小工具面同族，但是可随时切回带 plan/subagent 的 Standard。

[V4.1-Flash Table 4](../sources/deepseek-v41-flash.md) 同一 checkpoint 下 Minimal 的 Terminal-Bench 2.1 为 90.6，Standard 掉到 85.8——满配不一定更强。数字来自模型报告，不是 DSH 自己的实验。内核论文 Cordis（arXiv:2608.25512）只解释热插拔的 PL 语义，不提供 agent 分数。

### Prime Agent：表达性膜 + 权重冻结的 self-improvement

[Prime Agent](../sources/prime-agent.md) 把状态分成 L0 权重 / L1 可见上下文 / L2 REPL 与 subagent / L3 磁盘上的 history、memories、skills、prompts 和 subagent specs。L1–L2 边界是 model–context boundary：L2 的 Python 值必须被序列化才会进入生成。Continual Harness 在轨迹时间内把执行证据写成版本化 L3，不改 L0。ARC-AGI-3 上 Prime Agent + Opus 5 报 95.5% RHAE，对照官方 ARC harness 的 30.2%；作者明确这些外部点只是 situating，因为他们自己的 native 复跑低于官方分。

### Macaron-V1：把 harness 做成可版本化、可搜索的契约

[Macaron-V1](../sources/macaron-v1.md) 的 HCP 把 router、工具 allowlist、MCP、hooks、prompt/skills、session 与 workspace 写成 TOML 契约。最醒目的实验故意不更新权重：122 个 TerminalBench 2.1 基线必失败任务，69 个 adaptive HCP/skill/hook job 后覆盖 122/122。它支持「失败常常是未被正确 elicit 的能力」，但不能当成 adapter 学习曲线或跨代持续学习。与 Prime Agent 的差别：Macaron 搜索的是可评测 configuration portfolio；Prime Agent 在单条长轨迹里持续 refine 运行时状态。

### Dream-RSI：改探索策略代码，用发现树回放

[Dream-RSI](../sources/dream-rsi.md) 也不更新发现 agent 的权重。它在 coding agent 外面加一层编排，动作是「从根或叶子里挑一批、最多 $W$ 个并行延续，或交空批停止」。已完成的发现树被当成回放模拟器：替代策略只能揭开已经存好的子节点，根上的新分支按创建顺序揭开，分支内部不能重排。策略开发 agent 改的是这份策略代码；选中规则只保证固定历史上的平均回放分不低于当前策略。

这和上面几张膜的差别在信号从哪来。Macaron 在真实 harness 上搜 HCP；SoL-Pi 在能力地板内削 token；Prime Agent 在一条轨迹里写 L3 状态。Dream-RSI 的离线分数来自已实现搜索空间的重放，不学习动力学模型，所以它借用的 Dreamer 类比不能读成 world model。Lasso 上同模型调用约少 1.7×；摘要里相对 SimpleTES 的 162× 是 Gemini 调用次数对 gpt-oss-120b 的跨模型比。Pro 的六套 held-out 里只有 RCV1 快于固定探索。

### Harness²：无标签任务流上的两层递归

[Harness²](../sources/harness2.md) 也不更新执行器权重。它把递归拆成两层：任务级用 Propose–Probe–Compose，在没有地面真值的情况下，按同一题的行为对照改八个文件级组件；交卷后，harness 级递归把采纳的执行树和有对照支持的教训写入改进库。下一题仍从公共初始 harness 起步，库只在提议和组合时被查阅，不把上一题的执行 harness 自动挂上去。

主表编辑器是 Claude Opus 4.8，执行器是冻结的 GLM-5 或 Gemini 3.5 Flash，底盘是现成 OpenCode 或 Codex。单步最大增益是 WorkBuddy 宏平均 +11.5（GLM-5 / OpenCode，64.2→75.7）、JobBench +4.5、LAB Partial +11.8 个百分点。并行 $k=5$ 继续加分。Meta-Harness 在训练半或 LAB 的 24 题开发集上用分数做离线搜索，和这条只走测试流的协议不是同一设置；LAB 上 Gemini / Codex 的 Meta-Harness 76.5 仍高于单步 75.6。$k>1$ 的选择器只看匿名交付物，附录里出现过从 95 分候选里挑中 55 的情况。同一模型兼做执行器和编辑器的实验是 OpenCode、$k=2$，五模型都高于自己的 base，不能和主表的 $k=1$ 增益直接相减。

### UniClawBench：读分时 framework 与 model 同量级

[UniClawBench](../sources/uniclawbench.md) 在同一模型换框架：GPT-5.4 在 OpenClaw / EDICT / Nanobot 下 Overall PR 为 0.407 / 0.338 / 0.290。这是「harness 是一等变量」的评测侧证据，不是 expressivity 设计。OpenClaw 的优势被解释为集中式单 agent 信息损失最小；EDICT 付出多 agent token 却常卡在 coordination friction。

### Qwen-UI-Agent：通知 affair 与跨设备 planner

[Qwen-UI-Agent](../sources/qwen-ui-agent.md) 的 harness 不搜索 HCP，也不提供 REPL。它回答两件模型权重单独做不了的事：何时从手机通知主动开工，以及工作流如何在手机和电脑之间不断上下文。核心抽象把 event / affair / task 分开；低风险预备可以先做，改票和支付仍要用户确认。跨平台侧是 OpenClaw-like planner，把 Qwen-UI-Agent 当 GUI subagent 按设备调用，虚拟屏并行且不挡用户。这与 UI-Mate DemoCUA 的差别：后者改的是「同一任务可见哪段程序」；前者改的是任务何时开始、在哪台设备继续。当前证据是定性轨迹（Figure 11–12 / 28–29），没有开关膜的成对数字。

### UI-Mate DemoCUA：示范工作流是一层会改策略的膜

[UI-Mate](../sources/ui-mate.md) 的 Demo Workflow Harness 不搜索 HCP，也不提供 REPL。它把一条录屏变成指针驱动的 subtask checklist：当前目标、完成判据、不含坐标的里程碑，外加 `subtask_complete`。live 截图对示范有否决权，所以这层膜改变的是条件化，不是动作空间本身。33-task self-demo 上严格成功 17.2%→35.4%，是「同一模型换可见程序」的成对证据；作者同时承认 workflow 钉在上下文开头会破坏 KV 前缀，属于膜的实现税。它与 Prime Agent / Macaron 的差别：后两者在冻结 L0 时改 runtime 原语或配置；UI-Mate 还专门为这层膜做了 SFT（对齐 / 错位 / 无关示范），权重和膜是一起训的。§10 的 variant-demo 只有 10 题试点。原文没有「必须围着部分匹配示范做后训练，还是 self-demo SFT 已经够用」的对照。

### Qwen3.8-Omni：给已有 harness 补音视频，再单独做一张实时膜

[Qwen3.8-Omni](../sources/qwen3.8-omni.md) 把两张膜分开。Qwen-MM-Plugins 不替换 Claude Code / OpenClaw 这一类主循环，而是给它们补上音视频：字幕与笔记摘要、懒加载记忆、从教程视频抽出的技能，以及剪辑、翻译、MV 的现成模块。模型可以当主 agent，也可以当这些工具背后的子 agent。Qwen-Live-Harness 则是另一张实时膜：前台继续说话，Qwen Code / Codex / Claude Code 在统一 adapter 后异步执行；打断语音不取消已委托任务；主动提醒来自独立的感知会话；跨通话保留的是文本化的用户事实和画面观察。

Figure 3 把上下文压缩、工作记忆和外部音视频记忆画在主 agent 两侧。正文没有给「关掉这张膜」的成对分数。Table 7 能分开的是另一件事：同一 Omni-Flash，直接读输入对比用 Qwen Code 取证，LVOmniBench 63.3→73.6，OmniVideoBench 的 token/问约降 45.7%。那是执行设置，不是 Live-Harness 的消融。

### StepAudio 3：工具路由写在模型里

[StepAudio 3 Realtime](../sources/stepaudio-3-realtime.md) 的 Voice Agent 在直接回答、轻量工具和异步后端之间选择。用户可以在任务进行时继续说话；澄清、确认，以及没有证据就不要声称完成，都是训出来的模型行为。报告没有可替换的 loop、插件或状态契约。对照上面的 Qwen-Live-Harness：那边前台说话、后台工具跑在统一 adapter 上；这边异步执行留在模型自己的对话环里。

### 训练侧：不要让模型过拟合某一个 scaffold

多份后训练报告把 harness 当训练分布的一部分，而不是评测后才换上的皮肤：

- [KAT-Coder-V2.5](../sources/kat-coder-v2.5.md) 用 harness randomization（format / context-structure / control-flow）和 harness rewriting 对抗 scaffold overfitting。
- [Laguna](../sources/laguna-m1-xs2.md) 用 OpenHands / OpenCode2 / Mini-SWE-Agent 多 harness 训练，并加 IF judge。
- [Kimi K3](../sources/kimi-k3.md) 的 Unified White-Box RL Env 按配置实例化 Kimi Code / Claude Code / Codex / OpenClaw / Hermes，目标是 harness-agnostic RL。
- [MiMo-V2.6](../sources/mimo-v2.6.md) 不用生产 harness 直接做 RL：同一条最小 agent loop 重组出 mini-harness。DeepSWE v1.1 上，未参与训练的 codex、claude code、mini-swe-agent 的 pass@1 均值大约从 50% 到 66%（Figure 10）。这是训练分布里的迁移，不是换评测膜的成对消融。

这些和 Prime Agent 的「先做一张表达性膜再 co-train」方向相反但互补：前者防止模型绑死在一种 CLI，后者假设当前模型还不会用满一张更强的膜。Prime Agent §5 把围着 `rlm` / typed state 的后训练标为预期，原文没有实验拆开表达性膜的收益有多少来自模型本来就会用代码。

### ASPIRE：具身侧的 coding agent 膜

[ASPIRE](../sources/aspire.md) 把 coding agent、机器人执行引擎和 skill library 当成具身侧的膜。内侧标准化的是 per-primitive 多模态 traces、允许的感知/规划/控制 API，以及 coordinator 对可复用修复的入库审计；外侧仍把「写哪份程序、补哪处失败」留给冻结的 coder（仿真 Claude Opus 4.6，真机 GPT-5.5）。底层策略网络若存在也不被更新——对照表里的 OpenVLA / π0 / π0.5 只是评测基线。

与 Prime Agent / Macaron 的同构：L0 不动，改的是可复用程序（这里是 skill library 里的 code-as-policy 修复，不是 HCP TOML 或 REPL skill）。差异：痕迹是机器人感知–运动与接触动力学，不是终端日志；sim-to-real 运的是 know-how（in-context 指导），不是像素或权重。详见 [具身 skill 自进化](embodied-skill-self-evolution.md)。

### EmbodiedSkills：VLA 侧的 guarded runtime

[EmbodiedSkills](../sources/embodied-skills.md) 把膜做成 **policy–runtime 分离**：高层 Qwen3-VL 只提出结构化 skill call，runtime 检查 phase 兼容、输入齐全、artifact 新鲜、动作合法、状态转移合法之后才让低层 [π0.5](../models/pi0.5.md) 跑一个 bounded chunk。技能合同固定，不随任务写入新程序——和 ASPIRE 的库扩张相反。

证据分层要分开记账。RoboTwin 2.0 86.20% / LIBERO 97.40% 是任务特化低层策略相对 LingBot-VA π0.5 / 官方 OpenPI 的执行成绩（Table 2–3）。膜本身的受控证据是同一套 planner + 低层上的消融：去验证 −38.0 pp、去 subtask −51.8 pp、每 subtask 一个 32-step chunk 降到 19.5%（Table 5）。这更接近 UniClawBench「换膜改分」，但低层权重在 RoboTwin 上是按任务 fine-tune 的，不是冻结 L0。

### Agent Swarm：编排策略 ≠ 执行膜

[Agent Swarm](agent-swarm.md) 是 Kimi K2.5 里被 PARL 训出来的 orchestrator 策略：动态创建 frozen subagent、按 critical path 并行。Prime Agent 的递归 subagent 是同一 daemon 上的持久 session 加消息队列，**没有**报告 RL 训练编排器。Kimi 改的是「谁被梯度更新」；Prime Agent 改的是「子 agent 是否作为可恢复计算节点存在」。两者都做 context sharding，但一层是学到的拆任务，一层是 runtime 语义。

## 为什么重要

1. **Agent 分数默认是 (model, harness, budget, context policy) 的联合。** UniClawBench 和 Prime Agent 从两个方向重复这件事：换框架可以超过换模型；换一张更表达的膜可以大幅改变 ARC-AGI-3 曲线形状。[Pi](../sources/pi-coding-agent.md) 再补一条：极小工具面在 mixed-model Terminal-Bench 2.0 上也能进前十，但那不是同模型换膜。[UI-Mate](../sources/ui-mate.md) 再加一维：同一 verifier 下开关一条示范，严格成功可以从 17.2% 到 35.4%。[Qwen-UI-Agent](../sources/qwen-ui-agent.md) 则把「何时开始、在哪台设备继续」也放进膜，但还没有成对消融。读 [Agentic 评测体系](agentic-evaluation-benchmarks.md) 时，harness 和示范都不能再当脚注。
2. **冻结权重仍能改可达策略集。** Macaron 的 HCP 搜索和 Prime Agent 的 Continual Harness 都在 L0 不动时扩展策略。[Dream-RSI](../sources/dream-rsi.md) 同样不改发现 agent，改的是分支、并行和停止的策略代码，并用已记录发现树做回放。[Harness²](../sources/harness2.md) 把持久状态放在跨题改进库里，本题执行 harness 每题从公共 $\mathrm{H}_0$ 重挂；编辑当时看不到评测分数。[ASPIRE](../sources/aspire.md) 把同一判断搬到机器人：改 skill library 而不是 VLA 权重。[EmbodiedSkills](../sources/embodied-skills.md) 的高层 scheduler 也在冻结低层 VLA 上 SFT，但 RoboTwin headline 用的是按任务特化的 π0.5，不能写成「整层 L0 都没动」。这不是持续学习已经发生，而是「elicit vs 学会」必须分开记账。
3. **持久化会保存作弊。** Prime Agent 的 Factorio RCON skill 说明：refinement / memory 若没有独立校验和 rollback，self-improvement 会把 specification exploit 写成可复用程序。这是 [Agent 记忆生命周期](agent-memory-lifecycle.md) 的 gate/rollback 在 agent runtime 上的对应物。
4. **训练与评测正在抢同一层。** 一边随机化、多 harness、harness-agnostic RL，一边把评测膜做得更表达、更可记账。后续 model–harness co-learning 若真发生，这两条线会撞到同一组 primitive（工具 schema、上下文管理、subagent API）。Continual Harness 与 HCP 改的对象已经分开记账，本库不再追一份两者共用的标准契约。Dream-RSI 的选择准则只覆盖历史树上的平均回放分，原文没有下一轮在线质量是否同向的实验（见[来源页](../sources/dream-rsi.md)）。原文没有检验多 harness 训练会不会把策略训成在每张膜上都平庸，也没有把丢状态、记错成本或过早停从排行榜里单独审计出来的实验。

## 相关页面

- 来源：[MiMo-V2.6 技术报告](../sources/mimo-v2.6.md)、[Harness²](../sources/harness2.md)、[Dream-RSI](../sources/dream-rsi.md)、[Pi coding agent 设计博客](../sources/pi-coding-agent.md)、[SoL-Pi 官方博客](../sources/sol-pi.md)、[DeepSeek Harness 官方文档](../sources/deepseek-harness.md)、[EdgeBench 技术报告](../sources/edgebench.md)、[Databricks coding agent 内部评测博客](../sources/databricks-coding-agents.md)、[Prime Agent 技术报告](../sources/prime-agent.md)、[Macaron-V1 技术报告](../sources/macaron-v1.md)、[UniClawBench](../sources/uniclawbench.md)、[KAT-Coder-V2.5 技术报告](../sources/kat-coder-v2.5.md)、[Laguna M.1/XS.2 技术报告](../sources/laguna-m1-xs2.md)、[Kimi K3 技术报告](../sources/kimi-k3.md)、[UI-Mate 技术报告](../sources/ui-mate.md)、[Qwen-UI-Agent 技术报告](../sources/qwen-ui-agent.md)、[Qwen3.8-Omni 技术报告](../sources/qwen3.8-omni.md)、[StepAudio 3 Realtime 技术报告](../sources/stepaudio-3-realtime.md)、[ASPIRE](../sources/aspire.md)、[EmbodiedSkills](../sources/embodied-skills.md)
- 相邻概念：[Agentic engineering](agentic-engineering.md)、[Agent Swarm](agent-swarm.md)、[Agent 记忆生命周期](agent-memory-lifecycle.md)、[Agentic 评测体系](agentic-evaluation-benchmarks.md)、[Forge Agent-Native RL](forge-agent-native-rl.md)、[具身 skill 自进化](embodied-skill-self-evolution.md)
- 比较：[2026 前沿模型技术报告对比](../comparisons/2026-open-model-technical-reports.md)
