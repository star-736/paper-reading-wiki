---
type: Concept
title: "Agentic 评测体系"
description: "SWE-bench、Terminal-Bench、BrowseComp、MCP-Atlas 等 benchmark 的作用和可比性风险。"
tags: ["concept", "agentic-evaluation-benchmarks"]
timestamp: 2026-06-06
---

# Agentic 评测体系

## 为什么单一 benchmark 不够

Agentic model 的评测不只是回答正确率。它需要覆盖代码修改、终端操作、网页搜索、工具调用、长周期规划、经济任务和多轮环境反馈。已沉淀的模型报告与 agentic RL 算法论文都在强调 agentic benchmark，但每个 benchmark 衡量的能力不同。

## 常见 benchmark

| Benchmark | 主要能力 | 在报告中的用途 |
| --- | --- | --- |
| SWE-bench Verified | 真实 GitHub issue 修复，经过人工验证的子集 | GLM-5、MiMo-V2-Flash、DeepSeek-V4 都用来衡量 coding agent。 |
| SWE-bench Pro | 更偏行业级、长周期 repository repair | MiniMax-M2、Kimi K2.5 用来衡量复杂工程修复能力。 |
| SWE-bench Multilingual | 多语言软件工程问题修复 | 用于测试模型跨语言代码维护能力。 |
| Multi-SWE-bench | 多 repo / 多语言 issue resolving | MiniMax-M2 用来测试跨仓库迁移和更宽软件工程覆盖。 |
| Terminal-Bench 2.0 | 终端环境中的任务执行 | 测试 shell、文件、环境和多步操作能力。 |
| BrowseComp | 高难度网页搜索与多跳信息综合 | 测试 search agent 和 context management。 |
| WideSearch | 广域、多源信息搜索 | Kimi K2.5 和 MiniMax-M2 用来测试并行搜索、覆盖率和综合能力。 |
| MCP-Atlas | 使用 MCP servers 的多步工具工作流 | 测试真实工具协议下的 tool-use 能力。 |
| tau2-Bench | 双控制环境中的对话 agent | 测试代理在任务约束和用户互动中的执行能力。 |
| Toolathlon | 长周期异构工具任务 | 测试多工具、多步骤任务完成。 |
| Vending-Bench 2 | 模拟经营任务 | 测试长期规划和资源管理。 |
| GDPval-AA | 办公与经济价值任务的 Artificial Analysis 子集 | MiniMax-M2、Kimi K2.5 用来测试 office/cowork agent 能力。 |
| XBench / xbench-DeepSearch | 中文 deep search / deep research split | ARPO、WebSailor 等用来测试深搜索；读分数时要看中文 prompt、judge 和浏览器设置。 |
| AIME / HMMT / BeyondAIME | 数学竞赛与高难数学推理 | DAPO、GSPO、SAPO 常用来追踪 long-CoT / reasoning RL 的 policy optimization 效果；读数时要看 avg@k / pass@k、采样次数和是否只用 rule-based reward。 |
| LiveCodeBench / CodeForces | 代码生成与竞赛编程 | GSPO、SAPO 用来观察 RL 是否迁移到 coding；同名分数需看时间切分、采样次数和 Elo/Pass@1 口径。 |
| MLE Bench Lite | 自动机器学习工程任务 | MiniMax-M2 用来展示 M2.7 的 self-evolution 和 scaffold 修改能力。 |
| OSWorld / WebArena | GUI 与网页环境中的 computer-use | Kimi K2.5 用来测试视觉-操作结合的 agent 能力。 |
|| Mind2Web | 真实网页中的任务完成 | 测试 web agent 的端到端任务执行能力。 |
|| BFCL (v3) | 函数调用 / tool use | 测试模型选择和调用外部工具的能力。 |
|| UniClawBench | proactive agent 真实世界任务，capability-driven（5 维能力），三角色闭环评测 | 首个按能力分解而非场景分类的 proactive agent benchmark；400 双语任务，live web + Docker，跨模型 × 跨框架实验揭示 framework > model。 |
| KAT Code Bench | 快手内部 repository-level SWE 评测，12 语言，pin base commit + runtime + verification | KAT-Coder-V2.5 新建，覆盖 defect fix / feature completion / interface compatibility / cross-module edit / regression repair，压制不可复现环境/flaky test/verifier 耦合等噪声源。 |
| KAT Claw Bench | 快手内部业务导向 tool-use 评测，7 大类（个人办公 / 内容创作 / 软件开发 / 数据分析 / 信息检索 / 自动监控 / 投资分析） | KAT-Coder-V2.5 新建，覆盖短视频/直播/电商/广告/职场自动化场景，补现有 Claw benchmark 在任务粒度过细、场景覆盖不足、偏离业务上下文上的短板。 |
| PinchBench | OpenClaw 框架下的 agentic tool-use benchmark | KAT-Coder-V2/V2.5、GLM-5、MiniMax M2.7 等用来测真实 agent 任务执行效率。 |
| Claw-Eval | autonomous agent 评测 | KAT-Coder-V2 用来测 OpenClaw 框架下的 pass@3 和 average score。 |
| NL2Repo-Bench | 从自然语言规范端到端构建完整软件仓库，测试长周期 repository 构建、跨文件一致性和依赖管理 | Seed2.0 Model Card 新建，归入 Vibe Coding 评测维度。Seed2.0 Pro 仅 27.9，落后 GPT-5.2（49.3）和 Claude-Opus-4.5（43.2），被报告列为优先改进方向。 |
| ImageMining | 视觉中心深度搜索，要求多步工具调用主动挖掘视觉输入（局部裁剪放大再搜索） | GLM-5V-Turbo 自建，217 测试用例 / 7 领域 / 5 类推理，核心约束 \"Visual Jump\" 强制中间推理涉及视觉转换。GLM-5V-Turbo 30.7 vs Kimi K2.5 24.4。 |
| BrowseComp-VL | 视觉版网页浏览深度搜索 | GLM-5V-Turbo 51.9 vs Kimi K2.5 42.9 vs Claude Opus 4.6 35.9。 |
| Design2Code | 从设计图生成前端代码 | GLM-5V-Turbo 94.8，超过 Kimi K2.5（91.3）和 Claude Opus 4.6（77.3）。 |
| Vision2Web | 端到端视觉网站开发，任务规格含 PRD/mockup/reference page，用 workflow-based verification | GLM-5V-Turbo 31.0，弱于 Claude Opus 4.6（43.5）和 Kimi K2.5（33.2）——报告将此归因于 Claude 的更强代码生成能力。 |
| MMSearch / MMSearch-Plus | 多模态搜索 / 带来源溯源的多模态搜索 | GLM-5V-Turbo MMSearch 72.9 / MMSearch-Plus 30.0，前者较 GLM-4.6V 近八倍提升。 |
| AndroidWorld | Android GUI 环境中的动态 agent 任务 | GLM-5V-Turbo 75.7 vs Kimi K2.5 43.1 vs Claude Opus 4.6 62.0。 |
| WebVoyager | 端到端 web agent | GLM-5V-Turbo 88.5 vs Kimi K2.5 84.3 vs Claude Opus 4.6 88.0。 |
| Ainstain Bench | 科学计算编程，测试模型能否实现和操控科研工作流中的计算程序 | Seed2.0 Model Card 新建，归入 Science Discovery 评测维度。Seed2.0 Pro 47.7，超过 GPT-5.2（41.3）和 Claude-Sonnet-4.5（33.7）。 |
| GDPVal-Verified | GDPVal 的可靠子集 + rubric 自动评测，面向端到端真实世界任务 | Seed2.0 Model Card 新建，归入 Real-World Tasks 评测维度。 |

## UniClawBench 的差异化定位

UniClawBench（arXiv:2607.08768，HKU MMLab + Meituan）与上表其他 benchmark 的核心差异在于三点结构性设计，恰好对应它对现有 benchmark 的三个批评：

1. **capability-driven 而非 scenario-driven**：不按 office / research 等场景分类，而是按 5 维能力（Multimodal / Long Context / Skill Usage / Exploration / Cross-Platform）组织任务。场景分类的问题在于模型失败时无法定位瓶颈是视觉感知、长上下文推理还是工具使用——能力分解让 root cause 可诊断。
2. **三角色闭环评测**：executor agent（Docker 中执行）+ hidden supervisor agent（用 checkpoint rubric 评分，pass/fail/continue 三态）+ user simulator（只收可见轨迹 + 粗粒度信号，生成自然语言反馈）。Information Firewall 隔离评分标准与被测 agent，同时保留多轮人-agent 交互。多数现有 benchmark 是单轮的。
3. **真实环境执行**：任务在 fresh Docker 中跑，可访问 live web（非自建镜像或缓存页），缩小 sandbox-to-real-world 鸿沟。

其跨模型 × 跨框架实验有两个对 benchmark 解读直接有用的发现：

- **framework 选择对能力表现的影响常超过模型选择**。同一个 GPT-5.4 在 OpenClaw / EDICT / Nanobot 下 Overall PR 分别为 0.407 / 0.338 / 0.290——读任何 agent benchmark 分数时，framework/harness 是与 model 同量级的变量。
- **"半途失败"现象**：多数模型 intermediate AS 高但 final PR 显著低，说明 agent 能做部分进展但常在长执行链中犯不可恢复错误。这提示单看 step-level score 会高估能力。

详见 [UniClawBench 来源页](../sources/uniclawbench.md)。

## 外部来源

- SWE-bench Verified：OpenAI 对 SWE-bench 子集进行人工筛选，目标是减少原始 benchmark 中不明确或不可解的问题。参考：[Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/).
- BrowseComp：OpenAI 提出的网页浏览困难问题集，用于测试 agent 搜索和综合能力。参考：[arXiv:2504.12516](https://arxiv.org/abs/2504.12516).
- Terminal-Bench 2.0：终端任务 benchmark，GLM-5 和 DeepSeek-V4 都提到环境歧义或 verified subset 问题。参考：[Terminal-Bench 2.0 PDF](https://openreview.net/pdf/574281303882f822808ab57ac3a57a2bddfbc7a3.pdf).
- MCP-Atlas：面向 MCP tool-use 工作流的 benchmark。参考：[MCP Atlas report](https://static.scale.com/uploads/654197dc94d34f66c0f5184e/MCP_Atlas_v4.pdf).

## 解读注意事项

不同报告中的同名 benchmark 未必完全可比。差异可能来自：

- agent framework：OpenHands、Claude Code、内部 harness 等会改变工具集合和提示。
- context management：discard-all、keep-recent-k、hierarchical context management 会显著影响 BrowseComp。
- parallelism：Agent Swarm 等并行 agent 编排会改变 WideSearch、BrowseComp 的延迟和覆盖率。
- judge model：BrowseComp 等需要 judge 的任务会受 judge prompt 和 judge model 影响。
- timeout 和 step budget：Terminal-Bench、MCP-Atlas、BrowseComp 的最大步数和超时会改变结果。
- verified subset：Terminal-Bench 2.0 Verified 与原始 Terminal-Bench 2.0 结果不可直接混用。

## 对已沉淀报告的影响

GLM-5 的强项是非常系统地讨论了 agentic engineering 环境构建和 context management。MiMo-V2-Flash 更强调在较小模型规模下 SWE-bench 与 BrowseComp 的提升。DeepSeek-V4 则把 agent benchmark 放在 512K context、内部 harness 和 1M context 能力背景下理解。MiniMax-M2 更强调统一 scaffold、verifiable reward 和 self-evolution 工作流。Kimi K2.5 则把 Agent Swarm 作为 benchmark 变量，直接比较单 agent、context management 和并行 agent 编排。ARPO 不是发布新模型，而是把 GAIA / WebWalkerQA / HLE / XBench 作为算法测试床，强调同一个 backbone 下 rollout 采样结构也会显著改变 tool-use 分数。Seed2.0 Model Card 把 agentic 评测升级为四维框架（Science Discovery / Vibe Coding / Context Learning / Real-World Tasks），新建 NL2Repo-Bench / Ainstain Bench / GDPVal-Verified 三个内部 benchmark，并诚实标注 coding（SWE-Evo 8.5）和 repository 构建（NL2Repo-Bench 27.9）为短板。

因此，读 benchmark 表时要先问：模型本体、agent harness、工具集合、context strategy、rollout / sampling 策略、policy optimization 算法、reward / judge 设置分别是什么。

LoopCoder-v2 则提醒了另一个变量：**推理时计算量**。同一个 7B 模型，R=2 在 SWE-bench Verified 上 64.4%（超 Kimi-Dev-72B），R=3 掉到 27.6%（低于 R=1 baseline 43.0%），R=4 进一步降到 22.4%--loop count 是与 model / framework 同量级的性能变量，且呈强非单调。详见 [LoopCoder-v2 来源页](../sources/loopcoder-v2.md) 和 [Looped Transformers 概念页](looped-transformers.md)。

JoyAI-VL-Interaction 则展示了另一种评测思路：**不跑 offline benchmark，直接与真实部署产品做 head-to-head 人工盲评**。它选了 Doubao 和 Gemini 的视频通话功能作为 baseline，在 6 个 event-driven 场景（监控告警 / 实时计数 / 实时翻译 / 时间感知 / 实时解说 / 长程记忆）58 个 case 上让 5 名 LLM 研究者盲评 quality + timing 两轴。这种方法的核心论点是：流式交互场景的关键维度（是否在正确时刻行动）无法被 offline video-understanding benchmark 捕捉，只有与真实 turn-based 产品在 live event-driven 设置中对打才能暴露"turn-based 结构性缺陷"这一范式差距。详见 [JoyAI-VL-Interaction 来源页](../sources/joyai-vl-interaction.md)。
