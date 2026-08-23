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
| Terminal-Bench 3.0 | 终端环境中的更长周期任务执行 | GLM-5.3 报告 28.3；其披露为 Claude Code 2.1.207、400K context、128K max output、avg@3，单 rollout 最多 600 turns / 10 小时。 |
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
| BFCL V4 | BFCL v3 升级版，含 WebSearch / Memory / Multi-Turn / No-live / Live / Relevant / Irrelevant 多子域 | Agent-World 用作核心 agentic tool-use 评测之一（Avg 列）；Agent-World-14B 55.8 与 DeepSeek-V3.2-685B 54.1 颇具竞争力。 |
| MCP-Mark | 真实 MCP server 多步工具工作流评测，含 File / Github / Notion / Playground / Postgres 子域 | Agent-World 用作核心 agentic tool-use 评测之一；即便 GPT-5.2 High 也仅 53.1、Gemini-3 Pro 50.8，开源基础模型普遍 <6，是当前最难 MCP 评测之一。与 MCP-Atlas 同为 MCP tool-use benchmark 但来源不同。 |
| MCP-Universe | MCP 工具宇宙，含 Financial Analysis / Browser Automation / Web Searching / Location Navigation / Repository Management 五子域 | Agent-World 用作 Knowledge & MCP 泛化评测，五子域均大幅超过 Qwen3-8B / EnvScaler-8B。 |
| SkillsBench | 高级 AI assistant 长周期规划与执行 | Agent-World-8B 9.2 / 14B 12.6；多数开源基线均分 <20%。 |
| ARC-AGI-2 | 前沿 AI 推理新挑战 | Agent-World-8B 6.5 / 14B 8.5；与 SkillsBench / Claw-Eval 同属 Agent-World 的高级 assistant 泛化组。 |
| WebWalkerQA | 网页 walk 多跳信息检索 | Agent-World 用作 Agentic Search & Coding 评测，增益显著。 |
| GAIA | 通用 AI assistant 真实世界任务 | Agent-World 用作 Agentic Search & Coding 评测（部分子集加速）。 |
| HLE (Humanity's Last Exam) | 极难综合问答 | Agent-World 用作 Agentic Search & Coding 评测（部分子集加速）。 |
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
| AndroidWorld | Android GUI 环境中的动态 agent 任务 | GLM-5V-Turbo 75.7 vs Kimi K2.5 43.1 vs Claude Opus 4.6 62.0；Xiaomi-GUI-0 78.9（超过 UI-Venus-1.5-30B-A3B 77.6）。 |
| RealMobile | 真机 GUI agent benchmark，100 任务 / 14 应用 / 4 能力域 / 57% 跨应用，细粒度 sub-goal + veto + 双验证（XPath + logical semantic rules） | Xiaomi-GUI-0 自建，72.0% success / 85.8% progress。开源最强对手 MAI-UI-8B 仅 33%，接近 Gemini 3.1 Pro 85% / Seed 2.0 Pro 80%。 |
| MobileBench-OL | 中文真机在线 GUI agent benchmark，测任务执行 / 推理 / 噪声鲁棒性 | Xiaomi-GUI-0 引用为 prior real-device work。 |
| WebVoyager | 端到端 web agent | GLM-5V-Turbo 88.5 vs Kimi K2.5 84.3 vs Claude Opus 4.6 88.0。 |
| Ainstain Bench | 科学计算编程，测试模型能否实现和操控科研工作流中的计算程序 | Seed2.0 Model Card 新建，归入 Science Discovery 评测维度。Seed2.0 Pro 47.7，超过 GPT-5.2（41.3）和 Claude-Sonnet-4.5（33.7）。 |
| GDPVal-Verified | GDPVal 的可靠子集 + rubric 自动评测，面向端到端真实世界任务 | Seed2.0 Model Card 新建，归入 Real-World Tasks 评测维度。 |
| AgentWorldBench | language world model 评测，2,170 turn-level 样本 / 7 域 / 9 source benchmark / 5 frontier model / 5 维 rubric（Format/Factuality/Consistency/Realism/Quality） | Qwen-AgentWorld 自建，评估 LWM 模拟环境观测的保真度；每样本配真实环境执行的 ground-truth 观测，训练/评测在数据源级分区保证 OOD。 |

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

GLM-5 的强项是非常系统地讨论了 agentic engineering 环境构建和 context management。GLM-5.3 的发布页则罕见地给出 Terminal-Bench 3.0 的完整执行预算（harness、上下文、rollout 数、turn 与 timeout），也同时展示了为什么发布页的跨模型表不能直接当排名：同表项目的 context、timeout、采样数并不统一；Z.ai Code Bench 更是任务与 checklist 未公开的私有基准。MiMo-V2-Flash 更强调在较小模型规模下 SWE-bench 与 BrowseComp 的提升。DeepSeek-V4 则把 agent benchmark 放在 512K context、内部 harness 和 1M context 能力背景下理解。MiniMax-M2 更强调统一 scaffold、verifiable reward 和 self-evolution 工作流。Kimi K2.5 则把 Agent Swarm 作为 benchmark 变量，直接比较单 agent、context management 和并行 agent 编排。ARPO 不是发布新模型，而是把 GAIA / WebWalkerQA / HLE / XBench 作为算法测试床，强调同一个 backbone 下 rollout 采样结构也会显著改变 tool-use 分数。Seed2.0 Model Card 把 agentic 评测升级为四维框架（Science Discovery / Vibe Coding / Context Learning / Real-World Tasks），新建 NL2Repo-Bench / Ainstain Bench / GDPVal-Verified 三个内部 benchmark，并诚实标注 coding（SWE-Evo 8.5）和 repository 构建（NL2Repo-Bench 27.9）为短板。详见 [GLM-5.3 官方发布博客](../sources/glm-5-3-blog.md)。

因此，读 benchmark 表时要先问：模型本体、agent harness、工具集合、context strategy、rollout / sampling 策略、policy optimization 算法、reward / judge 设置分别是什么。

LoopCoder-v2 则提醒了另一个变量：**推理时计算量**。同一个 7B 模型，R=2 在 SWE-bench Verified 上 64.4%（超 Kimi-Dev-72B），R=3 掉到 27.6%（低于 R=1 baseline 43.0%），R=4 进一步降到 22.4%--loop count 是与 model / framework 同量级的性能变量，且呈强非单调。详见 [LoopCoder-v2 来源页](../sources/loopcoder-v2.md) 和 [Looped Transformers 概念页](looped-transformers.md)。

JoyAI-VL-Interaction 则展示了另一种评测思路：**不跑 offline benchmark，直接与真实部署产品做 head-to-head 人工盲评**。它选了 Doubao 和 Gemini 的视频通话功能作为 baseline，在 6 个 event-driven 场景（监控告警 / 实时计数 / 实时翻译 / 时间感知 / 实时解说 / 长程记忆）58 个 case 上让 5 名 LLM 研究者盲评 quality + timing 两轴。这种方法的核心论点是：流式交互场景的关键维度（是否在正确时刻行动）无法被 offline video-understanding benchmark 捕捉，只有与真实 turn-based 产品在 live event-driven 设置中对打才能暴露"turn-based 结构性缺陷"这一范式差距。详见 [JoyAI-VL-Interaction 来源页](../sources/joyai-vl-interaction.md)。

Xiaomi-GUI-0 的 RealMobile 则把"真机评测"推到另一极端：**benchmark 本身就在物理设备上跑 live 应用**。其与现有 GUI benchmark 的三点结构性差异恰好对应报告对 prior work 的三个批评：

1. **全真机真应用** vs 模拟器/mock 环境——主流商业应用的反模拟器检测使许多应用无法在虚拟化下稳定运行，验证码 / 支付验证 / 登录过期 / 风控拦截等异常态更难在模拟器中复现。AndroidWorld 等模拟器 benchmark 的状态分布偏向简化环境，无法完全捕获真实应用的账号态 / 页面动态 / 业务逻辑。
2. **细粒度 sub-goal 打分** vs 二元 pass/fail——每个任务 3–6 个 sub-goal 按 [0,1] 连续值给 partial credit，配合 veto 机制（不可恢复错误归零）和 conditional branching（多条有效路径均可得满分）。这让"完成大部分但最后一步失败"的 agent 不再与"立即偏离"的 agent 同分，提供诊断信号。
3. **双验证框架**——XML structure matching（XPath 查 UI hierarchy）+ logical semantic rules（sequential constraints 保证操作顺序 + consistency constraints 验证跨步信息传递如"QQ 音乐搜的歌必须和小红书歌单匹配"）。纯 XPath 对 UI 变化脆弱，纯代码检查需应用特定代码，双验证在 robustness 和 scalability 间平衡。

按能力域分解的结果揭示了一个跨模型共性：**Safety & Reflection 是所有模型最弱域**（Gemini 3.1 Pro 也仅 62.5%），安全感知和自纠正行为是当前 GUI agent 的共同瓶颈。而 Foundation 域 Xiaomi-GUI-0 达 100%，与 Gemini 3.1 Pro / Seed 2.0 Pro 并列——基础 UI 操作已接近饱和，不再区分能力。详见 [Xiaomi-GUI-0 来源页](../sources/xiaomi-gui-0.md)。

Agent-World 则展示了第三种评测思路：**把评测本身做成动态诊断 arena，而非静态 benchmark**。它不只在 23 个公开 benchmark 上打分，更把自建的 1978 环境 / 19822 工具生态同时当作训练源与诊断 arena——每轮按层次分类分层采样 K=5 环境/一级类别构造评测集，重新合成全新可验证任务（graph-based + programmatic，配 rubric 或可执行 `V_code`），环境与任务跨轮动态变化防过拟合。agentic diagnosis agent 从失败 trace + 错误分布 + 环境元数据定位弱环境与错误模式，输出定向任务生成指南驱动下一轮训练。这与 JoyAI（与真实产品 head-to-head 盲评）、Xiaomi-GUI-0（真机 live 应用评测）形成对照：JoyAI 解决"流式交互的 turn-based 结构性缺陷无法被 offline benchmark 捕捉"，Xiaomi-GUI-0 解决"模拟器状态分布偏离真实部署"，Agent-World 解决"静态评测无法持续诊断能力缺口并驱动定向学习"——三者都承认静态 offline benchmark 有盲区，但分别从交互范式、部署分布、诊断-学习闭环三个角度突围。另有一个对 benchmark 解读有用的发现：Agent-World 的环境数量 scaling 实验（0→2000 环境，四代表域均分 18.4%→38.5%）首次量化了"训练环境多样性"作为独立性能变量的作用——读 agent benchmark 分数时，除 model / framework / context / sampling 外，**训练时见过的环境多样性与自演化轮数**也是同量级变量。详见 [Agent-World 来源页](../sources/agent-world.md)。

Qwen-AgentWorld 的 AgentWorldBench 则展示了第四种评测思路：**评测 world model 本身而非 agent policy**。它不测 agent 能否完成任务，而测 LWM 能否忠实模拟环境观测——给定交互历史与当前 action，预测的下一观测是否匹配真实环境执行。三个结构设计值得 benchmark 解读时参考：

1. **Reference-grounded judging**——judge 同时拿 ground-truth 观测与预测观测，通过比较两者打分，把开放质量判断转为事实比较。这大幅收窄 judge 幻觉空间，是跨 judge 一致性高的主因：Gemini 3 Flash / Claude Sonnet 4.5 / GPT-5.2 三个 judge 绝对分系统性差异（Gemini 最宽松、GPT-5.2 最严），但模型级排名高度一致（pairwise Spearman ρ = 0.92–0.99）。
2. **Differentiated matching criteria**——把观测内容分三类匹配标准：Deterministic content（echo/文件读取/计算结果）须精确匹配；Pre-existing environment content（预装软件版本、非轨迹创建文件）仅验格式与合理性（模拟器无法复现特定 sandbox 的 gcc patch 版本）；Runtime metadata（timestamp/PID/内存地址/session token）仅验格式与范围。三类区分让 judge 奖励正确结构/语义行为而不惩罚不可复现细节——这对任何涉及环境模拟的 benchmark 都适用。
3. **Double-blind Turing test 校准 judge**——judge 拿两条候选观测（一条真实、一条 world model 生成，随机序）须识别哪条来自真实环境，用 Turing-test 准确率作优化信号迭代精炼域特定 judge prompt。这把 judge 选择从"挑最严/最宽"转为"挑最能区分真实与模拟的"。

另一个对 benchmark 解读有用的发现：Qwen-AgentWorld 在 GUI 域（Android/Web/OS）落后 Claude Opus 4.8/4.6 与 GPT-5.4，论文归因于 multimodal pre-training 优势——即便 GUI 域已用 accessibility tree 文本表示，纯文本 world modeling 仍不能完全捕获 GUI 状态推理。这暗示**评测 world model 时，模态表示本身是与模型质量同量级的变量**。详见 [Qwen-AgentWorld 来源页](../sources/qwen-agent-world.md)。

MinerU2.5-Pro 把「评测方法论」从 agentic 域拉回文档解析域，但它修正的两类偏差对任何 element-matching benchmark 都适用：(1) **匹配粒度偏差**——OmniDocBench v1.5 用固定粒度一对一匹配，会惩罚输出分割与 ground truth 不同的系统（语义完全正确的多行公式因被切成 k 块而非 1 块，分数从满分骤降到接近零）。v1.6 的 MGAM 在预测端搜最优粒度（直接匹配 / 按分隔符分裂 / 划分枚举三阶段取全局最优），使评测对输出粒度中立。这与 Qwen-AgentWorld 的 differentiated matching criteria（按内容类型区分匹配严格度）是同一类反思——**matching 算法本身是 scoring 变量，不是中立基础设施**。(2) **Hard 子集区分力**——top 模型在 Base 子集 0.5 分内近饱和，Hard 子集才拉开差距（MinerU2.5-Pro 94.08 vs GLM-OCR/PaddleOCR-VL-1.5 均 92.01）。这与 Xiaomi-GUI-0「Foundation 域已饱和、Safety & Reflection 才区分」、Agent-World「环境数量 scaling 才显差异」呼应——**饱和 benchmark 上读分要看 hard / 长尾子集，Full/Base 均分会掩盖真实差距**。一个对跨源读分直接有用的教训：同一模型（HunyuanOCR 1.0）自报 OmniDocBench v1.6 = 92.03，MinerU2.5-Pro 统一重测 = 89.87，差 2.16 分——**自报分与统一重测分不可混用**。详见 [MinerU2.5-Pro 来源页](../sources/mineru-2-5-pro.md)。
