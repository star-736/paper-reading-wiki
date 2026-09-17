---
title: "Looped Transformers"
type: Concept
description: "权重共享的循环 Transformer：用同一 block 反复执行增加有效深度；ITT 做层内 token 选择，MoR 联合路由与 KV 缓存，PLT 使延迟和 KV-cache 不随 loop count 增长"
tags: [looped-transformer, PLT, weight-sharing, test-time-compute, depth-recurrence]
timestamp: 2026-09-17
---

## 定义

**Looped Transformer** 用一个共享的 Transformer block $f_\theta$（L 层）反复执行 R 次来替代堆叠 R×L 个独立层，从而在不增加参数量的前提下增大有效计算深度：

$$h^{(0)} = \text{Embed}(x), \quad h^{(r)} = f_\theta(h^{(r-1)}), \quad r=1,\dots,R$$

参数量 N 固定，但有效深度 R·L 随 R 增长。这使得模型能在小参数预算上获得深度敏感任务的能力，是 **test-time compute scaling** 的一条路径——不生成额外推理 token，而在表征空间做迭代精炼（latent reasoning）。

## 谱系

### 标准 looped Transformer（序列式）

原始设计（Universal Transformer [Dehghani et al. 2018]）的 recurrence 是严格序列的：h(r) 必须在 h(r-1) 之后计算。代价随 R 线性增长：

- 延迟：O(R·C_block)——R 次顺序前向传播
- KV-cache：O(R·L·S·d)——每层每 loop 都缓存 KV

这使得深度 loop 在延迟/内存受限的部署中不实用。Giannou et al. [2023] 证明少量 encoder 层在 loop 中足以模拟通用指令集计算机；Yang et al. [2023] 证明 looped Transformer 用 <10% 参数在 in-context learning 上匹配标准 Transformer。

### 效率改进路线

| 方法 | 机制 | 解决的问题 |
|---|---|---|
| **MELT** [15] | 单个共享 KV cache per layer，learnable gating 更新 | KV-cache 随 R 线性增长 |
| **PLT** [16] | CLP（cross-loop position offset）打破序列依赖 + G-SWA（shared-KV gated SWA） | 延迟 + KV-cache 同时随 R 增长 |
| **[MoR](../sources/mixture-of-recursions.md)** | 共享中间层栈 + token 递归路由 + 递归局部 KV 或首轮 KV 共享 | 联合减少独立参数、活跃计算与缓存；另以深度批处理提升吞吐 |
| **LT2** [6] | 用线性/稀疏 attention 替代 softmax | loop 内 attention 的二次复杂度 |

**PLT 以跨 loop 并行化处理延迟和内存随循环增长的问题**：CLP 使 loop 间可并行（延迟 ≈ C_block），G-SWA 冻结 loop-1 KV cache 共享给所有后续 loop（内存 O(L·S·d) 不随 R 增长）。

### PLT 的结构性代价：CLP offset

PLT 的并行化不是免费的。CLP 使 loop r≥2 的输入为：

$$B^{(r)} = \text{Embed}(x) + \text{shift}(h^{(r-1)})$$

token $x_i$ 在 loop r 收到的是 $x_{i-1}$ 的 loop-(r-1) hidden state 而非自身的。这引入了 **per-token positional mismatch**——标准 looped Transformer 不存在此成本，但也不并行。

LoopCoder-v2 的核心贡献是量化这个代价：定义 intrinsic offset cost Ω(r) = (1/S) Σ_i ‖h(r-1)_i - h(r-1)_{i-1}‖₂，发现 Ω(r) 在各 loop 间大致**恒定**。而每个 loop 的 refinement gain 急速递减。固定成本 + 递减收益 = gain–cost 剪刀，使 PLT 在 R=2 饱和。

## 跨报告信号

- **[Mixture-of-Recursions（MoR）](../sources/mixture-of-recursions.md)**（arXiv:2507.10524v3）：Middle-Cycle 共享中间层，比较 expert-choice 与 token-choice，以及递归局部 KV 与首轮共享 KV。360M 基座同 FLOPs 下 MoR-2 的平均分 43.1 vs 42.3，但训练 tokens 为 27B vs 20B；1.7B 基座对照仍略低于普通模型。最高 2.06× 吞吐来自放大 batch 的深度批处理，计时排除 KV 写入／更新，不等于单请求延迟减半。
- **[Inner Thinking Transformer（ITT）](../sources/inner-thinking-transformer.md)**（ACL 2025）：百度与中科院等的层内循环方案，ATR 每步选择 token，RTC 累积输出，step encoding 区分步骤。主体 50B-token 对照中，162M ITT ×4 平均分 42.1，普通同规模 40.4；相对 Loop ×4 FLOPs 少约 30%，但仍高于普通模型。官方 PDF 的摘要规模、附录大模型数值和部分消融数字互相冲突，不能合并成统一的 scaling 结论（详见来源页）。
- **[LoopCoder-v2](../sources/loopcoder-v2.md)**（arXiv:2606.18023v1）：首个在 18T tokens 上从头训练 PLT coder 的大规模实验。7B 模型 R=2 最优（SWE-bench Verified 64.4%），R≥3 退化。gain–cost 框架 + per-loop 可解释性诊断（hidden-state dynamics / attention evolution / output-distribution shift 三镜头三角验证）。
- **[Looped Language Models Improve Compositional Tool Calling](../sources/looped-tool-calling.md)**（arXiv:2608.18171v1）：将 latent recurrence 的评测对象扩展到 tool-call DAG。受控 SFT 中，循环深度主要改善 BFCL 的独立多调用和 NESTful 的 output-to-input 依赖绑定；API-Bank 这类单调用 grounding 任务的收益小且不稳定。Ouro 的 adaptive exit 在保持接近最佳固定深度表现时降低平均循环次数；但原生 Ouro 没有同预训练条件的 non-looped 对照，最直接的架构隔离仍来自 OLMo / Llama retrofit。
- **[BDH-CQ](../sources/bdh-cq.md)**（arXiv:2608.09888v1）：把「从 demonstrations 得到当前任务」与「对 query 做多步 latent refinement」显式拆为 recurrent memory $S_t$ 与 workspace $H_r$。150M ARC 系统报告 29.5% pass@2 / $0.00070 per task，并以 controlled ladders 显示 propagation/copy 外推强、ordering/nesting 有边界。它是 recurrent latent reasoning 的相邻证据，**不是** weight-tied Transformer 或 PLT 的实证：update rule、共享 block、KV-cache 和 loop 并行性均未公开。
- **[Looped World Models](../sources/looped-world-models.md)**（arXiv:2606.18208v1）：把顺序循环接到 **world model 的 action-conditioned latent dynamics**，而不是 token 序列上的 LM。动力学核是 Huginn / Parcae 式 Prelude–Recurrent–Coda；线性保留 $\bar A$ 用负对角离散化保证 $\rho(\bar A)<1$，非线性 Transformer 残差不在该保证内。公开实验是 ScienceWorld / AlfWorld 的五步文本 next-state（约 1B vs 通用 LLM API），**没有**匹配的非循环 world-model 基线，也没有 PLT 的 CLP / shared-KV。延迟解码把观测重建挪到地平线终点；相对 Gemini 的逐步相对增益不能当成延迟解码消融。
- **Huginn-3.5B** [Geiping et al. 2025]：3.5B depth-recurrent Transformer，800B tokens 预训练，推理时最多 50 loops，等效 50B 参数计算预算。使用标准序列 loop（非 PLT），延迟随 loop count 线性增长。
- **Scaling law** [Schwethelm et al.]：loop 一个 block r 次等效 r^0.46 个独立参数层——远低于真正加层的线性等效。这从 scaling law 角度独立支持了 loop 收益递减的结论。
- **稳定性** [Yang et al.]：性能可能在中间 loop depth 达峰后崩溃，提出 fixed-point regularization 稳定循环动态。

## 为什么重要

**路由决定了缓存与可见上下文（本页综合）**：[MoR](../sources/mixture-of-recursions.md) 的递归局部缓存只让深层 token 关注同轮仍活跃的历史 token；首轮 KV 共享则保留完整历史，但复用较浅表征。因此减少 KV 的两种路径具有不同语义。MoR 的连续深度批处理是把不同请求的递归工作合批，提高吞吐；它不解除同一 token 的递归依赖，与 PLT 的并行化机制应分开比较。
**循环粒度与算力分配是两条轴（本页综合）**：[ITT](../sources/inner-thinking-transformer.md) 提醒我们，比较循环模型时除了“共享几层、循环几次”，还要问“哪些 token 在每一步参与”。其层内复用 + Top-K 预算是细粒度路线；不应因都叫自适应深度，就把它视为按置信度永久退出，也不能从 FLOPs 节省推断具备 PLT 的并行延迟性质。ITT 的证据是主表小模型预训练与预算调整，而非任意深度外推保证。

Looped Transformer 代表了一种与本 wiki 已收录的效率路线**正交**的 test-time compute scaling 思路：

- **高效长上下文注意力**（DSA / MSA / 线性注意力）解决的是**单次前向传播内**如何高效处理长序列；
- **MTP**（multi-token prediction）解决的是**单次前向传播内**如何预测多个 token 摊销计算；
- **Looped Transformer** 解决的是**跨多次前向传播**如何迭代精炼表征——增加的是**深度**而非宽度或 token 数。

三者在理论上可以叠加：一个 looped Transformer 的每次 loop 内部可以用稀疏注意力，也可以结合 MTP。但 PLT 的 CLP offset 成本 Ω(r) 是其独有的结构性约束，决定了 loop count 的上限。

**Latent loop 与 explicit CoT 互补**：LoopCoder-v2 发现 explicit CoT + latent loop 在 R=2 时呈超加性（LiveCodeBench +26.9，远超各自单独增益之和）。两机制在不同粒度操作：CoT 分解问题为文本步骤，latent loop 精炼每步底层表征。这与 thinking models（如 [GLM-5](../models/glm-5.md) 的 thinking mode）的 explicit CoT 是互补而非竞争关系。

**工具调用给出了不同于代码生成的行为读数**：[Looped Language Models Improve Compositional Tool Calling](../sources/looped-tool-calling.md) 显示循环的可观察效果不只是最终 task score：增加深度可以先后修正调用数、调用顺序、catalogue 外的幻觉函数、参数名，以及对先前结果的变量引用。这使循环计算成为「内部精炼工具工作流」的候选路径；它不替代外部 planner / execution graph，而是在生成该 graph 前多做 latent refinement。其证据仍限于 static single-turn benchmark，不能直接外推为真实 agent episode 的恢复能力。

**BDH-CQ 给出另一种状态分工**：[BDH-CQ](../sources/bdh-cq.md) 不是让整段 token hidden state 反复通过同一个公开 Transformer block，而是先让 demonstrations 改写 context memory，再在该 memory 条件下迭代 query workspace。它支持「in-context adaptation 和 latent refinement 可在同一 recurrent 系统共存」，却不支持 PLT 的 CLP / shared-KV 结论，也不能用来主张其低成本可迁移到文本 CoT、agent 或其他硬件。这个区分避免将「有 recurrence」误读为同一实现或同一效率曲线。

**LoopWM 把循环对象换成环境隐状态**：[Looped World Models](../sources/looped-world-models.md) 的内循环精炼的是单步转移估计，外循环才是环境时间；作者自己把「循环 = 物理定律反复作用」标成概念类比。它与 LoopCoder-v2 / Ouro 共享「参数共享 + 可变深度」，但不共享评测对象：前者问 next-state 文本是否像环境，后者问代码或 tool-call DAG 是否更好。因此不能用 ScienceWorld EM 68.4 去外推 PLT 的 R=2 饱和，也不能用 LoopCoder 的 gain–cost 剪刀去否定「复杂转移多循环几次」。

## 待追问

- **需实验或作者披露**：**PLT + 稀疏注意力 / 线性注意力**：如果 PLT 的每次 loop 内部用 DSA 或 GDN 替代 full attention，Ω(r) 会改变吗？loop-count 饱和点会移动吗？
- **需实验或作者披露**：**PLT + MTP 叠加**：latent loop 和 MTP 都是 test-time compute scaling 路径，两者是否如 latent loop + explicit CoT 一样互补？
- **需实验或作者披露**：**CLP offset 自适应**：当前 Ω(r) 恒定是 CLP 固定右移 1 位的结果。如果 offset 量随 loop 自适应减小（如 learnable shift amount），能否推迟饱和点？
- **需实验或作者披露**：**跨参数量 scaling**：7B 上 R=2 最优，更大模型上 gain–cost 交叉点是否会不同？Schwethelm 的 r^0.46 scaling law 是否暗示更大模型也需要更多 loop 才能饱和？
- **需实验或作者披露**：**World-model 循环是否也饱和**：LoopWM 主张复杂转移（碰撞、接触）多分迭代、简单转移早停，但没有给出平均 $T$、退出分布，也没有固定 $T=1/2/3$ 的配对。若环境隐状态也有 LoopCoder-v2 那种收益递减，自适应深度的节省幅度会远小于 §3.4 的 100 层思想实验。

## 相关页面

- [Mixture-of-Recursions（MoR）](../sources/mixture-of-recursions.md) — 两种递归路由、两种 KV 语义与吞吐计时边界
- [Inner Thinking Transformer（ITT）](../sources/inner-thinking-transformer.md) — 百度等的 ATR + RTC 层内循环；含正式版证据冲突
- [ITT 实验模型族](../models/inner-thinking-transformer.md) — LLaMA2 风格小模型与规模边界
- [LoopCoder-v2 来源页](../sources/loopcoder-v2.md) — PLT gain–cost 分析的一手出处
- [Looped Language Models Improve Compositional Tool Calling](../sources/looped-tool-calling.md) — 组合式 function calling 的循环深度证据
- [Looped World Models](../sources/looped-world-models.md) — 循环接到 world-model 隐状态；公开对照是通用 LLM，不是 RSSM
- [LoopWM 模型页](../models/loopwm.md) — 约 1B 的 looped latent world model
- [BDH-CQ](../sources/bdh-cq.md) — recurrent memory + latent workspace 的 ARC in-context 推理证据
- [LoopCoder-v2 模型页](../models/loopcoder-v2.md) — 7B PLT coder 模型族
- [Qwen-AgentWorld](../sources/qwen-agent-world.md) — native language world model，与 LoopWM 不是同一实现
- [高效长上下文注意力](efficient-long-context-attention.md) — 正交路线：单次前向传播内的长序列效率
- [多 token 预测](multi-token-prediction.md) — 正交路线：单次前向传播内的多 token 摊销
- [Agentic 评测体系](agentic-evaluation-benchmarks.md) — LoopCoder-v2 在 SWE-bench / Terminal-Bench 上的数据点

关联提问页：[LoopCoder-v2: Only Loop Once for Efficient Test-Time Computation Scaling](../sources/loopcoder-v2.md#相关追问)。
