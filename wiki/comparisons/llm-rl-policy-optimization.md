---
type: Comparison
title: "LLM RL policy optimization 对比"
description: "以 DeepSeekMath 的 GRPO 为基线，对照 VAPO / DAPO / GSPO / CTPO / SAPO / ARPO / GiGPO / HGPO / SAO / MGPO / IcePop / KPop / CISPO / ECHO：value-based credit assignment、GRPO recipe、sequence-level ratio、prefix-cumulative ratio、soft trust region、agentic partial rollout、history-aware step 组 advantage、异步单 rollout、prompt 权重、mismatch mask、asymmetric clip、环境观测辅助 CE。DPO / KTO 是离线偏好 HALO，与 DAPO 同名不同族，单独成节，不进 GRPO 主表。"
tags: ["comparison", "llm-rl-policy-optimization", "rl"]
timestamp: 2026-06-25
---

# LLM RL policy optimization 对比

## 为什么开这一页

[GRPO](../sources/deepseekmath.md)（DeepSeekMath，2024-04）是后面大多数方法默认对照的 **group-relative clipped objective**：丢掉 PPO critic，用同题组内相对奖励当 baseline。VAPO、DAPO、GSPO、SAPO、ARPO 都在名字上是 policy optimization，但它们解决的不是同一个问题：

- VAPO 修的是 **long-CoT value-model-based PPO 的 critic bias 与 GAE credit assignment**；
- DAPO 补的是 **GRPO 在 long-CoT 数学 RL 大规模复现时的 recipe 缺口**；
- GSPO 改的是 **importance ratio / clipping 的优化单元**，从 token-level 改到 sequence-level；
- CTPO 改的是 **importance ratio 的累积粒度**，从单 token 改到 prefix 连乘（到位置 $t$ 为止），并让 log-space clip 宽度随 $\sqrt t$ 增长；
- SAPO 改的是 **trust region 的形状**，从 hard clipping 改成 temperature-controlled soft gate；
- ARPO 改的是 **agentic rollout 的采样位置**，从完整轨迹采样改到工具反馈后的高熵 step 分叉；
- GiGPO 改的是 **advantage 的构造单元**，在已有轨迹组上用重复环境状态做 step-level 对照，不追加 rollout；
- HGPO 改的是 **step-group 的条件变量**，把同 state 对照再按共同历史拆成层次组，用 bias–variance 权重聚合，不追加 rollout；
- SAO 改的是 **异步下的采样单元**：放弃组采样，单条轨迹立即更新，并用 DIS mask + critic 换回 baseline；
- MGPO 改的是 **prompt-level 的梯度权重**，用最大熵权重降权全对/全错 prompt，聚焦能力边界 prompt；
- TIS / clip-or-pop 改的是 **mismatch token 的处理方式**：同一个 ratio 区间，是压低权重（阻尼）还是整段移除（丢弃）。
- ECHO 改的是 **哪些 token 进监督**：保留 GRPO 的 action PG，另加环境观测 token 的交叉熵；不是新的 ratio 或 clip。

如果把它们都简写成「比 GRPO 更好」，检索时会混掉层级。本页按抽象层级拆开。

**不要把 DPO 和 DAPO 混在一起。** [DPO](../sources/dpo.md)（Rafailov et al.，NeurIPS 2023）是离线偏好对上的闭式分类损失，目标是绕开 RL；[DAPO](../sources/dapo.md)（ByteDance Seed，2025）是 on-policy GRPO 的大规模 recipe。本页主表只收 RL 轴上的方法。DPO / KTO 的位置见下文「离线偏好：DPO / KTO 不在这张 RL 表里」。也不要把 [KTO](../sources/kto.md) 写成 GRPO 变体或 DAPO 近亲。

## 一张表

| 方法 | 论文 / 团队 | 改动层级 | 主要机制 | 解决的痛点 | 主要实验对象 | 和 GRPO 的关系 |
| --- | --- | --- | --- | --- | --- | --- |
| [GRPO](../sources/deepseekmath.md) | DeepSeek-AI，2024-04（DeepSeekMath） | **critic 的有无 / advantage 从哪来** | 同题采 $G$ 条；$\hat A_{i,t}=(r_i-\mathrm{mean})/\mathrm{std}$ 广播到整条；token-level PPO clip；KL 直接进 loss | PPO value model 显存大，且 LLM 常只在最后 token 给奖励 | DeepSeekMath 7B，GSM8K / MATH | **本页基线**：后续行默认在这条目标上改一层 |
| [VAPO](../sources/vapo.md) | ByteDance Seed，2025-04 | **critic / advantage estimation + PPO recipe** | Value-Pretraining、Decoupled-GAE、Length-Adaptive GAE、Clip-Higher、token-level loss、positive-example LM loss、Group-Sampling | critic 初始化偏差、长序列 reward 衰减、长度异质性、binary reward 下正样本稀缺 | Qwen2.5-32B，AIME 2024 | 明确保留 value model；同 prompt 多采样但 advantage 来自 critic，不是 group-relative reward baseline |
| [DAPO](../sources/dapo.md) | ByteDance Seed + 清华 AIR 等，2025-03 | **GRPO recipe / 系统工程** | Clip-Higher、Dynamic Sampling、token-level loss、Overlong Reward Shaping、DAPO-Math-17K | entropy collapse、全对/全错 prompt 零梯度、长 CoT token 权重稀释、截断 reward noise | Qwen2.5-32B，AIME 2024 | 保留 GRPO group-relative clipped objective，补齐 recipe |
| [GSPO](../sources/group-sequence-policy-optimization.md) | Qwen Team，2025-07 | **importance ratio / clipping 单元** | sequence likelihood ratio $s_i=(\pi_\theta(y_i)/\pi_{old}(y_i))^{1/|y_i|}$，response-level clipping | token-level ratio 与 sequence-level reward 不匹配；MoE expert routing 波动使 token ratio 失效 | Qwen3-30B-A3B，AIME / LiveCodeBench / CodeForces | 用 sequence-level ratio 替代 GRPO token ratio |
| [CTPO](../sources/cumulative-token-policy-optimization.md) | UIUC + Michigan + Amazon，2026-05（preprint） | **importance ratio 的累积粒度 + clip 宽度随位置** | prefix 连乘 $\rho^{\mathrm{cum}}_t=\prod_{t'\le t} r_{t'}$（Prop. 1 无偏、Prop. 2 方差严格低于 full sequence）；log-space clip 阈值按 $t^{0.5}$ 放大 | token ratio 丢 prefix 修正有偏；full sequence ratio 的 suffix 只加方差；固定 clip 与 $\mathrm{Var}(\log\rho^{\mathrm{cum}}_t)=t\sigma^2$ 不匹配 | Qwen3-4B / 14B，TIR 数学（AIME25/26、HMMT25、BRUMO25）avg@32 | 保留 GRPO 的 group-relative outcome advantage 与无 critic；只换 ratio 单元与 clip 形状 |
| [SAPO](../sources/soft-adaptive-policy-optimization.md) | Qwen Team，2025-12 | **soft trust region / gate 形状** | sigmoid soft gate + $sech^2$ gradient weight；$\tau_{neg}>\tau_{pos}$ | hard clipping 过脆；GSPO 整条 sequence 被裁、GRPO token 越界即零梯度 | Qwen3-30B-A3B、Qwen3-VL-30B-A3B | 保留 group-based RL，替换硬裁剪为软门控 |
| [ARPO](../sources/agentic-reinforced-policy-optimization.md) | 人大 + 快手，2025-07 | **agentic rollout 采样结构** | 工具反馈后监控 token entropy，在高熵 tool-call step 分叉 partial rollouts；advantage attribution | trajectory-level RL 忽略工具反馈后的 step-level 决策 | Qwen2.5 / Llama3.1 / Qwen3，math/QA/deep search | 把 DAPO/GRPO/REINFORCE++ 当 trajectory-level baseline |
| [GiGPO](../sources/gigpo.md) | NTU + Skywork，NeurIPS 2025 | **advantage 构造（episode + 同状态 step 组）** | 轨迹级 $A^E$ + anchor state grouping 的 $A^S$；折扣回报；不额外 rollout | 长周期稀疏奖励下轨迹级 GRPO 抹掉 step 好坏 | Qwen2.5-1.5B/3B/7B-Instruct，ALFWorld / WebShop / search QA | 保留 GRPO group 与 clipped objective；无重复状态时退回 GRPO |
| [HGPO](../sources/hierarchy-of-groups-policy-optimization.md) | NTU + 东南大学，ICLR 2026 | **advantage 构造（按历史一致性嵌套的 step 组）** | 同 state 的 $0\ldots K$-context groups；每层 relative advantage；按深度加权；不额外 rollout | finite-memory step-wise policy 中同 state step 的 effective prompt 不一致，令 GiGPO 式 state group baseline 偏差 | Qwen2.5-1.5B/7B-Instruct，ALFWorld / WebShop | 保留 group 和逐步 clipped objective；最高层组稀少，固定 $\alpha$ 权重交换 bias / variance |
| [SAO](../sources/single-rollout-asynchronous-optimization.md) | 清华 + Z.AI，2026-07 | **异步采样单元 + critic 回流 + DIS mask** | group size = 1；$r_t=\pi_\theta/\pi_{\mathrm{rollout}}$ 出界则 mask；更快 critic、冻结 attention、Skip-Observation GAE | 组采样在异步下引入 straggler 与更重 off-policy；单条轨迹没有组内 baseline | Qwen3-30B-A3B，TIR 数学 / SWE-Bench Verified；声明用于 GLM-5.2 | 放弃 GRPO 组；DIS 可单独接到 GRPO 上救命；完整 SAO 再加 critic |
| [MGPO](../sources/vibethinker-3b.md) | Sina Weibo，2025-11（VibeThinker-1.5B）-> 2026-06（3B） | **prompt-level 梯度权重** | 最大熵权重 $w(q)=\exp(-\gamma D_{ME}(p(q)\|0.5))$ 降权全对/全错 prompt；GRPO clipped objective + on-policy | 全对/全错 prompt 零梯度浪费；training-inference probability mismatch | Qwen2.5-Coder-3B（VibeThinker-3B），AIME/LiveCodeBench | 保留 GRPO group-relative clipped objective，加 prompt-level weight |
| [IcePop](../sources/ring-1t.md) / KPop | Inclusion AI（Ling Team），2025-10 / 2026-06 | **训练-推理 mismatch 的 mask 形状** | IcePop：GRPO 变体，$k=\pi_{\mathrm{train}}(\theta_{\mathrm{old}})/\pi_{\mathrm{infer}}$，$M(k)=k$ 若 $k\in[\alpha,\beta]$ 否则 $0$（默认 $[0.5,5]$），仍保留 PPO clip；KPop：symmetric binary KL $D_{KL}^B(\pi_{train}\|\pi_{infer})$，两方向都要求 $\leq\phi$ | MoE RL 中训练-推理引擎概率不对齐；KPop 认为固定比率过度 mask 低概率 token | IcePop：Ring-1T / Ring-mini-2.0；KPop：Ring-2.6-1T agentic coding RL | 不替代 GRPO 的 ratio 单元，只控制哪些 token 参与；IcePop 对越界 token **丢弃**（对照 TIS 的阻尼） |
| [CISPO](../sources/minimax-m1.md) | MiniMax-M1，2025-06；[Laguna](../sources/laguna-m1-xs2.md) 采用 2026 | **clip 对象：IS 权重 vs token 更新** | REINFORCE + stop-grad 夹过的 $r=\pi_\theta/\pi_{\theta_{\mathrm{old}}}$；原文 $\varepsilon_{\mathrm{IS}}^{\mathrm{low}}$ 取很大（不下界）、只调上界；全局 token-level 平均；GRPO 组内相对优势；无 KL。Laguna 另配 $(c_{low},c_{high})=(1,4)$ 与 length-weighted LOO | PPO/GRPO 把高 $r$ 的反思 token clip 掉，后续 off-policy 步无梯度 | Qwen2.5-32B 对照 AIME 2024（约 2× DAPO）；M1-456B 生产 RL；Laguna agentic coding | 不改 GRPO 的 advantage 定义；改「夹更新还是夹 IS 权重」，并保证所有 token 进梯度。M2 系列不是源头 |
| [ECHO](../sources/echo.md) | Microsoft Research，2026-05 | **哪些 token 进监督：action PG + env CE** | \(\mathcal{L}_{\mathrm{GRPO}}(\mathcal{A})+\lambda\mathcal{L}_{\mathrm{Env}}(\mathcal{O}')\)；只要终端输出块；同一前向、无额外 rollout | 稀疏终局奖励让失败轨迹零梯度，但观测已在 rollout 里 | Qwen3-8B/14B、OT-SFT；TerminalBench-2.0 | 保留整条 GRPO；另加 on-policy 观测 CE。不是独立 world model，也不是 MoE EP 的 ECHO |

## 关键分叉：critic、token、sequence、step 四个「单位」

### GRPO：丢掉 critic，用同题组内相对奖励当 baseline

[DeepSeekMath](../sources/deepseekmath.md) §4.1 是这条目标的定义文。它不改 PPO 的 token-level ratio / clip，只换 advantage：不再训 $V_\psi$，而对同一 $q$ 采 $\{o_i\}_{i=1}^{G}$，用组内标准化奖励。Outcome 把 $\tilde r_i$ 赋给该序列每个 token；Process 把后续步奖励求和。KL 用无偏估计直接加进 loss，不写进 $r_t$（Eq. 3–4、Figure 4）。

主实验是 7B 数学 Instruct → RL：只用 GSM8K+MATH 的 CoT 子集，MATH 46.8→51.7、GSM8K 82.9→88.2，且 OOD（CMATH 等）也涨。作者用 Figure 7 解释涨分：Maj@K 升、Pass@K 不动——「把正确答案从 TopK 抬稳」，不是扩张基本能力。Reward 仍是神经网络 RM，不是后来 RLVR 的 rule verifier。

后面每一行都默认读者已经知道这条目标。DAPO 补 recipe，GSPO 改 ratio 单元，VAPO 把 critic 请回来，SAO 拆掉组。不要把「2025–2026 的 GRPO 变体」回写成 DeepSeekMath 原文事实。

### VAPO：重新引入 critic，把 long-CoT credit assignment 做稳

VAPO 与 DAPO / GSPO / SAPO 的第一处分叉在于 **advantage 从哪里来**。后三者主要围绕 value-model-free group-relative advantage 调整 ratio、clipping 或 recipe；VAPO 保留独立 value model，用每个 token 前缀的 value estimate 做更细粒度 credit assignment。它用 Value-Pretraining 消 critic 初始化偏差，用 $\lambda_{critic}=1$ / 独立 $\lambda_{policy}$ 的 Decoupled-GAE 分开 critic 与 actor target，再令 $\lambda_{policy}=1-1/(\alpha l)$ 适应 response 长度。

VAPO 也复用了 DAPO 的 Clip-Higher 和 token-level loss，并把固定 8192 trajectories 从 8192 prompts × 1 response 改成 512 prompts × 16 responses。这个 Group-Sampling 只是 rollout allocation；VAPO 的 advantage 仍来自 critic，不能因“同 prompt 采 16 条”就归入 GRPO。论文在 Qwen2.5-32B / AIME 2024 上报告 60.4，DAPO 为 50，但没有报告 value model 的额外显存、FLOPs 或 wall-clock，因此当前只能说 update-step / benchmark efficiency 更高，不能说总训练成本更低。

### DAPO：还是 token-level GRPO，但把 recipe 做对

DAPO 的 objective 仍然是 [DeepSeekMath](../sources/deepseekmath.md) 定义的 token-level ratio + clipped objective + group reward normalization。它的贡献是把大规模 long-CoT RL 里几个会让这条目标跑不起来的工程细节补上：

- Clip-Higher 放宽上界，防止低概率探索 token 被上界压死；
- Dynamic Sampling 保证 batch 中每个 prompt 既有对也有错，从而有非零 advantage；
- token-level loss 让长 CoT 的 token 不被 sample-level 平均稀释；
- overlong shaping 避免把截断样本粗暴当错，降低 reward noise。

所以 DAPO 是「可复现的 GRPO 系统 recipe」，而不是从理论上否定 token-level ratio。

### GSPO：否定 token-level ratio，转到 sequence-level

GSPO 的判断更根本：既然 reward 是整条 response 的 reward，那么 importance sampling / clipping 也应该按整条 response 来。它认为 GRPO 的 token-level ratio 在长序列里会累积高方差，尤其 MoE 中 expert routing 波动会让 token ratio 失真。

GSPO 的 sequence-level ratio 让同一 response 内所有 token 共享一个权重，训练目标和 sequence reward 对齐。代价是它牺牲了局部 token adaptivity：一条 sequence 里少数 token off-policy，hard clipping 可能让整条 sequence 的有效信号被压掉。

### CTPO：不争 sequence 也不争单 token，而争 prefix 的精确修正

[CTPO](../sources/cumulative-token-policy-optimization.md) 把 GSPO 与 GRPO 的分叉重写成一条 bias–variance 谱：token ratio 有偏（丢 prefix 修正）、full sequence ratio 无偏但方差高、GSPO 的长度归一化 ratio 数值稳但**作为 IS 修正仍有偏**。它给出的第三点是 prefix 连乘 $\rho^{\mathrm{cum}}_t=\prod_{t'\le t}r_{t'}$：在 token-level policy-gradient 形式下，suffix $\epsilon_t=\prod_{t'>t}r_{t'}$ 的条件期望为 1，只乘进方差不减偏差，所以截到位置 $t$ 同时拿到无偏与严格更低的方差（Prop. 1–2）。

这与 GSPO 的动机不重合。GSPO 要的是「ratio 单元与 reward 单元对齐」加 MoE routing 稳定性，代价是接受 IS 偏差；CTPO 要的是「在 token-level MDP 的时间结构下做精确 prefix 修正」，代价是 ratio 的方差随位置增长——于是它必须配套第二个改动：$\mathrm{Var}(\log\rho^{\mathrm{cum}}_t)=t\sigma^2$ 让固定 clip 区间在晚 token 上 clip 到约 20%、早 token 近 0%，所以把 log-space 阈值按 $\sqrt t$ 放大。

注意 CTPO 的 clip 改动与 [SAPO](../sources/soft-adaptive-policy-optimization.md) 不在同一维度：SAPO 换 trust region 的**形状**（hard clip → sigmoid soft gate），CTPO 换 trust region 的**宽度随位置的变化**（仍是 hard clip，只是区间随 $t$ 变宽）。两者都批评固定 clip，但没有互相比较。

证据边界：CTPO 的实验只有 dense Qwen3-4B / 14B 上的 TIR 数学，GSPO 基线是它自己的复现而非 Qwen3-30B-A3B 的 MoE 场景；14B 上平均增益主要来自 AIME 两项，BRUMO 25 略低于 GSPO、HMMT 25 介于 GRPO 与 GSPO 之间。Table 1 把 GSPO 标为「有偏」是 IS 修正口径的判断，不等于否定 GSPO 在 MoE 稳定性上的实测结论——两边没有同一评测面的对照。

### SAPO：保留 sequence coherence，但恢复 token adaptivity

SAPO 继承 GSPO 对 sequence coherence 的追求，但不接受 GSPO 的 hard sequence clipping。它用 smooth gate 做 token-level attenuation：正常情况下平均 token gate 近似 sequence-level gate，满足 GSPO 的对齐诉求；异常情况下只下调 outlier token，不让 near-on-policy token 一起陪葬。

这就是 SAPO 的定位：**GSPO-like when ratios are coherent, token-adaptive when they are not**。

### ARPO：不争 ratio，而争 rollout 预算投到哪里

ARPO 和前三者不在同一轴上。它不主要讨论 token ratio 或 clipping 形式，而是问多轮 agent 轨迹里哪里值得采更多样本。它的回答是：工具返回后，模型前 10–50 个 token entropy 升高，说明这里是 step-level tool-use 行为的关键不确定点；因此从这里 branch partial rollouts 比单纯采完整 trajectory 更有效。

这意味着 ARPO 可以理论上叠加 DAPO / GSPO / SAPO 的 policy loss；它改的是 rollout collection / sampling structure。

### GiGPO：不争采样预算，而争已有轨迹组里怎么拆 step

[GiGPO](../sources/gigpo.md) 和 ARPO 问的是相邻问题——多轮 agent 的 step 好坏怎么分开——但花钱的位置相反。ARPO 在高熵工具步**额外分叉**；GiGPO 假设 group 已经按同一任务、同一初始状态采好，然后用重复环境状态做 **anchor state grouping**，把同一 $s$ 上的不同动作收成 step-level 组，用折扣回报算 $A^S$，再与轨迹级 $A^E$ 相加。

因此 GiGPO 改的是 **advantage 的构造单元**，不是采样图，也不是 importance ratio 的粒度（逐步 $\rho_\theta(a_t)$ 仍在）。无状态重复时 $A^S=0$，算法退回 GRPO；这是作者标明的下界，不是额外保证。附录把 DAPO 的 dynamic sampling + clip-higher 接到 GiGPO 上（`GiGPO_dynamic`），WebShop / 1.5B 成功率 GRPO 56.8 → DAPO 66.1 → 75.0，用来支持它与单轮 group recipe 正交。主实验是 ALFWorld / WebShop / search QA 上的 Qwen2.5-Instruct，不是 SWE 或生产 harness。

### HGPO：不否定 state group，而是补上它漏掉的历史条件

[HGPO](../sources/hierarchy-of-groups-policy-optimization.md) 以 GiGPO 的 step group 为 0-context 下界：共享当前 state 的 step 先在一起，但再检查最近 $1\ldots K$ 段 state history 是否也一致。历史越长的 group 更接近同一 effective prompt，却更小、更容易有高方差；因此 HGPO 不只取最深的 Oracle-like group，而是对每层 group-relative advantage 以 $(k+1)^\alpha$ 加权。它仍是同一组完整 rollout 上的离线 hashing，也仍使用逐步 clipped importance ratio；改动完全在 advantage estimator。

这与 GiGPO 的差别不是“又多一层 hierarchy”这么简单。GiGPO 将 episode advantage 与 same-state $A^S$ 相加；HGPO 用多个 history-conditioned $A_k^H$ 取代这两项，且 Table 5 显示再塞回 trajectory-level advantage 在多数格子会下降。HGPO 的 Figure 2 将不同 history 的 same-state comparison 与 Oracle comparison 的 advantage 差作为其偏差证据，但只在自己的 ALFWorld / WebShop training 统计，不能据此断言所有 GiGPO 应用均有同等问题。

HGPO 主表在 1.5B 的收益更明显；7B / $K=4$ 的 ALFWorld OOD 和 WebShop score 反而略低于 GiGPO。故当前最佳表述是：它在两种受控环境中提供了**history consistency 是可操作 credit-assignment 轴**的证据，不是对 state grouping 的统一替代。固定 $\alpha$、稀少的 deep group、摘要/latent memory 未定义怎样 hash，以及与 ARPO 的同预算比较，都仍缺少证据。

### SAO：不争组内怎么拆 step，而争异步下还要不要组

[SAO](../sources/single-rollout-asynchronous-optimization.md) 和 ARPO / GiGPO 不在同一层。后两者默认 group 已经采到，再决定探索预算或 step 对照；SAO 认为异步长周期里 **组本身是错配**——必须等最慢样本，而且在线环境常常只有一条反馈。

因此它同时改三件事：采样单元变成 1；baseline 从 group-relative 回到 critic（Length-Adaptive GAE 直接引用 VAPO）；trust region 从 PPO clip 变成 DIS 的出界 mask，并丢掉 $\pi_{\theta_{\mathrm{old}}}$。Figure 3 把前两项拆开：DIS 单独加到 GRPO 上就能跑满约 1000 step，单 rollout + critic 才在约 400 step 后继续涨。不要和 [SAPO](../sources/soft-adaptive-policy-optimization.md) 混名。

这使 SAO 更接近「异步版 VAPO」，而不是又一个 GRPO recipe。代价是重新承担 critic 显存与 value 预训练；论文没有报告这笔成本。生产声明（GLM-5.2 pipeline）和 30B 实验表不是同一份证据。

### MGPO：不争 ratio 也不争采样位置，而争 prompt 的梯度贡献

MGPO 和前述方法在又一条不同的轴上。它不改 token-level ratio（DAPO/GSPO/SAPO 的轴），不改 rollout 采样结构（ARPO 的轴），也不改同状态 step 组 advantage（GiGPO 的轴），而是改 **batch 内每个 prompt 对梯度的贡献权重**。

核心观察：GRPO 里一个 prompt 采 $G$ 条 response，如果全对（$p(q)\approx 1$）或全错（$p(q)\approx 0$），group-relative advantage $A_i$ 全为 0 或信号极弱，梯度贡献接近零——这些 prompt 在 batch 里浪费了采样预算。DAPO 的 Dynamic Sampling 用 hard filter 丢弃这些 prompt（只保留有对有错的），MGPO 则用连续的指数权重 $w(q)=\exp(-\gamma D_{ME}(p(q)\|0.5))$ 降权它们，保留在 batch 里但降低梯度影响。

两种策略的 trade-off：DAPO hard filter 更彻底（全对/全错直接丢弃，采样预算全部给有信号的 prompt），但需要一个额外的过滤+重采样 pass；MGPO soft weighting 实现更简单（权重直接乘进 loss），但全对/全错 prompt 仍占采样预算。两者动机完全一致——聚焦能力边界 prompt——区别在实现粒度。

MGPO 还有一个 DAPO 不涉及的维度：它显式处理 training-inference probability mismatch。VibeThinker-3B 发现 rollout engine 优化推理吞吐量后，training-inference 概率失配被放大，因此改为全 on-policy（参考 [14, 15] 的稳定化策略）。这是 GRPO 系实践中常见的工程问题，DAPO 论文未讨论。

### KPop / IcePop：不争 ratio 也不争采样，而争「哪些 token 参与」

KPop 和前述方法在又一条轴上。它不改 token-level ratio（DAPO/GSPO/SAPO 的轴），不改 rollout 采样结构（ARPO 的轴），也不改 prompt 权重（MGPO 的轴），而是控制**训练-推理 mismatch 下哪些 token 被允许参与梯度**。

[IcePop](../sources/ring-1t.md)（Ring-1T，2025-10）是这一层的一手出处。它是 GRPO 变体，目标里有**两条比**：校准比 $k=\pi_{\mathrm{train}}(\theta_{\mathrm{old}})/\pi_{\mathrm{infer}}$ 决定 $M(k)$（区间内乘 $k$、越界置 0），PPO 比 $r=\pi_{\mathrm{train}}(\theta)/\pi_{\mathrm{train}}(\theta_{\mathrm{old}})$ 仍做 clip。默认 $[\alpha,\beta]=[0.5,5]$。Appendix A.1 明确对照 TIS：TIS 对越界 token 仍更新、只加 moderating coefficient；IcePop **丢掉**这些梯度。作者经验是留下的小扰动会放大并把 benchmark 平台期化。Ring-mini-2.0 上 IcePop 相对 TIS 拉开 AIME25，Vanilla GRPO 约 100–150 step 后崩（Figure 5）。

[Ling-2.6](../sources/ling-2.6.md) §3.2.3 把前代 IcePop 转述成 uniform constant-ratio constraint，并主张固定比率会过度 mask 低概率 token。这是 **KPop 的动机陈述**，不是 IcePop 原文自己的定位。KPop 用 symmetric binary KL（全词表看成「当前 token vs 其余」）替代固定比率，两方向都要求 $<\phi$，单超参控制。这与 GLM-5 的 DIS、Miles 的 TIS / clip-or-pop 是同一层问题的不同边界形状，与 GSPO/SAPO 正交可组合——IcePop Appendix 也写它不依赖 sequence-level 优化。

详见 IcePop 原文 [Ring-1T](../sources/ring-1t.md) §2.3.2 / Eq. 1–3，以及 KPop 的 [Ling-2.6](../sources/ling-2.6.md) §3.2.3。

### TIS / clip-or-pop：不争 ratio 单元也不争 mask 依据，而争「阻尼还是丢弃」

[Miles v0.1](../sources/miles-v0-1.md) §3.4 提供的前缀修正与 KPop / IcePop 同层——都作用在 importance ratio 上、都不改 loss 的 ratio 单元。它把这一层做成与 advantage estimator **并列的可替换组件**，并给出两种形状：

- **TIS**（truncated importance sampling）把 $r=\exp(\log\pi_{\text{train}}-\log\pi_{\text{rollout}})$ 夹到配置区间后用作该 token policy-gradient loss 的 per-token 权重，默认区间 $[0,2]$。区间只作用在上尾，因为 ratio 不会低于 0。极端 token 被**阻尼**而不是丢弃。
- **clip-or-pop** 把区间外的 token 权重置零（等价于 loss mask），区间内原样通过。极端 token 被**丢弃**而不是阻尼。

两者的分工值得和上面的 KPop / IcePop 对照：IcePop 与 KPop 争的是「哪些 token 参与」（以及 IcePop 用什么量做门：$k=\pi_{\mathrm{train}}/\pi_{\mathrm{infer}}$），Miles 争的是「参与的方式」——同一个区间，是压低权重还是整段移除。IcePop 原文已经站在丢弃一侧反对 TIS 的阻尼；Miles 把这两种形状做成可替换组件。三种修正都上报同类指标（IcePop 报 1–2‰ clip ratio；Miles 报夹前 ratio、被裁比例、$|r-1|$ 的平均绝对偏差），因此可以在同一张曲线图上比较。

Miles 的 §9 案例给了这一层在真实配置下的量级：BF16 训练 + FP8 服务、GLM-5.2 744B、100 step 上 train–inference 分歧均值 0.0369，由 TIS 在更新里吸收。**本页综合**：这说明在 MoE + 低精度服务的生产配置里，「结构性成因去掉之后仍有可测残差」是常态而非异常，所以第四层修正不是可选装饰。

### CISPO：夹的是 IS 权重，不是 token 更新

[CISPO](../sources/minimax-m1.md)（MiniMax-M1，2025-06；MiniMax-M2 系列不是源头）和前述方法的第一处分叉是 **clip 对象**，不是 clip 区间的左右宽度。PPO/GRPO/DAPO 夹的是 token 更新：高 $r_{i,t}$ 的反思 token（`Wait` / `Aha`）第一次 on-policy 更新后就被裁掉，后面多轮 off-policy 吃不到梯度。M1 在 hybrid 架构、每 generation 16 轮 off-policy 时，DAPO Clip-Higher 也不够。CISPO 改成对 $r=\pi_\theta/\pi_{\theta_{\mathrm{old}}}$ 做 stop-grad 夹紧，再乘 $\hat A\log\pi_\theta$，**所有 token 都进梯度**。Advantage 仍用 GRPO 组内标准化；loss 是全局 token 池平均；无 KL。原文明确「不下 IS 下界、只调 $\varepsilon_{\mathrm{IS}}^{\mathrm{high}}$」，**没有**写出 $(1,4)$。

对照实验在 Qwen2.5-32B-base + DAPO 数学数据上：同 step 优于 GRPO/DAPO，约一半 step 追上 DAPO（M1 Figure 2）。这是算法效率证据，不是 456B hybrid 上的墙钟。

[Laguna](../sources/laguna-m1-xs2.md)（2026-05）是首个公开「消融 vs GRPO / GSPO 后选 CISPO」的团队，理由是「最终评测质量 + 训练稳定性组合最好」。Laguna 的采用配方才是 asymmetric $(c_{low},c_{high})=(1,4)$（有效 $[0,5]$）+ length-weighted leave-one-out（$b_i=\sum_{j\ne i}w_j r_j/\sum_{j\ne i}w_j$，$A_i=r_i-b_i$）。reward 只让 binary task verifier 给正分、其余全小负惩罚——长周期信用分配全靠末尾 verifier 经 advantage 传到每个 token。

Laguna 这组数字隐含的方向判断（几乎不约束 $\rho\to 0$、上界裁到 5）是**采用方超参**，不要回写成 M1 原文。它与 DAPO Clip-Higher「放宽上界」是否互补，以及 CISPO 的 token-level IS 权重能否接到 GSPO/SAPO 上，两边都没有消融。

## 离线偏好：DPO / KTO 不在这张 RL 表里

上面每一行都还在 **on-policy RL** 里改东西：advantage 从哪来、ratio 按什么单元 clip、batch 里哪些 prompt/token 进梯度。 [DPO](../sources/dpo.md) 问的是前一个问题：**KL-constrained reward max 能不能根本不跑 RL。**

它的回答是闭式的。最优策略 $\pi_r\propto\pi_{\mathrm{ref}}\exp(r/\beta)$ 反解出 $r=\beta\log(\pi/\pi_{\mathrm{ref}})+\beta\log Z$；Bradley-Terry 只看奖励差，$Z$ 消掉，得到对静态 $(y_w,y_l)$ 的 logistic 损失。训练时不采样、不拟合独立 RM、不需要 critic。这和 DAPO 的 Clip-Higher / Dynamic Sampling 没有共同改动点。

和本页方法的分叉有三层：

| | DPO | KTO | 本页 RL 方法（DAPO / GSPO / …） |
| --- | --- | --- | --- |
| 数据 | 离线偏好对，来自 $\pi_{\mathrm{ref}}$ / $\pi_{\mathrm{SFT}}$ | 离线二元 desirable / undesirable，**不需要 pair** | 当前策略 on-policy rollout |
| 监督 | 成对比较（Bradley-Terry） | 前景理论价值函数（增益凹 / 损失凸的 logistic） | 可验证奖励 / group-relative advantage / critic |
| 优化 | 一条分类损失，无 RL 环 | 一条 HALO 损失，无 RL 环 | clipped policy gradient（或 value-based PPO） |

它和 [OPD](on-policy-distillation.md) 也常被一起说成「不用 RL」，但 OPD 的监督是 teacher 在 **student 自己采样轨迹** 上的 reverse-KL，不是离线偏好标签。DPO 原文实验停在 6B、情感 / 摘要 / 单轮对话；2026 已收录报告的后训练主轴已经换到 RLVR + MOPD。把 DPO 放进本页是为了挡住「搜 DPO 落到 DAPO」和「把闭式偏好当 GRPO 变体」两条检索事故，不是主张它仍是当前 agentic 栈的一等算法。

[Iterative RPO](../sources/iterative-rpo.md)（Pang et al.，TRL `rpo_alpha`）仍在这条离线偏好轴上，只是给 winner 再加长度归一化 NLL：$\mathcal{L}=\mathcal{L}_{\mathrm{DPO}}+\alpha\mathcal{L}_{\mathrm{NLL}}(y_w)$，$\alpha=1$。动机是纯 DPO 会压低 chosen logprob。完整论文还按最终答案对错重新采样 pair 并迭代；TRL 默认只实现损失项。它和 [VAPO](../sources/vapo.md) 的 positive-example LM loss 同构（正样本再 SFT），一个挂 DPO、一个挂 PPO。不要把它写进上表当 GRPO 变体，也不要和 Regularized Preference Optimization（SFT 权重约 0.005）或 ORPO 混名。

[KTO](../sources/kto.md)（Ethayarajh et al.，ICML 2024）也仍在这条离线轴上，**仍然不进上面的 GRPO 主表**。它不改 DPO Eq. 7，而是换人类价值函数：二元信号 + logistic $v$ + KL 参考点。1B–30B 上匹配或超过 DPO，主数字是 Zephyr-β-SFT + UltraFeedback 的 GSM8K 40.0→53.5。不要把「不需要 pair」夸成生产 agentic RL 替代，也不要用它解释「2026 为何不用 DPO」——那条找论文也答不了。

## 与模型报告的关系

- [DeepSeekMath](../sources/deepseekmath.md)：GRPO 的定义文与发布检查点。主实验是 7B 数学 Instruct→RL、神经网络 RM，不是后来的 RLVR；Figure 7 的 Maj@K↑ / Pass@K≈ 是「涨分来自分布校准」的原文锚点。
- [MiniMax-M1](../sources/minimax-m1.md)：CISPO 的定义文与发布检查点。夹 IS 权重、不丢 token；原文不下 IS 下界。Laguna 的 $(1,4)$ 是采用方超参。
- [Qwen3 技术报告](../sources/qwen3.md)：官方 2025-05 报告的 reasoning RL 阶段写的是 GRPO；DAPO/GSPO/SAPO 都是后续或外部算法论文，不能回写成原报告事实。
- [VAPO](../sources/vapo.md)：主要实验直接使用 Qwen2.5-32B base 且不加 SFT；这是外部 RL 算法实验，不是 Qwen2.5 / Qwen3 官方训练 recipe，也不发布独立模型实体。
- [Qwen3-VL 技术报告](../sources/qwen3-vl.md)：原报告有 own post-training pipeline；SAPO 论文提供后续/配套 Qwen3-VL RL 训练证据，说明 SAPO 用在 Qwen3-VL-30B-A3B preliminary cold-start 上，但不替换源报告 pipeline。
- [VibeThinker-3B](../sources/vibethinker-3b.md)：MGPO 是 VibeThinker 系列（1.5B → 3B）的自研 RL 算法，不是外部算法论文的复现。VibeThinker-3B 的 Long2Short RL（零和 length-aware reward shift）是在 MGPO 之上的 efficiency 优化，与 DAPO 的 overlong shaping 解决的是问题的两面——DAPO 处理截断样本的 reward noise，Long2Short 主动 reshape 正确轨迹的长度偏好。
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)：本页可作为该概念页里「RL policy optimization 子谱系」的展开。
- [GLM-5 技术报告](../sources/glm-5.md) / [GLM-5.3](../models/glm-5-3.md)：报告里的 DIS 与 [SAO](../sources/single-rollout-asynchronous-optimization.md) 是同一层机制；SAO 论文补上单 rollout + critic，并声明用于 GLM-5.2。5.3 博客的 compaction 不在该 PDF。
- [MinerU2.5-Pro](../sources/mineru-2-5-pro.md)：Stage 3 是 GRPO + DAPO recipe 的**非 agentic 应用**——文档解析格式对齐。三个特点值得与 agentic RL 对照：(1) reward 直接用评测指标（edit distance / CDM / TEDS / IoU），无需单独 reward model，也无 LLM-as-judge，比 agentic RL 的 reward 设计更「机械」但更可信；(2) 训练数据按 reward 分布过滤（剔除过高/过低 reward，保留中段）以最大化有效 policy gradient 信号——与 [MGPO](../sources/vibethinker-3b.md) 的 prompt-level 加权、DAPO 的 dynamic sampling 同属「聚焦能力边界样本」家族，但用 hard filter 而非 soft weight；(3) 增益小（+0.45）但定向——主要提公式 CDM（+0.81），印证 GRPO 对「交叉熵等权 token 与 sequence-level 结构指标错位」的修补。这说明 GRPO+DAPO recipe 不限于 long-CoT reasoning / agentic，对任何「token-level loss 与任务级指标错位」的结构化输出任务都适用。

## 待追问

- VAPO 的 Length-Adaptive GAE 能否迁移到 code / agent 长轨迹？其 $\lambda(l)$ 实现是否需要下界 clipping，论文没有说明。
- VAPO 相对 DAPO 的 10.4 分与更少 update steps，扣除 value model 的额外显存、FLOPs 和预训练 50 steps 后，wall-clock / total-token 效率是否仍占优？
- DAPO 的 token-level loss 与 GSPO 的 sequence-level ratio 是否冲突？一个在 loss reduction 上按 token 平均，一个在 importance ratio 上按 sequence 加权；二者能否组合，需要实验。
- SAPO 是否会成为 GSPO 的严格替代，还是只在 outlier token 多 / hard clipping 脆的阶段更好？论文承认所有方法最终仍可能 instability。
- ARPO 的 partial rollout 如果配 GSPO / SAPO，shared prefix 与 branch token 的 advantage attribution 应该用 sequence-level 还是 token-adaptive gate？
- GiGPO 的状态匹配在开放工具 / GUI / 仓库编辑里是否还能维持 ALFWorld 那种 >65% 重复率？与 ARPO 同一预算对照仍然缺失。
- HGPO 的 $k$-state history 能否代表真实 memory prompt？summary / retrieval memory 的层次相似度、deep-group covariance 与 uncertainty-weighted aggregation 仍无实验。
- SAO 的 frozen-attention critic 在 dense 模型上是否还成立？DIS 硬 mask 与 SAPO soft gate、CISPO detached clip 能否组合？
- MoE 的 routing volatility 是 GSPO/SAPO 的核心动机之一；dense 模型上 sequence-level 方法相对 DAPO recipe 的收益是否同样大？
- DeepSeekMath Figure 7 的「RL 抬 Maj@K、不抬 Pass@K」是否在更大模型或可验证环境 RLVR 上仍成立？后续报告几乎不复现这条曲线。
- MiniMax-M1 没写 $\varepsilon_{\mathrm{IS}}^{\mathrm{high}}$ 的具体值；Laguna $(1,4)$ 与原文「不下下界」是否同一配方，两边都没有对照表。
- 2026 的 agentic / RLVR 栈几乎不用 DPO：是静态偏好对覆盖不了可验证环境，还是 length bias 等后续问题已经把它挤出生产？[KTO](../sources/kto.md) 把监督改成二元，仍不回答生产弃用因果。本页没有一手来源回答。
- CTPO 的 prefix 连乘在 MoE expert routing volatility 下是更敏感还是更钝？GSPO 的稳定性论据与 CTPO 的无偏论据没有同一评测面的对照，dense-only 实验无法回答。
- CTPO 在 TIR 多轮轨迹里如何对待环境返回的非策略生成 token（是否计入 $\rho^{\mathrm{cum}}$ 连乘、clip 是否作用其上）？正文与附录未说明，需看其 VERL 实现确认。

## 相关页面

- 来源：[DeepSeekMath](../sources/deepseekmath.md)（GRPO 一手出处）、[VAPO](../sources/vapo.md)、[DAPO](../sources/dapo.md)、[DPO](../sources/dpo.md)（离线偏好闭式解，不在主表）、[KTO](../sources/kto.md)（二元 HALO，不需要 pair，不在主表）、[Iterative RPO](../sources/iterative-rpo.md)（DPO+NLL / TRL `rpo_alpha`）、[Group Sequence Policy Optimization](../sources/group-sequence-policy-optimization.md)、[Cumulative Token Policy Optimization](../sources/cumulative-token-policy-optimization.md)（prefix-cumulative ratio + 位置自适应 clip）、[Soft Adaptive Policy Optimization](../sources/soft-adaptive-policy-optimization.md)、[Agentic Reinforced Policy Optimization](../sources/agentic-reinforced-policy-optimization.md)、[GiGPO](../sources/gigpo.md)、[HGPO](../sources/hierarchy-of-groups-policy-optimization.md)、[Single-Rollout Asynchronous Optimization](../sources/single-rollout-asynchronous-optimization.md)、[VibeThinker-3B](../sources/vibethinker-3b.md)、[Ring-1T](../sources/ring-1t.md)（IcePop 一手出处）、[Ling-2.6 技术报告](../sources/ling-2.6.md)（KPop）、[Laguna 技术报告](../sources/laguna-m1-xs2.md)（CISPO 采用 + vs GRPO/GSPO 消融）、CISPO 源头：[MiniMax-M1](../sources/minimax-m1.md)（MiniMax-M2 系列不是源头）、[ECHO](../sources/echo.md)（环境观测辅助 CE，不是新 ratio）
- 概念：[Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)、[异步 Agent RL](../concepts/asynchronous-agent-rl.md)、[训练—rollout 一致性](../concepts/train-rollout-consistency.md)、[Group-in-Group Policy Optimization](../concepts/group-in-group-policy-optimization.md)、[Hierarchy-of-Groups Policy Optimization](../concepts/hierarchy-of-groups-policy-optimization.md)、[Single-Rollout Asynchronous Optimization](../concepts/single-rollout-asynchronous-optimization.md)
- 系统侧来源：[R3](../sources/r3.md)（MoE 路由重放，与本页 ratio 方法正交）、[Miles v0.1](../sources/miles-v0-1.md)：不提出新算法，但把五类 advantage estimator（GRPO / GSPO / REINFORCE++ / PPO）与 TIS / clip-or-pop 做成同一层可替换组件，并给出低精度服务下 train–inference 残差的可测量级。
- 模型：[DeepSeekMath](../models/deepseekmath.md)（GRPO 发布检查点）、[MiniMax-M1](../models/minimax-m1.md)（CISPO 发布检查点）、[Qwen3](../models/qwen3.md)、[Qwen3-VL](../models/qwen3-vl.md)、[VibeThinker-3B](../models/vibethinker-3b.md)
