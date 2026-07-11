---
type: Comparison
title: "LLM RL policy optimization 对比"
description: "DAPO / GSPO / SAPO / ARPO 四类 LLM RL policy optimization 方法的抽象层级对比：GRPO recipe、sequence-level ratio、soft trust region、agentic partial rollout。"
tags: ["comparison", "llm-rl-policy-optimization", "rl"]
timestamp: 2026-06-25
---

# LLM RL policy optimization 对比

## 为什么开这一页

DAPO、GSPO、SAPO、ARPO 都在名字上是 policy optimization，但它们解决的不是同一个问题：

- DAPO 补的是 **GRPO 在 long-CoT 数学 RL 大规模复现时的 recipe 缺口**；
- GSPO 改的是 **importance ratio / clipping 的优化单元**，从 token-level 改到 sequence-level；
- SAPO 改的是 **trust region 的形状**，从 hard clipping 改成 temperature-controlled soft gate；
- ARPO 改的是 **agentic rollout 的采样位置**，从完整轨迹采样改到工具反馈后的高熵 step 分叉；
- MGPO 改的是 **prompt-level 的梯度权重**，用最大熵权重降权全对/全错 prompt，聚焦能力边界 prompt。

如果把它们都简写成「比 GRPO 更好」，检索时会混掉层级。本页按抽象层级拆开。

## 一张表

| 方法 | 论文 / 团队 | 改动层级 | 主要机制 | 解决的痛点 | 主要实验对象 | 和 GRPO 的关系 |
| --- | --- | --- | --- | --- | --- | --- |
| [DAPO](../sources/dapo.md) | ByteDance Seed + 清华 AIR 等，2025-03 | **GRPO recipe / 系统工程** | Clip-Higher、Dynamic Sampling、token-level loss、Overlong Reward Shaping、DAPO-Math-17K | entropy collapse、全对/全错 prompt 零梯度、长 CoT token 权重稀释、截断 reward noise | Qwen2.5-32B，AIME 2024 | 保留 GRPO group-relative clipped objective，补齐 recipe |
| [GSPO](../sources/group-sequence-policy-optimization.md) | Qwen Team，2025-07 | **importance ratio / clipping 单元** | sequence likelihood ratio $s_i=(\pi_\theta(y_i)/\pi_{old}(y_i))^{1/|y_i|}$，response-level clipping | token-level ratio 与 sequence-level reward 不匹配；MoE expert routing 波动使 token ratio 失效 | Qwen3-30B-A3B，AIME / LiveCodeBench / CodeForces | 用 sequence-level ratio 替代 GRPO token ratio |
| [SAPO](../sources/soft-adaptive-policy-optimization.md) | Qwen Team，2025-12 | **soft trust region / gate 形状** | sigmoid soft gate + $sech^2$ gradient weight；$\tau_{neg}>\tau_{pos}$ | hard clipping 过脆；GSPO 整条 sequence 被裁、GRPO token 越界即零梯度 | Qwen3-30B-A3B、Qwen3-VL-30B-A3B | 保留 group-based RL，替换硬裁剪为软门控 |
| [ARPO](../sources/agentic-reinforced-policy-optimization.md) | 人大 + 快手，2025-07 | **agentic rollout 采样结构** | 工具反馈后监控 token entropy，在高熵 tool-call step 分叉 partial rollouts；advantage attribution | trajectory-level RL 忽略工具反馈后的 step-level 决策 | Qwen2.5 / Llama3.1 / Qwen3，math/QA/deep search | 把 DAPO/GRPO/REINFORCE++ 当 trajectory-level baseline |
| [MGPO](../sources/vibethinker-3b.md) | Sina Weibo，2025-11（VibeThinker-1.5B）-> 2026-06（3B） | **prompt-level 梯度权重** | 最大熵权重 $w(q)=\exp(-\gamma D_{ME}(p(q)\|0.5))$ 降权全对/全错 prompt；GRPO clipped objective + on-policy | 全对/全错 prompt 零梯度浪费；training-inference probability mismatch | Qwen2.5-Coder-3B（VibeThinker-3B），AIME/LiveCodeBench | 保留 GRPO group-relative clipped objective，加 prompt-level weight |
| KPop / IcePop | Inclusion AI（Ling Team），2025-2026 | **训练-推理 mismatch 的 mask 形状** | IcePop：uniform fixed-ratio $[\alpha,\beta]$ + double-sided masking；KPop：symmetric binary KL divergence $D_{KL}^B(\pi_{train}\|\pi_{infer})$，两方向都要求 $\leq\phi$，单超参控制 | MoE RL 中训练-推理精度不对齐导致 token ratio 噪声；固定比率过度 mask 低概率 token | Ring-2.6-1T（1T MoE），agentic coding RL | 与 GRPO 系并列的 MoE RL 稳定化层，不替代 ratio/loss 而是控制哪些 token 参与 |

## 关键分叉：token、sequence、step 三个「单位」

### DAPO：还是 token-level GRPO，但把 recipe 做对

DAPO 的 objective 仍然是 token-level ratio + clipped objective + group reward normalization。它的贡献是把大规模 long-CoT RL 里几个会让 GRPO 跑不起来的工程细节补上：

- Clip-Higher 放宽上界，防止低概率探索 token 被上界压死；
- Dynamic Sampling 保证 batch 中每个 prompt 既有对也有错，从而有非零 advantage；
- token-level loss 让长 CoT 的 token 不被 sample-level 平均稀释；
- overlong shaping 避免把截断样本粗暴当错，降低 reward noise。

所以 DAPO 是「可复现的 GRPO 系统 recipe」，而不是从理论上否定 token-level ratio。

### GSPO：否定 token-level ratio，转到 sequence-level

GSPO 的判断更根本：既然 reward 是整条 response 的 reward，那么 importance sampling / clipping 也应该按整条 response 来。它认为 GRPO 的 token-level ratio 在长序列里会累积高方差，尤其 MoE 中 expert routing 波动会让 token ratio 失真。

GSPO 的 sequence-level ratio 让同一 response 内所有 token 共享一个权重，训练目标和 sequence reward 对齐。代价是它牺牲了局部 token adaptivity：一条 sequence 里少数 token off-policy，hard clipping 可能让整条 sequence 的有效信号被压掉。

### SAPO：保留 sequence coherence，但恢复 token adaptivity

SAPO 继承 GSPO 对 sequence coherence 的追求，但不接受 GSPO 的 hard sequence clipping。它用 smooth gate 做 token-level attenuation：正常情况下平均 token gate 近似 sequence-level gate，满足 GSPO 的对齐诉求；异常情况下只下调 outlier token，不让 near-on-policy token 一起陪葬。

这就是 SAPO 的定位：**GSPO-like when ratios are coherent, token-adaptive when they are not**。

### ARPO：不争 ratio，而争 rollout 预算投到哪里

ARPO 和前三者不在同一轴上。它不主要讨论 token ratio 或 clipping 形式，而是问多轮 agent 轨迹里哪里值得采更多样本。它的回答是：工具返回后，模型前 10–50 个 token entropy 升高，说明这里是 step-level tool-use 行为的关键不确定点；因此从这里 branch partial rollouts 比单纯采完整 trajectory 更有效。

这意味着 ARPO 可以理论上叠加 DAPO / GSPO / SAPO 的 policy loss；它改的是 rollout collection / sampling structure。

### MGPO：不争 ratio 也不争采样位置，而争 prompt 的梯度贡献

MGPO 和前四者在又一条不同的轴上。它不改 token-level ratio（DAPO/GSPO/SAPO 的轴），也不改 rollout 采样结构（ARPO 的轴），而是改 **batch 内每个 prompt 对梯度的贡献权重**。

核心观察：GRPO 里一个 prompt 采 $G$ 条 response，如果全对（$p(q)\approx 1$）或全错（$p(q)\approx 0$），group-relative advantage $A_i$ 全为 0 或信号极弱，梯度贡献接近零——这些 prompt 在 batch 里浪费了采样预算。DAPO 的 Dynamic Sampling 用 hard filter 丢弃这些 prompt（只保留有对有错的），MGPO 则用连续的指数权重 $w(q)=\exp(-\gamma D_{ME}(p(q)\|0.5))$ 降权它们，保留在 batch 里但降低梯度影响。

两种策略的 trade-off：DAPO hard filter 更彻底（全对/全错直接丢弃，采样预算全部给有信号的 prompt），但需要一个额外的过滤+重采样 pass；MGPO soft weighting 实现更简单（权重直接乘进 loss），但全对/全错 prompt 仍占采样预算。两者动机完全一致——聚焦能力边界 prompt——区别在实现粒度。

MGPO 还有一个 DAPO 不涉及的维度：它显式处理 training-inference probability mismatch。VibeThinker-3B 发现 rollout engine 优化推理吞吐量后，training-inference 概率失配被放大，因此改为全 on-policy（参考 [14, 15] 的稳定化策略）。这是 GRPO 系实践中常见的工程问题，DAPO 论文未讨论。

### KPop / IcePop：不争 ratio 也不争采样，而争「哪些 token 参与」

KPop 和前述方法在又一条轴上。它不改 token-level ratio（DAPO/GSPO/SAPO 的轴），不改 rollout 采样结构（ARPO 的轴），也不改 prompt 权重（MGPO 的轴），而是控制**训练-推理 mismatch 下哪些 token 被允许参与梯度**。

前代 IcePop 用 uniform constant-ratio constraint（固定 $[\alpha,\beta]$ + double-sided masking），隐含假设所有 token 的 ratio noise 相同。但实际 ratio divergence 依赖 token probability——低概率 token 的 ratio noise 更大，固定比率会过度 mask 它们。KPop 用 symmetric binary KL divergence（把全词表看成「当前 token vs 其余」二事件划分）替代固定比率，两个方向都要求小于阈值 $\phi$，单超参控制。这与 GLM-5 的 double-sided importance sampling 是同一层问题的不同解法，与 GSPO/SAPO 正交可组合。

详见 [Ling-2.6 技术报告](../sources/ling-2.6.md) § 3.2.3。

## 与模型报告的关系

- [Qwen3 技术报告](../sources/qwen3.md)：官方 2025-05 报告的 reasoning RL 阶段写的是 GRPO；DAPO/GSPO/SAPO 都是后续或外部算法论文，不能回写成原报告事实。
- [Qwen3-VL 技术报告](../sources/qwen3-vl.md)：原报告有 own post-training pipeline；SAPO 论文提供后续/配套 Qwen3-VL RL 训练证据，说明 SAPO 用在 Qwen3-VL-30B-A3B preliminary cold-start 上，但不替换源报告 pipeline。
- [VibeThinker-3B](../sources/vibethinker-3b.md)：MGPO 是 VibeThinker 系列（1.5B → 3B）的自研 RL 算法，不是外部算法论文的复现。VibeThinker-3B 的 Long2Short RL（零和 length-aware reward shift）是在 MGPO 之上的 efficiency 优化，与 DAPO 的 overlong shaping 解决的是问题的两面——DAPO 处理截断样本的 reward noise，Long2Short 主动 reshape 正确轨迹的长度偏好。
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)：本页可作为该概念页里「RL policy optimization 子谱系」的展开。
- [MinerU2.5-Pro](../sources/mineru-2-5-pro.md)：Stage 3 是 GRPO + DAPO recipe 的**非 agentic 应用**——文档解析格式对齐。三个特点值得与 agentic RL 对照：(1) reward 直接用评测指标（edit distance / CDM / TEDS / IoU），无需单独 reward model，也无 LLM-as-judge，比 agentic RL 的 reward 设计更「机械」但更可信；(2) 训练数据按 reward 分布过滤（剔除过高/过低 reward，保留中段）以最大化有效 policy gradient 信号——与 [MGPO](../sources/vibethinker-3b.md) 的 prompt-level 加权、DAPO 的 dynamic sampling 同属「聚焦能力边界样本」家族，但用 hard filter 而非 soft weight；(3) 增益小（+0.45）但定向——主要提公式 CDM（+0.81），印证 GRPO 对「交叉熵等权 token 与 sequence-level 结构指标错位」的修补。这说明 GRPO+DAPO recipe 不限于 long-CoT reasoning / agentic，对任何「token-level loss 与任务级指标错位」的结构化输出任务都适用。

## 待追问

- DAPO 的 token-level loss 与 GSPO 的 sequence-level ratio 是否冲突？一个在 loss reduction 上按 token 平均，一个在 importance ratio 上按 sequence 加权；二者能否组合，需要实验。
- SAPO 是否会成为 GSPO 的严格替代，还是只在 outlier token 多 / hard clipping 脆的阶段更好？论文承认所有方法最终仍可能 instability。
- ARPO 的 partial rollout 如果配 GSPO / SAPO，shared prefix 与 branch token 的 advantage attribution 应该用 sequence-level 还是 token-adaptive gate？
- MoE 的 routing volatility 是 GSPO/SAPO 的核心动机之一；dense 模型上 sequence-level 方法相对 DAPO recipe 的收益是否同样大？

## 相关页面

- 来源：[DAPO](../sources/dapo.md)、[Group Sequence Policy Optimization](../sources/group-sequence-policy-optimization.md)、[Soft Adaptive Policy Optimization](../sources/soft-adaptive-policy-optimization.md)、[Agentic Reinforced Policy Optimization](../sources/agentic-reinforced-policy-optimization.md)、[VibeThinker-3B](../sources/vibethinker-3b.md)、[Ling-2.6 技术报告](../sources/ling-2.6.md)（KPop / IcePop）
- 概念：[Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)、[异步 Agent RL](../concepts/asynchronous-agent-rl.md)
- 模型：[Qwen3](../models/qwen3.md)、[Qwen3-VL](../models/qwen3-vl.md)、[VibeThinker-3B](../models/vibethinker-3b.md)
