---
type: Concept
title: "Multi-Teacher On-Policy Distillation"
description: "MiMo-V2-Flash 的 MOPD 范式及其与 DeepSeek-V4 OPD 的关系，并含跨家共用的 OPD 数学依据（reverse-KL mode-seeking+unhackable / on-policy 消除 exposure bias / teacher 固定的良定义优化 / O(1)-vs-O(N) bits/episode / RL 子网络脆弱性 / phase-alternating + 多 teacher 混采的边界）。"
tags: ["concept", "multi-teacher-on-policy-distillation"]
timestamp: 2026-06-06
---

# Multi-Teacher On-Policy Distillation

> 想看各家技术报告里 OPD 的目的 / KL 形式 / pipeline 位置怎么分叉，请直接看 [On-Policy Distillation 跨报告对比](../comparisons/on-policy-distillation.md)。本页以 MiMo MOPD 为骨架讲机制细节，但「数学依据」一节适用于所有走 on-policy distillation 路线的家族（MiMo / DeepSeek-V4 / Qwen3 / Qwen3-VL / GLM-5 / Nemotron 3 Ultra）。

## 一句话定义

Multi-Teacher On-Policy Distillation（MOPD）是 [MiMo-V2-Flash](../models/mimo-v2-flash.md) 的核心后训练范式：student 从自己的分布采样，再从多个 domain-specialized teacher 获得 token-level supervision，把多个专家能力合并进同一个模型。

## 为什么需要 MOPD

顺序后训练常见问题是 capability see-saw：数学 RL 提升后，写作或代码退化；代码 RL 提升后，安全或通用问答退化。简单混合数据或合并权重也不总能保留每个 teacher 的峰值能力。

MOPD 的目标是让 student 同时学习多个 teacher 的专门能力，并尽量避免单领域优化带来的互相覆盖。

## 三阶段流程

1. SFT：先建立基础 instruction-following 能力。
2. Domain-specialized training：分别训练专门 teacher，覆盖搜索、代码、工具使用、数学推理、通用推理和安全等任务。
3. MOPD：student on-policy 采样，并根据 prompt 所属领域，从对应 teacher 获得 token-level KL reward。

## 技术直觉

普通离线蒸馏让 student 学 teacher 生成的数据，但这些数据可能不在 student 当前分布上。MOPD 让 student 学自己真实会生成的轨迹，因此更接近部署时遇到的分布。Teacher 不直接给最终答案，而是通过 logits 或 token-level KL 提供细粒度训练信号。

报告中的 surrogate loss 可以理解为：如果某个 token 在 domain teacher 下概率更高，而在 student 下概率较低，这个 token 对 student 形成正向 advantage；反之则形成惩罚。MOPD 还可以与 ORM/GRPO 的 advantage 混合。

## 数学依据：OPD 为什么 work

> 这一节的论证对所有 on-policy distillation 都成立——MiMo MOPD、DeepSeek-V4 OPD、Qwen3 Strong-to-Weak、GLM-5 cross-stage 的 loss 形式都落在同一个数学框架下，只是 KL 估计方式和 teacher 来源不同。

### 第一层：OPD loss 在数学上等价于最小化 reverse-KL

MiMo 公式 7-8、GLM-5 公式 2、Qwen3 §4.5、V4 公式 29 表面上各不相同，但都可以归到同一个目标：

$$\min_\theta \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)} \; D_{KL}\!\left(\pi_\theta(\cdot|x, y_{<t}) \,\Big\|\, \pi_T(\cdot|x, y_{<t})\right)$$

推导用 likelihood-ratio gradient（log-derivative trick）。对 reverse-KL 求梯度时，由于采样分布 π_θ 本身含 θ，结果是：

$$\nabla_\theta D_{KL}(\pi_\theta \,\|\, \pi_T) \;=\; \mathbb{E}_{y \sim \pi_\theta}\!\left[\underbrace{\log\frac{\pi_T(y)}{\pi_\theta(y)}}_{\text{当 advantage}}\;\nabla_\theta \log \pi_\theta(y)\right] \;+\; \text{熵正则项}$$

主项的形式正好是 policy gradient `E[Â · ∇log π_θ]`，只要把 advantage 定义成

$$\hat{A} \;=\; \text{sg}\!\left[\log \frac{\pi_T(y\mid x, y_{<t})}{\pi_\theta(y\mid x, y_{<t})}\right]$$

塞回 GRPO/PPO 框架即可。这就是 **GLM-5 公式 (2) 和 MiMo 公式 (8) 的来源**——两家用一模一样的代换。所以 OPD 不是新算法，是「把 RL 的 advantage 换成 KL log-ratio」的 reparameterization，infra 完全复用 RL 训练栈。

V4 的 full-vocabulary KL 是同一目标的**精确版本**——它不简化成 token-level 标量估计、而是直接算整个词表上的 D_KL：

$$\mathcal{L}_{OPD}(\theta) \;=\; \sum_{i=1}^N w_i \cdot D_{KL}\!\big(\pi_\theta \,\|\, \pi_{E_i}\big) \quad \text{（V4 公式 29）}$$

代价是要在每个位置 materialize teacher 完整 logits，工程难度大；但**优化目标和 MiMo / GLM-5 严格一致**。

**但「reverse-KL + 纯 on-policy 数据」只是这一层收敛出的配方，不是它的全部维度。** 这类目标的原始形式来自 [GKD](../sources/generalized-knowledge-distillation.md) `§3.1`，它把目标拆成两个独立旋钮：**student 自生成数据的比例 $\lambda \in [0,1]$** 与**发散度 $D$**（含 forward / reverse KL 与在两者间插值的 JSD(β) 谱系），并有 $\mathcal{L}_{\text{GKD}} = (1-\lambda)\mathbb{E}_{(X,Y)}[D] + \lambda\mathbb{E}_{x\sim X, y\sim p_S}[D]$。supervised KD 是 λ=0、on-policy KD 是 λ=1，ImitKD 与 f-distill 也都是该框架的实例（详见来源页）。已收录的各家报告事实上都取 λ=1 + student-first reverse KL，但 GKD 原文的 on-policy 标准实例用的是 **teacher-first 方向**（论文称 forward KL），且论文实测的最优发散度是 **task-dependent**。λ 这一维在现有报告里没有任何对应消融。

### 第二层：为什么 reverse-KL，不是 forward-KL

OPD 全家都选 reverse-KL `D_{KL}(\pi_\theta \,\|\, \pi_T)`，不是 forward-KL `D_{KL}(\pi_T \,\|\, \pi_\theta)`。这两个方向**不对称**，且对蒸馏行为影响巨大（注意这是 2025–2026 各家报告的**经验选择**：原始框架 [GKD](../sources/generalized-knowledge-distillation.md) 的 on-policy 标准实例用的正是 teacher-first 方向，reverse KL 只是它对比过的选项之一）：

| | forward-KL `D(π_T ∥ π_θ)` | reverse-KL `D(π_θ ∥ π_T)` |
| --- | --- | --- |
| 惩罚什么 | π_θ 在 π_T 有质量的地方为 0 | π_θ 在 π_T 为 0 的地方有质量 |
| 行为 | **mass-covering / mode-covering**：student 试图覆盖 teacher 所有 mode，必要时把概率"摊薄"在 mode 之间 | **mode-seeking / zero-forcing**：student 在每个 mode 周围"做选择"，把质量集中到 teacher 认可的若干 mode 上 |
| 多 teacher 场景 | student 被迫做加权平均，容量不够时两头不讨好 | student 在 prompt 路由下贴向**对应**teacher 的 mode，不被迫加权 |

> Bishop *PRML* §10.1.2 用高斯混合等高线图给了直观证明；Minka 2005 *Divergence Measures and Message Passing* TR（MSR-TR-2005-173）把 forward/reverse KL 列在 α-divergence 谱系的两端。

**这张表的一手出处与本表的适用边界。** 在本 wiki 实际引用的文献里，上述行为对照的直接出处是 [GKD](../sources/generalized-knowledge-distillation.md) 的 `Figure A.16`（`§A.7`）：论文原话是「容量失配下用 $Q_\theta$ 近似 $P$，最小化 reverse 与 forward KL 分别导致 mean-seeking 与 mode-seeking」，作图设置为**连续、单峰高斯 Q 拟合双峰 P**。把它外推到词表级离散蒸馏是推论而非该图结论。[AKL](../sources/akl.md)（Wu et al.，COLING 2025）已收原文：逐步 softmax 上 FKL 与 RKL 的驻点都是 \(q=p\)（公式 5–8），有限 epoch 里 FKL 先拟合 head、RKL 先拟合 tail。因此本表应读作「容量失配 + 连续分布」下的直觉，以及各家**为何在工程上这么选**的动机，而不是「离散 LLM 蒸馏下 reverse KL 必然 mode-seeking」的定理。**不要**把 AKL 的同驻点论证直接套到 2026 生产 OPD 的 sampled-token reverse-KL advantage——那是另一条估计器，AKL 没做。GKD 自己实测的发散度排序是 task-dependent（XSum 温度采样下 reverse KL / JSD(0.9) 最好，GSM8K 上 forward KL 并不差，指令微调上 reverse KL 大幅领先）；按 AKL 应读成没训到收敛时的路径差。同期另一支源头 [MiniLLM](../sources/minillm.md) 用的也是同一个连续 toy 设置（单峰高斯拟合高斯混合，其 `Figure 2`）。

对 GLM-5 这种「召回早期能力」的用法特别关键：reverse-KL 的 mode-seeking 性质让 student 能**强力把分布拉回**到 Reasoning_RL teacher 的某个 mode，而不是被强迫"同时覆盖 SFT + Reasoning_RL + General_RL 所有 mode 的并集"——后者在 student 容量受限时不可行。

**额外性质：reverse-KL 是 "unhackable"。** [Thinking Machines Lab On-Policy Distillation 博客](../sources/thinking-machines-on-policy-distillation.md)（OPD 在 GLM-5 ref [28]、MiMo §4 的共同算法源头）指出，reverse-KL 比 RL 里常见的 reward model 更不易被 hack：「low KL always corresponds to a high probability of desirable behavior from the teacher model's point of view」。RL reward model 是个学出来的 scalar 估计，可能被 policy 找到漏洞（superficial pattern 拿高分），但 reverse-KL 直接以 teacher 输出分布为锚，没有学习出来的中介——低 KL 必然意味着 student 在该状态下选择 teacher 高概率的 token。这是 GLM-5 / MiMo 把 advantage 换成 KL log-ratio 而非保留 outcome reward 的一个工程动机。

### 第三层：on-policy 采样让梯度只作用在 student 实际访问的状态上

OPD 的 KL 期望是对 **student 自己**的分布取的：`E_{y ∼ π_θ}[...]`。这和离线蒸馏（off-policy distill，用 teacher 生成的数据训 student）有根本差异：

- **off-policy**：student 学「teacher 会怎么写」，但 teacher 写过的状态 student 自己可能根本走不到。训完仍有 exposure bias，部署时跑偏。
- **on-policy**：监督直接打在 student 部署时**真实会经过**的状态分布上，没有 distribution shift。

> 数学根据：Ross et al. 2011 *DAgger*（AISTATS）证明 imitation learning 里 on-policy 数据让 cumulative regret 是 O(T)（线性），off-policy 是 O(T²)（平方）。OPD 沿用同一个论证。这也是 Qwen3 Table 21 里 on-policy distill **pass@64 涨**（93.3 vs 起点 90.0）、而 RL pass@64 不动（90.0）的解释——on-policy distill 在 student 自己的轨迹空间上拓宽了概率质量分布；RL 只 sharpen 已有 mode。

把 DAgger 的论证搬到自回归语言模型上、并把蒸馏明确写成「带交互式 expert 的 imitation learning」的是 [GKD](../sources/generalized-knowledge-distillation.md) `§3.1`（论文同时给出该框架的前置条件：student 必须已经能生成质量尚可的序列，实验里一律从 SFT 后的 student 起步——这解释了为什么现在的 OPD 都接在 SFT / 分域 RL 之后，而不是从 base 直接起）。这一层的**量化证据**来自 [MiniLLM](../sources/minillm.md) `Figure 6`：它用 ExAccErr（Arora et al. 2022 口径，把累积 regret 拆成 oracle 上下文误差与自生成前缀误差）度量纯 exposure bias 部分，三条监督式基线随生成长度持续累积、MiniLLM 在 >150 token 后趋平。

### 第四层：teacher 固定 → 良定义的收敛目标

teacher 参数 freeze，π_T 是个不动的目标分布。reverse-KL 朝它收敛是一个**单调、有界、有明确最优**的优化问题：

$$D_{KL}(\pi_\theta \,\|\, \pi_T) \;\geq\; 0, \quad \text{下界 0 当且仅当 } \pi_\theta \equiv \pi_T$$

理论上 student 可以完全召回 teacher 的能力（容量足够、采样足够、step 足够）。这一点比 RL 干净得多——RL 的 reward landscape 可能有多个不一致的 mode、可能 reward hacking、没有干净的全局最优。所以 GLM-5 §3.5 才敢用 **"swiftly recover"** 这种自信措辞——OPD 的优化目标允许 swift，RL 的不允许。

### 第五层：信息论效率 —— `O(1)` vs `O(N)` bits per episode

[Thinking Machines Lab 博客](../sources/thinking-machines-on-policy-distillation.md)（§ Discussion · Dense supervision）给出了 OPD 比 RL 高效的信息论解释：

- **RL** 每个 episode 只传递 `O(1)` bits 信息——最终对/错一个 scalar reward，与 token 数无关。
- **Distillation** 每个 episode 传 `O(N)` bits，N 是 token 数——每个 token 都有 reverse-KL 信号，credit assignment 自然 dense。

实验验证：同一目标下 distillation 比 RL 快 **7-10× gradient steps**、综合 compute 效率 **50-100×**（博客 § Discussion 自蒸馏实验：用同一 base model 跑 RL 训出 teacher，再 distill 回 base model，distillation 用 10 步达到 RL 70 步才到的水平）。

这条信息论分析也解释了 [Qwen3 Table 21](../sources/qwen3.md) 的 1/10 GPU·h 数字：不是工程优化造成的差距，是**reward density 本身的数量级差**。要 RL 追上 distillation 的 compute 效率，理论上需要 process reward modeling（[Lightman et al. 2023](https://arxiv.org/abs/2305.20050)）把 reward 也做到 token 粒度——但博客指出"训 RL 用 process supervision 一般很难"，所以 OPD 是这一类的更可行 instantiation。

### 第六层：RL 训练只动小子网络 —— 为什么"召回"是必须的

这一层不是 OPD 本身的数学性质，但解释了**为什么 GLM-5 / Thinking Machines 把 OPD 用作"召回工具"是必要的，而不是可选的**。

[Mukherjee et al. 2025 *Reinforcement Learning Finetunes Small Subnetworks in Large Language Models*](https://arxiv.org/abs/2505.11711) 证明：RL post-training **只调整 base model 的一小块子网络**，大部分参数保持原状。这有两个直接后果：

1. RL 学到的能力**脆弱**——继续在大规模数据上 SFT/mid-training 时，这块子网络容易被冲掉。
2. Catastrophic forgetting 不是数据脏，是子网络被覆盖。

Thinking Machines 博客 § Personalization 给的实验直接证据：Qwen3-8B（已 RL）+ 内部公司文档 SFT → IF-eval **85% → 45%**（即使 SFT 数据混 30% chat 数据也救不回来）。然后用**原版 Qwen3-8B** 当 teacher 在 Tulu3 prompt 上做 on-policy distill → IF-eval **召回到 83%**，且新学的 internal QA 知识 36→41%（不退化）。

**对 GLM-5 cross-stage distillation 的直接含义**：sequential RL 流水线（SFT → Reasoning RL → Agentic RL → General RL）每一段都在改造一小块子网络；到 General RL 末端时，Reasoning RL 阶段那块子网络很可能已经被冲淡。cross-stage distill 把 Reasoning RL 阶段的 checkpoint 当 teacher，相当于在 student 部署分布上**把那块子网络锚回去**——博客原文称作 *"re-invoke" capabilities lost during fine-tuning*。

这一层和「第四层 teacher 固定 → 良定义目标」组合起来，给出 GLM-5 流水线设计的完整数学/系统解释：**RL 子网络脆弱（必须召回）+ teacher 固定（召回目标良定义）+ on-policy 采样（在 student 真实分布上召回）+ reverse-KL mode-seeking（强力拉回，不被迫加权所有阶段）= 流水线终点的 cross-stage distillation**。

### 第七层：phase-alternating 框架 —— 连续学习的可行 recipe

Thinking Machines 博客 § Personalization 末段：

> fine-tune 学新知识 → on-policy distill 召回行为 → fine-tune 学新知识 → distill ……

引 [Cobbe et al. 2020 *Phasic Policy Gradient*](https://arxiv.org/abs/2009.04416)。这是 continual learning 的可行 recipe——不需要把所有任务塞进同一次训练，而是**承认 SFT 会冲淡 post-training 行为、但 OPD 能在每次 SFT 后召回**，两者交替推进。

GLM-5 是这条 recipe 的**单次实例**（只跑了一次 RL → distill 召回）。MiMo MOPD 的 "teacher-student co-evolution 循环"（§4.1 提到 distilled student 可再进 specialist RL 生成更强 teacher）是同一框架的**多次迭代版**，但目前两家都只跑了一轮。phase-alternating 是把这个循环跑多次的理论保证。

### 把七层拼起来

OPD 能 work（特别是 GLM-5 那种「召回早期能力」用法）建立在七个数学/系统事实的合力上：

1. **loss = reverse-KL** —— likelihood-ratio gradient 给出 advantage = log(π_T/π_θ) 的干净代换，infra 与 RL 共用。
2. **reverse-KL 的 mode-seeking + unhackable 性质** —— student 在每个 mode 周围做选择不被迫加权平均，且 reward 锚定在 teacher 输出分布而非学出来的 scalar 估计，没有 hack 漏洞。
3. **on-policy 消除 exposure bias** —— 梯度只作用在 student 部署时会访问的状态分布上，无 distribution shift（Ross 2011 的 O(T) vs O(T²) regret）。
4. **teacher 固定 → 良定义优化** —— reverse-KL ≥ 0 且 0 当且仅当分布相等，单调收敛，没有 RL 的 reward landscape 问题。
5. **`O(1)` vs `O(N)` bits/episode** —— dense reward 的信息论根源是 OPD 比 RL 快 7-10× steps、compute 省 50-100× 的解释，不是工程优化的功劳。
6. **RL 子网络脆弱（Mukherjee 2025）** —— sequential RL 阶段间互相冲掉对方子网络是真实现象，cross-stage distill 召回的就是这些子网络；这是 GLM-5 cross-stage distillation 必要性的数学依据。
7. **phase-alternating 框架（Cobbe 2020）** —— fine-tune 与 distill 交替是 continual learning 的可行 recipe；GLM-5 单次实例 / MiMo co-evolution 多次迭代都落在这个框架下。

### 数学上没闭合的地方：多 teacher 混采

**这四条直接论证「OPD 能逼近一个固定 teacher」，但不直接论证「多 teacher 混采能同时召回多个能力而不互相覆盖」。**

多 teacher 实际上把目标退化成一个**条件混合分布**：

$$\pi^*(y\mid x) \;=\; \pi_{T_{k(x)}}(y\mid x), \quad \text{其中 } k(x) = \text{teacher 路由}(x)$$

如果不同 teacher 的"专长 prompt 集合"几乎不重叠（数学 prompt / 通用 prompt / 搜索 prompt 各走各的），student 在不相交的输入区域上分别拟合各 π_{T_k}，**互不干涉**，前述四层论证逐 teacher 成立。但如果 prompt 集**重叠**（比如某条 prompt 同时合理由 Reasoning_RL teacher 和 General_RL teacher 解），它只被分配给一个 teacher，另一个的视角就丢了——MiMo Table 7 里 BrowseComp（−6.8）、Arena-Hard Creative Writing（−3.9）落后 best teacher 的差距，很可能就是这种 routing 错位。

这是 OPD 在数学上**没**完全闭合的部分，也是为什么 GLM-5 §3.5（"mixed in appropriate proportions"）和 MiMo §4 都把 teacher 选择 / domain routing / 采样比例挂在嘴边——这一步外包给了数据 curation，不是公式能解决的。

## 分布视角：SFT / RL / OPD 的三轴对照

> 以下框架来自 [nrehiew 博客](../sources/nrehiew-sft-rl-opd.md)（2026），它把三种 post-training 方法放在同一个分布视角下：**target distribution 是什么 / 数据从哪来 / KL 方向**。这与上面的七层数学依据互补--七层讲「OPD 为什么 work」，分布视角讲「把 SFT 和 RL 也拉进来，差异在哪一层」。

| | SFT | RL | OPD |
| --- | --- | --- | --- |
| target distribution | 固定外部分布（数据集） | 无明确外部目标 | teacher 分布 |
| 数据来源 | 外部数据集（off-policy） | 当前策略采样（on-policy） | 当前策略采样（on-policy） |
| KL 方向 | forward KL（mode-covering） | ~reverse KL（mode-seeking） | reverse KL |
| 梯度压力范围 | 均匀施加在所有 demonstrated token | 只在当前策略采样的高概率区域 | 只在 student 采样区域，朝 teacher 拉 |
| 遗忘 | 严重 | 轻微 | 轻微 |

SFT 的 forward KL 等价于最小化 $D_{KL}(p_{\text{data}} \| q_\theta)$，其中 $p_{\text{data}}$ 是数据集定义的固定分布。由于 NLL 不考虑起始分布，模型没有内置理由偏好邻近解--target 可以任意远。SFT 对每个 demonstrated token 一视同仁推高概率，不区分 task-critical token 和 style token（[Diao et al.](https://arxiv.org/abs/2601.02151) 发现 SFT 中存在大量 low-probability low-entropy token，模型本很自信却被迫拟合分歧 label）。

RL 的 [Chen et al., 2025](https://arxiv.org/abs/2510.18874) 指出可看作 reverse KL 最小化。但作者认为 KL 方向解释**不完整**--它依赖显式 KL 正则化，而 RLVR 去掉 KL 惩罚后仍抗遗忘。[Shenfeld et al.](https://arxiv.org/abs/2509.04259) 给出更底层解释：用 REINFORCE + binary 0/1 reward 时，reward 充当 filter（reward=1 贡献正信号，0 不贡献），RL 的隐式 target 是「所有 optimal policy 中离当前策略最近的」--on-policy 数据在每个时间步把训练约束在低 KL 区域。

### on-policy 数据是承重墙：对照实验

nrehiew 在 Minimal Code Editing 任务上做了直接对照：先分别用 SFT 和 RL 训出两个 teacher（RL 泛化更好、不遗忘；SFT 遗忘明显），再分别做 OPD 蒸馏到 student。

| Model | Pass@1 ↑ | Norm. Levenshtein ↓ | Added CC ↓ | LiveCodeBench v6 ↑ |
| --- | --- | --- | --- | --- |
| SFT teacher | 0.775 | 0.450 | 0.450 | 0.286 |
| RL teacher | 0.792 | 0.063 | 0.206 | 0.320 |
| OPD ← SFT teacher | **0.800** | 0.059 | **0.206** | 0.297 |
| OPD ← RL teacher | 0.787 | **0.055** | 0.228 | **0.314** |

**反直觉**：两个 OPD student 几乎一样，都略超 RL teacher、远超 SFT teacher。即使 teacher 是退化的 SFT 模型，student 的遗忘也比 SFT teacher 本身轻。

**含义**：teacher 提供信号，但 on-policy 采样决定了几何形状。这与七层数学依据中的「第三层 on-policy 消除 exposure bias」和「第六层 RL 子网络脆弱」吻合--on-policy 数据在 student 真实分布上召回/重塑子网络，teacher 质量不是决定性的。这暗示可以「暴力 SFT 过训练 expert → OPD 蒸馏 → 保留原模型大部分能力」。

### OPSD：On-Policy Self Distillation

一手出处：[OPSD](../sources/opsd.md)（Zhao et al.，arXiv:2601.18734v3）。同一套 LLM 权重拆成两种条件分布：student 只看题目，teacher 看题目 + 参考解答 \(y^\star\)；student 采样自己的轨迹，teacher 在同一前缀上给 dense 监督，**不生成 token**。训练时 teacher **冻结为初始策略**（再叠加 LoRA），所以不是「当前自己蒸当前自己」。

需要按原文校准的口径：主实验是 **full-vocab、teacher-first forward KL**（GKD 的 on-policy 标准实例），不是 2026 生产 OPD 的 reverse KL。Table 3 上 reverse KL 在 AIME25 从 36.7 走到 35.0。style token（`wait` / `alright`）的位置级 KL 仍比 math token 高一个数量级（Table 5，1.7B 上 0.85 vs 0.14），对策是对词表项 f-divergence 做 \(\min(\ell_{n,v},\tau)\)；无 clipping 会在 100 step 内把 AIME24 拉崩（Figure 4）。

[nrehiew](../sources/nrehiew-sft-rl-opd.md) 把 OPSD 读成「更接近 RLHF 而非 RLVR」——这是博客评价，不是论文结论。token 级质量控制仍和 [KAT-Coder-V2.5](../sources/kat-coder-v2.5.md) drift-aware truncation、[Keye-VL-2.0](../sources/keye-vl-2.md) top-k overlap 同层，但剪的对象不同：OPSD 剪 full-vocab 里的高贡献 style 词，不是长轨迹 drift。

### Student 为什么能超越 Teacher

[GKD（Agarwal et al., ICLR 2024）](../sources/generalized-knowledge-distillation.md) 已在 GSM8K 上报告此现象（来源页 `§A.1` 的自蒸馏实验：FLAN T5-Large teacher 20.5%，自蒸馏后 student 反超 teacher）。nrehiew 给出两个假设：

1. **OPD 监督更精准**：teacher 在 student 自己的 prefix 上给建议，而非 teacher 生成的轨迹。student 的错误不一定是 teacher 的错误--off-policy 蒸馏可能在 student 很少访问的分布区域给监督。
2. **KL matching ≠ reward maximization**：teacher 分布含 style、不确定性、替代路径、推理结构等信息。匹配它能在不复制 teacher greedy 行为的前提下重塑 student 分布，改善采样行为。即使 teacher 的采样输出不更好，student 仍能进步。

熵行为差异：OPD 的 entropy collapse 比 RL 更剧烈（reverse KL mode-seeking 的预期行为），reward 上升更突然。这部分是推测性的——**原始归属需要降级**：这条此前挂在 [Gu et al., 2023](https://arxiv.org/abs/2306.08543) 名下，但 [MiniLLM](../sources/minillm.md) 原文并未做 OPD vs RL 的熵曲线对照，它做的是 mode-seeking 论证与多样性持平检验（`Table 3`），见该来源页待追问。

## 报告中的结果

MiMo-V2-Flash 报告的 Table 7（MOPD 前后 student vs. best teacher 对比）：

| Benchmark | Student Before MOPD | Best Teacher | Student After MOPD | Δ(Student−Teacher) |
| --- | --- | --- | --- | --- |
| AIME 2025 | 89.3 | 93.9 (RL) | 94.1 | +0.2 |
| HMMT Feb. 2025 | 76.9 | 82.6 (RL) | 84.4 | +1.8 |
| LiveCodeBench | 77.5 | 82.6 (RL) | 83.2 | +0.6 |
| MMLU-Pro | 84.7 | 84.7 (Self) | 84.9 | +0.2 |
| GPQA-Diamond | 84.9 | 84.9 (Self) | 83.7 | −1.2 |
| HLE (w/o Tool) | 21.2 | 21.2 (Self) | 22.8 | +1.6 |
| Arena-Hard (Hard Prompt) | 50.0 | 50.0 (Self) | 54.1 | +4.1 |
| Arena-Hard (Creative Writing) | 90.1 | 90.1 (Self) | 86.2 | −3.9 |
| SWE-Bench Verified | 67.8 | 74.2 (RL) | 73.4 | −0.8 |
| Tau2-Bench | 75.9 | 79.6 (RL) | 80.3 | +0.7 |
| Tau2-Bench (Telecom) | 92.7 | 95.0 (RL) | 95.3 | +0.3 |
| BrowseComp | 42.5 | 51.7 (SFT) | 44.9 | −6.8 |

> Table 7（原文 §4.1）：Best Teacher 列括号标注其类型——RL（领域专项 RL teacher）、SFT（SFT teacher）、Self（student 自身即最强）。MOPD 后 student 在多数项上接近或超过 best teacher，但在 BrowseComp（−6.8）、Arena-Hard Creative Writing（−3.9）、GPQA-Diamond（−1.2）、SWE-Bench Verified（−0.8）上落后。

这说明 MOPD 不是自动合并所有峰值能力，teacher 选择、领域路由和 reward 设计仍然关键。

## 与 DeepSeek-V4 OPD 的关系

[DeepSeek-V4](../models/deepseek-v4.md) 也使用多 teacher OPD，但强调 full-vocabulary logit distillation。它认为 token-level KL 估计虽然省资源，但方差高、训练不稳定，因此通过 teacher hidden-state caching、按 teacher 排序调度和 TileLang kernel 来支持完整 logits 的 KL 计算。

## Keye-VL-2.0 的 Cross-Modal MOPD

[Keye-VL-2.0](../sources/keye-vl-2.md) 是 MOPD 在多模态场景的首次大规模应用。它与 MiMo MOPD 的核心思路一致（多 domain teacher + on-policy student rollout + token-level KL），但增加了四项工程增强：

1. **Top-k overlap estimator**：只在 teacher 和 student 都认为合理（TopK 交集）的 token 上计算 advantage，避免在极低概率 token 上的不稳定比较。$\Omega_{i,t} = \text{TopK}(\pi_T) \cap \text{TopK}(\pi_\theta)$，空集时 advantage 归零。
2. **SPRR（Segmented Prompt-Response Re-tokenization）**：分别处理 prompt 和 response，确保 teacher log-prob 与 student response token 严格对齐。
3. **Token-category-aware advantage scaling**：formatting token 降权、perception/reasoning token 升权。
4. **Localized repetition penalty**：在重复坍缩位置 $\tau_i$ 之后才施加惩罚，不影响正常生成长度。

13 个 domain teacher 覆盖 safety / 纯文本数学 / 指令跟随 / code / 视觉 STEM / OCR / grounding / counting / video / tool use 等，是已收录报告中 teacher 数最多的（MiMo 未明确数量，V4 ">10"，KAT 5 个）。详见 [OPD 跨报告对比](../comparisons/on-policy-distillation.md)。

## Mach-Mind-4-Flash 的 MOPD

[Mach-Mind-4-Flash](../sources/mach-mind-4-flash.md)（理想汽车，arXiv:2607.09375）是 MOPD 融合派的又一个大规模实例。与 MiMo MOPD 算法形式一致（token-level reverse-KL k1 estimator + PPO clipped surrogate），但工程实现有三处独有设计：

1. **统一 RL/OPD 训练框架**：把 RL 和 OPD 深度集成到同一框架（加权 loss `L = α·L_OPD + β·L_RL`，公式 1），支持纯 RL / 纯 OPD / 联合三模式切换。OPD 阶段直接复用 RL 框架的分布式调度、异步 reward routing（20+ task 并行）和 online sampling 闭环。这是已收录报告中唯一把 RL 和 OPD 统一到单一 loss 公式的实现--MiMo/V4/GLM-5 的 OPD 和 RL 在 pipeline 上是先后阶段，不混在一个 loss 里。

2. **Early Stopping Rollout**：max_response_length 截到 8K token（即使长 math/code/search），缩短每步 rollout、降 vLLM KV-cache 压力。引 Ziheng et al. [60]（Less is More: Early Stopping Rollout for OPD）。

3. **Teacher-student 参数量匹配**：匹配 teacher 与 student 参数量比用大 teacher 有更高 top-K overlap rate，引 Li et al. [61]（Rethinking OPD: Phenomenology, Mechanism, and Recipe）。这与"大 teacher 蒸小 student"的直觉相反。

MOPD 融合效果（Table 3）展示三种模式：(1) Reasoning 的 **capability anchoring**（专家防止融合退化，移除 Reasoning expert 导致 −2–4%）；(2) General 的 **full retention**（融合后微超专家）；(3) Agent 的 **mixed**（SWE 部分保留 −2.7pp，ClawBench/ClawEval 超越专家）。Appendix C 消融发现 code teacher 的跨域迁移效应：加 code teacher 后 AIME'25 +2.30 / AIME'26 +0.83（未加数学 teacher），但报告明确建议每个想保留的能力都应有 teacher 代表，不依赖跨域迁移。

## Kimi K3 的 MOPD

[Kimi K3](../sources/kimi-k3.md)（Moonshot AI，arXiv:2607.24653，首个开源 3T 级）的 MOPD 把 teacher 数推到 **9 个**（3 域 × 3 reasoning effort），是已收录报告中 teacher 数第二多（仅次于 Keye-VL-2.0 的 13 个），且首次把 **reasoning effort 作为正交 teacher 维度**：

- **9 teacher 矩阵**：3 域（general tasks / general agents / coding agents）× 3 reasoning effort（`{low, high, max}`）。给定 domain `d` 和采样 effort `e`，由对应 teacher `π_teacher^(d,e)` 监督。这是对 Mach-Mind「10+ 专家按域分」和 Keye-VL「13 teacher 按模态/任务分」的另一种切法——**同一域的不同 effort 是不同 teacher**，让 student 能在统一模型里按需切换 effort。
- **per-token OPD reward with R_max clip**：`r_opd = clip(sg(log π_teacher / π_θ), -R_max, R_max)`（`sg` stop-gradient，`R_max` clip 极端 advantage 稳定 RL 训练）。clip 形式与 MiMo/Mach-Mind 的 PPO clipped surrogate 同族，但 K3 明确把 clip 上界叫 `R_max` 并强调"constrain extreme advantage signals"——是对 OPD reward 尺度失控的直接防护。
- **dense reward 无缝接入 RL 框架**：OPD reward 是 per-token dense 信号，天然支持 partial rollout training（长 horizon 任务未完成轨迹跨 iteration 续跑）。试过更细的 **top-k distillation 目标**，但在收敛速度和最终性能上都没明显优势——这与 Mach-Mind 的观察一致（full-vocabulary OPD 够用，不需 top-k 精简）。
- **与 RL 阶段的衔接**：9 个 teacher 是 RL 阶段的产物（3 域 × 3 effort 的 9 个 RL 专家），MOPD 把它们融合成单一 student。这与 MiMo「先训多 teacher 再 MOPD」同构，但 K3 的 teacher 矩阵结构（域 × effort）让融合后的 student 能按 prompt 激活不同 effort，而非固定一个 effort。

详见 [OPD 跨报告对比](../comparisons/on-policy-distillation.md)。

## Miles 的框架侧 OPD：把信号折进 advantage

[Miles v0.1](../sources/miles-v0-1.md) §5.2 的 OPD 与上面各家不同——它的差异不在算法而在**框架接口**，因此也回答了本页「待追问」里几个训练系统层面的问题：

- **信号进 advantage，不进 loss**。算法形式仍是一样本 reverse KL（本页「第一层」的同一代换），但 Miles 在 advantage estimator 算完之后，从每个 token 的 advantage 里减去按系数缩放的 divergence 估计，然后 policy-gradient 更新照旧。两者都是固定输入（rollout 时记录，或由单独的 teacher pass 产生），所以惩罚表现为 dense per-token reward 而不是额外 loss 项。后果是它与 GRPO / GSPO / PPO / REINFORCE++ **可任意组合**，而不是替换其中某一个；任务 reward 可保留也可设 0 只做蒸馏。GLM-5 的「advantage = teacher gap」是同一思路，Miles 把它做成不绑定具体 estimator 的接口。
- **teacher 部署方式被显式拆开**。**served teacher**：外部 SGLang 服务器在 rollout 期间给每条完成的轨迹打分，log-prob 随轨迹进 trainer；teacher 可以与 student 架构不同、也可以大到装不下，但**必须共享 student 的 tokenizer**（打分在 student 的 token IDs 上做）。**in-process teacher**：Megatron 在 student 旁边加载第二个同架构模型，训练 step 里做一次专门前向。可以注册多个 served teacher 按 prompt metadata 的 tag 路由。top-K 变体（student 的 top-K 或双方 top-K 交集）**只在 served teacher 下可用**——这与本页「数学上没闭合的地方」把 teacher 路由和采样比例外包给数据 curation 的判断一致。
- **它给了本页唯一一条「蒸馏只缩长度、不涨分」的公开结论**。文档中的 Qwen3.5-35B-A3B 运行：teacher 是同一模型加五步可验证 reward 的 RL，student 从 base 起，任务 reward 设为 0，reverse-KL 惩罚提供全部训练信号。五步内 held-out DAPO prompt 上的 response 长度从 14,070 降到 6,132 token（−56%），accuracy 从 84.0% 到 85.2%——作者自己指出 1.2 个点落在该评测约 1.6 点的标准误内，因此支持得住的结论是「长度大降、精度无可信变化」，而不是 benchmark 提升。这是已收录报告里少见的反例：OPD 不必然换来能力增益，缩长度本身可以是全部收益。
- **不建模型实体、不给多 teacher 数量**：Miles 是训练系统，它的 OPD 覆盖的是「怎么把任意 teacher 接进任意 RL 循环」，teacher 数量与领域划分仍属于各家模型报告（MiMo / V4 / GLM-5 / KAT / Keye-VL / Mach-Mind / K3）。它与 [Mach-Mind-4-Flash](../sources/mach-mind-4-flash.md) 的统一 RL/OPD loss 是最近的邻居——两者都把 OPD 塞进 RL 框架而非当成后续阶段，但 Mach-Mind 是**加权 loss**（`L = α·L_OPD + β·L_RL`），Miles 是**advantage 修正**，后者不需要改 loss 形式因而能与更多 estimator 组合。

详见 [OPD 跨报告对比](../comparisons/on-policy-distillation.md) 与 [Miles v0.1](../sources/miles-v0-1.md)。

## Nemotron 3 Ultra：两轮 co-evolution 与按域恢复率

[Nemotron 3 Ultra](../sources/nemotron-3-ultra.md)（NVIDIA，arXiv:2606.15007）是目前唯一把 MiMo 口头提过的 **teacher–student 多轮循环**真正跑完两轮、并给出按域恢复率的生产报告。算法仍是 sampled-token reverse KL 当 advantage（公式 1–2），异步把 behavior policy 与 proximal policy 拆开，token mask 用 IcePop。

增量不在公式，在三件实证：

1. **两轮 co-evolution**（Figure 10）。SFT → 统一 RLVR → warmup 轻 SFT → MOPD1；从 MOPD1 新训一批 teacher、复用第一轮一部分，再 MOPD2。RLVR student 兼 self-teacher。**RL 保留，MOPD 做融合**——对照 [DeepSeek-V4](../models/deepseek-v4.md) 用 OPD 替换 mixed RL。
2. **按域恢复率**（Table 5，\((\mathrm{MOPD2}-\mathrm{RLVR})/(\mathrm{Teacher}-\mathrm{RLVR})\)）。Terminal Bench 2.0 **172.7%**（student 超过 teacher）、SWE-Bench Verified 88.1%、BrowseComp 67.0%；HLE no tools **16.9%**、LiveCodeBench v6 32.0%。作者原文把鸿沟写成 on-policy 的适用边界：STEM teacher 的优势来自额外大规模 SFT+RL（DeepSeek-V4-Pro 生成的推理混合），student 没见过这些路径，rollout 对 teacher 是 OOD，token-level 监督变差。Agentic 优势能写成 student **已经能采样到的**轨迹上的 token 偏好时，恢复才高。
3. **Warmup 几乎是 agentic 的前置条件**（Table 4）。teacher 与 student 若走不同 SFT，student 轨迹对 teacher 不可靠。轻 SFT 对齐后 GDPVal 28.9→46.7（无 warmup 只有 35.3），HLE 几乎不动。这与 [KAT-Coder-V2.5](../sources/kat-coder-v2.5.md) 的 off-policy cold start 同属「先把 student 拉进 teacher 支撑集」，但 Ultra 用短 SFT 而不是 truncation。

作者还报告：full-vocab / top-k logit matching 在 Terminal Bench 上**不如 sampled-token**（`§3.3.5`）。这与 V4「full-vocab 更稳」直接对照，两边都没有交叉复现。多数 agentic 任务实际用 PivotRL 式单轮 rollout，不是端到端多轮。

## 待追问

- MOPD 的 domain routing 如何定义？粗粒度领域错误是否会导致负迁移？
- Teacher 数量增加时，student 容量是否足够保留所有能力？
- **MOPD 与异步 agent RL 的循环**：[Nemotron 3 Ultra](../sources/nemotron-3-ultra.md) 已跑两轮。还没回答的是哪些域需要第二轮（GDPVal 在 MOPD2 持平）、以及统一 SFT 能否救回 HLE 那类「teacher 靠 off-policy 新数据」的缺口。
- KAT-Coder-V2.5 的 drift-aware dynamic truncation 中，top-k overlap 阈值 $\rho_t$ 和连续低兼容性 token 数 $m$ 如何调参？截断比例过高是否会导致长轨迹训练信号不足？cold start 阶段的步数选择依据是什么？
- **on-policy 数据 > teacher 的结论是否只在 niche task 上成立**？nrehiew 的实验用 minimal editing（适合测遗忘/泛化），在更 broad 的能力域上 teacher 质量是否会重新主导？
- **OPSD 的 pointwise clipping 与 KAT-V2.5 drift-aware truncation / Keye-VL-2.0 top-k overlap 是否在解同一个问题**？OPSD 原文剪的是 full-vocab 里高贡献的 style 词表项，KAT 剪长轨迹 drift，Keye 过滤双方低概率 token。统一框架仍没有。
- **OPD 比 RL 更剧烈的 entropy collapse 是否意味着多样性损失更严重**？这与 Qwen3 Table 21 里 OPD pass@64 也涨是否矛盾？

